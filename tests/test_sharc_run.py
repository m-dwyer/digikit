"""Synthetic tests for tools/sharc_run.py, the concrete single-path runner.

Follows tests/test_sharc_trace.py's convention: build tiny synthetic loader
streams and Instruction records by hand rather than reading real firmware
bytes (sections/, out/ and anything derived from them are Elektron's
copyright and must never end up here).
"""

import os
import struct
import sys
import unittest
from importlib import import_module
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools"))
sr = import_module("sharc_run")
T = import_module("sharc_trace")
Instruction = import_module("sharc_disasm").Instruction
L = import_module("sharcldr")


def loader_block(code, address, count, arg=0, payload=b""):
    """Build a checksum-valid synthetic loader block (test_sharc_trace.py's
    helper, duplicated rather than imported so this file stands alone)."""
    header = bytearray(struct.pack("<IIII", code | 0xAD000000, address, count, arg))
    header[2] = 0
    checksum = 0
    for byte in header:
        checksum ^= byte
    header[2] = checksum
    return bytes(header) + payload


def loader_memory(*blocks):
    return L.LoadedMemory.from_stream(b"".join(blocks))


def insn(name, fields, length=4, kind="confident"):
    return Instruction(0, length, name, fields, kind=kind)


class ResetUregsTest(unittest.TestCase):
    def test_every_ureg_is_seeded(self):
        uregs = sr.reset_uregs()
        self.assertEqual(len(uregs), len(T.UREG_NAMES))
        for code in range(len(T.UREG_NAMES)):
            self.assertIsInstance(uregs[code], T.Const)

    def test_core_reset_values_are_used_where_documented(self):
        uregs = sr.reset_uregs()
        self.assertEqual(uregs[T.UREG_CODES["MODE1"]], T.Const(0))
        self.assertEqual(uregs[T.UREG_CODES["ASTATX"]], T.Const(0))

    def test_undocumented_uregs_default_to_zero(self):
        uregs = sr.reset_uregs()
        self.assertEqual(uregs[T.UREG_CODES["R0"]], T.Const(0))
        self.assertEqual(uregs[T.UREG_CODES["L7"]], T.Const(0))

    def test_overrides_win_over_reset_values(self):
        uregs = sr.reset_uregs({"R4": 0x1234, "MODE1": 5})
        self.assertEqual(uregs[T.UREG_CODES["R4"]], T.Const(0x1234))
        self.assertEqual(uregs[T.UREG_CODES["MODE1"]], T.Const(5))

    def test_negative_override_matches_seed_value_semantics(self):
        uregs = sr.reset_uregs({"M7": -1})
        self.assertEqual(uregs[T.UREG_CODES["M7"]], T.Const(0xFFFFFFFF))


class MakeStateTest(unittest.TestCase):
    def test_requires_loaded_memory(self):
        with self.assertRaises(ValueError):
            sr.make_state(b"\x00\x00", 0x10)

    def test_default_dag_modify_regs_are_seeded(self):
        state = sr.make_state(loader_memory(), 0x10)
        self.assertEqual(state.uregs[T.UREG_CODES["M6"]], T.Const(1))
        self.assertEqual(state.uregs[T.UREG_CODES["M7"]], T.Const(0xFFFFFFFF))

    def test_record_events_defaults_off(self):
        state = sr.make_state(loader_memory(), 0x10)
        self.assertFalse(state.record_events)

    def test_poke_writes_through_dm_write(self):
        state = sr.make_state(loader_memory(), 0x10, pokes={0x31400: 0xCAFEBABE})
        self.assertEqual(T._dm_read(state, T.Const(0x31400), 4), T.Const(0xCAFEBABE))

    def test_bad_poke_raises(self):
        # Not an MMR and not in the 32-bit-normal-word external range, and
        # nothing loaded to alias against: _dm_write() cannot place it.
        with self.assertRaises(ValueError):
            sr.make_state(loader_memory(), 0x10, pokes={0x1000000: 1}, assume_nw32=False)


class RunnerStepTest(unittest.TestCase):
    def test_advances_pc_and_counts_one_instruction(self):
        runner = sr.Runner(loader_memory(), 0x10)
        runner._cache[0x10] = insn("21a", {}, length=6)
        runner.step()
        self.assertEqual(runner.state.pc_sw, 0x13)
        self.assertEqual(runner.instructions, 1)
        self.assertEqual(runner.form_counts["21a"], 1)

    def test_record_events_off_leaves_trace_empty_or_minimal(self):
        runner = sr.Runner(loader_memory(), 0x10)
        runner._cache[0x10] = insn("21a", {}, length=6)
        runner.step()
        # 21a is a pure NOP-advance with no _event() calls at all either way.
        self.assertEqual(runner.state.trace, [])

    def test_decode_cache_avoids_redecoding(self):
        runner = sr.Runner(loader_memory(), 0x10)
        runner._cache[0x10] = insn("21c", {}, length=2)
        with patch.object(sr.st, "decode_at") as mock_decode:
            first = runner._decode(0x10)
            second = runner._decode(0x10)
        mock_decode.assert_not_called()
        self.assertIs(first, second)

    def test_decode_cache_calls_through_on_miss(self):
        runner = sr.Runner(loader_memory(), 0x10)
        sentinel = insn("21c", {}, length=2)
        with patch.object(sr.st, "decode_at", return_value=sentinel) as mock_decode:
            got = runner._decode(0x10)
            got_again = runner._decode(0x10)
        mock_decode.assert_called_once()
        self.assertIs(got, sentinel)
        self.assertIs(got_again, sentinel)

    def test_invalidate_forces_redecode(self):
        runner = sr.Runner(loader_memory(), 0x10)
        first_insn = insn("21c", {}, length=2)
        second_insn = insn("21a", {}, length=6)
        with patch.object(sr.st, "decode_at", side_effect=[first_insn, second_insn]) as mock_decode:
            got = runner._decode(0x10)
            runner.invalidate(0x10)
            got_after = runner._decode(0x10)
        self.assertEqual(mock_decode.call_count, 2)
        self.assertIs(got, first_insn)
        self.assertIs(got_after, second_insn)

    def test_breakpoint_halts_before_executing(self):
        runner = sr.Runner(loader_memory(), 0x10, breakpoints=[0x10])
        runner._cache[0x10] = insn("21a", {}, length=6)
        with self.assertRaises(sr.Halt) as ctx:
            runner.step()
        self.assertEqual(ctx.exception.reason, "breakpoint")
        self.assertEqual(ctx.exception.pc_sw, 0x10)
        self.assertEqual(runner.instructions, 0)

    def _type9a_abs_fields(self, **changes):
        fields = {
            "b": 0,
            "a": 0,
            "cond[4:0]": 31,
            "pmi[2:2]": 1,
            "pmi[1:0]": 0,
            "pmm[2:0]": 5,
            "j": 0,
            "e": 0,
            "ci": 0,
            "compute[22:16]": 0,
            "compute[15:0]": 0,
        }
        fields.update(changes)
        return fields

    def test_depth0_return_is_a_clean_halt_not_an_exception_from_advance(self):
        # PGR's delayed-return idiom (pmm==6, j=1); tools/test_sharc_trace.py's
        # test_type9a_abs_i4_m6_delayed_is_return_with_compute confirms
        # sharc_trace itself reports this as state.stopped ==
        # "return without followed call" for an empty call_stack -- i.e. the
        # call-depth-0 RTS this runner's entry routines are expected to hit.
        runner = sr.Runner(loader_memory(), 0x10)
        runner._cache[0x10] = insn(
            "9a_abs", self._type9a_abs_fields(**{"pmm[2:0]": 6, "j": 1}), length=6
        )
        with self.assertRaises(sr.Halt) as ctx:
            runner.step()
        self.assertEqual(ctx.exception.reason, "return without followed call")
        self.assertEqual(ctx.exception.form, "9a_abs")

    def test_unresolved_predicate_is_reported_as_a_fork_halt(self):
        # An 8a_rel conditional branch whose predicate bit (SV, cond 0x07)
        # is not known forks sharc_trace into taken/not_taken -- exactly the
        # case a fully concrete run should never hit, and Runner.step() must
        # turn into a Halt instead of silently taking one branch.
        runner = sr.Runner(
            loader_memory(), 0x10, regs={"ASTATX": T.Unknown("forced for test")}
        )
        fields = {
            "b": 0,
            "a": 0,
            "cond[4:0]": 0x07,
            "j": 1,
            "ci": 0,
            "reladdr[23:16]": 0,
            "reladdr[15:0]": 16,
        }
        runner._cache[0x10] = insn("8a_rel", fields, length=4)
        with self.assertRaises(sr.Halt) as ctx:
            runner.step()
        self.assertTrue(ctx.exception.reason.startswith("fork"))
        self.assertEqual(ctx.exception.pc_sw, 0x10)
        self.assertEqual(ctx.exception.form, "8a_rel")
        # No half-executed state was kept as "the" state after a fork.
        self.assertEqual(runner.instructions, 0)


class RunnerRunTest(unittest.TestCase):
    def test_stops_at_max_steps(self):
        runner = sr.Runner(loader_memory(), 0x10)
        runner._cache[0x10] = insn("21c", {}, length=2)
        # 21c always advances by one short word, so pc_sw cycles through a
        # small range of addresses this test never has to pre-populate the
        # cache for beyond the first -- decode_at() on an empty LoadedMemory
        # returns kind="unknown", so cap steps at 1 and assert on that.
        result = runner.run(max_steps=1)
        self.assertEqual(result.halt.reason, "max-steps")
        self.assertEqual(result.instructions, 1)
        self.assertEqual(result.form_counts["21c"], 1)
        self.assertGreaterEqual(result.instructions_per_second, 0)

    def test_run_reports_halt_from_step(self):
        runner = sr.Runner(loader_memory(), 0x10, breakpoints=[0x10])
        runner._cache[0x10] = insn("21a", {}, length=6)
        result = runner.run(max_steps=100)
        self.assertEqual(result.halt.reason, "breakpoint")
        self.assertEqual(result.instructions, 0)
        self.assertEqual(result.final_pc_sw, 0x10)

    def test_to_json_shapes(self):
        runner = sr.Runner(loader_memory(), 0x10, breakpoints=[0x10])
        runner._cache[0x10] = insn("21a", {}, length=6)
        result = runner.run(max_steps=100)
        payload = result.to_json()
        self.assertEqual(payload["halt"]["reason"], "breakpoint")
        self.assertIn("instructions_per_second", payload)
        self.assertIn("form_counts", payload)


class ParseKvTest(unittest.TestCase):
    def test_parses_name_value_pairs(self):
        self.assertEqual(sr._parse_kv(["R4=0x10", "I0=5"], "--reg"), {"R4": "0x10", "I0": "5"})

    def test_rejects_missing_equals(self):
        with self.assertRaises(SystemExit):
            sr._parse_kv(["R4"], "--reg")


if __name__ == "__main__":
    unittest.main()
