"""Tests for tools/sharc_survey.py.

Pure-Python pieces (load_patch_table, apply_patches, category
classification, the ASTAT-group-from-unknowns mapping) are tested against
synthetic loader streams, following tests/test_sharc_run.py's convention.
The end-to-end SurveyStopTest class needs the real dt2-1.16 program
database and firmware blob (out/sharcdb, out/sections -- see the worktree's
symlinks); it is skipped, not marked slow, when they are not available,
matching tests/test_sharc_graph.py.
"""

import contextlib
import io
import json
import os
import pathlib
import sys
import tempfile
import unittest
from importlib import import_module

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools"))
sv = import_module("sharc_survey")
sr = import_module("sharc_run")
T = import_module("sharc_trace")
L = import_module("sharcldr")

DT2_116_DB = pathlib.Path("out/sharcdb/dt2-1.16.sqlite")
DT2_116_BLOB = pathlib.Path("out/sections/dt2-1.16/section_7_BLOB.bin")


def loader_memory():
    return L.LoadedMemory.from_stream(b"")


class LoadPatchTableTest(unittest.TestCase):
    def test_no_path_is_an_empty_table(self):
        self.assertEqual(sv.load_patch_table(None), {})

    def test_json_hex_string_keys_and_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "patch.json")
            with open(path, "w") as fh:
                json.dump({"0x10": [["reg", "R0", "0x40000000"]]}, fh)
            table = sv.load_patch_table(path)
        self.assertEqual(table, {0x10: [("reg", "R0", 0x40000000)]})

    def test_json_decimal_keys_and_int_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "patch.json")
            with open(path, "w") as fh:
                json.dump({"16": [["mem", 32, 255]]}, fh)
            table = sv.load_patch_table(path)
        self.assertEqual(table, {16: [("mem", 32, 255)]})

    def test_py_module_reads_patch_table_dict(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "patch.py")
            with open(path, "w") as fh:
                fh.write("PATCH_TABLE = {0x10: [('reg', 'R0', 5)]}\n")
            table = sv.load_patch_table(path)
        self.assertEqual(table, {0x10: [("reg", "R0", 5)]})

    def test_py_module_without_patch_table_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "patch.py")
            with open(path, "w") as fh:
                fh.write("X = 1\n")
            with self.assertRaises(SystemExit):
                sv.load_patch_table(path)


class ApplyPatchesTest(unittest.TestCase):
    def _runner(self):
        return sr.Runner(loader_memory(), 0x10, regs={"R0": 1})

    def test_reg_patch_by_name(self):
        runner = self._runner()
        applied = sv.apply_patches(runner, {0x10: [("reg", "R0", 0x40000000)]})
        self.assertEqual(runner.state.uregs[T.UREG_CODES["R0"]], T.Const(0x40000000))
        self.assertEqual(applied, ["reg R0 = 0x40000000"])

    def test_reg_patch_by_code(self):
        runner = self._runner()
        code = T.UREG_CODES["R1"]
        sv.apply_patches(runner, {0x10: [("reg", code, 7)]})
        self.assertEqual(runner.state.uregs[code], T.Const(7))

    def test_mem_patch_writes_through_dm_write(self):
        runner = sr.Runner(loader_memory(), 0x10)
        applied = sv.apply_patches(runner, {0x10: [("mem", 0x31400, 0xCAFEBABE)]})
        self.assertEqual(
            T._dm_read(runner.state, T.Const(0x31400), 4), T.Const(0xCAFEBABE)
        )
        self.assertEqual(applied, ["mem[0x31400] = 0xcafebabe"])

    def test_mem_patch_with_non_int_target_raises(self):
        runner = self._runner()
        with self.assertRaises(ValueError):
            sv.apply_patches(runner, {0x10: [("mem", "R0", 1)]})

    def test_unknown_kind_raises(self):
        runner = self._runner()
        with self.assertRaises(ValueError):
            sv.apply_patches(runner, {0x10: [("bogus", "R0", 1)]})

    def test_no_entry_for_this_pc_is_a_noop(self):
        runner = self._runner()
        applied = sv.apply_patches(runner, {0x1000: [("reg", "R0", 9)]})
        self.assertEqual(applied, [])
        self.assertEqual(runner.state.uregs[T.UREG_CODES["R0"]], T.Const(1))


class RunWithPatchesTest(unittest.TestCase):
    def test_patch_applied_before_the_triggering_pc_executes(self):
        # A patch at the CURRENT pc lands before that instruction runs: an
        # unconditional "21a" NOP-family instruction just advances pc, so
        # if the patch had not taken, the fork below (predicated on R0's
        # patched-vs-original value) could not distinguish the two cases.
        runner = sr.Runner(loader_memory(), 0x10, regs={"R0": 1}, diagnose_unknown=True)
        runner._cache[0x10] = _insn("21a", {}, length=4)
        halt = sv.run_with_patches(runner, {0x10: [("reg", "R0", 5)]}, max_steps=1)
        self.assertEqual(halt.reason, "max-steps")
        self.assertEqual(runner.state.uregs[T.UREG_CODES["R0"]], T.Const(5))

    def test_max_steps_budget_produces_a_max_steps_halt(self):
        runner = sr.Runner(loader_memory(), 0x10)
        runner._cache[0x10] = _insn("21a", {}, length=4)
        runner._cache[0x12] = _insn("21a", {}, length=4)
        halt = sv.run_with_patches(runner, {}, max_steps=2)
        self.assertEqual(halt.reason, "max-steps")
        self.assertEqual(runner.instructions, 2)


def _insn(name, fields, length=4, kind="confident"):
    Instruction = import_module("sharc_disasm").Instruction
    return Instruction(0, length, name, fields, kind=kind)


class AstatGroupsFromUnknownsTest(unittest.TestCase):
    def test_single_group(self):
        self.assertEqual(
            sv._astat_groups_from_unknowns(("cond=TF", "ASTATX.BTF")), {"BTF"}
        )

    def test_multiple_bits_same_group_collapse_to_one(self):
        self.assertEqual(
            sv._astat_groups_from_unknowns(
                ("cond=LT", "ASTATX.AF", "ASTATX.AN", "ASTATX.AZ", "ASTATX.AV")
            ),
            {"ALU"},
        )

    def test_two_different_registers_same_group(self):
        self.assertEqual(
            sv._astat_groups_from_unknowns(("cond=AC", "ASTATX.AC", "ASTATY.AC")),
            {"ALU"},
        )

    def test_mode1_only_is_empty(self):
        self.assertEqual(sv._astat_groups_from_unknowns(("cond=EQ", "MODE1")), set())

    def test_no_unknowns_is_empty(self):
        self.assertEqual(sv._astat_groups_from_unknowns(()), set())

    def test_reason_and_last_writer_notes_are_not_mistaken_for_bits(self):
        # note_reason()/note_writer() text (tools/sharc_run.py) never
        # matches _UNKNOWN_RE, so it must not spuriously contribute a group.
        unknowns = (
            "cond=AC",
            "ASTATX.AC",
            "last write to ASTATX at 0x10",
            "ASTATX reason: x",
        )
        self.assertEqual(sv._astat_groups_from_unknowns(unknowns), {"ALU"})


class SurveyStopCategoryTest(unittest.TestCase):
    def _stop(self, reason, text=""):
        halt = sr.Halt(reason, 0x10, "8a_rel", text)
        return sv.SurveyStop(
            halt=halt, runner=None, img=None, instructions=0, elapsed=0.0
        )

    def test_fork(self):
        self.assertEqual(self._stop("fork (2 successors): ...").category, "fork")

    def test_mmr(self):
        self.assertEqual(self._stop("mmr").category, "mmr")

    def test_watchpoint(self):
        self.assertEqual(self._stop("watchpoint").category, "watchpoint")

    def test_frame_returned(self):
        self.assertEqual(
            self._stop("return without followed call").category, "frame-returned"
        )

    def test_max_steps(self):
        self.assertEqual(self._stop("max-steps").category, "max-steps")

    def test_return_mismatch(self):
        reason = "return target 0x20 differs from recorded return 0x30"
        self.assertEqual(self._stop(reason).category, "return-mismatch")

    def test_breakpoint_falls_back_to_the_raw_reason(self):
        self.assertEqual(self._stop("breakpoint").category, "breakpoint")


@unittest.skipUnless(
    DT2_116_DB.exists() and DT2_116_BLOB.exists(),
    "out/sharcdb/dt2-1.16.sqlite or out/sections/dt2-1.16/section_7_BLOB.bin "
    "is not available",
)
class SurveyEndToEndTest(unittest.TestCase):
    """FUN_1c2b24 (the frame render root) from a bare state: the exact
    first stop lane-L7's survey found (docs/findings/06, "envelope or gain"
    reads its own unresolved R8 argument) -- a real, stable, deterministic
    execution of the actual firmware, not a synthetic fixture."""

    START = 0x1C2B24
    EXPECTED_HALT_PC = 0xB88E4B
    EXPECTED_INSTRUCTIONS = 15675

    def test_bare_survey_reaches_the_known_first_fork(self):
        stop = sv.survey(image="dt2-1.16", root=self.START)
        self.assertEqual(stop.category, "fork")
        self.assertEqual(stop.halt.pc_sw, self.EXPECTED_HALT_PC)
        self.assertEqual(stop.instructions, self.EXPECTED_INSTRUCTIONS)
        self.assertIn("cond=EQ", stop.halt.unknowns)
        self.assertIn("ASTATX.AZ", stop.halt.unknowns)
        self.assertTrue(
            any(u.startswith("last write to ASTATX at ") for u in stop.halt.unknowns)
        )

    def test_patch_table_advances_past_the_known_fork(self):
        patch_table = {0xB88E04: [("reg", "R8", 0x40000000)]}
        stop = sv.survey(image="dt2-1.16", root=self.START, patch_table=patch_table)
        self.assertGreater(stop.instructions, self.EXPECTED_INSTRUCTIONS)
        self.assertNotEqual(stop.halt.pc_sw, self.EXPECTED_HALT_PC)

    def test_report_stop_prints_the_documented_fields(self):
        stop = sv.survey(image="dt2-1.16", root=self.START)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            sv.report_stop(stop)
        text = buf.getvalue()
        self.assertIn("pc=0x%x" % self.EXPECTED_HALT_PC, text)
        self.assertIn("form=5a_move", text)
        self.assertIn("ASTATX.AZ", text)
        self.assertIn("last flag writer:", text)
        self.assertIn("slice (all inputs):", text)
        self.assertIn("call stack (return addrs, innermost last):", text)
        self.assertIn("FUN_", text)  # some function name resolved somewhere

    def test_json_output_round_trips_through_main(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            sv.main(
                [
                    "dt2-1.16",
                    "--root",
                    hex(self.START),
                    "--max-steps",
                    str(self.EXPECTED_INSTRUCTIONS + 10),
                    "--json",
                ]
            )
        payload = json.loads(buf.getvalue())
        self.assertEqual(payload["reason"].startswith("fork"), True)
        self.assertEqual(payload["pc_sw"], self.EXPECTED_HALT_PC)
        self.assertEqual(payload["category"], "fork")
        self.assertIn("ASTATX", payload["last_writer"])
        self.assertIsInstance(payload["call_stack"], list)


if __name__ == "__main__":
    unittest.main()
