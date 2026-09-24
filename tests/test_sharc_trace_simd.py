"""SIMD (PEx + PEy) execution tests for tools/sharc_trace.py.

Companion to tests/test_sharc_trace.py (not edited here, per instructions):
synthetic instruction-level tests for the PEy register file, per-PE
computation, per-PE conditions, SIMD-mode data-move companions, and one
real-image test over the render path's actual SIMD-enabled region
(0xb80105, called from FUN_1c642a at 0x1c65b3 and 0x1c6c72).
"""

import hashlib
import os
import sys
import unittest
from importlib import import_module

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools"))
sys.path.insert(0, os.path.dirname(__file__))
T = import_module("sharc_trace")
Instruction = import_module("sharc_disasm").Instruction
L = import_module("sharcldr")
from test_sharc_trace import insn, loader_memory  # noqa: E402  (shared fixtures)


def short_compute(opcode, rn, rx):
    return {"compute[11:0]": (opcode << 8) | (rn << 4) | rx}


class TraceHelpers(unittest.TestCase):
    def run_one(self, state, record):
        out = T._execute(state, record)
        self.assertEqual(len(out), 1, out)
        return out[0]


class CuregHelpersTest(unittest.TestCase):
    def test_r_and_s_registers_are_complementary_pairs(self):
        for code in range(16):
            self.assertEqual(T._cureg_code(code), 80 + code)
            self.assertEqual(T._cureg_code(80 + code), code)

    def test_astat_stky_ustat_px_pairs(self):
        for a, b in (
            ("ASTATX", "ASTATY"),
            ("STKYX", "STKYY"),
            ("USTAT1", "USTAT2"),
            ("USTAT3", "USTAT4"),
            ("PX1", "PX2"),
        ):
            self.assertEqual(T._cureg_code(T.UREG_CODES[a]), T.UREG_CODES[b])
            self.assertEqual(T._cureg_code(T.UREG_CODES[b]), T.UREG_CODES[a])

    def test_dag_and_combined_px_have_no_complement(self):
        # SHARC+ PRM p.15-12: "MODE1 (SREG) and LCNTR (UREG) ... have no
        # complements"; p.59: the combined PX register is itself
        # uncomplementary even though PX1/PX2 are.
        for name in ("I0", "M3", "L7", "B15", "PX", "MODE1", "LCNTR", "PC"):
            self.assertIsNone(T._cureg_code(T.UREG_CODES[name]))

    def test_simd_active_reads_mode1_bit_21(self):
        self.assertIsNone(T._simd_active(T.State(0, {})))
        self.assertFalse(
            T._simd_active(T.State(0, {T.UREG_CODES["MODE1"]: T.Const(0)}))
        )
        self.assertTrue(
            T._simd_active(T.State(0, {T.UREG_CODES["MODE1"]: T.Const(1 << 21)}))
        )


class SimdComputeTest(TraceHelpers):
    """Type2c (short compute): "R1 = R1 + R2" duplicated onto "S1 = S1 +
    S2" in SIMD mode (SHARC+ PRM p.101, p.3-39)."""

    def registers(self, mode1):
        regs = {
            1: T.Const(1),
            2: T.Const(-1 & 0xFFFFFFFF),  # R1 + R2 == 0 -> PEx AZ set
            81: T.Const(5),
            82: T.Const(3),  # S1 + S2 == 8 -> PEy AZ clear
        }
        if mode1 is not None:
            regs[T.UREG_CODES["MODE1"]] = T.Const(mode1)
        return regs

    def test_simd_active_duplicates_the_compute_onto_pey_independently(self):
        state = self.run_one(
            T.State(1, self.registers(1 << 21)), insn("2c", short_compute(0, 1, 2), 2)
        )
        self.assertEqual(state.uregs[1], T.Const(0))
        self.assertEqual(state.uregs[81], T.Const(8))
        astatx = state.uregs[T.UREG_CODES["ASTATX"]]
        astaty = state.uregs[T.UREG_CODES["ASTATY"]]
        self.assertTrue(T._astatx_known_bit(astatx, T.AZ_BIT))
        self.assertFalse(T._astatx_known_bit(astaty, T.AZ_BIT))
        pey_events = [e for e in state.trace if e["action"] == "compute-pey"]
        self.assertEqual(len(pey_events), 1)
        self.assertEqual(pey_events[0]["result_register"], "S1")
        self.assertEqual(pey_events[0]["value"], 8)

    def test_sisd_mode_never_touches_the_s_register_file(self):
        state = self.run_one(
            T.State(1, self.registers(0)), insn("2c", short_compute(0, 1, 2), 2)
        )
        self.assertEqual(state.uregs[1], T.Const(0))
        self.assertEqual(state.uregs[81], T.Const(5))  # unchanged
        self.assertEqual(state.uregs[82], T.Const(3))  # unchanged
        self.assertFalse(any(e["action"] == "compute-pey" for e in state.trace))

    def test_unresolved_mode1_behaves_like_sisd_not_a_stop(self):
        state = self.run_one(
            T.State(1, self.registers(None)), insn("2c", short_compute(0, 1, 2), 2)
        )
        self.assertIsNone(state.stopped)
        self.assertEqual(state.uregs[1], T.Const(0))
        self.assertEqual(state.uregs[81], T.Const(5))  # unchanged
        self.assertFalse(any(e["action"] == "compute-pey" for e in state.trace))

    def test_full_compute_2a_short_also_duplicates_in_simd(self):
        # cu=0 (fixed ALU), op selecting "add" (PRM full-compute table),
        # matching test_computes_and_old_value_parallel_move's own helper.
        def full(cu, op, rn, rx, ry):
            return {
                "compute[22:16]": ((cu << 4) | (op >> 4)),
                "compute[15:0]": ((op & 15) << 12) | (rn << 8) | (rx << 4) | ry,
            }

        fields = full(0, 0x01, 0, 1, 2)  # R0 = R1 + R2 (cu=0, op=0x01: ALU add)
        regs = {
            1: T.Const(10),
            2: T.Const(20),
            81: T.Const(100),
            82: T.Const(200),
            T.UREG_CODES["MODE1"]: T.Const(1 << 21),
        }
        state = self.run_one(T.State(1, regs), insn("2a_short", fields, 6))
        self.assertEqual(state.uregs[0], T.Const(30))
        self.assertEqual(state.uregs[80], T.Const(300))


class SimdRegisterMoveTest(TraceHelpers):
    """Type5b_move (unconditional ureg-to-ureg copy): SIMD companion rules
    from SHARC+ PRM p.28 and p.4-55 Table 4-22."""

    def test_uncomplementary_source_loads_both_pes_from_the_one_source(self):
        # "R5 = I8; loads R5 and S5 with I8" -- PRM p.28's own example.
        fields = {
            "srcureghigh[4:0]": 6,
            "srcureglow[1:1]": 0,
            "srcureglow[0:0]": 0,
            "dstureg[6:0]": 5,
            "cond[4:0]": 31,
        }
        state = self.run_one(
            T.State(
                1,
                {24: T.Const(0x77), T.UREG_CODES["MODE1"]: T.Const(1 << 21)},
            ),
            insn("5b_move", fields),
        )
        self.assertEqual(state.uregs[5], T.Const(0x77))
        self.assertEqual(state.uregs[85], T.Const(0x77))
        self.assertEqual(
            state.trace[-1]["simd_companion"],
            {"source": "I8", "destination": "S5"},
        )

    def test_complementary_source_and_destination_move_each_pes_own_register(self):
        fields = {
            "srcureghigh[4:0]": 0,
            "srcureglow[1:1]": 0,
            "srcureglow[0:0]": 1,
            "dstureg[6:0]": 5,
            "cond[4:0]": 31,
        }
        state = self.run_one(
            T.State(
                1,
                {
                    1: T.Const(0xAA),
                    81: T.Const(0xBB),
                    T.UREG_CODES["MODE1"]: T.Const(1 << 21),
                },
            ),
            insn("5b_move", fields),
        )
        self.assertEqual(state.uregs[5], T.Const(0xAA))  # R5 = R1
        self.assertEqual(state.uregs[85], T.Const(0xBB))  # S5 = S1 (its own PE)

    def test_uncomplementary_destination_gets_no_implicit_move(self):
        fields = {
            "srcureghigh[4:0]": 1,
            "srcureglow[1:1]": 0,
            "srcureglow[0:0]": 0,
            "dstureg[6:0]": 20,  # I4: no SIMD complement
            "cond[4:0]": 31,
        }
        state = self.run_one(
            T.State(
                1,
                {
                    4: T.Const(0x55),
                    84: T.Const(0x66),
                    T.UREG_CODES["MODE1"]: T.Const(1 << 21),
                },
            ),
            insn("5b_move", fields),
        )
        self.assertEqual(state.uregs[20], T.Const(0x55))
        self.assertIsNone(state.trace[-1]["simd_companion"])
        self.assertEqual(state.uregs[84], T.Const(0x66))  # untouched

    def test_conditional_ureg_copy_is_not_simd_duplicated(self):
        # Documented scope limit: cond != 0x1F is left PEx-only (see the
        # handler's own comment); this pins that choice rather than
        # silently drifting.
        fields = {
            "srcureghigh[4:0]": 6,
            "srcureglow[1:1]": 0,
            "srcureglow[0:0]": 0,
            "dstureg[6:0]": 5,
            "cond[4:0]": 0x04,  # AV: anything but "always"
        }
        # ALU overflow (AV) set -> predicate True (SIMPLE_COND_BITS does
        # not bail out on SIMD mode the way EQ/NE's _predicate path does).
        state = self.run_one(
            T.State(
                1,
                {
                    24: T.Const(0x77),
                    T.UREG_CODES["MODE1"]: T.Const(1 << 21),
                    T.UREG_CODES["ASTATX"]: T.Const(1 << T.AV_BIT),
                },
            ),
            insn("5b_move", fields),
        )
        self.assertEqual(state.uregs[5], T.Const(0x77))
        self.assertNotIn(85, state.uregs)


class SimdMemoryCompanionTest(TraceHelpers):
    """Type14a/15a/3b UREG<->memory companion transfers (SHARC+ PRM p.212,
    Table 6-10; p.13-17's Type3b SIMD description)."""

    def test_type14a_store_and_load_companion_at_plus_one_normal_word(self):
        fields = {
            "addr[31:16]": 0x3000,
            "addr[15:0]": 0x1000,
            "g": 0,
            "d": 1,
            "l": 0,
            "ureg[6:0]": 2,  # R2 (complementary)
        }
        regs = {
            2: T.Const(0x11111111),
            82: T.Const(0x22222222),
            T.UREG_CODES["MODE1"]: T.Const(1 << 21),
        }
        state = self.run_one(
            T.State(1, regs, concrete=loader_memory(), assume_nw32=True),
            insn("14a", fields, 6),
        )
        self.assertIsNone(state.stopped)
        self.assertEqual(state.trace[0]["simd_companion_possible"], True)
        pey = [e for e in state.trace if e["action"] == "store-pey"]
        self.assertEqual(len(pey), 1)
        self.assertEqual(pey[0]["ureg"], "S2")
        self.assertEqual(pey[0]["address"], 0x30001000 + 4)
        self.assertEqual(pey[0]["value"], 0x22222222)
        self.assertEqual(T._dm_read(state, 0x30001000, 4), T.Const(0x11111111))
        self.assertEqual(T._dm_read(state, 0x30001004, 4), T.Const(0x22222222))

        # Load direction, same addresses (poke both words first).
        load_state = T.State(
            1,
            {T.UREG_CODES["MODE1"]: T.Const(1 << 21)},
            concrete=loader_memory(),
            assume_nw32=True,
        )
        self.assertTrue(T._dm_write(load_state, 0x30001000, 4, T.Const(0xAAAA)))
        self.assertTrue(T._dm_write(load_state, 0x30001004, 4, T.Const(0xBBBB)))
        loaded = self.run_one(load_state, insn("14a", {**fields, "d": 0}, 6))
        self.assertEqual(loaded.uregs[2], T.Const(0xAAAA))
        self.assertEqual(loaded.uregs[82], T.Const(0xBBBB))
        pey_load = [e for e in loaded.trace if e["action"] == "load-pey"]
        self.assertEqual(pey_load[0]["ureg"], "S2")
        self.assertEqual(pey_load[0]["concrete_value"], 0xBBBB)

    def test_type14a_uncomplementary_ureg_has_no_companion_in_simd(self):
        fields = {
            "addr[31:16]": 0x3000,
            "addr[15:0]": 0x1000,
            "g": 0,
            "d": 1,
            "l": 0,
            "ureg[6:0]": 16,  # I0: no SIMD complement
        }
        state = self.run_one(
            T.State(
                1,
                {16: T.Const(1), T.UREG_CODES["MODE1"]: T.Const(1 << 21)},
                concrete=loader_memory(),
                assume_nw32=True,
            ),
            insn("14a", fields, 6),
        )
        self.assertFalse(any(e["action"] == "store-pey" for e in state.trace))

    def test_type14a_unresolved_mode1_stores_once_without_stopping(self):
        fields = {
            "addr[31:16]": 0x3000,
            "addr[15:0]": 0x1000,
            "g": 0,
            "d": 1,
            "l": 0,
            "ureg[6:0]": 2,  # complementary, but MODE1 is unknown
        }
        state = self.run_one(
            T.State(1, {2: T.Const(9)}, concrete=loader_memory(), assume_nw32=True),
            insn("14a", fields, 6),
        )
        self.assertIsNone(state.stopped)
        self.assertFalse(any(e["action"] == "store-pey" for e in state.trace))
        self.assertEqual(T._dm_read(state, 0x30001000, 4), T.Const(9))

    def test_type3b_post_modify_store_companion_and_index_update(self):
        fields = {
            "i[2:0]": 1,
            "m[2:0]": 2,
            "g": 0,
            "d": 1,
            "u": 1,
            "l": 0,
            "x": 1,
            "w": 1,  # normal-word
            "ureg[6:0]": 2,  # R2
            "cond[4:0]": 31,
        }
        regs = {
            17: T.Const(0x30001000),  # I1
            34: T.Const(4),  # M2
            2: T.Const(0x11111111),
            82: T.Const(0x22222222),
            T.UREG_CODES["MODE1"]: T.Const(1 << 21),
        }
        state = self.run_one(
            T.State(1, regs, concrete=loader_memory(), assume_nw32=True),
            insn("3b", fields),
        )
        self.assertEqual(T._dm_read(state, 0x30001000, 4), T.Const(0x11111111))
        self.assertEqual(T._dm_read(state, 0x30001004, 4), T.Const(0x22222222))
        # I1 post-modifies by M2 scaled to bytes (assume_nw32): 4 * 4 = 16.
        self.assertEqual(state.uregs[17], T.Const(0x30001010))
        pey = [e for e in state.trace if e["action"] == "store-pey"]
        self.assertEqual(pey[0]["address"], 0x30001004)

    def test_type3b_byte_access_with_complementary_ureg_stops_in_simd(self):
        # PRM pp.222-223's byte-space SIMD companion rule is not modelled;
        # this must stop rather than silently transfer only PEx's half.
        fields = {
            "i[2:0]": 1,
            "m[2:0]": 2,
            "g": 0,
            "d": 1,
            "u": 1,
            "l": 0,
            "x": 0,
            "w": 0,  # byte
            "ureg[6:0]": 2,  # R2 (complementary)
            "cond[4:0]": 31,
        }
        regs = {
            17: T.Const(0x30001000),
            34: T.Const(4),
            2: T.Const(1),
            T.UREG_CODES["MODE1"]: T.Const(1 << 21),
        }
        state = self.run_one(
            T.State(1, regs, concrete=loader_memory(), assume_nw32=True),
            insn("3b", fields),
        )
        self.assertIsNotNone(state.stopped)
        self.assertIn("unsupported SIMD companion access width", state.stopped)

    def test_type3b_long_word_never_gets_a_companion_even_in_simd(self):
        # PRM p.13-17: "(LW) ... override[s] SIMD mode, so these loads
        # always operate in SISD mode" -- a documented no-companion case.
        fields = {
            "i[2:0]": 1,
            "m[2:0]": 2,
            "g": 0,
            "d": 1,
            "u": 1,
            "l": 1,
            "x": 1,
            "w": 1,  # long-word
            "ureg[6:0]": 2,
            "cond[4:0]": 31,
        }
        regs = {
            17: T.Const(0x30001000),
            34: T.Const(4),
            2: T.Const(0x11111111),
            T.UREG_CODES["MODE1"]: T.Const(1 << 21),
        }
        state = self.run_one(
            T.State(1, regs, concrete=loader_memory(), assume_nw32=True),
            insn("3b", fields),
        )
        self.assertIsNone(state.stopped)
        self.assertFalse(any(e["action"] == "store-pey" for e in state.trace))

    def test_type15a_store_companion(self):
        fields = {
            "i[2:0]": 0,
            "g": 0,
            "d": 1,
            "l": 0,
            "ureg[6:0]": 4,  # R4
            "addr[31:16]": 0,
            "addr[15:0]": 0,
        }
        regs = {
            16: T.Const(0x30002000),  # I0
            4: T.Const(0xDEAD),
            84: T.Const(0xBEEF),
            T.UREG_CODES["MODE1"]: T.Const(1 << 21),
        }
        state = self.run_one(
            T.State(1, regs, concrete=loader_memory(), assume_nw32=True),
            insn("15a", fields, 6),
        )
        self.assertEqual(T._dm_read(state, 0x30002000, 4), T.Const(0xDEAD))
        self.assertEqual(T._dm_read(state, 0x30002004, 4), T.Const(0xBEEF))


class SimdBranchPredicateTest(TraceHelpers):
    """SHARC+ PRM p.4-54, Table 4-22: a branch/call/return ANDs PEx's and
    PEy's own conditions in SIMD mode."""

    def test_predicate_simd_branch_ands_divergent_pe_conditions(self):
        state = T.State(
            0,
            {
                T.UREG_CODES["MODE1"]: T.Const(1 << 21),
                T.UREG_CODES["ASTATX"]: T.Const(1 << T.AZ_BIT),  # PEx: EQ true
                T.UREG_CODES["ASTATY"]: T.Const(0),  # PEy: EQ false
            },
        )
        self.assertFalse(T._predicate_simd_branch(state, 0x00))  # EQ

    def test_predicate_simd_branch_is_pex_only_in_sisd(self):
        state = T.State(
            0,
            {
                T.UREG_CODES["MODE1"]: T.Const(0),
                T.UREG_CODES["ASTATX"]: T.Const(1 << T.AZ_BIT),
                T.UREG_CODES["ASTATY"]: T.Const(0),
            },
        )
        self.assertTrue(T._predicate_simd_branch(state, 0x00))

    def test_predicate_simd_branch_unresolved_mode1_resolves_when_modes_agree(self):
        # MODE1 unknown: SISD gives PEx, SIMD gives PEx AND PEy (EQ = cond 0x00).
        equal, unequal = T.Const(1 << T.AZ_BIT), T.Const(0)
        cases = (
            (equal, equal, True),  # both modes take the branch
            (unequal, equal, False),  # PEx false: false in both modes
            (unequal, T.Unknown("PEy"), False),
            (equal, unequal, None),  # SISD takes it, SIMD does not
            (equal, T.Unknown("PEy"), None),
        )
        for astatx, astaty, expected in cases:
            state = T.State(
                0,
                {T.UREG_CODES["ASTATX"]: astatx, T.UREG_CODES["ASTATY"]: astaty},
            )
            with self.subTest(astatx=astatx, astaty=astaty):
                self.assertIs(T._predicate_simd_branch(state, 0x00), expected)
                # The "always" condition never depends on PE state.
                self.assertTrue(T._predicate_simd_branch(state, 0x1F))

    def test_9b_abs_branch_is_not_taken_when_pey_disagrees(self):
        fields = {
            "b": 0,
            "a": 0,
            "cond[4:0]": 0x00,  # EQ
            "pmi[2:2]": 1,
            "pmi[1:0]": 1,
            "pmm[2:0]": 5,
            "j": 1,
            "ci": 0,
        }
        registers = {
            T.UREG_CODES["I13"]: T.Const(0x300),
            T.UREG_CODES["M13"]: T.Const(0),
            T.UREG_CODES["MODE1"]: T.Const(1 << 21),
            T.UREG_CODES["ASTATX"]: T.Const(1 << T.AZ_BIT),
            T.UREG_CODES["ASTATY"]: T.Const(0),
        }
        state = self.run_one(T.State(0x10, registers), insn("9b_abs", fields))
        self.assertIsNone(state.pending)
        self.assertEqual(state.pc_sw, 0x12)

    def test_9b_abs_branch_is_taken_pex_only_when_sisd(self):
        fields = {
            "b": 0,
            "a": 0,
            "cond[4:0]": 0x00,
            "pmi[2:2]": 1,
            "pmi[1:0]": 1,
            "pmm[2:0]": 5,
            "j": 1,
            "ci": 0,
        }
        registers = {
            T.UREG_CODES["I13"]: T.Const(0x300),
            T.UREG_CODES["M13"]: T.Const(0),
            T.UREG_CODES["MODE1"]: T.Const(0),
            T.UREG_CODES["ASTATX"]: T.Const(1 << T.AZ_BIT),
            T.UREG_CODES["ASTATY"]: T.Const(0),
        }
        state = self.run_one(T.State(0x10, registers), insn("9b_abs", fields))
        self.assertIsNotNone(state.pending)
        self.assertEqual(state.pending.target, 0x300)


class RealBlobSimdRegionTest(unittest.TestCase):
    """dt2-1.16's render path enables SIMD around 0xb80105 (called from
    FUN_1c642a at 0x1c65b3/0x1c6c72): MODE1.PEYEN is set at 0xb8012f and
    cleared at 0xb80150. Skips cleanly when the firmware isn't present
    (CLAUDE.md: never commit sections/out; matches this repo's
    skip-not-fail convention, e.g. tests/test_sharc_trace.py's
    RealBlobStage6NormalWordAddressingTest)."""

    BLOB = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "out",
        "sections",
        "dt2-1.16",
        "section_7_BLOB.bin",
    )
    BLOB_SHA256 = "0f514a12a2255f5c081e292c47f1f29462003177658da4bbae0a22fd737fffa2"
    START = 0xB80105
    PEYEN_SET_PC = 0xB8012F
    PEYEN_CLEAR_PC = 0xB80150

    @unittest.skipUnless(
        os.path.exists(BLOB),
        "out/sections/dt2-1.16/section_7_BLOB.bin is not available",
    )
    def test_simd_window_touches_only_uncomplementary_uregs(self):
        with open(self.BLOB, "rb") as fh:
            data = fh.read()
        self.assertEqual(hashlib.sha256(data).hexdigest(), self.BLOB_SHA256)
        memory = L.LoadedMemory.from_stream(data)
        states = T.trace(
            memory,
            None,
            self.START,
            {"I6": 0x90001000, "I7": 0x90000FF8},
            max_steps=60,
            max_states=32,
            concrete_memory=True,
            follow_loaded_calls=True,
            assume_nw32=True,
        )
        self.assertTrue(states)
        reached_clear = [
            s
            for s in states
            if any(e.get("pc_sw") == self.PEYEN_CLEAR_PC for e in s.trace)
        ]
        self.assertTrue(
            reached_clear,
            "no traced path reached the MODE1.PEYEN clear at 0xb80150",
        )
        for state in reached_clear:
            pc_range = range(self.PEYEN_SET_PC, self.PEYEN_CLEAR_PC + 1)
            window = [e for e in state.trace if e.get("pc_sw") in pc_range]
            self.assertTrue(window)
            # PEx's side of the window ran (housekeeping DAG stores of
            # M13/I12/0x0 -- all uncomplementary UREGs, PRM p.15-12).
            self.assertTrue(
                any(e.get("action") in ("store", "system-bit-op") for e in window)
            )
            # PEy's side never fired: none of this window's UREGs
            # (M13, I12, M4) have a SIMD complement, so this tracer's
            # companion logic correctly finds nothing to duplicate.
            pey_actions = {"store-pey", "load-pey", "compute-pey"}
            self.assertFalse(
                any(e.get("action") in pey_actions for e in state.trace),
                "a SIMD companion fired for a window that has none",
            )
            # PEy's own register file (S0-S15) and REGF_ASTATY were never
            # written -- the second half of "checking both PEs' results"
            # for a region where PEx is the only one doing anything.
            s_codes = range(80, 96)
            self.assertFalse(
                any(code in state.uregs for code in s_codes),
                "PEy's S register file was touched by an all-uncomplementary window",
            )
            self.assertNotIn(T.UREG_CODES["ASTATY"], state.uregs)
            # MODE1.PEYEN ends cleared (0xb80150 ran).
            mode1 = state.uregs.get(T.UREG_CODES["MODE1"])
            if isinstance(mode1, T.Const):
                self.assertEqual(mode1.value & (1 << 21), 0)


if __name__ == "__main__":
    unittest.main()
