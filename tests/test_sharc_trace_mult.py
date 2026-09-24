"""Tests for the multiplier (cu=0x1) full-compute opcodes added to
tools/sharc_trace.py: PRM Table 17-7 "MULOP Encode Table" (out/refs/
sharc-plus-prm/all.txt, p.17-7/17-8) rows this tracer did not previously
decode.

Kept in a separate file (rather than tests/test_sharc_trace.py) so this
opcode-set's edits do not collide with other agents' concurrent edits to
that file.
"""

import os
import sys
import unittest
from importlib import import_module

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools"))
T = import_module("sharc_trace")
Instruction = import_module("sharc_disasm").Instruction


def insn(name, fields, length=4, kind="confident"):
    return Instruction(0, length, name, fields, kind=kind)


def full_compute(cu, opcode, rn, rx, ry):
    """A full-compute field dict for _compute(f, short=False, ...): mirrors
    tests/test_sharc_trace.py's helper of the same name (PRM Figure 18-1,
    p.423: mf=0, cu[21:20], opcode[19:12], rn[11:8], rx[7:4], ry[3:0])."""
    field = (cu << 20) | (opcode << 12) | (rn << 8) | (rx << 4) | ry
    return {"compute[22:16]": field >> 16, "compute[15:0]": field & 0xFFFF}


def signed32(value):
    value &= 0xFFFFFFFF
    return value - (1 << 32) if value & 0x80000000 else value


class MultiplierFractionalHelperTest(unittest.TestCase):
    """Direct tests of _multiply_fractional, the shared 1.31/0.32 helper
    behind the new fractional-format opcodes."""

    def test_unsigned_fractional_exact_half_times_half(self):
        # 0.5 * 0.5 = 0.25 in U0.32 (PRM p.27-4): 0x80000000 * 0x80000000,
        # top 32 bits of the 64-bit product (PRM Figure 3-2, p.3-10).
        value = T._multiply_fractional(
            T.Const(0x80000000), T.Const(0x80000000), False, False, "test"
        )
        self.assertEqual(value, T.Const(0x40000000))

    def test_unsigned_fractional_near_one_times_near_one(self):
        value = T._multiply_fractional(
            T.Const(0xFFFFFFFF), T.Const(0xFFFFFFFF), False, False, "test"
        )
        self.assertEqual(value, T.Const(0xFFFFFFFE))

    def test_signed_fractional_negative_half_times_half(self):
        # -0.5 * 0.5 = -0.25 in 1.31 (PRM p.27-4), with the redundant-sign
        # left shift for signed*signed (PRM p.3-9) folded into >>31.
        value = T._multiply_fractional(
            T.Const(0xC0000000), T.Const(0x40000000), True, True, "test"
        )
        self.assertEqual(value, T.Const(0xE0000000))

    def test_unknown_operand_stays_unknown(self):
        value = T._multiply_fractional(T.Unknown("x"), T.Const(1), True, True, "expr")
        self.assertEqual(value, T.Unknown("expr"))

    def test_symbolic_affine_operand_stays_unknown(self):
        value = T._multiply_fractional(
            T.symbol("track_index"), T.Const(1), True, True, "expr"
        )
        self.assertEqual(value, T.Unknown("expr"))


class MultiplierComputeOpcodeTest(unittest.TestCase):
    """One numeric case per new cu=0x1 opcode, called directly through
    _compute (bypassing _execute) the same way
    tests/test_sharc_trace.py's AstatxFlagsTest does."""

    def compute(self, fields, values, special=None):
        return T._compute(fields, False, values, special)

    # -- 0x40: RN = RX*RY MOD1, UUI (PRM p.17-7/17-9) -----------------------

    def test_0x40_multiply_uui_wraps_mod_2_32(self):
        rn, value, op, astatx_update = self.compute(
            full_compute(1, 0x40, 5, 1, 2),
            {1: T.Const(0x80000000), 2: T.Const(3)},
        )
        self.assertEqual(rn, 5)
        self.assertEqual(value, T.Const(0x80000000))  # low 32 bits of 3*2**31
        self.assertEqual(op, "multiply")
        astatx = astatx_update(T.Const(0xFFFFFFFF))
        expected_mask = 0xFFFFFFFF & ~(
            (1 << T.MN_BIT) | (1 << T.MV_BIT) | (1 << T.MU_BIT)
        )
        self.assertEqual(
            astatx, T.PartialConst(expected_mask, expected_mask & ~(1 << T.MI_BIT))
        )

    # -- 0x48: RN = RX*RY MOD1, UUF (PRM p.17-7/17-9) -----------------------

    def test_0x48_multiply_uuf_takes_top_32_bits(self):
        rn, value, op, _ = self.compute(
            full_compute(1, 0x48, 5, 1, 2),
            {1: T.Const(0x80000000), 2: T.Const(0x80000000)},
        )
        self.assertEqual(rn, 5)
        self.assertEqual(value, T.Const(0x40000000))
        self.assertEqual(op, "multiply")

    # -- 0x74: mrf = RX*RY MOD1, SSI (no accumulate; PRM p.17-7/17-9) -------

    def test_0x74_multiply_mrf_ssi_negative_operand(self):
        rn, value, op, _ = self.compute(
            full_compute(1, 0x74, 0, 1, 2),
            {1: T.Const(0xFFFFFFFB), 2: T.Const(6)},  # -5 * 6 = -30
        )
        self.assertEqual(rn, "MRF")
        self.assertEqual(value, T.Const(0xFFFFFFE2))
        self.assertEqual(op, "multiply-mrf")

    # -- 0x7C: mrf = RX*RY MOD1, SSF (no accumulate; PRM p.17-7/17-9) -------

    def test_0x7c_multiply_mrf_ssf_negative_fractional(self):
        rn, value, op, _ = self.compute(
            full_compute(1, 0x7C, 0, 1, 2),
            {1: T.Const(0xC0000000), 2: T.Const(0x40000000)},  # -0.5 * 0.5
        )
        self.assertEqual(rn, "MRF")
        self.assertEqual(value, T.Const(0xE0000000))
        self.assertEqual(op, "multiply-mrf")

    # -- 0xBC: mrf = mrf + RX*RY MOD1, SSF (fractional twin of 0xB4) --------

    def test_0xbc_multiply_accumulate_ssf_positive(self):
        rn, value, op, astatx_update = self.compute(
            full_compute(1, 0xBC, 0, 1, 2),
            {1: T.Const(0x40000000), 2: T.Const(0x40000000)},  # 0.5 * 0.5
            special={"MRF": T.Const(0x10000000)},  # + 0.125
        )
        self.assertEqual(rn, "MRF")
        self.assertEqual(value, T.Const(0x30000000))  # 0.375
        self.assertEqual(op, "multiply-accumulate")
        astatx = astatx_update(T.Unknown("start"))
        self.assertEqual(astatx, T.PartialConst(1 << T.MI_BIT, 0))

    def test_0xbc_multiply_accumulate_ssf_negative_product(self):
        rn, value, op, _ = self.compute(
            full_compute(1, 0xBC, 0, 1, 2),
            {1: T.Const(0xC0000000), 2: T.Const(0x40000000)},  # -0.5 * 0.5
            special={"MRF": T.Const(0x40000000)},  # + 0.5
        )
        self.assertEqual(rn, "MRF")
        self.assertEqual(value, T.Const(0x20000000))  # 0.5 + (-0.25) = 0.25
        self.assertEqual(op, "multiply-accumulate")

    def test_0xbc_uninitialized_mrf_is_unknown(self):
        rn, value, op, _ = self.compute(
            full_compute(1, 0xBC, 0, 1, 2),
            {1: T.Const(0x40000000), 2: T.Const(0x40000000)},
            special={},
        )
        self.assertEqual(rn, "MRF")
        self.assertIsInstance(value, T.Unknown)
        self.assertEqual(op, "multiply-accumulate")

    # -- 0x09: RN = sat mrf MOD2, SF (PRM p.17-7/17-9; unmodeled 80-bit) ----

    def test_0x09_saturate_mrf_sf_stays_unknown_with_documented_flags(self):
        rn, value, op, astatx_update = self.compute(full_compute(1, 0x09, 3, 0, 0), {})
        self.assertEqual(rn, 3)
        self.assertIsInstance(value, T.Unknown)
        self.assertEqual(op, "saturate-mrf")
        astatx = astatx_update(T.Const(0xFFFFFFFF))
        expected_mask = 0xFFFFFFFF & ~((1 << T.MN_BIT) | (1 << T.MV_BIT))
        expected_bits = expected_mask & ~((1 << T.MU_BIT) | (1 << T.MI_BIT))
        self.assertEqual(astatx, T.PartialConst(expected_mask, expected_bits))

    # -- 0x10: undocumented in both public sources ---------------------------

    def test_0x10_undocumented_opcode_decodes_conservatively(self):
        rn, value, op, astatx_update = self.compute(
            full_compute(1, 0x10, 3, 1, 2), {1: T.Const(5), 2: T.Const(6)}
        )
        self.assertEqual(rn, 3)
        self.assertIsInstance(value, T.Unknown)
        self.assertEqual(op, "multiply-undocumented-10")
        astatx = astatx_update(T.Const(0xFFFFFFFF))
        self.assertEqual(
            astatx,
            T.PartialConst(
                0xFFFFFFFF & ~T.MULT_FLAGS_MASK, 0xFFFFFFFF & ~T.MULT_FLAGS_MASK
            ),
        )


class MultiplierComputeIntegrationTest(unittest.TestCase):
    """End-to-end through _execute/_apply_compute (Type 2a, always-true
    condition), mirroring test_sharc_trace.py's
    test_full_compute_register_to_mr_move_is_recorded."""

    def run_one(self, state, record):
        return T._execute(state, record)[0]

    def test_fractional_load_then_accumulate_then_multiply_sequence(self):
        state = T.State(
            1,
            {
                T.UREG_CODES["R1"]: T.Const(0x40000000),  # 0.5
                T.UREG_CODES["R2"]: T.Const(0x40000000),  # 0.5
            },
        )
        # mrf = R1*R2 (SSF), opcode 0x7C: loads mrf with no prior accumulator
        # state, unlike 0xBC below which needs one already established.
        field = (1 << 20) | (0x7C << 12) | (1 << 4) | 2
        state = self.run_one(
            state,
            insn(
                "2a",
                {
                    "cond[4:0]": 31,
                    "compute[22:16]": field >> 16,
                    "compute[15:0]": field & 0xFFFF,
                },
                6,
            ),
        )
        self.assertEqual(state.special["MRF"], T.Const(0x20000000))  # 0.25
        self.assertEqual(state.trace[-1]["operation"], "multiply-mrf")

        # mrf = mrf + R1*R2 (SSF), opcode 0xBC.
        field = (1 << 20) | (0xBC << 12) | (1 << 4) | 2
        state = self.run_one(
            state,
            insn(
                "2a",
                {
                    "cond[4:0]": 31,
                    "compute[22:16]": field >> 16,
                    "compute[15:0]": field & 0xFFFF,
                },
                6,
            ),
        )
        self.assertEqual(state.special["MRF"], T.Const(0x40000000))  # 0.5
        self.assertEqual(state.trace[-1]["operation"], "multiply-accumulate")

        # RN = R1*R2 (UUI), opcode 0x40: 2**30 * 2**30 = 2**60, whose low
        # 32 bits (the integer-format RN result) are 0.
        field = (1 << 20) | (0x40 << 12) | (5 << 8) | (1 << 4) | 2
        state = self.run_one(
            state,
            insn(
                "2a",
                {
                    "cond[4:0]": 31,
                    "compute[22:16]": field >> 16,
                    "compute[15:0]": field & 0xFFFF,
                },
                6,
            ),
        )
        self.assertEqual(state.uregs[T.UREG_CODES["R5"]], T.Const(0))
        self.assertEqual(state.trace[-1]["operation"], "multiply")


if __name__ == "__main__":
    unittest.main()
