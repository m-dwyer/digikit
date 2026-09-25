"""Tests for the 64-bit (IEEE double) compute ops added in lane F1: cu=0
opcodes 0x11-0x1f (ADSP-SC58x/2158x PRM Table 18-6, "ALUOP Encoding
(64-bit floating-point operations)"), cu=1 opcodes 0x31-0x33 (Table 18-8,
"MULOP Encode Table (64-bit floating-point operations)"), and the cu=3
opcode 0xe0 (still undocumented; only checked here for "does not raise").

Each case is worked from the manual definition cited in the implementation
(out/refs/sc58x-2158x-prm), independently of tools/sharc_core's own
arithmetic, so a broken implementation cannot pass by construction. Every
expected register-pair bit pattern is computed here with a hand-rolled
``f64`` helper (plain struct pack/unpack), not by importing anything from
tools/sharc_core/floats.py.
"""

import os
import struct
import sys
import unittest
from importlib import import_module

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools"))
T = import_module("sharc_trace")
Instruction = import_module("sharc_disasm").Instruction


def f64(value):
    """(hi, lo) 32-bit halves of VALUE's IEEE-754 binary64 pattern (the
    higher-numbered register holds the more-significant half -- SC58x/
    2158x PRM p.3-35 pair-alias convention)."""
    bits = struct.unpack(">Q", struct.pack(">d", value))[0]
    return (bits >> 32) & 0xFFFFFFFF, bits & 0xFFFFFFFF


def full_compute(cu, opcode, rn, rx, ry):
    """A full-compute field dict for _compute(f, short=False, ...)."""
    field = (cu << 20) | (opcode << 12) | (rn << 8) | (rx << 4) | ry
    return {"compute[22:16]": field >> 16, "compute[15:0]": field & 0xFFFF}


class ComputeHelperMixin:
    def astatx_after(self, fields, values, old_astatx, special=None):
        rn, value, operation, update = T._compute(fields, False, values, special)
        return rn, value, operation, update(old_astatx)


# --- 64-bit ALU (cu=0, opcodes 0x11-0x1f) -----------------------------------


class DoubleAluTests(ComputeHelperMixin, unittest.TestCase):
    def test_double_add(self):
        # F1:0 = 1.5, F3:2 = 2.25 -> F5:4 = 3.75 (PRM p.20-22).
        hi_a, lo_a = f64(1.5)
        hi_b, lo_b = f64(2.25)
        values = {
            1: T.Const(hi_a),
            0: T.Const(lo_a),
            3: T.Const(hi_b),
            2: T.Const(lo_b),
        }
        rn, value, op, astatx = self.astatx_after(
            full_compute(0, 0x11, 4, 0, 2), values, T.Unknown("start")
        )
        self.assertEqual(rn, (5, 4))
        hi_expect, lo_expect = f64(3.75)
        self.assertEqual(value, (T.Const(hi_expect), T.Const(lo_expect)))
        self.assertEqual(op, "double-add")
        self.assertEqual(T._astatx_known_bit(astatx, T.AN_BIT), False)
        self.assertEqual(T._astatx_known_bit(astatx, T.AZ_BIT), False)
        self.assertEqual(T._astatx_known_bit(astatx, T.AV_BIT), False)

    def test_double_subtract_negative_result_sets_an(self):
        # F1:0 = 1.5, F3:2 = 5.0 -> F5:4 = -3.5 (PRM p.20-23).
        hi_a, lo_a = f64(1.5)
        hi_b, lo_b = f64(5.0)
        values = {
            1: T.Const(hi_a),
            0: T.Const(lo_a),
            3: T.Const(hi_b),
            2: T.Const(lo_b),
        }
        rn, value, op, astatx = self.astatx_after(
            full_compute(0, 0x12, 4, 0, 2), values, T.Unknown("start")
        )
        hi_expect, lo_expect = f64(-3.5)
        self.assertEqual(value, (T.Const(hi_expect), T.Const(lo_expect)))
        self.assertEqual(op, "double-subtract")
        self.assertEqual(T._astatx_known_bit(astatx, T.AN_BIT), True)

    def test_double_add_nan_input_returns_all_ones_and_sets_ai(self):
        nan_hi, nan_lo = 0x7FF80000, 0x0  # quiet NaN
        hi_b, lo_b = f64(1.0)
        values = {
            1: T.Const(nan_hi),
            0: T.Const(nan_lo),
            3: T.Const(hi_b),
            2: T.Const(lo_b),
        }
        _rn, value, _op, astatx = self.astatx_after(
            full_compute(0, 0x11, 4, 0, 2), values, T.Unknown("start")
        )
        self.assertEqual(value, (T.Const(0xFFFFFFFF), T.Const(0xFFFFFFFF)))
        self.assertEqual(T._astatx_known_bit(astatx, T.AI_BIT), True)

    def test_double_compare_equal_sets_az(self):
        hi_a, lo_a = f64(2.5)
        values = {
            1: T.Const(hi_a),
            0: T.Const(lo_a),
            3: T.Const(hi_a),
            2: T.Const(lo_a),
        }
        _rn, value, op, astatx = self.astatx_after(
            full_compute(0, 0x13, 0, 0, 2), values, T.Unknown("start")
        )
        self.assertEqual(op, "double-compare")
        self.assertEqual(value.value & 0x1, 0x1)  # AZ-source bit0 set
        self.assertEqual(T._astatx_known_bit(astatx, T.AZ_BIT), True)
        self.assertEqual(T._astatx_known_bit(astatx, T.AN_BIT), False)

    def test_double_compare_less_than_sets_an(self):
        hi_a, lo_a = f64(1.0)
        hi_b, lo_b = f64(2.0)
        values = {
            1: T.Const(hi_a),
            0: T.Const(lo_a),
            3: T.Const(hi_b),
            2: T.Const(lo_b),
        }
        _rn, value, _op, astatx = self.astatx_after(
            full_compute(0, 0x13, 0, 0, 2), values, T.Unknown("start")
        )
        self.assertEqual(value.value & 0x4, 0x4)  # AN-source bit2 set
        self.assertEqual(T._astatx_known_bit(astatx, T.AN_BIT), True)

    def test_double_negate(self):
        hi_a, lo_a = f64(1.5)
        values = {1: T.Const(hi_a), 0: T.Const(lo_a)}
        rn, value, op, astatx = self.astatx_after(
            full_compute(0, 0x14, 4, 0, 0), values, T.Unknown("start")
        )
        self.assertEqual(rn, (5, 4))
        hi_expect, lo_expect = f64(-1.5)
        self.assertEqual(value, (T.Const(hi_expect), T.Const(lo_expect)))
        self.assertEqual(op, "double-negate")
        self.assertEqual(T._astatx_known_bit(astatx, T.AN_BIT), True)

    def test_double_abs(self):
        hi_a, lo_a = f64(-1.5)
        values = {1: T.Const(hi_a), 0: T.Const(lo_a)}
        rn, value, op, astatx = self.astatx_after(
            full_compute(0, 0x15, 4, 0, 0), values, T.Unknown("start")
        )
        hi_expect, lo_expect = f64(1.5)
        self.assertEqual(value, (T.Const(hi_expect), T.Const(lo_expect)))
        self.assertEqual(op, "double-abs")
        # AN is architecturally fixed 0 for abs (PRM p.20-25).
        self.assertEqual(T._astatx_known_bit(astatx, T.AN_BIT), False)

    def test_double_pass(self):
        hi_a, lo_a = f64(-1.5)
        values = {1: T.Const(hi_a), 0: T.Const(lo_a)}
        rn, value, op, _astatx = self.astatx_after(
            full_compute(0, 0x16, 4, 0, 0), values, T.Unknown("start")
        )
        self.assertEqual(value, (T.Const(hi_a), T.Const(lo_a)))
        self.assertEqual(op, "double-pass")

    def test_double_fix_rounds_to_nearest(self):
        mode1 = T.UREG_CODES["MODE1"]
        hi_a, lo_a = f64(2.5)
        values = {1: T.Const(hi_a), 0: T.Const(lo_a), mode1: T.Const(0)}
        _rn, value, op, astatx = self.astatx_after(
            full_compute(0, 0x17, 9, 0, 0), values, T.Unknown("start")
        )
        self.assertEqual(op, "double-fix")
        self.assertEqual(value, T.Const(2))  # round-to-nearest-even
        self.assertEqual(T._astatx_known_bit(astatx, T.AV_BIT), False)

    def test_double_trunc_toward_zero(self):
        mode1 = T.UREG_CODES["MODE1"]
        hi_a, lo_a = f64(-2.9)
        values = {1: T.Const(hi_a), 0: T.Const(lo_a), mode1: T.Const(0)}
        _rn, value, op, _astatx = self.astatx_after(
            full_compute(0, 0x19, 9, 0, 0), values, T.Unknown("start")
        )
        self.assertEqual(op, "double-trunc")
        self.assertEqual(value, T.Const((-2) & 0xFFFFFFFF))

    def test_double_float_from_int(self):
        values = {2: T.Const(7)}
        rn, value, op, astatx = self.astatx_after(
            full_compute(0, 0x1B, 4, 2, 0), values, T.Unknown("start")
        )
        self.assertEqual(rn, (5, 4))
        hi_expect, lo_expect = f64(7.0)
        self.assertEqual(value, (T.Const(hi_expect), T.Const(lo_expect)))
        self.assertEqual(op, "double-float")
        self.assertEqual(T._astatx_known_bit(astatx, T.AI_BIT), False)
        self.assertEqual(T._astatx_known_bit(astatx, T.AV_BIT), False)

    def test_double_scalb(self):
        # F1:0 = 1.5, scale by 3 (Rx) -> 1.5 * 2**3 = 12.0 (PRM p.20-27).
        hi_a, lo_a = f64(1.5)
        values = {1: T.Const(hi_a), 0: T.Const(lo_a), 2: T.Const(3)}
        rn, value, op, _astatx = self.astatx_after(
            full_compute(0, 0x1F, 4, 0, 2), values, T.Unknown("start")
        )
        self.assertEqual(rn, (5, 4))
        hi_expect, lo_expect = f64(12.0)
        self.assertEqual(value, (T.Const(hi_expect), T.Const(lo_expect)))
        self.assertEqual(op, "double-scalb")

    def test_cvt_single_to_double(self):
        single_bits = struct.unpack("<I", struct.pack("<f", 1.25))[0]
        values = {2: T.Const(single_bits)}
        rn, value, op, astatx = self.astatx_after(
            full_compute(0, 0x1D, 4, 2, 0), values, T.Unknown("start")
        )
        self.assertEqual(rn, (5, 4))
        hi_expect, lo_expect = f64(1.25)
        self.assertEqual(value, (T.Const(hi_expect), T.Const(lo_expect)))
        self.assertEqual(op, "float32-to-double")
        self.assertEqual(T._astatx_known_bit(astatx, T.AV_BIT), False)

    def test_cvt_double_to_single(self):
        hi_a, lo_a = f64(1.25)
        values = {1: T.Const(hi_a), 0: T.Const(lo_a)}
        _rn, value, op, astatx = self.astatx_after(
            full_compute(0, 0x1E, 4, 0, 0), values, T.Unknown("start")
        )
        expect = struct.unpack("<I", struct.pack("<f", 1.25))[0]
        self.assertEqual(value, T.Const(expect))
        self.assertEqual(op, "double-to-float32")
        self.assertEqual(T._astatx_known_bit(astatx, T.AV_BIT), False)


# --- 64-bit multiplier (cu=1, opcodes 0x31-0x33) ----------------------------


class DoubleMultTests(ComputeHelperMixin, unittest.TestCase):
    def test_double_multiply(self):
        # F1:0 = 1.5, F3:2 = 2.0 -> F5:4 = 3.0 (PRM p.23-2).
        hi_a, lo_a = f64(1.5)
        hi_b, lo_b = f64(2.0)
        values = {
            1: T.Const(hi_a),
            0: T.Const(lo_a),
            3: T.Const(hi_b),
            2: T.Const(lo_b),
        }
        rn, value, op, astatx = self.astatx_after(
            full_compute(1, 0x31, 4, 0, 2), values, T.Unknown("start")
        )
        self.assertEqual(rn, (5, 4))
        hi_expect, lo_expect = f64(3.0)
        self.assertEqual(value, (T.Const(hi_expect), T.Const(lo_expect)))
        self.assertEqual(op, "double-multiply")
        self.assertEqual(T._astatx_known_bit(astatx, T.MN_BIT), False)
        self.assertEqual(T._astatx_known_bit(astatx, T.MI_BIT), False)

    def test_double_multiply_nan_sets_mi(self):
        nan_hi, nan_lo = 0x7FF80000, 0x0
        hi_b, lo_b = f64(1.0)
        values = {
            1: T.Const(nan_hi),
            0: T.Const(nan_lo),
            3: T.Const(hi_b),
            2: T.Const(lo_b),
        }
        _rn, value, _op, astatx = self.astatx_after(
            full_compute(1, 0x31, 4, 0, 2), values, T.Unknown("start")
        )
        self.assertEqual(value, (T.Const(0xFFFFFFFF), T.Const(0xFFFFFFFF)))
        self.assertEqual(T._astatx_known_bit(astatx, T.MI_BIT), True)

    def test_double_multiply_single(self):
        # F1:0 = 2.5 (double), FY(=F2, plain single) = 2.0 -> 5.0 (PRM p.23-3).
        hi_a, lo_a = f64(2.5)
        fy_bits = struct.unpack("<I", struct.pack("<f", 2.0))[0]
        values = {1: T.Const(hi_a), 0: T.Const(lo_a), 2: T.Const(fy_bits)}
        rn, value, op, _astatx = self.astatx_after(
            full_compute(1, 0x32, 4, 0, 2), values, T.Unknown("start")
        )
        self.assertEqual(rn, (5, 4))
        hi_expect, lo_expect = f64(5.0)
        self.assertEqual(value, (T.Const(hi_expect), T.Const(lo_expect)))
        self.assertEqual(op, "double-multiply-single")

    def test_float_widening_multiply(self):
        # FX=1.5, FY=2.0 (both plain single) -> 3.0 double (PRM p.23-4).
        fx_bits = struct.unpack("<I", struct.pack("<f", 1.5))[0]
        fy_bits = struct.unpack("<I", struct.pack("<f", 2.0))[0]
        values = {1: T.Const(fx_bits), 2: T.Const(fy_bits)}
        rn, value, op, astatx = self.astatx_after(
            full_compute(1, 0x33, 4, 1, 2), values, T.Unknown("start")
        )
        self.assertEqual(rn, (5, 4))
        hi_expect, lo_expect = f64(3.0)
        self.assertEqual(value, (T.Const(hi_expect), T.Const(lo_expect)))
        self.assertEqual(op, "float-widening-multiply")
        # A float32*float32 product is always exact in a double: MV/MU
        # cannot fire.
        self.assertEqual(T._astatx_known_bit(astatx, T.MV_BIT), False)
        self.assertEqual(T._astatx_known_bit(astatx, T.MU_BIT), False)


# --- cu=3 (reserved, still undocumented) ------------------------------------


class ReservedCu3Tests(ComputeHelperMixin, unittest.TestCase):
    """0xd6 and 0xe0 are the only two cu=3 opcodes this project's static
    census (tools/sharc.py, lane F1) finds in real, in-function code across
    dt2-1.16/1.15C/dn2-1.11/1.10E; neither is explained by the SC58x/2158x
    PRM's 64-bit floating-point chapters (see compute.py's CU3_OPS
    docstring). This only checks that both still decode -- as Unknown, not
    a raise -- matching the pre-existing 0xd6 behaviour lane F1 extended to
    0xe0 (STATE.md: the harness previously stopped hard at 0xe0)."""

    def test_0xe0_does_not_raise(self):
        values = {}
        _rn, value, op, _astatx = self.astatx_after(
            full_compute(3, 0xE0, 0, 0, 0), values, T.Unknown("start")
        )
        self.assertEqual(op, "compute-reserved-cu3")
        self.assertIsInstance(value, T.Unknown)
        self.assertIn("0xe0", value.reason)

    def test_0xd6_does_not_raise(self):
        values = {}
        _rn, value, op, _astatx = self.astatx_after(
            full_compute(3, 0xD6, 0, 0, 0), values, T.Unknown("start")
        )
        self.assertEqual(op, "compute-reserved-cu3")
        self.assertIsInstance(value, T.Unknown)
        self.assertIn("0xd6", value.reason)

    def test_other_cu3_opcode_still_raises(self):
        with self.assertRaises(ValueError):
            T._compute(full_compute(3, 0x01, 0, 0, 0), False, {})


if __name__ == "__main__":
    unittest.main()
