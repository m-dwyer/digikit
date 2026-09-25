"""Tests for the shifter's bit-FIFO model (tools/sharc_core/compute_shift.py's
``_bff_words``/``_bff_extract``/``_bff_deposit`` and the ShiftImm opcode
0x14/0x19 BITEXT handler built on them).

Numeric cases are hand-derived from the PRM/PGR's documented pseudocode
(PRM p.3-18, out/refs/sharc-plus-prm/all.txt:3507-3511; PGR p.11-86/11-87/
11-90/11-91, pgr.txt:23287-23481) rather than lifted from the manuals'
own worked examples, which label bits with letters ("qwertyui...") instead
of giving actual numbers. Where a test's setup mirrors a manual listing's
*procedure* (out/refs/sharc-plus-prm/all.txt:3495-3521's Example of Header
Extraction: BFFWRP=0; BITDEP R10 by 32; R6 = BITEXT(6)) this is noted
inline, with concrete numbers standing in for the manual's letters and the
expected outputs verified independently of this file's own code (by hand,
in the docstring/comments below -- not by calling _bff_extract/_bff_deposit
to generate the "expected" value, which would test nothing).

Kept separate from tests/test_sharc_trace.py and tests/test_sharc_trace_forms.py
(which cover the ShiftImm opcode dispatch table itself) so this lane's
edits do not collide with other agents' concurrent edits to those files.
"""

import os
import sys
import unittest
from importlib import import_module

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools"))
T = import_module("sharc_trace")
compute_shift = import_module("sharc_core.compute_shift")


def shiftimm_fields(opcode, data8, rn, rx, dataex=0):
    """A ShiftImm field dict for T._shift_immediate (matches the helper of
    the same name in tests/test_sharc_trace.py)."""
    field = (opcode << 16) | (data8 << 8) | (rn << 4) | rx
    return {
        "shiftimm[22:16]": field >> 16,
        "shiftimm[15:0]": field & 0xFFFF,
        "dataex[3:0]": dataex,
    }


def bitext_fields(opcode, bitlen12, rn, rx=0):
    """A ShiftImm field dict encoding BITLEN12 the way opcode 0x14/0x19
    split it: dataex[3:0] holds bits [11:8], data8 holds bits [7:0]."""
    return shiftimm_fields(opcode, bitlen12 & 0xFF, rn, rx, (bitlen12 >> 8) & 0xF)


# ---------------------------------------------------------------------------
# _bff_extract: the 64-bit-int model shared by BITEXT and (unreached)
# BITDEP.
# ---------------------------------------------------------------------------


class BffExtractTest(unittest.TestCase):
    def test_extract_within_top_word(self):
        # hi = 0xF0000000 (top nibble set), lo = 0: the documented pseudo-
        # code (PGR p.11-91) is "Rn = FEXT BFF[63:32] BY <32-bitlen>:
        # <bitlen>" -- by hand, extracting the top 4 bits of 0xF0000000
        # gives the top nibble itself, 0xF, right-justified.
        extracted, new_hi, new_lo = compute_shift._bff_extract(
            T.Const(0xF0000000), T.Const(0), 4
        )
        self.assertEqual(extracted, T.Const(0xF))
        # Step 2 (BFF <<= bitlen) shifts the whole 64-bit register left by
        # 4: the extracted nibble falls off the top, and no lower bits
        # exist to replace it, so both halves become 0.
        self.assertEqual(new_hi, T.Const(0))
        self.assertEqual(new_lo, T.Const(0))

    def test_extract_zero_bits_is_a_no_op(self):
        # BITLEN=0 is always well-defined (PGR's own pseudocode reduces to
        # a no-op FEXT/shift), even when the FIFO content itself is not --
        # this must not manufacture an Unknown out of a trivially-known 0.
        extracted, new_hi, new_lo = compute_shift._bff_extract(
            T.Unknown("x"), T.Unknown("y"), 0
        )
        self.assertEqual(extracted, T.Const(0))
        self.assertEqual(new_hi, T.Unknown("x"))
        self.assertEqual(new_lo, T.Unknown("y"))

    def test_extract_unknown_fifo_is_unknown(self):
        extracted, new_hi, new_lo = compute_shift._bff_extract(
            T.Unknown("uninitialized bit FIFO"), T.Const(0), 6
        )
        self.assertIsInstance(extracted, T.Unknown)
        self.assertIsInstance(new_hi, T.Unknown)
        self.assertIsInstance(new_lo, T.Unknown)

    def test_extract_carries_low_word_bits_into_high_word(self):
        # hi = 0x00000001 (a single valid bit at the very bottom of the
        # top word -- i.e. the word is otherwise full), lo = 0x80000000
        # (the low word's own top bit, next in line once the top word
        # empties). This is PRM/PGR's own "BFF = BFF << bitlen" step
        # crossing the hi/lo boundary (all.txt:3507's "64-bit register",
        # not two independent 32-bit ones).
        #
        # By hand: combined = hi:lo = 0x0000000180000000 (64 bits). The
        # top 1 bit (bit 63) is 0, so BITEXT(1) extracts 0. Shifting the
        # 64-bit register left by 1 moves lo's bit 31 (the only set bit in
        # lo) up into hi's bit 0, and hi's own bit 0 (already 1) up into
        # bit 1 -- new hi = 0b11 = 3, new lo = 0.
        extracted, new_hi, new_lo = compute_shift._bff_extract(
            T.Const(1), T.Const(0x80000000), 1
        )
        self.assertEqual(extracted, T.Const(0))
        self.assertEqual(new_hi, T.Const(3))
        self.assertEqual(new_lo, T.Const(0))

    def test_extract_matches_header_extraction_listing(self):
        # out/refs/sharc-plus-prm/all.txt:3495-3521, "Example of Header
        # Extraction": BFFWRP = 0x0; BITDEP R10 by 32; R6 = BITEXT(6). The
        # manual labels R10's bits with letters; substitute a concrete
        # value, R10 = 0x12345678, and hand-verify BITEXT(6)'s result two
        # independent ways.
        #
        # After "BITDEP R10 by 32" into an empty FIFO, hi = R10 = 0x12345678
        # exactly (see BffDepositTest.test_deposit_into_empty_fifo_matches_
        # listing below) and lo = 0.
        #
        # Hand check 1 (bit string): 0x12345678 = 0001 0010 0011 0100 0101
        # 0110 0111 1000. Its top 6 bits are 000100 = 4, so BITEXT(6) must
        # extract 4.
        #
        # Hand check 2 (arithmetic, independent of check 1): the shifted
        # high word is (hi << 6) & 0xFFFFFFFF = (0x12345678 * 64) mod
        # 2**32 = 19546873344 mod 4294967296 = 2367004160 = 0x8D159E00.
        extracted, new_hi, new_lo = compute_shift._bff_extract(
            T.Const(0x12345678), T.Const(0), 6
        )
        self.assertEqual(extracted, T.Const(4))
        self.assertEqual(new_hi, T.Const(0x8D159E00))
        self.assertEqual(new_lo, T.Const(0))


# ---------------------------------------------------------------------------
# _bff_deposit: BITDEP's own pseudocode, unreachable through decode in this
# ISA (see the function's docstring) but exercised directly here.
# ---------------------------------------------------------------------------


class BffDepositTest(unittest.TestCase):
    def test_deposit_into_empty_fifo_matches_listing(self):
        # out/refs/sharc-plus-prm/all.txt:3495-3521: "BFFWRP = 0x0;
        # ... BITDEP R10 by 32". Depositing a full 32-bit word into an
        # empty FIFO (wrp=0) must land it exactly in the high word: PGR's
        # pseudocode position is 64-(wrp+bitlen) = 64-(0+32) = 32, i.e. the
        # deposited field starts exactly at the hi/lo boundary.
        new_hi, new_lo = compute_shift._bff_deposit(
            T.Const(0), T.Const(0), 0, T.Const(0x12345678), 32
        )
        self.assertEqual(new_hi, T.Const(0x12345678))
        self.assertEqual(new_lo, T.Const(0))

    def test_deposit_packs_below_existing_content(self):
        # A non-empty FIFO (wrp=4, top nibble of hi already holds 0b1010)
        # depositing 4 more bits (0b0110) packs them immediately below the
        # existing content: position = 64-(4+4) = 56, i.e. bits [59:56] of
        # the 64-bit register, which is bits [27:24] of hi.
        hi = 0b1010 << 28  # existing 4 valid bits, MSB-justified in hi
        new_hi, new_lo = compute_shift._bff_deposit(
            T.Const(hi), T.Const(0), 4, T.Const(0b0110), 4
        )
        self.assertEqual(new_hi, T.Const((0b1010 << 28) | (0b0110 << 24)))
        self.assertEqual(new_lo, T.Const(0))

    def test_deposit_overflow_is_undefined(self):
        # PGR p.11-87: "Attempts to append more bits than the bit FIFO has
        # room for results in an undefined bit FIFO and write pointer."
        new_hi, new_lo = compute_shift._bff_deposit(
            T.Const(0), T.Const(0), 60, T.Const(0xFF), 8
        )
        self.assertIsInstance(new_hi, T.Unknown)
        self.assertIsInstance(new_lo, T.Unknown)

    def test_deposit_zero_bits_is_a_no_op(self):
        new_hi, new_lo = compute_shift._bff_deposit(
            T.Unknown("x"), T.Unknown("y"), 5, T.Const(0xFF), 0
        )
        self.assertEqual(new_hi, T.Unknown("x"))
        self.assertEqual(new_lo, T.Unknown("y"))


# ---------------------------------------------------------------------------
# The wired-up ShiftImm opcode 0x14 (update) / 0x19 (NU) BITEXT handler.
# ---------------------------------------------------------------------------


class BitextOpcodeTest(unittest.TestCase):
    def test_bitext_update_extracts_and_advances_pointer(self):
        special = {
            "BFFWRP": T.Const(32),
            "BFF_HI": T.Const(0x12345678),
            "BFF_LO": T.Const(0),
        }
        rn, value, op, update = T._shift_immediate(
            bitext_fields(0x14, 6, rn=6), {}, special
        )
        self.assertEqual(op, "bit-extract")
        self.assertEqual(rn, (6, "BFFWRP", "BFF_HI", "BFF_LO"))
        self.assertEqual(
            value, (T.Const(4), T.Const(26), T.Const(0x8D159E00), T.Const(0))
        )
        astatx = update(T.Unknown("start"))
        self.assertEqual(T._astatx_known_bit(astatx, T.SS_BIT), False)
        self.assertEqual(T._astatx_known_bit(astatx, T.SV_BIT), False)
        self.assertEqual(T._astatx_known_bit(astatx, T.SZ_BIT), False)
        # Updated BFFWRP (26) < 32 -> SF clears.
        self.assertEqual(T._astatx_known_bit(astatx, T.SF_BIT), False)

    def test_bitext_nu_leaves_fifo_and_pointer_untouched(self):
        special = {
            "BFFWRP": T.Const(32),
            "BFF_HI": T.Const(0x12345678),
            "BFF_LO": T.Const(0),
        }
        rn, value, op, update = T._shift_immediate(
            bitext_fields(0x19, 6, rn=6), {}, special
        )
        self.assertEqual(op, "bit-extract-nu")
        self.assertEqual(rn, 6)
        self.assertEqual(value, T.Const(4))
        astatx = update(T.Unknown("start"))
        # SF reflects the *un-updated* pointer (32 >= 32 -> set), per PGR
        # p.11-91's "If NU modifier is used SF reflects the un-updated
        # Write pointer status" -- even though a real update would clear it
        # (26 < 32, as the update-variant test above shows).
        self.assertEqual(T._astatx_known_bit(astatx, T.SF_BIT), True)

    def test_bitext_over_32_undefines_rn_but_not_the_pointer(self):
        # PGR p.11-91's two error sentences have different scope: "A value
        # of more than 32 ... is prohibited and use of such a value sets
        # SV" says nothing about the pointer or FIFO (unlike the separate
        # "results in undefined pointer and bit FIFO" sentence for the
        # underflow case below) -- and step 1's FEXT genuinely cannot
        # return more than 32 bits from a single word, while step 2/3's
        # pointer/FIFO bookkeeping has no such limit. So RN is Unknown, but
        # BFFWRP still decrements mechanically: 64 - 40 = 24.
        special = {"BFFWRP": T.Const(64)}
        rn, value, op, update = T._shift_immediate(
            bitext_fields(0x14, 40, rn=6), {}, special
        )
        self.assertEqual(value[0], T.Unknown("bitext: undefined (bitlen 40 > 32)"))
        self.assertEqual(value[1], T.Const(24))
        astatx = update(T.Unknown("start"))
        self.assertEqual(T._astatx_known_bit(astatx, T.SV_BIT), True)
        self.assertEqual(T._astatx_known_bit(astatx, T.SF_BIT), False)  # 24 < 32

    def test_bitext_underflow_sets_sv_when_pointer_known(self):
        # PGR p.11-91: "Attempts to get more bits than those in the bit
        # FIFO results in undefined pointer and bit FIFO. SV is set in
        # that case" -- distinct from (and here, in isolation from) the
        # BITLEN12>32 case: bitlen=6 is legal in general, but exceeds a
        # BFFWRP of 4. Per the manual's own wording, only the *pointer and
        # FIFO* are declared undefined here -- RN is not mentioned, and the
        # FEXT-based extraction (step 1) is mechanically well defined
        # regardless of BFFWRP, so RN still comes out concrete: the top 6
        # bits of 0xF0000000 are 111100 = 0x3C = 60.
        special = {
            "BFFWRP": T.Const(4),
            "BFF_HI": T.Const(0xF0000000),
            "BFF_LO": T.Const(0),
        }
        rn, value, op, update = T._shift_immediate(
            bitext_fields(0x14, 6, rn=6), {}, special
        )
        self.assertEqual(value[0], T.Const(0x3C))
        self.assertIsInstance(value[1], T.Unknown)  # BFFWRP undefined
        self.assertIsInstance(value[2], T.Unknown)  # and the FIFO itself
        self.assertIsInstance(value[3], T.Unknown)
        astatx = update(T.Unknown("start"))
        self.assertEqual(T._astatx_known_bit(astatx, T.SV_BIT), True)

    def test_bitext_underflow_unknown_when_pointer_unknown(self):
        # No BFFWRP tracked at all (never written in this trace): SV must
        # stay unknown, not silently False -- this is the bug the old
        # "SV_BIT: bitlen12 > 32" formula had (it ignored this case
        # entirely). The extracted value is still computed mechanically,
        # since BITEXT's FEXT step never reads BFFWRP.
        special = {"BFF_HI": T.Const(0xF0000000), "BFF_LO": T.Const(0)}
        rn, value, op, update = T._shift_immediate(
            bitext_fields(0x14, 4, rn=6), {}, special
        )
        self.assertEqual(value[0], T.Const(0xF))
        astatx = update(T.Unknown("start"))
        self.assertIsNone(T._astatx_known_bit(astatx, T.SV_BIT))
        # BFFWRP was never known, so it stays Unknown ("uninitialized"),
        # not merely "undefined" -- these are different reasons but both
        # collapse to Unknown.
        self.assertIsInstance(value[1], T.Unknown)

    def test_bitext_absent_special_value_unknown_sv_now_honest(self):
        # The common case in dt2-1.16: no BITDEP ever runs and BFFWRP is
        # never written, so special is empty for this opcode. RN (and, on
        # the update variant, BFFWRP/the FIFO) stay Unknown, matching the
        # pre-existing default -- but SV changes from this file's old,
        # always-False formula (which only ever checked bitlen12 > 32) to
        # None here, since with no BFFWRP tracked this tracer genuinely
        # cannot know whether PGR p.11-91's "more bits than those in the
        # bit FIFO" condition holds. That correction is the point of this
        # test, not an implementation detail.
        rn, value, op, update = T._shift_immediate(
            bitext_fields(0x14, 6, rn=6), {}, None
        )
        self.assertIsInstance(value[0], T.Unknown)
        astatx = update(T.Unknown("start"))
        self.assertIsNone(T._astatx_known_bit(astatx, T.SV_BIT))
        self.assertIsNone(T._astatx_known_bit(astatx, T.SF_BIT))


if __name__ == "__main__":
    unittest.main()
