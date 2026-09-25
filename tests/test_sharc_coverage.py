"""tools/sharc_coverage.py's compare_decode_at(): tools/sharcdb.py's own
decode (the `insn` table) against tools/sharc_core.sequencer.decode_at() at
the same PC. See docs/findings/05-sharc-isa-and-decoding.md, "One decode
path"."""

import json
import os
import pathlib
import struct
import sys
import unittest

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"),
)

from test_sharc_disasm import encode  # noqa: E402
from test_sharcldr import block  # noqa: E402

import sharc  # noqa: E402
import sharc_coverage  # noqa: E402
import sharcldr  # noqa: E402

DT2_116_BLOB = pathlib.Path("out/sections/dt2-1.16/section_7_BLOB.bin")


class RawLeBytesTest(unittest.TestCase):
    """_raw_le_bytes() must invert the packing tools/sharc_isa.py's
    frame_of()/DecodedInstruction.raw use: 16-bit little-endian words, most
    significant short word first."""

    def test_single_word(self):
        self.assertEqual(sharc_coverage._raw_le_bytes(0xC000, 2), b"\x00\xc0")

    def test_three_words(self):
        # w0=0x0a3e, w1=0x0400, w2=0x0000 -- a real dt2-1.16 example (sw
        # 0x1c0701's insn.raw) that first exposed the naive
        # bytes.fromhex(raw_hex) comparison bug this function replaces.
        raw = 0x0A3E04000000
        self.assertEqual(
            sharc_coverage._raw_le_bytes(raw, 6), b"\x3e\x0a\x00\x04\x00\x00"
        )


class _FakeImage:
    """The minimal tools/sharc.py Image surface compare_decode_at() uses:
    .name, ._mem() and .sql()."""

    def __init__(self, name, mem, rows):
        self.name = name
        self._mem_obj = mem
        self._rows = rows

    def _mem(self):
        return self._mem_obj

    def sql(self, query, *args):
        return self._rows


def _image_at(sw, data, db_width, db_raw, db_form, db_fields):
    addr = sharcldr.sw_to_byte(sw)
    mem = sharcldr.LoadedMemory.from_stream(block(0, addr, len(data), payload=data))
    rows = [(sw, db_width, "%x" % db_raw, db_form, json.dumps(db_fields))]
    return _FakeImage("fake", mem, rows)


class CompareDecodeAtTest(unittest.TestCase):
    def test_agreeing_row_is_not_a_mismatch(self):
        # word0 alone would truncate 2b's own (wrongly) wider candidate
        # match; real code following it (as in a real image) is what lets
        # decode_confident() resolve the width at all -- see
        # tests/test_sharc_disasm.py's ConfidentDecodeTest.
        word0 = encode("2c", 0)
        filler = encode("17b", 0) * 10
        data = word0 + filler
        raw = struct.unpack("<H", word0)[0]
        img = _image_at(0x1000, data, 2, raw, "2c", {"compute[11:0]": 0})
        result = sharc_coverage.compare_decode_at(img)
        self.assertEqual((result["stale_bytes"], result["mismatches"]), (0, 0))

    def test_stale_bytes_row_is_skipped_not_counted_as_a_mismatch(self):
        # The database's raw bytes were never written to this image's
        # memory at all: decode_at can't agree or disagree with bytes it
        # cannot read, so this is not a decode mismatch (see
        # tools/sharc_coverage.compare_decode_at's docstring -- this is the
        # dt2-1.16 block-1 case, reproduced minimally).
        mem = sharcldr.LoadedMemory.from_stream(b"")
        rows = [(0x2000, 2, "c000", "2c", "{}")]
        img = _FakeImage("fake", mem, rows)
        result = sharc_coverage.compare_decode_at(img)
        self.assertEqual((result["stale_bytes"], result["mismatches"]), (1, 0))

    def test_form_mismatch_is_classified(self):
        data = encode("21c", 0)  # 16-bit, fully fixed, no fields
        raw = struct.unpack("<H", data)[0]
        img = _image_at(0x1000, data, 2, raw, "not-a-real-form", {})
        result = sharc_coverage.compare_decode_at(img)
        self.assertEqual(result["by_kind"], {"form": 1})
        self.assertEqual(result["by_key"][0]["decode_at_form"], "21c")

    def test_width_mismatch_is_classified(self):
        # word0 naively also matches Type2b (32 bits); with real code
        # following, decode_at correctly prefers the narrower Type2c (see
        # tests/test_sharc_disasm.py's ConfidentDecodeTest). Claim the
        # database recorded the wider (wrong) reading, with bytes that
        # really are what's in memory across those 4 bytes, so this is not
        # a stale_bytes row.
        word0 = encode("2c", 0)
        filler = encode("17b", 0) * 10
        data = word0 + filler
        w0, w1 = struct.unpack_from("<HH", data, 0)
        raw32 = (w0 << 16) | w1
        img = _image_at(0x1000, data, 4, raw32, "2b", {})
        result = sharc_coverage.compare_decode_at(img)
        self.assertEqual(result["by_kind"], {"width": 1})
        self.assertEqual(
            (
                result["by_key"][0]["decode_at_form"],
                result["by_key"][0]["decode_at_width"],
            ),
            ("2c", 2),
        )

    def test_fields_mismatch_is_classified(self):
        word0 = encode("2c", 5)
        filler = encode("17b", 0) * 10
        data = word0 + filler
        raw = struct.unpack("<H", word0)[0]
        img = _image_at(0x1000, data, 2, raw, "2c", {"compute[11:0]": 999})
        result = sharc_coverage.compare_decode_at(img)
        self.assertEqual(result["by_kind"], {"fields": 1})


@unittest.skipUnless(DT2_116_BLOB.exists(), "DT2 1.16 SHARC blob is not available")
class RealImageAgreementTest(unittest.TestCase):
    """The acceptance fact behind the "One decode path" fix: decode_at()
    agrees with the database's own successor-confidence decode at every
    aligned instruction of the real dt2-1.16 image (a few thousand rows are
    skipped as stale_bytes -- see compare_decode_at()'s docstring and
    docs/findings/05-sharc-isa-and-decoding.md; that is a loader-block
    lifecycle artifact, not a decode disagreement)."""

    def test_decode_at_matches_database_for_every_aligned_instruction(self):
        img = sharc.load("dt2-1.16")
        result = sharc_coverage.compare_decode_at(img)
        self.assertEqual(
            result["mismatches"],
            0,
            "decode_at disagreed with the database at: %s" % result["by_key"],
        )


if __name__ == "__main__":
    unittest.main()
