"""Tests for tools/sharc_capture_run.py's pure helpers: argument parsing and
the constants run_natural_track_refresh() depends on. No firmware or
snapshot needed -- see tests/test_sharc_replay.py's own docstring for the
project convention this follows (pure helpers always run; anything that
needs a real Machine/snapshot is exercised by hand, per the module's own
docstring).
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools"))
import sharc_capture_run as scr  # noqa: E402


class ParseTrackTypePokeTest(unittest.TestCase):
    def test_decimal(self):
        self.assertEqual(scr.parse_track_type_poke("3:2"), (3, 2))

    def test_hex(self):
        self.assertEqual(scr.parse_track_type_poke("0xf:0x5"), (15, 5))

    def test_missing_colon_rejected(self):
        with self.assertRaises(ValueError):
            scr.parse_track_type_poke("32")

    def test_track_out_of_range_rejected(self):
        with self.assertRaises(ValueError):
            scr.parse_track_type_poke("16:2")
        with self.assertRaises(ValueError):
            scr.parse_track_type_poke("-1:2")


class ParseMemRangeTest(unittest.TestCase):
    def test_hex_range(self):
        self.assertEqual(
            scr.parse_mem_range("0x8c000000:0x8c000010"), (0x8C000000, 0x8C000010)
        )

    def test_missing_colon_rejected(self):
        with self.assertRaises(ValueError):
            scr.parse_mem_range("0x8c000000")

    def test_hi_must_exceed_lo_rejected(self):
        with self.assertRaises(ValueError):
            scr.parse_mem_range("0x10:0x10")
        with self.assertRaises(ValueError):
            scr.parse_mem_range("0x10:0x5")


class ParseArgsPreInstrsTest(unittest.TestCase):
    """Lane A3: --pre-instrs lets the real per-track kit-load-and-refresh
    (run_natural_track_refresh()'s own docstring) fire before the timed
    capture phase, instead of every capture seeing the SRAM row's
    zero-initialized reset value. Default 0 must keep every existing
    capture's exact behaviour (an opt-in flag, per this lane's own scope)."""

    def _parse(self, extra=()):
        return scr.parse_args(
            ["snap.snap", "--out", "out.dt2cap", "--kind", "idle", *extra]
        )

    def test_default_is_zero(self):
        self.assertEqual(self._parse().pre_instrs, 0)

    def test_decimal_value(self):
        self.assertEqual(
            self._parse(["--pre-instrs", "15000000"]).pre_instrs, 15_000_000
        )

    def test_hex_value(self):
        self.assertEqual(self._parse(["--pre-instrs", "0xE00000"]).pre_instrs, 0xE00000)

    def test_combines_with_poke_track_type(self):
        args = self._parse(["--pre-instrs", "15000000", "--poke-track-type", "3:2"])
        self.assertEqual(args.pre_instrs, 15_000_000)
        self.assertEqual(args.poke_track_type, ["3:2"])


class NaturalRefreshConstantsTest(unittest.TestCase):
    """KIT_LOAD_FN/ROW_REFRESH_FN are docs/findings/04-coldfire-dsp-link.md
    addresses (FUN_4002d9c4, FUN_4002d438), pinned so a future edit that
    moves either constant is caught here rather than silently missing the
    hook at runtime."""

    def test_kit_load_fn_address(self):
        self.assertEqual(scr.KIT_LOAD_FN, 0x4002D9C4)

    def test_row_refresh_fn_address(self):
        self.assertEqual(scr.ROW_REFRESH_FN, 0x4002D438)

    def test_run_natural_track_refresh_is_callable(self):
        self.assertTrue(callable(scr.run_natural_track_refresh))


if __name__ == "__main__":
    unittest.main()
