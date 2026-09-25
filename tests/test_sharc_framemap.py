"""Tests for tools/sharc_framemap.py.

Pure-Python pieces (marker_value, _unalias, _region_label, _apply_fingerprint,
_load_frame_bytes) need neither the real firmware nor its program database.
The end-to-end run() test does (out/sharcdb, out/sections -- see the
worktree's symlinks) and is skipped, not marked slow, when they are not
available, matching tests/test_sharc_inputs.py; run() itself is marked slow
(run_init() alone is ~1.24M instructions).
"""

import os
import pathlib
import sys
import unittest
from importlib import import_module

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools"))
fm = import_module("sharc_framemap")
L = import_module("sharcldr")

DT2_116_DB = pathlib.Path("out/sharcdb/dt2-1.16.sqlite")
DT2_116_BLOB = pathlib.Path("out/sections/dt2-1.16/section_7_BLOB.bin")
IDLE_CAPTURE = pathlib.Path("out/captures/dt2-1.16-idle.dt2cap")


class MarkerValueTest(unittest.TestCase):
    def test_distinct_across_every_field_and_track(self):
        values = set()
        for field_index in range(len(fm.FINGERPRINT_FIELDS)):
            for track in range(fm.TRACK_COUNT):
                v = fm.marker_value(field_index, track)
                self.assertNotIn(v, values)
                values.add(v)

    def test_always_small_and_positive(self):
        # No width's sign/zero extension may change a marker -- see the
        # module docstring.
        for field_index in range(len(fm.FINGERPRINT_FIELDS)):
            for track in range(fm.TRACK_COUNT):
                v = fm.marker_value(field_index, track)
                self.assertGreaterEqual(v, 0)
                self.assertLess(v, 0x8000)


class UnaliasTest(unittest.TestCase):
    def test_raw_address_unchanged(self):
        self.assertEqual(fm._unalias(0x2558DC), 0x2558DC)

    def test_aliased_address_stripped(self):
        self.assertEqual(fm._unalias(L.SW_ALIAS_BASE + 0x2558DC), 0x2558DC)

    def test_boundary(self):
        self.assertEqual(fm._unalias(L.SW_ALIAS_BASE), 0)
        self.assertEqual(fm._unalias(L.SW_ALIAS_BASE - 1), L.SW_ALIAS_BASE - 1)


class RegionLabelTest(unittest.TestCase):
    def test_ring_buffer(self):
        self.assertTrue(fm._region_label(0x261CC8).startswith("ring:"))

    def test_gain_table(self):
        self.assertEqual(fm._region_label(0x255FB6), "gain/mixer table +0x0")
        self.assertEqual(fm._region_label(0x2560B8), "gain/mixer table +0x102")

    def test_voice_record(self):
        base = fm.VOICE_RECORDS_BASE
        self.assertEqual(fm._region_label(base), "voice record 0 +0x0")
        self.assertEqual(
            fm._region_label(base + fm.VOICE_RECORD_STRIDE + 4), "voice record 1 +0x4"
        )

    def test_frame_itself(self):
        self.assertEqual(fm._region_label(fm.FRAME_BASE + 0x94), "frame itself +0x94")

    def test_bare_workspace(self):
        label = fm._region_label(0x241000)
        self.assertTrue(label.startswith("workspace "))


class ApplyFingerprintTest(unittest.TestCase):
    def test_fingerprinted_offsets_get_distinct_be_markers(self):
        frame = bytearray(fm.FRAME_LEN)
        markers = fm._apply_fingerprint(frame)
        field_index = fm.FINGERPRINT_FIELDS.index(0x94)
        track = 3
        offset = 0x94 + 2 * track
        value = (frame[offset] << 8) | frame[offset + 1]
        self.assertEqual(value, fm.marker_value(field_index, track))
        self.assertEqual(markers[value], (0x94, track))

    def test_untouched_offsets_stay_real_data(self):
        frame = bytearray(fm.FRAME_LEN)
        frame[0x0] = 0xAB  # not one of FINGERPRINT_FIELDS
        fm._apply_fingerprint(frame)
        self.assertEqual(frame[0x0], 0xAB)

    def test_the_0x60_stride_parameter_page_is_never_touched(self):
        # docs/findings/06's "FUN_1c24e9 reads the SRC page" -- this range
        # can carry sample-length/loop bounds a synthetic value could turn
        # into a runaway loop, so fingerprinting must never write here.
        frame = bytearray(fm.FRAME_LEN)
        fm._apply_fingerprint(frame)
        for offset in range(0xDA, 0xDA + 16 * 0x60):
            self.assertEqual(frame[offset], 0, "byte 0x%x was touched" % offset)


class LoadFrameBytesTest(unittest.TestCase):
    def test_no_capture_is_all_zero(self):
        frame = fm._load_frame_bytes(None, 0)
        self.assertEqual(len(frame), fm.FRAME_LEN)
        self.assertEqual(frame, bytearray(fm.FRAME_LEN))

    def test_short_capture_frame_is_padded(self):
        # emu.sharc_capture.load() needs a real capture file; build a tiny
        # one directly instead of depending on out/captures/.
        import tempfile

        from emu.sharc_capture import CaptureWriter

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "short.dt2cap")
            with CaptureWriter(path, frame_bytes=10, kind="idle") as w:
                w.write_dspi2(0, b"\x01\x02\x03", b"")
            frame = fm._load_frame_bytes(path, 0)
        self.assertEqual(len(frame), fm.FRAME_LEN)
        self.assertEqual(frame[:3], b"\x01\x02\x03")
        self.assertEqual(frame[3], 0)

    def test_bad_frame_index_raises(self):
        import tempfile

        from emu.sharc_capture import CaptureWriter

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "one.dt2cap")
            with CaptureWriter(path, frame_bytes=10, kind="idle") as w:
                w.write_dspi2(0, b"\x00", b"")
            with self.assertRaises(ValueError):
                fm._load_frame_bytes(path, 5)


@pytest.mark.slow
@unittest.skipUnless(DT2_116_BLOB.exists(), "DT2 1.16 firmware bytes are not available")
class RunEndToEndTest(unittest.TestCase):
    """run() against the real image: run_init() alone is ~1.24M
    instructions, and a Watchpoint-logged render_frame call is slower
    still -- see tests/test_sharc_inputs.py's own DynamicViewTest."""

    @classmethod
    def setUpClass(cls):
        capture = str(IDLE_CAPTURE) if IDLE_CAPTURE.exists() else None
        cls.result = fm.run("dt2-1.16", capture=capture, fingerprint=True)

    def test_no_error(self):
        self.assertNotIn("error", self.result)

    def test_frame_mapping_confirmed_by_execution(self):
        # docs/findings/06's "The ColdFire frame is mapped into SHARC DM at
        # 0x2558dc": a real render_frame call must actually read a
        # substantial part of the poked frame before its first stop, not
        # just the eleven statically-known scalar bases.
        self.assertGreater(self.result["n_offsets_read"], 100)

    def test_known_machine_type_offset_is_read(self):
        offsets = {int(f["frame_offset"], 16) for f in self.result["fields"]}
        # docs/findings/04: machine type at 0x94 + 2*track, for every track.
        for track in range(fm.TRACK_COUNT):
            self.assertIn(0x94 + 2 * track, offsets)

    def test_reader_pcs_are_all_in_shal_dm_program_space(self):
        for f in self.result["fields"]:
            for pc in f["reader_pcs"]:
                self.assertTrue(pc.startswith("0x"))
                self.assertLess(int(pc, 16), L.SW_ALIAS_BASE)


if __name__ == "__main__":
    unittest.main()
