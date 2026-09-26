"""Tests for tools/sharc_memdiff.py.

Same two-group convention as tests/test_sharc_replay.py: the labelling/
coalescing helpers are pure and always run; memdiff() itself needs the real
DT2 1.16 SHARC+ image bytes (Elektron's copyright, never committed) and a
real capture file (out/captures/, also never committed), and is skipped
without either.
"""

import os
import pathlib
import sys
import unittest

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools"))
import sharc_memdiff as md  # noqa: E402

DT2_116_BLOB = pathlib.Path("out/sections/dt2-1.16/section_7_BLOB.bin")
IDLE_CAPTURE = pathlib.Path("out/captures/dt2-1.16-idle.dt2cap")


class CanonTest(unittest.TestCase):
    def test_low_address_is_unchanged(self):
        self.assertEqual(md._canon(0x2412CC), 0x2412CC)

    def test_aliased_address_is_folded_down(self):
        from sharcldr import SW_ALIAS_BASE

        self.assertEqual(md._canon(SW_ALIAS_BASE + 0x2412CC), 0x2412CC)

    def test_boundary_address_is_treated_as_aliased(self):
        from sharcldr import SW_ALIAS_BASE

        self.assertEqual(md._canon(SW_ALIAS_BASE), 0)


class RegionsTest(unittest.TestCase):
    """build_regions() needs a resolved symbol Profile (tools/sharc_symbols.py),
    which in turn needs the real firmware bytes -- but the region *shape* for
    the parts this lane's brief gives as literal addresses (special voices,
    the guard arrays) does not depend on any symbol resolution, so those are
    checked directly here without gating on firmware."""

    def test_special_voice_bases_match_brief(self):
        self.assertEqual(md.SPECIAL_VOICE_BASES, (0x252730, 0x252908))

    def test_guard_range_covers_brief_bytes_inclusive(self):
        self.assertEqual(md.GUARD_LO, 0x24F0D8)
        # The brief's own "0x24f0d8..0x24f137" names 0x24f137 as the last
        # byte, so the exclusive GUARD_HI must be one past it.
        self.assertEqual(md.GUARD_HI, 0x24F138)


class BucketTest(unittest.TestCase):
    def setUp(self):
        self.regions = [(0x100, 0x200, "voice0"), (0x200, 0x210, "guard")]

    def test_finds_containing_region(self):
        self.assertEqual(md._bucket(0x150, self.regions), ("voice0", 0x100))
        self.assertEqual(md._bucket(0x205, self.regions), ("guard", 0x200))

    def test_outside_every_region_is_unlabeled(self):
        self.assertEqual(md._bucket(0x50, self.regions), (None, None))
        self.assertEqual(md._bucket(0x210, self.regions), (None, None))


class CoalesceTest(unittest.TestCase):
    def setUp(self):
        self.regions = [(0x100, 0x104, "a"), (0x104, 0x108, "b")]

    def test_contiguous_same_label_run_is_one_range(self):
        ranges = md._coalesce([0x100, 0x101, 0x102], self.regions)
        self.assertEqual(len(ranges), 1)
        self.assertEqual(
            ranges[0], {"lo": 0x100, "hi": 0x103, "label": "a", "base": 0x100}
        )

    def test_label_change_splits_a_contiguous_run(self):
        # 0x103 (label "a") and 0x104 (label "b") are numerically adjacent
        # but belong to different regions -- must not merge into one range.
        ranges = md._coalesce([0x103, 0x104], self.regions)
        self.assertEqual(len(ranges), 2)
        self.assertEqual(ranges[0]["label"], "a")
        self.assertEqual(ranges[1]["label"], "b")

    def test_gap_splits_a_run_even_with_the_same_label(self):
        ranges = md._coalesce([0x100, 0x102], self.regions)
        self.assertEqual(len(ranges), 2)

    def test_unlabeled_addresses_still_coalesce(self):
        ranges = md._coalesce([0x400, 0x401], [])
        self.assertEqual(len(ranges), 1)
        self.assertIsNone(ranges[0]["label"])


class MmrRenderTest(unittest.TestCase):
    def test_none_is_none(self):
        self.assertIsNone(md._mmr_render(None))

    def test_const_renders_as_hex(self):
        import sharc_trace as st

        self.assertEqual(md._mmr_render(st.Const(0x1234)), "0x1234")

    def test_unknown_renders_with_its_reason(self):
        import sharc_trace as st

        self.assertEqual(
            md._mmr_render(st.Unknown("uninitialized")), "unknown:uninitialized"
        )


class ParseFramesTest(unittest.TestCase):
    def test_no_spec_means_whole_capture_from_zero(self):
        self.assertEqual(md._parse_frames(None), (0, None))

    def test_parses_inclusive_bounds(self):
        self.assertEqual(md._parse_frames("295:310"), (295, 310))


class ParseArgsTest(unittest.TestCase):
    def test_defaults(self):
        args = md.parse_args(["dt2-1.16", "a.dt2cap", "b.dt2cap"])
        self.assertEqual(args.image, "dt2-1.16")
        self.assertEqual(args.capture_a, "a.dt2cap")
        self.assertEqual(args.capture_b, "b.dt2cap")
        self.assertIsNone(args.frames)
        self.assertFalse(args.only_first_divergence)
        self.assertEqual(args.top, 20)


@pytest.mark.slow
@unittest.skipUnless(DT2_116_BLOB.exists(), "DT2 1.16 firmware bytes are not available")
@unittest.skipUnless(
    IDLE_CAPTURE.exists(), "dt2-1.16-idle.dt2cap capture is not available"
)
class MemdiffSelfCompareTest(unittest.TestCase):
    """A capture diffed against ITSELF, frame for frame, must never show a
    difference -- both runners see byte-identical TX content at every
    frame, from the same deterministic post-init/post-setup start. This is
    a strong regression test for diff_states()'s own candidate-key
    restriction (tools/sharc_memdiff.py's own module note: state.overlay is
    never scanned in full): if that restriction ever missed a real write,
    the two runners could drift apart even though their real inputs never
    did, but they never should here."""

    def test_no_divergence_against_itself(self):
        result = md.memdiff(
            "dt2-1.16", str(IDLE_CAPTURE), str(IDLE_CAPTURE), frame_hi=2
        )
        self.assertEqual(result["frames_executed"], 3)
        self.assertIsNone(result["first_divergence_frame"])
        for entry in result["per_frame"]:
            self.assertFalse(entry["has_diff"])
            self.assertEqual(entry["overlay_total_ranges"], 0)
            self.assertEqual(entry["mmr_changes"], [])


if __name__ == "__main__":
    unittest.main()
