"""Tests for tools/sharc_replay.py.

Two groups, the same convention tests/test_sharc_harness.py uses: pure
helpers (describe_frame/_mono/parse_args) need no firmware and always run;
replay() itself needs the real DT2 1.16 SHARC+ image bytes (Elektron's
copyright, never committed here) and a real capture file (out/captures/,
also never committed -- tools/sharc_capture_run.py's own output), and is
skipped without either.
"""

import os
import pathlib
import sys
import unittest

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools"))
import sharc_replay as replay_mod  # noqa: E402

DT2_116_BLOB = pathlib.Path("out/sections/dt2-1.16/section_7_BLOB.bin")
IDLE_CAPTURE = pathlib.Path("out/captures/dt2-1.16-idle.dt2cap")


class DescribeFrameTest(unittest.TestCase):
    """No firmware or capture file needed: describe_frame() only reads the
    documented byte offsets out of a plain bytes object."""

    def _tx(self, words=None):
        tx = bytearray(0x802)
        for offset, value in (words or {}).items():
            tx[offset] = (value >> 8) & 0xFF
            tx[offset + 1] = value & 0xFF
        return bytes(tx)

    def test_all_zero_frame_has_no_nonzero_tracks(self):
        info = replay_mod.describe_frame(self._tx())
        self.assertEqual(info["tx_len"], 0x802)
        self.assertEqual(len(info["tracks"]), 16)
        self.assertTrue(all(t["machine_type"] == 0 for t in info["tracks"]))

    def test_reads_documented_per_track_offsets(self):
        tx = self._tx(
            {
                replay_mod.MACHINE_TYPE_OFFSET + 2 * 3: 5,
                replay_mod.FLAG_A_OFFSET + 2 * 3: 1,
                replay_mod.FLAG_B_OFFSET + 2 * 3: 2,
            }
        )
        info = replay_mod.describe_frame(tx)
        track3 = info["tracks"][3]
        self.assertEqual(track3["machine_type"], 5)
        self.assertEqual(track3["flag_a"], 1)
        self.assertEqual(track3["flag_b"], 2)
        # Every other track is untouched.
        for i, t in enumerate(info["tracks"]):
            if i != 3:
                self.assertEqual(t["machine_type"], 0)


class MonoTest(unittest.TestCase):
    def test_averages_lr_pairs_across_blocks(self):
        blocks = [[1.0, 3.0, -1.0, -3.0], [0.5, 0.5]]
        self.assertEqual(replay_mod._mono(blocks), [2.0, -2.0, 0.5])

    def test_empty_blocks_produce_empty_mono(self):
        self.assertEqual(replay_mod._mono([]), [])


class ParseArgsTest(unittest.TestCase):
    def test_defaults(self):
        args = replay_mod.parse_args(["dt2-1.16", "cap.dt2cap"])
        self.assertEqual(args.image, "dt2-1.16")
        self.assertEqual(args.capture, "cap.dt2cap")
        self.assertIsNone(args.frames)
        self.assertEqual(args.freq, 1000.0)
        self.assertEqual(args.sample_len, 4096)
        self.assertFalse(args.poke_candidates)

    def test_poke_candidates_flag(self):
        args = replay_mod.parse_args(["dt2-1.16", "cap.dt2cap", "--poke-candidates"])
        self.assertTrue(args.poke_candidates)


@pytest.mark.slow
@unittest.skipUnless(DT2_116_BLOB.exists(), "DT2 1.16 firmware bytes are not available")
@unittest.skipUnless(
    IDLE_CAPTURE.exists(), "dt2-1.16-idle.dt2cap capture is not available"
)
class ReplayIdleCaptureTest(unittest.TestCase):
    """Replays the first two frames of the idle capture through the real
    block_handler call chain (tools/sharc_harness.call_frame_collect_all())
    and pins today's own result as a milestone -- the same convention
    tools/sharc_harness.py's FRAME_MILESTONE uses for a single bare frame
    call. Frame 0 (a `run_init()` Runner, one hand-set-up voice, no other
    per-track frame data -- this capture's own per-track TX fields are all
    zero at every one of its 38 frames, confirmed separately by a static
    scan) reaches the same "frame-returned" milestone FRAME_MILESTONE pins.
    Frame 1 -- the first call to render_frame a *second* time, on state
    carried over from frame 0 -- hits a stop FRAME_MILESTONE's own
    single-call measurement never reaches. It used to be forms_move.py's
    Type14a handler refusing a long-word access whose decoded UREG code is
    odd (0x1c32ad, USTAT2); that gap is fixed (sharc_core.state
    ._lw_pair_mate implements the PRM's odd-register and complementary
    -pair (LW) rules, p.2-4/2-9/2-12/6-5), so frame 1 now runs one
    instruction further and hits the next, unrelated gap: an unconfirmed
    opcode at 0x1c32b0. This is still a real, reproducible tools/sharc_core
    gap, not something this test papers over: an intended fix updates this
    pin and says why, exactly like FRAME_MILESTONE's own docstring asks.

    ring_a_mono stays all-zero: the mix gate (Z2-report.md's "mix_gate")
    blocks the hand-set-up voice's output before it reaches the master mix,
    so this pins silence, not a rendered tone -- a fix to the mix gate
    changes this pin too.
    """

    @classmethod
    def setUpClass(cls):
        sys.path.insert(
            0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools")
        )

    def test_two_frame_replay_milestone(self):
        result = replay_mod.replay(
            "dt2-1.16", str(IDLE_CAPTURE), n_frames=2, tone_freq=1000.0
        )
        self.assertEqual(result["frames_replayed"], 2)
        self.assertEqual(result["rx_base"], "%#x" % replay_mod.RX_BASE)
        self.assertEqual(result["other_voices_active_by_firmware"], {})

        frame0, frame1 = result["per_frame"]
        self.assertEqual(frame0["stop_reason"], "frame-returned")
        self.assertEqual(frame0["instructions"], 95982)

        # Type14a's odd-UREG-pair gap at 0x1c32ad is fixed (sharc_core.state
        # ._lw_pair_mate: an odd ureg pairs with ureg-1, and USTAT2's own
        # complementary-pair/load-only-the-named-register rule, PRM
        # p.2-9/2-12); frame 1 now executes one instruction further and
        # hits the next, unrelated gap -- an unconfirmed opcode.
        self.assertEqual(
            frame1["stop_reason"],
            "uncertain or undecodable form: source: firmware (undocumented; unconfirmed)",
        )
        self.assertEqual(frame1["stop_pc"], "0x1c32b0")
        self.assertEqual(frame1["instructions"], 82549)

        self.assertEqual(len(result["ring_a_mono"]), 64)
        self.assertEqual(max(abs(v) for v in result["ring_a_mono"]), 0.0)

        # Per-frame nonzero checks (track buffers / master mix / ring A) and
        # the per-frame ACTIVE-byte snapshot, added alongside the pretracks
        # capture survey (B1 2026-09-25): both frames are silent end to end
        # -- no track buffer, the master mix, or ring A ever goes nonzero,
        # and no voice other than the hand-set-up one (voice 0, excluded)
        # is ever marked ACTIVE by the firmware itself.
        for frame in (frame0, frame1):
            self.assertEqual(frame["voice_active_by_firmware"], {})
            self.assertEqual(
                frame["track_buffers_nonzero"], {t: False for t in range(16)}
            )
            self.assertFalse(frame["any_track_buffer_nonzero"])
            self.assertFalse(frame["master_mix_nonzero"])
            self.assertFalse(frame["ring_a_nonzero"])


if __name__ == "__main__":
    unittest.main()
