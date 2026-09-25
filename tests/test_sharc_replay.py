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
import sharc_harness as h  # noqa: E402
import sharc_replay as replay_mod  # noqa: E402
import sharc_run as sr  # noqa: E402

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


class FrameCommandTest(unittest.TestCase):
    """frame_command() reads the ColdFire's own header word (TX byte offset
    0, docs/findings/04's "a header written at +0x00"), not a hardcoded
    value -- every real capture on hand has this as 1 on its first frame
    and 3 on every frame after (see the module docstring's point 4)."""

    def _tx(self, header):
        tx = bytearray(0x802)
        tx[0] = (header >> 8) & 0xFF
        tx[1] = header & 0xFF
        return bytes(tx)

    def test_reads_header_word(self):
        self.assertEqual(replay_mod.frame_command(self._tx(1)), 1)
        self.assertEqual(replay_mod.frame_command(self._tx(3)), 3)

    def test_falls_back_to_render_on_implausibly_short_tx(self):
        self.assertEqual(replay_mod.frame_command(b""), 3)


class FirstNonzeroWriteTest(unittest.TestCase):
    """_first_nonzero_write() scans an sr.Runner.watch_log-shaped sequence
    for the first in-range write whose new value is nonzero, optionally
    restricted to a strided per-record field offset (SLOT_TYPE_WATCH's own
    convention)."""

    def _event(self, address, new_value, access="write", pc_sw=0x1234):
        return sr.WatchEvent(
            pc_sw=pc_sw,
            form=None,
            access=access,
            address=address,
            width=4,
            old_value=0,
            new_value=new_value,
        )

    def test_finds_first_nonzero_write_in_range(self):
        events = [
            self._event(0x100, 0, pc_sw=0x1),
            self._event(0x104, 0x2A, pc_sw=0x2),
            self._event(0x108, 0x99, pc_sw=0x3),
        ]
        hit = replay_mod._first_nonzero_write(events, 0x100, 0x110)
        self.assertEqual(hit, {"address": "0x104", "pc": "0x2", "value": "0x2a"})

    def test_ignores_reads_and_out_of_range(self):
        events = [
            self._event(0x104, 0x2A, access="read"),
            self._event(0x200, 0x2A),
        ]
        self.assertIsNone(replay_mod._first_nonzero_write(events, 0x100, 0x110))

    def test_returns_none_when_all_zero(self):
        events = [self._event(0x104, 0)]
        self.assertIsNone(replay_mod._first_nonzero_write(events, 0x100, 0x110))

    def test_strided_offset_restricts_to_matching_records(self):
        # record 0 at 0x100, record 1 at 0x200, stride 0x100, field +0x10.
        events = [
            self._event(0x108, 0x5),  # record 0, wrong offset
            self._event(0x210, 0x7, pc_sw=0x9),  # record 1, matching offset
        ]
        hit = replay_mod._first_nonzero_write(
            events, 0x100, 0x300, stride=0x100, offset=0x10
        )
        self.assertEqual(hit, {"address": "0x210", "pc": "0x9", "value": "0x7"})


class ParseArgsTest(unittest.TestCase):
    def test_defaults(self):
        args = replay_mod.parse_args(["dt2-1.16", "cap.dt2cap"])
        self.assertEqual(args.image, "dt2-1.16")
        self.assertEqual(args.capture, "cap.dt2cap")
        self.assertIsNone(args.frames)
        self.assertEqual(args.freq, 1000.0)
        self.assertEqual(args.sample_len, 4096)
        self.assertIsNone(args.force_command)
        self.assertEqual(args.provisional, [])

    def test_force_command_flag(self):
        args = replay_mod.parse_args(["dt2-1.16", "cap.dt2cap", "--force-command", "3"])
        self.assertEqual(args.force_command, 3)

    def test_provisional_flag_may_repeat(self):
        args = replay_mod.parse_args(
            [
                "dt2-1.16",
                "cap.dt2cap",
                "--provisional",
                "21p_undoc16=nop",
                "--provisional",
                "8p_undoc48=nop",
            ]
        )
        self.assertEqual(args.provisional, ["21p_undoc16=nop", "8p_undoc48=nop"])


class ParseProvisionalTest(unittest.TestCase):
    def test_empty_list_is_none(self):
        self.assertIsNone(replay_mod._parse_provisional([]))

    def test_parses_form_equals_mode_pairs(self):
        self.assertEqual(
            replay_mod._parse_provisional(["21p_undoc16=nop", "8p_undoc48=skip"]),
            {"21p_undoc16": "nop", "8p_undoc48": "skip"},
        )


@pytest.mark.slow
@unittest.skipUnless(DT2_116_BLOB.exists(), "DT2 1.16 firmware bytes are not available")
@unittest.skipUnless(
    IDLE_CAPTURE.exists(), "dt2-1.16-idle.dt2cap capture is not available"
)
class ReplayIdleCaptureTest(unittest.TestCase):
    """Replays the first three frames of the idle capture through the real
    block_handler call chain, delivering each frame's own captured TX bytes
    at the SPI-slave receive DMA's own real landing zone and "completing"
    the transfer the way real hardware would
    (sharc_harness.write_dma_transfer()/drive_dma_completion(), lane G1,
    2026-09-26) instead of poking RX_BASE/command_word directly -- and pins
    today's own result as a milestone, the same convention
    tools/sharc_harness.py's FRAME_MILESTONE uses for a single bare frame
    call.

    This capture's own header word is 1 (docs/findings/04's "a header
    written at +0x00", command "clears + stores 0") on frame 0 and 3
    ("renders") on every frame after -- true of every `.dt2cap` capture on
    hand, not just this one. So under real per-frame dispatch:

    - frame 0 (command 1) never reaches render_frame at all: a trivial
      clear, 192 instructions, "frame-returned" at the block handler's own
      return pc (FRAME_MILESTONE, 0x1c75d3) -- unchanged from the pre-lane-G1
      pin, since command 1 never reaches the DMA-delivered data at all.
    - frame 1 -- the FIRST real render call -- reaches the same
      "frame-returned" milestone at 95063 instructions: close to, but not
      identical to, the pre-lane-G1 pin (95982) -- expected, since
      render_frame's own companding-copy loop (sharc_harness.py's own
      section note above dma_landing_address()) now actually copies this
      frame's REAL, correctly byte-swapped transfer from the hardware
      landing zone, instead of copying whatever was left at RX_BASE by an
      earlier direct poke of the SAME bytes; the small instruction-count
      difference is data-dependent branching inside that copy/decode path,
      not a stop.
    - frame 2 -- the SECOND real render call, on state carried over from
      frame 1 -- ALSO reaches "frame-returned" cleanly, at 95049
      instructions, no longer hitting the unconfirmed opcode at 0x1c32b0
      the pre-lane-G1 pin hit here: that gap was in a data-dependent branch
      this real, ping-pong-delivered transfer does not exercise the same
      way an all-frames-share-one-buffer direct poke did. `first_stop` is
      therefore `None` for this whole three-frame run.

    ring_a_mono stays all-zero: the mix gate (docs/findings/06's still-open
    mix gate, DM(0x252d3c)) blocks the hand-set-up voice's output before it
    reaches the master mix, so this pins silence, not a rendered tone -- a
    fix to the mix gate changes this pin too. mix_gate_write/
    master_bus_decoded_write/slot_type_write all stay None across all three
    frames: even a fully successful render never writes any of lane C2's own
    target globals here, on this capture's own (all-zero machine-type)
    per-track data. fun_1c60a2_hits stays empty on every frame too --
    consistent with lane E2/F2's own finding that a correctly-delivered
    transfer still does not fill the companding record
    (command_record_table, 0x266220) this gate reads (see this lane's own
    report for why: the transfer fits inside command_word's own ring and
    never reaches that second one).
    """

    @classmethod
    def setUpClass(cls):
        sys.path.insert(
            0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools")
        )

    def test_three_frame_replay_milestone(self):
        result = replay_mod.replay(
            "dt2-1.16", str(IDLE_CAPTURE), n_frames=3, tone_freq=1000.0
        )
        self.assertEqual(result["frames_replayed"], 3)
        self.assertEqual(result["rx_base"], "%#x" % replay_mod.RX_BASE)
        self.assertEqual(result["other_voices_active_by_firmware"], {})
        self.assertIsNone(result["forced_command"])
        self.assertEqual(result["commands_seen"], {1: 1, 3: 2})
        self.assertEqual(result["fun_1c60a2_total_hits"], 0)
        self.assertFalse(result["any_companding_gate_pointer_nonzero"])
        self.assertFalse(result["any_signal_without_injection"])
        self.assertIsNone(result["provisional_interpretations"])
        self.assertEqual(result["provisional_used"], [])

        frame0, frame1, frame2 = result["per_frame"]

        self.assertEqual(frame0["command"], 1)
        self.assertEqual(frame0["stop_reason"], "frame-returned")
        self.assertEqual(frame0["stop_pc"], "0x1c75d3")
        self.assertEqual(frame0["instructions"], 192)
        self.assertFalse(frame0["master_bus_source_nonzero"])

        self.assertEqual(frame1["command"], 3)
        self.assertEqual(frame1["stop_reason"], "frame-returned")
        self.assertEqual(frame1["stop_pc"], "0x1c75d3")
        self.assertEqual(frame1["instructions"], 95063)
        self.assertTrue(frame1["master_bus_source_nonzero"])

        # The second real render call, unlike the pre-lane-G1 pin, does NOT
        # hit the unconfirmed opcode at 0x1c32b0 -- see the class docstring.
        self.assertEqual(frame2["command"], 3)
        self.assertEqual(frame2["stop_reason"], "frame-returned")
        self.assertEqual(frame2["stop_pc"], "0x1c75d3")
        self.assertEqual(frame2["instructions"], 95049)
        self.assertTrue(frame2["master_bus_source_nonzero"])

        self.assertIsNone(result["first_stop"])

        self.assertEqual(len(result["ring_a_mono"]), 96)
        self.assertEqual(max(abs(v) for v in result["ring_a_mono"]), 0.0)

        # Per-frame nonzero checks (track buffers / master mix / ring A),
        # the per-frame ACTIVE-byte snapshot, and lane C2's own target-global
        # writes: all three frames are silent end to end -- no track buffer,
        # the master mix, or ring A ever goes nonzero; no voice other than
        # the hand-set-up one (voice 0, excluded) is ever marked ACTIVE by
        # the firmware itself; and none of the mix gate, the master-bus
        # decode destinations, or a voice record's own slot-type byte is
        # ever written, even by a fully successful render.
        for frame in (frame0, frame1, frame2):
            self.assertEqual(frame["voice_active_by_firmware"], {})
            self.assertEqual(
                frame["track_buffers_nonzero"], {t: False for t in range(16)}
            )
            self.assertFalse(frame["any_track_buffer_nonzero"])
            self.assertFalse(frame["master_mix_nonzero"])
            self.assertFalse(frame["ring_a_nonzero"])
            self.assertIsNone(frame["mix_gate_write"])
            self.assertIsNone(frame["master_bus_decoded_write"])
            self.assertIsNone(frame["slot_type_write"])
            self.assertEqual(frame["fun_1c60a2_hits"], [])
            self.assertFalse(frame["companding_gate_pointer_nonzero"])


@pytest.mark.slow
@unittest.skipUnless(DT2_116_BLOB.exists(), "DT2 1.16 firmware bytes are not available")
class CallFrameCollectAllWithHitsTest(unittest.TestCase):
    """tools/sharc_replay.call_frame_collect_all_with_hits(): a real pc-hit
    breakpoint (Runner.breakpoints, checked natively inside
    sharc_survey.run_collect_all()'s own loop), not a memory watch on a
    write target a function is only believed to touch -- lane E2's own task
    1 (2026-09-25/26, docs/findings/06's "Lane E2" section)."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(
            0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools")
        )

    def _runner(self):
        memory = h.load_image_memory("dt2-1.16")
        init = h.run_init(memory, "dt2-1.16")
        self.assertTrue(init.ran, init.error)
        runner = h.new_runner(memory, "dt2-1.16", init=init)
        h.setup_voice(runner.state, "dt2-1.16", voice=0, sample_len=4096)
        h.setup_frame(runner.state, "dt2-1.16", command=3, ring_flag=0)
        return runner

    def test_reachable_pc_is_hit_and_frame_still_completes(self):
        # 0x1c3083: FUN_1c2b24's own CALL to FUN_1c642a -- docs/findings/06's
        # "Call convention" ("FUN_1c2b24 loads R4 = 0x2412c8 (0x1c3080) and
        # calls FUN_1c642a (0x1c3083)"), runs exactly once per frame.
        runner, result, hits = replay_mod.call_frame_collect_all_with_hits(
            self._runner(),
            "dt2-1.16",
            trace_pcs={0x1C3083},
            patch_table=h.FRAME_PATCH_TABLE,
        )
        self.assertEqual([hit["pc"] for hit in hits], ["0x1c3083"])
        # The hit did not stop the frame -- run_collect_all() resumed past
        # it and reached the same "frame-returned" milestone a plain
        # call_frame_collect_all() call reaches from this state.
        self.assertEqual(result.terminal.category, "frame-returned")
        self.assertEqual(result.terminal.pc, 0x1C75D3)

    def test_fun_1c60a2_is_never_reached_from_a_run_init_state(self):
        """FUN_1c60a2 (the machine-type change detector's own call target,
        docs/findings/06's "The SHARC machine-type consumer") does not
        execute at all from a run_init() state with one hand-set-up voice,
        in a bare command-3 frame call with no captured DSPI2 content --
        consistent with the same zero-hit result this lane found replaying
        real frames of out/captures/dt2-1.16-play-pretracks-fulltx.dt2cap
        (not committed -- Elektron-derived; see the lane's own report) and
        with a synthetic frame that forces every track's machine-type field
        to change. All three of FUN_1c60a2's own static call sites
        (0x1c32d2/0x1c330e/0x1c33f1, inside 0x1c3289-0x1c33fe) sit behind a
        32-iteration companding loop in FUN_1c2b24 itself (0x1c319a-
        0x1c31da) whose own chained GT branches never fall through to that
        block in any run tried here -- see the lane report for the
        concrete register values."""
        runner, result, hits = replay_mod.call_frame_collect_all_with_hits(
            self._runner(),
            "dt2-1.16",
            trace_pcs={0x1C60A2},
            patch_table=h.FRAME_PATCH_TABLE,
        )
        self.assertEqual(hits, [])
        self.assertEqual(result.terminal.category, "frame-returned")


if __name__ == "__main__":
    unittest.main()
