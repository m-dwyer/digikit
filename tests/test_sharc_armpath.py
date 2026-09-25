"""Tests for tools/sharc_armpath.py (lane H3, 2026-09-26).

Same two-group convention as tests/test_sharc_replay.py: `analyze_iterations()`
is pure grouping logic over a hand-built `hits` list and always runs;
`trace_arm_path()` needs the real DT2 1.16 SHARC+ image bytes (Elektron's
copyright, never committed) and a real capture file (out/captures/, also
never committed), and is skipped without either.
"""

import os
import pathlib
import sys
import unittest

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools"))
import sharc_armpath as armpath  # noqa: E402
import sharc_harness as h  # noqa: E402
import sharc_replay as replay_mod  # noqa: E402

DT2_116_BLOB = pathlib.Path("out/sections/dt2-1.16/section_7_BLOB.bin")
FULLTX_CAPTURE = pathlib.Path("out/captures/dt2-1.16-play-pretracks-fulltx.dt2cap")


def _hit(pc, instructions, sample=None):
    entry = {"pc": "%#x" % pc, "instructions": instructions}
    if sample is not None:
        entry["sample"] = sample
    return entry


class AnalyzeIterationsTest(unittest.TestCase):
    """`analyze_iterations()` needs no firmware: it only groups a
    caller-supplied `hits` list (the same shape
    `sharc_replay.call_frame_collect_all_with_hits()` returns) by
    `LOOP_TOP`. These fixtures encode the two real control-flow shapes this
    lane found by execution (see the module docstring and
    `analyze_iterations()`'s own docstring): `GUARD_SKIP`'s jump has two
    delay slots, so `AFTER_R6_LOAD` (0x1c653c, its own second delay slot)
    fires on *every* iteration regardless of `GUARD_SKIP`'s outcome, and
    `GUARD_A` (0x1c6540) is independently reachable only when `GUARD_SKIP`
    was NOT taken."""

    def test_guard_skip_taken_skips_guard_a_entirely(self):
        # field1 == 0 -> GUARD_SKIP taken -> AFTER_R6_LOAD (delay slot,
        # always runs) is immediately followed by FALLTHRU_A, never by
        # GUARD_A -- the real shape this lane found on every voice of a
        # real capture (field1/field2 always 0).
        hits = [
            _hit(armpath.LOOP_TOP, 0),
            _hit(armpath.AFTER_LOAD1, 1),
            _hit(armpath.GUARD_SKIP, 2),
            _hit(armpath.AFTER_R6_LOAD, 3),
            _hit(armpath.FALLTHRU_A, 4),
            _hit(armpath.AFTER_LOAD2, 5),
            _hit(armpath.GUARD_B, 6),
            _hit(armpath.DISPATCH_TOP, 7),
        ]
        iterations = armpath.analyze_iterations(hits)
        self.assertEqual(len(iterations), 1)
        it = iterations[0]
        self.assertTrue(it["guard_skip_taken"])
        self.assertFalse(it["guard_a_reached"])
        self.assertIsNone(it["guard_a_taken"])
        self.assertFalse(it["guard_b_taken"])
        self.assertTrue(it["reached_dispatch_top"])
        self.assertFalse(it["reached_arm_prep"])
        self.assertFalse(it["reached_arm_fn"])

    def test_guard_skip_not_taken_reaches_real_guard_a(self):
        # field1 != 0 -> GUARD_SKIP not taken -> AFTER_R6_LOAD falls
        # through normally to the real, independently reachable GUARD_A,
        # which here is taken (arms the voice) without ever reaching
        # GUARD_B/DISPATCH_TOP at all this iteration.
        hits = [
            _hit(armpath.LOOP_TOP, 0),
            _hit(armpath.AFTER_LOAD1, 1),
            _hit(armpath.GUARD_SKIP, 2),
            _hit(armpath.AFTER_R6_LOAD, 3),
            _hit(armpath.GUARD_A, 4),
            _hit(armpath.ARM_PREP, 5),
            _hit(armpath.ARM_CALL, 6),
            _hit(armpath.ARM_FN, 7),
        ]
        iterations = armpath.analyze_iterations(hits)
        it = iterations[0]
        self.assertFalse(it["guard_skip_taken"])
        self.assertTrue(it["guard_a_reached"])
        self.assertTrue(it["guard_a_taken"])
        self.assertIsNone(it["guard_b_taken"])
        self.assertFalse(it["reached_dispatch_top"])
        self.assertTrue(it["reached_arm_prep"])
        self.assertTrue(it["reached_arm_call"])
        self.assertTrue(it["reached_arm_fn"])

    def test_guard_b_taken_is_read_off_its_own_successor(self):
        hits = [
            _hit(armpath.LOOP_TOP, 0),
            _hit(armpath.AFTER_R6_LOAD, 1),
            _hit(armpath.FALLTHRU_A, 2),
            _hit(armpath.AFTER_LOAD2, 3),
            _hit(armpath.GUARD_B, 4),
            _hit(armpath.ARM_PREP, 5),
            _hit(armpath.ARM_CALL, 6),
            _hit(armpath.ARM_FN, 7),
        ]
        it = armpath.analyze_iterations(hits)[0]
        self.assertTrue(it["guard_skip_taken"])
        self.assertTrue(it["guard_b_taken"])
        self.assertTrue(it["reached_arm_fn"])

    def test_two_iterations_are_split_at_each_loop_top(self):
        hits = [
            _hit(armpath.LOOP_TOP, 0),
            _hit(armpath.AFTER_R6_LOAD, 1),
            _hit(armpath.FALLTHRU_A, 2),
            _hit(armpath.GUARD_B, 3),
            _hit(armpath.DISPATCH_TOP, 4),
            _hit(armpath.LOOP_TOP, 5),
            _hit(armpath.AFTER_R6_LOAD, 6),
            _hit(armpath.FALLTHRU_A, 7),
            _hit(armpath.GUARD_B, 8),
            _hit(armpath.DISPATCH_TOP, 9),
        ]
        iterations = armpath.analyze_iterations(hits)
        self.assertEqual([it["iteration"] for it in iterations], [0, 1])
        self.assertEqual(iterations[0]["n_hits"], 5)
        self.assertEqual(iterations[1]["n_hits"], 5)


@pytest.mark.slow
@unittest.skipUnless(
    DT2_116_BLOB.exists() and FULLTX_CAPTURE.exists(),
    "DT2 1.16 firmware bytes / real capture are not available",
)
class TraceArmPathTest(unittest.TestCase):
    """`trace_arm_path()` against a real capture -- pins this lane's own
    finding (docs/findings/06's Lane H3 section): on real DSPI2 frame
    content, FUN_1c642a's two arm guards never fire on any of the 32
    voices, because both guard-feeding bytes read 0 for every voice (see
    the module docstring)."""

    def test_render_frame_never_arms_a_voice_on_real_capture(self):
        result = armpath.trace_arm_path("dt2-1.16", str(FULLTX_CAPTURE), n_frames=2)
        self.assertEqual(result["frames_replayed"], 2)
        self.assertFalse(result["any_guard_a_taken"])
        self.assertFalse(result["any_guard_b_taken"])
        self.assertFalse(result["any_arm_fn_hit"])

        # Frame 0 is a command-1 "clear": it never reaches FUN_1c642a's own
        # dispatch loop at all.
        frame0, frame1 = result["per_frame"]
        self.assertEqual(frame0["command"], 1)
        self.assertEqual(frame0["n_iterations"], 0)

        # Frame 1 (command 3, "render") runs the real 32-voice loop, with
        # both guard-feeding bytes 0 on every voice.
        self.assertEqual(frame1["command"], 3)
        self.assertEqual(frame1["n_iterations"], 32)
        for it in frame1["iterations"]:
            self.assertEqual(it["field1_sample"]["R2"], "0x0")
            self.assertEqual(it["field2_sample"]["R2"], "0x0")
            self.assertTrue(it["guard_skip_taken"])
            self.assertFalse(it["guard_a_reached"])
            self.assertFalse(it["guard_b_taken"])
            self.assertFalse(it["reached_arm_fn"])

    def test_start_frame_skips_without_shifting_reported_index(self):
        """Lane J1: `start_frame` lets a caller bracket a real event deep
        into a long capture (e.g. a panel press near the end of a
        multi-hundred-frame idle-baseline run) without paying to replay
        every preceding frame -- `per_frame[i]["index"]` must still be the
        real capture index (`start_frame + i`), and starting at frame 1
        directly must agree with frame 1's own result from a `start_frame=0`
        run over the same two frames (this project's null-companding-record
        finding does not depend on prior frames -- see the function's own
        docstring)."""
        baseline = armpath.trace_arm_path("dt2-1.16", str(FULLTX_CAPTURE), n_frames=2)
        skipped = armpath.trace_arm_path(
            "dt2-1.16", str(FULLTX_CAPTURE), n_frames=1, start_frame=1
        )
        self.assertEqual(skipped["frames_replayed"], 1)
        self.assertEqual(skipped["start_frame"], 1)
        only = skipped["per_frame"][0]
        self.assertEqual(only["index"], 1)
        reference = baseline["per_frame"][1]
        self.assertEqual(only["command"], reference["command"])
        self.assertEqual(only["n_iterations"], reference["n_iterations"])
        self.assertEqual(only["companding_fields"], reference["companding_fields"])

    def test_forcing_the_second_guard_byte_arms_and_renders_a_voice(self):
        """A direct, diagnostic poke of one voice-iteration's own GUARD_B
        byte (not a DMA-transfer write: this lane confirmed by execution
        that neither guard byte lies inside the transfer's own landing
        ring or RX working copy -- see the module docstring and the lane
        report) reaches `FUN_1c4eaf` and marks a real voice record ACTIVE,
        which then renders a nonzero work buffer the same frame -- the
        causal chain this lane's own guard decode predicts, verified end
        to end. This is deliberately NOT what item 3 of the lane's task
        asked for (a crafted DMA transfer): no transfer content reaches
        this byte, so none was built (see the report's own "crafted
        transfer" verdict)."""
        image = "dt2-1.16"
        cap_path = str(FULLTX_CAPTURE)
        from emu import sharc_capture

        cap = sharc_capture.load(cap_path)
        memory = h.load_image_memory(image)
        init = h.run_init(memory, image)
        self.assertTrue(init.ran, init.error)
        runner = h.new_runner(memory, image, init=init)
        state = runner.state
        h.setup_voice(state, image, voice=0, sample_len=4096)
        h.setup_frame_dma(state, image)

        # The first loop iteration's own GUARD_B byte (I1(0) - 0x54,
        # confirmed by a wide read watchpoint in this lane's own
        # investigation -- see the report for the exact addresses).
        field2_iter0 = 0x24F0D8
        h._poke(state, field2_iter0, 1, width=1)

        for frame in cap.dspi2_frames[:2]:
            cmd = replay_mod.frame_command(frame.tx)
            h.write_dma_transfer(state, image, frame.tx)
            runner = h.drive_dma_completion(runner, image)
            state = runner.state
            if cmd != 3:
                continue
            runner, result, hits = replay_mod.call_frame_collect_all_with_hits(
                runner,
                image,
                trace_pcs={armpath.ARM_PREP, armpath.ARM_CALL, armpath.ARM_FN},
                patch_table=h.FRAME_PATCH_TABLE,
                max_hits=10,
            )
            state = runner.state
            break

        arm_fn_hits = [hit for hit in hits if hit["pc"] == "%#x" % armpath.ARM_FN]
        self.assertEqual(len(arm_fn_hits), 1)

        other_active = h.scan_voice_active(state, image, exclude=(0,))
        self.assertEqual(len(other_active), 1)
        ((armed_voice, active_value),) = other_active.items()
        self.assertEqual(active_value, 1)
        record = h.voice_record_address(image, armed_voice)
        work = h.read_voice_work_buffer_decimated(state, record)
        self.assertTrue(any(work))


if __name__ == "__main__":
    unittest.main()
