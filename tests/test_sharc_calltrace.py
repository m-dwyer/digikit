"""Tests for tools/sharc_calltrace.py.

Same two-group convention as tests/test_sharc_replay.py: the register/
stack-arg sampling helpers are pure (a hand-built sharc_core.state.State,
no firmware) and always run; calltrace() itself needs the real DT2 1.16
SHARC+ image bytes (Elektron's copyright, never committed) and a real
capture file (out/captures/, also never committed), and is skipped without
either.
"""

import os
import pathlib
import sys
import unittest

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools"))
import sharc_calltrace as ct  # noqa: E402
import sharc_trace as st  # noqa: E402

DT2_116_BLOB = pathlib.Path("out/sections/dt2-1.16/section_7_BLOB.bin")
IDLE_CAPTURE = pathlib.Path("out/captures/dt2-1.16-running-idle.dt2cap")


def _state_with_regs(**regs) -> st.State:
    uregs = {st.UREG_CODES[name]: st.Const(value) for name, value in regs.items()}
    return st.State(pc_sw=0, uregs=uregs)


class RegTest(unittest.TestCase):
    def test_reads_a_concrete_register(self):
        state = _state_with_regs(R0=0x1234)
        self.assertEqual(ct._reg(state, "R0"), 0x1234)

    def test_masks_to_32_bits(self):
        state = _state_with_regs(R0=-1)
        self.assertEqual(ct._reg(state, "R0"), 0xFFFFFFFF)

    def test_uninitialized_register_is_none(self):
        state = st.State(pc_sw=0)
        self.assertIsNone(ct._reg(state, "R0"))

    def test_unknown_value_is_none(self):
        state = st.State(pc_sw=0, uregs={st.UREG_CODES["R0"]: st.Unknown("test")})
        self.assertIsNone(ct._reg(state, "R0"))


class StackArgsTest(unittest.TestCase):
    def test_no_i6_gives_four_nones(self):
        self.assertEqual(
            ct._stack_args(st.State(pc_sw=0), None), [None, None, None, None]
        )

    def test_no_backing_memory_gives_four_nones(self):
        # state.concrete is None here, so _dm_read() can never resolve
        # anything -- this only proves the four-slot shape/order, not the
        # x4 normal-word scaling (that needs a real backing image; see
        # CalltraceArmFrameTest below for that, by construction: it asserts
        # concrete decimal stack_args values against a real replay).
        self.assertEqual(
            ct._stack_args(st.State(pc_sw=0), 0x1000), [None, None, None, None]
        )


class HexOrNoneTest(unittest.TestCase):
    def test_none_stays_none(self):
        self.assertIsNone(ct._hex_or_none(None))

    def test_int_renders_as_hex(self):
        self.assertEqual(ct._hex_or_none(0x2A), "0x2a")


class ParseFramesTest(unittest.TestCase):
    def test_no_spec_means_whole_capture_from_zero(self):
        self.assertEqual(ct._parse_frames(None), (0, None))

    def test_parses_inclusive_bounds(self):
        self.assertEqual(ct._parse_frames("295:310"), (295, 310))


class ParseArgsTest(unittest.TestCase):
    def test_defaults_to_the_builtin_entry_list(self):
        args = ct.parse_args(["dt2-1.16", "a.dt2cap"])
        self.assertEqual(args.image, "dt2-1.16")
        self.assertEqual(args.capture, "a.dt2cap")
        self.assertEqual(args.function, [])
        self.assertIsNone(args.frames)

    def test_function_is_repeatable(self):
        args = ct.parse_args(
            ["dt2-1.16", "a.dt2cap", "--function", "0x1c7442", "--function", "0x1c4e70"]
        )
        self.assertEqual(args.function, ["0x1c7442", "0x1c4e70"])


class DefaultEntryPcsTest(unittest.TestCase):
    def test_matches_the_briefs_own_list(self):
        self.assertEqual(
            ct.DEFAULT_ENTRY_PCS,
            (
                0x1C7442,
                0x1C4E70,
                0x1C4EAF,
                0x1C3289,
                0x1C60A2,
                0x1C642A,
                0x1C4ECF,
                0x1C4F81,
            ),
        )


@pytest.mark.slow
@unittest.skipUnless(DT2_116_BLOB.exists(), "DT2 1.16 firmware bytes are not available")
@unittest.skipUnless(
    IDLE_CAPTURE.exists(), "dt2-1.16-running-idle.dt2cap capture is not available"
)
class CalltraceIntegrationTest(unittest.TestCase):
    """A few real frames through the hand-set-up-voice-0 replay: FUN_1c4f81
    (voice render) is always reached once per frame this way, so tracing
    just that one entry is a cheap, deterministic smoke test that the
    breakpoint-attach/return-match machinery actually works end to end
    (entry recorded, its own return matched, R0 filled in) -- not a claim
    about the specific numbers, which tools/sharc_calltrace.py's own module
    docstring already documents as coming straight from real execution."""

    def test_voice_render_entry_and_return_are_matched(self):
        result = ct.calltrace(
            "dt2-1.16",
            str(IDLE_CAPTURE),
            entry_pcs=(0x1C4F81,),
            frame_hi=1,
        )
        self.assertEqual(result["frames_executed"], 2)
        self.assertEqual(result["entry_pcs"], ["0x1c4f81"])
        # Frame 0 is a bare "clear" command (docs/findings/04's command 1)
        # and never reaches render_frame at all; frame 1 is the first real
        # render and calls FUN_1c4f81 exactly once (one hand-set-up voice).
        frame0, frame1 = result["per_frame"]
        self.assertEqual(frame0["events"], [])
        self.assertEqual(len(frame1["events"]), 1)
        event = frame1["events"][0]
        self.assertEqual(event["pc"], "0x1c4f81")
        self.assertIsNotNone(event["return_value"])
        self.assertIsNotNone(event["return_instructions"])
        self.assertGreater(event["return_instructions"], event["instructions"])
        self.assertEqual(len(event["stack_args"]), 4)


if __name__ == "__main__":
    unittest.main()
