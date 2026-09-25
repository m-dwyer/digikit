"""Tests for tools/sharc_harness.py.

Two groups: pure-Python helpers (decimate/write_wav/reference_render) need
no firmware and always run; everything that calls into a real voice-render
(setup_voice/render_blocks/decode_overrides/check_correctness) needs the
real DT2 1.16 SHARC+ image bytes (out/sections/dt2-1.16/section_7_BLOB.bin
-- Elektron's copyright, never committed here) and is skipped without them,
the same convention tests/test_sharc_contract.py uses.
"""

import os
import pathlib
import struct
import sys
import unittest
import wave

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools"))
import sharc_harness as h  # noqa: E402
import sharc_run as sr  # noqa: E402

DT2_116_BLOB = pathlib.Path("out/sections/dt2-1.16/section_7_BLOB.bin")
SCRATCH = (
    pathlib.Path(os.environ.get("CLAUDE_SCRATCHPAD", "/tmp")) / "test_sharc_harness"
)


class DecimateTest(unittest.TestCase):
    def test_pairwise_average(self):
        self.assertEqual(h.decimate([1.0, 3.0, 2.0, 4.0]), [2.0, 3.0])

    def test_odd_trailing_sample_dropped(self):
        # docs/findings/06: "0xb80000 reads 64, writes 32, x0.5" -- an
        # always-even work buffer in practice, but decimate() should not
        # crash on a short/odd input either.
        self.assertEqual(h.decimate([1.0, 1.0, 1.0]), [1.0])

    def test_empty(self):
        self.assertEqual(h.decimate([]), [])


class WriteWavTest(unittest.TestCase):
    def test_round_trips_pcm16(self):
        SCRATCH.mkdir(parents=True, exist_ok=True)
        path = str(SCRATCH / "roundtrip.wav")
        h.write_wav(path, [0.0, 0.5, -0.5, 1.0, -1.0], sample_rate=48000)
        with wave.open(path, "rb") as wav:
            self.assertEqual(wav.getnchannels(), 1)
            self.assertEqual(wav.getsampwidth(), 2)
            self.assertEqual(wav.getframerate(), 48000)
            frames = wav.readframes(wav.getnframes())
        values = struct.unpack("<%dh" % (len(frames) // 2), frames)
        self.assertEqual(values[0], 0)
        self.assertGreater(values[1], 0)
        self.assertLess(values[2], 0)
        self.assertEqual(values[3], 32767)

    def test_clips_out_of_range(self):
        SCRATCH.mkdir(parents=True, exist_ok=True)
        path = str(SCRATCH / "clip.wav")
        h.write_wav(path, [10.0, -10.0], sample_rate=8000)
        with wave.open(path, "rb") as wav:
            frames = wav.readframes(wav.getnframes())
        values = struct.unpack("<2h", frames)
        self.assertEqual(values, (32767, -32767))


class ReferenceRenderTest(unittest.TestCase):
    def test_unity_step_identity_coefficients_pass_input_through(self):
        # A one-phase, one-tap "coefficient table" of [1.0] at unity step
        # degenerates the polyphase interpolator to a single-sample lookup
        # (at base - half + 1 + t = base + 1 for a 1-tap, half=0 table);
        # the following 2:1 decimation then averages consecutive pairs.
        samples = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        table = [[1.0]]
        out = h.reference_render(samples, table, step=1.0, n_output=3)
        self.assertEqual(len(out), 3)
        # interpolated[k] = samples[k+1] for k=0..5 (position 5 reads past
        # the end -> 0.0): [2,3,4,5,6,0], decimated pairwise.
        self.assertEqual(out, [2.5, 4.5, 3.0])

    def test_reads_past_the_end_as_zero(self):
        table = [[1.0]]
        # The lone real sample sits at index 1 so the tap offset above
        # (base + 1) actually reaches it, at output position 0.
        out = h.reference_render([0.0, 1.0], table, step=1.0, n_output=4)
        self.assertEqual(len(out), 4)
        self.assertEqual(out[0], 0.5)
        self.assertEqual(out[1:], [0.0, 0.0, 0.0])


class VoiceRecordAddressTest(unittest.TestCase):
    def test_voice_zero_matches_contract(self):
        self.assertEqual(h.voice_record_address(0), 0x2412CC)

    def test_stride_matches_contract(self):
        self.assertEqual(h.voice_record_address(1), 0x2412CC + 0x1D8)
        self.assertEqual(h.voice_record_address(31), 0x2412CC + 31 * 0x1D8)

    def test_rejects_out_of_range(self):
        with self.assertRaises(ValueError):
            h.voice_record_address(32)
        with self.assertRaises(ValueError):
            h.voice_record_address(-1)


@unittest.skipUnless(DT2_116_BLOB.exists(), "DT2 1.16 firmware bytes are not available")
class FirmwareBackedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.memory = h.load_image_memory("dt2-1.16")

    def test_decode_overrides_agree_with_the_database(self):
        # Regression guard: if out/sharcdb is rebuilt and these PCs no
        # longer decode as "2c" there (e.g. because the systematic decode
        # bug this module's docstring describes was fixed upstream), this
        # should fail loudly rather than silently keep forcing a stale
        # decode.
        overrides = h.decode_overrides("dt2-1.16")
        for pc, insn in overrides.items():
            self.assertEqual(insn.type_name, "2c", hex(pc))
            self.assertEqual(insn.kind, "confident", hex(pc))

    def test_setup_voice_writes_documented_fields(self):
        runner = h.new_runner(self.memory, "dt2-1.16")
        record = h.setup_voice(runner.state, 0, sample_len=256, pitch_step=1.0)
        self.assertEqual(record, 0x2412CC)

        def read32(offset):
            value = sr.st._dm_read(runner.state, record + offset, 4)
            return value.value

        self.assertEqual(read32(h.FIELD_STEP), 1 << 31)  # step 1.0 in Q31
        self.assertEqual(read32(h.FIELD_PHASE), 0)
        self.assertEqual(read32(h.FIELD_END), 256 << 31 & 0xFFFFFFFF)
        active = sr.st._dm_read(runner.state, record + h.FIELD_ACTIVE, 1)
        self.assertEqual(active.value, 1)

    def test_render_blocks_runs_the_real_render_path_without_error_halts(self):
        blocks, results = h.render_blocks(self.memory, "dt2-1.16", 0, 2)
        self.assertEqual(len(blocks), 2)
        for block in blocks:
            self.assertEqual(len(block), 64)
        ok_reasons = {"return without followed call", "max-steps"}
        for result in results:
            self.assertIn(
                result.halt.reason,
                ok_reasons,
                "unexpected halt: %s at %#x (%s)%s"
                % (
                    result.halt.reason,
                    result.halt.pc_sw,
                    result.halt.form,
                    (": " + result.halt.text) if result.halt.text else "",
                ),
            )
        # docs/findings/06's voice record contract: FUN_1c4ecf/0x1c4f81 is
        # not a handful-of-instructions stub; the first block's genuine
        # pass through the interpolation loop is at least a few hundred
        # instructions (this repo's own exploration during development saw
        # ~1000-4000 depending on the exact path taken). A later block can
        # legitimately be much shorter (this module's docstring's "Open
        # problem": block-to-block continuity is not yet established, and
        # this harness has already seen the render clear its own active
        # flag on some later call), so only the first block is asserted on.
        self.assertGreater(results[0].instructions, 200)

    def test_check_correctness_runs_and_reports_a_number(self):
        # See sharc_harness.py's module docstring and check_correctness()'s
        # own docstring: this does not assert a small error, because the
        # raw sample-data source this render path reads is not yet
        # correctly wired up (an open problem, not a bug in this test) --
        # it only asserts the machinery runs end to end and returns
        # well-formed numbers.
        report = h.check_correctness(self.memory, "dt2-1.16")
        self.assertEqual(len(report["rendered_decimated"]), 32)
        self.assertEqual(len(report["reference"]), 32)
        self.assertIsInstance(report["max_abs_error"], float)
        self.assertGreaterEqual(report["max_abs_error"], 0.0)

    def test_coeff_table_reads_plausible_taps(self):
        table = h.read_coeff_table(self.memory, phases=4)
        self.assertEqual(len(table), 4)
        for taps in table:
            self.assertEqual(len(taps), h.ADDRESSES["coeff_table_taps"])
            for tap in taps:
                # (swse) int16 / 2**15: always in [-1, 1).
                self.assertGreaterEqual(tap, -1.0)
                self.assertLess(tap, 1.0)


if __name__ == "__main__":
    unittest.main()
