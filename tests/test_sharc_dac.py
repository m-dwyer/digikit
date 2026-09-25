"""Tests for tools/sharc_dac.py.

Pure Python: every function here takes a plain `dm_read` callable (or
plain lists), so none of this needs the real firmware image bytes -- the
format itself (module docstring's static citations plus the isolated
`FUN_1c74a1` cross-check) is what tools/sharc_harness.py's own
tests/test_sharc_harness.py exercises against the real image.
"""

import os
import struct
import sys
import unittest
import wave

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools"))
import sharc_dac as dac  # noqa: E402


def _f2bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


class Q31RoundTripTest(unittest.TestCase):
    def test_zero(self):
        self.assertEqual(dac.float_to_q31(0.0), 0)
        self.assertEqual(dac.q31_to_float(0), 0.0)

    def test_positive_and_negative(self):
        for value in (0.5, -0.5, 0.25, -0.25, 0.999, -1.0):
            raw = dac.float_to_q31(value)
            back = dac.q31_to_float(raw)
            self.assertAlmostEqual(back, value, places=6)

    def test_full_scale_matches_prm_1_31_format(self):
        # out/refs/sharc-plus-prm p.67: "Ry = 31; Rn = FIX Fx BY Ry; /*
        # fixed-point 1.31 format */" -- full scale is 2**31, signed.
        self.assertEqual(dac.Q31_FULL_SCALE, 2**31)
        self.assertEqual(dac.float_to_q31(1.0 - 2**-31), 2**31 - 1)
        self.assertEqual(dac.float_to_q31(-1.0), -(2**31))

    def test_clips_rather_than_wraps(self):
        # A value at or past +1.0 must not two's-complement-wrap to a
        # negative raw word (the failure mode this lane's own report
        # ruled out for the real firmware conversion, but a reference
        # implementation used only by these tests must not reintroduce it).
        self.assertEqual(dac.float_to_q31(1.5), dac.Q31_FULL_SCALE - 1)
        self.assertEqual(dac.float_to_q31(-1.5), -dac.Q31_FULL_SCALE)

    def test_q31_to_float_masks_unsigned_input(self):
        # A caller may pass an already-masked (0..2**32-1) word, as
        # Runner.state reads give (see tools/sharc_harness.py's
        # read_ring_a() dm_read closure).
        self.assertAlmostEqual(dac.q31_to_float(0x80000000), -1.0, places=9)
        self.assertAlmostEqual(dac.q31_to_float(0x7FFFFFFF), 1.0, places=6)


class RingHalfTest(unittest.TestCase):
    def test_ring_half_base(self):
        self.assertEqual(dac.ring_half_base(0), dac.RING_A_BASE)
        self.assertEqual(dac.ring_half_base(1), dac.RING_A_BASE + dac.RING_A_HALF_BYTES)

    def test_ring_half_base_masks_bit0_only(self):
        # A raw DM(ring_flag) word may carry more than bit 0 -- only that
        # bit selects the half (module docstring's "Task loop" bullet).
        self.assertEqual(dac.ring_half_base(2), dac.RING_A_BASE)
        self.assertEqual(dac.ring_half_base(3), dac.RING_A_BASE + dac.RING_A_HALF_BYTES)

    def test_just_written_is_the_opposite_half(self):
        # block_handler toggles the flag AFTER the render (module
        # docstring), so "just written" is flag_value XOR 1.
        self.assertEqual(dac.ring_half_just_written(0), dac.RING_A_HALF_BYTES)
        self.assertEqual(dac.ring_half_just_written(1), 0)

    def test_none_flag_reads_half_zero(self):
        self.assertEqual(dac.ring_half_just_written(None), 0)


class DecodeRingAWordsTest(unittest.TestCase):
    def test_known_interleave(self):
        left = [0.1 + 0.01 * k for k in range(dac.SAMPLES_PER_HALF)]
        right = [-0.2 - 0.01 * k for k in range(dac.SAMPLES_PER_HALF)]
        words = []
        for l_val, r_val in zip(left, right, strict=True):
            words.append(dac.float_to_q31(l_val) & 0xFFFFFFFF)
            words.append(dac.float_to_q31(r_val) & 0xFFFFFFFF)
        frame = dac.decode_ring_a_words(words)
        for got, want in zip(frame.left, left, strict=True):
            self.assertAlmostEqual(got, want, places=6)
        for got, want in zip(frame.right, right, strict=True):
            self.assertAlmostEqual(got, want, places=6)

    def test_wrong_length_raises(self):
        with self.assertRaises(ValueError):
            dac.decode_ring_a_words([0] * 10)

    def test_none_words_decode_as_zero(self):
        words = [None] * (2 * dac.SAMPLES_PER_HALF)
        frame = dac.decode_ring_a_words(words)
        self.assertEqual(frame.left, [0.0] * dac.SAMPLES_PER_HALF)
        self.assertEqual(frame.right, [0.0] * dac.SAMPLES_PER_HALF)
        self.assertEqual(frame.left_raw, [0] * dac.SAMPLES_PER_HALF)

    def test_interleaved_q31_reconstructs_on_chip_order(self):
        words = list(range(2 * dac.SAMPLES_PER_HALF))
        # Use small positive values so the sign bit never trips.
        frame = dac.decode_ring_a_words(words)
        self.assertEqual(frame.interleaved_q31(), words)


class ReadRingAPcmTest(unittest.TestCase):
    def _memory(self, half0_words, half1_words):
        """A dict-backed dm_read over two 64-word halves at RING_A_BASE and
        RING_A_BASE + RING_A_HALF_BYTES."""
        mem = {}
        for i, word in enumerate(half0_words):
            mem[dac.RING_A_BASE + i * 4] = word & 0xFFFFFFFF
        for i, word in enumerate(half1_words):
            mem[dac.RING_A_BASE + dac.RING_A_HALF_BYTES + i * 4] = word & 0xFFFFFFFF
        return lambda addr: mem.get(addr)

    def test_reads_the_half_just_written(self):
        half0 = [1] * (2 * dac.SAMPLES_PER_HALF)
        half1 = [2] * (2 * dac.SAMPLES_PER_HALF)
        dm_read = self._memory(half0, half1)
        # ring_flag_value == 0 -> just-written half is 1 (offset
        # RING_A_HALF_BYTES) -- see ring_half_just_written()'s docstring.
        frame = dac.read_ring_a_pcm(dm_read, 0)
        self.assertEqual(frame.half_written, dac.RING_A_HALF_BYTES)
        self.assertEqual(frame.left_raw[0], 2)
        # ring_flag_value == 1 -> just-written half is 0.
        frame = dac.read_ring_a_pcm(dm_read, 1)
        self.assertEqual(frame.half_written, 0)
        self.assertEqual(frame.left_raw[0], 1)

    def test_unmapped_memory_reads_as_zero(self):
        frame = dac.read_ring_a_pcm(lambda addr: None, 0)
        self.assertEqual(frame.left, [0.0] * dac.SAMPLES_PER_HALF)
        self.assertEqual(frame.right, [0.0] * dac.SAMPLES_PER_HALF)


class ReadMasterMixPcmTest(unittest.TestCase):
    def test_planar_not_interleaved(self):
        # Module docstring's second bullet: MASTER_MIX_BASE holds 32
        # planar L floats then 32 planar R floats -- confirmed by
        # FUN_1c74a1's own +31-word displacement landing exactly on
        # MASTER_MIX_BASE + 0x80 + 4*k.
        left = [0.1 * k for k in range(dac.MASTER_MIX_CHANNEL_WORDS)]
        right = [-0.1 * k for k in range(dac.MASTER_MIX_CHANNEL_WORDS)]
        mem = {}
        for i, value in enumerate(left):
            mem[dac.MASTER_MIX_BASE + i * 4] = _f2bits(value)
        for i, value in enumerate(right):
            mem[dac.MASTER_MIX_BASE + (dac.MASTER_MIX_R_OFFSET_WORDS + i) * 4] = (
                _f2bits(value)
            )
        got_left, got_right = dac.read_master_mix_pcm(lambda addr: mem.get(addr))
        for got, want in zip(got_left, left, strict=True):
            self.assertAlmostEqual(got, want, places=6)
        for got, want in zip(got_right, right, strict=True):
            self.assertAlmostEqual(got, want, places=6)

    def test_unmapped_reads_as_zero(self):
        left, right = dac.read_master_mix_pcm(lambda addr: None)
        self.assertEqual(left, [0.0] * dac.MASTER_MIX_CHANNEL_WORDS)
        self.assertEqual(right, [0.0] * dac.MASTER_MIX_CHANNEL_WORDS)


class WriteWavStereoTest(unittest.TestCase):
    def setUp(self):
        scratch_dir = os.environ.get("CLAUDE_SCRATCHPAD", "/tmp")
        os.makedirs(scratch_dir, exist_ok=True)
        self.path = os.path.join(scratch_dir, "test_sharc_dac_stereo.wav")

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_round_trip(self):
        left = [0.5, -0.5, 0.0, 0.25]
        right = [-0.25, 0.25, 1.0, -1.0]
        dac.write_wav_stereo(self.path, left, right, sample_rate=48000)
        with wave.open(self.path, "rb") as wav:
            self.assertEqual(wav.getnchannels(), 2)
            self.assertEqual(wav.getsampwidth(), 2)
            self.assertEqual(wav.getframerate(), 48000)
            raw = wav.readframes(wav.getnframes())
        samples = struct.unpack("<%dh" % (2 * len(left)), raw)
        got_left = [samples[2 * i] / 32767 for i in range(len(left))]
        got_right = [samples[2 * i + 1] / 32767 for i in range(len(right))]
        for got, want in zip(got_left, left, strict=True):
            self.assertAlmostEqual(got, want, places=3)
        for got, want in zip(got_right, right, strict=True):
            self.assertAlmostEqual(got, want, places=3)

    def test_clips_out_of_range(self):
        dac.write_wav_stereo(self.path, [2.0], [-2.0], sample_rate=48000)
        with wave.open(self.path, "rb") as wav:
            raw = wav.readframes(wav.getnframes())
        left_sample, right_sample = struct.unpack("<hh", raw)
        self.assertEqual(left_sample, 32767)
        self.assertEqual(right_sample, -32767)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            dac.write_wav_stereo(self.path, [0.0, 0.0], [0.0])


if __name__ == "__main__":
    unittest.main()
