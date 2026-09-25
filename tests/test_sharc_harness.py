"""Tests for tools/sharc_harness.py.

Two groups: pure-Python helpers (decimate/write_wav/reference_render) need
no firmware and always run; everything that calls into a real voice-render
(setup_voice/render_blocks/check_correctness) needs the
real DT2 1.16 SHARC+ image bytes (out/sections/dt2-1.16/section_7_BLOB.bin
-- Elektron's copyright, never committed here) and is skipped without them,
the same convention tests/test_sharc_contract.py uses.
"""

import math
import os
import pathlib
import struct
import sys
import unittest
import wave

import pytest

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
        # at ``base + t`` = ``base`` for a 1-tap table -- forward taps, not
        # centered (this lane's own finding, 2026-09-25: an impulse-
        # response sweep against a concrete FUN_1c4f81 run showed the real
        # DO 64 loop reads ``sample[base + t]`` for t=0..5, never behind
        # ``base`` -- see reference_render()'s own docstring); the
        # following 2:1 decimation then averages consecutive pairs.
        samples = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        table = [[1.0]]
        out = h.reference_render(samples, table, step=1.0, n_output=3)
        self.assertEqual(len(out), 3)
        # interpolated[k] = samples[k] for k=0..5: [1,2,3,4,5,6], decimated
        # pairwise.
        self.assertEqual(out, [1.5, 3.5, 5.5])

    def test_reads_past_the_end_as_zero(self):
        table = [[1.0]]
        # The lone real sample sits at index 1 so the tap offset above
        # (base + 0) reaches it one step later, at output position 0's
        # second (odd) interpolated point -- still decimated into output
        # index 0.
        out = h.reference_render([0.0, 1.0], table, step=1.0, n_output=4)
        self.assertEqual(len(out), 4)
        self.assertEqual(out[0], 0.5)
        self.assertEqual(out[1:], [0.0, 0.0, 0.0])


class DefaultSampleLenTest(unittest.TestCase):
    """Task 1's --sample-len auto-sizing (main()'s own comment, and
    _default_sample_len()'s docstring): the same "+1 block, +32 tap-window
    pad" margin check_correctness() computes inline for each of its own
    cases."""

    def test_matches_check_correctness_own_formula_at_unity_step(self):
        n_blocks, pitch_step = 8, 1.0
        expected = int((n_blocks + 1) * 64 * pitch_step) + 32
        self.assertEqual(h._default_sample_len(n_blocks, pitch_step), expected)

    def test_scales_with_pitch_step(self):
        # A higher pitch_step consumes source samples faster per block, so
        # the auto-sized buffer must grow with it, not stay fixed.
        low = h._default_sample_len(8, 0.5)
        unity = h._default_sample_len(8, 1.0)
        high = h._default_sample_len(8, 2.0)
        self.assertLess(low, unity)
        self.assertLess(unity, high)

    def test_scales_with_blocks(self):
        self.assertLess(
            h._default_sample_len(4, 1.0),
            h._default_sample_len(16, 1.0),
        )


@unittest.skipUnless(DT2_116_BLOB.exists(), "DT2 1.16 firmware bytes are not available")
class FirmwareBackedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.memory = h.load_image_memory("dt2-1.16")

    def test_voice_zero_matches_contract(self):
        self.assertEqual(h.voice_record_address("dt2-1.16", 0), 0x2412CC)

    def test_stride_matches_contract(self):
        self.assertEqual(h.voice_record_address("dt2-1.16", 1), 0x2412CC + 0x1D8)
        self.assertEqual(h.voice_record_address("dt2-1.16", 31), 0x2412CC + 31 * 0x1D8)

    def test_rejects_out_of_range(self):
        with self.assertRaises(ValueError):
            h.voice_record_address("dt2-1.16", 32)
        with self.assertRaises(ValueError):
            h.voice_record_address("dt2-1.16", -1)

    def test_setup_voice_writes_documented_fields(self):
        runner = h.new_runner(self.memory, "dt2-1.16")
        record = h.setup_voice(
            runner.state,
            "dt2-1.16",
            0,
            sample_len=256,
            pitch_step=1.0,
            sample_base=0x310000,
        )
        self.assertEqual(record, 0x2412CC)

        def read32(offset):
            value = sr.st._dm_read(runner.state, record + offset, 4)
            return value.value

        # FIELD_SAMPLE_PTR (record+0): this lane's own finding, not
        # previously in docs/findings/06's contract table -- see
        # sharc_harness.py's module docstring's "The record+0 open
        # problem, resolved" section.
        self.assertEqual(read32(h.FIELD_SAMPLE_PTR), 0x310000)
        self.assertEqual(read32(h.FIELD_STEP), 1 << 31)  # step 1.0 in Q31
        self.assertEqual(read32(h.FIELD_PHASE), 0)
        self.assertEqual(read32(h.FIELD_END), 256 << 31 & 0xFFFFFFFF)
        active = sr.st._dm_read(runner.state, record + h.FIELD_ACTIVE, 1)
        self.assertEqual(active.value, 1)

        def read8(offset):
            return sr.st._dm_read(runner.state, record + offset, 1).value

        # The trigger's net effect (this lane's addition -- see
        # setup_voice()'s docstring): REVERSE/LOOP default False, and
        # SEED_PENDING ends cleared (the arm 0x1c4eaf sets it, but the
        # setter that follows always clears it again).
        self.assertEqual(read8(h.FIELD_REVERSE), 0)
        self.assertEqual(read8(h.FIELD_LOOP), 0)
        self.assertEqual(read8(h.FIELD_SEED_PENDING), 0)
        # Declick flags have no documented trigger-path writer; zeroed here
        # only so an init-state render is comparable to a bare-state one.
        self.assertEqual(read8(h.FIELD_FADE_IN), 0)
        self.assertEqual(read8(h.FIELD_ZERO_CROSS_MUTE), 0)
        self.assertEqual(read8(h.FIELD_RESEED), 0)

    def test_setup_voice_reverse_seeds_phase_to_end_minus_one_sample(self):
        # Finding 06's "Flags and seed": "if set, it clears it and sets the
        # phase to end - 1.0 sample ... when +0x1bb is set, else to start."
        runner = h.new_runner(self.memory, "dt2-1.16")
        record = h.setup_voice(
            runner.state,
            "dt2-1.16",
            0,
            sample_len=256,
            start=10,
            end=200,
            reverse=True,
            sample_base=0x310000,
        )

        reverse_byte = sr.st._dm_read(runner.state, record + h.FIELD_REVERSE, 1)
        self.assertEqual(reverse_byte.value, 1)
        self.assertEqual(
            sr.st._dm_read(runner.state, record + h.FIELD_PHASE, 4).value,
            (199 << 31) & 0xFFFFFFFF,
        )

    def test_call_render_passes_the_record_base_so_i0_latches_the_sample_ptr(self):
        # The corrected call convention (this module's docstring): R4 must
        # be the record address itself, not record+4 -- confirmed against
        # out/sharcdb's decode of FUN_1c642a's own call site (sw
        # 0x1c6ad3-0x1c6b00): R4 = DM(I6-4) + 4 = frame_workspace + 4 =
        # voice_records, the record base. The observable consequence: with
        # R4 = record, FUN_1c4f81's "I0 = DM(I4, M5) u=0" (sw 0x1c504a)
        # reads record+0 (FIELD_SAMPLE_PTR) exactly -- with the old
        # record+4 bug this instruction would have read record+4 (the
        # work buffer) instead, latching garbage.
        from sharc_core.encoding import UREG_CODES

        sample_ptr = 0x310000
        runner = h.new_runner(self.memory, "dt2-1.16")
        record = h.setup_voice(
            runner.state,
            "dt2-1.16",
            0,
            sample_len=256,
            pitch_step=1.0,
            sample_base=sample_ptr,
        )
        state = runner.state
        state.pc_sw = h.profile("dt2-1.16").voice_render
        state.stopped = None
        state.pending = None
        state.call_stack = []
        state.loops = []
        state.status_stack = []
        state.at_loaded_entry = False
        state.uregs[UREG_CODES["R4"]] = sr.st.Const(record)
        # sw 0x1c504c is the instruction right after "I0 = DM(I4, M5)" (sw
        # 0x1c504a) and right before I4 gets rebased to +0x1b8 -- stop
        # there so I0 is observed exactly once latched, before anything
        # later in the function could touch it again.
        runner.breakpoints = frozenset({0x1C504C})
        runner.run(max_steps=2000)
        i0 = state.uregs[UREG_CODES["I0"]]
        self.assertEqual(i0, sr.st.Const(sample_ptr))

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
        # instructions.
        self.assertGreater(results[0].instructions, 200)

    def test_check_correctness_runs_and_reports_numbers_per_case(self):
        # This lane (2026-09-25) fixed the bugs that used to make every
        # case's error meaningless: (1) call_render() left R12 (the real
        # caller's second argument -- see SESSION_TICK_ADDR's docstring) at
        # whatever run_init() happened to leave it, corrupting the past-
        # limit test; (2) setup_voice() never wrote FIELD_SAMPLE_LENGTH
        # (+0x188), so the voice deactivated one block after the record's
        # own END/START/LOOP_START span said it should not (see
        # check_correctness()'s own docstring); (3) sample_format="int16"
        # samples were packed at the wrong byte stride (2 bytes, not 4 --
        # see _SAMPLE_STRIDE_BYTES's docstring, corrected again once the
        # MODIFY (sw)/(nw) fix landed in tools/sharc_core/forms_dag.py) and
        # read_coeff_table() assumed 128 phases instead of the table's real
        # 256; (4) reference_render() centered its 6 taps around the
        # interpolation point (``base - half + 1 + t``) when the firmware's
        # own DO 64 loop is purely forward from ``base`` (``base + t`` --
        # confirmed by an impulse-response sweep against a concrete
        # FUN_1c4f81 run, see reference_render()'s own docstring); (5) --
        # the actual remaining root cause after (1)-(4), where a large
        # residual used to remain -- read_coeff_table() read the wrong
        # bytes of each 4-byte coefficient slot (first the low 16 bits,
        # then, in an intermediate version of this fix, only the upper 16;
        # each 4-byte slot is a single 32-bit Q31 fraction, and the real
        # multiplier consumes the whole 32-bit register, not either half --
        # see read_coeff_table()'s own docstring for the register-level MAC
        # trace that pinned this down). ``tools/sharc_core``'s MR-
        # accumulator housekeeping (sw 0x1c50df) and ``float_by`` (sw
        # 0x1c50e3) were both directly verified correct in the process
        # (register-level traces reproduce the PRM-documented bits-63:32
        # floor and the ``FLOAT Rx BY Ry`` operand order exactly) -- neither
        # needed a change. The remaining ~2-3e-5 max error (well inside this
        # module's 1e-4 target) is that same bits-63:32 extraction: a
        # PRM-documented *floor*, not a round (p.3-10), that a pure-float
        # reference cannot match past about 1 ULP of Q15 without
        # reimplementing the same truncation.
        report = h.check_correctness(self.memory, "dt2-1.16", n_blocks=8)
        self.assertEqual(
            set(report),
            {
                "sine_step1",
                "sine_step0.5",
                "sine_step2",
                "sine_step0.75",
                "ramp_step1",
            },
        )
        for name, case in report.items():
            self.assertEqual(case["n_blocks"], 8, name)
            self.assertIsInstance(case["max_abs_error"], float, name)
            self.assertGreaterEqual(case["max_abs_error"], 0.0, name)
            # The achieved bound: this lane measured max_abs_error of about
            # 2.4e-5 to 3.1e-5 across the default cases (a pure-float
            # reference against the hardware's bits-63:32 floor -- see this
            # test's own docstring); 1e-4 keeps headroom above that while
            # still meeting this module's target, and would catch a real
            # regression (the pre-fix value here was ~1.1e-4, itself a huge
            # improvement over the ~1.2-1.6 seen before the coefficient
            # extraction fix).
            self.assertLess(case["max_abs_error"], 1e-4, name)
            self.assertEqual(len(case["phase"]), 8, name)
            self.assertEqual(len(case["active"]), 8, name)
            # The deactivation-bug regression guard (task 3): every default
            # case's voice must stay active for all 8 blocks now that
            # FIELD_SAMPLE_LENGTH is set to match the record's own span.
            self.assertEqual(case["active"], [1] * 8, name)
            # This lane's per-block record diff diagnostic: one list of
            # {"offset", "before", "after"} dicts per block.
            self.assertEqual(len(case["record_diffs"]), 8, name)
            for diffs in case["record_diffs"]:
                for entry in diffs:
                    self.assertEqual(set(entry), {"offset", "before", "after"}, name)

    def test_coeff_table_reads_plausible_taps(self):
        table = h.read_coeff_table(self.memory, "dt2-1.16", phases=4)
        self.assertEqual(len(table), 4)
        for taps in table:
            self.assertEqual(len(taps), h.COEFF_TABLE_TAPS)
            for tap in taps:
                # (swse) int16 / 2**15: always in [-1, 1).
                self.assertGreaterEqual(tap, -1.0)
                self.assertLess(tap, 1.0)

    def test_coeff_table_default_is_256_phases_with_mirror_symmetry(self):
        # This lane's own finding (read_coeff_table()'s docstring): row i
        # and row 248-i are exact time-reversed mirrors of each other (a
        # full scan over all 256 rows confirms this holds for every i in
        # 0..248, with row 124 -- the fixed point of i -> 248-i -- reading
        # as a palindrome on its own) -- the textbook mirrored second half
        # of a linear-phase polyphase filter bank (pivoting near, not
        # exactly at, the table's own midpoint), not a pattern a wrong
        # stride into unrelated data would produce.
        table = h.read_coeff_table(self.memory, "dt2-1.16")
        self.assertEqual(len(table), 256)
        self.assertEqual(table[124], list(reversed(table[124])))
        for i in (0, 1, 4, 40, 120, 121, 200, 248):
            self.assertEqual(
                table[i],
                list(reversed(table[248 - i])),
                "row %d should mirror row %d" % (i, 248 - i),
            )

    def test_setup_voice_writes_sample_length(self):
        # FIELD_SAMPLE_LENGTH (+0x188): this lane's own finding (task 3) --
        # left at init's stale default, the past-limit test deactivates the
        # voice one block after the record's own END/START/LOOP_START span
        # says it should not (see FIELD_SAMPLE_LENGTH's own docstring).
        runner = h.new_runner(self.memory, "dt2-1.16")
        record = h.setup_voice(
            runner.state, "dt2-1.16", 0, sample_len=999, sample_base=0x310000
        )
        length = sr.st._dm_read(runner.state, record + h.FIELD_SAMPLE_LENGTH, 4)
        length_hi = sr.st._dm_read(runner.state, record + h.FIELD_SAMPLE_LENGTH_HI, 4)
        self.assertEqual(length.value, 999)
        self.assertEqual(length_hi.value, 0)

    def test_call_render_sets_r12_from_the_session_tick(self):
        # SESSION_TICK_ADDR's docstring: call_render() must read R12 fresh
        # from DM every call, not leave it at whatever run_init() (or a
        # prior call_render()) happened to leave the register at -- poke a
        # distinctive value there and confirm it reaches R12 by the time
        # the callee spills its doubled copy (sw 0x1c4f1b).
        from sharc_core.encoding import UREG_CODES

        runner = h.new_runner(self.memory, "dt2-1.16")
        state = runner.state
        h._poke(state, h.SESSION_TICK_ADDR, 5)
        record = h.setup_voice(
            state, "dt2-1.16", 0, sample_len=256, sample_base=0x310000
        )
        # Poison R12 first, the way a leftover register value would --
        # call_render() must overwrite it, not trust whatever is already
        # there.
        state.uregs[UREG_CODES["R12"]] = sr.st.Const(0xDEAD)
        state.pc_sw = h.profile("dt2-1.16").voice_render
        state.stopped = None
        state.pending = None
        state.call_stack = []
        state.loops = []
        state.status_stack = []
        state.at_loaded_entry = False
        tick = sr.st._dm_read(state, h.SESSION_TICK_ADDR, 4)
        state.uregs[UREG_CODES["R4"]] = sr.st.Const(record)
        state.uregs[UREG_CODES["R12"]] = sr.st.Const(tick.value)
        # sw 0x1c4f1b is right after "R0 = add(R2, R2)" (sw 0x1c4ef8, R0 =
        # 2*R12) spills that doubled value -- stop there and read it back.
        runner.breakpoints = frozenset({0x1C4F1B})
        runner.run(max_steps=2000)
        r0 = state.uregs[UREG_CODES["R0"]]
        self.assertEqual(r0, sr.st.Const(10))  # 2 * 5

    def test_setup_voice_loop_wraps_phase_and_stays_active(self):
        # Task 3's "also test loop=1", on a buffer shorter than one block's
        # 64 raw samples so a wrap happens inside the very first blocks.
        #
        # PHASE wrapping is a single-length correction, not a full modulo
        # done in one step (though it lands on the same result here since
        # the 64-sample-per-block advance never exceeds one buffer length):
        # with a 100-sample looped buffer, phase goes 0 -> 64 (block 0, no
        # wrap yet) -> 28 (block 1: 64+64=128, 128-100=28) -> 92 -> 56 ...,
        # consistent with docs/findings/06's "a wrap copies +0x1bc to
        # +0x1b8" describing a real wrap-detection mechanism, and with
        # FIELD_LOOP (+0x1bc) staying 1 the whole time (checked below).
        #
        # ACTIVE (+0x1b8) now stays 1 across every wrap (this lane's own
        # call_render() fixup -- see its docstring's "Loop-wrap ACTIVE
        # fixup" section for the root cause: a tools/sharc_core Type3d
        # access-width bug, confirmed by register-level single-stepping
        # both wrap sites, made the wrap's own read always see 0 instead of
        # FIELD_LOOP's real value). An earlier version of this test found
        # the *opposite* (ACTIVE clearing despite FIELD_LOOP=1) and reported
        # it as an unresolved discrepancy; that was itself the Type3d bug,
        # not a real firmware behavior -- see call_render()'s docstring.
        n_blocks = 4
        sample_len = 100
        samples = [0.1 * ((i % 7) - 3) for i in range(sample_len)]
        runner = h.new_runner(self.memory, "dt2-1.16")
        state = runner.state
        h._write_samples(state, 0x310000, samples, "int16")
        record = h.setup_voice(
            state,
            "dt2-1.16",
            0,
            sample_len=sample_len,
            loop=True,
            loop_start=0,
            end=sample_len,
            sample_base=0x310000,
        )
        phases, actives = [], []
        for _ in range(n_blocks):
            h.call_render(runner, "dt2-1.16", record)
            self.assertEqual(h._read_byte(state, record + h.FIELD_LOOP), 1)
            phases.append(h._read_q31_pair(state, record + h.FIELD_PHASE) / h.Q31)
            actives.append(h._read_byte(state, record + h.FIELD_ACTIVE))
        # Block 0 stays pre-wrap (64 < 100); block 1 wraps by exactly one
        # buffer length (64 + 64 - 100 = 28).
        self.assertEqual(phases[0], 64.0)
        self.assertEqual(phases[1], 28.0)
        self.assertEqual(actives, [1] * n_blocks)
        for phase in phases:
            self.assertGreaterEqual(phase, 0.0)
            self.assertLess(phase, sample_len)

    def test_setup_voice_loop_plays_at_least_16_blocks(self):
        # Task 3's own requirement: a looping voice must keep playing for
        # >= 16 blocks (the firmware supports this -- see call_render()'s
        # "Loop-wrap ACTIVE fixup" docstring section). 16 blocks * 64
        # samples/block advance = 1024 sample-positions worth of wraps
        # around this test's 100-sample buffer (over 10 full laps), so this
        # also exercises many repeated wraps, not just the first one.
        n_blocks = 16
        sample_len = 100
        samples = [0.1 * ((i % 7) - 3) for i in range(sample_len)]
        runner = h.new_runner(self.memory, "dt2-1.16")
        state = runner.state
        h._write_samples(state, 0x310000, samples, "int16")
        record = h.setup_voice(
            state,
            "dt2-1.16",
            0,
            sample_len=sample_len,
            loop=True,
            loop_start=0,
            end=sample_len,
            sample_base=0x310000,
        )
        phases, actives, halts = [], [], []
        for _ in range(n_blocks):
            result, floats = h.call_render(runner, "dt2-1.16", record)
            phases.append(h._read_q31_pair(state, record + h.FIELD_PHASE) / h.Q31)
            actives.append(h._read_byte(state, record + h.FIELD_ACTIVE))
            halts.append(result.halt.reason)
            self.assertEqual(len(floats), 64)
        self.assertEqual(actives, [1] * n_blocks)
        self.assertEqual(set(halts), {"return without followed call"})
        for phase in phases:
            self.assertGreaterEqual(phase, 0.0)
            self.assertLess(phase, sample_len)
        # Measured, exact phase sequence (64 mod 100 advanced 16 times, one
        # -100 correction whenever it would reach/exceed 100): a concrete
        # regression guard, not just a bounds check.
        self.assertEqual(
            phases,
            [
                64.0,
                28.0,
                92.0,
                56.0,
                20.0,
                84.0,
                48.0,
                12.0,
                76.0,
                40.0,
                4.0,
                68.0,
                32.0,
                96.0,
                60.0,
                24.0,
            ],
        )

    def test_call_render_does_not_reactivate_a_non_looping_voice(self):
        # The loop-wrap fixup (call_render()'s docstring) must not touch
        # ACTIVE for a non-looping voice -- it only fires when FIELD_LOOP
        # was set going into the call. A short non-looping buffer deactivates
        # past its own END exactly as check_correctness()'s own docstring
        # describes, and this fixup must not undo that.
        sample_len = 40  # short enough to run past END within a few blocks
        samples = [0.1 * ((i % 7) - 3) for i in range(sample_len)]
        runner = h.new_runner(self.memory, "dt2-1.16")
        state = runner.state
        h._write_samples(state, 0x310000, samples, "int16")
        record = h.setup_voice(
            state,
            "dt2-1.16",
            0,
            sample_len=sample_len,
            loop=False,
            sample_base=0x310000,
        )
        actives = []
        for _ in range(4):
            h.call_render(runner, "dt2-1.16", record)
            actives.append(h._read_byte(state, record + h.FIELD_ACTIVE))
        # Deactivates and stays deactivated -- not held active by the fixup.
        self.assertIn(0, actives)
        self.assertEqual(actives[-1], 0)


@unittest.skipUnless(DT2_116_BLOB.exists(), "DT2 1.16 firmware bytes are not available")
class InitStateRenderTest(unittest.TestCase):
    """--run-init actually feeding the render (HANDOVER-2026-09-25's "Next
    steps 1"): run_init() once here (it is ~1.24M instructions, several
    seconds) and share the resulting InitResult across every test method,
    the same way check_correctness()'s own cases now share one init run."""

    @classmethod
    def setUpClass(cls):
        cls.memory = h.load_image_memory("dt2-1.16")
        cls.init = h.run_init(cls.memory, "dt2-1.16")
        if not cls.init.ran:  # pragma: no cover - defensive, see run_init
            raise RuntimeError("run_init did not complete: %s" % cls.init.error)

    def test_run_init_returns_a_runner_with_ran_state(self):
        self.assertTrue(self.init.ran)
        self.assertIsNotNone(self.init.runner)
        self.assertIsNone(self.init.error)

    def test_new_runner_raises_on_failed_init(self):
        failed = h.InitResult(False, None, "max-steps at 0x0 (?)", None)
        with self.assertRaises(ValueError):
            h.new_runner(self.memory, "dt2-1.16", init=failed)

    def test_new_runner_with_init_does_not_mutate_the_source(self):
        # Finding 06's "Init writes": every voice is inactive with word +0
        # = the R8 argument (0x8045a6c8) before any trigger. new_runner()
        # must hand back a *copy* -- calling setup_voice()/call_render() on
        # it must not touch init.runner.state itself, so init can seed more
        # than one independent render.
        record = h.voice_record_address("dt2-1.16", 0)
        source_state = self.init.runner.state
        active_before = h._read_byte(source_state, record + h.FIELD_ACTIVE)
        sample_ptr_before = sr.st._dm_read(
            source_state, record + h.FIELD_SAMPLE_PTR, 4
        ).value
        self.assertEqual(active_before, 0)
        self.assertEqual(sample_ptr_before, 0x8045A6C8)

        runner = h.new_runner(self.memory, "dt2-1.16", init=self.init)
        self.assertIsNot(runner.state, source_state)
        record2 = h.setup_voice(runner.state, "dt2-1.16", 0, sample_len=256)
        h.call_render(runner, "dt2-1.16", record2)

        # The clone changed (ACTIVE poked to 1 by setup_voice at least);
        # the source init.runner.state did not.
        self.assertEqual(
            h._read_byte(source_state, record + h.FIELD_ACTIVE), active_before
        )
        self.assertEqual(
            sr.st._dm_read(source_state, record + h.FIELD_SAMPLE_PTR, 4).value,
            sample_ptr_before,
        )

    def test_two_new_runner_with_init_calls_are_independent(self):
        # Each check_correctness() case gets its own new_runner(init=...)
        # clone; one case's writes (e.g. voice 0's record) must not leak
        # into another's.
        runner_a = h.new_runner(self.memory, "dt2-1.16", init=self.init)
        runner_b = h.new_runner(self.memory, "dt2-1.16", init=self.init)
        record_a = h.setup_voice(
            runner_a.state, "dt2-1.16", 0, sample_len=256, sample_base=0x310000
        )
        record_b = h.setup_voice(
            runner_b.state, "dt2-1.16", 0, sample_len=256, sample_base=0x320000
        )
        self.assertEqual(record_a, record_b)
        ptr_a = sr.st._dm_read(runner_a.state, record_a + h.FIELD_SAMPLE_PTR, 4).value
        ptr_b = sr.st._dm_read(runner_b.state, record_b + h.FIELD_SAMPLE_PTR, 4).value
        self.assertEqual(ptr_a, 0x310000)
        self.assertEqual(ptr_b, 0x320000)

    def test_render_blocks_from_init_state_runs_without_error(self):
        blocks, results = h.render_blocks(self.memory, "dt2-1.16", 0, 2, init=self.init)
        self.assertEqual(len(blocks), 2)
        ok_reasons = {"return without followed call", "max-steps"}
        for result in results:
            self.assertIn(result.halt.reason, ok_reasons)

    def test_check_correctness_from_init_state_reports_active_and_diffs(self):
        report = h.check_correctness(
            self.memory, "dt2-1.16", n_blocks=2, init=self.init
        )
        for name, case in report.items():
            self.assertEqual(len(case["active"]), 2, name)
            self.assertEqual(len(case["record_diffs"]), 2, name)
            # Block 0's render is expected to touch the record (at least
            # the work buffer it writes every call) -- an empty diff would
            # mean the render did not run at all.
            self.assertTrue(case["record_diffs"][0], name)


@unittest.skipUnless(DT2_116_BLOB.exists(), "DT2 1.16 firmware bytes are not available")
class FrameRenderTest(unittest.TestCase):
    """Lane L13's frame render entry point: setup_frame()/call_frame()/
    render_frames(), calling FUN_1c2b24 (render_frame) via block_handler,
    the real firmware call chain, instead of bare with hand-set registers
    (see this module's own "Frame render entry point" section)."""

    @classmethod
    def setUpClass(cls):
        cls.memory = h.load_image_memory("dt2-1.16")

    def test_setup_frame_pokes_command_dispatch_and_returns_block_handler(self):
        import sharc_symbols as ss

        runner = h.new_runner(self.memory, "dt2-1.16")
        entry = h.setup_frame(runner.state, "dt2-1.16", command=3, ring_flag=1)
        p = ss.resolve(h.sharcmod.load("dt2-1.16"), device="dt2")
        self.assertEqual(entry, p.block_handler)

        def read32(addr):
            return sr.st._dm_read(runner.state, addr, 4).value

        self.assertEqual(read32(p.command_word_shift_src), 0)
        self.assertEqual(read32(p.command_word), 3)
        self.assertEqual(read32(p.ring_flag), 1)

    def test_read_master_mix_reads_back_poked_floats(self):
        runner = h.new_runner(self.memory, "dt2-1.16")
        for i in range(64):
            h._poke(
                runner.state,
                0x25F180 + i * 4,
                struct.unpack("<I", struct.pack("<f", float(i)))[0],
            )
        mix = h.read_master_mix(self.memory, "dt2-1.16", runner)
        self.assertEqual(len(mix), 64)
        self.assertEqual(mix, [float(i) for i in range(64)])

    def test_read_master_mix_defaults_to_zero_when_unwritten(self):
        runner = h.new_runner(self.memory, "dt2-1.16")
        self.assertEqual(h.read_master_mix(self.memory, "dt2-1.16", runner), [0.0] * 64)


@pytest.mark.slow
@unittest.skipUnless(DT2_116_BLOB.exists(), "DT2 1.16 firmware bytes are not available")
class FrameRenderFromInitTest(unittest.TestCase):
    """render_frames()'s real call chain from a run_init() state (~1.24M
    instructions, several seconds -- see InitStateRenderTest above): this
    lane's own cross-check that block_handler's real dispatch reaches the
    same known stops a bare `--root 0x1c2b24` run already found (this
    module's FRAME_PATCH_TABLE docstring), and does not regress. One
    run_init() shared across both methods, the same convention
    InitStateRenderTest uses."""

    @classmethod
    def setUpClass(cls):
        cls.memory = h.load_image_memory("dt2-1.16")

    def test_render_frames_without_patches_stops_at_the_first_known_fork(self):
        runner, results = h.render_frames(
            self.memory, "dt2-1.16", n_frames=1, patch_table=None
        )
        self.assertEqual(len(results), 1)
        halt = results[0].halt
        self.assertEqual(halt.reason.split(" ", 1)[0], "fork")
        # The first of the three known stops this lane's task brief and
        # handover both record for an (almost) empty synthetic frame.
        self.assertEqual(halt.pc_sw, 0xB88E4B)

    def test_render_frames_with_frame_patch_table_reaches_the_return_mismatch(self):
        runner, results = h.render_frames(self.memory, "dt2-1.16", n_frames=1)
        self.assertEqual(len(results), 1)
        halt = results[0].halt
        # FRAME_PATCH_TABLE's two hypotheses get past the first two known
        # stops (0xb88e4b, 0x1c4969); the Type18a bit test on a partly known
        # ASTATX now resolves the third (0x1c088e). The frame then stops on
        # a computed return in the 0xb82xxx routines.
        #
        # pc_sw and instructions both moved (from 0xb82a30/191363) once
        # tools/sharc_widthaudit.py's own audit found and this lane fixed
        # two sharc_core width bugs it reaches: Type3d's w=0 branch
        # ignoring l/x (tools/sharc_core/forms_move.py's _type_3d, see its
        # own docstring) and Type4b's local width table mis-keying (1, 1,
        # 1)/(0, 1, 1) against ACCESS_WIDTHS (_type_4b's own docstring).
        # Before those fixes, both handlers silently performed a
        # concretely wrong-width memory access instead of the correct one
        # (Type3d) or an Unknown/no-op (Type4b's now-declined long-word
        # case); the frame render used those wrong concrete values for
        # longer before finally diverging, reaching this same class of
        # halt 111,706 instructions later than it now does. This new stop
        # is the more trustworthy one -- it reflects a run that stopped
        # guessing on schedule (a wrong-width memory access misleading
        # nothing) -- not a regression.
        self.assertIn("differs from recorded return", halt.reason)
        self.assertEqual(halt.pc_sw, 0xB82BF1)
        self.assertEqual(results[0].instructions, 79625)


@unittest.skipUnless(DT2_116_BLOB.exists(), "DT2 1.16 firmware bytes are not available")
class CliFrequencyTest(unittest.TestCase):
    """Task 1's CLI semantics fix, checked end-to-end through main() itself
    (not just render_blocks()/setup_voice() internals): the WAV main()
    writes for a given --freq/--pitch-step must actually sound at
    freq * pitch_step Hz (see main()'s own comment for the corrected
    derivation, and SOURCE_SAMPLE_RATE's docstring for why).

    Verified by direct correlation against the *exact* expected frequency
    (continuous, not restricted to an FFT bin) rather than an FFT: for a
    single embedded sinusoid, correlating a window of N samples against
    cos/sin at its own exact frequency reconstructs that sinusoid's power
    with no leakage regardless of how many whole cycles fit the window (only
    an FFT bin readout has a coherent-sampling requirement) -- so this does
    not need this module's own THD+N measurement's coherent-bin machinery
    (see the module docstring's "CLI semantics corrected" section) to tell
    "the output is at the frequency main() claims" from "it is not".
    """

    @classmethod
    def setUpClass(cls):
        cls.memory = h.load_image_memory("dt2-1.16")
        SCRATCH.mkdir(parents=True, exist_ok=True)

    def _dominant_frequency_power_ratio(
        self, path: str, freq_hz: float, discard_samples: int = 32
    ) -> float:
        with wave.open(path, "rb") as wav:
            sample_rate = wav.getframerate()
            frames = wav.readframes(wav.getnframes())
        values = struct.unpack("<%dh" % (len(frames) // 2), frames)
        samples = [v / 32768.0 for v in values[discard_samples:]]
        n = len(samples)
        mean = sum(samples) / n
        total_power = sum((v - mean) ** 2 for v in samples) / n
        a_cos = (
            sum(
                samples[i] * math.cos(2 * math.pi * freq_hz * i / sample_rate)
                for i in range(n)
            )
            * 2.0
            / n
        )
        a_sin = (
            sum(
                samples[i] * math.sin(2 * math.pi * freq_hz * i / sample_rate)
                for i in range(n)
            )
            * 2.0
            / n
        )
        amplitude = math.hypot(a_cos, a_sin)
        fund_power = amplitude * amplitude / 2.0
        return fund_power / total_power if total_power else 0.0

    def _render_and_check(self, freq: float, pitch_step: float, blocks: int = 96):
        path = str(SCRATCH / ("cli_freq_%g_%g.wav" % (freq, pitch_step)))
        rc = h.main(
            [
                "dt2-1.16",
                "--blocks",
                str(blocks),
                "--pitch-step",
                str(pitch_step),
                "--freq",
                str(freq),
                "--out",
                path,
            ]
        )
        self.assertEqual(rc, 0)
        return self._dominant_frequency_power_ratio(path, freq * pitch_step)

    def test_unity_pitch_step_output_is_at_freq(self):
        ratio = self._render_and_check(500.0, 1.0)
        self.assertGreater(ratio, 0.9, "expected 500 Hz to dominate at pitch_step=1.0")

    def test_double_pitch_step_output_is_at_double_freq(self):
        # The old, wrong CLI would have put this at 2 * 500 * 2**2 = 4000 Hz
        # instead of 500 * 2 = 1000 Hz; checking the ratio at the *corrected*
        # target frequency (not the old formula's) is itself the regression
        # guard for task 1's fix.
        ratio = self._render_and_check(500.0, 2.0)
        self.assertGreater(
            ratio, 0.9, "expected 1000 Hz (500 * 2) to dominate at pitch_step=2.0"
        )

    def test_half_pitch_step_output_is_at_half_freq(self):
        ratio = self._render_and_check(250.0, 0.5)
        self.assertGreater(
            ratio, 0.9, "expected 125 Hz (250 * 0.5) to dominate at pitch_step=0.5"
        )

    def test_auto_sample_len_covers_the_requested_blocks_without_deactivating(self):
        # Task 1's other half: --sample-len omitted must auto-size large
        # enough that a long render does not run off the buffer (see
        # _default_sample_len()). 96 blocks * 64 samples/block is well past
        # the old fixed default of 4096.
        path = str(SCRATCH / "cli_auto_len.wav")
        rc = h.main(
            [
                "dt2-1.16",
                "--blocks",
                "96",
                "--pitch-step",
                "1.0",
                "--freq",
                "500",
                "--out",
                path,
            ]
        )
        self.assertEqual(rc, 0)
        with wave.open(path, "rb") as wav:
            # 96 blocks * 32 decimated samples/block.
            self.assertEqual(wav.getnframes(), 96 * 32)


if __name__ == "__main__":
    unittest.main()
