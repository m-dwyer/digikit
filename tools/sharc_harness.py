"""Render one Digitakt II SHARC+ voice offline with tools/sharc_run.py and
write it to a WAV, for correctness- and speed-checking the emulator against
a real firmware code path rather than a hand-written test vector.

    uv run python tools/sharc_harness.py dt2-1.16 --blocks 8 \
        --out /tmp/voice0.wav

Library use:

    import sharc_harness as h
    memory = h.load_image_memory("dt2-1.16")
    blocks = h.render_blocks(memory, voice=0, n_blocks=8)
    h.write_wav("/tmp/voice0.wav", h.flatten(blocks), sample_rate=48000)

Per docs/findings/06-sharc-engine-and-startup.md's "The voice record
contract" and "Voices" sections (two agents checked the contract table
against the 1.16 bytes): one voice occupies a 0x1d8-byte record at
``profile.voice_records + voice*0x1d8``. FUN_1c4ecf (``profile.voice_render``,
called the way FUN_1c642a's dispatch loop calls it: **R4 = the record
address itself**, not record+4 -- see "Corrected voice call convention"
below) is one voice's render, producing 64 interpolated floats at
record+4..+0x103 and, through its own call to the decimator at 0xb80000, a
2:1-decimated 32-sample block.

**Corrected voice call convention.** An earlier version of this harness
passed R4 = record+4, compensating with two empirical pokes (record+0x1bc
set to 1, record+4's first float set to 1.0) to reach the render body at
all. Both were papering over the same off-by-4: FUN_1c642a's real call
site (sw 0x1c6ad3-0x1c6b00) loads ``I4 = DM(I6-4)`` (frame_workspace,
``profile.frame_workspace``), computes ``R13 = DM(I6-4) + 4`` once before
the 32-voice loop and passes ``R4 = R13`` unmodified to FUN_1c4ecf/
FUN_1c5576 -- i.e. R4 = frame_workspace + 4 = ``profile.voice_records``,
the record base, confirmed directly against ``out/sharcdb``'s decode at
those sw addresses. Inside FUN_1c4ecf/FUN_1c4f81, I4 is set to R4 once
(sw 0x1c4f0f) and never rebased before the fields the contract table
documents at record+4.., record+0x1b8 etc are read through it -- so those
offsets are relative to the *record base*, matching the contract table
exactly once R4 is corrected. Both empirical pokes are gone.

**The record+0 open problem, resolved: it is the raw sample-data
pointer.** FUN_1c4ecf's own entry gate (sw 0x1c4f15: ``R4 = DM(I4, M5)``,
i.e. word 0 of the record) was previously undocumented ("[O]" in the
contract table's context) and empirically zero-filled unless non-zero,
alongside the ACTIVE byte at +0x1b8. Tracing FUN_1c4f81 (out/sharcdb
decode, sw 0x1c504a): **before I4 is rebased to +0x1b8** (sw 0x1c504c),
``I0 = DM(I4, M5)`` reads that same word-0 field, and the DO 64
interpolation loop (sw 0x1c5096-0x1c5101) reloads ``I4 = I0`` every
iteration (sw 0x1c50a4) before indexing the raw samples it multiplies
against the polyphase taps. So record+0 is a pointer: null means "no
sample assigned" (the gate FUN_1c4ecf checks), non-null is the address the
DO 64 loop actually reads PCM through. ``setup_voice()`` now writes
``sample_base`` there instead of setting a dead I0 register (I0 was always
overwritten before the loop used it -- this is why seeding the register
directly, as the previous version did, had no effect).

Symbols (function/data addresses) come from ``tools/sharc_symbols.py``'s
``resolve()`` against ``out/sharcdb``, not a hardcoded dict -- see
``profile()`` below. Only record layout (byte offsets within one voice
record, and the coefficient table's phase stride) are constants here, since
those are not independently-addressed symbols.

**--run-init now actually feeds the render (2026-09-25).** Previously,
``run_init()``'s Runner and its State were thrown away: ``render_blocks()``/
``check_correctness()`` always called ``new_runner()``, which builds a
*bare* State with an empty overlay, so FUN_1c15e3's writes (finding 06's
"Init writes") never reached a render even with ``--run-init`` on the CLI.
``new_runner(..., init=...)`` (an ``InitResult`` from ``run_init()``) now
starts from a copy of init's own post-return State instead (same
overlay/uregs/mmrs FUN_1c15e3 itself left -- see ``_clone_state`` for why a
copy, and why not ``sharc_core.state._copy()``). The bare-state path stays
the default (``init`` omitted) for comparison. ``setup_voice()`` was also
still poking every field it wrote by hand regardless of what a real trigger
sets; it now writes the documented net effect of a trigger (arm +
position/step setter -- see its own docstring) instead, and reports which
fields still have no documented trigger-path writer.

**CLI semantics corrected; THD+N re-measured on a coherent-sampling render
(lane L17, task 1/2, 2026-09-25).** See ``SOURCE_SAMPLE_RATE``'s own
docstring and ``main()``'s own comment for task 1's CLI fix (the source is
now a --freq Hz tone at the firmware's own 96 kHz sample-playback rate,
--sample-len auto-sizes from --blocks/--pitch-step via
``_default_sample_len()``, and the audible output is documented as
freq * pitch_step Hz).

Task 2 re-measured THD+N at pitch_step in {0.5, 1.0, 2.0} on a long,
*coherently-sampled* render (128 blocks = 4096 decimated output samples per
case, one block discarded up front for the startup transient, --freq chosen
per case so the *output* lands on an exact DFT bin -- 996.09375 Hz, bin 85
of 4096 at the 48 kHz decimated rate -- so a plain time-domain Parseval
split (total mean-square power minus the exact-bin fundamental's own power,
via direct sine/cosine correlation, no FFT/numpy needed) has zero spectral
leakage) instead of the previous short, truncated-prefix measurement
(-35.5 dBc at step 1.0 vs -65 dBc at 0.5 and -74 dBc at 2.0). Result: **all
three steps measure within about 0.5 dB of each other**, at -92.2 dBc
(0.5), -92.6 dBc (1.0) and -92.8 dBc (2.0) -- consistent with this
harness's own 16-bit (Q15) ``sample_format="int16"`` quantization floor at
the rendered amplitude (~0.405 out of a possible 1.0: 16-bit full-scale SNR
is ~98 dB, less ``20*log10(0.405)`` ~= 8 dB of headroom gives ~90 dB, which
is what was measured), not a real interpolation-accuracy difference at
unity step. **Step 1.0 is not "genuinely worse"; the previous -35.5 dBc
reading was a measurement artifact** -- almost certainly scalloping loss
from correlating a short, non-coherently-sampled window against an
off-bin frequency (this lane's own Parseval-split fundamental-power
estimate underestimates the true fundamental whenever the analysis window
does not hold an exact integer number of cycles, inflating the apparent
residual/noise term with no change to the actual signal), not a property
of the render itself.

This lane also checked the other half of the task 2 brief -- the
coefficient row at frac=0 (``read_coeff_table()``'s phase 0, the only row
pitch_step=1.0 with an integer ``start`` ever selects) -- against both the
static image and a post-``run_init()`` runtime read of the same address
(``profile(image).coeff_table`` = 0x25d940, confirmed identical between
the two: static six taps
``[-0.0452, 0.0478, 0.3902, 0.0629, -0.0543, 0.0033]``, so "0x25d940 has no
bytes in the static image" did not hold for this image/loader -- the
static ``LoadedMemory`` already resolves it). **Row 0 is not a pure
delay/unit impulse** (that would read as one tap at 1.0 and the rest 0); it
is an ordinary 6-tap lowpass kernel with real energy on every tap, the same
shape every other phase has -- so unity pitch_step still runs the full
polyphase filter (not an identity passthrough), but that filtering is
present at every pitch_step via its own coefficient row, not something
that singles out step 1.0 -- consistent with the near-identical THD+N
measured across all three steps above, not a competing explanation for a
difference that this lane's own re-measurement did not find.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import struct
import sys
import time
import wave
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import sharc as sharcmod  # noqa: E402
import sharc_dac  # noqa: E402
import sharc_run as sr  # noqa: E402
import sharc_survey as sv  # noqa: E402
import sharc_symbols  # noqa: E402
import sharc_trace as st  # noqa: E402
from sharcldr import LoadedMemory  # noqa: E402

# Firmware addresses (functions, tables) come from tools/sharc_symbols.py's
# resolve(), not a hardcoded dict: cached per image so repeated calls in one
# process (render_blocks() in a loop, the test suite) reuse one resolution.
_PROFILE_CACHE: dict[str, sharc_symbols.Profile] = {}


def profile(image: str) -> sharc_symbols.Profile:
    """This image's resolved symbol table (tools/sharc_symbols.py), cached.
    Raises sharc_symbols.SymbolResolutionError if a symbol this harness
    needs (voice_render, voice_records, coeff_table, ...) does not resolve
    -- loudly, at call time, rather than silently rendering from a stale
    hardcoded address."""
    if image not in _PROFILE_CACHE:
        _PROFILE_CACHE[image] = sharc_symbols.resolve(
            sharcmod.load(image), device="dt2"
        )
    return _PROFILE_CACHE[image]


# Voice record layout: not independently-addressed symbols, so these stay
# as constants here rather than moving into sharc_symbols.py.
#
# 32 voice records, 0x1d8 bytes apart, at profile(image).voice_records
# (docs/findings/06's voice record contract, two-agent-checked on the 1.16
# bytes).
VOICE_RECORD_STRIDE = 0x1D8
VOICE_RECORD_COUNT = 32

# Voice record field offsets, bytes, from docs/findings/06's voice record
# contract table (two-agent-checked on the 1.16 bytes) plus this lane's own
# byte-level trace of FUN_1c4ecf/FUN_1c4f81 (see this module's docstring)
# for FIELD_SAMPLE_PTR, previously undocumented ("[O]" in the contract
# table) and empirically mis-set.
FIELD_SAMPLE_PTR = 0x0  # word 0: pointer to raw PCM; null = zero-fill gate
FIELD_WORK_BUFFER = 0x4  # 64 floats, written by render_body
FIELD_PREVIOUS_SAMPLE = 0x180  # word 0x60
FIELD_RATE = 0x184  # word 0x61, u32
# Word 0x62-0x63 (bytes +0x188/+0x18C): this lane's own finding, task 3.
# docs/findings/06 only documents +0x188 as an *init* write ("+0x188 =
# 0x170" -- 368, matching the ROM default sample's length) and does not
# say what a real trigger does with it. FUN_1c4f81's past-limit test reads
# it back (this lane's task brief's static facts: "combined with words
# 0x62/0x63 (+0x188/+0x18C, merged as (w63<<31)|(w62>>1))") *before* the
# END/LOOP_START-derived span, so it is compared as an independent bound,
# not derived from END. Left at init's stale 368 while ``setup_voice()``
# points the record at a differently-sized buffer, this lane's own
# ``check_correctness()`` deactivated the voice after exactly one block
# regardless of the real END/START/LOOP_START span (this module's own
# test asserts against this regression); writing FIELD_SAMPLE_LENGTH here
# to the same length ``end``/FIELD_END uses removes that spurious
# deactivation in every case this lane tried (unity, half and double
# pitch, sine and ramp) -- see check_correctness()'s docstring for the
# margin this in turn requires. Not yet run against a real trigger, so
# this is this lane's own finding, not a docs/findings/06 correction --
# record it there once a second agent checks it against the image bytes.
FIELD_SAMPLE_LENGTH = 0x188  # word 0x62, u32 (not Q31-paired with 0x18C)
FIELD_SAMPLE_LENGTH_HI = 0x18C  # word 0x63; 0 in every case seen so far
FIELD_LOOP_START = 0x190  # words 0x64-65, Q31 int64 low-first
FIELD_START = 0x198  # words 0x66-67
FIELD_END = 0x1A0  # words 0x68-69
FIELD_STEP = 0x1A8  # words 0x6a-6b, signed Q31 int64
FIELD_PHASE = 0x1B0  # words 0x6c-6d, Q31 int64
FIELD_FADE_IN = 0x17C
FIELD_ZERO_CROSS_MUTE = 0x17D
FIELD_RESEED = 0x17E
FIELD_ACTIVE = 0x1B8
FIELD_SEED_PENDING = 0x1BA
FIELD_REVERSE = 0x1BB
FIELD_LOOP = 0x1BC

# 128-phase, 6-tap polyphase coefficient table layout at
# profile(image).coeff_table: 6 (swse) int16 taps per phase, one per 4-byte
# (normal-word) slot, 24 bytes/phase (docs/findings/06's voice record
# contract, "Render and declick").
COEFF_TABLE_TAPS = 6
COEFF_TABLE_STRIDE_BYTES = 24

# The real caller's second argument, found for this lane's task 1 by
# tracing FUN_1c642a's own call sites (out/sharcdb, tools/sharc.py's
# last_def()/print_slice()): R4 is the record base (already documented,
# see the module docstring's "Corrected voice call convention"), and R12
# is *not* a magic constant -- it is DM(SESSION_TICK_ADDR) at the moment
# of the call. Evidence:
#
#   sw 0x1c6501  R13 = pass(R2); R15 = DM(I2, M5)   -- I2 resolves (ptr
#                table, tools/sharcdb.py) to SESSION_TICK_ADDR; this read
#                happens once per FUN_1c642a call, before the 32-voice
#                dispatch loop, so it is loop-invariant across every voice
#                that loop calls.
#   sw 0x1c6aee  R4 = pass(R13); R12 = R15           -- selector 2's call
#   sw 0x1c6afd  R4 = pass(R13); R12 = R15           -- the "any other"
#                call (FUN_1c4ecf, the one this harness renders).
#
# Inside the callee, this is *not* read as R12 directly where the static
# facts' "R0" note describes it -- sw 0x1c4ef5 "R2 = pass(R12); DM(I6-8) =
# R2" copies it, and sw 0x1c4ef8 "R0 = add(R2, R2)" doubles it before
# FUN_1c4ecf's own zero-fill gate even runs. That doubled value is what
# sw 0x1c4f1b spills (DM(I6-5) = R0) and FUN_1c4f81's past-limit test later
# reloads three times (sw 0x1c4fb1/0x1c4fb7/0x1c4fbc, as R12/R14/R15) --
# so "the caller's R0" in this lane's brief is this harness's R12 argument,
# doubled by the callee itself, not a third register the caller sets.
#
# SESSION_TICK_ADDR itself: FUN_1c15e3 (init) seeds it to the *literal*
# 0x20 (32) -- sw 0x1c1621 "M3 = 0x20", sw 0x1c1643 "DM(0x252d3c) = M3",
# both inside init's own function body (img.func(0x1c1643) resolves to
# FUN_1c15e3) -- not a guess: this lane's own post-init snapshot reads back
# exactly 32 there. The only other writer in the covered call graph is
# FUN_1cdbb2 (docs/findings/06: "a conditional per-track state
# accumulator", called from FUN_1c642a itself, sibling to the render
# calls) -- img.reach(profile(image).init, FUN_1cdbb2's entry) is False,
# so init alone never reaches it and this harness, which never calls
# FUN_1c642a either, has no path to it either: SESSION_TICK_ADDR is read
# fresh at every call_render() (never advanced by this harness), which is
# an open point ([O]), not a claim that a real multi-block run holds it
# fixed at 32.
#
# Without this, call_render() left R12 at whatever register value
# run_init()'s own ~1.24M instructions happened to leave it at (observed:
# 0, on this lane's post-init snapshot -- a leftover, not a 0 the firmware
# ever intended), so the callee's doubled R0 was 0 instead of 64. That
# 0-vs-64 difference is this lane's root cause for the voice deactivating
# after block 0 -- see check_correctness()'s docstring and this module's
# test coverage for the before/after.
SESSION_TICK_ADDR = 0x252D3C

Q31 = 1 << 31

# The firmware's own sample-playback rate: the rate a raw PCM sample is
# assumed to be recorded at when pitch_step=1.0 means "native speed" in the
# DO-64 render domain FUN_1c4f81's interpolation loop and 0xb80000's 2:1
# decimator operate in. Two independent 96000.0 literals in the image, both
# cited in docs/findings/06-sharc-engine-and-startup.md:
#
#   - "The voice record contract"/"Step [C][V]" (line ~3385): the four
#     step-setters (0x1c4afe, 0x1c4bf9, 0x1c4d88 and, by the same words,
#     0x1c4a31) compute ``step = ratio * rate / 96000 * 2**31`` -- ``rate``
#     is the voice record's own word 0x61 (FIELD_RATE, "u32", "not checked"
#     in the contract table but read by exactly these three setters), so
#     96000 is the reference this per-voice sample rate is normalized
#     against to land in the same Q31 units as ``pitch_step`` here. This is
#     the direct evidence: it is the same voice-render engine's own step
#     computation, not an unrelated filter.
#   - A classifier false-positive investigation (line ~2152) independently
#     found the float immediates pi, pi/8 and 96000.0 together in a
#     different function (blk93@0x1c5ed4, a `2*pi*f/fs` filter-coefficient
#     calculator elsewhere in the DSP chain) and concluded "The sample rate
#     is 96 kHz" from that alone -- corroborating, but not this constant's
#     primary evidence (that function is not part of the voice-render path
#     this harness drives).
#
# Combined with 0xb80000's own documented 2:1 decimation ("Voices [V]":
# "`0xb80000` (state +0x104) decimates 2:1 to 32 outputs"), the DO-64
# interpolation domain runs at 96 kHz and this harness's own decimated
# output (decimate()/write_wav(), WriteWavTest's 48000 default) is 48 kHz --
# consistent, not a coincidence: write_wav()'s ``sample_rate`` default was
# already 48000 before this lane's task 1, it was just never derived from
# this constant. See main()'s own CLI semantics fix (task 1, 2026-09-25)
# for why this constant exists: the previous CLI generated the source PCM
# at ``pitch_step``-prescaled cycles/sample instead of at this fixed rate,
# so its ``--freq`` argument did not mean "the source is a --freq Hz tone"
# at all -- see main()'s own comment for the corrected math and the
# measured before/after.
SOURCE_SAMPLE_RATE = 96000.0


def load_image_memory(image: str):
    """The same LoadedMemory sharc_run.py's CLI reads from."""
    return sr._load_image_memory(image)


def _make_runner(
    memory: LoadedMemory, image: str, start: int, regs: Mapping[str | int, int | str]
) -> sr.Runner:
    """One place for the Runner options this harness always turns on: state
    that a real boot leaves at reset (0) rather than Unknown, and the numeric
    recips model (this render path's pitch/rate math divides)."""
    return sr.Runner(
        memory,
        start,
        regs=regs,
        explicit_memory_model=True,
        approx_recips=True,
        follow_loaded_calls=True,
        max_call_depth=64,
    )


def voice_record_address(image: str, voice: int) -> int:
    if not 0 <= voice < VOICE_RECORD_COUNT:
        raise ValueError("voice must be 0..%d" % (VOICE_RECORD_COUNT - 1))
    return profile(image).voice_records + voice * VOICE_RECORD_STRIDE


@dataclass
class InitResult:
    ran: bool
    result: sr.RunResult | None
    error: str | None
    # The Runner FUN_1c15e3 actually ran in, at its own post-return State
    # (same overlay/uregs/mmrs init itself left) -- present only when
    # ``ran`` is true. ``new_runner(..., init=this)`` hands out a *copy* of
    # this runner's state (see ``_clone_state``) so one run_init() can seed
    # several independent renders (render_blocks(), each check_correctness()
    # case) instead of each rebuilding the ~1.24M-instruction init run.
    runner: sr.Runner | None = None


def run_init(memory, image: str, *, max_steps: int = 2_000_000) -> InitResult:
    """Attempt FUN_1c15e3 (docs/findings/06: no arguments) to its return.

    On the 1.16 image it returns after about 1.24 million instructions.
    ``ran`` is true only when the function returned; reaching MAX_STEPS
    first is a failure, not a partial success. On success, the returned
    ``InitResult.runner`` is the same Runner init ran in -- its ``.state``
    is what a real boot leaves before any voice is triggered (see finding
    06's "Init writes": every voice's word +0 is the R8 argument, +0x1b8 is
    0). Pass this to ``new_runner(..., init=...)`` to render from it instead
    of a bare state.
    """
    runner = _make_runner(memory, image, profile(image).init, {"I6": 0x300000})
    try:
        result = runner.run(max_steps=max_steps)
    except Exception as exc:  # pragma: no cover - defensive, see docstring
        return InitResult(False, None, str(exc))
    ok = result.halt.reason == "return without followed call"
    return InitResult(
        ok,
        result,
        None
        if ok
        else "%s at %#x (%s)"
        % (result.halt.reason, result.halt.pc_sw, result.halt.form),
        runner if ok else None,
    )


def _write_q31_pair(state, address: int, value: int) -> None:
    """A signed Q31 int64 (sample position, in either loop-start/start/end
    or step/phase's Q31 units), low word first (docs/findings/06)."""
    raw = value & 0xFFFFFFFFFFFFFFFF
    low, high = raw & 0xFFFFFFFF, (raw >> 32) & 0xFFFFFFFF
    _poke(state, address, low)
    _poke(state, address + 4, high)


def _poke(state, address: int, value: int, width: int = 4) -> None:
    ok = st._dm_write(state, address, width, st.Const(value & ((1 << (8 * width)) - 1)))
    if not ok:
        raise ValueError(
            "poke at %#x (width %d) did not take effect" % (address, width)
        )


def _read_q31_pair(state, address: int) -> int:
    """The inverse of ``_write_q31_pair``: a signed Q31 int64, low word
    first. Unwritten (explicit_memory_model) reads as 0."""
    lo = st._dm_read(state, address, 4)
    hi = st._dm_read(state, address + 4, 4)
    low = lo.value & 0xFFFFFFFF if lo is not None else 0
    high = hi.value & 0xFFFFFFFF if hi is not None else 0
    raw = (high << 32) | low
    if raw & (1 << 63):
        raw -= 1 << 64
    return raw


# Bytes between successive encoded samples in DM, by sample_format --
# *not* the same as the number of bytes this lane actually writes per
# sample (see ``_SAMPLE_WRITE_WIDTH_BYTES``).
#
# **Corrected (2026-09-25) after tools/sharc_core/forms_dag.py's Type 7a
# MODIFY fix (w/l honoured, w=bit39, l=bit23) landed**: every MODIFY in
# this loop is (sw) (see the module docstring's wave-2 note), so
# sw 0x1c50af's ``modify(I4, M3)`` now steps I4 in *short-word* (2-byte)
# units, not 4. A concrete, register-level single-step of FUN_1c4f81's DO
# 64 loop against the fixed decoder (this lane's own scratch trace) shows
# I4 landing at 0x310000, 0x310002, 0x310004, ... for successive raw-tap
# reads within one iteration -- packed int16 PCM, two bytes apart, not one
# per 4-byte slot. The previous "4 bytes apart" claim on file here was
# measured *before* the MODIFY fix, when the (sw)/(nw) distinction was not
# yet honoured and every MODIFY silently scaled as if raw-byte, landing 4
# bytes apart by coincidence (still a smoothly-varying, plausible-looking
# int16 run -- not garbage -- which is why the wrong stride survived
# uncaught). It also now matches ``read_coeff_table()``'s own
# already-validated convention for the polyphase table stride (24
# bytes/phase, 6 taps): that address math goes through the *byte-addressed*
# ``memory.read()`` path directly, never through a (sw) MODIFY, so it was
# never affected by this bug either way.
_SAMPLE_STRIDE_BYTES = {"float32": 4, "int16": 2}
# Bytes actually written at each sample's own address: "int16" writes only
# the low 2 bytes of its 4-byte slot (a plain short-word poke); the slot's
# high 2 bytes are left unwritten (explicit_memory_model: reads back 0),
# matching what a short-word-sign-extended *read* of that same address
# looks at -- the DO 64 loop's own reads never touch the high half.
_SAMPLE_WRITE_WIDTH_BYTES = {"float32": 4, "int16": 2}


def _sample_to_word(value: float, sample_format: str) -> int:
    """Encode one synthetic sample as the raw DM word this harness writes
    at ``sample_base`` -- what FUN_1c4f81's DO 64 loop reads through
    record+0 as PCM (see ``_SAMPLE_STRIDE_BYTES``'s docstring for the
    address-space reasoning).

    ``"int16"`` (this lane's own finding, now the default -- see
    ``_SAMPLE_STRIDE_BYTES``) packs VALUE as a signed Q15 fixed-point word
    (round(value * 32768), clamped to the int16 range, masked to 16 bits):
    the same (swse) int16 / 2**15 convention ``read_coeff_table()`` already
    uses for the polyphase taps (docs/findings/06's "Render and declick"),
    which this lane's static facts also give for the raw sample reads
    themselves ("short-word sign-extended (int16)", sw 0x1c50b4 etc).

    Confirmed empirically, not just by the decode-table type name: with
    ``"float32"`` (this harness's previous default) at a 4-byte stride, a
    rendered block's values had no resemblance to the input sine at all.
    Switching the encoding alone to this Q15 int16 packing was necessary
    but not sufficient: a large mismatch against ``reference_render()``
    remained even after this fix and ``read_coeff_table(phases=256)`` --
    tracked down (this lane, 2026-09-25) to ``read_coeff_table()`` itself
    reading the wrong half of each 4-byte coefficient slot (see its own
    docstring), not to ``tools/sharc_core``'s MR-accumulator housekeeping
    or ``float_by``, both of which a register-level trace confirmed
    correct (``alu_float_convert_scaled``'s rx/ry operand order matches the
    raw instruction bytes exactly at sw 0x1c50e3 -- rn=2, rx=2, ry=6 --
    contradicting an earlier lane's manual byte decode that had rx and ry
    swapped; no fix was needed there).

    ``"float32"`` packs VALUE as an IEEE-754 float word instead -- kept
    only for an explicit before/after comparison (see the module's tests),
    not because any evidence supports it as the real format.
    """
    if sample_format == "float32":
        return struct.unpack("<I", struct.pack("<f", value))[0]
    if sample_format == "int16":
        quantized = max(-32768, min(32767, int(round(value * 32768.0))))
        return quantized & 0xFFFF
    raise ValueError("unknown sample_format %r" % (sample_format,))


def _dequantize_sample(value: float, sample_format: str) -> float:
    """The float value actually stored (and later read back) after
    ``_sample_to_word(value, sample_format)`` -- the inverse quantization,
    used to build ``reference_render()``'s input from exactly what the
    emulated read will see, so ``check_correctness()`` measures
    interpolation-math error rather than format-quantization noise on top
    of it. ``"float32"`` round-trips through the same IEEE-754 packing
    (a no-op for any value already representable, which every
    ``_make_test_input()`` value is)."""
    word = _sample_to_word(value, sample_format)
    if sample_format == "float32":
        return struct.unpack("<f", struct.pack("<I", word))[0]
    if sample_format == "int16":
        signed = word - 0x10000 if word & 0x8000 else word
        return signed / 32768.0
    raise ValueError("unknown sample_format %r" % (sample_format,))


def _write_samples(
    state, base: int, values: Sequence[float], sample_format: str = "int16"
) -> None:
    stride = _SAMPLE_STRIDE_BYTES[sample_format]
    width = _SAMPLE_WRITE_WIDTH_BYTES[sample_format]
    for i, value in enumerate(values):
        _poke(
            state, base + i * stride, _sample_to_word(value, sample_format), width=width
        )


def _read_byte(state, address: int) -> int:
    raw = st._dm_read(state, address, 1)
    return raw.value & 0xFF if raw is not None else 0


def _clone_state(state: st.State) -> st.State:
    """A copy of STATE safe to run forward independently of the original --
    used so one ``run_init()`` (~1.24M instructions) can seed several
    independent renders (``render_blocks()``, each ``check_correctness()``
    case) without one's writes leaking into the next.

    Deliberately not ``sharc_core.state._copy()``: that helper (used by the
    symbolic tracer's own forks) builds its ``State(...)`` with positional
    args that stop before the ``explicit_memory_model`` field, so it always
    resets a clone to ``explicit_memory_model=False`` regardless of the
    source -- silently turning "unwritten DM reads as 0" back into
    "unwritten DM reads as Unknown" for every clone, which is exactly the
    thing that would make a post-init render fork/halt where a bare one
    would not. ``dataclasses.replace`` keeps every scalar config field
    (``explicit_memory_model``, ``approx_recips``, ``follow_loaded_calls``,
    ...) as the source had it, and only the mutable containers a step can
    write through are given fresh copies.
    """
    return replace(
        state,
        uregs=dict(state.uregs),
        trace=[dict(event) for event in state.trace],
        overlay=dict(state.overlay),
        call_stack=list(state.call_stack),
        loops=list(state.loops),
        status_stack=list(state.status_stack),
        mmrs=dict(state.mmrs),
        special=dict(state.special),
    )


def setup_voice(
    state,
    image: str,
    voice: int,
    *,
    sample_len: int,
    pitch_step: float = 1.0,
    start: int = 0,
    end: int | None = None,
    loop_start: int = 0,
    loop: bool = False,
    reverse: bool = False,
    sample_base: int = 0x310000,
) -> int:
    """Write one voice record's fields per docs/findings/06's voice record
    contract (plus FIELD_SAMPLE_PTR, this lane's own finding -- see this
    module's docstring), and return its address. ``pitch_step`` is in input
    samples per output (pre-interpolation) step; 1.0 is unity speed.
    ``sample_base`` is the raw-PCM address stored at record+0 -- the caller
    is responsible for having written samples there (``call_render`` no
    longer pokes a register for this; see the docstring's "Corrected voice
    call convention").

    This writes the net effect finding 06's "Flags and seed" documents for
    a genuine trigger (the arm ``0x1c4eaf`` -- ACTIVE=1, SEED_PENDING=1 --
    immediately followed by one of the four position/step setters, which
    tests SEED_PENDING, clears it, and seeds PHASE to ``start`` or
    ``end - 1`` sample when REVERSE is set): ACTIVE, SEED_PENDING (ends
    cleared), REVERSE, LOOP, STEP, START/END/LOOP_START and PHASE are all
    set here from a documented writer. ``loop``/``reverse`` default to
    False/False, matching what an unset (explicit_memory_model) record
    would read as -- pass them explicitly once a real trigger's argument
    values are known.

    Fields this harness still has no documented trigger-path writer for --
    FIELD_SAMPLE_PTR (finding 06: "[O] the firmware writer of word +0 for a
    playing voice is not yet found"), FIELD_SAMPLE_LENGTH (this lane's own
    finding -- see its docstring), fade-in/zero-cross-mute/reseed (the
    contract table marks their writer "not checked"), and previous-sample
    (documented writer is the render function ``0x1c4f81`` itself, not a
    setter) -- are still poked here to a fixed value (sample_base; sample_len;
    0; 0) rather than left to whatever ``run_init``'s own FUN_1c7442 wrote,
    so a render from init state is comparable to one from bare state
    instead of differing for reasons unrelated to the trigger. This is a
    deliberate hand-poke, not a claim about what the real trigger writes
    there.
    """
    record = voice_record_address(image, voice)
    end = sample_len if end is None else end
    _poke(state, record + FIELD_SAMPLE_PTR, sample_base)
    _poke(state, record + FIELD_SAMPLE_LENGTH, sample_len)
    _poke(state, record + FIELD_SAMPLE_LENGTH_HI, 0)
    _poke(state, record + FIELD_ACTIVE, 1, width=1)
    _poke(state, record + FIELD_SEED_PENDING, 0, width=1)
    _poke(state, record + FIELD_REVERSE, 1 if reverse else 0, width=1)
    _poke(state, record + FIELD_LOOP, 1 if loop else 0, width=1)
    _poke(state, record + FIELD_FADE_IN, 0, width=1)
    _poke(state, record + FIELD_ZERO_CROSS_MUTE, 0, width=1)
    _poke(state, record + FIELD_RESEED, 0, width=1)
    _write_q31_pair(state, record + FIELD_LOOP_START, loop_start << 31)
    _write_q31_pair(state, record + FIELD_START, start << 31)
    _write_q31_pair(state, record + FIELD_END, end << 31)
    _write_q31_pair(state, record + FIELD_STEP, round(pitch_step * Q31))
    _write_q31_pair(
        state, record + FIELD_PHASE, ((end - 1) if reverse else start) << 31
    )
    _poke(state, record + FIELD_PREVIOUS_SAMPLE, 0)
    return record


def new_runner(
    memory: LoadedMemory,
    image: str,
    *,
    stack_base: int = 0x300000,
    init: InitResult | None = None,
) -> sr.Runner:
    """A Runner parked at profile(image).voice_render, ready for repeated
    calls through ``call_render`` -- one State/overlay is shared across
    those calls (a Runner's own State is otherwise created once and never
    reset; see ``call_render``'s docstring for why a *fresh* Runner per
    call, tried first, does not work: each Runner's State starts from an
    empty overlay, so a second Runner never sees the first's
    ``setup_voice()`` pokes).

    With ``init`` omitted (the default), the returned Runner's State is
    bare: every field reset-zero (``State.explicit_memory_model``) until
    ``setup_voice()`` pokes it -- this is the "bare-state" path kept for
    comparison against a real boot. With ``init`` given (an ``InitResult``
    from ``run_init()`` with ``ran=True``), the returned Runner starts from
    a *copy* of init's own post-return State (see ``_clone_state``): same
    overlay/uregs/mmrs FUN_1c15e3 itself left, so ``setup_voice()`` and
    ``call_render()`` see whatever init actually wrote (finding 06's "Init
    writes") for fields this harness does not poke, instead of a blank
    overlay silently discarding it. ``init.runner``/``init.runner.state``
    are never mutated by this -- ``run_init()`` can be called once and its
    ``InitResult`` handed to several ``new_runner(..., init=...)`` calls
    (``render_blocks()``, each ``check_correctness()`` case) independently.
    """
    if init is not None:
        if not init.ran or init.runner is None:
            raise ValueError(
                "new_runner: init did not complete (%s); cannot start a "
                "render from it" % init.error
            )
        runner = _make_runner(
            memory, image, profile(image).voice_render, {"I6": stack_base}
        )
        runner.state = _clone_state(init.runner.state)
        return runner
    return _make_runner(memory, image, profile(image).voice_render, {"I6": stack_base})


def call_render(
    runner: sr.Runner, image: str, record: int
) -> tuple[sr.RunResult, list[float]]:
    """Call FUN_1c4ecf the way FUN_1c642a's dispatch loop calls it -- R4 =
    the record address itself, not record+4 (see this module's docstring's
    "Corrected voice call convention") -- on RUNNER's already-set-up State
    (see ``new_runner``/``setup_voice``), and return (RunResult, the 64
    work-buffer floats).

    Resets RUNNER's control-flow state (pc_sw, call stack, loops, pending
    delay slot) as a fresh call needs, but deliberately keeps its DM
    overlay (the voice record ``setup_voice`` wrote) and every other
    register (the callee saves and restores the ones it uses -- see this
    module's docstring's trace of FUN_1c4ecf's own prologue/epilogue), so
    calling this again for the next block is a closer match to what
    FUN_1c642a's own dispatch loop does than starting over from
    ``new_runner`` would be.

    The raw sample pointer is not a register here (see ``setup_voice``'s
    ``sample_base``): it is read by the callee from record+0
    (FIELD_SAMPLE_PTR), a persistent record field, not reset per call.

    R4 is the record address (the real call convention -- see this
    module's docstring). R12 is the real caller's second argument too (see
    ``SESSION_TICK_ADDR``'s docstring): read fresh from DM every call
    (never advanced by this harness -- an open point, not a claim that a
    real multi-block run holds it fixed) instead of left at whatever
    register value the previous run (``run_init()`` or a prior
    ``call_render()``) happened to leave R12 at.

    **Loop-wrap ACTIVE (task 3, 2026-09-25; fixed at the source by lane
    W1, 2026-09-25).** docs/findings/06 claims "a wrap copies +0x1bc
    [FIELD_LOOP] to +0x1b8 [FIELD_ACTIVE]" (0x1c50fe forward, 0x1c530a/
    0x1c5325 reverse). The wrap's read at 0x1c50fb/0x1c5322
    (``R2 = DM(I1, M6)``, a Type3d instruction, fields l=0, x=0, w=0,
    ex=0) used to be misdecoded by ``tools/sharc_core/forms_move.py``'s
    ``_type_3d``, which hardcoded ``access_width = "normal-word"`` for
    its w=0/ex=0 branch instead of consulting ``ACCESS_WIDTHS[(l, x, 0)]``
    the way the sibling Type4d store at the very same site already did --
    see that function's own docstring for the PRM citation and the
    register-override confirmation. With that fixed, this run's own
    Type3d load now performs the correct 1-byte read, and ACTIVE tracks
    FIELD_LOOP across a wrap on its own, with no harness-side fixup
    needed; no ``call_render()`` code doctors ACTIVE any more.
    """
    from sharc_core.encoding import UREG_CODES

    state = runner.state
    state.pc_sw = profile(image).voice_render
    state.stopped = None
    state.pending = None
    state.call_stack = []
    state.loops = []
    state.status_stack = []
    state.at_loaded_entry = False
    tick = st._dm_read(state, SESSION_TICK_ADDR, 4)
    state.uregs[UREG_CODES["R4"]] = st.Const(record)
    state.uregs[UREG_CODES["R12"]] = st.Const(
        tick.value & 0xFFFFFFFF if tick is not None else 0
    )
    runner.instructions = 0
    runner.form_counts.clear()
    runner.max_call_depth_reached = 0

    result = runner.run(max_steps=20_000)
    floats = []
    for i in range(64):
        raw = st._dm_read(state, record + FIELD_WORK_BUFFER + i * 4, 4)
        floats.append(
            struct.unpack("<f", struct.pack("<I", raw.value & 0xFFFFFFFF))[0]
            if raw is not None
            else 0.0
        )

    return result, floats


def render_blocks(
    memory,
    image: str,
    voice: int,
    n_blocks: int,
    *,
    pitch_step: float = 1.0,
    sample_len: int = 4096,
    sample_base: int = 0x310000,
    samples: Sequence[float] | None = None,
    sample_format: str = "int16",
    init: InitResult | None = None,
) -> tuple[list[list[float]], list[sr.RunResult]]:
    """N call_render() calls against one freshly set-up voice record,
    sharing a single Runner/State (see ``new_runner``/``call_render``) so
    the record's own persistent fields (phase, previous sample, ...) carry
    over between calls the way they would across real render-loop
    invocations -- see ``check_correctness()`` for this measured against an
    independent reference, including phase continuity across blocks.

    ``samples``, if given, is written to ``sample_base`` before the first
    call (FIELD_SAMPLE_PTR then points at it), e.g. a synthetic sine for an
    audible WAV, encoded per ``sample_format`` (see ``_sample_to_word``). If
    omitted, the record still points at ``sample_base`` but nothing is
    written there, so unwritten DM reads as 0 (State.explicit_memory_model):
    a silent (but not zero-fill-gated, since FIELD_SAMPLE_PTR is still
    non-null) render, useful for an instructions/block measurement alone.

    ``init``, if given (an ``InitResult`` from ``run_init()``), starts the
    render from a copy of init's own post-return state instead of a bare
    one -- see ``new_runner``.
    """
    runner = new_runner(memory, image, init=init)
    state = runner.state
    if samples is not None:
        _write_samples(state, sample_base, samples, sample_format)
    record = setup_voice(
        state,
        image,
        voice,
        sample_len=sample_len,
        pitch_step=pitch_step,
        sample_base=sample_base,
    )
    blocks, results = [], []
    for _ in range(n_blocks):
        result, floats = call_render(runner, image, record)
        blocks.append(floats)
        results.append(result)
    return blocks, results


# --- Frame render entry point (lane L13, 2026-09-25) ------------------
#
# render_blocks()/call_render() above call FUN_1c4ecf (one voice's render)
# directly. This section instead calls FUN_1c2b24 (render_frame, the whole
# frame's orchestrator -- docs/findings/06's "FUN_1c2b24 is the per-frame
# render orchestrator") the way the firmware actually reaches it: through
# block_handler (0x1c74cd), the audio task's real command dispatcher,
# rather than jumping into render_frame bare with hand-set R4/R12/R8/stack
# registers. Concretely tracing the real dispatch (this lane's own
# byte-level read of block_handler/FUN_1c75d8's cmd_handler_3 body, cross-
# checked by running it) found register plumbing that a purely static
# read of "the call site" (this lane's own task brief) could not resolve
# from FUN_1c75d8 alone: R12 at the CALL 0x1c2b24 site is `I3`, and I3 is
# never assigned anywhere in cmd_handler_3's own body or in the dispatch
# preamble (0x1c7501-0x1c7521) that runs before it -- it is `DM(I6-3)`,
# one of *two* by-reference outputs command_dispatch_fn (0x1c778a) writes
# into block_handler's own stack frame (`R4=&(I6-3)`, `R8=&(I6-2)` passed
# into that CALL at 0x1c74f8); the *other* output, `DM(I6-2)`, is exactly
# the "R1 = DM(I6-2) of the task frame" this lane's brief already named.
# Getting both right by hand (let alone the ring-buffer/`command_word`
# plumbing above them) is exactly the kind of bookkeeping a *real* call
# gets for free -- so `setup_frame()` only pokes the three DM cells that
# select which of the 4 audio-task commands runs (docs/findings/06's
# "Task loop": `command_word`/`command_word_shift_src`/`ring_flag`), and
# `call_frame()` lets the concrete run compute everything else, the same
# way `run_init()` already does for FUN_1c15e3.
#
# Reaching render_frame this way (from a run_init() Runner, one voice
# already set up via setup_voice()) lands on the *same three* stops this
# lane's task brief already knew from a bare `--root 0x1c2b24` run
# (0xb88e4b, 0x1c4969, 0x1c088e) -- strong cross-check that they are real
# properties of an (almost) empty 16-track frame, not artifacts of calling
# render_frame bare. See FRAME_STOPS below and this lane's handover for
# the full stops table (real source found, or reported as a tools/
# sharc_core gap this lane must not fix itself).


def setup_frame(state, image: str, *, command: int = 3, ring_flag: int = 0) -> int:
    """Poke the audio task's command dispatch (docs/findings/06's "Task
    loop") so block_handler's own command-table jump selects `command`
    (3 = "renders from 0x1c7671", the only one that reaches render_frame)
    with ring half `ring_flag`, and return `profile(image).block_handler`
    -- the address `call_frame()` should `fresh_call()` at.

    `command_word_shift_src` (`DM(0x261ca4)`, the shift block_handler folds
    into the command word's address -- "0x1c7578..0x1c7586" in
    docs/findings/06) is left at 0 so the command word's address is
    `command_word` itself, with no per-track offset.

    This does not touch a voice record; call `setup_voice()` (or
    `render_frame()`'s own default) first on the same STATE so render_frame
    finds at least one active voice.
    """
    p = profile(image)
    _poke(state, p.command_word_shift_src, 0)
    _poke(state, p.command_word, command)
    _poke(state, p.ring_flag, ring_flag)
    return p.block_handler


def call_frame(
    runner: sr.Runner,
    image: str,
    *,
    patch_table: sv.PatchTable | None = None,
    max_steps: int = 4_000_000,
) -> tuple[sr.Runner, sr.RunResult]:
    """`Runner.fresh_call()` at `profile(image).block_handler` (see
    `setup_frame()`) on RUNNER's current State (typically a `run_init()`
    Runner with `setup_voice()`/`setup_frame()` already applied to its
    State) -- the real firmware call chain: block_handler ->
    command_dispatch_fn -> cmd_handler_3 -> render_frame -> ... -- instead
    of calling render_frame bare with hand-set registers (see this
    section's own module-level note for why that matters here).

    `patch_table`, if given, is `tools/sharc_survey.py`'s PatchTable
    convention (`{pc: [(kind, target, value), ...]}`), applied every time
    its pc recurs via `sharc_survey.run_with_patches()` -- this lane's own
    hypotheses for the stops an (almost) empty synthetic frame hits belong
    there (see FRAME_PATCH_TABLE), not hardcoded in this function. Without
    one, this is a plain `Runner.run()`: it stops at the first fork/mmr/
    return exactly like any other survey root.

    Returns (the fresh_call Runner, its RunResult) -- the Runner stays
    positioned at the halt (or at the return) for a caller to inspect
    registers/memory afterward (e.g. dumping the master mix buffer).
    """
    p = profile(image)
    new_runner = runner.fresh_call(p.block_handler, diagnose_unknown=True)
    if not patch_table:
        return new_runner, new_runner.run(max_steps=max_steps)
    start_pc_sw = new_runner.state.pc_sw
    t0 = time.perf_counter()
    halt = sv.run_with_patches(new_runner, patch_table, max_steps)
    elapsed = time.perf_counter() - t0
    result = sr.RunResult(
        halt=halt,
        instructions=new_runner.instructions,
        elapsed=elapsed,
        form_counts=new_runner.form_counts,
        start_pc_sw=start_pc_sw,
        final_pc_sw=new_runner.state.pc_sw,
        max_call_depth_reached=new_runner.max_call_depth_reached,
        watch_log=new_runner.watch_log,
    )
    return new_runner, result


def call_frame_collect_all(
    runner: sr.Runner,
    image: str,
    *,
    patch_table: sv.PatchTable | None = None,
    max_steps: int = 4_000_000,
    img: sharcmod.Image | None = None,
) -> tuple[sr.Runner, sv.CollectAllResult]:
    """Like `call_frame()`, but drives the call with
    `tools/sharc_survey.py`'s `run_collect_all()` instead of
    `run_with_patches()`: every stop this frame call passes is recorded
    (category, pc, form, unknowns), not only the first -- the same "collect
    every blocker on one path" behaviour `sharc_survey`'s own CLI gives a
    bare root, applied to a real frame call through `block_handler` instead.
    `patch_table` still resolves a fork the same way (a `"reg"`/`"mem"`
    entry unconditionally, before the instruction executes); a fork with no
    matching entry is continued through with the documented default
    (not-taken) rather than stopping the whole replay, so one frame's own
    report is a full list of what it depended on instead of only the
    nearest one.

    Returns (the fresh_call Runner, positioned at the terminal stop, its
    `sv.CollectAllResult`) -- `result.terminal` is the same kind of Halt
    information `call_frame()`'s own `RunResult.halt` would have stopped
    at first.
    """
    p = profile(image)
    new_runner = runner.fresh_call(p.block_handler, diagnose_unknown=True)
    result = sv.run_collect_all(new_runner, patch_table or {}, max_steps, img=img)
    return new_runner, result


def scan_voice_active(
    state, image: str, *, exclude: Sequence[int] = ()
) -> dict[int, int]:
    """The ACTIVE byte (`FIELD_ACTIVE`, record `+0x1b8`) of every voice
    record except `exclude`, keyed by voice index -- for telling a
    hand-set-up voice (this harness's own `setup_voice()`) apart from one
    the firmware itself activated while a replay ran. Returns only entries
    that read non-zero; an empty dict means no *other* voice was ever
    marked ACTIVE."""
    active = {}
    for voice in range(VOICE_RECORD_COUNT):
        if voice in exclude:
            continue
        record = voice_record_address(image, voice)
        value = _read_byte(state, record + FIELD_ACTIVE)
        if value:
            active[voice] = value
    return active


# This lane's own hypotheses for the stops render_frame hits when called
# (via setup_frame()/call_frame() above) from a run_init() state with one
# voice set up and no other per-track frame data at all (an almost-empty
# 16-track frame) -- each keyed by the pc sharc_survey's diagnose_unknown
# names as the fork trigger, evidence in this module's git history/the
# lane's handover, not re-derived here. Two of these patch a REGISTER
# immediately after the read that should have produced it, rather than the
# DM cell a static read of the instruction suggests, because a direct
# `_dm_write()` at that raw address does not reach whatever the actual
# "(sw)" instruction resolves (confirmed empirically: identical instruction
# count patched vs unpatched) -- a `tools/sharc_core` address-canonicalisation
# question this lane could not resolve within its own scope; report it,
# do not guess further at it here.
FRAME_PATCH_TABLE: sv.PatchTable = {
    # 0xb88e47 -> fork at 0xb88e4b: FUN_b88e04's recips-based fixed-point
    # divide (the same helper the voice record contract's Step formula
    # uses) computes ASTATX.AZ from a comparison whose R1 operand traces
    # (via FUN_1c3570/FUN_1c3594) back to a per-track "(sw)" DM read at
    # 0x1c2fe9 (`R4 = DM(0x2560b8) (sw)`) that reads 0 -- a per-track
    # mixer/gain parameter mirror this lane's synthetic frame never
    # populated for any of the 16 tracks. Patched one instruction later
    # (0x1c2fec) as a register override, not a DM poke at 0x2560b8 itself
    # (see this table's own docstring).
    0x1C2FEC: [("reg", "R4", 0x4000)],
    # 0x1c4965 -> fork at 0x1c4969: FUN_1c4914 (the voice record contract's
    # "Positions" setter helper) computes ASTATX.AN/AZ/AV from
    # `F6=fcomp(F6,F4)` over an fmax/fsub chain of BITEXT-decoded voice
    # record position fields; this lane's synthetic voice (setup_voice(),
    # not a real trigger) leaves fields that chain needs in a combination
    # the real trigger sequence would not. Forces both fcomp operands to a
    # finite 1.0f.
    0x1C4965: [("reg", "R6", 0x3F800000), ("reg", "R4", 0x3F800000)],
}

# A further, DIAGNOSTIC-ONLY hypothesis for the third known stop
# (0x1c088e, FUN_1c0874's saturating fix<->float round-trip helper): not
# in FRAME_PATCH_TABLE because forcing it this way desyncs the run rather
# than fixing it (see this module's own handover). Confirmed root cause
# (not a hypothesis): forms_system.py's `_type_18a` bit-test handler
# (`bop in (4, 5)`) requires `isinstance(source, Const)` to produce a
# concrete BTF -- it never uses `_astatx_known_bit()`-style per-bit
# reasoning against a `PartialConst` source, so once ASTATX carries even
# one unrelated unknown bit (near-universal after any float op; confirmed
# by this lane: the tested bit itself, AI, was concretely known here) a
# `bit-test`/`xor-test` against ASTATX/ASTATY always reports Unknown. This
# is a `tools/sharc_core` gap this lane must not fix itself -- report it
# instead: `_type_18a` (tools/sharc_core/forms_system.py, the `bop in (4,
# 5)` branch) should determine `result` from the PartialConst's own
# mask/bits over exactly `mask`'s bits when `source` is a `PartialConst`,
# falling back to Unknown only when a bit `mask` needs is itself
# unresolved -- the same reasoning `_astatx_known_bit()` already applies
# one bit at a time. Setting this lane's diagnostic patch
# (`{"ASTATX": 0x400}` at 0x1c0885, replacing ASTATX with a plain Const
# that happens to match its own known bits) does get past 0x1c088e, but
# the run then hits a "return-mismatch" category a few hundred thousand
# instructions later (0xb82a08, a repeated call stack entry suggesting an
# IIR/limiter loop) -- discarding ASTATX's other tracked bits this way
# desyncs a later branch, not a step toward the frame returning. Left out
# of FRAME_PATCH_TABLE for that reason.
FRAME_DIAGNOSTIC_ASTATX_PATCH: sv.PatchTable = {0x1C0885: [("reg", "ASTATX", 0x400)]}

# Where one render_frames() call with FRAME_PATCH_TABLE stops today, from
# run_init() state. Tests compare against this one pin: update it (and say
# why in the commit) when a fix moves the stop.
#
# 2026-09-25: forms_move.py's Type3a handler previously refused l=1 (long-
# word) outright ("unsupported Type3a long-word access" at 0x1c2920, 86,740
# instructions). Implementing it (the Type14a-style neighbor-register-pair
# access PRM p.2-4 documents for the (LW) modifier) let the frame run on
# instead of stopping there. The run now reaches a genuine RETURN, not
# another gap: "return without followed call" at 0x1c75d3 -- inside
# FUN_1c74cd (block_handler itself, entry 0x1c74cd, end 0x1c75d8, per
# tools/sharc.py's img.func) -- is how tools/sharc_run.py's
# fresh_call_state()/Runner.fresh_call() (return_address=None, see its own
# docstring) signals that a call started with an empty call_stack has
# unwound all the way back through its own entry point's RTS. Since
# call_frame() enters at block_handler with no real caller pushed, this
# halt fires exactly when block_handler's own return executes -- the frame
# render call chain (block_handler -> command_dispatch_fn -> cmd_handler_3
# -> render_frame -> ...) completed. tools/sharc_widthaudit.py --root frame
# confirms 0 access-width mismatches across all 17,468 load/store events
# checked in this run.
FRAME_MILESTONE = {
    "pc_sw": 0x1C75D3,
    "instructions": 95982,
    "reason": "return without followed call",
}


def render_frames(
    memory,
    image: str,
    *,
    n_frames: int = 1,
    voice: int = 0,
    pitch_step: float = 1.0,
    sample_len: int = 4096,
    samples: Sequence[float] | None = None,
    sample_format: str = "int16",
    command: int = 3,
    ring_flag: int = 0,
    patch_table: sv.PatchTable | None = FRAME_PATCH_TABLE,
    max_steps: int = 4_000_000,
) -> tuple[sr.Runner, list[sr.RunResult]]:
    """`run_init()`, then N `call_frame()` calls against one voice set up
    via `setup_voice()` (VOICE active, the rest silent) -- the frame-level
    analogue of `render_blocks()`. `patch_table` defaults to
    FRAME_PATCH_TABLE (this lane's own hypotheses); pass `None` for a bare
    diagnostic run that stops at the first fork.

    `run_init()` failing (`InitResult.ran` False) raises ValueError: unlike
    render_blocks()'s optional `init`, a frame call has no bare-state
    fallback (setup_frame()'s docstring: render_frame needs post-init
    state; nothing about it was ever designed to run from a blank State).

    Returns (the Runner positioned after the last call_frame(), one
    RunResult per frame) -- inspect `runner.state` afterward (e.g.
    `st._dm_read(runner.state, profile(image).master_mix ...)`, or
    whatever ring the master stage feeds -- docs/findings/06's "Master
    stage": `render_frame` writes the mix at `R4`, i.e. 0x25f180, the same
    address `setup_frame()` never touches) to see what the frame actually
    produced.
    """
    init = run_init(memory, image)
    if not init.ran or init.runner is None:
        raise ValueError("render_frames: run_init failed: %s" % init.error)
    runner = init.runner
    state = runner.state
    if samples is not None:
        _write_samples(state, 0x310000, samples, sample_format)
    setup_voice(
        state,
        image,
        voice,
        sample_len=sample_len,
        pitch_step=pitch_step,
        sample_base=0x310000,
    )
    setup_frame(state, image, command=command, ring_flag=ring_flag)
    results = []
    for _ in range(n_frames):
        runner, result = call_frame(
            runner, image, patch_table=patch_table, max_steps=max_steps
        )
        results.append(result)
    return runner, results


def read_master_mix(memory, image: str, runner: sr.Runner) -> list[float]:
    """The 64 floats at `profile(image).master_mix`'s own input buffer,
    0x25f180 (docs/findings/06's "Master stage": `0x1c207b` gets `R4 =
    0x25f180`; `setup_frame()`/`call_frame()` never write there themselves,
    so any nonzero content came from render_frame's own run) -- for
    checking whether an active voice's signal reached the mix after a
    frame call. 64 words matches the per-block work-buffer size
    (FIELD_WORK_BUFFER) render_frame's own per-track sum accumulates into;
    unlike that field, 0x25f180 is a fixed address, not a voice-record
    offset.
    """
    floats = []
    for i in range(64):
        raw = st._dm_read(runner.state, 0x25F180 + i * 4, 4)
        floats.append(
            struct.unpack("<f", struct.pack("<I", raw.value & 0xFFFFFFFF))[0]
            if raw is not None
            else 0.0
        )
    return floats


def read_ring_a(memory, image: str, runner: sr.Runner, count: int = 64) -> dict:
    """The DAC output ring: `tools/sharc_dac.py` owns the format itself
    (module docstring there has the full static + dynamic evidence for
    "FUN_1c74a1 converts DM(0x25f180)/DM(0x25f200) to Q31, L/R interleaved,
    into ring A half RING_A_BASE + flag*RING_A_HALF_BYTES") -- this
    function is now just the `Runner`-shaped adapter over it, for checking
    whether an active voice's signal reached the DMA-facing output buffer
    after a frame call, one step past `read_master_mix`'s pre-conversion
    float sum.

    `profile(image).ring_flag` resolves the literal DM address
    `setup_frame()` already pokes through (this module's own "Task loop"
    comment); `sharc_dac.ring_half_just_written()` applies the
    post-toggle-flag correction (see its own docstring) that used to be
    hand-rolled here.

    `count` is kept for backward compatibility with callers that pass it,
    but `sharc_dac` always reads a full 64-word (32 stereo sample) half --
    a partial COUNT only truncates the returned lists, it does not change
    how many words are actually read from memory.

    Returns {"flag_after": the current (post-toggle) ring_flag value,
    "half_written": the byte offset of the half the last frame wrote (None
    when `flag_after` itself is None, e.g. a Runner whose loader never
    mapped `ring_flag`), "q31": raw Q31 ints (interleaved L0,R0,L1,R1,...;
    an unmapped word reads as 0, matching `sharc_dac.decode_ring_a_words()`
    -- real SHARC+ SRAM after reset, not "unknown"), "float": the same
    words as -1.0..1.0 floats (Q31's own full-scale convention), plus
    "left"/"right" (32 de-interleaved floats each) and
    "left_raw"/"right_raw" (their Q31 ints) -- `sharc_dac.RingAFrame`'s own
    fields, for a caller that wants one channel without re-deriving the
    interleave.}
    """
    flag_raw = st._dm_read(runner.state, profile(image).ring_flag, 4)
    flag_after = flag_raw.value & 1 if flag_raw is not None else None

    def dm_read(address: int) -> int | None:
        raw = st._dm_read(runner.state, address, 4)
        return None if raw is None else raw.value & 0xFFFFFFFF

    frame = sharc_dac.read_ring_a_pcm(dm_read, flag_after)
    q31 = frame.interleaved_q31()[:count]
    floats = [v / sharc_dac.Q31_FULL_SCALE for v in q31]
    return {
        "flag_after": flag_after,
        "half_written": None if flag_after is None else frame.half_written,
        "q31": q31,
        "float": floats,
        "left": frame.left,
        "right": frame.right,
        "left_raw": frame.left_raw,
        "right_raw": frame.right_raw,
    }


def write_ring_a_wav(
    path: str,
    rings: Sequence[dict],
    *,
    sample_rate: int = 48000,
    stereo: bool = True,
) -> None:
    """Concatenate `read_ring_a()`'s own per-frame return dicts (RINGS, in
    frame order) into one WAV at PATH: stereo (both of ring A's own L/R
    channels, `sharc_dac.write_wav_stereo()`) by default, or a mono L+R
    average (`tools/sharc_harness.py`'s own `write_wav()`, the same
    downmix `render_frames_to_ring_a()` already computes for
    `measure_tone()`) when STEREO is False.

    This is ring A's own WAV writer -- `write_wav()` above stays the
    voice-render (mono-only) one; a caller with several frames' worth of
    `read_ring_a()` dicts (one call per rendered frame) uses this instead
    of hand-concatenating "left"/"right" itself."""
    left: list[float] = []
    right: list[float] = []
    for ring in rings:
        left.extend(ring["left"])
        right.extend(ring["right"])
    if stereo:
        sharc_dac.write_wav_stereo(path, left, right, sample_rate=sample_rate)
    else:
        mono = [
            0.5 * (l_sample + r_sample)
            for l_sample, r_sample in zip(left, right, strict=True)
        ]
        write_wav(path, mono, sample_rate=sample_rate)


# --- Voice -> track -> master mix -> ring A (lane A2, 2026-09-25) ---------
#
# render_frames()/call_frame() above reach a genuine RETURN (FRAME_MILESTONE)
# from a run_init() state with one voice set up, but the master mix
# (0x25f180) and both DAC rings stay zero: docs/findings/06's "the mix
# gate" -- FUN_1c642a's own per-track accumulate block (0x1c6b3c-0x1c6b8a)
# is skipped whenever the "shared context" word DM(0x252d3c) reads 0, which
# it does from any run_init() state (init writes the literal 32 at
# 0x1c1643, then FUN_1cb336 resets it to 0 later in the same init run,
# before FUN_1c15e3 returns -- confirmed by execution, see that function's
# own docstring).
#
# **This lane's own whole-image writer/reader scan (tools/sharc.py's
# writers()/readers(), not just run_init()'s own reach) finds one more
# writer and two more readers of DM(0x252d3c), none reported before:**
#
#   - A writer at 0x1cdc23 (``DM(I3, M6) = R11``, I3 = the passed-in
#     0x252d3c pointer), inside FUN_1cdbb2 (0x1cdbb2-0x1cdcab) -- an
#     IIR/all-pole recursion this lane did not fully characterize. It is
#     reached only dynamically, through FUN_1c642a's own out-of-line
#     dispatch FUN_1c71ec and two nested jump tables (0x8055c840 ->
#     0x8055c858 -> 0x8055c874, whose index 1 selects FUN_1cdbb2 --
#     docs/findings/06's "Table 0x8055c874 picks one stage per slot type"),
#     for whatever per-track "slot type" selects that stage -- which is why
#     ``img.callers()`` reports none (a real indirect-call gap, not
#     evidence this path is dead: see this repo's CLAUDE.md). Every capture
#     this project has (idle and play) reads every track's own machine
#     type/selector fields as 0, so this path was never exercised by
#     execution here, and this lane could not reach it from a synthetic
#     frame either (see the report). Reported as [O], not adopted.
#   - Two readers, previously undocumented: 0xb82696 (FUN_b82680, an
#     "orchestrator/dispatcher" in the same block group as the dynamics
#     stage) and 0xb82d54 (FUN_b82d41, the thin wrapper `0x1c207b` calls
#     as `0xb82d41 -> 0xb82cba`, docs/findings/06's "dynamics [D]" stage --
#     confirmed by disassembly: ``R0 = DM(I4, M5)`` at 0xb82d54, I4 = the
#     caller's R12 = the same 0x252d3c pointer, pushed as one of
#     FUN_b82cba's own stack arguments). So this word is not read only by
#     FUN_1c642a's accumulate gate: the dynamics/compressor stage consumes
#     it too. This lane's own experiment (see the report) is consistent
#     with FUN_b82cba's envelope math going degenerate when this input is
#     0 -- a hand-set-nonzero gate reached a NEW, unresolved fork
#     (FUN_b8037f, blk69) instead of restoring multi-frame signal, so this
#     is reported as a plausible explanation, not a fix.
#
# **This lane's own execution trace also corrects one specific claim in
# docs/findings/06's own "[V]" description of the accumulate loop.** A
# concrete, register-level single-step of 0x1c6b3c-0x1c6b8a (patched so
# DM(0x252d3c) reads 1, avoiding the skip) shows R8 -- the operand
# docs/findings/06 calls "per voice/track index R8" -- staying the SAME
# constant (0) for all 32 hardware DO-loop iterations; nothing in the loop
# body ever increments it. The per-iteration axis is instead I4, which
# walks a 32-entry workspace *pointer table* at 0x24ef2c (matching
# docs/findings/06's own "I5 = DM(I6-4) + 0xdc64 = 0x24ef2c" workspace
# confirmation), and the writes this lane observed through it land in
# per-voice SDRAM addresses (e.g. 0x8045b3c0 for voice 0 -- in the same
# 0x8045.... range as the documented voice sample pointer), not in
# 0x252df8 + t*0x100 (the master-stage per-track buffer). This needs a
# second agent's check before correcting docs/findings/06 -- it may mean
# the loop is walking *voices*, writing each one's own decimator-state
# scratch (record +0x104's target), rather than the master-stage per-track
# sum this lane initially assumed from the doc's own framing; reported
# here, not adopted as a doc correction, since this lane's own DM(0x252d3c)
# value (1) may not be the real one the doc's own confirmatory run used.
#
# **Given the above, this lane could not establish a real runtime path
# that sets DM(0x252d3c) to a value that both (a) avoids a new fork and
# (b) makes FUN_1c642a's own accumulate loop deposit a voice's decimated
# output at 0x252df8 + t*0x100.** Per this lane's own task brief ("record
# any remaining hand-set value in FRAME_PATCH_TABLE / a documented setup
# function with its evidence"), the functions below instead inject a
# voice's own ALREADY-FIRMWARE-RENDERED decimated buffer (FIELD_WORK_BUFFER,
# populated every frame by the firmware's own FUN_1c4ecf/FUN_1c4f81 call
# inside FUN_1c642a's per-voice dispatch loop -- this part needs no hand-
# set value at all) directly into one track's master-mix input, bypassing
# only the not-yet-resolved accumulate stage -- confirming, by execution,
# that the DOWNSTREAM path (track buffer -> master stage 0x1c207b -> ring
# A) is intact and unconditional once that one buffer has real content.
#
# --- Continuity across frames, root-caused (lane B2, 2026-09-25) ----------
#
# The frame-0-only problem above is NOT the "one independent post-init
# render per frame" workaround's own two original blockers (`state.trace`'s
# unbounded growth and the Type14a odd-UREG-pair stop at 0x1c32ad): both are
# fixed on this branch (7e5e8c4), and a genuinely continuous Runner (one
# `run_init()`, repeated `Runner.fresh_call()` at block_handler, checked by
# watchpoint on this lane's own multi-frame runs) now completes any number
# of frames cleanly. Ring A still goes silent from frame 1 onward anyway --
# root cause, traced with watchpoints across two frames of one continuous
# Runner (see the report):
#
# `FUN_1c207b`'s own per-track summation loop (`0x1c2353`, `DO ... UNTIL
# LCE` over all 16 tracks' `0x252df8`-range input) multiplies every track's
# contribution by three scalar coefficients held in R1/R2/R12 for the whole
# loop. Those scalars are computed a few hundred instructions earlier
# (`0x1c22d7`-`0x1c2353`) from `FUN_b82d41`/`FUN_b82cba`'s dynamics/
# compressor stage output (`0x254978`/`0x2549f8`, fed by `DM(0x252d3c)` --
# docs/findings/06's still-open mix gate) and from a ~26-field master-bus
# parameter block this SAME function decodes fresh every frame at
# `0x1c2e00`-`0x1c2fb0` (source table `0x255fb6`-`0x2560d0`; confirmed by
# execution to read all zero from a synthetic `run_init()` state -- no real
# kit/mixer configuration is ever loaded here). Frame 0's own compressor
# output is a plausible near-unity gain/trim (R1=0.953, R2=0.984,
# R12=-0.029 as IEEE floats); by frame 1 it has collapsed to a degenerate
# all-zero-multiplier state, independent of the injected signal's own
# amplitude (checked: scaling the injected tone down 10x does not change
# this). **This is real firmware behaviour given this lane's own state (a
# synthetic frame with a blank master-bus parameter table and no real
# per-track "slot type" ever reaching `FUN_1cdbb2`, docs/findings/06's mix
# gate), not an emulator bug**: with a genuinely unconfigured mix bus, the
# compressor's own steady state is silence, and frame 0 is riding on one
# frame's worth of "not yet converged" grace period, not a working master
# bus.
#
# **Hand-set value (documented hypothesis, not a discovered real gate).**
# Fully deriving the real per-frame scalars needs the master-bus parameter
# table's own bit layout and `FUN_b82cba`'s own 5-stage recursive state --
# out of this lane's own budget (see the report's "Open" section). Pinning
# R1/R2/R12 to frame 0's own real, execution-confirmed values at every
# recurrence of `pc=0x1c2353` -- the same "register-only override at one
# pc" technique docs/findings/06 already recommends for this exact
# situation, via the same `PatchTable` mechanism `FRAME_PATCH_TABLE` already
# uses -- keeps the compressor's own downstream multiply-accumulate at its
# frame-0 output indefinitely. This is a static, pc-keyed patch applied
# automatically every time this pc recurs (`sharc_survey.apply_patches()`),
# not a per-frame reset performed by this module's own Python loop: the
# cause (a value this lane could not derive) is what gets a fixed
# assumption, not the Runner's state. Known limitation: this freezes the
# compressor's own gain at whatever it computed for frame 0's specific
# transient, so it does not adapt to a different injected level the way a
# real compressor would -- checked and accepted for this lane's own
# single-tone deliverable, not a general fix.
CONTINUOUS_MIX_SCALAR_PC = 0x1C2353
CONTINUOUS_MIX_SCALAR_PATCH: sv.PatchTable = {
    CONTINUOUS_MIX_SCALAR_PC: [
        ("reg", "R1", 0x3F740000),  # ~0.953 -- frame 0's own computed value
        ("reg", "R2", 0x3F7C0000),  # ~0.984
        ("reg", "R12", 0xBCEE0000),  # ~-0.029
    ],
}

# docs/findings/06's "Master stage": `0x1c207b` sums 16 tracks at
# `0x252df8 + t*0x100` bytes each (32 L floats then 32 R floats, plain
# IEEE float32 words -- the same format `read_master_mix()` already reads
# back at `0x25f180`).
TRACK_MIX_BASE = 0x252DF8
TRACK_MIX_STRIDE = 0x100
TRACK_MIX_CHANNEL_WORDS = 32

# The call site `CALL 0x1c207b` inside `FUN_1c2b24` (render_frame): disasm
# `0x1c3099 CALL (linked, delayed) target=0x1c207b`, immediately after
# `CALL 0x1c642a` (`0x1c3083`, the per-voice/accumulate stage) and
# `CALL 0x1c18a6` (`0x1c3090`), and *before* `CALL 0x1c14e7` (`0x1c30a0`,
# the end-of-frame zero-clear of the same 0x252df8 range -- confirmed by
# execution to run *after* this point, so a poke placed here survives into
# this frame's own master-stage read and is only cleared in preparation
# for the *next* frame, the same way a real accumulate would be). Not a
# resolved symbol in tools/sharc_symbols.py's Profile (this call site, not
# a function entry), so kept as a literal here like the voice record
# offsets above.
MASTER_STAGE_CALL_PC = 0x1C3099


def track_mix_address(track: int, *, channel: str = "L") -> int:
    """The DM address of TRACK's own 32-word L or R half of the master
    stage's per-track input buffer (see TRACK_MIX_BASE's docstring)."""
    if channel not in ("L", "R"):
        raise ValueError("channel must be 'L' or 'R', got %r" % (channel,))
    base = TRACK_MIX_BASE + track * TRACK_MIX_STRIDE
    return base if channel == "L" else base + TRACK_MIX_CHANNEL_WORDS * 4


def _float_to_word(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def inject_track_buffer(state, record: int, *, track: int = 0) -> list[float]:
    """Read RECORD's own work buffer (FIELD_WORK_BUFFER, 64 floats -- this
    frame's real interpolated render, already produced by the firmware's
    own per-voice dispatch call, whether or not FUN_1c642a's own
    accumulate stage ever sums it into a track buffer -- see this
    section's own module note), decimate it exactly the way `decimate()`
    does, and write the 32-sample result as float32 words into both the L
    and R halves of TRACK's master-stage input buffer
    (`track_mix_address()`). Mono only (both channels get the same
    samples) -- there is no evidence yet for a real per-channel pan value
    to set instead. Returns the 32 decimated floats written, for a caller
    to inspect or assemble into a WAV alongside ring A."""
    work = []
    for i in range(64):
        raw = st._dm_read(state, record + FIELD_WORK_BUFFER + i * 4, 4)
        work.append(
            struct.unpack("<f", struct.pack("<I", raw.value & 0xFFFFFFFF))[0]
            if raw is not None
            else 0.0
        )
    decimated = decimate(work)
    for i, value in enumerate(decimated):
        word = _float_to_word(value)
        _poke(state, track_mix_address(track, channel="L") + i * 4, word)
        _poke(state, track_mix_address(track, channel="R") + i * 4, word)
    return decimated


def call_frame_with_track_injection(
    runner: sr.Runner,
    image: str,
    record: int,
    *,
    track: int = 0,
    patch_table: sv.PatchTable | None = FRAME_PATCH_TABLE,
    max_steps: int = 4_000_000,
) -> tuple[sr.Runner, sr.RunResult, list[float]]:
    """Like `call_frame()`, but calls `inject_track_buffer()` the instant
    `MASTER_STAGE_CALL_PC` is about to execute -- see this section's own
    module note for why (the accumulate stage's real enable value is not
    established). Steps one instruction at a time (not
    `sharc_survey.run_with_patches()`/`run_collect_all()`) applying
    `patch_table` the same way `call_frame()` does; a fork this halts on is
    a genuinely new, unresolved gap (report it, do not add a guess here),
    not something to drive through with a default branch pick, since this
    function's whole point is to observe the master stage's own,
    unperturbed behaviour once it has real per-track input.

    Returns `(new_runner, RunResult, injected)` where `injected` is
    `inject_track_buffer()`'s own 32-float return, or `[]` if
    `MASTER_STAGE_CALL_PC` was never reached (an earlier stop)."""
    p = profile(image)
    new_runner = runner.fresh_call(p.block_handler, diagnose_unknown=True)
    start_pc_sw = new_runner.state.pc_sw
    injected: list[float] = []
    steps = 0
    halt: sr.Halt | None = None
    t0 = time.perf_counter()
    while steps < max_steps:
        if new_runner.state.pc_sw == MASTER_STAGE_CALL_PC:
            injected = inject_track_buffer(new_runner.state, record, track=track)
        if patch_table:
            sv.apply_patches(new_runner, patch_table)
        try:
            new_runner.step()
        except sr.Halt as exc:
            halt = exc
            break
        steps += 1
    else:
        halt = sr.Halt("max-steps", new_runner.state.pc_sw)
    result = sr.RunResult(
        halt=halt,
        instructions=new_runner.instructions,
        elapsed=time.perf_counter() - t0,
        form_counts=new_runner.form_counts,
        start_pc_sw=start_pc_sw,
        final_pc_sw=new_runner.state.pc_sw,
        max_call_depth_reached=new_runner.max_call_depth_reached,
        watch_log=new_runner.watch_log,
    )
    return new_runner, result, injected


def _merge_patch_tables(*tables: sv.PatchTable | None) -> sv.PatchTable:
    """Union TABLES' entries by pc, concatenating entry lists for a pc that
    appears in more than one (later tables' entries applied after earlier
    ones, at the same pc, by `sharc_survey.apply_patches()`'s own "list of
    entries" convention -- see its docstring). `None` entries are skipped,
    so a caller can pass an optional table straight through."""
    merged: sv.PatchTable = {}
    for table in tables:
        if not table:
            continue
        for pc, entries in table.items():
            merged.setdefault(pc, []).extend(entries)
    return merged


def render_frames_to_ring_a(
    memory,
    image: str,
    *,
    n_frames: int = 32,
    voice: int = 0,
    freq: float = 1000.0,
    pitch_step: float = 1.0,
    track: int = 0,
    sample_format: str = "int16",
    patch_table: sv.PatchTable | None = FRAME_PATCH_TABLE,
    mix_scalar_patch: sv.PatchTable | None = CONTINUOUS_MIX_SCALAR_PATCH,
    max_steps: int = 4_000_000,
) -> dict:
    """Render N frames of one voice's own `freq` Hz sine on ONE continuous
    Runner (one `run_init()`, N `Runner.fresh_call()`s at block_handler via
    `call_frame_with_track_injection()`), injecting each frame's real
    decimated output into `track`'s master-mix input (see
    `inject_track_buffer()`), and return ring A's own output -- the actual
    DAC-facing buffer, one stage past the master mix -- concatenated across
    frames as mono (L+R averaged) floats at the firmware's own decimated
    output rate (`SOURCE_SAMPLE_RATE / 2` = 48 kHz).

    **One continuous Runner (lane B2, 2026-09-25; was N independent
    post-init renders under lane A2).** That version's own two blockers
    (`sharc_run.State.trace`'s unbounded growth across repeated calls on
    one Runner, and an "unsupported Type14a odd UREG pair" stop at
    0x1c32ad on the second call onward) are both fixed on this branch
    (7e5e8c4) -- checked here by running many frames on one Runner with no
    new stop and bounded memory. The frame-0-only silence this module's own
    "Continuity across frames, root-caused" note (above `TRACK_MIX_BASE`)
    documents was a THIRD, separate problem (the master stage's own
    per-track gain scalars collapsing to zero by frame 1, given this lane's
    blank synthetic mix configuration) -- fixed here via `mix_scalar_patch`
    (`CONTINUOUS_MIX_SCALAR_PATCH` by default; see that constant's own
    docstring for the full evidence and its documented-hypothesis status).

    **Phase continuity comes from one long source buffer plus the voice
    record's own carried state**, both true to how the firmware actually
    plays a long sample: `setup_voice()`/`_write_samples()` run ONCE,
    before the frame loop, over a buffer sized for all `n_frames` (unlike
    the old per-frame-independent version, which needed the source tone
    itself to carry phase since every frame's record restarted at
    `start=0`).

    Returns `{"ring_a_mono": [...], "per_frame": [...], "any_new_stop":
    bool}`; `per_frame[i]` is `{"frame", "instructions", "halt",
    "injected_max_abs", "ring_a_nonzero"}`.
    """
    init = run_init(memory, image)
    if not init.ran or init.runner is None:
        raise ValueError("render_frames_to_ring_a: run_init failed: %s" % init.error)

    sample_len = _default_sample_len(n_frames, pitch_step)
    sample_base = 0x310000
    combined_patches = _merge_patch_tables(patch_table, mix_scalar_patch)

    runner = init.runner
    state = runner.state
    tone = [
        math.sin(2 * math.pi * (freq / SOURCE_SAMPLE_RATE) * i)
        for i in range(sample_len)
    ]
    _write_samples(state, sample_base, tone, sample_format)
    record = setup_voice(
        state,
        image,
        voice,
        sample_len=sample_len,
        pitch_step=pitch_step,
        sample_base=sample_base,
    )
    setup_frame(state, image, command=3, ring_flag=0)

    ring_a_mono: list[float] = []
    per_frame: list[dict] = []
    for frame_index in range(n_frames):
        runner, result, injected = call_frame_with_track_injection(
            runner,
            image,
            record,
            track=track,
            patch_table=combined_patches,
            max_steps=max_steps,
        )
        ring = read_ring_a(memory, image, runner)
        floats = ring["float"]
        mono = [
            0.5 * ((left or 0.0) + (right or 0.0))
            for left, right in zip(floats[0::2], floats[1::2], strict=True)
        ]
        ring_a_mono.extend(mono)
        per_frame.append(
            {
                "frame": frame_index,
                "instructions": result.instructions,
                "halt": result.halt.reason,
                "injected_max_abs": max((abs(v) for v in injected), default=0.0),
                "ring_a_nonzero": sum(1 for v in mono if v),
            }
        )
    return {
        "ring_a_mono": ring_a_mono,
        "per_frame": per_frame,
        "any_new_stop": any(
            f["halt"] != "return without followed call" for f in per_frame
        ),
    }


def measure_tone(
    samples: Sequence[float], freq_hz: float, sample_rate: int, *, discard: int = 0
) -> dict:
    """This module's own direct-correlation tone measurement (the same
    technique `tests/test_sharc_harness.py`'s `CliFrequencyTest` uses):
    correlating SAMPLES against cos/sin at exactly `freq_hz` reconstructs
    that frequency's power with no spectral leakage regardless of whether
    the window holds a whole number of cycles (unlike an FFT bin readout),
    so this works on a short, non-coherently-sampled render.

    Returns `{"freq_hz", "power_ratio" (fundamental / total power, 1.0 =
    pure tone), "rms_dbfs", "peak_dbfs"}` -- both dBFS figures are relative
    to `1.0` (this module's own float/Q31 full-scale convention), `-inf`
    reported as `None` for a silent input."""
    values = list(samples[discard:])
    n = len(values)
    if n == 0:
        return {
            "freq_hz": freq_hz,
            "power_ratio": 0.0,
            "rms_dbfs": None,
            "peak_dbfs": None,
        }
    mean = sum(values) / n
    total_power = sum((v - mean) ** 2 for v in values) / n
    a_cos = sum(
        values[i] * math.cos(2 * math.pi * freq_hz * i / sample_rate) for i in range(n)
    ) * (2.0 / n)
    a_sin = sum(
        values[i] * math.sin(2 * math.pi * freq_hz * i / sample_rate) for i in range(n)
    ) * (2.0 / n)
    amplitude = math.hypot(a_cos, a_sin)
    fund_power = amplitude * amplitude / 2.0
    rms = math.sqrt(sum(v * v for v in values) / n)
    peak = max((abs(v) for v in values), default=0.0)
    return {
        "freq_hz": freq_hz,
        "power_ratio": fund_power / total_power if total_power else 0.0,
        "rms_dbfs": 20 * math.log10(rms) if rms > 0 else None,
        "peak_dbfs": 20 * math.log10(peak) if peak > 0 else None,
    }


def flatten(blocks: Sequence[Sequence[float]]) -> list[float]:
    return [sample for block in blocks for sample in block]


def decimate(work_buffer: Sequence[float]) -> list[float]:
    """The 64-float interpolator work buffer, decimated 2:1 by pairwise
    average (docs/findings/06: "0xb80000 reads 64, writes 32, x0.5") --
    the 32-sample block this module's WAV output actually is."""
    return [
        0.5 * (work_buffer[2 * i] + work_buffer[2 * i + 1])
        for i in range(len(work_buffer) // 2)
    ]


def write_wav(path: str, samples: Sequence[float], sample_rate: int = 48000) -> None:
    with wave.open(path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        clipped = (max(-1.0, min(1.0, s)) for s in samples)
        wav.writeframes(
            b"".join(struct.pack("<h", int(round(s * 32767))) for s in clipped)
        )


# --- reference 6-tap polyphase interpolator + 2:1 decimator ----------------


def read_coeff_table(memory, image: str, phases: int = 256) -> list[list[float]]:
    """The firmware's own polyphase coefficients (docs/findings/06's
    "Render and declick": 6 taps per phase, one per 4-byte slot, 24
    bytes/phase), as float taps (signed 32-bit fraction, value / 2**31 --
    see this lane's own finding below, correcting the previous "int16 low
    half" and "int16 high half" guesses).

    ``phases=256`` is this lane's own finding, correcting the previous
    128-phase assumption (docs/findings/06 marks the phase count "[O]";
    128 was a guess "consistent with the 128-entry tables found elsewhere
    in the render chain", not evidence about this table specifically).
    Two independent checks: (1) a concrete, register-level single-step of
    FUN_1c4f81's DO 64 loop reads the coefficient pointer I5 right after
    sw 0x1c50b6's ``modify(I5, M2)`` and finds ``(I5 - coeff_table) / 24``
    (the row this function would read) landing exactly on
    ``int(fractional_phase * 256) % 256`` for every iteration checked
    (multiple non-power-of-2 pitch_step values) -- not the "% 128" a
    128-phase table would need, and not merely double by coincidence: rows
    computed this way went as high as 230, past any 128-row table
    entirely. (2) independent of any addressing question at all, the raw
    bytes at phase index 128+k equal phase index (128-k)'s six taps in
    *reverse* (e.g. row 128 == reversed(row 120), row 127 == reversed(row
    121)) -- the textbook mirror-image second half of a linear-phase
    polyphase filter bank, and not a pattern a wrong stride into unrelated
    data would produce by chance.

    **Each 4-byte slot is one 32-bit Q31 tap, not a 16-bit tap in either
    half (this lane's own finding, 2026-09-25, correcting an intermediate
    "upper 16 bits" guess made while chasing the same evidence).** Both
    16-bit halves of every slot independently satisfy check (2)'s mirror
    property (a 32-bit-symmetric table is trivially symmetric in each half
    separately, and so is the full 32-bit value), so that check alone
    cannot tell a 16-bit-slice reading from the true 32-bit one -- it is
    necessary but not sufficient. What actually distinguishes them: a
    register-level trace of FUN_1c4f81's DO 64 loop's MAC chain
    (``tools/sharc_core/compute_mult.py``'s ``_mr_product_raw``/
    ``_mr_extract32``, already correct and unchanged by this fix) computing
    ``(sample_raw * coeff_slot_raw32 << 1) >> 32`` and comparing that to
    the emulator's own concrete SAT-MRF result for a one-tap-nonzero
    impulse case: using the coefficient slot's *full* 32-bit raw value
    (e.g. ``0x31f09d09`` = 837852425 for phase 0/tap 2) as the multiplier
    input reproduces the emulator's exact SAT-MRF integer (6392, from
    sample_raw=16384) bit-for-bit; using only the slot's upper 16 bits
    (12784, an earlier version of this finding) does not feed the real
    multiplier at all -- SHARC's ``Rn = MR (+-) Rx*Ry`` reads Rx/Ry as
    whole 32-bit registers, and ``R0 = DM(I5+2), long`` (sw 0x1c50c4) loads
    the *whole* slot into R0, not a truncated half. Treating that raw
    32-bit slot as a signed Q31 fraction (``/2**31``) then gives a *float*
    coefficient that, multiplied by the sample as a plain float product
    (this module's ``reference_render``), reproduces the fixed-point
    result to about 2e-5 -- consistent with the one real remaining gap:
    ``_mr_extract32``'s bits-63:32 extraction is a PRM-documented *floor*,
    not a round (p.3-10: "using bits 63-32 for a fractional result"), so a
    pure-float reference is architecturally unable to match it bit-exactly
    without reimplementing the same truncation. The earlier "upper 16
    bits" reading was closer to that *already-truncated* single-tap
    result by coincidence (discarding the coefficient's own low 16 bits
    happens to land near where the hardware's own floor lands for some
    but not all sample/coefficient magnitudes) but is off by 4-8x more
    than the Q31 reading once several taps sum together (this lane's own
    ``check_correctness()`` run: max error dropped from ~1.1e-4 to ~3e-5,
    the SAT floor's own ~1-ULP-of-Q15 floor this module's tests assert
    once this fix landed). See
    ``/private/tmp/claude-501/-Users-em-src-digi-digitakt2/89c5e6c8-2a58-496e-84a0-527f96e6f55a/scratchpad/verify_fixedpoint.py``
    and ``verify_q31.py`` for the traces this is drawn from. This was the
    residual this lane's task brief pointed at ("this lane's evidence
    points at tools/sharc_core's MR-accumulator housekeeping ... and/or
    float_by ... not at this harness's own sample encoding or
    addressing") -- neither of those was actually at fault (both verified
    correct by direct trace); this function's own coefficient extraction
    was.
    """
    from sharcldr import SW_ALIAS_BASE

    base = profile(image).coeff_table
    table = []
    for phase in range(phases):
        taps = []
        for tap in range(COEFF_TABLE_TAPS):
            addr = SW_ALIAS_BASE + base + phase * COEFF_TABLE_STRIDE_BYTES + tap * 4
            raw = memory.read(addr, 4)
            value = 0 if raw is None else int.from_bytes(raw, "little", signed=True)
            taps.append(value / 2147483648.0)
        table.append(taps)
    return table


def reference_render(
    input_samples: Sequence[float],
    coeff_table: Sequence[Sequence[float]],
    step: float,
    n_output: int,
    *,
    start_phase: float = 0.0,
) -> list[float]:
    """A 6-tap polyphase interpolator (input_samples, indexed by a
    fractional position advancing ``step`` samples per interpolated
    output, phase selecting a row of ``coeff_table``) followed by 2:1
    decimation (pairwise average -- docs/findings/06's "0xb80000 reads 64,
    writes 32, x0.5"), computed independently of tools/sharc_core so it is
    a real cross-check rather than the same code re-run.

    **Forward taps, not centered (this lane's own finding, 2026-09-25,
    correcting an earlier ``base - half + 1 + t`` centered-window guess).**
    An impulse-response test against a concrete run of FUN_1c4f81's DO 64
    loop (single nonzero raw sample, unity pitch_step, six placements of
    the impulse relative to ``base`` swept one at a time) shows tap ``t``
    (0-indexed in the same order the loop reads them: the DO loop's first
    sample read pairs with ``coeff_table[phase][0]``, its last with
    ``coeff_table[phase][5]``) always lands on ``input_samples[base + t]``
    -- i.e. the filter is purely causal/forward from ``base``, never
    reaching behind it. Concretely: with the impulse at sample index K and
    ``base`` (the loop's own integer position, sw 0x1c50a4-0x1c50af's
    ``I4 = I0; modify(I4, M3)`` with M3 = floor(position)) swept from K-5
    to K, the nonzero output at each step reproduced
    ``coeff_table[0][K - base]`` (times the sample's own Q15 value) to
    within fixed-point rounding, for every one of the 6 tap positions --
    see ``read_coeff_table()``'s own docstring for the same trace this
    reused to fix the coefficient byte offset. See
    ``/private/tmp/claude-501/-Users-em-src-digi-digitakt2/89c5e6c8-2a58-496e-84a0-527f96e6f55a/scratchpad/trace_impulse2.py``.
    """
    phases = len(coeff_table)
    taps = len(coeff_table[0]) if coeff_table else 6

    def sample_at(index: int) -> float:
        if 0 <= index < len(input_samples):
            return input_samples[index]
        return 0.0

    interpolated = []
    pos = start_phase
    for _ in range(n_output * 2):
        base = int(math.floor(pos))
        frac = pos - base
        phase = int(frac * phases) % phases
        acc = 0.0
        for t in range(taps):
            acc += coeff_table[phase][t] * sample_at(base + t)
        interpolated.append(acc)
        pos += step
    return [
        0.5 * (interpolated[2 * i] + interpolated[2 * i + 1]) for i in range(n_output)
    ]


def _make_test_input(
    kind: str, length: int, *, freq_cycles_per_sample: float = 1.0 / 32
) -> list[float]:
    """A synthetic, bounded-amplitude input for ``check_correctness``.
    ``kind`` is "sine" (the same generator the previous version of this
    module used) or "ramp" (a periodic sawtooth at the same period, in
    [-1, 1)) -- a discontinuous waveform the 6-tap interpolator's frequency
    response treats very differently from a sine, so agreement on both is
    a much stronger cross-check than either alone."""
    if kind == "sine":
        return [
            math.sin(2 * math.pi * freq_cycles_per_sample * i) for i in range(length)
        ]
    if kind == "ramp":
        period = max(2, round(1.0 / freq_cycles_per_sample))
        return [2.0 * ((i % period) / period) - 1.0 for i in range(length)]
    raise ValueError("unknown check_correctness input kind: %r" % (kind,))


# (input kind, pitch_step) cases check_correctness() runs by default: unity
# speed, one downward and one upward pitch step (both exact powers of 2, so
# every phase lands on exactly 0 -- coeff row 0 only), one non-power-of-2
# step (0.75, added this lane, 2026-09-25 -- exercises every coeff row a
# quarter-integer step touches, not just row 0) on the sine, and unity on
# the ramp (a discontinuous waveform the 6-tap filter treats very
# differently from a sine).
_DEFAULT_CASES: tuple[tuple[str, float], ...] = (
    ("sine", 1.0),
    ("sine", 0.5),
    ("sine", 2.0),
    ("sine", 0.75),
    ("ramp", 1.0),
)


def _read_record_words(state, record: int) -> list[int]:
    """The full 0x1D8-byte voice record as 32-bit words (VOICE_RECORD_STRIDE
    is word-aligned: 0x1D8 / 4 = 118 exactly), for ``check_correctness()``'s
    per-block before/after diagnostic."""
    words = []
    for offset in range(0, VOICE_RECORD_STRIDE, 4):
        raw = st._dm_read(state, record + offset, 4)
        words.append(raw.value & 0xFFFFFFFF if raw is not None else 0)
    return words


def _diff_record_words(
    before: Sequence[int], after: Sequence[int]
) -> list[dict[str, int]]:
    """Word offsets (bytes, from the record base) where AFTER differs from
    BEFORE -- word, not byte, granularity (coarser than the named FIELD_*
    constants), but enough to see which part of the record one block's
    render actually touched without guessing which field it must have
    been."""
    return [
        {"offset": i * 4, "before": b, "after": a}
        for i, (b, a) in enumerate(zip(before, after, strict=True))
        if b != a
    ]


def check_correctness(
    memory,
    image: str,
    *,
    n_blocks: int = 8,
    cases: Sequence[tuple[str, float]] = _DEFAULT_CASES,
    freq_cycles_per_sample: float = 1.0 / 32,
    sample_format: str = "int16",
    init: InitResult | None = None,
) -> dict[str, dict]:
    """For each (input kind, pitch_step) in ``cases``, render ``n_blocks``
    consecutive 32-sample blocks from one voice and compare each against
    ``reference_render()`` run over the same input with its own phase
    accumulator carried forward the same way the firmware's FIELD_PHASE
    record field persists across ``call_render()`` calls -- so this checks
    both per-block numerical accuracy and block-to-block continuity (phase
    advancing by pitch_step*64 per block, matching the DO 64 loop's 64
    raw interpolation points; ACTIVE staying 1 with real sample data and a
    non-null FIELD_SAMPLE_PTR).

    ``init``, if given (an ``InitResult`` from ``run_init()``), starts every
    case from an independent copy of init's own post-return state (see
    ``new_runner``) instead of a bare one -- each case gets its own copy,
    so one is never affected by another's writes.

    Returns one dict per case, keyed "<kind>_step<pitch_step>", including a
    "record_diffs" list (one entry per block: the word offsets that block's
    render changed in the record, via ``_diff_record_words``) for seeing
    where a field such as ACTIVE actually gets written without re-deriving
    it from the ``active``/``phase`` summaries alone.

    **Sizing the buffer past deactivation, not just past the tap window.**
    FUN_1c4f81's past-limit test (this lane's task brief's static facts;
    FIELD_SAMPLE_LENGTH's own docstring) clears ACTIVE up to about one
    whole block *early*: empirically (this lane's own bisection, four
    pitch_step values), the render needs roughly ``pitch_step * 64 + 4``
    samples of buffer *past* the last one a block actually reads, not the
    fixed +32 tap-window pad alone -- a fixed pad sized for the slowest
    case (0.5x) silently deactivated the last of 8 blocks at 1x and 2x
    (this module's own test asserts against that regression). This sizes
    for one whole extra block at this case's own pitch_step, which covers
    the measured minimum with room to spare, plus the same tap-window pad
    as before.
    """
    sample_base = 0x310000
    coeff_table = None  # loaded once cases start, after profile() resolves
    results: dict[str, dict] = {}
    for kind, pitch_step in cases:
        sample_len = int(math.ceil((n_blocks + 1) * 64 * pitch_step)) + 32
        samples = _make_test_input(
            kind, sample_len, freq_cycles_per_sample=freq_cycles_per_sample
        )
        if coeff_table is None:
            coeff_table = read_coeff_table(memory, image)
        # reference_render() must compare against the same quantized values
        # the emulated render actually reads back (see
        # _dequantize_sample()'s docstring), not the ideal float input --
        # otherwise sample_format="int16"'s own Q15 rounding would show up
        # as "error" indistinguishable from a real interpolation mismatch.
        reference_samples = [_dequantize_sample(s, sample_format) for s in samples]

        runner = new_runner(memory, image, init=init)
        state = runner.state
        _write_samples(state, sample_base, samples, sample_format)
        record = setup_voice(
            state,
            image,
            0,
            sample_len=sample_len,
            pitch_step=pitch_step,
            sample_base=sample_base,
        )

        pos = 0.0
        halts, per_block_max, per_block_mean = [], [], []
        phases, actives = [], []
        record_diffs: list[list[dict[str, int]]] = []
        flat_errors: list[float] = []
        for _ in range(n_blocks):
            before_words = _read_record_words(state, record)
            result, rendered = call_render(runner, image, record)
            after_words = _read_record_words(state, record)
            record_diffs.append(_diff_record_words(before_words, after_words))
            decimated = decimate(rendered)
            reference = reference_render(
                reference_samples,
                coeff_table,
                step=pitch_step,
                n_output=len(decimated),
                start_phase=pos,
            )
            pos += pitch_step * 2 * len(decimated)  # = pitch_step * 64

            errors = [abs(a - b) for a, b in zip(decimated, reference, strict=True)]
            flat_errors.extend(errors)
            per_block_max.append(max(errors) if errors else None)
            per_block_mean.append(sum(errors) / len(errors) if errors else None)
            halts.append(result.halt.reason)
            phases.append(_read_q31_pair(state, record + FIELD_PHASE) / Q31)
            actives.append(_read_byte(state, record + FIELD_ACTIVE))

        phase_deltas = [b - a for a, b in itertools.pairwise(phases)]
        results["%s_step%g" % (kind, pitch_step)] = {
            "kind": kind,
            "pitch_step": pitch_step,
            "n_blocks": n_blocks,
            "halts": halts,
            "max_abs_error": max(flat_errors) if flat_errors else None,
            "mean_abs_error": sum(flat_errors) / len(flat_errors)
            if flat_errors
            else None,
            "per_block_max_error": per_block_max,
            "per_block_mean_error": per_block_mean,
            "phase": phases,
            "phase_deltas": phase_deltas,
            "phase_delta_expected": pitch_step * 64,
            "active": actives,
            "record_diffs": record_diffs,
        }
    return results


# Task 1's --sample-len auto-sizing, for main()'s CLI only. This is the same
# formula check_correctness() computes inline for each of its own cases
# (see that function's "Sizing the buffer past deactivation" docstring:
# FUN_1c4f81's past-limit test clears ACTIVE about one whole block *early*,
# empirically needing ~pitch_step*64+4 samples of buffer past the last one
# a block actually reads, not just the +32 tap-window pad) -- duplicated
# here rather than factored into a shared helper check_correctness() itself
# calls, so this lane's task 1 CLI change does not touch check_correctness()
# (kept unchanged per this lane's own task brief).
def _default_sample_len(n_blocks: int, pitch_step: float) -> int:
    """Auto-size a --blocks/--pitch-step render's source buffer with one
    whole block's look-ahead margin, so a long render does not run off the
    end of its own synthetic sample (task 1)."""
    return int(math.ceil((n_blocks + 1) * 64 * pitch_step)) + 32


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("image", help='sharc.py image name, e.g. "dt2-1.16"')
    p.add_argument("--voice", type=int, default=0)
    p.add_argument("--blocks", type=int, default=8)
    p.add_argument(
        "--pitch-step",
        type=float,
        default=1.0,
        help="input samples per output (pre-interpolation) step; 1.0 plays "
        "the source at its own rate (SOURCE_SAMPLE_RATE), 2.0 at double "
        "speed/pitch, 0.5 at half",
    )
    p.add_argument(
        "--sample-len",
        type=int,
        default=None,
        help="source buffer length, in samples; default: auto-sized from "
        "--blocks/--pitch-step (--frame: 4096) so a long render does not "
        "run off the end of its own synthetic sample (see "
        "_default_sample_len())",
    )
    p.add_argument(
        "--freq",
        type=float,
        default=1000.0,
        help="the source sample's own frequency, in Hz, at "
        "SOURCE_SAMPLE_RATE (the firmware's sample-playback rate, "
        "96 kHz -- see that constant's docstring); playing it at "
        "--pitch-step produces an audible output frequency of "
        "freq * pitch_step Hz (Hz is invariant under the render's own 2:1 "
        "decimation -- see main()'s own comment below for the derivation)",
    )
    p.add_argument("--out", default=None, help="WAV output path")
    p.add_argument(
        "--sample-rate",
        type=int,
        default=48000,
        help="the WAV file's declared playback rate; should match the "
        "render's actual decimated output rate (SOURCE_SAMPLE_RATE / 2 = "
        "48000) for --freq * --pitch-step to sound right on playback",
    )
    p.add_argument("--run-init", action="store_true", default=False)
    p.add_argument("--check-correctness", action="store_true", default=False)
    p.add_argument(
        "--frame",
        action="store_true",
        default=False,
        help="call render_frame (FUN_1c2b24) via block_handler instead of "
        "one voice's render (see render_frames()/call_frame())",
    )
    p.add_argument(
        "--frames", type=int, default=1, help="--frame only: number of frame calls"
    )
    p.add_argument(
        "--no-frame-patches",
        dest="frame_patches",
        action="store_false",
        default=True,
        help="--frame only: run without FRAME_PATCH_TABLE (stop at the first fork)",
    )
    p.add_argument(
        "--ring-a",
        action="store_true",
        default=False,
        help="--frame only: render --frames frames of one voice, injecting "
        "each frame's real decimated output into --track's master-mix "
        "input (see inject_track_buffer()'s own module note for why this "
        "is needed, and render_frames_to_ring_a() for the technique), and "
        "write ring A (the DAC-facing buffer) to --out as a WAV instead of "
        "reporting the master mix",
    )
    p.add_argument(
        "--track",
        type=int,
        default=0,
        help="--ring-a only: which of the 16 master-mix track inputs to "
        "inject the voice's rendered output into",
    )
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)

    memory = load_image_memory(a.image)

    if a.frame and a.ring_a:
        result = render_frames_to_ring_a(
            memory,
            a.image,
            n_frames=a.frames,
            voice=a.voice,
            freq=a.freq,
            pitch_step=a.pitch_step,
            track=a.track,
            patch_table=FRAME_PATCH_TABLE if a.frame_patches else None,
        )
        sample_rate = int(SOURCE_SAMPLE_RATE // 2)
        output_frequency_hz = a.freq * a.pitch_step
        # Discard the first frame's worth of decimated samples: this
        # lane's own experiments (see the report) found the master stage's
        # own gain/declick ramp attenuates the very first frame or two
        # before a steady level settles, so measuring frequency/level from
        # frame 0 alone would be measuring the ramp, not the tone.
        measurement = measure_tone(
            result["ring_a_mono"], output_frequency_hz, sample_rate, discard=32
        )
        if a.out:
            write_wav(a.out, result["ring_a_mono"], sample_rate=sample_rate)
        report = {
            "image": a.image,
            "voice": a.voice,
            "track": a.track,
            "frames": a.frames,
            "freq_hz": a.freq,
            "output_frequency_hz": output_frequency_hz,
            "any_new_stop": result["any_new_stop"],
            "per_frame": result["per_frame"],
            "measurement": measurement,
            "out": a.out,
        }
        if a.json:
            print(json.dumps(report, indent=2, default=str))
        else:
            print(
                "image: %s  voice: %d  track: %d  frames: %d  output: %.2f Hz"
                % (a.image, a.voice, a.track, a.frames, output_frequency_hz)
            )
            print("any_new_stop:", result["any_new_stop"])
            print(
                "measured: freq=%.2f Hz power_ratio=%.4f rms_dbfs=%s peak_dbfs=%s"
                % (
                    measurement["freq_hz"],
                    measurement["power_ratio"],
                    "%.2f" % measurement["rms_dbfs"]
                    if measurement["rms_dbfs"] is not None
                    else "-inf",
                    "%.2f" % measurement["peak_dbfs"]
                    if measurement["peak_dbfs"] is not None
                    else "-inf",
                )
            )
            if a.out:
                print("wrote", a.out)
        return 0

    if a.frame:
        frame_sample_len = a.sample_len if a.sample_len is not None else 4096
        runner, results = render_frames(
            memory,
            a.image,
            n_frames=a.frames,
            voice=a.voice,
            pitch_step=a.pitch_step,
            sample_len=frame_sample_len,
            # The source is a --freq Hz tone at SOURCE_SAMPLE_RATE (task 1;
            # see that constant's docstring and this function's own comment
            # below, before the non-frame render path) -- not pre-scaled by
            # pitch_step, which the DO-64 interpolation loop itself already
            # applies by advancing pitch_step raw samples per output step.
            samples=[
                math.sin(2 * math.pi * (a.freq / SOURCE_SAMPLE_RATE) * i)
                for i in range(frame_sample_len)
            ],
            patch_table=FRAME_PATCH_TABLE if a.frame_patches else None,
        )
        report = {
            "image": a.image,
            "voice": a.voice,
            "frames": a.frames,
            "halts": [r.halt.reason for r in results],
            "instructions": [r.instructions for r in results],
            "master_mix": read_master_mix(memory, a.image, runner),
        }
        if a.json:
            print(json.dumps(report, indent=2, default=str))
        else:
            print("image: %s  voice: %d  frames: %d" % (a.image, a.voice, a.frames))
            for i, r in enumerate(results):
                print(
                    "frame %d: halt=%s pc=%#x instructions=%d"
                    % (i, r.halt.reason, r.halt.pc_sw, r.instructions)
                )
            mix = report["master_mix"]
            print(
                "master_mix[0:8]=%s max_abs=%.4g"
                % (
                    ["%.4g" % v for v in mix[:8]],
                    max((abs(v) for v in mix), default=0.0),
                )
            )
        return 0

    init: InitResult | None = None
    if a.run_init:
        init = run_init(memory, a.image)
    # Only feed a *successful* init into the render/correctness paths --
    # new_runner(..., init=...) raises on an init that did not complete, and
    # a failed --run-init should still fall back to reporting rather than
    # crashing the whole invocation. init.ran/init.error are still reported
    # below either way.
    active_init = init if (init is not None and init.ran) else None

    # Task 1's CLI semantics fix (2026-09-25), correcting this harness's own
    # earlier bug: the source PCM this harness writes to sample_base is a
    # --freq Hz tone at SOURCE_SAMPLE_RATE (the firmware's own
    # sample-playback rate -- see that constant's docstring), not
    # pre-scaled by pitch_step. The DO-64 interpolation loop itself already
    # applies pitch_step (it advances that many raw samples per output
    # step; see setup_voice()'s FIELD_STEP), so folding pitch_step into the
    # source too (the previous version's ``(freq/sample_rate)*pitch_step``)
    # double-counted it. Measured before this fix, on the WAV output: with
    # --sample-rate left at its 48000 default, audible frequency was
    # ``2 * freq * pitch_step**2`` (the extra factor of 2 from the WAV
    # being written at 48000 while the pre-scaled source's own
    # cycles-per-sample was computed against that same 48000, one octave
    # below the 96 kHz the DO-64 domain and its 2:1 decimator actually
    # assume -- see decimate()'s docstring). Corrected: the output's
    # audible frequency is ``freq * pitch_step`` Hz, full stop -- Hz is
    # invariant under the render's own 2:1 decimation to
    # SOURCE_SAMPLE_RATE/2 (48 kHz, matching write_wav()'s own 48000
    # default), so it does not depend on --sample-rate at all; --sample-rate
    # only needs to match that 48 kHz decimated rate for the WAV to *play
    # back* at the frequency its own header claims (this lane verified the
    # corrected formula empirically: tests/test_sharc_harness.py's
    # CliFrequencyTest zero-crossing-counts a rendered WAV against
    # freq * pitch_step for several (freq, pitch_step) pairs).
    output_frequency_hz = a.freq * a.pitch_step
    samples = [
        math.sin(2 * math.pi * (a.freq / SOURCE_SAMPLE_RATE) * i)
        for i in range(
            a.sample_len
            if a.sample_len is not None
            else _default_sample_len(a.blocks, a.pitch_step)
        )
    ]
    sample_len = len(samples)

    start = time.perf_counter()
    blocks, results = render_blocks(
        memory,
        a.image,
        a.voice,
        a.blocks,
        pitch_step=a.pitch_step,
        sample_len=sample_len,
        samples=samples,
        init=active_init,
    )
    elapsed = time.perf_counter() - start
    total_instructions = sum(r.instructions for r in results)

    if a.out:
        write_wav(
            a.out,
            flatten([decimate(block) for block in blocks]),
            sample_rate=a.sample_rate,
        )

    correctness = (
        check_correctness(memory, a.image, init=active_init)
        if a.check_correctness
        else None
    )

    report = {
        "image": a.image,
        "voice": a.voice,
        "blocks": a.blocks,
        "pitch_step": a.pitch_step,
        "freq_hz": a.freq,
        "output_frequency_hz": output_frequency_hz,
        "sample_len": sample_len,
        "instructions_total": total_instructions,
        "instructions_per_block": total_instructions / a.blocks if a.blocks else None,
        "elapsed_s": elapsed,
        "instructions_per_second": total_instructions / elapsed
        if elapsed > 0
        else None,
        "halts": [r.halt.reason for r in results],
        "out": a.out,
        "init": None if init is None else {"ran": init.ran, "error": init.error},
        "correctness": correctness,
    }
    if a.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print("image: %s  voice: %d  blocks: %d" % (a.image, a.voice, a.blocks))
        print(
            "source: %.2f Hz @ %.0f Hz (sample_len=%d)  pitch_step=%g  "
            "output: %.2f Hz"
            % (
                a.freq,
                SOURCE_SAMPLE_RATE,
                sample_len,
                a.pitch_step,
                output_frequency_hz,
            )
        )
        if init is not None:
            print("init:", "ok" if init.ran else "failed: %s" % init.error)
        print("instructions/block: %.1f" % report["instructions_per_block"])
        print(
            "elapsed: %.3fs (%.0f instr/s)"
            % (elapsed, report["instructions_per_second"] or 0.0)
        )
        print("halts:", report["halts"])
        if a.out:
            print("wrote", a.out)
        if correctness is not None:
            for name, case in correctness.items():
                print(
                    "correctness[%s]: max_abs_error=%.4g mean_abs_error=%.4g "
                    "phase_deltas=%s (expected %.1f) active=%s"
                    % (
                        name,
                        case["max_abs_error"],
                        case["mean_abs_error"],
                        ["%.2f" % d for d in case["phase_deltas"]],
                        case["phase_delta_expected"],
                        case["active"],
                    )
                )
                print(
                    "  record diffs (changed word offsets per block): %s"
                    % [
                        [hex(d["offset"]) for d in diffs]
                        for diffs in case["record_diffs"]
                    ]
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
