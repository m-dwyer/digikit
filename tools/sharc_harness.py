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
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import sharc as sharcmod  # noqa: E402
import sharc_run as sr  # noqa: E402
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

Q31 = 1 << 31


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


def run_init(memory, image: str, *, max_steps: int = 200_000) -> InitResult:
    """Attempt FUN_1c15e3 (docs/findings/06: no arguments) to its return.

    Best-effort: on the 1.16 image it runs 1,320 instructions and stops at
    sw 0x1c128a, a Type9a jump with a control modifier the tracer does not
    model yet. A caller does not have to
    treat this failing as fatal: every voice-record field
    ``setup_voice()`` writes is set directly from docs/findings/06's own
    contract, not read back from what FUN_1c15e3 would have written.
    """
    runner = _make_runner(memory, image, profile(image).init, {"I6": 0x300000})
    try:
        result = runner.run(max_steps=max_steps)
    except Exception as exc:  # pragma: no cover - defensive, see docstring
        return InitResult(False, None, str(exc))
    ok = result.halt.reason in ("return without followed call", "max-steps")
    return InitResult(
        ok,
        result,
        None
        if ok
        else "%s at %#x (%s)"
        % (result.halt.reason, result.halt.pc_sw, result.halt.form),
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


def _read_byte(state, address: int) -> int:
    raw = st._dm_read(state, address, 1)
    return raw.value & 0xFF if raw is not None else 0


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

    Fields this harness does not have a documented source for (fade-in,
    zero-cross mute, reseed, reverse/loop, seed-pending) are left at their
    reset-zero value via ``State.explicit_memory_model`` rather than
    guessed at.
    """
    record = voice_record_address(image, voice)
    end = sample_len if end is None else end
    _poke(state, record + FIELD_SAMPLE_PTR, sample_base)
    _poke(state, record + FIELD_ACTIVE, 1, width=1)
    _write_q31_pair(state, record + FIELD_LOOP_START, loop_start << 31)
    _write_q31_pair(state, record + FIELD_START, start << 31)
    _write_q31_pair(state, record + FIELD_END, end << 31)
    _write_q31_pair(state, record + FIELD_STEP, round(pitch_step * Q31))
    _write_q31_pair(state, record + FIELD_PHASE, start << 31)
    _poke(state, record + FIELD_PREVIOUS_SAMPLE, 0)
    return record


def new_runner(
    memory: LoadedMemory, image: str, *, stack_base: int = 0x300000
) -> sr.Runner:
    """A Runner parked at profile(image).voice_render, ready for repeated
    calls through ``call_render`` -- one State/overlay is shared across
    those calls (a Runner's own State is otherwise created once and never
    reset; see ``call_render``'s docstring for why a *fresh* Runner per
    call, tried first, does not work: each Runner's State starts from an
    empty overlay, so a second Runner never sees the first's
    ``setup_voice()`` pokes)."""
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
    state.uregs[UREG_CODES["R4"]] = st.Const(record)
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
) -> tuple[list[list[float]], list[sr.RunResult]]:
    """N call_render() calls against one freshly set-up voice record,
    sharing a single Runner/State (see ``new_runner``/``call_render``) so
    the record's own persistent fields (phase, previous sample, ...) carry
    over between calls the way they would across real render-loop
    invocations -- see ``check_correctness()`` for this measured against an
    independent reference, including phase continuity across blocks.

    ``samples``, if given, is written to ``sample_base`` before the first
    call (FIELD_SAMPLE_PTR then points at it), e.g. a synthetic sine for an
    audible WAV. If omitted, the record still points at ``sample_base`` but
    nothing is written there, so unwritten DM reads as 0
    (State.explicit_memory_model): a silent (but not zero-fill-gated, since
    FIELD_SAMPLE_PTR is still non-null) render, useful for an
    instructions/block measurement alone.
    """
    runner = new_runner(memory, image)
    state = runner.state
    if samples is not None:
        for i, value in enumerate(samples):
            bits = struct.unpack("<I", struct.pack("<f", value))[0]
            _poke(state, sample_base + i * 4, bits)
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


def read_coeff_table(memory, image: str, phases: int = 128) -> list[list[float]]:
    """The firmware's own polyphase coefficients (docs/findings/06's
    "Render and declick": 6 (swse) int16 taps per phase, one per 4-byte
    slot, 24 bytes/phase), as float taps (sign-extended int16 / 2**15).
    ``phases`` is not independently confirmed by docs/findings/06 (marked
    [O] there); 128 is this harness's assumption, consistent with the
    128-entry tables found elsewhere in the render chain.
    """
    from sharcldr import SW_ALIAS_BASE

    base = profile(image).coeff_table
    table = []
    for phase in range(phases):
        taps = []
        for tap in range(COEFF_TABLE_TAPS):
            addr = SW_ALIAS_BASE + base + phase * COEFF_TABLE_STRIDE_BYTES + tap * 4
            raw = memory.read(addr, 2)
            value = 0 if raw is None else int.from_bytes(raw, "little", signed=True)
            taps.append(value / 32768.0)
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
    """
    phases = len(coeff_table)
    taps = len(coeff_table[0]) if coeff_table else 6
    half = taps // 2

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
            acc += coeff_table[phase][t] * sample_at(base - half + 1 + t)
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
# speed, one downward and one upward pitch step, on both waveforms.
_DEFAULT_CASES: tuple[tuple[str, float], ...] = (
    ("sine", 1.0),
    ("sine", 0.5),
    ("sine", 2.0),
    ("ramp", 1.0),
)


def check_correctness(
    memory,
    image: str,
    *,
    n_blocks: int = 8,
    cases: Sequence[tuple[str, float]] = _DEFAULT_CASES,
    freq_cycles_per_sample: float = 1.0 / 32,
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

    Returns one dict per case, keyed "<kind>_step<pitch_step>".
    """
    sample_base = 0x310000
    coeff_table = None  # loaded once cases start, after profile() resolves
    results: dict[str, dict] = {}
    for kind, pitch_step in cases:
        # Enough input for n_blocks*64 raw interpolation points at this
        # pitch_step, plus the interpolator's 6-tap window margin either
        # side (reference_render's sample_at() zero-pads out of range, but
        # keeping real signal there makes the comparison meaningful for
        # taps that reach past a block's own span).
        sample_len = int(math.ceil(n_blocks * 64 * pitch_step)) + 32
        samples = _make_test_input(
            kind, sample_len, freq_cycles_per_sample=freq_cycles_per_sample
        )
        if coeff_table is None:
            coeff_table = read_coeff_table(memory, image)

        runner = new_runner(memory, image)
        state = runner.state
        for i, value in enumerate(samples):
            bits = struct.unpack("<I", struct.pack("<f", value))[0]
            _poke(state, sample_base + i * 4, bits)
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
        flat_errors: list[float] = []
        for _ in range(n_blocks):
            result, rendered = call_render(runner, image, record)
            decimated = decimate(rendered)
            reference = reference_render(
                samples,
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
        }
    return results


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("image", help='sharc.py image name, e.g. "dt2-1.16"')
    p.add_argument("--voice", type=int, default=0)
    p.add_argument("--blocks", type=int, default=8)
    p.add_argument("--pitch-step", type=float, default=1.0)
    p.add_argument("--sample-len", type=int, default=4096)
    p.add_argument("--freq", type=float, default=1000.0, help="WAV sine frequency, Hz")
    p.add_argument("--out", default=None, help="WAV output path")
    p.add_argument("--sample-rate", type=int, default=48000)
    p.add_argument("--run-init", action="store_true", default=False)
    p.add_argument("--check-correctness", action="store_true", default=False)
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)

    memory = load_image_memory(a.image)

    init: InitResult | None = None
    if a.run_init:
        init = run_init(memory, a.image)

    # The output WAV's audible frequency is freq/sample_rate cycles per
    # *decimated* output sample; the raw PCM this harness writes to
    # sample_base is consumed at pitch_step raw samples per output, so its
    # own cycles-per-sample must be scaled up by pitch_step to land on the
    # same audible frequency after resampling.
    cycles_per_sample = (a.freq / a.sample_rate) * a.pitch_step
    samples = [
        math.sin(2 * math.pi * cycles_per_sample * i) for i in range(a.sample_len)
    ]

    start = time.perf_counter()
    blocks, results = render_blocks(
        memory,
        a.image,
        a.voice,
        a.blocks,
        pitch_step=a.pitch_step,
        sample_len=a.sample_len,
        samples=samples,
    )
    elapsed = time.perf_counter() - start
    total_instructions = sum(r.instructions for r in results)

    if a.out:
        write_wav(
            a.out,
            flatten([decimate(block) for block in blocks]),
            sample_rate=a.sample_rate,
        )

    correctness = check_correctness(memory, a.image) if a.check_correctness else None

    report = {
        "image": a.image,
        "voice": a.voice,
        "blocks": a.blocks,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
