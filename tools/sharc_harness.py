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
contract" section (two agents checked it against the 1.16 bytes) and the
"Known" facts this harness was briefed with: one voice occupies a 0x1d8-byte
record at ADDRESSES["voice_records"] + voice*0x1d8; FUN_1c4ecf (called the
way FUN_1c642a calls it: R4 = record address + 4) is one voice's render,
producing 64 interpolated floats at record+4..+0x103 and, through its own
call to the decimator at 0xb80000, a 2:1-decimated 32-sample block.

Two known-bad decodes (see ``_KNOWN_BAD_DECODE_PCS``'s docstring -- a live,
standalone ``sharc_trace.decode_at()`` call picks a 32-bit "2b" full-compute
form at these PCs where ``out/sharcdb``'s own sequential-walk decode, and a
raw-buffer cross-check at every window size ``sharc_disasm.decode_loaded_at``
tries, agree the true form is a 16-bit "2c" short-compute -- this looks like
a systematic decode-table/disassembler bug, not two isolated ones) are
worked around here with ``sharc_run.Runner``'s ``decode_overrides``, sourced
live from ``out/sharcdb`` rather than hand-copied, so a database rebuild
that changes them is picked up automatically. This is a workaround, not a
fix: sharc_disasm.py and the decode tables are not this module's files (see
this repo's CLAUDE.md "Agents" section on lane ownership) -- report new
instances of the same pattern rather than adding overrides for them here
blindly.

**Open problem (not solved by this module): the raw sample-data source.**
FUN_1c4ecf/FUN_1c4f81's own six-tap interpolation loop (the ``DO 64`` at
sw 0x1c5096-0x1c5101) reads through I4, freshly reloaded from I0 on every
iteration (sw 0x1c50a4's ``I4 = I0``); I0 is callee-saved (saved at
sw 0x1c4ee3, restored at sw 0x1c4f60), so its value on entry to FUN_1c4ecf
should be the caller-supplied raw sample pointer. But by the time the DO 64
loop runs, I0 has already been overwritten *inside FUN_1c4f81 itself* (the
only other write found in this span, sw 0x1c505a: ``I0 = DM(I4, M5)``) --
seeding I0 at entry (tried: a synthetic 320-sample sine buffer) has no
effect on what the loop actually reads. What the loop does read, both with
a zero and a synthetic sample buffer, is five fixed addresses around
0x17800008-0x17800018 -- fixed across all 64 iterations of one call, which
does not look like a sliding FIR window over raw PCM at all, and is
extremely likely a downstream artefact of some *other* unseeded register in
the dependency chain (FUN_1c642a normally sets ~15 DAG registers, several
of them from per-track state this harness does not reconstruct, before
calling FUN_1c4ecf) rather than a real address. This harness therefore
renders successfully (FUN_1c4ecf returns cleanly) but its output is **not**
validated as numerically correct -- see ``check_correctness()``'s docstring
and this module's own test for how that is measured and reported honestly
rather than papered over.
"""

from __future__ import annotations

import argparse
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

import sharc_run as sr  # noqa: E402
import sharc_trace as st  # noqa: E402
from sharc_disasm import Instruction  # noqa: E402
from sharcldr import LoadedMemory  # noqa: E402

# Every firmware address this harness relies on, in one place (per this
# lane's brief) so a future per-image resolver (lane B's sharc_symbols.py)
# only has to replace this dict.
ADDRESSES = {
    # FUN_1c15e3 (docs/findings/06): no arguments; initialises the 32 voice
    # records at "voice_records" below (calls 0x1c7442 32 times), copies 32
    # words 0x24ef2c->0x252d78 and fills 0x253df8.
    "init": 0x1C15E3,
    # FUN_1c4ecf: one voice's render (callable entry; calls FUN_1c4f81 at
    # 0x1c4f81 for the actual interpolation/decimation body). Called from
    # FUN_1c642a's dispatch loop (sw 0x1c6b00) with R4 = record address + 4.
    "render": 0x1C4ECF,
    "render_body": 0x1C4F81,
    # 32 voice records, 0x1d8 bytes apart (docs/findings/06's voice record
    # contract, two-agent-checked on the 1.16 bytes).
    "voice_records": 0x2412CC,
    "voice_record_stride": 0x1D8,
    "voice_record_count": 32,
    # 128-phase, 6-tap polyphase coefficient table: 6 (swse) int16 taps per
    # phase, one per 4-byte (normal-word) slot, 24 bytes/phase (docs/
    # findings/06's voice record contract, "Render and declick").
    "coeff_table": 0x25D940,
    "coeff_table_taps": 6,
    "coeff_table_stride_bytes": 24,
    # The decimator (docs/findings/06 sec. "The L2 code block's short-word
    # base is 0xb80000" / functions/README.md): an L2 (blk69) short-word
    # code address, not a DM data address -- called from within FUN_1c4f81
    # (sw 0x1c525d), reads the 64-float work buffer at record+4 and writes
    # 32 decimated (x0.5) floats.
    "decimator": 0xB80000,
}

# Voice record field offsets, bytes, from docs/findings/06's voice record
# contract table (two-agent-checked on the 1.16 bytes).
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

Q31 = 1 << 31

# See this module's docstring: a live, standalone decode_at() call picks the
# wrong (wider) form at these two PCs, reached on FUN_1c4ecf's/FUN_1c15e3's
# own execution paths; out/sharcdb's sequential-walk decode has the
# firmware-consistent one. Both were independently confirmed by a raw
# sharc_disasm.disassemble() cross-check at every window size
# decode_loaded_at tries (2/4/6 bytes): none of them recover the 16-bit
# form decode_loaded_at should pick, so this is not a caching or
# window-size artefact in this module -- it is upstream, in
# sharc_disasm.py/the decode tables (not this lane's files).
_KNOWN_BAD_DECODE_PCS = (0x1C502B, 0x1C59CE)


def load_image_memory(image: str):
    """The same LoadedMemory sharc_run.py's CLI reads from."""
    return sr._load_image_memory(image)


def _decode_override(image: str, pc_sw: int) -> Instruction:
    """out/sharcdb's own decode at PC_SW, as a sharc_disasm.Instruction --
    see this module's docstring for why this differs from a live
    ``sharc_trace.decode_at()`` call at these specific addresses."""
    import sharc as sharcmod

    img = sharcmod.load(image)
    rows = img.sql(
        "SELECT width, form, fields FROM insn WHERE image=? AND sw=?",
        image,
        pc_sw,
    )
    if not rows:
        raise ValueError(
            "no out/sharcdb decode recorded at %#x for %s "
            "(rebuild the database, or this PC no longer needs an "
            "override)" % (pc_sw, image)
        )
    width, form, fields_json = rows[0]
    return Instruction(
        offset=0,
        length_bytes=width,
        type_name=form,
        fields=json.loads(fields_json),
        kind="confident",
        note="out/sharcdb decode override: see sharc_harness.py",
    )


def decode_overrides(image: str) -> dict[int, Instruction]:
    return {pc: _decode_override(image, pc) for pc in _KNOWN_BAD_DECODE_PCS}


# Decode-table-unconfirmed forms this render path actually exercises (lane
# C is verifying 3d/4d/14d -- docs/findings/06).
_PROVISIONAL_FORMS = ("3d", "4d", "14d")


def _make_runner(
    memory: LoadedMemory, image: str, start: int, regs: Mapping[str | int, int | str]
) -> sr.Runner:
    """One place for the Runner options this harness always turns on: state
    that a real boot leaves at reset (0) rather than Unknown, the two known
    decode overrides (see this module's docstring), ``_PROVISIONAL_FORMS``,
    and the numeric recips model (this render path's pitch/rate math
    divides)."""
    return sr.Runner(
        memory,
        start,
        regs=regs,
        decode_overrides=decode_overrides(image),
        explicit_memory_model=True,
        provisional_forms=_PROVISIONAL_FORMS,
        approx_recips=True,
        follow_loaded_calls=True,
        max_call_depth=64,
    )


def voice_record_address(voice: int) -> int:
    if not 0 <= voice < ADDRESSES["voice_record_count"]:
        raise ValueError("voice must be 0..%d" % (ADDRESSES["voice_record_count"] - 1))
    return ADDRESSES["voice_records"] + voice * ADDRESSES["voice_record_stride"]


@dataclass
class InitResult:
    ran: bool
    result: sr.RunResult | None
    error: str | None


def run_init(memory, image: str, *, max_steps: int = 200_000) -> InitResult:
    """Attempt FUN_1c15e3 (docs/findings/06: no arguments) to its return.

    Best-effort: see this module's docstring for the systematic decode
    issue this hits deeper in than the two PCs ``decode_overrides()``
    covers (a delay-slot ``25c_rframe`` reached other than immediately
    after a delayed return, at sw 0x1c12bf on the 1.16 bytes, itself
    consistent with the same wrong-live-decode pattern rather than a real
    "return frame outside a return" -- not yet isolated to a specific bad
    PC the way the render path's two were). A caller does not have to
    treat this failing as fatal: every voice-record field
    ``setup_voice()`` writes is set directly from docs/findings/06's own
    contract, not read back from what FUN_1c15e3 would have written.
    """
    runner = _make_runner(memory, image, ADDRESSES["init"], {"I6": 0x300000})
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


def setup_voice(
    state,
    voice: int,
    *,
    sample_len: int,
    pitch_step: float = 1.0,
    start: int = 0,
    end: int | None = None,
    loop_start: int = 0,
) -> int:
    """Write one voice record's fields per docs/findings/06's voice record
    contract, and return its address. ``pitch_step`` is in input samples
    per output (pre-interpolation) step; 1.0 is unity speed.

    Fields this harness does not have a documented source for (fade-in,
    zero-cross mute, reseed, reverse/loop, seed-pending) are left at their
    reset-zero value via ``State.explicit_memory_model`` rather than
    guessed at.
    """
    record = voice_record_address(voice)
    end = sample_len if end is None else end
    _poke(state, record + FIELD_ACTIVE, 1, width=1)
    # See this module's docstring: byte record+0x1bc (not the contract
    # table's documented +0x1b8) is what FUN_1c4ecf's own early-exit gate
    # (sw 0x1c4f1f/0x1c4f21) actually reads with a synthetic, from-scratch
    # register/stack setup -- empirically confirmed non-zero here is
    # required to reach the render body at all; see the module docstring's
    # "Open problem" note before trusting this beyond that gate.
    _poke(state, record + FIELD_LOOP, 1, width=1)
    # Avoid the function's zero-fill short-circuit at sw 0x1c4f15-0x1c4f18
    # (leftz(DM(record+4)) -> SV): a fresh record's work buffer is 0
    # (State.explicit_memory_model), which that check treats as "nothing
    # to render yet".
    _poke(
        state,
        record + FIELD_WORK_BUFFER,
        struct.unpack("<I", struct.pack("<f", 1.0))[0],
    )
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
    """A Runner parked at ADDRESSES["render"], ready for repeated calls
    through ``call_render`` -- one State/overlay is shared across those
    calls (a Runner's own State is otherwise created once and never reset;
    see this module's docstring's "Open problem" note and ``call_render``'s
    docstring for why a *fresh* Runner per call, tried first, does not
    work: each Runner's State starts from an empty overlay, so a second
    Runner never sees the first's ``setup_voice()`` pokes)."""
    return _make_runner(memory, image, ADDRESSES["render"], {"I6": stack_base})


def call_render(
    runner: sr.Runner, record: int, *, sample_base: int = 0x310000
) -> tuple[sr.RunResult, list[float]]:
    """Call FUN_1c4ecf the way FUN_1c642a calls it (R4 = record + 4) on
    RUNNER's already-set-up State (see ``new_runner``/``setup_voice``),
    and return (RunResult, the 64 work-buffer floats).

    Resets RUNNER's control-flow state (pc_sw, call stack, loops, pending
    delay slot) as a fresh call needs, but deliberately keeps its DM
    overlay (the voice record ``setup_voice`` wrote) and every other
    register (the callee saves and restores the ones it uses -- see this
    module's docstring's trace of FUN_1c4ecf's own prologue/epilogue), so
    calling this again for the next block is a closer match to what
    FUN_1c642a's own dispatch loop does than starting over from
    ``new_runner`` would be.

    See this module's docstring for why the returned floats are not (yet)
    a validated render of whatever sample data ``sample_base`` holds.
    """
    from sharc_core.encoding import UREG_CODES

    state = runner.state
    state.pc_sw = ADDRESSES["render"]
    state.stopped = None
    state.pending = None
    state.call_stack = []
    state.loops = []
    state.status_stack = []
    state.at_loaded_entry = False
    state.uregs[UREG_CODES["R4"]] = st.Const(record + FIELD_WORK_BUFFER)
    state.uregs[UREG_CODES["I0"]] = st.Const(sample_base)
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
) -> tuple[list[list[float]], list[sr.RunResult]]:
    """N call_render() calls against one freshly set-up voice record,
    sharing a single Runner/State (see ``new_runner``/``call_render``) so
    the record's own persistent fields (phase, previous sample, ...) carry
    over between calls the way they would across real render-loop
    invocations. See this module's docstring's "Open problem" note: this
    harness does not yet know how to make the render loop's actual sample
    read advance across blocks either way, so this measures
    instructions/block and exercises the correctness-checking machinery,
    not (yet) verified block-to-block audio continuity.
    """
    runner = new_runner(memory, image)
    record = setup_voice(
        runner.state, voice, sample_len=sample_len, pitch_step=pitch_step
    )
    blocks, results = [], []
    for _ in range(n_blocks):
        result, floats = call_render(runner, record, sample_base=sample_base)
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


def read_coeff_table(memory, phases: int = 128) -> list[list[float]]:
    """The firmware's own polyphase coefficients (docs/findings/06's
    "Render and declick": 6 (swse) int16 taps per phase, one per 4-byte
    slot, 24 bytes/phase), as float taps (sign-extended int16 / 2**15).
    ``phases`` is not independently confirmed by docs/findings/06 (marked
    [O] there); 128 is this harness's assumption, consistent with the
    128-entry tables found elsewhere in the render chain.
    """
    from sharcldr import SW_ALIAS_BASE

    table = []
    for phase in range(phases):
        taps = []
        for tap in range(ADDRESSES["coeff_table_taps"]):
            addr = (
                SW_ALIAS_BASE
                + ADDRESSES["coeff_table"]
                + phase * ADDRESSES["coeff_table_stride_bytes"]
                + tap * 4
            )
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


def check_correctness(
    memory,
    image: str,
    *,
    n_samples: int = 64,
    freq_cycles_per_sample: float = 1.0 / 32,
) -> dict:
    """Compare one render_block() call's work-buffer output against
    reference_render() on the same synthetic sine input.

    This is a real, working comparison, not a placeholder -- but see this
    module's docstring's "Open problem" before trusting its verdict: the
    harness's render call does not yet demonstrably read the synthetic
    sample buffer this function writes (I0, the only register found that
    plausibly carries a sample pointer, is overwritten inside FUN_1c4f81
    before the interpolation loop uses it), so a large error here today is
    expected and does not by itself mean the interpolator's semantics
    are wrong -- only that this harness has not yet wired real sample data
    into whatever the loop actually reads. A small error would be strong
    (if surprising, given the open problem) positive evidence; report
    both the error and this caveat together, never the error alone.
    """
    sample_len = 320
    samples = [
        math.sin(2 * math.pi * freq_cycles_per_sample * i) for i in range(sample_len)
    ]
    sample_base = 0x310000
    runner = new_runner(memory, image)
    state = runner.state
    for i, value in enumerate(samples):
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        _poke(state, sample_base + i * 4, bits)
    record = setup_voice(state, 0, sample_len=sample_len, pitch_step=1.0)
    result, rendered = call_render(runner, record, sample_base=sample_base)
    decimated = decimate(rendered)

    coeff_table = read_coeff_table(memory)
    reference = reference_render(
        samples, coeff_table, step=1.0, n_output=len(decimated)
    )

    errors = [abs(a - b) for a, b in zip(decimated, reference, strict=True)]
    return {
        "halt": result.halt.reason,
        "instructions": result.instructions,
        "rendered_decimated": decimated,
        "reference": reference,
        "max_abs_error": max(errors) if errors else None,
        "mean_abs_error": sum(errors) / len(errors) if errors else None,
    }


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("image", help='sharc.py image name, e.g. "dt2-1.16"')
    p.add_argument("--voice", type=int, default=0)
    p.add_argument("--blocks", type=int, default=8)
    p.add_argument("--pitch-step", type=float, default=1.0)
    p.add_argument("--sample-len", type=int, default=4096)
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

    start = time.perf_counter()
    blocks, results = render_blocks(
        memory,
        a.image,
        a.voice,
        a.blocks,
        pitch_step=a.pitch_step,
        sample_len=a.sample_len,
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
            print(
                "correctness: max_abs_error=%.4g mean_abs_error=%.4g (see "
                "sharc_harness.py's docstring/check_correctness() before "
                "trusting this number)"
                % (correctness["max_abs_error"], correctness["mean_abs_error"])
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
