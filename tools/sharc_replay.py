"""Replay a tools/sharc_capture_run.py capture into the real SHARC audio
task, from a post-init state.

    uv run python tools/sharc_replay.py dt2-1.16 CAPTURE.dt2cap \
        [--frames N] [--out OUT.wav] [--report OUT.json] [--freq HZ] \
        [--sample-len N] [--poke-candidates]

For each captured DSPI2 frame, in capture order:

1. **Report the frame's own content**, at the per-track offsets
   docs/findings/04-coldfire-dsp-link.md's TX frame map already establishes
   from the ColdFire's own build code (`FUN_4002d438`/the vector-191
   handler): machine type at byte offset `0x94 + 2i` and the two derived
   flags at `0x73c + 2i` / `0x75c + 2i`, for tracks 0-15. This does not
   depend on where -- or whether -- the payload lands on the SHARC side
   (see point 2): it is read straight from the captured bytes, the same way
   `docs/findings/04`'s own `tools/sharcframe.py` experiment read them.

2. **Best-effort, diagnostic only**: writes the captured payload verbatim
   (byte 0 of the payload -> byte 0 of the candidate buffer -- justified by
   the *same* finding: the SHARC-side reader in `FUN_1c2b24` indexes `I5 +
   0x94`/`+0x73c`/`+0x75c`, the identical offsets, which only makes sense if
   the transport is a byte-for-byte copy) at BOTH of
   docs/findings/04's still-open candidate SHARC receive-buffer addresses
   (`CANDIDATE_RX_BUFFERS` below, "The SHARC reads and change-tests the
   0x94 + 2i machine word": "The two strongest candidate receive states are
   now concrete but still not aliased to I5"). Neither address is
   established as the real one -- this tool reports whether poking them
   measurably changed anything (it does not gate the render on it either
   way), not a claim that it found the real buffer.

3. **Writes the captured payload to the confirmed SHARC receive buffer**
   (`RX_BASE = 0x2558dc`, docs/findings/06's "The ColdFire frame is mapped
   into SHARC DM at 0x2558dc" -- a whole-frame mapped range
   `[0x2558dc, 0x2560de]`, 0x802 bytes, exactly this tool's own captured
   `tx` length): byte 0 of the payload to byte 0 of that DM range, the same
   1:1 mapping the finding's own literal-subtraction table establishes.
   This supersedes the two `CANDIDATE_RX_BUFFERS` an earlier version of
   this tool poked (`0x261bac`/`0x261aa4`, docs/findings/04's "still not
   aliased to I5" candidates from before `0x2558dc` was confirmed by
   execution) -- kept below only as an additional, off-by-default,
   diagnostic poke (`--poke-candidates`), not the transport this tool now
   relies on.

4. **Drives the real audio-task call chain**, the same one
   tools/sharc_harness.py's `render_frames()`/`call_frame()` already proved
   reaches render_frame (block_handler -> command_dispatch_fn ->
   cmd_handler_3 -> render_frame -- docs/findings/06's "Frame call path"),
   with `FRAME_PATCH_TABLE` (that module's own hypotheses for the stops an
   almost-empty synthetic frame hits) via
   `sharc_harness.call_frame_collect_all()` -- every stop the frame call
   passes is recorded (`per_frame[i]["stops"]`), not only the first, so a
   replay's own report is a full list of what each frame depended on.

   **The command word now comes from each captured frame's own header**
   (`frame_command()` below), not a hardcoded 3. docs/findings/04's "Frame
   content: a header written at +0x00" already documents that the
   ColdFire's own frame-build code writes a header word at TX byte offset
   0; every `.dt2cap` capture on hand (idle, idle15M, note-track1,
   machine2-track0, play-pretracks) has that word as `1` on the capture's
   first frame and `3` on every frame after it -- an exact match to
   docs/findings/06's own command semantics ("1" clears the rings and
   stores 0, "3" renders). No static, `tools/sharc.py`-resolved data-flow
   edge from this TX byte into `command_word`
   (`0x264220 + (DM(command_word_shift_src=0x261ca4)<<12)`, itself written
   only by `FUN_1c77b4`'s bit-toggle, which has no static caller of its
   own) was found: the correlation is a same-value-sequence behavioural
   match across every capture on hand, not a proven store-to-load edge, so
   it is used here as the best available per-frame command signal, marked
   `[D]` in the lane report, not claimed as `[V]`. `--force-command N`
   keeps the old always-N behaviour for an A/B comparison.

   ONE voice is set up via `sharc_harness.setup_voice()`, with a `--freq`
   Hz sine (default 1000 Hz, at `sharc_harness.SOURCE_SAMPLE_RATE`) written
   to its sample buffer -- **not** from the captured frame, because the
   SHARC-side code path that would turn a received DSPI2 frame into an
   active voice record (a written record word +0 / ACTIVE) is not yet
   found: docs/findings/06's "Init writes" leaves "the firmware writer of
   word +0 for a playing voice" open. This tool's own
   `voice_activated_by_firmware` is therefore always `False` -- it names
   the missing link instead of quietly working around it with a
   hand-set-up voice and calling that "firmware-driven". `scan_voice_active`
   reports whether the firmware itself ever marked any *other* voice record
   ACTIVE (`+0x1b8`) while this replay ran, independent of the one this
   tool sets up by hand.

5. **Collects ring A** (DAC output, `0x261cc8 + (flag<<8)`,
   docs/findings/06's "Rings": already Q31, L/R-interleaved, at the final
   output rate -- unlike the single-voice work buffer, this is NOT run
   through `sharc_harness.decimate()`) after each call, and writes an
   L+R-averaged mono WAV from it. **The SHARC's TX reply** -- an actual
   outgoing DSPI2/SPI packet -- is not collected, because no code path that
   builds one is known in either image yet (docs/findings/04: "What happens
   to 2748 after the call is not known... No write to an SPI or DMA
   register has been found yet"); this tool does not invent one.

Stops are reported exactly as `tools/sharc_survey.py`'s `CollectStop`s give
them (category, pc, form, unknowns), via `FRAME_PATCH_TABLE` the same way
`sharc_harness.render_frames()` applies it. This tool never adds a new
patch-table hypothesis of its own -- one belongs in `FRAME_PATCH_TABLE`
(tools/sharc_harness.py, this lane's own file), not here.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import sharc_harness as h  # noqa: E402
import sharc_run as sr  # noqa: E402
import sharc_trace as st  # noqa: E402

sys.path.insert(0, os.path.dirname(HERE))
from emu import sharc_capture  # noqa: E402

# docs/findings/04-coldfire-dsp-link.md, "The SHARC reads and change-tests
# the 0x94 + 2i machine word": two candidate SHARC-side receive buffers,
# concrete but not established as the real one -- see the module docstring,
# point 2.
CANDIDATE_RX_BUFFERS = {
    "selector_1_0x261b18": 0x261BAC,
    "selector_2_0x261a10": 0x261AA4,
}

# docs/findings/06's "The ColdFire frame is mapped into SHARC DM at
# 0x2558dc": the confirmed (by execution) whole-frame mapped range
# [0x2558dc, 0x2560de], 0x802 bytes -- exactly this tool's own captured `tx`
# length (tools/sharc_capture_run.py hooks the firmware's own driver call,
# whose own argument length is 0x802). Byte i of the payload maps to DM byte
# RX_BASE + i, one-to-one.
RX_BASE = 0x2558DC

# docs/findings/04's per-track TX frame map: offsets into the ColdFire's own
# 0x802-byte payload, which is also this tool's captured `tx` bytes (see
# tools/sharc_capture_run.py: the driver-call hook captures the argument
# length the firmware itself passes, 0x802, not the padded 2748-byte wire
# frame emu/dspi2.py's eDMA model would see).
MACHINE_TYPE_OFFSET = 0x94
FLAG_A_OFFSET = 0x73C
FLAG_B_OFFSET = 0x75C
TRACK_COUNT = 16

# docs/findings/06's "Rings": ring A holds 32 L/R-interleaved Q31 sample
# pairs (64 words) at the final output rate.
RING_A_WORDS = 64

VOICE_ACTIVATED_REASON = (
    "No SHARC-side code path from a received DSPI2 frame to a voice "
    "record's own word+0/ACTIVE fields is established yet "
    "(docs/findings/06 'Init writes': 'the firmware writer of word +0 for "
    "a playing voice is not yet found'; docs/findings/04's two candidate "
    "receive buffers are not aliased to the frame reader's runtime I5: "
    "'No static path establishes either frame value'). This replay drives "
    "the render path with a hand-set-up voice "
    "(tools/sharc_harness.setup_voice), not one the firmware itself "
    "activated from the captured frame."
)


def _u16be(data: bytes, offset: int) -> int | None:
    if offset + 2 > len(data):
        return None
    return (data[offset] << 8) | data[offset + 1]


# The ColdFire's own frame-build code writes a header word at TX byte
# offset 0 (docs/findings/04, "Frame content: a header written at +0x00").
# Every `.dt2cap` capture on hand (idle, idle15M, note-track1,
# machine2-track0, play-pretracks) has this word as 1 on the capture's
# first frame and 3 on every frame after it -- exactly docs/findings/06's
# own command semantics for "clears + stores 0" and "renders". See the
# module docstring's point 4 for what is, and is not, established about
# this being the SAME word `command_dispatch_fn` (0x1c778a) reads at
# `command_word` (0x264220 + (DM(command_word_shift_src=0x261ca4)<<12)).
COMMAND_HEADER_OFFSET = 0x0


def frame_command(tx: bytes) -> int:
    """The command this captured frame's own header asks for -- the u16be
    at `COMMAND_HEADER_OFFSET` -- for driving `replay()`'s per-frame
    dispatch instead of a hardcoded 3. Falls back to 3 (render) only if TX
    is implausibly short to hold a header at all; every real capture has
    one."""
    value = _u16be(tx, COMMAND_HEADER_OFFSET)
    return value if value is not None else 3


def describe_frame(tx: bytes) -> dict:
    """-> per-track machine type / derived-flag words this captured frame
    carries, from the ColdFire's own documented TX layout -- independent of
    where, or whether, it lands on the SHARC side."""
    tracks = [
        {
            "track": i,
            "machine_type": _u16be(tx, MACHINE_TYPE_OFFSET + 2 * i),
            "flag_a": _u16be(tx, FLAG_A_OFFSET + 2 * i),
            "flag_b": _u16be(tx, FLAG_B_OFFSET + 2 * i),
        }
        for i in range(TRACK_COUNT)
    ]
    return {"tx_len": len(tx), "tracks": tracks}


def _write_bytes(state, base: int, data: bytes) -> None:
    for i, byte in enumerate(data):
        h._poke(state, base + i, byte, width=1)


def _read_ring_a(state, image: str) -> list[float]:
    p = h.profile(image)
    ring_flag = st._dm_read(state, p.ring_flag, 4)
    flag = (ring_flag.value & 1) if ring_flag is not None else 0
    base = p.ring_a + (flag << 8)
    out = []
    for i in range(RING_A_WORDS):
        raw = st._dm_read(state, base + i * 4, 4)
        value = raw.value & 0xFFFFFFFF if raw is not None else 0
        if value & 0x80000000:
            value -= 1 << 32
        out.append(value / float(1 << 31))
    return out


def _track_buffers_nonzero(state) -> dict[int, bool]:
    """Whether each of the 16 master-stage per-track input buffers
    (`sharc_harness.TRACK_MIX_BASE + t*TRACK_MIX_STRIDE`, docs/findings/06's
    "Master stage": `0x1c207b` sums these into the master mix) holds any
    nonzero float, L or R half, after a frame call -- read the same way
    `sharc_harness.inject_track_buffer()` writes them, but never itself
    written by this tool. Per-track (not just "any"), so a caller can tell
    which of the 16 tracks the render path actually deposited output into,
    keyed by track index 0-15."""
    out: dict[int, bool] = {}
    for track in range(16):
        base = h.TRACK_MIX_BASE + track * h.TRACK_MIX_STRIDE
        nonzero = False
        for i in range(2 * h.TRACK_MIX_CHANNEL_WORDS):
            raw = st._dm_read(state, base + i * 4, 4)
            if raw is not None and raw.value & 0xFFFFFFFF:
                nonzero = True
                break
        out[track] = nonzero
    return out


def _mono(ring_a_blocks: list[list[float]]) -> list[float]:
    """L/R-interleaved Q31 pairs -> mono, averaging each pair."""
    out = []
    for block in ring_a_blocks:
        for i in range(0, len(block) - 1, 2):
            out.append(0.5 * (block[i] + block[i + 1]))
    return out


# Lane C2's own item 1 targets: whether a real per-frame command dispatch
# (as opposed to a single hand-forced command=3 for every frame) ever lets
# the firmware's own execution populate these, and by which pc, instead of
# staying zero because a synthetic single-command-3 test never ran the
# commands (0-2) or repeated command-3 calls that would reach them.
#
# `MASTER_BUS_SOURCE` is the raw 26-field mixer/gain table this tool's own
# `_write_bytes(state, RX_BASE, frame.tx)` already maps byte-for-byte from
# the captured TX frame (docs/findings/06, "The ColdFire frame is mapped
# into SHARC DM at 0x2558dc") -- checked directly against the captured
# bytes, not via a watchpoint, since nothing in the SHARC image is a
# "writer" of it (it is a ColdFire input, tools/sharc_inputs.py's own
# `frame_label()` bucket). `MASTER_BUS_DECODED` is `FUN_1c2b24`'s own
# 0x1c2e00-0x1c2fb0 decode of that table into per-channel dynamics
# parameters (this branch's own "Continuity across frames, root-caused"
# note in tools/sharc_harness.py); `MIX_GATE` is `DM(0x252d3c)`
# (docs/findings/06's still-open mix gate, three known writers: init
# 0x1c1643, `FUN_1cb336`'s own reset at 0x1cb33a, and `FUN_1cdbb2`'s
# 0x1cdc23); `SLOT_TYPE_WATCH` covers the 32 voice records
# (`sharc_harness.VOICE_RECORD_STRIDE`-strided from 0x2412cc) at the
# `+0x1b9` byte docs/findings/06's "Where the SRC-page words go" names as
# the per-track flag `FUN_001c60a2` sets ("0x2412c8+4+t*0x1d8+0x1b9" ==
# record base 0x2412cc, `+0x1b9`).
MASTER_BUS_SOURCE = (0x255FB6, 0x2560D0)
MASTER_BUS_DECODED = (0x2524D0, 0x2526E8)
MIX_GATE = (0x252D3C, 0x252D40)
SLOT_TYPE_STRIDE = 0x1D8
SLOT_TYPE_OFFSET = 0x1B9
SLOT_TYPE_WATCH = (0x2412CC, 0x2412CC + 32 * SLOT_TYPE_STRIDE)


def _first_nonzero_write(
    events, lo: int, hi: int, *, stride: int | None = None, offset: int | None = None
) -> dict | None:
    """The first write in EVENTS (an `sr.Runner.watch_log`) landing in
    [lo, hi) -- optionally further restricted to a strided per-record field
    (`(address - lo) % stride == offset`, for `SLOT_TYPE_WATCH`) -- whose
    new value is nonzero: `{"address", "pc", "value"}` (all hex strings), or
    None if this frame's own watch_log never wrote a nonzero value there."""
    for event in events:
        if event.access != "write" or not (lo <= event.address < hi):
            continue
        if stride is not None and (event.address - lo) % stride != offset:
            continue
        if event.new_value:
            return {
                "address": "%#x" % event.address,
                "pc": "%#x" % event.pc_sw,
                "value": "%#x" % event.new_value,
            }
    return None


def replay(
    image: str,
    capture_path: str,
    *,
    n_frames: int | None = None,
    poke_candidates: bool = False,
    command: int | None = None,
    ring_flag: int = 0,
    tone_freq: float = 1000.0,
    sample_len: int = 4096,
) -> dict:
    """Replay CAPTURE_PATH's DSPI2 frames from a `run_init()` state, one
    voice set up with a `tone_freq` Hz sine (`sharc_harness.SOURCE_SAMPLE_RATE`
    -- matching `sharc_harness.render_frames()`'s own CLI convention) so the
    voice actually has audible input, not the all-zero (unwritten,
    `explicit_memory_model`) PCM an earlier version of this function left at
    `sample_base`. `poke_candidates` defaults to False now that `RX_BASE` is
    the confirmed transport (see the module docstring); the two stale
    candidates stay available for an explicit diagnostic comparison.

    `command` defaults to None: each frame's own command comes from
    `frame_command(frame.tx)` (that frame's own captured header), so the
    block handler runs whatever command that frame actually asked for
    instead of a hardcoded 3 -- see the module docstring's point 4. Pass an
    int to force every frame to that one command instead (the old
    behaviour), for an A/B comparison.
    """
    cap = sharc_capture.load(capture_path)
    frames = cap.dspi2_frames if n_frames is None else cap.dspi2_frames[:n_frames]

    memory = h.load_image_memory(image)
    init = h.run_init(memory, image)
    if not init.ran:
        return {
            "capture": capture_path,
            "kind": cap.kind,
            "frames_in_capture": len(cap.dspi2_frames),
            "frames_replayed": 0,
            "error": "run_init failed: %s" % init.error,
        }

    runner = h.new_runner(memory, image, init=init)
    state = runner.state
    sample_base = 0x310000
    tone = [
        math.sin(2 * math.pi * (tone_freq / h.SOURCE_SAMPLE_RATE) * i)
        for i in range(sample_len)
    ]
    h._write_samples(state, sample_base, tone, "int16")
    h.setup_voice(state, image, voice=0, sample_len=sample_len, sample_base=sample_base)
    p = h.profile(image)
    # setup_frame()'s own command=/ring_flag= pokes, without forcing a
    # command yet -- each frame below pokes command_word for itself, from
    # that frame's own header (or the forced override).
    h._poke(state, p.command_word_shift_src, 0)
    h._poke(state, p.ring_flag, ring_flag)

    per_frame = []
    ring_a_blocks = []
    first_stop = None
    other_voices_active: dict[int, int] = {}
    commands_seen: dict[int, int] = {}

    target_lo = min(MASTER_BUS_DECODED[0], MIX_GATE[0], SLOT_TYPE_WATCH[0])
    target_hi = max(MASTER_BUS_DECODED[1], MIX_GATE[1], SLOT_TYPE_WATCH[1])
    target_watch = sr.Watchpoint(
        target_lo,
        target_hi,
        on_read=False,
        on_write=True,
        stop=False,
        label="c2-targets",
    )

    for idx, frame in enumerate(frames):
        info = describe_frame(frame.tx)
        info["index"] = idx
        info["instr_count"] = frame.instr_count

        cmd = command if command is not None else frame_command(frame.tx)
        info["command"] = cmd
        commands_seen[cmd] = commands_seen.get(cmd, 0) + 1

        _write_bytes(state, RX_BASE, frame.tx)
        if poke_candidates:
            for addr in CANDIDATE_RX_BUFFERS.values():
                _write_bytes(state, addr, frame.tx)
        h._poke(state, p.command_word_shift_src, 0)
        h._poke(state, p.command_word, cmd)

        src_lo, src_hi = MASTER_BUS_SOURCE
        info["master_bus_source_nonzero"] = any(
            frame.tx[src_lo - RX_BASE : src_hi - RX_BASE]
        )

        runner, result = h.call_frame_collect_all(
            runner, image, patch_table=h.FRAME_PATCH_TABLE, watchpoints=[target_watch]
        )
        state = runner.state
        events = runner.watch_log
        ring_a = _read_ring_a(state, image)
        ring_a_blocks.append(ring_a)
        frame_voices_active = h.scan_voice_active(state, image, exclude=(0,))
        other_voices_active.update(frame_voices_active)
        track_buffers = _track_buffers_nonzero(state)
        master_mix = h.read_master_mix(memory, image, runner)

        terminal = result.terminal
        info["handler"] = "block_handler -> command_dispatch_fn -> cmd_handler_%d" % cmd
        info["stops"] = ["%s@%#x" % (s.category, s.pc) for s in result.stops]
        info["stop_reason"] = terminal.category
        info["stop_pc"] = "%#x" % terminal.pc
        info["instructions"] = result.instructions
        info["voice_active_by_firmware"] = frame_voices_active
        info["track_buffers_nonzero"] = track_buffers
        info["any_track_buffer_nonzero"] = any(track_buffers.values())
        info["master_mix_nonzero"] = any(v for v in master_mix)
        info["ring_a_nonzero"] = any(v for v in ring_a)
        info["mix_gate_write"] = _first_nonzero_write(events, *MIX_GATE)
        info["master_bus_decoded_write"] = _first_nonzero_write(
            events, *MASTER_BUS_DECODED
        )
        info["slot_type_write"] = _first_nonzero_write(
            events,
            *SLOT_TYPE_WATCH,
            stride=SLOT_TYPE_STRIDE,
            offset=SLOT_TYPE_OFFSET,
        )
        per_frame.append(info)

        if first_stop is None and terminal.category != "frame-returned":
            first_stop = {
                "frame": idx,
                "pc": "%#x" % terminal.pc,
                "category": terminal.category,
                "form": terminal.form,
                "instructions": result.instructions,
            }

    return {
        "capture": capture_path,
        "kind": cap.kind,
        "device": cap.device,
        "source_sha256": cap.source_sha256,
        "frames_in_capture": len(cap.dspi2_frames),
        "frames_replayed": len(frames),
        "rx_base": "%#x" % RX_BASE,
        "candidate_rx_buffers": {k: "%#x" % v for k, v in CANDIDATE_RX_BUFFERS.items()},
        "poked_candidates": poke_candidates,
        "tone_freq_hz": tone_freq,
        "commands_seen": commands_seen,
        "forced_command": command,
        "voice_activated_by_firmware": False,
        "voice_activated_by_firmware_reason": VOICE_ACTIVATED_REASON,
        "other_voices_active_by_firmware": other_voices_active,
        "per_frame": per_frame,
        "first_stop": first_stop,
        "ring_a_mono": _mono(ring_a_blocks),
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("image")
    p.add_argument("capture")
    p.add_argument("--frames", type=int, default=None)
    p.add_argument("--out", help="write ring A (L+R averaged) as a mono WAV")
    p.add_argument("--report", help="write the full replay report as JSON")
    p.add_argument(
        "--freq",
        type=float,
        default=1000.0,
        help="the hand-set-up voice's own sine source frequency, in Hz, at "
        "sharc_harness.SOURCE_SAMPLE_RATE (default 1000 Hz)",
    )
    p.add_argument(
        "--sample-len",
        type=int,
        default=4096,
        help="the hand-set-up voice's source buffer length, in samples",
    )
    p.add_argument(
        "--poke-candidates",
        action="store_true",
        help="also poke the two stale candidate RX buffers (see the module "
        "docstring) alongside the confirmed RX_BASE -- diagnostic only",
    )
    p.add_argument(
        "--force-command",
        type=int,
        default=None,
        help="force every frame to this command number instead of reading "
        "it from that frame's own header (frame_command()) -- diagnostic "
        "A/B against the old always-3 behaviour",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    result = replay(
        args.image,
        args.capture,
        n_frames=args.frames,
        poke_candidates=args.poke_candidates,
        command=args.force_command,
        tone_freq=args.freq,
        sample_len=args.sample_len,
    )
    if args.out and "ring_a_mono" in result:
        h.write_wav(args.out, result["ring_a_mono"], sample_rate=48000)
    if args.report:
        with open(args.report, "w") as fh:
            json.dump(result, fh, indent=1)
    print(
        "%s: %d/%d frame(s) replayed, commands_seen=%s, first_stop=%s, "
        "other_voices_active=%s"
        % (
            args.capture,
            result.get("frames_replayed", 0),
            result.get("frames_in_capture", 0),
            result.get("commands_seen"),
            result.get("first_stop"),
            result.get("other_voices_active_by_firmware"),
        )
    )
    return 0 if "error" not in result else 1


if __name__ == "__main__":
    sys.exit(main())
