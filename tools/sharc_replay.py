"""Replay a tools/sharc_capture_run.py capture into the real SHARC audio
task, from a post-init state.

    uv run python tools/sharc_replay.py dt2-1.16 CAPTURE.dt2cap \
        [--frames N] [--out OUT.wav] [--report OUT.json] [--freq HZ] \
        [--sample-len N] [--force-command N]

For each captured DSPI2 frame, in capture order:

1. **Report the frame's own content**, at the per-track offsets
   docs/findings/04-coldfire-dsp-link.md's TX frame map already establishes
   from the ColdFire's own build code (`FUN_4002d438`/the vector-191
   handler): machine type at byte offset `0x94 + 2i` and the two derived
   flags at `0x73c + 2i` / `0x75c + 2i`, for tracks 0-15. This does not
   depend on where -- or whether -- the payload lands on the SHARC side
   (see point 2): it is read straight from the captured bytes, the same way
   `docs/findings/04`'s own `tools/sharcframe.py` experiment read them.

2. **Delivers the transfer at its real hardware address and lets the
   firmware do the rest (lane G1, 2026-09-26)**, instead of poking a
   confirmed-by-execution but firmware-internal buffer directly. Lane G1
   traced the SPI-slave receive DMA's own descriptor-list ring
   (`tools/sharc_harness.py`'s own module note above
   `dma_landing_address()`: `FUN_1c7bd4` builds it at DM `0x2641b0`/
   `0x2641cc`, ADDRSTART fields pointing at `command_word`'s own two
   ping-pong instances, `0x264220`/`0x265220` on DT2 1.16 -- the SC58x/
   2158x Hardware Reference's own Descriptor-List Mode layout,
   out/refs/adsp-2156x-hwr Table 27-10) and confirmed the SAME address from
   `render_frame`'s (`FUN_1c2b24`) own consumer side: its `M10` (saved from
   its incoming `R12` argument, `command_word + (shift<<12)`) is the SOURCE
   of its own 512-long-word (2,048-byte) copy into `RX_BASE` (`0x2558dc`,
   still the confirmed-by-execution buffer `FUN_1c2b24`'s own per-track
   decode reads -- now understood to be a WORKING COPY render_frame makes
   every frame, not the landing zone itself).

   So this tool now calls `sharc_harness.write_dma_transfer()` (writes the
   captured frame's own TX bytes -- 16-bit-unit byte-swapped, see that
   function's own docstring for why -- at whichever ring `command_word_
   shift_src` is NOT currently pointing at) and `sharc_harness.
   drive_dma_completion()` (calls `FUN_1c77b4`, the SPI service's own
   registered completion callback, with an event code that has its own
   bit 5 set -- the one bit that function's own 9 instructions act on --
   toggling the ping-pong shift onto the buffer just written, and re-arming
   `sharc_harness.COPY_GATE_ADDRESS`, a second, still-open gate lane G1
   found guarding render_frame's own companding-copy block; see that
   module's own section note for exactly what is, and is not, traced about
   it) before every frame's own `call_frame_collect_all()`. It no longer
   pokes `RX_BASE`/`command_word` directly at all -- both are now filled by
   `render_frame`'s own execution, from the transfer this tool delivers at
   the real hardware address.

3. **Drives the real audio-task call chain**, the same one
   tools/sharc_harness.py's `render_frames()`/`call_frame()` already proved
   reaches render_frame (block_handler -> command_dispatch_fn ->
   cmd_handler_3 -> render_frame -- docs/findings/06's "Frame call path"),
   with `FRAME_PATCH_TABLE` (that module's own hypotheses for the stops an
   almost-empty synthetic frame hits) via
   `sharc_harness.call_frame_collect_all_with_hits()` -- every stop the
   frame call passes is recorded (`per_frame[i]["stops"]`), not only the
   first, so a replay's own report is a full list of what each frame
   depended on; a breakpoint at `0x1c60a2` (`FUN_1c60a2`, the machine-type
   change detector docs/findings/06's Lane E2 found unreached from a
   synthetic frame) is installed on every frame, so `per_frame[i][
   "fun_1c60a2_hits"]` answers, with a real hit count rather than an
   inferred one, whether a real, correctly-delivered transfer ever reaches
   it.

   **The command word itself is no longer a separate poke.** It comes from
   whatever the delivered transfer's own header (TX byte offset 0,
   docs/findings/04's "Frame content: a header written at +0x00") resolves
   to once `render_frame`/`command_dispatch_fn` read it back -- `cmd` in
   this tool's own per-frame report is `frame_command(frame.tx)` (or
   `--force-command N`'s override, now applied to the TRANSFERRED bytes'
   own header before delivery, not to `command_word` after the fact) purely
   for describing what was sent, not a separate write.

   ONE voice is set up via `sharc_harness.setup_voice()`, with a `--freq`
   Hz sine (default 1000 Hz, at `sharc_harness.SOURCE_SAMPLE_RATE`) written
   to its sample buffer -- **not** from the captured frame, because the
   SHARC-side code path that would turn a received DSPI2 frame into an
   active voice record (a written record word +0 / ACTIVE) is still not
   found even with the transfer delivered correctly this time (see this
   lane's own report). This tool's own `voice_activated_by_firmware` is
   therefore still always `False` for voice 0 (the one it sets up by
   hand) -- it names the missing link instead of quietly working around it
   and calling that "firmware-driven". `scan_voice_active` reports whether
   the firmware itself ever marked any *other* voice record ACTIVE
   (`+0x1b8`) while this replay ran, independent of the one this tool sets
   up by hand.

4. **Collects ring A** (DAC output, `0x261cc8 + (flag<<8)`,
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
import sharc_survey as sv  # noqa: E402
import sharc_trace as st  # noqa: E402

sys.path.insert(0, os.path.dirname(HERE))
from emu import sharc_capture  # noqa: E402

# docs/findings/06's "The ColdFire frame is mapped into SHARC DM at
# 0x2558dc": the confirmed (by execution) whole-frame mapped range
# [0x2558dc, 0x2560de], 0x802 bytes. **[C] Lane G1 (2026-09-26): this is
# render_frame's (FUN_1c2b24) own WORKING COPY of the real transfer, made by
# a 2,048-byte copy from the real hardware landing zone
# (sharc_harness.dma_landing_address()) every frame it runs -- not
# something this tool writes to any more (see the module docstring, point
# 2, and sharc_harness.py's own section note above dma_landing_address()).
# Kept here only for describe_frame()'s own per-track byte offsets below,
# which are relative to this same base either way.
RX_BASE = 0x2558DC

# emu/dspiframe.py's TX_PAYLOAD_BYTES["dt2"]: the real-payload length the
# ColdFire driver copies out of its own TX source buffer before padding the
# rest of the FRAME_BYTES=0xabc (2,748) byte wire frame with tag-only PUSHR
# entries (docs/findings/04-coldfire-dsp-link.md, "TX length ... are not two
# independently sized directions"). Used only by describe_frame() below to
# find a captured frame's own tail (past the real payload) -- lane G1
# delivers the whole transfer (up to sharc_harness.RING_SIZE_BYTES) at the
# real hardware address, not just this prefix.
TX_PAYLOAD_BYTES = 0x802

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
    where, or whether, it lands on the SHARC side. Also reports the
    transfer's tail (past `TX_PAYLOAD_BYTES`, see that constant's own
    docstring): a capture made before this lane's `tools/sharc_capture_run.py`
    fix only ever has `tx_len == TX_PAYLOAD_BYTES` (an empty tail, `None`
    fields below), not a claim the tail is absent on real hardware."""
    tracks = [
        {
            "track": i,
            "machine_type": _u16be(tx, MACHINE_TYPE_OFFSET + 2 * i),
            "flag_a": _u16be(tx, FLAG_A_OFFSET + 2 * i),
            "flag_b": _u16be(tx, FLAG_B_OFFSET + 2 * i),
        }
        for i in range(TRACK_COUNT)
    ]
    tail = tx[TX_PAYLOAD_BYTES:]
    return {
        "tx_len": len(tx),
        "tracks": tracks,
        "has_tail": len(tail) > 0,
        "tail_len": len(tail),
        "tail_nonzero": any(tail) if tail else None,
    }


def _write_bytes(state, base: int, data: bytes) -> None:
    """Byte-for-byte poke, unswapped -- `replay()` itself no longer uses
    this (see the module docstring, point 2: it delivers a transfer at the
    real hardware address via `sharc_harness.write_dma_transfer()` instead).
    Kept as a small utility for `tools/sharc_inputs.py`'s own `dynamic_view()`,
    which still models the pre-lane-G1 direct-poke behaviour for its own,
    separate purpose (a static "no writer" audit, not a claim about where a
    real transfer lands)."""
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
# `MASTER_BUS_SOURCE` is the raw 26-field mixer/gain table `render_frame`'s
# own 2,048-byte copy (lane G1: from the real DMA landing zone, into
# `RX_BASE`) maps byte-for-byte from the captured TX frame (docs/findings/06,
# "The ColdFire frame is mapped into SHARC DM at 0x2558dc") -- checked
# directly against the captured bytes, not via a watchpoint, since nothing in
# the SHARC image is a
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


def call_frame_collect_all_with_hits(
    runner: sr.Runner,
    image: str,
    *,
    trace_pcs,
    patch_table: sv.PatchTable | None = None,
    watchpoints=(),
    max_steps: int = 4_000_000,
    max_hits: int = 200,
    img=None,
    sample_fn=None,
) -> tuple[sr.Runner, sv.CollectAllResult, list[dict]]:
    """Like `sharc_harness.call_frame_collect_all()`, but also answers "does
    control ever reach any of TRACE_PCS during this one frame call", with a
    real pc-hit breakpoint (`Runner.breakpoints`, checked natively inside
    `sharc_survey.run_collect_all()`'s own loop before an instruction
    decodes) rather than inferring reachability from a memory watch on some
    address a function is *believed* to write (lane E2, 2026-09-25/26 --
    see docs/findings/06's "Lane E2" section for why this matters: a
    write-target watch at a candidate offset can read as empty either
    because the function never ran, or because it ran but the watched
    address was wrong -- a pc-hit breakpoint at the function's own entry
    (or any other pc of interest) tells the two apart).

    `Runner.fresh_call()` already carries `breakpoints` forward (`new_runner
    .breakpoints = self.breakpoints`, tools/sharc_run.py), so TRACE_PCS is
    installed on the fresh_call runner this makes internally -- a caller
    does not need to pre-set `runner.breakpoints` itself. A breakpoint hit
    is not fatal to the frame call: this function clears `breakpoints`
    (saving the set first), executes exactly the one instruction at that pc
    (`Runner.step()`, tools/sharc_run.py's own single-instruction API,
    unaffected by whatever `sharc_survey.run_collect_all()`'s own inlined
    stepping loop does elsewhere), restores `breakpoints`, and resumes
    `run_collect_all()` -- so a frame with several hits (e.g. a function
    called once per voice/track, in a `DO 32` loop) still runs to its real
    terminal (`frame-returned`, a genuine fork, an MMR trap, ...), with
    every intermediate hit recorded rather than only the first.

    Returns `(runner, result, hits)`: `runner` is the fresh_call Runner,
    positioned at the true terminal stop (not at a hit -- hits are always
    stepped past); `result` is that terminal leg's own `CollectAllResult`
    (earlier legs' own stops, if any, are not merged in -- only the hits
    list accumulates across legs); `hits` is
    `[{"pc": "0x...", "instructions": N}, ...]`, oldest first, `N` being
    `runner.instructions` (this frame's own running count, comparable
    across hits and to `result.instructions`) at the moment of that hit,
    before the pc's own instruction executed. Stops at `max_hits` hits
    (a runaway loop should not build an unbounded list) or `max_steps`
    total instructions, whichever comes first -- either way, `result`
    still reports whatever terminal `run_collect_all()` last reached.

    `sample_fn`, if given, is called as `sample_fn(new_runner, terminal.pc)`
    right when a hit is recorded -- *before* that pc's own instruction has
    executed, so it sees exactly the register/memory state the about-to-run
    instruction (often a conditional jump) will read. Its return value
    (any JSON-able dict) is merged under the hit's own `"sample"` key.
    Never called when `sample_fn` is None (the default), which keeps every
    existing caller's hit-dict shape (and tests/test_sharc_replay.py's own
    assertions on it) unchanged -- this is a pure opt-in addition, not a
    behaviour change (lane H3, 2026-09-26, tools/sharc_armpath.py's own
    caller: reading raw register state at a guard jump this way, rather
    than re-deriving it from the disassembly's own displacement units,
    which sharc_core's `modify()` resolves once and correctly already).
    """
    p = h.profile(image)
    new_runner = runner.fresh_call(p.block_handler, diagnose_unknown=True)
    if watchpoints:
        new_runner.attach_watchpoints(list(watchpoints))
    trace_pcs = frozenset(trace_pcs)
    new_runner.breakpoints = frozenset(new_runner.breakpoints) | trace_pcs
    hits: list[dict] = []
    result = None
    remaining = max_steps
    while remaining > 0 and len(hits) < max_hits:
        result = sv.run_collect_all(new_runner, patch_table or {}, remaining, img=img)
        terminal = result.terminal
        if terminal.category != "breakpoint" or terminal.pc not in trace_pcs:
            break
        hit = {"pc": "%#x" % terminal.pc, "instructions": new_runner.instructions}
        if sample_fn is not None:
            hit["sample"] = sample_fn(new_runner, terminal.pc)
        hits.append(hit)
        saved = new_runner.breakpoints
        new_runner.breakpoints = frozenset()
        try:
            new_runner.step()
        except sr.Halt as exc:
            hits[-1]["step_halt"] = str(exc)
            return new_runner, result, hits
        new_runner.breakpoints = saved
        remaining = max_steps - new_runner.instructions
    return new_runner, result, hits


FUN_1C60A2 = 0x1C60A2  # docs/findings/06's Lane E2 machine-type change detector.

# docs/findings/06's Lane F2: the pointer FUN_1c60a2's own gate dereferences
# -- null (via State.explicit_memory_model) everywhere this project has run
# with an unfilled companding record. A real, correctly-delivered transfer
# (lane G1) does not fill DM(0x266220) either (see sharc_harness.py's own
# section note: the transfer fits inside command_word's own ring, and never
# reaches command_record_table's), so this stays a diagnostic read, not
# something this tool expects to change -- reported per frame either way.
COMPANDING_GATE_POINTER = 0x254D78


def replay(
    image: str,
    capture_path: str,
    *,
    n_frames: int | None = None,
    command: int | None = None,
    ring_flag: int = 0,
    tone_freq: float = 1000.0,
    sample_len: int = 4096,
    provisional_interpretations: dict[str, str] | None = None,
) -> dict:
    """Replay CAPTURE_PATH's DSPI2 frames from a `run_init()` state, one
    voice set up with a `tone_freq` Hz sine (`sharc_harness.SOURCE_SAMPLE_RATE`
    -- matching `sharc_harness.render_frames()`'s own CLI convention) so the
    voice actually has audible input, not the all-zero (unwritten,
    `explicit_memory_model`) PCM an earlier version of this function left at
    `sample_base`.

    **Delivery (lane G1, 2026-09-26): every frame's own captured TX bytes
    are written at the real SPI-slave receive DMA's own landing zone**
    (`sharc_harness.write_dma_transfer()`) and the transfer is "completed"
    the way real hardware would (`sharc_harness.drive_dma_completion()`,
    toggling the ping-pong shift and the companding-copy gate -- see that
    module's own section note) -- not poked into `RX_BASE`/`command_word`
    directly any more. See the module docstring's point 2.

    `command` defaults to None: each frame's own command comes from
    `frame_command(frame.tx)` (that frame's own captured header) -- the
    block handler runs whatever command that frame actually asked for,
    read back from the transfer itself, not a hardcoded 3. Pass an int to
    OVERRIDE that frame's own header bytes (TX offset 0-1, big-endian)
    before delivering it, for an A/B comparison -- this still goes through
    `write_dma_transfer()`/`drive_dma_completion()`, it does not poke
    `command_word` after the fact.

    `provisional_interpretations`, if given, is threaded through
    `sharc_harness.run_init()` to every frame call afterward (`fresh_call()`
    keeps a State's own `provisional_interpretations`/`provisional_
    interpreted` -- see `tools/sharc_run.py`'s `fresh_call_state()`): an
    opt-in way past an undocumented form this replay would otherwise stop
    at (e.g. `{"21p_undoc16": "nop"}`), reported back per frame in
    `provisional_hits` so a caller can tell a provisional-assisted run from
    a clean one -- label any result built with this **[D]**, not **[V]**.
    """
    cap = sharc_capture.load(capture_path)
    frames = cap.dspi2_frames if n_frames is None else cap.dspi2_frames[:n_frames]

    memory = h.load_image_memory(image)
    init = h.run_init(
        memory,
        image,
        provisional_interpretations=provisional_interpretations,
    )
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
    h.setup_frame_dma(state, image, ring_flag=ring_flag)

    per_frame = []
    ring_a_blocks = []
    first_stop = None
    other_voices_active: dict[int, int] = {}
    commands_seen: dict[int, int] = {}
    fun_1c60a2_total_hits = 0
    provisional_used: set[str] = set()

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

        cmd = frame_command(frame.tx)
        tx = frame.tx
        if command is not None and cmd != command:
            tx = bytearray(tx)
            tx[0] = (command >> 8) & 0xFF
            tx[1] = command & 0xFF
            tx = bytes(tx)
            cmd = command
        info["command"] = cmd
        commands_seen[cmd] = commands_seen.get(cmd, 0) + 1

        landing_base = h.write_dma_transfer(state, image, tx)
        runner = h.drive_dma_completion(runner, image)
        state = runner.state
        info["landing_base"] = "%#x" % landing_base

        src_lo, src_hi = MASTER_BUS_SOURCE
        info["master_bus_source_nonzero"] = any(tx[src_lo - RX_BASE : src_hi - RX_BASE])

        runner, result, hits = call_frame_collect_all_with_hits(
            runner,
            image,
            trace_pcs={FUN_1C60A2},
            patch_table=h.FRAME_PATCH_TABLE,
            watchpoints=[target_watch],
        )
        state = runner.state
        fun_1c60a2_total_hits += len(hits)
        events = runner.watch_log
        ring_a = _read_ring_a(state, image)
        ring_a_blocks.append(ring_a)
        frame_voices_active = h.scan_voice_active(state, image, exclude=(0,))
        other_voices_active.update(frame_voices_active)
        track_buffers = _track_buffers_nonzero(state)
        master_mix = h.read_master_mix(memory, image, runner)
        companding_fields = h.companding_record_fields(state, image)
        if state.provisional_interpreted:
            provisional_used.update(state.provisional_interpreted)

        terminal = result.terminal
        info["handler"] = "block_handler -> command_dispatch_fn -> cmd_handler_%d" % cmd
        info["stops"] = ["%s@%#x" % (s.category, s.pc) for s in result.stops]
        info["stop_reason"] = terminal.category
        info["stop_pc"] = "%#x" % terminal.pc
        info["instructions"] = result.instructions
        info["fun_1c60a2_hits"] = [hit["pc"] for hit in hits]
        info["voice_active_by_firmware"] = frame_voices_active
        info["track_buffers_nonzero"] = track_buffers
        info["any_track_buffer_nonzero"] = any(track_buffers.values())
        info["master_mix_nonzero"] = any(v for v in master_mix)
        info["ring_a_nonzero"] = any(v for v in ring_a)
        info["companding_fields"] = ["%#x" % v for v in companding_fields]
        info["companding_gate_pointer_nonzero"] = bool(companding_fields[0])
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
        "tx_payload_bytes": "%#x" % TX_PAYLOAD_BYTES,
        "tone_freq_hz": tone_freq,
        "commands_seen": commands_seen,
        "forced_command": command,
        "voice_activated_by_firmware": False,
        "voice_activated_by_firmware_reason": VOICE_ACTIVATED_REASON,
        "other_voices_active_by_firmware": other_voices_active,
        "fun_1c60a2_total_hits": fun_1c60a2_total_hits,
        "companding_gate_pointer": "%#x" % COMPANDING_GATE_POINTER,
        "any_companding_gate_pointer_nonzero": any(
            f["companding_gate_pointer_nonzero"] for f in per_frame
        ),
        "any_signal_without_injection": any(
            f["any_track_buffer_nonzero"] or f["master_mix_nonzero"] for f in per_frame
        ),
        "provisional_interpretations": provisional_interpretations,
        "provisional_used": sorted(provisional_used),
        "per_frame": per_frame,
        "first_stop": first_stop,
        "ring_a_mono": _mono(ring_a_blocks),
    }


# --- Lane K1 (2026-09-26): a voice record's own word+0 across a real,
# continuous replay, and a demo render of whichever voice the firmware
# itself arms ------------------------------------------------------------
#
# Lane J1 replayed `out/captures/dt2-1.16-running-trig.dt2cap` from a
# `--start-frame 300` shortcut (skipping frames 0-299 without executing
# them) and found voice 4's own FIELD_SAMPLE_PTR (word+0,
# `sharc_harness.FIELD_SAMPLE_PTR`) already null (`0x0`) from the very
# first frame that shortcut fed -- on BOTH the TRIG and the idle capture --
# even though voice 4's record reads a real SDRAM address (`0x8045a6c8`)
# right after `run_init()`. J1's own report flagged this as possibly a
# cold-start artifact of the shortcut itself, not a genuine frame-300
# event, and left it open. This section answers that with a real,
# continuous replay from frame 0 (no `start_frame` shortcut) -- see
# `replay_armed_voice()`.

# One log-only (`stop=False`), write-only 4-byte Watchpoint per voice
# record's own word+0, for every `sharc_harness.VOICE_RECORD_COUNT` voice
# record (`sharc_harness.profile(image).voice_records`,
# `sharc_harness.VOICE_RECORD_STRIDE` apart). Labelled so a WatchEvent's own
# `label` (carried through unchanged by `sharc_run._WatchTracker`, never
# re-derived from its address) identifies which voice wrote, without
# re-deriving the voice index from `(address - base) // stride` -- exact
# even if a canonicalized range ever shifted the reported address.
VOICE_WORD0_LABEL_PREFIX = "voice-word0-"
VOICE_ACTIVE_LABEL_PREFIX = "voice-active-"


def _voice_word0_watchpoints(image: str) -> list[sr.Watchpoint]:
    base = h.profile(image).voice_records
    return [
        sr.Watchpoint(
            base + i * h.VOICE_RECORD_STRIDE + h.FIELD_SAMPLE_PTR,
            base + i * h.VOICE_RECORD_STRIDE + h.FIELD_SAMPLE_PTR + 4,
            on_read=False,
            on_write=True,
            stop=False,
            label="%s%d" % (VOICE_WORD0_LABEL_PREFIX, i),
        )
        for i in range(h.VOICE_RECORD_COUNT)
    ]


def _voice_active_watchpoint(image: str, voice: int) -> sr.Watchpoint:
    """Log-only, 1-byte Watchpoint on VOICE's own FIELD_ACTIVE
    (`+0x1b8`) -- separate from the word+0 watches above so this lane can
    name the exact pc of a deactivating write (docs/findings/06's
    `0x1c5008` past-limit / `0x1c50fe` wrap candidates), not just infer a
    frame-level transition from re-reading the byte after each call."""
    record = h.voice_record_address(image, voice)
    return sr.Watchpoint(
        record + h.FIELD_ACTIVE,
        record + h.FIELD_ACTIVE + 1,
        on_read=False,
        on_write=True,
        stop=False,
        label="%s%d" % (VOICE_ACTIVE_LABEL_PREFIX, voice),
    )


def _voice_index_from_label(label: str, prefix: str) -> int | None:
    if not label.startswith(prefix):
        return None
    return int(label[len(prefix) :])


def _hex_or_none(value: int | None) -> str | None:
    return None if value is None else "%#x" % value


def replay_armed_voice(
    image: str,
    capture_path: str,
    *,
    voice: int = 4,
    extra_frames: int = 20,
    n_frames: int | None = None,
    start_frame: int = 0,
) -> dict:
    """Replay CAPTURE_PATH continuously from frame 0 by default (no
    `start_frame` shortcut -- lane J1's own open question, see this
    section's module note; the report this lane wrote used `start_frame=0`
    throughout) on ONE Runner, through `sharc_harness.call_frame_with_track_
    injection()` every frame (real DMA delivery via `sharc_harness.
    write_dma_transfer()`/`drive_dma_completion()`, the same as `replay()`
    above -- NOT the older `write_capture_frame()`/`CAPTURE_FRAME_BASE`
    path `render_frames_to_ring_a()` uses, which lane G1 already found does
    not reach the real arming code), with two things attached to every
    frame's own `fresh_call()`:

    `START_FRAME` (default 0), if given, skips that many of CAPTURE's own
    leading frames without executing them at all -- lane J1's own
    `tools/sharc_armpath.py` shortcut, `per_frame`'s own real capture index
    preserved (every reported frame number below is `start_frame +
    local_idx`, never a re-based 0). Only safe once whatever it skips is
    known not to matter: this lane's own from-frame-0 run (see the module
    note) found VOICE's own word+0 already null by frame ~2, independent
    of the capture's own TRIG byte, so `start_frame=300` (well past that)
    reproduces this same run's own `arm_frame`/`active_frames`/
    `deactivate_pc`/`sample_pointer_at_arm` for THIS capture exactly --
    used by this lane's own slow test to pin the demo WAV without a ~3
    minute full replay on every run. Do not assume this for a different
    capture or a different voice without checking word+0's own null-timing
    there first.

    1. A log-only write Watchpoint on word+0 (`FIELD_SAMPLE_PTR`) of all
       32 voice records (`_voice_word0_watchpoints()`) plus one on VOICE's
       own FIELD_ACTIVE (`_voice_active_watchpoint()`) -- every hit is
       collected into `word0_writes`/`active_writes` (frame, pc, old/new),
       oldest first, across the WHOLE replay, not just a bracketed window.

    2. VOICE's own decimated render output injected straight into the
       master mix (`inject_track=False, write_master_mix=True` --
       `sharc_harness.inject_master_mix()`, the SAME "existing harness
       path" `sharc_harness.render_frames_to_ring_a()`'s `--ring-a` CLI
       already uses, just pointed at VOICE's record instead of a
       hand-driven one) on every frame -- harmless before VOICE is ever
       armed (its own work buffer reads as silence), and the one thing
       that lets a firmware-armed voice's real output reach ring A without
       a second, separate render pass, since master mix
       (`sharc_dac.MASTER_MIX_BASE`) is a disjoint memory range from every
       voice record and every byte this lane's own watchpoints track --
       this injection cannot perturb the arming logic itself. **This
       injection is the one hand step in this replay** (see `main()`'s own
       `--armed-voice-wav` help text).

    VOICE's own FIELD_ACTIVE is re-read (not just watched) after every
    frame's call, into `active_by_frame` (`bool`, oldest first): the
    firmware's own `+0x1b8` state right after that frame finished, the
    same convention lane J1 used ("ACTIVE is 1 at frame 304"). `arm_frame`
    is the first frame index where this is True (`None` if VOICE is never
    armed in this replay); `active_frames` counts how many CONSECUTIVE
    frames from `arm_frame` stay True; `deactivate_pc` is the first
    `active_writes` entry at or after `arm_frame` whose new value is 0 (the
    exact pc of the `+0x1b8` clear -- docs/findings/06's `0x1c5008`
    past-limit or `0x1c50fe` wrap), or `None` if the run ends first.
    `sample_pointer_at_arm` is VOICE's own word+0 value from JUST BEFORE
    `arm_frame`'s own frame was delivered (`sample_ptr_before_frame[arm_frame]`
    -- matching lane J1's own "just before this lane's first fed frame"
    phrasing) -- not the value after that frame's own render, which may
    already have changed it.

    Ring A (`sharc_harness.read_ring_a()`, deinterleaved L/R -- NOT this
    module's own `_read_ring_a()`/`_mono()`, whose plain `0.5*(L+R)`
    downmix cancels to silence for a coherent signal identically written
    to both channels, per `sharc_harness.py`'s own "ring A's own L channel
    is deliberately negated" module note) is read every frame and kept
    per-frame; `render_left`/`render_right` are the concatenation of frames
    `[arm_frame, arm_frame + active_frames + EXTRA_FRAMES)` -- the demo
    window the task brief asks for -- or `[]` if VOICE was never armed.

    Returns a dict with `capture`, `frames_replayed`, `voice`,
    `word0_writes`, `active_writes`, `active_by_frame`, `arm_frame`,
    `active_frames`, `deactivate_pc`, `sample_pointer_at_arm`,
    `render_left`, `render_right`, `sample_rate_hz`
    (`sharc_dac`'s own ring-A rate, `int(sharc_harness.SOURCE_SAMPLE_RATE //
    2)` = 48000), and `error` (only present, and everything else absent,
    if `run_init()` itself failed)."""
    cap = sharc_capture.load(capture_path)
    end = None if n_frames is None else start_frame + n_frames
    frames = cap.dspi2_frames[start_frame:end]

    memory = h.load_image_memory(image)
    init = h.run_init(memory, image)
    if not init.ran:
        return {
            "capture": capture_path,
            "frames_in_capture": len(cap.dspi2_frames),
            "frames_replayed": 0,
            "error": "run_init failed: %s" % init.error,
        }

    runner = h.new_runner(memory, image, init=init)
    state = runner.state
    h.setup_frame_dma(state, image, ring_flag=0)

    record = h.voice_record_address(image, voice)
    watchpoints = [
        *_voice_word0_watchpoints(image),
        _voice_active_watchpoint(image, voice),
    ]
    active_label = "%s%d" % (VOICE_ACTIVE_LABEL_PREFIX, voice)

    word0_writes: list[dict] = []
    active_writes: list[dict] = []
    active_by_frame: list[bool] = []
    sample_ptr_before_frame: list[int | None] = []
    ring_left_by_frame: list[list[float]] = []
    ring_right_by_frame: list[list[float]] = []

    for local_idx, frame in enumerate(frames):
        idx = start_frame + local_idx
        ptr_before = st._dm_read(state, record + h.FIELD_SAMPLE_PTR, 4)
        sample_ptr_before_frame.append(
            ptr_before.value & 0xFFFFFFFF if ptr_before is not None else None
        )

        h.write_dma_transfer(state, image, frame.tx)
        runner = h.drive_dma_completion(runner, image)
        state = runner.state

        runner, _result, _injected = h.call_frame_with_track_injection(
            runner,
            image,
            record,
            track=0,
            patch_table=h.FRAME_PATCH_TABLE,
            write_master_mix=True,
            inject_track=False,
            watchpoints=watchpoints,
        )
        state = runner.state

        for event in runner.watch_log:
            if event.access != "write":
                continue
            vi = _voice_index_from_label(event.label, VOICE_WORD0_LABEL_PREFIX)
            if vi is not None:
                word0_writes.append(
                    {
                        "frame": idx,
                        "pc": "%#x" % event.pc_sw,
                        "voice": vi,
                        "old": _hex_or_none(event.old_value),
                        "new": _hex_or_none(event.new_value),
                    }
                )
            elif event.label == active_label:
                active_writes.append(
                    {
                        "frame": idx,
                        "pc": "%#x" % event.pc_sw,
                        "old": event.old_value,
                        "new": event.new_value,
                    }
                )

        active_raw = st._dm_read(state, record + h.FIELD_ACTIVE, 1)
        active_by_frame.append(bool(active_raw is not None and active_raw.value))

        ring = h.read_ring_a(memory, image, runner)
        ring_left_by_frame.append([v or 0.0 for v in ring["left"]])
        ring_right_by_frame.append([v or 0.0 for v in ring["right"]])

    local_arm_idx = next((i for i, a in enumerate(active_by_frame) if a), None)
    arm_frame = None if local_arm_idx is None else start_frame + local_arm_idx
    active_frames = 0
    deactivate_pc = None
    if local_arm_idx is not None:
        i = local_arm_idx
        while i < len(active_by_frame) and active_by_frame[i]:
            active_frames += 1
            i += 1
        for w in active_writes:
            if w["frame"] >= arm_frame and w["new"] == 0:
                deactivate_pc = w["pc"]
                break

    render_left: list[float] = []
    render_right: list[float] = []
    if local_arm_idx is not None:
        hi = min(len(frames), local_arm_idx + active_frames + extra_frames)
        for i in range(local_arm_idx, hi):
            render_left.extend(ring_left_by_frame[i])
            render_right.extend(ring_right_by_frame[i])

    return {
        "capture": capture_path,
        "frames_in_capture": len(cap.dspi2_frames),
        "frames_replayed": len(frames),
        "start_frame": start_frame,
        "voice": voice,
        "record": "%#x" % record,
        "word0_writes": word0_writes,
        "active_writes": active_writes,
        "active_by_frame": active_by_frame,
        "arm_frame": arm_frame,
        "active_frames": active_frames,
        "deactivate_pc": deactivate_pc,
        "sample_pointer_at_arm": (
            _hex_or_none(sample_ptr_before_frame[local_arm_idx])
            if local_arm_idx is not None
            else None
        ),
        "render_left": render_left,
        "render_right": render_right,
        "sample_rate_hz": int(h.SOURCE_SAMPLE_RATE // 2),
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
        "--force-command",
        type=int,
        default=None,
        help="override every frame's own header bytes (TX offset 0-1) to "
        "this command number before delivering the transfer, instead of "
        "reading it from that frame's own captured header (frame_command()) "
        "-- diagnostic A/B, still delivered through the real DMA path",
    )
    p.add_argument(
        "--provisional",
        action="append",
        default=[],
        metavar="FORM=MODE",
        help="an opt-in tools/sharc_core provisional interpretation (e.g. "
        "21p_undoc16=nop) to get a replay past an undocumented form -- may "
        "be given more than once; any result built with this is provisional "
        "([D], not [V]) -- see replay()'s own docstring",
    )
    p.add_argument(
        "--armed-voice-wav",
        action="store_true",
        default=False,
        help="lane K1: instead of the hand-set-up-voice-0 replay() above, "
        "run replay_armed_voice() -- a continuous, from-frame-0 replay "
        "(no start_frame shortcut) that watches all 32 voice records' own "
        "word+0 and --voice's own FIELD_ACTIVE (log-only, never stops the "
        "run), and writes ring A's own output for whichever voice the "
        "FIRMWARE ITSELF arms (--voice, default 4) as a stereo WAV to "
        "--out, from the frame it arms through that many active frames "
        "plus --extra-frames. HAND STEP: this still calls sharc_harness."
        "inject_master_mix() every frame (via call_frame_with_track_"
        "injection(inject_track=False, write_master_mix=True)) to get "
        "--voice's own decimated render output into the master mix -- no "
        "SHARC-side code path from the real per-track mix/gate tables to "
        "the master mix is established yet (see sharc_harness.py's "
        "MASTER_MIX_INJECT_PC module note), so without this hand step "
        "ring A stays silent even for a voice the firmware itself armed "
        "and rendered. --out is a stereo WAV (ring A's own L channel is "
        "the firmware's own negation of R for an identical signal on both "
        "channels -- see that same module note -- so an L+R mono downmix "
        "of this output cancels to silence; play/measure L and R, not "
        "their average).",
    )
    p.add_argument(
        "--voice",
        type=int,
        default=4,
        help="--armed-voice-wav only: which voice record to watch/render "
        "(default 4, docs/findings/06 Lane J1's own finding for "
        "out/captures/dt2-1.16-running-trig.dt2cap)",
    )
    p.add_argument(
        "--extra-frames",
        type=int,
        default=20,
        help="--armed-voice-wav only: frames to render past the voice's "
        "own active window",
    )
    p.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="--armed-voice-wav only: skip this many of the capture's own "
        "leading frames without executing them (see replay_armed_voice()'s "
        "own docstring for when this is, and is not, safe to use) -- "
        "default 0, a genuinely continuous replay from frame 0",
    )
    return p.parse_args(argv)


def _parse_provisional(pairs: list[str]) -> dict[str, str] | None:
    if not pairs:
        return None
    out = {}
    for pair in pairs:
        form, _, mode = pair.partition("=")
        out[form] = mode
    return out


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.armed_voice_wav:
        result = replay_armed_voice(
            args.image,
            args.capture,
            voice=args.voice,
            extra_frames=args.extra_frames,
            n_frames=args.frames,
            start_frame=args.start_frame,
        )
        if args.out and result.get("render_left"):
            h.sharc_dac.write_wav_stereo(
                args.out,
                result["render_left"],
                result["render_right"],
                sample_rate=result["sample_rate_hz"],
            )
        if args.report:
            with open(args.report, "w") as fh:
                json.dump(result, fh, indent=1)
        print(
            "%s: %d/%d frame(s) replayed, voice=%d, arm_frame=%s, "
            "active_frames=%d, deactivate_pc=%s, sample_pointer_at_arm=%s"
            % (
                args.capture,
                result.get("frames_replayed", 0),
                result.get("frames_in_capture", 0),
                args.voice,
                result.get("arm_frame"),
                result.get("active_frames", 0),
                result.get("deactivate_pc"),
                result.get("sample_pointer_at_arm"),
            )
        )
        return 0 if "error" not in result else 1
    result = replay(
        args.image,
        args.capture,
        n_frames=args.frames,
        command=args.force_command,
        tone_freq=args.freq,
        sample_len=args.sample_len,
        provisional_interpretations=_parse_provisional(args.provisional),
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
