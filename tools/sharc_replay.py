"""Replay a tools/sharc_capture_run.py capture into the real SHARC audio
task, from a post-init state.

    uv run python tools/sharc_replay.py dt2-1.16 CAPTURE.dt2cap \
        [--frames N] [--out OUT.wav] [--report OUT.json] [--no-candidates]

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

3. **Drives the real audio-task call chain**, the same one
   tools/sharc_harness.py's `render_frames()`/`call_frame()` already proved
   reaches render_frame (block_handler -> command_dispatch_fn ->
   cmd_handler_3 -> render_frame -- docs/findings/06's "Frame call path"),
   with `FRAME_PATCH_TABLE` (that module's own hypotheses for the stops an
   almost-empty synthetic frame hits). ONE voice is set up via
   `sharc_harness.setup_voice()` -- **not** from the captured frame,
   because the SHARC-side code path that would turn a received DSPI2 frame
   into an active voice record (a written record word +0 / ACTIVE) is not
   yet found: docs/findings/06's "Init writes" leaves "the firmware writer
   of word +0 for a playing voice" open, and docs/findings/04's own
   candidate-buffer investigation could not alias either candidate to the
   frame reader's runtime `I5` ("No static path establishes either frame
   value"). This tool's own `voice_activated_by_firmware` is therefore
   always `False` -- it names the missing link instead of quietly working
   around it with a hand-set-up voice and calling that "firmware-driven".

4. **Collects ring A** (DAC output, `0x261cc8 + (flag<<8)`,
   docs/findings/06's "Rings": already Q31, L/R-interleaved, at the final
   output rate -- unlike the single-voice work buffer, this is NOT run
   through `sharc_harness.decimate()`) after each call, and writes an
   L+R-averaged mono WAV from it. **The SHARC's TX reply** -- an actual
   outgoing DSPI2/SPI packet -- is not collected, because no code path that
   builds one is known in either image yet (docs/findings/04: "What happens
   to 2748 after the call is not known... No write to an SPI or DMA
   register has been found yet"); this tool does not invent one.

Stops are reported exactly as `tools/sharc_run.py`'s `Halt` gives them
(reason, pc, form, instruction count), via `FRAME_PATCH_TABLE` the same way
`sharc_harness.render_frames()` applies it. This tool never patches past a
stop on its own -- a new patch hypothesis belongs in `FRAME_PATCH_TABLE`
(tools/sharc_harness.py, a different lane's file), not here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import sharc_harness as h  # noqa: E402
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

NOT_A_RETURN = "return without followed call"

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


def _mono(ring_a_blocks: list[list[float]]) -> list[float]:
    """L/R-interleaved Q31 pairs -> mono, averaging each pair."""
    out = []
    for block in ring_a_blocks:
        for i in range(0, len(block) - 1, 2):
            out.append(0.5 * (block[i] + block[i + 1]))
    return out


def replay(
    image: str,
    capture_path: str,
    *,
    n_frames: int | None = None,
    poke_candidates: bool = True,
    command: int = 3,
    ring_flag: int = 0,
) -> dict:
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
    h.setup_voice(state, image, voice=0, sample_len=4096)
    h.setup_frame(state, image, command=command, ring_flag=ring_flag)

    per_frame = []
    ring_a_blocks = []
    first_stop = None

    for idx, frame in enumerate(frames):
        info = describe_frame(frame.tx)
        info["index"] = idx
        info["instr_count"] = frame.instr_count

        if poke_candidates:
            for addr in CANDIDATE_RX_BUFFERS.values():
                _write_bytes(state, addr, frame.tx)

        runner, result = h.call_frame(runner, image, patch_table=h.FRAME_PATCH_TABLE)
        state = runner.state
        ring_a_blocks.append(_read_ring_a(state, image))

        halt = result.halt
        info["handler"] = (
            "block_handler -> command_dispatch_fn -> cmd_handler_%d" % command
        )
        info["stop_reason"] = halt.reason
        info["stop_pc"] = "%#x" % halt.pc_sw
        info["instructions"] = result.instructions
        per_frame.append(info)

        if first_stop is None and halt.reason != NOT_A_RETURN:
            first_stop = {
                "frame": idx,
                "pc": "%#x" % halt.pc_sw,
                "reason": halt.reason,
                "form": halt.form,
                "instructions": result.instructions,
            }

    return {
        "capture": capture_path,
        "kind": cap.kind,
        "device": cap.device,
        "source_sha256": cap.source_sha256,
        "frames_in_capture": len(cap.dspi2_frames),
        "frames_replayed": len(frames),
        "candidate_rx_buffers": {k: "%#x" % v for k, v in CANDIDATE_RX_BUFFERS.items()},
        "poked_candidates": poke_candidates,
        "voice_activated_by_firmware": False,
        "voice_activated_by_firmware_reason": VOICE_ACTIVATED_REASON,
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
        "--no-candidates",
        action="store_true",
        help="skip poking the two candidate RX buffers (see the module docstring)",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    result = replay(
        args.image,
        args.capture,
        n_frames=args.frames,
        poke_candidates=not args.no_candidates,
    )
    if args.out and "ring_a_mono" in result:
        h.write_wav(args.out, result["ring_a_mono"], sample_rate=48000)
    if args.report:
        with open(args.report, "w") as fh:
            json.dump(result, fh, indent=1)
    print(
        "%s: %d/%d frame(s) replayed, first_stop=%s"
        % (
            args.capture,
            result.get("frames_replayed", 0),
            result.get("frames_in_capture", 0),
            result.get("first_stop"),
        )
    )
    return 0 if "error" not in result else 1


if __name__ == "__main__":
    sys.exit(main())
