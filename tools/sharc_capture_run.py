"""Capture real ColdFire<->SHARC DSPI2 frames (and, best-effort, SSI0 "audio
in" requests) from a running Digitakt II 1.16 snapshot, for
tools/sharc_replay.py.

    uv run python tools/sharc_capture_run.py SNAPSHOT --out CAPTURE.dt2cap \
        --kind idle|note|play --instrs N [--syx SYX] [--ssi0-hz N]
        [--trig-at N] [--force-period N] [--unblock]
        [--poke-track-type TRACK:TYPE ...]

Two things a run needs, both handled here:

**Getting a DSPI2 driver call to happen at all.** The vector-191 ISR
(docs/findings/04-coldfire-dsp-link.md, "The ColdFire tells the SHARC
through a periodic DSPI2 frame"; addresses from tools/framelink.py) builds
the TX payload and calls the DSPI2 driver (`FUN_400cd2bc`) with it. This
tool hooks that call directly the way tools/sharcframe.py already does
(intercept the stack args via `emu.dspiframe.read_driver_call`, fake the
return) rather than modelling eDMA/DSPI2 registers -- nothing about
capturing needs the transport modelled, only what crossed it. It also opens
the frame-build gate (docs/findings/04, "The frame capture runs; the frame
build is switched off": the long at the profile's `gate` address must be 0
or the handler sends stale/zero bytes) the same way `--open-gate` does.

Getting vector 191 to *fire*, so that call happens, is two independent
mechanisms, both always on:

1. **SSI0 model, for "audio in" only** (best-effort): `--ssi0-hz` wires up
   `emu.ssi.Ssi0Dma` via `emu.longrun.build(ssi0_request_hz=...,
   ssi0_legacy_upgrade=True)`, claiming the snapshot's already-programmed
   SSI0 TCDs (the same trick `tools/guirun.py` uses) so
   `emu.sharc_capture.CapturingPeer` can observe genuine SSI0 "audio in"
   requests (`peer.rx()`) as the run proceeds under `emu.longrun.spin()`.
   Measured on `boot400M.snap`, `--unblock` off: this reaches real
   TCD50 major-loop completions and a delivered vector 170 every time
   (`Ssi0Dma.vector170` increments normally), but the vector-170 ISR that
   ran there did not write `INTFRCH1`'s force bit even after ten
   deliveries (`Ssi0Dma.force_asserted` stayed `False` throughout) --
   so on this snapshot/config the documented "SSI0-paced eDMA50 completion
   -> vector 170 -> software-forced vector 191" chain (`emu/ssi.py`'s own
   docstring) does not by itself reach the DSPI2 driver call within a
   bounded run. Recorded here as a finding, not chased further (out of
   this lane's scope: `emu/ssi.py`/the vector-170 handler's *own*
   forward-progress conditions belong to whichever lane owns that model).
   If claiming the snapshot's SSI0 TCDs fails outright (a shape mismatch --
   `Ssi0Dma.arm_legacy()` raises `RuntimeError`), SSI0 capture is simply
   skipped; DSPI2 capture (below) never depends on it either way. Either
   outcome is recorded in the capture's own JSON header (`ssi0`: "natural"
   or "unavailable: <reason>").
2. **Forcing it directly, for DSPI2** (always on, and what actually
   produces every DSPI2 frame this tool captures): every `--force-period`
   instructions, from inside `spin()`'s own `on_chunk` hook, clear the
   handler's pacing counter and call `Machine.raise_vector(191)` directly
   -- the same "force it, run the handler like anything else" technique
   `tools/sharcframe.py` already uses and has proven safe, just paced
   through the normal instruction stream (so it shares one Machine and one
   instruction clock with kind='note' below) instead of a separate
   call/return dance per frame.

**Triggering a note deterministically, with no framebuffer reading.**
`--kind note` injects a panel "TRIG 1" press+release at ColdFire instruction
count `--trig-at` (default: a quarter of `--instrs`), via `emu/panelin.py`
(the front-panel UART8 input path -- no GPIO/key-matrix to model, see that
module's own docstring) -- the exact wire bytes
(`encode_buttons(3, 1)` then `encode_buttons(3, 0)`, delivered as one
`feed()` so the firmware's own ISR drains both from one raised vector, per
`emu/panelin.py`'s docstring) a prior lane host-replayed and confirmed reach
the firmware's real panel-input path end to end
(docs/findings/03-ui-and-panel.md, "A panel-path track trigger joins
machine invalidation to the refresh queue": panel machine commit -> ...
-> panel TRIG 1 -> record construction -> queue append -> vector 191 ->
`FUN_4002d438` -> track-0 row -> TX frame `0x94` = machine type). Track 0
in that chain is the panel's own "Track 1" (1-indexed on the hardware,
0-indexed internally) -- this is the "note triggered on track 1" this tool
captures. No UI/framebuffer read is needed: the whole chain is inferred
from the resulting TX frame bytes this tool already captures.

**Starting the sequencer instead of a single panel trigger.** `--kind play`
injects a panel "PLAY" press+release at `--trig-at` the same way (wire bytes
`encode_buttons(2, 8)` then `encode_buttons(2, 0)` -- PLAY is channel 2 bit
3, read back from the running image's own button-name table,
`emu/panelin.py`'s `control_names(m, profile, 'button')`: code 20 = "PLAY",
matching `code_for`'s `channel*8+bit+1` for channels 0-5). This does not
need a `--poke-track-type`: the snapshot's already-loaded pattern has real
active trigs on it (docs/findings/03-ui-and-panel.md, "A panel-path track
trigger joins machine invalidation to the refresh queue": the causal
control "machine change followed by `PLAY` scheduled records only for the
current pattern's tracks 3, 12, 13 and 15" -- so starting the transport on
a stock `boot400M.snap`-derived snapshot already reaches the same
record/queue path as a bare panel TRIG, for those four tracks, with no
pattern-data poke needed). Use it to test whether firmware-driven sequencer
playback reaches the SHARC by a path a single hand-pressed TRIG does not
exercise (real timing, real step advance) -- `--instrs` needs to cover at
least one full pattern loop (16 steps) for every active track to be hit at
least once; at the default tempo that is on the order of 10-20M ColdFire
instructions from the press, not the few million `--kind note` needs.

**Changing a track's machine, without re-running the real UI navigation.**
The same finding's own causal-control run (`out/experiments/a2-queue-
trigger/report.json`) already established, end to end and byte-checked,
that "machine change on track 0, then TRIG 1" is what actually flips the TX
frame's machine-type word (`0x94 + 2*track`) from `0x0000` to a real type --
"TRIG 1 without a machine change" or "machine change without TRIG 1" alone
do not. But reaching that "machine menu commit" state through the real panel
(`MACHINE SEL` chord, encoder turns, `YES`) took that prior run 46-160M
ColdFire instructions of UI navigation just to get there, before the part
this tool cares about (the DSPI2 frame) ever happens -- far more than this
capture tool's own budget. `--poke-track-type TRACK:TYPE` instead seeds the
*result* of that commit directly: the per-track machine-type mirror byte
`0x80003cd0 + track*0x9a` (`tools/framelink.py`'s `track_9a`, corrected base
per docs/findings/04) that `FUN_4002d438`/the vector-191 handler already
reads on every real commit -- the same address and mechanism
`tools/sharcframe.py` used for its own `[V]`-marked single-poke experiment
("Measured, not only read", docs/findings/04). The vector-191 handler does
not care why that byte changed; it copies it into the TX frame exactly the
same way whether a real MACHINE SEL commit, a MIDI program change, or this
poke set it. This is not synthesizing the frame -- the DSPI2 driver call is
still the real one, hooked the same way as always -- it only stands in for
the (already independently `[V]`-established, just prohibitively expensive
to re-run here) UI front-end that would normally produce that same byte
change. Combine with `--kind note` (`--trig-at`) for the full causal chain
that report.json's own `machine_trig1` run used: a poked machine change,
then a real panel TRIG 1.
"""

from __future__ import annotations

import argparse
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unicorn import UC_HOOK_MEM_WRITE  # noqa: E402
from unicorn.m68k_const import (  # noqa: E402
    UC_M68K_REG_A7,
    UC_M68K_REG_D0,
    UC_M68K_REG_PC,
)

import framelink  # noqa: E402
from emu import config, dspiframe, panelin, symbols  # noqa: E402
from emu.dtim import Dtims, Timers  # noqa: E402
from emu.longrun import build, spin  # noqa: E402
from emu.pit import Pits  # noqa: E402
from emu.sharc_capture import CaptureWriter, CapturingPeer  # noqa: E402

# The DSPI1 driver, docs/findings/04-coldfire-dsp-link.md "eDMA and DSPI
# transfer inventory" (channel 14, `0xfc03c034` PUSHR -- a *different*
# physical DSPI block than the SHARC's own `0xec038034`). Confirmed at this
# address on DT2 1.16 only (`out/ghidra/dt2-1.16-emac`); not re-checked on
# 1.15C, so the hook below is gated on the resolved profile's name.
DSPI1_DRIVER_ADDR = 0x400CD48A
DSPI1_PROFILE_NAME = "Digitakt II 1.16"

# The FlexBus window this lane tested as a candidate sample-data/control
# path and ruled out: confirmed by `tools/refscan.py` on the 1.16 image plus
# a decompile of the two functions that touch it (`FUN_400ccda0`,
# `FUN_400ccf74`) to be the SHARC program loader's own boot-time GPIO/status
# handshake, not a data channel -- see docs/findings/04, "Ruled out as the
# control link" (1.16 correction). Watched by default on 1.16 so a capture
# still records if that reading is ever wrong, or if some other code starts
# using the window after boot.
FLEXBUS_WATCH = (0x8C000000, 0x8C000010)

# docs/findings/03-ui-and-panel.md's "A panel-path track trigger joins
# machine invalidation to the refresh queue": the host-replayed wire event
# was channel 3 bit 0 (bytes 0x23 0x01 press, 0x23 0x00 release), and it
# drove track 0 (the panel's "TRIG 1", 1-indexed on the hardware).
TRIG_1_CHANNEL = 3
TRIG_1_BIT = 0

# PLAY's (channel, bit), read back from the running 1.16 image's own button
# name table (`emu/panelin.py`'s `control_names(m, profile, 'button')`:
# code 20 = "PLAY", code 19 = "RECORD", code 21 = "STOP", codes 25-40 =
# "TRIG 1".."TRIG 16" -- matching `panelin.code_for`'s channel*8+bit+1 for
# channels 0-5: code 20 -> channel 2, bit 3). Used for `--kind play`: start
# the sequencer on the pattern already loaded in the snapshot (the a2-queue-
# trigger experiment's "current pattern enqueues tracks 3,12,13,15 only"
# recorded in docs/findings/03-ui-and-panel.md, "A panel-path track trigger
# joins machine invalidation to the refresh queue" -- so a stock snapshot's
# default pattern already has active trigs, with no pattern-data poke
# needed) instead of a single panel TRIG, to test whether firmware-driven
# sequencer playback reaches the SHARC by a different path than a bare panel
# trigger.
PLAY_CHANNEL = 2
PLAY_BIT = 3

# tools/framelink.py TABLES's 'track_9a' entry (base corrected to 0x80003cd0
# in docs/findings/04-coldfire-dsp-link.md, "Corrections to the SRAM
# layout"): 16 rows of 0x9a bytes; byte 0 of each row is the track's machine
# type, per "The machine type reaches the SHARC, at TX frame offset
# 0x94 + 2i".
TRACK_9A_BASE = 0x80003CD0
TRACK_9A_STRIDE = 0x9A
TRACK_TYPE_OFFSET = 0


def _resolve_panel_profile(main_img: bytes):
    return symbols.resolve(main_img)


def parse_track_type_poke(spec: str) -> tuple[int, int]:
    """'TRACK:TYPE' -> (track, type), both plain ints (0x-prefixed or
    decimal). Raises ValueError with SPEC in the message on a bad shape, so
    argparse reports a useful error."""
    if ":" not in spec:
        raise ValueError("bad --poke-track-type %r (want TRACK:TYPE)" % spec)
    track_s, type_s = spec.split(":", 1)
    track, type_ = int(track_s, 0), int(type_s, 0)
    if not 0 <= track < 16:
        raise ValueError("bad --poke-track-type %r: track must be 0..15" % spec)
    return track, type_


def parse_mem_range(spec: str) -> tuple[int, int]:
    """'LO:HI' -> (lo, hi), both plain ints (0x-prefixed or decimal), for
    `--watch-mem`: every write with lo <= address < hi becomes a
    `emu.sharc_capture.CaptureWriter.write_mem_write` record."""
    if ":" not in spec:
        raise ValueError("bad --watch-mem %r (want LO:HI)" % spec)
    lo_s, hi_s = spec.split(":", 1)
    lo, hi = int(lo_s, 0), int(hi_s, 0)
    if hi <= lo:
        raise ValueError("bad --watch-mem %r: HI must be > LO" % spec)
    return lo, hi


def install_dspi1_observer(at, writer, counter) -> None:
    """Log every call to the DSPI1 driver (`DSPI1_DRIVER_ADDR`'s own
    docstring), without altering control flow: unlike the DSPI2 driver hook
    in `run()`, which must fake a reply since nothing here answers as a real
    SHARC would, this is a plain observer -- the same idiom
    `emu.longrun.build()`'s own `task_create`/`do_print` hooks use -- so
    whatever the real firmware does next (including a real hang, if this
    turns out to matter) happens exactly as it would with no hook at all."""

    def dspi1_hook(uc, addr, size, data):
        sp = uc.reg_read(UC_M68K_REG_A7)
        _ret, length, src, callback = struct.unpack(">IIII", bytes(uc.mem_read(sp, 16)))
        writer.write_dspi1_call(counter(), length, src, callback)

    at(DSPI1_DRIVER_ADDR, dspi1_hook)


def install_mem_write_watch(m, writer, counter, lo: int, hi: int) -> None:
    """Record every guest write in `[lo, hi)` as a
    `emu.sharc_capture.CaptureWriter.write_mem_write` record. `value` comes
    straight from Unicorn's own `UC_HOOK_MEM_WRITE` callback (the same
    convention `emu/console.py`'s `on_write` uses for UART8), packed
    big-endian to `size` bytes -- the ColdFire's own byte order, matching
    every other field this format records."""

    def mem_write_hook(uc, access, address, size, value, data):
        writer.write_mem_write(
            counter(), address, (value & ((1 << (8 * size)) - 1)).to_bytes(size, "big")
        )

    m.uc.hook_add(UC_HOOK_MEM_WRITE, mem_write_hook, begin=lo, end=hi - 1)


def poke_track_type(m, track: int, type_: int) -> None:
    """Set TRACK's machine-type mirror byte directly (see TRACK_9A_BASE's
    own docstring) -- the same address/mechanism docs/findings/04's
    tools/sharcframe.py experiment used, standing in for a real (but, per
    this module's own docstring, prohibitively expensive to re-run here)
    MACHINE SEL commit. Does not touch the frame itself: the next real
    vector-191 firing reads this byte through FUN_4002d438/the handler's
    own copy exactly as it would a firmware-driven change."""
    m.uc.mem_write(
        TRACK_9A_BASE + track * TRACK_9A_STRIDE + TRACK_TYPE_OFFSET,
        bytes([type_ & 0xFF]),
    )


def run(
    snapshot: str,
    out_path: str,
    *,
    kind: str,
    instrs: int,
    syx: str | None = None,
    ssi0_hz: int = 1000,
    trig_at: int | None = None,
    force_period: int = 50_000,
    chunk: int = 200_000,
    unblock: bool = False,
    poke_track_types: tuple[tuple[int, int], ...] = (),
    watch_mem: tuple[tuple[int, int], ...] = (),
) -> dict:
    if kind not in ("idle", "note", "play"):
        raise ValueError("kind must be 'idle', 'note' or 'play', got %r" % (kind,))
    if kind in ("note", "play") and trig_at is None:
        trig_at = instrs // 4

    image = config.main_image()
    image_sha256, prof = framelink.profile_for(image)
    with open(image, "rb") as fh:
        main_img = fh.read()

    ssi0_status = "natural"
    build_kwargs: dict[str, object] = dict(
        unblock=unblock,
        softfloat=True,
        bitmap=True,
        dsp=True,
        ssi0_request_hz=ssi0_hz,
        ssi0_legacy_upgrade=True,
    )
    if syx:
        build_kwargs["syx"] = syx
    try:
        m, ev, _st, pc, _inq, at = build(snapshot, **build_kwargs)
    except RuntimeError as exc:
        # Ssi0Dma.arm_legacy() shape mismatch, or the force-RTE symbol
        # missing -- fall back to no SSI0 model at all (see module
        # docstring, "Fallback"): DSPI2 capture must not depend on it.
        ssi0_status = "unavailable: %s" % exc
        build_kwargs.pop("ssi0_request_hz")
        build_kwargs.pop("ssi0_legacy_upgrade")
        m, ev, _st, pc, _inq, at = build(snapshot, **build_kwargs)

    panel_profile = _resolve_panel_profile(main_img)
    m.uc.mem_write(prof["gate"], bytes(4))  # open the frame-build gate
    for track, type_ in poke_track_types:
        poke_track_type(m, track, type_)

    pits = Timers(Pits(m), Dtims(m, channels=(3,)))
    ssi0 = ev.get("ssi0_dma")
    if ssi0 is not None:
        ssi0.align(pits.now)

    writer = CaptureWriter(
        out_path,
        frame_bytes=dspiframe.FRAME_BYTES,
        kind=kind,
        device="dt2",
        source_sha256=image_sha256,
        extra={
            "profile": prof["name"],
            "ssi0": ssi0_status,
            "ssi0_hz": ssi0_hz if ssi0 is not None else None,
            "poke_track_types": ["%d:%d" % (t, y) for t, y in poke_track_types],
        },
    )
    peer = CapturingPeer(writer, counter=lambda: pits.now)
    if ssi0 is not None:
        ssi0.peer = peer

    def driver_hook(uc, addr, size, data):
        sp = uc.reg_read(UC_M68K_REG_A7)
        ret, tx_len, tx, rx_len, rx = dspiframe.read_driver_call(uc, sp)
        tx_bytes = bytes(uc.mem_read(tx, tx_len)) if tx and tx_len else b""
        rx_bytes = peer.exchange(tx_bytes)
        if rx and rx_len:
            fill = rx_bytes[:rx_len].ljust(rx_len, b"\x00")
            uc.mem_write(rx, fill)
        uc.reg_write(UC_M68K_REG_D0, 0)
        uc.reg_write(UC_M68K_REG_A7, sp + 4)
        uc.reg_write(UC_M68K_REG_PC, ret)

    at(prof["driver"], driver_hook)

    if prof["name"] == DSPI1_PROFILE_NAME:
        install_dspi1_observer(at, writer, lambda: pits.now)

    # Watch the FlexBus boot-loader window by default on 1.16 (see
    # FLEXBUS_WATCH's own docstring), plus whatever the caller asked for.
    mem_ranges = tuple(watch_mem)
    if prof["name"] == DSPI1_PROFILE_NAME:
        mem_ranges = (FLEXBUS_WATCH, *mem_ranges)
    for lo, hi in mem_ranges:
        install_mem_write_watch(m, writer, lambda: pits.now, lo, hi)

    triggered = {"done": False}
    forced = {"last": 0}

    def on_chunk(pc_, done):
        if kind in ("note", "play") and not triggered["done"] and done >= trig_at:
            channel, bit = (
                (TRIG_1_CHANNEL, TRIG_1_BIT)
                if kind == "note"
                else (PLAY_CHANNEL, PLAY_BIT)
            )
            data = panelin.encode_buttons(channel, 1 << bit) + panelin.encode_buttons(
                channel, 0
            )
            panelin.feed(m, panel_profile, data)
            triggered["done"] = True
        if done - forced["last"] >= force_period:
            m.uc.mem_write(prof["counter"], bytes(4))
            m.raise_vector(prof["vector"])
            forced["last"] = done

    async_events = tuple(e for e in (ssi0,) if e is not None)
    try:
        pc, done, stop = spin(
            m,
            pc,
            instrs,
            chunk,
            on_chunk=on_chunk,
            pits=pits,
            async_events=async_events,
        )
    finally:
        writer.close()
        m.close()

    return {
        "stop": stop,
        "instructions": done,
        "frames": writer.counts["dspi2_frames"],
        "ssi0_rx": writer.counts["ssi0_rx"],
        "dspi1_calls": writer.counts["dspi1_calls"],
        "mem_writes": writer.counts["mem_writes"],
        "ssi0_status": ssi0_status,
        "triggered": triggered["done"] if kind in ("note", "play") else None,
        "image_sha256": image_sha256,
        "out": out_path,
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Capture real ColdFire<->SHARC DSPI2/SSI0 traffic from a snapshot."
    )
    p.add_argument("snapshot")
    p.add_argument("--out", required=True)
    p.add_argument(
        "--kind",
        choices=("idle", "note", "play"),
        required=True,
        help="idle: no panel input. note: panel TRIG 1 press+release at "
        "--trig-at. play: panel PLAY press+release at --trig-at (starts "
        "the loaded pattern; see this module's own docstring)",
    )
    p.add_argument("--instrs", type=lambda s: int(s, 0), default=2_000_000)
    p.add_argument("--syx")
    p.add_argument("--ssi0-hz", type=int, default=1000)
    p.add_argument("--trig-at", type=lambda s: int(s, 0), default=None)
    p.add_argument("--force-period", type=lambda s: int(s, 0), default=50_000)
    p.add_argument("--chunk", type=lambda s: int(s, 0), default=200_000)
    p.add_argument("--unblock", action="store_true")
    p.add_argument(
        "--poke-track-type",
        dest="poke_track_type",
        action="append",
        default=[],
        metavar="TRACK:TYPE",
        help="seed a track's machine-type mirror byte before capturing (see "
        "poke_track_type()'s own docstring); repeatable",
    )
    p.add_argument(
        "--watch-mem",
        dest="watch_mem",
        action="append",
        default=[],
        metavar="LO:HI",
        help="record every write in [LO, HI) as a REC_MEM_WRITE record (see "
        "parse_mem_range()'s own docstring); repeatable. On the 1.16 "
        "profile, FLEXBUS_WATCH (0x8c000000-0x8c000010) is always added, "
        "in addition to whatever this gives",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    poke_track_types = tuple(
        parse_track_type_poke(spec) for spec in args.poke_track_type
    )
    watch_mem = tuple(parse_mem_range(spec) for spec in args.watch_mem)
    result = run(
        args.snapshot,
        args.out,
        kind=args.kind,
        instrs=args.instrs,
        syx=args.syx,
        ssi0_hz=args.ssi0_hz,
        trig_at=args.trig_at,
        force_period=args.force_period,
        chunk=args.chunk,
        unblock=args.unblock,
        poke_track_types=poke_track_types,
        watch_mem=watch_mem,
    )
    print(
        "%s: %d frame(s), %d ssi0-rx, %d dspi1-calls, %d mem-writes, "
        "stop=%s, instructions=%d, ssi0=%s%s"
        % (
            args.out,
            result["frames"],
            result["ssi0_rx"],
            result["dspi1_calls"],
            result["mem_writes"],
            result["stop"],
            result["instructions"],
            result["ssi0_status"],
            ""
            if result["triggered"] is None
            else (", triggered" if result["triggered"] else ", NOT triggered"),
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
