"""DSPI2 (`0xEC038000`) + eDMA channels 28/29: the periodic ColdFire<->SHARC
frame link.

`FUN_400cd2bc` (1.16; `FUN_400cf9c4` on 1.15C) is a DSPI2 send/receive driver
built on eDMA channels 28 and 29. `CTAR0` is set to `0xFA010000` (16-bit SPI
frames); the last PUSHR entry of a burst gets EOQ. The two eDMA word counts
are set from one shared variable (`0xabc` = 2748 bytes on Digitakt II), so
this is a single fixed-length full-duplex exchange, not two independently
sized transfers: the `0x802`-byte figure quoted elsewhere is how many of
those 2748 TX bytes are real payload, the rest tag entries. See
docs/findings/04-coldfire-dsp-link.md, "The ColdFire tells the SHARC through
a periodic DSPI2 frame" and "eDMA and DSPI transfer inventory". The frame's
constants (register block, channel/vector numbers, sizes, per-track layout)
are defined once in `emu/dspiframe.py` and imported here; see that module's
docstring for how the several partial copies of these facts were unified.

**Channel roles.** TCD **29**'s fields are programmed with `DADDR =
0xec038034` (PUSHR) and `SADDR = 0x80001bc0` (SRAM staging, rebuilt per
call) -- channel 29 is TX. TCD **28** gets `SADDR = 0xec03803a` (a POPR-area
hardware address) and `DADDR = 0x80001000` (a *different* SRAM buffer from
the TX staging one) -- channel 28 is RX. Both are armed by `EDMA_CERQ`
(`0xfc044019`, disabling first) then `EDMA_SERQ` (`0xfc044018`), each
written with channel 29 then channel 28, in that order. `TX_CHAN, RX_CHAN =
29, 28` below follow this reading, confirmed two ways independently -- the
Ghidra decompile/disassembly dump and a raw-byte scan of
`sections/section_3_MAIN_OS.bin` at `0x400cd2bc`-`0x400cd482` -- and cross-
checked against Digitone II 1.11's driver (`FUN_400cf7be`), which programs
the same two TCDs the same way at the same offsets. `docs/findings/04-
coldfire-dsp-link.md`'s "eDMA and DSPI transfer inventory" table is corrected
to match, citing both checks.

## The real mechanism: a polled status bit, not an interrupt

`FUN_400cd2bc` never touches the DSPI2/eDMA registers unconditionally. Right
after its two argument-validity checks, it reads `0xEC03802C` (DSPI2 SR) and
does `btst.b #$1c, d0` (bit 28) followed by `beq.w <past-the-whole-transfer>`
-- if that bit is clear, the function returns immediately without arming
either channel. After programming both TCDs and writing SERQ, the function
does not install or reference any interrupt vector anywhere in its body; it
just falls through to its epilogue and returns. That is the whole completion
story: **the driver polls one status bit to see whether the previous transfer
finished, and if not, silently skips this cycle** (consistent with "each
firing sends the frame built by the previous firing", the one-cycle-delayed
double buffer already on record) -- there is no interrupt to deliver for this
transport at all, and no evidence one was ever installed. This was found by
reading the disassembly, then confirmed by running it: with the pre-existing
`0xEC03802C` mock in `emu/longrun.py` (`0x80000000`, bit 31, for the
unrelated SHARC-boot-upload status pair `emu/dspboot.py` already documents)
and nothing forcing bit 28, a vector-191 pass through the real, unstubbed
driver produced *zero* eDMA/DSPI2 writes -- the `beq` was always taken.
`emu/longrun.py`'s `build()` now ORs bit 28 into that same mock, but *only*
when `dspi2_peer` is given (see `build()`'s docstring and the comment at that
line), so a run that does not opt into this module sees exactly the old
constant. This is the one piece of hardware state this module needs the host
Machine to cooperate on; `Dspi2Link` itself only ever touches TCD registers
and SERQ/CINT, matching how `emu/edma.py` and `emu/ssi.py` touch only their
own eDMA registers and leave the peripheral's own status register alone.

An eDMA major-loop-complete interrupt is *also* modeled (see "Completion
vector" below) purely defensively, in case some other, unobserved firmware
path does set `CSR.INT_MAJOR` on these TCDs; nothing in the disassembled
span above does, so by default it never fires and costs nothing.

## Model

Registered as a pair of `UC_HOOK_MEM_WRITE` hooks on `EDMA_SERQ`
(`emu.edma.SERQ`), exactly like the existing UART8 (`emu/edma.py`) and SSI0
(`emu/ssi.py`) channels: arming `TX_CHAN` or `RX_CHAN` runs that TCD's whole
major loop eagerly, in Python, inside the write hook -- SPI is fast enough
relative to a ColdFire instruction that there is no reason to pace it the way
`Ssi0Dma` paces the audio-clocked SSI0 channels. `TX_CHAN`'s *source* bytes
are captured (SRAM staging, "rebuilt per call"); `RX_CHAN`'s *destination*
bytes are supplied by `peer.exchange()` and written into its programmed SRAM
buffer. Both channels are read generically from their live TCD fields
(`SADDR`/`DADDR`, `ATTR` SSIZE/DSIZE, `SOFF`/`DOFF`, `SLAST`/`DLAST`,
`CITER`/`BITER`, and `CSR` for `E_SG`/`INT_MAJOR`) rather than hardcoded, so
it plays back whatever the firmware actually programs instead of an assumed
layout -- the same discipline `emu/edma.py`'s `TxChannel` and `emu/ssi.py`'s
`Ssi0Dma` already follow, for the reasons given in their docstrings
(duplicating firmware struct layout is the mistake to avoid). PUSHR/POPR
themselves are never touched: like `TxChannel`, this collapses the SPI shift
register and hands the moved bytes straight to the peer instead of modeling
the hardware FIFO.

Whichever channel arms *second* (of the pair, since real firmware arms both
before the exchange can be meaningful -- here, RX before TX, per the CERQ/SERQ
order above) triggers `peer.exchange()`; if RX arms first the exchange is
deferred until TX's bytes are captured, and vice versa -- `_maybe_exchange()`
below is written to not care which order actually happens. `peer.exchange()`
must return exactly as many bytes as it was given (a full-duplex link cannot
change length); a peer that returns a different length is a bug in the peer,
and raises rather than silently truncating/padding.

## Completion vector: defensive, off by evidence, not by default

Every other eDMA channel this emulator already models is delivered through
the channel's own INTC vector, and the mapping is uniform across every
verified case in this codebase: channel 34 (panel UART TX) -> vector 154
(`emu/panelin.py`), channel 35 (console UART TX) -> vector 155
(`emu/edma.py`, "verified in the vector table"), channel 48 (SSI0 RX, no
completion vector needed) -> none, channel 50 (SSI0 TX) -> vector 170
(`emu/ssi.py`). All four fit `vector = channel + 120`, and `VBR + 4*vector`
for each matches a *statically confirmed* absolute-address install site
(`tools/refscan.py --range <slot> <slot+4>` on `section_3_MAIN_OS.bin` finds
1-3 literal `move.l dN,$<slot>.l` writes for 34, 35 and 50's slots; only 48,
which needs no completion vector, has none). The same scan on channels 28/29's
slots (`0x40000250`/`0x40000254`) finds **zero** literal writes, and reading
`FUN_400cd2bc` end to end (see "The real mechanism" above) confirms why: this
driver installs no vector at all, for either channel, anywhere in its body --
completion here is the polled `SR_LINK_IDLE` bit, not an interrupt.

This module still raises `vector = channel + 120` (149 for TX/29, 148 for
RX/28) when the TCD's own `CSR` `INT_MAJOR` bit is set (0x0002, exactly like
`Ssi0Dma._run_minor`'s `csr & 0x0002` check for channel 50), purely as a
defensive fallback for a firmware path this worktree's single verification
run did not exercise -- it costs nothing when unused, since `FUN_400cd2bc`
never sets `INT_MAJOR` on either TCD, so by default it never fires.
`raise_completion` (default `True`) turns it off entirely if it ever proves
wrong (an unhandled-vector fault, or any other symptom that goes away with it
off) without losing the exchange itself, which does not depend on it. There
is no existing semaphore for *this* transport to fall back to either -- the
completion semaphore `emu/dspboot.py` patches (`0x44e4d69c`) belongs to the
unrelated mutex/kick/pend transport wrapper at `0x40128c7c`, whose four call
sites do not include this driver's call site (`0x4002dd74`) or anywhere
inside it.

Like `emu/edma.py`'s `TxChannel`, the interrupt is never raised from inside
the `UC_HOOK_MEM_WRITE` hook itself (Unicorn cannot safely redirect PC from a
memory hook, and interrupts could not be taken there on real hardware
either, since the firmware masks interrupts around the SERQ write in the
comparable UART8 case). It is queued (`_pending_vector`) and delivered from
`service()`, at a chunk boundary where interrupts are enabled and the IPL is
known -- see "Lockstep interface" below.

## Peer interface

    peer.exchange(tx: bytes) -> bytes   # same length as tx

Called once per completed frame, with the raw bytes eDMA `TX_CHAN` moved
out of the TX staging buffer; must return exactly `len(tx)` bytes, which are
written into `RX_CHAN`'s destination the same way. Two peers are provided:

- `ZeroPeer` (the default): returns `bytes(len(tx))`. This is what the
  unmodeled hardware already implies -- an untouched, zero-initialized SRAM
  RX buffer -- so installing this module with the default peer does not
  change behavior versus today beyond actually letting the DSPI2 driver's
  call site return (see the module's `install()` for the opt-in gate).
- `RecordingPeer(inner=None, counter=None)`: forwards to `inner.exchange()`
  (default `ZeroPeer()`) and additionally appends `(instr_count, tx_bytes)`
  to `.frames` for every call, where `instr_count = counter()` if `counter`
  is given (a caller-supplied zero-arg callable, e.g. `lambda: st['n']` from
  `emu/longrun.py`'s stats dict) else `None`. This module has no notion of
  an instruction clock of its own -- see the next section -- so the counter
  is threaded in by whoever wires the model up, the same way `Ssi0Dma` is
  handed `instr_per_sec` rather than measuring it.

## Lockstep interface for a future SHARC stepper

`emu.longrun.spin()` accepts `async_events`: objects with `step(done,
remaining=None) -> int | None` (instructions until this source's next
deadline, or `None` for "no deadline of its own") and `service(done)`
(perform whatever is due by instruction count `done`; called at *every*
chunk boundary regardless of what `step()` returned -- see `Pits` and
`Ssi0Dma` in `emu/pit.py` / `emu/ssi.py`). `Dspi2Link` already implements
this pair, purely to deliver its own queued completion vector safely (see
above); its `step()` always returns `None` because the exchange itself is
write-triggered, not clock-paced, so it imposes no deadline of its own.

A future SHARC stepper is a second, richer implementation of the same pair,
added to the same `async_events` list passed to `spin()`, that additionally
tracks a ColdFire-instruction-to-SHARC-cycle ratio and:

- in `step()`, returns a bound on how soon it needs to run again -- driven
  by whatever the SHARC side needs to stay responsive to (a newly-captured
  DSPI2 TX frame it has not yet consumed, SSI0 samples arriving via the
  `peer.rx`/`peer.tx` hook documented in `emu/ssi.py`'s "SSI0 peer hook",
  or its own periodic workload) rather than by any fixed cadence;
- in `service(done)`, advances the SHARC core the corresponding number of
  cycles, consuming any pending input and updating whatever state answers
  the *next* exchange.

It plugs into the two exchange points documented above, not into this
module's internals: construct it once, hand it to `Dspi2Link(m, peer=stepper)`
(implementing `exchange()`) and to `emu.ssi.Ssi0Dma(..., peer=stepper)`
(implementing that module's `rx`/`tx` hook), and add the stepper itself to
`async_events` for its own scheduling. `Dspi2Link` and `Ssi0Dma` stay exactly
as ignorant of what is on the other end of the link as they are today; they
only know the peer/hook protocol. This module does not build that stepper.
"""

# pyright: reportMissingImports=false
from __future__ import annotations

import struct

from unicorn import UC_HOOK_MEM_WRITE
from unicorn.m68k_const import UC_M68K_REG_SR

from emu.dspiframe import (
    E_SG,
    INT_MAJOR,
    RX_CHAN,
    RX_VECTOR,
    TX_CHAN,
    TX_VECTOR,
)
from emu.dspiframe import (
    SR_LINK_IDLE as SR_LINK_IDLE,  # re-exported: emu.longrun imports it from here
)
from emu.edma import (
    ATTR,
    BITER,
    CITER,
    CSR,
    DADDR,
    DLAST,
    DOFF,
    EDMA_BASE,
    NBYTES,
    SADDR,
    SERQ,
    SLAST,
    SOFF,
    TCD_BASE,
)
from emu.pit import interrupt_level

CINT = EDMA_BASE + 0x1C


def _signed(value, bits):
    sign = 1 << (bits - 1)
    return value - (1 << bits) if value & sign else value


class ZeroPeer:
    """Default peer: what the unmodeled hardware already implies.

    An untouched RX destination buffer reads as whatever it was
    zero-initialized to, so returning `bytes(len(tx))` keeps that behavior
    once `Dspi2Link` starts actually writing the buffer.
    """

    def exchange(self, tx):
        return bytes(len(tx))


class RecordingPeer:
    """Wraps another peer (default `ZeroPeer`) and records every TX frame.

    `counter`, if given, is a zero-arg callable returning the current
    instruction count (e.g. `lambda: st['n']`); recorded as `None` when not
    supplied, since this module has no instruction clock of its own.
    """

    def __init__(self, inner=None, counter=None):
        self.inner = inner if inner is not None else ZeroPeer()
        self.counter = counter
        self.frames = []  # [(instr_count_or_None, tx_bytes), ...]

    def exchange(self, tx):
        tx_bytes = bytes(tx)
        self.frames.append((self.counter() if self.counter else None, tx_bytes))
        return self.inner.exchange(tx_bytes)


class Dspi2Link:
    """DSPI2 + eDMA 28/29 model. See the module docstring for the mechanism."""

    def __init__(
        self, m, peer=None, tx_chan=TX_CHAN, rx_chan=RX_CHAN, raise_completion=True
    ):
        self.m = m
        self.peer = peer if peer is not None else ZeroPeer()
        self.tx_chan, self.rx_chan = tx_chan, rx_chan
        self.raise_completion = raise_completion

        self._tx_ready = None  # captured TX bytes, waiting for RX to arm
        self._rx_armed = False  # RX armed, waiting for TX bytes
        self._pending_vector = {}  # vector -> True, queued for service()

        self.frames = 0
        self.tx_bytes = 0

        m.uc.hook_add(UC_HOOK_MEM_WRITE, self._on_serq, begin=SERQ, end=SERQ)
        m.uc.hook_add(UC_HOOK_MEM_WRITE, self._on_cint, begin=CINT, end=CINT)

    # -- TCD field access ----------------------------------------------------
    def _tcd(self, chan):
        return TCD_BASE + chan * 0x20

    def _u32(self, chan, off):
        return struct.unpack(">I", self.m.uc.mem_read(self._tcd(chan) + off, 4))[0]

    def _u16(self, chan, off):
        return struct.unpack(">H", self.m.uc.mem_read(self._tcd(chan) + off, 2))[0]

    def _w32(self, chan, off, value):
        self.m.uc.mem_write(
            self._tcd(chan) + off, struct.pack(">I", value & 0xFFFFFFFF)
        )

    def _w16(self, chan, off, value):
        self.m.uc.mem_write(self._tcd(chan) + off, struct.pack(">H", value & 0xFFFF))

    # -- SERQ / CINT hooks -----------------------------------------------------
    def _on_serq(self, uc, access, addr, size, value, data):
        if size != 1 or value & 0x80:
            return
        channels = (self.tx_chan, self.rx_chan) if value & 0x40 else (value & 0x3F,)
        for chan in channels:
            if chan == self.tx_chan:
                self._tx_ready = self._capture(self.tx_chan)
            elif chan == self.rx_chan:
                self._rx_armed = True
        self._maybe_exchange()

    def _on_cint(self, uc, access, addr, size, value, data):
        if size != 1:
            return
        channels = (self.tx_chan, self.rx_chan) if value & 0x40 else (value & 0x3F,)
        for chan in channels:
            self._pending_vector.pop(chan, None)

    def _maybe_exchange(self):
        if self._tx_ready is None or not self._rx_armed:
            return
        tx = self._tx_ready
        rx = self.peer.exchange(tx)
        if len(rx) != len(tx):
            raise ValueError(
                "Dspi2Link peer returned %d bytes for a %d-byte frame"
                % (len(rx), len(tx))
            )
        self._deliver(self.rx_chan, rx)
        self._tx_ready = None
        self._rx_armed = False
        self.frames += 1
        self.tx_bytes += len(tx)
        if self._int_major(self.tx_chan):
            self._pending_vector[self.tx_chan] = True
        if self._int_major(self.rx_chan):
            self._pending_vector[self.rx_chan] = True

    def _int_major(self, chan):
        return bool(self._u16(chan, CSR) & INT_MAJOR)

    # -- major-loop transfer, eager and TCD-driven (see module docstring) ----
    def _capture(self, chan):
        """Run chan's whole major loop now, reading SOURCE. -> bytes."""
        citer = self._u16(chan, CITER) & 0x7FFF
        biter = self._u16(chan, BITER) & 0x7FFF
        if not citer:
            return b""
        attr = self._u16(chan, ATTR)
        elem = 1 << (attr & 0x7)
        nbytes = self._u32(chan, NBYTES)
        if not nbytes or nbytes % elem:
            raise RuntimeError(
                "Dspi2Link: NBYTES not a multiple of the source element size"
            )
        src = self._u32(chan, SADDR)
        soff = _signed(self._u16(chan, SOFF), 16)
        smod = (attr >> 11) & 0x1F
        mask = (1 << smod) - 1 if smod else 0
        base = src & ~mask if mask else 0

        out = bytearray()
        for _ in range(citer):
            for _ in range(nbytes // elem):
                word = self.m.uc.mem_read(src, elem)
                if elem == 4:
                    # A 4-byte source element is DSPI2's own PUSHR-formatted
                    # SRAM entry: upper 16 bits are the driver's CONT/PCS/EOQ
                    # command tag, consumed inside the DSPI2 peripheral and
                    # never shifted onto the wire; only the low 16 bits
                    # (TXDATA) reach the SHARC. Confirmed both by
                    # docs/findings/04-coldfire-dsp-link.md ("Each PUSHR
                    # entry is 0x8001xxxx") and by disassembling
                    # FUN_400cd2bc's frame builder directly: `move.w
                    # #$8001,(a0)` then `move.w (a2)+,-$2(a0)` -- the tag at
                    # the low address, the real data 2 bytes later. Keep only
                    # the data half, so `peer.exchange()` sees the logical
                    # payload the SHARC actually receives, at the same byte
                    # width as the RX channel's plain (untagged) buffer.
                    out += bytes(word[2:4])
                else:
                    out += bytes(word)
                src = (src + soff) & 0xFFFFFFFF
                if mask:
                    src = base | (src & mask)
        src = (src + _signed(self._u32(chan, SLAST), 32)) & 0xFFFFFFFF
        if mask:
            src = base | (src & mask)
        self._w32(chan, SADDR, src)
        self._reload(chan, biter)
        return bytes(out)

    def _deliver(self, chan, data):
        """Run chan's whole major loop now, writing `data` to DEST."""
        citer = self._u16(chan, CITER) & 0x7FFF
        biter = self._u16(chan, BITER) & 0x7FFF
        if not citer:
            return
        attr = self._u16(chan, ATTR)
        elem = 1 << ((attr >> 8) & 0x7)
        nbytes = self._u32(chan, NBYTES)
        if not nbytes or nbytes % elem:
            raise RuntimeError(
                "Dspi2Link: NBYTES not a multiple of the dest element size"
            )
        expect = citer * (nbytes // elem) * elem
        if len(data) != expect:
            raise ValueError(
                "Dspi2Link: peer frame is %d bytes, TCD%d expects %d"
                % (len(data), chan, expect)
            )
        dst = self._u32(chan, DADDR)
        doff = _signed(self._u16(chan, DOFF), 16)
        # DMOD (dest modulo wraparound): bits [7:3] of ATTR by the standard
        # eDMA convention this repo has not independently exercised for a
        # destination side (both existing models -- UART8 TX, SSI0 -- only
        # ever use source-side SMOD). Supported defensively; the RX buffer
        # this feeds is documented as a flat, fixed-size SRAM block, so DMOD
        # is expected to read 0 in practice.
        dmod = (attr >> 3) & 0x1F
        mask = (1 << dmod) - 1 if dmod else 0
        base = dst & ~mask if mask else 0

        pos = 0
        for _ in range(citer):
            for _ in range(nbytes // elem):
                self.m.uc.mem_write(dst, data[pos : pos + elem])
                pos += elem
                dst = (dst + doff) & 0xFFFFFFFF
                if mask:
                    dst = base | (dst & mask)
        dst = (dst + _signed(self._u32(chan, DLAST), 32)) & 0xFFFFFFFF
        if mask:
            dst = base | (dst & mask)
        self._w32(chan, DADDR, dst)
        self._reload(chan, biter)

    def _reload(self, chan, biter):
        """Major-loop completion: scatter-gather reload, or CITER<-BITER."""
        csr = self._u16(chan, CSR)
        if csr & E_SG:
            pointer = self._u32(chan, DLAST)
            if pointer & 0x1F:
                raise RuntimeError(
                    "Dspi2Link: scatter/gather pointer is not 32-byte aligned"
                )
            descriptor = bytes(self.m.uc.mem_read(pointer, 0x20))
            self.m.uc.mem_write(self._tcd(chan), descriptor)
        else:
            self._w16(chan, CITER, biter)

    # -- lockstep interface (emu.longrun.spin's async_events) ----------------
    def step(self, done, remaining=None):
        """No deadline of its own: the exchange is write-triggered, not
        clock-paced. -> None, always. See the module docstring."""
        return None

    def service(self, done):
        """Deliver any queued completion vector(s) the IPL currently allows.

        Never raises from the SERQ write hook itself -- see the module
        docstring and `emu.edma.TxChannel.deliver`, the precedent for this
        two-step queue/deliver split.
        """
        if not self.raise_completion or not self._pending_vector:
            return
        for chan, vector in ((self.tx_chan, TX_VECTOR), (self.rx_chan, RX_VECTOR)):
            if chan not in self._pending_vector:
                continue
            level = interrupt_level(self.m, vector)
            if level is None:
                continue
            sr = self.m.uc.reg_read(UC_M68K_REG_SR)
            if ((sr >> 8) & 0x07) >= level:
                continue
            if self.m.raise_vector(vector, level=level):
                del self._pending_vector[chan]


def install(m, peer=None, tx_chan=TX_CHAN, rx_chan=RX_CHAN, raise_completion=True):
    """Model the DSPI2/eDMA 28/29 SHARC link. -> the Dspi2Link.

    Not wired into `emu.longrun.build()` by default (see `dspi2_peer=` there);
    installing this always requires an explicit peer or accepts `ZeroPeer`,
    matching how `emu/edma.py` and `emu/ssi.py`'s models are opt-in.
    """
    return Dspi2Link(
        m,
        peer=peer,
        tx_chan=tx_chan,
        rx_chan=rx_chan,
        raise_completion=raise_completion,
    )
