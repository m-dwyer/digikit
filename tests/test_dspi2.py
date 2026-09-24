"""Synthetic DSPI2/eDMA-28/29 link model tests; no firmware image required.

Run with: uv run --with pytest python -m pytest tests/test_dspi2.py -q
"""

# pyright: reportMissingImports=false

import struct
import unittest

from unicorn import UC_HOOK_MEM_WRITE
from unicorn.m68k_const import UC_M68K_REG_SR

from emu.edma import (
    ATTR,
    BITER,
    CITER,
    CSR,
    DADDR,
    DLAST,
    DOFF,
    NBYTES,
    SADDR,
    SLAST,
    SOFF,
    TCD_BASE,
)
from emu.dspi2 import (
    CINT,
    E_SG,
    RX_CHAN,
    RX_VECTOR,
    TX_CHAN,
    TX_VECTOR,
    Dspi2Link,
    RecordingPeer,
    ZeroPeer,
)


SERQ = 0xFC044018  # emu.edma.SERQ; re-derived here to catch an import drift


class FakeUc:
    """Unlike test_ssi.py's FakeUc, this one actually dispatches
    UC_HOOK_MEM_WRITE hooks registered on a single byte address, since
    Dspi2Link's SERQ/CINT protocol is exercised here through real
    single-byte `mem_write` calls (`serq()`/`cint()` below) rather than by
    calling its hook methods directly.
    """

    def __init__(self):
        self.memory = {}
        self.regs = {}
        self.write_hooks = []  # [(begin, end, callback)]

    def hook_add(self, kind, callback, begin=None, end=None):
        if kind == UC_HOOK_MEM_WRITE:
            self.write_hooks.append((begin, end, callback))
        return 1

    def mem_read(self, address, size):
        return bytes(self.memory.get(address + i, 0) for i in range(size))

    def mem_write(self, address, data):
        for i, value in enumerate(data):
            self.memory[address + i] = value
        if len(data) == 1:
            value = data[0]
            for begin, end, callback in self.write_hooks:
                if begin <= address <= end:
                    callback(self, None, address, 1, value, None)

    def reg_read(self, register):
        return self.regs.get(register, 0)

    def reg_write(self, register, value):
        self.regs[register] = value


class FakeMachine:
    def __init__(self):
        self.uc = FakeUc()
        self.vectors = []

    def raise_vector(self, vector, level=None):
        self.vectors.append((vector, level))
        return True


def put_tcd(machine, channel, *, source, dest, citer=4, biter=None, csr=0,
            soff=2, doff=2, slast=0, dlast=0, attr=0x0101, nbytes=2):
    biter = citer if biter is None else biter
    base = TCD_BASE + channel * 0x20
    raw = bytearray(0x20)
    struct.pack_into(">I", raw, SADDR, source)
    struct.pack_into(">H", raw, ATTR, attr)
    struct.pack_into(">h", raw, SOFF, soff)
    struct.pack_into(">I", raw, NBYTES, nbytes)
    struct.pack_into(">i", raw, SLAST, slast)
    struct.pack_into(">I", raw, DADDR, dest)
    struct.pack_into(">H", raw, CITER, citer)
    struct.pack_into(">h", raw, DOFF, doff)
    struct.pack_into(">I", raw, DLAST, dlast)
    struct.pack_into(">H", raw, BITER, biter)
    struct.pack_into(">H", raw, CSR, csr)
    machine.uc.mem_write(base, raw)


def enable_vector(machine, vector, level):
    """INTC1 (vectors 128-191): ICR byte at base+0x40+source, low nibble = level."""
    source = vector - 128
    machine.uc.mem_write(0xFC04C000 + 0x40 + source, bytes([level]))


def serq(machine, channel):
    machine.uc.mem_write(SERQ, bytes([channel]))


def cint(machine, channel):
    machine.uc.mem_write(CINT, bytes([channel]))


class Dspi2LinkTest(unittest.TestCase):
    def test_default_zero_peer_round_trips_length_and_echoes_zeros(self):
        machine = FakeMachine()
        machine.uc.mem_write(0x1000, bytes(range(1, 9)))  # 4 words, TX source
        put_tcd(machine, TX_CHAN, source=0x1000, dest=0xDEAD)
        put_tcd(machine, RX_CHAN, source=0xBEEF, dest=0x2000)
        link = Dspi2Link(machine, peer=ZeroPeer())

        serq(machine, TX_CHAN)
        serq(machine, RX_CHAN)

        self.assertEqual(link.frames, 1)
        self.assertEqual(machine.uc.mem_read(0x2000, 8), bytes(8))
        # CITER reloaded from BITER after the major loop completes.
        self.assertEqual(link._u16(TX_CHAN, CITER), 4)
        self.assertEqual(link._u16(RX_CHAN, CITER), 4)
        # SADDR/DADDR advanced by citer*elem then SLAST/DLAST (0 here).
        self.assertEqual(link._u32(TX_CHAN, SADDR), 0x1008)
        self.assertEqual(link._u32(RX_CHAN, DADDR), 0x2008)

    def test_rx_armed_before_tx_still_exchanges(self):
        machine = FakeMachine()
        machine.uc.mem_write(0x1000, bytes(range(1, 9)))
        put_tcd(machine, TX_CHAN, source=0x1000, dest=0xDEAD)
        put_tcd(machine, RX_CHAN, source=0xBEEF, dest=0x2000)
        link = Dspi2Link(machine, peer=ZeroPeer())

        serq(machine, RX_CHAN)
        self.assertEqual(link.frames, 0)  # deferred: TX bytes not captured yet
        serq(machine, TX_CHAN)
        self.assertEqual(link.frames, 1)

    def test_custom_peer_echoes_transformed_bytes(self):
        class InvertPeer:
            def exchange(self, tx):
                return bytes(b ^ 0xFF for b in tx)

        machine = FakeMachine()
        machine.uc.mem_write(0x1000, bytes(range(1, 9)))
        put_tcd(machine, TX_CHAN, source=0x1000, dest=0xDEAD)
        put_tcd(machine, RX_CHAN, source=0xBEEF, dest=0x2000)
        link = Dspi2Link(machine, peer=InvertPeer())

        serq(machine, TX_CHAN)
        serq(machine, RX_CHAN)

        self.assertEqual(machine.uc.mem_read(0x2000, 8),
                          bytes(b ^ 0xFF for b in range(1, 9)))

    def test_peer_length_mismatch_raises(self):
        class ShortPeer:
            def exchange(self, tx):
                return tx[:-1]

        machine = FakeMachine()
        machine.uc.mem_write(0x1000, bytes(range(1, 9)))
        put_tcd(machine, TX_CHAN, source=0x1000, dest=0xDEAD)
        put_tcd(machine, RX_CHAN, source=0xBEEF, dest=0x2000)
        link = Dspi2Link(machine, peer=ShortPeer())

        serq(machine, TX_CHAN)
        with self.assertRaisesRegex(ValueError, "7 bytes for a 8-byte frame"):
            serq(machine, RX_CHAN)

    def test_recording_peer_captures_frames_with_instruction_count(self):
        machine = FakeMachine()
        machine.uc.mem_write(0x1000, bytes(range(1, 9)))
        put_tcd(machine, TX_CHAN, source=0x1000, dest=0xDEAD)
        put_tcd(machine, RX_CHAN, source=0xBEEF, dest=0x2000)
        n = {"count": 41}
        peer = RecordingPeer(counter=lambda: n["count"])
        link = Dspi2Link(machine, peer=peer)

        n["count"] = 100
        serq(machine, TX_CHAN)
        serq(machine, RX_CHAN)

        self.assertEqual(peer.frames, [(100, bytes(range(1, 9)))])
        self.assertEqual(link.frames, 1)

    def test_no_vector_when_csr_int_major_unset(self):
        machine = FakeMachine()
        put_tcd(machine, TX_CHAN, source=0x1000, dest=0xDEAD, csr=0)
        put_tcd(machine, RX_CHAN, source=0xBEEF, dest=0x2000, csr=0)
        enable_vector(machine, TX_VECTOR, 5)
        enable_vector(machine, RX_VECTOR, 5)
        link = Dspi2Link(machine, peer=ZeroPeer())

        serq(machine, TX_CHAN)
        serq(machine, RX_CHAN)
        link.service(1)

        self.assertEqual(machine.vectors, [])

    def test_vector_queued_then_delivered_at_service_when_ipl_allows(self):
        machine = FakeMachine()
        put_tcd(machine, TX_CHAN, source=0x1000, dest=0xDEAD, csr=0x0002)
        put_tcd(machine, RX_CHAN, source=0xBEEF, dest=0x2000, csr=0x0002)
        enable_vector(machine, TX_VECTOR, 5)
        enable_vector(machine, RX_VECTOR, 6)
        link = Dspi2Link(machine, peer=ZeroPeer())

        serq(machine, TX_CHAN)
        serq(machine, RX_CHAN)
        # Not raised from inside the write hook itself.
        self.assertEqual(machine.vectors, [])

        machine.uc.reg_write(UC_M68K_REG_SR, 0)
        link.service(1)
        self.assertEqual(sorted(machine.vectors), sorted([(TX_VECTOR, 5), (RX_VECTOR, 6)]))

    def test_vector_delivery_blocked_by_current_ipl_then_retried(self):
        machine = FakeMachine()
        put_tcd(machine, TX_CHAN, source=0x1000, dest=0xDEAD, csr=0x0002)
        put_tcd(machine, RX_CHAN, source=0xBEEF, dest=0x2000, csr=0x0002)
        enable_vector(machine, TX_VECTOR, 5)
        enable_vector(machine, RX_VECTOR, 5)
        link = Dspi2Link(machine, peer=ZeroPeer())

        serq(machine, TX_CHAN)
        serq(machine, RX_CHAN)
        machine.uc.reg_write(UC_M68K_REG_SR, 0x0700)  # IPL 7: blocks everything
        link.service(1)
        self.assertEqual(machine.vectors, [])

        machine.uc.reg_write(UC_M68K_REG_SR, 0)
        link.service(2)
        self.assertEqual(sorted(machine.vectors), sorted([(TX_VECTOR, 5), (RX_VECTOR, 5)]))

    def test_cint_clears_pending_vector(self):
        machine = FakeMachine()
        put_tcd(machine, TX_CHAN, source=0x1000, dest=0xDEAD, csr=0x0002)
        put_tcd(machine, RX_CHAN, source=0xBEEF, dest=0x2000, csr=0x0002)
        enable_vector(machine, TX_VECTOR, 5)
        enable_vector(machine, RX_VECTOR, 5)
        link = Dspi2Link(machine, peer=ZeroPeer())

        serq(machine, TX_CHAN)
        serq(machine, RX_CHAN)
        cint(machine, TX_CHAN)
        machine.uc.reg_write(UC_M68K_REG_SR, 0)
        link.service(1)

        self.assertEqual(machine.vectors, [(RX_VECTOR, 5)])  # only RX; TX was CINT'd

    def test_raise_completion_false_suppresses_vectors_but_still_exchanges(self):
        machine = FakeMachine()
        machine.uc.mem_write(0x1000, bytes(range(1, 9)))
        put_tcd(machine, TX_CHAN, source=0x1000, dest=0xDEAD, csr=0x0002)
        put_tcd(machine, RX_CHAN, source=0xBEEF, dest=0x2000, csr=0x0002)
        enable_vector(machine, TX_VECTOR, 5)
        enable_vector(machine, RX_VECTOR, 5)
        link = Dspi2Link(machine, peer=ZeroPeer(), raise_completion=False)

        serq(machine, TX_CHAN)
        serq(machine, RX_CHAN)
        machine.uc.reg_write(UC_M68K_REG_SR, 0)
        link.service(1)

        self.assertEqual(link.frames, 1)
        self.assertEqual(machine.uc.mem_read(0x2000, 8), bytes(8))
        self.assertEqual(machine.vectors, [])

    def test_step_never_imposes_a_deadline(self):
        machine = FakeMachine()
        link = Dspi2Link(machine, peer=ZeroPeer())
        self.assertIsNone(link.step(0))
        self.assertIsNone(link.step(1_000_000, remaining=5))

    def test_scatter_gather_reload_on_tx(self):
        machine = FakeMachine()
        machine.uc.mem_write(0x1000, bytes(range(1, 9)))
        # Next descriptor at 0x3000 (32-byte aligned), for a fresh 2-word TX.
        put_tcd(machine, TX_CHAN, source=0x1000, dest=0xDEAD, dlast=0x3000, csr=E_SG)
        next_desc = bytearray(0x20)
        struct.pack_into(">I", next_desc, SADDR, 0x9000)
        struct.pack_into(">H", next_desc, ATTR, 0x0101)
        struct.pack_into(">h", next_desc, SOFF, 2)
        struct.pack_into(">I", next_desc, NBYTES, 2)
        struct.pack_into(">H", next_desc, CITER, 2)
        struct.pack_into(">H", next_desc, BITER, 2)
        machine.uc.mem_write(0x3000, bytes(next_desc))
        machine.uc.mem_write(0x9000, bytes([0xAA, 0xBB, 0xCC, 0xDD]))

        link = Dspi2Link(machine, peer=ZeroPeer())
        captured = link._capture(TX_CHAN)

        self.assertEqual(captured, bytes(range(1, 9)))
        # The TCD slot now holds the scatter-gathered descriptor.
        self.assertEqual(link._u32(TX_CHAN, SADDR), 0x9000)
        self.assertEqual(link._u16(TX_CHAN, CITER), 2)


if __name__ == "__main__":
    unittest.main()
