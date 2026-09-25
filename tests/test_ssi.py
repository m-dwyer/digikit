"""Synthetic SSI0/eDMA event-source tests; no firmware image required."""

# pyright: reportMissingImports=false

import struct
import unittest

from unicorn.m68k_const import UC_M68K_REG_A7, UC_M68K_REG_SR

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
from emu.ssi import (
    AUDIO_SSI0_REQUEST_HZ,
    INTFRCH1,
    RX_CHAN,
    RX_REGISTER,
    RxHandoverPeer,
    Ssi0Dma,
    TX_CHAN,
    TX_REGISTER,
)


class FakeUc:
    def __init__(self):
        self.memory = {}
        self.regs = {}

    def hook_add(self, *args, **kwargs):
        return 1

    def mem_read(self, address, size):
        return bytes(self.memory.get(address + i, 0) for i in range(size))

    def mem_write(self, address, data):
        for i, value in enumerate(data):
            self.memory[address + i] = value

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


class FakePeer:
    """Records rx()/tx() calls; rx() always returns `rx_data` verbatim."""

    def __init__(self, rx_data=b""):
        self.rx_data = rx_data
        self.rx_calls = []  # [nbytes, ...]
        self.tx_calls = []  # [bytes, ...]

    def rx(self, nbytes):
        self.rx_calls.append(nbytes)
        return self.rx_data

    def tx(self, data):
        self.tx_calls.append(data)


def put_tcd(machine, channel, *, source, dest, citer=64, link=0, csr=0):
    base = TCD_BASE + channel * 0x20
    raw = bytearray(0x20)
    struct.pack_into(">I", raw, SADDR, source)
    struct.pack_into(">H", raw, ATTR, 0x0202)
    struct.pack_into(">h", raw, SOFF, 4 if channel == TX_CHAN else 0)
    struct.pack_into(">I", raw, NBYTES, 0x20)
    struct.pack_into(">i", raw, SLAST, 0)
    struct.pack_into(">I", raw, DADDR, dest)
    struct.pack_into(">H", raw, CITER, citer)
    struct.pack_into(">h", raw, DOFF, 0 if channel == TX_CHAN else 4)
    struct.pack_into(">I", raw, DLAST, link)
    struct.pack_into(">H", raw, BITER, citer)
    struct.pack_into(">H", raw, CSR, csr)
    machine.uc.mem_write(base, raw)


class Ssi0DmaTest(unittest.TestCase):
    def make_source(self, request_hz=1, ips=10):
        machine = FakeMachine()
        # INTC1 source 42/vector 170 at level 6; all mask bits clear.
        machine.uc.mem_write(0xFC04C06A, b"\x06")
        machine.uc.reg_write(UC_M68K_REG_SR, 0)
        source = Ssi0Dma(machine, request_hz=request_hz, instr_per_sec=ips)
        return machine, source

    def test_63_64_boundary_and_scatter_gather(self):
        machine, source = self.make_source()
        put_tcd(machine, RX_CHAN, source=RX_REGISTER, dest=0x5000)
        put_tcd(
            machine,
            TX_CHAN,
            source=0x1000,
            dest=TX_REGISTER,
            link=0x3000,
            csr=0x12,
        )
        active = machine.uc.mem_read(TCD_BASE + TX_CHAN * 0x20, 0x20)
        put_tcd(
            machine,
            TX_CHAN,
            source=0x2000,
            dest=TX_REGISTER,
            link=0x3040,
            csr=0x12,
        )
        following = machine.uc.mem_read(TCD_BASE + TX_CHAN * 0x20, 0x20)
        machine.uc.mem_write(0x3000, following)
        machine.uc.mem_write(TCD_BASE + TX_CHAN * 0x20, active)
        machine.uc.mem_write(0x1000, bytes(range(32)) * 64)
        source.enabled.update((RX_CHAN, TX_CHAN))
        source.align(0)

        for request in range(1, 64):
            source.service(request * 10)
        self.assertEqual(source._u16(TX_CHAN, CITER), 1)
        self.assertEqual(source.vector170, 0)
        self.assertEqual(machine.vectors, [])

        source.service(640)
        self.assertEqual(source._u32(TX_CHAN, SADDR), 0x2000)
        self.assertEqual(source.scatter_gathers[TX_CHAN], 1)
        self.assertEqual(source.major_loops[TX_CHAN], 1)
        self.assertEqual(source.vector170, 1)
        self.assertEqual(machine.vectors, [(170, 6)])

    def test_rx_minor_advances_descriptor_without_fabricating_payload(self):
        machine, source = self.make_source()
        before = bytes(range(32))
        machine.uc.mem_write(0x5000, before)
        put_tcd(machine, RX_CHAN, source=RX_REGISTER, dest=0x5000, citer=2)
        source.enabled.add(RX_CHAN)

        source._run_minor(RX_CHAN, capture_tx=False)

        self.assertEqual(machine.uc.mem_read(0x5000, 32), before)
        self.assertEqual(source._u32(RX_CHAN, DADDR), 0x5020)
        self.assertEqual(source._u16(RX_CHAN, CITER), 1)

    def test_rx_peer_hook_supplies_destination_bytes(self):
        machine, source = self.make_source()
        machine.uc.mem_write(0x5000, bytes(32))  # untouched-by-default baseline
        put_tcd(machine, RX_CHAN, source=RX_REGISTER, dest=0x5000, citer=2)
        source.enabled.add(RX_CHAN)
        provided = bytes(range(1, 33))
        peer = FakePeer(rx_data=provided)
        source.peer = peer

        source._run_minor(RX_CHAN, capture_tx=False)

        self.assertEqual(machine.uc.mem_read(0x5000, 32), provided)
        self.assertEqual(peer.rx_calls, [32])

    def test_tx_peer_hook_receives_captured_bytes(self):
        machine, source = self.make_source()
        payload = bytes(range(32))
        machine.uc.mem_write(0x1000, payload)
        put_tcd(machine, TX_CHAN, source=0x1000, dest=TX_REGISTER, citer=2)
        source.enabled.add(TX_CHAN)
        peer = FakePeer(rx_data=b"")
        source.peer = peer

        source._run_minor(TX_CHAN, capture_tx=True)

        self.assertEqual(peer.tx_calls, [payload])

    def test_peer_rx_length_mismatch_raises(self):
        machine, source = self.make_source()
        put_tcd(machine, RX_CHAN, source=RX_REGISTER, dest=0x5000, citer=2)
        source.enabled.add(RX_CHAN)
        source.peer = FakePeer(rx_data=b"\x00" * 4)  # short: expects 32

        with self.assertRaisesRegex(ValueError, "expected 32"):
            source._run_minor(RX_CHAN, capture_tx=False)

    def test_force_bit_is_delivered_only_at_safe_rte_hook(self):
        machine, source = self.make_source()
        # INTC1 source 63/vector 191 at level 5. INTFRCH ignores IMR.
        machine.uc.mem_write(0xFC04C07F, b"\x05")
        machine.uc.reg_write(UC_M68K_REG_A7, 0x8000)
        machine.uc.mem_write(0x8002, b"\x00\x00")
        source._on_intfrch1(machine.uc, None, INTFRCH1, 4, 0x80000000, None)
        self.assertEqual(machine.vectors, [])

        source._on_force_rte(machine.uc, 0, 0, None)
        source._on_force_rte(machine.uc, 0, 0, None)
        self.assertEqual(machine.vectors, [(191, 5)])

        source._on_intfrch1(machine.uc, None, INTFRCH1, 4, 0, None)
        self.assertFalse(source.force_asserted)

    def test_force_waits_when_restored_cpu_ipl_blocks_it(self):
        machine, source = self.make_source()
        machine.uc.mem_write(0xFC04C07F, b"\x05")
        machine.uc.reg_write(UC_M68K_REG_A7, 0x8000)
        machine.uc.mem_write(0x8002, b"\x25\x00")  # restored IPL 5
        source._on_intfrch1(machine.uc, None, INTFRCH1, 4, 0x80000000, None)

        source._on_force_rte(machine.uc, 0, 0, None)
        self.assertEqual(machine.vectors, [])
        self.assertTrue(source.force_asserted)

        machine.uc.reg_write(UC_M68K_REG_SR, 0x2000)
        source.service(1)
        self.assertEqual(machine.vectors, [(191, 5)])

    def test_checkpoint_round_trip_and_rate_mismatch(self):
        machine, original = self.make_source(request_hz=48000, ips=4680000)
        original.enabled.update((RX_CHAN, TX_CHAN))
        original.align(123)
        original.requests = 7
        state = original.checkpoint_state()

        restored = Ssi0Dma(machine, request_hz=48000, instr_per_sec=1)
        restored.restore_checkpoint_state(state)
        self.assertEqual(restored.checkpoint_state(), state)

        mismatch = Ssi0Dma(machine, request_hz=44100, instr_per_sec=4680000)
        with self.assertRaisesRegex(RuntimeError, "request-rate mismatch"):
            mismatch.restore_checkpoint_state(state)

    def test_legacy_arm_validates_observed_peripheral_ends(self):
        machine, source = self.make_source()
        put_tcd(machine, RX_CHAN, source=RX_REGISTER, dest=0x5000, csr=0x10)
        put_tcd(machine, TX_CHAN, source=0x1000, dest=TX_REGISTER, csr=0x12)
        source.arm_legacy()
        self.assertEqual(source.enabled, {RX_CHAN, TX_CHAN})

        put_tcd(machine, TX_CHAN, source=0x1000, dest=0xDEADBEEF, csr=0x12)
        with self.assertRaisesRegex(RuntimeError, "TCD50 shape mismatch"):
            source.arm_legacy()

        put_tcd(machine, TX_CHAN, source=0x1000, dest=TX_REGISTER, csr=0x12)
        machine.uc.mem_write(
            TCD_BASE + TX_CHAN * 0x20 + NBYTES, struct.pack(">I", 4)
        )
        with self.assertRaisesRegex(RuntimeError, "TCD50 shape mismatch"):
            source.arm_legacy()


class RxHandoverPeerTest(unittest.TestCase):
    def test_marker_written_only_at_major_loop_start(self):
        machine = FakeMachine()
        put_tcd(machine, RX_CHAN, source=RX_REGISTER, dest=0x5000, citer=64)
        peer = RxHandoverPeer(machine, channel=RX_CHAN)

        first = peer.rx(32)
        self.assertEqual(struct.unpack(">I", first[:4])[0], 0x007FFFFF)
        self.assertEqual(first[4:], bytes(28))

        # Simulate the mid-major-loop state _run_minor would leave behind:
        # CITER decremented, BITER unchanged.
        machine.uc.mem_write(TCD_BASE + RX_CHAN * 0x20 + CITER, struct.pack(">H", 63))
        second = peer.rx(32)
        self.assertEqual(second, bytes(32))

    def test_marker_reappears_after_reload_survives_resume(self):
        machine = FakeMachine()
        put_tcd(machine, RX_CHAN, source=RX_REGISTER, dest=0x5000, citer=64)
        peer = RxHandoverPeer(machine, channel=RX_CHAN)

        # A resumed checkpoint may hand the peer a fresh instance mid-major
        # loop; only the *reload* (CITER restored to BITER), not object
        # construction order, decides where the marker lands.
        machine.uc.mem_write(TCD_BASE + RX_CHAN * 0x20 + CITER, struct.pack(">H", 5))
        self.assertEqual(peer.rx(32), bytes(32))
        machine.uc.mem_write(TCD_BASE + RX_CHAN * 0x20 + CITER, struct.pack(">H", 64))
        self.assertEqual(struct.unpack(">I", peer.rx(32)[:4])[0], 0x007FFFFF)

    def test_integrates_with_run_minor_major_loop_boundary(self):
        machine, source = Ssi0DmaTest().make_source()
        machine.uc.mem_write(0x5000, bytes(32))
        put_tcd(machine, RX_CHAN, source=RX_REGISTER, dest=0x5000, citer=2)
        source.enabled.add(RX_CHAN)
        source.peer = RxHandoverPeer(machine, channel=RX_CHAN)

        source._run_minor(RX_CHAN, capture_tx=False)  # first of 2: marker
        self.assertEqual(
            struct.unpack(">I", machine.uc.mem_read(0x5000, 4))[0], 0x007FFFFF
        )
        source._run_minor(RX_CHAN, capture_tx=False)  # second: no marker
        self.assertEqual(machine.uc.mem_read(0x5020, 4), bytes(4))


class AudioSsi0RequestHzTest(unittest.TestCase):
    def test_matches_documented_block_cadence(self):
        # 96 kHz requests / 64-minor major loop = 1,500 Hz vector 170, the
        # SHARC's own documented block cadence (32 stereo samples/block at
        # 48 kHz). See emu/ssi.py's AUDIO_SSI0_REQUEST_HZ derivation.
        self.assertEqual(AUDIO_SSI0_REQUEST_HZ, 96_000)
        self.assertEqual(AUDIO_SSI0_REQUEST_HZ / 64, 1500)


if __name__ == "__main__":
    unittest.main()
