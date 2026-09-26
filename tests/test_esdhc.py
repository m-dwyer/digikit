"""Synthetic eSDHC/eDMA data-path coverage; no firmware input required."""

import struct
import unittest

from emu.edma import (
    BITER,
    CITER,
    CSR,
    DADDR,
    NBYTES,
    SADDR,
    SOFF,
    TCD_BASE,
)
from emu.esdhc import BASE, CMDARG, DPSEL, DTDSEL, Esdhc, Card


class FakeUc:
    def __init__(self):
        self.memory = {}

    def hook_add(self, *args, **kwargs):
        return 1

    def mem_read(self, addr, size):
        return bytes(self.memory.get(addr + offset, 0) for offset in range(size))

    def mem_write(self, addr, data):
        for offset, value in enumerate(data):
            self.memory[addr + offset] = value


class FakeMachine:
    def __init__(self):
        self.uc = FakeUc()

    def ensure(self, addr):
        pass


def put16(uc, addr, value):
    uc.mem_write(addr, struct.pack(">H", value))


def put32(uc, addr, value):
    uc.mem_write(addr, struct.pack(">I", value))


def get32(uc, addr):
    return struct.unpack(">I", uc.mem_read(addr, 4))[0]


class EsdhcBulkReadTest(unittest.TestCase):
    def test_cmd18_moves_backing_sector_and_posts_both_data_completions(self):
        machine = FakeMachine()
        image = bytes((i & 0xFF) for i in range(1024))
        cmd_sem, data_sem, dma_sem = 0x50001000, 0x50001010, 0x50001020
        drv_status, destination = 0x50001030, 0x50002000
        model = Esdhc(
            machine,
            card=Card(image=image),
            drv_status=drv_status,
            cmd_sem=cmd_sem,
            data_sem=data_sem,
            dma_sem=dma_sem,
        )

        channel = 59
        tcd = TCD_BASE + channel * 0x20
        put32(machine.uc, tcd + SADDR, BASE + 0x20)
        put32(machine.uc, tcd + NBYTES, 4)
        put32(machine.uc, tcd + DADDR, destination)
        put16(machine.uc, tcd + CITER, 128)
        put16(machine.uc, tcd + BITER, 128)
        put16(machine.uc, tcd + CSR, 0)
        put32(machine.uc, BASE + CMDARG, 1)

        model._on_serq(None, None, 0, 1, channel, None)
        model._on_serq(None, None, 0, 1, 35, None)
        self.assertEqual(model.armed, channel)
        model._on_xfertyp(
            None, None, 0, 4, (18 << 24) | DPSEL | DTDSEL, None
        )

        self.assertEqual(machine.uc.mem_read(destination, 512), image[512:])
        self.assertEqual(get32(machine.uc, dma_sem), 1)
        self.assertEqual(get32(machine.uc, data_sem), 1)
        self.assertEqual(get32(machine.uc, cmd_sem), 1)
        self.assertEqual(get32(machine.uc, drv_status), 0)
        self.assertEqual(get32(machine.uc, tcd + DADDR), destination + 512)
        csr = struct.unpack(">H", machine.uc.mem_read(tcd + CSR, 2))[0]
        self.assertEqual(csr & 0x80, 0x80)

    def test_command_log_is_off_by_default(self):
        machine = FakeMachine()
        model = Esdhc(machine, card=Card(image=bytes(1024)))
        put32(machine.uc, BASE + CMDARG, 0)
        model._on_xfertyp(None, None, 0, 4, (0 << 24), None)
        self.assertEqual(model.command_log, [])
        # The plain (idx, arg) log is unaffected either way.
        self.assertEqual(model.log, [(0, 0)])

    def test_command_log_records_cmd18_multiblock_read_details(self):
        machine = FakeMachine()
        image = bytes((i & 0xFF) for i in range(1024))
        cmd_sem, data_sem, dma_sem = 0x50001000, 0x50001010, 0x50001020
        tcb_ptr, tcb_value = 0x50001040, 0x60000123
        put32(machine.uc, tcb_ptr, tcb_value)
        model = Esdhc(
            machine, card=Card(image=image), cmd_sem=cmd_sem,
            data_sem=data_sem, dma_sem=dma_sem, command_log=True,
            current_tcb=tcb_ptr,
        )

        channel = 59
        tcd = TCD_BASE + channel * 0x20
        put32(machine.uc, tcd + SADDR, BASE + 0x20)
        put32(machine.uc, tcd + NBYTES, 512)
        put32(machine.uc, tcd + DADDR, 0x50002000)
        put16(machine.uc, tcd + CITER, 1)
        put16(machine.uc, tcd + BITER, 1)
        put16(machine.uc, tcd + CSR, 0)
        put32(machine.uc, BASE + 0x04, (4 << 16) | 512)  # BLKATTR: 4 blocks
        put32(machine.uc, BASE + CMDARG, 1)              # sector 1

        model._on_serq(None, None, 0, 1, channel, None)
        model._on_xfertyp(None, None, 0, 4, (18 << 24) | DPSEL | DTDSEL, None)

        self.assertEqual(len(model.command_log), 1)
        entry = model.command_log[0]
        self.assertEqual(entry['cmd'], 18)
        self.assertEqual(entry['arg'], 1)
        self.assertEqual(entry['direction'], 'read')
        self.assertEqual(entry['blkattr_count'], 4)
        self.assertEqual(entry['blkattr_size'], 512)
        self.assertEqual(entry['dma_bytes'], 512)
        self.assertEqual(entry['dma_channel'], channel)
        self.assertEqual(entry['dma_dst'], 0x50002000)
        self.assertTrue(entry['payload_available'])
        self.assertFalse(entry['truncated'])
        self.assertEqual(set(entry['sems_posted']),
                         {'dma_sem', 'data_sem', 'cmd_sem'})
        self.assertIsNone(entry['pc'])          # FakeUc has no reg_read
        self.assertEqual(entry['task'], tcb_value)

    def test_command_log_flags_data_command_issued_without_arming_dma(self):
        """A data-phase command with no eDMA channel armed for it: dma_sem
        must not be posted (there is nothing to complete), and the log says
        so explicitly rather than silently omitting it."""
        machine = FakeMachine()
        cmd_sem, data_sem, dma_sem = 0x50001000, 0x50001010, 0x50001020
        model = Esdhc(
            machine, card=Card(image=bytes(1024)), cmd_sem=cmd_sem,
            data_sem=data_sem, dma_sem=dma_sem, command_log=True,
        )
        put32(machine.uc, BASE + CMDARG, 0)

        model._on_xfertyp(None, None, 0, 4, (18 << 24) | DPSEL | DTDSEL, None)

        entry = model.command_log[0]
        self.assertNotIn('dma_sem', entry['sems_posted'])
        self.assertTrue(any('NOT posted' in s for s in entry['sems_posted']))
        self.assertIn('data_sem', entry['sems_posted'])
        self.assertIn('cmd_sem', entry['sems_posted'])
        self.assertEqual(get32(machine.uc, dma_sem), 0)
        self.assertEqual(get32(machine.uc, data_sem), 1)

    def test_command_log_flags_stale_unconsumed_semaphore(self):
        """A completion semaphore that is already positive when this command
        would post it means an earlier post was never consumed -- the
        signature of a task that stopped pending, not of a missing post."""
        machine = FakeMachine()
        cmd_sem = 0x50001000
        put32(machine.uc, cmd_sem, 1)   # already posted, never consumed
        model = Esdhc(machine, card=Card(image=bytes(512)), cmd_sem=cmd_sem,
                      command_log=True)
        put32(machine.uc, BASE + CMDARG, 0)

        model._on_xfertyp(None, None, 0, 4, (0 << 24), None)

        entry = model.command_log[0]
        self.assertIn('(cmd_sem already set, not reposted)',
                      entry['sems_posted'])

    def test_cmd25_consumes_host_buffer_and_is_visible_to_cmd18(self):
        machine = FakeMachine()
        data_sem, dma_sem = 0x50001010, 0x50001020
        source = 0x50002000
        payload = bytes((255 - i) & 0xFF for i in range(512))
        model = Esdhc(machine, data_sem=data_sem, dma_sem=dma_sem)

        channel = 59
        tcd = TCD_BASE + channel * 0x20
        machine.uc.mem_write(source, payload)
        put32(machine.uc, tcd + SADDR, source)
        put16(machine.uc, tcd + SOFF, 16)
        put32(machine.uc, tcd + NBYTES, 16)
        put32(machine.uc, tcd + DADDR, BASE + 0x20)
        put16(machine.uc, tcd + CITER, 32)
        put16(machine.uc, tcd + BITER, 32)
        put16(machine.uc, tcd + CSR, 0)
        put32(machine.uc, BASE + CMDARG, 7)

        model._on_serq(None, None, 0, 1, channel, None)
        model._on_xfertyp(None, None, 0, 4, (25 << 24) | DPSEL, None)

        self.assertEqual(model.card.data_for(18, 7, 512), payload)
        self.assertEqual(get32(machine.uc, dma_sem), 1)
        self.assertEqual(get32(machine.uc, data_sem), 1)

        restored = Esdhc(FakeMachine())
        restored.restore_checkpoint_state(model.checkpoint_state())
        self.assertEqual(restored.card.data_for(18, 7, 512), payload)


if __name__ == "__main__":
    unittest.main()
