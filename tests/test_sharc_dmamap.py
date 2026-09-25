"""Tests for tools/sharc_dmamap.py.

Two groups, the same convention tests/test_sharc_replay.py uses: pure
helpers (the CLI's own key=value parsers, MmrWrite's own to_json/__str__)
need no firmware and always run; trace_mmr_writes() itself needs the real
DT2 1.16 SHARC+ image bytes (out/sections/dt2-1.16/section_7_BLOB.bin --
Elektron's copyright, never committed here) and is skipped without them.
"""

import os
import pathlib
import sys
import unittest

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools"))
import sharc_dmamap as dm  # noqa: E402
import sharc_harness as h  # noqa: E402
import sharc_run as sr  # noqa: E402

DT2_116_BLOB = pathlib.Path("out/sections/dt2-1.16/section_7_BLOB.bin")


class ParseRegsTest(unittest.TestCase):
    def test_empty_spec(self):
        self.assertEqual(dm._parse_regs(""), {})

    def test_single_pair(self):
        self.assertEqual(dm._parse_regs("R4=0xabc"), {"R4": 0xABC})

    def test_multiple_pairs_and_whitespace(self):
        self.assertEqual(dm._parse_regs(" R4=0xabc , R8=2 "), {"R4": 0xABC, "R8": 2})


class ParsePokesTest(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(dm._parse_pokes([]), {})

    def test_hex_address_and_value(self):
        self.assertEqual(
            dm._parse_pokes(["0x31089400=0", "0x1000=0xff"]),
            {0x31089400: 0, 0x1000: 0xFF},
        )


class ParseProvisionalTest(unittest.TestCase):
    def test_empty(self):
        forms, interpretations = dm._parse_provisional([])
        self.assertEqual(forms, [])
        self.assertEqual(interpretations, {})

    def test_one_entry(self):
        forms, interpretations = dm._parse_provisional(["21p_undoc16=nop"])
        self.assertEqual(forms, ["21p_undoc16"])
        self.assertEqual(interpretations, {"21p_undoc16": "nop"})


class MmrWriteTest(unittest.TestCase):
    def test_to_json_round_trips_fields(self):
        write = dm.MmrWrite(
            0x1C1000, 0x3102D200, None, 0x2641B0, "DMA27 (SPI2 RX) DSCPTR_NXT"
        )
        payload = write.to_json()
        self.assertEqual(payload["pc_sw"], 0x1C1000)
        self.assertEqual(payload["address"], 0x3102D200)
        self.assertIsNone(payload["old_value"])
        self.assertEqual(payload["new_value"], 0x2641B0)
        self.assertEqual(payload["name"], "DMA27 (SPI2 RX) DSCPTR_NXT")

    def test_str_shows_unknown_old_value_as_question_mark(self):
        write = dm.MmrWrite(
            0x1C1000, 0x3102D200, None, 0x2641B0, "DMA27 (SPI2 RX) DSCPTR_NXT"
        )
        text = str(write)
        self.assertIn("?", text)
        self.assertIn("0x2641b0", text)

    def test_str_shows_old_value_when_known(self):
        write = dm.MmrWrite(0x1C1000, 0x3102D200, 0x0, 0x2641B0, None)
        text = str(write)
        self.assertIn("0x0", text)
        self.assertNotIn("(", text)  # no name suffix when name is None


@pytest.mark.slow
@unittest.skipUnless(DT2_116_BLOB.exists(), "DT2 1.16 firmware bytes are not available")
class TraceMmrWritesTest(unittest.TestCase):
    """FUN_1c7bd4 (the SPI2 driver's own descriptor-ring setup, called from
    FUN_1c7ff9 at 0x1c80a2 with R4=0xabc -- HANDOVER-2026-09-26-sharc-audio.md's
    "Update (J1, J2, K1 and static reading)") resets and configures DMA
    channels 26/27 (SPI2 TX/RX, tools/sharcimm.py's DMA_CHANNELS table)
    before this project's tracer hits an unmodeled 64-bit ("long") memory
    access inside a nested driver-table-scan helper -- see this module's
    own docstring and this lane's own report for the full trace. This test
    pins that reachable prefix: every MMR write up to the halt, not the
    (still unreached) per-ring DSCPTR_NXT assignment."""

    @classmethod
    def setUpClass(cls):
        memory = h.load_image_memory("dt2-1.16")
        init = h.run_init(memory, "dt2-1.16")
        assert init.ran, "run_init() must succeed for this test to mean anything"
        cls.init = init

    def _runner(self):
        runner = self.init.runner.fresh_call(
            0x1C7BD4, regs={"R4": 0xABC}, diagnose_unknown=True
        )
        dm._apply_pokes(runner, {0x31089400: 0, 0x31089A28: 0})
        return runner

    def test_resets_both_spi2_dma_channels(self):
        writes, halt = dm.trace_mmr_writes(self._runner(), limit=20_000)
        self.assertIsInstance(halt, sr.Halt)
        addresses = {w.address for w in writes}
        # DSCPTR_NXT for DMA26 (SPI2 TX) and DMA27 (SPI2 RX) -- see
        # tools/sharcimm.py's DMA_REGS/DMA_CHANNELS tables.
        self.assertIn(0x3102D200, addresses)  # DMA26 DSCPTR_NXT
        self.assertIn(0x3102D280, addresses)  # DMA27 DSCPTR_NXT
        for address in (0x3102D200, 0x3102D280):
            names = [w.name for w in writes if w.address == address]
            self.assertTrue(names and names[0] and "SPI2" in names[0])

    def test_configures_spi2_register_block(self):
        writes, _ = dm.trace_mmr_writes(self._runner(), limit=20_000)
        by_address = {w.address: w for w in writes}
        # SPI2 CTL (tools/sharcimm.py's SPI_REGS offset 0x04 inside the
        # 0x31030000 SPI2 block) -- the driver's own slave-mode enable word.
        self.assertIn(0x31030004, by_address)
        self.assertEqual(by_address[0x31030004].new_value, 0x50)
        self.assertEqual(by_address[0x31030004].name, "SPI2 CTL")

    def test_channel_resets_precede_register_config_in_execution_order(self):
        """trace_mmr_writes() records writes as they actually execute, not
        by address: the DMA26/27 channel-reset loop (0xb8c398-0xb8c3cf, a
        library helper at a HIGHER pc) runs before the SPI2 register-block
        config (0x1c8954-0x1c89da, LOWER pc, inside FUN_1c7bd4's own
        caller) -- so program order here is the opposite of address order,
        which is exactly why this dict diff (not a Watchpoint keyed by
        address range) is needed."""
        writes, _ = dm.trace_mmr_writes(self._runner(), limit=20_000)
        self.assertTrue(len(writes) > 10)
        first_reset_index = next(
            i for i, w in enumerate(writes) if w.address == 0x3102D200
        )
        ctl_index = next(i for i, w in enumerate(writes) if w.address == 0x31030004)
        self.assertLess(first_reset_index, ctl_index)


if __name__ == "__main__":
    unittest.main()
