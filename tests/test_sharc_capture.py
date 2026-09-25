"""emu/sharc_capture.py: capture file format round-trip and CapturingPeer.

No firmware image or Unicorn required -- this only exercises the file
format and the peer contract (emu.dspi2's `.exchange`, emu.ssi's
`.rx`/`.tx`). tools/sharc_capture_run.py, which drives a real Machine, is
exercised by hand against a real snapshot instead (see its own docstring
and tests/test_sharcframe.py's precedent: "capture needs a snapshot").

Run with: uv run python -m pytest tests/test_sharc_capture.py -q
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emu.sharc_capture import (  # noqa: E402
    MAGIC,
    CaptureWriter,
    CapturingPeer,
    load,
)


class CaptureRoundTripTest(unittest.TestCase):
    def _write(self, path, *, kind="idle", frames=3, ssi0=2, counter_start=0):
        writer = CaptureWriter(
            path,
            frame_bytes=2748,
            kind=kind,
            device="dt2",
            source_sha256="deadbeef",
            extra={"note": "test"},
        )
        counter = [counter_start]
        peer = CapturingPeer(writer, counter=lambda: counter[0])
        tx_frames = []
        for i in range(frames):
            tx = bytes([i]) * 2748
            rx = peer.exchange(tx)
            tx_frames.append((tx, rx))
            counter[0] += 1000
        ssi_reqs = []
        for _ in range(ssi0):
            data = peer.rx(32)
            ssi_reqs.append(data)
            peer.tx(b"\x00" * 32)  # not recorded; must not raise
            counter[0] += 10
        writer.close()
        return tx_frames, ssi_reqs

    def test_round_trip_dspi2_and_ssi0(self):
        path = "/tmp/test_sharc_capture_roundtrip.dt2cap"
        try:
            tx_frames, ssi_reqs = self._write(path, kind="note", frames=3, ssi0=2)
            cap = load(path)
            self.assertEqual(cap.frame_bytes, 2748)
            self.assertEqual(cap.kind, "note")
            self.assertEqual(cap.device, "dt2")
            self.assertEqual(cap.source_sha256, "deadbeef")
            self.assertEqual(cap.meta["note"], "test")

            self.assertEqual(len(cap.dspi2_frames), 3)
            for i, frame in enumerate(cap.dspi2_frames):
                self.assertEqual(frame.tx, tx_frames[i][0])
                self.assertEqual(frame.rx, tx_frames[i][1])
                self.assertEqual(len(frame.rx), len(frame.tx))
                self.assertEqual(frame.instr_count, i * 1000)

            self.assertEqual(len(cap.ssi0_rx), 2)
            for i, rec in enumerate(cap.ssi0_rx):
                self.assertEqual(rec.data, ssi_reqs[i])
                self.assertEqual(len(rec.data), 32)
                self.assertEqual(rec.instr_count, 3000 + i * 10)
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_kind_must_be_valid(self):
        with self.assertRaises(ValueError):
            CaptureWriter(
                "/tmp/test_sharc_capture_bad_kind.dt2cap",
                frame_bytes=2748,
                kind="wat",
            )

    def test_kind_play_round_trips(self):
        # tools/sharc_capture_run.py's `--kind play` (presses the panel PLAY
        # button instead of a single TRIG, to start the loaded pattern's own
        # sequencer -- see that module's docstring): CaptureWriter/load must
        # accept and preserve "play" the same way as "idle"/"note".
        path = "/tmp/test_sharc_capture_kind_play.dt2cap"
        try:
            tx_frames, _ = self._write(path, kind="play", frames=2, ssi0=0)
            cap = load(path)
            self.assertEqual(cap.kind, "play")
            self.assertEqual(len(cap.dspi2_frames), 2)
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_counter_none_round_trips_as_none(self):
        path = "/tmp/test_sharc_capture_no_counter.dt2cap"
        try:
            writer = CaptureWriter(path, frame_bytes=2748, kind="idle")
            peer = CapturingPeer(writer)  # no counter supplied
            peer.exchange(b"\x01\x02\x03\x04")
            writer.close()
            cap = load(path)
            self.assertEqual(len(cap.dspi2_frames), 1)
            self.assertIsNone(cap.dspi2_frames[0].instr_count)
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_bad_magic_rejected(self):
        path = "/tmp/test_sharc_capture_bad_magic.dt2cap"
        try:
            with open(path, "wb") as fh:
                fh.write(b"NOTACAP\n")
            with self.assertRaises(ValueError):
                load(path)
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_magic_is_stable(self):
        # Pinned: changing this breaks every capture already on disk.
        self.assertEqual(MAGIC, b"DT2CAP1\n")

    def test_dspi1_call_round_trips(self):
        # tools/sharc_capture_run.py's observational hook on FUN_400cd48a
        # (docs/findings/04-coldfire-dsp-link.md, "eDMA and DSPI transfer
        # inventory"): records the call's own arguments only.
        path = "/tmp/test_sharc_capture_dspi1.dt2cap"
        try:
            writer = CaptureWriter(path, frame_bytes=2748, kind="idle")
            writer.write_dspi1_call(123, 0x300, 0x4FE7A340, 0x42948B04)
            writer.write_dspi1_call(None, 0x10, 0, 0)
            writer.close()
            cap = load(path)
            self.assertEqual(len(cap.dspi1_calls), 2)
            first, second = cap.dspi1_calls
            self.assertEqual(first.instr_count, 123)
            self.assertEqual(first.length, 0x300)
            self.assertEqual(first.src, 0x4FE7A340)
            self.assertEqual(first.callback, 0x42948B04)
            self.assertIsNone(second.instr_count)
            self.assertEqual(second.length, 0x10)
            self.assertEqual(second.src, 0)
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_mem_write_round_trips(self):
        # tools/sharc_capture_run.py's --watch-mem hook (e.g. the FlexBus
        # 0x8c000000 window): address plus the bytes actually written.
        path = "/tmp/test_sharc_capture_memwrite.dt2cap"
        try:
            writer = CaptureWriter(path, frame_bytes=2748, kind="idle")
            writer.write_mem_write(500, 0x8C000002, b"\xff\x81")
            writer.write_mem_write(600, 0x8C00000A, b"\x80")
            writer.close()
            cap = load(path)
            self.assertEqual(len(cap.mem_writes), 2)
            self.assertEqual(cap.mem_writes[0].address, 0x8C000002)
            self.assertEqual(cap.mem_writes[0].data, b"\xff\x81")
            self.assertEqual(cap.mem_writes[1].address, 0x8C00000A)
            self.assertEqual(cap.mem_writes[1].data, b"\x80")
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_old_capture_without_new_record_types_still_loads(self):
        # Format-extension contract (see the module docstring): a capture
        # written before REC_DSPI1_CALL/REC_MEM_WRITE existed has none, and
        # must still load with empty lists, not an error.
        path = "/tmp/test_sharc_capture_old_format.dt2cap"
        try:
            self._write(path, kind="idle", frames=1, ssi0=0)
            cap = load(path)
            self.assertEqual(cap.dspi1_calls, [])
            self.assertEqual(cap.mem_writes, [])
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_mixed_record_types_preserve_order_per_type(self):
        # A real run interleaves DSPI2 frames, DSPI1 calls and memory
        # writes; each type's own list must come back in the order written.
        path = "/tmp/test_sharc_capture_mixed.dt2cap"
        try:
            writer = CaptureWriter(path, frame_bytes=4, kind="play")
            peer = CapturingPeer(writer, counter=lambda: 0)
            peer.exchange(b"\x01\x02\x03\x04")
            writer.write_dspi1_call(10, 4, 0, 0)
            writer.write_mem_write(20, 0x8C000002, b"\x00")
            peer.exchange(b"\x05\x06\x07\x08")
            writer.write_mem_write(30, 0x8C00000A, b"\x80")
            writer.close()
            cap = load(path)
            self.assertEqual(len(cap.dspi2_frames), 2)
            self.assertEqual(cap.dspi2_frames[1].tx, b"\x05\x06\x07\x08")
            self.assertEqual(len(cap.dspi1_calls), 1)
            self.assertEqual(len(cap.mem_writes), 2)
            self.assertEqual(cap.mem_writes[1].address, 0x8C00000A)
        finally:
            if os.path.exists(path):
                os.remove(path)


class CapturingPeerContractTest(unittest.TestCase):
    """CapturingPeer must satisfy the same shapes emu.dspi2/emu.ssi expect
    of any peer: exchange() returns exactly len(tx) bytes, rx() returns
    exactly nbytes, both never invent nonzero data."""

    def test_exchange_never_invents_data(self):
        writer = CaptureWriter(
            "/tmp/test_sharc_capture_peer_exchange.dt2cap", frame_bytes=8, kind="idle"
        )
        try:
            peer = CapturingPeer(writer)
            tx = b"\xff" * 8
            rx = peer.exchange(tx)
            self.assertEqual(len(rx), len(tx))
            self.assertEqual(rx, bytes(8))
        finally:
            writer.close()
            os.remove(writer.path)

    def test_rx_never_invents_data(self):
        writer = CaptureWriter(
            "/tmp/test_sharc_capture_peer_rx.dt2cap", frame_bytes=8, kind="idle"
        )
        try:
            peer = CapturingPeer(writer)
            data = peer.rx(16)
            self.assertEqual(len(data), 16)
            self.assertEqual(data, bytes(16))
        finally:
            writer.close()
            os.remove(writer.path)


if __name__ == "__main__":
    unittest.main()
