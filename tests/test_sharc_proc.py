"""tools/sharc_proc.py: framing, backends and the dispatch loop, in-process
(no subprocess, no firmware). See tests/test_sharc_peer.py for the
subprocess-level client tests.

Run with: uv run --with pytest python -m pytest tests/test_sharc_proc.py -q
"""

import io
import json
import os
import sys
import unittest
from importlib import import_module

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"),
)

sharc_proc = import_module("sharc_proc")
ADVANCE = sharc_proc.ADVANCE
ADVANCE_OK = sharc_proc.ADVANCE_OK
AUDIO_RX = sharc_proc.AUDIO_RX
AUDIO_RX_OK = sharc_proc.AUDIO_RX_OK
AUDIO_TX = sharc_proc.AUDIO_TX
AUDIO_TX_OK = sharc_proc.AUDIO_TX_OK
ERROR = sharc_proc.ERROR
EXCHANGE = sharc_proc.EXCHANGE
EXCHANGE_OK = sharc_proc.EXCHANGE_OK
HELLO = sharc_proc.HELLO
HELLO_OK = sharc_proc.HELLO_OK
PROTOCOL_VERSION = sharc_proc.PROTOCOL_VERSION
RESET = sharc_proc.RESET
RESET_OK = sharc_proc.RESET_OK
SHUTDOWN = sharc_proc.SHUTDOWN
SHUTDOWN_OK = sharc_proc.SHUTDOWN_OK
STATS = sharc_proc.STATS
STATS_OK = sharc_proc.STATS_OK
RunnerBackend = sharc_proc.RunnerBackend
Server = sharc_proc.Server
StubBackend = sharc_proc.StubBackend
read_message = sharc_proc.read_message
send_json = sharc_proc.send_json
send_message = sharc_proc.send_message


class FramingTest(unittest.TestCase):
    def test_round_trips_raw_payload(self):
        stream = io.BytesIO()
        send_message(stream, EXCHANGE, b"\x01\x02\x03")
        stream.seek(0)
        msg_type, payload = read_message(stream)
        self.assertEqual(msg_type, EXCHANGE)
        self.assertEqual(payload, b"\x01\x02\x03")

    def test_round_trips_json_payload(self):
        stream = io.BytesIO()
        send_json(stream, STATS_OK, {"frames": 3})
        stream.seek(0)
        msg_type, payload = read_message(stream)
        self.assertEqual(msg_type, STATS_OK)
        self.assertEqual(json.loads(payload), {"frames": 3})

    def test_empty_stream_is_clean_eof(self):
        self.assertIsNone(read_message(io.BytesIO(b"")))

    def test_truncated_message_raises(self):
        stream = io.BytesIO()
        send_message(stream, EXCHANGE, b"\x01\x02\x03")
        truncated = io.BytesIO(stream.getvalue()[:-1])
        with self.assertRaises(EOFError):
            read_message(truncated)


class StubBackendTest(unittest.TestCase):
    def test_exchange_same_length_stamped_with_frame_counter(self):
        backend = StubBackend()
        rx1 = backend.exchange(bytes(8))
        rx2 = backend.exchange(bytes(8))
        self.assertEqual(len(rx1), 8)
        self.assertEqual(rx1, b"\x00\x00\x00\x01\x00\x00\x00\x00")
        self.assertEqual(rx2, b"\x00\x00\x00\x02\x00\x00\x00\x00")

    def test_exchange_short_frame_still_same_length(self):
        backend = StubBackend()
        rx = backend.exchange(b"\x01\x02")
        self.assertEqual(rx, b"\x00\x00")  # too short for the counter stamp

    def test_audio_rx_silence_is_zeros(self):
        backend = StubBackend(audio_mode="silence")
        self.assertEqual(backend.audio_rx(16), bytes(16))

    def test_audio_rx_tone_is_deterministic_ramp(self):
        backend = StubBackend(audio_mode="tone")
        first = backend.audio_rx(4)
        second = backend.audio_rx(4)
        self.assertEqual(first, bytes([0, 1, 2, 3]))
        self.assertEqual(second, bytes([4, 5, 6, 7]))

    def test_audio_tx_tracked_in_stats(self):
        backend = StubBackend()
        backend.audio_tx(bytes(10))
        backend.audio_tx(bytes(5))
        self.assertEqual(backend.stats()["audio_tx_bytes"], 15)

    def test_advance_accumulates_per_unit(self):
        backend = StubBackend()
        self.assertEqual(backend.advance("instructions", 100), 100)
        self.assertEqual(backend.advance("samples", 7), 7)
        self.assertEqual(backend.advance("instructions", 50), 50)
        self.assertEqual(
            backend.stats()["advanced"], {"instructions": 150, "samples": 7}
        )

    def test_advance_rejects_unknown_unit(self):
        backend = StubBackend()
        with self.assertRaises(ValueError):
            backend.advance("furlongs", 1)

    def test_reset_clears_counters(self):
        backend = StubBackend()
        backend.exchange(bytes(4))
        backend.audio_tx(bytes(4))
        backend.advance("instructions", 9)
        backend.reset()
        self.assertEqual(
            backend.stats(),
            {
                "backend": "stub",
                "audio_mode": "silence",
                "frames": 0,
                "tx_bytes": 0,
                "audio_rx_bytes": 0,
                "audio_tx_bytes": 0,
                "advanced": {"instructions": 0, "samples": 0},
            },
        )


class RunnerBackendTest(unittest.TestCase):
    def test_every_method_raises_not_implemented_with_a_clear_message(self):
        backend = RunnerBackend(image_path="blob.bin", image_sha256="abc")
        calls = [
            (backend.reset, ()),
            (backend.exchange, (b"\x00",)),
            (backend.audio_rx, (4,)),
            (backend.audio_tx, (b"\x00",)),
            (backend.advance, ("instructions", 1)),
            (backend.stats, ()),
        ]
        for fn, args in calls:
            with self.assertRaises(NotImplementedError) as ctx:
                fn(*args)
            self.assertIn("not implemented", str(ctx.exception))
            self.assertIn("lane A2", str(ctx.exception))


class ServerTest(unittest.TestCase):
    def _server(self, backend=None, device="dt2"):
        instream = io.BytesIO()
        outstream = io.BytesIO()
        server = Server(
            backend or StubBackend(),
            device=device,
            instream=instream,
            outstream=outstream,
            log=io.StringIO(),
        )
        return server, instream, outstream

    def _feed(self, server, instream, *messages):
        """Write `messages` ([(type, json_or_bytes_payload, is_json)]) into
        `instream`, rewind it, then drain every reply -> [(type, payload)]."""
        for msg_type, payload, is_json in messages:
            if is_json:
                send_json(instream, msg_type, payload)
            else:
                send_message(instream, msg_type, payload)
        instream.seek(0)
        replies = []
        while server.handle_one():
            pass
        out = server.outstream
        out.seek(0)
        while True:
            msg = read_message(out)
            if msg is None:
                break
            replies.append(msg)
        return replies

    def test_hello_then_exchange_round_trip(self):
        server, instream, _out = self._server()
        replies = self._feed(
            server,
            instream,
            (
                HELLO,
                {
                    "version": PROTOCOL_VERSION,
                    "image_path": None,
                    "image_sha256": None,
                    "device": "dt2",
                },
                True,
            ),
            (EXCHANGE, bytes(4), False),
        )
        self.assertEqual(replies[0][0], HELLO_OK)
        hello = json.loads(replies[0][1])
        self.assertEqual(hello["version"], PROTOCOL_VERSION)
        self.assertEqual(hello["backend"], "StubBackend")
        self.assertIn("pid", hello)
        self.assertEqual(replies[1], (EXCHANGE_OK, b"\x00\x00\x00\x01"))

    def test_message_before_hello_is_an_error(self):
        server, instream, _out = self._server()
        replies = self._feed(server, instream, (STATS, {}, True))
        self.assertEqual(replies[0][0], ERROR)
        self.assertIn("HELLO", json.loads(replies[0][1])["message"])

    def test_version_mismatch_is_an_error_and_does_not_set_hello_seen(self):
        server, instream, _out = self._server()
        replies = self._feed(
            server,
            instream,
            (
                HELLO,
                {
                    "version": PROTOCOL_VERSION + 1,
                    "image_path": None,
                    "image_sha256": None,
                    "device": "dt2",
                },
                True,
            ),
            (STATS, {}, True),
        )
        self.assertEqual(replies[0][0], ERROR)
        self.assertIn("version mismatch", json.loads(replies[0][1])["message"])
        self.assertEqual(replies[1][0], ERROR)  # still no HELLO accepted

    def test_full_message_table_round_trip(self):
        server, instream, _out = self._server()
        replies = self._feed(
            server,
            instream,
            (
                HELLO,
                {
                    "version": PROTOCOL_VERSION,
                    "image_path": None,
                    "image_sha256": None,
                    "device": "dt2",
                },
                True,
            ),
            (RESET, {}, True),
            (EXCHANGE, bytes(4), False),
            (AUDIO_RX, {"nbytes": 4}, True),
            (AUDIO_TX, bytes(4), False),
            (ADVANCE, {"unit": "instructions", "count": 10}, True),
            (STATS, {}, True),
            (SHUTDOWN, {}, True),
        )
        types = [msg_type for msg_type, _payload in replies]
        self.assertEqual(
            types,
            [
                HELLO_OK,
                RESET_OK,
                EXCHANGE_OK,
                AUDIO_RX_OK,
                AUDIO_TX_OK,
                ADVANCE_OK,
                STATS_OK,
                SHUTDOWN_OK,
            ],
        )
        advance_reply = json.loads(replies[types.index(ADVANCE_OK)][1])
        self.assertEqual(advance_reply["executed"], 10)
        stats_reply = json.loads(replies[types.index(STATS_OK)][1])
        self.assertEqual(stats_reply["frames"], 1)

    def test_runner_backend_error_does_not_kill_the_server(self):
        server, instream, _out = self._server(backend=RunnerBackend())
        replies = self._feed(
            server,
            instream,
            (
                HELLO,
                {
                    "version": PROTOCOL_VERSION,
                    "image_path": None,
                    "image_sha256": None,
                    "device": "dt2",
                },
                True,
            ),
            (RESET, {}, True),
            (STATS, {}, True),
        )
        self.assertEqual(replies[0][0], HELLO_OK)
        self.assertEqual(replies[1][0], ERROR)
        self.assertIn("not implemented", json.loads(replies[1][1])["message"])
        self.assertEqual(replies[2][0], ERROR)  # server kept serving

    def test_unknown_message_type_is_an_error(self):
        server, instream, _out = self._server()
        replies = self._feed(
            server,
            instream,
            (
                HELLO,
                {
                    "version": PROTOCOL_VERSION,
                    "image_path": None,
                    "image_sha256": None,
                    "device": "dt2",
                },
                True,
            ),
            (254, {}, True),
        )
        self.assertEqual(replies[1][0], ERROR)

    def test_exchange_length_mismatch_from_a_bad_backend_is_reported(self):
        class ShortBackend(StubBackend):
            def exchange(self, tx):
                return super().exchange(tx)[:-1]

        server, instream, _out = self._server(backend=ShortBackend())
        replies = self._feed(
            server,
            instream,
            (
                HELLO,
                {
                    "version": PROTOCOL_VERSION,
                    "image_path": None,
                    "image_sha256": None,
                    "device": "dt2",
                },
                True,
            ),
            (EXCHANGE, bytes(4), False),
        )
        self.assertEqual(replies[1][0], ERROR)
        self.assertIn(
            "3 bytes for a 4-byte frame", json.loads(replies[1][1])["message"]
        )


if __name__ == "__main__":
    unittest.main()
