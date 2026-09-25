"""emu/sharc_peer.py: the client half of the tools/sharc_proc.py protocol.

Most tests here launch the real server as a subprocess -- fast (process
startup only, no firmware) since `--backend stub` needs nothing but the
standard library. `command=[sys.executable, SERVER_PATH]` is used throughout
instead of the default PyPy-preferring launch, so these do not depend on a
PyPy interpreter being installed; `test_default_command_*` cover that
selection logic separately with `shutil.which` mocked out, and one slow test
exercises the real default (PyPy-preferring) launch end to end.

Run with: uv run --with pytest python -m pytest tests/test_sharc_peer.py -q
"""

import os
import shutil
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emu.sharc_peer import (  # noqa: E402
    SharcServerDied,
    SharcServerError,
    SharcServerPeer,
    default_command,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER_PATH = os.path.join(ROOT, "tools", "sharc_proc.py")
CPYTHON_COMMAND = [sys.executable, SERVER_PATH]


def stub_peer(**kwargs):
    return SharcServerPeer(command=CPYTHON_COMMAND, **kwargs)


class DefaultCommandTest(unittest.TestCase):
    def test_prefers_pypy_on_path(self):
        with patch(
            "shutil.which",
            side_effect=lambda name: (
                "/usr/bin/pypy3.11" if name == "pypy3.11" else None
            ),
        ):
            self.assertEqual(
                default_command("server.py"), ["/usr/bin/pypy3.11", "server.py"]
            )

    def test_falls_back_to_uv_managed_pypy(self):
        def which(name):
            return "/usr/bin/uv" if name == "uv" else None

        class Probe:
            returncode = 0
            stdout = "/opt/uv/pypy3.11\n"

        with (
            patch("shutil.which", side_effect=which),
            patch("subprocess.run", return_value=Probe()),
        ):
            self.assertEqual(
                default_command("server.py"),
                [
                    "/usr/bin/uv",
                    "run",
                    "--no-project",
                    "--python",
                    "pypy3.11",
                    "python",
                    "server.py",
                ],
            )

    def test_falls_back_to_cpython_when_neither_found(self):
        with patch("shutil.which", return_value=None):
            self.assertEqual(
                default_command("server.py"), [sys.executable, "server.py"]
            )

    def test_prefer_cpython_always_uses_sys_executable(self):
        with patch("shutil.which", return_value="/usr/bin/pypy3.11"):
            self.assertEqual(
                default_command("server.py", prefer="cpython"),
                [sys.executable, "server.py"],
            )

    def test_rejects_unknown_prefer(self):
        with self.assertRaises(ValueError):
            default_command("server.py", prefer="jython")


class SharcServerPeerStubTest(unittest.TestCase):
    def setUp(self):
        self.peer = stub_peer(image_path="blob.bin", image_sha256="ab" * 16)
        self.addCleanup(self.peer.close)

    def test_hello_records_backend_and_pid(self):
        self.assertEqual(self.peer.backend_name, "StubBackend")
        self.assertIsInstance(self.peer.server_pid, int)
        self.assertNotEqual(self.peer.server_pid, os.getpid())

    def test_exchange_same_length_and_frame_counted(self):
        rx1 = self.peer.exchange(bytes(2748))
        rx2 = self.peer.exchange(bytes(2748))
        self.assertEqual(len(rx1), 2748)
        self.assertEqual(rx1[:4], b"\x00\x00\x00\x01")
        self.assertEqual(rx2[:4], b"\x00\x00\x00\x02")
        self.assertEqual(self.peer.frames, 2)
        self.assertEqual(self.peer.exchange_count, 2)
        self.assertGreater(self.peer.exchange_seconds, 0.0)

    def test_audio_rx_and_tx_round_trip(self):
        rx = self.peer.rx(16)
        self.assertEqual(rx, bytes(16))
        self.peer.tx(bytes(32))  # no exception -> acked
        stats = self.peer.stats()
        self.assertEqual(stats["audio_rx_bytes"], 16)
        self.assertEqual(stats["audio_tx_bytes"], 32)

    def test_advance_reports_executed(self):
        self.assertEqual(self.peer.advance("instructions", 500), 500)
        self.assertEqual(self.peer.stats()["advanced"]["instructions"], 500)

    def test_reset_clears_backend_state(self):
        self.peer.exchange(bytes(4))
        self.peer.reset()
        self.assertEqual(self.peer.stats()["frames"], 0)

    def test_step_returns_none_without_advance_every(self):
        self.assertIsNone(self.peer.step(0))
        self.assertIsNone(self.peer.step(1_000_000, remaining=10))

    def test_close_is_idempotent(self):
        self.peer.close()
        self.peer.close()  # must not raise
        self.assertIsNotNone(self.peer.proc.poll())

    def test_context_manager_closes_on_exit(self):
        with stub_peer() as peer:
            peer.exchange(bytes(4))
        self.assertIsNotNone(peer.proc.poll())


class SharcServerPeerAdvanceCadenceTest(unittest.TestCase):
    def test_step_and_service_send_advance_at_cadence(self):
        peer = stub_peer(advance_every=1000)
        self.addCleanup(peer.close)
        self.assertEqual(peer.step(0), 1000)
        peer.service(500)  # not due yet
        self.assertEqual(peer.stats()["advanced"]["instructions"], 0)
        self.assertEqual(peer.step(500), 500)
        peer.service(1000)  # due now
        self.assertEqual(peer.stats()["advanced"]["instructions"], 1000)
        self.assertEqual(peer.step(1000), 1000)  # next deadline is 2000


class RunnerBackendErrorTest(unittest.TestCase):
    def test_runner_backend_surfaces_a_clear_error(self):
        peer = SharcServerPeer(command=CPYTHON_COMMAND, backend="runner")
        self.addCleanup(peer.close)
        with self.assertRaises(SharcServerError) as ctx:
            peer.reset()
        self.assertIn("not implemented", str(ctx.exception))
        self.assertIn("lane A2", str(ctx.exception))
        # The server itself is still alive and answers the next request.
        with self.assertRaises(SharcServerError):
            peer.exchange(bytes(4))


class ChildDiesTest(unittest.TestCase):
    def test_dying_before_hello_raises_with_stderr_tail(self):
        command = [
            sys.executable,
            "-c",
            "import sys; sys.stderr.write('boom from child\\n'); "
            "sys.stderr.flush(); sys.exit(3)",
        ]
        with self.assertRaises(SharcServerDied) as ctx:
            SharcServerPeer(command=command)
        message = str(ctx.exception)
        self.assertIn("boom from child", message)
        self.assertIn("exit=3", message)

    def test_dying_mid_conversation_raises_with_stderr_tail(self):
        # Answers HELLO like a real server, then exits before EXCHANGE_OK.
        script = textwrap.dedent("""
            import json, struct, sys

            def read_exact(n):
                buf = b""
                while len(buf) < n:
                    chunk = sys.stdin.buffer.read(n - len(buf))
                    if not chunk:
                        sys.exit(2)
                    buf += chunk
                return buf

            length = struct.unpack(">I", read_exact(4))[0]
            read_exact(length)  # the HELLO request; ignored
            reply = json.dumps({"version": 1, "backend": "StubBackend",
                                 "device": "dt2", "pid": 1}).encode("utf-8")
            body = bytes([2]) + reply  # 2 == HELLO_OK
            sys.stdout.buffer.write(struct.pack(">I", len(body)) + body)
            sys.stdout.buffer.flush()
            sys.stderr.write("died after hello\\n")
            sys.stderr.flush()
            sys.exit(1)
        """)
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(script)
            path = f.name
        self.addCleanup(os.unlink, path)
        peer = SharcServerPeer(command=[sys.executable, path])
        with self.assertRaises(SharcServerDied) as ctx:
            peer.exchange(bytes(4))
        self.assertIn("died after hello", str(ctx.exception))


@pytest.mark.slow
class DefaultLaunchTest(unittest.TestCase):
    def test_default_command_launch_works_end_to_end(self):
        if shutil.which("pypy3.11") is None and shutil.which("uv") is None:
            self.skipTest("neither pypy3.11 nor uv on PATH")
        peer = SharcServerPeer()  # prefer='pypy' default
        try:
            rx = peer.exchange(bytes(16))
            self.assertEqual(len(rx), 16)
        finally:
            peer.close()


if __name__ == "__main__":
    unittest.main()
