#!/usr/bin/env python3
"""Generate the two synthetic test WAVs used by tools/plusdrive.py's tests
and by an emulator +Drive image, into samples/.

Nothing under samples/*.wav is committed (see samples/README.md) -- these
are small, fully synthetic, and regenerated on demand instead:

    uv run python tools/gen_test_samples.py

Writes samples/test-sine-1khz.wav (a 1 kHz sine, 48 kHz/16-bit/mono, 0.5 s)
and samples/test-perc-click.wav (a short exponentially-decaying click, same
format, 50 ms) -- the second stands in for a percussive drum sample without
needing a real recording.
"""

import math
import os
import struct
import sys
import wave

SAMPLES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "samples"
)
RATE = 48000


def _write_wav(path, samples):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(b"".join(struct.pack("<h", s) for s in samples))


def sine_1khz(duration=0.5, freq=1000.0, amplitude=0.5):
    n = int(RATE * duration)
    return [
        int(amplitude * 32767 * math.sin(2 * math.pi * freq * i / RATE))
        for i in range(n)
    ]


def perc_click(duration=0.05, freq=180.0, amplitude=0.9):
    n = int(RATE * duration)
    out = []
    for i in range(n):
        t = i / RATE
        env = math.exp(-t * 60.0)
        # A little noise on top of a low thump reads as "percussive", not a
        # pure tone, without needing a real recording.
        noise = (((i * 2654435761) & 0xFFFF) / 65535.0 - 0.5) * 0.6
        s = env * (math.sin(2 * math.pi * freq * t) * 0.7 + noise)
        out.append(int(max(-1.0, min(1.0, s)) * amplitude * 32767))
    return out


def main():
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    sine_path = os.path.join(SAMPLES_DIR, "test-sine-1khz.wav")
    click_path = os.path.join(SAMPLES_DIR, "test-perc-click.wav")
    _write_wav(sine_path, sine_1khz())
    _write_wav(click_path, perc_click())
    for path in (sine_path, click_path):
        print("wrote %s (%d bytes)" % (path, os.path.getsize(path)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
