# samples/

WAV files here become the emulated Digitakt II's +Drive: `tools/plusdrive.py
build samples/ -o out/plusdrive/dt2.img` reads every `.wav` file directly in
this folder (not recursively) and writes them into a card image in the
firmware's own +Drive filesystem format
(`docs/findings/14-plus-drive-format.md`), which the emulator can then boot
from (`--card-image out/plusdrive/dt2.img` on the runners that accept it).

**No `.wav` file in this folder is committed** (`.gitignore` excludes
`samples/*.wav`), including the two test files below -- generate them
instead of expecting them to already be here:

```
uv run python tools/gen_test_samples.py
```

That writes:

- `test-sine-1khz.wav` -- a 1 kHz sine, 48 kHz/16-bit/mono, 0.5 s.
- `test-perc-click.wav` -- a short exponentially-decaying click (a low thump
  plus a little noise), same format, 50 ms -- stands in for a percussive
  drum sample without needing a real recording.

Both are fully synthetic and safe to regenerate at any time; add your own
`.wav` files alongside them for a real test.

`tools/plusdrive.py` stores each input file's bytes **verbatim** (the whole
file, including its RIFF header) -- the firmware's own write path does no
format validation or conversion (see the finding), so this tool doesn't
either. What sample rate / bit depth / channel count the firmware's loader
actually expects at playback time is one of that finding's open questions;
until that's confirmed by an emulator run, stick to 48 kHz/16-bit/mono (the
device's apparent native format) for anything you want to actually hear play
back correctly.
