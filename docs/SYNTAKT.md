# Syntakt

Elektron Syntakt, OS 1.41 (`Syntakt_OS1.41.syx`, sha256 `8e2488f4…d19e`, listed in `devices/syntakt.toml`).
Same MCF5441x ColdFire and the same RTOS block as the Digitakt II, linked at a different base; the resolver's
relocated-`Fixed` rule and the Syntakt-only symbols in `emu/symbols.py` cover the difference.

## Status

- Cold boot to the main screen (project "A01", the calibration dialog) in about 331 M instructions, no
  `FancyErrorView`. Panel input works through `--feed` (buttons and encoders).
- Not modelled: the DSP (its loader gives up after about 690 M instructions and the UI shows "OOPS"), and the
  audio-frame interrupt, so engine code that runs from it (the LFOs, for example) has to be called directly.

## Running it

```
python -m emu.extract Syntakt_OS1.41.syx -o sections/
python -m emu.checkpoint make 120000000 snapshots/boot Syntakt_OS1.41.syx
python tools/guirun.py snapshots/boot120M.snap --syx Syntakt_OS1.41.syx --intro-timers pit3 --free-dtcn 2 \
    --limit 331000000 --save-at 330M:snapshots/post330M.snap --png-at 330M:main.png
```

`--free-dtcn 2` is needed: the OS busy-waits on DMA timer 2's counter, which the bootloader leaves running and
the model otherwise never advances.

## Panel (measured by use, not with tools/panelsweep.py)

Button message `2c mask` (the whole state of channel c), encoder message `3c dd` (signed delta). Channel 0 bits
0-7 = tracks 1-8, channel 1 bits 0-3 = tracks 9-12, channel 2 bit 7 = LFO page, channel 3 bit 3 = NO. Encoders
A-H = channels 0-7, LEVEL = 8. The first encoder message after a page or parameter change only shows the value;
each later one moves by delta/6 positions. The `[panel]` groups in `devices/syntakt.toml` are the Digitakt II
layout as a placeholder.
