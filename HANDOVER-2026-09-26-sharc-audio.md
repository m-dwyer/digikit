# Handover 2026-09-26: SHARC audio output

State and next steps only. Results are in `docs/findings/` (06 last sections,
05 Type 7a, 07 tooling). Branch `work/sharc-emulator` @ 28aaccf, pushed.

## Goal

Hear the Digitakt II (dt2-1.16) SHARC make sound in our emulator, driven by
the firmware's own state, then patch it.

## Where we are

| Stage | State |
|---|---|
| One voice (FUN_1c4f81) | Correct: `check_correctness` max error ~3e-5 at steps 0.5-2.0 |
| Frame (FUN_1c2b24 via block handler 0x1c74cd) | Returns every frame; ~96k instr/frame; multi-frame runs keep memory flat |
| DAC | Known: ring A halves 0x261cc8/0x261dc8, 32 stereo Q31 samples/block, L,R interleaved, from planar master mix 0x25f180 (L) / 0x25f200 (R) via FUN_1c74a1; firmware negates L (0x1c758b) |
| ColdFire link | DSPI2 transfer (2,748 B) lands in the command ring 0x264220/0x265220 (descriptor ring built by FUN_1c7bd4); FUN_1c2b24 copies 2,048 B into 0x2558dc. Replay delivers real captures there |
| Sound | Clean tone in ring A only with a hand step (`inject_master_mix`); pitch in the frame path is 0.632x (caused by the remaining frame patch) |

Speed: ~90-100k instr/s CPython, ~270k PyPy (about 5 min of PyPy per audio
second).

## Hand steps still in use (remove as real state replaces them)

- `FRAME_PATCH_TABLE[0x1c4965]`: fork at 0x1c4969 from BITEXT(NU) with
  BITLEN12 = 95 at 0xb88fa4 (manual: "prohibited", no result defined).
  Causes the frame-path pitch error.
- `inject_master_mix`: the voice output is written into the master mix at
  the FUN_1c74a1 call site. The per-track accumulate in FUN_1c642a
  (0x1c6b3c-0x1c6b8a) is gated by DM(0x252d3c) (init leaves 0; only other
  writer FUN_1cdbb2 via slot-type dispatch).
- `setup_voice` pokes the voice record (no firmware path arms a voice).
- `--provisional 21p_undoc16=nop` (opt-in) for 0x1c32b0; next stop
  8p_undoc48 at 0x1c32b4. cu=3 opcodes 0xe0/0xd6 decode as reserved.

## Blocker: no firmware-armed voice

- The trigger path FUN_1c60a2 never runs: the companding loop in FUN_1c2b24
  reads through DM(0x254d78), which is null. FUN_1c2b24 stores it from the
  record at 0x266220 + (page<<12) (dispatcher 0x1c778a's second output);
  nothing reachable writes that record, and the DSPI2 transfer does not
  land there.
- The arm call 0x1c6582 (FUN_1c642a -> FUN_1c4eaf) is guarded at
  0x1c6540/0x1c6549; all 32 voice iterations take 0x1c6553 instead.
- ColdFire captures: panel keys reach the firmware, the kit loads with
  `--pre-instrs`, but PLAY and TRIG change no edge-triggered byte in the
  transfer. The ColdFire emulator's PIT/DTIM timer service delays the kit
  load (lane A3), so the sequencer may not be running in real time.

## Next steps (recommended order)

1. Decide whether the ColdFire emits notes at all: during PLAY, check that
   its sequencer step counter advances and that note events are created on
   the ColdFire side (observe records, not the UI). If not, fix the timer
   model first: no SHARC-side search can find a trigger that is never sent.
2. BITEXT with BITLEN12 > 32: use the frame-path pitch as the oracle.
   Candidate semantics are right only if a 1 kHz request gives 1 kHz.
3. Arm path, top-down: what state makes FUN_1c642a take 0x1c6540/0x1c6549
   and call FUN_1c4eaf; which memory holds it; who writes it
   (`tools/sharc_inputs.py`, `tools/sharc_survey.py --collect-all`).

## Tools added this session

`tools/sharc_widthaudit.py` (runtime widths vs database), `sharc_survey.py`
(`--collect-all`), `sharc_inputs.py` (input sorter), `sharc_framemap.py`,
`sharc_dac.py`, `sharc_replay.py`, `sharc_capture_run.py` +
`emu/sharc_capture.py` (captures under `out/captures/`, never commit),
`sharc_run.py` (snapshots, watchpoints, `fresh_call`, `--provisional`),
`sharc.py` (`cfg`/`callgraph`/`defuse`/`slice` with per-group ASTAT).
`sharc_harness.FRAME_MILESTONE` pins the frame stop once.

## How we worked

Sonnet lanes in worktrees, each owning files, reading a one-page state
file, committing in its worktree and returning a short JSON verdict; merged
by cherry-pick. `tests/test_sharc_pcode.py` GeneratedLanguage fails until
`tools/ghidra/install-sharc.sh` is run (decode_table gained Type7a w/l).
