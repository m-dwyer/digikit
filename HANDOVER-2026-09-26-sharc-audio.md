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

## Update (J1, J2, K1 and static reading, @ K1 merge)

- `running.snap` (RTOS running, kit loaded) and `running-audio.snap` (SSI0
  clocked at the audio rate, vector 191 fires by itself) exist. A real TRIG
  on track 2 arms voice 4 through FUN_1c642a -> FUN_1c4eaf (no pokes).
- The note path is mapped: FUN_1c2b24's tail loop (FUN_1c3289,
  0x1c3289-0x1c33fe) handles new notes. Per voice it calls FUN_1c7442
  (0x1c3371) -> FUN_1c4e70, which loads a sample into the voice (word +0
  pointer, +0x188 length as Q31, +0x194 loop start; zero arguments clear
  the voice, which is what we see every frame), and it writes the guard
  bytes that make FUN_1c642a arm the voice. Its inputs come through the
  pointers 0x254d78..0x254d90, copied each frame from the record at
  0x266220 + (page<<12).
- FUN_1c7bd4 (called from 0x1c80a2) builds two DMA descriptor rings with
  the same length: ring 1 at 0x2641b0/0x2641cc -> 0x265220/0x264220 (the
  command pages replay fills) and ring 2 at 0x2641e8/0x264204 ->
  0x267220/0x266220 (the record pages). So the record is filled by DMA,
  not by code: no static writer exists. Replay only fills ring 1.
- Next: find which DMA channel and peripheral use ring 2 and its direction
  (the second half of a full-duplex SPI? DSPI1?), capture that traffic from
  running.snap, and deliver it in replay. Then the tail loop gets real
  sample pointers and lengths.

## Update (lanes H1-H3, @ e8056cf)

- Root cause of "no notes": no capture ever came from a ColdFire with its
  RTOS running. `boot400M.snap` is mid-intro; `tools/sharc_capture_run.py`
  skipped the intro hold/release other tools do, and its forced vector 191
  re-entered the handler (fixed). The sequencer never ran.
- Arm path confirmed: FUN_1c642a arms a voice (0x1c6549 -> 0x1c6582 ->
  FUN_1c4eaf) when a per-voice guard byte (DM(I1-0x54), array
  0x24f0d8..0x24f0f7) is nonzero. FUN_1c2b24's tail loop (0x1c3289-0x1c33fe,
  gated by the null pointer 0x254d78) writes those bytes. Poking one guard
  byte arms a voice through firmware code and it renders.
- BITEXT BITLEN12 > 32 is moot (the bit FIFO is never filled in dt2-1.16).
  Frame-path pitch suspect: voice-record words 0x62/0x63 as read by
  FUN_1c4afe.

Next: get a 1.16 ColdFire to a running RTOS with kit and pattern loaded
(emu/longrun.py handles the intro and couples to the SHARC process), save a
snapshot there, and capture PLAY from it. Everything SHARC-side waits on it.

## Next steps (before H1-H3; kept for reference)

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
