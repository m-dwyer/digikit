# fmt: off
"""Free-running DMA-timer counters (guirun --free-dtcn CH).

emu/dtim.py models the DMA timers as interrupt sources only: it never writes
DTCNn, the read-only counter register. A firmware that uses a timer as a time
base -- reading DTCN to busy-wait or to timestamp -- then sees a counter that
never moves. Syntakt OS 1.41 leaves DTIM2 running from the bootloader (MAIN
OS never writes DTMR2) and busy-waits on DTCN2 at 0x40000f78 for +0x10000
ticks, which hung every run at that loop.

This serves DTCNn from guest time: `clock()` is the emulator's instruction
count (guirun's `clock`, block-stepper aware), scaled by `ticks_per_instr`.
The real rate is unknown (set by the bootloader); the default assumes bus
clock / 16 at the model's nominal instruction rate. Reads inside one stepping
chunk see the same clock() value, so the counter is also nudged forward by
`min_step` per read to keep a busy-wait terminating.
"""
import struct

from unicorn import UC_HOOK_MEM_READ

from emu.dtim import BASES, DTCN
from emu.pit import F_BUS, INSTR_PER_SEC


def install_free_counters(m, clock, channels=(2,), ticks_per_instr=None, min_step=64):
    tpi = ticks_per_instr if ticks_per_instr is not None else (F_BUS / 16.0) / INSTR_PER_SEC
    state = {}
    for ch in channels:
        addr = BASES[ch] + DTCN
        m.ensure(addr)
        st = state[ch] = {'last': 0, 'reads': 0}

        def hook(uc, typ, a, size, val, data, addr=addr, st=st):
            est = int(clock() * tpi) & 0xFFFFFFFF
            v = est if est > st['last'] else (st['last'] + min_step) & 0xFFFFFFFF
            st['last'] = v
            st['reads'] += 1
            uc.mem_write(addr, struct.pack('>I', v))

        m.uc.hook_add(UC_HOOK_MEM_READ, hook, begin=addr, end=addr + 3)
    return state
