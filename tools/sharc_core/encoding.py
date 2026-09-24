"""Register codes, reset values, flag bits, access widths and instruction-field access.

Moved verbatim from tools/sharc_trace.py.
"""

from __future__ import annotations

from collections.abc import Mapping

UREG_NAMES = tuple(
    [f"R{i}" for i in range(16)]
    + [f"I{i}" for i in range(16)]
    + [f"M{i}" for i in range(16)]
    + [f"L{i}" for i in range(16)]
    + [f"B{i}" for i in range(16)]
    + [f"S{i}" for i in range(16)]
    + [
        "FADDR",
        "DADDR",
        "UREG_RESERVED_62",
        "PC",
        "PCSTK",
        "PCSTKP",
        "LADDR",
        "CURLCNTR",
        "LCNTR",
        "EMUCLK",
        "EMUCLK2",
        "PX",
        "PX1",
        "PX2",
        "TPERIOD",
        "TCOUNT",
        "USTAT1",
        "USTAT2",
        "MODE1",
        "MMASK",
        "MODE2",
        "FLAGS",
        "ASTATX",
        "ASTATY",
        "STKYX",
        "STKYY",
        "IRPTL",
        "IMASK",
        "IMASKP",
        "MODE1STK",
        "USTAT3",
        "USTAT4",
    ]
)
UREG_CODES = {name: code for code, name in enumerate(UREG_NAMES)}

# Public SHARC+ register tables document these reset values.  Keep this list
# deliberately bounded to core state used by startup rather than treating
# every absent UREG as zero.
CORE_UREG_RESET_VALUES = {
    name: 0
    for name in (
        "MODE1",
        "MMASK",
        "MODE1STK",
        "MODE2",
        "PCSTK",
        "PCSTKP",
        "LADDR",
        "LCNTR",
        "CURLCNTR",
        "ASTATX",
        "ASTATY",
        "STKYX",
        "STKYY",
        "IRPTL",
        "IMASK",
        "IMASKP",
    )
}
CORE_MMR_RESET_VALUES = {
    0x30024: 0,  # CMMR_SYSCTL
    0x31400: 0,  # SHBTB_CFG
    0x31401: 0,  # SHBTB_LOCK_START
    0x31402: 0,  # SHBTB_LOCK_END
    0x3E000: 0,  # SHL1C_CFG
    0x3E002: 0,  # SHL1C_CFG2
}

# ADSP-2156x L1 block 3 aliases.  The normal-word window is the one used by
# the reset path's PM(...)=PX table read; the loader records the same physical
# storage through the short-word/system-byte view.
L1_BLOCK3_NW_BASE = 0x000E0000
L1_BLOCK3_NW_LIMIT = 0x000E8000
L1_BLOCK3_SW_BASE = 0x001C0000

# ASTATX/ASTATY bit positions (SHARC+ PRM ch.4 REGF_ASTATX/REGF_ASTATY).
AZ_BIT, AV_BIT, AN_BIT, AC_BIT, AS_BIT, AI_BIT = 0, 1, 2, 3, 4, 5
MN_BIT, MV_BIT, MU_BIT, MI_BIT = 6, 7, 8, 9
AF_BIT = 10
SV_BIT, SZ_BIT, SS_BIT = 11, 12, 13
SF_BIT = 14  # Shifter Bit FIFO (PRM Table 28-2, REGF_ASTATX bit 14, all.txt:29659)
BTF_BIT = 18
ALUSAT_BIT = 13  # MODE1.ALUSAT
TRUNCATE_BIT = 15  # MODE1.TRUNCATE (PRM Table 28-19, p.28-63): rounding mode
# select for FIX -- 0 rounds to nearest (ties to even), 1 truncates toward
# zero. TRUNC always truncates and does not consult this bit (PRM p.24-.. /
# PGR p.11-37: "The trunc operation always truncates toward 0. The TRUNCATE
# bit does not influence operation of the trunc instruction.").

# Bits every fixed-point ALU op (add/sub/inc/dec/pass/not/and/or/xor/compare)
# defines: AZ/AV/AN/AC/AS/AI, plus AF which ch.19's intro says every
# fixed-point ALU op clears (PRM p.439).
ALU_FLAGS_MASK = (
    (1 << AZ_BIT)
    | (1 << AV_BIT)
    | (1 << AN_BIT)
    | (1 << AC_BIT)
    | (1 << AS_BIT)
    | (1 << AI_BIT)
    | (1 << AF_BIT)
)
# Multiplier-result flags (MN/MV/MU/MI); the tracer does not model the
# multiplier result format, so these are always left unknown except for the
# MR data-move, which the PRM (p.493) documents as clearing all four.
MULT_FLAGS_MASK = (1 << MN_BIT) | (1 << MV_BIT) | (1 << MU_BIT) | (1 << MI_BIT)
# Shifter-result flags (SV/SZ/SS); forgotten (not guessed) for an
# undocumented shifter opcode or an unrecognized compute unit.
SHIFT_FLAGS_MASK = (1 << SV_BIT) | (1 << SZ_BIT) | (1 << SS_BIT)

# Type3b's (l, x, w) ACCESS/BH/BHSE encode table (SHARC+ Core Programming
# Reference rev. 1.4, pp. 13-16--13-19), used by this module's own "3b"
# _execute branch below; Type4b/4d share the identical 3-bit l/x/w table
# (PRM pp.13-31/13-32/13-34, no "ex" bit -- unlike Type3d/14d, which add
# one), so tools/sharcdb.py's extract_mem_access() imports this rather than
# re-deriving it.
ACCESS_WIDTHS = {
    (0, 1, 1): "normal-word",
    (0, 0, 0): "byte",
    (0, 1, 0): "byte-sign-extended",
    (1, 0, 0): "short-word",
    (1, 1, 0): "short-word-sign-extended",
    (1, 1, 1): "long-word",
}

# IF-condition codes (PGR Table 10-4) that read a single ASTATX bit,
# optionally complemented.
SIMPLE_COND_BITS = {
    0x03: (AC_BIT, False),
    0x13: (AC_BIT, True),
    0x04: (AV_BIT, False),
    0x14: (AV_BIT, True),
    0x05: (MV_BIT, False),
    0x15: (MV_BIT, True),
    0x06: (MN_BIT, False),
    0x16: (MN_BIT, True),
    0x07: (SV_BIT, False),
    0x17: (SV_BIT, True),
    0x08: (SZ_BIT, False),
    0x18: (SZ_BIT, True),
    0x0D: (BTF_BIT, False),
    0x1D: (BTF_BIT, True),
}


def _field(f: Mapping[str, int], stem: str) -> int:
    for key, value in f.items():
        if key == stem or key.startswith(stem + "["):
            return value
    raise KeyError(stem)


def _wide(f: Mapping[str, int], stem: str) -> int:
    return (_field(f, stem + "[31:16]") << 16) | _field(f, stem + "[15:0]")
