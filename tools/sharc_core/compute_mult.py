"""Multiplier compute (cu=1) and MR data moves: handler bodies dispatched
by tools/sharc_core/compute.py's MULT_OPS table (plus MR_DATAMOVE_REGISTERS/
_mr_data_move for the separate MRDATAMOVE encoding, and the two
multiply-accumulate helpers compute.py calls directly ahead of its
mf/cu dispatch -- see compute.py's docstring for why).

Handler bodies moved verbatim from the cu==1 branches of
tools/sharc_trace.py's old _compute if-chain; each still carries that
branch's own PRM/PGR citation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from .flags import (
    _astatx_mult_clear,
    _astatx_mult_fixed,
    _astatx_mult_forget,
    _astatx_mult_sat,
)
from .floats import _float32, _float32_bits
from .state import _ureg
from .values import (
    ComputeResult,
    Const,
    Operand,
    Unknown,
    Value,
    _add,
    _multiply,
    _multiply_fractional,
)

Handler = Callable[
    [
        int,
        int,
        int,
        Operand,
        Operand,
        Mapping[int, Value],
        Mapping[str, Operand] | None,
        bool,
    ],
    ComputeResult,
]


# PRM Table 18-29 MRDATAMOVE (p.438), cross-checked against
# tools/sharcspec/compute_table.json's mrdatamove table (PGR Table 12-10,
# pgr.txt:22827-22841): opcode[15:12] selects which of the six banked
# multiplier-result registers a data move addresses.
MR_DATAMOVE_REGISTERS = {
    0x0: "MR0F",
    0x1: "MR1F",
    0x2: "MR2F",
    0x4: "MR0B",
    0x5: "MR1B",
    0x6: "MR2B",
}


def _mr_data_move(
    mr_name: str,
    rn: int,
    direction: int,
    values: Mapping[int, Value],
    special: Mapping[str, Operand] | None,
) -> ComputeResult:
    """PRM Table 18-29 MRDATAMOVE (p.438): moves a 32-bit value between the
    register file and one of the six banked multiplier-result registers.
    MR0F is the low 32 bits of the same 80-bit accumulator the
    multiply-accumulate rows below call "MRF" (PRM p.3-10: "The REGF_MRF
    register ... is comprised of the REGF_MR2F, REGF_MR1F, and REGF_MR0F
    registers"), so it reuses that special-dict key; the other five (guard
    bits and the alternate/background bank, PRM p.3-10/4-77, "Each
    multiplier has a primary or foreground register ... and alternate or
    background") get their own key since this tracer does not otherwise
    model their contents. Flags: PRM p.493, MU/MN/MI/MV all cleared for
    every direction and register.

    The returned destination is MR_NAME itself (not the aliased key) so the
    trace event names the register the instruction actually addresses;
    ``_apply_compute`` applies the MR0F->"MRF" alias when it commits the
    write to state.special."""
    key = "MRF" if mr_name == "MR0F" else mr_name
    if direction:  # register file -> MR register
        return mr_name, _ureg(values, rn), "mr-data-move", _astatx_mult_clear
    value = (special or {}).get(key, Unknown("uninitialized %s" % mr_name))
    return rn, value, "mr-data-move", _astatx_mult_clear


def multiply_accumulate_mrf(
    rx: int, ry: int, values: Mapping[int, Value], special: Mapping[str, Operand] | None
) -> tuple:
    """PRM multiplier compute table: MRF = MRF + RX * RY (MOD1). Preserve
    the accumulator separately from the UREG file so later MR transfers do
    not masquerade as architectural UREGs."""
    accumulator = (special or {}).get("MRF", Unknown("uninitialized MRF"))
    product = _multiply(_ureg(values, rx), _ureg(values, ry), "R%d * R%d" % (rx, ry))
    return (
        "MRF",
        _add(accumulator, product, "MRF + R%d * R%d" % (rx, ry)),
        "multiply-accumulate",
        _astatx_mult_forget,
    )


def multiply_add_mrf(
    rn: int,
    rx: int,
    ry: int,
    values: Mapping[int, Value],
    special: Mapping[str, Operand] | None,
) -> tuple:
    accumulator = (special or {}).get("MRF", Unknown("uninitialized MRF"))
    product = _multiply(_ureg(values, rx), _ureg(values, ry), "R%d * R%d" % (rx, ry))
    return (
        rn,
        _add(accumulator, product, "MRF + R%d * R%d" % (rx, ry)),
        "multiply-add-mrf",
        _astatx_mult_forget,
    )


# PRM Table 17-7: MULOP 0000 F00x writes a saturated MRF value to RN.
# The tracer does not model the full-width multiplier accumulator or MOD2
# format bits, so preserve the documented data dependency conservatively.
def mult_saturate_mrf_ui(
    rn, rx, ry, left, right, values, special, approx_recips
) -> tuple:
    return (
        rn,
        Unknown("saturated MRF (unmodeled MOD2)"),
        "saturate-mrf",
        _astatx_mult_forget,
    )


def mult_raw_ssi(rn, rx, ry, left, right, values, special, approx_recips) -> tuple:
    value = _multiply(left, right, "R%d * R%d" % (rx, ry))
    return rn, value, "multiply", _astatx_mult_forget


# PRM Table 18-7 (p.428-429): MULOP 00110000 is Fn = Fx * Fy. Flags are
# the multiplier's MN/MV/MU/MI (PGR p.11-57), the same unmodeled-result
# group the fixed-point multiply above forgets via
# ``_astatx_mult_forget``; unlike the ALU's NaN-input quirk, the PGR
# text for this op does not document an all-1s override, so ordinary
# IEEE NaN propagation applies.
def mult_float(rn, rx, ry, left, right, values, special, approx_recips) -> tuple:
    a, b = _float32(left), _float32(right)
    value: Operand
    if a is None or b is None:
        value = Unknown("F%d * F%d" % (rx, ry))
    else:
        bits, _overflowed = _float32_bits(a * b)
        value = Const(bits)
    return rn, value, "float-multiply", _astatx_mult_forget


# PRM Table 17-7 (p.17-7), "(RN|mrf|mrb) = RX*RY MOD1" row, MOD1 UUI
# sub-option (PRM p.17-9): same row as opcode 0x70 (SSI) above with
# RX/RY unsigned instead of signed. The low 32 bits of a 32x32 product
# do not depend on operand signedness (two's-complement wraparound is
# identical either way), so this shares 0x70's raw-multiply semantics.
def mult_raw_uui(rn, rx, ry, left, right, values, special, approx_recips) -> tuple:
    value = _multiply(left, right, "R%d * R%d" % (rx, ry))
    return rn, value, "multiply", _astatx_mult_fixed


# Same row, MOD1 UUF sub-option (fractional, unsigned*unsigned, no
# round): PRM p.3-9/27-3 -- the register result is the top 32 bits of
# the 64-bit unsigned product (no redundant-sign shift; that only
# applies when both inputs are signed).
def mult_fractional_uuf(
    rn, rx, ry, left, right, values, special, approx_recips
) -> tuple:
    value = _multiply_fractional(left, right, False, False, "R%d * R%d" % (rx, ry))
    return rn, value, "multiply", _astatx_mult_fixed


# PRM Table 17-7, "mrf = RX*RY MOD1" row (no accumulate), MOD1 SSI
# sub-option: the plain-load twin of opcode 0xB4's accumulate above,
# same raw-multiply-into-MRF data dependency.
def mult_mrf_ssi(rn, rx, ry, left, right, values, special, approx_recips) -> tuple:
    value = _multiply(left, right, "R%d * R%d" % (rx, ry))
    return "MRF", value, "multiply-mrf", _astatx_mult_fixed


# Same row, MOD1 SSF sub-option (fractional, signed*signed, no round):
# PRM p.3-9 -- both inputs signed, so the redundant-sign left shift
# applies (folded into _multiply_fractional's >>31).
def mult_mrf_ssf(rn, rx, ry, left, right, values, special, approx_recips) -> tuple:
    value = _multiply_fractional(left, right, True, True, "R%d * R%d" % (rx, ry))
    return "MRF", value, "multiply-mrf", _astatx_mult_fixed


# PRM Table 17-7, "mrf = mrf + RX*RY MOD1" row, MOD1 SSF sub-option:
# the fractional twin of opcode 0xB4 (SSI, integer) above.
def mult_accumulate_ssf(
    rn, rx, ry, left, right, values, special, approx_recips
) -> tuple:
    accumulator = (special or {}).get("MRF", Unknown("uninitialized MRF"))
    product = _multiply_fractional(left, right, True, True, "R%d * R%d" % (rx, ry))
    value = _add(accumulator, product, "MRF + R%d * R%d (SSF)" % (rx, ry))
    return "MRF", value, "multiply-accumulate", _astatx_mult_fixed


# PRM Table 17-7, "RN = sat mrf MOD2" row, MOD2 SF sub-option: same row
# opcode 0x00 above handles as UI. PRM p.3-11/Table 3-5 defines
# saturation against the fractional maximum, which needs the unmodeled
# 80-bit MRF value, so this stays Unknown for the same reason 0x00
# does; only the ASTATX rule differs (MU is fixed 0 on this row, not
# merely unknown -- PRM Table 3-7, p.3-12).
def mult_saturate_mrf_sf(
    rn, rx, ry, left, right, values, special, approx_recips
) -> tuple:
    return (
        rn,
        Unknown("saturated MRF (unmodeled MOD2, SF)"),
        "saturate-mrf",
        _astatx_mult_sat,
    )


# Undocumented in both public sources: PRM Table 17-7 and PGR Table
# 12-5 both list only mrf/mrb=0 (0001 0100/0110) and rnd MOD3
# (0001 100x-111x) under the "0001 xxxx" opcode prefix -- neither has a
# 0001 0000 row. Decode it (so the walk does not desync) and leave the
# result and flags Unknown, same as the shifter's undocumented 0xB0 gap
# in compute_shift.py.
def mult_undocumented_10(
    rn, rx, ry, left, right, values, special, approx_recips
) -> tuple:
    label = "multiply opcode 0x10 R%d, R%d (undocumented; no public source)" % (rx, ry)
    return rn, Unknown(label), "multiply-undocumented-10", _astatx_mult_forget


MULT_OPS: dict[int, Handler] = {
    0x00: mult_saturate_mrf_ui,
    0x70: mult_raw_ssi,
    0x30: mult_float,
    0x40: mult_raw_uui,
    0x48: mult_fractional_uuf,
    0x74: mult_mrf_ssi,
    0x7C: mult_mrf_ssf,
    0xBC: mult_accumulate_ssf,
    0x09: mult_saturate_mrf_sf,
    # Undocumented; kept here (not a separate table) so cu=1 dispatch stays
    # a single lookup -- see the docstring above mult_undocumented_10.
    0x10: mult_undocumented_10,
}
