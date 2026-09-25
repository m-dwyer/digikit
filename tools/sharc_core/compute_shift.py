"""Shifter compute (cu=2) and the ShiftImm form: handler bodies dispatched
by tools/sharc_core/compute.py's SHIFT_OPS table, plus _shift_immediate
(the ShiftImm 6a_mem/6b_shiftimm forms call this directly, not through
_compute -- see its own docstring).

Handler bodies moved verbatim from the cu==2 branches of
tools/sharc_trace.py's old _compute if-chain and from _shift_immediate;
each still carries that branch's own PRM/PGR citation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from .encoding import (
    ALU_FLAGS_MASK,
    SF_BIT,
    SHIFT_FLAGS_MASK,
    SS_BIT,
    SV_BIT,
    SZ_BIT,
    _field,
)
from .flags import (
    _astatx_bit_field,
    _astatx_btst,
    _astatx_fext,
    _astatx_forget,
    _astatx_from_updates,
    _astatx_lefto,
    _astatx_leftz,
    _astatx_shift,
)
from .state import MR, _ureg
from .values import (
    ComputeResult,
    Const,
    Operand,
    Unknown,
    Value,
    _bitwise,
    _signed,
    _signed32,
)

Handler = Callable[
    [
        int,
        int,
        int,
        Operand,
        Operand,
        Mapping[int, Value],
        Mapping[str, Operand | MR] | None,
        bool,
    ],
    ComputeResult,
]


def _field_deposit_or(
    dest: Operand,
    source: Operand,
    position: int,
    length: int,
    sign_extend: bool,
    label: str,
) -> Operand:
    """RN = RN or fdep RX by BIT6:LEN6[(SE)] (PGR p.11-70/11-74, cross-
    checked against compute_table.json's shiftop_shiftimm rows 011011/
    011101): deposits the low LENGTH bits of SOURCE at bit POSITION of DEST
    -- sign-extending the deposited field's own top bit upward through bit
    31 first when SIGN_EXTEND, mirroring the sign-extend convention the
    already-implemented FEXT-SE (opcode 0x12) uses -- then ORs the result
    into DEST (bits outside [POSITION, min(POSITION+LENGTH,32)) are left as
    DEST already had them, unlike the plain, non-OR FDEP forms this tracer
    does not implement)."""
    if length == 0:
        return dest
    if not isinstance(source, Const) or not isinstance(dest, Const):
        return Unknown(label)
    field = source.value & ((1 << length) - 1)
    if sign_extend and (field >> (length - 1)) & 1:
        field |= (0xFFFFFFFF << length) & 0xFFFFFFFF
    deposited = (field << position) & 0xFFFFFFFF
    return Const((dest.value & 0xFFFFFFFF) | deposited)


def _shift_immediate(
    f: Mapping[str, int],
    values: Mapping[int, Value],
    special: Mapping[str, Operand | MR] | None = None,
) -> ComputeResult:
    """Execute the documented ShiftImm subset seen on qualifying paths.

    SPECIAL is the same special-register mapping ``_compute`` reads MRF
    from (currently just "BFFWRP", the bit-FIFO write pointer opcodes 0x14/
    0x19/0x1f below read and update); it defaults to None for every caller
    that does not need those opcodes."""
    field = (_field(f, "shiftimm[22:16]") << 16) | _field(f, "shiftimm[15:0]")
    opcode = (field >> 16) & 0x3F
    data8 = (field >> 8) & 0xFF
    rn, rx = (field >> 4) & 0xF, field & 0xF
    source = _ureg(values, rx)
    if opcode in (0x00, 0x01, 0x08, 0x09):
        amount = _signed(data8, 8)
        base = opcode & 0x01
        name = "lshift" if base == 0x00 else "ashift"
        if amount == 0:
            shifted = source
        elif not isinstance(source, Const):
            shifted = Unknown("%s R%d by %d" % (name, rx, amount))
        elif amount >= 32:
            shifted = Const(0)
        elif amount <= -32:
            shifted = (
                Const(0xFFFFFFFF)
                if base == 0x01 and source.value & 0x80000000
                else Const(0)
            )
        elif amount > 0:
            shifted = Const(source.value << amount)
        elif base == 0x01:
            shifted = Const(_signed32(source.value) >> -amount)
        else:
            shifted = Const(source.value >> -amount)
        if opcode in (0x08, 0x09):
            # PRM Table 17-9, shiftimm 001000/001001 (p. 17-10): RN = RN or
            # (l/a)shift RX by DATA8.
            value = _bitwise(
                _ureg(values, rn),
                shifted,
                "R%d or %s R%d by %d" % (rn, name, rx, amount),
                lambda a, b: a | b,
            )
            operation = (
                "logical-shift-or-immediate"
                if base == 0x00
                else "arithmetic-shift-or-immediate"
            )
        else:
            value = shifted
            operation = (
                "logical-shift-immediate"
                if base == 0x00
                else "arithmetic-shift-immediate"
            )
        # SZ is defined from the shifted value before any OR (PRM pp.509-510:
        # "Set if the shifted result is zero"); SS is cleared for every one
        # of these forms except OR-ashift (opcode 0x09), whose entry omits
        # the SS line.
        ss_mode = "forget" if opcode == 0x09 else "clear"
        return rn, value, operation, _astatx_shift(amount, shifted, ss_mode)
    if opcode == 0x10:
        position = data8 & 0x3F
        length = (_field(f, "dataex[3:0]") << 2) | (data8 >> 6)
        if length == 0:
            value = Const(0)
        elif not isinstance(source, Const):
            value = Unknown("fext R%d by %d:%d" % (rx, position, length))
        else:
            value = Const((source.value >> position) & ((1 << min(length, 32)) - 1))
        return (
            rn,
            value,
            "field-extract-immediate",
            _astatx_fext(position + length, value),
        )
    if opcode == 0x12:
        # PRM Table 17-9 p.17-10/17-11 (out/refs/sharc-plus-prm/all.txt
        # lines 22781-22785): shiftimm 010010 is
        # "RREG = fext RREG by BIT6:LEN6 (se)" -- the sign-extending twin
        # of the already-implemented opcode 0x10. Field layout (position,
        # length) is identical to 0x10; NOTE at PRM p.3-17 (all.txt
        # line 3486) says the (SE) option "sign extends the left bits" of
        # the extracted field, i.e. bits above the extracted field take the
        # value of the field's own sign (MSB) instead of being cleared.
        # CALIBRATION NOTE: added in a scratch copy of sharc_trace.py for
        # this task only; not part of the tracked tool.
        position = data8 & 0x3F
        length = (_field(f, "dataex[3:0]") << 2) | (data8 >> 6)
        if length == 0:
            value = Const(0)
        elif not isinstance(source, Const):
            value = Unknown("fext R%d by %d:%d (se)" % (rx, position, length))
        else:
            value = Const(
                _signed(source.value >> position, min(length, 32)) & 0xFFFFFFFF
            )
        return (
            rn,
            value,
            "field-extract-immediate-se",
            _astatx_fext(position + length, value),
        )
    if opcode in (0x30, 0x31):
        position = data8
        if position > 31:
            value = source
        else:
            calculate = (
                (lambda a, b: a | b) if opcode == 0x30 else (lambda a, b: a & ~b)
            )
            name = "bset" if opcode == 0x30 else "bclr"
            value = _bitwise(
                source,
                Const(1 << position),
                "%s R%d by %d" % (name, rx, position),
                calculate,
            )
        operation = "bit-set-immediate" if opcode == 0x30 else "bit-clear-immediate"
        return rn, value, operation, _astatx_bit_field(position, value)
    if opcode == 0x32:
        position = data8
        if position > 31:
            value = source
        else:
            value = _bitwise(
                source,
                Const(1 << position),
                "btgl R%d by %d" % (rx, position),
                lambda a, b: a ^ b,
            )
        return rn, value, "bit-toggle-immediate", _astatx_bit_field(position, value)
    if opcode == 0x33:
        # PRM Table 17-9: ShiftImm 110011 is btst RX by DATA8, the immediate
        # form of the 11001100 register operation. It updates status only, so
        # RN keeps its value.
        return rn, source, "bit-test", _astatx_btst(source, Const(data8))
    if opcode == 0x1D:
        # compute_table.json shiftop_shiftimm row 011101 / PGR p.11-73/11-74
        # (pgr.txt:22883): RN = RN or fdep RX by BIT6:LEN6 (SE). Same
        # bit6/len6 packing as the FEXT-SE opcode (0x12) above.
        position = data8 & 0x3F
        length = (_field(f, "dataex[3:0]") << 2) | (data8 >> 6)
        dest = _ureg(values, rn)
        label = "R%d or fdep R%d by %d:%d (se)" % (rn, rx, position, length)
        value = _field_deposit_or(dest, source, position, length, True, label)
        return rn, value, "field-deposit-or-se", _astatx_fext(position + length, value)
    if opcode == 0x1F:
        # compute_table.json shiftop_shiftimm row 011111 / PGR p.11-89
        # (pgr.txt:23379): BFFWRP = DATA7 -- the immediate form of cu=2
        # opcode 0x7c above (writes the bit-FIFO write-pointer special
        # register, not an RN).
        new_wrp = Const(data8 & 0x7F)
        updates: dict[int, bool | None] = {
            SS_BIT: False,
            SZ_BIT: False,
            SV_BIT: new_wrp.value > 64,
            SF_BIT: new_wrp.value >= 32,
        }
        return "BFFWRP", new_wrp, "bffwrp-write", _astatx_from_updates(updates)
    if opcode in (0x14, 0x19):
        # compute_table.json shiftop_shiftimm rows 010100/011001 / PGR
        # p.11-86/11-90 (pgr.txt:23271-23336): RN = BITEXT RX|BITLEN12(,NU).
        # Extracts the top BITLEN12 bits of the internal 64-bit bit FIFO
        # into RN, left-shifts the FIFO by that amount, and decrements
        # BFFWRP by the same amount; 0x19's NU modifier skips the FIFO/
        # pointer update (and leaves SF untouched -- PGR: "the SF flag is
        # not updated"). This tracer does not model the FIFO's 64-bit
        # content (no instruction in this image's coverage scan writes it --
        # BITDEP never appears), so RN is always Unknown; BFFWRP itself is
        # still tracked from BFFWRP= writes (opcode 0x1f above / 0x7c in
        # _compute) where known, so SF/SV stay meaningful even though RN
        # does not.
        no_update = opcode == 0x19
        bitlen12 = (_field(f, "dataex[3:0]") << 8) | data8
        value = Unknown("bitext by %d%s" % (bitlen12, " (nu)" if no_update else ""))
        updates = {SS_BIT: False, SV_BIT: bitlen12 > 32, SZ_BIT: None}
        if no_update:
            return rn, value, "bit-extract-nu", _astatx_from_updates(updates)
        old_wrp = (special or {}).get("BFFWRP")
        wrp_after: Operand = (
            Const(old_wrp.value - bitlen12)
            if isinstance(old_wrp, Const)
            else Unknown("uninitialized BFFWRP")
        )
        updates[SF_BIT] = (
            wrp_after.value >= 32 if isinstance(wrp_after, Const) else None
        )
        return (
            (rn, "BFFWRP"),
            (value, wrp_after),
            "bit-extract",
            _astatx_from_updates(updates),
        )
    raise ValueError("unsupported ShiftImm opcode %#x" % opcode)


# PRM Table 17-9: SHIFTOP 00000000 is RN = LSHIFT RX by RY. The signed
# low byte of RY selects a left (positive) or logical right (negative)
# shift; magnitudes of 32 or more produce zero.
def shift_logical(rn, rx, ry, left, right, values, special, approx_recips) -> tuple:
    amount: int | None = None
    value: Operand
    if not isinstance(right, Const):
        value = Unknown("lshift R%d by R%d" % (rx, ry))
    else:
        amount = _signed(right.value & 0xFF, 8)
        if amount == 0:
            value = left
        elif not isinstance(left, Const):
            value = Unknown("lshift R%d by %d" % (rx, amount))
        elif amount >= 32 or amount <= -32:
            value = Const(0)
        elif amount > 0:
            value = Const(left.value << amount)
        else:
            value = Const(left.value >> -amount)
    return rn, value, "logical-shift", _astatx_shift(amount, value, "clear")


# PRM Table 17-9 p.17-10 (out/refs/sharc-plus-prm/all.txt line 22765):
# SHIFTOP 00000100 is RN = ASHIFT RX by RY -- the register-operand twin
# of the already-implemented ShiftImm opcode 0x01 (arithmetic shift by
# an 8-bit immediate). Same signed-low-byte amount, same 32-magnitude
# saturation, differing only in that a right shift (negative amount) is
# sign-extending, mirroring _shift_immediate's opcode==0x01 branch
# exactly.
# CALIBRATION NOTE: added in a scratch copy of sharc_trace.py for this
# task only; not part of the tracked tool.
def shift_arithmetic(rn, rx, ry, left, right, values, special, approx_recips) -> tuple:
    amount = None
    value: Operand
    if not isinstance(right, Const):
        value = Unknown("ashift R%d by R%d" % (rx, ry))
    else:
        amount = _signed(right.value & 0xFF, 8)
        if amount == 0:
            value = left
        elif not isinstance(left, Const):
            value = Unknown("ashift R%d by %d" % (rx, amount))
        elif amount >= 32:
            value = Const(0)
        elif amount <= -32:
            value = Const(0xFFFFFFFF) if left.value & 0x80000000 else Const(0)
        elif amount > 0:
            value = Const(left.value << amount)
        else:
            value = Const(_signed32(left.value) >> -amount)
    return rn, value, "arithmetic-shift", _astatx_shift(amount, value, "clear")


# PRM Table 17-9 (compute_table.json shiftop_shiftimm row 00100000):
# SHIFTOP 00100000 is RN = RN OR LSHIFT RX by RY -- the register-operand
# twin of the already-implemented ShiftImm opcode 0x08 (OR-lshift by an
# 8-bit immediate). Same signed-low-byte amount/32-magnitude rule as
# plain lshift (0x00 above), computed independently here rather than
# shared, then ORed into RN instead of replacing it.
def shift_logical_or(rn, rx, ry, left, right, values, special, approx_recips) -> tuple:
    or_amount: int | None = None
    shifted: Operand
    if not isinstance(right, Const):
        shifted = Unknown("lshift R%d by R%d" % (rx, ry))
    else:
        or_amount = _signed(right.value & 0xFF, 8)
        if or_amount == 0:
            shifted = left
        elif not isinstance(left, Const):
            shifted = Unknown("lshift R%d by %d" % (rx, or_amount))
        elif or_amount >= 32 or or_amount <= -32:
            shifted = Const(0)
        elif or_amount > 0:
            shifted = Const(left.value << or_amount)
        else:
            shifted = Const(left.value >> -or_amount)
    value = _bitwise(
        _ureg(values, rn),
        shifted,
        "R%d or lshift R%d by R%d" % (rn, rx, ry),
        lambda a, b: a | b,
    )
    return rn, value, "logical-shift-or", _astatx_shift(or_amount, shifted, "clear")


# PRM Table 17-9: SHIFTOP 10001000 is RN = leftz RX.
def shift_leftz(rn, rx, ry, left, right, values, special, approx_recips) -> tuple:
    value = (
        Const(32 if left.value == 0 else 32 - left.value.bit_length())
        if isinstance(left, Const)
        else Unknown("leftz R%d" % rx)
    )
    return rn, value, "leftz", _astatx_leftz(left, value)


# PGR p.11-83 (pgr.txt:23171): SHIFTOP 10001100 is RN = LEFTO RX --
# leading 1s, the complement of leftz above (leading 0s of ~RX).
def shift_lefto(rn, rx, ry, left, right, values, special, approx_recips) -> tuple:
    value: Operand
    if isinstance(left, Const):
        inverted = (~left.value) & 0xFFFFFFFF
        value = Const(32 if inverted == 0 else 32 - inverted.bit_length())
    else:
        value = Unknown("lefto R%d" % rx)
    return rn, value, "lefto", _astatx_lefto(left, value)


# PRM Table 18-9: SHIFTOP 11000000/11000100 are variable bit set/clear.
def _shift_bitset_impl(rn, rx, ry, left, right, opcode: int) -> tuple:
    name = "bset" if opcode == 0xC0 else "bclr"
    value: Operand
    if not isinstance(right, Const):
        value = Unknown("%s R%d by R%d" % (name, rx, ry))
    elif right.value > 31:
        value = left
    else:
        calculate = (lambda a, b: a | b) if opcode == 0xC0 else (lambda a, b: a & ~b)
        value = _bitwise(
            left, Const(1 << right.value), "%s R%d by R%d" % (name, rx, ry), calculate
        )
    return (
        rn,
        value,
        "bit-set" if opcode == 0xC0 else "bit-clear",
        _astatx_bit_field(right, value),
    )


def shift_bset(rn, rx, ry, left, right, values, special, approx_recips) -> tuple:
    return _shift_bitset_impl(rn, rx, ry, left, right, 0xC0)


def shift_bclr(rn, rx, ry, left, right, values, special, approx_recips) -> tuple:
    return _shift_bitset_impl(rn, rx, ry, left, right, 0xC4)


# PRM Table 17-9 and p. 23-5: SHIFTOP 11001000 is
# RN = btgl RX by RY.  Positions outside the 32-bit field leave RX
# unchanged.
def shift_btgl(rn, rx, ry, left, right, values, special, approx_recips) -> tuple:
    value: Operand
    if not isinstance(right, Const):
        value = Unknown("btgl R%d by R%d" % (rx, ry))
    elif right.value > 31:
        value = left
    else:
        value = _bitwise(
            left,
            Const(1 << right.value),
            "btgl R%d by R%d" % (rx, ry),
            lambda a, b: a ^ b,
        )
    return rn, value, "bit-toggle", _astatx_bit_field(right, value)


# PRM Table 18-9 and pp. 24-5--24-6: SHIFTOP 11001100 is
# btst RX by RY. It changes status flags only and has no RN result.
def shift_btst(rn, rx, ry, left, right, values, special, approx_recips) -> tuple:
    return rn, left, "bit-test", _astatx_btst(left, right)


# PGR p.11-88 (pgr.txt:23353), opcode 0111 0000: RN = BFFWRP -- reads
# the bit-FIFO write pointer this tracer tracks via ShiftImm opcode
# 0x1f/cu=2 opcode 0x7c below (special-dict key "BFFWRP"; Unknown until
# one of those has run). SF is documented "Not affected" so it is left
# out of UPDATES entirely (stays whatever it already was).
def shift_bffwrp_read(rn, rx, ry, left, right, values, special, approx_recips) -> tuple:
    value = (special or {}).get("BFFWRP", Unknown("uninitialized BFFWRP"))
    updates: dict[int, bool | None] = {SS_BIT: False, SZ_BIT: False, SV_BIT: False}
    return rn, value, "bffwrp-read", _astatx_from_updates(updates)


# PGR p.11-89 (pgr.txt:23379), opcode 0111 1100: BFFWRP = RN|<data7> --
# the register-operand twin of ShiftImm opcode 0x1f below. The register
# form reads the RN-positioned field as its SOURCE, not a destination,
# per PGR's own text: "Updates write pointer from Rn ... Only 7 least
# significant bits of Rn are written."
def shift_bffwrp_write(
    rn, rx, ry, left, right, values, special, approx_recips
) -> tuple:
    source = _ureg(values, rn)
    new_wrp = (
        Const(source.value & 0x7F)
        if isinstance(source, Const)
        else Unknown("BFFWRP = R%d" % rn)
    )
    updates = {
        SS_BIT: False,
        SZ_BIT: False,
        SV_BIT: (new_wrp.value > 64) if isinstance(new_wrp, Const) else None,
        SF_BIT: (new_wrp.value >= 32) if isinstance(new_wrp, Const) else None,
    }
    return "BFFWRP", new_wrp, "bffwrp-write", _astatx_from_updates(updates)


# Shifter opcode 1011 0000: absent from both public sources' shifter
# tables (PRM Table 17-9, p.17-10/17-11, and PGR Table 12-11, p.580-581,
# transcribed in full -- neither lists any 0xA0-0xBF row). Seen at
# `sw 0x1cd002`. Rather than guess an operation from an undocumented
# opcode, decode it (so the walk does not desync) and leave both the
# result and its flags Unknown, per this file's existing rule for gaps
# the manuals do not cover.
def shift_undocumented_b0(
    rn, rx, ry, left, right, values, special, approx_recips
) -> tuple:
    label = "shift opcode 0xb0 R%d, R%d (undocumented; no public source)" % (rx, ry)
    return (
        rn,
        Unknown(label),
        "shift-undocumented-b0",
        lambda astatx: _astatx_forget(astatx, ALU_FLAGS_MASK),
    )


# Shifter opcode 0001 0100: absent from PRM Table 17-9, PGR Table 12-11
# and tools/sharcspec/compute_table.json's shiftop_shiftimm table alike
# (all three transcribed in full for this project; none has a 00010100
# row). Seen at `sw 0x1cadac`, between two otherwise-ordinary
# instructions, so not obviously misdecoded data -- decoded like the
# 0xb0 case above (so the walk does not desync) rather than guessed.
def shift_undocumented_14(
    rn, rx, ry, left, right, values, special, approx_recips
) -> tuple:
    label = "shift opcode 0x14 R%d, R%d (undocumented; no public source)" % (rx, ry)
    return (
        rn,
        Unknown(label),
        "shift-undocumented-14",
        lambda astatx: _astatx_forget(astatx, SHIFT_FLAGS_MASK),
    )


SHIFT_OPS: dict[int, Handler] = {
    0x00: shift_logical,
    0x04: shift_arithmetic,
    0x20: shift_logical_or,
    0x88: shift_leftz,
    0x8C: shift_lefto,
    0xC0: shift_bset,
    0xC4: shift_bclr,
    0xC8: shift_btgl,
    0xCC: shift_btst,
    0x70: shift_bffwrp_read,
    0x7C: shift_bffwrp_write,
    # Undocumented; kept in this same table (not a separate one) so cu=2
    # dispatch stays a single lookup -- see each handler's own docstring.
    0xB0: shift_undocumented_b0,
    0x14: shift_undocumented_14,
}
