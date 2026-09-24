"""Fixed and float ALU compute (cu=0): handler bodies dispatched by
tools/sharc_core/compute.py's ALU_OPS table, plus the dual add/subtract
range match (opcode top nibble 0111/1111) that compute.py calls directly.

Handler bodies moved verbatim from the cu==0 branches of
tools/sharc_trace.py's old _compute if-chain; each still carries that
branch's own PRM/PGR citation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from .encoding import (
    AC_BIT,
    AF_BIT,
    AI_BIT,
    AN_BIT,
    AS_BIT,
    AV_BIT,
    AZ_BIT,
    UREG_CODES,
)
from .flags import (
    _alu_arith_updates,
    _astatx_abs,
    _astatx_alu_arith,
    _astatx_alu_arith_ci,
    _astatx_alu_logical,
    _astatx_compare,
    _astatx_compare_float,
    _astatx_from_updates,
    _compare_flags,
    _compare_flags_float,
    _float_alu_updates,
    _or_updates,
)
from .floats import (
    _approx_recips,
    _fixed_to_float,
    _fixed_to_float_scaled,
    _float_binary,
    _float_clip,
    _float_copysign,
    _float_logb,
    _float_mantissa,
    _float_max,
    _float_min,
    _float_round32,
    _float_scalb,
    _float_to_fixed,
    _float_to_fixed_trunc,
    _float_unary,
    _scale_fixed_input,
)
from .state import _ureg_raw
from .values import (
    Const,
    Operand,
    Unknown,
    Value,
    _add,
    _astatx_known_bit,
    _bitwise,
    _not,
    _signed32,
    _subtract,
)

Handler = Callable[
    [
        int,
        int,
        int,
        Value,
        Value,
        Mapping[int, Value],
        Mapping[str, Value] | None,
        bool,
    ],
    tuple,
]


def dual_add_subtract(
    rn: int,
    rs: int,
    rx: int,
    ry: int,
    left: Operand,
    right: Operand,
    float_form: bool,
) -> tuple:
    """PRM Table 18-10 (p.433) / Table 18-13 (p.434): Dual Add/Subtract is a
    single-function ALU op (mf=0, not multifunction) whose opcode top
    nibble (bits 19:16, i.e. this OPCODE's top nibble) is 0111 (fixed) or
    1111 (float); the low nibble (bits 15:12) is not part of the opcode at
    all -- it is RS, a second 4-bit result register alongside
    RA=RN/FN at bits 11:8 (already read above as RN)."""
    if float_form:
        add_value, add_overflow, add_invalid = _float_binary(
            left, right, "F%d + F%d" % (rx, ry), lambda a, b: a + b
        )
        sub_value, sub_overflow, sub_invalid = _float_binary(
            left, right, "F%d - F%d" % (rx, ry), lambda a, b: a - b
        )
        updates = _or_updates(
            _float_alu_updates(add_value, av=add_overflow, ai=add_invalid),
            _float_alu_updates(sub_value, av=sub_overflow, ai=sub_invalid),
        )
        operation = "float-dual-add-subtract"
    else:
        add_value = _add(left, right, "R%d + R%d" % (rx, ry))
        sub_value = _subtract(left, right, "R%d - R%d" % (rx, ry), same_source=rx == ry)
        updates = _or_updates(
            _alu_arith_updates(left, right, False),
            _alu_arith_updates(left, right, True, same_source=rx == ry),
        )
        operation = "dual-add-subtract"
    return (
        (rn, rs),
        (add_value, sub_value),
        operation,
        _astatx_from_updates(updates),
    )


# PRM Table 17-5: ALUOP 00000001/00000010 are add/subtract.
def alu_add(rn, rx, ry, left, right, values, special, approx_recips) -> tuple:
    value = _add(left, right, "R%d + R%d" % (rx, ry))
    return rn, value, "add", _astatx_alu_arith(left, right, False)


def alu_subtract(rn, rx, ry, left, right, values, special, approx_recips) -> tuple:
    same_source = rx == ry
    value = _subtract(left, right, "R%d - R%d" % (rx, ry), same_source=same_source)
    return (
        rn,
        value,
        "subtract",
        _astatx_alu_arith(left, right, True, same_source=same_source),
    )


# PRM p.438-439 / PGR Table 12-3 (p.573): ALUOP 00000101 is
# RN = RX + RY + ci (add with carry) and 00000110 is RN = RX - RY + ci -
# 1 (subtract with borrow), both using ASTATX's AC bit as an explicit
# carry-in read before this instruction's own flags are written. The
# identity RX - RY + ci - 1 = RX + ~RY + ci (see ``_arith_flag_bits_ci``)
# lets both share one value formula: add RY (or its one's complement for
# subtract) plus a constant offset of ci (add) or ci-1 (subtract).
def _alu_carry_impl(rn, rx, ry, left, right, values, subtract: bool) -> tuple:
    astatx = _ureg_raw(values, UREG_CODES["ASTATX"])
    carry_in = _astatx_known_bit(astatx, AC_BIT)
    label = "R%d %s R%d + ci%s" % (
        rx,
        "-" if subtract else "+",
        ry,
        " - 1" if subtract else "",
    )
    value: Operand
    if carry_in is None:
        value = Unknown(label)
    else:
        base = (
            _subtract(left, right, label, same_source=rx == ry)
            if subtract
            else _add(left, right, label)
        )
        offset = (1 if carry_in else 0) - (1 if subtract else 0)
        value = _add(base, Const(offset), label)
    operation = "subtract-with-borrow" if subtract else "add-with-carry"
    return (
        rn,
        value,
        operation,
        _astatx_alu_arith_ci(left, right, subtract, carry_in),
    )


def alu_add_with_carry(
    rn, rx, ry, left, right, values, special, approx_recips
) -> tuple:
    return _alu_carry_impl(rn, rx, ry, left, right, values, False)


def alu_subtract_with_borrow(
    rn, rx, ry, left, right, values, special, approx_recips
) -> tuple:
    return _alu_carry_impl(rn, rx, ry, left, right, values, True)


# PGR p.11-9/11-10 (pgr.txt:20570/20606), opcodes 0010 0101/0010 0110:
# RN = RX + ci / RN = RX + ci - 1 -- the single-operand twins of
# 0x05/0x06 above (no RY; PGR's flag table is identical to the RY form),
# so they reuse the same value/flags formulas with RY forced to 0.
def _alu_carry_noy_impl(rn, rx, left, values, subtract: bool) -> tuple:
    astatx = _ureg_raw(values, UREG_CODES["ASTATX"])
    carry_in = _astatx_known_bit(astatx, AC_BIT)
    label = "R%d + ci%s" % (rx, " - 1" if subtract else "")
    value: Operand
    if carry_in is None:
        value = Unknown(label)
    else:
        offset = (1 if carry_in else 0) - (1 if subtract else 0)
        value = _add(left, Const(offset), label)
    operation = "subtract-with-borrow" if subtract else "add-with-carry"
    return (
        rn,
        value,
        operation,
        _astatx_alu_arith_ci(left, Const(0), subtract, carry_in),
    )


def alu_add_with_carry_noy(
    rn, rx, ry, left, right, values, special, approx_recips
) -> tuple:
    return _alu_carry_noy_impl(rn, rx, left, values, False)


def alu_subtract_with_borrow_noy(
    rn, rx, ry, left, right, values, special, approx_recips
) -> tuple:
    return _alu_carry_noy_impl(rn, rx, left, values, True)


# PRM Table 18-5: ALUOP 00001010 is signed comp(RX, RY) and 00001011 is
# unsigned compu(RX, RY). Both update status only, so the tracer records
# the comparison without writing RN; the value carries the new flags.
def _alu_compare_impl(rn, rx, ry, left, right, signed: bool) -> tuple:
    label = "%s R%d, R%d" % ("comp" if signed else "compu", rx, ry)
    value = _compare_flags(left, right, signed, label)
    return rn, value, "compare", _astatx_compare(value)


def alu_compare_signed(
    rn, rx, ry, left, right, values, special, approx_recips
) -> tuple:
    return _alu_compare_impl(rn, rx, ry, left, right, True)


def alu_compare_unsigned(
    rn, rx, ry, left, right, values, special, approx_recips
) -> tuple:
    return _alu_compare_impl(rn, rx, ry, left, right, False)


def alu_pass(rn, rx, ry, left, right, values, special, approx_recips) -> tuple:
    return rn, left, "pass", _astatx_alu_logical(left)


# PRM Table 18-5 and p. 19-10: ALUOP 00100010 is RN = -RX, the two's
# complement, with the same flags as 0 - RX.
def alu_negate(rn, rx, ry, left, right, values, special, approx_recips) -> tuple:
    value = _subtract(Const(0), left, "-R%d" % rx)
    return rn, value, "negate", _astatx_alu_arith(Const(0), left, True)


def alu_increment(rn, rx, ry, left, right, values, special, approx_recips) -> tuple:
    value = _add(left, Const(1), "R%d + 1" % rx)
    return rn, value, "increment", _astatx_alu_arith(left, Const(1), False)


# PRM Table 18-5 and p. 19-9: ALUOP 00101010 is RN = RX - 1.
def alu_decrement(rn, rx, ry, left, right, values, special, approx_recips) -> tuple:
    value = _add(left, Const(-1), "R%d - 1" % rx)
    return rn, value, "decrement", _astatx_alu_arith(left, Const(1), True)


# PGR p.11-13/11-14 (pgr.txt:20750), opcode 0011 0000: RN = ABS RX. Value
# and AC/AV/AN/AZ come from the same 0-RX adder as negate (0x22 above,
# "The ABS of the minimum negative number ... causes an overflow" only
# makes sense if ABS always runs the 0-RX path, even for a positive RX);
# unlike negate, AS is data-dependent here (RX's own sign) rather than
# architecturally cleared.
def alu_abs(rn, rx, ry, left, right, values, special, approx_recips) -> tuple:
    if isinstance(left, Const):
        signed = _signed32(left.value)
        value = left if signed >= 0 else _subtract(Const(0), left, "abs R%d" % rx)
    else:
        value = Unknown("abs R%d" % rx)
    return rn, value, "abs", _astatx_abs(left)


# PRM Table 18-5: ALUOP 01000000..01000010 are the integer logical
# operations AND, OR, and XOR.
_LOGICAL_OPS = {
    0x40: ("and", lambda a, b: a & b),
    0x41: ("or", lambda a, b: a | b),
    0x42: ("xor", lambda a, b: a ^ b),
}


def _alu_logical_binary(opcode, rn, rx, ry, left, right) -> tuple:
    name, operation = _LOGICAL_OPS[opcode]
    value = _bitwise(left, right, "R%d %s R%d" % (rx, name, ry), operation)
    return rn, value, name, _astatx_alu_logical(value)


def alu_and(rn, rx, ry, left, right, values, special, approx_recips) -> tuple:
    return _alu_logical_binary(0x40, rn, rx, ry, left, right)


def alu_or(rn, rx, ry, left, right, values, special, approx_recips) -> tuple:
    return _alu_logical_binary(0x41, rn, rx, ry, left, right)


def alu_xor(rn, rx, ry, left, right, values, special, approx_recips) -> tuple:
    return _alu_logical_binary(0x42, rn, rx, ry, left, right)


# PGR p.11-19 (pgr.txt:20918), opcode 0100 0011: RN = NOT RX. Same
# AZ/AN-from-result, AC/AV/AS/AI-cleared rule as pass/and/or/xor.
def alu_not(rn, rx, ry, left, right, values, special, approx_recips) -> tuple:
    value = _not(left, "not R%d" % rx)
    return rn, value, "not", _astatx_alu_logical(value)


# PRM Table 18-5 (p.425-427), float rows; per-op flags cited at each
# branch (PRM Table 3-3, pp.3-8/3-9, cross-checked against the classic
# PGR's per-instruction pages, which spell out AZ/AN/AV/AI exactly where
# the SHARC+ PRM only marks a column "*"/data-dependent).
def alu_float_add(rn, rx, ry, left, right, values, special, approx_recips) -> tuple:
    value, overflow, invalid = _float_binary(
        left, right, "F%d + F%d" % (rx, ry), lambda a, b: a + b
    )
    return (
        rn,
        value,
        "float-add",
        _astatx_from_updates(_float_alu_updates(value, av=overflow, ai=invalid)),
    )


def alu_float_subtract(
    rn, rx, ry, left, right, values, special, approx_recips
) -> tuple:
    value, overflow, invalid = _float_binary(
        left, right, "F%d - F%d" % (rx, ry), lambda a, b: a - b
    )
    return (
        rn,
        value,
        "float-subtract",
        _astatx_from_updates(_float_alu_updates(value, av=overflow, ai=invalid)),
    )


# PGR p.11-27 (pgr.txt:21185), opcode 1001 0010: Fn = abs(Fx - Fy).
# Magnitude (and so AV/AZ) is identical to plain float-subtract above;
# only the sign bit changes (cleared) and AN is architecturally fixed 0
# rather than following the result's sign.
def alu_float_abs_subtract(
    rn, rx, ry, left, right, values, special, approx_recips
) -> tuple:
    diff, overflow, invalid = _float_binary(
        left, right, "F%d - F%d" % (rx, ry), lambda a, b: a - b
    )
    value = Const(diff.value & 0x7FFFFFFF) if isinstance(diff, Const) else diff
    return (
        rn,
        value,
        "float-abs-subtract",
        _astatx_from_updates(
            _float_alu_updates(value, av=overflow, an_zero=True, ai=invalid)
        ),
    )


# PGR p.11-29: comp(Fx, Fy).
def alu_float_compare(rn, rx, ry, left, right, values, special, approx_recips) -> tuple:
    label = "comp F%d, F%d" % (rx, ry)
    value, invalid = _compare_flags_float(left, right, label)
    return rn, value, "float-compare", _astatx_compare_float(value, invalid)


# PGR p.11-32: Fn = pass Fx.
def alu_float_pass(rn, rx, ry, left, right, values, special, approx_recips) -> tuple:
    value, overflow, invalid = _float_unary(left, "pass F%d" % rx, lambda a: a)
    return (
        rn,
        value,
        "float-pass",
        _astatx_from_updates(_float_alu_updates(value, av=False, ai=invalid)),
    )


# PGR p.11-30: Fn = -Fx.
def alu_float_negate(rn, rx, ry, left, right, values, special, approx_recips) -> tuple:
    value, overflow, invalid = _float_unary(left, "-F%d" % rx, lambda a: -a)
    return (
        rn,
        value,
        "float-negate",
        _astatx_from_updates(_float_alu_updates(value, av=False, ai=invalid)),
    )


# PRM Table 18-5 opcode 1010 0101 (p.20-8) / PGR Table 12-4 opcode
# 1010 0101, p.11-33: Fn = rnd Fx.
def alu_float_round32(rn, rx, ry, left, right, values, special, approx_recips) -> tuple:
    value, invalid = _float_round32(left, "rnd F%d" % rx)
    return (
        rn,
        value,
        "float-round32",
        _astatx_from_updates(_float_alu_updates(value, av=False, ai=invalid)),
    )


# PGR p.11-34/11-35: Rn = mant Fx. Bespoke flag dict, not
# ``_float_alu_updates``: the result is an unsigned-magnitude fixed
# word (no sign bit of its own to derive AZ/AN from), AN is
# architecturally fixed 0 (PRM Table 3-3), and AS/AV/AI come from the
# *input*'s sign/infinity/NAN rather than the result.
def alu_float_mant(rn, rx, ry, left, right, values, special, approx_recips) -> tuple:
    value, overflow, negative, invalid = _float_mantissa(left, "mant F%d" % rx)
    updates = {
        AC_BIT: False,
        AF_BIT: True,
        AN_BIT: False,
        AV_BIT: overflow,
        AS_BIT: negative,
        AI_BIT: invalid,
        AZ_BIT: (value.value == 0) if isinstance(value, Const) else None,
    }
    return rn, value, "mant", _astatx_from_updates(updates)


# PGR p.11-36 (pgr.txt:21522), opcode 1100 0001: RN = LOGB FX. Bespoke
# flag dict like ``mant`` above: RESULT is a plain fixed-point integer
# (not float), so AN/AZ come from its own bits rather than through
# ``_float_alu_updates``, and AF is set (float-unit op with a
# fixed-point result, same convention as ``mant``/``fix``/``trunc``).
def alu_float_logb(rn, rx, ry, left, right, values, special, approx_recips) -> tuple:
    mode1 = _ureg_raw(values, UREG_CODES["MODE1"])
    value, overflow, invalid = _float_logb(left, mode1, "logb F%d" % rx)
    updates = {
        AC_BIT: False,
        AF_BIT: True,
        AS_BIT: False,
        AV_BIT: overflow,
        AI_BIT: invalid,
        AZ_BIT: (value.value == 0) if isinstance(value, Const) else None,
        AN_BIT: bool(value.value & 0x80000000) if isinstance(value, Const) else None,
    }
    return rn, value, "logb", _astatx_from_updates(updates)


# PGR p.11-31: Fn = abs Fx. AN fixed 0; AS carries the *input*'s sign.
def alu_float_abs(rn, rx, ry, left, right, values, special, approx_recips) -> tuple:
    value, overflow, invalid = _float_unary(left, "abs F%d" % rx, abs)
    return (
        rn,
        value,
        "float-abs",
        _astatx_from_updates(
            _float_alu_updates(
                value, av=False, an_zero=True, as_source=left, ai=invalid
            )
        ),
    )


# PGR p.11-33: Fn = scalb Fx by Ry. Unlike abs/pass/etc., AN here
# follows the *result*'s sign (PRM Table 3-3 marks AN '*', not 0), so
# this reuses ``_float_alu_updates``'s default (as_source=None,
# an_zero=False) rather than the abs-style override.
def alu_float_scalb(rn, rx, ry, left, right, values, special, approx_recips) -> tuple:
    value, overflow, invalid = _float_scalb(left, right, "scalb F%d by R%d" % (rx, ry))
    return (
        rn,
        value,
        "float-scalb",
        _astatx_from_updates(_float_alu_updates(value, av=overflow, ai=invalid)),
    )


# PGR p.11-20/11-21: Rn = min/max(Rx, Ry) -- fixed-point, not the
# float min/max at 0xE1/0xE2 below. AV/AC/AS/AI/AF are all fixed 0
# (PRM Table 3-2, "AF Flag = 0"); only AZ/AN follow the chosen
# operand, the same rule ``_astatx_alu_logical`` already implements
# for pass/not/and/or/xor.
def _alu_minmax_fixed_impl(rn, rx, ry, left, right, minimum: bool) -> tuple:
    name = "min" if minimum else "max"
    value: Operand
    if isinstance(left, Const) and isinstance(right, Const):
        a, b = _signed32(left.value), _signed32(right.value)
        pick_left = (a <= b) if minimum else (a >= b)
        value = left if pick_left else right
    else:
        value = Unknown("%s(R%d, R%d)" % (name, rx, ry))
    return rn, value, name, _astatx_alu_logical(value)


def alu_min_fixed(rn, rx, ry, left, right, values, special, approx_recips) -> tuple:
    return _alu_minmax_fixed_impl(rn, rx, ry, left, right, True)


def alu_max_fixed(rn, rx, ry, left, right, values, special, approx_recips) -> tuple:
    return _alu_minmax_fixed_impl(rn, rx, ry, left, right, False)


# PRM p.19-19, opcode 1110 0000 / PGR Table 12-4 p.574, same opcode: Fn =
# Fx copysign Fy.
def alu_float_copysign(
    rn, rx, ry, left, right, values, special, approx_recips
) -> tuple:
    value, invalid = _float_copysign(left, right, "F%d copysign F%d" % (rx, ry))
    return (
        rn,
        value,
        "float-copysign",
        _astatx_from_updates(_float_alu_updates(value, av=False, ai=invalid)),
    )


# PGR p.11-46/11-47: Fn = min/max(Fx, Fy).
def _alu_minmax_float_impl(rn, rx, ry, left, right, minimum: bool) -> tuple:
    name = "min" if minimum else "max"
    combine = _float_min if minimum else _float_max
    value, overflow, invalid = _float_binary(
        left, right, "%s(F%d, F%d)" % (name, rx, ry), combine
    )
    return (
        rn,
        value,
        "float-" + name,
        _astatx_from_updates(_float_alu_updates(value, av=False, ai=invalid)),
    )


def alu_min_float(rn, rx, ry, left, right, values, special, approx_recips) -> tuple:
    return _alu_minmax_float_impl(rn, rx, ry, left, right, True)


def alu_max_float(rn, rx, ry, left, right, values, special, approx_recips) -> tuple:
    return _alu_minmax_float_impl(rn, rx, ry, left, right, False)


# PGR p.11-48 / PRM p.3-6: Fn = clip Fx by Fy.
def alu_float_clip(rn, rx, ry, left, right, values, special, approx_recips) -> tuple:
    value, overflow, invalid = _float_binary(
        left, right, "clip F%d by F%d" % (rx, ry), _float_clip
    )
    return (
        rn,
        value,
        "float-clip",
        _astatx_from_updates(_float_alu_updates(value, av=False, ai=invalid)),
    )


# PRM p.427/PGR p.11-39 "without scaling factor": Fn = float Rx.
def alu_float_convert(rn, rx, ry, left, right, values, special, approx_recips) -> tuple:
    value, invalid = _fixed_to_float(left, "float R%d" % rx)
    return (
        rn,
        value,
        "float-convert",
        _astatx_from_updates(_float_alu_updates(value, av=False, ai=invalid)),
    )


# PGR p.11-39 "with scaling factor" / PRM p.19-.. : Fn = float Rx by Ry.
# AV is data-dependent here (unlike the unscaled form above, where an
# int32 input can never overflow float32 range), per PGR Table 3-3.
def alu_float_convert_scaled(
    rn, rx, ry, left, right, values, special, approx_recips
) -> tuple:
    value, overflow = _fixed_to_float_scaled(left, right, "float R%d by R%d" % (rx, ry))
    return (
        rn,
        value,
        "float-convert-scaled",
        _astatx_from_updates(_float_alu_updates(value, av=overflow, ai=False)),
    )


# PRM p.24-.. / PGR p.11-36..11-38, opcode 1100 1001: Rn = fix Fx
# (rounds to nearest or truncates per MODE1.TRUNCATE; see
# ``_float_to_fixed``).
def alu_fix(rn, rx, ry, left, right, values, special, approx_recips) -> tuple:
    mode1 = _ureg_raw(values, UREG_CODES["MODE1"])
    value, overflow, invalid = _float_to_fixed(left, mode1, False, "fix F%d" % rx)
    return (
        rn,
        value,
        "fix",
        _astatx_from_updates(_float_alu_updates(value, av=overflow, ai=invalid)),
    )


# PRM p.427/PGR p.11-37: Rn = trunc Fx.
def alu_trunc(rn, rx, ry, left, right, values, special, approx_recips) -> tuple:
    mode1 = _ureg_raw(values, UREG_CODES["MODE1"])
    value, overflow, invalid = _float_to_fixed_trunc(left, mode1, "trunc F%d" % rx)
    return (
        rn,
        value,
        "trunc",
        _astatx_from_updates(_float_alu_updates(value, av=overflow, ai=invalid)),
    )


# PGR p.11-36..11-38, opcode 1101 1001: Rn = fix Fx by Ry -- Ry's
# exponent-add (``_scale_fixed_input``) applied before the same fix
# conversion as 0xC9; PGR Table 3-3 gives it the identical flag columns
# as the unscaled form.
def alu_fix_scaled(rn, rx, ry, left, right, values, special, approx_recips) -> tuple:
    mode1 = _ureg_raw(values, UREG_CODES["MODE1"])
    scaled = _scale_fixed_input(left, right, "F%d * 2**R%d" % (rx, ry))
    value, overflow, invalid = _float_to_fixed(
        scaled, mode1, False, "fix F%d by R%d" % (rx, ry)
    )
    return (
        rn,
        value,
        "fix-scaled",
        _astatx_from_updates(_float_alu_updates(value, av=overflow, ai=invalid)),
    )


# PGR p.11-36..11-38, opcode 1101 1101: Rn = trunc Fx by Ry -- Ry's
# exponent-add applied before the same truncation as 0xCD; PGR Table 3-3
# again gives it the unscaled form's flag columns.
def alu_trunc_scaled(rn, rx, ry, left, right, values, special, approx_recips) -> tuple:
    mode1 = _ureg_raw(values, UREG_CODES["MODE1"])
    scaled = _scale_fixed_input(left, right, "F%d * 2**R%d" % (rx, ry))
    value, overflow, invalid = _float_to_fixed(
        scaled, mode1, True, "trunc F%d by R%d" % (rx, ry)
    )
    return (
        rn,
        value,
        "trunc-scaled",
        _astatx_from_updates(_float_alu_updates(value, av=overflow, ai=invalid)),
    )


# PRM p.427 / PGR p.11-44/11-45: Fn = recips/rsqrts Fx -- iterative
# reciprocal/reciprocal-sqrt seed instructions. The seed mantissa comes
# from an ROM lookup table the public manuals do not print, so this
# tracer decodes the instruction (unblocking whatever reads its flags or
# continues past it) without claiming a numeric seed value it cannot
# verify; AV/AI are genuinely data-dependent here and left unknown too.
def _alu_recip_seed_impl(rn, rx, left, name: str, approx_recips: bool) -> tuple:
    # --approx-recips only covers recips: rsqrts's seed exponent rule
    # (floor(e/2), PRM p.19-18) couples to the exponent's LSB in a way
    # that is not a trivial mirror of recips's rule, so it is left
    # Unknown until that is separately worked out.
    if name == "recips" and approx_recips:
        value, updates = _approx_recips(left)
        return rn, value, "float-recips-seed-approx", _astatx_from_updates(updates)
    label = "%s F%d (iterative seed, not numerically modeled)" % (name, rx)
    return (
        rn,
        Unknown(label),
        "float-" + name + "-seed",
        _astatx_from_updates(_float_alu_updates(Unknown(label), av=None, ai=None)),
    )


def alu_recips_seed(rn, rx, ry, left, right, values, special, approx_recips) -> tuple:
    return _alu_recip_seed_impl(rn, rx, left, "recips", approx_recips)


def alu_rsqrts_seed(rn, rx, ry, left, right, values, special, approx_recips) -> tuple:
    return _alu_recip_seed_impl(rn, rx, left, "rsqrts", approx_recips)


ALU_OPS: dict[int, Handler] = {
    0x01: alu_add,
    0x02: alu_subtract,
    0x05: alu_add_with_carry,
    0x06: alu_subtract_with_borrow,
    0x25: alu_add_with_carry_noy,
    0x26: alu_subtract_with_borrow_noy,
    0x0A: alu_compare_signed,
    0x0B: alu_compare_unsigned,
    0x21: alu_pass,
    0x22: alu_negate,
    0x29: alu_increment,
    0x2A: alu_decrement,
    0x30: alu_abs,
    0x40: alu_and,
    0x41: alu_or,
    0x42: alu_xor,
    0x43: alu_not,
    0x81: alu_float_add,
    0x82: alu_float_subtract,
    0x92: alu_float_abs_subtract,
    0x8A: alu_float_compare,
    0xA1: alu_float_pass,
    0xA2: alu_float_negate,
    0xA5: alu_float_round32,
    0xAD: alu_float_mant,
    0xC1: alu_float_logb,
    0xB0: alu_float_abs,
    0xBD: alu_float_scalb,
    0x61: alu_min_fixed,
    0x62: alu_max_fixed,
    0xE0: alu_float_copysign,
    0xE1: alu_min_float,
    0xE2: alu_max_float,
    0xE3: alu_float_clip,
    0xCA: alu_float_convert,
    0xDA: alu_float_convert_scaled,
    0xC9: alu_fix,
    0xCD: alu_trunc,
    0xD9: alu_fix_scaled,
    0xDD: alu_trunc_scaled,
    0xC4: alu_recips_seed,
    0xC5: alu_rsqrts_seed,
}
