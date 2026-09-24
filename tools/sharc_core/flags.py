"""ASTATX flag updates for compute results.

Moved verbatim from tools/sharc_trace.py.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Callable, Dict, Optional

from .encoding import (
    AC_BIT,
    AF_BIT,
    AI_BIT,
    ALU_FLAGS_MASK,
    AN_BIT,
    AS_BIT,
    AV_BIT,
    AZ_BIT,
    MI_BIT,
    MN_BIT,
    MULT_FLAGS_MASK,
    MU_BIT,
    MV_BIT,
    SS_BIT,
    SV_BIT,
    SZ_BIT,
)
from .values import (
    Const,
    PartialConst,
    Unknown,
    Value,
    _astatx_known_bit,
    _signed32,
)
from .floats import (
    _float32,
)


def _compare_flags(left: Value, right: Value, signed: bool, label: str) -> Value:
    """Return AZ (bit 0), AN (bit 2) and the new CACC MSB (bit 31) of a compare.

    PRM comp/compu (pp. 18-5, 18-6): AZ when RX equals RY, AN when RX is
    smaller, and the CACC MSB when RX is greater.
    """
    if not isinstance(left, Const) or not isinstance(right, Const):
        return Unknown(label)
    x, y = left.value & 0xFFFFFFFF, right.value & 0xFFFFFFFF
    if signed:
        x, y = _signed32(x), _signed32(y)
    return Const(
        (0x1 if x == y else 0)
        | (0x4 if x < y else 0)
        | (0x80000000 if x > y else 0)
    )


def _compare_flags_float(
    left: Value, right: Value, label: str
) -> tuple[Value, Optional[bool]]:
    """comp(FX, FY) (PRM Table 18-5 opcode 0x8A, p.426; PGR p.11-29).

    Same bit-0 (AZ)/bit-2 (AN)/bit-31 (new CACC MSB) value encoding
    ``_compare_flags`` uses for the fixed-point comp/compu, consumed by
    ``_astatx_compare``'s CACC shift-register logic. An unordered (NAN)
    compare sets none of those bits (PGR doesn't document AZ/AN/CACC firing
    on an unordered compare) and instead reports the invalid flag, which
    the caller applies on top via ``_astatx_compare``'s AI override.
    """
    a, b = _float32(left), _float32(right)
    if a is None or b is None:
        return Unknown(label), None
    if math.isnan(a) or math.isnan(b):
        return Const(0), True
    return (
        Const(
            (0x1 if a == b else 0)
            | (0x4 if a < b else 0)
            | (0x80000000 if a > b else 0)
        ),
        False,
    )


def _bits_to_updates(mask: int, bits: Optional[int]) -> Dict[int, Optional[bool]]:
    """Expand an optional MASK-shaped flag bit pattern into an
    ``_astatx_apply_bits()`` updates dict; BITS=None forgets every bit in
    MASK."""
    updates: Dict[int, Optional[bool]] = {}
    bit = 0
    while (1 << bit) <= mask:
        if mask & (1 << bit):
            updates[bit] = None if bits is None else bool(bits & (1 << bit))
        bit += 1
    return updates


def _or_updates(
    a: Dict[int, Optional[bool]], b: Dict[int, Optional[bool]]
) -> Dict[int, Optional[bool]]:
    """Kleene-OR two ASTATX update dicts bit by bit (PRM p.3-21/3-22:
    "Multifunction Computations ... in the dual add/subtract computation,
    the ALU flags from the two operations are ORed together"). True beats
    anything; a bit present in only one dict keeps that dict's own value."""
    merged = dict(a)
    for bit, b_value in b.items():
        a_value = merged.get(bit, False)
        if a_value is True or b_value is True:
            merged[bit] = True
        elif a_value is None or b_value is None:
            merged[bit] = None
        else:
            merged[bit] = False
    return merged


def _alu_arith_updates(
    a: Value, b: Value, subtract: bool, *, same_source: bool = False
) -> Dict[int, Optional[bool]]:
    """Dict-returning counterpart of ``_astatx_alu_arith`` (PRM pp.439-440,
    446-447), for callers -- the fixed-point dual add/subtract -- that need
    to OR two such results together before applying either to ASTATX.

    SAME_SOURCE mirrors ``_subtract``'s: with SUBTRACT=True it means A and B
    are the same operand read twice, so the flags are those of 0-0 (AZ/AC
    set, AN/AV clear) regardless of what value that operand held."""
    if same_source and subtract:
        return _bits_to_updates(ALU_FLAGS_MASK, _arith_flag_bits(Const(0), Const(0), True))
    if isinstance(a, Const) and isinstance(b, Const):
        return _bits_to_updates(ALU_FLAGS_MASK, _arith_flag_bits(a, b, subtract))
    return _bits_to_updates(ALU_FLAGS_MASK, None)


def _float_alu_updates(
    result: Value,
    *,
    av: Optional[bool] = False,
    an_zero: bool = False,
    as_source: Optional[Value] = None,
    ai: Optional[bool] = None,
) -> Dict[int, Optional[bool]]:
    """ASTATX update dict shared by the float ALU ops (PRM Table 3-3,
    pp.3-8/3-9; per-op PGR pages cited at each call site).

    AC is always 0 and AF is always 1 for a float ALU result. AZ/AN come
    from RESULT's bit pattern (both +0.0 and -0.0 count as AZ) unless the
    op's AN column is architecturally fixed to 0 (the abs-family:
    AN_ZERO=True). AS is 0 unless AS_SOURCE is given (FN=abs FX carries the
    *input*'s sign, PGR p.11-31). AV/AI are per-op data: pass the
    (overflowed, invalid) pair ``_float_binary``/``_float_unary`` computed,
    or an explicit fixed value for an op the table/PGR documents as always
    0 (e.g. FN=float RX's AV and AI).
    """
    updates: Dict[int, Optional[bool]] = {
        AC_BIT: False,
        AF_BIT: True,
        AV_BIT: av,
        AI_BIT: ai,
        AS_BIT: False if as_source is None else _astatx_known_bit(as_source, 31),
    }
    if isinstance(result, Const):
        bits = result.value
        updates[AZ_BIT] = (bits & 0x7FFFFFFF) == 0
        updates[AN_BIT] = False if an_zero else bool(bits & 0x80000000)
    else:
        updates[AZ_BIT] = None
        updates[AN_BIT] = False if an_zero else None
    return updates


def _astatx_from_updates(updates: Dict[int, Optional[bool]]) -> "Callable[[Value], Value]":
    """Wrap a pre-built updates dict as an ASTATX updater function, matching
    the ``Callable[[Value], Value]`` contract every other compute-table
    branch returns."""
    return lambda astatx: _astatx_apply_bits(astatx, updates)


def _astatx_compare_float(value: Value, invalid: Optional[bool]) -> "Callable[[Value], Value]":
    """Float comp (PRM Table 3-3 AI='*'; PGR p.11-29 spells it out: "Set if
    either of the input operands is a NAN"). Identical to
    ``_astatx_compare``'s AC/AV/AS-clear, AZ/AN/CACC-from-VALUE and
    CACC-shift behaviour (which needs the *old* ASTATX, so it is reused
    rather than duplicated); only AI and AF differ from the fixed-point
    comp/compu version, which the PRM documents as always 0/0 rather than
    float compare's AI=data-dependent, AF=1.
    """

    def update(astatx: Value) -> Value:
        base = _astatx_compare(value)(astatx)
        return _astatx_apply_bits(base, {AI_BIT: invalid, AF_BIT: True})

    return update


def _astatx_define(old: Value, mask: int, bits: int) -> Value:
    """Return OLD with MASK's bits set definitively to BITS (masked to MASK);
    bits outside MASK keep whatever knowledge OLD already carried."""
    mask &= 0xFFFFFFFF
    bits &= mask
    if isinstance(old, Const):
        return Const((old.value & ~mask) | bits)
    if isinstance(old, PartialConst):
        new_mask = old.mask | mask
        new_bits = (old.bits & ~mask) | bits
        return Const(new_bits) if new_mask == 0xFFFFFFFF else PartialConst(new_mask, new_bits)
    # Unknown (or a stray non-ASTATX Value type): only MASK becomes known.
    if not mask:
        return old
    return Const(bits) if mask == 0xFFFFFFFF else PartialConst(mask, bits)


def _astatx_forget(old: Value, mask: int) -> Value:
    """Return OLD with MASK's bits downgraded to unknown; other bits keep
    whatever knowledge OLD already carried."""
    mask &= 0xFFFFFFFF
    if isinstance(old, Const):
        new_mask = 0xFFFFFFFF & ~mask
        new_bits = old.value & new_mask
    elif isinstance(old, PartialConst):
        new_mask = old.mask & ~mask
        new_bits = old.bits & new_mask
    else:
        return old
    return Unknown("astatx bits forgotten") if new_mask == 0 else PartialConst(new_mask, new_bits)


def _astatx_apply_bits(old: Value, updates: Mapping[int, Optional[bool]]) -> Value:
    """Apply per-bit updates to an ASTATX-like value: True/False defines that
    bit, None forgets it (downgrades to unknown). Bits not mentioned in
    UPDATES are left exactly as OLD had them."""
    define_mask = define_bits = forget_mask = 0
    for bit, known in updates.items():
        if known is None:
            forget_mask |= 1 << bit
        else:
            define_mask |= 1 << bit
            if known:
                define_bits |= 1 << bit
    result = old
    if define_mask:
        result = _astatx_define(result, define_mask, define_bits)
    if forget_mask:
        result = _astatx_forget(result, forget_mask)
    return result


def _alu_result_bits(value: Const) -> int:
    """AN/AZ for a pass/not/and/or/xor result (PRM pp.449-452); AC/AV/AS/AI
    are always 0 for these."""
    bits = 0
    if value.value & 0x80000000:
        bits |= 1 << AN_BIT
    if value.value == 0:
        bits |= 1 << AZ_BIT
    return bits


def _arith_flag_bits(a: Const, b: Const, subtract: bool) -> int:
    """AC/AV/AN/AZ for add/subtract/increment/decrement (PRM pp.439-440,
    446-447); AS/AI are always 0.

    AC is the carry out of the MSB adder stage; AV is the XOR of the carries
    into and out of the MSB adder stage (the standard two's-complement
    signed-overflow test). Subtraction is modelled the way the ALU does it:
    add the one's complement of B with a forced carry-in of 1 (so decrement,
    RX - 1, is add(RX, 1, subtract=True), matching the PRM wording exactly).
    """
    A = a.value & 0xFFFFFFFF
    if subtract:
        b_eff, carry_in = (~b.value) & 0xFFFFFFFF, 1
    else:
        b_eff, carry_in = b.value & 0xFFFFFFFF, 0
    low31 = (A & 0x7FFFFFFF) + (b_eff & 0x7FFFFFFF) + carry_in
    carry_into_msb = (low31 >> 31) & 1
    full = A + b_eff + carry_in
    carry_out = (full >> 32) & 1
    result = full & 0xFFFFFFFF
    bits = 0
    if carry_out:
        bits |= 1 << AC_BIT
    if carry_into_msb ^ carry_out:
        bits |= 1 << AV_BIT
    if result & 0x80000000:
        bits |= 1 << AN_BIT
    if result == 0:
        bits |= 1 << AZ_BIT
    return bits


def _astatx_alu_logical(value: Value) -> "Callable[[Value], Value]":
    """pass/not/and/or/xor: AC/AV/AS/AI/AF cleared; AN/AZ from VALUE."""

    def update(astatx: Value) -> Value:
        if isinstance(value, Const):
            return _astatx_define(astatx, ALU_FLAGS_MASK, _alu_result_bits(value))
        return _astatx_forget(astatx, ALU_FLAGS_MASK)

    return update


def _astatx_alu_arith(
    a: Value, b: Value, subtract: bool, *, same_source: bool = False
) -> "Callable[[Value], Value]":
    """add/subtract/increment/decrement: AC/AV/AN/AZ from A and B; AS/AI/AF
    cleared.

    SAME_SOURCE mirrors ``_subtract``'s: with SUBTRACT=True it means A and B
    are the same operand read twice (the "Rn = Rn - Rn" self-clear idiom),
    so the flags are those of 0-0 (AZ/AC set, AN/AV clear) regardless of
    what value that operand held, even an Unknown one."""

    def update(astatx: Value) -> Value:
        if same_source and subtract:
            return _astatx_define(
                astatx, ALU_FLAGS_MASK, _arith_flag_bits(Const(0), Const(0), True)
            )
        if isinstance(a, Const) and isinstance(b, Const):
            return _astatx_define(astatx, ALU_FLAGS_MASK, _arith_flag_bits(a, b, subtract))
        return _astatx_forget(astatx, ALU_FLAGS_MASK)

    return update


def _astatx_abs(source: Value) -> "Callable[[Value], Value]":
    """abs (PGR p.11-13/11-14): AC/AV/AN/AZ come from the same adder the
    value itself is computed with -- 0-RX (subtract) when RX is negative,
    0+RX (add, i.e. an ordinary passthrough) when it is not -- so AN/AZ
    always agree with the actual returned value, and AC/AV are trivially 0
    on the positive branch (adding 0 cannot carry or overflow) but can be
    set on the negative branch (ABS(INT_MIN) overflows, matching the PGR
    text, exactly like negate(INT_MIN)). AS is set from RX's own sign
    (unlike negate, whose AS is always cleared); AI cleared."""

    def update(astatx: Value) -> Value:
        if isinstance(source, Const):
            negative = bool(source.value & 0x80000000)
            bits = _arith_flag_bits(Const(0), source, negative)
            updates: Dict[int, Optional[bool]] = {
                AZ_BIT: bool(bits & (1 << AZ_BIT)),
                AV_BIT: bool(bits & (1 << AV_BIT)),
                AN_BIT: bool(bits & (1 << AN_BIT)),
                AC_BIT: bool(bits & (1 << AC_BIT)),
                AS_BIT: negative,
                AI_BIT: False,
                AF_BIT: False,  # every fixed-point ALU op clears AF (PRM p.439)
            }
            return _astatx_apply_bits(astatx, updates)
        return _astatx_forget(astatx, ALU_FLAGS_MASK)

    return update


def _arith_flag_bits_ci(a: Const, b: Const, subtract: bool, carry_in: bool) -> int:
    """AC/AV/AN/AZ for RN = RX+RY+ci / RN = RX-RY+ci-1 (PRM p.438-439; PGR
    Table 12-3 opcodes 0000 0101/0000 0110, p.573); AS/AI are always 0,
    matching plain add/subtract (PRM: "AS Cleared", "AI Cleared" for both).

    Same two's-complement adder model as ``_arith_flag_bits``, with the
    ASTATX AC bit supplied as an explicit carry-in. The identity RX - RY +
    ci - 1 = RX + ~RY + ci (two's complement: ~RY = -RY-1) means the
    subtract-with-borrow row needs no forced +1 of its own -- CI itself
    supplies the carry that ordinary subtract hard-codes to 1 -- so this
    reuses the exact same b_eff = ~B one's-complement substitution as
    ``_arith_flag_bits``, just with CI standing in for the fixed carry_in.
    """
    A = a.value & 0xFFFFFFFF
    b_eff = (~b.value if subtract else b.value) & 0xFFFFFFFF
    ci = 1 if carry_in else 0
    low31 = (A & 0x7FFFFFFF) + (b_eff & 0x7FFFFFFF) + ci
    carry_into_msb = (low31 >> 31) & 1
    full = A + b_eff + ci
    carry_out = (full >> 32) & 1
    result = full & 0xFFFFFFFF
    bits = 0
    if carry_out:
        bits |= 1 << AC_BIT
    if carry_into_msb ^ carry_out:
        bits |= 1 << AV_BIT
    if result & 0x80000000:
        bits |= 1 << AN_BIT
    if result == 0:
        bits |= 1 << AZ_BIT
    return bits


def _astatx_alu_arith_ci(
    a: Value, b: Value, subtract: bool, carry_in: Optional[bool]
) -> "Callable[[Value], Value]":
    """add-with-carry/subtract-with-borrow: AC/AV/AN/AZ from A, B and the
    ASTATX AC carry-in; AS/AI/AF cleared, same as ``_astatx_alu_arith``.
    Forgets the flags (rather than defining them) whenever the carry-in
    itself is unknown, not just when A or B is."""

    def update(astatx: Value) -> Value:
        if isinstance(a, Const) and isinstance(b, Const) and carry_in is not None:
            return _astatx_define(
                astatx, ALU_FLAGS_MASK, _arith_flag_bits_ci(a, b, subtract, carry_in)
            )
        return _astatx_forget(astatx, ALU_FLAGS_MASK)

    return update


def _astatx_compare(value: Value) -> "Callable[[Value], Value]":
    """PRM comp/compu: AC/AV/AS/AI/AF clear; AZ/AN from VALUE (bits 0, 2);
    CACC (bits 31:24) is an 8-bit shift register, newest bit (VALUE bit 31)
    entering at bit 31. The shift needs the old CACC bits, so it is only
    computed exactly when the old ASTATX is fully known; otherwise CACC
    becomes unknown while the other newly defined bits do not.
    """

    def update(astatx: Value) -> Value:
        if not isinstance(value, Const):
            return _astatx_forget(_astatx_forget(astatx, ALU_FLAGS_MASK), 0xFF000000)
        new_low = value.value & ((1 << AZ_BIT) | (1 << AN_BIT))
        if isinstance(astatx, Const):
            old = astatx.value
            cacc = (old >> 1) & 0x7F000000
            preserve = 0x00FFFFC0 & ~(1 << AF_BIT)  # bits 6-23 minus AF
            return Const((old & preserve) | cacc | new_low | (value.value & 0x80000000))
        result = _astatx_define(astatx, ALU_FLAGS_MASK, new_low)
        return _astatx_forget(result, 0xFF000000)

    return update


def _astatx_mult_forget(astatx: Value) -> Value:
    """multiply/multiply-add-mrf/saturate-mrf/multiply-accumulate: the
    tracer does not model the multiplier result format, so MN/MV/MU/MI are
    always unknown."""
    return _astatx_forget(astatx, MULT_FLAGS_MASK)


def _astatx_mult_clear(astatx: Value) -> Value:
    """mr-data-move: PRM p.493 documents MU/MN/MI/MV all cleared."""
    return _astatx_define(astatx, MULT_FLAGS_MASK, 0)


def _astatx_mult_fixed(astatx: Value) -> Value:
    """Fixed-point multiply/multiply-mrf/multiply-accumulate rows of PRM
    Table 3-7 (p.3-12): MN/MV/MU are data-dependent on the unmodeled
    multiplier result format, so they stay unknown like
    ``_astatx_mult_forget``; MI is documented 0 on every fixed-point row
    there (it only ever applies to the floating-point row), so it is
    defined rather than forgotten.
    """
    return _astatx_apply_bits(
        astatx, {MN_BIT: None, MV_BIT: None, MU_BIT: None, MI_BIT: False}
    )


def _astatx_mult_sat(astatx: Value) -> Value:
    """sat mrf/mrb MOD2 row of PRM Table 3-7 (p.3-12): MN/MV are
    data-dependent and unmodeled, but that row documents MU and MI as fixed
    0 (unlike the plain multiply/accumulate rows, where only MI is fixed).
    """
    return _astatx_apply_bits(
        astatx, {MN_BIT: None, MV_BIT: None, MU_BIT: False, MI_BIT: False}
    )


def _astatx_bit_field(position: Value | int, result: Value) -> "Callable[[Value], Value]":
    """bset/bclr/btgl reg and immediate (PRM pp.511-513): SS cleared; SZ =
    output == 0; SV = bit position > 31."""
    pos = position.value if isinstance(position, Const) else (
        position if isinstance(position, int) else None
    )

    def update(astatx: Value) -> Value:
        updates: Dict[int, Optional[bool]] = {SS_BIT: False}
        if pos is None:
            updates[SV_BIT] = None
            updates[SZ_BIT] = None
        else:
            updates[SV_BIT] = pos > 31
            updates[SZ_BIT] = (result.value == 0) if isinstance(result, Const) else None
        return _astatx_apply_bits(astatx, updates)

    return update


def _astatx_fext(span: int, result: Value) -> "Callable[[Value], Value]":
    """fext immediate (PRM pp.518-519): SS cleared; SZ = output == 0; SV =
    len6 + bit6 > 32. SPAN is len6+bit6, always known from the immediate."""

    def update(astatx: Value) -> Value:
        updates: Dict[int, Optional[bool]] = {
            SS_BIT: False,
            SV_BIT: span > 32,
            SZ_BIT: (result.value == 0) if isinstance(result, Const) else None,
        }
        return _astatx_apply_bits(astatx, updates)

    return update


def _astatx_leftz(source: Value, result: Value) -> "Callable[[Value], Value]":
    """leftz (PRM p.521): SS cleared; SZ = MSB of RX is 1; SV = result == 32."""

    def update(astatx: Value) -> Value:
        updates: Dict[int, Optional[bool]] = {SS_BIT: False}
        updates[SZ_BIT] = (
            bool(source.value & 0x80000000) if isinstance(source, Const) else None
        )
        updates[SV_BIT] = (result.value == 32) if isinstance(result, Const) else None
        return _astatx_apply_bits(astatx, updates)

    return update


def _astatx_lefto(source: Value, result: Value) -> "Callable[[Value], Value]":
    """lefto (PGR p.11-83): SS cleared; SZ = MSB of RX is 0; SV = result ==
    32. The mirror image of ``_astatx_leftz``'s SZ polarity (leading 1s are
    zero in count exactly when RX starts with a 0 bit)."""

    def update(astatx: Value) -> Value:
        updates: Dict[int, Optional[bool]] = {SS_BIT: False}
        updates[SZ_BIT] = (
            not bool(source.value & 0x80000000) if isinstance(source, Const) else None
        )
        updates[SV_BIT] = (result.value == 32) if isinstance(result, Const) else None
        return _astatx_apply_bits(astatx, updates)

    return update


def _astatx_btst(source: Value, position: Value) -> "Callable[[Value], Value]":
    """btst reg (PRM p.513): SS cleared; SZ set if the tested bit is 0 or the
    position is out of range, cleared if the tested bit is 1; SV = position >
    31. BTF is unaffected."""

    def update(astatx: Value) -> Value:
        updates: Dict[int, Optional[bool]] = {SS_BIT: False}
        if not isinstance(position, Const):
            updates[SV_BIT] = None
            updates[SZ_BIT] = None
        else:
            pos = position.value
            out_of_range = pos > 31
            updates[SV_BIT] = out_of_range
            if out_of_range:
                updates[SZ_BIT] = True
            elif isinstance(source, Const):
                updates[SZ_BIT] = not bool(source.value & (1 << pos))
            else:
                updates[SZ_BIT] = None
        return _astatx_apply_bits(astatx, updates)

    return update


def _astatx_shift(
    amount: Optional[int], shifted: Value, ss_mode: str
) -> "Callable[[Value], Value]":
    """lshift/ashift reg and immediate, OR-lshift/OR-ashift immediate (PRM
    pp.508-510): SZ = the shifted value (before any OR) is zero; SV = the
    shift amount is a left shift (> 0).

    SS is cleared for every one of these forms except OR-ashift, whose PRM
    entry omits an SS line entirely (unlike its OR-lshift sibling, which
    repeats "SS Cleared"); pass ss_mode="forget" there so SS becomes unknown
    instead of guessed, without touching any other already-known bit.
    """

    def update(astatx: Value) -> Value:
        updates: Dict[int, Optional[bool]] = {
            SS_BIT: False if ss_mode == "clear" else None,
            SV_BIT: None if amount is None else amount > 0,
            SZ_BIT: (shifted.value == 0) if isinstance(shifted, Const) else None,
        }
        return _astatx_apply_bits(astatx, updates)

    return update
