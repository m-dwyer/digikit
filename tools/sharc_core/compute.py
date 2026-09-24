"""Compute operations (ALU, multiplier, shifter, multifunction) and their application.

Moved verbatim from tools/sharc_trace.py.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from sharc_disasm import Instruction

from .encoding import (
    AC_BIT,
    AF_BIT,
    AI_BIT,
    ALU_FLAGS_MASK,
    AN_BIT,
    AS_BIT,
    AV_BIT,
    AZ_BIT,
    SF_BIT,
    SHIFT_FLAGS_MASK,
    SS_BIT,
    SV_BIT,
    SZ_BIT,
    UREG_CODES,
    _field,
)
from .flags import (
    _alu_arith_updates,
    _astatx_abs,
    _astatx_alu_arith,
    _astatx_alu_arith_ci,
    _astatx_alu_logical,
    _astatx_apply_bits,
    _astatx_bit_field,
    _astatx_btst,
    _astatx_compare,
    _astatx_compare_float,
    _astatx_fext,
    _astatx_forget,
    _astatx_from_updates,
    _astatx_lefto,
    _astatx_leftz,
    _astatx_mult_clear,
    _astatx_mult_fixed,
    _astatx_mult_forget,
    _astatx_mult_sat,
    _astatx_shift,
    _compare_flags,
    _compare_flags_float,
    _float_alu_updates,
    _or_updates,
)
from .floats import (
    _approx_recips,
    _fixed_to_float,
    _fixed_to_float_scaled,
    _float32,
    _float32_bits,
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
from .state import (
    State,
    _event,
    _json_value,
    _simd_active,
    _ureg,
    _ureg_raw,
)
from .values import (
    Const,
    Unknown,
    Value,
    _add,
    _astatx_known_bit,
    _bitwise,
    _multiply,
    _multiply_fractional,
    _not,
    _signed,
    _signed32,
    _subtract,
)


def _field_deposit_or(
    dest: Value,
    source: Value,
    position: int,
    length: int,
    sign_extend: bool,
    label: str,
) -> Value:
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
    special: Mapping[str, Value] | None = None,
) -> tuple[int | str | tuple, Value, str, Callable[[Value], Value]]:
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
        new_wrp = (
            Const(old_wrp.value - bitlen12)
            if isinstance(old_wrp, Const)
            else Unknown("uninitialized BFFWRP")
        )
        updates[SF_BIT] = new_wrp.value >= 32 if isinstance(new_wrp, Const) else None
        return (
            (rn, "BFFWRP"),
            (value, new_wrp),
            "bit-extract",
            _astatx_from_updates(updates),
        )
    raise ValueError("unsupported ShiftImm opcode %#x" % opcode)


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
    special: Mapping[str, Value] | None,
) -> tuple[int | str, Value, str, Callable[[Value], Value]]:
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


def _compute(
    f: Mapping[str, int],
    short: bool,
    values: Mapping[int, Value],
    special: Mapping[str, Value] | None = None,
    *,
    approx_recips: bool = False,
) -> tuple[int | str, Value, str, Callable[[Value], Value]] | None:
    """Decode the small public-table subset, reading every operand from VALUES.

    The 4th element of a non-None result is an ASTATX updater: a function
    from the old ASTATX Value to the new one, computed here (where the
    operands are in scope) and applied by ``_apply_compute``.
    """
    field = (
        _field(f, "compute")
        if short
        else ((_field(f, "compute[22:16]") << 16) | _field(f, "compute[15:0]"))
    )
    if not short and field == 0:
        return None
    # PRM Table 18-29: fixed bits 22:17=100000 select an MR data move.
    # The target-guided SPORT setup path uses the register-to-MR direction.
    if not short and field >> 17 == 0b100000:
        direction = (field >> 16) & 1
        opcode = (field >> 12) & 0xF
        rn = (field >> 8) & 0xF
        mr_name = MR_DATAMOVE_REGISTERS.get(opcode)
        if mr_name is None:
            raise ValueError("unsupported MR data move %#x" % field)
        return _mr_data_move(mr_name, rn, direction, values, special)
    # PRM multiplier compute table: MRF = MRF + RX * RY (MOD1).  Preserve
    # the accumulator separately from the UREG file so later MR transfers do
    # not masquerade as architectural UREGs.
    if not short and ((field >> 20) & 3) == 1 and ((field >> 12) & 0xFF) == 0xB4:
        rx, ry = (field >> 4) & 0xF, field & 0xF
        accumulator = (special or {}).get("MRF", Unknown("uninitialized MRF"))
        product = _multiply(
            _ureg(values, rx), _ureg(values, ry), "R%d * R%d" % (rx, ry)
        )
        return (
            "MRF",
            _add(accumulator, product, "MRF + R%d * R%d" % (rx, ry)),
            "multiply-accumulate",
            _astatx_mult_forget,
        )
    if not short and ((field >> 20) & 3) == 1 and ((field >> 12) & 0xFF) == 0xB0:
        rn, rx, ry = (field >> 8) & 0xF, (field >> 4) & 0xF, field & 0xF
        accumulator = (special or {}).get("MRF", Unknown("uninitialized MRF"))
        product = _multiply(
            _ureg(values, rx), _ureg(values, ry), "R%d * R%d" % (rx, ry)
        )
        return (
            rn,
            _add(accumulator, product, "MRF + R%d * R%d" % (rx, ry)),
            "multiply-add-mrf",
            _astatx_mult_forget,
        )
    if short:
        opcode, rn, rx = (field >> 8) & 0xF, (field >> 4) & 0xF, field & 0xF
        left, right = _ureg(values, rn), _ureg(values, rx)
        # Each entry is (name, calculate, astatx_kind): astatx_kind is None
        # for the value-only logical rule (pass/not/and/or/xor), an
        # (a, b, subtract) triple for the arithmetic-flags rule, or "mult"
        # to forget the (unmodelled) multiplier flags.
        operations = {
            0: (
                "add",
                lambda: _add(left, right, "R%d + R%d" % (rn, rx)),
                (left, right, False),
            ),
            1: (
                "subtract",
                lambda: _subtract(
                    left, right, "R%d - R%d" % (rn, rx), same_source=rn == rx
                ),
                (left, right, True, rn == rx),
            ),
            2: ("pass", lambda: right, None),
            4: (
                "not",
                lambda: _not(right, "not R%d" % rx),
                None,
            ),
            5: (
                "increment",
                lambda: _add(right, Const(1), "R%d + 1" % rx),
                (right, Const(1), False),
            ),
            6: (
                "decrement",
                lambda: _add(right, Const(-1), "R%d - 1" % rx),
                (right, Const(1), True),
            ),
            7: (
                "multiply",
                lambda: _multiply(left, right, "R%d * R%d" % (rn, rx)),
                "mult",
            ),
            0xC: (
                "and",
                lambda: _bitwise(
                    left, right, "R%d and R%d" % (rn, rx), lambda a, b: a & b
                ),
                None,
            ),
            0xD: (
                "or",
                lambda: _bitwise(
                    left, right, "R%d or R%d" % (rn, rx), lambda a, b: a | b
                ),
                None,
            ),
            0xE: (
                "xor",
                lambda: _bitwise(
                    left, right, "R%d xor R%d" % (rn, rx), lambda a, b: a ^ b
                ),
                None,
            ),
        }
        if opcode == 3:
            # PRM ShortCompute table (p. 17-3): 0011 is the signed comp(RN, RX).
            value = _compare_flags(left, right, True, "comp R%d, R%d" % (rn, rx))
            return rn, value, "compare", _astatx_compare(value)
        # PRM Table 18-2 (p.423-425)/PGR "Short Compute Opcodes"
        # (pgr.txt:23108-23120): 1000-1011 and 1111 are the float
        # ShortCompute rows -- the same ops as the full-compute float table
        # above, just the compact 16-bit Type 2c encoding where RN doubles
        # as both the Y input and the result (Table 18-22: "RN = RN op RX").
        if opcode == 0x8:
            value, overflow, invalid = _float_binary(
                left, right, "F%d + F%d" % (rn, rx), lambda a, b: a + b
            )
            return (
                rn,
                value,
                "float-add",
                _astatx_from_updates(
                    _float_alu_updates(value, av=overflow, ai=invalid)
                ),
            )
        if opcode == 0x9:
            value, overflow, invalid = _float_binary(
                left, right, "F%d - F%d" % (rn, rx), lambda a, b: a - b
            )
            return (
                rn,
                value,
                "float-subtract",
                _astatx_from_updates(
                    _float_alu_updates(value, av=overflow, ai=invalid)
                ),
            )
        if opcode == 0xA:
            # FN = float RX: unlike the other short float rows, RN is not
            # read as an input here (only RX is converted); RN is purely the
            # destination.
            value, invalid = _fixed_to_float(right, "float R%d" % rx)
            return (
                rn,
                value,
                "float-convert",
                _astatx_from_updates(_float_alu_updates(value, av=False, ai=invalid)),
            )
        if opcode == 0xB:
            label = "comp F%d, F%d" % (rn, rx)
            value, invalid = _compare_flags_float(left, right, label)
            return rn, value, "float-compare", _astatx_compare_float(value, invalid)
        if opcode == 0xF:
            a, b = _float32(left), _float32(right)
            if a is None or b is None:
                value = Unknown("F%d * F%d" % (rn, rx))
            else:
                bits, _overflowed = _float32_bits(a * b)
                value = Const(bits)
            return rn, value, "float-multiply", _astatx_mult_forget
        if opcode not in operations:
            raise ValueError("unsupported short compute opcode %#x" % opcode)
        operation, calculate, astatx_kind = operations[opcode]
        value = calculate()
        if astatx_kind is None:
            astatx_update = _astatx_alu_logical(value)
        elif astatx_kind == "mult":
            astatx_update = _astatx_mult_forget
        else:
            a, b, subtract, *rest = astatx_kind
            astatx_update = _astatx_alu_arith(
                a, b, subtract, same_source=rest[0] if rest else False
            )
        return rn, value, operation, astatx_update
    # PRM Table 18-1/Figure 18-1 (p.423): bit22 is MF, the multifunction
    # selector. A multifunction op's register sub-fields (PRM Table
    # 18-15..18-19, p.434-435) do not line up with the SINGLEFN rn/rx/ry
    # layout computed below, so it is decoded separately and always
    # returns/raises before falling through to that layout.
    mf = (field >> 22) & 1
    if mf:
        category = (field >> 16) & 0x3F
        rm, ra = (field >> 12) & 0xF, (field >> 8) & 0xF
        # PRM Table 18-16/18-17 (p.434): the four multifunction INPUT
        # operands are 2-bit fields, each selecting within a fixed quad --
        # Fxm in F0-3, Fym in F4-7, Fxa in F8-11, Fya in F12-15.
        rxm_reg = (field >> 6) & 0x3
        rym_reg = 4 + ((field >> 4) & 0x3)
        rxa_reg = 8 + ((field >> 2) & 0x3)
        rya_reg = 12 + (field & 0x3)
        fxm, fym = _ureg(values, rxm_reg), _ureg(values, rym_reg)
        fxa, fya = _ureg(values, rxa_reg), _ureg(values, rya_reg)
        # PGR Table 12-12 (pgr.txt:23129-23198), opcode[21:16] 011000/011001:
        # FM = FXM*FYM, FA = FXA+-FYA -- the only MUL/ALU multifunction rows
        # this firmware's audio code uses. Flags follow the single-function
        # rule for each half (PRM p.3-21/3-22: multifunction "handle[s]
        # flags in the same way as the single function computations" except
        # for dual add/subtract), so the ALU half reuses
        # ``_float_alu_updates`` and the multiplier half stays forgotten via
        # ``_astatx_mult_forget`` exactly as the plain float multiply below.
        if category in (0x18, 0x19, 0x1A, 0x1C, 0x1D, 0x1E, 0x1F):
            # The multiplier half uses ordinary IEEE NaN propagation like
            # the plain float multiply below, not the ALU's NaN-input
            # all-1s quirk, so it is computed directly rather than through
            # ``_float_binary``. Shared by every multifunction category
            # below: only the ALU half's operation differs.
            fm_a, fm_b = _float32(fxm), _float32(fym)
            if fm_a is None or fm_b is None:
                fm_value: Value = Unknown("F%d * F%d" % (rxm_reg, rym_reg))
            else:
                fm_bits, _fm_overflowed = _float32_bits(fm_a * fm_b)
                fm_value = Const(fm_bits)

            def with_mult(fa_updates, operation):
                def astatx_update(astatx: Value, updates=fa_updates) -> Value:
                    return _astatx_mult_forget(_astatx_apply_bits(astatx, updates))

                return (rm, ra), (fm_value, fa_value), operation, astatx_update

            if category in (0x18, 0x19):
                subtract = category == 0x19
                fa_op = (lambda a, b: a - b) if subtract else (lambda a, b: a + b)
                fa_value, fa_overflow, fa_invalid = _float_binary(
                    fxa,
                    fya,
                    "F%d %s F%d" % (rxa_reg, "-" if subtract else "+", rya_reg),
                    fa_op,
                )
                return with_mult(
                    _float_alu_updates(fa_value, av=fa_overflow, ai=fa_invalid),
                    "float-mulalu-subtract" if subtract else "float-mulalu-add",
                )
            # PGR Table 12-12 opcode 011010 (p.587): FM = FXM*FYM,
            # FA = FLOAT RXA by RYA -- the ALU half is the scaled
            # fixed->float convert (0xDA above), reading RXA/RYA as fixed
            # rather than float registers (same physical register file
            # slots, ``fxa``/``fya`` above are just the raw bit patterns).
            if category == 0x1A:
                fa_value, fa_overflow = _fixed_to_float_scaled(
                    fxa, fya, "float R%d by R%d" % (rxa_reg, rya_reg)
                )
                return with_mult(
                    _float_alu_updates(fa_value, av=fa_overflow, ai=False),
                    "float-mulalu-convert",
                )
            # PGR Table 12-12 opcodes 011110/011111 (p.588): FM = FXM*FYM,
            # FA = MAX/MIN(FXA, FYA) -- the ALU half is the same
            # single-function float min/max as 0xE1/0xE2 above (AV fixed 0
            # there too).
            if category in (0x1E, 0x1F):
                minimum = category == 0x1F
                name = "min" if minimum else "max"
                combine = _float_min if minimum else _float_max
                fa_value, fa_overflow, fa_invalid = _float_binary(
                    fxa, fya, "%s(F%d, F%d)" % (name, rxa_reg, rya_reg), combine
                )
                return with_mult(
                    _float_alu_updates(fa_value, av=False, ai=fa_invalid),
                    "float-mul" + name,
                )
            # PGR Table 12-12 opcode 011100 (p.588): FM = FXM*FYM,
            # FA = (FXA + FYA)/2 -- the ALU half is the single-function
            # float average (PGR p.11-28: AV architecturally fixed 0, never
            # data-dependent, unlike plain float add).
            if category == 0x1C:
                fa_value, _fa_overflow, fa_invalid = _float_binary(
                    fxa,
                    fya,
                    "(F%d + F%d)/2" % (rxa_reg, rya_reg),
                    lambda a, b: (a + b) / 2,
                )
                return with_mult(
                    _float_alu_updates(fa_value, av=False, ai=fa_invalid),
                    "float-mulalu-average",
                )
            # PGR Table 12-12 opcode 011101 (p.588): FM = FXM*FYM,
            # FA = ABS FXA -- the ALU half is the same single-function float
            # abs as 0xB0 above (AN fixed 0, AS carries FXA's own sign).
            if category == 0x1D:
                fa_value, _fa_overflow, fa_invalid = _float_unary(
                    fxa, "abs F%d" % rxa_reg, abs
                )
                return with_mult(
                    _float_alu_updates(
                        fa_value, av=False, an_zero=True, as_source=fxa, ai=fa_invalid
                    ),
                    "float-mulalu-abs",
                )
        # PRM ch.24 p.528-529 "Floating-Point Multiplier and ALU (dual Add
        # and Subtract)": Fm=F3-0*F7-4, Fa=F11-8+F15-12, Fs=F11-8-F15-12 --
        # the multifunction twin of the single-function dual add/subtract
        # above, for FFT butterflies. Neither PRM nor PGR prints this row's
        # opcode bits (compute_table.json's multifn_mul_dual_addsub note),
        # but tools/sharcdb.py's independently-built register-def/use table
        # (_compute_regdef_reguse's is_dual_addsub branch) already extracts
        # RS from bits 19:16 -- i.e. category's own low 4 bits, with only
        # category's top 2 bits (here, "11") acting as the fixed selector --
        # confirmed against this decoder's own database. FS shares FXA/FYA
        # with FA (same operand pair, opposite sign), matching the
        # single-function dual add/subtract's OR'd-flags convention.
        if (category >> 4) == 0b11:
            rs = category & 0xF
            fm_a, fm_b = _float32(fxm), _float32(fym)
            if fm_a is None or fm_b is None:
                fm_value = Unknown("F%d * F%d" % (rxm_reg, rym_reg))
            else:
                fm_bits, _fm_overflowed = _float32_bits(fm_a * fm_b)
                fm_value = Const(fm_bits)
            add_value, add_overflow, add_invalid = _float_binary(
                fxa, fya, "F%d + F%d" % (rxa_reg, rya_reg), lambda a, b: a + b
            )
            sub_value, sub_overflow, sub_invalid = _float_binary(
                fxa, fya, "F%d - F%d" % (rxa_reg, rya_reg), lambda a, b: a - b
            )
            updates = _or_updates(
                _float_alu_updates(add_value, av=add_overflow, ai=add_invalid),
                _float_alu_updates(sub_value, av=sub_overflow, ai=sub_invalid),
            )

            def dual_astatx_update(astatx: Value, u=updates) -> Value:
                return _astatx_mult_forget(_astatx_apply_bits(astatx, u))

            return (
                (rm, ra, rs),
                (fm_value, add_value, sub_value),
                "float-mul-dual-add-subtract",
                dual_astatx_update,
            )
        raise ValueError(
            "unsupported multifunction category=%#04x rm=%d ra=%d" % (category, rm, ra)
        )
    cu, opcode = (field >> 20) & 3, (field >> 12) & 0xFF
    rn, rx, ry = (field >> 8) & 0xF, (field >> 4) & 0xF, field & 0xF
    left, right = _ureg(values, rx), _ureg(values, ry)
    # PRM Table 18-10 (p.433) / Table 18-13 (p.434): Dual Add/Subtract is a
    # single-function ALU op (mf=0, not multifunction) whose opcode top
    # nibble (bits 19:16, i.e. this OPCODE's top nibble) is 0111 (fixed) or
    # 1111 (float); the low nibble (bits 15:12) is not part of the opcode at
    # all -- it is RS, a second 4-bit result register alongside
    # RA=RN/FN at bits 11:8 (already read above as RN).
    if cu == 0 and (opcode >> 4) in (0x7, 0xF):
        float_form = (opcode >> 4) == 0xF
        rs = opcode & 0xF
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
            sub_value = _subtract(
                left, right, "R%d - R%d" % (rx, ry), same_source=rx == ry
            )
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
    if cu == 0 and opcode == 0x01:
        value = _add(left, right, "R%d + R%d" % (rx, ry))
        return rn, value, "add", _astatx_alu_arith(left, right, False)
    if cu == 0 and opcode == 0x02:
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
    if cu == 0 and opcode in (0x05, 0x06):
        subtract = opcode == 0x06
        astatx = _ureg_raw(values, UREG_CODES["ASTATX"])
        carry_in = _astatx_known_bit(astatx, AC_BIT)
        label = "R%d %s R%d + ci%s" % (
            rx,
            "-" if subtract else "+",
            ry,
            " - 1" if subtract else "",
        )
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
    # PGR p.11-9/11-10 (pgr.txt:20570/20606), opcodes 0010 0101/0010 0110:
    # RN = RX + ci / RN = RX + ci - 1 -- the single-operand twins of
    # 0x05/0x06 above (no RY; PGR's flag table is identical to the RY form),
    # so they reuse the same value/flags formulas with RY forced to 0.
    if cu == 0 and opcode in (0x25, 0x26):
        subtract = opcode == 0x26
        astatx = _ureg_raw(values, UREG_CODES["ASTATX"])
        carry_in = _astatx_known_bit(astatx, AC_BIT)
        label = "R%d + ci%s" % (rx, " - 1" if subtract else "")
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
    # PRM Table 18-5: ALUOP 00001010 is signed comp(RX, RY) and 00001011 is
    # unsigned compu(RX, RY). Both update status only, so the tracer records
    # the comparison without writing RN; the value carries the new flags.
    if cu == 0 and opcode in (0x0A, 0x0B):
        signed = opcode == 0x0A
        label = "%s R%d, R%d" % ("comp" if signed else "compu", rx, ry)
        value = _compare_flags(left, right, signed, label)
        return rn, value, "compare", _astatx_compare(value)
    if cu == 0 and opcode == 0x21:
        return rn, left, "pass", _astatx_alu_logical(left)
    # PRM Table 18-5 and p. 19-10: ALUOP 00100010 is RN = -RX, the two's
    # complement, with the same flags as 0 - RX.
    if cu == 0 and opcode == 0x22:
        value = _subtract(Const(0), left, "-R%d" % rx)
        return rn, value, "negate", _astatx_alu_arith(Const(0), left, True)
    if cu == 0 and opcode == 0x29:
        value = _add(left, Const(1), "R%d + 1" % rx)
        return rn, value, "increment", _astatx_alu_arith(left, Const(1), False)
    # PRM Table 18-5 and p. 19-9: ALUOP 00101010 is RN = RX - 1.
    if cu == 0 and opcode == 0x2A:
        value = _add(left, Const(-1), "R%d - 1" % rx)
        return rn, value, "decrement", _astatx_alu_arith(left, Const(1), True)
    # PGR p.11-13/11-14 (pgr.txt:20750), opcode 0011 0000: RN = ABS RX. Value
    # and AC/AV/AN/AZ come from the same 0-RX adder as negate (0x22 above,
    # "The ABS of the minimum negative number ... causes an overflow" only
    # makes sense if ABS always runs the 0-RX path, even for a positive RX);
    # unlike negate, AS is data-dependent here (RX's own sign) rather than
    # architecturally cleared.
    if cu == 0 and opcode == 0x30:
        if isinstance(left, Const):
            signed = _signed32(left.value)
            value = left if signed >= 0 else _subtract(Const(0), left, "abs R%d" % rx)
        else:
            value = Unknown("abs R%d" % rx)
        return rn, value, "abs", _astatx_abs(left)
    # PRM Table 18-5: ALUOP 01000000..01000010 are the integer logical
    # operations AND, OR, and XOR.
    if cu == 0 and opcode in (0x40, 0x41, 0x42):
        name, operation = {
            0x40: ("and", lambda a, b: a & b),
            0x41: ("or", lambda a, b: a | b),
            0x42: ("xor", lambda a, b: a ^ b),
        }[opcode]
        value = _bitwise(
            left,
            right,
            "R%d %s R%d" % (rx, name, ry),
            operation,
        )
        return rn, value, name, _astatx_alu_logical(value)
    # PGR p.11-19 (pgr.txt:20918), opcode 0100 0011: RN = NOT RX. Same
    # AZ/AN-from-result, AC/AV/AS/AI-cleared rule as pass/and/or/xor.
    if cu == 0 and opcode == 0x43:
        value = _not(left, "not R%d" % rx)
        return rn, value, "not", _astatx_alu_logical(value)
    # PRM Table 18-5 (p.425-427), float rows; per-op flags cited at each
    # branch (PRM Table 3-3, pp.3-8/3-9, cross-checked against the classic
    # PGR's per-instruction pages, which spell out AZ/AN/AV/AI exactly where
    # the SHARC+ PRM only marks a column "*"/data-dependent).
    if cu == 0 and opcode == 0x81:
        value, overflow, invalid = _float_binary(
            left, right, "F%d + F%d" % (rx, ry), lambda a, b: a + b
        )
        return (
            rn,
            value,
            "float-add",
            _astatx_from_updates(_float_alu_updates(value, av=overflow, ai=invalid)),
        )
    if cu == 0 and opcode == 0x82:
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
    if cu == 0 and opcode == 0x92:
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
    if cu == 0 and opcode == 0x8A:
        label = "comp F%d, F%d" % (rx, ry)
        value, invalid = _compare_flags_float(left, right, label)
        return rn, value, "float-compare", _astatx_compare_float(value, invalid)
    # PGR p.11-32: Fn = pass Fx.
    if cu == 0 and opcode == 0xA1:
        value, overflow, invalid = _float_unary(left, "pass F%d" % rx, lambda a: a)
        return (
            rn,
            value,
            "float-pass",
            _astatx_from_updates(_float_alu_updates(value, av=False, ai=invalid)),
        )
    # PGR p.11-30: Fn = -Fx.
    if cu == 0 and opcode == 0xA2:
        value, overflow, invalid = _float_unary(left, "-F%d" % rx, lambda a: -a)
        return (
            rn,
            value,
            "float-negate",
            _astatx_from_updates(_float_alu_updates(value, av=False, ai=invalid)),
        )
    # PRM Table 18-5 opcode 1010 0101 (p.20-8) / PGR Table 12-4 opcode
    # 1010 0101, p.11-33: Fn = rnd Fx.
    if cu == 0 and opcode == 0xA5:
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
    if cu == 0 and opcode == 0xAD:
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
    if cu == 0 and opcode == 0xC1:
        mode1 = _ureg_raw(values, UREG_CODES["MODE1"])
        value, overflow, invalid = _float_logb(left, mode1, "logb F%d" % rx)
        updates = {
            AC_BIT: False,
            AF_BIT: True,
            AS_BIT: False,
            AV_BIT: overflow,
            AI_BIT: invalid,
            AZ_BIT: (value.value == 0) if isinstance(value, Const) else None,
            AN_BIT: bool(value.value & 0x80000000)
            if isinstance(value, Const)
            else None,
        }
        return rn, value, "logb", _astatx_from_updates(updates)
    # PGR p.11-31: Fn = abs Fx. AN fixed 0; AS carries the *input*'s sign.
    if cu == 0 and opcode == 0xB0:
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
    if cu == 0 and opcode == 0xBD:
        value, overflow, invalid = _float_scalb(
            left, right, "scalb F%d by R%d" % (rx, ry)
        )
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
    if cu == 0 and opcode in (0x61, 0x62):
        name = "min" if opcode == 0x61 else "max"
        if isinstance(left, Const) and isinstance(right, Const):
            a, b = _signed32(left.value), _signed32(right.value)
            pick_left = (a <= b) if name == "min" else (a >= b)
            value = left if pick_left else right
        else:
            value = Unknown("%s(R%d, R%d)" % (name, rx, ry))
        return rn, value, name, _astatx_alu_logical(value)
    # PRM p.19-19, opcode 1110 0000 / PGR Table 12-4 p.574, same opcode: Fn =
    # Fx copysign Fy.
    if cu == 0 and opcode == 0xE0:
        value, invalid = _float_copysign(left, right, "F%d copysign F%d" % (rx, ry))
        return (
            rn,
            value,
            "float-copysign",
            _astatx_from_updates(_float_alu_updates(value, av=False, ai=invalid)),
        )
    # PGR p.11-46/11-47: Fn = min/max(Fx, Fy).
    if cu == 0 and opcode in (0xE1, 0xE2):
        name = "min" if opcode == 0xE1 else "max"
        combine = _float_min if opcode == 0xE1 else _float_max
        value, overflow, invalid = _float_binary(
            left, right, "%s(F%d, F%d)" % (name, rx, ry), combine
        )
        return (
            rn,
            value,
            "float-" + name,
            _astatx_from_updates(_float_alu_updates(value, av=False, ai=invalid)),
        )
    # PGR p.11-48 / PRM p.3-6: Fn = clip Fx by Fy.
    if cu == 0 and opcode == 0xE3:
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
    if cu == 0 and opcode == 0xCA:
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
    if cu == 0 and opcode == 0xDA:
        value, overflow = _fixed_to_float_scaled(
            left, right, "float R%d by R%d" % (rx, ry)
        )
        return (
            rn,
            value,
            "float-convert-scaled",
            _astatx_from_updates(_float_alu_updates(value, av=overflow, ai=False)),
        )
    # PRM p.24-.. / PGR p.11-36..11-38, opcode 1100 1001: Rn = fix Fx
    # (rounds to nearest or truncates per MODE1.TRUNCATE; see
    # ``_float_to_fixed``).
    if cu == 0 and opcode == 0xC9:
        mode1 = _ureg_raw(values, UREG_CODES["MODE1"])
        value, overflow, invalid = _float_to_fixed(left, mode1, False, "fix F%d" % rx)
        return (
            rn,
            value,
            "fix",
            _astatx_from_updates(_float_alu_updates(value, av=overflow, ai=invalid)),
        )
    # PRM p.427/PGR p.11-37: Rn = trunc Fx.
    if cu == 0 and opcode == 0xCD:
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
    if cu == 0 and opcode == 0xD9:
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
    if cu == 0 and opcode == 0xDD:
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
    if cu == 0 and opcode in (0xC4, 0xC5):
        name = "recips" if opcode == 0xC4 else "rsqrts"
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
    # PRM Table 17-7: MULOP 0000 F00x writes a saturated MRF value to RN.
    # The tracer does not model the full-width multiplier accumulator or MOD2
    # format bits, so preserve the documented data dependency conservatively.
    if cu == 1 and opcode == 0x00:
        return (
            rn,
            Unknown("saturated MRF (unmodeled MOD2)"),
            "saturate-mrf",
            _astatx_mult_forget,
        )
    if cu == 1 and opcode == 0x70:
        value = _multiply(left, right, "R%d * R%d" % (rx, ry))
        return rn, value, "multiply", _astatx_mult_forget
    # PRM Table 18-7 (p.428-429): MULOP 00110000 is Fn = Fx * Fy. Flags are
    # the multiplier's MN/MV/MU/MI (PGR p.11-57), the same unmodeled-result
    # group the fixed-point multiply above forgets via
    # ``_astatx_mult_forget``; unlike the ALU's NaN-input quirk, the PGR
    # text for this op does not document an all-1s override, so ordinary
    # IEEE NaN propagation applies.
    if cu == 1 and opcode == 0x30:
        a, b = _float32(left), _float32(right)
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
    if cu == 1 and opcode == 0x40:
        value = _multiply(left, right, "R%d * R%d" % (rx, ry))
        return rn, value, "multiply", _astatx_mult_fixed
    # Same row, MOD1 UUF sub-option (fractional, unsigned*unsigned, no
    # round): PRM p.3-9/27-3 -- the register result is the top 32 bits of
    # the 64-bit unsigned product (no redundant-sign shift; that only
    # applies when both inputs are signed).
    if cu == 1 and opcode == 0x48:
        value = _multiply_fractional(left, right, False, False, "R%d * R%d" % (rx, ry))
        return rn, value, "multiply", _astatx_mult_fixed
    # PRM Table 17-7, "mrf = RX*RY MOD1" row (no accumulate), MOD1 SSI
    # sub-option: the plain-load twin of opcode 0xB4's accumulate above,
    # same raw-multiply-into-MRF data dependency.
    if cu == 1 and opcode == 0x74:
        value = _multiply(left, right, "R%d * R%d" % (rx, ry))
        return "MRF", value, "multiply-mrf", _astatx_mult_fixed
    # Same row, MOD1 SSF sub-option (fractional, signed*signed, no round):
    # PRM p.3-9 -- both inputs signed, so the redundant-sign left shift
    # applies (folded into _multiply_fractional's >>31).
    if cu == 1 and opcode == 0x7C:
        value = _multiply_fractional(left, right, True, True, "R%d * R%d" % (rx, ry))
        return "MRF", value, "multiply-mrf", _astatx_mult_fixed
    # PRM Table 17-7, "mrf = mrf + RX*RY MOD1" row, MOD1 SSF sub-option:
    # the fractional twin of opcode 0xB4 (SSI, integer) above.
    if cu == 1 and opcode == 0xBC:
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
    if cu == 1 and opcode == 0x09:
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
    # below.
    if cu == 1 and opcode == 0x10:
        label = "multiply opcode 0x10 R%d, R%d (undocumented; no public source)" % (
            rx,
            ry,
        )
        return rn, Unknown(label), "multiply-undocumented-10", _astatx_mult_forget
    # Shifter opcode 1011 0000: absent from both public sources' shifter
    # tables (PRM Table 17-9, p.17-10/17-11, and PGR Table 12-11, p.580-581,
    # transcribed in full -- neither lists any 0xA0-0xBF row). Seen at
    # `sw 0x1cd002`. Rather than guess an operation from an undocumented
    # opcode, decode it (so the walk does not desync) and leave both the
    # result and its flags Unknown, per this file's existing rule for gaps
    # the manuals do not cover.
    if cu == 2 and opcode == 0xB0:
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
    if cu == 2 and opcode == 0x14:
        label = "shift opcode 0x14 R%d, R%d (undocumented; no public source)" % (rx, ry)
        return (
            rn,
            Unknown(label),
            "shift-undocumented-14",
            lambda astatx: _astatx_forget(astatx, SHIFT_FLAGS_MASK),
        )
    # PRM Table 17-9: SHIFTOP 00000000 is RN = LSHIFT RX by RY. The signed
    # low byte of RY selects a left (positive) or logical right (negative)
    # shift; magnitudes of 32 or more produce zero.
    if cu == 2 and opcode == 0x00:
        amount: int | None = None
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
    if cu == 2 and opcode == 0x04:
        amount = None
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
    if cu == 2 and opcode == 0x20:
        or_amount: int | None = None
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
    if cu == 2 and opcode == 0x88:
        value = (
            Const(32 if left.value == 0 else 32 - left.value.bit_length())
            if isinstance(left, Const)
            else Unknown("leftz R%d" % rx)
        )
        return rn, value, "leftz", _astatx_leftz(left, value)
    # PGR p.11-83 (pgr.txt:23171): SHIFTOP 10001100 is RN = LEFTO RX --
    # leading 1s, the complement of leftz above (leading 0s of ~RX).
    if cu == 2 and opcode == 0x8C:
        if isinstance(left, Const):
            inverted = (~left.value) & 0xFFFFFFFF
            value = Const(32 if inverted == 0 else 32 - inverted.bit_length())
        else:
            value = Unknown("lefto R%d" % rx)
        return rn, value, "lefto", _astatx_lefto(left, value)
    # PRM Table 18-9: SHIFTOP 11000000/11000100 are variable bit set/clear.
    if cu == 2 and opcode in (0xC0, 0xC4):
        name = "bset" if opcode == 0xC0 else "bclr"
        if not isinstance(right, Const):
            value = Unknown("%s R%d by R%d" % (name, rx, ry))
        elif right.value > 31:
            value = left
        else:
            calculate = (
                (lambda a, b: a | b) if opcode == 0xC0 else (lambda a, b: a & ~b)
            )
            value = _bitwise(
                left,
                Const(1 << right.value),
                "%s R%d by R%d" % (name, rx, ry),
                calculate,
            )
        return (
            rn,
            value,
            "bit-set" if opcode == 0xC0 else "bit-clear",
            _astatx_bit_field(right, value),
        )
    # PRM Table 17-9 and p. 23-5: SHIFTOP 11001000 is
    # RN = btgl RX by RY.  Positions outside the 32-bit field leave RX
    # unchanged.
    if cu == 2 and opcode == 0xC8:
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
    if cu == 2 and opcode == 0xCC:
        return rn, left, "bit-test", _astatx_btst(left, right)
    # PGR p.11-88 (pgr.txt:23353), opcode 0111 0000: RN = BFFWRP -- reads
    # the bit-FIFO write pointer this tracer tracks via ShiftImm opcode
    # 0x1f/cu=2 opcode 0x7c below (special-dict key "BFFWRP"; Unknown until
    # one of those has run). SF is documented "Not affected" so it is left
    # out of UPDATES entirely (stays whatever it already was).
    if cu == 2 and opcode == 0x70:
        value = (special or {}).get("BFFWRP", Unknown("uninitialized BFFWRP"))
        updates: dict[int, bool | None] = {SS_BIT: False, SZ_BIT: False, SV_BIT: False}
        return rn, value, "bffwrp-read", _astatx_from_updates(updates)
    # PGR p.11-89 (pgr.txt:23379), opcode 0111 1100: BFFWRP = RN|<data7> --
    # the register-operand twin of ShiftImm opcode 0x1f below. The register
    # form reads the RN-positioned field as its SOURCE, not a destination,
    # per PGR's own text: "Updates write pointer from Rn ... Only 7 least
    # significant bits of Rn are written."
    if cu == 2 and opcode == 0x7C:
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
    # PRM top-level compute selector (tools/sharcspec/compute_table.json's
    # top_level_structure.single_function_selector): cu=11 is "reserved
    # (not used by SINGLEFN)" -- no compute unit is documented for it, in
    # either public source. Only the one opcode this image's coverage scan
    # actually found (0xd6, `sw 0x1c32ba`, sitting between two otherwise-
    # ordinary instructions -- not an obvious data-as-code region) is
    # decoded, like the cu=2 0xb0/0x14 undocumented cases above, rather than
    # guessed at; every other cu=3 opcode still raises, since there is no
    # evidence it is real or what it would mean.
    if cu == 3 and opcode == 0xD6:
        label = (
            "reserved compute unit cu=3 opcode=%#x R%d, R%d (PRM: cu=11 not used by SINGLEFN)"
            % (
                opcode,
                rx,
                ry,
            )
        )
        return (
            rn,
            Unknown(label),
            "compute-reserved-cu3",
            lambda astatx: _astatx_forget(astatx, ALU_FLAGS_MASK | SHIFT_FLAGS_MASK),
        )
    raise ValueError("unsupported full compute cu=%#x opcode=%#x" % (cu, opcode))


def _apply_compute(
    state: State,
    insn: Instruction,
    result: tuple[int | str, Value, str, Callable[[Value], Value]],
) -> None:
    rn, value, operation, astatx_update = result
    astatx_code = UREG_CODES["ASTATX"]
    state.uregs[astatx_code] = astatx_update(_ureg_raw(state.uregs, astatx_code))
    if isinstance(rn, str):
        _event(
            state,
            insn,
            "compute",
            operation=operation,
            result_register=rn,
            value=value,
        )
        # "MR0F" (PRM Table 18-29 MRDATAMOVE) is the same physical register
        # the multiply-accumulate rows call "MRF" (PRM p.3-10); every other
        # string key (other MR registers, "BFFWRP") is its own special-dict
        # slot, keyed by the name the caller returned.
        state.special["MRF" if rn == "MR0F" else rn] = value
        return
    if isinstance(rn, tuple):
        # Dual/triple-result compute (dual add/subtract, MUL/ALU
        # multifunction, MUL dual add/subtract, BITEXT's paired RN+BFFWRP
        # write): destinations sharing one ASTATX update, already combined
        # by the caller (PRM p.3-21/3-22). A string destination is a
        # special-dict slot (as in the single-string case above); an int is
        # an ordinary register.
        names = [reg if isinstance(reg, str) else "R%d" % reg for reg in rn]
        _event(
            state,
            insn,
            "compute",
            operation=operation,
            result_register=names,
            value=[_json_value(v) for v in value],
        )
        for reg, val in zip(rn, value, strict=True):
            if isinstance(reg, str):
                state.special["MRF" if reg == "MR0F" else reg] = val
            else:
                state.uregs[reg] = val
        return
    if operation in ("compare", "bit-test", "float-compare"):
        _event(state, insn, "compute", operation=operation, status_only=True)
    else:
        _event(
            state,
            insn,
            "compute",
            operation=operation,
            result_register="R%d" % rn,
            value=value,
        )
        state.uregs[rn] = value
    if operation == "float-recips-seed-approx":
        # --approx-recips produced a numeric value with no ROM table behind
        # it (see _approx_recips); mark this path and this instant so any
        # report can find and discount it, matching how "predicate-assumption"
        # tags a forked branch guess.
        state.approx_recips_used = True
        _event(state, insn, "approximate-recips", value=value)


def _compute_pey_values(values: Mapping[int, Value]) -> dict[int, Value]:
    """A PEy view of the register file for _compute: R/F codes 0-15 read
    the paired S/SF register instead (SHARC+ PRM p.3-39, "Compute
    Instructions in SIMD Mode": "S0 = S1 + S2; /* implicit ALU instruction
    */" -- the PEy compute is decoded from the *same* instruction bits as
    PEx, just re-targeted at the S file, so re-running _compute unchanged
    against a shifted register map is exactly this rule)."""
    shifted = dict(values)
    for code in range(16):
        shifted[code] = values.get(80 + code, Unknown("uninitialized S%d" % code))
    return shifted


def _compute_pey(
    f: Mapping[str, int],
    short: bool,
    values: Mapping[int, Value],
    special: Mapping[str, Value] | None = None,
    *,
    approx_recips: bool = False,
) -> tuple[int | str, Value, str, Callable[[Value], Value]] | None:
    """PEy's half of a SIMD compute (SHARC+ PRM p.101, "SIMD Mode":
    "Executes the same instruction simultaneously in both processing
    elements"), decoded against the S/SF register file and the PEy
    multiplier accumulator (PRM p.101, "Multiplier Result Register Swap":
    "swapping also occurs with the PEY unit based registers (REGF_MS0F,
    REGF_MS2F, and REGF_MS0B, REGF_MS2B)"). _compute always names its MR
    data-move/multiply-accumulate destination "MRF" regardless of PE (see
    its own docstring); disguise state.special["MSF"] as "MRF" on the way
    in so that accumulator read is correct, and _apply_compute_pey below
    undoes the disguise on the way out.
    """
    pey_special = {"MRF": (special or {}).get("MSF", Unknown("uninitialized MSF"))}
    return _compute(
        f,
        short,
        _compute_pey_values(values),
        pey_special,
        approx_recips=approx_recips,
    )


def _apply_compute_pey(
    state: State,
    insn: Instruction,
    result: tuple[int | str, Value, str, Callable[[Value], Value]],
) -> None:
    """PEy's half of _apply_compute: the identical shape, redirected to the
    S/SF register file, REGF_ASTATY, and the PEy multiplier accumulator
    (SHARC+ PRM p.66, Table 3-1: ASTATx/ASTATy and STKYx/STKYy are the
    per-PE computation-status register pairs)."""
    rn, value, operation, astatx_update = result
    astaty_code = UREG_CODES["ASTATY"]
    state.uregs[astaty_code] = astatx_update(_ureg_raw(state.uregs, astaty_code))
    if isinstance(rn, str):
        _event(
            state,
            insn,
            "compute-pey",
            operation=operation,
            result_register="MSF",
            value=value,
        )
        state.special["MSF"] = value
        return
    if isinstance(rn, tuple):
        names = ["S%d" % reg for reg in rn]
        _event(
            state,
            insn,
            "compute-pey",
            operation=operation,
            result_register=names,
            value=[_json_value(v) for v in value],
        )
        for reg, val in zip(rn, value, strict=True):
            state.uregs[80 + reg] = val
        return
    if operation in ("compare", "bit-test", "float-compare"):
        _event(state, insn, "compute-pey", operation=operation, status_only=True)
    else:
        _event(
            state,
            insn,
            "compute-pey",
            operation=operation,
            result_register="S%d" % rn,
            value=value,
        )
        state.uregs[80 + rn] = value
    if operation == "float-recips-seed-approx":
        state.approx_recips_used = True
        _event(state, insn, "approximate-recips-pey", value=value)


def _compute_simd(
    state: State,
    f: Mapping[str, int],
    short: bool,
    values: Mapping[int, Value],
    special: Mapping[str, Value] | None = None,
    *,
    approx_recips: bool = False,
) -> tuple[
    tuple[int | str, Value, str, Callable[[Value], Value]] | None,
    tuple[int | str, Value, str, Callable[[Value], Value]] | None,
]:
    """Decode a compute for PEx, and for PEy too when MODE1.PEYEN is
    concretely set (SHARC+ PRM p.101, "SIMD Mode": "Dispatches a single
    instruction to both processing element's computational units").  SISD
    mode, or an unresolved MODE1, returns a PEy result of None -- exactly
    like SISD, the tracer does not invent a PEy effect it cannot confirm.
    """
    result_x = _compute(f, short, values, special, approx_recips=approx_recips)
    if _simd_active(state) is not True:
        return result_x, None
    result_y = _compute_pey(f, short, values, special, approx_recips=approx_recips)
    return result_x, result_y


def _apply_compute_simd(
    state: State,
    insn: Instruction,
    result_x: tuple[int | str, Value, str, Callable[[Value], Value]] | None,
    result_y: tuple[int | str, Value, str, Callable[[Value], Value]] | None,
) -> None:
    if result_x is not None:
        _apply_compute(state, insn, result_x)
    if result_y is not None:
        _apply_compute_pey(state, insn, result_y)
