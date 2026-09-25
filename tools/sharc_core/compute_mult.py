"""Multiplier compute (cu=1) and MR data moves: handler bodies dispatched
by tools/sharc_core/compute.py's MULT_OPS table (plus MR_DATAMOVE_REGISTERS/
_mr_data_move for the separate MRDATAMOVE encoding, and the two
multiply-accumulate helpers compute.py calls directly ahead of its
mf/cu dispatch -- see compute.py's docstring for why).

The multiplier's 80-bit result accumulators (MRF/MRB, or MSF/MSB for PEy --
SHARC+ PRM p.3-10) are modelled as ``state.MR`` values (see state.py),
carried through ``special["MRF"]``/``special["MRB"]`` the same way this
module always has. Every fixed-point row of PRM Table 17-7 ("MULOP Encode
Table", out/refs/sharc-plus-prm/all.txt:22612-22660) is generated
programmatically below (``_build_mult_ops``) from the same MOD1/MOD2/MOD3
bit layout the manual documents (all.txt:22660-22734), rather than
hand-written per opcode: each opcode's fixed/variable bits are derived once
in one of the five handler-builder functions
(``_mult_handler``/``_mac_handler``/``_sat_handler``/``_rnd_handler``/
``_clear_handler``), and ``tests/test_sharc_compute_table.py`` cross-checks
every generated key against ``tools/sharcspec/compute_table.json``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from .flags import _astatx_apply_bits, _astatx_mult_clear, _astatx_mult_forget
from .floats import _float_binary
from .state import MR, MR_ZERO, _mr_from_signed, _mr_read_word, _mr_write_word, _ureg
from .values import (
    ComputeDest,
    ComputeValue,
    Const,
    Operand,
    Unknown,
    Value,
    _multiply,
    _signed32,
)

# A multiplier handler's result may carry a full 80-bit MR (state.py, a
# layer above values.py's Operand/ComputeValue by design -- values.py
# cannot import MR without an import cycle), so this module uses this
# locally widened result type instead of values.py's ComputeResult.
MultResult = tuple[ComputeDest, "ComputeValue | MR", str, Callable[[Value], Value]]

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
    MultResult,
]

# ASTATX/ASTATY bit positions this module updates (SHARC+ PRM ch.4/ch.28
# REGF_ASTATX). Imported directly from encoding.py rather than through
# flags.py, since flags.py only re-exports these as opaque updater
# functions, not the raw bit numbers this module's own flag formula needs.
from .encoding import MI_BIT, MN_BIT, MU_BIT, MV_BIT  # noqa: E402

# 80-bit two's-complement field mask (state.py's MR uses the same one, but
# does not export it; redefined here from the same PRM p.3-10 fact -- MR2F
# is bits 79:64 -- rather than reaching into state.py's private name).
_MR_MASK = (1 << 80) - 1


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
# mr_name -> (accumulator key in state.special, word index 0/1/2 within it).
_MR_DATAMOVE_WORD = {
    "MR0F": ("MRF", 0),
    "MR1F": ("MRF", 1),
    "MR2F": ("MRF", 2),
    "MR0B": ("MRB", 0),
    "MR1B": ("MRB", 1),
    "MR2B": ("MRB", 2),
}


def _mr_data_move(
    mr_name: str,
    rn: int,
    direction: int,
    values: Mapping[int, Value],
    special: Mapping[str, Operand | MR] | None,
) -> MultResult:
    """PRM Table 18-29 MRDATAMOVE (p.438): moves a 32-bit value between the
    register file and one 32/32/16-bit word (MR0/MR1/MR2) of one of the two
    80-bit accumulators (PRM p.3-10/4-77: "Each multiplier has a primary or
    foreground register ... and alternate or background" -- MRF/MRB).
    ``state._mr_read_word``/``_mr_write_word`` hold the actual word-slicing
    and MR1->MR2 sign-extension-on-write rule; this just picks which
    accumulator and word. Flags: PRM p.493, MU/MN/MI/MV all cleared for
    every direction and register."""
    reg_key, word = _MR_DATAMOVE_WORD[mr_name]
    op_name = "mr-data-move-%s" % mr_name.lower()
    current = (special or {}).get(reg_key, Unknown("uninitialized %s" % reg_key))
    if direction:  # register file -> MR word
        new_mr = _mr_write_word(current, word, _ureg(values, rn))
        return reg_key, new_mr, op_name, _astatx_mult_clear
    value = _mr_read_word(current, word)
    return rn, value, op_name, _astatx_mult_clear


def _mr_flags(
    raw80: int, fractional: bool, signed_result: bool
) -> tuple[bool, bool, bool]:
    """MN/MV/MU (SHARC+ PRM p.28-5/28-6, REGF_ASTATX.MN/MV/MU field
    descriptions) for a concrete 80-bit multiplier result RAW80 (the
    unsigned bit pattern), evaluated against the FORMAT (fractional or
    integer, signed or unsigned) the instruction itself declares via its
    MOD1/MOD2/MOD3 bits -- which need not be the format that originally
    produced RAW80 (e.g. ``sat mrf (UI)`` reads back whatever mrf holds as
    an unsigned integer, regardless of how it got there).

    MN is the sign of the full 80-bit field (bit 79), for every format.
    MV/MU each check how many of the upper bits are pure sign/zero padding:
    MV when the value does not fit the format's usable width ("Two's-
    complement fractional with the upper 17 bits of MR not all zeros or all
    ones", etc.), MU (fractional only -- the PRM's MU bullets list only the
    two fractional cases, never an integer one, so integer results never
    underflow) when the value is nonzero but entirely within the sub-32-bit
    region MR0 alone provides.
    """
    mn = bool((raw80 >> 79) & 1)
    if fractional:
        upper48 = (raw80 >> 32) & 0xFFFFFFFFFFFF
        lower32 = raw80 & 0xFFFFFFFF
        if signed_result:
            upper17 = (raw80 >> 63) & 0x1FFFF
            mv = upper17 not in (0, 0x1FFFF)
            mu_upper_ok = upper48 in (0, 0xFFFFFFFFFFFF)
        else:
            upper16 = (raw80 >> 64) & 0xFFFF
            mv = upper16 != 0
            mu_upper_ok = upper48 == 0
        mu = mu_upper_ok and lower32 != 0
    else:
        upper49 = (raw80 >> 31) & 0x1FFFFFFFFFFFF
        if signed_result:
            mv = upper49 not in (0, (1 << 49) - 1)
        else:
            upper48 = (raw80 >> 32) & 0xFFFFFFFFFFFF
            mv = upper48 != 0
        mu = False
    return mn, mv, mu


def _astatx_mult_from(
    raw80: int | None, fractional: bool, signed_result: bool, *, sat: bool = False
) -> Callable[[Value], Value]:
    """ASTATX/ASTATY updater for a fixed-point multiplier result: MN/MV/MU
    from RAW80 via ``_mr_flags``, MI fixed 0 (PRM Table 3-7: every
    fixed-point row lists MI as a flat "0" -- MI only ever applies to the
    floating-point multiply row, see ``_astatx_mult_float``). SAT=True
    additionally fixes MV to 0 (PRM Table 3-7's sat row alone: "MU * MN *
    MV 0 MI 0" -- unlike every other fixed-point row, whose MV column is
    data-dependent). RAW80=None (an unknown operand) forgets MN/MV/MU
    instead of guessing.
    """
    if raw80 is None:
        mn = mu = None
        mv = False if sat else None
    else:
        mn, mv, mu = _mr_flags(raw80, fractional, signed_result)
        if sat:
            mv = False
    updates = {MN_BIT: mn, MV_BIT: mv, MU_BIT: mu, MI_BIT: False}
    return lambda astatx: _astatx_apply_bits(astatx, updates)


def _astatx_mult_float(
    result: Value, overflowed: bool | None, invalid: bool | None
) -> Callable[[Value], Value]:
    """PRM Table 3-9 "FN = FX * FY": MU/MN/MV/MI all data-dependent (unlike
    every fixed-point row). MN is RESULT's own sign bit; MV is OVERFLOWED
    (post-rounded exponent > 127, exactly what ``_float_binary`` already
    reports via struct's OverflowError path); MU is a denormal RESULT
    (biased exponent 0, nonzero mantissa -- "the floating-point result
    underflows (unbiased exponent < -126)", and a denormal's unbiased
    exponent is exactly -127); MI is INVALID as ``_float_binary`` already
    computes it (NaN input)."""
    mn = mv = mu = None
    if isinstance(result, Const):
        bits = result.value
        mn = bool(bits & 0x80000000)
        mv = bool(overflowed)
        mu = (bits & 0x7F800000) == 0 and (bits & 0x007FFFFF) != 0
    updates = {MN_BIT: mn, MV_BIT: mv, MU_BIT: mu, MI_BIT: invalid}
    return lambda astatx: _astatx_apply_bits(astatx, updates)


def _mr_product_raw(
    x: int, y: int, signed_x: bool, signed_y: bool, fractional: bool
) -> int:
    """The 80-bit two's-complement bit pattern (mod 2**80) a fixed-point
    multiply places in the accumulator (PRM p.3-9/3-10, Figure 3-2): the
    exact product of the two operands (each interpreted per SIGNED_X/
    SIGNED_Y), doubled once when both are signed and the format is
    fractional ("If both inputs are fractional and signed, the multiplier
    automatically shifts the result left one bit to remove the redundant
    sign bit"). That single left shift is *all* that distinguishes the
    fractional and integer placements: an integer product's meaningful
    32-bit truncation is bits 31:0 of the result (ordinary 32x32->32
    wraparound multiply), a fractional product's is bits 63:32 (Figure
    3-2's "FRACTIONAL RESULT" region) -- both simply fall out of masking
    the same raw product into the 80-bit field, no extra positional shift
    needed (cross-checked against ``_multiply_fractional``'s existing
    >>31/>>32 register-direct extraction, which ``_mr_extract32`` below
    reduces to when there is no accumulate step)."""
    xv = _signed32(x) if signed_x else (x & 0xFFFFFFFF)
    yv = _signed32(y) if signed_y else (y & 0xFFFFFFFF)
    product = xv * yv
    if fractional and signed_x and signed_y:
        product <<= 1
    return product & _MR_MASK


def _mr_extract32(raw80: int, fractional: bool) -> int:
    """The 32-bit value sent to a register-file location (PRM p.3-10:
    "using bits 6332 for a fractional result or bits 310 for an integer
    result")."""
    return (raw80 >> 32) & 0xFFFFFFFF if fractional else raw80 & 0xFFFFFFFF


def _mr_round(raw80: int) -> int:
    """Round-to-nearest of the 80-bit value at the MR1F/MR0F (bit 32)
    boundary (PRM p.3-11 "Round MRx Instruction": "performs a round to
    nearest of the 80-bit MRF value at bit 32"): add a half-ULP at bit 31
    in the full 80-bit two's-complement field (so it correctly carries into
    MR2F on overflow), then zero the low 32 bits (discarding MR0F -- "the
    rounded result in MR1F can be sent to the register file", matching how
    the same section describes *not* rounding as "discarding bits 310 [...]
    truncates a fractional result")."""
    rounded = (raw80 + (1 << 31)) & _MR_MASK
    return rounded & (_MR_MASK ^ 0xFFFFFFFF)


# (fractional, signed) -> (max, min) representable value (PRM Table 3-5,
# p.3-11 "Fixed-Point Format Maximum Values (Saturation)"), as plain Python
# ints -- masking either bound into 80 bits (``& _MR_MASK``) reproduces the
# table's hex rows exactly, since two's-complement sign-extension of a
# negative Python int under ``&`` already fills MR2F correctly.
_SAT_BOUNDS: dict[tuple[bool, bool], tuple[int, int]] = {
    (True, True): ((1 << 63) - 1, -(1 << 63)),  # fractional, signed
    (True, False): ((1 << 64) - 1, 0),  # fractional, unsigned
    (False, True): ((1 << 31) - 1, -(1 << 31)),  # integer, signed
    (False, False): ((1 << 32) - 1, 0),  # integer, unsigned
}


def _mac_accumulator(
    src_key: str, special: Mapping[str, Operand | MR] | None
) -> tuple[Operand | MR, int | None]:
    accumulator = (special or {}).get(src_key, Unknown("uninitialized %s" % src_key))
    acc_signed = accumulator.signed() if isinstance(accumulator, MR) else None
    return accumulator, acc_signed


def _mult_handler(
    signed_x: bool, signed_y: bool, fractional: bool, round_: bool, dest: str
) -> Handler:
    """PRM Table 17-7 "01yx f00r"/"F10r"/"F11r" rows (p.17-7): (RN|mrf|mrb)
    = RX * RY MOD1 -- a plain multiply, no accumulate. DEST is "rn", "mrf"
    or "mrb"."""
    dest_key = {"mrf": "MRF", "mrb": "MRB"}.get(dest)
    signed_result = signed_x or signed_y
    op_name = (
        "multiply"
        if dest_key is None
        else "multiply-mrf"
        if dest_key == "MRF"
        else "multiply-mrb"
    )

    def handler(rn, rx, ry, left, right, values, special, approx_recips):
        if isinstance(left, Const) and isinstance(right, Const):
            raw = _mr_product_raw(
                left.value, right.value, signed_x, signed_y, fractional
            )
            if round_:
                raw = _mr_round(raw)
            astatx = _astatx_mult_from(raw, fractional, signed_result)
            if dest_key is None:
                return rn, Const(_mr_extract32(raw, fractional)), op_name, astatx
            return dest_key, _mr_from_signed(raw), op_name, astatx
        astatx = _astatx_mult_from(None, fractional, signed_result)
        if dest_key is not None:
            return dest_key, Unknown("R%d * R%d MOD1" % (rx, ry)), op_name, astatx
        # RN destination, no accumulate: preserve the tracer's existing
        # symbolic-affine multiply-by-constant capability for the plain
        # *integer*, unrounded case (PRM p.3-9: the low 32 bits of an
        # integer product do not depend on rounding or the redundant-sign
        # shift, so this is exactly ``_multiply``'s domain -- the same
        # helper the pre-80-bit-MR version of this handler called
        # directly). A fractional or rounded RN result still has no
        # symbolic model, matching ``_multiply_fractional``'s own Unknown
        # fallback for anything but two Consts.
        value = (
            Unknown("R%d * R%d MOD1" % (rx, ry))
            if fractional or round_
            else _multiply(left, right, "R%d * R%d" % (rx, ry))
        )
        return rn, value, op_name, astatx

    return handler


def _mac_handler(
    signed_x: bool,
    signed_y: bool,
    fractional: bool,
    round_: bool,
    dest: str,
    subtract: bool,
    src: str,
) -> Handler:
    """PRM Table 17-7 "10yx.../11yx..." rows (p.17-7): (RN|mrf|mrb) =
    (mrf|mrb) [+-] RX * RY MOD1. DEST is "rn", "mrf" or "mrb"; SRC ("mrf" or
    "mrb") is the accumulator read (and, unless DEST=="rn", also written)."""
    src_key = "MRF" if src == "mrf" else "MRB"
    dest_key = {"mrf": "MRF", "mrb": "MRB"}.get(dest)
    signed_result = signed_x or signed_y
    op_name = "multiply-subtract" if subtract else "multiply-accumulate"

    def handler(rn, rx, ry, left, right, values, special, approx_recips):
        _accumulator, acc_signed = _mac_accumulator(src_key, special)
        if acc_signed is None or not (
            isinstance(left, Const) and isinstance(right, Const)
        ):
            astatx = _astatx_mult_from(None, fractional, signed_result)
            label = "%s %s R%d * R%d MOD1" % (src_key, "-" if subtract else "+", rx, ry)
            return (dest_key or rn, Unknown(label), op_name, astatx)
        product = _mr_product_raw(
            left.value, right.value, signed_x, signed_y, fractional
        )
        product_signed = product - (1 << 80) if product & (1 << 79) else product
        raw = (
            (acc_signed - product_signed) if subtract else (acc_signed + product_signed)
        ) & _MR_MASK
        if round_:
            raw = _mr_round(raw)
        astatx = _astatx_mult_from(raw, fractional, signed_result)
        if dest_key is None:
            return rn, Const(_mr_extract32(raw, fractional)), op_name, astatx
        return dest_key, _mr_from_signed(raw), op_name, astatx

    return handler


def _sat_handler(fractional: bool, signed_: bool, dest: str, src: str) -> Handler:
    """PRM Table 17-7 "0000 F--x" rows (p.17-7): (RN|mrf|mrb) = sat
    (mrf|mrb) MOD2 -- clamp the accumulator to the MOD2-declared format's
    representable range (Table 3-5). MV is architecturally fixed 0 for this
    row (saturation removes overflow by definition); MN/MU still reflect
    the pre-saturation accumulator (a saturated result can never itself
    underflow or be negative-but-clamped-positive, so evaluating them
    post-clamp would just report the same thing the clamp already forces)."""
    src_key = "MRF" if src == "mrf" else "MRB"
    dest_key = {"mrf": "MRF", "mrb": "MRB"}.get(dest)
    op_name = "saturate-mrf" if src_key == "MRF" else "saturate-mrb"
    max_value, min_value = _SAT_BOUNDS[(fractional, signed_)]

    def handler(rn, rx, ry, left, right, values, special, approx_recips):
        _accumulator, acc_signed = _mac_accumulator(src_key, special)
        if acc_signed is None:
            astatx = _astatx_mult_from(None, fractional, signed_, sat=True)
            return (
                dest_key or rn,
                Unknown("saturated %s (uninitialized)" % src_key),
                op_name,
                astatx,
            )
        clamped = max(min_value, min(max_value, acc_signed))
        astatx = _astatx_mult_from(acc_signed & _MR_MASK, fractional, signed_, sat=True)
        raw = clamped & _MR_MASK
        if dest_key is None:
            return rn, Const(_mr_extract32(raw, fractional)), op_name, astatx
        return dest_key, _mr_from_signed(raw), op_name, astatx

    return handler


def _rnd_handler(signed_: bool, dest: str, src: str) -> Handler:
    """PRM Table 17-7 "0001 1..." rows (p.17-7): (RN|mrf|mrb) = rnd
    (mrf|mrb) MOD3. Always fractional -- MOD3 only has SF/UF options ("The
    RND operation ... applies only to fractional results", PRM p.3-11)."""
    src_key = "MRF" if src == "mrf" else "MRB"
    dest_key = {"mrf": "MRF", "mrb": "MRB"}.get(dest)
    op_name = "round-mrf" if src_key == "MRF" else "round-mrb"

    def handler(rn, rx, ry, left, right, values, special, approx_recips):
        _accumulator, acc_signed = _mac_accumulator(src_key, special)
        if acc_signed is None:
            astatx = _astatx_mult_from(None, True, signed_)
            return (
                dest_key or rn,
                Unknown("rounded %s (uninitialized)" % src_key),
                op_name,
                astatx,
            )
        raw = _mr_round(acc_signed & _MR_MASK)
        astatx = _astatx_mult_from(raw, True, signed_)
        if dest_key is None:
            return rn, Const(_mr_extract32(raw, True)), op_name, astatx
        return dest_key, _mr_from_signed(raw), op_name, astatx

    return handler


def _clear_handler(dest: str) -> Handler:
    """PRM Table 17-7 "0001 01d0" rows (p.17-7): (mrf|mrb) = 0. PRM Table
    3-7: all four flags fixed 0, the same pattern ``_astatx_mult_clear``
    already gives the MR-data-move rows."""
    dest_key = "MRF" if dest == "mrf" else "MRB"
    op_name = "clear-mrf" if dest_key == "MRF" else "clear-mrb"

    def handler(rn, rx, ry, left, right, values, special, approx_recips):
        return dest_key, MR_ZERO, op_name, _astatx_mult_clear

    return handler


# PRM Table 18-7 (p.428-429): MULOP 00110000 is Fn = Fx * Fy. Flags are the
# multiplier's MN/MV/MU/MI (PRM Table 3-9), computed by _astatx_mult_float
# above rather than forgotten -- unlike the fixed-point rows, every one of
# these four is genuinely data-dependent for a float multiply.
def mult_float(rn, rx, ry, left, right, values, special, approx_recips) -> tuple:
    result, overflowed, invalid = _float_binary(
        left, right, "F%d * F%d" % (rx, ry), lambda a, b: a * b
    )
    astatx = _astatx_mult_float(result, overflowed, invalid)
    return rn, result, "float-multiply", astatx


# Undocumented in both public sources: PRM Table 17-7 and PGR Table 12-5
# both list only mrf/mrb=0 (0001 0100/0110) and rnd MOD3 (0001 100x-111x)
# under the "0001 xxxx" opcode prefix -- neither has a 0001 0000 row.
# Decode it (so the walk does not desync) and leave the result and flags
# Unknown, same as the shifter's undocumented 0xB0 gap in compute_shift.py.
def mult_undocumented_10(
    rn, rx, ry, left, right, values, special, approx_recips
) -> tuple:
    label = "multiply opcode 0x10 R%d, R%d (undocumented; no public source)" % (rx, ry)
    return rn, Unknown(label), "multiply-undocumented-10", _astatx_mult_forget


def _build_mult_ops() -> dict[int, Handler]:
    """PRM Table 17-7's full fixed-point MULOP encode space (p.17-7/17-8),
    generated from the same MOD1/MOD2/MOD3 bit layout the manual documents
    (p.17-9: y=bit5 Y-signed, x=bit4 X-signed, f=bit3 fractional, r=bit0
    round) rather than one handler per opcode by hand. 0xB4/0xB0 (the two
    multiply-accumulate-into-MRF rows compute.py's dispatch matches on the
    raw field *before* the mf bit, ahead of ever consulting this table --
    see compute.py's module docstring) are deliberately excluded here, even
    though their bit patterns fall out of the same "10yx F10r"/"F00r" loops
    below; ``multiply_accumulate_mrf``/``multiply_add_mrf`` are built the
    same way, just not inserted into MULT_OPS.
    """
    ops: dict[int, Handler] = {}

    # "0000 Fbdx": (RN|mrf|mrb) = sat (mrf|mrb) MOD2. bit3=F (MOD2
    # fractional/integer), bit2=b (0=mrf source,1=mrb), bit1=d (0=dest RN,
    # 1=dest is the same register as the source), bit0=s (MOD2 signed).
    for frac_bit, fractional in ((0, False), (1, True)):
        for b_bit, src in ((0, "mrf"), (1, "mrb")):
            for d_bit, dest in ((0, "rn"), (1, src)):
                for s_bit, signed_ in ((0, False), (1, True)):
                    opcode = (frac_bit << 3) | (b_bit << 2) | (d_bit << 1) | s_bit
                    ops[opcode] = _sat_handler(fractional, signed_, dest, src)

    # "0001 01d0": (mrf|mrb) = 0.
    for d_bit, dest in ((0, "mrf"), (1, "mrb")):
        opcode = 0b00010100 | (d_bit << 1)
        ops[opcode] = _clear_handler(dest)

    # "0001 1bdx": (RN|mrf|mrb) = rnd (mrf|mrb) MOD3. bit2=b (source),
    # bit1=d (dest), bit0=MOD3 sign.
    for b_bit, src in ((0, "mrf"), (1, "mrb")):
        for d_bit, dest in ((0, "rn"), (1, src)):
            for s_bit, signed_ in ((0, False), (1, True)):
                opcode = 0b00011000 | (b_bit << 2) | (d_bit << 1) | s_bit
                ops[opcode] = _rnd_handler(signed_, dest, src)

    # "01yx f00r"/"F10r"/"F11r": (RN|mrf|mrb) = RX*RY MOD1 (no accumulate).
    # bits2:1 select the destination (00=RN, 10=mrf, 11=mrb).
    for y_bit, signed_y in ((0, False), (1, True)):
        for x_bit, signed_x in ((0, False), (1, True)):
            for f_bit, fractional in ((0, False), (1, True)):
                for r_bit, round_ in ((0, False), (1, True)):
                    if r_bit and not f_bit:
                        continue  # MOD1 has no integer-round ("...IR") option
                    for dest_bits, dest in ((0b00, "rn"), (0b10, "mrf"), (0b11, "mrb")):
                        opcode = (
                            0b01000000
                            | (y_bit << 5)
                            | (x_bit << 4)
                            | (f_bit << 3)
                            | (dest_bits << 1)
                            | r_bit
                        )
                        ops[opcode] = _mult_handler(
                            signed_x, signed_y, fractional, round_, dest
                        )

    # "10yx.../11yx...": (RN|mrf|mrb) = (mrf|mrb) [+-] RX*RY MOD1. bit7:6
    # selects add (10, 0x80) / subtract (11, 0xC0); bits2:1 select
    # dest/source together, since a RN destination still has to say which
    # accumulator was read (00=RN from mrf, 01=RN from mrb, 10=mrf, 11=mrb).
    for subtract, hi_bits in ((False, 0b10), (True, 0b11)):
        for y_bit, signed_y in ((0, False), (1, True)):
            for x_bit, signed_x in ((0, False), (1, True)):
                for f_bit, fractional in ((0, False), (1, True)):
                    for r_bit, round_ in ((0, False), (1, True)):
                        if r_bit and not f_bit:
                            continue
                        for dest_bits, dest, src in (
                            (0b00, "rn", "mrf"),
                            (0b01, "rn", "mrb"),
                            (0b10, "mrf", "mrf"),
                            (0b11, "mrb", "mrb"),
                        ):
                            opcode = (
                                (hi_bits << 6)
                                | (y_bit << 5)
                                | (x_bit << 4)
                                | (f_bit << 3)
                                | (dest_bits << 1)
                                | r_bit
                            )
                            if opcode in (0xB0, 0xB4):
                                continue  # compute.py intercepts these first
                            ops[opcode] = _mac_handler(
                                signed_x,
                                signed_y,
                                fractional,
                                round_,
                                dest,
                                subtract,
                                src,
                            )

    ops[0x30] = mult_float
    ops[0x10] = mult_undocumented_10
    return ops


MULT_OPS: dict[int, Handler] = _build_mult_ops()

# The two multiply-accumulate-into-MRF rows compute.py's dispatch matches
# ahead of the mf/cu table (see its module docstring): opcode 0xB4 ("mrf =
# mrf + RX*RY MOD1 SSI") and 0xB0 ("RN = mrf + RX*RY MOD1 SSI"), built from
# the exact same _mac_handler this module uses for every other MOD1
# combination -- both a real 80-bit accumulate now, not the previous
# 32-bit-truncating placeholder.
multiply_accumulate_mrf: Handler = _mac_handler(
    True, True, False, False, "mrf", False, "mrf"
)
multiply_add_mrf: Handler = _mac_handler(True, True, False, False, "rn", False, "mrf")
