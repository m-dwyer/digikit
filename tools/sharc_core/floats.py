"""32-bit IEEE float and fixed/float conversion helpers.

Moved verbatim from tools/sharc_trace.py.
"""

from __future__ import annotations

import math
import struct

from .encoding import (
    AC_BIT,
    AI_BIT,
    ALUSAT_BIT,
    AN_BIT,
    AS_BIT,
    AV_BIT,
    AZ_BIT,
    TRUNCATE_BIT,
)
from .values import (
    Const,
    Unknown,
    Value,
    _astatx_known_bit,
    _signed,
    _signed32,
)

# ---------------------------------------------------------------------------
# Floating-point compute support.
#
# The register file (R0-R15/F0-F15) is a flat 32-bit store either way; a
# float ALU/multiplier op just reinterprets the same bits as IEEE-754 single
# precision (PRM p.3-4: "floating-point instructions operate on 32-bit ...
# operands"). The SHARC+ ALU/multiplier additionally support an optional
# 40-bit extended-precision float format for *intermediate* results (PRM
# p.3-37, active when MODE1.RND32=0, the reset default: "eight additional
# LSBs of mantissa"), but that only matters to values forwarded between
# back-to-back compute ops without ever reaching the register file; this
# tracer has no pipeline/forwarding model and every UREG it tracks is a
# plain 32-bit value, so every float result here is computed and stored at
# IEEE-754 single precision -- the same width a real RN/FN write-back uses
# regardless of the extended-precision mode. Denormal-flush-to-zero, which
# several PGR entries below document for their inputs/outputs, is also not
# modeled (struct's round-trip preserves denormals exactly); this only
# matters for subnormal magnitudes, which real audio sample/parameter data
# essentially never produces.
# ---------------------------------------------------------------------------


def _float32(value: Value) -> float | None:
    """Reinterpret VALUE's 32-bit pattern as IEEE-754 single precision.

    Returns None when VALUE isn't a fully known Const: an Affine (symbolic
    address arithmetic) or Unknown source is not float data and must not be
    silently coerced into one.
    """
    if not isinstance(value, Const):
        return None
    return struct.unpack("<f", struct.pack("<I", value.value))[0]


def _float32_bits(value: float) -> tuple[int, bool]:
    """Round VALUE to IEEE-754 single precision; return (bits, overflowed).

    struct raises OverflowError for a finite double outside the float32
    range. SHARC+ float overflow rounds to signed infinity in the default
    round-to-nearest mode (PRM Table 3-3 / PGR p.11-24 AV description:
    "post-rounded result overflows ... returns +-infinity"); reproduce that
    by hand when struct refuses, since struct has no float32-infinity
    fallback of its own.
    """
    try:
        return struct.unpack("<I", struct.pack("<f", value))[0], False
    except OverflowError:
        sign = 0x80000000 if math.copysign(1.0, value) < 0 else 0
        return (0x7F800000 | sign), True


_FLOAT_ALL_ONES = Const(0xFFFFFFFF)


def _float_binary(
    left: Value, right: Value, expression: str, operation
) -> tuple[Value, bool | None, bool | None]:
    """Evaluate a float ALU/multiplier binary OPERATION; return (result,
    overflowed, invalid).

    Several PGR float-ALU entries (e.g. p.11-24 Fx+Fy, p.11-46 MIN, p.11-48
    CLIP) document "A NAN input returns an all 1s result" -- a fixed
    sentinel pattern, not whatever IEEE NaN OPERATION would naturally
    produce -- so an explicit NaN *input* is special-cased before OPERATION
    ever runs. A NaN produced by OPERATION itself from two non-NaN inputs
    (e.g. +infinity + -infinity, PGR p.11-24's "opposite-signed infinities"
    AI case) is not overridden: it keeps the ordinary computed NaN pattern.
    Either input not being a known Const makes the whole result unknown.
    """
    a, b = _float32(left), _float32(right)
    if a is None or b is None:
        return Unknown(expression), None, None
    if math.isnan(a) or math.isnan(b):
        return _FLOAT_ALL_ONES, False, True
    raw = operation(a, b)
    bits, overflowed = _float32_bits(raw)
    return Const(bits), overflowed, math.isnan(raw)


def _float_unary(
    value: Value, expression: str, operation
) -> tuple[Value, bool | None, bool | None]:
    """Unary counterpart of ``_float_binary`` (see its docstring)."""
    a = _float32(value)
    if a is None:
        return Unknown(expression), None, None
    if math.isnan(a):
        return _FLOAT_ALL_ONES, False, True
    raw = operation(a)
    bits, overflowed = _float32_bits(raw)
    return Const(bits), overflowed, math.isnan(raw)


def _float_min(a: float, b: float) -> float:
    """PGR p.11-46: smaller operand; min(+0, -0) is documented as -0."""
    if a == 0.0 and b == 0.0:
        return -0.0
    return a if a < b else b


def _float_max(a: float, b: float) -> float:
    """PGR p.11-47: larger operand; max(+0, -0) is documented as +0."""
    if a == 0.0 and b == 0.0:
        return 0.0
    return a if a > b else b


def _float_clip(a: float, b: float) -> float:
    """PGR p.11-48 / PRM p.3-6 CLIP: FX if |FX| < |FY|, else +-|FY| with
    FX's sign (copysign handles the FX=+-0 boundary the same as the PGR
    text's "if Fx is positive")."""
    return a if abs(a) < abs(b) else math.copysign(abs(b), a)


def _float_mantissa(
    value: Value, expression: str
) -> tuple[Value, bool | None, bool | None, bool | None]:
    """RN = mant FX (PRM Table 18-5 opcode 0xAD, p.427; PGR p.11-34/11-35).

    Extracts the hidden bit plus the 23-bit fraction, left-justified as an
    unsigned-magnitude 1.31 fixed-point word (bit31 the hidden bit, bits
    30-8 the fraction, bits 7-0 zero-filled); the 24 significant bits
    always fit exactly, so no rounding is performed (PGR: "no rounding is
    performed because all results are inherently exact"). Denormal and
    zero inputs flush to a zero mantissa (PGR: "Denormal inputs are
    flushed to +-zero"). A NAN *or an infinity* input returns the fixed
    all-1s sentinel (PGR: "A NAN or an infinity input returns an all 1s
    result") -- unlike the arithmetic float ALU ops above, which only
    override NAN, MANT also overrides infinity, since it is not an
    ordinary IEEE operation. Returns (result, overflow=is-infinity,
    negative=input-sign, invalid=is-NAN); AN is always cleared for this
    op (PRM Table 3-3) and is not returned here.
    """
    if not isinstance(value, Const):
        return Unknown(expression), None, None, None
    bits = value.value
    sign = bool(bits & 0x80000000)
    exponent = (bits >> 23) & 0xFF
    fraction = bits & 0x7FFFFF
    if exponent == 0xFF:
        is_nan = fraction != 0
        return _FLOAT_ALL_ONES, not is_nan, sign, is_nan
    if exponent == 0:
        return Const(0), False, sign, False
    return Const((0x800000 | fraction) << 8), False, sign, False


def _float_logb(
    value: Value, mode1: Value, expression: str
) -> tuple[Value, bool | None, bool]:
    """RN = logb FX (PGR p.11-36, pgr.txt:21522): the unbiased two's-
    complement exponent of FX as a fixed-point integer (biased_exp - 127).
    A NAN input returns the all-1s sentinel. A +-infinity or +-zero input
    (denormals flush to zero first, same general rule ``_float_mantissa``
    already cites) returns a value gated on MODE1.ALUSAT exactly like
    ``_float_to_fixed``'s saturation branches: unsaturated, PGR says these
    return the *floating-point* +infinity/-infinity bit pattern verbatim in
    the (nominally fixed-point) result register; saturated, the maximum
    positive/negative fixed-point integer. Returns (result,
    overflow=input-is-inf-or-zero, invalid=is-NAN); AV/AI feed AV/AI
    directly, AN/AZ come from RESULT's own bits at the call site (RESULT is
    a plain integer here, not a float, so ``_float_alu_updates`` does not
    apply)."""
    if not isinstance(value, Const):
        return Unknown(expression), None, False
    bits = value.value & 0xFFFFFFFF
    biased_exp = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    if biased_exp == 0xFF and mantissa != 0:
        return _FLOAT_ALL_ONES, False, True
    if biased_exp == 0xFF or biased_exp == 0:
        saturating = _astatx_known_bit(mode1, ALUSAT_BIT)
        if saturating is None:
            return Unknown(expression), True, False
        if biased_exp == 0xFF:
            result = Const(0x7FFFFFFF) if saturating else Const(0x7F800000)
        else:
            result = Const(0x80000000) if saturating else Const(0xFF800000)
        return result, True, False
    return Const(_signed(biased_exp - 127, 32) & 0xFFFFFFFF), False, False


def _float_scalb(
    value: Value, scale: Value, expression: str
) -> tuple[Value, bool | None, bool | None]:
    """FN = scalb FX by RY (PRM Table 18-5 opcode 0xBD, p.427; PGR p.11-33).

    Adds the two's-complement fixed-point integer RY to FX's exponent
    (i.e. FX * 2**RY). Overflow rounds to +-infinity (round-to-nearest,
    the only rounding mode this tracer models, matching every other float
    op here); a result whose magnitude underflows below the smallest
    float32 normal (2**-126) flushes to +-zero rather than becoming a
    subnormal (PGR: "Denormal returns +-zero" -- an explicit override of
    struct's ordinary IEEE denormal rounding, the same kind of override
    ``_float_to_fixed_trunc`` already applies for its own corner cases). A
    NAN input returns the same all-1s sentinel ``_float_binary`` uses;
    zero and infinity inputs pass through unchanged (``math.ldexp``
    preserves both, matching the PRM, which documents no special case for
    them). Returns (result, overflow, invalid).
    """
    a = _float32(value)
    if a is None or not isinstance(scale, Const):
        return Unknown(expression), None, None
    if math.isnan(a):
        return _FLOAT_ALL_ONES, False, True
    shift = _signed32(scale.value)
    try:
        scaled = math.ldexp(a, shift)
    except OverflowError:
        scaled = math.copysign(math.inf, a)
    if scaled != 0.0 and not math.isinf(scaled) and abs(scaled) < 2.0**-126:
        return Const(0x80000000 if scaled < 0 else 0), False, False
    bits, overflowed = _float32_bits(scaled)
    return Const(bits), overflowed, False


def _fixed_to_float(value: Value, expression: str) -> tuple[Value, bool | None]:
    """FN = float RX (PRM Table 18-5 opcode 0xCA, p.427; PGR p.11-39 "without
    scaling factor"): numeric int32->float32 conversion, not a bit
    reinterpretation. PGR documents AV and AI both fixed 0 for the
    no-scaling form actually used here (RN=FLOAT RX BY RY, which also takes
    a scale factor, is not implemented). Returns (result, invalid) where
    invalid is always False when computable, matching that fixed AI=0.
    """
    if not isinstance(value, Const):
        return Unknown(expression), None
    bits, _ = _float32_bits(float(_signed32(value.value)))
    return Const(bits), False


def _float_to_fixed(
    value: Value, mode1: Value, always_truncate: bool, expression: str
) -> tuple[Value, bool | None, bool | None]:
    """RN = FIX FX / RN = TRUNC FX (PRM Table 18-5 opcodes 0xC9/0xCD, p.427;
    PGR p.11-36..11-38), and their scaled BY RY siblings 0xD9/0xDD (the
    caller pre-scales VALUE via ``_scale_fixed_input`` -- PGR Table 3-3,
    p.3-11 marks the BY RY forms with the identical AZ/AV/AN/AC/AS/AI
    columns as the unscaled ones, so no separate flag rule is needed here).

    TRUNC always truncates toward zero, ignoring MODE1 (ALWAYS_TRUNCATE=True
    -- PGR p.11-37: "The trunc operation always truncates toward 0. The
    TRUNCATE bit does not influence operation of the trunc instruction.").
    FIX instead rounds to nearest, ties to even (Python's ``round()`` on a
    float already implements this) when MODE1.TRUNCATE=0, or truncates
    toward zero when MODE1.TRUNCATE=1 (PGR p.11-37 / PRM p.24-..); the whole
    result is Unknown when TRUNCATE itself is unknown, rather than guessing
    which rounding applied.

    A result within int32 range needs no saturation. Out-of-range
    magnitudes and NAN/+-infinity inputs are governed by MODE1.ALUSAT (PGR:
    "In saturation mode ... positive overflows and +infinity return
    0x7FFFFFFF, and negative overflows and -infinity return 0x80000000");
    when ALUSAT is known clear the PGR instead documents a "floating-point
    all 1s" *Rn* pattern for that corner, an architecturally odd case this
    tracer does not attempt to reproduce bit-for-bit, so it reports Unknown
    there and whenever MODE1 itself isn't known, rather than guessing.
    Returns (result, overflow, invalid).
    """
    a = _float32(value)
    if a is None:
        return Unknown(expression), None, None
    saturating = _astatx_known_bit(mode1, ALUSAT_BIT)
    if math.isnan(a) or math.isinf(a):
        if saturating:
            return (
                Const(0x7FFFFFFF if (math.isnan(a) or a > 0) else 0x80000000),
                True,
                True,
            )
        if saturating is False:
            return Unknown(expression + " (unsaturated NAN/infinity fix)"), True, True
        return Unknown(expression), None, True
    if always_truncate:
        rounded = math.trunc(a)
    else:
        truncate_mode = _astatx_known_bit(mode1, TRUNCATE_BIT)
        if truncate_mode is None:
            return Unknown(expression), None, False
        rounded = math.trunc(a) if truncate_mode else int(round(a))
    if -(1 << 31) <= rounded <= (1 << 31) - 1:
        return Const(rounded & 0xFFFFFFFF), False, False
    if saturating:
        return Const(0x7FFFFFFF if rounded > 0 else 0x80000000), True, False
    if saturating is False:
        return Unknown(expression + " (unsaturated fix overflow)"), True, False
    return Unknown(expression), None, False


def _float_to_fixed_trunc(
    value: Value, mode1: Value, expression: str
) -> tuple[Value, bool | None, bool | None]:
    """RN = TRUNC FX: ``_float_to_fixed`` with ALWAYS_TRUNCATE=True. Kept as
    a named wrapper since opcode 0xCD's call site predates the shared
    FIX/TRUNC helper and reads more clearly with its own name."""
    return _float_to_fixed(value, mode1, True, expression)


def _scale_fixed_input(value: Value, scale: Value, expression: str) -> Value:
    """Fx * 2**Ry (exponent add), the shared first step of RN = FIX/TRUNC FX
    BY RY (PGR p.11-37: "the fixed-point two's-complement integer in Ry is
    added to the exponent of the floating-point operand in Fx before the
    conversion"). Reuses ``_float_scalb``'s ldexp/overflow/denormal-flush
    rule -- the same exponent-add primitive FN=SCALB uses -- and discards
    its own (overflow, invalid) pair, since the caller's FIX/TRUNC applies
    its own overflow/NAN handling to the scaled value afterward.
    """
    scaled, _overflow, _invalid = _float_scalb(value, scale, expression)
    return scaled


def _fixed_to_float_scaled(
    value: Value, scale: Value, expression: str
) -> tuple[Value, bool | None]:
    """FN = FLOAT RX BY RY (PRM p.19-.. ; PGR p.11-39 "with scaling factor"):
    numeric int32->float32 conversion as ``_fixed_to_float``, then the
    fixed-point two's-complement integer in RY is added to the result's
    exponent (PGR: "the fixed-point two's-complement integer in Ry is added
    to the exponent of the floating-point result"). Overflow (unbiased
    exponent > 127) returns +-infinity; underflow (unbiased exponent <
    -126) flushes to +-zero (PGR: "Overflow generates a return of
    +-infinity ...; underflow generates a return of +-zero"). Unlike the
    unscaled form -- where an int32 input can never land in the subnormal
    range, so AV is architecturally fixed 0 (PGR Table 3-3) -- AV here is
    genuinely data-dependent, so this returns (result, overflow) rather
    than the unscaled helper's implicit always-False.
    """
    if not isinstance(value, Const) or not isinstance(scale, Const):
        return Unknown(expression), None
    unscaled = float(_signed32(value.value))
    shift = _signed32(scale.value)
    try:
        scaled = math.ldexp(unscaled, shift)
    except OverflowError:
        scaled = math.copysign(math.inf, unscaled) if unscaled != 0.0 else 0.0
    if scaled != 0.0 and not math.isinf(scaled) and abs(scaled) < 2.0**-126:
        return Const(0x80000000 if scaled < 0 else 0), False
    bits, overflowed = _float32_bits(scaled)
    return Const(bits), overflowed


def _float_copysign(
    left: Value, right: Value, expression: str
) -> tuple[Value, bool | None]:
    """FN = FX copysign FY (PRM p.19-19, opcode 1110 0000; PGR p.11-45,
    Table 12-4 opcode 1110 0000): copies FY's sign bit onto FX's exponent
    and mantissa unchanged. A denormal FX input flushes to zero before the
    sign copy (PRM/PGR: "A denormal input is flushed to +-zero"). A NAN
    input -- either operand -- returns the fixed all-1s sentinel
    (``_float_binary``'s convention). Returns (result, invalid); AC/AS/AV
    are architecturally fixed for this op (PRM Table 3-3 / PGR Table 3-3)
    and are not returned here.
    """
    a, b = _float32(left), _float32(right)
    if a is None or b is None:
        return Unknown(expression), None
    if math.isnan(a) or math.isnan(b):
        return _FLOAT_ALL_ONES, True
    magnitude = abs(a)
    if 0.0 < magnitude < 2.0**-126:
        magnitude = 0.0
    negative = math.copysign(1.0, b) < 0
    result = -magnitude if negative else magnitude
    bits, _overflowed = _float32_bits(result)
    return Const(bits), False


def _float_round32(value: Value, expression: str) -> tuple[Value, bool | None]:
    """FN = rnd FX (PRM Table 18-5 opcode 1010 0101, p.20-8 "32-bit and
    40-bit Operations"; PGR Table 12-4 opcode 1010 0101, pp.12-3/12-4, and
    p.11-33): rounds FX to a 32-bit floating-point boundary.

    The PRM's own wording for the rounding-mode choice ("as defined by the
    REGF_MODE1.RND32 bit") is inconsistent with what RND32 documents
    elsewhere in the same manual (register-map chapter, p.29-55: RND32
    selects whether the computational units round floating-point data to
    32 bits or 40 bits -- an output-width choice, not a nearest-vs-truncate
    one) and with the classic PGR's wording for the identical op ("the
    rounding mode bit in MODE1"); this follows the PGR and every other
    rounding-mode citation in this file (MODE1.TRUNCATE -- see
    ``_float_to_fixed``'s docstring).

    That rounding-mode choice is not observable here regardless: per this
    module's "Floating-point compute support" header comment, every UREG
    this tracer tracks is already stored at IEEE-754 single precision, so
    FX is already rounded to the 32-bit boundary this op targets, and
    re-rounding an already-32-bit-precision, finite, normal input is a
    no-op under either rounding rule. The "post-rounded overflow" corner
    the manual documents only arises from rounding away mantissa bits
    beyond 32-bit precision, which this tracer never carries between ops;
    the float-pass op (opcode 0xA1, via ``_float_unary``) already fixes
    AV=False on the same reasoning, so this does too. A denormal input
    still flushes to +-zero (this op documents that override explicitly,
    the same as ``_float_copysign`` above); a NAN input returns the fixed
    all-1s sentinel. Returns (result, invalid).
    """
    a = _float32(value)
    if a is None:
        return Unknown(expression), None
    if math.isnan(a):
        return _FLOAT_ALL_ONES, True
    magnitude = abs(a)
    if 0.0 < magnitude < 2.0**-126:
        return Const(0x80000000 if a < 0 else 0), False
    bits, _overflowed = _float32_bits(a)
    return Const(bits), False


def _approx_recips(left: Value) -> tuple[Value, dict[int, bool | None]]:
    """Opt-in ``--approx-recips`` model of ``FN = recips FX``.

    PRM p.19-16/19-17 (out/refs/sharc-plus-prm, quoted in ``_compute``'s
    recips/rsqrts branch below): "Creates an 8-bit accurate seed for
    1/Fx... The mantissa of the seed is determined from a ROM table using
    the 7 MSBs (excluding the hidden bit) of the Fx mantissa as an index."
    That ROM table's contents are not published, so this cannot reproduce
    the real hardware seed bit for bit. It instead:

      - reproduces every documented special case exactly: NaN input ->
        all-1s result (PRM p.417-418 IEEE-754-compatibility bullet: "NAN
        inputs ... return a quiet NAN (all 1s)"); +-zero input -> +-infinity
        with the overflow flag; an Fx unbiased exponent > +125 -> +-zero;
      - flushes a denormal input to +-zero first, per the same PRM section's
        general rule ("Denormal operands ... flush to zero when input to a
        computational unit"), which recips's own page does not restate but
        which applies to every computational unit;
      - for the ordinary case, derives the seed's exponent from the
        documented rule (unbiased exponent of Fn = -e-1, e = Fx's unbiased
        exponent) and approximates its mantissa as the true mathematical
        reciprocal's mantissa, truncated to the documented 8-bit accuracy
        (the low 15 of 23 mantissa bits zeroed) so as not to claim
        precision no public source confirms.

    Every value this returns is an approximation the caller must not treat
    as ground truth; ``_apply_compute`` tags it with an "approximate-recips"
    trace event so a report can always tell it apart from a real seed.
    """
    updates: dict[int, bool | None] = {AC_BIT: False, AS_BIT: False}
    if not isinstance(left, Const):
        updates.update({AV_BIT: None, AI_BIT: None, AN_BIT: None, AZ_BIT: None})
        return Unknown("recips seed (symbolic input)"), updates
    bits = left.value & 0xFFFFFFFF
    sign = (bits >> 31) & 1
    biased_exp = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    if biased_exp == 0xFF and mantissa != 0:  # NaN
        updates.update({AI_BIT: True, AN_BIT: bool(sign), AV_BIT: False, AZ_BIT: False})
        return Const(0xFFFFFFFF), updates
    updates[AI_BIT] = False
    updates[AN_BIT] = bool(sign)
    if biased_exp == 0:  # +-zero, or a denormal flushed to zero on input
        updates[AV_BIT] = True
        updates[AZ_BIT] = False
        return Const((sign << 31) | (0xFF << 23)), updates  # +-infinity
    updates[AV_BIT] = False
    unbiased_exp = biased_exp - 127
    if unbiased_exp > 125:
        updates[AZ_BIT] = True
        return Const(sign << 31), updates  # +-zero
    updates[AZ_BIT] = False
    x = struct.unpack("<f", struct.pack("<I", bits))[0]
    seed_bits, _overflowed = _float32_bits(1.0 / x)
    # "8-bit accurate seed": keep sign, exponent and the top 8 mantissa
    # bits; zero the low 15 mantissa bits this model cannot claim.
    seed_bits &= 0xFFFF8000
    return Const(seed_bits), updates
