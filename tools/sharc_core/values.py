"""Concrete and symbolic values and the integer arithmetic over them.

Moved verbatim from tools/sharc_trace.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Union


@dataclass(frozen=True)
class Const:
    value: int

    def __post_init__(self):
        object.__setattr__(self, "value", self.value & 0xFFFFFFFF)


@dataclass(frozen=True)
class Affine:
    """A canonical 32-bit affine expression, constant plus named terms."""

    constant: int
    terms: tuple[tuple[str, int], ...]

    def __post_init__(self):
        coefficients: dict[str, int] = {}
        for name, coefficient in self.terms:
            if not _SYMBOL_RE.fullmatch(name):
                raise ValueError("invalid symbol name: " + repr(name))
            coefficients[name] = (coefficients.get(name, 0) + coefficient) & 0xFFFFFFFF
        object.__setattr__(self, "constant", self.constant & 0xFFFFFFFF)
        object.__setattr__(
            self,
            "terms",
            tuple(
                sorted(
                    (name, coefficient)
                    for name, coefficient in coefficients.items()
                    if coefficient
                )
            ),
        )


@dataclass(frozen=True)
class Unknown:
    reason: str


@dataclass(frozen=True)
class PartialConst:
    """A 32-bit value known only at some bit positions.

    Used for ASTATX/ASTATY: different instruction classes each define a
    disjoint group of bits (ALU flags, shifter flags, multiplier flags, BTF,
    CACC), so full 32-bit knowledge is rare in practice, but bit-level
    knowledge is common and is all the condition predicates ever need (each
    reads at most a handful of specific bits). ``mask`` has a 1 at every
    known bit position; ``bits`` holds the known value at those positions and
    is canonicalized to 0 elsewhere so two PartialConst values with the same
    knowledge compare and hash equal regardless of what an unknown position
    happened to hold before.
    """

    mask: int
    bits: int

    def __post_init__(self):
        object.__setattr__(self, "mask", self.mask & 0xFFFFFFFF)
        object.__setattr__(self, "bits", self.bits & self.mask)


Value = Union[Const, Affine, Unknown, PartialConst]
_SYMBOL_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _affine(constant: int, terms: tuple[tuple[str, int], ...]) -> Const | Affine:
    """Build a canonical affine value, collapsing a constant expression."""
    value = Affine(constant, terms)
    return Const(value.constant) if not value.terms else value


def symbol(name: str) -> Affine:
    """Return the named symbolic value NAME."""
    if not _SYMBOL_RE.fullmatch(name):
        raise ValueError("invalid symbol name: " + repr(name))
    return Affine(0, ((name, 1),))


def _signed(value: int, bits: int) -> int:
    return value - (1 << bits) if value & (1 << (bits - 1)) else value


def _signed32(value: int) -> int:
    return _signed(value & 0xFFFFFFFF, 32)


def _terms(value: Const | Affine) -> tuple[int, tuple[tuple[str, int], ...]]:
    return (
        (value.value, ()) if isinstance(value, Const) else (value.constant, value.terms)
    )


# A symbol name this module recognises as denoting a value some caller has
# already bounded to a known numeric range: either an entry-time seed in
# the convention tools/sharcwriters.py's seed_sets()/ENTRY_SEED_NAMES uses
# ("I6e", "B7e", ...: one or more uppercase letters, one or more digits,
# then "e"), or one of this module's own CIRC_SYMBOL_PREFIX-tagged symbols
# (below). This module does not itself know the numeric bound -- that is
# the caller's fact to state (tools/sharcwriters.py's STACK_SYMBOLS /
# CIRC_WRAP_SLACK) -- it only recognises the *shape* of a name a caller is
# likely to have bounded, so it knows when re-deriving a fresh symbol
# through a circular MODIFY is meaningful rather than fabricating a bound
# for an arbitrary, unrelated value that merely happens to be a bare named
# term (e.g. a loop-count symbol).
_BOUNDED_SYMBOL_RE = re.compile(r"^[A-Z]+\d+e$")
CIRC_SYMBOL_PREFIX = "circ_"


def _stack_bounded_symbol(value: Value) -> Optional[tuple[str, int]]:
    """-> (name, signed constant offset), if `value` is exactly one named
    symbol with coefficient 1 (any constant offset) whose name matches
    _BOUNDED_SYMBOL_RE or starts with CIRC_SYMBOL_PREFIX -- otherwise None.
    A second term, or a coefficient other than 1, means the value's range
    is no longer provably tied to the symbol's own bound (e.g. a scaled or
    summed expression), so the caller falls back to Unknown rather than
    guess."""
    if isinstance(value, Affine) and len(value.terms) == 1:
        name, coefficient = value.terms[0]
        if coefficient == 1 and (
            _BOUNDED_SYMBOL_RE.match(name) or name.startswith(CIRC_SYMBOL_PREFIX)
        ):
            return name, _signed(value.constant, 32)
    return None


def _add(left: Value, right: Value, expression: str) -> Value:
    if isinstance(left, Unknown) or isinstance(right, Unknown):
        return Unknown(expression)
    constant, terms = _terms(left)
    other_constant, other_terms = _terms(right)
    return _affine(constant + other_constant, terms + other_terms)


def _negate(value: Value, expression: str) -> Value:
    if isinstance(value, Unknown):
        return Unknown(expression)
    constant, terms = _terms(value)
    return _affine(
        -constant, tuple((name, -coefficient) for name, coefficient in terms)
    )


def _subtract(
    left: Value, right: Value, expression: str, *, same_source: bool = False
) -> Value:
    """LEFT - RIGHT, with an explicit fold for the self-subtract idiom.

    SAME_SOURCE=True is the caller's promise that LEFT and RIGHT are two
    reads of the exact same register/operand at this instant (e.g. the
    SHARC+ "Rn = Rn - Rn" self-clear idiom, PRM Table 17-5 / 18-10 ALUOP
    add/subtract with RX=RY): whatever that shared value is -- even an
    Unknown/symbolic one -- X - X is exactly 0 in 32-bit modular
    arithmetic, so fold to Const(0) directly rather than letting an
    Unknown operand swallow the whole expression (Unknown - Unknown would
    otherwise stay Unknown forever, e.g. a subsequent DO-loop compare
    against it never resolving concretely and forking every iteration)."""
    if same_source:
        return Const(0)
    return _add(left, _negate(right, expression), expression)


def _multiply(left: Value, right: Value, expression: str) -> Value:
    if isinstance(left, Unknown) or isinstance(right, Unknown):
        return Unknown(expression)
    if isinstance(left, Const) and isinstance(right, Const):
        return Const(left.value * right.value)
    if isinstance(left, Const):
        constant, terms = _terms(right)
        return _affine(
            left.value * constant,
            tuple((name, left.value * coefficient) for name, coefficient in terms),
        )
    if isinstance(right, Const):
        constant, terms = _terms(left)
        return _affine(
            right.value * constant,
            tuple((name, right.value * coefficient) for name, coefficient in terms),
        )
    return Unknown(expression + " (non-affine multiplication)")


def _multiply_fractional(
    left: Value, right: Value, signed_x: bool, signed_y: bool, expression: str
) -> Value:
    """RX * RY MOD1 in 1.31/0.32 fractional format (PRM "Fixed-Point
    Formats", p.27-3/27-4): a 32-bit fractional operand's value is its raw
    bit pattern scaled by 2**-31 (signed) or 2**-32 (unsigned), so the
    64-bit product is scaled by 2**-62/2**-63/2**-64 depending on operand
    signs. The register-file/MRF result keeps the top 32 bits of that
    product (PRM Figure 3-2, p.3-10: "bits 63-0 for a fractional result").
    When both inputs are signed, PRM p.3-9 documents an extra left shift by
    one to remove the redundant sign bit before that truncation, which
    folds into dividing by 2**31 instead of 2**32 below. Only the doubly
    Const case is evaluated; anything else (Unknown, or a still-symbolic
    Affine, which this shift does not distribute over) stays Unknown.
    """
    if not (isinstance(left, Const) and isinstance(right, Const)):
        return Unknown(expression)
    x = _signed32(left.value) if signed_x else left.value
    y = _signed32(right.value) if signed_y else right.value
    shift = 31 if (signed_x and signed_y) else 32
    return Const((x * y) >> shift)


def _aconv_symbol(value: Affine, direction: str, source_code: int, pc_sw: int) -> Affine:
    """Return an opaque, stable symbolic result for map-dependent ACONV.

    B2W is not affine when the source's low two bits are unknown.  The PRM's
    address-map/ILAD exception also prevents treating a symbolic source as an
    unconditional shift.  Retaining a source-derived opaque symbol lets the
    bounded writer tracer continue without asserting a false linear relation.
    """
    pieces = [direction, str(source_code), "%x" % pc_sw, "%x" % value.constant]
    pieces.extend("%s_%x" % (name, coefficient) for name, coefficient in value.terms)
    return symbol("aconv_" + "_".join(pieces))


def _aconv(value: Value, w2b: bool, source_code: int, pc_sw: int) -> Value:
    """Apply the PRM-likely ACONV arithmetic without inventing ILAD behavior."""
    if isinstance(value, Const):
        return Const(value.value << 2 if w2b else value.value >> 2)
    if not isinstance(value, Affine):
        return Unknown("ACONV source is not symbolic")
    if w2b:
        return _multiply(value, Const(4), "ACONV W2B")
    if value.constant % 4 == 0 and all(coefficient % 4 == 0 for _, coefficient in value.terms):
        return _affine(
            value.constant // 4,
            tuple((name, coefficient // 4) for name, coefficient in value.terms),
        )
    return _aconv_symbol(value, "b2w", source_code, pc_sw)


def _bitwise(left: Value, right: Value, expression: str, operation) -> Value:
    if isinstance(left, Const) and isinstance(right, Const):
        return Const(operation(left.value, right.value))
    return Unknown(expression)


def _not(value: Value, expression: str) -> Value:
    return Const(~value.value) if isinstance(value, Const) else Unknown(expression)


def _astatx_known_bit(value: Value, bit: int) -> Optional[bool]:
    """Return ASTATX/ASTATY bit BIT if known, else None."""
    if isinstance(value, Const):
        return bool(value.value & (1 << bit))
    if isinstance(value, PartialConst):
        if value.mask & (1 << bit):
            return bool(value.bits & (1 << bit))
        return None
    return None
