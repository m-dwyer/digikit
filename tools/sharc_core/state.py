"""Machine state, register access and the trace event log.

Moved verbatim from tools/sharc_trace.py.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from sharc_disasm import Instruction
from sharcldr import LoadedMemory

from .encoding import (
    UREG_CODES,
    UREG_NAMES,
)
from .values import (
    Affine,
    Const,
    Operand,
    PartialConst,
    Unknown,
    Value,
    _bitwise,
    _signed32,
)

# 80-bit multiplier-result accumulator (SHARC+ PRM p.3-10, Figure 3-2: MR2F
# is bits 79:64, MR1F is bits 63:32, MR0F is bits 31:0). _MR_MASK is the
# canonical 80-bit unsigned mask MR uses the same way Const uses 0xFFFFFFFF.
_MR_BITS = 80
_MR_MASK = (1 << _MR_BITS) - 1
_MR_WORD_SLICE = {0: (0, 32), 1: (32, 32), 2: (64, 16)}  # word -> (shift, width)


@dataclass(frozen=True)
class MR:
    """A fully or partially known 80-bit two's-complement multiplier-result
    accumulator (REGF_MRF/REGF_MRB, or REGF_MSF/REGF_MSB for PEy).

    Tracks knowledge the same way PartialConst does for ASTATX/ASTATY: MASK
    has a 1 at every known bit position, BITS holds the known value there
    and is canonicalized to 0 elsewhere. A program that only ever moved
    MR0F (PRM Table 18-29 MRDATAMOVE) says nothing about MR1F/MR2F, and this
    lets that partial knowledge survive instead of collapsing the whole
    accumulator to Unknown; ``signed()`` is only ever non-None once every
    bit is known, which is what a multiply/accumulate/round/saturate needs
    (PRM p.3-10: those instructions read/write the full 80-bit field at
    once, never a single MR0/MR1/MR2 word)."""

    mask: int
    bits: int

    def __post_init__(self):
        object.__setattr__(self, "mask", self.mask & _MR_MASK)
        object.__setattr__(self, "bits", self.bits & self.mask)

    @property
    def known(self) -> bool:
        return self.mask == _MR_MASK

    def signed(self) -> int | None:
        """The two's-complement integer value of the full 80-bit field, or
        None unless every bit is known."""
        if not self.known:
            return None
        return (
            self.bits - (1 << _MR_BITS)
            if self.bits & (1 << (_MR_BITS - 1))
            else self.bits
        )


def _mr_from_signed(value: int) -> MR:
    """A fully known MR from a Python int (any width; reduced mod 2**80,
    matching Const's mod-2**32 reduction)."""
    return MR(_MR_MASK, value & _MR_MASK)


MR_ZERO = _mr_from_signed(0)


def _mr_read_word(mr: MR | Operand, word: int) -> Operand:
    """UREG-facing 32-bit value of MR0x/MR1x/MR2x (PRM p.3-11): MR0/MR1 are
    their 32-bit slice verbatim; MR2 sign-extends its 16 stored bits to 32
    ("When data is read from the REGF_MR2F register (guard bits), it is
    sign-extended to 32 bits"). Unknown if MR is not an MR, or that word's
    bits are not all known."""
    shift, width = _MR_WORD_SLICE[word]
    if not isinstance(mr, MR):
        return Unknown("uninitialized MR word %d" % word)
    word_mask = ((1 << width) - 1) << shift
    if (mr.mask & word_mask) != word_mask:
        return Unknown("partially known MR word %d" % word)
    raw = (mr.bits >> shift) & ((1 << width) - 1)
    if word == 2 and raw & (1 << 15):
        raw |= 0xFFFF0000
    return Const(raw)


def _mr_write_word(mr: MR | Operand, word: int, value: Operand) -> MR | Unknown:
    """Write UREG VALUE into MR0x/MR1x/MR2x, returning the updated MR (PRM
    p.3-11): "Data written to the REGF_MR0F register is not sign-extended"
    (word 0, plain 32-bit slice) and "Data written to the REGF_MR1F
    register is sign-extended to REGF_MR2F, repeating the MSB of REGF_MR1F
    in the 16 bits of the REGF_MR2F register" (word 1 also overwrites word
    2); a direct write to MR2F (word 2) only ever touches its own 16 bits.
    A non-Const VALUE clobbers (forgets) the word(s) it would have written,
    rather than leaving stale prior knowledge in place."""
    shift, width = _MR_WORD_SLICE[word]
    old_mask = mr.mask if isinstance(mr, MR) else 0
    old_bits = mr.bits if isinstance(mr, MR) else 0
    word_mask = ((1 << width) - 1) << shift
    mr2_shift, mr2_width = _MR_WORD_SLICE[2]
    mr2_mask = ((1 << mr2_width) - 1) << mr2_shift
    touched_mask = word_mask | (mr2_mask if word == 1 else 0)
    if not isinstance(value, Const):
        new_mask = old_mask & ~touched_mask
        new_bits = old_bits & ~touched_mask
        return (
            MR(new_mask, new_bits)
            if new_mask
            else Unknown("uninitialized MR after unknown write to word %d" % word)
        )
    bits = (value.value & ((1 << width) - 1)) << shift
    new_mask = (old_mask & ~word_mask) | word_mask
    new_bits = (old_bits & ~word_mask) | bits
    if word == 1:
        sign = 0xFFFF if (value.value >> 31) & 1 else 0x0000
        new_mask |= mr2_mask
        new_bits = (new_bits & ~mr2_mask) | (sign << mr2_shift)
    return MR(new_mask, new_bits)


@dataclass(frozen=True)
class Pending:
    # A None target marks the delay slots of a conditional transfer not taken.
    target: int | None
    call: bool = False
    slots: int = 2
    return_from_call: bool = False
    return_sw: int | None = None


# Pending.return_sw placeholder for a delayed call: the return address is the
# PC after the second delay slot, known only once both slots have executed.
AFTER_DELAY_SLOTS = -1


@dataclass(frozen=True)
class Loop:
    start_sw: int
    end_sw: int
    remaining: int
    mode: int


@dataclass
class State:
    pc_sw: int
    uregs: dict[int, Value] = field(default_factory=dict)
    trace: list[dict] = field(default_factory=list)
    pending: Pending | None = None
    steps: int = 0
    stopped: str | None = None
    # Concrete mode is deliberately loader-only.  OVERLAY is per path, so a
    # conditional fork cannot mutate another path or the immutable boot image.
    concrete: LoadedMemory | None = None
    overlay: dict[int, int] = field(default_factory=dict)
    base_sw: int | None = None
    follow_loaded_calls: bool = False
    continue_external_calls: bool = False
    dossier_bytes: int = 0
    max_call_depth: int = 0
    call_stack: list[int] = field(default_factory=list)
    skip_provisional_entries: bool = False
    at_loaded_entry: bool = False
    assume_nw32: bool = False
    loops: list[Loop] = field(default_factory=list)
    status_stack: list[tuple[Value, Value, Value]] = field(default_factory=list)
    core_reset_state: bool = False
    mmrs: dict[int, Value] = field(default_factory=dict)
    data_memory_tainted: bool = False
    # MRF/MRB/MSF/MSB (the 80-bit multiplier accumulators -- see MR above)
    # and other per-instruction-class registers this tracer does not give
    # their own State field (e.g. BFFWRP) share this one dict, keyed by
    # name; every other value here is an ordinary Operand.
    special: dict[str, Operand | MR] = field(default_factory=dict)
    # Forms this run may execute although the table marks them unconfirmed,
    # and the ones it actually did. A state that used any is calibration.
    provisional_forms: tuple[str, ...] = ()
    provisional_used: tuple[str, ...] = ()
    # Opt-in --approx-recips: whether this path may substitute a documented-
    # but-unverified numeric model for recips's undocumented ROM seed, and
    # whether it actually did so at least once (calibration, like above).
    approx_recips: bool = False
    approx_recips_used: bool = False
    # Concrete single-path execution (tools/sharc_run.py) sets this False to
    # skip the per-step trace log. _event() still appends a minimal dict so
    # the few call sites that immediately do trace[-1].update(...)/[...] =
    # (the predicate-resolved Type3a/etc. idiom) keep working; only the
    # pc_sw/form/_json_value bookkeeping is skipped.
    record_events: bool = True
    # Opt-in (tools/sharc_harness.py): a DM read that a real boot/init never
    # wrote reads as 0 (internal RAM only -- see sharc_core/memory.py's
    # _dm_read) instead of Unknown, and a core/system MMR (see
    # sharc_core/encoding.py's CORE_MMR_RANGE/SYSTEM_MMR_RANGE) with no
    # known reset value and no harness-set value raises
    # sharc_core.memory.UnmodeledMMR instead of also going Unknown. Default
    # False: this must not change tools/sharc_run.py's default CLI output
    # or tests/test_sharc_golden.py's hashes.
    explicit_memory_model: bool = False


def _render(value: Value | MR | int) -> str:
    if isinstance(value, MR):
        signed = value.signed()
        if signed is not None:
            return "mr:%#x" % signed
        return "mr:partial(%#x/%#x)" % (value.mask, value.bits)
    if isinstance(value, Const):
        return _render(value.value)
    if isinstance(value, Affine):
        parts: list[tuple[int, str]] = []
        for name, coefficient in value.terms:
            coefficient = _signed32(coefficient)
            magnitude = (
                name if abs(coefficient) == 1 else "%d*%s" % (abs(coefficient), name)
            )
            parts.append((coefficient, magnitude))
        constant = _signed32(value.constant)
        if constant:
            parts.append((constant, hex(abs(constant))))
        if not parts:
            return "0x0"
        first_sign, first = parts[0]
        rendered = ("-" if first_sign < 0 else "") + first
        for sign, magnitude in parts[1:]:
            rendered += (" - " if sign < 0 else " + ") + magnitude
        return rendered
    if isinstance(value, PartialConst):
        return "partial(known=%#010x, bits=%#010x)" % (value.mask, value.bits)
    if isinstance(value, Unknown):
        return value.reason
    return ("-" if value < 0 else "") + hex(abs(value))


def _json_value(value: Value | MR | int) -> int | dict:
    """Render tracer values without leaking internal dataclasses into CLI JSON."""
    if isinstance(value, MR):
        return (
            {"mr": value.signed()}
            if value.known
            else {"mr_partial": {"mask": value.mask, "bits": value.bits}}
        )
    if isinstance(value, Const):
        return value.value
    if isinstance(value, Affine):
        return {
            "affine": {
                "constant": value.constant,
                "terms": [list(term) for term in value.terms],
            }
        }
    if isinstance(value, PartialConst):
        return {"partial": {"known_mask": value.mask, "known_bits": value.bits}}
    if isinstance(value, Unknown):
        return {"unknown": value.reason}
    return value


def _event(state: State, insn: Instruction, action: str, **extra) -> None:
    if not state.record_events:
        # A few call sites (the predicate-resolved Type3a/2a/5a_move/9a_abs
        # idiom) do trace[-1].update(...) or trace[-1][...] = ... right
        # after this call, on the non-forking, always-taken path -- so this
        # must still append one dict, just not the full one.
        state.trace.append({"action": action})
        return
    for key in ("address", "value", "concrete_value"):
        if key in extra:
            extra[key] = _json_value(extra[key])
    state.trace.append(
        {"pc_sw": state.pc_sw, "form": insn.type_name, "action": action, **extra}
    )


def _stop(state: State, insn: Instruction | None, reason: str) -> State:
    form = insn.type_name if insn else None
    state.trace.append(
        {"pc_sw": state.pc_sw, "form": form, "action": "stop", "reason": reason}
    )
    state.stopped = reason
    return state


def _copy(state: State) -> State:
    return State(
        state.pc_sw,
        dict(state.uregs),
        [dict(event) for event in state.trace],
        state.pending,
        state.steps,
        state.stopped,
        state.concrete,
        dict(state.overlay),
        state.base_sw,
        state.follow_loaded_calls,
        state.continue_external_calls,
        state.dossier_bytes,
        state.max_call_depth,
        list(state.call_stack),
        state.skip_provisional_entries,
        state.at_loaded_entry,
        state.assume_nw32,
        list(state.loops),
        list(state.status_stack),
        state.core_reset_state,
        dict(state.mmrs),
        state.data_memory_tainted,
        dict(state.special),
        state.provisional_forms,
        state.provisional_used,
        state.approx_recips,
        state.approx_recips_used,
        state.record_events,
    )


def _ureg_raw(values: Mapping[int, Value], code: int) -> Value:
    """Read UREG CODE exactly as stored, including a PartialConst for
    ASTATX/ASTATY. Only the flag/predicate code that understands
    PartialConst (see the ``_astatx_*`` helpers, ``_apply_compute``,
    ``_predicate``, the Type18a BTF writers, and the status-stack push) may
    call this. Everything else — arithmetic, addressing, memory, UREG
    moves, dossiers — must use ``_ureg``, which never lets a PartialConst
    escape into generic code that only understands Const/Affine/Unknown
    (``_terms``/``_add``/``_negate``/``_multiply``/``_bitwise`` would
    otherwise crash or silently misbehave on one).
    """
    return values.get(code, Unknown("uninitialized " + UREG_NAMES[code]))


def _ureg(values: Mapping[int, Value], code: int) -> Operand:
    """Read UREG CODE as a value any generic consumer can handle.

    A PartialConst (only ever stored at ASTATX/ASTATY) never escapes this
    function: a fully-known one becomes Const, a partially-known one
    becomes Unknown. This is what makes "R0 = ASTATX" (a Type5 UREG move),
    an ASTATX value used as a compute operand or DM address, or a status
    register read by a dossier all safe by construction, without each of
    those call sites needing to know about PartialConst.
    """
    value = _ureg_raw(values, code)
    if isinstance(value, PartialConst):
        return (
            Const(value.bits)
            if value.mask == 0xFFFFFFFF
            else Unknown("partially known ASTATx")
        )
    return value


# SHARC+ Core Programming Reference (out/refs/sharc-plus-prm) p.62 (Table
# 2-3, "Universal and System Register Complementary Pairs") and p.4-55
# footnote *1 ("Complementary universal register pairs (CUreg) ... include
# PEx/y data registers and USTAT1/2, USTAT3/4, ASTATx/y, STKYx/y, and PX1/2
# Uregs"): the UREG codes with a SIMD companion register.  Any code not in
# this map -- every DAG register (I/M/L/B), PC/PCSTK/loop and interrupt
# state, the combined PX, MODE1/MMASK/MODE2/FLAGS, and the timers -- "has
# no complements, so they do not operate differently in SIMD mode" (p.15-3,
# the MODE1/LCNTR example) and this tracer's single-PE handling of them is
# already correct in SIMD mode as well as SISD.
_CUREG_PAIRS: dict[int, int] = {code: code + 80 for code in range(16)}
_CUREG_PAIRS.update({code + 80: code for code in range(16)})
for _pair in (
    ("USTAT1", "USTAT2"),
    ("USTAT3", "USTAT4"),
    ("PX1", "PX2"),
    ("ASTATX", "ASTATY"),
    ("STKYX", "STKYY"),
):
    _CUREG_PAIRS[UREG_CODES[_pair[0]]] = UREG_CODES[_pair[1]]
    _CUREG_PAIRS[UREG_CODES[_pair[1]]] = UREG_CODES[_pair[0]]
del _pair


def _cureg_code(code: int) -> int | None:
    """The SIMD companion (Cureg) UREG code for CODE, or None if CODE has
    no SIMD complement (see _CUREG_PAIRS)."""
    return _CUREG_PAIRS.get(code)


def _simd_active(state: State) -> bool | None:
    """MODE1.PEYEN (bit 21, SHARC+ PRM p.101): True/False when MODE1 is
    concretely known, else None."""
    mode1 = _ureg(state.uregs, UREG_CODES["MODE1"])
    if not isinstance(mode1, Const):
        return None
    return bool(mode1.value & (1 << 21))


def _sync_pc_stack(state: State) -> None:
    """Mirror the tracer's architectural PC stack into its public registers."""
    state.uregs[UREG_CODES["PCSTKP"]] = Const(len(state.call_stack))
    state.uregs[UREG_CODES["PCSTK"]] = (
        Const(state.call_stack[-1]) if state.call_stack else Const(0x7FFFFFFF)
    )
    stkyx_code = UREG_CODES["STKYX"]
    state.uregs[stkyx_code] = _bitwise(
        _ureg(state.uregs, stkyx_code),
        Const(1 << 22),
        "PC stack empty" if not state.call_stack else "PC stack nonempty",
        (lambda value, mask: value | mask)
        if not state.call_stack
        else (lambda value, mask: value & ~mask),
    )
