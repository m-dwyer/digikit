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
    special: dict[str, Operand] = field(default_factory=dict)
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


def _render(value: Value | int) -> str:
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


def _json_value(value: Value | int) -> int | dict:
    """Render tracer values without leaking internal dataclasses into CLI JSON."""
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
