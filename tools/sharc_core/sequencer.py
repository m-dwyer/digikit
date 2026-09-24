"""Decode, program flow, conditions, delayed branches, calls, returns and loops.

Moved verbatim from tools/sharc_trace.py.
"""

from __future__ import annotations

from sharc_disasm import Instruction, decode_loaded_at, disassemble
from sharcldr import LoadedMemory

from .encoding import (
    AF_BIT,
    ALUSAT_BIT,
    AN_BIT,
    AV_BIT,
    AZ_BIT,
    SIMPLE_COND_BITS,
    UREG_CODES,
    _field,
)
from .memory import (
    _dossier,
)
from .state import (
    AFTER_DELAY_SLOTS,
    Loop,
    Pending,
    State,
    _copy,
    _event,
    _simd_active,
    _stop,
    _sync_pc_stack,
    _ureg,
    _ureg_raw,
)
from .values import (
    Const,
    Unknown,
    _astatx_known_bit,
    _bitwise,
    _signed,
)


def decode_at(
    data: bytes | LoadedMemory, base_sw: int | None, pc_sw: int
) -> Instruction:
    """Decode exactly at PC_SW from a flat image or loader-backed memory."""
    if isinstance(data, LoadedMemory):
        return decode_loaded_at(data, pc_sw)
    if base_sw is None:
        raise ValueError("base_sw is required for flat image decoding")
    offset = (pc_sw - base_sw) * 2
    if offset < 0 or offset >= len(data):
        return Instruction(
            offset, None, "unknown", kind="unknown", note="PC outside image"
        )
    return next(disassemble(data, start_offset=offset, count=1))


def _advance(state: State, insn: Instruction) -> list[State]:
    state.steps += 1
    if insn.length_bytes is None:
        raise ValueError("cannot advance an instruction without a decoded length")
    next_pc = state.pc_sw + insn.length_bytes // 2
    if state.pending is None:
        if state.loops and state.pc_sw == state.loops[-1].end_sw:
            loop = state.loops[-1]
            remaining = loop.remaining - 1
            state.uregs[UREG_CODES["CURLCNTR"]] = Const(max(remaining, 0))
            if remaining > 0:
                _event(
                    state,
                    insn,
                    "loop-back",
                    target_sw=loop.start_sw,
                    remaining=remaining,
                    mode=loop.mode,
                )
                state.loops[-1] = Loop(loop.start_sw, loop.end_sw, remaining, loop.mode)
                state.pc_sw = loop.start_sw
                return [state]
            _event(state, insn, "loop-exit", remaining=0, mode=loop.mode)
            state.loops.pop()
            if not state.call_stack or state.call_stack[-1] != loop.start_sw:
                return [_stop(state, insn, "loop PC-stack mismatch")]
            state.call_stack.pop()
            _sync_pc_stack(state)
            state.uregs[UREG_CODES["CURLCNTR"]] = (
                Const(state.loops[-1].remaining) if state.loops else Const(0xFFFFFFFF)
            )
            if not state.loops:
                stkyx_code = UREG_CODES["STKYX"]
                state.uregs[stkyx_code] = _bitwise(
                    _ureg(state.uregs, stkyx_code),
                    Const(1 << 26),
                    "loop stacks empty",
                    lambda a, b: a | b,
                )
        state.pc_sw = next_pc
        return [state]
    p = state.pending
    if p.slots == 1:
        if p.return_from_call:
            if not state.call_stack:
                return [_stop(state, insn, "return without followed call")]
            if state.loops and state.call_stack[-1] == state.loops[-1].start_sw:
                return [_stop(state, insn, "return reached loop PC-stack entry")]
            state.pc_sw = state.call_stack.pop()
            _sync_pc_stack(state)
            state.pending = None
            _event(state, insn, "loaded-call-return", return_sw=state.pc_sw)
            return [state]
        if p.call:
            if p.target is None:
                return [_stop(state, insn, "call without target")]
            target = p.target
            return_sw = next_pc if p.return_sw == AFTER_DELAY_SLOTS else p.return_sw
            loaded = False
            if (
                state.follow_loaded_calls
                and state.concrete is not None
                and target is not None
                and target >= 0
            ):
                decoded = decode_at(state.concrete, None, target)
                loaded = decoded.kind != "unknown"
            if loaded:
                followed_depth = len(state.call_stack) - len(state.loops)
                if followed_depth >= state.max_call_depth:
                    return [_stop(state, insn, "max-call-depth")]
                if return_sw is None:
                    return [_stop(state, insn, "call without architectural return")]
                state.call_stack.append(return_sw)
                _sync_pc_stack(state)
                state.pending = None
                state.pc_sw = target
                state.at_loaded_entry = True
                _event(
                    state,
                    insn,
                    "loaded-call-enter",
                    target_sw=target,
                    return_sw=return_sw,
                )
                return [state]
            if return_sw is None:
                return [_stop(state, insn, "call without architectural return")]
            dossier = _dossier(state, target, return_sw)
            if not state.continue_external_calls:
                _stop(state, insn, "external-call")
                # The default endpoint remains the historical stop event; its
                # dossier explicitly labels the otherwise opaque boundary.
                state.trace[-1].update(dossier)
                state.trace[-1]["opaque_external_call"] = True
                return [state]
            _event(state, insn, "opaque-external-call", **dossier)
            # Conservative ABI boundary: results can be clobbered; memory and
            # pointer arguments are deliberately untouched.
            for code in range(16):
                state.uregs[code] = Unknown("opaque-external-call result")
            state.special["MRF"] = Unknown("opaque-external-call result")
            state.pending = None
            state.pc_sw = return_sw
            _event(
                state,
                insn,
                "external-call-continue",
                clobbered=["R%d" % n for n in range(16)] + ["MRF"],
            )
            return [state]
        state.pending = None
        state.pc_sw = next_pc if p.target is None else p.target
        return [state]
    state.pending = Pending(
        p.target, p.call, p.slots - 1, p.return_from_call, p.return_sw
    )
    state.pc_sw = next_pc
    return [state]


def _lt_ge_le_gt(state: State, cond: int) -> bool | None:
    """PGR Table 4-37 (p.4-93) / PRM p.4-53:

    X = (NOT AF AND (AN XOR (AV AND NOT ALUSAT))) OR (AF AND AN) OR AZ
    LE iff X, GT iff NOT X.
    Y = (NOT AF AND (AN XOR (AV AND NOT ALUSAT))) OR (AF AND AN AND NOT AZ)
    LT iff Y, GE iff NOT Y.

    (At AF=0 this is X = Y OR AZ, i.e. LE = LT OR EQ, matching intuition.)
    ALUSAT is only read when it would actually change the answer (AF=0 and
    AV=1); this lets a comparison that clearly did not overflow resolve
    without needing MODE1 to be known.
    """
    astatx = _ureg_raw(state.uregs, UREG_CODES["ASTATX"])
    af = _astatx_known_bit(astatx, AF_BIT)
    an = _astatx_known_bit(astatx, AN_BIT)
    az = _astatx_known_bit(astatx, AZ_BIT)
    if af is None or an is None or az is None:
        return None
    if af:
        x = an or az
        y = an and not az
    else:
        av = _astatx_known_bit(astatx, AV_BIT)
        if av is None:
            return None
        if not av:
            term = an  # AN xor (AV and not ALUSAT), with AV=0
        else:
            mode1 = _ureg(state.uregs, UREG_CODES["MODE1"])
            if not isinstance(mode1, Const):
                return None
            alusat = bool(mode1.value & (1 << ALUSAT_BIT))
            term = an != (not alusat)  # AN xor (True and not ALUSAT)
        x = term or az
        y = term
    if cond in (0x02, 0x12):  # LE / GT
        return x if cond == 0x02 else not x
    return y if cond == 0x01 else not y  # LT / GE


def _predicate(state: State, cond: int) -> bool | None:
    if cond == 0x1F:
        return True
    if cond in (0x00, 0x10):
        # Conditional branches in SIMD mode combine the PEx/PEy conditions.
        # The tracer does not yet model the companion PASS, so only consume
        # AZ when execution is concretely SISD.
        mode1 = _ureg(state.uregs, UREG_CODES["MODE1"])
        astatx = _ureg_raw(state.uregs, UREG_CODES["ASTATX"])
        equal = _astatx_known_bit(astatx, AZ_BIT)
        if not isinstance(mode1, Const) or mode1.value & (1 << 21) or equal is None:
            return None
        return equal if cond == 0x00 else not equal
    if cond in (0x01, 0x02, 0x11, 0x12):
        return _lt_ge_le_gt(state, cond)
    if cond in SIMPLE_COND_BITS:
        bit, negate = SIMPLE_COND_BITS[cond]
        astatx = _ureg_raw(state.uregs, UREG_CODES["ASTATX"])
        known = _astatx_known_bit(astatx, bit)
        if known is None:
            return None
        return (not known) if negate else known
    return None


def _lt_ge_le_gt_pe(state: State, cond: int, pe: str) -> bool | None:
    """_lt_ge_le_gt read against one PE's own status (SHARC+ PRM p.4-53's
    rule, applied to REGF_ASTATY when pe == "y" instead of REGF_ASTATX --
    p.66 Table 3-1 pairs them as the identical per-PE status)."""
    astat_code = UREG_CODES["ASTATX"] if pe == "x" else UREG_CODES["ASTATY"]
    astat = _ureg_raw(state.uregs, astat_code)
    af = _astatx_known_bit(astat, AF_BIT)
    an = _astatx_known_bit(astat, AN_BIT)
    az = _astatx_known_bit(astat, AZ_BIT)
    if af is None or an is None or az is None:
        return None
    if af:
        x = an or az
        y = an and not az
    else:
        av = _astatx_known_bit(astat, AV_BIT)
        if av is None:
            return None
        if not av:
            term = an
        else:
            mode1 = _ureg(state.uregs, UREG_CODES["MODE1"])
            if not isinstance(mode1, Const):
                return None
            alusat = bool(mode1.value & (1 << ALUSAT_BIT))
            term = an != (not alusat)
        x = term or az
        y = term
    if cond in (0x02, 0x12):
        return x if cond == 0x02 else not x
    return y if cond == 0x01 else not y


def _predicate_pe(state: State, cond: int, pe: str) -> bool | None:
    """Evaluate COND against exactly one processing element's own status
    (SHARC+ PRM p.4-54, Table 4-22: a conditional compute or register/
    memory move "[e]xecutes ... depending on condition test in each PE").
    Unlike _predicate, this never bails out because SIMD mode is active or
    unresolved -- reading a single PE's own condition is exactly what SIMD
    mode calls for, and callers that need the combined branch condition use
    _predicate_simd_branch instead."""
    if cond == 0x1F:
        return True
    if cond in (0x00, 0x10):
        astat_code = UREG_CODES["ASTATX"] if pe == "x" else UREG_CODES["ASTATY"]
        astat = _ureg_raw(state.uregs, astat_code)
        equal = _astatx_known_bit(astat, AZ_BIT)
        if equal is None:
            return None
        return equal if cond == 0x00 else not equal
    if cond in (0x01, 0x02, 0x11, 0x12):
        return _lt_ge_le_gt_pe(state, cond, pe)
    if cond in SIMPLE_COND_BITS:
        bit, negate = SIMPLE_COND_BITS[cond]
        astat_code = UREG_CODES["ASTATX"] if pe == "x" else UREG_CODES["ASTATY"]
        astat = _ureg_raw(state.uregs, astat_code)
        known = _astatx_known_bit(astat, bit)
        if known is None:
            return None
        return (not known) if negate else known
    return None


def _predicate_and(a: bool | None, b: bool | None) -> bool | None:
    """Three-valued AND, used to combine PEx's and PEy's conditions for a
    SIMD branch (SHARC+ PRM p.4-54): a concrete False on either side makes
    the whole AND False even if the other side is unresolved; otherwise an
    unresolved side makes the result unresolved."""
    if a is False or b is False:
        return False
    if a is None or b is None:
        return None
    return a and b


def _predicate_simd_branch(state: State, cond: int) -> bool | None:
    """A branch/call/return's predicate (SHARC+ PRM p.4-54, Table 4-22:
    "Executes in sequencer depending on AND'ing condition test on both
    PEs"). SISD mode uses PEx's own condition only; SIMD mode ANDs PEx's
    and PEy's. An unresolved MODE1.PEYEN leaves the choice between those
    two rules unresolved too, except for the unconditional ("always")
    branch, which needs neither PE's status."""
    if cond == 0x1F:
        return True
    simd = _simd_active(state)
    if simd is None:
        return None
    pex = _predicate_pe(state, cond, "x")
    if not simd:
        return pex
    pey = _predicate_pe(state, cond, "y")
    return _predicate_and(pex, pey)


def _check_return_target(state: State) -> str | None:
    """The firmware returns through JUMP (M14, I12) (DB). When both registers
    are known, the jump target must equal the recorded return address."""
    index = _ureg(state.uregs, UREG_CODES["I12"])
    modifier = _ureg(state.uregs, UREG_CODES["M14"])
    if not isinstance(index, Const) or not isinstance(modifier, Const):
        return None
    target = (index.value + modifier.value) & 0xFFFFFF
    if target != state.call_stack[-1]:
        return "return target %#x differs from recorded return %#x" % (
            target,
            state.call_stack[-1],
        )
    return None


def _transfer(
    state: State, insn: Instruction, target: int, call: bool, cond: bool | None
) -> list[State]:
    if state.pending:
        return [_stop(state, insn, "nested delayed transfer")]
    if insn.length_bytes is None:
        raise ValueError("cannot transfer from an instruction without a decoded length")
    fall = state.pc_sw + insn.length_bytes // 2
    _event(state, insn, "call" if call else "branch", target_sw=target, predicate=cond)
    state.steps += 1
    if cond is False:
        state.pc_sw = fall
        return [state]
    # A delayed CALL returns to the instruction after its second delay slot.
    # The firmware's CJUMP idiom stores that address - 1 in the second slot, so
    # the short-word offset depends on the slot widths (7 after a 16-bit push,
    # 9 after a 48-bit one). Resolve it when the slots complete.
    return_sw = AFTER_DELAY_SLOTS if call else None
    if cond is True:
        state.pc_sw, state.pending = fall, Pending(target, call, return_sw=return_sw)
        return [state]
    taken, not_taken = _copy(state), _copy(state)
    taken.pc_sw, taken.pending = fall, Pending(target, call, return_sw=return_sw)
    not_taken.pc_sw, not_taken.pending = fall, Pending(None)
    not_taken.trace[-1]["action"] = "branch-not-taken"
    return [taken, not_taken]


def _immediate_transfer(
    state: State, insn: Instruction, target: int, call: bool, cond: bool | None
) -> list[State]:
    """Execute a Type 8 transfer without the instruction's DB modifier."""
    if state.pending:
        return [_stop(state, insn, "nested delayed transfer")]
    if insn.length_bytes is None:
        raise ValueError("cannot transfer from an instruction without a decoded length")
    fall = state.pc_sw + insn.length_bytes // 2
    _event(state, insn, "call" if call else "branch", target_sw=target, predicate=cond)
    if cond is False:
        return _advance(state, insn)
    if cond is True:
        state.pending = Pending(target, call, slots=1, return_sw=fall if call else None)
        return _advance(state, insn)
    taken, not_taken = _copy(state), _copy(state)
    taken.pending = Pending(target, call, slots=1, return_sw=fall if call else None)
    not_taken.trace[-1]["action"] = "branch-not-taken"
    return _advance(taken, insn) + _advance(not_taken, insn)


def _return_transfer(
    state: State, insn: Instruction, predicate: bool | None, delayed: bool
) -> list[State]:
    """Execute a documented RTS against the tracer's followed-call stack."""
    if state.pending:
        return [_stop(state, insn, "nested delayed transfer")]
    if insn.length_bytes is None:
        raise ValueError("cannot return from an instruction without a decoded length")
    length_bytes = insn.length_bytes
    _event(state, insn, "return", predicate=predicate, delayed=delayed)
    if predicate is False:
        state.trace[-1]["action"] = "return-not-taken"
        return _advance(state, insn)

    def take_return(taken: State) -> list[State]:
        if not taken.call_stack:
            return [_stop(taken, insn, "return without followed call")]
        if taken.loops and taken.call_stack[-1] == taken.loops[-1].start_sw:
            return [_stop(taken, insn, "return reached loop PC-stack entry")]
        if delayed:
            taken.steps += 1
            taken.pc_sw += length_bytes // 2
            taken.pending = Pending(None, slots=2, return_from_call=True)
        else:
            taken.steps += 1
            taken.pc_sw = taken.call_stack.pop()
            _sync_pc_stack(taken)
            _event(taken, insn, "loaded-call-return", return_sw=taken.pc_sw)
        return [taken]

    if predicate is True:
        return take_return(state)
    taken, not_taken = _copy(state), _copy(state)
    not_taken.trace[-1]["action"] = "return-not-taken"
    return take_return(taken) + _advance(not_taken, insn)


def _start_counted_loop(state: State, insn: Instruction, count: int) -> list[State]:
    if count == 0:
        return [_stop(state, insn, "unsupported zero-count Type12a loop")]
    reladdr = (_field(insn.fields, "reladdr[22:16]") << 16) | _field(
        insn.fields, "reladdr[15:0]"
    )
    if insn.length_bytes is None:
        raise ValueError("cannot start a loop from an instruction without a length")
    end_sw = state.pc_sw + _signed(reladdr, 23)
    start_sw = state.pc_sw + insn.length_bytes // 2
    mode = _field(insn.fields, "mode")
    state.uregs[UREG_CODES["LCNTR"]] = Const(count)
    state.uregs[UREG_CODES["CURLCNTR"]] = Const(count)
    stkyx_code = UREG_CODES["STKYX"]
    state.uregs[stkyx_code] = _bitwise(
        _ureg(state.uregs, stkyx_code),
        Const(1 << 26),
        "loop stacks nonempty",
        lambda a, b: a & ~b,
    )
    state.loops.append(Loop(start_sw, end_sw, count, mode))
    state.call_stack.append(start_sw)
    _sync_pc_stack(state)
    _event(
        state,
        insn,
        "loop-setup",
        start_sw=start_sw,
        end_sw=end_sw,
        count=count,
        mode=mode,
    )
    return _advance(state, insn)
