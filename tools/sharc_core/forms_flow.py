"""Program flow forms: jumps, calls, returns and hardware loops.

Each handler executes one decoded instruction and returns the successor
states. FORMS maps form names to handlers; sharc_core.forms merges the
family tables.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import List, Optional

from sharc_disasm import Instruction

from .encoding import (
    UREG_CODES,
    _field,
)
from .values import (
    Const,
    Unknown,
    _signed,
)
from .state import (
    Pending,
    State,
    _copy,
    _event,
    _json_value,
    _stop,
    _ureg,
)
from .memory import (
    _dm_read,
)
from .compute import (
    _apply_compute,
    _compute,
)
from .sequencer import (
    _advance,
    _check_return_target,
    _immediate_transfer,
    _predicate,
    _predicate_simd_branch,
    _return_transfer,
    _start_counted_loop,
    _transfer,
)


def _type_12a_imm(
    state: State, insn: Instruction, f: Mapping[str, int], name: str
) -> List[State]:
    """12a_imm."""
    count = (_field(f, "data[15:8]") << 8) | _field(f, "data[7:0]")
    return _start_counted_loop(state, insn, count)


def _type_12a_ureg(
    state: State, insn: Instruction, f: Mapping[str, int], name: str
) -> List[State]:
    """12a_ureg."""
    count = _ureg(state.uregs, _field(f, "ureg"))
    if not isinstance(count, Const):
        return [_stop(state, insn, "nonconcrete Type12a UREG loop count")]
    return _start_counted_loop(state, insn, count.value)


def _type_11c(
    state: State, insn: Instruction, f: Mapping[str, int], name: str
) -> List[State]:
    """11c."""
    if _field(f, "x"):
        return [_stop(state, insn, "unsupported Type11c RTI")]
    if _field(f, "lr"):
        return [_stop(state, insn, "unsupported Type11c loop reentry")]
    return _return_transfer(
        state,
        insn,
        _predicate_simd_branch(state, _field(f, "cond")),
        bool(_field(f, "j")),
    )


def _type_11a(
    state: State, insn: Instruction, f: Mapping[str, int], name: str
) -> List[State]:
    """11a."""
    # PGR "Type 11a ISA/VISA (cond + branch return + comp/else comp)"
    # (out/refs/adsp-2136x_2137x_214xx_pgr_rev2.4/all.txt lines
    # 17818-17862, printed pp.9-44/9-45): IF COND RTS/RTI (DB) (LR),
    # compute / ELSE compute. x selects RTS (0) or RTI (1), the same
    # bit Type11c's own "x" already gates; RTI additionally pops the
    # status/loop stacks and clears IRPTL/IMASKP, none of which this
    # tracer models, so it fails closed exactly like Type11c's RTI
    # check. LR (loop reentry) changes how a loop's PC-stack entry is
    # consumed and is also unmodeled; fail closed rather than guess.
    # j is the (DB) delayed-return modifier, reusing
    # ``_return_transfer``'s existing Type9b/11c-verified delay-slot
    # handling. e selects a plain compute (e=0, runs when the return
    # is taken) or an ELSE compute (e=1, runs only when the return is
    # NOT taken) -- the same "compute unless ELSE" convention Type9a's
    # own "e" bit already implements below (PGR p.9-45: "If a compute
    # operation is specified with the ELSE, it is performed only when
    # the If condition is false").
    if state.pending:
        return [_stop(state, insn, "nested delayed transfer")]
    if _field(f, "x"):
        return [_stop(state, insn, "unsupported Type11a RTI")]
    if _field(f, "lr"):
        return [_stop(state, insn, "unsupported Type11a loop reentry")]
    cond = _field(f, "cond")
    delayed = bool(_field(f, "j"))
    compute_when_taken = not bool(_field(f, "e"))

    def apply_compute(executed: State) -> Optional[str]:
        try:
            compute = _compute(
                f,
                False,
                dict(executed.uregs),
                executed.special,
                approx_recips=executed.approx_recips,
            )
        except ValueError as error:
            return str(error)
        if compute is not None:
            _apply_compute(executed, insn, compute)
        return None

    predicate = _predicate(state, cond)
    if predicate is not None:
        if predicate == compute_when_taken:
            error = apply_compute(state)
            if error:
                return [_stop(state, insn, error)]
        return _return_transfer(state, insn, predicate, delayed)
    taken, not_taken = _copy(state), _copy(state)
    compute_state = taken if compute_when_taken else not_taken
    error = apply_compute(compute_state)
    if error:
        return [_stop(compute_state, insn, error)]
    _event(
        taken, insn, "predicate-assumption", condition=cond, predicate_assumption=True
    )
    _event(
        not_taken,
        insn,
        "predicate-assumption",
        condition=cond,
        predicate_assumption=False,
    )
    return _return_transfer(taken, insn, True, delayed) + _return_transfer(
        not_taken, insn, False, delayed
    )


def _type_9a_abs(
    state: State, insn: Instruction, f: Mapping[str, int], name: str
) -> List[State]:
    """9a_abs."""
    # PRM Type 9a (pp. 14-5, 14-8): JUMP/CALL (Md, Ic) with an optional
    # compute. I pre-modified by M gives the target; I is unchanged.
    if state.pending:
        return [_stop(state, insn, "nested delayed transfer")]
    if _field(f, "a") or _field(f, "ci"):
        return [_stop(state, insn, "unsupported Type9a control modifier")]
    pmi = (_field(f, "pmi[2:2]") << 2) | _field(f, "pmi[1:0]")
    pmm = _field(f, "pmm")
    cond = _field(f, "cond")
    compute_when_taken = not bool(_field(f, "e"))

    def apply_compute(executed: State) -> Optional[str]:
        try:
            compute = _compute(
                f,
                False,
                dict(executed.uregs),
                executed.special,
                approx_recips=executed.approx_recips,
            )
        except ValueError as error:
            return str(error)
        if compute is not None:
            _apply_compute(executed, insn, compute)
        return None

    if (
        _field(f, "b") == 0
        and cond == 0x1F
        and pmi == 4
        and pmm == 6
        and _field(f, "j") == 1
    ):
        # The verified I12/M14 (DB) return idiom of 9b_abs, plus the compute.
        if not state.call_stack:
            return [_stop(state, insn, "return without followed call")]
        mismatch = _check_return_target(state)
        if mismatch:
            return [_stop(state, insn, mismatch)]
        if compute_when_taken:
            error = apply_compute(state)
            if error:
                return [_stop(state, insn, error)]
        _event(state, insn, "return-branch", index="I12", modifier="M14")
        state.steps += 1
        state.pc_sw = state.pc_sw + insn.length_bytes // 2
        state.pending = Pending(None, slots=2, return_from_call=True)
        return [state]
    # Type 9 indirect branches use DAG2: Ic is I8-I15 and Md is M8-M15.
    i_value = _ureg(state.uregs, UREG_CODES["I%d" % (8 + pmi)])
    m_value = _ureg(state.uregs, UREG_CODES["M%d" % (8 + pmm)])
    if not isinstance(i_value, Const) or not isinstance(m_value, Const):
        return [
            _stop(
                state,
                insn,
                "unknown 9a_abs indirect target through I%d/M%d" % (8 + pmi, 8 + pmm),
            )
        ]
    target = (i_value.value + m_value.value) & 0xFFFFFF
    predicate = _predicate(state, cond)
    call = bool(_field(f, "b"))
    transfer = _transfer if _field(f, "j") else _immediate_transfer
    if predicate is not None:
        if predicate == compute_when_taken:
            error = apply_compute(state)
            if error:
                return [_stop(state, insn, error)]
        return transfer(state, insn, target, call, predicate)
    taken, not_taken = _copy(state), _copy(state)
    compute_state = taken if compute_when_taken else not_taken
    error = apply_compute(compute_state)
    if error:
        return [_stop(compute_state, insn, error)]
    _event(
        taken, insn, "predicate-assumption", condition=cond, predicate_assumption=True
    )
    _event(
        not_taken,
        insn,
        "predicate-assumption",
        condition=cond,
        predicate_assumption=False,
    )
    return transfer(taken, insn, target, call, True) + transfer(
        not_taken, insn, target, call, False
    )


# The verified compiler return is a TRUE 9b_abs jump through I12/M14,
# with two delay slots, one of which is the confident 25c_rframe form.
# Do not treat rframe alone, its provisional 48-bit sibling, or another
# register-indirect jump as a return.
def _type_9b_abs(
    state: State, insn: Instruction, f: Mapping[str, int], name: str
) -> List[State]:
    """9b_abs."""
    pmi = (_field(f, "pmi[2:2]") << 2) | _field(f, "pmi[1:0]")
    pmm = _field(f, "pmm")
    if (
        _field(f, "b") == 0
        and _field(f, "cond") == 0x1F
        and pmi == 4
        and pmm == 6
        and _field(f, "j") == 1
    ):
        if not state.call_stack:
            return [_stop(state, insn, "return without followed call")]
        mismatch = _check_return_target(state)
        if mismatch:
            return [_stop(state, insn, mismatch)]
        _event(state, insn, "return-branch", index="I12", modifier="M14")
        state.steps += 1
        state.pc_sw = state.pc_sw + insn.length_bytes // 2
        state.pending = Pending(None, slots=2, return_from_call=True)
        return [state]
    # Any other Type 9b JUMP/CALL (Md, Ic): DAG2 I(8+pmi) + M(8+pmm).
    if state.pending:
        return [_stop(state, insn, "nested delayed transfer")]
    if _field(f, "a") or _field(f, "ci"):
        return [_stop(state, insn, "unsupported Type9b control modifier")]
    i_value = _ureg(state.uregs, UREG_CODES["I%d" % (8 + pmi)])
    m_value = _ureg(state.uregs, UREG_CODES["M%d" % (8 + pmm)])
    if not isinstance(i_value, Const) or not isinstance(m_value, Const):
        return [
            _stop(
                state,
                insn,
                "unknown 9b_abs indirect target through I%d/M%d" % (8 + pmi, 8 + pmm),
            )
        ]
    target = (i_value.value + m_value.value) & 0xFFFFFF
    transfer = _transfer if _field(f, "j") else _immediate_transfer
    return transfer(
        state,
        insn,
        target,
        bool(_field(f, "b")),
        _predicate_simd_branch(state, _field(f, "cond")),
    )


def _type_25c_rframe(
    state: State, insn: Instruction, f: Mapping[str, int], name: str
) -> List[State]:
    """25c_rframe."""
    if state.pending and state.pending.return_from_call:
        frame = _ureg(state.uregs, UREG_CODES["I6"])
        state.uregs[UREG_CODES["I7"]] = frame
        if isinstance(frame, Const):
            restored = _dm_read(state, frame.value, 4)
            if restored is None:
                state.uregs[UREG_CODES["I6"]] = Unknown(
                    "RFRAME load from unavailable memory"
                )
            else:
                state.uregs[UREG_CODES["I6"]] = restored
        else:
            state.uregs[UREG_CODES["I6"]] = Unknown(
                "RFRAME load through nonconcrete I6"
            )
        _event(
            state,
            insn,
            "rframe",
            frame=_json_value(frame),
            restored_i6=_json_value(_ureg(state.uregs, UREG_CODES["I6"])),
        )
        return _advance(state, insn)
    return [_stop(state, insn, "rframe outside verified return delay slots")]


def _type_9a_rel(
    state: State, insn: Instruction, f: Mapping[str, int], name: str
) -> List[State]:
    """9a_rel."""
    if state.pending:
        return [_stop(state, insn, "nested delayed transfer")]
    if _field(f, "a") or _field(f, "ci") or not _field(f, "j"):
        return [_stop(state, insn, "unsupported Type9a control modifier")]
    relative = (_field(f, "reladdr[5:5]") << 5) | _field(f, "reladdr[4:0]")
    target = (state.pc_sw + _signed(relative, 6)) & 0xFFFFFF
    predicate = _predicate(state, _field(f, "cond"))

    def apply_compute(executed: State) -> Optional[str]:
        try:
            compute = _compute(
                f,
                False,
                dict(executed.uregs),
                executed.special,
                approx_recips=executed.approx_recips,
            )
        except ValueError as error:
            return str(error)
        if compute is not None:
            _apply_compute(executed, insn, compute)
        return None

    compute_when_taken = not bool(_field(f, "e"))
    if predicate is not None:
        if predicate == compute_when_taken:
            error = apply_compute(state)
            if error:
                return [_stop(state, insn, error)]
        return _transfer(state, insn, target, bool(_field(f, "b")), predicate)

    taken, not_taken = _copy(state), _copy(state)
    compute_state = taken if compute_when_taken else not_taken
    error = apply_compute(compute_state)
    if error:
        return [_stop(compute_state, insn, error)]
    _event(
        taken,
        insn,
        "predicate-assumption",
        condition=_field(f, "cond"),
        predicate_assumption=True,
    )
    _event(
        not_taken,
        insn,
        "predicate-assumption",
        condition=_field(f, "cond"),
        predicate_assumption=False,
    )
    return _transfer(taken, insn, target, bool(_field(f, "b")), True) + _transfer(
        not_taken, insn, target, bool(_field(f, "b")), False
    )


def _type_25a_direct(
    state: State, insn: Instruction, f: Mapping[str, int], name: str
) -> List[State]:
    """25a_direct, 25a_pcrel, 8a_abs, 8a_rel."""
    if state.pending:
        return [_stop(state, insn, "nested delayed transfer")]
    stem = "addr" if name.endswith("direct") or name.endswith("abs") else "reladdr"
    raw = (_field(f, stem + "[23:16]") << 16) | _field(f, stem + "[15:0]")
    # The sequencer generates 24-bit short-word instruction addresses;
    # reduce a signed PC-relative sum to that architectural width.
    target = (
        raw
        if name.endswith(("direct", "abs"))
        else (state.pc_sw + _signed(raw, 24)) & 0xFFFFFF
    )
    call = name.startswith("25a") or bool(_field(f, "b"))
    if name.startswith("25a"):
        previous_i6 = _ureg(state.uregs, UREG_CODES["I6"])
        new_i6 = _ureg(state.uregs, UREG_CODES["I7"])
        state.uregs[UREG_CODES["R2"]] = previous_i6
        state.uregs[UREG_CODES["I6"]] = new_i6
        _event(
            state,
            insn,
            "cjump-frame",
            saved_i6=_json_value(previous_i6),
            frame=_json_value(new_i6),
        )
    cond = (
        True
        if name.startswith("25a")
        else _predicate_simd_branch(state, _field(f, "cond"))
    )
    delayed = name.startswith("25a") or bool(_field(f, "j"))
    transfer = _transfer if delayed else _immediate_transfer
    return transfer(state, insn, target, call, cond)


FORMS = {
    "12a_imm": _type_12a_imm,
    "12a_ureg": _type_12a_ureg,
    "11c": _type_11c,
    "11a": _type_11a,
    "9a_abs": _type_9a_abs,
    "9b_abs": _type_9b_abs,
    "25c_rframe": _type_25c_rframe,
    "9a_rel": _type_9a_rel,
    "25a_direct": _type_25a_direct,
    "25a_pcrel": _type_25a_direct,
    "8a_abs": _type_25a_direct,
    "8a_rel": _type_25a_direct,
}
