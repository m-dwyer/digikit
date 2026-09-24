"""Per-form instruction execution: the _execute dispatch.

Moved verbatim from tools/sharc_trace.py.
"""

from __future__ import annotations

from typing import List, Optional

from sharc_disasm import Instruction

from .encoding import (
    ACCESS_WIDTHS,
    BTF_BIT,
    UREG_CODES,
    UREG_NAMES,
    _field,
    _wide,
)
from .values import (
    Affine,
    CIRC_SYMBOL_PREFIX,
    Const,
    PartialConst,
    Unknown,
    _aconv,
    _add,
    _bitwise,
    _multiply,
    _signed,
    _stack_bounded_symbol,
)
from .state import (
    Pending,
    State,
    _copy,
    _cureg_code,
    _event,
    _json_value,
    _render,
    _simd_active,
    _stop,
    _sync_pc_stack,
    _ureg,
    _ureg_raw,
)
from .memory import (
    _access_modifier_scale,
    _circular_wrap_const,
    _dm_read,
    _dm_write,
    _load_normal_ureg,
    _simd_ureg_mem_companion,
)
from .flags import (
    _astatx_define,
    _astatx_forget,
)
from .compute import (
    _apply_compute,
    _apply_compute_simd,
    _compute,
    _compute_simd,
    _shift_immediate,
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


def _execute(state: State, insn: Instruction) -> List[State]:
    if insn.kind != "confident" or insn.length_bytes is None:
        if insn.length_bytes is None or insn.type_name not in state.provisional_forms:
            return [_stop(state, insn, "uncertain or undecodable form: " + insn.note)]
        if insn.type_name not in state.provisional_used:
            state.provisional_used = tuple(
                sorted(set(state.provisional_used) | {insn.type_name})
            )
    state.at_loaded_entry = False
    f, name = insn.fields, insn.type_name
    if name in ("21a", "21c"):
        return _advance(state, insn)
    if name == "6b_shiftimm":
        if _field(f, "cond") != 0x1F:
            return [_stop(state, insn, "unsupported Type6b predicate")]
        try:
            result = _shift_immediate(f, dict(state.uregs), state.special)
        except ValueError as error:
            return [_stop(state, insn, str(error))]
        _apply_compute(state, insn, result)
        return _advance(state, insn)
    if name == "6a_mem":
        # PRM Type 6a performs a ShiftImm and a normal-word memory transfer
        # in parallel, then post-modifies the selected I register by M.
        if _field(f, "cond") != 0x1F:
            return [_stop(state, insn, "unsupported Type6a predicate")]
        old = dict(state.uregs)
        try:
            result = _shift_immediate(f, old, state.special)
        except ValueError as error:
            return [_stop(state, insn, str(error))]
        bank = 8 if _field(f, "g") else 0
        index, modifier = _field(f, "i") + bank, _field(f, "m") + bank
        iv, mv = _ureg(old, 16 + index), _ureg(old, 32 + modifier)
        scale = _access_modifier_scale("normal-word", state.assume_nw32)
        scaled_mv = _multiply(mv, Const(scale), "M%d * %d" % (modifier, scale))
        space = "PM" if bank else "DM"
        dreg = _field(f, "dreg")
        if _field(f, "d"):
            value = _ureg(old, dreg)
            _event(
                state,
                insn,
                "store",
                space=space,
                dreg="R%d" % dreg,
                value=value,
                address=iv,
                expression=_render(iv),
                concrete_write=_dm_write(state, iv, 4, value)
                if space == "DM"
                else False,
                addressing_mode="post-modify",
                access_width="normal-word",
            )
        else:
            loaded = _load_normal_ureg(state, space, iv, dreg)
            _event(
                state,
                insn,
                "load",
                space=space,
                dreg="R%d" % dreg,
                address=iv,
                expression=_render(iv),
                concrete_value=loaded,
                addressing_mode="post-modify",
                access_width="normal-word",
            )
        state.uregs[16 + index] = _add(
            iv, scaled_mv, "I%d + M%d * %d" % (index, modifier, scale)
        )
        _apply_compute(state, insn, result)
        return _advance(state, insn)
    if name == "18a":
        bop = _field(f, "bop")
        sreg = _field(f, "sreg")
        if bop in (4, 5):
            operation = "bit-test" if bop == 4 else "xor-test"
            code = UREG_CODES["USTAT1"] + sreg
            mask = _wide(f, "data")
            source = _ureg(state.uregs, code)
            if isinstance(source, Const):
                result = (
                    (source.value & mask) == mask if bop == 4 else source.value == mask
                )
            else:
                result = None
            mode1 = _ureg(state.uregs, UREG_CODES["MODE1"])
            simd = bool(mode1.value & (1 << 21)) if isinstance(mode1, Const) else None
            astatx_code = UREG_CODES["ASTATX"]
            astatx = _ureg_raw(state.uregs, astatx_code)
            if result is None:
                state.uregs[astatx_code] = _astatx_forget(astatx, 1 << BTF_BIT)
            else:
                state.uregs[astatx_code] = _astatx_define(
                    astatx, 1 << BTF_BIT, (1 << BTF_BIT) if result else 0
                )
            # In SIMD mode the complementary STKY/ASTAT pair is evaluated
            # independently.  Preserve that uncertainty unless both MODE1
            # and the complementary source are concrete.
            if sreg in (6, 7, 8, 9) and simd is not False:
                complement = {6: 7, 7: 6, 8: 9, 9: 8}[sreg]
                complement_source = _ureg(
                    state.uregs, UREG_CODES["USTAT1"] + complement
                )
                if simd is True and isinstance(complement_source, Const):
                    complement_result = (
                        (complement_source.value & mask) == mask
                        if bop == 4
                        else complement_source.value == mask
                    )
                else:
                    complement_result = None
                astaty_code = UREG_CODES["ASTATY"]
                astaty = _ureg_raw(state.uregs, astaty_code)
                if complement_result is None:
                    state.uregs[astaty_code] = _astatx_forget(astaty, 1 << BTF_BIT)
                else:
                    state.uregs[astaty_code] = _astatx_define(
                        astaty, 1 << BTF_BIT, (1 << BTF_BIT) if complement_result else 0
                    )
            _event(
                state,
                insn,
                "system-bit-test",
                register=UREG_NAMES[code],
                operation=operation,
                mask=mask,
                result=result,
                simd=simd,
            )
            return _advance(state, insn)
        operations = {
            0: ("set", lambda a, b: a | b),
            1: ("clear", lambda a, b: a & ~b),
            2: ("toggle", lambda a, b: a ^ b),
        }
        if bop not in operations:
            return [_stop(state, insn, "unsupported Type18a BOP %#x" % bop)]
        # ASTATx/y and STKYx/y have implicit complementary-register behavior
        # in SIMD mode.  Stop rather than invent MODE1/PE state for those
        # register pairs; the other SYSREG selections have no companion.
        if sreg in (6, 7, 8, 9):
            return [
                _stop(
                    state,
                    insn,
                    "unsupported Type18a SIMD-sensitive system register",
                )
            ]
        code = UREG_CODES["USTAT1"] + sreg
        mask = _wide(f, "data")
        previous = _ureg(state.uregs, code)
        operation, calculate = operations[bop]
        value = _bitwise(
            previous,
            Const(mask),
            "%s %s %#x" % (operation, UREG_NAMES[code], mask),
            calculate,
        )
        state.uregs[code] = value
        _event(
            state,
            insn,
            "system-bit-op",
            register=UREG_NAMES[code],
            operation=operation,
            mask=mask,
            previous=_json_value(previous),
            value=value,
        )
        return _advance(state, insn)
    if name == "20a":
        push_fields = ("lpu", "spu", "ppu")
        pop_fields = ("lpo", "spo", "ppo")
        if any(_field(f, field) for field in push_fields) and any(
            _field(f, field) for field in pop_fields
        ):
            return [_stop(state, insn, "invalid Type20a mixed push and pop")]
        unsupported = [
            field
            for field in (
                "lpu",
                "ppu",
                "llii",
                "lldwb",
                "lldi",
                "llpwb",
                "llpi",
            )
            if _field(f, field)
        ]
        if unsupported:
            return [
                _stop(
                    state,
                    insn,
                    "unsupported Type20a operations: " + ", ".join(unsupported),
                )
            ]
        push_status = bool(_field(f, "spu"))
        pop_status = bool(_field(f, "spo"))
        pop_loop = bool(_field(f, "lpo"))
        pop_pc = bool(_field(f, "ppo"))
        flush_cache = bool(_field(f, "fc"))
        astatx_code = UREG_CODES["ASTATX"]
        astaty_code = UREG_CODES["ASTATY"]
        mode1_code = UREG_CODES["MODE1"]
        stkyx_code = UREG_CODES["STKYX"]
        if push_status:
            # PUSH STS saves the exact ASTATX/ASTATY register, including any
            # partial knowledge, not a value moved to a general register: use
            # _ureg_raw so a PartialConst round-trips through POP STS intact.
            state.status_stack.append(
                (
                    _ureg_raw(state.uregs, astatx_code),
                    _ureg_raw(state.uregs, astaty_code),
                    _ureg(state.uregs, mode1_code),
                )
            )
            state.uregs[mode1_code] = _bitwise(
                _ureg(state.uregs, mode1_code),
                _ureg(state.uregs, UREG_CODES["MMASK"]),
                "MODE1 masked by PUSH STS",
                lambda mode1, mmask: mode1 & ~mmask,
            )
            state.uregs[stkyx_code] = _bitwise(
                _ureg(state.uregs, stkyx_code),
                Const(1 << 24),
                "status stack nonempty",
                lambda value, mask: value & ~mask,
            )
        if pop_status:
            if state.status_stack:
                astatx, astaty, mode1 = state.status_stack.pop()
                state.uregs[astatx_code] = astatx
                state.uregs[astaty_code] = astaty
                state.uregs[mode1_code] = mode1
            if not state.status_stack:
                state.uregs[stkyx_code] = _bitwise(
                    _ureg(state.uregs, stkyx_code),
                    Const(1 << 24),
                    "status stack empty",
                    lambda value, mask: value | mask,
                )
        if pop_loop:
            if state.loops:
                state.loops.pop()
            state.uregs[UREG_CODES["CURLCNTR"]] = (
                Const(state.loops[-1].remaining) if state.loops else Const(0xFFFFFFFF)
            )
            if not state.loops:
                state.uregs[stkyx_code] = _bitwise(
                    _ureg(state.uregs, stkyx_code),
                    Const(1 << 26),
                    "loop stacks empty",
                    lambda value, mask: value | mask,
                )
        if pop_pc:
            if state.call_stack:
                state.call_stack.pop()
            _sync_pc_stack(state)
        _event(
            state,
            insn,
            "stack-control",
            push_status=push_status,
            pop_status=pop_status,
            pop_loop=pop_loop,
            pop_pc=pop_pc,
            flush_cache=flush_cache,
            status_depth=len(state.status_stack),
        )
        return _advance(state, insn)
    if name == "12a_imm":
        count = (_field(f, "data[15:8]") << 8) | _field(f, "data[7:0]")
        return _start_counted_loop(state, insn, count)
    if name == "12a_ureg":
        count = _ureg(state.uregs, _field(f, "ureg"))
        if not isinstance(count, Const):
            return [_stop(state, insn, "nonconcrete Type12a UREG loop count")]
        return _start_counted_loop(state, insn, count.value)
    if state.pending and name in ("25a_direct", "25a_pcrel", "8a_abs", "8a_rel"):
        return [_stop(state, insn, "nested delayed transfer")]
    if name == "11c":
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
    if name == "11a":
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
            not_taken, insn, "predicate-assumption", condition=cond, predicate_assumption=False
        )
        return _return_transfer(taken, insn, True, delayed) + _return_transfer(
            not_taken, insn, False, delayed
        )
    if name == "9a_abs":
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
                    "unknown 9a_abs indirect target through I%d/M%d"
                    % (8 + pmi, 8 + pmm),
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
            not_taken, insn, "predicate-assumption", condition=cond, predicate_assumption=False
        )
        return transfer(taken, insn, target, call, True) + transfer(
            not_taken, insn, target, call, False
        )
    # The verified compiler return is a TRUE 9b_abs jump through I12/M14,
    # with two delay slots, one of which is the confident 25c_rframe form.
    # Do not treat rframe alone, its provisional 48-bit sibling, or another
    # register-indirect jump as a return.
    if name == "9b_abs":
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
                    "unknown 9b_abs indirect target through I%d/M%d"
                    % (8 + pmi, 8 + pmm),
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
    if name == "25c_rframe":
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
    if name in ("17a", "17b"):
        value = (
            _wide(f, "data") if name == "17a" else _signed(_field(f, "data[15:0]"), 16)
        )
        code = _field(f, "ureg")
        state.uregs[code] = Const(value)
        _event(
            state, insn, "ureg-write", ureg=UREG_NAMES[code], value=value & 0xFFFFFFFF
        )
        return _advance(state, insn)
    if name == "7a":
        # Type 7a is MODIFY: the manual guarantees an index-register update in
        # parallel with its optional compute.  The table now carries the M
        # register selector at bits 29-27, the same field Type7b uses.
        cond = _field(f, "cond")
        if cond not in (0x1F, 0x17):
            return [_stop(state, insn, "unsupported Type7a predicate")]
        bank = 8 if _field(f, "g") else 0
        source_low = _field(f, "is[2:2]") << 2 | _field(f, "is[1:0]")
        destination_low = source_low ^ _field(f, "idis")
        source, destination = source_low + bank, destination_low + bank
        modifier = _field(f, "m") + bank
        conditional = cond == 0x17
        circular_wrap: Optional[Const] = None
        if conditional:
            if _field(f, "compute[22:16]") or _field(f, "compute[15:0]"):
                return [_stop(state, insn, "unsupported Type7a conditional compute")]
            length = _ureg(state.uregs, UREG_CODES["L%d" % source])
            if isinstance(length, Const) and length.value != 0:
                # The MODIFY instruction wraps whenever L is nonzero,
                # independent of MODE1.CBUFEN (out/refs/sharc-plus-prm
                # lines 9344-9346, 10410-10412): unlike an ordinary
                # load/store post-modify, this is not gated by CBUFEN.
                base = _ureg(state.uregs, UREG_CODES["B%d" % source])
                index_now = _ureg(state.uregs, 16 + source)
                modifier_now = _ureg(state.uregs, 32 + modifier)
                scale_now = _access_modifier_scale("normal-word", state.assume_nw32)
                if (
                    isinstance(base, Const)
                    and isinstance(index_now, Const)
                    and isinstance(modifier_now, Const)
                ):
                    wrapped_value = _circular_wrap_const(
                        index_now.value,
                        base.value,
                        length.value,
                        modifier_now.value * scale_now,
                    )
                    if wrapped_value is None:
                        return [
                            _stop(
                                state,
                                insn,
                                "circular modifier is not smaller than L%d" % source,
                            )
                        ]
                    circular_wrap = Const(wrapped_value)
                else:
                    return [_stop(state, insn, "unsupported Type7a circular modify")]
            elif not isinstance(length, Const):
                return [_stop(state, insn, "unsupported Type7a circular modify")]
            predicate = _predicate(state, cond)
            mode1 = _ureg(state.uregs, UREG_CODES["MODE1"])
            if predicate is False and isinstance(mode1, Const) and not (
                mode1.value & (1 << 21)
            ):
                _event(
                    state,
                    insn,
                    "i-modify-skipped",
                    source="I%d" % source,
                    destination="I%d" % destination,
                    predicate="PEx false, SISD",
                )
                return _advance(state, insn)
            if predicate is not True:
                state.uregs[16 + destination] = Unknown(
                    "conditional Type7a modify outcome"
                )
                _event(
                    state,
                    insn,
                    "i-modify-uncertain",
                    source="I%d" % source,
                    destination="I%d" % destination,
                    predicate="PEx unknown" if predicate is None else "PEx false, SIMD unknown",
                    scale_assumption="assume_nw32" if state.assume_nw32 else "unscaled normal-word",
                )
                return _advance(state, insn)
        try:
            compute = _compute(
                f,
                False,
                dict(state.uregs),
                state.special,
                approx_recips=state.approx_recips,
            )
        except ValueError as error:
            return [_stop(state, insn, str(error))]
        if circular_wrap is not None:
            state.uregs[16 + destination] = circular_wrap
            _event(
                state,
                insn,
                "i-modify",
                source="I%d" % source,
                destination="I%d" % destination,
                modifier="M%d" % modifier,
                circular=True,
            )
        else:
            index_value = _ureg(state.uregs, 16 + source)
            modifier_value = _ureg(state.uregs, 32 + modifier)
            scale = _access_modifier_scale("normal-word", state.assume_nw32)
            scaled_modifier = _multiply(
                modifier_value, Const(scale), "M%d * %d" % (modifier, scale)
            )
            state.uregs[16 + destination] = _add(
                index_value,
                scaled_modifier,
                "I%d + M%d * %d" % (source, modifier, scale),
            )
            _event(
                state,
                insn,
                "i-modify",
                source="I%d" % source,
                destination="I%d" % destination,
                modifier="M%d" % modifier,
                **({"scale_assumption": "assume_nw32"} if conditional and state.assume_nw32 else {}),
            )
        if compute is not None:
            _apply_compute(state, insn, compute)
        return _advance(state, insn)
    if name == "7b":
        # Type 7b VISA MODIFY (out/refs/sharc-plus-prm pp.13-49/13-50): the
        # same index-register update as Type7a's MODIFY, minus the parallel
        # compute option, over the full condition-code range (not just
        # Type7a's 0x1F/0x17). "If the DAG's Lx and Bx registers that
        # correspond to Ia or Ic are set up for circular buffering, the
        # modify operation always executes circular buffer wraparound,
        # independent of the state of the CBUFEN bit" (same page), so --
        # unlike an ordinary load/store post-modify -- this form is not
        # gated by MODE1.CBUFEN.
        cond = _field(f, "cond")
        bank = 8 if _field(f, "g") else 0
        source_low = _field(f, "is[2:2]") << 2 | _field(f, "is[1:0]")
        destination_low = source_low ^ _field(f, "idis")
        source, destination = source_low + bank, destination_low + bank
        modifier = _field(f, "m") + bank

        def modify7b(executed: State) -> None:
            length = _ureg(executed.uregs, UREG_CODES["L%d" % source])
            index_value = _ureg(executed.uregs, 16 + source)
            modifier_value = _ureg(executed.uregs, 32 + modifier)
            scale = _access_modifier_scale("normal-word", executed.assume_nw32)
            if (
                isinstance(length, Const)
                and length.value != 0
                and isinstance(index_value, Const)
                and isinstance(modifier_value, Const)
            ):
                base = _ureg(executed.uregs, UREG_CODES["B%d" % source])
                if isinstance(base, Const):
                    wrapped = _circular_wrap_const(
                        index_value.value,
                        base.value,
                        length.value,
                        modifier_value.value * scale,
                    )
                    executed.uregs[16 + destination] = (
                        Const(wrapped)
                        if wrapped is not None
                        else Unknown(
                            "circular modifier is not smaller than L%d" % source
                        )
                    )
                    _event(
                        executed,
                        insn,
                        "i-modify",
                        source="I%d" % source,
                        destination="I%d" % destination,
                        modifier="M%d" % modifier,
                        circular=True,
                    )
                    return
            scaled_modifier = _multiply(
                modifier_value, Const(scale), "M%d * %d" % (modifier, scale)
            )
            executed.uregs[16 + destination] = _add(
                index_value,
                scaled_modifier,
                "I%d + M%d * %d" % (source, modifier, scale),
            )
            _event(
                executed,
                insn,
                "i-modify",
                source="I%d" % source,
                destination="I%d" % destination,
                modifier="M%d" % modifier,
            )

        predicate = _predicate(state, cond)
        if predicate is True:
            modify7b(state)
            return _advance(state, insn)
        if predicate is False:
            _event(
                state,
                insn,
                "i-modify-skipped",
                source="I%d" % source,
                destination="I%d" % destination,
                condition=cond,
            )
            return _advance(state, insn)
        executed, skipped = _copy(state), _copy(state)
        modify7b(executed)
        _event(
            skipped,
            insn,
            "i-modify-skipped",
            source="I%d" % source,
            destination="I%d" % destination,
            condition=cond,
        )
        return _advance(executed, insn) + _advance(skipped, insn)
    if name == "7d":
        # SHARC+ Core Programming Reference (out/refs/sharc-plus-prm), ACONV
        # (Type 7d), Figure 13-21 p.352 and its Encode Table (same page):
        # this handler is deliberately only the pure ACONV row: cond=11111
        # and an empty compute (Table 13-22, p.350). decode_table.json pins
        # those fields into the mask/value, so cond/compute are not free here.
        # The PRM also describes conditional/compute-parallel Type7d rows;
        # they are outside this bounded decoder contract and must not silently
        # enter this handler as pure ACONV. g selects
        # DAG1/DAG2 (add 8, as for Type7a/Type19a); breg selects the I or B
        # register class; toby selects W2B (1) vs B2W (0); the destination
        # register is the source XOR idis, the same trick as Type7a/Type19a.
        #
        # Table 6-4 "Switch Address Instruction Semantics" (same PRM p.200,
        # printed 6-16; identical table in out/refs/sc58x-2158x-prm) hedges
        # the shift:
        #   "Id = B2W(Is) ... Base addr in byte-addressed space: Convert
        #   byte pointer to word pointer. Likely semantics Id <- Is >> 2.
        #   Exact semantics depend on address map and must work correctly
        #   for all addresses in both internal and external memory. In case
        #   of byte addresses not having word space equivalent Is will be
        #   retained as is i.e. Id = Is and illegal address space (ILAD)
        #   interrupt is generated."
        #   "Id = W2B(Is) ... Likely semantics Id <- Is << 2 ... [same ILAD
        #   hedge]." (Bd/Bs rows mirror Id/Is.)
        # This decoder does not model the address map or the ILAD trap, so
        # it only applies the documented "likely" shift, tags the event
        # semantics="prm-likely", and stops when the source is not concrete
        # rather than guess whether the trap fires.
        bank = 8 if _field(f, "g") else 0
        source_low = _field(f, "is[2:2]") << 2 | _field(f, "is[1:0]")
        destination_low = source_low ^ _field(f, "idis")
        source, destination = source_low + bank, destination_low + bank
        breg = bool(_field(f, "breg"))
        reg_class = "B" if breg else "I"
        base_code = UREG_CODES["B0"] if breg else UREG_CODES["I0"]
        src_code, dst_code = base_code + source, base_code + destination
        value = _ureg(state.uregs, src_code)
        w2b = bool(_field(f, "toby"))
        direction = "w2b" if w2b else "b2w"
        if isinstance(value, Unknown) or isinstance(value, PartialConst):
            return [
                _stop(
                    state,
                    insn,
                    "Type7d %s(%s%d) source is not concrete"
                    % (direction.upper(), reg_class, source),
                )
            ]
        result = _aconv(value, w2b, src_code, state.pc_sw)
        state.uregs[dst_code] = result
        _event(
            state,
            insn,
            "aconv",
            direction=direction,
            source="%s%d" % (reg_class, source),
            destination="%s%d" % (reg_class, destination),
            value=_json_value(result),
            semantics="prm-likely",
        )
        return _advance(state, insn)
    if name == "3a":
        # PRM Type 3a is a conditional compute plus one normal-word DM/PM
        # transfer. Table 13-1's syntax row is "IF cond compute, DM(Ia,Mb)
        # = Ureg" -- cond gates the *whole* instruction, not just the
        # compute half (PRM p.7924: a false condition "generate[s] NOPs
        # on the processing element"), so a resolved-false predicate skips
        # both the transfer and the compute, and an unresolved predicate
        # forks into executed/skipped states exactly like every other
        # conditional form here (2a, 5a_move, 9a_abs). Long-word pairs
        # remain deliberately unsupported.
        if _field(f, "l"):
            return [_stop(state, insn, "unsupported Type3a long-word access")]
        cond = _field(f, "cond")
        old = dict(state.uregs)
        compute_fields = dict(f)
        compute_field = _field(f, "compute")
        compute_fields["compute[22:16]"] = compute_field >> 16
        compute_fields["compute[15:0]"] = compute_field & 0xFFFF
        try:
            compute = _compute(
                compute_fields,
                False,
                old,
                state.special,
                approx_recips=state.approx_recips,
            )
        except ValueError as error:
            return [_stop(state, insn, str(error))]

        def run_transfer(target: State) -> None:
            bank = 8 if _field(f, "g") else 0
            index, modifier = _field(f, "i") + bank, _field(f, "m") + bank
            post_modify = bool(_field(f, "u"))
            space = "PM" if bank else "DM"
            iv, mv = _ureg(old, 16 + index), _ureg(old, 32 + modifier)
            scale = _access_modifier_scale("normal-word", target.assume_nw32)
            scaled_mv = _multiply(mv, Const(scale), "M%d * %d" % (modifier, scale))
            modified = _add(iv, scaled_mv, "I%d + M%d * %d" % (index, modifier, scale))
            address = iv if post_modify else modified
            ureg = _field(f, "ureg")
            if _field(f, "d"):
                value = _ureg(old, ureg)
                _event(
                    target,
                    insn,
                    "store",
                    space=space,
                    ureg=UREG_NAMES[ureg],
                    value=value,
                    address=address,
                    expression=_render(address),
                    concrete_write=_dm_write(target, address, 4, value)
                    if space == "DM"
                    else False,
                    addressing_mode="post-modify" if post_modify else "pre-modify",
                    access_width="normal-word",
                )
            else:
                loaded = _load_normal_ureg(target, space, address, ureg)
                _event(
                    target,
                    insn,
                    "load",
                    space=space,
                    ureg=UREG_NAMES[ureg],
                    address=address,
                    expression=_render(address),
                    concrete_value=loaded,
                    addressing_mode="post-modify" if post_modify else "pre-modify",
                    access_width="normal-word",
                )
            if post_modify:
                target.uregs[16 + index] = modified
            if compute is not None:
                _apply_compute(target, insn, compute)

        predicate = _predicate(state, cond)
        if predicate is True:
            run_transfer(state)
            state.trace[-1].update(condition=cond, predicate_assumption=True)
            return _advance(state, insn)
        if predicate is False:
            _event(
                state, insn, "type3a-skipped", condition=cond, predicate_assumption=False
            )
            return _advance(state, insn)
        executed, skipped = _copy(state), _copy(state)
        run_transfer(executed)
        executed.trace[-1].update(condition=cond, predicate_assumption=True)
        _event(
            skipped, insn, "type3a-skipped", condition=cond, predicate_assumption=False
        )
        return _advance(executed, insn) + _advance(skipped, insn)
    if name == "14a":
        if _field(f, "l"):
            code = _field(f, "ureg")
            if _field(f, "g"):
                return [_stop(state, insn, "unsupported Type14a PM long-word access")]
            if code & 1 or code + 1 >= len(UREG_NAMES):
                return [_stop(state, insn, "unsupported Type14a odd UREG pair")]
            address = _wide(f, "addr")
            rendered = _render(Const(address))
            pair = (code, code + 1)
            if _field(f, "d"):
                values = tuple(_ureg(state.uregs, item) for item in pair)
                writes = tuple(
                    _dm_write(state, address + 4 * offset, 4, value)
                    for offset, value in enumerate(values)
                )
                concrete_write = all(writes)
                _event(
                    state,
                    insn,
                    "store",
                    space="DM",
                    ureg_pair=[UREG_NAMES[item] for item in pair],
                    values=[_json_value(value) for value in values],
                    address=address,
                    expression=rendered,
                    access_width="long-word",
                    concrete_write=concrete_write,
                    simd_companion_possible=False,
                )
            else:
                values = tuple(
                    _dm_read(state, address + 4 * offset, 4)
                    for offset in range(2)
                )
                for item, value, offset in zip(pair, values, range(2)):
                    state.uregs[item] = value or Unknown(
                        "memory-address " + _render(Const(address + 4 * offset))
                    )
                _event(
                    state,
                    insn,
                    "load",
                    space="DM",
                    ureg_pair=[UREG_NAMES[item] for item in pair],
                    address=address,
                    expression=rendered,
                    concrete_values=[
                        _json_value(value)
                        if value is not None
                        else {"unknown": "unavailable memory"}
                        for value in values
                    ],
                    access_width="long-word",
                    simd_companion_possible=False,
                )
            return _advance(state, insn)
        address = _wide(f, "addr")
        rendered = _render(Const(address))
        code = _field(f, "ureg")
        space = "PM" if _field(f, "g") else "DM"
        try:
            companion = _simd_ureg_mem_companion(state, code, Const(address))
        except ValueError as error:
            return [_stop(state, insn, str(error))]
        if _field(f, "d"):
            value = _ureg(state.uregs, code)
            _event(
                state,
                insn,
                "store",
                space=space,
                ureg=UREG_NAMES[code],
                value=value,
                address=address,
                expression=rendered,
                simd_companion_possible=True,
                **(
                    {"concrete_write": _dm_write(state, address, 4, value)}
                    if space == "DM" and state.concrete is not None
                    else {}
                ),
            )
            if companion is not None and space == "DM":
                companion_value = _ureg(state.uregs, companion[0])
                _dm_write(state, companion[1], 4, companion_value)
                _event(
                    state,
                    insn,
                    "store-pey",
                    space=space,
                    ureg=UREG_NAMES[companion[0]],
                    value=companion_value,
                    address=companion[1],
                    expression=_render(companion[1]),
                )
        else:
            loaded = _load_normal_ureg(state, space, address, code)
            _event(
                state,
                insn,
                "load",
                space=space,
                ureg=UREG_NAMES[code],
                address=address,
                expression=rendered,
                concrete_value=loaded,
                simd_companion_possible=True,
            )
            if companion is not None and space == "DM":
                companion_loaded = _load_normal_ureg(
                    state, space, companion[1], companion[0]
                )
                _event(
                    state,
                    insn,
                    "load-pey",
                    space=space,
                    ureg=UREG_NAMES[companion[0]],
                    address=companion[1],
                    expression=_render(companion[1]),
                    concrete_value=companion_loaded,
                )
        return _advance(state, insn)
    if name == "14d":
        # SHARC+ Core Programming Reference (out/refs/sharc-plus-prm)
        # pp.384-387, Figure 15-2 ("Type14d Instruction Opcode"): a direct-
        # address DM <-> R-register-file move, an "extension (exclusive
        # access) to 14a instruction". The w/ex/d/l opcode table (p.384-385)
        # lists only EX/LWEX rows (Dreg = dm(addr32) EX/LWEX and the mirror
        # store) at w=1,ex=1; every w=0 row is BH/BHEX (store, l selects
        # byte/short) or BHSE/BHSEEX (load, l selects byte/short and x
        # selects zero- vs sign-extend, p.386 BHSE/BHSEEX Encode Tables).
        # BWSE/SWSE are load-only per the Description on p.386. This
        # decoder does not model exclusive-access monitors, so it stops on
        # ex=1 (EX/BHEX/BHSEEX/LWEX) and on the undocumented w=1,ex=0
        # combination the opcode table has no row for.
        if _field(f, "ex"):
            return [_stop(state, insn, "unsupported Type14d exclusive access")]
        if _field(f, "w"):
            return [
                _stop(state, insn, "undocumented Type14d encoding (w=1, ex=0)")
            ]
        store = bool(_field(f, "d"))
        l_bit, x_bit = _field(f, "l"), _field(f, "x")
        if store:
            if x_bit:
                return [
                    _stop(
                        state,
                        insn,
                        "undocumented Type14d store encoding (x=1)",
                    )
                ]
            access_width, width, signed = (
                ("byte", 1, False),
                ("short-word", 2, False),
            )[l_bit]
        else:
            access_width, width, signed = {
                (0, 0): ("byte", 1, False),
                (1, 0): ("short-word", 2, False),
                (0, 1): ("byte-sign-extended", 1, True),
                (1, 1): ("short-word-sign-extended", 2, True),
            }[(l_bit, x_bit)]
        address = _wide(f, "addr")
        rendered = _render(Const(address))
        code = _field(f, "dreg")
        if store:
            value = _ureg(state.uregs, code)
            _event(
                state,
                insn,
                "store",
                space="DM",
                dreg="R%d" % code,
                value=value,
                address=address,
                expression=rendered,
                access_width=access_width,
                concrete_write=_dm_write(state, address, width, value),
            )
        else:
            loaded = _dm_read(state, address, width, signed)
            state.uregs[code] = loaded or Unknown("memory-address " + rendered)
            _event(
                state,
                insn,
                "load",
                space="DM",
                dreg="R%d" % code,
                address=address,
                expression=rendered,
                concrete_value=loaded,
                access_width=access_width,
            )
        return _advance(state, insn)
    if name in ("5a_move", "5b_move"):
        cond = _field(f, "cond")
        old = dict(state.uregs)
        compute = None
        if name == "5a_move":
            try:
                compute = _compute(
                    f, False, old, state.special, approx_recips=state.approx_recips
                )
            except ValueError as error:
                return [_stop(state, insn, str(error))]
        src = (
            _field(f, "srcureghigh") << 2
            | _field(f, "srcureglow[1:1]") << 1
            | _field(f, "srcureglow[0:0]")
        )
        dst = _field(f, "dstureg")
        copied = _ureg(old, src)
        # The unconditional ("always") case is PE-independent by
        # construction, so its SIMD companion can be resolved without
        # forking on the predicate at all (SHARC+ PRM p.28, "Data and
        # Complementary Data Register Transfers": a complementary
        # destination with an uncomplementary source loads both PEs from
        # that one source -- "R5 = I8; loads R5 and S5 with I8"; a
        # complementary source and destination move each PE from its own
        # register; an uncomplementary destination gets no implicit move
        # at all, p.4-55 Table 4-22). Conditional (cond != 0x1F) ureg
        # copies are not SIMD-duplicated here: PEx's and PEy's predicates
        # can differ, which would need this form's existing predicate-fork
        # logic doubled for the PEy side too. An unresolved MODE1.PEYEN is
        # treated like SISD (no companion), exactly like
        # _simd_ureg_mem_companion: the explicit copy is correct either
        # way, so unknown MODE1 only leaves a companion unmodelled.
        cureg_dst = _cureg_code(dst) if cond == 0x1F else None
        simd_companion_source = None
        if cureg_dst is not None and _simd_active(state) is True:
            cureg_src = _cureg_code(src)
            simd_companion_source = cureg_src if cureg_src is not None else src
        predicate = _predicate(state, cond)
        if predicate is False:
            _event(
                state,
                insn,
                "ureg-copy-skipped",
                source=UREG_NAMES[src],
                destination=UREG_NAMES[dst],
                condition=cond,
                predicate_assumption=False,
            )
            return _advance(state, insn)
        executed = state if predicate is True else _copy(state)
        # The Type 5a data move and compute both consume the pre-instruction file.
        executed.uregs[dst] = copied
        if simd_companion_source is not None:
            executed.uregs[cureg_dst] = _ureg(old, simd_companion_source)
        if compute is not None:
            _apply_compute(executed, insn, compute)
        _event(
            executed,
            insn,
            "ureg-copy",
            source=UREG_NAMES[src],
            destination=UREG_NAMES[dst],
            condition=cond,
            predicate_assumption=True,
            simd_companion=(
                {
                    "source": UREG_NAMES[simd_companion_source],
                    "destination": UREG_NAMES[cureg_dst],
                }
                if simd_companion_source is not None
                else None
            ),
        )
        if predicate is True:
            return _advance(executed, insn)
        skipped = _copy(state)
        _event(
            skipped,
            insn,
            "ureg-copy-skipped",
            source=UREG_NAMES[src],
            destination=UREG_NAMES[dst],
            condition=cond,
            predicate_assumption=False,
        )
        return _advance(executed, insn) + _advance(skipped, insn)
    if name == "2c":
        # Unconditional (no cond field): a SIMD-active MODE1 duplicates
        # this onto PEy's S register file too (PRM p.101, p.3-39).
        try:
            compute_x, compute_y = _compute_simd(state, f, True, dict(state.uregs))
        except ValueError as error:
            return [_stop(state, insn, str(error))]
        if compute_x is None:
            return [_stop(state, insn, "empty short compute")]
        _apply_compute_simd(state, insn, compute_x, compute_y)
        return _advance(state, insn)
    if name in ("2a_short", "2b"):
        # Both are 32-bit unconditional full-compute forms with no
        # condition field (Type2b: PRM prefix 0xc0, decode_table.json
        # "prm figure (overrides PGR; firmware-confirmed)"): always execute.
        # A SIMD-active MODE1 duplicates this onto PEy too (PRM p.101).
        try:
            compute_x, compute_y = _compute_simd(
                state,
                f,
                False,
                dict(state.uregs),
                state.special,
                approx_recips=state.approx_recips,
            )
        except ValueError as error:
            return [_stop(state, insn, str(error))]
        if compute_x is None:
            return [_stop(state, insn, "empty full compute")]
        _apply_compute_simd(state, insn, compute_x, compute_y)
        return _advance(state, insn)
    if name == "2a":
        # Type 2a conditionally executes a full compute.  Decode against the
        # pre-instruction register file before either predicate assumption mutates it.
        try:
            compute = _compute(
                f,
                False,
                dict(state.uregs),
                state.special,
                approx_recips=state.approx_recips,
            )
        except ValueError as error:
            return [_stop(state, insn, str(error))]
        if compute is None:
            return [_stop(state, insn, "empty full compute")]
        cond = _field(f, "cond")
        predicate = _predicate(state, cond)
        if predicate is True:
            _apply_compute(state, insn, compute)
            state.trace[-1].update(condition=cond, predicate_assumption=True)
            return _advance(state, insn)
        if predicate is False:
            _event(
                state,
                insn,
                "compute-skipped",
                condition=cond,
                predicate_assumption=False,
            )
            return _advance(state, insn)
        executed, skipped = _copy(state), _copy(state)
        _apply_compute(executed, insn, compute)
        executed.trace[-1].update(condition=cond, predicate_assumption=True)
        _event(
            skipped,
            insn,
            "compute-skipped",
            condition=cond,
            predicate_assumption=False,
        )
        return _advance(executed, insn) + _advance(skipped, insn)
    if name == "4a":
        if _field(f, "cond") != 0x1F:
            return [_stop(state, insn, "unsupported predicate")]
        old = dict(state.uregs)
        try:
            compute = _compute(
                f, False, old, state.special, approx_recips=state.approx_recips
            )
        except ValueError as error:
            return [_stop(state, insn, str(error))]
        index = _field(f, "i") + (8 if _field(f, "g") else 0)
        offset = _signed((_field(f, "data[5:5]") << 5) | _field(f, "data[4:0]"), 6)
        # The immediate modifier is in normal-word address units.  Only turn
        # it into a byte displacement when the caller has explicitly fixed
        # internal normal words at 32 bits.
        if state.assume_nw32:
            offset *= 4
        iv = _ureg(old, 16 + index)
        space = "PM" if _field(f, "g") else "DM"
        if _field(f, "u"):
            address, next_i = iv, _add(iv, Const(offset), "I%d + %d" % (index, offset))
        else:
            address, next_i = _add(iv, Const(offset), "I%d + %d" % (index, offset)), iv
        code = _field(f, "dreg")
        if _field(f, "d"):
            value = _ureg(old, code)
            _event(
                state,
                insn,
                "store",
                space=space,
                dreg="R%d" % code,
                value=value,
                address=address,
                expression=_render(address),
                concrete_write=_dm_write(state, address, 4, value)
                if space == "DM"
                else False,
            )
        else:
            loaded = _dm_read(state, address, 4) if space == "DM" else None
            state.uregs[code] = loaded or Unknown("memory-address " + _render(address))
            _event(
                state,
                insn,
                "load",
                space=space,
                dreg="R%d" % code,
                address=address,
                expression=_render(address),
                concrete_value=loaded,
            )
        state.uregs[16 + index] = next_i
        if compute is not None:
            _apply_compute(state, insn, compute)
        return _advance(state, insn)
    if name == "4b":
        # SHARC+ Core Programming Reference rev. 1.4, pp. 13-29--13-32:
        # conditional DM/PM transfer with a signed six-bit immediate modifier.
        width_fields = (_field(f, "l"), _field(f, "x"), _field(f, "w"))
        widths = {
            (1, 1, 1): ("normal-word", 4, False),
            (0, 0, 0): ("byte", 1, False),
            (1, 0, 0): ("short-word", 2, False),
            (0, 1, 0): ("byte-sign-extended", 1, True),
            (1, 1, 0): ("short-word-sign-extended", 2, True),
        }
        access_spec = widths.get(width_fields)
        if access_spec is None:
            return [_stop(state, insn, "unsupported Type4b access width")]
        access_width, width, signed = access_spec
        store = bool(_field(f, "d"))
        if store and signed:
            return [_stop(state, insn, "unsupported Type4b sign-extended store")]
        bank = 8 if _field(f, "g") else 0
        index = _field(f, "i") + bank
        offset = _signed((_field(f, "data[5:5]") << 5) | _field(f, "data[4:0]"), 6)
        offset *= _access_modifier_scale(access_width, state.assume_nw32)
        post_modify = bool(_field(f, "u"))
        space = "PM" if bank else "DM"
        code = _field(f, "dreg")
        cond = _field(f, "cond")

        def access_memory(executed: State) -> None:
            old = dict(executed.uregs)
            iv = _ureg(old, 16 + index)
            address = (
                iv
                if post_modify
                else _add(iv, Const(offset), "I%d + %d" % (index, offset))
            )
            if store:
                value = _ureg(old, code)
                _event(
                    executed,
                    insn,
                    "store",
                    space=space,
                    dreg="R%d" % code,
                    value=value,
                    address=address,
                    expression=_render(address),
                    concrete_write=_dm_write(executed, address, width, value)
                    if space == "DM"
                    else False,
                    addressing_mode="post-modify" if post_modify else "pre-modify",
                    access_width=access_width,
                    condition=cond,
                    predicate_assumption=True,
                )
            else:
                loaded = (
                    _dm_read(executed, address, width, signed)
                    if space == "DM"
                    else None
                )
                executed.uregs[code] = loaded or Unknown(
                    "memory-address " + _render(address)
                )
                _event(
                    executed,
                    insn,
                    "load",
                    space=space,
                    dreg="R%d" % code,
                    address=address,
                    expression=_render(address),
                    concrete_value=loaded,
                    addressing_mode="post-modify" if post_modify else "pre-modify",
                    access_width=access_width,
                    condition=cond,
                    predicate_assumption=True,
                )
            if post_modify:
                executed.uregs[16 + index] = _add(
                    iv, Const(offset), "I%d + %d" % (index, offset)
                )

        predicate = _predicate(state, cond)
        if predicate is True:
            access_memory(state)
            return _advance(state, insn)
        if predicate is False:
            _event(
                state,
                insn,
                "memory-access-skipped",
                condition=cond,
                predicate_assumption=False,
            )
            return _advance(state, insn)
        executed, skipped = _copy(state), _copy(state)
        access_memory(executed)
        _event(
            skipped,
            insn,
            "memory-access-skipped",
            condition=cond,
            predicate_assumption=False,
        )
        return _advance(executed, insn) + _advance(skipped, insn)
    if name == "3b":
        # SHARC+ Core Programming Reference rev. 1.4, pp. 13-16--13-19.
        # Validate and decode the complete access before making a predicate
        # assumption, so unsupported forms stop rather than creating paths.
        width_fields = (_field(f, "l"), _field(f, "x"), _field(f, "w"))
        access_width = ACCESS_WIDTHS.get(width_fields)
        if access_width is None:
            return [_stop(state, insn, "unsupported Type3b access width")]
        store = bool(_field(f, "d"))
        if store and access_width.endswith("sign-extended"):
            return [_stop(state, insn, "unsupported Type3b sign-extended store")]
        bank = 8 if _field(f, "g") else 0
        index, modifier = _field(f, "i") + bank, _field(f, "m") + bank
        post_modify = bool(_field(f, "u"))
        addressing_mode = "post-modify" if post_modify else "pre-modify"
        space = "PM" if bank else "DM"
        ureg = _field(f, "ureg")
        cond = _field(f, "cond")

        def access(executed: State) -> Optional[str]:
            old = dict(executed.uregs)
            iv, mv = _ureg(old, 16 + index), _ureg(old, 32 + modifier)
            widths = {
                "normal-word": 4,
                "byte": 1,
                "byte-sign-extended": 1,
                "short-word": 2,
                "short-word-sign-extended": 2,
                "long-word": 8,
            }
            width = widths[access_width]
            scale = _access_modifier_scale(access_width, executed.assume_nw32)
            scaled_mv = _multiply(mv, Const(scale), f"M{modifier} * {scale}")
            address = (
                iv
                if post_modify
                else _add(iv, scaled_mv, f"I{index} + M{modifier} * {scale}")
            )
            try:
                companion = _simd_ureg_mem_companion(
                    executed, ureg, address, access_width
                )
            except ValueError as error:
                return str(error)
            if store:
                value = _ureg(old, ureg)
                _event(
                    executed,
                    insn,
                    "store",
                    space=space,
                    ureg=UREG_NAMES[ureg],
                    value=value,
                    address=address,
                    expression=_render(address),
                    concrete_write=_dm_write(executed, address, width, value)
                    if space == "DM"
                    else False,
                    addressing_mode=addressing_mode,
                    access_width=access_width,
                    condition=cond,
                    predicate_assumption=True,
                )
                if companion is not None and space == "DM":
                    companion_value = _ureg(old, companion[0])
                    _dm_write(executed, companion[1], width, companion_value)
                    _event(
                        executed,
                        insn,
                        "store-pey",
                        space=space,
                        ureg=UREG_NAMES[companion[0]],
                        value=companion_value,
                        address=companion[1],
                        expression=_render(companion[1]),
                        addressing_mode=addressing_mode,
                        access_width=access_width,
                    )
            else:
                if access_width == "normal-word":
                    loaded: Optional[Const | dict[str, int]] = _load_normal_ureg(
                        executed, space, address, ureg
                    )
                else:
                    scalar_loaded = (
                        _dm_read(
                            executed,
                            address,
                            width,
                            access_width.endswith("sign-extended"),
                        )
                        if space == "DM"
                        else None
                    )
                    executed.uregs[ureg] = scalar_loaded or Unknown(
                        "memory-address " + _render(address)
                    )
                    loaded = scalar_loaded
                _event(
                    executed,
                    insn,
                    "load",
                    space=space,
                    ureg=UREG_NAMES[ureg],
                    address=address,
                    expression=_render(address),
                    concrete_value=loaded,
                    addressing_mode=addressing_mode,
                    access_width=access_width,
                    condition=cond,
                    predicate_assumption=True,
                )
                if companion is not None and space == "DM" and access_width == "normal-word":
                    companion_loaded = _load_normal_ureg(
                        executed, space, companion[1], companion[0]
                    )
                    _event(
                        executed,
                        insn,
                        "load-pey",
                        space=space,
                        ureg=UREG_NAMES[companion[0]],
                        address=companion[1],
                        expression=_render(companion[1]),
                        concrete_value=companion_loaded,
                        addressing_mode=addressing_mode,
                        access_width=access_width,
                    )
            if post_modify:
                executed.uregs[16 + index] = _add(
                    iv, scaled_mv, "I%d + M%d * %d" % (index, modifier, scale)
                )
            return None

        predicate = _predicate(state, cond)
        if predicate is True:
            error = access(state)
            if error:
                return [_stop(state, insn, error)]
            return _advance(state, insn)
        executed, skipped = _copy(state), _copy(state)
        error = access(executed)
        if error:
            return [_stop(executed, insn, error)]
        _event(
            skipped,
            insn,
            "memory-access-skipped",
            space=space,
            ureg=UREG_NAMES[ureg],
            addressing_mode=addressing_mode,
            access_width=access_width,
            condition=cond,
            predicate_assumption=False,
        )
        return _advance(executed, insn) + _advance(skipped, insn)
    if name == "3c":
        index, modifier = _field(f, "dmi"), _field(f, "dmm")
        old = dict(state.uregs)
        iv, mv = _ureg(old, 16 + index), _ureg(old, 32 + modifier)
        scale = _access_modifier_scale("normal-word", state.assume_nw32)
        scaled_mv = _multiply(mv, Const(scale), "M%d * %d" % (modifier, scale))
        address = iv
        state.uregs[16 + index] = _add(
            iv, scaled_mv, "I%d + M%d * %d" % (index, modifier, scale)
        )
        code = _field(f, "dreg")
        if _field(f, "d"):
            value = _ureg(old, code)
            _event(
                state,
                insn,
                "store",
                space="DM",
                dreg="R%d" % code,
                value=value,
                address=address,
                expression=_render(address),
                concrete_write=_dm_write(state, address, 4, value),
            )
        else:
            loaded = _dm_read(state, address, 4)
            state.uregs[code] = loaded or Unknown("memory-address " + _render(address))
            _event(
                state,
                insn,
                "load",
                space="DM",
                dreg="R%d" % code,
                address=address,
                expression=_render(address),
                concrete_value=loaded,
            )
        return _advance(state, insn)
    if name in ("16a", "16b"):
        if name == "16a" and (_field(f, "by") or _field(f, "sl")):
            return [_stop(state, insn, "unsupported Type16a by/sl")]
        index, modifier = (
            _field(f, "i") + (8 if _field(f, "g") else 0),
            _field(f, "m") + (8 if _field(f, "g") else 0),
        )
        old = dict(state.uregs)
        iv, mv = _ureg(old, 16 + index), _ureg(old, 32 + modifier)
        scale = _access_modifier_scale(
            "normal-word", state.assume_nw32 and not bool(_field(f, "g"))
        )
        scaled_mv = _multiply(mv, Const(scale), "M%d * %d" % (modifier, scale))
        address = iv
        value = Const(
            _wide(f, "data") if name == "16a" else _signed(_field(f, "data[15:0]"), 16)
        )
        _event(
            state,
            insn,
            "store",
            space="PM" if _field(f, "g") else "DM",
            address=address,
            expression=_render(address),
            value=value,
            by=_field(f, "by") if name == "16a" else 0,
            sl=_field(f, "sl") if name == "16a" else 0,
            concrete_write=_dm_write(state, address, 4, value)
            if not _field(f, "g")
            else False,
        )
        state.uregs[16 + index] = _add(
            iv, scaled_mv, "I%d + M%d * %d" % (index, modifier, scale)
        )
        return _advance(state, insn)
    if name == "15b":
        index = _field(f, "i") + (8 if _field(f, "g") else 0)
        offset = _signed(_field(f, "data[6:0]"), 7)
        # Type 15b's immediate modifier follows the selected memory width.
        # The opt-in 32-bit normal-word interpretation therefore makes an
        # unqualified (non-LW) displacement four bytes wide.
        if state.assume_nw32 and not _field(f, "l"):
            offset *= 4
        iv = state.uregs.get(16 + index, Unknown("uninitialized I%d" % index))
        address = _add(iv, Const(offset), "I%d + %d" % (index, offset))
        code = _field(f, "ureg")
        width = 8 if _field(f, "l") else 4
        if _field(f, "d"):
            value = state.uregs.get(code, Unknown("uninitialized " + UREG_NAMES[code]))
            _event(
                state,
                insn,
                "store",
                ureg=UREG_NAMES[code],
                address=address,
                expression=_render(address),
                long_word=bool(_field(f, "l")),
                concrete_write=_dm_write(state, address, width, value),
            )
        else:
            loaded = _dm_read(state, address, width)
            state.uregs[code] = loaded or Unknown("memory-address " + _render(address))
            _event(
                state,
                insn,
                "load",
                ureg=UREG_NAMES[code],
                address=address,
                expression=_render(address),
                long_word=bool(_field(f, "l")),
                concrete_value=loaded,
            )
        return _advance(state, insn)
    if name == "15a":
        # SHARC+ Core Programming Reference (out/refs/sharc-plus-prm)
        # pp.387-390, Figure 15-3 p.390 ("Type15a Instruction Opcode"):
        # DM(<data32>,Ia) = Ureg / Ureg = DM(<data32>,Ia), and the PM/Ic
        # form when g=1 (opcode table p.387: g=0 -> dm/I1REG(DAG1), g=1 ->
        # pm/I2REG(DAG2)). p.388 Description: "The I register is pre-
        # modified with an immediate value specified in the instruction.
        # The I register is not updated" -- pre-modify without writeback,
        # unlike Type19a's post-modify MODIFY. The optional (lw) "forces
        # register pair access" (p.389), modelled the same way as Type14a's
        # own (lw) register-pair form, with no SIMD companion.
        # SHARC+ Core Programming Reference (out/refs/sharc-plus-prm) pp.6-9
        # -6-10 ("Enhanced Modify Instruction for Address Scaling") and
        # Table 6-2 p.6-10/6-11: in byte-addressed space, an immediate
        # displacement on a load/store is scaled by the access size (the
        # (lw) row scales the same as an unqualified/(nw) access, not by 8);
        # in word-addressed space it is not scaled at all. This tracer's
        # opt-in 32-bit-normal-word model represents such pointers in byte
        # space (matching Type4a/4b/15b's own modifier scaling above), so
        # <data32> needs the same four-byte scaling those forms already
        # apply -- this 32-bit displacement was previously added unscaled,
        # which put a Type15a access at a different address than a Type4a/
        # 4b/15b access using the same architectural word offset from the
        # same I register (e.g. a stage-6 wavetable local stored via Type4a
        # at DM(I6-4) and re-read via Type15a at DM(0xfffffffc,I6)).
        bank = 8 if _field(f, "g") else 0
        index = _field(f, "i[2:0]") + bank
        addr = _wide(f, "addr") * _access_modifier_scale(
            "normal-word", state.assume_nw32
        )
        iv = _ureg(state.uregs, 16 + index)
        address = _add(iv, Const(addr), "I%d + %d" % (index, addr))
        rendered = _render(address)
        space = "PM" if bank else "DM"
        if _field(f, "l"):
            code = _field(f, "ureg")
            if code & 1 or code + 1 >= len(UREG_NAMES):
                return [_stop(state, insn, "unsupported Type15a odd UREG pair")]
            pair = (code, code + 1)
            offsets = tuple(
                _add(address, Const(4 * offset), "%s + %d" % (rendered, 4 * offset))
                for offset in range(2)
            )
            if _field(f, "d"):
                values = tuple(_ureg(state.uregs, item) for item in pair)
                writes = tuple(
                    _dm_write(state, offset_address, 4, value)
                    if space == "DM"
                    else False
                    for offset_address, value in zip(offsets, values)
                )
                concrete_write = all(writes)
                _event(
                    state,
                    insn,
                    "store",
                    space=space,
                    ureg_pair=[UREG_NAMES[item] for item in pair],
                    values=[_json_value(value) for value in values],
                    address=address,
                    expression=rendered,
                    access_width="long-word",
                    concrete_write=concrete_write,
                    simd_companion_possible=False,
                )
            else:
                values = tuple(
                    _dm_read(state, offset_address, 4) if space == "DM" else None
                    for offset_address in offsets
                )
                for item, value in zip(pair, values):
                    state.uregs[item] = value or Unknown(
                        "memory-address " + rendered
                    )
                _event(
                    state,
                    insn,
                    "load",
                    space=space,
                    ureg_pair=[UREG_NAMES[item] for item in pair],
                    address=address,
                    expression=rendered,
                    concrete_values=[
                        _json_value(value)
                        if value is not None
                        else {"unknown": "unavailable memory"}
                        for value in values
                    ],
                    access_width="long-word",
                    simd_companion_possible=False,
                )
            return _advance(state, insn)
        code = _field(f, "ureg")
        try:
            companion = _simd_ureg_mem_companion(state, code, address)
        except ValueError as error:
            return [_stop(state, insn, str(error))]
        if _field(f, "d"):
            value = _ureg(state.uregs, code)
            _event(
                state,
                insn,
                "store",
                space=space,
                ureg=UREG_NAMES[code],
                value=value,
                address=address,
                expression=rendered,
                simd_companion_possible=True,
                concrete_write=_dm_write(state, address, 4, value)
                if space == "DM"
                else False,
            )
            if companion is not None and space == "DM":
                companion_value = _ureg(state.uregs, companion[0])
                _dm_write(state, companion[1], 4, companion_value)
                _event(
                    state,
                    insn,
                    "store-pey",
                    space=space,
                    ureg=UREG_NAMES[companion[0]],
                    value=companion_value,
                    address=companion[1],
                    expression=_render(companion[1]),
                )
        else:
            loaded = _load_normal_ureg(state, space, address, code)
            _event(
                state,
                insn,
                "load",
                space=space,
                ureg=UREG_NAMES[code],
                address=address,
                expression=rendered,
                concrete_value=loaded,
                simd_companion_possible=True,
            )
            if companion is not None and space == "DM":
                companion_loaded = _load_normal_ureg(
                    state, space, companion[1], companion[0]
                )
                _event(
                    state,
                    insn,
                    "load-pey",
                    space=space,
                    ureg=UREG_NAMES[companion[0]],
                    address=companion[1],
                    expression=_render(companion[1]),
                    concrete_value=companion_loaded,
                )
        return _advance(state, insn)
    if name in ("19a", "19a_scaled"):
        bank = 8 if _field(f, "g") else 0
        src_low = _field(f, "is")
        # PGR Type 19 encodes the destination as Id XOR Is, not as a direct
        # register number (Table 17-2 and Figure 17-2).
        dst_low = src_low ^ _field(f, "idis")
        src, dst = src_low + bank, dst_low + bank
        v = state.uregs.get(16 + src, Unknown("uninitialized I%d" % src))
        delta = _signed(_wide(f, "data"), 32)
        scale = 1
        scaled_width = None
        if name == "19a_scaled":
            scaled_width = "normal-word" if _field(f, "w") else "short-word"
            # The opt-in normal-word model represents the loaded program's
            # internal pointers in byte space. Enhanced MODIFY therefore
            # scales NW/SW immediates by four/two bytes respectively.
            if state.assume_nw32:
                scale = 4 if _field(f, "w") else 2
                delta *= scale

        result = _add(v, Const(delta), "I%d + %d" % (src, delta))
        circular = False
        wrapped = False
        if name == "19a_scaled":
            base = _ureg(state.uregs, UREG_CODES["B%d" % src])
            length = _ureg(state.uregs, UREG_CODES["L%d" % src])
            if isinstance(length, Const) and length.value == 0:
                pass
            elif (
                isinstance(v, Const)
                and isinstance(base, Const)
                and isinstance(length, Const)
            ):
                circular = True
                byte_length = length.value * scale
                # PRM Sec. 6 "Circular Buffering": the modifier's magnitude
                # may be up to and including the buffer length L -- a step
                # of exactly L is a full lap and lands back on the start
                # value. Only a magnitude greater than L is out of the
                # single-correction case this tracer models.
                if byte_length < abs(delta):
                    result = Unknown("circular modifier is not smaller than L%d" % src)
                else:
                    candidate = _circular_wrap_const(
                        v.value, base.value, byte_length, delta
                    )
                    wrapped = candidate != (v.value + delta) & 0xFFFFFFFF
                    result = Const(candidate)
            else:
                bounded = _stack_bounded_symbol(v)
                byte_length = length.value * scale if isinstance(length, Const) else None
                if (
                    bounded is not None
                    and byte_length is not None
                    and abs(bounded[1]) < byte_length
                ):
                    # v has no proof yet of its own concrete value, but it
                    # is a symbol some caller has already bounded (an
                    # entry-time seed, or an earlier circular-MODIFY-
                    # derived symbol -- recursively, ultimately grounded in
                    # an entry-time seed), offset by less than one buffer
                    # length. PRM p.6-23: "If the index pointer falls
                    # outside the buffer, the DAG subtracts or adds the
                    # buffer length to the index value, wrapping the index
                    # pointer back within the start and end boundaries of
                    # the buffer" -- one +-byte_length correction. So the
                    # true (concrete) result is v + delta, corrected by at
                    # most one +-byte_length: within one buffer length of
                    # wherever v's own bound places it -- never a claim
                    # that the result equals v, another modify site's
                    # result, or the same site's own value on a different
                    # visit. A FRESH symbol (never v's own name) is minted
                    # so two circular-MODIFY results are never treated as
                    # equal or made to cancel by the Affine algebra; this
                    # module does not itself know or state the numeric
                    # bound -- that is the caller's fact (tools/
                    # sharcwriters.py's STACK_SYMBOLS / CIRC_WRAP_SLACK).
                    fresh = "%s%d_%x" % (CIRC_SYMBOL_PREFIX, src, state.pc_sw)
                    result = Affine(constant=0, terms=((fresh, 1),))
                    circular = True
                else:
                    result = Unknown("scaled circular modify I%d" % src)
        state.uregs[16 + dst] = result
        _event(
            state,
            insn,
            "i-add",
            source="I%d" % src,
            destination="I%d" % dst,
            offset=delta,
            scaled_width=scaled_width,
            circular=circular,
            wrapped=wrapped,
        )
        return _advance(state, insn)
    if name == "9a_rel":
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
    if name in ("25a_direct", "25a_pcrel", "8a_abs", "8a_rel"):
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
    if name == "1a":
        # SHARC+ Core Programming Reference (out/refs/sharc-plus-prm)
        # pp.13-3--13-6, Figure 13-1: an unconditional compute in parallel
        # with a DM transfer (fixed to DAG1, I0-7/M0-7) and a PM transfer
        # (fixed to DAG2, I8-15/M8-15). "The I values are post-modified and
        # updated by the specified M registers. Pre-modify offset
        # addressing is not supported" (p.13-4) -- unlike Type3a/3b/4a/4b,
        # there is no pre/post "u" bit; both accesses are always
        # post-modify. This tracer does not model the SIMD Y-element
        # (PEy/Sn) companion access the same page describes.
        old = dict(state.uregs)
        try:
            compute = _compute(
                f, False, old, state.special, approx_recips=state.approx_recips
            )
        except ValueError as error:
            return [_stop(state, insn, str(error))]
        scale = _access_modifier_scale("normal-word", state.assume_nw32)

        def access1a(space: str, index: int, modifier: int, dreg: int, store: bool) -> None:
            iv, mv = _ureg(old, 16 + index), _ureg(old, 32 + modifier)
            scaled_mv = _multiply(mv, Const(scale), "M%d * %d" % (modifier, scale))
            address = iv
            if store:
                value = _ureg(old, dreg)
                _event(
                    state,
                    insn,
                    "store",
                    space=space,
                    dreg="R%d" % dreg,
                    value=value,
                    address=address,
                    expression=_render(address),
                    concrete_write=_dm_write(state, address, 4, value)
                    if space == "DM"
                    else False,
                    addressing_mode="post-modify",
                    access_width="normal-word",
                )
            else:
                loaded = _load_normal_ureg(state, space, address, dreg)
                _event(
                    state,
                    insn,
                    "load",
                    space=space,
                    dreg="R%d" % dreg,
                    address=address,
                    expression=_render(address),
                    concrete_value=loaded,
                    addressing_mode="post-modify",
                    access_width="normal-word",
                )
            state.uregs[16 + index] = _add(
                iv, scaled_mv, "I%d + M%d * %d" % (index, modifier, scale)
            )

        dm_index, dm_modifier = _field(f, "dmi[2:0]"), _field(f, "dmm[2:0]")
        pm_index = ((_field(f, "pmi[2:2]") << 2) | _field(f, "pmi[1:0]")) + 8
        pm_modifier = _field(f, "pmm[2:0]") + 8
        access1a("DM", dm_index, dm_modifier, _field(f, "dmdreg[3:0]"), bool(_field(f, "dmd")))
        access1a("PM", pm_index, pm_modifier, _field(f, "pmdreg[3:0]"), bool(_field(f, "pmd")))
        if compute is not None:
            _apply_compute(state, insn, compute)
        return _advance(state, insn)
    if name == "22c":
        # SHARC+ Core Programming Reference pp.16-13/16-14, Figure 16-8:
        # idle/emuidle. "The processor remains in the low power state
        # until an interrupt occurs. On return from the interrupt,
        # execution continues at the instruction following the Idle
        # instruction." This tracer does not model interrupts arriving, so
        # it advances straight to that following instruction -- the state
        # the manual says execution reaches -- rather than stopping on an
        # unmodeled halt.
        emu = bool(_field(f, "emu"))
        _event(state, insn, "idle", mode="emuidle" if emu else "idle")
        return _advance(state, insn)
    if name == "26a":
        # SHARC+ Core Programming Reference p.16-19/16-20, Figure 16-13:
        # SYNC, a fully fixed 48-bit word with no operand fields.
        # "Ensures completion of all pending writes on the system
        # interface as well as the internal memory (L1) interface. The
        # core pipeline is stalled until SYNC completes." This tracer does
        # not model write buffering or pipeline timing, so SYNC has no
        # register or memory effect to apply; it just advances.
        _event(state, insn, "sync")
        return _advance(state, insn)
    if name == "5a_swap":
        # SHARC+ Core Programming Reference pp.13-37/13-38, Figure 13-14:
        # Dreg <-> CDreg (the PEx Rn register swaps with its PEy
        # complementary Sn register), with an optional condition and a
        # parallel compute. This tracer's register file models only PEx
        # (Rn); the PEy companion (Sn) is not tracked at all (see the
        # "18a" ASTATY handling above for the same limitation on flags).
        # Since Rn's new value after the swap is whatever untracked value
        # was in Sn, that new value is Unknown -- there is nothing to
        # concretely compute -- but the parallel compute and the
        # condition/predicate-fork machinery are still modeled exactly
        # like Type5a (move) above.
        cond = _field(f, "cond")
        old = dict(state.uregs)
        try:
            compute = _compute(
                f, False, old, state.special, approx_recips=state.approx_recips
            )
        except ValueError as error:
            return [_stop(state, insn, str(error))]
        dreg = _field(f, "dreg")
        cdreg = _field(f, "cdreg")
        predicate = _predicate(state, cond)
        if predicate is False:
            _event(
                state,
                insn,
                "dreg-swap-skipped",
                dreg="R%d" % dreg,
                condition=cond,
                predicate_assumption=False,
            )
            return _advance(state, insn)
        executed = state if predicate is True else _copy(state)
        executed.uregs[dreg] = Unknown(
            "Type5a swap from untracked complementary S%d" % cdreg
        )
        if compute is not None:
            _apply_compute(executed, insn, compute)
        _event(
            executed,
            insn,
            "dreg-swap",
            dreg="R%d" % dreg,
            cdreg="S%d" % cdreg,
            condition=cond,
            predicate_assumption=True,
        )
        if predicate is True:
            return _advance(executed, insn)
        skipped = _copy(state)
        _event(
            skipped,
            insn,
            "dreg-swap-skipped",
            dreg="R%d" % dreg,
            condition=cond,
            predicate_assumption=False,
        )
        return _advance(executed, insn) + _advance(skipped, insn)
    if name == "4d":
        # SHARC+ Core Programming Reference pp.13-32--13-35, Figure 13-12:
        # a 48-bit re-encoding of Type4a's index+6-bit-immediate transfer
        # that adds the byte/short access-width options Type4a itself does
        # not support ("does not support compute option", p.13-32 NOTE),
        # so unlike Type4a there is no compute field here. Same u/g/d/i
        # layout and index arithmetic as Type4a and Type4b; the l/w/x
        # width table is Type3b/4b's shared ACCESS_WIDTHS (l, x, w).
        width_fields = (_field(f, "l"), _field(f, "x"), _field(f, "w"))
        access_width = ACCESS_WIDTHS.get(width_fields)
        if access_width is None:
            return [_stop(state, insn, "unsupported Type4d access width")]
        store = bool(_field(f, "d"))
        if store and access_width.endswith("sign-extended"):
            return [_stop(state, insn, "unsupported Type4d sign-extended store")]
        if _field(f, "cond") != 0x1F:
            return [_stop(state, insn, "unsupported Type4d predicate")]
        widths = {
            "normal-word": 4,
            "byte": 1,
            "byte-sign-extended": 1,
            "short-word": 2,
            "short-word-sign-extended": 2,
            "long-word": 8,
        }
        width = widths[access_width]
        bank = 8 if _field(f, "g") else 0
        index = _field(f, "i") + bank
        offset = _signed((_field(f, "data[5:5]") << 5) | _field(f, "data[4:0]"), 6)
        offset *= _access_modifier_scale(access_width, state.assume_nw32)
        post_modify = bool(_field(f, "u"))
        space = "PM" if bank else "DM"
        code = _field(f, "dreg")
        iv = _ureg(state.uregs, 16 + index)
        address = (
            iv if post_modify else _add(iv, Const(offset), "I%d + %d" % (index, offset))
        )
        if store:
            value = _ureg(state.uregs, code)
            _event(
                state,
                insn,
                "store",
                space=space,
                dreg="R%d" % code,
                value=value,
                address=address,
                expression=_render(address),
                concrete_write=_dm_write(state, address, width, value)
                if space == "DM"
                else False,
                addressing_mode="post-modify" if post_modify else "pre-modify",
                access_width=access_width,
            )
        else:
            loaded = (
                _load_normal_ureg(state, space, address, code)
                if access_width == "normal-word"
                else (
                    _dm_read(state, address, width, access_width.endswith("sign-extended"))
                    if space == "DM"
                    else None
                )
            )
            if access_width != "normal-word":
                state.uregs[code] = loaded or Unknown("memory-address " + _render(address))
            _event(
                state,
                insn,
                "load",
                space=space,
                dreg="R%d" % code,
                address=address,
                expression=_render(address),
                concrete_value=loaded,
                addressing_mode="post-modify" if post_modify else "pre-modify",
                access_width=access_width,
            )
        if post_modify:
            state.uregs[16 + index] = _add(iv, Const(offset), "I%d + %d" % (index, offset))
        return _advance(state, insn)
    if name == "3d":
        # SHARC+ Core Programming Reference pp.13-22--13-25, Figure 13-9: a
        # 48-bit re-encoding of Type3a's index+M-register transfer that
        # adds byte/short and exclusive-access options Type3a does not
        # support ("extension to 3a instruction (exclusive access without
        # compute option)", p.13-22 NOTE), so there is no compute field.
        # w selects the ACCESS (0) vs WACCESS (1) group and ex marks an
        # exclusive-access monitor this tracer does not model, matching
        # the existing "14d" handler's ex=1 stop above; both are left
        # unsupported here rather than guessed. The w=0/ex=0 ACCESS group
        # is a plain normal-word transfer (l/x unused); w=0/ex=1 is
        # BH/BHSE, the same (l, x, 0) slice of ACCESS_WIDTHS as Type3b/4d
        # use, but always exclusive, so it also stops.
        if _field(f, "w"):
            return [_stop(state, insn, "unsupported Type3d WACCESS")]
        if _field(f, "ex"):
            return [_stop(state, insn, "unsupported Type3d exclusive access")]
        access_width = "normal-word"
        store = bool(_field(f, "d"))
        bank = 8 if _field(f, "g") else 0
        index, modifier = _field(f, "i") + bank, _field(f, "m") + bank
        post_modify = bool(_field(f, "u"))
        space = "PM" if bank else "DM"
        ureg = _field(f, "ureg")
        cond = _field(f, "cond")

        def access3d(executed: State) -> None:
            old = dict(executed.uregs)
            iv, mv = _ureg(old, 16 + index), _ureg(old, 32 + modifier)
            scale = _access_modifier_scale(access_width, executed.assume_nw32)
            scaled_mv = _multiply(mv, Const(scale), "M%d * %d" % (modifier, scale))
            address = (
                iv
                if post_modify
                else _add(iv, scaled_mv, "I%d + M%d * %d" % (index, modifier, scale))
            )
            if store:
                value = _ureg(old, ureg)
                _event(
                    executed,
                    insn,
                    "store",
                    space=space,
                    ureg=UREG_NAMES[ureg],
                    value=value,
                    address=address,
                    expression=_render(address),
                    concrete_write=_dm_write(executed, address, 4, value)
                    if space == "DM"
                    else False,
                    addressing_mode="post-modify" if post_modify else "pre-modify",
                    access_width=access_width,
                )
            else:
                loaded = _load_normal_ureg(executed, space, address, ureg)
                _event(
                    executed,
                    insn,
                    "load",
                    space=space,
                    ureg=UREG_NAMES[ureg],
                    address=address,
                    expression=_render(address),
                    concrete_value=loaded,
                    addressing_mode="post-modify" if post_modify else "pre-modify",
                    access_width=access_width,
                )
            if post_modify:
                executed.uregs[16 + index] = _add(
                    iv, scaled_mv, "I%d + M%d * %d" % (index, modifier, scale)
                )

        predicate = _predicate(state, cond)
        if predicate is True:
            access3d(state)
            return _advance(state, insn)
        if predicate is False:
            _event(
                state,
                insn,
                "type3d-skipped",
                condition=cond,
                predicate_assumption=False,
            )
            return _advance(state, insn)
        executed, skipped = _copy(state), _copy(state)
        access3d(executed)
        _event(
            skipped, insn, "type3d-skipped", condition=cond, predicate_assumption=False
        )
        return _advance(executed, insn) + _advance(skipped, insn)
    if name in ("8p_undoc48", "21p_undoc16", "22p_undoc48"):
        # Confirmed real (non-misaligned) code in places, but with no
        # known semantics: docs/findings/05-sharc-isa-and-decoding.md
        # marks what Type8p/Type22p words do as "Open" (lines 330, 528-
        # 530), and this session's own sample of 21p_undoc16/22p_undoc48
        # instances off the chosen render path (SW 0x16b8f1-0x16b930)
        # found them clustered with other undecoded/gap forms and a
        # garbage-offset "15a" (PM(I14 + 0xb8bd4a)), i.e. inside a run
        # that looks like misaligned data, not confirmed instructions.
        # Guessing an execution semantics for a jump/call-shaped
        # (8p_undoc48) or fully unknown (21p/22p) opcode risks silently
        # mistracing control flow, so this stops with the specific reason
        # instead of the generic fallback below.
        return [
            _stop(
                state,
                insn,
                "undocumented form %s has no confirmed semantics" % name,
            )
        ]
    return [_stop(state, insn, "unsupported form " + str(name))]
