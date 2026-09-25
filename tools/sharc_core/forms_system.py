"""System forms: bit operations on system registers, stacks, NOP, IDLE, SYNC and forms with no confirmed semantics.

Each handler executes one decoded instruction and returns the successor
states. FORMS maps form names to handlers; sharc_core.forms merges the
family tables.
"""

from __future__ import annotations

from collections.abc import Mapping

from sharc_disasm import Instruction

from .encoding import (
    BTF_BIT,
    UREG_CODES,
    UREG_NAMES,
    _field,
    _wide,
)
from .flags import (
    _astatx_define,
    _astatx_forget,
)
from .sequencer import (
    _advance,
    _pop_loop_stack,
    _pop_pc_stack,
)
from .state import (
    State,
    _event,
    _json_value,
    _stop,
    _ureg,
    _ureg_raw,
)
from .values import (
    Const,
    _bitwise,
)


def _type_21a(
    state: State, insn: Instruction, f: Mapping[str, int], name: str
) -> list[State]:
    """21a, 21c."""
    return _advance(state, insn)


def _type_18a(
    state: State, insn: Instruction, f: Mapping[str, int], name: str
) -> list[State]:
    """18a."""
    bop = _field(f, "bop")
    sreg = _field(f, "sreg")
    if bop in (4, 5):
        operation = "bit-test" if bop == 4 else "xor-test"
        code = UREG_CODES["USTAT1"] + sreg
        mask = _wide(f, "data")
        source = _ureg(state.uregs, code)
        if isinstance(source, Const):
            result = (source.value & mask) == mask if bop == 4 else source.value == mask
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
            complement_source = _ureg(state.uregs, UREG_CODES["USTAT1"] + complement)
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


def _type_20a(
    state: State, insn: Instruction, f: Mapping[str, int], name: str
) -> list[State]:
    """20a."""
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
        _pop_loop_stack(state)
    if pop_pc:
        _pop_pc_stack(state)
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


def _type_22c(
    state: State, insn: Instruction, f: Mapping[str, int], name: str
) -> list[State]:
    """22c."""
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


def _type_26a(
    state: State, insn: Instruction, f: Mapping[str, int], name: str
) -> list[State]:
    """26a."""
    # SHARC+ Core Programming Reference p.16-19/16-20, Figure 16-13:
    # SYNC, a fully fixed 48-bit word with no operand fields.
    # "Ensures completion of all pending writes on the system
    # interface as well as the internal memory (L1) interface. The
    # core pipeline is stalled until SYNC completes." This tracer does
    # not model write buffering or pipeline timing, so SYNC has no
    # register or memory effect to apply; it just advances.
    _event(state, insn, "sync")
    return _advance(state, insn)


def _type_8p_undoc48(
    state: State, insn: Instruction, f: Mapping[str, int], name: str
) -> list[State]:
    """8p_undoc48, 21p_undoc16, 22p_undoc48."""
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


FORMS = {
    "21a": _type_21a,
    "21c": _type_21a,
    "18a": _type_18a,
    "20a": _type_20a,
    "22c": _type_22c,
    "26a": _type_26a,
    "8p_undoc48": _type_8p_undoc48,
    "21p_undoc16": _type_8p_undoc48,
    "22p_undoc48": _type_8p_undoc48,
}
