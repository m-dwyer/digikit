"""Data move forms: register/memory transfers and immediate loads and stores.

Each handler executes one decoded instruction and returns the successor
states. FORMS maps form names to handlers; sharc_core.forms merges the
family tables.
"""

from __future__ import annotations

from collections.abc import Mapping

from sharc_disasm import Instruction

from .compute import (
    _apply_compute,
    _compute,
)
from .encoding import (
    ACCESS_WIDTHS,
    UREG_NAMES,
    _field,
    _wide,
)
from .memory import (
    _access_modifier_scale,
    _dm_read,
    _dm_write,
    _load_normal_ureg,
    _simd_ureg_mem_companion,
)
from .sequencer import (
    _advance,
    _predicate,
)
from .state import (
    State,
    _copy,
    _cureg_code,
    _event,
    _json_value,
    _render,
    _simd_active,
    _stop,
    _ureg,
)
from .values import (
    Const,
    Unknown,
    _add,
    _multiply,
    _signed,
)


def _type_17a(
    state: State, insn: Instruction, f: Mapping[str, int], name: str
) -> list[State]:
    """17a, 17b."""
    value = _wide(f, "data") if name == "17a" else _signed(_field(f, "data[15:0]"), 16)
    code = _field(f, "ureg")
    state.uregs[code] = Const(value)
    _event(state, insn, "ureg-write", ureg=UREG_NAMES[code], value=value & 0xFFFFFFFF)
    return _advance(state, insn)


def _type_3a(
    state: State, insn: Instruction, f: Mapping[str, int], name: str
) -> list[State]:
    """3a."""
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
    _event(skipped, insn, "type3a-skipped", condition=cond, predicate_assumption=False)
    return _advance(executed, insn) + _advance(skipped, insn)


def _type_14a(
    state: State, insn: Instruction, f: Mapping[str, int], name: str
) -> list[State]:
    """14a."""
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
            loaded_values = tuple(
                _dm_read(state, address + 4 * offset, 4) for offset in range(2)
            )
            for item, mem_value, offset in zip(
                pair, loaded_values, range(2), strict=True
            ):
                state.uregs[item] = mem_value or Unknown(
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
                    for value in loaded_values
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


def _type_14d(
    state: State, insn: Instruction, f: Mapping[str, int], name: str
) -> list[State]:
    """14d."""
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
        return [_stop(state, insn, "undocumented Type14d encoding (w=1, ex=0)")]
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


def _type_5a_move(
    state: State, insn: Instruction, f: Mapping[str, int], name: str
) -> list[State]:
    """5a_move, 5b_move."""
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
    simd_companion = None
    if simd_companion_source is not None:
        # simd_companion_source is only ever set inside the "cureg_dst is
        # not None" branch above, so this implies cureg_dst is not None too.
        assert cureg_dst is not None
        executed.uregs[cureg_dst] = _ureg(old, simd_companion_source)
        simd_companion = {
            "source": UREG_NAMES[simd_companion_source],
            "destination": UREG_NAMES[cureg_dst],
        }
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
        simd_companion=simd_companion,
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


def _type_4a(
    state: State, insn: Instruction, f: Mapping[str, int], name: str
) -> list[State]:
    """4a."""
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


def _type_4b(
    state: State, insn: Instruction, f: Mapping[str, int], name: str
) -> list[State]:
    """4b."""
    # SHARC+ Core Programming Reference rev. 1.4, pp. 13-29--13-32:
    # conditional DM/PM transfer with a signed six-bit immediate modifier,
    # the same 3-bit ACCESS/BH/BHSE (l, x, w) table as Type3b/Type4d --
    # ACCESS_WIDTHS (sharc_core/encoding.py), not a separate hand-rolled
    # one. This handler previously carried its own local ``widths`` dict
    # that mis-keyed two of ACCESS_WIDTHS' six entries: (1, 1, 1) mapped
    # here to ("normal-word", 4, False) -- ACCESS_WIDTHS' own (1, 1, 1) is
    # "long-word" -- and (0, 1, 1) (ACCESS_WIDTHS' real normal-word key)
    # was missing entirely, so that combination always stopped as
    # "unsupported". Found by tools/sharc_widthaudit.py on dt2-1.16's
    # frame render: every Type4b (l, x, w) = (1, 1, 1) access there (317
    # occurrences) read/wrote a 4-byte normal word where the decoded
    # fields select an 8-byte long-word access.
    width_fields = (_field(f, "l"), _field(f, "x"), _field(f, "w"))
    access_width = ACCESS_WIDTHS.get(width_fields)
    if access_width is None:
        return [_stop(state, insn, "unsupported Type4b access width")]
    widths = {
        "normal-word": 4,
        "byte": 1,
        "byte-sign-extended": 1,
        "short-word": 2,
        "short-word-sign-extended": 2,
        "long-word": 8,
    }
    width = widths[access_width]
    signed = access_width.endswith("sign-extended")
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
            iv if post_modify else _add(iv, Const(offset), "I%d + %d" % (index, offset))
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
                _dm_read(executed, address, width, signed) if space == "DM" else None
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


def _type_3b(
    state: State, insn: Instruction, f: Mapping[str, int], name: str
) -> list[State]:
    """3b."""
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

    def access(executed: State) -> str | None:
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
            companion = _simd_ureg_mem_companion(executed, ureg, address, access_width)
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
                loaded: Const | dict[str, int] | None = _load_normal_ureg(
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
            if (
                companion is not None
                and space == "DM"
                and access_width == "normal-word"
            ):
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
    if predicate is False:
        # Matching _type_3a's identical fast path above: a concretely
        # False predicate must not fork (the missing case here was
        # forking on every False predicate, even a fully known one,
        # which a single-path concrete runner cannot resolve).
        _event(
            state,
            insn,
            "memory-access-skipped",
            space=space,
            ureg=UREG_NAMES[ureg],
            addressing_mode=addressing_mode,
            access_width=access_width,
            condition=cond,
            predicate_assumption=False,
        )
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


def _type_3c(
    state: State, insn: Instruction, f: Mapping[str, int], name: str
) -> list[State]:
    """3c."""
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


def _type_16a(
    state: State, insn: Instruction, f: Mapping[str, int], name: str
) -> list[State]:
    """16a, 16b."""
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


def _type_15b(
    state: State, insn: Instruction, f: Mapping[str, int], name: str
) -> list[State]:
    """15b.

    SHARC+ Core Programming Reference (out/refs/sharc-plus-prm) pp.389-392,
    Figure 15-4 p.392: DM(<data7>,Ia) = Ureg / Ureg = DM(<data7>,Ia).
    p.391 Description: "The optional (LW) in this syntax lets programs
    specify long word addressing, overriding default addressing from the
    memory map... if the instruction type uses optional forced long word
    modifier (LW)... register pair access is done" -- the same
    register-pair (ureg, ureg+1) at (address, address+4) Type15a's own
    (lw) already implements (this form's docstring, citing p.389), not a
    single wider (8-byte) transfer into one register. That was the
    previous model here; sharc_core.memory._dm_read intentionally refuses
    any width > 4 ("a register pair, which this tracer does not model"),
    so every (LW) Type15b site read Unknown rather than a wrong value --
    found tracing docs/findings/06's voice render polyphase interpolator
    (FUN_1c4f81's coefficient-table reads at sw 0x1c50b9 etc., all (LW)
    Type15b), which needs six int16 taps per phase from three (LW) pair
    reads, not six separate ones.
    p.6-10's (lw) row "scales the same as an unqualified/(nw) access, not
    by 8" (Type15a's docstring cites the same table): the <data7>
    displacement is the normal-word (x4 under assume_nw32) scale
    unconditionally, independent of l -- only the pair's own second word
    is a fixed +4, exactly as Type15a's l=1 branch already does.
    """
    index = _field(f, "i") + (8 if _field(f, "g") else 0)
    offset = _signed(_field(f, "data[6:0]"), 7)
    if state.assume_nw32:
        offset *= 4
    iv = _ureg(state.uregs, 16 + index)
    address = _add(iv, Const(offset), "I%d + %d" % (index, offset))
    rendered = _render(address)
    code = _field(f, "ureg")
    if _field(f, "l"):
        if code & 1 or code + 1 >= len(UREG_NAMES):
            return [_stop(state, insn, "unsupported Type15b odd UREG pair")]
        pair = (code, code + 1)
        offsets = tuple(
            _add(address, Const(4 * off), "%s + %d" % (rendered, 4 * off))
            for off in range(2)
        )
        if _field(f, "d"):
            values = tuple(_ureg(state.uregs, item) for item in pair)
            writes = tuple(
                _dm_write(state, offset_address, 4, value)
                for offset_address, value in zip(offsets, values, strict=True)
            )
            _event(
                state,
                insn,
                "store",
                ureg_pair=[UREG_NAMES[item] for item in pair],
                values=[_json_value(value) for value in values],
                address=address,
                expression=rendered,
                access_width="long-word",
                long_word=True,
                concrete_write=all(writes),
            )
        else:
            loaded_values = tuple(
                _dm_read(state, offset_address, 4) for offset_address in offsets
            )
            for item, mem_value in zip(pair, loaded_values, strict=True):
                state.uregs[item] = mem_value or Unknown("memory-address " + rendered)
            _event(
                state,
                insn,
                "load",
                ureg_pair=[UREG_NAMES[item] for item in pair],
                address=address,
                expression=rendered,
                access_width="long-word",
                long_word=True,
                concrete_values=[
                    _json_value(value)
                    if value is not None
                    else {"unknown": "unavailable memory"}
                    for value in loaded_values
                ],
            )
        return _advance(state, insn)
    if _field(f, "d"):
        value = state.uregs.get(code, Unknown("uninitialized " + UREG_NAMES[code]))
        _event(
            state,
            insn,
            "store",
            ureg=UREG_NAMES[code],
            address=address,
            expression=rendered,
            long_word=False,
            concrete_write=_dm_write(state, address, 4, value),
        )
    else:
        loaded = _dm_read(state, address, 4)
        state.uregs[code] = loaded or Unknown("memory-address " + rendered)
        _event(
            state,
            insn,
            "load",
            ureg=UREG_NAMES[code],
            address=address,
            expression=rendered,
            long_word=False,
            concrete_value=loaded,
        )
    return _advance(state, insn)


def _type_15a(
    state: State, insn: Instruction, f: Mapping[str, int], name: str
) -> list[State]:
    """15a."""
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
    addr = _wide(f, "addr") * _access_modifier_scale("normal-word", state.assume_nw32)
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
                _dm_write(state, offset_address, 4, value) if space == "DM" else False
                for offset_address, value in zip(offsets, values, strict=True)
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
            loaded_values = tuple(
                _dm_read(state, offset_address, 4) if space == "DM" else None
                for offset_address in offsets
            )
            for item, mem_value in zip(pair, loaded_values, strict=True):
                state.uregs[item] = mem_value or Unknown("memory-address " + rendered)
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
                    for value in loaded_values
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


def _type_1a(
    state: State, insn: Instruction, f: Mapping[str, int], name: str
) -> list[State]:
    """1a."""
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
    access1a(
        "DM", dm_index, dm_modifier, _field(f, "dmdreg[3:0]"), bool(_field(f, "dmd"))
    )
    access1a(
        "PM", pm_index, pm_modifier, _field(f, "pmdreg[3:0]"), bool(_field(f, "pmd"))
    )
    if compute is not None:
        _apply_compute(state, insn, compute)
    return _advance(state, insn)


def _type_5a_swap(
    state: State, insn: Instruction, f: Mapping[str, int], name: str
) -> list[State]:
    """5a_swap."""
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


def _type_4d(
    state: State, insn: Instruction, f: Mapping[str, int], name: str
) -> list[State]:
    """4d."""
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
            # The dict[str, int] PX1/PX2 summary only comes back from the
            # access_width == "normal-word" branch above (_load_normal_ureg's
            # combined-PX case), which this guard excludes.
            assert not isinstance(loaded, dict)
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


def _type_3d(
    state: State, insn: Instruction, f: Mapping[str, int], name: str
) -> list[State]:
    """3d."""
    # SHARC+ Core Programming Reference pp.13-22--13-25, Figure 13-9: a
    # 48-bit re-encoding of Type3a's index+M-register transfer that
    # adds byte/short and exclusive-access options Type3a does not
    # support ("extension to 3a instruction (exclusive access without
    # compute option)", p.13-22 NOTE), so there is no compute field.
    # w selects the ACCESS (0) vs WACCESS (1) group; ex marks an
    # exclusive-access monitor this tracer does not model, matching
    # the existing "14d" handler's ex=1 stop below -- left unsupported
    # rather than guessed.
    #
    # docs/findings/06 ("Wrap copies +0x1bc"): the w=0 ACCESS group's own
    # encode table (p.324, "ACCESS (Type 3d)") names every u/g/d row's
    # modifier from the BH ("... BH (Type 3d)", a store) or BHSE
    # ("... BHSE (Type 3d)", a load) sub-table that immediately follows
    # it -- both indexed by ex, l and (BHSE only) x -- not from a
    # separate "no modifier when ex=0" case. Those sub-tables' own (l, x)
    # columns are exactly Type3b/Type4d's byte/short(-sign-extended)
    # ACCESS_WIDTHS (l, x, w=0) slice (PRM pp.13-16--13-19/13-31--13-35).
    # This handler previously hardcoded w=0 (any ex) as normal-word,
    # l/x unused ("the plain 48-bit re-encoding of Type3a"): a concrete
    # run of dt2-1.16's voice render refuted that (sw 0x1c50fb/0x1c5322,
    # decoded fields l=0, x=0, w=0, ex=0 -- confirmed via tools/sharc.py's
    # insn table). With the hardcoded normal-word read, a looping voice's
    # ACTIVE flag was cleared on every wrap instead of tracking
    # FIELD_LOOP the way docs/findings/06's "Flags and seed" documents;
    # overriding the load to the ACCESS_WIDTHS-correct 1-byte value
    # (matching the sibling Type4d byte store at the very same site)
    # reproduced the documented wrap behaviour. So w=0 uses
    # ACCESS_WIDTHS[(l, x, 0)] regardless of ex -- only WACCESS (w=1)
    # and the exclusive-monitor semantics of ex=1 remain unsupported.
    if _field(f, "w"):
        return [_stop(state, insn, "unsupported Type3d WACCESS")]
    if _field(f, "ex"):
        return [_stop(state, insn, "unsupported Type3d exclusive access")]
    access_width = ACCESS_WIDTHS.get((_field(f, "l"), _field(f, "x"), _field(f, "w")))
    if access_width is None:
        return [_stop(state, insn, "unsupported Type3d access width")]
    store = bool(_field(f, "d"))
    if store and access_width.endswith("sign-extended"):
        return [_stop(state, insn, "unsupported Type3d sign-extended store")]
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
                concrete_write=_dm_write(executed, address, width, value)
                if space == "DM"
                else False,
                addressing_mode="post-modify" if post_modify else "pre-modify",
                access_width=access_width,
            )
        else:
            loaded = (
                _load_normal_ureg(executed, space, address, ureg)
                if access_width == "normal-word"
                else (
                    _dm_read(
                        executed,
                        address,
                        width,
                        access_width.endswith("sign-extended"),
                    )
                    if space == "DM"
                    else None
                )
            )
            if access_width != "normal-word":
                # The dict[str, int] PX1/PX2 summary only comes back from
                # the access_width == "normal-word" branch above (see
                # Type4d's identical guard); this branch never sees it.
                assert not isinstance(loaded, dict)
                executed.uregs[ureg] = loaded or Unknown(
                    "memory-address " + _render(address)
                )
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
    _event(skipped, insn, "type3d-skipped", condition=cond, predicate_assumption=False)
    return _advance(executed, insn) + _advance(skipped, insn)


FORMS = {
    "17a": _type_17a,
    "17b": _type_17a,
    "3a": _type_3a,
    "14a": _type_14a,
    "14d": _type_14d,
    "5a_move": _type_5a_move,
    "5b_move": _type_5a_move,
    "4a": _type_4a,
    "4b": _type_4b,
    "3b": _type_3b,
    "3c": _type_3c,
    "16a": _type_16a,
    "16b": _type_16a,
    "15b": _type_15b,
    "15a": _type_15a,
    "1a": _type_1a,
    "5a_swap": _type_5a_swap,
    "4d": _type_4d,
    "3d": _type_3d,
}
