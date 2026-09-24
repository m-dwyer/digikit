"""Per-form instruction execution: _execute dispatches on the decoded form
through FORMS, merged from the family modules (forms_compute, forms_move,
forms_dag, forms_flow, forms_system).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Callable, Dict, List

from sharc_disasm import Instruction

from .state import (
    State,
    _stop,
)
from .forms_compute import FORMS as _FORMS_COMPUTE
from .forms_move import FORMS as _FORMS_MOVE
from .forms_dag import FORMS as _FORMS_DAG
from .forms_flow import FORMS as _FORMS_FLOW
from .forms_system import FORMS as _FORMS_SYSTEM


def _merge(
    *tables: Mapping[str, Callable[..., List[State]]],
) -> Dict[str, Callable[..., List[State]]]:
    merged: Dict[str, Callable[..., List[State]]] = {}
    for table in tables:
        for form, handler in table.items():
            if form in merged:
                raise ValueError("form %s has two handlers" % form)
            merged[form] = handler
    return merged


# Form name -> handler(state, insn, fields, name).
FORMS = _merge(_FORMS_COMPUTE, _FORMS_MOVE, _FORMS_DAG, _FORMS_FLOW, _FORMS_SYSTEM)


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
    handler = FORMS.get(name)
    if handler is not None:
        return handler(state, insn, f, name)
    return [_stop(state, insn, "unsupported form " + str(name))]
