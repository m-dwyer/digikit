"""Concrete, single-path SHARC+ runner built on tools/sharc_trace.py.

tools/sharc_trace.py is a *symbolic* tracer: registers and memory it has not
been told about read back as ``Unknown``, an unresolved conditional forks
into several states, and every state deep-copies on each fork. That is right
for exploring what a routine's callers could pass; it is the wrong shape for
"run this one routine forward with real inputs to see what it actually
does" -- the question this module answers.

This module seeds every register with a real value (0, or the documented
SHARC+ reset value where sharc_trace already has one), keeps memory as the
loader image plus one flat write overlay (no per-path copies), and steps
:func:`tools.sharc_trace._execute` one instruction at a time. With fully
concrete inputs, every predicate sharc_trace would need to fork on should
already be resolvable, so this runner treats a fork -- or any state it
tracks as ``stopped`` -- as a hard stop: see :class:`Halt`.

    uv run python tools/sharc_run.py dt2-1.16 --start 0x1c4ecf --max-steps 20000

Library use goes through :class:`Runner` directly, without sharc.py's
sqlite/networkx machinery, given any already-loaded
:class:`tools.sharcldr.LoadedMemory`:

    import sharc_run
    runner = sharc_run.Runner(loaded_memory, start=0x1c4ecf)
    result = runner.run(max_steps=20000)
    print(result.halt.reason, result.instructions_per_second)
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import sharc_trace as st  # noqa: E402
from sharc_core.memory import UnmodeledMMR  # noqa: E402
from sharc_disasm import Instruction  # noqa: E402
from sharcldr import LoadedMemory  # noqa: E402

# The firmware's own hardware DAG-modify reset values (docs/findings/06),
# duplicated from tools/sharc.py's _DEFAULT_TRACE_REGS rather than importing
# sharc.py -- that module's sqlite/networkx machinery is otherwise unneeded
# here, and this runner also has to work from a bare LoadedMemory in tests.
DEFAULT_REGS: dict[str, int] = {
    "M5": 0,
    "M6": 1,
    "M7": -1,
    "M13": 0,
    "M14": 1,
    "M15": -1,
}


class Halt(Exception):
    """Why a single-path concrete run stopped.

    Every stop this runner recognises -- a natural tracer stop
    (``state.stopped``, which is also how a depth-0 RTS, a max-call-depth
    trip, or an unfollowable call surface), a breakpoint, running out of
    ``max_steps``, or a fork sharc_trace would need but concrete state
    cannot resolve -- is reported through this one exception so callers have
    a single place to look.
    """

    def __init__(
        self,
        reason: str,
        pc_sw: int,
        form: str | None = None,
        text: str = "",
    ) -> None:
        self.reason = reason
        self.pc_sw = pc_sw
        self.form = form
        self.text = text
        message = "%s at %#x (%s)" % (reason, pc_sw, form or "?")
        if text:
            message += ": " + text
        super().__init__(message)

    def to_json(self) -> dict:
        return {
            "reason": self.reason,
            "pc_sw": self.pc_sw,
            "form": self.form,
            "text": self.text,
        }


def reset_uregs(
    overrides: Mapping[str | int, int | str] | None = None,
) -> dict[int, st.Value]:
    """Every UREG code seeded with a real value.

    sharc_trace.CORE_UREG_RESET_VALUES already documents the reset value for
    the core status/loop registers it needs for its own ``--core-reset-state``
    option; every other UREG (R0-15, I/M/L/B/S0-15, PX, TPERIOD, USTAT1-4,
    ...) resets to 0 on real SHARC+ hardware, and sharc_trace leaves an
    unseeded UREG as ``Unknown`` -- exactly the thing a concrete, single-path
    run cannot tolerate, since it is what forces a fork or a stop.
    """
    uregs: dict[int, st.Value] = {
        code: st.Const(st.CORE_UREG_RESET_VALUES.get(name, 0))
        for code, name in enumerate(st.UREG_NAMES)
    }
    for key, value in (overrides or {}).items():
        uregs[st._seed_code(key)] = st._seed_value(value)
    return uregs


def make_state(
    data: LoadedMemory,
    start: int,
    *,
    regs: Mapping[str | int, int | str] | None = None,
    pokes: Mapping[int, int] | None = None,
    follow_loaded_calls: bool = True,
    continue_external_calls: bool = False,
    assume_nw32: bool = True,
    max_call_depth: int = 64,
    provisional_forms: Sequence[str] = (),
    explicit_memory_model: bool = False,
    approx_recips: bool = False,
) -> st.State:
    """A fully concrete State ready to step, with sharc_trace's own
    _dm_write() used to apply pokes -- the same canonicalisation (loader
    alias, MMR routing, width gating) a real store instruction gets.

    ``provisional_forms`` and ``explicit_memory_model`` are both opt-in and
    change nothing when left at their defaults: see sharc_core/state.py's
    ``State.provisional_forms``/``State.explicit_memory_model`` docstrings.
    ``approx_recips`` is sharc_trace.py's own pre-existing State field
    (State.approx_recips), exposed here too: a voice render's pitch/rate
    math (docs/findings/06's "voice record contract") goes through recips,
    whose ROM seed is undocumented without it.
    """
    if not isinstance(data, LoadedMemory):
        raise ValueError(
            "sharc_run needs a LoadedMemory image; concrete execution has "
            "nothing else to read memory from"
        )
    combined_regs: dict[str | int, int | str] = dict(DEFAULT_REGS.items())
    combined_regs.update(regs or {})
    mmrs: dict[int, st.Value] = {
        address: st.Const(value) for address, value in st.CORE_MMR_RESET_VALUES.items()
    }
    state = st.State(
        pc_sw=start,
        uregs=reset_uregs(combined_regs),
        concrete=data,
        base_sw=None,
        follow_loaded_calls=follow_loaded_calls,
        continue_external_calls=continue_external_calls,
        max_call_depth=max_call_depth,
        assume_nw32=assume_nw32,
        core_reset_state=True,
        mmrs=mmrs,
        record_events=False,
        provisional_forms=tuple(provisional_forms),
        explicit_memory_model=explicit_memory_model,
        approx_recips=approx_recips,
    )
    for address, value in sorted((pokes or {}).items()):
        if not st._dm_write(state, address, 4, st.Const(value & 0xFFFFFFFF)):
            raise ValueError(
                "poke at %#x did not take effect (outside a mapped DM "
                "region, or not 32-bit-normal-word addressable)" % address
            )
    return state


@dataclass
class RunResult:
    halt: Halt
    instructions: int
    elapsed: float
    form_counts: collections.Counter[str]
    start_pc_sw: int
    final_pc_sw: int
    max_call_depth_reached: int

    @property
    def instructions_per_second(self) -> float:
        return self.instructions / self.elapsed if self.elapsed > 0 else float("inf")

    def to_json(self) -> dict:
        return {
            "start_pc_sw": self.start_pc_sw,
            "final_pc_sw": self.final_pc_sw,
            "instructions": self.instructions,
            "elapsed_s": self.elapsed,
            "instructions_per_second": self.instructions_per_second,
            "max_call_depth_reached": self.max_call_depth_reached,
            "halt": self.halt.to_json(),
            "form_counts": dict(self.form_counts.most_common()),
        }


class Runner:
    """Steps one State through sharc_trace._execute, one instruction at a
    time, with a decode cache and no per-step trace/copy overhead.

    decode_at() (sharc_trace.py) reads only the immutable loader image, not
    a path's write overlay, so a cached Instruction can never go stale from
    a store this runner itself performs -- the firmware is not documented
    to self-modify PM either. invalidate() exists as a deliberate safety
    net anyway (call it with a pc_sw if that ever changes); it is not wired
    to every store automatically because nothing currently can make it
    fire, and a heuristic that fired on every store would defeat the cache.
    """

    def __init__(
        self,
        data: LoadedMemory,
        start: int,
        *,
        regs: Mapping[str | int, int | str] | None = None,
        pokes: Mapping[int, int] | None = None,
        follow_loaded_calls: bool = True,
        continue_external_calls: bool = False,
        assume_nw32: bool = True,
        max_call_depth: int = 64,
        breakpoints: Sequence[int] = (),
        provisional_forms: Sequence[str] = (),
        explicit_memory_model: bool = False,
        approx_recips: bool = False,
    ) -> None:
        self.data = data
        self.state = make_state(
            data,
            start,
            regs=regs,
            pokes=pokes,
            follow_loaded_calls=follow_loaded_calls,
            continue_external_calls=continue_external_calls,
            assume_nw32=assume_nw32,
            max_call_depth=max_call_depth,
            provisional_forms=provisional_forms,
            explicit_memory_model=explicit_memory_model,
            approx_recips=approx_recips,
        )
        self.breakpoints = frozenset(breakpoints)
        self.instructions = 0
        self.form_counts: collections.Counter[str] = collections.Counter()
        self.max_call_depth_reached = 0
        self._cache: dict[int, Instruction] = {}

    def invalidate(self, pc_sw: int) -> None:
        """Evict pc_sw from the decode cache. See the class docstring for
        why nothing calls this automatically today."""
        self._cache.pop(pc_sw, None)

    def _decode(self, pc_sw: int) -> Instruction:
        insn = self._cache.get(pc_sw)
        if insn is None:
            insn = st.decode_at(self.data, None, pc_sw)
            self._cache[pc_sw] = insn
        return insn

    def step(self) -> None:
        """Execute exactly one instruction, or raise Halt."""
        state = self.state
        if state.pc_sw in self.breakpoints:
            raise Halt("breakpoint", state.pc_sw)
        insn = self._decode(state.pc_sw)
        try:
            out = st._execute(state, insn)
        except UnmodeledMMR as exc:
            raise Halt(
                "mmr",
                state.pc_sw,
                insn.type_name,
                "unmodeled MMR %#x (%s)" % (exc.address, exc.name or "unnamed"),
            ) from exc
        if len(out) != 1:
            raise Halt(
                "fork (%d successors): a predicate or address went Unknown "
                "despite concrete input -- see the form's _execute branch "
                "for what read an unseeded value" % len(out),
                state.pc_sw,
                insn.type_name,
                insn.note,
            )
        state = out[0]
        self.state = state
        if state.stopped:
            raise Halt(state.stopped, state.pc_sw, insn.type_name, insn.note)
        self.instructions += 1
        # sharc_disasm.disassemble() always sets type_name to a decode-table
        # name or the literal string "unknown", never None (see the same
        # assertion in sharc_core/forms.py's _execute).
        assert insn.type_name is not None, "instruction with no type_name"
        self.form_counts[insn.type_name] += 1
        depth = len(state.call_stack)
        if depth > self.max_call_depth_reached:
            self.max_call_depth_reached = depth

    def run(self, max_steps: int) -> RunResult:
        start_pc_sw = self.state.pc_sw
        start_time = time.perf_counter()
        halt: Halt
        try:
            for _ in range(max_steps):
                self.step()
        except Halt as exc:
            halt = exc
        else:
            halt = Halt("max-steps", self.state.pc_sw)
        elapsed = time.perf_counter() - start_time
        return RunResult(
            halt=halt,
            instructions=self.instructions,
            elapsed=elapsed,
            form_counts=self.form_counts,
            start_pc_sw=start_pc_sw,
            final_pc_sw=self.state.pc_sw,
            max_call_depth_reached=self.max_call_depth_reached,
        )


def _parse_kv(items: Sequence[str], flag: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for item in items:
        try:
            name, value = item.split("=", 1)
        except ValueError:
            raise SystemExit("%s must be NAME=VALUE, got %r" % (flag, item)) from None
        values[name] = value
    return values


def _load_image_memory(name: str) -> LoadedMemory:
    """tools/sharc.py's Image._mem(), reused rather than duplicated -- it
    already knows how the DB name maps to out/sections/<name>/section_7_BLOB.bin.
    Imported lazily so a library caller (or a test building its own
    LoadedMemory) never pays for sharc.py's sqlite3/networkx imports."""
    import sharc

    img = sharc.load(name)
    if img.meta.get("cpu") == "coldfire":
        raise SystemExit(
            "sharc_run.py: %r is a ColdFire image; this runner only "
            "executes SHARC+ code" % name
        )
    return img._mem()


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("image", help='sharc.py image name, e.g. "dt2-1.16"')
    p.add_argument("--start", required=True, type=lambda x: int(x, 0))
    p.add_argument("--max-steps", type=int, default=100_000)
    p.add_argument(
        "--poke",
        dest="pokes",
        action="append",
        default=[],
        metavar="ADDR=HEX",
        help="seed a 32-bit DM word before running (repeatable)",
    )
    p.add_argument(
        "--reg",
        dest="regs",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="override one register's reset value (repeatable)",
    )
    p.add_argument(
        "--breakpoint",
        dest="breakpoints",
        action="append",
        default=[],
        type=lambda x: int(x, 0),
        metavar="PC_SW",
        help="stop before executing this short-word PC (repeatable)",
    )
    p.add_argument("--max-call-depth", type=int, default=64)
    p.add_argument(
        "--no-follow-calls",
        dest="follow_loaded_calls",
        action="store_false",
        default=True,
        help="stop at the first call into loaded code instead of entering it",
    )
    p.add_argument(
        "--allow-provisional",
        dest="provisional_forms",
        action="append",
        default=[],
        metavar="FORM",
        help="execute this decode-table-unconfirmed form instead of "
        "halting on it (repeatable); sets state.provisional_forms",
    )
    p.add_argument(
        "--explicit-memory-model",
        action="store_true",
        default=False,
        help="internal RAM never written by the loader or this run reads "
        "as 0 instead of Unknown; a core/system MMR with no known reset "
        "value and no --poke halts naming the register instead of forking "
        "(sharc_core.State.explicit_memory_model; off by default)",
    )
    p.add_argument(
        "--approx-recips",
        action="store_true",
        default=False,
        help="substitute a documented-but-unverified numeric model for "
        "recips's undocumented ROM seed instead of leaving it Unknown "
        "(sharc_core.State.approx_recips, same as tools/sharc_trace.py's "
        "own --approx-recips; off by default)",
    )
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)

    pokes: dict[int, int] = {}
    for name, value in _parse_kv(a.pokes, "--poke").items():
        # HEX per --help, but int(value, 0) also takes a "0x..."-prefixed
        # value without double-interpreting it.
        pokes[int(name, 0)] = (
            int(value, 0) if value.lower().startswith("0x") else int(value, 16)
        )

    regs: dict[str | int, int | str] = {}
    for name, value in _parse_kv(a.regs, "--reg").items():
        regs[name] = int(value, 0)

    memory = _load_image_memory(a.image)
    runner = Runner(
        memory,
        a.start,
        regs=regs,
        pokes=pokes,
        follow_loaded_calls=a.follow_loaded_calls,
        max_call_depth=a.max_call_depth,
        breakpoints=a.breakpoints,
        provisional_forms=a.provisional_forms,
        explicit_memory_model=a.explicit_memory_model,
        approx_recips=a.approx_recips,
    )
    result = runner.run(a.max_steps)

    if a.json:
        print(json.dumps(result.to_json(), indent=2))
    else:
        print("start:  %#x" % result.start_pc_sw)
        print("final:  %#x" % result.final_pc_sw)
        print("instructions: %d" % result.instructions)
        print(
            "elapsed: %.3fs (%.0f instr/s)"
            % (result.elapsed, result.instructions_per_second)
        )
        print("max call depth reached: %d" % result.max_call_depth_reached)
        print(
            "halt: %s at %#x (%s)%s"
            % (
                result.halt.reason,
                result.halt.pc_sw,
                result.halt.form,
                (": " + result.halt.text) if result.halt.text else "",
            )
        )
        print("top forms:")
        for form, count in result.form_counts.most_common(15):
            print("  %-16s %d" % (form, count))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
