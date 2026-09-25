"""Reusable frame-render/routine survey driver.

Promoted from a scratch driver (a lane-L7 survey of FUN_1c2b24, the frame
render root) into a general tool: start from an already-booted state (a
tools/sharc_run.py snapshot, or a fresh tools/sharc_harness.run_init()),
jump to an arbitrary root with Runner.fresh_call(), apply an accumulated
table of register/memory hypotheses each time a given pc is about to
execute, run with diagnose_unknown=True, and print a rich diagnosis of
whatever it stops on.

    uv run python tools/sharc_survey.py dt2-1.16 --root 0x1c2b24 \\
        --snapshot /path/to/init_cache.snap

    uv run python tools/sharc_survey.py dt2-1.16 --root 0x1c2b24 --run-init

    uv run python tools/sharc_survey.py dt2-1.16 --root 0x1c2b24 \\
        --snapshot /path/to/init_cache.snap --patch-table hyps.py

Workflow: run once, read the printed diagnosis (unknown bits, the last
instruction that wrote each one, a bounded backward slice from the stop,
and the call stack with function names), add a hypothesis to a patch-table
file for whatever it found, and re-run -- the patch table is applied fresh
on every pc recurrence, so an already-diagnosed stop is skipped over on the
next run and only a genuinely new one halts it. A patch-table file's
hypotheses accumulate; this tool itself never resumes from where it halted
(no ad hoc, unrepeatable in-memory state) -- see run_with_patches()'s and
--patch-table's docstrings.

Library use:

    import sharc_survey as sv
    stop = sv.survey(image="dt2-1.16", root=0x1c2b24, snapshot=path)
    print(stop.halt.reason)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import sharc  # noqa: E402
import sharc_run as sr  # noqa: E402
import sharc_trace as st  # noqa: E402

# {trigger pc_sw: [("reg", NAME, value) | ("mem", addr, value), ...]},
# applied the instant that pc is about to execute, every time it recurs --
# see run_with_patches().
PatchTable = dict[int, list[tuple[str, int | str, int]]]

DEFAULT_MAX_STEPS = 4_000_000
DEFAULT_SLICE_DEPTH = 6

# REG.BIT name (e.g. "ASTATX.BTF", as tools/sharc_run.py's Halt.unknowns
# reports it -- see its _ASTAT_BIT_NAME) -> the tools/sharc.py ASTAT group
# pseudo-register name it belongs to (e.g. "ASTAT.BTF"), composed from both
# modules' own bit tables rather than a third, hand-copied one: sharc_run's
# bit-number -> name and sharc's bit-number -> group must already agree on
# every bit number, or a Halt's own "ASTATX.BTF" text and sharc.py's
# defuse()/slice() group split would disagree about what that text means.
_UNKNOWN_RE = re.compile(r"^(ASTATX|ASTATY)\.(\w+)$")
_BIT_NAME_TO_GROUP = {
    name: sharc._ASTAT_GROUP_BIT[bit]
    for bit, name in sr._ASTAT_BIT_NAME.items()
    if bit in sharc._ASTAT_GROUP_BIT
}


def _astat_groups_from_unknowns(unknowns: Sequence[str]) -> set[str]:
    """The tools/sharc.py ASTAT group(s) (see Image._ASTAT_GROUPS)
    implicated by a fork Halt's own unknowns tuple -- e.g. {"BTF"} for
    ("cond=TF", "ASTATX.BTF"). Empty when nothing in UNKNOWNS names an
    ASTATX/ASTATY bit at all (a MODE1-only fork, or diagnose_unknown gave
    up entirely -- see _fork_diagnosis()'s own docstring)."""
    groups: set[str] = set()
    for text in unknowns:
        match = _UNKNOWN_RE.match(text)
        if match is None:
            continue
        group = _BIT_NAME_TO_GROUP.get(match.group(2))
        if group is not None:
            groups.add(group)
    return groups


def load_patch_table(path: str | None) -> PatchTable:
    """A PatchTable from PATH: a .json file (keys are decimal or "0x..."
    strings, values are [kind, target, value] lists -- JSON has no tuples)
    or a .py file defining a module-level ``PATCH_TABLE`` dict (the same
    convention the scratch survey driver this tool was promoted from used:
    edit a Python literal between runs, keys already ints, values already
    tuples). None (no --patch-table) is an empty table."""
    if path is None:
        return {}
    if path.endswith(".json"):
        with open(path) as fh:
            raw = json.load(fh)
        table: PatchTable = {}
        for key, entries in raw.items():
            pc = int(key, 0) if isinstance(key, str) else int(key)
            table[pc] = [
                (
                    kind,
                    target,
                    int(value, 0) if isinstance(value, str) else int(value),
                )
                for kind, target, value in entries
            ]
        return table
    spec = importlib.util.spec_from_file_location("_sharc_survey_patch_table", path)
    if spec is None or spec.loader is None:
        raise SystemExit("--patch-table: cannot load %r" % path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    py_table: PatchTable | None = getattr(module, "PATCH_TABLE", None)
    if py_table is None:
        raise SystemExit("--patch-table: %r defines no PATCH_TABLE dict" % path)
    return py_table


def apply_patches(runner: sr.Runner, patch_table: PatchTable) -> list[str]:
    """Apply every PATCH_TABLE entry for runner.state.pc_sw, if any (in
    place, before that instruction executes); returns a one-line-per-patch
    description of what was applied, for the caller to log."""
    entries = patch_table.get(runner.state.pc_sw)
    if not entries:
        return []
    applied = []
    for kind, target, value in entries:
        if kind == "reg":
            if isinstance(target, str):
                code, name = st.UREG_CODES[target], target
            else:
                code, name = target, st.UREG_NAMES[target]
            runner.state.uregs[code] = st.Const(value)
            applied.append("reg %s = %#x" % (name, value))
        elif kind == "mem":
            if not isinstance(target, int):
                raise ValueError(
                    "mem patch target must be an int address, got %r" % (target,)
                )
            ok = st._dm_write(runner.state, target, 4, st.Const(value & 0xFFFFFFFF))
            applied.append(
                "mem[%#x] = %#x%s" % (target, value, "" if ok else " (FAILED)")
            )
        else:
            raise ValueError("bad patch kind %r at pc=%#x" % (kind, runner.state.pc_sw))
    return applied


def run_with_patches(
    runner: sr.Runner, patch_table: PatchTable, max_steps: int, *, verbose: bool = False
) -> sr.Halt:
    """Step RUNNER forward, applying PATCH_TABLE's entries for the current
    pc immediately before it executes (every time that pc recurs, e.g. on
    each iteration of a loop) until it halts or MAX_STEPS is used up.
    Returns the Halt (a "max-steps" one if the budget ran out)."""
    steps = 0
    while steps < max_steps:
        applied = apply_patches(runner, patch_table)
        if verbose and applied:
            print("  patched at %#x: %s" % (runner.state.pc_sw, "; ".join(applied)))
        try:
            runner.step()
        except sr.Halt as exc:
            return exc
        steps += 1
    return sr.Halt("max-steps", runner.state.pc_sw)


@dataclass
class SurveyStop:
    """One survey run's outcome: the Halt itself, the Runner that produced
    it (still positioned at the halt, so its state/call_stack/_last_writer
    are all inspectable), and the Image used to diagnose it."""

    halt: sr.Halt
    runner: sr.Runner
    img: sharc.Image
    instructions: int
    elapsed: float

    @property
    def category(self) -> str:
        """A short label for the halt's kind, beyond Halt.reason's own free
        text: "return-mismatch" gets its own category (see the module
        docstring and _check_return_target() in sharc_core/sequencer.py) --
        a prior poke or an inherited call_stack entry disagreeing with what
        the sequencer's own I12/M14 computed is a sign that THIS RUN's
        state is wrong, not a new thing to explain about the firmware."""
        reason = self.halt.reason
        if (
            reason.startswith("return target ")
            and " differs from recorded return " in reason
        ):
            return "return-mismatch"
        if reason.startswith("fork"):
            return "fork"
        if reason == "mmr":
            return "mmr"
        if reason == "watchpoint":
            return "watchpoint"
        if reason == "return without followed call":
            return "frame-returned"
        if reason == "max-steps":
            return "max-steps"
        return reason


def _call_stack_lines(img: sharc.Image, runner: sr.Runner) -> list[str]:
    lines = []
    for addr in runner.state.call_stack:
        fn = img.func(addr)
        name = fn["name"] if fn else None
        lines.append("0x%x%s" % (addr, (" (%s)" % name) if name else ""))
    return lines


def _last_writer_lines(runner: sr.Runner) -> list[str]:
    lines = []
    for name in sr._DIAGNOSE_TRACKED_UREGS:
        pc = runner._last_writer.get(name)
        lines.append(
            "%-8s %s"
            % (name, ("0x%x" % pc) if pc is not None else "(never written this run)")
        )
    return lines


def _print_bounded_slice(
    img: sharc.Image, pc_sw: int, *, reg: str | None, depth: int, label: str
) -> None:
    print("  %s:" % label)
    try:
        text = img.print_slice(pc_sw, reg=reg, depth=depth)
    except Exception as exc:  # noqa: BLE001 -- report and move on, never abort the survey
        print("    (unavailable: %s)" % exc)
        return
    if not text:
        print("    (empty)")


def report_stop(stop: SurveyStop, *, slice_depth: int = DEFAULT_SLICE_DEPTH) -> None:
    """Print pc, form, unknown bits, last flag writer, a bounded
    img.print_slice from the stop, and the call stack with function names
    -- the fixed shape every survey stop gets, regardless of category."""
    halt, runner, img = stop.halt, stop.runner, stop.img
    print(
        "instructions=%d elapsed=%.3fs (%.0f instr/s)"
        % (
            stop.instructions,
            stop.elapsed,
            stop.instructions / stop.elapsed if stop.elapsed > 0 else float("inf"),
        )
    )
    print("halt: reason=%s pc=%#x form=%s" % (halt.reason, halt.pc_sw, halt.form))
    if halt.text:
        print("  text: %s" % halt.text)
    print("  category: %s" % stop.category)
    if halt.unknowns:
        print("  unknowns:")
        for text in halt.unknowns:
            print("    %s" % text)
    print("  last flag writer:")
    for line in _last_writer_lines(runner):
        print("    %s" % line)
    fn = img.func(halt.pc_sw)
    print("  function: %s" % (fn["name"] if fn else "(none)"))
    _print_bounded_slice(
        img, halt.pc_sw, reg=None, depth=slice_depth, label="slice (all inputs)"
    )
    if stop.category == "fork":
        groups = _astat_groups_from_unknowns(halt.unknowns)
        if len(groups) == 1:
            group = next(iter(groups))
            reg = sharc._ASTAT_GROUP_REG[group]
            _print_bounded_slice(
                img,
                halt.pc_sw,
                reg=reg,
                depth=slice_depth,
                label="slice (reg=%r -- the precise group this cond reads)" % reg,
            )
        elif groups:
            print(
                "  (cond reads more than one ASTAT group: %s -- see the "
                "whole-register slice above)" % ", ".join(sorted(groups))
            )
    elif stop.category == "mmr":
        match = re.search(r"0x[0-9a-fA-F]+", halt.text)
        if match:
            addr = int(match.group(0), 16)
            print("  writers(%#x): %s" % (addr, img.writers(addr)))
            print("  readers(%#x): %s" % (addr, img.readers(addr)))
            print("  refs(%#x): %s" % (addr, img.refs(addr)))
    elif stop.category == "return-mismatch":
        print(
            "  a prior poke or state is wrong: the sequencer's own I12/M14 "
            "computed a return address that disagrees with the pushed "
            "call_stack entry (see run_init()/fresh_call()'s own note on "
            "why pending/stopped/call_stack must all be cleared for a new "
            "root) -- this is not new firmware behaviour to explain."
        )
    print("  call stack (return addrs, innermost last):")
    stack_lines = _call_stack_lines(img, runner)
    if not stack_lines:
        print("    (empty)")
    for line in stack_lines:
        print("    %s" % line)


def survey(
    *,
    image: str,
    root: int,
    snapshot: str | None = None,
    run_init: bool = False,
    regs: Mapping[str | int, int | str] | None = None,
    return_address: int | None = None,
    patch_table: PatchTable | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    max_call_depth: int = 64,
    explicit_memory_model: bool = True,
    approx_recips: bool = True,
    follow_loaded_calls: bool = True,
    verbose: bool = False,
) -> SurveyStop:
    """Run ROOT from SNAPSHOT, a fresh run_init(), or (neither given) a
    bare Runner started directly at ROOT, applying PATCH_TABLE, and return
    the resulting SurveyStop. See the module docstring for the two
    documented starting points and why a snapshot/run_init Runner needs
    Runner.fresh_call() rather than just overwriting its pc_sw."""
    img = sharc.load(image)
    memory = sr._load_image_memory(image)
    patch_table = patch_table or {}

    if snapshot is not None:
        base = sr.load_snapshot(snapshot, memory)
        runner = base.fresh_call(
            root, regs=regs, return_address=return_address, diagnose_unknown=True
        )
    elif run_init:
        import sharc_harness as h  # lazy: only this path needs the harness

        init = h.run_init(memory, image)
        if not init.ran or init.runner is None:
            raise SystemExit("run_init failed: %s" % init.error)
        runner = init.runner.fresh_call(
            root, regs=regs, return_address=return_address, diagnose_unknown=True
        )
    else:
        runner = sr.Runner(
            memory,
            root,
            regs=regs,
            follow_loaded_calls=follow_loaded_calls,
            max_call_depth=max_call_depth,
            explicit_memory_model=explicit_memory_model,
            approx_recips=approx_recips,
            diagnose_unknown=True,
        )
        if return_address is not None:
            runner.state.call_stack = [return_address]

    t0 = time.perf_counter()
    halt = run_with_patches(runner, patch_table, max_steps, verbose=verbose)
    elapsed = time.perf_counter() - t0
    return SurveyStop(
        halt=halt,
        runner=runner,
        img=img,
        instructions=runner.instructions,
        elapsed=elapsed,
    )


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("image", help='sharc.py image name, e.g. "dt2-1.16"')
    p.add_argument(
        "--root", required=True, type=lambda x: int(x, 0), help="pc_sw to call"
    )
    start = p.add_mutually_exclusive_group()
    start.add_argument(
        "--snapshot",
        metavar="PATH",
        help="sr.save_snapshot() file to fresh_call() from",
    )
    start.add_argument(
        "--run-init",
        action="store_true",
        help="run tools/sharc_harness.run_init() first",
    )
    p.add_argument(
        "--reg",
        dest="regs",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="register value for the fresh call (repeatable)",
    )
    p.add_argument(
        "--return-address",
        type=lambda x: int(x, 0),
        default=None,
        help="push this pc_sw as the call's return address",
    )
    p.add_argument("--patch-table", metavar="PATH", help=".json or .py PatchTable file")
    p.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    p.add_argument("--max-call-depth", type=int, default=64)
    p.add_argument("--slice-depth", type=int, default=DEFAULT_SLICE_DEPTH)
    p.add_argument(
        "--no-explicit-memory-model",
        dest="explicit_memory_model",
        action="store_false",
        default=True,
        help="bare-Runner mode only: do not assume unwritten internal RAM reads as 0",
    )
    p.add_argument(
        "--no-approx-recips",
        dest="approx_recips",
        action="store_false",
        default=True,
        help="bare-Runner mode only: do not substitute recips's approximate seed",
    )
    p.add_argument("--verbose", action="store_true", help="log every applied patch")
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)

    regs: dict[str | int, int | str] = {}
    for name, value in sr._parse_kv(a.regs, "--reg").items():
        regs[name] = int(value, 0)

    patch_table = load_patch_table(a.patch_table)

    stop = survey(
        image=a.image,
        root=a.root,
        snapshot=a.snapshot,
        run_init=a.run_init,
        regs=regs,
        return_address=a.return_address,
        patch_table=patch_table,
        max_steps=a.max_steps,
        max_call_depth=a.max_call_depth,
        explicit_memory_model=a.explicit_memory_model,
        approx_recips=a.approx_recips,
        verbose=a.verbose,
    )

    if a.json:
        payload = stop.halt.to_json()
        payload["category"] = stop.category
        payload["instructions"] = stop.instructions
        payload["elapsed_s"] = stop.elapsed
        payload["last_writer"] = dict(stop.runner._last_writer)
        payload["call_stack"] = _call_stack_lines(stop.img, stop.runner)
        print(json.dumps(payload, indent=2))
    else:
        report_stop(stop, slice_depth=a.slice_depth)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
