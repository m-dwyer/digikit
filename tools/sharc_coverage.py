"""Semantic coverage of tools/sharc_trace.py over a SHARC program database.

Answers "which real instructions would the tracer refuse to execute?" from
the database alone, without running a trace:

- form: the instruction's form has no branch in sharc_trace._execute and
  falls through to "unsupported form".
- provisional: the form's decode confidence is 'uncertain', so _execute
  stops unless the form is passed as a provisional form.
- compute: the compute field raises in sharc_trace._compute or
  _shift_immediate (probed with every UREG = 1.0f).

Other stops are not listed (SIMD-sensitive registers, RTI, circular
modify, long-word access, unresolved predicates): they depend on state.

    uv run python tools/sharc_coverage.py dt2-1.16 --root 0x1c2b24
    uv run python tools/sharc_coverage.py dt2-1.16 --all-roots --json
"""

from __future__ import annotations

import argparse
import ast
import collections
import inspect
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import networkx as nx  # noqa: E402
import sharc  # noqa: E402
import sharc_trace  # noqa: E402

ONE = sharc_trace.Const(0x3F800000)
SHIFT_IMMEDIATE_FORMS = ("6b_shiftimm", "6a_mem")
SHORT_COMPUTE_FORMS = ("2c",)


def handled_forms():
    """Form names that sharc_trace._execute compares `name` against."""
    tree = ast.parse(inspect.getsource(sharc_trace._execute))
    forms = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare) or not isinstance(node.left, ast.Name):
            continue
        if node.left.id != "name":
            continue
        for comparator in node.comparators:
            for const in ast.walk(comparator):
                if isinstance(const, ast.Constant) and isinstance(const.value, str):
                    forms.add(const.value)
    return forms


def function_set(img, root=None, all_roots=False):
    """Function entries in scope: reach from ROOT over call and jump edges,
    the database's reach table for ALL_ROOTS, or None for the whole image."""
    if all_roots:
        return {r[0] for r in img.sql("SELECT DISTINCT function_sw FROM reach WHERE image=?", img.name)}
    if root is None:
        return None
    graph = nx.DiGraph()
    graph.add_edges_from(img.sql(
        "SELECT DISTINCT from_function, to_function FROM edges WHERE image=? "
        "AND from_function IS NOT NULL AND to_function IS NOT NULL "
        "AND kind IN ('call', 'jump', 'cond_jump')", img.name))
    graph.add_node(root)
    return nx.descendants(graph, root) | {root}


def probe_compute(form, fields):
    """None when the compute field decodes, else the tracer's error text."""
    values = {code: ONE for code in range(256)}
    try:
        if form in SHIFT_IMMEDIATE_FORMS:
            sharc_trace._shift_immediate(fields, values)
            return None
        if form in SHORT_COMPUTE_FORMS:
            sharc_trace._compute(fields, True, values)
            return None
        if "compute" in fields and "compute[22:16]" not in fields:
            fields = dict(fields)
            fields["compute[22:16]"] = fields["compute"] >> 16
            fields["compute[15:0]"] = fields["compute"] & 0xFFFF
        if "compute[22:16]" not in fields:
            return None
        sharc_trace._compute(fields, False, values, {}, approx_recips=True)
    except Exception as error:  # the tracer signals unsupported encodings with ValueError
        return str(error)
    return None


def coverage(img, functions=None):
    """Gap rows {kind, form, detail, count, example_sw} for real (aligned)
    instructions in FUNCTIONS (None = whole image), most frequent first."""
    handled = handled_forms()
    gaps = collections.OrderedDict()
    total = 0
    rows = img.sql(
        "SELECT sw, form, fields, confidence, function_sw FROM insn WHERE image=? AND aligned=1", img.name)
    for sw, form, fields, confidence, function_sw in rows:
        if functions is not None and function_sw not in functions:
            continue
        total += 1
        if form not in handled:
            key = ("form", form, "unsupported form")
        elif confidence != "confident":
            key = ("provisional", form, "decode confidence %s" % confidence)
        else:
            detail = probe_compute(form, json.loads(fields or "{}"))
            if detail is None:
                continue
            key = ("compute", form, detail)
        entry = gaps.setdefault(key, {"kind": key[0], "form": key[1], "detail": key[2],
                                      "count": 0, "example_sw": "0x%x" % sw})
        entry["count"] += 1
    ordered = sorted(gaps.values(), key=lambda g: (-g["count"], g["kind"], g["form"], g["detail"]))
    return {"image": img.name, "instructions": total,
            "gap_instructions": sum(g["count"] for g in ordered), "gaps": ordered}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("image", help="database name, e.g. dt2-1.16")
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--root", type=lambda s: int(s, 0), help="function reach from this entry sw")
    scope.add_argument("--all-roots", action="store_true", help="the database's reach table")
    parser.add_argument("--kind", choices=("form", "provisional", "compute"), help="show one kind only")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--fail", action="store_true", help="exit 1 when any gap remains")
    args = parser.parse_args(argv)

    img = sharc.load(args.image)
    result = coverage(img, function_set(img, args.root, args.all_roots))
    if args.kind:
        result["gaps"] = [g for g in result["gaps"] if g["kind"] == args.kind]
        result["gap_instructions"] = sum(g["count"] for g in result["gaps"])
    if args.json:
        print(json.dumps(result, indent=1))
    else:
        print("%s: %d instructions, %d not executable, %d distinct gaps"
              % (result["image"], result["instructions"], result["gap_instructions"], len(result["gaps"])))
        for g in result["gaps"]:
            print("%6d  %-11s %-12s %-10s %s" % (g["count"], g["kind"], g["form"], g["example_sw"], g["detail"]))
    return 1 if args.fail and result["gaps"] else 0


if __name__ == "__main__":
    sys.exit(main())
