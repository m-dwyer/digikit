"""Ruff lint and format checks for the files that have been cleaned.

The rules live in pyproject.toml. Add a path here once it passes both
`ruff check` and `ruff format --check`; the list should only grow.
"""

import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CLEAN = [
    "emu/dspi2.py",
    "emu/dspiframe.py",
    "tests/test_dspi2.py",
    "tests/test_lint.py",
    "tests/test_pypy.py",
    "tests/test_sharc_compute_table.py",
    "tests/test_sharc_contract.py",
    "tests/test_sharc_disasm.py",
    "tests/test_sharc_golden.py",
    "tests/test_sharc_run.py",
    "tests/test_sharc_symbols.py",
    "tests/test_sharc_trace_alu.py",
    "tests/test_sharc_trace_forms.py",
    "tests/test_sharc_trace_mult.py",
    "tests/test_sharc_trace_simd.py",
    "tests/test_sharcldr.py",
    "tests/test_types.py",
    "tools/sharc.py",
    "tools/sharc_contract.py",
    "tools/sharc_core",
    "tools/sharc_coverage.py",
    "tools/sharc_disasm.py",
    "tools/sharc_run.py",
    "tools/sharc_symbols.py",
    "tools/sharcldr.py",
]


def ruff(*args):
    return subprocess.run(
        [sys.executable, "-m", "ruff", *args, *CLEAN],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "args",
    [("check", "--no-cache"), ("format", "--check", "--no-cache")],
    ids=["check", "format"],
)
def test_ruff(args):
    result = ruff(*args)
    assert result.returncode == 0, result.stdout + result.stderr
