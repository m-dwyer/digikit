"""The SHARC core's own tests, run under PyPy 3.11.

tools/sharc_core and tools/sharc_run.py are written to also run under PyPy
(see [tool.mypy]'s python_version = "3.11" in pyproject.toml, checked by
tests/test_types.py), since a symbolic trace or a concrete run is pure
Python arithmetic that benefits from PyPy's JIT. This test proves that
claim by actually running the core's test suite under a PyPy interpreter,
in a subprocess, instead of just type-checking against 3.11.

The file list below is every required test file that exercises sharc_core/
sharc_trace/sharc_run without a CPython-only dependency (unicorn, capstone,
pypcode); tools/sharc.py (imported by test_sharc_contract.py) needs
networkx, a pure-Python package that runs fine under PyPy, so it is
installed alongside pytest.
"""

import os
import shutil
import subprocess
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PYPY_TESTS = [
    "tests/test_sharc_compute_table.py",
    "tests/test_sharc_trace.py",
    "tests/test_sharc_trace_mult.py",
    "tests/test_sharc_trace_alu.py",
    "tests/test_sharc_trace_forms.py",
    "tests/test_sharc_trace_simd.py",
    "tests/test_sharc_run.py",
    "tests/test_sharc_contract.py",
]


@pytest.mark.slow
def test_sharc_core_under_pypy():
    if shutil.which("uv") is None:
        pytest.skip("uv not on PATH")
    start = time.perf_counter()
    result = subprocess.run(
        [
            "uv",
            "run",
            "--no-project",
            "--python",
            "pypy3.11",
            "--with",
            "pytest",
            "--with",
            "networkx",
            "python",
            "-m",
            "pytest",
            *PYPY_TESTS,
            "-q",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - start
    assert result.returncode == 0, "%s\n%s\n(%.1fs)" % (
        result.stdout,
        result.stderr,
        elapsed,
    )
