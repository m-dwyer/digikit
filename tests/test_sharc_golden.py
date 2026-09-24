"""Golden outputs of the SHARC core on the DT2 1.16 image.

Six outputs are hashed and compared with tests/golden/sharc_golden.json:
coverage for the render path and all roots, the concrete runner from the
voice and frame renders, and two symbolic trace summaries. Any difference
is a behaviour change. When the change is intended, update the hashes and
say why in the commit message:

    uv run python tests/test_sharc_golden.py --update

Only hashes are stored, because the outputs derive from Elektron's
firmware. The test skips when the image is absent or is not the one the
hashes were made from.
"""

import contextlib
import hashlib
import importlib
import io
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
BLOB = "out/sections/dt2-1.16/section_7_BLOB.bin"
GOLDEN = os.path.join(ROOT, "tests", "golden", "sharc_golden.json")

TRACE = [BLOB, "--blob", "--max-states", "64", "--summary"]
CASES = {
    "cov_render": ("sharc_coverage", ["dt2-1.16", "--root", "0x1c2b24", "--json"]),
    "cov_all": ("sharc_coverage", ["dt2-1.16", "--all-roots", "--json"]),
    "run_voice": (
        "sharc_run",
        ["dt2-1.16", "--start", "0x1c4ecf", "--max-steps", "200000", "--json"],
    ),
    "run_frame": (
        "sharc_run",
        ["dt2-1.16", "--start", "0x1c2b24", "--max-steps", "200000", "--json"],
    ),
    "trace_voice": (
        "sharc_trace",
        [*TRACE, "--start", "0x1c4ecf", "--max-steps", "2000"],
    ),
    "trace_frame": (
        "sharc_trace",
        [
            *TRACE,
            "--start",
            "0x1c2b24",
            "--max-steps",
            "5000",
            "--allow-provisional-form",
            "14d",
        ],
    ),
}
# Wall-clock fields vary run to run.
RUN_TIMING = ("elapsed_s", "instructions_per_second")


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def image_sha256():
    path = os.path.join(ROOT, BLOB)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as fh:
        return _sha256(fh.read())


def output(name):
    """The case's CLI output, exactly as printed, run in-process."""
    module_name, argv = CASES[name]
    if TOOLS not in sys.path:
        sys.path.insert(0, TOOLS)
    module = importlib.import_module(module_name)
    text = io.StringIO()
    cwd = os.getcwd()
    os.chdir(ROOT)  # the tools resolve out/ relative to the working directory
    try:
        with contextlib.redirect_stdout(text):
            module.main(argv)
    finally:
        os.chdir(cwd)
    result = text.getvalue()
    if module_name == "sharc_run":
        data = json.loads(result)
        for key in RUN_TIMING:
            data.pop(key, None)
        result = json.dumps(data, sort_keys=True) + "\n"
    return result


def load_golden():
    with open(GOLDEN) as fh:
        return json.load(fh)


@pytest.mark.parametrize("name", sorted(CASES))
def test_golden(name):
    golden = load_golden()
    image = image_sha256()
    if image is None:
        pytest.skip("firmware image %s is absent" % BLOB)
    if image != golden["image_sha256"]:
        pytest.skip("golden hashes are for image %s" % golden["image_sha256"])
    actual = _sha256(output(name).encode())
    assert actual == golden["outputs"][name], (
        "%s changed. If intended: uv run python tests/test_sharc_golden.py --update"
        % name
    )


def update():
    image = image_sha256()
    if image is None:
        raise SystemExit("firmware image %s is absent" % BLOB)
    golden = {
        "image_sha256": image,
        "outputs": {name: _sha256(output(name).encode()) for name in sorted(CASES)},
    }
    os.makedirs(os.path.dirname(GOLDEN), exist_ok=True)
    with open(GOLDEN, "w") as fh:
        json.dump(golden, fh, indent=1, sort_keys=True)
        fh.write("\n")
    print(json.dumps(golden, indent=1, sort_keys=True))


if __name__ == "__main__":
    if sys.argv[1:] != ["--update"]:
        raise SystemExit("usage: python tests/test_sharc_golden.py --update")
    update()
