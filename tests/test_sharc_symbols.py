"""Tests for tools/sharc_symbols.py, the SHARC+ program-address resolver.

Two kinds of test here:

* Rule unit tests (FuncMatch/LiteralAt/Offset/IVTSlot/EnclosingFunction)
  against small fake Image/reference objects -- no database needed, these
  always run.
* Resolution tests against the real DT2 1.16 / DN2 1.11 / DT2 1.15C /
  DN2 1.10E databases in out/sharcdb -- these use the `images` fixture,
  which skips (not fails) whenever one of those .sqlite files is absent, the
  same convention tests/test_sharcdb.py uses for the firmware-derived
  databases this repo never commits.
"""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

import sharc  # noqa: E402
import sharc_symbols  # noqa: E402

DB_DIR = os.path.join(ROOT, "out", "sharcdb")
IMAGE_NAMES = ["dt2-1.16", "dn2-1.11", "dt2-1.15C", "dn2-1.10E"]


def _db_path(name):
    return os.path.join(DB_DIR, name + ".sqlite")


@pytest.fixture(scope="module")
def images():
    missing = [n for n in IMAGE_NAMES if not os.path.exists(_db_path(n))]
    if missing:
        pytest.skip("out/sharcdb/*.sqlite missing for: %s" % ", ".join(missing))
    loaded = {n: sharc.load(n) for n in IMAGE_NAMES}
    yield loaded
    for img in loaded.values():
        img.close()


# --------------------------------------------------------------------------
# Rule unit tests: fake Image/reference objects, no database required.
# --------------------------------------------------------------------------


class _FakeRef:
    """Enough of tools/sharc.py's Image to drive FuncMatch.resolve() in
    isolation: `.name` plus a canned `.match()` result."""

    def __init__(self, name, matches):
        self.name = name
        self._matches = matches

    def match(self, other_img, func):
        return self._matches


class _FakeImg:
    """Enough of tools/sharc.py's Image to drive the sql()-based rules in
    isolation: `.name` plus a callback deciding what `.sql()` returns."""

    def __init__(self, name, sql_fn):
        self.name = name
        self._sql_fn = sql_fn

    def sql(self, query, *args):
        return self._sql_fn(query, args)


def test_funcmatch_no_hits_is_unresolved_not_an_exception():
    ref = _FakeRef("dt2-1.16", [])
    tgt = _FakeImg("dn2-1.11", lambda q, a: [])
    value, detail = sharc_symbols.FuncMatch(0x1C75D8).resolve(tgt, ref, {})
    assert value is None
    assert "no func_hash match" in detail


def test_funcmatch_ambiguous_hits_are_reported_not_guessed():
    ref = _FakeRef(
        "dt2-1.16",
        [
            {"entry_sw": "0x1c1000", "exact_match": False},
            {"entry_sw": "0x1c2000", "exact_match": False},
        ],
    )
    tgt = _FakeImg("other-image", lambda q, a: [])
    value, detail = sharc_symbols.FuncMatch(0x1C75D8).resolve(tgt, ref, {})
    assert value is None
    assert "2 distinct" in detail
    assert "0x1c1000" in detail and "0x1c2000" in detail


def test_funcmatch_unique_hit_resolves_and_reports_exactness():
    ref = _FakeRef("dt2-1.16", [{"entry_sw": "0x1c9e76", "exact_match": False}])
    tgt = _FakeImg("dn2-1.11", lambda q, a: [])
    value, detail = sharc_symbols.FuncMatch(0x1C75D8).resolve(tgt, ref, {})
    assert value == 0x1C9E76
    assert "reloc-only" in detail


def test_funcmatch_on_the_reference_image_checks_functions_table():
    ref = _FakeImg("dt2-1.16", lambda q, a: [])

    def sql_fn(query, *args):
        assert "functions" in query
        return []

    ref.sql = sql_fn
    value, detail = sharc_symbols.FuncMatch(0x1C75D8).resolve(ref, ref, {})
    assert value is None
    assert "missing from" in detail


def test_literalat_depends_on_unresolved_base():
    tgt = _FakeImg("x", lambda q, a: [])
    value, detail = sharc_symbols.LiteralAt("base", 4).resolve(
        tgt, None, {"base": None}
    )
    assert value is None
    assert "unresolved 'base'" in detail


def test_literalat_needs_exactly_one_row():
    tgt = _FakeImg("x", lambda q, a: [(1, "17a"), (2, "17a")])
    value, detail = sharc_symbols.LiteralAt("base", 4).resolve(
        tgt, None, {"base": 0x100}
    )
    assert value is None
    assert "2 literal(s)" in detail


def test_literalat_checks_the_expected_form():
    tgt = _FakeImg("x", lambda q, a: [(0x2412CC, "19a")])
    value, detail = sharc_symbols.LiteralAt("base", 4, form="17a").resolve(
        tgt, None, {"base": 0x100}
    )
    assert value is None
    assert "has form '19a', expected '17a'" in detail


def test_literalat_masks_a_sign_extended_value_to_unsigned_32_bit():
    # tools/sharcdb.py's extract_literal sign-extends; float_table_a
    # (0x8045a6c8) reads back negative from a signed column.
    tgt = _FakeImg("x", lambda q, a: [(0x8045A6C8 - (1 << 32), "17a")])
    value, detail = sharc_symbols.LiteralAt("base", 4, form="17a").resolve(
        tgt, None, {"base": 0x100}
    )
    assert value == 0x8045A6C8
    assert "0x8045a6c8" in detail


def test_offset_depends_on_unresolved_base():
    tgt = _FakeImg("x", lambda q, a: [])
    value, detail = sharc_symbols.Offset("base", 4).resolve(tgt, None, {"base": None})
    assert value is None
    assert "unresolved 'base'" in detail


def test_offset_requires_an_aligned_instruction():
    tgt = _FakeImg("x", lambda q, a: [])
    value, detail = sharc_symbols.Offset("base", 4).resolve(tgt, None, {"base": 0x100})
    assert value is None
    assert "not an aligned instruction start" in detail


def test_offset_resolves_when_aligned():
    tgt = _FakeImg("x", lambda q, a: [(1,)])
    value, detail = sharc_symbols.Offset("base", 4).resolve(tgt, None, {"base": 0x100})
    assert value == 0x104


def test_ivtslot_needs_exactly_one_root():
    tgt = _FakeImg("x", lambda q, a: [])
    value, detail = sharc_symbols.IVTSlot(15, "SECI").resolve(tgt, None, {})
    assert value is None
    assert "0 IVT root(s)" in detail


def test_enclosingfunction_depends_on_unresolved_base():
    tgt = _FakeImg("x", lambda q, a: [])
    value, detail = sharc_symbols.EnclosingFunction("base").resolve(
        tgt, None, {"base": None}
    )
    assert value is None
    assert "unresolved 'base'" in detail


def test_enclosingfunction_needs_exactly_one_enclosing_function():
    tgt = _FakeImg("x", lambda q, a: [])
    value, detail = sharc_symbols.EnclosingFunction("base").resolve(
        tgt, None, {"base": 0x100}
    )
    assert value is None
    assert "0 function(s) enclose" in detail


def test_device_of_recognizes_both_products():
    assert sharc_symbols.device_of("dt2-1.16") == "dt2"
    assert sharc_symbols.device_of("dn2-1.11") == "dn2"
    assert sharc_symbols.device_of("dt2-1.15C") == "dt2"
    assert sharc_symbols.device_of("something-else") is None


# --------------------------------------------------------------------------
# Real-database resolution tests.
# --------------------------------------------------------------------------

# Known-good DT2 1.16 addresses (docs/findings/06-sharc-engine-and-startup.md
# and docs/findings/11-sharc-cross-image-comparison.md), cross-checked
# against out/sharcdb/dt2-1.16.sqlite while tools/sharc_symbols.py's table
# was built. Kept in lockstep with SYMBOLS by test_every_symbol_is_checked.
KNOWN_DT2_116 = {
    "audio_task_fn": 0x1C75D8,
    "block_handler": 0x1C74CD,
    "command_dispatch_fn": 0x1C778A,
    "simd_helper": 0xB80105,
    "seci_dispatch": 0x1C0B7B,
    "seci_isr": 0x1C0B1D,
    "command_table": 0x25F7B0,
    "cmd_handler_0": 0x1C7524,
    "cmd_handler_1": 0x1C75D8,
    "cmd_handler_2": 0x1C763C,
    "cmd_handler_3": 0x1C7671,
    "ring_a": 0x261CC8,
    "ring_b": 0x261EC8,
    "ring_d": 0x263138,
    "ring_flag": 0x25F780,
    "command_word": 0x264220,
    "command_word_shift_src": 0x261CA4,
    "render_frame": 0x1C2B24,
    "unpack_track": 0x1C24E9,
    "slot_dispatch": 0x1C642A,
    "master_mix": 0x1C207B,
    "voice_render": 0x1C4ECF,
    "voice_render_tail": 0x1C4F81,
    "init": 0x1C15E3,
    "voice_alloc_scan": 0x1C149B,
    "voice_record_init_a": 0x1C7442,
    "voice_record_init_b": 0x1C4E70,
    "decimator": 0xB80000,
    "voice_records": 0x2412CC,
    "coeff_table": 0x25D940,
    "frame_workspace": 0x2412C8,
    "mix_table_base": 0x252D78,
    "source_words": 0x24EF2C,
    "float_table_a": 0x8045A6C8,
    "track_buffers": 0x252DF8,
    "selector_table": 0x2567C0,
    "machine_type_cache": 0x255970,
}

# DT2-only symbols: DN2's FM engine has no equivalent (docs/findings/11), so
# FuncMatch/LiteralAt correctly report these as unresolved on a dn2-* image.
DT2_ONLY = frozenset(
    {
        "render_frame",
        "unpack_track",
        "slot_dispatch",
        "master_mix",
        "voice_render",
        "voice_render_tail",
        "init",
        "voice_alloc_scan",
        "voice_record_init_a",
        "voice_record_init_b",
        "decimator",
        "voice_records",
        "coeff_table",
        "frame_workspace",
        "mix_table_base",
        "source_words",
        "float_table_a",
        "track_buffers",
        "selector_table",
        "machine_type_cache",
    }
)


def test_every_symbol_is_checked():
    # Guards KNOWN_DT2_116/DT2_ONLY above against silently going stale if
    # SYMBOLS grows.
    names = {name for name, _, _ in sharc_symbols.SYMBOLS}
    assert names == set(KNOWN_DT2_116)
    assert names >= DT2_ONLY


def test_dt2_116_resolves_every_symbol_to_its_known_address(images):
    profile = sharc_symbols.resolve(images["dt2-1.16"], device="dt2")
    assert profile.unresolved == []
    for name, addr in KNOWN_DT2_116.items():
        assert profile[name] == addr, "%s: got %r, want 0x%x" % (
            name,
            profile[name],
            addr,
        )


def test_dn2_11_required_common_symbols_resolve(images):
    profile = sharc_symbols.resolve(images["dn2-1.11"])
    assert profile.device == "dn2"
    for name, _, required in sharc_symbols.SYMBOLS:
        if "dn2" in required:
            assert profile[name] is not None, "%s should resolve on dn2-1.11" % name


def test_dn2_11_lacks_the_dt2_voice_engine(images):
    profile = sharc_symbols.resolve(images["dn2-1.11"])
    for name in DT2_ONLY:
        assert profile[name] is None, "%s unexpectedly resolved on dn2-1.11" % name
    assert set(profile.unresolved) == DT2_ONLY


def test_dn2_11_shared_addresses_differ_numerically_from_dt2(images):
    # Same rule, same relative position -- but DN2's own memory layout, not
    # DT2's copy-pasted.
    dt2 = sharc_symbols.resolve(images["dt2-1.16"])
    dn2 = sharc_symbols.resolve(images["dn2-1.11"])
    for name in (
        "ring_a",
        "ring_b",
        "ring_d",
        "ring_flag",
        "command_table",
        "command_word",
        "command_word_shift_src",
    ):
        assert dn2[name] is not None
        assert dn2[name] != dt2[name]


def test_dn2_11_audio_task_fn_opens_by_touching_ring_a(images):
    # Spot check by reading the matched code (per the task brief), not just
    # trusting the hash: DN2's audio_task_fn should still open the same way
    # DT2's does, addressing the *DN2* ring_a this resolver just computed.
    dn2 = sharc_symbols.resolve(images["dn2-1.11"])
    text = " ".join(m for _, m in images["dn2-1.11"].listing(dn2.audio_task_fn, 8))
    assert "0x%x" % dn2.ring_a in text


def test_dn2_11_block_handler_reads_ring_flag_before_dispatch(images):
    dn2 = sharc_symbols.resolve(images["dn2-1.11"])
    text = " ".join(m for _, m in images["dn2-1.11"].listing(dn2.block_handler, 60))
    assert "0x%x" % dn2.ring_flag in text
    assert "0x%x" % dn2.command_table in text


def test_dt2_15c_audio_task_hash_mismatch_fails_loudly(images):
    # Regression pin for a real, checked result: 1.15C's audio task does not
    # func_hash-match 1.16's (see docs/findings/06's function-bounds
    # corrections), so a REQUIRED_COMMON symbol comes back unresolved and
    # resolve() must raise rather than silently return a partial profile.
    with pytest.raises(sharc_symbols.SymbolResolutionError) as exc_info:
        sharc_symbols.resolve(images["dt2-1.15C"])
    assert "audio_task_fn" in str(exc_info.value)


def test_reference_image_device_is_dt2(images):
    profile = sharc_symbols.resolve(images["dt2-1.16"])
    assert profile.device == "dt2"


def test_report_lists_every_symbol(images):
    profile = sharc_symbols.resolve(images["dt2-1.16"])
    report = profile.report()
    for name in KNOWN_DT2_116:
        assert name in report
