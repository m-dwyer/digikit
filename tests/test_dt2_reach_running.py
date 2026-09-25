"""Tests for tools/dt2_reach_running.py's pure helpers: the MAIN_OS_RUNNING
check computation and the raw-memory reader it is built on. No firmware or
snapshot needed -- see tests/test_sharc_capture_run.py's own docstring for
the project convention this follows (pure helpers always run; anything that
needs a real Machine/snapshot is exercised by hand, per the module's own
docstring).
"""

import os
import struct
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools"))
import dt2_reach_running as rr  # noqa: E402


class _FakeUc:
    """Just enough of Unicorn's `uc` surface for u32()/observe(): a flat
    byte-addressed memory dict, read 4 (or 1) bytes at a time big-endian."""

    def __init__(self, words):
        self._words = dict(words)

    def mem_read(self, addr, size):
        if size == 4:
            value = self._words.get(addr, 0)
            return struct.pack(">I", value)
        raise NotImplementedError(size)


class _FakeMachine:
    def __init__(self, words):
        self.uc = _FakeUc(words)


class _FakeProfile:
    def __init__(self, **kw):
        self.current_tcb = kw.get("current_tcb")
        self.ready_cursor = kw.get("ready_cursor")
        self.intro_pit3_isr = kw.get("intro_pit3_isr", 0x400D0668)


class _FakeTimers:
    def __init__(self, fired):
        self._fired = dict(fired)

    @property
    def fired(self):
        return dict(self._fired)


class U32Test(unittest.TestCase):
    def test_reads_big_endian_word(self):
        m = _FakeMachine({0x1000: 0xDEADBEEF})
        self.assertEqual(rr.u32(m, 0x1000), 0xDEADBEEF)

    def test_missing_address_reads_zero_not_none(self):
        # _FakeUc defaults an unlisted address to 0, matching a real,
        # mapped-but-untouched guest page -- the try/except in u32() is for
        # an UNMAPPED address (mem_read raising), not a zero value.
        m = _FakeMachine({})
        self.assertEqual(rr.u32(m, 0x2000), 0)

    def test_unmapped_address_returns_none(self):
        class _RaisingUc:
            def mem_read(self, addr, size):
                raise RuntimeError("unmapped")

        class _RaisingMachine:
            uc = _RaisingUc()

        self.assertIsNone(rr.u32(_RaisingMachine(), 0x3000))


class ObserveTest(unittest.TestCase):
    """observe()'s main_os_running verdict must come from timers.fired's
    STRING-keyed ("PIT3", "DTIM3") property, not a raw Pits/Dtims object's
    own int-keyed counters -- reading the latter here was a bug (always 0,
    so main_os_running could never turn True) that this test pins."""

    def _profile(self, vec208_isr=0x400D0668):
        return _FakeProfile(
            current_tcb=None, ready_cursor=None, intro_pit3_isr=vec208_isr
        )

    def _ev(self):
        return {"tasks": [1, 2], "switch": {1: 1}}

    def test_all_checks_true_reports_running(self):
        m = _FakeMachine({0x40000340: 0x40133518, rr.KIT_PTR: 0x426532B8})
        profile = self._profile()
        mark = {"mainloop": 1, "job_pump": 3}
        timers = _FakeTimers({"PIT3": 418, "DTIM3": 428})
        obs = rr.observe(m, self._ev(), profile, mark, timers)
        self.assertTrue(obs["main_os_running"])
        self.assertTrue(all(obs["checks"].values()))
        self.assertEqual(obs["kit_ptr"], 0x426532B8)
        self.assertEqual(obs["tasks_created"], 2)

    def test_missing_dtim3_keeps_it_false(self):
        m = _FakeMachine({0x40000340: 0x40133518})
        profile = self._profile()
        mark = {"mainloop": 1, "job_pump": 3}
        timers = _FakeTimers({"PIT3": 418})  # no DTIM3 key at all
        obs = rr.observe(m, self._ev(), profile, mark, timers)
        self.assertFalse(obs["main_os_running"])
        self.assertFalse(obs["checks"]["DTIM3 firing"])

    def test_vector_208_still_on_intro_isr_keeps_it_false(self):
        # vec208 equal to the intro's own ISR means the display module has
        # not yet reclaimed the vector -- see emu/pit.py's intro_running().
        isr = 0x400D0668
        m = _FakeMachine({0x40000340: isr})
        profile = self._profile(vec208_isr=isr)
        mark = {"mainloop": 1, "job_pump": 3}
        timers = _FakeTimers({"PIT3": 1, "DTIM3": 1})
        obs = rr.observe(m, self._ev(), profile, mark, timers)
        self.assertFalse(obs["checks"]["vector 208 handed to display"])
        self.assertFalse(obs["main_os_running"])

    def test_int_keyed_fired_never_satisfies_the_check(self):
        # Regression for the bug this module's own observe() docstring
        # describes: passing a raw Pits/Dtims-style int-keyed Counter
        # instead of the combined Timers' string-keyed property must not
        # accidentally read as "firing" through some other coincidence.
        m = _FakeMachine({0x40000340: 0x40133518})
        profile = self._profile()
        mark = {"mainloop": 1, "job_pump": 3}
        timers = _FakeTimers({3: 418, 0: 428})  # wrong (int) keys
        obs = rr.observe(m, self._ev(), profile, mark, timers)
        self.assertFalse(obs["checks"]["PIT3 firing"])
        self.assertFalse(obs["checks"]["DTIM3 firing"])
        self.assertFalse(obs["main_os_running"])


if __name__ == "__main__":
    unittest.main()
