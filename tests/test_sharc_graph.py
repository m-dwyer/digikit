"""Checks for tools/sharc.py's networkx graph layer (cfg/callgraph/defuse/
slice) against the real dt2-1.16 program database (out/sharcdb/
dt2-1.16.sqlite -- see the worktree's `out`/`sections` symlinks). Skipped,
not marked slow: it only opens the already-built database, the same thing
tools/sharc.py itself does by default (see tests/test_sharc_contract.py).

FUN_1c4f81 (0x1c4f81-0x1c5334, "interpolating table lookup / wavetable
oscillator" per its `functions.label`) is used throughout: its two
conditionally-executed clears at 0x1c5008 and 0x1c5048 ("IF SV DM(I4, M1)
u=0 = M13") are docs/findings/06's per-track record +0x1b8 field, guarded by
a `leftz` a few instructions earlier -- a real slice question, not a
synthetic one.
"""

import os
import pathlib
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools"))
import sharc  # noqa: E402

DT2_116_DB = pathlib.Path("out/sharcdb/dt2-1.16.sqlite")

FUN_1C4F81 = 0x1C4F81  # entry of the wavetable-oscillator function
CLEAR_SITE_1 = 0x1C5008
CLEAR_SITE_2 = 0x1C5048
COND_SW = 0x1C5006  # `R2 = leftz(R2, R0)`, sets the SV flag CLEAR_SITE_1 reads
LOAD_SW = 0x1C5004  # `R2 = DM(I4, M4) u=0`, defines the R2 leftz consumes
LIT_SW = 0x1C5002  # `M4 = 0x1bc`, defines the M4 the load above uses

# A real Type18a system bit-test/xor-test (bop 4/5) and the BTF-reading
# branch right after it, in dt2-1.16's own image: FUN_0xb88200's
# `bit-test(MODE1STK, mask=0x10000) -> BTF` at BTF_WRITE_SW, consumed by
# `JUMP IF TF non-delayed` at BTF_BRANCH_SW. Before this file's own
# per-group ASTAT modelling, defuse()'s whole-register "ASTAT" rule had no
# def at all for a Type18a bit-test (tools/sharcdb.py's register_effects
# records no regdef row for one -- it only reads a source into BTF, never
# writes a data register), so slice(reg="ASTAT") here used to walk straight
# past it to the last unrelated compute (BTF_UNRELATED_COMPUTE_SW, an R14
# ALU/whatever compute architecturally unable to touch BTF -- PRM: "BTF is
# unaffected" by every ALU op).
FUN_B88200 = 0xB88200
BTF_WRITE_SW = 0xB883A5
BTF_BRANCH_SW = 0xB883A8
BTF_UNRELATED_COMPUTE_SW = 0xB882F6

# A real float multiply and the MN-reading (`IF MS`) branch right after it:
# FUN_0xb88dc9's `F0 = F0 * F8; R12 = M6` at MULT_WRITE_SW, consumed by
# `IF MS R4 = neg(R4); R12 = M7` at MULT_BRANCH_SW.
FUN_B88DC9 = 0xB88DC9
MULT_WRITE_SW = 0xB88DED
MULT_BRANCH_SW = 0xB88DF0


@unittest.skipUnless(
    DT2_116_DB.exists(), "out/sharcdb/dt2-1.16.sqlite is not available"
)
class CfgTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.img = sharc.load("dt2-1.16")

    def test_cfg_node_count_matches_bblocks(self):
        g = self.img.cfg(FUN_1C4F81)
        (n,) = self.img.db.execute(
            "SELECT COUNT(*) FROM bblocks WHERE image=? AND function_sw=?",
            ("dt2-1.16", FUN_1C4F81),
        ).fetchone()
        self.assertEqual(g.number_of_nodes(), n)
        self.assertGreater(g.number_of_edges(), 0)

    def test_cfg_edges_carry_succ_kind(self):
        g = self.img.cfg(FUN_1C4F81)
        kinds = {d["kind"] for _u, _v, d in g.edges(data=True)}
        self.assertTrue(kinds)
        self.assertTrue(
            kinds
            <= {
                "fallthrough",
                "jump",
                "cond_taken",
                "cond_not_taken",
                "call_return",
                "loop_back",
                "loop_exit",
                "return",
                "indirect",
            }
        )

    def test_cfg_marks_the_hardware_loop_header(self):
        # FUN_1c4ecf's own DO 0x1c4f45 UNTIL LCE (its listing has the loop
        # at sw 0x1c4f42..0x1c4f45) -- the block starting at its body (the
        # only block in that function with a succ row of kind 'loop_back'
        # into it) must come back with is_loop_header=True.
        rows = self.img.db.execute(
            "SELECT header_block FROM loops WHERE image=? AND function_sw=0x1c4ecf",
            ("dt2-1.16",),
        ).fetchall()
        self.assertTrue(rows, "expected at least one loop header in FUN_1c4ecf")
        g = self.img.cfg(0x1C4ECF)
        for (header,) in rows:
            self.assertTrue(g.nodes[header]["is_loop_header"])

    def test_cfg_is_cached(self):
        self.assertIs(self.img.cfg(FUN_1C4F81), self.img.cfg(FUN_1C4F81))


@unittest.skipUnless(
    DT2_116_DB.exists(), "out/sharcdb/dt2-1.16.sqlite is not available"
)
class CallgraphTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.img = sharc.load("dt2-1.16")

    def test_callgraph_matches_functions_and_call_edges(self):
        g = self.img.callgraph()
        (n_funcs,) = self.img.db.execute(
            "SELECT COUNT(*) FROM functions WHERE image=?", ("dt2-1.16",)
        ).fetchone()
        (n_edges,) = self.img.db.execute(
            "SELECT COUNT(*) FROM (SELECT DISTINCT from_function, to_function FROM edges "
            "WHERE image=? AND kind='call' AND from_function IS NOT NULL AND to_function IS NOT NULL)",
            ("dt2-1.16",),
        ).fetchone()
        self.assertEqual(g.number_of_nodes(), n_funcs)
        self.assertEqual(g.number_of_edges(), n_edges)

    def test_callgraph_has_the_known_render_call_edge(self):
        # docs/findings/06's call convention section: "FUN_1c2b24 ...
        # calls FUN_1c642a (0x1c3083)" -- a real edges.kind='call' row, not
        # the cond_jump/jump/fallthrough tail-call edges callgraph() (CALL
        # edges only, by design) must not pick up.
        g = self.img.callgraph()
        self.assertIn(0x1C642A, set(g.successors(0x1C2B24)))

    def test_callgraph_is_cached_and_reused_by_cards(self):
        g1 = self.img.callgraph()
        g2 = self.img.callgraph()
        self.assertIs(g1, g2)
        # cards() must not mutate the cached graph (root-filtering subgraphs
        # it instead).
        list(self.img.cards(root=0x1C15E3, limit=1))
        self.assertIs(self.img.callgraph(), g1)


@unittest.skipUnless(
    DT2_116_DB.exists(), "out/sharcdb/dt2-1.16.sqlite is not available"
)
class DefuseAndSliceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.img = sharc.load("dt2-1.16")

    def _preds(self, g, sw, reg):
        return {u for u, _v, d in g.in_edges(sw, data=True) if reg in d.get("regs", ())}

    def test_defuse_node_count_matches_aligned_instructions(self):
        g = self.img.defuse(FUN_1C4F81)
        (n,) = self.img.db.execute(
            "SELECT COUNT(*) FROM insn WHERE image=? AND function_sw=? AND aligned=1",
            ("dt2-1.16", FUN_1C4F81),
        ).fetchone()
        self.assertEqual(g.number_of_nodes(), n)

    def test_defuse_agrees_with_last_def_within_one_block(self):
        # A same-block, unconditional chain: no may-def/ASTAT subtlety, so
        # defuse()'s def->use edge and last_def()'s CFG walk must agree.
        g = self.img.defuse(FUN_1C4F81)
        self.assertEqual(self._preds(g, COND_SW, "R2"), {LOAD_SW})
        self.assertEqual(self.img.last_def("R2", COND_SW), LOAD_SW)
        self.assertEqual(self._preds(g, LOAD_SW, "M4"), {LIT_SW})
        self.assertEqual(self.img.last_def("M4", LOAD_SW), LIT_SW)

    def test_last_def_does_not_re_admit_the_queried_sws_own_def(self):
        # Regression for the recursive CTE's upper-bound bug: block
        # [0x1c501e, 0x1c5022) has no R6 def before 0x1c5020 (the R6
        # compute at 0x1c5020 is the queried sw itself), so the walk must
        # expand into its predecessor [0x1c5016, 0x1c501e) and beyond, not
        # re-admit 0x1c5020's own def by widening the search past the
        # current block's end. That predecessor is itself a merge point
        # (0x1c4f81/0x1c4ff9/0x1c5002 all reach it), so the two blocks that
        # actually carry an R6 def disagree and this returns a list -- one
        # of them was 0x1c5020 itself before the fix.
        r6_before = 0x1C5020
        self.assertNotEqual(self.img.last_def("R6", r6_before), r6_before)
        self.assertEqual(self.img.last_def("R6", r6_before), [0x1C4FD2, 0x1C500D])
        # Cross-check against defuse()'s reaching-definitions for the same
        # (use sw, reg): this is a plain multi-block merge, no conditional
        # def involved, so the two analyses must agree exactly.
        g = self.img.defuse(FUN_1C4F81)
        self.assertEqual(self._preds(g, r6_before, "R6"), {0x1C4FD2, 0x1C500D})

    def test_defuse_models_astat_as_a_flags_pseudo_register(self):
        # regdef has no ASTAT/flags entry at all (see sharcdb.py's SCHEMA
        # comment on regdef.kind) -- defuse() must still connect the
        # conditional clear at CLEAR_SITE_1 to the compute that set the
        # flag it reads.
        g = self.img.defuse(FUN_1C4F81)
        self.assertEqual(self._preds(g, CLEAR_SITE_1, "ASTAT"), {COND_SW})
        # The second clear (CLEAR_SITE_2) has its own, later flag source.
        self.assertEqual(self._preds(g, CLEAR_SITE_2, "ASTAT"), {0x1C5046})

    def test_defuse_models_astat_shift_group_for_leftz(self):
        # `leftz` (SHIFT compute unit, PRM p.521) only ever touches SS/SZ/SV
        # (tools/sharc_core/flags.py's _astatx_leftz) -- CLEAR_SITE_1's own
        # SV-reading cond must resolve through the precise "ASTAT.SHIFT"
        # pseudo-register, exactly like the whole-register "ASTAT" one.
        g = self.img.defuse(FUN_1C4F81)
        self.assertEqual(self._preds(g, CLEAR_SITE_1, "ASTAT.SHIFT"), {COND_SW})
        # leftz cannot touch an ALU/MULT/BTF bit, so it must not appear as
        # a reaching def for those groups at all.
        self.assertEqual(self._preds(g, CLEAR_SITE_1, "ASTAT.ALU"), set())
        self.assertEqual(self._preds(g, CLEAR_SITE_1, "ASTAT.MULT"), set())
        self.assertEqual(self._preds(g, CLEAR_SITE_1, "ASTAT.BTF"), set())

    def test_defuse_marks_unresolved_uses_as_boundaries(self):
        # I4/M1/M13 are never written inside FUN_1c4f81 (I4/M1 are restored
        # in the physically-preceding FUN_1c4ecf's shared epilogue; M13 is
        # the hardware DAG-modify reset constant, docs/findings/06) -- an
        # intra-procedural analysis must report them as boundaries, not
        # silently omit them or invent a def.
        g = self.img.defuse(FUN_1C4F81)
        self.assertEqual(g.nodes[CLEAR_SITE_1].get("unresolved"), {"I4", "M1", "M13"})

    def test_defuse_mem_load_and_literal_node_attrs(self):
        g = self.img.defuse(FUN_1C4F81)
        self.assertIn("mem", g.nodes[LOAD_SW])
        self.assertEqual(g.nodes[LIT_SW].get("literal"), "0x1bc")

    def test_defuse_is_cached(self):
        self.assertIs(self.img.defuse(FUN_1C4F81), self.img.defuse(FUN_1C4F81))

    def test_slice_reaches_the_literal_offset_two_hops_back(self):
        g = self.img.slice(CLEAR_SITE_1)
        self.assertIn(COND_SW, g)
        self.assertIn(LOAD_SW, g)
        self.assertIn(LIT_SW, g)

    def test_slice_reg_filter_seeds_from_only_that_edge(self):
        g_astat = self.img.slice(CLEAR_SITE_1, reg="ASTAT")
        self.assertIn(COND_SW, g_astat.predecessors(CLEAR_SITE_1))
        self.assertEqual(set(g_astat.predecessors(CLEAR_SITE_1)), {COND_SW})

    def test_slice_depth_bounds_the_walk(self):
        g_full = self.img.slice(CLEAR_SITE_1)
        g_1 = self.img.slice(CLEAR_SITE_1, depth=1)
        self.assertIn(COND_SW, g_1)
        self.assertNotIn(LOAD_SW, g_1)
        self.assertLess(g_1.number_of_nodes(), g_full.number_of_nodes())

    def test_slice_rejects_a_sw_outside_any_function(self):
        with self.assertRaises(ValueError):
            self.img.slice(0)

    def test_print_slice_renders_the_known_chain(self):
        text = self.img.print_slice(CLEAR_SITE_1)
        self.assertIn("0x1c5008", text)
        self.assertIn("leftz", text)
        self.assertIn("literal=0x1bc", text)
        self.assertIn("unresolved: I4,M1,M13", text)


@unittest.skipUnless(
    DT2_116_DB.exists(), "out/sharcdb/dt2-1.16.sqlite is not available"
)
class AstatGroupsBtfAndMultTest(unittest.TestCase):
    """The exact bug the frame-render survey found (a slice for a
    BTF-reading branch pointed at an unrelated ALU/MULT/SHIFT compute, since
    the old whole-ASTAT rule had no def at all for a Type18a bit-test) --
    against real, fixed addresses in dt2-1.16's own image, not synthetic
    ones."""

    @classmethod
    def setUpClass(cls):
        cls.img = sharc.load("dt2-1.16")

    def test_btf_branch_reaches_the_real_bit_test_not_an_unrelated_compute(
        self,
    ):
        g = self.img.defuse(FUN_B88200)
        preds_btf = {
            u
            for u, _v, d in g.in_edges(BTF_BRANCH_SW, data=True)
            if "ASTAT.BTF" in d.get("regs", ())
        }
        self.assertEqual(preds_btf, {BTF_WRITE_SW})
        self.assertNotIn(BTF_UNRELATED_COMPUTE_SW, preds_btf)
        # The old whole-register pseudo-register must also see it now
        # (previously it did not either -- see the class docstring).
        preds_whole = {
            u
            for u, _v, d in g.in_edges(BTF_BRANCH_SW, data=True)
            if "ASTAT" in d.get("regs", ())
        }
        self.assertEqual(preds_whole, {BTF_WRITE_SW})

    def test_btf_slice_walks_back_to_the_bit_test(self):
        text = self.img.print_slice(BTF_BRANCH_SW, reg="ASTAT.BTF")
        self.assertIn("0x%x" % BTF_WRITE_SW, text)
        self.assertIn("bit-test", text)
        self.assertNotIn("0x%x" % BTF_UNRELATED_COMPUTE_SW, text)

    def test_mult_branch_reaches_the_float_multiply(self):
        # `IF MS` (cond=0x06, PGR Table 10-4) reads MN (SIMPLE_COND_BITS):
        # the float multiply at MULT_WRITE_SW is the only thing that can
        # touch it.
        g = self.img.defuse(FUN_B88DC9)
        preds_mult = {
            u
            for u, _v, d in g.in_edges(MULT_BRANCH_SW, data=True)
            if "ASTAT.MULT" in d.get("regs", ())
        }
        self.assertEqual(preds_mult, {MULT_WRITE_SW})
        preds_shift = {
            u
            for u, _v, d in g.in_edges(MULT_BRANCH_SW, data=True)
            if "ASTAT.SHIFT" in d.get("regs", ())
        }
        self.assertEqual(preds_shift, set())


if __name__ == "__main__":
    unittest.main()
