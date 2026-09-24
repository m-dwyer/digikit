"""tools/sharc_visa_tables.py, tools/sharc_disasm.py and tools/sharccompare.py on words built from the table."""

import os
import struct
import sys
import unittest

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"),
)

import sharc_disasm  # noqa: E402
import sharc_visa_tables as T  # noqa: E402
import sharccompare  # noqa: E402


def encode(name, extra=0):
    """-> little-endian bytes of a form-`name` instruction: its fixed bits, plus `extra` elsewhere."""
    t = T.get_type(name)
    insn = t["opcode_value"] | (extra & ~t["opcode_mask"] & ((1 << t["bits"]) - 1))
    words = [
        (insn >> (t["bits"] - 16 * (i + 1))) & 0xFFFF for i in range(t["bits"] // 16)
    ]
    return struct.pack("<%dH" % len(words), *words)


def undecodable_word():
    for w in range(0x10000):
        if T.decode([w, 0, 0]) == (None, []):
            return w
    raise AssertionError("every word decodes")


class TableTest(unittest.TestCase):
    def test_names(self):
        names = [t["name"] for t in T.TYPES]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(T.form_name("Type5b (move)"), "5b_move")
        self.assertEqual(T.form_name("Type8a_abs"), "8a_abs")

    def test_masks_and_fields_fit_the_width(self):
        for t in T.TYPES:
            low = (1 << (48 - t["bits"])) - 1
            self.assertEqual(t["frame_mask"] & low, 0, t["name"])
            self.assertLess(t["opcode_mask"], 1 << t["bits"], t["name"])
            for label, (hi, lo) in t["fields"].items():
                self.assertTrue(0 <= lo <= hi < t["bits"], (t["name"], label))

    def test_15b_fixes_seven_bits(self):
        t = T.get_type("15b")
        self.assertEqual(t["fixed_bits"], 7)
        self.assertEqual(bin(t["opcode_mask"]).count("1"), 7)


class DisasmTest(unittest.TestCase):
    def test_02_prefix_is_a_48_bit_shift_immediate(self):
        data = struct.pack("<HHH", 0x023E, 0x3810, 0x8022)
        rec = next(sharc_disasm.disassemble(data))
        self.assertEqual(
            (rec.type_name, rec.length_bytes, rec.kind), ("6b_shiftimm", 6, "confident")
        )
        self.assertEqual(rec.raw, 0x023E38108022)
        self.assertEqual(
            rec.fields,
            {
                "cond[4:0]": 0x1F,
                "dataex[3:0]": 7,
                "shiftimm[22:16]": 0x10,
                "shiftimm[15:0]": 0x8022,
            },
        )

    def test_01_prefix_bit39_selects_32bit_compute(self):
        data = bytes.fromhex("a80180820e0700001d00")
        recs = list(sharc_disasm.disassemble(data))
        self.assertEqual(
            (recs[0].type_name, recs[0].length_bytes, recs[0].kind),
            ("2a_short", 4, "confident"),
        )
        self.assertEqual(
            recs[0].fields, {"compute[22:16]": 0x28, "compute[15:0]": 0x8280}
        )
        self.assertEqual(recs[1].type_name, "8a_rel")

    def test_01_prefix_bit39_clear_stays_48bit_type2a(self):
        rec = next(sharc_disasm.disassemble(bytes.fromhex("280180820e07")))
        self.assertEqual(
            (rec.type_name, rec.length_bytes, rec.kind), ("2a", 6, "confident")
        )
        self.assertEqual(rec.fields["cond[4:0]"], 20)

    def test_15_prefix_is_documented_scaled_type19_modify(self):
        data = bytes.fromhex("8715fffffeff")
        rec = next(sharc_disasm.disassemble(data))
        self.assertEqual(
            (rec.type_name, rec.length_bytes, rec.kind),
            ("19a_scaled", 6, "confident"),
        )
        self.assertEqual(rec.raw, 0x1587FFFFFFFE)
        self.assertEqual(
            rec.fields,
            {
                "w": 1,
                "g": 0,
                "idis[2:0]": 0,
                "is[2:0]": 7,
                "data[31:16]": 0xFFFF,
                "data[15:0]": 0xFFFE,
            },
        )

    def test_walk(self):
        t = T.get_type("17b")
        hi, lo = t["fields"]["ureg[6:0]"]
        data = encode("17b", 0x55 << lo) + encode("15b")
        recs = list(sharc_disasm.disassemble(data))
        self.assertEqual(
            [(r.offset, r.type_name, r.length_bytes) for r in recs],
            [(0, "17b", 4), (4, "15b", 4)],
        )
        self.assertEqual(recs[0].fields["ureg[6:0]"], 0x55)

    def test_unknown_stops(self):
        data = (
            encode("17b")
            + struct.pack("<HHH", undecodable_word(), 0, 0)
            + encode("17b")
        )
        recs = list(sharc_disasm.disassemble(data))
        self.assertEqual([r.kind for r in recs], ["confident", "unknown"])
        self.assertEqual(recs[-1].offset, 4)
        report = sharc_disasm.walk_and_report(data)
        self.assertEqual((report.end_offset, report.instructions), (4, 1))

    def test_buffer_ends_inside_an_instruction(self):
        wide = next(
            t["name"]
            for t in T.TYPES
            if t["bits"] == 48
            and not t["uncertain"]
            and T.decode(struct.unpack("<3H", encode(t["name"])))[0] is not None
            and T.decode(struct.unpack("<3H", encode(t["name"])))[0]["name"]
            == t["name"]
        )
        recs = list(sharc_disasm.disassemble(encode(wide)[:4]))
        self.assertEqual(recs[-1].kind, "unknown")

    def test_raise_and_empty(self):
        with self.assertRaises(sharc_disasm.Desync):
            list(sharc_disasm.disassemble(b"\x00", on_unknown="raise"))
        self.assertEqual(list(sharc_disasm.disassemble(b"")), [])


class CompareTest(unittest.TestCase):
    def test_names_agree(self):
        self.assertTrue(sharccompare.names_agree("8a", "8a_abs"))
        self.assertTrue(sharccompare.names_agree("15b", "15b"))
        self.assertFalse(sharccompare.names_agree("5a_move", "5a_swap"))
        self.assertFalse(sharccompare.names_agree("17b", "15b"))
        self.assertFalse(sharccompare.names_agree(None, "15b"))

    def test_sweeps_agree_on_built_words(self):
        data = encode("17b") + struct.pack("<H", undecodable_word()) + encode("15b")
        ours = sharccompare.sweep_ours(sharc_disasm, data, 0)
        spec = sharccompare.sweep_spec(data, 0)
        self.assertEqual(ours[0], (4, "17b"))
        self.assertEqual(ours[4], (None, None))
        self.assertEqual(
            sharccompare.compare(ours, spec, 5)["same_form"],
            sharccompare.compare(ours, spec, 5)["common_instructions"],
        )


if __name__ == "__main__":
    unittest.main()
