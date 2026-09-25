"""dt2/elz.py against the repo's own packer and, when present, 1.15C's sections."""

import hashlib
import os
import random
import unittest

from dt2 import aplib, elz
from dt2.container import sections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYX = os.path.join(ROOT, 'Digitakt_II_OS1.15C.syx')
SECTIONS = os.path.join(ROOT, 'sections')


def sections_from_syx():
    """True when sections/ was extracted from the 1.15C .syx in the repo root."""
    marker = os.path.join(SECTIONS, '.source-sha256')
    if not (os.path.exists(SYX) and os.path.exists(marker)):
        return False
    with open(SYX, 'rb') as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    with open(marker) as f:
        return f.read().strip() == digest


class ElzTest(unittest.TestCase):
    def test_round_trip_with_the_store_only_packer(self):
        data = bytes(random.Random(1).randrange(256) for _ in range(5000))
        self.assertEqual(elz.depack_section(aplib.pack_section(data)), data)

    def test_padding_after_the_end_marker_is_accepted(self):
        # a same-size repack pads the stream past the end marker; the device
        # stops at the marker, so the padding must not be an error
        packed = bytearray(aplib.pack_section(b'hello world'))
        packed[3] += 4  # declare four more stream bytes than the end marker uses
        self.assertEqual(elz.depack_section(bytes(packed) + bytes(4)), b'hello world')

    def test_end_marker_past_the_declared_length_is_an_error(self):
        packed = bytearray(aplib.pack_section(b'hello world'))
        packed[3] -= 4  # declare four fewer stream bytes than the end marker needs
        with self.assertRaises(EOFError):  # depack runs out of declared stream first
            elz.depack_section(bytes(packed))

    @unittest.skipUnless(sections_from_syx(), 'needs the 1.15C .syx and sections/ extracted from it')
    def test_matches_the_device_depacker_on_1_15c(self):
        c, secs = sections(SYX)
        names = {2: 'section_2_DSP.bin', 3: 'section_3_MAIN_OS.bin', 7: 'section_7_BLOB.bin'}
        for sid, off, clen, dest in secs:
            if sid in names:
                with self.subTest(section=sid):
                    with open(os.path.join(SECTIONS, names[sid]), 'rb') as f:
                        expected = f.read()
                    self.assertEqual(elz.depack_section(bytes(c[off:off + clen])), expected)


if __name__ == '__main__':
    unittest.main()
