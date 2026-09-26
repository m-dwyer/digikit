#!/usr/bin/env python3
"""Build and read a +Drive card image, in the firmware's own on-disk format.

The format is reverse engineered in docs/findings/14-plus-drive-format.md.
Short version: +Drive has two independent regions at fixed, hard-coded
absolute sector numbers -- the ``MmcFs`` project/sound/kit pools (untouched
here) and a separate, real, path-addressable filesystem (128-byte metadata
records, extent-mapped 32 KiB content pages, hierarchical directories) that
sample files live in. This tool only writes the second region: a header
sector, an empty pool table, a root directory, and one file entry per input
WAV, all under the root.

    uv run python tools/plusdrive.py build samples/ -o out/plusdrive/dt2.img
    uv run python tools/plusdrive.py ls out/plusdrive/dt2.img

``build`` also doubles as the only writer of this format, and ``ls`` as a
reader for any card image in it (including one dumped from a real firmware
card overlay), since both share the same decode/encode tables below.

To boot the emulator from a built image, pass its path as ``card_image=`` to
``emu.dspboot.run``/``emu.longrun.build`` (or ``--card-image`` on the CLI
runners that expose it); both construct the card via
``emu.esdhc.Card.from_file``, which mmaps the file read-only -- writes during
emulation go to the in-RAM overlay (already how snapshots capture +Drive
writes; see ``Esdhc.checkpoint_state``), and this file is never modified.

Open questions the finding doc flags, most relevant to this tool:

* The exact attribute-byte values at record offsets 0x00/0x01/0x0c for
  "regular file" vs "directory" are inferred, not proven (see ATTR_* below).
* Sample payload bytes are stored verbatim (whole input WAV file, including
  its RIFF header) -- this is the only choice consistent with the proven
  "write path does no interpretation" finding, not a confirmed fact about
  what the browser/loader expects to unpack.
* The 0x10000 (name-hash) and 0x10002 (id-sorted) auxiliary index pages are
  deliberately NOT written -- believed unnecessary for on-device browsing
  and loading (as opposed to by-path MIDI RPC access), unconfirmed.
"""

import argparse
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SECTOR = 512
PAGE = 0x8000  # 32 KiB, 0x40 sectors
RECORD_SIZE = 0x80  # 128 bytes
RECORDS_PER_PAGE = PAGE // RECORD_SIZE  # 256

# Region 1 (MmcFs pools): left untouched (all zero), but the header and the
# empty pool-occupancy table are written so first-boot format detection
# passes and the pools read as validly-empty rather than corrupt. See
# docs/findings/14-plus-drive-format.md.
HEADER_SECTOR = 0
HEADER_MAGIC = 0xBEEFBACE
POOL_TABLE_SECTOR = 0x800

# Region 2 (the real filesystem): absolute sector numbers, hard-coded in the
# firmware, not derived from card capacity.
ID_BITMAP_SECTOR = 0x5D8040
PAGE_BITMAP_SECTOR = 0x5D80C0
PAGE_BITMAP_SECTOR_END = 0x5D8180
RECORD_AREA_SECTOR = 0x5D8180
CONTENT_AREA_SECTOR = 0x5EE180

ROOT_ID = 2
ROOT_PARENT = 2  # the root is its own parent; nothing reads this for id 2

# Reserved logical pages inside a directory's own extent list.
LOGICAL_INDEX_PAGE = 0x10001  # readdir's "jump to entry N" index; required
# LOGICAL_HASH_PAGE = 0x10000   # by-name binary search; not written here
# LOGICAL_ID_PAGE   = 0x10002   # id-sorted index; not written here

# Record offset 0x00 / 0x01 bit-0 semantics are not disambiguated in the
# finding (candidates: is-directory, protected/read-only, a third flag).
# Best inference: offset 0x00 bit 0 marks a directory (this is the byte the
# directory-entry writer copies into its own per-entry "type" field, and
# what FileSystemDirectory's attribute vfuncs test), offset 0x01 keeps the
# allocator's own default (2, bit 0 clear) for both kinds.
ATTR_DIR = 0x01
ATTR_FILE = 0x00
DEFAULT_OFFSET01 = 0x02

DEFAULT_CAPACITY_BLOCKS = 0x00760000  # matches emu/esdhc.py's Card default


def _u16(v):
    return struct.pack(">H", v & 0xFFFF)


def _u32(v):
    return struct.pack(">I", v & 0xFFFFFFFF)


class Image:
    """An in-progress (or already-built) +Drive card image, as a sparse file.

    Only the bytes this tool actually writes are ever touched on disk; a
    freshly `truncate()`d file reads as zero everywhere else, matching how
    the real card behaves before anything is written there.
    """

    def __init__(self, path, capacity_blocks=DEFAULT_CAPACITY_BLOCKS):
        self.path = path
        self.capacity_blocks = capacity_blocks
        self._f = open(path, "w+b")  # noqa: SIM115 -- kept open for the object's life
        self._f.truncate(capacity_blocks * SECTOR)

    def close(self):
        self._f.close()

    def write(self, sector, data):
        self._f.seek(sector * SECTOR)
        self._f.write(data)

    def read(self, sector, length):
        self._f.seek(sector * SECTOR)
        data = self._f.read(length)
        if len(data) < length:
            data = data + b"\0" * (length - len(data))
        return data

    def set_bits(self, base_sector, bit_indices):
        """OR the given bit numbers into a bitmap starting at base_sector.

        All the bit numbers this tool ever sets are tiny (single digits),
        so they land in the first word of the first page; a real 32 KiB
        bitmap page is still allocated (zero-filled by the sparse-file
        default) so the read side sees a well-formed page, not a hole.
        """
        page = bytearray(self.read(base_sector, PAGE))
        for bit in bit_indices:
            page[bit >> 3] |= 1 << (bit & 7)
        self.write(base_sector, bytes(page))


def _record_offset(record_id):
    group, slot = divmod(record_id, RECORDS_PER_PAGE)
    return (RECORD_AREA_SECTOR + group * (PAGE // SECTOR)) * SECTOR + slot * RECORD_SIZE


def _write_record(img, record_id, attr0, size, parent, extents):
    """extents: list of (logical_start_page, length_pages, physical_page)."""
    if len(extents) > 8:
        raise ValueError(
            "more than 8 extents needs the indirect-page format (unimplemented)"
        )
    rec = bytearray(RECORD_SIZE)
    rec[0x00] = attr0
    rec[0x01] = DEFAULT_OFFSET01
    struct.pack_into(">H", rec, 0x02, 1)  # link count
    struct.pack_into(">I", rec, 0x04, size)
    struct.pack_into(">I", rec, 0x08, parent)
    struct.pack_into(">I", rec, 0x0C, 0)
    struct.pack_into(">I", rec, 0x10, record_id)  # sequence number, any unique value
    struct.pack_into(">H", rec, 0x1E, len(extents))
    for i, (logical, length, phys) in enumerate(extents):
        off = 0x20 + i * 12
        struct.pack_into(">III", rec, off, logical, length, phys)
    off = _record_offset(record_id)
    img._f.seek(off)
    img._f.write(bytes(rec))


def _page_offset(page_id):
    return (CONTENT_AREA_SECTOR + page_id * (PAGE // SECTOR)) * SECTOR


def _write_page(img, page_id, data):
    if len(data) > PAGE:
        raise ValueError("page payload exceeds 32 KiB")
    img._f.seek(_page_offset(page_id))
    img._f.write(data)


def build(samples_dir, out_path, capacity_blocks=DEFAULT_CAPACITY_BLOCKS):
    """Build a +Drive card image at out_path from every file directly under
    samples_dir (non-recursive; anything other than .wav is skipped with a
    warning). -> list of (name, size) written."""
    entries = []
    for name in sorted(os.listdir(samples_dir)):
        if not name.lower().endswith(".wav"):
            continue
        path = os.path.join(samples_dir, name)
        if not os.path.isfile(path):
            continue
        with open(path, "rb") as f:
            data = f.read()
        entries.append((name, data))
    if not entries:
        raise ValueError("no .wav files found directly under %r" % samples_dir)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    img = Image(out_path, capacity_blocks)
    try:
        # Region 1: header + empty pool table, so first-boot format
        # detection sees a valid, already-formatted (if sample-only) card.
        header = bytearray(SECTOR)
        struct.pack_into(">I", header, 0x00, HEADER_MAGIC)
        struct.pack_into(">I", header, 0x04, 1)
        header[0x08] = 1
        header[0x09] = 1
        img.write(HEADER_SECTOR, bytes(header))
        img.write(POOL_TABLE_SECTOR, bytes(SECTOR))  # all-zero: nothing allocated

        # Region 2: allocate physical pages up front -- page 0 is the root
        # directory's content (the child listing), page 1 is its 0x10001
        # enumeration index, then each sample gets ceil(size/32KiB)
        # contiguous pages so a single inline extent covers it.
        next_page = 0
        root_content_page = next_page
        next_page += 1
        root_index_page = next_page
        next_page += 1

        dir_content = bytearray()
        file_records = []  # (id, name, size)
        next_id = ROOT_ID + 1
        for name, data in entries:
            name_bytes = name.encode("ascii", "replace")
            if len(name_bytes) > 0xFF:
                name_bytes = name_bytes[:0xFF]
            n_pages = max(1, (len(data) + PAGE - 1) // PAGE)
            start_page = next_page
            next_page += n_pages
            for i in range(n_pages):
                chunk = data[i * PAGE : (i + 1) * PAGE]
                _write_page(img, start_page + i, chunk)

            record_id = next_id
            next_id += 1
            _write_record(
                img,
                record_id,
                ATTR_FILE,
                len(data),
                ROOT_ID,
                [(0, n_pages, start_page)],
            )

            slot_len = 8 + len(name_bytes)
            entry = bytearray(slot_len)
            struct.pack_into(">I", entry, 0x00, record_id)
            struct.pack_into(">H", entry, 0x04, slot_len)
            entry[0x06] = len(name_bytes)
            entry[0x07] = ATTR_FILE
            entry[0x08 : 0x08 + len(name_bytes)] = name_bytes
            file_records.append((record_id, name, len(data), len(dir_content)))
            dir_content += entry

        _write_page(img, root_content_page, bytes(dir_content))

        index_page = bytearray(6 + 4 * len(file_records))
        struct.pack_into(">H", index_page, 0x00, len(file_records))
        for i, (_, _, _, byte_offset) in enumerate(file_records):
            if byte_offset >= PAGE:
                raise ValueError(
                    "root directory content exceeds one 32 KiB page (unimplemented)"
                )
            packed = byte_offset & 0x7FFF  # page field 0 (all entries on page 0)
            struct.pack_into(">I", index_page, 6 + i * 4, packed)
        _write_page(img, root_index_page, bytes(index_page))

        _write_record(
            img,
            ROOT_ID,
            ATTR_DIR,
            len(dir_content),
            ROOT_PARENT,
            [(0, 1, root_content_page), (LOGICAL_INDEX_PAGE, 1, root_index_page)],
        )

        img.set_bits(
            ID_BITMAP_SECTOR, [ROOT_ID] + [rid for rid, _, _, _ in file_records]
        )
        img.set_bits(PAGE_BITMAP_SECTOR, list(range(next_page)))
    finally:
        img.close()

    return [(name, len(data)) for name, data in entries]


def _read_record(img, record_id):
    group, slot = divmod(record_id, RECORDS_PER_PAGE)
    sector = RECORD_AREA_SECTOR + group * (PAGE // SECTOR)
    page = img.read(sector, PAGE)
    rec = page[slot * RECORD_SIZE : slot * RECORD_SIZE + RECORD_SIZE]
    attr0 = rec[0]
    size = struct.unpack(">I", rec[4:8])[0]
    parent = struct.unpack(">I", rec[8:12])[0]
    n_extents = struct.unpack(">H", rec[0x1E:0x20])[0]
    extents = []
    for i in range(min(n_extents, 8)):
        off = 0x20 + i * 12
        extents.append(struct.unpack(">III", rec[off : off + 12]))
    return {
        "id": record_id,
        "attr0": attr0,
        "size": size,
        "parent": parent,
        "extents": extents,
    }


def _read_page(img, page_id):
    return img.read(_page_offset(page_id) // SECTOR, PAGE)


def ls(image_path):
    """-> list of dicts, one per entry directly under the root directory.

    Reads back exactly the layout `build` writes: the root's own record for
    its content-page extent, then the variable-length directory-entry list
    on that page. Works as a generic reader for any image in this format,
    not just ones this tool wrote -- e.g. a real firmware card overlay,
    once it holds sample files under `/`.
    """
    with open(image_path, "rb") as f:
        img = _ReadOnlyImage(f)
        root = _read_record(img, ROOT_ID)
        content_extents = [e for e in root["extents"] if e[0] < 0x10000]
        if not content_extents:
            return []
        _, length, phys = content_extents[0]
        content = b"".join(_read_page(img, phys + i) for i in range(length))
        content = content[: root["size"]]

        out = []
        pos = 0
        while pos < len(content):
            record_id = struct.unpack(">I", content[pos : pos + 4])[0]
            if record_id == 0:
                break
            slot_len = struct.unpack(">H", content[pos + 4 : pos + 6])[0]
            name_len = content[pos + 6]
            attr = content[pos + 7]
            name = content[pos + 8 : pos + 8 + name_len].decode("ascii", "replace")
            rec = _read_record(img, record_id)
            out.append(
                {
                    "id": record_id,
                    "name": name,
                    "type": attr,
                    "size": rec["size"],
                }
            )
            pos += slot_len
        return out


class _ReadOnlyImage:
    """Just enough of Image's read() to share ls()/build() sector math."""

    def __init__(self, f):
        self._f = f

    def read(self, sector, length):
        self._f.seek(sector * SECTOR)
        data = self._f.read(length)
        if len(data) < length:
            data = data + b"\0" * (length - len(data))
        return data


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build a +Drive image from a folder of WAVs")
    b.add_argument("samples_dir")
    b.add_argument("-o", "--out", required=True)
    b.add_argument(
        "--capacity-blocks", type=lambda s: int(s, 0), default=DEFAULT_CAPACITY_BLOCKS
    )

    lsp = sub.add_parser("ls", help="list a +Drive image's root directory")
    lsp.add_argument("image")

    args = p.parse_args(argv)
    if args.cmd == "build":
        entries = build(args.samples_dir, args.out, args.capacity_blocks)
        print(
            "wrote %s (%d bytes logical) with %d sample(s):"
            % (args.out, args.capacity_blocks * SECTOR, len(entries))
        )
        for name, size in entries:
            print("  %-32s %d bytes" % (name, size))
    elif args.cmd == "ls":
        entries = ls(args.image)
        if not entries:
            print("(root directory is empty or unreadable)")
        for e in entries:
            kind = "dir " if e["type"] == ATTR_DIR else "file"
            print("id=%-6d %s %-32s %d bytes" % (e["id"], kind, e["name"], e["size"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
