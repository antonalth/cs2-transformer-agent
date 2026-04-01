#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import struct
from pathlib import Path


PT_GNU_STACK = 0x6474E551
PF_X = 0x1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--backup", action="store_true")
    return parser.parse_args()


def read_fmt(data: bytes, offset: int, fmt: str) -> tuple[int, ...]:
    size = struct.calcsize(fmt)
    return struct.unpack(fmt, data[offset : offset + size])


def main() -> int:
    args = parse_args()
    path = Path(args.path)
    blob = bytearray(path.read_bytes())

    if blob[:4] != b"\x7fELF":
        raise SystemExit(f"{path} is not an ELF file")

    elf_class = blob[4]
    endian = blob[5]
    if endian == 1:
        prefix = "<"
    elif endian == 2:
        prefix = ">"
    else:
        raise SystemExit(f"{path}: unsupported ELF endianness {endian}")

    if elf_class == 1:
        e_phoff = read_fmt(blob, 28, prefix + "I")[0]
        e_phentsize = read_fmt(blob, 42, prefix + "H")[0]
        e_phnum = read_fmt(blob, 44, prefix + "H")[0]
        ph_fmt = prefix + "IIIIIIII"
        p_type_index = 0
        p_flags_index = 6
    elif elf_class == 2:
        e_phoff = read_fmt(blob, 32, prefix + "Q")[0]
        e_phentsize = read_fmt(blob, 54, prefix + "H")[0]
        e_phnum = read_fmt(blob, 56, prefix + "H")[0]
        ph_fmt = prefix + "IIQQQQQQ"
        p_type_index = 0
        p_flags_index = 1
    else:
        raise SystemExit(f"{path}: unsupported ELF class {elf_class}")

    ph_size = struct.calcsize(ph_fmt)
    if e_phentsize < ph_size:
        raise SystemExit(f"{path}: unexpected program header size {e_phentsize}")

    changed = False
    for i in range(e_phnum):
        offset = e_phoff + i * e_phentsize
        header = list(read_fmt(blob, offset, ph_fmt))
        if header[p_type_index] != PT_GNU_STACK:
            continue
        old_flags = header[p_flags_index]
        new_flags = old_flags & ~PF_X
        if new_flags == old_flags:
            print(f"{path}: PT_GNU_STACK already non-executable ({old_flags:#x})")
            return 0
        header[p_flags_index] = new_flags
        blob[offset : offset + ph_size] = struct.pack(ph_fmt, *header)
        print(f"{path}: PT_GNU_STACK flags {old_flags:#x} -> {new_flags:#x}")
        changed = True
        break

    if not changed:
        raise SystemExit(f"{path}: PT_GNU_STACK header not found")

    if args.backup:
        backup = path.with_suffix(path.suffix + ".bak")
        if not backup.exists():
            shutil.copy2(path, backup)
            print(f"backup: {backup}")

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(blob)
    os.replace(tmp, path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
