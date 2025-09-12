#!/usr/bin/env python3
# get_network_versions.py
# Usage: python get_network_versions.py /path/to/folder

import sys
from pathlib import Path

def get_network_version(demo_path: Path) -> str:
    from awpy import Demo
    dem = Demo(str(demo_path), verbose=False)
    header = dem.parse_header()  # fast: header-only
    # awpy returns 'network_protocol' (CS:2). Be defensive with a few variants.
    return (
        header.get("network_protocol")
        or header.get("networkProtocol")
        or header.get("Network Protocol")
        or "unknown"
    )

def main():
    if len(sys.argv) != 2:
        print("Usage: python get_network_versions.py /path/to/folder", file=sys.stderr)
        sys.exit(1)

    folder = Path(sys.argv[1])
    if not folder.is_dir():
        print(f"Not a directory: {folder}", file=sys.stderr)
        sys.exit(1)

    # Find .dem files (recursively). Change to glob('*.dem') if you only want the top-level.
    demos = sorted(folder.rglob("*.dem"))
    if not demos:
        print("No .dem files found.", file=sys.stderr)
        sys.exit(2)

    for demo_path in demos:
        try:
            netver = get_network_version(demo_path)
            print(f"{demo_path.name} - {netver}")
        except Exception as e:
            print(f"{demo_path.name} - ERROR: {e.__class__.__name__}: {e}")

if __name__ == "__main__":
    main()
