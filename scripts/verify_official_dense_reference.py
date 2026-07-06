#!/usr/bin/env python
"""Compare local dense-reference files against upstream OpenTAD.

This script is intentionally read-only. Use it when dense AdaTAD / ActionFormer
behavior must be audited against the official OpenTAD source instead of the
copies in this sparse-head route repository.
"""

from __future__ import annotations

import argparse
import difflib
from pathlib import Path
import sys
from urllib.request import urlopen


REFERENCE_FILES = (
    "opentad/models/dense_heads/anchor_free_head.py",
    "opentad/models/dense_heads/prior_generator/point_generator.py",
    "opentad/models/necks/fpn.py",
)
OFFICIAL_URLS = {
    "opentad/models/dense_heads/anchor_free_head.py": "https://raw.githubusercontent.com/sming256/OpenTAD/main/opentad/models/dense_heads/anchor_free_head.py",
    "opentad/models/dense_heads/prior_generator/point_generator.py": "https://raw.githubusercontent.com/sming256/OpenTAD/main/opentad/models/dense_heads/prior_generator/point_generator.py",
    "opentad/models/necks/fpn.py": "https://raw.githubusercontent.com/sming256/OpenTAD/main/opentad/models/necks/fpn.py",
}


def read_local(root: Path, rel_path: str) -> str:
    return (root / rel_path).read_text(encoding="utf-8")


def read_official(args: argparse.Namespace, rel_path: str) -> str:
    if args.official_root is not None:
        return read_local(args.official_root, rel_path)
    with urlopen(OFFICIAL_URLS[rel_path], timeout=args.timeout) as response:
        return response.read().decode("utf-8")


def compare_file(local_root: Path, args: argparse.Namespace, rel_path: str) -> str:
    local_text = read_local(local_root, rel_path)
    official_text = read_official(args, rel_path)
    if local_text == official_text:
        return ""
    diff = difflib.unified_diff(
        official_text.splitlines(keepends=True),
        local_text.splitlines(keepends=True),
        fromfile=f"official/{rel_path}",
        tofile=f"local/{rel_path}",
    )
    return "".join(diff)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Path to this local OpenTAD route repository.",
    )
    parser.add_argument(
        "--official-root",
        type=Path,
        default=None,
        help="Optional local checkout of official OpenTAD. If omitted, raw GitHub URLs are fetched.",
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="Network timeout in seconds.")
    parser.add_argument(
        "--fail-on-diff",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Return non-zero when any reference file differs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    local_root = args.local_root.resolve()
    diffs = []
    for rel_path in REFERENCE_FILES:
        diff = compare_file(local_root, args, rel_path)
        if diff:
            diffs.append(diff)

    if not diffs:
        print("Local dense reference files match upstream OpenTAD.")
        return 0

    print("\n".join(diffs))
    if args.fail_on_diff:
        print("Dense reference drift detected.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
