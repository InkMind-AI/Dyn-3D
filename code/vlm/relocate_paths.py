#!/usr/bin/env python3
"""Rewrite portable media placeholders in released JSONL files.

Released data stores media paths as ``__PROJECT_ROOT__/data/...`` so no
internal filesystem path is exposed. Run this once after cloning or unpacking.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MARKER = "__PROJECT_ROOT__"


def rewrite(value: Any, root: str) -> Any:
    if isinstance(value, str):
        return value.replace(MARKER, root)
    if isinstance(value, list):
        return [rewrite(item, root) for item in value]
    if isinstance(value, dict):
        return {key: rewrite(item, root) for key, item in value.items()}
    return value


def rewrite_jsonl(path: Path, root: str, in_place: bool) -> tuple[int, Path]:
    output = path if in_place else path.with_suffix(".local.jsonl")
    # Never open the source file for writing while it is being read.  This is
    # particularly important for the large released JSONL files.
    write_path = path.with_suffix(path.suffix + ".tmp") if in_place else output
    rows = 0
    with path.open("r", encoding="utf-8") as source, write_path.open("w", encoding="utf-8") as target:
        for line in source:
            if not line.strip():
                continue
            target.write(json.dumps(rewrite(json.loads(line), root), ensure_ascii=False) + "\n")
            rows += 1
    if in_place:
        write_path.replace(path)
    return rows, output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Absolute release root after unpacking.")
    parser.add_argument("--in-place", action="store_true", help="Rewrite released JSONL files in place.")
    args = parser.parse_args()

    root = str(Path(args.root).resolve())
    release = Path(root)
    files = list((release / "data").rglob("*.jsonl"))
    total = 0
    for path in files:
        rows, output = rewrite_jsonl(path, root, args.in_place)
        total += rows
        print(f"{path.relative_to(release)} -> {output.relative_to(release)} ({rows} rows)")
    print(f"Rewrote {total} rows across {len(files)} JSONL files.")


if __name__ == "__main__":
    main()
