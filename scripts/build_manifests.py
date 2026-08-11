#!/usr/bin/env python3
"""Build deterministic SHA-256 inventories for immutable publication inputs."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
#: ``outputs/`` is derived: its digests are recorded in
#: ``manifests/PAPER_OBJECTS.csv`` and its report is regenerated on demand.
INCLUDED = (
    "src",
    "config",
    "data",
    "experiments",
    "results",
    "provenance",
    "audit",
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    paths: list[Path] = []
    for name in INCLUDED:
        base = ROOT / name
        if base.exists():
            paths.extend(
                path
                for path in base.rglob("*")
                if path.is_file()
                and "__pycache__" not in path.parts
                and path.suffix not in {".pyc", ".pyo"}
            )
    rows = [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": digest(path),
        }
        for path in sorted(set(paths), key=lambda item: item.relative_to(ROOT).as_posix())
    ]
    destination = ROOT / "manifests/SHA256SUMS.csv"
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "size_bytes", "sha256"))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} entries to {destination}")


if __name__ == "__main__":
    main()
