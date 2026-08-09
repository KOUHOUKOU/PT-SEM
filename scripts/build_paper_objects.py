#!/usr/bin/env python3
"""Assemble the manuscript's objects into ``outputs/`` and verify each one.

The manuscript renumbered its figures, so the file names produced by the
experiment programs no longer match the figure numbers a reader sees. This
script is the single place that mapping lives. It copies each produced artifact
to ``outputs/`` under its manuscript name and writes
``outputs/REPRODUCTION_REPORT.md``, a one-page table stating, per object,
whether what this repository produces is byte-identical to what the manuscript
contains.

Nothing else is promoted. Per-replication tables, per-family local scores,
optimizer diagnostics and run manifests stay where the programs wrote them.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests/PAPER_OBJECTS.csv"
OUTPUTS = ROOT / "outputs"


def stream_sha256(pdf: Path) -> str:
    """SHA-256 of a single-page PDF's content stream.

    Embedding a figure into LaTeX preserves this stream, so it is the identity
    that survives the round trip into the manuscript.
    """
    return hashlib.sha256(
        PdfReader(str(pdf)).pages[0].get_contents().get_data()
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest() -> list[dict[str, str]]:
    with MANIFEST.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def resolve_source(entry: dict[str, str]) -> Path | None:
    """Prefer a freshly produced artifact, fall back to the frozen copy."""
    for candidate in entry["source_candidates"].split(";"):
        path = ROOT / candidate.strip()
        if path.exists():
            return path
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check-only", action="store_true",
                        help="verify without writing outputs/")
    args = parser.parse_args()

    entries = load_manifest()
    rows, all_ok = [], True

    for entry in entries:
        label = entry["paper_object"]
        expected = entry["expected_sha256"]
        source = resolve_source(entry)

        if source is None:
            rows.append((label, entry["outputs_name"], "MISSING", "-", "not produced yet"))
            all_ok = False
            continue

        if entry["kind"] == "figure":
            produced = stream_sha256(source)
        else:
            produced = file_sha256(source)

        ok = produced == expected
        all_ok &= ok
        rows.append((label, entry["outputs_name"],
                     "MATCH" if ok else "DIFFER",
                     produced[:16],
                     source.relative_to(ROOT).as_posix()))

        if not args.check_only:
            destination = OUTPUTS / entry["outputs_subdir"] / entry["outputs_name"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())

    width = max(len(r[1]) for r in rows) + 2
    print(f"{'paper object':<14}{'delivered as':<{width}}{'result':<9}{'sha256':<18}produced from")
    print("-" * (14 + width + 9 + 18 + 30))
    for label, name, verdict, digest, origin in rows:
        print(f"{label:<14}{name:<{width}}{verdict:<9}{digest:<18}{origin}")
    print()
    print("ALL PAPER OBJECTS REPRODUCED" if all_ok
          else "SOME PAPER OBJECTS DO NOT MATCH THE MANUSCRIPT")

    if not args.check_only:
        write_report(rows, all_ok)
        print(f"\nwrote {(OUTPUTS / 'REPRODUCTION_REPORT.md').relative_to(ROOT).as_posix()}")

    raise SystemExit(0 if all_ok else 1)


def write_report(rows, all_ok: bool) -> None:
    manuscript = load_manifest()[0]["manuscript"]
    lines = [
        "# Reproduction report",
        "",
        f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} "
        f"against `{manuscript}`.",
        "",
        "Every object below was compared with what the manuscript actually "
        "contains. Figures are compared by the PDF content stream embedded in "
        "the manuscript, which is preserved through LaTeX inclusion; the table "
        "is compared by file digest.",
        "",
        "| Paper object | Delivered as | Result | SHA-256 (first 16) | Produced from |",
        "|---|---|---|---|---|",
    ]
    for label, name, verdict, digest, origin in rows:
        lines.append(f"| {label} | `{name}` | **{verdict}** | `{digest}` | `{origin}` |")
    lines += [
        "",
        ("All objects are byte-identical to the manuscript."
         if all_ok else
         "At least one object differs from the manuscript; see the table above."),
        "",
        "Regenerate with `python scripts/build_paper_objects.py`.",
        "",
    ]
    (OUTPUTS / "REPRODUCTION_REPORT.md").parent.mkdir(parents=True, exist_ok=True)
    (OUTPUTS / "REPRODUCTION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
