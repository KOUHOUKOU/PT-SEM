#!/usr/bin/env python3
"""Assemble the paper's figures and table into ``outputs/`` and verify them.

Each object is copied to ``outputs/`` under its paper name and compared with
the expected digest recorded in ``manifests/PAPER_OBJECTS.csv``. Figures are
compared by PDF content stream, which is preserved when a figure is included
in a document. The result is written to ``outputs/REPRODUCTION_REPORT.md``.

Optionally, the released paper objects can be cross-checked against a supplied
manuscript PDF via ``--manuscript <path>``. The PDF is treated as external
input: it is not recorded in the manifest or anywhere else in the repository.

Per-replication tables, per-family local scores, optimizer diagnostics and run
manifests are left where the programs wrote them.
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
    """SHA-256 of a single-page PDF's content stream."""
    return hashlib.sha256(
        PdfReader(str(pdf)).pages[0].get_contents().get_data()
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def embedded_stream_digests(manuscript: Path) -> set[str]:
    """SHA-256 of every figure stream embedded in a manuscript PDF."""
    digests = set()
    for page in PdfReader(str(manuscript)).pages:
        resources = page.get("/Resources")
        if not resources:
            continue
        xobjects = resources.get("/XObject")
        if not xobjects:
            continue
        for reference in xobjects.values():
            obj = reference.get_object()
            if obj.get("/Subtype") == "/Form":
                digests.add(hashlib.sha256(obj.get_data()).hexdigest())
    return digests


def load_manifest() -> list[dict[str, str]]:
    with MANIFEST.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def resolve_source(entry: dict[str, str]) -> Path | None:
    """Prefer a freshly produced artifact, fall back to the released copy."""
    for candidate in entry["source_candidates"].split(";"):
        path = ROOT / candidate.strip()
        if path.exists():
            return path
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--check-only", action="store_true",
                        help="verify without writing outputs/")
    parser.add_argument("--manuscript", type=Path,
                        help="optional manuscript PDF to cross-check the figures against")
    args = parser.parse_args()

    manuscript_digests: set[str] | None = None
    if args.manuscript is not None:
        if not args.manuscript.exists():
            raise SystemExit(f"No such manuscript PDF: {args.manuscript}")
        manuscript_digests = embedded_stream_digests(args.manuscript)

    entries = load_manifest()
    rows, all_ok = [], True

    for entry in entries:
        label = entry["paper_object"]
        expected = entry["expected_sha256"]
        source = resolve_source(entry)

        if source is None:
            rows.append((label, entry["outputs_name"], "MISSING", "-", "not produced yet", ""))
            all_ok = False
            continue

        if entry["kind"] == "figure":
            produced = stream_sha256(source)
        else:
            produced = file_sha256(source)

        ok = produced == expected
        all_ok &= ok

        cross = ""
        if manuscript_digests is not None and entry["kind"] == "figure":
            embedded = produced in manuscript_digests
            all_ok &= embedded
            cross = "EMBEDDED" if embedded else "NOT FOUND"

        rows.append((label, entry["outputs_name"],
                     "MATCH" if ok else "DIFFER",
                     produced[:16],
                     source.relative_to(ROOT).as_posix(),
                     cross))

        if not args.check_only:
            destination = OUTPUTS / entry["outputs_subdir"] / entry["outputs_name"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())

    width = max(len(r[1]) for r in rows) + 2
    header = f"{'paper object':<14}{'delivered as':<{width}}{'result':<9}{'sha256':<18}produced from"
    if manuscript_digests is not None:
        header += "   in manuscript"
    print(header)
    print("-" * len(header))
    for label, name, verdict, digest, origin, cross in rows:
        line = f"{label:<14}{name:<{width}}{verdict:<9}{digest:<18}{origin}"
        if manuscript_digests is not None:
            line += f"   {cross}"
        print(line)
    print()
    print("ALL PAPER OBJECTS VERIFIED" if all_ok
          else "SOME PAPER OBJECTS DID NOT VERIFY")

    if not args.check_only:
        write_report(rows, all_ok, manuscript_digests is not None)
        print(f"\nwrote {(OUTPUTS / 'REPRODUCTION_REPORT.md').relative_to(ROOT).as_posix()}")

    raise SystemExit(0 if all_ok else 1)


def write_report(rows, all_ok: bool, cross_checked: bool) -> None:
    lines = [
        "# Reproduction report",
        "",
        f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by "
        "`scripts/build_paper_objects.py`.",
        "",
        "Figures are compared by PDF content stream; the table is compared by "
        "file digest. Expected values are recorded in "
        "`manifests/PAPER_OBJECTS.csv`.",
        "",
    ]
    head = "| Paper object | Delivered as | Result | SHA-256 (first 16) | Produced from |"
    rule = "|---|---|---|---|---|"
    if cross_checked:
        head = head[:-1] + " In manuscript |"
        rule = rule[:-1] + "---|"
    lines += [head, rule]
    for label, name, verdict, digest, origin, cross in rows:
        row = f"| {label} | `{name}` | **{verdict}** | `{digest}` | `{origin}` |"
        if cross_checked:
            row = row[:-1] + f" {cross} |"
        lines.append(row)
    lines += [
        "",
        ("All objects match their expected digests."
         if all_ok else
         "At least one object did not verify; see the table above."),
        "",
    ]
    (OUTPUTS / "REPRODUCTION_REPORT.md").parent.mkdir(parents=True, exist_ok=True)
    (OUTPUTS / "REPRODUCTION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
