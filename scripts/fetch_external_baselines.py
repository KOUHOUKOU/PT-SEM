#!/usr/bin/env python3
"""Fetch and verify the exact author baseline sources used by PT-SEM.

The source directories are gitignored because neither pinned upstream commit
contains a redistribution license.  This script never substitutes a proxy and
never accepts a different commit.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "config/external_baselines.json"
DEFAULT_REPORT = ROOT / "work/external_baselines.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_git(directory: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(directory), *arguments],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"git {' '.join(arguments)} failed in {directory}: "
            f"{result.stderr.strip()}"
        )
    return result.stdout.strip()


def load_spec() -> dict[str, dict[str, Any]]:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


def fetch_sources() -> None:
    for name, entry in load_spec().items():
        target = ROOT / entry["directory"]
        if target.exists():
            if not (target / ".git").is_dir():
                raise RuntimeError(
                    f"Refusing to replace existing non-git path {target}"
                )
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            result = subprocess.run(
                ["git", "clone", "--no-checkout", entry["repository"], str(target)],
                text=True,
                check=False,
            )
            if result.returncode:
                raise RuntimeError(f"Could not clone {name} from {entry['repository']}")
        run_git(target, "fetch", "origin", entry["commit"])
        run_git(target, "checkout", "--detach", entry["commit"])


def _dependency_report() -> dict[str, dict[str, str | bool]]:
    modules = ("causallearn", "KDEpy", "sklearn", "scipy", "statsmodels", "tqdm")
    report: dict[str, dict[str, str | bool]] = {}
    for name in modules:
        available = importlib.util.find_spec(name) is not None
        version = ""
        if available:
            module = importlib.import_module(name)
            version = str(getattr(module, "__version__", "unknown"))
        report[name] = {"available": available, "version": version}
    return report


def preflight_external_baselines(import_check: bool = True) -> dict[str, Any]:
    report: dict[str, Any] = {
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "spec_sha256": sha256(SPEC_PATH),
        "dependencies": _dependency_report(),
        "sources": {},
    }
    failures: list[str] = []
    for dependency, state in report["dependencies"].items():
        if not state["available"]:
            failures.append(f"missing dependency: {dependency}")

    for name, entry in load_spec().items():
        target = (ROOT / entry["directory"]).resolve()
        source_state: dict[str, Any] = {
            "directory": str(target),
            "repository": entry["repository"],
            "required_commit": entry["commit"],
            "license_state": entry["upstream_license_state"],
            "file_sha256": {},
        }
        report["sources"][name] = source_state
        if not (target / ".git").is_dir():
            failures.append(f"{name}: missing git checkout at {target}")
            continue
        try:
            head = run_git(target, "rev-parse", "HEAD")
            origin = run_git(target, "remote", "get-url", "origin")
            source_state.update(head=head, origin=origin)
            if head != entry["commit"]:
                failures.append(f"{name}: HEAD {head} != pinned {entry['commit']}")
            if origin.rstrip("/") != entry["repository"].rstrip("/"):
                failures.append(f"{name}: origin URL differs from pinned repository")
            for relative in entry["required_files"]:
                path = target / relative
                if not path.is_file():
                    failures.append(f"{name}: missing required file {relative}")
                else:
                    source_state["file_sha256"][relative] = sha256(path)
            license_files = sorted(
                path.name
                for pattern in ("LICENSE*", "COPYING*", "NOTICE*")
                for path in target.glob(pattern)
                if path.is_file()
            )
            source_state["license_files"] = sorted(set(license_files))
        except Exception as exc:
            failures.append(f"{name}: {type(exc).__name__}: {exc}")

    if import_check and not failures:
        try:
            sys.path.insert(0, str(ROOT / "src"))
            import nba_core as d

            d._load_pbscm_author_modules()
            d._load_pbscm_pgf_author_modules()
            report["import_check"] = "passed"
        except Exception as exc:
            failures.append(f"author implementation import failed: {type(exc).__name__}: {exc}")
            report["import_check"] = "failed"

    if failures:
        report["status"] = "failed"
        report["failures"] = failures
    return report


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    if args.fetch:
        fetch_sources()
    report = preflight_external_baselines(import_check=True)
    atomic_json(args.report.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
