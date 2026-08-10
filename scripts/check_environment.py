#!/usr/bin/env python3
"""Verify the interpreter against the pinned reproduction environment.

Refits reproduce the committed values only under the pinned versions. A
2015-16 NBA control run matched exactly under NumPy 1.26.4 and differed by
2.3e-05 under NumPy 2.2.6, changing one node's selected exogenous family.
Entry points that recompute fits therefore exit rather than run in an
unpinned interpreter.

Hash-only verification (``scripts/verify_frozen_results.py``) compares
committed bytes and does not call this module.
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

#: The two reproduction lines need different NumPy versions and must not share
#: one environment. See docs/ENVIRONMENT.md.
PROFILES = {
    "simulation": ROOT / "requirements-simulation.txt",
    "nba": ROOT / "requirements-nba.txt",
}

#: Packages whose versions change fitted numerical output.
NUMERICAL_PACKAGES = ("numpy", "scipy", "pandas")

PYTHON_EXPECTED = (3, 11)


def pinned_versions(profile: str) -> dict[str, str]:
    """Read ``name==version`` pins for one reproduction profile."""
    try:
        requirements = PROFILES[profile]
    except KeyError:
        raise SystemExit(
            f"Unknown profile {profile!r}; expected one of {sorted(PROFILES)}"
        ) from None
    pins: dict[str, str] = {}
    for line in requirements.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        name, version = line.split("==", 1)
        pins[name.strip().lower()] = version.strip()
    return pins


def installed_version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def check(profile: str, packages: tuple[str, ...] = NUMERICAL_PACKAGES) -> list[str]:
    """Return a list of human-readable environment problems (empty if clean)."""
    problems: list[str] = []
    if sys.version_info[:2] != PYTHON_EXPECTED:
        problems.append(
            f"python is {sys.version_info.major}.{sys.version_info.minor}, "
            f"expected {PYTHON_EXPECTED[0]}.{PYTHON_EXPECTED[1]}"
        )
    pins = pinned_versions(profile)
    for package in packages:
        expected = pins.get(package.lower())
        if expected is None:
            problems.append(f"{package} is not pinned for profile {profile}")
            continue
        actual = installed_version(package)
        if actual is None:
            problems.append(f"{package} is not installed (expected {expected})")
        elif actual != expected:
            problems.append(f"{package} is {actual}, expected {expected}")
    return problems


def require(profile: str, packages: tuple[str, ...] = NUMERICAL_PACKAGES) -> None:
    """Abort unless the interpreter matches the pinned reproduction environment."""
    problems = check(profile, packages)
    if not problems:
        return
    requirements = PROFILES[profile].name
    lines = [
        f"Refusing to recompute {profile} results in an unpinned environment.",
        "",
        "Problems:",
        *(f"  - {problem}" for problem in problems),
        "",
        f"Create the {profile} environment first:",
        "",
        f"  python -m venv .venv-{profile}",
        f"  .venv-{profile}/Scripts/python -m pip install -r {requirements}   # Windows",
        f"  .venv-{profile}/bin/python -m pip install -r {requirements}       # Linux/macOS",
        "",
        "then rerun this command with that interpreter.",
    ]
    raise SystemExit("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=sorted(PROFILES))
    parser.add_argument(
        "--packages",
        nargs="+",
        default=list(NUMERICAL_PACKAGES),
        help="packages to check (default: the numerically relevant ones)",
    )
    args = parser.parse_args()
    problems = check(args.profile, tuple(args.packages))
    pins = pinned_versions(args.profile)
    print(f"profile {args.profile}  ({PROFILES[args.profile].name})")
    print(f"python {sys.version.split()[0]}")
    for package in args.packages:
        print(
            f"{package} {installed_version(package)} "
            f"(pinned {pins.get(package.lower(), 'unpinned')})"
        )
    if problems:
        print("\nENVIRONMENT MISMATCH")
        for problem in problems:
            print(f"  - {problem}")
        raise SystemExit(1)
    print("\nenvironment matches the pinned reproduction environment")


if __name__ == "__main__":
    main()
