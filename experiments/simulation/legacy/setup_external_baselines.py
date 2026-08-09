"""Install Python dependencies and pin external author repositories."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def ensure_package(import_name: str, pip_spec: str) -> None:
    if importlib.util.find_spec(import_name) is None:
        run([sys.executable, "-m", "pip", "install", pip_spec])


def ensure_repository(relative_path: str, url: str, commit: str) -> None:
    destination = ROOT / relative_path
    if (destination / ".git").exists():
        current_commit = subprocess.run(
            ["git", "-C", str(destination), "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if current_commit == commit:
            print(
                f"+ {relative_path} already pinned at {commit}; no fetch needed.",
                flush=True,
            )
            return
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", url, str(destination)])
    run(["git", "-C", str(destination), "fetch", "--depth", "1", "origin", commit])
    run(["git", "-C", str(destination), "checkout", "--detach", commit])


def find_rscript() -> Path | None:
    located = shutil.which("Rscript")
    if located:
        return Path(located)
    candidates = sorted(
        Path("C:/Program Files/R").glob("R-*/bin/Rscript.exe"), reverse=True
    )
    return candidates[0] if candidates else None


def ensure_cpcm_r_environment() -> None:
    rscript = find_rscript()
    if rscript is None:
        if shutil.which("winget") is None:
            raise RuntimeError(
                "CPCM requires R. Install R >= 4.4 or make winget available."
            )
        run(
            [
                "winget",
                "install",
                "--id",
                "RProject.R",
                "--exact",
                "--silent",
                "--accept-package-agreements",
                "--accept-source-agreements",
                "--disable-interactivity",
            ]
        )
        rscript = find_rscript()
    if rscript is None:
        raise RuntimeError("R installation completed but Rscript could not be found.")

    r_library = ROOT / "external" / "R_library"
    r_library.mkdir(parents=True, exist_ok=True)
    expression = (
        f".libPaths(c({r_library.as_posix()!r}, .libPaths()));"
        "pkgs<-c('dHSIC','bnlearn','gamlss','stringr','dplyr');"
        "missing<-pkgs[!sapply(pkgs,requireNamespace,quietly=TRUE)];"
        "if(length(missing)) install.packages(missing,repos='https://cloud.r-project.org',Ncpus=4);"
        "required<-c('mgcv','dHSIC','bnlearn','MASS','gamlss','stringr','dplyr');"
        "if(!all(sapply(required,requireNamespace,quietly=TRUE))) quit(status=2)"
    )
    environment = os.environ.copy()
    environment["R_LIBS_USER"] = str(r_library)
    print("+", rscript, "-e <install CPCM packages>", flush=True)
    subprocess.run(
        [str(rscript), "--vanilla", "-e", expression],
        cwd=ROOT,
        env=environment,
        check=True,
    )


def main() -> None:
    if shutil.which("git") is None:
        raise RuntimeError("Git is required to fetch the author baseline repositories.")
    ensure_package("causallearn", "causal-learn")
    ensure_package("KDEpy", "KDEpy==1.1.4")
    ensure_package("sklearn", "scikit-learn")
    ensure_package("statsmodels", "statsmodels")
    ensure_package("tqdm", "tqdm")
    ensure_repository(
        "external/PBSCM",
        "https://github.com/DMIRLAB-Group/PBSCM.git",
        "08ba9ba806eacd8162f5f907c941d946a0ce69f0",
    )
    ensure_repository(
        "external/PBSCM_PGF",
        "https://github.com/DMIRLAB-Group/PBSCM-PGF.git",
        "5fdea2b7e1b796fd0b48c6e0b3be51959b4fa2c1",
    )
    print("PC-RCIT, PB-SCM (cumulant and PGF), and ODS-GLMLasso environments are ready.")


if __name__ == "__main__":
    main()
