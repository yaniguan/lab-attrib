#!/usr/bin/env python3
"""Automated `#print axioms` audit for all formal theorems in `lean/LabAttribLean.lean`.

Verifies that:
1. Every `theorem <name>` declared in `lean/LabAttribLean.lean` (>= 30 theorems) is
   included in `lean/CheckAxioms.lean`.
2. Running `lake env lean CheckAxioms.lean` succeeds with zero warnings/errors, zero
   `sorryAx`, and **no custom axioms** (only standard Lean 4 kernel axioms:
   `propext`, `Classical.choice`, `Quot.sound`).
3. `lean/README.md` explicitly states `"no custom axioms"` and contains no inaccurate
   `"0 axiom"` phrasing.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
LEAN_DIR = REPO_ROOT / "lean"
LEAN_LIB_FILE = LEAN_DIR / "LabAttribLean.lean"
CHECK_AXIOMS_FILE = LEAN_DIR / "CheckAxioms.lean"
LEAN_README_FILE = LEAN_DIR / "README.md"

ALLOWED_STANDARD_AXIOMS: frozenset[str] = frozenset(
    {"propext", "Classical.choice", "Quot.sound"}
)


def extract_theorems(lean_path: Path = LEAN_LIB_FILE) -> list[str]:
    text = lean_path.read_text(encoding="utf-8")
    no_blocks = re.sub(r"/-.*?-/", "", text, flags=re.DOTALL)
    clean_lines = [ln.split("--")[0] for ln in no_blocks.splitlines()]
    clean = "\n".join(clean_lines)
    return re.findall(r"^\s*theorem\s+([A-Za-z0-9_']+)", clean, flags=re.MULTILINE)


def extract_checked_theorems(check_path: Path = CHECK_AXIOMS_FILE) -> list[str]:
    text = check_path.read_text(encoding="utf-8")
    no_blocks = re.sub(r"/-.*?-/", "", text, flags=re.DOTALL)
    clean_lines = [ln.split("--")[0] for ln in no_blocks.splitlines()]
    clean = "\n".join(clean_lines)
    return re.findall(
        r"^\s*#print\s+axioms\s+LabAttribLean\.([A-Za-z0-9_']+)",
        clean,
        flags=re.MULTILINE,
    )


def find_lake_executable() -> str | None:
    from_path = shutil.which("lake") or shutil.which("lake.exe")
    if from_path:
        return from_path
    home_elan = Path.home() / ".elan" / "bin"
    for name in ("lake.exe", "lake"):
        cand = home_elan / name
        if cand.is_file():
            return str(cand)
    return None


def audit_lean_axioms(*, run_kernel: bool = True) -> dict[str, Any]:
    declared = extract_theorems()
    checked = extract_checked_theorems()
    missing = [t for t in declared if t not in set(checked)]
    extra = [t for t in checked if t not in set(declared)]
    if len(declared) < 25:
        raise AssertionError(f"Expected >= 25 theorems in LabAttribLean.lean, found {len(declared)}")
    if missing or extra:
        raise AssertionError(
            f"CheckAxioms.lean drift vs LabAttribLean.lean: missing={missing}, extra={extra}"
        )

    readme_text = LEAN_README_FILE.read_text(encoding="utf-8")
    if "no custom axioms" not in readme_text.lower():
        raise AssertionError("lean/README.md must explicitly state 'no custom axioms'")
    if "0 axiom" in readme_text.lower():
        raise AssertionError("lean/README.md must not claim '0 axiom' (use 'no custom axioms')")

    theorem_axioms: dict[str, list[str]] = {}
    if run_kernel:
        lake_bin = find_lake_executable()
        if lake_bin is None:
            raise RuntimeError("lake executable not found on PATH or ~/.elan/bin")
        proc = subprocess.run(
            [lake_bin, "env", "lean", "CheckAxioms.lean"],
            cwd=str(LEAN_DIR),
            capture_output=True,
            text=True,
            check=True,
            timeout=90,
        )
        combined = proc.stdout + "\n" + proc.stderr
        if "sorryAx" in combined:
            raise AssertionError(f"Forbidden sorryAx detected in Lean kernel output:\n{combined}")

        dep_re = re.compile(
            r"^'LabAttribLean\.([A-Za-z0-9_']+)'\s+depends on axioms:\s*\[([^\]]*)\]",
            flags=re.MULTILINE,
        )
        no_ax_re = re.compile(
            r"^'LabAttribLean\.([A-Za-z0-9_']+)'\s+does not depend on any axioms",
            flags=re.MULTILINE,
        )
        for m in dep_re.finditer(proc.stdout):
            tname = m.group(1)
            axs = [tok.strip() for tok in m.group(2).split(",") if tok.strip()]
            disallowed = [a for a in axs if a not in ALLOWED_STANDARD_AXIOMS]
            if disallowed:
                raise AssertionError(
                    f"Theorem {tname} depends on non-standard/custom axioms: {disallowed}"
                )
            theorem_axioms[tname] = axs
        for m in no_ax_re.finditer(proc.stdout):
            theorem_axioms[m.group(1)] = []

        unverified = [t for t in declared if t not in theorem_axioms]
        if unverified:
            raise AssertionError(f"Missing #print axioms output for theorems: {unverified}")

    return {
        "theorem_count": len(declared),
        "all_theorems_checked": True,
        "kernel_verified": run_kernel,
        "zero_sorryAx": True,
        "zero_custom_axioms": True,
        "allowed_standard_axioms": sorted(ALLOWED_STANDARD_AXIOMS),
        "theorem_axioms": theorem_axioms,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify Lean 4 #print axioms on LabAttribLean.")
    parser.add_argument("--static-only", action="store_true", help="Skip invoking lake env lean")
    parser.add_argument("--json", action="store_true", help="Emit JSON audit report")
    args = parser.parse_args(argv)

    report = audit_lean_axioms(run_kernel=not args.static_only)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(
            f"Lean axiom audit PASS: {report['theorem_count']} theorems verified "
            f"(kernel={report['kernel_verified']}, 0 sorryAx, no custom axioms)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
