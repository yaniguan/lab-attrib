"""Hub-parity benchmark entrypoint for the Attributed `Chem-RRT` Frontier Campaign."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lab_attrib.attributed_frontier import run_attributed_frontier_benchmark


def run_benchmark(csv_path: str | Path | None = None) -> dict[str, Any]:
    """Execute the 4-regime closed-loop attributed frontier benchmark and return summary dict."""
    return run_attributed_frontier_benchmark(csv_path=csv_path)
