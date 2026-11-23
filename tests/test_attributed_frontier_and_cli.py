"""Tests for closed-loop Attributed `Chem-RRT` Frontier Campaign, CLI, and hub extension."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from community_extensions import run_benchmark as run_extension_benchmark
from lab_attrib.attributed_frontier import (
    build_synthetic_esol_pool,
    run_attributed_frontier_benchmark,
)
from lab_attrib.cli import main as cli_main


def test_synthetic_esol_pool_is_deterministic_and_contains_frontier_hits() -> None:
    pool_a = build_synthetic_esol_pool(n_molecules=600, seed=7)
    pool_b = build_synthetic_esol_pool(n_molecules=600, seed=7)
    assert pool_a == pool_b
    hits = [m for m in pool_a if m.is_true_frontier_hit]
    assert len(hits) >= 10


def test_attributed_frontier_benchmark_synthetic_fallback_invariants() -> None:
    res = run_attributed_frontier_benchmark(
        csv_path=Path("nonexistent_delaney_esol_fallback.csv"),
        seeds=(11, 23, 37),
        n_rounds=8,
        batch_size=18,
    )
    assert "synthetic_esol_fallback" in res["dataset"]
    regimes = res["regimes"]

    naive = regimes["naive_raw_writeback"]
    static = regimes["pre_gate_only_static"]
    unattr = regimes["unattributed_frontier_probes"]
    full = regimes["lab_attrib_full_closed_loop"]

    # Full closed-loop `lab-attrib` guarantees zero false discovery and zero poisoned memory
    assert full["false_discovery_rate_pct"] == 0.0
    assert full["mean_false_positive_hits"] == 0.0
    assert full["mean_poisoned_memory_entries"] == 0.0
    assert full["all_ledger_chains_valid"] is True
    assert full["mean_certified_ledger_entries"] > 50.0
    assert full["mean_final_tau"] > full["mean_initial_tau"]

    # Naive and unattributed regimes suffer severe memory poisoning and false positives
    assert naive["mean_poisoned_memory_entries"] > 20.0
    assert naive["false_discovery_rate_pct"] > 15.0
    assert unattr["mean_poisoned_memory_entries"] > 20.0

    # Full closed-loop discovers more verified multi-ring hits than static pre-gate
    # and achieves substantially better conformal calibration & lower RMSE than unattributed probes
    assert full["mean_verified_multi_ring_hits"] > static["mean_verified_multi_ring_hits"]
    assert full["conformal_coverage_pct"] >= 82.0
    assert full["mean_frontier_rmse"] < unattr["mean_frontier_rmse"]


def test_community_extension_entrypoint_returns_valid_summary() -> None:
    summary = run_extension_benchmark(csv_path=Path("nonexistent_fallback.csv"))
    assert summary["benchmark"] == "lab_attrib_attributed_chem_rrt_frontier"
    assert set(summary["regimes"].keys()) == {
        "naive_raw_writeback",
        "pre_gate_only_static",
        "unattributed_frontier_probes",
        "lab_attrib_full_closed_loop",
    }


def test_cli_demo_text_and_json(capsys: pytest.CaptureFixture[str]) -> None:
    rc = cli_main(["demo"])
    assert rc == 0
    out_text = capsys.readouterr().out
    assert "A-Lab Precursor Ghost-Peak Trap" in out_text

    rc_json = cli_main(["demo", "--json"])
    assert rc_json == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["certified_ledger_demo"]["chain_valid"] is True


def test_cli_attribute_inline_and_file_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tel_file = tmp_path / "telemetry.json"
    tel_file.write_text(
        json.dumps(
            {
                "dispense_error_permille": 10,
                "aspiration_clean": True,
                "incubation_window_clean": True,
                "seal_intact": True,
                "positive_control_passed": True,
                "blank_uncontaminated": True,
                "detector_calibrated": True,
                "precursor_overlap_permille": 500,
                "spectral_residual_permille": 40,
                "snr_tenths": 80,
                "target_signal_present": True,
            }
        ),
        encoding="utf-8",
    )
    orth_inline = json.dumps(
        {
            "modality": "Raman",
            "instrument_calibrated": True,
            "orthogonal_snr_tenths": 90,
            "selectivity_clean": True,
            "target_confirmed": False,
        }
    )
    rc = cli_main(
        ["attribute", "--telemetry", str(tel_file), "--orthogonal", orth_inline, "--json"]
    )
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["outcome"] == "CERTIFIED_NEGATIVE"
    assert data["resolved_via_orthogonal"] is True


def test_cli_benchmark_writes_json_artifact(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out_json = tmp_path / "bench_out.json"
    rc = cli_main(
        [
            "benchmark",
            "--csv",
            str(tmp_path / "missing.csv"),
            "--rounds",
            "4",
            "--batch-size",
            "12",
            "--output",
            str(out_json),
            "--json",
        ]
    )
    assert rc == 0
    stdout_payload = json.loads(capsys.readouterr().out)
    file_payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert stdout_payload == file_payload
    assert file_payload["regimes"]["lab_attrib_full_closed_loop"]["false_discovery_rate_pct"] == 0.0
