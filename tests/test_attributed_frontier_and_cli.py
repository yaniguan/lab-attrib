"""Tests for closed-loop Attributed `Chem-RRT` Frontier Campaign, CLI, and hub extension."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import lab_attrib.attributed_frontier as af_mod
from community_extensions import run_benchmark as run_extension_benchmark
from lab_attrib.attributed_frontier import (
    DEFAULT_CACHED_ESOL_CSV,
    SimulatedWetLabRun,
    _run_single_seed_regime,
    build_synthetic_esol_pool,
    is_locally_supported_in_tree,
    resolve_molecule_pool,
    run_attributed_frontier_benchmark,
)
from lab_attrib.blame import AssayTelemetry, SpectralFitMethod
from lab_attrib.cli import main as cli_main

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_esol_pool_is_deterministic_and_contains_frontier_hits() -> None:
    pool_a = build_synthetic_esol_pool(n_molecules=600, seed=7)
    pool_b = build_synthetic_esol_pool(n_molecules=600, seed=7)
    assert pool_a == pool_b
    hits = [m for m in pool_a if m.is_true_frontier_hit]
    assert len(hits) >= 10


def test_attributed_frontier_benchmark_synthetic_fallback_invariants() -> None:
    res = run_attributed_frontier_benchmark(
        force_synthetic_fallback=True,
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
    eq_budget = regimes["lab_attrib_equal_budget"]

    assert full["false_discovery_rate_pct"] == 0.0
    assert full["mean_false_positive_hits"] == 0.0
    assert full["mean_poisoned_memory_entries"] == 0.0
    assert full["all_ledger_chains_valid"] is True
    assert full["mean_certified_ledger_entries"] > 50.0
    assert full["mean_final_tau"] > full["mean_initial_tau"]
    assert full["mean_total_measurements"] > full["mean_primary_assays"]

    # Equal-budget regime strictly respects 8 * 18 = 144 total measurements (including retries + secondary)
    assert eq_budget["mean_total_measurements"] <= 8 * 18
    assert eq_budget["false_discovery_rate_pct"] == 0.0
    assert eq_budget["mean_poisoned_memory_entries"] == 0.0
    assert eq_budget["mean_verified_multi_ring_hits"] > static["mean_verified_multi_ring_hits"]

    assert naive["mean_poisoned_memory_entries"] > 20.0
    assert naive["false_discovery_rate_pct"] > 15.0
    assert unattr["mean_poisoned_memory_entries"] > 20.0

    assert full["mean_verified_multi_ring_hits"] > static["mean_verified_multi_ring_hits"]
    assert full["conformal_coverage_pct"] >= 82.0
    assert full["mean_frontier_rmse"] < unattr["mean_frontier_rmse"]

    # Sensor-recall sweep present in JSON and shows non-zero FDR & poisoned entries when recall < 1.0
    sweep = {row["fault_flag_recall"]: row for row in res["sensor_recall_sweep"]}
    assert set(sweep.keys()) == {1.0, 0.9, 0.8}
    assert sweep[0.8]["lab_attrib_full_closed_loop"]["mean_poisoned_memory_entries"] > 0.0
    assert sweep[0.8]["lab_attrib_full_closed_loop"]["false_discovery_rate_pct"] > 0.0


def test_issue15_single_sensor_missed_fault_increments_poisoned_memory_in_full_closed_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = build_synthetic_esol_pool(n_molecules=400, seed=0)
    seed_dom = [m for m in pool if m.rings <= 2 and m.features[1] <= 220.0]
    front_dom = [m for m in pool if not (m.rings <= 2 and m.features[1] <= 220.0)]

    real_sim = af_mod._simulate_wet_lab_execution
    injected_missed_faults = 0

    def _inject_one_missed_fault(
        mol: af_mod.MoleculeCandidate,
        *,
        seed: int,
        attempt: int,
        fault_flag_recall: float = 1.0,
        false_alarm_rate: float = 0.0,
    ) -> SimulatedWetLabRun:
        nonlocal injected_missed_faults
        if injected_missed_faults == 0 and attempt == 0:
            injected_missed_faults += 1
            return SimulatedWetLabRun(
                raw_measured_log_s=-8.85,
                telemetry=AssayTelemetry(
                    dispense_error_permille=12,
                    aspiration_clean=True,
                    incubation_window_clean=True,
                    seal_intact=True,
                    positive_control_passed=True,
                    blank_uncontaminated=True,
                    detector_calibrated=True,
                    precursor_overlap_permille=0,
                    spectral_residual_permille=25,
                    snr_tenths=95,
                    target_signal_present=False,
                    fit_method=SpectralFitMethod.STICK_NNLS,
                ),
                true_is_corrupted=True,
                true_has_ghost_peak=False,
            )
        return real_sim(
            mol,
            seed=seed,
            attempt=attempt,
            fault_flag_recall=fault_flag_recall,
            false_alarm_rate=false_alarm_rate,
        )

    monkeypatch.setattr(af_mod, "_simulate_wet_lab_execution", _inject_one_missed_fault)
    metrics = _run_single_seed_regime(
        "lab_attrib_full_closed_loop",
        seed_dom[:140],
        seed_dom[140:190],
        front_dom[50:],
        front_dom[:50],
        seed=11,
        n_rounds=2,
        batch_size=10,
        fault_flag_recall=1.0,
    )
    assert injected_missed_faults == 1
    assert metrics["poisoned_memory_entries"] == 1


def test_issue15_false_alarm_rate_increases_robot_retries() -> None:
    pool = build_synthetic_esol_pool(n_molecules=400, seed=0)
    seed_dom = [m for m in pool if m.rings <= 2 and m.features[1] <= 220.0]
    front_dom = [m for m in pool if not (m.rings <= 2 and m.features[1] <= 220.0)]

    m_zero = _run_single_seed_regime(
        "lab_attrib_full_closed_loop",
        seed_dom[:140],
        seed_dom[140:190],
        front_dom[50:],
        front_dom[:50],
        seed=11,
        n_rounds=3,
        batch_size=12,
        false_alarm_rate=0.0,
    )
    m_fa = _run_single_seed_regime(
        "lab_attrib_full_closed_loop",
        seed_dom[:140],
        seed_dom[140:190],
        front_dom[50:],
        front_dom[:50],
        seed=11,
        n_rounds=3,
        batch_size=12,
        false_alarm_rate=0.25,
    )
    assert m_fa["robot_retries_triggered"] > m_zero["robot_retries_triggered"]


def test_issue10_no_user_specific_paths_single_pass_csv_and_directional_knn_locality(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 1. Zero user-specific or machine-specific hardcoded paths in src/lab_attrib/
    assert "Users" not in DEFAULT_CACHED_ESOL_CSV.parts or REPO_ROOT in DEFAULT_CACHED_ESOL_CSV.parents
    for src_file in (REPO_ROOT / "src" / "lab_attrib").glob("*.py"):
        src_text = src_file.read_text(encoding="utf-8")
        assert "yanig" not in src_text
        assert "c71e01d3" not in src_text

    # 2. Single-pass CSV read in resolve_molecule_pool
    mini_csv = tmp_path / "mini_esol.csv"
    mini_csv.write_text(
        "Compound ID,smiles,Minimum Degree,Molecular Weight,Number of H-Bond Donors,"
        "Number of Rings,Number of Rotatable Bonds,Polar Surface Area,"
        "measured log solubility in mols per litre\n"
        "Ethanol,CCO,1,46.07,1,0,0,20.23,1.12\n"
        "Benzene,c1ccccc1,2,78.11,0,1,0,0.00,-1.64\n",
        encoding="utf-8",
    )
    read_calls = 0
    real_loader = af_mod.load_delaney_esol_csv

    def _counting_loader(path: Path) -> list[af_mod.MoleculeCandidate]:
        nonlocal read_calls
        read_calls += 1
        return real_loader(path)

    monkeypatch.setattr(af_mod, "load_delaney_esol_csv", _counting_loader)
    loaded, label = resolve_molecule_pool(mini_csv)
    assert read_calls == 1
    assert len(loaded) == 2
    assert "2 molecules" in label

    # 3. Directional Chem-RRT support: stepping-stone at +3.0 sigma along z_1 NEVER unlocks -3.0 sigma along z_4
    seed_cloud = [
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.2, -0.1, 0.0, 0.1, 0.0, 0.0),
        (-0.1, 0.2, 0.1, 0.0, -0.1, 0.0),
    ]
    support_with_z1_ray = [
        *seed_cloud,
        (1.2, 0.0, 0.0, 0.0, 0.0, 0.0),
        (2.1, 0.0, 0.0, 0.0, 0.0, 0.0),
        (3.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    ]
    near_z1_probe = (2.85, 0.05, 0.0, 0.0, 0.0, 0.0)
    antipodal_z4_unprobed = (0.0, 0.0, 0.0, -3.0, 0.0, 0.0)

    assert is_locally_supported_in_tree(
        near_z1_probe, support_with_z1_ray, local_tau=1.25, k_neighbors=3
    ) is True
    assert is_locally_supported_in_tree(
        antipodal_z4_unprobed, support_with_z1_ray, local_tau=1.25, k_neighbors=3
    ) is False


def test_issue10_short_budget_campaign_prevents_isotropic_tau_doubling_and_preserves_coverage() -> None:
    pool = build_synthetic_esol_pool(n_molecules=720, seed=0)
    seed_dom = [m for m in pool if m.rings <= 2 and m.features[1] <= 220.0]
    front_dom = [m for m in pool if not (m.rings <= 2 and m.features[1] <= 220.0)]

    coverages: list[float] = []
    for s in (11, 23, 37, 53, 71):
        metrics = _run_single_seed_regime(
            "lab_attrib_full_closed_loop",
            seed_dom[:200],
            seed_dom[200:260],
            front_dom[60:],
            front_dom[:60],
            seed=s,
            n_rounds=2,
            batch_size=10,
            eta_permille=420,
        )
        # Cannot double global tau on a 2-round under-sampled campaign
        assert metrics["final_tau"] < 1.65 * metrics["initial_tau"]
        coverages.append(float(metrics["conformal_coverage_90"]))

    mean_cov = sum(coverages) / len(coverages)
    assert mean_cov >= 0.90


def test_community_extension_entrypoint_returns_valid_summary() -> None:
    summary = run_extension_benchmark(csv_path=Path("nonexistent_fallback.csv"))
    assert summary["benchmark"] == "lab_attrib_attributed_chem_rrt_frontier"
    assert set(summary["regimes"].keys()) == {
        "naive_raw_writeback",
        "pre_gate_only_static",
        "unattributed_frontier_probes",
        "lab_attrib_full_closed_loop",
        "lab_attrib_equal_budget",
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


def test_cli_benchmark_writes_json_artifact_and_checks_drift(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out_json = tmp_path / "bench_out.json"
    rc = cli_main(
        [
            "benchmark",
            "--synthetic-fallback",
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

    # Zero-drift check passes on matching reference JSON
    rc_ok = cli_main(
        [
            "benchmark",
            "--synthetic-fallback",
            "--rounds",
            "4",
            "--batch-size",
            "12",
            "--check-drift",
            str(out_json),
        ]
    )
    assert rc_ok == 0
    capsys.readouterr()

    # Drift check fails non-zero (2) when reference JSON is tampered
    tampered = dict(file_payload)
    tampered["n_seeds"] = 999
    tampered_path = tmp_path / "tampered.json"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    rc_bad = cli_main(
        [
            "benchmark",
            "--synthetic-fallback",
            "--rounds",
            "4",
            "--batch-size",
            "12",
            "--check-drift",
            str(tampered_path),
        ]
    )
    assert rc_bad == 2
