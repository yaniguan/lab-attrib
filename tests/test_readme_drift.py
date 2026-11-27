"""Documentation & benchmark drift guards (`README.md`, `docs/ARCHITECTURE_AND_HISTORY.md`).

Ensures that:
1. `results/real_spectra_benchmark.json` is committed, verified against SHA-256-pinned
   real experimental assets (`primary_data_source == "real_experimental"`), and matches
   the Real vs. Controlled-Synthetic summary table in BOTH `README.md` and
   `docs/ARCHITECTURE_AND_HISTORY.md` byte-for-byte.
2. `scripts/benchmark_real_spectra.py --offline` fails non-zero (`2`) without cached
   SHA-256-verified real spectra unless `--allow-synthetic` is explicitly passed.
3. Closed-loop ESOL metrics (`benchmarks/results/lab_attrib_benchmark.json`), open
   dataset registry IDs, and Lean 4 theorem count (`30` theorems, `no custom axioms`)
   have zero drift across documentation files.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from lab_attrib import OPEN_CHARACTERIZATION_DATASETS

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(name: str, rel_path: str) -> Any:
    script_path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_real_spectra_benchmark_json_and_readme_tables_zero_drift() -> None:
    bench_mod = _load_script_module(
        "benchmark_real_spectra_mod", "scripts/benchmark_real_spectra.py"
    )
    results_path = REPO_ROOT / "results" / "real_spectra_benchmark.json"
    assert results_path.is_file(), "Missing committed results/real_spectra_benchmark.json"

    payload = json.loads(results_path.read_text(encoding="utf-8"))
    assert payload["benchmark_schema"] == "lab-attrib-real-spectra/v1"
    assert payload["primary_data_source"] == "real_experimental"
    assert payload["real_experimental_suites_executed"] is True
    assert payload["sha256_all_verified"] is True
    assert payload["pinned_asset_count"] == len(bench_mod.PINNED_REAL_SPECTRA_ASSETS)
    assert payload["pinned_asset_count"] >= 45

    suites = payload["suites"]
    for real_suite_id in (
        "alab_gpss_xrd",
        "rruff_dara_powder_xrd",
        "rruff_rod_raman",
        "massspecgym_msms",
    ):
        assert real_suite_id in suites
        assert suites[real_suite_id]["data_source"] == "real_experimental"

    assert suites["synthetic_controlled_ablation"]["data_source"] == "synthetic_ablation"

    # Quantitative sanity invariants across all 4 real experimental suites
    alab_m = suites["alab_gpss_xrd"]["methods"]
    assert alab_m["shift_voigt_l1_nnls"]["false_discovery_rate_pct"] == 0.0
    assert (
        alab_m["shift_voigt_l1_nnls"]["false_discovery_rate_pct"]
        < alab_m["naive_peak_cosine"]["false_discovery_rate_pct"]
    )
    assert (
        alab_m["shift_voigt_l1_nnls"]["mean_rwp_permille"]
        < alab_m["rigid_gaussian_nnls"]["mean_rwp_permille"]
    )

    xrd_m = suites["rruff_dara_powder_xrd"]["methods"]
    assert (
        xrd_m["shift_voigt_l1_nnls"]["false_discovery_rate_pct"]
        < xrd_m["rigid_gaussian_nnls"]["false_discovery_rate_pct"]
    )
    assert (
        xrd_m["shift_voigt_l1_nnls"]["mean_rwp_permille"]
        < xrd_m["rigid_gaussian_nnls"]["mean_rwp_permille"]
    )

    ram_m = suites["rruff_rod_raman"]["methods"]
    assert ram_m["shift_voigt_l1_nnls"]["false_discovery_rate_pct"] == 0.0
    assert (
        ram_m["shift_voigt_l1_nnls"]["mean_rwp_permille"]
        < ram_m["rigid_gaussian_nnls"]["mean_rwp_permille"]
    )

    ms_m = suites["massspecgym_msms"]["methods"]
    assert (
        ms_m["weighted_spectral_entropy_chimera_gate"]["top1_accuracy_pct"]
        > ms_m["msms_cosine"]["top1_accuracy_pct"]
    )
    assert (
        ms_m["weighted_spectral_entropy_chimera_gate"]["false_discovery_rate_pct"]
        < ms_m["msms_cosine"]["false_discovery_rate_pct"]
    )
    assert ms_m["weighted_spectral_entropy_chimera_gate"]["chimera_quarantine_rate_pct"] >= 90.0

    # Byte-for-byte table parity in README.md and docs/ARCHITECTURE_AND_HISTORY.md
    expected_table = bench_mod.render_real_vs_synthetic_markdown_table(payload)
    begin_m = bench_mod.TABLE_BEGIN_MARKER
    end_m = bench_mod.TABLE_END_MARKER

    for doc_rel in ("README.md", "docs/ARCHITECTURE_AND_HISTORY.md"):
        doc_text = (REPO_ROOT / doc_rel).read_text(encoding="utf-8")
        assert begin_m in doc_text and end_m in doc_text, (
            f"Missing Real vs. Controlled-Synthetic table markers in {doc_rel}"
        )
        start = doc_text.index(begin_m)
        stop = doc_text.index(end_m) + len(end_m)
        actual_block = doc_text[start:stop]
        assert actual_block == expected_table, (
            f"Real vs. Controlled-Synthetic table in {doc_rel} drifted from "
            f"results/real_spectra_benchmark.json"
        )


def test_benchmark_real_spectra_offline_fails_closed_without_cache_unless_allow_synthetic(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bench_mod = _load_script_module(
        "benchmark_real_spectra_offline_mod", "scripts/benchmark_real_spectra.py"
    )
    empty_cache = tmp_path / "empty_open_datasets"
    empty_cache.mkdir()

    # 1. --offline WITHOUT --allow-synthetic MUST exit non-zero (2)
    rc_fail = bench_mod.main(["--offline", "--data-dir", str(empty_cache), "--no-write"])
    captured = capsys.readouterr()
    assert rc_fail != 0
    assert rc_fail == 2
    assert "Offline real-spectra verification failed" in captured.err

    # 2. --offline WITH --allow-synthetic succeeds and explicitly marks synthetic_ablation
    synth_out = tmp_path / "synth_only.json"
    rc_synth = bench_mod.main(
        [
            "--offline",
            "--allow-synthetic",
            "--data-dir",
            str(empty_cache),
            "--output",
            str(synth_out),
        ]
    )
    assert rc_synth == 0
    synth_payload = json.loads(synth_out.read_text(encoding="utf-8"))
    assert synth_payload["primary_data_source"] == "synthetic_ablation"
    assert synth_payload["real_experimental_suites_executed"] is False
    assert list(synth_payload["suites"]) == ["synthetic_controlled_ablation"]

    # 3. Corrupted SHA-256 file in cache also fails non-zero under --offline
    first_asset = bench_mod.PINNED_REAL_SPECTRA_ASSETS[0]
    corrupted_file = empty_cache / first_asset.relative_path
    corrupted_file.parent.mkdir(parents=True, exist_ok=True)
    corrupted_file.write_bytes(b"corrupted_bytes")
    rc_corrupt = bench_mod.main(["--offline", "--data-dir", str(empty_cache), "--no-write"])
    err_corrupt = capsys.readouterr().err
    assert rc_corrupt == 2
    assert "SHA-256 mismatch" in err_corrupt


def test_closed_loop_benchmark_dataset_registry_and_lean_axiom_docs_parity() -> None:
    axiom_mod = _load_script_module("check_lean_axioms_drift_mod", "scripts/check_lean_axioms.py")
    theorems = axiom_mod.extract_theorems()
    n_thms = len(theorems)
    assert n_thms == 30

    esol_bench = json.loads(
        (REPO_ROOT / "benchmarks" / "results" / "lab_attrib_benchmark.json").read_text(
            encoding="utf-8"
        )
    )
    full_loop = esol_bench["regimes"]["lab_attrib_full_closed_loop"]

    readme_text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    arch_text = (REPO_ROOT / "docs" / "ARCHITECTURE_AND_HISTORY.md").read_text(encoding="utf-8")
    lean_readme = (REPO_ROOT / "lean" / "README.md").read_text(encoding="utf-8")

    for ds in OPEN_CHARACTERIZATION_DATASETS:
        assert f"`{ds.dataset_id}`" in readme_text
        assert f"`{ds.dataset_id}`" in arch_text

    for doc_name, doc_text in (
        ("README.md", readme_text),
        ("docs/ARCHITECTURE_AND_HISTORY.md", arch_text),
        ("lean/README.md", lean_readme),
    ):
        assert "no custom axioms" in doc_text.lower(), f"Missing 'no custom axioms' in {doc_name}"
        assert "0 axiom" not in doc_text.lower(), f"Stale '0 axiom' claim in {doc_name}"
        assert f"{n_thms}" in doc_text, f"Theorem count {n_thms} missing in {doc_name}"

    expected_hits = f"{full_loop['mean_verified_multi_ring_hits']:.2f}"
    expected_cov = f"{full_loop['conformal_coverage_pct']:.2f}%"
    expected_rmse = f"{full_loop['mean_frontier_rmse']:.4f}"
    for doc_text in (readme_text, arch_text):
        assert expected_hits in doc_text
        assert expected_cov in doc_text
        assert expected_rmse in doc_text
