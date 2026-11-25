"""Tests for 2026 open characterization/synthesis datasets, spectral parsers, and CLI."""

from __future__ import annotations

import importlib.util
import io
import json
import random
from pathlib import Path
from typing import Any

import pytest

from lab_attrib import (
    OPEN_CHARACTERIZATION_DATASETS,
    REFERENCE_MINERAL_PHASES,
    VALID_DATASET_MODALITIES,
    AssayTelemetry,
    AttributionOutcome,
    OpenCharacterizationDataset,
    OrthogonalModality,
    deconvolve_xy_diffractogram,
    diagnose_assay,
    evaluate_msms_orthogonal_readout,
    evaluate_raman_orthogonal_readout,
    get_open_dataset,
    get_reference_mineral_phase,
    list_open_datasets,
    load_alab_gpss_summary,
    load_aqsoldb_csv,
    load_massspecgym_spectra,
    load_raman_spectrum,
    load_xy_diffractogram,
    msms_cosine_similarity,
    render_reference_raman_spectrum,
    render_reference_xrd_spectrum,
    route_orthogonal_assay,
    run_attributed_frontier_benchmark,
    run_offline_reference_panel_benchmark,
)
from lab_attrib.cli import main as cli_main

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_fetch_script_module() -> Any:
    script_path = REPO_ROOT / "scripts" / "fetch_open_spectra.py"
    spec = importlib.util.spec_from_file_location("fetch_open_spectra", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_open_dataset_registry_schema_modalities_and_license_filters() -> None:
    assert len(OPEN_CHARACTERIZATION_DATASETS) >= 6
    modalities_covered = {ds.modality for ds in OPEN_CHARACTERIZATION_DATASETS}
    assert modalities_covered == set(VALID_DATASET_MODALITIES)

    seen_ids: set[str] = set()
    for ds in OPEN_CHARACTERIZATION_DATASETS:
        assert ds.dataset_id not in seen_ids
        seen_ids.add(ds.dataset_id)
        assert ds.url.startswith("https://")
        assert ds.spdx_license
        assert isinstance(ds.commercial_use_ok, bool)
        assert ds.n_records >= 100
        as_dict = ds.to_dict()
        assert as_dict["dataset_id"] == ds.dataset_id
        assert as_dict["modality"] == ds.modality

    # Canonical anchor datasets required by specification
    for expected_id in (
        "rruff_powder_xrd",
        "rruff_raman_zenodo",
        "cod_crystallography",
        "mp20_crystal_phases",
        "alab_gpss_solid_state",
        "massspecgym_msms",
        "nmrshiftdb2_nmr",
        "aqsoldb_solubility",
    ):
        assert get_open_dataset(expected_id).dataset_id == expected_id

    with pytest.raises(KeyError):
        get_open_dataset("nonexistent_dataset_xyz")

    # Commercial-use filter excludes OPL-1.0 NMRShiftDB2 while keeping CC0-1.0 BMRB
    all_nmr = list_open_datasets(modality="1H-NMR", commercial_only=False)
    comm_nmr = list_open_datasets(modality="1H-NMR", commercial_only=True)
    assert any(d.dataset_id == "nmrshiftdb2_nmr" for d in all_nmr)
    assert all(d.dataset_id != "nmrshiftdb2_nmr" for d in comm_nmr)
    assert any(d.dataset_id == "bmrb_metabolomics_nmr" for d in comm_nmr)

    with pytest.raises(ValueError, match="Unknown modality"):
        list_open_datasets(modality="FTIR")

    with pytest.raises(ValueError, match="https://"):
        OpenCharacterizationDataset(
            dataset_id="bad_http",
            modality="XRD",
            url="http://insecure.example.org/xrd",
            spdx_license="CC0-1.0",
            commercial_use_ok=True,
            n_records=10,
        )


def test_offline_reference_mineral_panel_deconvolution_and_orthogonal_routing() -> None:
    assert len(REFERENCE_MINERAL_PHASES) >= 10
    grid_xrd = [15.0 + 0.5 * i for i in range(90)]
    grid_ram = [100.0 + 5.0 * i for i in range(200)]

    # Centrosymmetric rocksalt LiCl has zero first-order Raman scattering
    licl_ram = render_reference_raman_spectrum("LiCl_rocksalt", grid_ram)
    assert all(v == 0.0 for v in licl_ram)
    assert get_reference_mineral_phase("LiCl_rocksalt").centrosymmetric is True

    calcite_xrd = render_reference_xrd_spectrum("Calcite_R050048", grid_xrd)
    assert max(calcite_xrd) > 0.8

    with pytest.raises(KeyError):
        get_reference_mineral_phase("Unknown_Phase")

    bench = run_offline_reference_panel_benchmark()
    spinel = bench["halide_spinel_ghost_trap"]
    assert spinel["naive_claims_hit"] is True
    assert spinel["precursor_overlap_permille"] > 150
    assert spinel["primary_outcome"] == "AMBIGUOUS_CHARACTERIZATION"
    assert spinel["centrosymmetric_selected_modality"] == "TEM"

    zro2 = bench["zirconia_raman_disambiguation"]
    assert zro2["xrd_naive_claims_hit"] is True
    assert zro2["xrd_precursor_overlap_permille"] > 150
    assert zro2["selected_modality"] == "Raman"
    assert zro2["post_raman_refutation_outcome"] == "CERTIFIED_NEGATIVE"
    assert zro2["post_raman_confirmation_outcome"] == "CERTIFIED_POSITIVE"

    bto = bench["barium_titanate_clean_hit"]
    assert bto["outcome"] == "CERTIFIED_POSITIVE"
    assert bto["lean_theorem_tag"] == "LabAttribLean.certified_positive_sound"
    assert bench["certified_ledger"]["chain_valid"] is True
    assert bench["certified_ledger"]["entries"] == 3


def test_xy_rruff_and_xrdml_diffractogram_parsers_feed_nnls_deconvolver(
    tmp_path: Path,
) -> None:
    # 1. RRUFF XY format with ##KEY=VALUE headers
    rruff_text = "\n".join(
        [
            "##NAMES=Calcite",
            "##RRUFFID=R050048",
            "##IDEAL CHEMISTRY=CaCO_3_",
            "20.0, 12.0",
            "25.0, 18.0",
            "29.4, 980.0",
            "36.0, 140.0",
            "43.2, 175.0",
            "##END=",
        ]
    )
    rruff_file = tmp_path / "calcite_rruff.txt"
    rruff_file.write_text(rruff_text, encoding="utf-8")
    cal_rec = load_xy_diffractogram(rruff_file, min_two_theta=22.0, max_two_theta=45.0)
    assert cal_rec.sample_id == "R050048"
    assert cal_rec.metadata["NAMES"] == "Calcite"
    assert cal_rec.two_theta_deg == (25.0, 29.4, 36.0, 43.2)

    # 2. PANalytical XRDML XML format (A-Lab GPSS schema)
    common_grid = [20.0 + 0.5 * i for i in range(61)]  # 20..50 deg
    prec_y = render_reference_xrd_spectrum("TiO2_anatase", common_grid, fwhm_deg=0.7)
    targ_y = render_reference_xrd_spectrum("BaTiO3_tetragonal", common_grid, fwhm_deg=0.7)
    mix_y = [0.05 * prec_y[i] + 1.20 * targ_y[i] for i in range(len(common_grid))]
    counts_str = " ".join(f"{v:.5f}" for v in mix_y)

    xrdml_text = f"""<?xml version="1.0"?>
<xrdMeasurements xmlns="http://www.xrdml.com/XRDMeasurement/2.3">
  <sample><id>GPSS_BaTiO3_Test</id></sample>
  <xrdMeasurement>
    <usedWavelength><kAlpha1>1.5405980</kAlpha1></usedWavelength>
    <scan>
      <dataPoints>
        <positions axis="2Theta" unit="deg">
          <startPosition>20.0</startPosition>
          <endPosition>50.0</endPosition>
        </positions>
        <intensities unit="counts">{counts_str}</intensities>
      </dataPoints>
    </scan>
  </xrdMeasurement>
</xrdMeasurements>
"""
    xrdml_path = tmp_path / "pattern.xrdml"
    xrdml_path.write_text(xrdml_text, encoding="utf-8")
    obs_rec = load_xy_diffractogram(xrdml_path)
    assert obs_rec.sample_id == "GPSS_BaTiO3_Test"
    assert len(obs_rec.two_theta_deg) == 61

    # Precursor on a coarser 0.25-deg grid resampled automatically via deconvolve_xy_diffractogram
    fine_grid = [19.0 + 0.25 * i for i in range(130)]
    prec_fine = render_reference_xrd_spectrum("TiO2_anatase", fine_grid, fwhm_deg=0.7)
    targ_fine = render_reference_xrd_spectrum("BaTiO3_tetragonal", fine_grid, fwhm_deg=0.7)
    prec_rec = load_xy_diffractogram(
        "\n".join(f"{fine_grid[i]} {prec_fine[i]} 0.01" for i in range(len(fine_grid)))
    )
    targ_rec = load_xy_diffractogram(
        "\n".join(f"{fine_grid[i]},{targ_fine[i]}" for i in range(len(fine_grid)))
    )
    assert prec_rec.uncertainties is not None

    deconv = deconvolve_xy_diffractogram(
        obs_rec,
        [prec_rec],
        targ_rec,
        min_hit_amplitude=0.25,
        noise_sigma=0.03,
    )
    report = diagnose_assay(deconv.to_assay_telemetry())
    assert report.outcome is AttributionOutcome.CERTIFIED_POSITIVE


def test_alab_gpss_summary_parser_routes_spinel_runs() -> None:
    alab_json = json.dumps(
        [
            {
                "sample_index": 0,
                "composition": "Li2MgCl4",
                "xrd_analysis_result": {
                    "weight_fractions_of_each_phases": {
                        "spinel (Fd-3m)": 0.9616,
                        "LiCl (Fm-3m)": 0.0384,
                    },
                    "Rwp (%)": "8.89%",
                },
            },
            {
                "sample_index": 12,
                "composition": "Li1.6Mn0.8Zr0.2Cl4",
                "xrd_analysis_result": {
                    "weight_fractions_of_each_phases": {
                        "spinel (Fd-3m)": 0.62,
                        "LiCl (Fm-3m)": 0.38,
                    },
                    "Rwp (%)": "11.40%",
                },
            },
        ]
    )
    parsed = load_alab_gpss_summary(alab_json)
    assert len(parsed) == 2
    assert parsed[0]["attribution"]["outcome"] == "CERTIFIED_POSITIVE"
    assert parsed[0]["orthogonal_routing"] is None
    assert parsed[1]["attribution"]["outcome"] == "AMBIGUOUS_CHARACTERIZATION"
    assert parsed[1]["orthogonal_routing"] is not None


def test_jcamp_dx_and_csv_raman_parsers_resolve_ambiguous_xrd() -> None:
    # Build compressed JCAMP-DX ##XYDATA=(X++(Y..Y)) spectrum with XFACTOR & YFACTOR
    wn_grid = [120.0 + 4.0 * i for i in range(110)]  # 120..556 cm^-1
    quartz_y = render_reference_raman_spectrum("Quartz_alpha_COD9009666", wn_grid)
    calcite_y = render_reference_raman_spectrum("Calcite_R050048", wn_grid)

    jdx_rows: list[str] = [
        "##TITLE=Quartz_Sample_JCAMP",
        "##JCAMP-DX=4.24",
        "##XFACTOR=1.0",
        "##YFACTOR=0.001",
        "##DELTAX=4.0",
        "##XYDATA=(X++(Y..Y))",
    ]
    for start in range(0, len(wn_grid), 5):
        chunk_y = " ".join(str(int(round(1000.0 * v))) for v in quartz_y[start : start + 5])
        jdx_rows.append(f"{wn_grid[start]:.1f} {chunk_y}")
    jdx_rows.append("##END=")

    obs_raman = load_raman_spectrum("\n".join(jdx_rows))
    assert obs_raman.sample_id == "Quartz_Sample_JCAMP"
    assert len(obs_raman.wavenumber_cm1) == len(wn_grid)

    csv_quartz = "wavenumber_cm1,intensity\n" + "\n".join(
        f"{wn_grid[i]},{quartz_y[i]:.6f}" for i in range(len(wn_grid))
    )
    csv_calcite = "wavenumber_cm1,intensity\n" + "\n".join(
        f"{wn_grid[i]},{calcite_y[i]:.6f}" for i in range(len(wn_grid))
    )
    ref_quartz = load_raman_spectrum(csv_quartz)
    ref_calcite = load_raman_spectrum(csv_calcite)

    _, raman_readout = evaluate_raman_orthogonal_readout(
        obs_raman,
        [ref_calcite],
        ref_quartz,
    )
    assert raman_readout.modality is OrthogonalModality.RAMAN
    assert raman_readout.target_confirmed is True

    ambiguous_primary = AssayTelemetry(
        dispense_error_permille=12,
        aspiration_clean=True,
        incubation_window_clean=True,
        seal_intact=True,
        positive_control_passed=True,
        blank_uncontaminated=True,
        detector_calibrated=True,
        precursor_overlap_permille=520,
        spectral_residual_permille=45,
        snr_tenths=75,
        target_signal_present=True,
    )
    resolved = diagnose_assay(ambiguous_primary, raman_readout)
    assert resolved.outcome is AttributionOutcome.CERTIFIED_POSITIVE
    assert resolved.resolved_via_orthogonal is True


def test_massspecgym_mgf_tsv_json_parsers_and_lcms_orthogonal_resolution() -> None:
    mgf_text = """BEGIN IONS
IDENTIFIER=MassSpecGymID0001
PRECURSOR_MZ=195.0877
SMILES=Cn1cnc2c1c(=O)n(C)c(=O)n2C
INCHIKEY=RYYVLZVUVIJVGH-UHFFFAOYSA-N
ADDUCT=[M+H]+
INSTRUMENT_TYPE=Orbitrap
FOLD=val
110.0712 240.0
138.0662 1000.0
195.0877 620.0
END IONS
"""
    tsv_text = (
        "identifier\tmzs\tintensities\tsmiles\tinchikey\tprecursor_mz\tadduct\tfold\n"
        "MassSpecGymID0001_obs\t110.0720,138.0665,195.0880\t0.25,1.0,0.60\t"
        "Cn1cnc2c1c(=O)n(C)c(=O)n2C\tRYYVLZVUVIJVGH-UHFFFAOYSA-N\t195.0880\t[M+H]+\tval\n"
    )
    json_text = json.dumps(
        [
            {
                "identifier": "MassSpecGymID9999_decoy",
                "mzs": [85.028, 121.065, 163.075],
                "intensities": [0.50, 1.00, 0.40],
                "precursor_mz": 195.0879,
                "smiles": "COc1ccc(CC(=O)O)cc1",
                "adduct": "[M+H]+",
            }
        ]
    )

    mgf_recs = load_massspecgym_spectra(mgf_text)
    tsv_recs = load_massspecgym_spectra(tsv_text)
    json_recs = load_massspecgym_spectra(json_text)
    assert len(mgf_recs) == 1 and len(tsv_recs) == 1 and len(json_recs) == 1

    cos_match = msms_cosine_similarity(tsv_recs[0], mgf_recs[0], mz_tolerance_da=0.02)
    cos_decoy = msms_cosine_similarity(json_recs[0], mgf_recs[0], mz_tolerance_da=0.02)
    assert cos_match > 0.99
    assert cos_decoy == 0.0

    ambiguous_solution_tel = AssayTelemetry(
        dispense_error_permille=10,
        aspiration_clean=True,
        incubation_window_clean=True,
        seal_intact=True,
        positive_control_passed=True,
        blank_uncontaminated=True,
        detector_calibrated=True,
        precursor_overlap_permille=480,
        spectral_residual_permille=145,
        snr_tenths=60,
        target_signal_present=True,
    )
    routing = route_orthogonal_assay(
        ambiguous_solution_tel,
        prior_hit_prob=0.30,
        available_volume_ul=8.0,
        sample_phase="solution",
        analyte_concentration_um=1.0,
    )
    assert routing.selected_modality is OrthogonalModality.LC_MS

    confirm_readout = evaluate_msms_orthogonal_readout(tsv_recs[0], mgf_recs[0])
    refute_readout = evaluate_msms_orthogonal_readout(json_recs[0], mgf_recs[0])
    assert diagnose_assay(ambiguous_solution_tel, confirm_readout).outcome is AttributionOutcome.CERTIFIED_POSITIVE
    assert diagnose_assay(ambiguous_solution_tel, refute_readout).outcome is AttributionOutcome.CERTIFIED_NEGATIVE

    # Isobaric co-eluting chimera sets selectivity_clean=False -> stays AMBIGUOUS_CHARACTERIZATION
    chimera_rec = load_massspecgym_spectra(
        json.dumps(
            [
                {
                    "identifier": "chimera",
                    "mzs": [85.028, 121.065, 138.0662, 163.075, 195.0877],
                    "intensities": [0.55, 1.00, 0.90, 0.45, 0.55],
                    "precursor_mz": 195.0878,
                }
            ]
        )
    )[0]
    chimera_readout = evaluate_msms_orthogonal_readout(
        chimera_rec,
        mgf_recs[0],
        interferent_reference=json_recs[0],
    )
    assert chimera_readout.selectivity_clean is False
    assert (
        diagnose_assay(ambiguous_solution_tel, chimera_readout).outcome
        is AttributionOutcome.AMBIGUOUS_CHARACTERIZATION
    )


def test_aqsoldb_csv_and_tsv_parsers_feed_attributed_frontier_campaign(
    tmp_path: Path,
) -> None:
    rng = random.Random(42)
    tab_rows = [
        "ID\tName\tSMILES\tSolubility\tMolWt\tNumHDonors\tNumRotatableBonds\tRingCount\tTPSA"
    ]
    for i in range(220):
        rings = 1 if i < 120 else rng.choice([3, 4])
        mw = round(140.0 + 35.0 * rings + rng.uniform(-15.0, 25.0), 2)
        hbd = rng.choice([0, 1, 2])
        rot = rng.choice([1, 2, 3])
        tpsa = round(22.0 + 18.0 * hbd + rng.uniform(0.0, 20.0), 2)
        logs = round(-1.2 - 0.55 * rings + 0.25 * hbd + rng.gauss(0.0, 0.55), 4)
        smi = "c1ccccc1O" if rings == 1 else "c1ccc2cc3ccccc3cc2c1O"
        tab_rows.append(f"A-{i:04d}\tComp_{i}\t{smi}\t{logs}\t{mw}\t{hbd}\t{rot}\t{rings}\t{tpsa}")

    aqsoldb_tab = tmp_path / "curated-solubility-dataset.tab"
    aqsoldb_tab.write_text("\n".join(tab_rows) + "\n", encoding="utf-8")

    mols = load_aqsoldb_csv(aqsoldb_tab)
    assert len(mols) == 220
    assert any(m.is_true_frontier_hit for m in mols)

    # Also verify HF comma-separated Y column schema
    hf_csv = (
        "ID,SMILES,Y,MolWt,NumHDonors,NumRotatableBonds,RingCount,TPSA\n"
        "F-1054,COc1ccc2cc(C(C)C(=O)OCCC)ccc2c1,-3.12,272.34,0,5,2,35.53\n"
    )
    hf_mols = load_aqsoldb_csv(hf_csv)
    assert len(hf_mols) == 1
    assert hf_mols[0].rings == 2

    # Run full 4-regime attributed frontier campaign directly on AqSolDB TSV
    campaign = run_attributed_frontier_benchmark(
        csv_path=aqsoldb_tab,
        seeds=(11, 23),
        n_rounds=4,
        batch_size=14,
    )
    full = campaign["regimes"]["lab_attrib_full_closed_loop"]
    assert full["false_discovery_rate_pct"] == 0.0
    assert full["mean_poisoned_memory_entries"] == 0.0
    assert full["all_ledger_chains_valid"] is True


def test_cli_datasets_and_fetch_open_spectra_script(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rc = cli_main(["datasets", "--modality", "XRD", "--commercial-only", "--json"])
    assert rc == 0
    out_xrd = json.loads(capsys.readouterr().out)
    assert out_xrd["n_datasets"] >= 4
    assert all(d["modality"] == "XRD" and d["commercial_use_ok"] for d in out_xrd["datasets"])

    rc_panel = cli_main(["datasets", "--benchmark-panel"])
    assert rc_panel == 0
    panel_text = capsys.readouterr().out
    assert "Bundled Reference Mineral/Oxide Panel Benchmark" in panel_text

    # Test scripts/fetch_open_spectra.py offline panel export + mocked HTTP download
    fetch_mod = _load_fetch_script_module()
    rc_export = fetch_mod.main(["--offline-panel", "--output-dir", str(tmp_path), "--json"])
    assert rc_export == 0
    export_json = json.loads(capsys.readouterr().out)
    assert export_json["exported_file_count"] >= 20

    exported_xy = load_xy_diffractogram(tmp_path / "reference_panel" / "Calcite_R050048_xrd.xy")
    exported_jdx = load_raman_spectrum(tmp_path / "reference_panel" / "Calcite_R050048_raman.jdx")
    assert len(exported_xy.two_theta_deg) == 241
    assert len(exported_jdx.wavenumber_cm1) == 526

    class _FakeHttpResponse(io.BytesIO):
        status = 200

        def __enter__(self) -> _FakeHttpResponse:
            return self

        def __exit__(self, *args: object) -> None:
            self.close()

    def _fake_urlopen(req: object, timeout: float = 10.0) -> _FakeHttpResponse:
        return _FakeHttpResponse(b"##NAMES=Calcite\n29.40, 1000.0\n39.41, 180.0\n")

    monkeypatch.setattr(fetch_mod.urllib.request, "urlopen", _fake_urlopen)
    rc_fetch = fetch_mod.main(
        ["--dataset", "rruff_powder_xrd", "--output-dir", str(tmp_path), "--json"]
    )
    assert rc_fetch == 0
    fetch_payload = json.loads(capsys.readouterr().out)
    assert fetch_payload["fetched"][0]["dataset_id"] == "rruff_powder_xrd"
    assert (tmp_path / "rruff_powder_xrd" / "manifest.json").is_file()

    rc_verify = fetch_mod.main(["--verify-urls", "--modality", "XRD", "--json"])
    assert rc_verify == 0
    verify_payload = json.loads(capsys.readouterr().out)
    assert verify_payload["all_urls_reachable"] is True

    # Ensure data/ is gitignored and no user-specific strings appear in scripts/
    gitignore_text = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "data/" in gitignore_text
    script_src = (REPO_ROOT / "scripts" / "fetch_open_spectra.py").read_text(encoding="utf-8")
    assert "c71e01d3" not in script_src
