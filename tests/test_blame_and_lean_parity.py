"""Unit, boundary, and runtime Lean 4 `--batch-tsv` parity tests for `blame.py`."""

from __future__ import annotations

import itertools
import os
import re
import subprocess
from pathlib import Path

import pytest

from lab_attrib.blame import (
    MAX_DISPENSE_ERROR_PERMILLE,
    MAX_PRECURSOR_OVERLAP_PERMILLE,
    MAX_SPECTRAL_RESIDUAL_PERMILLE,
    MAX_STEPPING_STONE_VOLUME_UL,
    MAX_STICK_NNLS_RESIDUAL_PERMILLE,
    MIN_POS_CONTROL_RECOVERY_PERMILLE,
    MIN_POS_CONTROL_ZSCORE_TENTHS,
    MIN_RESOLVABLE_SNR_TENTHS,
    MIN_Z_PRIME_TENTHS,
    AssayTelemetry,
    AttributedSteppingStone,
    AttributionOutcome,
    DiagnosticAction,
    OrthogonalAssayReadout,
    OrthogonalModality,
    SpectralFitMethod,
    admit_to_scientific_memory,
    attribute_assay_outcome,
    characterization_identifiable,
    check_attributed_frontier_update,
    diagnose_assay,
    effective_attribution,
    execution_clean,
    max_residual_for_fit_method,
    orthogonal_readout_clean,
    resolve_with_orthogonal_assay,
    route_diagnostic_action,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CJK_PATTERN = re.compile(
    r"[\u2e80-\u2eff\u3000-\u303f\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef]"
)
CHECKED_SUFFIXES = {
    ".py",
    ".lean",
    ".md",
    ".toml",
    ".yml",
    ".yaml",
    ".json",
    ".tex",
    ".bib",
    ".sty",
}


def _clean_telemetry(**overrides: int | bool | str | SpectralFitMethod) -> AssayTelemetry:
    base: dict[str, int | bool | str | SpectralFitMethod] = {
        "dispense_error_permille": 12,
        "aspiration_clean": True,
        "incubation_window_clean": True,
        "seal_intact": True,
        "positive_control_passed": True,
        "blank_uncontaminated": True,
        "detector_calibrated": True,
        "precursor_overlap_permille": 20,
        "spectral_residual_permille": 35,
        "snr_tenths": 65,
        "target_signal_present": False,
        "positive_control_recovery_permille": 1000,
        "plate_z_prime_tenths": 70,
        "positive_control_zscore_tenths": 50,
        "fit_method": SpectralFitMethod.RIETVELD,
    }
    base.update(overrides)
    return AssayTelemetry(**base)  # type: ignore[arg-type]


def _find_lean_binary() -> Path | None:
    bin_dir = REPO_ROOT / "lean" / ".lake" / "build" / "bin"
    for candidate in (bin_dir / "lab-attrib-lean.exe", bin_dir / "lab-attrib-lean"):
        if candidate.is_file():
            return candidate
    return None


def test_lean_guard_examples_exact_parity() -> None:
    clean_neg = _clean_telemetry()
    clean_pos = _clean_telemetry(
        dispense_error_permille=15,
        precursor_overlap_permille=45,
        spectral_residual_permille=42,
        snr_tenths=110,
        target_signal_present=True,
    )
    cavitation = _clean_telemetry(
        dispense_error_permille=180,
        aspiration_clean=False,
        precursor_overlap_permille=10,
        spectral_residual_permille=25,
        snr_tenths=50,
        target_signal_present=False,
    )
    alab_ghost = _clean_telemetry(
        dispense_error_permille=18,
        precursor_overlap_permille=780,
        spectral_residual_permille=60,
        snr_tenths=85,
        target_signal_present=True,
    )

    assert attribute_assay_outcome(clean_neg) is AttributionOutcome.CERTIFIED_NEGATIVE
    assert attribute_assay_outcome(clean_pos) is AttributionOutcome.CERTIFIED_POSITIVE
    assert attribute_assay_outcome(cavitation) is AttributionOutcome.EXECUTION_ARTIFACT
    assert (
        route_diagnostic_action(attribute_assay_outcome(cavitation))
        is DiagnosticAction.RETRY_ROBOT
    )
    assert admit_to_scientific_memory(attribute_assay_outcome(cavitation)) is False

    assert attribute_assay_outcome(alab_ghost) is AttributionOutcome.AMBIGUOUS_CHARACTERIZATION
    assert (
        route_diagnostic_action(attribute_assay_outcome(alab_ghost))
        is DiagnosticAction.RUN_ORTHOGONAL_ASSAY
    )
    assert admit_to_scientific_memory(attribute_assay_outcome(alab_ghost)) is False


def test_exhaustive_four_way_partition_across_full_11_clause_grid() -> None:
    # Full 2^11 = 2,048 combinations covering ALL execution & characterization clauses
    count = 0
    for (
        disp,
        asp,
        inc,
        seal,
        pos_ctrl,
        blank_ok,
        det,
        prec,
        rwp,
        snr,
        sig,
    ) in itertools.product(
        (MAX_DISPENSE_ERROR_PERMILLE, MAX_DISPENSE_ERROR_PERMILLE + 1),
        (True, False),
        (True, False),
        (True, False),
        (True, False),
        (True, False),
        (True, False),
        (MAX_PRECURSOR_OVERLAP_PERMILLE, MAX_PRECURSOR_OVERLAP_PERMILLE + 1),
        (MAX_SPECTRAL_RESIDUAL_PERMILLE, MAX_SPECTRAL_RESIDUAL_PERMILLE + 1),
        (MIN_RESOLVABLE_SNR_TENTHS - 1, MIN_RESOLVABLE_SNR_TENTHS),
        (True, False),
    ):
        count += 1
        t = _clean_telemetry(
            dispense_error_permille=disp,
            aspiration_clean=asp,
            incubation_window_clean=inc,
            seal_intact=seal,
            positive_control_passed=pos_ctrl,
            blank_uncontaminated=blank_ok,
            detector_calibrated=det,
            precursor_overlap_permille=prec,
            spectral_residual_permille=rwp,
            snr_tenths=snr,
            target_signal_present=sig,
        )
        out = attribute_assay_outcome(t)
        exec_ok = execution_clean(t)
        ident_ok = characterization_identifiable(t)

        c1 = out is AttributionOutcome.EXECUTION_ARTIFACT and not exec_ok
        c2 = out is AttributionOutcome.AMBIGUOUS_CHARACTERIZATION and exec_ok and not ident_ok
        c3 = out is AttributionOutcome.CERTIFIED_NEGATIVE and exec_ok and ident_ok and not sig
        c4 = out is AttributionOutcome.CERTIFIED_POSITIVE and exec_ok and ident_ok and sig
        assert sum([c1, c2, c3, c4]) == 1
        assert admit_to_scientific_memory(out) == (exec_ok and ident_ok)
    assert count == 2048


# --- Issue #4 & #5: Dual-fault diagnostics, quantitative control drift, and fault scope ---


def test_issue4_compound_execution_and_characterization_fault_recalibration_ordering() -> None:
    tel = AssayTelemetry(
        dispense_error_permille=65,
        aspiration_clean=True,
        incubation_window_clean=True,
        seal_intact=True,
        positive_control_passed=True,
        blank_uncontaminated=True,
        detector_calibrated=False,
        precursor_overlap_permille=720,
        spectral_residual_permille=45,
        snr_tenths=80,
        target_signal_present=True,
    )
    rep = diagnose_assay(tel)
    assert rep.outcome is AttributionOutcome.EXECUTION_ARTIFACT
    assert rep.action is DiagnosticAction.RETRY_ROBOT
    assert rep.admit_to_memory is False
    assert rep.dual_fault_detected is True
    assert rep.compound_fault is True
    assert rep.fault_scope == "WELL_MECHANICAL"
    assert "DISPENSE_ERROR_EXCEEDED" in rep.violated_execution_clauses
    assert "DETECTOR_UNCALIBRATED_OR_SATURATED" in rep.violated_characterization_clauses
    assert "PRECURSOR_GHOST_PEAK_OVERLAP" in rep.violated_characterization_clauses
    assert "RECALIBRATE_AND_RETRY" in rep.diagnostic_notes
    assert rep.remediation_steps.index("RECALIBRATE_DETECTOR") < rep.remediation_steps.index(
        "RETRY_ROBOT"
    )
    assert "PRE_STAGE_ORTHOGONAL_ASSAY" in rep.remediation_steps


def test_issue5_quantitative_control_drift_and_plate_reagent_contamination_scope() -> None:
    # Sub-threshold positive-control recovery (< 800), low Z' (< 50), low z-score (< 20)
    rec_drift = _clean_telemetry(
        positive_control_recovery_permille=MIN_POS_CONTROL_RECOVERY_PERMILLE - 1,
        target_signal_present=False,
    )
    zprime_drift = _clean_telemetry(
        plate_z_prime_tenths=MIN_Z_PRIME_TENTHS - 1,
        target_signal_present=True,
    )
    zscore_drift = _clean_telemetry(
        positive_control_zscore_tenths=MIN_POS_CONTROL_ZSCORE_TENTHS - 1,
        target_signal_present=False,
    )
    for t_drift, expected_clause in (
        (rec_drift, "POSITIVE_CONTROL_RECOVERY_DRIFT"),
        (zprime_drift, "LOW_PLATE_Z_PRIME_FACTOR"),
        (zscore_drift, "POSITIVE_CONTROL_ZSCORE_DRIFT"),
    ):
        rep = diagnose_assay(t_drift)
        assert rep.outcome is AttributionOutcome.EXECUTION_ARTIFACT
        assert rep.admit_to_memory is False
        assert expected_clause in rep.violated_execution_clauses
        assert rep.fault_scope == "PLATE_REAGENT_CONTAMINATION"
        assert rep.remediation_steps[0] == "REFRESH_PLATE_CONTROLS_AND_BLANKS"

    # Boolean positive-control / blank failure also sets PLATE_REAGENT_CONTAMINATION
    batch_ctrl_fail = diagnose_assay(
        AssayTelemetry(10, True, True, True, False, False, True, 20, 30, 80, False)
    )
    assert batch_ctrl_fail.fault_scope == "PLATE_REAGENT_CONTAMINATION"
    assert "REFRESH_PLATE_CONTROLS_AND_BLANKS" in batch_ctrl_fail.remediation_steps


# --- Issue #8: Runtime differential parity against compiled `lab-attrib-lean --batch-tsv` ---


def test_issue8_compiled_lean_batch_tsv_differential_parity(tmp_path: Path) -> None:
    lean_bin = _find_lean_binary()
    require_bin = os.environ.get("LAB_ATTRIB_REQUIRE_LEAN_BIN") == "1"
    if lean_bin is None:
        if require_bin:
            pytest.fail(
                "LAB_ATTRIB_REQUIRE_LEAN_BIN=1 set, but compiled lab-attrib-lean binary was not found"
            )
        pytest.skip("Compiled lab-attrib-lean binary not present (clean tree without .lake)")

    # Construct >= 2,300 boundary vectors covering all 14 telemetry lanes, orthogonal resolution,
    # and stepping-stone shell/volume/containment boundaries.
    vectors: list[tuple[str, AssayTelemetry, OrthogonalAssayReadout | None, AttributedSteppingStone]] = []
    idx = 0
    for (
        disp,
        asp,
        inc,
        seal,
        pos_ctrl,
        blank_ok,
        det,
        prec,
        rwp,
        snr,
        sig,
    ) in itertools.product(
        (MAX_DISPENSE_ERROR_PERMILLE, MAX_DISPENSE_ERROR_PERMILLE + 1),
        (True, False),
        (True, False),
        (True, False),
        (True, False),
        (True, False),
        (True, False),
        (MAX_PRECURSOR_OVERLAP_PERMILLE, MAX_PRECURSOR_OVERLAP_PERMILLE + 1),
        (MAX_SPECTRAL_RESIDUAL_PERMILLE, MAX_SPECTRAL_RESIDUAL_PERMILLE + 1),
        (MIN_RESOLVABLE_SNR_TENTHS - 1, MIN_RESOLVABLE_SNR_TENTHS),
        (True, False),
    ):
        t = _clean_telemetry(
            dispense_error_permille=disp,
            aspiration_clean=asp,
            incubation_window_clean=inc,
            seal_intact=seal,
            positive_control_passed=pos_ctrl,
            blank_uncontaminated=blank_ok,
            detector_calibrated=det,
            precursor_overlap_permille=prec,
            spectral_residual_permille=rwp,
            snr_tenths=snr,
            target_signal_present=sig,
        )
        orth: OrthogonalAssayReadout | None = None
        if idx % 4 == 1:
            orth = OrthogonalAssayReadout(
                modality=OrthogonalModality.RAMAN,
                instrument_calibrated=True,
                orthogonal_snr_tenths=75,
                selectivity_clean=True,
                target_confirmed=(idx % 8 == 1),
            )
        elif idx % 4 == 2:
            orth = OrthogonalAssayReadout(
                modality=OrthogonalModality.LC_MS,
                instrument_calibrated=(idx % 8 == 2),
                orthogonal_snr_tenths=25 if (idx % 16 == 2) else 90,
                selectivity_clean=True,
                target_confirmed=True,
            )

        tau = 300
        eta = 250 if (idx % 5 != 0) else 550
        dist = 350 if (idx % 3 != 0) else 390
        prop_tau = dist if (idx % 7 != 0) else 300
        vol = 8 if (idx % 6 != 0) else 12
        probe = AttributedSteppingStone(
            tau_permille=tau,
            eta_permille=eta,
            dist_permille=dist,
            proposed_new_tau_permille=prop_tau,
            volume_ul=vol,
            uses_only_stock_reagents=(idx % 11 != 0),
            within_hazard_envelope=(idx % 13 != 0),
            telemetry=t,
            orthogonal_readout=orth,
        )
        vectors.append((f"ROW-{idx:04d}", t, orth, probe))
        idx += 1

    # Add quantitative control-drift boundary vectors (including negative Z'-factor / z-score)
    for rec_p, zp_t, zs_t in itertools.product(
        (799, 800, 1000),
        (-15, 49, 50, 70),
        (-10, 19, 20, 50),
    ):
        t_q = _clean_telemetry(
            positive_control_recovery_permille=rec_p,
            plate_z_prime_tenths=zp_t,
            positive_control_zscore_tenths=zs_t,
            target_signal_present=(rec_p == 1000),
        )
        probe_q = AttributedSteppingStone(
            tau_permille=300,
            eta_permille=250,
            dist_permille=350,
            proposed_new_tau_permille=350,
            volume_ul=8,
            uses_only_stock_reagents=True,
            within_hazard_envelope=True,
            telemetry=t_q,
        )
        vectors.append((f"QCTRL-{idx:04d}", t_q, None, probe_q))
        idx += 1

    # Add stepping-stone shell / micro-volume / containment boundary grid (192 vectors)
    for tau_v, eta_v, dist_v, prop_v, vol_v, stock_ok, haz_ok in itertools.product(
        (0, 300),
        (250, 500, 501),
        (300, 375, 376),
        (300, 375),
        (0, 10, 11),
        (True, False),
        (True,),
    ):
        t_step = _clean_telemetry(target_signal_present=(vol_v == 10))
        probe_s = AttributedSteppingStone(
            tau_permille=tau_v,
            eta_permille=eta_v,
            dist_permille=dist_v,
            proposed_new_tau_permille=prop_v,
            volume_ul=vol_v,
            uses_only_stock_reagents=stock_ok,
            within_hazard_envelope=haz_ok,
            telemetry=t_step,
        )
        vectors.append((f"STEP-{idx:04d}", t_step, None, probe_s))
        idx += 1

    # Add Issue #17 fit_method ("rietveld" vs "stick_nnls") boundary vectors (100, 101, 200, 201)
    for fit_m in (SpectralFitMethod.RIETVELD, SpectralFitMethod.STICK_NNLS):
        for rwp_v in (
            MAX_SPECTRAL_RESIDUAL_PERMILLE,
            MAX_SPECTRAL_RESIDUAL_PERMILLE + 1,
            MAX_STICK_NNLS_RESIDUAL_PERMILLE,
            MAX_STICK_NNLS_RESIDUAL_PERMILLE + 1,
        ):
            for sig_v in (True, False):
                t_fit = _clean_telemetry(
                    spectral_residual_permille=rwp_v,
                    target_signal_present=sig_v,
                    fit_method=fit_m,
                )
                probe_fit = AttributedSteppingStone(300, 250, 350, 350, 8, True, True, t_fit)
                vectors.append((f"FIT-{idx:04d}", t_fit, None, probe_fit))
                idx += 1

    # Add explicit Issue #8 telemetry-tampering & same-length sampleId collision vectors
    t_neg_base = _clean_telemetry(dispense_error_permille=12, snr_tenths=65, target_signal_present=False)
    t_neg_mut = _clean_telemetry(dispense_error_permille=50, snr_tenths=30, target_signal_present=False)
    base_probe = AttributedSteppingStone(300, 250, 350, 350, 8, True, True, t_neg_base)
    mut_probe = AttributedSteppingStone(300, 250, 350, 350, 8, True, True, t_neg_mut)
    vectors.append(("NEG-1", t_neg_base, None, base_probe))
    vectors.append(("FORGE", t_neg_base, None, base_probe))
    vectors.append(("NEG-1", t_neg_mut, None, mut_probe))

    assert len(vectors) >= 2300

    tsv_lines: list[str] = []
    for row_id, t, orth, p in vectors:
        has_orth = "1" if orth is not None else "0"
        orth_mod = orth.modality.value if orth is not None else "NONE"
        orth_cal = ("1" if orth.instrument_calibrated else "0") if orth is not None else "0"
        orth_snr = str(orth.orthogonal_snr_tenths) if orth is not None else "0"
        orth_sel = ("1" if orth.selectivity_clean else "0") if orth is not None else "0"
        orth_conf = ("1" if orth.target_confirmed else "0") if orth is not None else "0"
        cols = [
            row_id,
            str(t.dispense_error_permille),
            "1" if t.aspiration_clean else "0",
            "1" if t.incubation_window_clean else "0",
            "1" if t.seal_intact else "0",
            "1" if t.positive_control_passed else "0",
            "1" if t.blank_uncontaminated else "0",
            "1" if t.detector_calibrated else "0",
            str(t.precursor_overlap_permille),
            str(t.spectral_residual_permille),
            str(t.snr_tenths),
            "1" if t.target_signal_present else "0",
            str(t.positive_control_recovery_permille),
            str(t.plate_z_prime_tenths),
            str(t.positive_control_zscore_tenths),
            has_orth,
            orth_mod,
            orth_cal,
            orth_snr,
            orth_sel,
            orth_conf,
            str(p.tau_permille),
            str(p.eta_permille),
            str(p.dist_permille),
            str(p.proposed_new_tau_permille),
            str(p.volume_ul),
            "1" if p.uses_only_stock_reagents else "0",
            "1" if p.within_hazard_envelope else "0",
            t.fit_method.value,
        ]
        tsv_lines.append("\t".join(cols))

    tsv_path = tmp_path / "lean_parity_batch.tsv"
    tsv_path.write_text("\n".join(tsv_lines) + "\n", encoding="utf-8")

    proc = subprocess.run(
        [str(lean_bin), "--batch-tsv", str(tsv_path)],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    out_lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    assert len(out_lines) == len(vectors)

    hashes_tail: list[int] = []
    for (row_id, t, orth, p), line in zip(vectors, out_lines):
        parts = line.split("\t")
        assert len(parts) == 12
        (
            l_id,
            l_exec,
            l_ident,
            l_dual,
            l_pout,
            l_pact,
            l_padm,
            l_eout,
            l_eact,
            l_eadm,
            l_front,
            l_hash,
        ) = parts

        py_exec = execution_clean(t)
        py_ident = characterization_identifiable(t)
        py_dual = (not py_exec) and (not py_ident)
        py_pout = attribute_assay_outcome(t)
        py_pact = route_diagnostic_action(py_pout)
        py_padm = admit_to_scientific_memory(py_pout)
        py_eout = effective_attribution(t, orth)
        py_eact = route_diagnostic_action(py_eout)
        py_eadm = admit_to_scientific_memory(py_eout)
        py_front = check_attributed_frontier_update(p)

        assert l_id == row_id
        assert (l_exec == "true") is py_exec
        assert (l_ident == "true") is py_ident
        assert (l_dual == "true") is py_dual
        assert l_pout == py_pout.value
        assert l_pact == py_pact.value
        assert (l_padm == "true") is py_padm
        assert l_eout == py_eout.value
        assert l_eact == py_eact.value
        assert (l_eadm == "true") is py_eadm
        assert (l_front == "true") is py_front
        hashes_tail.append(int(l_hash))

    # Last 3 rows are ("NEG-1", base), ("FORGE", base), ("NEG-1", mut): all 3 hashes MUST differ!
    h_neg1_base, h_forge_base, h_neg1_mut = hashes_tail[-3:]
    assert len({h_neg1_base, h_forge_base, h_neg1_mut}) == 3


@pytest.mark.parametrize(
    ("field", "bad_val", "expected_code"),
    [
        ("dispense_error_permille", 51, "DISPENSE_ERROR_EXCEEDED"),
        ("aspiration_clean", False, "ASPIRATION_CAVITATION_OR_CLOG"),
        ("incubation_window_clean", False, "INCUBATION_WINDOW_VIOLATION"),
        ("seal_intact", False, "SEAL_INTEGRITY_BREACH"),
        ("positive_control_passed", False, "POSITIVE_CONTROL_FAILED"),
        ("blank_uncontaminated", False, "BLANK_CONTAMINATED"),
    ],
)
def test_each_hardware_fault_blocks_memory_and_routes_to_retry(
    field: str,
    bad_val: int | bool,
    expected_code: str,
) -> None:
    t = _clean_telemetry(target_signal_present=False, **{field: bad_val})
    report = diagnose_assay(t)
    assert report.outcome is AttributionOutcome.EXECUTION_ARTIFACT
    assert report.action is DiagnosticAction.RETRY_ROBOT
    assert report.admit_to_memory is False
    assert expected_code in report.violated_execution_clauses


@pytest.mark.parametrize(
    ("field", "bad_val", "expected_code"),
    [
        ("detector_calibrated", False, "DETECTOR_UNCALIBRATED_OR_SATURATED"),
        ("precursor_overlap_permille", 151, "PRECURSOR_GHOST_PEAK_OVERLAP"),
        ("spectral_residual_permille", 101, "HIGH_SPECTRAL_RWP_RESIDUAL"),
        ("snr_tenths", 29, "LOW_ASSAY_SNR"),
    ],
)
def test_each_characterization_fault_blocks_certification_and_routes_orthogonal(
    field: str,
    bad_val: int | bool,
    expected_code: str,
) -> None:
    t = _clean_telemetry(target_signal_present=True, **{field: bad_val})
    report = diagnose_assay(t)
    assert report.outcome is AttributionOutcome.AMBIGUOUS_CHARACTERIZATION
    assert report.action is DiagnosticAction.RUN_ORTHOGONAL_ASSAY
    assert report.admit_to_memory is False
    assert expected_code in report.violated_characterization_clauses


def test_orthogonal_confirm_and_refute_promote_clean_runs() -> None:
    ghost = _clean_telemetry(precursor_overlap_permille=680, target_signal_present=True)
    raman_refute = OrthogonalAssayReadout(
        modality=OrthogonalModality.RAMAN,
        instrument_calibrated=True,
        orthogonal_snr_tenths=75,
        selectivity_clean=True,
        target_confirmed=False,
    )
    lcms_confirm = OrthogonalAssayReadout(
        modality=OrthogonalModality.LC_MS,
        instrument_calibrated=True,
        orthogonal_snr_tenths=92,
        selectivity_clean=True,
        target_confirmed=True,
    )
    rep_neg = diagnose_assay(ghost, raman_refute)
    assert rep_neg.outcome is AttributionOutcome.CERTIFIED_NEGATIVE
    assert rep_neg.admit_to_memory is True
    assert rep_neg.resolved_via_orthogonal is True
    assert rep_neg.lean_theorem_tag == "LabAttribLean.orthogonal_refute_promotes_certified_negative"

    rep_pos = diagnose_assay(ghost, lcms_confirm)
    assert rep_pos.outcome is AttributionOutcome.CERTIFIED_POSITIVE
    assert rep_pos.admit_to_memory is True
    assert rep_pos.resolved_via_orthogonal is True
    assert rep_pos.lean_theorem_tag == "LabAttribLean.orthogonal_confirm_promotes_certified_positive"


def test_orthogonal_cannot_launder_corrupted_execution() -> None:
    corrupted = _clean_telemetry(aspiration_clean=False, precursor_overlap_permille=600)
    lcms_confirm = OrthogonalAssayReadout(
        modality=OrthogonalModality.LC_MS,
        instrument_calibrated=True,
        orthogonal_snr_tenths=95,
        selectivity_clean=True,
        target_confirmed=True,
    )
    assert (
        resolve_with_orthogonal_assay(corrupted, lcms_confirm)
        is AttributionOutcome.EXECUTION_ARTIFACT
    )
    assert effective_attribution(corrupted, lcms_confirm) is AttributionOutcome.EXECUTION_ARTIFACT


def test_unconfirmed_orthogonal_readout_remains_ambiguous() -> None:
    ghost = _clean_telemetry(precursor_overlap_permille=500, target_signal_present=True)
    noisy_orth = OrthogonalAssayReadout(
        modality=OrthogonalModality.NMR_1H,
        instrument_calibrated=True,
        orthogonal_snr_tenths=20,
        selectivity_clean=True,
        target_confirmed=True,
    )
    assert orthogonal_readout_clean(noisy_orth) is False
    assert (
        resolve_with_orthogonal_assay(ghost, noisy_orth)
        is AttributionOutcome.AMBIGUOUS_CHARACTERIZATION
    )


def test_attributed_stepping_stone_gate_enforces_shell_volume_and_certification() -> None:
    clean_neg = _clean_telemetry()
    valid_probe = AttributedSteppingStone(
        tau_permille=300,
        eta_permille=250,
        dist_permille=360,
        proposed_new_tau_permille=360,
        volume_ul=8,
        uses_only_stock_reagents=True,
        within_hazard_envelope=True,
        telemetry=clean_neg,
    )
    assert check_attributed_frontier_update(valid_probe) is True

    deep_ood = AttributedSteppingStone(
        tau_permille=300,
        eta_permille=250,
        dist_permille=380,
        proposed_new_tau_permille=380,
        volume_ul=8,
        uses_only_stock_reagents=True,
        within_hazard_envelope=True,
        telemetry=clean_neg,
    )
    assert check_attributed_frontier_update(deep_ood) is False

    over_vol = AttributedSteppingStone(
        tau_permille=300,
        eta_permille=250,
        dist_permille=360,
        proposed_new_tau_permille=360,
        volume_ul=MAX_STEPPING_STONE_VOLUME_UL + 1,
        uses_only_stock_reagents=True,
        within_hazard_envelope=True,
        telemetry=clean_neg,
    )
    assert check_attributed_frontier_update(over_vol) is False

    corrupted_probe = AttributedSteppingStone(
        tau_permille=300,
        eta_permille=250,
        dist_permille=360,
        proposed_new_tau_permille=360,
        volume_ul=8,
        uses_only_stock_reagents=True,
        within_hazard_envelope=True,
        telemetry=_clean_telemetry(seal_intact=False),
    )
    assert check_attributed_frontier_update(corrupted_probe) is False

    amb_probe = AttributedSteppingStone(
        tau_permille=300,
        eta_permille=250,
        dist_permille=360,
        proposed_new_tau_permille=360,
        volume_ul=8,
        uses_only_stock_reagents=True,
        within_hazard_envelope=True,
        telemetry=_clean_telemetry(precursor_overlap_permille=450),
    )
    assert check_attributed_frontier_update(amb_probe) is False


def test_telemetry_and_orthogonal_validation_rejects_bad_types() -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        _clean_telemetry(dispense_error_permille=-1)
    with pytest.raises(ValueError, match="non-negative integer"):
        _clean_telemetry(snr_tenths=True)
    with pytest.raises(ValueError, match="must be a bool"):
        _clean_telemetry(seal_intact=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-negative integer"):
        OrthogonalAssayReadout(
            modality=OrthogonalModality.RAMAN,
            instrument_calibrated=True,
            orthogonal_snr_tenths=-5,
            selectivity_clean=True,
            target_confirmed=False,
        )
    with pytest.raises(ValueError, match="must be a bool"):
        AssayTelemetry.from_dict(
            {**_clean_telemetry().to_dict(), "aspiration_clean": "false"}
        )
    with pytest.raises(ValueError, match="must be an integer"):
        AssayTelemetry.from_dict(
            {**_clean_telemetry().to_dict(), "dispense_error_permille": True}
        )
    with pytest.raises(ValueError, match="must be a bool"):
        OrthogonalAssayReadout.from_dict(
            {
                "modality": "Raman",
                "instrument_calibrated": "false",
                "orthogonal_snr_tenths": 80,
                "selectivity_clean": True,
                "target_confirmed": True,
            }
        )


def test_lean_print_axioms_audit_and_readme_no_custom_axioms() -> None:
    import importlib.util

    script_path = REPO_ROOT / "scripts" / "check_lean_axioms.py"
    spec = importlib.util.spec_from_file_location("check_lean_axioms", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    require_bin = os.environ.get("LAB_ATTRIB_REQUIRE_LEAN_BIN") == "1"
    has_built_olean = (REPO_ROOT / "lean" / ".lake" / "build" / "lib").is_dir()
    run_kernel = bool(require_bin or (has_built_olean and mod.find_lake_executable() is not None))
    report = mod.audit_lean_axioms(run_kernel=run_kernel)
    assert report["theorem_count"] >= 30
    assert report["zero_sorryAx"] is True
    assert report["zero_custom_axioms"] is True
    if require_bin:
        assert report["kernel_verified"] is True
        assert len(report["theorem_axioms"]) == report["theorem_count"]


def test_lean_source_theorem_and_guard_counts_zero_sorry_zero_axiom() -> None:
    lean_files = [
        REPO_ROOT / "lean" / "LabAttribLean.lean",
        REPO_ROOT / "lean" / "Main.lean",
        REPO_ROOT / "lean" / "CheckAxioms.lean",
    ]
    total_theorems: list[str] = []
    total_guards: list[str] = []

    for lf in lean_files:
        assert lf.is_file(), f"Missing Lean file: {lf}"
        content = lf.read_text(encoding="utf-8")
        no_block_comments = re.sub(r"/-.*?-/", "", content, flags=re.DOTALL)
        code_lines = [line.split("--")[0] for line in no_block_comments.splitlines()]
        clean_code = "\n".join(code_lines)

        total_theorems.extend(
            re.findall(r"^\s*theorem\s+([A-Za-z0-9_']+)", clean_code, flags=re.MULTILINE)
        )
        total_guards.extend(re.findall(r"^\s*#guard\b", clean_code, flags=re.MULTILINE))
        sorries = re.findall(r"\bsorry\b", clean_code)
        axioms = re.findall(r"^\s*axiom\b", clean_code, flags=re.MULTILINE)
        assert sorries == [], f"Found forbidden sorry tokens in {lf.name}: {sorries}"
        assert axioms == [], f"Found forbidden axiom declarations in {lf.name}: {axioms}"

    assert len(total_theorems) >= 30, f"Expected >= 30 theorems, found {len(total_theorems)}"
    assert "tampered_telemetry_breaks_entry_hash" in total_theorems
    assert len(total_guards) >= 16, f"Expected >= 16 #guard checks, found {len(total_guards)}"


def test_fit_method_rwp_threshold_distinguishes_rietveld_and_stick_nnls() -> None:
    assert max_residual_for_fit_method("rietveld") == MAX_SPECTRAL_RESIDUAL_PERMILLE == 100
    assert max_residual_for_fit_method("stick_nnls") == MAX_STICK_NNLS_RESIDUAL_PERMILLE == 200

    # Real A-Lab spinel stick-NNLS hit range (e.g. Rwp = 140 permille, overlap = 41 permille):
    # quarantined under Rietveld limit (100), certified under stick_nnls limit (200)
    t_rietveld = _clean_telemetry(
        precursor_overlap_permille=41,
        spectral_residual_permille=140,
        snr_tenths=90,
        target_signal_present=True,
        fit_method="rietveld",
    )
    t_stick = _clean_telemetry(
        precursor_overlap_permille=41,
        spectral_residual_permille=140,
        snr_tenths=90,
        target_signal_present=True,
        fit_method="stick_nnls",
    )
    assert diagnose_assay(t_rietveld).outcome is AttributionOutcome.AMBIGUOUS_CHARACTERIZATION
    assert diagnose_assay(t_stick).outcome is AttributionOutcome.CERTIFIED_POSITIVE

    # Stick-NNLS above 200 permille still quarantines
    t_stick_high = _clean_telemetry(
        precursor_overlap_permille=41,
        spectral_residual_permille=201,
        snr_tenths=90,
        target_signal_present=True,
        fit_method="stick_nnls",
    )
    assert diagnose_assay(t_stick_high).outcome is AttributionOutcome.AMBIGUOUS_CHARACTERIZATION


def test_entire_repository_contains_strictly_zero_cjk_characters() -> None:
    skip_dirs = {".git", ".lake", "__pycache__", ".pytest_cache", ".ruff_cache", "data"}
    offending: list[str] = []
    checked_count = 0
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in skip_dirs or part.endswith(".egg-info") for part in path.parts):
            continue
        if path.suffix.lower() not in CHECKED_SUFFIXES:
            continue
        checked_count += 1
        text = path.read_text(encoding="utf-8")
        match = CJK_PATTERN.search(text)
        if match is not None:
            rel = path.relative_to(REPO_ROOT)
            offending.append(f"{rel} (char {match.group(0)!r} at offset {match.start()})")
    assert checked_count >= 10
    assert offending == [], f"CJK characters found in repository files: {offending}"
