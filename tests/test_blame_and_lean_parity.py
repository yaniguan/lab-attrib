"""Unit, boundary, and property tests for 4-way blame calculus, Lean 4 parity, and zero-CJK."""

from __future__ import annotations

import itertools
import re
from pathlib import Path

import pytest

from lab_attrib.blame import (
    MAX_DISPENSE_ERROR_PERMILLE,
    MAX_PRECURSOR_OVERLAP_PERMILLE,
    MAX_SPECTRAL_RESIDUAL_PERMILLE,
    MAX_STEPPING_STONE_VOLUME_UL,
    MIN_RESOLVABLE_SNR_TENTHS,
    AssayTelemetry,
    AttributedSteppingStone,
    AttributionOutcome,
    DiagnosticAction,
    OrthogonalAssayReadout,
    OrthogonalModality,
    admit_to_scientific_memory,
    attribute_assay_outcome,
    characterization_identifiable,
    check_attributed_frontier_update,
    diagnose_assay,
    effective_attribution,
    execution_clean,
    orthogonal_readout_clean,
    resolve_with_orthogonal_assay,
    route_diagnostic_action,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CJK_PATTERN = re.compile(
    r"[\u2e80-\u2eff\u3000-\u303f\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef]"
)
CHECKED_SUFFIXES = {".py", ".lean", ".md", ".toml", ".yml", ".yaml", ".json", ".tex", ".bib", ".sty"}


def _clean_telemetry(**overrides: int | bool) -> AssayTelemetry:
    base: dict[str, int | bool] = {
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
    }
    base.update(overrides)
    return AssayTelemetry(**base)  # type: ignore[arg-type]


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
    assert route_diagnostic_action(attribute_assay_outcome(cavitation)) is DiagnosticAction.RETRY_ROBOT
    assert admit_to_scientific_memory(attribute_assay_outcome(cavitation)) is False

    assert attribute_assay_outcome(alab_ghost) is AttributionOutcome.AMBIGUOUS_CHARACTERIZATION
    assert (
        route_diagnostic_action(attribute_assay_outcome(alab_ghost))
        is DiagnosticAction.RUN_ORTHOGONAL_ASSAY
    )
    assert admit_to_scientific_memory(attribute_assay_outcome(alab_ghost)) is False


def test_exhaustive_four_way_partition_across_boolean_and_threshold_grid() -> None:
    for (
        disp,
        asp,
        inc,
        seal,
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
    assert resolve_with_orthogonal_assay(corrupted, lcms_confirm) is AttributionOutcome.EXECUTION_ARTIFACT
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
    assert resolve_with_orthogonal_assay(ghost, noisy_orth) is AttributionOutcome.AMBIGUOUS_CHARACTERIZATION


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

    # Deep OOD: 380 > 1.25 * 300 = 375
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

    # Exceeding micro volume (> 10 uL)
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

    # Corrupted telemetry inside shell
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

    # Ambiguous telemetry without orthogonal resolution inside shell
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
        _clean_telemetry(snr_tenths=True)  # bool is subclass of int; must be rejected!
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


def test_lean_source_theorem_and_guard_counts_zero_sorry_zero_axiom() -> None:
    lean_file = REPO_ROOT / "lean" / "LabAttribLean.lean"
    assert lean_file.is_file()
    content = lean_file.read_text(encoding="utf-8")
    # Strip comments before scanning for `sorry` and `axiom`
    no_block_comments = re.sub(r"/-.*?-/", "", content, flags=re.DOTALL)
    code_lines = [
        line.split("--")[0] for line in no_block_comments.splitlines()
    ]
    clean_code = "\n".join(code_lines)

    theorems = re.findall(r"^\s*theorem\s+([A-Za-z0-9_']+)", clean_code, flags=re.MULTILINE)
    guards = re.findall(r"^\s*#guard\b", clean_code, flags=re.MULTILINE)
    sorries = re.findall(r"\bsorry\b", clean_code)
    axioms = re.findall(r"^\s*axiom\b", clean_code, flags=re.MULTILINE)

    assert len(theorems) >= 18, f"Expected >= 18 theorems, found {len(theorems)}"
    assert len(guards) >= 12, f"Expected >= 12 #guard checks, found {len(guards)}"
    assert sorries == [], f"Found forbidden sorry tokens: {sorries}"
    assert axioms == [], f"Found forbidden axiom declarations: {axioms}"


def test_entire_repository_contains_strictly_zero_cjk_characters() -> None:
    skip_dirs = {".git", ".lake", "__pycache__", ".pytest_cache", ".ruff_cache"}
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
