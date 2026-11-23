"""Tests for SHA-256 `CertifiedLedger` and Bayesian EVOI `orthogonal_router`."""

from __future__ import annotations

from dataclasses import replace

import pytest

from lab_attrib.blame import (
    AssayTelemetry,
    AttributionOutcome,
    OrthogonalAssayReadout,
    OrthogonalModality,
)
from lab_attrib.ledger import (
    GENESIS_HASH,
    CertifiedLedger,
    LedgerIntegrityError,
    LedgerRejectionError,
)
from lab_attrib.orthogonal_router import route_orthogonal_assay


def _make_telemetry(**overrides: int | bool) -> AssayTelemetry:
    base: dict[str, int | bool] = {
        "dispense_error_permille": 14,
        "aspiration_clean": True,
        "incubation_window_clean": True,
        "seal_intact": True,
        "positive_control_passed": True,
        "blank_uncontaminated": True,
        "detector_calibrated": True,
        "precursor_overlap_permille": 25,
        "spectral_residual_permille": 40,
        "snr_tenths": 75,
        "target_signal_present": False,
    }
    base.update(overrides)
    return AssayTelemetry(**base)  # type: ignore[arg-type]


def test_ledger_rejects_corrupted_and_ambiguous_runs() -> None:
    ledger = CertifiedLedger()
    corrupted = _make_telemetry(aspiration_clean=False)
    with pytest.raises(LedgerRejectionError, match="EXECUTION_ARTIFACT"):
        ledger.append(
            sample_id="BAD-ROBOT-1",
            candidate_features={"mw": 210.0},
            measured_value=-8.5,
            telemetry=corrupted,
        )

    ambiguous = _make_telemetry(precursor_overlap_permille=520, target_signal_present=True)
    with pytest.raises(LedgerRejectionError, match="AMBIGUOUS_CHARACTERIZATION"):
        ledger.append(
            sample_id="GHOST-1",
            candidate_features={"mw": 310.0},
            measured_value=-1.9,
            telemetry=ambiguous,
        )
    assert len(ledger) == 0
    assert ledger.head_hash == GENESIS_HASH


def test_ledger_hash_chaining_json_roundtrip_and_export() -> None:
    ledger = CertifiedLedger()
    e0 = ledger.append(
        sample_id="NEG-001",
        candidate_features={"rings": 2.0, "mw": 198.2},
        measured_value=-4.6,
        telemetry=_make_telemetry(target_signal_present=False),
    )
    e1 = ledger.append(
        sample_id="HIT-002",
        candidate_features={"rings": 3.0, "mw": 274.3},
        measured_value=-2.3,
        telemetry=_make_telemetry(target_signal_present=True),
    )
    ghost = _make_telemetry(precursor_overlap_permille=610, target_signal_present=True)
    raman = OrthogonalAssayReadout(
        modality=OrthogonalModality.RAMAN,
        instrument_calibrated=True,
        orthogonal_snr_tenths=88,
        selectivity_clean=True,
        target_confirmed=False,
    )
    e2 = ledger.append(
        sample_id="GHOST-REFUTED-003",
        candidate_features={"rings": 4.0, "mw": 352.1},
        measured_value=-5.1,
        telemetry=ghost,
        orthogonal_readout=raman,
    )

    assert e0.prev_hash == GENESIS_HASH
    assert e1.prev_hash == e0.entry_hash
    assert e2.prev_hash == e1.entry_hash
    assert e2.outcome is AttributionOutcome.CERTIFIED_NEGATIVE
    assert ledger.verify_chain() == (True, None)

    exported = ledger.export_training_pairs()
    assert len(exported) == 3
    assert exported[0]["is_positive_hit"] is False
    assert exported[1]["is_positive_hit"] is True
    assert exported[2]["is_positive_hit"] is False

    raw_json = ledger.to_json()
    reloaded = CertifiedLedger.from_json(raw_json)
    assert len(reloaded) == 3
    assert reloaded.head_hash == ledger.head_hash


def test_ledger_detects_value_hash_and_outcome_tampering() -> None:
    ledger = CertifiedLedger()
    ledger.append(
        sample_id="NEG-001",
        candidate_features={"rings": 2.0},
        measured_value=-4.5,
        telemetry=_make_telemetry(target_signal_present=False),
    )
    ledger.append(
        sample_id="HIT-002",
        candidate_features={"rings": 3.0},
        measured_value=-2.1,
        telemetry=_make_telemetry(target_signal_present=True),
    )
    orig_entries = list(ledger.entries)

    # 1. Mutate measured_value without updating hash
    tampered_val = CertifiedLedger([replace(orig_entries[0], measured_value=-1.0), orig_entries[1]])
    ok, err = tampered_val.verify_chain()
    assert not ok and err is not None and "SHA-256 digest mismatch" in err
    with pytest.raises(LedgerIntegrityError):
        tampered_val.export_training_pairs()

    # 2. Break prev_hash link
    tampered_prev = CertifiedLedger([orig_entries[0], replace(orig_entries[1], prev_hash="f" * 64)])
    ok2, err2 = tampered_prev.verify_chain()
    assert not ok2 and err2 is not None and "Broken prev_hash link" in err2

    # 3. Forge outcome against raw telemetry
    tampered_out = CertifiedLedger(
        [replace(orig_entries[0], outcome=AttributionOutcome.CERTIFIED_POSITIVE), orig_entries[1]]
    )
    ok3, err3 = tampered_out.verify_chain()
    assert not ok3 and err3 is not None and "Outcome forgery" in err3


def test_evoi_router_selects_raman_for_solid_ghost_peak_and_tem_for_ultralow_volume() -> None:
    solid_ghost = _make_telemetry(precursor_overlap_permille=620, spectral_residual_permille=50)
    dec_raman = route_orthogonal_assay(
        solid_ghost,
        prior_hit_prob=0.30,
        available_volume_ul=5.0,
        sample_phase="solid",
    )
    assert dec_raman.discard_sample is False
    assert dec_raman.selected_modality is OrthogonalModality.RAMAN
    assert dec_raman.best_net_evoi_usd > 0.0

    # At 1.1 uL solid sample, Raman requires 1.5 uL -> only TEM (1.0 uL) is feasible!
    dec_tem = route_orthogonal_assay(
        solid_ghost,
        prior_hit_prob=0.60,
        available_volume_ul=1.1,
        sample_phase="solid",
    )
    assert dec_tem.discard_sample is False
    assert dec_tem.selected_modality is OrthogonalModality.TEM


def test_evoi_router_selects_lcms_in_solution_under_high_rwp_and_discards_unprofitable() -> None:
    sol_multiphase = _make_telemetry(
        precursor_overlap_permille=320,
        spectral_residual_permille=260,
    )
    dec_lcms = route_orthogonal_assay(
        sol_multiphase,
        prior_hit_prob=0.35,
        available_volume_ul=8.0,
        sample_phase="solution",
    )
    assert dec_lcms.discard_sample is False
    assert dec_lcms.selected_modality is OrthogonalModality.LC_MS

    # Zero discovery & negative-info value -> Net EVOI < 0 -> discard without poisoning memory
    dec_discard = route_orthogonal_assay(
        sol_multiphase,
        prior_hit_prob=0.0,
        available_volume_ul=8.0,
        sample_phase="solution",
        discovery_value_usd=100.0,
        certified_negative_info_value_usd=0.0,
    )
    assert dec_discard.discard_sample is True
    assert dec_discard.selected_modality is None


def test_evoi_router_rejects_non_ambiguous_input() -> None:
    clean = _make_telemetry(target_signal_present=True)
    with pytest.raises(ValueError, match="AMBIGUOUS_CHARACTERIZATION"):
        route_orthogonal_assay(clean)


def test_evoi_router_validates_probability_and_phase_bounds() -> None:
    amb = _make_telemetry(precursor_overlap_permille=400)
    with pytest.raises(ValueError, match="prior_hit_prob"):
        route_orthogonal_assay(amb, prior_hit_prob=1.5)
    with pytest.raises(ValueError, match="sample_phase"):
        route_orthogonal_assay(amb, sample_phase="gas")


def test_ledger_rejects_empty_sample_id_and_non_list_json() -> None:
    ledger = CertifiedLedger()
    with pytest.raises(ValueError, match="non-empty string"):
        ledger.append(
            sample_id="   ",
            candidate_features={"rings": 1.0},
            measured_value=-2.0,
            telemetry=_make_telemetry(),
        )
    with pytest.raises(ValueError, match="list of entry objects"):
        CertifiedLedger.from_json('{"not": "a list"}')


def test_ledger_detects_index_reordering_and_certificate_tag_tampering() -> None:
    ledger = CertifiedLedger()
    ledger.append(
        sample_id="A-1",
        candidate_features={"rings": 1.0},
        measured_value=-3.0,
        telemetry=_make_telemetry(target_signal_present=False),
    )
    ledger.append(
        sample_id="A-2",
        candidate_features={"rings": 3.0},
        measured_value=-2.0,
        telemetry=_make_telemetry(target_signal_present=True),
    )
    entries = list(ledger.entries)
    reordered = CertifiedLedger([entries[1], entries[0]])
    ok_idx, err_idx = reordered.verify_chain()
    assert not ok_idx and err_idx is not None and "Index mismatch" in err_idx

    bad_tag = CertifiedLedger([replace(entries[0], lean_certificate_tag="FakeTheorem"), entries[1]])
    ok_tag, err_tag = bad_tag.verify_chain()
    assert not ok_tag and err_tag is not None and "Certificate tag mismatch" in err_tag

