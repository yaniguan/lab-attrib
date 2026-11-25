"""Tests for SHA-256 `CertifiedLedger` and Bayesian EVOI `orthogonal_router`."""

from __future__ import annotations

import hmac
import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

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
    verify_inclusion_proof,
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

    tampered_val = CertifiedLedger(
        [replace(orig_entries[0], measured_value=-1.0), orig_entries[1]]
    )
    ok, err = tampered_val.verify_chain()
    assert not ok and err is not None and "SHA-256 digest mismatch" in err
    with pytest.raises(LedgerIntegrityError):
        tampered_val.export_training_pairs()

    tampered_prev = CertifiedLedger(
        [orig_entries[0], replace(orig_entries[1], prev_hash="f" * 64)]
    )
    ok2, err2 = tampered_prev.verify_chain()
    assert not ok2 and err2 is not None and "Broken prev_hash link" in err2

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


# --- Issue #6: Unclamped Bayes risk, correlated modality blindspots, low-SNR NMR attenuation ---


def test_issue6_unclamped_false_positive_loss_discards_low_prior_sample() -> None:
    amb_solid = AssayTelemetry(12, True, True, True, True, True, True, 650, 50, 80, True)
    dec = route_orthogonal_assay(
        amb_solid,
        prior_hit_prob=0.01,
        available_volume_ul=5.0,
        sample_phase="solid",
        discovery_value_usd=400.0,
        certified_negative_info_value_usd=45.0,
        false_positive_penalty_usd=1200.0,
    )
    assert dec.discard_sample is True
    assert dec.selected_modality is None
    raman_eval = next(e for e in dec.rankings if e.modality is OrthogonalModality.RAMAN)
    assert raman_eval.net_evoi_usd == pytest.approx(-3.4095, abs=1e-3)


def test_issue6_centrosymmetric_and_fluorescent_blindspots_and_low_snr_nmr_attenuation() -> None:
    amb_solid = _make_telemetry(
        precursor_overlap_permille=650,
        spectral_residual_permille=50,
        snr_tenths=80,
        target_signal_present=True,
    )

    # Centrosymmetric Fm-3m cubic phase: skips Raman -> routes to TEM when EVOI > 0
    dec_centro_tem = route_orthogonal_assay(
        amb_solid,
        prior_hit_prob=0.60,
        available_volume_ul=5.0,
        sample_phase="solid",
        centrosymmetric_crystal=True,
    )
    assert dec_centro_tem.discard_sample is False
    assert dec_centro_tem.selected_modality is OrthogonalModality.TEM
    raman_centro = next(
        e for e in dec_centro_tem.rankings if e.modality is OrthogonalModality.RAMAN
    )
    assert raman_centro.feasible is False

    # Same centrosymmetric crystal at low prior where TEM cost exceeds EVOI -> discard_sample=True
    dec_centro_discard = route_orthogonal_assay(
        amb_solid,
        prior_hit_prob=0.10,
        available_volume_ul=5.0,
        sample_phase="solid",
        centrosymmetric_crystal=True,
    )
    assert dec_centro_discard.discard_sample is True
    assert dec_centro_discard.selected_modality is None

    # High fluorescence risk / shared_blindspot_tags in solution skips Raman and picks LC-MS
    dec_fluor = route_orthogonal_assay(
        amb_solid,
        prior_hit_prob=0.30,
        available_volume_ul=8.0,
        sample_phase="solution",
        shared_blindspot_tags=("fluorescence",),
    )
    assert dec_fluor.selected_modality is OrthogonalModality.LC_MS

    # Low-SNR trace sample (snr_tenths < 30) with plenty of volume (20 uL) down-weights 1H-NMR vs LC-MS
    low_snr_trace = _make_telemetry(
        precursor_overlap_permille=40,
        spectral_residual_permille=45,
        snr_tenths=14,
        target_signal_present=True,
    )
    dec_trace = route_orthogonal_assay(
        low_snr_trace,
        prior_hit_prob=0.35,
        available_volume_ul=20.0,
        sample_phase="solution",
    )
    lcms_eval = next(e for e in dec_trace.rankings if e.modality is OrthogonalModality.LC_MS)
    nmr_eval = next(e for e in dec_trace.rankings if e.modality is OrthogonalModality.NMR_1H)
    assert lcms_eval.effective_resolvability > nmr_eval.effective_resolvability
    assert dec_trace.selected_modality is OrthogonalModality.LC_MS


# --- Issue #7: NaN/Inf rejection, 8-worker concurrency, Merkle proofs, atomic save & seals ---


def test_issue7_ledger_rejects_nan_and_infinity_in_append_and_from_json() -> None:
    ledger = CertifiedLedger()
    t_neg = _make_telemetry(target_signal_present=False)

    for bad_float in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="finite"):
            ledger.append(
                sample_id="NAN-1",
                candidate_features={"mw": 200.0},
                measured_value=bad_float,
                telemetry=t_neg,
            )
        with pytest.raises(ValueError, match="finite"):
            ledger.append(
                sample_id="NAN-2",
                candidate_features={"mw": bad_float},
                measured_value=-3.2,
                telemetry=t_neg,
            )

    # RFC-8259-invalid literal NaN in raw JSON must also be rejected with ValueError
    raw_nan_json = (
        '[{"index":0,"sample_id":"X","candidate_features":{"mw":NaN},'
        '"measured_value":-2.0,"telemetry":{},"orthogonal_readout":null,'
        '"outcome":"CERTIFIED_NEGATIVE","lean_certificate_tag":"x",'
        '"prev_hash":"' + ("0" * 64) + '","entry_hash":"' + ("a" * 64) + '"}]'
    )
    with pytest.raises(ValueError):
        CertifiedLedger.from_json(raw_nan_json)


def test_issue7_concurrent_8_worker_200_appends_strictly_sequential() -> None:
    ledger = CertifiedLedger()

    def _worker(i: int) -> str:
        t = _make_telemetry(target_signal_present=(i % 3 == 0))
        entry = ledger.append(
            sample_id=f"CONC-RUN-{i:04d}",
            candidate_features={"rings": float(i % 5), "mw": 180.0 + float(i)},
            measured_value=-4.0 + 0.01 * float(i),
            telemetry=t,
        )
        return entry.entry_hash

    with ThreadPoolExecutor(max_workers=8) as pool:
        hashes = list(pool.map(_worker, range(200)))

    assert len(hashes) == 200
    assert len(set(hashes)) == 200
    assert len(ledger) == 200
    assert [e.index for e in ledger.entries] == list(range(200))
    assert ledger.verify_chain() == (True, None)


def test_issue7_merkle_proof_and_hmac_ed25519_seals_detect_negative_history_pruning(
    tmp_path: Path,
) -> None:
    secret = b"workcell-hsm-anchor-key-2026"
    honest = CertifiedLedger(secret_key=secret)
    e0 = honest.append(
        sample_id="TRUE-NEG-001",
        candidate_features={"rings": 3.0},
        measured_value=-5.2,
        telemetry=_make_telemetry(target_signal_present=False),
    )
    e1 = honest.append(
        sample_id="TRUE-HIT-002",
        candidate_features={"rings": 3.0},
        measured_value=-2.1,
        telemetry=_make_telemetry(target_signal_present=True),
    )
    e2 = honest.append(
        sample_id="TRUE-NEG-003",
        candidate_features={"rings": 4.0},
        measured_value=-4.9,
        telemetry=_make_telemetry(target_signal_present=False),
    )

    pinned_root = honest.merkle_root()
    seal = honest.seal_head()
    assert honest.verify_chain(expected_merkle_root=pinned_root, head_seal=seal) == (True, None)

    for idx, entry in enumerate((e0, e1, e2)):
        proof = honest.inclusion_proof(idx)
        assert verify_inclusion_proof(entry.entry_hash, idx, proof, pinned_root) is True
        assert verify_inclusion_proof(entry.entry_hash, proof, pinned_root) is True

    # Selective negative pruning attack: rebuild chain keeping only hits and recomputing prev_hash
    pruned = CertifiedLedger()
    pruned.append(
        sample_id=e1.sample_id,
        candidate_features=e1.candidate_features,
        measured_value=e1.measured_value,
        telemetry=e1.telemetry,
    )
    ok_root, err_root = pruned.verify_chain(expected_merkle_root=pinned_root)
    assert not ok_root and err_root is not None and "Merkle root mismatch" in err_root
    assert (
        verify_inclusion_proof(
            pruned.entries[0].entry_hash, 0, pruned.inclusion_proof(0), pinned_root
        )
        is False
    )
    ok_seal, _ = pruned.verify_head_seal(seal, secret_key=secret)
    assert ok_seal is False

    # Atomic file-locked save/load + JSONL WAL with optimistic CAS
    saved_path = tmp_path / "certified_ledger.json"
    honest.save(saved_path)
    reloaded = CertifiedLedger.load(saved_path, secret_key=secret)
    assert reloaded.merkle_root() == pinned_root

    wal_path = tmp_path / "wal_ledger.jsonl"
    w0 = CertifiedLedger.append_to_jsonl(
        wal_path,
        sample_id="WAL-001",
        candidate_features={"mw": 210.0},
        measured_value=-3.5,
        telemetry=_make_telemetry(target_signal_present=False),
        expected_head_hash=GENESIS_HASH,
    )
    with pytest.raises(LedgerIntegrityError, match="CAS conflict"):
        CertifiedLedger.append_to_jsonl(
            wal_path,
            sample_id="WAL-002-STALE",
            candidate_features={"mw": 220.0},
            measured_value=-2.5,
            telemetry=_make_telemetry(target_signal_present=True),
            expected_head_hash=GENESIS_HASH,
        )
    CertifiedLedger.append_to_jsonl(
        wal_path,
        sample_id="WAL-002",
        candidate_features={"mw": 220.0},
        measured_value=-2.5,
        telemetry=_make_telemetry(target_signal_present=True),
        expected_head_hash=w0.entry_hash,
    )
    wal_loaded = CertifiedLedger.load_jsonl(wal_path)
    assert len(wal_loaded) == 2

    # Also test Ed25519 sign/verify interface via mock asymmetric signer
    class _MockEd25519Signer:
        def sign(self, data: bytes) -> bytes:
            return hmac.new(b"ed25519-sim", data, hashlib.sha512).digest()

        def verify(self, sig: bytes, data: bytes) -> None:
            expected = hmac.new(b"ed25519-sim", data, hashlib.sha512).digest()
            if not hmac.compare_digest(sig, expected):
                raise ValueError("bad signature")

    signer = _MockEd25519Signer()
    ed_seal = honest.seal_head(ed25519_private_key=signer)
    assert ed_seal["scheme"] == "ed25519"
    assert honest.verify_head_seal(ed_seal, ed25519_public_key=signer) == (True, None)


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

    bad_tag = CertifiedLedger(
        [replace(entries[0], lean_certificate_tag="FakeTheorem"), entries[1]]
    )
    ok_tag, err_tag = bad_tag.verify_chain()
    assert not ok_tag and err_tag is not None and "Certificate tag mismatch" in err_tag
