"""Tests for `autolab_wire.py` (`autolab-wire/v1` Layer L4 post-execution causal blame node)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

import autolab_wire
from lab_attrib.blame import diagnose_assay
from lab_attrib.ledger import CertifiedLedger
from lab_attrib.xrd_deconvolve import build_gaussian_spectrum, deconvolve_spectrum


def _two_theta_grid() -> list[float]:
    return [10.0 + 1.25 * idx for idx in range(48)]


def test_wire_metadata_and_cwd_independent_dynamic_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert autolab_wire.REPO_ID == "lab-attrib"
    assert autolab_wire.LAYER == "L4"
    assert autolab_wire.SCHEMA_VERSION == "autolab-wire/v1"
    assert autolab_wire.REPO_DIR.is_dir()

    monkeypatch.chdir(tmp_path)
    wire_path = autolab_wire.REPO_DIR / "autolab_wire.py"
    spec = importlib.util.spec_from_file_location("lab_attrib_dynamic_wire", wire_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    out = module.process_packet({"run_id": "cwd-check-001", "candidate_smiles": "CCO"})
    assert out["halted"] is False
    assert out["attribution"]["outcome"] == "CERTIFIED_POSITIVE"
    assert len(out["certified_ledger_entries"]) == 1


def test_clean_certified_positive_and_certified_negative_ledger_append() -> None:
    shared_ledger = CertifiedLedger()
    original_packet: dict[str, Any] = {
        "run_id": "run-pos-001",
        "candidate_smiles": "CC(=O)Oc1ccccc1C(=O)O",
        "edge_runtime": {
            "dispense_error_permille": 11,
            "aspiration_clean": True,
            "incubation_window_clean": True,
            "seal_intact": True,
        },
    }

    pos_out = autolab_wire.process_packet(original_packet, ledger=shared_ledger)
    assert "attribution" not in original_packet
    assert pos_out["schema_version"] == "autolab-wire/v1"
    assert pos_out["halted"] is False
    assert pos_out["halt_reason"] is None

    pos_attr = pos_out["attribution"]
    assert pos_attr["outcome"] == "CERTIFIED_POSITIVE"
    assert pos_attr["action"] == "COMMIT_CERTIFIED_HIT"
    assert pos_attr["admit_to_memory"] is True
    assert pos_attr["orthogonal_modality"] is None
    assert isinstance(pos_attr["ledger_entry_sha256"], str)
    assert len(pos_attr["ledger_entry_sha256"]) == 64

    assert len(pos_out["certified_ledger_entries"]) == 1
    pos_entry = pos_out["certified_ledger_entries"][0]
    assert pos_entry == {
        "run_id": "run-pos-001",
        "candidate_smiles": "CC(=O)Oc1ccccc1C(=O)O",
        "outcome": "CERTIFIED_POSITIVE",
        "admit_to_memory": True,
        "ledger_entry_sha256": pos_attr["ledger_entry_sha256"],
    }
    assert pos_out["trace"][-1]["repo"] == "lab-attrib"
    assert pos_out["trace"][-1]["layer"] == "L4"
    assert pos_out["trace"][-1]["status"] == "PASS"
    assert (
        pos_out["bottom_reasoning_traces"][-1]["invariant"]
        == "corrupted_run_never_poisons_memory & unidentifiable_spectrum_never_certified"
    )

    neg_in = dict(pos_out)
    neg_in["run_id"] = "run-neg-002"
    neg_in["candidate_smiles"] = "c1ccccc1"
    neg_in["telemetry_override"] = {"target_signal_present": False, "snr_tenths": 90}

    neg_out = autolab_wire.process_packet(neg_in, ledger=shared_ledger)
    assert neg_out["halted"] is False
    assert neg_out["halt_reason"] is None

    neg_attr = neg_out["attribution"]
    assert neg_attr["outcome"] == "CERTIFIED_NEGATIVE"
    assert neg_attr["action"] == "COMMIT_CERTIFIED_NEGATIVE"
    assert neg_attr["admit_to_memory"] is True
    assert neg_attr["orthogonal_modality"] is None
    assert isinstance(neg_attr["ledger_entry_sha256"], str)
    assert len(neg_attr["ledger_entry_sha256"]) == 64
    assert neg_attr["ledger_entry_sha256"] != pos_attr["ledger_entry_sha256"]

    assert len(neg_out["certified_ledger_entries"]) == 2
    assert neg_out["certified_ledger_entries"][1] == {
        "run_id": "run-neg-002",
        "candidate_smiles": "c1ccccc1",
        "outcome": "CERTIFIED_NEGATIVE",
        "admit_to_memory": True,
        "ledger_entry_sha256": neg_attr["ledger_entry_sha256"],
    }
    assert len(shared_ledger) == 2
    assert shared_ledger.entries[1].prev_hash == shared_ledger.entries[0].entry_hash
    assert shared_ledger.verify_chain() == (True, None)


def test_live_nnls_xrd_mixture_ghost_peaks_quarantines_and_routes_evoi_modality() -> None:
    grid = _two_theta_grid()
    prec_clean = build_gaussian_spectrum(grid, [(16.0, 1.0, 1.8)])
    prec_ghost = build_gaussian_spectrum(
        grid, [(34.0, 0.96, 2.1), (52.0, 0.90, 2.1), (22.0, 0.35, 1.7)]
    )
    target_basis = build_gaussian_spectrum(grid, [(34.0, 1.0, 2.0), (52.0, 0.92, 2.0)])
    observed = [0.20 * prec_clean[i] + 1.10 * prec_ghost[i] for i in range(len(grid))]

    packet: dict[str, Any] = {
        "run_id": "run-ghost-xrd",
        "candidate_smiles": "O=C1NC(=O)c2ccccc21",
        "xrd_mixture": {
            "observed": observed,
            "target_basis": target_basis,
            "precursor_bases": [prec_clean, prec_ghost],
        },
    }

    out = autolab_wire.process_packet(packet)
    assert out["halted"] is True
    assert out["halt_reason"] == "L4_ATTRIB_QUARANTINE"
    assert out["certified_ledger_entries"] == []

    attr = out["attribution"]
    assert attr["outcome"] == "AMBIGUOUS_CHARACTERIZATION"
    assert attr["action"] == "RUN_ORTHOGONAL_ASSAY"
    assert attr["admit_to_memory"] is False
    assert attr["precursor_overlap_permille"] > 700
    assert attr["orthogonal_modality"] in {"Raman", "LC-MS", "TEM", "1H-NMR"}
    assert attr["orthogonal_modality"] == "Raman"
    assert attr["ledger_entry_sha256"] is None
    assert out["trace"][-1]["status"] == "QUARANTINE"

    mystery_impurity = build_gaussian_spectrum(grid, [(48.0, 1.25, 2.0), (61.0, 1.05, 2.0)])
    observed_rwp = [
        0.55 * prec_ghost[i] + 0.95 * mystery_impurity[i] for i in range(len(grid))
    ]
    out_lcms = autolab_wire.process_packet(
        {
            "run_id": "run-multiphase-rwp",
            "prior_hit_prob": 0.35,
            "available_volume_ul": 8.0,
            "sample_phase": "solution",
            "xrd_mixture": {
                "observed": observed_rwp,
                "target_basis": target_basis,
                "precursor_bases": [prec_clean, prec_ghost],
            },
        }
    )
    assert out_lcms["halted"] is True
    assert out_lcms["halt_reason"] == "L4_ATTRIB_QUARANTINE"
    assert out_lcms["attribution"]["outcome"] == "AMBIGUOUS_CHARACTERIZATION"
    assert out_lcms["attribution"]["spectral_residual_permille"] > 100
    assert out_lcms["attribution"]["orthogonal_modality"] == "LC-MS"


def test_robotic_dispense_error_85_permille_quarantines_execution_artifact() -> None:
    packet: dict[str, Any] = {
        "run_id": "run-dispense-slip",
        "candidate_smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "edge_runtime": {
            "dispense_error_permille": 85,
            "aspiration_clean": True,
            "incubation_window_clean": True,
            "seal_intact": True,
        },
        "telemetry_override": {
            "target_signal_present": False,
        },
    }

    out = autolab_wire.process_packet(packet)
    assert out["halted"] is True
    assert out["halt_reason"] == "L4_ATTRIB_QUARANTINE"
    assert out["certified_ledger_entries"] == []

    attr = out["attribution"]
    assert attr["outcome"] == "EXECUTION_ARTIFACT"
    assert attr["action"] == "RETRY_ROBOT"
    assert attr["admit_to_memory"] is False
    assert attr["orthogonal_modality"] is None
    assert attr["ledger_entry_sha256"] is None
    assert out["trace"][-1]["status"] == "QUARANTINE"


def test_halted_packet_passthrough_is_unchanged() -> None:
    upstream_halted: dict[str, Any] = {
        "schema_version": "autolab-wire/v1",
        "halted": True,
        "halt_reason": "L4_REHEARSAL_BLOCK",
        "trace": [{"repo": "lab-rehearsal", "layer": "L4", "status": "BLOCK", "summary": "overfill"}],
        "bottom_reasoning_traces": [],
        "certified_ledger_entries": [],
    }
    out = autolab_wire.process_packet(upstream_halted)
    assert out == upstream_halted
    assert out is not upstream_halted
    assert "attribution" not in out


# --- Issue #9: Two-turn orthogonal resolution & deconvolved snr_tenths propagation ---


def test_issue9_two_turn_orthogonal_resolution_unhalts_and_commits_ledger() -> None:
    grid = _two_theta_grid()
    prec_clean = build_gaussian_spectrum(grid, [(16.0, 1.0, 1.8)])
    prec_ghost = build_gaussian_spectrum(
        grid, [(34.0, 0.96, 2.1), (52.0, 0.90, 2.1), (22.0, 0.35, 1.7)]
    )
    target_basis = build_gaussian_spectrum(grid, [(34.0, 1.0, 2.0), (52.0, 0.92, 2.0)])
    observed_ghost = [0.20 * prec_clean[i] + 1.10 * prec_ghost[i] for i in range(len(grid))]

    turn1 = autolab_wire.process_packet(
        {
            "run_id": "ghost-2turn-001",
            "candidate_smiles": "c1ccc2c(c1)ccn2C",
            "xrd_mixture": {
                "observed": observed_ghost,
                "target_basis": target_basis,
                "precursor_bases": [prec_clean, prec_ghost],
            },
        }
    )
    assert turn1["halted"] is True
    assert turn1["halt_reason"] == "L4_ATTRIB_QUARANTINE"
    assert turn1["attribution"]["outcome"] == "AMBIGUOUS_CHARACTERIZATION"
    assert turn1["attribution"]["orthogonal_modality"] == "Raman"

    # Turn 2a: Orthogonal Raman refutation -> CERTIFIED_NEGATIVE, unhalted, committed!
    turn2_neg_in = dict(turn1)
    turn2_neg_in["orthogonal_readout"] = {
        "modality": turn1["attribution"]["orthogonal_modality"],
        "instrument_calibrated": True,
        "orthogonal_snr_tenths": 90,
        "selectivity_clean": True,
        "target_confirmed": False,
    }
    turn2_neg = autolab_wire.process_packet(turn2_neg_in)
    assert turn2_neg["halted"] is False
    assert turn2_neg["halt_reason"] is None
    assert turn2_neg["attribution"]["outcome"] == "CERTIFIED_NEGATIVE"
    assert turn2_neg["attribution"]["resolved_via_orthogonal"] is True
    assert turn2_neg["attribution"]["orthogonal_modality"] == "Raman"
    assert len(turn2_neg["certified_ledger_entries"]) == 1
    assert len(turn2_neg["attribution"]["ledger_entry_sha256"]) == 64

    # Turn 2b: Orthogonal LC-MS confirmation -> CERTIFIED_POSITIVE, unhalted, committed!
    turn2_pos_in = dict(turn1)
    turn2_pos_in["orthogonal_readout"] = {
        "modality": "LC-MS",
        "instrument_calibrated": True,
        "orthogonal_snr_tenths": 95,
        "selectivity_clean": True,
        "target_confirmed": True,
    }
    turn2_pos = autolab_wire.process_packet(turn2_pos_in)
    assert turn2_pos["halted"] is False
    assert turn2_pos["halt_reason"] is None
    assert turn2_pos["attribution"]["outcome"] == "CERTIFIED_POSITIVE"
    assert turn2_pos["attribution"]["resolved_via_orthogonal"] is True
    assert turn2_pos["attribution"]["orthogonal_modality"] == "LC-MS"
    assert len(turn2_pos["certified_ledger_entries"]) == 1

    # Upstream halts MUST remain untouched even if orthogonal_readout is attached
    for upstream_reason in ("L4_REHEARSAL_BLOCK", "L6_ABSTAIN"):
        upstream_pkt: dict[str, Any] = {
            "halted": True,
            "halt_reason": upstream_reason,
            "orthogonal_readout": turn2_pos_in["orthogonal_readout"],
        }
        res_up = autolab_wire.process_packet(upstream_pkt)
        assert res_up["halted"] is True
        assert res_up["halt_reason"] == upstream_reason
        assert "attribution" not in res_up


def test_issue9_high_noise_xrd_mixture_propagates_deconvolved_snr_and_quarantines() -> None:
    grid = _two_theta_grid()
    prec_clean = build_gaussian_spectrum(grid, [(16.0, 1.0, 1.8)])
    target_basis = build_gaussian_spectrum(grid, [(34.0, 1.0, 2.0), (52.0, 0.92, 2.0)])

    deconv_noisy = deconvolve_spectrum(
        [0.0] * len(grid), [prec_clean], target_basis, min_hit_amplitude=0.20, noise_sigma=5.0
    )
    assert deconv_noisy.snr_tenths < 30
    assert diagnose_assay(deconv_noisy.to_assay_telemetry()).outcome.value == "AMBIGUOUS_CHARACTERIZATION"

    wire_noisy = autolab_wire.process_packet(
        {
            "run_id": "unresolvable-snr-002",
            "candidate_smiles": "CCO",
            "xrd_mixture": {
                "observed": [0.0] * len(grid),
                "target_basis": target_basis,
                "precursor_bases": [prec_clean],
                "noise_sigma": 5.0,
            },
        }
    )
    assert wire_noisy["halted"] is True
    assert wire_noisy["halt_reason"] == "L4_ATTRIB_QUARANTINE"
    assert wire_noisy["attribution"]["outcome"] == "AMBIGUOUS_CHARACTERIZATION"
    assert "LOW_ASSAY_SNR" in wire_noisy["attribution"]["violated_characterization_clauses"]


# --- Issue #4, #5, #7: Compound wire diagnostics, bare-packet rejection, stateless prev_hash ---


def test_issue4_and_issue5_wire_compound_clauses_and_readout_free_quarantine() -> None:
    # Issue #4: seal breach + uncalibrated detector surfaces exact fault clauses & dual_fault
    pkt_dual = autolab_wire.process_packet(
        {
            "run_id": "seal-and-detector-fault",
            "edge_runtime": {"dispense_error_permille": 12, "seal_intact": False},
            "telemetry_override": {"detector_calibrated": False},
        }
    )
    assert pkt_dual["attribution"]["dual_fault_detected"] is True
    assert "SEAL_INTEGRITY_BREACH" in pkt_dual["attribution"]["violated_execution_clauses"]
    assert (
        "DETECTOR_UNCALIBRATED_OR_SATURATED"
        in pkt_dual["attribution"]["violated_characterization_clauses"]
    )
    summary = pkt_dual["trace"][-1]["summary"]
    assert "SEAL_INTEGRITY_BREACH" in summary
    assert "DETECTOR_UNCALIBRATED_OR_SATURATED" in summary
    assert "RECALIBRATE_AND_RETRY" in summary

    # Issue #5: Readout-free / bare packets quarantine instead of default-allow CERTIFIED_POSITIVE
    for bare_pkt in ({}, {"run_id": "no-sensor-data-001"}):
        out_bare = autolab_wire.process_packet(bare_pkt)
        assert out_bare["halted"] is True
        assert out_bare["halt_reason"] == "L4_ATTRIB_QUARANTINE"
        assert out_bare["attribution"]["admit_to_memory"] is False
        assert out_bare["certified_ledger_entries"] == []

    # strict_evidence_required=True also rejects SMILES-only packet lacking execution/sensor data
    out_strict = autolab_wire.process_packet(
        {"run_id": "strict-001", "candidate_smiles": "CCO"},
        strict_evidence_required=True,
    )
    assert out_strict["halted"] is True
    assert "MISSING_EXPERIMENTAL_EVIDENCE" in out_strict["attribution"]["violated_characterization_clauses"]


def test_issue7_stateless_multi_turn_wire_packets_preserve_prev_hash_continuity() -> None:
    stateful = CertifiedLedger()
    p1_in = {
        "run_id": "chain-001",
        "candidate_smiles": "CCO",
        "edge_runtime": {"dispense_error_permille": 10},
    }
    stateless_1 = autolab_wire.process_packet(p1_in)
    stateful_1 = autolab_wire.process_packet(p1_in, ledger=stateful)
    assert (
        stateless_1["attribution"]["ledger_entry_sha256"]
        == stateful_1["attribution"]["ledger_entry_sha256"]
    )

    p2_stateless_in = dict(stateless_1)
    p2_stateless_in["run_id"] = "chain-002"
    p2_stateless_in["candidate_smiles"] = "c1ccccc1"
    p2_stateless_in["telemetry_override"] = {"target_signal_present": False, "snr_tenths": 80}

    p2_stateful_in = dict(stateful_1)
    p2_stateful_in["run_id"] = "chain-002"
    p2_stateful_in["candidate_smiles"] = "c1ccccc1"
    p2_stateful_in["telemetry_override"] = {"target_signal_present": False, "snr_tenths": 80}

    stateless_2 = autolab_wire.process_packet(p2_stateless_in)
    stateful_2 = autolab_wire.process_packet(p2_stateful_in, ledger=stateful)
    assert (
        stateless_2["attribution"]["ledger_entry_sha256"]
        == stateful_2["attribution"]["ledger_entry_sha256"]
    )
