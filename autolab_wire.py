"""Auto-Lab 15-repository wire contract node (`autolab-wire/v1`) for `lab-attrib` (Layer L4).

Post-execution causal blame gate, NNLS precursor ghost-peak deconvolver, Bayesian EVOI
orthogonal assay router, and SHA-256 proof-carrying `CertifiedLedger` firewall sitting
between wet-lab execution (`stepbot`, `lab-feet`) and downstream learning/recalibration
(`twinstep`, `labbench-posttrain`, `lab-decide`).
"""

import copy
import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ID: str = "lab-attrib"
LAYER: str = "L4"
SCHEMA_VERSION: str = "autolab-wire/v1"
REPO_DIR: Path = Path(__file__).resolve().parent

_SRC_DIR = REPO_DIR / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from lab_attrib.blame import (  # noqa: E402
    AssayTelemetry,
    AttributionOutcome,
    DiagnosticAction,
    OrthogonalAssayReadout,
    diagnose_assay,
)
from lab_attrib.ledger import (  # noqa: E402
    GENESIS_HASH,
    CertifiedLedger,
    LedgerIntegrityError,
)
from lab_attrib.orthogonal_router import route_orthogonal_assay  # noqa: E402
from lab_attrib.xrd_deconvolve import DeconvolutionResult, deconvolve_spectrum  # noqa: E402

WIRE_INVARIANT: str = (
    "corrupted_run_never_poisons_memory & unidentifiable_spectrum_never_certified"
)


def _nonneg_int(value: Any, default: int) -> int:
    if value is None or isinstance(value, bool):
        return default
    fval = float(value)
    if not math.isfinite(fval):
        raise ValueError(f"Telemetry integer field must be finite, got {value!r}")
    return max(0, int(round(fval)))


def _signed_int(value: Any, default: int) -> int:
    if value is None or isinstance(value, bool):
        return default
    fval = float(value)
    if not math.isfinite(fval):
        raise ValueError(f"Telemetry signed integer field must be finite, got {value!r}")
    return int(round(fval))


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    return bool(value)


def _extract_candidate_features(
    packet: Mapping[str, Any], candidate_smiles: str
) -> dict[str, float]:
    raw_features = packet.get("candidate_features")
    if isinstance(raw_features, Mapping) and raw_features:
        return {str(k): float(v) for k, v in raw_features.items()}
    parsed_spans = packet.get("parsed_spans")
    vol_ul = (
        float(parsed_spans.get("volume_ul", 100.0))
        if isinstance(parsed_spans, Mapping)
        else 100.0
    )
    return {
        "smiles_length": float(len(candidate_smiles)),
        "knn_distance": float(packet.get("knn_distance", 0.412)),
        "volume_ul": vol_ul,
    }


def _packet_has_sensor_or_exec_evidence(packet: Mapping[str, Any]) -> bool:
    for mapping_key in (
        "xrd_mixture",
        "edge_runtime",
        "stepbot_execution",
        "telemetry_override",
        "orthogonal_readout",
        "execution_receipt",
    ):
        val = packet.get(mapping_key)
        if isinstance(val, Mapping) and bool(val):
            return True
    if packet.get("measured_value") is not None:
        return True
    if packet.get("dispense_error_permille") is not None:
        return True
    return False


def _packet_has_candidate_context(packet: Mapping[str, Any]) -> bool:
    if str(packet.get("candidate_smiles") or "").strip():
        return True
    for ctx_key in ("candidate_features", "parsed_spans", "protocol", "trace"):
        val = packet.get(ctx_key)
        if val:
            return True
    return False


def process_packet(
    packet: dict[str, Any],
    *,
    ledger: CertifiedLedger | None = None,
    strict_evidence_required: bool = False,
) -> dict[str, Any]:
    """Execute the `L4` `lab-attrib` post-execution verification gate on an `autolab-wire/v1` packet."""
    out: dict[str, Any] = copy.deepcopy(packet)
    if out.get("schema_version") is None:
        out["schema_version"] = SCHEMA_VERSION
    if out.get("halted") is None:
        out["halted"] = False
    if "halt_reason" not in out:
        out["halt_reason"] = None
    if out.get("trace") is None:
        out["trace"] = []
    if out.get("bottom_reasoning_traces") is None:
        out["bottom_reasoning_traces"] = []
    if out.get("certified_ledger_entries") is None:
        out["certified_ledger_entries"] = []

    orth_raw = out.get("orthogonal_readout")
    has_orthogonal_turn2 = isinstance(orth_raw, Mapping) and bool(orth_raw)
    has_retry_turn2 = _nonneg_int(out.get("retry_attempt"), 0) > 0

    if bool(out["halted"]):
        # Upstream halts (e.g., L4_REHEARSAL_BLOCK, L6_ABSTAIN) ALWAYS pass through untouched.
        if out.get("halt_reason") != "L4_ATTRIB_QUARANTINE":
            return out
        # L4_ATTRIB_QUARANTINE resumes ONLY when secondary orthogonal readout or explicit retry arrives.
        if not (has_orthogonal_turn2 or has_retry_turn2):
            return out

    strict_mode = bool(
        strict_evidence_required or _as_bool(out.get("strict_evidence_required"), False)
    )
    has_evidence = _packet_has_sensor_or_exec_evidence(packet)
    has_context = _packet_has_candidate_context(packet)

    # Reject readout-free packets when strict_evidence_required=True OR when packet has
    # neither sensor/execution evidence nor any candidate/pipeline context (e.g. `{}` or `{"run_id": ...}`).
    if (strict_mode and not has_evidence) or (not has_evidence and not has_context):
        out["halted"] = True
        out["halt_reason"] = "L4_ATTRIB_QUARANTINE"
        missing_clauses = ["MISSING_EXPERIMENTAL_EVIDENCE", "LOW_ASSAY_SNR"]
        out["attribution"] = {
            "outcome": AttributionOutcome.AMBIGUOUS_CHARACTERIZATION.value,
            "action": DiagnosticAction.RUN_ORTHOGONAL_ASSAY.value,
            "admit_to_memory": False,
            "precursor_overlap_permille": 0,
            "spectral_residual_permille": 0,
            "orthogonal_modality": None,
            "resolved_via_orthogonal": False,
            "violated_execution_clauses": [],
            "violated_characterization_clauses": missing_clauses,
            "dual_fault_detected": False,
            "compound_fault": False,
            "fault_scope": "NONE",
            "remediation_steps": ["RUN_ORTHOGONAL_ASSAY"],
            "diagnostic_notes": ["MISSING_EXPERIMENTAL_EVIDENCE"],
            "ledger_entry_sha256": None,
            "lean_theorem_tag": "LabAttribLean.unidentifiable_spectrum_never_certified",
        }
        out["trace"].append(
            {
                "repo": REPO_ID,
                "layer": LAYER,
                "status": "QUARANTINE",
                "summary": (
                    "outcome=AMBIGUOUS_CHARACTERIZATION, "
                    "clauses=MISSING_EXPERIMENTAL_EVIDENCE,LOW_ASSAY_SNR"
                ),
            }
        )
        out["bottom_reasoning_traces"].append(
            {
                "repo": REPO_ID,
                "layer": LAYER,
                "invariant": WIRE_INVARIANT,
                "lean_theorem_tag": "LabAttribLean.unidentifiable_spectrum_never_certified",
                "first_principles": (
                    "Readout-free packets lacking wet-lab execution receipts or analytical "
                    "characterization cannot be promoted to certified scientific memory."
                ),
            }
        )
        return out

    edge_raw = out.get("edge_runtime")
    edge: Mapping[str, Any] = edge_raw if isinstance(edge_raw, Mapping) else {}
    stepbot_raw = out.get("stepbot_execution")
    stepbot_exec: Mapping[str, Any] = (
        stepbot_raw if isinstance(stepbot_raw, Mapping) else {}
    )
    override_raw = out.get("telemetry_override")
    telemetry_override: Mapping[str, Any] = (
        override_raw if isinstance(override_raw, Mapping) else {}
    )

    deconv: DeconvolutionResult | None = None
    xrd_mix = out.get("xrd_mixture")
    if isinstance(xrd_mix, Mapping) and xrd_mix:
        observed = xrd_mix.get("observed", xrd_mix.get("observed_y"))
        target_basis = xrd_mix.get("target_basis", xrd_mix.get("target_spectrum"))
        precursor_bases_raw = xrd_mix.get(
            "precursor_bases", xrd_mix.get("precursor_spectra", [])
        )
        if observed is None or target_basis is None:
            raise ValueError(
                "xrd_mixture must provide both 'observed' and 'target_basis' spectra"
            )
        precursor_bases = list(precursor_bases_raw)
        if precursor_bases and isinstance(precursor_bases[0], (int, float)):
            precursor_bases = [precursor_bases]
        deconv = deconvolve_spectrum(
            observed,
            precursor_bases,
            target_basis,
            min_hit_amplitude=float(xrd_mix.get("min_hit_amplitude", 0.20)),
            noise_sigma=float(xrd_mix.get("noise_sigma", 0.025)),
            weights=xrd_mix.get("weights"),
            two_theta_grid=xrd_mix.get("two_theta_grid"),
            max_peak_shift_deg=float(xrd_mix.get("max_peak_shift_deg", 0.0)),
            allow_peak_shift_bins=int(xrd_mix.get("allow_peak_shift_bins", 0)),
            max_strain_scale=float(xrd_mix.get("max_strain_scale", 0.0)),
            background_poly_degree=xrd_mix.get("background_poly_degree"),
            use_numpy=xrd_mix.get("use_numpy"),
        )
        precursor_overlap_permille = deconv.precursor_overlap_permille
        spectral_residual_permille = deconv.spectral_residual_permille
    else:
        precursor_overlap_permille = _nonneg_int(
            telemetry_override.get("precursor_overlap_permille"), 35
        )
        spectral_residual_permille = _nonneg_int(
            telemetry_override.get("spectral_residual_permille"), 42
        )

    dispense_default = _nonneg_int(
        edge.get(
            "dispense_error_permille",
            stepbot_exec.get("dispense_error_permille", out.get("dispense_error_permille")),
        ),
        12,
    )
    dispense_error_permille = _nonneg_int(
        telemetry_override.get("dispense_error_permille"),
        dispense_default,
    )
    aspiration_clean = _as_bool(
        telemetry_override.get(
            "aspiration_clean",
            edge.get("aspiration_clean", stepbot_exec.get("aspiration_clean", True)),
        ),
        True,
    )
    incubation_window_clean = _as_bool(
        telemetry_override.get(
            "incubation_window_clean",
            edge.get(
                "incubation_window_clean",
                stepbot_exec.get("incubation_window_clean", True),
            ),
        ),
        True,
    )
    seal_intact = _as_bool(
        telemetry_override.get(
            "seal_intact",
            edge.get("seal_intact", stepbot_exec.get("seal_intact", True)),
        ),
        True,
    )
    positive_control_passed = _as_bool(
        telemetry_override.get(
            "positive_control_passed", edge.get("positive_control_passed", True)
        ),
        True,
    )
    blank_uncontaminated = _as_bool(
        telemetry_override.get(
            "blank_uncontaminated", edge.get("blank_uncontaminated", True)
        ),
        True,
    )
    detector_calibrated = _as_bool(
        telemetry_override.get(
            "detector_calibrated", edge.get("detector_calibrated", True)
        ),
        True,
    )
    pos_ctrl_recovery = _nonneg_int(
        telemetry_override.get(
            "positive_control_recovery_permille",
            edge.get("positive_control_recovery_permille"),
        ),
        1000,
    )
    plate_z_prime = _signed_int(
        telemetry_override.get("plate_z_prime_tenths", edge.get("plate_z_prime_tenths")),
        70,
    )
    pos_ctrl_zscore = _signed_int(
        telemetry_override.get(
            "positive_control_zscore_tenths", edge.get("positive_control_zscore_tenths")
        ),
        50,
    )

    default_snr = (
        (deconv.snr_tenths if deconv.converged else 0) if deconv is not None else 85
    )
    snr_tenths = _nonneg_int(telemetry_override.get("snr_tenths"), default_snr)
    default_signal = deconv.target_signal_present if deconv is not None else True
    target_signal_present = _as_bool(
        telemetry_override.get("target_signal_present", default_signal),
        default_signal,
    )

    telemetry = AssayTelemetry(
        dispense_error_permille=dispense_error_permille,
        aspiration_clean=aspiration_clean,
        incubation_window_clean=incubation_window_clean,
        seal_intact=seal_intact,
        positive_control_passed=positive_control_passed,
        blank_uncontaminated=blank_uncontaminated,
        detector_calibrated=detector_calibrated,
        precursor_overlap_permille=precursor_overlap_permille,
        spectral_residual_permille=spectral_residual_permille,
        snr_tenths=snr_tenths,
        target_signal_present=target_signal_present,
        positive_control_recovery_permille=pos_ctrl_recovery,
        plate_z_prime_tenths=plate_z_prime,
        positive_control_zscore_tenths=pos_ctrl_zscore,
    )

    secondary = (
        OrthogonalAssayReadout.from_dict(dict(orth_raw))
        if has_orthogonal_turn2 and isinstance(orth_raw, Mapping)
        else None
    )

    report = diagnose_assay(telemetry, secondary)
    outcome = report.outcome
    action = report.action
    admit_to_memory = report.admit_to_memory

    chosen_modality: str | None = None
    if report.resolved_via_orthogonal and secondary is not None:
        chosen_modality = secondary.modality.value
    elif outcome is AttributionOutcome.AMBIGUOUS_CHARACTERIZATION:
        sample_phase = str(
            out.get(
                "sample_phase",
                xrd_mix.get("sample_phase", "solution")
                if isinstance(xrd_mix, Mapping)
                else "solution",
            )
        )
        if sample_phase not in ("solid", "solution"):
            sample_phase = "solution"
        available_vol = float(out.get("available_volume_ul", 8.0))
        prior_prob = min(1.0, max(0.0, float(out.get("prior_hit_prob", 0.25))))
        blindspot_tags = out.get("shared_blindspot_tags", ())
        routing = route_orthogonal_assay(
            telemetry,
            prior_hit_prob=prior_prob,
            available_volume_ul=available_vol,
            sample_phase=sample_phase,
            centrosymmetric_crystal=bool(out.get("centrosymmetric_crystal", False)),
            high_fluorescence_risk=bool(out.get("high_fluorescence_risk", False)),
            analyte_concentration_um=(
                float(out["analyte_concentration_um"])
                if out.get("analyte_concentration_um") is not None
                else None
            ),
            shared_blindspot_tags=(
                list(blindspot_tags)
                if isinstance(blindspot_tags, (list, tuple, set))
                else ()
            ),
        )
        chosen_modality = (
            routing.selected_modality.value
            if routing.selected_modality is not None
            else None
        )

    entry_hash: str | None = None
    if admit_to_memory:
        # Un-halt packet if resuming from a Turn-1 L4_ATTRIB_QUARANTINE resolution
        out["halted"] = False
        out["halt_reason"] = None

        raw_run_id = out.get("run_id", "run-001")
        run_id = str(raw_run_id).strip() or "run-001"
        candidate_smiles = str(out.get("candidate_smiles", ""))
        features = _extract_candidate_features(out, candidate_smiles)

        if isinstance(orth_raw, Mapping) and orth_raw.get("measured_value") is not None:
            measured_value = float(orth_raw["measured_value"])
        elif out.get("measured_value") is not None:
            measured_value = float(out["measured_value"])
        elif report.resolved_via_orthogonal:
            measured_value = (
                1.0 if outcome is AttributionOutcome.CERTIFIED_POSITIVE else 0.0
            )
        elif deconv is not None:
            measured_value = float(deconv.target_amplitude)
        else:
            measured_value = (
                1.0 if outcome is AttributionOutcome.CERTIFIED_POSITIVE else 0.0
            )

        if ledger is not None:
            active_ledger = ledger
        else:
            prior_entries = out.get("certified_ledger_entries") or []
            if prior_entries and isinstance(prior_entries[-1], Mapping):
                last_sha = str(prior_entries[-1].get("ledger_entry_sha256") or GENESIS_HASH)
                active_ledger = CertifiedLedger(
                    start_index=len(prior_entries),
                    initial_prev_hash=last_sha,
                )
            else:
                active_ledger = CertifiedLedger()

        committed = active_ledger.append(
            sample_id=run_id,
            candidate_features=features,
            measured_value=measured_value,
            telemetry=telemetry,
            orthogonal_readout=secondary,
        )
        chain_ok, chain_err = active_ledger.verify_chain()
        if not chain_ok:
            raise LedgerIntegrityError(
                f"CertifiedLedger verification failed: {chain_err}"
            )
        entry_hash = committed.entry_hash
        out["certified_ledger_entries"].append(
            {
                "run_id": run_id,
                "candidate_smiles": candidate_smiles,
                "outcome": outcome.value,
                "admit_to_memory": True,
                "ledger_entry_sha256": entry_hash,
            }
        )
    else:
        out["halted"] = True
        out["halt_reason"] = "L4_ATTRIB_QUARANTINE"

    out["attribution"] = {
        "outcome": outcome.value,
        "action": action.value,
        "admit_to_memory": admit_to_memory,
        "precursor_overlap_permille": precursor_overlap_permille,
        "spectral_residual_permille": spectral_residual_permille,
        "orthogonal_modality": chosen_modality,
        "resolved_via_orthogonal": report.resolved_via_orthogonal,
        "violated_execution_clauses": list(report.violated_execution_clauses),
        "violated_characterization_clauses": list(report.violated_characterization_clauses),
        "dual_fault_detected": report.dual_fault_detected,
        "compound_fault": report.compound_fault,
        "fault_scope": report.fault_scope,
        "remediation_steps": list(report.remediation_steps),
        "diagnostic_notes": list(report.diagnostic_notes),
        "ledger_entry_sha256": entry_hash,
        "lean_theorem_tag": report.lean_theorem_tag,
    }

    trace_status = "QUARANTINE" if out["halted"] else "PASS"
    all_clauses = (
        report.violated_execution_clauses + report.violated_characterization_clauses
    )
    clauses_str = ",".join(all_clauses) if all_clauses else "NONE"

    if admit_to_memory and entry_hash is not None:
        orth_note = (
            f", resolved_via={chosen_modality}" if report.resolved_via_orthogonal else ""
        )
        summary = (
            f"outcome={outcome.value}, action={action.value}{orth_note}, "
            f"ledger_sha256={entry_hash[:12]}..."
        )
    elif outcome is AttributionOutcome.AMBIGUOUS_CHARACTERIZATION:
        summary = (
            f"outcome={outcome.value}, clauses={clauses_str}, "
            f"overlap={precursor_overlap_permille} permille, "
            f"rwp={spectral_residual_permille} permille, route={chosen_modality}"
        )
    else:
        dual_tag = ", dual_fault=RECALIBRATE_AND_RETRY" if report.dual_fault_detected else ""
        summary = (
            f"outcome={outcome.value}, action={action.value}, "
            f"fault_scope={report.fault_scope}, clauses={clauses_str}{dual_tag}"
        )

    out["trace"].append(
        {
            "repo": REPO_ID,
            "layer": LAYER,
            "status": trace_status,
            "summary": summary,
        }
    )
    out["bottom_reasoning_traces"].append(
        {
            "repo": REPO_ID,
            "layer": LAYER,
            "invariant": WIRE_INVARIANT,
            "lean_theorem_tag": report.lean_theorem_tag,
            "first_principles": (
                "Post-execution observations conflate liquid-handling/incubation artifacts "
                "and unreacted precursor diffraction ghost peaks with biochemical truth; "
                "NNLS multiphase deconvolution and 4-way Lean causal blame quarantine "
                "non-identifiable or corrupted runs before SHA-256 ledger admission."
            ),
        }
    )
    return out
