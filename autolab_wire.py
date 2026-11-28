"""Auto-Lab 15-repository wire contract node (`autolab-wire/v1`) for `lab-attrib` (Layer L4).

Post-execution causal blame gate, NNLS precursor ghost-peak deconvolver, Bayesian EVOI
orthogonal assay router, and SHA-256 proof-carrying `CertifiedLedger` firewall sitting
between wet-lab execution (`stepbot`, `lab-feet`) and downstream learning/recalibration
(`twinstep`, `labbench-posttrain`, `lab-decide`).

Fail-Closed Contract:
- Default-deny evidence gate: packets lacking explicit wet-lab execution QC telemetry
  (`dispense_error_permille`, `aspiration_clean`, `incubation_window_clean`, `seal_intact` from
  measured `edge_runtime`, `execution_qc`, `stepbot_execution`, or `telemetry_override`) OR
  lacking explicit analytical characterization (`xrd_mixture`, or `characterization` / `readout` /
  `telemetry_override` with `snr_tenths` + `target_signal_present`) never default to clean
  `snr_tenths = 85` or `target_signal_present = True`; they quarantine with `halted = True`,
  `halt_reason = "L4_ATTRIB_QUARANTINE"`, `admit_to_memory = False`, `certified_ledger_entries = []`,
  and `"MISSING_EXPERIMENTAL_EVIDENCE"` unless `allow_default_telemetry=True` is explicitly opted in.
- Malformed, non-boolean, non-integer, negative, or non-finite (`NaN`/`+-Inf`) telemetry
  and envelope fields never silently default to clean execution and never raise unhandled
  `ValueError`/`TypeError`; they fail closed with `halted = True` and
  `halt_reason = "L4_ATTRIB_SCHEMA_REJECT"` (`admit_to_memory = False`).
"""

from __future__ import annotations

import copy
import math
import numbers
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ID: str = "lab-attrib"
LAYER: str = "L4"
SCHEMA_VERSION: str = "autolab-wire/v1"
SCHEMA_REJECT_REASON: str = "L4_ATTRIB_SCHEMA_REJECT"
QUARANTINE_REASON: str = "L4_ATTRIB_QUARANTINE"
REPO_DIR: Path = Path(__file__).resolve().parent

_SRC_DIR = REPO_DIR / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from lab_attrib.blame import (  # noqa: E402
    AssayTelemetry,
    AttributionOutcome,
    DiagnosticAction,
    OrthogonalAssayReadout,
    SpectralFitMethod,
    diagnose_assay,
)
from lab_attrib.ledger import (  # noqa: E402
    GENESIS_HASH,
    CertifiedLedger,
    LedgerIntegrityError,
    LedgerRejectionError,
)
from lab_attrib.orthogonal_router import route_orthogonal_assay  # noqa: E402
from lab_attrib.xrd_deconvolve import DeconvolutionResult, deconvolve_spectrum  # noqa: E402

WIRE_INVARIANT: str = (
    "corrupted_run_never_poisons_memory & unidentifiable_spectrum_never_certified"
)


class _SchemaRejectError(Exception):
    """Internal fail-closed wire schema error converted to `L4_ATTRIB_SCHEMA_REJECT`."""


def _is_strict_bool(value: Any) -> bool:
    return isinstance(value, bool) or type(value).__name__ == "bool_"


def _strict_nonneg_int(value: Any, default: int, field: str) -> int:
    if value is None:
        return default
    if _is_strict_bool(value) or not isinstance(value, numbers.Integral):
        raise _SchemaRejectError(
            f"Field {field!r} must be a non-negative integer (bool/float/str forbidden), got {value!r}"
        )
    ival = int(value)
    if ival < 0:
        raise _SchemaRejectError(f"Field {field!r} must be >= 0, got {ival}")
    return ival


def _strict_signed_int(value: Any, default: int, field: str) -> int:
    if value is None:
        return default
    if _is_strict_bool(value) or not isinstance(value, numbers.Integral):
        raise _SchemaRejectError(
            f"Field {field!r} must be a signed integer (bool/float/str forbidden), got {value!r}"
        )
    return int(value)


def _strict_bool(value: Any, default: bool, field: str) -> bool:
    if value is None:
        return default
    if not _is_strict_bool(value):
        raise _SchemaRejectError(
            f"Field {field!r} must be a strict bool, got {type(value).__name__} {value!r}"
        )
    return bool(value)


def _strict_finite_float(
    value: Any,
    default: float | None,
    field: str,
    *,
    min_val: float | None = None,
    max_val: float | None = None,
    positive: bool = False,
) -> float | None:
    if value is None:
        return default
    if _is_strict_bool(value) or not isinstance(value, (int, float, numbers.Real)):
        raise _SchemaRejectError(
            f"Field {field!r} must be a finite real number, got {value!r}"
        )
    fval = float(value)
    if not math.isfinite(fval):
        raise _SchemaRejectError(f"Field {field!r} must be finite (no NaN/Inf), got {value!r}")
    if positive and fval <= 0.0:
        raise _SchemaRejectError(f"Field {field!r} must be > 0, got {fval}")
    if min_val is not None and fval < min_val:
        raise _SchemaRejectError(f"Field {field!r} must be >= {min_val}, got {fval}")
    if max_val is not None and fval > max_val:
        raise _SchemaRejectError(f"Field {field!r} must be <= {max_val}, got {fval}")
    return fval


def _strict_submapping(packet: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    if key not in packet or packet[key] is None:
        return {}
    raw = packet[key]
    if not isinstance(raw, Mapping):
        raise _SchemaRejectError(
            f"Envelope field {key!r} must be a mapping/dict, got {type(raw).__name__}"
        )
    return raw


def _resolve_layered_nonneg_int_tracked(
    sources: list[tuple[str, Mapping[str, Any]]],
    key: str,
    default: int,
) -> tuple[int, bool]:
    chosen: int | None = None
    found = False
    for prefix, mapping in sources:
        if key in mapping and mapping[key] is not None:
            val = _strict_nonneg_int(mapping[key], default, f"{prefix}.{key}")
            if not found:
                chosen = val
                found = True
    return (default if not found or chosen is None else chosen), found


def _resolve_layered_nonneg_int(
    sources: list[tuple[str, Mapping[str, Any]]],
    key: str,
    default: int,
) -> int:
    val, _ = _resolve_layered_nonneg_int_tracked(sources, key, default)
    return val


def _resolve_layered_signed_int(
    sources: list[tuple[str, Mapping[str, Any]]],
    key: str,
    default: int,
) -> int:
    chosen: int | None = None
    for prefix, mapping in sources:
        if key in mapping and mapping[key] is not None:
            val = _strict_signed_int(mapping[key], default, f"{prefix}.{key}")
            if chosen is None:
                chosen = val
    return default if chosen is None else chosen


def _resolve_layered_bool_tracked(
    sources: list[tuple[str, Mapping[str, Any]]],
    key: str,
    default: bool,
) -> tuple[bool, bool]:
    chosen: bool | None = None
    found = False
    for prefix, mapping in sources:
        if key in mapping and mapping[key] is not None:
            val = _strict_bool(mapping[key], default, f"{prefix}.{key}")
            if not found:
                chosen = val
                found = True
    return (default if not found or chosen is None else chosen), found


def _resolve_layered_bool(
    sources: list[tuple[str, Mapping[str, Any]]],
    key: str,
    default: bool,
) -> bool:
    val, _ = _resolve_layered_bool_tracked(sources, key, default)
    return val


def _extract_candidate_features(
    packet: Mapping[str, Any], candidate_smiles: str
) -> dict[str, float]:
    raw_features = _strict_submapping(packet, "candidate_features")
    if raw_features:
        out_feats: dict[str, float] = {}
        for k, v in raw_features.items():
            fval = _strict_finite_float(v, None, f"candidate_features[{k!r}]")
            assert fval is not None
            out_feats[str(k)] = fval
        return out_feats

    parsed_spans = _strict_submapping(packet, "parsed_spans")
    vol_ul = _strict_finite_float(
        parsed_spans.get("volume_ul"), 100.0, "parsed_spans.volume_ul", positive=True
    )
    knn_dist = _strict_finite_float(
        packet.get("knn_distance"), 0.412, "knn_distance", min_val=0.0
    )
    assert vol_ul is not None and knn_dist is not None
    return {
        "smiles_length": float(len(candidate_smiles)),
        "knn_distance": knn_dist,
        "volume_ul": vol_ul,
    }


def _edge_runtime_has_real_interlocks(edge: Mapping[str, Any]) -> bool:
    """Return False when `lab-feet` filled `edge_runtime` interlocks from passthrough defaults."""
    if not edge:
        return False
    if edge.get("interlocks_source") == "packet_passthrough":
        return False
    if edge.get("sensor_source") == "derived_from_protocol_hash":
        return False
    return True


def _validate_candidate_smiles_field(packet: Mapping[str, Any]) -> str:
    raw_smiles = packet.get("candidate_smiles")
    if raw_smiles is not None and not isinstance(raw_smiles, str):
        raise _SchemaRejectError(
            f"Field 'candidate_smiles' must be a str, got {type(raw_smiles).__name__}"
        )
    return str(raw_smiles or "")


def _emit_schema_reject(out: dict[str, Any], reason: str) -> dict[str, Any]:
    """Populate a fail-closed `L4_ATTRIB_SCHEMA_REJECT` packet without admitting to memory."""
    out["halted"] = True
    out["halt_reason"] = SCHEMA_REJECT_REASON
    if not isinstance(out.get("trace"), list):
        out["trace"] = []
    if not isinstance(out.get("bottom_reasoning_traces"), list):
        out["bottom_reasoning_traces"] = []
    if not isinstance(out.get("certified_ledger_entries"), list):
        out["certified_ledger_entries"] = []

    out["attribution"] = {
        "outcome": AttributionOutcome.AMBIGUOUS_CHARACTERIZATION.value,
        "action": DiagnosticAction.RUN_ORTHOGONAL_ASSAY.value,
        "admit_to_memory": False,
        "defaulted_fields": [],
        "demo_default_telemetry_used": False,
        "schema_valid": False,
        "schema_error": reason,
        "precursor_overlap_permille": 0,
        "spectral_residual_permille": 0,
        "orthogonal_modality": None,
        "resolved_via_orthogonal": False,
        "violated_execution_clauses": ["MALFORMED_TELEMETRY_SCHEMA"],
        "violated_characterization_clauses": ["MALFORMED_TELEMETRY_SCHEMA"],
        "dual_fault_detected": False,
        "compound_fault": False,
        "fault_scope": "SCHEMA_REJECT",
        "remediation_steps": ["FIX_TELEMETRY_PACKET_SCHEMA"],
        "diagnostic_notes": [reason],
        "ledger_entry_sha256": None,
        "lean_theorem_tag": "LabAttribLean.unidentifiable_spectrum_never_certified",
    }
    out["trace"].append(
        {
            "repo": REPO_ID,
            "layer": LAYER,
            "status": "REJECT",
            "summary": f"schema_reject={SCHEMA_REJECT_REASON}: {reason}",
        }
    )
    out["bottom_reasoning_traces"].append(
        {
            "repo": REPO_ID,
            "layer": LAYER,
            "invariant": WIRE_INVARIANT,
            "lean_theorem_tag": "LabAttribLean.unidentifiable_spectrum_never_certified",
            "first_principles": (
                "Malformed, non-boolean, non-integer, or non-finite post-execution telemetry "
                "packets fail closed before attribution or SHA-256 ledger admission."
            ),
        }
    )
    return out


def process_packet(
    packet: dict[str, Any],
    *,
    ledger: CertifiedLedger | None = None,
    strict_evidence_required: bool = True,
    allow_default_telemetry: bool = False,
) -> dict[str, Any]:
    """Execute the `L4` `lab-attrib` post-execution verification gate on an `autolab-wire/v1` packet."""
    if not isinstance(packet, Mapping):
        return _emit_schema_reject(
            {"schema_version": SCHEMA_VERSION},
            f"Packet must be a mapping/dict, got {type(packet).__name__}",
        )

    out: dict[str, Any] = copy.deepcopy(dict(packet))
    if out.get("schema_version") is None:
        out["schema_version"] = SCHEMA_VERSION
    if "halt_reason" not in out:
        out["halt_reason"] = None

    try:
        for list_key in ("trace", "bottom_reasoning_traces", "certified_ledger_entries"):
            if out.get(list_key) is None:
                out[list_key] = []
            elif not isinstance(out[list_key], list):
                raise _SchemaRejectError(
                    f"Envelope field {list_key!r} must be a list, got {type(out[list_key]).__name__}"
                )

        if out.get("halted") is None:
            out["halted"] = False
        elif not _is_strict_bool(out["halted"]):
            raise _SchemaRejectError(
                f"Envelope field 'halted' must be a strict bool, got {out['halted']!r}"
            )
        else:
            out["halted"] = bool(out["halted"])

        # Upstream halts (e.g., L4_REHEARSAL_BLOCK, L6_ABSTAIN, L1_MCP_BLOCK) ALWAYS pass through untouched.
        if out["halted"] and out.get("halt_reason") not in (
            QUARANTINE_REASON,
            SCHEMA_REJECT_REASON,
        ):
            return out

        orth_raw = _strict_submapping(out, "orthogonal_readout")
        has_orthogonal_turn2 = bool(orth_raw)
        has_retry_turn2 = _strict_nonneg_int(out.get("retry_attempt"), 0, "retry_attempt") > 0

        if out["halted"]:
            if out.get("halt_reason") == SCHEMA_REJECT_REASON:
                return out
            # L4_ATTRIB_QUARANTINE resumes ONLY when secondary orthogonal readout or explicit retry arrives.
            if not (has_orthogonal_turn2 or has_retry_turn2):
                return out

        if not _is_strict_bool(strict_evidence_required):
            raise _SchemaRejectError("strict_evidence_required argument must be a bool")
        _strict_bool(out.get("strict_evidence_required"), True, "strict_evidence_required")

        if not _is_strict_bool(allow_default_telemetry):
            raise _SchemaRejectError("allow_default_telemetry argument must be a bool")
        pkt_allow_defaults = _strict_bool(
            out.get("allow_default_telemetry"), False, "allow_default_telemetry"
        )
        allow_defaults = bool(allow_default_telemetry or pkt_allow_defaults)

        edge = _strict_submapping(out, "edge_runtime")
        stepbot_exec = _strict_submapping(out, "stepbot_execution")
        telemetry_override = _strict_submapping(out, "telemetry_override")
        execution_qc = _strict_submapping(out, "execution_qc")
        exec_receipt = _strict_submapping(out, "execution_receipt")
        xrd_mix = _strict_submapping(out, "xrd_mixture")
        char_map = _strict_submapping(out, "characterization")
        readout_map = _strict_submapping(out, "readout")
        _validate_candidate_smiles_field(out)

        # Validate routing/envelope fields upfront on every active packet so malformed fields
        # NEVER bypass validation even when sensor/readout telemetry is incomplete.
        raw_phase = out.get("sample_phase", xrd_mix.get("sample_phase", "solution"))
        if not isinstance(raw_phase, str) or raw_phase not in ("solid", "solution"):
            raise _SchemaRejectError(
                f"Field 'sample_phase' must be 'solid' or 'solution', got {raw_phase!r}"
            )
        available_vol = _strict_finite_float(
            out.get("available_volume_ul"), 8.0, "available_volume_ul", positive=True
        )
        prior_prob = _strict_finite_float(
            out.get("prior_hit_prob"), 0.25, "prior_hit_prob", min_val=0.0, max_val=1.0
        )
        centro = _strict_bool(
            out.get("centrosymmetric_crystal"), False, "centrosymmetric_crystal"
        )
        fluor = _strict_bool(
            out.get("high_fluorescence_risk"), False, "high_fluorescence_risk"
        )
        conc_um = _strict_finite_float(
            out.get("analyte_concentration_um"),
            None,
            "analyte_concentration_um",
            min_val=0.0,
        )
        blindspot_tags = out.get("shared_blindspot_tags", ())
        if blindspot_tags is not None and (
            isinstance(blindspot_tags, (str, bytes))
            or not isinstance(blindspot_tags, (list, tuple, set))
        ):
            raise _SchemaRejectError(
                "Field 'shared_blindspot_tags' must be a list/tuple/set of strings"
            )
        assert available_vol is not None and prior_prob is not None

        # Validate measured_value upfront so NaN/bool never slips by even on quarantined packets
        orth_mv = _strict_finite_float(
            orth_raw.get("measured_value"), None, "orthogonal_readout.measured_value"
        )
        pkt_mv = _strict_finite_float(out.get("measured_value"), None, "measured_value")

        defaulted_fields: list[str] = []

        deconv: DeconvolutionResult | None = None
        if xrd_mix:
            observed = xrd_mix.get("observed", xrd_mix.get("observed_y"))
            target_basis = xrd_mix.get("target_basis", xrd_mix.get("target_spectrum"))
            precursor_bases_raw = xrd_mix.get(
                "precursor_bases", xrd_mix.get("precursor_spectra", [])
            )
            if observed is None or target_basis is None:
                raise _SchemaRejectError(
                    "xrd_mixture must provide both 'observed' and 'target_basis' spectra"
                )
            if isinstance(observed, (str, bytes)) or not isinstance(observed, (list, tuple)):
                raise _SchemaRejectError("xrd_mixture.observed must be a numeric list/tuple")
            if isinstance(target_basis, (str, bytes)) or not isinstance(target_basis, (list, tuple)):
                raise _SchemaRejectError("xrd_mixture.target_basis must be a numeric list/tuple")
            if isinstance(precursor_bases_raw, (str, bytes)) or not isinstance(
                precursor_bases_raw, (list, tuple)
            ):
                raise _SchemaRejectError("xrd_mixture.precursor_bases must be a list/tuple")

            precursor_bases = list(precursor_bases_raw)
            if (
                precursor_bases
                and isinstance(precursor_bases[0], (int, float))
                and not _is_strict_bool(precursor_bases[0])
            ):
                precursor_bases = [precursor_bases]

            min_hit_amp = _strict_finite_float(
                xrd_mix.get("min_hit_amplitude"), 0.20, "xrd_mixture.min_hit_amplitude", positive=True
            )
            noise_sig = _strict_finite_float(
                xrd_mix.get("noise_sigma"), 0.025, "xrd_mixture.noise_sigma", positive=True
            )
            max_shift_deg = _strict_finite_float(
                xrd_mix.get("max_peak_shift_deg"), 0.0, "xrd_mixture.max_peak_shift_deg", min_val=0.0
            )
            shift_bins = _strict_nonneg_int(
                xrd_mix.get("allow_peak_shift_bins"), 0, "xrd_mixture.allow_peak_shift_bins"
            )
            max_strain = _strict_finite_float(
                xrd_mix.get("max_strain_scale"), 0.0, "xrd_mixture.max_strain_scale", min_val=0.0
            )
            l1_reg = _strict_finite_float(
                xrd_mix.get("l1_alpha"), 0.0, "xrd_mixture.l1_alpha", min_val=0.0
            )
            pv_fwhm = _strict_finite_float(
                xrd_mix.get("pseudo_voigt_fwhm_add"),
                0.0,
                "xrd_mixture.pseudo_voigt_fwhm_add",
                min_val=0.0,
            )
            pv_eta = _strict_finite_float(
                xrd_mix.get("pseudo_voigt_eta"),
                0.5,
                "xrd_mixture.pseudo_voigt_eta",
                min_val=0.0,
                max_val=1.0,
            )
            shift_mode = str(xrd_mix.get("shift_search_mode", "dictionary"))
            bg_deg_raw = xrd_mix.get("background_poly_degree")
            bg_deg = (
                _strict_nonneg_int(bg_deg_raw, 0, "xrd_mixture.background_poly_degree")
                if bg_deg_raw is not None
                else None
            )
            use_np_raw = xrd_mix.get("use_numpy")
            use_np = (
                _strict_bool(use_np_raw, False, "xrd_mixture.use_numpy")
                if use_np_raw is not None
                else None
            )
            assert (
                min_hit_amp is not None
                and noise_sig is not None
                and max_shift_deg is not None
                and max_strain is not None
                and l1_reg is not None
                and pv_fwhm is not None
                and pv_eta is not None
            )
            try:
                deconv = deconvolve_spectrum(
                    observed,
                    precursor_bases,
                    target_basis,
                    min_hit_amplitude=min_hit_amp,
                    noise_sigma=noise_sig,
                    weights=xrd_mix.get("weights"),
                    two_theta_grid=xrd_mix.get("two_theta_grid"),
                    max_peak_shift_deg=max_shift_deg,
                    allow_peak_shift_bins=shift_bins,
                    max_strain_scale=max_strain,
                    background_poly_degree=bg_deg,
                    l1_alpha=l1_reg,
                    pseudo_voigt_fwhm_add=pv_fwhm,
                    pseudo_voigt_eta=pv_eta,
                    shift_search_mode=shift_mode,
                    use_numpy=use_np,
                )
            except (ValueError, TypeError, RuntimeError) as exc:
                raise _SchemaRejectError(f"Invalid xrd_mixture payload: {exc}") from exc

        char_sources: list[tuple[str, Mapping[str, Any]]] = [
            ("telemetry_override", telemetry_override),
            ("characterization", char_map),
            ("readout", readout_map),
            ("edge_runtime", edge),
        ]

        # Validate all layered execution sources (including edge_runtime even if passthrough-defaulted)
        all_exec_sources: list[tuple[str, Mapping[str, Any]]] = [
            ("telemetry_override", telemetry_override),
            ("execution_qc", execution_qc),
            ("execution_receipt", exec_receipt),
            ("edge_runtime", edge),
            ("stepbot_execution", stepbot_exec),
            ("packet", out),
        ]
        # Sources that constitute genuine non-defaulted execution QC evidence
        explicit_exec_sources: list[tuple[str, Mapping[str, Any]]] = [
            ("telemetry_override", telemetry_override),
            ("execution_qc", execution_qc),
            ("execution_receipt", exec_receipt),
            *(
                [("edge_runtime", edge)]
                if _edge_runtime_has_real_interlocks(edge)
                else []
            ),
            ("stepbot_execution", stepbot_exec),
            ("packet", out),
        ]

        dispense_error_permille = _resolve_layered_nonneg_int(
            all_exec_sources, "dispense_error_permille", 12
        )
        _, has_disp = _resolve_layered_nonneg_int_tracked(
            explicit_exec_sources, "dispense_error_permille", 12
        )
        if not has_disp:
            defaulted_fields.append("dispense_error_permille")

        aspiration_clean = _resolve_layered_bool(all_exec_sources, "aspiration_clean", True)
        _, has_asp = _resolve_layered_bool_tracked(
            explicit_exec_sources, "aspiration_clean", True
        )
        if not has_asp:
            defaulted_fields.append("aspiration_clean")

        incubation_window_clean = _resolve_layered_bool(
            all_exec_sources, "incubation_window_clean", True
        )
        _, has_inc = _resolve_layered_bool_tracked(
            explicit_exec_sources, "incubation_window_clean", True
        )
        if not has_inc:
            defaulted_fields.append("incubation_window_clean")

        seal_intact = _resolve_layered_bool(all_exec_sources, "seal_intact", True)
        _, has_seal = _resolve_layered_bool_tracked(explicit_exec_sources, "seal_intact", True)
        if not has_seal:
            defaulted_fields.append("seal_intact")

        control_and_char_sources: list[tuple[str, Mapping[str, Any]]] = [
            ("telemetry_override", telemetry_override),
            ("characterization", char_map),
            ("readout", readout_map),
            ("execution_qc", execution_qc),
            ("execution_receipt", exec_receipt),
            ("edge_runtime", edge),
            ("stepbot_execution", stepbot_exec),
            ("packet", out),
        ]
        positive_control_passed = _resolve_layered_bool(
            control_and_char_sources, "positive_control_passed", True
        )
        blank_uncontaminated = _resolve_layered_bool(
            control_and_char_sources, "blank_uncontaminated", True
        )
        detector_calibrated = _resolve_layered_bool(
            control_and_char_sources, "detector_calibrated", True
        )
        pos_ctrl_recovery = _resolve_layered_nonneg_int(
            control_and_char_sources, "positive_control_recovery_permille", 1000
        )
        plate_z_prime = _resolve_layered_signed_int(
            control_and_char_sources, "plate_z_prime_tenths", 70
        )
        pos_ctrl_zscore = _resolve_layered_signed_int(
            control_and_char_sources, "positive_control_zscore_tenths", 50
        )

        has_explicit_char_metric = False
        if deconv is not None:
            has_explicit_char_metric = True
            precursor_overlap_permille = _resolve_layered_nonneg_int(
                char_sources, "precursor_overlap_permille", deconv.precursor_overlap_permille
            )
            spectral_residual_permille = _resolve_layered_nonneg_int(
                char_sources, "spectral_residual_permille", deconv.spectral_residual_permille
            )
            default_snr = deconv.snr_tenths if deconv.converged else 0
            snr_tenths = _resolve_layered_nonneg_int(char_sources, "snr_tenths", default_snr)
            target_signal_present = _resolve_layered_bool(
                char_sources, "target_signal_present", deconv.target_signal_present
            )
        else:
            snr_tenths, has_snr = _resolve_layered_nonneg_int_tracked(
                char_sources, "snr_tenths", 85 if allow_defaults else 0
            )
            target_signal_present, has_sig = _resolve_layered_bool_tracked(
                char_sources, "target_signal_present", True if allow_defaults else False
            )
            if not has_snr:
                defaulted_fields.append("snr_tenths")
            if not has_sig:
                defaulted_fields.append("target_signal_present")

            has_explicit_readout_pair = has_snr and has_sig
            fallback_overlap = 35 if allow_defaults else 0
            fallback_rwp = 42 if allow_defaults else 0
            precursor_overlap_permille, has_ov = _resolve_layered_nonneg_int_tracked(
                char_sources, "precursor_overlap_permille", fallback_overlap
            )
            spectral_residual_permille, has_rwp = _resolve_layered_nonneg_int_tracked(
                char_sources, "spectral_residual_permille", fallback_rwp
            )
            if not has_explicit_readout_pair:
                if not has_ov:
                    defaulted_fields.append("precursor_overlap_permille")
                if not has_rwp:
                    defaulted_fields.append("spectral_residual_permille")
            has_explicit_char_metric = bool(has_snr or has_sig or has_ov or has_rwp)

        raw_fit_method = None
        for prefix, src_map in [
            ("telemetry_override", telemetry_override),
            ("characterization", char_map),
            ("readout", readout_map),
            ("xrd_mixture", xrd_mix),
            ("packet", out),
        ]:
            if "fit_method" in src_map and src_map["fit_method"] is not None:
                cand_fit = src_map["fit_method"]
                if not isinstance(cand_fit, str) or cand_fit not in ("rietveld", "stick_nnls"):
                    raise _SchemaRejectError(
                        f"Field '{prefix}.fit_method' must be 'rietveld' or 'stick_nnls', got {cand_fit!r}"
                    )
                raw_fit_method = cand_fit
                break
        resolved_fit_method = SpectralFitMethod(
            raw_fit_method
            if raw_fit_method is not None
            else ("stick_nnls" if deconv is not None else "rietveld")
        )

        try:
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
                fit_method=resolved_fit_method,
            )
        except (ValueError, TypeError) as exc:
            raise _SchemaRejectError(f"Invalid AssayTelemetry fields: {exc}") from exc

        secondary: OrthogonalAssayReadout | None = None
        if has_orthogonal_turn2:
            try:
                secondary = OrthogonalAssayReadout.from_dict(dict(orth_raw))
            except (KeyError, ValueError, TypeError) as exc:
                raise _SchemaRejectError(
                    f"Invalid orthogonal_readout payload: {exc}"
                ) from exc

        report = diagnose_assay(telemetry, secondary)
        outcome = report.outcome
        action = report.action
        admit_to_memory = report.admit_to_memory
        exec_clauses = list(report.violated_execution_clauses)
        char_clauses = list(report.violated_characterization_clauses)
        remediation_steps = list(report.remediation_steps)
        diagnostic_notes = list(report.diagnostic_notes)
        lean_tag = report.lean_theorem_tag
        resolved_via_orth = report.resolved_via_orthogonal

        # Fail-closed gate: refuse memory/ledger admission whenever required telemetry or
        # characterization fields were filled from defaults (unless allow_default_telemetry=True).
        missing_evidence_blocked = bool(defaulted_fields and not allow_defaults)
        if missing_evidence_blocked:
            admit_to_memory = False
            resolved_via_orth = False
            if "MISSING_EXPERIMENTAL_EVIDENCE" not in diagnostic_notes:
                diagnostic_notes.insert(0, "MISSING_EXPERIMENTAL_EVIDENCE")
            if outcome is not AttributionOutcome.EXECUTION_ARTIFACT:
                outcome = AttributionOutcome.AMBIGUOUS_CHARACTERIZATION
                action = DiagnosticAction.RUN_ORTHOGONAL_ASSAY
                lean_tag = "LabAttribLean.unidentifiable_spectrum_never_certified"
                if "MISSING_EXPERIMENTAL_EVIDENCE" not in char_clauses:
                    char_clauses.insert(0, "MISSING_EXPERIMENTAL_EVIDENCE")
                if "RUN_ORTHOGONAL_ASSAY" not in remediation_steps:
                    remediation_steps = ["RUN_ORTHOGONAL_ASSAY"]

        chosen_modality: str | None = None
        if resolved_via_orth and secondary is not None:
            chosen_modality = secondary.modality.value
        elif (
            report.outcome is AttributionOutcome.AMBIGUOUS_CHARACTERIZATION
            and outcome is AttributionOutcome.AMBIGUOUS_CHARACTERIZATION
            and has_explicit_char_metric
        ):
            routing = route_orthogonal_assay(
                telemetry,
                prior_hit_prob=prior_prob,
                available_volume_ul=available_vol,
                sample_phase=raw_phase,
                centrosymmetric_crystal=centro,
                high_fluorescence_risk=fluor,
                analyte_concentration_um=conc_um,
                shared_blindspot_tags=(
                    [str(t) for t in blindspot_tags] if blindspot_tags is not None else ()
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
            if raw_run_id is not None and _is_strict_bool(raw_run_id):
                raise _SchemaRejectError(f"Field 'run_id' cannot be a bool, got {raw_run_id!r}")
            run_id = str(raw_run_id).strip() if raw_run_id is not None else "run-001"
            if not run_id:
                run_id = "run-001"

            raw_smiles = out.get("candidate_smiles", "")
            if raw_smiles is not None and not isinstance(raw_smiles, str):
                raise _SchemaRejectError(
                    f"Field 'candidate_smiles' must be a str, got {type(raw_smiles).__name__}"
                )
            candidate_smiles = str(raw_smiles or "")
            features = _extract_candidate_features(out, candidate_smiles)

            if orth_mv is not None:
                measured_value = orth_mv
            elif pkt_mv is not None:
                measured_value = pkt_mv
            elif resolved_via_orth:
                measured_value = (
                    1.0 if outcome is AttributionOutcome.CERTIFIED_POSITIVE else 0.0
                )
            elif deconv is not None:
                measured_value = float(deconv.target_amplitude)
            else:
                measured_value = (
                    1.0 if outcome is AttributionOutcome.CERTIFIED_POSITIVE else 0.0
                )

            try:
                if ledger is not None:
                    active_ledger = ledger
                else:
                    prior_entries = out.get("certified_ledger_entries") or []
                    if prior_entries:
                        if not isinstance(prior_entries[-1], Mapping):
                            raise _SchemaRejectError(
                                "certified_ledger_entries[-1] must be a mapping"
                            )
                        last_sha = str(
                            prior_entries[-1].get("ledger_entry_sha256") or GENESIS_HASH
                        )
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
            except (ValueError, TypeError, LedgerRejectionError, LedgerIntegrityError) as exc:
                raise _SchemaRejectError(f"Ledger commitment rejected: {exc}") from exc

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
            out["halt_reason"] = QUARANTINE_REASON

    except _SchemaRejectError as exc:
        return _emit_schema_reject(out, str(exc))

    out["attribution"] = {
        "outcome": outcome.value,
        "action": action.value,
        "admit_to_memory": admit_to_memory,
        "defaulted_fields": defaulted_fields,
        "demo_default_telemetry_used": bool(defaulted_fields and allow_defaults),
        "precursor_overlap_permille": precursor_overlap_permille,
        "spectral_residual_permille": spectral_residual_permille,
        "orthogonal_modality": chosen_modality,
        "resolved_via_orthogonal": resolved_via_orth,
        "violated_execution_clauses": exec_clauses,
        "violated_characterization_clauses": char_clauses,
        "dual_fault_detected": report.dual_fault_detected,
        "compound_fault": report.compound_fault,
        "fault_scope": report.fault_scope,
        "remediation_steps": remediation_steps,
        "diagnostic_notes": diagnostic_notes,
        "ledger_entry_sha256": entry_hash,
        "lean_theorem_tag": lean_tag,
    }

    trace_status = "QUARANTINE" if out["halted"] else "PASS"
    all_clauses = exec_clauses + char_clauses
    clauses_str = ",".join(all_clauses) if all_clauses else "NONE"

    if admit_to_memory and entry_hash is not None:
        orth_note = (
            f", resolved_via={chosen_modality}" if resolved_via_orth else ""
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
            "lean_theorem_tag": lean_tag,
            "first_principles": (
                "Post-execution observations conflate liquid-handling/incubation artifacts "
                "and unreacted precursor diffraction ghost peaks with biochemical truth; "
                "NNLS multiphase deconvolution and 4-way Lean causal blame quarantine "
                "non-identifiable or corrupted runs before SHA-256 ledger admission."
            ),
        }
    )
    return out
