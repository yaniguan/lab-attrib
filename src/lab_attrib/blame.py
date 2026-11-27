"""Exact Python mirror of the Lean 4.36.0 post-execution causal blame calculus (`LabAttribLean`).

Separates post-execution wet-lab outcomes into four mutually exclusive causal verdicts:
1. ``EXECUTION_ARTIFACT`` -> ``RETRY_ROBOT`` (liquid handling, incubation, seal, or control failure)
2. ``AMBIGUOUS_CHARACTERIZATION`` -> ``RUN_ORTHOGONAL_ASSAY`` (detector, precursor ghost-peak overlap, R_wp, or SNR failure)
3. ``CERTIFIED_NEGATIVE`` -> ``COMMIT_CERTIFIED_NEGATIVE`` (clean execution + identifiable absence of target)
4. ``CERTIFIED_POSITIVE`` -> ``COMMIT_CERTIFIED_HIT`` (clean execution + identifiable presence of target)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import numbers
from typing import Any

# Exact integer permille / tenths thresholds mirrored 1-to-1 with `lean/LabAttribLean.lean`
MAX_DISPENSE_ERROR_PERMILLE: int = 50
MAX_PRECURSOR_OVERLAP_PERMILLE: int = 150
MAX_SPECTRAL_RESIDUAL_PERMILLE: int = 100
MIN_RESOLVABLE_SNR_TENTHS: int = 30
MAX_STEPPING_STONE_VOLUME_UL: int = 10
MAX_EXPANSION_ETA_PERMILLE: int = 500
MIN_POS_CONTROL_RECOVERY_PERMILLE: int = 800
MIN_Z_PRIME_TENTHS: int = 50
MIN_POS_CONTROL_ZSCORE_TENTHS: int = 20

_PLATE_CONTROL_FAULT_CODES: frozenset[str] = frozenset(
    {
        "POSITIVE_CONTROL_FAILED",
        "BLANK_CONTAMINATED",
        "POSITIVE_CONTROL_RECOVERY_DRIFT",
        "LOW_PLATE_Z_PRIME_FACTOR",
        "POSITIVE_CONTROL_ZSCORE_DRIFT",
    }
)


class AttributionOutcome(str, Enum):
    """Exhaustive 4-way causal attribution verdict after wet-lab execution."""

    EXECUTION_ARTIFACT = "EXECUTION_ARTIFACT"
    AMBIGUOUS_CHARACTERIZATION = "AMBIGUOUS_CHARACTERIZATION"
    CERTIFIED_NEGATIVE = "CERTIFIED_NEGATIVE"
    CERTIFIED_POSITIVE = "CERTIFIED_POSITIVE"


class DiagnosticAction(str, Enum):
    """Operational closed-loop routing action paired with each causal verdict."""

    RETRY_ROBOT = "RETRY_ROBOT"
    RUN_ORTHOGONAL_ASSAY = "RUN_ORTHOGONAL_ASSAY"
    COMMIT_CERTIFIED_NEGATIVE = "COMMIT_CERTIFIED_NEGATIVE"
    COMMIT_CERTIFIED_HIT = "COMMIT_CERTIFIED_HIT"


class OrthogonalModality(str, Enum):
    """Secondary analytical modalities capable of resolving primary spectral ambiguity."""

    LC_MS = "LC-MS"
    RAMAN = "Raman"
    TEM = "TEM"
    NMR_1H = "1H-NMR"


@dataclass(frozen=True)
class AssayTelemetry:
    """Post-execution robotic, environmental, control-well, and deconvolved spectral telemetry."""

    dispense_error_permille: int
    aspiration_clean: bool
    incubation_window_clean: bool
    seal_intact: bool
    positive_control_passed: bool
    blank_uncontaminated: bool
    detector_calibrated: bool
    precursor_overlap_permille: int
    spectral_residual_permille: int
    snr_tenths: int
    target_signal_present: bool
    positive_control_recovery_permille: int = 1000
    plate_z_prime_tenths: int = 70
    positive_control_zscore_tenths: int = 50

    def __post_init__(self) -> None:
        for name in (
            "dispense_error_permille",
            "precursor_overlap_permille",
            "spectral_residual_permille",
            "snr_tenths",
            "positive_control_recovery_permille",
        ):
            val = getattr(self, name)
            if isinstance(val, bool) or not isinstance(val, int) or val < 0:
                raise ValueError(f"{name} must be a non-negative integer, got {val!r}")
        for name in ("plate_z_prime_tenths", "positive_control_zscore_tenths"):
            val = getattr(self, name)
            if isinstance(val, bool) or not isinstance(val, int):
                raise ValueError(f"{name} must be an integer, got {val!r}")
        for name in (
            "aspiration_clean",
            "incubation_window_clean",
            "seal_intact",
            "positive_control_passed",
            "blank_uncontaminated",
            "detector_calibrated",
            "target_signal_present",
        ):
            val = getattr(self, name)
            if not isinstance(val, bool):
                raise ValueError(f"{name} must be a bool, got {val!r}")

    def violated_execution_clauses(self) -> list[str]:
        """Return ordered root-cause fault codes for any violated physical or batch-control checks."""
        faults: list[str] = []
        if self.dispense_error_permille > MAX_DISPENSE_ERROR_PERMILLE:
            faults.append("DISPENSE_ERROR_EXCEEDED")
        if not self.aspiration_clean:
            faults.append("ASPIRATION_CAVITATION_OR_CLOG")
        if not self.incubation_window_clean:
            faults.append("INCUBATION_WINDOW_VIOLATION")
        if not self.seal_intact:
            faults.append("SEAL_INTEGRITY_BREACH")
        if not self.positive_control_passed:
            faults.append("POSITIVE_CONTROL_FAILED")
        if not self.blank_uncontaminated:
            faults.append("BLANK_CONTAMINATED")
        if self.positive_control_recovery_permille < MIN_POS_CONTROL_RECOVERY_PERMILLE:
            faults.append("POSITIVE_CONTROL_RECOVERY_DRIFT")
        if self.plate_z_prime_tenths < MIN_Z_PRIME_TENTHS:
            faults.append("LOW_PLATE_Z_PRIME_FACTOR")
        if self.positive_control_zscore_tenths < MIN_POS_CONTROL_ZSCORE_TENTHS:
            faults.append("POSITIVE_CONTROL_ZSCORE_DRIFT")
        return faults

    def violated_characterization_clauses(self) -> list[str]:
        """Return ordered root-cause fault codes for any violated analytical checks."""
        faults: list[str] = []
        if not self.detector_calibrated:
            faults.append("DETECTOR_UNCALIBRATED_OR_SATURATED")
        if self.precursor_overlap_permille > MAX_PRECURSOR_OVERLAP_PERMILLE:
            faults.append("PRECURSOR_GHOST_PEAK_OVERLAP")
        if self.spectral_residual_permille > MAX_SPECTRAL_RESIDUAL_PERMILLE:
            faults.append("HIGH_SPECTRAL_RWP_RESIDUAL")
        if self.snr_tenths < MIN_RESOLVABLE_SNR_TENTHS:
            faults.append("LOW_ASSAY_SNR")
        return faults

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AssayTelemetry:
        def _req_int(key: str, default: Any = None, *, has_default: bool = False) -> int:
            raw = data.get(key, default) if has_default else data[key]
            if isinstance(raw, bool) or not isinstance(raw, numbers.Integral):
                raise ValueError(f"{key} must be an integer, got {raw!r}")
            return int(raw)

        def _req_bool(key: str) -> bool:
            raw = data[key]
            if not isinstance(raw, bool) and type(raw).__name__ != "bool_":
                raise ValueError(f"{key} must be a bool, got {raw!r}")
            return bool(raw)

        return cls(
            dispense_error_permille=_req_int("dispense_error_permille"),
            aspiration_clean=_req_bool("aspiration_clean"),
            incubation_window_clean=_req_bool("incubation_window_clean"),
            seal_intact=_req_bool("seal_intact"),
            positive_control_passed=_req_bool("positive_control_passed"),
            blank_uncontaminated=_req_bool("blank_uncontaminated"),
            detector_calibrated=_req_bool("detector_calibrated"),
            precursor_overlap_permille=_req_int("precursor_overlap_permille"),
            spectral_residual_permille=_req_int("spectral_residual_permille"),
            snr_tenths=_req_int("snr_tenths"),
            target_signal_present=_req_bool("target_signal_present"),
            positive_control_recovery_permille=_req_int(
                "positive_control_recovery_permille", 1000, has_default=True
            ),
            plate_z_prime_tenths=_req_int("plate_z_prime_tenths", 70, has_default=True),
            positive_control_zscore_tenths=_req_int(
                "positive_control_zscore_tenths", 50, has_default=True
            ),
        )


@dataclass(frozen=True)
class OrthogonalAssayReadout:
    """Secondary readout emitted by an orthogonal analytical modality."""

    modality: OrthogonalModality
    instrument_calibrated: bool
    orthogonal_snr_tenths: int
    selectivity_clean: bool
    target_confirmed: bool

    def __post_init__(self) -> None:
        if not isinstance(self.modality, OrthogonalModality):
            object.__setattr__(self, "modality", OrthogonalModality(self.modality))
        if (
            isinstance(self.orthogonal_snr_tenths, bool)
            or not isinstance(self.orthogonal_snr_tenths, int)
            or self.orthogonal_snr_tenths < 0
        ):
            raise ValueError(
                f"orthogonal_snr_tenths must be a non-negative integer, got {self.orthogonal_snr_tenths!r}"
            )
        for name in ("instrument_calibrated", "selectivity_clean", "target_confirmed"):
            val = getattr(self, name)
            if not isinstance(val, bool):
                raise ValueError(f"{name} must be a bool, got {val!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "modality": self.modality.value,
            "instrument_calibrated": self.instrument_calibrated,
            "orthogonal_snr_tenths": self.orthogonal_snr_tenths,
            "selectivity_clean": self.selectivity_clean,
            "target_confirmed": self.target_confirmed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OrthogonalAssayReadout:
        raw_snr = data["orthogonal_snr_tenths"]
        if isinstance(raw_snr, bool) or not isinstance(raw_snr, numbers.Integral):
            raise ValueError(
                f"orthogonal_snr_tenths must be a non-negative integer, got {raw_snr!r}"
            )

        def _req_bool(key: str) -> bool:
            raw = data[key]
            if not isinstance(raw, bool) and type(raw).__name__ != "bool_":
                raise ValueError(f"{key} must be a bool, got {raw!r}")
            return bool(raw)

        return cls(
            modality=OrthogonalModality(data["modality"]),
            instrument_calibrated=_req_bool("instrument_calibrated"),
            orthogonal_snr_tenths=int(raw_snr),
            selectivity_clean=_req_bool("selectivity_clean"),
            target_confirmed=_req_bool("target_confirmed"),
        )


@dataclass(frozen=True)
class AttributionReport:
    """Complete structured diagnostic verdict for a wet-lab assay run."""

    outcome: AttributionOutcome
    action: DiagnosticAction
    admit_to_memory: bool
    execution_clean: bool
    characterization_identifiable: bool
    violated_execution_clauses: tuple[str, ...]
    violated_characterization_clauses: tuple[str, ...]
    resolved_via_orthogonal: bool
    lean_theorem_tag: str
    dual_fault_detected: bool = False
    compound_fault: bool = False
    fault_scope: str = "NONE"
    remediation_steps: tuple[str, ...] = ()
    diagnostic_notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome.value,
            "action": self.action.value,
            "admit_to_memory": self.admit_to_memory,
            "execution_clean": self.execution_clean,
            "characterization_identifiable": self.characterization_identifiable,
            "violated_execution_clauses": list(self.violated_execution_clauses),
            "violated_characterization_clauses": list(self.violated_characterization_clauses),
            "resolved_via_orthogonal": self.resolved_via_orthogonal,
            "lean_theorem_tag": self.lean_theorem_tag,
            "dual_fault_detected": self.dual_fault_detected,
            "compound_fault": self.compound_fault,
            "fault_scope": self.fault_scope,
            "remediation_steps": list(self.remediation_steps),
            "diagnostic_notes": list(self.diagnostic_notes),
        }


def execution_clean(telemetry: AssayTelemetry) -> bool:
    """Mirror of `LabAttribLean.executionClean`."""
    return (
        telemetry.dispense_error_permille <= MAX_DISPENSE_ERROR_PERMILLE
        and telemetry.aspiration_clean
        and telemetry.incubation_window_clean
        and telemetry.seal_intact
        and telemetry.positive_control_passed
        and telemetry.blank_uncontaminated
        and telemetry.positive_control_recovery_permille >= MIN_POS_CONTROL_RECOVERY_PERMILLE
        and telemetry.plate_z_prime_tenths >= MIN_Z_PRIME_TENTHS
        and telemetry.positive_control_zscore_tenths >= MIN_POS_CONTROL_ZSCORE_TENTHS
    )


def characterization_identifiable(telemetry: AssayTelemetry) -> bool:
    """Mirror of `LabAttribLean.characterizationIdentifiable`."""
    return (
        telemetry.detector_calibrated
        and telemetry.precursor_overlap_permille <= MAX_PRECURSOR_OVERLAP_PERMILLE
        and telemetry.spectral_residual_permille <= MAX_SPECTRAL_RESIDUAL_PERMILLE
        and telemetry.snr_tenths >= MIN_RESOLVABLE_SNR_TENTHS
    )


def attribute_assay_outcome(telemetry: AssayTelemetry) -> AttributionOutcome:
    """Mirror of `LabAttribLean.attributeAssayOutcome`."""
    if not execution_clean(telemetry):
        return AttributionOutcome.EXECUTION_ARTIFACT
    if not characterization_identifiable(telemetry):
        return AttributionOutcome.AMBIGUOUS_CHARACTERIZATION
    if telemetry.target_signal_present:
        return AttributionOutcome.CERTIFIED_POSITIVE
    return AttributionOutcome.CERTIFIED_NEGATIVE


def route_diagnostic_action(outcome: AttributionOutcome) -> DiagnosticAction:
    """Mirror of `LabAttribLean.routeDiagnosticAction`."""
    if outcome is AttributionOutcome.EXECUTION_ARTIFACT:
        return DiagnosticAction.RETRY_ROBOT
    if outcome is AttributionOutcome.AMBIGUOUS_CHARACTERIZATION:
        return DiagnosticAction.RUN_ORTHOGONAL_ASSAY
    if outcome is AttributionOutcome.CERTIFIED_NEGATIVE:
        return DiagnosticAction.COMMIT_CERTIFIED_NEGATIVE
    if outcome is AttributionOutcome.CERTIFIED_POSITIVE:
        return DiagnosticAction.COMMIT_CERTIFIED_HIT
    raise ValueError(f"Unsupported outcome: {outcome!r}")


def admit_to_scientific_memory(outcome: AttributionOutcome) -> bool:
    """Mirror of `LabAttribLean.admitToScientificMemory`."""
    return outcome in (
        AttributionOutcome.CERTIFIED_NEGATIVE,
        AttributionOutcome.CERTIFIED_POSITIVE,
    )


def orthogonal_readout_clean(readout: OrthogonalAssayReadout) -> bool:
    """Mirror of `LabAttribLean.orthogonalReadoutClean`."""
    return (
        readout.instrument_calibrated
        and readout.orthogonal_snr_tenths >= MIN_RESOLVABLE_SNR_TENTHS
        and readout.selectivity_clean
    )


def resolve_with_orthogonal_assay(
    primary: AssayTelemetry,
    secondary: OrthogonalAssayReadout,
) -> AttributionOutcome:
    """Mirror of `LabAttribLean.resolveAmbiguousSample`."""
    if not execution_clean(primary):
        return AttributionOutcome.EXECUTION_ARTIFACT
    if not orthogonal_readout_clean(secondary):
        return AttributionOutcome.AMBIGUOUS_CHARACTERIZATION
    if secondary.target_confirmed:
        return AttributionOutcome.CERTIFIED_POSITIVE
    return AttributionOutcome.CERTIFIED_NEGATIVE


def effective_attribution(
    primary: AssayTelemetry,
    secondary: OrthogonalAssayReadout | None = None,
) -> AttributionOutcome:
    """Mirror of `LabAttribLean.effectiveAttribution`."""
    primary_outcome = attribute_assay_outcome(primary)
    if secondary is None:
        return primary_outcome
    if primary_outcome is AttributionOutcome.AMBIGUOUS_CHARACTERIZATION:
        return resolve_with_orthogonal_assay(primary, secondary)
    return primary_outcome


def lean_certificate_tag_for(
    primary: AssayTelemetry,
    secondary: OrthogonalAssayReadout | None = None,
) -> str:
    """Return the exact `LabAttribLean` theorem name certifying the attribution decision."""
    primary_outcome = attribute_assay_outcome(primary)
    if primary_outcome is AttributionOutcome.EXECUTION_ARTIFACT:
        return "LabAttribLean.corrupted_run_never_poisons_memory"
    if primary_outcome is AttributionOutcome.CERTIFIED_NEGATIVE:
        return "LabAttribLean.certified_negative_sound"
    if primary_outcome is AttributionOutcome.CERTIFIED_POSITIVE:
        return "LabAttribLean.certified_positive_sound"
    # Primary was AMBIGUOUS_CHARACTERIZATION
    if secondary is None:
        return "LabAttribLean.unidentifiable_spectrum_never_certified"
    resolved = resolve_with_orthogonal_assay(primary, secondary)
    if resolved is AttributionOutcome.CERTIFIED_POSITIVE:
        return "LabAttribLean.orthogonal_confirm_promotes_certified_positive"
    if resolved is AttributionOutcome.CERTIFIED_NEGATIVE:
        return "LabAttribLean.orthogonal_refute_promotes_certified_negative"
    return "LabAttribLean.unconfirmed_orthogonal_never_admitted"


def diagnose_assay(
    primary: AssayTelemetry,
    secondary: OrthogonalAssayReadout | None = None,
) -> AttributionReport:
    """Produce a full diagnostic report combining blame calculus, routing, and Lean proof tags."""
    exec_ok = execution_clean(primary)
    ident_ok = characterization_identifiable(primary)
    exec_clauses = tuple(primary.violated_execution_clauses())
    char_clauses = tuple(primary.violated_characterization_clauses())
    dual_fault = (not exec_ok) and (not ident_ok)

    has_plate_fault = any(code in _PLATE_CONTROL_FAULT_CODES for code in exec_clauses)
    if has_plate_fault:
        fault_scope = "PLATE_REAGENT_CONTAMINATION"
    elif exec_clauses:
        fault_scope = "WELL_MECHANICAL"
    else:
        fault_scope = "NONE"

    primary_out = attribute_assay_outcome(primary)
    final_out = effective_attribution(primary, secondary)
    final_action = route_diagnostic_action(final_out)
    resolved_orth = (
        secondary is not None
        and primary_out is AttributionOutcome.AMBIGUOUS_CHARACTERIZATION
        and admit_to_scientific_memory(final_out)
    )

    steps: list[str] = []
    notes: list[str] = []
    if not exec_ok:
        if not primary.detector_calibrated:
            steps.append("RECALIBRATE_DETECTOR")
        if has_plate_fault:
            steps.append("REFRESH_PLATE_CONTROLS_AND_BLANKS")
            notes.append(
                "PLATE_REAGENT_CONTAMINATION: refresh control/blank troughs before robotic retry"
            )
        steps.append("RETRY_ROBOT")
        if dual_fault:
            notes.append("RECALIBRATE_AND_RETRY")
            if any(c != "DETECTOR_UNCALIBRATED_OR_SATURATED" for c in char_clauses):
                steps.append("PRE_STAGE_ORTHOGONAL_ASSAY")
    elif final_out is AttributionOutcome.AMBIGUOUS_CHARACTERIZATION:
        if not primary.detector_calibrated:
            steps.append("RECALIBRATE_DETECTOR")
        steps.append("RUN_ORTHOGONAL_ASSAY")
    else:
        steps.append(final_action.value)

    return AttributionReport(
        outcome=final_out,
        action=final_action,
        admit_to_memory=admit_to_scientific_memory(final_out),
        execution_clean=exec_ok,
        characterization_identifiable=ident_ok,
        violated_execution_clauses=exec_clauses,
        violated_characterization_clauses=char_clauses,
        resolved_via_orthogonal=resolved_orth,
        lean_theorem_tag=lean_certificate_tag_for(primary, secondary),
        dual_fault_detected=dual_fault,
        compound_fault=dual_fault,
        fault_scope=fault_scope,
        remediation_steps=tuple(steps),
        diagnostic_notes=tuple(notes),
    )


@dataclass(frozen=True)
class AttributedSteppingStone:
    """Exact Python mirror of `LabAttribLean.AttributedSteppingStone`."""

    tau_permille: int
    eta_permille: int
    dist_permille: int
    proposed_new_tau_permille: int
    volume_ul: int
    uses_only_stock_reagents: bool
    within_hazard_envelope: bool
    telemetry: AssayTelemetry
    orthogonal_readout: OrthogonalAssayReadout | None = None


def in_annular_shell(tau_permille: int, eta_permille: int, dist_permille: int) -> bool:
    """Mirror of `LabAttribLean.inAnnularShell`."""
    return (
        tau_permille > 0
        and 0 <= eta_permille <= MAX_EXPANSION_ETA_PERMILLE
        and tau_permille < dist_permille
        and 1000 * dist_permille <= (1000 + eta_permille) * tau_permille
    )


def micro_containment_clean(probe: AttributedSteppingStone) -> bool:
    """Mirror of `LabAttribLean.microContainmentClean`."""
    return (
        1 <= probe.volume_ul <= MAX_STEPPING_STONE_VOLUME_UL
        and probe.uses_only_stock_reagents
        and probe.within_hazard_envelope
    )


def check_attributed_frontier_update(probe: AttributedSteppingStone) -> bool:
    """Mirror of `LabAttribLean.checkAttributedFrontierUpdate`."""
    outcome = effective_attribution(probe.telemetry, probe.orthogonal_readout)
    return (
        in_annular_shell(probe.tau_permille, probe.eta_permille, probe.dist_permille)
        and micro_containment_clean(probe)
        and admit_to_scientific_memory(outcome)
        and probe.tau_permille < probe.proposed_new_tau_permille <= probe.dist_permille
    )
