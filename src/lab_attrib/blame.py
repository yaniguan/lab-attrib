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
from typing import Any

# Exact integer permille / tenths thresholds mirrored 1-to-1 with `lean/LabAttribLean.lean`
MAX_DISPENSE_ERROR_PERMILLE: int = 50
MAX_PRECURSOR_OVERLAP_PERMILLE: int = 150
MAX_SPECTRAL_RESIDUAL_PERMILLE: int = 100
MIN_RESOLVABLE_SNR_TENTHS: int = 30
MAX_STEPPING_STONE_VOLUME_UL: int = 10
MAX_EXPANSION_ETA_PERMILLE: int = 500


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

    def __post_init__(self) -> None:
        for name in (
            "dispense_error_permille",
            "precursor_overlap_permille",
            "spectral_residual_permille",
            "snr_tenths",
        ):
            val = getattr(self, name)
            if isinstance(val, bool) or not isinstance(val, int) or val < 0:
                raise ValueError(f"{name} must be a non-negative integer, got {val!r}")
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
        """Return ordered root-cause fault codes for any violated physical execution checks."""
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
        return cls(
            dispense_error_permille=int(data["dispense_error_permille"]),
            aspiration_clean=bool(data["aspiration_clean"]),
            incubation_window_clean=bool(data["incubation_window_clean"]),
            seal_intact=bool(data["seal_intact"]),
            positive_control_passed=bool(data["positive_control_passed"]),
            blank_uncontaminated=bool(data["blank_uncontaminated"]),
            detector_calibrated=bool(data["detector_calibrated"]),
            precursor_overlap_permille=int(data["precursor_overlap_permille"]),
            spectral_residual_permille=int(data["spectral_residual_permille"]),
            snr_tenths=int(data["snr_tenths"]),
            target_signal_present=bool(data["target_signal_present"]),
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
        return cls(
            modality=OrthogonalModality(data["modality"]),
            instrument_calibrated=bool(data["instrument_calibrated"]),
            orthogonal_snr_tenths=int(data["orthogonal_snr_tenths"]),
            selectivity_clean=bool(data["selectivity_clean"]),
            target_confirmed=bool(data["target_confirmed"]),
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
    primary_out = attribute_assay_outcome(primary)
    final_out = effective_attribution(primary, secondary)
    resolved_orth = (
        secondary is not None
        and primary_out is AttributionOutcome.AMBIGUOUS_CHARACTERIZATION
        and admit_to_scientific_memory(final_out)
    )
    return AttributionReport(
        outcome=final_out,
        action=route_diagnostic_action(final_out),
        admit_to_memory=admit_to_scientific_memory(final_out),
        execution_clean=exec_ok,
        characterization_identifiable=ident_ok,
        violated_execution_clauses=tuple(primary.violated_execution_clauses()),
        violated_characterization_clauses=tuple(primary.violated_characterization_clauses()),
        resolved_via_orthogonal=resolved_orth,
        lean_theorem_tag=lean_certificate_tag_for(primary, secondary),
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
