"""Expected Value of Information (EVOI) Orthogonal Assay Router.

When `attribute_assay_outcome(telemetry) == AMBIGUOUS_CHARACTERIZATION`, single-modality
primary characterization (such as powder XRD under precursor peak overlap or plate-reader
absorbance under excipient turbidity) cannot soundly certify either a positive discovery
or a true negative.

This router evaluates candidate secondary analytical modalities (`LC-MS`, `Raman`, `TEM`,
`1H-NMR`) against sample volume constraints, ambiguity physics, sensitivity/specificity,
and assay cost to select the Net-EVOI-optimal confirmatory experiment (or discard if no
modality achieves positive Net EVOI).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from lab_attrib.blame import (
    MAX_PRECURSOR_OVERLAP_PERMILLE,
    MAX_SPECTRAL_RESIDUAL_PERMILLE,
    AssayTelemetry,
    AttributionOutcome,
    OrthogonalModality,
    attribute_assay_outcome,
)


@dataclass(frozen=True)
class ModalitySpec:
    """Instrument specification for a secondary orthogonal characterization modality."""

    modality: OrthogonalModality
    min_sample_volume_ul: float
    assay_cost_usd: float
    turnaround_minutes: float
    base_sensitivity: float
    base_specificity: float
    resolves_precursor_ghost: bool
    resolves_multiphase_rwp: bool
    suitable_for_solid_state: bool
    suitable_for_solution: bool


DEFAULT_MODALITY_SPECS: tuple[ModalitySpec, ...] = (
    ModalitySpec(
        modality=OrthogonalModality.RAMAN,
        min_sample_volume_ul=1.5,
        assay_cost_usd=28.0,
        turnaround_minutes=8.0,
        base_sensitivity=0.93,
        base_specificity=0.985,
        resolves_precursor_ghost=True,
        resolves_multiphase_rwp=False,
        suitable_for_solid_state=True,
        suitable_for_solution=True,
    ),
    ModalitySpec(
        modality=OrthogonalModality.LC_MS,
        min_sample_volume_ul=4.0,
        assay_cost_usd=45.0,
        turnaround_minutes=18.0,
        base_sensitivity=0.97,
        base_specificity=0.992,
        resolves_precursor_ghost=True,
        resolves_multiphase_rwp=True,
        suitable_for_solid_state=False,
        suitable_for_solution=True,
    ),
    ModalitySpec(
        modality=OrthogonalModality.NMR_1H,
        min_sample_volume_ul=15.0,
        assay_cost_usd=85.0,
        turnaround_minutes=35.0,
        base_sensitivity=0.96,
        base_specificity=0.996,
        resolves_precursor_ghost=True,
        resolves_multiphase_rwp=True,
        suitable_for_solid_state=False,
        suitable_for_solution=True,
    ),
    ModalitySpec(
        modality=OrthogonalModality.TEM,
        min_sample_volume_ul=1.0,
        assay_cost_usd=160.0,
        turnaround_minutes=75.0,
        base_sensitivity=0.98,
        base_specificity=0.998,
        resolves_precursor_ghost=True,
        resolves_multiphase_rwp=True,
        suitable_for_solid_state=True,
        suitable_for_solution=False,
    ),
)


@dataclass(frozen=True)
class ModalityEvaluation:
    """Per-modality Bayesian EVOI evaluation record."""

    modality: OrthogonalModality
    feasible: bool
    effective_resolvability: float
    gross_evoi_usd: float
    total_cost_usd: float
    net_evoi_usd: float
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "modality": self.modality.value,
            "feasible": self.feasible,
            "effective_resolvability": round(self.effective_resolvability, 4),
            "gross_evoi_usd": round(self.gross_evoi_usd, 4),
            "total_cost_usd": round(self.total_cost_usd, 4),
            "net_evoi_usd": round(self.net_evoi_usd, 4),
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class OrthogonalRoutingDecision:
    """Selected EVOI-optimal secondary assay routing decision."""

    selected_modality: OrthogonalModality | None
    discard_sample: bool
    best_net_evoi_usd: float
    prior_hit_probability: float
    rankings: tuple[ModalityEvaluation, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_modality": (
                self.selected_modality.value if self.selected_modality is not None else None
            ),
            "discard_sample": self.discard_sample,
            "best_net_evoi_usd": round(self.best_net_evoi_usd, 4),
            "prior_hit_probability": round(self.prior_hit_probability, 4),
            "rankings": [r.to_dict() for r in self.rankings],
        }


def route_orthogonal_assay(
    telemetry: AssayTelemetry,
    *,
    prior_hit_prob: float = 0.25,
    available_volume_ul: float = 8.0,
    sample_phase: str = "solution",
    discovery_value_usd: float = 400.0,
    certified_negative_info_value_usd: float = 22.0,
    false_positive_penalty_usd: float = 1200.0,
    time_cost_per_minute_usd: float = 0.50,
    modality_specs: Sequence[ModalitySpec] = DEFAULT_MODALITY_SPECS,
) -> OrthogonalRoutingDecision:
    """Compute Bayesian EVOI across orthogonal modalities for an ambiguous sample.

    Raises `ValueError` if called on a sample whose primary outcome is not
    `AMBIGUOUS_CHARACTERIZATION`.
    """
    primary_outcome = attribute_assay_outcome(telemetry)
    if primary_outcome is not AttributionOutcome.AMBIGUOUS_CHARACTERIZATION:
        raise ValueError(
            "Orthogonal assay router requires AMBIGUOUS_CHARACTERIZATION primary outcome, "
            f"got {primary_outcome.value}"
        )
    if not (0.0 <= prior_hit_prob <= 1.0):
        raise ValueError(f"prior_hit_prob must be in [0, 1], got {prior_hit_prob}")
    if sample_phase not in ("solid", "solution"):
        raise ValueError(f"sample_phase must be 'solid' or 'solution', got {sample_phase!r}")

    has_ghost_overlap = telemetry.precursor_overlap_permille > MAX_PRECURSOR_OVERLAP_PERMILLE
    has_high_rwp = telemetry.spectral_residual_permille > MAX_SPECTRAL_RESIDUAL_PERMILLE

    evaluations: list[ModalityEvaluation] = []
    for spec in modality_specs:
        total_cost = spec.assay_cost_usd + time_cost_per_minute_usd * spec.turnaround_minutes
        if available_volume_ul < spec.min_sample_volume_ul:
            evaluations.append(
                ModalityEvaluation(
                    modality=spec.modality,
                    feasible=False,
                    effective_resolvability=0.0,
                    gross_evoi_usd=0.0,
                    total_cost_usd=total_cost,
                    net_evoi_usd=-total_cost,
                    rationale=(
                        f"Insufficient sample volume ({available_volume_ul:.1f} uL < "
                        f"{spec.min_sample_volume_ul:.1f} uL required)"
                    ),
                )
            )
            continue

        phase_compatible = (
            spec.suitable_for_solid_state
            if sample_phase == "solid"
            else spec.suitable_for_solution
        )
        if not phase_compatible:
            evaluations.append(
                ModalityEvaluation(
                    modality=spec.modality,
                    feasible=False,
                    effective_resolvability=0.0,
                    gross_evoi_usd=0.0,
                    total_cost_usd=total_cost,
                    net_evoi_usd=-total_cost,
                    rationale=f"Modality incompatible with {sample_phase} sample state",
                )
            )
            continue

        # Compute probability that the secondary assay produces a clean, selective resolution
        resolvability = 0.96
        if has_ghost_overlap and not spec.resolves_precursor_ghost:
            resolvability *= 0.25
        if has_high_rwp and not spec.resolves_multiphase_rwp:
            resolvability *= 0.55

        # Under default-deny (`U_0 = 0`), running modality `m` yields:
        #   - Expected hit utility when confirmed cleanly:
        #     max(0, p * TPR * V_hit - (1 - p) * (1 - TNR) * L_FP)
        #   - Expected active-learning utility of certifying a clean outcome (hit or negative):
        #     (1 - p) * TNR * V_neg_info
        expected_confirm_utility = max(
            0.0,
            prior_hit_prob * spec.base_sensitivity * discovery_value_usd
            - (1.0 - prior_hit_prob) * (1.0 - spec.base_specificity) * false_positive_penalty_usd,
        )
        expected_neg_utility = (
            (1.0 - prior_hit_prob) * spec.base_specificity * certified_negative_info_value_usd
        )
        gross_evoi = resolvability * (expected_confirm_utility + expected_neg_utility)
        net_evoi = gross_evoi - total_cost

        evaluations.append(
            ModalityEvaluation(
                modality=spec.modality,
                feasible=True,
                effective_resolvability=resolvability,
                gross_evoi_usd=gross_evoi,
                total_cost_usd=total_cost,
                net_evoi_usd=net_evoi,
                rationale=(
                    f"Feasible for {sample_phase} phase; resolvability={resolvability:.2f}, "
                    f"gross_EVOI=${gross_evoi:.2f} vs total_cost=${total_cost:.2f}"
                ),
            )
        )

    evaluations.sort(key=lambda e: (e.feasible, e.net_evoi_usd), reverse=True)
    best = evaluations[0] if evaluations else None
    if best is not None and best.feasible and best.net_evoi_usd > 0.0:
        return OrthogonalRoutingDecision(
            selected_modality=best.modality,
            discard_sample=False,
            best_net_evoi_usd=best.net_evoi_usd,
            prior_hit_probability=prior_hit_prob,
            rankings=tuple(evaluations),
        )
    return OrthogonalRoutingDecision(
        selected_modality=None,
        discard_sample=True,
        best_net_evoi_usd=0.0,
        prior_hit_probability=prior_hit_prob,
        rankings=tuple(evaluations),
    )
