"""Expected Value of Information (EVOI) Orthogonal Assay Router.

When `attribute_assay_outcome(telemetry) == AMBIGUOUS_CHARACTERIZATION`, single-modality
primary characterization (such as powder XRD under precursor peak overlap or plate-reader
absorbance under excipient turbidity) cannot soundly certify either a positive discovery
or a true negative.

This router evaluates candidate secondary analytical modalities (`LC-MS`, `Raman`, `TEM`,
`1H-NMR`) against sample volume constraints, ambiguity physics, correlated modality
blindspots (e.g., centrosymmetric Raman selection rules, fluorescence, dilute trace LOD),
unclamped Bayes risk reduction, and assay cost to select the Net-EVOI-optimal confirmatory
experiment (or discard if no modality achieves positive Net EVOI).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from lab_attrib.blame import (
    MAX_PRECURSOR_OVERLAP_PERMILLE,
    MIN_RESOLVABLE_SNR_TENTHS,
    AssayTelemetry,
    AttributionOutcome,
    OrthogonalModality,
    attribute_assay_outcome,
    max_residual_for_fit_method,
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
    shared_blindspot_tags: tuple[str, ...] = ()
    min_concentration_um: float = 0.0


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
        shared_blindspot_tags=("centrosymmetric_crystal", "fluorescence", "high_fluorescence"),
        min_concentration_um=5.0,
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
        shared_blindspot_tags=("non_ionizable",),
        min_concentration_um=0.05,
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
        shared_blindspot_tags=("trace_concentration", "low_snr", "paramagnetic"),
        min_concentration_um=50.0,
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
        shared_blindspot_tags=("beam_sensitive_volatile",),
        min_concentration_um=0.0,
    ),
)

_HARD_INFEASIBLE_BLINDSPOTS: frozenset[str] = frozenset(
    {
        "centrosymmetric_crystal",
        "fluorescence",
        "high_fluorescence",
        "non_ionizable",
        "paramagnetic",
        "beam_sensitive_volatile",
    }
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
    posterior_hit_probability: float | None = None

    def __post_init__(self) -> None:
        if self.posterior_hit_probability is None:
            object.__setattr__(self, "posterior_hit_probability", self.prior_hit_probability)

    def to_dict(self) -> dict[str, Any]:
        post_p = (
            self.posterior_hit_probability
            if self.posterior_hit_probability is not None
            else self.prior_hit_probability
        )
        return {
            "selected_modality": (
                self.selected_modality.value if self.selected_modality is not None else None
            ),
            "discard_sample": self.discard_sample,
            "best_net_evoi_usd": round(self.best_net_evoi_usd, 4),
            "prior_hit_probability": round(self.prior_hit_probability, 4),
            "posterior_hit_probability": round(post_p, 4),
            "rankings": [r.to_dict() for r in self.rankings],
        }


def _compute_posterior_hit_prob(
    telemetry: AssayTelemetry,
    prior_hit_prob: float,
    *,
    update_belief_from_primary: bool,
) -> float:
    """Update prior `p(Z=1)` to `p(Z=1 | y_primary)` when primary channel is unconfounded."""
    if not update_belief_from_primary or prior_hit_prob <= 0.0 or prior_hit_prob >= 1.0:
        return prior_hit_prob
    if (
        not telemetry.detector_calibrated
        or telemetry.precursor_overlap_permille > MAX_PRECURSOR_OVERLAP_PERMILLE
    ):
        # Primary peak channel is confounded by ghost overlap or uncalibrated detector -> LR = 1
        return prior_hit_prob

    snr_ratio = min(
        1.0,
        max(0.0, float(telemetry.snr_tenths) / float(MIN_RESOLVABLE_SNR_TENTHS)),
    )
    rwp_attenuation = (
        0.5
        if telemetry.spectral_residual_permille > max_residual_for_fit_method(telemetry.fit_method)
        else 1.0
    )
    info_weight = snr_ratio * rwp_attenuation
    delta_log_odds = info_weight * (0.55 if telemetry.target_signal_present else -0.55)
    prior_odds = prior_hit_prob / (1.0 - prior_hit_prob)
    post_odds = prior_odds * math.exp(delta_log_odds)
    return min(1.0, max(0.0, post_odds / (1.0 + post_odds)))


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
    centrosymmetric_crystal: bool = False,
    high_fluorescence_risk: bool = False,
    analyte_concentration_um: float | None = None,
    shared_blindspot_tags: Sequence[str] = (),
    update_belief_from_primary: bool = True,
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
    if not math.isfinite(prior_hit_prob) or not (0.0 <= prior_hit_prob <= 1.0):
        raise ValueError(f"prior_hit_prob must be in [0, 1], got {prior_hit_prob}")
    if sample_phase not in ("solid", "solution"):
        raise ValueError(f"sample_phase must be 'solid' or 'solution', got {sample_phase!r}")
    if analyte_concentration_um is not None and (
        not math.isfinite(analyte_concentration_um) or analyte_concentration_um < 0.0
    ):
        raise ValueError(
            f"analyte_concentration_um must be a non-negative finite float, got {analyte_concentration_um}"
        )

    has_ghost_overlap = telemetry.precursor_overlap_permille > MAX_PRECURSOR_OVERLAP_PERMILLE
    has_high_rwp = (
        telemetry.spectral_residual_permille > max_residual_for_fit_method(telemetry.fit_method)
    )
    has_low_snr = telemetry.snr_tenths < MIN_RESOLVABLE_SNR_TENTHS

    active_blindspots: set[str] = {
        str(tag).strip().lower() for tag in shared_blindspot_tags if str(tag).strip()
    }
    if centrosymmetric_crystal:
        active_blindspots.add("centrosymmetric_crystal")
    if high_fluorescence_risk:
        active_blindspots.add("fluorescence")
        active_blindspots.add("high_fluorescence")

    posterior_p = _compute_posterior_hit_prob(
        telemetry,
        prior_hit_prob,
        update_belief_from_primary=update_belief_from_primary,
    )

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

        spec_blindspots = {t.strip().lower() for t in spec.shared_blindspot_tags if t.strip()}
        overlapping_blindspots = sorted(active_blindspots & spec_blindspots)
        if any(tag in _HARD_INFEASIBLE_BLINDSPOTS for tag in overlapping_blindspots):
            evaluations.append(
                ModalityEvaluation(
                    modality=spec.modality,
                    feasible=False,
                    effective_resolvability=0.0,
                    gross_evoi_usd=0.0,
                    total_cost_usd=total_cost,
                    net_evoi_usd=-total_cost,
                    rationale=(
                        f"Correlated modality blindspot ({', '.join(overlapping_blindspots)}) "
                        f"prevents selective {spec.modality.value} resolution"
                    ),
                )
            )
            continue

        resolvability = 0.96
        eff_sensitivity = spec.base_sensitivity
        penalty_notes: list[str] = []

        if has_ghost_overlap and not spec.resolves_precursor_ghost:
            resolvability *= 0.25
        if has_high_rwp and not spec.resolves_multiphase_rwp:
            resolvability *= 0.55

        if overlapping_blindspots:
            resolvability *= 0.20
            eff_sensitivity *= 0.45
            penalty_notes.append(f"shared_blindspot={','.join(overlapping_blindspots)}")

        dilute_sample = (
            analyte_concentration_um is not None
            and analyte_concentration_um < spec.min_concentration_um
        )
        if dilute_sample or (has_low_snr and spec.min_concentration_um >= 25.0):
            resolvability *= 0.25
            eff_sensitivity *= 0.55
            penalty_notes.append("low_snr_high_lod_attenuation")
        elif has_low_snr and sample_phase == "solution" and spec.min_concentration_um >= 1.0:
            # Spontaneous solution-phase optical scattering loses sensitivity on trace low-SNR samples
            resolvability *= 0.55
            eff_sensitivity *= 0.80
            penalty_notes.append("solution_trace_snr_attenuation")

        # Exact unclamped Bayes risk reduction consistent with downstream
        # `resolve_with_orthogonal_assay` unconditional positive/negative commitment:
        expected_confirm_utility = (
            posterior_p * eff_sensitivity * discovery_value_usd
            - (1.0 - posterior_p) * (1.0 - spec.base_specificity) * false_positive_penalty_usd
        )
        expected_neg_utility = (
            (1.0 - posterior_p) * spec.base_specificity * certified_negative_info_value_usd
        )
        gross_evoi = resolvability * (expected_confirm_utility + expected_neg_utility)
        net_evoi = gross_evoi - total_cost

        extra = f" [{'; '.join(penalty_notes)}]" if penalty_notes else ""
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
                    f"gross_EVOI=${gross_evoi:.2f} vs total_cost=${total_cost:.2f}{extra}"
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
            posterior_hit_probability=posterior_p,
        )
    return OrthogonalRoutingDecision(
        selected_modality=None,
        discard_sample=True,
        best_net_evoi_usd=0.0,
        prior_hit_probability=prior_hit_prob,
        rankings=tuple(evaluations),
        posterior_hit_probability=posterior_p,
    )
