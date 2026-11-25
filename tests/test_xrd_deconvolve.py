"""Tests for NNLS Precursor Peak-Overlap & R_wp Deconvolver (`xrd_deconvolve.py`)."""

from __future__ import annotations

import math
import random

import pytest

from lab_attrib.blame import (
    AttributionOutcome,
    attribute_assay_outcome,
    diagnose_assay,
)
from lab_attrib.xrd_deconvolve import (
    build_gaussian_spectrum,
    deconvolve_spectrum,
)


def _sample_grid() -> list[float]:
    return [10.0 + 1.25 * i for i in range(48)]


def test_pure_target_phase_recovers_certified_positive() -> None:
    grid = _sample_grid()
    prec_1 = build_gaussian_spectrum(grid, [(18.0, 1.0, 1.8), (44.0, 0.7, 1.8)])
    prec_2 = build_gaussian_spectrum(grid, [(26.0, 0.9, 1.8)])
    target = build_gaussian_spectrum(grid, [(35.0, 1.0, 2.0), (56.0, 0.85, 2.0)])

    observed = [1.25 * v for v in target]
    res = deconvolve_spectrum(observed, [prec_1, prec_2], target, min_hit_amplitude=0.25)

    assert res.target_amplitude == pytest.approx(1.25, abs=1e-4)
    assert res.precursor_overlap_permille <= 5
    assert res.spectral_residual_permille <= 5
    assert res.target_signal_present is True
    assert res.converged is True
    assert res.kkt_max_violation <= 1e-8

    telemetry = res.to_assay_telemetry()
    assert attribute_assay_outcome(telemetry) is AttributionOutcome.CERTIFIED_POSITIVE


def test_alab_ghost_peak_blocks_false_positive_despite_high_naive_cosine() -> None:
    grid = _sample_grid()
    prec_nonoverlap = build_gaussian_spectrum(grid, [(16.0, 1.0, 1.8)])
    prec_ghost = build_gaussian_spectrum(
        grid, [(34.0, 0.96, 2.1), (52.0, 0.90, 2.1), (22.0, 0.35, 1.7)]
    )
    target = build_gaussian_spectrum(grid, [(34.0, 1.0, 2.0), (52.0, 0.92, 2.0)])

    observed = [0.20 * prec_nonoverlap[i] + 1.10 * prec_ghost[i] for i in range(len(grid))]
    res = deconvolve_spectrum(
        observed, [prec_nonoverlap, prec_ghost], target, min_hit_amplitude=0.25
    )

    assert res.naive_peak_match_cosine > 0.88
    assert res.naive_claims_hit is True
    assert res.precursor_overlap_permille > 700

    telemetry = res.to_assay_telemetry()
    assert attribute_assay_outcome(telemetry) is AttributionOutcome.AMBIGUOUS_CHARACTERIZATION


def test_clean_non_overlapping_precursor_only_yields_certified_negative() -> None:
    grid = _sample_grid()
    prec_1 = build_gaussian_spectrum(grid, [(18.0, 1.0, 1.8), (44.0, 0.8, 1.8)])
    target = build_gaussian_spectrum(grid, [(34.0, 1.0, 2.0), (58.0, 0.9, 2.0)])

    observed = [0.95 * v for v in prec_1]
    res = deconvolve_spectrum(
        observed, [prec_1], target, min_hit_amplitude=0.25, noise_sigma=0.03
    )

    assert res.target_signal_present is False
    assert res.precursor_overlap_permille <= 5
    assert res.spectral_residual_permille <= 10
    assert res.snr_tenths >= 30
    assert attribute_assay_outcome(res.to_assay_telemetry()) is AttributionOutcome.CERTIFIED_NEGATIVE


def test_clean_zero_baseline_buffer_yields_certified_negative() -> None:
    grid = _sample_grid()
    prec_1 = build_gaussian_spectrum(grid, [(20.0, 1.0, 1.8)])
    target = build_gaussian_spectrum(grid, [(38.0, 1.0, 2.0)])

    rng = random.Random(42)
    observed = [max(0.0, rng.gauss(0.0, 0.004)) for _ in grid]
    res = deconvolve_spectrum(
        observed, [prec_1], target, min_hit_amplitude=0.25, noise_sigma=0.03
    )

    assert res.target_signal_present is False
    assert res.spectral_residual_permille <= 100
    assert res.precursor_overlap_permille <= 50
    assert attribute_assay_outcome(res.to_assay_telemetry()) is AttributionOutcome.CERTIFIED_NEGATIVE


def test_unaccounted_impurity_phase_triggers_high_rwp_ambiguity() -> None:
    grid = _sample_grid()
    prec_1 = build_gaussian_spectrum(grid, [(18.0, 1.0, 1.8)])
    target = build_gaussian_spectrum(grid, [(36.0, 1.0, 2.0)])
    mystery_polymorph = build_gaussian_spectrum(grid, [(50.0, 1.3, 2.2), (62.0, 1.1, 2.0)])

    observed = [0.8 * target[i] + 1.0 * mystery_polymorph[i] for i in range(len(grid))]
    res = deconvolve_spectrum(observed, [prec_1], target, min_hit_amplitude=0.25)

    assert res.spectral_residual_permille > 400
    assert attribute_assay_outcome(res.to_assay_telemetry()) is AttributionOutcome.AMBIGUOUS_CHARACTERIZATION


def test_numpy_and_pure_stdlib_nnls_backends_match_to_machine_precision() -> None:
    pytest.importorskip("numpy")
    grid = _sample_grid()
    p1 = build_gaussian_spectrum(grid, [(20.0, 1.0, 2.2), (38.0, 0.5, 2.0)])
    p2 = build_gaussian_spectrum(grid, [(30.0, 0.9, 2.0), (50.0, 0.7, 2.2)])
    target = build_gaussian_spectrum(grid, [(38.0, 1.0, 2.1), (56.0, 0.9, 2.1)])

    rng = random.Random(99)
    observed = [
        max(0.0, 0.45 * p1[i] + 0.70 * p2[i] + 0.65 * target[i] + rng.gauss(0.0, 0.015))
        for i in range(len(grid))
    ]

    res_np = deconvolve_spectrum(observed, [p1, p2], target, use_numpy=True)
    res_py = deconvolve_spectrum(observed, [p1, p2], target, use_numpy=False)

    assert res_np.used_numpy is True
    assert res_py.used_numpy is False
    assert res_np.target_amplitude == pytest.approx(res_py.target_amplitude, abs=1e-8)
    for w_np, w_py in zip(res_np.precursor_weights, res_py.precursor_weights):
        assert w_np == pytest.approx(w_py, abs=1e-8)
    assert res_np.precursor_overlap_permille == res_py.precursor_overlap_permille
    assert res_np.spectral_residual_permille == res_py.spectral_residual_permille


def test_nnls_nonnegativity_and_kkt_across_random_trials() -> None:
    grid = _sample_grid()
    p1 = build_gaussian_spectrum(grid, [(18.0, 1.0, 2.5), (36.0, 0.6, 2.5)])
    p2 = build_gaussian_spectrum(grid, [(28.0, 1.0, 2.5), (46.0, 0.8, 2.5)])
    target = build_gaussian_spectrum(grid, [(36.0, 1.0, 2.5), (54.0, 1.0, 2.5)])

    rng = random.Random(1234)
    for _ in range(12):
        y = [
            max(0.0, rng.uniform(-0.2, 1.2) * p1[i] + rng.uniform(0.0, 1.0) * target[i])
            for i in range(len(grid))
        ]
        res = deconvolve_spectrum(y, [p1, p2], target, use_numpy=False)
        assert all(w >= 0.0 for w in res.precursor_weights)
        assert res.target_amplitude >= 0.0
        assert res.kkt_max_violation <= 1e-8


# --- Issue #1: Collinear polymorph convergence, rank-deficient duplicates, and weighted NNLS ---


@pytest.mark.parametrize("use_np", [False, True])
def test_issue1_nearly_collinear_polymorph_pure_target_converges_cleanly(use_np: bool) -> None:
    if use_np:
        pytest.importorskip("numpy")
    grid = [15.0 + 0.1 * i for i in range(500)]
    p1 = build_gaussian_spectrum(grid, [(25.0, 1.0, 0.45), (40.0, 0.8, 0.45)])
    target_col = build_gaussian_spectrum(grid, [(25.008, 1.0, 0.45), (40.008, 0.8, 0.45)])

    obs_pure_target = [1.0 * v for v in target_col]
    res = deconvolve_spectrum(
        obs_pure_target, [p1], target_col, max_iter=2000, use_numpy=use_np
    )

    assert res.converged is True
    assert res.max_basis_cosine > 0.9999
    assert "NEAR_COLLINEAR_BASIS" in res.diagnostic_notes
    assert res.kkt_max_violation <= 1e-10
    assert res.target_amplitude == pytest.approx(1.0, abs=1e-5)
    assert res.precursor_weights[0] == pytest.approx(0.0, abs=1e-5)
    assert res.precursor_overlap_permille == 0
    assert diagnose_assay(res.to_assay_telemetry()).outcome is AttributionOutcome.CERTIFIED_POSITIVE


def test_issue1_rank_deficient_duplicate_precursors_and_weighted_nnls_parity() -> None:
    pytest.importorskip("numpy")
    grid = [15.0 + 0.1 * i for i in range(300)]
    p1 = build_gaussian_spectrum(grid, [(21.0, 1.0, 0.5), (35.0, 0.6, 0.5)])
    target = build_gaussian_spectrum(grid, [(35.0, 1.0, 0.5), (41.0, 0.9, 0.5)])

    # Rank-deficient duplicate precursor columns [p1, p1]
    obs_dup = [0.75 * p1[i] + 0.90 * target[i] for i in range(len(grid))]
    res_dup_py = deconvolve_spectrum(obs_dup, [p1, p1], target, use_numpy=False)
    res_dup_np = deconvolve_spectrum(obs_dup, [p1, p1], target, use_numpy=True)
    assert res_dup_py.converged and res_dup_np.converged
    assert res_dup_py.kkt_max_violation <= 1e-10
    assert sum(res_dup_py.precursor_weights) == pytest.approx(0.75, abs=1e-5)
    assert res_dup_py.target_amplitude == pytest.approx(0.90, abs=1e-5)
    assert res_dup_np.target_amplitude == pytest.approx(0.90, abs=1e-5)

    # Heteroscedastic Poisson Rietveld weights: noisy spike on low-weight channel
    obs_het = [0.60 * p1[i] + 0.85 * target[i] for i in range(len(grid))]
    # Corrupt precursor-peak channels with disturbance where weight is small (high count variance)
    weights = [1.0 / (0.05 + p1[i] * 8.0) for i in range(len(grid))]
    for i in range(50, 70):
        obs_het[i] += 0.25 * p1[i]

    res_w_py = deconvolve_spectrum(obs_het, [p1], target, weights=weights, use_numpy=False)
    res_w_np = deconvolve_spectrum(obs_het, [p1], target, weights=weights, use_numpy=True)
    res_unw = deconvolve_spectrum(obs_het, [p1], target, weights=None, use_numpy=False)

    assert res_w_py.target_amplitude == pytest.approx(res_w_np.target_amplitude, abs=1e-8)
    assert res_w_py.precursor_weights[0] == pytest.approx(res_w_np.precursor_weights[0], abs=1e-8)

    def _weighted_rss(r: object) -> float:
        assert hasattr(r, "precursor_weights") and hasattr(r, "target_amplitude")
        c0 = r.precursor_weights[0]
        bt = r.target_amplitude
        return sum(
            weights[i] * (obs_het[i] - (c0 * p1[i] + bt * target[i])) ** 2
            for i in range(len(grid))
        )

    assert _weighted_rss(res_w_py) < _weighted_rss(res_unw)


# --- Issue #2: Bragg 2-theta shift & Vegard solid-solution lattice strain alignment ---


def test_issue2_vegard_peak_shift_recovers_hit_while_shifted_ghost_and_large_shift_quarantine() -> None:
    grid = [15.0 + 0.1 * i for i in range(500)]
    p_clean = build_gaussian_spectrum(grid, [(18.0, 1.0, 0.42)])
    p_ghost = build_gaussian_spectrum(
        grid, [(33.2, 0.96, 0.44), (41.5, 0.74, 0.44), (22.0, 0.40, 0.42)]
    )
    t_ref = build_gaussian_spectrum(
        grid, [(33.2, 1.0, 0.42), (41.5, 0.75, 0.42), (54.0, 0.60, 0.42)]
    )

    # 1. Pure novel solid-solution target shifted by +0.35 deg 2-theta
    obs_vegard = build_gaussian_spectrum(
        grid, [(33.55, 1.0, 0.42), (41.85, 0.75, 0.42), (54.35, 0.60, 0.42)]
    )
    res_vegard = deconvolve_spectrum(
        obs_vegard,
        [p_clean],
        t_ref,
        min_hit_amplitude=0.25,
        two_theta_grid=grid,
        max_peak_shift_deg=0.5,
    )
    assert res_vegard.spectral_residual_permille <= 50
    assert res_vegard.target_amplitude == pytest.approx(1.0, abs=0.03)
    assert res_vegard.fitted_shift_deg == pytest.approx(0.35, abs=0.05)
    assert (
        diagnose_assay(res_vegard.to_assay_telemetry()).outcome
        is AttributionOutcome.CERTIFIED_POSITIVE
    )

    # Also check allow_peak_shift_bins without two_theta_grid
    res_bins = deconvolve_spectrum(
        obs_vegard,
        [p_clean],
        t_ref,
        min_hit_amplitude=0.25,
        allow_peak_shift_bins=5,
    )
    assert res_bins.spectral_residual_permille <= 50
    assert (
        diagnose_assay(res_bins.to_assay_telemetry()).outcome
        is AttributionOutcome.CERTIFIED_POSITIVE
    )

    # 2. +0.35 deg shifted unreacted ghost precursor MUST still trigger overlap quarantine
    obs_shifted_ghost = build_gaussian_spectrum(
        grid, [(33.55, 0.96, 0.44), (41.85, 0.74, 0.44), (22.35, 0.40, 0.42)]
    )
    res_ghost = deconvolve_spectrum(
        obs_shifted_ghost,
        [p_clean, p_ghost],
        t_ref,
        min_hit_amplitude=0.25,
        two_theta_grid=grid,
        max_peak_shift_deg=0.5,
    )
    assert res_ghost.precursor_overlap_permille > 150
    assert (
        diagnose_assay(res_ghost.to_assay_telemetry()).outcome
        is AttributionOutcome.AMBIGUOUS_CHARACTERIZATION
    )

    # 3. Large +2.0 deg shift (distinct polymorph) beyond max_peak_shift_deg=0.5 triggers high R_wp
    obs_distinct_poly = build_gaussian_spectrum(
        grid, [(35.2, 1.0, 0.42), (43.5, 0.75, 0.42), (56.0, 0.60, 0.42)]
    )
    res_poly = deconvolve_spectrum(
        obs_distinct_poly,
        [p_clean],
        t_ref,
        min_hit_amplitude=0.25,
        two_theta_grid=grid,
        max_peak_shift_deg=0.5,
    )
    rep_poly = diagnose_assay(res_poly.to_assay_telemetry())
    assert rep_poly.outcome is AttributionOutcome.AMBIGUOUS_CHARACTERIZATION
    assert "HIGH_SPECTRAL_RWP_RESIDUAL" in rep_poly.violated_characterization_clauses


# --- Issue #3 & #5: Amorphous background halo, NaN/Inf guards, zero-weight guard, empirical SNR ---


def test_issue3_amorphous_background_halo_and_fluorescence_saturation_guard() -> None:
    grid = [15.0 + 0.1 * i for i in range(500)]
    p_clean = build_gaussian_spectrum(grid, [(18.0, 1.0, 0.42)])
    t_ref = build_gaussian_spectrum(grid, [(33.2, 1.0, 0.42), (41.5, 0.75, 0.42)])

    n = len(grid)
    obs_halo = [
        0.08 + 0.03 * (i / (n - 1)) * (1.0 - i / (n - 1)) for i in range(n)
    ]
    res_halo = deconvolve_spectrum(
        obs_halo,
        [p_clean],
        t_ref,
        min_hit_amplitude=0.25,
        background_poly_degree=2,
    )
    assert res_halo.spectral_residual_permille <= 50
    assert res_halo.target_amplitude == pytest.approx(0.0, abs=1e-4)
    assert res_halo.precursor_overlap_permille == 0
    assert (
        diagnose_assay(res_halo.to_assay_telemetry()).outcome
        is AttributionOutcome.CERTIFIED_NEGATIVE
    )

    # Extreme fluorescence saturation is quarantined despite background basis
    obs_saturated = [10.0] * n
    res_sat = deconvolve_spectrum(
        obs_saturated,
        [p_clean],
        t_ref,
        min_hit_amplitude=0.25,
        background_poly_degree=2,
    )
    assert "EXCESSIVE_BACKGROUND_FLUORESCENCE" in res_sat.diagnostic_notes
    assert (
        diagnose_assay(res_sat.to_assay_telemetry()).outcome
        is AttributionOutcome.AMBIGUOUS_CHARACTERIZATION
    )


def test_issue3_nan_inf_and_zero_weight_guards_and_issue5_empirical_hf_noise_floor() -> None:
    grid = [15.0 + 0.1 * i for i in range(100)]
    p_clean = build_gaussian_spectrum(grid, [(18.0, 1.0, 0.42)])
    t_ref = build_gaussian_spectrum(grid, [(21.0, 1.0, 0.42)])

    # NaN / Inf in observed_y, precursor_spectra, target_spectrum, weights, two_theta_grid
    with pytest.raises(ValueError, match="finite"):
        deconvolve_spectrum([float("nan")] * len(grid), [p_clean], t_ref)
    with pytest.raises(ValueError, match="finite"):
        deconvolve_spectrum([float("inf")] * len(grid), [p_clean], t_ref)
    with pytest.raises(ValueError, match="finite"):
        deconvolve_spectrum([0.1] * len(grid), [[float("nan")] * len(grid)], t_ref)
    with pytest.raises(ValueError, match="finite"):
        deconvolve_spectrum([0.1] * len(grid), [p_clean], [float("-inf")] * len(grid))
    with pytest.raises(ValueError, match="finite"):
        deconvolve_spectrum(
            [0.1] * len(grid), [p_clean], t_ref, weights=[float("nan")] * len(grid)
        )

    # All-zero weights or weights orthogonal to target_spectrum
    with pytest.raises(ValueError, match="positive"):
        deconvolve_spectrum([0.5] * len(grid), [p_clean], t_ref, weights=[0.0] * len(grid))
    orth_weights = [1.0 if v == 0.0 else 0.0 for v in t_ref]
    if sum(orth_weights) > 0:
        with pytest.raises(ValueError, match="positive support on target_spectrum"):
            deconvolve_spectrum([0.5] * len(grid), [p_clean], t_ref, weights=orth_weights)

    # Issue #5: High-frequency empirical residual RMS degrades snr_tenths < 30 even if noise_sigma is tiny
    target_norm = math.sqrt(sum(v * v for v in t_ref))
    floor_int = 0.25 * target_norm
    # Alternating channel noise with empirical HF RMS > floor_int / 3.0
    amp = 0.6 * floor_int
    noisy_obs = [amp if (i % 2 == 0) else 0.0 for i in range(len(grid))]
    res_hf = deconvolve_spectrum(
        noisy_obs, [p_clean], t_ref, min_hit_amplitude=0.25, noise_sigma=0.005
    )
    assert res_hf.empirical_noise_rms > floor_int / 3.0
    assert res_hf.snr_tenths < 30
    assert (
        diagnose_assay(res_hf.to_assay_telemetry()).outcome
        is AttributionOutcome.AMBIGUOUS_CHARACTERIZATION
    )


def test_deconvolve_input_validation_errors() -> None:
    with pytest.raises(ValueError, match="at least 2 channels"):
        deconvolve_spectrum([1.0], [], [1.0])
    with pytest.raises(ValueError, match="target_spectrum length"):
        deconvolve_spectrum([1.0, 2.0], [], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="precursor_spectra\\[0\\] length"):
        deconvolve_spectrum([1.0, 2.0], [[1.0]], [1.0, 2.0])
    with pytest.raises(ValueError, match="non-zero L2 norm"):
        deconvolve_spectrum([1.0, 2.0], [], [0.0, 0.0])
    with pytest.raises(ValueError, match="Peak sigma must be positive"):
        build_gaussian_spectrum([10.0, 20.0], [(15.0, 1.0, 0.0)])
