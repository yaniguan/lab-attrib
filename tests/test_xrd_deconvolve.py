"""Tests for NNLS Precursor Peak-Overlap & R_wp Deconvolver (`xrd_deconvolve.py`)."""

from __future__ import annotations

import random

import pytest

from lab_attrib.blame import (
    AttributionOutcome,
    attribute_assay_outcome,
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
    assert res.kkt_max_violation <= 1e-8

    telemetry = res.to_assay_telemetry()
    assert attribute_assay_outcome(telemetry) is AttributionOutcome.CERTIFIED_POSITIVE


def test_alab_ghost_peak_blocks_false_positive_despite_high_naive_cosine() -> None:
    grid = _sample_grid()
    prec_nonoverlap = build_gaussian_spectrum(grid, [(16.0, 1.0, 1.8)])
    # Precursor B shares both main Bragg peaks with putative novel target!
    prec_ghost = build_gaussian_spectrum(
        grid, [(34.0, 0.96, 2.1), (52.0, 0.90, 2.1), (22.0, 0.35, 1.7)]
    )
    target = build_gaussian_spectrum(grid, [(34.0, 1.0, 2.0), (52.0, 0.92, 2.0)])

    # True reaction mixture contains ZERO novel target — only unreacted precursors!
    observed = [0.20 * prec_nonoverlap[i] + 1.10 * prec_ghost[i] for i in range(len(grid))]
    res = deconvolve_spectrum(observed, [prec_nonoverlap, prec_ghost], target, min_hit_amplitude=0.25)

    assert res.naive_peak_match_cosine > 0.88
    assert res.naive_claims_hit is True
    assert res.precursor_overlap_permille > 700

    # Even if an upstream peak caller claims target_signal_present=True, blame calculus blocks it!
    telemetry = res.to_assay_telemetry()
    assert attribute_assay_outcome(telemetry) is AttributionOutcome.AMBIGUOUS_CHARACTERIZATION


def test_clean_non_overlapping_precursor_only_yields_certified_negative() -> None:
    grid = _sample_grid()
    prec_1 = build_gaussian_spectrum(grid, [(18.0, 1.0, 1.8), (44.0, 0.8, 1.8)])
    target = build_gaussian_spectrum(grid, [(34.0, 1.0, 2.0), (58.0, 0.9, 2.0)])

    observed = [0.95 * v for v in prec_1]
    res = deconvolve_spectrum(observed, [prec_1], target, min_hit_amplitude=0.25, noise_sigma=0.03)

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
    res = deconvolve_spectrum(observed, [prec_1], target, min_hit_amplitude=0.25, noise_sigma=0.03)

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
        y = [max(0.0, rng.uniform(-0.2, 1.2) * p1[i] + rng.uniform(0.0, 1.0) * target[i]) for i in range(len(grid))]
        res = deconvolve_spectrum(y, [p1, p2], target, use_numpy=False)
        assert all(w >= 0.0 for w in res.precursor_weights)
        assert res.target_amplitude >= 0.0
        assert res.kkt_max_violation <= 1e-8


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
