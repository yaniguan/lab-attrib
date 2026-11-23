"""Non-Negative Least-Squares (NNLS) Precursor Peak-Overlap & R_wp Deconvolver.

Resolves the `A-Lab` XRD / optical ghost-phase false-positive failure mode:
unreacted starting precursors frequently share diffraction or spectral peaks with a
putative novel target phase. Naive cosine or top-k peak matchers misclassify precursor
peaks as target formation.

This module solves the non-negative multiphase decomposition:
    min_{c >= 0, beta >= 0} 0.5 * || y - (sum_j c_j p_j + beta * s_target) ||_2^2
via cyclic Projected Coordinate Descent (PCD) with exact KKT stopping criteria,
supporting both NumPy acceleration and a zero-dependency pure-Python stdlib fallback.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from lab_attrib.blame import AssayTelemetry

try:
    import numpy as _np  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    _np = None


@dataclass(frozen=True)
class DeconvolutionResult:
    """Structured multiphase NNLS spectral deconvolution output."""

    precursor_weights: tuple[float, ...]
    target_amplitude: float
    precursor_overlap_permille: int
    spectral_residual_permille: int
    snr_tenths: int
    target_signal_present: bool
    naive_peak_match_cosine: float
    naive_claims_hit: bool
    kkt_max_violation: float
    iterations: int
    used_numpy: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "precursor_weights": [round(w, 6) for w in self.precursor_weights],
            "target_amplitude": round(self.target_amplitude, 6),
            "precursor_overlap_permille": self.precursor_overlap_permille,
            "spectral_residual_permille": self.spectral_residual_permille,
            "snr_tenths": self.snr_tenths,
            "target_signal_present": self.target_signal_present,
            "naive_peak_match_cosine": round(self.naive_peak_match_cosine, 6),
            "naive_claims_hit": self.naive_claims_hit,
            "kkt_max_violation": round(self.kkt_max_violation, 12),
            "iterations": self.iterations,
            "used_numpy": self.used_numpy,
        }

    def to_assay_telemetry(
        self,
        *,
        dispense_error_permille: int = 12,
        aspiration_clean: bool = True,
        incubation_window_clean: bool = True,
        seal_intact: bool = True,
        positive_control_passed: bool = True,
        blank_uncontaminated: bool = True,
        detector_calibrated: bool = True,
    ) -> AssayTelemetry:
        """Lift deconvolved analytical metrics + wet-lab QC into `AssayTelemetry`."""
        return AssayTelemetry(
            dispense_error_permille=dispense_error_permille,
            aspiration_clean=aspiration_clean,
            incubation_window_clean=incubation_window_clean,
            seal_intact=seal_intact,
            positive_control_passed=positive_control_passed,
            blank_uncontaminated=blank_uncontaminated,
            detector_calibrated=detector_calibrated,
            precursor_overlap_permille=self.precursor_overlap_permille,
            spectral_residual_permille=self.spectral_residual_permille,
            snr_tenths=self.snr_tenths,
            target_signal_present=self.target_signal_present,
        )


def build_gaussian_spectrum(
    two_theta_grid: Sequence[float],
    peaks: Sequence[tuple[float, float, float]],
) -> list[float]:
    """Synthesize a non-negative diffraction/optical profile from `(center, height, sigma)` peaks."""
    spectrum: list[float] = []
    for x in two_theta_grid:
        val = 0.0
        for center, height, sigma in peaks:
            if sigma <= 0.0:
                raise ValueError(f"Peak sigma must be positive, got {sigma}")
            z = (x - center) / sigma
            val += max(0.0, height) * math.exp(-0.5 * z * z)
        spectrum.append(val)
    return spectrum


def _solve_nnls_stdlib(
    columns: list[list[float]],
    y: list[float],
    tol: float,
    max_iter: int,
) -> tuple[list[float], list[float], float, int]:
    """Pure-Python cyclic Projected Coordinate Descent NNLS solver."""
    m = len(y)
    p = len(columns)
    scales: list[float] = []
    norm_cols: list[list[float]] = []
    for col in columns:
        sq = sum(v * v for v in col)
        norm = math.sqrt(sq)
        if norm > 1e-15:
            scales.append(norm)
            norm_cols.append([v / norm for v in col])
        else:
            scales.append(0.0)
            norm_cols.append([0.0] * m)

    # Precompute Gram matrix G = U^T U and linear term b = U^T y on unit-norm columns
    gram: list[list[float]] = [[0.0] * p for _ in range(p)]
    rhs: list[float] = [0.0] * p
    for j in range(p):
        uj = norm_cols[j]
        rhs[j] = sum(uj[i] * y[i] for i in range(m))
        for k in range(j, p):
            uk = norm_cols[k]
            g_jk = sum(uj[i] * uk[i] for i in range(m))
            gram[j][k] = g_jk
            gram[k][j] = g_jk

    z = [0.0] * p
    # Maintain g = G z - b (gradient of 0.5 * ||y - U z||^2)
    grad = [-b_j for b_j in rhs]

    iters = 0
    kkt_violation = float("inf")
    for it in range(1, max_iter + 1):
        iters = it
        max_step = 0.0
        for j in range(p):
            if scales[j] == 0.0:
                continue
            old_zj = z[j]
            # Unit column -> G[j][j] == 1.0
            new_zj = max(0.0, old_zj - grad[j])
            delta = new_zj - old_zj
            if abs(delta) > 0.0:
                z[j] = new_zj
                row_j = gram[j]
                for k in range(p):
                    grad[k] += row_j[k] * delta
                if abs(delta) > max_step:
                    max_step = abs(delta)

        # Compute exact KKT optimality violation:
        # z_j >= 0, grad_j >= -tol, and if z_j > 0 then |grad_j| <= tol
        kkt_violation = 0.0
        for j in range(p):
            if scales[j] == 0.0:
                continue
            gj = grad[j]
            viol = abs(gj) if z[j] > 1e-13 else max(0.0, -gj)
            if viol > kkt_violation:
                kkt_violation = viol
        if kkt_violation <= tol and max_step <= tol:
            break

    weights = [z[j] / scales[j] if scales[j] > 0.0 else 0.0 for j in range(p)]
    fitted = [0.0] * m
    for j in range(p):
        zj = z[j]
        if zj > 0.0:
            uj = norm_cols[j]
            for i in range(m):
                fitted[i] += zj * uj[i]
    return weights, fitted, kkt_violation, iters


def _solve_nnls_numpy(
    columns: list[list[float]],
    y: list[float],
    tol: float,
    max_iter: int,
) -> tuple[list[float], list[float], float, int]:
    """NumPy-accelerated cyclic Projected Coordinate Descent NNLS solver."""
    assert _np is not None
    mat = _np.asarray(columns, dtype=_np.float64).T  # shape (M, P)
    y_vec = _np.asarray(y, dtype=_np.float64)
    scales = _np.linalg.norm(mat, axis=0)
    active = scales > 1e-15
    u_mat = _np.zeros_like(mat)
    u_mat[:, active] = mat[:, active] / scales[active]

    gram = u_mat.T @ u_mat
    rhs = u_mat.T @ y_vec
    p = mat.shape[1]
    z = _np.zeros(p, dtype=_np.float64)
    grad = -rhs.copy()

    iters = 0
    kkt_violation = float("inf")
    for it in range(1, max_iter + 1):
        iters = it
        max_step = 0.0
        for j in range(p):
            if not active[j]:
                continue
            old_zj = float(z[j])
            new_zj = max(0.0, old_zj - float(grad[j]))
            delta = new_zj - old_zj
            if delta != 0.0:
                z[j] = new_zj
                grad += gram[j] * delta
                if abs(delta) > max_step:
                    max_step = abs(delta)

        kkt_violation = 0.0
        for j in range(p):
            if not active[j]:
                continue
            gj = float(grad[j])
            viol = abs(gj) if z[j] > 1e-13 else max(0.0, -gj)
            if viol > kkt_violation:
                kkt_violation = viol
        if kkt_violation <= tol and max_step <= tol:
            break

    weights_arr = _np.zeros(p, dtype=_np.float64)
    weights_arr[active] = z[active] / scales[active]
    fitted_arr = u_mat @ z
    return (
        [float(v) for v in weights_arr.tolist()],
        [float(v) for v in fitted_arr.tolist()],
        kkt_violation,
        iters,
    )


def deconvolve_spectrum(
    observed_y: Sequence[float],
    precursor_spectra: Sequence[Sequence[float]],
    target_spectrum: Sequence[float],
    *,
    min_hit_amplitude: float = 0.20,
    noise_sigma: float = 0.025,
    naive_cosine_threshold: float = 0.70,
    weights: Sequence[float] | None = None,
    tol: float = 1e-10,
    max_iter: int = 2000,
    use_numpy: bool | None = None,
) -> DeconvolutionResult:
    """Deconvolve `observed_y` into non-negative precursor phases + target phase.

    Parameters
    ----------
    observed_y:
        Measured 1D diffraction / optical spectrum of length `M >= 2`.
    precursor_spectra:
        Sequence of `K >= 0` starting-precursor reference spectra, each of length `M`.
    target_spectrum:
        Putative novel target phase reference spectrum of length `M`.
    min_hit_amplitude:
        Minimum fitted target phase scale `beta` required to declare `target_signal_present`
        and reference detection-limit scale used for assay SNR and precursor shadowing.
    noise_sigma:
        Estimated per-channel detector noise standard deviation `sigma > 0`.
    naive_cosine_threshold:
        Cosine similarity threshold used by a naive peak matcher (demonstrating A-Lab false hits).
    weights:
        Optional positive channel weights `u_i` for weighted profile residual `R_wp`.
    use_numpy:
        `True` forces NumPy, `False` forces pure-Python stdlib, `None` uses NumPy if available.
    """
    y = [max(0.0, float(v)) for v in observed_y]
    m = len(y)
    if m < 2:
        raise ValueError("Spectrum must contain at least 2 channels")
    s_target = [max(0.0, float(v)) for v in target_spectrum]
    if len(s_target) != m:
        raise ValueError("target_spectrum length must match observed_y length")

    prec_cols: list[list[float]] = []
    for idx, p_spec in enumerate(precursor_spectra):
        col = [max(0.0, float(v)) for v in p_spec]
        if len(col) != m:
            raise ValueError(f"precursor_spectra[{idx}] length {len(col)} != {m}")
        prec_cols.append(col)

    if min_hit_amplitude <= 0.0:
        raise ValueError("min_hit_amplitude must be positive")
    if noise_sigma <= 0.0:
        raise ValueError("noise_sigma must be positive")

    target_norm = math.sqrt(sum(v * v for v in s_target))
    if target_norm <= 1e-15:
        raise ValueError("target_spectrum must have non-zero L2 norm")

    w_profile = [1.0] * m if weights is None else [max(0.0, float(v)) for v in weights]
    if len(w_profile) != m:
        raise ValueError("weights length must match observed_y length")

    all_cols = [*prec_cols, s_target]
    run_np = (_np is not None) if use_numpy is None else bool(use_numpy)
    if run_np and _np is None:  # pragma: no cover
        raise RuntimeError("use_numpy=True requested, but numpy is not installed")

    if run_np:
        coeff, fitted, kkt_viol, iters = _solve_nnls_numpy(all_cols, y, tol=tol, max_iter=max_iter)
    else:
        coeff, fitted, kkt_viol, iters = _solve_nnls_stdlib(all_cols, y, tol=tol, max_iter=max_iter)

    prec_weights = tuple(coeff[:-1])
    beta_target = coeff[-1]

    # Reconstruct precursor-only and target-only components
    y_prec = [0.0] * m
    for c_j, p_col in zip(prec_weights, prec_cols):
        if c_j > 0.0:
            for i in range(m):
                y_prec[i] += c_j * p_col[i]
    y_target = [beta_target * v for v in s_target]

    # 1. Weighted profile residual R_wp in permille:
    #    For near-empty background baselines (||y||_w < reference floor), normalize by the
    #    minimum detectable target profile energy so pure Gaussian zero-baseline noise does
    #    not blow up 0 / 0 into artificial R_wp = 1000 permille.
    weighted_err_sq = sum(w_profile[i] * (y[i] - fitted[i]) ** 2 for i in range(m))
    weighted_obs_sq = sum(w_profile[i] * (y[i] ** 2) for i in range(m))
    min_ref_sq = sum(w_profile[i] * ((min_hit_amplitude * s_target[i]) ** 2) for i in range(m))
    denom_sq = max(weighted_obs_sq, min_ref_sq, 1e-18)
    r_wp = math.sqrt(weighted_err_sq / denom_sq)
    spectral_residual_permille = min(1000, max(0, int(round(1000.0 * r_wp))))

    # 2. Precursor peak-overlap shadowing index on target channels in permille:
    #    rho_overlap = <y_prec, s_hat> / (<y_prec, s_hat> + max(<y_target, s_hat>, I_floor))
    s_unit = [v / target_norm for v in s_target]
    prec_on_target = sum(y_prec[i] * s_unit[i] for i in range(m))
    target_on_target = sum(y_target[i] * s_unit[i] for i in range(m))
    floor_intensity = min_hit_amplitude * target_norm
    overlap_ratio = prec_on_target / (prec_on_target + max(target_on_target, floor_intensity))
    precursor_overlap_permille = min(1000, max(0, int(round(1000.0 * overlap_ratio))))

    # 3. Assay resolvability / detection-power SNR in tenths:
    #    Uses max(fitted target projection, minimum detectable target projection) / noise_sigma.
    #    When noise_sigma is small relative to the detection threshold, both hits and clean
    #    baselines carry high statistical power (SNR >= 3.0 -> snr_tenths >= 30).
    effective_signal_norm = max(target_on_target, floor_intensity)
    assay_snr = effective_signal_norm / noise_sigma
    snr_tenths = max(0, int(round(10.0 * assay_snr)))

    # 4. Naive peak-matcher cosine similarity between raw y and s_target
    obs_norm = math.sqrt(sum(v * v for v in y))
    if obs_norm > 1e-12:
        naive_cos = sum(y[i] * s_unit[i] for i in range(m)) / obs_norm
    else:
        naive_cos = 0.0
    naive_cos = max(0.0, min(1.0, naive_cos))

    target_present = beta_target >= min_hit_amplitude
    naive_hit = naive_cos >= naive_cosine_threshold and obs_norm >= 0.5 * floor_intensity

    return DeconvolutionResult(
        precursor_weights=prec_weights,
        target_amplitude=beta_target,
        precursor_overlap_permille=precursor_overlap_permille,
        spectral_residual_permille=spectral_residual_permille,
        snr_tenths=snr_tenths,
        target_signal_present=target_present,
        naive_peak_match_cosine=naive_cos,
        naive_claims_hit=naive_hit,
        kkt_max_violation=kkt_viol,
        iterations=iters,
        used_numpy=run_np,
    )
