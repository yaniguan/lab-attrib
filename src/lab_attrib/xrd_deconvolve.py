"""Non-Negative Least-Squares (NNLS) Precursor Peak-Overlap & R_wp Deconvolver.

Resolves the `A-Lab` XRD / optical ghost-phase false-positive failure mode:
unreacted starting precursors frequently share diffraction or spectral peaks with a
putative novel target phase. Naive cosine or top-k peak matchers misclassify precursor
peaks as target formation.

This module solves the weighted non-negative multiphase decomposition:
    min_{c >= 0, beta >= 0, b >= 0}
        0.5 * sum_i w_i * (y_i - (sum_j c_j p_j(i) + beta * s_target(i) + y_bg(i)))^2
via an active-set Lawson-Hanson NNLS solver backed by Householder QR least-squares
(with cyclic Projected Coordinate Descent fallback) and exact KKT stopping criteria,
supporting both NumPy acceleration and a zero-dependency pure-Python stdlib backend.

Also supports bounded Bragg 2-theta peak-shift / Vegard solid-solution lattice-strain
alignment, non-negative Bernstein polynomial amorphous background subtraction, strict
finite-value validation, and empirical high-frequency residual noise-floor estimation.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Any, Sequence

from lab_attrib.blame import AssayTelemetry, SpectralFitMethod

try:
    import numpy as _np  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    _np = None


def _require_finite_float(value: Any, name: str) -> float:
    """Convert `value` to `float` while rejecting booleans, `NaN`, and `+-Infinity`."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a real number, got bool {value!r}")
    val = float(value)
    if not math.isfinite(val):
        raise ValueError(f"{name} must be finite (no NaN or Infinity), got {value!r}")
    return val


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
    converged: bool = True
    fitted_shift_deg: float = 0.0
    max_basis_cosine: float = 0.0
    background_weights: tuple[float, ...] = ()
    empirical_noise_rms: float = 0.0
    diagnostic_notes: tuple[str, ...] = ()

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
            "converged": self.converged,
            "fitted_shift_deg": round(self.fitted_shift_deg, 6),
            "max_basis_cosine": round(self.max_basis_cosine, 6),
            "background_weights": [round(b, 6) for b in self.background_weights],
            "empirical_noise_rms": round(self.empirical_noise_rms, 6),
            "diagnostic_notes": list(self.diagnostic_notes),
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
        fit_method: SpectralFitMethod | str = SpectralFitMethod.STICK_NNLS,
    ) -> AssayTelemetry:
        """Lift deconvolved analytical metrics + wet-lab QC into `AssayTelemetry`."""
        effective_snr = self.snr_tenths if self.converged else 0
        resolved_fit = (
            fit_method
            if isinstance(fit_method, SpectralFitMethod)
            else SpectralFitMethod(str(fit_method))
        )
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
            snr_tenths=effective_snr,
            target_signal_present=self.target_signal_present,
            fit_method=resolved_fit,
        )


def build_gaussian_spectrum(
    two_theta_grid: Sequence[float],
    peaks: Sequence[tuple[float, float, float]],
) -> list[float]:
    """Synthesize a non-negative diffraction/optical profile from `(center, height, sigma)` peaks."""
    validated_peaks: list[tuple[float, float, float]] = []
    for idx, (center, height, sigma) in enumerate(peaks):
        c_val = _require_finite_float(center, f"peaks[{idx}].center")
        h_val = _require_finite_float(height, f"peaks[{idx}].height")
        s_val = _require_finite_float(sigma, f"peaks[{idx}].sigma")
        if s_val <= 0.0:
            raise ValueError(f"Peak sigma must be positive, got {sigma}")
        validated_peaks.append((c_val, max(0.0, h_val), s_val))

    spectrum: list[float] = []
    for idx, x_raw in enumerate(two_theta_grid):
        x = _require_finite_float(x_raw, f"two_theta_grid[{idx}]")
        val = 0.0
        for c_val, h_val, s_val in validated_peaks:
            z = (x - c_val) / s_val
            val += h_val * math.exp(-0.5 * z * z)
        spectrum.append(val)
    return spectrum


def _catmull_rom_warp(
    spectrum: Sequence[float],
    grid_x: Sequence[float],
    shift_deg: float,
    strain_scale: float = 0.0,
    *,
    is_angles_deg: bool = True,
) -> list[float]:
    """Evaluate `spectrum(x - shift - strain * tan(theta))` via non-negative Catmull-Rom interpolation."""
    m = len(spectrum)
    if shift_deg == 0.0 and strain_scale == 0.0:
        return [float(v) for v in spectrum]

    x0 = grid_x[0]
    x_end = grid_x[-1]
    out: list[float] = []
    deg_to_rad_half = math.pi / 360.0

    for i in range(m):
        xi = grid_x[i]
        strain_offset = (
            strain_scale * math.tan(xi * deg_to_rad_half)
            if (strain_scale != 0.0 and is_angles_deg)
            else strain_scale * ((xi - x0) / max(1e-9, x_end - x0))
        )
        query = xi - shift_deg - strain_offset
        if query < x0 or query > x_end:
            out.append(0.0)
            continue

        hi = bisect.bisect_right(grid_x, query)
        if hi <= 0:
            out.append(max(0.0, float(spectrum[0])))
            continue
        if hi >= m:
            out.append(max(0.0, float(spectrum[m - 1])))
            continue

        i1 = hi - 1
        i2 = hi
        dx = grid_x[i2] - grid_x[i1]
        if dx <= 1e-15:
            out.append(max(0.0, float(spectrum[i1])))
            continue
        t = (query - grid_x[i1]) / dx

        y0 = float(spectrum[max(0, i1 - 1)])
        y1 = float(spectrum[i1])
        y2 = float(spectrum[i2])
        y3 = float(spectrum[min(m - 1, i2 + 1)])

        t2 = t * t
        t3 = t2 * t
        interp = 0.5 * (
            (2.0 * y1)
            + (-y0 + y2) * t
            + (2.0 * y0 - 5.0 * y1 + 4.0 * y2 - y3) * t2
            + (-y0 + 3.0 * y1 - 3.0 * y2 + y3) * t3
        )
        out.append(max(0.0, interp))
    return out


def _build_bernstein_background_columns(m: int, degree: int) -> list[list[float]]:
    """Construct `degree + 1` non-negative Bernstein basis polynomials on `[0, 1]`."""
    cols: list[list[float]] = []
    denom = float(max(1, m - 1))
    for k in range(degree + 1):
        binom = float(math.comb(degree, k))
        col: list[float] = []
        for i in range(m):
            u = float(i) / denom
            val = binom * (u ** k) * ((1.0 - u) ** (degree - k))
            col.append(val)
        cols.append(col)
    return cols


def _householder_lstsq_stdlib(
    cols_sub: list[list[float]],
    rhs_vec: list[float],
) -> tuple[list[float], bool]:
    """Solve full-rank `min_s ||U_sub s - rhs||_2` via Householder QR without squaring condition number."""
    k = len(cols_sub)
    m = len(rhs_vec)
    if k == 0:
        return [], True
    if k > m:
        return [0.0] * k, False

    a = [col[:] for col in cols_sub]
    b = rhs_vec[:]

    for col in range(k):
        col_vec = a[col]
        norm_sq = sum(col_vec[i] * col_vec[i] for i in range(col, m))
        norm_x = math.sqrt(norm_sq)
        if norm_x <= 1e-13:
            return [0.0] * k, False

        alpha_h = -math.copysign(norm_x, col_vec[col])
        v0 = col_vec[col] - alpha_h
        tail_sq = norm_sq - col_vec[col] * col_vec[col]
        v_norm_sq = v0 * v0 + tail_sq
        col_vec[col] = alpha_h

        if v_norm_sq <= 1e-30:
            continue

        inv_half = 2.0 / v_norm_sq
        for c2 in range(col + 1, k):
            target_col = a[c2]
            dot = v0 * target_col[col] + sum(
                col_vec[i] * target_col[i] for i in range(col + 1, m)
            )
            scale = inv_half * dot
            target_col[col] -= scale * v0
            for i in range(col + 1, m):
                target_col[i] -= scale * col_vec[i]

        dot_b = v0 * b[col] + sum(col_vec[i] * b[i] for i in range(col + 1, m))
        scale_b = inv_half * dot_b
        b[col] -= scale_b * v0
        for i in range(col + 1, m):
            b[i] -= scale_b * col_vec[i]

    sol = [0.0] * k
    for c in range(k - 1, -1, -1):
        diag = a[c][c]
        if abs(diag) <= 1e-12:
            return [0.0] * k, False
        rhs_c = b[c] - sum(a[c2][c] * sol[c2] for c2 in range(c + 1, k))
        sol[c] = rhs_c / diag
    return sol, True


def _eval_kkt_and_grad_stdlib(
    norm_cols: list[list[float]],
    scales: list[float],
    y: list[float],
    z: list[float],
) -> tuple[list[float], list[float], float]:
    """Compute fitted vector `U z`, residual-space gradient `-U^T (y - U z)`, and max KKT violation."""
    m = len(y)
    p = len(norm_cols)
    fitted = [0.0] * m
    for j in range(p):
        zj = z[j]
        if zj > 0.0 and scales[j] > 0.0:
            uj = norm_cols[j]
            for i in range(m):
                fitted[i] += zj * uj[i]

    res = [y[i] - fitted[i] for i in range(m)]
    grad = [0.0] * p
    kkt_violation = 0.0
    for j in range(p):
        if scales[j] == 0.0:
            continue
        uj = norm_cols[j]
        gj = -sum(uj[i] * res[i] for i in range(m))
        grad[j] = gj
        viol = abs(gj) if z[j] > 1e-13 else max(0.0, -gj)
        if viol > kkt_violation:
            kkt_violation = viol
    return fitted, grad, kkt_violation


def _solve_nnls_stdlib(
    columns: list[list[float]],
    y: list[float],
    tol: float,
    max_iter: int,
) -> tuple[list[float], list[float], float, int]:
    """Pure-Python Lawson-Hanson active-set NNLS solver (Householder QR + PCD fallback)."""
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

    z = [0.0] * p
    passive: list[int] = []
    in_passive = [False] * p
    blocked: set[int] = set()
    iters = 0

    fitted, grad, kkt_violation = _eval_kkt_and_grad_stdlib(norm_cols, scales, y, z)
    while iters < max_iter and kkt_violation > tol:
        best_j = -1
        best_neg_grad = tol
        for j in range(p):
            if scales[j] == 0.0 or in_passive[j] or j in blocked:
                continue
            if -grad[j] > best_neg_grad:
                best_neg_grad = -grad[j]
                best_j = j
        if best_j < 0:
            break

        iters += 1
        passive.append(best_j)
        in_passive[best_j] = True

        inner_guard = 0
        max_inner = max(16, 4 * p)
        while passive and inner_guard < max_inner:
            inner_guard += 1
            s_sub, full_rank = _householder_lstsq_stdlib([norm_cols[j] for j in passive], y)
            if not full_rank:
                passive.pop()
                in_passive[best_j] = False
                blocked.add(best_j)
                break

            if min(s_sub) > 1e-15:
                for idx_p, j in enumerate(passive):
                    z[j] = s_sub[idx_p]
                blocked.clear()
                break

            alpha_step = 1.0
            for idx_p, j in enumerate(passive):
                sj = s_sub[idx_p]
                if sj <= 1e-15:
                    zj = z[j]
                    denom = zj - sj
                    ratio = (zj / denom) if denom > 0.0 else 0.0
                    if ratio < alpha_step:
                        alpha_step = ratio

            new_passive: list[int] = []
            for idx_p, j in enumerate(passive):
                updated = max(0.0, z[j] + alpha_step * (s_sub[idx_p] - z[j]))
                if updated <= 1e-15:
                    z[j] = 0.0
                    in_passive[j] = False
                else:
                    z[j] = updated
                    new_passive.append(j)
            passive = new_passive

        fitted, grad, kkt_violation = _eval_kkt_and_grad_stdlib(norm_cols, scales, y, z)

    # Cyclic PCD polish if rank-degeneracy left any KKT violation and budget remains
    if kkt_violation > tol and iters < max_iter:
        gram: list[list[float]] = [[0.0] * p for _ in range(p)]
        for j in range(p):
            uj = norm_cols[j]
            for k in range(j, p):
                uk = norm_cols[k]
                g_jk = sum(uj[i] * uk[i] for i in range(m))
                gram[j][k] = g_jk
                gram[k][j] = g_jk
        while iters < max_iter and kkt_violation > tol:
            iters += 1
            max_step = 0.0
            for j in range(p):
                if scales[j] == 0.0:
                    continue
                old_zj = z[j]
                new_zj = max(0.0, old_zj - grad[j])
                delta = new_zj - old_zj
                if abs(delta) > 0.0:
                    z[j] = new_zj
                    row_j = gram[j]
                    for k in range(p):
                        grad[k] += row_j[k] * delta
                    if abs(delta) > max_step:
                        max_step = abs(delta)
            fitted, grad, kkt_violation = _eval_kkt_and_grad_stdlib(norm_cols, scales, y, z)
            if kkt_violation <= tol and max_step <= tol:
                break

    weights = [z[j] / scales[j] if scales[j] > 0.0 else 0.0 for j in range(p)]
    return weights, fitted, kkt_violation, iters


def _eval_kkt_and_grad_numpy(
    u_mat: Any,
    active: Any,
    y_vec: Any,
    z: Any,
) -> tuple[Any, Any, float]:
    """NumPy evaluation of fitted vector, direct residual-space gradient, and max KKT violation."""
    assert _np is not None
    fitted_arr = u_mat @ z
    res_arr = y_vec - fitted_arr
    grad_arr = -(u_mat.T @ res_arr)
    grad_arr[~active] = 0.0

    kkt_violation = 0.0
    for j in range(int(z.shape[0])):
        if not bool(active[j]):
            continue
        gj = float(grad_arr[j])
        viol = abs(gj) if float(z[j]) > 1e-13 else max(0.0, -gj)
        if viol > kkt_violation:
            kkt_violation = viol
    return fitted_arr, grad_arr, kkt_violation


def _solve_nnls_numpy(
    columns: list[list[float]],
    y: list[float],
    tol: float,
    max_iter: int,
) -> tuple[list[float], list[float], float, int]:
    """NumPy-accelerated Lawson-Hanson active-set NNLS solver (`lstsq` + PCD fallback)."""
    assert _np is not None
    mat = _np.asarray(columns, dtype=_np.float64).T  # shape (M, P)
    y_vec = _np.asarray(y, dtype=_np.float64)
    scales = _np.linalg.norm(mat, axis=0)
    active = scales > 1e-15
    u_mat = _np.zeros_like(mat)
    u_mat[:, active] = mat[:, active] / scales[active]

    p = int(mat.shape[1])
    z = _np.zeros(p, dtype=_np.float64)
    passive: list[int] = []
    in_passive = [False] * p
    blocked: set[int] = set()
    iters = 0

    fitted_arr, grad_arr, kkt_violation = _eval_kkt_and_grad_numpy(u_mat, active, y_vec, z)
    while iters < max_iter and kkt_violation > tol:
        best_j = -1
        best_neg_grad = tol
        for j in range(p):
            if not bool(active[j]) or in_passive[j] or j in blocked:
                continue
            neg_g = -float(grad_arr[j])
            if neg_g > best_neg_grad:
                best_neg_grad = neg_g
                best_j = j
        if best_j < 0:
            break

        iters += 1
        passive.append(best_j)
        in_passive[best_j] = True

        inner_guard = 0
        max_inner = max(16, 4 * p)
        while passive and inner_guard < max_inner:
            inner_guard += 1
            u_sub = u_mat[:, passive]
            s_arr, _, rank, svals = _np.linalg.lstsq(u_sub, y_vec, rcond=1e-12)
            full_rank = int(rank) == len(passive) and (
                len(svals) == 0 or float(svals[-1]) > 1e-12
            )
            if not full_rank:
                passive.pop()
                in_passive[best_j] = False
                blocked.add(best_j)
                break

            s_sub = [float(v) for v in s_arr.tolist()]
            if min(s_sub) > 1e-15:
                for idx_p, j in enumerate(passive):
                    z[j] = s_sub[idx_p]
                blocked.clear()
                break

            alpha_step = 1.0
            for idx_p, j in enumerate(passive):
                sj = s_sub[idx_p]
                if sj <= 1e-15:
                    zj = float(z[j])
                    denom = zj - sj
                    ratio = (zj / denom) if denom > 0.0 else 0.0
                    if ratio < alpha_step:
                        alpha_step = ratio

            new_passive: list[int] = []
            for idx_p, j in enumerate(passive):
                updated = max(0.0, float(z[j]) + alpha_step * (s_sub[idx_p] - float(z[j])))
                if updated <= 1e-15:
                    z[j] = 0.0
                    in_passive[j] = False
                else:
                    z[j] = updated
                    new_passive.append(j)
            passive = new_passive

        fitted_arr, grad_arr, kkt_violation = _eval_kkt_and_grad_numpy(u_mat, active, y_vec, z)

    if kkt_violation > tol and iters < max_iter:
        gram = u_mat.T @ u_mat
        while iters < max_iter and kkt_violation > tol:
            iters += 1
            max_step = 0.0
            for j in range(p):
                if not bool(active[j]):
                    continue
                old_zj = float(z[j])
                new_zj = max(0.0, old_zj - float(grad_arr[j]))
                delta = new_zj - old_zj
                if delta != 0.0:
                    z[j] = new_zj
                    grad_arr += gram[j] * delta
                    if abs(delta) > max_step:
                        max_step = abs(delta)
            fitted_arr, grad_arr, kkt_violation = _eval_kkt_and_grad_numpy(u_mat, active, y_vec, z)
            if kkt_violation <= tol and max_step <= tol:
                break

    weights_arr = _np.zeros(p, dtype=_np.float64)
    weights_arr[active] = z[active] / scales[active]
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
    two_theta_grid: Sequence[float] | None = None,
    max_peak_shift_deg: float = 0.0,
    allow_peak_shift_bins: int = 0,
    max_strain_scale: float = 0.0,
    background_poly_degree: int | None = None,
    max_background_ratio: float = 5.0,
    l1_alpha: float = 0.0,
    l2_alpha: float = 0.0,
    pseudo_voigt_fwhm_add: float = 0.0,
    pseudo_voigt_eta: float = 0.5,
    shift_search_mode: str = "dictionary",
    tol: float = 1e-10,
    max_iter: int = 2000,
    use_numpy: bool | None = None,
) -> DeconvolutionResult:
    """Deconvolve `observed_y` into non-negative precursor phases + target phase (+ optional baseline).

    Parameters
    ----------
    observed_y:
        Measured 1D diffraction / optical spectrum of length `M >= 2` (all finite).
    precursor_spectra:
        Sequence of `K >= 0` starting-precursor reference spectra, each of length `M`.
    target_spectrum:
        Putative novel target phase reference spectrum of length `M`.
    min_hit_amplitude:
        Minimum fitted target phase scale `beta` required to declare `target_signal_present`
        and reference detection-limit scale used for assay SNR and precursor shadowing.
    noise_sigma:
        Nominal per-channel detector noise standard deviation `sigma > 0` (bounded below by
        the empirical Rice first-difference high-frequency residual RMS).
    naive_cosine_threshold:
        Cosine similarity threshold used by a naive peak matcher (demonstrating A-Lab false hits).
    weights:
        Optional non-negative Rietveld channel weights `w_i` minimized in the NNLS objective
        `0.5 * sum_i w_i (y_i - fitted_i)^2` and used in `R_wp`; must have positive support on
        `target_spectrum`.
    two_theta_grid:
        Optional monotonic 2-theta coordinate axis of length `M` for Bragg shift/strain alignment.
    max_peak_shift_deg:
        Maximum symmetric Bragg peak shift tolerance `+-delta` in 2-theta degrees (`0.0` disables).
    allow_peak_shift_bins:
        Optional shift tolerance in channel bins (`0` disables); works with or without `two_theta_grid`.
    max_strain_scale:
        Optional isotropic lattice-strain scale modeling `Delta(2theta) = delta + eps * tan(theta)`.
    background_poly_degree:
        Optional degree `d in [0, 8]` of non-negative Bernstein polynomial amorphous background basis.
    max_background_ratio:
        Maximum allowed ratio `max(y_bg) / max(target_spectrum)` before flagging fluorescence saturation.
    use_numpy:
        `True` forces NumPy, `False` forces pure-Python stdlib, `None` uses NumPy if available.
    """
    y = [max(0.0, _require_finite_float(v, f"observed_y[{i}]")) for i, v in enumerate(observed_y)]
    m = len(y)
    if m < 2:
        raise ValueError("Spectrum must contain at least 2 channels")

    if len(target_spectrum) != m:
        raise ValueError("target_spectrum length must match observed_y length")
    s_target = [
        max(0.0, _require_finite_float(v, f"target_spectrum[{i}]"))
        for i, v in enumerate(target_spectrum)
    ]

    prec_cols: list[list[float]] = []
    for idx, p_spec in enumerate(precursor_spectra):
        if len(p_spec) != m:
            raise ValueError(f"precursor_spectra[{idx}] length {len(p_spec)} != {m}")
        col = [
            max(0.0, _require_finite_float(v, f"precursor_spectra[{idx}][{i}]"))
            for i, v in enumerate(p_spec)
        ]
        prec_cols.append(col)

    min_hit_amp = _require_finite_float(min_hit_amplitude, "min_hit_amplitude")
    sigma_val = _require_finite_float(noise_sigma, "noise_sigma")
    naive_thresh = _require_finite_float(naive_cosine_threshold, "naive_cosine_threshold")
    tol_val = _require_finite_float(tol, "tol")
    shift_deg_val = _require_finite_float(max_peak_shift_deg, "max_peak_shift_deg")
    strain_val = _require_finite_float(max_strain_scale, "max_strain_scale")
    max_bg_ratio = _require_finite_float(max_background_ratio, "max_background_ratio")
    l1_val = _require_finite_float(l1_alpha, "l1_alpha")
    l2_val = _require_finite_float(l2_alpha, "l2_alpha")
    pv_fwhm_val = _require_finite_float(pseudo_voigt_fwhm_add, "pseudo_voigt_fwhm_add")
    pv_eta_val = _require_finite_float(pseudo_voigt_eta, "pseudo_voigt_eta")

    if min_hit_amp <= 0.0:
        raise ValueError("min_hit_amplitude must be positive")
    if sigma_val <= 0.0:
        raise ValueError("noise_sigma must be positive")
    if tol_val <= 0.0:
        raise ValueError("tol must be positive")
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    if isinstance(allow_peak_shift_bins, bool) or not isinstance(allow_peak_shift_bins, int) or allow_peak_shift_bins < 0:
        raise ValueError("allow_peak_shift_bins must be a non-negative integer")
    if shift_deg_val < 0.0:
        raise ValueError("max_peak_shift_deg must be non-negative")
    if strain_val < 0.0:
        raise ValueError("max_strain_scale must be non-negative")
    if l1_val < 0.0:
        raise ValueError("l1_alpha must be non-negative")
    if l2_val < 0.0:
        raise ValueError("l2_alpha must be non-negative")
    if pv_fwhm_val < 0.0:
        raise ValueError("pseudo_voigt_fwhm_add must be non-negative")
    if not (0.0 <= pv_eta_val <= 1.0):
        raise ValueError("pseudo_voigt_eta must lie in [0, 1]")
    if shift_search_mode not in ("dictionary", "golden_section"):
        raise ValueError("shift_search_mode must be 'dictionary' or 'golden_section'")
    if background_poly_degree is not None:
        if (
            isinstance(background_poly_degree, bool)
            or not isinstance(background_poly_degree, int)
            or not (0 <= background_poly_degree <= 8)
        ):
            raise ValueError("background_poly_degree must be an integer in [0, 8]")

    target_norm = math.sqrt(sum(v * v for v in s_target))
    if target_norm <= 1e-15:
        raise ValueError("target_spectrum must have non-zero L2 norm")

    if weights is None:
        w_profile = [1.0] * m
    else:
        if len(weights) != m:
            raise ValueError("weights length must match observed_y length")
        w_profile = []
        for i, w_raw in enumerate(weights):
            w_val = _require_finite_float(w_raw, f"weights[{i}]")
            if w_val < 0.0:
                raise ValueError(f"weights[{i}] must be non-negative, got {w_val}")
            w_profile.append(w_val)
        if sum(w_profile) <= 0.0:
            raise ValueError("weights must contain positive entries (all-zero weights forbidden)")

    min_ref_sq = sum(w_profile[i] * ((min_hit_amp * s_target[i]) ** 2) for i in range(m))
    if min_ref_sq <= 1e-18:
        raise ValueError("weights must have positive support on target_spectrum")

    if two_theta_grid is not None:
        if len(two_theta_grid) != m:
            raise ValueError("two_theta_grid length must match observed_y length")
        grid_x = [
            _require_finite_float(v, f"two_theta_grid[{i}]")
            for i, v in enumerate(two_theta_grid)
        ]
        for i in range(m - 1):
            if grid_x[i + 1] <= grid_x[i]:
                raise ValueError("two_theta_grid must be strictly increasing")
        is_angles_deg = True
    else:
        grid_x = [float(i) for i in range(m)]
        is_angles_deg = False

    dx_mean = (grid_x[-1] - grid_x[0]) / float(m - 1)
    effective_shift_bound = max(shift_deg_val, float(allow_peak_shift_bins) * dx_mean)

    if shift_search_mode == "golden_section":
        from lab_attrib.deconvolution import deconvolve_frontier_spectrum

        fwhm_cands = (0.0, pv_fwhm_val) if pv_fwhm_val > 0.0 else (0.0, 0.16, 0.34, 0.55)
        return deconvolve_frontier_spectrum(
            y,
            prec_cols,
            s_target,
            two_theta_grid=grid_x,
            max_peak_shift_deg=effective_shift_bound,
            broadening_fwhm_candidates=fwhm_cands,
            pseudo_voigt_eta=pv_eta_val,
            background_poly_degree=background_poly_degree,
            l1_alpha=(l1_val if l1_val > 0.0 else None),
            min_hit_amplitude=min_hit_amp,
            noise_sigma=sigma_val,
            naive_cosine_threshold=naive_thresh,
            weights=w_profile,
            use_numpy=use_numpy,
        )

    if pv_fwhm_val > 0.0:
        from lab_attrib.deconvolution import convolve_pseudo_voigt_broadening

        broadened_precs: list[list[float]] = []
        for p_col in prec_cols:
            p_max = max(p_col)
            conv_p = convolve_pseudo_voigt_broadening(
                p_col, grid_x, fwhm_add=pv_fwhm_val, eta=pv_eta_val
            )
            c_max = max(conv_p)
            scale_p = (p_max / c_max) if (p_max > 1e-15 and c_max > 1e-15) else 1.0
            broadened_precs.append([v * scale_p for v in conv_p])
        prec_cols = broadened_precs

        t_max = max(s_target)
        conv_t = convolve_pseudo_voigt_broadening(
            s_target, grid_x, fwhm_add=pv_fwhm_val, eta=pv_eta_val
        )
        ct_max = max(conv_t)
        scale_t = (t_max / ct_max) if (t_max > 1e-15 and ct_max > 1e-15) else 1.0
        s_target = [v * scale_t for v in conv_t]
        target_norm = math.sqrt(sum(v * v for v in s_target))

    # Collinearity condition diagnostic across precursor and target reference bases
    s_unit_unshifted = [v / target_norm for v in s_target]
    max_basis_cos = 0.0
    for p_col in prec_cols:
        p_norm = math.sqrt(sum(v * v for v in p_col))
        if p_norm > 1e-15:
            cos_pt = sum(p_col[i] * s_unit_unshifted[i] for i in range(m)) / p_norm
            if cos_pt > max_basis_cos:
                max_basis_cos = min(1.0, max(0.0, cos_pt))

    # Construct optional shift/strain dictionary per crystalline phase
    if effective_shift_bound > 0.0:
        fine_step = min(0.025 if is_angles_deg else 0.25, 0.5 * dx_mean)
        n_half = min(20, max(2, int(math.ceil(effective_shift_bound / max(1e-9, fine_step)))))
        shift_grid = [effective_shift_bound * (k / float(n_half)) for k in range(-n_half, n_half + 1)]
    else:
        shift_grid = [0.0]

    strain_grid = [-strain_val, 0.0, strain_val] if strain_val > 0.0 else [0.0]
    warp_params = [(sh, st) for sh in shift_grid for st in strain_grid]

    design_cols: list[list[float]] = []
    prec_slices: list[tuple[int, int]] = []
    for p_col in prec_cols:
        start_idx = len(design_cols)
        for sh, st in warp_params:
            design_cols.append(
                _catmull_rom_warp(p_col, grid_x, sh, st, is_angles_deg=is_angles_deg)
            )
        prec_slices.append((start_idx, len(design_cols)))

    target_start = len(design_cols)
    target_warped_cols: list[list[float]] = []
    for sh, st in warp_params:
        warped_t = _catmull_rom_warp(s_target, grid_x, sh, st, is_angles_deg=is_angles_deg)
        design_cols.append(warped_t)
        target_warped_cols.append(warped_t)
    target_end = len(design_cols)

    bg_start = len(design_cols)
    bg_cols: list[list[float]] = (
        _build_bernstein_background_columns(m, background_poly_degree)
        if background_poly_degree is not None
        else []
    )
    design_cols.extend(bg_cols)
    bg_end = len(design_cols)

    # Pre-condition design matrix and observation by W^{1/2} for exact weighted Rietveld NNLS
    sqrt_w = [math.sqrt(w) for w in w_profile]
    y_weighted = [sqrt_w[i] * y[i] for i in range(m)]
    weighted_cols = [[sqrt_w[i] * col[i] for i in range(m)] for col in design_cols]

    run_np = (_np is not None) if use_numpy is None else bool(use_numpy)
    if run_np and _np is None:  # pragma: no cover
        raise RuntimeError("use_numpy=True requested, but numpy is not installed")

    if l1_val > 0.0 or l2_val > 0.0:
        from lab_attrib.deconvolution import elastic_nnls

        n_phase_cols = bg_start
        n_bg_cols = bg_end - bg_start
        l1_per_col = [l1_val] * n_phase_cols + [0.0] * n_bg_cols
        l2_per_col = [l2_val] * n_phase_cols + [0.0] * n_bg_cols
        el_res = elastic_nnls(
            design_cols,
            y,
            l1_alpha=l1_per_col,
            l2_alpha=l2_per_col,
            sample_weights=w_profile,
            tol=tol_val,
            max_iter=max_iter,
            use_numpy=run_np,
        )
        coeff = list(el_res.weights)
        kkt_viol = el_res.kkt_max_violation
        iters = el_res.iterations
    elif run_np:
        coeff, _, kkt_viol, iters = _solve_nnls_numpy(
            weighted_cols, y_weighted, tol=tol_val, max_iter=max_iter
        )
    else:
        coeff, _, kkt_viol, iters = _solve_nnls_stdlib(
            weighted_cols, y_weighted, tol=tol_val, max_iter=max_iter
        )

    notes: list[str] = []
    if max_basis_cos >= 0.995:
        notes.append("NEAR_COLLINEAR_BASIS")
    converged = kkt_viol <= max(tol_val * 100.0, 1e-6)
    if not converged:
        notes.append("NNLS_NON_CONVERGENCE")

    # Aggregate precursor, target, and amorphous background components in original intensity units
    prec_weights_list: list[float] = []
    y_prec = [0.0] * m
    for start_idx, end_idx in prec_slices:
        phase_sum = 0.0
        for col_idx in range(start_idx, end_idx):
            c_val = coeff[col_idx]
            if c_val > 0.0:
                phase_sum += c_val
                col_vec = design_cols[col_idx]
                for i in range(m):
                    y_prec[i] += c_val * col_vec[i]
        prec_weights_list.append(phase_sum)
    prec_weights = tuple(prec_weights_list)

    beta_target = 0.0
    weighted_shift_sum = 0.0
    y_target = [0.0] * m
    for local_idx, col_idx in enumerate(range(target_start, target_end)):
        c_val = coeff[col_idx]
        if c_val > 0.0:
            beta_target += c_val
            weighted_shift_sum += c_val * warp_params[local_idx][0]
            col_vec = design_cols[col_idx]
            for i in range(m):
                y_target[i] += c_val * col_vec[i]
    fitted_shift_deg = weighted_shift_sum / beta_target if beta_target > 1e-12 else 0.0

    bg_weights = tuple(coeff[bg_start:bg_end])
    y_bg = [0.0] * m
    for c_val, bg_col in zip(bg_weights, bg_cols):
        if c_val > 0.0:
            for i in range(m):
                y_bg[i] += c_val * bg_col[i]

    fitted = [y_prec[i] + y_target[i] + y_bg[i] for i in range(m)]

    # 1. Weighted profile residual R_wp in permille
    weighted_err_sq = sum(w_profile[i] * (y[i] - fitted[i]) ** 2 for i in range(m))
    weighted_obs_sq = sum(w_profile[i] * (y[i] ** 2) for i in range(m))
    denom_sq = max(weighted_obs_sq, min_ref_sq, 1e-18)
    r_wp = math.sqrt(weighted_err_sq / denom_sq)
    spectral_residual_permille = min(1000, max(0, int(round(1000.0 * r_wp))))

    # 2. Precursor peak-overlap shadowing index on target channels in permille:
    #    Project onto aligned target profile for target_on_target and take maximum precursor
    #    shadowing across allowed target shifts so shifted unreacted precursors cannot escape.
    y_target_norm = math.sqrt(sum(v * v for v in y_target))
    if y_target_norm > 1e-12:
        s_eff_unit = [v / y_target_norm for v in y_target]
    else:
        s_eff_unit = s_unit_unshifted
    target_on_target = sum(y_target[i] * s_eff_unit[i] for i in range(m))

    prec_on_target = sum(y_prec[i] * s_eff_unit[i] for i in range(m))
    for warped_t in target_warped_cols:
        wt_norm = math.sqrt(sum(v * v for v in warped_t))
        if wt_norm > 1e-15:
            proj = sum(y_prec[i] * warped_t[i] for i in range(m)) / wt_norm
            if proj > prec_on_target:
                prec_on_target = proj

    floor_intensity = min_hit_amp * target_norm
    overlap_ratio = prec_on_target / (prec_on_target + max(target_on_target, floor_intensity))
    precursor_overlap_permille = min(1000, max(0, int(round(1000.0 * overlap_ratio))))

    # 3. Assay resolvability / detection-power SNR in tenths:
    #    Bound noise floor below by empirical Rice first-difference high-frequency residual RMS.
    res_unweighted = [y[i] - fitted[i] for i in range(m)]
    hf_diff_sq = sum((res_unweighted[i + 1] - res_unweighted[i]) ** 2 for i in range(m - 1))
    empirical_hf_rms = math.sqrt(hf_diff_sq / float(max(1, 2 * (m - 1))))
    effective_noise = max(sigma_val, empirical_hf_rms)

    effective_signal_norm = max(target_on_target, floor_intensity)
    assay_snr = effective_signal_norm / effective_noise
    snr_tenths = max(0, int(round(10.0 * assay_snr)))

    # Guard extreme amorphous fluorescence saturation or NNLS non-convergence
    max_target_peak = max(s_target)
    if bg_cols and max(y_bg) > max_bg_ratio * max(max_target_peak, 1e-12):
        notes.append("EXCESSIVE_BACKGROUND_FLUORESCENCE")
        snr_tenths = min(snr_tenths, 10)
    if not converged:
        snr_tenths = 0

    # 4. Naive peak-matcher cosine similarity between raw y and unshifted s_target
    obs_norm = math.sqrt(sum(v * v for v in y))
    if obs_norm > 1e-12:
        naive_cos = sum(y[i] * s_unit_unshifted[i] for i in range(m)) / obs_norm
    else:
        naive_cos = 0.0
    naive_cos = max(0.0, min(1.0, naive_cos))

    target_present = converged and (beta_target >= min_hit_amp)
    naive_hit = naive_cos >= naive_thresh and obs_norm >= 0.5 * floor_intensity

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
        converged=converged,
        fitted_shift_deg=fitted_shift_deg,
        max_basis_cosine=max_basis_cos,
        background_weights=bg_weights,
        empirical_noise_rms=empirical_hf_rms,
        diagnostic_notes=tuple(notes),
    )
