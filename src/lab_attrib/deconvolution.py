"""Frontier Spectral Deconvolution & Tandem MS/MS Entropy Similarity Engine.

Upgrades `lab-attrib`'s post-execution analytical characterization beyond fixed-grid,
unregularized Gaussian NNLS and raw MS/MS dot-product cosine matching:

1. **Pseudo-Voigt Peak Synthesis, Caglioti Resolution & Instrumental Broadening**:
   Models mixed Gaussian microstrain/slit optics and Lorentzian crystallite-size tails
   (`V_p = eta * L + (1 - eta) * G`) plus angle-dependent Caglioti `FWHM(2theta)`.
2. **Sparsity-Regularized Weighted Elastic-Net NNLS (`elastic_nnls`)**:
   Solves `min_{c >= 0} 0.5 * ||W^{1/2}(A c - y)||_2^2 + sum_j l1_j c_j + 0.5 * sum_j l2_j c_j^2`
   to KKT tolerance `<= 1e-10` via active-set QR + unit-diagonal coordinate descent in both
   NumPy and zero-dependency pure-Python stdlib backends, eliminating spurious minor-phase
   hallucinations (`FDR`) in multi-mineral libraries.
3. **Shift-Tolerant Pseudo-Voigt `L1-NNLS` (`deconvolve_multiphase_library`, `deconvolve_frontier_spectrum`)**:
   Combines coarse-grid + continuous Golden-Section zero-point `2-theta` alignment (`+-0.35 deg`),
   per-phase Vegard solid-solution lattice-shift refinement, Pseudo-Voigt broadening selection,
   and Bernstein polynomial amorphous background subtraction.
4. **MS/MS Spectral Entropy Similarity & Near-Isobaric Chimera Quarantine**:
   Implements unweighted and dynamically weighted MS/MS spectral entropy similarity
   (Li et al., *Nature Methods* 18, 1524-1531, 2021) alongside convex two-component mixture
   entropy maximization to quarantine co-eluting near-isobaric chimeras.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Literal, Sequence

from lab_attrib.xrd_deconvolve import (
    DeconvolutionResult,
    _build_bernstein_background_columns,
    _catmull_rom_warp,
    _require_finite_float,
)

if TYPE_CHECKING:
    from lab_attrib.open_datasets import MassSpecRecord

try:
    import numpy as _np  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    _np = None

_GOLDEN_PHI: float = (math.sqrt(5.0) - 1.0) * 0.5
_LN4: float = math.log(4.0)
_FOUR_LN2: float = 4.0 * math.log(2.0)


def _expand_nonneg_vector(
    param: float | Sequence[float],
    length: int,
    name: str,
) -> list[float]:
    """Validate scalar or per-column non-negative regularization weights."""
    if isinstance(param, (int, float)) and not isinstance(param, bool):
        val = _require_finite_float(param, name)
        if val < 0.0:
            raise ValueError(f"{name} must be non-negative, got {val}")
        return [val] * length
    if isinstance(param, bool) or not isinstance(param, Sequence):
        raise ValueError(f"{name} must be a non-negative float or sequence of length {length}")
    if len(param) != length:
        raise ValueError(f"{name} length {len(param)} != {length}")
    out: list[float] = []
    for idx, raw in enumerate(param):
        val = _require_finite_float(raw, f"{name}[{idx}]")
        if val < 0.0:
            raise ValueError(f"{name}[{idx}] must be non-negative, got {val}")
        out.append(val)
    return out


# ---------------------------------------------------------------------------
# 1. Pseudo-Voigt Profiles, Caglioti FWHM & Broadening Convolution
# ---------------------------------------------------------------------------


def caglioti_fwhm(
    two_theta_deg: float,
    *,
    u: float = 0.02,
    v: float = -0.01,
    w: float = 0.09,
    min_fwhm: float = 0.03,
) -> float:
    """Compute angle-dependent Caglioti powder diffraction FWHM (`sqrt(U tan^2 theta + V tan theta + W)`)."""
    ang = _require_finite_float(two_theta_deg, "two_theta_deg")
    u_val = _require_finite_float(u, "u")
    v_val = _require_finite_float(v, "v")
    w_val = _require_finite_float(w, "w")
    floor_val = _require_finite_float(min_fwhm, "min_fwhm")
    if floor_val <= 0.0:
        raise ValueError("min_fwhm must be positive")
    theta_rad = math.radians(max(0.0, min(89.5, 0.5 * abs(ang))))
    tan_t = math.tan(theta_rad)
    h_sq = u_val * (tan_t * tan_t) + v_val * tan_t + w_val
    return math.sqrt(max(floor_val * floor_val, h_sq))


def pseudo_voigt_profile(
    grid_x: Sequence[float],
    center: float,
    height: float,
    fwhm: float,
    *,
    eta: float = 0.5,
) -> list[float]:
    """Evaluate a peak-height-normalized Pseudo-Voigt peak `height * (eta * L + (1 - eta) * G)`."""
    c_val = _require_finite_float(center, "center")
    h_val = max(0.0, _require_finite_float(height, "height"))
    w_val = _require_finite_float(fwhm, "fwhm")
    eta_val = _require_finite_float(eta, "eta")
    if w_val <= 0.0:
        raise ValueError(f"fwhm must be positive, got {fwhm}")
    if not (0.0 <= eta_val <= 1.0):
        raise ValueError(f"eta must lie in [0, 1], got {eta}")

    inv_w_sq = 1.0 / (w_val * w_val)
    out: list[float] = []
    for idx, x_raw in enumerate(grid_x):
        x = _require_finite_float(x_raw, f"grid_x[{idx}]")
        dx_sq = (x - c_val) * (x - c_val)
        scaled = dx_sq * inv_w_sq
        gauss = math.exp(-_FOUR_LN2 * scaled)
        lorentz = 1.0 / (1.0 + 4.0 * scaled)
        out.append(h_val * (eta_val * lorentz + (1.0 - eta_val) * gauss))
    return out


def build_pseudo_voigt_spectrum(
    grid_x: Sequence[float],
    peaks: Sequence[tuple[float, ...]],
    *,
    default_fwhm: float = 0.35,
    default_eta: float = 0.45,
    caglioti_uvw: tuple[float, float, float] | None = None,
) -> list[float]:
    """Synthesize a multi-peak Pseudo-Voigt spectrum from `(center, height[, fwhm[, eta]])` tuples."""
    if not grid_x:
        raise ValueError("grid_x must be non-empty")
    xs = [_require_finite_float(v, f"grid_x[{i}]") for i, v in enumerate(grid_x)]
    def_w = _require_finite_float(default_fwhm, "default_fwhm")
    def_eta = _require_finite_float(default_eta, "default_eta")
    if def_w <= 0.0:
        raise ValueError("default_fwhm must be positive")
    if not (0.0 <= def_eta <= 1.0):
        raise ValueError("default_eta must lie in [0, 1]")

    parsed_peaks: list[tuple[float, float, float, float]] = []
    for idx, pk in enumerate(peaks):
        if len(pk) < 2 or len(pk) > 4:
            raise ValueError(
                f"peaks[{idx}] must have 2 to 4 elements (center, height[, fwhm[, eta]]), got {len(pk)}"
            )
        c_val = _require_finite_float(pk[0], f"peaks[{idx}].center")
        h_val = max(0.0, _require_finite_float(pk[1], f"peaks[{idx}].height"))
        if len(pk) >= 3:
            w_val = _require_finite_float(pk[2], f"peaks[{idx}].fwhm")
        elif caglioti_uvw is not None:
            w_val = caglioti_fwhm(
                c_val, u=caglioti_uvw[0], v=caglioti_uvw[1], w=caglioti_uvw[2]
            )
        else:
            w_val = def_w
        eta_val = _require_finite_float(pk[3], f"peaks[{idx}].eta") if len(pk) == 4 else def_eta
        if w_val <= 0.0:
            raise ValueError(f"peaks[{idx}].fwhm must be positive, got {w_val}")
        if not (0.0 <= eta_val <= 1.0):
            raise ValueError(f"peaks[{idx}].eta must lie in [0, 1], got {eta_val}")
        parsed_peaks.append((c_val, h_val, w_val, eta_val))

    out = [0.0] * len(xs)
    for c_val, h_val, w_val, eta_val in parsed_peaks:
        if h_val <= 0.0:
            continue
        inv_w_sq = 1.0 / (w_val * w_val)
        for i, x in enumerate(xs):
            scaled = (x - c_val) * (x - c_val) * inv_w_sq
            if scaled > 625.0 and eta_val == 0.0:
                continue
            gauss = math.exp(-_FOUR_LN2 * scaled) if scaled < 625.0 else 0.0
            lorentz = 1.0 / (1.0 + 4.0 * scaled)
            out[i] += h_val * (eta_val * lorentz + (1.0 - eta_val) * gauss)
    return out


def convolve_pseudo_voigt_broadening(
    spectrum: Sequence[float],
    grid_x: Sequence[float],
    *,
    fwhm_add: float = 0.0,
    eta: float = 0.5,
) -> list[float]:
    """Convolve a 1D reference spectrum with a unit-sum discrete Pseudo-Voigt broadening kernel."""
    m = len(spectrum)
    if m < 2 or len(grid_x) != m:
        raise ValueError("spectrum and grid_x must have matching length >= 2")
    ys = [max(0.0, _require_finite_float(v, f"spectrum[{i}]")) for i, v in enumerate(spectrum)]
    w_add = _require_finite_float(fwhm_add, "fwhm_add")
    eta_val = _require_finite_float(eta, "eta")
    if w_add < 0.0:
        raise ValueError("fwhm_add must be non-negative")
    if not (0.0 <= eta_val <= 1.0):
        raise ValueError("eta must lie in [0, 1]")
    if w_add <= 1e-9:
        return ys

    dx = (_require_finite_float(grid_x[-1], "grid_x[-1]") - _require_finite_float(grid_x[0], "grid_x[0]")) / float(m - 1)
    if dx <= 1e-12:
        return ys

    half_span = min(m - 1, max(2, int(math.ceil(8.0 * w_add / dx))))
    inv_w_sq = 1.0 / (w_add * w_add)
    kernel: list[float] = []
    for k in range(-half_span, half_span + 1):
        u = float(k) * dx
        scaled = u * u * inv_w_sq
        gauss = math.exp(-_FOUR_LN2 * scaled) if scaled < 625.0 else 0.0
        lorentz = 1.0 / (1.0 + 4.0 * scaled)
        kernel.append(eta_val * lorentz + (1.0 - eta_val) * gauss)
    k_sum = sum(kernel)
    if k_sum <= 1e-15:
        return ys
    kernel = [kv / k_sum for kv in kernel]

    if _np is not None:
        conv = _np.convolve(
            _np.asarray(ys, dtype=_np.float64),
            _np.asarray(kernel, dtype=_np.float64),
            mode="same",
        )
        return [max(0.0, float(v)) for v in conv.tolist()]

    out = [0.0] * m
    for i in range(m):
        acc = 0.0
        j_lo = max(0, i - half_span)
        j_hi = min(m - 1, i + half_span)
        for j in range(j_lo, j_hi + 1):
            acc += ys[j] * kernel[(i - j) + half_span]
        out[i] = max(0.0, acc)
    return out


# ---------------------------------------------------------------------------
# 2. Sparsity-Regularized Weighted Elastic-Net NNLS (`elastic_nnls`)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ElasticNNLSResult:
    """Structured output of weighted sparsity-regularized non-negative least squares."""

    weights: tuple[float, ...]
    fitted: tuple[float, ...]
    objective: float
    kkt_max_violation: float
    iterations: int
    converged: bool
    used_numpy: bool
    support_indices: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights": [round(w, 8) for w in self.weights],
            "objective": round(self.objective, 10),
            "kkt_max_violation": round(self.kkt_max_violation, 12),
            "iterations": self.iterations,
            "converged": self.converged,
            "used_numpy": self.used_numpy,
            "support_indices": list(self.support_indices),
        }


def _householder_qr_upper_stdlib(
    cols_sub: list[list[float]],
    l2_diag: list[float],
) -> tuple[list[list[float]], bool]:
    """Compute upper-triangular `R` (`k x k`) of `[U_sub; diag(sqrt(l2_diag))]` via Householder QR."""
    k = len(cols_sub)
    if k == 0:
        return [], True
    m_base = len(cols_sub[0])
    has_l2 = any(v > 0.0 for v in l2_diag)
    m_aug = m_base + (k if has_l2 else 0)
    if k > m_aug:
        return [], False

    a: list[list[float]] = []
    for c_idx, col in enumerate(cols_sub):
        aug = col[:] + ([0.0] * k if has_l2 else [])
        if has_l2 and l2_diag[c_idx] > 0.0:
            aug[m_base + c_idx] = math.sqrt(l2_diag[c_idx])
        a.append(aug)

    for col in range(k):
        col_vec = a[col]
        norm_sq = sum(col_vec[i] * col_vec[i] for i in range(col, m_aug))
        norm_x = math.sqrt(norm_sq)
        if norm_x <= 1e-13:
            return [], False
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
                col_vec[i] * target_col[i] for i in range(col + 1, m_aug)
            )
            scale = inv_half * dot
            target_col[col] -= scale * v0
            for i in range(col + 1, m_aug):
                target_col[i] -= scale * col_vec[i]

    r_mat = [[a[c2][r] if c2 >= r else 0.0 for c2 in range(k)] for r in range(k)]
    for d in range(k):
        if abs(r_mat[d][d]) <= 1e-12:
            return [], False
    return r_mat, True


def _solve_passive_elastic_stdlib(
    cols_sub: list[list[float]],
    l1_sub: list[float],
    l2_sub: list[float],
    uty_sub: list[float],
) -> tuple[list[float], bool]:
    """Solve `(U_P^T U_P + diag(l2_P)) s_P = U_P^T y - l1_P` using thin Householder `R` factor."""
    k = len(cols_sub)
    if k == 0:
        return [], True
    r_mat, ok = _householder_qr_upper_stdlib(cols_sub, l2_sub)
    if not ok:
        return [0.0] * k, False

    rhs = [uty_sub[i] - l1_sub[i] for i in range(k)]
    # Forward solve R^T u = rhs
    u_vec = [0.0] * k
    for i in range(k):
        acc = rhs[i] - sum(r_mat[r][i] * u_vec[r] for r in range(i))
        u_vec[i] = acc / r_mat[i][i]
    # Back solve R s = u_vec
    s_vec = [0.0] * k
    for i in range(k - 1, -1, -1):
        acc = u_vec[i] - sum(r_mat[i][c] * s_vec[c] for c in range(i + 1, k))
        s_vec[i] = acc / r_mat[i][i]
    return s_vec, True


def _eval_elastic_kkt_stdlib(
    norm_cols: list[list[float]],
    scales: list[float],
    y_w: list[float],
    z: list[float],
    l1_tilde: list[float],
    l2_tilde: list[float],
) -> tuple[list[float], list[float], float]:
    m = len(y_w)
    p = len(norm_cols)
    fitted_w = [0.0] * m
    for j in range(p):
        zj = z[j]
        if zj > 0.0 and scales[j] > 0.0:
            uj = norm_cols[j]
            for i in range(m):
                fitted_w[i] += zj * uj[i]

    res = [y_w[i] - fitted_w[i] for i in range(m)]
    grad = [0.0] * p
    kkt_violation = 0.0
    for j in range(p):
        if scales[j] == 0.0:
            continue
        uj = norm_cols[j]
        gj = -sum(uj[i] * res[i] for i in range(m)) + l1_tilde[j] + l2_tilde[j] * z[j]
        grad[j] = gj
        viol = abs(gj) if z[j] > 1e-13 else max(0.0, -gj)
        if viol > kkt_violation:
            kkt_violation = viol
    return fitted_w, grad, kkt_violation


def _solve_elastic_nnls_stdlib(
    weighted_cols: list[list[float]],
    y_w: list[float],
    l1_vec: list[float],
    l2_vec: list[float],
    tol: float,
    max_iter: int,
) -> tuple[list[float], list[float], float, int]:
    m = len(y_w)
    p = len(weighted_cols)
    scales: list[float] = []
    norm_cols: list[list[float]] = []
    l1_tilde: list[float] = []
    l2_tilde: list[float] = []
    uty: list[float] = []

    for j, col in enumerate(weighted_cols):
        col_sq = sum(v * v for v in col)
        diag_h = col_sq + l2_vec[j]
        if col_sq > 1e-30 and diag_h > 1e-30:
            sj = math.sqrt(diag_h)
            scales.append(sj)
            uj = [v / sj for v in col]
            norm_cols.append(uj)
            l1_tilde.append(l1_vec[j] / sj)
            l2_tilde.append(l2_vec[j] / (sj * sj))
            uty.append(sum(uj[i] * y_w[i] for i in range(m)))
        else:
            scales.append(0.0)
            norm_cols.append([0.0] * m)
            l1_tilde.append(0.0)
            l2_tilde.append(0.0)
            uty.append(0.0)

    z = [0.0] * p
    passive: list[int] = []
    in_passive = [False] * p
    blocked: set[int] = set()
    iters = 0

    fitted_w, grad, kkt_violation = _eval_elastic_kkt_stdlib(
        norm_cols, scales, y_w, z, l1_tilde, l2_tilde
    )
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
            s_sub, full_rank = _solve_passive_elastic_stdlib(
                [norm_cols[j] for j in passive],
                [l1_tilde[j] for j in passive],
                [l2_tilde[j] for j in passive],
                [uty[j] for j in passive],
            )
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

        fitted_w, grad, kkt_violation = _eval_elastic_kkt_stdlib(
            norm_cols, scales, y_w, z, l1_tilde, l2_tilde
        )

    # Exact unit-diagonal coordinate-descent polish (H_jj == 1 exactly in z-coordinates)
    if kkt_violation > tol and iters < max_iter:
        gram: list[list[float]] = [[0.0] * p for _ in range(p)]
        for j in range(p):
            uj = norm_cols[j]
            for k in range(j, p):
                uk = norm_cols[k]
                g_jk = sum(uj[i] * uk[i] for i in range(m)) + (l2_tilde[j] if j == k else 0.0)
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
                if delta != 0.0:
                    z[j] = new_zj
                    row_j = gram[j]
                    for k in range(p):
                        grad[k] += row_j[k] * delta
                    if abs(delta) > max_step:
                        max_step = abs(delta)
            fitted_w, grad, kkt_violation = _eval_elastic_kkt_stdlib(
                norm_cols, scales, y_w, z, l1_tilde, l2_tilde
            )
            if kkt_violation <= tol and max_step <= tol:
                break

    weights = [z[j] / scales[j] if scales[j] > 0.0 else 0.0 for j in range(p)]
    return weights, fitted_w, kkt_violation, iters


def _eval_elastic_kkt_numpy(
    u_mat: Any,
    active: Any,
    y_vec: Any,
    z: Any,
    l1_tilde: Any,
    l2_tilde: Any,
) -> tuple[Any, Any, float]:
    assert _np is not None
    fitted_arr = u_mat @ z
    res_arr = y_vec - fitted_arr
    grad_arr = -(u_mat.T @ res_arr) + l1_tilde + l2_tilde * z
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


def _solve_elastic_nnls_numpy(
    weighted_cols: list[list[float]],
    y_w: list[float],
    l1_vec: list[float],
    l2_vec: list[float],
    tol: float,
    max_iter: int,
) -> tuple[list[float], list[float], float, int]:
    assert _np is not None
    mat = _np.asarray(weighted_cols, dtype=_np.float64).T  # shape (M, P)
    y_vec = _np.asarray(y_w, dtype=_np.float64)
    l1_arr = _np.asarray(l1_vec, dtype=_np.float64)
    l2_arr = _np.asarray(l2_vec, dtype=_np.float64)

    col_sq = _np.sum(mat * mat, axis=0)
    diag_h = col_sq + l2_arr
    active = (col_sq > 1e-30) & (diag_h > 1e-30)
    scales = _np.zeros(mat.shape[1], dtype=_np.float64)
    scales[active] = _np.sqrt(diag_h[active])

    u_mat = _np.zeros_like(mat)
    u_mat[:, active] = mat[:, active] / scales[active]
    l1_tilde = _np.zeros_like(l1_arr)
    l1_tilde[active] = l1_arr[active] / scales[active]
    l2_tilde = _np.zeros_like(l2_arr)
    l2_tilde[active] = l2_arr[active] / (scales[active] * scales[active])
    uty = u_mat.T @ y_vec

    p = int(mat.shape[1])
    z = _np.zeros(p, dtype=_np.float64)
    passive: list[int] = []
    in_passive = [False] * p
    blocked: set[int] = set()
    iters = 0

    fitted_arr, grad_arr, kkt_violation = _eval_elastic_kkt_numpy(
        u_mat, active, y_vec, z, l1_tilde, l2_tilde
    )
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
            k_sub = len(passive)
            u_sub = u_mat[:, passive]
            l2_sub = l2_tilde[passive]
            if bool(_np.any(l2_sub > 0.0)):
                aug = _np.vstack([u_sub, _np.diag(_np.sqrt(l2_sub))])
            else:
                aug = u_sub
            r_mat = _np.linalg.qr(aug, mode="r")
            diag_r = _np.abs(_np.diag(r_mat))
            full_rank = r_mat.shape == (k_sub, k_sub) and bool(_np.all(diag_r > 1e-12))
            if not full_rank:
                passive.pop()
                in_passive[best_j] = False
                blocked.add(best_j)
                break

            rhs = uty[passive] - l1_tilde[passive]
            try:
                u_step = _np.linalg.solve(r_mat.T, rhs)
                s_arr = _np.linalg.solve(r_mat, u_step)
            except _np.linalg.LinAlgError:  # pragma: no cover
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

        fitted_arr, grad_arr, kkt_violation = _eval_elastic_kkt_numpy(
            u_mat, active, y_vec, z, l1_tilde, l2_tilde
        )

    if kkt_violation > tol and iters < max_iter:
        gram = (u_mat.T @ u_mat) + _np.diag(l2_tilde)
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
            fitted_arr, grad_arr, kkt_violation = _eval_elastic_kkt_numpy(
                u_mat, active, y_vec, z, l1_tilde, l2_tilde
            )
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


def elastic_nnls(
    columns: Sequence[Sequence[float]],
    observed_y: Sequence[float],
    *,
    l1_alpha: float | Sequence[float] = 0.0,
    l2_alpha: float | Sequence[float] = 0.0,
    sample_weights: Sequence[float] | None = None,
    tol: float = 1e-10,
    max_iter: int = 2000,
    debias_support: bool = False,
    unpenalized_Debias_mask: Sequence[bool] | None = None,
    support_threshold: float = 1e-6,
    use_numpy: bool | None = None,
) -> ElasticNNLSResult:
    """Solve weighted sparsity-regularized `L1/L2` non-negative least squares (`elastic_nnls`).

    Minimizes over `c >= 0`:
        `0.5 * sum_i w_i * (y_i - (A c)_i)^2 + sum_j l1_j * c_j + 0.5 * sum_j l2_j * c_j^2`.
    """
    y = [max(0.0, _require_finite_float(v, f"observed_y[{i}]")) for i, v in enumerate(observed_y)]
    m = len(y)
    if m < 2:
        raise ValueError("observed_y must have length >= 2")

    p = len(columns)
    cols_clean: list[list[float]] = []
    for j, col in enumerate(columns):
        if len(col) != m:
            raise ValueError(f"columns[{j}] length {len(col)} != {m}")
        cols_clean.append(
            [_require_finite_float(v, f"columns[{j}][{i}]") for i, v in enumerate(col)]
        )

    tol_val = _require_finite_float(tol, "tol")
    thresh_val = _require_finite_float(support_threshold, "support_threshold")
    if tol_val <= 0.0:
        raise ValueError("tol must be positive")
    if thresh_val < 0.0:
        raise ValueError("support_threshold must be non-negative")
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")

    l1_vec = _expand_nonneg_vector(l1_alpha, p, "l1_alpha")
    l2_vec = _expand_nonneg_vector(l2_alpha, p, "l2_alpha")

    if sample_weights is None:
        w_vec = [1.0] * m
    else:
        if len(sample_weights) != m:
            raise ValueError("sample_weights length must match observed_y length")
        w_vec = []
        for i, w_raw in enumerate(sample_weights):
            wv = _require_finite_float(w_raw, f"sample_weights[{i}]")
            if wv < 0.0:
                raise ValueError(f"sample_weights[{i}] must be non-negative")
            w_vec.append(wv)
        if sum(w_vec) <= 0.0:
            raise ValueError("sample_weights must contain positive entries")

    run_np = (_np is not None) if use_numpy is None else bool(use_numpy)
    if run_np and _np is None:  # pragma: no cover
        raise RuntimeError("use_numpy=True requested, but numpy is not installed")

    if p == 0:
        obj0 = 0.5 * sum(w_vec[i] * y[i] * y[i] for i in range(m))
        return ElasticNNLSResult(
            weights=(),
            fitted=tuple([0.0] * m),
            objective=obj0,
            kkt_max_violation=0.0,
            iterations=0,
            converged=True,
            used_numpy=run_np,
            support_indices=(),
        )

    sqrt_w = [math.sqrt(w) for w in w_vec]
    y_w = [sqrt_w[i] * y[i] for i in range(m)]
    w_cols = [[sqrt_w[i] * cols_clean[j][i] for i in range(m)] for j in range(p)]

    solver_fn = _solve_elastic_nnls_numpy if run_np else _solve_elastic_nnls_stdlib
    weights, _, kkt_viol, iters = solver_fn(w_cols, y_w, l1_vec, l2_vec, tol_val, max_iter)

    support = [j for j, c_val in enumerate(weights) if c_val > thresh_val]
    if debias_support and any(v > 0.0 for v in l1_vec) and support:
        always_keep = (
            [j for j, keep in enumerate(unpenalized_Debias_mask) if keep]
            if unpenalized_Debias_mask is not None
            else []
        )
        kept = sorted(set(support) | set(always_keep))
        sub_cols = [w_cols[j] for j in kept]
        sub_l2 = [l2_vec[j] for j in kept]
        sub_w, _, sub_kkt, sub_iters = solver_fn(
            sub_cols, y_w, [0.0] * len(kept), sub_l2, tol_val, max_iter
        )
        full_w = [0.0] * p
        for local_i, orig_j in enumerate(kept):
            full_w[orig_j] = sub_w[local_i]
        weights = full_w
        kkt_viol = sub_kkt
        iters += sub_iters
        support = [j for j, c_val in enumerate(weights) if c_val > thresh_val]
        eval_l1 = [0.0] * p
    else:
        eval_l1 = l1_vec

    fitted = [0.0] * m
    for j, c_val in enumerate(weights):
        if c_val > 0.0:
            col_j = cols_clean[j]
            for i in range(m):
                fitted[i] += c_val * col_j[i]

    rss = 0.5 * sum(w_vec[i] * ((y[i] - fitted[i]) ** 2) for i in range(m))
    reg_l1 = sum(eval_l1[j] * weights[j] for j in range(p))
    reg_l2 = 0.5 * sum(l2_vec[j] * (weights[j] ** 2) for j in range(p))
    converged = kkt_viol <= max(tol_val * 100.0, 1e-6)

    return ElasticNNLSResult(
        weights=tuple(weights),
        fitted=tuple(fitted),
        objective=rss + reg_l1 + reg_l2,
        kkt_max_violation=kkt_viol,
        iterations=iters,
        converged=converged,
        used_numpy=run_np,
        support_indices=tuple(support),
    )


# ---------------------------------------------------------------------------
# 3. Golden-Section Shift Alignment + Pseudo-Voigt `L1-NNLS` Phase Attribution
# ---------------------------------------------------------------------------


def golden_section_minimize(
    func: Callable[[float], float],
    lo: float,
    hi: float,
    *,
    tol: float = 1e-4,
    max_evals: int = 36,
) -> tuple[float, float]:
    """Minimize unimodal 1D scalar `func` on `[lo, hi]` via exact Golden-Section Search."""
    a = float(lo)
    b = float(hi)
    if b <= a:
        val = func(a)
        return a, val

    c = b - _GOLDEN_PHI * (b - a)
    d = a + _GOLDEN_PHI * (b - a)
    fc = func(c)
    fd = func(d)
    evals = 2

    while (b - a) > tol and evals < max_evals:
        evals += 1
        if fc <= fd:
            b = d
            d = c
            fd = fc
            c = b - _GOLDEN_PHI * (b - a)
            fc = func(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + _GOLDEN_PHI * (b - a)
            fd = func(d)

    if fc <= fd:
        return c, fc
    return d, fd


@dataclass(frozen=True)
class MultiphaseAttributionResult:
    """Quantitative multi-mineral / multi-polymorph spectral attribution output."""

    method: str
    phase_ids: tuple[str, ...]
    phase_weights: tuple[float, ...]
    phase_fractions: tuple[float, ...]
    detected_phases: tuple[str, ...]
    rwp_permille: int
    fitted_zero_shift: float
    per_phase_shifts: tuple[float, ...]
    fitted_fwhm_broadening: float
    fitted_eta: float
    l1_alpha_used: float
    empirical_noise_rms: float
    kkt_max_violation: float
    converged: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "phase_ids": list(self.phase_ids),
            "phase_weights": [round(w, 6) for w in self.phase_weights],
            "phase_fractions": [round(f, 6) for f in self.phase_fractions],
            "detected_phases": list(self.detected_phases),
            "rwp_permille": self.rwp_permille,
            "fitted_zero_shift": round(self.fitted_zero_shift, 6),
            "per_phase_shifts": [round(s, 6) for s in self.per_phase_shifts],
            "fitted_fwhm_broadening": round(self.fitted_fwhm_broadening, 6),
            "fitted_eta": round(self.fitted_eta, 4),
            "l1_alpha_used": round(self.l1_alpha_used, 8),
            "empirical_noise_rms": round(self.empirical_noise_rms, 6),
            "kkt_max_violation": round(self.kkt_max_violation, 12),
            "converged": self.converged,
        }


def deconvolve_multiphase_library(
    observed_y: Sequence[float],
    grid_x: Sequence[float],
    library_phases: Sequence[tuple[str, Sequence[float]]],
    *,
    method: Literal[
        "naive_peak_cosine",
        "rigid_gaussian_nnls",
        "shift_voigt_l1_nnls",
    ] = "shift_voigt_l1_nnls",
    max_zero_shift: float = 0.35,
    max_per_phase_shift: float = 0.12,
    broadening_fwhm_candidates: Sequence[float] = (0.0, 0.18, 0.38, 0.65),
    pseudo_voigt_eta_candidates: Sequence[float] = (0.20, 0.55),
    background_poly_degree: int | None = 3,
    l1_alpha: float | None = None,
    l2_alpha: float = 1e-6,
    min_phase_fraction: float = 0.05,
    min_phase_amplitude: float = 0.04,
    naive_cosine_threshold: float = 0.32,
    is_angles_deg: bool = True,
    use_numpy: bool | None = None,
) -> MultiphaseAttributionResult:
    """Identify and quantify crystalline/vibrational phases from a candidate reference library.

    Supports direct head-to-head comparison across:
    - `naive_peak_cosine`: Independent cosine / top-peak correlation thresholding (baseline).
    - `rigid_gaussian_nnls`: Fixed-grid, zero-shift, unregularized NNLS (`l1=0`, `shift=0`).
    - `shift_voigt_l1_nnls`: Coarse-grid + Golden-Section zero-point `+-0.35 deg` alignment,
      Pseudo-Voigt instrumental/crystallite broadening selection, per-phase Vegard shift polish,
      Bernstein amorphous baseline subtraction, and noise-adaptive `L1-NNLS` + support debiasing.
    """
    y_raw = [max(0.0, _require_finite_float(v, f"observed_y[{i}]")) for i, v in enumerate(observed_y)]
    m = len(y_raw)
    if m < 4 or len(grid_x) != m:
        raise ValueError("observed_y and grid_x must have matching length >= 4")
    xs = [_require_finite_float(v, f"grid_x[{i}]") for i, v in enumerate(grid_x)]
    for i in range(m - 1):
        if xs[i + 1] <= xs[i]:
            raise ValueError("grid_x must be strictly increasing")
    if not library_phases:
        raise ValueError("library_phases must contain at least 1 candidate phase")

    y_max = max(y_raw)
    if y_max <= 1e-15:
        phase_ids_zero = tuple(pid for pid, _ in library_phases)
        k0 = len(phase_ids_zero)
        return MultiphaseAttributionResult(
            method=method,
            phase_ids=phase_ids_zero,
            phase_weights=tuple([0.0] * k0),
            phase_fractions=tuple([0.0] * k0),
            detected_phases=(),
            rwp_permille=0,
            fitted_zero_shift=0.0,
            per_phase_shifts=tuple([0.0] * k0),
            fitted_fwhm_broadening=0.0,
            fitted_eta=0.5,
            l1_alpha_used=0.0,
            empirical_noise_rms=0.0,
            kkt_max_violation=0.0,
            converged=True,
        )

    # Peak-normalize observed and reference profiles to [0, 1] for scale-invariant L1 regularization
    y = [v / y_max for v in y_raw]
    obs_sq = max(1e-15, sum(v * v for v in y))
    obs_norm = math.sqrt(obs_sq)

    phase_ids: list[str] = []
    unit_max_bases: list[list[float]] = []
    for idx, (pid, raw_spec) in enumerate(library_phases):
        if len(raw_spec) != m:
            raise ValueError(f"library_phases[{idx}] ({pid}) length {len(raw_spec)} != {m}")
        cleaned = [max(0.0, _require_finite_float(v, f"{pid}[{i}]")) for i, v in enumerate(raw_spec)]
        p_max = max(cleaned)
        if p_max <= 1e-15:
            raise ValueError(f"library_phases[{idx}] ({pid}) must have non-zero intensity")
        phase_ids.append(str(pid))
        unit_max_bases.append([v / p_max for v in cleaned])

    k_phases = len(phase_ids)
    hf_diff_sq = sum((y[i + 1] - y[i]) ** 2 for i in range(m - 1))
    emp_noise = math.sqrt(hf_diff_sq / float(max(1, 2 * (m - 1))))

    # --- Baseline 1: Naive independent peak/cosine matcher ---
    if method == "naive_peak_cosine":
        # Estimate smooth rolling-minimum baseline for fair naive peak correlation
        win = max(5, m // 16)
        y_sub: list[float] = []
        for i in range(m):
            lo_w = max(0, i - win)
            hi_w = min(m, i + win + 1)
            y_sub.append(max(0.0, y[i] - min(y[lo_w:hi_w])))
        y_sub_norm = math.sqrt(sum(v * v for v in y_sub)) or obs_norm

        cos_scores: list[float] = []
        for col in unit_max_bases:
            c_norm = math.sqrt(sum(v * v for v in col))
            cos_v = sum(y_sub[i] * col[i] for i in range(m)) / max(1e-15, y_sub_norm * c_norm)
            cos_scores.append(max(0.0, cos_v))
        raw_w = [sc if sc >= naive_cosine_threshold else 0.0 for sc in cos_scores]
        sum_w = sum(raw_w)
        fracs = [w / sum_w if sum_w > 0.0 else 0.0 for w in raw_w]
        # Best scalar least-squares scale of naive combination for R_wp reporting
        combo = [sum(fracs[j] * unit_max_bases[j][i] for j in range(k_phases)) for i in range(m)]
        combo_sq = sum(v * v for v in combo)
        scale_opt = (sum(y[i] * combo[i] for i in range(m)) / combo_sq) if combo_sq > 1e-15 else 0.0
        fitted_naive = [scale_opt * v for v in combo]
        rwp = math.sqrt(sum((y[i] - fitted_naive[i]) ** 2 for i in range(m)) / obs_sq)
        detected = tuple(phase_ids[j] for j, w in enumerate(raw_w) if w >= naive_cosine_threshold)
        return MultiphaseAttributionResult(
            method=method,
            phase_ids=tuple(phase_ids),
            phase_weights=tuple(scale_opt * f for f in fracs),
            phase_fractions=tuple(fracs),
            detected_phases=detected,
            rwp_permille=min(1000, max(0, int(round(1000.0 * rwp)))),
            fitted_zero_shift=0.0,
            per_phase_shifts=tuple([0.0] * k_phases),
            fitted_fwhm_broadening=0.0,
            fitted_eta=0.0,
            l1_alpha_used=0.0,
            empirical_noise_rms=emp_noise,
            kkt_max_violation=0.0,
            converged=True,
        )

    # --- Baseline 2: Rigid fixed-grid Gaussian NNLS (`shift=0`, `broadening=0`, `l1=0`) ---
    if method == "rigid_gaussian_nnls":
        res_rigid = elastic_nnls(
            unit_max_bases,
            y,
            l1_alpha=0.0,
            l2_alpha=0.0,
            use_numpy=use_numpy,
        )
        p_weights = list(res_rigid.weights)
        sum_pw = sum(p_weights)
        fracs = [w / sum_pw if sum_pw > 1e-12 else 0.0 for w in p_weights]
        detected = tuple(
            phase_ids[j]
            for j in range(k_phases)
            if p_weights[j] >= min_phase_amplitude and fracs[j] >= min_phase_fraction
        )
        rwp = math.sqrt(sum((y[i] - res_rigid.fitted[i]) ** 2 for i in range(m)) / obs_sq)
        return MultiphaseAttributionResult(
            method=method,
            phase_ids=tuple(phase_ids),
            phase_weights=tuple(p_weights),
            phase_fractions=tuple(fracs),
            detected_phases=detected,
            rwp_permille=min(1000, max(0, int(round(1000.0 * rwp)))),
            fitted_zero_shift=0.0,
            per_phase_shifts=tuple([0.0] * k_phases),
            fitted_fwhm_broadening=0.0,
            fitted_eta=0.0,
            l1_alpha_used=0.0,
            empirical_noise_rms=emp_noise,
            kkt_max_violation=res_rigid.kkt_max_violation,
            converged=res_rigid.converged,
        )

    if method != "shift_voigt_l1_nnls":
        raise ValueError(f"Unsupported method {method!r}")

    # --- Frontier Solver: Shift-Tolerant Pseudo-Voigt `L1-NNLS` ---
    bg_cols = (
        _build_bernstein_background_columns(m, background_poly_degree)
        if background_poly_degree is not None
        else []
    )
    n_bg = len(bg_cols)

    # Noise- and norm-calibrated L1 sparsity weight (penalizes crystalline phases, not smooth background)
    mean_col_norm_sq = sum(sum(v * v for v in col) for col in unit_max_bases) / float(k_phases)
    auto_l1 = max(0.012 * mean_col_norm_sq, 1.8 * emp_noise * math.sqrt(mean_col_norm_sq))
    eff_l1 = _require_finite_float(l1_alpha, "l1_alpha") if l1_alpha is not None else auto_l1
    l1_vec = [eff_l1] * k_phases + [0.0] * n_bg
    l2_vec = [l2_alpha] * k_phases + [1e-4] * n_bg
    bg_mask = [False] * k_phases + [True] * n_bg

    # Precompute Pseudo-Voigt broadened library variants `(fwhm_add, eta)`
    broadened_bank: list[tuple[float, float, list[list[float]]]] = []
    seen_zero = False
    for w_cand in broadening_fwhm_candidates:
        if w_cand <= 1e-9:
            if not seen_zero:
                broadened_bank.append((0.0, 0.0, unit_max_bases))
                seen_zero = True
            continue
        for eta_cand in pseudo_voigt_eta_candidates:
            b_cols: list[list[float]] = []
            for base_col in unit_max_bases:
                conv_col = convolve_pseudo_voigt_broadening(
                    base_col, xs, fwhm_add=w_cand, eta=eta_cand
                )
                c_max = max(conv_col) or 1.0
                b_cols.append([v / c_max for v in conv_col])
            broadened_bank.append((float(w_cand), float(eta_cand), b_cols))
    if not broadened_bank:
        broadened_bank.append((0.0, 0.0, unit_max_bases))

    def _eval_shift_and_broadening(
        shift_val: float,
        bases_for_width: list[list[float]],
        *,
        debias: bool = False,
    ) -> ElasticNNLSResult:
        warped = [
            _catmull_rom_warp(col, xs, shift_val, 0.0, is_angles_deg=is_angles_deg)
            for col in bases_for_width
        ]
        return elastic_nnls(
            warped + bg_cols,
            y,
            l1_alpha=l1_vec,
            l2_alpha=l2_vec,
            debias_support=debias,
            unpenalized_Debias_mask=bg_mask,
            support_threshold=min_phase_amplitude * 0.5,
            use_numpy=use_numpy,
        )

    # Step A: Coarse grid scan over zero-point shift in [-max_zero_shift, +max_zero_shift] x Pseudo-Voigt widths
    shift_bound = max(0.0, _require_finite_float(max_zero_shift, "max_zero_shift"))
    if shift_bound > 0.0:
        n_coarse = 9
        coarse_shifts = [
            -shift_bound + 2.0 * shift_bound * (idx / float(n_coarse - 1))
            for idx in range(n_coarse)
        ]
    else:
        coarse_shifts = [0.0]

    best_bank_idx = 0
    best_coarse_idx = 0
    best_coarse_obj = float("inf")
    coarse_step = (2.0 * shift_bound / float(max(1, len(coarse_shifts) - 1))) if shift_bound > 0.0 else 0.0

    for b_idx, (_, _, b_cols) in enumerate(broadened_bank):
        for s_idx, sh in enumerate(coarse_shifts):
            res_c = _eval_shift_and_broadening(sh, b_cols, debias=False)
            if res_c.objective < best_coarse_obj:
                best_coarse_obj = res_c.objective
                best_bank_idx = b_idx
                best_coarse_idx = s_idx

    best_fwhm, best_eta, chosen_bases = broadened_bank[best_bank_idx]

    # Step B: Continuous Golden-Section Search around optimal coarse zero-point bracket
    if shift_bound > 0.0:
        center_sh = coarse_shifts[best_coarse_idx]
        gs_lo = max(-shift_bound, center_sh - 1.25 * coarse_step)
        gs_hi = min(shift_bound, center_sh + 1.25 * coarse_step)
        opt_zero_shift, _ = golden_section_minimize(
            lambda s: _eval_shift_and_broadening(s, chosen_bases, debias=False).objective,
            gs_lo,
            gs_hi,
            tol=1e-4,
            max_evals=28,
        )
    else:
        opt_zero_shift = 0.0

    initial_fit = _eval_shift_and_broadening(opt_zero_shift, chosen_bases, debias=True)
    phase_shifts = [opt_zero_shift] * k_phases
    active_phase_indices = [
        j for j in range(k_phases) if initial_fit.weights[j] >= min_phase_amplitude * 0.5
    ]

    # Step C: Per-phase solid-solution Vegard shift polish via Golden-Section on active phases
    phase_shift_bound = max(0.0, _require_finite_float(max_per_phase_shift, "max_per_phase_shift"))
    if phase_shift_bound > 0.0 and active_phase_indices:
        warped_active_cols = [
            _catmull_rom_warp(chosen_bases[j], xs, phase_shifts[j], 0.0, is_angles_deg=is_angles_deg)
            for j in active_phase_indices
        ]
        for local_a, j_idx in enumerate(active_phase_indices):
            other_cols = [
                warped_active_cols[idx_o]
                for idx_o in range(len(active_phase_indices))
                if idx_o != local_a
            ]

            def _local_phase_rss(delta_sh: float) -> float:
                cand_col = _catmull_rom_warp(
                    chosen_bases[j_idx],
                    xs,
                    opt_zero_shift + delta_sh,
                    0.0,
                    is_angles_deg=is_angles_deg,
                )
                cols_loc = [cand_col] + other_cols + bg_cols
                l2_loc = [l2_alpha] * (1 + len(other_cols)) + [1e-4] * n_bg
                fit_loc = elastic_nnls(
                    cols_loc, y, l1_alpha=0.0, l2_alpha=l2_loc, use_numpy=use_numpy
                )
                return fit_loc.objective

            best_delta, _ = golden_section_minimize(
                _local_phase_rss,
                -phase_shift_bound,
                phase_shift_bound,
                tol=2e-4,
                max_evals=20,
            )
            phase_shifts[j_idx] = opt_zero_shift + best_delta
            warped_active_cols[local_a] = _catmull_rom_warp(
                chosen_bases[j_idx], xs, phase_shifts[j_idx], 0.0, is_angles_deg=is_angles_deg
            )

    # Final continuous Pseudo-Voigt width & eta polish on detected active phases + debiased refit
    if active_phase_indices and broadening_fwhm_candidates:
        max_w_cand = max(float(w) for w in broadening_fwhm_candidates)
        if max_w_cand > 0.0:
            raw_active_unbroadened = [
                _catmull_rom_warp(
                    unit_max_bases[j], xs, phase_shifts[j], 0.0, is_angles_deg=is_angles_deg
                )
                for j in active_phase_indices
            ]
            best_pol_rss = float("inf")
            best_pol_w = best_fwhm
            best_pol_eta = best_eta
            for eta_try in (0.0, 0.35, 0.65, 0.90):
                def _width_rss(w_try: float) -> float:
                    if w_try <= 1e-6:
                        cols_w = raw_active_unbroadened
                    else:
                        cols_w = []
                        for col_u in raw_active_unbroadened:
                            cv = convolve_pseudo_voigt_broadening(
                                col_u, xs, fwhm_add=w_try, eta=eta_try
                            )
                            cm = max(cv) or 1.0
                            cols_w.append([v / cm for v in cv])
                    l2_w = [l2_alpha] * len(cols_w) + [1e-4] * n_bg
                    return elastic_nnls(
                        cols_w + bg_cols, y, l1_alpha=0.0, l2_alpha=l2_w, use_numpy=use_numpy
                    ).objective

                w_opt, rss_opt = golden_section_minimize(
                    _width_rss, 0.0, max_w_cand * 1.15, tol=5e-3, max_evals=14
                )
                if rss_opt < best_pol_rss:
                    best_pol_rss = rss_opt
                    best_pol_w = w_opt
                    best_pol_eta = eta_try if w_opt > 1e-4 else 0.0
            best_fwhm = best_pol_w
            best_eta = best_pol_eta
            if best_fwhm > 1e-4:
                chosen_bases = []
                for base_col in unit_max_bases:
                    cv = convolve_pseudo_voigt_broadening(
                        base_col, xs, fwhm_add=best_fwhm, eta=best_eta
                    )
                    cm = max(cv) or 1.0
                    chosen_bases.append([v / cm for v in cv])
            else:
                chosen_bases = unit_max_bases

    # Final sparse L1-NNLS + support-debiased Rietveld refit on aligned Pseudo-Voigt library
    final_phase_cols = [
        _catmull_rom_warp(chosen_bases[j], xs, phase_shifts[j], 0.0, is_angles_deg=is_angles_deg)
        for j in range(k_phases)
    ]
    final_res = elastic_nnls(
        final_phase_cols + bg_cols,
        y,
        l1_alpha=l1_vec,
        l2_alpha=l2_vec,
        debias_support=True,
        unpenalized_Debias_mask=bg_mask,
        support_threshold=min_phase_amplitude * 0.6,
        use_numpy=use_numpy,
    )

    raw_phase_w = list(final_res.weights[:k_phases])
    pruned_w = [w if w >= min_phase_amplitude else 0.0 for w in raw_phase_w]
    sum_pw = sum(pruned_w)
    fracs = [w / sum_pw if sum_pw > 1e-12 else 0.0 for w in pruned_w]
    detected = tuple(
        phase_ids[j]
        for j in range(k_phases)
        if pruned_w[j] >= min_phase_amplitude and fracs[j] >= min_phase_fraction
    )
    # Remove any sub-fraction phase and compute final R_wp
    for j in range(k_phases):
        if phase_ids[j] not in detected:
            pruned_w[j] = 0.0
    sum_clean = sum(pruned_w)
    fracs_clean = [w / sum_clean if sum_clean > 1e-12 else 0.0 for w in pruned_w]

    rwp = math.sqrt(sum((y[i] - final_res.fitted[i]) ** 2 for i in range(m)) / obs_sq)
    res_unw = [y[i] - final_res.fitted[i] for i in range(m)]
    post_hf_rms = math.sqrt(
        sum((res_unw[i + 1] - res_unw[i]) ** 2 for i in range(m - 1)) / float(max(1, 2 * (m - 1)))
    )

    return MultiphaseAttributionResult(
        method=method,
        phase_ids=tuple(phase_ids),
        phase_weights=tuple(pruned_w),
        phase_fractions=tuple(fracs_clean),
        detected_phases=detected,
        rwp_permille=min(1000, max(0, int(round(1000.0 * rwp)))),
        fitted_zero_shift=opt_zero_shift,
        per_phase_shifts=tuple(phase_shifts),
        fitted_fwhm_broadening=best_fwhm,
        fitted_eta=best_eta,
        l1_alpha_used=eff_l1,
        empirical_noise_rms=post_hf_rms,
        kkt_max_violation=final_res.kkt_max_violation,
        converged=final_res.converged,
    )


def deconvolve_frontier_spectrum(
    observed_y: Sequence[float],
    precursor_spectra: Sequence[Sequence[float]],
    target_spectrum: Sequence[float],
    *,
    two_theta_grid: Sequence[float],
    max_peak_shift_deg: float = 0.35,
    max_precursor_shift_deg: float | None = None,
    broadening_fwhm_candidates: Sequence[float] = (0.0, 0.16, 0.34, 0.55),
    pseudo_voigt_eta: float = 0.45,
    background_poly_degree: int | None = 3,
    l1_alpha: float | None = None,
    min_hit_amplitude: float = 0.20,
    noise_sigma: float = 0.025,
    naive_cosine_threshold: float = 0.70,
    weights: Sequence[float] | None = None,
    use_numpy: bool | None = None,
) -> DeconvolutionResult:
    """Shift-tolerant (`+-0.35 deg` Golden-Section) + Pseudo-Voigt + `L1-NNLS` target/precursor deconvolver."""
    y = [max(0.0, _require_finite_float(v, f"observed_y[{i}]")) for i, v in enumerate(observed_y)]
    m = len(y)
    if m < 4 or len(two_theta_grid) != m or len(target_spectrum) != m:
        raise ValueError("observed_y, target_spectrum, and two_theta_grid must match length >= 4")
    xs = [_require_finite_float(v, f"two_theta_grid[{i}]") for i, v in enumerate(two_theta_grid)]
    for i in range(m - 1):
        if xs[i + 1] <= xs[i]:
            raise ValueError("two_theta_grid must be strictly increasing")

    s_target = [max(0.0, _require_finite_float(v, f"target[{i}]")) for i, v in enumerate(target_spectrum)]
    target_norm = math.sqrt(sum(v * v for v in s_target))
    if target_norm <= 1e-15:
        raise ValueError("target_spectrum must have non-zero L2 norm")

    prec_cols = [
        [max(0.0, _require_finite_float(v, f"prec[{j}][{i}]")) for i, v in enumerate(p_spec)]
        for j, p_spec in enumerate(precursor_spectra)
    ]
    for j, col in enumerate(prec_cols):
        if len(col) != m:
            raise ValueError(f"precursor_spectra[{j}] length {len(col)} != {m}")

    w_profile = [1.0] * m if weights is None else [
        _require_finite_float(w, f"weights[{i}]") for i, w in enumerate(weights)
    ]
    bg_cols = (
        _build_bernstein_background_columns(m, background_poly_degree)
        if background_poly_degree is not None
        else []
    )
    k_prec = len(prec_cols)
    n_bg = len(bg_cols)

    hf_diff_sq = sum((y[i + 1] - y[i]) ** 2 for i in range(m - 1))
    emp_hf_rms = math.sqrt(hf_diff_sq / float(max(1, 2 * (m - 1))))
    eff_l1 = (
        _require_finite_float(l1_alpha, "l1_alpha")
        if l1_alpha is not None
        else max(0.005 * target_norm, 1.25 * emp_hf_rms * target_norm)
    )
    l1_vec = [eff_l1] * (k_prec + 1) + [0.0] * n_bg
    l2_vec = [1e-6] * (k_prec + 1) + [1e-4] * n_bg
    bg_mask = [False] * (k_prec + 1) + [True] * n_bg

    # Build broadened phase candidates while preserving peak scale relative to input reference
    all_raw_phases = prec_cols + [s_target]
    width_candidates = [0.0] + [float(w) for w in broadening_fwhm_candidates if w > 1e-9]
    broadened_sets: list[tuple[float, list[list[float]]]] = []
    for w_add in width_candidates:
        if w_add <= 1e-9:
            broadened_sets.append((0.0, all_raw_phases))
        else:
            b_set: list[list[float]] = []
            for ph in all_raw_phases:
                orig_max = max(ph)
                conv = convolve_pseudo_voigt_broadening(ph, xs, fwhm_add=w_add, eta=pseudo_voigt_eta)
                conv_max = max(conv)
                scale = (orig_max / conv_max) if (orig_max > 1e-15 and conv_max > 1e-15) else 1.0
                b_set.append([v * scale for v in conv])
            broadened_sets.append((w_add, b_set))

    prec_shift_bound = (
        max(0.0, _require_finite_float(max_precursor_shift_deg, "max_precursor_shift_deg"))
        if max_precursor_shift_deg is not None
        else None
    )

    def _eval_at(shift_deg: float, ph_set: list[list[float]], *, debias: bool = False) -> ElasticNNLSResult:
        if prec_shift_bound is None:
            warped = [_catmull_rom_warp(col, xs, shift_deg, 0.0, is_angles_deg=True) for col in ph_set]
        else:
            p_sh = max(-prec_shift_bound, min(prec_shift_bound, shift_deg))
            warped = [
                _catmull_rom_warp(ph_set[j], xs, p_sh, 0.0, is_angles_deg=True)
                for j in range(k_prec)
            ] + [_catmull_rom_warp(ph_set[k_prec], xs, shift_deg, 0.0, is_angles_deg=True)]
        return elastic_nnls(
            warped + bg_cols,
            y,
            l1_alpha=l1_vec,
            l2_alpha=l2_vec,
            sample_weights=w_profile,
            debias_support=debias,
            unpenalized_Debias_mask=bg_mask,
            support_threshold=min_hit_amplitude * 0.25,
            use_numpy=use_numpy,
        )

    shift_bound = max(0.0, _require_finite_float(max_peak_shift_deg, "max_peak_shift_deg"))
    coarse_shifts = (
        [-shift_bound + 2.0 * shift_bound * (k / 8.0) for k in range(9)]
        if shift_bound > 0.0
        else [0.0]
    )
    coarse_step = (2.0 * shift_bound / 8.0) if shift_bound > 0.0 else 0.0

    best_w_idx = 0
    best_s_idx = 0
    best_obj = float("inf")
    for w_idx, (_, ph_set) in enumerate(broadened_sets):
        for s_idx, sh in enumerate(coarse_shifts):
            fit_c = _eval_at(sh, ph_set, debias=False)
            if fit_c.objective < best_obj:
                best_obj = fit_c.objective
                best_w_idx = w_idx
                best_s_idx = s_idx

    best_w_add, chosen_ph_set = broadened_sets[best_w_idx]
    if shift_bound > 0.0:
        c_sh = coarse_shifts[best_s_idx]
        opt_shift, _ = golden_section_minimize(
            lambda s: _eval_at(s, chosen_ph_set, debias=False).objective,
            max(-shift_bound, c_sh - 1.25 * coarse_step),
            min(shift_bound, c_sh + 1.25 * coarse_step),
            tol=1e-4,
            max_evals=28,
        )
    else:
        opt_shift = 0.0

    if prec_shift_bound is None:
        opt_p_shift = opt_shift
        warped_chosen = [
            _catmull_rom_warp(col, xs, opt_shift, 0.0, is_angles_deg=True) for col in chosen_ph_set
        ]
    else:
        # Refine precursor specimen-displacement zero-point independently within +-prec_shift_bound
        aligned_t_fixed = _catmull_rom_warp(
            chosen_ph_set[k_prec], xs, opt_shift, 0.0, is_angles_deg=True
        )
        if k_prec > 0 and prec_shift_bound > 0.0:
            opt_p_shift, _ = golden_section_minimize(
                lambda ps: elastic_nnls(
                    [
                        _catmull_rom_warp(chosen_ph_set[j], xs, ps, 0.0, is_angles_deg=True)
                        for j in range(k_prec)
                    ]
                    + [aligned_t_fixed]
                    + bg_cols,
                    y,
                    l1_alpha=l1_vec,
                    l2_alpha=l2_vec,
                    sample_weights=w_profile,
                    use_numpy=use_numpy,
                ).objective,
                -prec_shift_bound,
                prec_shift_bound,
                tol=2e-4,
                max_evals=18,
            )
        else:
            opt_p_shift = 0.0
        warped_chosen = [
            _catmull_rom_warp(chosen_ph_set[j], xs, opt_p_shift, 0.0, is_angles_deg=True)
            for j in range(k_prec)
        ] + [aligned_t_fixed]

    # Continuous Pseudo-Voigt FWHM & Lorentzian/Gaussian mixing polish on aligned phases
    max_w_cand = max(width_candidates) if width_candidates else 0.0
    if max_w_cand > 0.0:
        warped_raw = [
            _catmull_rom_warp(all_raw_phases[j], xs, opt_p_shift, 0.0, is_angles_deg=True)
            for j in range(k_prec)
        ] + [_catmull_rom_warp(all_raw_phases[k_prec], xs, opt_shift, 0.0, is_angles_deg=True)]
        best_pol_obj = elastic_nnls(
            warped_chosen + bg_cols,
            y,
            l1_alpha=l1_vec,
            l2_alpha=l2_vec,
            sample_weights=w_profile,
            use_numpy=use_numpy,
        ).objective

        def _broaden_warped_raw(w_try: float, eta_try: float) -> list[list[float]]:
            if w_try <= 1e-6:
                return warped_raw
            cols_out: list[list[float]] = []
            for ph_u in warped_raw:
                u_max = max(ph_u)
                cv = convolve_pseudo_voigt_broadening(ph_u, xs, fwhm_add=w_try, eta=eta_try)
                c_max = max(cv)
                sc = (u_max / c_max) if (u_max > 1e-15 and c_max > 1e-15) else 1.0
                cols_out.append([v * sc for v in cv])
            return cols_out

        best_pol_cols: list[list[float]] | None = None
        for eta_try in (0.0, 0.35, 0.65, 0.90):
            w_opt, obj_opt = golden_section_minimize(
                lambda wt, et=eta_try: elastic_nnls(
                    _broaden_warped_raw(wt, et) + bg_cols,
                    y,
                    l1_alpha=l1_vec,
                    l2_alpha=l2_vec,
                    sample_weights=w_profile,
                    use_numpy=use_numpy,
                ).objective,
                0.0,
                max_w_cand * 1.15,
                tol=5e-3,
                max_evals=14,
            )
            if obj_opt + 1e-12 < best_pol_obj:
                best_pol_obj = obj_opt
                best_pol_cols = _broaden_warped_raw(w_opt, eta_try)
        if best_pol_cols is not None:
            warped_chosen = best_pol_cols
    final_fit = elastic_nnls(
        warped_chosen + bg_cols,
        y,
        l1_alpha=l1_vec,
        l2_alpha=l2_vec,
        sample_weights=w_profile,
        debias_support=True,
        unpenalized_Debias_mask=bg_mask,
        support_threshold=min_hit_amplitude * 0.25,
        use_numpy=use_numpy,
    )

    prec_w = tuple(final_fit.weights[:k_prec])
    beta_target = float(final_fit.weights[k_prec])
    bg_w = tuple(final_fit.weights[k_prec + 1 :])

    y_prec = [0.0] * m
    for j in range(k_prec):
        cj = prec_w[j]
        if cj > 0.0:
            col_j = warped_chosen[j]
            for i in range(m):
                y_prec[i] += cj * col_j[i]

    aligned_target = warped_chosen[k_prec]
    y_target = [beta_target * v for v in aligned_target]
    fitted = list(final_fit.fitted)

    min_ref_sq = sum(w_profile[i] * ((min_hit_amplitude * aligned_target[i]) ** 2) for i in range(m))
    weighted_err_sq = sum(w_profile[i] * ((y[i] - fitted[i]) ** 2) for i in range(m))
    weighted_obs_sq = sum(w_profile[i] * (y[i] ** 2) for i in range(m))
    r_wp = math.sqrt(weighted_err_sq / max(weighted_obs_sq, min_ref_sq, 1e-18))
    spectral_residual_permille = min(1000, max(0, int(round(1000.0 * r_wp))))

    aligned_t_norm = math.sqrt(sum(v * v for v in aligned_target)) or target_norm
    s_eff_unit = [v / aligned_t_norm for v in aligned_target]
    target_on_target = sum(y_target[i] * s_eff_unit[i] for i in range(m))
    prec_on_target = sum(y_prec[i] * s_eff_unit[i] for i in range(m))

    floor_intensity = min_hit_amplitude * aligned_t_norm
    overlap_ratio = prec_on_target / (prec_on_target + max(target_on_target, floor_intensity))
    precursor_overlap_permille = min(1000, max(0, int(round(1000.0 * overlap_ratio))))

    res_unw = [y[i] - fitted[i] for i in range(m)]
    post_hf_rms = math.sqrt(
        sum((res_unw[i + 1] - res_unw[i]) ** 2 for i in range(m - 1)) / float(max(1, 2 * (m - 1)))
    )
    eff_noise = max(noise_sigma, post_hf_rms)
    snr_tenths = (
        max(0, int(round(10.0 * (max(target_on_target, floor_intensity) / eff_noise))))
        if final_fit.converged
        else 0
    )

    s_unit_unshifted = [v / target_norm for v in s_target]
    obs_norm = math.sqrt(sum(v * v for v in y))
    naive_cos = (
        max(0.0, min(1.0, sum(y[i] * s_unit_unshifted[i] for i in range(m)) / obs_norm))
        if obs_norm > 1e-12
        else 0.0
    )
    max_basis_cos = 0.0
    for j in range(k_prec):
        p_norm = math.sqrt(sum(v * v for v in warped_chosen[j]))
        if p_norm > 1e-15:
            cos_pt = sum(warped_chosen[j][i] * s_eff_unit[i] for i in range(m)) / p_norm
            if cos_pt > max_basis_cos:
                max_basis_cos = min(1.0, max(0.0, cos_pt))

    notes: list[str] = [f"PSEUDO_VOIGT_FWHM_ADD={best_w_add:.3f}", f"L1_ALPHA={eff_l1:.6f}"]
    if max_basis_cos >= 0.995:
        notes.append("NEAR_COLLINEAR_BASIS")

    return DeconvolutionResult(
        precursor_weights=prec_w,
        target_amplitude=beta_target,
        precursor_overlap_permille=precursor_overlap_permille,
        spectral_residual_permille=spectral_residual_permille,
        snr_tenths=snr_tenths,
        target_signal_present=bool(final_fit.converged and beta_target >= min_hit_amplitude),
        naive_peak_match_cosine=naive_cos,
        naive_claims_hit=bool(naive_cos >= naive_cosine_threshold and obs_norm >= 0.5 * floor_intensity),
        kkt_max_violation=final_fit.kkt_max_violation,
        iterations=final_fit.iterations,
        used_numpy=final_fit.used_numpy,
        converged=final_fit.converged,
        fitted_shift_deg=opt_shift,
        max_basis_cosine=max_basis_cos,
        background_weights=bg_w,
        empirical_noise_rms=post_hf_rms,
        diagnostic_notes=tuple(notes),
    )


# ---------------------------------------------------------------------------
# 4. MS/MS Spectral Entropy Similarity (Li et al., Nature Methods 2021) & Chimeras
# ---------------------------------------------------------------------------


def _prepare_ms_probability_peaks(
    mzs: Sequence[float],
    intensities: Sequence[float],
    *,
    weighted: bool,
    clean_noise_rel: float = 0.01,
    merge_within_da: float = 0.0,
) -> tuple[list[float], list[float], float]:
    """Normalize MS/MS peaks to `sum(p) == 1` and optionally apply Li et al. (2021) dynamic weighting."""
    if len(mzs) != len(intensities):
        raise ValueError("mzs and intensities must have matching length")
    if not mzs:
        return [], [], 0.0

    raw_pairs = sorted(
        (_require_finite_float(mz, "mz"), max(0.0, _require_finite_float(inten, "intensity")))
        for mz, inten in zip(mzs, intensities)
    )
    max_i = max((inten for _, inten in raw_pairs), default=0.0)
    if max_i <= 0.0:
        return [], [], 0.0

    floor = max(0.0, clean_noise_rel) * max_i
    filtered = [(mz, inten) for mz, inten in raw_pairs if inten > 0.0 and inten >= floor]
    if not filtered:
        filtered = [(mz, inten) for mz, inten in raw_pairs if inten > 0.0]

    # Centroid adjacent centroids if merge_within_da > 0
    if merge_within_da > 0.0 and len(filtered) > 1:
        merged: list[tuple[float, float]] = []
        cur_mz, cur_i = filtered[0]
        for mz, inten in filtered[1:]:
            if (mz - cur_mz) <= merge_within_da:
                tot = cur_i + inten
                cur_mz = (cur_mz * cur_i + mz * inten) / max(1e-15, tot)
                cur_i = tot
            else:
                merged.append((cur_mz, cur_i))
                cur_mz, cur_i = mz, inten
        merged.append((cur_mz, cur_i))
        filtered = merged

    sum_i = sum(inten for _, inten in filtered)
    probs = [inten / sum_i for _, inten in filtered]
    raw_entropy = -sum(p * math.log(p) for p in probs if p > 0.0)

    if weighted and raw_entropy < 3.0:
        # Li et al., Nature Methods 2021 Eq. 3: w = 0.25 + 0.25 * S for S < 3 (natural log)
        weight_power = 0.25 + 0.25 * raw_entropy
        warped = [p ** weight_power for p in probs]
        sum_w = sum(warped)
        probs = [v / sum_w for v in warped]
        eff_entropy = -sum(p * math.log(p) for p in probs if p > 0.0)
    else:
        eff_entropy = raw_entropy

    return [mz for mz, _ in filtered], probs, eff_entropy


def spectral_entropy(
    intensities: Sequence[float],
    *,
    weighted: bool = False,
    clean_noise_rel: float = 0.0,
) -> float:
    """Compute Shannon spectral entropy `S = -sum p_i ln p_i` (nats) of an MS/MS intensity vector."""
    dummy_mzs = [float(i + 1) for i in range(len(intensities))]
    _, _, ent = _prepare_ms_probability_peaks(
        dummy_mzs,
        intensities,
        weighted=weighted,
        clean_noise_rel=clean_noise_rel,
    )
    return ent


def _entropy_similarity_from_prepared(
    mzs_a: Sequence[float],
    probs_a: Sequence[float],
    ent_a: float,
    mzs_b: Sequence[float],
    probs_b: Sequence[float],
    ent_b: float,
    mz_tolerance_da: float,
) -> float:
    """Compute `1 - (2 * S(0.5*(P_A + P_B)) - S(P_A) - S(P_B)) / ln(4)` with 1-to-1 peak matching."""
    if not probs_a or not probs_b:
        return 0.0

    # Greedy 1-to-1 matching prioritizing largest combined fragment probability within tolerance
    candidates: list[tuple[float, float, int, int]] = []
    for i, mz_a in enumerate(mzs_a):
        lo = bisect.bisect_left(mzs_b, mz_a - mz_tolerance_da)
        hi = bisect.bisect_right(mzs_b, mz_a + mz_tolerance_da)
        for j in range(lo, hi):
            candidates.append((probs_a[i] + probs_b[j], -abs(mz_a - mzs_b[j]), i, j))
    candidates.sort(reverse=True)

    used_a: set[int] = set()
    used_b: set[int] = set()
    mix_probs: list[float] = []
    for _, _, i, j in candidates:
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
            mix_probs.append(0.5 * (probs_a[i] + probs_b[j]))

    for i, pa in enumerate(probs_a):
        if i not in used_a and pa > 0.0:
            mix_probs.append(0.5 * pa)
    for j, pb in enumerate(probs_b):
        if j not in used_b and pb > 0.0:
            mix_probs.append(0.5 * pb)

    ent_ab = -sum(pm * math.log(pm) for pm in mix_probs if pm > 0.0)
    js_divergence_2 = max(0.0, 2.0 * ent_ab - ent_a - ent_b)
    sim = 1.0 - (js_divergence_2 / _LN4)
    return max(0.0, min(1.0, sim))


def msms_entropy_similarity(
    spec_a: MassSpecRecord,
    spec_b: MassSpecRecord,
    *,
    mz_tolerance_da: float = 0.02,
    weighted: bool = True,
    clean_noise_rel: float = 0.01,
) -> float:
    """Compute unweighted or weighted MS/MS spectral entropy similarity (Li et al., *Nat. Methods* 2021)."""
    tol_da = _require_finite_float(mz_tolerance_da, "mz_tolerance_da")
    if tol_da <= 0.0:
        raise ValueError("mz_tolerance_da must be positive")
    noise_rel = _require_finite_float(clean_noise_rel, "clean_noise_rel")
    if noise_rel < 0.0:
        raise ValueError("clean_noise_rel must be non-negative")

    mzs_a, probs_a, ent_a = _prepare_ms_probability_peaks(
        spec_a.mzs,
        spec_a.intensities,
        weighted=weighted,
        clean_noise_rel=noise_rel,
        merge_within_da=tol_da * 0.5,
    )
    mzs_b, probs_b, ent_b = _prepare_ms_probability_peaks(
        spec_b.mzs,
        spec_b.intensities,
        weighted=weighted,
        clean_noise_rel=noise_rel,
        merge_within_da=tol_da * 0.5,
    )
    return _entropy_similarity_from_prepared(
        mzs_a, probs_a, ent_a, mzs_b, probs_b, ent_b, tol_da
    )


def _merge_two_reference_spectra(
    spec_target: MassSpecRecord,
    spec_interferent: MassSpecRecord,
    alpha_interferent: float,
    mz_tolerance_da: float,
) -> tuple[list[float], list[float]]:
    """Construct convex mixture `(1 - alpha) * P_target + alpha * P_interferent` in probability space."""
    mzs_t, probs_t, _ = _prepare_ms_probability_peaks(
        spec_target.mzs, spec_target.intensities, weighted=False, clean_noise_rel=0.005
    )
    mzs_i, probs_i, _ = _prepare_ms_probability_peaks(
        spec_interferent.mzs, spec_interferent.intensities, weighted=False, clean_noise_rel=0.005
    )
    w_t = max(0.0, min(1.0, 1.0 - alpha_interferent))
    w_i = max(0.0, min(1.0, alpha_interferent))

    combined = sorted(
        [(mz, w_t * p) for mz, p in zip(mzs_t, probs_t) if w_t * p > 0.0]
        + [(mz, w_i * p) for mz, p in zip(mzs_i, probs_i) if w_i * p > 0.0]
    )
    if not combined:
        return [], []
    merged_mzs: list[float] = []
    merged_int: list[float] = []
    cur_mz, cur_p = combined[0]
    for mz, p in combined[1:]:
        if (mz - cur_mz) <= mz_tolerance_da:
            tot = cur_p + p
            cur_mz = (cur_mz * cur_p + mz * p) / max(1e-15, tot)
            cur_p = tot
        else:
            merged_mzs.append(cur_mz)
            merged_int.append(cur_p)
            cur_mz, cur_p = mz, p
    merged_mzs.append(cur_mz)
    merged_int.append(cur_p)
    return merged_mzs, merged_int


@dataclass(frozen=True)
class ChimeraAnalysisResult:
    """Near-isobaric MS/MS co-elution / chimera ambiguity analysis."""

    is_chimera: bool
    target_entropy_similarity: float
    interferent_id: str | None
    interferent_entropy_similarity: float
    mixture_alpha_interferent: float
    mixture_entropy_similarity: float
    entropy_gain_over_single: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_chimera": self.is_chimera,
            "target_entropy_similarity": round(self.target_entropy_similarity, 6),
            "interferent_id": self.interferent_id,
            "interferent_entropy_similarity": round(self.interferent_entropy_similarity, 6),
            "mixture_alpha_interferent": round(self.mixture_alpha_interferent, 4),
            "mixture_entropy_similarity": round(self.mixture_entropy_similarity, 6),
            "entropy_gain_over_single": round(self.entropy_gain_over_single, 6),
        }


def detect_msms_chimera(
    observed: MassSpecRecord,
    target_reference: MassSpecRecord,
    *,
    interferent_reference: MassSpecRecord | None = None,
    candidate_pool: Sequence[MassSpecRecord] = (),
    mz_tolerance_da: float = 0.02,
    isolation_window_da: float = 1.0,
    min_interferent_alpha: float = 0.18,
    min_mixture_gain: float = 0.045,
    weighted: bool = True,
) -> ChimeraAnalysisResult:
    """Detect co-eluting near-isobaric tandem MS/MS chimeras within a Q1 precursor window.

    Evaluates whether `observed` is significantly better explained as a convex combination
    `(1 - alpha) * target + alpha * interferent` (`alpha in [min_interferent_alpha, 0.85]`)
    of two structurally distinct precursors inside `+-isolation_window_da`.
    """
    tol_da = _require_finite_float(mz_tolerance_da, "mz_tolerance_da")
    iso_da = _require_finite_float(isolation_window_da, "isolation_window_da")
    if tol_da <= 0.0 or iso_da <= 0.0:
        raise ValueError("mz_tolerance_da and isolation_window_da must be positive")

    sim_target = msms_entropy_similarity(
        observed, target_reference, mz_tolerance_da=tol_da, weighted=weighted
    )

    mzs_obs, probs_obs, ent_obs = _prepare_ms_probability_peaks(
        observed.mzs,
        observed.intensities,
        weighted=weighted,
        clean_noise_rel=0.01,
        merge_within_da=tol_da * 0.5,
    )

    pool: list[MassSpecRecord] = []
    if interferent_reference is not None:
        pool.append(interferent_reference)
    pool.extend(candidate_pool)

    target_ik14 = target_reference.inchikey[:14] if target_reference.inchikey else ""

    best_interferent: MassSpecRecord | None = None
    best_sim_inter = 0.0
    best_alpha = 0.0
    best_mix_sim = sim_target

    for cand in pool:
        if cand.identifier == target_reference.identifier:
            continue
        cand_ik14 = cand.inchikey[:14] if cand.inchikey else ""
        if target_ik14 and cand_ik14 and cand_ik14 == target_ik14:
            continue
        if observed.precursor_mz is not None and cand.precursor_mz is not None:
            if abs(observed.precursor_mz - cand.precursor_mz) > iso_da:
                continue

        sim_cand = msms_entropy_similarity(
            observed, cand, mz_tolerance_da=tol_da, weighted=weighted
        )
        if sim_cand < 0.22:
            continue

        def _neg_mix_sim(alpha_val: float) -> float:
            mix_mzs, mix_int = _merge_two_reference_spectra(
                target_reference, cand, alpha_val, tol_da
            )
            m_mzs, m_probs, m_ent = _prepare_ms_probability_peaks(
                mix_mzs, mix_int, weighted=weighted, clean_noise_rel=0.01
            )
            return -_entropy_similarity_from_prepared(
                mzs_obs, probs_obs, ent_obs, m_mzs, m_probs, m_ent, tol_da
            )

        alpha_opt, neg_best = golden_section_minimize(
            _neg_mix_sim, 0.05, 0.95, tol=1e-2, max_evals=18
        )
        mix_sim = -neg_best
        if mix_sim > best_mix_sim:
            best_mix_sim = mix_sim
            best_alpha = alpha_opt
            best_interferent = cand
            best_sim_inter = sim_cand

    best_single = max(sim_target, best_sim_inter)
    gain = max(0.0, best_mix_sim - best_single)
    dual_high_overlap = (
        best_interferent is not None
        and sim_target >= 0.45
        and best_sim_inter >= 0.50
        and min_interferent_alpha <= best_alpha <= 0.88
    )
    is_chimera = bool(
        best_interferent is not None
        and min_interferent_alpha <= best_alpha <= 0.88
        and (gain >= min_mixture_gain or dual_high_overlap)
    )

    return ChimeraAnalysisResult(
        is_chimera=is_chimera,
        target_entropy_similarity=sim_target,
        interferent_id=best_interferent.identifier if best_interferent is not None else None,
        interferent_entropy_similarity=best_sim_inter,
        mixture_alpha_interferent=best_alpha if best_interferent is not None else 0.0,
        mixture_entropy_similarity=best_mix_sim,
        entropy_gain_over_single=gain,
    )
