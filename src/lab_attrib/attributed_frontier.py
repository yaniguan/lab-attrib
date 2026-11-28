"""Closed-Loop Attributed `Chem-RRT` Frontier Campaign on Delaney ESOL & AqSolDB.

Benchmarks five experimental feedback regimes across multi-round molecular discovery:
1. `naive_raw_writeback`:
   Jumps directly to deep OOD candidates without pre-execution containment or
   post-execution attribution; writes all robotic zero-signal artifacts and
   precursor ghost-peak false positives directly into surrogate memory.
2. `pre_gate_only_static`:
   Enforces a static pre-execution trust radius `d(x) <= tau_0` without stepping-stone
   expansion and without post-execution causal attribution.
3. `unattributed_frontier_probes`:
   Uses annular `Chem-RRT` micro-probes (`tau_t < d(x) <= (1 + eta) * tau_t`, `<= 10 uL`)
   to expand `tau_{t+1}`, but omits post-execution attribution—allowing corrupted and
   spectrally ambiguous probes to expand `tau` and poison surrogate memory.
4. `lab_attrib_full_closed_loop`:
   Combines pre-execution annular micro-containment and local `k`-NN `Chem-RRT` support
   gating with synchronized end-of-round frontier commits, unclamped post-execution NNLS
   spectral deconvolution (`xrd_deconvolve`), 4-way causal blame (`blame`), EVOI orthogonal
   assay routing (`orthogonal_router`), SHA-256 `CertifiedLedger` writeback (`ledger`),
   stratified holdout calibration + Gibbs-Candes online ACI (`alpha_t`), and
   `check_attributed_frontier_update`.
5. `lab_attrib_equal_budget`:
   Identical post-execution verification, retry, EVOI routing, and `CertifiedLedger`
   architecture as `lab_attrib_full_closed_loop`, but deducts EVERY hardware retry and
   secondary orthogonal assay directly from the fixed `n_rounds * batch_size` total
   measurement budget (`180` total assays on the default `9 x 20` campaign).
"""

from __future__ import annotations

import csv
import hashlib
import heapq
import math
import os
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

from lab_attrib.blame import (
    AssayTelemetry,
    AttributedSteppingStone,
    AttributionOutcome,
    OrthogonalAssayReadout,
    OrthogonalModality,
    SpectralFitMethod,
    admit_to_scientific_memory,
    attribute_assay_outcome,
    check_attributed_frontier_update,
    diagnose_assay,
    effective_attribution,
    in_annular_shell,
)
from lab_attrib.ledger import CertifiedLedger
from lab_attrib.orthogonal_router import route_orthogonal_assay
from lab_attrib.xrd_deconvolve import build_gaussian_spectrum, deconvolve_spectrum

DEFAULT_CACHED_ESOL_CSV: Path = (
    Path(__file__).resolve().parents[2] / "benchmarks" / "data" / "delaney-processed.csv"
)

REGIME_NAMES: tuple[str, ...] = (
    "naive_raw_writeback",
    "pre_gate_only_static",
    "unattributed_frontier_probes",
    "lab_attrib_full_closed_loop",
    "lab_attrib_equal_budget",
)

HIT_MIN_RINGS: int = 3
HIT_MIN_LOG_SOLUBILITY: float = -3.50
MIN_FRONTIER_CALIBRATION_SAMPLES: int = 15


@dataclass(frozen=True)
class MoleculeCandidate:
    """Physicochemical descriptor record for an ESOL or synthetic molecule."""

    compound_id: str
    smiles: str
    features: tuple[float, float, float, float, float, float]
    rings: int
    true_log_s: float

    @property
    def is_true_frontier_hit(self) -> bool:
        return self.rings >= HIT_MIN_RINGS and self.true_log_s >= HIT_MIN_LOG_SOLUBILITY

    def feature_dict(self) -> dict[str, float]:
        names = (
            "min_degree",
            "mol_weight",
            "hbond_donors",
            "rings",
            "rotatable_bonds",
            "polar_surface_area",
        )
        return {k: float(v) for k, v in zip(names, self.features)}


def load_delaney_esol_csv(csv_path: Path) -> list[MoleculeCandidate]:
    """Load Delaney ESOL or AqSolDB solubility molecules from CSV/TSV (single-pass read-only)."""
    molecules: list[MoleculeCandidate] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        first_line = handle.readline()
        if not first_line:
            return molecules
        delimiter = "\t" if "\t" in first_line else ","
        handle.seek(0)
        reader = csv.DictReader(handle, delimiter=delimiter)
        for idx, row in enumerate(reader):
            cid = (
                row.get("Compound ID")
                or row.get("ID")
                or row.get("Name")
                or f"ESOL-{idx:04d}"
            ).strip()
            smiles = (row.get("smiles") or row.get("SMILES") or "").strip()
            raw_mw = row.get("Molecular Weight") or row.get("MolWt")
            raw_hbd = row.get("Number of H-Bond Donors") or row.get("NumHDonors")
            raw_rings = row.get("Number of Rings") or row.get("RingCount")
            raw_rot = row.get("Number of Rotatable Bonds") or row.get("NumRotatableBonds")
            raw_psa = row.get("Polar Surface Area") or row.get("TPSA")
            raw_logs = (
                row.get("measured log solubility in mols per litre")
                or row.get("Solubility")
                or row.get("Y")
            )
            if any(v in (None, "") for v in (raw_mw, raw_hbd, raw_rings, raw_rot, raw_psa, raw_logs)):
                continue
            mw = float(str(raw_mw))
            hbd = float(str(raw_hbd))
            rings = int(round(float(str(raw_rings))))
            rot = float(str(raw_rot))
            psa = float(str(raw_psa))
            log_s = float(str(raw_logs))
            raw_min_deg = row.get("Minimum Degree")
            if raw_min_deg not in (None, ""):
                min_deg = float(str(raw_min_deg))
            else:
                min_deg = 1.0 if (rings == 0 or hbd >= 1.0 or rot >= 1.0) else 2.0
            molecules.append(
                MoleculeCandidate(
                    compound_id=f"{cid}#{idx}",
                    smiles=smiles,
                    features=(min_deg, mw, hbd, float(rings), rot, psa),
                    rings=rings,
                    true_log_s=log_s,
                )
            )
    return molecules


def build_synthetic_esol_pool(n_molecules: int = 720, seed: int = 0) -> list[MoleculeCandidate]:
    """Deterministic physicochemical molecular benchmark when external ESOL CSV is absent."""
    rng = random.Random(seed)
    pool: list[MoleculeCandidate] = []
    ring_weights = [0.26, 0.34, 0.20, 0.12, 0.06, 0.02]
    for idx in range(n_molecules):
        u = rng.random()
        cum = 0.0
        rings = 0
        for r_val, w in enumerate(ring_weights):
            cum += w
            if u <= cum:
                rings = r_val
                break
        min_deg = float(1 if rng.random() < 0.72 else 2)
        hbd = float(rng.choices([0, 1, 2, 3, 4], weights=[0.38, 0.30, 0.18, 0.10, 0.04])[0])
        rot = float(rng.randint(0, 7))
        psa = max(0.0, round(hbd * 21.5 + rng.gauss(26.0, 22.0), 2))
        mw = max(
            65.0,
            round(82.0 + rings * 58.0 + rot * 14.5 + psa * 0.65 + rng.gauss(0.0, 24.0), 2),
        )
        # Realistic ESOL-like solubility relationship (~24% hit rate on multi-ring scaffolds)
        true_log_s = (
            0.15
            - 0.0105 * mw
            - 0.66 * rings
            - 0.06 * rot
            + 0.0205 * psa
            + 0.26 * hbd
            + rng.gauss(0.0, 0.62)
        )
        pool.append(
            MoleculeCandidate(
                compound_id=f"SYN-MOL-{idx:04d}",
                smiles=f"C1=CC=CC=C1_R{rings}_{idx}",
                features=(min_deg, mw, hbd, float(rings), rot, psa),
                rings=rings,
                true_log_s=round(true_log_s, 4),
            )
        )
    return pool


def resolve_molecule_pool(
    csv_path: str | Path | None = None,
    *,
    force_synthetic_fallback: bool = False,
) -> tuple[list[MoleculeCandidate], str]:
    """Load Delaney ESOL in a single pass if available, else return deterministic synthetic pool."""
    if not force_synthetic_fallback:
        candidates_to_check: list[Path] = []
        if csv_path is not None:
            candidates_to_check.append(Path(csv_path))
        env_csv = os.environ.get("LAB_ATTRIB_ESOL_CSV")
        if env_csv:
            candidates_to_check.append(Path(env_csv))
        if csv_path is None and not env_csv and DEFAULT_CACHED_ESOL_CSV.is_file():
            candidates_to_check.append(DEFAULT_CACHED_ESOL_CSV)

        for candidate in candidates_to_check:
            if candidate.is_file():
                loaded = load_delaney_esol_csv(candidate)
                return loaded, f"delaney_esol_csv ({len(loaded)} molecules)"
    syn = build_synthetic_esol_pool()
    return syn, f"synthetic_esol_fallback ({len(syn)} molecules)"


def _standardize_fit(
    xs: Sequence[Sequence[float]],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    n = len(xs)
    d = len(xs[0])
    means = [sum(row[j] for row in xs) / n for j in range(d)]
    stds: list[float] = []
    for j in range(d):
        var = sum((row[j] - means[j]) ** 2 for row in xs) / max(1, n - 1)
        stds.append(max(1e-6, math.sqrt(var)))
    return tuple(means), tuple(stds)


def _standardize_apply(
    x: Sequence[float],
    means: Sequence[float],
    stds: Sequence[float],
) -> tuple[float, ...]:
    return tuple((x[j] - means[j]) / stds[j] for j in range(len(x)))


def _nearest_seed_distance(
    z: Sequence[float],
    seed_cloud: Sequence[Sequence[float]],
) -> float:
    best_sq = float("inf")
    for ref in seed_cloud:
        sq = sum((z[j] - ref[j]) ** 2 for j in range(len(z)))
        if sq < best_sq:
            best_sq = sq
    return math.sqrt(best_sq)


def knn_support_distance(
    z: Sequence[float],
    support_cloud: Sequence[Sequence[float]],
    *,
    k: int = 3,
) -> float:
    """Compute RMS Euclidean distance from `z` to its `k` nearest certified support neighbors."""
    if not support_cloud:
        return float("inf")
    k_eff = max(1, min(k, len(support_cloud)))
    sq_dists = [sum((z[j] - ref[j]) ** 2 for j in range(len(z))) for ref in support_cloud]
    smallest_sq = heapq.nsmallest(k_eff, sq_dists)
    return math.sqrt(sum(smallest_sq) / float(k_eff))


def is_locally_supported_in_tree(
    z: Sequence[float],
    certified_support: Sequence[Sequence[float]],
    *,
    local_tau: float,
    k_neighbors: int = 3,
) -> bool:
    """Directional `Chem-RRT` local density check preventing single-ray isotropic `tau` jumps.

    Requires `z` to lie within `local_tau` of its nearest certified node AND within
    `1.35 * local_tau` RMS distance of at least `k_neighbors` certified support points.
    """
    if not certified_support or local_tau <= 0.0:
        return False
    nearest = _nearest_seed_distance(z, certified_support)
    if nearest > local_tau:
        return False
    knn_rms = knn_support_distance(z, certified_support, k=k_neighbors)
    return knn_rms <= 1.35 * local_tau


def _basis_expand(z: Sequence[float]) -> list[float]:
    """13-term quadratic feature basis `[1, z_1..z_6, z_1^2..z_6^2]`."""
    out = [1.0]
    out.extend(z)
    out.extend(v * v for v in z)
    return out


def _fit_ridge_linear_system(
    z_rows: Sequence[Sequence[float]],
    y_vals: Sequence[float],
    l2_reg: float = 1.5,
) -> list[float]:
    """Pure-Python Cholesky/Gaussian-elimination ridge regression solver."""
    p = 13
    ata = [[0.0] * p for _ in range(p)]
    aty = [0.0] * p
    for z, y in zip(z_rows, y_vals):
        phi = _basis_expand(z)
        for i in range(p):
            pi = phi[i]
            aty[i] += pi * y
            row_i = ata[i]
            for j in range(i, p):
                row_i[j] += pi * phi[j]
    for i in range(p):
        for j in range(i):
            ata[i][j] = ata[j][i]
        ata[i][i] += l2_reg if i > 0 else 1e-6

    # Gaussian elimination with partial pivoting
    aug = [ata[i][:] + [aty[i]] for i in range(p)]
    for col in range(p):
        pivot = max(range(col, p), key=lambda r: abs(aug[r][col]))
        aug[col], aug[pivot] = aug[pivot], aug[col]
        denom = aug[col][col]
        if abs(denom) < 1e-12:
            denom = 1e-12
        for j in range(col, p + 1):
            aug[col][j] /= denom
        for r in range(p):
            if r != col:
                factor = aug[r][col]
                if factor != 0.0:
                    for j in range(col, p + 1):
                        aug[r][j] -= factor * aug[col][j]
    return [aug[i][p] for i in range(p)]


def _predict_ridge(weights: Sequence[float], z: Sequence[float]) -> float:
    phi = _basis_expand(z)
    return sum(w * f for w, f in zip(weights, phi))


def _conformal_quantile(scores: Sequence[float], alpha: float = 0.10) -> float:
    """Finite-sample split-conformal quantile at level `ceil((n + 1) * (1 - alpha)) / n`."""
    if not scores:
        return 1.5
    eff_alpha = min(0.99, max(1e-6, alpha))
    sorted_s = sorted(scores)
    n = len(sorted_s)
    rank = min(n, max(1, math.ceil((n + 1) * (1.0 - eff_alpha))))
    return sorted_s[rank - 1]


# Precomputed spectral templates on a 28-point diffraction / optical grid for fast NNLS calls
_SPECTRAL_GRID: tuple[float, ...] = tuple(10.0 + 2.0 * i for i in range(28))
_PREC_CLEAN_SPECTRUM: tuple[float, ...] = tuple(
    build_gaussian_spectrum(_SPECTRAL_GRID, [(16.0, 1.0, 2.2), (56.0, 0.8, 2.2)])
)
_PREC_GHOST_SPECTRUM: tuple[float, ...] = tuple(
    build_gaussian_spectrum(
        _SPECTRAL_GRID, [(32.0, 0.95, 2.5), (42.0, 0.85, 2.5), (16.0, 0.35, 2.2)]
    )
)
_TARGET_SPECTRUM: tuple[float, ...] = tuple(
    build_gaussian_spectrum(_SPECTRAL_GRID, [(32.0, 1.0, 2.4), (42.0, 0.90, 2.4)])
)


@dataclass(frozen=True)
class SimulatedWetLabRun:
    """Simulated wet-lab readout + spectral NNLS telemetry for a single candidate execution."""

    raw_measured_log_s: float
    telemetry: AssayTelemetry
    true_is_corrupted: bool
    true_has_ghost_peak: bool


def _deterministic_rng(*parts: object) -> random.Random:
    """Cross-platform, PYTHONHASHSEED-independent SHA-256 deterministic PRNG."""
    joined = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(joined).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _simulate_wet_lab_execution(
    mol: MoleculeCandidate,
    *,
    seed: int,
    attempt: int,
    fault_flag_recall: float = 1.0,
    false_alarm_rate: float = 0.0,
) -> SimulatedWetLabRun:
    """Deterministic per-(seed, compound_id, attempt) wet-lab + unclamped NNLS spectral simulator.

    - `fault_flag_recall` in `[0, 1]` controls sensor/library recall on hardware execution faults
      and interfering ghost-precursor reference coverage. At `< 1.0`, missed hardware faults
      report clean execution QC flags, and uncatalogued ghost precursors are omitted from the
      NNLS precursor reference basis so `deconvolve_spectrum` genuinely mistakes overlapping
      precursor peaks for the target phase.
    - `false_alarm_rate` in `[0, 1]` injects spurious execution sensor alarms on physically clean
      runs, triggering extra hardware retries.
    """
    if not math.isfinite(fault_flag_recall) or not (0.0 <= fault_flag_recall <= 1.0):
        raise ValueError(f"fault_flag_recall must be in [0, 1], got {fault_flag_recall}")
    if not math.isfinite(false_alarm_rate) or not (0.0 <= false_alarm_rate <= 1.0):
        raise ValueError(f"false_alarm_rate must be in [0, 1], got {false_alarm_rate}")

    rng = _deterministic_rng("WETLAB", seed, mol.compound_id, attempt)
    sensor_rng = _deterministic_rng("SENSOR_MODEL", seed, mol.compound_id, attempt)

    primary_sigma = 0.13
    # Hardware fault rate: 22% on initial robotic execution, 3% on targeted hardware retry
    fault_prob = 0.22 if attempt == 0 else 0.03
    u_fault = rng.random()
    if u_fault < fault_prob:
        # Robotic cavitation / missed incubation / broken seal -> apparent insoluble precipitate
        corrupted_log_s = round(-8.65 + rng.gauss(0.0, 0.20), 4)
        fault_kind = rng.randint(0, 2)
        fault_detected = sensor_rng.random() < fault_flag_recall
        obs_fault = [
            max(0.0, 0.35 * _PREC_CLEAN_SPECTRUM[i] + sensor_rng.gauss(0.0, 0.010))
            for i in range(len(_SPECTRAL_GRID))
        ]
        deconv_fault = deconvolve_spectrum(
            obs_fault,
            [_PREC_CLEAN_SPECTRUM, _PREC_GHOST_SPECTRUM],
            _TARGET_SPECTRUM,
            min_hit_amplitude=0.25,
            noise_sigma=0.04,
            use_numpy=False,
        )
        telemetry = AssayTelemetry(
            dispense_error_permille=(165 if fault_kind == 0 else 18) if fault_detected else 18,
            aspiration_clean=(fault_kind != 0) if fault_detected else True,
            incubation_window_clean=(fault_kind != 1) if fault_detected else True,
            seal_intact=(fault_kind != 2) if fault_detected else True,
            positive_control_passed=True,
            blank_uncontaminated=True,
            detector_calibrated=True,
            precursor_overlap_permille=deconv_fault.precursor_overlap_permille,
            spectral_residual_permille=deconv_fault.spectral_residual_permille,
            snr_tenths=deconv_fault.snr_tenths,
            target_signal_present=False,
            fit_method=SpectralFitMethod.STICK_NNLS,
        )
        return SimulatedWetLabRun(
            raw_measured_log_s=corrupted_log_s,
            telemetry=telemetry,
            true_is_corrupted=True,
            true_has_ghost_peak=False,
        )

    # Physically clean execution; check for unreacted precursor ghost-peak interference (18%)
    u_ghost = rng.random()
    true_amp = max(0.0, min(2.0, (mol.true_log_s - (-4.5)) * 0.35))
    if u_ghost < 0.18:
        ghost_c = 0.95 + 0.25 * rng.random()
        obs_spec = [
            max(
                0.0,
                ghost_c * _PREC_GHOST_SPECTRUM[i]
                + 0.12 * true_amp * _TARGET_SPECTRUM[i]
                + rng.gauss(0.0, 0.012),
            )
            for i in range(len(_SPECTRAL_GRID))
        ]
        ghost_catalogued = sensor_rng.random() < fault_flag_recall
        spurious_alarm = sensor_rng.random() < false_alarm_rate
        prec_basis = (
            [_PREC_CLEAN_SPECTRUM, _PREC_GHOST_SPECTRUM]
            if ghost_catalogued
            else [_PREC_CLEAN_SPECTRUM]
        )
        deconv = deconvolve_spectrum(
            obs_spec,
            prec_basis,
            _TARGET_SPECTRUM,
            min_hit_amplitude=0.25,
            noise_sigma=0.04,
            use_numpy=False,
        )
        # Naive optical/peak readout sees high apparent target absorbance -> false soluble readout
        apparent_ghost_log_s = round(-1.85 + rng.gauss(0.0, 0.18), 4)
        telemetry = AssayTelemetry(
            dispense_error_permille=14,
            aspiration_clean=not spurious_alarm,
            incubation_window_clean=True,
            seal_intact=True,
            positive_control_passed=True,
            blank_uncontaminated=True,
            detector_calibrated=True,
            precursor_overlap_permille=deconv.precursor_overlap_permille,
            spectral_residual_permille=deconv.spectral_residual_permille,
            snr_tenths=deconv.snr_tenths,
            target_signal_present=(
                deconv.target_signal_present or apparent_ghost_log_s >= HIT_MIN_LOG_SOLUBILITY
            ),
            fit_method=SpectralFitMethod.STICK_NNLS,
        )
        return SimulatedWetLabRun(
            raw_measured_log_s=apparent_ghost_log_s,
            telemetry=telemetry,
            true_is_corrupted=False,
            true_has_ghost_peak=True,
        )

    # Clean spectrum without overlapping ghost precursor (zero clamps!)
    clean_log_s = round(mol.true_log_s + rng.gauss(0.0, primary_sigma), 4)
    clean_prec_c = 0.35
    target_c = max(0.0, (clean_log_s - (-4.2)) * 0.45)
    obs_spec = [
        max(
            0.0,
            clean_prec_c * _PREC_CLEAN_SPECTRUM[i]
            + target_c * _TARGET_SPECTRUM[i]
            + rng.gauss(0.0, 0.010),
        )
        for i in range(len(_SPECTRAL_GRID))
    ]
    spurious_alarm = sensor_rng.random() < false_alarm_rate
    deconv = deconvolve_spectrum(
        obs_spec,
        [_PREC_CLEAN_SPECTRUM, _PREC_GHOST_SPECTRUM],
        _TARGET_SPECTRUM,
        min_hit_amplitude=0.25,
        noise_sigma=0.04,
        use_numpy=False,
    )
    if mol.rings >= HIT_MIN_RINGS:
        decision_snr = abs(clean_log_s - HIT_MIN_LOG_SOLUBILITY) / primary_sigma
        snr_tenths = int(round(10.0 * decision_snr))
    else:
        snr_tenths = deconv.snr_tenths

    telemetry = AssayTelemetry(
        dispense_error_permille=12,
        aspiration_clean=not spurious_alarm,
        incubation_window_clean=True,
        seal_intact=True,
        positive_control_passed=True,
        blank_uncontaminated=True,
        detector_calibrated=True,
        precursor_overlap_permille=deconv.precursor_overlap_permille,
        spectral_residual_permille=deconv.spectral_residual_permille,
        snr_tenths=snr_tenths,
        target_signal_present=(
            mol.rings >= HIT_MIN_RINGS and clean_log_s >= HIT_MIN_LOG_SOLUBILITY
        ),
        fit_method=SpectralFitMethod.STICK_NNLS,
    )
    return SimulatedWetLabRun(
        raw_measured_log_s=clean_log_s,
        telemetry=telemetry,
        true_is_corrupted=False,
        true_has_ghost_peak=False,
    )


def _simulate_orthogonal_lcms(
    mol: MoleculeCandidate,
    modality: OrthogonalModality,
    *,
    seed: int,
) -> tuple[float, OrthogonalAssayReadout]:
    """High-selectivity secondary orthogonal measurement (`sigma = 0.012`)."""
    rng = _deterministic_rng("ORTH", seed, modality.value, mol.compound_id)
    orth_sigma = 0.012
    orth_log_s = round(mol.true_log_s + rng.gauss(0.0, orth_sigma), 4)
    decision_snr = (
        abs(orth_log_s - HIT_MIN_LOG_SOLUBILITY) / orth_sigma
        if mol.rings >= HIT_MIN_RINGS
        else 7.5
    )
    readout = OrthogonalAssayReadout(
        modality=modality,
        instrument_calibrated=True,
        orthogonal_snr_tenths=int(round(10.0 * decision_snr)),
        selectivity_clean=True,
        target_confirmed=(mol.rings >= HIT_MIN_RINGS and orth_log_s >= HIT_MIN_LOG_SOLUBILITY),
    )
    return orth_log_s, readout


def _run_single_seed_regime(
    regime: str,
    seed_train: Sequence[MoleculeCandidate],
    seed_cal: Sequence[MoleculeCandidate],
    exploration_pool: Sequence[MoleculeCandidate],
    frontier_eval_set: Sequence[MoleculeCandidate],
    *,
    seed: int,
    n_rounds: int,
    batch_size: int,
    eta_permille: int = 350,
    alpha: float = 0.10,
    fault_flag_recall: float = 1.0,
    false_alarm_rate: float = 0.0,
) -> dict[str, Any]:
    """Execute one closed-loop campaign seed for a single experimental regime."""
    if regime not in REGIME_NAMES:
        raise ValueError(f"Unknown regime {regime!r}; expected one of {REGIME_NAMES}")

    is_gated_regime = regime in ("lab_attrib_full_closed_loop", "lab_attrib_equal_budget")
    is_equal_budget = regime == "lab_attrib_equal_budget"
    max_total_measurements = n_rounds * batch_size

    means, stds = _standardize_fit([m.features for m in seed_train])
    z_seed_train = [_standardize_apply(m.features, means, stds) for m in seed_train]
    z_seed_cal = [_standardize_apply(m.features, means, stds) for m in seed_cal]

    cal_dists = [_nearest_seed_distance(z, z_seed_train) for z in z_seed_cal]
    sorted_cal_d = sorted(cal_dists)
    tau_0 = sorted_cal_d[min(len(sorted_cal_d) - 1, int(0.94 * len(sorted_cal_d)))]
    tau_permille = max(150, int(round(1000.0 * tau_0)))
    initial_tau_permille = tau_permille

    cal_knn_dists = [knn_support_distance(z, z_seed_train, k=3) for z in z_seed_cal]
    sorted_knn_d = sorted(cal_knn_dists)
    local_tau_0 = sorted_knn_d[min(len(sorted_knn_d) - 1, int(0.94 * len(sorted_knn_d)))]

    # Pre-deployment split-conformal quantile calibrated on clean seed domain
    init_weights = _fit_ridge_linear_system(z_seed_train, [m.true_log_s for m in seed_train])
    init_cal_scores = [
        abs(m_c.true_log_s - _predict_ridge(init_weights, z_c))
        for m_c, z_c in zip(seed_cal, z_seed_cal)
    ]
    static_q_hat = _conformal_quantile(init_cal_scores, alpha=alpha)

    cand_z = {m.compound_id: _standardize_apply(m.features, means, stds) for m in exploration_pool}
    cand_dist = {
        m.compound_id: _nearest_seed_distance(cand_z[m.compound_id], z_seed_train)
        for m in exploration_pool
    }
    cand_dist_permille = {
        cid: max(1, int(round(1000.0 * d))) for cid, d in cand_dist.items()
    }

    eval_z = [_standardize_apply(m.features, means, stds) for m in frontier_eval_set]
    eval_dist = [_nearest_seed_distance(z, z_seed_train) for z in eval_z]

    # Precompute top-3 nearest squared distances to z_seed_train for O(1) per-round kNN updates
    cand_top3_sq: dict[str, list[float]] = {}
    if is_gated_regime:
        for cid, zv in cand_z.items():
            sq_list = [sum((zv[j] - ref[j]) ** 2 for j in range(len(zv))) for ref in z_seed_train]
            cand_top3_sq[cid] = sorted(heapq.nsmallest(min(3, len(sq_list)), sq_list))

    def _cand_locally_supported(cid: str, local_tau: float) -> bool:
        top3 = cand_top3_sq[cid]
        if not top3 or local_tau <= 0.0:
            return False
        if math.sqrt(top3[0]) > local_tau:
            return False
        rms = math.sqrt(sum(top3) / float(len(top3)))
        return rms <= 1.35 * local_tau

    def _cand_nearest_support(cid: str) -> float:
        top3 = cand_top3_sq[cid]
        return math.sqrt(top3[0]) if top3 else float("inf")

    # Surrogate memory (training features + measured targets), local certified Chem-RRT support,
    # and strictly held-out exchangeable calibration split + Gibbs-Candes online ACI state.
    mem_z: list[tuple[float, ...]] = list(z_seed_train)
    mem_y: list[float] = [m.true_log_s for m in seed_train]
    certified_support: list[tuple[float, ...]] = list(z_seed_train)

    frontier_cal_z: list[tuple[float, ...]] = []
    frontier_cal_y: list[float] = []
    alpha_t = alpha
    gamma_aci = 0.03

    ledger = CertifiedLedger()
    remaining = {m.compound_id: m for m in exploration_pool}

    committed_hits: list[MoleculeCandidate] = []
    poisoned_memory_entries = 0
    primary_assays_executed = 0
    robot_retries_triggered = 0
    orthogonal_assays_triggered = 0
    total_measurements_used = 0
    certified_negatives_committed = 0
    certified_positives_committed = 0
    stepping_stone_expansions = 0

    round_mae_curve: list[float] = []
    round_poison_curve: list[int] = []
    round_hits_curve: list[int] = []

    eta_scale = 1.0 + (eta_permille / 1000.0)

    for round_idx in range(n_rounds):
        budget_exhausted = is_equal_budget and (total_measurements_used >= max_total_measurements)
        if not remaining or budget_exhausted:
            # Pad remaining round trajectory slots with stationary post-budget state
            cur_w = _fit_ridge_linear_system(mem_z, mem_y)
            cur_mae = sum(
                abs(frontier_eval_set[i].true_log_s - _predict_ridge(cur_w, eval_z[i]))
                for i in range(len(frontier_eval_set))
            ) / max(1, len(frontier_eval_set))
            cur_true_hits = sum(1 for m in committed_hits if m.is_true_frontier_hit)
            while len(round_mae_curve) < n_rounds:
                round_mae_curve.append(cur_mae)
                round_poison_curve.append(poisoned_memory_entries)
                round_hits_curve.append(cur_true_hits)
            break

        weights = _fit_ridge_linear_system(mem_z, mem_y)
        round_tau_permille = tau_permille

        current_cal_scores = [
            abs(y_c - _predict_ridge(weights, z_c))
            for z_c, y_c in zip(frontier_cal_z, frontier_cal_y)
        ] or [
            abs(m_c.true_log_s - _predict_ridge(weights, z_c))
            for m_c, z_c in zip(seed_cal, z_seed_cal)
        ]
        online_q_t = _conformal_quantile(current_cal_scores, alpha=alpha_t)

        available = list(remaining.values())
        cal_holdout_ids: set[str] = set()

        if regime == "naive_raw_writeback":
            scored = sorted(
                available,
                key=lambda m: (
                    (1.2 if m.rings >= HIT_MIN_RINGS else 0.0)
                    + _predict_ridge(weights, cand_z[m.compound_id])
                    + 0.25 * cand_dist[m.compound_id]
                ),
                reverse=True,
            )
            batch = scored[:batch_size]
        elif regime == "pre_gate_only_static":
            admissible = [
                m for m in available if cand_dist_permille[m.compound_id] <= initial_tau_permille
            ]
            admissible.sort(
                key=lambda m: (
                    (1.2 if m.rings >= HIT_MIN_RINGS else 0.0)
                    + _predict_ridge(weights, cand_z[m.compound_id])
                ),
                reverse=True,
            )
            batch = admissible[:batch_size]
        elif regime == "unattributed_frontier_probes":
            shell_probes = [
                m
                for m in available
                if in_annular_shell(
                    round_tau_permille, eta_permille, cand_dist_permille[m.compound_id]
                )
            ]
            shell_probes.sort(
                key=lambda m: (
                    cand_dist_permille[m.compound_id],
                    _predict_ridge(weights, cand_z[m.compound_id]),
                ),
                reverse=True,
            )
            in_domain = [
                m for m in available if cand_dist_permille[m.compound_id] <= round_tau_permille
            ]
            in_domain.sort(
                key=lambda m: (
                    (1.5 if m.rings >= HIT_MIN_RINGS else 0.0)
                    + _predict_ridge(weights, cand_z[m.compound_id])
                ),
                reverse=True,
            )
            n_probes = min(len(shell_probes), max(2, batch_size // 5))
            chosen_ids = {m.compound_id for m in shell_probes[:n_probes]}
            batch = list(shell_probes[:n_probes])
            for m in in_domain:
                if len(batch) >= batch_size:
                    break
                if m.compound_id not in chosen_ids:
                    batch.append(m)
                    chosen_ids.add(m.compound_id)
        else:
            # Gated regimes (`lab_attrib_full_closed_loop` & `lab_attrib_equal_budget`):
            # 1. Exploitation requires BOTH global shell bound AND local Chem-RRT kNN support!
            in_domain = [
                m
                for m in available
                if cand_dist_permille[m.compound_id] <= round_tau_permille
                and _cand_locally_supported(m.compound_id, local_tau_0 * 1.15)
            ]
            # 2. Annular micro-probes step outward from certified support within `(1 + eta)` shell
            max_shell_permille = int((1000 + eta_permille) * round_tau_permille // 1000)
            shell_probes = [
                m
                for m in available
                if in_annular_shell(
                    round_tau_permille, eta_permille, cand_dist_permille[m.compound_id]
                )
                or (
                    cand_dist_permille[m.compound_id] <= max_shell_permille
                    and not _cand_locally_supported(m.compound_id, local_tau_0 * 1.15)
                    and _cand_nearest_support(m.compound_id) <= local_tau_0 * eta_scale
                )
            ]

            # 3. Stratified uniform random calibration holdout BEFORE greedy UCB sorting
            cal_rng = _deterministic_rng("CAL_SPLIT", seed, round_idx)
            shell_copy = list(shell_probes)
            cal_rng.shuffle(shell_copy)
            n_cal_probes = min(len(shell_copy), max(2, batch_size // 6))
            cal_probes = shell_copy[:n_cal_probes]
            cal_holdout_ids = {m.compound_id for m in cal_probes}

            remaining_shell = [m for m in shell_probes if m.compound_id not in cal_holdout_ids]
            remaining_shell.sort(
                key=lambda m: (
                    cand_dist_permille[m.compound_id],
                    _predict_ridge(weights, cand_z[m.compound_id]),
                ),
                reverse=True,
            )
            in_domain.sort(
                key=lambda m: (
                    (1.5 if m.rings >= HIT_MIN_RINGS else 0.0)
                    + _predict_ridge(weights, cand_z[m.compound_id])
                ),
                reverse=True,
            )

            n_exp_probes = min(len(remaining_shell), max(2, batch_size // 5))
            batch = list(cal_probes) + list(remaining_shell[:n_exp_probes])
            chosen_ids = {m.compound_id for m in batch}
            for m in in_domain:
                if len(batch) >= batch_size:
                    break
                if m.compound_id not in chosen_ids:
                    batch.append(m)
                    chosen_ids.add(m.compound_id)
            for m in remaining_shell[n_exp_probes:]:
                if len(batch) >= batch_size:
                    break
                if m.compound_id not in chosen_ids:
                    batch.append(m)
                    chosen_ids.add(m.compound_id)

        if not batch:
            cur_w = _fit_ridge_linear_system(mem_z, mem_y)
            cur_mae = sum(
                abs(frontier_eval_set[i].true_log_s - _predict_ridge(cur_w, eval_z[i]))
                for i in range(len(frontier_eval_set))
            ) / max(1, len(frontier_eval_set))
            cur_true_hits = sum(1 for m in committed_hits if m.is_true_frontier_hit)
            while len(round_mae_curve) < n_rounds:
                round_mae_curve.append(cur_mae)
                round_poison_curve.append(poisoned_memory_entries)
                round_hits_curve.append(cur_true_hits)
            break

        new_round_support: list[tuple[float, ...]] = []
        approved_round_taus: list[int] = []

        for mol in batch:
            if is_equal_budget and total_measurements_used >= max_total_measurements:
                break
            remaining.pop(mol.compound_id, None)
            d_perm = cand_dist_permille[mol.compound_id]
            z_vec = cand_z[mol.compound_id]

            primary_assays_executed += 1
            total_measurements_used += 1
            run = _simulate_wet_lab_execution(
                mol,
                seed=seed,
                attempt=0,
                fault_flag_recall=fault_flag_recall,
                false_alarm_rate=false_alarm_rate,
            )

            if not is_gated_regime:
                mem_z.append(z_vec)
                mem_y.append(run.raw_measured_log_s)
                if run.true_is_corrupted or run.true_has_ghost_peak:
                    poisoned_memory_entries += 1
                if (
                    mol.rings >= HIT_MIN_RINGS
                    and run.raw_measured_log_s >= HIT_MIN_LOG_SOLUBILITY
                ):
                    committed_hits.append(mol)

                if regime == "unattributed_frontier_probes":
                    if (
                        in_annular_shell(tau_permille, eta_permille, d_perm)
                        and d_perm > tau_permille
                    ):
                        tau_permille = d_perm
                        stepping_stone_expansions += 1
                continue

            # Gated regimes (`lab_attrib_full_closed_loop` & `lab_attrib_equal_budget`)
            diag_report = diagnose_assay(run.telemetry)
            primary_outcome = diag_report.outcome
            orth_readout: OrthogonalAssayReadout | None = None
            final_measured_log_s = run.raw_measured_log_s

            if primary_outcome is AttributionOutcome.EXECUTION_ARTIFACT:
                if (not is_equal_budget) or (total_measurements_used < max_total_measurements):
                    robot_retries_triggered += 1
                    total_measurements_used += 1
                    run = _simulate_wet_lab_execution(
                        mol,
                        seed=seed,
                        attempt=1,
                        fault_flag_recall=fault_flag_recall,
                        false_alarm_rate=false_alarm_rate,
                    )
                    if not run.telemetry.detector_calibrated:
                        run = SimulatedWetLabRun(
                            raw_measured_log_s=run.raw_measured_log_s,
                            telemetry=replace(run.telemetry, detector_calibrated=True),
                            true_is_corrupted=run.true_is_corrupted,
                            true_has_ghost_peak=run.true_has_ghost_peak,
                        )
                    primary_outcome = attribute_assay_outcome(run.telemetry)
                    final_measured_log_s = run.raw_measured_log_s

            if primary_outcome is AttributionOutcome.AMBIGUOUS_CHARACTERIZATION:
                decision = route_orthogonal_assay(
                    run.telemetry,
                    prior_hit_prob=0.32 if mol.rings >= HIT_MIN_RINGS else 0.15,
                    available_volume_ul=8.0,
                    sample_phase="solution",
                )
                if (
                    not decision.discard_sample
                    and decision.selected_modality is not None
                    and ((not is_equal_budget) or (total_measurements_used < max_total_measurements))
                ):
                    orthogonal_assays_triggered += 1
                    total_measurements_used += 1
                    final_measured_log_s, orth_readout = _simulate_orthogonal_lcms(
                        mol,
                        decision.selected_modality,
                        seed=seed,
                    )

            final_outcome = effective_attribution(run.telemetry, orth_readout)
            if not admit_to_scientific_memory(final_outcome):
                continue

            # Ground-truth poisoned-memory accounting across ALL regimes (including gated!)
            if run.true_is_corrupted or (run.true_has_ghost_peak and orth_readout is None):
                poisoned_memory_entries += 1

            ledger.append(
                sample_id=mol.compound_id,
                candidate_features=mol.feature_dict(),
                measured_value=final_measured_log_s,
                telemetry=run.telemetry,
                orthogonal_readout=orth_readout,
            )

            if final_outcome is AttributionOutcome.CERTIFIED_POSITIVE:
                certified_positives_committed += 1
                committed_hits.append(mol)
            else:
                certified_negatives_committed += 1

            new_round_support.append(z_vec)

            if mol.compound_id in cal_holdout_ids:
                # Online Gibbs-Candes Adaptive Conformal Inference (`alpha_t`) update
                err_indicator = (
                    1.0
                    if abs(final_measured_log_s - _predict_ridge(weights, z_vec)) > online_q_t
                    else 0.0
                )
                alpha_t = min(0.25, max(0.01, alpha_t + gamma_aci * (alpha - err_indicator)))
                frontier_cal_z.append(z_vec)
                frontier_cal_y.append(final_measured_log_s)
            else:
                mem_z.append(z_vec)
                mem_y.append(final_measured_log_s)

            # Evaluate formal AttributedSteppingStone gate against fixed `round_tau_permille`
            if d_perm > round_tau_permille:
                probe = AttributedSteppingStone(
                    tau_permille=round_tau_permille,
                    eta_permille=eta_permille,
                    dist_permille=d_perm,
                    proposed_new_tau_permille=d_perm,
                    volume_ul=8,
                    uses_only_stock_reagents=True,
                    within_hazard_envelope=True,
                    telemetry=run.telemetry,
                    orthogonal_readout=orth_readout,
                )
                if check_attributed_frontier_update(probe):
                    approved_round_taus.append(d_perm)

        # Synchronized end-of-plate-round frontier commit (order-independent within batch!)
        if is_gated_regime:
            if new_round_support:
                certified_support.extend(new_round_support)
                for cid in remaining:
                    zv = cand_z[cid]
                    top3 = cand_top3_sq[cid]
                    for ref in new_round_support:
                        sq = sum((zv[j] - ref[j]) ** 2 for j in range(len(zv)))
                        if len(top3) < 3:
                            top3.append(sq)
                            top3.sort()
                        elif sq < top3[-1]:
                            top3[-1] = sq
                            top3.sort()
            if approved_round_taus:
                best_new_tau = max(approved_round_taus)
                if best_new_tau > tau_permille:
                    tau_permille = best_new_tau
                    stepping_stone_expansions += 1

        # End-of-round held-out surrogate MAE & cumulative trajectory snapshot
        post_round_w = _fit_ridge_linear_system(mem_z, mem_y)
        post_round_mae = sum(
            abs(frontier_eval_set[i].true_log_s - _predict_ridge(post_round_w, eval_z[i]))
            for i in range(len(frontier_eval_set))
        ) / max(1, len(frontier_eval_set))
        round_mae_curve.append(post_round_mae)
        round_poison_curve.append(poisoned_memory_entries)
        round_hits_curve.append(sum(1 for m in committed_hits if m.is_true_frontier_hit))

    # Final post-campaign evaluation on clean held-out frontier molecules
    final_weights = _fit_ridge_linear_system(mem_z, mem_y)
    claimed_tau_permille = tau_permille

    if is_gated_regime:
        eff_alpha = min(alpha, alpha_t)
        frontier_cal_scores = [
            abs(y_c - _predict_ridge(final_weights, z_c))
            for z_c, y_c in zip(frontier_cal_z, frontier_cal_y)
        ]
        seed_cal_final_scores = [
            abs(m_c.true_log_s - _predict_ridge(final_weights, z_c))
            for m_c, z_c in zip(seed_cal, z_seed_cal)
        ]
        if len(frontier_cal_scores) >= MIN_FRONTIER_CALIBRATION_SAMPLES:
            q_hat = max(
                static_q_hat,
                _conformal_quantile(frontier_cal_scores, alpha=eff_alpha),
            )
        else:
            cal_frac = len(frontier_cal_scores) / float(MIN_FRONTIER_CALIBRATION_SAMPLES)
            claimed_tau_permille = int(
                round(initial_tau_permille + cal_frac * (tau_permille - initial_tau_permille))
            )
            tau_permille = claimed_tau_permille
            pooled_scores = seed_cal_final_scores + frontier_cal_scores
            conservative_alpha = max(0.01, eff_alpha * max(0.25, cal_frac))
            radial_shift = max(0.0, (claimed_tau_permille - initial_tau_permille) / 1000.0)
            q_hat = (
                max(static_q_hat, _conformal_quantile(pooled_scores, alpha=conservative_alpha))
                + 0.35 * radial_shift
            )
    else:
        q_hat = static_q_hat

    if regime == "naive_raw_writeback":
        eval_indices = list(range(len(frontier_eval_set)))
    elif is_gated_regime:
        eval_indices = [
            i
            for i, d_val in enumerate(eval_dist)
            if int(round(1000.0 * d_val)) <= claimed_tau_permille
            and is_locally_supported_in_tree(
                eval_z[i],
                certified_support,
                local_tau=local_tau_0 * eta_scale,
                k_neighbors=3,
            )
        ]
        if not eval_indices:
            eval_indices = [
                i
                for i, d_val in enumerate(eval_dist)
                if int(round(1000.0 * d_val)) <= claimed_tau_permille
            ]
    else:
        eval_indices = [
            i
            for i, d_val in enumerate(eval_dist)
            if int(round(1000.0 * d_val)) <= tau_permille
        ]
    if not eval_indices:
        eval_indices = list(range(len(frontier_eval_set)))

    covered = 0
    for i in eval_indices:
        mol_e = frontier_eval_set[i]
        pred = _predict_ridge(final_weights, eval_z[i])
        if abs(mol_e.true_log_s - pred) <= q_hat:
            covered += 1

    abs_errors = [
        abs(frontier_eval_set[i].true_log_s - _predict_ridge(final_weights, eval_z[i]))
        for i in range(len(frontier_eval_set))
    ]
    frontier_mae = sum(abs_errors) / max(1, len(abs_errors))
    frontier_rmse = math.sqrt(sum(e * e for e in abs_errors) / max(1, len(abs_errors)))
    conformal_coverage = covered / max(1, len(eval_indices))

    true_hits = sum(1 for m in committed_hits if m.is_true_frontier_hit)
    false_hits = len(committed_hits) - true_hits
    fdr = false_hits / len(committed_hits) if committed_hits else 0.0

    ledger_ok, ledger_err = ledger.verify_chain()
    assert ledger_ok, f"Internal ledger verification failed: {ledger_err}"

    return {
        "verified_multi_ring_hits": true_hits,
        "false_positive_hits": false_hits,
        "total_claimed_hits": len(committed_hits),
        "false_discovery_rate": fdr,
        "poisoned_memory_entries": poisoned_memory_entries,
        "conformal_coverage_90": conformal_coverage,
        "frontier_surrogate_mae": frontier_mae,
        "frontier_surrogate_rmse": frontier_rmse,
        "initial_tau": initial_tau_permille / 1000.0,
        "final_tau": tau_permille / 1000.0,
        "stepping_stone_expansions": stepping_stone_expansions,
        "primary_assays_executed": primary_assays_executed,
        "robot_retries_triggered": robot_retries_triggered,
        "orthogonal_assays_triggered": orthogonal_assays_triggered,
        "total_measurements_used": total_measurements_used,
        "certified_ledger_size": len(ledger),
        "certified_negatives": certified_negatives_committed,
        "certified_positives": certified_positives_committed,
        "frontier_calibration_size": len(frontier_cal_z),
        "aci_alpha_final": round(alpha_t, 4),
        "ledger_chain_valid": ledger_ok,
        "round_mae_curve": round_mae_curve,
        "round_poison_curve": round_poison_curve,
        "round_hits_curve": round_hits_curve,
    }


def _summarize_regime_runs(runs: Sequence[dict[str, Any]], n_rounds: int) -> dict[str, Any]:
    n_s = float(max(1, len(runs)))
    round_mae_mean = [
        round(sum(float(r["round_mae_curve"][k]) for r in runs) / n_s, 4)
        for k in range(n_rounds)
    ]
    round_poison_mean = [
        round(sum(float(r["round_poison_curve"][k]) for r in runs) / n_s, 2)
        for k in range(n_rounds)
    ]
    round_hits_mean = [
        round(sum(float(r["round_hits_curve"][k]) for r in runs) / n_s, 2)
        for k in range(n_rounds)
    ]
    return {
        "mean_verified_multi_ring_hits": round(
            sum(r["verified_multi_ring_hits"] for r in runs) / n_s, 2
        ),
        "mean_false_positive_hits": round(
            sum(r["false_positive_hits"] for r in runs) / n_s, 2
        ),
        "false_discovery_rate_pct": round(
            100.0 * sum(r["false_discovery_rate"] for r in runs) / n_s, 2
        ),
        "mean_poisoned_memory_entries": round(
            sum(r["poisoned_memory_entries"] for r in runs) / n_s, 2
        ),
        "conformal_coverage_pct": round(
            100.0 * sum(r["conformal_coverage_90"] for r in runs) / n_s, 2
        ),
        "mean_frontier_mae": round(
            sum(r["frontier_surrogate_mae"] for r in runs) / n_s, 4
        ),
        "mean_frontier_rmse": round(
            sum(r["frontier_surrogate_rmse"] for r in runs) / n_s, 4
        ),
        "mean_initial_tau": round(sum(r["initial_tau"] for r in runs) / n_s, 3),
        "mean_final_tau": round(sum(r["final_tau"] for r in runs) / n_s, 3),
        "mean_stepping_stone_expansions": round(
            sum(r["stepping_stone_expansions"] for r in runs) / n_s, 2
        ),
        "mean_primary_assays": round(
            sum(r["primary_assays_executed"] for r in runs) / n_s, 2
        ),
        "mean_robot_retries": round(
            sum(r["robot_retries_triggered"] for r in runs) / n_s, 2
        ),
        "mean_orthogonal_assays": round(
            sum(r["orthogonal_assays_triggered"] for r in runs) / n_s, 2
        ),
        "mean_total_measurements": round(
            sum(r["total_measurements_used"] for r in runs) / n_s, 2
        ),
        "mean_certified_ledger_entries": round(
            sum(r["certified_ledger_size"] for r in runs) / n_s, 2
        ),
        "all_ledger_chains_valid": all(r["ledger_chain_valid"] for r in runs),
        "round_trajectories": {
            "rounds": list(range(1, n_rounds + 1)),
            "mean_frontier_mae": round_mae_mean,
            "mean_cumulative_poisoned_entries": round_poison_mean,
            "mean_cumulative_verified_hits": round_hits_mean,
        },
    }


def run_attributed_frontier_benchmark(
    *,
    csv_path: str | Path | None = None,
    force_synthetic_fallback: bool = False,
    seeds: Sequence[int] = (11, 23, 37, 53, 71),
    n_rounds: int = 9,
    batch_size: int = 20,
    eta_permille: int = 420,
    fault_flag_recall: float = 1.0,
    false_alarm_rate: float = 0.0,
    recall_sweep: Sequence[float] = (1.0, 0.9, 0.8),
) -> dict[str, Any]:
    """Run the 5-regime closed-loop attributed frontier benchmark + sensor-recall sweep."""
    molecules, dataset_label = resolve_molecule_pool(
        csv_path, force_synthetic_fallback=force_synthetic_fallback
    )
    seed_domain_all = [m for m in molecules if m.rings <= 2 and m.features[1] <= 220.0]
    frontier_all = [m for m in molecules if not (m.rings <= 2 and m.features[1] <= 220.0)]

    seed_splits: list[
        tuple[
            int,
            list[MoleculeCandidate],
            list[MoleculeCandidate],
            list[MoleculeCandidate],
            list[MoleculeCandidate],
        ]
    ] = []
    for seed in seeds:
        rng = random.Random(seed * 9973)
        seed_shuffled = list(seed_domain_all)
        rng.shuffle(seed_shuffled)
        n_cal = max(40, min(120, len(seed_shuffled) // 5))
        n_train = min(260, len(seed_shuffled) - n_cal)
        seed_train = seed_shuffled[:n_train]
        seed_cal = seed_shuffled[n_train : n_train + n_cal]

        frontier_shuffled = list(frontier_all)
        rng.shuffle(frontier_shuffled)
        n_eval = max(50, min(100, len(frontier_shuffled) // 4))
        frontier_eval = frontier_shuffled[:n_eval]
        exploration_pool = frontier_shuffled[n_eval:]
        seed_splits.append((seed, seed_train, seed_cal, exploration_pool, frontier_eval))

    per_regime_runs: dict[str, list[dict[str, Any]]] = {r: [] for r in REGIME_NAMES}
    for seed, seed_train, seed_cal, exploration_pool, frontier_eval in seed_splits:
        for regime in REGIME_NAMES:
            metrics = _run_single_seed_regime(
                regime,
                seed_train,
                seed_cal,
                exploration_pool,
                frontier_eval,
                seed=seed,
                n_rounds=n_rounds,
                batch_size=batch_size,
                eta_permille=eta_permille,
                fault_flag_recall=fault_flag_recall,
                false_alarm_rate=false_alarm_rate,
            )
            per_regime_runs[regime].append(metrics)

    summary_by_regime: dict[str, dict[str, Any]] = {
        regime: _summarize_regime_runs(runs, n_rounds=n_rounds)
        for regime, runs in per_regime_runs.items()
    }

    sweep_entries: list[dict[str, Any]] = []
    for rec_level in recall_sweep:
        rec_val = float(rec_level)
        if abs(rec_val - fault_flag_recall) < 1e-12:
            full_sum = summary_by_regime["lab_attrib_full_closed_loop"]
            eq_sum = summary_by_regime["lab_attrib_equal_budget"]
        else:
            full_runs = [
                _run_single_seed_regime(
                    "lab_attrib_full_closed_loop",
                    st,
                    sc,
                    ep,
                    fe,
                    seed=sd,
                    n_rounds=n_rounds,
                    batch_size=batch_size,
                    eta_permille=eta_permille,
                    fault_flag_recall=rec_val,
                    false_alarm_rate=false_alarm_rate,
                )
                for sd, st, sc, ep, fe in seed_splits
            ]
            eq_runs = [
                _run_single_seed_regime(
                    "lab_attrib_equal_budget",
                    st,
                    sc,
                    ep,
                    fe,
                    seed=sd,
                    n_rounds=n_rounds,
                    batch_size=batch_size,
                    eta_permille=eta_permille,
                    fault_flag_recall=rec_val,
                    false_alarm_rate=false_alarm_rate,
                )
                for sd, st, sc, ep, fe in seed_splits
            ]
            full_sum = _summarize_regime_runs(full_runs, n_rounds=n_rounds)
            eq_sum = _summarize_regime_runs(eq_runs, n_rounds=n_rounds)

        def _compact_sweep_metrics(s: dict[str, Any]) -> dict[str, Any]:
            return {
                "mean_verified_multi_ring_hits": s["mean_verified_multi_ring_hits"],
                "mean_false_positive_hits": s["mean_false_positive_hits"],
                "false_discovery_rate_pct": s["false_discovery_rate_pct"],
                "mean_poisoned_memory_entries": s["mean_poisoned_memory_entries"],
                "conformal_coverage_pct": s["conformal_coverage_pct"],
                "mean_frontier_mae": s["mean_frontier_mae"],
                "mean_frontier_rmse": s["mean_frontier_rmse"],
                "mean_total_measurements": s["mean_total_measurements"],
                "mean_certified_ledger_entries": s["mean_certified_ledger_entries"],
            }

        sweep_entries.append(
            {
                "fault_flag_recall": round(rec_val, 4),
                "false_alarm_rate": round(false_alarm_rate, 4),
                "lab_attrib_full_closed_loop": _compact_sweep_metrics(full_sum),
                "lab_attrib_equal_budget": _compact_sweep_metrics(eq_sum),
            }
        )

    return {
        "benchmark": "lab_attrib_attributed_chem_rrt_frontier",
        "dataset": dataset_label,
        "n_seeds": len(seeds),
        "seeds": list(seeds),
        "n_rounds": n_rounds,
        "batch_size": batch_size,
        "primary_budget_assays": n_rounds * batch_size,
        "fault_flag_recall": round(fault_flag_recall, 4),
        "false_alarm_rate": round(false_alarm_rate, 4),
        "nominal_conformal_target_pct": 90.0,
        "hit_definition": (
            f"rings >= {HIT_MIN_RINGS} and true_log_solubility >= {HIT_MIN_LOG_SOLUBILITY}"
        ),
        "regimes": summary_by_regime,
        "sensor_recall_sweep": sweep_entries,
    }
