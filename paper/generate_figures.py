"""Generate 4 publication-grade vector PDF figures for the standalone `lab-attrib` research paper.

Reads all closed-loop active-learning trajectories and multi-regime frontier metrics directly
from `benchmarks/results/lab_attrib_benchmark.json` (zero hardcoded closed-loop numbers).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
from matplotlib.patches import FancyBboxPatch
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent / "figures"
BENCH_JSON = REPO_ROOT / "benchmarks" / "results" / "lab_attrib_benchmark.json"

INK = "#111827"
SLATE = "#374151"
MUTED = "#6B7280"
GRID = "#E5E7EB"
BLUE = "#1D4ED8"
TEAL = "#0D9488"
AMBER = "#D97706"
CRIMSON = "#BE123C"
PURPLE = "#6D28D9"

plt.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.edgecolor": "#D1D5DB",
        "axes.linewidth": 0.8,
        "xtick.color": SLATE,
        "ytick.color": SLATE,
        "text.color": INK,
    }
)


def _despine(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    sub: str,
    *,
    fc: str,
    ec: str,
    tc: str = INK,
) -> None:
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.2,
        zorder=2,
    )
    ax.add_patch(p)
    ax.text(
        x + w / 2,
        y + h * 0.63,
        title,
        ha="center",
        va="center",
        fontsize=8.1,
        fontweight="bold",
        color=tc,
        zorder=3,
    )
    ax.text(
        x + w / 2,
        y + h * 0.28,
        sub,
        ha="center",
        va="center",
        fontsize=6.85,
        color=SLATE,
        zorder=3,
    )


def make_fig1_scm_and_calculus() -> None:
    """Page-1 Hero Diagram: Confounded Wet-Lab Readout SCM -> Lean 4 Blame Partition -> Routing."""
    fig, ax = plt.subplots(figsize=(10.2, 2.85))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.155,
        0.94,
        "A. Latent Wet-Lab Causal Factors",
        ha="center",
        fontsize=8.4,
        fontweight="bold",
        color=SLATE,
    )
    ax.text(
        0.50,
        0.94,
        "B. LabAttribLean 4-Way Blame Calculus (31 Thms)",
        ha="center",
        fontsize=8.4,
        fontweight="bold",
        color=BLUE,
    )
    ax.text(
        0.845,
        0.94,
        "C. Certified Action & Memory Routing",
        ha="center",
        fontsize=8.4,
        fontweight="bold",
        color=TEAL,
    )

    _box(
        ax,
        0.02,
        0.66,
        0.27,
        0.20,
        "Execution Cleanliness E in {0,1}",
        "Acoustic · Wait Window · Tactile · +/- Ctrls",
        fc="#F9FAFB",
        ec="#9CA3AF",
    )
    _box(
        ax,
        0.02,
        0.39,
        0.27,
        0.20,
        "Spectral Identifiability I in {0,1}",
        "Fit-Calibrated R_wp <= R_max(m) & Overlap <= tau",
        fc="#F9FAFB",
        ec="#9CA3AF",
    )
    _box(
        ax,
        0.02,
        0.12,
        0.27,
        0.20,
        "True Biochemistry Z in {0,1}",
        "Latent Target Phase / Molecular Activity",
        fc="#EFF6FF",
        ec=BLUE,
        tc=BLUE,
    )

    panel = FancyBboxPatch(
        (0.345, 0.08),
        0.31,
        0.80,
        boxstyle="round,pad=0.012,rounding_size=0.022",
        facecolor="#EFF6FF",
        edgecolor=BLUE,
        lw=1.5,
        zorder=1,
    )
    ax.add_patch(panel)
    _box(
        ax,
        0.365,
        0.68,
        0.27,
        0.16,
        "E = 0  =>  .executionArtifact",
        "corrupted_run_never_poisons_memory",
        fc="#FFF1F2",
        ec=CRIMSON,
        tc=CRIMSON,
    )
    _box(
        ax,
        0.365,
        0.49,
        0.27,
        0.16,
        "E = 1, I = 0  =>  .ambiguousChar",
        "unidentifiable_spectrum_never_certified",
        fc="#FFFBEB",
        ec=AMBER,
        tc="#B45309",
    )
    _box(
        ax,
        0.365,
        0.30,
        0.27,
        0.16,
        "E = 1, I = 1, Y >= theta  =>  .certPos",
        "certified_positive_sound",
        fc="#F0FDFA",
        ec=TEAL,
        tc="#0F766E",
    )
    _box(
        ax,
        0.365,
        0.11,
        0.27,
        0.16,
        "E = 1, I = 1, Y < theta   =>  .certNeg",
        "certified_negative_sound",
        fc="#F0FDFA",
        ec=TEAL,
        tc="#0F766E",
    )

    _box(
        ax,
        0.71,
        0.68,
        0.27,
        0.16,
        "Quarantine & Hardware Retry",
        "0 Surrogate Write-Back · Re-queue Deck",
        fc="#FFFFFF",
        ec=CRIMSON,
        tc=CRIMSON,
    )
    _box(
        ax,
        0.71,
        0.49,
        0.27,
        0.16,
        "EVOI Orthogonal Assay Router",
        "Dispatch Raman / LC-MS / TEM Check",
        fc="#FFFFFF",
        ec=AMBER,
        tc="#B45309",
    )
    _box(
        ax,
        0.71,
        0.18,
        0.27,
        0.25,
        "SHA-256 Certified Ledger",
        "Admit Hit & Proof-Carrying Negative\nRecalibrate Annular Chem-RRT tau_{t+1}",
        fc="#F0FDFA",
        ec=TEAL,
        tc="#0F766E",
    )

    arr = dict(arrowstyle="-|>,head_width=0.2,head_length=0.32", lw=1.25, color=BLUE)
    for yy in (0.76, 0.49, 0.22):
        ax.annotate("", xy=(0.342, yy), xytext=(0.292, yy), arrowprops=arr)
    for yy, col in ((0.76, CRIMSON), (0.57, AMBER), (0.38, TEAL), (0.19, TEAL)):
        dest_y = 0.76 if yy > 0.65 else (0.57 if yy > 0.45 else 0.30)
        ax.annotate(
            "",
            xy=(0.708, dest_y),
            xytext=(0.638, yy),
            arrowprops=dict(arrowstyle="-|>,head_width=0.2,head_length=0.32", lw=1.25, color=col),
        )

    fig.tight_layout(pad=0.2)
    fig.savefig(OUT_DIR / "fig1_scm_and_calculus.pdf", bbox_inches="tight")
    plt.close(fig)


def make_fig2_xrd_deconvolution() -> None:
    """2-panel XRD Precursor Ghost-Peak Deconvolution (`A-Lab` Mode + Real RRUFF/DARA & A-Lab Audit)."""
    real_json = REPO_ROOT / "results" / "real_spectra_benchmark.json"
    real_data = json.loads(real_json.read_text(encoding="utf-8"))
    xrd_m = real_data["suites"]["rruff_dara_powder_xrd"]["methods"]

    rng = np.random.default_rng(42)
    two_theta = np.linspace(15.0, 65.0, 500)

    def peaks(centers: list[tuple[float, float]]) -> np.ndarray:
        y = np.zeros_like(two_theta)
        for c, amp in centers:
            y += amp * np.exp(-0.5 * ((two_theta - c) / 0.42) ** 2)
        return y

    p1 = peaks([(22.4, 0.85), (33.1, 1.00), (47.6, 0.62)])
    p2 = peaks([(28.6, 0.92), (38.9, 0.70), (54.2, 0.55)])
    putative_target = peaks([(22.6, 0.80), (33.2, 0.95), (41.5, 0.65), (54.0, 0.50)])

    obs_ghost = 0.72 * p1 + 0.64 * p2 + rng.normal(0, 0.02, size=two_theta.size)
    obs_ghost = np.clip(obs_ghost, 0, None)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 2.75))

    ax = axes[0]
    ax.plot(two_theta, obs_ghost, color=INK, lw=1.4, label="Observed Diffractogram (0% Novel Phase)")
    ax.plot(two_theta, 0.72 * p1 + 0.64 * p2, "--", color=CRIMSON, lw=1.3, label="NNLS Precursor Fit (Overlap > 15%)")
    ax.plot(two_theta, 0.55 * putative_target, ":", color=BLUE, lw=1.25, label="Putative Target Template (Missing 41.5 deg)")
    ax.axvspan(40.5, 42.5, color="#FEF3C7", alpha=0.8, label="Absent Diagnostic Target Peak")
    ax.set_xlabel("Diffraction Angle 2-theta (deg)", fontsize=8.0)
    ax.set_ylabel("Normalized Intensity (a.u.)", fontsize=8.0)
    ax.set_title("A. Illustrative Ghost-Phase Mechanism (A-Lab Mode)", fontsize=8.5, fontweight="bold", pad=6)
    ax.legend(frameon=False, fontsize=6.5, loc="upper right")
    ax.grid(True, color=GRID, lw=0.6)
    _despine(ax)

    ax = axes[1]
    cats = ["Naive Peak\nCosine", "Rigid Zero-Shift\nGaussian NNLS", "Shift-Voigt\nL1-NNLS (Ours)"]
    fdr = [
        float(xrd_m["naive_peak_cosine"]["false_discovery_rate_pct"]),
        float(xrd_m["rigid_gaussian_nnls"]["false_discovery_rate_pct"]),
        float(xrd_m["shift_voigt_l1_nnls"]["false_discovery_rate_pct"]),
    ]
    recall = [
        float(xrd_m["naive_peak_cosine"]["recall_pct"]),
        float(xrd_m["rigid_gaussian_nnls"]["recall_pct"]),
        float(xrd_m["shift_voigt_l1_nnls"]["recall_pct"]),
    ]
    x = np.arange(len(cats))
    w = 0.34
    b1 = ax.bar(x - w / 2, fdr, width=w, color=CRIMSON, label="False Discovery Rate (%) down", zorder=3)
    b2 = ax.bar(x + w / 2, recall, width=w, color=TEAL, label="True Phase Recall (%) up", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=7.4)
    ax.set_ylim(0, 128)
    ax.set_ylabel("Rate (%)", fontsize=8.0)
    ax.set_title("B. Real RRUFF + DARA Powder XRD Benchmark (N = 8)", fontsize=8.5, fontweight="bold", pad=6)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    _despine(ax)
    ax.legend(frameon=False, fontsize=6.9, loc="upper right")
    for b in list(b1) + list(b2):
        h = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2,
            h + 2.0,
            f"{h:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7.0,
            fontweight="bold",
        )

    fig.tight_layout(pad=0.6)
    fig.savefig(OUT_DIR / "fig2_xrd_ghost_deconvolution.pdf", bbox_inches="tight")
    plt.close(fig)


def make_fig3_active_learning_divergence(bench: dict[str, Any]) -> None:
    """3-panel closed-loop ESOL active learning read directly from `lab_attrib_benchmark.json`."""
    regimes = bench["regimes"]
    naive_r = regimes["naive_raw_writeback"]
    eq_r = regimes["lab_attrib_equal_budget"]
    full_r = regimes["lab_attrib_full_closed_loop"]

    rounds = full_r["round_trajectories"]["rounds"]
    naive_mae = naive_r["round_trajectories"]["mean_frontier_mae"]
    eq_mae = eq_r["round_trajectories"]["mean_frontier_mae"]
    attrib_mae = full_r["round_trajectories"]["mean_frontier_mae"]

    naive_poison = naive_r["round_trajectories"]["mean_cumulative_poisoned_entries"]
    attrib_poison = full_r["round_trajectories"]["mean_cumulative_poisoned_entries"]

    naive_hits = naive_r["round_trajectories"]["mean_cumulative_verified_hits"]
    eq_hits = eq_r["round_trajectories"]["mean_cumulative_verified_hits"]
    attrib_hits = full_r["round_trajectories"]["mean_cumulative_verified_hits"]

    fig, axes = plt.subplots(1, 3, figsize=(10.3, 2.75))

    ax = axes[0]
    ax.plot(rounds, naive_mae, "-^", color=CRIMSON, lw=1.9, ms=4.2, label="Naive Raw Write-Back")
    ax.plot(rounds, eq_mae, "--s", color=BLUE, lw=1.8, ms=3.8, label="Equal-Budget ATTRIB (180)")
    ax.plot(rounds, attrib_mae, "-o", color=TEAL, lw=2.2, ms=4.2, label="Full Closed-Loop ATTRIB")
    ax.set_xlabel("Active-Learning Campaign Round", fontsize=7.9)
    ax.set_ylabel("Frontier Surrogate MAE (log mol/L) down", fontsize=7.8)
    mae_all = list(naive_mae) + list(eq_mae) + list(attrib_mae)
    ax.set_ylim(max(0.5, min(mae_all) * 0.88), max(mae_all) * 1.12)
    ax.set_title("A. Surrogate Error Trajectory (ESOL)", fontsize=8.5, fontweight="bold", pad=6)
    ax.grid(True, color=GRID, lw=0.6)
    _despine(ax)
    ax.legend(frameon=False, fontsize=6.4, loc="best")

    ax = axes[1]
    ax.plot(rounds, naive_poison, "-^", color=CRIMSON, lw=2.0, ms=4.2, label=f"Naive ({naive_poison[-1]:.1f} poisoned)")
    ax.plot(rounds, attrib_poison, "-o", color=BLUE, lw=2.2, ms=4.2, label=f"ATTRIB recall=1.0 ({attrib_poison[-1]:.1f})")
    ax.fill_between(rounds, naive_poison, color=CRIMSON, alpha=0.10)
    ax.set_xlabel("Active-Learning Campaign Round", fontsize=7.9)
    ax.set_ylabel("Cumulative Poisoned Records down", fontsize=7.8)
    ax.set_ylim(-2.5, max(naive_poison) * 1.18)
    ax.set_title("B. Memory Poisoning Accumulation", fontsize=8.5, fontweight="bold", pad=6)
    ax.grid(True, color=GRID, lw=0.6)
    _despine(ax)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")

    ax = axes[2]
    ax.plot(
        rounds,
        naive_hits,
        "--^",
        color=CRIMSON,
        lw=1.8,
        ms=4.2,
        label=f"Naive ({naive_hits[-1]:.1f}, FDR {naive_r['false_discovery_rate_pct']:.1f}%)",
    )
    ax.plot(
        rounds,
        eq_hits,
        "--s",
        color=BLUE,
        lw=1.8,
        ms=3.8,
        label=f"Equal-Budget ({eq_hits[-1]:.1f}, FDR {eq_r['false_discovery_rate_pct']:.1f}%)",
    )
    ax.plot(
        rounds,
        attrib_hits,
        "-o",
        color=TEAL,
        lw=2.2,
        ms=4.2,
        label=f"Full ATTRIB ({attrib_hits[-1]:.1f}, FDR {full_r['false_discovery_rate_pct']:.1f}%)",
    )
    ax.set_xlabel("Active-Learning Campaign Round", fontsize=7.9)
    ax.set_ylabel("Cumulative Verified Multi-Ring Hits up", fontsize=7.8)
    ax.set_ylim(0, max(max(attrib_hits), max(naive_hits)) * 1.25)
    ax.set_title("C. True Discovery Yield (Delaney ESOL)", fontsize=8.5, fontweight="bold", pad=6)
    ax.grid(True, color=GRID, lw=0.6)
    _despine(ax)
    ax.legend(frameon=False, fontsize=6.3, loc="upper left")

    fig.tight_layout(pad=0.6)
    fig.savefig(OUT_DIR / "fig3_active_learning_divergence.pdf", bbox_inches="tight")
    plt.close(fig)


def make_fig4_attributed_frontier(bench: dict[str, Any]) -> None:
    """2-panel Attributed Annular `Chem-RRT` Stepping-Stone Calibration & Sensor-Recall Sweep."""
    regimes = bench["regimes"]
    sweep = {float(row["fault_flag_recall"]): row for row in bench["sensor_recall_sweep"]}

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 2.8))

    ax = axes[0]
    labels = [
        "Static\nPre-Gate",
        "Naive Raw\nWrite-Back",
        "Unattrib.\nProbes",
        "Equal-Budget\nATTRIB (180)",
        "Full Closed-\nLoop ATTRIB",
    ]
    keys = [
        "pre_gate_only_static",
        "naive_raw_writeback",
        "unattributed_frontier_probes",
        "lab_attrib_equal_budget",
        "lab_attrib_full_closed_loop",
    ]
    hits_vals = [float(regimes[k]["mean_verified_multi_ring_hits"]) for k in keys]
    cov_vals = [float(regimes[k]["conformal_coverage_pct"]) for k in keys]
    cols = [AMBER, CRIMSON, PURPLE, BLUE, TEAL]
    bars = ax.bar(labels, hits_vals, color=cols, width=0.54, zorder=3)
    ax.set_ylabel("Verified Multi-Ring Hits up", fontsize=8.0)
    ax.set_ylim(0, max(hits_vals) * 1.32)
    ax.set_title("A. 5-Regime Delaney ESOL Campaign (9x20 Primary Budget)", fontsize=8.4, fontweight="bold", pad=6)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    _despine(ax)
    ax.tick_params(axis="x", labelsize=6.9)
    for b, hv, cv in zip(bars, hits_vals, cov_vals):
        ax.text(
            b.get_x() + b.get_width() / 2,
            hv + 0.6,
            f"{hv:.1f}\n({cv:.1f}% cov)",
            ha="center",
            va="bottom",
            fontsize=6.5,
            fontweight="bold",
        )

    ax = axes[1]
    recalls = sorted(sweep.keys(), reverse=True)
    rec_labels = [f"Recall = {r:.1f}" for r in recalls]
    fdr_vals = [float(sweep[r]["lab_attrib_full_closed_loop"]["false_discovery_rate_pct"]) for r in recalls]
    pois_vals = [float(sweep[r]["lab_attrib_full_closed_loop"]["mean_poisoned_memory_entries"]) for r in recalls]
    x = np.arange(len(recalls))
    w = 0.34
    b_fdr = ax.bar(x - w / 2, fdr_vals, width=w, color=CRIMSON, label="Full-Loop FDR (%)", zorder=3)
    b_pois = ax.bar(x + w / 2, pois_vals, width=w, color=AMBER, label="Poisoned Memory Entries", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(rec_labels, fontsize=7.6)
    ax.set_ylim(0, max(max(fdr_vals + pois_vals), 5.0) * 1.30)
    ax.set_ylabel("Rate (%) / Count", fontsize=8.0)
    ax.set_title("B. Imperfect-Sensor Recall Sweep (Full Closed-Loop)", fontsize=8.4, fontweight="bold", pad=6)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    _despine(ax)
    ax.legend(frameon=False, fontsize=6.9, loc="upper left")
    for b in list(b_fdr) + list(b_pois):
        h = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2,
            h + 0.35,
            f"{h:.2f}",
            ha="center",
            va="bottom",
            fontsize=6.9,
            fontweight="bold",
        )

    fig.tight_layout(pad=0.6)
    fig.savefig(OUT_DIR / "fig4_attributed_chem_rrt.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bench = json.loads(BENCH_JSON.read_text(encoding="utf-8"))
    make_fig1_scm_and_calculus()
    make_fig2_xrd_deconvolution()
    make_fig3_active_learning_divergence(bench)
    make_fig4_attributed_frontier(bench)
    print(f"Generated 4 vector PDF figures in {OUT_DIR} from {BENCH_JSON.name}")


if __name__ == "__main__":
    main()
