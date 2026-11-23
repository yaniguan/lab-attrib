"""Command-line interface for `lab-attrib` (`demo`, `attribute`, `benchmark`)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from lab_attrib.attributed_frontier import run_attributed_frontier_benchmark
from lab_attrib.blame import (
    AssayTelemetry,
    OrthogonalAssayReadout,
    OrthogonalModality,
    diagnose_assay,
)
from lab_attrib.ledger import CertifiedLedger
from lab_attrib.orthogonal_router import route_orthogonal_assay
from lab_attrib.xrd_deconvolve import build_gaussian_spectrum, deconvolve_spectrum


def _build_demo_payload() -> dict[str, Any]:
    grid = [10.0 + 1.5 * i for i in range(40)]
    precursor_a = build_gaussian_spectrum(grid, [(24.0, 1.0, 2.0), (48.0, 0.85, 2.0)])
    precursor_b_ghost = build_gaussian_spectrum(
        grid, [(34.0, 0.95, 2.2), (52.0, 0.90, 2.2), (20.0, 0.40, 1.8)]
    )
    target_phase = build_gaussian_spectrum(grid, [(34.0, 1.0, 2.1), (52.0, 0.92, 2.1)])

    # 1. A-Lab ghost-peak trap: only unreacted precursor B is present (zero novel phase!)
    ghost_mixture = [1.05 * precursor_b_ghost[i] + 0.15 * precursor_a[i] for i in range(len(grid))]
    ghost_deconv = deconvolve_spectrum(
        ghost_mixture,
        [precursor_a, precursor_b_ghost],
        target_phase,
        min_hit_amplitude=0.25,
        noise_sigma=0.03,
    )
    ghost_telemetry = AssayTelemetry(
        dispense_error_permille=15,
        aspiration_clean=True,
        incubation_window_clean=True,
        seal_intact=True,
        positive_control_passed=True,
        blank_uncontaminated=True,
        detector_calibrated=True,
        precursor_overlap_permille=ghost_deconv.precursor_overlap_permille,
        spectral_residual_permille=ghost_deconv.spectral_residual_permille,
        snr_tenths=ghost_deconv.snr_tenths,
        target_signal_present=True,
    )
    ghost_primary_report = diagnose_assay(ghost_telemetry)
    evoi_decision = route_orthogonal_assay(
        ghost_telemetry,
        prior_hit_prob=0.30,
        available_volume_ul=5.0,
        sample_phase="solid",
    )
    raman_refutation = OrthogonalAssayReadout(
        modality=OrthogonalModality.RAMAN,
        instrument_calibrated=True,
        orthogonal_snr_tenths=82,
        selectivity_clean=True,
        target_confirmed=False,
    )
    ghost_resolved_report = diagnose_assay(ghost_telemetry, raman_refutation)

    # 2. Robotic cavitation zero-signal fault (Sakana / La Agente Optima negative conflation trap)
    cavitation_telemetry = AssayTelemetry(
        dispense_error_permille=175,
        aspiration_clean=False,
        incubation_window_clean=True,
        seal_intact=True,
        positive_control_passed=True,
        blank_uncontaminated=True,
        detector_calibrated=True,
        precursor_overlap_permille=10,
        spectral_residual_permille=30,
        snr_tenths=55,
        target_signal_present=False,
    )
    cavitation_report = diagnose_assay(cavitation_telemetry)

    # 3. Clean certified discovery
    hit_mixture = [0.10 * precursor_a[i] + 1.15 * target_phase[i] for i in range(len(grid))]
    hit_deconv = deconvolve_spectrum(
        hit_mixture,
        [precursor_a, precursor_b_ghost],
        target_phase,
        min_hit_amplitude=0.25,
        noise_sigma=0.03,
    )
    hit_telemetry = hit_deconv.to_assay_telemetry()
    hit_report = diagnose_assay(hit_telemetry)

    # 4. Append certified outcomes to SHA-256 hash-chained CertifiedLedger
    ledger = CertifiedLedger()
    ledger.append(
        sample_id="ALAB-GHOST-001",
        candidate_features={"rings": 4.0, "mw": 342.4},
        measured_value=-4.85,
        telemetry=ghost_telemetry,
        orthogonal_readout=raman_refutation,
    )
    ledger.append(
        sample_id="NOVEL-HIT-002",
        candidate_features={"rings": 3.0, "mw": 288.3},
        measured_value=-2.45,
        telemetry=hit_telemetry,
    )
    chain_ok, _ = ledger.verify_chain()

    return {
        "a_lab_ghost_peak_demo": {
            "naive_peak_match_cosine": round(ghost_deconv.naive_peak_match_cosine, 4),
            "naive_claims_hit": ghost_deconv.naive_claims_hit,
            "nnls_precursor_overlap_permille": ghost_deconv.precursor_overlap_permille,
            "primary_diagnosis": ghost_primary_report.to_dict(),
            "evoi_orthogonal_routing": evoi_decision.to_dict(),
            "post_raman_diagnosis": ghost_resolved_report.to_dict(),
        },
        "robotic_cavitation_zero_signal_demo": cavitation_report.to_dict(),
        "clean_target_discovery_demo": hit_report.to_dict(),
        "certified_ledger_demo": {
            "entries": len(ledger),
            "head_hash": ledger.head_hash,
            "chain_valid": chain_ok,
        },
    }


def _cmd_demo(args: argparse.Namespace) -> int:
    payload = _build_demo_payload()
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    ghost = payload["a_lab_ghost_peak_demo"]
    cav = payload["robotic_cavitation_zero_signal_demo"]
    hit = payload["clean_target_discovery_demo"]
    led = payload["certified_ledger_demo"]

    print("=== lab-attrib: Post-Execution Causal Blame & Certified Negative Ledger ===")
    print(
        f"[1] A-Lab Precursor Ghost-Peak Trap:\n"
        f"    Naive cosine={ghost['naive_peak_match_cosine']:.3f} (naive_claims_hit={ghost['naive_claims_hit']})\n"
        f"    NNLS precursor_overlap={ghost['nnls_precursor_overlap_permille']} permille (> 150 guard)\n"
        f"    Primary verdict : {ghost['primary_diagnosis']['outcome']} -> {ghost['primary_diagnosis']['action']}\n"
        f"    EVOI router     : selected={ghost['evoi_orthogonal_routing']['selected_modality']} "
        f"(Net EVOI=${ghost['evoi_orthogonal_routing']['best_net_evoi_usd']:.2f})\n"
        f"    Raman resolution: {ghost['post_raman_diagnosis']['outcome']} "
        f"[{ghost['post_raman_diagnosis']['lean_theorem_tag']}]"
    )
    print(
        f"[2] Robotic Cavitation Zero-Signal Fault:\n"
        f"    Verdict         : {cav['outcome']} -> {cav['action']} "
        f"(admit_to_memory={cav['admit_to_memory']})\n"
        f"    Fault clauses   : {', '.join(cav['violated_execution_clauses'])}"
    )
    print(
        f"[3] Clean Target Phase Hit:\n"
        f"    Verdict         : {hit['outcome']} -> {hit['action']} "
        f"[{hit['lean_theorem_tag']}]"
    )
    print(
        f"[4] SHA-256 Proof-Carrying Ledger:\n"
        f"    Committed entries={led['entries']}, chain_valid={led['chain_valid']}, "
        f"head_hash={led['head_hash'][:16]}..."
    )
    return 0


def _load_json_arg(raw: str) -> dict[str, Any]:
    candidate_path = Path(raw)
    if candidate_path.is_file():
        return dict(json.loads(candidate_path.read_text(encoding="utf-8")))
    return dict(json.loads(raw))


def _cmd_attribute(args: argparse.Namespace) -> int:
    if args.telemetry is None:
        telemetry = AssayTelemetry(
            dispense_error_permille=18,
            aspiration_clean=True,
            incubation_window_clean=True,
            seal_intact=True,
            positive_control_passed=True,
            blank_uncontaminated=True,
            detector_calibrated=True,
            precursor_overlap_permille=640,
            spectral_residual_permille=55,
            snr_tenths=80,
            target_signal_present=True,
        )
    else:
        telemetry = AssayTelemetry.from_dict(_load_json_arg(args.telemetry))

    secondary = (
        OrthogonalAssayReadout.from_dict(_load_json_arg(args.orthogonal))
        if args.orthogonal is not None
        else None
    )
    report = diagnose_assay(telemetry, secondary)
    out = report.to_dict()
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print(
            f"Outcome={out['outcome']} | Action={out['action']} | "
            f"Admit={out['admit_to_memory']} | Theorem={out['lean_theorem_tag']}"
        )
    return 0


def _default_benchmark_output_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "benchmarks" / "results" / "lab_attrib_benchmark.json"


def _cmd_benchmark(args: argparse.Namespace) -> int:
    results = run_attributed_frontier_benchmark(
        csv_path=args.csv,
        n_rounds=args.rounds,
        batch_size=args.batch_size,
    )
    out_path = Path(args.output) if args.output else _default_benchmark_output_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    formatted = json.dumps(results, indent=2) + "\n"
    out_path.write_text(formatted, encoding="utf-8")

    if args.json:
        print(formatted, end="")
    else:
        print(f"Benchmark written to: {out_path}")
        for regime, stats in results["regimes"].items():
            print(
                f"  {regime:28s} | hits={stats['mean_verified_multi_ring_hits']:5.2f} "
                f"| FDR={stats['false_discovery_rate_pct']:5.2f}% "
                f"| poisoned={stats['mean_poisoned_memory_entries']:5.2f} "
                f"| cov90={stats['conformal_coverage_pct']:5.2f}%"
            )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lab-attrib",
        description=(
            "Post-execution causal blame calculus, NNLS precursor ghost-peak deconvolution, "
            "and proof-carrying negative ledger."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_demo = sub.add_parser("demo", help="Run end-to-end diagnostic & ledger showcase")
    p_demo.add_argument("--json", action="store_true", help="Emit structured JSON")
    p_demo.set_defaults(func=_cmd_demo)

    p_attr = sub.add_parser("attribute", help="Attribute a post-execution assay telemetry record")
    p_attr.add_argument(
        "--telemetry",
        default=None,
        help="Path to JSON file or inline JSON object for AssayTelemetry",
    )
    p_attr.add_argument(
        "--orthogonal",
        default=None,
        help="Optional path to JSON file or inline JSON for OrthogonalAssayReadout",
    )
    p_attr.add_argument("--json", action="store_true", help="Emit structured JSON")
    p_attr.set_defaults(func=_cmd_attribute)

    p_bench = sub.add_parser(
        "benchmark",
        help="Run the 4-regime closed-loop attributed Chem-RRT frontier campaign",
    )
    p_bench.add_argument("--csv", default=None, help="Optional path to Delaney ESOL CSV")
    p_bench.add_argument("--rounds", type=int, default=9, help="Campaign rounds per seed")
    p_bench.add_argument("--batch-size", type=int, default=20, help="Assays per round")
    p_bench.add_argument("--output", default=None, help="Output JSON path")
    p_bench.add_argument("--json", action="store_true", help="Emit benchmark JSON to stdout")
    p_bench.set_defaults(func=_cmd_benchmark)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
