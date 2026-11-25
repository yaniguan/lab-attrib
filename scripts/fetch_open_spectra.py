#!/usr/bin/env python3
"""On-demand fetcher & live URL verifier for `lab-attrib` open characterization datasets.

Downloads representative spectral samples or full dataset files from verified public archives
(`RRUFF`, `COD`, `MP-20`, `A-Lab GPSS`, `MassSpecGym`, `NMRShiftDB2`, `BMRB`, `AqSolDB`) into
the gitignored `data/open_datasets/` directory, recording SHA-256 checksums and SPDX provenance.
Also exports the bundled zero-network CC0/permissive oxide & mineral XRD + Raman reference panel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from lab_attrib.open_datasets import (  # noqa: E402
    OPEN_CHARACTERIZATION_DATASETS,
    REFERENCE_MINERAL_PHASES,
    VALID_DATASET_MODALITIES,
    OpenCharacterizationDataset,
    get_open_dataset,
    list_open_datasets,
    render_reference_raman_spectrum,
    render_reference_xrd_spectrum,
    run_offline_reference_panel_benchmark,
)

DEFAULT_DATA_DIR = REPO_ROOT / "data" / "open_datasets"
USER_AGENT = "Mozilla/5.0 (compatible; lab-attrib-open-datasets/0.1)"


def _infer_filename(ds: OpenCharacterizationDataset, url: str) -> str:
    tail = url.rstrip("/").rsplit("/", 1)[-1].split("?", 1)[0]
    if "." in tail and len(tail) <= 96:
        return tail
    if ds.dataset_id == "aqsoldb_solubility":
        return "curated-solubility-dataset.tab"
    if "api/" in url or url.endswith("/cod/") or url.endswith(".io/"):
        return f"{ds.dataset_id}_meta.txt"
    return f"{ds.dataset_id}_sample.dat"


def verify_dataset_url(url: str, *, timeout_sec: float = 12.0) -> dict[str, Any]:
    """Perform a lightweight live HTTP GET check reading at most 2 KiB."""
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Range": "bytes=0-2047",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            chunk = resp.read(2048)
            status = int(getattr(resp, "status", 200))
            return {
                "url": url,
                "ok": 200 <= status < 400 and len(chunk) > 0,
                "status_code": status,
                "preview_bytes": len(chunk),
            }
    except urllib.error.HTTPError as exc:
        # Retry without Range header if server rejects Range with 416/400
        if exc.code in (400, 416):
            req_plain = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            try:
                with urllib.request.urlopen(req_plain, timeout=timeout_sec) as resp2:
                    chunk2 = resp2.read(2048)
                    status2 = int(getattr(resp2, "status", 200))
                    return {
                        "url": url,
                        "ok": 200 <= status2 < 400 and len(chunk2) > 0,
                        "status_code": status2,
                        "preview_bytes": len(chunk2),
                    }
            except Exception as inner_exc:
                return {"url": url, "ok": False, "error": str(inner_exc)}
        return {"url": url, "ok": False, "status_code": int(exc.code), "error": str(exc)}
    except Exception as exc:
        return {"url": url, "ok": False, "error": str(exc)}


def fetch_dataset_sample(
    ds: OpenCharacterizationDataset,
    output_root: Path,
    *,
    max_bytes: int = 2_000_000,
    timeout_sec: float = 25.0,
) -> dict[str, Any]:
    """Download `ds.sample_download_url` (or `ds.url`) into gitignored `output_root/<dataset_id>/`."""
    target_url = ds.sample_download_url or ds.url
    ds_dir = output_root / ds.dataset_id
    ds_dir.mkdir(parents=True, exist_ok=True)
    filename = _infer_filename(ds, target_url)
    dest_path = ds_dir / filename

    req = urllib.request.Request(target_url, headers={"User-Agent": USER_AGENT})
    hasher = hashlib.sha256()
    bytes_written = 0
    truncated = False

    with urllib.request.urlopen(req, timeout=timeout_sec) as resp, dest_path.open("wb") as out_fh:
        while True:
            to_read = 65536 if max_bytes <= 0 else min(65536, max(0, max_bytes - bytes_written))
            if max_bytes > 0 and to_read == 0:
                extra = resp.read(1)
                truncated = len(extra) > 0
                break
            chunk = resp.read(to_read)
            if not chunk:
                break
            out_fh.write(chunk)
            hasher.update(chunk)
            bytes_written += len(chunk)

    # For partial previews, trim incomplete trailing record so downstream parsers succeed cleanly
    if truncated and dest_path.suffix.lower() in (".tsv", ".csv", ".tab", ".txt"):
        raw = dest_path.read_bytes()
        last_nl = raw.rfind(b"\n")
        if last_nl > 0:
            trimmed = raw[: last_nl + 1]
            dest_path.write_bytes(trimmed)
            bytes_written = len(trimmed)
            hasher = hashlib.sha256(trimmed)
    elif truncated and dest_path.suffix.lower() == ".json":
        raw_text = dest_path.read_bytes().decode("utf-8", errors="ignore")
        if raw_text.lstrip().startswith("["):
            cut = raw_text.rfind("\n  },")
            if cut > 0:
                repaired = (raw_text[: cut + 4] + "\n]\n").encode("utf-8")
                dest_path.write_bytes(repaired)
                bytes_written = len(repaired)
                hasher = hashlib.sha256(repaired)

    manifest = {
        "dataset_id": ds.dataset_id,
        "modality": ds.modality,
        "spdx_license": ds.spdx_license,
        "commercial_use_ok": ds.commercial_use_ok,
        "source_url": target_url,
        "saved_file": str(dest_path.name),
        "bytes_written": bytes_written,
        "truncated_preview": truncated,
        "sha256": hasher.hexdigest(),
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (ds_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def export_offline_reference_panel(output_root: Path) -> dict[str, Any]:
    """Materialize bundled XRD `.xy` and Raman `.jdx` reference files into `output_root/reference_panel/`."""
    panel_dir = output_root / "reference_panel"
    panel_dir.mkdir(parents=True, exist_ok=True)
    xrd_grid = [10.0 + 0.25 * i for i in range(241)]
    raman_grid = [100.0 + 2.0 * i for i in range(526)]

    exported_files: list[str] = []
    for phase in REFERENCE_MINERAL_PHASES:
        xrd_y = render_reference_xrd_spectrum(phase.phase_id, xrd_grid)
        xy_lines = [
            f"##NAMES={phase.mineral_name}",
            f"##IDEAL CHEMISTRY={phase.formula}",
            f"##SPACE GROUP={phase.space_group}",
            f"##EXTERNAL_REF={phase.external_reference_id}",
            *[f"{xrd_grid[i]:.2f}, {xrd_y[i]:.6f}" for i in range(len(xrd_grid))],
        ]
        xy_path = panel_dir / f"{phase.phase_id}_xrd.xy"
        xy_path.write_text("\n".join(xy_lines) + "\n", encoding="utf-8")
        exported_files.append(xy_path.name)

        if phase.raman_shift_peaks_cm1:
            ram_y = render_reference_raman_spectrum(phase.phase_id, raman_grid)
            jdx_lines = [
                f"##TITLE={phase.phase_id}",
                "##JCAMP-DX=5.01",
                "##DATA TYPE=RAMAN SPECTRUM",
                "##XUNITS=1/CM",
                "##YUNITS=RELATIVE INTENSITY",
                "##XFACTOR=1.0",
                "##YFACTOR=1.0",
                f"##FIRSTX={raman_grid[0]:.2f}",
                f"##LASTX={raman_grid[-1]:.2f}",
                f"##NPOINTS={len(raman_grid)}",
                "##XYPOINTS=(XY..XY)",
                *[f"{raman_grid[i]:.2f}, {ram_y[i]:.6f}" for i in range(len(raman_grid))],
                "##END=",
            ]
            jdx_path = panel_dir / f"{phase.phase_id}_raman.jdx"
            jdx_path.write_text("\n".join(jdx_lines) + "\n", encoding="utf-8")
            exported_files.append(jdx_path.name)

    bench = run_offline_reference_panel_benchmark()
    summary = {
        "panel_directory": str(panel_dir),
        "exported_file_count": len(exported_files),
        "exported_files": exported_files,
        "offline_benchmark": bench,
    }
    (panel_dir / "panel_manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect, verify, or download open spectroscopy & synthesis datasets for lab-attrib."
    )
    parser.add_argument("--list", action="store_true", help="List registered open datasets")
    parser.add_argument(
        "--modality",
        choices=list(VALID_DATASET_MODALITIES),
        default=None,
        help="Filter datasets by analytical modality",
    )
    parser.add_argument(
        "--commercial-only",
        action="store_true",
        help="Only show or fetch datasets with commercial_use_ok=True",
    )
    parser.add_argument(
        "--verify-urls",
        action="store_true",
        help="Perform live HTTP verification of dataset and sample URLs",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Fetch specific dataset_id(s) into gitignored data/open_datasets/",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Fetch sample slices for all matching registered datasets",
    )
    parser.add_argument(
        "--offline-panel",
        action="store_true",
        help="Export bundled zero-network XRD + Raman mineral reference panel into data/",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=2_000_000,
        help="Max bytes per downloaded file (0 for unlimited full archive download)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Destination directory (defaults to gitignored data/open_datasets/)",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    selected = list_open_datasets(
        modality=args.modality,
        commercial_only=args.commercial_only,
    )

    if args.verify_urls:
        checks = []
        for ds in selected:
            primary_res = verify_dataset_url(ds.url)
            sample_res = (
                verify_dataset_url(ds.sample_download_url)
                if ds.sample_download_url and ds.sample_download_url != ds.url
                else primary_res
            )
            checks.append(
                {
                    "dataset_id": ds.dataset_id,
                    "modality": ds.modality,
                    "primary": primary_res,
                    "sample": sample_res,
                }
            )
        all_ok = all(c["primary"]["ok"] and c["sample"]["ok"] for c in checks)
        payload = {"all_urls_reachable": all_ok, "results": checks}
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            for c in checks:
                print(
                    f"{c['dataset_id']:24s} | {c['modality']:10s} | "
                    f"url_ok={c['primary']['ok']} | sample_ok={c['sample']['ok']}"
                )
        return 0 if all_ok else 1

    if args.offline_panel:
        summary = export_offline_reference_panel(args.output_dir)
        if args.json:
            print(json.dumps(summary, indent=2))
        else:
            print(
                f"Exported {summary['exported_file_count']} reference XRD/Raman files to "
                f"{summary['panel_directory']}"
            )
        return 0

    if args.dataset or args.all:
        to_fetch = (
            selected
            if args.all
            else [get_open_dataset(ds_id) for ds_id in args.dataset]
        )
        if args.commercial_only:
            to_fetch = [ds for ds in to_fetch if ds.commercial_use_ok]
        manifests = [
            fetch_dataset_sample(ds, args.output_dir, max_bytes=args.max_bytes)
            for ds in to_fetch
        ]
        if args.json:
            print(json.dumps({"fetched": manifests}, indent=2))
        else:
            for m in manifests:
                print(
                    f"Fetched {m['dataset_id']} -> {m['saved_file']} "
                    f"({m['bytes_written']} B, sha256={m['sha256'][:12]}...)"
                )
        return 0

    # Default or --list: display dataset registry
    payload = {
        "n_datasets": len(selected),
        "n_bundled_reference_phases": len(REFERENCE_MINERAL_PHASES),
        "datasets": [ds.to_dict() for ds in selected],
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        for ds in selected:
            comm = "commercial=OK" if ds.commercial_use_ok else "commercial=NO"
            print(
                f"{ds.dataset_id:24s} | {ds.modality:10s} | {ds.spdx_license:10s} | "
                f"{comm:13s} | n={ds.n_records:7d} | {ds.url}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
