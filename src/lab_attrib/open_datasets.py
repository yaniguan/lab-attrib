"""Canonical Open-Source Characterization & Synthesis Dataset Registry and Multi-Modal Parsers.

Grounds `lab-attrib`'s post-execution NNLS spectral deconvolver (`xrd_deconvolve`), Bayesian EVOI
orthogonal assay router (`orthogonal_router`), 4-way causal blame gate (`blame`), and closed-loop
`Chem-RRT` frontier campaign (`attributed_frontier`) in verified public datasets across five
analytical modalities (`XRD`, `Raman`, `LC-MS`, `1H-NMR`, `Solubility`):

1. `rruff_powder_xrd` (`XRD`, `Apache-2.0`): RRUFF mineral powder XRD XY & DIF reference patterns.
2. `rruff_raman_zenodo` (`Raman`, `CC-BY-4.0`): RRUFF/WURM/SLOPP-E mineral Raman shift spectra.
3. `cod_crystallography` (`XRD`, `CC0-1.0`): Crystallography Open Database (`530,000+` CIF entries).
4. `mp20_crystal_phases` (`XRD`, `CC-BY-4.0`): Materials Project `MP-20` inorganic crystal phases.
5. `alab_gpss_solid_state` (`XRD`, `CC-BY-4.0`): Ceder Group `A-Lab GPSS` solid-state XRD release.
6. `massspecgym_msms` (`LC-MS`, `MIT`): NeurIPS 2024 `MassSpecGym` (`231,104` MS/MS spectra).
7. `nmrshiftdb2_nmr` (`1H-NMR`, `OPL-1.0`): `NMRShiftDB2` open 1H/13C NMR spectral archive.
8. `bmrb_metabolomics_nmr` (`1H-NMR`, `CC0-1.0`): BMRB public-domain small-molecule 1H/13C NMR.
9. `aqsoldb_solubility` (`Solubility`, `CC-BY-4.0`): `AqSolDB` (`9,982` aqueous logS compounds).

Also bundles a compact CC0/permissive curated reference mini-panel of real oxide, halide, and
carbonate mineral Bragg 2-theta peak tables (`Cu K-alpha`, `1.5406 A`) paired with Raman shift
peak tables (`cm^-1`) for zero-network offline NNLS deconvolution and orthogonal confirmation.
"""

from __future__ import annotations

import bisect
import csv
import io
import json
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

from lab_attrib.attributed_frontier import MoleculeCandidate
from lab_attrib.blame import (
    MAX_PRECURSOR_OVERLAP_PERMILLE,
    MAX_SPECTRAL_RESIDUAL_PERMILLE,
    MIN_RESOLVABLE_SNR_TENTHS,
    AssayTelemetry,
    OrthogonalAssayReadout,
    OrthogonalModality,
    diagnose_assay,
)
from lab_attrib.ledger import CertifiedLedger
from lab_attrib.orthogonal_router import OrthogonalRoutingDecision, route_orthogonal_assay
from lab_attrib.xrd_deconvolve import (
    DeconvolutionResult,
    build_gaussian_spectrum,
    deconvolve_spectrum,
)

DatasetModality = Literal["XRD", "Raman", "LC-MS", "1H-NMR", "Solubility"]

VALID_DATASET_MODALITIES: tuple[DatasetModality, ...] = (
    "XRD",
    "Raman",
    "LC-MS",
    "1H-NMR",
    "Solubility",
)

_FWHM_TO_SIGMA: float = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))


def _require_finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric, got bool {value!r}")
    val = float(value)
    if not math.isfinite(val):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return val


def _is_probable_path(source: str | Path) -> Path | None:
    if isinstance(source, Path):
        return source
    s_head = source.lstrip()
    if (
        "\n" in source
        or "\r" in source
        or len(source) > 1024
        or s_head.startswith(("[", "{", "<", "##", "BEGIN IONS"))
    ):
        return None
    candidate = Path(source)
    try:
        if candidate.is_file():
            return candidate
    except OSError:
        return None
    return candidate if "\t" not in source else None


def _read_text_or_path(source: str | Path, default_name: str) -> tuple[str, str]:
    if isinstance(source, Path):
        return source.read_text(encoding="utf-8"), source.stem
    s_head = source.lstrip()
    if (
        "\n" in source
        or "\r" in source
        or s_head.startswith(("[", "{", "<", "##", "BEGIN IONS"))
    ):
        return source, default_name
    candidate = Path(source)
    try:
        if candidate.exists():
            return candidate.read_text(encoding="utf-8"), candidate.stem
    except OSError:
        pass
    raise FileNotFoundError(f"Input spectrum path does not exist: {source}")


def _linear_resample(
    xs: Sequence[float],
    ys: Sequence[float],
    target_grid: Sequence[float],
) -> list[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("Source spectrum must have matching x/y sequences of length >= 2")
    if len(target_grid) < 2:
        raise ValueError("target_grid must contain at least 2 points")

    out: list[float] = []
    x0 = xs[0]
    x_end = xs[-1]
    n = len(xs)
    prev_q: float | None = None
    for idx, q_raw in enumerate(target_grid):
        q = _require_finite(q_raw, f"target_grid[{idx}]")
        if prev_q is not None and q <= prev_q:
            raise ValueError("target_grid must be strictly increasing")
        prev_q = q
        if q < x0 or q > x_end:
            out.append(0.0)
            continue
        hi = bisect.bisect_right(xs, q)
        if hi <= 0:
            out.append(max(0.0, float(ys[0])))
        elif hi >= n:
            out.append(max(0.0, float(ys[-1])))
        else:
            x1, x2 = xs[hi - 1], xs[hi]
            y1, y2 = ys[hi - 1], ys[hi]
            dx = x2 - x1
            t = (q - x1) / dx if dx > 1e-15 else 0.0
            out.append(max(0.0, float(y1 + t * (y2 - y1))))
    return out


# ---------------------------------------------------------------------------
# 1. Typed Open Characterization & Synthesis Dataset Registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OpenCharacterizationDataset:
    """Metadata and license contract for an open spectroscopy or synthesis dataset."""

    dataset_id: str
    modality: DatasetModality
    url: str
    spdx_license: str
    commercial_use_ok: bool
    n_records: int
    title: str = ""
    citation: str = ""
    sample_download_url: str | None = None
    mirror_url: str | None = None
    paired_modalities: tuple[str, ...] = ()
    supported_formats: tuple[str, ...] = ()
    attrib_integration: str = ""

    def __post_init__(self) -> None:
        if not self.dataset_id or not self.dataset_id.strip():
            raise ValueError("dataset_id must be non-empty")
        if self.modality not in VALID_DATASET_MODALITIES:
            raise ValueError(
                f"modality must be one of {VALID_DATASET_MODALITIES}, got {self.modality!r}"
            )
        if not self.url.startswith("https://"):
            raise ValueError(f"url must be an https:// URL, got {self.url!r}")
        if not self.spdx_license or not self.spdx_license.strip():
            raise ValueError("spdx_license must be non-empty")
        if not isinstance(self.commercial_use_ok, bool):
            raise ValueError("commercial_use_ok must be a bool")
        if isinstance(self.n_records, bool) or not isinstance(self.n_records, int) or self.n_records < 1:
            raise ValueError(f"n_records must be a positive integer, got {self.n_records!r}")
        if self.sample_download_url is not None and not self.sample_download_url.startswith("https://"):
            raise ValueError(
                f"sample_download_url must be an https:// URL, got {self.sample_download_url!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "modality": self.modality,
            "url": self.url,
            "spdx_license": self.spdx_license,
            "commercial_use_ok": self.commercial_use_ok,
            "n_records": self.n_records,
            "title": self.title,
            "citation": self.citation,
            "sample_download_url": self.sample_download_url,
            "mirror_url": self.mirror_url,
            "paired_modalities": list(self.paired_modalities),
            "supported_formats": list(self.supported_formats),
            "attrib_integration": self.attrib_integration,
        }


OPEN_CHARACTERIZATION_DATASETS: tuple[OpenCharacterizationDataset, ...] = (
    OpenCharacterizationDataset(
        dataset_id="rruff_powder_xrd",
        modality="XRD",
        url="https://huggingface.co/datasets/AI4Spectro/rruff",
        spdx_license="Apache-2.0",
        commercial_use_ok=True,
        n_records=5842,
        title="RRUFF Mineral Powder XRD XY & Crystal Structure Archive (AI4Spectro Open Mirror)",
        citation="Lafuente, B. et al. The power of databases: The RRUFF project (2015).",
        sample_download_url=(
            "https://huggingface.co/datasets/AI4Spectro/rruff/raw/main/"
            "Calcite__R050048-1__Powder__Xray_Data_XY_RAW__541.txt"
        ),
        paired_modalities=("Raman",),
        supported_formats=("XY", "RRUFF-TXT", "DIF"),
        attrib_integration="load_xy_diffractogram -> deconvolve_spectrum -> diagnose_assay",
    ),
    OpenCharacterizationDataset(
        dataset_id="rruff_raman_zenodo",
        modality="Raman",
        url="https://zenodo.org/records/15742626",
        spdx_license="CC-BY-4.0",
        commercial_use_ok=True,
        n_records=21490,
        title="RRUFF / WURM / SLOPP-E Open Mineral & Polymer Raman Spectral Archive",
        citation="Lafuente, B. et al. RRUFF Project (2015); Zenodo Record 15742626 (2025).",
        sample_download_url="https://zenodo.org/api/records/15742626",
        paired_modalities=("XRD",),
        supported_formats=("JCAMP-DX", "RRUFF-TXT", "CSV"),
        attrib_integration="load_raman_spectrum -> route_orthogonal_assay -> resolve_with_orthogonal_assay",
    ),
    OpenCharacterizationDataset(
        dataset_id="cod_crystallography",
        modality="XRD",
        url="https://www.crystallography.net/cod/",
        spdx_license="CC0-1.0",
        commercial_use_ok=True,
        n_records=530000,
        title="Crystallography Open Database (COD) Public-Domain CIF & Diffraction Library",
        citation="Grazulis, S. et al. Crystallography Open Database (COD), Nucleic Acids Res. 40, D420-D427 (2012).",
        sample_download_url="https://www.crystallography.net/cod/9009666.cif",
        supported_formats=("CIF", "XY"),
        attrib_integration="Reference precursor/target phase basis for xrd_deconvolve",
    ),
    OpenCharacterizationDataset(
        dataset_id="mp20_crystal_phases",
        modality="XRD",
        url="https://github.com/txie-93/cdvae/tree/main/data/mp_20",
        spdx_license="CC-BY-4.0",
        commercial_use_ok=True,
        n_records=45231,
        title="Materials Project MP-20 Inorganic Crystal Phase Benchmark (<=20 atoms/cell)",
        citation="Jain, A. et al. APL Mater. 1, 011002 (2013); Xie, T. et al. CDVAE, ICLR (2022).",
        sample_download_url="https://raw.githubusercontent.com/txie-93/cdvae/main/data/mp_20/val.csv",
        mirror_url="https://huggingface.co/datasets/materials-toolkits/mp-20",
        supported_formats=("CSV", "CIF"),
        attrib_integration="Competing thermodynamic polymorph & precursor screening in xrd_deconvolve",
    ),
    OpenCharacterizationDataset(
        dataset_id="alab_gpss_solid_state",
        modality="XRD",
        url="https://github.com/CederGroupHub/alab_gpss_public",
        spdx_license="CC-BY-4.0",
        commercial_use_ok=True,
        n_records=352,
        title="Ceder Group A-Lab GPSS Autonomous Solid-State Synthesis & XRD Release",
        citation=(
            "Szymanski, N. J. et al. Nature 624, 86-91 (2023); "
            "CederGroupHub alab_gpss_public & Zenodo DOI 10.5281/zenodo.19396297 (2026)."
        ),
        sample_download_url=(
            "https://raw.githubusercontent.com/CederGroupHub/alab_gpss_public/main/data/dataset.json"
        ),
        mirror_url="https://zenodo.org/records/19396297",
        supported_formats=("XRDML", "JSON"),
        attrib_integration="load_xy_diffractogram (.xrdml) & load_alab_gpss_summary -> blame & EVOI router",
    ),
    OpenCharacterizationDataset(
        dataset_id="massspecgym_msms",
        modality="LC-MS",
        url="https://huggingface.co/datasets/roman-bushuiev/MassSpecGym",
        spdx_license="MIT",
        commercial_use_ok=True,
        n_records=231104,
        title="MassSpecGym Tandem MS/MS De Novo & Retrieval Benchmark (NeurIPS 2024 Spotlight)",
        citation="Bushuiev, R. et al. MassSpecGym: A benchmark for the discovery and identification of molecules, NeurIPS (2024).",
        sample_download_url=(
            "https://huggingface.co/datasets/roman-bushuiev/MassSpecGym/resolve/main/data/MassSpecGym.tsv"
        ),
        supported_formats=("MGF", "TSV", "JSON"),
        attrib_integration="load_massspecgym_spectra -> evaluate_msms_orthogonal_readout (LC-MS)",
    ),
    OpenCharacterizationDataset(
        dataset_id="nmrshiftdb2_nmr",
        modality="1H-NMR",
        url="https://nmrshiftdb.nmr.uni-koeln.de/",
        spdx_license="OPL-1.0",
        commercial_use_ok=False,
        n_records=70029,
        title="NMRShiftDB2 Open 1H / 13C / Heteronuclear NMR Spectral Database",
        citation="Kuhn, S. & Schlorer, N. E. Magn. Reson. Chem. 53, 582-589 (2015).",
        sample_download_url="https://huggingface.co/api/datasets/structure-epflai/nmrshiftdb2",
        mirror_url="https://sourceforge.net/projects/nmrshiftdb2/files/data/",
        supported_formats=("JCAMP-DX", "SDF", "Parquet"),
        attrib_integration="route_orthogonal_assay (1H-NMR) open-content reference shifts",
    ),
    OpenCharacterizationDataset(
        dataset_id="bmrb_metabolomics_nmr",
        modality="1H-NMR",
        url="https://bmrb.io/",
        spdx_license="CC0-1.0",
        commercial_use_ok=True,
        n_records=5200,
        title="Biological Magnetic Resonance Data Bank (BMRB) Public-Domain Small-Molecule NMR",
        citation="Hoch, J. C. et al. Biological Magnetic Resonance Data Bank, Nucleic Acids Res. 51, D368-D376 (2023).",
        sample_download_url="https://bmrb.io/",
        mirror_url="https://huggingface.co/datasets/mahynski/bmrb-hsqc-nmr-1H13C",
        supported_formats=("NMR-STAR", "CSV"),
        attrib_integration="Commercial-safe CC0-1.0 1H-NMR secondary confirmation reference",
    ),
    OpenCharacterizationDataset(
        dataset_id="aqsoldb_solubility",
        modality="Solubility",
        url="https://huggingface.co/datasets/maomlab/AqSolDB",
        spdx_license="CC-BY-4.0",
        commercial_use_ok=True,
        n_records=9982,
        title="AqSolDB Curated Reference Set of Aqueous Solubility (logS) & 2D Descriptors",
        citation="Sorkun, M. C., Khetan, A. & Er, S. AqSolDB, Scientific Data 6, 143 (2019); DOI 10.7910/DVN/OVHAW8.",
        sample_download_url="https://dataverse.harvard.edu/api/access/datafile/3407241",
        mirror_url="https://huggingface.co/datasets/maomlab/AqSolDB/resolve/main/AqSolDB/test.csv",
        supported_formats=("CSV", "TSV"),
        attrib_integration="load_aqsoldb_csv -> MoleculeCandidate -> run_attributed_frontier_benchmark",
    ),
)

_DATASETS_BY_ID: dict[str, OpenCharacterizationDataset] = {
    ds.dataset_id: ds for ds in OPEN_CHARACTERIZATION_DATASETS
}


def list_open_datasets(
    *,
    modality: str | None = None,
    commercial_only: bool = False,
) -> list[OpenCharacterizationDataset]:
    """Return registered open characterization datasets filtered by modality and license policy."""
    if modality is not None and modality not in VALID_DATASET_MODALITIES:
        raise ValueError(
            f"Unknown modality {modality!r}; expected one of {VALID_DATASET_MODALITIES}"
        )
    out: list[OpenCharacterizationDataset] = []
    for ds in OPEN_CHARACTERIZATION_DATASETS:
        if modality is not None and ds.modality != modality:
            continue
        if commercial_only and not ds.commercial_use_ok:
            continue
        out.append(ds)
    return out


def get_open_dataset(dataset_id: str) -> OpenCharacterizationDataset:
    """Look up a single registered `OpenCharacterizationDataset` by `dataset_id`."""
    try:
        return _DATASETS_BY_ID[dataset_id]
    except KeyError as exc:
        known = ", ".join(sorted(_DATASETS_BY_ID))
        raise KeyError(f"Unknown dataset_id {dataset_id!r}. Known datasets: {known}") from exc


# ---------------------------------------------------------------------------
# 2. Bundled CC0 / Permissive Reference Oxide & Mineral XRD + Raman Mini-Panel
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReferenceMineralPhase:
    """Curated crystallographic Cu K-alpha Bragg 2-theta & paired Raman shift peak table."""

    phase_id: str
    formula: str
    mineral_name: str
    space_group: str
    centrosymmetric: bool
    source_dataset_id: str
    external_reference_id: str
    xrd_bragg_peaks_cu_ka: tuple[tuple[float, float], ...]
    raman_shift_peaks_cm1: tuple[tuple[float, float], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase_id": self.phase_id,
            "formula": self.formula,
            "mineral_name": self.mineral_name,
            "space_group": self.space_group,
            "centrosymmetric": self.centrosymmetric,
            "source_dataset_id": self.source_dataset_id,
            "external_reference_id": self.external_reference_id,
            "xrd_bragg_peaks_cu_ka": [
                {"two_theta_deg": t, "rel_intensity": h} for t, h in self.xrd_bragg_peaks_cu_ka
            ],
            "raman_shift_peaks_cm1": [
                {"wavenumber_cm1": w, "rel_intensity": h} for w, h in self.raman_shift_peaks_cm1
            ],
        }


REFERENCE_MINERAL_PHASES: tuple[ReferenceMineralPhase, ...] = (
    ReferenceMineralPhase(
        phase_id="LiCl_rocksalt",
        formula="LiCl",
        mineral_name="Halite-type Lithium Chloride",
        space_group="Fm-3m",
        centrosymmetric=True,
        source_dataset_id="alab_gpss_solid_state",
        external_reference_id="COD-9008674",
        xrd_bragg_peaks_cu_ka=(
            (30.09, 1.00),
            (34.89, 0.82),
            (50.17, 0.52),
            (59.61, 0.28),
            (62.55, 0.16),
        ),
        raman_shift_peaks_cm1=(),  # First-order Raman silent (centrosymmetric Fm-3m rocksalt)
    ),
    ReferenceMineralPhase(
        phase_id="MgCl2_chloromagnesite",
        formula="MgCl2",
        mineral_name="Chloromagnesite",
        space_group="R-3m",
        centrosymmetric=True,
        source_dataset_id="alab_gpss_solid_state",
        external_reference_id="COD-9009129",
        xrd_bragg_peaks_cu_ka=(
            (15.05, 0.72),
            (30.34, 0.25),
            (34.96, 1.00),
            (50.32, 0.44),
            (62.88, 0.30),
        ),
        raman_shift_peaks_cm1=(
            (156.0, 0.34),
            (242.0, 1.00),
        ),
    ),
    ReferenceMineralPhase(
        phase_id="Li2MgCl4_spinel",
        formula="Li2MgCl4",
        mineral_name="Inverse Chloride Spinel",
        space_group="Fd-3m",
        centrosymmetric=True,
        source_dataset_id="alab_gpss_solid_state",
        external_reference_id="ALAB-GPSS-000",
        xrd_bragg_peaks_cu_ka=(
            (14.74, 0.24),
            (29.74, 0.68),
            (34.47, 1.00),
            (42.36, 0.14),
            (49.55, 0.58),
            (58.86, 0.30),
        ),
        raman_shift_peaks_cm1=(
            (132.0, 0.28),
            (254.0, 1.00),
            (336.0, 0.42),
        ),
    ),
    ReferenceMineralPhase(
        phase_id="ZrO2_cubic",
        formula="ZrO2",
        mineral_name="Cubic Fluorite Zirconia",
        space_group="Fm-3m",
        centrosymmetric=False,
        source_dataset_id="cod_crystallography",
        external_reference_id="COD-5000038",
        xrd_bragg_peaks_cu_ka=(
            (30.17, 1.00),
            (34.98, 0.22),
            (50.30, 0.55),
            (59.79, 0.32),
            (62.74, 0.12),
        ),
        raman_shift_peaks_cm1=(
            (615.0, 1.00),
        ),
    ),
    ReferenceMineralPhase(
        phase_id="Y2O3_bixbyite",
        formula="Y2O3",
        mineral_name="Yttria (Bixbyite-type)",
        space_group="Ia-3",
        centrosymmetric=False,
        source_dataset_id="cod_crystallography",
        external_reference_id="COD-1513303",
        xrd_bragg_peaks_cu_ka=(
            (20.50, 0.14),
            (29.15, 1.00),
            (33.79, 0.26),
            (48.54, 0.44),
            (57.63, 0.31),
        ),
        raman_shift_peaks_cm1=(
            (330.0, 0.12),
            (377.0, 1.00),
            (469.0, 0.15),
            (592.0, 0.09),
        ),
    ),
    ReferenceMineralPhase(
        phase_id="ZrO2_tetragonal",
        formula="ZrO2",
        mineral_name="Tetragonal Zirconia",
        space_group="P42/nmc",
        centrosymmetric=False,
        source_dataset_id="cod_crystallography",
        external_reference_id="COD-1525706",
        xrd_bragg_peaks_cu_ka=(
            (30.18, 1.00),
            (34.62, 0.15),
            (35.17, 0.12),
            (50.18, 0.50),
            (59.35, 0.16),
            (60.15, 0.22),
        ),
        raman_shift_peaks_cm1=(
            (148.0, 0.55),
            (264.0, 1.00),
            (318.0, 0.35),
            (462.0, 0.50),
            (602.0, 0.22),
            (642.0, 0.85),
        ),
    ),
    ReferenceMineralPhase(
        phase_id="TiO2_anatase",
        formula="TiO2",
        mineral_name="Anatase",
        space_group="I41/amd",
        centrosymmetric=False,
        source_dataset_id="rruff_powder_xrd",
        external_reference_id="RRUFF-R060277",
        xrd_bragg_peaks_cu_ka=(
            (25.28, 1.00),
            (37.80, 0.20),
            (48.05, 0.35),
            (53.89, 0.20),
            (55.06, 0.21),
            (62.69, 0.14),
        ),
        raman_shift_peaks_cm1=(
            (144.0, 1.00),
            (197.0, 0.06),
            (399.0, 0.18),
            (515.0, 0.22),
            (639.0, 0.32),
        ),
    ),
    ReferenceMineralPhase(
        phase_id="BaCO3_witherite",
        formula="BaCO3",
        mineral_name="Witherite",
        space_group="Pmcn",
        centrosymmetric=False,
        source_dataset_id="rruff_powder_xrd",
        external_reference_id="RRUFF-R040040",
        xrd_bragg_peaks_cu_ka=(
            (23.90, 1.00),
            (24.30, 0.48),
            (34.09, 0.24),
            (34.60, 0.30),
            (41.98, 0.26),
            (44.85, 0.22),
        ),
        raman_shift_peaks_cm1=(
            (136.0, 0.35),
            (155.0, 0.42),
            (691.0, 0.24),
            (1060.0, 1.00),
        ),
    ),
    ReferenceMineralPhase(
        phase_id="BaTiO3_tetragonal",
        formula="BaTiO3",
        mineral_name="Barium Titanate (Tetragonal Perovskite)",
        space_group="P4mm",
        centrosymmetric=False,
        source_dataset_id="mp20_crystal_phases",
        external_reference_id="COD-1507756",
        xrd_bragg_peaks_cu_ka=(
            (22.24, 0.22),
            (31.56, 1.00),
            (38.89, 0.28),
            (45.24, 0.42),
            (50.92, 0.16),
            (56.18, 0.38),
        ),
        raman_shift_peaks_cm1=(
            (260.0, 0.65),
            (305.0, 0.85),
            (518.0, 1.00),
            (716.0, 0.45),
        ),
    ),
    ReferenceMineralPhase(
        phase_id="Calcite_R050048",
        formula="CaCO3",
        mineral_name="Calcite",
        space_group="R-3c",
        centrosymmetric=False,
        source_dataset_id="rruff_powder_xrd",
        external_reference_id="RRUFF-R050048",
        xrd_bragg_peaks_cu_ka=(
            (23.05, 0.12),
            (29.40, 1.00),
            (35.97, 0.14),
            (39.41, 0.18),
            (43.16, 0.18),
            (47.50, 0.17),
            (48.51, 0.19),
        ),
        raman_shift_peaks_cm1=(
            (155.0, 0.22),
            (281.0, 0.55),
            (712.0, 0.18),
            (1085.0, 1.00),
        ),
    ),
    ReferenceMineralPhase(
        phase_id="Quartz_alpha_COD9009666",
        formula="SiO2",
        mineral_name="Alpha-Quartz",
        space_group="P3221",
        centrosymmetric=False,
        source_dataset_id="cod_crystallography",
        external_reference_id="COD-9009666",
        xrd_bragg_peaks_cu_ka=(
            (20.86, 0.22),
            (26.64, 1.00),
            (36.54, 0.08),
            (39.46, 0.07),
            (50.14, 0.14),
            (59.96, 0.09),
        ),
        raman_shift_peaks_cm1=(
            (128.0, 0.25),
            (206.0, 0.38),
            (464.0, 1.00),
        ),
    ),
)

_PHASES_BY_ID: dict[str, ReferenceMineralPhase] = {
    phase.phase_id: phase for phase in REFERENCE_MINERAL_PHASES
}


def get_reference_mineral_phase(phase_id: str) -> ReferenceMineralPhase:
    """Return a bundled `ReferenceMineralPhase` by `phase_id`."""
    try:
        return _PHASES_BY_ID[phase_id]
    except KeyError as exc:
        known = ", ".join(sorted(_PHASES_BY_ID))
        raise KeyError(f"Unknown phase_id {phase_id!r}. Known phases: {known}") from exc


def render_reference_xrd_spectrum(
    phase_id: str,
    two_theta_grid: Sequence[float],
    *,
    fwhm_deg: float = 0.55,
) -> list[float]:
    """Synthesize a continuous powder XRD profile from curated Cu K-alpha Bragg peaks."""
    if fwhm_deg <= 0.0 or not math.isfinite(fwhm_deg):
        raise ValueError(f"fwhm_deg must be a positive finite float, got {fwhm_deg}")
    phase = get_reference_mineral_phase(phase_id)
    sigma = fwhm_deg * _FWHM_TO_SIGMA
    peaks = [(center, height, sigma) for center, height in phase.xrd_bragg_peaks_cu_ka]
    return build_gaussian_spectrum(two_theta_grid, peaks)


def render_reference_raman_spectrum(
    phase_id: str,
    wavenumber_grid: Sequence[float],
    *,
    fwhm_cm1: float = 8.0,
) -> list[float]:
    """Synthesize a continuous Raman spectrum (`cm^-1`) from curated vibrational shift peaks."""
    if fwhm_cm1 <= 0.0 or not math.isfinite(fwhm_cm1):
        raise ValueError(f"fwhm_cm1 must be a positive finite float, got {fwhm_cm1}")
    phase = get_reference_mineral_phase(phase_id)
    if not phase.raman_shift_peaks_cm1:
        return [0.0 for _ in wavenumber_grid]
    sigma = fwhm_cm1 * _FWHM_TO_SIGMA
    peaks = [(center, height, sigma) for center, height in phase.raman_shift_peaks_cm1]
    return build_gaussian_spectrum(wavenumber_grid, peaks)


# ---------------------------------------------------------------------------
# 3. Multi-Modal Open Spectrum & Solubility Parsers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiffractogramRecord:
    """Parsed 1D powder X-ray diffractogram (`2theta` in degrees vs non-negative intensity)."""

    sample_id: str
    two_theta_deg: tuple[float, ...]
    intensities: tuple[float, ...]
    uncertainties: tuple[float, ...] | None
    metadata: dict[str, str]

    def resample(self, target_grid: Sequence[float]) -> list[float]:
        """Interpolate diffractogram intensities onto `target_grid` (`0.0` outside scan bounds)."""
        return _linear_resample(self.two_theta_deg, self.intensities, target_grid)


def _parse_xrdml_diffractogram(text: str, sample_id: str) -> DiffractogramRecord:
    """Parse a PANalytical `.xrdml` powder diffractogram (`CederGroupHub/alab_gpss_public`)."""
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise ValueError(f"Invalid XRDML XML document: {exc}") from exc

    def _strip_ns(tag: str) -> str:
        return tag.rsplit("}", 1)[-1] if "}" in tag else tag

    meta: dict[str, str] = {}
    start_pos: float | None = None
    end_pos: float | None = None
    raw_intensities: list[float] = []

    for elem in root.iter():
        tag = _strip_ns(elem.tag)
        if tag == "id" and elem.text and elem.text.strip() and "SAMPLE_ID" not in meta:
            meta["SAMPLE_ID"] = elem.text.strip()
        elif tag == "kAlpha1" and elem.text and elem.text.strip():
            meta["WAVELENGTH_KA1_A"] = elem.text.strip()
        elif tag == "positions" and elem.attrib.get("axis", "").lower() in ("2theta", "twotheta"):
            for child in elem:
                ctag = _strip_ns(child.tag)
                if ctag == "startPosition" and child.text:
                    start_pos = _require_finite(child.text.strip(), "startPosition")
                elif ctag == "endPosition" and child.text:
                    end_pos = _require_finite(child.text.strip(), "endPosition")
        elif tag in ("intensities", "counts") and elem.text and elem.text.strip() and not raw_intensities:
            raw_intensities = [
                max(0.0, _require_finite(tok, "xrdml_intensity"))
                for tok in elem.text.split()
            ]

    if start_pos is None or end_pos is None or len(raw_intensities) < 2:
        raise ValueError("XRDML file missing 2Theta start/end positions or >=2 intensity channels")
    if end_pos <= start_pos:
        raise ValueError(f"XRDML endPosition ({end_pos}) must exceed startPosition ({start_pos})")

    m = len(raw_intensities)
    step = (end_pos - start_pos) / float(m - 1)
    angles = tuple(round(start_pos + i * step, 8) for i in range(m))
    sid = meta.get("SAMPLE_ID", sample_id)
    return DiffractogramRecord(
        sample_id=sid,
        two_theta_deg=angles,
        intensities=tuple(raw_intensities),
        uncertainties=None,
        metadata=meta,
    )


def load_xy_diffractogram(
    source: str | Path,
    *,
    min_two_theta: float | None = None,
    max_two_theta: float | None = None,
    resample_grid: Sequence[float] | None = None,
) -> DiffractogramRecord:
    """Parse an XY / XYE / CSV / RRUFF `##KEY=VALUE` / PANalytical `.xrdml` powder diffractogram."""
    text, default_id = _read_text_or_path(source, "xy_diffractogram")
    stripped_head = text.lstrip()
    if stripped_head.startswith("<?xml") or "<xrdMeasurements" in stripped_head[:512]:
        rec = _parse_xrdml_diffractogram(text, default_id)
    else:
        meta: dict[str, str] = {}
        points: dict[float, tuple[float, float | None]] = {}
        for line in text.splitlines():
            s = line.strip()
            if not s:
                continue
            if s.startswith("##"):
                if s.upper().startswith("##END"):
                    break
                if "=" in s:
                    k, v = s[2:].split("=", 1)
                    meta[k.strip().upper()] = v.strip()
                continue
            if s[0] in ("#", "!", ";", "*", "'", "/"):
                continue

            tokens = [tok for tok in re.split(r"[\s,;]+", s) if tok]
            if len(tokens) < 2:
                continue
            try:
                x_val = _require_finite(tokens[0], "two_theta")
                y_val = _require_finite(tokens[1], "intensity")
                err_val = _require_finite(tokens[2], "sigma") if len(tokens) >= 3 else None
            except ValueError:
                # Skip column header rows such as "two_theta,intensity"
                if not points:
                    continue
                raise

            points[x_val] = (max(0.0, y_val), max(0.0, err_val) if err_val is not None else None)

        sorted_angles = sorted(points)
        if len(sorted_angles) < 2:
            raise ValueError("Diffractogram must contain at least 2 valid (2theta, intensity) rows")
        ys = tuple(points[a][0] for a in sorted_angles)
        has_err = all(points[a][1] is not None for a in sorted_angles)
        errs = tuple(float(points[a][1] or 0.0) for a in sorted_angles) if has_err else None
        sid = meta.get("RRUFFID") or meta.get("NAMES") or meta.get("TITLE") or default_id
        rec = DiffractogramRecord(
            sample_id=sid,
            two_theta_deg=tuple(sorted_angles),
            intensities=ys,
            uncertainties=errs,
            metadata=meta,
        )

    if min_two_theta is not None or max_two_theta is not None:
        lo = -float("inf") if min_two_theta is None else float(min_two_theta)
        hi = float("inf") if max_two_theta is None else float(max_two_theta)
        kept_indices = [i for i, a in enumerate(rec.two_theta_deg) if lo <= a <= hi]
        if len(kept_indices) < 2:
            raise ValueError("Filtered [min_two_theta, max_two_theta] window left < 2 channels")
        rec = DiffractogramRecord(
            sample_id=rec.sample_id,
            two_theta_deg=tuple(rec.two_theta_deg[i] for i in kept_indices),
            intensities=tuple(rec.intensities[i] for i in kept_indices),
            uncertainties=(
                tuple(rec.uncertainties[i] for i in kept_indices)
                if rec.uncertainties is not None
                else None
            ),
            metadata=rec.metadata,
        )

    if resample_grid is not None:
        resampled_y = rec.resample(resample_grid)
        rec = DiffractogramRecord(
            sample_id=rec.sample_id,
            two_theta_deg=tuple(float(v) for v in resample_grid),
            intensities=tuple(resampled_y),
            uncertainties=None,
            metadata=rec.metadata,
        )
    return rec


def deconvolve_xy_diffractogram(
    observed: DiffractogramRecord,
    precursors: Sequence[DiffractogramRecord],
    target: DiffractogramRecord,
    *,
    grid: Sequence[float] | None = None,
    **deconv_kwargs: Any,
) -> DeconvolutionResult:
    """Align `DiffractogramRecord` inputs onto a shared 2-theta grid and invoke `deconvolve_spectrum`."""
    common_grid = tuple(float(v) for v in grid) if grid is not None else observed.two_theta_deg
    obs_y = (
        list(observed.intensities)
        if common_grid == observed.two_theta_deg
        else observed.resample(common_grid)
    )
    prec_ys = [
        list(p.intensities) if common_grid == p.two_theta_deg else p.resample(common_grid)
        for p in precursors
    ]
    targ_y = (
        list(target.intensities)
        if common_grid == target.two_theta_deg
        else target.resample(common_grid)
    )
    return deconvolve_spectrum(
        obs_y,
        prec_ys,
        targ_y,
        two_theta_grid=common_grid,
        **deconv_kwargs,
    )


@dataclass(frozen=True)
class RamanSpectrumRecord:
    """Parsed 1D Raman vibrational spectrum (`wavenumber_cm1` vs non-negative intensity)."""

    sample_id: str
    wavenumber_cm1: tuple[float, ...]
    intensities: tuple[float, ...]
    metadata: dict[str, str]

    def resample(self, target_grid: Sequence[float]) -> list[float]:
        """Interpolate Raman intensities onto `target_grid` (`0.0` outside scan bounds)."""
        return _linear_resample(self.wavenumber_cm1, self.intensities, target_grid)


def load_raman_spectrum(
    source: str | Path,
    *,
    resample_grid: Sequence[float] | None = None,
) -> RamanSpectrumRecord:
    """Parse a JCAMP-DX (`##XYDATA` / `##XYPOINTS` / RRUFF) or CSV/ASCII Raman spectrum."""
    text, default_id = _read_text_or_path(source, "raman_spectrum")
    meta: dict[str, str] = {}
    in_xydata_multi_y = False
    points: dict[float, float] = {}

    x_factor = 1.0
    y_factor = 1.0
    delta_x: float | None = None
    first_x: float | None = None
    last_x: float | None = None
    npoints: int | None = None

    raw_lines = text.splitlines()
    for line in raw_lines:
        s = line.strip()
        if not s:
            continue
        if s.startswith("##"):
            upper = s.upper()
            if upper.startswith("##END"):
                break
            if "=" in s:
                key_raw, val_raw = s[2:].split("=", 1)
                key = key_raw.strip().upper()
                val = val_raw.strip()
                meta[key] = val
                if key == "XFACTOR":
                    x_factor = _require_finite(val.split()[0], "XFACTOR")
                elif key == "YFACTOR":
                    y_factor = _require_finite(val.split()[0], "YFACTOR")
                elif key == "DELTAX":
                    delta_x = _require_finite(val.split()[0], "DELTAX")
                elif key == "FIRSTX":
                    first_x = _require_finite(val.split()[0], "FIRSTX")
                elif key == "LASTX":
                    last_x = _require_finite(val.split()[0], "LASTX")
                elif key == "NPOINTS":
                    npoints = int(float(val.split()[0]))
                elif key == "XYDATA":
                    in_xydata_multi_y = "X++(Y..Y)" in val.upper().replace(" ", "")
                elif key == "XYPOINTS":
                    in_xydata_multi_y = False
            continue

        if s[0] in ("#", "$", ";", "!"):
            continue

        if in_xydata_multi_y:
            if delta_x is None:
                if first_x is not None and last_x is not None and npoints and npoints > 1:
                    delta_x = (last_x - first_x) / float(npoints - 1)
                else:
                    delta_x = 1.0
            # Split uncompressed/AFFN JCAMP-DX line: X0 Y0 Y1 Y2 ... (handling +/- glued numbers)
            normalized = re.sub(r"(?<=[0-9])([+-])", r" \1", s)
            toks = [t for t in re.split(r"[\s,;]+", normalized) if t]
            if len(toks) >= 2:
                x0 = _require_finite(toks[0], "jcamp_x0") * x_factor
                for offset_idx, y_tok in enumerate(toks[1:]):
                    yv = _require_finite(y_tok, "jcamp_y") * y_factor
                    xv = round(x0 + offset_idx * delta_x, 6)
                    points[xv] = max(0.0, yv)
            continue

        # Standard `(x, y)` pairs or JCAMP `##XYPOINTS=(XY..XY)` semicolon/comma/whitespace pairs
        if ";" in s and "," in s:
            pair_chunks = [c.strip() for c in s.split(";") if c.strip()]
            for chunk in pair_chunks:
                sub = [t for t in re.split(r"[\s,()]+", chunk) if t]
                if len(sub) >= 2:
                    xv = _require_finite(sub[0], "wavenumber") * x_factor
                    yv = _require_finite(sub[1], "intensity") * y_factor
                    points[xv] = max(0.0, yv)
            continue

        toks = [t for t in re.split(r"[\s,;]+", s) if t]
        if len(toks) < 2:
            continue
        try:
            xv = _require_finite(toks[0], "wavenumber") * x_factor
            yv = _require_finite(toks[1], "intensity") * y_factor
        except ValueError:
            if not points:
                continue
            raise
        points[xv] = max(0.0, yv)

    sorted_wn = sorted(points)
    if len(sorted_wn) < 2:
        raise ValueError("Raman spectrum must contain at least 2 valid (wavenumber, intensity) points")
    sid = meta.get("RRUFFID") or meta.get("TITLE") or meta.get("NAMES") or default_id
    rec = RamanSpectrumRecord(
        sample_id=sid,
        wavenumber_cm1=tuple(sorted_wn),
        intensities=tuple(points[w] for w in sorted_wn),
        metadata=meta,
    )
    if resample_grid is not None:
        resampled = rec.resample(resample_grid)
        return RamanSpectrumRecord(
            sample_id=rec.sample_id,
            wavenumber_cm1=tuple(float(v) for v in resample_grid),
            intensities=tuple(resampled),
            metadata=rec.metadata,
        )
    return rec


def evaluate_raman_orthogonal_readout(
    observed: RamanSpectrumRecord,
    precursor_raman: Sequence[RamanSpectrumRecord],
    target_raman: RamanSpectrumRecord,
    *,
    grid: Sequence[float] | None = None,
    min_hit_amplitude: float = 0.20,
    noise_sigma: float = 0.03,
) -> tuple[DeconvolutionResult, OrthogonalAssayReadout]:
    """Deconvolve a secondary Raman spectrum and emit a Lean-compatible `OrthogonalAssayReadout`."""
    common_grid = tuple(float(v) for v in grid) if grid is not None else observed.wavenumber_cm1
    obs_y = (
        list(observed.intensities)
        if common_grid == observed.wavenumber_cm1
        else observed.resample(common_grid)
    )
    prec_ys = [
        list(p.intensities) if common_grid == p.wavenumber_cm1 else p.resample(common_grid)
        for p in precursor_raman
    ]
    targ_y = (
        list(target_raman.intensities)
        if common_grid == target_raman.wavenumber_cm1
        else target_raman.resample(common_grid)
    )
    deconv = deconvolve_spectrum(
        obs_y,
        prec_ys,
        targ_y,
        min_hit_amplitude=min_hit_amplitude,
        noise_sigma=noise_sigma,
    )
    selectivity_ok = (
        deconv.converged
        and deconv.precursor_overlap_permille <= MAX_PRECURSOR_OVERLAP_PERMILLE
        and deconv.spectral_residual_permille <= MAX_SPECTRAL_RESIDUAL_PERMILLE
    )
    readout = OrthogonalAssayReadout(
        modality=OrthogonalModality.RAMAN,
        instrument_calibrated=deconv.converged,
        orthogonal_snr_tenths=deconv.snr_tenths if deconv.converged else 0,
        selectivity_clean=selectivity_ok,
        target_confirmed=bool(selectivity_ok and deconv.target_signal_present),
    )
    return deconv, readout


@dataclass(frozen=True)
class MassSpecRecord:
    """Standardized tandem MS/MS spectrum record (`MassSpecGym` MGF / TSV / JSON)."""

    identifier: str
    mzs: tuple[float, ...]
    intensities: tuple[float, ...]
    precursor_mz: float | None = None
    smiles: str = ""
    inchikey: str = ""
    adduct: str = ""
    instrument_type: str = ""
    fold: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "identifier": self.identifier,
            "mzs": list(self.mzs),
            "intensities": list(self.intensities),
            "precursor_mz": self.precursor_mz,
            "smiles": self.smiles,
            "inchikey": self.inchikey,
            "adduct": self.adduct,
            "instrument_type": self.instrument_type,
            "fold": self.fold,
        }


def _parse_float_sequence(raw: Any, label: str) -> tuple[float, ...]:
    if isinstance(raw, (list, tuple)):
        return tuple(_require_finite(v, label) for v in raw)
    s = str(raw).strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    if not s:
        return ()
    return tuple(_require_finite(tok.strip(), label) for tok in s.split(",") if tok.strip())


def _normalize_ms_peaks(
    mzs: Sequence[float],
    intensities: Sequence[float],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if len(mzs) != len(intensities) or len(mzs) == 0:
        raise ValueError("MS/MS spectrum must have matching non-empty mzs and intensities")
    pairs = sorted(
        (float(mz), max(0.0, float(inten)))
        for mz, inten in zip(mzs, intensities)
    )
    max_i = max(inten for _, inten in pairs)
    if max_i <= 0.0:
        raise ValueError("MS/MS spectrum must contain at least one positive-intensity peak")
    return (
        tuple(mz for mz, _ in pairs),
        tuple(inten / max_i for _, inten in pairs),
    )


def _record_from_mapping(row: dict[str, Any], idx: int) -> MassSpecRecord:
    norm = {str(k).strip().lower(): v for k, v in row.items() if k is not None}
    ident = str(
        norm.get("identifier")
        or norm.get("spectrum_id")
        or norm.get("title")
        or f"MSGYM-{idx:06d}"
    ).strip()
    mzs_raw = _parse_float_sequence(norm.get("mzs", ()), "mzs")
    intens_raw = _parse_float_sequence(norm.get("intensities", ()), "intensities")
    mzs, intens = _normalize_ms_peaks(mzs_raw, intens_raw)
    prec_raw = norm.get("precursor_mz") if norm.get("precursor_mz") not in (None, "") else norm.get("parent_mass")
    if prec_raw in (None, ""):
        prec_raw = norm.get("pepmass")
    prec_mz = (
        _require_finite(str(prec_raw).split()[0], "precursor_mz")
        if prec_raw not in (None, "")
        else None
    )
    return MassSpecRecord(
        identifier=ident,
        mzs=mzs,
        intensities=intens,
        precursor_mz=prec_mz,
        smiles=str(norm.get("smiles") or "").strip(),
        inchikey=str(norm.get("inchikey") or "").strip(),
        adduct=str(norm.get("adduct") or "").strip(),
        instrument_type=str(norm.get("instrument_type") or "").strip(),
        fold=str(norm.get("fold") or "").strip(),
    )


def load_massspecgym_spectra(
    source: str | Path,
    *,
    max_records: int | None = None,
) -> list[MassSpecRecord]:
    """Parse `MassSpecGym` tandem MS/MS records from MGF (`BEGIN IONS`), JSON/JSONL, or TSV/CSV."""
    if max_records is not None and (isinstance(max_records, bool) or max_records < 1):
        raise ValueError("max_records must be a positive integer when provided")

    path_obj = _is_probable_path(source)
    if path_obj is not None and path_obj.is_file():
        suffix = path_obj.suffix.lower()
        text = path_obj.read_text(encoding="utf-8")
    else:
        text, _ = _read_text_or_path(source, "massspecgym")
        suffix = ""

    head = text.lstrip()
    records: list[MassSpecRecord] = []

    # 1. MGF (`BEGIN IONS` ... `END IONS`)
    if suffix == ".mgf" or "BEGIN IONS" in head[:512]:
        in_block = False
        meta: dict[str, str] = {}
        block_mzs: list[float] = []
        block_intens: list[float] = []
        for raw_line in text.splitlines():
            s = raw_line.strip()
            if not s or s.startswith("#"):
                continue
            upper = s.upper()
            if upper == "BEGIN IONS":
                in_block = True
                meta = {}
                block_mzs = []
                block_intens = []
                continue
            if upper == "END IONS":
                if in_block and block_mzs:
                    row_map: dict[str, Any] = dict(meta)
                    row_map["mzs"] = block_mzs
                    row_map["intensities"] = block_intens
                    records.append(_record_from_mapping(row_map, len(records)))
                    if max_records is not None and len(records) >= max_records:
                        break
                in_block = False
                continue
            if not in_block:
                continue
            if "=" in s:
                k, v = s.split("=", 1)
                meta[k.strip().lower()] = v.strip()
            else:
                toks = s.split()
                if len(toks) >= 2:
                    block_mzs.append(_require_finite(toks[0], "mgf_mz"))
                    block_intens.append(_require_finite(toks[1], "mgf_intensity"))
        if not records:
            raise ValueError("No valid MS/MS ion blocks found in MGF input")
        return records

    # 2. JSON array / object or JSONL
    if suffix in (".json", ".jsonl") or head.startswith("[") or head.startswith("{"):
        if head.startswith("["):
            parsed = json.loads(text)
            for item in parsed:
                records.append(_record_from_mapping(dict(item), len(records)))
                if max_records is not None and len(records) >= max_records:
                    break
        else:
            try:
                single = json.loads(text)
                items = single.get("spectra", [single]) if isinstance(single, dict) else list(single)
                for item in items:
                    records.append(_record_from_mapping(dict(item), len(records)))
                    if max_records is not None and len(records) >= max_records:
                        break
            except json.JSONDecodeError:
                for line in text.splitlines():
                    if not line.strip():
                        continue
                    records.append(_record_from_mapping(dict(json.loads(line)), len(records)))
                    if max_records is not None and len(records) >= max_records:
                        break
        if not records:
            raise ValueError("No valid MS/MS records found in JSON/JSONL input")
        return records

    # 3. Tabular TSV / CSV (`MassSpecGym.tsv`)
    first_line = head.splitlines()[0] if head else ""
    delimiter = "\t" if "\t" in first_line else ","
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    for row in reader:
        records.append(_record_from_mapping(row, len(records)))
        if max_records is not None and len(records) >= max_records:
            break
    if not records:
        raise ValueError("No valid MS/MS records found in tabular MassSpecGym input")
    return records


def msms_cosine_similarity(
    spec_a: MassSpecRecord,
    spec_b: MassSpecRecord,
    *,
    mz_tolerance_da: float = 0.02,
) -> float:
    """Compute greedy tolerance-matched normalized MS/MS fragment cosine similarity in `[0, 1]`."""
    if mz_tolerance_da <= 0.0 or not math.isfinite(mz_tolerance_da):
        raise ValueError("mz_tolerance_da must be positive and finite")
    norm_a = math.sqrt(sum(v * v for v in spec_a.intensities))
    norm_b = math.sqrt(sum(v * v for v in spec_b.intensities))
    if norm_a <= 1e-15 or norm_b <= 1e-15:
        return 0.0

    candidates: list[tuple[float, int, int]] = []
    for i, mz_a in enumerate(spec_a.mzs):
        lo = bisect.bisect_left(spec_b.mzs, mz_a - mz_tolerance_da)
        hi = bisect.bisect_right(spec_b.mzs, mz_a + mz_tolerance_da)
        for j in range(lo, hi):
            prod = spec_a.intensities[i] * spec_b.intensities[j]
            if prod > 0.0:
                candidates.append((prod, i, j))
    candidates.sort(reverse=True)
    used_a: set[int] = set()
    used_b: set[int] = set()
    dot = 0.0
    for prod, i, j in candidates:
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
            dot += prod
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))


def evaluate_msms_orthogonal_readout(
    observed: MassSpecRecord,
    target_reference: MassSpecRecord,
    *,
    mz_tolerance_da: float = 0.02,
    precursor_tolerance_da: float = 0.05,
    min_confirm_cosine: float = 0.70,
    noise_floor_rel: float = 0.02,
    interferent_reference: MassSpecRecord | None = None,
) -> OrthogonalAssayReadout:
    """Evaluate a `MassSpecGym` LC-MS/MS readout into a Lean-verified `OrthogonalAssayReadout`."""
    cos_target = msms_cosine_similarity(
        observed, target_reference, mz_tolerance_da=mz_tolerance_da
    )
    precursor_match = True
    if observed.precursor_mz is not None and target_reference.precursor_mz is not None:
        precursor_match = (
            abs(observed.precursor_mz - target_reference.precursor_mz) <= precursor_tolerance_da
        )

    selectivity_clean = True
    if interferent_reference is not None:
        cos_interferent = msms_cosine_similarity(
            observed, interferent_reference, mz_tolerance_da=mz_tolerance_da
        )
        # Flag co-eluting isobaric chimera when interferent dominates and shadows target
        if (
            observed.precursor_mz is not None
            and interferent_reference.precursor_mz is not None
            and abs(observed.precursor_mz - interferent_reference.precursor_mz) <= precursor_tolerance_da
            and cos_interferent > 0.55
            and cos_target > 0.45
        ):
            selectivity_clean = False

    signal_rms = math.sqrt(sum(v * v for v in observed.intensities) / max(1, len(observed.intensities)))
    snr_tenths = max(0, int(round(10.0 * (signal_rms / max(1e-6, noise_floor_rel)))))
    confirmed = bool(
        selectivity_clean
        and precursor_match
        and cos_target >= min_confirm_cosine
        and snr_tenths >= MIN_RESOLVABLE_SNR_TENTHS
    )
    return OrthogonalAssayReadout(
        modality=OrthogonalModality.LC_MS,
        instrument_calibrated=True,
        orthogonal_snr_tenths=snr_tenths,
        selectivity_clean=selectivity_clean,
        target_confirmed=confirmed,
    )


def _estimate_min_degree_from_smiles(smiles: str, rings: int, hbd: float, rot: float) -> float:
    """Infer terminal heavy-atom degree (`1.0` vs `2.0`) when `Minimum Degree` column is absent."""
    if rings == 0 or hbd >= 1.0 or rot >= 1.0:
        return 1.0
    # Terminal substituent tokens (=O, #N, -F, -Cl, -Br, -I, -O, -N, methyl C outside ring closures)
    if any(tok in smiles for tok in ("=O", "#N", "F", "Cl", "Br", "I", "O", "[O-]", "C(", ")C")):
        return 1.0
    return 2.0


def load_aqsoldb_csv(
    source: str | Path,
    *,
    max_molecules: int | None = None,
) -> list[MoleculeCandidate]:
    """Load `AqSolDB` solubility CSV/TSV (`Solubility` or `Y` column) into `MoleculeCandidate` records."""
    text, _ = _read_text_or_path(source, "aqsoldb")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        raise ValueError("AqSolDB file must contain a header and at least 1 compound row")

    delimiter = "\t" if "\t" in lines[0] else ","
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    molecules: list[MoleculeCandidate] = []

    for idx, row in enumerate(reader):
        norm = {str(k).strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items() if k}
        smiles = str(norm.get("SMILES") or norm.get("smiles") or "").strip()
        cid_base = str(
            norm.get("ID") or norm.get("Compound ID") or norm.get("Name") or f"AQSOL-{idx:05d}"
        ).strip()

        raw_logs = (
            norm.get("Solubility")
            if norm.get("Solubility") not in (None, "")
            else (
                norm.get("Y")
                if norm.get("Y") not in (None, "")
                else norm.get("measured log solubility in mols per litre")
            )
        )
        raw_mw = norm.get("MolWt") if norm.get("MolWt") not in (None, "") else norm.get("Molecular Weight")
        raw_hbd = (
            norm.get("NumHDonors")
            if norm.get("NumHDonors") not in (None, "")
            else norm.get("Number of H-Bond Donors")
        )
        raw_rings = (
            norm.get("RingCount")
            if norm.get("RingCount") not in (None, "")
            else norm.get("Number of Rings")
        )
        raw_rot = (
            norm.get("NumRotatableBonds")
            if norm.get("NumRotatableBonds") not in (None, "")
            else norm.get("Number of Rotatable Bonds")
        )
        raw_psa = (
            norm.get("TPSA")
            if norm.get("TPSA") not in (None, "")
            else norm.get("Polar Surface Area")
        )

        if any(v in (None, "") for v in (raw_logs, raw_mw, raw_hbd, raw_rings, raw_rot, raw_psa)):
            continue
        try:
            log_s = _require_finite(raw_logs, "logS")
            mw = _require_finite(raw_mw, "MolWt")
            hbd = _require_finite(raw_hbd, "NumHDonors")
            rings = int(round(_require_finite(raw_rings, "RingCount")))
            rot = _require_finite(raw_rot, "NumRotatableBonds")
            psa = _require_finite(raw_psa, "TPSA")
        except ValueError:
            continue

        if norm.get("Minimum Degree") not in (None, ""):
            min_deg = _require_finite(norm["Minimum Degree"], "Minimum Degree")
        else:
            min_deg = _estimate_min_degree_from_smiles(smiles, rings, hbd, rot)

        molecules.append(
            MoleculeCandidate(
                compound_id=f"{cid_base}#{idx}",
                smiles=smiles,
                features=(min_deg, mw, hbd, float(rings), rot, psa),
                rings=rings,
                true_log_s=log_s,
            )
        )
        if max_molecules is not None and len(molecules) >= max_molecules:
            break

    if not molecules:
        raise ValueError("No valid solubility rows parsed from AqSolDB input")
    return molecules


def load_alab_gpss_summary(source: str | Path) -> list[dict[str, Any]]:
    """Convert `CederGroupHub/alab_gpss_public` `data/dataset.json` into `lab-attrib` diagnostics."""
    text, _ = _read_text_or_path(source, "alab_gpss_dataset")
    items = json.loads(text)
    if not isinstance(items, list):
        raise ValueError("A-Lab GPSS summary JSON must be a list of experiment objects")

    results: list[dict[str, Any]] = []
    for entry in items:
        idx = int(entry.get("sample_index", len(results)))
        comp = str(entry.get("composition") or f"GPSS-{idx:03d}")
        xrd = dict(entry.get("xrd_analysis_result") or {})
        weights = {str(k): float(v) for k, v in dict(xrd.get("weight_fractions_of_each_phases") or {}).items()}
        rwp_raw = str(xrd.get("Rwp (%)") or "0.0").replace("%", "").strip()
        rwp_pct = _require_finite(rwp_raw, "Rwp (%)") if rwp_raw else 0.0
        rwp_permille = min(1000, max(0, int(round(10.0 * rwp_pct))))

        target_frac = sum(w for name, w in weights.items() if "spinel" in name.lower())
        precursor_frac = sum(w for name, w in weights.items() if "spinel" not in name.lower())
        denom = max(1e-9, target_frac + precursor_frac)
        overlap_permille = min(1000, max(0, int(round(1000.0 * (precursor_frac / denom)))))

        telemetry = AssayTelemetry(
            dispense_error_permille=15,
            aspiration_clean=True,
            incubation_window_clean=True,
            seal_intact=True,
            positive_control_passed=True,
            blank_uncontaminated=True,
            detector_calibrated=True,
            precursor_overlap_permille=overlap_permille,
            spectral_residual_permille=rwp_permille,
            snr_tenths=85 if (target_frac + precursor_frac) > 0.0 else 15,
            target_signal_present=(target_frac >= 0.50),
        )
        report = diagnose_assay(telemetry)
        routing: OrthogonalRoutingDecision | None = None
        if report.outcome.value == "AMBIGUOUS_CHARACTERIZATION":
            routing = route_orthogonal_assay(
                telemetry,
                prior_hit_prob=0.35,
                available_volume_ul=5.0,
                sample_phase="solid",
                centrosymmetric_crystal=True,
            )
        results.append(
            {
                "sample_index": idx,
                "composition": comp,
                "phase_fractions": weights,
                "rwp_permille": rwp_permille,
                "precursor_overlap_permille": overlap_permille,
                "attribution": report.to_dict(),
                "orthogonal_routing": routing.to_dict() if routing is not None else None,
            }
        )
    return results


# ---------------------------------------------------------------------------
# 4. Zero-Network Offline Reference Mineral/Oxide Deconvolution & Router Suite
# ---------------------------------------------------------------------------


def run_offline_reference_panel_benchmark() -> dict[str, Any]:
    """Run deterministic offline NNLS deconvolution + EVOI routing on curated XRD + Raman phases."""
    xrd_grid = [10.0 + 0.25 * i for i in range(241)]  # 10..70 deg 2-theta
    raman_grid = [100.0 + 2.0 * i for i in range(526)]  # 100..1150 cm^-1

    # Case 1: Halide Spinel Precursor Ghost Trap (LiCl + MgCl2 shadowing Li2MgCl4)
    licl_xrd = render_reference_xrd_spectrum("LiCl_rocksalt", xrd_grid, fwhm_deg=1.05)
    mgcl2_xrd = render_reference_xrd_spectrum("MgCl2_chloromagnesite", xrd_grid, fwhm_deg=1.05)
    spinel_xrd = render_reference_xrd_spectrum("Li2MgCl4_spinel", xrd_grid, fwhm_deg=1.05)
    unreacted_halide_obs = [0.65 * licl_xrd[i] + 0.55 * mgcl2_xrd[i] for i in range(len(xrd_grid))]

    spinel_deconv = deconvolve_spectrum(
        unreacted_halide_obs,
        [licl_xrd, mgcl2_xrd],
        spinel_xrd,
        two_theta_grid=xrd_grid,
        min_hit_amplitude=0.25,
        noise_sigma=0.03,
    )
    spinel_tel = AssayTelemetry(
        dispense_error_permille=12,
        aspiration_clean=True,
        incubation_window_clean=True,
        seal_intact=True,
        positive_control_passed=True,
        blank_uncontaminated=True,
        detector_calibrated=True,
        precursor_overlap_permille=spinel_deconv.precursor_overlap_permille,
        spectral_residual_permille=spinel_deconv.spectral_residual_permille,
        snr_tenths=spinel_deconv.snr_tenths,
        target_signal_present=True,
    )
    spinel_primary = diagnose_assay(spinel_tel)
    spinel_routing = route_orthogonal_assay(
        spinel_tel,
        prior_hit_prob=0.35,
        available_volume_ul=4.0,
        sample_phase="solid",
        discovery_value_usd=750.0,
        centrosymmetric_crystal=True,
    )

    # Case 2: Cubic vs Tetragonal Zirconia XRD Shadowing Disambiguated by Paired Raman Shifts
    zro2_cub_xrd = render_reference_xrd_spectrum("ZrO2_cubic", xrd_grid, fwhm_deg=0.65)
    y2o3_xrd = render_reference_xrd_spectrum("Y2O3_bixbyite", xrd_grid, fwhm_deg=0.65)
    zro2_tet_xrd = render_reference_xrd_spectrum("ZrO2_tetragonal", xrd_grid, fwhm_deg=0.65)

    # 2a: Ghost mixture (cubic ZrO2 + Y2O3 only) -> Raman refutes tetragonal ZrO2 -> CERTIFIED_NEGATIVE
    zro2_ghost_xrd_obs = [1.10 * zro2_cub_xrd[i] + 0.18 * y2o3_xrd[i] for i in range(len(xrd_grid))]
    zro2_ghost_deconv = deconvolve_spectrum(
        zro2_ghost_xrd_obs,
        [zro2_cub_xrd, y2o3_xrd],
        zro2_tet_xrd,
        two_theta_grid=xrd_grid,
        min_hit_amplitude=0.25,
        noise_sigma=0.03,
    )
    zro2_ghost_tel = AssayTelemetry(
        dispense_error_permille=14,
        aspiration_clean=True,
        incubation_window_clean=True,
        seal_intact=True,
        positive_control_passed=True,
        blank_uncontaminated=True,
        detector_calibrated=True,
        precursor_overlap_permille=zro2_ghost_deconv.precursor_overlap_permille,
        spectral_residual_permille=zro2_ghost_deconv.spectral_residual_permille,
        snr_tenths=zro2_ghost_deconv.snr_tenths,
        target_signal_present=True,
    )
    zro2_routing = route_orthogonal_assay(
        zro2_ghost_tel,
        prior_hit_prob=0.30,
        available_volume_ul=4.0,
        sample_phase="solid",
        centrosymmetric_crystal=False,
    )

    cub_raman_rec = RamanSpectrumRecord(
        sample_id="ZrO2_cubic",
        wavenumber_cm1=tuple(raman_grid),
        intensities=tuple(render_reference_raman_spectrum("ZrO2_cubic", raman_grid)),
        metadata={},
    )
    y2o3_raman_rec = RamanSpectrumRecord(
        sample_id="Y2O3_bixbyite",
        wavenumber_cm1=tuple(raman_grid),
        intensities=tuple(render_reference_raman_spectrum("Y2O3_bixbyite", raman_grid)),
        metadata={},
    )
    tet_raman_rec = RamanSpectrumRecord(
        sample_id="ZrO2_tetragonal",
        wavenumber_cm1=tuple(raman_grid),
        intensities=tuple(render_reference_raman_spectrum("ZrO2_tetragonal", raman_grid)),
        metadata={},
    )
    ghost_raman_obs = RamanSpectrumRecord(
        sample_id="zro2_ghost_sample",
        wavenumber_cm1=tuple(raman_grid),
        intensities=tuple(
            1.05 * cub_raman_rec.intensities[i] + 0.22 * y2o3_raman_rec.intensities[i]
            for i in range(len(raman_grid))
        ),
        metadata={},
    )
    _, raman_refute_readout = evaluate_raman_orthogonal_readout(
        ghost_raman_obs,
        [cub_raman_rec, y2o3_raman_rec],
        tet_raman_rec,
    )
    zro2_refuted_report = diagnose_assay(zro2_ghost_tel, raman_refute_readout)

    # 2b: True tetragonal ZrO2 product under XRD cubic precursor overlap -> Raman confirms -> CERTIFIED_POSITIVE
    true_tet_raman_obs = RamanSpectrumRecord(
        sample_id="zro2_true_tetragonal_sample",
        wavenumber_cm1=tuple(raman_grid),
        intensities=tuple(
            0.12 * cub_raman_rec.intensities[i] + 0.95 * tet_raman_rec.intensities[i]
            for i in range(len(raman_grid))
        ),
        metadata={},
    )
    _, raman_confirm_readout = evaluate_raman_orthogonal_readout(
        true_tet_raman_obs,
        [cub_raman_rec, y2o3_raman_rec],
        tet_raman_rec,
    )
    zro2_confirmed_report = diagnose_assay(zro2_ghost_tel, raman_confirm_readout)

    # Case 3: Clean BaCO3 + TiO2 -> BaTiO3 Perovskite Synthesis -> Primary CERTIFIED_POSITIVE
    baco3_xrd = render_reference_xrd_spectrum("BaCO3_witherite", xrd_grid, fwhm_deg=0.55)
    tio2_xrd = render_reference_xrd_spectrum("TiO2_anatase", xrd_grid, fwhm_deg=0.55)
    batio3_xrd = render_reference_xrd_spectrum("BaTiO3_tetragonal", xrd_grid, fwhm_deg=0.55)
    batio3_obs = [
        0.04 * baco3_xrd[i] + 0.03 * tio2_xrd[i] + 1.15 * batio3_xrd[i]
        for i in range(len(xrd_grid))
    ]
    batio3_deconv = deconvolve_spectrum(
        batio3_obs,
        [baco3_xrd, tio2_xrd],
        batio3_xrd,
        two_theta_grid=xrd_grid,
        min_hit_amplitude=0.25,
        noise_sigma=0.03,
    )
    batio3_tel = batio3_deconv.to_assay_telemetry()
    batio3_report = diagnose_assay(batio3_tel)

    ledger = CertifiedLedger()
    ledger.append(
        sample_id="PANEL-ZRO2-GHOST-NEG",
        candidate_features={"t_c": 950.0, "y_mol_frac": 0.08},
        measured_value=0.0,
        telemetry=zro2_ghost_tel,
        orthogonal_readout=raman_refute_readout,
    )
    ledger.append(
        sample_id="PANEL-ZRO2-TET-POS",
        candidate_features={"t_c": 1150.0, "y_mol_frac": 0.03},
        measured_value=0.95,
        telemetry=zro2_ghost_tel,
        orthogonal_readout=raman_confirm_readout,
    )
    ledger.append(
        sample_id="PANEL-BATIO3-POS",
        candidate_features={"t_c": 1100.0, "ba_ti_ratio": 1.0},
        measured_value=1.15,
        telemetry=batio3_tel,
    )
    chain_ok, _ = ledger.verify_chain()

    return {
        "n_registered_datasets": len(OPEN_CHARACTERIZATION_DATASETS),
        "n_reference_mineral_phases": len(REFERENCE_MINERAL_PHASES),
        "halide_spinel_ghost_trap": {
            "naive_cosine": round(spinel_deconv.naive_peak_match_cosine, 4),
            "naive_claims_hit": spinel_deconv.naive_claims_hit,
            "precursor_overlap_permille": spinel_deconv.precursor_overlap_permille,
            "primary_outcome": spinel_primary.outcome.value,
            "centrosymmetric_selected_modality": (
                spinel_routing.selected_modality.value
                if spinel_routing.selected_modality is not None
                else None
            ),
        },
        "zirconia_raman_disambiguation": {
            "xrd_naive_cosine": round(zro2_ghost_deconv.naive_peak_match_cosine, 4),
            "xrd_naive_claims_hit": zro2_ghost_deconv.naive_claims_hit,
            "xrd_precursor_overlap_permille": zro2_ghost_deconv.precursor_overlap_permille,
            "selected_modality": (
                zro2_routing.selected_modality.value
                if zro2_routing.selected_modality is not None
                else None
            ),
            "post_raman_refutation_outcome": zro2_refuted_report.outcome.value,
            "post_raman_confirmation_outcome": zro2_confirmed_report.outcome.value,
        },
        "barium_titanate_clean_hit": {
            "target_amplitude": round(batio3_deconv.target_amplitude, 4),
            "precursor_overlap_permille": batio3_deconv.precursor_overlap_permille,
            "outcome": batio3_report.outcome.value,
            "lean_theorem_tag": batio3_report.lean_theorem_tag,
        },
        "certified_ledger": {
            "entries": len(ledger),
            "chain_valid": chain_ok,
            "head_hash": ledger.head_hash,
        },
    }
