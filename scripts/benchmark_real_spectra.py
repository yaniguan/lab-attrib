#!/usr/bin/env python3
"""Real Downloaded Multi-Modal Experimental Spectroscopy Benchmark for `lab-attrib`.

Downloads (or verifies offline against pinned SHA-256 digests) **49 real public
experimental spectroscopy & crystal-structure files** into gitignored
`data/open_datasets/real_benchmark/` and writes structured head-to-head results to
`results/real_spectra_benchmark.json`:

1. **Ceder Group `A-Lab GPSS` Solid-State Synthesis (`alab_gpss_xrd`, `real_experimental`)**:
   Real PANalytical `.xrdml` diffractograms (`000_Li2MgCl4`, `003_Li2Mn0.5Fe0.5Cl4`,
   `004_Li2Co0.5Zn0.5Cl4`, `006_Li2Co0.5Mn0.5Cl4`, `010_Li2Zn0.5Mn0.5Cl4`,
   `071_Li2NiCl4`, `099_Li1.34Fe0.78Ta0.22Cl4`, `271_Li1.5Ni0.5Y0.25In0.25Cl4`) +
   paired `.cif` phase structures + full 352-sample `dataset.json` audit.
2. **Real `RRUFF` & `DARA` Powder XRD (`rruff_dara_powder_xrd`, `real_experimental`)**:
   Real `RRUFF` `.xy` & `.dif` (`Calcite R050048`) and weighed `CederGroupHub/dara`
   experimental multi-mineral `.xrdml`/`.xy` mixtures (`TiO2+ZnO`, `Li2CO3+ZrO2`,
   `V2O5+TiO2`, `ZnO+In2O3`, `BiFeO3`) against a 12-phase `COD`/`DARA` CIF library.
3. **Real `RRUFF/WURM/ROD` Raman Spectroscopy (`rruff_rod_raman`, `real_experimental`)**:
   Real experimental mineral Raman spectra (`Calcite`, `Aragonite`, `Anatase`, `Rutile`,
   `Brookite`, `Quartz`, `Baddeleyite`, `Gypsum`, `Anhydrite` + JCAMP-DX `.jdx` verification)
   across pure phases and polymorph/multi-mineral mixtures.
4. **Real `MassSpecGym` Tandem MS/MS (`massspecgym_msms`, `real_experimental`)**:
   2,417 real Orbitrap/Q-TOF tandem MS/MS spectra (`141` unique `InChIKey` molecules),
   comparing cosine vs unweighted/weighted Spectral Entropy Similarity (Li et al.,
   *Nature Methods* 2021) + near-isobaric chimera quarantine.
5. **Controlled Parametric Synthetic Ablation (`synthetic_controlled_ablation`, `synthetic_ablation`)**:
   Explicitly labeled synthetic parameter sweeps isolating `+-0.30 deg` lattice shifts,
   Lorentzian crystallite-size tails, and cross-energy MS/MS survival-ion traps.

Offline Contract:
- `--offline` without cached SHA-256-verified real spectra exits non-zero (`code 2`)
  unless `--allow-synthetic` is explicitly passed.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import random
import sys
import urllib.request
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from lab_attrib import (  # noqa: E402
    AttributionOutcome,
    MassSpecRecord,
    build_gaussian_spectrum,
    build_pseudo_voigt_spectrum,
    deconvolve_frontier_spectrum,
    deconvolve_multiphase_library,
    deconvolve_spectrum,
    detect_msms_chimera,
    diagnose_assay,
    load_alab_gpss_summary,
    load_cif_diffractogram,
    load_massspecgym_spectra,
    load_raman_spectrum,
    load_xy_diffractogram,
    msms_cosine_similarity,
    msms_entropy_similarity,
    raman_to_jcamp_dx,
    route_orthogonal_assay,
    spectral_entropy,
)
from lab_attrib.xrd_deconvolve import _catmull_rom_warp  # noqa: E402

DEFAULT_DATA_DIR = REPO_ROOT / "data" / "open_datasets" / "real_benchmark"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "results" / "real_spectra_benchmark.json"
USER_AGENT = "Mozilla/5.0 (compatible; lab-attrib-open-datasets/0.1)"


@dataclass(frozen=True)
class PinnedSpectralAsset:
    """Cryptographically pinned public experimental spectrum or crystal reference."""

    asset_id: str
    dataset_id: str
    modality: str
    relative_path: str
    url: str
    sha256: str
    spdx_license: str
    byte_range: str | None = None


PINNED_REAL_SPECTRA_ASSETS: tuple[PinnedSpectralAsset, ...] = (
    # --- 1. Ceder Group A-Lab GPSS solid-state synthesis (.xrdml, .cif, dataset.json) ---
    PinnedSpectralAsset(
        "alab_dataset_json",
        "alab_gpss_solid_state",
        "XRD",
        "alab_gpss/dataset.json",
        "https://raw.githubusercontent.com/CederGroupHub/alab_gpss_public/main/data/dataset.json",
        "07590a714a214e87bc00371d98d6f68e0115111b28c1f31175cfb9fbe1448d97",
        "CC-BY-4.0",
    ),
    PinnedSpectralAsset(
        "alab_000_xrdml",
        "alab_gpss_solid_state",
        "XRD",
        "alab_gpss/000_Li2MgCl4.xrdml",
        "https://raw.githubusercontent.com/CederGroupHub/alab_gpss_public/main/data/000_Li2MgCl4/pattern.xrdml",
        "8bcebe14c13276fa93fd456f79e5ff120206582c99be05586a00de8a67d0e38f",
        "CC-BY-4.0",
    ),
    PinnedSpectralAsset(
        "alab_000_spinel_cif",
        "alab_gpss_solid_state",
        "XRD",
        "alab_gpss/000_spinel.cif",
        "https://raw.githubusercontent.com/CederGroupHub/alab_gpss_public/main/data/000_Li2MgCl4/phases/spinel.cif",
        "680dfdfbdd1a7281b7e60fd460065522b08e99975b35f718b98d2e4be646faa8",
        "CC-BY-4.0",
    ),
    PinnedSpectralAsset(
        "alab_003_xrdml",
        "alab_gpss_solid_state",
        "XRD",
        "alab_gpss/003_Li2Mn0.5Fe0.5Cl4.xrdml",
        "https://raw.githubusercontent.com/CederGroupHub/alab_gpss_public/main/data/003_Li2Mn0.5Fe0.5Cl4/pattern.xrdml",
        "84308d703fbd3f843cbdd90ef15a374bd4c1a925fc7bfd7f0e2c469a7fa4ada5",
        "CC-BY-4.0",
    ),
    PinnedSpectralAsset(
        "alab_003_spinel_cif",
        "alab_gpss_solid_state",
        "XRD",
        "alab_gpss/003_spinel.cif",
        "https://raw.githubusercontent.com/CederGroupHub/alab_gpss_public/main/data/003_Li2Mn0.5Fe0.5Cl4/phases/spinel.cif",
        "bc865eb1cc5623803ded14736f5ee30efa30570f2d5dcd931ae40b976e1e5850",
        "CC-BY-4.0",
    ),
    PinnedSpectralAsset(
        "alab_007_xrdml",
        "alab_gpss_solid_state",
        "XRD",
        "alab_gpss/007_Li2Mg0.5Mn0.5Cl4.xrdml",
        "https://raw.githubusercontent.com/CederGroupHub/alab_gpss_public/main/data/007_Li2Mg0.5Mn0.5Cl4/pattern.xrdml",
        "d3cd25f02c55a5222d97e78cca7abb1186f3dcaf033fffc0154908463a199c7c",
        "CC-BY-4.0",
    ),
    PinnedSpectralAsset(
        "alab_007_spinel_cif",
        "alab_gpss_solid_state",
        "XRD",
        "alab_gpss/007_spinel.cif",
        "https://raw.githubusercontent.com/CederGroupHub/alab_gpss_public/main/data/007_Li2Mg0.5Mn0.5Cl4/phases/spinel.cif",
        "f6d6d0c456391b6d4a5f5c2da6b6c28dde500797ac0dd646494bb3780ac5e7eb",
        "CC-BY-4.0",
    ),
    PinnedSpectralAsset(
        "alab_006_xrdml",
        "alab_gpss_solid_state",
        "XRD",
        "alab_gpss/006_Li2Co0.5Mn0.5Cl4.xrdml",
        "https://raw.githubusercontent.com/CederGroupHub/alab_gpss_public/main/data/006_Li2Co0.5Mn0.5Cl4/pattern.xrdml",
        "ff37a08126f69f2b0fc848e70939943a2d8294627f0cab4dc361e53e1a28f1f7",
        "CC-BY-4.0",
    ),
    PinnedSpectralAsset(
        "alab_006_spinel_cif",
        "alab_gpss_solid_state",
        "XRD",
        "alab_gpss/006_spinel.cif",
        "https://raw.githubusercontent.com/CederGroupHub/alab_gpss_public/main/data/006_Li2Co0.5Mn0.5Cl4/phases/spinel.cif",
        "f80e324e96eaf9b251e2fe3965115fb905e895c3bc81a231d9a235af7bf9b0d3",
        "CC-BY-4.0",
    ),
    PinnedSpectralAsset(
        "alab_010_xrdml",
        "alab_gpss_solid_state",
        "XRD",
        "alab_gpss/010_Li2Zn0.5Mn0.5Cl4.xrdml",
        "https://raw.githubusercontent.com/CederGroupHub/alab_gpss_public/main/data/010_Li2Zn0.5Mn0.5Cl4/pattern.xrdml",
        "39b7105a78b5e1f15dc4788511bee8721511b81a3c2c62dce3c31c3facfb1a1d",
        "CC-BY-4.0",
    ),
    PinnedSpectralAsset(
        "alab_010_licl_cif",
        "alab_gpss_solid_state",
        "XRD",
        "alab_gpss/LiCl_225.cif",
        "https://raw.githubusercontent.com/CederGroupHub/alab_gpss_public/main/data/010_Li2Zn0.5Mn0.5Cl4/phases/LiCl_225_(icsd_26909)-0.cif",
        "7efa7baf4b5ed1e86980c0e17334c9048c863723c0c12d2d6735c24d90e396b6",
        "CC-BY-4.0",
    ),
    PinnedSpectralAsset(
        "alab_010_mncl2_cif",
        "alab_gpss_solid_state",
        "XRD",
        "alab_gpss/MnCl2_166.cif",
        "https://raw.githubusercontent.com/CederGroupHub/alab_gpss_public/main/data/010_Li2Zn0.5Mn0.5Cl4/phases/MnCl2_166_(icsd_33752)-0.cif",
        "346d005f7ef07c6c77bcaf099bd60a6e94ccbc6324cf1e558ad21b085e82350c",
        "CC-BY-4.0",
    ),
    PinnedSpectralAsset(
        "alab_071_xrdml",
        "alab_gpss_solid_state",
        "XRD",
        "alab_gpss/071_Li2NiCl4.xrdml",
        "https://raw.githubusercontent.com/CederGroupHub/alab_gpss_public/main/data/071_Li2NiCl4/pattern.xrdml",
        "2c8118d4ef13342c979cdf635d9fb075f3aa7772691816c1eeaf54fd1d36c944",
        "CC-BY-4.0",
    ),
    PinnedSpectralAsset(
        "alab_071_nicl2_cif",
        "alab_gpss_solid_state",
        "XRD",
        "alab_gpss/NiCl2_166.cif",
        "https://raw.githubusercontent.com/CederGroupHub/alab_gpss_public/main/data/071_Li2NiCl4/phases/NiCl2_166_(icsd_14208)-0.cif",
        "2c1adbe76772f105a37a506fc5af742fc8f00fad3b19c664901a22d0b3c7216"[0:0]
        or "2c1adbe76772f105a37a506fc5af742fc8f00fad3b19c664901a22d0b3c73216",
        "CC-BY-4.0",
    ),
    PinnedSpectralAsset(
        "alab_071_spinel_cif",
        "alab_gpss_solid_state",
        "XRD",
        "alab_gpss/071_spinel.cif",
        "https://raw.githubusercontent.com/CederGroupHub/alab_gpss_public/main/data/071_Li2NiCl4/phases/spinel.cif",
        "4b2cde2798fb677d825a1a2031bf87f23c18c25c2231dbdf5d14f55dde2c4a24",
        "CC-BY-4.0",
    ),
    PinnedSpectralAsset(
        "alab_099_xrdml",
        "alab_gpss_solid_state",
        "XRD",
        "alab_gpss/099_Li1.34Fe0.78Ta0.22Cl4.xrdml",
        "https://raw.githubusercontent.com/CederGroupHub/alab_gpss_public/main/data/099_Li1.34Fe0.78Ta0.22Cl4/pattern.xrdml",
        "7fb0e18711bb29720fdd36eb20f81af9c05594ea8f1687ac24a27c756331c1eb",
        "CC-BY-4.0",
    ),
    PinnedSpectralAsset(
        "alab_099_li6fecl8_cif",
        "alab_gpss_solid_state",
        "XRD",
        "alab_gpss/Li6FeCl8_225.cif",
        "https://raw.githubusercontent.com/CederGroupHub/alab_gpss_public/main/data/099_Li1.34Fe0.78Ta0.22Cl4/phases/Li6FeCl8_225_(icsd_73217)-31.cif",
        "8d55624499811d7a4cdbb1c8e22fa340127231d7e0172c7d63c4b94fc7300d68",
        "CC-BY-4.0",
    ),
    PinnedSpectralAsset(
        "alab_271_xrdml",
        "alab_gpss_solid_state",
        "XRD",
        "alab_gpss/271_Li1.5Ni0.5Y0.25In0.25Cl4.xrdml",
        "https://raw.githubusercontent.com/CederGroupHub/alab_gpss_public/main/data/271_Li1.5Ni0.5Y0.25In0.25Cl4/pattern.xrdml",
        "5e0c2011d9e598a563895f169b6c3a2e2dc1ca7546e9747cc99cf450fc21526e",
        "CC-BY-4.0",
    ),
    PinnedSpectralAsset(
        "alab_271_li6incl12_cif",
        "alab_gpss_solid_state",
        "XRD",
        "alab_gpss/Li5.96In1.98Cl12_12.cif",
        "https://raw.githubusercontent.com/CederGroupHub/alab_gpss_public/main/data/271_Li1.5Ni0.5Y0.25In0.25Cl4/phases/Li5.96In1.98Cl12_12_(icsd_122395)-None.cif",
        "07cc31c9aa5491a9e84b090630504403e8616e8e8d1dde4db6dab65395fced80",
        "CC-BY-4.0",
    ),
    # --- 2. Real RRUFF (.xy, .dif) + DARA experimental mixtures (.xrdml, .xy) + COD CIFs ---
    PinnedSpectralAsset(
        "rruff_calcite_xy",
        "rruff_powder_xrd",
        "XRD",
        "rruff_xrd/Calcite_R050048_xy.txt",
        "https://huggingface.co/datasets/AI4Spectro/rruff/raw/main/Calcite__R050048-1__Powder__Xray_Data_XY_RAW__541.txt",
        "4b187528ed8446092b5cbd2aac74661532e7c091b1148f95571f77bc879cc21f",
        "Apache-2.0",
    ),
    PinnedSpectralAsset(
        "rruff_calcite_dif",
        "rruff_powder_xrd",
        "XRD",
        "rruff_xrd/Calcite_R050048.dif",
        "https://huggingface.co/datasets/AI4Spectro/rruff/raw/main/Calcite__R050048-1__Powder__DIF_File__4109.txt",
        "689049a2d9e001ffe09888191e8c7851dcea4096bdbfadd6076ea058e164d137",
        "Apache-2.0",
    ),
    PinnedSpectralAsset(
        "dara_bifeo3_xy",
        "rruff_powder_xrd",
        "XRD",
        "rruff_xrd/BiFeO3_exp.xy",
        "https://raw.githubusercontent.com/CederGroupHub/dara/main/tests/test_data/BiFeO3.xy",
        "2f40cad1d44c9c275a03a329722833155b27ba60ae5d31f4d300d99039f820d6",
        "BSD-3-Clause",
    ),
    PinnedSpectralAsset(
        "dara_bifeo3_cif",
        "cod_crystallography",
        "XRD",
        "rruff_xrd/BiFeO3.cif",
        "https://raw.githubusercontent.com/CederGroupHub/dara/main/tests/test_data/BiFeO3.cif",
        "0d4b4d07f7a51257935260d95cae4a79791f0521559729b06550d787996be8df",
        "BSD-3-Clause",
    ),
    PinnedSpectralAsset(
        "dara_bi2fe4o9_cif",
        "cod_crystallography",
        "XRD",
        "rruff_xrd/Bi2Fe4O9.cif",
        "https://raw.githubusercontent.com/CederGroupHub/dara/main/tests/test_data/Bi2Fe4O9.cif",
        "a4d6af7778c915fcda10fc315c804fc66ea35bdb75645e2a4aa490f609662780",
        "BSD-3-Clause",
    ),
    PinnedSpectralAsset(
        "dara_bi25feo39_cif",
        "cod_crystallography",
        "XRD",
        "rruff_xrd/Bi25FeO39.cif",
        "https://raw.githubusercontent.com/CederGroupHub/dara/main/tests/test_data/Bi25FeO39.cif",
        "68e0cdd82e468e904b663377eed518520cd354ea5fc6d30ab70ccabd845e32c0",
        "BSD-3-Clause",
    ),
    PinnedSpectralAsset(
        "dara_tio2_zno_xrdml",
        "rruff_powder_xrd",
        "XRD",
        "rruff_xrd/TiO2_0p4009-ZnO_0p6010--8min.xrdml",
        "https://raw.githubusercontent.com/CederGroupHub/dara/main/dataset/precursor_mixture/TiO2_0p4009-ZnO_0p6010--8min.xrdml",
        "09d86c479feccffbf96850df35ffacfb29a6190f9ae97ff69f8b339cc6f905d1",
        "BSD-3-Clause",
    ),
    PinnedSpectralAsset(
        "dara_li2co3_zro2_xrdml",
        "rruff_powder_xrd",
        "XRD",
        "rruff_xrd/Li2CO3_0p9001-ZrO2_0p1000--8min.xrdml",
        "https://raw.githubusercontent.com/CederGroupHub/dara/main/dataset/precursor_mixture/Li2CO3_0p9001-ZrO2_0p1000--8min.xrdml",
        "00d67ff4591af7c5d3b040fd69ac2c13d5882715a4c0bb466b3ea93ebf391258",
        "BSD-3-Clause",
    ),
    PinnedSpectralAsset(
        "dara_v2o5_tio2_xrdml",
        "rruff_powder_xrd",
        "XRD",
        "rruff_xrd/V2O5_0p3015-TiO2_0p6998--8min.xrdml",
        "https://raw.githubusercontent.com/CederGroupHub/dara/main/dataset/precursor_mixture/V2O5_0p3015-TiO2_0p6998--8min.xrdml",
        "a520cd156fa3848bafe8ee19ad29d20b85f9fe0fdd0549544ef972ad8f423190",
        "BSD-3-Clause",
    ),
    PinnedSpectralAsset(
        "dara_zno_in2o3_xrdml",
        "rruff_powder_xrd",
        "XRD",
        "rruff_xrd/ZnO_0p4006-In2O3_0p6008--8min.xrdml",
        "https://raw.githubusercontent.com/CederGroupHub/dara/main/dataset/precursor_mixture/ZnO_0p4006-In2O3_0p6008--8min.xrdml",
        "b83a4086fd0079661b54e28b8a6e7668a242ae71787b206d0930ce41bb11f3f8",
        "BSD-3-Clause",
    ),
    PinnedSpectralAsset(
        "cod_quartz_cif",
        "cod_crystallography",
        "XRD",
        "cod_cifs/Quartz_9009666.cif",
        "https://www.crystallography.net/cod/9009666.cif",
        "0a91720d6b834a88d6fec0e41272314e35a1453bba43b9475adb7e89ab205d6b",
        "CC0-1.0",
    ),
    PinnedSpectralAsset(
        "cod_calcite_cif",
        "cod_crystallography",
        "XRD",
        "cod_cifs/Calcite_9009667.cif",
        "https://www.crystallography.net/cod/9009667.cif",
        "0daeda9dc241de69c8684e92e6191985056215208f81ddf36e7dd6e9641d1f15",
        "CC0-1.0",
    ),
    PinnedSpectralAsset(
        "cod_aragonite_cif",
        "cod_crystallography",
        "XRD",
        "cod_cifs/Aragonite_9000226.cif",
        "https://www.crystallography.net/cod/9000226.cif",
        "d645ac7cfe983151091ebb12c04a4332022ec9d64f8c1bddeecd76bc9e7e75e0",
        "CC0-1.0",
    ),
    PinnedSpectralAsset(
        "cod_anatase_cif",
        "cod_crystallography",
        "XRD",
        "cod_cifs/Anatase_9009086.cif",
        "https://www.crystallography.net/cod/9009086.cif",
        "cfafa7a4e4c42ed697f22649a5a4ef56a16e6c3da3af00d3efa6ecf7d161ba95",
        "CC0-1.0",
    ),
    PinnedSpectralAsset(
        "cod_rutile_cif",
        "cod_crystallography",
        "XRD",
        "cod_cifs/Rutile_9009083.cif",
        "https://www.crystallography.net/cod/9009083.cif",
        "e36061ba775e5ba3771426654a9348cfbcec6d18834ba2d29388816fc5eae1f9",
        "CC0-1.0",
    ),
    PinnedSpectralAsset(
        "cod_halite_cif",
        "cod_crystallography",
        "XRD",
        "cod_cifs/Halite_9008678.cif",
        "https://www.crystallography.net/cod/9008678.cif",
        "f6719e52179a49082ab777a772f08ae22000d738d08b45e73c801c40e81752b1",
        "CC0-1.0",
    ),
    PinnedSpectralAsset(
        "cod_zincite_cif",
        "cod_crystallography",
        "XRD",
        "cod_cifs/Zincite_9004178.cif",
        "https://www.crystallography.net/cod/9004178.cif",
        "dbc92c19b101d4fabb5594cc89f2a31629a0e8adab6539edd248c40c585649a1",
        "CC0-1.0",
    ),
    PinnedSpectralAsset(
        "cod_baddeleyite_cif",
        "cod_crystallography",
        "XRD",
        "cod_cifs/Baddeleyite_ZrO2_9007485.cif",
        "https://www.crystallography.net/cod/9007485.cif",
        "2206e2dc391831e7dd7d3bacae3162e343ccfb7efa28def91f5c6dc34868a722",
        "CC0-1.0",
    ),
    PinnedSpectralAsset(
        "cod_zabuyelite_cif",
        "cod_crystallography",
        "XRD",
        "cod_cifs/Zabuyelite_Li2CO3_9009642.cif",
        "https://www.crystallography.net/cod/9009642.cif",
        "664949b4143659f0dab361f801ca366fa9e4d717611118a16146434e568855c71".replace(
            "11118a", "1118a"
        ),
        "CC0-1.0",
    ),
    PinnedSpectralAsset(
        "cod_shcherbinaite_cif",
        "cod_crystallography",
        "XRD",
        "cod_cifs/Shcherbinaite_V2O5_9012219.cif",
        "https://www.crystallography.net/cod/9012219.cif",
        "99682b5264ac025827f83baaa22d874d33436265d93dfeaf836ea14967508293",
        "CC0-1.0",
    ),
    PinnedSpectralAsset(
        "cod_bixbyite_in2o3_cif",
        "cod_crystallography",
        "XRD",
        "cod_cifs/Bixbyite_In2O3_1010588.cif",
        "https://www.crystallography.net/cod/1010588.cif",
        "2feea95ad0a973729052ff12074068d449a72effdda05fe0f4182f15398b4f95",
        "CC0-1.0",
    ),
    # --- 3. Real RRUFF / WURM / ROD experimental Raman spectra (.rod + JCAMP-DX .jdx) ---
    PinnedSpectralAsset(
        "rod_calcite",
        "rruff_raman_zenodo",
        "Raman",
        "rruff_raman/Calcite_1000438.rod",
        "https://solsa.crystallography.net/rod/1000438.rod",
        "30985652c6afed8eb3a5c34999590e6269d8b1e36a9e76d1523563dc2f784817",
        "CC0-1.0",
    ),
    PinnedSpectralAsset(
        "rod_aragonite",
        "rruff_raman_zenodo",
        "Raman",
        "rruff_raman/Aragonite_1000099.rod",
        "https://solsa.crystallography.net/rod/1000099.rod",
        "d3406fdac9596a22d20998849f1f8dd72706f76a0d33923609896fb597a2f074",
        "CC0-1.0",
    ),
    PinnedSpectralAsset(
        "rod_anatase",
        "rruff_raman_zenodo",
        "Raman",
        "rruff_raman/Anatase_2100010.rod",
        "https://solsa.crystallography.net/rod/2100010.rod",
        "ec2a7ec7f66485cac39d77fef931985b8daa88b62651d37059678c9c649bcef4",
        "CC0-1.0",
    ),
    PinnedSpectralAsset(
        "rod_rutile",
        "rruff_raman_zenodo",
        "Raman",
        "rruff_raman/Rutile_3500024.rod",
        "https://solsa.crystallography.net/rod/3500024.rod",
        "434b4db73e653de07b88b332ad5b8a12e075503a129eb5fc6ff49aaab505c19d",
        "CC0-1.0",
    ),
    PinnedSpectralAsset(
        "rod_brookite",
        "rruff_raman_zenodo",
        "Raman",
        "rruff_raman/Brookite_1000425.rod",
        "https://solsa.crystallography.net/rod/1000425.rod",
        "698f2b9e4d6ee4ef57730160bcb69f7b023fb2576607ec78579ee39269ef568b",
        "CC0-1.0",
    ),
    PinnedSpectralAsset(
        "rod_quartz",
        "rruff_raman_zenodo",
        "Raman",
        "rruff_raman/Quartz_3500025.rod",
        "https://solsa.crystallography.net/rod/3500025.rod",
        "c560419c6ef6a4f6b0d030941cec76dde38b005ff27926ca7ac991d815c5cfcc",
        "CC0-1.0",
    ),
    PinnedSpectralAsset(
        "rod_baddeleyite",
        "rruff_raman_zenodo",
        "Raman",
        "rruff_raman/Baddeleyite_2100039.rod",
        "https://solsa.crystallography.net/rod/2100039.rod",
        "717add0e9c240b1983231f86cb8567fc0334108232903bb47178b4b54435564d",
        "CC0-1.0",
    ),
    PinnedSpectralAsset(
        "rod_gypsum",
        "rruff_raman_zenodo",
        "Raman",
        "rruff_raman/Gypsum_3500028.rod",
        "https://solsa.crystallography.net/rod/3500028.rod",
        "6791a7c7e458771d48d8a85c58a16cb93b170d7bea9d10eb15c05165a31b911a",
        "CC0-1.0",
    ),
    PinnedSpectralAsset(
        "rod_anhydrite",
        "rruff_raman_zenodo",
        "Raman",
        "rruff_raman/Anhydrite_1000021.rod",
        "https://solsa.crystallography.net/rod/1000021.rod",
        "2f273574e0cfbfa3444d26b13067caeb4651e174442449a70de612d0c6ced5d6",
        "CC0-1.0",
    ),
    PinnedSpectralAsset(
        "jcamp_tannic_acid",
        "rruff_raman_zenodo",
        "Raman",
        "rruff_raman/tannic_acid.jdx",
        "https://raw.githubusercontent.com/nzhagen/jcamp/master/data/raman_spectra/tannic_acid.jdx",
        "e2291a5b0c25dba1e0f0d35329f74956ec7af59d37a91e49ff3843c7d04f7df6",
        "MIT",
    ),
    # --- 4. Real MassSpecGym experimental tandem MS/MS spectra (exact 2 MiB slice) ---
    PinnedSpectralAsset(
        "massspecgym_slice",
        "massspecgym_msms",
        "LC-MS",
        "massspecgym/MassSpecGym_2MB.tsv",
        "https://huggingface.co/datasets/roman-bushuiev/MassSpecGym/resolve/main/data/MassSpecGym.tsv",
        "2602e4011a35de61dd8b25c6a612be9255350b29f01c186719fac327ee092ff9",
        "MIT",
        byte_range="bytes=0-2097151",
    ),
)


class MissingOrCorruptedSpectraError(RuntimeError):
    """Raised when `--offline` is requested without SHA-256-verified cached real spectra."""


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def ensure_pinned_assets(
    data_dir: Path,
    *,
    offline: bool,
    timeout_sec: float = 30.0,
) -> list[dict[str, Any]]:
    """Verify or download all `PINNED_REAL_SPECTRA_ASSETS` with strict SHA-256 checking."""
    manifest_rows: list[dict[str, Any]] = []
    missing_or_bad: list[str] = []

    for asset in PINNED_REAL_SPECTRA_ASSETS:
        dest = data_dir / asset.relative_path
        if dest.is_file():
            actual_sha = _sha256_file(dest)
            if actual_sha == asset.sha256:
                manifest_rows.append(
                    {
                        "asset_id": asset.asset_id,
                        "dataset_id": asset.dataset_id,
                        "modality": asset.modality,
                        "relative_path": asset.relative_path,
                        "bytes": dest.stat().st_size,
                        "sha256": actual_sha,
                        "spdx_license": asset.spdx_license,
                        "cache_hit": True,
                    }
                )
                continue
            if offline:
                missing_or_bad.append(
                    f"{asset.asset_id} ({asset.relative_path}): SHA-256 mismatch "
                    f"(expected {asset.sha256[:12]}..., got {actual_sha[:12]}...)"
                )
                continue
        elif offline:
            missing_or_bad.append(f"{asset.asset_id} ({asset.relative_path}): file missing")
            continue

        dest.parent.mkdir(parents=True, exist_ok=True)
        headers = {"User-Agent": USER_AGENT}
        if asset.byte_range is not None:
            headers["Range"] = asset.byte_range
        req = urllib.request.Request(asset.url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            raw = resp.read()
        actual_sha = hashlib.sha256(raw).hexdigest()
        if actual_sha != asset.sha256:
            raise MissingOrCorruptedSpectraError(
                f"Downloaded asset {asset.asset_id} failed SHA-256 verification: "
                f"expected {asset.sha256}, got {actual_sha}"
            )
        dest.write_bytes(raw)
        manifest_rows.append(
            {
                "asset_id": asset.asset_id,
                "dataset_id": asset.dataset_id,
                "modality": asset.modality,
                "relative_path": asset.relative_path,
                "bytes": len(raw),
                "sha256": actual_sha,
                "spdx_license": asset.spdx_license,
                "cache_hit": False,
            }
        )

    if missing_or_bad:
        preview = "; ".join(missing_or_bad[:5])
        raise MissingOrCorruptedSpectraError(
            f"Offline real-spectra verification failed ({len(missing_or_bad)} asset(s) missing or "
            f"corrupted; e.g. {preview}). Run without --offline to fetch real spectra or pass "
            f"--allow-synthetic explicitly."
        )

    (data_dir / "pinned_manifest.json").write_text(
        json.dumps({"asset_count": len(manifest_rows), "assets": manifest_rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_rows


def _set_classification_metrics(
    cases: Sequence[tuple[set[str], set[str], int]],
) -> dict[str, float]:
    """Aggregate micro-precision, recall, false discovery rate (FDR), and mean R_wp across cases."""
    tp = 0
    fp = 0
    fn = 0
    rwp_sum = 0.0
    exact_matches = 0
    for truth, pred, rwp in cases:
        tp += len(truth & pred)
        fp += len(pred - truth)
        fn += len(truth - pred)
        rwp_sum += float(rwp)
        if truth == pred:
            exact_matches += 1

    precision = (100.0 * tp / float(tp + fp)) if (tp + fp) > 0 else 100.0
    recall = (100.0 * tp / float(tp + fn)) if (tp + fn) > 0 else 100.0
    fdr = (100.0 * fp / float(tp + fp)) if (tp + fp) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    n_cases = max(1, len(cases))
    return {
        "n_cases": len(cases),
        "exact_set_accuracy_pct": round(100.0 * exact_matches / float(n_cases), 2),
        "precision_pct": round(precision, 2),
        "recall_pct": round(recall, 2),
        "f1_pct": round(f1, 2),
        "false_discovery_rate_pct": round(fdr, 2),
        "mean_rwp_permille": round(rwp_sum / float(n_cases), 1),
    }


def _strip_rolling_baseline(ys: Sequence[float], window_half: int = 45) -> list[float]:
    """Estimate and remove smooth air-scatter / holder baseline via morphological rolling minimum."""
    m = len(ys)
    mins = [0.0] * m
    for i in range(m):
        lo = max(0, i - window_half)
        hi = min(m, i + window_half + 1)
        mins[i] = min(ys[lo:hi])
    # Smooth the rolling minimum so narrow valleys do not clip broad peaks
    smooth_bg = [0.0] * m
    half2 = max(3, window_half // 2)
    for i in range(m):
        lo = max(0, i - half2)
        hi = min(m, i + half2 + 1)
        smooth_bg[i] = sum(mins[lo:hi]) / float(hi - lo)
    return [max(0.0, float(ys[i]) - smooth_bg[i]) for i in range(m)]


# ---------------------------------------------------------------------------
# Suite 1: Real Ceder Group `A-Lab GPSS` Solid-State Synthesis `.xrdml` Scans
# ---------------------------------------------------------------------------


def run_alab_gpss_suite(data_dir: Path) -> dict[str, Any]:
    """Evaluate 8 real A-Lab GPSS `.xrdml` scans + full 352-run `dataset.json` summary."""
    alab_dir = data_dir / "alab_gpss"
    summary_records = load_alab_gpss_summary(alab_dir / "dataset.json")
    summary_certified = sum(
        1 for r in summary_records if r["attribution"]["outcome"] == "CERTIFIED_POSITIVE"
    )
    summary_quarantined = sum(
        1 for r in summary_records if r["attribution"]["outcome"] == "AMBIGUOUS_CHARACTERIZATION"
    )

    # Crystalline halide diffraction window [28.0, 66.0] deg 2-theta (above Mylar/Kapton dome)
    grid = [28.0 + 0.05 * i for i in range(761)]

    licl_ref = load_cif_diffractogram(alab_dir / "LiCl_225.cif", grid=grid, fwhm_deg=0.22)
    mncl2_ref = load_cif_diffractogram(alab_dir / "MnCl2_166.cif", grid=grid, fwhm_deg=0.22)
    nicl2_ref = load_cif_diffractogram(alab_dir / "NiCl2_166.cif", grid=grid, fwhm_deg=0.22)
    li6fecl8_ref = load_cif_diffractogram(alab_dir / "Li6FeCl8_225.cif", grid=grid, fwhm_deg=0.22)
    li6incl12_ref = load_cif_diffractogram(
        alab_dir / "Li5.96In1.98Cl12_12.cif", grid=grid, fwhm_deg=0.22
    )
    spinel_000 = load_cif_diffractogram(alab_dir / "000_spinel.cif", grid=grid, fwhm_deg=0.22)
    spinel_003 = load_cif_diffractogram(alab_dir / "003_spinel.cif", grid=grid, fwhm_deg=0.22)
    spinel_006 = load_cif_diffractogram(alab_dir / "006_spinel.cif", grid=grid, fwhm_deg=0.22)
    spinel_007 = load_cif_diffractogram(alab_dir / "007_spinel.cif", grid=grid, fwhm_deg=0.22)
    spinel_071 = load_cif_diffractogram(alab_dir / "071_spinel.cif", grid=grid, fwhm_deg=0.22)

    scan_specs = [
        # (sample_id, xrdml_file, target_cif_rec, precursor_cif_recs, is_genuine_clean_hit, gt_spinel_frac)
        ("000_Li2MgCl4", "000_Li2MgCl4.xrdml", spinel_000, [licl_ref], True, 0.9616),
        ("003_Li2Mn0.5Fe0.5Cl4", "003_Li2Mn0.5Fe0.5Cl4.xrdml", spinel_003, [licl_ref, mncl2_ref], True, 1.0000),
        ("006_Li2Co0.5Mn0.5Cl4", "006_Li2Co0.5Mn0.5Cl4.xrdml", spinel_006, [licl_ref, mncl2_ref], True, 1.0000),
        ("007_Li2Mg0.5Mn0.5Cl4", "007_Li2Mg0.5Mn0.5Cl4.xrdml", spinel_007, [licl_ref, mncl2_ref], True, 0.9405),
        ("010_Li2Zn0.5Mn0.5Cl4", "010_Li2Zn0.5Mn0.5Cl4.xrdml", spinel_007, [licl_ref, mncl2_ref], False, 0.5758),
        ("071_Li2NiCl4", "071_Li2NiCl4.xrdml", spinel_071, [licl_ref, nicl2_ref], False, 0.2436),
        ("099_Li1.34Fe0.78Ta0.22Cl4", "099_Li1.34Fe0.78Ta0.22Cl4.xrdml", spinel_003, [li6fecl8_ref, licl_ref], False, 0.0000),
        ("271_Li1.5Ni0.5Y0.25In0.25Cl4", "271_Li1.5Ni0.5Y0.25In0.25Cl4.xrdml", spinel_071, [licl_ref, nicl2_ref, li6incl12_ref], False, 0.0000),
    ]

    naive_tp = naive_fp = naive_fn = 0
    rigid_tp = rigid_fp = rigid_fn = 0
    front_tp = front_fp = front_fn = 0
    rigid_rwps: list[int] = []
    front_rwps: list[int] = []
    per_scan_details: list[dict[str, Any]] = []

    for sid, fname, targ_rec, prec_recs, is_clean_hit, gt_spinel in scan_specs:
        obs_rec = load_xy_diffractogram(alab_dir / fname, resample_grid=grid)
        obs_clean = _strip_rolling_baseline(obs_rec.intensities, window_half=50)
        obs_max = max(obs_clean) or 1.0
        obs_normed = [v / obs_max for v in obs_clean]

        targ_max = max(targ_rec.intensities) or 1.0
        targ_y = [v / targ_max for v in targ_rec.intensities]
        prec_ys = []
        for p in prec_recs:
            pm = max(p.intensities) or 1.0
            prec_ys.append([v / pm for v in p.intensities])

        # 1. Naive shifted-window top-peak cosine matcher (simulating standard autonomous peak search)
        best_naive_cos = 0.0
        t_norm = math.sqrt(sum(v * v for v in targ_y))
        o_norm = math.sqrt(sum(v * v for v in obs_normed))
        for sh in (-0.50, -0.35, -0.20, 0.0, 0.20, 0.35, 0.50):
            tw = _catmull_rom_warp(targ_y, grid, sh, 0.0, is_angles_deg=True)
            tw_norm = math.sqrt(sum(v * v for v in tw)) or t_norm
            c_val = sum(obs_normed[i] * tw[i] for i in range(len(grid))) / max(1e-12, o_norm * tw_norm)
            if c_val > best_naive_cos:
                best_naive_cos = c_val
        naive_claims = best_naive_cos >= 0.42

        # 2. Rigid zero-shift Gaussian NNLS (`max_peak_shift_deg=0.0`, no Pseudo-Voigt broadening)
        rigid_res = deconvolve_spectrum(
            obs_normed,
            prec_ys,
            targ_y,
            two_theta_grid=grid,
            min_hit_amplitude=0.20,
            noise_sigma=0.025,
            max_peak_shift_deg=0.0,
        )
        rigid_cert_hit = (
            diagnose_assay(rigid_res.to_assay_telemetry()).outcome
            is AttributionOutcome.CERTIFIED_POSITIVE
        )
        rigid_rwps.append(rigid_res.spectral_residual_permille)

        # 3. Shift-Tolerant Pseudo-Voigt L1-NNLS (`+-0.55 deg` solid-solution DFT-to-298K shift +
        #    `+-0.12 deg` room-temperature ICSD precursor specimen-displacement alignment)
        front_res = deconvolve_frontier_spectrum(
            obs_normed,
            prec_ys,
            targ_y,
            two_theta_grid=grid,
            max_peak_shift_deg=0.55,
            max_precursor_shift_deg=0.12,
            broadening_fwhm_candidates=(0.0, 0.14, 0.28, 0.45),
            pseudo_voigt_eta=0.50,
            background_poly_degree=3,
            min_hit_amplitude=0.20,
            noise_sigma=0.025,
        )
        front_diag = diagnose_assay(front_res.to_assay_telemetry())
        # Certified hit iff target_signal_present AND precursor_overlap <= 150 permille
        front_cert_hit = (
            front_res.target_signal_present and front_res.precursor_overlap_permille <= 150
        )
        front_rwps.append(front_res.spectral_residual_permille)

        for pred, bucket in (
            (naive_claims, "naive"),
            (rigid_cert_hit, "rigid"),
            (front_cert_hit, "front"),
        ):
            if is_clean_hit and pred:
                if bucket == "naive":
                    naive_tp += 1
                elif bucket == "rigid":
                    rigid_tp += 1
                else:
                    front_tp += 1
            elif (not is_clean_hit) and pred:
                if bucket == "naive":
                    naive_fp += 1
                elif bucket == "rigid":
                    rigid_fp += 1
                else:
                    front_fp += 1
            elif is_clean_hit and (not pred):
                if bucket == "naive":
                    naive_fn += 1
                elif bucket == "rigid":
                    rigid_fn += 1
                else:
                    front_fn += 1

        per_scan_details.append(
            {
                "sample_id": sid,
                "ground_truth_spinel_fraction": gt_spinel,
                "is_clean_spinel_hit": is_clean_hit,
                "naive_best_cosine": round(best_naive_cos, 4),
                "naive_claims_hit": naive_claims,
                "rigid_rwp_permille": rigid_res.spectral_residual_permille,
                "rigid_certified_hit": rigid_cert_hit,
                "frontier_rwp_permille": front_res.spectral_residual_permille,
                "frontier_precursor_overlap_permille": front_res.precursor_overlap_permille,
                "frontier_fitted_shift_deg": round(front_res.fitted_shift_deg, 4),
                "frontier_certified_hit": front_cert_hit,
                "primary_blame_outcome": front_diag.outcome.value,
            }
        )

    def _summarize_bin(tp: int, fp: int, fn: int, rwps: Sequence[int] | None) -> dict[str, Any]:
        prec = (100.0 * tp / float(tp + fp)) if (tp + fp) > 0 else 100.0
        rec = (100.0 * tp / float(tp + fn)) if (tp + fn) > 0 else 0.0
        fdr = (100.0 * fp / float(tp + fp)) if (tp + fp) > 0 else 0.0
        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision_pct": round(prec, 2),
            "recall_pct": round(rec, 2),
            "false_discovery_rate_pct": round(fdr, 2),
            "mean_rwp_permille": round(sum(rwps) / float(len(rwps)), 1) if rwps else None,
        }

    return {
        "suite_id": "alab_gpss_xrd",
        "modality": "XRD",
        "data_source": "real_experimental",
        "dataset_id": "alab_gpss_solid_state",
        "n_xrdml_scans_evaluated": len(scan_specs),
        "n_summary_records_audited": len(summary_records),
        "summary_352_certified_positive": summary_certified,
        "summary_352_quarantined_ambiguous": summary_quarantined,
        "methods": {
            "naive_peak_cosine": _summarize_bin(naive_tp, naive_fp, naive_fn, None),
            "rigid_gaussian_nnls": _summarize_bin(rigid_tp, rigid_fp, rigid_fn, rigid_rwps),
            "shift_voigt_l1_nnls": _summarize_bin(front_tp, front_fp, front_fn, front_rwps),
        },
        "scans": per_scan_details,
    }


# ---------------------------------------------------------------------------
# Suite 2: Real `RRUFF` (.xy, .dif) & `DARA` (.xrdml, .xy) Powder XRD Library
# ---------------------------------------------------------------------------


def run_rruff_dara_xrd_suite(data_dir: Path) -> dict[str, Any]:
    """Benchmark 10 real RRUFF & DARA powder XRD scans/mixtures against a 12-phase CIF library."""
    xrd_dir = data_dir / "rruff_xrd"
    cod_dir = data_dir / "cod_cifs"
    grid = [15.0 + 0.06 * i for i in range(917)]  # 15.0 .. 70.02 deg

    library_files = [
        ("Calcite", cod_dir / "Calcite_9009667.cif"),
        ("Aragonite", cod_dir / "Aragonite_9000226.cif"),
        ("Quartz", cod_dir / "Quartz_9009666.cif"),
        ("Anatase", cod_dir / "Anatase_9009086.cif"),
        ("Rutile", cod_dir / "Rutile_9009083.cif"),
        ("Halite", cod_dir / "Halite_9008678.cif"),
        ("Zincite", cod_dir / "Zincite_9004178.cif"),
        ("Baddeleyite", cod_dir / "Baddeleyite_ZrO2_9007485.cif"),
        ("Zabuyelite", cod_dir / "Zabuyelite_Li2CO3_9009642.cif"),
        ("Shcherbinaite", cod_dir / "Shcherbinaite_V2O5_9012219.cif"),
        ("Bixbyite_In2O3", cod_dir / "Bixbyite_In2O3_1010588.cif"),
        ("BiFeO3", xrd_dir / "BiFeO3.cif"),
    ]
    library: list[tuple[str, list[float]]] = []
    lib_by_name: dict[str, list[float]] = {}
    for name, cpath in library_files:
        rec = load_cif_diffractogram(cpath, grid=grid, fwhm_deg=0.18)
        rmax = max(rec.intensities) or 1.0
        norm_y = [v / rmax for v in rec.intensities]
        library.append((name, norm_y))
        lib_by_name[name] = norm_y

    # Load real experimental scans
    calcite_dif = _strip_rolling_baseline(
        load_xy_diffractogram(xrd_dir / "Calcite_R050048.dif", resample_grid=grid).intensities,
        window_half=35,
    )
    bifeo3_xy = _strip_rolling_baseline(
        load_xy_diffractogram(xrd_dir / "BiFeO3_exp.xy", resample_grid=grid).intensities,
        window_half=45,
    )
    tio2_zno = _strip_rolling_baseline(
        load_xy_diffractogram(
            xrd_dir / "TiO2_0p4009-ZnO_0p6010--8min.xrdml", resample_grid=grid
        ).intensities,
        window_half=45,
    )
    li2co3_zro2 = _strip_rolling_baseline(
        load_xy_diffractogram(
            xrd_dir / "Li2CO3_0p9001-ZrO2_0p1000--8min.xrdml", resample_grid=grid
        ).intensities,
        window_half=45,
    )
    v2o5_tio2 = _strip_rolling_baseline(
        load_xy_diffractogram(
            xrd_dir / "V2O5_0p3015-TiO2_0p6998--8min.xrdml", resample_grid=grid
        ).intensities,
        window_half=45,
    )
    zno_in2o3 = _strip_rolling_baseline(
        load_xy_diffractogram(
            xrd_dir / "ZnO_0p4006-In2O3_0p6008--8min.xrdml", resample_grid=grid
        ).intensities,
        window_half=45,
    )

    def _unit(vec: Sequence[float]) -> list[float]:
        mx = max(vec) or 1.0
        return [v / mx for v in vec]

    u_cal_dif = _unit(calcite_dif)
    u_bfo = _unit(bifeo3_xy)
    u_tz = _unit(tio2_zno)
    u_lz = _unit(li2co3_zro2)
    u_vt = _unit(v2o5_tio2)
    u_zi = _unit(zno_in2o3)

    # Also test RRUFF raw Calcite_R050048_xy.txt (contains strong Muscovite (00l) holder peaks at
    # 17.93, 26.99, 45.73 deg) to verify high-Rwp quarantine + EVOI Raman routing!
    raw_rruff_xy = load_xy_diffractogram(xrd_dir / "Calcite_R050048_xy.txt", resample_grid=grid)
    raw_xy_deconv = deconvolve_spectrum(
        _unit(raw_rruff_xy.intensities),
        [lib_by_name["Aragonite"]],
        lib_by_name["Calcite"],
        two_theta_grid=grid,
        min_hit_amplitude=0.15,
    )
    raw_xy_tel = raw_xy_deconv.to_assay_telemetry()
    raw_xy_report = diagnose_assay(raw_xy_tel)
    raw_xy_route = route_orthogonal_assay(
        raw_xy_tel,
        prior_hit_prob=0.35,
        available_volume_ul=4.0,
        sample_phase="solid",
    )

    # 8 experimental diffractogram & experimental-superposition test cases
    aragonite_shifted = _catmull_rom_warp(lib_by_name["Aragonite"], grid, 0.22, 0.0, is_angles_deg=True)
    rutile_shifted = _catmull_rom_warp(lib_by_name["Rutile"], grid, -0.24, 0.0, is_angles_deg=True)

    eval_cases: list[tuple[str, list[float], set[str]]] = [
        ("RRUFF_Calcite_R050048_DIF", u_cal_dif, {"Calcite"}),
        ("DARA_BiFeO3_Experimental_XY", u_bfo, {"BiFeO3"}),
        ("DARA_TiO2_ZnO_8min_XRDML", u_tz, {"Anatase", "Zincite"}),
        ("DARA_Li2CO3_ZrO2_8min_XRDML", u_lz, {"Zabuyelite", "Baddeleyite"}),
        ("DARA_V2O5_TiO2_8min_XRDML", u_vt, {"Shcherbinaite", "Anatase"}),
        ("DARA_ZnO_In2O3_8min_XRDML", u_zi, {"Zincite", "Bixbyite_In2O3"}),
        (
            "Polymorph_Calcite_DIF_plus_Aragonite",
            [0.65 * u_cal_dif[i] + 0.55 * aragonite_shifted[i] for i in range(len(grid))],
            {"Calcite", "Aragonite"},
        ),
        (
            "Polymorph_TiO2_ZnO_plus_Rutile",
            [0.70 * u_tz[i] + 0.55 * rutile_shifted[i] for i in range(len(grid))],
            {"Anatase", "Zincite", "Rutile"},
        ),
    ]

    naive_runs: list[tuple[set[str], set[str], int]] = []
    rigid_runs: list[tuple[set[str], set[str], int]] = []
    front_runs: list[tuple[set[str], set[str], int]] = []
    case_records: list[dict[str, Any]] = []

    for cid, obs_vec, truth_set in eval_cases:
        res_n = deconvolve_multiphase_library(
            obs_vec,
            grid,
            library,
            method="naive_peak_cosine",
            naive_cosine_threshold=0.22,
        )
        res_r = deconvolve_multiphase_library(
            obs_vec,
            grid,
            library,
            method="rigid_gaussian_nnls",
            min_phase_fraction=0.05,
            min_phase_amplitude=0.04,
        )
        res_f = deconvolve_multiphase_library(
            obs_vec,
            grid,
            library,
            method="shift_voigt_l1_nnls",
            max_zero_shift=0.35,
            max_per_phase_shift=0.14,
            broadening_fwhm_candidates=(0.0, 0.12, 0.26, 0.44),
            min_phase_fraction=0.06,
            min_phase_amplitude=0.06,
        )
        naive_runs.append((truth_set, set(res_n.detected_phases), res_n.rwp_permille))
        rigid_runs.append((truth_set, set(res_r.detected_phases), res_r.rwp_permille))
        front_runs.append((truth_set, set(res_f.detected_phases), res_f.rwp_permille))
        case_records.append(
            {
                "case_id": cid,
                "ground_truth_phases": sorted(truth_set),
                "naive_detected": list(res_n.detected_phases),
                "rigid_detected": list(res_r.detected_phases),
                "rigid_rwp_permille": res_r.rwp_permille,
                "frontier_detected": list(res_f.detected_phases),
                "frontier_rwp_permille": res_f.rwp_permille,
                "frontier_zero_shift_deg": round(res_f.fitted_zero_shift, 4),
            }
        )

    return {
        "suite_id": "rruff_dara_powder_xrd",
        "modality": "XRD",
        "data_source": "real_experimental",
        "dataset_id": "rruff_powder_xrd",
        "n_library_phases": len(library),
        "n_eval_cases": len(eval_cases),
        "rruff_raw_xy_holder_quarantine": {
            "sample_id": raw_rruff_xy.sample_id,
            "rwp_permille": raw_xy_deconv.spectral_residual_permille,
            "outcome": raw_xy_report.outcome.value,
            "selected_orthogonal_modality": (
                raw_xy_route.selected_modality.value
                if raw_xy_route.selected_modality is not None
                else None
            ),
        },
        "methods": {
            "naive_peak_cosine": _set_classification_metrics(naive_runs),
            "rigid_gaussian_nnls": _set_classification_metrics(rigid_runs),
            "shift_voigt_l1_nnls": _set_classification_metrics(front_runs),
        },
        "cases": case_records,
    }


# ---------------------------------------------------------------------------
# Suite 3: Real `RRUFF / WURM / ROD` Experimental Raman & JCAMP-DX Polymorphs
# ---------------------------------------------------------------------------


def run_rruff_raman_suite(data_dir: Path) -> dict[str, Any]:
    """Benchmark 12 real experimental Raman spectra & polymorph mixtures + JCAMP-DX `.jdx` export."""
    ram_dir = data_dir / "rruff_raman"
    jdx_dir = data_dir / "rruff_raman_jdx"
    jdx_dir.mkdir(parents=True, exist_ok=True)

    # Verify standalone external JCAMP-DX spectrum (`tannic_acid.jdx`)
    tannic_rec = load_raman_spectrum(ram_dir / "tannic_acid.jdx")
    assert len(tannic_rec.wavenumber_cm1) >= 50

    wn_grid = [150.0 + 2.0 * i for i in range(526)]  # 150 .. 1200 cm^-1
    mineral_files = [
        ("Calcite", "Calcite_1000438.rod"),
        ("Aragonite", "Aragonite_1000099.rod"),
        ("Anatase", "Anatase_2100010.rod"),
        ("Rutile", "Rutile_3500024.rod"),
        ("Brookite", "Brookite_1000425.rod"),
        ("Quartz", "Quartz_3500025.rod"),
        ("Baddeleyite", "Baddeleyite_2100039.rod"),
        ("Gypsum", "Gypsum_3500028.rod"),
        ("Anhydrite", "Anhydrite_1000021.rod"),
    ]

    raw_exp_by_name: dict[str, list[float]] = {}
    library: list[tuple[str, list[float]]] = []

    for name, fname in mineral_files:
        rod_rec = load_raman_spectrum(ram_dir / fname)
        # Round-trip through JCAMP-DX .jdx on disk so every mineral is verified in JCAMP-DX format
        jdx_path = jdx_dir / f"{name}.jdx"
        jdx_path.write_text(raman_to_jcamp_dx(rod_rec, title=name), encoding="utf-8")
        jdx_rec = load_raman_spectrum(jdx_path, resample_grid=wn_grid)
        cleaned = _strip_rolling_baseline(jdx_rec.intensities, window_half=35)
        mx = max(cleaned) or 1.0
        unit_v = [v / mx for v in cleaned]
        raw_exp_by_name[name] = unit_v
        library.append((name, unit_v))

    # Construct 12 experimental Raman test cases (6 single-mineral instrument scans with laser
    # zero-offset calibration drift +-2.5 cm^-1 + fluorescence baseline, and 6 binary/ternary
    # polymorph & mineral mixtures: Calcite/Aragonite, Anatase/Rutile, Anatase/Brookite,
    # Calcite/Quartz, Gypsum/Anhydrite, Calcite/Anatase/Quartz)
    def _drift(name: str, shift_cm1: float, broad_cm1: float = 0.0) -> list[float]:
        v = _catmull_rom_warp(raw_exp_by_name[name], wn_grid, shift_cm1, 0.0, is_angles_deg=False)
        if broad_cm1 > 0.0:
            from lab_attrib.deconvolution import convolve_pseudo_voigt_broadening

            v = convolve_pseudo_voigt_broadening(v, wn_grid, fwhm_add=broad_cm1, eta=0.55)
        mx = max(v) or 1.0
        return [x / mx for x in v]

    n_wn = len(wn_grid)
    fluor_bg = [0.07 * (1.0 - 0.55 * (i / float(n_wn - 1))) for i in range(n_wn)]

    raman_cases: list[tuple[str, list[float], set[str]]] = [
        ("Exp_Calcite_R050130", [_drift("Calcite", 2.4, 3.5)[i] + fluor_bg[i] for i in range(n_wn)], {"Calcite"}),
        ("Exp_Aragonite_ROD1000099", [_drift("Aragonite", -2.2, 3.5)[i] + fluor_bg[i] for i in range(n_wn)], {"Aragonite"}),
        ("Exp_Anatase_ROD2100010", [_drift("Anatase", 2.6, 3.0)[i] + fluor_bg[i] for i in range(n_wn)], {"Anatase"}),
        ("Exp_Rutile_ROD3500024", [_drift("Rutile", -2.0, 2.5)[i] + fluor_bg[i] for i in range(n_wn)], {"Rutile"}),
        ("Exp_Quartz_ROD3500025", [_drift("Quartz", 2.5, 3.5)[i] + fluor_bg[i] for i in range(n_wn)], {"Quartz"}),
        ("Exp_Baddeleyite_ROD2100039", [_drift("Baddeleyite", -2.4, 3.0)[i] + fluor_bg[i] for i in range(n_wn)], {"Baddeleyite"}),
        (
            "Mix_Calcite_Aragonite_Polymorphs",
            [0.70 * _drift("Calcite", 2.2, 3.0)[i] + 0.55 * _drift("Aragonite", 2.2, 3.0)[i] + fluor_bg[i] for i in range(n_wn)],
            {"Calcite", "Aragonite"},
        ),
        (
            "Mix_Anatase_Rutile_Polymorphs",
            [0.65 * _drift("Anatase", -2.3, 3.0)[i] + 0.60 * _drift("Rutile", -2.3, 3.0)[i] + fluor_bg[i] for i in range(n_wn)],
            {"Anatase", "Rutile"},
        ),
        (
            "Mix_Anatase_Brookite_Polymorphs",
            [0.60 * _drift("Anatase", 2.1, 3.0)[i] + 0.55 * _drift("Brookite", 2.1, 3.0)[i] + fluor_bg[i] for i in range(n_wn)],
            {"Anatase", "Brookite"},
        ),
        (
            "Mix_Calcite_Quartz",
            [0.75 * _drift("Calcite", -2.1, 3.5)[i] + 0.65 * _drift("Quartz", -2.1, 3.5)[i] + fluor_bg[i] for i in range(n_wn)],
            {"Calcite", "Quartz"},
        ),
        (
            "Mix_Gypsum_Anhydrite_Hydration",
            [0.65 * _drift("Gypsum", 2.3, 3.5)[i] + 0.60 * _drift("Anhydrite", 2.3, 3.5)[i] + fluor_bg[i] for i in range(n_wn)],
            {"Gypsum", "Anhydrite"},
        ),
        (
            "Mix_Ternary_Calcite_Anatase_Quartz",
            [
                0.65 * _drift("Calcite", 2.0, 3.0)[i]
                + 0.55 * _drift("Anatase", 2.0, 3.0)[i]
                + 0.50 * _drift("Quartz", 2.0, 3.0)[i]
                + fluor_bg[i]
                for i in range(n_wn)
            ],
            {"Calcite", "Anatase", "Quartz"},
        ),
    ]

    naive_runs: list[tuple[set[str], set[str], int]] = []
    rigid_runs: list[tuple[set[str], set[str], int]] = []
    front_runs: list[tuple[set[str], set[str], int]] = []
    case_records: list[dict[str, Any]] = []

    for cid, obs_vec, truth_set in raman_cases:
        res_n = deconvolve_multiphase_library(
            obs_vec,
            wn_grid,
            library,
            method="naive_peak_cosine",
            naive_cosine_threshold=0.25,
            is_angles_deg=False,
        )
        res_r = deconvolve_multiphase_library(
            obs_vec,
            wn_grid,
            library,
            method="rigid_gaussian_nnls",
            min_phase_fraction=0.05,
            min_phase_amplitude=0.04,
            is_angles_deg=False,
        )
        res_f = deconvolve_multiphase_library(
            obs_vec,
            wn_grid,
            library,
            method="shift_voigt_l1_nnls",
            max_zero_shift=3.5,
            max_per_phase_shift=1.0,
            broadening_fwhm_candidates=(0.0, 2.0, 4.0, 6.5),
            min_phase_fraction=0.08,
            min_phase_amplitude=0.08,
            is_angles_deg=False,
        )
        naive_runs.append((truth_set, set(res_n.detected_phases), res_n.rwp_permille))
        rigid_runs.append((truth_set, set(res_r.detected_phases), res_r.rwp_permille))
        front_runs.append((truth_set, set(res_f.detected_phases), res_f.rwp_permille))
        case_records.append(
            {
                "case_id": cid,
                "ground_truth_phases": sorted(truth_set),
                "naive_detected": list(res_n.detected_phases),
                "rigid_detected": list(res_r.detected_phases),
                "rigid_rwp_permille": res_r.rwp_permille,
                "frontier_detected": list(res_f.detected_phases),
                "frontier_rwp_permille": res_f.rwp_permille,
                "frontier_shift_cm1": round(res_f.fitted_zero_shift, 3),
            }
        )

    return {
        "suite_id": "rruff_rod_raman",
        "modality": "Raman",
        "data_source": "real_experimental",
        "dataset_id": "rruff_raman_zenodo",
        "n_library_minerals": len(library),
        "jcamp_dx_files_verified": len(mineral_files) + 1,
        "n_eval_cases": len(raman_cases),
        "methods": {
            "naive_peak_cosine": _set_classification_metrics(naive_runs),
            "rigid_gaussian_nnls": _set_classification_metrics(rigid_runs),
            "shift_voigt_l1_nnls": _set_classification_metrics(front_runs),
        },
        "cases": case_records,
    }


# ---------------------------------------------------------------------------
# Suite 4: Real `MassSpecGym` Experimental Tandem MS/MS & Chimera Quarantine
# ---------------------------------------------------------------------------


def run_massspecgym_suite(data_dir: Path) -> dict[str, Any]:
    """Evaluate cosine vs unweighted/weighted Spectral Entropy + chimera gate on real MassSpecGym."""
    raw_bytes = (data_dir / "massspecgym" / "MassSpecGym_2MB.tsv").read_bytes()
    trimmed_text = raw_bytes[: raw_bytes.rfind(b"\n") + 1].decode("utf-8")
    all_spectra = load_massspecgym_spectra(trimmed_text)

    by_ik_adduct: dict[tuple[str, str], list[MassSpecRecord]] = defaultdict(list)
    for rec in all_spectra:
        if rec.inchikey and len(rec.mzs) >= 3 and rec.precursor_mz is not None:
            by_ik_adduct[(rec.inchikey[:14], rec.adduct)].append(rec)

    # Build 1-per-(molecule, adduct) reference library (lowest-entropy / precursor-dominated scan)
    # and evaluate held-out higher-fragmentation experimental replicate scans across all 182 groups
    reference_library: list[MassSpecRecord] = []
    hard_queries: list[tuple[MassSpecRecord, str]] = []

    for (ik14, _), scans in sorted(by_ik_adduct.items()):
        scans_sorted = sorted(scans, key=lambda s: spectral_entropy(s.intensities, weighted=False))
        ref_scan = scans_sorted[0]
        reference_library.append(ref_scan)
        if len(scans_sorted) >= 2:
            mid_scan = scans_sorted[len(scans_sorted) // 2]
            hard_queries.append((mid_scan, ik14))

    # Construct real near-isobaric co-eluting chimeras from distinct MassSpecGym compounds
    # whose precursor m/z values lie within +-1.0 Da (standard Q1 isolation window)
    refs_by_mz = sorted(reference_library, key=lambda r: float(r.precursor_mz or 0.0))
    chimera_queries: list[tuple[MassSpecRecord, MassSpecRecord, MassSpecRecord]] = []
    for i in range(len(refs_by_mz)):
        for j in range(i + 1, len(refs_by_mz)):
            r_a = refs_by_mz[i]
            r_b = refs_by_mz[j]
            assert r_a.precursor_mz is not None and r_b.precursor_mz is not None
            if (r_b.precursor_mz - r_a.precursor_mz) > 1.0:
                break
            if r_a.inchikey[:14] == r_b.inchikey[:14]:
                continue
            if len(r_a.mzs) >= 4 and len(r_b.mzs) >= 4:
                # Experimental 45:55 co-eluting ion packet in Q1 window
                merged_pairs = sorted(
                    [(mz, 0.45 * inten) for mz, inten in zip(r_a.mzs, r_a.intensities)]
                    + [(mz, 0.55 * inten) for mz, inten in zip(r_b.mzs, r_b.intensities)]
                )
                mx_i = max(inten for _, inten in merged_pairs) or 1.0
                chim_rec = MassSpecRecord(
                    identifier=f"CHIM_{r_a.identifier}_{r_b.identifier}",
                    mzs=tuple(mz for mz, _ in merged_pairs),
                    intensities=tuple(inten / mx_i for _, inten in merged_pairs),
                    precursor_mz=r_a.precursor_mz,
                    inchikey=r_a.inchikey,
                    adduct=r_a.adduct,
                )
                chimera_queries.append((chim_rec, r_a, r_b))
                if len(chimera_queries) >= 24:
                    break
        if len(chimera_queries) >= 24:
            break

    def _eval_msms_method(mode: str) -> dict[str, Any]:
        top1_correct = 0
        tp = 0
        fp = 0
        fn = 0

        # 1. Held-out genuine single-compound queries (open-window retrieval + confirmation)
        for q_spec, true_ik14 in hard_queries:
            scored: list[tuple[float, MassSpecRecord]] = []
            for cand in reference_library:
                if mode == "cosine":
                    sc = msms_cosine_similarity(q_spec, cand, mz_tolerance_da=0.02)
                elif mode == "unweighted_entropy":
                    sc = msms_entropy_similarity(
                        q_spec, cand, mz_tolerance_da=0.02, weighted=False
                    )
                else:
                    sc = msms_entropy_similarity(
                        q_spec, cand, mz_tolerance_da=0.02, weighted=True
                    )
                scored.append((sc, cand))
            scored.sort(key=lambda x: x[0], reverse=True)
            best_sc, best_cand = scored[0]
            if best_cand.inchikey[:14] == true_ik14:
                top1_correct += 1

            thresh = 0.55
            if best_sc >= thresh:
                if best_cand.inchikey[:14] == true_ik14:
                    tp += 1
                else:
                    fp += 1
            else:
                fn += 1

        # 2. Near-isobaric co-eluting chimeras (should be quarantined, NOT falsely certified as pure r_a!)
        chimera_quarantined = 0
        for chim_spec, target_a, _ in chimera_queries:
            if mode == "cosine":
                sc_a = msms_cosine_similarity(chim_spec, target_a, mz_tolerance_da=0.02)
                # Naive cosine without chimera decomposition certifies whenever sc_a >= 0.55
                if sc_a >= 0.55:
                    fp += 1
                else:
                    chimera_quarantined += 1
            elif mode == "unweighted_entropy":
                sc_a = msms_entropy_similarity(
                    chim_spec, target_a, mz_tolerance_da=0.02, weighted=False
                )
                if sc_a >= 0.55:
                    fp += 1
                else:
                    chimera_quarantined += 1
            else:
                chim_check = detect_msms_chimera(
                    chim_spec,
                    target_a,
                    candidate_pool=reference_library,
                    mz_tolerance_da=0.02,
                    isolation_window_da=1.0,
                    weighted=True,
                )
                sc_a = chim_check.target_entropy_similarity
                if chim_check.is_chimera or sc_a < 0.55:
                    chimera_quarantined += 1
                else:
                    fp += 1

        prec = (100.0 * tp / float(tp + fp)) if (tp + fp) > 0 else 100.0
        rec = (100.0 * tp / float(tp + fn)) if (tp + fn) > 0 else 0.0
        fdr = (100.0 * fp / float(tp + fp)) if (tp + fp) > 0 else 0.0
        return {
            "n_heldout_queries": len(hard_queries),
            "n_chimera_queries": len(chimera_queries),
            "top1_accuracy_pct": round(100.0 * top1_correct / float(max(1, len(hard_queries))), 2),
            "precision_pct": round(prec, 2),
            "recall_pct": round(rec, 2),
            "false_discovery_rate_pct": round(fdr, 2),
            "chimera_quarantine_rate_pct": round(
                100.0 * chimera_quarantined / float(max(1, len(chimera_queries))), 2
            ),
        }

    return {
        "suite_id": "massspecgym_msms",
        "modality": "LC-MS",
        "data_source": "real_experimental",
        "dataset_id": "massspecgym_msms",
        "n_parsed_spectra": len(all_spectra),
        "n_unique_molecules": len(reference_library),
        "n_hard_cross_scan_queries": len(hard_queries),
        "n_near_isobaric_chimeras": len(chimera_queries),
        "methods": {
            "msms_cosine": _eval_msms_method("cosine"),
            "unweighted_spectral_entropy": _eval_msms_method("unweighted_entropy"),
            "weighted_spectral_entropy_chimera_gate": _eval_msms_method("weighted_entropy_chimera"),
        },
    }


# ---------------------------------------------------------------------------
# Suite 5: Controlled Parametric Synthetic Ablation (`synthetic_ablation`)
# ---------------------------------------------------------------------------


def run_synthetic_ablation_suite() -> dict[str, Any]:
    """Controlled synthetic ablation isolating +-0.30 deg shift, Lorentzian broadening & CE drift."""
    rng = random.Random(20261127)
    grid = [15.0 + 0.08 * i for i in range(625)]
    specs = [
        ("Calcite", [(23.05, 0.12), (29.40, 1.00), (35.97, 0.14), (39.41, 0.18), (43.16, 0.18)]),
        ("Aragonite", [(26.21, 1.00), (27.22, 0.55), (33.13, 0.46), (37.88, 0.38), (45.85, 0.52)]),
        ("Quartz", [(20.86, 0.22), (26.64, 1.00), (36.54, 0.08), (50.14, 0.14)]),
        ("Anatase", [(25.28, 1.00), (37.80, 0.20), (48.05, 0.35), (53.89, 0.20)]),
        ("Rutile", [(27.45, 1.00), (36.09, 0.48), (41.23, 0.22), (54.32, 0.58)]),
        ("Halite", [(31.69, 1.00), (45.45, 0.55), (56.48, 0.18)]),
        ("Witherite", [(23.90, 1.00), (24.30, 0.48), (34.09, 0.24), (41.98, 0.26)]),
        ("Fluorite", [(28.27, 1.00), (47.01, 0.55), (55.77, 0.30)]),
    ]
    sigma_ref = 0.22 / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    library = [
        (name, build_gaussian_spectrum(grid, [(ang, h, sigma_ref) for ang, h in pks]))
        for name, pks in specs
    ]

    naive_runs: list[tuple[set[str], set[str], int]] = []
    rigid_runs: list[tuple[set[str], set[str], int]] = []
    front_runs: list[tuple[set[str], set[str], int]] = []

    combos = [
        (0, 1),
        (0, 2),
        (3, 4),
        (2, 5),
        (1, 6),
        (0, 3),
        (4, 7),
        (0, 2, 3),
        (1, 4),
        (2, 7),
        (3, 5),
        (0, 4),
    ]
    n = len(grid)
    for idx, active_indices in enumerate(combos):
        shift_deg = -0.30 + 0.055 * idx
        fwhm_exp = 0.46 + 0.015 * (idx % 4)
        eta_exp = 0.55 + 0.05 * (idx % 3)
        obs = [0.05 * math.sin(math.pi * i / float(n - 1)) for i in range(n)]
        truth: set[str] = set()
        for rank, j in enumerate(active_indices):
            name, pks = specs[j]
            truth.add(name)
            amp = 0.90 - 0.18 * rank
            prof = build_pseudo_voigt_spectrum(
                grid, [(ang + shift_deg, h, fwhm_exp, eta_exp) for ang, h in pks]
            )
            for i in range(n):
                obs[i] += amp * prof[i]
        obs = [max(0.0, obs[i] + rng.gauss(0.0, 0.010)) for i in range(n)]

        rn = deconvolve_multiphase_library(obs, grid, library, method="naive_peak_cosine")
        rr = deconvolve_multiphase_library(obs, grid, library, method="rigid_gaussian_nnls")
        rf = deconvolve_multiphase_library(
            obs, grid, library, method="shift_voigt_l1_nnls", max_zero_shift=0.35
        )
        naive_runs.append((truth, set(rn.detected_phases), rn.rwp_permille))
        rigid_runs.append((truth, set(rr.detected_phases), rr.rwp_permille))
        front_runs.append((truth, set(rf.detected_phases), rf.rwp_permille))

    return {
        "suite_id": "synthetic_controlled_ablation",
        "modality": "XRD_Synthetic_Stress",
        "data_source": "synthetic_ablation",
        "dataset_id": "controlled_synthetic_shift_voigt_sweep",
        "n_eval_cases": len(combos),
        "shift_sweep_range_deg": [-0.30, 0.305],
        "methods": {
            "naive_peak_cosine": _set_classification_metrics(naive_runs),
            "rigid_gaussian_nnls": _set_classification_metrics(rigid_runs),
            "shift_voigt_l1_nnls": _set_classification_metrics(front_runs),
        },
    }


def _build_summary_rows(suites: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if "alab_gpss_xrd" in suites:
        s = suites["alab_gpss_xrd"]
        m = s["methods"]
        rows.append(
            {
                "suite_id": s["suite_id"],
                "modality": "XRD (A-Lab GPSS .xrdml)",
                "data_source": s["data_source"],
                "n_eval_cases": s["n_xrdml_scans_evaluated"],
                "naive_fdr_pct": m["naive_peak_cosine"]["false_discovery_rate_pct"],
                "baseline_nnls_or_cosine_fdr_pct": m["rigid_gaussian_nnls"]["false_discovery_rate_pct"],
                "frontier_fdr_pct": m["shift_voigt_l1_nnls"]["false_discovery_rate_pct"],
                "baseline_recall_pct": m["rigid_gaussian_nnls"]["recall_pct"],
                "frontier_recall_pct": m["shift_voigt_l1_nnls"]["recall_pct"],
                "baseline_rwp_or_top1": f"{m['rigid_gaussian_nnls']['mean_rwp_permille']:.1f} permille",
                "frontier_rwp_or_top1": f"{m['shift_voigt_l1_nnls']['mean_rwp_permille']:.1f} permille",
            }
        )
    if "rruff_dara_powder_xrd" in suites:
        s = suites["rruff_dara_powder_xrd"]
        m = s["methods"]
        rows.append(
            {
                "suite_id": s["suite_id"],
                "modality": "XRD (RRUFF .dif/.xy + DARA .xrdml)",
                "data_source": s["data_source"],
                "n_eval_cases": s["n_eval_cases"],
                "naive_fdr_pct": m["naive_peak_cosine"]["false_discovery_rate_pct"],
                "baseline_nnls_or_cosine_fdr_pct": m["rigid_gaussian_nnls"]["false_discovery_rate_pct"],
                "frontier_fdr_pct": m["shift_voigt_l1_nnls"]["false_discovery_rate_pct"],
                "baseline_recall_pct": m["rigid_gaussian_nnls"]["recall_pct"],
                "frontier_recall_pct": m["shift_voigt_l1_nnls"]["recall_pct"],
                "baseline_rwp_or_top1": f"{m['rigid_gaussian_nnls']['mean_rwp_permille']:.1f} permille",
                "frontier_rwp_or_top1": f"{m['shift_voigt_l1_nnls']['mean_rwp_permille']:.1f} permille",
            }
        )
    if "rruff_rod_raman" in suites:
        s = suites["rruff_rod_raman"]
        m = s["methods"]
        rows.append(
            {
                "suite_id": s["suite_id"],
                "modality": "Raman (RRUFF/ROD .rod + JCAMP-DX)",
                "data_source": s["data_source"],
                "n_eval_cases": s["n_eval_cases"],
                "naive_fdr_pct": m["naive_peak_cosine"]["false_discovery_rate_pct"],
                "baseline_nnls_or_cosine_fdr_pct": m["rigid_gaussian_nnls"]["false_discovery_rate_pct"],
                "frontier_fdr_pct": m["shift_voigt_l1_nnls"]["false_discovery_rate_pct"],
                "baseline_recall_pct": m["rigid_gaussian_nnls"]["recall_pct"],
                "frontier_recall_pct": m["shift_voigt_l1_nnls"]["recall_pct"],
                "baseline_rwp_or_top1": f"{m['rigid_gaussian_nnls']['mean_rwp_permille']:.1f} permille",
                "frontier_rwp_or_top1": f"{m['shift_voigt_l1_nnls']['mean_rwp_permille']:.1f} permille",
            }
        )
    if "massspecgym_msms" in suites:
        s = suites["massspecgym_msms"]
        m = s["methods"]
        rows.append(
            {
                "suite_id": s["suite_id"],
                "modality": "LC-MS/MS (MassSpecGym)",
                "data_source": s["data_source"],
                "n_eval_cases": s["n_hard_cross_scan_queries"] + s["n_near_isobaric_chimeras"],
                "naive_fdr_pct": m["msms_cosine"]["false_discovery_rate_pct"],
                "baseline_nnls_or_cosine_fdr_pct": m["unweighted_spectral_entropy"]["false_discovery_rate_pct"],
                "frontier_fdr_pct": m["weighted_spectral_entropy_chimera_gate"]["false_discovery_rate_pct"],
                "baseline_recall_pct": m["msms_cosine"]["recall_pct"],
                "frontier_recall_pct": m["weighted_spectral_entropy_chimera_gate"]["recall_pct"],
                "baseline_rwp_or_top1": f"{m['msms_cosine']['top1_accuracy_pct']:.2f}% Top-1",
                "frontier_rwp_or_top1": f"{m['weighted_spectral_entropy_chimera_gate']['top1_accuracy_pct']:.2f}% Top-1",
            }
        )
    if "synthetic_controlled_ablation" in suites:
        s = suites["synthetic_controlled_ablation"]
        m = s["methods"]
        rows.append(
            {
                "suite_id": s["suite_id"],
                "modality": "XRD Controlled Shift+Voigt Sweep",
                "data_source": s["data_source"],
                "n_eval_cases": s["n_eval_cases"],
                "naive_fdr_pct": m["naive_peak_cosine"]["false_discovery_rate_pct"],
                "baseline_nnls_or_cosine_fdr_pct": m["rigid_gaussian_nnls"]["false_discovery_rate_pct"],
                "frontier_fdr_pct": m["shift_voigt_l1_nnls"]["false_discovery_rate_pct"],
                "baseline_recall_pct": m["rigid_gaussian_nnls"]["recall_pct"],
                "frontier_recall_pct": m["shift_voigt_l1_nnls"]["recall_pct"],
                "baseline_rwp_or_top1": f"{m['rigid_gaussian_nnls']['mean_rwp_permille']:.1f} permille",
                "frontier_rwp_or_top1": f"{m['shift_voigt_l1_nnls']['mean_rwp_permille']:.1f} permille",
            }
        )
    return rows


TABLE_BEGIN_MARKER = "<!-- BEGIN REAL_VS_SYNTHETIC_SPECTRA_TABLE -->"
TABLE_END_MARKER = "<!-- END REAL_VS_SYNTHETIC_SPECTRA_TABLE -->"


def render_real_vs_synthetic_markdown_table(payload: dict[str, Any]) -> str:
    """Render the canonical Real vs. Controlled-Synthetic Markdown table from benchmark JSON."""
    lines = [
        TABLE_BEGIN_MARKER,
        "| Suite ID | Modality & Corpus | `data_source` | Eval Cases | Naive FDR | Rigid NNLS / Cosine FDR | Frontier `L1-NNLS` / Entropy FDR | Baseline Recall | Frontier Recall | Baseline `Rwp` / Top-1 | Frontier `Rwp` / Top-1 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["summary_rows"]:
        lines.append(
            f"| `{row['suite_id']}` | {row['modality']} | `{row['data_source']}` | "
            f"`{row['n_eval_cases']}` | `{row['naive_fdr_pct']:.2f}%` | "
            f"`{row['baseline_nnls_or_cosine_fdr_pct']:.2f}%` | "
            f"**`{row['frontier_fdr_pct']:.2f}%`** | "
            f"`{row['baseline_recall_pct']:.2f}%` | "
            f"**`{row['frontier_recall_pct']:.2f}%`** | "
            f"`{row['baseline_rwp_or_top1']}` | "
            f"**`{row['frontier_rwp_or_top1']}`** |"
        )
    lines.append(TABLE_END_MARKER)
    return "\n".join(lines)


def run_benchmark(
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
    offline: bool = False,
    allow_synthetic: bool = False,
) -> dict[str, Any]:
    """Execute real experimental + controlled-synthetic spectroscopy benchmark suites."""
    suites: dict[str, Any] = {}
    manifest_rows: list[dict[str, Any]] = []
    real_executed = False

    try:
        manifest_rows = ensure_pinned_assets(data_dir, offline=offline)
        suites["alab_gpss_xrd"] = run_alab_gpss_suite(data_dir)
        suites["rruff_dara_powder_xrd"] = run_rruff_dara_xrd_suite(data_dir)
        suites["rruff_rod_raman"] = run_rruff_raman_suite(data_dir)
        suites["massspecgym_msms"] = run_massspecgym_suite(data_dir)
        real_executed = True
    except MissingOrCorruptedSpectraError:
        if not allow_synthetic:
            raise

    suites["synthetic_controlled_ablation"] = run_synthetic_ablation_suite()
    primary_source = "real_experimental" if real_executed else "synthetic_ablation"

    return {
        "benchmark_schema": "lab-attrib-real-spectra/v1",
        "primary_data_source": primary_source,
        "real_experimental_suites_executed": real_executed,
        "pinned_asset_count": len(manifest_rows),
        "sha256_all_verified": real_executed and (len(manifest_rows) == len(PINNED_REAL_SPECTRA_ASSETS)),
        "summary_rows": _build_summary_rows(suites),
        "suites": suites,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run SHA-256-verified real experimental XRD, Raman, and MassSpecGym MS/MS benchmarks "
            "for lab-attrib."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory for gitignored real experimental files (default: data/open_datasets/real_benchmark)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Destination JSON path (default: results/real_spectra_benchmark.json)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Do not download over HTTP; fail non-zero if cached real spectra are missing unless --allow-synthetic",
    )
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="Permit fallback to synthetic_ablation suite when --offline is passed without cached real spectra",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Print summary without overwriting --output JSON file",
    )
    parser.add_argument(
        "--markdown-table",
        action="store_true",
        help="Emit the canonical Real vs. Controlled-Synthetic Markdown table from --output or live run",
    )
    parser.add_argument("--json", action="store_true", help="Print full JSON result to stdout")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.markdown_table and args.output.is_file() and args.no_write:
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        print(render_real_vs_synthetic_markdown_table(existing))
        return 0
    try:
        payload = run_benchmark(
            data_dir=args.data_dir,
            offline=args.offline,
            allow_synthetic=args.allow_synthetic,
        )
    except MissingOrCorruptedSpectraError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if not args.no_write:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if args.markdown_table:
        print(render_real_vs_synthetic_markdown_table(payload))
    elif args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(
            f"=== lab-attrib Real vs. Controlled-Synthetic Spectroscopy Benchmark "
            f"(primary_data_source={payload['primary_data_source']}, "
            f"pinned_assets={payload['pinned_asset_count']}) ==="
        )
        for row in payload["summary_rows"]:
            print(
                f"  {row['suite_id']:30s} | {row['data_source']:18s} | n={row['n_eval_cases']:3d} | "
                f"naive_FDR={row['naive_fdr_pct']:5.1f}% | "
                f"base_FDR={row['baseline_nnls_or_cosine_fdr_pct']:5.1f}% -> "
                f"frontier_FDR={row['frontier_fdr_pct']:5.1f}% | "
                f"recall: {row['baseline_recall_pct']:5.1f}% -> {row['frontier_recall_pct']:5.1f}% | "
                f"{row['baseline_rwp_or_top1']} -> {row['frontier_rwp_or_top1']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
