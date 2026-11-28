# Architecture and History — `lab-attrib`

> **Auto-Lab Ecosystem Role**: Layer `L4 · Post-Execution Causal Blame Calculus & Proof-Carrying Negative Ledger` in [Robot AI Auto-Lab (`auto-lab`)](https://github.com/yaniguan/auto-lab) ([Project Studio](https://yaniguan.github.io/auto-lab/)).

---

## 1. Object

### Problem Statement
Existing autonomous and self-driving laboratory (`SDL`) verifiers terminate at **pre-execution** physical safety and schema checks, leaving a severe epistemic blind spot **after** wet-lab execution completes:

1. **Analytical Ghost-Phase False Positives (`A-Lab` XRD controversy)**: Unreacted starting-precursor diffraction or optical peaks frequently overlap putative novel target peaks. Naive cosine or top-$k$ peak matchers misclassify precursor peaks as target formation, claiming false discoveries.
2. **Negative-Result Epistemic Conflation (`Sakana AI Scientist-v2` debate & `La Agente Optima` sensor faults)**: Robotic dispense slip, acoustic cavitation, tip clogs, missed incubation windows, or unsealed plate evaporation yield zero target signal. Naive pipelines write these corrupted runs directly into scientific memory (`assayfold`, Bayesian optimization surrogates, RLVR reward tables) as "true biochemical negative results", poisoning downstream active learning.
3. **Uncertified Stepping-Stone Conformal Drift**: Expanding an out-of-distribution conformal trust radius via annular `Chem-RRT` stepping-stone micro-probes ($\tau_t < d(x) \le (1 + \eta)\tau_t$) collapses if stepping-stone readouts themselves are physically corrupted or spectrally ambiguous.

`lab-attrib` defines the **fail-closed post-execution causal diagnostic gate and proof-carrying scientific ledger** across four tightly coupled components:
1. **4-Way Causal Blame Calculus & Fit-Method-Calibrated $R_{\mathrm{wp}}$ Gate** (`src/lab_attrib/blame.py` + `lean/LabAttribLean.lean`, **31 theorems**, **17 `#guard` checks**, `0 sorry`, `no custom axioms`),
2. **Shift-Tolerant Pseudo-Voigt `L1-NNLS` Precursor Peak-Overlap Deconvolver & MS/MS Spectral Entropy + Residual Chimera Engine** (`src/lab_attrib/deconvolution.py`, `src/lab_attrib/xrd_deconvolve.py`),
3. **Expected Value of Information (`EVOI`) Orthogonal Assay Router** (`src/lab_attrib/orthogonal_router.py`), and
4. **SHA-256 Hash-Chained Proof-Carrying Ledger & Attributed `Chem-RRT` Frontier Gate** (`src/lab_attrib/ledger.py`, `src/lab_attrib/attributed_frontier.py`).

### Role in the Auto-Lab `L0`–`L6` Hierarchy
Within the [Auto-Lab](https://github.com/yaniguan/auto-lab) Brain/Body architecture (`L0`–`L6`), `lab-attrib` resides in **`L4 · Digital-Twin Rehearsal, Pre/Post-Execution Verification & Causal Attribution`**:
- **Upstream (`L1`–`L2` Wet-Lab Body & Analytical Instruments)**: Ingests raw liquid-handling, thermal/incubation, plate-seal, control-well, and 1D diffraction/optical readout vectors immediately after benchtop execution ([`stepbot`](https://github.com/yaniguan/stepbot), [`twinstep`](https://github.com/yaniguan/twinstep)).
- **Peer (`L4` Verification Suite)**: Complements pre-execution verifiers [`lab-verify`](https://github.com/yaniguan/lab-verify) and [`lab-rehearsal`](https://github.com/yaniguan/lab-rehearsal) by closing the **post-execution** verification loop, backed by standalone Lean 4.36.0 proofs in `lean/LabAttribLean.lean`.
- **Downstream (`L0` Ledger & `L3`/`L5` Brain)**: Admits only `CERTIFIED_NEGATIVE` and `CERTIFIED_POSITIVE` records backed by explicit physical execution QC and analytical readout evidence into structured experiment memory ([`assayfold`](https://github.com/yaniguan/assayfold)), active-learning surrogates ([`lab-decide`](https://github.com/yaniguan/lab-decide)), and verifiable post-training reward pipelines ([`labbench-posttrain`](https://github.com/yaniguan/labbench-posttrain)).

---

## 2. Data

### Open Characterization & Solid-State Synthesis Registry (`src/lab_attrib/open_datasets.py`)
`src/lab_attrib/open_datasets.py` and `scripts/fetch_open_spectra.py` define a typed `OpenCharacterizationDataset` registry (`9` live-verified public archives across `XRD`, `Raman`, `LC-MS`, `1H-NMR`, and `Solubility`, fetched on-demand into gitignored `data/open_datasets/` with SHA-256 manifests):
1. **`rruff_powder_xrd` (`XRD`, `Apache-2.0`, `5,842` records)**: Open RRUFF mineral powder XRD XY & DIF scans (`AI4Spectro/rruff`).
2. **`rruff_raman_zenodo` (`Raman`, `CC-BY-4.0`, `21,490` records)**: Compiled RRUFF, WURM, and SLOPP/E Raman vibrational spectra (`Zenodo 15742626`) + public-domain Raman Open Database `.rod` records.
3. **`cod_crystallography` (`XRD`, `CC0-1.0`, `530,000` records)**: Crystallography Open Database public-domain CIF and powder diffraction reference structures (`crystallography.net/cod/`).
4. **`mp20_crystal_phases` (`XRD`, `CC-BY-4.0`, `45,231` records)**: Materials Project `MP-20` inorganic crystal phases ($\le 20$ atoms/unit cell).
5. **`alab_gpss_solid_state` (`XRD`, `CC-BY-4.0`, `352` records)**: Ceder Group `A-Lab GPSS` autonomous solid-state halide spinel synthesis release (`CederGroupHub/alab_gpss_public` & Zenodo `10.5281/zenodo.19396297`), parsed from PANalytical `.xrdml` scans (`load_xy_diffractogram`) and phase/peak-anomaly summaries (`load_alab_gpss_summary`, `calibrate_alab_rwp_thresholds`).
6. **`massspecgym_msms` (`LC-MS`, `MIT`, `231,104` records)**: NeurIPS 2024 `MassSpecGym` tandem MS/MS fragment spectra (`roman-bushuiev/MassSpecGym`) with official scaffold-disjoint `fold` (`train`/`val`/`test`) annotations (`load_massspecgym_spectra`).
7. **`nmrshiftdb2_nmr` (`1H-NMR`, `OPL-1.0`, `70,029` records)** & **`bmrb_metabolomics_nmr` (`1H-NMR`, `CC0-1.0`, `5,200` records)**: Open 1H/13C NMR reference archives (`nmrshiftdb2_nmr` marked `commercial_use_ok=False`; `bmrb_metabolomics_nmr` `commercial_use_ok=True`).
8. **`aqsoldb_solubility` (`Solubility`, `CC-BY-4.0`, `9,982` records)**: Curated aqueous solubility ($\log S$) and 2D physicochemical descriptors (`load_aqsoldb_csv`).
9. **Bundled Zero-Network Mineral/Oxide Reference Mini-Panel (`11` phases)**: Curated $\mathrm{Cu\,K\alpha}$ Bragg $2\theta$ peak tables paired with Raman shifts ($\mathrm{cm}^{-1}$) (`lab-attrib datasets --benchmark-panel`).

### Real Empirical Benchmark Input (Delaney ESOL & AqSolDB) + Committed Synthetic Fallback
`src/lab_attrib/attributed_frontier.py` evaluates closed-loop frontier discovery on:
1. **Delaney ESOL (`1,128` molecules, SHA-256 `8c06a76f0c6487d29ab0f903e6a7a7139f189ab3c1178f159c8be8964602f189`)**:
   - **Seed core (`657` molecules)**: Low-MW fragments (`Rings <= 2` and `MW <= 220 Da`) split per seed into seed surrogate training (`260`) and seed conformal calibration (`120`).
   - **Multi-ring frontier pool (`471` molecules)**: Higher-MW 2-ring bridges and all `204` multi-ring scaffolds (`Rings >= 3`, up to `8` rings), containing **`44` genuine soluble multi-ring leads** (`Rings >= 3` and $\log S \ge -3.50$).
2. **Deterministic Synthetic ESOL Fallback (`720` molecules, `benchmarks/results/lab_attrib_synthetic_fallback.json`)**:
   - Verified live on every GitHub Actions CI run (`lab-attrib benchmark --synthetic-fallback --check-drift benchmarks/results/lab_attrib_synthetic_fallback.json`) to guarantee zero code-vs-JSON drift.

---

## 3. Model

`lab-attrib` trains **no neural network weights**. Its verification architecture comprises four deterministic mathematical engines mirrored between Python (`src/lab_attrib/`) and Lean 4.36.0 (`lean/LabAttribLean.lean`):

### 3.1 Four-Way Post-Execution Blame Calculus & Fit-Method-Calibrated $R_{\mathrm{wp}}$ (`blame.py` & `LabAttribLean.lean`)
Given `AssayTelemetry` $t$:
- **Physical Execution Gate** (`executionClean`):
  $$\mathrm{dispenseErrorPermille} \le 50 \;\land\; \mathrm{aspirationClean} \;\land\; \mathrm{incubationWindowClean} \;\land\; \mathrm{sealIntact} \;\land\; \mathrm{positiveControlPassed} \;\land\; \mathrm{blankUncontaminated} \;\land\; \mathrm{controlWindowClean}$$
- **Analytical Identifiability Gate** (`characterizationIdentifiable`):
  $$\mathrm{detectorCalibrated} \;\land\; \mathrm{precursorOverlapPermille} \le 150 \;\land\; \mathrm{spectralResidualPermille} \le R_{\max}(t.\mathrm{fitMethod}) \;\land\; \mathrm{snrTenths} \ge 30$$
  where $R_{\max}(\texttt{rietveld}) = 100$ permille ($10.0\%$ full-profile Rietveld residual) and $R_{\max}(\texttt{stick\_nnls}) = 200$ permille ($20.0\%$ 1D isotropic-CIF stick-pattern NNLS residual, theorem `stick_nnls_calibrated_rwp_sound`).
- **352-Run `A-Lab GPSS` Split Calibration**: `load_alab_gpss_summary` stores `impurity_weight_permille` separately from crystallographic Bragg projection + `extra_peaks`/`missing_peaks` overlap (`precursor_overlap_permille`). Across `176` train (`even sample_index`) and `176` held-out (`odd sample_index`) runs, the Rietveld gate achieves **`100.00%` precision (`0.00%` FDR)**, **`69.81%` held-out positive recall**, and **`29.41%` held-out primary negative certification**, yielding **`231 CERTIFIED_POSITIVE`**, **`15 CERTIFIED_NEGATIVE`**, and **`106 AMBIGUOUS_CHARACTERIZATION`** across all `352` runs.

### 3.2 Shift-Tolerant Pseudo-Voigt `L1-NNLS` Deconvolver & MS/MS Chimera Gate (`deconvolution.py`, `xrd_deconvolve.py`)
Solves weighted sparsity-regularized non-negative multiphase decomposition (`elastic_nnls`) coupled with Golden-Section `2-theta` zero-point & Vegard lattice alignment and Pseudo-Voigt peak broadening to KKT violation $\le 10^{-10}$. For tandem MS/MS (`MassSpecGym`), pairs unweighted/weighted **Spectral Entropy Similarity** (Li et al., *Nature Methods* 2021) with convex two-component mixture entropy maximization AND library-free Q1 isolation-window dual-precursor + unexplained-fragment residual entropy checks (`detect_msms_chimera`), elevating leave-interferent-out chimera quarantine from `16.67%` (`25.00%` at `0.55` cutoff) to **`75.00%`** (`95.83%` when the interferent is also in the library).

### 3.3 Bayesian EVOI Orthogonal Router (`orthogonal_router.py`)
Evaluates `Raman` (`1.5 uL`), `LC-MS` (`4.0 uL`), `1H-NMR` (`15.0 uL`), and `TEM` (`1.0 uL`) against sample volume, phase state, fit-method-calibrated residual flags, and turnaround cost to maximize $\mathrm{NetEVOI}(m) = \mathrm{GrossEVOI}(m) - C_{\mathrm{total}}(m)$.

### 3.4 Closed-Loop Attributed `Chem-RRT` Benchmark & Sensor-Recall Sweep (`benchmarks/results/lab_attrib_benchmark.json`)
Across `5` seeds $\times$ `9` rounds $\times$ `20` primary assays/round on Delaney ESOL (`1,128` molecules), with unclamped `deconvolve_spectrum` outputs and ground-truth `poisoned_memory_entries` tracked across all five regimes:

| Regime | Verified Multi-Ring Hits | False Hits | FDR (`%`) | Poisoned Memory Entries | Total Measurements | 90% Conformal Coverage (`%`) | Frontier RMSE | Certified Ledger Entries |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `naive_raw_writeback` | `21.40` | `15.00` | `40.22%` | `66.20` | `180.00` | `66.80%` | `2.0715` | `0.00` |
| `pre_gate_only_static` | `0.00` | `0.00` | `0.00%` | `15.00` | `39.20` | `78.33%` | `4.4604` | `0.00` |
| `unattributed_frontier_probes` | `22.60` | `14.00` | `37.91%` | `64.20` | `180.00` | `76.01%` | `1.3852` | `0.00` |
| **`lab_attrib_equal_budget`** | **`17.60`** | **`0.00`** | **`0.00%`** | **`0.00`** | **`180.00`** | **`97.67%`** | **`1.6430`** | **`118.40`** |
| **`lab_attrib_full_closed_loop`** | **`27.00`** | **`0.00`** | **`0.00%`** | **`0.00`** | **`269.60`** | **`98.27%`** | **`1.1797`** | **`177.20`** |

- **Imperfect-Sensor Sweep (`fault_flag_recall = 1.0 / 0.9 / 0.8`)**:
  - `recall = 1.0`: Full-loop FDR `0.00%`, `0.00` poisoned memory entries (`180`-assay equal-budget: `0.00%` FDR, `0.00` poisoned, `17.60` hits).
  - `recall = 0.9`: Full-loop FDR `10.13%` (`3.00` false hits), `7.00` poisoned memory entries, RMSE `1.2122` (`180`-assay equal-budget: `9.88%` FDR, `5.20` poisoned, `17.40` hits).
  - `recall = 0.8`: Full-loop FDR `17.80%` (`6.00` false hits), `13.20` poisoned memory entries, RMSE `1.2301` (`180`-assay equal-budget: `16.92%` FDR, `9.00` poisoned, `17.20` hits).

### 3.5 Real Experimental vs. Controlled-Synthetic Spectroscopy Benchmark (`results/real_spectra_benchmark.json`)
Evaluated under equal per-suite scoring rules and 95% Wilson confidence intervals across **72 SHA-256-pinned public spectroscopy & crystal-structure files** (`scripts/benchmark_real_spectra.py`, failing closed under `--offline` without cached verified spectra unless `--allow-synthetic` is passed):

<!-- BEGIN REAL_VS_SYNTHETIC_SPECTRA_TABLE -->
| Suite ID | Modality & Corpus | `data_source` | Eval Cases | Naive FDR | Rigid NNLS / Entropy FDR | Frontier FDR (95% Wilson CI) | Baseline Recall | Frontier Recall (95% Wilson CI) | Baseline `Rwp` / Top-1 | Frontier `Rwp` / Top-1 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `alab_gpss_xrd` | XRD (A-Lab GPSS .xrdml, full gate; relaxed 25%->75%) | `real_experimental` | `8` | `33.33%` | `0.00%` | **`0.00%`** `[0.0, 56.1]` | `0.00%` | **`75.00%`** `[30.1, 95.4]` | `626.4 permille` | **`256.6 permille`** |
| `rruff_dara_powder_xrd` | XRD (RRUFF .dif + DARA .xrdml) | `real_experimental` | `8` | `40.91%` | `44.44%` | **`6.25%`** `[1.1, 28.3]` | `100.00%` | **`100.00%`** `[79.6, 100.0]` | `608.8 permille` | **`167.8 permille`** |
| `rruff_rod_raman` | Raman (Independent ROD/RRUFF Specimens) | `real_experimental` | `14` | `67.44%` | `33.33%` | **`27.78%`** `[12.5, 50.9]` | `100.00%` | **`92.86%`** `[68.5, 98.7]` | `371.9 permille` | **`287.0 permille`** |
| `massspecgym_msms` | LC-MS/MS (MassSpecGym val/test; held-out chimera 17%->75.0%, full-pool 95.8%) | `real_experimental` | `110` | `64.06%` | `53.52%` | **`36.73%`** `[24.7, 50.7]` | `33.82%` | **`41.89%`** `[31.3, 53.3]` | `46.51% Top-1` | **`60.47% Top-1`** |
| `perturbed_reference_stress` | Raman/XRD Perturbed Library & Superpositions | `real_reference_perturbed` | `14` | `53.06%` | `25.00%` | **`0.00%`** `[0.0, 13.8]` | `100.00%` | **`100.00%`** `[86.2, 100.0]` | `337.1 permille` | **`41.2 permille`** |
| `synthetic_controlled_ablation` | XRD Controlled Shift+Voigt Sweep | `synthetic_ablation` | `12` | `7.69%` | `39.02%` | **`0.00%`** `[0.0, 13.3]` | `100.00%` | **`100.00%`** `[86.7, 100.0]` | `739.6 permille` | **`71.0 permille`** |
<!-- END REAL_VS_SYNTHETIC_SPECTRA_TABLE -->

---

## 4. Training

- **Zero Parametric Neural Training**: `lab-attrib` requires no GPU and trains no deep neural weights. Closed-loop surrogate evaluation in `attributed_frontier.py` uses exact Cholesky/Gaussian-elimination quadratic ridge regression (`13` basis terms) paired with finite-sample split-conformal quantile estimation.
- **Verifiable Post-Training & Active-Learning Shield (`L0` / `L3`)**: `CertifiedLedger.export_training_pairs()` verifies the full SHA-256 chain and Lean-mirrored admissibility invariants before emitting supervised `(features, measured_value, is_positive_hit, lean_certificate_tag)` tuples.

---

## 5. History

| Commit | Date | Author | Branch / Ref | Summary |
|---|---|---|---|---|
| Initial release (`v0.1.0`) | `2026-11-23` | Yani Guan | `main` | Created `lab-attrib` (`15th` core Auto-Lab repository): 4-way causal blame calculus, dual-backend NNLS precursor deconvolver, Bayesian EVOI orthogonal assay router, SHA-256 `CertifiedLedger`, closed-loop Delaney ESOL `Chem-RRT` campaign, and Lean 4.36.0 formalization (`25` theorems). |
| Wire contract (`autolab-wire/v1`) | `2026-11-23` | Yani Guan | `main` | Added `autolab_wire.py` (`REPO_ID = "lab-attrib"`, `LAYER = "L4"`) and `tests/test_autolab_wire.py`. |
| Phase-2 audit & formal parity (`fix/phase2-audit-and-wire`) | `2026-11-24` | Yani Guan | `fix/phase2-audit-and-wire` | Closed issues `#1..#10`: weighted Householder-QR NNLS + shift/baseline subtraction, compound dual-fault diagnostics, correlated EVOI blindspots, Merkle + HMAC/Ed25519 `CertifiedLedger`, directional kNN `Chem-RRT` + ACI, and `lab-attrib-lean --batch-tsv` CLI (`30` theorems). |
| Open characterization & synthesis datasets (`feat/open-datasets-2026`) | `2026-11-25` | Yani Guan | `feat/open-datasets-2026` | Added `src/lab_attrib/open_datasets.py`, `scripts/fetch_open_spectra.py`, and `lab-attrib datasets` CLI (`9` open spectroscopy/crystallography/solubility datasets + `11`-phase bundled panel). |
| Frontier real-spectra benchmark, Pseudo-Voigt `L1-NNLS`, MS/MS entropy & Lean axiom CI (`feat/frontier-real-spectra-voigt-nnls-and-axioms`) | `2026-11-27` | Yani Guan | `feat/frontier-real-spectra-voigt-nnls-and-axioms` | Added `src/lab_attrib/deconvolution.py`, `scripts/benchmark_real_spectra.py` -> `results/real_spectra_benchmark.json`, `lean/CheckAxioms.lean` + `scripts/check_lean_axioms.py`, and `tests/test_readme_drift.py`. |
| Phase-3 audit resolution (`fix/audit-14-17-failclosed-raman-alab-negatives`) | `2026-11-28` | Yani Guan | `fix/audit-14-17-failclosed-raman-alab-negatives` | Closed issues `#14..#17`: fail-closed default-deny `autolab_wire.py` requiring both explicit execution QC and characterization readout (`defaulted_fields`, `MISSING_EXPERIMENTAL_EVIDENCE`), unclamped closed-loop benchmark with ground-truth `poisoned_memory_entries`, `lab_attrib_equal_budget` (`180` assays), imperfect-sensor `fault_flag_recall = 1.0 / 0.9 / 0.8` sweep, live CI synthetic-fallback drift guard, fit-method-calibrated `Rwp` gate (`SpectralFitMethod`: `rietveld` `100` vs `stick_nnls` `200`, `31` Lean theorems, `15` `CERTIFIED_NEGATIVE`s on 352-run `A-Lab GPSS`), equal spectroscopy scoring rules, `14` independent ROD Raman specimens, separated `real_reference_perturbed` suite, MassSpecGym `fold` split + held-out Q1/residual chimera gate (`75.0%` vs `95.8%`), and 95% Wilson CIs. |

---

## 6. Unique Properties

1. **Proof-Carrying Certified Negative Results**: Treats a negative experimental result not as the mere absence of signal, but as an affirmative formal proof that physical execution (`executionClean`) and analytical detection power (`characterizationIdentifiable`) both succeeded while target formation was absent (`LabAttribLean.certified_negative_sound`, verified on real `A-Lab GPSS` failed spinel syntheses).
2. **Formal + Numerical Resolution of the `A-Lab` Ghost-Peak Trap**: Combines KKT-convergent `L1-NNLS` multiphase spectral decomposition with fit-method-calibrated Lean 4 guards (`unidentifiable_spectrum_never_certified`, `stick_nnls_calibrated_rwp_sound`) and secondary orthogonal assay resolution (`orthogonal_refute_promotes_certified_negative`).
3. **Non-Laundering Secondary Characterization Guarantee**: Proves formally (`orthogonal_cannot_launder_corrupted_execution`) that even a clean secondary `LC-MS` or `Raman` measurement cannot launder a physically botched wet-lab execution into scientific memory.
4. **Certified Annular `Chem-RRT` Trust Expansion**: Couples micro-scale (`<= 10 uL`) stock-reagent annular containment ($\tau_t < d(x) \le (1 + \eta)\tau_t$) with mandatory post-execution certification (`valid_tau_expansion_requires_certified_micro_probe`), preserving `97.67%`–`98.27%` empirical conformal coverage vs `76.01%` unattributed shell probes on Delaney ESOL.
5. **Zero-Dependency Core + Exact Dual-Backend Parity**: Runs out-of-the-box on pure Python 3.10+ standard library while automatically accelerating NNLS spectral deconvolution with `numpy` when installed (`< 1e-8` numerical parity).

---

## 15-Repository Auto-Lab Wire Contract (`autolab-wire/v1`)

`autolab_wire.py` exposes `REPO_ID = "lab-attrib"`, `LAYER = "L4"`, `SCHEMA_VERSION = "autolab-wire/v1"`, and CWD-independent fail-closed `process_packet(packet: dict) -> dict`:

- **Execution & Feedback Topology**: `stepbot, lab-feet -> lab-attrib -> twinstep, labbench-posttrain, lab-decide`
- **Fail-Closed Default-Deny Evidence Contract**:
  - Requires **both** explicit non-defaulted execution QC (`dispense_error_permille`, `aspiration_clean`, `incubation_window_clean`, `seal_intact`) **and** explicit characterization readout (`xrd_mixture`, or `snr_tenths` + `target_signal_present` in `characterization` / `readout` / `telemetry_override` / `edge_runtime`).
  - Context-only packets (`candidate_smiles`, `candidate_features`, `parsed_spans`, `protocol`, `trace`), bare `measured_value` (`0.0`), and `lab-feet` passthrough/protocol-hash placeholders are quarantined (`halted = True`, `halt_reason = "L4_ATTRIB_QUARANTINE"`, `admit_to_memory = False`, `certified_ledger_entries = []`, fault `"MISSING_EXPERIMENTAL_EVIDENCE"`) unless `allow_default_telemetry = True` is explicitly opted in for demo mode.
  - Always emits `out["attribution"]["defaulted_fields"]` listing any telemetry/readout fields filled from defaults.
