# Architecture and History — `lab-attrib`

> **Auto-Lab Ecosystem Role**: Layer `L4 · Post-Execution Causal Blame Calculus & Proof-Carrying Negative Ledger` in [Robot AI Auto-Lab (`auto-lab`)](https://github.com/yaniguan/auto-lab) ([Project Studio](https://yaniguan.github.io/auto-lab/)).

---

## 1. Object

### Problem Statement
Existing autonomous and self-driving laboratory (`SDL`) verifiers terminate at **pre-execution** physical safety and schema checks, leaving a severe epistemic blind spot **after** wet-lab execution completes:

1. **Analytical Ghost-Phase False Positives (`A-Lab` XRD controversy)**: Unreacted starting-precursor diffraction or optical peaks frequently overlap putative novel target peaks. Naive cosine or top-$k$ peak matchers misclassify precursor peaks as target formation, claiming false discoveries.
2. **Negative-Result Epistemic Conflation (`Sakana AI Scientist-v2` debate & `La Agente Optima` sensor faults)**: Robotic dispense slip, acoustic cavitation, tip clogs, missed incubation windows, or unsealed plate evaporation yield zero target signal. Naive pipelines write these corrupted runs directly into scientific memory (`assayfold`, Bayesian optimization surrogates, RLVR reward tables) as "true biochemical negative results", poisoning downstream active learning.
3. **Uncertified Stepping-Stone Conformal Drift**: Expanding an out-of-distribution conformal trust radius via annular `Chem-RRT` stepping-stone micro-probes ($\tau_t < d(x) \le (1 + \eta)\tau_t$) collapses if stepping-stone readouts themselves are physically corrupted or spectrally ambiguous.

`lab-attrib` defines the **post-execution causal diagnostic gate and proof-carrying scientific ledger** across four tightly coupled components:
1. **4-Way Causal Blame Calculus** (`src/lab_attrib/blame.py` + `lean/LabAttribLean.lean`),
2. **Non-Negative Least-Squares (`NNLS`) Precursor Peak-Overlap & $R_{\mathrm{wp}}$ Deconvolver** (`src/lab_attrib/xrd_deconvolve.py`),
3. **Expected Value of Information (`EVOI`) Orthogonal Assay Router** (`src/lab_attrib/orthogonal_router.py`), and
4. **SHA-256 Hash-Chained Proof-Carrying Ledger & Attributed `Chem-RRT` Frontier Gate** (`src/lab_attrib/ledger.py`, `src/lab_attrib/attributed_frontier.py`).

### Role in the Auto-Lab `L0`–`L6` Hierarchy
Within the [Auto-Lab](https://github.com/yaniguan/auto-lab) Brain/Body architecture (`L0`–`L6`), `lab-attrib` resides in **`L4 · Digital-Twin Rehearsal, Pre/Post-Execution Verification & Causal Attribution`**:
- **Upstream (`L1`–`L2` Wet-Lab Body & Analytical Instruments)**: Ingests raw liquid-handling, thermal/incubation, plate-seal, control-well, and 1D diffraction/optical readout vectors immediately after benchtop execution ([`stepbot`](https://github.com/yaniguan/stepbot), [`twinstep`](https://github.com/yaniguan/twinstep)).
- **Peer (`L4` Verification Suite)**: Complements pre-execution verifiers [`lab-verify`](https://github.com/yaniguan/lab-verify) (4-layer SQL/proof/MCP default-deny gate) and [`lab-rehearsal`](https://github.com/yaniguan/lab-rehearsal) (stateful deck rehearsal) by closing the **post-execution** verification loop, backed by standalone Lean 4.36.0 proofs in `lean/LabAttribLean.lean`.
- **Downstream (`L0` Ledger & `L3`/`L5` Brain)**: Admits only `CERTIFIED_NEGATIVE` and `CERTIFIED_POSITIVE` records into structured experiment memory ([`assayfold`](https://github.com/yaniguan/assayfold)), active-learning surrogates ([`lab-decide`](https://github.com/yaniguan/lab-decide)), and verifiable post-training reward pipelines ([`labbench-posttrain`](https://github.com/yaniguan/labbench-posttrain)).

### Inputs, Outputs, and Scope Boundaries
- **Inputs**:
  - Post-execution `AssayTelemetry` (volumetric dispense error in permille, aspiration cavitation status, incubation window status, seal integrity, positive/blank control outcomes, detector calibration, deconvolved `precursor_overlap_permille`, `spectral_residual_permille`, `snr_tenths`, and `target_signal_present`).
  - Optional raw 1D multi-channel spectra `(observed_y, precursor_spectra, target_spectrum)` for `deconvolve_spectrum(...)` and optional secondary `OrthogonalAssayReadout` (`LC-MS`, `Raman`, `TEM`, `1H-NMR`).
- **Outputs**:
  - Deterministic `AttributionReport` pairing `AttributionOutcome` (`EXECUTION_ARTIFACT`, `AMBIGUOUS_CHARACTERIZATION`, `CERTIFIED_NEGATIVE`, `CERTIFIED_POSITIVE`) with `DiagnosticAction` (`RETRY_ROBOT`, `RUN_ORTHOGONAL_ASSAY`, `COMMIT_CERTIFIED_NEGATIVE`, `COMMIT_CERTIFIED_HIT`) and the exact `LabAttribLean` theorem tag.
  - Ranked Bayesian `OrthogonalRoutingDecision` (`Net EVOI` in USD across feasible secondary instruments) when primary characterization is ambiguous.
  - Cryptographically tamper-evident `CertifiedLedgerEntry` chains (`SHA-256`) and validated $\tau_{t+1}$ annular trust-radius updates.
- **Scope Boundaries**:
  - `lab-attrib` performs symbolic, optimization-grounded (`NNLS`), Bayesian decision-theoretic (`EVOI`), and cryptographically chained post-execution verification. Hardware driver actuation lives in `L1`/`L2`, and pre-execution deck simulation lives in `lab-rehearsal`.

---

## 2. Data

### Open Characterization & Solid-State Synthesis Registry (`src/lab_attrib/open_datasets.py`)
`src/lab_attrib/open_datasets.py` and `scripts/fetch_open_spectra.py` define a typed `OpenCharacterizationDataset` registry (`9` live-verified public archives across `XRD`, `Raman`, `LC-MS`, `1H-NMR`, and `Solubility`, fetched on-demand into gitignored `data/open_datasets/` with SHA-256 manifests) plus multi-modal parsers directly feeding `deconvolve_spectrum`, `route_orthogonal_assay`, and `attributed_frontier`:
1. **`rruff_powder_xrd` (`XRD`, `Apache-2.0`, `5,842` records)**: Open RRUFF mineral powder XRD XY & DIF scans (`AI4Spectro/rruff`), parsed via `load_xy_diffractogram` and `deconvolve_xy_diffractogram`.
2. **`rruff_raman_zenodo` (`Raman`, `CC-BY-4.0`, `21,490` records)**: Compiled RRUFF, WURM, and SLOPP/E Raman vibrational spectra (`Zenodo 15742626`), parsed via `load_raman_spectrum` (`JCAMP-DX` `##XYDATA`/`##XYPOINTS`, RRUFF `##KEY=VALUE`, and CSV) into `evaluate_raman_orthogonal_readout`.
3. **`cod_crystallography` (`XRD`, `CC0-1.0`, `530,000` records)**: Crystallography Open Database public-domain CIF and powder diffraction reference structures (`crystallography.net/cod/`).
4. **`mp20_crystal_phases` (`XRD`, `CC-BY-4.0`, `45,231` records)**: Materials Project `MP-20` inorganic crystal phases ($\le 20$ atoms/unit cell) for competing polymorph screening.
5. **`alab_gpss_solid_state` (`XRD`, `CC-BY-4.0`, `352` records)**: Ceder Group `A-Lab GPSS` autonomous solid-state halide spinel synthesis release (`CederGroupHub/alab_gpss_public` & Zenodo `10.5281/zenodo.19396297`), natively parsed from PANalytical `.xrdml` scans (`load_xy_diffractogram`) and phase/residual summaries (`load_alab_gpss_summary`).
6. **`massspecgym_msms` (`LC-MS`, `MIT`, `231,104` records)**: NeurIPS 2024 `MassSpecGym` tandem MS/MS fragment spectra (`roman-bushuiev/MassSpecGym`), parsed from `.mgf`, `.tsv`, and `.json`/`.jsonl` (`load_massspecgym_spectra`) into `evaluate_msms_orthogonal_readout` (including co-eluting isobaric chimera selectivity blocking).
7. **`nmrshiftdb2_nmr` (`1H-NMR`, `OPL-1.0`, `70,029` records)** & **`bmrb_metabolomics_nmr` (`1H-NMR`, `CC0-1.0`, `5,200` records)**: Open 1H/13C NMR reference archives (`nmrshiftdb2_nmr` explicitly marked `commercial_use_ok=False` under `OPL-1.0`; `bmrb_metabolomics_nmr` `commercial_use_ok=True` under `CC0-1.0`).
8. **`aqsoldb_solubility` (`Solubility`, `CC-BY-4.0`, `9,982` records)**: Curated aqueous solubility ($\log S$) and 2D physicochemical descriptors (`DOI 10.7910/DVN/OVHAW8` & `maomlab/AqSolDB`), parsed via `load_aqsoldb_csv` directly into `MoleculeCandidate` pools for `run_attributed_frontier_benchmark`.
9. **Bundled Zero-Network Mineral/Oxide Reference Mini-Panel (`11` phases)**: Curated $\mathrm{Cu\,K\alpha}$ Bragg $2\theta$ peak tables paired with Raman shifts ($\mathrm{cm}^{-1}$) covering halide spinels (`LiCl`, `MgCl2`, `Li2MgCl4`), zirconia polymorphs (`ZrO2` cubic/tetragonal, `Y2O3`), titanate perovskites (`BaCO3`, `TiO2`, `BaTiO3`), `Calcite` (`R050048`), and `Alpha-Quartz` (`COD-9009666`) (`run_offline_reference_panel_benchmark`, `lab-attrib datasets --benchmark-panel`).

### Real Empirical Benchmark Input (Delaney ESOL & AqSolDB) + Deterministic Fallback
`src/lab_attrib/attributed_frontier.py` evaluates closed-loop frontier discovery on:
1. **Delaney ESOL (`1,128` molecules, read-only CSV when present)**:
   - Uses 6 continuous/discrete molecular descriptors (`Minimum Degree`, `Molecular Weight`, `Number of H-Bond Donors`, `Number of Rings`, `Number of Rotatable Bonds`, `Polar Surface Area`) and measured aqueous log solubility ($\log S$ in $\mathrm{mol/L}$).
   - **Seed core (`657` molecules)**: Low-MW fragments (`Rings <= 2` and `MW <= 220 Da`) split per seed into seed surrogate training (`260`) and seed conformal calibration (`120`).
   - **Multi-ring frontier pool (`471` molecules)**: Higher-MW 2-ring bridges and all `204` multi-ring scaffolds (`Rings >= 3`, up to `8` rings), containing **`44` genuine soluble multi-ring leads** (`Rings >= 3` and $\log S \ge -3.50$).
2. **Deterministic Synthetic ESOL Fallback (`720` molecules)**:
   - Automatically engaged when no external CSV is supplied (e.g., standalone CI on `ubuntu-latest`), preserving identical physicochemical descriptor semantics, ring distribution, and multi-regime invariants with zero external file dependencies.

### Synthetic Spectral & Fault Telemetry Suite
- **28- to 48-channel Gaussian diffraction/optical spectra** (`build_gaussian_spectrum`) modeling clean precursors, target-overlapping ghost precursors (`A-Lab` regime), clean target hits, zero-signal baselines, and unaccounted third-party polymorphs (high $R_{\mathrm{wp}}$).
- **Seeded Wet-Lab Fault Injection**: Primary runs experience realistic `22%` physical execution faults (dispense slip, acoustic cavitation, incubation excursion, seal breach -> raw readout $\approx -8.65$) and `18%` precursor ghost-peak interference on physically clean runs (naive cosine $> 0.85$ -> apparent readout $\approx -1.85$).

---

## 3. Model

`lab-attrib` trains **no neural network weights**. Its verification architecture comprises four deterministic mathematical engines mirrored between Python (`src/lab_attrib/`) and Lean 4.36.0 (`lean/LabAttribLean.lean`):

### 3.1 Four-Way Post-Execution Blame Calculus (`blame.py` & `LabAttribLean.lean`)
Given `AssayTelemetry` $t$:
- **Physical Execution Gate** (`executionClean`):
  $$\mathrm{dispenseErrorPermille} \le 50 \;\land\; \mathrm{aspirationClean} \;\land\; \mathrm{incubationWindowClean} \;\land\; \mathrm{sealIntact} \;\land\; \mathrm{positiveControlPassed} \;\land\; \mathrm{blankUncontaminated}$$
- **Analytical Identifiability Gate** (`characterizationIdentifiable`):
  $$\mathrm{detectorCalibrated} \;\land\; \mathrm{precursorOverlapPermille} \le 150 \;\land\; \mathrm{spectralResidualPermille} \le 100 \;\land\; \mathrm{snrTenths} \ge 30$$
- **Sequential Partition**:
  - $\neg \mathrm{executionClean}(t) \implies \texttt{EXECUTION\_ARTIFACT} \to \texttt{RETRY\_ROBOT}$ (`admit = False`)
  - $\mathrm{executionClean}(t) \land \neg \mathrm{characterizationIdentifiable}(t) \implies \texttt{AMBIGUOUS\_CHARACTERIZATION} \to \texttt{RUN\_ORTHOGONAL\_ASSAY}$ (`admit = False`)
  - $\mathrm{executionClean}(t) \land \mathrm{characterizationIdentifiable}(t) \land \neg \mathrm{targetSignalPresent}(t) \implies \texttt{CERTIFIED\_NEGATIVE} \to \texttt{COMMIT\_CERTIFIED\_NEGATIVE}$ (`admit = True`)
  - $\mathrm{executionClean}(t) \land \mathrm{characterizationIdentifiable}(t) \land \mathrm{targetSignalPresent}(t) \implies \texttt{CERTIFIED\_POSITIVE} \to \texttt{COMMIT\_CERTIFIED\_HIT}$ (`admit = True`)

### 3.2 Shift-Tolerant Pseudo-Voigt `L1-NNLS` Deconvolver & MS/MS Spectral Entropy (`deconvolution.py`, `xrd_deconvolve.py`)
Solves weighted sparsity-regularized non-negative multiphase decomposition (`elastic_nnls`) over precursor/library phases $A = [p_1, \dots, p_K, s_{\mathrm{target}}, B_{\mathrm{bg}}]$:
$$\min_{c \ge 0} \frac{1}{2} \sum_{i=1}^M w_i \left(y_i - (A c)_i\right)^2 + \sum_{j \in \mathcal{P}} \lambda_{1,j} c_j + \frac{1}{2} \sum_{j} \lambda_{2,j} c_j^2$$
via active-set Householder-QR + unit-diagonal coordinate descent to KKT violation $\le 10^{-10}$ (implemented in both NumPy and pure-Python stdlib to $< 10^{-8}$ parity), coupled with coarse-grid + continuous Golden-Section `2-theta` zero-point & solid-solution Vegard lattice alignment ($\pm 0.35^\circ$) and Pseudo-Voigt Gaussian/Lorentzian peak broadening ($V_p = \eta L + (1 - \eta) G$). For tandem MS/MS (`MassSpecGym`), pairs cosine matching with unweighted and low-entropy dynamically weighted **Spectral Entropy Similarity** ($w = 0.25 + 0.25 S$ for $S < 3$, Li et al., *Nature Methods* 2021) plus convex two-component mixture entropy maximization to quarantine near-isobaric co-eluting chimeras. Computes weighted profile residual $R_{\mathrm{wp}}$ and precursor peak-shadowing ratio:
$$\rho_{\mathrm{overlap}} = \frac{\langle \hat{y}_{\mathrm{prec}}, \hat{s} \rangle}{\langle \hat{y}_{\mathrm{prec}}, \hat{s} \rangle + \max(\langle \hat{y}_{\mathrm{target}}, \hat{s} \rangle, I_{\mathrm{floor}})} \in [0, 1].$$

### 3.3 Bayesian EVOI Orthogonal Router (`orthogonal_router.py`)
Evaluates `Raman` (`1.5 uL`), `LC-MS` (`4.0 uL`), `1H-NMR` (`15.0 uL`), and `TEM` (`1.0 uL`) against available sample volume, solid/solution phase compatibility, ambiguity mode, and turnaround cost to maximize $\mathrm{NetEVOI}(m) = \mathrm{GrossEVOI}(m) - C_{\mathrm{total}}(m)$ relative to default-deny baseline utility $U_0 = 0$.

### 3.4 Closed-Loop Attributed `Chem-RRT` Benchmark (`benchmarks/results/lab_attrib_benchmark.json`)
Across `5` seeds $\times$ `9` rounds $\times$ `20` assays/round (`180` primary executions/seed) on Delaney ESOL (`1,128` molecules):

| Regime | Verified Multi-Ring Hits | False Hits | FDR (`%`) | Poisoned Memory Entries | 90% Conformal Coverage (`%`) | Frontier RMSE | Certified Ledger Entries |
|---|---:|---:|---:|---:|---:|---:|---:|
| `naive_raw_writeback` | `21.40` | `15.00` | `40.22%` | `66.20` | `66.80%` | `2.0715` | `0.00` |
| `pre_gate_only_static` | `0.00` | `0.00` | `0.00%` | `15.00` | `78.33%` | `4.4604` | `0.00` |
| `unattributed_frontier_probes` | `22.60` | `14.00` | `37.91%` | `64.20` | `76.01%` | `1.3852` | `0.00` |
| **`lab_attrib_full_closed_loop`** | **`31.00`** | **`0.00`** | **`0.00%`** | **`0.00`** | **`96.39%`** | **`1.2286`** | **`177.60`** |

### 3.5 Real Experimental vs. Controlled-Synthetic Spectroscopy Benchmark (`results/real_spectra_benchmark.json`)
Evaluated across **51 SHA-256-pinned public experimental spectroscopy & crystal-structure files** (`scripts/benchmark_real_spectra.py`, gitignored `data/open_datasets/real_benchmark/`, failing closed under `--offline` without cached verified spectra unless `--allow-synthetic` is explicitly passed):

<!-- BEGIN REAL_VS_SYNTHETIC_SPECTRA_TABLE -->
| Suite ID | Modality & Corpus | `data_source` | Eval Cases | Naive FDR | Rigid NNLS / Cosine FDR | Frontier `L1-NNLS` / Entropy FDR | Baseline Recall | Frontier Recall | Baseline `Rwp` / Top-1 | Frontier `Rwp` / Top-1 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `alab_gpss_xrd` | XRD (A-Lab GPSS .xrdml) | `real_experimental` | `8` | `50.00%` | `0.00%` | **`0.00%`** | `0.00%` | **`75.00%`** | `641.2 permille` | **`258.0 permille`** |
| `rruff_dara_powder_xrd` | XRD (RRUFF .dif/.xy + DARA .xrdml) | `real_experimental` | `8` | `50.00%` | `48.28%` | **`6.25%`** | `100.00%` | **`100.00%`** | `649.4 permille` | **`199.6 permille`** |
| `rruff_rod_raman` | Raman (RRUFF/ROD .rod + JCAMP-DX) | `real_experimental` | `12` | `50.00%` | `40.62%` | **`0.00%`** | `100.00%` | **`100.00%`** | `309.5 permille` | **`13.2 permille`** |
| `massspecgym_msms` | LC-MS/MS (MassSpecGym) | `real_experimental` | `206` | `35.20%` | `28.68%` | **`12.60%`** | `51.27%` | **`66.47%`** | `69.23% Top-1` | **`82.42% Top-1`** |
| `synthetic_controlled_ablation` | XRD Controlled Shift+Voigt Sweep | `synthetic_ablation` | `12` | `0.00%` | `50.00%` | **`0.00%`** | `100.00%` | **`100.00%`** | `739.6 permille` | **`71.1 permille`** |
<!-- END REAL_VS_SYNTHETIC_SPECTRA_TABLE -->

---

## 4. Training

- **Zero Parametric Neural Training**: `lab-attrib` requires no GPU and trains no deep neural weights. Closed-loop surrogate evaluation in `attributed_frontier.py` uses exact Cholesky/Gaussian-elimination quadratic ridge regression (`13` basis terms) paired with finite-sample split-conformal quantile estimation.
- **Verifiable Post-Training & Active-Learning Shield (`L0` / `L3`)**: `CertifiedLedger.export_training_pairs()` verifies the full SHA-256 chain and Lean-mirrored admissibility invariants before emitting supervised `(features, measured_value, is_positive_hit, lean_certificate_tag)` tuples, preventing corrupted negative results and ghost-peak false positives from contaminating [`assayfold`](https://github.com/yaniguan/assayfold), [`lab-decide`](https://github.com/yaniguan/lab-decide), and RLVR reward tables in [`labbench-posttrain`](https://github.com/yaniguan/labbench-posttrain).

---

## 5. History

| Commit | Date | Author | Branch / Ref | Summary |
|---|---|---|---|---|
| Initial release (`v0.1.0`) | `2026-11-23` | Yani Guan | `main` | Created `lab-attrib` (`15th` core Auto-Lab repository): implemented 4-way causal blame calculus (`blame.py`), dual-backend NNLS precursor peak-overlap deconvolver (`xrd_deconvolve.py`), Bayesian EVOI orthogonal assay router (`orthogonal_router.py`), SHA-256 proof-carrying `CertifiedLedger` (`ledger.py`), 4-regime closed-loop Delaney ESOL `Chem-RRT` campaign (`attributed_frontier.py`), standalone Lean 4.36.0 formalization (`lean/LabAttribLean.lean`, `25` machine-checked theorems, `14` `#guard` checks, `0 sorry`, `no custom axioms`), `42` pytest checks, and GitHub Actions CI (`python-tests` 3.10–3.12 + `lean-proofs`). |
| Wire contract (`autolab-wire/v1`) | `2026-11-23` | Yani Guan | `main` | Added `autolab_wire.py` (`REPO_ID = "lab-attrib"`, `LAYER = "L4"`) and `tests/test_autolab_wire.py`, wiring `stepbot, lab-feet -> lab-attrib -> twinstep, labbench-posttrain, lab-decide` with live NNLS `xrd_mixture` deconvolution, EVOI secondary routing, SHA-256 `CertifiedLedger` append, and `L4_ATTRIB_QUARANTINE` memory firewall. |
| Phase-2 audit & formal parity (`fix/phase2-audit-and-wire`) | `2026-11-24` | Yani Guan | `fix/phase2-audit-and-wire` | Closed issues `#1..#10`: added weighted Lawson-Hanson Householder-QR NNLS + Bragg 2-theta shift/strain alignment + Bernstein amorphous baseline subtraction + NaN/Inf/zero-weight guards (`xrd_deconvolve.py`), compound dual-fault diagnostics + quantitative control-drift guards (`blame.py`), correlated modality blindspots + unclamped Bayes risk EVOI (`orthogonal_router.py`), thread/file-locked `CertifiedLedger` with Merkle proofs & HMAC/Ed25519 seals (`ledger.py`), two-turn `L4_ATTRIB_QUARANTINE` orthogonal resolution & deconvolved SNR propagation (`autolab_wire.py`), directional kNN `Chem-RRT` support & split-calibration ACI (`attributed_frontier.py`), and compiled `lab-attrib-lean --batch-tsv` CLI with full telemetry hash commitment (`30` theorems, `67` pytest checks). |
| Open characterization & synthesis datasets (`feat/open-datasets-2026`) | `2026-11-25` | Yani Guan | `feat/open-datasets-2026` | Added `src/lab_attrib/open_datasets.py`, `scripts/fetch_open_spectra.py` (gitignored `data/`), `lab-attrib datasets` CLI subcommand, and `tests/test_open_datasets_2026.py`: integrated `9` live-verified open-source `XRD`, `Raman`, `LC-MS`, `1H-NMR`, and `Solubility` datasets (`RRUFF` XRD & Raman, `COD`, `MP-20`, `A-Lab GPSS`, `MassSpecGym`, `NMRShiftDB2`, `BMRB`, `AqSolDB`), bundled an `11`-phase zero-network oxide/halide/mineral Bragg $2\theta$ + Raman reference panel, and added multi-modal XY/XRDML, JCAMP-DX/CSV Raman, MGF/TSV/JSON MS/MS, and AqSolDB parsers (`75` pytest checks). |
| Frontier real-spectra benchmark, Pseudo-Voigt `L1-NNLS`, MS/MS entropy & Lean axiom CI (`feat/frontier-real-spectra-voigt-nnls-and-axioms`) | `2026-11-27` | Yani Guan | `feat/frontier-real-spectra-voigt-nnls-and-axioms` | Added `src/lab_attrib/deconvolution.py` (Golden-Section `2-theta` shift alignment, Pseudo-Voigt peak broadening, active-set + coordinate-descent `elastic_nnls` `L1/L2-NNLS`, and Li et al. *Nature Methods* 2021 unweighted/weighted MS/MS Spectral Entropy Similarity + near-isobaric chimera quarantine), native `.dif`/`.cif`/`.rod` parsers in `open_datasets.py`, `scripts/benchmark_real_spectra.py` -> `results/real_spectra_benchmark.json` (`51` SHA-256-pinned real `A-Lab GPSS`, `RRUFF`/`DARA`/`COD`, `RRUFF/ROD` Raman, and `MassSpecGym` files), fail-closed `L4_ATTRIB_SCHEMA_REJECT` telemetry validation in `autolab_wire.py`, automated `#print axioms` verification (`lean/CheckAxioms.lean`, `scripts/check_lean_axioms.py`, `30` theorems, `0 sorryAx`, `no custom axioms`), and `tests/test_readme_drift.py`. |

---

## 6. Unique Properties

1. **Proof-Carrying Certified Negative Results**: Treats a negative experimental result not as the mere absence of signal, but as an affirmative formal proof that physical execution (`executionClean`) and analytical detection power (`characterizationIdentifiable`) both succeeded while target formation was absent (`LabAttribLean.certified_negative_sound`).
2. **Formal + Numerical Resolution of the `A-Lab` Ghost-Peak Trap**: Combines KKT-convergent NNLS multiphase spectral decomposition (`xrd_deconvolve.py`) with a machine-checked Lean 4 guard (`unidentifiable_spectrum_never_certified`) and secondary orthogonal assay resolution (`orthogonal_refute_promotes_certified_negative`).
3. **Non-Laundering Secondary Characterization Guarantee**: Proves formally (`orthogonal_cannot_launder_corrupted_execution`) that even a clean secondary `LC-MS` or `Raman` measurement cannot launder a physically botched wet-lab execution into scientific memory.
4. **Certified Annular `Chem-RRT` Trust Expansion**: Couples micro-scale (`<= 10 uL`) stock-reagent annular containment ($\tau_t < d(x) \le (1 + \eta)\tau_t$) with mandatory post-execution certification (`valid_tau_expansion_requires_certified_micro_probe`), eliminating uncertified stepping-stone conformal drift (`96.39%` vs `76.01%` empirical 90% coverage on Delaney ESOL).
5. **Zero-Dependency Core + Exact Dual-Backend Parity**: Runs out-of-the-box on pure Python 3.10+ standard library while automatically accelerating NNLS spectral deconvolution with `numpy` when installed (verified to `< 1e-8` numerical parity).

---

## 15-Repository Auto-Lab Wire Contract (`autolab-wire/v1`)

`autolab_wire.py` exposes `REPO_ID = "lab-attrib"`, `LAYER = "L4"`, `SCHEMA_VERSION = "autolab-wire/v1"`, and CWD-independent `process_packet(packet: dict) -> dict` (`REPO_DIR = Path(__file__).resolve().parent`):

- **Execution & Feedback Topology**: `stepbot, lab-feet -> lab-attrib -> twinstep, labbench-posttrain, lab-decide`
- **Inputs Consumed**:
  - `edge_runtime` from `lab-feet` (`L0`) and execution state from `stepbot` (`L1`): `dispense_error_permille`, `aspiration_clean`, `incubation_window_clean`, `seal_intact` (with optional `telemetry_override`).
  - Optional `xrd_mixture` (`observed`, `target_basis`, `precursor_bases`), solved live via `lab_attrib.xrd_deconvolve.deconvolve_spectrum` to derive exact `precursor_overlap_permille` and `spectral_residual_permille` ($R_{\mathrm{wp}}$).
- **Invariants & Outputs Emitted**:
  - Short-circuits already-halted upstream `autolab-wire/v1` packets unchanged after deep-copy envelope initialization, and fails closed (`halted = True`, `halt_reason = "L4_ATTRIB_SCHEMA_REJECT"`, `admit_to_memory = False`) on any malformed, non-boolean, non-integer, negative, or non-finite (`NaN`/`+-Inf`) telemetry or envelope field.
  - Evaluates `diagnose_assay(telemetry)`:
    - On `CERTIFIED_POSITIVE` or `CERTIFIED_NEGATIVE` (`admit_to_memory = True`): appends a SHA-256 hash-chained record to `CertifiedLedger`, verifies chain integrity, and appends `{run_id, candidate_smiles, outcome, admit_to_memory: True, ledger_entry_sha256}` to `out["certified_ledger_entries"]` (`trace` status `"PASS"`).
    - On `AMBIGUOUS_CHARACTERIZATION`: invokes `route_orthogonal_assay` to select the Net-EVOI-optimal confirmatory instrument (`"Raman"`, `"LC-MS"`, `"TEM"`, or `"1H-NMR"`) and quarantines the packet (`halted = True`, `halt_reason = "L4_ATTRIB_QUARANTINE"`, `trace` status `"QUARANTINE"`).
    - On `EXECUTION_ARTIFACT`: routes to `RETRY_ROBOT` and quarantines the packet (`halted = True`, `halt_reason = "L4_ATTRIB_QUARANTINE"`, `trace` status `"QUARANTINE"`).
  - Populates `out["attribution"]` and appends bottom-up invariant `"corrupted_run_never_poisons_memory & unidentifiable_spectrum_never_certified"` to `out["bottom_reasoning_traces"]` for downstream `twinstep` (`L3`), `labbench-posttrain` (`L3`, paired with `labfill`), and `lab-decide` (`L6`) across the 15-repository stack (`lab-ears`, `lab-ask-why`, `lab-decide`, `lab-automation-atlas`, `buy-or-make`, `labshift`, `lab-rehearsal`, `lab-verify`, `stepbot`, `lab-hands`, `lab-feet`, `lab-attrib`, `twinstep`, `labfill`, `labbench-posttrain`, plus `assayfold` `L0` memory).
