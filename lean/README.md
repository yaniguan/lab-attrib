# `LabAttribLean` — Formal Lean 4.36.0 Verification Package for `lab-attrib`

`LabAttribLean` (`lean/LabAttribLean.lean`) is a standalone, zero-dependency **Lean 4.36.0** formalization of the post-execution causal blame calculus, secondary orthogonal-assay resolution, attributed annular `Chem-RRT` frontier expansion, and hash-chained proof-carrying scientific ledger implemented in `src/lab_attrib/`.

## Guarantees (`0 sorry`, `no custom axioms`, `0 warnings`)

Running `lake build` and `lake env lean CheckAxioms.lean` inside `lean/` compiles both the `LabAttribLean` proof library and the `lab-attrib-lean` (`Main.lean`) `--batch-tsv` runtime differential oracle, machine-checking **30 formal theorems** and **16 `#guard` compile-time test assertions** across four verification layers with **`0 sorry`** (`zero sorryAx`) and **no custom axioms** (depending exclusively on Lean 4's standard foundational axioms `propext`, `Classical.choice`, and `Quot.sound`):

### 1. 4-Way Mutually Exclusive & Exhaustive Blame Calculus
- `outcome_exhaustive_four_way`: Proves that `attributeAssayOutcome` partitions all post-execution telemetry vectors into four mutually exclusive causal regimes (`executionArtifact`, `ambiguousCharacterization`, `certifiedNegative`, `certifiedPositive`).
- `corrupted_run_never_poisons_memory` & `hardware_fault_disjunction_never_poisons_memory`: Any liquid-handling dispense error (`> 50` permille), acoustic cavitation/clog, missed incubation window, broken seal, failed positive control, or contaminated blank is unconditionally denied admission to scientific memory.
- `unidentifiable_spectrum_never_certified`: Formal `A-Lab` precursor ghost-peak (`> 150` permille) and weighted profile residual $R_{\mathrm{wp}}$ (`> 100` permille) guard blocking both false discoveries and unpowered negatives.
- `certified_negative_sound`, `certified_positive_sound`, `admit_iff_exec_clean_and_identifiable`: Bi-directional soundness and completeness of certified scientific memory writeback.
- `execution_artifact_routes_to_hardware_retry` & `ambiguous_routes_to_orthogonal_assay`: Deterministic closed-loop diagnostic routing.

### 2. Orthogonal-Assay Secondary Resolution
- `orthogonal_confirm_promotes_certified_positive`: Clean secondary confirmation (`LC-MS`, `Raman`, `TEM`, `1H-NMR`) on a physically clean run promotes an ambiguous primary readout to `certifiedPositive`.
- `orthogonal_refute_promotes_certified_negative`: Clean secondary refutation converts an `A-Lab`-style precursor ghost-peak artifact into a `certifiedNegative`.
- `unconfirmed_orthogonal_never_admitted`, `orthogonal_cannot_launder_corrupted_execution`, `effectiveAttribution_admissible_iff`, `effectiveAttribution_admissible_outcomes`: Secondary characterization can never launder a physically botched wet-lab run into scientific memory.

### 3. Attributed Annular `Chem-RRT` Frontier Stepping-Stones
- `deep_ood_stepping_stone_blocked`: Probes beyond $(1 + \eta)\tau_t$ (`1000 * d > (1000 + eta) * tau`) never expand conformal trust radius $\tau_{t+1}$.
- `corrupted_probe_never_expands_tau` & `ambiguous_probe_never_expands_tau`: Eliminates uncertified stepping-stone drift caused by corrupted or spectrally ambiguous micro-probes.
- `exceeding_micro_volume_blocks_tau_expansion` & `valid_tau_expansion_requires_certified_micro_probe`: Every valid $\tau_{t+1}$ expansion is backed by a micro-scale (`<= 10 uL`), stock-reagent, hazard-bounded annular probe with `admitToScientificMemory = true`.

### 4. Hash-Chained Certified Scientific Ledger
- `unattributed_entry_rejected_by_ledger`: Default-deny rejection of unattributed runs at the ledger boundary.
- `ledgerChainValidAux_append_singleton` & `appendCertified_preserves_ledger_validity`: Inductive invariance of `LedgerValid` across arbitrary certified append sequences.
- `ledgerChainValidAux_entries_admissible`, `ledger_entries_all_scientifically_admissible`, `tampered_prevHash_breaks_ledger_validity`, `tampered_entryHash_breaks_ledger_validity`, `tampered_telemetry_breaks_entry_hash`: Every record in any valid ledger chain commits full `sampleId` codepoints, all 14 `AssayTelemetry` fields, and `Option OrthogonalAssayReadout` into `computeAbstractEntryHash`, preventing telemetry or sample-ID tampering.
- `control_drift_blocks_memory_admission` & `dual_fault_quarantines_memory`: Quantitative control-window drift (`recovery < 800`, `Z' < 0.50`, `z < 2.0`) and compound physical+analytical faults are unconditionally quarantined from scientific memory.

## Reproducing Locally

```bash
cd lean
lake build
```

Note: `.lake/` and `lake-manifest.json` are excluded from version control via `.gitignore` to keep the repository lightweight and reproducible from `lean-toolchain` (`leanprover/lean4:v4.36.0`).
