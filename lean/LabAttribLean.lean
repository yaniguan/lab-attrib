/-!
# LabAttribLean: Formal L4 Post-Execution Causal Blame Calculus & Proof-Carrying Negative Ledger

Self-contained Lean 4.36.0 formalization for `lab-attrib` (`L4` Post-Execution Diagnostic Gate
and Certified Negative Ledger). Formalizes four tightly coupled verification modules:

1. **4-Way Mutually Exclusive & Exhaustive Blame Calculus** (`AssayTelemetry`,
   `AttributionOutcome`, `executionClean`, `characterizationIdentifiable`,
   `attributeAssayOutcome`, `admitToScientificMemory`):
   Separates robotic/incubation/batch-control execution faults (`Sakana AI Scientist-v2` /
   `La Agente Optima` zero-signal conflation) and analytical precursor ghost-peak overlap
   (`A-Lab` XRD controversy) from genuine certified negative and certified positive discoveries.
2. **Orthogonal-Assay Secondary Resolution** (`OrthogonalModality`, `OrthogonalAssayReadout`,
   `resolveAmbiguousSample`, `effectiveAttribution`):
   Promotes spectrally ambiguous samples to certified outcomes iff primary wet-lab execution was
   clean and an orthogonal analytical modality (`LC-MS`, `Raman`, `TEM`, `1H-NMR`) cleanly resolves
   target identity.
3. **Attributed Annular `Chem-RRT` Frontier Stepping-Stones** (`AttributedSteppingStone`,
   `checkAttributedFrontierUpdate`):
   Requires micro-scale (`<= 10 uL`) stock-reagent annular probes (`tau < d <= (1 + eta) * tau`)
   to achieve post-execution `admitToScientificMemory = true` before expanding conformal trust
   radius `tau_{t+1}`.
4. **Hash-Chained Certified Scientific Ledger** (`LedgerEntry`, `LedgerValid`,
   `appendCertifiedEntry`, `computeAbstractEntryHash`):
   Commits full sample-ID codepoints, all `AssayTelemetry` fields, and `Option OrthogonalAssayReadout`
   into every ledger link, proving by induction that append-only certified admission preserves
   hash-chain integrity and rejects telemetry tampering.
-/
namespace LabAttribLean

/-! ## 1. Threshold Constants (Exact Integer Permille & Tenths) -/

/-- Maximum tolerated volumetric dispense error in permille (50 = 5.0%). -/
def maxDispenseErrorPermille : Nat := 50

/-- Maximum tolerated precursor peak-overlap fraction on target channels in permille (150 = 15.0%). -/
def maxPrecursorOverlapPermille : Nat := 150

/-- Maximum tolerated Rietveld full-profile weighted residual R_wp in permille (100 = 10.0%). -/
def maxSpectralResidualPermille : Nat := 100

/-- Maximum tolerated 1D stick-pattern / isotropic-CIF NNLS residual R_wp in permille (200 = 20.0%). -/
def maxStickNnlsResidualPermille : Nat := 200

/-- Spectral profile fitting method governing the calibrated R_wp residual threshold. -/
inductive SpectralFitMethod where
  | rietveld  : SpectralFitMethod
  | stickNnls : SpectralFitMethod
  deriving DecidableEq, Repr

/-- Fit-method-calibrated R_wp residual threshold in permille (`100` Rietveld vs `200` stick-NNLS). -/
def maxResidualForFitMethod (m : SpectralFitMethod) : Nat :=
  match m with
  | .rietveld  => maxSpectralResidualPermille
  | .stickNnls => maxStickNnlsResidualPermille

/-- Minimum resolvable assay detection-power signal-to-noise ratio in tenths (30 = SNR 3.0). -/
def minResolvableSnrTenths : Nat := 30

/-- Maximum micro-scale stepping-stone probe volume in microliters (10 uL). -/
def maxSteppingStoneVolumeUl : Nat := 10

/-- Maximum single-step annular frontier expansion factor eta in permille (500 = 50.0%). -/
def maxExpansionEtaPermille : Nat := 500

/-- Minimum tolerated positive-control signal recovery in permille (800 = 80.0%). -/
def minPosControlRecoveryPermille : Nat := 800

/-- Minimum tolerated screening plate Z'-factor in tenths (50 = Z' 0.50). -/
def minZPrimeTenths : Int := 50

/-- Minimum tolerated positive-control separation z-score in tenths (20 = 2.0 sigma). -/
def minPosControlZscoreTenths : Int := 20

/-! ## 2. 4-Way Post-Execution Blame Calculus -/

/-- Post-execution robotic, environmental, control-well, and deconvolved spectral telemetry. -/
structure AssayTelemetry where
  dispenseErrorPermille           : Nat
  aspirationClean                 : Bool
  incubationWindowClean           : Bool
  sealIntact                      : Bool
  positiveControlPassed           : Bool
  blankUncontaminated             : Bool
  detectorCalibrated              : Bool
  precursorOverlapPermille        : Nat
  spectralResidualPermille        : Nat
  snrTenths                       : Nat
  targetSignalPresent             : Bool
  positiveControlRecoveryPermille : Nat := 1000
  plateZPrimeTenths               : Int := 70
  positiveControlZscoreTenths     : Int := 50
  fitMethod                       : SpectralFitMethod := .rietveld
  deriving DecidableEq, Repr

/-- Exhaustive 4-way causal attribution verdict after wet-lab execution. -/
inductive AttributionOutcome where
  | executionArtifact         : AttributionOutcome
  | ambiguousCharacterization : AttributionOutcome
  | certifiedNegative         : AttributionOutcome
  | certifiedPositive         : AttributionOutcome
  deriving DecidableEq, Repr

/-- Operational closed-loop routing action paired with each causal verdict. -/
inductive DiagnosticAction where
  | retryRobot         : DiagnosticAction
  | runOrthogonalAssay : DiagnosticAction
  | commitCertifiedNeg : DiagnosticAction
  | commitCertifiedHit : DiagnosticAction
  deriving DecidableEq, Repr

/-- Physical wet-lab execution predicate: liquid handling, incubation, seal, and control window. -/
def executionClean (t : AssayTelemetry) : Bool :=
  decide (t.dispenseErrorPermille ≤ maxDispenseErrorPermille) &&
  t.aspirationClean &&
  t.incubationWindowClean &&
  t.sealIntact &&
  t.positiveControlPassed &&
  t.blankUncontaminated &&
  decide (minPosControlRecoveryPermille ≤ t.positiveControlRecoveryPermille) &&
  decide (minZPrimeTenths ≤ t.plateZPrimeTenths) &&
  decide (minPosControlZscoreTenths ≤ t.positiveControlZscoreTenths)

/-- Analytical characterization predicate: calibrated detector, A-Lab precursor guard, fit-method R_wp, SNR. -/
def characterizationIdentifiable (t : AssayTelemetry) : Bool :=
  t.detectorCalibrated &&
  decide (t.precursorOverlapPermille ≤ maxPrecursorOverlapPermille) &&
  decide (t.spectralResidualPermille ≤ maxResidualForFitMethod t.fitMethod) &&
  decide (minResolvableSnrTenths ≤ t.snrTenths)

/-- Compound physical + analytical fault indicator (`dual_fault_detected`). -/
def dualFaultDetected (t : AssayTelemetry) : Bool :=
  !executionClean t && !characterizationIdentifiable t

/-- Deterministic post-execution causal attribution function. -/
def attributeAssayOutcome (t : AssayTelemetry) : AttributionOutcome :=
  if !executionClean t then
    .executionArtifact
  else if !characterizationIdentifiable t then
    .ambiguousCharacterization
  else if t.targetSignalPresent then
    .certifiedPositive
  else
    .certifiedNegative

/-- Closed-loop diagnostic router mapping each attribution outcome to its workflow action. -/
def routeDiagnosticAction (o : AttributionOutcome) : DiagnosticAction :=
  match o with
  | .executionArtifact         => .retryRobot
  | .ambiguousCharacterization => .runOrthogonalAssay
  | .certifiedNegative         => .commitCertifiedNeg
  | .certifiedPositive         => .commitCertifiedHit

/-- Default-deny scientific memory gate: only certified negatives and certified positives enter. -/
def admitToScientificMemory (o : AttributionOutcome) : Bool :=
  match o with
  | .executionArtifact         => false
  | .ambiguousCharacterization => false
  | .certifiedNegative         => true
  | .certifiedPositive         => true

/-- Theorem 1: `attributeAssayOutcome` induces an exact 4-way semantic partition of telemetry space. -/
theorem outcome_exhaustive_four_way (t : AssayTelemetry) :
    (attributeAssayOutcome t = .executionArtifact ∧ executionClean t = false) ∨
    (attributeAssayOutcome t = .ambiguousCharacterization ∧
      executionClean t = true ∧ characterizationIdentifiable t = false) ∨
    (attributeAssayOutcome t = .certifiedNegative ∧
      executionClean t = true ∧ characterizationIdentifiable t = true ∧
      t.targetSignalPresent = false) ∨
    (attributeAssayOutcome t = .certifiedPositive ∧
      executionClean t = true ∧ characterizationIdentifiable t = true ∧
      t.targetSignalPresent = true) := by
  unfold attributeAssayOutcome
  cases hExec : executionClean t with
  | false =>
    simp
  | true =>
    cases hIdent : characterizationIdentifiable t with
    | false =>
      simp
    | true =>
      cases t.targetSignalPresent <;> simp

/-- Theorem 2: Any run failing physical execution QC is strictly blocked from scientific memory. -/
theorem corrupted_run_never_poisons_memory (t : AssayTelemetry)
    (hCorrupt : executionClean t = false) :
    admitToScientificMemory (attributeAssayOutcome t) = false := by
  simp [attributeAssayOutcome, admitToScientificMemory, hCorrupt]

/-- Theorem 3 (Hardware Corollary): Any of the 6 physical wet-lab faults blocks memory writeback. -/
theorem hardware_fault_disjunction_never_poisons_memory (t : AssayTelemetry)
    (hFault :
      maxDispenseErrorPermille < t.dispenseErrorPermille ∨
      t.aspirationClean = false ∨
      t.incubationWindowClean = false ∨
      t.sealIntact = false ∨
      t.positiveControlPassed = false ∨
      t.blankUncontaminated = false) :
    admitToScientificMemory (attributeAssayOutcome t) = false := by
  apply corrupted_run_never_poisons_memory
  unfold executionClean
  rcases hFault with hDisp | hAsp | hInc | hSeal | hPos | hBlank
  · have hNotLe : ¬ (t.dispenseErrorPermille ≤ maxDispenseErrorPermille) := Nat.not_le.mpr hDisp
    simp [hNotLe]
  · simp [hAsp]
  · simp [hInc]
  · simp [hSeal]
  · simp [hPos]
  · simp [hBlank]

/-- Quantitative batch-control drift (`recovery < 800`, `Z' < 0.50`, or `z < 2.0`) blocks memory admission. -/
theorem control_drift_blocks_memory_admission (t : AssayTelemetry)
    (hDrift :
      t.positiveControlRecoveryPermille < minPosControlRecoveryPermille ∨
      t.plateZPrimeTenths < minZPrimeTenths ∨
      t.positiveControlZscoreTenths < minPosControlZscoreTenths) :
    executionClean t = false ∧
    admitToScientificMemory (attributeAssayOutcome t) = false := by
  have hExecFalse : executionClean t = false := by
    unfold executionClean
    rcases hDrift with hRec | hZp | hZs
    · have hNotLe : ¬ (minPosControlRecoveryPermille ≤ t.positiveControlRecoveryPermille) :=
        Nat.not_le.mpr hRec
      simp [hNotLe]
    · have hNotLe : ¬ (minZPrimeTenths ≤ t.plateZPrimeTenths) :=
        Int.not_le.mpr hZp
      simp [hNotLe]
    · have hNotLe : ¬ (minPosControlZscoreTenths ≤ t.positiveControlZscoreTenths) :=
        Int.not_le.mpr hZs
      simp [hNotLe]
  exact ⟨hExecFalse, corrupted_run_never_poisons_memory t hExecFalse⟩

/-- Dual-fault runs preserve `executionArtifact` memory quarantine while flagging both fault domains. -/
theorem dual_fault_quarantines_memory (t : AssayTelemetry)
    (hDual : dualFaultDetected t = true) :
    executionClean t = false ∧
    characterizationIdentifiable t = false ∧
    attributeAssayOutcome t = .executionArtifact ∧
    admitToScientificMemory (attributeAssayOutcome t) = false := by
  simp only [dualFaultDetected, Bool.and_eq_true, Bool.not_eq_true'] at hDual
  rcases hDual with ⟨hExec, hIdent⟩
  exact ⟨hExec, hIdent, by simp [attributeAssayOutcome, hExec],
    corrupted_run_never_poisons_memory t hExec⟩

/-- Theorem 4: Formal A-Lab precursor-overlap & fit-method R_wp guard — unidentifiable spectra are never certified. -/
theorem unidentifiable_spectrum_never_certified (t : AssayTelemetry)
    (hAmb :
      t.detectorCalibrated = false ∨
      maxPrecursorOverlapPermille < t.precursorOverlapPermille ∨
      maxResidualForFitMethod t.fitMethod < t.spectralResidualPermille ∨
      t.snrTenths < minResolvableSnrTenths) :
    attributeAssayOutcome t ≠ .certifiedPositive ∧
    attributeAssayOutcome t ≠ .certifiedNegative ∧
    admitToScientificMemory (attributeAssayOutcome t) = false := by
  have hIdentFalse : characterizationIdentifiable t = false := by
    unfold characterizationIdentifiable
    rcases hAmb with hDet | hPrec | hRwp | hSnr
    · simp [hDet]
    · have hNotLe : ¬ (t.precursorOverlapPermille ≤ maxPrecursorOverlapPermille) :=
        Nat.not_le.mpr hPrec
      simp [hNotLe]
    · have hNotLe : ¬ (t.spectralResidualPermille ≤ maxResidualForFitMethod t.fitMethod) :=
        Nat.not_le.mpr hRwp
      simp [hNotLe]
    · have hNotLe : ¬ (minResolvableSnrTenths ≤ t.snrTenths) :=
        Nat.not_le.mpr hSnr
      simp [hNotLe]
  unfold attributeAssayOutcome
  cases executionClean t <;> simp [hIdentFalse, admitToScientificMemory]

/-- Theorem 5: Soundness and completeness of Certified Negative attribution. -/
theorem certified_negative_sound (t : AssayTelemetry) :
    attributeAssayOutcome t = .certifiedNegative ↔
      executionClean t = true ∧
      characterizationIdentifiable t = true ∧
      t.targetSignalPresent = false := by
  unfold attributeAssayOutcome
  cases executionClean t <;> cases characterizationIdentifiable t <;> cases t.targetSignalPresent <;>
    simp

/-- Theorem 6: Soundness and completeness of Certified Positive attribution. -/
theorem certified_positive_sound (t : AssayTelemetry) :
    attributeAssayOutcome t = .certifiedPositive ↔
      executionClean t = true ∧
      characterizationIdentifiable t = true ∧
      t.targetSignalPresent = true := by
  unfold attributeAssayOutcome
  cases executionClean t <;> cases characterizationIdentifiable t <;> cases t.targetSignalPresent <;>
    simp

/-- Theorem 7: Primary assay admission iff both physical execution and characterization pass. -/
theorem admit_iff_exec_clean_and_identifiable (t : AssayTelemetry) :
    admitToScientificMemory (attributeAssayOutcome t) = true ↔
      executionClean t = true ∧ characterizationIdentifiable t = true := by
  unfold attributeAssayOutcome admitToScientificMemory
  cases executionClean t <;> cases characterizationIdentifiable t <;> cases t.targetSignalPresent <;>
    simp

/-- Fit-method R_wp calibration soundness: a clean stick-NNLS fit in `(100, 200]` permille is
    quarantined under Rietveld thresholds while certified under calibrated stick-NNLS thresholds. -/
theorem stick_nnls_calibrated_rwp_sound (t : AssayTelemetry)
    (hExec : executionClean t = true)
    (hDet : t.detectorCalibrated = true)
    (hPrec : t.precursorOverlapPermille ≤ maxPrecursorOverlapPermille)
    (hSnr : minResolvableSnrTenths ≤ t.snrTenths)
    (hAboveRietveld : maxSpectralResidualPermille < t.spectralResidualPermille)
    (hWithinStick : t.spectralResidualPermille ≤ maxStickNnlsResidualPermille) :
    attributeAssayOutcome { t with fitMethod := .rietveld } = .ambiguousCharacterization ∧
    admitToScientificMemory (attributeAssayOutcome { t with fitMethod := .stickNnls }) = true := by
  have hExecR : executionClean { t with fitMethod := .rietveld } = true := hExec
  have hExecS : executionClean { t with fitMethod := .stickNnls } = true := hExec
  have hNotLeR : ¬ (t.spectralResidualPermille ≤ maxSpectralResidualPermille) :=
    Nat.not_le.mpr hAboveRietveld
  have hIdentR : characterizationIdentifiable { t with fitMethod := .rietveld } = false := by
    simp [characterizationIdentifiable, maxResidualForFitMethod, hDet, hNotLeR]
  have hIdentS : characterizationIdentifiable { t with fitMethod := .stickNnls } = true := by
    simp [characterizationIdentifiable, maxResidualForFitMethod, hDet, hPrec, hWithinStick, hSnr]
  constructor
  · simp [attributeAssayOutcome, hExecR, hIdentR]
  · rw [admit_iff_exec_clean_and_identifiable]
    exact ⟨hExecS, hIdentS⟩

/-- Theorem 8: Physical execution artifacts always route to robotic hardware retry. -/
theorem execution_artifact_routes_to_hardware_retry (t : AssayTelemetry)
    (hCorrupt : executionClean t = false) :
    attributeAssayOutcome t = .executionArtifact ∧
    routeDiagnosticAction (attributeAssayOutcome t) = .retryRobot := by
  simp [attributeAssayOutcome, routeDiagnosticAction, hCorrupt]

/-- Theorem 9: Clean execution with unidentifiable spectrum always routes to orthogonal assay. -/
theorem ambiguous_routes_to_orthogonal_assay (t : AssayTelemetry)
    (hExec : executionClean t = true)
    (hIdent : characterizationIdentifiable t = false) :
    attributeAssayOutcome t = .ambiguousCharacterization ∧
    routeDiagnosticAction (attributeAssayOutcome t) = .runOrthogonalAssay := by
  simp [attributeAssayOutcome, routeDiagnosticAction, hExec, hIdent]

/-! ## 3. Orthogonal-Assay Secondary Resolution -/

/-- Orthogonal secondary analytical modalities used to resolve primary spectral ambiguity. -/
inductive OrthogonalModality where
  | lcMs   : OrthogonalModality
  | raman  : OrthogonalModality
  | temEds : OrthogonalModality
  | nmr1H  : OrthogonalModality
  deriving DecidableEq, Repr

/-- Secondary readout emitted by an orthogonal analytical instrument. -/
structure OrthogonalAssayReadout where
  modality             : OrthogonalModality
  instrumentCalibrated : Bool
  orthogonalSnrTenths  : Nat
  selectivityClean     : Bool
  targetConfirmed      : Bool
  deriving DecidableEq, Repr

/-- Secondary orthogonal readout quality gate. -/
def orthogonalReadoutClean (r : OrthogonalAssayReadout) : Bool :=
  r.instrumentCalibrated &&
  decide (minResolvableSnrTenths ≤ r.orthogonalSnrTenths) &&
  r.selectivityClean

/-- Resolves a primary-ambiguous sample using a secondary orthogonal readout,
    while never allowing secondary characterization to launder a failed physical run. -/
def resolveAmbiguousSample (primary : AssayTelemetry) (secondary : OrthogonalAssayReadout) :
    AttributionOutcome :=
  if !executionClean primary then
    .executionArtifact
  else if !orthogonalReadoutClean secondary then
    .ambiguousCharacterization
  else if secondary.targetConfirmed then
    .certifiedPositive
  else
    .certifiedNegative

/-- Combined primary + optional orthogonal attribution pipeline. -/
def effectiveAttribution (primary : AssayTelemetry)
    (secondaryOpt : Option OrthogonalAssayReadout := none) : AttributionOutcome :=
  match secondaryOpt with
  | none => attributeAssayOutcome primary
  | some sec =>
      match attributeAssayOutcome primary with
      | .ambiguousCharacterization => resolveAmbiguousSample primary sec
      | other => other

/-- Theorem 10: Clean orthogonal confirmation on a physically clean run promotes to Certified Positive. -/
theorem orthogonal_confirm_promotes_certified_positive
    (primary : AssayTelemetry) (secondary : OrthogonalAssayReadout)
    (hExec : executionClean primary = true)
    (hOrth : orthogonalReadoutClean secondary = true)
    (hConf : secondary.targetConfirmed = true) :
    resolveAmbiguousSample primary secondary = .certifiedPositive ∧
    admitToScientificMemory (resolveAmbiguousSample primary secondary) = true := by
  simp [resolveAmbiguousSample, admitToScientificMemory, hExec, hOrth, hConf]

/-- Theorem 11: Clean orthogonal refutation (e.g. Raman refuting XRD precursor ghost peak)
    promotes to Certified Negative. -/
theorem orthogonal_refute_promotes_certified_negative
    (primary : AssayTelemetry) (secondary : OrthogonalAssayReadout)
    (hExec : executionClean primary = true)
    (hOrth : orthogonalReadoutClean secondary = true)
    (hRefute : secondary.targetConfirmed = false) :
    resolveAmbiguousSample primary secondary = .certifiedNegative ∧
    admitToScientificMemory (resolveAmbiguousSample primary secondary) = true := by
  simp [resolveAmbiguousSample, admitToScientificMemory, hExec, hOrth, hRefute]

/-- Theorem 12: Unconfirmed/noisy orthogonal readouts or failed primary executions are never admitted. -/
theorem unconfirmed_orthogonal_never_admitted
    (primary : AssayTelemetry) (secondary : OrthogonalAssayReadout)
    (hFail : orthogonalReadoutClean secondary = false ∨ executionClean primary = false) :
    admitToScientificMemory (resolveAmbiguousSample primary secondary) = false := by
  unfold resolveAmbiguousSample
  rcases hFail with hOrthFalse | hExecFalse
  · cases executionClean primary <;> simp [hOrthFalse, admitToScientificMemory]
  · simp [hExecFalse, admitToScientificMemory]

/-- Theorem 13: Secondary characterization can never launder a physically corrupted wet-lab run. -/
theorem orthogonal_cannot_launder_corrupted_execution
    (primary : AssayTelemetry) (secondaryOpt : Option OrthogonalAssayReadout)
    (hCorrupt : executionClean primary = false) :
    effectiveAttribution primary secondaryOpt = .executionArtifact ∧
    admitToScientificMemory (effectiveAttribution primary secondaryOpt) = false := by
  cases secondaryOpt with
  | none =>
    simp [effectiveAttribution, attributeAssayOutcome, admitToScientificMemory, hCorrupt]
  | some _ =>
    simp [effectiveAttribution, attributeAssayOutcome, admitToScientificMemory, hCorrupt]

/-- Theorem 14: Exact admissibility characterization across primary + optional orthogonal resolution. -/
theorem effectiveAttribution_admissible_iff
    (primary : AssayTelemetry) (secondaryOpt : Option OrthogonalAssayReadout) :
    admitToScientificMemory (effectiveAttribution primary secondaryOpt) = true ↔
      executionClean primary = true ∧
      (characterizationIdentifiable primary = true ∨
        ∃ sec, secondaryOpt = some sec ∧ orthogonalReadoutClean sec = true) := by
  cases secondaryOpt with
  | none =>
    simp [effectiveAttribution, admit_iff_exec_clean_and_identifiable]
  | some sec =>
    unfold effectiveAttribution attributeAssayOutcome resolveAmbiguousSample admitToScientificMemory
    cases executionClean primary with
    | false =>
      simp
    | true =>
      cases characterizationIdentifiable primary with
      | false =>
        cases hOrth : orthogonalReadoutClean sec with
        | false => simp [hOrth]
        | true => cases hConf : sec.targetConfirmed <;> simp [hOrth, hConf]
      | true =>
        cases primary.targetSignalPresent <;> simp

/-- Every admissible effective attribution resolves strictly to CertifiedNegative or CertifiedPositive. -/
theorem effectiveAttribution_admissible_outcomes
    (primary : AssayTelemetry) (secondaryOpt : Option OrthogonalAssayReadout)
    (hAdmit : admitToScientificMemory (effectiveAttribution primary secondaryOpt) = true) :
    effectiveAttribution primary secondaryOpt = .certifiedNegative ∨
    effectiveAttribution primary secondaryOpt = .certifiedPositive := by
  cases hOut : effectiveAttribution primary secondaryOpt <;>
    simp [hOut, admitToScientificMemory] at hAdmit ⊢

/-! ## 4. Attributed Annular `Chem-RRT` Frontier Stepping-Stones -/

/-- Stepping-stone frontier expansion proposal combining pre-execution micro-containment
    and post-execution causal attribution. -/
structure AttributedSteppingStone where
  tauPermille            : Nat
  etaPermille            : Nat
  distPermille           : Nat
  proposedNewTauPermille : Nat
  volumeUl               : Nat
  usesOnlyStockReagents  : Bool
  withinHazardEnvelope   : Bool
  telemetry              : AssayTelemetry
  orthogonalReadout      : Option OrthogonalAssayReadout := none
  deriving DecidableEq, Repr

/-- Exact integer check for annular shell `tau < d <= (1 + eta) * tau` with `0 < tau` and `eta <= 0.50`. -/
def inAnnularShell (tauPermille etaPermille distPermille : Nat) : Bool :=
  decide (0 < tauPermille) &&
  decide (etaPermille ≤ maxExpansionEtaPermille) &&
  decide (tauPermille < distPermille) &&
  decide (1000 * distPermille ≤ (1000 + etaPermille) * tauPermille)

/-- Micro-scale stock-only containment envelope (`1 <= volumeUl <= 10`). -/
def microContainmentClean (p : AttributedSteppingStone) : Bool :=
  decide (1 ≤ p.volumeUl) &&
  decide (p.volumeUl ≤ maxSteppingStoneVolumeUl) &&
  p.usesOnlyStockReagents &&
  p.withinHazardEnvelope

/-- Full attributed frontier radius update gate: requires annular shell, micro-containment,
    and post-execution certified admissibility before advancing `tau_{t+1}`. -/
def checkAttributedFrontierUpdate (p : AttributedSteppingStone) : Bool :=
  inAnnularShell p.tauPermille p.etaPermille p.distPermille &&
  microContainmentClean p &&
  admitToScientificMemory (effectiveAttribution p.telemetry p.orthogonalReadout) &&
  decide (p.tauPermille < p.proposedNewTauPermille) &&
  decide (p.proposedNewTauPermille ≤ p.distPermille)

/-- Theorem 15: Deep OOD probes beyond `(1 + eta) * tau` are strictly blocked from expanding `tau`. -/
theorem deep_ood_stepping_stone_blocked (p : AttributedSteppingStone)
    (hDeepOod : (1000 + p.etaPermille) * p.tauPermille < 1000 * p.distPermille) :
    checkAttributedFrontierUpdate p = false := by
  unfold checkAttributedFrontierUpdate inAnnularShell
  have hNotLe : ¬ (1000 * p.distPermille ≤ (1000 + p.etaPermille) * p.tauPermille) :=
    Nat.not_le.mpr hDeepOod
  simp [hNotLe]

/-- Theorem 16: Physically corrupted stepping-stone probes never expand conformal radius `tau`. -/
theorem corrupted_probe_never_expands_tau (p : AttributedSteppingStone)
    (hCorrupt : executionClean p.telemetry = false) :
    checkAttributedFrontierUpdate p = false := by
  have hNoAdmit := (orthogonal_cannot_launder_corrupted_execution p.telemetry p.orthogonalReadout hCorrupt).2
  unfold checkAttributedFrontierUpdate
  simp [hNoAdmit]

/-- Theorem 17: Spectrally ambiguous stepping-stone probes without clean orthogonal confirmation
    never expand conformal radius `tau`. -/
theorem ambiguous_probe_never_expands_tau (p : AttributedSteppingStone)
    (hIdentFalse : characterizationIdentifiable p.telemetry = false)
    (hNoOrth :
      p.orthogonalReadout = none ∨
      ∃ sec, p.orthogonalReadout = some sec ∧ orthogonalReadoutClean sec = false) :
    checkAttributedFrontierUpdate p = false := by
  have hNotAdmit : admitToScientificMemory (effectiveAttribution p.telemetry p.orthogonalReadout) = false := by
    cases hBool : admitToScientificMemory (effectiveAttribution p.telemetry p.orthogonalReadout) with
    | false => rfl
    | true =>
      have hIff := (effectiveAttribution_admissible_iff p.telemetry p.orthogonalReadout).mp hBool
      rcases hIff with ⟨_, hCharOrOrth⟩
      rcases hCharOrOrth with hCharTrue | ⟨sec, hSome, hOrthTrue⟩
      · rw [hIdentFalse] at hCharTrue
        contradiction
      · rcases hNoOrth with hNone | ⟨sec2, hSome2, hOrthFalse⟩
        · rw [hNone] at hSome
          contradiction
        · rw [hSome] at hSome2
          injection hSome2 with hEq
          subst hEq
          rw [hOrthFalse] at hOrthTrue
          contradiction
  unfold checkAttributedFrontierUpdate
  simp [hNotAdmit]

/-- Theorem 18: Micro-volume containment — probes exceeding `10 uL` never expand `tau`. -/
theorem exceeding_micro_volume_blocks_tau_expansion (p : AttributedSteppingStone)
    (hVol : maxSteppingStoneVolumeUl < p.volumeUl) :
    checkAttributedFrontierUpdate p = false := by
  unfold checkAttributedFrontierUpdate microContainmentClean
  have hNotLe : ¬ (p.volumeUl ≤ maxSteppingStoneVolumeUl) := Nat.not_le.mpr hVol
  simp [hNotLe]

/-- Theorem 19: Every approved `tau_{t+1}` expansion is backed by a micro-scale (`<= 10 uL`),
    stock-only, annular stepping-stone probe with certified post-execution attribution. -/
theorem valid_tau_expansion_requires_certified_micro_probe (p : AttributedSteppingStone)
    (hOk : checkAttributedFrontierUpdate p = true) :
    p.volumeUl ≤ maxSteppingStoneVolumeUl ∧
    p.usesOnlyStockReagents = true ∧
    p.withinHazardEnvelope = true ∧
    p.tauPermille < p.distPermille ∧
    1000 * p.distPermille ≤ (1000 + p.etaPermille) * p.tauPermille ∧
    p.tauPermille < p.proposedNewTauPermille ∧
    p.proposedNewTauPermille ≤ p.distPermille ∧
    admitToScientificMemory (effectiveAttribution p.telemetry p.orthogonalReadout) = true ∧
    executionClean p.telemetry = true := by
  simp only [checkAttributedFrontierUpdate, inAnnularShell, microContainmentClean,
    Bool.and_eq_true, decide_eq_true_eq] at hOk
  simp_all [effectiveAttribution_admissible_iff]

/-! ## 5. Hash-Chained Proof-Carrying Certified Ledger (Full Telemetry Commitment) -/

/-- Immutable ledger record binding sample identity, raw telemetry, causal outcome, and hash chain. -/
structure LedgerEntry where
  index      : Nat
  sampleId   : String
  telemetry  : AssayTelemetry
  orthogonal : Option OrthogonalAssayReadout
  outcome    : AttributionOutcome
  prevHash   : Nat
  entryHash  : Nat
  deriving DecidableEq, Repr

/-- Genesis chain anchor hash. -/
def GenesisHash : Nat := 0

/-- Mersenne prime modulus `2^61 - 1` for collision-sensitive rolling hash commitment. -/
def HashModulus : Nat := 2305843009213693951

/-- Mix one natural number lane into the polynomial rolling hash accumulator. -/
def mixHashLane (acc : Nat) (val : Nat) : Nat :=
  (acc * 1000003 + val + 97) % HashModulus

/-- Bijective ZigZag encoding from `Int` to `Nat` for signed Z'-factor and z-score lanes. -/
def encodeSignedInt (z : Int) : Nat :=
  if 0 ≤ z then 2 * z.toNat else 2 * (-z).toNat - 1

/-- Encode `Bool` as distinct non-zero lane tag. -/
def encodeBoolLane (b : Bool) : Nat :=
  if b then 2 else 1

/-- Fold `s.length` and every Unicode codepoint of `s` into the hash accumulator. -/
def hashStringCodepoints (acc : Nat) (s : String) : Nat :=
  s.toList.foldl (fun st c => mixHashLane st c.toNat) (mixHashLane acc s.length)

/-- Encode `SpectralFitMethod` as distinct non-zero lane tag. -/
def encodeFitMethodLane (m : SpectralFitMethod) : Nat :=
  match m with
  | .rietveld  => 1
  | .stickNnls => 2

/-- Fold all 15 `AssayTelemetry` fields (including `fitMethod`) into the rolling hash accumulator. -/
def hashAssayTelemetry (acc : Nat) (t : AssayTelemetry) : Nat :=
  let s0  := mixHashLane acc t.dispenseErrorPermille
  let s1  := mixHashLane s0  (encodeBoolLane t.aspirationClean)
  let s2  := mixHashLane s1  (encodeBoolLane t.incubationWindowClean)
  let s3  := mixHashLane s2  (encodeBoolLane t.sealIntact)
  let s4  := mixHashLane s3  (encodeBoolLane t.positiveControlPassed)
  let s5  := mixHashLane s4  (encodeBoolLane t.blankUncontaminated)
  let s6  := mixHashLane s5  (encodeBoolLane t.detectorCalibrated)
  let s7  := mixHashLane s6  t.precursorOverlapPermille
  let s8  := mixHashLane s7  t.spectralResidualPermille
  let s9  := mixHashLane s8  t.snrTenths
  let s10 := mixHashLane s9  (encodeBoolLane t.targetSignalPresent)
  let s11 := mixHashLane s10 t.positiveControlRecoveryPermille
  let s12 := mixHashLane s11 (encodeSignedInt t.plateZPrimeTenths)
  let s13 := mixHashLane s12 (encodeSignedInt t.positiveControlZscoreTenths)
  mixHashLane s13 (encodeFitMethodLane t.fitMethod)

/-- Fold optional secondary `OrthogonalAssayReadout` into the rolling hash accumulator. -/
def hashOrthogonalOpt (acc : Nat) (orth : Option OrthogonalAssayReadout) : Nat :=
  match orth with
  | none => mixHashLane acc 0
  | some r =>
      let modCode := match r.modality with
        | .lcMs   => 11
        | .raman  => 12
        | .temEds => 13
        | .nmr1H  => 14
      let u0 := mixHashLane acc modCode
      let u1 := mixHashLane u0 (encodeBoolLane r.instrumentCalibrated)
      let u2 := mixHashLane u1 r.orthogonalSnrTenths
      let u3 := mixHashLane u2 (encodeBoolLane r.selectivityClean)
      mixHashLane u3 (encodeBoolLane r.targetConfirmed)

/-- Deterministic collision-sensitive abstract hash mixer committing `prevHash`, `index`,
    full `sampleId` codepoints, all `AssayTelemetry` fields, `Option OrthogonalAssayReadout`,
    and `AttributionOutcome`. -/
def computeAbstractEntryHash (index : Nat) (sampleId : String)
    (telemetry : AssayTelemetry) (orthogonal : Option OrthogonalAssayReadout)
    (outcome : AttributionOutcome) (prevHash : Nat) : Nat :=
  let outcomeCode := match outcome with
    | .executionArtifact         => 1
    | .ambiguousCharacterization => 2
    | .certifiedNegative         => 3
    | .certifiedPositive         => 4
  let h0 := mixHashLane prevHash index
  let h1 := hashStringCodepoints h0 sampleId
  let h2 := hashAssayTelemetry h1 telemetry
  let h3 := hashOrthogonalOpt h2 orthogonal
  (mixHashLane h3 outcomeCode) + 1

/-- Single-entry well-formedness invariant at `(expectedIndex, expectedPrevHash)`. -/
def EntryWellFormed (expectedIndex expectedPrevHash : Nat) (e : LedgerEntry) : Prop :=
  e.index = expectedIndex ∧
  e.prevHash = expectedPrevHash ∧
  e.outcome = effectiveAttribution e.telemetry e.orthogonal ∧
  admitToScientificMemory e.outcome = true ∧
  e.entryHash =
    computeAbstractEntryHash e.index e.sampleId e.telemetry e.orthogonal e.outcome e.prevHash

/-- Inductive hash-chain validity predicate parameterized by running index and predecessor hash. -/
def LedgerChainValidAux : Nat → Nat → List LedgerEntry → Prop
  | _, _, [] => True
  | idx, prevH, e :: rest =>
      EntryWellFormed idx prevH e ∧ LedgerChainValidAux (idx + 1) e.entryHash rest

/-- Top-level ledger validity starting from index `0` and `GenesisHash`. -/
def LedgerValid (chain : List LedgerEntry) : Prop :=
  LedgerChainValidAux 0 GenesisHash chain

/-- Returns the terminal hash of a ledger chain (or `prevH` if empty). -/
def lastChainHash : Nat → List LedgerEntry → Nat
  | prevH, [] => prevH
  | _, e :: rest => lastChainHash e.entryHash rest

/-- Default-deny append operation admitting only certified outcomes into the hash chain. -/
def appendCertifiedEntry (chain : List LedgerEntry) (sampleId : String)
    (t : AssayTelemetry) (orth : Option OrthogonalAssayReadout := none) :
    Option (List LedgerEntry) :=
  let out := effectiveAttribution t orth
  if admitToScientificMemory out then
    let idx := chain.length
    let prevH := lastChainHash GenesisHash chain
    let h := computeAbstractEntryHash idx sampleId t orth out prevH
    let entry : LedgerEntry := {
      index      := idx
      sampleId   := sampleId
      telemetry  := t
      orthogonal := orth
      outcome    := out
      prevHash   := prevH
      entryHash  := h
    }
    some (chain ++ [entry])
  else
    none

/-- Theorem 20: Unattributed (corrupted or unresolved ambiguous) runs are rejected by the ledger. -/
theorem unattributed_entry_rejected_by_ledger
    (chain : List LedgerEntry) (sampleId : String)
    (t : AssayTelemetry) (orth : Option OrthogonalAssayReadout)
    (hReject : admitToScientificMemory (effectiveAttribution t orth) = false) :
    appendCertifiedEntry chain sampleId t orth = none := by
  simp [appendCertifiedEntry, hReject]

/-- Structural append lemma decomposing `LedgerChainValidAux` at the tail of a chain. -/
theorem ledgerChainValidAux_append_singleton (chain : List LedgerEntry)
    (idx prevH : Nat) (e : LedgerEntry) :
    LedgerChainValidAux idx prevH (chain ++ [e]) ↔
      LedgerChainValidAux idx prevH chain ∧
      EntryWellFormed (idx + chain.length) (lastChainHash prevH chain) e := by
  induction chain generalizing idx prevH with
  | nil =>
    simp [LedgerChainValidAux, lastChainHash]
  | cons hd tl ih =>
    simp only [List.cons_append, LedgerChainValidAux, lastChainHash, List.length_cons]
    rw [ih (idx + 1) hd.entryHash]
    have hIdx : idx + 1 + tl.length = idx + (tl.length + 1) := by omega
    rw [hIdx]
    constructor
    · rintro ⟨hHd, hTl, hEnd⟩
      exact ⟨⟨hHd, hTl⟩, hEnd⟩
    · rintro ⟨⟨hHd, hTl⟩, hEnd⟩
      exact ⟨hHd, hTl, hEnd⟩

/-- Theorem 21: `appendCertifiedEntry` preserves `LedgerValid` across arbitrary append sequences. -/
theorem appendCertified_preserves_ledger_validity
    (chain chain' : List LedgerEntry) (sampleId : String)
    (t : AssayTelemetry) (orth : Option OrthogonalAssayReadout)
    (hValid : LedgerValid chain)
    (hAppend : appendCertifiedEntry chain sampleId t orth = some chain') :
    LedgerValid chain' := by
  dsimp only [appendCertifiedEntry] at hAppend
  cases hAdmit : admitToScientificMemory (effectiveAttribution t orth) with
  | false =>
    simp [hAdmit] at hAppend
  | true =>
    simp [hAdmit] at hAppend
    subst hAppend
    unfold LedgerValid at hValid ⊢
    rw [ledgerChainValidAux_append_singleton]
    refine ⟨hValid, ?_⟩
    simp [EntryWellFormed, hAdmit]

/-- Auxiliary invariant: every entry in `LedgerChainValidAux` is scientifically admissible. -/
theorem ledgerChainValidAux_entries_admissible
    (idx prevH : Nat) (chain : List LedgerEntry)
    (hValid : LedgerChainValidAux idx prevH chain) :
    ∀ e ∈ chain,
      admitToScientificMemory e.outcome = true ∧
      executionClean e.telemetry = true ∧
      (e.outcome = .certifiedNegative ∨ e.outcome = .certifiedPositive) := by
  induction chain generalizing idx prevH with
  | nil =>
    intro e hMem
    cases hMem
  | cons hd tl ih =>
    rcases hValid with ⟨hWell, hRest⟩
    rcases hWell with ⟨_, _, hOutEq, hAdmit, _⟩
    intro e hMem
    rcases List.mem_cons.mp hMem with rfl | hInTl
    · have hAdmitEff : admitToScientificMemory (effectiveAttribution e.telemetry e.orthogonal) = true := by
        rw [← hOutEq]
        exact hAdmit
      have hExecClean := (effectiveAttribution_admissible_iff e.telemetry e.orthogonal).mp hAdmitEff |>.1
      have hCertCases := effectiveAttribution_admissible_outcomes e.telemetry e.orthogonal hAdmitEff
      rw [← hOutEq] at hCertCases
      exact ⟨hAdmit, hExecClean, hCertCases⟩
    · exact ih (idx + 1) hd.entryHash hRest e hInTl

/-- Theorem 22: Every entry in a valid ledger is scientifically admissible, backed by clean
    robotic execution, and strictly classified as CertifiedNegative or CertifiedPositive. -/
theorem ledger_entries_all_scientifically_admissible
    (chain : List LedgerEntry) (hValid : LedgerValid chain) :
    ∀ e ∈ chain,
      admitToScientificMemory e.outcome = true ∧
      executionClean e.telemetry = true ∧
      (e.outcome = .certifiedNegative ∨ e.outcome = .certifiedPositive) :=
  ledgerChainValidAux_entries_admissible 0 GenesisHash chain hValid

/-- Theorem 23: Tampering with the head entry's `prevHash` immediately invalidates `LedgerValid`. -/
theorem tampered_prevHash_breaks_ledger_validity
    (e : LedgerEntry) (rest : List LedgerEntry)
    (hTamper : e.prevHash ≠ GenesisHash) :
    ¬ LedgerValid (e :: rest) := by
  intro hValid
  rcases hValid with ⟨⟨_, hPrevEq, _, _, _⟩, _⟩
  exact hTamper hPrevEq

/-- Tampering with `sampleId`, `telemetry`, or `orthogonal` without updating `entryHash`
    invalidates `LedgerValid`. -/
theorem tampered_entryHash_breaks_ledger_validity
    (e : LedgerEntry) (rest : List LedgerEntry)
    (hTamper : e.entryHash ≠
      computeAbstractEntryHash e.index e.sampleId e.telemetry e.orthogonal e.outcome e.prevHash) :
    ¬ LedgerValid (e :: rest) := by
  intro hValid
  rcases hValid with ⟨⟨_, _, _, _, hHashEq⟩, _⟩
  exact hTamper hHashEq

/-! ## 6. Concrete Compile-Time `#guard` Verification Suite & Telemetry-Tamper Theorem -/

/-- Canonical clean negative assay (true biochemical negative, clean execution, clean baseline). -/
def sampleCleanNegative : AssayTelemetry := {
  dispenseErrorPermille    := 12
  aspirationClean          := true
  incubationWindowClean    := true
  sealIntact               := true
  positiveControlPassed    := true
  blankUncontaminated      := true
  detectorCalibrated       := true
  precursorOverlapPermille := 20
  spectralResidualPermille := 35
  snrTenths                := 65
  targetSignalPresent      := false
}

/-- Canonical clean positive discovery (true hit, low precursor overlap, low R_wp). -/
def sampleCleanPositive : AssayTelemetry := {
  dispenseErrorPermille    := 15
  aspirationClean          := true
  incubationWindowClean    := true
  sealIntact               := true
  positiveControlPassed    := true
  blankUncontaminated      := true
  detectorCalibrated       := true
  precursorOverlapPermille := 45
  spectralResidualPermille := 42
  snrTenths                := 110
  targetSignalPresent      := true
}

/-- Robotic cavitation zero-signal fault (Sakana / La Agente Optima negative conflation trap). -/
def sampleCavitationFault : AssayTelemetry := {
  dispenseErrorPermille    := 180
  aspirationClean          := false
  incubationWindowClean    := true
  sealIntact               := true
  positiveControlPassed    := true
  blankUncontaminated      := true
  detectorCalibrated       := true
  precursorOverlapPermille := 10
  spectralResidualPermille := 25
  snrTenths                := 50
  targetSignalPresent      := false
}

/-- A-Lab XRD precursor ghost-peak false-positive trap (78.0% precursor overlap on target peaks). -/
def sampleALabGhostPeak : AssayTelemetry := {
  dispenseErrorPermille    := 18
  aspirationClean          := true
  incubationWindowClean    := true
  sealIntact               := true
  positiveControlPassed    := true
  blankUncontaminated      := true
  detectorCalibrated       := true
  precursorOverlapPermille := 780
  spectralResidualPermille := 60
  snrTenths                := 85
  targetSignalPresent      := true
}

/-- Clean Raman secondary readout refuting the A-Lab XRD ghost peak. -/
def sampleRamanRefutation : OrthogonalAssayReadout := {
  modality             := .raman
  instrumentCalibrated := true
  orthogonalSnrTenths  := 75
  selectivityClean     := true
  targetConfirmed      := false
}

/-- Clean LC-MS secondary readout confirming a genuinely active overshadowed hit. -/
def sampleLcMsConfirmation : OrthogonalAssayReadout := {
  modality             := .lcMs
  instrumentCalibrated := true
  orthogonalSnrTenths  := 92
  selectivityClean     := true
  targetConfirmed      := true
}

/-- Same-outcome telemetry forgery (`dispenseErrorPermille = 50`, `snrTenths = 30`) and
    same-length `sampleId` forgery (`"NEG-1"` vs `"FORGE"`) strictly change `computeAbstractEntryHash`. -/
theorem tampered_telemetry_breaks_entry_hash :
    computeAbstractEntryHash 0 "NEG-1" sampleCleanNegative none .certifiedNegative GenesisHash ≠
      computeAbstractEntryHash 0 "FORGE" sampleCleanNegative none .certifiedNegative GenesisHash ∧
    computeAbstractEntryHash 0 "NEG-1" sampleCleanNegative none .certifiedNegative GenesisHash ≠
      computeAbstractEntryHash 0 "NEG-1"
        { sampleCleanNegative with dispenseErrorPermille := 50, snrTenths := 30 }
        none .certifiedNegative GenesisHash := by
  decide

-- Guard 1: Clean negative -> certifiedNegative
#guard attributeAssayOutcome sampleCleanNegative == .certifiedNegative

-- Guard 2: Clean positive -> certifiedPositive
#guard attributeAssayOutcome sampleCleanPositive == .certifiedPositive

-- Guard 3: Cavitation zero-signal fault -> executionArtifact (never certifiedNegative!)
#guard attributeAssayOutcome sampleCavitationFault == .executionArtifact

-- Guard 4: Cavitation fault routes to retryRobot and is denied memory admission
#guard routeDiagnosticAction (attributeAssayOutcome sampleCavitationFault) == .retryRobot &&
       admitToScientificMemory (attributeAssayOutcome sampleCavitationFault) == false

-- Guard 5: A-Lab ghost peak -> ambiguousCharacterization (never certifiedPositive!)
#guard attributeAssayOutcome sampleALabGhostPeak == .ambiguousCharacterization

-- Guard 6: A-Lab ghost peak routes to runOrthogonalAssay and is denied memory admission
#guard routeDiagnosticAction (attributeAssayOutcome sampleALabGhostPeak) == .runOrthogonalAssay &&
       admitToScientificMemory (attributeAssayOutcome sampleALabGhostPeak) == false

-- Guard 7: Raman secondary refutation turns A-Lab ghost peak into certifiedNegative
#guard resolveAmbiguousSample sampleALabGhostPeak sampleRamanRefutation == .certifiedNegative

-- Guard 8: LC-MS secondary confirmation promotes ambiguous sample into certifiedPositive
#guard resolveAmbiguousSample sampleALabGhostPeak sampleLcMsConfirmation == .certifiedPositive

-- Guard 9: Orthogonal readout CANNOT launder a corrupted physical execution
#guard effectiveAttribution sampleCavitationFault (some sampleLcMsConfirmation) == .executionArtifact

-- Guard 10: Valid micro-scale (8 uL) certified annular stepping-stone expands tau (300 -> 360)
#guard checkAttributedFrontierUpdate {
  tauPermille            := 300
  etaPermille            := 250
  distPermille           := 360
  proposedNewTauPermille := 360
  volumeUl               := 8
  usesOnlyStockReagents  := true
  withinHazardEnvelope   := true
  telemetry              := sampleCleanNegative
} == true

-- Guard 11: Deep OOD stepping-stone (dist 420 > 1.25 * 300 = 375) is blocked
#guard checkAttributedFrontierUpdate {
  tauPermille            := 300
  etaPermille            := 250
  distPermille           := 420
  proposedNewTauPermille := 420
  volumeUl               := 8
  usesOnlyStockReagents  := true
  withinHazardEnvelope   := true
  telemetry              := sampleCleanNegative
} == false

-- Guard 12: Corrupted stepping-stone inside annular shell is blocked from expanding tau
#guard checkAttributedFrontierUpdate {
  tauPermille            := 300
  etaPermille            := 250
  distPermille           := 350
  proposedNewTauPermille := 350
  volumeUl               := 8
  usesOnlyStockReagents  := true
  withinHazardEnvelope   := true
  telemetry              := sampleCavitationFault
} == false

-- Guard 13: Ledger rejects unattributed ghost-peak run (`none`)
#guard (appendCertifiedEntry [] "ALAB-001" sampleALabGhostPeak none).isNone == true

-- Guard 14: Ledger admits certified negative + Raman-refuted negative with sequential indices
#guard match appendCertifiedEntry [] "NEG-001" sampleCleanNegative none with
  | none => false
  | some c1 =>
      match appendCertifiedEntry c1 "ALAB-001-RAMAN" sampleALabGhostPeak (some sampleRamanRefutation) with
      | none => false
      | some c2 => c2.length == 2 && (c2.getLast? |>.map (·.prevHash)) == (c2.head? |>.map (·.entryHash))

-- Guard 15: Quantitative positive-control recovery drift (< 800 permille) blocks memory admission
#guard attributeAssayOutcome { sampleCleanNegative with positiveControlRecoveryPermille := 620 } == .executionArtifact

-- Guard 16: Telemetry or same-length sampleId tampering changes abstract entry hash
#guard computeAbstractEntryHash 0 "NEG-1" sampleCleanNegative none .certifiedNegative GenesisHash !=
       computeAbstractEntryHash 0 "FORGE" sampleCleanNegative none .certifiedNegative GenesisHash

-- Guard 17: Stick-NNLS calibrated R_wp (140 permille) certifies clean hit while Rietveld (max 100) quarantines
#guard attributeAssayOutcome { sampleCleanPositive with spectralResidualPermille := 140, fitMethod := .rietveld } == .ambiguousCharacterization &&
       attributeAssayOutcome { sampleCleanPositive with spectralResidualPermille := 140, fitMethod := .stickNnls } == .certifiedPositive

end LabAttribLean
