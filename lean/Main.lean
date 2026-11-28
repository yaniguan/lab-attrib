import LabAttribLean

/-!
# `lab-attrib-lean`: Runtime Differential Verification Oracle (`--batch-tsv`)

Evaluates `LabAttribLean` predicates (`executionClean`, `characterizationIdentifiable`,
`attributeAssayOutcome`, `routeDiagnosticAction`, `admitToScientificMemory`,
`effectiveAttribution`, `checkAttributedFrontierUpdate`, and `computeAbstractEntryHash`)
over arbitrary batch TSV inputs for 1-to-1 runtime parity testing against Python `blame.py`.
-/

namespace LabAttribCli

open LabAttribLean

private def stripLineEndings (s : String) : String :=
  String.ofList (s.toList.filter (fun c => c != '\r' && c != '\n'))

private def splitOnChar (s : String) (sep : Char) : List String :=
  let rec loop (chars : List Char) (cur : List Char) (acc : List String) : List String :=
    match chars with
    | [] => (acc.reverse ++ [String.ofList cur.reverse])
    | c :: rest =>
        if c == sep then
          loop rest [] (String.ofList cur.reverse :: acc)
        else
          loop rest (c :: cur) acc
  loop s.toList [] []

private def parseBoolField (raw : String) : Except String Bool :=
  match raw with
  | "1" | "true" | "True" | "TRUE"    => .ok true
  | "0" | "false" | "False" | "FALSE" => .ok false
  | _ => .error s!"Invalid boolean token: {raw}"

private def parseNatField (raw : String) : Except String Nat :=
  match raw.toNat? with
  | some n => .ok n
  | none   => .error s!"Invalid Nat token: {raw}"

private def parseIntField (raw : String) : Except String Int :=
  match raw.toInt? with
  | some z => .ok z
  | none   => .error s!"Invalid Int token: {raw}"

private def parseModality (raw : String) : Except String OrthogonalModality :=
  match raw with
  | "LC-MS"  => .ok .lcMs
  | "Raman"  => .ok .raman
  | "TEM"    => .ok .temEds
  | "1H-NMR" => .ok .nmr1H
  | _        => .error s!"Invalid OrthogonalModality token: {raw}"

private def parseFitMethod (raw : String) : Except String SpectralFitMethod :=
  match raw with
  | "rietveld"   => .ok .rietveld
  | "stick_nnls" => .ok .stickNnls
  | _            => .error s!"Invalid SpectralFitMethod token: {raw}"

private def formatBool (b : Bool) : String :=
  if b then "true" else "false"

private def formatOutcome (o : AttributionOutcome) : String :=
  match o with
  | .executionArtifact         => "EXECUTION_ARTIFACT"
  | .ambiguousCharacterization => "AMBIGUOUS_CHARACTERIZATION"
  | .certifiedNegative         => "CERTIFIED_NEGATIVE"
  | .certifiedPositive         => "CERTIFIED_POSITIVE"

private def formatAction (a : DiagnosticAction) : String :=
  match a with
  | .retryRobot         => "RETRY_ROBOT"
  | .runOrthogonalAssay => "RUN_ORTHOGONAL_ASSAY"
  | .commitCertifiedNeg => "COMMIT_CERTIFIED_NEGATIVE"
  | .commitCertifiedHit => "COMMIT_CERTIFIED_HIT"

/-- Evaluate one 28- or 29-column TSV row and emit deterministic 12-column TSV output. -/
private def evaluateTsvRow (cols : List String) : Except String String := do
  let (coreCols, fitMethodRaw) ←
    match cols with
    | [rowId, dispS, aspS, incS, sealS, posS, blankS, detS, precS, rwpS, snrS, sigS,
       posRecS, zPrimeS, posZsS, hasOrthS, orthModS, orthCalS, orthSnrS, orthSelS, orthConfS,
       tauS, etaS, distS, propTauS, volS, stockS, hazS] =>
        pure ([rowId, dispS, aspS, incS, sealS, posS, blankS, detS, precS, rwpS, snrS, sigS,
               posRecS, zPrimeS, posZsS, hasOrthS, orthModS, orthCalS, orthSnrS, orthSelS, orthConfS,
               tauS, etaS, distS, propTauS, volS, stockS, hazS], "rietveld")
    | [rowId, dispS, aspS, incS, sealS, posS, blankS, detS, precS, rwpS, snrS, sigS,
       posRecS, zPrimeS, posZsS, hasOrthS, orthModS, orthCalS, orthSnrS, orthSelS, orthConfS,
       tauS, etaS, distS, propTauS, volS, stockS, hazS, fitMethodS] =>
        pure ([rowId, dispS, aspS, incS, sealS, posS, blankS, detS, precS, rwpS, snrS, sigS,
               posRecS, zPrimeS, posZsS, hasOrthS, orthModS, orthCalS, orthSnrS, orthSelS, orthConfS,
               tauS, etaS, distS, propTauS, volS, stockS, hazS], fitMethodS)
    | _ =>
        .error s!"Expected 28 or 29 tab-separated columns, got {cols.length}"

  match coreCols with
  | [rowId, dispS, aspS, incS, sealS, posS, blankS, detS, precS, rwpS, snrS, sigS,
     posRecS, zPrimeS, posZsS, hasOrthS, orthModS, orthCalS, orthSnrS, orthSelS, orthConfS,
     tauS, etaS, distS, propTauS, volS, stockS, hazS] =>
      let disp    ← parseNatField dispS
      let asp     ← parseBoolField aspS
      let inc     ← parseBoolField incS
      let sealOk  ← parseBoolField sealS
      let pos     ← parseBoolField posS
      let blank   ← parseBoolField blankS
      let det     ← parseBoolField detS
      let prec    ← parseNatField precS
      let rwp     ← parseNatField rwpS
      let snr     ← parseNatField snrS
      let sig     ← parseBoolField sigS
      let posRec  ← parseNatField posRecS
      let zPrime  ← parseIntField zPrimeS
      let posZs   ← parseIntField posZsS
      let hasOrth ← parseBoolField hasOrthS
      let fitM    ← parseFitMethod fitMethodRaw

      let telemetry : AssayTelemetry := {
        dispenseErrorPermille           := disp
        aspirationClean                 := asp
        incubationWindowClean           := inc
        sealIntact                      := sealOk
        positiveControlPassed           := pos
        blankUncontaminated             := blank
        detectorCalibrated              := det
        precursorOverlapPermille        := prec
        spectralResidualPermille        := rwp
        snrTenths                       := snr
        targetSignalPresent             := sig
        positiveControlRecoveryPermille := posRec
        plateZPrimeTenths               := zPrime
        positiveControlZscoreTenths     := posZs
        fitMethod                       := fitM
      }

      let orthOpt : Option OrthogonalAssayReadout ←
        if hasOrth then
          let m    ← parseModality orthModS
          let cal  ← parseBoolField orthCalS
          let osnr ← parseNatField orthSnrS
          let sel  ← parseBoolField orthSelS
          let conf ← parseBoolField orthConfS
          pure (some {
            modality             := m
            instrumentCalibrated := cal
            orthogonalSnrTenths  := osnr
            selectivityClean     := sel
            targetConfirmed      := conf
          })
        else
          pure none

      let tau     ← parseNatField tauS
      let eta     ← parseNatField etaS
      let dist    ← parseNatField distS
      let propTau ← parseNatField propTauS
      let vol     ← parseNatField volS
      let stock   ← parseBoolField stockS
      let haz     ← parseBoolField hazS

      let probe : AttributedSteppingStone := {
        tauPermille            := tau
        etaPermille            := eta
        distPermille           := dist
        proposedNewTauPermille := propTau
        volumeUl               := vol
        usesOnlyStockReagents  := stock
        withinHazardEnvelope   := haz
        telemetry              := telemetry
        orthogonalReadout      := orthOpt
      }

      let execOk   := executionClean telemetry
      let identOk  := characterizationIdentifiable telemetry
      let dualErr  := dualFaultDetected telemetry
      let primOut  := attributeAssayOutcome telemetry
      let primAct  := routeDiagnosticAction primOut
      let primAdm  := admitToScientificMemory primOut
      let effOut   := effectiveAttribution telemetry orthOpt
      let effAct   := routeDiagnosticAction effOut
      let effAdm   := admitToScientificMemory effOut
      let frontOk  := checkAttributedFrontierUpdate probe
      let entryH   := computeAbstractEntryHash 0 rowId telemetry orthOpt effOut GenesisHash

      let outFields := [
        rowId,
        formatBool execOk,
        formatBool identOk,
        formatBool dualErr,
        formatOutcome primOut,
        formatAction primAct,
        formatBool primAdm,
        formatOutcome effOut,
        formatAction effAct,
        formatBool effAdm,
        formatBool frontOk,
        toString entryH
      ]
      return String.intercalate "\t" outFields
  | _ =>
      .error "Unreachable column destructuring"

private partial def readAllStdinLines (stdin : IO.FS.Stream) (acc : List String) : IO (List String) := do
  let line ← stdin.getLine
  if line.isEmpty then
    return acc.reverse
  else
    readAllStdinLines stdin (line :: acc)

def runBatchTsv (rawLines : List String) : IO UInt32 := do
  let stdout ← IO.getStdout
  let stderr ← IO.getStderr
  let mut lineNo : Nat := 0
  for rawLine in rawLines do
    lineNo := lineNo + 1
    let cleaned := stripLineEndings rawLine
    if cleaned.isEmpty || cleaned.startsWith "#" || cleaned.startsWith "row_id\t" then
      continue
    let cols := splitOnChar cleaned '\t'
    match evaluateTsvRow cols with
    | .ok rendered =>
        stdout.putStrLn rendered
    | .error errMsg =>
        stderr.putStrLn s!"Line {lineNo}: {errMsg}"
        return 1
  return 0

end LabAttribCli

def main (args : List String) : IO UInt32 := do
  match args with
  | ["--version"] =>
      IO.println "lab-attrib-lean 0.1.0 (Lean 4.36.0)"
      return 0
  | ["--self-test"] =>
      let ok :=
        LabAttribLean.attributeAssayOutcome LabAttribLean.sampleCleanNegative == .certifiedNegative &&
        LabAttribLean.attributeAssayOutcome LabAttribLean.sampleCleanPositive == .certifiedPositive
      if ok then
        IO.println "PASS"
        return 0
      else
        IO.eprintln "FAIL"
        return 1
  | ["--batch-tsv"] | ["--batch-tsv", "-"] =>
      let stdin ← IO.getStdin
      let lines ← LabAttribCli.readAllStdinLines stdin []
      LabAttribCli.runBatchTsv lines
  | ["--batch-tsv", inputPath] =>
      let content ← IO.FS.readFile (System.FilePath.mk inputPath)
      let lines := content.splitOn "\n"
      LabAttribCli.runBatchTsv lines
  | _ =>
      IO.eprintln "Usage: lab-attrib-lean [--version | --self-test | --batch-tsv [path|-]]"
      return 2
