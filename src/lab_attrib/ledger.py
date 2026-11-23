"""Append-Only, SHA-256 Hash-Chained Proof-Carrying Negative & Positive Ledger.

Python implementation of `LabAttribLean.LedgerEntry`, `LedgerValid`, and
`appendCertifiedEntry`. Guarantees:
1. Default-deny admission: `EXECUTION_ARTIFACT` and unresolved `AMBIGUOUS_CHARACTERIZATION`
   runs raise `LedgerRejectionError` and can never enter scientific memory.
2. Cryptographic tamper-evidence: every entry commits via canonical JSON SHA-256 to its
   index, sample ID, molecular descriptors, measured readout, raw execution & spectral
   telemetry, optional orthogonal assay readout, causal attribution outcome, Lean theorem
   certificate tag, and predecessor `entry_hash`.
3. Clean downstream export: `export_training_pairs()` verifies the entire hash chain and
   re-evaluates every attribution predicate from raw telemetry before emitting training
   records for BO surrogates, `assayfold`, or RLVR post-training (`labbench-posttrain`).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

from lab_attrib.blame import (
    AssayTelemetry,
    AttributionOutcome,
    OrthogonalAssayReadout,
    admit_to_scientific_memory,
    effective_attribution,
    execution_clean,
    lean_certificate_tag_for,
)

GENESIS_HASH: str = "0" * 64


class LedgerRejectionError(ValueError):
    """Raised when attempting to append an uncertified or corrupted run to the ledger."""


class LedgerIntegrityError(RuntimeError):
    """Raised when cryptographic or semantic hash-chain verification detects tampering."""


def _canonical_entry_sha256(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CertifiedLedgerEntry:
    """Single immutable record in `CertifiedLedger`."""

    index: int
    sample_id: str
    candidate_features: dict[str, float]
    measured_value: float
    telemetry: AssayTelemetry
    orthogonal_readout: OrthogonalAssayReadout | None
    outcome: AttributionOutcome
    lean_certificate_tag: str
    prev_hash: str
    entry_hash: str

    def hash_payload(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "sample_id": self.sample_id,
            "candidate_features": {
                str(k): round(float(v), 8) for k, v in sorted(self.candidate_features.items())
            },
            "measured_value": round(float(self.measured_value), 8),
            "telemetry": self.telemetry.to_dict(),
            "orthogonal_readout": (
                self.orthogonal_readout.to_dict() if self.orthogonal_readout is not None else None
            ),
            "outcome": self.outcome.value,
            "lean_certificate_tag": self.lean_certificate_tag,
            "prev_hash": self.prev_hash,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.hash_payload()
        payload["entry_hash"] = self.entry_hash
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CertifiedLedgerEntry:
        orth_raw = data.get("orthogonal_readout")
        return cls(
            index=int(data["index"]),
            sample_id=str(data["sample_id"]),
            candidate_features={
                str(k): float(v) for k, v in dict(data["candidate_features"]).items()
            },
            measured_value=float(data["measured_value"]),
            telemetry=AssayTelemetry.from_dict(dict(data["telemetry"])),
            orthogonal_readout=(
                OrthogonalAssayReadout.from_dict(dict(orth_raw)) if orth_raw is not None else None
            ),
            outcome=AttributionOutcome(str(data["outcome"])),
            lean_certificate_tag=str(data["lean_certificate_tag"]),
            prev_hash=str(data["prev_hash"]),
            entry_hash=str(data["entry_hash"]),
        )


class CertifiedLedger:
    """Append-only SHA-256 hash-chained ledger for certified positive and negative results."""

    def __init__(self, entries: list[CertifiedLedgerEntry] | None = None) -> None:
        self._entries: list[CertifiedLedgerEntry] = list(entries) if entries is not None else []

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> tuple[CertifiedLedgerEntry, ...]:
        return tuple(self._entries)

    @property
    def head_hash(self) -> str:
        return self._entries[-1].entry_hash if self._entries else GENESIS_HASH

    def append(
        self,
        *,
        sample_id: str,
        candidate_features: Mapping[str, float],
        measured_value: float,
        telemetry: AssayTelemetry,
        orthogonal_readout: OrthogonalAssayReadout | None = None,
    ) -> CertifiedLedgerEntry:
        """Validate causal attribution and append a certified entry to the hash chain."""
        if not sample_id or not sample_id.strip():
            raise ValueError("sample_id must be a non-empty string")
        outcome = effective_attribution(telemetry, orthogonal_readout)
        if not admit_to_scientific_memory(outcome):
            raise LedgerRejectionError(
                f"Sample {sample_id!r} rejected by CertifiedLedger: outcome={outcome.value} "
                f"(Lean guard: {lean_certificate_tag_for(telemetry, orthogonal_readout)})"
            )

        idx = len(self._entries)
        prev_h = self.head_hash
        cert_tag = lean_certificate_tag_for(telemetry, orthogonal_readout)
        norm_features = {
            str(k): round(float(v), 8) for k, v in sorted(candidate_features.items())
        }
        rounded_val = round(float(measured_value), 8)

        provisional = CertifiedLedgerEntry(
            index=idx,
            sample_id=sample_id,
            candidate_features=norm_features,
            measured_value=rounded_val,
            telemetry=telemetry,
            orthogonal_readout=orthogonal_readout,
            outcome=outcome,
            lean_certificate_tag=cert_tag,
            prev_hash=prev_h,
            entry_hash="",
        )
        digest = _canonical_entry_sha256(provisional.hash_payload())
        committed = CertifiedLedgerEntry(
            index=idx,
            sample_id=sample_id,
            candidate_features=norm_features,
            measured_value=rounded_val,
            telemetry=telemetry,
            orthogonal_readout=orthogonal_readout,
            outcome=outcome,
            lean_certificate_tag=cert_tag,
            prev_hash=prev_h,
            entry_hash=digest,
        )
        self._entries.append(committed)
        return committed

    def verify_chain(self) -> tuple[bool, str | None]:
        """Re-verify sequential indices, SHA-256 chain links, and telemetry admissibility."""
        expected_prev = GENESIS_HASH
        for expected_idx, entry in enumerate(self._entries):
            if entry.index != expected_idx:
                return (
                    False,
                    f"Index mismatch at position {expected_idx}: found index={entry.index}",
                )
            if entry.prev_hash != expected_prev:
                return (
                    False,
                    f"Broken prev_hash link at index {expected_idx}: "
                    f"expected {expected_prev[:12]}..., got {entry.prev_hash[:12]}...",
                )
            recomputed_outcome = effective_attribution(entry.telemetry, entry.orthogonal_readout)
            if entry.outcome is not recomputed_outcome:
                return (
                    False,
                    f"Outcome forgery at index {expected_idx}: stored {entry.outcome.value} "
                    f"!= recomputed {recomputed_outcome.value}",
                )
            if not admit_to_scientific_memory(recomputed_outcome) or not execution_clean(
                entry.telemetry
            ):
                return (
                    False,
                    f"Inadmissible entry stored at index {expected_idx}: {recomputed_outcome.value}",
                )
            expected_tag = lean_certificate_tag_for(entry.telemetry, entry.orthogonal_readout)
            if entry.lean_certificate_tag != expected_tag:
                return (
                    False,
                    f"Certificate tag mismatch at index {expected_idx}: "
                    f"stored {entry.lean_certificate_tag!r} != {expected_tag!r}",
                )
            recomputed_hash = _canonical_entry_sha256(entry.hash_payload())
            if entry.entry_hash != recomputed_hash:
                return (
                    False,
                    f"SHA-256 digest mismatch at index {expected_idx}: "
                    f"stored {entry.entry_hash[:12]}... != {recomputed_hash[:12]}...",
                )
            expected_prev = entry.entry_hash
        return True, None

    def export_training_pairs(self) -> list[dict[str, Any]]:
        """Verify ledger integrity and export clean supervised / RLVR training examples."""
        ok, err = self.verify_chain()
        if not ok:
            raise LedgerIntegrityError(f"Cannot export from tampered ledger: {err}")
        rows: list[dict[str, Any]] = []
        for entry in self._entries:
            rows.append(
                {
                    "index": entry.index,
                    "sample_id": entry.sample_id,
                    "features": dict(entry.candidate_features),
                    "measured_value": entry.measured_value,
                    "is_positive_hit": entry.outcome is AttributionOutcome.CERTIFIED_POSITIVE,
                    "outcome": entry.outcome.value,
                    "lean_certificate_tag": entry.lean_certificate_tag,
                    "entry_hash": entry.entry_hash,
                }
            )
        return rows

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps([e.to_dict() for e in self._entries], indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, raw_json: str) -> CertifiedLedger:
        parsed = json.loads(raw_json)
        if not isinstance(parsed, list):
            raise ValueError("Ledger JSON root must be a list of entry objects")
        ledger = cls([CertifiedLedgerEntry.from_dict(item) for item in parsed])
        ok, err = ledger.verify_chain()
        if not ok:
            raise LedgerIntegrityError(f"Loaded JSON failed ledger verification: {err}")
        return ledger
