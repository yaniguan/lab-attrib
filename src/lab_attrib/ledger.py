"""Append-Only, SHA-256 Hash-Chained Proof-Carrying Negative & Positive Ledger.

Python implementation of `LabAttribLean.LedgerEntry`, `LedgerValid`, and
`appendCertifiedEntry`. Guarantees:
1. Default-deny admission: `EXECUTION_ARTIFACT` and unresolved `AMBIGUOUS_CHARACTERIZATION`
   runs raise `LedgerRejectionError` and can never enter scientific memory.
2. Strict RFC-8259 finite-float canonicalization (`allow_nan=False`): `NaN` and `+-Infinity`
   in `measured_value` or `candidate_features` are rejected before hashing or export.
3. Concurrency & durability: thread-safe `RLock` synchronization, cross-platform file locking
   (`msvcrt.locking` on Windows, `fcntl.flock` on POSIX), atomic temp-file `save()`, and
   optimistic compare-and-swap JSONL write-ahead logging (`append_to_jsonl()`).
4. Anti-pruning cryptographic commitments: RFC-6962 domain-separated SHA-256 Merkle tree
   roots (`merkle_root()`), Merkle inclusion proofs (`inclusion_proof()`,
   `verify_inclusion_proof()`), and HMAC-SHA256 / Ed25519 chain-head attestation seals.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

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
_HEX_CHARS: frozenset[str] = frozenset("0123456789abcdef")
_IN_PROCESS_PATH_LOCKS_GUARD = threading.Lock()
_IN_PROCESS_PATH_LOCKS: dict[str, threading.RLock] = {}


class LedgerRejectionError(ValueError):
    """Raised when attempting to append an uncertified or corrupted run to the ledger."""


class LedgerIntegrityError(RuntimeError):
    """Raised when cryptographic or semantic hash-chain verification detects tampering."""


def _require_finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number, got bool {value!r}")
    val = float(value)
    if not math.isfinite(val):
        raise ValueError(f"{name} must be finite (NaN/Infinity forbidden), got {value!r}")
    return val


def _validate_sha256_hex(digest: str, name: str) -> str:
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in _HEX_CHARS for c in digest):
        raise ValueError(f"{name} must be a 64-character lowercase hex SHA-256 digest, got {digest!r}")
    return digest


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return canonical.encode("utf-8")


def _canonical_entry_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _merkle_leaf_hash(entry_hash: str) -> str:
    _validate_sha256_hex(entry_hash, "entry_hash")
    return hashlib.sha256(b"\x00" + entry_hash.encode("ascii")).hexdigest()


def _merkle_parent_hash(left_hash: str, right_hash: str) -> str:
    return hashlib.sha256(
        b"\x01" + left_hash.encode("ascii") + right_hash.encode("ascii")
    ).hexdigest()


def verify_inclusion_proof(
    entry_hash: str,
    proof_or_index: Sequence[Mapping[str, str]] | int,
    expected_root_or_proof: str | Sequence[Mapping[str, str]],
    maybe_expected_root: str | None = None,
) -> bool:
    """Verify an RFC-6962 domain-separated Merkle inclusion proof against a pinned `expected_root`.

    Supports both `verify_inclusion_proof(entry_hash, proof, expected_root)` and
    `verify_inclusion_proof(entry_hash, index, proof, expected_root)`.
    """
    if isinstance(proof_or_index, int) and not isinstance(proof_or_index, bool):
        if not isinstance(expected_root_or_proof, Sequence) or isinstance(
            expected_root_or_proof, (str, bytes)
        ):
            raise TypeError("Expected proof sequence as third argument when index is passed")
        if maybe_expected_root is None:
            raise TypeError("Expected expected_root as fourth argument when index is passed")
        if proof_or_index < 0:
            return False
        proof = expected_root_or_proof
        expected_root = maybe_expected_root
    else:
        if not isinstance(expected_root_or_proof, str):
            raise TypeError("Expected expected_root string as third argument")
        proof = proof_or_index  # type: ignore[assignment]
        expected_root = expected_root_or_proof

    try:
        _validate_sha256_hex(entry_hash, "entry_hash")
        _validate_sha256_hex(expected_root, "expected_root")
    except ValueError:
        return False

    current = _merkle_leaf_hash(entry_hash)
    for step in proof:
        if not isinstance(step, Mapping):
            return False
        pos = step.get("position")
        sibling = step.get("hash")
        if not isinstance(sibling, str):
            return False
        try:
            _validate_sha256_hex(sibling, "sibling_hash")
        except ValueError:
            return False
        if pos == "left":
            current = _merkle_parent_hash(sibling, current)
        elif pos == "right":
            current = _merkle_parent_hash(current, sibling)
        else:
            return False
    return hmac.compare_digest(current, expected_root)


class _CrossPlatformFileLock:
    """Cross-thread and cross-process advisory file lock (`msvcrt` on Windows, `fcntl` on POSIX)."""

    def __init__(self, target_path: Path, timeout_seconds: float = 10.0) -> None:
        self._lock_path = target_path.resolve().with_name(f"{target_path.name}.lock")
        self._timeout = timeout_seconds
        self._fh: Any = None
        key = str(self._lock_path)
        with _IN_PROCESS_PATH_LOCKS_GUARD:
            if key not in _IN_PROCESS_PATH_LOCKS:
                _IN_PROCESS_PATH_LOCKS[key] = threading.RLock()
            self._thread_lock = _IN_PROCESS_PATH_LOCKS[key]

    def __enter__(self) -> _CrossPlatformFileLock:
        self._thread_lock.acquire()
        try:
            self._lock_path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self._lock_path.open("a+b")
            self._fh.seek(0, os.SEEK_END)
            if self._fh.tell() == 0:
                self._fh.write(b"\x00")
                self._fh.flush()
            self._fh.seek(0)

            deadline = time.monotonic() + self._timeout
            if os.name == "nt":
                import msvcrt

                while True:
                    try:
                        self._fh.seek(0)
                        msvcrt.locking(self._fh.fileno(), msvcrt.LK_NBLCK, 1)
                        break
                    except OSError:
                        if time.monotonic() >= deadline:
                            raise TimeoutError(
                                f"Timed out acquiring ledger lock {self._lock_path}"
                            ) from None
                        time.sleep(0.005)
            else:  # pragma: no cover - exercised on Ubuntu CI
                import fcntl

                while True:
                    try:
                        fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except OSError:
                        if time.monotonic() >= deadline:
                            raise TimeoutError(
                                f"Timed out acquiring ledger lock {self._lock_path}"
                            ) from None
                        time.sleep(0.005)
            return self
        except Exception:
            if self._fh is not None:
                self._fh.close()
                self._fh = None
            self._thread_lock.release()
            raise

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        try:
            if self._fh is not None:
                try:
                    if os.name == "nt":
                        import msvcrt

                        self._fh.seek(0)
                        msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
                    else:  # pragma: no cover - exercised on Ubuntu CI
                        import fcntl

                        fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
                finally:
                    self._fh.close()
                    self._fh = None
        finally:
            self._thread_lock.release()


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

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise ValueError(f"index must be a non-negative integer, got {self.index!r}")
        if not isinstance(self.sample_id, str) or not self.sample_id.strip():
            raise ValueError("sample_id must be a non-empty string")
        _require_finite_float(self.measured_value, "measured_value")
        for key, val in self.candidate_features.items():
            _require_finite_float(val, f"candidate_features[{key!r}]")
        _validate_sha256_hex(self.prev_hash, "prev_hash")
        if self.entry_hash != "":
            _validate_sha256_hex(self.entry_hash, "entry_hash")

    def hash_payload(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "sample_id": self.sample_id,
            "candidate_features": {
                str(k): round(_require_finite_float(v, f"candidate_features[{k!r}]"), 8)
                for k, v in sorted(self.candidate_features.items())
            },
            "measured_value": round(_require_finite_float(self.measured_value, "measured_value"), 8),
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
        raw_features = dict(data["candidate_features"])
        checked_features = {
            str(k): _require_finite_float(v, f"candidate_features[{k!r}]")
            for k, v in raw_features.items()
        }
        checked_measured = _require_finite_float(data["measured_value"], "measured_value")
        return cls(
            index=int(data["index"]),
            sample_id=str(data["sample_id"]),
            candidate_features=checked_features,
            measured_value=checked_measured,
            telemetry=AssayTelemetry.from_dict(dict(data["telemetry"])),
            orthogonal_readout=(
                OrthogonalAssayReadout.from_dict(dict(orth_raw)) if orth_raw is not None else None
            ),
            outcome=AttributionOutcome(str(data["outcome"])),
            lean_certificate_tag=str(data["lean_certificate_tag"]),
            prev_hash=str(data["prev_hash"]),
            entry_hash=str(data["entry_hash"]),
        )


def _normalize_secret_key(secret_key: bytes | str) -> bytes:
    raw = secret_key.encode("utf-8") if isinstance(secret_key, str) else bytes(secret_key)
    if not raw:
        raise ValueError("secret_key must be non-empty")
    return raw


class CertifiedLedger:
    """Thread-safe, append-only SHA-256 hash-chained & Merkle-committed scientific ledger."""

    def __init__(
        self,
        entries: list[CertifiedLedgerEntry] | None = None,
        *,
        start_index: int = 0,
        initial_prev_hash: str = GENESIS_HASH,
        secret_key: bytes | str | None = None,
    ) -> None:
        if isinstance(start_index, bool) or not isinstance(start_index, int) or start_index < 0:
            raise ValueError(f"start_index must be a non-negative integer, got {start_index!r}")
        _validate_sha256_hex(initial_prev_hash, "initial_prev_hash")
        self._lock = threading.RLock()
        self._start_index = start_index
        self._initial_prev_hash = initial_prev_hash
        self._secret_key: bytes | None = (
            _normalize_secret_key(secret_key) if secret_key is not None else None
        )
        self._entries: list[CertifiedLedgerEntry] = list(entries) if entries is not None else []

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    @property
    def entries(self) -> tuple[CertifiedLedgerEntry, ...]:
        with self._lock:
            return tuple(self._entries)

    @property
    def head_hash(self) -> str:
        with self._lock:
            return self._entries[-1].entry_hash if self._entries else self._initial_prev_hash

    def append(
        self,
        *,
        sample_id: str,
        candidate_features: Mapping[str, float],
        measured_value: float,
        telemetry: AssayTelemetry,
        orthogonal_readout: OrthogonalAssayReadout | None = None,
    ) -> CertifiedLedgerEntry:
        """Validate causal attribution and atomically append a certified entry to the hash chain."""
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError("sample_id must be a non-empty string")
        checked_val = round(_require_finite_float(measured_value, "measured_value"), 8)
        norm_features = {
            str(k): round(_require_finite_float(v, f"candidate_features[{k!r}]"), 8)
            for k, v in sorted(candidate_features.items())
        }

        outcome = effective_attribution(telemetry, orthogonal_readout)
        if not admit_to_scientific_memory(outcome):
            raise LedgerRejectionError(
                f"Sample {sample_id!r} rejected by CertifiedLedger: outcome={outcome.value} "
                f"(Lean guard: {lean_certificate_tag_for(telemetry, orthogonal_readout)})"
            )
        cert_tag = lean_certificate_tag_for(telemetry, orthogonal_readout)

        with self._lock:
            idx = self._start_index + len(self._entries)
            prev_h = self._entries[-1].entry_hash if self._entries else self._initial_prev_hash

            provisional = CertifiedLedgerEntry(
                index=idx,
                sample_id=sample_id,
                candidate_features=norm_features,
                measured_value=checked_val,
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
                measured_value=checked_val,
                telemetry=telemetry,
                orthogonal_readout=orthogonal_readout,
                outcome=outcome,
                lean_certificate_tag=cert_tag,
                prev_hash=prev_h,
                entry_hash=digest,
            )
            self._entries.append(committed)
            return committed

    def merkle_root(self) -> str:
        """Compute RFC-6962 domain-separated SHA-256 Merkle root over all ledger entries."""
        with self._lock:
            if not self._entries:
                return GENESIS_HASH
            level = [_merkle_leaf_hash(e.entry_hash) for e in self._entries]
            while len(level) > 1:
                next_level: list[str] = []
                for i in range(0, len(level) - 1, 2):
                    next_level.append(_merkle_parent_hash(level[i], level[i + 1]))
                if len(level) % 2 == 1:
                    next_level.append(level[-1])
                level = next_level
            return level[0]

    def inclusion_proof(self, position: int) -> list[dict[str, str]]:
        """Generate an audit Merkle inclusion proof for the entry at `position` (`0 .. len-1`)."""
        with self._lock:
            n = len(self._entries)
            if isinstance(position, bool) or not isinstance(position, int) or not (0 <= position < n):
                raise IndexError(f"Entry position {position} out of range for ledger of size {n}")
            level = [_merkle_leaf_hash(e.entry_hash) for e in self._entries]
            pos = position
            proof: list[dict[str, str]] = []
            while len(level) > 1:
                if pos % 2 == 0:
                    if pos + 1 < len(level):
                        proof.append({"position": "right", "hash": level[pos + 1]})
                else:
                    proof.append({"position": "left", "hash": level[pos - 1]})

                next_level: list[str] = []
                for i in range(0, len(level) - 1, 2):
                    next_level.append(_merkle_parent_hash(level[i], level[i + 1]))
                if len(level) % 2 == 1:
                    next_level.append(level[-1])
                level = next_level
                pos //= 2
            return proof

    def _head_commitment_payload(self) -> dict[str, Any]:
        return {
            "start_index": self._start_index,
            "length": len(self._entries),
            "head_hash": self.head_hash,
            "merkle_root": self.merkle_root(),
        }

    def seal_head(
        self,
        *,
        secret_key: bytes | str | None = None,
        ed25519_private_key: Any = None,
    ) -> dict[str, Any]:
        """Create a cryptographic attestation seal over `(length, head_hash, merkle_root)`."""
        with self._lock:
            commitment = self._head_commitment_payload()
            msg = _canonical_json_bytes(commitment)
            if ed25519_private_key is not None:
                sig_bytes = bytes(ed25519_private_key.sign(msg))
                return {
                    **commitment,
                    "scheme": "ed25519",
                    "signature_hex": sig_bytes.hex(),
                }
            eff_key = (
                _normalize_secret_key(secret_key)
                if secret_key is not None
                else self._secret_key
            )
            if eff_key is None:
                raise ValueError(
                    "seal_head requires either secret_key (HMAC-SHA256) or ed25519_private_key"
                )
            mac_hex = hmac.new(eff_key, msg, hashlib.sha256).hexdigest()
            return {
                **commitment,
                "scheme": "hmac-sha256",
                "signature_hex": mac_hex,
            }

    def verify_head_seal(
        self,
        seal: Mapping[str, Any],
        *,
        secret_key: bytes | str | None = None,
        ed25519_public_key: Any = None,
    ) -> tuple[bool, str | None]:
        """Verify an HMAC-SHA256 or Ed25519 chain-head seal against the current ledger state."""
        with self._lock:
            current = self._head_commitment_payload()
            for field in ("start_index", "length", "head_hash", "merkle_root"):
                if seal.get(field) != current[field]:
                    return (
                        False,
                        f"Head seal {field} mismatch: seal={seal.get(field)!r} != ledger={current[field]!r}",
                    )
            scheme = seal.get("scheme")
            sig_hex = seal.get("signature_hex")
            if not isinstance(sig_hex, str) or not sig_hex:
                return False, "Head seal missing signature_hex"
            msg = _canonical_json_bytes(current)

            if scheme == "ed25519":
                if ed25519_public_key is None:
                    return False, "ed25519_public_key required to verify ed25519 head seal"
                try:
                    ed25519_public_key.verify(bytes.fromhex(sig_hex), msg)
                except Exception as exc:
                    return False, f"Ed25519 signature verification failed: {exc}"
                return True, None

            if scheme == "hmac-sha256":
                eff_key = (
                    _normalize_secret_key(secret_key)
                    if secret_key is not None
                    else self._secret_key
                )
                if eff_key is None:
                    return False, "secret_key required to verify hmac-sha256 head seal"
                expected_mac = hmac.new(eff_key, msg, hashlib.sha256).hexdigest()
                if not hmac.compare_digest(sig_hex, expected_mac):
                    return False, "HMAC-SHA256 head signature mismatch"
                return True, None

            return False, f"Unsupported seal scheme: {scheme!r}"

    def verify_chain(
        self,
        *,
        expected_merkle_root: str | None = None,
        head_seal: Mapping[str, Any] | None = None,
        secret_key: bytes | str | None = None,
        ed25519_public_key: Any = None,
    ) -> tuple[bool, str | None]:
        """Re-verify sequential indices, finite floats, SHA-256 chain links, and optional seals."""
        with self._lock:
            expected_prev = self._initial_prev_hash
            for offset, entry in enumerate(self._entries):
                expected_idx = self._start_index + offset
                if entry.index != expected_idx:
                    return (
                        False,
                        f"Index mismatch at position {offset}: expected {expected_idx}, found index={entry.index}",
                    )
                if entry.prev_hash != expected_prev:
                    return (
                        False,
                        f"Broken prev_hash link at index {expected_idx}: "
                        f"expected {expected_prev[:12]}..., got {entry.prev_hash[:12]}...",
                    )
                try:
                    payload = entry.hash_payload()
                    recomputed_hash = _canonical_entry_sha256(payload)
                except (ValueError, TypeError) as exc:
                    return False, f"Non-finite or invalid entry payload at index {expected_idx}: {exc}"

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
                if not hmac.compare_digest(entry.entry_hash, recomputed_hash):
                    return (
                        False,
                        f"SHA-256 digest mismatch at index {expected_idx}: "
                        f"stored {entry.entry_hash[:12]}... != {recomputed_hash[:12]}...",
                    )
                expected_prev = entry.entry_hash

            if expected_merkle_root is not None:
                actual_root = self.merkle_root()
                if not hmac.compare_digest(actual_root, expected_merkle_root):
                    return (
                        False,
                        f"Merkle root mismatch: expected {expected_merkle_root[:12]}..., "
                        f"got {actual_root[:12]}...",
                    )

            if head_seal is not None:
                seal_ok, seal_err = self.verify_head_seal(
                    head_seal,
                    secret_key=secret_key,
                    ed25519_public_key=ed25519_public_key,
                )
                if not seal_ok:
                    return False, seal_err

            return True, None

    def export_training_pairs(self) -> list[dict[str, Any]]:
        """Verify ledger integrity and export clean supervised / RLVR training examples."""
        with self._lock:
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
        with self._lock:
            return json.dumps(
                [e.to_dict() for e in self._entries],
                indent=indent,
                sort_keys=True,
                allow_nan=False,
            )

    @classmethod
    def from_json(cls, raw_json: str) -> CertifiedLedger:
        def _reject_nonfinite_constant(token: str) -> Any:
            raise ValueError(f"RFC-8259 invalid non-finite JSON constant {token!r}")

        parsed = json.loads(raw_json, parse_constant=_reject_nonfinite_constant)
        if not isinstance(parsed, list):
            raise ValueError("Ledger JSON root must be a list of entry objects")
        ledger = cls([CertifiedLedgerEntry.from_dict(item) for item in parsed])
        ok, err = ledger.verify_chain()
        if not ok:
            raise LedgerIntegrityError(f"Loaded JSON failed ledger verification: {err}")
        return ledger

    def save(
        self,
        path: str | Path,
        *,
        secret_key: bytes | str | None = None,
        ed25519_private_key: Any = None,
    ) -> dict[str, Any] | None:
        """Atomically persist the verified ledger (and optional `.seal.json`) under a file lock."""
        target = Path(path)
        with self._lock:
            ok, err = self.verify_chain()
            if not ok:
                raise LedgerIntegrityError(f"Refusing to save invalid ledger: {err}")
            serialized = self.to_json(indent=2) + "\n"
            seal_payload: dict[str, Any] | None = None
            if (
                secret_key is not None
                or ed25519_private_key is not None
                or self._secret_key is not None
            ):
                seal_payload = self.seal_head(
                    secret_key=secret_key,
                    ed25519_private_key=ed25519_private_key,
                )

        with _CrossPlatformFileLock(target):
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = target.with_name(
                f".{target.name}.{os.getpid()}.{threading.get_ident()}.tmp"
            )
            try:
                with tmp_path.open("w", encoding="utf-8", newline="\n") as fh:
                    fh.write(serialized)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp_path, target)
            finally:
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except OSError:
                        pass

            if seal_payload is not None:
                seal_path = target.with_name(f"{target.name}.seal.json")
                tmp_seal = target.with_name(
                    f".{target.name}.{os.getpid()}.{threading.get_ident()}.seal.tmp"
                )
                seal_text = json.dumps(seal_payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
                try:
                    with tmp_seal.open("w", encoding="utf-8", newline="\n") as fh:
                        fh.write(seal_text)
                        fh.flush()
                        os.fsync(fh.fileno())
                    os.replace(tmp_seal, seal_path)
                finally:
                    if tmp_seal.exists():
                        try:
                            tmp_seal.unlink()
                        except OSError:
                            pass
        return seal_payload

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        secret_key: bytes | str | None = None,
        ed25519_public_key: Any = None,
    ) -> CertifiedLedger:
        """Load and verify a persisted JSON ledger (and optional `.seal.json` if key supplied)."""
        target = Path(path)
        with _CrossPlatformFileLock(target):
            raw = target.read_text(encoding="utf-8")
            seal_path = target.with_name(f"{target.name}.seal.json")
            seal_data: dict[str, Any] | None = None
            if seal_path.is_file():
                seal_data = dict(json.loads(seal_path.read_text(encoding="utf-8")))

        ledger = cls.from_json(raw)
        if secret_key is not None:
            ledger._secret_key = _normalize_secret_key(secret_key)
        if seal_data is not None and (
            secret_key is not None or ed25519_public_key is not None
        ):
            ok, err = ledger.verify_head_seal(
                seal_data,
                secret_key=secret_key,
                ed25519_public_key=ed25519_public_key,
            )
            if not ok:
                raise LedgerIntegrityError(f"Loaded ledger failed head seal check: {err}")
        return ledger

    @classmethod
    def append_to_jsonl(
        cls,
        path: str | Path,
        *,
        sample_id: str,
        candidate_features: Mapping[str, float],
        measured_value: float,
        telemetry: AssayTelemetry,
        orthogonal_readout: OrthogonalAssayReadout | None = None,
        expected_head_hash: str | None = None,
    ) -> CertifiedLedgerEntry:
        """Cross-process locked append-only WAL writer with optional CAS `expected_head_hash`."""
        target = Path(path)
        with _CrossPlatformFileLock(target):
            existing_entries: list[CertifiedLedgerEntry] = []
            if target.is_file():
                for line_no, line in enumerate(
                    target.read_text(encoding="utf-8").splitlines(), start=1
                ):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    item = json.loads(stripped)
                    existing_entries.append(CertifiedLedgerEntry.from_dict(item))
            ledger = cls(existing_entries)
            ok, err = ledger.verify_chain()
            if not ok:
                raise LedgerIntegrityError(f"Existing JSONL WAL corrupted: {err}")
            if expected_head_hash is not None and ledger.head_hash != expected_head_hash:
                raise LedgerIntegrityError(
                    f"Optimistic CAS conflict: expected head_hash={expected_head_hash[:12]}..., "
                    f"found {ledger.head_hash[:12]}..."
                )
            committed = ledger.append(
                sample_id=sample_id,
                candidate_features=candidate_features,
                measured_value=measured_value,
                telemetry=telemetry,
                orthogonal_readout=orthogonal_readout,
            )
            serialized_line = json.dumps(
                committed.to_dict(),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a", encoding="utf-8", newline="\n") as fh:
                fh.write(serialized_line + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            return committed

    @classmethod
    def load_jsonl(cls, path: str | Path) -> CertifiedLedger:
        """Load and verify a JSONL write-ahead log ledger."""
        target = Path(path)
        with _CrossPlatformFileLock(target):
            entries: list[CertifiedLedgerEntry] = []
            for line in target.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped:
                    entries.append(CertifiedLedgerEntry.from_dict(json.loads(stripped)))
        ledger = cls(entries)
        ok, err = ledger.verify_chain()
        if not ok:
            raise LedgerIntegrityError(f"JSONL ledger verification failed: {err}")
        return ledger
