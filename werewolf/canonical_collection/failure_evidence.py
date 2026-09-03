"""Immutable failure and partial evidence for one claimed collection attempt."""

from __future__ import annotations

import errno
import json
import os
import re
import stat
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

from werewolf.artifact_io import canonical_json_bytes, sha256_bytes
from werewolf.canonical_collection.attempt_ledger import (
    AttemptClaim,
    CollectionPlan,
    PartialEvidenceReference,
    validate_attempt_claim,
    validate_collection_plan,
)
from werewolf.canonical_collection.public_history import PLAYER_IDS, PUBLIC_PHASES
from werewolf.canonical_collection.frozen_json import (
    FrozenJSONValue,
    freeze_json_object,
)


CANONICAL_PARTIAL_EVIDENCE_SCHEMA_VERSION = "classic7_partial_evidence_v1"
CANONICAL_FAILURE_ATTEMPT_SCHEMA_VERSION = "classic7_failure_attempt_v1"
CANONICAL_FAILURE_EVIDENCE_SCHEMA_VERSION = "classic7_failure_evidence_v1"
CANONICAL_FAILURE_SUMMARY_SCHEMA_VERSION = "classic7_failure_summary_v1"
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class CanonicalFailureStage(str, Enum):
    BELIEF_OBSERVATION = "belief_observation"
    SPEECH_PERCEPTION = "speech_perception"
    GAMEPLAY_ACTION = "gameplay_action"
    DETERMINISTIC_REPLAY = "deterministic_replay"
    RUNTIME = "runtime"
    INTERRUPTED = "interrupted"


class CanonicalEvidenceValidationError(ValueError):
    """Attempt evidence is malformed or not bound to its durable claim."""


class CanonicalEvidenceConflictError(FileExistsError):
    """An immutable final evidence path already exists."""


@dataclass(frozen=True)
class CanonicalPartialEvidence:
    schema_version: str
    collection_id: str
    collection_plan_digest: str
    claim_record_digest: str
    attempt_id: str
    ordinal: int
    seed: int
    evidence_id: str
    evidence_type: str
    payload: FrozenJSONValue
    record_digest: str

    def to_record(self) -> dict[str, Any]:
        return {
            **_partial_record_without_digest(self),
            "record_digest": self.record_digest,
        }


@dataclass(frozen=True)
class VerifiedCanonicalPartialEvidence:
    path: Path
    evidence: CanonicalPartialEvidence
    file_sha256: str
    ledger_reference: PartialEvidenceReference


@dataclass(frozen=True)
class CanonicalFailureAttempt:
    schema_version: str
    attempt_index: int
    call_id: str
    backend_identity: str
    model_identity: str
    parser_identity: str
    prompt_identity: str
    retry_policy_identity: str
    call_budget_identity: str
    error_category: str
    error_message: str
    response_digest: str | None
    attempt_digest: str

    def to_record(self) -> dict[str, Any]:
        return {
            **_failure_attempt_record_without_digest(self),
            "attempt_digest": self.attempt_digest,
        }


@dataclass(frozen=True)
class CanonicalFailureEvidence:
    schema_version: str
    collection_id: str
    collection_plan_digest: str
    claim_record_digest: str
    attempt_id: str
    ordinal: int
    seed: int
    game_id: str
    stage: CanonicalFailureStage
    error_category: str
    boundary_id: str | None
    observer_id: str | None
    day: int | None
    phase: str | None
    retry_exhausted: bool
    attempt_evidence: tuple[CanonicalFailureAttempt, ...]
    partial_evidence: tuple[PartialEvidenceReference, ...]
    failure_digest: str

    def to_record(self) -> dict[str, Any]:
        return {
            **_failure_record_without_digest(self),
            "failure_digest": self.failure_digest,
        }


@dataclass(frozen=True)
class VerifiedCanonicalFailureEvidence:
    path: Path
    evidence: CanonicalFailureEvidence
    file_sha256: str


@dataclass(frozen=True)
class CanonicalFailureSummary:
    schema_version: str
    total_failure_count: int
    failure_count_by_stage: tuple[tuple[str, int], ...]
    failure_count_by_category: tuple[tuple[str, int], ...]
    summary_digest: str

    def to_record(self) -> dict[str, Any]:
        return {
            **_summary_record_without_digest(self),
            "summary_digest": self.summary_digest,
        }


def _required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty text")
    return value


def _safe_component(value: Any, field_name: str) -> str:
    value = _required_text(value, field_name)
    if _SAFE_COMPONENT.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be one safe path component")
    return value


def _sha256(value: Any, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _nonnegative_integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _canonical_mapping(
    value: Mapping[str, Any] | FrozenJSONValue,
    field_name: str,
) -> FrozenJSONValue:
    return freeze_json_object(value, field_name)


def _partial_record_without_digest(
    evidence: CanonicalPartialEvidence,
) -> dict[str, Any]:
    return {
        "schema_version": evidence.schema_version,
        "collection_id": evidence.collection_id,
        "collection_plan_digest": evidence.collection_plan_digest,
        "claim_record_digest": evidence.claim_record_digest,
        "attempt_id": evidence.attempt_id,
        "ordinal": evidence.ordinal,
        "seed": evidence.seed,
        "evidence_id": evidence.evidence_id,
        "evidence_type": evidence.evidence_type,
        "payload": evidence.payload.to_value(),
    }


def construct_canonical_partial_evidence(
    *,
    plan: CollectionPlan,
    claim: AttemptClaim,
    evidence_id: str,
    evidence_type: str,
    payload: Mapping[str, Any] | FrozenJSONValue,
) -> CanonicalPartialEvidence:
    validate_collection_plan(plan)
    validate_attempt_claim(claim, plan)
    provisional = CanonicalPartialEvidence(
        schema_version=CANONICAL_PARTIAL_EVIDENCE_SCHEMA_VERSION,
        collection_id=plan.collection_id,
        collection_plan_digest=plan.plan_digest,
        claim_record_digest=claim.record_digest,
        attempt_id=claim.attempt_id,
        ordinal=claim.ordinal,
        seed=claim.seed,
        evidence_id=_safe_component(evidence_id, "evidence_id"),
        evidence_type=_required_text(evidence_type, "evidence_type"),
        payload=_canonical_mapping(payload, "partial evidence payload"),
        record_digest="",
    )
    return replace(
        provisional,
        record_digest=sha256_bytes(
            canonical_json_bytes(_partial_record_without_digest(provisional))
        ),
    )


def _failure_record_without_digest(
    evidence: CanonicalFailureEvidence,
) -> dict[str, Any]:
    return {
        "schema_version": evidence.schema_version,
        "collection_id": evidence.collection_id,
        "collection_plan_digest": evidence.collection_plan_digest,
        "claim_record_digest": evidence.claim_record_digest,
        "attempt_id": evidence.attempt_id,
        "ordinal": evidence.ordinal,
        "seed": evidence.seed,
        "game_id": evidence.game_id,
        "stage": evidence.stage.value,
        "error_category": evidence.error_category,
        "boundary_id": evidence.boundary_id,
        "observer_id": evidence.observer_id,
        "day": evidence.day,
        "phase": evidence.phase,
        "retry_exhausted": evidence.retry_exhausted,
        "attempt_evidence": [item.to_record() for item in evidence.attempt_evidence],
        "partial_evidence": [item.to_record() for item in evidence.partial_evidence],
    }


def _failure_attempt_record_without_digest(
    attempt: CanonicalFailureAttempt,
) -> dict[str, Any]:
    return {
        "schema_version": attempt.schema_version,
        "attempt_index": attempt.attempt_index,
        "call_id": attempt.call_id,
        "backend_identity": attempt.backend_identity,
        "model_identity": attempt.model_identity,
        "parser_identity": attempt.parser_identity,
        "prompt_identity": attempt.prompt_identity,
        "retry_policy_identity": attempt.retry_policy_identity,
        "call_budget_identity": attempt.call_budget_identity,
        "status": "error",
        "error_category": attempt.error_category,
        "error_message": attempt.error_message,
        "response_digest": attempt.response_digest,
    }


def construct_canonical_failure_attempt(
    *,
    attempt_index: int,
    call_id: str,
    backend_identity: str,
    model_identity: str,
    parser_identity: str,
    prompt_identity: str,
    retry_policy_identity: str,
    call_budget_identity: str,
    error_category: str,
    error_message: str,
    response_digest: str | None = None,
) -> CanonicalFailureAttempt:
    attempt_index = _nonnegative_integer(attempt_index, "attempt_index")
    if attempt_index == 0:
        raise ValueError("failure attempt indices must start at one")
    if response_digest is not None:
        response_digest = _sha256(response_digest, "response_digest")
    provisional = CanonicalFailureAttempt(
        schema_version=CANONICAL_FAILURE_ATTEMPT_SCHEMA_VERSION,
        attempt_index=attempt_index,
        call_id=_required_text(call_id, "call_id"),
        backend_identity=_required_text(backend_identity, "backend_identity"),
        model_identity=_required_text(model_identity, "model_identity"),
        parser_identity=_required_text(parser_identity, "parser_identity"),
        prompt_identity=_required_text(prompt_identity, "prompt_identity"),
        retry_policy_identity=_required_text(
            retry_policy_identity,
            "retry_policy_identity",
        ),
        call_budget_identity=_required_text(
            call_budget_identity,
            "call_budget_identity",
        ),
        error_category=_required_text(error_category, "attempt error_category"),
        error_message=_required_text(error_message, "attempt error_message"),
        response_digest=response_digest,
        attempt_digest="",
    )
    return replace(
        provisional,
        attempt_digest=sha256_bytes(
            canonical_json_bytes(_failure_attempt_record_without_digest(provisional))
        ),
    )


def construct_canonical_failure_evidence(
    *,
    plan: CollectionPlan,
    claim: AttemptClaim,
    game_id: str,
    stage: CanonicalFailureStage,
    error_category: str,
    boundary_id: str | None,
    observer_id: str | None,
    day: int | None,
    phase: str | None,
    retry_exhausted: bool,
    attempt_evidence: Sequence[CanonicalFailureAttempt],
    partial_evidence: Sequence[VerifiedCanonicalPartialEvidence],
) -> CanonicalFailureEvidence:
    validate_collection_plan(plan)
    validate_attempt_claim(claim, plan)
    if not isinstance(stage, CanonicalFailureStage):
        raise TypeError("stage must be CanonicalFailureStage")
    if boundary_id is not None:
        boundary_id = _required_text(boundary_id, "boundary_id")
    if observer_id is not None and observer_id not in PLAYER_IDS:
        raise ValueError("observer_id must be an exact player ID")
    if day is not None:
        day = _nonnegative_integer(day, "day")
    if phase is not None and phase not in PUBLIC_PHASES:
        raise ValueError("failure evidence has an unknown Public Phase")
    if not isinstance(retry_exhausted, bool):
        raise TypeError("retry_exhausted must be boolean")
    if stage in {
        CanonicalFailureStage.BELIEF_OBSERVATION,
        CanonicalFailureStage.SPEECH_PERCEPTION,
    } and (
        boundary_id is None
        or observer_id is None
        or day is None
        or phase is None
    ):
        raise ValueError(
            "causal observer failure requires boundary, observer, day, and phase"
        )
    if stage in {
        CanonicalFailureStage.BELIEF_OBSERVATION,
        CanonicalFailureStage.SPEECH_PERCEPTION,
    } and not retry_exhausted:
        raise ValueError("bounded observer/perception failure must exhaust retries")
    attempts = tuple(attempt_evidence)
    if any(not isinstance(item, CanonicalFailureAttempt) for item in attempts):
        raise TypeError("attempt_evidence requires CanonicalFailureAttempt values")
    if tuple(item.attempt_index for item in attempts) != tuple(
        range(1, len(attempts) + 1)
    ):
        raise ValueError("failure attempt indices must be contiguous from one")
    if len({item.call_id for item in attempts}) != len(attempts):
        raise ValueError("failure attempt call IDs must be unique")
    for item in attempts:
        expected = construct_canonical_failure_attempt(
            attempt_index=item.attempt_index,
            call_id=item.call_id,
            backend_identity=item.backend_identity,
            model_identity=item.model_identity,
            parser_identity=item.parser_identity,
            prompt_identity=item.prompt_identity,
            retry_policy_identity=item.retry_policy_identity,
            call_budget_identity=item.call_budget_identity,
            error_category=item.error_category,
            error_message=item.error_message,
            response_digest=item.response_digest,
        )
        if item != expected:
            raise ValueError("failure attempt digest mismatch")
        frozen_identities = {
            "backend_identity": plan.backend_identity,
            "model_identity": plan.model_identity,
            "parser_identity": plan.parser_identity,
            "prompt_identity": plan.prompt_identity,
            "retry_policy_identity": plan.retry_policy_identity,
            "call_budget_identity": plan.call_budget_identity,
        }
        for field_name, expected_value in frozen_identities.items():
            if getattr(item, field_name) != expected_value:
                raise ValueError(
                    f"failure attempt {field_name} does not match Collection Plan"
                )
    if stage is not CanonicalFailureStage.INTERRUPTED and not attempts:
        raise ValueError("normal failure requires attempt evidence")
    if attempts and attempts[-1].error_category != error_category:
        raise ValueError("failure category must match the exhausted final attempt")
    references: list[PartialEvidenceReference] = []
    for item in partial_evidence:
        if not isinstance(item, VerifiedCanonicalPartialEvidence):
            raise TypeError("partial_evidence requires verified handles")
        if (
            item.evidence.collection_plan_digest != plan.plan_digest
            or item.evidence.claim_record_digest != claim.record_digest
        ):
            raise ValueError("partial evidence provenance mismatch")
        references.append(item.ledger_reference)
    references.sort(key=lambda item: item.relative_path)
    provisional = CanonicalFailureEvidence(
        schema_version=CANONICAL_FAILURE_EVIDENCE_SCHEMA_VERSION,
        collection_id=plan.collection_id,
        collection_plan_digest=plan.plan_digest,
        claim_record_digest=claim.record_digest,
        attempt_id=claim.attempt_id,
        ordinal=claim.ordinal,
        seed=claim.seed,
        game_id=_required_text(game_id, "game_id"),
        stage=stage,
        error_category=_required_text(error_category, "error_category"),
        boundary_id=boundary_id,
        observer_id=observer_id,
        day=day,
        phase=phase,
        retry_exhausted=retry_exhausted,
        attempt_evidence=attempts,
        partial_evidence=tuple(references),
        failure_digest="",
    )
    return replace(
        provisional,
        failure_digest=sha256_bytes(
            canonical_json_bytes(_failure_record_without_digest(provisional))
        ),
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_attempt_directories(collection_directory: Path, attempt_id: str) -> Path:
    attempt_id = _safe_component(attempt_id, "attempt_id")
    chain = [
        collection_directory,
        collection_directory / "attempts",
        collection_directory / "attempts" / attempt_id,
    ]
    for path in chain:
        path.mkdir(exist_ok=True)
        if path.is_symlink() or not path.is_dir():
            raise CanonicalEvidenceValidationError(
                "attempt evidence directory is invalid"
            )
        _fsync_directory(path)
        if path != collection_directory:
            _fsync_directory(path.parent)
    return chain[-1]


def _publish_record_noreplace(
    *,
    attempt_directory: Path,
    final_path: Path,
    data: bytes,
) -> None:
    staging_directory = attempt_directory / ".staging"
    staging_directory.mkdir(exist_ok=True)
    _fsync_directory(attempt_directory)
    staging_path = staging_directory / f"{final_path.name}.{uuid4().hex}.staging"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(staging_path, flags, 0o600)
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    final_path.parent.mkdir(exist_ok=True)
    _fsync_directory(final_path.parent)
    _fsync_directory(final_path.parent.parent)
    try:
        try:
            os.link(staging_path, final_path)
        except OSError as error:
            if error.errno == errno.EEXIST:
                raise CanonicalEvidenceConflictError(final_path) from error
            raise
        _fsync_directory(final_path.parent)
    finally:
        if staging_path.exists():
            staging_path.unlink()
            _fsync_directory(staging_directory)


def _read_record(path: Path) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink():
        raise CanonicalEvidenceValidationError("attempt evidence cannot be a symlink")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise CanonicalEvidenceValidationError("attempt evidence is missing") from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise CanonicalEvidenceValidationError(
                "attempt evidence must be a regular file"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    data = b"".join(chunks)
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeError, ValueError) as error:
        raise CanonicalEvidenceValidationError(
            "attempt evidence is invalid JSON"
        ) from error
    if not isinstance(value, dict) or canonical_json_bytes(value) != data:
        raise CanonicalEvidenceValidationError("attempt evidence is not canonical JSON")
    return value, data


def _validate_parent_fields(
    value: Mapping[str, Any],
    plan: CollectionPlan,
    claim: AttemptClaim,
) -> None:
    expected = {
        "collection_id": plan.collection_id,
        "collection_plan_digest": plan.plan_digest,
        "claim_record_digest": claim.record_digest,
        "attempt_id": claim.attempt_id,
        "ordinal": claim.ordinal,
        "seed": claim.seed,
    }
    for field_name, expected_value in expected.items():
        if value.get(field_name) != expected_value:
            raise CanonicalEvidenceValidationError(
                f"attempt evidence {field_name} mismatch"
            )


def publish_canonical_partial_evidence(
    collection_directory: Path | str,
    *,
    plan: CollectionPlan,
    claim: AttemptClaim,
    evidence: CanonicalPartialEvidence,
) -> VerifiedCanonicalPartialEvidence:
    validate_collection_plan(plan)
    validate_attempt_claim(claim, plan)
    if not isinstance(evidence, CanonicalPartialEvidence):
        raise TypeError("evidence must be CanonicalPartialEvidence")
    _validate_parent_fields(evidence.to_record(), plan, claim)
    expected = construct_canonical_partial_evidence(
        plan=plan,
        claim=claim,
        evidence_id=evidence.evidence_id,
        evidence_type=evidence.evidence_type,
        payload=evidence.payload,
    )
    if evidence != expected:
        raise CanonicalEvidenceValidationError(
            "partial evidence contains a stale nested digest"
        )
    collection_directory = Path(collection_directory)
    attempt_directory = _ensure_attempt_directories(
        collection_directory,
        claim.attempt_id,
    )
    final_path = (
        attempt_directory / "partial_evidence" / f"{evidence.evidence_id}.json"
    )
    _publish_record_noreplace(
        attempt_directory=attempt_directory,
        final_path=final_path,
        data=canonical_json_bytes(evidence.to_record()),
    )
    return validate_canonical_partial_evidence(
        final_path,
        plan=plan,
        claim=claim,
        collection_directory=collection_directory,
    )


def validate_canonical_partial_evidence(
    path: Path | str,
    *,
    plan: CollectionPlan,
    claim: AttemptClaim,
    collection_directory: Path | str | None = None,
) -> VerifiedCanonicalPartialEvidence:
    validate_collection_plan(plan)
    validate_attempt_claim(claim, plan)
    path = Path(path)
    value, data = _read_record(path)
    _validate_parent_fields(value, plan, claim)
    if value.get("schema_version") != CANONICAL_PARTIAL_EVIDENCE_SCHEMA_VERSION:
        raise CanonicalEvidenceValidationError("unsupported partial evidence schema")
    try:
        evidence = construct_canonical_partial_evidence(
            plan=plan,
            claim=claim,
            evidence_id=value["evidence_id"],
            evidence_type=value["evidence_type"],
            payload=value["payload"],
        )
    except KeyError as error:
        raise CanonicalEvidenceValidationError(
            "partial evidence field is missing"
        ) from error
    if evidence.to_record() != value:
        raise CanonicalEvidenceValidationError("partial evidence digest mismatch")
    root = (
        Path(collection_directory)
        if collection_directory is not None
        else path.parents[3]
    )
    expected_path = (
        root
        / "attempts"
        / claim.attempt_id
        / "partial_evidence"
        / f"{evidence.evidence_id}.json"
    )
    if path != expected_path:
        raise CanonicalEvidenceValidationError(
            "partial evidence path does not match its identity"
        )
    try:
        relative_path = path.relative_to(root).as_posix()
    except ValueError as error:
        raise CanonicalEvidenceValidationError(
            "partial evidence is outside its collection"
        ) from error
    return VerifiedCanonicalPartialEvidence(
        path=path,
        evidence=evidence,
        file_sha256=sha256_bytes(data),
        ledger_reference=PartialEvidenceReference(
            relative_path=relative_path,
            sha256=sha256_bytes(data),
        ),
    )


def publish_canonical_failure_evidence(
    collection_directory: Path | str,
    *,
    plan: CollectionPlan,
    claim: AttemptClaim,
    evidence: CanonicalFailureEvidence,
) -> VerifiedCanonicalFailureEvidence:
    validate_collection_plan(plan)
    validate_attempt_claim(claim, plan)
    if not isinstance(evidence, CanonicalFailureEvidence):
        raise TypeError("evidence must be CanonicalFailureEvidence")
    _validate_parent_fields(evidence.to_record(), plan, claim)
    collection_directory = Path(collection_directory)
    verified_partials: list[VerifiedCanonicalPartialEvidence] = []
    for reference in evidence.partial_evidence:
        verified = validate_canonical_partial_evidence(
            collection_directory / reference.relative_path,
            plan=plan,
            claim=claim,
            collection_directory=collection_directory,
        )
        if verified.file_sha256 != reference.sha256:
            raise CanonicalEvidenceValidationError(
                "failure evidence partial digest mismatch"
            )
        verified_partials.append(verified)
    expected = construct_canonical_failure_evidence(
        plan=plan,
        claim=claim,
        game_id=evidence.game_id,
        stage=evidence.stage,
        error_category=evidence.error_category,
        boundary_id=evidence.boundary_id,
        observer_id=evidence.observer_id,
        day=evidence.day,
        phase=evidence.phase,
        retry_exhausted=evidence.retry_exhausted,
        attempt_evidence=evidence.attempt_evidence,
        partial_evidence=tuple(verified_partials),
    )
    if evidence != expected:
        raise CanonicalEvidenceValidationError(
            "failure evidence contains a stale nested digest"
        )
    attempt_directory = _ensure_attempt_directories(
        collection_directory,
        claim.attempt_id,
    )
    final_path = attempt_directory / "failure_evidence.json"
    _publish_record_noreplace(
        attempt_directory=attempt_directory,
        final_path=final_path,
        data=canonical_json_bytes(evidence.to_record()),
    )
    return validate_canonical_failure_evidence(
        final_path,
        plan=plan,
        claim=claim,
    )


def validate_canonical_failure_evidence(
    path: Path | str,
    *,
    plan: CollectionPlan,
    claim: AttemptClaim,
) -> VerifiedCanonicalFailureEvidence:
    validate_collection_plan(plan)
    validate_attempt_claim(claim, plan)
    path = Path(path)
    collection_directory = path.parents[2]
    expected_path = (
        collection_directory
        / "attempts"
        / claim.attempt_id
        / "failure_evidence.json"
    )
    if path != expected_path:
        raise CanonicalEvidenceValidationError(
            "failure evidence path does not match its claim"
        )
    value, data = _read_record(path)
    _validate_parent_fields(value, plan, claim)
    try:
        verified_partials: list[VerifiedCanonicalPartialEvidence] = []
        for item in value["partial_evidence"]:
            reference = PartialEvidenceReference(
                relative_path=item["relative_path"],
                sha256=item["sha256"],
            )
            partial_path = collection_directory / reference.relative_path
            verified = validate_canonical_partial_evidence(
                partial_path,
                plan=plan,
                claim=claim,
                collection_directory=collection_directory,
            )
            if verified.file_sha256 != reference.sha256:
                raise CanonicalEvidenceValidationError(
                    "failure evidence partial digest mismatch"
                )
            verified_partials.append(verified)
        attempts: list[CanonicalFailureAttempt] = []
        for item in value["attempt_evidence"]:
            if item.get("status") != "error":
                raise ValueError("failure attempt status must be error")
            attempt = construct_canonical_failure_attempt(
                attempt_index=item["attempt_index"],
                call_id=item["call_id"],
                backend_identity=item["backend_identity"],
                model_identity=item["model_identity"],
                parser_identity=item["parser_identity"],
                prompt_identity=item["prompt_identity"],
                retry_policy_identity=item["retry_policy_identity"],
                call_budget_identity=item["call_budget_identity"],
                error_category=item["error_category"],
                error_message=item["error_message"],
                response_digest=item["response_digest"],
            )
            if attempt.to_record() != item:
                raise ValueError("failure attempt record is not canonical")
            attempts.append(attempt)
        evidence = construct_canonical_failure_evidence(
            plan=plan,
            claim=claim,
            game_id=value["game_id"],
            stage=CanonicalFailureStage(value["stage"]),
            error_category=value["error_category"],
            boundary_id=value["boundary_id"],
            observer_id=value["observer_id"],
            day=value["day"],
            phase=value["phase"],
            retry_exhausted=value["retry_exhausted"],
            attempt_evidence=tuple(attempts),
            partial_evidence=tuple(verified_partials),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise CanonicalEvidenceValidationError(
            "invalid failure evidence record"
        ) from error
    if evidence.schema_version != CANONICAL_FAILURE_EVIDENCE_SCHEMA_VERSION:
        raise CanonicalEvidenceValidationError("unsupported failure evidence schema")
    if evidence.to_record() != value:
        raise CanonicalEvidenceValidationError("failure evidence digest mismatch")
    return VerifiedCanonicalFailureEvidence(
        path=path,
        evidence=evidence,
        file_sha256=sha256_bytes(data),
    )


def _summary_record_without_digest(summary: CanonicalFailureSummary) -> dict[str, Any]:
    return {
        "schema_version": summary.schema_version,
        "total_failure_count": summary.total_failure_count,
        "failure_count_by_stage": dict(summary.failure_count_by_stage),
        "failure_count_by_category": dict(summary.failure_count_by_category),
    }


def summarize_canonical_failures(
    failures: Sequence[CanonicalFailureEvidence],
) -> CanonicalFailureSummary:
    frozen = tuple(failures)
    if any(not isinstance(item, CanonicalFailureEvidence) for item in frozen):
        raise TypeError("failures must contain CanonicalFailureEvidence values")
    by_stage = tuple(sorted(Counter(item.stage.value for item in frozen).items()))
    by_category = tuple(sorted(Counter(item.error_category for item in frozen).items()))
    provisional = CanonicalFailureSummary(
        schema_version=CANONICAL_FAILURE_SUMMARY_SCHEMA_VERSION,
        total_failure_count=len(frozen),
        failure_count_by_stage=by_stage,
        failure_count_by_category=by_category,
        summary_digest="",
    )
    return replace(
        provisional,
        summary_digest=sha256_bytes(
            canonical_json_bytes(_summary_record_without_digest(provisional))
        ),
    )
