"""Crash-atomic Canonical Collection Plan and Attempt Ledger contracts."""

from __future__ import annotations

import errno
import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

from werewolf.artifact_io import (
    ArtifactValidationError,
    canonical_json_bytes,
    sha256_bytes,
    verify_artifact,
)
from werewolf.artifact_io.canonical import ensure_durable_directory


COLLECTION_PLAN_SCHEMA_VERSION = "classic7_collection_plan_v1"
ATTEMPT_CLAIM_SCHEMA_VERSION = "classic7_attempt_claim_v1"
ATTEMPT_TERMINAL_SCHEMA_VERSION = "classic7_attempt_terminal_v1"
CANONICAL_GAME_BUNDLE_ARTIFACT_TYPE = "canonical_game_bundle"
_FINAL_RECORD_NAME = re.compile(
    r"^(?P<ordinal>0|[1-9][0-9]*)-(?P<seed>0|[1-9][0-9]*)"
    r"\.(?P<record_type>claim|terminal)\.json$"
)


class TerminalOutcome(str, Enum):
    CANONICAL_SUCCESS = "canonical_success"
    CANONICAL_FAILURE = "canonical_failure"
    INTERRUPTED_FAILURE = "interrupted_failure"


class LedgerPublishStep(str, Enum):
    STAGING_CREATED = "staging_created"
    COMPLETE_WRITE = "complete_write"
    FILE_FSYNCED = "file_fsynced"
    HARD_LINK_PUBLISHED = "hard_link_published"
    LEDGER_DIRECTORY_FSYNCED = "ledger_directory_fsynced"


class LedgerValidationError(ValueError):
    """An authoritative ledger record or state is invalid."""


class LedgerRecordConflictError(FileExistsError):
    """A final immutable ledger path already exists."""


class LedgerDurabilityUnsupportedError(RuntimeError):
    """The filesystem cannot provide the required durable publication."""


class InjectedLedgerCrash(RuntimeError):
    """A deterministic test crash at one publication boundary."""


class CollectionTargetReached(RuntimeError):
    """The Collection Plan already has its target number of successes."""


class CollectionSeedPoolExhausted(RuntimeError):
    """No unclaimed planned seed remains before target success."""


@dataclass(frozen=True)
class CollectionPlan:
    schema_version: str
    collection_id: str
    ordered_seed_pool: tuple[int, ...]
    target_canonical_success_count: int
    runtime_identity: str
    agent_identity: str
    backend_identity: str
    model_identity: str
    parser_identity: str
    prompt_identity: str
    retry_policy_identity: str
    call_budget_identity: str
    public_event_schema_version: str
    pre_prefix_schema_version: str
    belief_observation_schema_version: str
    v1_annotation_schema_version: str
    bundle_schema_version: str
    source_revision: str
    environment_provenance: tuple[tuple[str, str], ...]
    runtime_provenance_digest: str
    plan_digest: str

    def to_record(self) -> dict[str, Any]:
        return {**_plan_record_without_digest(self), "plan_digest": self.plan_digest}


@dataclass(frozen=True)
class AttemptClaim:
    schema_version: str
    plan_digest: str
    collection_id: str
    ordinal: int
    seed: int
    attempt_id: str
    runtime_provenance_digest: str
    claim_timestamp_utc: str
    record_digest: str

    def to_record(self) -> dict[str, Any]:
        return {
            **_claim_record_without_digest(self),
            "record_digest": self.record_digest,
        }


@dataclass(frozen=True)
class PartialEvidenceReference:
    relative_path: str
    sha256: str

    def to_record(self) -> dict[str, str]:
        return {"relative_path": self.relative_path, "sha256": self.sha256}


@dataclass(frozen=True)
class AttemptTerminal:
    schema_version: str
    plan_digest: str
    collection_id: str
    ordinal: int
    seed: int
    attempt_id: str
    runtime_provenance_digest: str
    outcome: TerminalOutcome
    canonical_game_bundle_id: str | None
    canonical_game_bundle_digest: str | None
    failure_evidence_digest: str | None
    partial_evidence: tuple[PartialEvidenceReference, ...]
    terminal_timestamp_utc: str
    record_digest: str

    def to_record(self) -> dict[str, Any]:
        return {
            **_terminal_record_without_digest(self),
            "record_digest": self.record_digest,
        }


@dataclass(frozen=True)
class PlannedAttempt:
    ordinal: int
    seed: int


@dataclass(frozen=True)
class AttemptLedgerState:
    plan_digest: str
    claims: tuple[AttemptClaim, ...]
    terminals: tuple[AttemptTerminal, ...]
    canonical_success_count: int
    open_claim: AttemptClaim | None
    target_reached: bool


def _required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty text")
    return value


def _sha256(value: Any, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _optional_sha256(value: Any, field_name: str) -> str | None:
    return None if value is None else _sha256(value, field_name)


def _utc_timestamp(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{field_name} must be an RFC 3339 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError(
            f"{field_name} must be an RFC 3339 UTC timestamp"
        ) from error
    if parsed.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must be an RFC 3339 UTC timestamp")
    return value


def _ordinal(value: Any, plan: CollectionPlan) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("attempt ordinal must be a non-negative integer")
    if value >= len(plan.ordered_seed_pool):
        raise ValueError("attempt ordinal is outside the ordered seed pool")
    return value


def _ordered_seeds(value: Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("ordered seed pool must be a sequence")
    seeds = tuple(value)
    if not seeds:
        raise ValueError("ordered seed pool must be non-empty")
    if any(
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
        for seed in seeds
    ):
        raise ValueError("collection seeds must be non-negative integers")
    if len(seeds) != len(set(seeds)):
        raise ValueError("ordered seed pool must contain unique seeds")
    return seeds


def _target(value: Any, seed_count: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("target canonical-success count must be a positive integer")
    if value > seed_count:
        raise ValueError("target canonical-success count exceeds the seed pool")
    return value


def _environment_provenance(
    value: Mapping[str, str],
) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("environment provenance must be a non-empty mapping")
    items: list[tuple[str, str]] = []
    for key, item in value.items():
        items.append(
            (
                _required_text(key, "environment provenance key"),
                _required_text(item, f"environment provenance value for {key!r}"),
            )
        )
    if len({key for key, _ in items}) != len(items):
        raise ValueError("environment provenance keys must be unique")
    return tuple(sorted(items))


def _runtime_provenance_record(plan: CollectionPlan) -> dict[str, Any]:
    return {
        "runtime_identity": plan.runtime_identity,
        "agent_identity": plan.agent_identity,
        "backend_identity": plan.backend_identity,
        "model_identity": plan.model_identity,
        "parser_identity": plan.parser_identity,
        "prompt_identity": plan.prompt_identity,
        "retry_policy_identity": plan.retry_policy_identity,
        "call_budget_identity": plan.call_budget_identity,
        "source_revision": plan.source_revision,
        "environment_provenance": dict(plan.environment_provenance),
    }


def _runtime_provenance_digest(plan: CollectionPlan) -> str:
    return sha256_bytes(canonical_json_bytes(_runtime_provenance_record(plan)))


def _plan_record_without_digest(plan: CollectionPlan) -> dict[str, Any]:
    return {
        "schema_version": plan.schema_version,
        "collection_id": plan.collection_id,
        "ordered_seed_pool": list(plan.ordered_seed_pool),
        "target_canonical_success_count": plan.target_canonical_success_count,
        **_runtime_provenance_record(plan),
        "public_event_schema_version": plan.public_event_schema_version,
        "pre_prefix_schema_version": plan.pre_prefix_schema_version,
        "belief_observation_schema_version": (
            plan.belief_observation_schema_version
        ),
        "v1_annotation_schema_version": plan.v1_annotation_schema_version,
        "bundle_schema_version": plan.bundle_schema_version,
        "runtime_provenance_digest": plan.runtime_provenance_digest,
    }


def construct_collection_plan(
    *,
    collection_id: str,
    ordered_seed_pool: Sequence[int],
    target_canonical_success_count: int,
    runtime_identity: str,
    agent_identity: str,
    backend_identity: str,
    model_identity: str,
    parser_identity: str,
    prompt_identity: str,
    retry_policy_identity: str,
    call_budget_identity: str,
    public_event_schema_version: str,
    pre_prefix_schema_version: str,
    belief_observation_schema_version: str,
    v1_annotation_schema_version: str,
    bundle_schema_version: str,
    source_revision: str,
    environment_provenance: Mapping[str, str],
) -> CollectionPlan:
    """Freeze one production-only Collection Plan identity."""

    seeds = _ordered_seeds(ordered_seed_pool)
    provisional = CollectionPlan(
        schema_version=COLLECTION_PLAN_SCHEMA_VERSION,
        collection_id=_required_text(collection_id, "collection_id"),
        ordered_seed_pool=seeds,
        target_canonical_success_count=_target(
            target_canonical_success_count,
            len(seeds),
        ),
        runtime_identity=_required_text(runtime_identity, "runtime_identity"),
        agent_identity=_required_text(agent_identity, "agent_identity"),
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
        public_event_schema_version=_required_text(
            public_event_schema_version,
            "public_event_schema_version",
        ),
        pre_prefix_schema_version=_required_text(
            pre_prefix_schema_version,
            "pre_prefix_schema_version",
        ),
        belief_observation_schema_version=_required_text(
            belief_observation_schema_version,
            "belief_observation_schema_version",
        ),
        v1_annotation_schema_version=_required_text(
            v1_annotation_schema_version,
            "v1_annotation_schema_version",
        ),
        bundle_schema_version=_required_text(
            bundle_schema_version,
            "bundle_schema_version",
        ),
        source_revision=_required_text(source_revision, "source_revision"),
        environment_provenance=_environment_provenance(
            environment_provenance
        ),
        runtime_provenance_digest="",
        plan_digest="",
    )
    with_runtime_digest = replace(
        provisional,
        runtime_provenance_digest=_runtime_provenance_digest(provisional),
    )
    return replace(
        with_runtime_digest,
        plan_digest=sha256_bytes(
            canonical_json_bytes(_plan_record_without_digest(with_runtime_digest))
        ),
    )


def validate_collection_plan(plan: CollectionPlan) -> CollectionPlan:
    """Validate one frozen Collection Plan without repairing it."""

    if not isinstance(plan, CollectionPlan):
        raise TypeError("plan must be a CollectionPlan")
    if plan.schema_version != COLLECTION_PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported Collection Plan schema_version")
    _required_text(plan.collection_id, "collection_id")
    seeds = _ordered_seeds(plan.ordered_seed_pool)
    if seeds != plan.ordered_seed_pool:
        raise ValueError("ordered seed pool is not frozen canonically")
    _target(plan.target_canonical_success_count, len(seeds))
    for field_name in (
        "runtime_identity",
        "agent_identity",
        "backend_identity",
        "model_identity",
        "parser_identity",
        "prompt_identity",
        "retry_policy_identity",
        "call_budget_identity",
        "public_event_schema_version",
        "pre_prefix_schema_version",
        "belief_observation_schema_version",
        "v1_annotation_schema_version",
        "bundle_schema_version",
        "source_revision",
    ):
        _required_text(getattr(plan, field_name), field_name)
    if _environment_provenance(dict(plan.environment_provenance)) != (
        plan.environment_provenance
    ):
        raise ValueError("environment provenance is not frozen canonically")
    if plan.runtime_provenance_digest != _runtime_provenance_digest(plan):
        raise ValueError("Collection Plan runtime provenance digest mismatch")
    expected_plan_digest = sha256_bytes(
        canonical_json_bytes(_plan_record_without_digest(plan))
    )
    if plan.plan_digest != expected_plan_digest:
        raise ValueError("Collection Plan digest mismatch")
    return plan


def _claim_record_without_digest(claim: AttemptClaim) -> dict[str, Any]:
    return {
        "schema_version": claim.schema_version,
        "record_type": "claim",
        "plan_digest": claim.plan_digest,
        "collection_id": claim.collection_id,
        "ordinal": claim.ordinal,
        "seed": claim.seed,
        "attempt_id": claim.attempt_id,
        "runtime_provenance_digest": claim.runtime_provenance_digest,
        "claim_timestamp_utc": claim.claim_timestamp_utc,
    }


def construct_attempt_claim(
    plan: CollectionPlan,
    *,
    ordinal: int,
    attempt_id: str,
    claim_timestamp_utc: str,
) -> AttemptClaim:
    """Bind one planned seed before its sole permitted attempt."""

    validate_collection_plan(plan)
    ordinal = _ordinal(ordinal, plan)
    provisional = AttemptClaim(
        schema_version=ATTEMPT_CLAIM_SCHEMA_VERSION,
        plan_digest=plan.plan_digest,
        collection_id=plan.collection_id,
        ordinal=ordinal,
        seed=plan.ordered_seed_pool[ordinal],
        attempt_id=_safe_artifact_identity(attempt_id, "attempt_id"),
        runtime_provenance_digest=plan.runtime_provenance_digest,
        claim_timestamp_utc=_utc_timestamp(
            claim_timestamp_utc,
            "claim_timestamp_utc",
        ),
        record_digest="",
    )
    return replace(
        provisional,
        record_digest=sha256_bytes(
            canonical_json_bytes(_claim_record_without_digest(provisional))
        ),
    )


def validate_attempt_claim(
    claim: AttemptClaim,
    plan: CollectionPlan,
) -> AttemptClaim:
    """Validate a claim against its exact immutable Collection Plan."""

    validate_collection_plan(plan)
    if not isinstance(claim, AttemptClaim):
        raise TypeError("claim must be an AttemptClaim")
    if claim.schema_version != ATTEMPT_CLAIM_SCHEMA_VERSION:
        raise ValueError("unsupported attempt claim schema_version")
    if claim.plan_digest != plan.plan_digest:
        raise ValueError("attempt claim Collection Plan digest mismatch")
    if claim.collection_id != plan.collection_id:
        raise ValueError("attempt claim collection identity mismatch")
    ordinal = _ordinal(claim.ordinal, plan)
    if claim.seed != plan.ordered_seed_pool[ordinal]:
        raise ValueError("attempt claim seed does not match its plan ordinal")
    _safe_artifact_identity(claim.attempt_id, "attempt_id")
    if claim.runtime_provenance_digest != plan.runtime_provenance_digest:
        raise ValueError("attempt claim runtime provenance mismatch")
    _utc_timestamp(claim.claim_timestamp_utc, "claim_timestamp_utc")
    expected_digest = sha256_bytes(
        canonical_json_bytes(_claim_record_without_digest(claim))
    )
    if claim.record_digest != expected_digest:
        raise ValueError("attempt claim record digest mismatch")
    return claim


def _safe_artifact_identity(value: Any, field_name: str) -> str:
    identity = _required_text(value, field_name)
    if identity in {".", ".."} or "/" in identity or "\\" in identity:
        raise ValueError(f"{field_name} must be one safe path component")
    return identity


def _partial_evidence_references(
    values: Sequence[PartialEvidenceReference],
    attempt_id: str,
) -> tuple[PartialEvidenceReference, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("partial_evidence must be a sequence")
    references = tuple(values)
    expected_prefix = PurePosixPath(
        "attempts",
        attempt_id,
        "partial_evidence",
    )
    seen_paths: set[str] = set()
    for reference in references:
        if not isinstance(reference, PartialEvidenceReference):
            raise TypeError(
                "partial_evidence must contain PartialEvidenceReference values"
            )
        path = PurePosixPath(reference.relative_path)
        if (
            path.is_absolute()
            or path.as_posix() != reference.relative_path
            or ".." in path.parts
            or path.parent == PurePosixPath(".")
            or not path.is_relative_to(expected_prefix)
            or path == expected_prefix
        ):
            raise ValueError("partial evidence path is outside its claimed attempt")
        if reference.relative_path in seen_paths:
            raise ValueError("partial evidence paths must be unique")
        _sha256(reference.sha256, "partial evidence digest")
        seen_paths.add(reference.relative_path)
    if references != tuple(sorted(references, key=lambda item: item.relative_path)):
        raise ValueError("partial evidence references must use canonical path order")
    return references


def _validate_terminal_evidence(
    outcome: TerminalOutcome,
    canonical_game_bundle_id: str | None,
    canonical_game_bundle_digest: str | None,
    failure_evidence_digest: str | None,
    partial_evidence: tuple[PartialEvidenceReference, ...],
) -> None:
    if outcome is TerminalOutcome.CANONICAL_SUCCESS:
        if (
            canonical_game_bundle_id is None
            or canonical_game_bundle_digest is None
        ):
            raise ValueError("canonical success requires a bundle digest")
        if failure_evidence_digest is not None or partial_evidence:
            raise ValueError("canonical success cannot bind failure evidence")
    elif outcome is TerminalOutcome.CANONICAL_FAILURE:
        if failure_evidence_digest is None:
            raise ValueError("canonical failure requires a failure evidence digest")
        if (
            canonical_game_bundle_id is not None
            or canonical_game_bundle_digest is not None
        ):
            raise ValueError("canonical failure cannot bind a bundle digest")
    elif outcome is TerminalOutcome.INTERRUPTED_FAILURE:
        if (
            canonical_game_bundle_id is not None
            or canonical_game_bundle_digest is not None
        ):
            raise ValueError("interrupted failure cannot bind a bundle digest")


def _terminal_record_without_digest(
    terminal: AttemptTerminal,
) -> dict[str, Any]:
    return {
        "schema_version": terminal.schema_version,
        "record_type": "terminal",
        "plan_digest": terminal.plan_digest,
        "collection_id": terminal.collection_id,
        "ordinal": terminal.ordinal,
        "seed": terminal.seed,
        "attempt_id": terminal.attempt_id,
        "runtime_provenance_digest": terminal.runtime_provenance_digest,
        "outcome": terminal.outcome.value,
        "canonical_game_bundle_id": terminal.canonical_game_bundle_id,
        "canonical_game_bundle_digest": terminal.canonical_game_bundle_digest,
        "failure_evidence_digest": terminal.failure_evidence_digest,
        "partial_evidence": [
            reference.to_record() for reference in terminal.partial_evidence
        ],
        "terminal_timestamp_utc": terminal.terminal_timestamp_utc,
    }


def construct_attempt_terminal(
    plan: CollectionPlan,
    claim: AttemptClaim,
    *,
    outcome: TerminalOutcome,
    terminal_timestamp_utc: str,
    canonical_game_bundle_id: str | None = None,
    canonical_game_bundle_digest: str | None = None,
    failure_evidence_digest: str | None = None,
    partial_evidence: Sequence[PartialEvidenceReference] = (),
) -> AttemptTerminal:
    """Close a claimed attempt with exactly one immutable terminal outcome."""

    validate_attempt_claim(claim, plan)
    if not isinstance(outcome, TerminalOutcome):
        raise TypeError("outcome must be a TerminalOutcome")
    bundle_id = (
        None
        if canonical_game_bundle_id is None
        else _safe_artifact_identity(
            canonical_game_bundle_id,
            "canonical game bundle ID",
        )
    )
    bundle_digest = _optional_sha256(
        canonical_game_bundle_digest,
        "canonical game bundle digest",
    )
    failure_digest = _optional_sha256(
        failure_evidence_digest,
        "failure evidence digest",
    )
    partial_references = _partial_evidence_references(
        partial_evidence,
        claim.attempt_id,
    )
    _validate_terminal_evidence(
        outcome,
        bundle_id,
        bundle_digest,
        failure_digest,
        partial_references,
    )
    provisional = AttemptTerminal(
        schema_version=ATTEMPT_TERMINAL_SCHEMA_VERSION,
        plan_digest=plan.plan_digest,
        collection_id=plan.collection_id,
        ordinal=claim.ordinal,
        seed=claim.seed,
        attempt_id=claim.attempt_id,
        runtime_provenance_digest=plan.runtime_provenance_digest,
        outcome=outcome,
        canonical_game_bundle_id=bundle_id,
        canonical_game_bundle_digest=bundle_digest,
        failure_evidence_digest=failure_digest,
        partial_evidence=partial_references,
        terminal_timestamp_utc=_utc_timestamp(
            terminal_timestamp_utc,
            "terminal_timestamp_utc",
        ),
        record_digest="",
    )
    return replace(
        provisional,
        record_digest=sha256_bytes(
            canonical_json_bytes(_terminal_record_without_digest(provisional))
        ),
    )


def validate_attempt_terminal(
    terminal: AttemptTerminal,
    plan: CollectionPlan,
    claim: AttemptClaim,
) -> AttemptTerminal:
    """Validate a terminal against its exact plan and prior claim."""

    validate_attempt_claim(claim, plan)
    if not isinstance(terminal, AttemptTerminal):
        raise TypeError("terminal must be an AttemptTerminal")
    if terminal.schema_version != ATTEMPT_TERMINAL_SCHEMA_VERSION:
        raise ValueError("unsupported attempt terminal schema_version")
    if (
        terminal.plan_digest != plan.plan_digest
        or terminal.collection_id != plan.collection_id
    ):
        raise ValueError("attempt terminal Collection Plan identity mismatch")
    if (
        terminal.ordinal != claim.ordinal
        or terminal.seed != claim.seed
        or terminal.attempt_id != claim.attempt_id
    ):
        raise ValueError("attempt terminal does not match its claim")
    if terminal.runtime_provenance_digest != plan.runtime_provenance_digest:
        raise ValueError("attempt terminal runtime provenance mismatch")
    if not isinstance(terminal.outcome, TerminalOutcome):
        raise TypeError("attempt terminal outcome must be TerminalOutcome")
    bundle_id = (
        None
        if terminal.canonical_game_bundle_id is None
        else _safe_artifact_identity(
            terminal.canonical_game_bundle_id,
            "canonical game bundle ID",
        )
    )
    bundle_digest = _optional_sha256(
        terminal.canonical_game_bundle_digest,
        "canonical game bundle digest",
    )
    failure_digest = _optional_sha256(
        terminal.failure_evidence_digest,
        "failure evidence digest",
    )
    partial_references = _partial_evidence_references(
        terminal.partial_evidence,
        terminal.attempt_id,
    )
    _validate_terminal_evidence(
        terminal.outcome,
        bundle_id,
        bundle_digest,
        failure_digest,
        partial_references,
    )
    _utc_timestamp(terminal.terminal_timestamp_utc, "terminal_timestamp_utc")
    expected_digest = sha256_bytes(
        canonical_json_bytes(_terminal_record_without_digest(terminal))
    )
    if terminal.record_digest != expected_digest:
        raise ValueError("attempt terminal record digest mismatch")
    return terminal


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _durability_error(error: OSError) -> LedgerDurabilityUnsupportedError:
    return LedgerDurabilityUnsupportedError(
        "filesystem does not support crash-atomic Attempt Ledger publication"
    )


def _ensure_attempt_ledger_directories(
    ledger_directory: Path | str,
) -> tuple[Path, Path]:
    ledger_directory = Path(ledger_directory)
    try:
        ensure_durable_directory(ledger_directory.parent)
    except OSError as error:
        raise _durability_error(error) from error
    if ledger_directory.is_symlink():
        raise LedgerValidationError("Attempt Ledger directory cannot be a symlink")
    created_ledger = not ledger_directory.exists()
    ledger_directory.mkdir(exist_ok=True)
    if not ledger_directory.is_dir():
        raise LedgerValidationError("Attempt Ledger path must be a directory")
    staging_directory = ledger_directory / ".staging"
    if staging_directory.is_symlink():
        raise LedgerValidationError("Attempt Ledger staging cannot be a symlink")
    staging_directory.mkdir(exist_ok=True)
    if not staging_directory.is_dir():
        raise LedgerValidationError("Attempt Ledger staging must be a directory")
    if created_ledger:
        try:
            _fsync_directory(ledger_directory.parent)
        except OSError as error:
            raise _durability_error(error) from error
    return ledger_directory, staging_directory


def preflight_attempt_ledger(ledger_directory: Path | str) -> Path:
    """Prove hard-link no-replace and directory-fsync support."""

    ledger_directory, staging_directory = _ensure_attempt_ledger_directories(
        ledger_directory
    )

    source = staging_directory / f".preflight-{uuid4().hex}"
    linked = ledger_directory / f".preflight-link-{uuid4().hex}"
    descriptor: int | None = None
    try:
        _fsync_directory(ledger_directory)
        descriptor = os.open(source, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        _write_all(descriptor, b"ledger-preflight")
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        if os.stat(source).st_dev != os.stat(ledger_directory).st_dev:
            raise LedgerDurabilityUnsupportedError(
                "Attempt Ledger staging and final paths must share a filesystem"
            )
        os.link(source, linked, follow_symlinks=False)
        try:
            os.link(source, linked, follow_symlinks=False)
        except FileExistsError:
            pass
        else:
            raise LedgerDurabilityUnsupportedError(
                "hard-link publication did not enforce no-replace semantics"
            )
        _fsync_directory(staging_directory)
        _fsync_directory(ledger_directory)
    except LedgerDurabilityUnsupportedError:
        raise
    except OSError as error:
        raise _durability_error(error) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        for path in (linked, source):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        try:
            _fsync_directory(staging_directory)
            _fsync_directory(ledger_directory)
        except OSError:
            pass
    return ledger_directory


def _record_path(
    ledger_directory: Path,
    record: AttemptClaim | AttemptTerminal,
) -> Path:
    record_type = "claim" if isinstance(record, AttemptClaim) else "terminal"
    return ledger_directory / (
        f"{record.ordinal}-{record.seed}.{record_type}.json"
    )


_PLAN_RECORD_FIELDS = frozenset(
    {
        "schema_version",
        "collection_id",
        "ordered_seed_pool",
        "target_canonical_success_count",
        "runtime_identity",
        "agent_identity",
        "backend_identity",
        "model_identity",
        "parser_identity",
        "prompt_identity",
        "retry_policy_identity",
        "call_budget_identity",
        "source_revision",
        "environment_provenance",
        "public_event_schema_version",
        "pre_prefix_schema_version",
        "belief_observation_schema_version",
        "v1_annotation_schema_version",
        "bundle_schema_version",
        "runtime_provenance_digest",
        "plan_digest",
    }
)


def collection_plan_from_record(value: dict[str, Any]) -> CollectionPlan:
    if set(value) != _PLAN_RECORD_FIELDS:
        raise LedgerValidationError("collection_plan.json has an invalid field set")
    environment = value["environment_provenance"]
    seeds = value["ordered_seed_pool"]
    if not isinstance(environment, dict) or not isinstance(seeds, list):
        raise LedgerValidationError("collection_plan.json has invalid value types")
    try:
        plan = construct_collection_plan(
            collection_id=value["collection_id"],
            ordered_seed_pool=seeds,
            target_canonical_success_count=value[
                "target_canonical_success_count"
            ],
            runtime_identity=value["runtime_identity"],
            agent_identity=value["agent_identity"],
            backend_identity=value["backend_identity"],
            model_identity=value["model_identity"],
            parser_identity=value["parser_identity"],
            prompt_identity=value["prompt_identity"],
            retry_policy_identity=value["retry_policy_identity"],
            call_budget_identity=value["call_budget_identity"],
            public_event_schema_version=value["public_event_schema_version"],
            pre_prefix_schema_version=value["pre_prefix_schema_version"],
            belief_observation_schema_version=value[
                "belief_observation_schema_version"
            ],
            v1_annotation_schema_version=value[
                "v1_annotation_schema_version"
            ],
            bundle_schema_version=value["bundle_schema_version"],
            source_revision=value["source_revision"],
            environment_provenance=environment,
        )
    except (TypeError, ValueError) as error:
        raise LedgerValidationError("invalid collection_plan.json") from error
    if plan.to_record() != value:
        raise LedgerValidationError("collection_plan.json digest mismatch")
    return plan


def _validate_bound_collection_plan(
    collection_directory: Path,
    expected_plan: CollectionPlan,
) -> CollectionPlan:
    plan_path = collection_directory / "collection_plan.json"
    if plan_path.is_symlink() or not plan_path.is_file():
        raise LedgerValidationError("collection_plan.json is missing or invalid")
    plan = collection_plan_from_record(_load_canonical_record(plan_path))
    if plan.plan_digest != expected_plan.plan_digest:
        raise LedgerRecordConflictError(
            "collection directory is bound to a different Collection Plan"
        )
    return plan


def load_collection_plan(collection_directory: Path | str) -> CollectionPlan:
    """Open the immutable Collection Plan bound to one collection directory."""

    collection_directory = Path(collection_directory)
    plan_path = collection_directory / "collection_plan.json"
    if plan_path.is_symlink() or not plan_path.is_file():
        raise LedgerValidationError("collection_plan.json is missing or invalid")
    return collection_plan_from_record(_load_canonical_record(plan_path))


def initialize_attempt_ledger(
    collection_directory: Path | str,
    plan: CollectionPlan,
) -> Path:
    """Durably bind one Collection Plan and initialize its Attempt Ledger."""

    validate_collection_plan(plan)
    collection_directory = Path(collection_directory)
    ledger_directory = preflight_attempt_ledger(
        collection_directory / "attempt_ledger"
    )
    plan_path = collection_directory / "collection_plan.json"
    if os.path.lexists(plan_path):
        _validate_bound_collection_plan(collection_directory, plan)
        try:
            _fsync_directory(collection_directory)
        except OSError as error:
            raise _durability_error(error) from error
        return ledger_directory
    try:
        _publish_bytes_noreplace(
            staging_directory=ledger_directory / ".staging",
            final_path=plan_path,
            durable_directory=collection_directory,
            data=canonical_json_bytes(plan.to_record()),
        )
    except LedgerRecordConflictError:
        _validate_bound_collection_plan(collection_directory, plan)
        try:
            _fsync_directory(collection_directory)
        except OSError as error:
            raise _durability_error(error) from error
    _validate_bound_collection_plan(collection_directory, plan)
    return ledger_directory


def _load_canonical_record(path: Path) -> dict[str, Any]:
    try:
        data = path.read_bytes()
        value = json.loads(data.decode("utf-8"))
        if not isinstance(value, dict) or canonical_json_bytes(value) != data:
            raise ValueError("ledger record is not canonical JSON")
        return value
    except (OSError, UnicodeError, ValueError, TypeError) as error:
        raise LedgerValidationError(f"invalid ledger record: {path.name}") from error


def _read_fsynced_regular_file(path: Path, description: str) -> bytes:
    if path.is_symlink():
        raise LedgerValidationError(f"{description} cannot be a symlink")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise LedgerValidationError(f"{description} is missing or invalid") from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise LedgerValidationError(f"{description} must be a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        os.fsync(descriptor)
    except OSError as error:
        raise LedgerValidationError(f"{description} cannot be verified") from error
    finally:
        os.close(descriptor)
    try:
        _fsync_directory(path.parent)
    except OSError as error:
        raise LedgerValidationError(
            f"{description} parent directory is not durable"
        ) from error
    return b"".join(chunks)


def _failure_evidence_path(
    collection_directory: Path,
    attempt_id: str,
) -> Path:
    return _validated_attempt_directory(
        collection_directory,
        attempt_id,
    ) / "failure_evidence.json"


def _validated_attempt_directory(
    collection_directory: Path,
    attempt_id: str,
) -> Path:
    _safe_artifact_identity(attempt_id, "attempt_id")
    attempts_directory = collection_directory / "attempts"
    if os.path.lexists(attempts_directory) and (
        attempts_directory.is_symlink() or not attempts_directory.is_dir()
    ):
        raise LedgerValidationError("attempts directory is invalid")
    attempt_directory = attempts_directory / attempt_id
    if os.path.lexists(attempt_directory) and (
        attempt_directory.is_symlink() or not attempt_directory.is_dir()
    ):
        raise LedgerValidationError("attempt evidence directory is invalid")
    return attempt_directory


def _actual_failure_evidence_digest(
    collection_directory: Path,
    attempt_id: str,
) -> str | None:
    path = _failure_evidence_path(collection_directory, attempt_id)
    if not os.path.lexists(path):
        return None
    return sha256_bytes(_read_fsynced_regular_file(path, "failure evidence"))


def _scan_partial_evidence(
    collection_directory: Path,
    attempt_id: str,
) -> tuple[PartialEvidenceReference, ...]:
    root = _validated_attempt_directory(
        collection_directory,
        attempt_id,
    ) / "partial_evidence"
    if not os.path.lexists(root):
        return ()
    if root.is_symlink() or not root.is_dir():
        raise LedgerValidationError("partial evidence root is invalid")
    references: list[PartialEvidenceReference] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise LedgerValidationError("partial evidence cannot contain symlinks")
        if path.is_dir():
            continue
        data = _read_fsynced_regular_file(path, "partial evidence")
        references.append(
            PartialEvidenceReference(
                relative_path=path.relative_to(collection_directory).as_posix(),
                sha256=sha256_bytes(data),
            )
        )
    try:
        _fsync_directory(root)
    except OSError as error:
        raise LedgerValidationError(
            "partial evidence directory is not durable"
        ) from error
    return tuple(references)


def _fsync_attempt_evidence_lineage(
    collection_directory: Path,
    attempt_id: str,
) -> None:
    attempt_directory = _validated_attempt_directory(
        collection_directory,
        attempt_id,
    )
    if not os.path.lexists(attempt_directory):
        return
    directories = [
        path
        for path in attempt_directory.rglob("*")
        if path.is_dir() and not path.is_symlink()
    ]
    if any(path.is_symlink() for path in attempt_directory.rglob("*")):
        raise LedgerValidationError("attempt evidence cannot contain symlinks")
    directories.sort(key=lambda path: len(path.parts), reverse=True)
    directories.extend(
        [
            attempt_directory,
            attempt_directory.parent,
            collection_directory,
        ]
    )
    try:
        for directory in directories:
            _fsync_directory(directory)
    except OSError as error:
        raise LedgerValidationError(
            "attempt evidence directory lineage is not durable"
        ) from error


def _verify_terminal_artifacts_if_needed(
    record: AttemptClaim | AttemptTerminal,
    plan: CollectionPlan,
    collection_directory: Path,
) -> None:
    if isinstance(record, AttemptClaim):
        return
    actual_failure_digest = _actual_failure_evidence_digest(
        collection_directory,
        record.attempt_id,
    )
    if actual_failure_digest != record.failure_evidence_digest:
        raise LedgerValidationError("failure evidence digest mismatch")
    actual_partial_evidence = _scan_partial_evidence(
        collection_directory,
        record.attempt_id,
    )
    if actual_partial_evidence != record.partial_evidence:
        raise LedgerValidationError("partial evidence set or digest mismatch")
    _fsync_attempt_evidence_lineage(
        collection_directory,
        record.attempt_id,
    )

    if record.outcome is not TerminalOutcome.CANONICAL_SUCCESS:
        return
    assert record.canonical_game_bundle_id is not None
    assert record.canonical_game_bundle_digest is not None
    try:
        verified = verify_artifact(
            collection_directory
            / "games"
            / record.canonical_game_bundle_id,
            expected_artifact_type=CANONICAL_GAME_BUNDLE_ARTIFACT_TYPE,
            expected_schema_version=plan.bundle_schema_version,
        )
    except ArtifactValidationError as error:
        raise LedgerValidationError(
            "canonical game bundle is missing or invalid"
        ) from error
    if verified.manifest_digest != record.canonical_game_bundle_digest:
        raise LedgerValidationError("canonical game bundle digest mismatch")
    try:
        _fsync_directory(verified.path.parent)
        _fsync_directory(collection_directory)
    except OSError as error:
        raise LedgerValidationError(
            "canonical game bundle directory lineage is not durable"
        ) from error


_CLAIM_RECORD_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "plan_digest",
        "collection_id",
        "ordinal",
        "seed",
        "attempt_id",
        "runtime_provenance_digest",
        "claim_timestamp_utc",
        "record_digest",
    }
)
_TERMINAL_RECORD_FIELDS = frozenset(
    {
        "schema_version",
        "record_type",
        "plan_digest",
        "collection_id",
        "ordinal",
        "seed",
        "attempt_id",
        "runtime_provenance_digest",
        "outcome",
        "canonical_game_bundle_id",
        "canonical_game_bundle_digest",
        "failure_evidence_digest",
        "partial_evidence",
        "terminal_timestamp_utc",
        "record_digest",
    }
)


def _claim_from_record(value: dict[str, Any], plan: CollectionPlan) -> AttemptClaim:
    if set(value) != _CLAIM_RECORD_FIELDS or value.get("record_type") != "claim":
        raise LedgerValidationError("attempt claim has an invalid field set")
    claim = AttemptClaim(
        schema_version=value["schema_version"],
        plan_digest=value["plan_digest"],
        collection_id=value["collection_id"],
        ordinal=value["ordinal"],
        seed=value["seed"],
        attempt_id=value["attempt_id"],
        runtime_provenance_digest=value["runtime_provenance_digest"],
        claim_timestamp_utc=value["claim_timestamp_utc"],
        record_digest=value["record_digest"],
    )
    try:
        return validate_attempt_claim(claim, plan)
    except (TypeError, ValueError) as error:
        raise LedgerValidationError("invalid attempt claim") from error


def _terminal_from_record(
    value: dict[str, Any],
    plan: CollectionPlan,
) -> AttemptTerminal:
    if (
        set(value) != _TERMINAL_RECORD_FIELDS
        or value.get("record_type") != "terminal"
    ):
        raise LedgerValidationError("attempt terminal has an invalid field set")
    try:
        outcome = TerminalOutcome(value["outcome"])
    except (TypeError, ValueError) as error:
        raise LedgerValidationError("invalid attempt terminal outcome") from error
    partial_value = value["partial_evidence"]
    if not isinstance(partial_value, list):
        raise LedgerValidationError(
            "attempt terminal partial evidence must be a JSON array"
        )
    partial_references: list[PartialEvidenceReference] = []
    for reference in partial_value:
        if not isinstance(reference, dict) or set(reference) != {
            "relative_path",
            "sha256",
        }:
            raise LedgerValidationError(
                "attempt terminal has an invalid partial evidence reference"
            )
        partial_references.append(
            PartialEvidenceReference(
                relative_path=reference["relative_path"],
                sha256=reference["sha256"],
            )
        )
    return AttemptTerminal(
        schema_version=value["schema_version"],
        plan_digest=value["plan_digest"],
        collection_id=value["collection_id"],
        ordinal=value["ordinal"],
        seed=value["seed"],
        attempt_id=value["attempt_id"],
        runtime_provenance_digest=value["runtime_provenance_digest"],
        outcome=outcome,
        canonical_game_bundle_id=value["canonical_game_bundle_id"],
        canonical_game_bundle_digest=value["canonical_game_bundle_digest"],
        failure_evidence_digest=value["failure_evidence_digest"],
        partial_evidence=tuple(partial_references),
        terminal_timestamp_utc=value["terminal_timestamp_utc"],
        record_digest=value["record_digest"],
    )


def _state_from_records(
    plan: CollectionPlan,
    claims: Sequence[AttemptClaim],
    terminals: Sequence[AttemptTerminal],
) -> AttemptLedgerState:
    claims_by_ordinal: dict[int, AttemptClaim] = {}
    attempt_ids: set[str] = set()
    for claim in claims:
        try:
            validate_attempt_claim(claim, plan)
        except (TypeError, ValueError) as error:
            raise LedgerValidationError("invalid attempt claim") from error
        if claim.ordinal in claims_by_ordinal:
            raise LedgerValidationError("duplicate attempt claim ordinal")
        if claim.attempt_id in attempt_ids:
            raise LedgerValidationError("duplicate attempt_id in Attempt Ledger")
        claims_by_ordinal[claim.ordinal] = claim
        attempt_ids.add(claim.attempt_id)
    ordered_ordinals = sorted(claims_by_ordinal)
    if ordered_ordinals != list(range(len(ordered_ordinals))):
        raise LedgerValidationError("Attempt Ledger has an out-of-order claim gap")

    terminals_by_ordinal: dict[int, AttemptTerminal] = {}
    for terminal in terminals:
        if terminal.ordinal in terminals_by_ordinal:
            raise LedgerValidationError("duplicate attempt terminal ordinal")
        claim = claims_by_ordinal.get(terminal.ordinal)
        if claim is None:
            raise LedgerValidationError("attempt terminal exists without a claim")
        try:
            validate_attempt_terminal(terminal, plan, claim)
        except (TypeError, ValueError) as error:
            raise LedgerValidationError("invalid attempt terminal") from error
        terminals_by_ordinal[terminal.ordinal] = terminal

    ordered_claims = tuple(claims_by_ordinal[index] for index in ordered_ordinals)
    open_claims = tuple(
        claim
        for claim in ordered_claims
        if claim.ordinal not in terminals_by_ordinal
    )
    if len(open_claims) > 1 or (
        open_claims and open_claims[0] is not ordered_claims[-1]
    ):
        raise LedgerValidationError("a new claim follows an unclosed attempt")

    success_count = 0
    for claim in ordered_claims:
        terminal = terminals_by_ordinal.get(claim.ordinal)
        if terminal is not None and (
            terminal.outcome is TerminalOutcome.CANONICAL_SUCCESS
        ):
            success_count += 1
        if success_count >= plan.target_canonical_success_count and (
            claim is not ordered_claims[-1]
        ):
            raise LedgerValidationError(
                "Attempt Ledger contains a claim after target success"
            )

    ordered_terminals = tuple(
        terminals_by_ordinal[index] for index in sorted(terminals_by_ordinal)
    )
    return AttemptLedgerState(
        plan_digest=plan.plan_digest,
        claims=ordered_claims,
        terminals=ordered_terminals,
        canonical_success_count=success_count,
        open_claim=open_claims[0] if open_claims else None,
        target_reached=success_count >= plan.target_canonical_success_count,
    )


def validate_attempt_ledger(
    ledger_directory: Path | str,
    plan: CollectionPlan,
) -> AttemptLedgerState:
    """Read and validate only authoritative final ledger records."""

    validate_collection_plan(plan)
    ledger_directory = Path(ledger_directory)
    if ledger_directory.is_symlink() or not ledger_directory.is_dir():
        raise LedgerValidationError("Attempt Ledger directory is missing or invalid")
    staging_directory = ledger_directory / ".staging"
    if staging_directory.is_symlink() or not staging_directory.is_dir():
        raise LedgerValidationError("Attempt Ledger staging is missing or invalid")
    _validate_bound_collection_plan(ledger_directory.parent, plan)

    claims: list[AttemptClaim] = []
    terminal_values: list[AttemptTerminal] = []
    for path in sorted(ledger_directory.iterdir(), key=lambda item: item.name):
        if path.name == ".staging":
            continue
        if path.is_symlink() or not path.is_file():
            raise LedgerValidationError(
                f"unexpected Attempt Ledger entry: {path.name}"
            )
        match = _FINAL_RECORD_NAME.fullmatch(path.name)
        if match is None:
            raise LedgerValidationError(
                f"unexpected Attempt Ledger record name: {path.name}"
            )
        value = _load_canonical_record(path)
        if match.group("record_type") == "claim":
            record: AttemptClaim | AttemptTerminal = _claim_from_record(value, plan)
            claims.append(record)
        else:
            record = _terminal_from_record(value, plan)
            terminal_values.append(record)
        if path != _record_path(ledger_directory, record):
            raise LedgerValidationError("Attempt Ledger filename does not match record")
    state = _state_from_records(plan, claims, terminal_values)
    for terminal in state.terminals:
        _verify_terminal_artifacts_if_needed(
            terminal,
            plan,
            ledger_directory.parent,
        )
    return state


def next_planned_attempt(
    plan: CollectionPlan,
    state: AttemptLedgerState,
) -> PlannedAttempt:
    """Return the sole next claim slot or fail at a terminal plan state."""

    validate_collection_plan(plan)
    if not isinstance(state, AttemptLedgerState) or state.plan_digest != plan.plan_digest:
        raise LedgerValidationError("Attempt Ledger state has the wrong plan identity")
    if state.open_claim is not None:
        raise LedgerValidationError("the current claimed attempt is not terminal")
    if state.target_reached:
        raise CollectionTargetReached("target canonical-success count reached")
    ordinal = len(state.claims)
    if ordinal >= len(plan.ordered_seed_pool):
        raise CollectionSeedPoolExhausted(
            "ordered seed pool exhausted before target canonical success"
        )
    return PlannedAttempt(ordinal=ordinal, seed=plan.ordered_seed_pool[ordinal])


def _validate_record_for_publication(
    plan: CollectionPlan,
    state: AttemptLedgerState,
    record: AttemptClaim | AttemptTerminal,
) -> None:
    if isinstance(record, AttemptClaim):
        validate_attempt_claim(record, plan)
        slot = next_planned_attempt(plan, state)
        if (record.ordinal, record.seed) != (slot.ordinal, slot.seed):
            raise LedgerValidationError("claim is not the next ordered plan seed")
        if record.attempt_id in {claim.attempt_id for claim in state.claims}:
            raise LedgerValidationError("attempt_id is already claimed")
        return
    if isinstance(record, AttemptTerminal):
        if state.open_claim is None:
            raise LedgerValidationError("terminal requires one open claim")
        validate_attempt_terminal(record, plan, state.open_claim)
        return
    raise TypeError("ledger record must be an AttemptClaim or AttemptTerminal")


def _write_all(descriptor: int, data: bytes) -> None:
    offset = 0
    while offset < len(data):
        written = os.write(descriptor, data[offset:])
        if written <= 0:
            raise OSError(errno.EIO, "ledger staging write made no progress")
        offset += written


def _inject_crash(
    crash_after: LedgerPublishStep | None,
    completed_step: LedgerPublishStep,
) -> None:
    if crash_after is completed_step:
        raise InjectedLedgerCrash(completed_step.value)


def _publish_bytes_noreplace(
    *,
    staging_directory: Path,
    final_path: Path,
    durable_directory: Path,
    data: bytes,
    crash_after: LedgerPublishStep | None = None,
) -> Path:
    if os.path.lexists(final_path):
        raise LedgerRecordConflictError(final_path)
    staging_path = staging_directory / (
        f"{final_path.name}.{uuid4().hex}.staging"
    )
    descriptor = os.open(
        staging_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        _inject_crash(crash_after, LedgerPublishStep.STAGING_CREATED)
        _write_all(descriptor, data)
        _inject_crash(crash_after, LedgerPublishStep.COMPLETE_WRITE)
        os.fsync(descriptor)
        _inject_crash(crash_after, LedgerPublishStep.FILE_FSYNCED)
    finally:
        os.close(descriptor)

    try:
        os.link(staging_path, final_path, follow_symlinks=False)
    except FileExistsError as error:
        raise LedgerRecordConflictError(final_path) from error
    _inject_crash(crash_after, LedgerPublishStep.HARD_LINK_PUBLISHED)
    _fsync_directory(durable_directory)
    _inject_crash(crash_after, LedgerPublishStep.LEDGER_DIRECTORY_FSYNCED)
    staging_path.unlink()
    _fsync_directory(staging_directory)
    return final_path


def publish_ledger_record(
    ledger_directory: Path | str,
    plan: CollectionPlan,
    record: AttemptClaim | AttemptTerminal,
    *,
    crash_after: LedgerPublishStep | None = None,
) -> Path:
    """Durably publish one immutable claim or terminal without replacement."""

    if crash_after is not None and not isinstance(crash_after, LedgerPublishStep):
        raise TypeError("crash_after must be a LedgerPublishStep")
    ledger_directory = preflight_attempt_ledger(ledger_directory)
    _validate_bound_collection_plan(ledger_directory.parent, plan)
    final_path = _record_path(ledger_directory, record)
    if os.path.lexists(final_path):
        raise LedgerRecordConflictError(final_path)
    state = validate_attempt_ledger(ledger_directory, plan)
    _validate_record_for_publication(plan, state, record)
    _verify_terminal_artifacts_if_needed(
        record,
        plan,
        ledger_directory.parent,
    )
    return _publish_bytes_noreplace(
        staging_directory=ledger_directory / ".staging",
        final_path=final_path,
        durable_directory=ledger_directory,
        data=canonical_json_bytes(record.to_record()),
        crash_after=crash_after,
    )


def _remove_non_authoritative_staging(ledger_directory: Path) -> None:
    staging_directory = ledger_directory / ".staging"
    for path in staging_directory.iterdir():
        if path.is_dir() and not path.is_symlink():
            raise LedgerValidationError(
                "Attempt Ledger staging contains an unexpected directory"
            )
        path.unlink()
    _fsync_directory(staging_directory)


def recover_attempt_ledger(
    ledger_directory: Path | str,
    plan: CollectionPlan,
    *,
    interruption_timestamp_utc: str | None = None,
) -> AttemptLedgerState:
    """Remove staging and close a durable unclosed claim without rerunning it."""

    validate_collection_plan(plan)
    ledger_directory, _ = _ensure_attempt_ledger_directories(ledger_directory)
    _remove_non_authoritative_staging(ledger_directory)
    preflight_attempt_ledger(ledger_directory)
    state = validate_attempt_ledger(ledger_directory, plan)
    if state.open_claim is None:
        return state
    if interruption_timestamp_utc is None:
        raise LedgerValidationError(
            "recovery needs an interruption timestamp for the open claim"
        )
    failure_evidence_digest = _actual_failure_evidence_digest(
        ledger_directory.parent,
        state.open_claim.attempt_id,
    )
    partial_evidence = _scan_partial_evidence(
        ledger_directory.parent,
        state.open_claim.attempt_id,
    )
    terminal = construct_attempt_terminal(
        plan,
        state.open_claim,
        outcome=TerminalOutcome.INTERRUPTED_FAILURE,
        terminal_timestamp_utc=interruption_timestamp_utc,
        failure_evidence_digest=failure_evidence_digest,
        partial_evidence=partial_evidence,
    )
    publish_ledger_record(ledger_directory, plan, terminal)
    return validate_attempt_ledger(ledger_directory, plan)
