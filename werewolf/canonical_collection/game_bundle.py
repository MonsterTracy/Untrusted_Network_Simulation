"""Immutable Canonical Game Bundle publication and sole read interface."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from werewolf.artifact_io import (
    ArtifactValidationError,
    VerifiedArtifact,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    publish_artifact,
    sha256_bytes,
    verify_artifact,
)
from werewolf.canonical_collection.attempt_ledger import (
    AttemptClaim,
    CollectionPlan,
    validate_attempt_claim,
    validate_collection_plan,
)
from werewolf.canonical_collection.pre import (
    AUTHORITATIVE_PRE_PREFIX_SCHEMA_VERSION,
    AuthoritativePREPrefix,
    SpeakerPREBeliefHandoff,
    construct_authoritative_pre_prefix,
    construct_speaker_pre_belief_handoff,
)
from werewolf.canonical_collection.public_history import (
    PUBLIC_EVENT_SCHEMA_VERSION,
    PublicEventHistory,
    freeze_public_event_history,
)
from werewolf.canonical_collection.speech import (
    V1_ANNOTATION_SCHEMA_VERSION,
    V1AnnotationStatus,
    V1PerceptionAttempt,
    V1SpeechAction,
    V1SpeechAnnotation,
    construct_v1_speech_annotation,
)
from werewolf.canonical_collection.trajectory_evidence import (
    BELIEF_OBSERVATION_SCHEMA_VERSION,
    PRIVATE_REPLAY_EVIDENCE_SCHEMA_VERSION,
    BackendCallEvidence,
    BackendCallPurpose,
    BackendCallStatus,
    BeliefObservation,
    BeliefObservationStatus,
    CallBudgetSummary,
    CanonicalGameEvidence,
    ParserSummary,
    PrivateReplayEvidence,
    SubmittedGameplayAction,
    construct_backend_call_evidence,
    construct_belief_observation,
    construct_call_budget_summary,
    construct_canonical_game_evidence,
    construct_private_replay_evidence,
    construct_submitted_gameplay_action,
    validate_canonical_game_evidence,
)


CANONICAL_GAME_BUNDLE_ARTIFACT_TYPE = "canonical_game_bundle"
CANONICAL_GAME_BUNDLE_SCHEMA_VERSION = "classic7-canonical-game-bundle-v1"
DETERMINISTIC_REPLAY_RESULT_SCHEMA_VERSION = "classic7_replay_result_v1"

_BUNDLE_FILE_PATHS = frozenset(
    {
        "public/public_event_stream.jsonl",
        "public/authoritative_pre_prefixes.jsonl",
        "public/belief_observations.jsonl",
        "public/speech_annotations_v1.jsonl",
        "audit/call_budget_summary.json",
        "audit/parser_summary.json",
        "private/submitted_gameplay_actions.jsonl",
        "private/backend_call_evidence.jsonl",
        "private/private_replay_evidence.json",
    }
)
_BUNDLE_DOMAIN_MANIFEST_FIELDS = frozenset(
    {
        "artifact_type",
        "schema_version",
        "artifact_identity",
        "collection_id",
        "collection_plan_digest",
        "claim_record_digest",
        "attempt_id",
        "ordinal",
        "seed",
        "game_id",
        "source_revision",
        "runtime_identity",
        "runtime_provenance_digest",
        "runtime_configuration_digest",
        "public_event_schema_version",
        "pre_prefix_schema_version",
        "belief_observation_schema_version",
        "v1_annotation_schema_version",
        "private_replay_evidence_schema_version",
        "replay_executor_identity",
        "replay_result_digest",
        "canonical_eligibility",
    }
)
_FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "actual_role",
        "backend_prompt",
        "backend_response",
        "hard_knowledge",
        "private_check",
        "private_prompt",
        "private_response",
        "private_state",
        "role_assignment",
        "role_id",
        "seer_check",
        "teammate_knowledge",
        "witch_information",
        "wolf_teammates",
    }
)
_SAFE_IDENTITY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class DeterministicReplayExecutor:
    """One provenance-identified replay boundary supplied by Game Runtime."""

    identity: str
    execute: Callable[
        [PrivateReplayEvidence, tuple[SubmittedGameplayAction, ...]],
        Sequence[Mapping[str, Any]],
    ]

    def __post_init__(self) -> None:
        if not isinstance(self.identity, str) or not self.identity.strip():
            raise ValueError("replay executor identity must be non-empty text")
        if not callable(self.execute):
            raise TypeError("replay executor must be callable")


@dataclass(frozen=True)
class DeterministicReplayResult:
    schema_version: str
    replay_executor_identity: str
    public_event_digest: str
    replay_result_digest: str

    def to_record(self) -> dict[str, str]:
        return {
            "schema_version": self.schema_version,
            "replay_executor_identity": self.replay_executor_identity,
            "public_event_digest": self.public_event_digest,
            "replay_result_digest": self.replay_result_digest,
        }


@dataclass(frozen=True)
class VerifiedCanonicalGameBundle:
    """A Bundle whose envelope, semantics, provenance, and replay all passed."""

    artifact: VerifiedArtifact
    game_id: str
    public_event_stream: PublicEventHistory
    authoritative_pre_prefixes: tuple[AuthoritativePREPrefix, ...]
    belief_observations: tuple[BeliefObservation, ...]
    speech_annotations_v1: tuple[V1SpeechAnnotation, ...]
    call_budget_summary: CallBudgetSummary
    parser_summary: ParserSummary
    submitted_gameplay_actions: tuple[SubmittedGameplayAction, ...]
    backend_call_evidence: tuple[BackendCallEvidence, ...]
    private_replay_evidence: PrivateReplayEvidence
    replay_result: DeterministicReplayResult

    @property
    def manifest(self) -> dict[str, Any]:
        return self.artifact.manifest

    @property
    def manifest_digest(self) -> str:
        return self.artifact.manifest_digest

    @property
    def path(self) -> Path:
        return self.artifact.path


def _replay_result(
    public_event_digest: str,
    executor_identity: str,
) -> DeterministicReplayResult:
    without_digest = {
        "schema_version": DETERMINISTIC_REPLAY_RESULT_SCHEMA_VERSION,
        "replay_executor_identity": executor_identity,
        "public_event_digest": public_event_digest,
    }
    return DeterministicReplayResult(
        **without_digest,
        replay_result_digest=sha256_bytes(canonical_json_bytes(without_digest)),
    )


def validate_deterministic_replay(
    *,
    expected_public_event_stream: PublicEventHistory,
    private_replay_evidence: PrivateReplayEvidence,
    submitted_gameplay_actions: tuple[SubmittedGameplayAction, ...],
    replay_executor: DeterministicReplayExecutor,
) -> DeterministicReplayResult:
    """Replay private evidence and require exact authoritative public outcomes."""

    try:
        replayed = freeze_public_event_history(
            replay_executor.execute(
                private_replay_evidence,
                submitted_gameplay_actions,
            )
        )
    except (TypeError, ValueError) as error:
        raise ArtifactValidationError(
            f"deterministic replay produced invalid public history: {error}"
        ) from error
    if replayed.to_records() != expected_public_event_stream.to_records():
        raise ArtifactValidationError(
            "deterministic replay diverged from authoritative public trajectory"
        )
    if replayed.digest != private_replay_evidence.expected_public_event_digest:
        raise ArtifactValidationError(
            "deterministic replay diverged from Private Replay Evidence"
        )
    return _replay_result(replayed.digest, replay_executor.identity)


def _bundle_files(evidence: CanonicalGameEvidence) -> dict[str, bytes]:
    return {
        "public/public_event_stream.jsonl": canonical_jsonl_bytes(
            evidence.public_event_stream.to_records()
        ),
        "public/authoritative_pre_prefixes.jsonl": canonical_jsonl_bytes(
            prefix.to_record() for prefix in evidence.authoritative_pre_prefixes
        ),
        "public/belief_observations.jsonl": canonical_jsonl_bytes(
            item.to_record() for item in evidence.belief_observations
        ),
        "public/speech_annotations_v1.jsonl": canonical_jsonl_bytes(
            item.to_record() for item in evidence.speech_annotations_v1
        ),
        "audit/call_budget_summary.json": canonical_json_bytes(
            evidence.call_budget_summary.to_record()
        ),
        "audit/parser_summary.json": canonical_json_bytes(
            evidence.parser_summary.to_record()
        ),
        "private/submitted_gameplay_actions.jsonl": canonical_jsonl_bytes(
            item.to_record() for item in evidence.submitted_gameplay_actions
        ),
        "private/backend_call_evidence.jsonl": canonical_jsonl_bytes(
            item.to_record() for item in evidence.backend_call_evidence
        ),
        "private/private_replay_evidence.json": canonical_json_bytes(
            evidence.private_replay_evidence.to_record()
        ),
    }


def _manifest_fields(
    *,
    plan: CollectionPlan,
    claim: AttemptClaim,
    evidence: CanonicalGameEvidence,
    replay_result: DeterministicReplayResult,
) -> dict[str, Any]:
    return {
        "artifact_type": CANONICAL_GAME_BUNDLE_ARTIFACT_TYPE,
        "schema_version": CANONICAL_GAME_BUNDLE_SCHEMA_VERSION,
        "artifact_identity": evidence.game_id,
        "collection_id": plan.collection_id,
        "collection_plan_digest": plan.plan_digest,
        "claim_record_digest": claim.record_digest,
        "attempt_id": claim.attempt_id,
        "ordinal": claim.ordinal,
        "seed": claim.seed,
        "game_id": evidence.game_id,
        "source_revision": plan.source_revision,
        "runtime_identity": plan.runtime_identity,
        "runtime_provenance_digest": plan.runtime_provenance_digest,
        "runtime_configuration_digest": sha256_bytes(
            canonical_json_bytes(
                evidence.private_replay_evidence.runtime_configuration.to_value()
            )
        ),
        "public_event_schema_version": plan.public_event_schema_version,
        "pre_prefix_schema_version": plan.pre_prefix_schema_version,
        "belief_observation_schema_version": (
            plan.belief_observation_schema_version
        ),
        "v1_annotation_schema_version": plan.v1_annotation_schema_version,
        "private_replay_evidence_schema_version": (
            PRIVATE_REPLAY_EVIDENCE_SCHEMA_VERSION
        ),
        "replay_executor_identity": replay_result.replay_executor_identity,
        "replay_result_digest": replay_result.replay_result_digest,
        "canonical_eligibility": True,
    }


def _validate_parent_contract(
    plan: CollectionPlan,
    claim: AttemptClaim,
    evidence: CanonicalGameEvidence,
) -> None:
    validate_collection_plan(plan)
    validate_attempt_claim(claim, plan)
    if plan.bundle_schema_version != CANONICAL_GAME_BUNDLE_SCHEMA_VERSION:
        raise ValueError("Collection Plan bundle schema version mismatch")
    if plan.public_event_schema_version != PUBLIC_EVENT_SCHEMA_VERSION:
        raise ValueError("Collection Plan public-event schema version mismatch")
    if plan.pre_prefix_schema_version != AUTHORITATIVE_PRE_PREFIX_SCHEMA_VERSION:
        raise ValueError("Collection Plan PRE-prefix schema version mismatch")
    if plan.belief_observation_schema_version != BELIEF_OBSERVATION_SCHEMA_VERSION:
        raise ValueError("Collection Plan belief-observation schema version mismatch")
    if plan.v1_annotation_schema_version != V1_ANNOTATION_SCHEMA_VERSION:
        raise ValueError("Collection Plan V1 annotation schema version mismatch")
    if evidence.private_replay_evidence.seed != claim.seed:
        raise ValueError("Private Replay Evidence seed does not match claim")
    if any(
        observation.attempt_id != claim.attempt_id
        for observation in evidence.belief_observations
    ):
        raise ValueError("Belief Observation attempt identity does not match claim")
    frozen_call_identities = {
        "backend_identity": plan.backend_identity,
        "model_identity": plan.model_identity,
        "parser_identity": plan.parser_identity,
        "prompt_identity": plan.prompt_identity,
        "retry_policy_identity": plan.retry_policy_identity,
        "call_budget_identity": plan.call_budget_identity,
    }
    for call in evidence.backend_call_evidence:
        for field_name, expected_value in frozen_call_identities.items():
            if getattr(call, field_name) != expected_value:
                raise ValueError(
                    f"backend call {field_name} does not match Collection Plan"
                )
    for annotation in evidence.speech_annotations_v1:
        for attempt in annotation.attempts:
            identities = {
                "backend_identity": attempt.backend_id,
                "model_identity": attempt.model_id,
                "parser_identity": attempt.parser_version,
                "prompt_identity": attempt.prompt_version,
            }
            for field_name, actual_value in identities.items():
                if actual_value != frozen_call_identities[field_name]:
                    raise ValueError(
                        f"V1 attempt {field_name} does not match Collection Plan"
                    )


def publish_canonical_game_bundle(
    destination: Path | str,
    *,
    plan: CollectionPlan,
    claim: AttemptClaim,
    evidence: CanonicalGameEvidence,
    replay_executor: DeterministicReplayExecutor,
) -> VerifiedCanonicalGameBundle:
    """Publish one eligible Bundle, then reopen it through the sole validator."""

    if not isinstance(evidence, CanonicalGameEvidence):
        raise TypeError("evidence must be CanonicalGameEvidence")
    validate_canonical_game_evidence(evidence)
    if _SAFE_IDENTITY.fullmatch(evidence.game_id) is None:
        raise ValueError("game_id must be one safe path component")
    if Path(destination).name != evidence.game_id:
        raise ValueError("Bundle destination must end with its game_id")
    _validate_parent_contract(plan, claim, evidence)
    replay_result = validate_deterministic_replay(
        expected_public_event_stream=evidence.public_event_stream,
        private_replay_evidence=evidence.private_replay_evidence,
        submitted_gameplay_actions=evidence.submitted_gameplay_actions,
        replay_executor=replay_executor,
    )
    publish_artifact(
        destination,
        manifest_fields=_manifest_fields(
            plan=plan,
            claim=claim,
            evidence=evidence,
            replay_result=replay_result,
        ),
        files=_bundle_files(evidence),
    )
    return validate_canonical_game_bundle(
        destination,
        plan=plan,
        claim=claim,
        replay_executor=replay_executor,
    )


def _load_json(path: Path) -> Any:
    data = path.read_bytes()
    value = json.loads(data.decode("utf-8"))
    if canonical_json_bytes(value) != data:
        raise ArtifactValidationError(f"non-canonical JSON child: {path.name}")
    return value


def _load_jsonl(path: Path) -> list[Any]:
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        raise ArtifactValidationError(
            f"JSONL child lacks terminal newline: {path.name}"
        )
    records = [json.loads(line.decode("utf-8")) for line in data.splitlines()]
    if canonical_jsonl_bytes(records) != data:
        raise ArtifactValidationError(f"non-canonical JSONL child: {path.name}")
    return records


def _assert_no_private_fields(value: Any, location: str) -> None:
    if isinstance(value, Mapping):
        leaked = _FORBIDDEN_PUBLIC_KEYS.intersection(value)
        if leaked:
            raise ArtifactValidationError(
                f"private field leaked into {location}: {sorted(leaked)[0]}"
            )
        for nested in value.values():
            _assert_no_private_fields(nested, location)
    elif isinstance(value, list):
        for nested in value:
            _assert_no_private_fields(nested, location)


def _raw_public_event(record: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        key: value
        for key, value in record.items()
        if key not in {"day", "phase"}
    }
    if record.get("event_type") == "phase_change":
        result["day"] = record.get("day")
        result["phase"] = record.get("phase")
    return result


def _history_from_records(records: Sequence[Mapping[str, Any]]) -> PublicEventHistory:
    return freeze_public_event_history(
        [_raw_public_event(record) for record in records]
    )


def _annotation_from_record(
    record: Mapping[str, Any],
    full_history: PublicEventHistory,
) -> V1SpeechAnnotation:
    try:
        event_index = record["event_index"]
        history = _history_from_records(
            full_history.to_records()[: event_index + 1]
        )
        annotation = construct_v1_speech_annotation(
            history,
            status=V1AnnotationStatus(record["status"]),
            actions=tuple(V1SpeechAction(*value) for value in record["actions"]),
            attempts=tuple(
                V1PerceptionAttempt(
                    attempt_index=value["attempt_index"],
                    call_id=value["call_id"],
                    backend_id=value["backend_id"],
                    model_id=value["model_id"],
                    prompt_version=value["prompt_version"],
                    parser_version=value["parser_version"],
                    status=V1AnnotationStatus(value["status"]),
                    raw_response=value["raw_response"],
                    error_category=value["error_category"],
                    error_message=value["error_message"],
                )
                for value in record["attempts"]
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ArtifactValidationError(
            f"invalid V1 annotation record: {error}"
        ) from error
    if annotation.to_record() != record:
        raise ArtifactValidationError("V1 annotation record is not canonical")
    return annotation


def _prefix_from_record(record: Mapping[str, Any]) -> AuthoritativePREPrefix:
    try:
        history = _history_from_records(record["public_events"])
        annotations = tuple(
            _annotation_from_record(value, history)
            for value in record["v1_annotations"]
        )
        links = {
            value["observer_id"]: value["observation_id"]
            for value in record["belief_observation_links"]
        }
        prefix = construct_authoritative_pre_prefix(
            game_id=record["game_id"],
            boundary_id=record["boundary_id"],
            step_index=record["step_index"],
            report_trigger_id=record["report_trigger_id"],
            current_speaker=record["current_speaker"],
            alive_observer_ids=tuple(record["alive_observer_ids"]),
            public_event_history=history,
            v1_annotations=annotations,
            belief_observation_ids_by_observer=links,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ArtifactValidationError(f"invalid PRE Prefix record: {error}") from error
    if prefix.to_record() != record:
        raise ArtifactValidationError(
            "Authoritative PRE Prefix record is not canonical"
        )
    return prefix


def _observation_from_record(record: Mapping[str, Any]) -> BeliefObservation:
    try:
        observation = construct_belief_observation(
            game_id=record["game_id"],
            attempt_id=record["attempt_id"],
            boundary_id=record["boundary_id"],
            prefix_digest=record["prefix_digest"],
            observation_id=record["observation_id"],
            observer_id=record["observer_id"],
            observer_alive=record["observer_alive"],
            day=record["day"],
            phase=record["phase"],
            status=BeliefObservationStatus(record["status"]),
            suspicion_support=tuple(record["suspicion_support"]),
            attempts=tuple(record["attempts"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ArtifactValidationError(
            f"invalid Belief Observation record: {error}"
        ) from error
    if observation.to_record() != record:
        raise ArtifactValidationError("Belief Observation record is not canonical")
    return observation


def _call_budget_from_record(record: Mapping[str, Any]) -> CallBudgetSummary:
    try:
        summary = construct_call_budget_summary(
            configured_call_limit=record["configured_call_limit"],
            used_calls=record["used_calls"],
            retry_calls=record["retry_calls"],
            fallback_action_count=record["fallback_action_count"],
            second_speaker_belief_count=record["second_speaker_belief_count"],
            opaque_call_digests=tuple(record["opaque_call_digests"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ArtifactValidationError(
            f"invalid call-budget summary: {error}"
        ) from error
    if summary.to_record() != record:
        raise ArtifactValidationError("call-budget summary is not canonical")
    return summary


def _private_replay_from_record(record: Mapping[str, Any]) -> PrivateReplayEvidence:
    try:
        evidence = construct_private_replay_evidence(
            game_id=record["game_id"],
            seed=record["seed"],
            role_assignment=record["role_assignment"],
            runtime_configuration=record["runtime_configuration"],
            initial_runtime_state=record["initial_runtime_state"],
            replay_inputs=record["replay_inputs"],
            expected_public_event_digest=record["expected_public_event_digest"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ArtifactValidationError(
            f"invalid Private Replay Evidence: {error}"
        ) from error
    if evidence.to_record() != record:
        raise ArtifactValidationError("Private Replay Evidence is not canonical")
    return evidence


def _backend_call_from_record(record: Mapping[str, Any]) -> BackendCallEvidence:
    try:
        evidence = construct_backend_call_evidence(
            call_id=record["call_id"],
            operation_id=record["operation_id"],
            purpose=BackendCallPurpose(record["purpose"]),
            status=BackendCallStatus(record["status"]),
            boundary_id=record["boundary_id"],
            observer_id=record["observer_id"],
            attempt_index=record["attempt_index"],
            backend_identity=record["backend_identity"],
            model_identity=record["model_identity"],
            parser_identity=record["parser_identity"],
            prompt_identity=record["prompt_identity"],
            retry_policy_identity=record["retry_policy_identity"],
            call_budget_identity=record["call_budget_identity"],
            private_payload=record["private_payload"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ArtifactValidationError(
            f"invalid backend call evidence: {error}"
        ) from error
    if evidence.to_record() != record:
        raise ArtifactValidationError("backend call evidence is not canonical")
    return evidence


def _handoff_from_record(
    record: Mapping[str, Any] | None,
    prefixes_by_boundary: Mapping[str, AuthoritativePREPrefix],
    observations_by_id: Mapping[str, BeliefObservation],
) -> SpeakerPREBeliefHandoff | None:
    if record is None:
        return None
    try:
        prefix = prefixes_by_boundary[record["boundary_id"]]
        observation = observations_by_id[record["observation_id"]]
        handoff = construct_speaker_pre_belief_handoff(
            prefix,
            observation_id=observation.observation_id,
            observation_digest=observation.observation_digest,
            observer_id=observation.observer_id,
            observation_status=observation.status.value,
            suspicion_support=observation.suspicion_support,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ArtifactValidationError(
            f"invalid Speaker PRE handoff: {error}"
        ) from error
    if handoff.to_record() != record:
        raise ArtifactValidationError("Speaker PRE Belief Handoff is not canonical")
    return handoff


def _action_from_record(
    record: Mapping[str, Any],
    prefixes_by_boundary: Mapping[str, AuthoritativePREPrefix],
    observations_by_id: Mapping[str, BeliefObservation],
) -> SubmittedGameplayAction:
    try:
        action = construct_submitted_gameplay_action(
            action_id=record["action_id"],
            step_index=record["step_index"],
            actor_id=record["actor_id"],
            action_type=record["action_type"],
            action_payload=record["action_payload"],
            resulting_public_event_ids=tuple(record["resulting_public_event_ids"]),
            boundary_id=record["boundary_id"],
            speaker_pre_belief_handoff=_handoff_from_record(
                record["speaker_pre_belief_handoff"],
                prefixes_by_boundary,
                observations_by_id,
            ),
            fallback_used=record["fallback_used"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ArtifactValidationError(
            f"invalid submitted gameplay action: {error}"
        ) from error
    if action.to_record() != record:
        raise ArtifactValidationError("submitted gameplay action is not canonical")
    return action


def _validate_manifest(
    manifest: Mapping[str, Any],
    *,
    plan: CollectionPlan,
    claim: AttemptClaim,
    replay_executor: DeterministicReplayExecutor,
) -> None:
    domain_fields = set(manifest) - {"file_table", "manifest_digest"}
    if domain_fields != _BUNDLE_DOMAIN_MANIFEST_FIELDS:
        raise ArtifactValidationError("Bundle manifest has an invalid field set")
    expected = {
        "artifact_type": CANONICAL_GAME_BUNDLE_ARTIFACT_TYPE,
        "schema_version": CANONICAL_GAME_BUNDLE_SCHEMA_VERSION,
        "collection_id": plan.collection_id,
        "collection_plan_digest": plan.plan_digest,
        "claim_record_digest": claim.record_digest,
        "attempt_id": claim.attempt_id,
        "ordinal": claim.ordinal,
        "seed": claim.seed,
        "source_revision": plan.source_revision,
        "runtime_identity": plan.runtime_identity,
        "runtime_provenance_digest": plan.runtime_provenance_digest,
        "public_event_schema_version": PUBLIC_EVENT_SCHEMA_VERSION,
        "pre_prefix_schema_version": AUTHORITATIVE_PRE_PREFIX_SCHEMA_VERSION,
        "belief_observation_schema_version": BELIEF_OBSERVATION_SCHEMA_VERSION,
        "v1_annotation_schema_version": V1_ANNOTATION_SCHEMA_VERSION,
        "private_replay_evidence_schema_version": (
            PRIVATE_REPLAY_EVIDENCE_SCHEMA_VERSION
        ),
        "replay_executor_identity": replay_executor.identity,
        "canonical_eligibility": True,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ArtifactValidationError(f"Bundle manifest {field} mismatch")
    if manifest.get("artifact_identity") != manifest.get("game_id"):
        raise ArtifactValidationError("Bundle artifact/game identity mismatch")
    if set(manifest["file_table"]) != _BUNDLE_FILE_PATHS:
        raise ArtifactValidationError("Bundle child file partition is incomplete")


def validate_canonical_game_bundle(
    path: Path | str,
    *,
    plan: CollectionPlan,
    claim: AttemptClaim,
    replay_executor: DeterministicReplayExecutor,
) -> VerifiedCanonicalGameBundle:
    """The sole interface that opens a Canonical Game Bundle."""

    validate_collection_plan(plan)
    validate_attempt_claim(claim, plan)
    artifact = verify_artifact(
        path,
        expected_artifact_type=CANONICAL_GAME_BUNDLE_ARTIFACT_TYPE,
        expected_schema_version=CANONICAL_GAME_BUNDLE_SCHEMA_VERSION,
    )
    if artifact.path.name != artifact.manifest.get("game_id"):
        raise ArtifactValidationError("Bundle path does not match game identity")
    _validate_manifest(
        artifact.manifest,
        plan=plan,
        claim=claim,
        replay_executor=replay_executor,
    )
    root = artifact.path
    public_records = {
        relative_path: _load_jsonl(root / relative_path)
        for relative_path in _BUNDLE_FILE_PATHS
        if relative_path.startswith("public/")
    }
    audit_records = {
        relative_path: _load_json(root / relative_path)
        for relative_path in _BUNDLE_FILE_PATHS
        if relative_path.startswith("audit/")
    }
    for relative_path, records in {**public_records, **audit_records}.items():
        _assert_no_private_fields(records, relative_path)

    try:
        public_event_stream = _history_from_records(
            public_records["public/public_event_stream.jsonl"]
        )
        annotations = tuple(
            _annotation_from_record(item, public_event_stream)
            for item in public_records["public/speech_annotations_v1.jsonl"]
        )
        prefixes = tuple(
            _prefix_from_record(item)
            for item in public_records["public/authoritative_pre_prefixes.jsonl"]
        )
        observations = tuple(
            _observation_from_record(item)
            for item in public_records["public/belief_observations.jsonl"]
        )
        call_budget = _call_budget_from_record(
            audit_records["audit/call_budget_summary.json"]
        )
        private_replay = _private_replay_from_record(
            _load_json(root / "private/private_replay_evidence.json")
        )
        backend_records = tuple(
            _backend_call_from_record(item)
            for item in _load_jsonl(
                root / "private/backend_call_evidence.jsonl"
            )
        )
        prefixes_by_boundary = {item.boundary_id: item for item in prefixes}
        observations_by_id = {item.observation_id: item for item in observations}
        actions = tuple(
            _action_from_record(item, prefixes_by_boundary, observations_by_id)
            for item in _load_jsonl(
                root / "private/submitted_gameplay_actions.jsonl"
            )
        )
        evidence = construct_canonical_game_evidence(
            game_id=artifact.manifest["game_id"],
            public_event_stream=public_event_stream,
            authoritative_pre_prefixes=prefixes,
            belief_observations=observations,
            speech_annotations_v1=annotations,
            call_budget_summary=call_budget,
            submitted_gameplay_actions=actions,
            backend_call_evidence=backend_records,
            private_replay_evidence=private_replay,
        )
    except ArtifactValidationError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise ArtifactValidationError(
            f"invalid Canonical Game Bundle: {error}"
        ) from error

    parser_record = audit_records["audit/parser_summary.json"]
    if evidence.parser_summary.to_record() != parser_record:
        raise ArtifactValidationError("parser summary disagrees with V1 annotations")
    runtime_configuration_digest = sha256_bytes(
        private_replay.runtime_configuration.canonical_bytes
    )
    if (
        artifact.manifest["runtime_configuration_digest"]
        != runtime_configuration_digest
    ):
        raise ArtifactValidationError("Bundle runtime configuration digest mismatch")
    _validate_parent_contract(plan, claim, evidence)
    replay_result = validate_deterministic_replay(
        expected_public_event_stream=public_event_stream,
        private_replay_evidence=private_replay,
        submitted_gameplay_actions=actions,
        replay_executor=replay_executor,
    )
    if artifact.manifest["replay_result_digest"] != replay_result.replay_result_digest:
        raise ArtifactValidationError("Bundle replay result digest mismatch")
    return VerifiedCanonicalGameBundle(
        artifact=artifact,
        game_id=evidence.game_id,
        public_event_stream=public_event_stream,
        authoritative_pre_prefixes=prefixes,
        belief_observations=observations,
        speech_annotations_v1=annotations,
        call_budget_summary=call_budget,
        parser_summary=evidence.parser_summary,
        submitted_gameplay_actions=actions,
        backend_call_evidence=evidence.backend_call_evidence,
        private_replay_evidence=private_replay,
        replay_result=replay_result,
    )
