from dataclasses import replace
import json

import pytest

from werewolf.artifact_io import (
    ArtifactConflictError,
    ArtifactValidationError,
    canonical_json_bytes,
    publish_artifact,
    sha256_bytes,
)
from werewolf.canonical_collection import (
    BELIEF_OBSERVATION_SCHEMA_VERSION,
    CANONICAL_GAME_BUNDLE_SCHEMA_VERSION,
    V1_ANNOTATION_SCHEMA_VERSION,
    V1_SPEECH_PARSER_VERSION,
    V1_SPEECH_PROMPT_VERSION,
    BeliefObservationStatus,
    BackendCallPurpose,
    BackendCallStatus,
    CanonicalEvidenceConflictError,
    CanonicalEvidenceValidationError,
    CanonicalFailureStage,
    DeterministicReplayExecutor,
    TerminalOutcome,
    V1AnnotationStatus,
    V1PerceptionAttempt,
    V1SpeechAction,
    construct_attempt_claim,
    construct_authoritative_pre_prefix,
    construct_belief_observation,
    construct_backend_call_evidence,
    construct_call_budget_summary,
    construct_canonical_failure_evidence,
    construct_canonical_failure_attempt,
    construct_canonical_game_evidence,
    construct_canonical_partial_evidence,
    construct_attempt_terminal,
    construct_collection_plan,
    construct_private_replay_evidence,
    construct_speaker_pre_belief_handoff,
    construct_submitted_gameplay_action,
    construct_v1_speech_annotation,
    freeze_public_event_history,
    initialize_attempt_ledger,
    publish_canonical_failure_evidence,
    publish_canonical_game_bundle,
    publish_canonical_partial_evidence,
    publish_ledger_record,
    summarize_canonical_failures,
    validate_canonical_failure_evidence,
    validate_canonical_game_bundle,
    validate_canonical_partial_evidence,
)


def _plan(**overrides):
    arguments = {
        "collection_id": "collection-commit4",
        "ordered_seed_pool": (101, 202),
        "target_canonical_success_count": 1,
        "runtime_identity": "classic7-runtime-v1",
        "agent_identity": "canonical-playing-agent-v1",
        "backend_identity": "deterministic-fixture-backend-v1",
        "model_identity": "fixture-model-v1",
        "parser_identity": V1_SPEECH_PARSER_VERSION,
        "prompt_identity": V1_SPEECH_PROMPT_VERSION,
        "retry_policy_identity": "bounded-retries-v1",
        "call_budget_identity": "fixture-call-budget-v1",
        "public_event_schema_version": "classic7_public_event_history_v1",
        "pre_prefix_schema_version": "classic7_authoritative_pre_prefix_v1",
        "belief_observation_schema_version": BELIEF_OBSERVATION_SCHEMA_VERSION,
        "v1_annotation_schema_version": V1_ANNOTATION_SCHEMA_VERSION,
        "bundle_schema_version": CANONICAL_GAME_BUNDLE_SCHEMA_VERSION,
        "source_revision": "ee0f7081dadbab4492ced62b93abcb40a04b9989",
        "environment_provenance": {
            "implementation": "phase1-commit4",
            "python": "3.12",
        },
    }
    arguments.update(overrides)
    return construct_collection_plan(**arguments)


def _claim(plan=None):
    plan = _plan() if plan is None else plan
    return construct_attempt_claim(
        plan,
        ordinal=0,
        attempt_id="attempt-000",
        claim_timestamp_utc="2026-09-03T01:02:03Z",
    )


def _public_events():
    return [
        {
            "event_id": "event-0",
            "event_index": 0,
            "event_type": "phase_change",
            "day": 0,
            "phase": "night",
        },
        {
            "event_id": "event-1",
            "event_index": 1,
            "event_type": "death_announcement",
            "dead_players": [],
        },
        {
            "event_id": "event-2",
            "event_index": 2,
            "event_type": "phase_change",
            "day": 1,
            "phase": "discussion",
        },
        {
            "event_id": "event-3",
            "event_index": 3,
            "event_type": "turn_start",
            "speaker": "player1",
        },
        {
            "event_id": "event-4",
            "event_index": 4,
            "event_type": "public_speech",
            "speaker": "player1",
            "raw_text": "player3 is suspicious",
        },
    ]


def _call_provenance(plan):
    return {
        "backend_identity": plan.backend_identity,
        "model_identity": plan.model_identity,
        "parser_identity": plan.parser_identity,
        "prompt_identity": plan.prompt_identity,
        "retry_policy_identity": plan.retry_policy_identity,
        "call_budget_identity": plan.call_budget_identity,
    }


def _annotation(full_history, plan):
    return construct_v1_speech_annotation(
        full_history,
        status=V1AnnotationStatus.OK,
        actions=(
            V1SpeechAction("player1", "point_as_werewolf", "player3"),
        ),
        attempts=(
            V1PerceptionAttempt(
                attempt_index=1,
                call_id="perception-call-1",
                backend_id=plan.backend_identity,
                model_id=plan.model_identity,
                prompt_version=plan.prompt_identity,
                parser_version=plan.parser_identity,
                status=V1AnnotationStatus.OK,
                raw_response="player1,point_as_werewolf,player3",
                error_category=None,
                error_message=None,
            ),
        ),
    )


def _fixture(plan=None, claim=None):
    plan = _plan() if plan is None else plan
    claim = _claim(plan) if claim is None else claim
    full_history = freeze_public_event_history(_public_events())
    alive = tuple(f"player{seat}" for seat in range(1, 8))
    observation_ids = {
        observer: f"observation-{observer}" for observer in alive
    }
    prefix = construct_authoritative_pre_prefix(
        game_id="game-000",
        boundary_id="boundary-003",
        step_index=3,
        report_trigger_id="pre-public-speech-003",
        current_speaker="player1",
        alive_observer_ids=alive,
        public_event_history=freeze_public_event_history(_public_events()[:4]),
        v1_annotations=(),
        belief_observation_ids_by_observer=observation_ids,
    )
    observations = tuple(
        construct_belief_observation(
            game_id="game-000",
            attempt_id=claim.attempt_id,
            boundary_id=prefix.boundary_id,
            prefix_digest=prefix.prefix_digest,
            observation_id=observation_ids[observer],
            observer_id=observer,
            observer_alive=True,
            day=1,
            phase="discussion",
            status=BeliefObservationStatus.SUCCESS,
            suspicion_support=(
                "player3" if observer != "player3" else "player4",
            ),
            attempts=(
                {
                    "attempt_index": 1,
                    "call_id": f"belief-call-{observer}",
                    "status": "success",
                    "response_digest": f"{int(observer[-1]):064x}",
                    "error_category": None,
                    "error_message": None,
                },
            ),
        )
        for observer in alive
    )
    speaker_observation = observations[0]
    handoff = construct_speaker_pre_belief_handoff(
        prefix,
        observation_id=speaker_observation.observation_id,
        observation_digest=speaker_observation.observation_digest,
        observer_id="player1",
        observation_status="success",
        suspicion_support=speaker_observation.suspicion_support,
    )
    submitted_action = construct_submitted_gameplay_action(
        action_id="action-001",
        step_index=3,
        actor_id="player1",
        action_type="public_speech",
        action_payload="player3 is suspicious",
        resulting_public_event_ids=("event-4",),
        boundary_id=prefix.boundary_id,
        speaker_pre_belief_handoff=handoff,
        fallback_used=False,
    )
    private_replay = construct_private_replay_evidence(
        game_id="game-000",
        seed=claim.seed,
        role_assignment={
            "player1": "Werewolf",
            "player2": "Werewolf",
            "player3": "Villager",
            "player4": "Villager",
            "player5": "Villager",
            "player6": "Seer",
            "player7": "Witch",
        },
        runtime_configuration={"night0": True, "player_count": 7},
        initial_runtime_state={"night_index": 0},
        replay_inputs={"fixture": "deterministic-v1"},
        expected_public_event_digest=full_history.digest,
    )
    backend_call_evidence = (
        *(
            construct_backend_call_evidence(
                call_id=f"belief-call-player{seat}",
                operation_id=f"observation-player{seat}",
                purpose=BackendCallPurpose.BELIEF_OBSERVATION,
                status=BackendCallStatus.SUCCESS,
                boundary_id=prefix.boundary_id,
                observer_id=f"player{seat}",
                attempt_index=1,
                private_payload={
                    "private_prompt": "private role-aware cognition",
                    "private_response": "private response",
                },
                **_call_provenance(plan),
            )
            for seat in range(1, 8)
        ),
        construct_backend_call_evidence(
            call_id="perception-call-1",
            operation_id="event-4",
            purpose=BackendCallPurpose.SPEECH_PERCEPTION,
            status=BackendCallStatus.SUCCESS,
            boundary_id=prefix.boundary_id,
            observer_id="player1",
            attempt_index=1,
            private_payload={
                "private_prompt": "public-speech semantic perception",
                "private_response": "parser response",
            },
            **_call_provenance(plan),
        ),
    )
    evidence = construct_canonical_game_evidence(
        game_id="game-000",
        public_event_stream=full_history,
        authoritative_pre_prefixes=(prefix,),
        belief_observations=observations,
        speech_annotations_v1=(_annotation(full_history, plan),),
        call_budget_summary=construct_call_budget_summary(
            configured_call_limit=32,
            used_calls=8,
            retry_calls=0,
            fallback_action_count=0,
            second_speaker_belief_count=0,
            opaque_call_digests=tuple(
                record.call_digest
                for record in backend_call_evidence
            ),
        ),
        submitted_gameplay_actions=(submitted_action,),
        backend_call_evidence=backend_call_evidence,
        private_replay_evidence=private_replay,
    )
    executor = DeterministicReplayExecutor(
        identity="deterministic-fixture-replay-v1",
        execute=lambda _private, _actions: _public_events(),
    )
    return plan, claim, evidence, executor


def test_complete_evidence_publishes_one_verified_three_partition_bundle(
    tmp_path,
):
    plan, claim, evidence, executor = _fixture()
    destination = tmp_path / "games" / evidence.game_id

    published = publish_canonical_game_bundle(
        destination,
        plan=plan,
        claim=claim,
        evidence=evidence,
        replay_executor=executor,
    )
    verified = validate_canonical_game_bundle(
        destination,
        plan=plan,
        claim=claim,
        replay_executor=executor,
    )

    assert published.manifest_digest == verified.manifest_digest
    assert verified.game_id == "game-000"
    assert verified.manifest["canonical_eligibility"] is True
    assert set(verified.manifest["file_table"]) == {
        "audit/call_budget_summary.json",
        "audit/parser_summary.json",
        "private/backend_call_evidence.jsonl",
        "private/private_replay_evidence.json",
        "private/submitted_gameplay_actions.jsonl",
        "public/authoritative_pre_prefixes.jsonl",
        "public/belief_observations.jsonl",
        "public/public_event_stream.jsonl",
        "public/speech_annotations_v1.jsonl",
    }
    assert verified.public_event_stream.digest == (
        evidence.public_event_stream.digest
    )
    assert verified.private_replay_evidence.role_assignment[0] == (
        "player1",
        "Werewolf",
    )
    assert b"role_assignment" not in (
        destination / "public" / "belief_observations.jsonl"
    ).read_bytes()
    assert b"private_prompt" not in (
        destination / "audit" / "call_budget_summary.json"
    ).read_bytes()


@pytest.mark.parametrize(
    "stage",
    [
        CanonicalFailureStage.BELIEF_OBSERVATION,
        CanonicalFailureStage.SPEECH_PERCEPTION,
        CanonicalFailureStage.GAMEPLAY_ACTION,
    ],
)
def test_failure_and_partial_evidence_are_formal_round_trip_artifacts(
    tmp_path,
    stage,
):
    plan = _plan()
    claim = _claim(plan)
    partial = construct_canonical_partial_evidence(
        plan=plan,
        claim=claim,
        evidence_id=f"partial-{stage.value}",
        evidence_type="bounded_attempt",
        payload={"call_id": "failed-call-1", "status": "error"},
    )
    published_partial = publish_canonical_partial_evidence(
        tmp_path,
        plan=plan,
        claim=claim,
        evidence=partial,
    )
    evidence = construct_canonical_failure_evidence(
        plan=plan,
        claim=claim,
        game_id="game-000",
        stage=stage,
        error_category="bounded_attempts_exhausted",
        boundary_id="boundary-003",
        observer_id="player1",
        day=1,
        phase="discussion",
        retry_exhausted=True,
        attempt_evidence=(
            construct_canonical_failure_attempt(
                attempt_index=1,
                call_id="failed-call-1",
                **_call_provenance(plan),
                error_category="bounded_attempts_exhausted",
                error_message="bounded attempt failed",
            ),
        ),
        partial_evidence=(published_partial,),
    )

    published_failure = publish_canonical_failure_evidence(
        tmp_path,
        plan=plan,
        claim=claim,
        evidence=evidence,
    )
    verified_partial = validate_canonical_partial_evidence(
        published_partial.path,
        plan=plan,
        claim=claim,
    )
    verified_failure = validate_canonical_failure_evidence(
        published_failure.path,
        plan=plan,
        claim=claim,
    )
    summary = summarize_canonical_failures((verified_failure.evidence,))

    assert verified_partial.evidence == partial
    assert verified_partial.ledger_reference == published_partial.ledger_reference
    assert verified_failure.evidence == evidence
    assert verified_failure.file_sha256 == sha256_bytes(
        published_failure.path.read_bytes()
    )
    assert summary.total_failure_count == 1
    assert summary.failure_count_by_stage == ((stage.value, 1),)

    with pytest.raises(CanonicalEvidenceConflictError):
        publish_canonical_failure_evidence(
            tmp_path,
            plan=plan,
            claim=claim,
            evidence=evidence,
        )


def test_bundle_replay_divergence_fails_closed(tmp_path):
    plan, claim, evidence, executor = _fixture()
    destination = tmp_path / "games" / evidence.game_id
    publish_canonical_game_bundle(
        destination,
        plan=plan,
        claim=claim,
        evidence=evidence,
        replay_executor=executor,
    )
    divergent_events = _public_events()
    divergent_events[-1] = {**divergent_events[-1], "raw_text": "changed"}
    divergent = DeterministicReplayExecutor(
        identity=executor.identity,
        execute=lambda _private, _actions: divergent_events,
    )

    with pytest.raises(ArtifactValidationError, match="replay diverged"):
        validate_canonical_game_bundle(
            destination,
            plan=plan,
            claim=claim,
            replay_executor=divergent,
        )


def test_bundle_rejects_child_corruption_and_never_repairs_it(tmp_path):
    plan, claim, evidence, executor = _fixture()
    destination = tmp_path / "games" / evidence.game_id
    publish_canonical_game_bundle(
        destination,
        plan=plan,
        claim=claim,
        evidence=evidence,
        replay_executor=executor,
    )
    child = destination / "public" / "public_event_stream.jsonl"
    child.write_bytes(child.read_bytes() + b"corruption")

    with pytest.raises(ArtifactValidationError, match="file size mismatch"):
        validate_canonical_game_bundle(
            destination,
            plan=plan,
            claim=claim,
            replay_executor=executor,
        )


def test_bundle_atomic_publication_never_replaces_an_existing_identity(tmp_path):
    plan, claim, evidence, executor = _fixture()
    destination = tmp_path / "games" / evidence.game_id
    original = publish_canonical_game_bundle(
        destination,
        plan=plan,
        claim=claim,
        evidence=evidence,
        replay_executor=executor,
    )
    changed_executor = DeterministicReplayExecutor(
        identity="different-replay-identity",
        execute=executor.execute,
    )

    with pytest.raises(ArtifactConflictError):
        publish_canonical_game_bundle(
            destination,
            plan=plan,
            claim=claim,
            evidence=evidence,
            replay_executor=changed_executor,
        )
    assert validate_canonical_game_bundle(
        destination,
        plan=plan,
        claim=claim,
        replay_executor=executor,
    ).manifest_digest == original.manifest_digest


def _republish_with_mutation(source, destination, *, manifest_updates=None, files=None):
    manifest = json.loads((source / "manifest.json").read_text())
    manifest_fields = {
        key: value
        for key, value in manifest.items()
        if key not in {"file_table", "manifest_digest"}
    }
    manifest_fields.update(manifest_updates or {})
    payloads = {
        relative_path: (source / relative_path).read_bytes()
        for relative_path in manifest["file_table"]
    }
    payloads.update(files or {})
    return publish_artifact(
        destination,
        manifest_fields=manifest_fields,
        files=payloads,
    )


def test_bundle_rejects_private_field_in_public_or_audit_partition(tmp_path):
    plan, claim, evidence, executor = _fixture()
    source = tmp_path / "games" / evidence.game_id
    publish_canonical_game_bundle(
        source,
        plan=plan,
        claim=claim,
        evidence=evidence,
        replay_executor=executor,
    )
    leaked = canonical_json_bytes(
        {
            **evidence.call_budget_summary.to_record(),
            "role_assignment": {"player1": "Werewolf"},
        }
    )
    destination = tmp_path / "leaked" / "games" / evidence.game_id
    _republish_with_mutation(
        source,
        destination,
        files={"audit/call_budget_summary.json": leaked},
    )

    with pytest.raises(ArtifactValidationError, match="private field leaked"):
        validate_canonical_game_bundle(
            destination,
            plan=plan,
            claim=claim,
            replay_executor=executor,
        )


@pytest.mark.parametrize(
    ("manifest_updates", "match"),
    [
        ({"collection_plan_digest": "0" * 64}, "collection_plan_digest"),
        ({"schema_version": "unsupported-bundle-schema"}, "schema_version"),
    ],
)
def test_bundle_rejects_parent_provenance_or_schema_drift(
    tmp_path,
    manifest_updates,
    match,
):
    plan, claim, evidence, executor = _fixture()
    source = tmp_path / "games" / evidence.game_id
    publish_canonical_game_bundle(
        source,
        plan=plan,
        claim=claim,
        evidence=evidence,
        replay_executor=executor,
    )
    destination = tmp_path / "mutated" / "games" / evidence.game_id
    _republish_with_mutation(
        source,
        destination,
        manifest_updates=manifest_updates,
    )

    with pytest.raises(ArtifactValidationError, match=match):
        validate_canonical_game_bundle(
            destination,
            plan=plan,
            claim=claim,
            replay_executor=executor,
        )


def test_ineligible_collection_evidence_cannot_be_published_as_a_bundle():
    _plan_value, _claim_value, evidence, _executor = _fixture()
    failed_observation = replace(
        evidence.belief_observations[1],
        status=BeliefObservationStatus.ERROR,
        label_observed=False,
        suspicion_support=(),
    )

    with pytest.raises(ValueError, match="failed Belief Observation"):
        construct_canonical_game_evidence(
            game_id=evidence.game_id,
            public_event_stream=evidence.public_event_stream,
            authoritative_pre_prefixes=evidence.authoritative_pre_prefixes,
            belief_observations=(
                evidence.belief_observations[0],
                failed_observation,
                *evidence.belief_observations[2:],
            ),
            speech_annotations_v1=evidence.speech_annotations_v1,
            call_budget_summary=evidence.call_budget_summary,
            submitted_gameplay_actions=evidence.submitted_gameplay_actions,
            backend_call_evidence=evidence.backend_call_evidence,
            private_replay_evidence=evidence.private_replay_evidence,
        )


def test_every_public_speech_requires_one_pre_prefix_and_observation_set():
    plan, _claim_value, evidence, _executor = _fixture()
    extended_events = [
        *_public_events(),
        {
            "event_id": "event-5",
            "event_index": 5,
            "event_type": "turn_start",
            "speaker": "player2",
        },
        {
            "event_id": "event-6",
            "event_index": 6,
            "event_type": "public_speech",
            "speaker": "player2",
            "raw_text": "I suspect player4",
        },
    ]
    extended_history = freeze_public_event_history(extended_events)
    second_annotation = construct_v1_speech_annotation(
        extended_history,
        status=V1AnnotationStatus.NO_ACTION,
        actions=(),
        attempts=(
            V1PerceptionAttempt(
                attempt_index=1,
                call_id="perception-call-2",
                backend_id=plan.backend_identity,
                model_id=plan.model_identity,
                prompt_version=plan.prompt_identity,
                parser_version=plan.parser_identity,
                status=V1AnnotationStatus.NO_ACTION,
                raw_response="NONE",
                error_category=None,
                error_message=None,
            ),
        ),
    )

    with pytest.raises(ValueError, match="exactly match every public speech"):
        construct_canonical_game_evidence(
            game_id=evidence.game_id,
            public_event_stream=extended_history,
            authoritative_pre_prefixes=evidence.authoritative_pre_prefixes,
            belief_observations=evidence.belief_observations,
            speech_annotations_v1=(
                evidence.speech_annotations_v1[0],
                second_annotation,
            ),
            call_budget_summary=evidence.call_budget_summary,
            submitted_gameplay_actions=evidence.submitted_gameplay_actions,
            backend_call_evidence=evidence.backend_call_evidence,
            private_replay_evidence=replace(
                evidence.private_replay_evidence,
                expected_public_event_digest=extended_history.digest,
            ),
        )


def test_unlinked_second_speaker_belief_call_makes_bundle_ineligible():
    _plan_value, _claim_value, evidence, _executor = _fixture()
    prefix = evidence.authoritative_pre_prefixes[0]
    extra = construct_backend_call_evidence(
        call_id="second-speaker-belief",
        operation_id="second-speaker-belief-operation",
        purpose=BackendCallPurpose.BELIEF_OBSERVATION,
        status=BackendCallStatus.SUCCESS,
        boundary_id=prefix.boundary_id,
        observer_id=prefix.current_speaker,
        attempt_index=1,
        private_payload={"private_prompt": "second estimate"},
        **_call_provenance(_plan_value),
    )
    calls = (*evidence.backend_call_evidence, extra)
    summary = construct_call_budget_summary(
        configured_call_limit=32,
        used_calls=len(calls),
        retry_calls=0,
        fallback_action_count=0,
        second_speaker_belief_count=0,
        opaque_call_digests=tuple(item.call_digest for item in calls),
    )

    with pytest.raises(ValueError, match="exactly equal collected observations"):
        construct_canonical_game_evidence(
            game_id=evidence.game_id,
            public_event_stream=evidence.public_event_stream,
            authoritative_pre_prefixes=evidence.authoritative_pre_prefixes,
            belief_observations=evidence.belief_observations,
            speech_annotations_v1=evidence.speech_annotations_v1,
            call_budget_summary=summary,
            submitted_gameplay_actions=evidence.submitted_gameplay_actions,
            backend_call_evidence=calls,
            private_replay_evidence=evidence.private_replay_evidence,
        )


def test_bundle_rejects_backend_call_provenance_outside_collection_plan(tmp_path):
    plan, claim, evidence, executor = _fixture()
    original = evidence.backend_call_evidence[0]
    mismatched = construct_backend_call_evidence(
        call_id=original.call_id,
        operation_id=original.operation_id,
        purpose=original.purpose,
        status=original.status,
        boundary_id=original.boundary_id,
        observer_id=original.observer_id,
        attempt_index=original.attempt_index,
        private_payload=original.private_payload,
        backend_identity="unplanned-backend",
        model_identity=plan.model_identity,
        parser_identity=plan.parser_identity,
        prompt_identity=plan.prompt_identity,
        retry_policy_identity=plan.retry_policy_identity,
        call_budget_identity=plan.call_budget_identity,
    )
    calls = (mismatched, *evidence.backend_call_evidence[1:])
    candidate = construct_canonical_game_evidence(
        game_id=evidence.game_id,
        public_event_stream=evidence.public_event_stream,
        authoritative_pre_prefixes=evidence.authoritative_pre_prefixes,
        belief_observations=evidence.belief_observations,
        speech_annotations_v1=evidence.speech_annotations_v1,
        call_budget_summary=construct_call_budget_summary(
            configured_call_limit=32,
            used_calls=len(calls),
            retry_calls=0,
            fallback_action_count=0,
            second_speaker_belief_count=0,
            opaque_call_digests=tuple(item.call_digest for item in calls),
        ),
        submitted_gameplay_actions=evidence.submitted_gameplay_actions,
        backend_call_evidence=calls,
        private_replay_evidence=evidence.private_replay_evidence,
    )
    destination = tmp_path / "games" / evidence.game_id

    with pytest.raises(ValueError, match="backend_identity"):
        publish_canonical_game_bundle(
            destination,
            plan=plan,
            claim=claim,
            evidence=candidate,
            replay_executor=executor,
        )
    assert not destination.exists()


def test_call_budget_retry_count_covers_every_backend_operation():
    plan, _claim_value, evidence, _executor = _fixture()
    retry_calls = tuple(
        construct_backend_call_evidence(
            call_id=f"gameplay-call-{attempt_index}",
            operation_id="gameplay-operation-1",
            purpose=BackendCallPurpose.GAMEPLAY_ACTION,
            status=(
                BackendCallStatus.ERROR
                if attempt_index == 1
                else BackendCallStatus.SUCCESS
            ),
            boundary_id="boundary-003",
            observer_id="player1",
            attempt_index=attempt_index,
            private_payload={"attempt": attempt_index},
            **_call_provenance(plan),
        )
        for attempt_index in (1, 2)
    )
    calls = (*evidence.backend_call_evidence, *retry_calls)

    with pytest.raises(ValueError, match="retry count"):
        construct_canonical_game_evidence(
            game_id=evidence.game_id,
            public_event_stream=evidence.public_event_stream,
            authoritative_pre_prefixes=evidence.authoritative_pre_prefixes,
            belief_observations=evidence.belief_observations,
            speech_annotations_v1=evidence.speech_annotations_v1,
            call_budget_summary=construct_call_budget_summary(
                configured_call_limit=32,
                used_calls=len(calls),
                retry_calls=0,
                fallback_action_count=0,
                second_speaker_belief_count=0,
                opaque_call_digests=tuple(item.call_digest for item in calls),
            ),
            submitted_gameplay_actions=evidence.submitted_gameplay_actions,
            backend_call_evidence=calls,
            private_replay_evidence=evidence.private_replay_evidence,
        )


def test_invalid_frozen_evidence_cannot_poison_final_no_replace_paths(tmp_path):
    plan, claim, evidence, executor = _fixture()
    destination = tmp_path / "games" / evidence.game_id
    forged_observation = replace(
        evidence.belief_observations[0],
        observation_digest="0" * 64,
    )
    forged_bundle = replace(
        evidence,
        belief_observations=(
            forged_observation,
            *evidence.belief_observations[1:],
        ),
    )

    with pytest.raises(ValueError, match="stale nested digest"):
        publish_canonical_game_bundle(
            destination,
            plan=plan,
            claim=claim,
            evidence=forged_bundle,
            replay_executor=executor,
        )
    assert not destination.exists()

    mutable_payload = {"nested": {"attempts": [1]}}
    partial = construct_canonical_partial_evidence(
        plan=plan,
        claim=claim,
        evidence_id="immutable-partial",
        evidence_type="bounded_attempt",
        payload=mutable_payload,
    )
    mutable_payload["nested"]["attempts"].append(2)
    assert partial.to_record()["payload"] == {"nested": {"attempts": [1]}}

    forged_partial = replace(partial, record_digest="0" * 64)
    with pytest.raises(
        CanonicalEvidenceValidationError,
        match="stale nested digest",
    ):
        publish_canonical_partial_evidence(
            tmp_path,
            plan=plan,
            claim=claim,
            evidence=forged_partial,
        )
    assert not (
        tmp_path
        / "attempts"
        / claim.attempt_id
        / "partial_evidence"
        / "immutable-partial.json"
    ).exists()

    published_partial = publish_canonical_partial_evidence(
        tmp_path,
        plan=plan,
        claim=claim,
        evidence=partial,
    )
    failure = construct_canonical_failure_evidence(
        plan=plan,
        claim=claim,
        game_id=evidence.game_id,
        stage=CanonicalFailureStage.BELIEF_OBSERVATION,
        error_category="backend_timeout",
        boundary_id="boundary-003",
        observer_id="player1",
        day=1,
        phase="discussion",
        retry_exhausted=True,
        attempt_evidence=(
            construct_canonical_failure_attempt(
                attempt_index=1,
                call_id="failed-call-1",
                **_call_provenance(plan),
                error_category="backend_timeout",
                error_message="bounded attempt failed",
            ),
        ),
        partial_evidence=(published_partial,),
    )
    forged_failure = replace(failure, failure_digest="0" * 64)
    with pytest.raises(
        CanonicalEvidenceValidationError,
        match="stale nested digest",
    ):
        publish_canonical_failure_evidence(
            tmp_path,
            plan=plan,
            claim=claim,
            evidence=forged_failure,
        )
    assert not (
        tmp_path / "attempts" / claim.attempt_id / "failure_evidence.json"
    ).exists()

    error_annotation = construct_v1_speech_annotation(
        evidence.public_event_stream,
        status=V1AnnotationStatus.ERROR,
        actions=(),
        attempts=(
            V1PerceptionAttempt(
                attempt_index=1,
                call_id="perception-call-1",
                backend_id="fixture-backend",
                model_id="fixture-model",
                prompt_version=V1_SPEECH_PROMPT_VERSION,
                parser_version=V1_SPEECH_PARSER_VERSION,
                status=V1AnnotationStatus.ERROR,
                raw_response=None,
                error_category="backend_timeout",
                error_message="bounded attempts exhausted",
            ),
        ),
    )
    with pytest.raises(ValueError, match="V1 perception failure"):
        construct_canonical_game_evidence(
            game_id=evidence.game_id,
            public_event_stream=evidence.public_event_stream,
            authoritative_pre_prefixes=evidence.authoritative_pre_prefixes,
            belief_observations=evidence.belief_observations,
            speech_annotations_v1=(error_annotation,),
            call_budget_summary=evidence.call_budget_summary,
            submitted_gameplay_actions=evidence.submitted_gameplay_actions,
            backend_call_evidence=evidence.backend_call_evidence,
            private_replay_evidence=evidence.private_replay_evidence,
        )

    with pytest.raises(ValueError, match="fallback gameplay action"):
        construct_canonical_game_evidence(
            game_id=evidence.game_id,
            public_event_stream=evidence.public_event_stream,
            authoritative_pre_prefixes=evidence.authoritative_pre_prefixes,
            belief_observations=evidence.belief_observations,
            speech_annotations_v1=evidence.speech_annotations_v1,
            call_budget_summary=evidence.call_budget_summary,
            submitted_gameplay_actions=(
                replace(evidence.submitted_gameplay_actions[0], fallback_used=True),
            ),
            backend_call_evidence=evidence.backend_call_evidence,
            private_replay_evidence=evidence.private_replay_evidence,
        )


def test_belief_observation_rejects_self_suspicion():
    plan = _plan()
    claim = _claim(plan)

    with pytest.raises(ValueError, match="self-suspicion"):
        construct_belief_observation(
            game_id="game-000",
            attempt_id=claim.attempt_id,
            boundary_id="boundary-003",
            prefix_digest="a" * 64,
            observation_id="observation-player1",
            observer_id="player1",
            observer_alive=True,
            day=1,
            phase="discussion",
            status=BeliefObservationStatus.SUCCESS,
            suspicion_support=("player1",),
            attempts=(
                {
                    "attempt_index": 1,
                    "call_id": "belief-call-player1",
                    "status": "success",
                    "response_digest": "b" * 64,
                    "error_category": None,
                    "error_message": None,
                },
            ),
        )


def test_observer_or_perception_failure_requires_exhausted_retry_evidence():
    plan = _plan()
    claim = _claim(plan)

    with pytest.raises(ValueError, match="must exhaust retries"):
        construct_canonical_failure_evidence(
            plan=plan,
            claim=claim,
            game_id="game-000",
            stage=CanonicalFailureStage.BELIEF_OBSERVATION,
            error_category="backend_timeout",
            boundary_id="boundary-003",
            observer_id="player1",
            day=1,
            phase="discussion",
            retry_exhausted=False,
            attempt_evidence=(
                construct_canonical_failure_attempt(
                    attempt_index=1,
                    call_id="failed-call-1",
                    **_call_provenance(plan),
                    error_category="backend_timeout",
                    error_message="bounded attempt failed",
                ),
            ),
            partial_evidence=(),
        )


def test_failure_attempt_provenance_must_equal_collection_plan():
    plan = _plan()
    claim = _claim(plan)
    mismatched = construct_canonical_failure_attempt(
        attempt_index=1,
        call_id="failed-call-1",
        backend_identity="unplanned-backend",
        model_identity=plan.model_identity,
        parser_identity=plan.parser_identity,
        prompt_identity=plan.prompt_identity,
        retry_policy_identity=plan.retry_policy_identity,
        call_budget_identity=plan.call_budget_identity,
        error_category="backend_timeout",
        error_message="bounded attempt failed",
    )

    with pytest.raises(ValueError, match="backend_identity"):
        construct_canonical_failure_evidence(
            plan=plan,
            claim=claim,
            game_id="game-000",
            stage=CanonicalFailureStage.BELIEF_OBSERVATION,
            error_category="backend_timeout",
            boundary_id="boundary-003",
            observer_id="player1",
            day=1,
            phase="discussion",
            retry_exhausted=True,
            attempt_evidence=(mismatched,),
            partial_evidence=(),
        )


def test_failure_attempts_must_form_one_contiguous_exhausted_sequence():
    plan = _plan()
    claim = _claim(plan)

    attempts = tuple(
        construct_canonical_failure_attempt(
            attempt_index=attempt_index,
            call_id=f"failed-call-{attempt_index}",
            **_call_provenance(plan),
            error_category="backend_timeout",
            error_message="bounded attempt failed",
        )
        for attempt_index in (1, 3)
    )

    with pytest.raises(ValueError, match="contiguous from one"):
        construct_canonical_failure_evidence(
            plan=plan,
            claim=claim,
            game_id="game-000",
            stage=CanonicalFailureStage.BELIEF_OBSERVATION,
            error_category="backend_timeout",
            boundary_id="boundary-003",
            observer_id="player1",
            day=1,
            phase="discussion",
            retry_exhausted=True,
            attempt_evidence=attempts,
            partial_evidence=(),
        )


def test_formal_failure_artifacts_bind_directly_to_attempt_terminal(tmp_path):
    plan = _plan()
    claim = _claim(plan)
    ledger_directory = initialize_attempt_ledger(tmp_path, plan)
    publish_ledger_record(ledger_directory, plan, claim)
    partial = publish_canonical_partial_evidence(
        tmp_path,
        plan=plan,
        claim=claim,
        evidence=construct_canonical_partial_evidence(
            plan=plan,
            claim=claim,
            evidence_id="belief-attempt-1",
            evidence_type="belief_attempt",
            payload={"status": "error"},
        ),
    )
    failure = publish_canonical_failure_evidence(
        tmp_path,
        plan=plan,
        claim=claim,
        evidence=construct_canonical_failure_evidence(
            plan=plan,
            claim=claim,
            game_id="game-000",
            stage=CanonicalFailureStage.BELIEF_OBSERVATION,
            error_category="backend_timeout",
            boundary_id="boundary-003",
            observer_id="player1",
            day=1,
            phase="discussion",
            retry_exhausted=True,
            attempt_evidence=(
                construct_canonical_failure_attempt(
                    attempt_index=1,
                    call_id="failed-call-1",
                    **_call_provenance(plan),
                    error_category="backend_timeout",
                    error_message="bounded attempt failed",
                ),
            ),
            partial_evidence=(partial,),
        ),
    )
    terminal = construct_attempt_terminal(
        plan,
        claim,
        outcome=TerminalOutcome.CANONICAL_FAILURE,
        terminal_timestamp_utc="2026-09-03T01:03:00Z",
        failure_evidence_digest=failure.file_sha256,
        partial_evidence=(partial.ledger_reference,),
    )

    publish_ledger_record(ledger_directory, plan, terminal)
