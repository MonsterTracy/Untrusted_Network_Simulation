"""Small, explicitly synthetic trajectories through real canonical artifact I/O."""

from dataclasses import replace

import pytest

from tests.canonical_collection.test_game_bundle import _fixture, _plan
from werewolf.canonical_collection import (
    BackendCallPurpose, BeliefObservationStatus, DeterministicReplayExecutor,
    TerminalOutcome, V1SpeechAction, V1AnnotationStatus,
    construct_attempt_claim, construct_attempt_terminal,
    construct_authoritative_pre_prefix, construct_backend_call_evidence,
    construct_belief_observation, construct_call_budget_summary,
    construct_canonical_game_evidence, construct_private_replay_evidence,
    construct_speaker_pre_belief_handoff, construct_submitted_gameplay_action,
    construct_v1_speech_annotation, freeze_public_event_history,
    initialize_attempt_ledger, publish_canonical_game_bundle, publish_ledger_record,
)
from werewolf.canonical_collection.public_history import PLAYER_IDS
from werewolf.development_publication import open_verified_collection, publish_development


def speech_events(*, phase="discussion", dead=(), actor="player1"):
    events = []

    def add(kind, **fields):
        index = len(events)
        events.append({"event_id": f"event-{index}", "event_index": index,
                       "event_type": kind, **fields})

    add("phase_change", day=0, phase="night")
    add("death_announcement", dead_players=list(dead))
    add("phase_change", day=1, phase="discussion")
    if phase == "pk_discussion":
        alive = [p for p in PLAYER_IDS if p not in dead]
        half = len(alive) // 2
        targets = [alive[0]] * half + [alive[1]] * half + [None] * (len(alive) % 2)
        add("phase_change", day=1, phase="vote")
        add("vote_result", votes=[{"voter": p, "target": t} for p, t in zip(alive, targets)])
        add("exile_result", exiled_players=[])
        add("phase_change", day=1, phase="pk_discussion")
    add("turn_start", speaker=actor)
    add("public_speech", speaker=actor, raw_text="controlled fixture speech")
    return events


def prefix_for(*, phase="discussion", dead=(), actor="player1", game_id="controlled-game"):
    events = speech_events(phase=phase, dead=dead, actor=actor)
    alive = tuple(p for p in PLAYER_IDS if p not in dead)
    return construct_authoritative_pre_prefix(
        game_id=game_id, boundary_id="controlled-pre", step_index=len(events) - 2,
        report_trigger_id="controlled-trigger", current_speaker=actor,
        alive_observer_ids=alive, public_event_history=freeze_public_event_history(events[:-1]),
        v1_annotations=(), belief_observation_ids_by_observer={p: f"report-{p}" for p in alive},
    )


def observation_for(context, support=(), *, status=BeliefObservationStatus.SUCCESS):
    success = status is BeliefObservationStatus.SUCCESS
    return construct_belief_observation(
        game_id=context.game_id, attempt_id="controlled-attempt",
        boundary_id=context.boundary_id, prefix_digest=context.prefix_digest,
        observation_id=f"report-{context.observer_id}", observer_id=context.observer_id,
        observer_alive=True, day=context.day, phase=context.phase, status=status,
        suspicion_support=support,
        attempts=({"attempt_index": 1, "call_id": "controlled-call", "status": status.value,
                   "response_digest": "1" * 64 if success else None,
                   "error_category": None if success else "fixture_failure",
                   "error_message": None if success else "controlled failed report"},),
    )


def controlled_evidence(plan, claim, game_id, *, phase, dead, actor, supports, wolves):
    _, _, original, _ = _fixture(plan, claim, game_id=game_id)
    events = speech_events(phase=phase, dead=dead, actor=actor)
    history = freeze_public_event_history(events)
    prefix = prefix_for(phase=phase, dead=dead, actor=actor, game_id=game_id)
    observations = tuple(construct_belief_observation(
        game_id=game_id, attempt_id=claim.attempt_id, boundary_id=prefix.boundary_id,
        prefix_digest=prefix.prefix_digest, observation_id=f"report-{p}", observer_id=p,
        observer_alive=True, day=1, phase=phase, status=BeliefObservationStatus.SUCCESS,
        suspicion_support=supports.get(p, ()), attempts=({
            "attempt_index": 1, "call_id": f"belief-call-{p}", "status": "success",
            "response_digest": "1" * 64, "error_category": None, "error_message": None,
        },),
    ) for p in prefix.alive_observer_ids)
    speaker = next(o for o in observations if o.observer_id == actor)
    handoff = construct_speaker_pre_belief_handoff(
        prefix, observation_id=speaker.observation_id, observation_digest=speaker.observation_digest,
        observer_id=actor, observation_status="success", suspicion_support=speaker.suspicion_support,
    )
    action = construct_submitted_gameplay_action(
        action_id="controlled-action", step_index=prefix.step_index, actor_id=actor,
        action_type="public_speech", action_payload=events[-1]["raw_text"],
        resulting_public_event_ids=(events[-1]["event_id"],), boundary_id=prefix.boundary_id,
        speaker_pre_belief_handoff=handoff, fallback_used=False,
    )
    calls = []
    for old in original.backend_call_evidence:
        if old.purpose is BackendCallPurpose.BELIEF_OBSERVATION and old.observer_id in dead:
            continue
        record = old.to_record()
        for key in ("schema_version", "call_digest"):
            record.pop(key)
        record.update(purpose=old.purpose, status=old.status, boundary_id=prefix.boundary_id)
        if old.purpose is BackendCallPurpose.BELIEF_OBSERVATION:
            record["operation_id"] = f"report-{old.observer_id}"
        else:
            record.update(operation_id=events[-1]["event_id"], observer_id=actor)
        calls.append(construct_backend_call_evidence(**record))
    target = next(p for p in PLAYER_IDS if p != actor)
    annotation = construct_v1_speech_annotation(
        history, status=V1AnnotationStatus.OK,
        actions=(V1SpeechAction(actor, "point_as_werewolf", target),),
        attempts=(replace(original.speech_annotations_v1[0].attempts[0],
                          raw_response=f"{actor},point_as_werewolf,{target}"),),
    )
    old_private = original.private_replay_evidence.to_record()
    for key in ("schema_version", "evidence_digest"):
        old_private.pop(key)
    old_private["expected_public_event_digest"] = history.digest
    nonwolf_roles = iter(("Villager", "Villager", "Villager", "Seer", "Witch"))
    old_private["role_assignment"] = {
        p: "Werewolf" if p in wolves else next(nonwolf_roles) for p in PLAYER_IDS}
    evidence = construct_canonical_game_evidence(
        game_id=game_id, public_event_stream=history, authoritative_pre_prefixes=(prefix,),
        belief_observations=observations, speech_annotations_v1=(annotation,),
        call_budget_summary=construct_call_budget_summary(
            configured_call_limit=32, used_calls=len(calls), retry_calls=0,
            fallback_action_count=0, second_speaker_belief_count=0,
            opaque_call_digests=tuple(c.call_digest for c in calls)),
        submitted_gameplay_actions=(action,), backend_call_evidence=tuple(calls),
        private_replay_evidence=construct_private_replay_evidence(**old_private),
    )
    replay = DeterministicReplayExecutor("observer-controlled-replay-v1", lambda _p, _a: events)
    return evidence, replay


@pytest.fixture
def publication_factory(tmp_path):
    def create(*, name="source", phase="discussion", dead=(), actor="player1", supports=None,
               wolves=("player1", "player2")):
        root = tmp_path / name
        plan = _plan(collection_id="observer-controlled", ordered_seed_pool=tuple(range(100, 105)),
                     target_canonical_success_count=5)
        ledger = initialize_attempt_ledger(root / "collection", plan)
        for ordinal in range(5):
            claim = construct_attempt_claim(plan, ordinal=ordinal, attempt_id=f"attempt-{ordinal}",
                                             claim_timestamp_utc=f"2026-09-04T00:00:0{ordinal}Z")
            publish_ledger_record(ledger, plan, claim)
            game_id = f"game-{ordinal:03d}"
            evidence, replay = controlled_evidence(plan, claim, game_id,
                phase=phase, dead=dead, actor=actor, supports=supports or {}, wolves=wolves)
            bundle = publish_canonical_game_bundle(root / "collection/games" / game_id,
                plan=plan, claim=claim, evidence=evidence, replay_executor=replay)
            publish_ledger_record(ledger, plan, construct_attempt_terminal(
                plan, claim, outcome=TerminalOutcome.CANONICAL_SUCCESS,
                canonical_game_bundle_id=game_id, canonical_game_bundle_digest=bundle.manifest_digest,
                terminal_timestamp_utc=f"2026-09-04T00:01:0{ordinal}Z"))
        collection = open_verified_collection(root / "collection", replay_executor=replay)
        return publish_development(collection, root / "observer-controlled", publication_id="observer-controlled")
    return create
