"""Real env/recorder transitions, scripted Actor/parser; no sealed model load."""
from copy import deepcopy
from dataclasses import asdict, fields, replace
from unittest.mock import Mock

import pytest

from scripts.speech_planning import SpeechPlan, PlanningAction as A
from scripts.speech_realization import expected_action, realize_parse_verify, PublicRealizationContext
from scripts.verified_speech_commit import realize_and_commit
from scripts.counterfactual_tom import build_continuation
from tests.tom.test_counterfactual_consumer import CAPACITY
from tests.tom.test_speech_realization import ActorBackend
from tests.canonical_collection.test_final_capacity import _night
from tests.canonical_collection.test_runtime_game_evidence import _runtime, ROLES
from werewolf.agents.gpt_agent import GPTAgent
from werewolf.canonical_collection.public_history import freeze_public_event_history
from werewolf.speech.verified_commit import bind_verified_speech, VerifiedSpeechCommit


def prepare(phase='speech'):
    _, _, env, _, _, recorder, backend = _runtime()
    env.reset(roles=ROLES)
    _night(env, 0)
    if phase == 'speech_pk':
        from tests.tom.test_speech_realization import ParserBackend
        original_backend = env.speech_perceiver.backend
        while env.phase == 'speech':
            env.speech_perceiver.backend = ParserBackend(
                f'player{env.current_act_idx + 1} | no_commitment | NONE')
            env.step(('speech', '我暂时不作明确表态。'))
        votes = (2, 3, 1, 1, 2, 3, 0)
        while env.phase == 'vote':
            env.step(('vote', votes[env.current_act_idx]))
        assert env.phase == 'speech_pk'
        env.speech_perceiver.backend = original_backend
    recorder.start(env, roles=ROLES)
    return env, recorder, backend


def pending(env, recorder):
    return recorder.before_agent_act(env, step_idx=0, acting_player_id=env.current_act_idx + 1,
        delivered_observation=env.get_observation(), speech_kind=env.phase)


def snapshot(env, recorder):
    return deepcopy({
        "events": env.public_events, "logs": [vars(log) for log in env.game_log],
        "speech_queue": env.speech_queue, "vote_queue": env.vote_queue,
        "phase": env.phase, "day": env.day, "speaker": env.current_act_idx,
        "rng": env._rng.getstate(), "annotations": env.speech_annotations,
        "pending": recorder._pending, "prefixes": recorder.prefixes,
        "observations": recorder.observations, "submitted": recorder.submitted_gameplay_actions,
        "raw": recorder._raw_actions,
    })


def response_for(expected):
    return f"{expected.subject} | {expected.action} | {expected.object or 'NONE'}"


def actor_for(expected):
    # Generic natural-language fixture includes the required explicit seat.
    text = f"我对{expected.object or expected.subject}的判断暂时保留。"
    return GPTAgent(backend=ActorBackend(text), model_name="test-actor"), text


def envelope_for(env, payload, response):
    from tests.tom.test_speech_realization import ParserBackend
    # Only tests substitute a backend, still invoking the actual parser.
    env.speech_perceiver.backend = ParserBackend(response)
    speaker = f"player{env.current_act_idx + 1}"
    actor, _ = actor_for(expected_action(payload, speaker))
    result = realize_parse_verify(payload, PublicRealizationContext(speaker, env.day,
        'discussion' if env.phase == 'speech' else 'pk_discussion'),
        actor=actor, perceiver=env.speech_perceiver)
    return bind_verified_speech(speech=result.speech, expected=result.expected,
        perception=result.perception, day=env.day, phase=env.phase,
        public_history_digest=freeze_public_event_history(env.public_events).digest,
        perceiver=env.speech_perceiver)


@pytest.mark.parametrize("action", tuple(A))
def test_six_actions_four_layers_and_single_perception(action):
    env, recorder, backend = prepare()
    # Commit one genuine prior public speech so SUPPORT/OPPOSE are eligible.
    first = f"player{env.current_act_idx + 1}"
    pending(env, recorder)
    backend.perception_response = f"{first} | no_commitment | NONE"
    actor, _ = actor_for(expected_action(SpeechPlan(A.NO_COMMITMENT).public_payload(), first))
    realize_and_commit(SpeechPlan(A.NO_COMMITMENT).public_payload(), env=env, actor=actor, recorder=recorder)
    pending(env, recorder)
    speaker = f"player{env.current_act_idx + 1}"
    plan = SpeechPlan(action, None if action in (A.SELF_DEFEND, A.NO_COMMITMENT) else first)
    expected = expected_action(plan.public_payload(), speaker)
    continuation = build_continuation(
        recorder._pending.prefix, plan.for_speaker(speaker), capacity=CAPACITY)
    semantic = continuation.tokens[-2]
    assert semantic.token_type == "speech_action"
    assert semantic.source == expected.subject
    assert semantic.action == expected.action
    assert semantic.target == expected.object
    backend.perception_response = response_for(expected)
    env.speech_perceiver.parse_with_audit = Mock(wraps=env.speech_perceiver.parse_with_audit)
    actor, text = actor_for(expected)
    rng = env._rng.getstate()
    realize_and_commit(plan.public_payload(), env=env, actor=actor, recorder=recorder)
    assert env.speech_perceiver.parse_with_audit.call_count == 1
    assert env.speech_annotations[-1].actions == (expected,)
    assert env.game_log[-1].content['speech_content'] == text
    assert [e for e in env.public_events if e['event_type'] == 'public_speech'][-1]['raw_text'] == text
    assert recorder._pending is None
    assert recorder.submitted_gameplay_actions[-1].action_payload.to_value() == text
    assert env._rng.getstate() == rng


@pytest.mark.parametrize("phase", ['speech', 'speech_pk'])
@pytest.mark.parametrize("kind", ['wrong_action', 'wrong_target', 'extra', 'parse_failure', 'empty'])
def test_semantic_failure_has_no_gameplay_or_recorder_mutation(kind, phase):
    env, recorder, backend = prepare(phase)
    pending(env, recorder)
    speaker = f"player{env.current_act_idx + 1}"
    targets = [f"player{i+1}" for i in range(7) if i != env.current_act_idx]
    payload = SpeechPlan(A.NO_COMMITMENT if kind == 'empty' else A.CLEAR,
                         None if kind == 'empty' else targets[0]).public_payload()
    expected = expected_action(payload, speaker)
    backend.perception_response = {
        'wrong_action': f'{speaker} | oppose | {targets[0]}',
        'wrong_target': f'{speaker} | point_as_non_werewolf | {targets[1]}',
        'extra': response_for(expected) + f'\n{speaker} | support | {targets[1]}',
        'parse_failure': 'not valid parser output', 'empty': 'NONE',
    }[kind]
    before = snapshot(env, recorder)
    env.speech_perceiver.parse_with_audit = Mock(wraps=env.speech_perceiver.parse_with_audit)
    actor, _ = actor_for(expected)
    with pytest.raises(ValueError):
        realize_and_commit(payload, env=env, actor=actor, recorder=recorder)
    assert snapshot(env, recorder) == before
    assert env.speech_perceiver.parse_with_audit.call_count == 1
    assert len(actor.backend.calls) == 1


@pytest.mark.parametrize("tamper", ['text', 'actions', 'attempt', 'boundary'])
def test_commit_rejects_post_verification_mutation(tamper):
    env, recorder, _ = prepare()
    pending(env, recorder)
    speaker = f"player{env.current_act_idx + 1}"
    envelope = envelope_for(env, SpeechPlan(A.NO_COMMITMENT).public_payload(), f'{speaker} | no_commitment | NONE')
    if tamper == 'text':
        envelope = replace(envelope, speech=envelope.speech + '篡改')
    elif tamper == 'actions':
        envelope.perception.normalized_actions.clear()
    elif tamper == 'attempt':
        envelope.perception.generation_attempts[0]['raw_response'] = 'NONE'
    else:
        envelope = replace(envelope, public_history_digest='wrong')
    before = snapshot(env, recorder)
    env.speech_perceiver.parse_with_audit = Mock(side_effect=AssertionError('second parse'))
    with pytest.raises(ValueError):
        recorder.commit_verified_speech(env, envelope)
    assert snapshot(env, recorder) == before
    env.speech_perceiver.parse_with_audit.assert_not_called()


def test_staged_recorder_failure_leaves_env_unchanged(monkeypatch):
    from werewolf.canonical_collection.runtime import CanonicalGameRecorder
    env, recorder, _ = prepare()
    pending(env, recorder)
    speaker = f"player{env.current_act_idx + 1}"
    envelope = envelope_for(env, SpeechPlan(A.NO_COMMITMENT).public_payload(), f'{speaker} | no_commitment | NONE')
    before = snapshot(env, recorder)
    monkeypatch.setattr(CanonicalGameRecorder, 'after_env_step', Mock(side_effect=ValueError('construction failed')))
    with pytest.raises(ValueError, match='construction failed'):
        recorder.commit_verified_speech(env, envelope)
    assert snapshot(env, recorder) == before


def test_envelope_fields_contain_no_planner_diagnostics():
    assert {f.name for f in fields(VerifiedSpeechCommit)} == {
        'speech', 'expected', 'day', 'phase', 'public_history_digest', 'backend_id',
        'model_id', 'perception', 'snapshot', 'digest'}


@pytest.mark.parametrize("semantic_match", [True, False])
def test_audited_contexts_are_sequential_and_exit_on_semantic_failure(semantic_match):
    from werewolf.canonical_collection.call_audit import audited_backends
    from werewolf.canonical_collection.trajectory_evidence import BackendCallPurpose
    from scripts.speech_realization import SemanticVerificationError

    env, recorder, backend = prepare()
    handoff = pending(env, recorder)
    audit = recorder.call_audit
    speaker = f'player{env.current_act_idx + 1}'
    payload = SpeechPlan(A.NO_COMMITMENT).public_payload()
    actor, _ = actor_for(expected_action(payload, speaker))
    actor.backend = audited_backends({'actor': actor.backend}, audit)['actor']
    backend.perception_response = f'{speaker} | no_commitment | NONE' if semantic_match else 'NONE'
    env.speech_perceiver.parse_with_audit = Mock(wraps=env.speech_perceiver.parse_with_audit)
    before = len(audit.records)
    if semantic_match:
        realize_and_commit(payload, env=env, actor=actor, recorder=recorder)
    else:
        with pytest.raises(SemanticVerificationError):
            realize_and_commit(payload, env=env, actor=actor, recorder=recorder)
    calls = audit.records[before:]
    assert [call.purpose for call in calls] == [
        BackendCallPurpose.SPEECH_GENERATION, BackendCallPurpose.SPEECH_PERCEPTION]
    assert all(call.boundary_id == handoff.boundary_id for call in calls)
    assert env.speech_perceiver.parse_with_audit.call_count == 1
    # Public context API proves cleanup without inspecting or changing _active.
    with audit.action_context(acting_player_id=int(speaker[-1]),
            boundary_id=handoff.boundary_id, is_public_speech=True):
        # The no-nesting contract remains enforced after both outcomes.
        with pytest.raises(RuntimeError, match='backend call contexts cannot be nested'):
            with audit.speech_perception_context(event_id='cleanup-probe',
                    boundary_id=handoff.boundary_id, speaker_id=int(speaker[-1])):
                pytest.fail('nested context was accepted')


@pytest.mark.parametrize("phase", ['speech', 'speech_pk'])
def test_original_and_verified_env_semantics_and_progression_match(phase):
    left, _, _ = prepare(phase)
    right, _, _ = prepare(phase)
    from tests.tom.test_speech_realization import ParserBackend
    # Traverse the complete normal speech phase, including its final speaker.
    while left.phase == phase:
        assert right.current_act_idx == left.current_act_idx
        speaker = f'player{left.current_act_idx + 1}'
        response = f'{speaker} | no_commitment | NONE'
        envelope = envelope_for(right, SpeechPlan(A.NO_COMMITMENT).public_payload(), response)
        left.speech_perceiver.backend = ParserBackend(response)
        left.step((phase, envelope.speech))
        right.speech_perceiver.parse_with_audit = Mock(side_effect=AssertionError('second parse'))
        right.step_verified_speech(envelope)
        # Restore real method for producing the next envelope.
        del right.speech_perceiver.parse_with_audit
        assert left.public_events == right.public_events
        assert left.speech_annotations == right.speech_annotations
        assert [vars(x) for x in left.game_log] == [vars(x) for x in right.game_log]
        assert (left.phase, left.current_act_idx, left.speech_queue, left.vote_queue) == (
            right.phase, right.current_act_idx, right.speech_queue, right.vote_queue)
    assert left.phase == right.phase == ('vote' if phase == 'speech' else 'vote_pk')
