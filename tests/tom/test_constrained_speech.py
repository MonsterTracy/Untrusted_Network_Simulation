"""Phase 4A runner and public treatment boundary; no model service."""
from dataclasses import fields
from copy import deepcopy
import inspect
import re
import subprocess
import sys
from unittest.mock import Mock

import pytest

from run_random import eval as run_game
from scripts import constrained_speech as c
from scripts.speech_planning import SpeechPlan, PlanningAction as A
from tests.tom.test_counterfactual_consumer import PublicWitness, P
from tests.tom.test_verified_speech_commit import prepare, pending, snapshot, actor_for
from tests.canonical_collection.test_runtime_game_evidence import _runtime, _Agent, ROLES
from werewolf.agents.gpt_agent import GPTAgent


def no_commitment(opportunity):
    return next(p for p in opportunity.candidates if p.action == A.NO_COMMITMENT)


@pytest.mark.parametrize('phase', ['discussion', 'pk_discussion'])
def test_public_gate_is_shared_role_independent_and_last_speaker_explicit(phase):
    w = PublicWitness(phase)
    parent = w.pre()
    assert tuple(inspect.signature(c.public_eligibility).parameters) == ('parent',)
    assert c.public_eligibility(parent) == c.public_eligibility(parent)
    eligible = c.public_eligibility(parent)
    assert eligible.eligible
    assert {f.name for f in fields(eligible.opportunity)} == {'parent', 'speaker', 'candidates'}
    for unused_provider in (no_commitment, Mock(side_effect=RuntimeError('model failure'))):
        assert c.public_eligibility(parent) == eligible
    with pytest.raises(TypeError):
        c.public_eligibility(parent, role_truth=ROLES)
    w.speech(P[0], 'no_commitment', None, '我暂时不作明确表态。')
    w.event('turn_start', speaker=P[1])
    last = c.public_eligibility(w.pre())
    assert not last.eligible and last.reason == 'LAST_SAME_PHASE_SPEAKER'


@pytest.mark.parametrize('phase', ['speech', 'speech_pk'])
@pytest.mark.parametrize('outcome', ['success', 'semantic_failure', 'provider_failure', 'invalid_plan'])
def test_real_pending_opportunity_commit_or_fail_closed(phase, outcome):
    env, recorder, backend = prepare(phase)
    pending(env, recorder)
    speaker = f'player{env.current_act_idx + 1}'
    from scripts.speech_realization import expected_action
    actor, _ = actor_for(expected_action(SpeechPlan(A.NO_COMMITMENT).public_payload(), speaker))
    backend.perception_response = 'NONE' if outcome == 'semantic_failure' else f'{speaker} | no_commitment | NONE'
    provider = Mock(side_effect=RuntimeError('provider failed')) if outcome == 'provider_failure' else Mock(
        return_value=object() if outcome == 'invalid_plan' else SpeechPlan(A.NO_COMMITMENT))
    env.step = Mock(side_effect=AssertionError('must not call default step'))
    env.speech_perceiver.parse_with_audit = Mock(wraps=env.speech_perceiver.parse_with_audit)
    before = snapshot(env, recorder)
    trace = []
    if outcome == 'success':
        result = c.handle_speech(env=env, actor=actor, recorder=recorder, provider=provider,
                                 assigned=True, trace=trace)
        assert result is not None and recorder._pending is None
        assert len(env.speech_annotations) == len(before['annotations']) + 1
        assert trace[-1].status == 'VERIFIED_COMMIT_SUCCEEDED'
    else:
        with pytest.raises(c.ConstrainedExecutionFailure):
            c.handle_speech(env=env, actor=actor, recorder=recorder, provider=provider,
                            assigned=True, trace=trace)
        assert snapshot(env, recorder) == before
        assert trace[-1].status == 'CONSTRAINED_EXECUTION_FAILED'
    provider.assert_called_once()
    env.step.assert_not_called()
    assert env.speech_perceiver.parse_with_audit.call_count == (0 if outcome in ('provider_failure', 'invalid_plan') else 1)
    assert trace[-1].public_eligible is True
    assert all('planning_mode' not in event and 'selected_plan' not in event for event in env.public_events)


def runner_fixture():
    _, _, env, fixtures, audit, recorder, backend = _runtime()
    raw_chat = backend.chat
    def chat(*, messages, **kwargs):
        prompt = messages[-1]['content']
        if '当前发言者：' in prompt:
            speaker = re.search(r'当前发言者：(player[1-7])', prompt).group(1)
            return f'{speaker} | no_commitment | NONE'
        return raw_chat(messages=messages, **kwargs)
    backend.chat = chat
    backend.chat_with_metadata = lambda **kwargs: ('我暂时不作明确表态。', {'finish_reason': 'stop'})
    calls = []
    class RunnerActor(GPTAgent):
        reset = _Agent.reset
        report_suspected_werewolves_readonly = _Agent.report_suspected_werewolves_readonly
        def act(self, observation):
            calls.append(('non_speech', observation['phase']))
            return _Agent.act(self, observation)
        def act_with_pre_speech_belief(self, observation, *, pre_speech_belief):
            calls.append(('original_speech', observation['phase']))
            return _Agent.act_with_pre_speech_belief(self, observation, pre_speech_belief=pre_speech_belief)
    agents = []
    for original in fixtures:
        agent = RunnerActor(backend=original.backend, model_name=original.model_name)
        agent.backend_id, agent.seat = original.backend_id, original.seat
        agents.append(agent)
    recorder.belief_collector.agents = tuple(agents)
    env.step = Mock(wraps=env.step)
    env.speech_perceiver.parse_with_audit = Mock(wraps=env.speech_perceiver.parse_with_audit)
    return env, agents, audit, recorder, calls


def test_runner_eligible_consumed_once_last_speaker_untreated(monkeypatch):
    env, agents, audit, recorder, calls = runner_fixture()
    provider, trace = Mock(side_effect=no_commitment), []
    commit = Mock(wraps=c.realize_and_commit)
    monkeypatch.setattr(c, 'realize_and_commit', commit)
    # Controlled fixture terminates on the first ORIGINAL speech, so eligible
    # commits must continue until the final speaker takes the declared untreated path.
    run_game(env, agents, ROLES, canonical_recorder=recorder, call_audit=audit,
        planning_mode='constrained', plan_provider=provider,
        speech_treatment=c.SpeechTreatment(players=P, roles=()), ablation_trace=trace)
    assert trace[-1].status == 'INELIGIBLE_FOR_CONSTRAINED_TREATMENT'
    assert trace[-1].public_eligible is False
    assert all(t.status == 'VERIFIED_COMMIT_SUCCEEDED' for t in trace[:-1])
    assert provider.call_count == commit.call_count == len(trace) - 1
    assert sum(kind == 'original_speech' for kind, _ in calls) == 1
    assert sum(call.args[0][0] in ('speech', 'speech_pk') for call in env.step.call_args_list) == 1
    assert env.speech_perceiver.parse_with_audit.call_count == len(trace)
    assert any(kind == 'non_speech' for kind, _ in calls)
    assert len([e for e in env.public_events if e['event_type'] == 'public_speech']) == len(trace)


@pytest.mark.parametrize('mode', ['original', 'constrained_unassigned'])
def test_runner_default_and_non_treated_use_original(mode):
    env, agents, audit, recorder, calls = runner_fixture()
    provider, trace = Mock(side_effect=AssertionError('provider forbidden')), []
    options = {} if mode == 'original' else dict(planning_mode='constrained',
        speech_treatment=c.SpeechTreatment(roles=()), plan_provider=provider, ablation_trace=trace)
    run_game(env, agents, ROLES, canonical_recorder=recorder, call_audit=audit, **options)
    provider.assert_not_called()
    assert sum(kind == 'original_speech' for kind, _ in calls) == 1
    assert env.speech_perceiver.parse_with_audit.call_count == 1
    if mode == 'original':
        assert trace == []
    else:
        assert trace[0].status == 'UNTREATED_NOT_ASSIGNED'


@pytest.mark.parametrize('failure', ['provider', 'semantic'])
def test_runner_execution_failure_never_falls_back(failure):
    env, agents, audit, recorder, calls = runner_fixture()
    if failure == 'semantic':
        # Actual parser returns no action: a semantic failure for NO_COMMITMENT.
        raw = env.speech_perceiver.backend._backend
        original_chat = raw.chat
        raw.chat = lambda **kwargs: 'NONE' if '当前发言者：' in kwargs['messages'][-1]['content'] else original_chat(**kwargs)
    provider = Mock(side_effect=RuntimeError('provider failed')) if failure == 'provider' else Mock(side_effect=no_commitment)
    trace = []
    from werewolf.canonical_collection.collector import CanonicalAttemptFailure
    with pytest.raises(CanonicalAttemptFailure):
        run_game(env, agents, ROLES, canonical_recorder=recorder, call_audit=audit,
            planning_mode='constrained', plan_provider=provider,
            speech_treatment=c.SpeechTreatment(players=P, roles=()), ablation_trace=trace)
    provider.assert_called_once()
    assert not any(kind == 'original_speech' for kind, _ in calls)
    assert not any(call.args[0][0] in ('speech', 'speech_pk') for call in env.step.call_args_list)
    assert env.speech_perceiver.parse_with_audit.call_count == (failure == 'semantic')
    assert trace[-1].status == 'CONSTRAINED_EXECUTION_FAILED'
    assert trace[-1].public_eligible is True


def test_private_assignment_and_payload_are_separate():
    assignment = c.SpeechTreatment()
    assert assignment.assigned('player1', 'Werewolf')
    assert not assignment.assigned('player1', 'Villager')
    plan = no_commitment(c.public_eligibility(PublicWitness().pre()).opportunity)
    assert set(plan.public_payload()) == {'action', 'target'}


def test_gameplay_imports_do_not_load_sealed_predictor():
    subprocess.run([sys.executable, '-c',
        "import sys; import run_random; import scripts.constrained_speech; "
        "assert 'werewolf.tom.final_evaluation' not in sys.modules"], check=True)


def test_original_default_matches_explicit_original_canonical_behavior(monkeypatch):
    monkeypatch.setattr(c, 'public_eligibility', Mock(side_effect=AssertionError('original must not gate')))
    results = []
    for options in ({}, {'planning_mode': 'original'}):
        env, agents, audit, recorder, calls = runner_fixture()
        run_game(env, agents, ROLES, canonical_recorder=recorder, call_audit=audit, **options)
        results.append((deepcopy(env.public_events), tuple(env.speech_annotations),
                        deepcopy(recorder.submitted_gameplay_actions), env._rng.getstate(), calls))
    assert results[0] == results[1]


def test_vote_turn_bypasses_provider_after_speech_phase():
    from werewolf.envs.werewolf_text_env_v0 import WerewolfTextEnvV0
    env, agents, audit, recorder, calls = runner_fixture()
    # End this controlled test after one real vote instead of after one speech.
    def through_vote(action):
        observation, reward, done, info = WerewolfTextEnvV0.step(env, action)
        if action[0] == 'vote':
            return observation, reward, True, {'Werewolf': -1}
        return observation, reward, done, info
    env.step = Mock(side_effect=through_vote)
    provider, trace = Mock(side_effect=no_commitment), []
    run_game(env, agents, ROLES, canonical_recorder=recorder, call_audit=audit,
        planning_mode='constrained', plan_provider=provider,
        speech_treatment=c.SpeechTreatment(players=P, roles=()), ablation_trace=trace)
    assert any(kind == 'non_speech' and 'vote' in phase for kind, phase in calls)
    assert provider.call_count == len(trace) - 1
    assert all(t.phase == 'discussion' for t in trace)
