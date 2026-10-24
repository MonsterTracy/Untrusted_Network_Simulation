"""Synthetic source + real PRE/Actor/parser/verified env commit, no model service."""
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from scripts.speech_planning import PlanningAction as A
from tests.observer_dtrs.conftest import prefix_for
from tests.observer_dtrs.test_contextual_policy import fitted
from tests.tom.test_counterfactual_consumer import PublicWitness, P
from tests.tom.test_speech_realization import ActorBackend
from tests.tom.test_verified_speech_commit import pending, snapshot
from tests.canonical_collection.test_runtime_game_evidence import _runtime, ROLES
from tests.canonical_collection.test_final_capacity import _night
from werewolf.agents.gpt_agent import GPTAgent
from werewolf.artifact_io import canonical_json_bytes
from werewolf.observer_dtrs.d3_protocol import load_protocol
from werewolf.observer_dtrs.decision import Region, TiePolicy
from werewolf.observer_dtrs.estimation import FittedModel
from werewolf.observer_dtrs.probabilities import ProbabilityMatrix
from werewolf.observer_dtrs.speech_runtime import (
    Comparison, ResearchPolicy, SealedQwen3Source, SpeechStrategy, SyntheticPrediction,
    _legal_team, execute_pending_speech, select_observer, strategy_plan,
)


REFERENCE = SpeechStrategy(A.SELF_DEFEND, A.NO_COMMITMENT, A.NO_COMMITMENT, A.NO_COMMITMENT)
EVIDENCE = SpeechStrategy(A.SELF_DEFEND, A.OPPOSE, A.OPPOSE, A.NO_COMMITMENT)
POLICY = ResearchPolicy(load_protocol().loss(22, TiePolicy.TERMINAL_NEG_FIRST), REFERENCE, EVIDENCE)


@pytest.fixture
def estimator():
    return fitted()


def witness(*, target=P[0], phase='discussion'):
    w = PublicWitness(phase)
    # Replace ONLY the final synthetic public speech before the current turn.
    speaker = P[6] if phase == 'discussion' else P[2]
    w.events = w.events[:-2]
    w.annotations.pop()
    w.speech(speaker, 'point_as_werewolf', target, f'我认为{target}是狼人。')
    w.event('turn_start', speaker=P[0])
    return w


def source(pre, r=.4):
    rows = []
    for observer in P:
        if observer in P[:2]:
            rows.append(tuple(0 if p == observer else 1/6 for p in P))
        else:
            values = [r/2, r/2, 0., 0., 0., 0., 0.]
            values[P.index(next(p for p in P[2:] if p != observer))] = 1-r
            rows.append(tuple(values))
    return SyntheticPrediction(ProbabilityMatrix(pre.game_id, pre.boundary_id, pre.prefix_digest, tuple(rows)))


def runtime(phase='discussion', target=P[0]):
    _, _, env, _, audit, recorder, backend = _runtime()
    env.reset(roles=ROLES)  # Synthetic fixture only; runtime adapter never receives this table.
    _night(env, 0)
    def prior_speech(text):
        with audit.speech_perception_context(event_id=f'event-{len(env.public_events):06d}',
                boundary_id=f'synthetic-prelude-{len(env.public_events)}', speaker_id=env.current_act_idx+1):
            env.step((env.phase, text))
    if phase == 'pk_discussion':
        while env.phase == 'speech':
            speaker = f'player{env.current_act_idx+1}'
            backend.perception_response = f'{speaker} | no_commitment | NONE'
            prior_speech('我暂时不作明确表态。')
        votes = (2, 3, 1, 1, 2, 3, 0)
        while env.phase == 'vote':
            env.step(('vote', votes[env.current_act_idx]))
    while env.current_act_idx != 0:
        speaker = f'player{env.current_act_idx+1}'
        if target is None:
            backend.perception_response = f'{speaker} | no_commitment | NONE'
            prior_speech('我暂时不作明确表态。')
        else:
            backend.perception_response = f'{speaker} | point_as_werewolf | {target}'
            prior_speech(f'我认为{target}是狼人。')
    recorder.start(env, roles=ROLES)
    pending(env, recorder)
    return env, recorder, backend


def invoke(env, recorder, actor, estimator, *, src=None, condition=Comparison.CONTEXTUAL_SPEECH, policy=POLICY):
    protocol, model = estimator
    return execute_pending_speech(env=env, recorder=recorder, actor=actor,
        source=src or source(recorder._pending.prefix), model=model, protocol=protocol,
        condition=condition, policy=policy)


def test_selection_uses_direction_then_recency_not_model_scores():
    pre = witness().pre()
    selected = select_observer(pre, P[:2])
    assert selected.observer_id == P[6] and selected.relation == 'SELF_ACCUSED'
    assert selected.reason == 'PUBLIC_SELF_ACCUSATION'
    assert selected.annotation_digest and selected.event_id
    team = select_observer(witness(target=P[1]).pre(), P[:2])
    assert team.relation == 'TEAM_ACCUSED'
    # Generic no-commitment speeches are evidence of speaking, not accusations.
    neutral = select_observer(PublicWitness().pre(), P[:2])
    assert neutral.observer_id == P[6] and neutral.relation == 'NO_EXPLICIT_ACCUSATION'
    no_evidence = select_observer(prefix_for(), P[:2])
    assert no_evidence.observer_id == P[2] and no_evidence.reason == 'CANONICAL_SEAT_NO_EVIDENCE'


def test_latest_statement_does_not_resurrect_an_old_accusation():
    w = witness()
    w.speech(P[0], 'no_commitment', None, '我暂不表态。')
    w.event('turn_start', speaker=P[1])
    w.speech(P[1], 'no_commitment', None, '我暂不表态。')
    w.event('phase_change', day=1, phase='vote')
    w.event('vote_result', votes=[{'voter': p, 'target': None} for p in P])
    w.event('exile_result', exiled_players=[])
    w.event('phase_change', day=1, phase='night')
    w.event('death_announcement', dead_players=[])
    w.event('phase_change', day=2, phase='discussion')
    w.event('turn_start', speaker=P[6])
    w.speech(P[6], 'point_as_non_werewolf', P[0], '我认为player1不是狼人。')
    w.event('turn_start', speaker=P[0])
    assert select_observer(w.pre(), P[:2]).relation == 'NO_EXPLICIT_ACCUSATION'


@pytest.mark.parametrize('phase', ['discussion', 'pk_discussion'])
def test_evidence_direction_restricts_configurable_intents(estimator, phase):
    from werewolf.observer_dtrs.decision import decide
    pre = witness(target=P[1], phase=phase).pre()
    sel = select_observer(pre, P[:2])
    positive = decide(.99, POLICY.static_loss)
    plan, _ = strategy_plan(pre, sel, positive, EVIDENCE)
    assert plan.action is A.OPPOSE and plan.target == sel.observer_id
    plan, _ = strategy_plan(pre, sel, positive, REFERENCE)
    assert plan.action is A.NO_COMMITMENT  # No self-defense for a teammate-only claim.
    neutral_pre = PublicWitness(phase).pre()
    plan, reason = strategy_plan(neutral_pre, select_observer(neutral_pre, P[:2]), positive, EVIDENCE)
    assert plan.action is A.NO_COMMITMENT and reason == 'NO_DIRECTIONAL_PUBLIC_EVIDENCE'


@pytest.mark.parametrize('bad', ['OPPOSE', A.ACCUSE_WOLF, A.CLEAR, A.SUPPORT])
def test_strategy_rejects_unadmitted_role_claim_or_arbitrary_action(bad):
    with pytest.raises(ValueError):
        SpeechStrategy(A.SELF_DEFEND, bad, A.OPPOSE, A.NO_COMMITMENT)


@pytest.mark.parametrize('phase', ['discussion', 'pk_discussion'])
@pytest.mark.parametrize('target', [P[0], P[1]])
def test_real_pending_closed_loop_and_one_time_commit(estimator, phase, target):
    env, recorder, backend = runtime(phase, target)
    pre = recorder._pending.prefix
    selection = select_observer(pre, P[:2])
    # r=.8 -> high D2 event p, but the public accusation direction chooses the intent.
    semantic = 'point_as_non_werewolf' if target == P[0] else 'oppose'
    obj = P[0] if target == P[0] else selection.observer_id
    text = f'我认为{P[0]}不是狼人。' if target == P[0] else f'我质疑{obj}此前的判断。'
    actor_backend = ActorBackend(text)
    actor = GPTAgent(backend=actor_backend, model_name='controlled-actor')
    backend.perception_response = f'player1 | {semantic} | {obj}'
    raw_before = len(recorder.submitted_gameplay_actions)
    rng_before = env._rng.getstate()
    result = invoke(env, recorder, actor, estimator, src=source(pre, .8))
    assert result.audit['prediction']['scope'] == 'SYNTHETIC_TEST_ONLY'
    assert result.audit['selection']['observer_id'] == selection.observer_id
    assert result.audit['decision']['region'] == 'POS'
    assert result.audit['decision']['probability'] != result.audit['features']['r']
    assert recorder._pending is None and len(recorder.submitted_gameplay_actions) == raw_before+1
    assert recorder.submitted_gameplay_actions[-1].action_payload.to_value() == text
    assert env.speech_annotations[-1].actions[0].to_record() == [P[0], semantic, obj]
    assert env._rng.getstate() == rng_before
    assert len(actor_backend.calls) == 1
    request = json.dumps(actor_backend.calls)
    assert all(v not in request for v in (result.audit['model_digest'], pre.prefix_digest, 'known_wolf_ids', 'probability', 'loss_digest'))
    assert all('selection' not in e and 'prediction' not in e and 'condition' not in e for e in env.public_events)
    with pytest.raises(ValueError, match='pending'):
        invoke(env, recorder, actor, estimator, src=source(pre, .8))
    assert len(actor_backend.calls) == 1


@pytest.mark.parametrize('bad_parse', ['NONE', 'player1 | point_as_werewolf | player4'])
def test_parser_failure_is_atomic_and_does_not_fallback(estimator, bad_parse):
    env, recorder, backend = runtime()
    actor = GPTAgent(backend=ActorBackend('我认为player1不是狼人。'), model_name='controlled')
    backend.perception_response = bad_parse
    before = snapshot(env, recorder)
    with pytest.raises(ValueError):
        invoke(env, recorder, actor, estimator, src=source(recorder._pending.prefix, .8))
    assert snapshot(env, recorder) == before and len(actor.backend.calls) == 1


def test_three_comparisons_share_probability_reference_mapping_and_explicit_tie(estimator):
    results = []
    for condition in Comparison:
        env, recorder, backend = runtime('pk_discussion')
        positive = condition is not Comparison.STATIC_D3  # same p=.5; static alpha=.75, PK alpha=3/7
        text = '我认为player1不是狼人。' if positive else '我暂时不作明确表态。'
        backend.perception_response = f"player1 | {'point_as_non_werewolf | player1' if positive else 'no_commitment | NONE'}"
        actor = GPTAgent(backend=ActorBackend(text), model_name='controlled')
        results.append(invoke(env, recorder, actor, estimator, condition=condition).audit)
    assert {r['decision']['probability'] for r in results} == {.5}
    assert len({r['model_digest'] for r in results}) == 1
    assert [r['decision']['region'] for r in results] == ['BND', 'POS', 'POS']
    assert results[0]['policy'] == results[1]['policy'] == results[2]['policy']
    assert results[0]['strategy'] == results[1]['strategy']
    assert results[1]['loss'] == results[2]['loss']
    assert results[1]['strategy'] != results[2]['strategy']


@pytest.mark.parametrize('kind', ['raw_matrix', 'unproven_provider', 'mismatched_pre', 'wrong_sealed_type'])
def test_source_rejection_precedes_language_and_env_mutation(estimator, kind):
    env, recorder, _ = runtime()
    pre = recorder._pending.prefix
    src = {'raw_matrix': source(pre).matrix, 'unproven_provider': SimpleNamespace(probabilities=lambda _: source(pre).matrix),
           'mismatched_pre': source(prefix_for()), 'wrong_sealed_type': SealedQwen3Source(object(), 'a'*64, 'b'*64)}[kind]
    actor = GPTAgent(backend=ActorBackend('unused'), model_name='controlled')
    before = snapshot(env, recorder)
    with pytest.raises((TypeError, ValueError)):
        invoke(env, recorder, actor, estimator, src=src)
    assert snapshot(env, recorder) == before and not actor.backend.calls


def test_formal_estimator_transfer_is_not_assumed_from_a_model_digest(estimator):
    env, recorder, _ = runtime()
    protocol, model = estimator
    record = model.record
    record['source']['scope'] = 'formal_training_partition'
    actor = GPTAgent(backend=ActorBackend('unused'), model_name='controlled')
    with pytest.raises(ValueError, match='separately admitted'):
        invoke(env, recorder, actor, (protocol, FittedModel(canonical_json_bytes(record))))
    assert not actor.backend.calls


@pytest.mark.parametrize('field,value', [('identity', 'Seer'), ('observer_id', 2), ('current_act_idx', 2), ('phase', '2_day_speech')])
def test_other_identity_or_stale_observation_cannot_supply_team(field, value):
    env, recorder, _ = runtime()
    observation = env.get_observation()
    observation[field] = value
    with pytest.raises(ValueError, match='bind PRE'):
        _legal_team(recorder._pending.prefix, observation)


def test_comparison_requires_registered_d3_loss_and_explicit_policy():
    with pytest.raises(ValueError, match='registered'):
        ResearchPolicy(replace(POLICY.static_loss, c_fp=999), REFERENCE, EVIDENCE)
    with pytest.raises(TypeError):
        ResearchPolicy(POLICY.static_loss, None, EVIDENCE)


def test_added_strategy_changes_intent_without_changing_contextual_risk(estimator):
    results = []
    for condition in (Comparison.CONTEXTUAL_LOSS, Comparison.CONTEXTUAL_SPEECH):
        env, recorder, backend = runtime(target=P[1])
        selection = select_observer(recorder._pending.prefix, P[:2])
        oppose = condition is Comparison.CONTEXTUAL_SPEECH
        text = f'我质疑{selection.observer_id}的判断。' if oppose else '我暂时不作明确表态。'
        backend.perception_response = f"player1 | oppose | {selection.observer_id}" if oppose else 'player1 | no_commitment | NONE'
        actor = GPTAgent(backend=ActorBackend(text), model_name='controlled')
        results.append(invoke(env, recorder, actor, estimator, condition=condition,
                              src=source(recorder._pending.prefix, .8)).audit)
    assert results[0]['decision'] == results[1]['decision']
    assert results[0]['loss'] == results[1]['loss']
    assert results[0]['public_payload']['action'] == 'NO_COMMITMENT'
    assert results[1]['public_payload']['action'] == 'OPPOSE'


def test_no_q_or_event_probability_averaging_across_observers(estimator):
    env, recorder, backend = runtime()
    pre = recorder._pending.prefix
    selected = select_observer(pre, P[:2]).observer_id
    rows = list(source(pre, .1).matrix.values)
    rows[P.index(selected)] = source(pre, .8).matrix.row(selected).values
    src = SyntheticPrediction(replace(source(pre).matrix, values=tuple(rows)))
    actor = GPTAgent(backend=ActorBackend('我认为player1不是狼人。'), model_name='controlled')
    backend.perception_response = 'player1 | point_as_non_werewolf | player1'
    result = invoke(env, recorder, actor, estimator, src=src)
    assert result.audit['selection']['observer_id'] == selected
    assert result.audit['features']['r'] == .8
    assert result.audit['decision']['probability'] > .75
    assert len(result.audit['prediction']['matrix_digest']) == 64


def test_explicit_self_accusation_precedes_newer_teammate_accusation():
    w = PublicWitness()
    w.events = w.events[:3]
    w.annotations = []
    for observer in P[2:]:
        target = P[0] if observer == P[2] else P[1]
        w.event('turn_start', speaker=observer)
        w.speech(observer, 'point_as_werewolf', target, f'我认为{target}是狼人。')
    w.event('turn_start', speaker=P[0])
    assert select_observer(w.pre(), P[:2]).observer_id == P[2]
