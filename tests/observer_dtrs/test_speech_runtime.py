"""Synthetic source + real PRE/Actor/parser/verified env commit, no model service."""
from collections import Counter
from dataclasses import replace
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from werewolf.strategy.speech_planning import PlanningAction as A, generate_candidates
from tests.observer_dtrs.conftest import prefix_for
from tests.observer_dtrs.test_contextual_policy import fitted
from tests.tom.test_counterfactual_consumer import PublicWitness, P
from tests.tom.test_speech_realization import ActorBackend
from tests.tom.test_verified_speech_commit import pending, snapshot
from tests.canonical_collection.test_runtime_game_evidence import _runtime, ROLES
from tests.canonical_collection.test_final_capacity import _night
from werewolf.agents.gpt_agent import GPTAgent
from werewolf.artifact_io import canonical_json_bytes
from werewolf.canonical_collection.call_audit import AuditedBackend
from werewolf.observer_dtrs.context import contexts_for_pre
from werewolf.observer_dtrs.d3_protocol import load_protocol
from werewolf.observer_dtrs.decision import Region, TiePolicy
from werewolf.observer_dtrs.estimation import FittedModel
from werewolf.observer_dtrs.final_tom_client import CapsuleGameplayPredictor
from werewolf.observer_dtrs.probabilities import ProbabilityMatrix
from werewolf.observer_dtrs.speech_runtime import (
    Comparison, ReportEventArm, ReportEventGamePolicy, ResearchPolicy, SealedQwen3Source,
    SpeechStrategy, SyntheticPrediction,
    _legal_team, _report_event_panel, execute_pending_speech, select_observer, strategy_plan,
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


def runtime(phase='discussion', target=P[0], *, native=False):
    _, claim, env, _, audit, recorder, backend = _runtime()
    if native:
        from werewolf.envs.werewolf_text_env_v0 import WerewolfTextEnvV0
        env = WerewolfTextEnvV0(speech_perceiver=env.speech_perceiver, random_seed=claim.seed,
                                log_save_path=None)
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


@pytest.fixture
def report_event_parameters():
    from scripts import fit_observer_dtrs_report_event as fit

    features = fit.FEATURES['M1']
    slope = [8. if name == 'actor_mass' else 0. for name in features]
    return {'features': features, 'standardization': {'mean': [0.] * 3, 'std': [1.] * 3},
            'base': {'intercept': -4., 'coefficients': slope},
            'calibrator': {'intercept': 0., 'coefficients': [1.]}}


def report_event_prediction(pre, player1_mass=.5):
    matrix = source(pre).matrix
    rows = list(matrix.values)
    for observer in P[2:]:
        rows[P.index(observer)] = (player1_mass, 1 - player1_mass, 0., 0., 0., 0., 0.)
    return SyntheticPrediction(replace(matrix, values=tuple(rows))).probabilities(pre)


@pytest.fixture
def report_event_predictor(monkeypatch):
    predict = Mock(side_effect=report_event_prediction)
    monkeypatch.setattr(CapsuleGameplayPredictor, 'require_report_event_ready', lambda _: None)
    monkeypatch.setattr(CapsuleGameplayPredictor, 'predict_report_event', lambda _, pre: predict(pre))
    return object.__new__(CapsuleGameplayPredictor), predict


def handle_report_event(policy, env, recorder):
    return policy.handle_pre(env=env, recorder=recorder, call_audit=recorder.call_audit,
                             observation=env.get_observation(), handoff=recorder._pending.handoff)


class ReportEventBackend(ActorBackend):
    supports_json_schema = True

    def __init__(self, speech, choose):
        super().__init__(speech)
        self.choose = choose

    def chat_with_metadata(self, **kwargs):
        if 'response_format' not in kwargs:
            return super().chat_with_metadata(**kwargs)
        assert kwargs['response_format']['json_schema']['name'] == 'report_event_plan'
        self.calls.append(kwargs)
        context = json.loads(kwargs['messages'][0]['content'].rsplit('\n\n', 1)[1])
        return json.dumps(self.choose(context)), {'finish_reason': 'stop'}


def plan_choice(context, action, target, basis_ref):
    index = context['candidates'].index({'action': action, 'target': target})
    return {'candidate_index': index, 'basis_ref': basis_ref}


@pytest.mark.parametrize(('arm', 'directions'), [
    (ReportEventArm.TOM_ONLY, [['POS', 'NEG']] * 5),
    (ReportEventArm.TWO_WAY, [['NEG'], ['NEG'], ['NEG'], ['POS'], ['POS']]),
    (ReportEventArm.THREE_WAY, [['NEG'], [], [], [], ['POS']]),
])
def test_report_event_commitments_at_frozen_boundaries(monkeypatch, report_event_parameters, arm, directions):
    from scripts import analyze_observer_dtrs_report_event as study

    probabilities = dict(zip(P[2:], [.10, .25, .50, .75, .90], strict=True))
    monkeypatch.setattr(study, 'rebuild_probabilities', lambda *_: probabilities)
    pre = witness().pre()
    result = _report_event_panel(pre, contexts_for_pre(pre, P[:2]), report_event_prediction(pre),
                                 m1_parameters=report_event_parameters, arm=arm)
    assert result['loss'] == study.LOSS.to_record()
    assert result['panel'] == {
        seat: {'probability': p, 'adoptable_directions': allowed}
        for (seat, p), allowed in zip(probabilities.items(), directions, strict=True)
    }


def test_report_event_arms_share_unaccused_pre_and_legal_inputs(report_event_parameters, report_event_predictor):
    predictor, predict = report_event_predictor
    common_inputs, public_requests = [], []
    for arm in ReportEventArm:
        env, recorder, backend = runtime(target=None, native=True)
        pre = recorder._pending.prefix
        assert select_observer(pre, P[:2]).relation == 'NO_EXPLICIT_ACCUSATION'
        backend.perception_response = 'player1 | point_as_non_werewolf | player1'
        actor_backend = ReportEventBackend('我认为player1不是狼人。',
            lambda context: plan_choice(context, 'SELF_DEFEND', None, 'independent'))
        actor = GPTAgent(backend=actor_backend, model_name='controlled')
        evidence = {}
        policy = ReportEventGamePolicy(recorder=recorder, agents=(actor,) * 7, predictor=predictor,
            arm=arm, probability=1/3, m1_parameters=report_event_parameters, record_evidence=evidence.__setitem__)
        predict.reset_mock()
        assert handle_report_event(policy, env, recorder)[2] is False
        predict.assert_called_once_with(pre)
        private, public = actor_backend.calls
        prompt, raw_context = private['messages'][0]['content'].rsplit('\n\n', 1)
        context = json.loads(raw_context)
        assert context['candidates'] == [plan.public_payload() for plan in generate_candidates(pre)]
        assert context['known_wolf_team'] == list(P[:2])
        assert context['public_events'] == pre.public_event_history.to_records()
        assert context['report_predictions'] == evidence['execution']['panel']
        assert {seat: row['probability'] for seat, row in context['report_predictions'].items()} == dict.fromkeys(P[2:], .5)
        assert evidence['execution']['public_payload'] == {'action': 'SELF_DEFEND', 'target': None}
        assert evidence['execution']['basis_ref'] == 'independent'
        assert evidence['execution']['tom_integration'] == {'A': 1, 'P': 1, 'S': 'UNKNOWN'}
        for row in context['report_predictions'].values():
            del row['adoptable_directions']
        call_settings = {key: private[key] for key in ('model', 'temperature', 'max_tokens', 'extra_body')}
        common_inputs.append((prompt, context, call_settings))
        public_requests.append(public)
    assert common_inputs[0] == common_inputs[1] == common_inputs[2]
    assert public_requests[0] == public_requests[1] == public_requests[2]


@pytest.mark.parametrize('problem', ['index', 'bnd_prediction'])
def test_report_event_rejects_invalid_private_selection(report_event_parameters, report_event_predictor, problem):
    env, recorder, backend = runtime(native=True)
    predictor, _ = report_event_predictor

    def choose(context):
        if problem == 'index':
            return {'candidate_index': len(context['candidates']), 'basis_ref': 'independent'}
        return plan_choice(context, 'SELF_DEFEND', None, 'prediction:player3:NEG')

    actor_backend = ReportEventBackend('不会生成这段发言。', choose)
    actor = GPTAgent(backend=actor_backend, model_name='controlled')
    evidence = {}
    policy = ReportEventGamePolicy(recorder=recorder, agents=(actor,) * 7, predictor=predictor,
        arm=ReportEventArm.THREE_WAY, probability=1/3, m1_parameters=report_event_parameters,
        record_evidence=evidence.__setitem__)
    before, parser_calls = snapshot(env, recorder), len(backend.requests)
    with pytest.raises(ValueError, match='合法候选索引和相容的依据引用'):
        handle_report_event(policy, env, recorder)
    assert snapshot(env, recorder) == before and len(backend.requests) == parser_calls
    assert len(actor_backend.calls) == 1
    assert evidence['execution']['status'] == 'FAILED' and 'intent' not in evidence


def test_report_event_semantic_failure_leaves_pre_uncommitted(report_event_parameters, report_event_predictor):
    env, recorder, backend = runtime(native=True)
    predictor, _ = report_event_predictor
    actor_backend = ReportEventBackend('我认为player1不是狼人。',
        lambda context: plan_choice(context, 'SELF_DEFEND', None, 'independent'))
    actor = GPTAgent(backend=actor_backend, model_name='controlled')
    backend.perception_response = 'player1 | point_as_werewolf | player4'
    evidence = {}
    policy = ReportEventGamePolicy(recorder=recorder, agents=(actor,) * 7, predictor=predictor,
        arm=ReportEventArm.TOM_ONLY, probability=1/3, m1_parameters=report_event_parameters,
        record_evidence=evidence.__setitem__)
    before = snapshot(env, recorder)
    with pytest.raises(ValueError, match='does not exactly match'):
        handle_report_event(policy, env, recorder)
    assert snapshot(env, recorder) == before
    assert evidence['execution']['status'] == 'FAILED' and 'commit' not in evidence['execution']
    assert len(actor_backend.calls) == 2


def test_report_event_wrong_q_binding_stops_before_language(report_event_parameters, report_event_predictor):
    env, recorder, _ = runtime(native=True)
    predictor, predict = report_event_predictor
    actor_backend = ActorBackend('unused')
    actor = GPTAgent(backend=actor_backend, model_name='controlled')
    other = replace(source(recorder._pending.prefix).matrix, boundary_id='unrelated-pre')
    predict.side_effect = None
    predict.return_value = other, {'scope': 'SYNTHETIC_TEST_ONLY'}
    evidence = {}
    policy = ReportEventGamePolicy(recorder=recorder, agents=(actor,) * 7, predictor=predictor,
        arm=ReportEventArm.THREE_WAY, probability=1/3, m1_parameters=report_event_parameters,
        record_evidence=evidence.__setitem__)
    before = snapshot(env, recorder)
    with pytest.raises(ValueError, match='Q 与当前 PRE 不一致'):
        handle_report_event(policy, env, recorder)
    assert snapshot(env, recorder) == before and not actor_backend.calls
    assert evidence['execution']['status'] == 'FAILED'
    assert evidence['execution']['tom_integration'] == {'A': 1, 'P': 0, 'S': 0}


@pytest.mark.parametrize("problem", ["unknown_response", "missing_payload"])
def test_report_event_requires_actual_matching_decision_request(
        report_event_parameters, report_event_predictor, problem, monkeypatch):
    env, recorder, parser = runtime(native=True)
    predictor, _ = report_event_predictor
    parser.perception_response = 'player1 | point_as_non_werewolf | player1'

    def choose(context):
        if problem == "unknown_response":
            raise TimeoutError("no decision receipt")
        assert not context['report_predictions']
        return plan_choice(context, 'SELF_DEFEND', None, 'independent')

    backend = ReportEventBackend('我认为player1不是狼人。', choose)
    actor = GPTAgent(backend=AuditedBackend(backend, recorder.call_audit), model_name='controlled')
    original = actor.select_report_event_plan
    if problem == "missing_payload":
        def without_payload(context, *, basis_refs):
            return original({**context, 'report_predictions': {}}, basis_refs=basis_refs)
        monkeypatch.setattr(actor, 'select_report_event_plan', without_payload)
    evidence = {}
    policy = ReportEventGamePolicy(recorder=recorder, agents=(actor,) * 7, predictor=predictor,
        arm=ReportEventArm.TOM_ONLY, probability=1/3, m1_parameters=report_event_parameters,
        record_evidence=evidence.__setitem__)
    if problem == "unknown_response":
        with pytest.raises(TimeoutError, match="no decision receipt"):
            handle_report_event(policy, env, recorder)
    else:
        assert handle_report_event(policy, env, recorder)[2] is False
    row = evidence['execution']
    assert row['tom_integration'] == {'A': 1, 'P': 1, 'S': 'UNKNOWN' if problem == "unknown_response" else 0}
    assert not row['decision_call_ids'] and row['plan_calls'] == 1
    assert len(row['unconfirmed_decision_call_ids']) == (1 if problem == "unknown_response" else 0)


@pytest.mark.parametrize('arm', list(ReportEventArm))
def test_report_event_game_hook_records_assignment_and_intent_before_language(
        report_event_parameters, report_event_predictor, arm):
    env, recorder, parser = runtime(native=True)
    handoff = recorder._pending.handoff
    pre = recorder.public_pre_for_handoff(handoff)
    selection = select_observer(pre, P[:2])
    public_ref = f'public:{selection.event_id}'

    def choose(context):
        if context['report_predictions'][selection.observer_id]['adoptable_directions'] == ['NEG']:
            return plan_choice(context, 'ACCUSE_WOLF', P[2], f'prediction:{selection.observer_id}:NEG')
        basis_ref = 'keep_both' if context['report_predictions'][selection.observer_id]['adoptable_directions'] else public_ref
        return plan_choice(context, 'SELF_DEFEND', None, basis_ref)

    binary = arm is ReportEventArm.TWO_WAY
    text = '我认为player3是狼人。' if binary else '我认为player1不是狼人。'
    parser.perception_response = ('player1 | point_as_werewolf | player3' if binary
                                  else 'player1 | point_as_non_werewolf | player1')
    parser_calls = len(parser.requests)
    actor_backend = ReportEventBackend(text, choose)
    actor = GPTAgent(backend=AuditedBackend(actor_backend, recorder.call_audit), model_name='controlled')
    predictor, predict = report_event_predictor
    evidence = {}

    def record(stage, row):
        if stage == 'assignment':
            assert not actor_backend.calls
        elif stage == 'intent':
            assert len(actor_backend.calls) == 1 and len(parser.requests) == parser_calls
        assert stage not in evidence
        evidence[stage] = row

    policy = ReportEventGamePolicy(recorder=recorder, agents=(actor,) * 7, predictor=predictor,
                                arm=arm, probability=1/3, m1_parameters=report_event_parameters,
                                record_evidence=record)
    assert handle_report_event(policy, env, recorder)[2] is False
    with pytest.raises(ValueError, match='unconsumed canonical PRE'):
        recorder.public_pre_for_handoff(handoff)
    predict.assert_called_once_with(pre)
    assert [stage for stage in evidence] == ['assignment', 'intent', 'execution']
    assert evidence['assignment']['arm'] == arm.value and evidence['assignment']['probability'] == 1/3
    expected_plan = {'action': 'ACCUSE_WOLF', 'target': P[2]} if binary else {'action': 'SELF_DEFEND', 'target': None}
    assert evidence['intent']['public_payload'] == expected_plan
    if arm is ReportEventArm.THREE_WAY:
        assert evidence['intent']['basis_ref'] == public_ref
    execution = evidence['execution']
    assert execution['status'] == 'COMMITTED' and execution['prefix_digest'] == pre.prefix_digest
    assert [execution[k] for k in ('q_requests', 'plan_calls', 'actor_calls', 'parser_calls')] == [1, 1, 1, 1]
    assert execution['tom_integration'] == {'A': 1, 'P': 1, 'S': 1}
    assert len(execution['decision_call_ids']) == 1
    submitted, = recorder.submitted_gameplay_actions
    annotation = env.speech_annotations[-1]
    assert execution['commit'] == {'action_id': submitted.action_id, 'event_id': annotation.event_id,
                                   'annotation_digest': annotation.annotation_digest}
    assert execution['assignment_ref'] == evidence['intent']['assignment_ref']
    assert submitted.action_payload.to_value() == text and recorder._pending is None
    assert len(actor_backend.calls) == 2 and len(parser.requests) == parser_calls + 1
    public_transcript = json.dumps([actor_backend.calls[1], parser.requests[-1], env.public_events])
    for private_value in ('report_predictions', 'known_wolf_team', 'adoptable_directions',
                          'basis_ref', execution['assignment_ref'], execution['model_digest'], arm.value):
        assert private_value not in public_transcript


def test_report_event_entry_assigns_three_arms_and_records_each_pre(
        monkeypatch, tmp_path, report_event_parameters, report_event_predictor):
    from scripts import analyze_observer_dtrs_report_event as study
    from tests.canonical_collection.test_attempt_ledger import _claim, _plan
    from werewolf.canonical_collection import attempt_ledger, collector, production_runtime
    from werewolf.interventions import gameplay_server

    seeds = (101, 202, 303, 404, 505)
    plan = _plan(ordered_seed_pool=seeds, target_canonical_success_count=len(seeds))
    reference = SimpleNamespace(collection_id='paper-canonical-qwen35-9b-5800-v1', ordered_seed_pool=(1, 2),
        plan_digest='ccc63353fe626ba5b76b06174a94272874a650f9a12b404de7c98c646b1e6bbe')
    monkeypatch.setattr(attempt_ledger, 'load_collection_plan', lambda _: reference)
    load_frozen = Mock(return_value={'models': {'M1': report_event_parameters}})
    monkeypatch.setattr(study, 'open_fitted_models', load_frozen)
    predictor, predict = report_event_predictor
    monkeypatch.setattr(CapsuleGameplayPredictor, 'identity', property(lambda _self: {'fixture': 'Q'}))
    runtime_factory = object.__new__(production_runtime.Classic7RuntimeFactory)
    runtime_factory.f0_control = None
    monkeypatch.setattr(production_runtime.Classic7RuntimeFactory, '__call__', lambda _self, *, plan, claim:
        SimpleNamespace(agents=(None,) * 7, recorder=SimpleNamespace(claim=claim, plan=plan,
            game_id=f'fixture-{claim.ordinal}', runtime_configuration={})))
    replay = object()
    monkeypatch.setattr(production_runtime, 'classic7_full_game_replay_executor', lambda _: replay)
    writes = {}

    def write_once(path, row):
        assert path not in writes
        writes[path] = row

    monkeypatch.setattr(gameplay_server, 'write_once', write_once)
    result = object()

    def collect(*, plan, runtime_factory, destination):
        assert destination == tmp_path / 'pilot'
        for ordinal in range(len(seeds)):
            runtime = runtime_factory(plan=plan, claim=_claim(plan, ordinal=ordinal, attempt_id=f'attempt-{ordinal}'))
            assert runtime.replay_executor is replay
            if ordinal == 0:
                for step in (5, 11):
                    for stage in ('intent', 'execution'):
                        runtime.online_pilot._record(stage, {'step_index': step})
        return result

    monkeypatch.setattr(collector, 'collect', collect)
    assert gameplay_server.execute_report_event_games(plan=plan, reference_collection=tmp_path / 'reference',
        assignment_seed=1729, destination=tmp_path / 'pilot', runtime_factory=runtime_factory, predictor=predictor) is result
    load_frozen.assert_called_once_with()
    predict.assert_not_called()
    allocations = writes[tmp_path / 'pilot' / 'assignments.json']['allocations']
    assert tuple(row['seed'] for row in allocations) == seeds
    counts = Counter(row['arm'] for row in allocations)
    assert set(counts) == {arm.value for arm in ReportEventArm} and sorted(counts.values()) == [1, 2, 2]
    for ordinal, row in enumerate(allocations):
        actual = writes[tmp_path / 'pilot' / 'speech-evidence' / f'fixture-{ordinal}' / 'assignment.json']
        assert {key: actual[key] for key in row} == row
        assert actual['probability'] == counts[row['arm']] / len(seeds)
    recorded = tmp_path / 'pilot' / 'speech-evidence' / 'fixture-0'
    for step in (5, 11):
        for stage in ('intent', 'execution'):
            assert writes[recorded / f'{step:06d}-{stage}.json'] == {'step_index': step}


@pytest.mark.parametrize('fail_at', [None, 5])
def test_report_event_full_game_tracks_each_wolf_pre(
        tmp_path, report_event_parameters, report_event_predictor, fail_at):
    from tests.runtime.test_runtime_config import new_config
    from werewolf.canonical_collection.collector import CanonicalAttemptFailure
    from werewolf.canonical_collection.game_bundle import publish_canonical_game_bundle
    from werewolf.canonical_collection.production_runtime import _RecordedRuntime, classic7_full_game_replay_executor
    from werewolf.envs.werewolf_text_env_v0 import WerewolfTextEnvV0
    from werewolf.strategy.counterfactual_tom import derive_public_phase_speaker_order

    plan, claim, setup, agents, audit, recorder, parser = _runtime()
    env = WerewolfTextEnvV0(random_seed=claim.seed, log_save_path=None, speech_perceiver=setup.speech_perceiver)
    original_calls = []
    private_inputs = []

    def choose(context, seat):
        assert context['speaker'] == seat
        private_inputs.append(context)
        parser.perception_response = f'{seat} | no_commitment | NONE'
        return plan_choice(context, 'NO_COMMITMENT', None, 'keep_both')

    for i, original in enumerate(agents):
        def original_speech(observation, *, pre_speech_belief, original=original.act_with_pre_speech_belief):
            assert observation['identity'] != 'Werewolf'
            original_calls.append(observation['phase'])
            parser.perception_response = 'NONE'
            return original(observation, pre_speech_belief=pre_speech_belief)

        def act(observation, original_act=original.act):
            phase = observation['phase']
            seat = observation['current_act_idx']
            original_calls.append(phase)
            if phase.endswith(('skill_wolf', 'vote_pk')):
                target = 0
            elif phase == '1_day_vote':
                target = (2, 1, 1, 1, 2, 2, 0)[seat - 1]  # 两狼并列进入 PK。
            elif phase.endswith('_day_vote'):
                exile = 1 if phase.startswith('2_') else 2
                target = exile if seat != exile else 0
            else:
                return original_act(observation)
            return next(action for action in observation['valid_action'] if action[1] == target)

        if i < 2:
            actor_backend = ReportEventBackend('我暂时不作明确表态。', lambda context, seat=P[i]: choose(context, seat))
            agents[i] = GPTAgent(backend=AuditedBackend(actor_backend, audit), model_name='controlled')
        agents[i].act_with_pre_speech_belief = original_speech
        agents[i].act = act

    predictor, predict = report_event_predictor

    def predict_for_pre(pre):
        if predict.call_count == fail_at:
            raise ValueError('fixture Q budget exhausted')
        state = pre.public_temporal_state
        mass = .5 if state.phase.value == 'pk_discussion' else (.9 if state.day == 1 else .1)
        return report_event_prediction(pre, mass)

    predict.side_effect = predict_for_pre
    rows = []
    policy = ReportEventGamePolicy(recorder=recorder, agents=agents, predictor=predictor,
        arm=ReportEventArm.THREE_WAY, probability=1/3, m1_parameters=report_event_parameters,
        record_evidence=lambda stage, row: rows.append((stage, row)))
    replay = classic7_full_game_replay_executor(new_config())
    runtime = _RecordedRuntime(env, agents, list(ROLES), recorder, audit, replay, online_pilot=policy)
    if fail_at is not None:
        with pytest.raises(CanonicalAttemptFailure, match='fixture Q budget exhausted'):
            runtime.run()
        executions = [row for stage, row in rows if stage == 'execution']
        assert [row['status'] for row in executions] == ['COMMITTED'] * (fail_at - 1) + ['FAILED']
        assert executions[-1]['q_requests'] == 1 and executions[-1]['q_successes'] == 0
        assert executions[-1]['tom_integration'] == {'A': 1, 'P': 0, 'S': 0}
        assert len(private_inputs) == fail_at - 1 and predict.call_count == fail_at
        assert recorder._pending.prefix.boundary_id == executions[-1]['boundary_id']
        assert [stage for stage, _ in rows].count('runtime') == 0 and env.phase != 'end_game'
        return

    product = runtime.run()
    all_pres = product.evidence.authoritative_pre_prefixes
    wolf_pres = [pre for pre in all_pres if pre.current_speaker in P[:2]]
    assert predict.call_count == len(wolf_pres) == 7
    assert [call.args[0] for call in predict.call_args_list] == wolf_pres
    assert len(private_inputs) == len(wolf_pres)
    assignment = rows[0][1]
    assert [stage for stage, _ in rows] == ['assignment'] + ['intent', 'execution'] * len(wolf_pres) + ['runtime']
    summary = rows[-1][1]
    assert summary['q_requests'] == summary['q_successes'] == len(wolf_pres)
    assert summary['tom_integration']['S'] == len(wolf_pres)
    submitted = {action.boundary_id: action for action in product.evidence.submitted_gameplay_actions}
    wolf_submissions = [action for action in product.evidence.submitted_gameplay_actions
                        if action.actor_id in P[:2] and action.action_type == 'public_speech']
    assert [action.boundary_id for action in wolf_submissions] == [pre.boundary_id for pre in wolf_pres]
    executions = [row for stage, row in rows if stage == 'execution']
    for pre, context, execution in zip(wolf_pres, private_inputs, executions, strict=True):
        assert execution['arm'] == assignment['arm'] == 'tom_3wd'
        assert execution['assignment_ref'] == assignment['assignment_ref']
        assert (execution['game_id'], execution['attempt_id']) == (recorder.game_id, claim.attempt_id)
        assert (execution['acting_wolf'], execution['step_index']) == (pre.current_speaker, pre.step_index)
        assert execution['status'] == 'COMMITTED'
        assert execution['commit']['action_id'] == submitted[pre.boundary_id].action_id
        assert [execution[k] for k in ('q_requests', 'plan_calls', 'actor_calls', 'parser_calls')] == [1, 1, 1, 1]
        assert set(context) == {'speaker', 'alive_players', 'known_wolf_team', 'public_events',
                                'report_predictions', 'candidates'}
        assert context['public_events'] == pre.public_event_history.to_records()
        assert context['report_predictions'] == execution['panel']
        assert set(execution['panel']) == set(pre.alive_observer_ids).difference(P[:2])
        assert all(set(row) == {'probability', 'adoptable_directions'} for row in execution['panel'].values())

    pk = [pre for pre in wolf_pres if pre.public_temporal_state.phase.value == 'pk_discussion']
    assert len(pk) == 2 and pk[-1].current_speaker == derive_public_phase_speaker_order(pk[-1])[-1]
    player1_rows = [row['panel'][P[2]] for row in executions if row['acting_wolf'] == P[0]]
    assert [row['adoptable_directions'] for row in player1_rows] == [['POS'], [], ['NEG']]
    assert player1_rows[0]['probability'] > player1_rows[1]['probability'] > player1_rows[2]['probability']
    player2_rows = [row['panel'][P[2]] for row in executions if row['acting_wolf'] == P[1]]
    assert [row['adoptable_directions'] for row in player2_rows] == [['NEG'], [], ['POS'], ['POS']]
    assert any('skill_wolf' in phase for phase in original_calls) and any('vote' in phase for phase in original_calls)
    assert sum('speech' in phase for phase in original_calls) == len(all_pres) - len(wolf_pres)
    assert env.phase == 'end_game' and env.is_done()[1:] == (True, {'Werewolf': -1})
    bundle = publish_canonical_game_bundle(tmp_path / recorder.game_id, plan=plan, claim=claim,
                                          evidence=product.evidence, replay_executor=product.replay_executor)
    assert bundle.manifest['canonical_eligibility'] is True
