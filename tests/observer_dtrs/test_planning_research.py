"""Artificial PREs, inference tables and B2 tensors only; no real model/game."""
from dataclasses import replace
import inspect
import random

import numpy as np
import pytest
import torch

from scripts import counterfactual_tom as cf
from scripts.speech_planning import generate_candidates, evaluate_candidates
from tests.tom.test_counterfactual_consumer import PublicWitness, P
from werewolf.artifact_io import canonical_json_bytes
from werewolf.observer_dtrs.d3_protocol import load_protocol
from werewolf.observer_dtrs.decision import TiePolicy, decide
from werewolf.observer_dtrs.features import observer_features
from werewolf.observer_dtrs.online_context import ShadowRequest, digest
from werewolf.observer_dtrs.online_model import FrozenEventEstimator, TomIdentity, TYPE, VERSION as MODEL_VERSION
from werewolf.observer_dtrs.planning_research import (
    ControlledB2, D3_PROTOCOL_DIGEST, analyze_pre,
)
from werewolf.observer_dtrs.probabilities import ProbabilityMatrix
from werewolf.observer_dtrs.shadow import QPrediction
from werewolf.structured_history import plan_structured_history
from werewolf.tom.dataset import ExperimentCapacity


POLICY = TiePolicy.TERMINAL_NEG_FIRST


def witness(phase='discussion', dead=(), last=False):
    """Explicit public event witness, not an executed or sampled game."""
    alive = tuple(p for p in P if p not in dead)
    w = PublicWitness()
    w.events, w.annotations, w.alive = [], [], alive
    w.event('phase_change', day=0, phase='night')
    w.event('death_announcement', dead_players=list(dead))
    w.event('phase_change', day=1, phase='discussion')
    if phase == 'pk_discussion':
        for p in alive:
            w.event('turn_start', speaker=p)
            w.speech(p, 'no_commitment', None, '本轮暂不表态。')
        other = next(p for p in alive if p != P[0])
        w.event('phase_change', day=1, phase='vote')
        w.event('vote_result', votes=[{'voter': p, 'target': other if p == P[0] else
                                     P[0] if p == other else None} for p in alive])
        w.event('exile_result', exiled_players=[])
        w.event('phase_change', day=1, phase='pk_discussion')
        prior = (other,) if last else ()
    else:
        # Starting at the next living seat makes player1 the final speaker.
        prior = alive[1:] if last else ()
    for p in prior:
        w.event('turn_start', speaker=p)
        w.speech(p, 'no_commitment', None, '本轮暂不表态。')
    w.event('turn_start', speaker=P[0])
    return w


def inputs(*, phase='discussion', dead=(), last=False, probability=.5):
    w = witness(phase, dead, last)
    pre = w.pre()
    request = ShadowRequest(pre, P[:2], 'controlled-planning', 'attempt-1', captured_at_ns=0)
    tom = TomIdentity('controlled_fixture', None, '1'*64, '2'*64, '3'*64, '4'*64, None)
    rows = [[0. if i == j else 1/6 for j in range(7)] for i in range(7)]
    rows[2] = [.1, .2, 0., .1, .1, .2, .3]
    q = ProbabilityMatrix(pre.game_id, pre.boundary_id, pre.prefix_digest, rows)
    prediction = QPrediction(q, tom, plan_structured_history(pre).plan_digest)
    # Hand-declared constant EQ0 inference table. No fit, label or training read.
    record = {'schema_version': MODEL_VERSION, 'scope': 'controlled_fixture', 'model_id': 'OBS-EQ0',
              'weighting': 'game_equal', 'tom': tom.to_record(), 'source_model_digest': '5'*64,
              'source_evidence_digest': '6'*64, 'selection_rule_digest': '7'*64,
              'inference': {'root': probability, 'wolf_nodes': {}, 'state_nodes': {}}}
    payload = canonical_json_bytes(record)
    manifest = {'artifact_type': TYPE, 'schema_version': MODEL_VERSION, 'scope': 'controlled_fixture',
                'inference_digest': digest(record),
                'file_table': {'inference.json': {'byte_size': len(payload), 'sha256': digest(record)}}}
    manifest['manifest_digest'] = digest(manifest)
    estimator = FrozenEventEstimator(payload, manifest['manifest_digest'], canonical_json_bytes(manifest))
    return request, prediction, estimator


class Consumer:
    # Deliberately seal-shaped strings: these still cannot confer sealed scope.
    provenance = cf.ConsumerProvenance(cf.CONSUMER_SCHEMA_VERSION, cf.CONSUMER_IMPLEMENTATION_VERSION,
        'a'*64, 'b'*64, 'c'*64, 'explicit_day_phase', 'd'*64, cf.BUILDER_VERSION)

    def __init__(self, fail_at=None, corrupt=False, mutate_source=False):
        self.calls = []
        self.fail_at, self.corrupt, self.mutate_source = fail_at, corrupt, mutate_source

    def probabilities(self, parent, candidate):
        self.calls.append((parent.prefix_digest, candidate))
        if len(self.calls) == self.fail_at:
            raise RuntimeError('controlled batch failure')
        if self.mutate_source:
            self.provenance = replace(self.provenance, checkpoint_digest='e'*64)
        q = torch.full((7, 7), 1/6, dtype=torch.float64)
        q.fill_diagonal_(0)
        if self.corrupt:
            q[2, 0] = float('nan')
        return q


def run(args, **kwargs):
    return analyze_pre(*args, expected_deployment_digest=args[2].manifest_digest,
                       tie_policy=kwargs.pop('tie_policy', POLICY), **kwargs)


@pytest.mark.parametrize('phase', ['discussion', 'pk_discussion'])
@pytest.mark.parametrize('dead', [(), (P[1], P[6])])
@pytest.mark.parametrize('policy', [POLICY, TiePolicy.DEFER_FIRST])
def test_full_population_matrix_features_costs_and_independent_sources(phase, dead, policy):
    args = inputs(phase=phase, dead=dead)
    consumer = Consumer()
    record = run(args, tie_policy=policy,
                 proxy=ControlledB2(consumer, ExperimentCapacity(256), 'artificial-b2')).record
    req, prediction, _ = args
    assert record['cost_source'] == {'kind': 'FROZEN_D3_REGISTERED_45',
        'protocol_digest': D3_PROTOCOL_DIGEST, 'tie_policy': policy.value}
    assert [r['context']['observer_id'] for r in record['observer_states']] == [
        p for p in req.prefix.alive_observer_ids if p not in req.known_wolf_ids]
    assert len(record['observer_states']) == (4 if dead else 5)
    row = next(r for r in record['observer_states'] if r['context']['observer_id'] == P[2])
    assert row['features'] == observer_features(req.contexts[0], prediction.matrix).to_record()
    assert row['features']['m'] == (0 if dead else .2)
    assert row['features']['dW'] == (None if dead else .1)
    assert row['context']['step_index'] == req.prefix.step_index
    assert record['current_Q'][2][1] == .2 and record['current_Q'][2][6] == .3
    assert len(row['decisions']) == 45
    protocol = load_protocol()
    for i, entry in enumerate(row['decisions']):
        assert entry['scenario_id'] == f'C{i:02}'
        assert entry['decision'] == decide(.5, protocol.loss(i, policy)).to_record()
    plans = generate_candidates(req.prefix)
    assert record['candidates'] == [p.public_payload() for p in plans]
    assert len(consumer.calls) == len(plans)
    assert record['proxy_batch']['source']['scope'] == 'CONTROLLED_TEST_ONLY_NOT_SEALED_B2'
    assert record['current_source']['tom'] == prediction.tom.to_record()
    assert all(r['evaluation']['delta'] is None for r in record['proxy_batch']['rows'])
    assert not any(k in record for k in ('selected_plan', 'reward', 'action'))


@pytest.mark.parametrize('phase', ['discussion', 'pk_discussion'])
def test_last_speaker_keeps_every_legal_plan_and_current_cognition_without_proxy_calls(phase):
    args = inputs(phase=phase, last=True)
    consumer = Consumer()
    record = run(args, proxy=ControlledB2(consumer, ExperimentCapacity(256), 'fixture')).record
    plans = generate_candidates(args[0].prefix)
    assert len(record['candidates']) == len(plans) > 2
    assert len(record['proxy_batch']['rows']) == len(plans)
    assert record['proxy_batch']['status'] == 'UNSUPPORTED_LAST_SAME_PHASE_SPEAKER'
    assert consumer.calls == [] and len(record['observer_states']) == 5
    assert all(r['evaluation'] is None for r in record['proxy_batch']['rows'])


def test_capacity_and_missing_proxy_are_explicit_without_scores_or_calls():
    args = inputs()
    consumer = Consumer()
    count = plan_structured_history(args[0].prefix).token_count
    record = run(args, proxy=ControlledB2(consumer, ExperimentCapacity(count+2), 'too-small')).record
    assert record['proxy_batch']['status'] == 'UNSUPPORTED_COMPLETE_SEQUENCE_CAPACITY'
    assert consumer.calls == []
    assert run(args).record['proxy_batch']['status'] == 'PROXY_SOURCE_UNAVAILABLE'


def test_proxy_values_match_original_batch_exactly_and_no_cross_source_delta():
    args = inputs(phase='pk_discussion')
    req = args[0]
    candidates = generate_candidates(req.prefix)
    original = evaluate_candidates(req.prefix, candidates, Consumer(), alive_wolves=req.contexts[0].alive_wolf_ids)
    current = run(args, proxy=ControlledB2(Consumer(), ExperimentCapacity(256), 'oracle')).record
    for expected, got in zip(original, current['proxy_batch']['rows'], strict=True):
        assert got['plan'] == expected.plan.public_payload()
        assert got['evaluation']['suspicion']['raw_team_mass'] == expected.suspicion.raw_team_mass
        assert got['evaluation']['suspicion']['alive_conditional_team_mass'] == expected.primary_score
        assert got['evaluation']['current_suspicion'] is None and got['evaluation']['delta'] is None


@pytest.mark.parametrize('kind', ['exception', 'bad_Q', 'source_changed'])
def test_batch_failure_never_publishes_partial_results_or_retries(kind):
    consumer = Consumer(fail_at=2 if kind == 'exception' else None,
                        corrupt=kind == 'bad_Q', mutate_source=kind == 'source_changed')
    record = run(inputs(), proxy=ControlledB2(consumer, ExperimentCapacity(256), 'failure')).record
    assert record['proxy_batch']['status'] == 'BATCH_FAILED_NO_RESULTS'
    assert all(r['evaluation'] is None for r in record['proxy_batch']['rows'])
    assert record['proxy_batch']['error_type'] in ('RuntimeError', 'ValueError')
    assert len(consumer.calls) == (2 if kind == 'exception' else 1 if kind == 'bad_Q' else len(record['candidates']))


@pytest.mark.parametrize('kind', ['Q_pre', 'plan', 'tom', 'deployment'])
def test_current_source_conflicts_fail_before_proxy(kind):
    req, prediction, model = inputs()
    if kind == 'Q_pre':
        prediction = replace(prediction, matrix=replace(prediction.matrix, boundary_id='different'))
    elif kind == 'plan':
        prediction = replace(prediction, plan_digest='0'*64)
    elif kind == 'tom':
        prediction = replace(prediction, tom=replace(prediction.tom, weights_digest='9'*64))
    consumer = Consumer()
    with pytest.raises(ValueError, match='source mismatch'):
        analyze_pre(req, prediction, model, expected_deployment_digest='0'*64 if kind == 'deployment' else model.manifest_digest,
                    tie_policy=POLICY, proxy=ControlledB2(consumer, ExperimentCapacity(256), 'fixture'))
    assert consumer.calls == []


def test_arbitrary_cost_source_losses_and_unregistered_tie_cannot_claim_registry():
    args = inputs()
    for extra in ({'cost_source': 'FROZEN_D3'}, {'losses': []}):
        with pytest.raises(TypeError):
            run(args, **extra)
    for policy in (POLICY.value, None, TiePolicy.TERMINAL_POS_FIRST):
        with pytest.raises(ValueError, match='registered D3'):
            run(args, tie_policy=policy)


def test_raw_or_subclass_fixture_cannot_impersonate_sealed_b2():
    args = inputs()
    for fake in (Consumer(), object()):
        with pytest.raises(TypeError, match='ControlledB2'):
            run(args, proxy=fake, expected_proxy_provenance=Consumer.provenance)
    class FakeSealed(cf.CounterfactualToMConsumer):
        def __init__(self):
            self.provenance = Consumer.provenance
    with pytest.raises(TypeError):
        run(args, proxy=FakeSealed(), expected_proxy_provenance=Consumer.provenance)
    with pytest.raises(ValueError, match='cannot claim'):
        run(args, proxy=ControlledB2(Consumer(), ExperimentCapacity(256), 'fixture'),
            expected_proxy_provenance=Consumer.provenance)


def test_exact_d3_ties_retain_all_bayes_optima_and_explicit_policy():
    # Scenario C22 has alpha=3/4 and beta=1/4 exactly.
    for p in (.25, .75):
        args = inputs(probability=p)
        terminal = run(args).record['observer_states'][0]['decisions'][22]['decision']
        defer = run(args, tie_policy=TiePolicy.DEFER_FIRST).record['observer_states'][0]['decisions'][22]['decision']
        assert len(terminal['minimizing_actions']) == 2
        assert terminal['minimizing_actions'] == defer['minimizing_actions']
        assert terminal['region'] != defer['region'] and defer['region'] == 'BND'


def test_repeatability_no_mutation_io_rng_actor_parser_or_supervision(monkeypatch, tmp_path):
    args = inputs()
    pre_bytes = canonical_json_bytes(args[0].prefix.to_record())
    q_bytes = canonical_json_bytes(args[1].matrix.values)
    import scripts.verified_speech_commit as commits
    import scripts.speech_realization as language
    py = random.getstate()
    numpy = np.random.get_state()
    torch_rng = torch.get_rng_state().clone()
    def forbidden(*a, **k):
        pytest.fail('research inspection attempted speech/commit/supervision')
    monkeypatch.setattr(commits, 'realize_and_commit', forbidden)
    monkeypatch.setattr(language, 'realize_parse_verify', forbidden)
    # Reject output files even if an implementation starts writing unexpectedly.
    from pathlib import Path
    original_open = Path.open
    def readonly_open(path, mode='r', *a, **k):
        assert not any(c in mode for c in 'wax+')
        from werewolf.observer_dtrs.d3_protocol import CONFIG_PATH
        assert path.resolve() == CONFIG_PATH.resolve(), 'unexpected research input read'
        return original_open(path, mode, *a, **k)
    monkeypatch.setattr(Path, 'open', readonly_open)
    monkeypatch.chdir(tmp_path)
    first = run(args, proxy=ControlledB2(Consumer(), ExperimentCapacity(256), 'fixture'))
    assert first == run(args, proxy=ControlledB2(Consumer(), ExperimentCapacity(256), 'fixture'))
    assert canonical_json_bytes(args[0].prefix.to_record()) == pre_bytes
    assert canonical_json_bytes(args[1].matrix.values) == q_bytes
    assert random.getstate() == py
    now = np.random.get_state()
    assert now[0] == numpy[0] and np.array_equal(now[1], numpy[1]) and now[2:] == numpy[2:]
    assert torch.equal(torch.get_rng_state(), torch_rng)
    assert list(tmp_path.iterdir()) == []
    # Returned dictionaries are fresh; no mutable cache or hidden game state.
    changed = first.record
    changed['observer_states'].clear()
    assert first.record['observer_states']
    from werewolf.observer_dtrs import planning_research
    source = inspect.getsource(planning_research)
    assert not any(s in source for s in ('open_supervision', 'realize_and_commit', 'select_minimum_suspicion', 'env.step'))


def test_supervision_is_not_an_input_and_q_change_does_not_relabel_proxy_source():
    args = inputs()
    for field in ('Y', 'supervision', 'role_sidecar', 'future_events', 'action_values'):
        with pytest.raises(TypeError):
            run(args, **{field: object()})
    altered = args[1].matrix.values
    altered = (*altered[:2], (.2, .1, 0., .1, .1, .2, .3), *altered[3:])
    other = (args[0], replace(args[1], matrix=replace(args[1].matrix, values=altered)), args[2])
    a = run(args, proxy=ControlledB2(Consumer(), ExperimentCapacity(256), 'fixture')).record
    b = run(other, proxy=ControlledB2(Consumer(), ExperimentCapacity(256), 'fixture')).record
    assert a['current_Q_digest'] != b['current_Q_digest']
    assert a['proxy_batch'] == b['proxy_batch']
