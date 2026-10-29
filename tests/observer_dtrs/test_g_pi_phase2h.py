"""Phase-2H interface acceptance; artificial PRE/Q/tables, never a game policy."""
from dataclasses import replace
from fractions import Fraction as F
from pathlib import Path
import json
import random

import numpy as np
import pytest
import torch

from scripts.speech_planning import PlanningAction as A, SpeechPlan, generate_candidates
from tests.observer_dtrs.test_planning_research import Consumer, inputs, run
from tests.tom.test_counterfactual_consumer import P, PublicWitness
from werewolf.artifact_io import canonical_json_bytes
from werewolf.observer_dtrs.candidate_appraisal import (
    BasisKind as B, CandidateBasis, GateMode, PublicReference,
    appraise_candidates, bind_bases,
)
from werewolf.observer_dtrs.d3_protocol import CONFIG_PATH, load_protocol
from werewolf.observer_dtrs.decision import Region, TiePolicy, decide
from werewolf.observer_dtrs.features import observer_features
from werewolf.observer_dtrs.online_context import digest
from werewolf.observer_dtrs.online_model import FrozenEventEstimator
from werewolf.observer_dtrs.planning_research import ControlledB2
from werewolf.observer_dtrs.probabilities import ProbabilityMatrix
from werewolf.observer_dtrs.speech_evidence import (
    ObserverCognition, SelectionMode, SpeechStrategy, process_cognition,
    select_speech,
)
from werewolf.structured_history import plan_structured_history
from werewolf.tom.dataset import ExperimentCapacity

POLICY = TiePolicy.TERMINAL_NEG_FIRST
# The existing S1a EVIDENCE hypothesis, not a new or learned action preference.
STRATEGY = SpeechStrategy(A.SELF_DEFEND, A.OPPOSE, A.OPPOSE, A.NO_COMMITMENT)


def declared_table(model):
    """Hand-declared EQ1 table with the old 30-scene p values; no fitting/Y."""
    record = model.record
    record['model_id'] = 'OBS-EQ1'
    record['inference'] = {'root': .5, 'wolf_nodes': {'1': .5, '2': .5},
        'state_nodes': {f'{phase}/{n}': {'probability': .5, 'edges': [.2, .4, .6, .8],
            'leaves': [.125, .25, .5, .75, .875]}
            for phase in ('discussion', 'pk_discussion') for n in (1, 2)}}
    payload = canonical_json_bytes(record)
    manifest = json.loads(model.manifest_payload)
    manifest['inference_digest'] = digest(record)
    manifest['file_table'] = {'inference.json': {'byte_size': len(payload), 'sha256': digest(record)}}
    manifest['manifest_digest'] = digest({k: v for k, v in manifest.items() if k != 'manifest_digest'})
    return FrozenEventEstimator(payload, manifest['manifest_digest'], canonical_json_bytes(manifest))


def directional_witness(phase, target):
    """Reuse the existing public witness and its scripted V1 setup, no env."""
    witness = PublicWitness(phase)
    if target is not None:
        previous = witness.events[-2]['speaker']
        witness.events = witness.events[:-2]
        witness.annotations.pop()
        witness.speech(previous, 'point_as_werewolf', target, f'我认为{target}是狼人。')
        witness.event('turn_start', speaker=P[0])
    return witness


def bound_inputs(pre, r):
    """Only an artificial seven-seat probability table, not a model provider."""
    req, prediction, model = inputs()
    rows = []
    for observer in P:
        if observer in P[:2]:
            rows.append(tuple(0. if player == observer else 1/6 for player in P))
        else:
            row = [r/2, r/2, 0., 0., 0., 0., 0.]
            row[P.index(next(player for player in P[2:] if player != observer))] = 1-r
            rows.append(tuple(row))
    q = ProbabilityMatrix(pre.game_id, pre.boundary_id, pre.prefix_digest, tuple(rows))
    return (replace(req, prefix=pre),
            replace(prediction, matrix=q, plan_digest=plan_structured_history(pre).plan_digest),
            declared_table(model))


def panel_from_snapshot(args, snapshot, *, scenario=22):
    """Test-only seam check: bind original typed values to EVERY snapshot row.

    No runtime adapter or alternative feature/decision implementation is added.
    Reconstruct with the original pure functions and require full record equality.
    """
    record = snapshot.record
    req, prediction, estimator = args
    protocol = load_protocol()
    policy = TiePolicy(record['cost_source']['tie_policy'])
    loss = protocol.loss(scenario, policy)
    assert canonical_json_bytes(record['request']) == canonical_json_bytes(req.identity)
    assert record['current_source']['tom'] == prediction.tom.to_record() == estimator.tom.to_record()
    assert record['current_source']['deployment_manifest_digest'] == estimator.manifest_digest
    assert record['current_source']['estimator_digest'] == estimator.digest
    assert record['current_Q'] == [list(row) for row in prediction.matrix.values]
    assert len(record['observer_states']) == len(req.contexts)
    panel = []
    for context, row in zip(req.contexts, record['observer_states'], strict=True):
        features = observer_features(context, prediction.matrix)
        decision = decide(row['estimate']['probability'], loss)
        assert canonical_json_bytes(row['context']) == canonical_json_bytes(context.to_record())
        assert row['features'] == features.to_record()
        assert row['decisions'][scenario] == {'scenario_id': f'C{scenario:02}',
            'loss': loss.to_record(), 'decision': decision.to_record()}
        panel.append(ObserverCognition(context, features, estimator.digest,
            row['estimate']['level'], loss, decision))
    return tuple(panel)


@pytest.mark.parametrize('phase', ['discussion', 'pk_discussion'])
@pytest.mark.parametrize('dead', [(), (P[1], P[6])])
@pytest.mark.parametrize('tie', [POLICY, TiePolicy.DEFER_FIRST])
def test_full_snapshot_to_G_and_complete_pi_without_action_values(phase, dead, tie):
    args = inputs(phase=phase, dead=dead)
    snapshot = run(args, tie_policy=tie)
    record = snapshot.record
    evidence = process_cognition(args[0].prefix, panel=panel_from_snapshot(args, snapshot))
    assert evidence.to_record()['panel_coverage'] == 'ALL_ELIGIBLE'
    assert len(evidence.rows) == (4 if dead else 5)
    assert record['current_Q'][2][1] == .2 and record['current_Q'][2][6] == .3
    candidates = generate_candidates(args[0].prefix)
    catalog = bind_bases(args[0].prefix, candidates=candidates, bases=tuple(
        CandidateBasis(f'public-{i}', plan, B.PUBLIC_ONLY) for i, plan in enumerate(candidates)))
    for mode in GateMode:
        appraisal = appraise_candidates(args[0].prefix, evidence, catalog, mode=mode)
        assert appraisal.status == 'ACTION_VALUE_UNAVAILABLE' and appraisal.selected_plan is None
        assert appraisal.audit['legal_candidate_count'] == len(candidates)
        assert appraisal.audit['assessable_candidate_count'] == len(candidates)
    # Deployment-only lookup level is not Kish/game support or a BND confidence flag.
    assert all(set(row['estimate']) == {'probability', 'level'} for row in record['observer_states'])
    assert all(row.cognition.estimator_level == 'global' for row in evidence.rows)


@pytest.mark.parametrize('case', ['absent', 'last', 'capacity', 'batch_failure', 'evaluated'])
def test_proxy_status_never_becomes_action_value_or_deletes_a_candidate(case):
    args = inputs(last=case == 'last')
    consumer = Consumer(fail_at=2 if case == 'batch_failure' else None)
    capacity = (plan_structured_history(args[0].prefix).token_count+2 if case == 'capacity' else 256)
    kwargs = {} if case == 'absent' else {'proxy': ControlledB2(consumer, ExperimentCapacity(capacity), 'p2h-test')}
    snapshot = run(args, **kwargs)
    record = snapshot.record
    expected = {'absent': 'PROXY_SOURCE_UNAVAILABLE', 'last': 'UNSUPPORTED_LAST_SAME_PHASE_SPEAKER',
        'capacity': 'UNSUPPORTED_COMPLETE_SEQUENCE_CAPACITY', 'batch_failure': 'BATCH_FAILED_NO_RESULTS',
        'evaluated': 'EVALUATED'}[case]
    assert record['proxy_batch']['status'] == expected
    candidates = generate_candidates(args[0].prefix)
    assert record['candidates'] == [p.public_payload() for p in candidates]
    assert len(record['proxy_batch']['rows']) == len(candidates)
    if case != 'evaluated':
        assert all(row['evaluation'] is None for row in record['proxy_batch']['rows'])
    evidence = process_cognition(args[0].prefix, panel=panel_from_snapshot(args, snapshot))
    catalog = bind_bases(args[0].prefix, candidates=candidates, bases=tuple(
        CandidateBasis(f'p{i}', plan, B.PUBLIC_ONLY) for i, plan in enumerate(candidates)))
    result = appraise_candidates(args[0].prefix, evidence, catalog, mode=GateMode.THREE_WAY)
    assert result.status == 'ACTION_VALUE_UNAVAILABLE' and result.selected_plan is None
    assert record['interpretation'] == 'B2_SUSPICION_PROXY_NOT_EVENT_PROBABILITY_OR_SPEECH_UTILITY'
    assert not {'selected_plan', 'action_value', 'reward'} & record.keys()
    assert all(row['evaluation'] is None or row['evaluation']['delta'] is None
               for row in record['proxy_batch']['rows'])


def test_successive_legal_wolf_PRE_rebinds_Q_p_G_and_candidates_no_cache():
    witness = directional_witness('discussion', P[0])
    first = witness.pre()
    witness.speech(P[0], 'no_commitment', None, '本轮暂不表态。')
    witness.event('turn_start', speaker=P[1])
    second = witness.pre()
    a = bound_inputs(first, .1)
    b = bound_inputs(second, .9)
    assert a[2] == b[2]  # The SAME hand-declared estimator bytes, not a second fit.
    first_snapshot, second_snapshot = run(a), run(b)
    ga = process_cognition(first, panel=panel_from_snapshot(a, first_snapshot))
    gb = process_cognition(second, panel=panel_from_snapshot(b, second_snapshot))
    assert first.game_id == second.game_id and first.step_index < second.step_index
    assert first.prefix_digest != second.prefix_digest and first.boundary_id != second.boundary_id
    assert {r.cognition.decision.probability for r in ga.rows} == {F(1, 8)}
    assert {r.cognition.decision.probability for r in gb.rows} == {F(7, 8)}
    assert {r.cognition.decision.region for r in ga.rows} == {Region.NEG}
    assert {r.cognition.decision.region for r in gb.rows} == {Region.POS}
    with pytest.raises(ValueError, match='source mismatch'):
        run((b[0], a[1], b[2]))
    with pytest.raises(ValueError, match='PRE/context'):
        process_cognition(second, panel=tuple(r.cognition for r in ga.rows))
    old_catalog = bind_bases(first, candidates=generate_candidates(first), bases=())
    with pytest.raises(ValueError, match='complete canonical legal candidate'):
        appraise_candidates(second, gb, old_catalog, mode=GateMode.THREE_WAY)
    current_catalog = bind_bases(second, candidates=generate_candidates(second), bases=())
    stale_identity = replace(current_catalog, boundary_id=first.boundary_id,
                             prefix_digest=first.prefix_digest)
    with pytest.raises(ValueError, match='another PRE'):
        appraise_candidates(second, gb, stale_identity, mode=GateMode.THREE_WAY)
    assert second_snapshot == run(b)


def test_exact_tie_keeps_all_optima_but_P0_selected_gate_is_an_extra_assumption():
    args = inputs(probability=.25)
    observations = []
    for tie in (POLICY, TiePolicy.DEFER_FIRST):
        snapshot = run(args, tie_policy=tie)
        evidence = process_cognition(args[0].prefix, panel=panel_from_snapshot(args, snapshot))
        cognition = evidence.rows[0].cognition
        assert set(cognition.decision.minimizing_actions) == {Region.NEG, Region.BND}
        candidates = generate_candidates(args[0].prefix)
        catalog = bind_bases(args[0].prefix, candidates=candidates, bases=(
            CandidateBasis('both', candidates[0], B.KEEP_BOTH, cognition.context.observer_id),))
        result = appraise_candidates(args[0].prefix, evidence, catalog, mode=GateMode.THREE_WAY)
        review = result.audit['candidates'][0]['basis_reviews'][0]
        assert review['cognitive_risk']['excess_exact'] == '0'
        observations.append((result.status, review['admitted']))
    assert observations == [('NO_ADMISSIBLE_BASIS', False), ('ACTION_VALUE_UNAVAILABLE', True)]


def test_public_reference_binds_origin_but_does_not_prove_candidate_claim():
    pre = directional_witness('discussion', P[0]).pre()
    args = bound_inputs(pre, .9)
    evidence = process_cognition(pre, panel=panel_from_snapshot(args, run(args)))
    candidates = generate_candidates(pre)
    annotation = pre.v1_annotations[-1]
    assert annotation.actions[0].object == P[0]
    plan = SpeechPlan(A.ACCUSE_WOLF, P[3])
    basis = CandidateBasis('origin-only', plan, B.PUBLIC_ONLY,
        public_references=(PublicReference(annotation.event_id, annotation.annotation_digest),))
    catalog = bind_bases(pre, candidates=candidates, bases=(basis,))
    result = appraise_candidates(pre, evidence, catalog, mode=GateMode.THREE_WAY)
    entry = result.audit['candidates'][candidates.index(plan)]['basis_reviews'][0]
    assert entry['admitted'] and entry['reason'] == 'PUBLIC_ONLY_NO_COGNITIVE_GATE'
    assert result.status == 'ACTION_VALUE_UNAVAILABLE'
    assert catalog.to_record()['meaning'].startswith('predeclared_argument_schema_not_verified')


@pytest.mark.parametrize('family', ['d2_oof', 'final_tom'])
def test_snapshot_cannot_alias_fixture_Q_as_OOF_or_Final(family):
    args = inputs()
    tom = replace(args[1].tom, family=family, source_revision='f'*40,
                  fold_index=0 if family == 'd2_oof' else None)
    with pytest.raises(ValueError, match='source mismatch'):
        run((args[0], replace(args[1], tom=tom), args[2]))


def test_pure_cross_interface_path_no_actor_parser_model_env_or_output(monkeypatch, tmp_path):
    import scripts.speech_realization as language
    import scripts.verified_speech_commit as commit
    from werewolf.speech.speech_perceiver import SpeechPerceiver
    args = bound_inputs(directional_witness('discussion', P[0]).pre(), .5)
    before = canonical_json_bytes(args[0].prefix.to_record())
    py, nps, tr = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    def forbidden(*a, **k):
        pytest.fail('pure research path invoked execution, model, parser or unapproved I/O')
    original_open = Path.open
    def readonly(path, mode='r', *a, **k):
        assert path.resolve() == CONFIG_PATH.resolve() and mode in ('r', 'rb')
        return original_open(path, mode, *a, **k)
    with monkeypatch.context() as m:
        m.setattr(language, 'realize_parse_verify', forbidden)
        m.setattr(commit, 'realize_and_commit', forbidden)
        m.setattr(SpeechPerceiver, 'parse_with_audit', forbidden)
        m.setattr(torch.nn.Module, '__call__', forbidden)
        m.setattr('builtins.open', forbidden)
        m.setattr(Path, 'open', readonly)
        m.chdir(tmp_path)
        snapshot = run(args)
        evidence = process_cognition(args[0].prefix, panel=panel_from_snapshot(args, snapshot))
        candidates = generate_candidates(args[0].prefix)
        results = [select_speech(args[0].prefix, evidence, candidates=candidates,
                    strategy=STRATEGY, mode=mode) for mode in SelectionMode]
        assert results[0].plan == results[1].plan
        assert snapshot == run(args)
    assert list(tmp_path.iterdir()) == [] and random.getstate() == py
    current = np.random.get_state()
    assert current[0] == nps[0] and np.array_equal(current[1], nps[1]) and current[2:] == nps[2:]
    assert torch.equal(torch.get_rng_state(), tr)
    assert canonical_json_bytes(args[0].prefix.to_record()) == before
