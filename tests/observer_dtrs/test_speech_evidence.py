"""S1a pure G/pi boundary tests and unchanged verified-speech transport."""
from dataclasses import FrozenInstanceError, fields, replace
from fractions import Fraction as F
import json
import random

import numpy as np
import pytest

from scripts.speech_planning import PlanningAction as A, generate_candidates
from tests.observer_dtrs.conftest import prefix_for, speech_events
from tests.observer_dtrs.test_speech_runtime import (
    EVIDENCE, POLICY, REFERENCE, runtime, source, witness,
)
from tests.tom.test_counterfactual_consumer import PublicWitness, P
from werewolf.canonical_collection import construct_authoritative_pre_prefix, freeze_public_event_history
from werewolf.observer_dtrs.context import contexts_for_pre
from werewolf.observer_dtrs.decision import Region, TiePolicy, candidate_losses, decide, thresholds
from werewolf.observer_dtrs.features import observer_features
from werewolf.observer_dtrs.speech_evidence import (
    CognitiveHandling, ObserverCognition, SelectionMode, process_cognition, select_observer, select_speech,
)


def panel(pre, p=F(1, 2), *, loss=None):
    loss = POLICY.static_loss if loss is None else loss
    prediction = source(pre).matrix
    return tuple(ObserverCognition(c, observer_features(c, prediction), '1'*64, 'leaf', loss, decide(p, loss))
                 for c in contexts_for_pre(pre, P[:2]))


def choose(pre, evidence, *, mode=SelectionMode.EXPLICIT_THREE_WAY, strategy=EVIDENCE, candidates=None):
    return select_speech(pre, evidence, candidates=generate_candidates(pre) if candidates is None else candidates,
                         strategy=strategy, mode=mode)


def legal_population_pre(phase, dead):
    if phase == 'discussion':
        return prefix_for(dead=dead)
    events = speech_events(phase=phase, dead=dead)
    alive = tuple(p for p in P if p not in dead)
    first, second = alive[0], alive[1]
    half = len(alive)//2
    others = [p for p in alive if p not in (first, second)]
    targets = [first]*(half-1)+[second]*(half-1)+[None]*(len(alive)%2)
    votes = {first: second, second: first, **dict(zip(others, targets, strict=True))}
    # Real positive maximum tie, full living coverage, no self/dead vote.
    next(e for e in events if e['event_type'] == 'vote_result')['votes'] = [
        {'voter': p, 'target': votes[p]} for p in alive]
    return construct_authoritative_pre_prefix(game_id='s1a-artificial', boundary_id='s1a-pre',
        step_index=len(events)-2, report_trigger_id='s1a-trigger', current_speaker=P[0],
        alive_observer_ids=alive, public_event_history=freeze_public_event_history(events[:-1]),
        v1_annotations=(), belief_observation_ids_by_observer={p: f's1a-{p}' for p in alive})


@pytest.mark.parametrize('phase', ['discussion', 'pk_discussion'])
@pytest.mark.parametrize('dead', [(), (P[1],), (P[5],)])
def test_complete_panel_matches_legal_domain_and_does_not_choose_speech(phase, dead):
    pre = legal_population_pre(phase, dead)
    evidence = process_cognition(pre, panel=panel(pre))
    record = evidence.to_record()
    assert record['panel_coverage'] == 'ALL_ELIGIBLE'
    assert tuple(record['processed_observers']) == tuple(p for p in pre.alive_observer_ids if p not in P[:2])
    assert all(r.cognition.features.nW == (1 if P[1] in dead else 2) for r in evidence.rows)
    if P[1] in dead:
        assert all(r.cognition.features.dW is None for r in evidence.rows)
    assert all(r.handling is CognitiveHandling.KEEP_BOTH for r in evidence.rows)
    assert not {'plan', 'action', 'target'} & record.keys()
    with pytest.raises(FrozenInstanceError):
        evidence.selected = None


def test_subset_is_explicit_and_cannot_replace_public_selection_using_high_probability():
    pre = witness().pre()
    rows = panel(pre)
    selected = select_observer(pre, P[:2]).observer_id
    row = next(r for r in rows if r.context.observer_id == selected)
    subset = process_cognition(pre, panel=(row,))
    assert subset.to_record()['panel_coverage'] == 'EXPLICIT_SUBSET'
    assert choose(pre, subset).plan == choose(pre, process_cognition(pre, panel=rows)).plan
    modified_panel = tuple(r if r is row else replace(r, decision=decide(F(1), r.loss)) for r in rows)
    assert choose(pre, process_cognition(pre, panel=modified_panel)).plan == choose(pre, subset).plan
    others = tuple(replace(r, decision=decide(F(1), r.loss)) for r in rows if r is not row)
    with pytest.raises(ValueError, match='omits'):
        process_cognition(pre, panel=others)


@pytest.mark.parametrize('mutation', ['duplicate', 'reorder', 'other_pre', 'wrong_loss', 'wrong_model'])
def test_panel_rejects_duplicate_stale_and_mixed_sources(mutation):
    pre = witness().pre()
    rows = panel(pre)
    if mutation == 'duplicate':
        rows = rows + rows[-1:]
    elif mutation == 'reorder':
        rows = rows[::-1]
    elif mutation == 'other_pre':
        rows = panel(prefix_for(game_id='another-game'))
    elif mutation == 'wrong_model':
        rows = (replace(rows[0], model_digest='2'*64), *rows[1:])
    else:
        loss = candidate_losses(tie_policy=TiePolicy.TERMINAL_NEG_FIRST)[0]
        rows = (replace(rows[0], loss=loss, decision=decide(F(1, 2), loss)), *rows[1:])
    with pytest.raises(ValueError):
        process_cognition(pre, panel=rows)


def test_supplied_d3_risks_cannot_be_detached_from_probability_cost_or_region():
    row = panel(witness().pre())[0]
    for decision in (replace(row.decision, region=Region.POS), decide(F(1), row.loss),
                     replace(row.decision, risk_gap=F(42))):
        # A different self-consistent p is allowed; mixing that p with old risks is not.
        if decision == decide(F(1), row.loss):
            decision = replace(row.decision, probability=F(1))
        with pytest.raises(ValueError, match='binding'):
            replace(row, decision=decision)


@pytest.mark.parametrize('tie', list(TiePolicy))
def test_continuous_and_explicit_representation_agree_at_every_registered_exact_boundary(tie):
    pre = witness().pre()
    context = contexts_for_pre(pre, P[:2])[-1]
    features = observer_features(context, source(pre).matrix)
    for loss in candidate_losses(tie_policy=tie):
        th = thresholds(loss)
        for p in (F(0), th.beta, (th.beta+th.alpha)/2, th.alpha, F(1)):
            row = ObserverCognition(context, features, '1'*64, 'leaf', loss, decide(p, loss))
            evidence = process_cognition(pre, panel=(row,))
            continuous = choose(pre, evidence, mode=SelectionMode.CONTINUOUS_RISK)
            explicit = choose(pre, evidence)
            assert continuous.plan == explicit.plan
            assert continuous.cognitive_branch == explicit.cognitive_branch == row.decision.region
            assert continuous.reason == explicit.reason


def test_regions_have_no_fixed_speech_action_and_public_direction_gates_mapping():
    pre = witness(target=P[1]).pre()
    evidence = process_cognition(pre, panel=panel(pre, p=F(1)))
    assert choose(pre, evidence).plan.action is A.OPPOSE
    assert choose(pre, evidence, strategy=REFERENCE).plan.action is A.NO_COMMITMENT
    neutral = PublicWitness().pre()
    for p in (F(0), F(1, 2), F(1)):
        chosen = choose(neutral, process_cognition(neutral, panel=panel(neutral, p)))
        assert chosen.plan.action is A.NO_COMMITMENT
        assert chosen.reason == 'NO_DIRECTIONAL_PUBLIC_EVIDENCE'


def test_selection_rejects_modified_evidence_candidates_or_implicit_policy():
    pre = witness().pre()
    evidence = process_cognition(pre, panel=panel(pre))
    with pytest.raises(ValueError, match='evidence changed'):
        choose(pre, replace(evidence, selected=replace(evidence.selected, reason='invented')))
    with pytest.raises(ValueError, match='handling'):
        replace(evidence.rows[0], handling=CognitiveHandling.CONSIDER_HIT)
    candidates = generate_candidates(pre)
    for bad in (candidates[:-1], candidates[::-1], candidates+candidates[:1], list(candidates)):
        with pytest.raises(ValueError, match='candidate tuple'):
            choose(pre, evidence, candidates=bad)
    with pytest.raises(TypeError):
        choose(pre, evidence, mode='explicit_three_way')
    with pytest.raises(TypeError):
        choose(pre, evidence, strategy=None)


def test_supervision_counterfactual_is_not_an_input_and_pure_calls_preserve_rng(monkeypatch):
    pre = witness().pre()
    lawful_rows = panel(pre)
    rng, np_rng = random.getstate(), np.random.get_state()
    outputs = []
    for hidden_truth in (dict(Y=0, report=('player4',)), dict(Y=1, report=('player1',))):
        evidence = process_cognition(pre, panel=lawful_rows)
        outputs.append(choose(pre, evidence).to_record())
        with pytest.raises(TypeError):
            process_cognition(pre, panel=lawful_rows, supervision=hidden_truth)
    assert outputs[0] == outputs[1]
    assert not {'Y', 'report', 'supervision', 'sidecar', 'support'} & {f.name for f in fields(ObserverCognition)}
    assert random.getstate() == rng
    current = np.random.get_state()
    assert current[0] == np_rng[0] and np.array_equal(current[1], np_rng[1]) and current[2:] == np_rng[2:]
    def no_read(*args, **kwargs):
        raise AssertionError('pure G/pi must not read files')
    from pathlib import Path
    with monkeypatch.context() as m:
        m.setattr('builtins.open', no_read)
        m.setattr(Path, 'open', no_read)
        assert choose(pre, process_cognition(pre, panel=lawful_rows)).to_record() == outputs[0]


@pytest.mark.parametrize('phase', ['discussion', 'pk_discussion'])
def test_both_modes_keep_identical_original_actor_parser_and_commit_path(phase):
    from scripts.compare_observer_dtrs_synthetic import ControlledActor, ControlledParser, synthetic_model
    from werewolf.agents.gpt_agent import GPTAgent
    from werewolf.observer_dtrs.speech_runtime import Comparison, execute_pending_speech
    protocol, model = synthetic_model()
    outputs = []
    for mode in SelectionMode:
        env, recorder, _ = runtime(phase)
        pre = recorder._pending.prefix
        actor_backend, parser_backend = ControlledActor(), ControlledParser()
        env.speech_perceiver.backend = parser_backend
        actor = GPTAgent(backend=actor_backend, model_name='artificial-grammar')
        before = len(recorder.submitted_gameplay_actions)
        rng = env._rng.getstate()
        result = execute_pending_speech(env=env, recorder=recorder, actor=actor, source=source(pre, .5),
            model=model, protocol=protocol, condition=Comparison.CONTEXTUAL_SPEECH,
            policy=POLICY, selection_mode=mode)
        assert len(actor_backend.calls) == len(parser_backend.calls) == 1
        assert len(recorder.submitted_gameplay_actions) == before+1 and recorder._pending is None
        assert env._rng.getstate() == rng
        assert result.audit['cognitive_processing']['panel_coverage'] == 'EXPLICIT_SUBSET'
        assert result.audit['candidate_selection']['mode'] == mode.value
        assert not any(key in json.dumps(actor_backend.calls) for key in ('model_digest', 'cognitive_processing', 'risks_exact'))
        outputs.append((actor_backend.calls, actor_backend.speeches, parser_backend.responses,
                        env.public_events, env.speech_annotations[-1].to_record()))
    assert outputs[0] == outputs[1]


def test_paired_factorial_comparison_has_complete_fixed_budget_and_reports_no_difference():
    from scripts.compare_observer_dtrs_s1a import compare_scene, scenarios, summarize, synthetic_model
    protocol, model = synthetic_model()
    rows = [row for scene in scenarios() for row in compare_scene(scene, protocol, model)]
    summary = summarize(rows)
    assert summary['rows'] == 120 and summary['pairing_ok']
    assert [(c['region_flips'], c['intent_changes']) for c in summary['contrasts']] == [(0, 0), (0, 0), (3, 1), (3, 1)]
    with pytest.raises(ValueError, match='coverage'):
        summarize(rows[:-1])
    with pytest.raises(ValueError, match='coverage'):
        summarize(rows+rows[:1])
    rows[0] = {**rows[0], 'pairing': {**rows[0]['pairing'], 'model_digest': '2'*64}}
    with pytest.raises(ValueError, match='unpaired'):
        summarize(rows)


def test_controlled_cli_seals_complete_grid_and_refuses_to_overwrite(tmp_path):
    import hashlib
    from pathlib import Path
    import subprocess
    import sys
    output = tmp_path.resolve()/'selection-only'
    cmd = [sys.executable, '-B', '-m', 'scripts.compare_observer_dtrs_s1a', '--output', str(output)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['rows'] == 120
    manifest = json.loads((output/'manifest.json').read_text())
    for name, record in manifest['files'].items():
        data = (output/name).read_bytes()
        assert len(data) == record['bytes'] and hashlib.sha256(data).hexdigest() == record['sha256']
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    repeat = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
    assert repeat.returncode != 0 and 'FileExistsError' in repeat.stderr
    assert {p.name: p.read_bytes() for p in output.iterdir()} == before
    repo = Path(__file__).resolve().parents[2]
    bad = subprocess.run([*cmd[:-1], str(repo/'must-not-create-s1a-output')],
                         capture_output=True, text=True, timeout=45)
    assert bad.returncode != 0 and not (repo/'must-not-create-s1a-output').exists()
