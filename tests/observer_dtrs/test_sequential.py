"""序贯实验只用合成 PRE、Q、报告与整局抽样验证。"""

from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import analyze_observer_dtrs_sequential as seq
from tests.observer_dtrs.conftest import observation_for, prefix_for
from tests.tom.test_counterfactual_consumer import P, PublicWitness
from werewolf.canonical_collection import freeze_public_event_history
from werewolf.canonical_collection.trajectory_evidence import BeliefObservationStatus
from werewolf.observer_dtrs.context import contexts_for_pre
from werewolf.observer_dtrs.scoring import score_predictions


TEAM = P[:2]
ATTEMPT = 'controlled-attempt'


def trajectory(*, ending=None):
    witness = PublicWitness()
    prefixes = []

    def speak():
        pre = witness.pre()
        witness.speech(pre.current_speaker, 'no_commitment', None, '本轮暂不作明确表态。')
        prefixes.append(pre)

    speak()
    witness.event('turn_start', speaker=P[1])
    speak()
    witness.event('phase_change', day=1, phase='vote')
    witness.event('vote_result', votes=[{'voter': p, 'target': t} for p, t in zip(
        P, (P[1], P[0], P[0], P[0], P[1], P[1], None), strict=True)])
    witness.event('exile_result', exiled_players=[])
    witness.event('phase_change', day=1, phase='pk_discussion')
    for speaker in TEAM:
        witness.event('turn_start', speaker=speaker)
        speak()
    witness.event('phase_change', day=1, phase='pk_vote')
    witness.event('vote_result', votes=[{'voter': p, 'target': P[1] if p != P[1] else None} for p in P])
    witness.event('exile_result', exiled_players=[P[1]])
    witness.event('phase_change', day=1, phase='night')
    witness.event('death_announcement', dead_players=[P[0], P[2]] if ending == 'terminal' else [])
    if ending != 'terminal':
        witness.alive = tuple(p for p in P if p != P[1])
        witness.event('phase_change', day=2, phase='discussion')
        witness.event('turn_start', speaker=P[0])
        speak()
    game = SimpleNamespace(game_id=prefixes[0].game_id, bundle_digest='bundle',
                           authoritative_pre_prefixes=tuple(reversed(prefixes)),
                           public_event_stream=freeze_public_event_history(witness.events))
    return game, prefixes


def pairs_for(game):
    boundaries, pairs = seq.pair_game(game, TEAM, publication_digest='publication', attempt_id=ATTEMPT)
    assert [pre.step_index for pre, _ in boundaries] == sorted(pre.step_index for pre, _ in boundaries)
    return pairs


def test_authoritative_pairing_skips_teammate_but_not_missing_predictions():
    game, prefixes = trajectory()
    pairs = pairs_for(game)
    same = [p for p in pairs if (p.start.acting_wolf, p.start.observer_id) == (P[0], P[2])]
    assert [row.start.step_index for row in same] == [prefixes[j].step_index for j in (0, 2, 4)]
    assert same[0].followup == same[1].start and same[1].followup == same[2].start
    assert same[2].followup is None and same[2].terminal is None
    assert [(p.start.day, p.start.phase, p.start.wolf_count) for p in same] == [
        (1, 'discussion', 2), (1, 'pk_discussion', 2), (2, 'discussion', 1)]
    assert all(p.sequence_id == ('publication', game.game_id, ATTEMPT, 'bundle', P[0], P[2]) for p in same)
    latest_teammate = [p for p in pairs if p.start.acting_wolf == P[1] and p.followup is None]
    assert latest_teammate and all(p.wolf_death is not None and p.terminal is None for p in latest_teammate)

    ids = [seq.sample_id(p.start) for p in same]
    report = seq.summarize(same, {ids[0]: .5, ids[2]: .9}, dict.fromkeys(ids, (1, 'valid')), replicates=4)
    bnd = report['by_initial_region']['BND']
    assert [bnd['population'][name]['rows'] for name in ('I', 'O', 'P', 'V')] == [1, 1, 0, 0]
    assert bnd['h1']['delta_pred']['na_reason'] == 'no_support'
    assert report['start_prediction_invalid']['rows'] == 1
    assert report['by_initial_region']['POS']['no_next']['by_death_and_terminal'][
        'no_recorded_death/terminal_unknown']['rows'] == 1


def test_death_and_confirmed_terminal_are_simultaneous_facts():
    game, _ = trajectory(ending='terminal')
    last = [p for p in pairs_for(game) if p.followup is None]
    simultaneous = next(p for p in last if p.start.acting_wolf == P[0] and p.start.observer_id == P[2])
    assert simultaneous.wolf_death == simultaneous.observer_death == simultaneous.terminal
    probabilities = {seq.sample_id(p.start): .5 for p in last}
    labels = dict.fromkeys(probabilities, (0, 'valid'))
    bnd = seq.summarize(last, probabilities, labels, replicates=3)['by_initial_region']['BND']
    assert bnd['no_next']['by_death_and_terminal']['both_dead/terminal_confirmed']['rows'] == 2
    assert not bnd['next_opportunity_strata']['step_gap']


@pytest.fixture
def static_pre():
    pre = prefix_for()
    contexts = contexts_for_pre(pre, TEAM)
    observations = {f'report-{ctx.observer_id}': observation_for(ctx) for ctx in contexts}
    q = {'game_id': pre.game_id, 'boundary_id': pre.boundary_id, 'step_index': pre.step_index,
         'prefix_digest': pre.prefix_digest, 'q': [[0 if p == i else 1/6 for p in P] for i in P]}
    return pre, contexts, observations, q


def test_raw_report_binding_distinguishes_empty_failed_missing_and_wrong_attempt(static_pre):
    pre, contexts, observations, q = static_pre
    samples, labels, status = seq.join_pre(pre, contexts, TEAM, observations, q, ATTEMPT)
    assert status == 'valid' and all(r.successful_empty and r.y == 0 for r in samples)
    observations['report-player4'] = observation_for(contexts[1], status=BeliefObservationStatus.ERROR)
    del observations['report-player5']
    observations['report-player6'] = replace(observations['report-player6'], attempt_id='wrong')
    features, labels, status = seq.join_pre(pre, contexts, TEAM, observations, q, ATTEMPT)
    assert status == 'valid' and len(features) == len(contexts)
    assert [label for label in labels.values()] == [(0, 'valid'), (None, 'error'), (None, 'missing'),
                                                    (None, 'misbound_or_invalid'), (0, 'valid')]
    assert seq.join_pre(pre, contexts, TEAM, observations, None, ATTEMPT) == ((), labels, 'missing')


def test_h1_h2_weights_bootstrap_and_directional_errors(monkeypatch):
    base = contexts_for_pre(prefix_for(), TEAM)[0]
    pairs, probabilities, labels = [], {}, {}
    for game, values in (('long', ((.5, 0, 2), (.5, 1, 3), (.9, 1, None), (.1, 1, None))),
                         ('short', ((.5, 1, 1), (.5, 0, None)))):
        contexts = [replace(base, game_id=game, step_index=i, boundary_id=str(i)) for i in range(len(values))]
        for context, (p, y, followup) in zip(contexts, values, strict=True):
            pairs.append(seq.Pair('publication', ATTEMPT, game, context, contexts[followup] if followup is not None else None))
            probabilities[seq.sample_id(context)] = p
            labels[seq.sample_id(context)] = (y, 'valid')

    def draws(games, repeats, seed):
        assert (games, repeats, seed) == (['long', 'short'], 3, seq.fit.BOOTSTRAP_SEED)
        return np.asarray([[0, 0], [1, 1], [0, 1]])

    monkeypatch.setattr(seq, 'bootstrap_indices', draws)
    bnd = seq.summarize(pairs, probabilities, labels, replicates=3)['by_initial_region']['BND']
    assert bnd['population']['I']['rows'] == 4
    assert bnd['population']['V']['rows'] == 3 and bnd['population']['V']['games'] == 2
    assert bnd['h1']['stale_brier']['value'] == .25
    assert bnd['h1']['updated_brier']['value'] == pytest.approx(.33)
    assert bnd['h1']['delta_pred']['value'] == pytest.approx(-.08)
    assert bnd['h1']['delta_pred']['bootstrap']['percentile_95'] == pytest.approx([-.156, -.004])
    assert bnd['resolution']['occurrence_given_I']['value'] == pytest.approx(1/3)
    h2 = bnd['h2']
    assert h2['directions']['POS']['error']['value'] == 0
    assert h2['directions']['NEG']['error']['value'] == 1
    assert h2['q_given_V']['value'] == h2['e_given_decided']['value'] == .5
    assert h2['delta_dec']['value'] == h2['identity_value'] == -.0625
    assert h2['losses']['updated_binary']['value'] == .125  # .5 并列按原规则取 NEG，且该后续 y=0。
    assert bnd['label_transition_given_V']['1/0']['rows'] == 1
    assert bnd['state_transition_given_P']['BND']['value'] == .5


def test_no_decision_no_label_and_insufficient_bootstrap():
    first = contexts_for_pre(prefix_for(), TEAM)[0]
    later = replace(first, boundary_id='next', step_index=first.step_index + 2)
    other = replace(first, game_id='other')
    pairs = [seq.Pair('p', 'a', 'b', first, later), seq.Pair('p', 'a', 'b', later, None),
             seq.Pair('p', 'a2', 'b2', other, None)]
    p = dict.fromkeys([seq.sample_id(pair.start) for pair in pairs], .5)
    labels = dict.fromkeys(p, (0, 'valid'))
    bnd = seq.summarize(pairs, p, labels, replicates=100)['by_initial_region']['BND']
    assert bnd['h2']['q_given_V']['value'] == 0 and bnd['h2']['e_given_decided']['value'] is None
    assert bnd['h2']['delta_dec']['value'] == bnd['h2']['identity_value'] == 0
    assert bnd['h2']['delta_dec']['bootstrap']['na_reason'] == 'insufficient_valid_replicates'

    labels[seq.sample_id(later)] = (None, 'missing')
    bnd = seq.summarize(pairs, p, labels, replicates=4)['by_initial_region']['BND']
    assert [bnd['population'][name]['rows'] for name in ('I', 'O', 'P', 'V')] == [2, 1, 1, 0]
    assert bnd['state_transition_given_P']['BND']['value'] == 1
    assert bnd['h2']['delta_dec']['value'] is None and bnd['h2']['identity_value'] is None


def test_entry_reproduces_static_scores_before_sequential_estimates(monkeypatch, capsys, static_pre):
    pre, contexts, observations, q = static_pre
    fit, study = seq.fit, seq.study
    rows = list(fit.pre_samples(pre, TEAM, observations, q))
    population = fit.population(rows)
    monkeypatch.setattr(study, 'FOLD4', {'games': 1, 'pres': 1, 'rows': len(rows)})
    game = SimpleNamespace(game_id=pre.game_id, bundle_digest='bundle', public_event_stream=pre.public_event_history,
                           authoritative_pre_prefixes=(pre,), belief_observations=tuple(observations.values()))
    folds = SimpleNamespace(folds=(SimpleNamespace(game_ids=(pre.game_id,), fold_index=4),), manifest_digest='folds')
    member = {'game_id': pre.game_id, 'bundle_digest': game.bundle_digest, 'attempt_id': ATTEMPT}
    publication = SimpleNamespace(manifest_digest=fit.PUBLICATION_DIGEST, manifest={'development_games': [member]},
                                  public_view=SimpleNamespace(games=(game,), fold_manifest=folds))
    archive = SimpleNamespace(manifest_digest=fit.Q_DIGEST, manifest={'row_and_column_seat_order': P})
    monkeypatch.setattr(seq, 'open_artifact_envelope', lambda path, **_: publication if path == fit.DEVELOPMENT else archive)
    monkeypatch.setattr(seq, 'read_artifact_file', lambda *_: json.dumps(q).encode())
    monkeypatch.setattr(fit, 'open_development_input', lambda *_, **kw: publication)
    role_record = SimpleNamespace(game_id=pre.game_id, bundle_digest=game.bundle_digest,
                                  role_assignment=tuple((p, 'Werewolf' if p in TEAM else 'Other') for p in P))
    monkeypatch.setattr(fit, 'open_role_sidecar', lambda _: SimpleNamespace(games=(role_record,)))
    m1 = {'features': fit.FEATURES['M1'], 'standardization': {'mean': [0]*3, 'std': [1]*3},
          'base': {'intercept': 0., 'coefficients': [0.] * len(fit.FEATURES['M1'])},
          'calibrator': {'intercept': 0., 'coefficients': [1.]}}
    historical = score_predictions(rows, dict.fromkeys((r.sample_id for r in rows), .5))
    saved = {'folds': {'4': {**population, 'game_ids': [pre.game_id]}}, 'models': {'M1': m1},
             'development_fold_manifest_digest': 'folds', 'evaluation': {'calibrated': {'M1': historical}}}
    monkeypatch.setattr(study, 'open_fitted_models', lambda: saved)
    assert seq.main() == 0
    assert json.loads(capsys.readouterr().out)['m1_reproduction']['rows_matched'] == len(rows)
    historical['game_equal']['brier'] += .01
    assert seq.main() == 1
    result = json.loads(capsys.readouterr().out)
    assert 'score_mismatch' in result['static_population'] and 'by_initial_region' not in result
    q['step_index'] += 1
    assert seq.main() == 1
    result = json.loads(capsys.readouterr().out)
    assert result['static_population']['legal']['rows'] == len(rows)
    assert result['static_population']['q_status_by_pre'] == {'misbound': 1}
    assert 'by_initial_region' not in result
