"""用真实格式的合成 PRE 检查离线复询及统计，调用全部由 Mock 提供。"""

import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts import analyze_observer_dtrs_premise_sensitivity as experiment
from tests.observer_dtrs.conftest import prefix_for
from tests.observer_dtrs.test_speech_runtime import P, report_event_prediction
from werewolf.agents.gpt_agent import GPTAgent


def agent_with_replies(*replies):
    backend = Mock(supports_json_schema=True)
    backend.chat_with_metadata.side_effect = [(json.dumps(reply), {'finish_reason': 'stop'}) for reply in replies]
    return GPTAgent(backend=backend, model_name='mock', temperature=.7, gameplay_max_tokens=2048)


def choice(index, basis='independent'):
    return {'candidate_index': index, 'basis_ref': basis}


def inputs(agent):
    requests = [c.kwargs for c in agent.backend.chat_with_metadata.call_args_list]
    parts = [r['messages'][0]['content'].rsplit('\n\n', 1) for r in requests]
    return requests, [(prompt, json.loads(context)) for prompt, context in parts]


def test_shared_online_inputs_focal_tie_and_blind_preference():
    pre = prefix_for()
    probabilities = dict(zip(reversed(P[2:]), [.9, .75, .375, .625, .25], strict=True))
    agent = agent_with_replies(choice(0), choice(1), choice(2, 'keep_both'),
                               {'preference': 'B'}, {'preference': 'tie'})
    row = experiment.evaluate_pre(agent, pre, P[:2], probabilities, three_way_first=False, repeat=False)
    assert row['focal_observer'] == P[3]
    assert row['z'] == 1
    assert row['preference']['option_a_arm'] == experiment.TWO
    assert row['selections'][experiment.THREE]['basis_ref'] == 'keep_both'

    requests, parts = inputs(agent)
    assert len(parts) == 5
    assert all((r['model'], r['temperature'], r['max_tokens']) == ('mock', .7, 2048) for r in requests)
    prompt, first = parts[0]
    allowed = {'speaker', 'alive_players', 'known_wolf_team', 'public_events', 'report_predictions', 'candidates'}
    assert set(first) == allowed
    for text, context in parts[:3]:
        assert text == prompt
        assert {k: context[k] for k in allowed - {'report_predictions'}} == {
            k: first[k] for k in allowed - {'report_predictions'}}
        assert {s: v['probability'] for s, v in context['report_predictions'].items()} == probabilities
    assert [context['report_predictions'][P[3]]['adoptable_directions'] for _, context in parts[:3]] == [
        ['POS', 'NEG'], ['POS'], []]

    (includes, left), (excludes, right) = parts[3:]
    assert f'假设 {P[3]} 当前成功取得的原始嫌疑报告包含你' in includes
    assert f'假设 {P[3]} 当前成功取得的原始嫌疑报告不包含你' in excludes
    assert left == right
    assert set(left) == allowed | {'comparison'}
    assert left['report_predictions'] == probabilities
    assert left['comparison'] == {'A': first['candidates'][1], 'B': first['candidates'][2]}
    for request in requests:
        assert len(request['messages']) == 1
        payload = request['messages'][0]['content']
        assert all(value not in payload for value in ('belief_observations', 'observation_id', 'winner', 'basis_ref":'))


@pytest.mark.parametrize('probability,three_way_index,z', [(0.5, 1, 0), (0.9, 2, None), (0.5, 99, None)])
def test_preference_requires_bnd_and_two_distinct_valid_plans(probability, three_way_index, z):
    agent = agent_with_replies(choice(0), choice(1), choice(three_way_index, 'keep_both'))
    row = experiment.evaluate_pre(agent, prefix_for(), P[:2], dict.fromkeys(P[2:], probability),
                                   three_way_first=True, repeat=False)
    assert row['z'] == z and 'preference' not in row
    requests, _ = inputs(agent)
    assert len(requests) == 3
    if probability == 0.9:
        assert requests[1] == requests[2]


@pytest.mark.parametrize('includes,excludes,expected', [
    ('A', 'tie', 1), ('tie', 'B', -1), ('A', 'B', 0), ('tie', 'tie', 0), ('invalid', 'A', None),
])
def test_dominance_and_independent_repeat(includes, excludes, expected):
    agent = agent_with_replies(choice(0), choice(1), choice(2), choice(0), choice(3), choice(4),
                               {'preference': includes}, {'preference': excludes},
                               {'preference': 'A'}, {'preference': 'tie'})
    row = experiment.evaluate_pre(agent, prefix_for(), P[:2], dict.fromkeys(P[2:], .5),
                                   three_way_first=True, repeat=True)
    assert row['z'] == expected
    requests, _ = inputs(agent)
    assert requests[:3] == requests[3:6]
    assert requests[6:8] == requests[8:10]
    assert row['repeated_selections'][experiment.TWO] != row['selections'][experiment.TWO]
    if expected is None:
        assert row['preference']['responses']['includes'] == {'error_type': 'GameplayActionValidationError'}


def test_game_equal_denominators_keep_full_pre_population():
    def row(game, *, focal=True, same=False, z=None, invalid=False):
        two = {'plan': {'action': 'SELF_DEFEND', 'target': None}}
        three = {'plan': {'action': 'NO_COMMITMENT', 'target': None}}
        result = {'game_id': game, 'focal_observer': P[2] if focal else None, 'z': z,
                  'selections': {experiment.ARMS[0]: two, experiment.TWO: two,
                                 experiment.THREE: {} if invalid else two if same else three}}
        if focal and not same and not invalid:
            result['preference'] = {}
        return result

    rows = [row('g1', z=1), row('g1', same=True, z=0), row('g1'), row('g1', focal=False),
            row('g2', z=-1), row('g3', invalid=True)]
    result = experiment.summarize(rows, replicates=250)
    p = result['preference']
    assert p['initial_bnd'] == {'pres': 5, 'games': 3, 'game_equal_mass': pytest.approx(2.75 / 3)}
    assert p['invalid_pair']['pres'] == 1
    assert p['two_or_three_selection_invalid']['pres'] == 1
    assert p['same_plan']['pres'] == 1
    assert p['delta_pref_valid_bnd']['estimate'] == pytest.approx(-0.5)
    assert p['delta_pref_valid_different_plan']['estimate'] == pytest.approx(-0.6)
    assert p['delta_pref_valid_bnd']['bootstrap']['valid_replicates'] < 250
    pair = f'{experiment.TWO}/{experiment.THREE}'
    assert result['action_disagreement']['all'][pair]['valid']['estimate'] == pytest.approx(.875)
    assert result['action_disagreement']['no_bnd'][pair]['valid']['estimate'] == 1
    assert result['action_disagreement']['all'][pair]['invalid']['pres'] == 1


def test_fold_loader_joins_q_without_accessing_private_reports(monkeypatch):
    pre = prefix_for()
    matrix, _ = report_event_prediction(pre)
    fit, study = experiment.fit, experiment.study
    population = {'games': 1, 'pres': 1, 'rows': 5}
    publication = SimpleNamespace(manifest_digest=fit.PUBLICATION_DIGEST,
                                  manifest={'development_games': [{'game_id': pre.game_id}, {'game_id': 'another-fold'}]})
    archive = SimpleNamespace(manifest_digest=fit.Q_DIGEST, manifest={'row_and_column_seat_order': P})
    monkeypatch.setattr(experiment, 'open_artifact_envelope', lambda path, **_: publication if path == fit.DEVELOPMENT else archive)
    q_record = {'game_id': pre.game_id, 'boundary_id': pre.boundary_id, 'q': matrix.values,
                'prefix_digest': pre.prefix_digest, 'step_index': pre.step_index}
    monkeypatch.setattr(experiment, 'read_artifact_file', lambda *_: json.dumps(q_record).encode())
    game = SimpleNamespace(game_id=pre.game_id, bundle_digest='bundle', authoritative_pre_prefixes=(pre,))
    folds = SimpleNamespace(folds=(SimpleNamespace(game_ids=(pre.game_id,), fold_index=4),), manifest_digest='folds')
    batch = SimpleNamespace(public_view=SimpleNamespace(games=(game,), fold_manifest=folds))
    reader = Mock(return_value=batch)
    monkeypatch.setattr(fit, 'open_development_input', reader)
    roles = SimpleNamespace(game_id=pre.game_id, bundle_digest='bundle', role_assignment=tuple(
        (p, 'Werewolf' if p in P[:2] else 'Other Role') for p in P))
    monkeypatch.setattr(fit, 'open_role_sidecar', lambda _: SimpleNamespace(games=(roles,)))
    monkeypatch.setattr(study, 'FOLD4', population)
    m1 = {'features': fit.FEATURES['M1'], 'standardization': {'mean': [0, 0, 0], 'std': [1, 1, 1]},
          'base': {'intercept': 0., 'coefficients': [0.] * len(fit.FEATURES['M1'])},
          'calibrator': {'intercept': 0., 'coefficients': [1.]}}
    saved = {'folds': {'4': {**population, 'game_ids': [pre.game_id]}},
             'models': {'M1': m1}, 'development_fold_manifest_digest': 'folds'}
    assert experiment.load_opportunities(saved) == [(pre, P[:2], dict.fromkeys(P[2:], .5))]
    assert reader.call_args.kwargs['game_ids'] == (pre.game_id,)
    q_record['step_index'] += 1
    with pytest.raises(ValueError, match='联结失败'):
        experiment.load_opportunities(saved)


def test_entry_point_balances_anonymous_options(monkeypatch, capsys):
    runtime = Path(__file__).resolve().parents[2] / 'configs/runtime/local-qwen35-9b.yaml'
    monkeypatch.setattr(sys, 'argv', ['experiment', '--runtime-config', str(runtime), '--profile', 'local_qwen'])
    monkeypatch.setattr(experiment.study, 'open_fitted_models', lambda: {'development_fold_manifest_digest': 'mock'})
    monkeypatch.setattr(experiment, 'load_opportunities', lambda _: [
        (prefix_for(game_id=game), P[:2], dict.fromkeys(P[2:], .5)) for game in ('g1', 'g2')])
    backend = Mock(supports_json_schema=True)

    def respond(**request):
        schema = request['response_format']['json_schema']['name']
        context = json.loads(request['messages'][0]['content'].rsplit('\n\n', 1)[1])
        if schema == 'report_event_plan':
            reply = choice(1 if context['report_predictions'][P[2]]['adoptable_directions'] else 2)
        else:
            reply = {'preference': 'tie'}
        return json.dumps(reply), {'finish_reason': 'stop'}

    backend.chat_with_metadata.side_effect = respond
    monkeypatch.setattr(experiment, 'load_named_backends', lambda *_a, **_kw: {'local_qwen': backend})
    experiment.main()
    header, first, second, summary = map(json.loads, capsys.readouterr().out.splitlines())
    assert [r['type'] for r in (header, first, second, summary)] == ['settings', 'pre', 'pre', 'summary']
    assert {first['preference']['option_a_arm'], second['preference']['option_a_arm']} == {experiment.TWO, experiment.THREE}
    assert sum('repeated_selections' in row for row in (first, second)) == header['repeat_pres'] == 1
    assert backend.chat_with_metadata.call_count == 15
