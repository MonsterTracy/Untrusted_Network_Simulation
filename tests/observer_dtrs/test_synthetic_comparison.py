"""Controlled comparison checks, not evidence of real LLM or game benefit."""
from copy import deepcopy
from fractions import Fraction as F
import hashlib
import json

import pytest

from scripts import compare_observer_dtrs_synthetic as c
from scripts.speech_realization import expected_action
from werewolf.observer_dtrs.decision import Region, conditional_risks
from werewolf.observer_dtrs.features import ObserverFeatures


@pytest.fixture(scope='module')
def comparison(tmp_path_factory):
    output = tmp_path_factory.mktemp('paired-parent')/'results'
    summary = c.run(output)
    return output, summary, [json.loads(line) for line in (output/'rows.jsonl').read_text().splitlines()]


def test_grid_is_complete_cartesian_not_selected_by_outcome():
    scenes = c.scenarios()
    assert len(scenes) == len({s['scene_id'] for s in scenes}) == 30
    assert {(s['phase'], s['evidence'], s['q_r']) for s in scenes} == {
        (p, e, r) for p in c.PHASES for e in c.EVIDENCE for r in c.R_VALUES}


def test_fixed_genuine_d2_fixture_gives_exact_ties_not_raw_q():
    _, model = c.synthetic_model()
    assert model.record['source']['scope'] == 'controlled_fixture'
    assert len(model.record['source']['game_sizes']) == 120
    for phase in c.PHASES:
        for r, p in zip(c.R_VALUES, c.P_VALUES):
            actual = model.predict(ObserverFeatures(r/2, r/2, r, 0, phase, 2))
            assert F(actual) == p
            if r != .5:
                assert actual != r
    assert model.digest == c.synthetic_model()[1].digest


def test_all_ninety_calls_are_paired_and_reach_real_verified_commit(comparison):
    output, summary, rows = comparison
    assert summary['status'] == 'PASS' and summary['pairing_ok'] and len(rows) == 90
    for scene in c.scenarios():
        paired = [r for r in rows if r['scene'] == scene]
        assert len(paired) == 3 and len({json.dumps(r['pairing'], sort_keys=True) for r in paired}) == 1
        assert len({r['audit']['decision']['probability_exact'] for r in paired}) == 1
        assert all(r['committed'] and r['parse_consistent'] for r in paired)
        for r in paired:
            assert r['audit']['prediction']['scope'] == 'SYNTHETIC_TEST_ONLY'
            assert r['parse']['actions'] == [expected_action(r['audit']['public_payload'], 'player1').to_record()]
            assert len(r['speech']) == len(r['parser_responses']) == len(r['parse']['attempts']) == 1
            assert r['public_evidence']
            assert r['pairing']['selection']['relation'] == {
                'self_accused': 'SELF_ACCUSED', 'team_accused': 'TEAM_ACCUSED',
                'no_accusation': 'NO_EXPLICIT_ACCUSATION'}[scene['evidence']]
    assert json.loads((output/'plan.json').read_text())['reference_loss'] == c.POLICY.static_loss.to_record()


def test_cost_changes_are_separated_from_strategy_changes(comparison):
    _, summary, rows = comparison
    assert [v['region_flips'] for v in summary['contrasts']] == [3, 0, 3]
    assert [v['intent_changes'] for v in summary['contrasts']] == [1, 7, 8]
    for scene in c.scenarios():
        a, b, d = [r['audit'] for r in rows if r['scene'] == scene]
        assert a['strategy'] == b['strategy']
        assert b['decision'] == d['decision'] and b['loss'] == d['loss']
        if scene['phase'] == 'discussion':
            assert a['decision']['region'] == b['decision']['region']
        if scene['evidence'] == 'no_accusation':
            assert a['public_payload'] == b['public_payload'] == d['public_payload'] == {
                'action': 'NO_COMMITMENT', 'target': None}


def test_common_reference_risk_uses_original_d3_and_static_argmin(comparison):
    _, summary, rows = comparison
    for r in rows:
        audit = r['audit']
        risks = conditional_risks(F(audit['decision']['probability_exact']), c.POLICY.static_loss)
        assert F(r['reference_selected_risk_exact']) == risks[Region(audit['decision']['region'])]
        assert F(r['reference_excess_exact']) >= 0
        if r['condition'] == 'static_d3':
            assert r['reference_excess_exact'] == '0'
    assert [v['mean_reference_risk_exact'] for v in summary['conditions'].values()] == ['1/10', '9/80', '9/80']
    assert [v['mean_reference_risk_difference_exact'] for v in summary['contrasts']] == ['1/80', '0', '1/80']
    assert 'by definition' in summary['reference_caveat']


def test_float_execution_ties_and_separate_exact_rational_diagnostic(comparison):
    output, _, rows = comparison
    assert {r['audit']['decision']['probability_exact'] for r in rows if len(
        r['audit']['decision']['minimizing_actions']) == 2} == {'1/4', '3/4'}
    diagnostics = json.loads((output/'rational-ties.json').read_text())
    pk = [r for r in diagnostics if r['phase'] == 'pk_discussion' and
          r['loss_name'] == 'contextual' and r['boundary'] == 'alpha']
    assert [r['decision']['region'] for r in pk] == ['BND', 'POS', 'POS']
    assert pk[1]['decision']['probability_exact'] == '3/7'
    assert pk[1]['decision']['minimizing_actions'] == ['POS', 'BND']
    assert all(r['scope'] == 'RATIONAL_D3_ONLY_NOT_D2_OR_GAMEPLAY' for r in diagnostics)


def test_failed_or_missing_rows_never_get_complete_case_risk_estimate(comparison):
    _, _, rows = comparison
    failed = deepcopy(rows)
    failed[1].update(status='failed', audit=None, committed=False, parse_consistent=False)
    result = c.summarize(failed)
    assert result['status'] == 'FAIL' and result['complete_grid'] and not result['contrasts']
    assert result['conditions']['contextual_loss']['legal_commit_rate'] == 29/30
    assert all(v['mean_reference_risk_exact'] is None for v in result['conditions'].values())
    for broken in (rows[:-1], [*rows[:-1], rows[0]]):
        result = c.summarize(broken)
        assert result['status'] == 'FAIL' and not result['complete_grid']
    drift = deepcopy(rows)
    drift[0]['pairing']['q_digest'] = '0'*64
    assert not c.summarize(drift)['pairing_ok']


def test_output_is_self_describing_and_hashed_and_never_overwritten(comparison):
    output, summary, _ = comparison
    hashes = json.loads((output/'sha256.json').read_text())
    assert all(hashlib.sha256((output/name).read_bytes()).hexdigest() == digest for name, digest in hashes.items())
    assert 'results.csv' in hashes and 'plan.json' in hashes and 'synthetic-model.json' in hashes
    assert json.loads((output/'summary.json').read_text()) == summary
    with pytest.raises(FileExistsError):
        c.run(output)
    assert hashes == json.loads((output/'sha256.json').read_text())


def test_output_cannot_be_a_source_checkout():
    with pytest.raises(ValueError, match='outside'):
        c.run(c.Path(__file__).resolve().parents[2]/'unapproved-output')


def test_parser_observes_text_not_a_prediction_or_intent_record():
    parser = c.ControlledParser()
    message = [{'content': '待解析的玩家发言：\nplayer1: 我质疑player3此前的判断。'}]
    assert parser.chat(messages=message) == 'player1 | oppose | player3'
    with pytest.raises(ValueError, match='grammar'):
        parser.chat(messages=[{'content': '待解析的玩家发言：\nplayer1: 未声明的话语。'}])
