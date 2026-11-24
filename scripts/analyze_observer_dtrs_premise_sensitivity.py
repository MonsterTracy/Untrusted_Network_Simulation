"""在冻结 Development Fold 4 的同一 PRE 上复询三臂并盲评计划。"""

import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path
from random import Random

import numpy as np
import yaml

from scripts import analyze_observer_dtrs_report_event as study
from scripts import fit_observer_dtrs_report_event as fit
from werewolf.agents.gpt_agent import GPTAgent, GameplayActionValidationError
from werewolf.artifact_io import open_artifact_envelope, read_artifact_file, sha256_bytes
from werewolf.backends import BackendError, load_named_backends
from werewolf.observer_dtrs.context import contexts_for_pre
from werewolf.observer_dtrs.probabilities import ProbabilityMatrix
from werewolf.observer_dtrs.speech_runtime import (
    ReportEventArm, report_event_directions, report_event_plan_input, report_event_probabilities,
)
from werewolf.runtime_config import normalize_runtime_config
from werewolf.strategy.speech_planning import generate_candidates
from werewolf.tom.protocol import bootstrap_indices


ARMS = tuple(arm.value for arm in ReportEventArm)
_, TWO, THREE = ARMS
HYPOTHESES = ('includes', 'excludes')
PREFERENCES = ('A', 'B', 'tie')


def load_opportunities(saved):
    """沿用固定输入身份，只联结 PRE、当前狼的队伍和 Q。"""
    held = saved['folds']['4']
    games = set(held['game_ids'])
    publication = open_artifact_envelope(fit.DEVELOPMENT, expected_artifact_type=fit.PUBLICATION_TYPE,
                                         expected_schema_version=fit.PUBLICATION_VERSION)
    if publication.manifest_digest != fit.PUBLICATION_DIGEST:
        raise ValueError('Development Publication 与封存 M1 不一致')
    game_ids = tuple(g['game_id'] for g in publication.manifest['development_games'] if g['game_id'] in games)
    archive = open_artifact_envelope(fit.Q_PATH, expected_artifact_type=fit.Q_TYPE,
                                     expected_schema_version=fit.Q_TYPE)
    if (archive.manifest_digest != fit.Q_DIGEST
            or tuple(archive.manifest['row_and_column_seat_order']) != fit.PLAYER_IDS):
        raise ValueError('Q 档案或座位顺序与封存 M1 不一致')
    by_pre = {}
    for line in read_artifact_file(archive, 'predictions.jsonl').splitlines():
        record = json.loads(line)
        if record['game_id'] in games:
            key = record['game_id'], record['boundary_id']
            if key in by_pre:
                raise ValueError(f'Fold 4 Q 重复：{key}')
            by_pre[key] = record

    opportunities, teams = [], None
    for offset in range(0, len(game_ids), 120):
        batch_ids = game_ids[offset:offset + 120]
        batch = fit.open_development_input(fit.DEVELOPMENT, expected_publication_digest=fit.PUBLICATION_DIGEST,
                                           expected_split_digest=fit.SPLIT_DIGEST, game_ids=batch_ids)
        if teams is None:
            folds = batch.public_view.fold_manifest
            fold4 = next(f for f in folds.folds if f.fold_index == fit.EVAL_FOLD)
            if set(fold4.game_ids) != games or folds.manifest_digest != saved['development_fold_manifest_digest']:
                raise ValueError('Fold 4 名单与封存 M1 不一致')
            teams = {(g.game_id, g.bundle_digest): tuple(p for p, r in g.role_assignment if r == 'Werewolf')
                     for g in fit.open_role_sidecar(batch).games if g.game_id in games}
        if tuple(g.game_id for g in batch.public_view.games) != batch_ids:
            raise ValueError('Development 批次与 Fold 4 名单不一致')
        for game in batch.public_view.games:
            team = teams[game.game_id, game.bundle_digest]
            for pre in game.authoritative_pre_prefixes:
                if pre.current_speaker not in team or pre.public_temporal_state.phase.value not in fit.SPEECH_PHASES:
                    continue
                key = pre.game_id, pre.boundary_id
                record = by_pre.pop(key, None)
                contexts = contexts_for_pre(pre, team)
                if (record is None or record['prefix_digest'] != pre.prefix_digest
                        or record['step_index'] != pre.step_index or not contexts):
                    raise ValueError(f'合法 PRE、观察者或 Q 联结失败：{key}')
                matrix = ProbabilityMatrix(*key, pre.prefix_digest, record['q'])
                probabilities = report_event_probabilities(pre, contexts, matrix, m1_parameters=saved['models']['M1'])
                opportunities.append((pre, team, probabilities))

    population = {'games': len({pre.game_id for pre, _, _ in opportunities}), 'pres': len(opportunities),
                  'rows': sum(len(probabilities) for _, _, probabilities in opportunities)}
    if (by_pre or {pre.game_id for pre, _, _ in opportunities} != games or population != study.FOLD4
            or any(held[key] != value for key, value in population.items())):
        raise ValueError(f'Fold 4 起点人口不一致：{population}，未联结 Q {len(by_pre)} 条')
    return opportunities


def focal_observer(three_way_panel):
    # min 保留第一个并列项。按固定座位顺序扫描，不依赖字典顺序。
    candidates = (seat for seat in fit.PLAYER_IDS
                  if seat in three_way_panel and not three_way_panel[seat]['adoptable_directions'])
    return min(candidates, key=lambda seat: abs(three_way_panel[seat]['probability'] - 0.5), default=None)


def compare_plans(agent, context, *, observer, hypothesis, plan_a, plan_b):
    """单次无状态询问。候选来源、承诺权限和原策略依据均不提供。"""
    premise = '包含' if hypothesis == 'includes' else '不包含'
    prompt = (
        '你是当前发言的狼人，最终目标是狼队获胜。考虑放逐威胁、公开证据、发言可信度、队友利益和后续策略空间。\n'
        f'仅在本次比较中假设 {observer} 当前成功取得的原始嫌疑报告{premise}你。'
        '这是讨论用的假设，私人报告仍未知。report_predictions 保留提出假设前的预测概率，'
        '没有对其他观察者进行联合信念更新。公开发言只证明有人说过这些话。\n'
        '从匿名计划 A 和 B 中选更有利于狼队的计划，也可选 tie 表示并列。'
        'ACCUSE_WOLF 为指狼，CLEAR 为认好，SUPPORT 和 OPPOSE 为支持或反对公开观点，'
        'SELF_DEFEND 为自辩，NO_COMMITMENT 为不作明确表态。仅返回 preference。'
    )
    temperature, max_tokens = agent._request_limits()
    content, metadata = agent._chat_with_metadata(
        [{'role': 'user', 'content': prompt + '\n\n' + json.dumps(
            {**context, 'comparison': {'A': plan_a, 'B': plan_b}}, ensure_ascii=False)}],
        temperature=temperature, max_tokens=max_tokens,
        response_format=agent._decision_response_format(
            'report_event_plan_preference', preference={'type': 'string', 'enum': list(PREFERENCES)}),
        extra_body={'chat_template_kwargs': {'enable_thinking': False}},
    )
    try:
        payload = json.loads(content)
    except (TypeError, json.JSONDecodeError) as exc:
        raise GameplayActionValidationError('条件偏好响应必须为 JSON') from exc
    if (metadata['finish_reason'] == 'length' or type(payload) is not dict
            or set(payload) != {'preference'} or payload['preference'] not in PREFERENCES):
        raise GameplayActionValidationError('条件偏好必须返回 A、B 或 tie')
    return payload['preference']


def select_plan(agent, context, refs):
    try:
        index, basis = agent.select_report_event_plan(context, basis_refs=refs)
    except (GameplayActionValidationError, BackendError) as exc:
        return {'error_type': type(exc).__name__}
    return {'plan': context['candidates'][index], 'basis_ref': basis}


def preference_pair(agent, context, observer, plan_a, plan_b):
    responses = {}
    for hypothesis in HYPOTHESES:
        try:
            responses[hypothesis] = {'choice': compare_plans(
                agent, context, observer=observer, hypothesis=hypothesis, plan_a=plan_a, plan_b=plan_b)}
        except (GameplayActionValidationError, BackendError) as exc:
            responses[hypothesis] = {'error_type': type(exc).__name__}
    return responses


def dominance(responses, *, three_way_first):
    choices = {reply.get('choice') for reply in responses.values()}
    if None in choices:
        return None
    three, two = ('A', 'B') if three_way_first else ('B', 'A')
    return int(three in choices and two not in choices) - int(two in choices and three not in choices)


def evaluate_pre(agent, pre, team, probabilities, *, three_way_first, repeat):
    candidates = generate_candidates(pre)
    panels = {arm.value: report_event_directions(probabilities, arm) for arm in ReportEventArm}
    inputs = {arm: report_event_plan_input(pre, team, panel, candidates) for arm, panel in panels.items()}
    observer = focal_observer(panels[THREE])
    selections = {arm: select_plan(agent, *inputs[arm]) for arm in ARMS}
    two, three = (selections[arm].get('plan') for arm in (TWO, THREE))
    record = {'game_id': pre.game_id, 'boundary_id': pre.boundary_id, 'prefix_digest': pre.prefix_digest,
              'step_index': pre.step_index, 'acting_wolf': pre.current_speaker,
              'probabilities': probabilities, 'focal_observer': observer, 'selections': selections,
              'z': None}
    if repeat:
        record['repeated_selections'] = {arm: select_plan(agent, *inputs[arm]) for arm in ARMS}
    if observer is not None and two is not None and three is not None:
        if two == three:
            record['z'] = 0
        else:
            context, _ = inputs[THREE]
            context = {**context, 'report_predictions': probabilities}
            plan_a, plan_b = (three, two) if three_way_first else (two, three)
            responses = preference_pair(agent, context, observer, plan_a, plan_b)
            record['preference'] = {'option_a_arm': THREE if three_way_first else TWO, 'responses': responses}
            record['z'] = dominance(responses, three_way_first=three_way_first)
            if repeat:
                record['repeated_preference'] = preference_pair(agent, context, observer, plan_a, plan_b)
    return record


def summarize(rows, *, replicates=fit.BOOTSTRAP_REPLICATES):
    sizes = Counter(row['game_id'] for row in rows)
    games = sorted(sizes)
    game_index = {game: j for j, game in enumerate(games)}
    weights = np.asarray([1 / sizes[game] for game in games])
    draws = bootstrap_indices(games, replicates, fit.BOOTSTRAP_SEED)

    def support(records):
        counts = Counter(row['game_id'] for row in records)
        return {'pres': len(records), 'games': len(counts),
                'game_equal_mass': sum(n / sizes[g] for g, n in counts.items()) / len(games)}

    def estimate(records, values):
        numerator, denominator = np.zeros(len(games)), np.zeros(len(games))
        for row, value in zip(records, values, strict=True):
            index = game_index[row['game_id']]
            numerator[index] += value
            denominator[index] += 1
        return {**support(records), 'estimate': float(weights @ numerator / (weights @ denominator)) if records else None,
                'bootstrap': study.interval(study.bootstrap_ratio(numerator, denominator, weights, draws))}

    def disagreement(group, left, right, field):
        valid, different = [], []
        for row in group:
            a, b = left(row).get(field), right(row).get(field)
            if a is not None and b is not None:
                valid.append(row)
                different.append(a != b)
        return {'population': support(group), 'valid': estimate(valid, different),
                'invalid': support([row for row in group if left(row).get(field) is None or right(row).get(field) is None])}

    bnd = [row for row in rows if row['focal_observer'] is not None]
    no_bnd = [row for row in rows if row['focal_observer'] is None]
    queried = [row for row in bnd if 'preference' in row]
    valid_z = [row for row in bnd if row['z'] is not None]
    valid_queried = [row for row in queried if row['z'] is not None]
    repeats = [row for row in rows if 'repeated_selections' in row]
    repeated_preferences = [row for row in repeats if 'repeated_preference' in row]
    return {
        'action_disagreement': {
            group_name: {f'{a}/{b}': disagreement(group, lambda r: r['selections'][a],
                                                  lambda r: r['selections'][b], 'plan')
                         for a, b in combinations(ARMS, 2)}
            for group_name, group in (('all', rows), ('bnd', bnd), ('no_bnd', no_bnd))},
        'preference': {
            'initial_bnd': support(bnd), 'no_bnd': support(no_bnd),
            'two_or_three_selection_invalid': support([r for r in bnd if any(
                'plan' not in r['selections'][arm] for arm in (TWO, THREE))]),
            'same_plan': support([r for r in valid_z if 'preference' not in r]),
            'different_plan_queried': support(queried),
            'invalid_pair': support([r for r in queried if r['z'] is None]),
            'z_support': {str(z): support([r for r in valid_z if r['z'] == z]) for z in (-1, 0, 1)},
            'delta_pref_valid_bnd': estimate(valid_z, (r['z'] for r in valid_z)),
            'delta_pref_valid_different_plan': estimate(valid_queried, (r['z'] for r in valid_queried)),
        },
        'same_condition_repeat': {
            'selections': {arm: disagreement(repeats, lambda r: r['selections'][arm],
                                             lambda r: r['repeated_selections'][arm], 'plan') for arm in ARMS},
            'preferences': {h: disagreement(repeated_preferences, lambda r: r['preference']['responses'][h],
                                            lambda r: r['repeated_preference'][h], 'choice') for h in HYPOTHESES},
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-config', type=Path, required=True)
    parser.add_argument('--profile', required=True, help='现有 Runtime 中的 GPTAgent profile_name')
    args = parser.parse_args()
    raw = args.runtime_config.read_bytes()
    config = normalize_runtime_config(yaml.safe_load(raw))
    profile = next((p for p in config['agent_config']['all_candidates'] if p['profile_name'] == args.profile), None)
    if profile is None or not config['backends'][profile['backend']]['supports_json_schema']:
        parser.error('选择支持 JSON Schema 的现有 GPTAgent profile')
    saved = study.open_fitted_models()
    opportunities = load_opportunities(saved)
    backends = load_named_backends(config, env_file=None, max_retries=0)
    agent = GPTAgent(backend=backends[profile['backend']], model_name=profile['model'], **profile['model_params'])
    rng = Random(fit.BOOTSTRAP_SEED)
    repeat_indices = set(rng.sample(range(len(opportunities)), max(1, len(opportunities) // 20)))
    three_way_first = bool(rng.randrange(2))
    print(json.dumps({'type': 'settings', 'scope': 'development_fold4_premise_sensitivity',
                      'publication_digest': fit.PUBLICATION_DIGEST,
                      'fold_manifest_digest': saved['development_fold_manifest_digest'],
                      'fit_digest': study.FIT_DIGEST, 'q_digest': fit.Q_DIGEST,
                      'runtime_config_sha256': sha256_bytes(raw), 'profile': profile, 'population': study.FOLD4,
                      'repeat_pres': len(repeat_indices), 'seed': fit.BOOTSTRAP_SEED}), flush=True)
    rows = []
    for index, (pre, team, probabilities) in enumerate(opportunities):
        row = evaluate_pre(agent, pre, team, probabilities, three_way_first=three_way_first,
                           repeat=index in repeat_indices)
        rows.append(row)
        if 'preference' in row:
            three_way_first = not three_way_first
        print(json.dumps({'type': 'pre', **row}), flush=True)
    print(json.dumps({'type': 'summary', **summarize(rows)}), flush=True)


if __name__ == '__main__':
    main()
