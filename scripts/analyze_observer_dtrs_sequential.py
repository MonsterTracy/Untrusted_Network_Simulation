"""在冻结 Development Fold 4 上作同一狼人和观察者的序贯认知评价。"""

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
import sys
from types import SimpleNamespace

import numpy as np

from scripts import analyze_observer_dtrs_report_event as study
from scripts import fit_observer_dtrs_report_event as fit
from werewolf.artifact_io import open_artifact_envelope, read_artifact_file
from werewolf.observer_dtrs.context import ObserverContext, contexts_for_pre
from werewolf.observer_dtrs.decision import Region, decide, forced_binary
from werewolf.observer_dtrs.decision_offline import realized_loss
from werewolf.observer_dtrs.labels import label_from_observation
from werewolf.observer_dtrs.probabilities import ProbabilityMatrix
from werewolf.tom.protocol import bootstrap_indices


@dataclass(frozen=True)
class Pair:
    publication_digest: str
    attempt_id: str
    bundle_digest: str
    start: ObserverContext
    followup: ObserverContext | None
    wolf_death: int | None = None
    observer_death: int | None = None
    terminal: int | None = None

    @property
    def sequence_id(self):
        return (self.publication_digest, self.start.game_id, self.attempt_id, self.bundle_digest,
                self.start.acting_wolf, self.start.observer_id)


def sample_id(context):
    return context.game_id, context.boundary_id, context.observer_id


def pair_game(game, team, *, publication_digest, attempt_id):
    """同一游戏尝试内按权威步号取紧邻机会，完全独立于 Q 与报告。"""
    prefixes = sorted(game.authoritative_pre_prefixes, key=lambda pre: pre.step_index)
    steps = [pre.step_index for pre in prefixes]
    turns = [pre.public_event_history.events[-1].event_index for pre in prefixes]
    if any(a >= b for axis in (steps, turns) for a, b in zip(axis, axis[1:])):
        raise ValueError(f'权威步号与公开顺序不一致：{game.game_id}')

    deaths = {}
    events = game.public_event_stream.events
    for event in events:
        if event.event_type in ('death_announcement', 'exile_result'):
            for seat in event.affected_players:
                deaths.setdefault(seat, event.event_index)
    living = set(fit.PLAYER_IDS) - deaths.keys()
    wolves, others = len(living.intersection(team)), len(living.difference(team))
    terminal = None
    # 原生终止检查紧跟公开死亡或放逐记录。流末本身不能证明终局。
    if (events and events[-1].event_type in ('death_announcement', 'exile_result')
            and (wolves == 0 or wolves >= others)):
        terminal = events[-1].event_index

    boundaries, pairs, following = [], [], {}
    for pre in reversed(prefixes):
        if pre.current_speaker not in team or pre.public_temporal_state.phase.value not in fit.SPEECH_PHASES:
            continue
        contexts = contexts_for_pre(pre, team)
        if not contexts:
            raise ValueError(f'狼人 PRE 缺少合法观察者：{game.game_id}/{pre.boundary_id}')
        boundaries.append((pre, contexts))
        turn = pre.public_event_history.events[-1].event_index
        for ctx in reversed(contexts):
            key = ctx.acting_wolf, ctx.observer_id
            later = following.get(key)
            facts = {}
            if later is None:
                for name, index in (('wolf_death', deaths.get(ctx.acting_wolf)),
                                    ('observer_death', deaths.get(ctx.observer_id)), ('terminal', terminal)):
                    if index is not None and index > turn:
                        facts[name] = index
            pairs.append(Pair(publication_digest, attempt_id, game.bundle_digest, ctx, later, **facts))
            following[key] = ctx
    return list(reversed(boundaries)), list(reversed(pairs))


def join_pre(pre, contexts, team, observations, q, attempt_id):
    """正常路径复用静态样本。异常 PRE 保留单侧可用证据及失败类别。"""
    links = {link.observer_id: link.observation_id for link in pre.belief_observation_links}
    if q is not None and all(
            getattr(observations.get(links[ctx.observer_id]), 'attempt_id', None) == attempt_id for ctx in contexts):
        try:
            samples = tuple(fit.pre_samples(pre, team, observations, q))
        except (KeyError, TypeError, ValueError):
            pass
        else:
            return samples, {s.sample_id: (s.y, 'valid') for s in samples}, 'valid'

    labels = {}
    for ctx in contexts:
        oid = links.get(ctx.observer_id)
        report = observations.get(oid)
        status, y = 'missing', None
        if report is not None:
            try:
                label = label_from_observation(ctx, report)
                if label.observation_id != oid or label.attempt_id != attempt_id:
                    raise ValueError('报告尝试身份不符')
            except (TypeError, ValueError):
                status = 'misbound_or_invalid'
            else:
                status = 'valid' if label.label_observed else label.observation_status
                if label.label_observed:
                    y = int(ctx.acting_wolf in label.raw_suspicion_support)
        labels[sample_id(ctx)] = (y, status)

    if q is None:
        return (), labels, 'missing'
    identity = (pre.game_id, pre.boundary_id, pre.prefix_digest, pre.step_index)
    if tuple(q.get(field) for field in ('game_id', 'boundary_id', 'prefix_digest', 'step_index')) != identity:
        return (), labels, 'misbound'
    try:
        matrix = ProbabilityMatrix(*identity[:3], q['q'])
    except (KeyError, TypeError, ValueError):
        return (), labels, 'invalid'
    rows = tuple(SimpleNamespace(sample_id=sample_id(ctx), features=x)
                 for ctx, _, x in fit.legal_features(contexts, matrix))
    return rows, labels, 'valid'


def load_fold4(saved):
    """复用冻结 Publication、Q、角色 sidecar 与历史 M1 的读取合同。"""
    held = saved['folds']['4']
    games = set(held['game_ids'])
    publication = open_artifact_envelope(fit.DEVELOPMENT, expected_artifact_type=fit.PUBLICATION_TYPE,
                                         expected_schema_version=fit.PUBLICATION_VERSION)
    if publication.manifest_digest != fit.PUBLICATION_DIGEST:
        raise ValueError('Development Publication 身份不符')
    members = {g['game_id']: g for g in publication.manifest['development_games'] if g['game_id'] in games}
    archive = open_artifact_envelope(fit.Q_PATH, expected_artifact_type=fit.Q_TYPE,
                                     expected_schema_version=fit.Q_TYPE)
    if (archive.manifest_digest != fit.Q_DIGEST or tuple(archive.manifest['row_and_column_seat_order']) != fit.PLAYER_IDS):
        raise ValueError('Q 档案身份或绝对座位顺序不符')
    by_pre = {}
    for line in read_artifact_file(archive, 'predictions.jsonl').splitlines():
        record = json.loads(line)
        if record['game_id'] in games:
            key = record['game_id'], record['boundary_id']
            if key in by_pre:
                raise ValueError(f'Fold 4 Q 重复：{key}')
            by_pre[key] = record

    pairs, rows, labels, q_status, teams = [], [], {}, {}, None
    ids = tuple(members)
    for offset in range(0, len(ids), 120):
        batch_ids = ids[offset:offset + 120]
        batch = fit.open_development_input(fit.DEVELOPMENT, expected_publication_digest=fit.PUBLICATION_DIGEST,
                                           expected_split_digest=fit.SPLIT_DIGEST, game_ids=batch_ids)
        if teams is None:
            folds = batch.public_view.fold_manifest
            fold = next(f for f in folds.folds if f.fold_index == fit.EVAL_FOLD)
            if set(fold.game_ids) != games or folds.manifest_digest != saved['development_fold_manifest_digest']:
                raise ValueError('Fold 4 名单与封存拟合不一致')
            teams = {(g.game_id, g.bundle_digest): tuple(p for p, r in g.role_assignment if r == 'Werewolf')
                     for g in fit.open_role_sidecar(batch).games if g.game_id in games}
        if tuple(g.game_id for g in batch.public_view.games) != batch_ids:
            raise ValueError('Development 批次与 Fold 4 名单不一致')
        for game in batch.public_view.games:
            member = members[game.game_id]
            attempt = member['attempt_id']
            if not attempt or game.bundle_digest != member['bundle_digest']:
                raise ValueError(f'成员、游戏尝试与 Bundle 身份不符：{game.game_id}')
            team = teams[game.game_id, game.bundle_digest]
            boundaries, game_pairs = pair_game(game, team, publication_digest=batch.manifest_digest, attempt_id=attempt)
            pairs.extend(game_pairs)
            observations = {o.observation_id: o for o in game.belief_observations}
            for pre, contexts in boundaries:
                key = pre.game_id, pre.boundary_id
                joined, raw_labels, status = join_pre(pre, contexts, team, observations, by_pre.pop(key, None), attempt)
                rows.extend(joined)
                labels.update(raw_labels)
                q_status[key] = status

    probabilities = study.rebuild_probabilities(rows, 'M1', saved['models']['M1']) if rows else {}
    invalid_values = [key for key, p in probabilities.items() if not math.isfinite(p) or not 0 <= p <= 1]
    for key in invalid_values:
        del probabilities[key]
    complete = [r for r in rows if r.sample_id in probabilities and isinstance(r, fit.Sample)]
    diagnostics = {
        'legal': {'games': len({pair.start.game_id for pair in pairs}), 'pres': len(q_status), 'rows': len(pairs)},
        'label_status_by_row': dict(Counter(status for _, status in labels.values())),
        'q_status_by_pre': dict(Counter(q_status.values())), 'invalid_probability_rows': len(invalid_values),
        'usable_predictions': len(probabilities), 'complete_static': fit.population(complete),
        'unused_q_pres': len(by_pre),
        'q_issue_examples': [(key, status) for key, status in q_status.items() if status != 'valid'][:10],
        'label_issue_examples': [(key, status) for key, (_, status) in labels.items() if status != 'valid'][:10],
    }
    expected = study.FOLD4
    diagnostics['archived_static'] = {key: held[key] for key in diagnostics['complete_static']}
    if (diagnostics['legal'] != expected or any(held[k] != n for k, n in expected.items())
            or len(held['game_ids']) != len(games) or {p.start.game_id for p in pairs} != games
            or diagnostics['complete_static'] != diagnostics['archived_static'] or by_pre):
        return pairs, probabilities, labels, diagnostics, None
    try:
        reproduction = study.verify_scores(complete, 'M1', probabilities, saved['evaluation']['calibrated']['M1'])
    except ValueError as exc:
        diagnostics['score_mismatch'] = str(exc)
        reproduction = None
    return pairs, probabilities, labels, diagnostics, reproduction


def summarize(pairs, probabilities, labels, *, replicates=fit.BOOTSTRAP_REPLICATES):
    sizes = Counter(pair.start.game_id for pair in pairs)
    games = sorted(sizes)
    index = {g: i for i, g in enumerate(games)}
    weights = np.asarray([1 / sizes[g] for g in games])
    game = np.asarray([index[p.start.game_id] for p in pairs])
    draws = bootstrap_indices(games, replicates, fit.BOOTSTRAP_SEED)

    def totals(mask, values=None):
        return np.bincount(game[mask], weights=None if values is None else values[mask], minlength=len(games))

    def support(mask):
        counts = totals(mask)
        return {'rows': int(counts.sum()), 'games': int(np.count_nonzero(counts)),
                'game_equal_mass': float(weights @ counts / len(games))}

    def estimate(mask, values, *, bootstrap=False):
        den = totals(mask)
        num = totals(mask, values)
        mass = weights @ den
        result = {'value': float(weights @ num / mass) if mass else None,
                  'na_reason': None if mass else 'no_support'}
        if bootstrap:
            interval = study.interval(study.bootstrap_ratio(num, den, weights, draws))
            interval['na_reason'] = (None if interval['percentile_95'] is not None else
                                     'insufficient_valid_replicates' if mass else 'no_support')
            result['bootstrap'] = interval
        return result

    def rate(numerator, denominator):
        return {**support(numerator), **estimate(denominator, numerator.astype(float))}

    def as_mask(indices):
        mask = np.zeros(len(pairs), dtype=bool)
        mask[indices] = True
        return mask

    start_ids = [sample_id(pair.start) for pair in pairs]
    next_ids = [sample_id(pair.followup) if pair.followup else None for pair in pairs]
    p_t = np.asarray([probabilities.get(key, np.nan) for key in start_ids])
    p_u = np.asarray([probabilities.get(key, np.nan) for key in next_ids])
    y_t = np.asarray([labels.get(key, (None, 'missing'))[0] for key in start_ids], dtype=float)
    y_u = np.asarray([labels.get(key, (None, 'missing'))[0] for key in next_ids], dtype=float)
    regions = {key: decide(p, study.LOSS).region.value for key, p in probabilities.items()}
    d_t = np.asarray([regions.get(key, '') for key in start_ids])
    d_u = np.asarray([regions.get(key, '') for key in next_ids])
    initial = np.isfinite(p_t) & np.isfinite(y_t)
    opportunity = np.asarray([key is not None for key in next_ids])
    predicted = opportunity & np.isfinite(p_u)
    valid = predicted & np.isfinite(y_u)
    decided = predicted & (d_u != Region.BND.value)
    stale, updated = (p_t - y_u) ** 2, (p_u - y_u) ** 2
    report = {}
    for region in Region:
        starts = initial & (d_t == region.value)
        i, o, p, v = starts, starts & opportunity, starts & predicted, starts & valid
        no_next = i & ~opportunity
        end_cases = defaultdict(list)
        strata = {name: defaultdict(list) for name in ('phase', 'day_gap', 'step_gap', 'alive_wolves')}
        for j in np.flatnonzero(no_next):
            pair = pairs[j]
            deaths = ('both_dead' if pair.wolf_death is not None and pair.observer_death is not None else
                      'wolf_dead' if pair.wolf_death is not None else
                      'observer_dead' if pair.observer_death is not None else 'no_recorded_death')
            terminal = 'terminal_confirmed' if pair.terminal is not None else 'terminal_unknown'
            end_cases[f'{deaths}/{terminal}'].append(j)
        for j in np.flatnonzero(o):
            start, followup = pairs[j].start, pairs[j].followup
            strata['phase'][f'{start.phase}/{followup.phase}'].append(j)
            strata['day_gap'][str(followup.day - start.day)].append(j)
            strata['step_gap'][str(followup.step_index - start.step_index)].append(j)
            strata['alive_wolves'][f'{start.wolf_count}/{followup.wolf_count}'].append(j)

        strata_support = {}
        for field, groups in strata.items():
            strata_support[field] = {}
            for key, js in sorted(groups.items()):
                mask = as_mask(js)
                strata_support[field][key] = {'O': support(mask), 'P': support(mask & p), 'V': support(mask & v)}
        report[region.value] = {
            'population': {name: support(mask) for name, mask in zip(('I', 'O', 'P', 'V'), (i, o, p, v), strict=True)},
            'no_next': {'total': support(no_next), 'by_death_and_terminal':
                        {key: support(as_mask(js)) for key, js in sorted(end_cases.items())}},
            'state_transition_given_P': {dest.value: rate(p & (d_u == dest.value), p) for dest in Region},
            'label_transition_given_V': {f'{old}/{new}': rate(v & (y_t == old) & (y_u == new), v)
                                         for old in (0, 1) for new in (0, 1)},
            'next_opportunity_strata': strata_support,
            'h1': {'support': support(v), 'stale_brier': estimate(v, stale), 'updated_brier': estimate(v, updated),
                   'delta_pred': estimate(v, stale - updated, bootstrap=True)},
        }
        if region is not Region.BND:
            continue

        deferred, binary, three = np.zeros((3, len(pairs)))
        for j in np.flatnonzero(v):
            y = int(y_u[j])
            deferred[j] = realized_loss(Region.BND, y, study.LOSS)
            binary[j] = realized_loss(forced_binary(p_u[j], study.LOSS, tie_policy=study.BINARY_TIE_POLICY), y, study.LOSS)
            three[j] = realized_loss(Region(d_u[j]), y, study.LOSS)
        errors = ((d_u == Region.POS.value) & (y_u == 0)) | ((d_u == Region.NEG.value) & (y_u == 1))
        decided_valid = v & decided
        q, e = rate(decided_valid, v), rate(decided_valid & errors, decided_valid)
        delta = estimate(v, deferred - three, bootstrap=True)
        identity = None
        if q['value'] is not None:
            identity = q['value'] * (float(study.LOSS.b0) - float(study.LOSS.c_fp) * (e['value'] or 0))
            if not math.isclose(delta['value'], identity, rel_tol=0, abs_tol=study.SCORE_ATOL):
                raise ValueError('H2 六损失恒等式核对失败')

        report[region.value]['resolution'] = {
            'occurrence_given_I': rate(p & decided, i), 'resolved_given_P': rate(p & decided, p),
            'persistent_given_P': rate(p & ~decided, p),
        }
        report[region.value]['h2'] = {
            'support': support(v), 'q_given_V': q, 'e_given_decided': e,
            'directions': {dest.value: {
                'support': support(v & (d_u == dest.value)),
                'error': rate(v & (d_u == dest.value) & errors, v & (d_u == dest.value)),
            } for dest in (Region.POS, Region.NEG)},
            'losses': {'continue_BND': estimate(v, deferred), 'updated_binary': estimate(v, binary),
                       'updated_three_way': estimate(v, three)},
            'delta_dec': delta, 'identity_value': identity,
            'binary_minus_three_way': estimate(v, binary - three, bootstrap=True),
        }

    return {'start_valid': support(initial),
            'start_prediction_invalid': support(~np.isfinite(p_t)),
            'start_label_invalid': support(~np.isfinite(y_t)), 'by_initial_region': report}


def main():
    try:
        saved = study.open_fitted_models()
        pairs, probabilities, labels, diagnostics, reproduction = load_fold4(saved)
    except ValueError as exc:
        print(json.dumps({'status': 'static_reproduction_failed', 'load_error': str(exc)}, ensure_ascii=False))
        return 1
    receipt = {'scope': 'development_fold4_exploratory_sequential', 'publication_digest': fit.PUBLICATION_DIGEST,
               'split_digest': fit.SPLIT_DIGEST, 'fit_digest': study.FIT_DIGEST, 'q_digest': fit.Q_DIGEST,
               'fold_manifest_digest': saved['development_fold_manifest_digest'],
               'loss': study.LOSS.to_record(), 'binary_tie_policy': study.BINARY_TIE_POLICY.value,
               'bootstrap_seed': fit.BOOTSTRAP_SEED, 'bootstrap_replicates': fit.BOOTSTRAP_REPLICATES,
               'static_population': diagnostics, 'm1_reproduction': reproduction}
    if reproduction is None:
        print(json.dumps({**receipt, 'status': 'static_reproduction_failed'}, ensure_ascii=False))
        return 1
    print(json.dumps({**receipt, 'status': 'evaluated', **summarize(pairs, probabilities, labels)}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
