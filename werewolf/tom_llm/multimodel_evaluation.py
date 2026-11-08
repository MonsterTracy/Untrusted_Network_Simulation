"""共享输入核对和补充概率统计；不包含独立的 Mock 执行链。"""
from werewolf.artifact_io import sha256_bytes
from werewolf.tom.evaluation import score_prediction_rows
from werewolf.tom.protocol import bootstrap_indices
from werewolf.tom.scoring import paired_summary
from werewolf.tom_llm.common import digest, identity, require
from werewolf.tom_llm.task import validate_task
from werewolf.tom_llm.policy import POLICY


def reference_binding(reference):
    return {'scope':'historical_v1_explicit_day_phase', 'seal_digest':reference.seal_digest,
            'publication_digest':reference.publication_digest, 'fold_manifest_digest':reference.fold_manifest_digest,
            'folds_digest':digest(reference.folds), 'prediction_digests':reference.prediction_digests,
            'bootstrap_digest':reference.bootstrap_digest}


def validate_public_coverage(tasks, reference, games):
    require(tasks and len({identity(t) for t in tasks}) == len(tasks), 'duplicate/empty public cohort')
    for task in tasks:
        validate_task(task)
    expected = {identity(r): r for r in reference.rows if r['observer_alive'] and r['game_id'] in set(games)}
    require(set(expected) == {identity(t) for t in tasks}, 'public-alive cohort mismatch; no intersection')
    require({t['game_id'] for t in tasks} == set(games), 'selection game coverage mismatch')
    for task in tasks:
        ref = expected[identity(task)]
        require(task['plan_digest'] == ref['plan_digest'] and task['fold'] == ref['fold'], 'cohort PRE/fold mismatch')


def nonuniform_prediction_rows(reference, predictions, games):
    refs = [r for r in reference.selected(games) if sum(q > 0 for q in r['q']) < 6]
    indexed = {identity(r):r for r in predictions}
    llm = [{**r,'probability':indexed[identity(r)]['prediction']['probability'],
            'non_self_log_probability':indexed[identity(r)]['prediction']['non_self_log_probability']} for r in refs]
    return refs,llm


def paired_extras(reference, predictions, games, result):
    cells = result['cells']
    indices = (reference.bootstrap_indices if tuple(games) == reference.game_ids else
               bootstrap_indices(list(games),POLICY['bootstrap']['replicates'],POLICY['bootstrap']['seed']))
    tv = paired_summary([cells['llm']['game_scores'][g]['total_variation'] for g in games],
                        [cells['qwen3_oof']['game_scores'][g]['total_variation'] for g in games],indices,.95)
    nu_games = result['nu_game_ids']
    if not nu_games:
        return {'Delta_TV':tv,'Delta_Top1_NU':None}
    refs,llm = nonuniform_prediction_rows(reference,predictions,games)
    _, left = score_prediction_rows(llm,nu_games)
    _, right = score_prediction_rows(refs,nu_games)
    nu_indices = bootstrap_indices(nu_games,POLICY['bootstrap']['replicates'],POLICY['bootstrap']['seed'])
    require(sha256_bytes(nu_indices.tobytes()) == result['nu_bootstrap_indices_digest'], 'NU bootstrap mismatch')
    nu = paired_summary([left[g]['argmax_in_target_support'] for g in nu_games],
                        [right[g]['argmax_in_target_support'] for g in nu_games],nu_indices,.95)
    return {'Delta_TV':tv,'Delta_Top1_NU':nu}
