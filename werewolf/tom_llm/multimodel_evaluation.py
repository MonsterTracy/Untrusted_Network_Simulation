"""Shared Selection/OOF cohort binding and sealed model comparisons. Mock only."""

from pathlib import Path

from werewolf.artifact_io import (publish_artifact, verify_artifact, canonical_json_bytes,
                                  canonical_jsonl_bytes, sha256_bytes)
from werewolf.tom.evaluation import score_prediction_rows
from werewolf.tom.protocol import bootstrap_indices
from werewolf.tom.scoring import paired_summary
from werewolf.tom_llm.common import digest, identity, require, strict_json
from werewolf.tom_llm.evaluation import compare
from werewolf.tom_llm.model_spec import registry
from werewolf.tom_llm.multimodel import (COHORT_VERSION, VERSION, source_identity, read_cohort,
                                        open_model_predictions, open_model_mock)
from werewolf.tom_llm.policy import POLICY
from werewolf.tom_llm.selection_binding import open_selection_binding
from werewolf.tom_llm.task import task_binding, validate_task


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


def publish_mock_cohort(destination, tasks, reference, *, selection_root, expected, provenance, prepared_manifest):
    """Tasks were independently constructed from public PREs; labels never enter this artifact."""
    selection = open_selection_binding(selection_root,reference.folds,expected=expected,
        provenance=provenance,prepared_manifest=prepared_manifest)
    require(selection['publication_digest'] == reference.publication_digest
            and selection['fold_manifest_digest'] == reference.fold_manifest_digest,
            'Selection/Reference publication mismatch')
    raw_selection = strict_json((Path(selection_root)/'selection.json').read_bytes())
    games = raw_selection['game_ids']
    validate_public_coverage(tasks,reference,games)
    return publish_artifact(destination,manifest_fields={
        'artifact_type':'llm_multimodel_public_cohort','schema_version':COHORT_VERSION,'engineering_only':True,
        'task':task_binding(),'source':source_identity(), 'reference':reference_binding(reference),
        'selection':{**selection,'game_ids':games},'selection_root':str(Path(selection_root).absolute()),
        'requests_digest':digest(tasks),'row_identity_digest':digest([identity(t) for t in tasks])},
        files={'requests.jsonl':canonical_jsonl_bytes(tasks)})


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


def compare_models(reference, cohort_root, *, expected_cohort_digest, runs, expected_seal_digests):
    """Open each complete model seal; never accept anonymous, mixed or partial rows."""
    cohort,tasks = read_cohort(cohort_root,expected_cohort_digest)
    require(cohort.manifest['reference'] == reference_binding(reference), 'shared OOF Reference drift')
    require(set(runs) == set(expected_seal_digests) == set(registry()), 'exact three-model comparison required')
    games = tuple(cohort.manifest['selection']['game_ids'])
    validate_public_coverage(tasks,reference,games)
    results, anchors = {}, {}
    for key,(root,execution) in runs.items():
        require(execution.spec.key == key, 'model result label/spec mismatch')
        contract,_,_ = open_model_mock(root,execution)
        seal,rows = open_model_predictions(root,execution)
        require(seal.manifest_digest == expected_seal_digests[key]
                and seal.manifest['cohort_digest'] == cohort.manifest_digest, 'model seal/cohort anchor mismatch')
        require([r['task_digest'] for r in rows] == [digest(t) for t in tasks], 'model task sequence mismatch')
        result = compare(reference,rows,games)
        result.update(paired_extras(reference,rows,games,result))
        result['limitations'] = [x for x in POLICY['limitations']
                                if key == 'qwen3' or x != 'qwen_family_related_to_generator']
        result['paired_difference_direction'] = 'LLM_minus_Qwen3_block_Observer_ToM'
        results[key] = result
        anchors[key] = {'seal_digest':seal.manifest_digest,'model_spec_digest':execution.spec.digest,
                        'execution_digest':contract.manifest['execution_digest']}
    require(len({r['bootstrap_indices_digest'] for r in results.values()}) == 1
            and len({r['nu_bootstrap_indices_digest'] for r in results.values()}) == 1,
            'cross-model bootstrap identities differ')
    return {'schema_version':VERSION+'_comparison','engineering_only':True,'formal_result':False,
            'cohort_digest':cohort.manifest_digest,'selection':cohort.manifest['selection'],
            'reference':reference_binding(reference),'models':anchors,'comparisons':results}
