"""三个独立模型 Run 的唯一评价入口；保留原科学计分与配对统计。"""

from collections import Counter
import csv
from io import StringIO
from itertools import combinations

from werewolf.artifact_io import (
    canonical_json_bytes, canonical_jsonl_bytes, open_artifact_envelope,
    publish_artifact, read_artifact_file, sha256_bytes,
)
from werewolf.tom_llm.common import digest, identity, require, require_digest, strict_json, valid_distribution
from werewolf.tom_llm.experiment import VERSION, open_run, require_formal_cohort, safe_path
from werewolf.tom_llm.live_runner import read_predictions
from werewolf.tom_llm.multimodel import COHORT_VERSION, source_identity
from werewolf.tom_llm.evaluation import compare, compare_sets
from werewolf.tom_llm.multimodel_evaluation import paired_extras,nonuniform_prediction_rows,reference_binding,validate_public_coverage
from werewolf.tom_llm.reference import open_reference
from werewolf.tom.protocol import bootstrap_indices
from werewolf.tom.scoring import paired_summary
from werewolf.tom.evaluation import probability_cell, predict_structured_observers, score_prediction_rows, uniform_prediction_rows
from werewolf.tom_llm.policy import POLICY

def compute(reference,run,tasks,predictions):
    require(run['reference']==reference_binding(reference),'analysis Reference binding drift')
    require(set(predictions)==set(run['models']),'all planned models required; no success-only subset')
    games=run['games'];validate_public_coverage(tasks,reference,games)
    results={};nu={}
    for key,rows in predictions.items():
        require([r['task_digest'] for r in rows]==run['task_digests'],'prediction task sequence differs')
        r=compare(reference,rows,games);r.update(paired_extras(reference,rows,games,r))
        if key!='qwen3':r['limitations']=[s for s in r['limitations'] if s!='qwen_family_related_to_generator']
        results[key]=r;nu_games=r['nu_game_ids']
        _,nu_rows=nonuniform_prediction_rows(reference,rows,games)
        nu[key]=score_prediction_rows(nu_rows,nu_games)[1] if nu_games else {}
    require(len({r['primary_row_identity_digest'] for r in results.values()})==1
            and len({r['bootstrap_indices_digest'] for r in results.values()})==1
            and len({r['nu_bootstrap_indices_digest'] for r in results.values()})==1,'cross-model evaluation populations differ')
    indices=reference.bootstrap_indices if tuple(games)==reference.game_ids else bootstrap_indices(games,POLICY['bootstrap']['replicates'],POLICY['bootstrap']['seed'])
    nu_indices=bootstrap_indices(nu_games,POLICY['bootstrap']['replicates'],POLICY['bootstrap']['seed']) if nu_games else None
    pairs=[]
    for left,right in combinations(run['model_order'],2):
        cells=[results[k]['cells']['llm']['game_scores'] for k in (left,right)]
        row={'left':left,'right':right,'direction':'left_minus_right'}
        for label,metric in [('Delta_KL','kl'),('Delta_TV','total_variation')]:
            row[label]=paired_summary([cells[0][g][metric] for g in games],[cells[1][g][metric] for g in games],indices,.95)
        row['Delta_Top1_NU']=None if not nu_games else paired_summary([nu[left][g]['argmax_in_target_support'] for g in nu_games],
            [nu[right][g]['argmax_in_target_support'] for g in nu_games],nu_indices,.95)
        pairs.append(row)
    result = {'formal_result':run['mode']=='formal','mode':run['mode'],'run_id':run['run_id'],
        'game_count':len(games),'selection':run['selection'],'reference':run['reference'],'source':run['source'],
        'protocol':run['protocol'],'models':results,'llm_pairwise':pairs,
        'direction':'LLM_minus_Qwen3_block_Observer_ToM; pairwise left_minus_right; lower KL/TV and higher Top-1_NU are better'}
    from werewolf.tom_llm.reference import raw_set_labels
    result['sets'] = compare_sets(reference,predictions,games,raw_set_labels(reference,run['publication'],games),
        engineering_only=run['mode']=='mock')
    result['sets']['formal_result'] = run['mode']=='formal'
    return result


def csv_bytes(rows):
    stream = StringIO()
    writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def report_files(result):
    """报告以集合终点为主，概率评价为补充；不生成额外图形缓存。"""
    metrics = []
    differences = []
    for population,group in result['sets']['populations'].items():
        for model,cell in group['cells'].items():
            for metric,value in cell['metrics'].items():
                metrics.append({'model':model,'population':population,'metric':metric,
                    'point':None if value is None else value['point'],
                    'ci95_low':None if value is None else value['interval'][0],
                    'ci95_high':None if value is None else value['interval'][1],
                    'games':cell['game_count'],'rows':cell['row_count'],'tier':group['reporting_tier']})
        for pair in group['paired']:
            for metric,value in pair['metrics'].items():
                differences.append({'population':population,'left':pair['left'],'right':pair['right'],
                    'metric':metric,'point':None if value is None else value['point'],
                    'ci95_low':None if value is None else value['interval'][0],
                    'ci95_high':None if value is None else value['interval'][1],'tier':pair['tier'],
                    'direction':'left_minus_right'})
    supplementary = []
    first = next(iter(result['models'].values()))
    cells = {k:r['cells']['llm'] for k,r in result['models'].items()}
    cells.update({'Observer-ToM':first['cells']['qwen3_oof'],'Uniform':first['cells']['uniform_non_self']})
    for model,cell in cells.items():
        for metric in ('kl','tv','top1_nu'):
            value = cell[metric]
            supplementary.append({'model':model,'metric':metric,'point':None if value is None else value['point'],
                'ci95_low':None if value is None else value['interval'][0],
                'ci95_high':None if value is None else value['interval'][1],
                'tier':'supplementary_exploratory'})
    marker = 'FORMAL' if result['formal_result'] else 'CPU/MOCK ONLY'
    lines = ['# TomBaseline 怀疑集合比较','',marker,'',
        '主要描述性终点：Full Primary Set-F1。NU 中加入固定 OOF 投影。',
        '所有非法输出保留统一分母；缺失预测阻断报告。游戏等权，10,000 次配对游戏 Bootstrap。',
        '区间是点态描述／探索区间，不支持未经多重性控制的联合优越性声明。',
        '150 局已被观察；V1/V2 差异不是输出表示改变的因果效应。','',
        '| Model | Full Set-F1 | 95% CI |','|---|---:|---|']
    for row in metrics:
        if row['population']=='full_primary_raw' and row['metric']=='set_f1':
            lines.append(f"| {row['model']} | {row['point']} | [{row['ci95_low']}, {row['ci95_high']}] |")
    lines += ['', '执行协议：'+result['protocol']['schema_version'],
        '执行 Source：'+result['source']['source_revision'],
        '计时仅为收到并落盘响应的客户端请求耗时；不含模型启动和停止，也不声称纯 GPU 速度。',
        '完整计分、逐局统计、失败原因与身份见 analysis.json。']
    perf = [{'model':k,**r['performance']} for k,r in result['models'].items()]
    return {'analysis.json':canonical_json_bytes(result),'metrics.csv':csv_bytes(metrics),
        'paired.csv':csv_bytes(differences),'supplementary.csv':csv_bytes(supplementary),
        'performance.csv':csv_bytes(perf),'report.md':('\n'.join(lines)+'\n').encode()}


def analyze_cross_runs(sealed_runs, destination):
    """仅接收三个独立封存 Run；一次读取共享输入，禁止成功子集分析。"""
    keys = ('qwen3','yi','mistral')
    require(type(sealed_runs) is dict and set(sealed_runs)==set(keys),'exactly three model runs required')
    roots = set()
    run_ids = set()
    first = None
    predictions = {}
    inputs = {}
    configurations = {}
    seals = {}
    for key in keys:
        item = sealed_runs[key]
        require(type(item) is dict and set(item)=={'path','digest'},'external prediction Seal digest required')
        require_digest(item['digest'])
        root = safe_path(item['path'])
        require(root not in roots,'distinct Run roots required')
        roots.add(root)
        run = open_run(root,with_tasks=first is None)
        require(run.key==key and run.record['run_id'] not in run_ids,'model/Run identity mismatch')
        run_ids.add(run.record['run_id'])
        if first is None:
            first = run
        else:
            for field in ('source','mode','protocol','cohort_root','cohort_digest','selection','reference','games',
                          'game_count','request_count','task_digests','row_identity_digest','publication','evaluation'):
                require(run.record[field]==first.record[field],'cross-run '+field+' mismatch')
            run.tasks = first.tasks
        seal,rows = read_predictions(run,item['digest'])
        predictions[key] = rows
        seals[key] = seal
        configurations[key] = run.record
        inputs[key] = {'path':str(root),'run_id':run.record['run_id'],
                      'contract_digest':run.artifact.manifest_digest,'prediction_seal_digest':seal.manifest_digest}
    context = {**first.record,'run_id':None,
        'models':{k:configurations[k]['models'][k] for k in keys},'model_order':list(keys)}
    reference = open_reference(context['publication'],context['evaluation'],
                               expected_seal_digest=context['reference']['seal_digest'])
    result = compute(reference,context,first.tasks,predictions)
    result.update(independent_runs=inputs,experiment_configurations=configurations,analysis_source=source_identity())
    for key,rows in predictions.items():
        summary = seals[key].manifest['summary']
        result['models'][key]['performance'] = {
            'completed_requests':len(rows),'http_attempts':summary['attempts'],
            'valid_prediction_rate':summary['valid_rate'],'received_request_seconds':summary['elapsed_seconds'],
            'input_tokens':sum(r['usage']['input_tokens'] for r in rows),
            'output_tokens':sum(r['usage']['output_tokens'] for r in rows)}
    dest = safe_path(destination)
    require(not dest.exists(),'report output exists; no overwrite')
    if result['formal_result']:
        from werewolf.tom_llm.deployment import baseline_root
        baseline_root(dest)
    return publish_artifact(dest,manifest_fields={'artifact_type':'tom_baseline_analysis','schema_version':VERSION,
        'prediction_seals':{k:a.manifest_digest for k,a in seals.items()},
        'formal_result':result['formal_result'],'protocol_digest':digest(result['protocol'])},files=report_files(result))


# 已核验的三个历史 Run Seal 锚定各自的 Contract 与 Prediction Seal。
V1_RUN_SEALS = {
    "qwen3": "a59c658d60d2c96a8f0446803e8f07a06a58d85ad5294b0b62201ed21534901e",
    "yi": "b65ed863786668550e7ebed9663d44d235aee1f3c2d4fdab1ccde9eb8f260bf4",
    "mistral": "d8a191a514e9f74a3b19dbfd2fd2ec8434e85a067ca6a23e931a9e46288dfadb",
}
V1_SOURCE_REVISION = "98d8e038c86c35b87490c433acb9d97caf886c2d"
V1_EXPERIMENT = "classic7_multimodel_experiment_v1"
V1_COHORT_DIGEST = "7f6b7f4f13478f5e0d805503758d9a98f29c307ce298d53191ac81fdf5b3b4b8"
V1_PRIMARY_ROWS_DIGEST = "dddcf67133864986320ff5b2d47db7cf5da201220b6ba59e759ba1d88cc8921a"
V1_BOOTSTRAP_DIGEST = "fd5cfc1829e5bade1075179a4c224c99f3b1bb6fd33b2658ca94d6acb549efcf"
V1_PRE_COUNT, V1_PRIMARY_COUNT, V1_NU_COUNT = 1886, 7480, 5502
ROW_FIELDS = ("game_id", "boundary_id", "prefix_digest", "plan_digest", "observer", "fold")
PROBABILITY_FIELDS = ("probability", "non_self_log_probability")


def _read_v1_record(path, kind, expected):
    """消费冻结的 V1 record.json，校验外部摘要并保留原始数值。"""
    artifact = open_artifact_envelope(path, expected_artifact_type=kind, expected_schema_version=V1_EXPERIMENT)
    manifest = artifact.manifest
    require(artifact.manifest_digest == expected and set(manifest["file_table"]) == {"record.json"},
            f"historical V1 {kind} seal/inventory mismatch")
    require(manifest["record_digest"] == manifest["file_table"]["record.json"]["sha256"],
            f"historical V1 {kind} record digest mismatch")
    return strict_json(read_artifact_file(artifact, "record.json"))


def _read_v1_comparison_inputs(run_paths):
    """按已验证的 Seal 读取公开 Cohort 和三份预测，供预测及评分阶段各取所需。"""
    from werewolf.tom_llm.task import VERSION as V1_TASK

    require(set(run_paths) == set(V1_RUN_SEALS), "three historical V1 Run paths required")
    context = None
    predictions, provenance = {}, {}
    for model, seal_digest in V1_RUN_SEALS.items():
        root = safe_path(run_paths[model])
        seal = _read_v1_record(root / "seal", "mm_baseline_run_seal", seal_digest)
        run = _read_v1_record(root / "contract", "mm_baseline_experiment", seal["run_digest"])
        require(seal["mode"] == run["mode"] == "formal" and set(seal["models"]) == set(run["models"]) == {model}
                and run["source"] == seal["source"] and run["source"]["source_revision"] == V1_SOURCE_REVISION
                and run["protocol"]["schema_version"] == V1_EXPERIMENT, f"historical V1 {model} Run identity mismatch")
        shared = {field: run[field] for field in ("cohort_root", "cohort_digest", "selection", "reference",
                  "publication", "evaluation", "games", "game_count", "request_count", "row_identity_digest", "task_digests")}
        require(context is None or shared == context, f"historical V1 {model} shared population mismatch")
        context = shared
        prediction_digest = seal["models"][model]
        prediction = _read_v1_record(root / "models" / model / "seal", "mm_baseline_predictions", prediction_digest)
        require(prediction["run_digest"] == seal["run_digest"] and prediction["model"] == model
                and prediction["mode"] == "formal" and prediction["formal_result"] is True
                and prediction["model_spec_digest"] == run["models"][model]
                and prediction["task_digests"] == run["task_digests"], f"historical V1 {model} prediction binding mismatch")
        predictions[model] = prediction["rows"]
        provenance[model] = {"path": str(root), "run_id": run["run_id"], "contract_digest": seal["run_digest"],
                             "run_seal_digest": seal_digest, "prediction_seal_digest": prediction_digest,
                             "model_spec_digest": run["models"][model], "execution_source": run["source"]}

    cohort = open_artifact_envelope(safe_path(context["cohort_root"]),
        expected_artifact_type="llm_multimodel_public_cohort", expected_schema_version=COHORT_VERSION)
    manifest = cohort.manifest
    require(cohort.manifest_digest == context["cohort_digest"] == V1_COHORT_DIGEST
            and manifest["task"]["schema_version"] == V1_TASK and set(manifest["file_table"]) == {"requests.jsonl"}
            and all(manifest[k] == context[k] for k in ("selection", "reference", "row_identity_digest")),
            "historical V1 Cohort identity mismatch")
    tasks = [strict_json(line) for line in read_artifact_file(cohort, "requests.jsonl").splitlines()]
    require_formal_cohort(manifest, tasks)
    keys = [identity(task) for task in tasks]
    require(len(set(keys)) == len(tasks) == context["request_count"]
            and sorted({task["game_id"] for task in tasks}) == context["games"] == sorted(manifest["selection"]["game_ids"])
            and context["game_count"] == len(context["games"])
            and digest(keys) == context["row_identity_digest"]
            and digest(tasks) == manifest["requests_digest"]
            and [digest(task) for task in tasks] == context["task_digests"], "historical V1 ordered request coverage mismatch")
    for model, rows in predictions.items():
        require(len(rows) == len(tasks), f"historical V1 {model} missing or extra prediction rows")
        for row, task, task_digest in zip(rows, tasks, context["task_digests"], strict=True):
            require(all(row[k] == task[k] for k in ROW_FIELDS) and row["task_digest"] == task_digest
                    and row["canonical_shift"] == 0 and row["temporal_condition"] == "explicit_day_phase",
                    f"historical V1 {model} prediction row identity mismatch")
            p = row["prediction"]
            require(p["observer"] == row["observer"] and p["status"] in ("valid", "fallback_uniform"),
                    f"historical V1 {model} prediction seat/status mismatch")
            valid_distribution(p)
    return context, tasks, predictions, provenance


def _predict_epoch35(tasks, task_digests, model, device):
    """仅从公开计划构造张量，每个 PRE 前向一次，按请求顺序回填观察者。"""
    from werewolf.structured_history import StructuredTokenDescriptor, StructuredTokenPlan
    from werewolf.tom.makto import EPOCH35_DIGEST
    from werewolf.tom_llm.prompt import validate_plan

    records = {}
    for task in tasks:
        require(set(task) == {*ROW_FIELDS, "public_plan", "messages", "response_schema"},
                "historical V1 prediction inputs must be public tasks")
        key = task["game_id"], task["boundary_id"]
        record = task["public_plan"]
        require(all(record[k] == task[k] for k in ("prefix_digest", "plan_digest"))
                and records.setdefault(key, record) == record, f"inconsistent public plan for PRE {key}")
    require(len(records) == V1_PRE_COUNT, "historical V1 PRE count mismatch")
    plans = {}
    for key, record in records.items():
        plan = validate_plan(StructuredTokenPlan(**{**record,
            "tokens": tuple(StructuredTokenDescriptor(**token) for token in record["tokens"])}))
        require(plan.token_count <= model.capacity.max_seq_len
                and all(token.day < len(model.temporal.day) for token in plan.tokens),
                f"Epoch35 capacity/Temporal mismatch for PRE {key}")
        plans[key] = plan

    row_predictions = {key: predict_structured_observers(plan, model, device) for key, plan in plans.items()}

    rows = []
    for task, task_digest in zip(tasks, task_digests, strict=True):
        probability, logp = row_predictions[task["game_id"], task["boundary_id"]]
        seat = task["observer"]
        rows.append({**{k: task[k] for k in ROW_FIELDS}, "task_digest": task_digest, "canonical_shift": 0,
                     "checkpoint_digest": EPOCH35_DIGEST, "temporal_condition": "explicit_day_phase",
                     "probability": probability[seat], "non_self_log_probability": logp[seat]})
    return rows


def _score_epoch35_v1(reference, games, tasks, large_rows, v1_rows):
    """共用同一原始 q、观察者行和游戏级 Bootstrap。"""
    from werewolf.tom_llm.multimodel_evaluation import validate_reference_coverage

    validate_reference_coverage(tasks, reference, games)
    require(len(large_rows) == len(tasks) and all(all(row[k] == task[k] for k in ROW_FIELDS)
            for row, task in zip(large_rows, tasks, strict=True)), "Epoch35 ordered prediction coverage mismatch")
    primary = reference.selected(games)
    primary_keys = [identity(row) for row in primary]
    nu_games = sorted({row["game_id"] for row in primary if sum(q > 0 for q in row["q"]) < 6})
    require(len(primary) == V1_PRIMARY_COUNT and digest(primary_keys) == V1_PRIMARY_ROWS_DIGEST
            and sum(sum(q > 0 for q in row["q"]) < 6 for row in primary) == V1_NU_COUNT
            and nu_games == games, "historical V1 Primary/NU population mismatch")

    positions = {identity(row): i for i, row in enumerate(tasks)}
    selected = [positions[key] for key in primary_keys]
    sampling = POLICY["bootstrap"]
    indices = bootstrap_indices(games, sampling["replicates"], sampling["seed"])
    indices_digest = sha256_bytes(indices.tobytes())
    require(indices_digest == V1_BOOTSTRAP_DIGEST, "historical V1 Bootstrap identity mismatch")
    inputs = {"uniform_non_self": uniform_prediction_rows(primary), "historical_qwen3_oof": primary,
              "epoch35": [{**r, **{k: large_rows[i][k] for k in PROBABILITY_FIELDS}}
                          for r, i in zip(primary, selected, strict=True)]}
    failures = {}
    for key, rows in v1_rows.items():
        sealed = [row["prediction"] for row in rows]
        inputs[key] = [{**r, **{k: sealed[i][k] for k in PROBABILITY_FIELDS}}
                       for r, i in zip(primary, selected, strict=True)]
        failures[key] = {
            "public_alive_failure_types": dict(Counter(p["failure_type"] for p in sealed if p["status"] == "fallback_uniform")),
            "primary_failure_types": dict(Counter(sealed[i]["failure_type"] for i in selected
                                                 if sealed[i]["status"] == "fallback_uniform")),
        }
    cells, nu_scores = {}, {}
    for key, rows in inputs.items():
        cells[key], nu_scores[key] = probability_cell(rows, games, indices, nu_games, indices)
        cells[key]["nu_game_scores"] = nu_scores[key]
    paired = {}
    for key in v1_rows:
        llm, large = cells[key]["game_scores"], cells["epoch35"]["game_scores"]
        paired[key] = {
            "kl": paired_summary([llm[g]["kl"] for g in games], [large[g]["kl"] for g in games], indices, .95),
            "tv": paired_summary([llm[g]["total_variation"] for g in games],
                                 [large[g]["total_variation"] for g in games], indices, .95),
            "top1_nu": paired_summary([nu_scores["epoch35"][g] for g in nu_games],
                                      [nu_scores[key][g] for g in nu_games], indices, .95),
        }
    return {"game_ids": games, "public_request_count": len(tasks), "primary_row_count": len(primary),
            "primary_row_identity_digest": V1_PRIMARY_ROWS_DIGEST, "nu_row_count": V1_NU_COUNT,
            "nu_game_ids": nu_games, "bootstrap": {**sampling, "indices_digest": indices_digest},
            "cells": cells, "failure_counts": failures, "paired": paired,
            "paired_direction": {"kl": "LLM_minus_Epoch35", "tv": "LLM_minus_Epoch35",
                                 "top1_nu": "Epoch35_minus_LLM", "positive": "favors_Epoch35"}}


def analyze_epoch35_v1(run_paths, checkpoint, temporal, destination, *, device="cpu"):
    """先完成公开 PRE 预测，再打开历史 OOF 监督，原子发布一份配对结果。"""
    from werewolf.tom.makto import EPOCH35_DIGEST, load_epoch35_model

    dest = safe_path(destination)
    require(not dest.exists() and not any((parent / "manifest.json").exists()
            or (parent / "contract" / "manifest.json").exists()
            or (parent / "phase_codebook.manifest.json").exists() for parent in dest.parents),
            "output exists or is inside frozen evidence")
    context, tasks, v1_rows, historical_runs = _read_v1_comparison_inputs(run_paths)
    source = source_identity()
    model = load_epoch35_model(safe_path(checkpoint), safe_path(temporal)).to(device)
    large_rows = _predict_epoch35(tasks, context["task_digests"], model, device)
    model_identity = {"epoch": 35, "checkpoint_digest": EPOCH35_DIGEST, "graph": model.graph,
                      "temporal_artifact_digest": model.temporal.artifact_digest, "device": device}
    del model
    tasks = [{k: task[k] for k in ROW_FIELDS} for task in tasks]

    reference = open_reference(context["publication"], context["evaluation"],
                               expected_seal_digest=context["reference"]["seal_digest"])
    require(reference_binding(reference) == context["reference"], "historical OOF Reference binding mismatch")
    result = _score_epoch35_v1(reference, context["games"], tasks, large_rows, v1_rows)
    result.update(population="previously_observed_historical_150", cohort_digest=context["cohort_digest"],
                  public_row_identity_digest=context["row_identity_digest"], pre_count=V1_PRE_COUNT,
                  epoch35=model_identity, prediction_source=source, analysis_source=source,
                  v1_runs=historical_runs, v1_execution_protocol=V1_EXPERIMENT,
                  reference=context["reference"], historical_qwen3_oof=reference.provenance)
    return publish_artifact(dest, manifest_fields={
        "artifact_type": "epoch35_v1_paired_evaluation", "schema_version": "classic7_epoch35_v1_paired_evaluation_v1",
        "cohort_digest": context["cohort_digest"], "epoch35": model_identity,
    }, files={"predictions.jsonl": canonical_jsonl_bytes(large_rows), "analysis.json": canonical_json_bytes(result)})
