"""三个独立模型 Run 的唯一评价入口；保留原科学计分与配对统计。"""

import csv
from io import StringIO
from itertools import combinations

from werewolf.artifact_io import canonical_json_bytes,publish_artifact
from werewolf.tom_llm.common import digest,require,require_digest
from werewolf.tom_llm.experiment import VERSION,open_run,safe_path
from werewolf.tom_llm.live_runner import read_predictions
from werewolf.tom_llm.multimodel import source_identity
from werewolf.tom_llm.evaluation import compare,compare_sets
from werewolf.tom_llm.multimodel_evaluation import paired_extras,nonuniform_prediction_rows,reference_binding,validate_public_coverage
from werewolf.tom_llm.reference import open_reference
from werewolf.tom.protocol import bootstrap_indices
from werewolf.tom.scoring import paired_summary
from werewolf.tom.evaluation import score_prediction_rows
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
