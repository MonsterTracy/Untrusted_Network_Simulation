"""Publication reports rebuilt exclusively from complete, verified prediction seals."""

from collections import Counter
import csv
from io import StringIO,BytesIO
from itertools import combinations
import html
import math
import os
from pathlib import Path
import tempfile

from werewolf.artifact_io import publish_artifact, canonical_json_bytes
from werewolf.tom_llm.common import digest, require
from werewolf.tom_llm.experiment import VERSION, protocol, open_run, read_record, safe_path
from werewolf.tom_llm.live_runner import read_predictions, owned
from werewolf.tom_llm.evaluation import compare
from werewolf.tom_llm.multimodel_evaluation import paired_extras, nonuniform_prediction_rows, reference_binding, validate_public_coverage
from werewolf.tom_llm.reference import open_reference
from werewolf.tom.protocol import bootstrap_indices
from werewolf.tom.scoring import paired_summary
from werewolf.tom.evaluation import score_prediction_rows
from werewolf.tom_llm.policy import POLICY


def performance(root,key,tasks,rows):
    contract,_,_=open_run(root,current_source=False);attempts=Counter();durations=[]
    for task in tasks:
        path=Path(root)/'models'/key/'requests'/digest(task)
        for i in (1,2):
            if (path/f'failure-{i}').exists():
                _,r=owned(path/f'failure-{i}',contract,key,task)
                attempts[r['outcome']]+=1;durations.append(r['elapsed_seconds'])
            elif (path/f'response-{i}').exists():
                _,r=owned(path/f'response-{i}',contract,key,task);durations.append(r['elapsed_seconds']);attempts['reply_received']+=1
    session_times=[];incomplete_sessions=0
    for session_root in (Path(root)/'models'/key/'sessions').iterdir():
        if session_root.name.endswith('-complete'):continue
        sa,sr=read_record(session_root,'mm_baseline_session')
        require(sr['run_digest']==contract.manifest_digest and sr['model']==key,'performance session ownership drift')
        completed=session_root.parent/(session_root.name+'-complete')
        if completed.exists():
            _,cr=read_record(completed,'mm_baseline_session_complete')
            require(cr['session_digest']==sa.manifest_digest and type(cr['elapsed_seconds']) in (int,float) and math.isfinite(cr['elapsed_seconds']) and cr['elapsed_seconds']>=0,'session duration drift')
            session_times.append(cr['elapsed_seconds'])
        else:incomplete_sessions+=1
    failures=Counter(r['prediction']['failure_type'] for r in rows if r['prediction']['status']=='fallback_uniform')
    wall=math.fsum(durations);out=sum(r['usage']['output_tokens'] for r in rows)
    return {'completed_requests':len(rows),'http_attempts':sum(attempts.values()),'attempt_outcomes':dict(attempts),
        'valid_predictions':len(rows)-sum(failures.values()),'valid_prediction_rate':1-sum(failures.values())/len(rows),
        'fallback_predictions':sum(failures.values()),'fallback_reasons':dict(failures),
        'legal_uniform_predictions':sum(r['prediction']['status']=='valid' and len(set(r['prediction']['raw_probability']))==1 for r in rows),
        'input_tokens':sum(r['usage']['input_tokens'] for r in rows),'output_tokens':out,
        'active_client_wall_seconds':wall,'output_tokens_per_client_wall_second':None if wall==0 else out/wall,
        'completed_request_loop_wall_seconds':sum(session_times),'sessions_without_end_marker':incomplete_sessions,
        'session_wall_is_total':incomplete_sessions==0,'ttft_seconds':None,'allocator_peak_bytes':None,'scope':protocol()['efficiency_scope'],
        'unknown_attempt_cost':'no unknown outcome can appear in a complete seal; blocked runs have unmeasured work, never zero-cost success'}


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
    return {'formal_result':run['mode']=='formal','mode':run['mode'],'run_id':run['run_id'],
        'game_count':len(games),'selection':run['selection'],'reference':run['reference'],'source':run['source'],
        'protocol':protocol(),'models':results,'llm_pairwise':pairs,
        'direction':'LLM_minus_Qwen3_block_Observer_ToM; pairwise left_minus_right; lower KL/TV and higher Top-1_NU are better'}


def csv_bytes(rows):
    stream=StringIO();writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
    writer.writeheader();writer.writerows(rows);return stream.getvalue().encode()


def report_files(result):
    metrics=[];differences=[];perf=[]
    first=next(iter(result['models'].values()))
    cells={k:r['cells']['llm'] for k,r in result['models'].items()}
    cells.update({'Qwen3_Observer_ToM':first['cells']['qwen3_oof'],'uniform':first['cells']['uniform_non_self']})
    for key,cell in cells.items():
        for label,metric in [('KL','kl'),('TV','tv'),('Top-1_NU','top1_nu')]:
            value=cell[metric]
            metrics.append({'model':key,'metric':label,'point':None if value is None else value['point'],
                'ci95_low':None if value is None else value['interval'][0],'ci95_high':None if value is None else value['interval'][1],
                'games':cell['nu_game_count'] if metric=='top1_nu' else cell['game_count'],
                'rows':cell['nu_row_count'] if metric=='top1_nu' else cell['row_count'],'mode':result['mode']})
    pair_rows=list(result['llm_pairwise'])+[{'left':k,'right':'Qwen3_Observer_ToM',**{n:r[n] for n in ('Delta_KL','Delta_TV','Delta_Top1_NU')}} for k,r in result['models'].items()]
    for pair in pair_rows:
        for name in ('Delta_KL','Delta_TV','Delta_Top1_NU'):
            v=pair[name];differences.append({'left':pair['left'],'right':pair['right'],'metric':name,
                'point':None if v is None else v['point'],'ci95_low':None if v is None else v['interval'][0],
                'ci95_high':None if v is None else v['interval'][1],'direction':'left_minus_right'})
    for key,r in result['models'].items():
        p=r['performance'];perf.append({'model':key,**{n:p[n] for n in ('completed_requests','http_attempts','valid_prediction_rate',
            'fallback_predictions','legal_uniform_predictions','input_tokens','output_tokens','active_client_wall_seconds',
            'completed_request_loop_wall_seconds','sessions_without_end_marker','session_wall_is_total','ttft_seconds','allocator_peak_bytes','output_tokens_per_client_wall_second')},'mode':result['mode']})
    marker='FORMAL' if result['formal_result'] else 'ENGINEERING ONLY — '+result['mode'].upper()
    lines=['# ToM Multimodel Baseline', '',marker,'',f"Run: {result['run_id']}; games: {result['game_count']}",
        '', 'Game-macro metrics, paired game bootstrap, 95% percentile intervals. All three predeclared metrics and fallbacks are retained.',
        '', '| Model | Metric | Point | 95% CI |','|---|---|---:|---|']
    for r in metrics:lines.append(f"| {r['model']} | {r['metric']} | {r['point']} | [{r['ci95_low']}, {r['ci95_high']}] |")
    lines+=['',f"Execution Source: {result['source']['source_revision']} / {digest(result['source'])}",
        f"Analysis Source: {result['analysis_source']['source_revision']} / {digest(result['analysis_source'])}",
        f"Selection artifact: {result['selection']['selection_artifact_manifest_digest']}",
        f"Reference seal: {result['reference']['seal_digest']}",
        *[f"Model {key}: {spec['model']} @ {spec['revision']}" for key,spec in result['model_specs'].items()],
        '', 'Differences: left minus right. Lower KL/TV and higher Top-1_NU are better.',
        '', 'Efficiency is per-attempt nonstreaming client wall, including request conversion, HTTP/service and confirmed-not-executed attempts. Request-loop wall additionally includes per-row validation/artifacts; startup, physical preparation, final sealing and analysis are excluded; interrupted sessions are disclosed as incomplete timing. TTFT, GPU allocator peak and Observer-ToM inference time were not measured; no pure-GPU or ToM speedup claim is supported.',
        '', 'Method-level comparison includes different pretraining/posttraining and explicit rules. Qwen family relation to the generator applies to Qwen. It is not an architecture-only ablation.',
        '', 'See metrics.csv, paired.csv, performance.csv and analysis.json for complete paired intervals, provenance, failure reasons and per-game scores.']
    markdown='\n'.join(lines)+'\n'
    # Matplotlib caches are scoped to a disposable report directory, never shared server caches.
    from matplotlib.figure import Figure
    fig=Figure(figsize=(12,4));axes=fig.subplots(1,3)
    for ax,(label,metric) in zip(axes,[('KL','kl'),('TV','tv'),('Top-1_NU','top1_nu')]):
        data=[(k,c[metric]) for k,c in cells.items() if c[metric] is not None]
        for i,(key,v) in enumerate(data):
            ax.barh(i,v['point']);ax.plot(v['interval'],[i,i],color='black')
        ax.set_yticks(range(len(data)),[k for k,v in data]);ax.set_title(label)
    fig.suptitle(marker);fig.tight_layout();stream=BytesIO();fig.savefig(stream,format='svg');metric_svg=stream.getvalue()
    fig=Figure(figsize=(8,4));ax=fig.subplots()
    ax.barh([r['model'] for r in perf],[r['active_client_wall_seconds'] for r in perf]);ax.set_xlabel('Per-attempt client wall seconds (not pure GPU time)')
    ax.set_title(marker);fig.tight_layout();stream=BytesIO();fig.savefig(stream,format='svg')
    page='<meta charset="utf-8"><title>ToM Baseline</title><pre>'+html.escape(markdown)+'</pre><img src="metrics.svg"><img src="efficiency.svg">'
    return {'analysis.json':canonical_json_bytes(result),'metrics.csv':csv_bytes(metrics),'paired.csv':csv_bytes(differences),
            'performance.csv':csv_bytes(perf),'report.md':markdown.encode(),'report.html':page.encode(),
            'metrics.svg':metric_svg,'efficiency.svg':stream.getvalue()}


def analyze(root,destination,*,expected_run_seal=None):
    contract,run,tasks=open_run(root,current_source=False)
    seal,final=read_record(Path(root)/'seal','mm_baseline_run_seal',expected_run_seal)
    require(final['run_digest']==contract.manifest_digest and final['mode']==run['mode'] and final['source']==run['source']
            and set(final['models'])==set(run['models']),'run seal model/source/coverage drift')
    predictions={};anchors={}
    for key,wanted in final['models'].items():
        a,rows=read_predictions(root,key,expected=wanted);predictions[key]=rows;anchors[key]=a.manifest_digest
    reference=open_reference(run['publication'],run['evaluation'],expected_seal_digest=run['reference']['seal_digest'])
    result=compute(reference,run,tasks,predictions);result['prediction_seals']=anchors
    from werewolf.tom_llm.experiment import source_v3
    result['analysis_source']=source_v3()
    result['experiment_configuration']=run;result['experiment_root']=str(Path(root).absolute())
    from werewolf.tom_llm.model_spec import ModelSpec
    result['model_specs']={key:ModelSpec(key).record() for key in run['models']}
    import importlib.metadata,sys
    result['analysis_environment']={'python':sys.version,'packages':{name:importlib.metadata.version(name)
        for name in ('torch','numpy','transformers','tokenizers','Jinja2','matplotlib')}}
    result['execution_sessions']={key:[read_record(p,'mm_baseline_session')[1]
        for p in sorted((Path(root)/'models'/key/'sessions').iterdir()) if not p.name.endswith('-complete')] for key in run['models']}
    for key in result['models']:result['models'][key]['performance']=performance(root,key,tasks,predictions[key])
    dest=safe_path(destination);require(not dest.exists(),'report output exists; no overwrite')
    if run['mode']!='fake':
        from werewolf.tom_llm.deployment import BASE
        require(dest.is_relative_to(BASE),'live report must remain inside independent b2 root')
    dest.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.report-cache-',dir=dest.parent) as temp:
        previous=os.environ.get('MPLCONFIGDIR');os.environ['MPLCONFIGDIR']=temp
        try:files=report_files(result)
        finally:
            if previous is None:os.environ.pop('MPLCONFIGDIR',None)
            else:os.environ['MPLCONFIGDIR']=previous
    return publish_artifact(dest,manifest_fields={'artifact_type':'mm_baseline_analysis','schema_version':VERSION,
        'run_seal_digest':seal.manifest_digest,'prediction_seals':anchors,'formal_result':run['mode']=='formal',
        'protocol_digest':digest(protocol())},files=files)
