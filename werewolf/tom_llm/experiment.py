"""单一集合 Baseline 的输入合同、执行协议和运行上下文。"""

from dataclasses import dataclass
from pathlib import Path
import re

from werewolf.artifact_io import publish_artifact, verify_artifact, canonical_json_bytes
from werewolf.tom_llm.common import digest, identity, require, strict_json
from werewolf.tom_llm.model_spec import ModelSpec, check_cpu_parity
from werewolf.tom_llm.multimodel import source_identity, read_cohort
from werewolf.tom_llm.task import SET_VERSION, task_binding

VERSION = 'classic7_suspicion_set_execution_v2'
APPROVED_PROTOCOL_VERSION = 'classic7_suspicion_set_research_protocol_v2'
FORMAL_PUBLIC_REQUESTS = 10714
FORMAL_PUBLIC_ROWS_DIGEST = '8fd129c892d9657ed429d984c83acc278792637a5e96e06eb9c148f3b98e80c8'


def protocol():
    """执行协议另立身份；研究定义仍指向已批准的集合方法。"""
    return {'schema_version':VERSION,'research_protocol':APPROVED_PROTOCOL_VERSION,
        'historical_research_digest':'e135168b1927c1542d7e0c991aaeb5fe9dc8814317b9316071dd1ffc655b71dc',
        'task':task_binding(SET_VERSION),'population':{'games':150,'public_requests':FORMAL_PUBLIC_REQUESTS,
        'row_identity_digest':FORMAL_PUBLIC_ROWS_DIGEST},
        'metrics':['Set-F1','Set-Precision','Set-Recall','Jaccard','Exact-Set-Match','Valid-Output-Rate'],
        'qualification_rounds':0,'physical_admission_required':False,'execution_permit_required':False,
        'model_loading':'one_owned_instance_per_model','automatic_request_retries':0,
        'unknown_outcome':'stop_without_resend','persistence':'append_only_hash_chained_journal',
        'weight_verification':'registered_revision_index_and_stat_not_repeated_full_sha',
        'checks':{'source':'start_and_seal','cohort':'once_per_loaded_context',
                  'native_tokens_and_budget':'once_when_preparing_run','status_stop':'owned_process_only'}}


def require_formal_cohort(manifest, tasks):
    """消费已经核验的 Cohort；折归属来自任务及 Selection，不虚构绑定字段。"""
    from werewolf.tom_llm.reference import fixed_oof_contract
    require(len(manifest['selection']['game_ids'])==150 and len(tasks)==FORMAL_PUBLIC_REQUESTS
            and manifest['selection']['scope']=='subset150'
            and all(len({t['game_id'] for t in tasks if t['fold']==f})==30 for f in range(5))
            and manifest['row_identity_digest']==FORMAL_PUBLIC_ROWS_DIGEST,
            'complete frozen 150/10714 cohort required')
    require(manifest['reference']==fixed_oof_contract()['reference'],'fixed OOF Reference required')


def safe_path(path):
    p = Path(path).absolute()
    require('..' not in p.parts and not any(q.is_symlink() for q in (p,*p.parents)),
            'absolute path without traversal/symlinks required')
    return p


def publish_record(path, kind, record):
    require(not safe_path(path).exists(),'artifact exists; no overwrite')
    return publish_artifact(path,manifest_fields={'artifact_type':kind,'schema_version':VERSION,
        'record_digest':digest(record)},files={'record.json':canonical_json_bytes(record)})


def read_record(path, kind, expected=None):
    a = verify_artifact(safe_path(path),expected_artifact_type=kind,expected_schema_version=VERSION)
    require(expected is None or a.manifest_digest==expected,'artifact anchor drift')
    require(set(a.manifest['file_table'])=={'record.json'},'record inventory drift')
    record = strict_json((a.path/'record.json').read_bytes())
    require(digest(record)==a.manifest['record_digest'],'record digest drift')
    return a,record


@dataclass
class Run:
    root: Path
    artifact: object
    record: dict
    tasks: list | None
    spec: ModelSpec

    @property
    def key(self):
        return self.record['model_order'][0]


def prepare_run(destination, *, run_id, model, cohort_root, expected_cohort, publication,
                evaluation, expected_reference, runtime=None, tokenizer_assets=None, mode='formal',
                count_tokens=None):
    """仅在此处核对完整人口和原生预算；不会读取 Gold、加载权重或启动服务。"""
    from werewolf.tom_llm.physical import verify_model
    from werewolf.tom_llm.model_spec import render
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}',run_id) is not None,'invalid run-id')
    require(mode in ('formal','mock'),'unsupported execution mode')
    destination = safe_path(destination)
    require(not destination.exists(),'new Run directory required')
    cohort,tasks = read_cohort(cohort_root,expected_cohort)
    m = cohort.manifest
    require(m['reference']['seal_digest']==expected_reference,'Reference anchor drift')
    source = source_identity()
    spec = ModelSpec(model)
    if mode=='formal':
        require(source['source_inventory_clean'],'formal execution requires committed Source')
        require_formal_cohort(m,tasks)
        require(runtime and tokenizer_assets and count_tokens is None,'real runtime and tokenizer assets required')
        model_evidence = verify_model(spec,runtime['argv_options']['model'])
        budget = check_cpu_parity(tasks,spec,tokenizer_assets)
    else:
        require(runtime is None and callable(count_tokens),'Mock requires an explicit synthetic token counter')
        counts = [count_tokens(render(t,spec)['text']) for t in tasks]
        budget = {'request_count':len(tasks),'input_token_counts':counts,
                  'maximum_input_tokens':max(counts),'input_tokens_total':sum(counts),'status':'MOCK_NOT_CPU_PARITY'}
        model_evidence = None
    counts = budget['input_token_counts']
    require(len(counts)==len(tasks) and all(type(n) is int and 0<n<=7680 for n in counts),
            'full PRE exceeds token budget; no filtering/truncation')
    record = {'run_id':run_id,'mode':mode,'models':{model:spec.digest},'model_order':[model],
        'model_spec':spec.record(),'source':source,'protocol':protocol(),
        'cohort_root':str(cohort.path),'cohort_digest':cohort.manifest_digest,
        'selection':m['selection'],'reference':m['reference'],'games':sorted(m['selection']['game_ids']),
        'game_count':len(m['selection']['game_ids']),'request_count':len(tasks),
        'task_digests':[digest(t) for t in tasks],'row_identity_digest':digest([identity(t) for t in tasks]),
        'publication':str(safe_path(publication)),'evaluation':str(safe_path(evaluation)),
        'runtime':runtime,'model_evidence':model_evidence,'tokenizer_assets':str(tokenizer_assets) if tokenizer_assets else None,
        'budget':budget}
    a = publish_record(destination/'contract','tom_baseline_run',record)
    return Run(destination,a,record,tasks,spec)


def open_run(root, expected=None, *, with_tasks=True, check_source=False):
    """只读取一次上下文；status/stop 不打开 Cohort、Budget 或模型资产。"""
    root = safe_path(root)
    a,r = read_record(root/'contract','tom_baseline_run',expected)
    require(r['protocol']==protocol() and r['mode'] in ('formal','mock'),'execution protocol drift')
    require(len(r['model_order'])==1 and set(r['models'])==set(r['model_order']),'one model per Run required')
    spec = ModelSpec(r['model_order'][0])
    require(r['model_spec']==spec.record() and r['models'][spec.key]==spec.digest,'ModelSpec drift')
    ids = r['task_digests']
    require(len(ids)==r['request_count'] and len(set(ids))==len(ids),'Run task identities drift')
    require(r['game_count']==len(r['games']) and r['games']==sorted(r['selection']['game_ids'])
            and len(set(r['games']))==r['game_count'],'Run game identities drift')
    counts = r['budget']['input_token_counts']
    require(len(counts)==len(ids) and all(type(n) is int and 0<n<=7680 for n in counts)
            and r['budget']['request_count']==len(ids)
            and r['budget']['maximum_input_tokens']==max(counts)
            and r['budget']['input_tokens_total']==sum(counts),'Run token budget drift')
    if r['mode']=='formal':
        from werewolf.tom_llm.reference import fixed_oof_contract
        require(r['source']['source_inventory_clean'] and r['game_count']==150
                and len(ids)==FORMAL_PUBLIC_REQUESTS and r['row_identity_digest']==FORMAL_PUBLIC_ROWS_DIGEST
                and r['selection']['scope']=='subset150'
                and r['reference']==fixed_oof_contract()['reference'],'formal population/Source drift')
        require(r['budget']['status']=='CPU_PARITY_PASS' and r['budget']['model_spec_digest']==spec.digest
                and r['runtime'] and r['model_evidence'] and r['tokenizer_assets'],'formal preparation missing')
    else:
        require(r['runtime'] is None and r['model_evidence'] is None
                and r['budget']['status']=='MOCK_NOT_CPU_PARITY','Mock cannot claim live evidence')
    if check_source:
        require(r['source']==source_identity(),'execution Source changed; no automatic migration')
    tasks = None
    if with_tasks:
        cohort,tasks = read_cohort(r['cohort_root'],r['cohort_digest'])
        require([digest(t) for t in tasks]==ids and r['selection']==cohort.manifest['selection']
                and r['reference']==cohort.manifest['reference']
                and r['row_identity_digest']==cohort.manifest['row_identity_digest'],'Run/cohort mismatch')
        if r['mode']=='formal':
            require_formal_cohort(cohort.manifest,tasks)
    return Run(root,a,r,tasks,spec)
