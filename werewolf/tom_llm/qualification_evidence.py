"""CPU validation of actual engine/caller/memory evidence. No inferred GPU success."""

from collections import Counter
import math

from werewolf.tom_llm.common import digest, require, strict_json
from werewolf.tom_llm.model_spec import ModelSpec
from werewolf.tom_llm.output import parse_output
from werewolf.tom_llm.qualification_inputs import validate_suite, SCHEDULE

VERSION = 'classic7_multimodel_gpu_evidence_v1'
CAPTURE_SITE = 'AsyncLLM._add_request/EngineCoreRequest'
MEMORY = {'method':'nvidia_smi_sampled_device_used','interval_seconds':.1,'allocator_peak_measured':False}
NUMERIC = {'n':1,'temperature':0,'top_p':1,'top_k':0,'min_p':0,'max_tokens':512,'seed':20261014,
           'presence_penalty':0,'frequency_penalty':0,'repetition_penalty':1,'min_tokens':0}


def validate_sampling(params,schema,spec):
    keys=set(NUMERIC)|{'stop','stop_token_ids','eos_token_ids','primary_eos_token_id','ignore_eos',
        'json_schema','logit_bias','allowed_token_ids','bad_words','logits_processors'}
    require(type(params) is dict and set(params)==keys,'effective SamplingParams inventory drift')
    require(all(type(params[k]) in (int,float) and math.isfinite(params[k]) and params[k]==v for k,v in NUMERIC.items()),
            'effective decoding parameter drift')
    row=spec.record(); eos=row['termination']['generation_eos_token_id']
    eos=set(eos if type(eos) is list else [eos])|{row['tokenizer']['eos_token_id']}
    require(params['stop'] in (None,[]) and params['ignore_eos'] is False
            and params['eos_token_ids']==sorted(eos) and params['primary_eos_token_id'] in eos
            and type(params['stop_token_ids']) is list and set(params['stop_token_ids'])<=eos, 'effective EOS/termination drift')
    require(all(params[k] is None for k in ('logit_bias','allowed_token_ids','bad_words','logits_processors')),
            'undeclared logits modifiers')
    require(params['json_schema']==schema,'effective six-seat JSON Schema drift')


def validate_memory(samples,peak,gpu_uuid):
    require(type(samples) is list and len(samples)>=2,'missing sampled GPU memory evidence')
    require(all(set(s)=={'monotonic_seconds','used_bytes','gpu_uuid'} and s['gpu_uuid']==gpu_uuid
        and type(s['monotonic_seconds']) in (int,float) and math.isfinite(s['monotonic_seconds'])
        and type(s['used_bytes']) is int and 0<s['used_bytes']<=24576*1024**2 for s in samples), 'GPU memory sample drift')
    require([s['monotonic_seconds'] for s in samples]==sorted(s['monotonic_seconds'] for s in samples)
            and type(peak) is int and peak==max(s['used_bytes'] for s in samples),'GPU peak/sample mismatch')


def validate_round(deployment,suite,instance,events,calls,run):
    spec=ModelSpec(deployment['key']);validate_suite(suite,spec)
    require(run in (0,1) and type(run) is int and len(calls)==38 and len(events)==77,'round needs 38 calls and startup+76 engine events')
    require(instance['instance_digest']==digest({k:v for k,v in instance.items() if k!='instance_digest'}), 'instance digest drift')
    require(instance['gpu']==deployment['profile']['gpu'],'instance GPU identity drift')
    startup=events[0]
    common={'deployment_digest':digest(deployment),'model_manifest_digest':digest(deployment['model']),
            'process_lifetime_digest':digest({k:instance[k] for k in ('pid','boot_id','start_ticks')})}
    require(startup=={**common,'kind':'startup','argv_options':deployment['profile']['argv_options'],
                     'frontend':deployment['profile']['frontend'],'sdk_hidden_retries':False,'engine_request_retries':0},
            'missing/mismatched actual server startup contract')
    seen=set();outputs={};schema_cases=set();statuses=Counter()
    for ordinal,(phase,case,repeat) in enumerate(SCHEDULE):
        call=calls[ordinal];inp,out=events[1+ordinal*2:3+ordinal*2];expected=suite['cases'][case]
        for event,kind in [(inp,'input'),(out,'output')]:
            require(all(event.get(k)==v for k,v in common.items()) and event['kind']==kind
                and event['case_index']==case,'engine identity/schedule drift')
        key=inp['engine_request_id']
        require(type(key) is str and key and key not in seen and out['engine_request_id']==key, 'duplicate/missing engine request ID')
        seen.add(key)
        require(inp['capture_site']==CAPTURE_SITE and inp['input_ids']==expected['input_ids'], 'final engine input_ids drift')
        validate_sampling(inp['sampling'],expected['task']['response_schema'],spec)
        require(call['run']==run and (call['phase'],call['case_index'],call['repeat'])==(phase,case,repeat)
            and call['engine_request_id']==key and call['wire_digest']==expected['wire_digest']
            and call['instance_digest']==instance['instance_digest'] and call['content']==out['content']
            and call['finish_reason']==out['finish_reason'], 'caller/engine reply or instance mismatch')
        tokens=out['output_token_ids']
        require(type(tokens) is list and 0<len(tokens)<=512 and all(type(t) is int and 0<=t<suite['vocabulary_size'] for t in tokens),
                'output IDs invalid')
        if spec.key=='qwen3':require(not {151667,151668}&set(tokens),'Non-thinking output drift')
        require(out['finish_reason'] in ('stop','length') and type(out['content']) is str,'invalid service finish')
        reason=out['stop_reason']
        termination=inp['sampling']['eos_token_ids']
        require(reason is None or type(reason) is int and reason in termination,'undeclared stop reason')
        if out['finish_reason']=='stop':
            value=strict_json(out['content']);seats=expected['task']['response_schema']['properties']['probabilities']['required']
            require(type(value) is dict and set(value)=={'probabilities'} and type(value['probabilities']) is dict
                and set(value['probabilities'])==set(seats) and all(type(v) in (int,float) and math.isfinite(v) and 0<=v<=1
                for v in value['probabilities'].values()), 'served JSON violates schema')
            schema_cases.add(case)
        statuses[parse_output(out['content'],expected['task']['observer'],finish_reason=out['finish_reason']).status]+=1
        usage={'input_tokens':len(expected['input_ids']),'output_tokens':len(tokens),'total_tokens':len(expected['input_ids'])+len(tokens)}
        require(call['usage']==usage and type(call['elapsed_seconds']) in (int,float) and math.isfinite(call['elapsed_seconds'])
            and call['elapsed_seconds']>=0 and call['memory_measurement']==MEMORY,'usage/timing/memory method mismatch')
        validate_memory(call['memory_samples'],call['gpu_peak_bytes'],instance['gpu']['uuid'])
        if phase=='measurement':
            fingerprint=digest([tokens,out['content'],out['finish_reason']])
            require(outputs.setdefault(case,fingerprint)==fingerprint,'within-instance repeat mismatch')
    require(schema_cases==set(range(12)),'no completed schema response for some cases')
    return {'outputs_by_case':outputs,'conversion_status_counts':dict(statuses),'engine_request_ids':sorted(seen)}


def validate_qualification(deployment,rounds):
    require(type(rounds) is list and len(rounds)==2,'two independent rounds required')
    reports=[];lifetimes=[]
    for run,r in enumerate(rounds):
        require(r['run']==run and r['deployment_digest']==digest(deployment)
            and r['evidence_kind']=='physical_probe','wrong probe domain/run/deployment')
        instance=r['instance'];lifetimes.append(digest({k:instance[k] for k in ('pid','boot_id','start_ticks')}))
        reports.append(validate_round(deployment,deployment['suite'],instance,r['events'],r['calls'],run))
    require(all(rounds[0]['instance'][k]==rounds[1]['instance'][k] for k in ('driver_version','kernel_release')), 'restart driver/kernel drift')
    require(lifetimes[0]!=lifetimes[1] and rounds[0]['instance']['instance_digest']!=rounds[1]['instance']['instance_digest'],
            'independent restart evidence required')
    require(reports[0]['outputs_by_case']==reports[1]['outputs_by_case'],'restart output reproducibility mismatch')
    conversion_counts=Counter()
    for report in reports:conversion_counts.update(report['conversion_status_counts'])
    measured=[call for r in rounds for call in r['calls'] if call['phase']=='measurement']
    wall=sum(c['elapsed_seconds'] for c in measured);tokens=sum(c['usage']['output_tokens'] for c in measured)
    return {'schema_version':VERSION,'deployment_digest':digest(deployment),'model_spec_digest':ModelSpec(deployment['key']).digest,
        'suite_digest':digest(deployment['suite']),'round_digests':[digest(r) for r in rounds],
        'dispatches':76,'measurement_dispatches':72,'warmup_dispatches':4,'input_ids_equal':True,
        'effective_sampling_equal':True,'restart_reproducible':True,'qualification_scope':'synthetic_only',
        'conversion_status_counts':dict(conversion_counts),
        'driver_version':rounds[0]['instance']['driver_version'],'kernel_release':rounds[0]['instance']['kernel_release'],
        'measurement_input_tokens':sum(c['usage']['input_tokens'] for c in measured),
        'measurement_output_tokens':tokens,'measurement_client_wall_seconds':wall,
        'diagnostic_output_tokens_per_second':tokens/wall if wall>0 else None,
        'sampled_device_peak_bytes':max(c['gpu_peak_bytes'] for r in rounds for c in r['calls']),
        'timing_scope':'client_wall_including_predispatch_identity_guard',
        'formal_efficiency_benchmark':False,'formal_execution_authorized':False}
