"""Serial durable real/fake HTTP execution. Unknown outcomes are never resent."""

from contextlib import contextmanager
import datetime
import fcntl
import http.client
import math
from pathlib import Path
import time
from urllib.parse import urlparse
import uuid

from werewolf.tom_llm.common import digest, identity, require, strict_json
from werewolf.tom_llm.experiment import open_run, publish_record, read_record, safe_path
from werewolf.tom_llm.model_spec import ModelSpec, wire, render, load_cpu_tokenizer
from werewolf.tom_llm.output import parse_output
from werewolf.tom_llm.provider_semantics import NotExecuted, OutcomeUnknown
from werewolf.tom_llm.inference import InfrastructureFailure
from werewolf.tom_llm.qualification_evidence import validate_sampling


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()


class HttpTransport:
    """One connection/request, no SDK, proxy, redirect, reconnect, or hidden retry."""
    evidence_kind='fake_http'
    def __init__(self, endpoint, *, timeout=120):
        u=urlparse(endpoint)
        require(u.scheme=='http' and u.hostname=='127.0.0.1' and u.port and u.path in ('','/')
                and not u.query and not u.fragment and not u.username, 'only explicit loopback HTTP endpoint allowed')
        require(type(timeout) in (int,float) and math.isfinite(timeout) and timeout>0,'invalid HTTP timeout')
        self.endpoint=endpoint;self.port=u.port;self.timeout=timeout
    def guard(self):return {'kind':self.evidence_kind,'endpoint':self.endpoint}
    def send(self, request):
        c=http.client.HTTPConnection('127.0.0.1',self.port,timeout=self.timeout)
        try:
            try:c.connect()
            except OSError as e:raise NotExecuted('connect failed before request transmission') from e
            body={k:v for k,v in request.items() if k!='extra_body'}
            require(not set(body)&set(request['extra_body']),'extra-body key collision')
            body.update(request['extra_body'])
            from werewolf.artifact_io import canonical_json_bytes
            try:
                c.request('POST','/v1/chat/completions',body=canonical_json_bytes(body),headers={'Content-Type':'application/json'})
                r=c.getresponse();raw=r.read(8*1024**2+1)
                require(len(raw)<=8*1024**2,'HTTP response exceeds evidence limit')
                if r.status==429:raise NotExecuted('explicit HTTP 429 rejection')
                if r.status!=200:raise OutcomeUnknown('HTTP failure may follow accepted execution: '+str(r.status))
                return {'status':r.status,'headers':list(r.getheaders()),'body':raw.decode('utf-8')}
            except NotExecuted:raise
            except Exception as e:raise OutcomeUnknown('request may have reached server; automatic resend forbidden') from e
        finally:c.close()
    def proof(self, task):return None


class LiveTransport(HttpTransport):
    evidence_kind='physical_live'
    def __init__(self, root, key, expected_permit, pid):
        from werewolf.tom_llm.live_admission import open_permit, capture_live_instance
        from werewolf.tom_llm.candidate_client import model_stats
        self.permit,self.record=open_permit(root,key,expected_permit)
        self.instance=capture_live_instance(self.permit,self.record,pid)
        self.pid=pid;self.capture=capture_live_instance
        # open_permit already checked every weight SHA and the physical environment.
        from werewolf.tom_llm.physical import asset_spec
        self.model_record={'model':{'directory':self.record['profile']['argv_options']['model'],
            'files':set(asset_spec(ModelSpec(key))['files'])|{'HF_REVISION'}}}
        self.stats=model_stats(self.model_record);self.stats_fn=model_stats
        self.tokenizer=load_cpu_tokenizer(ModelSpec(key),self.record['tokenizer_assets']);self.spec=ModelSpec(key)
        with Path(self.instance['engine_log']).open('rb') as stream:
            self.startup=strict_json(stream.readline());self.offset=stream.tell()
        require(self.startup['kind']=='startup' and self.startup['permit_digest']==expected_permit,'engine not ready')
        super().__init__('http://127.0.0.1:'+str(self.record['profile']['argv_options']['port']))
    def guard(self):
        require(self.capture(self.permit,self.record,self.pid)==self.instance and self.stats_fn(self.model_record)==self.stats,
                'physical service/model lifetime drift')
        return {'kind':self.evidence_kind,'endpoint':self.endpoint,'permit_digest':self.permit.manifest_digest,
                'instance':self.instance,'engine_startup':self.startup}
    def proof(self, task):
        with Path(self.instance['engine_log']).open('rb') as f:
            f.seek(self.offset);raw=f.read();self.offset=f.tell()
        require(raw.endswith(b'\n'),'incomplete durable engine log')
        events=[strict_json(line) for line in raw.splitlines()]
        require(len(events)==2 and events[0]['kind']=='input' and events[1]['kind']=='output', 'missing/extra engine events')
        return {'input':events[0],'output':events[1]}


def validate_reply(raw, task, spec, proof=None, *, tokenizer=None, session=None):
    require(set(raw)=={'status','headers','body'} and raw['status']==200,'unaccepted HTTP response')
    body=strict_json(raw['body'])
    require(body['model']==spec.record()['model'] and len(body['choices'])==1,'reply model/choice drift')
    choice=body['choices'][0];message=choice['message'];content=message.get('content')
    if message.get('refusal') is not None:content=None
    usage={a:body['usage'][b] for a,b in [('input_tokens','prompt_tokens'),('output_tokens','completion_tokens'),('total_tokens','total_tokens')]}
    require(all(type(v) is int and v>=0 for v in usage.values()) and usage['input_tokens']>0
            and usage['input_tokens']+usage['output_tokens']==usage['total_tokens'],'invalid token usage')
    if proof is not None:
        require(session is not None and tokenizer is not None,'physical proof requires native tokenizer and session')
        inp,out=proof['input'],proof['output'];permit=session['identity']['permit_digest']
        life={k:session['identity']['instance'][k] for k in ('pid','boot_id','start_ticks')}
        startup=session['identity']['engine_startup']
        require(all(e['process_lifetime_digest']==digest(life) for e in (startup,inp,out)),'engine/session process lifetime mismatch')
        require(startup['kind']=='startup' and startup['permit_digest']==permit
                and startup['task_digests']==session['remaining_task_digests'],'engine startup contract drift')
        i=session['remaining_task_digests'].index(digest(task))
        require(inp['kind']=='input' and out['kind']=='output' and all(e['permit_digest']==permit and e['ordinal']==i
                and e['task_digest']==digest(task) for e in (inp,out)), 'engine task/permit/order mismatch')
        require(inp['engine_request_id']==out['engine_request_id']==body['id'],'HTTP/engine request ID mismatch')
        text=render(task,spec)['text'];ids=tokenizer.encode(text,add_special_tokens=False)
        require(ids==inp['input_ids']==tokenizer.apply_chat_template(task['messages'],tokenize=True,return_dict=False,
                add_generation_prompt=True,**spec.record()['chat_template']['kwargs']) and 0<len(ids)<=7680,'native engine input_ids/context drift')
        validate_sampling(inp['sampling'],task['response_schema'],spec)
        require(out['content']==message.get('content') and out['finish_reason']==choice['finish_reason']
                and usage['input_tokens']==len(ids) and usage['output_tokens']==len(out['output_token_ids']), 'engine/response/usage drift')
        require(all(type(n) is int and 0<=n<len(tokenizer) for n in out['output_token_ids']) and len(out['output_token_ids'])<=512,'output IDs invalid')
        if spec.key=='qwen3':require(not {151667,151668}&set(out['output_token_ids']),'Non-thinking output drift')
        require(type(out['engine_wall_seconds']) in (int,float) and math.isfinite(out['engine_wall_seconds'])
                and out['engine_wall_seconds']>=0,'engine timing missing')
        from werewolf.tom_llm.live_service import _eos
        require(out['stop_reason'] is None or out['stop_reason'] in _eos(spec),'stop token drift')
    prediction=parse_output(content,task['observer'],finish_reason=choice['finish_reason']).to_record()
    return prediction,usage


def event(root, contract, key, task, fields):
    return publish_record(root,'mm_baseline_event',{'run_digest':contract.manifest_digest,'model':key,
        'task_digest':digest(task),**fields})


def owned(path, contract, key, task):
    a,r=read_record(path,'mm_baseline_event')
    require(r['run_digest']==contract.manifest_digest and r['model']==key and r['task_digest']==digest(task),'cross-run/model/task event')
    return a,r


def inspect_request(path, contract, key, task):
    """Validate durable prefixes BEFORE any network operation or recovery write."""
    if not path.exists():return [],None
    names={p.name for p in path.iterdir()};known={'result'};attempts=[];received=None
    for i in (1,2):
        d=path/f'dispatch-{i}'
        if not d.exists():break
        da,dispatch=owned(d,contract,key,task)
        require(dispatch['number']==i and dispatch['wire']==wire(task,ModelSpec(key)), 'dispatch wire/number drift')
        known.add(d.name);failure=path/f'failure-{i}';response=path/f'response-{i}';accepted=path/f'accepted-{i}'
        if failure.exists():
            fa,f=owned(failure,contract,key,task);known.add(failure.name)
            if response.exists():owned(response,contract,key,task);known.add(response.name)
            require(f['outcome']=='not_executed' and not response.exists() and not accepted.exists(), 'unknown/rejected outcome; quarantine without resend')
            attempts.append(da.manifest_digest);continue
        require(response.exists() and accepted.exists(),'in-flight or unaccepted reply; no automatic resend')
        ra,r=owned(response,contract,key,task);aa,accept=owned(accepted,contract,key,task)
        require(r['session_digest']==dispatch['session_digest'],'response/dispatch session mismatch')
        require(accept['response_digest']==ra.manifest_digest and accept['dispatch_digest']==da.manifest_digest,'reply acceptance chain drift')
        attempts.append(da.manifest_digest);known.update((response.name,accepted.name));received=({**r,'engine_proof':accept['engine_proof']},ra.manifest_digest,dispatch)
        break
    require(names<=known and len(names-{'result'})==len(known-{'result'}),'extra/noncontiguous request state')
    require(received is not None or 'result' not in names,'result without reply')
    require(received is not None or len(attempts)<2,'two confirmed-not-executed attempts exhausted')
    return attempts,received


def pending_tasks(root, key, tasks):
    contract,run,_=open_run(root);remaining=[];tokenizer=None;cache={}
    request_root=Path(root)/'models'/key/'requests'
    require(not request_root.exists() or {p.name for p in request_root.iterdir()}<=set(run['task_digests']),
            'extra public task before recovery; no intersection')
    for t in tasks:
        path=Path(root)/'models'/key/'requests'/digest(t)
        attempts,received=inspect_request(path,contract,key,t)
        if received is not None:
            if run['mode']!='fake' and tokenizer is None:
                from werewolf.tom_llm.live_admission import open_permit
                permit_path=Path(root)/'models'/key/'permit'
                expected=strict_json((permit_path/'manifest.json').read_bytes())['manifest_digest']
                _,permit=open_permit(root,key,expected,physical=False)
                tokenizer=load_cpu_tokenizer(ModelSpec(key),permit['tokenizer_assets'])
            row=make_row(contract,key,t,attempts,received,tokenizer,cache)
            if (path/'result').exists():
                require(owned(path/'result',contract,key,t)[1]['row']==row,'stored result drift before recovery')
        elif not (path/'result').exists():remaining.append(t)
    return remaining


def make_row(contract, key, task, attempts, received, tokenizer=None, session_cache=None):
    response,response_digest,dispatch=received
    cache={} if session_cache is None else session_cache
    session_key=(response['session_root'],response['session_digest'])
    if session_key not in cache:cache[session_key]=read_record(response['session_root'],'mm_baseline_session',response['session_digest'])
    sa,session=cache[session_key]
    require(sa.path.parent==contract.path.parent/'models'/key/'sessions' and session['run_root']==str(contract.path.parent),'session path escaped model/run')
    require(session['run_digest']==contract.manifest_digest and session['model']==key
            and digest(task) in session['remaining_task_digests'],'session ownership drift')
    run=strict_json((contract.path/'record.json').read_bytes())
    require(len(set(session['remaining_task_digests']))==len(session['remaining_task_digests']) and set(session['remaining_task_digests'])<=set(run['task_digests']),'session has duplicate/foreign tasks')
    require(session['mode']==run['mode'] and session['identity']['kind']==('fake_http' if run['mode']=='fake' else 'physical_live'),'session evidence domain drift')
    if run['mode']!='fake':
        permit_manifest=strict_json((Path(session['run_root'])/'models'/key/'permit/manifest.json').read_bytes())
        require(session['identity']['permit_digest']==permit_manifest['manifest_digest'],'session permit drift')
        permit=strict_json((Path(session['run_root'])/'models'/key/'permit/permit.json').read_bytes())
        instance=session['identity']['instance'];profile=permit['profile']
        require(instance['permit_digest']==permit_manifest['manifest_digest'] and instance['gpu']==profile['gpu']
            and session['identity']['endpoint']=='http://127.0.0.1:'+str(profile['argv_options']['port']), 'recorded live instance model/GPU/endpoint drift')
        life={k:instance[k] for k in ('pid','boot_id','start_ticks')}
        require(instance['engine_log']==str(Path(profile['engine_log_root'])/digest(life)/'engine-events.jsonl')
            and type(instance['pid']) is int and instance['pid']>0 and instance['driver'] and instance['kernel'],'recorded live instance log/lifetime drift')
    pred,usage=validate_reply(response['raw'],task,ModelSpec(key),response['engine_proof'],tokenizer=tokenizer,session=session)
    require((response['engine_proof'] is None)==(session['mode']=='fake'),'physical/fake proof confusion')
    durations=[];outcomes=[]
    for number in range(1,len(attempts)):
        failed,record=owned(contract.path.parent/'models'/key/'requests'/digest(task)/f'failure-{number}',contract,key,task)
        require(record['outcome']=='not_executed','unknown outcome cannot supply a prediction')
        outcomes.append(failed.manifest_digest);durations.append(record['elapsed_seconds'])
    outcomes.append(response_digest);durations.append(response['elapsed_seconds'])
    require(all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in durations),'invalid dispatch time')
    return {**{k:task[k] for k in ('game_id','boundary_id','prefix_digest','plan_digest','observer','fold')},
        'canonical_shift':0,'temporal_condition':'explicit_day_phase','prediction':pred,'usage':usage,
        'attempt_digests':attempts,'attempt_outcome_digests':outcomes,'dispatch_elapsed_seconds':durations,
        'task_digest':digest(task),'wire_digest':digest(wire(task,ModelSpec(key))),
        'rendered_prompt_digest':render(task,ModelSpec(key))['text_sha256'],'response_digest':response_digest,
        'session_digest':sa.manifest_digest,'model_spec_digest':ModelSpec(key).digest}


def run_model(root, key, transport, *, after_accepted=None):
    contract,run,tasks=open_run(root);require(key in run['models'],'unplanned model')
    physical=run['mode']!='fake'
    require((type(transport) is LiveTransport) if physical else transport.evidence_kind=='fake_http','transport execution domain mismatch')
    model=Path(root)/'models'/key
    require(not (model/'seal').exists(),'model already sealed')
    remaining=pending_tasks(root,key,tasks)
    identity_before=transport.guard()
    require(identity_before['kind']==('physical_live' if physical else 'fake_http'),'service domain drift')
    session=publish_record(model/'sessions'/uuid.uuid4().hex,'mm_baseline_session',{'run_digest':contract.manifest_digest,
        'model':key,'mode':run['mode'],'run_root':str(Path(root).absolute()),'identity':identity_before,'started_utc':utc(),
        'remaining_task_digests':[digest(t) for t in remaining]})
    tokenizer=getattr(transport,'tokenizer',None);session_cache={};session_start=time.perf_counter()
    for task in tasks:
        path=model/'requests'/digest(task)
        attempts,received=inspect_request(path,contract,key,task)
        if (path/'result').exists():
            _,old=owned(path/'result',contract,key,task)
            require(old['row']==make_row(contract,key,task,attempts,received,tokenizer,session_cache),'stored result drift');continue
        for number in range(len(attempts)+1,3) if received is None else []:
            require(transport.guard()==identity_before,'pre-dispatch service drift')
            da=event(path/f'dispatch-{number}',contract,key,task,{'number':number,'wire':wire(task,ModelSpec(key)),
                'session_digest':session.manifest_digest,'started_utc':utc()})
            attempts.append(da.manifest_digest);start=time.perf_counter();response_written=False
            try:
                raw=transport.send(wire(task,ModelSpec(key)))
                elapsed=time.perf_counter()-start
                # Raw reply is durable before guard/proof/parsing checks.
                ra=event(path/f'response-{number}',contract,key,task,{'raw':raw,'engine_proof':None,
                    'elapsed_seconds':elapsed,'session_root':str(session.path),'session_digest':session.manifest_digest})
                response_written=True
                require(transport.guard()==identity_before,'post-response service drift')
                proof=transport.proof(task)
                # Store proof separately: raw response remains immutable.
                accepted_response={**strict_json((ra.path/'record.json').read_bytes()),'engine_proof':proof}
                validate_reply(raw,task,ModelSpec(key),proof,tokenizer=tokenizer,
                    session=read_record(session.path,'mm_baseline_session')[1])
                event(path/f'accepted-{number}',contract,key,task,{'response_digest':ra.manifest_digest,
                    'dispatch_digest':da.manifest_digest,'engine_proof':proof})
                if after_accepted:after_accepted(task)
                received=(accepted_response,ra.manifest_digest,{'number':number});break
            except Exception as error:
                outcome='not_executed' if isinstance(error,NotExecuted) and not response_written else 'unknown'
                event(path/f'failure-{number}',contract,key,task,{'outcome':outcome,'exception_type':type(error).__name__,
                    'elapsed_seconds':time.perf_counter()-start,'response_received':response_written})
                if outcome=='not_executed' and number<2:continue
                raise InfrastructureFailure('execution stopped; '+outcome+'; no fabricated prediction') from error
        require(received is not None,'two confirmed-not-executed attempts exhausted')
        row=make_row(contract,key,task,attempts,received,tokenizer,session_cache)
        event(path/'result',contract,key,task,{'row':row})
    publish_record(session.path.parent/(session.path.name+'-complete'),'mm_baseline_session_complete',
        {'session_digest':session.manifest_digest,'completed_utc':utc(),'elapsed_seconds':time.perf_counter()-session_start})
    return seal_model(root,key,tokenizer=tokenizer)


def read_predictions(root,key,*,expected=None,tokenizer=None):
    contract,run,tasks=open_run(root)
    a,sealed=read_record(Path(root)/'models'/key/'seal','mm_baseline_predictions',expected)
    require(sealed['run_digest']==contract.manifest_digest and sealed['mode']==run['mode']
            and sealed['model']==key and sealed['model_spec_digest']==ModelSpec(key).digest,'prediction seal domain/model drift')
    require(sealed['session_artifacts']==session_artifacts(root,key),'sealed session evidence drift')
    rows=collect_rows(root,key,tokenizer=tokenizer)
    require(sealed['rows']==rows and sealed['task_digests']==run['task_digests'],'sealed coverage/result mismatch')
    return a,rows


def collect_rows(root,key,*,tokenizer=None):
    contract,run,tasks=open_run(root)
    if run['mode']!='fake' and tokenizer is None:
        from werewolf.tom_llm.live_admission import open_permit
        permit_path=Path(root)/'models'/key/'permit'
        digest_anchor=strict_json((permit_path/'manifest.json').read_bytes())['manifest_digest']
        _,permit=open_permit(root,key,digest_anchor)
        tokenizer=load_cpu_tokenizer(ModelSpec(key),permit['tokenizer_assets'])
    request_root=Path(root)/'models'/key/'requests'
    require(request_root.is_dir() and {p.name for p in request_root.iterdir()}==set(run['task_digests']),'missing/extra public task; no intersection')
    rows=[];session_cache={}
    for task in tasks:
        path=request_root/digest(task);attempts,received=inspect_request(path,contract,key,task)
        require(received is not None,'incomplete task cannot seal')
        _,record=owned(path/'result',contract,key,task)
        row=make_row(contract,key,task,attempts,received,tokenizer,session_cache)
        require(record['row']==row,'prediction differs from durable raw reply')
        rows.append(row)
    return rows


def session_artifacts(root,key):
    return {p.name:read_record(p,'mm_baseline_session_complete' if p.name.endswith('-complete') else 'mm_baseline_session')[0].manifest_digest
        for p in sorted((Path(root)/'models'/key/'sessions').iterdir())}


def seal_model(root,key,*,tokenizer=None):
    contract,run,tasks=open_run(root);rows=collect_rows(root,key,tokenizer=tokenizer)
    return publish_record(Path(root)/'models'/key/'seal','mm_baseline_predictions',{'run_digest':contract.manifest_digest,
        'model':key,'mode':run['mode'],'formal_result':run['mode']=='formal','model_spec_digest':ModelSpec(key).digest,
        'task_digests':run['task_digests'],'rows':rows,'session_artifacts':session_artifacts(root,key),'sealed_utc':utc()})


def finalize_received(root,key):
    """Recover only accepted local responses; no network or re-dispatch."""
    contract,run,tasks=open_run(root);tokenizer=None;session_cache={}
    require(not pending_tasks(root,key,tasks),'outstanding requests still require execution')
    if run['mode']!='fake':
        from werewolf.tom_llm.live_admission import open_permit
        path=Path(root)/'models'/key/'permit'
        _,permit=open_permit(root,key,strict_json((path/'manifest.json').read_bytes())['manifest_digest'])
        tokenizer=load_cpu_tokenizer(ModelSpec(key),permit['tokenizer_assets'])
    for task in tasks:
        path=Path(root)/'models'/key/'requests'/digest(task)
        attempts,received=inspect_request(path,contract,key,task)
        if not (path/'result').exists():event(path/'result',contract,key,task,{'row':make_row(contract,key,task,attempts,received,tokenizer,session_cache)})
    return seal_model(root,key,tokenizer=tokenizer)


def seal_run(root):
    contract,run,_=open_run(root);seals={}
    for key in run['models']:seals[key]=read_predictions(root,key)[0].manifest_digest
    return publish_record(Path(root)/'seal','mm_baseline_run_seal',{'run_digest':contract.manifest_digest,
        'mode':run['mode'],'models':seals,'source':run['source'],'completed_utc':utc()})


@contextmanager
def run_lock(root):
    path=safe_path(root);require((path/'contract').is_dir(),'missing run contract')
    with (path/'execution.lock').open('a+b') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)


def status(root):
    contract,r,tasks=open_run(root);models={}
    for key in r['models']:
        done=blocked=0;errors=[]
        for task in tasks:
            p=Path(root)/'models'/key/'requests'/digest(task)
            try:
                inspect_request(p,contract,key,task)
                done+=int((p/'result').exists())
            except (ValueError,InfrastructureFailure) as e:blocked+=1;errors.append(str(e))
        sealed=False
        if (Path(root)/'models'/key/'seal').exists():
            try:read_predictions(root,key);sealed=True
            except (ValueError,OSError,KeyError,TypeError,InfrastructureFailure) as e:errors.append('invalid prediction seal: '+str(e))
        models[key]={'completed_requests':done,'blocked_requests':blocked,'sealed':sealed,'errors':errors}
    return {'run_id':r['run_id'],'mode':r['mode'],'games':r['game_count'],'requests_per_model':len(tasks),'models':models}
