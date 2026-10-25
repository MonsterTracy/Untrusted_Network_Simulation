"""Fresh, bounded, synthetic-only physical qualification calls. No formal runner."""

import os
from pathlib import Path
import subprocess
import threading
import time

from werewolf.artifact_io import publish_artifact, canonical_json_bytes
from werewolf.tom_llm.common import digest, require, strict_json
from werewolf.tom_llm.deployment import open_deployment, capture_instance
from werewolf.tom_llm.model_spec import ModelSpec, wire
from werewolf.tom_llm.multimodel import source_identity
from werewolf.tom_llm.physical import capture_environment, regular_root
from werewolf.tom_llm.provider_semantics import classify_provider_error, NotExecuted, OutcomeUnknown
from werewolf.tom_llm.qualification_inputs import SCHEDULE
from werewolf.tom_llm.qualification_evidence import VERSION, MEMORY, validate_round


class SyntheticCandidateClient:
    def __init__(self,client,*,deployment,record,instance,guard):
        self.client,self.deployment,self.record,self.instance,self.guard=client,deployment,record,instance,guard
        self.index=0;self.endpoint=f"http://127.0.0.1:{record['profile']['argv_options']['port']}/v1"
        self.check()

    def check(self):
        c=self.client
        require(str(c.base_url).rstrip('/')==self.endpoint and c.max_retries==0
            and c._client._trust_env is False and c._client._transport._pool._retries==0,
            'endpoint/proxy/SDK/HTTP hidden retry drift')
        self.guard()

    def dispatch(self,request):
        try:
            self.check();require(self.index<38,'synthetic dispatch budget exhausted')
            case=self.record['suite']['cases'][SCHEDULE[self.index][1]]
            require(digest(request)==case['wire_digest'],'only fixed native synthetic wire accepted')
        except Exception as error:
            raise NotExecuted('local pre-dispatch refusal; no HTTP call was made') from error
        self.index+=1  # Before network, even failures consume this ordinal. No retry API.
        return self.client.chat.completions.create(**request)

    def close(self):self.client.close()


def model_stats(record):
    root=regular_root(record['model']['directory']);wanted=set(record['model']['files'])
    entries=list(root.rglob('*'))
    require(not any(p.is_symlink() for p in entries) and
        {p.relative_to(root).as_posix() for p in entries if p.is_file()}==wanted,'model stat inventory drift')
    return {name:[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns] for name in sorted(wanted) if (s:=(root/name).stat())}


def create_candidate(root,expected_digest,*,pid):
    artifact,r=open_deployment(root,expected_digest,physical_recheck=True)
    require(r['source']['source_inventory_clean'],'committed source required before physical client')
    instance=capture_instance(artifact,r,pid);stats=model_stats(r)
    def guard():
        require(source_identity()==r['source'] and capture_instance(artifact,r,pid)==instance
            and model_stats(r)==stats,'physical source/model/service drift')
        require(capture_environment(r['profile']['environment_prefix'],prior_audit_digest=r['environment']['prior_audit_digest'])
            ==r['environment'],'runtime environment drift')
    # Construction does not issue HTTP requests. Explicit collection is separate.
    import openai
    import httpx2
    client=openai.OpenAI(api_key='local-synthetic-qualification',max_retries=0,timeout=120,
        base_url=f"http://127.0.0.1:{r['profile']['argv_options']['port']}/v1",
        http_client=openai.DefaultHttpx2Client(trust_env=False,transport=httpx2.HTTPTransport(retries=0)))
    try:return SyntheticCandidateClient(client,deployment=artifact,record=r,instance=instance,guard=guard)
    except Exception:
        client.close();raise


class SampledGpuMemory:
    measurement_identity=MEMORY

    def __init__(self,gpu_uuid):self.gpu_uuid=gpu_uuid

    def _sample(self):
        text=subprocess.check_output(['nvidia-smi','--id='+self.gpu_uuid,'--query-gpu=uuid,memory.used',
            '--format=csv,noheader,nounits'],text=True,timeout=5).strip()
        uuid,amount=[s.strip() for s in text.split(',')]
        require(uuid==self.gpu_uuid,'memory monitor GPU identity drift')
        self.samples.append({'monotonic_seconds':time.perf_counter(),'used_bytes':int(amount)*1024**2,'gpu_uuid':uuid})
        self.peak_bytes=max(self.peak_bytes,self.samples[-1]['used_bytes'])

    def __enter__(self):
        self.samples=[];self.peak_bytes=0;self.error=None;self.stop=threading.Event();self._sample()
        def poll():
            while not self.stop.wait(.1):
                try:self._sample()
                except Exception as error:self.error=error;return
        self.worker=threading.Thread(target=poll,daemon=True);self.worker.start();return self

    def __exit__(self,kind,error,tb):
        self.stop.set();self.worker.join(timeout=6)
        require(not self.worker.is_alive(),'GPU memory monitor did not stop')
        if error is None:
            if self.error:raise self.error
            self._sample()
        return False


def normalize_reply(reply,model):
    require(reply.model==model and len(reply.choices)==1,'reply model/choice mismatch')
    choice=reply.choices[0]
    require(choice.message.refusal is None,'synthetic schema refusal')
    usage=reply.usage
    return {'content':choice.message.content,'finish_reason':choice.finish_reason,
            'usage':{'input_tokens':usage.prompt_tokens,'output_tokens':usage.completion_tokens,'total_tokens':usage.total_tokens}}


def collect_round(destination,transport,*,run,event_reader,memory_monitor):
    """No resume or automatic resend, including a known NotExecuted qualification call."""
    require(type(transport) is SyntheticCandidateClient and transport.index==0 and type(run) is int and run in (0,1),
            'fresh native synthetic candidate and run required')
    root=Path(destination);require(not root.exists(),'probe destination exists; no automatic recovery/replay')
    initial=event_reader();require(len(initial)==1 and initial[0]['kind']=='startup','fresh isolated engine log required')
    contract=publish_artifact(root/'contract',manifest_fields={'artifact_type':'mm_probe_round_contract','schema_version':VERSION,
        'deployment_manifest_digest':transport.deployment.manifest_digest,'deployment_digest':digest(transport.record),
        'instance':transport.instance,'run':run,'dispatch_limit':38,'evidence_kind':'physical_probe'},files={})
    calls=[];spec=ModelSpec(transport.record['key'])
    for ordinal,(phase,case,repeat) in enumerate(SCHEDULE):
        task=transport.record['suite']['cases'][case]['task'];request=wire(task,spec)
        publish_artifact(root/'dispatches'/str(ordinal),manifest_fields={'artifact_type':'mm_probe_dispatch','schema_version':VERSION,
            'contract_digest':contract.manifest_digest,'ordinal':ordinal,'wire_digest':digest(request),'outcome':'dispatched'},files={})
        try:
            monitor=memory_monitor()
            require(monitor.measurement_identity==MEMORY,'memory monitor method drift')
            with monitor:
                start=time.perf_counter();reply=transport.dispatch(request);elapsed=time.perf_counter()-start
                # Record received bytes before identity/evidence/semantic checks.
                publish_artifact(root/'responses'/str(ordinal),manifest_fields={'artifact_type':'mm_probe_received','schema_version':VERSION,
                    'contract_digest':contract.manifest_digest,'ordinal':ordinal,'outcome':'reply_received'},
                    files={'reply.json':canonical_json_bytes(reply.model_dump(mode='json'))})
            transport.check()
        except Exception as error:
            classified=error if isinstance(error,(NotExecuted,OutcomeUnknown)) else classify_provider_error(error)
            outcome='confirmed_not_executed' if isinstance(classified,NotExecuted) else (
                'reply_received_invalid' if (root/'responses'/str(ordinal)).exists() else 'outcome_unknown')
            publish_artifact(root/'failures'/str(ordinal),manifest_fields={'artifact_type':'mm_probe_failure','schema_version':VERSION,
                'contract_digest':contract.manifest_digest,'ordinal':ordinal,'outcome':outcome,
                'exception_type':type(error).__name__,'automatic_retry':False},files={})
            raise classified from error
        events=event_reader();require(len(events)==1+(ordinal+1)*2,'missing/extra engine events; round remains unsealed')
        inp,out=events[-2:]
        normalized=normalize_reply(reply,spec.record()['model'])
        call={'run':run,'phase':phase,'case_index':case,'repeat':repeat,'engine_request_id':inp['engine_request_id'],
              'wire_digest':digest(request),'instance_digest':transport.instance['instance_digest'],**normalized,
              'elapsed_seconds':elapsed,'gpu_peak_bytes':monitor.peak_bytes,'memory_samples':monitor.samples,'memory_measurement':MEMORY}
        publish_artifact(root/'calls'/str(ordinal),manifest_fields={'artifact_type':'mm_probe_call','schema_version':VERSION,
            'contract_digest':contract.manifest_digest,'ordinal':ordinal},files={'call.json':canonical_json_bytes(call)})
        calls.append(call)
    events=event_reader();validate_round(transport.record,transport.record['suite'],transport.instance,events,calls,run)
    result={'schema_version':VERSION,'deployment_digest':digest(transport.record),'evidence_kind':'physical_probe',
            'run':run,'instance':transport.instance,'events':events,'calls':calls}
    return publish_artifact(root/'seal',manifest_fields={'artifact_type':'mm_probe_round_seal','schema_version':VERSION,
        'contract_digest':contract.manifest_digest,'record_digest':digest(result),'deployment_digest':digest(transport.record)},
        files={'round.json':canonical_json_bytes(result)})


def collect_physical_round(destination,deployment_root,expected_digest,*,pid,run,execute=False):
    require(execute,'explicit separately authorized synthetic GPU calls required')
    transport=create_candidate(deployment_root,expected_digest,pid=pid)
    instance=transport.instance
    process={k:instance[k] for k in ('pid','boot_id','start_ticks')}
    log=Path(transport.record['profile']['engine_log_root'])/digest(process)/'engine-events.jsonl'
    def reader():return [strict_json(line) for line in log.read_bytes().splitlines()]
    try:
        return collect_round(destination,transport,run=run,event_reader=reader,
            memory_monitor=lambda:SampledGpuMemory(instance['gpu']['uuid']))
    finally:transport.close()
