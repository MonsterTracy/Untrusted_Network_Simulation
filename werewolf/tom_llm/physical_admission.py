"""New physical qualification reader/publisher. Never upgrades W5 or Mock receipts."""

from pathlib import Path
import sys

from werewolf.artifact_io import publish_artifact, verify_artifact, canonical_json_bytes
from werewolf.tom_llm.common import digest, require, require_digest, strict_json
from werewolf.tom_llm.deployment import open_deployment
from werewolf.tom_llm.qualification_inputs import SCHEDULE
from werewolf.tom_llm.qualification_evidence import VERSION, validate_round, validate_qualification

ADMISSION_VERSION = 'classic7_multimodel_physical_admission_v1'


def open_round(root,expected_digest,deployment,*,deployment_manifest_digest):
    require_digest(expected_digest);root=Path(root)
    a=verify_artifact(root/'seal',expected_artifact_type='mm_probe_round_seal',expected_schema_version=VERSION)
    require(a.manifest_digest==expected_digest and set(a.manifest['file_table'])=={'round.json'}, 'probe round seal identity drift')
    record=strict_json((a.path/'round.json').read_bytes())
    require(digest(record)==a.manifest['record_digest'] and record['deployment_digest']==digest(deployment), 'probe round/deployment payload drift')
    c=verify_artifact(root/'contract',expected_artifact_type='mm_probe_round_contract',expected_schema_version=VERSION)
    require(c.manifest['deployment_manifest_digest']==deployment_manifest_digest
        and record['instance']['deployment_manifest_digest']==deployment_manifest_digest, 'probe service/deployment manifest mismatch')
    require(c.manifest_digest==a.manifest['contract_digest'] and c.manifest['deployment_digest']==digest(deployment)
        and c.manifest['evidence_kind']=='physical_probe' and c.manifest['instance']==record['instance']
        and c.manifest['run']==record['run'] and c.manifest['dispatch_limit']==38, 'probe contract drift')
    require({p.name for p in root.iterdir()}=={'contract','dispatches','responses','calls','seal'}, 'unsealed/failed/extra probe state')
    for directory in ('dispatches','responses','calls'):
        require({p.name for p in (root/directory).iterdir()}=={str(i) for i in range(38)}, 'probe durable coverage mismatch')
    for i,(_,case,_) in enumerate(SCHEDULE):
        data={}
        for folder,kind in [('dispatches','mm_probe_dispatch'),('responses','mm_probe_received'),('calls','mm_probe_call')]:
            event=verify_artifact(root/folder/str(i),expected_artifact_type=kind,expected_schema_version=VERSION)
            require(event.manifest['contract_digest']==c.manifest_digest and event.manifest['ordinal']==i, 'probe event ownership drift')
            expected_files={'reply.json'} if folder=='responses' else {'call.json'} if folder=='calls' else set()
            require(set(event.manifest['file_table'])==expected_files, 'probe event file inventory drift')
            data[folder]=event
        require(data['dispatches'].manifest['wire_digest']==deployment['suite']['cases'][case]['wire_digest']
            and data['dispatches'].manifest['outcome']=='dispatched'
            and data['responses'].manifest['outcome']=='reply_received', 'probe dispatch/received status drift')
        call=strict_json((data['calls'].path/'call.json').read_bytes())
        raw=strict_json((data['responses'].path/'reply.json').read_bytes())
        require(call==record['calls'][i] and raw['model']==deployment['model_spec']['model'] and len(raw['choices'])==1,
                'probe call/raw reply drift')
        choice=raw['choices'][0]
        require(choice['message']['refusal'] is None and choice['message']['content']==call['content']
            and choice['finish_reason']==call['finish_reason'] and call['usage']=={
                'input_tokens':raw['usage']['prompt_tokens'],'output_tokens':raw['usage']['completion_tokens'],
                'total_tokens':raw['usage']['total_tokens']},'probe seal differs from durable raw reply')
    validate_round(deployment,deployment['suite'],record['instance'],record['events'],record['calls'],record['run'])
    return a,record


def publish_admission(destination,deployment_root,expected_deployment,*,round_roots,expected_rounds):
    require(sys.platform=='linux','physical GPU admission cannot be published from Mac/Mock')
    a,deployment=open_deployment(deployment_root,expected_deployment,physical_recheck=True)
    require(deployment['source']['source_inventory_clean'],'physical admission requires committed Source')
    require(len(round_roots)==len(expected_rounds)==2 and len(set(map(str,round_roots)))==2,'two distinct round roots required')
    require(Path(destination).absolute().is_relative_to(Path(deployment['profile']['root'])), 'admission must stay in model directory')
    rounds=[open_round(root,wanted,deployment,deployment_manifest_digest=a.manifest_digest)[1] for root,wanted in zip(round_roots,expected_rounds,strict=True)]
    report=validate_qualification(deployment,rounds)
    record={'schema_version':ADMISSION_VERSION,'evidence_kind':'physical_gpu_synthetic',
        'deployment_root':str(a.path),'deployment_manifest_digest':a.manifest_digest,'deployment_digest':digest(deployment),
        'model_spec_digest':deployment['profile']['model_spec_digest'],
        'round_roots':[str(Path(p).absolute()) for p in round_roots],'round_manifest_digests':expected_rounds,
        'qualification':report,'formal_execution_authorized':False}
    return publish_artifact(destination,manifest_fields={'artifact_type':'mm_physical_gpu_admission','schema_version':ADMISSION_VERSION,
        'record_digest':digest(record),'qualification_scope':'synthetic_only','formal_execution_authorized':False},
        files={'admission.json':canonical_json_bytes(record)})


def open_admission(root,*,expected_manifest,expected_deployment,model_spec):
    require_digest(expected_manifest);require_digest(expected_deployment)
    a=verify_artifact(root,expected_artifact_type='mm_physical_gpu_admission',expected_schema_version=ADMISSION_VERSION)
    require(a.manifest_digest==expected_manifest and set(a.manifest['file_table'])=={'admission.json'}, 'new physical admission identity required')
    record=strict_json((a.path/'admission.json').read_bytes())
    require(record['evidence_kind']=='physical_gpu_synthetic' and record['formal_execution_authorized'] is False
        and digest(record)==a.manifest['record_digest'] and record['model_spec_digest']==model_spec.digest
        and record['deployment_manifest_digest']==expected_deployment,'admission domain/model/deployment drift')
    active,deployment=open_deployment(record['deployment_root'],expected_deployment,current_source=False)
    require(deployment['source']['source_inventory_clean'],'admission source is not clean')
    require(deployment['key']==model_spec.key and digest(deployment)==record['deployment_digest'],'admitted model mismatch')
    rounds=[open_round(p,h,deployment,deployment_manifest_digest=active.manifest_digest)[1] for p,h in zip(record['round_roots'],record['round_manifest_digests'],strict=True)]
    require(validate_qualification(deployment,rounds)==record['qualification'],'admission qualification changed')
    return a,record
