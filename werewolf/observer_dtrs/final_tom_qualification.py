"""Offline, CPU-only numerical QUAL evidence; never CAL/TEST authorization.

The historical graph reference is a separately imported frozen checkout. It is
not open_fit/verify_seal/SealedQwen3Predictor recovery replay. Zero numerical
tolerance is declared before execution; mismatch stops, never widens tolerance.
"""
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from werewolf.artifact_io import canonical_json_bytes, publish_artifact, read_artifact_file, sha256_bytes, verify_artifact
from werewolf.observer_dtrs.qual_sources import ROOT, capture_sources, digest, new_external_path, require, verify_snapshot

VERSION = 'observer_dtrs_cpu_qualification_v1'
PRE_VERSION = 'observer_dtrs_qual_public_pres_v1'


def read_pres(path, raw_sha256):
    path = Path(path)
    require(path.stat().st_size <= 16*1024*1024, 'public PRE packet too large')
    raw = path.read_bytes()
    require(sha256_bytes(raw) == raw_sha256, 'external PRE SHA mismatch')
    r = json.loads(raw)
    require(canonical_json_bytes(r) == raw and set(r) == {'schema_version','scope','cases'}
            and r['schema_version'] == PRE_VERSION
            and r['scope'] in ('CONTROLLED_FIXTURE','PUBLIC_PRE_QUALIFICATION_ONLY')
            and type(r['cases']) is list and 1 <= len(r['cases']) <= 32, 'public PRE packet schema')
    from werewolf.canonical_collection.game_bundle import _prefix_from_record
    from werewolf.tom.final_capacity import derived_capacity, validate_final_pre
    cases, seen = [], set()
    for row in r['cases']:
        require(type(row) is dict and set(row) == {'case_id','pre'}
                and type(row['case_id']) is str and 0 < len(row['case_id']) <= 128
                and row['case_id'] not in seen, 'case identity/duplicate')
        seen.add(row['case_id'])
        prefix = _prefix_from_record(row['pre'])
        require(prefix.to_record() == row['pre'], 'PRE roundtrip/allowlist')
        validate_final_pre(prefix,derived_capacity(1024))
        cases.append((row['case_id'],prefix))
    return r['scope'], cases


def _tensor(value):
    x = value.detach().cpu().numpy()
    return {'dtype':x.dtype.str,'shape':list(x.shape),'sha256':sha256_bytes(x.tobytes(order='C'))}


def compare_reference(model, cases, reference, *, inputs_sha256):
    """Offline inspector only; private model state never becomes online input."""
    import torch
    from werewolf.observer_dtrs.final_tom_capsule import encode_public_pre
    origin = model.identity['origin']
    require(set(reference) == {'schema_version','reference_kind','source_revision','implementation_digest',
            'capsule_digest','inputs_raw_sha256','model_tensors','temporal','full_semantic_recovery',
            'original_sealed_predictor','cases','runtime'}
            and reference['schema_version'] == 'observer_dtrs_qual_reference_v1'
            and reference['reference_kind'] == 'PINNED_ORIGINAL_GRAPH_CPU_SAME_CAPSULE_NOT_SEALED_LIFECYCLE'
            and reference['source_revision'] == origin['source_revision']
            and reference['implementation_digest'] == origin['implementation_digest']
            and reference['capsule_digest'] == model.identity['capsule_digest']
            and reference['inputs_raw_sha256'] == inputs_sha256
            and reference['full_semantic_recovery'] == reference['original_sealed_predictor'] == 'NOT_RUN',
            'reference origin/scope mismatch')
    runtime=reference['runtime'];observed=model.identity['runtime']
    require(type(runtime) is dict and set(runtime)=={'python','packages','device','dtype','attention_backend',
                'torch_threads','torch_interop_threads'}
            and all(runtime[n]==observed[n] for n in ('python','packages','device','dtype','attention_backend','torch_threads'))
            and type(runtime['torch_interop_threads']) is int and runtime['torch_interop_threads']>=1,
            'reference CPU runtime mismatch')
    require(reference['model_tensors'] == {k:_tensor(v) for k,v in model._model.state_dict().items()},
            'restored tensor mismatch')
    require(reference['temporal'] == {k:sha256_bytes(getattr(model._model.temporal,k).tobytes())
                                     for k in ('day','phase')}, 'Temporal value mismatch')
    rows = reference['cases']
    require(type(rows) is list and len(rows) == len(cases), 'reference coverage')
    keyed = {r['case_id']:r for r in rows}
    require(len(keyed) == len(rows) and set(keyed) == {n for n,p in cases}, 'reference case identity')
    results=[]
    for case_id, prefix in cases:
        r = keyed[case_id]
        plan, arrays = encode_public_pre(prefix)
        require(set(r) == {'case_id','pre_sha256','plan_digest','token_count','inputs','log_q_nonself','q'}
                and r['pre_sha256'] == digest(prefix.to_record())
                and r['plan_digest'] == plan.plan_digest and r['token_count'] == plan.token_count,
                'reference PRE/plan mismatch')
        require(r['inputs'] == {k:_tensor(v[0]) for k,v in arrays.items()}, 'input encoding mismatch')
        with torch.inference_mode():
            logq = model._model(**arrays)[0]
        nonself = [[float(logq[i,j]) for j in range(7) if i != j] for i in range(7)]
        require(nonself == r['log_q_nonself'], 'log-Q mismatch (atol=rtol=0)')
        prediction = model.predict(prefix)
        require([list(row) for row in prediction.probabilities] == r['q'], 'Q conversion mismatch (atol=rtol=0)')
        require(model.predict(prefix).probabilities == prediction.probabilities, 'non-deterministic repeated PRE')
        results.append({'case_id':case_id,'game_id':prefix.game_id,'boundary_id':prefix.boundary_id,
            'prefix_digest':prefix.prefix_digest,'plan_digest':plan.plan_digest,'token_count':plan.token_count,
            'phase':prefix.public_temporal_state.phase.value,'alive_players':len(prefix.alive_observer_ids),
            'input_tensor_digest':digest(r['inputs']),'Q_digest':digest(r['q']),
            'status':'PASS_EXACT_CPU_GRAPH_ENCODING_LOGQ_Q_AND_REPEAT'})
    return results


def capture_reference(checkout, model, inputs, inputs_sha256, *, deadline_s=120):
    require(type(deadline_s) in (int,float) and 0 < deadline_s <= 3600, 'bounded reference subprocess')
    origin = model.identity['origin']
    command=[sys.executable,'-B',str(ROOT/'scripts/export_final_tom_qual_reference.py'),
        '--checkout',str(checkout),'--revision',origin['source_revision'],
        '--implementation-digest',origin['implementation_digest'],
        '--capsule',str(inputs['capsule_path']),'--capsule-digest',model.identity['capsule_digest'],
        '--inputs',str(inputs['pre_path']),'--inputs-sha256',inputs_sha256,
        '--threads',str(model.identity['runtime']['torch_threads'])]
    env={k:v for k,v in os.environ.items() if k not in ('PYTHONPATH','PYTHONHOME') and not k.startswith('GIT_')}
    env.update(PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false')
    result=subprocess.run(command,env=env,cwd=checkout,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=deadline_s)
    require(result.returncode == 0 and len(result.stdout) <= 4*1024*1024, 'historical reference failed')
    record=json.loads(result.stdout)
    require(canonical_json_bytes(record) == result.stdout, 'reference canonical output')
    return record


def verify_provenance_packet(model, path, raw_sha256):
    """Reopen externally anchored chain-files evidence; no fit/recovery execution."""
    from werewolf.observer_dtrs.final_tom_capsule_build import _audit_packet
    from werewolf.observer_dtrs.final_tom_audit import _Pins
    r=_audit_packet(path,raw_sha256,_Pins(),controlled=False)
    origin=model.identity['origin'];c=r['checks']
    require(origin['audit_file_raw_sha256']==raw_sha256 and origin['audit_evidence_digest']==r['evidence_digest']
            and origin['fit_digest']==c['fit']['manifest_digest']
            and origin['source_revision']==c['source']['source_revision']
            and origin['implementation_digest']==c['source']['implementation_digest']
            and all(origin[n]==c['recovery'][n] for n in ('terminal_digest','seal_digest',
                'model_bin_raw_sha256','model_container_descriptor_sha256','ancestry_digest')),
            'capsule/provenance evidence mismatch')
    require(c['temporal']['manifest_digest']==origin['temporal_manifest_digest'],
            'capsule/provenance Temporal mismatch')
    return {'audit_file_raw_sha256':raw_sha256,'audit_evidence_digest':r['evidence_digest'],
            'scope':'CHAIN_FILES_BYTES_AND_CODEC_NOT_SEMANTIC_RECOVERY'}


def _qualify_cpu(*, capsule_path, capsule_digest, source_path, source_digest, pre_path, pre_sha256,
                 reference_checkout, destination, limits=None, capsule_scope, pre_scope,
                 audit_path=None, audit_raw_sha256=None):
    """Public entry accepts real pinned capsules only; source snapshot is explicit.

    The frozen original GRAPH is the numerical reference, not a reconstructed
    training lifecycle. No semantic-recovery or OS-sandbox PASS is inferred.
    """
    snapshot = verify_snapshot(source_path,source_digest)
    output = new_external_path(destination,protected=(capsule_path,source_path,pre_path,reference_checkout,
        *((audit_path,) if audit_path is not None else ())))
    scope, cases = read_pres(pre_path,pre_sha256)
    require(scope == pre_scope, 'PRE qualification scope mismatch')
    from werewolf.observer_dtrs.final_tom_capsule import CapsulePredictor
    from werewolf.observer_dtrs.final_tom_client import ResidentOwner, ServiceTrust
    from werewolf.observer_dtrs.final_tom_broker import ResidentBroker
    from werewolf.observer_dtrs.final_tom_wire import Limits
    limits = limits or Limits()
    started=time.monotonic()
    output.mkdir(mode=0o700)
    owner=None
    stage='capsule_load'
    try:
        model=CapsulePredictor(capsule_path,expected_manifest_digest=capsule_digest,
                              expected_scope=capsule_scope,device='cpu')
        stage='externally_anchored_chain_files_evidence'
        provenance=(verify_provenance_packet(model,audit_path,audit_raw_sha256)
                    if capsule_scope=='PINNED_FINAL_INFERENCE_UNQUALIFIED'
                    else {'scope':'CONTROLLED_FIXTURE_NOT_REAL_PROVENANCE'})
        stage='historical_graph_reference'
        reference=capture_reference(reference_checkout,model,
            {'capsule_path':capsule_path,'pre_path':pre_path},pre_sha256)
        stage='tensor_temporal_encoding_logq_q_comparison'
        numerical=compare_reference(model,cases,reference,inputs_sha256=pre_sha256)
        verify_snapshot(source_path,source_digest)
        stage='resident_lifecycle'
        identity=model.identity
        trust=ServiceTrust(capsule_digest,identity['scope'],
            identity['loader_source']['loader']['implementation_digest'],digest(identity['runtime']))
        owner=ResidentOwner(capsule_path,trust=trust,journal_root=output/'resident',limits=limits,
            cpu_threads=identity['runtime']['torch_threads'],interop_threads=identity['runtime']['torch_interop_threads'])
        cold=time.monotonic();require(owner.start(), 'resident start failed')
        cold=time.monotonic()-cold;pid=owner.pid;times=[]
        broker=ResidentBroker(owner);lease=None
        try:
            lease=broker.lease()
            for case_id,prefix in cases:
                # The broker adds its lease namespace; keep the owner ID bounded.
                request_id=digest({'case_id':case_id})
                t=time.monotonic();status,result=lease.predict(
                    prefix,request_id=request_id,timeout_s=limits.query_s)
                require(status == 'SUBMITTED', 'resident PRE not admitted')
                require(result is not None and result.status == 'SUCCEEDED'
                        and result.matrix.values == model.predict(prefix).probabilities, 'resident numerical mismatch')
                times.append(time.monotonic()-t)
                require(owner.pid == pid and lease.predict(
                    prefix,request_id=request_id,timeout_s=limits.query_s)[0] == 'DUPLICATE',
                        'model reload/dedup mismatch')
        finally:
            if lease is not None: lease.close()
            broker.close()
        close=owner.close();require(close == 'CLOSED' and owner.cleanup['pid_exit_confirmed'], 'resident cleanup failed')
        audit=owner.inspect(expected_head=owner.journal.head)
        require(not audit['incomplete'] and not audit['execution_unconfirmed']
                and sum(e['kind']=='READY' for e in audit['events']) == 1, 'incomplete lifecycle')
        verify_snapshot(source_path,source_digest)
        # Reverify exact immutable payloads after inference; never repair a capsule.
        require(verify_artifact(capsule_path,expected_artifact_type='observer_dtrs_final_tom_inference_capsule',
            expected_schema_version='observer_dtrs_final_tom_inference_capsule_v1').manifest_digest == capsule_digest,
            'capsule changed')
        report={'schema_version':VERSION,'status':('REAL_MODEL_LOCAL_VALIDATED' if capsule_scope ==
                'PINNED_FINAL_INFERENCE_UNQUALIFIED' else 'QUAL_PREPARATION_ONLY'),
            'scope':capsule_scope,
            'authorization':'NOT_FORMAL_DEPLOYMENT_NOT_CAL_TEST', 'source_snapshot_digest':source_digest,
            'source_inventory_digest':snapshot['inventory_digest'],'base_head':snapshot['base_head'],
            'capsule_identity':identity,'inputs_raw_sha256':pre_sha256,'cases':numerical,
            'provenance':provenance,
            'comparison':{'atol':0,'rtol':0,'reference_kind':reference['reference_kind'],
                          'reference_runtime':reference['runtime']},
            'resident':{'journal_head':owner.journal.head,'submitted':audit['submitted'],
                'terminal':audit['terminal'],'one_load_one_PID':True,'exit_confirmed':True,
                'cold_ready_seconds':cold,'query_seconds':times,'limits':asdict(limits)},
            'wall_seconds':time.monotonic()-started,
            'resource_evidence':{'OS_ACL_sandbox':'NOT_ESTABLISHED',
                'process_tree':'OWNED_INFERENCE_PID_EXIT_ONLY',
                'RSS_and_VRAM_budget_qualification':'NOT_RUN',
                'GPU_or_remote_cancel':'NOT_IMPLEMENTED_OR_QUALIFIED'},
            'not_established':['original_sealed_predictor_initialization','full_semantic_recovery',
                'OS_ACL_sandbox','process_tree_and_GPU_release','real_fault_campaign',
                'formal_CAL_package','Shadow_game_hook_adapter','CAL_TEST_authorization']}
        return publish_artifact(output/'complete',manifest_fields={'artifact_type':VERSION,'schema_version':VERSION},
            files={'report.json':canonical_json_bytes(report),'reference.json':canonical_json_bytes(reference)})
    except Exception as error:
        if owner is not None: owner.close()
        checks={m:code for m,code in (
            ('reference origin/scope mismatch','REFERENCE_ORIGIN'),
            ('reference CPU runtime mismatch','REFERENCE_RUNTIME'),
            ('restored tensor mismatch','MODEL_RESTORE'),('Temporal value mismatch','TEMPORAL'),
            ('reference coverage','REFERENCE_COVERAGE'),('reference case identity','REFERENCE_IDENTITY'),
            ('reference PRE/plan mismatch','PRE_PLAN'),('input encoding mismatch','INPUT_ENCODING'),
            ('log-Q mismatch (atol=rtol=0)','LOG_Q'),('Q conversion mismatch (atol=rtol=0)','Q_CONVERSION'),
            ('non-deterministic repeated PRE','REPEATED_PRE'),('resident numerical mismatch','RESIDENT_Q'))}
        publish_artifact(output/'failure',manifest_fields={'artifact_type':VERSION,'schema_version':VERSION},
            files={'report.json':canonical_json_bytes({'schema_version':VERSION,'status':'QUAL_BLOCKED',
                'source_snapshot_digest':source_digest,'capsule_digest':capsule_digest,
                'reason':'QUAL_MISMATCH_OR_LIFECYCLE_FAILURE_NO_FALLBACK','failed_stage':stage,
                'failed_check':checks.get(str(error),'UNCLASSIFIED_STAGE_FAILURE'),
                'cleanup':dict(owner.cleanup) if owner else None})})
        raise


def qualify_cpu(**kwargs):
    """Real pinned model CPU check; cannot select artificial origin via the CLI."""
    return _qualify_cpu(**kwargs,capsule_scope='PINNED_FINAL_INFERENCE_UNQUALIFIED',
                        pre_scope='PUBLIC_PRE_QUALIFICATION_ONLY')
