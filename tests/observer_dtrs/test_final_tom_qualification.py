"""Artificial tensors only; real-final checks are NOT_RUN without external inputs."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import torch

from tests.observer_dtrs.conftest import prefix_for
from tests.observer_dtrs.test_final_tom_capsule import artificial,load,republish
from tests.observer_dtrs.test_final_tom_audit import git
from werewolf.artifact_io import canonical_json_bytes,read_artifact_file,sha256_bytes
from werewolf.observer_dtrs import qual_sources as sources
from werewolf.observer_dtrs.final_tom_audit import checkout_identity
from werewolf.observer_dtrs import final_tom_qualification as qual
from werewolf.observer_dtrs.final_tom_capsule import encode_public_pre


def packet(path,scope='CONTROLLED_FIXTURE'):
    path.parent.mkdir(parents=True,exist_ok=True)
    from tests.canonical_collection.test_final_capacity import sparse_pre
    p=prefix_for(actor='player5',dead=('player2',),game_id='artificial-short')
    pk=prefix_for(phase='pk_discussion',actor='player5',game_id='artificial-pk')
    r={'schema_version':qual.PRE_VERSION,'scope':scope,
       'cases':[{'case_id':str(i),'pre':v.to_record()} for i,v in enumerate((p,pk,sparse_pre(256)))]}
    path.write_bytes(canonical_json_bytes(r))
    return path,sha256_bytes(path.read_bytes())


def reference(model,cases,sha):
    rows=[]
    for case,prefix in cases:
        plan,x=encode_public_pre(prefix)
        with torch.inference_mode(): logq=model._model(**x)[0]
        rows.append({'case_id':case,'pre_sha256':sources.digest(prefix.to_record()),
            'plan_digest':plan.plan_digest,'token_count':plan.token_count,
            'inputs':{k:qual._tensor(v[0]) for k,v in x.items()},
            'log_q_nonself':[[float(logq[i,j]) for j in range(7) if i!=j] for i in range(7)],
            'q':logq.exp().tolist()})
    origin=model.identity['origin']
    return {'schema_version':'observer_dtrs_qual_reference_v1',
        'reference_kind':'PINNED_ORIGINAL_GRAPH_CPU_SAME_CAPSULE_NOT_SEALED_LIFECYCLE',
        'source_revision':origin['source_revision'],'implementation_digest':origin['implementation_digest'],
        'runtime':{k:model.identity['runtime'][k] for k in ('python','packages','device','dtype',
                   'attention_backend','torch_threads','torch_interop_threads')},
        'capsule_digest':model.identity['capsule_digest'],'inputs_raw_sha256':sha,
        'model_tensors':{k:qual._tensor(v) for k,v in model._model.state_dict().items()},
        'temporal':{k:sha256_bytes(getattr(model._model.temporal,k).tobytes()) for k in ('day','phase')},
        'full_semantic_recovery':'NOT_RUN','original_sealed_predictor':'NOT_RUN','cases':rows}


def test_snapshot_exact_bytes_and_added_file_rejection(tmp_path):
    a=sources.publish_snapshot(tmp_path/'source')
    r=sources.verify_snapshot(a.path,a.manifest_digest)
    assert r['git_head_is_complete_D4_identity'] is False
    assert 'werewolf/observer_dtrs/final_tom_client.py' in r['inventory']
    mirror=tmp_path/'mirror'
    for n in r['inventory']:
        target=mirror/n;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(read_artifact_file(a,'sources/'+n));target.chmod(r['inventory'][n]['mode'])
    sources.verify_snapshot(a.path,a.manifest_digest,root=mirror)
    (mirror/'werewolf/unreviewed.py').write_text('unreviewed = True\n')
    with pytest.raises(ValueError,match='inventory'):sources.verify_snapshot(a.path,a.manifest_digest,root=mirror)


@pytest.mark.parametrize('change',['modify','alias','wrong_pin'])
def test_snapshot_tamper_or_alias(tmp_path,change):
    a=sources.publish_snapshot(tmp_path/'source')
    if change=='modify':(a.path/'snapshot.json').write_bytes(b'{}')
    elif change=='alias':
        alias=tmp_path/'alias';alias.symlink_to(sources.ROOT,target_is_directory=True)
        with pytest.raises(ValueError): sources.new_external_path(alias/'bad')
        return
    with pytest.raises(ValueError):sources.verify_snapshot(a.path,'e'*64 if change=='wrong_pin' else a.manifest_digest)


@pytest.mark.parametrize('change',['extra_target','duplicate','missing','raw_sha','over_capacity','nan'])
def test_public_packet_rejects_bad_identity_supervision_and_capacity(tmp_path,change):
    path,sha=packet(tmp_path/'pre.json');r=json.loads(path.read_bytes())
    if change=='extra_target':r['cases'][0]['Y']=1
    elif change=='duplicate':r['cases'].append(r['cases'][0])
    elif change=='missing':r['cases']=[]
    elif change=='over_capacity':
        from tests.canonical_collection.test_final_capacity import sparse_pre
        r['cases'][0]['pre']=sparse_pre(257).to_record()
    elif change=='nan':r['cases'][0]['pre']['step_index']='NaN'
    if change!='raw_sha':path.write_bytes(canonical_json_bytes(r));sha=sha256_bytes(path.read_bytes())
    else:sha='e'*64
    with pytest.raises((ValueError,TypeError,KeyError)):qual.read_pres(path,sha)


@pytest.mark.parametrize('change',['none','source','tensor','temporal','encoding','logq','q','missing','duplicate','scope'])
def test_exact_comparison_stages(artificial,tmp_path,change):
    model=load(artificial['capsule']);path,sha=packet(tmp_path/'pre.json');_,cases=qual.read_pres(path,sha)
    r=reference(model,cases,sha)
    if change=='source':r['source_revision']='e'*40
    elif change=='tensor':next(iter(r['model_tensors'].values()))['sha256']='e'*64
    elif change=='temporal':r['temporal']['day']='e'*64
    elif change=='encoding':next(iter(r['cases'][0]['inputs'].values()))['sha256']='e'*64
    elif change=='logq':r['cases'][0]['log_q_nonself'][0][0]+=1e-7
    elif change=='q':r['cases'][0]['q'][0][1]+=1e-7
    elif change=='missing':r['cases'].pop()
    elif change=='duplicate':r['cases'][1]=deepcopy(r['cases'][0])
    elif change=='scope':r['original_sealed_predictor']='PASS'
    if change=='none':assert len(qual.compare_reference(model,cases,r,inputs_sha256=sha))==3
    else:
        with pytest.raises(ValueError):qual.compare_reference(model,cases,r,inputs_sha256=sha)


@pytest.fixture(scope='module')
def original_fixture(tmp_path_factory,artificial):
    root=tmp_path_factory.mktemp('isolated-historical-graph')
    shutil.copytree(sources.ROOT/'werewolf',root/'werewolf',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(sources.ROOT/'run_random.py',root/'run_random.py')
    shutil.copytree(sources.ROOT/'configs',root/'configs')
    git(root,'init','-q');git(root,'add','--','werewolf','run_random.py','configs')
    git(root,'commit','-qm','controlled reference bytes only')
    identity=checkout_identity(root)
    a=republish(artificial['capsule'],root.parent/(root.name+'-capsule'),
        manifest_change=lambda m:m['origin'].update({k:identity[k] for k in ('source_revision','implementation_digest')}))
    return root,a


def test_isolated_original_graph_subprocess_and_full_controlled_qual(original_fixture,tmp_path,monkeypatch):
    from werewolf.observer_dtrs.final_tom_broker import ResidentBroker
    calls=[];original_lease=ResidentBroker.lease
    def traced_lease(broker):
        lease=original_lease(broker);predict=lease.predict
        def traced_predict(*args,**kwargs):
            calls.append(kwargs['request_id'])
            return predict(*args,**kwargs)
        monkeypatch.setattr(lease,'predict',traced_predict)
        return lease
    monkeypatch.setattr(ResidentBroker,'lease',traced_lease)
    root,a=original_fixture;model=load(a);path,sha=packet(tmp_path/'inputs/pre.json')
    inputs=json.loads(path.read_bytes());inputs['cases'][0]['case_id']='x'*128
    path.write_bytes(canonical_json_bytes(inputs));sha=sha256_bytes(path.read_bytes())
    _,cases=qual.read_pres(path,sha)
    r=qual.capture_reference(root,model,{'capsule_path':a.path,'pre_path':path},sha)
    assert len(qual.compare_reference(model,cases,r,inputs_sha256=sha))==3
    assert r['original_sealed_predictor']==r['full_semantic_recovery']=='NOT_RUN'
    snapshot=sources.publish_snapshot(tmp_path/'source')
    result=qual._qualify_cpu(capsule_path=a.path,capsule_digest=a.manifest_digest,
        source_path=snapshot.path,source_digest=snapshot.manifest_digest,pre_path=path,pre_sha256=sha,
        reference_checkout=root,destination=tmp_path/'run',capsule_scope='CONTROLLED_FIXTURE',pre_scope='CONTROLLED_FIXTURE')
    report=json.loads(read_artifact_file(result,'report.json'))
    assert report['status']=='QUAL_PREPARATION_ONLY'
    assert report['resident']['one_load_one_PID'] and report['resident']['exit_confirmed']
    assert calls == [sources.digest({'case_id':case_id}) for case_id,_ in cases for _ in range(2)]
    assert len(report['cases'])==3 and 'formal_CAL_package' in report['not_established']


def test_public_cpu_entry_rejects_artificial_pre_without_model_load(tmp_path):
    path,sha=packet(tmp_path/'inputs/pre.json');snapshot=sources.publish_snapshot(tmp_path/'source')
    (tmp_path/'absent').mkdir()  # exists, but loading must not be reached
    with pytest.raises(ValueError,match='scope'):
        qual.qualify_cpu(capsule_path=tmp_path/'absent',capsule_digest='e'*64,source_path=snapshot.path,
            source_digest=snapshot.manifest_digest,pre_path=path,pre_sha256=sha,
            reference_checkout=sources.ROOT,destination=tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_public_cli_snapshot_and_verify(tmp_path):
    command=[sys.executable,'-B',str(sources.ROOT/'scripts/qualify_final_tom_capsule.py')]
    r=subprocess.run(command+['snapshot','--output',str(tmp_path/'source')],capture_output=True,text=True,timeout=30)
    assert r.returncode==0,r.stdout+r.stderr
    d=json.loads(r.stdout)['source_snapshot_manifest']
    r=subprocess.run(command+['verify-source','--snapshot',str(tmp_path/'source'),'--snapshot-digest',d],
                     capture_output=True,text=True,timeout=30)
    assert r.returncode==0 and json.loads(r.stdout)['source_bytes']=='PASS'


def test_provenance_packet_reuses_existing_reader_without_fixture_release(artificial,tmp_path,monkeypatch):
    from werewolf.observer_dtrs import final_tom_capsule_build as build
    model=load(artificial['capsule'])
    # The actual production reader must reject the synthetic pins/scope.
    with pytest.raises(ValueError):
        qual.verify_provenance_packet(model,artificial['audit'],artificial['audit_sha'])
    with monkeypatch.context() as m:
        m.setattr(build,'_audit_packet',lambda *a,**kw:deepcopy(artificial['report']))
        r=qual.verify_provenance_packet(model,artificial['audit'],artificial['audit_sha'])
        assert r['scope']=='CHAIN_FILES_BYTES_AND_CODEC_NOT_SEMANTIC_RECOVERY'
        with pytest.raises(ValueError,match='provenance'):
            qual.verify_provenance_packet(model,artificial['audit'],'e'*64)


@pytest.mark.parametrize('case',['direct','symlink','dotdot','protected'])
def test_output_refusal_keeps_sources_unchanged(tmp_path,case):
    protected=tmp_path/'input';protected.mkdir();(protected/'sentinel').write_bytes(b'KEEP')
    if case=='direct':target=sources.ROOT/'QUAL_FORBIDDEN'
    elif case=='symlink':
        alias=tmp_path/'alias';alias.symlink_to(protected,target_is_directory=True);target=alias/'output'
    elif case=='dotdot':target=tmp_path/'input/../output'
    else:target=protected/'output'
    with pytest.raises(ValueError):sources.new_external_path(target,protected=(protected,))
    assert sorted(p.name for p in protected.iterdir())==['sentinel']
    assert (protected/'sentinel').read_bytes()==b'KEEP'


def test_failed_numerical_run_seals_failure_without_resident_or_fallback(original_fixture,tmp_path,monkeypatch):
    from werewolf.artifact_io import verify_artifact
    root,a=original_fixture;model=load(a);path,sha=packet(tmp_path/'inputs/pre.json')
    _,cases=qual.read_pres(path,sha);ref=reference(model,cases,sha)
    ref['cases'][0]['log_q_nonself'][0][0]+=1e-7
    snapshot=sources.publish_snapshot(tmp_path/'source')
    monkeypatch.setattr(qual,'capture_reference',lambda *a,**kw:ref)
    with pytest.raises(ValueError,match='log-Q mismatch'):
        qual._qualify_cpu(capsule_path=a.path,capsule_digest=a.manifest_digest,
            source_path=snapshot.path,source_digest=snapshot.manifest_digest,pre_path=path,pre_sha256=sha,
            reference_checkout=root,destination=tmp_path/'run',capsule_scope='CONTROLLED_FIXTURE',pre_scope='CONTROLLED_FIXTURE')
    failure=verify_artifact(tmp_path/'run/failure',expected_artifact_type=qual.VERSION,expected_schema_version=qual.VERSION)
    r=json.loads(read_artifact_file(failure,'report.json'))
    assert r['status']=='QUAL_BLOCKED' and r['failed_stage']=='tensor_temporal_encoding_logq_q_comparison'
    assert r['failed_check']=='LOG_Q'
    assert not (tmp_path/'run/complete').exists() and not (tmp_path/'run/resident').exists()
