"""Artificial resident process + independent spawned client. No game integration."""
from dataclasses import replace
from copy import deepcopy
import json
import multiprocessing as mp
import time

import pytest

from tests.observer_dtrs.conftest import prefix_for
from tests.observer_dtrs.test_final_tom_capsule import artificial
from tests.observer_dtrs.test_final_tom_service import factory
from tests.observer_dtrs.qual_broker_fixtures import roundtrip,disappear_after_dispatch
from werewolf.observer_dtrs.final_tom_broker import ResidentBroker,_message
from werewolf.observer_dtrs.final_tom_wire import Channel,Limits,ProtocolError,PeerClosed,WireTimeout


def wait(predicate):
    deadline=time.monotonic()+5
    while time.monotonic()<deadline:
        if predicate():return
        time.sleep(.01)
    pytest.fail('bounded fixture wait expired')


def test_spawn_client_multiple_PRE_without_owner_pickle(factory,tmp_path):
    make,model=factory;owner=make();assert owner.start()
    broker=ResidentBroker(owner);lease=broker.lease()
    pres=[prefix_for(game_id='broker-a'),prefix_for(game_id='broker-b',phase='pk_discussion',dead=('player2',))]
    child=mp.get_context('spawn').Process(target=roundtrip,args=(lease,[p.to_record() for p in pres],tmp_path/'result.json'))
    try:
        child.start();broker.release_parent_copy(lease);child.join(timeout=15)
        assert child.exitcode==0
        r=json.loads((tmp_path/'result.json').read_text())
        assert r['restricted_imports'] is False
        assert all(x['admission']=='SUBMITTED' and x['status']=='SUCCEEDED' for x in r['rows'])
        assert [x['matrix'] for x in r['rows']]==[[list(row) for row in model.predict(p).probabilities] for p in pres]
        assert owner.inspect()['submitted']==2
    finally:
        if child.is_alive():child.kill();child.join(timeout=2)
        broker.close()
    assert owner.close()=='CLOSED'


def test_disconnected_spawn_client_cancels_underlying_execution(factory):
    make,_=factory;owner=make('delay');assert owner.start()
    broker=ResidentBroker(owner);lease=broker.lease()
    child=mp.get_context('spawn').Process(target=disappear_after_dispatch,args=(lease,prefix_for().to_record()))
    try:
        child.start();broker.release_parent_copy(lease);child.join(timeout=10)
        assert child.exitcode==23
        wait(lambda:len([e for e in owner.journal.events if e['kind']=='OUTCOME'])==1)
        r=owner.inspect()
        assert r['terminal']==1 and not r['execution_unconfirmed']
        assert 'SUCCEEDED' not in r['outcome_counts']
        assert owner.state=='IDLE'  # actual thread joined before accepting more
    finally:
        if child.is_alive():child.kill();child.join(timeout=2)
        broker.close()


def test_broker_disconnect_hard_hang_cannot_reuse_old_service(factory):
    make,_=factory;owner=make('hang',replace(Limits(),cancel_s=.1));assert owner.start()
    broker=ResidentBroker(owner);lease=broker.lease()
    child=mp.get_context('spawn').Process(target=disappear_after_dispatch,args=(lease,prefix_for().to_record()))
    try:
        child.start();broker.release_parent_copy(lease);child.join(timeout=10)
        wait(lambda:owner.state=='UNHEALTHY' and owner._process.poll() is not None)
        with pytest.raises(ValueError):broker.lease()
        assert owner.cleanup['pid_exit_confirmed']
        wait(lambda:owner._pending==0)
        assert owner.recover()
        new=broker.lease();assert new.binding['epoch']==owner.epoch
        new.close()
    finally:
        if child.is_alive():child.kill();child.join(timeout=2)
        broker.close()


@pytest.mark.parametrize('case',['token','pre','deadline','extra','request_id'])
def test_bad_broker_input_never_admits_inference(factory,case):
    make,_=factory;owner=make();assert owner.start()
    broker=ResidentBroker(owner);lease=broker.lease();binding=lease.binding
    payload={'pre':prefix_for().to_record(),'deadline_monotonic':time.monotonic()+2}
    if case=='token':binding['token']='e'*64
    elif case=='pre':payload['pre']['Y']=1
    elif case=='deadline':payload['deadline_monotonic']=time.monotonic()-1
    elif case=='extra':payload['support']=0
    request_id='x'*129 if case=='request_id' else 'bad'
    channel=Channel(lease.sock,lease.limits)
    try:
        channel.write(_message('QUERY',binding,request_id,payload),time.monotonic()+1)
        with pytest.raises((PeerClosed,WireTimeout,ProtocolError)):
            channel.read(time.monotonic()+1)
        assert owner.inspect()['submitted']==0
    finally:
        lease.close();broker.close()


def test_lease_bound_duplicate_and_same_PID(factory):
    make,_=factory;owner=make();assert owner.start();pid=owner.pid
    broker=ResidentBroker(owner);lease=broker.lease()
    try:
        with pytest.raises(ValueError):broker.lease()
        p=prefix_for()
        assert lease.predict(p,request_id='one',timeout_s=3)[1].status=='SUCCEEDED'
        assert lease.predict(p,request_id='one',timeout_s=3)==('DUPLICATE',None)
        assert owner.pid==pid and owner.inspect()['submitted']==1
    finally:
        lease.close();broker.close()


def test_broker_request_id_boundaries_preserve_dedup_and_lease_isolation(factory):
    make,_=factory;owner=make();assert owner.start();pid=owner.pid
    broker=ResidentBroker(owner);lease=broker.lease();second=None
    try:
        for length in (111,112,128):
            request_id='x'*length;p=prefix_for(game_id=f'broker-id-{length}')
            admission,result=lease.predict(p,request_id=request_id,timeout_s=3)
            assert admission=='SUBMITTED' and result.status=='SUCCEEDED'
            assert lease.predict(p,request_id=request_id,timeout_s=3)==('DUPLICATE',None)
        lease.close();wait(lambda:not broker._sessions[0][1].is_alive())
        second=broker.lease()
        # The same public ID in a new lease belongs to a separate namespace.
        admission,result=second.predict(prefix_for(game_id='broker-second-lease'),request_id='x'*128,timeout_s=3)
        assert admission=='SUBMITTED' and result.status=='SUCCEEDED'
        audit=owner.inspect()
        assert audit['submitted']==audit['terminal']==4 and owner.pid==pid
        internal_ids=[e['request_id'] for e in owner.journal.events if e['kind']=='ADMISSION']
        assert len(set(internal_ids))==4 and all(len(rid)<=128 for rid in internal_ids)
    finally:
        lease.close()
        if second is not None:second.close()
        broker.close()


def test_broker_lifetime_lease_quota_is_bounded(factory):
    make,_=factory;owner=make(limits=replace(Limits(),max_requests=1));assert owner.start()
    broker=ResidentBroker(owner);lease=broker.lease()
    lease.close();wait(lambda:not broker._sessions[0][1].is_alive())
    try:
        with pytest.raises(ValueError):broker.lease()
        assert owner.inspect()['submitted']==0
    finally:
        broker.close()


@pytest.fixture
def capsule_client(artificial,tmp_path,monkeypatch):
    """仅隔离真实来源回执验证；Owner、Broker、PRE 和进程实际运行。"""
    from werewolf.observer_dtrs import final_tom_client as client
    from tests.observer_dtrs.test_final_tom_capsule import load
    model=load(artificial['capsule']);clients=[]
    monkeypatch.setattr(client,'_qualified_capsule',lambda *args:{
        'capsule_identity':model.identity,'base_head':'a'*40,'source_snapshot_digest':'b'*64})
    def make():
        result=client.CapsuleGameplayPredictor(artificial['capsule'].path,
            capsule_digest=artificial['capsule'].manifest_digest,
            qualification_path=tmp_path/'unused',qualification_digest='c'*64,
            qualification_source=tmp_path/'unused-source',journal_root=tmp_path/f'interface-{len(clients)}')
        clients.append(result);return result
    yield make,model
    for c in clients:c.close()


def test_capsule_predict_interface_Q_identity_duplicate_and_close(capsule_client):
    from werewolf.phase2_online_preflight import _frozen_q_runtime_ready
    from werewolf.phase2_online_runner import predict_q,predictor_source_digest
    make,model=capsule_client;c=make()
    assert not _frozen_q_runtime_ready(c,controlled=True)
    with c:
        p=prefix_for(dead=('player2',));pid=c._owner.pid
        assert _frozen_q_runtime_ready(c,controlled=True)
        assert not _frozen_q_runtime_ready(c)
        with pytest.raises(ValueError,match='SCOPE_NOT_QUALIFIED'):predict_q(c,p)
        q=predict_q(c,p,controlled=True)
        assert q==model.predict(p).probabilities and len(q)==7 and all(len(row)==7 for row in q)
        assert q[2][1]>0 and predictor_source_digest(c)!=c.identity['model_origin']['seal_digest']
        assert c.identity['execution']['capsule']['origin']==c.identity['model_origin']
        assert c.identity['scope']=='CAPSULE_CPU_QUALIFICATION_ONLY'
        from werewolf.phase2_online_runner import OnlineTerminalPilotRunnerV1
        runner=object.__new__(OnlineTerminalPilotRunnerV1);runner.predictor=c
        with pytest.raises(ValueError,match='SCOPE_NOT_QUALIFIED'):runner.start_game('blocked')
        with pytest.raises(ValueError,match='duplicate PRE'):c.predict(p)
        assert c._owner.pid==pid and c._owner.inspect()['submitted']==1
    assert c.close()=='CLOSED' and c._owner.cleanup['pid_exit_confirmed']
    assert not _frozen_q_runtime_ready(c,controlled=True)
    with pytest.raises(ValueError):c.predict(p)


def test_controlled_preflight_rejects_duck_typed_provider():
    from types import SimpleNamespace
    from werewolf.phase2_online_preflight import assess_capsule_q_interface,_frozen_q_runtime_ready
    fake=SimpleNamespace(predict=lambda p:(),require_ready=lambda **kw:None,seal_digest='a'*64)
    assert not _frozen_q_runtime_ready(fake,controlled=True)
    with pytest.raises(ValueError):assess_capsule_q_interface(fake,prefix_for())


def test_capsule_interface_never_retries_uncertain_request(capsule_client,monkeypatch):
    make,_=capsule_client;c=make();c.start();calls=[]
    def uncertain(*args,**kwargs):
        calls.append(kwargs['request_id']);raise TimeoutError('unknown request result')
    monkeypatch.setattr(c._lease,'predict',uncertain)
    with pytest.raises(TimeoutError):c.predict(prefix_for())
    with pytest.raises(ValueError):c.predict(prefix_for())
    assert len(calls)==1 and c._owner.cleanup['pid_exit_confirmed']


@pytest.mark.parametrize('lane',['gameplay','phase2'])
def test_real_server_Q_interface_branch_never_builds_gameplay(capsule_client,tmp_path,monkeypatch,capsys,lane):
    from types import SimpleNamespace
    from werewolf.observer_dtrs.final_tom_wire import ROOT
    from werewolf import phase2_gameplay_server as gameplay,phase2_online_server as phase2
    make,model=capsule_client
    def forbidden(*args,**kwargs):pytest.fail('full campaign must not be entered')
    monkeypatch.setattr(gameplay,'server_inputs',forbidden)
    monkeypatch.setattr(phase2,'load_server_inputs',forbidden)
    with make() as c:
        p=prefix_for(game_id=lane)
        if lane=='gameplay':
            r=gameplay.execute_server(None,repo=ROOT,work=tmp_path/'unused',destination=tmp_path/'unused-pub',
                runtime_check_only=True,predictor=c,public_pre=p)
            with pytest.raises(ValueError):
                gameplay.execute_server(None,repo=ROOT,work=tmp_path/'unused',destination=tmp_path/'unused-pub',
                    allow_gpu=True,predictor=c,public_pre=p)
        else:
            args=SimpleNamespace(repo=ROOT,preflight_only=True,resume=False)
            assert phase2.execute_server_campaign(args,predictor=c,public_pre=p)==0
            r=json.loads(capsys.readouterr().out)['q_interface']
            args.preflight_only=False
            with pytest.raises(ValueError):phase2.execute_server_campaign(args,predictor=c,public_pre=p)
        assert r['status']=='CONTROLLED_Q_INTERFACE_PASS'
        assert r['online_gameplay_ready'] is False and r['full_campaign_preflight_passed'] is False
        assert 'CAPSULE_ONLINE_SCOPE_NOT_QUALIFIED' in r['blockers']
        assert r['pre']['prefix_digest']==p.prefix_digest
        assert r['q_source_digest']==c.source_digest
        assert tuple(map(tuple,r['Q']))==model.predict(p).probabilities
    assert not (tmp_path/'unused').exists() and not (tmp_path/'unused-pub').exists()


@pytest.mark.parametrize('change',['gold','future','invalid_Q','wrong_plan','wrong_identity'])
def test_capsule_interface_rejects_invalid_input_or_result(capsule_client,monkeypatch,change):
    from werewolf.observer_dtrs.final_tom_client import ResidentResult
    make,_=capsule_client
    with make() as c:
        p=prefix_for()
        if change in ('gold','future'):
            bad=p.to_record();bad['Y' if change=='gold' else 'future_events']=1
            with pytest.raises((TypeError,ValueError,AttributeError)):c.predict(bad)
            assert c._owner.inspect()['submitted']==0
            return
        original=c._lease.predict
        def corrupt(*args,**kwargs):
            admission,r=original(*args,**kwargs);record=r.record
            if change=='wrong_plan':record['prediction']['plan_digest']='e'*64
            elif change=='wrong_identity':record['prediction']['capsule_identity']['origin']['seal_digest']='e'*64
            else:record['prediction']['probabilities'][0][0]=.5
            from werewolf.artifact_io import canonical_json_bytes
            return admission,ResidentResult(canonical_json_bytes(record))
        monkeypatch.setattr(c._lease,'predict',corrupt)
        with pytest.raises(ValueError):c.predict(p)
        assert c._closed and c._owner.cleanup['pid_exit_confirmed']


@pytest.fixture(scope='module')
def qualified_metadata(artificial,tmp_path_factory):
    """人工回执用于来源拒绝测试；不会作为真实模型资格证据。"""
    from tests.observer_dtrs.test_final_tom_capsule import load,republish
    from werewolf.observer_dtrs.qual_sources import publish_snapshot
    from werewolf.observer_dtrs.final_tom_audit import _Pins
    from werewolf.artifact_io import publish_artifact,canonical_json_bytes
    root=tmp_path_factory.mktemp('qualified-interface-metadata');identity=load(artificial['capsule']).identity
    pins=_Pins();identity['origin'].update(source_revision=pins.revision,fit_digest=pins.fit,
        terminal_digest=pins.terminal,seal_digest=pins.seal,model_container_descriptor_sha256=pins.model_descriptor)
    identity['scope']='PINNED_FINAL_INFERENCE_UNQUALIFIED'
    a=republish(artificial['capsule'],root/'capsule',manifest_change=lambda m:m.update(
        scope=identity['scope'],origin=identity['origin']))
    identity['capsule_digest']=a.manifest_digest
    source=publish_snapshot(root/'source');saved=json.loads((source.path/'snapshot.json').read_bytes())
    runtime=identity['runtime'];state={'parameters_frozen':True,'model_training':False,'forward_inference_mode':True}
    ref={'runtime':{**runtime,'parameters_frozen':True,'model_training':False},
         'capsule_digest':a.manifest_digest,'source_revision':pins.revision,
         'implementation_digest':identity['origin']['implementation_digest'],'inputs_raw_sha256':'d'*64,
         'original_sealed_predictor':'NOT_RUN','full_semantic_recovery':'NOT_RUN',
         'reference_kind':'PINNED_ORIGINAL_GRAPH_CPU_SAME_CAPSULE_NOT_SEALED_LIFECYCLE'}
    report={'capsule_identity':identity,'status':'REAL_MODEL_LOCAL_VALIDATED',
        'authorization':'NOT_FORMAL_DEPLOYMENT_NOT_CAL_TEST','scope':identity['scope'],
        'comparison':{'atol':0,'rtol':0,'inference_state':state,'reference_runtime':ref['runtime'],
                      'reference_kind':ref['reference_kind']},
        'cases':[{'max_abs_log_q_error':0,'max_abs_Q_error':0,'status':'PASS_EXACT_CPU_GRAPH_ENCODING_LOGQ_Q_AND_REPEAT'}],
        'resident':{'one_load_one_PID':True,'exit_confirmed':True,'submitted':1,'terminal':1},
        'base_head':saved['base_head'],'source_snapshot_digest':source.manifest_digest,'inputs_raw_sha256':'d'*64}
    manifest={'artifact_type':'observer_dtrs_cpu_qualification_v1','schema_version':'observer_dtrs_cpu_qualification_v1',
        'capsule_digest':a.manifest_digest,'source_snapshot_digest':source.manifest_digest,'source_commit':saved['base_head'],
        'model_origin':{n:identity['origin'][n] for n in ('source_revision','fit_digest','seal_digest','model_bin_raw_sha256')},
        'runtime':runtime,'inputs_raw_sha256':'d'*64}
    return root,a,source,report,ref,manifest


@pytest.mark.parametrize('change',['none','pin','failed','scope','origin','loader','runtime','error','exit','source','reference'])
def test_interface_requires_pinned_qualified_core(qualified_metadata,tmp_path,change):
    from werewolf.artifact_io import publish_artifact,canonical_json_bytes
    from werewolf.observer_dtrs.final_tom_client import _qualified_capsule
    root,a,source,original,reference,manifest=qualified_metadata
    r,ref,m=deepcopy(original),deepcopy(reference),deepcopy(manifest)
    if change=='failed':r['status']='QUAL_BLOCKED'
    elif change=='scope':r['scope']='FORMAL_GAMEPLAY'
    elif change=='origin':r['capsule_identity']['origin']['seal_digest']='e'*64
    elif change=='loader':r['capsule_identity']['loader_source']['loader']['implementation_digest']='e'*64
    elif change=='runtime':m['runtime']['device']='cuda'
    elif change=='error':r['cases'][0]['max_abs_Q_error']=1e-8
    elif change=='exit':r['resident']['exit_confirmed']=False
    elif change=='source':m['source_commit']='e'*40
    elif change=='reference':ref['original_sealed_predictor']='PASS'
    q=publish_artifact(tmp_path/'qualification',manifest_fields=m,files={
        'report.json':canonical_json_bytes(r),'reference.json':canonical_json_bytes(ref)})
    def check():return _qualified_capsule(a.path,a.manifest_digest,q.path,
        'e'*64 if change=='pin' else q.manifest_digest,source.path)
    if change=='none':assert check()['capsule_identity']['capsule_digest']==a.manifest_digest
    else:
        with pytest.raises(ValueError):check()
