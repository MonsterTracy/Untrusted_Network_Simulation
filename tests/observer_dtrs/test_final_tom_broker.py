"""Artificial resident process + independent spawned client. No game integration."""
from dataclasses import replace
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
