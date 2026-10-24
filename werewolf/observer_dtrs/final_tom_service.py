"""Resident CPU ToM service, owned local IPC only; no Agent/Shadow/targets imports.

The control thread stays responsive while ONE inference thread runs. Cancellation
is acknowledged only after join; a stuck kernel requires owner process teardown.
"""
import os
from pathlib import Path
from queue import Queue, Empty
import secrets
import socket
import sys
from threading import Thread
import time

from werewolf.artifact_io import canonical_json_bytes, sha256_bytes
from werewolf.observer_dtrs.final_tom_wire import (
    Channel, Limits, ROOT, digest, envelope, frame, hex_id, require, service_source, WireTimeout,
)


def serve(fd):
    # Bootstrap has a hard, fixed ceiling; the owner then supplies tighter limits.
    channel = Channel(socket.socket(fileno=fd), Limits())
    initial = channel.read(time.monotonic()+10.)
    require(type(initial) is dict and initial.get('kind') == 'START')
    epoch, boot = initial['epoch'], initial['token']
    hex_id(epoch);hex_id(boot)
    _, config = envelope(initial, epoch=epoch, instance=None, token=boot, request_id=None)
    require(set(config) == {'capsule_path', 'trust', 'limits', 'cpu_threads', 'interop_threads',
                            'entry_sha256', 'transport_scope'})
    limits = Limits(**config['limits']);channel.limits = limits
    require(config['transport_scope'] in ('CAPSULE_CPU_QUALIFICATION_ONLY', 'CONTROLLED_FAULT_FIXTURE'))
    require(all(type(config[n]) is int and 1 <= config[n] <= 64 for n in ('cpu_threads', 'interop_threads')))
    instance = secrets.token_hex(32)
    send = lambda kind, token, request, payload: channel.write(
        frame(kind, epoch, instance, token, request, payload), time.monotonic()+limits.write_s)
    source = service_source()
    require(sha256_bytes(Path(sys.argv[0]).read_bytes()) == config['entry_sha256'], 'entry source mismatch')
    send('BOOTED', boot, None, {'pid': os.getpid(), 'source': source,
         'entry_sha256': config['entry_sha256'], 'transport_scope': config['transport_scope']})
    # Only the isolated child imports torch / the capsule loader.
    import torch
    from werewolf.observer_dtrs.final_tom_capsule import CapsulePredictor
    from werewolf.canonical_collection.game_bundle import _prefix_from_record
    from werewolf.tom.final_capacity import derived_capacity, validate_final_pre
    torch.set_num_threads(config['cpu_threads'])
    torch.set_num_interop_threads(config['interop_threads'])
    trust = config['trust']
    require(set(trust) == {'capsule_digest', 'capsule_scope', 'loader_digest', 'runtime_digest'})
    try:
        predictor = CapsulePredictor(config['capsule_path'], expected_manifest_digest=trust['capsule_digest'],
                                     expected_scope=trust['capsule_scope'], device='cpu')
        identity = predictor.identity
        require(identity['loader_source']['loader']['implementation_digest'] == trust['loader_digest']
                and digest(identity['runtime']) == trust['runtime_digest'], 'loader/runtime trust mismatch')
    except Exception:
        send('START_FAILED', boot, None, {'status': 'SOURCE_OR_LOAD_ERROR', 'state': 'UNHEALTHY'})
        return 2
    session = {'schema_version': 'observer_dtrs_capsule_service_source_v1', 'epoch': epoch, 'instance': instance,
               'capsule': identity, 'service_source': source, 'entry_sha256': config['entry_sha256'],
               'transport_scope': config['transport_scope'], 'qualification': 'NOT_ESTABLISHED'}
    send('READY', boot, None, {'session': session, 'state': 'IDLE', 'pid': os.getpid(), 'model_load_count': 1})
    results = Queue(maxsize=1)
    active = None
    seen_ids, seen_pres = set(), set()
    last = None

    def compute(prefix):
        try:
            prediction = predictor.predict(prefix)
            results.put_nowait(('SUCCEEDED', {
                'game_id': prediction.game_id, 'boundary_id': prediction.boundary_id,
                'prefix_digest': prediction.prefix_digest, 'plan_digest': prediction.plan_digest,
                'token_count': prediction.token_count, 'probabilities': prediction.probabilities,
                'capsule_identity': prediction.identity}))
        except Exception:
            results.put_nowait(('INFERENCE_ERROR', None))

    while True:
        if active is not None:
            if time.monotonic() >= active['deadline']:
                active['timed_out'] = True
            if not active['thread'].is_alive():
                active['thread'].join(timeout=0)
                status, prediction = results.get_nowait()
                if active['timed_out']:
                    status, prediction = 'SERVICE_TIMEOUT', None
                elif active['cancelled']:
                    status, prediction = 'CANCELLED', None
                send('RESULT', active['token'], active['request_id'], {
                    'status': status, 'prediction': prediction, 'session': session,
                    'execution_stopped': True, 'state': 'IDLE'})
                last, active = active, None
        try:
            message = channel.read(time.monotonic()+.01)
        except WireTimeout:
            continue
        request_id, token = message.get('request_id'), message.get('token')
        if message.get('kind') == 'CLOSE':
            kind, payload = envelope(message, epoch=epoch, instance=instance, token=boot, request_id=None)
            require(active is None and payload == {}, 'close while executing')
            send('CLOSED', boot, None, {'state': 'CLOSED', 'session': session})
            return 0
        require(type(request_id) is str and 0 < len(request_id) <= 128)
        hex_id(token)
        kind, payload = envelope(message, epoch=epoch, instance=instance, token=token, request_id=request_id)
        if kind == 'CANCEL':
            require(payload == {})
            work = active if active is not None else last
            require(work is not None and (work['request_id'], work['token']) == (request_id, token), 'unknown cancel lease')
            if active is not None:
                active['cancelled'] = True  # never acknowledge while thread/kernel is live
            else:
                send('RESULT', token, request_id, {'status': 'CANCELLED', 'prediction': None,
                    'session': session, 'execution_stopped': True, 'state': 'IDLE'})
            continue
        require(kind == 'QUERY' and active is None and len(seen_ids) < limits.max_requests)
        require(set(payload) == {'pre', 'pre_identity', 'deadline_monotonic', 'session_digest'})
        require(payload['session_digest'] == digest(session))
        require(len(canonical_json_bytes(payload['pre'])) <= limits.max_request_bytes, 'PRE too large')
        prefix = _prefix_from_record(payload['pre'])
        require(prefix.to_record() == payload['pre'], 'PRE canonical roundtrip')
        plan = validate_final_pre(prefix, derived_capacity(1024))
        pre = {'game_id': prefix.game_id, 'boundary_id': prefix.boundary_id, 'prefix_digest': prefix.prefix_digest,
               'plan_digest': plan.plan_digest, 'token_count': plan.token_count}
        require(pre == payload['pre_identity'])
        pre_key = (prefix.game_id, prefix.boundary_id, prefix.prefix_digest)
        require(request_id not in seen_ids and pre_key not in seen_pres, 'duplicate query')
        deadline = payload['deadline_monotonic']
        require(type(deadline) in (float, int) and time.monotonic() < deadline <= time.monotonic()+limits.query_s)
        seen_ids.add(request_id);seen_pres.add(pre_key)
        thread = Thread(target=compute, args=(prefix,), daemon=True, name='capsule-only-inference')
        active = {'thread': thread, 'token': token, 'request_id': request_id,
                  'deadline': min(deadline, time.monotonic()+limits.service_s),
                  'timed_out': False, 'cancelled': False}
        thread.start()
        send('STARTED', token, request_id, {'state': 'RUNNING', 'session': session})
