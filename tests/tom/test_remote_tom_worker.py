"""Model-free worker adapter tests using official PRE reconstruction/planning."""
from copy import deepcopy
from dataclasses import asdict, fields
from io import StringIO
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts import remote_tom_protocol as protocol, remote_tom_worker as worker
from scripts import speech_planning as planning, counterfactual_tom as cf
from tests.tom.test_remote_tom_provider import IDENTITY, opportunity, provider
from tests.tom.test_speech_planning import RecordingConsumer
from scripts.remote_tom_provider import build_request
from werewolf.tom.final_capacity import derived_capacity


def fixture(phase='discussion'):
    planner = SimpleNamespace(CANDIDATE_ORDER_VERSION=planning.CANDIDATE_ORDER_VERSION,
        generate_candidates=Mock(wraps=planning.generate_candidates),
        evaluate_candidates=Mock(wraps=planning.evaluate_candidates),
        select_minimum_suspicion=Mock(wraps=planning.select_minimum_suspicion))
    consumer = RecordingConsumer()
    instance = worker.Worker(IDENTITY, consumer, planner, cf.derive_public_phase_speaker_order,
                             cf.QUEUE_RULE_VERSION, derived_capacity(96))
    request = build_request(opportunity(phase), alive_wolves=('player1',), worker_identity=IDENTITY)
    return instance, request


def rebind(request):
    return protocol.bind_request({k: v for k, v in request.items() if k not in ('request_id', 'request_digest')})


@pytest.mark.parametrize('phase', ['discussion', 'pk_discussion'])
def test_valid_request_uses_official_reconstruction_and_existing_planner(phase):
    instance, request = fixture(phase)
    response = instance.handle(request)
    instance.planner.generate_candidates.assert_called_once()
    instance.planner.evaluate_candidates.assert_called_once()
    instance.planner.select_minimum_suspicion.assert_called_once()
    parent = instance.planner.evaluate_candidates.call_args.args[0]
    assert parent.to_record() == request['parent_pre']
    assert len(instance.consumer.calls) == len(request['candidates'])
    index = protocol.validate_response(response, request)
    assert response['selected_plan'] == request['candidates'][index]
    assert set(response) == {'protocol_version', 'request_id', 'request_digest', 'status',
        'selected_candidate_index', 'selected_plan', 'worker_identity'}


@pytest.mark.parametrize('case', ['digest', 'pre', 'identity', 'adapter', 'condition',
    'missing_candidate', 'extra_candidate', 'reordered', 'duplicate_wolf', 'invalid_wolf',
    'unsorted_wolves', 'queue_version', 'candidate_version', 'boundary', 'protocol'])
def test_invalid_request_zero_evaluation_and_forward(case):
    instance, request = fixture()
    request = deepcopy(request)
    if case == 'digest': request['request_digest'] = '0'*64
    elif case == 'pre': request['parent_pre']['public_event_digest'] = '0'*64
    elif case in ('identity', 'adapter'):
        request['expected_worker_identity']['adapter_digest' if case == 'adapter' else 'runtime_digest'] = '0'*64
    elif case == 'condition': request['condition'] = 'explicit_day_phase'
    elif case == 'missing_candidate': request['candidates'].pop()
    elif case == 'extra_candidate': request['candidates'].append(request['candidates'][0])
    elif case == 'reordered': request['candidates'].reverse()
    elif case == 'duplicate_wolf': request['alive_wolves'] = ['player1', 'player1']
    elif case == 'invalid_wolf': request['alive_wolves'] = ['player1', 'player8']
    elif case == 'unsorted_wolves': request['alive_wolves'] = ['player2', 'player1']
    elif case == 'queue_version': request['public_queue_rule_version'] = 'wrong'
    elif case == 'candidate_version': request['candidate_order_version'] = 'wrong'
    elif case == 'boundary': request['boundary_id'] = 'wrong'
    elif case == 'protocol': request['protocol_version'] = 'wrong'
    if case != 'digest': request = rebind(request)
    with pytest.raises((ValueError, RuntimeError)):
        instance.handle(request)
    instance.planner.evaluate_candidates.assert_not_called()
    instance.planner.select_minimum_suspicion.assert_not_called()
    assert instance.consumer.calls == []
    if case in ('missing_candidate', 'extra_candidate', 'reordered'):
        instance.planner.generate_candidates.assert_called_once()


def test_loop_errors_continue_and_stdout_is_protocol_only():
    instance, request = fixture()
    original = instance.planner.evaluate_candidates
    def noisy(*args, **kwargs):
        print('planner debug output')
        return original(*args, **kwargs)
    instance.planner.evaluate_candidates = noisy
    stdin = StringIO('{"status":1,"status":2}\nnot json\n' + protocol.encode_json_line(request))
    stdout, stderr = StringIO(), StringIO()
    worker.serve(instance, stdin, stdout, stderr)
    lines = stdout.getvalue().splitlines(keepends=True)
    responses = [protocol.decode_json_line(line) for line in lines]
    assert [r['status'] for r in responses] == ['error', 'error', 'ok']
    assert all('request_id' not in r for r in responses[:2])
    assert 'planner debug output' not in stdout.getvalue()
    assert 'planner debug output' in stderr.getvalue()


def test_in_memory_jsonl_provider_round_trip():
    instance, _ = fixture()
    class Transport:
        def request(self, line):
            stdout = StringIO()
            worker.serve(instance, StringIO(line), stdout, StringIO())
            return stdout.getvalue()
    op = opportunity()
    selected = provider(Transport())(op)
    assert any(selected is candidate for candidate in op.candidates)


def test_final_identity_and_adapter_digest(tmp_path):
    assert {f.name for f in fields(protocol.WorkerIdentity)} == {
        'planner_baseline_commit', 'adapter_digest', 'runtime_digest', 'source_digest',
        'experiment_digest', 'seal_digest', 'checkpoint_digest', 'condition'}
    values = asdict(IDENTITY)
    del values['adapter_digest']
    with pytest.raises(TypeError): protocol.WorkerIdentity(**values)
    for name in worker.ADAPTER_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'adapter fixture')
    first = worker.adapter_digest(tmp_path)
    assert first == worker.adapter_digest(tmp_path)
    (tmp_path / worker.ADAPTER_FILES[0]).write_bytes(b'changed')
    assert first != worker.adapter_digest(tmp_path)


def test_bootstrap_failure_exits_without_serving(monkeypatch):
    monkeypatch.setattr(worker.sys, 'argv', ['worker', '--experiment', '/fixture', '--condition', 'implicit'])
    monkeypatch.setattr(worker, 'bootstrap', Mock(side_effect=ValueError('seal failed')))
    serve = Mock()
    monkeypatch.setattr(worker, 'serve', serve)
    assert worker.main() == 1
    serve.assert_not_called()
