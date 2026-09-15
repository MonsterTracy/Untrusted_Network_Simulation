"""Protocol fixtures only; no subprocess worker/model is launched."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
import subprocess
import sys
from unittest.mock import Mock

import pytest

from scripts import remote_tom_provider as r
from scripts.constrained_speech import public_eligibility, SpeechTreatment
from scripts.speech_planning import PlanningAction as A
from tests.tom.test_counterfactual_consumer import PublicWitness, P
from tests.tom.test_constrained_speech import runner_fixture
from run_random import eval as run_game
from tests.canonical_collection.test_runtime_game_evidence import ROLES
from werewolf.artifact_io import canonical_json_bytes, sha256_bytes


IDENTITY = r.WorkerIdentity('8886af9505036b55662d54ae9a108b09121e8c71',
    'a'*64, '1'*64, '2'*64, '3'*64, '4'*64, '5'*64, 'implicit')  # Fixture adapter/runtime digests only.


def opportunity(phase='discussion'):
    return public_eligibility(PublicWitness(phase).pre()).opportunity


def success(request):
    index = next(i for i, p in enumerate(request['candidates']) if p['action'] == 'NO_COMMITMENT')
    return dict(protocol_version=request['protocol_version'], request_id=request['request_id'],
        request_digest=request['request_digest'], status='ok', selected_candidate_index=index,
        selected_plan=request['candidates'][index], worker_identity=request['expected_worker_identity'])


class FakeTransport:
    def __init__(self, mutate=lambda response: None):
        self.calls = []
        self.mutate = mutate
    def request(self, line):
        request = r.decode_json_line(line)
        self.calls.append(request)
        response = deepcopy(success(request))
        self.mutate(response)
        return r.encode_json_line(response)


def provider(transport):
    return r.RemoteToMPlanProvider(transport=transport, worker_identity=IDENTITY,
        alive_wolves_for=lambda op: (op.speaker,))


@pytest.mark.parametrize('phase', ['discussion', 'pk_discussion'])
def test_request_deterministic_complete_pre_and_canonical_return(phase):
    op = opportunity(phase)
    request = r.build_request(op, alive_wolves=(P[0],), worker_identity=IDENTITY)
    assert request == r.build_request(op, alive_wolves=(P[0],), worker_identity=IDENTITY)
    assert request['parent_pre'] == op.parent.to_record()
    assert request['parent_digest'] == op.parent.prefix_digest
    assert request['candidates'] == [p.public_payload() for p in op.candidates]
    wire = r.decode_json_line(r.encode_json_line(request))
    digest = wire.pop('request_digest')
    assert digest == sha256_bytes(canonical_json_bytes(wire))
    transport = FakeTransport()
    selected = provider(transport)(op)
    assert selected is op.candidates[-1] and selected.action == A.NO_COMMITMENT
    assert len(transport.calls) == 1
    assert set(selected.public_payload()) == {'action', 'target'}
    changed = r.build_request(op, alive_wolves=(P[0], P[1]), worker_identity=IDENTITY)
    assert changed['request_digest'] != request['request_digest']
    reversed_candidates = r.build_request(replace(op, candidates=op.candidates[::-1]),
        alive_wolves=(P[0],), worker_identity=IDENTITY)
    assert reversed_candidates['request_digest'] != request['request_digest']
    # A wire mutation cannot preserve the verified binding.
    for field in ('parent_pre', 'parent_digest', 'candidate_order_version', 'public_queue_rule_version'):
        changed = deepcopy(wire)
        changed[field] = 'tampered'
        assert sha256_bytes(canonical_json_bytes(changed)) != digest


@pytest.mark.parametrize('field', ['protocol_version', 'request_id', 'request_digest'])
def test_response_binding_mismatch(field):
    transport = FakeTransport(lambda response: response.__setitem__(field, 'wrong'))
    with pytest.raises(r.RemoteToMError):
        provider(transport)(opportunity())
    assert len(transport.calls) == 1


@pytest.mark.parametrize('field', list(asdict(IDENTITY)))
def test_worker_identity_mismatch(field):
    transport = FakeTransport(lambda response: response['worker_identity'].__setitem__(field, 'wrong'))
    with pytest.raises(r.RemoteToMError):
        provider(transport)(opportunity())


@pytest.mark.parametrize('case', ['negative', 'large', 'bool', 'unknown_plan', 'index_plan',
    'missing', 'score', 'matrix', 'worker_error'])
def test_invalid_response_fail_closed(case):
    def mutate(response):
        if case in ('negative', 'large', 'bool'):
            response['selected_candidate_index'] = {'negative': -1, 'large': 999, 'bool': True}[case]
        elif case == 'unknown_plan':
            response['selected_plan'] = {'action': 'vote_intent', 'target': 'player2'}
        elif case == 'index_plan':
            response['selected_candidate_index'] = 0
        elif case == 'missing':
            del response['selected_plan']
        elif case in ('score', 'matrix'):
            response[case] = []
        else:
            response['status'] = 'error'
            del response['selected_candidate_index'], response['selected_plan'], response['request_digest']
            response.update(error_code='INFERENCE_FAILED', error_message='private worker details')
    transport = FakeTransport(mutate)
    with pytest.raises(r.RemoteToMError) as exc:
        provider(transport)(opportunity())
    assert 'private worker details' not in str(exc.value)
    assert len(transport.calls) == 1


@pytest.mark.parametrize('line', ['not json\n', '{}', '{}\n{}\n', '[]\n',
    '{"status":"ok","status":"error"}\n', '{"nested":{"x":1,"x":2}}\n', '{"x":NaN}\n'])
def test_malformed_json_or_duplicate_keys(line):
    transport = Mock()
    transport.request.return_value = line
    with pytest.raises(r.RemoteToMError):
        provider(transport)(opportunity())
    transport.request.assert_called_once()


@pytest.mark.parametrize('error', [TimeoutError(), BrokenPipeError(), ConnectionError(), EOFError()])
def test_transport_failures_do_not_retry(error):
    transport = Mock()
    transport.request.side_effect = error
    with pytest.raises(r.RemoteToMError):
        provider(transport)(opportunity())
    transport.request.assert_called_once()


@pytest.mark.parametrize('value', [None, 'unknown', 'latest', ''])
def test_incomplete_worker_config_rejected(value):
    with pytest.raises(ValueError):
        replace(IDENTITY, runtime_digest=value)


def test_remote_failure_through_runner_has_no_fallback_or_private_trace():
    from werewolf.canonical_collection.collector import CanonicalAttemptFailure
    env, agents, audit, recorder, calls = runner_fixture()
    transport = Mock()
    transport.request.side_effect = TimeoutError()
    trace = []
    with pytest.raises(CanonicalAttemptFailure):
        run_game(env, agents, ROLES, canonical_recorder=recorder, call_audit=audit,
            planning_mode='constrained', plan_provider=provider(transport),
            speech_treatment=SpeechTreatment(players=P, roles=()), ablation_trace=trace)
    transport.request.assert_called_once()
    assert not any(kind == 'original_speech' for kind, _ in calls)
    assert trace[-1].status == 'CONSTRAINED_EXECUTION_FAILED'
    assert trace[-1].public_eligible is True
    assert not {'request_id', 'request_digest', 'alive_wolves'} & asdict(trace[-1]).keys()
    assert not any(e['event_type'] == 'public_speech' for e in env.public_events)


@pytest.mark.parametrize('mode', ['original', 'unassigned', 'last'])
def test_untreated_does_not_call_remote_transport(mode):
    env, agents, audit, recorder, calls = runner_fixture()
    transport = FakeTransport()
    remote = provider(transport)
    trace = []
    options = dict(planning_mode='constrained', plan_provider=remote, ablation_trace=trace,
                   speech_treatment=SpeechTreatment(roles=()))
    if mode == 'original':
        options['planning_mode'] = 'original'
    elif mode == 'last':
        # Public gate runs before provider even if all players are assigned.
        from scripts.constrained_speech import handle_speech
        from tests.tom.test_verified_speech_commit import prepare, pending
        env, recorder, backend = prepare()
        from tests.tom.test_speech_realization import ParserBackend
        while env.speech_queue:
            env.speech_perceiver.backend = ParserBackend(
                f'player{env.current_act_idx+1} | no_commitment | NONE')
            env.step(('speech', '我暂时不作明确表态。'))
        pending(env, recorder)
        assert handle_speech(env=env, actor=None, recorder=recorder, provider=remote,
                             assigned=True, trace=trace) is None
        assert trace[-1].status == 'INELIGIBLE_FOR_CONSTRAINED_TREATMENT'
        assert transport.calls == []
        return
    run_game(env, agents, ROLES, canonical_recorder=recorder, call_audit=audit, **options)
    assert transport.calls == []


def test_remote_module_has_no_inference_imports():
    subprocess.run([sys.executable, '-c',
        "import sys; import scripts.remote_tom_provider; "
        "assert 'scripts.counterfactual_tom' not in sys.modules; "
        "assert 'scripts.speech_planning' not in sys.modules; "
        "assert not any(k.startswith('werewolf.tom') for k in sys.modules)"], check=True)


def test_remote_success_does_not_expose_private_request_to_actor_or_trace():
    env, agents, audit, recorder, _ = runner_fixture()
    transport, trace = FakeTransport(), []
    run_game(env, agents, ROLES, canonical_recorder=recorder, call_audit=audit,
        planning_mode='constrained', plan_provider=provider(transport),
        speech_treatment=SpeechTreatment(players=P, roles=()), ablation_trace=trace)
    assert transport.calls and all('alive_wolves' in request for request in transport.calls)
    assert all(not {'request_id', 'request_digest', 'alive_wolves'} & asdict(row).keys() for row in trace)
    assert all(row.selected_plan is None or set(row.selected_plan.public_payload()) == {'action', 'target'} for row in trace)
    # Requests to the remote transport never enter local Actor/perception audit.
    assert 'alive_wolves' not in json.dumps([call.private_payload.to_value() for call in audit.records])
    assert 'alive_wolves' not in json.dumps(env.public_events)
    public_surfaces = repr((trace, [call.private_payload.to_value() for call in audit.records],
                            env.public_events, [vars(log) for log in env.game_log]))
    for request in transport.calls:
        assert request['request_id'] not in public_surfaces
        assert request['request_digest'] not in public_surfaces


@pytest.mark.parametrize('field', ['planner_baseline_commit', 'adapter_digest'])
def test_baseline_and_adapter_identities_are_required_and_bound(field):
    op = opportunity()
    request = r.build_request(op, alive_wolves=(P[0],), worker_identity=IDENTITY)
    changed = replace(IDENTITY, **{field: 'b'*(40 if field == 'planner_baseline_commit' else 64)})
    other = r.build_request(op, alive_wolves=(P[0],), worker_identity=changed)
    assert request['expected_worker_identity'][field] == getattr(IDENTITY, field)
    assert request['request_id'] != other['request_id']
    assert request['request_digest'] != other['request_digest']
    with pytest.raises(ValueError):
        replace(IDENTITY, **{field: None})


@pytest.mark.parametrize('phase', ['discussion', 'pk_discussion'])
def test_supplier_wolf_order_does_not_change_canonical_request(phase):
    op = opportunity(phase)
    transports = [FakeTransport(), FakeTransport()]
    for transport, wolves in zip(transports, ((P[2], P[0]), (P[0], P[2]))):
        remote = r.RemoteToMPlanProvider(transport=transport, worker_identity=IDENTITY,
            alive_wolves_for=lambda opportunity, wolves=wolves: wolves)
        remote(op)
    assert transports[0].calls == transports[1].calls
    assert transports[0].calls[0]['alive_wolves'] == [P[0], P[2]]


@pytest.mark.parametrize('wolves', [(P[0], P[0]), (P[0], 'player8'), (P[0], 'player0')])
def test_invalid_or_duplicate_wolf_ids_fail_before_transport(wolves):
    transport = FakeTransport()
    remote = r.RemoteToMPlanProvider(transport=transport, worker_identity=IDENTITY,
        alive_wolves_for=lambda opportunity: wolves)
    with pytest.raises(ValueError):
        remote(opportunity())
    assert transport.calls == []
