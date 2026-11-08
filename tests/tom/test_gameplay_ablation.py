"""The only direct gameplay treatment is one wolf day-cognition request."""

from contextlib import contextmanager, nullcontext
from copy import deepcopy
from io import BytesIO
import json
import os
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from run_random import eval as run_game
from scripts import qwen3_gameplay_predictor as process
from scripts.qwen3_gameplay_predictor import _decode, _wire
from scripts.tom_gameplay_ablation import eligible_observers, run_ablation, tom_payload
from tests.agents.test_agent_backend import (
    MetadataBackend, _belief_for, _day_cognition, _observation,
    _pre_speech_belief, _role_observation,
)
from werewolf.agents.gpt_agent import GPTAgent
from werewolf.agents.prompt_template_v0 import (
    build_day_cognition_prompt, build_public_claim_catalog,
    derive_belief_constraints, freeze_discussion_candidates,
)
from werewolf.helper.log_utils import Log


TOM_BLOCK_HEADER = (
    "PRIVATE ToM INFORMATION\n"
    "Each row gives the model-predicted distribution of the named observer's\n"
    "suspected_werewolves self-report over target_ids.\n"
    "These values are not ground-truth role probabilities.\n"
)


def _matrix():
    return [[0.0 if observer == target else (target + 1) /
             sum(seat + 1 for seat in range(7) if seat != observer)
             for target in range(7)] for observer in range(7)]


def _wolf_observation(phase="speech", alive=None):
    team = Log(viewer=[1], source=0, target=[1, 2],
               content={"wolf_team": [1, 2]}, day=1,
               time="第1天白天", event="werewolf_team_info")
    observation = _role_observation("Werewolf", 1, logs=(team,),
                                    phase=f"1_day_{phase}")
    if phase == "speech_pk":
        observation["game_log"].append(Log(
            viewer=list(range(1, 8)), source=0, target=0,
            content={"vote_outcome": "draw", "speech_queue": [1, 3, 5]},
            day=1, time="第1天白天", event="end_vote"))
    observation["authoritative_public_state"]["alive_players"] = (
        list(range(1, 8)) if alive is None else alive)
    return observation


class _OneActionEnv:
    def __init__(self, observation):
        self.observation = observation
        self.public_events = []
        self.phase = observation["phase"].rsplit("_", 1)[-1]
        if observation["phase"].endswith("speech_pk"):
            self.phase = "speech_pk"
        elif observation["phase"].endswith("vote_pk"):
            self.phase = "vote_pk"
        self.actions = []

    def reset(self, *, roles):
        return deepcopy(self.observation)

    def step(self, action):
        self.actions.append(action)
        return self.observation, None, True, {"Werewolf": -1}


class _Recorder:
    def __init__(self, observation, *, alive=None):
        self.prefix = SimpleNamespace(
            game_id="game-1", boundary_id="game-1-pre-1", prefix_digest="a" * 64,
            current_speaker="player1",
            alive_observer_ids=tuple(f"player{seat}" for seat in (
                range(1, 8) if alive is None else alive)),
        )
        self.handoff = _pre_speech_belief(observation)
        self.handoff = self.handoff.__class__(
            **{**self.handoff.__dict__, "boundary_id": self.prefix.boundary_id})
        self._pending = SimpleNamespace(prefix=self.prefix)

    def start(self, *_args, **_kwargs):
        pass

    def before_agent_act(self, *_args, speech_kind, **_kwargs):
        return self.handoff if speech_kind is not None else None

    def after_agent_act(self, action):
        self.action = action

    def after_env_step(self, *_args, **_kwargs):
        pass

    def finish(self, *_args, **_kwargs):
        pass

    def failure_from_exception(self, error):
        return error


class _Audit:
    def action_context(self, **_kwargs):
        return nullcontext()

    def speech_perception_context(self, **_kwargs):
        return nullcontext()


def _runtime(observation, agent, *, alive=None):
    agents = [agent] * 7
    return _OneActionEnv(observation), agents, _Recorder(observation, alive=alive), _Audit()


def _agent(backend):
    return GPTAgent(backend=backend, model_name="fixture-model")


def test_notom_prompt_matches_frozen_prechange_bytes():
    observation = _observation()
    arguments = dict(candidate_snapshot=freeze_discussion_candidates(observation),
                     claim_catalog=build_public_claim_catalog(observation),
                     pre_speech_belief=_pre_speech_belief(observation))
    expected = (Path(__file__).parent / "fixtures" /
                "no_tom_day_cognition_prompt.txt").read_bytes()
    assert build_day_cognition_prompt(observation, **arguments).encode("utf-8") == expected
    assert build_day_cognition_prompt(
        observation, **arguments, tom_context=None).encode("utf-8") == expected


@pytest.mark.parametrize("phase", ("speech", "speech_pk"))
def test_notom_uses_exact_original_speech_request(phase):
    observation = _wolf_observation(phase)
    results = []
    for ablation in (False, True):
        backend = MetadataBackend([_day_cognition(observation)])
        env, agents, recorder, audit = _runtime(observation, _agent(backend))
        if ablation:
            run_ablation(env, agents, (), recorder=recorder, call_audit=audit,
                         arm="Wolf-NoToM")
        else:
            run_game(env, agents, (), canonical_recorder=recorder, call_audit=audit)
        results.append((deepcopy(backend.calls), env.actions))
    assert results[0] == results[1]
    assert len(results[0][0]) == 2


@pytest.mark.parametrize("phase", ("speech", "speech_pk"))
def test_tom_changes_only_cognition_request(phase):
    observation = _wolf_observation(phase)
    baseline = MetadataBackend([_day_cognition(observation)])
    env, agents, recorder, audit = _runtime(observation, _agent(baseline))
    run_ablation(env, agents, (), recorder=recorder, call_audit=audit,
                 arm="Wolf-NoToM")

    treated = MetadataBackend([_day_cognition(observation)])
    env, agents, recorder, audit = _runtime(observation, _agent(treated))
    predictor = SimpleNamespace(predict=Mock(return_value=_matrix()),
                                fit_digest="fit", seal_digest="seal")
    treatment_audit = []
    run_ablation(env, agents, (), recorder=recorder, call_audit=audit,
                 arm="Wolf+ToM", predictor=predictor,
                 treatment_audit=treatment_audit)
    payload = tom_payload(recorder.prefix, observation, _matrix())
    context = _wire(payload).decode().rstrip("\n")
    baseline_prompt = baseline.calls[0]["messages"][0]["content"]
    treated_prompt = treated.calls[0]["messages"][0]["content"]
    assert treated_prompt == baseline_prompt.replace(
        "\n\nDISCUSSION ACTION SEMANTICS",
        "\n\n" + TOM_BLOCK_HEADER + context + "\n\nDISCUSSION ACTION SEMANTICS",
        1)
    assert treated_prompt.count(TOM_BLOCK_HEADER) == 1
    assert (treated_prompt.index("PRE-SPEECH PRIVATE BELIEF") <
            treated_prompt.index(TOM_BLOCK_HEADER) <
            treated_prompt.index("DISCUSSION ACTION SEMANTICS"))
    assert treated_prompt.endswith(
        "Return only the JSON object required by the response schema. The three fields are\n"
        "public_content_selection, public_vote_stance_index and evidence_selection.")
    assert context not in treated_prompt.split("DISCUSSION ACTION SEMANTICS", 1)[1]
    assert {k: v for k, v in treated.calls[0].items() if k != "messages"} == {
        k: v for k, v in baseline.calls[0].items() if k != "messages"}
    assert treated.calls[1] == baseline.calls[1]
    assert TOM_BLOCK_HEADER not in treated.calls[1]["messages"][0]["content"]
    assert context not in treated.calls[1]["messages"][0]["content"]
    predictor.predict.assert_called_once_with(recorder.prefix)
    assert len(treatment_audit) == 1
    assert set(treatment_audit[0]) == {"game_id", "boundary_id", "prefix_digest",
                                      "observer_ids", "payload_digest",
                                      "fit_digest", "seal_digest"}


def test_cognition_retry_reuses_one_prediction():
    observation = _wolf_observation()
    backend = MetadataBackend(["invalid cognition", _day_cognition(observation)])
    env, agents, recorder, audit = _runtime(observation, _agent(backend))
    predictor = SimpleNamespace(predict=Mock(return_value=_matrix()),
                                fit_digest="fit", seal_digest="seal")
    run_ablation(env, agents, (), recorder=recorder, call_audit=audit,
                 arm="Wolf+ToM", predictor=predictor, treatment_audit=[])
    predictor.predict.assert_called_once()
    assert backend.calls[0]["messages"] == backend.calls[1]["messages"]
    assert all(call["messages"][0]["content"].count(TOM_BLOCK_HEADER) == 1
               for call in backend.calls[:2])
    assert len(backend.calls) == 3


def test_tom_is_not_retained_in_vote_pk_vote_or_night_requests():
    speech = _wolf_observation()
    vote = _wolf_observation("vote")
    vote_pk = _wolf_observation("vote_pk")
    night = _wolf_observation("skill_wolf")
    night["phase"] = "1_night_skill_wolf"
    night["valid_action"] = [("kill", 3)]
    vote_roles = derive_belief_constraints(vote)[1]
    pk_roles = derive_belief_constraints(vote_pk)[1]
    backend = MetadataBackend([
        _day_cognition(speech), _belief_for(vote_roles), '{"target":0}',
        _belief_for(pk_roles), '{"target":0}', '{"action_index":0}',
    ])
    agent = _agent(backend)
    context = '{"target_ids":["player1"],"rows":[{"observer_id":"player3","probabilities":[0.125]}]}'
    agent.act_with_pre_speech_belief(
        speech, pre_speech_belief=_pre_speech_belief(speech),
        tom_context=context)
    assert agent.act(vote)[0] == "vote"
    assert agent.act(vote_pk)[0] == "vote_pk"
    assert agent.act(night) == ("kill", 3)
    assert context in backend.calls[0]["messages"][0]["content"]
    assert TOM_BLOCK_HEADER in backend.calls[0]["messages"][0]["content"]
    assert all(context not in call["messages"][0]["content"] and
               TOM_BLOCK_HEADER not in call["messages"][0]["content"]
               for call in backend.calls[1:])


@pytest.mark.parametrize("identity,alive", (("Villager", list(range(1, 8))),
                                             ("Werewolf", [2, 3, 4, 5, 6, 7])))
def test_nonwolf_and_dead_wolf_never_request_prediction(identity, alive):
    observation = (_wolf_observation(alive=alive) if identity == "Werewolf"
                   else _role_observation("Villager", 1))
    class Agent:
        def reset(self):
            pass
        def act_with_pre_speech_belief(self, observation, *, pre_speech_belief):
            return "speech", "unchanged"
    env, agents, recorder, audit = _runtime(observation, Agent(), alive=alive)
    predictor = SimpleNamespace(predict=Mock(side_effect=AssertionError("forbidden")),
                                fit_digest="fit", seal_digest="seal")
    run_ablation(env, agents, (), recorder=recorder, call_audit=audit,
                 arm="Wolf+ToM", predictor=predictor, treatment_audit=[])
    predictor.predict.assert_not_called()


@pytest.mark.parametrize("phase", ("vote", "vote_pk", "skill_wolf"))
def test_vote_and_night_do_not_request_prediction_or_context(phase):
    observation = _wolf_observation(phase)
    class Agent:
        def reset(self):
            pass
        def act(self, observation):
            return "unchanged", 0
    env, agents, recorder, audit = _runtime(observation, Agent())
    predictor = SimpleNamespace(predict=Mock(side_effect=AssertionError("forbidden")),
                                fit_digest="fit", seal_digest="seal")
    run_ablation(env, agents, (), recorder=recorder, call_audit=audit,
                 arm="Wolf+ToM", predictor=predictor, treatment_audit=[])
    predictor.predict.assert_not_called()
    assert env.actions == [("unchanged", 0)]


def test_legal_rows_and_all_unsorted_target_columns():
    observation = _wolf_observation(alive=[1, 2, 3, 5, 7])
    prefix = _Recorder(observation, alive=[1, 2, 3, 5, 7]).prefix
    assert eligible_observers(prefix, observation) == ("player3", "player5", "player7")
    matrix = _matrix()
    payload = tom_payload(prefix, observation, matrix)
    assert list(payload) == ["target_ids", "rows"]
    assert payload["target_ids"] == [f"player{seat}" for seat in range(1, 8)]
    assert [row["observer_id"] for row in payload["rows"]] == [
        "player3", "player5", "player7"]
    assert [row["probabilities"] for row in payload["rows"]] == [
        matrix[2], matrix[4], matrix[6]]
    assert len(payload["rows"][0]["probabilities"]) == 7
    assert json.loads(_wire(payload))["rows"][0]["probabilities"] == matrix[2]


def test_tom_payload_rejects_simplex_error_over_one_part_per_million():
    observation = _wolf_observation()
    matrix = _matrix()
    matrix[2][0] += 2e-6
    with pytest.raises(ValueError, match="invalid distribution"):
        tom_payload(_Recorder(observation).prefix, observation, matrix)


def test_prediction_failure_stops_before_agent_action_or_env_step():
    observation = _wolf_observation()
    class Agent:
        def reset(self):
            pass
        def act_with_pre_speech_belief(self, *_args, **_kwargs):
            raise AssertionError("agent must not act")
    env, agents, recorder, audit = _runtime(observation, Agent())
    predictor = SimpleNamespace(predict=Mock(side_effect=RuntimeError("worker failed")),
                                fit_digest="fit", seal_digest="seal")
    with pytest.raises(RuntimeError, match="worker failed"):
        run_ablation(env, agents, (), recorder=recorder, call_audit=audit,
                     arm="Wolf+ToM", predictor=predictor, treatment_audit=[])
    predictor.predict.assert_called_once()
    assert env.actions == []


@pytest.mark.parametrize("field,value", (("boundary_id", "wrong-boundary"),
                                          ("prefix_digest", "b" * 64),
                                          ("observer_id", "player2")))
def test_tom_boundary_identity_mismatch_fails_before_prediction(field, value):
    observation = _wolf_observation()
    env, agents, recorder, audit = _runtime(observation, Mock())
    recorder.handoff = recorder.handoff.__class__(
        **{**recorder.handoff.__dict__, field: value})
    predictor = SimpleNamespace(predict=Mock(side_effect=AssertionError("forbidden")),
                                fit_digest="fit", seal_digest="seal")
    with pytest.raises(ValueError, match="PRE/handoff mismatch"):
        run_ablation(env, agents, (), recorder=recorder, call_audit=audit,
                     arm="Wolf+ToM", predictor=predictor, treatment_audit=[])
    predictor.predict.assert_not_called()


def test_predictor_wire_rejects_noncanonical_and_duplicate_keys():
    assert _decode(_wire({"request_id": 0})) == {"request_id": 0}
    with pytest.raises(ValueError, match="noncanonical"):
        _decode(b'{"request_id": 0}\n')
    with pytest.raises(ValueError, match="duplicate"):
        _decode(b'{"request_id":0,"request_id":1}\n')


@contextmanager
def _predictor_watchdog():
    """仅为旧实现的阻塞回归提供外层熔断，不能替代客户端期限。"""
    previous = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    def expired(*args):
        raise AssertionError('client exceeded the test watchdog')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, 2)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.signal(signal.SIGALRM, previous)


@pytest.fixture
def predictor_worker(monkeypatch, tmp_path):
    """用真实本地管道运行小型 CPU 协议替身，不加载模型。"""
    popen = process.subprocess.Popen
    children = []
    ready = {'status': 'ready', 'source_revision': process.SOURCE_REVISION,
             'fit_digest': process.FIT_DIGEST, 'seal_digest': process.SEAL_DIGEST}
    header = ("import json, os, sys, time\n"
        "def wire(value):\n"
        " return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()+b'\\n'\n"
        "def send(value):\n"
        " sys.stdout.buffer.write(wire(value)); sys.stdout.buffer.flush()\n"
        f"def ready():\n send({ready!r})\n"
        "def reply(request):\n"
        " return {'request_id':request['request_id'],'prefix_digest':'"+'a'*64+"',"
        f"'fit_digest':{process.FIT_DIGEST!r},'seal_digest':{process.SEAL_DIGEST!r},"
        f"'probabilities':{_matrix()!r}}}\n")
    monkeypatch.setattr(process, '_STARTUP_TIMEOUT_SECONDS', .25, raising=False)
    monkeypatch.setattr(process, '_STOP_TIMEOUT_SECONDS', .25, raising=False)
    monkeypatch.setattr(process, '_TIMEOUT_SECONDS', 1)

    def start(body):
        def launch(command, **kwargs):
            child = popen([sys.executable, '-u', '-c', header+body],
                cwd=kwargs['cwd'], stdin=process.subprocess.PIPE,
                stdout=process.subprocess.PIPE, bufsize=kwargs.get('bufsize', -1))
            children.append(child)
            return child
        monkeypatch.setattr(process.subprocess, 'Popen', launch)
        return process.Qwen3GameplayPredictorClient(checkout=tmp_path, fit_path=tmp_path/'fit')

    try:
        with _predictor_watchdog():
            yield start, children
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
            child.wait(timeout=2)
            child.stdin.close()
            child.stdout.close()


def _predictor_prefix(record=None):
    return SimpleNamespace(prefix_digest='a'*64,
        to_record=lambda: {'public': True} if record is None else record)


def test_predictor_normal_ready_requests_and_idempotent_close(predictor_worker):
    start, children = predictor_worker
    client = start('ready()\nfor line in sys.stdin.buffer:\n send(reply(json.loads(line)))\n')
    assert client.predict(_predictor_prefix()) == _matrix()
    assert client.predict(_predictor_prefix()) == _matrix()
    assert client._next_request == 2 and len(children) == 1
    client.close()
    client.close()
    assert children[0].poll() is not None
    assert children[0].stdin.closed and children[0].stdout.closed
    with pytest.raises(RuntimeError, match='closed'):
        client.predict(_predictor_prefix())


@pytest.mark.parametrize('output', ('none', 'partial', 'eof', 'wrong'))
def test_predictor_startup_failure_is_bounded_and_reports_state(predictor_worker, output):
    start, children = predictor_worker
    bodies = {'none': 'time.sleep(5)\n',
        'partial': "sys.stdout.buffer.write(b'{'); sys.stdout.buffer.flush(); time.sleep(5)\n",
        'eof': 'sys.exit(3)\n', 'wrong': "send({'status':'wrong'}); time.sleep(5)\n"}
    with pytest.raises(RuntimeError, match='startup.*pid=.*returncode='):
        start(bodies[output])
    assert len(children) == 1 and children[0].poll() is not None
    assert children[0].stdin.closed and children[0].stdout.closed


@pytest.mark.parametrize('failure', ('partial', 'eof', 'invalid', 'identity', 'trickle'))
def test_predictor_response_failure_never_resends(predictor_worker, monkeypatch, failure):
    start, children = predictor_worker
    bodies = {
        'partial': "sys.stdout.buffer.write(b'{'); sys.stdout.buffer.flush(); time.sleep(5)\n",
        'eof': 'sys.exit(0)\n',
        'invalid': "sys.stdout.buffer.write(b'not-json\\n'); sys.stdout.buffer.flush(); time.sleep(5)\n",
        'identity': "response=reply(request); response['prefix_digest']='wrong'; send(response); time.sleep(5)\n",
        'trickle': "for value in wire(reply(request)):\n os.write(1,bytes([value])); time.sleep(.04)\n",
    }
    client = start('ready()\nrequest=json.loads(sys.stdin.buffer.readline())\n'+bodies[failure])
    monkeypatch.setattr(process, '_TIMEOUT_SECONDS', .15)
    written = []
    original = os.write
    fd = children[0].stdin.fileno()
    def write(descriptor, data):
        count = original(descriptor, data)
        if descriptor == fd:
            written.append(bytes(data[:count]))
        return count
    monkeypatch.setattr(process.os, 'write', write)
    with pytest.raises(ValueError if failure in ('identity', 'invalid') else RuntimeError):
        client.predict(_predictor_prefix())
    assert client._next_request == 1 and len(children) == 1
    # 分块写不是重发：成功写出的字节只能组成一个请求。
    assert b''.join(written).count(b'\n') == 1
    assert children[0].stdin.closed and children[0].stdout.closed
    client.close()
    with pytest.raises(RuntimeError, match='closed'):
        client.predict(_predictor_prefix())


def test_predictor_write_backpressure_obeys_request_deadline(predictor_worker, monkeypatch):
    start, children = predictor_worker
    client = start('ready()\ntime.sleep(5)\n')
    monkeypatch.setattr(process, '_TIMEOUT_SECONDS', .15)
    with pytest.raises(RuntimeError, match='request write.*timed out'):
        client.predict(_predictor_prefix({'public': 'x'*(1024*1024)}))
    assert client._next_request == 1 and len(children) == 1
    assert children[0].stdin.closed and children[0].stdout.closed


def test_predictor_partial_writes_and_transient_readiness(predictor_worker, monkeypatch):
    start, _ = predictor_worker
    client = start('ready()\nrequest=json.loads(sys.stdin.buffer.readline())\nsend(reply(request))\n')
    original = os.write
    chunks = []
    def write(fd, data):
        if not chunks:
            chunks.append(b'')
            raise BlockingIOError()
        count = original(fd, data[:17])
        chunks.append(bytes(data[:count]))
        return count
    monkeypatch.setattr(process.os, 'write', write)
    assert client.predict(_predictor_prefix()) == _matrix()
    assert len(chunks) > 2
    assert _decode(b''.join(chunks))['request_id'] == 0
    client.close()


def test_predictor_write_and_read_share_one_budget(predictor_worker, monkeypatch):
    start, _ = predictor_worker
    client = start('ready()\nrequest=json.loads(sys.stdin.buffer.readline())\ntime.sleep(.1)\nsend(reply(request))\n')
    original = os.write
    def write(fd, data):
        count = original(fd, data)
        time.sleep(.1)
        return count
    # 分别模拟写入与读取开销；累计超时不能被阶段切换重置。
    original_read = process.os.read
    def read(fd, count):
        result = original_read(fd, count)
        time.sleep(.1)
        return result
    monkeypatch.setattr(process.os, 'write', write)
    monkeypatch.setattr(process.os, 'read', read)
    monkeypatch.setattr(process, '_TIMEOUT_SECONDS', .15)
    with pytest.raises(RuntimeError, match='timed out'):
        client.predict(_predictor_prefix())


def test_predictor_identity_validation_is_inside_deadline(predictor_worker, monkeypatch):
    start, _ = predictor_worker
    client = start('ready()\nrequest=json.loads(sys.stdin.buffer.readline())\nsend(reply(request))\n')
    original = process._decode
    def decode(line):
        value = original(line)
        time.sleep(.1)
        return value
    monkeypatch.setattr(process, '_decode', decode)
    monkeypatch.setattr(process, '_TIMEOUT_SECONDS', .05)
    with pytest.raises(RuntimeError, match='timed out'):
        client.predict(_predictor_prefix())


def test_predictor_closed_input_pipe_fails_without_retry(predictor_worker):
    start, children = predictor_worker
    client = start('sys.stdin.close()\nready()\ntime.sleep(5)\n')
    with pytest.raises((OSError, RuntimeError)):
        client.predict(_predictor_prefix())
    assert client._next_request == 1 and len(children) == 1
    assert children[0].stdin.closed and children[0].stdout.closed


def test_predictor_stop_timeout_remains_unconfirmed_and_closes_descriptors(monkeypatch):
    child = Mock(pid=123, stdin=BytesIO(), stdout=BytesIO())
    child.poll.return_value = None
    child.wait.side_effect = process.subprocess.TimeoutExpired('fixture', .1)
    client = object.__new__(process.Qwen3GameplayPredictorClient)
    client._process, client._closed, client._terminate_sent = child, False, False
    monkeypatch.setattr(process, '_STOP_TIMEOUT_SECONDS', .1, raising=False)
    for _ in range(2):
        with pytest.raises(RuntimeError, match='stop.*unconfirmed'):
            client.close()
    child.terminate.assert_called_once()
    child.kill.assert_not_called()
    assert child.stdin.closed and child.stdout.closed
    assert all(kwargs['timeout'] <= .1 for args, kwargs in child.wait.call_args_list)
    child.poll.return_value = 0
    client.close()


def test_predictor_interrupt_during_startup_releases_owned_child(predictor_worker, monkeypatch):
    start, children = predictor_worker
    monkeypatch.setattr(process.Qwen3GameplayPredictorClient, '_read',
        Mock(side_effect=KeyboardInterrupt('controlled startup interruption')))
    with pytest.raises(KeyboardInterrupt):
        start('ready()\ntime.sleep(5)\n')
    assert children[0].poll() is not None
    assert children[0].stdin.closed and children[0].stdout.closed


def test_predictor_launch_failure_reports_startup_and_no_child(monkeypatch, tmp_path):
    calls = []
    def launch(*args, **kwargs):
        calls.append(True)
        raise OSError('launch refused')
    monkeypatch.setattr(process.subprocess, 'Popen', launch)
    with pytest.raises(RuntimeError, match='startup.*launch refused.*pid=None.*returncode=None'):
        process.Qwen3GameplayPredictorClient(checkout=tmp_path, fit_path=tmp_path/'fit')
    assert calls == [True]


def test_predictor_close_releases_stdout_even_if_stdin_close_fails():
    child = Mock(pid=123, stdout=BytesIO())
    child.poll.return_value = 0
    child.stdin.close.side_effect = OSError('controlled close failure')
    client = object.__new__(process.Qwen3GameplayPredictorClient)
    client._process, client._closed, client._terminate_sent = child, False, False
    with pytest.raises(OSError, match='controlled close failure'):
        client.close()
    assert child.stdout.closed and client._closed
    child.terminate.assert_not_called()


def test_predictor_failed_cleanup_preserves_request_failure_and_unconfirmed_state(monkeypatch):
    child = Mock(pid=123, stdin=BytesIO(), stdout=BytesIO())
    child.poll.return_value = None
    child.wait.side_effect = process.subprocess.TimeoutExpired('fixture', .1)
    client = object.__new__(process.Qwen3GameplayPredictorClient)
    client._process, client._closed, client._terminate_sent = child, False, False
    client._next_request = 0
    client._write = Mock()
    client._read = Mock(return_value={'request_id': 999})
    with pytest.raises(RuntimeError, match='response identity mismatch.*cleanup failed:.*stop unconfirmed'):
        client.predict(_predictor_prefix())
    assert client._next_request == 1
    assert child.stdin.closed and child.stdout.closed
    child.kill.assert_not_called()


@pytest.mark.parametrize("head,status", (("wrong-revision", ""),
                                         (process.SOURCE_REVISION, "?? stray.py")))
def test_worker_rejects_wrong_or_dirty_checkout(monkeypatch, tmp_path, head, status):
    monkeypatch.chdir(tmp_path)
    def git(command, *, env, text, stderr):
        del env, text, stderr
        key = tuple(command[3:])
        return {("rev-parse", "--show-toplevel"): str(tmp_path),
                ("rev-parse", "HEAD"): head,
                ("status", "--porcelain=v1", "--untracked-files=all",
                 "--ignore-submodules=none"): status}[key]
    monkeypatch.setattr(process.subprocess, "check_output", git)
    with pytest.raises(RuntimeError, match="clean pinned source checkout"):
        process._clean_checkout(tmp_path)


def test_tom_requires_an_existing_predictor_before_game_starts():
    observation = _wolf_observation()
    agent = Mock()
    env, agents, recorder, audit = _runtime(observation, agent)
    with pytest.raises(ValueError, match="existing predictor"):
        run_ablation(env, agents, (), recorder=recorder, call_audit=audit,
                     arm="Wolf+ToM", treatment_audit=[])
    agent.reset.assert_not_called()
    assert env.actions == []


def test_notom_rejects_predictor_before_game_starts():
    observation = _wolf_observation()
    agent = Mock()
    env, agents, recorder, audit = _runtime(observation, agent)
    with pytest.raises(ValueError, match="NoToM must not receive"):
        run_ablation(env, agents, (), recorder=recorder, call_audit=audit,
                     arm="Wolf-NoToM", predictor=Mock())
    agent.reset.assert_not_called()
    assert env.actions == []
