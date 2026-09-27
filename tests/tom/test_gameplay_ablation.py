"""The only direct gameplay treatment is one wolf day-cognition request."""

from contextlib import nullcontext
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

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


def test_predictor_client_rejects_identity_mismatch_without_retry(monkeypatch, tmp_path):
    child = Mock()
    child.stdin, child.stdout = BytesIO(), BytesIO()
    child.poll.return_value = None
    monkeypatch.setattr(process.subprocess, "Popen", Mock(return_value=child))
    monkeypatch.setattr(process.Qwen3GameplayPredictorClient, "_read", Mock(
        side_effect=[{"status": "ready", "source_revision": process.SOURCE_REVISION,
                      "fit_digest": process.FIT_DIGEST,
                      "seal_digest": process.SEAL_DIGEST},
                     {"request_id": 0, "prefix_digest": "wrong",
                      "fit_digest": process.FIT_DIGEST,
                      "seal_digest": process.SEAL_DIGEST,
                      "probabilities": _matrix()}]))
    client = process.Qwen3GameplayPredictorClient(
        checkout=tmp_path, fit_path=tmp_path / "fit")
    prefix = SimpleNamespace(prefix_digest="a" * 64, to_record=lambda: {"public": True})
    with pytest.raises(ValueError, match="response identity mismatch"):
        client.predict(prefix)
    assert process.Qwen3GameplayPredictorClient._read.call_args_list == [
        call(timeout_seconds=None), call(timeout_seconds=600)]
    child.terminate.assert_called_once()
    assert process.subprocess.Popen.call_count == 1


@pytest.mark.parametrize("timeout_seconds,readable,message", (
    (None, True, "terminated without a response"),
    (600, False, "response timed out"),
))
def test_predictor_read_fails_closed_on_eof_or_prediction_timeout(
        monkeypatch, timeout_seconds, readable, message):
    stream = BytesIO()
    client = object.__new__(process.Qwen3GameplayPredictorClient)
    client._process = SimpleNamespace(stdout=stream)
    select_call = Mock(return_value=([stream] if readable else [], [], []))
    monkeypatch.setattr(process.select, "select", select_call)
    with pytest.raises(RuntimeError, match=message):
        client._read(timeout_seconds=timeout_seconds)
    select_call.assert_called_once_with([stream], [], [], timeout_seconds)


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
