from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256
from inspect import getsourcelines
import json
import random
import re
import signal
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pytest

from werewolf import game_runtime
from werewolf.backends import OpenAICompatibleBackend
from werewolf.backends.openai_compatible import _request_context
from werewolf.game_runtime import build_runtime, eval as run_game, verify_gameplay_replay


ROLES = [
    "Werewolf",
    "Villager",
    "Seer",
    "Witch",
    "Villager",
    "Villager",
    "Werewolf",
]


class RecordingBackend:
    def __init__(self, name):
        self.name = name
        self.calls = []

    def chat(
        self,
        messages,
        model=None,
        temperature=0.7,
        max_tokens=None,
        response_format=None,
        **kwargs,
    ):
        self.calls.append(
            {
                "messages": messages,
                "model": model,
            }
        )
        return "[]"


def runtime_config():
    return {
        "backends": {
            "parser-api": {
                "type": "openai_compatible",
            },
            "agent-api-a": {
                "type": "openai_compatible",
            },
            "agent-api-b": {
                "type": "openai_compatible",
            },
        },
        "parser": {
            "backend": "parser-api",
            "model": "parser-model",
        },
        "env_config": {
            "n_player": 7,
            "n_role": 4,
            "n_werewolf": 2,
            "n_seer": 1,
            "n_guard": 0,
            "n_witch": 1,
            "n_hunter": 0,
            "n_villager": 3,
        },
        "agent_config": {
            "must_include": ["profile-b"],
            "all_candidates": [
                {
                    "profile_name": "profile-a",
                    "agent_type": "gpt",
                    "backend": "agent-api-a",
                    "model": "agent-model-a",
                    "model_params": {"temperature": 0.2},
                    "sample_ratio": 1.0,
                },
                {
                    "profile_name": "profile-b",
                    "agent_type": "gpt",
                    "backend": "agent-api-b",
                    "model": "agent-model-b",
                    "model_params": {"temperature": 0.7},
                    "sample_ratio": 1.0,
                },
            ],
        },
    }


class RunRandomProfileTest(unittest.TestCase):
    def setUp(self):
        self.parser_backend = RecordingBackend("parser")
        self.agent_backend_a = RecordingBackend("agent-a")
        self.agent_backend_b = RecordingBackend("agent-b")
        self.backends = {
            "parser-api": self.parser_backend,
            "agent-api-a": self.agent_backend_a,
            "agent-api-b": self.agent_backend_b,
        }

    def test_root_source_module_exports_the_same_runtime(self):
        """保留的根Source位置只导出同一实现，不拥有第二套执行逻辑。"""
        import run_random
        from werewolf import game_runtime

        for name in ('eval', 'build_runtime', 'assign_agents', 'PUBLIC_SPEECH_RUNTIME_PHASES'):
            self.assertIs(getattr(run_random, name), getattr(game_runtime, name))
        self.assertEqual(game_runtime.eval.__module__, 'werewolf.game_runtime')
        self.assertFalse(hasattr(run_random, 'main_cli'))

    def test_internal_callers_load_package_runtime_without_root_module(self):
        """生产采集与Phase-2直接使用正式Runtime，不依赖根文件反向装配。"""
        code = '''
import sys
from werewolf import game_runtime
from werewolf.canonical_collection import production_runtime
from werewolf.interventions.online import runner as phase2_online_runner
assert production_runtime.build_runtime is game_runtime.build_runtime
assert production_runtime.run_game is game_runtime.eval
assert phase2_online_runner.run_canonical_game is game_runtime.eval
assert 'run_random' not in sys.modules
'''
        subprocess.run([sys.executable, '-B', '-c', code], check=True)

    def test_source_inventories_cover_the_runtime_implementation(self):
        """显式Source清单必须绑定迁移后的实现，不能只记录根转发文件。"""
        from pathlib import Path
        from werewolf import game_runtime
        from werewolf.artifact_io import sha256_bytes
        from werewolf.observer_dtrs.shadow_worker import _implementation
        from werewolf.interventions.online.preflight import SOURCE_FILES

        name = 'werewolf/game_runtime.py'
        self.assertIn(name, SOURCE_FILES)
        self.assertEqual(_implementation()[name], sha256_bytes(Path(game_runtime.__file__).read_bytes()))

    def test_seed_preserves_roles_profiles_and_next_random_draw(self):
        """固定迁移前观察到的角色、Profile和后续RNG状态，防止改变随机调用顺序。"""
        expected = (
            (0, ['Villager', 'Villager', 'Werewolf', 'Werewolf', 'Seer', 'Villager', 'Witch'],
             ['profile-b', 'profile-b', 'profile-a', 'profile-a', 'profile-b', 'profile-b', 'profile-b'],
             2577854692418029171),
            (3, ['Werewolf', 'Villager', 'Villager', 'Witch', 'Seer', 'Villager', 'Werewolf'],
             ['profile-b', 'profile-a', 'profile-a', 'profile-a', 'profile-a', 'profile-a', 'profile-b'],
             9979206796010795124),
            (17, ['Seer', 'Werewolf', 'Werewolf', 'Witch', 'Villager', 'Villager', 'Villager'],
             ['profile-b', 'profile-a', 'profile-a', 'profile-b', 'profile-b', 'profile-b', 'profile-b'],
             15858755842042771698),
        )
        state = random.getstate()
        try:
            for seed, expected_roles, expected_profiles, expected_draw in expected:
                with self.subTest(seed=seed):
                    _, _, roles, profiles = build_runtime(
                        runtime_config(), None, random_seed=seed, backends=self.backends)
                    self.assertEqual(roles, expected_roles)
                    self.assertEqual(profiles, expected_profiles)
                    self.assertEqual(random.getrandbits(64), expected_draw)
        finally:
            random.setstate(state)

    def test_random_ins_gets_the_game_seed_without_changing_assignment(self):
        config = runtime_config()
        state = random.getstate()
        try:
            _, _, expected_roles, expected_profiles = build_runtime(
                config, None, random_seed=3, backends=self.backends,
            )
            expected_rng_state = random.getstate()
            config["agent_config"]["all_candidates"][1]["model_params"]["decision_method"] = "dpins:random"
            _, agents, roles, profiles = build_runtime(
                config, None, random_seed=3, backends=self.backends,
            )

            self.assertEqual(roles, expected_roles)
            self.assertEqual(profiles, expected_profiles)
            self.assertEqual(random.getstate(), expected_rng_state)
            for agent, profile in zip(agents, profiles):
                if profile == "profile-b":
                    self.assertEqual((agent.decision_method, agent.game_seed), ("dpins:random", 3))
                    self.assertIs(agent.backend, self.agent_backend_b)
                else:
                    self.assertIsNone(agent.decision_method)
            with self.assertRaisesRegex(ValueError, "requires an integer game seed"):
                build_runtime(config, None, backends=self.backends)
        finally:
            random.setstate(state)

    def test_new_schema_wires_parser_and_profile_backends(self):
        env, agents, roles, profile_names = build_runtime(
            runtime_config(),
            log_save_path=None,
            roles=ROLES,
            random_seed=3,
            backends=self.backends,
        )

        self.assertEqual(roles, ROLES)
        self.assertIs(
            env.speech_perceiver.backend,
            self.parser_backend,
        )
        self.assertEqual(
            env.speech_perceiver.model_name,
            "parser-model",
        )
        self.assertEqual(profile_names[0], "profile-b")
        self.assertIn("profile-a", profile_names)
        for agent, profile_name in zip(agents, profile_names):
            if profile_name == "profile-a":
                self.assertIs(agent.backend, self.agent_backend_a)
                self.assertEqual(agent.model_name, "agent-model-a")
                self.assertEqual(agent.temperature, 0.2)
            else:
                self.assertIs(agent.backend, self.agent_backend_b)
                self.assertEqual(agent.model_name, "agent-model-b")
                self.assertEqual(agent.temperature, 0.7)

    @patch("werewolf.game_runtime.load_named_backends")
    def test_default_runtime_path_loads_named_backends(
        self,
        load_named_backends,
    ):
        load_named_backends.return_value = self.backends

        build_runtime(
            runtime_config(),
            log_save_path=None,
            roles=ROLES,
            random_seed=3,
        )

        load_named_backends.assert_called_once()
        normalized = load_named_backends.call_args.args[0]
        self.assertIn("backends", normalized)
        self.assertEqual(
            normalized["parser"]["backend"],
            "parser-api",
        )

    def test_empty_eligible_profiles_raise_value_error(self):
        config = runtime_config()
        config["agent_config"]["must_include"] = []
        config["agent_config"]["all_candidates"] = [
            config["agent_config"]["all_candidates"][0]
        ]

        with self.assertRaises(ValueError) as raised:
            build_runtime(
                config,
                log_save_path=None,
                roles=ROLES,
                random_seed=3,
                backends=self.backends,
            )

        self.assertIn("eligible", str(raised.exception))


class _GameplayBackend:
    supports_json_schema = True

    def __init__(self, winner):
        self.winner = winner
        self.requests = []
        self.pre_supports = []
        self.fail_request = None

    def chat(self, *, messages, model, response_format=None, **_kwargs):
        prompt = messages[-1]["content"]
        kind = response_format["json_schema"]["name"] if response_format else "speech_parser"
        self.requests.append((model, kind, prompt))
        if kind == self.fail_request:
            return "invalid output"
        if kind == "speech_parser":
            return "NONE"
        assert kind == "private_belief_report"
        required = prompt.split("MUST INCLUDE: ", 1)[1].splitlines()[0]
        candidates = response_format["json_schema"]["schema"]["properties"]
        legal = candidates["suspected_werewolves"]["items"]["enum"]
        support = legal[:1] if required == "<none>" else required.split(", ")
        self.pre_supports.append(support)
        return json.dumps({"suspected_werewolves": support})

    def chat_with_metadata(self, *, messages, model, response_format=None, **_kwargs):
        prompt = messages[-1]["content"]
        if response_format is None:
            self.requests.append((model, "speech_realization", prompt))
            return "我暂不表态。", {"finish_reason": "stop"}
        schema = response_format["json_schema"]
        kind = schema["name"]
        self.requests.append((model, kind, prompt))
        if kind == self.fail_request:
            return "invalid output", {"finish_reason": "stop"}
        properties = schema["schema"]["properties"]
        if kind == "night_action_selection":
            # 狼人选择首个有效刀口，预言家选择首个查验对象，女巫保留药。
            response = {"action_index": 1 if "0: {'杀害':'否'}" in prompt else 0}
        elif kind == "belief_report":
            response = {
                "belief": "本次根据当前信息判断。", "concise": "继续观察。",
                "roles": {player: "unknown" for player in properties["roles"]["properties"]},
            }
        elif kind == "day_cognition_report_v4":
            response = {
                "public_content_selection": {"mode": "none"},
                "public_vote_stance_index": 0,
                "evidence_selection": {"mode": "none"},
            }
        elif kind == "discussion_tactic":
            response = {"strategy": "honest_evidence"}
        elif kind == "direct_gameplay_speech":
            response = {"thought": "当前合法的公开信息有限。", "speech": "我暂不表态。"}
        else:
            assert kind == "vote"
            response = {"target": self._vote(prompt, properties["target"]["enum"])}
        return json.dumps(response, ensure_ascii=False), {"finish_reason": "stop"}

    def _vote(self, prompt, legal):
        actor = int(re.search(r"Current player: player([1-7])", prompt).group(1))
        phase = re.search(r"【当前阶段】第(\d+)天白天(平票)?投票", prompt)
        day = int(phase.group(1))
        if phase.group(2):
            return 0
        if day == 1:
            # 六名存活玩家以 3:3 将两名狼人送入 PK，随后弃票进入下一夜。
            target = {1: 7, 3: 1, 4: 1, 5: 7, 6: 7, 7: 1}[actor]
        elif self.winner == "Werewolf":
            target = 4
        else:
            target = 1 if day == 2 else 7
        return target if target in legal else 0


class _GameplayClient:
    """在同一个脚本化对局前复用真实 Backend 的 SDK 调用边界。"""

    def __init__(self, script):
        self.script = script
        self.chat = SimpleNamespace(completions=self)
        self.options = []
        self.calls = []
        self.request_arguments = []
        self.error_kind = None
        self.error = None
        self.repair_kind = None

    def with_options(self, **options):
        self.options.append(options)
        return self

    def create(self, *, messages, model, response_format=None, **kwargs):
        self.request_arguments.append(deepcopy(dict(
            messages=messages, model=model, response_format=response_format, **kwargs,
        )))
        kind = response_format["json_schema"]["name"] if response_format else (
            "speech_parser" if model == "parser-model" else "speech_realization"
        )
        self.calls.append((kind, kwargs.get("timeout")))
        if kind == self.error_kind:
            raise self.error
        usage = {"prompt_tokens": 11, "completion_tokens": 2, "total_tokens": 13}
        metadata = {"finish_reason": "stop"}
        if kind == self.repair_kind:
            self.repair_kind = None
            content = "invalid output"
            usage = None
        elif kind in {"private_belief_report", "speech_parser"}:
            content = self.script.chat(messages=messages, model=model, response_format=response_format)
        else:
            content, metadata = self.script.chat_with_metadata(
                messages=messages, model=model, response_format=response_format,
            )
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=content), finish_reason=metadata["finish_reason"],
        )], usage=usage)


def _build_gameplay_fixture(method, *, winner="Werewolf", use_openai_backend=False):
    config = runtime_config()
    config["agent_config"]["all_candidates"][0]["model_params"]["decision_method"] = method
    seats = ["profile-a" if role == "Werewolf" else "profile-b" for role in ROLES]
    backend = _GameplayBackend(winner)
    serving_backend = backend
    if use_openai_backend:
        serving_backend = OpenAICompatibleBackend(client=_GameplayClient(backend), supports_json_schema=True)
    runtime = build_runtime(
        config, None, roles=ROLES, random_seed=17, seat_profiles=seats,
        backends={name: serving_backend for name in config["backends"]},
    )
    return backend, runtime


@pytest.mark.parametrize("method", ["react", "dpins:no", "dpins:llm", "dpins:random"])
def test_gameplay_only_native_terminal_and_replay(method):
    backend, (env, agents, roles, profiles) = _build_gameplay_fixture(method, use_openai_backend=True)
    assert not backend.requests

    result = run_game(env, agents, roles, gameplay_only=True, max_steps=80,
                      record_requests=True, request_timeout=7.5)

    assert result.failure_type is None, result
    assert result.terminal_reached and env.phase == "end_game" and env.day == 2
    assert result.winner == result.replay_winner == "Werewolf"
    assert env.wolf_win_count == 1
    assert result.seed == 17 and result.roles == tuple(env.roles) == tuple(ROLES)
    assert result.seat_profiles == tuple(profiles)
    assert result.decision_methods == (method, None, None, None, None, None, method)
    assert result.seat_profiles == ("profile-a",) + ("profile-b",) * 5 + ("profile-a",)
    assert re.fullmatch(r"[0-9a-f]{40}", result.source_identity["commit"])
    assert re.fullmatch(r"[0-9a-f]{64}", result.source_identity["tracked_source_diff_sha256"])
    assert {action for action, _ in result.submitted_actions} == {
        "kill", "check", "witch_pass", "speech", "speech_pk", "vote", "vote_pk",
    }
    assert len(result.submitted_actions) == 38
    assert result.speech_parser_attempts == sum(kind == "speech_parser" for _, kind, _ in backend.requests) == 13
    assert result.completed_steps == len(result.submitted_actions)
    assert result.elapsed_seconds > 0 and result.request_timeout_seconds == 7.5
    assert re.fullmatch(r"[0-9a-f]{64}", result.source_identity["file_sha256"]["werewolf/game_runtime.py"])
    assert result.model_requests == len(backend.requests) == len(result.request_records)
    assert result.retries == 0
    assert result.token_usage == {"input_tokens": 11 * result.model_requests,
                                 "output_tokens": 2 * result.model_requests,
                                 "total_tokens": 13 * result.model_requests}
    client = agents[0].backend.client
    assert client.options == [{"max_retries": 0}] * result.model_requests
    assert all(timeout == 7.5 for _, timeout in client.calls)
    for row in result.request_records:
        assert row["game_seed"] == 17 and 1 <= row["seat"] <= 7
        assert row["method"] == (result.decision_methods[row["seat"] - 1] or "default")
        assert row["status"] == "response_received" and row["sdk_retries"] == 0
    traced_speeches = {row["step"]: row for row in result.request_records if row["kind"] == "speech_realization"}
    for speech, annotation in zip(result.speech_results, env.speech_annotations, strict=True):
        row = traced_speeches[speech["step"]]
        assert speech["kind"] == result.submitted_actions[speech["step"]][0]
        assert speech["seat"] == row["seat"]
        assert speech["event_id"] == annotation.event_id and speech["v1_status"] == "no_action"
        assert speech["strategy"] == row["strategy"]
        if row["method"] == "dpins:llm":
            assert speech["strategy"] == "honest_evidence"
        elif row["method"] == "dpins:random":
            assert speech["strategy"] in {
                "honest_evidence", "deceptive_evidence", "honest_accusation",
                "deceptive_accusation", "honest_defense", "deceptive_defense",
            }
        else:
            assert speech["strategy"] is None
    requests = list(backend.requests)
    assert verify_gameplay_replay(result) == "Werewolf"
    assert backend.requests == requests

    pre_count = sum(kind == "private_belief_report" for _, kind, _ in requests)
    background_speeches = [prompt for model, kind, prompt in requests if model == "agent-model-b" and kind == "day_cognition_report_v4"]
    assert pre_count == len(background_speeches) == len(backend.pre_supports) == 7
    for prompt, support in zip(background_speeches, backend.pre_supports, strict=True):
        pre_payload = prompt.split("PRE-SPEECH PRIVATE BELIEF (IMMUTABLE INPUT)\n", 1)[1].splitlines()[0]
        assert json.loads(pre_payload)["suspected_werewolves"] == support
    wolf_calls = [kind for model, kind, _ in requests if model == "agent-model-a"]
    assert "private_belief_report" not in wolf_calls and "day_cognition_report_v4" not in wolf_calls
    assert wolf_calls.count("direct_gameplay_speech") == 6
    assert wolf_calls.count("belief_report") == (0 if method == "react" else 12)
    assert wolf_calls.count("discussion_tactic") == (6 if method == "dpins:llm" else 0)


def test_direct_tom_only_appends_one_private_block_to_react_requests():
    from scripts.tom_gameplay_ablation import direct_tom_coverage, direct_tom_payload
    from tests.tom.test_gameplay_ablation import _matrix
    from werewolf.agents.prompt_template_v0 import DIRECT_SPEECH_TOM_PREFIX

    prefixes = []

    def predict(prefix, *, record_forward):
        prefixes.append(prefix)
        record_forward(0.125)
        return _matrix()

    def treatment(prefix, observation, evidence):
        return direct_tom_payload(prefix, observation, predictor=SimpleNamespace(predict=predict), evidence=evidence)

    def play(**options):
        _, (env, agents, roles, _) = _build_gameplay_fixture("react", use_openai_backend=True)
        client = agents[0].backend.client
        client.repair_kind = "direct_gameplay_speech"
        result = run_game(env, agents, roles, gameplay_only=True, max_steps=80,
                          record_requests=True, request_timeout=7.5, **options)
        assert result.failure_type is None and result.winner == result.replay_winner == "Werewolf"
        return result, client

    legacy, old_client = play()
    control, control_client = play(direct_tom_game_id="control")
    treated, client = play(direct_tom_game_id="direct", direct_wolf_speech_tom=treatment)
    assert old_client.request_arguments == control_client.request_arguments
    assert legacy.request_records[0].keys() == control.request_records[0].keys()
    assert legacy.submitted_actions == control.submitted_actions == treated.submitted_actions
    assert legacy.public_event_digest == control.public_event_digest == treated.public_event_digest
    assert not any("direct_tom" in speech for speech in legacy.speech_results)

    by_boundary = {prefix.boundary_id: prefix for prefix in prefixes}
    assert len(prefixes) == len(by_boundary) == 6
    sent = []
    for actual, original, record in zip(client.request_arguments, control_client.request_arguments,
                                        treated.request_records, strict=True):
        if "direct_tom" not in record:
            assert DIRECT_SPEECH_TOM_PREFIX not in actual["messages"][0]["content"]
            assert actual == original
            continue
        assert record["kind"] == "speech_realization" and record["method"] == "react" and record["seat"] in (1, 7)
        assert actual["response_format"]["json_schema"]["name"] == "direct_gameplay_speech"
        message = actual["messages"][0]["content"]
        first, rest = message.split(DIRECT_SPEECH_TOM_PREFIX, 1)
        payload, end = json.JSONDecoder().raw_decode(rest)
        wire = rest[:end]
        actual["messages"][0]["content"] = first + rest[end:]
        assert actual == original
        receipt = record["direct_tom"]
        assert receipt["payload_digest"] == sha256(wire.encode()).hexdigest()
        prefix = by_boundary[receipt["boundary_id"]]
        assert receipt["prefix_digest"] == prefix.prefix_digest
        assert payload["target_ids"] == [f"player{seat}" for seat in range(1, 8)]
        assert [row["observer_id"] for row in payload["rows"]] == [
            seat for seat in prefix.alive_observer_ids if seat not in ("player1", "player7")
        ]
        assert all(row["probabilities"] == _matrix()[int(row["observer_id"][-1]) - 1] for row in payload["rows"])
        sent.append(record)

    assert len(sent) == 7 and sent[0]["direct_tom"] == sent[1]["direct_tom"]
    assert sent[0]["validation_error_type"] == "GameplaySpeechQualityError"
    assert {speech["kind"] for speech in treated.speech_results if "direct_tom" in speech} == {"speech", "speech_pk"}
    stats = direct_tom_coverage([asdict(treated)], applicable=True)
    assert [stats[field] for field in ("E", "A", "P", "U", "S", "speech_committed")] == [6] * 6
    assert stats["by_phase"]["speech_pk"]["S"] == 2 and stats["unmatched_receipts"] == 0
    assert stats["forward_seconds"]["observed_sum"] == 0.75
    stats = direct_tom_coverage([asdict(control)], applicable=False)
    assert [stats[field] for field in ("E", "A", "P", "U", "S", "speech_committed")] == [6, 0, 0, 0, 0, 6]


@pytest.mark.parametrize(("fault", "stage", "prediction", "sent", "received", "submitted"), [
    ("prediction", "direct_tom_prediction", 0, 0, 0, False),
    ("interrupt", "direct_tom_prediction", "UNKNOWN", 0, 0, False),
    ("digest", "agent_action", 1, 0, 0, False),
    ("transport", "agent_action", 1, 1, "UNKNOWN", False),
    ("parser", "environment_step", 1, 1, 1, True),
])
def test_direct_tom_retains_partial_evidence_without_fallback(monkeypatch, fault, stage, prediction, sent, received, submitted):
    from scripts.tom_gameplay_ablation import direct_tom_coverage, direct_tom_payload
    from tests.tom.test_gameplay_ablation import _matrix
    from werewolf.backends import openai_compatible

    _, (env, agents, roles, _) = _build_gameplay_fixture("react", use_openai_backend=True)
    client = agents[0].backend.client

    def interrupt_digest(_payload):
        raise KeyboardInterrupt()

    def predict(prefix, *, record_forward):
        if fault == "prediction":
            raise ValueError("scripted predictor failure")
        if fault == "interrupt":
            raise KeyboardInterrupt()
        record_forward(0.1)
        if fault == "digest":
            monkeypatch.setattr(openai_compatible, "sha256", interrupt_digest)
        else:
            client.error_kind = "direct_gameplay_speech" if fault == "transport" else "speech_parser"
            client.error = TimeoutError("scripted transport failure")
        return _matrix()

    def treatment(prefix, observation, evidence):
        return direct_tom_payload(prefix, observation, predictor=SimpleNamespace(predict=predict), evidence=evidence)

    result = run_game(env, agents, roles, gameplay_only=True, record_requests=True,
                      direct_tom_game_id="partial", direct_wolf_speech_tom=treatment)
    assert result.failure_stage == stage and result.winner is None and not result.terminal_reached
    if fault in ("interrupt", "digest"):
        assert result.failure_type == "KeyboardInterrupt"
    detail, = (speech["direct_tom"] for speech in result.speech_results if "direct_tom" in speech)
    assert detail["speech_submitted"] is submitted and not detail["speech_committed"]
    assert sum(kind == "direct_gameplay_speech" for kind, _ in client.calls) == sent
    stats = direct_tom_coverage([asdict(result)], applicable=True)
    assert [stats[field] for field in ("E", "A", "P", "U", "S", "speech_committed")] == [1, 1, prediction, sent, received, 0]
    if fault == "transport":
        result.request_records[-1]["direct_tom"]["payload_digest"] = "f" * 64
        stats = direct_tom_coverage([asdict(result)], applicable=True)
        assert stats["unmatched_receipts"] == 1 and stats["U"] == stats["S"] == 0


class _SpeechLengthScorer:
    """受控终止评分依次偏好不同动作长度，仍运行真实 MCTS。"""

    def __init__(self, lengths=(0,)):
        import torch

        self.lengths = lengths
        self.histories = []
        self.score_counts = []
        self.uniform = torch.full((7, 7), 1 / 6).fill_diagonal_(0)
        self.preferred = []
        for column in range(7):
            matrix = torch.full((7, 7), 1 / 5).fill_diagonal_(0)
            matrix[:, column] = 0
            matrix[column] = self.uniform[column]
            self.preferred.append(matrix)

    def score_candidate(self, history, speaker, actions):
        if not self.histories or self.histories[-1] is not history:
            self.histories.append(history)
            self.score_counts.append(0)
        self.score_counts[-1] += 1
        target_length = self.lengths[(len(self.histories) - 1) % len(self.lengths)]
        return self.preferred[int(speaker[-1]) - 1] if len(actions) == target_length else self.uniform


def test_multimind_native_game_uses_real_mcts_and_v1_realizations(monkeypatch):
    from werewolf.canonical_collection.public_history import freeze_public_event_history
    from werewolf.strategy import multimind_planning
    from werewolf.tom.multimind import encode_history

    expected_lengths = (0, 1, 2, 3, 1, 0)
    scorer = _SpeechLengthScorer(expected_lengths)
    script, (env, agents, roles, profiles) = _build_gameplay_fixture("multimind", use_openai_backend=True)
    planner = multimind_planning.plan_speech
    planned, realized, speeches = [], [], {}
    assert all(agents[i].model_name == "agent-model-a" for i in (0, 6)) and agents[0] is not agents[6]
    assert all(agent.backend is agents[0].backend for agent in agents)

    def observe_plan(history, speaker, **kwargs):
        scope = _request_context.get()
        assert scope.method == "multimind" and speaker == f"player{scope.seat}"
        assert kwargs["reasoner"] is scorer and kwargs["known_wolves"] == ["player1", "player7"]
        assert kwargs["alive_players"] == tuple(f"player{i}" for i, alive in enumerate(env.alive, 1) if alive)
        assert kwargs.get("budget", 500) == 500
        public = freeze_public_event_history(env.public_events)
        assert history == encode_history(public, tuple(env.speech_annotations))
        for annotation in env.speech_annotations:
            raw = public.events[annotation.event_index].raw_text
            if raw in speeches:
                assert [a.to_record() for a in annotation.actions] == speeches[raw]
        assert script.requests[-1][1] == "belief_report"
        plan = planner(history, speaker, **kwargs)
        planned.append([action.to_record() for action in plan.actions])
        return plan

    monkeypatch.setattr(multimind_planning, "plan_speech", observe_plan)
    native_generation = script.chat_with_metadata
    native_parse = script.chat

    def generate(*, messages, model, response_format=None, **kwargs):
        output = native_generation(messages=messages, model=model, response_format=response_format, **kwargs)
        scope = _request_context.get()
        if scope.method != "multimind" or scope.kind != "speech_realization":
            return output
        prompt = messages[-1]["content"]
        assert "真实狼队信息（仅用于内部策略）：player1, player7" in prompt
        assert "CURRENT PRIVATE BELIEF" in prompt and "本次根据当前信息判断。" in prompt
        assert "Game rules:" in prompt and "player8" not in prompt
        assert len(planned) == len(realized) + 1
        actual = list(planned[-1])
        labels = {
            "point_as_werewolf": "指认为狼人", "point_as_villager": "指认为普通村民",
            "point_as_seer": "指认为预言家", "point_as_witch": "指认为女巫",
            "support": "公开支持", "oppose": "公开质疑",
        }
        if actual:
            ordered_lines = prompt.split("本轮必须完整表达、且只能表达以下冻结意图：\n", 1)[1].split("\n\n", 1)[0].splitlines()
            assert len(ordered_lines) == len(actual)
            for line, (_, predicate, target) in zip(ordered_lines, actual, strict=True):
                assert line.startswith(f"- {labels[predicate]}：") and target in line
        else:
            assert "本轮计划没有指定语义行动" in prompt and "冻结意图：" not in prompt

        index = len(realized)
        if index == 2:  # 合法省略整条意图，即使省掉目标也保留为遵从证据。
            actual = []
        elif index == 3:
            actual.reverse()
        elif index in (4, 5):
            actual.append([f"player{scope.seat}", "no_commitment", None])
        text_templates = {
            "point_as_werewolf": "{}是狼人", "point_as_villager": "{}就是普通村民",
            "point_as_seer": "{}是预言家", "point_as_witch": "{}是女巫",
            "support": "我支持{}", "oppose": "我质疑{}", "no_commitment": "本轮我明确暂不表态",
        }
        body = "。".join(text_templates[predicate].format(target) for _, predicate, target in actual)
        text = f"这一轮我说第{index + 1}句。" + (body + "。" if body else "声音都听得到。")
        realized.append(actual)
        speeches[text] = actual
        return text, output[1]

    def parse(*, messages, model, response_format=None, **kwargs):
        default = native_parse(messages=messages, model=model, response_format=response_format, **kwargs)
        prompt = messages[-1]["content"]
        raw = prompt.rsplit(": ", 1)[-1]
        if model != "parser-model" or raw not in speeches:
            return default
        actions = speeches[raw]
        lines = [f"{subject} | {predicate} | {target or 'NONE'}" for subject, predicate, target in actions]
        if len(actions) == 1:
            lines += lines  # 真实 V1 Parser 去重后才进入环境历史。
        return "\n".join(lines) or "NONE"

    monkeypatch.setattr(script, "chat_with_metadata", generate)
    monkeypatch.setattr(script, "chat", parse)
    result = run_game(env, agents, roles, gameplay_only=True, record_requests=True,
                      max_steps=80, request_timeout=7.5, multimind_reasoner=scorer)
    assert result.failure_type is None, result.failure_message
    assert result.terminal_reached and result.winner == result.replay_winner == "Werewolf"
    assert result.seat_profiles == tuple(profiles)
    assert result.decision_methods == ("multimind", None, None, None, None, None, "multimind")
    assert verify_gameplay_replay(result) == result.winner and env.day == 2 and result.completed_steps == 38
    assert result.model_requests == len(result.request_records) == len(script.requests)
    assert "werewolf/strategy/multimind_planning.py" in result.source_identity["file_sha256"]
    rows = [row for row in result.speech_results if "multimind" in row]
    assert len(rows) == len(scorer.histories) == 6 and scorer.score_counts == [500] * 6
    assert {row["kind"] for row in rows} == {"speech", "speech_pk"}
    assert {row["seat"] for row in rows} == {1, 7}
    assert any(row["multimind"]["last_same_phase_speaker"] for row in rows)
    assert tuple(map(len, planned)) == expected_lengths
    public = freeze_public_event_history(env.public_events)
    events = {event.event_id: event for event in public.events}
    for index, row in enumerate(rows):
        detail = row["multimind"]
        assert detail["planned_actions"] == planned[index] and detail["realized_actions"] == realized[index]
        assert detail["realized_speech"] == result.submitted_actions[row["step"]][1]
        event = events[row["event_id"]]
        assert event.raw_text == detail["realized_speech"]
        assert public.events[event.event_index - 1].event_id == detail["boundary_event_id"]
        next_event = public.events[event.event_index + 1]
        assert detail["last_same_phase_speaker"] == (next_event.temporal_state != event.temporal_state)
        assert detail["planner_attempted"] and detail["reasoner_prediction_succeeded"] and detail["planner_completed"]
        assert detail["plan_delivered_to_actor"] and detail["speech_submitted"] and detail["speech_committed"]
        assert detail["completed_simulations"] == detail["candidate_score_calls"] == 500
        assert detail["reasoner_forward_calls"] is None and detail["mcts_elapsed_seconds"] > 0
        assert detail["failure_stage"] is None and row["v1_status"] == ("ok" if realized[index] else "no_action")
        assert [r["kind"] for r in result.request_records if r["step"] == row["step"]] == [
            "belief", "speech_realization", "v1_parser",
        ]
    alignments = [row["multimind"]["alignment"] for row in rows]
    assert alignments[0] == dict(exact_match=True, plan_precision=None, plan_recall=None,
                                 order_match=None, stop_compliance=True)
    assert alignments[1]["exact_match"] and alignments[1]["plan_precision"] == 1
    assert not alignments[2]["exact_match"] and alignments[2]["plan_recall"] == 0
    assert not alignments[3]["order_match"] and alignments[3]["plan_recall"] == 1
    assert alignments[4]["plan_precision"] == 0.5 and alignments[4]["plan_recall"] == 1
    assert alignments[5]["stop_compliance"] is False and alignments[5]["plan_recall"] is None
    wolf_requests = [r for r in result.request_records if r["method"] == "multimind"]
    assert {r["kind"] for r in wolf_requests} == {"night_action", "belief", "speech_realization", "v1_parser", "vote"}
    assert {r["phase"].split("_day_")[-1] for r in wolf_requests if r["kind"] == "vote"} == {"vote", "vote_pk"}
    for request in wolf_requests:
        if request["kind"] == "vote":
            assert [r["kind"] for r in wolf_requests if r["step"] == request["step"]] == ["belief", "vote"]
    assert all("multimind" not in row for row in result.speech_results if row["seat"] not in (1, 7))


def test_multimind_alignment_treats_actual_duplicates_as_extra_and_keeps_matched_order():
    first = ["player1", "support", "player2"]
    second = ["player1", "oppose", "player3"]
    third = ["player1", "point_as_seer", "player4"]
    alignment = game_runtime._multimind_plan_alignment([first, second, third], [second, second, first])
    assert alignment == dict(exact_match=False, plan_precision=2 / 3, plan_recall=2 / 3,
                             order_match=False, stop_compliance=None)
    assert game_runtime._multimind_plan_alignment([first], None) is None


@pytest.mark.parametrize(("fault", "stage", "error_type", "actor_attempts", "parser_attempts"), [
    ("belief", "multimind_belief", "GameplayGenerationExhausted", 0, 0),
    ("planner", "multimind_planner", "ValueError", 0, 0),
    ("reasoner", "multimind_reasoner", "RuntimeError", 0, 0),
    ("capacity", "multimind_reasoner", "ValueError", 0, 0),
    ("actor_quality", "multimind_actor", "GameplayGenerationExhausted", 3, 0),
    ("actor_timeout", "multimind_actor", "BackendError", 1, 0),
    ("parser_quality", "environment_step", "V1SpeechPerceptionExhausted", 1, 3),
    ("parser_timeout", "environment_step", "BackendError", 1, 1),
])
def test_multimind_failures_keep_stage_and_partial_speech_without_fallback(
    monkeypatch, fault, stage, error_type, actor_attempts, parser_attempts,
):
    from werewolf.strategy import multimind_planning

    script, (env, agents, roles, _) = _build_gameplay_fixture("multimind", use_openai_backend=True)
    scorer = _SpeechLengthScorer()
    if fault == "planner":
        def fail_plan(*args, **kwargs):
            raise ValueError("scripted planner failure")
        monkeypatch.setattr(multimind_planning, "plan_speech", fail_plan)
    if fault in ("reasoner", "capacity"):
        from werewolf.tom.dataset import ExperimentCapacity
        from werewolf.tom.multimind import MultiMindReasoner

        capacity = ExperimentCapacity(1 if fault == "capacity" else 96)
        scorer = MultiMindReasoner(capacity, hidden_size=16, num_layers=1, num_heads=2).eval()
        if fault == "reasoner":
            forward = scorer.forward
            entered = 0

            def fail_second_forward(*args, **kwargs):
                nonlocal entered
                entered += 1
                if entered == 2:
                    raise RuntimeError("scripted Reasoner failure")
                return forward(*args, **kwargs)

            monkeypatch.setattr(scorer, "forward", fail_second_forward)
    client = agents[0].backend.client
    create = client.create

    def fail_selected_request(**kwargs):
        scope = _request_context.get()
        if scope.method == "multimind" and "speech" in scope.phase:
            if (fault, scope.kind) in {("belief", "belief"), ("parser_quality", "v1_parser")}:
                client.repair_kind = {"belief": "belief_report", "v1_parser": "speech_parser"}[scope.kind]
            if (fault, scope.kind) in {("actor_timeout", "speech_realization"), ("parser_timeout", "v1_parser")}:
                client.error_kind = "speech_realization" if scope.kind == "speech_realization" else "speech_parser"
                client.error = TimeoutError("scripted MultiMind timeout")
        return create(**kwargs)

    monkeypatch.setattr(client, "create", fail_selected_request)
    generation = script.chat_with_metadata

    def invalid_actor(**kwargs):
        content, metadata = generation(**kwargs)
        scope = _request_context.get()
        if fault == "actor_quality" and scope.method == "multimind" and scope.kind == "speech_realization":
            return "player8 不在合法座位内。", metadata
        return content, metadata

    # 文本无效与请求结果未知沿用现有 Backend 和 Agent 的重试边界。
    monkeypatch.setattr(script, "chat_with_metadata", invalid_actor)
    result = run_game(env, agents, roles, gameplay_only=True, record_requests=True,
                      request_timeout=7.5, max_steps=80, multimind_reasoner=scorer)
    assert result.failure_stage == stage and result.failure_type == error_type
    assert not result.terminal_reached and result.winner is None and result.replay_winner is None
    row = result.speech_results[-1]
    detail = row["multimind"]
    assert detail["failure_stage"] == stage and detail["failure_type"] == error_type
    assert detail["failure_message"] and row["step"] == result.completed_steps
    assert detail["realized_actions"] is None and detail["alignment"] is None and not detail["speech_committed"]
    assert detail["planner_attempted"] == (fault != "belief")
    assert detail["planner_completed"] == detail["plan_delivered_to_actor"] == (actor_attempts > 0)
    if fault in ("reasoner", "capacity"):
        actual_counts = (detail["completed_simulations"], detail["candidate_score_calls"],
                         detail["reasoner_forward_calls"], detail["reasoner_prediction_succeeded"])
        assert actual_counts == ((1, 2, 2, True) if fault == "reasoner" else (0, 1, 0, False))
        assert not scorer._forward_pre_hooks
    else:
        expected_scores = 500 if actor_attempts else {"belief": 0, "planner": None}[fault]
        assert detail["candidate_score_calls"] == expected_scores
    assert detail["speech_submitted"] == (parser_attempts > 0)
    if fault == "parser_quality":
        assert row["v1_status"] == "error" and detail["v1_error_message"]
    else:
        assert row["v1_status"] == "UNKNOWN"
    requests = [r for r in result.request_records if r["step"] == row["step"]]
    assert [r["kind"] for r in requests] == (
        ["belief"] * (3 if fault == "belief" else 1)
        + ["speech_realization"] * actor_attempts + ["v1_parser"] * parser_attempts
    )
    if fault.endswith("timeout"):
        assert requests[-1]["response_received"] == "UNKNOWN" and requests[-1]["sdk_retries"] == 0


def test_multimind_keeps_speech_evidence_when_request_recording_is_off():
    _, (env, agents, roles, _) = _build_gameplay_fixture("multimind")
    result = run_game(env, agents, roles, gameplay_only=True, max_steps=11,
                      multimind_reasoner=_SpeechLengthScorer())
    assert result.failure_stage == "step_limit" and result.request_records == ()
    assert result.speech_results
    for row in result.speech_results:
        detail = row["multimind"]
        assert row["seat"] in (1, 7) and row["v1_status"] == "no_action"
        assert detail["plan_delivered_to_actor"] and detail["speech_committed"]
        assert detail["alignment"]["stop_compliance"]


def test_multimind_requires_wolf_assignment_and_explicit_resource_before_gameplay():
    script, (env, agents, roles, profiles) = _build_gameplay_fixture("multimind")
    with pytest.raises(ValueError, match="bound Reasoner"):
        run_game(env, agents, roles, gameplay_only=True)
    scorer = _SpeechLengthScorer()
    misplaced = list(agents)
    misplaced[0], misplaced[1] = misplaced[1], misplaced[0]
    with pytest.raises(ValueError, match="requires Werewolf seats"):
        run_game(env, misplaced, roles, gameplay_only=True, multimind_reasoner=scorer)
    with pytest.raises(ValueError, match="only supported for Gameplay-only"):
        run_game(env, agents, roles, multimind_reasoner=scorer)
    assert not script.requests and not scorer.histories and env.game_count == 0

    config = runtime_config()
    config["agent_config"]["all_candidates"][0]["model_params"]["decision_method"] = "multimind"
    seats = list(profiles)
    seats[0], seats[1] = seats[1], seats[0]
    config["agent_config"]["allow_cross_team_profiles"] = True
    with pytest.raises(ValueError, match="MultiMind profile requires a Werewolf seat"):
        build_runtime(config, None, roles=roles, random_seed=17, seat_profiles=seats,
                      backends={name: script for name in config["backends"]})
    assert not script.requests


@pytest.fixture
def source_snapshot():
    return {
        "commit": "a" * 40,
        "branch": "twd/mainline",
        "file_sha256": {"werewolf/game_runtime.py": "b" * 64},
        "tracked_source_diff_sha256": "c" * 64,
        "worktree_status": {"entries": [" M werewolf/game_runtime.py"]},
    }


@pytest.mark.parametrize("method", ["react", "dpins:no", "dpins:llm", "dpins:random"])
def test_gameplay_source_snapshot_preserves_runtime(method, source_snapshot):
    _, (env, agents, roles, _) = _build_gameplay_fixture(method, use_openai_backend=True)
    ordinary = run_game(env, agents, roles, gameplay_only=True, record_requests=True,
                        request_timeout=7.5, max_steps=80)
    ordinary_client = agents[0].backend.client
    assert re.fullmatch(r"[0-9a-f]{40}", ordinary.source_identity["commit"])

    _, (env, agents, roles, _) = _build_gameplay_fixture(method, use_openai_backend=True)
    original_snapshot = deepcopy(source_snapshot)
    with patch.object(game_runtime, "_gameplay_source_identity", side_effect=AssertionError("unexpected Git lookup")) as collect:
        supplied = run_game(env, agents, roles, gameplay_only=True, record_requests=True,
                            request_timeout=7.5, max_steps=80, source_identity=source_snapshot,
                            check_gameplay_interrupt=lambda: None)
    collect.assert_not_called()
    assert supplied.source_identity == source_snapshot == original_snapshot
    supplied.source_identity["worktree_status"]["entries"].append("?? added.py")
    assert source_snapshot == original_snapshot

    supplied_client = agents[0].backend.client
    assert supplied_client.request_arguments == ordinary_client.request_arguments
    assert supplied_client.options == ordinary_client.options

    def without_identity_and_timing(result):
        return replace(
            result, source_identity={}, elapsed_seconds=None,
            request_records=tuple({k: v for k, v in row.items() if k != "elapsed_seconds"}
                                  for row in result.request_records),
        )

    assert ordinary.terminal_reached and ordinary.winner == ordinary.replay_winner
    assert without_identity_and_timing(supplied) == without_identity_and_timing(ordinary)


@pytest.mark.parametrize(("field", "value"), [
    ("commit", "UNKNOWN"),
    ("branch", " "),
    ("tracked_source_diff_sha256", "UNKNOWN"),
    ("file_sha256", {}),
    ("file_sha256", {"werewolf/game_runtime.py": "UNKNOWN"}),
])
def test_gameplay_rejects_invalid_source_snapshot_before_reset(source_snapshot, field, value):
    backend, (env, agents, roles, _) = _build_gameplay_fixture("react")
    source_snapshot[field] = value
    with pytest.raises(ValueError, match=f"source_identity.{field}"):
        run_game(env, agents, roles, gameplay_only=True, source_identity=source_snapshot)
    assert env.game_count == 0 and not backend.requests


def test_source_snapshot_is_gameplay_only(source_snapshot):
    backend, (env, agents, roles, _) = _build_gameplay_fixture("react")
    with pytest.raises(ValueError, match="source_identity is only supported for Gameplay-only"):
        run_game(env, agents, roles, source_identity=source_snapshot)
    assert env.game_count == 0 and not backend.requests


@pytest.mark.parametrize(("boundary", "signum", "stage"), [
    ("request", signal.SIGTERM, "agent_action"),
    ("step_return", signal.SIGTERM, "step_commit"),
    ("between_steps", signal.SIGHUP, "between_steps"),
    ("result_assembly", signal.SIGINT, "result_assembly"),
])
def test_gameplay_signal_preserves_committed_action_evidence(monkeypatch, boundary, signum, stage):
    def build_game():
        _, (env, agents, roles, _) = _build_gameplay_fixture("react", use_openai_backend=True)
        return env, agents, roles

    options = {"gameplay_only": True, "record_requests": True, "request_timeout": 7.5, "max_steps": 80}
    normal = run_game(*build_game(), **options)
    assert normal.terminal_reached and normal.failure_type is None
    env, agents, roles = build_game()
    accepted = []
    observed_signal = None

    def checkpoint():
        if observed_signal is not None:
            raise InterruptedError(f"Pilot interrupted by {observed_signal}")

    def handle_signal(number, _frame):
        nonlocal observed_signal
        observed_signal = signal.Signals(number).name
        scope = _request_context.get()
        if scope is not None and scope.kind != "UNKNOWN":
            checkpoint()

    class NextObservation(dict):
        def __getitem__(self, key):
            if key == "current_act_idx":
                signal.raise_signal(signum)
            return super().__getitem__(key)

    native_step = env.step

    def step(action):
        observation, reward, done, info = native_step(action)
        accepted.append(action)
        if boundary == "between_steps":
            observation = NextObservation(observation)
        return observation, reward, done, info

    monkeypatch.setattr(env, "step", step)
    if boundary == "request":
        client = agents[0].backend.client
        create = client.create

        def interrupted_request(**kwargs):
            if accepted:
                signal.raise_signal(signum)
            return create(**kwargs)

        monkeypatch.setattr(client, "create", interrupted_request)
    elif boundary == "result_assembly":
        assemble = game_runtime._complete_gameplay_result

        def interrupted_assembly(result, **metrics):
            completed = assemble(result, **metrics)
            if result.failure_type is None:
                signal.raise_signal(signum)
            return completed

        monkeypatch.setattr(game_runtime, "_complete_gameplay_result", interrupted_assembly)

    previous = signal.signal(signum, handle_signal)
    previous_trace = sys.gettrace()
    try:
        if boundary == "step_return":
            lines, start = getsourcelines(run_game)
            before_count = next(start + i for i, line in enumerate(lines)
                                if line.strip() == "step = len(submitted_actions)")

            def at_commit(frame, event, _arg):
                if frame.f_code is not run_game.__code__:
                    return None
                if event == "line" and frame.f_lineno == before_count:
                    signal.raise_signal(signum)
                return at_commit

            sys.settrace(at_commit)
        result = run_game(env, agents, roles, **options, check_gameplay_interrupt=checkpoint)
    finally:
        if boundary == "step_return":
            sys.settrace(previous_trace)
        signal.signal(signum, previous)

    terminal = boundary == "result_assembly"
    action_count = normal.completed_steps if terminal else 1
    assert result.completed_steps == len(accepted) == action_count
    assert result.submitted_actions == tuple(accepted) == normal.submitted_actions[:action_count]
    assert result.terminal_reached is terminal and result.failure_stage == stage
    with pytest.raises(ValueError, match="completed gameplay result"):
        verify_gameplay_replay(result)
    if terminal:
        assert result.winner == result.replay_winner == normal.winner
        assert result.public_event_digest == normal.public_event_digest
    else:
        assert result.winner is result.replay_winner is None
    if boundary == "request":
        assert result.failure_type == "BackendError" and result.failure_cause_type == "InterruptedError"
        assert result.failure_message.endswith(f"Pilot interrupted by {signum.name}")
        row = result.request_records[-1]
        assert row["error_type"] == "InterruptedError" and row["step"] == action_count
        assert row["response_received"] == row["total_tokens"] == "UNKNOWN"
    else:
        assert result.failure_type == "InterruptedError"
        assert result.failure_message == f"Pilot interrupted by {signum.name}"


def test_gameplay_interruption_without_step_receipt_does_not_add_action(monkeypatch):
    _, (env, agents, roles, _) = _build_gameplay_fixture("react", use_openai_backend=True)

    def interrupted_before_acceptance(_action):
        raise InterruptedError("step acceptance unknown")

    monkeypatch.setattr(env, "step", interrupted_before_acceptance)
    result = run_game(env, agents, roles, gameplay_only=True, record_requests=True)
    assert result.completed_steps == 0 and result.submitted_actions == ()
    assert result.failure_stage == "environment_step" and result.failure_type == "InterruptedError"
    assert result.failure_message == "step acceptance unknown"
    assert not result.terminal_reached and result.winner is None


@pytest.mark.parametrize(("sdk_kind", "kind", "error", "stage"), [
    ("private_belief_report", "pre_report", TimeoutError, "speaker_pre"),
    ("speech_parser", "v1_parser", ConnectionError, "environment_step"),
    ("night_action_selection", "night_action", KeyboardInterrupt, "agent_action"),
])
def test_gameplay_only_stops_without_retry_when_response_is_unknown(sdk_kind, kind, error, stage):
    _, (env, agents, roles, _) = _build_gameplay_fixture("react", use_openai_backend=True)
    client = agents[0].backend.client
    client.error_kind = sdk_kind
    client.error = error("scripted transport failure")

    result = run_game(env, agents, roles, gameplay_only=True, max_steps=6,
                      record_requests=True, request_timeout=7.5)

    assert result.winner is None and not result.terminal_reached and result.replay_winner is None
    assert result.failure_stage == stage and result.elapsed_seconds >= 0
    if error is KeyboardInterrupt:
        assert result.failure_type == "KeyboardInterrupt" and result.failure_cause_type is None
    else:
        assert (result.failure_type, result.failure_cause_type) == ("BackendError", error.__name__)
    row = result.request_records[-1]
    assert (row["kind"], row["attempt"], row["status"], row["error_type"]) == (kind, 1, "request_error", error.__name__)
    assert row["step"] == result.completed_steps and row["phase"] == result.failure_phase
    assert row["response_received"] == "UNKNOWN" and row["sdk_retries"] == 0
    assert row["input_tokens"] == row["output_tokens"] == row["total_tokens"] == "UNKNOWN"
    assert result.model_requests == len(client.calls) and result.retries == 0
    assert client.calls.count((sdk_kind, 7.5)) == 1
    assert result.token_usage == dict.fromkeys(("input_tokens", "output_tokens", "total_tokens"), "UNKNOWN")
    if kind == "v1_parser":
        assert result.speech_results[-1]["v1_status"] == "UNKNOWN"
        assert result.completed_steps + 1 == len(result.submitted_actions)
    else:
        assert result.completed_steps == len(result.submitted_actions)


@pytest.mark.parametrize(("sdk_kind", "kind"), [
    ("night_action_selection", "night_action"),
    ("private_belief_report", "pre_report"),
    ("speech_parser", "v1_parser"),
])
def test_gameplay_only_retries_known_invalid_text_with_separate_attempts(sdk_kind, kind):
    _, (env, agents, roles, _) = _build_gameplay_fixture("react", use_openai_backend=True)
    client = agents[0].backend.client
    client.repair_kind = sdk_kind

    result = run_game(env, agents, roles, gameplay_only=True, max_steps=6,
                      record_requests=True, request_timeout=7.5)

    assert result.failure_stage == "step_limit" and result.completed_steps == 6
    assert result.model_requests == len(client.calls) and result.retries == 1
    first = next(row for row in result.request_records if row["kind"] == kind)
    pair = [row for row in result.request_records if row["kind"] == kind and row["step"] == first["step"]]
    assert [(row["attempt"], row["status"], row["response_received"]) for row in pair] == [
        (1, "response_received", True), (2, "response_received", True),
    ]
    assert pair[0]["input_tokens"] == "UNKNOWN" and pair[1]["input_tokens"] == 11
    assert all(row["sdk_retries"] == 0 for row in pair)
    if kind != "pre_report":
        assert pair[0]["validation_error_type"] in {"GameplayActionValidationError", "SpeechActionValidationError"}


def test_gameplay_only_random_ins_repeats_and_replay_checks_the_result():
    results = []
    choices = []
    for _ in range(2):
        backend, (env, agents, roles, _) = _build_gameplay_fixture("dpins:random", winner="Villager")
        result = run_game(env, agents, roles, gameplay_only=True, max_steps=80)
        assert result.failure_type is None, result
        assert result.winner == result.replay_winner == "Villager" and env.day == 3
        results.append(result)
        choices.append([
            prompt.split("本次发言必须执行的讨论策略：", 1)[1].splitlines()[0]
            for _, kind, prompt in backend.requests if kind == "direct_gameplay_speech"
        ])
    assert choices[0] and choices[0] == choices[1]
    assert results[0].submitted_actions == results[1].submitted_actions
    assert results[0].public_event_digest == results[1].public_event_digest

    result = results[0]
    actions = result.submitted_actions
    with pytest.raises(ValueError, match="recorded winner"):
        verify_gameplay_replay(replace(result, winner="Werewolf"))
    with pytest.raises(ValueError, match="native game terminal"):
        verify_gameplay_replay(replace(result, submitted_actions=actions[:-1]))
    with pytest.raises(ValueError, match="not legal"):
        verify_gameplay_replay(replace(result, submitted_actions=(("check", actions[0][1]),) + actions[1:]))
    with pytest.raises(ValueError, match="diverged"):
        verify_gameplay_replay(replace(result, public_event_digest="0" * 64))


@pytest.mark.parametrize(("fault", "stage", "error_type"), [
    ("illegal", "agent_action", "ValueError"),
    ("night_action_selection", "agent_action", "GameplayGenerationExhausted"),
    ("private_belief_report", "speaker_pre", "RuntimeError"),
    ("speech_parser", "environment_step", "V1SpeechPerceptionExhausted"),
    ("limit", "step_limit", "TimeoutError"),
    ("early_done", "native_terminal", "ValueError"),
])
def test_gameplay_only_failure_has_no_winner(monkeypatch, fault, stage, error_type):
    backend, (env, agents, roles, _) = _build_gameplay_fixture("react")
    if fault == "illegal":
        monkeypatch.setattr(agents[0], "act", lambda _observation: ("check", 2))
    elif fault == "early_done":
        native_step = env.step

        def pretend_done(action):
            observation, reward, _, _ = native_step(action)
            return observation, reward, True, {"Werewolf": 1}

        monkeypatch.setattr(env, "step", pretend_done)
    else:
        backend.fail_request = fault
    max_steps = 1 if fault == "limit" else 80

    result = run_game(env, agents, roles, gameplay_only=True, max_steps=max_steps)

    assert not result.terminal_reached and result.winner is None and result.replay_winner is None
    assert result.failure_stage == stage and result.failure_type == error_type
    assert result.failure_phase is not None and result.failure_message
    if fault == "illegal":
        assert result.submitted_actions == () and not backend.requests
    if fault == "private_belief_report":
        assert not any(kind == "day_cognition_report_v4" for _, kind, _ in backend.requests)
    if fault == "speech_parser":
        assert result.submitted_actions[-1][0] == "speech"
        assert result.speech_parser_attempts == 3
    with pytest.raises(ValueError, match="completed gameplay result"):
        verify_gameplay_replay(result)


def test_gameplay_only_rejects_canonical_objects_before_reset():
    backend, (env, agents, roles, _) = _build_gameplay_fixture("react")
    with pytest.raises(ValueError, match="cannot use Canonical"):
        run_game(env, agents, roles, gameplay_only=True, canonical_recorder=object(), call_audit=object())
    assert env.game_count == 0 and not backend.requests


def test_gameplay_only_rejects_a_mismatched_random_ins_seed():
    backend, (env, agents, roles, _) = _build_gameplay_fixture("dpins:random")
    agents[0].game_seed = 18
    with pytest.raises(ValueError, match="agent seed does not match"):
        run_game(env, agents, roles, gameplay_only=True)
    assert env.game_count == 0 and not backend.requests


@pytest.mark.parametrize(("seats", "message"), [
    (["profile-a"] * 6, "one profile name per role"),
    (["unknown"] * 7, "unknown profile name"),
    (["profile-a"] * 7, "same profile on both factions"),
])
def test_gameplay_seat_assignment_rejects_invalid_fixed_profiles(seats, message):
    config = runtime_config()
    with pytest.raises(ValueError, match=message):
        build_runtime(
            config, None, roles=ROLES, random_seed=17, seat_profiles=seats,
            backends={name: RecordingBackend(name) for name in config["backends"]},
        )


if __name__ == "__main__":
    unittest.main()
