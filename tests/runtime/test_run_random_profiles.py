from copy import deepcopy
from dataclasses import replace
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
