from collections import Counter
from dataclasses import asdict
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from scripts import run_werewolf_baseline as baseline
from tests.canonical_collection.test_public_history import _event, _phase
from tests.tom.test_multimind import multimind_checkpoint
from werewolf.artifact_io import canonical_json_bytes
from werewolf.backends import OpenAICompatibleBackend
from werewolf.canonical_collection.public_history import PLAYER_IDS, freeze_public_event_history
from werewolf.game_runtime import GameplayOnlyResult, count_game_votes, summarize_game_votes
from werewolf.interventions.gameplay import digest, write_once

from .test_run_random_profiles import _GameplayBackend, _GameplayClient


METHODS = ("react", "dpins:no", "dpins:llm", "dpins:random")
RUNTIME_YAML = Path(__file__).resolve().parents[2] / "configs/runtime/local-qwen35-9b.yaml"


def _run_first(plan_path, config_path):
    return baseline.main([
        "run", "--plan", str(plan_path), "--runtime-config", str(config_path), "--first-scenario",
    ])


def _summary(plan_path, capsys):
    capsys.readouterr()
    assert baseline.main(["summarize", "--plan", str(plan_path)]) == 0
    return json.loads(capsys.readouterr().out)


@pytest.fixture
def frozen_plan(tmp_path, monkeypatch):
    directory = tmp_path.resolve()
    config = yaml.safe_load(RUNTIME_YAML.read_bytes())
    # 现有 SDK Mock 通过模型名区分 V1 Parser 和自由文本生成。
    config["parser"]["model"] = "parser-model"
    config_path = directory / "runtime.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    source = {
        "commit": "a" * 40, "branch": "twd/mainline", "tracked_source_diff_sha256": "b" * 64,
        "file_sha256": {"werewolf/game_runtime.py": "c" * 64},
    }
    monkeypatch.setattr(baseline, "_source_identity", lambda: source)

    def unexpected_backend(*args, **kwargs):
        pytest.fail("测试必须经 Mock SDK 才能派发请求")

    monkeypatch.setattr(baseline, "load_named_backends", unexpected_backend)
    plan_path = directory / "scenarios.json"
    assert baseline.main(["plan", "--plan", str(plan_path), "--runtime-config", str(config_path)]) == 0
    return plan_path, config_path, json.loads(plan_path.read_bytes())


@pytest.fixture
def mock_sdk(frozen_plan, monkeypatch):
    plan_path, _, plan = frozen_plan
    scene = plan["scenarios"][0]

    class FirstSceneBackend(_GameplayBackend):
        def _vote(self, _prompt, legal):
            return next((seat for seat in legal if seat and scene["roles"][seat - 1] != "Werewolf"), 0)

    client = _GameplayClient(FirstSceneBackend("Werewolf"))
    backend = OpenAICompatibleBackend(client=client, supports_json_schema=True)
    load = Mock(return_value={"local_qwen": backend})
    monkeypatch.setattr(baseline, "load_named_backends", load)
    dispatch = client.create
    run_game = baseline.game_runtime.eval
    games = []

    def verify_seats_and_run(env, agents, roles, **kwargs):
        method = agents[0].decision_method
        assert list(roles) == scene["roles"] and env.random_seed == scene["seed"]
        assert [agent.profile_name for agent in agents] == ["baseline-wolf"] * 2 + ["local_qwen"] * 5
        assert [agent.decision_method for agent in agents] == [method] * 2 + [None] * 5
        assert kwargs["gameplay_only"] and kwargs["record_requests"]
        assert kwargs["max_steps"] == 128 and kwargs["source_identity"] == plan["source_identity"]
        if method == "dpins:random":
            assert [agent.game_seed for agent in agents[:2]] == [scene["seed"]] * 2
        games.append((method, env, agents))
        return run_game(env, agents, roles, **kwargs)

    def verify_attempt_and_dispatch(**kwargs):
        method = games[-1][0]
        attempt_path, result_path = baseline._task_paths(plan_path.parent, scene, method)
        assert json.loads(attempt_path.read_bytes())["task"] == {
            "plan_sha256": plan["plan_sha256"], "scenario_id": "scenario-10000", "method": method,
        }
        assert not result_path.exists()
        assert kwargs["timeout"] == 300 and client.options[-1] == {"max_retries": 0}
        return dispatch(**kwargs)

    monkeypatch.setattr(baseline.game_runtime, "eval", verify_seats_and_run)
    monkeypatch.setattr(client, "create", verify_attempt_and_dispatch)
    return client, load, games


def _sample_result(plan, scene, method, **changes):
    fields = {
        "seed": scene["seed"], "roles": tuple(scene["roles"]), "seat_profiles": tuple(scene["seat_profiles"]),
        "decision_methods": tuple(baseline._decision_methods(scene, method)),
        "source_identity": plan["source_identity"], "request_timeout_seconds": 300,
        "terminal_reached": True, "winner": "Werewolf", "replay_winner": "Werewolf",
        "public_event_digest": "d" * 64, "submitted_actions": (("kill", 3),), "completed_steps": 1,
    }
    fields.update(changes)
    return asdict(GameplayOnlyResult(**fields))


def _publish_sample(plan_path, plan, scene, method, game=None):
    attempt_path, result_path = baseline._task_paths(plan_path.parent, scene, method)
    task = baseline._task_key(plan, scene, method)
    attempt = {"task": task, "started_at_utc": "2026-01-01T00:00:00+00:00"}
    write_once(attempt_path, attempt)
    if game is not None:
        write_once(result_path, {
            "task": task, "attempt_sha256": digest(attempt),
            "gameplay_sha256": digest(game), "gameplay": game,
        })
    return attempt_path, result_path


def test_plan_is_deterministic_balanced_and_immutable(frozen_plan):
    plan_path, config_path, plan = frozen_plan
    original = plan_path.read_bytes()
    copy_path = plan_path.with_name("copy.json")
    assert baseline.main(["plan", "--plan", str(copy_path), "--runtime-config", str(config_path)]) == 0
    assert copy_path.read_bytes() == original
    assert baseline.main(["plan", "--plan", str(plan_path), "--runtime-config", str(config_path)]) == 1
    assert plan_path.read_bytes() == original
    digest_body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    assert sha256(canonical_json_bytes(digest_body)).hexdigest() == plan["plan_sha256"]

    pair_counts, seats, positions = Counter(), Counter(), Counter()
    scenarios = plan["scenarios"]
    assert len(scenarios) == len({scene["scenario_id"] for scene in scenarios}) == 210
    assert {scene["seed"] for scene in scenarios} == set(range(10000, 10210))
    assert scenarios[0]["roles"] == ["Werewolf", "Werewolf", "Villager", "Villager", "Villager", "Seer", "Witch"]
    for index, scene in enumerate(scenarios):
        assert Counter(scene["roles"]) == {"Werewolf": 2, "Seer": 1, "Witch": 1, "Villager": 3}
        wolves = tuple(seat for seat, role in enumerate(scene["roles"], 1) if role == "Werewolf")
        pair_counts[scene["batch"], wolves] += 1
        for seat, role in enumerate(scene["roles"], 1):
            seats[scene["batch"], seat, role] += 1
        offset = index % 4
        assert tuple(scene["method_order"]) == METHODS[offset:] + METHODS[:offset]
        for position, method in enumerate(scene["method_order"]):
            positions[method, position] += 1

    assert pair_counts == {(batch, pair): 5 for batch in (1, 2) for pair in combinations(range(1, 8), 2)}
    for batch in (1, 2):
        for seat in range(1, 8):
            assert [seats[batch, seat, role] for role in ("Werewolf", "Seer", "Witch", "Villager")] == [30, 15, 15, 45]
    for method in METHODS:
        assert sorted(positions[method, position] for position in range(4)) == [52, 52, 53, 53]


def test_first_scene_uses_frozen_seats_before_dispatch_and_archives_replay(frozen_plan, mock_sdk, capsys):
    plan_path, config_path, _ = frozen_plan
    client, load, games = mock_sdk
    assert _run_first(plan_path, config_path) == 0
    assert [method for method, _, _ in games] == list(METHODS)
    assert len({id(env) for _, env, _ in games}) == 4
    assert len({id(agent) for _, _, agents in games for agent in agents}) == 28
    assert load.call_count == 1 and load.call_args.kwargs == {"env_file": None, "max_retries": 0}
    paths = sorted(plan_path.parent.glob("*/*.json"))
    archives = {path: path.read_bytes() for path in paths}
    assert len(archives) == 8
    call_count = len(client.calls)
    assert _run_first(plan_path, config_path) == 0
    assert len(client.calls) == call_count and load.call_count == 1
    assert {path: path.read_bytes() for path in paths} == archives
    summary = _summary(plan_path, capsys)
    assert {row["method"] for row in summary["games"]} == set(METHODS)
    assert all(row["vote_counts"]["normal_vote_rounds"] >= 1 for row in summary["games"])
    assert {path: path.read_bytes() for path in archives} == archives
    for stats in summary["methods"].values():
        assert (stats["planned"], stats["not_started"], stats["started"]) == (210, 209, 1)
        assert stats["native_terminal"] == stats["replay_passed"] == stats["wolf_wins"] == 1
        assert stats["conditional_wolf_win_rate"] == 1
        assert stats["votes"]["games_with_vote_evidence"] == 1
        assert stats["tom_integration"]["complete_games"]["integration_success_rate"]["value"] is None
        assert stats["tom_integration"]["all_observed_attempts"]["prediction_success_rate"]["value"] is None


def test_transport_unknown_is_archived_and_never_redispatched(frozen_plan, mock_sdk, capsys):
    plan_path, config_path, plan = frozen_plan
    client, _, games = mock_sdk
    client.error_kind = "night_action_selection"
    client.error = TimeoutError("scripted transport failure")
    assert _run_first(plan_path, config_path) == 1
    assert client.calls == [("night_action_selection", 300)]
    attempt, result = baseline._task_paths(plan_path.parent, plan["scenarios"][0], "react")
    originals = attempt.read_bytes(), result.read_bytes()
    game = json.loads(originals[1])["gameplay"]
    assert game["winner"] is None and game["request_records"][0]["response_received"] == "UNKNOWN"
    client.error_kind = None
    assert _run_first(plan_path, config_path) == 0
    assert [method for method, _, _ in games] == list(METHODS)
    assert (attempt.read_bytes(), result.read_bytes()) == originals
    stats = _summary(plan_path, capsys)["methods"]["react"]
    assert (stats["unknown"], stats["failed"], stats["native_terminal"], stats["wolf_wins"]) == (1, 0, 0, 0)
    assert stats["conditional_wolf_win_rate"] is None
    assert stats["cost"]["model_requests"] == {"sum": 1, "games_available": 1}
    assert stats["cost"]["total_tokens"] == {"sum": None, "games_available": 0}


def test_summary_excludes_failures_after_confirmed_replay(frozen_plan, capsys):
    plan_path, _, plan = frozen_plan
    scene = plan["scenarios"][0]
    cases = (
        ("react", "RuntimeError", "failed"),
        ("dpins:no", "KeyboardInterrupt", "unknown"),
    )
    archived = {}
    for method, failure, _ in cases:
        _, result = _publish_sample(plan_path, plan, scene, method, _sample_result(
            plan, scene, method, failure_type=failure, failure_stage="result_assembly",
        ))
        archived[result] = result.read_bytes()

    summary = _summary(plan_path, capsys)
    assert summary["record_errors"] == []
    assert summary["games"] == []
    for method, _, status in cases:
        stats = summary["methods"][method]
        assert stats[status] == 1
        assert stats["completed"] == stats["wolf_wins"] == 0
        assert stats["native_terminal"] == stats["replay_passed"] == 1
        assert stats["conditional_wolf_win_rate"] is None
    assert {path: path.read_bytes() for path in archived} == archived


def test_native_votes_use_tied_pk_targets_living_seats_and_daytime_exile():
    roles = dict(zip(PLAYER_IDS, ["Werewolf", "Werewolf", "Villager", "Villager", "Villager", "Seer", "Witch"]))

    def result(index, ballots):
        return _event(index, "vote_result", votes=[
            {"voter": f"player{voter}", "target": f"player{target}" if target else None}
            for voter, target in ballots.items()
        ])

    history = freeze_public_event_history([
        _phase(0, 0, "night"),
        _event(1, "death_announcement", dead_players=[]),
        _phase(2, 1, "discussion"),
        _phase(3, 1, "vote"),
        result(4, {1: 2, 2: 4, 3: 1, 4: 1, 5: 4, 6: None, 7: None}),
        _event(5, "exile_result", exiled_players=[]),
        _phase(6, 1, "pk_discussion"),
        _phase(7, 1, "pk_vote"),
        result(8, {1: 4, 2: 1, 3: 1, 4: 1, 5: 4, 6: None, 7: None}),
        _event(9, "exile_result", exiled_players=["player1"]),
        _phase(10, 1, "night"),
        _event(11, "death_announcement", dead_players=["player6"]),
        _phase(12, 2, "discussion"),
        _phase(13, 2, "vote"),
        result(14, {2: 3, 3: 4, 4: 2, 5: 3, 7: None}),
        _event(15, "exile_result", exiled_players=["player3"]),
        _phase(16, 2, "night"),
        _event(17, "death_announcement", dead_players=["player2"]),
    ])
    counts = count_game_votes(history, roles)
    assert counts == {
        "wolf_participations": 2, "wolf_votes_received": 7, "wolf_votes_from_villagers": 5,
        "wolf_votes_from_teammates": 2, "wolf_eligible_rounds": 4, "villager_valid_votes": 9,
        "wolves_exiled": 1, "villagers_exiled": 1, "day1_wolf_exile_games": 1,
        "wolf_ballots": 5, "villager_ballots": 14, "wolf_abstentions": 0, "villager_abstentions": 5,
        "normal_vote_rounds": 2, "pk_vote_rounds": 1,
    }
    summary = summarize_game_votes([counts, {**counts, "wolf_eligible_rounds": 6}])
    assert summary["AVR"] == {"numerator": 14, "denominator": 4, "value": 3.5}
    assert summary["team_AVR"]["value"] == 7
    assert summary["VPR"] == {"numerator": 14, "denominator": 10, "value": 1.4}
    assert summary["WVS"]["value"] == 5 / 9
    assert summary["wolf_exile_rate"]["value"] == .5
    assert summary["D1WER"] == {"numerator": 2, "denominator": 2, "value": 1.0}
    assert summary["GMR"] == {"numerator": 2, "denominator": 4, "value": .5}
    assert summary["abstention_rate"] == {"numerator": 10, "denominator": 38, "value": 5 / 19}
    assert summary["wolf_abstention_rate"]["value"] == 0
    assert summary["villager_abstention_rate"]["value"] == 5 / 14


def test_vote_rates_zero_denominator_is_na():
    summary = summarize_game_votes([])
    for field in ("AVR", "team_AVR", "VPR", "WVS", "wolf_exile_rate", "D1WER", "GMR",
                  "abstention_rate", "wolf_abstention_rate", "villager_abstention_rate"):
        assert summary[field] == {"numerator": 0, "denominator": 0, "value": None}

    roles = dict(zip(PLAYER_IDS, ["Werewolf", "Werewolf", "Villager", "Villager", "Villager", "Seer", "Witch"]))
    history = freeze_public_event_history([
        _phase(0, 0, "night"),
        _event(1, "death_announcement", dead_players=["player1"]),
        _phase(2, 1, "discussion"),
        _phase(3, 1, "vote"),
        _event(4, "vote_result", votes=[{"voter": seat, "target": None} for seat in PLAYER_IDS[1:]]),
        _event(5, "exile_result", exiled_players=[]),
    ])
    counts = count_game_votes(history, roles)
    assert counts["wolf_eligible_rounds"] == 1 and counts["wolves_exiled"] == 0
    assert summarize_game_votes([counts])["WVS"] == {"numerator": 0, "denominator": 0, "value": None}


def test_restart_preserves_completed_failed_and_claim_only_tasks(frozen_plan, mock_sdk, capsys):
    plan_path, config_path, plan = frozen_plan
    _, _, games = mock_sdk
    scene = plan["scenarios"][0]
    _publish_sample(plan_path, plan, scene, "react", _sample_result(plan, scene, "react"))
    _publish_sample(plan_path, plan, scene, "dpins:no", _sample_result(
        plan, scene, "dpins:no", winner=None, replay_winner=None, failure_type="ValueError", failure_stage="replay",
    ))
    _, missing_result = _publish_sample(plan_path, plan, scene, "dpins:llm")
    assert _run_first(plan_path, config_path) == 0
    assert [method for method, _, _ in games] == ["dpins:random"]
    assert not missing_result.exists()

    summary = _summary(plan_path, capsys)["methods"]
    assert summary["react"]["completed"] == 1
    failed = summary["dpins:no"]
    assert failed["failed"] == failed["native_terminal"] == 1
    assert failed["replay_passed"] == failed["wolf_wins"] == 0
    assert failed["conditional_wolf_win_rate"] is None
    unknown = summary["dpins:llm"]
    assert unknown["started"] == unknown["unknown"] == 1
    assert unknown["failed"] == unknown["wolf_wins"] == 0
    assert unknown["conditional_wolf_win_rate"] is None


@pytest.mark.parametrize("corruption", ("attempt", "gameplay"))
def test_result_identity_must_match_frozen_task(frozen_plan, capsys, corruption):
    plan_path, config_path, plan = frozen_plan
    scene = plan["scenarios"][0]
    _, result = _publish_sample(plan_path, plan, scene, "react", _sample_result(plan, scene, "react"))
    original = result.read_bytes()
    damaged = json.loads(original)
    if corruption == "attempt":
        damaged["attempt_sha256"] = "e" * 64
    else:
        damaged["gameplay"]["seed"] += 1
        damaged["gameplay_sha256"] = digest(damaged["gameplay"])
    with pytest.raises(FileExistsError):
        write_once(result, damaged)
    assert result.read_bytes() == original
    result.write_bytes(canonical_json_bytes(damaged))
    assert _run_first(plan_path, config_path) == 1
    summary = _summary(plan_path, capsys)
    assert len(summary["record_errors"]) == 1
    assert summary["methods"]["react"]["unknown"] == 1
    assert summary["methods"]["react"]["wolf_wins"] == 0


@pytest.mark.parametrize("changed", ("source", "config"))
def test_run_requires_frozen_source_and_yaml(frozen_plan, monkeypatch, changed):
    plan_path, config_path, plan = frozen_plan
    if changed == "config":
        config_path.write_bytes(config_path.read_bytes() + b"\n# config changed\n")
    else:
        source = {**plan["source_identity"], "commit": "d" * 40}
        monkeypatch.setattr(baseline, "_source_identity", lambda: source)
    assert _run_first(plan_path, config_path) == 1
    assert not (plan_path.parent / "attempts").exists()


@pytest.fixture
def paired_plan(frozen_plan, multimind_checkpoint, monkeypatch):
    base_path, config_path, base_plan = frozen_plan
    old_path = base_path.parent / "historical" / "scenarios.json"
    write_once(old_path, base_plan)
    new_path = base_path.parent / "multimind" / "scenarios.json"
    checkpoint_path, identity, *_ = multimind_checkpoint
    monkeypatch.setattr(baseline, "_source_identity", lambda *, multimind=False: base_plan["source_identity"])
    options = ["--runtime-config", str(config_path), "--reasoner-checkpoint", str(checkpoint_path),
               "--reasoner-device", "cpu"]
    assert baseline.main(["plan", "--plan", str(new_path), "--method", "multimind", "--paired-plan", str(old_path),
                          "--reasoner-checkpoint-digest", identity, *options]) == 0
    return old_path, new_path, baseline._read_plan(new_path), options


def test_multimind_reuses_frozen_scenarios_and_keeps_old_plan_readable(paired_plan, capsys):
    old, new, plan, options = paired_plan
    original = old.read_bytes()
    parent = baseline._read_plan(old)
    assert "methods" not in parent and plan["methods"] == ["multimind"]
    assert plan["paired_plan_sha256"] == parent["plan_sha256"]
    assert plan["scenarios"] == parent["scenarios"] and plan["runtime_config_sha256"] == parent["runtime_config_sha256"]
    assert set(_summary(old, capsys)["methods"]) == set(METHODS)
    summary = _summary(new, capsys)["methods"]["multimind"]
    assert summary["planned"] == summary["not_started"] == 210
    assert summary["tom_integration"]["all_observed_attempts"]["integration_success_rate"]["value"] is None
    assert baseline.main(["plan", "--plan", str(old.with_name("multimind.json")), "--method", "multimind",
                          "--paired-plan", str(old), "--reasoner-checkpoint-digest",
                          plan["reasoner"]["checkpoint_sha256"], *options]) == 1
    assert old.read_bytes() == original and not old.with_name("multimind.json").exists()


def test_multimind_loads_one_real_checkpoint_for_two_mock_games(paired_plan, monkeypatch, capsys):
    from werewolf.tom import multimind

    old, plan_path, plan, options = paired_plan
    original = old.read_bytes()
    scenes = plan["scenarios"][:2]
    tested_scenes = {scene["scenario_id"] for scene in scenes}

    class ScenarioBackend(_GameplayBackend):
        def _vote(self, _prompt, legal):
            return next((seat for seat in legal if seat and self.roles[seat - 1] != "Werewolf"), 0)

    script = ScenarioBackend("Werewolf")
    client = _GameplayClient(script)
    backend = OpenAICompatibleBackend(client=client, supports_json_schema=True)
    named = Mock(return_value={"local_qwen": backend})
    monkeypatch.setattr(baseline, "load_named_backends", named)
    loader = Mock(wraps=multimind.load_multimind_checkpoint)
    monkeypatch.setattr(multimind, "load_multimind_checkpoint", loader)
    native_eval = baseline.game_runtime.eval
    reasoners = []

    def record_model(env, agents, roles, **kwargs):
        script.roles = list(roles)
        assert [agent.decision_method for agent in agents] == baseline._decision_methods(
            scenes[len(reasoners)], "multimind",
        )
        reasoner = kwargs["multimind_reasoner"]
        assert not reasoner.training and next(reasoner.parameters()).device.type == "cpu"
        reasoners.append(reasoner)
        return native_eval(env, agents, roles, **kwargs)

    monkeypatch.setattr(baseline.game_runtime, "eval", record_model)
    read_task = baseline._read_task
    with monkeypatch.context() as patch:
        def two_tasks(plan, scene, method, *paths):
            return (read_task(plan, scene, method, *paths) if scene["scenario_id"] in tested_scenes
                    else (True, "completed", None, None))

        patch.setattr(baseline, "_read_task", two_tasks)
        assert baseline.main(["run", "--plan", str(plan_path), "--batch", "1", *options]) == 0
        assert baseline.main(["run", "--plan", str(plan_path), "--batch", "1", *options]) == 0
    assert loader.call_count == named.call_count == 1
    assert len(reasoners) == 2 and reasoners[0] is reasoners[1]
    assert old.read_bytes() == original

    stats = _summary(plan_path, capsys)["methods"]["multimind"]
    assert stats["planned"] == 210 and stats["completed"] == stats["replay_passed"] == stats["wolf_wins"] == 2
    assert stats["votes"]["games_with_vote_evidence"] == 2
    assert stats["cost"]["model_requests"]["sum"] == len(client.calls)
    metrics = stats["multimind"]
    logical_count = metrics["planning_attempts"]
    assert logical_count > 0 and metrics["completed_plans"] == metrics["actor_confirmed_boundaries"] == logical_count
    assert metrics["cost"]["candidate_score_calls"]["sum"] == 500 * logical_count
    assert metrics["cost"]["reasoner_forward_calls"]["sum"] == 500 * logical_count
    assert metrics["alignment"]["exact_match"]["denominator"] == logical_count
    assert stats["tom_integration"]["all_observed_attempts"]["integration_success_rate"]["value"] == 1


def test_multimind_summary_preserves_partial_unknown_and_matches_actor_receipts(paired_plan, capsys):
    _, plan_path, plan, _ = paired_plan
    first, second, third = plan["scenarios"][:3]

    def publish_failure(scene, kind, *, receipt):
        seat = next(i for i, role in enumerate(scene["roles"], 1) if role == "Werewolf")
        game = _sample_result(
            plan, scene, "multimind", decision_methods=tuple(baseline._decision_methods(scene, "multimind")),
            terminal_reached=False, winner=None, replay_winner=None, failure_type="RuntimeError",
            public_event_digest=None, submitted_actions=(), completed_steps=0,
            speech_results=({"seat": seat, "step": 0, "kind": kind, "multimind": {
                "day": 1, "planner_attempted": True, "reasoner_prediction_succeeded": True,
                "planner_completed": True, "plan_delivered_to_actor": True, "last_same_phase_speaker": kind == "speech_pk",
                "candidate_score_calls": 500, "reasoner_forward_calls": None, "mcts_elapsed_seconds": 0.1,
                "alignment": None,
            }},),
            request_records=({"kind": "speech_realization", "method": "multimind", "step": 0,
                              "phase": f"1_day_{kind}", "seat": seat if receipt else (seat % 7) + 1,
                              "game_seed": scene["seed"], "dispatched": True, "response_received": True,
                              "status": "response_received"},),
        )
        _publish_sample(plan_path, plan, scene, "multimind", game)

    publish_failure(first, "speech", receipt=True)
    publish_failure(second, "speech_pk", receipt=False)
    _publish_sample(plan_path, plan, third, "multimind")
    stats = _summary(plan_path, capsys)["methods"]["multimind"]
    assert stats["failed"] == 2 and stats["unknown"] == 1 and stats["wolf_wins"] == 0
    assert stats["conditional_wolf_win_rate"] is None
    integration = stats["tom_integration"]["all_observed_attempts"]
    assert integration["verified_minimum"] == {"A": 2, "P": 2, "S": 1}
    assert integration["integration_success_rate"]["value"] == "UNKNOWN"
    evidence = stats["multimind"]
    assert evidence["missing_game_evidence"] == 1 and evidence["actor_plan_handoffs"] == 2
    assert evidence["actor_confirmed_boundaries"] == 1 and evidence["phase_support"]["last_same_phase_speaker"]["boundaries"] == 1
    assert evidence["cost"]["reasoner_forward_calls"]["unknown_boundaries"] == 2
    assert evidence["alignment"]["stop_compliance"]["value"] is None


def test_multimind_requires_pinned_device_and_checkpoint_before_dispatch(paired_plan):
    _, plan_path, _, options = paired_plan
    assert baseline.main(["run", "--plan", str(plan_path), "--first-scenario", *options[:-1], "cuda:0"]) == 1
    wrong = list(options)
    wrong[3] = str(plan_path.parent / "missing-checkpoint")
    assert baseline.main(["run", "--plan", str(plan_path), "--first-scenario", *wrong]) == 1
    assert not (plan_path.parent / "attempts").exists()


def _direct_tom_plan(frozen_plan, monkeypatch, *, smoke=False):
    from scripts import tom_gameplay_collection as collection

    base_path, config_path, parent = frozen_plan
    old = base_path.parent / "historical" / "scenarios.json"
    write_once(old, parent)
    monkeypatch.setattr(collection, "BASELINE_PLAN_SHA256", parent["plan_sha256"])
    monkeypatch.setattr(baseline, "_source_identity", lambda **kwargs: parent["source_identity"])
    directory = base_path.parent / baseline.DIRECT_TOM_EXPERIMENT
    destination = (directory / "smoke" if smoke else directory) / "scenarios.json"
    plan_options = ["--runtime-config", str(config_path), "--direct-tom", "--paired-plan", str(old)]
    if smoke:
        plan_options.append("--smoke")
    assert baseline.main(["plan", "--plan", str(destination), *plan_options]) == 0
    run_options = ["run", "--plan", str(destination), "--runtime-config", str(config_path), "--first-scenario",
                   "--checkpoint", str(base_path.parent / "checkpoint"), "--temporal", str(base_path.parent / "temporal")]
    return old, destination, baseline._read_plan(destination), run_options


@pytest.mark.parametrize(("smoke", "arm_only"), ((False, False), (True, False), (False, True)))
def test_direct_tom_paired_plan_run_and_summary(frozen_plan, monkeypatch, capsys, smoke, arm_only):
    from scripts import tom_gameplay_ablation as ablation
    from tests.tom.test_gameplay_ablation import _matrix

    old, destination, plan, run = _direct_tom_plan(frozen_plan, monkeypatch, smoke=smoke)
    if arm_only:
        run += ["--arm", "Direct-ToM"]
    original = old.read_bytes()
    parent = baseline._read_plan(old)
    assert plan["paired_plan_sha256"] == parent["plan_sha256"]
    scene = plan["scenarios"][0]
    if smoke:
        assert len(plan["scenarios"]) == 1 and scene["seed"] == baseline.DIRECT_TOM_SMOKE_SEED
        assert not any(item["seed"] == scene["seed"] for item in parent["scenarios"])
    else:
        assert plan["scenarios"] == parent["scenarios"]
        empty = _summary(destination, capsys)
        assert len(empty["tasks"]) == 420 and empty["paired"]["N"] == 0
        assert [task["arm"] for task in empty["tasks"][:4]] == ["NoToM", "Direct-ToM", "Direct-ToM", "NoToM"]
        assert [task["arm"] for task in empty["tasks"][210:212]] == ["Direct-ToM", "NoToM"]

    class ScenarioBackend(_GameplayBackend):
        def _vote(self, _prompt, legal):
            return next((seat for seat in legal if seat and scene["roles"][seat - 1] != "Werewolf"), 0)

    client = _GameplayClient(ScenarioBackend("Werewolf"))
    backend = OpenAICompatibleBackend(client=client, supports_json_schema=True)
    monkeypatch.setattr(baseline, "load_named_backends", Mock(return_value={"local_qwen": backend}))
    prefixes = []

    def predict(prefix, *, record_forward):
        prefixes.append(prefix.boundary_id)
        record_forward(0.25)
        return _matrix()

    def load(checkpoint, temporal, device):
        assert checkpoint.name == "checkpoint" and temporal.name == "temporal" and device == "cpu"
        assert client.calls
        control_paths = baseline._task_paths(destination.parent, scene, "NoToM")
        attempt, _ = baseline._task_paths(destination.parent, scene, "Direct-ToM")
        assert all(path.exists() is not arm_only for path in control_paths) and attempt.exists()
        return SimpleNamespace(identity=plan["direct_tom"]["predictor"], predict=predict)

    loader = Mock(side_effect=load)
    monkeypatch.setattr(ablation, "Epoch35GameplayPredictor", loader)
    assert baseline.main(run) == 0
    requests = len(client.calls)
    assert baseline.main(run) == 0
    assert len(client.calls) == requests and loader.call_count == 1
    assert len(prefixes) == len(set(prefixes)) > 0 and old.read_bytes() == original
    summary = _summary(destination, capsys)
    assert all(row["method"] == "react" and row["winner"] == "Werewolf" for row in summary["games"])
    stats = summary["arms"]["Direct-ToM"]
    assert stats["completed"] == stats["replay_passed"] == 1
    assert stats["cost"]["llm_backend_seconds"]["sum"] > 0
    evidence = stats["direct_tom"]["valid_games"]
    assert evidence["A"] == evidence["P"] == evidence["U"] == evidence["S"] == len(prefixes)
    assert evidence["forward_seconds"]["observed_sum"] == 0.25 * len(prefixes)
    if smoke:
        assert summary["smoke_verified"] and "paired" not in summary and len(summary["tasks"]) == 2
        assert evidence["by_phase"]["speech_pk"]["E"] == 0

        def fail_replay(_result):
            raise ValueError("scripted independent replay failure")

        monkeypatch.setattr(baseline.game_runtime, "replay_gameplay_history", fail_replay)
        without_votes = _summary(destination, capsys)
        assert not without_votes["smoke_verified"]
        assert all(arm["completed"] == 1 and arm["votes"]["games_with_vote_evidence"] == 0
                   for arm in without_votes["arms"].values())
    elif arm_only:
        assert summary["paired"]["N"] == 0 and len(summary["games"]) == 1
        assert summary["arms"]["NoToM"]["not_started"] == 210
        assert requests == stats["cost"]["model_requests"]["sum"]
        assert not any(destination.parent.glob("*/scenario-*-notom.json"))
    else:
        paired = summary["paired"]
        assert paired["N"] == paired["n11"] == 1 and paired["exact_mcnemar_p"] == 1
        assert paired["metrics"]["WWR"]["bootstrap_95_ci"] == [0.0, 0.0]
        assert paired["bootstrap_replicates"] == 100_000


def test_direct_tom_resume_preserves_explicit_failure_and_unfinished_claim(frozen_plan, monkeypatch, capsys):
    old, destination, plan, run = _direct_tom_plan(frozen_plan, monkeypatch)
    scene = plan["scenarios"][0]
    _, no_tom = _publish_sample(destination, plan, scene, "NoToM", _sample_result(
        plan, scene, "react", winner=None, replay_winner=None, terminal_reached=False, failure_type="ValueError",
    ))
    _, pending = _publish_sample(destination, plan, scene, "Direct-ToM")
    original = no_tom.read_bytes()
    assert baseline.main(run) == 0  # fixture 会拒绝任何非 Mock 的后端加载。
    summary = _summary(destination, capsys)
    assert no_tom.read_bytes() == original and not pending.exists()
    assert [task["status"] for task in summary["tasks"][:2]] == ["failed", "unknown"]
    evidence = summary["arms"]["Direct-ToM"]["direct_tom"]["all_observed_attempts"]
    assert evidence["E"] == evidence["A"] == evidence["P"] == evidence["U"] == evidence["S"] == "UNKNOWN"
    assert summary["paired"]["N"] == 0


def test_direct_tom_pair_keeps_valid_games_without_exposure_and_recomputes_vote_denominators():
    from werewolf import game_runtime

    def row(scene, arm, wins, eligible_rounds, votes):
        counts = dict.fromkeys(game_runtime._VOTE_FIELDS, 0)
        counts.update(wolf_eligible_rounds=eligible_rounds, wolf_votes_received=votes)
        return {"scenario_id": scene, "arm": arm, "winner": "Werewolf" if wins else "Villager",
                "vote_counts": counts, "E": 0, "U": 0}

    plan = {"scenarios": [{"scenario_id": scene} for scene in ("first", "second", "unpaired")]}
    paired = baseline._direct_tom_pairs(plan, [
        row("first", "NoToM", False, 2, 1), row("first", "Direct-ToM", True, 2, 2),
        row("second", "NoToM", True, 4, 0), row("second", "Direct-ToM", True, 2, 1),
        row("unpaired", "Direct-ToM", False, 2, 0),
    ])
    assert (paired["N"], paired["n00"], paired["n01"], paired["n10"], paired["n11"]) == (2, 0, 1, 0, 1)
    assert paired["metrics"]["WWR"]["delta"] == 0.5 and paired["metrics"]["WWR"]["bootstrap_95_ci"] == [0.0, 1.0]
    assert paired["metrics"]["VPR"]["delta"] == pytest.approx(3 / 4 - 1 / 6)
    assert paired["exact_mcnemar_p"] == 1


def test_direct_tom_arm_batches_skip_started_tasks(frozen_plan, monkeypatch, capsys):
    _, destination, plan, run = _direct_tom_plan(frozen_plan, monkeypatch)
    first, failed, pending, interrupted = plan["scenarios"][:4]
    by_seed = {scene["seed"]: scene for scene in plan["scenarios"]}
    backend = object()
    monkeypatch.setattr(baseline, "load_named_backends", Mock(return_value={"local_qwen": backend}))
    dispatched = []

    def evaluate(env, agents, roles, **kwargs):
        scene = by_seed[env.random_seed]
        dispatched.append(scene["scenario_id"])
        assert tuple(roles) == tuple(scene["roles"])
        assert tuple(agent.decision_method for agent in agents) == tuple(baseline._decision_methods(scene, "react"))
        assert kwargs.get("direct_wolf_speech_tom") and kwargs["direct_tom_game_id"].endswith("-direct-tom")
        attempt, result = baseline._task_paths(destination.parent, scene, "Direct-ToM")
        assert attempt.exists() and not result.exists()
        failure = {"failure_type": "KeyboardInterrupt", "terminal_reached": False, "winner": None, "replay_winner": None}
        return GameplayOnlyResult(**_sample_result(plan, scene, "react", **(failure if scene == interrupted else {})))

    monkeypatch.setattr(baseline.game_runtime, "eval", evaluate)
    run += ["--arm", "Direct-ToM"]
    assert baseline.main(run) == 0
    assert dispatched == [first["scenario_id"]]
    paths = baseline._task_paths(destination.parent, first, "Direct-ToM")
    paths += _publish_sample(destination, plan, failed, "Direct-ToM", _sample_result(
        plan, failed, "react", failure_type="ValueError", winner=None, replay_winner=None, terminal_reached=False,
    ))
    attempt, result = _publish_sample(destination, plan, pending, "Direct-ToM")
    original = {path: path.read_bytes() for path in (*paths, attempt)}
    batch = [arg for arg in run if arg != "--first-scenario"] + ["--batch"]
    assert baseline.main(batch + ["1"]) == 1
    assert dispatched == [first["scenario_id"], interrupted["scenario_id"]]
    assert baseline.main(batch + ["1"]) == 0 and baseline.main(batch + ["2"]) == 0
    assert len(dispatched) == len(set(dispatched)) == 208
    assert dispatched == [scene["scenario_id"] for scene in plan["scenarios"] if scene not in (failed, pending)]
    assert {path: path.read_bytes() for path in original} == original and not result.exists()
    assert not any(destination.parent.glob("*/scenario-*-notom.json"))
    stats = _summary(destination, capsys)["arms"]
    treated = stats["Direct-ToM"]
    assert (treated["completed"], treated["failed"], treated["unknown"]) == (207, 1, 2)
    assert stats["NoToM"]["not_started"] == 210 and stats["NoToM"]["started"] == 0


@pytest.mark.parametrize("source_status", ("compatible", "incompatible"))
def test_historical_react_audit_locks_valid_records_before_pairing(frozen_plan, mock_sdk, monkeypatch, capsys, source_status):
    old, destination, plan, _ = _direct_tom_plan(frozen_plan, monkeypatch)
    history = baseline._read_plan(old)
    base_path, config_path, _ = frozen_plan
    assert _run_first(base_path, config_path) == 0  # 取得原生完成及回放的脚本化局。
    valid, failed, pending, bad_identity, bad_replay, malformed = history["scenarios"][:6]
    _, native = baseline._task_paths(base_path.parent, valid, "react")
    game = json.loads(native.read_bytes())["gameplay"]
    _publish_sample(old, history, valid, "react", game)
    _publish_sample(old, history, failed, "react", _sample_result(
        history, failed, "react", failure_type="ValueError", failure_message="scripted failure",
        winner=None, replay_winner=None, terminal_reached=False,
    ))
    _publish_sample(old, history, pending, "react")
    _publish_sample(old, history, bad_identity, "react", _sample_result(history, bad_identity, "react", seed=-1))
    _publish_sample(old, history, bad_replay, "react", _sample_result(history, bad_replay, "react"))
    claim, broken = _publish_sample(old, history, malformed, "react")
    broken.write_text("{invalid", encoding="utf-8")
    original = {path: path.read_bytes() for path in old.parent.rglob("*.json")}
    monkeypatch.setattr(baseline, "_react_source_comparison", lambda *_: {"status": source_status})
    backend_loader = Mock(side_effect=AssertionError("审阅和汇总不可加载 Backend"))
    monkeypatch.setattr(baseline, "load_named_backends", backend_loader)
    audit = ["audit-react", "--plan", str(destination), "--historical-plan", str(old),
             "--runtime-config", str(config_path)]
    reference_path = destination.parent.parent / "historical_react_reference.json"
    capsys.readouterr()
    assert baseline.main(audit) == 0 and not reference_path.exists()
    preview = json.loads(capsys.readouterr().out)
    assert preview["summary"]["planned"] == 210 and preview["summary"]["WWR"] == {
        "numerator": 1, "denominator": 1, "value": 1.0,
    }
    assert preview["summary"]["votes"]["games_with_vote_evidence"] == 1
    included, = preview["included"]
    assert included["scenario_id"] == valid["scenario_id"] and included["gameplay_sha256"] == digest(game)
    assert included["cost"]["model_requests"] > 0 and included["result_sha256"]
    excluded = {row["scenario_id"]: row for row in preview["excluded"]}
    assert [excluded[scene["scenario_id"]]["status"] for scene in (failed, pending, bad_identity, bad_replay)] == [
        "failed", "unknown", "unknown", "completed",
    ]
    assert excluded[failed["scenario_id"]]["reason"] == "scripted failure"
    assert "no result" in excluded[pending["scenario_id"]]["reason"]
    assert "identity mismatch" in excluded[bad_identity["scenario_id"]]["reason"]
    assert "independent replay failed" in excluded[bad_replay["scenario_id"]]["reason"]
    assert excluded[malformed["scenario_id"]]["attempt_sha256"] == digest(json.loads(claim.read_bytes()))
    assert excluded[malformed["scenario_id"]]["status"] == "unknown"
    assert preview["historical_service"]["request_model_aliases"]
    assert preview["historical_service"]["model_manifest_sha256"] == "UNKNOWN"
    assert baseline.main(audit + ["--output", str(old.parent / reference_path.name)]) == 1
    assert baseline.main(audit + ["--output", str(reference_path)]) == 0
    reference_bytes = reference_path.read_bytes()
    assert baseline.main(audit + ["--output", str(reference_path)]) == 1
    assert reference_path.read_bytes() == reference_bytes

    _publish_sample(destination, plan, valid, "Direct-ToM", game)
    assert baseline.main(audit + ["--output", str(destination.parent / reference_path.name)]) == 1
    summary_args = ["summarize", "--plan", str(destination), "--historical-reference", str(reference_path)]
    capsys.readouterr()
    read_record = baseline._read_record

    def read_without_original(path):
        assert not path.is_relative_to(old.parent)
        return read_record(path)

    with monkeypatch.context() as patch:
        patch.setattr(baseline, "_read_record", read_without_original)
        assert baseline.main(summary_args) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["paired"]["N"] == 0 and summary["arms"]["NoToM"]["not_started"] == 210
    comparison = summary["historical_comparison"]
    assert comparison["direct_tom"]["completed"] == comparison["historical_react"]["completed"] == 1
    assert comparison["status"] == source_status
    paired = comparison["paired"]
    if source_status == "compatible":
        assert paired["N"] == paired["n11"] == 1 and paired["exact_mcnemar_p"] == 1
        assert paired["difference"] == "Direct-ToM - Historical ReAct"
        assert paired["bootstrap_replicates"] == 100_000 and paired["metrics"]["WWR"]["bootstrap_95_ci"] == [0.0, 0.0]
    else:
        assert paired is None
    assert {path: path.read_bytes() for path in old.parent.rglob("*.json")} == original
    assert not backend_loader.called

    reference = json.loads(reference_bytes)
    for field, wrong in (("seed", 999), ("roles", list(reversed(valid["roles"])))):
        wrong_ref = {**reference, "included": [{**included, field: wrong}]}
        del wrong_ref["reference_sha256"]
        wrong_ref["reference_sha256"] = digest(wrong_ref)
        wrong_path = reference_path.with_name(f"wrong-{field}.json")
        write_once(wrong_path, wrong_ref)
        assert baseline.main(summary_args[:-1] + [str(wrong_path)]) == 1


def test_historical_react_source_audits_cross_day_fix_using_archived_file_digests():
    def source(revision):
        def git(*args):
            return subprocess.check_output(["git", "--no-optional-locks", *args], cwd=RUNTIME_YAML.parents[2])

        return {"commit": git("rev-parse", revision).decode().strip(), "branch": "twd/mainline",
                "tracked_source_diff_sha256": sha256(b"").hexdigest(),
                "file_sha256": {path: sha256(git("show", f"{revision}:{path}")).hexdigest()
                                for path in baseline.game_runtime._GAMEPLAY_SOURCE_FILES}}

    fixed = baseline.REACT_CROSS_DAY_FIX
    target = source(baseline.REVIEWED_DIRECT_TOM_SOURCE)
    after = baseline._react_source_comparison(source(fixed), target)
    assert after["status"] == "compatible" and after["cross_day_fix"]["in_historical_git_ancestry"] is True
    before = baseline._react_source_comparison(source(f"{fixed}^"), target)
    assert before["status"] == "incompatible" and before["cross_day_fix"]["historical_validator"] == "reject_cross_day"
    modified = source(fixed)
    modified["file_sha256"]["werewolf/agents/llm_agent.py"] = "f" * 64
    unknown = baseline._react_source_comparison(modified, target)
    assert unknown["status"] == "review_required" and unknown["cross_day_fix"]["historical_validator"] == "UNKNOWN"
