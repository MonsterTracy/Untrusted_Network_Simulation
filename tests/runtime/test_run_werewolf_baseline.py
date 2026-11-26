from collections import Counter
from dataclasses import asdict
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

from scripts import run_werewolf_baseline as baseline
from tests.canonical_collection.test_public_history import _event, _phase
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
        "decision_methods": (method, method, None, None, None, None, None),
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
