"""按冻结的 Classic7 场景运行狼人阵营 Baseline，并汇总实际终局。"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import combinations
import json
from pathlib import Path
from random import Random
import subprocess
import sys

import yaml

from werewolf import game_runtime
from werewolf.artifact_io import sha256_bytes
from werewolf.artifact_io.canonical import _load_canonical_json, require_sha256
from werewolf.backends import load_named_backends
from werewolf.canonical_collection.public_history import PLAYER_IDS
from werewolf.game_runtime import UNKNOWN, count_game_votes, summarize_game_votes, summarize_tom_integration
from werewolf.interventions.gameplay import digest, safe_path, write_once
from werewolf.runtime_config import DECISION_METHODS, normalize_runtime_config


SEATS = tuple(range(1, 8))
WOLF_PAIRS = tuple(combinations(SEATS, 2))
ROLE_COUNTS = {"Werewolf": 2, "Seer": 1, "Witch": 1, "Villager": 3}
METHOD_NAMES = {"react": "react", "dpins:no": "belief", "dpins:llm": "llm-ins", "dpins:random": "random-ins"}
WOLF_PROFILE = "baseline-wolf"
BATCH_SIZE = 5 * len(WOLF_PAIRS)
MISSING = object()


def _method_order(index):
    offset = index % len(DECISION_METHODS)
    return list(DECISION_METHODS[offset:] + DECISION_METHODS[:offset])


def _seat_profiles(roles, default_profile):
    return [WOLF_PROFILE if role == "Werewolf" else default_profile for role in roles]


def _scenarios(default_profile):
    orders = []
    for first, second in WOLF_PAIRS:
        nonwolves = [seat for seat in SEATS if seat not in (first, second)]
        Random(f"werewolf-baseline:seats:{first}:{second}").shuffle(nonwolves)
        orders.append(nonwolves)

    scenarios = []
    for batch in (1, 2):
        for rotation in range(5):
            for index, (wolves, nonwolves) in enumerate(zip(WOLF_PAIRS, orders)):
                seed = 10000 + 10 * index + 5 * (batch - 1) + rotation
                roles = ["Villager"] * len(SEATS)
                for seat in wolves:
                    roles[seat - 1] = "Werewolf"
                roles[nonwolves[rotation] - 1] = "Seer"
                roles[nonwolves[(rotation + batch) % 5] - 1] = "Witch"
                scenarios.append({
                    "scenario_id": f"scenario-{seed}", "batch": batch, "seed": seed,
                    "roles": roles, "seat_profiles": _seat_profiles(roles, default_profile),
                    "method_order": _method_order(len(scenarios)),
                })
    return scenarios


def _validate_scenarios(scenarios, default_profile):
    if len(scenarios) != 2 * BATCH_SIZE:
        raise ValueError("expected 210 scenarios across two batches")
    assignments = Counter()
    for ordinal, scene in enumerate(scenarios):
        batch_index, within_batch = divmod(ordinal, BATCH_SIZE)
        rotation, pair_index = divmod(within_batch, len(WOLF_PAIRS))
        batch = batch_index + 1
        wolves = WOLF_PAIRS[pair_index]
        seed = 10000 + 10 * pair_index + 5 * batch_index + rotation
        roles = scene["roles"]
        if (
            (scene["scenario_id"], scene["batch"], scene["seed"]) != (f"scenario-{seed}", batch, seed)
            or Counter(roles) != ROLE_COUNTS
            or tuple(seat for seat, role in enumerate(roles, 1) if role == "Werewolf") != wolves
            or scene["seat_profiles"] != _seat_profiles(roles, default_profile)
            or scene["method_order"] != _method_order(ordinal)
        ):
            raise ValueError(f"invalid scenario at position {ordinal}")
        for seat, role in enumerate(roles, 1):
            assignments[batch, wolves, seat, role] += 1
    for batch in (1, 2):
        for wolves in WOLF_PAIRS:
            for seat in SEATS:
                expected = {"Werewolf": 5} if seat in wolves else {"Seer": 1, "Witch": 1, "Villager": 3}
                if any(assignments[batch, wolves, seat, role] != expected.get(role, 0) for role in ROLE_COUNTS):
                    raise ValueError(f"unbalanced roles for batch {batch}, wolves {wolves}, seat {seat}")


def _read_config(path):
    raw = path.read_bytes()
    config = normalize_runtime_config(yaml.safe_load(raw))
    profiles = config["agent_config"]["all_candidates"]
    if len(profiles) != 1 or "decision_method" in profiles[0]["model_params"]:
        raise ValueError("runtime YAML must supply one default GPTAgent profile")
    if profiles[0]["profile_name"] == WOLF_PROFILE:
        raise ValueError(f"{WOLF_PROFILE!r} is reserved for the tested wolf method")
    expected = {"n_player": 7, "n_role": 4, "n_werewolf": 2, "n_villager": 3, "n_seer": 1, "n_witch": 1}
    env = config["env_config"]
    if (any(env.get(key) != count for key, count in expected.items())
            or env.get("n_guard", 0) or env.get("n_hunter", 0)):
        raise ValueError("runtime YAML must use the Classic7 role composition")
    return config, sha256_bytes(raw)


def _source_identity():
    entry = Path(__file__).resolve()
    repo = entry.parents[1]
    identity = game_runtime._gameplay_source_identity(include_file_digests=True)
    identity["branch"] = subprocess.check_output(
        ["git", "--no-optional-locks", "symbolic-ref", "--short", "HEAD"], cwd=repo, text=True,
    ).strip()
    identity["file_sha256"][entry.relative_to(repo).as_posix()] = sha256_bytes(entry.read_bytes())
    return game_runtime._copy_gameplay_source_identity(identity)


def _read_record(path):
    try:
        return _load_canonical_json(safe_path(path).read_bytes())
    except FileNotFoundError:
        return MISSING


def _read_plan(path):
    plan = _read_record(path)
    if not isinstance(plan, dict):
        raise ValueError(f"no valid plan at {path}")
    return _validate_plan(plan)


def _validate_plan(plan):
    body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if plan.get("plan_sha256") != digest(body):
        raise ValueError("plan SHA-256 mismatch")
    _validate_scenarios(plan["scenarios"], plan["default_profile"])
    return plan


def _write_plan(args):
    config, config_sha256 = _read_config(args.runtime_config)
    default_profile = config["agent_config"]["all_candidates"][0]["profile_name"]
    scenarios = _scenarios(default_profile)
    _validate_scenarios(scenarios, default_profile)
    plan = {
        "default_profile": default_profile, "runtime_config_sha256": config_sha256,
        "source_identity": _source_identity(), "max_steps": 128, "request_timeout_seconds": 300, "sdk_retries": 0,
        "scenarios": scenarios,
    }
    plan["plan_sha256"] = digest(plan)
    write_once(args.plan, plan)
    _read_plan(args.plan)
    print(json.dumps({"plan": str(args.plan), "plan_sha256": plan["plan_sha256"], "scenarios": len(scenarios)}))


def _task_key(plan, scene, method):
    return {"plan_sha256": plan["plan_sha256"], "scenario_id": scene["scenario_id"], "method": method}


def _task_paths(output, scene, method):
    filename = f"{scene['scenario_id']}-{METHOD_NAMES[method]}.json"
    return output / "attempts" / filename, output / "results" / filename


def _decision_methods(scene, method):
    return [method if role == "Werewolf" else None for role in scene["roles"]]


def _game_status(game, plan, scene, method):
    expected = {
        "seed": scene["seed"], "roles": scene["roles"], "seat_profiles": scene["seat_profiles"],
        "decision_methods": _decision_methods(scene, method), "source_identity": plan["source_identity"],
        "request_timeout_seconds": plan["request_timeout_seconds"],
    }
    if not isinstance(game, dict) or any(game.get(key) != value for key, value in expected.items()):
        raise ValueError("gameplay result identity mismatch")
    terminal, winner, replay = game["terminal_reached"], game["winner"], game["replay_winner"]
    actions, steps = game["submitted_actions"], game["completed_steps"]
    if type(terminal) is not bool or not isinstance(actions, list) or type(steps) is not int or not (
        0 <= steps <= len(actions) <= plan["max_steps"]
    ):
        raise ValueError("invalid terminal or action count in gameplay result")
    if winner not in (None, "Werewolf", "Villager") or (winner is not None and not terminal):
        raise ValueError("winner requires a recognized native terminal")

    records = game["request_records"]
    if not isinstance(records, list) or any(not isinstance(record, dict) for record in records):
        raise ValueError("invalid gameplay request records")
    unknown_request = any(record.get("dispatched") is True and record.get("response_received") is not True
                          for record in records)
    failure = game["failure_type"]
    interrupted = {"KeyboardInterrupt", "InterruptedError"}.intersection((failure, game["failure_cause_type"]))
    if replay is not None:
        if not terminal or replay != winner or not 0 < steps == len(actions):
            raise ValueError("replay does not agree with the native terminal and actions")
        require_sha256(game["public_event_digest"], "public_event_digest")
    if unknown_request or interrupted:
        return "unknown"
    if failure is not None:
        return "failed"
    return "completed" if replay is not None else "unknown"


def _read_task(plan, scene, method, attempt_path, result_path):
    started = False
    try:
        task = _task_key(plan, scene, method)
        attempt = _read_record(attempt_path)
        if attempt is not MISSING:
            if not isinstance(attempt, dict) or attempt.get("task") != task:
                raise ValueError("attempt identity mismatch")
            if not isinstance(attempt.get("started_at_utc"), str):
                raise ValueError("attempt is missing its start time")
            started = True
        result = _read_record(result_path)
        if result is MISSING:
            return started, "unknown" if started else "not_started", None, None
        if not started or not isinstance(result, dict) or result.get("attempt_sha256") != digest(attempt):
            raise ValueError("result is missing a matching attempt")
        if result.get("task") != task:
            raise ValueError("result task identity mismatch")
        game = result["gameplay"]
        if result.get("gameplay_sha256") != digest(game):
            raise ValueError("gameplay result SHA-256 mismatch")
        return started, _game_status(game, plan, scene, method), game, None
    except (OSError, ValueError, KeyError, TypeError) as error:
        return started, "unknown", None, f"{result_path.name}: {error}"


def _run(args, plan):
    config, config_sha256 = _read_config(args.runtime_config)
    if config_sha256 != plan["runtime_config_sha256"] or _source_identity() != plan["source_identity"]:
        raise ValueError("runtime configuration or source has changed since the plan was frozen")
    scenes = plan["scenarios"][:1] if args.first_scenario else [
        scene for scene in plan["scenarios"] if scene["batch"] == args.batch
    ]
    default = config["agent_config"]["all_candidates"][0]
    backends = None
    for scene in scenes:
        for method in scene["method_order"]:
            attempt_path, result_path = _task_paths(args.plan.parent, scene, method)
            _, status, _, problem = _read_task(plan, scene, method, attempt_path, result_path)
            if problem is not None:
                raise ValueError(problem)
            if status != "not_started":
                continue
            if backends is None:
                backends = load_named_backends(config, env_file=None, max_retries=0)
            wolf = {**default, "profile_name": WOLF_PROFILE,
                    "model_params": {**default["model_params"], "decision_method": method}}
            game_config = {**config, "agent_config": {
                **config["agent_config"], "all_candidates": [default, wolf],
            }}
            env, agents, roles, profiles = game_runtime.build_runtime(
                game_config, None, roles=scene["roles"], random_seed=scene["seed"],
                seat_profiles=scene["seat_profiles"], backends=backends,
            )
            if (list(roles) != scene["roles"] or list(profiles) != scene["seat_profiles"]
                    or [agent.decision_method for agent in agents] != _decision_methods(scene, method)):
                raise ValueError("runtime seats do not match the frozen scenario and method")
            task = _task_key(plan, scene, method)
            attempt = {"task": task, "started_at_utc": datetime.now(timezone.utc).isoformat()}
            write_once(attempt_path, attempt)
            result = game_runtime.eval(
                env, agents, roles, gameplay_only=True, record_requests=True,
                max_steps=plan["max_steps"], request_timeout=plan["request_timeout_seconds"],
                source_identity=plan["source_identity"],
            )
            game = asdict(result)
            write_once(result_path, {
                "task": task, "attempt_sha256": digest(attempt),
                "gameplay_sha256": digest(game), "gameplay": game,
            })
            _, status, _, problem = _read_task(plan, scene, method, attempt_path, result_path)
            if problem is not None:
                raise ValueError(problem)
            print(json.dumps({**task, "status": status, "result": str(result_path)}), flush=True)
            if status == "unknown" or {game["failure_type"], game["failure_cause_type"]} & {
                "KeyboardInterrupt", "InterruptedError",
            }:
                return 1
    return 0


def _summarize(output, plan):
    metrics = ("elapsed_seconds", "model_requests", "input_tokens", "output_tokens", "total_tokens")
    counts = {
        method: dict.fromkeys(("planned", "not_started", "started", "native_terminal", "replay_passed",
                               "completed", "failed", "unknown", "wolf_wins"), 0)
        for method in DECISION_METHODS
    }
    totals = {method: Counter() for method in DECISION_METHODS}
    available = {method: Counter() for method in DECISION_METHODS}
    no_tom = dict.fromkeys(("A", "P", "S"), 0)
    games = []
    problems = []
    for scene in plan["scenarios"]:
        for method in scene["method_order"]:
            started, status, game, problem = _read_task(plan, scene, method, *_task_paths(output, scene, method))
            stats = counts[method]
            stats["planned"] += 1
            stats["started"] += started
            stats[status] += 1
            if problem is not None:
                problems.append(problem)
            if game is None:
                continue
            stats["native_terminal"] += game["terminal_reached"]
            stats["replay_passed"] += game["replay_winner"] is not None
            stats["wolf_wins"] += status == "completed" and game["winner"] == "Werewolf"
            if status == "completed":
                row = {"scenario_id": scene["scenario_id"], "seed": scene["seed"], "method": method,
                       "winner": game["winner"], "public_event_digest": game["public_event_digest"],
                       "vote_counts": UNKNOWN, "tom_integration": no_tom}
                try:
                    history, _ = game_runtime.replay_gameplay_history(game_runtime.GameplayOnlyResult(**game))
                    row["vote_counts"] = count_game_votes(history, dict(zip(PLAYER_IDS, game["roles"], strict=True)))
                except (ValueError, TypeError, KeyError) as error:
                    row["vote_evidence_error"] = str(error)
                games.append(row)
            usage = game.get("token_usage")
            values = {"elapsed_seconds": game.get("elapsed_seconds"), "model_requests": game.get("model_requests")}
            if isinstance(usage, dict):
                values.update((field, usage.get(field)) for field in metrics[2:])
            for field, value in values.items():
                if type(value) in (int, float) and value >= 0:
                    totals[method][field] += value
                    available[method][field] += 1
    for method, stats in counts.items():
        completed = stats["completed"]
        stats["native_terminal_rate"] = stats["native_terminal"] / stats["planned"]
        stats["valid_result_coverage"] = completed / stats["planned"]
        stats["conditional_wolf_win_rate"] = stats["wolf_wins"] / completed if completed else None
        stats["cost"] = {
            field: {"sum": totals[method][field] if available[method][field] else None,
                    "games_available": available[method][field]} for field in metrics
        }
        stats["votes"] = summarize_game_votes([game["vote_counts"] for game in games if game["method"] == method])
        stats["tom_integration"] = {
            "complete_games": summarize_tom_integration([no_tom] * completed, applicable=False),
            "all_observed_attempts": summarize_tom_integration([no_tom] * stats["started"], applicable=False),
        }
    return {"plan_sha256": plan["plan_sha256"], "methods": counts, "games": games, "record_errors": problems}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan_parser = commands.add_parser("plan", help="生成并冻结 210 个场景")
    run_parser = commands.add_parser("run", help="运行尚未开始的任务")
    summary_parser = commands.add_parser("summarize", help="只读汇总全部计划任务")
    for command in (plan_parser, run_parser, summary_parser):
        command.add_argument("--plan", type=Path, required=True)
    for command in (plan_parser, run_parser):
        command.add_argument("--runtime-config", type=Path, required=True)
    selection = run_parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--first-scenario", action="store_true")
    selection.add_argument("--batch", type=int, choices=(1, 2))
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            _write_plan(args)
            return 0
        plan = _read_plan(args.plan)
        if args.command == "run":
            return _run(args, plan)
        print(json.dumps(_summarize(args.plan.parent, plan), ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError, yaml.YAMLError) as error:
        print(f"werewolf baseline: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
