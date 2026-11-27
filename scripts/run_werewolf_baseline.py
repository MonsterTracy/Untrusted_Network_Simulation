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
from werewolf.game_runtime import UNKNOWN, count_game_votes, gameplay_ratio, summarize_game_votes, summarize_tom_integration
from werewolf.interventions.gameplay import digest, safe_path, write_once
from werewolf.runtime_config import DECISION_METHODS, MULTIMIND_METHOD, normalize_runtime_config


SEATS = tuple(range(1, 8))
WOLF_PAIRS = tuple(combinations(SEATS, 2))
ROLE_COUNTS = {"Werewolf": 2, "Seer": 1, "Witch": 1, "Villager": 3}
METHOD_NAMES = {"react": "react", "dpins:no": "belief", "dpins:llm": "llm-ins", "dpins:random": "random-ins",
                MULTIMIND_METHOD: MULTIMIND_METHOD}
DIRECT_TOM_EXPERIMENT = "paper-tom-gameplay-ablation-direct"
DIRECT_TOM_ARMS = ("NoToM", "Direct-ToM")
DIRECT_TOM_SMOKE_SEED = 17
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


def _source_identity(*, multimind=False, direct_tom=False):
    entry = Path(__file__).resolve()
    repo = entry.parents[1]
    identity = game_runtime._gameplay_source_identity(include_file_digests=True, multimind=multimind)
    identity["branch"] = subprocess.check_output(
        ["git", "--no-optional-locks", "symbolic-ref", "--short", "HEAD"], cwd=repo, text=True,
    ).strip()
    identity["file_sha256"][entry.relative_to(repo).as_posix()] = sha256_bytes(entry.read_bytes())
    if direct_tom:
        for name in ("scripts/tom_gameplay_ablation.py", "scripts/analyze_tom_gameplay_ablation.py",
                     "werewolf/structured_history.py", "werewolf/tom/dataset.py", "werewolf/tom/evaluation.py",
                     "werewolf/tom/final_evaluation.py", "werewolf/tom/makto.py", "werewolf/tom/qwen3_large.py",
                     "werewolf/tom/temporal.py"):
            identity["file_sha256"][name] = sha256_bytes((repo / name).read_bytes())
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


def _direct_tom_smoke_scene(default_profile):
    return {**_scenarios(default_profile)[0], "scenario_id": f"scenario-{DIRECT_TOM_SMOKE_SEED}",
            "seed": DIRECT_TOM_SMOKE_SEED}


def _validate_plan(plan):
    body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if plan.get("plan_sha256") != digest(body):
        raise ValueError("plan SHA-256 mismatch")
    if "direct_tom" in plan:
        from scripts.tom_gameplay_ablation import epoch35_identity
        from scripts.tom_gameplay_collection import BASELINE_PLAN_SHA256

        direct = plan["direct_tom"]
        if ({"methods", "reasoner"} & plan.keys() or plan.get("sdk_retries") != 0
                or plan.get("paired_plan_sha256") != BASELINE_PLAN_SHA256
                or direct["experiment_id"] != DIRECT_TOM_EXPERIMENT or type(direct["smoke"]) is not bool
                or direct["predictor"] != epoch35_identity(direct["predictor"]["device"])):
            raise ValueError("Direct-ToM plan differs from the frozen experiment inputs")
        if direct["smoke"]:
            if plan["scenarios"] != [_direct_tom_smoke_scene(plan["default_profile"])]:
                raise ValueError("Direct-ToM Smoke requires its independent fixed scenario")
        else:
            _validate_scenarios(plan["scenarios"], plan["default_profile"])
        return plan
    _validate_scenarios(plan["scenarios"], plan["default_profile"])
    if {"methods", "paired_plan_sha256", "reasoner"} & plan.keys():
        if plan.get("methods") != [MULTIMIND_METHOD] or plan.get("sdk_retries") != 0:
            raise ValueError("paired plan requires only MultiMind and zero SDK retries")
        require_sha256(plan.get("paired_plan_sha256"), "paired plan digest")
        reasoner = plan.get("reasoner")
        if (not isinstance(reasoner, dict) or set(reasoner) != {"checkpoint_sha256", "device"}
                or not isinstance(reasoner["device"], str) or not reasoner["device"]):
            raise ValueError("MultiMind plan requires a pinned checkpoint and explicit device")
        require_sha256(reasoner["checkpoint_sha256"], "MultiMind checkpoint digest")
    return plan


def _read_paired_plan(path, output, config_sha256, default_profile):
    source_root, output_root = safe_path(path).parent, safe_path(output).parent
    if source_root.is_relative_to(output_root) or output_root.is_relative_to(source_root):
        raise ValueError("paired and historical plan output roots must be disjoint")
    parent = _read_plan(path)
    if {"methods", "direct_tom"} & parent.keys():
        raise ValueError("--paired-plan must be the frozen four-method plan")
    if (parent["runtime_config_sha256"], parent["default_profile"]) != (config_sha256, default_profile):
        raise ValueError("the paired plan's exact runtime YAML and default profile are required")
    return parent


def _write_plan(args):
    config, config_sha256 = _read_config(args.runtime_config)
    default_profile = config["agent_config"]["all_candidates"][0]["profile_name"]
    checkpoint_options = (args.reasoner_checkpoint, args.reasoner_checkpoint_digest, args.reasoner_device)
    if args.direct_tom:
        from scripts.tom_gameplay_ablation import epoch35_identity
        from scripts.tom_gameplay_collection import BASELINE_PLAN_SHA256

        if not args.paired_plan or any(checkpoint_options):
            raise ValueError("Direct-ToM requires --paired-plan without MultiMind Reasoner settings")
        parent = _read_paired_plan(args.paired_plan, args.plan, config_sha256, default_profile)
        if parent["plan_sha256"] != BASELINE_PLAN_SHA256:
            raise ValueError("Direct-ToM requires the exact frozen four-method plan")
        scenarios = parent["scenarios"]
        if args.smoke:
            if any(scene["seed"] == DIRECT_TOM_SMOKE_SEED for scene in scenarios):
                raise ValueError("Direct-ToM Smoke seed overlaps the formal plan")
            scenarios = [_direct_tom_smoke_scene(default_profile)]
        plan = {**parent, "scenarios": scenarios, "source_identity": _source_identity(direct_tom=True),
                "paired_plan_sha256": parent["plan_sha256"], "direct_tom": {
                    "experiment_id": DIRECT_TOM_EXPERIMENT, "smoke": args.smoke,
                    "predictor": epoch35_identity(args.predictor_device or "cpu"),
                }}
        del plan["plan_sha256"]
    elif args.smoke or args.predictor_device is not None:
        raise ValueError("--smoke and --predictor-device require --direct-tom")
    elif args.method == MULTIMIND_METHOD:
        from werewolf.tom.multimind import multimind_device, open_multimind_checkpoint

        if not args.paired_plan or not all(checkpoint_options):
            raise ValueError("MultiMind plan requires --paired-plan, checkpoint path and digest, and device")
        parent = _read_paired_plan(args.paired_plan, args.plan, config_sha256, default_profile)
        checkpoint = open_multimind_checkpoint(safe_path(args.reasoner_checkpoint), args.reasoner_checkpoint_digest)
        scenarios = parent["scenarios"]
        plan = {**parent, "source_identity": _source_identity(multimind=True), "methods": [MULTIMIND_METHOD],
                "paired_plan_sha256": parent["plan_sha256"],
                "reasoner": {"checkpoint_sha256": checkpoint.manifest_digest,
                             "device": str(multimind_device(args.reasoner_device))}}
        del plan["plan_sha256"]
    else:
        if args.paired_plan or any(checkpoint_options):
            raise ValueError("paired plan requires --method multimind or --direct-tom")
        scenarios = _scenarios(default_profile)
        plan = {
            "default_profile": default_profile, "runtime_config_sha256": config_sha256,
            "source_identity": _source_identity(), "max_steps": 128, "request_timeout_seconds": 300, "sdk_retries": 0,
            "scenarios": scenarios,
        }
    plan["plan_sha256"] = digest(plan)
    _validate_plan(plan)
    write_once(args.plan, plan)
    _read_plan(args.plan)
    print(json.dumps({"plan": str(args.plan), "plan_sha256": plan["plan_sha256"], "scenarios": len(scenarios)}))


def _task_order(plan, scene, ordinal):
    if "direct_tom" in plan:
        return DIRECT_TOM_ARMS if ordinal % 2 == 0 else DIRECT_TOM_ARMS[::-1]
    return plan.get("methods", scene["method_order"])


def _task_key(plan, scene, method):
    key = {"plan_sha256": plan["plan_sha256"], "scenario_id": scene["scenario_id"], "method": method}
    if "direct_tom" in plan:
        key.update(method="react", arm=method)
    return key


def _task_paths(output, scene, method):
    name = method.lower() if method in DIRECT_TOM_ARMS else METHOD_NAMES[method]
    filename = f"{scene['scenario_id']}-{name}.json"
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
        return started, _game_status(game, plan, scene, task["method"]), game, None
    except (OSError, ValueError, KeyError, TypeError) as error:
        return started, "unknown", None, f"{result_path.name}: {error}"


def _run(args, plan):
    config, config_sha256 = _read_config(args.runtime_config)
    multimind = "reasoner" in plan
    direct = plan.get("direct_tom")
    if direct is not None:
        source = _source_identity(direct_tom=True)
        if args.checkpoint is None or args.temporal is None:
            raise ValueError("Direct-ToM run requires --checkpoint and --temporal")
        if direct["smoke"] and not args.first_scenario:
            raise ValueError("Direct-ToM Smoke runs with --first-scenario")
    else:
        source = _source_identity(multimind=True) if multimind else _source_identity()
        if args.checkpoint is not None or args.temporal is not None:
            raise ValueError("--checkpoint and --temporal require a Direct-ToM plan")
    if config_sha256 != plan["runtime_config_sha256"] or source != plan["source_identity"]:
        raise ValueError("runtime configuration or source has changed since the plan was frozen")
    if multimind and (args.reasoner_checkpoint is None or args.reasoner_device != plan["reasoner"]["device"]):
        raise ValueError("MultiMind run requires an explicit checkpoint path and the frozen --reasoner-device")
    if not multimind and (args.reasoner_checkpoint is not None or args.reasoner_device is not None):
        raise ValueError("Reasoner settings require a MultiMind plan")
    scenes = list(enumerate(plan["scenarios"]))
    scenes = scenes[:1] if args.first_scenario else [
        (ordinal, scene) for ordinal, scene in scenes if scene["batch"] == args.batch
    ]
    default = config["agent_config"]["all_candidates"][0]
    backends = None
    reasoner = None
    epoch35 = None

    def predict(prefix, observation, evidence):
        from scripts.tom_gameplay_ablation import Epoch35GameplayPredictor, direct_tom_payload

        nonlocal epoch35
        if epoch35 is None:
            model = Epoch35GameplayPredictor(safe_path(args.checkpoint), safe_path(args.temporal),
                                             direct["predictor"]["device"])
            if model.identity != direct["predictor"]:
                raise ValueError("Epoch 35 predictor differs from the frozen Direct-ToM plan")
            epoch35 = model
        return direct_tom_payload(prefix, observation, predictor=epoch35, evidence=evidence)

    for ordinal, scene in scenes:
        for label in _task_order(plan, scene, ordinal):
            attempt_path, result_path = _task_paths(args.plan.parent, scene, label)
            _, status, _, problem = _read_task(plan, scene, label, attempt_path, result_path)
            if problem is not None:
                raise ValueError(problem)
            if status != "not_started":
                continue
            if multimind and reasoner is None:
                from werewolf.tom.multimind import load_multimind_checkpoint

                try:
                    reasoner = load_multimind_checkpoint(
                        safe_path(args.reasoner_checkpoint), plan["reasoner"]["checkpoint_sha256"], device=args.reasoner_device,
                    )
                except RuntimeError as error:
                    raise ValueError(f"MultiMind Reasoner could not be loaded on {args.reasoner_device}: {error}") from error
            if backends is None:
                backends = load_named_backends(config, env_file=None, max_retries=0)
            task = _task_key(plan, scene, label)
            method = task["method"]
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
            options = {"multimind_reasoner": reasoner} if multimind else {}
            if direct is not None:
                options["direct_tom_game_id"] = f"{scene['scenario_id']}-{label.lower()}"
                if label == "Direct-ToM":
                    options["direct_wolf_speech_tom"] = predict
            attempt = {"task": task, "started_at_utc": datetime.now(timezone.utc).isoformat()}
            write_once(attempt_path, attempt)
            result = game_runtime.eval(
                env, agents, roles, gameplay_only=True, record_requests=True,
                max_steps=plan["max_steps"], request_timeout=plan["request_timeout_seconds"],
                source_identity=plan["source_identity"],
                **options,
            )
            game = asdict(result)
            write_once(result_path, {
                "task": task, "attempt_sha256": digest(attempt),
                "gameplay_sha256": digest(game), "gameplay": game,
            })
            _, status, _, problem = _read_task(plan, scene, label, attempt_path, result_path)
            if problem is not None:
                raise ValueError(problem)
            print(json.dumps({**task, "status": status, "result": str(result_path)}), flush=True)
            if status == "unknown" or {game["failure_type"], game["failure_cause_type"]} & {
                "KeyboardInterrupt", "InterruptedError",
            }:
                return 1
    return 0


def _multimind_evidence(game):
    """每个真实发言边界最多计一次，Actor 回执按同局、同座位和同一步匹配。"""
    unknown = dict.fromkeys(("A", "P", "S"), UNKNOWN)
    if game is None or not isinstance(game.get("speech_results"), list):
        return unknown, [], True
    responses = {}
    for request in game["request_records"]:
        if (request.get("kind") == "speech_realization" and request.get("method") == MULTIMIND_METHOD
                and request.get("game_seed") == game["seed"] and request.get("dispatched") is True):
            step, seat, phase = request.get("step"), request.get("seat"), request.get("phase")
            if type(step) is int and type(seat) is int and isinstance(phase, str):
                responses.setdefault((step, seat, phase), []).append(request)

    boundaries, rows = [], []
    recorded_steps = set()
    missing = False
    for speech in game["speech_results"]:
        if (not isinstance(speech, dict) or type(speech.get("step")) is not int or speech["step"] < 0
                or speech.get("kind") not in ("speech", "speech_pk")):
            missing = True
            continue
        step, seat = speech["step"], speech.get("seat")
        if step in recorded_steps:
            missing = True
            continue
        recorded_steps.add(step)
        if type(seat) is not int or not 1 <= seat <= len(SEATS):
            missing = True
            continue
        if game["decision_methods"][seat - 1] != MULTIMIND_METHOD:
            continue
        detail = speech.get("multimind")
        if not isinstance(detail, dict):
            missing = True
            continue
        rows.append((speech["kind"], detail))
        attempted = detail.get("planner_attempted")
        predicted = detail.get("reasoner_prediction_succeeded")
        delivered = detail.get("plan_delivered_to_actor")
        confirmed = UNKNOWN
        if delivered is False:
            confirmed = 0
        elif delivered is True and attempted is True and predicted is True and detail.get("planner_completed") is True:
            phase = f"{detail.get('day')}_day_{speech['kind']}"
            matching = responses.get((step, seat, phase), [])
            if any(r.get("response_received") is True
                   and r.get("status") in {"response_received", "invalid_response"} for r in matching):
                confirmed = 1
            elif matching and all(r.get("response_received") is True for r in matching):
                confirmed = 0
        boundaries.append({"A": int(attempted) if type(attempted) is bool else UNKNOWN,
                           "P": int(predicted) if type(predicted) is bool else UNKNOWN, "S": confirmed})
    if any(kind in game_runtime.PUBLIC_SPEECH_RUNTIME_PHASES and index not in recorded_steps
           for index, (kind, _) in enumerate(game["submitted_actions"])):
        missing = True
    if missing:
        boundaries.append(unknown)
    aggregate = summarize_tom_integration(boundaries)
    counts = {field: aggregate[field] for field in ("A", "P", "S")}
    counts.update({f"{field}_verified": value for field, value in aggregate["verified_minimum"].items()})
    return counts, rows, missing


def _summarize_multimind(evidence):
    rows = [row for _, speeches, _ in evidence for row in speeches]
    phases = {}
    for name in ("speech", "speech_pk", "last_same_phase_speaker"):
        selected = [detail for kind, detail in rows if kind == name or detail.get(name) is True]
        phases[name] = {"boundaries": len(selected),
                        "planning_attempts": sum(detail.get("planner_attempted") is True for detail in selected),
                        "alignment_available": sum(isinstance(detail.get("alignment"), dict) for detail in selected)}
    costs = {}
    for field in ("completed_simulations", "candidate_score_calls", "reasoner_forward_calls", "mcts_elapsed_seconds"):
        values = [detail[field] for _, detail in rows if type(detail.get(field)) in (int, float)]
        costs[field] = {"sum": sum(values) if values else None, "boundaries_available": len(values),
                        "unknown_boundaries": len(rows) - len(values)}
    alignment = {}
    for field in ("exact_match", "plan_precision", "plan_recall", "order_match", "stop_compliance"):
        values = [record[field] for _, detail in rows if isinstance(record := detail.get("alignment"), dict)
                  and type(record.get(field)) in (bool, int, float)]
        alignment[field] = gameplay_ratio(sum(values), len(values))
    return {
        "missing_game_evidence": sum(missing for _, _, missing in evidence),
        "observed_speech_boundaries": len(rows),
        "planning_attempts": sum(detail.get("planner_attempted") is True for _, detail in rows),
        "completed_plans": sum(detail.get("planner_completed") is True for _, detail in rows),
        "actor_plan_handoffs": sum(detail.get("plan_delivered_to_actor") is True for _, detail in rows),
        "actor_confirmed_boundaries": sum(counts.get("S_verified", 0) for counts, _, _ in evidence),
        "alignment_available": sum(isinstance(detail.get("alignment"), dict) for _, detail in rows),
        "phase_support": phases, "cost": costs, "alignment": alignment,
    }


def _direct_tom_pairs(plan, games):
    from scripts.analyze_tom_gameplay_ablation import (
        BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, _exact_mcnemar, _paired_ratio_effects,
    )

    by_scene = {(row["scenario_id"], row["arm"]): row for row in games}
    rows = []
    for scene in plan["scenarios"]:
        controls, treated = (by_scene.get((scene["scenario_id"], arm)) for arm in DIRECT_TOM_ARMS)
        if controls is not None and treated is not None:
            rows.append((controls, treated))
    outcomes = Counter((control["winner"] == "Werewolf", treated["winner"] == "Werewolf")
                       for control, treated in rows)
    n00, n01, n10, n11 = (outcomes[a, b] for a, b in ((0, 0), (0, 1), (1, 0), (1, 1)))
    metrics = _paired_ratio_effects(rows)
    del metrics["coverage"]  # 接入统计由逐臂 PRE 证据汇总。
    return {"N": len(rows), "n00": n00, "n01": n01, "n10": n10, "n11": n11,
            "difference": "Direct-ToM - NoToM", "metrics": metrics,
            "exact_mcnemar_p": _exact_mcnemar(n01, n10) if rows else None,
            "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_generator": "NumPy PCG64"}


def _summarize(output, plan):
    metrics = ("elapsed_seconds", "model_requests", "input_tokens", "output_tokens", "total_tokens")
    direct = plan.get("direct_tom")
    methods = DIRECT_TOM_ARMS if direct is not None else plan.get("methods", DECISION_METHODS)
    if direct is not None:
        from scripts.tom_gameplay_ablation import direct_tom_coverage

        metrics += ("llm_backend_seconds", "retries")
        direct_attempts = {arm: [] for arm in DIRECT_TOM_ARMS}
        direct_completed = {arm: [] for arm in DIRECT_TOM_ARMS}
        tasks = []
    counts = {
        method: dict.fromkeys(("planned", "not_started", "started", "native_terminal", "replay_passed",
                               "completed", "failed", "unknown", "wolf_wins"), 0)
        for method in methods
    }
    totals = {method: Counter() for method in methods}
    available = {method: Counter() for method in methods}
    no_tom = dict.fromkeys(("A", "P", "S"), 0)
    multimind_attempts, multimind_completed = [], []
    games = []
    problems = []
    for ordinal, scene in enumerate(plan["scenarios"]):
        for method in _task_order(plan, scene, ordinal):
            started, status, game, problem = _read_task(plan, scene, method, *_task_paths(output, scene, method))
            stats = counts[method]
            stats["planned"] += 1
            stats["started"] += started
            stats[status] += 1
            if problem is not None:
                problems.append(problem)
            if direct is not None:
                task = {**_task_key(plan, scene, method), "status": status, "started": started}
                if game is not None:
                    task.update({key: game[key] for key in ("failure_type", "failure_stage", "failure_phase")})
                tasks.append(task)
                if started or status == "unknown":
                    direct_attempts[method].append(game)
                if status == "completed":
                    direct_completed[method].append(game)
            tom = no_tom
            if method == MULTIMIND_METHOD and (started or status == "unknown"):
                evidence = _multimind_evidence(game)
                multimind_attempts.append(evidence)
                tom = evidence[0]
                if status == "completed":
                    multimind_completed.append(tom)
            if game is None:
                continue
            stats["native_terminal"] += game["terminal_reached"]
            stats["replay_passed"] += game["replay_winner"] is not None
            stats["wolf_wins"] += status == "completed" and game["winner"] == "Werewolf"
            if status == "completed":
                row = {"scenario_id": scene["scenario_id"], "seed": scene["seed"], "method": method,
                       "winner": game["winner"], "public_event_digest": game["public_event_digest"],
                       "vote_counts": UNKNOWN, "tom_integration": tom}
                if direct is not None:
                    row.update(method="react", arm=method)
                    del row["tom_integration"]
                try:
                    history, _ = game_runtime.replay_gameplay_history(game_runtime.GameplayOnlyResult(**game))
                    row["vote_counts"] = count_game_votes(history, dict(zip(PLAYER_IDS, game["roles"], strict=True)))
                except (ValueError, TypeError, KeyError) as error:
                    row["vote_evidence_error"] = str(error)
                games.append(row)
            usage = game.get("token_usage")
            values = {"elapsed_seconds": game.get("elapsed_seconds"), "model_requests": game.get("model_requests")}
            if isinstance(usage, dict):
                values.update((field, usage.get(field)) for field in metrics[2:5])
            if direct is not None:
                durations = [request.get("elapsed_seconds") for request in game["request_records"]]
                if all(type(seconds) in (int, float) for seconds in durations) and game["model_requests"] == len(durations):
                    values["llm_backend_seconds"] = sum(durations)
                values["retries"] = game["retries"]
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
        stats["votes"] = summarize_game_votes([game["vote_counts"] for game in games
                                               if game.get("arm", game["method"]) == method])
        if direct is not None:
            stats["WWR"] = gameplay_ratio(stats["wolf_wins"], completed)
            stats["direct_tom"] = {
                "valid_games": direct_tom_coverage(direct_completed[method], applicable=method == "Direct-ToM"),
                "all_observed_attempts": direct_tom_coverage(direct_attempts[method], applicable=method == "Direct-ToM"),
            }
        elif method == MULTIMIND_METHOD:
            stats["tom_integration"] = {
                "complete_games": summarize_tom_integration(multimind_completed),
                "all_observed_attempts": summarize_tom_integration([counts for counts, _, _ in multimind_attempts]),
            }
            stats["multimind"] = _summarize_multimind(multimind_attempts)
        else:
            stats["tom_integration"] = {
                "complete_games": summarize_tom_integration([no_tom] * completed, applicable=False),
                "all_observed_attempts": summarize_tom_integration([no_tom] * stats["started"], applicable=False),
            }
    if direct is not None:
        summary = {"plan_sha256": plan["plan_sha256"], "paired_plan_sha256": plan["paired_plan_sha256"],
                   "direct_tom": direct, "arms": counts, "games": games, "tasks": tasks, "record_errors": problems}
        if direct["smoke"]:
            control, treated = (counts[arm]["direct_tom"]["valid_games"] for arm in DIRECT_TOM_ARMS)
            summary["smoke_verified"] = (all(counts[arm]["completed"] == counts[arm]["votes"]["games_with_vote_evidence"] == 1
                                             for arm in DIRECT_TOM_ARMS)
                                         and type(treated["E"]) is int and treated["E"] > 0
                                         and all(treated[field] == treated["E"] for field in (
                                             "A", "P", "S", "U", "response_and_submitted_action_decisions"))
                                         and all(control[field] == 0 for field in ("A", "P", "S", "U"))
                                         and control["unmatched_receipts"] == treated["unmatched_receipts"] == 0)
        else:
            summary["paired"] = _direct_tom_pairs(plan, games)
        return summary
    summary = {"plan_sha256": plan["plan_sha256"], "methods": counts, "games": games, "record_errors": problems}
    if "reasoner" in plan:
        summary.update(paired_plan_sha256=plan["paired_plan_sha256"], reasoner=plan["reasoner"])
    return summary


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
        command.add_argument("--reasoner-checkpoint", type=Path)
        command.add_argument("--reasoner-device")
    mode = plan_parser.add_mutually_exclusive_group()
    mode.add_argument("--method", choices=(MULTIMIND_METHOD,))
    mode.add_argument("--direct-tom", action="store_true")
    plan_parser.add_argument("--paired-plan", type=Path)
    plan_parser.add_argument("--reasoner-checkpoint-digest")
    plan_parser.add_argument("--predictor-device", choices=("cpu", "cuda"))
    plan_parser.add_argument("--smoke", action="store_true")
    run_parser.add_argument("--checkpoint", type=Path, help="Direct-ToM 的冻结 Epoch 35 Checkpoint")
    run_parser.add_argument("--temporal", type=Path, help="Direct-ToM 的冻结 Temporal")
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
