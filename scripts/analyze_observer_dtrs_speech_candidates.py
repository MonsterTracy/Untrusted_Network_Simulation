"""在固定 Development PRE 上检查候选生成质量。"""

import argparse
from collections import Counter
from dataclasses import asdict
from itertools import product
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from scripts import fit_observer_dtrs_report_event as frozen
from scripts.analyze_observer_dtrs_team_event import PRE_IDENTITY, SELECTION_FOLDS, _frozen_q
from werewolf.agents.gpt_agent import GPTAgent
from werewolf.agents.llm_agent import GameplayActionValidationError, GameplaySpeechQualityError
from werewolf.artifact_io import canonical_json_bytes, open_artifact_envelope, read_artifact_file, sha256_bytes
from werewolf.backends import BackendError, load_named_backends
from werewolf.backends.openai_compatible import model_request_context
from werewolf.development_publication import open_role_sidecar
from werewolf.observer_dtrs.context import SPEECH_PHASES
from werewolf.observer_dtrs.probabilities import ProbabilityMatrix
from werewolf.observer_dtrs.speech_candidates import SPEECH_CANDIDATE_PROMPT, build_candidate_context
from werewolf.runtime_config import normalize_runtime_config
from werewolf.strategy.counterfactual_tom import derive_public_phase_speaker_order
from werewolf.tom.development_input import open_development_input


POSITIONS = ("discussion_first", "discussion_middle", "discussion_last", "pk_nonlast", "pk_last")
PER_STRATUM = 3
PROFILE = "local_qwen"
MODEL = "qwen35-9b"


def pre_key(row):
    return row["game_id"], row["boundary_id"]


def speech_window(pre, team):
    order = derive_public_phase_speaker_order(pre)
    index = order.index(pre.current_speaker)
    observers = tuple(seat for seat in pre.alive_observer_ids if seat not in team)
    replyable = tuple(seat for seat in order[index + 1:] if seat in observers)
    return order, index, observers, replyable


def opportunity(pre, team, *, fold, attempt_id):
    order, index, _, replyable = speech_window(pre, team)
    last = index == len(order) - 1
    if pre.public_temporal_state.phase.value == "pk_discussion":
        position = "pk_last" if last else "pk_nonlast"
    elif index == 0:
        position = "discussion_first"
    else:
        position = "discussion_last" if last else "discussion_middle"
    return {
        **{name: getattr(pre, name) for name in PRE_IDENTITY},
        "attempt_id": attempt_id, "fold": fold, "position": position,
        "alive_wolves": sum(seat in pre.alive_observer_ids for seat in team),
        "replyable_observer_count": len(replyable),
    }


def _select_stratum(rows, publication_digest):
    def rank(row):
        identity = [publication_digest, row["fold"], row["attempt_id"], *(row[key] for key in PRE_IDENTITY)]
        return sha256_bytes(canonical_json_bytes(identity)), pre_key(row)

    ranked = sorted(rows, key=rank)
    buckets = [min(row["replyable_observer_count"], 2) for row in ranked]
    representatives = [{}, {}, {}]
    for index, (row, bucket) in enumerate(zip(ranked, buckets, strict=True)):
        group = representatives[bucket]
        if len(group) < PER_STRATUM:
            group.setdefault(row["game_id"], index)

    # 每类最前的三个不同游戏已足够，任一代表至多需避开另外两个游戏。
    choices = product(*((None, *group.values()) for group in representatives))
    proposals = (tuple(sorted(index for index in choice if index is not None)) for choice in choices)
    selected = list(min(
        (indices for indices in proposals if len({ranked[i]["game_id"] for i in indices}) == len(indices)),
        key=lambda indices: (-len(indices), indices),
    ))
    while len(selected) < min(PER_STRATUM, len(ranked)):
        games = {ranked[i]["game_id"] for i in selected}
        covered = {buckets[i] for i in selected}
        selected.append(min(
            (i for i in range(len(ranked)) if i not in selected),
            key=lambda i: (ranked[i]["game_id"] in games, buckets[i] in covered, i),
        ))
    return [ranked[i] for i in sorted(selected)]


def select_samples(population, publication_digest):
    groups = {(position, wolves): [] for position in POSITIONS for wolves in (1, 2)}
    for row in population:
        groups[row["position"], row["alive_wolves"]].append(row)
    strata, selected = [], []
    for (position, wolves), rows in groups.items():
        picked = _select_stratum(rows, publication_digest)
        strata.append({
            "position": position, "alive_wolves": wolves, "available_pres": len(rows),
            "available_games": len({row["game_id"] for row in rows}),
            "replyable_observer_counts": dict(sorted(Counter(row["replyable_observer_count"] for row in rows).items())),
            "selected_pres": len(picked),
        })
        selected.extend(picked)
    return strata, selected


def open_games(game_ids):
    return open_development_input(
        frozen.DEVELOPMENT, expected_publication_digest=frozen.PUBLICATION_DIGEST,
        expected_split_digest=frozen.SPLIT_DIGEST, game_ids=game_ids,
    )


def error_detail(error):
    return {"type": type(error).__name__, "message": str(error)}


def bind_q(pre, record):
    if record is None:
        raise ValueError(f"合法 PRE 缺少冻结 Q：{pre.game_id}/{pre.boundary_id}")
    identity = tuple(getattr(pre, name) for name in PRE_IDENTITY)
    if tuple(record[name] for name in PRE_IDENTITY) != identity:
        raise ValueError(f"冻结 Q 与 PRE 身份不符：{pre.game_id}/{pre.boundary_id}")
    return ProbabilityMatrix(*identity[:3], record["q"])


def load_sample():
    publication = open_artifact_envelope(
        frozen.DEVELOPMENT, expected_artifact_type=frozen.PUBLICATION_TYPE,
        expected_schema_version=frozen.PUBLICATION_VERSION,
    )
    manifest = publication.manifest
    if publication.manifest_digest != frozen.PUBLICATION_DIGEST or manifest["split_manifest_digest"] != frozen.SPLIT_DIGEST:
        raise ValueError("Development Publication 或 Split 身份不符")
    folds = json.loads(read_artifact_file(publication, "public/development_fold_manifest.json"))
    game_folds = {game["game_id"]: fold["fold_index"] for fold in folds["folds"]
                  if fold["fold_index"] in SELECTION_FOLDS for game in fold["held_out_games"]}
    members = {game["game_id"]: game for game in manifest["development_games"] if game["game_id"] in game_folds}
    q_records = _frozen_q(manifest, set(members))
    population, q_inputs, teams = [], {}, None
    coverage = dict.fromkeys(("matched", "missing", "invalid"), 0)
    game_ids = tuple(members)
    for offset in range(0, len(game_ids), 120):
        batch_ids = game_ids[offset:offset + 120]
        batch = open_games(batch_ids)
        if batch.public_view.game_ids != batch_ids:
            raise ValueError("Development 批次与所选折名单不符")
        if teams is None:
            teams = {(g.game_id, g.bundle_digest): tuple(seat for seat, role in g.role_assignment if role == "Werewolf")
                     for g in open_role_sidecar(batch).games if g.game_id in members}
        for game in batch.public_view.games:
            team = teams[game.game_id, game.bundle_digest]
            for pre in game.authoritative_pre_prefixes:
                if pre.current_speaker not in team or pre.public_temporal_state.phase.value not in SPEECH_PHASES:
                    continue
                row = opportunity(pre, team, fold=game_folds[game.game_id], attempt_id=members[game.game_id]["attempt_id"])
                key = pre_key(row)
                if key in q_inputs:
                    raise ValueError(f"Development 中有重复 PRE：{key}")
                q_record = q_records.pop(key, None)
                try:
                    matrix = bind_q(pre, q_record)
                except (KeyError, TypeError, ValueError) as error:
                    coverage["missing" if q_record is None else "invalid"] += 1
                    q_inputs[key] = (None, error_detail(error))
                else:
                    coverage["matched"] += 1
                    q_inputs[key] = (matrix, None)
                population.append(row)

    strata, selected = select_samples(population, publication.manifest_digest)
    plan = {
        "type": "sample", "folds": SELECTION_FOLDS,
        "sources": {"publication_digest": publication.manifest_digest, "split_digest": frozen.SPLIT_DIGEST,
                    "fold_manifest_digest": manifest["development_fold_manifest_digest"],
                    "role_sidecar_digest": manifest["role_sidecar_digest"], "q_manifest_digest": frozen.Q_DIGEST},
        "q_coverage": {**coverage, "unmatched_q_records": len(q_records)},
        "strata": strata, "selected": selected, "request_limit": len(selected),
    }
    inputs = {pre_key(row): (
        teams[row["game_id"], members[row["game_id"]]["bundle_digest"]], *q_inputs[pre_key(row)],
    ) for row in selected}
    return plan, inputs


def load_selected_pres(selected):
    if not selected:
        return {}
    ids = tuple(dict.fromkeys(row["game_id"] for row in selected))
    keys = {pre_key(row) for row in selected}
    games = open_games(ids).public_view.games
    return {(pre.game_id, pre.boundary_id): pre for game in games for pre in game.authoritative_pre_prefixes
            if (pre.game_id, pre.boundary_id) in keys}


def generate_record(row, pre, team, q, q_error, agent):
    result = {"type": "pre", **row, "status": "input_error", "error": None,
              "input": None, "phase_speaker_order": None, "request": None, "topic": None, "candidates": []}
    if pre is None or any(getattr(pre, name) != row[name] for name in PRE_IDENTITY):
        result["error"] = error_detail(ValueError("入选 PRE 缺失或与固定名单身份不符"))
        return result

    order, index, observers, replyable = speech_window(pre, team)
    result["phase_speaker_order"] = order
    if q_error is not None:
        result.update(error=q_error, input={
            "speaker": pre.current_speaker, "day": pre.public_temporal_state.day,
            "phase": pre.public_temporal_state.phase.value, "alive_players": pre.alive_observer_ids,
            "known_wolf_team": team, "public_events": pre.public_event_history.to_records(),
            "remaining_speakers": order[index + 1:], "replyable_observers": replyable,
            "wolf_suspicion_mass": {"observer_ids": observers,
                                    "wolf_ids": tuple(seat for seat in team if seat in pre.alive_observer_ids),
                                    "values": None},
        })
        return result
    try:
        result["input"] = build_candidate_context(pre, team, q)
    except (TypeError, ValueError) as error:
        result["error"] = error_detail(error)
        return result

    requests = []
    try:
        with model_request_context(requests, step=pre.step_index, seat=int(pre.current_speaker.removeprefix("player")),
                                   method="speech_candidates", phase=pre.public_temporal_state.phase.value):
            batch = agent.generate_speech_candidates(pre, known_wolf_ids=team, q=q)
    except (GameplayActionValidationError, GameplaySpeechQualityError) as error:
        result.update(status="validation_error", error=error_detail(error))
    except BackendError as error:
        received = requests and requests[0]["status"] in ("invalid_response", "response_received")
        result.update(status="validation_error" if received else "backend_error", error=error_detail(error))
    else:
        result.update(status="ok", topic=batch.topic, candidates=[asdict(candidate) for candidate in batch.candidates])
    if requests:
        fields = ("status", "response_received", "error_type", "finish_reason", "elapsed_seconds", "sdk_retries",
                  "input_tokens", "output_tokens", "total_tokens")
        result["request"] = {field: requests[0][field] for field in fields}
    return result


def load_agent(runtime_path):
    raw = runtime_path.read_bytes()
    config = normalize_runtime_config(yaml.safe_load(raw))
    profile = next((p for p in config["agent_config"]["all_candidates"] if p["profile_name"] == PROFILE), None)
    backend_config = config["backends"].get(PROFILE)
    if (profile is None or profile["agent_type"] != "gpt" or profile["backend"] != PROFILE
            or profile["model"] != MODEL or not backend_config or not backend_config["supports_json_schema"]):
        raise ValueError("请使用支持 JSON Schema 的既有 local_qwen / qwen35-9b profile")
    params = profile["model_params"]
    if (params.get("temperature"), params.get("gameplay_max_tokens")) != (0.7, 2048):
        raise ValueError("首轮须使用既有 0.7 temperature、2048 gameplay_max_tokens")
    backend = load_named_backends(config, env_file=None, max_retries=0)[PROFILE]
    agent = GPTAgent(backend=backend, model_name=profile["model"], **params)
    source_files = (Path(__file__).resolve(), ROOT / "werewolf/observer_dtrs/speech_candidates.py",
                    ROOT / "werewolf/agents/gpt_agent.py")
    settings = {
        "type": "settings", "runtime_config_sha256": sha256_bytes(raw),
        "source_sha256": {str(path.relative_to(ROOT)): sha256_bytes(path.read_bytes()) for path in source_files},
        "profile": PROFILE, "backend": PROFILE, "base_url": backend_config["base_url"], "model": profile["model"],
        "temperature": params["temperature"], "max_tokens": params["gameplay_max_tokens"], "max_retries": 0,
        "prompt_prefix": SPEECH_CANDIDATE_PROMPT, "response_schema": "speech_candidates", "declared_basis_fields": True,
        "chat_template_kwargs": {"enable_thinking": False}, "semantic_quality": "human_review_pending",
    }
    return agent, settings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--sample-only", action="store_true", help="只读输出固定名单及人口统计")
    mode.add_argument("--runtime-config", type=Path, help="按固定名单执行一次候选生成")
    args = parser.parse_args(argv)
    plan, inputs = load_sample()
    print(json.dumps(plan, ensure_ascii=False, allow_nan=False), flush=True)
    if args.sample_only:
        return
    selected = plan["selected"]
    pres = load_selected_pres(selected)
    agent, settings = load_agent(args.runtime_config)
    print(json.dumps(settings, ensure_ascii=False, allow_nan=False), flush=True)
    for row in selected:
        key = pre_key(row)
        record = generate_record(row, pres.get(key), *inputs[key], agent)
        print(json.dumps(record, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
