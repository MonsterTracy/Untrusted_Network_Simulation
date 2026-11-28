"""评价冻结 Final-ToM 的发言后果。读取固定候选，或回放实际发言。"""

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from itertools import combinations
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from scripts import analyze_observer_dtrs_report_event as scoring  # noqa: E402
from scripts import analyze_observer_dtrs_speech_candidates as candidate_study  # noqa: E402
from scripts import analyze_observer_dtrs_team_event as development  # noqa: E402
from scripts import fit_observer_dtrs_report_event as frozen  # noqa: E402
from scripts import fit_observer_dtrs_report_state as probability_study  # noqa: E402
from werewolf.artifact_io import (  # noqa: E402
    canonical_json_bytes, canonical_jsonl_bytes, open_artifact_envelope, publish_artifact, read_artifact_file, sha256_bytes,
)
from werewolf.backends import load_named_backends, resolve_backend  # noqa: E402
from werewolf.canonical_collection.game_bundle import _history_from_records  # noqa: E402
from werewolf.canonical_collection.public_history import PLAYER_IDS  # noqa: E402
from werewolf.canonical_collection.speech import V1_SPEECH_PARSER_VERSION, V1_SPEECH_PROMPT_VERSION  # noqa: E402
from werewolf.development_publication import open_role_sidecar  # noqa: E402
from werewolf.observer_dtrs import report_state_probability as probability  # noqa: E402
from werewolf.observer_dtrs import speech_consequences as consequences  # noqa: E402
from werewolf.observer_dtrs.speech_candidates import SPEECH_FUNCTIONS, parse_speech_candidates  # noqa: E402
from werewolf.runtime_config import normalize_runtime_config  # noqa: E402
from werewolf.speech.speech_perceiver import SpeechPerceiver  # noqa: E402
from werewolf.structured_history import plan_public_history, plan_structured_history  # noqa: E402
from werewolf.tom.dataset import belief_target  # noqa: E402
from werewolf.tom.protocol import BOOTSTRAP_VERSION, bootstrap_indices  # noqa: E402
from werewolf.tom.scoring import row_kl  # noqa: E402


ARTIFACT_TYPE = "observer_dtrs_speech_consequences"
DESTINATIONS = {name: frozen.DATA / "dtrs" / directory for name, directory in (
    ("factual", "speech-consequence-replay"), ("candidates", "speech-consequence-candidates"),
)}
BOOTSTRAP_REPLICATES = probability_study.BOOTSTRAP_REPLICATES


def read_candidate_study(path):
    if not path.is_file():
        raise FileNotFoundError(f"缺少既有固定样本的候选生成 JSONL：{path}")
    raw = path.read_bytes()
    plan, *records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if plan["type"] != "sample" or not records or records[0]["type"] != "settings":
        raise ValueError("需要含名单、设置和已执行生成记录的 JSONL")
    settings, *generated = records
    selected = plan["selected"]
    if not selected or len(selected) > 30 or len({candidate_study.pre_key(row) for row in selected}) != len(selected):
        raise ValueError("固定 PRE 名单必须为 1–30 个不同的机会")
    if len(selected) != len(generated):
        raise ValueError("候选记录与固定 PRE 名单数量不符")
    for entry, row in zip(selected, generated, strict=True):
        if row["type"] != "pre" or any(row.get(key) != value for key, value in entry.items()):
            raise ValueError("候选生成记录与固定 PRE 名单不符")
    return plan, settings, generated, sha256_bytes(raw)


def load_candidate_inputs(plan, publication):
    manifest = publication.manifest
    actual_source = {"publication_digest": publication.manifest_digest, "split_digest": frozen.SPLIT_DIGEST,
                     "fold_manifest_digest": manifest["development_fold_manifest_digest"],
                     "role_sidecar_digest": manifest["role_sidecar_digest"], "q_manifest_digest": frozen.Q_DIGEST}
    if plan["sources"] != actual_source or tuple(plan["folds"]) != development.SELECTION_FOLDS:
        raise ValueError("候选名单与冻结 Development 来源不符")
    ids = tuple(dict.fromkeys(row["game_id"] for row in plan["selected"]))
    selected_games = set(ids)
    batch = candidate_study.open_games(ids)
    if batch.public_view.game_ids != ids:
        raise ValueError("读取的人口与固定名单不符")
    teams = {(g.game_id, g.bundle_digest): tuple(s for s, role in g.role_assignment if role == "Werewolf")
             for g in open_role_sidecar(batch).games if g.game_id in selected_games}
    fold_of = {game_id: fold.fold_index for fold in batch.public_view.fold_manifest.folds for game_id in fold.game_ids}
    members = {g["game_id"]: g for g in manifest["development_games"] if g["game_id"] in selected_games}
    games = {game.game_id: game for game in batch.public_view.games}
    pres = {(game.game_id, pre.boundary_id): pre for game in games.values() for pre in game.authoritative_pre_prefixes}
    q_records = development._frozen_q(manifest, selected_games)
    selected = {}
    for row in plan["selected"]:
        key = candidate_study.pre_key(row)
        pre, member = pres[key], members[row["game_id"]]
        game = games[pre.game_id]
        team = teams[game.game_id, game.bundle_digest]
        if (game.bundle_digest != member["bundle_digest"] or fold_of[pre.game_id] not in development.SELECTION_FOLDS
                or candidate_study.opportunity(pre, team, fold=fold_of[pre.game_id], attempt_id=member["attempt_id"]) != row):
            raise ValueError(f"固定候选 PRE 与权威游戏或合法机会不符：{key}")
        selected[key] = pre, team, q_records.get(key)
    return selected


def load_perceiver(config_path):
    raw = config_path.read_bytes()
    config = normalize_runtime_config(yaml.safe_load(raw))
    parser = config["parser"]
    backend = resolve_backend(parser["backend"], load_named_backends(config, env_file=None, max_retries=0))
    return SpeechPerceiver(backend=backend, model_name=parser["model"]), {
        "runtime_config_sha256": sha256_bytes(raw), "backend": parser["backend"], "model": parser["model"],
        "prompt_version": V1_SPEECH_PROMPT_VERSION, "parser_version": V1_SPEECH_PARSER_VERSION,
    }


def evaluate_candidates(generated, inputs, model, *, perceiver, predictor):
    results = []
    for row in generated:
        identity = {key: row[key] for key in (*development.PRE_IDENTITY, "attempt_id", "fold", "position", "alive_wolves")}
        if row["status"] != "ok":
            results.append({**identity, "status": row["status"], "error": row["error"], "candidates": []})
            continue
        pre, team, raw_q = inputs[candidate_study.pre_key(row)]
        q = candidate_study.bind_q(pre, raw_q)
        # 核对原调用确实消费了同一席位 Q、队伍及完整公开前缀。
        canonical = candidate_study.build_candidate_context(pre, team, q)
        if canonical_json_bytes(row["input"]) != canonical_json_bytes(canonical):
            raise ValueError(f"已生成候选的输入与冻结 PRE 不符：{pre.boundary_id}")
        batch = parse_speech_candidates(json.dumps({"topic": row["topic"], "candidates": row["candidates"]}),
                                        pre=pre, replyable_observers=canonical["replyable_observers"])
        result = consequences.score_candidates(batch, team, q, model, perceiver=perceiver, predictor=predictor)
        results.append({**identity, "status": "evaluated", **result})
    return results


def ratio(n, d):
    return n / d if d else None


def _candidate_metrics(groups):
    counts = Counter()
    for candidates in groups:
        counts["candidates"] += len(candidates)
        parsed = [c for c in candidates if c["v1_status"] != "parser_error"]
        scores = [c["prediction"] for c in candidates if c["prediction"] is not None]
        counts["parse_success"] += len(parsed)
        counts["no_action"] += sum(c["v1_status"] == "no_action" for c in parsed)
        for a, b in combinations(parsed, 2):
            counts["parsed_pairs"] += 1
            counts["equal_v1_pairs"] += a["parse"]["normalized_actions"] == b["parse"]["normalized_actions"]
        counts["pres_two_model_inputs"] += len({p["model_input_digest"] for p in scores}) >= 2
        if len(scores) >= 2:
            counts["pres_two_scores"] += 1
            counts["pres_all_scores_tied"] += len({s["V_soc"] for s in scores}) == 1
    counts["pres"] = len(groups)
    return {**{name: counts[name] for name in ("pres", "candidates", "parse_success", "no_action", "parsed_pairs",
                                                "equal_v1_pairs", "pres_two_model_inputs", "pres_two_scores", "pres_all_scores_tied")},
            "parse_success_rate": ratio(counts["parse_success"], counts["candidates"]),
            "no_action_rate_given_success": ratio(counts["no_action"], counts["parse_success"]),
            "v1_pair_collision_rate": ratio(counts["equal_v1_pairs"], counts["parsed_pairs"]),
            "pres_two_model_inputs_rate": ratio(counts["pres_two_model_inputs"], counts["pres"]),
            "all_scores_tied_rate_given_two_scores": ratio(counts["pres_all_scores_tied"], counts["pres_two_scores"])}


def summarize_candidates(records):
    by_pre, examples, pair_counts = [], {}, {}
    for row in records:
        candidates = row["candidates"]
        scored = [c["prediction"]["V_soc"] for c in candidates if c["prediction"] is not None]
        by_pre.append({"game_id": row["game_id"], "boundary_id": row["boundary_id"], "scored_candidates": len(scored),
                       "V_soc_range": max(scored) - min(scored) if len(scored) >= 2 else None})
        for a, b in combinations((c for c in candidates if c["v1_status"] != "parser_error"), 2):
            pair = "/".join(sorted((a["function"], b["function"])))
            counts = pair_counts.setdefault(pair, {"pairs": 0, "equal_v1": 0})
            counts["pairs"] += 1
            actions = a["parse"]["normalized_actions"]
            if actions == b["parse"]["normalized_actions"]:
                counts["equal_v1"] += 1
                examples.setdefault(pair, {"game_id": row["game_id"], "boundary_id": row["boundary_id"],
                    "candidates": [{key: c[key] for key in ("function", "text", "addressee")} for c in (a, b)],
                    "shared_normalized_actions": actions})
    by_function = {}
    for function in SPEECH_FUNCTIONS:
        groups = [[c for c in row["candidates"] if c["function"] == function] for row in records]
        by_function[function] = _candidate_metrics([group for group in groups if group])
    return {"overall": _candidate_metrics([row["candidates"] for row in records]),
            "by_function": by_function,
            "pairs_by_function": {key: {**counts, "collision_rate": ratio(counts["equal_v1"], counts["pairs"])}
                                  for key, counts in pair_counts.items()},
            "by_pre": by_pre, "semantic_compression_examples": examples,
            "generation_statuses": dict(Counter(row["status"] for row in records))}


class ReplayGame:
    """一局的真实边界索引，只在离线对照和标签联结中使用。"""

    def __init__(self, game):
        self.history = game.public_event_stream
        self.annotations = {a.event_id: a for a in game.speech_annotations_v1}
        self.pres = {pre.public_event_history.events[-1].event_id: pre for pre in game.authoritative_pre_prefixes}

    def realized_speech(self, pre):
        prefix = pre.public_event_history.events
        events = self.history.events
        offset = len(prefix)
        if (events[:offset] != prefix or offset >= len(events) or events[offset].event_type != "public_speech"
                or events[offset].speaker != pre.current_speaker or events[offset].temporal_state != pre.public_temporal_state):
            raise ValueError(f"真实发言与权威 PRE 不匹配：{pre.boundary_id}")
        speech = events[offset]
        annotation = self.annotations[speech.event_id]
        if not annotation.is_successful:
            raise ValueError(f"Canonical 发言缺少成功 V1：{speech.event_id}")
        return speech, annotation

    def observed_boundary(self, pre, plan, speech, annotation):
        """只在原生紧邻后缀取切片，投票末位以真实阶段切换作为公开证据。"""
        events = self.history.events
        index = speech.event_index + 1
        if index >= len(events):
            return None, None, ("missing_immediate_public_boundary",)
        observed = events[index]
        marker = plan.marker
        if (observed.event_type, observed.speaker, observed.temporal_state.day, observed.temporal_state.phase.value) != (
                marker.token_type, marker.source, marker.day, marker.phase):
            return None, observed, ("immediate_public_marker_mismatch",)
        if marker.token_type == "phase_change":
            history = _history_from_records([event.to_record() for event in events[:index + 1]])
            actions = {item.event_id: item.actions for item in (*pre.v1_annotations, annotation)}
            actual = plan_public_history(history, actions, boundary_digest=history.digest)
            followup = None
        else:
            followup = self.pres.get(observed.event_id)
            if (followup is None or followup.game_id != pre.game_id or followup.current_speaker != marker.source
                    or followup.public_event_history.events != events[:index + 1]
                    or followup.alive_observer_ids != pre.alive_observer_ids or followup.step_index <= pre.step_index
                    or followup.v1_annotations != (*pre.v1_annotations, annotation)):
                return None, observed, ("missing_or_misbound_adjacent_pre",)
            actual = plan_structured_history(followup)
        return followup, observed, consequences.model_input_differences(plan.token_plan, actual)


def _paired_report(pre, followup, seat, report, report_id, attempt_id):
    if report is None:
        return "missing", None
    if ((report.game_id, report.attempt_id, report.boundary_id, report.prefix_digest, report.observer_id,
         report.observation_id, report.day, report.phase) != (pre.game_id, attempt_id, followup.boundary_id,
         followup.prefix_digest, seat, report_id, followup.public_temporal_state.day, followup.public_temporal_state.phase.value)
            or not report.observer_alive):
        return "invalid_or_misbound", None
    if report.status.value != "success":
        return report.status.value, None
    return "success", report.suspicion_support


@dataclass(frozen=True)
class ReplayRecord:
    data: dict

    @property
    def game_id(self):
        return self.data["game_id"]

    @property
    def sample_id(self):
        return self.game_id, self.data["boundary_id"]


def replay_speech(game, pre, team, reports, current_q, model, *, attempt_id, predictor):
    contexts, current = consequences.current_exposure(pre, team, current_q, model)
    speech, annotation = game.realized_speech(pre)
    order = consequences.derive_public_phase_speaker_order(pre)
    last = order[-1] == pre.current_speaker
    record = {
        **{key: getattr(pre, key) for key in development.PRE_IDENTITY}, "attempt_id": attempt_id,
        "acting_wolf": pre.current_speaker, "alive_wolf_ids": contexts[0].alive_wolf_ids,
        "phase": contexts[0].phase, "position": "last" if last else "nonlast",
        "eligible_observers": len(contexts), "speech_event_id": speech.event_id, "speech_text": speech.raw_text,
        "v1_actions": [item.to_record() for item in annotation.actions], "reports": [], "prediction": None,
        "report_boundary": "immediate_vote_report_uncollected" if last else "adjacent_speech_pre",
        "input_verified": False,
    }
    try:
        plan = consequences.build_consequence_plan(pre, speech.raw_text, annotation.actions)
    except consequences.ConsequenceCapacityError:
        return ReplayRecord({**record, "status": "capacity_exceeded"})

    followup, observed, differences = game.observed_boundary(pre, plan, speech, annotation)
    record.update(public_boundary_event_id=observed.event_id if observed is not None else None,
                  followup_boundary_id=followup.boundary_id if followup is not None else None,
                  model_input_differences=differences, input_verified=not differences)
    if differences:
        return ReplayRecord({**record, "status": "unmatched_public_boundary"})

    forecast = predictor.forecast(plan)
    updated = consequences.consequence_exposure(contexts, plan, forecast.matrix, model)
    record.update(status="predicted", current_exposure=asdict(current),
                  prediction=consequences.forecast_record(plan, forecast, updated, current))
    if followup is None:
        return ReplayRecord(record)

    record["next_speaker_faction"] = "wolf" if followup.current_speaker in team else "non_wolf"
    linked = {link.observer_id: link.observation_id for link in followup.belief_observation_links}
    before_log = torch.as_tensor(current_q.values, dtype=torch.float64).log()
    for ctx in contexts:
        seat = ctx.observer_id
        report_id = linked.get(seat)
        status, raw = _paired_report(pre, followup, seat, reports.get(report_id), report_id, attempt_id)
        row = {"observer_id": seat, "observation_id": report_id, "status": status}
        record["reports"].append(row)
        if raw is None:
            continue
        state = 2 if pre.current_speaker in raw else int(any(wolf in raw for wolf in ctx.alive_wolf_ids))
        ys, yT = int(state == 2), int(state != 0)
        col = PLAYER_IDS.index(seat)
        target, _ = belief_target(col, raw, status)
        target = target.to(torch.float64)
        row.update(state=state, ys=ys, yT=yT, successful_empty=not raw)
        for name, exposure, log_q in (("carry", current, before_log), ("updated", updated, forecast.log_q)):
            p = exposure.observers[seat]
            row[name] = {"scores": {"team_brier": (p["pT"] - yT) ** 2,
                                    "speaker_brier": (p["ps"] - ys) ** 2,
                                    "joint_brier": .5 * sum((p[f"q{k}"] - int(state == k)) ** 2 for k in range(3)),
                                    "q_kl": float(row_kl(log_q[col], target))},
                         "probabilities": p}
    return ReplayRecord(record)


def evaluate_factual(model, predictor):
    current_game = None

    def visit_game(fold, game, team, *, publication_digest, attempt_id):
        nonlocal current_game
        current_game = ReplayGame(game)

    def build_samples(pre, team, reports, q_record, *, attempt_id):
        q = development.bind_q(pre, q_record)
        return (replay_speech(current_game, pre, team, reports, q, model, attempt_id=attempt_id, predictor=predictor),)

    parts, lineage = development.load_development((development.EVALUATION_FOLD,),
                                                  build_samples=build_samples, visit_game=visit_game)
    return [row.data for row in parts[development.EVALUATION_FOLD]], lineage


def _support(records):
    predicted = [r for r in records if r["status"] == "predicted"]
    reports = [row for r in predicted for row in r["reports"]]
    statuses = Counter(row["status"] for row in reports)
    observers = sum(r["eligible_observers"] for r in records)
    scored = [r for r in predicted if any(row["status"] == "success" for row in r["reports"])]
    return {"games": len({r["game_id"] for r in records}), "legal_pres": len(records), "legal_observers": observers,
            "verified_public_boundaries": sum(r["input_verified"] for r in records),
            "predicted_pres": len(predicted), "adjacent_speech_pres": sum(bool(r["reports"]) for r in predicted),
            "immediate_reports_uncollected_observers": sum(r["eligible_observers"] for r in records if r["position"] == "last"),
            "successful_reports": statuses["success"], "report_statuses": dict(statuses),
            "supervision_coverage": ratio(statuses["success"], observers),
            "scored_pres": len(scored), "scored_games": len({r["game_id"] for r in scored}),
            "statuses": dict(Counter(r["status"] for r in records)),
            "alignment_exclusions": dict(Counter(reason for r in records for reason in r.get("model_input_differences", ())))}


def summarize_factual(records, *, replicates=BOOTSTRAP_REPLICATES):
    report = {"population": {"overall": _support(records), "by_phase_position_wolves": {
        f"{phase}/{position}/{wolves}": _support([r for r in records if (r["phase"], r["position"], len(r["alive_wolf_ids"]))
                                                    == (phase, position, wolves)])
        for phase in consequences.VOTE_PHASE for position in ("nonlast", "last") for wolves in (1, 2)}}, "scores": {}}
    rows = [(pre, row) for pre in records for row in pre["reports"] if row["status"] == "success"]
    if not rows:
        return report
    games = tuple(sorted({pre["game_id"] for pre, _ in rows}))
    positions = {game: i for i, game in enumerate(games)}
    by_game = np.asarray([positions[r["game_id"]] for r, _ in rows])
    weights = 1 / np.bincount(by_game, minlength=len(games))
    draws = bootstrap_indices(games, replicates, frozen.BOOTSTRAP_SEED)
    groups = {"overall": np.ones(len(rows), dtype=bool)}
    for phase in consequences.VOTE_PHASE:
        groups[f"phase/{phase}"] = np.asarray([r["phase"] == phase for r, _ in rows])
    for wolves in (1, 2):
        groups[f"wolves/{wolves}"] = np.asarray([len(r["alive_wolf_ids"]) == wolves for r, _ in rows])
    for faction in ("wolf", "non_wolf"):
        groups[f"next_speaker/{faction}"] = np.asarray([r["next_speaker_faction"] == faction for r, _ in rows])
    groups["report/empty"] = np.asarray([row["successful_empty"] for _, row in rows])
    groups["report/nonempty"] = ~groups["report/empty"]
    for name, mask in groups.items():
        selected = [rows[i] for i in np.flatnonzero(mask)]
        counts = np.bincount(by_game[mask], minlength=len(games))
        mass = weights @ counts
        entry = report["scores"][name] = {"rows": len(selected), "games": int(np.count_nonzero(counts)),
                                         "game_equal_mass": float(mass / len(games)), "metrics": None, "calibration": None}
        if not len(selected):
            continue
        entry["metrics"] = {}
        for metric in ("team_brier", "speaker_brier", "joint_brier", "q_kl"):
            before, after = (np.bincount(by_game[mask], weights=[row[mode]["scores"][metric] for _, row in selected],
                                         minlength=len(games)) for mode in ("carry", "updated"))
            entry["metrics"][metric] = {"carry": float(weights @ before / mass), "updated": float(weights @ after / mass),
                                       "improvement": float(weights @ (before - after) / mass),
                                       "paired_bootstrap": scoring.interval(scoring.bootstrap_ratio(before - after, counts, weights, draws))}

        w = weights[by_game[mask]] / mass
        entry["calibration"] = {}
        events = {"pT": lambda state: state != 0, "ps": lambda state: state == 2,
                  **{f"q{k}": (lambda state, k=k: state == k) for k in range(3)}}
        for event, is_hit in events.items():
            binary = [SimpleNamespace(game_id=pre["game_id"], sample_id=(pre["game_id"], pre["boundary_id"], row["observer_id"]),
                                      y=int(is_hit(row["state"]))) for pre, row in selected]
            weight_map = frozen.probability_map(binary, w)
            entry["calibration"][event] = {mode: frozen.reliability(binary, frozen.probability_map(
                binary, (row[mode]["probabilities"][event] for _, row in selected)), weight_map) for mode in ("carry", "updated")}
    return report


def load_probability_model(path, publication):
    archive = open_artifact_envelope(path, expected_artifact_type=probability.ARTIFACT_TYPE,
                                     expected_schema_version=probability.ARTIFACT_TYPE)
    expected = {"publication_digest": publication.manifest_digest, "split_digest": frozen.SPLIT_DIGEST,
                "fold_manifest_digest": publication.manifest["development_fold_manifest_digest"],
                "q_manifest_digest": frozen.Q_DIGEST, "q_predictions_sha256": frozen.Q_SHA256}
    if any(archive.manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("第一层概率产物与固定 Development、Fold 或 Q 不匹配")
    model = probability.load_development_model(path)
    return model, {**expected, "probability_manifest_digest": archive.manifest_digest,
                   "method": model["method"], "lambda": model["lambda"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    factual = commands.add_parser("factual", help="回放冻结 Development Fold 4 的真实发言")
    candidates = commands.add_parser("candidates", help="读取已经生成的固定样本候选")
    for command in (factual, candidates):
        command.add_argument("--capsule", type=Path, required=True)
        command.add_argument("--capsule-digest", required=True)
        command.add_argument("--probability-artifact", type=Path, default=probability_study.DESTINATION)
    candidates.add_argument("--candidate-jsonl", type=Path, required=True)
    candidates.add_argument("--runtime-config", type=Path, required=True)
    args = parser.parse_args(argv)

    destination = DESTINATIONS[args.mode]
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"已有发言后果评价产物：{destination}")
    if args.mode == "candidates":
        plan, settings, generated, candidate_digest = read_candidate_study(args.candidate_jsonl)
    publication = open_artifact_envelope(frozen.DEVELOPMENT, expected_artifact_type=frozen.PUBLICATION_TYPE,
                                         expected_schema_version=frozen.PUBLICATION_VERSION)
    if publication.manifest_digest != frozen.PUBLICATION_DIGEST or publication.manifest["split_manifest_digest"] != frozen.SPLIT_DIGEST:
        raise ValueError("冻结 Development 或 Split 不匹配")
    model, probability_source = load_probability_model(args.probability_artifact, publication)
    torch.set_num_threads(1)
    predictor = consequences.SpeechConsequencePredictor(args.capsule, expected_manifest_digest=args.capsule_digest,
                                                         expected_scope="PINNED_FINAL_INFERENCE_UNQUALIFIED", device="cpu")

    lineage = {"probability_source": probability_source, "final_tom": predictor.identity}
    if args.mode == "factual":
        rows, development_source = evaluate_factual(model, predictor)
        # PINNED Capsule 的 Final-ToM 训练合同对应早期 1500 局 Development。
        results = {"scope": "development_fold4_factual_replay", "source": development_source,
                   "final_tom_training_population": {"partition": "all_development", "games": 1500,
                                                     "observer_population": "non_wolf_alive",
                                                     "publication_digest": predictor.identity["origin"]["publication_digest"]},
                   "primary_metric": "team_brier", "bootstrap": {"unit": "game", "version": BOOTSTRAP_VERSION,
                        "replicates": BOOTSTRAP_REPLICATES, "seed": frozen.BOOTSTRAP_SEED}, **summarize_factual(rows)}
    else:
        inputs = load_candidate_inputs(plan, publication)
        perceiver, parser_source = load_perceiver(args.runtime_config)
        rows = evaluate_candidates(generated, inputs, model, perceiver=perceiver, predictor=predictor)
        results = {"scope": "fixed_candidate_representation", "candidate_jsonl_sha256": candidate_digest,
                   "candidate_settings": settings, "parser": parser_source, **summarize_candidates(rows)}

    result = publish_artifact(destination, manifest_fields={
        "artifact_type": ARTIFACT_TYPE, "schema_version": ARTIFACT_TYPE, "artifact_identity": destination.name,
        "mode": args.mode, **lineage,
        "source_sha256": {str(path.relative_to(ROOT)): sha256_bytes(path.read_bytes()) for path in (
            Path(__file__).resolve(), Path(consequences.__file__).resolve(), Path(probability.__file__).resolve())},
    }, files={"results.json": canonical_json_bytes({**lineage, **results}), "records.jsonl": canonical_jsonl_bytes(rows)})
    print(json.dumps({"path": str(destination), "manifest_digest": result.manifest_digest, "records": len(rows)}), flush=True)


if __name__ == "__main__":
    main()
