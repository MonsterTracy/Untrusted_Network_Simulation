"""拟合固定 C0/C1，评价历史行动条件下的序贯认知改善。"""

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from scripts import analyze_observer_dtrs_report_event as evaluation  # noqa: E402
from scripts import analyze_observer_dtrs_report_state_risk as risk  # noqa: E402
from scripts import analyze_observer_dtrs_sequential as e4  # noqa: E402
from scripts import analyze_observer_dtrs_team_event as development  # noqa: E402
from scripts import evaluate_observer_dtrs_speech_consequences as speech_replay  # noqa: E402
from scripts import fit_observer_dtrs_report_event as frozen  # noqa: E402
from scripts import fit_observer_dtrs_report_state as probability_study  # noqa: E402
from werewolf.artifact_io import (  # noqa: E402
    canonical_json_bytes, canonical_jsonl_bytes, open_artifact_envelope, publish_artifact, read_artifact_file, sha256_bytes,
)
from werewolf.observer_dtrs import cognitive_value as cognitive  # noqa: E402
from werewolf.observer_dtrs import report_state_probability as probability  # noqa: E402
from werewolf.observer_dtrs.report_state_decision import event_probabilities  # noqa: E402
from werewolf.tom.protocol import BOOTSTRAP_VERSION, bootstrap_indices  # noqa: E402


DESTINATION = frozen.DATA / "dtrs" / "cognitive-value"
BOOTSTRAP_REPLICATES = probability_study.BOOTSTRAP_REPLICATES
TARGETS = dict(zip(cognitive.HEADS, ("O", "K", "G_total"), strict=True))


@dataclass(frozen=True, slots=True)
class Anchor:
    history: str
    numeric: dict
    speech: str
    speech_event_id: str
    speech_sha256: str


@dataclass(frozen=True, slots=True)
class Trial:
    pair: e4.Pair
    anchor: Anchor
    inputs: cognitive.InputRow | None
    outcome: dict

    @property
    def sample_id(self):
        return e4.sample_id(self.pair.start)

    @property
    def game_id(self):
        return self.pair.start.game_id

    @property
    def can_reply(self):
        return bool(self.anchor.numeric[self.pair.start.observer_id][cognitive.REPLY_COLUMN])


def load_data():
    pairs = {fold: [] for fold in development.DEVELOPMENT_FOLDS}
    anchors, current = {}, None

    def visit_game(fold, game, team, *, publication_digest, attempt_id):
        nonlocal current
        _, game_pairs = e4.pair_game(game, team, publication_digest=publication_digest, attempt_id=attempt_id)
        pairs[fold].extend(game_pairs)
        current = speech_replay.ReplayGame(game)

    def build_samples(pre, team, reports, q_record, *, attempt_id):
        samples = tuple(probability_study.pre_samples(pre, team, reports, q_record, attempt_id=attempt_id))
        history, numeric = cognitive.pre_inputs(pre, [(s.context, s.features) for s in samples], q_record["q"])
        speech, _ = current.realized_speech(pre)
        anchors[pre.game_id, pre.boundary_id] = Anchor(history, numeric, speech.raw_text, speech.event_id,
                                                      sha256_bytes(speech.raw_text.encode("utf-8")))
        return samples

    partitions, source = development.load_development(development.DEVELOPMENT_FOLDS, build_samples=build_samples,
                                                      visit_game=visit_game)
    return partitions, pairs, anchors, source


def read_oof(partitions, records, outer_selection):
    aligned, selections = risk.match_oof(partitions, records, outer_selection)
    return {row.sample.sample_id: cognitive.joint_values(record)
            for row, record in zip(aligned, records, strict=True)}, selections


def frozen_joint(samples, model):
    features = np.asarray([sample.features for sample in samples])
    joint = probability.probabilities(probability.log_probabilities(model, features), features)
    return {sample.sample_id: tuple(map(float, q)) for sample, q in zip(samples, joint, strict=True)}


def brier(q, state):
    return 0.5 * sum((value - int(k == state)) ** 2 for k, value in enumerate(q))


def label_pair(pair, q_t, q_u, next_sample, *, trajectory_complete):
    """机会由已固定的紧邻配对决定，缺失监督不改变配对。"""
    followup = pair.followup
    opportunity = followup is not None
    if not opportunity and not trajectory_complete:
        # 已记录死亡或终局也无法排除档案缺口内曾有机会。
        opportunity = None
    label = {"O": opportunity, "K": None, "M": None, "G_total": None, "E_u": None,
             "carry_brier": None, "updated_brier": None, "q_t": q_t, "q_u": q_u,
             "trajectory_complete": trajectory_complete, "report_status": None, "successful_empty": None}
    if followup is None:
        return label

    state = next_sample.state if next_sample is not None else None
    stable = pair.start.alive_wolf_ids == followup.alive_wolf_ids
    valid = q_t is not None and q_u is not None and state is not None
    label.update(K=stable, M=valid, E_u=state,
                 report_status=next_sample.report_status if next_sample is not None else "missing",
                 successful_empty=next_sample.successful_empty if state is not None else None)
    if stable and valid:
        carry, updated = brier(q_t, state), brier(q_u, state)
        label.update(carry_brier=carry, updated_brier=updated, G_total=carry - updated)
    return label


def build_trials(pairs, samples, anchors, joint, *, trajectory_complete):
    by_sample = {s.sample_id: s for s in samples}
    starts = [e4.sample_id(pair.start) for pair in pairs]
    if len(set(starts)) != len(starts) or len(by_sample) != len(samples) or set(by_sample) != set(starts):
        raise ValueError("E4 起点与合法观察者 PRE 人口不一致")
    trials = []
    for key, pair in zip(starts, pairs, strict=True):
        current = by_sample[key]
        next_key = e4.sample_id(pair.followup) if pair.followup is not None else None
        next_sample = by_sample.get(next_key)
        for sample, context in ((current, pair.start), (next_sample, pair.followup)):
            if sample is not None and (sample.context != context or sample.attempt_id != pair.attempt_id):
                raise ValueError(f"E4 游戏尝试或 PRE 身份不匹配：{sample.sample_id}")
        q_t, q_u = joint.get(key), joint.get(next_key)
        anchor = anchors[key[:2]]
        inputs = None if q_t is None else cognitive.InputRow(key[:2], anchor.history, anchor.speech,
                                                               (*anchor.numeric[key[2]], *q_t))
        trials.append(Trial(pair, anchor, inputs, label_pair(pair, q_t, q_u, next_sample,
                                                            trajectory_complete=trajectory_complete)))
    return trials


def base_weights(trials):
    observers = Counter(row.sample_id[:2] for row in trials)
    pres = Counter(game for game, _ in observers)
    return np.asarray([1 / (pres[row.game_id] * observers[row.sample_id[:2]]) for row in trials])


def target_values(trials, key):
    return np.asarray([row.outcome[key] for row in trials], dtype=float)


def fit_and_predict(train, held):
    """拟合、词表和数值变换只消费 train，两个方法使用同一标签和权重。"""
    train_idx = [i for i, row in enumerate(train) if row.inputs is not None]
    held_idx = [i for i, row in enumerate(held) if row.inputs is not None]
    train_rows = [train[i].inputs for i in train_idx]
    weights = base_weights(train)[train_idx]
    if not train_rows:
        raise ValueError("Fold 0–3 没有具备合法预测输入的起点")
    transforms = cognitive.fit_transforms(train_rows, weights)
    targets = {name: target_values(train, key)[train_idx] for name, key in TARGETS.items()}
    targets["cognitive_gain"] = (targets["cognitive_gain"] + 1) / 2
    models, predictions = {}, {}
    for method in cognitive.METHODS:
        design = cognitive.design_matrix(train_rows, transforms, method)
        models[method] = {head: cognitive.fit_head(design, target, weights) for head, target in targets.items()}
        del design
        predictions[method] = {head: np.full(len(held), np.nan) for head in cognitive.HEADS}
        if held_idx:
            design = cognitive.design_matrix([held[i].inputs for i in held_idx], transforms, method)
            for head, pred in cognitive.predict_heads(design, models[method]).items():
                predictions[method][head][held_idx] = pred
            del design
    return {"specification": cognitive.SPECIFICATION, "transforms": transforms, "models": models}, predictions


def _support(trials, weights, mask):
    chosen = np.flatnonzero(mask)
    return {"observer_rows": len(chosen), "games": len({trials[i].game_id for i in chosen}),
            "pres": len({trials[i].sample_id[:2] for i in chosen}),
            "game_weight": float(weights[mask].sum())}


def population(trials, weights, mask):
    selected = [trials[i] for i in np.flatnonzero(mask)]
    statuses = Counter(row.outcome["report_status"] for row in selected if row.outcome["O"] is True)
    result = {"legal": _support(trials, weights, mask), "actual_actions": len({
        (row.game_id, row.anchor.speech_event_id) for row in selected}), "report_statuses_when_O": dict(statuses)}
    for name, checks in (
        ("opportunity", {"yes": ("O", True), "no": ("O", False), "unknown": ("O", None)}),
        ("target_stability", {"same": ("K", True), "changed": ("K", False)}),
        ("evaluation_evidence", {"valid": ("M", True), "invalid": ("M", False)}),
    ):
        result[name] = {key: _support(trials, weights, mask & np.asarray([r.outcome[field] is state for r in trials]))
                        for key, (field, state) in checks.items()}
    for name, indicator in (
        ("valid_G_total", np.isfinite(target_values(trials, "G_total"))),
        ("successful_reports", np.asarray([r.outcome["E_u"] is not None for r in trials])),
        ("successful_empty", np.asarray([r.outcome["successful_empty"] is True for r in trials])),
        ("trajectory_incomplete", np.asarray([not r.outcome["trajectory_complete"] for r in trials])),
        ("start_prediction_missing", np.asarray([r.outcome["q_t"] is None for r in trials])),
        ("followup_prediction_missing", np.asarray([r.outcome["O"] is True and r.outcome["q_u"] is None for r in trials])),
    ):
        result[name] = _support(trials, weights, mask & indicator)
    result["no_followup_by_evidence"] = dict(Counter(
        "/".join(name for name in ("wolf_death", "observer_death", "terminal") if getattr(row.pair, name) is not None)
        or "no_recorded_end_reason" for row in selected if row.outcome["O"] is not True))
    return result


def gain_distribution(trials, weights, mask, gain):
    valid = mask & np.isfinite(gain)
    if not valid.any():
        return None
    g, w = gain[valid], weights[valid]
    bins = np.linspace(-1, 1, 17)
    counts = np.histogram(g, bins=bins)[0].tolist()
    masses = (np.histogram(g, bins=bins, weights=w)[0] / w.sum()).tolist()
    return {"mean": float(np.average(g, weights=w)), "min": float(g.min()), "max": float(g.max()),
            "histogram": {"edges": bins.tolist(), "counts": counts, "weighted_proportions": masses},
            "improvement_proportions": {name: float(w[keep].sum() / w.sum()) for name, keep in (
                ("negative", g < 0), ("zero", g == 0), ("positive", g > 0))},
            **{key: float(np.average(target_values(trials, key)[valid], weights=w))
               for key in ("carry_brier", "updated_brier")}}


def summarize(held, predictions, *, replicates=BOOTSTRAP_REPLICATES):
    weights = base_weights(held)
    games = tuple(sorted({row.game_id for row in held}))
    positions = {game: index for index, game in enumerate(games)}
    game_idx = np.asarray([positions[row.game_id] for row in held])
    draws = bootstrap_indices(games, replicates, frozen.BOOTSTRAP_SEED)
    phase = np.asarray([row.pair.start.phase for row in held])
    dual = np.asarray([row.pair.start.wolf_count == 2 for row in held])
    reply = np.asarray([row.can_reply for row in held])
    groups = {"overall": np.ones(len(held), dtype=bool), "discussion": phase == "discussion",
              "pk_discussion": phase == "pk_discussion", "single_wolf": ~dual, "two_wolves": dual,
              "observer_can_reply": reply, "observer_cannot_reply": ~reply}
    targets = {head: target_values(held, name) for head, name in TARGETS.items()}
    report = {}
    for group, mask in groups.items():
        result = {"population": population(held, weights, mask),
                  "G_total_distribution": gain_distribution(held, weights, mask, targets["cognitive_gain"]),
                  "models": {method: {} for method in cognitive.METHODS}, "G_mse_C0_minus_C1": None}
        report[group] = result
        for head, target in targets.items():
            available = mask & np.isfinite(target)
            common = available & np.logical_and.reduce([np.isfinite(predictions[m][head]) for m in cognitive.METHODS])
            scored, valid = _support(held, weights, common), _support(held, weights, available)
            metric = "mse" if head == "cognitive_gain" else "brier"
            losses = {}
            for method in cognitive.METHODS:
                prediction = predictions[method][head][common]
                error = losses[method] = (prediction - target[common]) ** 2
                entry = {"label_support": valid, "scored_support": scored, metric: None}
                result["models"][method][head] = entry
                if not common.any():
                    continue
                entry[metric] = float(np.average(error, weights=weights[common]))
                if metric == "brier":
                    score_rows = [SimpleNamespace(sample_id=held[i].sample_id, game_id=held[i].game_id, y=int(target[i]))
                                  for i in np.flatnonzero(common)]
                    normalized = weights[common] / weights[common].sum()
                    entry["calibration"] = frozen.reliability(score_rows, frozen.probability_map(score_rows, prediction),
                                                              frozen.probability_map(score_rows, normalized))
            if metric == "mse" and common.any():
                delta = losses["C0"] - losses["C1"]
                numerator, denominator = (np.bincount(game_idx[common], weights=value, minlength=len(games))
                                          for value in (weights[common] * delta, weights[common]))
                intervals = evaluation.interval(evaluation.bootstrap_ratio(numerator, denominator, np.ones(len(games)), draws))
                result["G_mse_C0_minus_C1"] = {"value": float(np.average(delta, weights=weights[common])), **intervals}
    return report


def decomposition(path, held, probability_source):
    """只消费已有第三轮事实回放的同源 q_post，辅助分析限于 Fold 4。"""
    archive = open_artifact_envelope(path, expected_artifact_type=speech_replay.ARTIFACT_TYPE,
                                     expected_schema_version=speech_replay.ARTIFACT_TYPE)
    old = archive.manifest["probability_source"]
    if (archive.manifest["mode"] != "factual" or old["probability_manifest_digest"] != probability_source["manifest_digest"]
            or (old["method"], old["lambda"]) != (probability_source["method"], probability_source["lambda"])):
        raise ValueError("阶段分解需要同一冻结概率模型的第三轮事实回放")
    records = {}
    for line in read_artifact_file(archive, "records.jsonl").splitlines():
        row = json.loads(line)
        key = row["game_id"], row["boundary_id"]
        if key in records:
            raise ValueError(f"第三轮事实回放包含重复 PRE：{key}")
        records[key] = row

    stats = {}
    for trial in held:
        outcome = trial.outcome
        row = records.get(trial.sample_id[:2])
        if outcome["G_total"] is None or row is None or row["status"] != "predicted":
            continue
        c, anchor = trial.pair.start, trial.anchor
        identity = (row["attempt_id"], row["prefix_digest"], row["step_index"], row["acting_wolf"],
                    tuple(row["alive_wolf_ids"]), row["speech_event_id"], row["speech_text"])
        if identity != (trial.pair.attempt_id, c.prefix_digest, c.step_index, c.acting_wolf, c.alive_wolf_ids,
                        anchor.speech_event_id, anchor.speech):
            raise ValueError(f"第三轮发言与第四轮起点身份不符：{trial.sample_id}")
        current = row["current_exposure"]["observers"][c.observer_id]
        post = row["prediction"]["exposure"]["observers"][c.observer_id]
        event_probabilities(post, c.wolf_count)
        if not np.allclose(cognitive.joint_values(current), outcome["q_t"], rtol=0, atol=1e-12):
            raise ValueError(f"阶段分解的起点概率不符：{trial.sample_id}")
        q_post = cognitive.joint_values(post)
        loss = brier(q_post, outcome["E_u"])
        stats[trial.sample_id] = {"q_post": q_post, "G_self": outcome["carry_brier"] - loss,
                                  "G_external": loss - outcome["updated_brier"]}
    mask = np.asarray([trial.sample_id in stats for trial in held])
    weights = base_weights(held)
    return stats, {"status": "evaluated", "artifact_digest": archive.manifest_digest, "support": _support(held, weights, mask),
                   "means": {key: float(np.average([stats[held[i].sample_id][key] for i in np.flatnonzero(mask)],
                                                   weights=weights[mask])) if mask.any() else None
                             for key in ("G_self", "G_external")}}


def label_records(by_fold, predictions, stages):
    for fold, trials in by_fold.items():
        for i, (row, weight) in enumerate(zip(trials, base_weights(trials), strict=True)):
            pair, start = row.pair, row.pair.start
            yield {"fold": fold, "sequence_id": pair.sequence_id, "game_id": start.game_id,
                   "boundary_id": start.boundary_id, "prefix_digest": start.prefix_digest, "step_index": start.step_index,
                   "observer_id": start.observer_id, "acting_wolf": start.acting_wolf, "alive_wolf_ids": start.alive_wolf_ids,
                   "attempt_id": pair.attempt_id, "phase": start.phase, "observer_can_reply": row.can_reply,
                   "followup": pair.followup.to_record() if pair.followup is not None else None,
                   "speech_event_id": row.anchor.speech_event_id, "speech_sha256": row.anchor.speech_sha256,
                   "wolf_death": pair.wolf_death, "observer_death": pair.observer_death, "terminal": pair.terminal,
                   "base_weight": float(weight), **row.outcome,
                   "stage_decomposition": stages.get(row.sample_id),
                   "predictions": {m: cognitive.observer_prediction(heads, i) for m, heads in predictions.items()}
                   if fold == development.EVALUATION_FOLD else None}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probability-artifact", type=Path, default=probability_study.DESTINATION)
    parser.add_argument("--speech-consequences", type=Path, help="可选的第三轮事实回放产物，仅用于阶段分解")
    args = parser.parse_args(argv)
    if DESTINATION.exists() or DESTINATION.is_symlink():
        raise FileExistsError(f"已有主动认知价值产物：{DESTINATION}")
    archive = open_artifact_envelope(args.probability_artifact, expected_artifact_type=probability.ARTIFACT_TYPE,
                                     expected_schema_version=probability.ARTIFACT_TYPE)
    previous = json.loads(read_artifact_file(archive, "results.json"))
    model = probability.load_development_model(args.probability_artifact)
    selection = previous["selection"]
    method = selection["selected_method"]
    if (model["method"], model["lambda"]) != (method, selection["best_lambda"][method]):
        raise ValueError("冻结概率模型与第一轮 Development 选择不一致")

    partitions, pairs, anchors, lineage = load_data()
    if any(archive.manifest.get(key) != value for key, value in lineage.items()):
        raise ValueError("第一轮概率产物与当前 Development、Fold 或 Q 来源不符")
    records = [json.loads(line) for line in read_artifact_file(archive, "oof_predictions.jsonl").splitlines()]
    joint_oof, outer = read_oof(partitions, records, previous["outer_selection"])
    held_fold = development.EVALUATION_FOLD
    joint_held = frozen_joint(partitions[held_fold], model)
    # Development Publication 来自完整 Canonical 成功对局，未配对原因仍单独保留。
    by_fold = {fold: build_trials(pairs[fold], samples, anchors, joint_held if fold == held_fold else joint_oof,
                                  trajectory_complete=True) for fold, samples in partitions.items()}
    train = [row for fold in development.SELECTION_FOLDS for row in by_fold[fold]]
    held = by_fold[held_fold]
    probability_source = {"manifest_digest": archive.manifest_digest, "method": model["method"],
                          "lambda": model["lambda"], "outer_selection": {
                              str(fold): {**value, "training_folds": previous["outer_selection"][str(fold)]["selection_folds"]}
                              for fold, value in outer.items()}}
    stages, stage_summary = ({}, {"status": "not_requested"}) if args.speech_consequences is None else decomposition(
        args.speech_consequences, held, probability_source)
    print(json.dumps({"stage": "fitting_fixed_models", "train_rows": len(train), "evaluation_rows": len(held),
                      "specification": cognitive.SPECIFICATION}, ensure_ascii=False), flush=True)
    with threadpool_limits(limits=1, user_api="blas"):
        models, predictions = fit_and_predict(train, held)
    source_files = (Path(__file__).resolve(), Path(cognitive.__file__).resolve())
    source = {"git_head": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
              "file_sha256": {str(path.relative_to(ROOT)): sha256_bytes(path.read_bytes()) for path in source_files}}
    results = {"scope": "development_factual_sequential_prediction", "fit_folds": development.SELECTION_FOLDS,
               "evaluation_fold": held_fold, "probability_source": probability_source,
               "fit_population": population(train, base_weights(train), np.ones(len(train), dtype=bool)),
               "population_by_fold": {str(fold): population(rows, base_weights(rows), np.ones(len(rows), dtype=bool))
                                      for fold, rows in by_fold.items()},
               "head_statuses": {m: {h: value["status"] for h, value in heads.items()} for m, heads in models["models"].items()},
               "stage_decomposition": stage_summary, "evaluation": summarize(held, predictions, replicates=BOOTSTRAP_REPLICATES),
               "bootstrap": {"unit": "game", "version": BOOTSTRAP_VERSION, "seed": frozen.BOOTSTRAP_SEED,
                             "replicates": BOOTSTRAP_REPLICATES}, "specification": cognitive.SPECIFICATION}
    saved = publish_artifact(DESTINATION, manifest_fields={
        "artifact_type": cognitive.ARTIFACT_TYPE, "schema_version": cognitive.ARTIFACT_TYPE,
        "artifact_identity": DESTINATION.name, **lineage, "source": source, "probability_source": probability_source,
    }, files={"models.json": canonical_json_bytes(models), "results.json": canonical_json_bytes(results),
              "labels.jsonl": canonical_jsonl_bytes(label_records(by_fold, predictions, stages))})
    print(json.dumps({"path": str(DESTINATION), "manifest_digest": saved.manifest_digest,
                      "head_statuses": results["head_statuses"]}), flush=True)


if __name__ == "__main__":
    main()
