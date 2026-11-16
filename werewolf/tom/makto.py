"""MaKTO 七人 Seer-Witch 外部代理标签；不进入 Canonical Development。"""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
from urllib.request import urlopen

import torch

from werewolf.artifact_io import (
    canonical_json_bytes, canonical_jsonl_bytes, publish_artifact,
    open_artifact_envelope, read_artifact_file, sha256_bytes,
)
from werewolf.canonical_collection.public_history import PLAYER_IDS, freeze_public_event_history
from werewolf.canonical_collection.speech import V1SpeechAction, V1_SPEECH_ONTOLOGY_VERSION
from werewolf.speech.speech_perceiver import SpeechPerceiver
from werewolf.structured_history import (
    STRUCTURED_TOKEN_PLANNER_VERSION, StructuredTokenDescriptor, StructuredTokenPlan,
    plan_public_history,
)
from werewolf.tom.dataset import (
    CanonicalToMSample, ExperimentCapacity, belief_target, tensorize_structured_plan,
)


REVISION = "7d7d516f780879dbe9f512ef29655f51548f7578"
PREPARED_DIGEST = "eadb9dc8c5d31a0594f734561c8144485c1f8407432dbebc928d050cb17be8ad"
EPOCH35_DIGEST = "6229037415e6aec74f3e615f3f4838f269a6ff1634b6bd9e83dbd291b9ef1609"
DATASET = "ReneeYe/werewolf_game_reasoning"
VERSION = "makto_seer_witch_proxy_belief_v1"
SPLITS = {"train": range(21, 41), "test": range(6, 14)}
# 只隔离 Train 审阅已经确认的冲突；不根据 Test 结果修改规则或补标签。
CONFLICTS = {
    ("train", 23, 34): "reason称1号狼人，role_prediction将1号标为seer",
    ("train", 40, 36): "reason称6号狼人，role_prediction将6号标为seer",
    ("train", 36, 33): "reason称2号seer，role_prediction将2号标为witch",
}
ROLES = {"werewolf", "simple_villager", "seer", "witch"}


def source_paths():
    return [f"raw/{split}/7_player_game/seer_witch/game_{game}/event_zh.json"
            for split, games in SPLITS.items() for game in games]


def download_sources():
    """固定版本及28个明确文件，无隐式缓存、模型下载或其他游戏读取。"""
    root = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/"
    return {path: urlopen(root + path, timeout=60).read() for path in source_paths()}


def speech_records(sources):
    """给标注者的视图只含公开发言及位置，不含角色或任何监督。"""
    rows = []
    for path, payload in sources.items():
        for index, event in enumerate(json.loads(payload)):
            if event["event"] == "speech":
                content = event["content"]
                rows.append({"key": path.rsplit("/", 1)[0] + f"/{index}",
                             "speaker": content["player"], "day": content["day"],
                             "text": content["context"],
                             "text_sha256": sha256_bytes(content["context"].encode())})
    return rows


def convert_label(content, *, quarantined=False):
    """候选角色到支持集的保守转换；真实角色不进入本函数。"""
    actor = content["player"]
    labels = content.get("role_prediction")
    reasons = []
    support = []
    values = []
    if not isinstance(labels, dict) or any(str(j) not in labels for j in range(1, 8) if j != actor):
        reasons.append("missing_target")
    else:
        values = [labels[str(j)] for j in range(1, 8) if j != actor]
        if any(not isinstance(v, list) or not v or len(v) != len(set(v))
               or set(v) - ROLES - {"NA"} for v in values):
            reasons.append("invalid_candidates")
        else:
            support = [f"player{j}" for j in range(1, 8)
                       if j != actor and labels[str(j)] == ["werewolf"]]
            if any("NA" in v for v in values):
                reasons.append("NA")
            if any("werewolf" in v and set(v) != {"werewolf"} for v in values):
                reasons.append("mixed_wolf_nonwolf")
            if not support:
                reasons.append("empty_support")
    if quarantined:
        reasons.append("confirmed_text_label_conflict")
    q, observed = belief_target(actor - 1, tuple(support) if not reasons else (),
                                "success" if not reasons else None)
    return {"raw_role_prediction": labels, "wolf_support": support,
            "label_usable": observed, "q": q.tolist(), "exclusion_reasons": reasons,
            "quarantined": quarantined}


def extract_game(path, payload, annotations):
    """公开事件与监督分别构造；投票阶段开始时冻结输入，不看当轮选票。"""
    split, game_number = path.split("/")[1], int(path.split("/")[-2].split("_")[1])
    game_id = f"makto/{split}/seer_witch/game_{game_number}"
    raw = json.loads(payload)
    roles = {e["content"]["player"]: e["content"]["role"] for e in raw if e["event"] == "roles"}
    if Counter(roles.values()) != Counter(werewolf=2, simple_villager=3, seer=1, witch=1):
        raise ValueError("not a seven-player Seer-Witch source")
    public, actions, boundaries, labels = [], {}, [], []
    alive, problems = set(range(1, 8)), []
    current_boundary = None
    day, phase, latest_speech_round = 0, "night", None

    def emit(kind, **fields):
        record = {"event_id": f"event-{len(public):06d}", "event_index": len(public),
                  "event_type": kind, **fields}
        public.append(record)
        return record["event_id"]

    def change(next_day, next_phase):
        nonlocal day, phase
        day, phase = next_day, next_phase
        emit("phase_change", day=day, phase=phase)

    for index, event in enumerate(raw):
        kind, c = event["event"], event.get("content")
        if kind == "cycle_round" and c["status"] == "night":
            change(c["round"] - 1, "night")
        elif kind == "cycle_round" and c["status"] == "day":
            text = c.get("context")
            if text is None:
                problems.append({"reason": "missing_public_death_announcement", "source_index": index})
            elif text == "今天是平安夜" or text.startswith("死亡玩家："):
                dead = sorted(set(map(int, re.findall(r"(\d+)号", text))))
                if any(j not in alive for j in dead):
                    raise ValueError("public death announcement repeats a dead seat")
                emit("death_announcement", dead_players=[f"player{j}" for j in dead])
                alive.difference_update(dead)
            else:
                raise ValueError("unrecognized public death announcement")
            change(c["round"], "discussion")
        elif kind == "speech":
            round_day, round_stage = map(int, c["day"].split("-"))
            if round_day != day:
                raise ValueError("speech day differs from public phase")
            if round_stage > 1 and phase in {"vote", "pk_vote"}:
                change(day, "pk_discussion")
            latest_speech_round = c["day"]
            speaker = f"player{c['player']}"
            emit("turn_start", speaker=speaker)
            event_id = emit("public_speech", speaker=speaker, raw_text=c["context"])
            key = path.rsplit("/", 1)[0] + f"/{index}"
            annotation = annotations[key]
            if annotation["text_sha256"] != sha256_bytes(c["context"].encode()):
                raise ValueError("speech annotation text identity mismatch")
            triples = SpeechPerceiver._normalize(parsed=annotation["actions"], speaker=c["player"])
            actions[event_id] = [V1SpeechAction(*triple) for triple in triples]
        elif kind == "vote_start":
            change(day, "pk_vote" if phase == "pk_discussion" else "vote")
            boundary_id = f"{game_id}/vote-start-{index}"
            snapshot = list(public)
            digest = sha256_bytes(canonical_json_bytes({"game_id": game_id,
                "boundary_id": boundary_id, "public_events": snapshot}))
            plan = None
            if not problems:
                plan = plan_public_history(freeze_public_event_history(snapshot), actions,
                                           boundary_digest=digest)
            current_boundary = {"game_id": game_id, "split": split,
                "boundary_id": boundary_id, "source_index": index,
                "day": day, "phase": phase, "prefix_digest": digest,
                "public_events": snapshot, "input_problems": list(problems),
                "alive": [j in alive for j in range(1, 8)] if not problems else None,
                "plan": plan.to_record() if plan else None}
            boundaries.append(current_boundary)
        elif kind == "voted":
            if current_boundary is None or c["day"] != latest_speech_round:
                raise ValueError("vote has no matching current public boundary")
            converted = convert_label(c, quarantined=(split, game_number, index) in CONFLICTS)
            actor = c["player"]
            nonwolf = roles[actor] != "werewolf"
            if not problems and actor not in alive:
                raise ValueError("a dead observer submitted a vote label")
            eligible = nonwolf and not problems
            reasons = list(converted["exclusion_reasons"])
            if not nonwolf:
                reasons.append("wolf_observer")
            reasons.extend(p["reason"] for p in problems)
            labels.append({"game_id": game_id, "split": split,
                "boundary_id": current_boundary["boundary_id"],
                "source_index": index, "source_sha256": sha256_bytes(payload),
                "observer_id": f"player{actor}", "day": c["day"],
                **converted, "non_wolf_observer": nonwolf,
                "quarantine_reason": CONFLICTS.get((split, game_number, index)),
                "primary_eligible": eligible,
                "effective": eligible and converted["label_usable"],
                "exclusion_reasons": list(dict.fromkeys(reasons))})
        elif kind == "vote_results":
            votes = [{"voter": f"player{k}", "target": None if v is None else f"player{v}"}
                     for k, v in sorted(c.items(), key=lambda pair: int(pair[0]))]
            emit("vote_result", votes=votes)
            counts = Counter(v for v in c.values() if v is not None)
            leaders = [j for j, n in counts.items() if n == max(counts.values())] if counts else []
            exiled = leaders if len(leaders) == 1 else []
            emit("exile_result", exiled_players=[f"player{j}" for j in sorted(exiled)])
            alive.difference_update(exiled)
            current_boundary = None
        # roles只供外围Mask；所有私人夜间行动、notes、summary、review、winner均无输入路径。
    return {"game_id": game_id, "split": split, "source_path": path,
            "source_sha256": sha256_bytes(payload), "public_events": public,
            "speech_actions": {key: [a.to_record() for a in value] for key, value in actions.items()}}, boundaries, labels


def prepare(sources, annotations):
    """一次准备，两组固定规则；原始文件与候选/排除审计全部保留。"""
    if set(sources) != set(source_paths()):
        raise ValueError("exactly the fixed 20 Train and 8 Test sources are required")
    expected = {row["key"] for row in speech_records(sources)}
    if set(annotations) != expected:
        raise ValueError("exact public speech annotation coverage is required")
    games, inputs, labels = [], [], []
    for path in source_paths():
        game, boundaries, rows = extract_game(path, sources[path], annotations)
        games.append(game)
        inputs.extend(boundaries)
        labels.extend(rows)
    report = {}
    for split in SPLITS:
        group = [g for g in games if g["split"] == split]
        rows = [row for row in labels if row["split"] == split]
        primary = [row for row in rows if row["non_wolf_observer"]]
        boundary_rows = [row for row in inputs if row["split"] == split]
        report[split] = {
            "games": len(group), "candidate_vote_rows": len(rows),
            "non_wolf_candidate_rows": len(primary),
            "label_usable_rows": sum(row["label_usable"] for row in primary),
            "quarantined_rows": sum(row["quarantined"] for row in primary),
            "effective_rows": sum(row["effective"] for row in rows),
            "effective_games": len({row["game_id"] for row in rows if row["effective"]}),
            "vote_boundaries": len(boundary_rows),
            "usable_input_boundaries": sum(row["plan"] is not None for row in boundary_rows),
            "public_speeches": sum(len(g["speech_actions"]) for g in group),
            "public_events": sum(len(g["public_events"]) for g in group),
            "speech_actions": sum(len(a) for g in group for a in g["speech_actions"].values()),
            "max_input_tokens": max((row["plan"]["token_count"] for row in boundary_rows if row["plan"]), default=0),
            "exclusion_counts_nonexclusive": dict(Counter(reason for row in primary for reason in row["exclusion_reasons"])),
            "label_exclusion_counts_exclusive": dict(Counter(
                row["exclusion_reasons"][0] for row in primary if not row["label_usable"])),
            "per_game": [{
                "game_id": game["game_id"],
                "candidate_vote_rows": sum(row["game_id"] == game["game_id"] for row in rows),
                "effective_rows": sum(row["game_id"] == game["game_id"] and row["effective"] for row in rows),
            } for game in group],
        }
    if any(row["plan"] and row["plan"]["token_count"] > 1024 for row in inputs):
        raise ValueError("complete public input exceeds model capacity; truncation forbidden")
    return games, inputs, labels, report


def publish(destination, sources, annotations, *, previous=None, corrections=()):
    games, inputs, labels, report = prepare(sources, annotations)
    revision_fields, revision_files = {}, {}
    if previous is not None:
        parent = open_artifact_envelope(previous, expected_artifact_type="makto_proxy_belief",
                                        expected_schema_version=VERSION)
        if parent.manifest_digest != PREPARED_DIGEST:
            raise ValueError("annotation correction parent identity mismatch")
        if parent.manifest["source_digests"] != {p: sha256_bytes(b) for p, b in sources.items()}:
            raise ValueError("annotation correction cannot change original sources")
        old = {r["key"]: r for r in map(json.loads, read_artifact_file(parent, "annotations.jsonl").splitlines())}
        changed = {k for k in annotations if old[k]["actions"] != annotations[k]["actions"]}
        if len({c["key"] for c in corrections}) != len(corrections) or changed != {c["key"] for c in corrections}:
            raise ValueError("correction log must cover exactly the changed annotations")
        for c in corrections:
            if (c["before"] != old[c["key"]]["actions"] or c["after"] != annotations[c["key"]]["actions"]
                    or c["text_sha256"] != old[c["key"]]["text_sha256"] or not c["reason"] or not c["evidence"]):
                raise ValueError("correction evidence identity mismatch")
        for split in SPLITS:
            if canonical_jsonl_bytes(r for r in labels if r["split"] == split) != read_artifact_file(parent, f"supervision/{split}.jsonl"):
                raise ValueError("annotation correction cannot change supervision")
            old_inputs = [json.loads(line) for line in read_artifact_file(parent, f"public/{split}.jsonl").splitlines()]
            new_inputs = [r for r in inputs if r["split"] == split]
            if ([{k: v for k, v in r.items() if k != "plan"} for r in old_inputs]
                    != [{k: v for k, v in r.items() if k != "plan"} for r in new_inputs]):
                raise ValueError("annotation correction cannot change public boundaries")
        revision_fields = {"annotation_correction": {
            "parent_digest": parent.manifest_digest, "changed_speeches": len(corrections),
            "scope": "targeted_V1_public_speech_semantics_not_full_independent_review",
            "rules_selected_on": "original_train_only", "official_test_rule_selection": False,
            "supervision_and_boundaries": "unchanged",
            "v1_parser_source_digest": sha256_bytes(Path(__file__).parents[1].joinpath("speech/speech_perceiver.py").read_bytes()),
        }}
        revision_files = {"corrections.jsonl": canonical_jsonl_bytes(sorted(corrections, key=lambda c: c["key"]))}
    elif corrections:
        raise ValueError("correction log requires the original artifact")
    package = Path(__file__).parents[1]
    return publish_artifact(destination, manifest_fields={
        "artifact_type": "makto_proxy_belief", "schema_version": VERSION,
        "dataset": DATASET, "revision": REVISION, "license": "MIT (dataset card)",
        "target": "human_expert_role_judgment_proxy_not_verified_private_belief",
        "label_rule": "six_complete_no_NA_no_wolf_nonwolf_mix_nonempty_support_uniform",
        "boundary": "before_current_vote_results_at_vote_start",
        "annotation_origin": "Codex_public_text_V1_annotation_not_independently_reviewed",
        "speech_ontology": V1_SPEECH_ONTOLOGY_VERSION,
        "token_planner": STRUCTURED_TOKEN_PLANNER_VERSION,
        "label_timing": "UNKNOWN", "report": report,
        "implementation_digests": {name: sha256_bytes((package / name).read_bytes())
                                   for name in ("tom/makto.py", "tom/dataset.py", "structured_history.py")},
        "source_digests": {p: sha256_bytes(b) for p, b in sources.items()},
        **revision_fields,
    }, files={
        **sources,
        **revision_files,
        "public/games.jsonl": canonical_jsonl_bytes(games),
        **{f"public/{split}.jsonl": canonical_jsonl_bytes(row for row in inputs if row["split"] == split)
           for split in SPLITS},
        **{f"supervision/{split}.jsonl": canonical_jsonl_bytes(row for row in labels if row["split"] == split)
           for split in SPLITS},
        "annotations.jsonl": canonical_jsonl_bytes({"key": k, **v} for k, v in sorted(annotations.items())),
        "report.json": canonical_json_bytes(report),
    })


class MaKTOToMDataset:
    """只打开指定划分的公开输入与标签；不读取原始角色和私人事件。"""

    def __init__(self, path, split, *, capacity=ExperimentCapacity(1024)):
        if split not in SPLITS:
            raise ValueError("split must be train or test")
        envelope = open_artifact_envelope(path, expected_artifact_type="makto_proxy_belief", expected_schema_version=VERSION)
        if envelope.manifest["revision"] != REVISION:
            raise ValueError("MaKTO revision mismatch")
        inputs = [json.loads(line) for line in read_artifact_file(envelope, f"public/{split}.jsonl").splitlines()]
        labels = [json.loads(line) for line in read_artifact_file(envelope, f"supervision/{split}.jsonl").splitlines()]
        self.samples, self.primary_masks = [], {}
        self.split = split
        self.source_digest = envelope.manifest_digest
        self.annotation_correction = envelope.manifest.get("annotation_correction")
        boundaries = set()
        for row in inputs:
            if row["split"] != split:
                raise ValueError("input split identity mismatch")
            if row["boundary_id"] in boundaries:
                raise ValueError("duplicate vote boundary")
            boundaries.add(row["boundary_id"])
            if row["plan"] is None:
                continue
            value = row["plan"]
            if (value["prefix_digest"] != row["prefix_digest"]
                    or value["token_count"] != len(value["tokens"])
                    or sha256_bytes(canonical_json_bytes({k: v for k, v in value.items()
                                                         if k != "plan_digest"})) != value["plan_digest"]):
                raise ValueError("structured plan identity mismatch")
            plan = StructuredTokenPlan(value["planner_version"], value["prefix_digest"],
                tuple(StructuredTokenDescriptor(**token) for token in value["tokens"]),
                value["token_count"], value["plan_digest"])
            q, observed, primary = torch.zeros(7, 7), torch.zeros(7, dtype=torch.bool), torch.zeros(7, dtype=torch.bool)
            seats = set()
            for label in labels:
                if label["boundary_id"] != row["boundary_id"]:
                    continue
                if label["split"] != split or label["game_id"] != row["game_id"]:
                    raise ValueError("label identity mismatch")
                seat = PLAYER_IDS.index(label["observer_id"])
                if seat in seats:
                    raise ValueError("duplicate observer at vote boundary")
                seats.add(seat)
                expected_q, _ = belief_target(seat, tuple(label["wolf_support"]) if label["label_usable"] else (),
                                              "success" if label["label_usable"] else None)
                if label["q"] != expected_q.tolist():
                    raise ValueError("proxy target does not match conservative label conversion")
                q[seat] = expected_q
                observed[seat], primary[seat] = label["label_usable"], label["primary_eligible"]
            self.primary_masks[row["boundary_id"]] = primary
            self.samples.append(CanonicalToMSample(row["game_id"], row["boundary_id"],
                row["prefix_digest"], plan.plan_digest, plan.token_count,
                tensorize_structured_plan(plan, capacity), q, observed, torch.tensor(row["alive"])))
        self.effective_game_ids = tuple(sorted({sample.game_id for sample in self.samples
                                               if self.effective_mask(sample).any()}))

    def effective_mask(self, sample):
        """部分监督只作用于非狼人、存活且标签通过规则的观察者。"""
        return self.primary_masks[sample.boundary_id] & sample.label_observed & sample.observer_alive

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def internal_split(data):
    """仅按Train的监督行数分层，再用固定哈希一次选定整局Validation。"""
    if data.split != "train":
        raise ValueError("internal split must use original Train only")
    counts = Counter()
    for sample in data.samples:
        counts[sample.game_id] += int(data.effective_mask(sample).sum())
    strata = {
        "one_or_two": ([g for g, n in counts.items() if 1 <= n <= 2], 2),
        "three": ([g for g, n in counts.items() if n == 3], 1),
        "four_or_more": ([g for g, n in counts.items() if n >= 4], 1),
    }
    validation = []
    for name, (games, take) in strata.items():
        if len(games) < take:
            raise ValueError("insufficient supervised games for the fixed split strata")
        ordered = sorted(games, key=lambda g: sha256_bytes(
            canonical_json_bytes(["makto_internal_holdout", 42, name, g])))
        validation.extend(ordered[:take])
    record = {
        "schema_version": "makto_internal_holdout_v1", "dataset_digest": data.source_digest,
        "seed": 42, "selection": "sha256_within_effective_row_count_strata_2_1_1",
        "training_game_ids": sorted(g for g, n in counts.items() if n and g not in validation),
        "validation_game_ids": sorted(validation),
        "unsupervised_game_ids": sorted(g for g, n in counts.items() if not n),
        "effective_rows_by_game": dict(sorted(counts.items())),
        "official_test_used": False,
    }
    return {**record, "split_digest": sha256_bytes(canonical_json_bytes(record))}


def validation_scores(data, split, model=None):
    """同一Mask逐行KL、局内平均后游戏等权；不更新模型或优化器。"""
    from werewolf.tom.scoring import row_kl

    if split != internal_split(data):
        raise ValueError("internal split identity mismatch")
    totals = {g: {"rows": 0, "uniform": 0., "model": 0.}
              for g in split["validation_game_ids"]}
    identity = []
    was_training = model.training if model is not None else None
    try:
        if model is not None:
            model.eval()
        with torch.inference_mode():
            for sample in data.samples:
                if sample.game_id not in totals:
                    continue
                mask = data.effective_mask(sample)
                if not mask.any():
                    continue
                q = sample.q.double()
                if not torch.allclose(q.sum(-1)[mask], torch.ones_like(q.sum(-1)[mask])):
                    raise ValueError("effective proxy target must sum to one")
                log_uniform = torch.full((7, 7), -math.log(6), dtype=torch.float64)
                log_uniform.fill_diagonal_(-torch.inf)
                result = totals[sample.game_id]
                result["rows"] += int(mask.sum())
                result["uniform"] += float(row_kl(log_uniform, q)[mask].sum())
                if model is not None:
                    public = {name: value[:sample.token_count].unsqueeze(0)
                              for name, value in sample.public.kwargs().items()}
                    logp = model(**public).squeeze(0).double()
                    result["model"] += float(row_kl(logp, q)[mask].sum())
                identity.extend({"game_id": sample.game_id, "boundary_id": sample.boundary_id,
                                 "prefix_digest": sample.prefix_digest, "plan_digest": sample.plan_digest,
                                 "observer_id": PLAYER_IDS[i]}
                                for i in mask.nonzero().flatten().tolist())
    finally:
        if model is not None:
            model.train(was_training)
    games = [{"game_id": g, "effective_rows": v["rows"],
              "uniform_kl": v["uniform"] / v["rows"],
              "epoch35_kl": v["model"] / v["rows"] if model is not None else None}
             for g, v in totals.items()]
    uniform = math.fsum(g["uniform_kl"] for g in games) / len(games)
    original = math.fsum(g["epoch35_kl"] for g in games) / len(games) if model is not None else None
    difference = original - uniform if original is not None else None
    return {"aggregation": "game_macro", "direction": "KL(q||p)", "unit": "nats",
            "row_identity_digest": sha256_bytes(canonical_json_bytes(sorted(
                identity, key=lambda r: (r["game_id"], r["boundary_id"], r["observer_id"])))),
            "games": games, "uniform_kl": uniform, "epoch35_kl": original,
            "absolute_difference_epoch35_minus_uniform": difference,
            "relative_difference": difference / uniform if difference is not None and uniform else None}


def load_epoch35_model(checkpoint_path, temporal_path):
    """只读原模型容器及绑定的时间表；不读取或恢复旧优化器、RNG和AI数据。"""
    from werewolf.artifact_io.tensor_state import decode_tensor_container
    from werewolf.tom.qwen3_large import Qwen3ToMLarge, graph_identity
    from werewolf.tom.state import restore_model_tensors
    from werewolf.tom.temporal import TemporalCodeProvider, TEMPORAL_VERSION

    checkpoint = open_artifact_envelope(checkpoint_path,
        expected_artifact_type="qwen3_large_training_checkpoint",
        expected_schema_version="classic7_qwen3_large_epoch_training_v1")
    m = checkpoint.manifest
    if (checkpoint.manifest_digest != EPOCH35_DIGEST or m["completed_epoch"] != 35
            or m["selection"]["best_epoch"] != 35 or m["binding"]["graph"] != graph_identity()):
        raise ValueError("Epoch 35 checkpoint digest/epoch/graph mismatch")
    temporal_envelope = open_artifact_envelope(temporal_path,
        expected_artifact_type="deterministic_temporal_code", expected_schema_version=TEMPORAL_VERSION,
        manifest_name="phase_codebook.manifest.json")
    if temporal_envelope.manifest_digest != m["binding"]["temporal_artifact_digest"]:
        raise ValueError("Epoch 35 temporal artifact identity mismatch")
    temporal = TemporalCodeProvider.explicit(temporal_path, temporal_envelope.manifest["parent_digest"])
    with torch.device("cpu"):
        model = Qwen3ToMLarge(temporal,
                             initialization_seed=m["binding"]["protocol"]["initialization_seed"])
    tensors = decode_tensor_container(m["containers"]["model"],
                                     read_artifact_file(checkpoint, "model.bin"), payload_path="model.bin")
    restore_model_tensors(model, tensors)
    return model.eval()


def _write_record(path, record):
    """小型实验记录只写一次；不向已有数据Artifact内部增加文件。"""
    path = Path(path).absolute()
    if any((parent / "manifest.json").exists() for parent in path.parents):
        raise ValueError("record cannot be written inside an immutable artifact")
    payload = canonical_json_bytes(record)
    if path.exists():
        if path.read_bytes() == payload:
            return
        raise FileExistsError("different experiment record already exists")
    with path.open("xb") as handle:
        handle.write(payload)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--split-record", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--temporal", type=Path)
    parser.add_argument("--expected-data-digest")
    args = parser.parse_args(argv)
    if args.baseline:
        if args.annotations is not None or args.data is None or args.split_record is None:
            parser.error("baseline requires --data/--split-record and forbids --annotations")
        if bool(args.checkpoint) != bool(args.temporal):
            parser.error("--checkpoint and --temporal must be supplied together")
        if args.output.absolute() == args.split_record.absolute():
            parser.error("split and baseline records must use different paths")
        data = MaKTOToMDataset(args.data, "train")
        if data.source_digest != (args.expected_data_digest or PREPARED_DIGEST):
            raise ValueError("prepared MaKTO artifact digest mismatch")
        split = internal_split(data)
        _write_record(args.split_record, split)
        model = load_epoch35_model(args.checkpoint, args.temporal) if args.checkpoint else None
        result = {"schema_version": "makto_validation_baseline_v1", "dataset_digest": data.source_digest,
                  "split_digest": split["split_digest"], "expected_checkpoint_digest": EPOCH35_DIGEST,
                  "checkpoint_digest": EPOCH35_DIGEST if model is not None else None,
                  "epoch35_status": "PASS" if model is not None else "NOT_RUN",
                  "epoch35_not_run_reason": None if model is not None else "checkpoint_and_temporal_paths_not_provided",
                  "evaluation_code_digest": sha256_bytes(Path(__file__).read_bytes()),
                  "quality_status": ("TARGETED_CORRECTIONS_APPLIED_NOT_FULL_CERTIFICATION"
                                     if getattr(data, "annotation_correction", None) else "ANNOTATION_SEMANTICS_REVIEW_REQUIRED"),
                  "interpretation": "diagnostic_on_existing_frozen_proxy_data",
                  **validation_scores(data, split, model)}
        _write_record(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.annotations is None or any((args.data, args.split_record, args.checkpoint, args.temporal, args.expected_data_digest)):
        parser.error("extraction requires --annotations and forbids baseline-only arguments")
    annotations = {}
    for line in args.annotations.read_bytes().splitlines():
        row = json.loads(line)
        key = row.pop("key")
        if key in annotations:
            raise ValueError("duplicate annotation key")
        annotations[key] = row
    artifact = publish(args.output, download_sources(), annotations)
    print(json.dumps({"path": str(artifact.path), "digest": artifact.manifest_digest,
                      "report": artifact.manifest["report"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
