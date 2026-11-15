"""Pure, versioned schedules and bootstrap draws frozen before training."""

import hashlib
import json
from collections import Counter

import numpy as np

from werewolf.artifact_io import canonical_json_bytes, sha256_bytes

SCHEDULE_VERSION = "classic7_balanced_seven_shift_v2"
ORDER_VERSION = "classic7_round_game_order_v2"
BOOTSTRAP_VERSION = "classic7_sha256_rejection_bootstrap_v1"
FIELD_ENCODING = "canonical_json_utf8_u64be_length_prefix_v1"
DEVELOPMENT_HOLDOUT_VERSION = "classic7_development_train4500_validation300_v1"
DEVELOPMENT_HOLDOUT_SEED = 20261115


def load_bootstrap_plan(experiment):
    raw = experiment.file("bootstrap/game_cluster_bootstrap_plan.manifest.json")
    metadata = json.loads(raw)
    if canonical_json_bytes(metadata) != raw:
        raise ValueError("noncanonical bootstrap metadata")
    data = experiment.file("bootstrap/game_cluster_bootstrap_indices.bin")
    expected = {"schema_version": BOOTSTRAP_VERSION, "parent_digest": experiment.manifest["protocol_digest"],
        "game_ids": experiment.manifest["game_ids"], "shape": [experiment.config.bootstrap_replicates, len(experiment.manifest["game_ids"])],
        "dtype": "<i4", "layout": "C", "seed": experiment.config.bootstrap_seed, "digest": sha256_bytes(data)}
    if metadata != expected:
        raise ValueError("bootstrap metadata/digest mismatch")
    indices = np.frombuffer(data, dtype="<i4").reshape(metadata["shape"])
    if np.any(indices < 0) or np.any(indices >= len(metadata["game_ids"])):
        raise ValueError("bootstrap index out of range")
    return indices, metadata["digest"]


def digest_fields(*values):
    encoded = [canonical_json_bytes(value) for value in values]
    return hashlib.sha256(b"".join(len(v).to_bytes(8, "big") + v for v in encoded)).digest()


def development_holdout_split(game_ids, *, development_game_ids):
    """仅按游戏身份生成固定划分，不读取标签、角色、Fold 或 Test。

    development_game_ids 必须来自既有 Development 读取器的 public_view；
    它是可信全集，不能用待检查的外部身份列表代替来源核验。
    排序键为 digest_fields(规则版本, 固定种子, Game ID)，并以 Game ID 破同序。
    """
    games = tuple(game_ids)
    development = tuple(development_game_ids)
    for ids in (development, games):
        if any(type(game_id) is not str or not game_id.strip() for game_id in ids):
            raise ValueError("Development game identities must be nonempty strings")
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate Development game identity")
    if len(development) != 4800:
        raise ValueError("holdout requires exactly 4800 verified Development games")
    if not set(games) <= set(development):
        raise ValueError("game identity is outside verified Development")
    if set(games) != set(development):
        raise ValueError("holdout must cover every verified Development game")
    ranked = sorted(games, key=lambda game_id: (
        digest_fields(DEVELOPMENT_HOLDOUT_VERSION, DEVELOPMENT_HOLDOUT_SEED, game_id),
        game_id,
    ))
    training = sorted(ranked[300:])
    validation = sorted(ranked[:300])
    return {
        "rule_version": DEVELOPMENT_HOLDOUT_VERSION,
        "seed": DEVELOPMENT_HOLDOUT_SEED,
        "field_encoding": FIELD_ENCODING,
        "development_game_ids_digest": sha256_bytes(canonical_json_bytes(sorted(development))),
        "training_game_ids": training,
        "validation_game_ids": validation,
        "training_game_ids_digest": sha256_bytes(canonical_json_bytes(training)),
        "validation_game_ids_digest": sha256_bytes(canonical_json_bytes(validation)),
    }


def training_schedule(schedule_seed, fold, game_ids, cycles, batch_size):
    """Control order/rotation only from frozen settings and training identities."""
    if type(schedule_seed) is not int or not 0 <= schedule_seed < 2**63:
        raise ValueError("schedule seed must be an explicit integer in 0..2**63-1")
    games = tuple(sorted(game_ids))
    if not games or len(set(games)) != len(games):
        raise ValueError("training games must be nonempty and unique")
    if any(type(n) is not int or n <= 0 for n in (cycles, batch_size)) or type(fold) is not int or not 0 <= fold < 5:
        raise ValueError("invalid training schedule configuration")
    batches, coefficients = _balanced_schedule(schedule_seed, fold, games, cycles, batch_size)
    return {"schedule_version": SCHEDULE_VERSION, "order_version": ORDER_VERSION,
            "field_encoding": FIELD_ENCODING, "schedule_seed": schedule_seed, "fold": fold,
            "rotation_cycles": cycles, "game_batch_size": batch_size, "optimizer_steps": len(batches),
            "batches": batches, "cumulative_game_coefficients": coefficients}


def _balanced_schedule(schedule_seed, lineage, games, cycles, batch_size):
    batches = []
    for cycle in range(cycles):
        offsets = {g: int.from_bytes(digest_fields(SCHEDULE_VERSION, schedule_seed, lineage, cycle, g), "big") % 7 for g in games}
        for round_index in range(7):
            ordered = sorted(games, key=lambda g: (digest_fields(ORDER_VERSION, schedule_seed, lineage, cycle, round_index, g), g))
            rows = [{"game_id": g, "shift": (offsets[g] + round_index) % 7, "cycle": cycle, "round": round_index} for g in ordered]
            batches.extend(rows[i:i+batch_size] for i in range(0, len(rows), batch_size))
    counts = Counter((r["game_id"], r["shift"]) for b in batches for r in b)
    if counts != Counter({(g, s): cycles for g in games for s in range(7)}):
        raise ValueError("incomplete seven-shift schedule")
    coefficients = {g: sum(1 / len(b) for b in batches if any(r["game_id"] == g for r in b)) for g in games}
    return batches, coefficients


def final_training_schedule(schedule_seed, game_ids, cycles, batch_size):
    """No partition: all development identities, paired across conditions."""
    if type(schedule_seed) is not int or not 0 <= schedule_seed < 2**63:
        raise ValueError("invalid schedule seed")
    games = tuple(sorted(game_ids))
    if not games or len(set(games)) != len(games) or any(type(n) is not int or n <= 0 for n in (cycles, batch_size)):
        raise ValueError("invalid final schedule")
    batches, coefficients = _balanced_schedule(schedule_seed, {"lifecycle": "classic7_final_fit_v1"}, games, cycles, batch_size)
    return {"schedule_version": "classic7_final_seven_shift_v1", "order_version": ORDER_VERSION,
            "field_encoding": FIELD_ENCODING, "schedule_seed": schedule_seed,
            "rotation_cycles": cycles, "game_batch_size": batch_size, "optimizer_steps": len(batches),
            "batches": batches, "cumulative_game_coefficients": coefficients}


def bootstrap_indices(game_ids, replicates, seed):
    if tuple(game_ids) != tuple(sorted(set(game_ids))) or not game_ids:
        raise ValueError("bootstrap game IDs must be canonical and unique")
    if type(replicates) is not int or replicates < 1:
        raise ValueError("bootstrap replicate count must be positive")
    n = len(game_ids)
    if n >= 2**31:
        raise ValueError("bootstrap game count exceeds canonical int32")
    limit = 2**256 - 2**256 % n
    draws = []
    counter = 0
    while len(draws) < replicates * n:
        value = int.from_bytes(digest_fields(BOOTSTRAP_VERSION, seed, list(game_ids), counter), "big")
        counter += 1
        if value < limit:
            draws.append(value % n)
    return np.array(draws, dtype="<i4").reshape(replicates, n)
