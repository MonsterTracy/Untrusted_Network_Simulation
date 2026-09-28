"""Read-only analysis of the completed formal paired ToM gameplay collection."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

from scripts import collect_games as operator
from scripts import tom_gameplay_collection as collection
from werewolf import cli


EXPECTED_FINAL_DIGEST = "067e1f18ccb9dc684fe720d740ee7b3206d227a7f3ab2d49a3d3ffc747252a18"
EXPECTED_ATTEMPTED = 42
EXPECTED_INVALID = {
    14: (collection.NOTOM, "GameplayGenerationExhausted"),
    15: (collection.TOM, "GameplayGenerationExhausted"),
}
BOOTSTRAP_SEED = 20260928
BOOTSTRAP_REPLICATES = 100000
_HEX = set("0123456789abcdef")


def _digest(value: object) -> bool:
    return type(value) is str and len(value) == 64 and set(value) <= _HEX


def _int(value: object, *, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def _plan(root: Path):
    record, artifact = collection._read_record(
        root / "experiment_plan", "tom_gameplay_experiment_plan"
    )
    if record != collection._experiment_record(
        record["source_revision"], record["runtime_provenance"],
        record["configured_call_limit"],
    ):
        raise ValueError("experiment plan differs from frozen protocol or seed pool")
    return record, artifact.manifest_digest


def _final(root: Path, plan: dict, plan_digest: str, expected_digest: str):
    record, artifact = collection._read_record(root / "final", "tom_gameplay_final")
    if artifact.manifest_digest != expected_digest:
        raise ValueError("formal final collection digest mismatch")
    required = {
        "experiment_id", "source_revision", "runtime_provenance",
        "experiment_plan_digest", "candidate_seed_pool_digest",
        "attempted_candidate_count", "completed_valid_count", "included_pairs",
        "invalid_pairs", "failure_reason_counts", "pair_record_digests",
    }
    if (
        set(record) != required
        or record["experiment_id"] != collection.EXPERIMENT_ID
        or record["source_revision"] != plan["source_revision"]
        or record["runtime_provenance"] != plan["runtime_provenance"]
        or record["experiment_plan_digest"] != plan_digest
        or record["candidate_seed_pool_digest"] != plan["candidate_seed_pool_digest"]
        or record["attempted_candidate_count"] != EXPECTED_ATTEMPTED
        or record["completed_valid_count"] != collection.TARGET_PAIRS
        or len(record["pair_record_digests"]) != EXPECTED_ATTEMPTED
        or len(record["included_pairs"]) != collection.TARGET_PAIRS
        or len(record["invalid_pairs"]) != len(EXPECTED_INVALID)
    ):
        raise ValueError("formal final collection is incomplete or inconsistent")
    return record, artifact.manifest_digest


def _check_arm(plan: dict, ordinal: int, seed: int, arm: str, status: dict):
    if type(status) is not dict or status.get("arm") != arm:
        raise ValueError("pair arm identity mismatch")
    if status.get("status") == "not_run":
        if status != {"arm": arm, "status": "not_run"}:
            raise ValueError("invalid not-run arm record")
        return
    arm_plan = collection._arm_plan(plan, ordinal, arm)
    if (
        status.get("collection_id") != arm_plan.collection_id
        or status.get("plan_digest") != arm_plan.plan_digest
        or status.get("source_revision") != plan["source_revision"]
        or status.get("runtime_provenance_digest") != arm_plan.runtime_provenance_digest
        or status.get("predictor_fit_digest") != (
            collection.FIT_DIGEST if arm == collection.TOM else None
        )
        or status.get("predictor_seal_digest") != (
            collection.SEAL_DIGEST if arm == collection.TOM else None
        )
        or not _digest(status.get("claim_digest"))
        or not _digest(status.get("terminal_digest"))
    ):
        raise ValueError("pair arm provenance or terminal identity mismatch")
    if status.get("status") == "success":
        expected_game_id = (
            f"{arm_plan.collection_id}-game-000000-seed-{seed}"
        )
        if (
            status.get("terminal_outcome") != "canonical_success"
            or status.get("game_id") != expected_game_id
            or status.get("winner") not in {"Werewolf", "Villager"}
            or not _digest(status.get("bundle_digest"))
            or not _digest(status.get("treatment_audit_digest"))
            or type(status.get("roles")) is not dict
            or not _int(status.get("backend_calls"))
            or not _int(status.get("speech_boundaries"))
            or not _int(status.get("treatment_count"))
            or (arm == collection.NOTOM and status["treatment_count"] != 0)
            or status.get("failure_stage") is not None
            or status.get("failure_type") is not None
        ):
            raise ValueError("invalid canonical-success arm record")
    elif status.get("status") == "failure":
        if (
            status.get("terminal_outcome") not in {
                "canonical_failure", "interrupted_failure"
            }
            or type(status.get("failure_stage")) is not str
            or type(status.get("failure_type")) is not str
            or status.get("winner") is not None
            or status.get("bundle_digest") is not None
        ):
            raise ValueError("invalid technical-failure arm record")
    else:
        raise ValueError("unknown pair arm status")


def _pairs(root: Path, plan: dict, final: dict):
    included = []
    valid_pairs = []
    invalid = []
    seen_ordinals = set()
    seen_seeds = set()
    reasons = Counter()
    for ordinal, reference in enumerate(final["pair_record_digests"]):
        seed = plan["ordered_seed_pool"][ordinal]
        if (
            type(reference) is not dict
            or set(reference) != {"candidate_ordinal", "seed", "pair_digest"}
            or reference["candidate_ordinal"] != ordinal
            or reference["seed"] != seed
            or not _digest(reference["pair_digest"])
            or ordinal in seen_ordinals
            or seed in seen_seeds
        ):
            raise ValueError("duplicate or reordered candidate/seed reference")
        seen_ordinals.add(ordinal)
        seen_seeds.add(seed)
        pair, artifact = collection._read_record(
            collection._pair_path(root, ordinal), "tom_gameplay_pair"
        )
        if artifact.manifest_digest != reference["pair_digest"]:
            raise ValueError("pair artifact digest differs from final manifest")
        if (
            pair.get("experiment_id") != collection.EXPERIMENT_ID
            or pair.get("candidate_ordinal") != ordinal
            or pair.get("seed") != seed
            or pair.get("candidate_seed_pool_digest") != plan["candidate_seed_pool_digest"]
            or pair.get("arm_execution_order") != list(collection._order(ordinal))
            or type(pair.get("arms")) is not dict
            or set(pair["arms"]) != set(collection.ARMS)
        ):
            raise ValueError("pair identity, seed, or arm order mismatch")
        for arm in collection.ARMS:
            _check_arm(plan, ordinal, seed, arm, pair["arms"][arm])
        expected = collection._pair_record(
            plan, ordinal, collection._order(ordinal), {
                arm: None if pair["arms"][arm]["status"] == "not_run"
                else pair["arms"][arm] for arm in collection.ARMS
            }
        )
        if pair != expected:
            raise ValueError("pair status or paired-role validation mismatch")
        if pair["status"] == "completed_valid":
            if (
                pair["roles_equal"] is not True
                or any(pair["arms"][arm]["status"] != "success"
                       for arm in collection.ARMS)
            ):
                raise ValueError("included pair lacks two valid arms")
            included.append({"candidate_ordinal": ordinal, "seed": seed})
            valid_pairs.append(pair)
        else:
            item = {
                "candidate_ordinal": ordinal, "seed": seed,
                "failure_arm": pair["failure_arm"],
                "failure_stage": pair["failure_stage"],
                "failure_type": pair["failure_type"],
                "pair_digest": artifact.manifest_digest,
            }
            invalid.append(item)
            reasons[f"{item['failure_stage']}:{item['failure_type']}"] += 1
    if (
        included != final["included_pairs"]
        or invalid != final["invalid_pairs"]
        or dict(reasons) != final["failure_reason_counts"]
        or len(included) != collection.TARGET_PAIRS
        or {item["candidate_ordinal"]: (
            item["failure_arm"], item["failure_type"]
        ) for item in invalid} != EXPECTED_INVALID
    ):
        raise ValueError("final inclusion or formal failure diagnostics mismatch")
    return valid_pairs, invalid


def _exact_mcnemar(n01: int, n10: int) -> float:
    discordant = n01 + n10
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(n01, n10) + 1))
    return min(1.0, 2.0 * tail / (1 << discordant))


def _percentile(sorted_values: list[float], fraction: float) -> float:
    index = (len(sorted_values) - 1) * fraction
    low = math.floor(index)
    high = math.ceil(index)
    return sorted_values[low] + (index - low) * (
        sorted_values[high] - sorted_values[low]
    )


def _paired_bootstrap(effects: list[int], *, seed: int = BOOTSTRAP_SEED,
                      replicates: int = BOOTSTRAP_REPLICATES) -> list[float]:
    if not effects or any(effect not in {-1, 0, 1} for effect in effects):
        raise ValueError("bootstrap requires binary paired effects")
    if not _int(replicates, minimum=1):
        raise ValueError("bootstrap replicate count must be positive")
    generator = random.Random(seed)
    size = len(effects)
    estimates = sorted(
        sum(effects[generator.randrange(size)] for _ in range(size)) / size
        for _ in range(replicates)
    )
    return [_percentile(estimates, 0.025), _percentile(estimates, 0.975)]


def analyze_collection(root: Path, *, expected_final_digest: str = EXPECTED_FINAL_DIGEST):
    root = Path(root)
    plan, plan_digest = _plan(root)
    final, final_digest = _final(root, plan, plan_digest, expected_final_digest)
    valid_pairs, invalid = _pairs(root, plan, final)
    outcomes = Counter()
    effects = []
    treatment_counts = []
    calls = {arm: [] for arm in collection.ARMS}
    speeches = {arm: [] for arm in collection.ARMS}
    for pair in valid_pairs:
        arms = pair["arms"]
        notom_win = arms[collection.NOTOM]["winner"] == "Werewolf"
        tom_win = arms[collection.TOM]["winner"] == "Werewolf"
        outcomes[int(notom_win), int(tom_win)] += 1
        effects.append(int(tom_win) - int(notom_win))
        treatment_counts.append(arms[collection.TOM]["treatment_count"])
        for arm in collection.ARMS:
            calls[arm].append(arms[arm]["backend_calls"])
            speeches[arm].append(arms[arm]["speech_boundaries"])
    n = len(valid_pairs)
    n00, n01, n10, n11 = (outcomes[a, b] for a, b in ((0, 0), (0, 1), (1, 0), (1, 1)))
    if n00 + n01 + n10 + n11 != n:
        raise ValueError("paired outcome counts do not sum to N")
    wins = {collection.NOTOM: n10 + n11, collection.TOM: n01 + n11}
    delta = (wins[collection.TOM] - wins[collection.NOTOM]) / n
    if delta != (n01 - n10) / n or delta != sum(effects) / n:
        raise ValueError("paired effect disagrees with discordant counts")
    arm_results = {}
    for arm in collection.ARMS:
        arm_results[arm] = {
            "wins": wins[arm], "losses": n - wins[arm], "win_rate": wins[arm] / n,
            "backend_calls": {
                "mean": statistics.mean(calls[arm]),
                "median": statistics.median(calls[arm]),
            },
            "speech_boundaries": {
                "mean": statistics.mean(speeches[arm]),
                "median": statistics.median(speeches[arm]),
            },
        }
    return {
        "experiment_id": collection.EXPERIMENT_ID,
        "source_collection_digest": final_digest,
        "N": n,
        "attempted_candidates": final["attempted_candidate_count"],
        "invalid_candidate_count": len(invalid),
        "invalid_candidates": [
            {key: item[key] for key in (
                "candidate_ordinal", "seed", "failure_arm", "failure_stage",
                "failure_type",
            )} for item in invalid
        ],
        "arms": arm_results,
        "paired": {
            "n00": n00, "n01": n01, "n10": n10, "n11": n11,
            "discordant_count": n01 + n10,
            "discordant_rate": (n01 + n10) / n,
            "delta_hat": delta,
            "bootstrap_95_ci": _paired_bootstrap(effects),
            "exact_mcnemar_p": _exact_mcnemar(n01, n10),
        },
        "treatment": {
            "zero_treatment_games": sum(count == 0 for count in treatment_counts),
            "positive_treatment_games": sum(count > 0 for count in treatment_counts),
            "total_treatments": sum(treatment_counts),
            "mean_treatments": statistics.mean(treatment_counts),
        },
        "analysis": {
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        },
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.collection_root is None:
        storage = cli._storage_root(operator.STORAGE)
        root = cli._artifact_path(
            storage, Path("gameplay_ablations") / collection.EXPERIMENT_ID
        )
    else:
        root = args.collection_root
    root = root.resolve(strict=True)
    output = (args.output_dir or root.parent / f"{collection.EXPERIMENT_ID}-analysis").resolve()
    if output == root or root in output.parents:
        raise ValueError("analysis output directory must be outside formal collection")
    result = analyze_collection(root, expected_final_digest=EXPECTED_FINAL_DIGEST)
    payload = (json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "analysis.json"
    try:
        with destination.open("xb") as handle:
            handle.write(payload)
    except FileExistsError:
        if destination.read_bytes() != payload:
            raise ValueError("existing analysis.json differs from reproducible result") from None
    sys.stdout.buffer.write(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
