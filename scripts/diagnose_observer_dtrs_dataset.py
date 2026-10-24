"""Read-only D2a descriptive diagnostics; no model fitting or learned bin edges."""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import defaultdict
import json
import hashlib
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from werewolf.artifact_io import canonical_json_bytes, read_artifact_file, sha256_bytes
from werewolf.observer_dtrs.dataset import open_dataset
from werewolf.observer_dtrs.dataset_offline import open_supervision, verify_dataset


VERSION = "observer_dtrs_d2a_descriptive_diagnostics_v1"
SPLIT_DOMAIN = "observer-dtrs-d2-development-split-v1"
SPLIT_SEED = 20261014
STRATA = tuple(f"{p}/{n}" for p in ("discussion", "pk_discussion") for n in (1, 2))


def weighted_quantile(values, weights, probability):
    """Left inverse weighted empirical CDF; ties never split."""
    if not values:
        return None
    pairs = sorted(zip(values, weights, strict=True))
    target = probability * sum(weights)
    total = 0.0
    for value, weight in pairs:
        total += weight
        if total >= target:
            return value
    return pairs[-1][0]


def game_partition(game_ids, *, tom_fold, folds, parent_outer=None):
    """Label-free split preview; no seed search or strata/label stratification."""
    def rank(game_id):
        return sha256_bytes(canonical_json_bytes([SPLIT_DOMAIN, SPLIT_SEED,
            tom_fold, parent_outer, game_id])), game_id
    return {game: index % folds for index, game in enumerate(sorted(game_ids, key=rank))}


def historical_rng_seed(training_rng):
    """Audit the historical numerical seed without importing/running a trainer.

    Matches digest_fields(RNG_VERSION, binding['training_rng']) in training
    revision ebf98b7. A provenance field is a numerical input here, not just an ID.
    """
    parts = [canonical_json_bytes(value) for value in
             ("backbone_training_rng_v1", training_rng)]
    digest = hashlib.sha256(b"".join(len(part).to_bytes(8, "big") + part
                                     for part in parts)).digest()
    return int.from_bytes(digest, "big") % 2**32


def describe(rows, game_sizes):
    ids = {r.legal.context.game_id for r in rows}
    weights = [1 / game_sizes[r.legal.context.game_id] for r in rows]
    total = sum(weights)
    positive = [r.supervision.event_label_y for r in rows]
    values = [r.legal.features.r for r in rows]
    game_mass = defaultdict(float)
    for row, weight in zip(rows, weights, strict=True):
        game_mass[row.legal.context.game_id] += weight
    return {"rows": len(rows), "games": len(ids), "positive": sum(positive),
        "negative": len(rows) - sum(positive),
        "positive_games": len({r.legal.context.game_id for r in rows if r.supervision.event_label_y == 1}),
        "negative_games": len({r.legal.context.game_id for r in rows if r.supervision.event_label_y == 0}),
        "successful_empty": sum(not r.supervision.raw_suspicion_support for r in rows),
        "record_prevalence": sum(positive) / len(rows) if rows else None,
        "game_weighted_conditional_prevalence": sum(w * y for w, y in zip(weights, positive, strict=True)) / total if total else None,
        "game_equivalent_mass": total,
        "effective_games_kish": total ** 2 / sum(w ** 2 for w in game_mass.values()) if total else None,
        "r_quantiles_record": {str(p): weighted_quantile(values, [1] * len(rows), p)
            for p in (0, .01, .05, .1, .25, .5, .75, .9, .95, .99, 1)},
        "r_quantiles_game_weighted": {str(p): weighted_quantile(values, weights, p)
            for p in (0, .1, .25, .5, .75, .9, 1)}}


def stratum(row):
    f = row.legal.features
    return f"{f.phase}/{f.nW}"


def diagnose(path, *, expected_digest):
    audit = verify_dataset(path, expected_digest=expected_digest)
    view = open_dataset(path, expected_digest=expected_digest)
    rows = open_supervision(view)
    m = view.artifact.manifest
    capsule = json.loads(read_artifact_file(view.artifact, "restricted/upstream_manifests.json"))
    sizes = {g["game_id"]: g["row_count"] for g in m["games"]}
    folds = {f: {g["game_id"] for g in m["games"] if g["fold_index"] == f} for f in range(5)}
    prepared = {r["fold"]: r for r in capsule["preparation"]["folds"]}
    overlaps = [[len(folds[k] & set(prepared[l]["training_game_ids"])) for l in range(5)] for k in range(5)]
    report = {"schema_version": VERSION, "scope": "descriptive_only_not_model_validation",
        "dataset_digest": view.manifest_digest, "implementation_snapshot": m["implementation"],
        "historical_training_rng": {
            "interpretation": "provenance_hash_drives_numeric_rng_not_proof_of_optimistic_bias",
            "training_revision": "ebf98b7e3249f78d3f1fa1fd46c20affe0486913",
            "source_functions": ["werewolf/tom/backbone_execution.py:_binding",
                                 "werewolf/tom/backbone_execution.py:_reset_rng"],
            "folds": [{"fold": f,
                "training_rng": capsule[f"terminal/{f}"]["binding"]["training_rng"],
                "derived_numerical_seed": historical_rng_seed(
                    capsule[f"terminal/{f}"]["binding"]["training_rng"])}
                for f in range(5)]},
        "source_training_test_overlap_matrix": overlaps,
        "source_fold_checks": [{"fold": f, "held_out_games": len(prepared[f]["held_out_game_ids"]),
            "training_games": len(prepared[f]["training_game_ids"]),
            "held_out_training_overlap": len(set(prepared[f]["held_out_game_ids"]) & set(prepared[f]["training_game_ids"]))}
            for f in range(5)],
        "overall": describe(rows, sizes), "saved_audit_overall": audit["overall"],
        "game_sample_count_quantiles": {str(p): weighted_quantile(list(sizes.values()), [1] * len(sizes), p)
                                       for p in (0, .1, .25, .5, .75, .9, 1)},
        "by_phase_nW": {}, "by_tom_fold_phase_nW": {}, "fixed_r_histograms": {}}
    for state in STRATA:
        chosen = [r for r in rows if stratum(r) == state]
        report["by_phase_nW"][state] = describe(chosen, sizes)
        for f in range(5):
            report["by_tom_fold_phase_nW"][f"{f}/{state}"] = describe(
                [r for r in chosen if r.legal.context.game_id in folds[f]], sizes)
        buckets = [[] for _ in range(10)]
        cuts = [i / 10 for i in range(1, 10)]
        for row in chosen:
            buckets[bisect_right(cuts, row.legal.features.r)].append(row)
        report["fixed_r_histograms"][state] = [describe(bucket, sizes) for bucket in buckets]
    report["by_report"] = {kind: describe([r for r in rows if
        bool(r.supervision.raw_suspicion_support) == (kind == "nonempty")], sizes)
        for kind in ("empty", "nonempty")}
    assignments, checks = {}, {name: {state: [] for state in STRATA}
                               for name in ("outer_train", "outer_test", "inner_train", "inner_validation")}
    by_game = defaultdict(list)
    for row in rows:
        by_game[row.legal.context.game_id].append(row)

    def support(ids):
        grouped = defaultdict(list)
        for game in ids:
            for row in by_game[game]:
                grouped[stratum(row)].append(row)
        return {state: {"games": len({r.legal.context.game_id for r in grouped[state]}),
                        "rows": len(grouped[state]),
                        "negative": sum(r.supervision.event_label_y == 0 for r in grouped[state]),
                        "negative_games": len({r.legal.context.game_id for r in grouped[state]
                                               if r.supervision.event_label_y == 0})} for state in STRATA}

    def collect(kind, ids):
        stats = support(ids)
        for state in STRATA:
            checks[kind][state].append(stats[state])

    for f in range(5):
        outer = game_partition(folds[f], tom_fold=f, folds=5)
        assignments[str(f)] = outer
        for j in range(5):
            train = {g for g, part in outer.items() if part != j}
            test = folds[f] - train
            collect("outer_train", train)
            collect("outer_test", test)
            inner = game_partition(train, tom_fold=f, folds=3, parent_outer=j)
            for h in range(3):
                collect("inner_train", {g for g, part in inner.items() if part != h})
                collect("inner_validation", {g for g, part in inner.items() if part == h})
    report["split_preview"] = {"only_support_diagnosis": True, "domain": SPLIT_DOMAIN,
        "protocol_seed": SPLIT_SEED, "dataset_digest_used_as_assignment_input": False,
        "outer_assignment_digest": sha256_bytes(canonical_json_bytes(assignments)),
        "outer_splits": 25, "inner_splits": 75, "support_ranges": {}}
    for kind, states in checks.items():
        report["split_preview"]["support_ranges"][kind] = {state: {
            **{key + "_min_max": [min(r[key] for r in records), max(r[key] for r in records)]
               for key in ("games", "rows", "negative", "negative_games")},
            "zero_negative_splits": sum(r["negative"] == 0 for r in records)}
            for state, records in states.items()}
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--expected-dataset-digest", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(diagnose(args.dataset, expected_digest=args.expected_dataset_digest), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
