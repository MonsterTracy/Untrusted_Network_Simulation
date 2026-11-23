"""生成独立 Test 的公开 Q，并在回读核验后评价固定报告事件模型。"""

import argparse
from collections import Counter
import io
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))

from werewolf.artifact_io import (  # noqa: E402
    canonical_json_bytes, open_artifact_envelope, publish_artifact, read_artifact_file, sha256_bytes,
)
from werewolf.canonical_collection.game_bundle import _prefix_from_record  # noqa: E402
from werewolf.canonical_collection.public_history import PLAYER_IDS  # noqa: E402


DATA = Path("/data/yuxiao/Untrusted_Network_Simulation")
COLLECTION = DATA / "canonical/paper-canonical-qwen35-9b-5800-v1"
SPLIT = DATA / "splits/paper-canonical-qwen35-9b-5800-v1-release-v1"
SPLIT_DIGEST = "edc271676649d9b23e01f505f0f23deb38fdbd0b0ab0ad65753a90cd562ee6e8"
TEST_DIGEST = "a427d80344cf90eb73ab946cc0d5e642a11c5f7c2806fcebdfd9d05873d6b42b"
TEST_GAMES = 1000
CAPSULE = DATA / "observer-dtrs/final-tom-cpu-c10b459fd6ea06a49c2cea5c58bae0c08e54cdfd/capsule"
CAPSULE_DIGEST = "001ca7cc0b7a5cb5c0a77500c1a4d2fa33d673617c3714e022f726565b56bf35"
CAPSULE_SCOPE = "PINNED_FINAL_INFERENCE_UNQUALIFIED"
SNAPSHOT = DATA / "tom/qual/run-01/source"
SNAPSHOT_DIGEST = "f204ffd81d72de2d8471a0e1a4a3df2e35a6f147d90888db4635106649e03f7c"
SNAPSHOT_TYPE = "observer_dtrs_qual_source_snapshot_v1"
LOADER_DIGEST = "bd80bdbe10dcd005b708018f910835b59350e71be2707ffbb68651b250a37fd2"
RUNTIME = {
    "python": "3.10.21", "platform": "Linux-6.14.0-29-generic-x86_64-with-glibc2.39",
    "packages": {"numpy": "2.2.6", "torch": "2.12.1+cu130", "transformers": "5.16.1"},
    "device": "cpu", "dtype": "float32", "attention_backend": "eager",
    "torch_threads": 1, "torch_interop_threads": 1, "deterministic_algorithms": False, "cuda_build": "13.0",
}
Q_PATH = DATA / "dtrs/final-tom-test-q"
RESULT_PATH = DATA / "dtrs/report-event-test"
Q_TYPE = "observer_dtrs_independent_test_q"
RESULT_TYPE = "observer_dtrs_report_event_test"
DECISION_DIGEST = "3556ea7b7e4eef9004a8ae97efd224a8ea1d89641726fabb7ae95878633b3cf1"
PRE_FIELDS = ("game_id", "boundary_id", "prefix_digest", "step_index")
Q_FIELDS = {*PRE_FIELDS, "plan_digest", "token_count", "q"}


def read_line(raw):
    value = json.loads(raw)
    if canonical_json_bytes(value) + b"\n" != raw or not isinstance(value, dict):
        raise ValueError("expected a canonical JSONL object")
    return value


def write_line(stream, value):
    stream.write(canonical_json_bytes(value) + b"\n")
    stream.flush()


def capsule_worker():
    """子进程只在冻结项目路径下导入 ToM 和 PRE 读取接口。"""
    import torch
    from werewolf.observer_dtrs.final_tom_capsule import CapsulePredictor

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    started = time.monotonic()
    predictor = CapsulePredictor(CAPSULE, expected_manifest_digest=CAPSULE_DIGEST,
                                 expected_scope=CAPSULE_SCOPE, device="cpu")
    load_seconds = time.monotonic() - started
    identity = predictor.identity
    model = predictor._model
    if (identity["runtime"] != RUNTIME or model.training or torch.cuda.is_initialized()
            or model.transformer.config._attn_implementation != "eager"
            or any(p.device.type != "cpu" or p.dtype != torch.float32 or p.requires_grad for p in model.parameters())
            or any(not Path(module.__file__).resolve().is_relative_to(SNAPSHOT / "sources")
                   for name, module in sys.modules.items()
                   if name == "werewolf" or name.startswith("werewolf."))):
        raise ValueError("unexpected Capsule runtime or import origin")
    out = sys.stdout.buffer
    write_line(out, {"capsule": identity, "seats": PLAYER_IDS, "loads": 1, "load_seconds": load_seconds,
                     "python_executable": str(Path(sys.executable).resolve())})
    count = 0
    for raw in sys.stdin.buffer:
        prefix = _prefix_from_record(read_line(raw))
        try:
            q = predictor.predict(prefix)
        except Exception as exc:
            raise ValueError(f"frozen inference failed at {prefix.game_id}/{prefix.boundary_id}") from exc
        write_line(out, {**{name: getattr(q, name) for name in PRE_FIELDS[:-1]}, "step_index": prefix.step_index,
                         "plan_digest": q.plan_digest, "token_count": q.token_count, "q": q.probabilities})
        count += 1
    if torch.cuda.is_initialized():
        raise ValueError("unexpected CUDA context")
    usage = resource.getrusage(resource.RUSAGE_SELF)
    write_line(out, {"predictions": count, "cpu_user_seconds": usage.ru_utime,
                     "cpu_system_seconds": usage.ru_stime, "peak_rss_kib": usage.ru_maxrss})


def start_worker():
    command = (
        "import runpy, sys, sysconfig\n"
        "sys.path[:0] = [sys.argv[2], *dict.fromkeys(sysconfig.get_paths()[k] for k in ('purelib', 'platlib'))]\n"
        "runpy.run_path(sys.argv[1])['capsule_worker']()"
    )
    return subprocess.Popen(
        [sys.executable, "-I", "-B", "-S", "-u", "-c", command, str(Path(__file__).resolve()),
         str(SNAPSHOT / "sources")],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
    )


def open_test():
    from werewolf.canonical_split import open_canonical_split

    split = open_canonical_split(SPLIT, expected_digest=SPLIT_DIGEST)
    members = tuple(sorted(split.members("independent_test"), key=lambda row: row["game_id"]))
    if (split.manifest["independent_test_identity_digest"] != TEST_DIGEST
            or len({row["game_id"] for row in members}) != len(members) or len(members) != TEST_GAMES):
        raise ValueError("Independent Test identity or membership mismatch")
    return split, members


def public_games(split, members):
    from werewolf.canonical_collection.game_bundle import open_split_game_bundle, read_bundle_pre_prefixes

    for member in members:
        bundle = open_split_game_bundle(COLLECTION, split.plan, member)
        yield bundle, read_bundle_pre_prefixes(bundle)


def validate_q(record, prefix):
    from werewolf.artifact_io.canonical import require_sha256
    from werewolf.observer_dtrs.probabilities import ProbabilityMatrix

    identity = tuple(getattr(prefix, name) for name in PRE_FIELDS)
    if (record.keys() != Q_FIELDS or tuple(record[name] for name in PRE_FIELDS) != identity
            or type(record["step_index"]) is not int):
        raise ValueError(f"Q identity mismatch for {prefix.boundary_id}")
    require_sha256(record["plan_digest"], "plan_digest")
    if type(record["token_count"]) is not int or not 1 <= record["token_count"] <= 1024:
        raise ValueError(f"invalid frozen Token count for {prefix.boundary_id}")
    ProbabilityMatrix(*identity[:3], record["q"])


def source_identity():
    return {
        "git_head": subprocess.check_output(
            ["git", "--no-optional-locks", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "script_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "test_sha256": sha256_bytes((ROOT / "tests/observer_dtrs/test_report_event_test.py").read_bytes()),
    }


def test_binding():
    return {"split_manifest_digest": SPLIT_DIGEST, "independent_test_identity_digest": TEST_DIGEST,
            "capsule_digest": CAPSULE_DIGEST, "capsule_scope": CAPSULE_SCOPE, "source_snapshot_digest": SNAPSHOT_DIGEST,
            "loader_implementation_digest": LOADER_DIGEST, "seat_order": list(PLAYER_IDS)}


def verify_worker(header, snapshot):
    capsule = header["capsule"]
    loader = capsule["loader_source"]["loader"]
    if (header["seats"] != list(PLAYER_IDS) or header["loads"] != 1
            or (capsule["capsule_digest"], capsule["scope"], capsule["runtime"]) !=
            (CAPSULE_DIGEST, CAPSULE_SCOPE, RUNTIME) or loader["implementation_digest"] != LOADER_DIGEST):
        raise ValueError("Capsule worker identity mismatch")
    for path, digest in loader["file_sha256"].items():
        if snapshot.manifest["file_table"][f"sources/{path}"]["sha256"] != digest:
            raise ValueError(f"Capsule loader differs from frozen snapshot: {path}")


def require_new(path):
    if os.path.lexists(path) or path.parent.resolve(strict=True) != path.parent:
        raise ValueError(f"expected an unused canonical output path: {path}")


def publish_and_readback(path, kind, fields, name, payload):
    artifact = publish_artifact(path, manifest_fields={"artifact_type": kind, "schema_version": kind,
                                                       "artifact_identity": path.name, **fields},
                                files={name: payload})
    readback = open_artifact_envelope(path, expected_artifact_type=kind, expected_schema_version=kind)
    if readback.manifest_digest != artifact.manifest_digest or read_artifact_file(readback, name) != payload:
        raise ValueError(f"published {kind} failed readback")
    return {"path": str(path), "manifest_digest": readback.manifest_digest,
            f"{Path(name).stem}_sha256": readback.manifest["file_table"][name]["sha256"], "readback": "PASS"}


def predict():
    require_new(Q_PATH)
    started = last_report = time.monotonic()
    split, members = open_test()
    source = source_identity()
    snapshot = open_artifact_envelope(SNAPSHOT, expected_artifact_type=SNAPSHOT_TYPE,
                                      expected_schema_version=SNAPSHOT_TYPE)
    if snapshot.manifest_digest != SNAPSHOT_DIGEST:
        raise ValueError("unexpected inference source snapshot")
    payload = io.BytesIO()
    bundles = {}
    count = 0
    with start_worker() as worker:
        try:
            header = read_line(worker.stdout.readline())
            verify_worker(header, snapshot)
            print(json.dumps({"stage": "capsule_loaded", **header}), flush=True)
            for bundle, prefixes in public_games(split, members):
                for prefix in prefixes:
                    write_line(worker.stdin, prefix.to_record())
                    raw = worker.stdout.readline()
                    validate_q(read_line(raw), prefix)
                    payload.write(raw)
                    count += 1
                bundles[bundle.manifest["game_id"]] = bundle.manifest_digest
                now = time.monotonic()
                if now - last_report >= 55:
                    print(json.dumps({"stage": "predicting", "games": len(bundles), "Q": count,
                                      "wall_seconds": now - started}), flush=True)
                    last_report = now
            worker.stdin.close()
            usage = read_line(worker.stdout.readline())
            if worker.stdout.read(1) or worker.wait() != 0 or usage["predictions"] != count:
                raise ValueError("Capsule worker did not finish the complete public PRE stream")
        except BaseException:
            if worker.poll() is None:
                worker.kill()
            raise
    if bundles != {row["game_id"]: row["bundle_digest"] for row in members} or not count:
        raise ValueError("public game/PRE coverage mismatch")
    fields = {**test_binding(), "game_bundles": bundles, "public_pres": count, "complete_q": count, "source": source,
              "worker": header, "resources": {**usage, "wall_seconds_before_publish": time.monotonic() - started}}
    published = publish_and_readback(Q_PATH, Q_TYPE, fields, "predictions.jsonl", payload.getvalue())
    print(json.dumps({"status": "TEST_Q_COMPLETE", **published, "games": len(bundles),
                      "public_pres": count, "resources": fields["resources"]}), flush=True)


def verify_test_q(split, members, expected_digest):
    """完整消费公开记录并验证 Q。返回后才允许打开监督。"""
    from werewolf.artifact_io.canonical import require_sha256

    require_sha256(expected_digest, "expected Test Q digest")
    q = open_artifact_envelope(Q_PATH, expected_artifact_type=Q_TYPE, expected_schema_version=Q_TYPE)
    manifest = q.manifest
    if (q.manifest_digest != expected_digest
            or any(manifest.get(key) != value for key, value in test_binding().items())
            or manifest["artifact_identity"] != Q_PATH.name or set(manifest["file_table"]) != {"predictions.jsonl"}
            or manifest["game_bundles"] != {row["game_id"]: row["bundle_digest"] for row in members}):
        raise ValueError("unexpected Independent Test Q identity or game population")
    aligned = []
    count = 0
    stream = io.BytesIO(read_artifact_file(q, "predictions.jsonl"))
    for bundle, prefixes in public_games(split, members):
        records = []
        for prefix in prefixes:
            record = read_line(stream.readline())
            validate_q(record, prefix)
            records.append(record)
        aligned.append((bundle, records))
        count += len(records)
    if stream.read(1) or manifest["public_pres"] != count or manifest["complete_q"] != count or not count:
        raise ValueError("Independent Test Q PRE population is incomplete or has extra predictions")
    return q, aligned


def test_samples(aligned):
    from scripts import fit_observer_dtrs_report_event as fit
    from werewolf.canonical_collection.game_bundle import read_bundle_pre_prefixes, read_bundle_supervision

    rows = []
    zero_games = []
    n_legal_pres = 0
    for bundle, records in aligned:
        prefixes = read_bundle_pre_prefixes(bundle)
        observations, roles = read_bundle_supervision(bundle, prefixes)
        wolves = tuple(p for p in PLAYER_IDS if roles[p] == "Werewolf")
        if len(wolves) != 2:
            raise ValueError("Independent Test role population mismatch")
        before = len(rows)
        for prefix, record in zip(prefixes, records, strict=True):
            if (prefix.public_temporal_state.phase.value not in fit.SPEECH_PHASES
                    or prefix.current_speaker not in wolves or prefix.current_speaker not in prefix.alive_observer_ids):
                continue
            n_legal_pres += 1
            if any(player not in wolves for player in prefix.alive_observer_ids):
                rows.extend(fit.pre_samples(prefix, wolves, observations, record))
        if len(rows) == before:
            zero_games.append(bundle.manifest["game_id"])
    return rows, {**fit.population(rows), "source_games": len(aligned), "legal_wolf_pres": n_legal_pres,
                  "missing_or_failed_reports": 0, "supervision_binding_errors": 0,
                  "zero_opportunity_game_ids": zero_games}


def fixed_probabilities(rows, model, parameters):
    from scripts import fit_observer_dtrs_report_event as fit

    if tuple(parameters["features"]) != fit.FEATURES[model]:
        raise ValueError(f"frozen feature order mismatch for {model}")
    z = fit.logits(fit.standardize(fit.design(rows, model), parameters["standardization"]), parameters["base"])
    return {"raw": fit.probability_map(rows, fit.sigmoid(z)),
            "calibrated": fit.probability_map(rows, fit.sigmoid(fit.logits(z[:, None], parameters["calibrator"])))}


def fixed_parameters():
    from scripts import analyze_observer_dtrs_report_event as decision

    artifact = open_artifact_envelope(decision.DESTINATION, expected_artifact_type=decision.RESULT_TYPE,
                                      expected_schema_version=decision.RESULT_TYPE)
    if artifact.manifest_digest != DECISION_DIGEST:
        raise ValueError("unexpected fixed Decision source")
    saved = json.loads(read_artifact_file(artifact, "results.json"))
    if (saved["loss"], saved["binary_tie_policy"], saved["probability_fit_manifest_digest"]) != (
            decision.LOSS.to_record(), decision.BINARY_TIE_POLICY.value, decision.FIT_DIGEST):
        raise ValueError("fixed Test evaluation disagrees with the Decision source")
    return decision.open_fitted_models()


def evaluate_test(rows, saved):
    from scripts import analyze_observer_dtrs_report_event as decision
    from scripts import fit_observer_dtrs_report_event as fit
    import numpy as np

    sizes = Counter(row.game_id for row in rows)
    games = tuple(sorted(sizes))
    n = np.asarray([sizes[g] for g in games], dtype=np.float64)
    draws = fit.bootstrap_indices(games, fit.BOOTSTRAP_REPLICATES, fit.BOOTSTRAP_SEED)
    weights = {"game_equal": fit.probability_map(rows, fit.game_weights(rows)),
               "record_equal": fit.probability_map(rows, [1 / len(rows)] * len(rows))}
    groups = {f"{phase}/{wolves}": np.asarray([r.phase == phase and r.alive_wolves == wolves for r in rows])
              for phase in fit.SPEECH_PHASES for wolves in (1, 2)}
    subgroups = {name: [r for r, include in zip(rows, mask, strict=True) if include] for name, mask in groups.items()}
    subgroups.update({f"phase/{phase}": [r for r in rows if r.phase == phase] for phase in fit.SPEECH_PHASES})
    subgroups.update({f"alive_wolves/{k}": [r for r in rows if r.alive_wolves == k] for k in (1, 2)})
    subgroups.update({f"day/{day}": [r for r in rows if r.day == day] for day in sorted({r.day for r in rows})})
    probability = {"raw": {}, "calibrated": {}}
    cognitive, totals = {}, {}
    for model in fit.MODEL_NAMES:
        probabilities = fixed_probabilities(rows, model, saved["models"][model])
        for version, p in probabilities.items():
            probability[version][model] = {
                **fit.score_predictions(rows, p),
                "calibration": {w: fit.reliability(rows, p, values) for w, values in weights.items()},
                "subgroups": fit.grouped_scores(subgroups, p, sizes),
            }
        if model in decision.MODELS:
            cognitive[model], totals[model] = decision.evaluate_model(
                rows, probabilities["calibrated"], games, sizes, groups, draws)
    comparisons = {}
    for version, models in probability.items():
        comparisons[version] = {}
        for first, second in (("M0", "M1"), ("M1", "M2")):
            a, b = models[first], models[second]
            pair = fit.paired_intervals(a, b, games, draws)
            for metric, value in pair.items():
                delta = n * np.asarray([b["by_game"][g][metric] - a["by_game"][g][metric] for g in games])
                value["record_equal_percentile_95"] = decision.interval(
                    decision.bootstrap_ratio(delta, n, np.ones_like(n), draws))["percentile_95"]
            comparisons[version][f"{second}-{first}"] = pair
    return {
        "probability": probability, "paired_probability_differences": comparisons, "decision": cognitive,
        "paired_loss_differences": decision.paired_losses({m: v["overall"] for m, v in cognitive.items()}, totals, n, draws),
        "loss": decision.LOSS.to_record(), "thresholds": decision.thresholds(decision.LOSS).to_record(),
        "binary_tie_policy": decision.BINARY_TIE_POLICY.value,
        "subgroup_population": {name: fit.support(group) for name, group in subgroups.items()},
        "bootstrap": {"version": fit.BOOTSTRAP_VERSION, "seed": fit.BOOTSTRAP_SEED, "replicates": len(draws),
                      "unit": "game", "scope": "fixed_predictions_and_rules",
                      "minimum_valid_fraction_for_region_interval": decision.MIN_VALID_BOOTSTRAP},
    }


def score(expected_q_digest):
    from scripts import analyze_observer_dtrs_report_event as decision

    require_new(RESULT_PATH)
    started = time.monotonic()
    split, members = open_test()
    q, aligned = verify_test_q(split, members, expected_q_digest)
    source = source_identity()
    if q.manifest["source"] != source:
        raise ValueError("scoring source differs from public Q generation")
    saved = fixed_parameters()
    print(json.dumps({"stage": "test_q_verified_before_supervision", "games": len(aligned),
                      "public_pres": q.manifest["public_pres"], "q_manifest_digest": q.manifest_digest}), flush=True)
    rows, population = test_samples(aligned)
    print(json.dumps({"stage": "test_population_aligned", "population": population}), flush=True)
    lineage = {**test_binding(), "q_manifest_digest": q.manifest_digest,
               "predictions_sha256": q.manifest["file_table"]["predictions.jsonl"]["sha256"],
               "probability_fit_manifest_digest": decision.FIT_DIGEST,
               "probability_fit_results_sha256": decision.FIT_RESULTS_SHA,
               "development_decision_manifest_digest": DECISION_DIGEST, "source": source,
               "fixed_fit_source_sha256": decision.FIT_SOURCE_SHA,
               "decision_analysis_source_sha256": sha256_bytes(Path(decision.__file__).read_bytes())}
    result = {
        **lineage, "target": saved["target"], "population": population,
        "scope": "independent_test_fixed_report_event_probability_and_decision",
        **evaluate_test(rows, saved),
        "limitations": ["The same Test games were previously used for the Epoch 35 ToM evaluation.",
                        "Current fixed models were not fitted on these games.",
                        "Percentile intervals summarize empirical risk under fixed rules.",
                        "Two-sided risk certification and effects on gameplay were not assessed."],
        "wall_seconds_before_publish": time.monotonic() - started,
    }
    published = publish_and_readback(RESULT_PATH, RESULT_TYPE, lineage, "results.json", canonical_json_bytes(result))
    print(json.dumps({"status": "INDEPENDENT_TEST_COMPLETE", **published}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    operations = parser.add_subparsers(dest="operation", required=True)
    operations.add_parser("predict")
    operations.add_parser("score").add_argument("--q-manifest-digest", required=True)
    args = parser.parse_args()
    if args.operation == "predict":
        predict()
    else:
        score(args.q_manifest_digest)
