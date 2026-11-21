"""历史 Final 评价，以及固定 Canonical Test 的独立预测与评分。"""

import json
from pathlib import Path

import numpy as np
import torch

from werewolf.artifact_io import (
    ArtifactConflictError, ArtifactValidationError, canonical_json_bytes, canonical_jsonl_bytes,
    open_artifact_envelope, publish_artifact, read_artifact_file, sha256_bytes,
)
from werewolf.canonical_collection.game_bundle import (
    open_split_game_bundle, read_bundle_pre_prefixes, read_bundle_supervision,
)
from werewolf.canonical_collection.public_history import PLAYER_IDS
from werewolf.canonical_split import open_canonical_split
from werewolf.final_publication import FINAL_PUBLICATION_VERSION, open_final_publication, publish_final_publication
from werewolf.development_publication import open_publication
from werewolf.structured_history import plan_structured_history
from werewolf.tom.dataset import CanonicalToMDataset, ExperimentCapacity, PublicTensors, belief_target, tensorize_public_pre
from werewolf.tom.experiment import TEMPORAL_CONDITIONS, validate_runtime
from werewolf.tom.final_capacity import validate_final_pre
from werewolf.tom.final_training import verify_final_seal, verify_final_terminal
from werewolf.tom.training import build_model
from werewolf.tom.state import load_model_state
from werewolf.tom.population import PRIMARY_SELECTOR_VERSION, select_final_primary_population, build_all_alive_eligibility
from werewolf.tom.evaluation import (
    predict_structured_observers, probability_cell, score_prediction_rows, uniform_prediction_rows,
)
from werewolf.tom.paper_study import attest_source
from werewolf.tom.reporting import summarize_scores
from werewolf.tom.protocol import bootstrap_indices, BOOTSTRAP_VERSION
from werewolf.tom.scoring import paired_summary, SCORING_VERSION
from werewolf.tom.run_records import publish_record, publish_run_bytes, read_record


TEST_SPLIT_DIGEST = "edc271676649d9b23e01f505f0f23deb38fdbd0b0ab0ad65753a90cd562ee6e8"
TEST_GAME_COUNT = 1000
TEST_TEMPORAL_DIGEST = "16830ce07a968c5d8c811912f6b6b189fefffd1f296cae1e304b8b37111e48b6"
TEST_SCHEMA = "classic7_independent_test"
TEST_PREDICTIONS = "independent_test_predictions"
TEST_EVALUATION = "independent_test_evaluation"
PROBABILITY_FIELDS = {"probability", "non_self_log_probability"}


def _independent_test_bundles(split_path, collection):
    """仅打开冻结 Split 和它列出的 Test Bundle 清单。"""
    split = open_canonical_split(split_path, expected_digest=TEST_SPLIT_DIGEST)
    members = sorted(split.members("independent_test"), key=lambda member: member["game_id"])
    games = {member["game_id"]: member["bundle_digest"] for member in members}
    if len(members) != len(games) or len(games) != TEST_GAME_COUNT:
        raise ArtifactValidationError("Independent Test must cover the complete frozen game population")
    binding = {"split_id": split.manifest["artifact_identity"], "split_manifest_digest": split.digest,
               "independent_test_identity_digest": split.manifest["independent_test_identity_digest"],
               "collection_identity_digest": split.manifest["collection_identity_digest"],
               "game_bundles": games}
    return binding, [open_split_game_bundle(collection, split.plan, member) for member in members]


def _independent_test_model():
    from werewolf.tom.makto import EPOCH35_DIGEST
    from werewolf.tom.qwen3_large import graph_identity

    return {"checkpoint_digest": EPOCH35_DIGEST, "graph_digest": graph_identity()["graph_digest"],
            "temporal_artifact_digest": TEST_TEMPORAL_DIGEST, "temporal_condition": "explicit_day_phase",
            "canonical_shift": 0}


def _test_output_path(destination, collection):
    path = Path(destination).absolute()
    if (".." in path.parts or Path(collection).resolve() in (path, *path.parents)
            or any(p.is_symlink() for p in (path, *path.parents))):
        raise ArtifactConflictError("Test output requires a separate directory outside the Canonical Collection")
    if path.exists() or any((parent / name).exists() for parent in path.parents for name in (
            "manifest.json", "contract/manifest.json", "phase_codebook.manifest.json",
            "experiment_manifest.json", "final_experiment_manifest.json")):
        raise ArtifactConflictError("Test output exists or is inside frozen evidence")
    return path


def _test_row_identity(game_id, boundary_id, alive, plan, observer):
    return {"game_id": game_id, "boundary_id": boundary_id, "prefix_digest": plan.prefix_digest,
            "plan_digest": plan.plan_digest, "observer": observer, "observer_alive": PLAYER_IDS[observer] in alive}


def _test_coverage(binding, pre_count, rows):
    return {"game_count": len(binding["game_bundles"]), "pre_count": pre_count,
            "public_prediction_row_count": len(rows),
            "public_alive_row_count": sum(r["observer_alive"] for r in rows)}


def predict_independent_test(split_path, collection, checkpoint, temporal, destination, *, device="cpu"):
    """用公开 PRE 生成完整七席预测，成功后一次性封存。"""
    from werewolf.tom.makto import load_epoch35_model

    destination = _test_output_path(destination, collection)
    source = attest_source()
    binding, bundles = _independent_test_bundles(split_path, collection)
    identity = _independent_test_model()
    model = load_epoch35_model(checkpoint, temporal)
    if (model.graph["graph_digest"] != identity["graph_digest"]
            or model.temporal.artifact_digest != identity["temporal_artifact_digest"]):
        raise ArtifactValidationError("Independent Test requires the frozen Epoch 35 graph and Temporal")

    plans = []
    for bundle in bundles:
        for prefix in read_bundle_pre_prefixes(bundle):
            plan = plan_structured_history(prefix)
            if plan.token_count > model.capacity.max_seq_len:
                raise ValueError(f"Independent Test PRE exceeds model capacity: {prefix.game_id}/{prefix.boundary_id}")
            if prefix.maximum_public_day >= len(model.temporal.day):
                raise ValueError(f"Independent Test PRE exceeds frozen Temporal: {prefix.game_id}/{prefix.boundary_id}")
            plans.append((prefix.game_id, prefix.boundary_id, prefix.alive_observer_ids, plan))

    model.to(device)
    rows = []
    for game_id, boundary_id, alive, plan in plans:
        probabilities, logs = predict_structured_observers(plan, model, device)
        for observer in range(len(PLAYER_IDS)):
            rows.append({**_test_row_identity(game_id, boundary_id, alive, plan, observer),
                         "probability": probabilities[observer], "non_self_log_probability": logs[observer]})
    coverage = _test_coverage(binding, len(plans), rows)
    return publish_artifact(destination, manifest_fields={
        "artifact_type": TEST_PREDICTIONS, "schema_version": TEST_SCHEMA, **binding, **identity,
        "source": source, "device": device, "coverage": coverage,
    }, files={"predictions.jsonl": canonical_jsonl_bytes(rows)})


def _align_independent_test_predictions(prediction, bundles):
    """在读取第一条监督前检查整个预测封存及其公开人口。"""
    from werewolf.tom_llm.common import valid_distribution

    data = read_artifact_file(prediction, "predictions.jsonl")
    rows = [json.loads(line) for line in data.splitlines()]
    if canonical_jsonl_bytes(rows) != data:
        raise ArtifactValidationError("Independent Test predictions must be canonical JSONL")
    ordered_rows = iter(rows)
    inputs = []
    for bundle in bundles:
        prefixes = read_bundle_pre_prefixes(bundle)
        game_rows = []
        for prefix in prefixes:
            plan = plan_structured_history(prefix)
            for observer in range(len(PLAYER_IDS)):
                identity = _test_row_identity(prefix.game_id, prefix.boundary_id, prefix.alive_observer_ids, plan, observer)
                row = next(ordered_rows, None)
                if (not isinstance(row, dict) or set(row) != set(identity) | PROBABILITY_FIELDS
                        or type(row["observer_alive"]) is not bool
                        or any(row[name] != value for name, value in identity.items())):
                    raise ArtifactValidationError(f"missing or misaligned Independent Test prediction: {identity}")
                valid_distribution(row)
                game_rows.append(row)
        inputs.append((bundle, prefixes, game_rows))
    pre_count = sum(len(pres) for _, pres, _ in inputs)
    if len(rows) != pre_count * len(PLAYER_IDS):
        raise ArtifactValidationError("extra Independent Test prediction rows")
    coverage = _test_coverage(prediction.manifest, pre_count, rows)
    if coverage != prediction.manifest["coverage"]:
        raise ArtifactValidationError("Independent Test prediction coverage mismatch")
    return inputs, coverage


def score_independent_test(split_path, collection, predictions, expected_prediction_digest, destination):
    """核验全量公开预测后才读取 Test 监督并发布游戏级指标。"""
    from werewolf.tom_llm.policy import POLICY

    destination = _test_output_path(destination, collection)
    source = attest_source()
    prediction = open_artifact_envelope(predictions, expected_artifact_type=TEST_PREDICTIONS,
                                        expected_schema_version=TEST_SCHEMA)
    if prediction.manifest_digest != expected_prediction_digest or set(prediction.manifest["file_table"]) != {"predictions.jsonl"}:
        raise ArtifactValidationError("Independent Test prediction Seal mismatch")
    binding, bundles = _independent_test_bundles(split_path, collection)
    identity = _independent_test_model()
    if any(prediction.manifest[k] != v for k, v in {**binding, **identity}.items()):
        raise ArtifactValidationError("Independent Test prediction model or Split binding mismatch")
    inputs, coverage = _align_independent_test_predictions(prediction, bundles)

    primary, population = [], []
    for bundle, prefixes, rows in inputs:
        observations, roles = read_bundle_supervision(bundle, prefixes)
        reports = {(o.boundary_id, o.observer_id): o for o in observations.values()}
        for row in rows:
            seat = row["observer"]
            player = PLAYER_IDS[seat]
            if not row["observer_alive"] or roles[player] == "Werewolf":
                continue
            observation = reports[row["boundary_id"], player]
            q, observed = belief_target(seat, observation.suspicion_support, observation.status.value)
            if observed:
                primary.append({**row, "q": q.tolist()})
                population.append({key: row[key] for key in ("game_id", "boundary_id", "prefix_digest", "plan_digest", "observer")}
                                  | {"nu": int(q.count_nonzero()) < len(PLAYER_IDS) - 1})

    games = sorted(binding["game_bundles"])
    primary_games = {r["game_id"] for r in population}
    missing = set(games) - primary_games
    if missing:
        raise ValueError(f"Independent Test has games without effective Primary rows: {sorted(missing)}")
    nu_games = sorted({r["game_id"] for r in population if r["nu"]})
    sampling = POLICY["bootstrap"]
    indices = bootstrap_indices(games, sampling["replicates"], sampling["seed"])
    nu_indices = None
    if nu_games:
        nu_indices = indices if nu_games == games else bootstrap_indices(nu_games, sampling["replicates"], sampling["seed"])
    cells = {}
    # 每种方法使用同一 q、同一游戏排序和同一抽样。
    for method, rows in (("epoch35", primary), ("uniform_non_self", uniform_prediction_rows(primary))):
        cell, nu_scores = probability_cell(rows, games, indices, nu_games, nu_indices)
        cells[method] = {**cell, "nu_game_scores": nu_scores}
    coverage.update(primary_row_count=len(primary), primary_game_count=len(primary_games),
                    nu_row_count=sum(r["nu"] for r in population), nu_game_count=len(nu_games))
    bootstrap = {**sampling, "version": BOOTSTRAP_VERSION, "indices_digest": sha256_bytes(indices.tobytes()),
                 "nu_indices_digest": None if nu_indices is None else sha256_bytes(nu_indices.tobytes())}
    report = {**binding, **identity, "prediction_seal_digest": prediction.manifest_digest,
              "prediction_source": prediction.manifest["source"], "scoring_source": source,
              "coverage": coverage, "game_ids": games, "nu_game_ids": nu_games, "bootstrap": bootstrap,
              "scoring_version": SCORING_VERSION, "population_version": PRIMARY_SELECTOR_VERSION,
              "cells": cells}
    return publish_artifact(destination, manifest_fields={
        "artifact_type": TEST_EVALUATION, "schema_version": TEST_SCHEMA,
        "prediction_seal_digest": prediction.manifest_digest, "split_manifest_digest": binding["split_manifest_digest"],
        "independent_test_identity_digest": binding["independent_test_identity_digest"],
    }, files={"analysis.json": canonical_json_bytes(report), "population.jsonl": canonical_jsonl_bytes(population)})


class SealedFinalPredictor:
    """Public PRE includes V1 annotations; observer is a zero-based seat."""
    def __init__(self, experiment, condition):
        self.seal = verify_final_seal(experiment)
        validate_runtime(experiment)
        self.experiment = experiment
        self.condition = condition
        self.checkpoint = verify_final_terminal(experiment, condition)
        self.model = build_model(experiment, condition)
        load_model_state(self.checkpoint.path, self.model, self.checkpoint.manifest_digest)
        self.model.eval()

    def log_probabilities(self, prefix):
        validate_final_pre(prefix, self.experiment.manifest["protocol_inputs"]["capacity"])
        _, tensors = tensorize_public_pre(prefix, ExperimentCapacity(self.experiment.config.max_seq_len))
        public = PublicTensors.stack([tensors])
        with torch.inference_mode():
            result = self.model(**{k: v.to(self.experiment.config.device) for k, v in public.kwargs().items()})[0].cpu()
        for observer in range(7):
            non_self = result[observer, [j for j in range(7) if j != observer]]
            if not torch.isfinite(non_self).all() or result[observer, observer] != -torch.inf or not torch.isclose(non_self.exp().sum(), torch.tensor(1.), atol=1e-6):
                raise ValueError("prediction violates fixed non-self simplex")
        return result

    def predict(self, prefix, observer):
        if type(observer) is not int or not 0 <= observer < 7:
            raise ValueError("observer must be a zero-based seat")
        return self.log_probabilities(prefix)[observer].exp()


def publish_final_evaluation(experiment, collection, destination, *, publication_id):
    seal = verify_final_seal(experiment)
    return publish_final_publication(collection, destination, publication_id=publication_id,
                                     model_seal_digest=seal["record_digest"])


def _independent(experiment, publication):
    development = open_publication(experiment.manifest["publication_path"])
    if development.manifest_digest != experiment.manifest["publication_digest"]:
        raise ValueError("development identity changed")
    d, f = development.manifest, publication.manifest
    if d["collection_id"] == f["collection_id"] or d["collection_identity_digest"] == f["collection_identity_digest"]:
        raise ValueError("final collection must be independent of development")
    for field in ("game_id", "bundle_digest", "seed"):
        if {g[field] for g in d["development_games"]} & {g[field] for g in f["games"]}:
            raise ValueError(f"development/final overlap: {field}")
    development_seeds = {g["seed"] for g in d["development_games"]} | {g["seed"] for g in d["complete_case_exclusions"]}
    final_seeds = {g["seed"] for g in f["games"]} | {g["seed"] for g in f["exclusions"]}
    if development_seeds & final_seeds:
        raise ValueError("development/final attempted seeds overlap")


def _consume(experiment, publication_path, resume):
    seal = verify_final_seal(experiment)  # before even the final manifest is opened
    validate_runtime(experiment)  # environment failures must precede final consumption
    envelope = open_artifact_envelope(publication_path, expected_artifact_type="final_evaluation_publication",
                                      expected_schema_version=FINAL_PUBLICATION_VERSION)
    if envelope.manifest["model_seal_digest"] != seal["record_digest"]:
        raise ValueError("final publication belongs to another model seal")
    path = experiment.runs_path / "final_evaluation_consumption.json"
    if path.exists() and not resume:
        raise ValueError("final evaluation already consumed; explicit identical-run resume required")
    if resume and not path.exists():
        raise ValueError("no final evaluation consumption to resume")
    return publish_record(path, {"schema_version": "classic7_final_evaluation_consumption_v1",
        "experiment_digest": experiment.digest, "model_seal_digest": seal["record_digest"],
        "publication_digest": envelope.manifest_digest, "publication_path": str(envelope.path.resolve()),
        "scoring_version": SCORING_VERSION})


def _report(experiment, publication, consumption, rows_by_condition):
    ids = sorted(publication.public_view.game_ids)
    primary = select_final_primary_population(publication)
    alive = build_all_alive_eligibility(publication.public_view)
    config = experiment.config
    indices = bootstrap_indices(ids, config.bootstrap_replicates, config.bootstrap_seed)
    cells = {}
    for condition in TEMPORAL_CONDITIONS:
        for name, eligibility in (("primary", primary), ("all_alive_identifiability_stress", alive)):
            selected = [r for r in rows_by_condition[condition] if r["label_observed"] and eligibility.rows[r["game_id"]][r["boundary_id"]][r["observer"]]]
            scores, games = score_prediction_rows(selected, ids)
            cells[f"{condition}/{name}"] = {"population": name, "population_provenance": eligibility.metadata,
                "game_scores": games, "row_scores": scores,
                **summarize_scores(games, scores, ids, indices, config.confidence_level)}
    def kl(condition, population):
        return [cells[f"{condition}/{population}"]["game_scores"][g]["kl"] for g in ids]
    return {"schema_version": "classic7_final_report_v1", "experiment_digest": experiment.digest,
        "consumption_digest": consumption["record_digest"], "model_seal_digest": consumption["model_seal_digest"],
        "publication_digest": publication.manifest_digest, "game_ids": ids, "scoring_version": SCORING_VERSION,
        "prediction_digests": {c: sha256_bytes(canonical_jsonl_bytes(rows_by_condition[c])) for c in TEMPORAL_CONDITIONS},
        "bootstrap": {"schema_version": BOOTSTRAP_VERSION, "seed": config.bootstrap_seed,
            "replicates": config.bootstrap_replicates, "shape": list(indices.shape), "dtype": "<i4", "layout": "C",
            "confidence_level": config.confidence_level, "digest": sha256_bytes(indices.tobytes())},
        "cells": cells, "paired_headlines": {
            "primary_temporal_information_effect": paired_summary(kl("implicit", "primary"), kl("explicit_day_phase", "primary"), indices, config.confidence_level),
            "identifiability_stress_penalty": {c: paired_summary(kl(c, "all_alive_identifiability_stress"), kl(c, "primary"), indices, config.confidence_level) for c in TEMPORAL_CONDITIONS}}}, indices


def run_final_evaluation(experiment, publication_path, *, resume=False):
    consumption = _consume(experiment, publication_path, resume)
    root = experiment.runs_path / "final_evaluation"
    if (root / "failure.json").exists():
        raise ValueError("final evaluation failed closed; no subset report is permitted")
    if (root / "report.json").exists():
        return validate_final_evaluation(experiment)
    try:
        publication = open_final_publication(publication_path, model_seal_digest=consumption["model_seal_digest"])
        _independent(experiment, publication)
        # Validate the entire final set before the first forward. No filtering.
        for game in publication.public_view.games:
            for prefix in game.authoritative_pre_prefixes:
                validate_final_pre(prefix, experiment.manifest["protocol_inputs"]["capacity"])
        dataset = CanonicalToMDataset(publication.public_view, sorted(publication.public_view.game_ids), ExperimentCapacity(experiment.config.max_seq_len))
        prefixes = {(g.game_id, p.boundary_id): p for g in publication.public_view.games for p in g.authoritative_pre_prefixes}
        rows_by_condition = {}
        for condition in TEMPORAL_CONDITIONS:
            predictor = SealedFinalPredictor(experiment, condition)
            rows = []
            for sample in dataset:
                logp = predictor.log_probabilities(prefixes[sample.game_id, sample.boundary_id])
                for observer in range(7):
                    rows.append({"game_id": sample.game_id, "boundary_id": sample.boundary_id,
                        "prefix_digest": sample.prefix_digest, "plan_digest": sample.plan_digest, "observer": observer,
                        "q": sample.q[observer].tolist(), "label_observed": bool(sample.label_observed[observer]),
                        "observer_alive": bool(sample.observer_alive[observer]), "probability": logp[observer].exp().tolist(),
                        "non_self_log_probability": [logp[observer, j].item() for j in range(7) if j != observer],
                        "checkpoint_digest": predictor.checkpoint.manifest_digest, "temporal_condition": condition})
            rows_by_condition[condition] = rows
            del predictor
        report, indices = _report(experiment, publication, consumption, rows_by_condition)
        for condition, rows in rows_by_condition.items():
            publish_run_bytes(root / f"{condition}_predictions.jsonl", canonical_jsonl_bytes(rows))
        publish_run_bytes(root / "bootstrap_indices.bin", indices.tobytes())
        return publish_record(root / "report.json", report)
    except Exception as error:
        publish_record(root / "failure.json", {"schema_version": "classic7_final_evaluation_failure_v1",
            "consumption_digest": consumption["record_digest"], "error": type(error).__name__, "message": str(error)})
        raise


def validate_final_evaluation(experiment):
    from werewolf.tom.experiment import read_json
    seal = verify_final_seal(experiment)
    consumption = read_record(experiment.runs_path / "final_evaluation_consumption.json")
    if set(consumption) != {"schema_version", "experiment_digest", "model_seal_digest", "publication_digest", "publication_path", "scoring_version", "record_digest"}:
        raise ValueError("final evaluation consumption fields mismatch")
    if (consumption["experiment_digest"] != experiment.digest or consumption["model_seal_digest"] != seal["record_digest"]
        or consumption["schema_version"] != "classic7_final_evaluation_consumption_v1" or consumption["scoring_version"] != SCORING_VERSION):
        raise ValueError("final evaluation consumption mismatch")
    publication = open_final_publication(consumption["publication_path"], model_seal_digest=seal["record_digest"])
    if publication.manifest_digest != consumption["publication_digest"]:
        raise ValueError("final evaluation publication changed")
    _independent(experiment, publication)
    for game in publication.public_view.games:
        for prefix in game.authoritative_pre_prefixes:
            validate_final_pre(prefix, experiment.manifest["protocol_inputs"]["capacity"])
    dataset = CanonicalToMDataset(publication.public_view, sorted(publication.public_view.game_ids), ExperimentCapacity(experiment.config.max_seq_len))
    root = experiment.runs_path / "final_evaluation"
    if (root / "failure.json").exists():
        raise ValueError("final evaluation is failed")
    rows_by_condition = {}
    for condition in TEMPORAL_CONDITIONS:
        rows = [read_json(line) for line in (root / f"{condition}_predictions.jsonl").read_bytes().splitlines()]
        if len(rows) != len(dataset) * 7:
            raise ValueError("final prediction coverage mismatch")
        checkpoint = verify_final_terminal(experiment, condition)
        for index, sample in enumerate(dataset):
            for observer in range(7):
                row = rows[7 * index + observer]
                expected = {"game_id": sample.game_id, "boundary_id": sample.boundary_id, "prefix_digest": sample.prefix_digest,
                    "plan_digest": sample.plan_digest, "observer": observer, "q": sample.q[observer].tolist(),
                    "label_observed": bool(sample.label_observed[observer]), "observer_alive": bool(sample.observer_alive[observer]),
                    "checkpoint_digest": checkpoint.manifest_digest, "temporal_condition": condition}
                if set(row) != set(expected) | {"probability", "non_self_log_probability"} or any(row[k] != v for k, v in expected.items()):
                    raise ValueError("final prediction identity/target mismatch")
                p, logs = np.array(row["probability"]), np.array(row["non_self_log_probability"])
                if p.shape != (7,) or logs.shape != (6,) or not np.isfinite(p).all() or not np.isfinite(logs).all() or p[observer] != 0 or np.any(p < 0) or not np.isclose(p.sum(), 1, atol=1e-6) or not np.allclose(np.delete(p, observer), np.exp(logs), atol=1e-7):
                    raise ValueError("final prediction simplex mismatch")
        rows_by_condition[condition] = rows
    expected, indices = _report(experiment, publication, consumption, rows_by_condition)
    report = read_record(root / "report.json")
    if {k: v for k, v in report.items() if k != "record_digest"} != expected or (root / "bootstrap_indices.bin").read_bytes() != indices.tobytes():
        raise ValueError("final report does not match canonical scoring")
    return report
