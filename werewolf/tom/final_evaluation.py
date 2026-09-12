"""Sealed label-free prediction and one-shot independent evaluation."""

import numpy as np
import torch

from werewolf.artifact_io import canonical_jsonl_bytes, sha256_bytes, open_artifact_envelope
from werewolf.final_publication import FINAL_PUBLICATION_VERSION, open_final_publication, publish_final_publication
from werewolf.development_publication import open_publication
from werewolf.tom.dataset import CanonicalToMDataset, ExperimentCapacity, PublicTensors, tensorize_public_pre
from werewolf.tom.experiment import TEMPORAL_CONDITIONS, validate_runtime
from werewolf.tom.final_capacity import validate_final_pre
from werewolf.tom.final_training import verify_final_seal, verify_final_terminal
from werewolf.tom.training import build_model
from werewolf.tom.state import load_model_state
from werewolf.tom.population import select_final_primary_population, build_all_alive_eligibility
from werewolf.tom.evaluation import score_prediction_rows
from werewolf.tom.reporting import summarize_scores
from werewolf.tom.protocol import bootstrap_indices, BOOTSTRAP_VERSION
from werewolf.tom.scoring import paired_summary, SCORING_VERSION
from werewolf.tom.run_records import publish_record, publish_run_bytes, read_record


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
