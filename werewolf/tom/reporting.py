"""OOF orchestration: finish all training, seal, then perform pure reporting."""

import numpy as np

from werewolf.artifact_io import canonical_jsonl_bytes
from werewolf.tom.evaluation import (predict_held_out_fold, evaluate_primary, evaluate_all_alive,
    open_predictions, primary_fold_report_value, all_alive_fold_report_value)
from werewolf.tom.experiment import TEMPORAL_CONDITIONS, open_experiment
from werewolf.tom.protocol import load_bootstrap_plan
from werewolf.tom.run_records import publish_record, publish_run_bytes, read_record, record_with_digest
from werewolf.tom.scoring import game_macro_summary, paired_summary
from werewolf.tom.training import train_primary_fold, seal_checkpoint_set, verify_checkpoint_set, lineage_path


def _aggregate_cell(experiment, seal, indices, bootstrap_digest, condition, operation, reports):
    game_ids = experiment.manifest["game_ids"]
    games = {}
    rows = []
    for fold, report in enumerate(reports):
        if set(report["game_scores"]) != set(experiment.fold(fold)["held_out_game_ids"]):
            raise ValueError("report game/fold coverage mismatch")
        if set(games) & set(report["game_scores"]):
            raise ValueError("duplicate held-out game")
        games.update(report["game_scores"])
        rows.extend(report["row_scores"])
    if sorted(games) != game_ids:
        raise ValueError("OOF report omitted a development game")
    scores = [games[g]["kl"] for g in game_ids]
    reference = [games[g]["uniform_kl"] for g in game_ids]
    secondary = {name: float(np.mean([games[g][name] for g in game_ids])) for name in
        ("cross_entropy", "total_variation", "mean_absolute_error", "argmax_in_target_support", "target_support_probability")}
    gap_rows = [r["normalized_reducible_gap_improvement"] for r in rows
                if r["normalized_reducible_gap_improvement"] is not None]
    cell = {"artifact_type": operation + "_aggregate", "schema_version": "classic7_oof_cell_v1",
        "experiment_digest": experiment.digest, "checkpoint_set_digest": seal["record_digest"],
        "temporal_condition": condition, "bootstrap_digest": bootstrap_digest,
        "fold_report_digests": [r["record_digest"] for r in reports], "game_scores": games,
        "headline": game_macro_summary(scores, indices, experiment.config.confidence_level),
        "uniform_non_self_reference": game_macro_summary(reference, indices, experiment.config.confidence_level),
        "secondary_game_macro_diagnostics": secondary,
        "observer_row_weighted_diagnostic": {"kl": float(np.mean([r["kl"] for r in rows])), "row_count": len(rows),
            "normalized_reducible_gap_improvement_nonzero_gap_only": float(np.mean(gap_rows)) if gap_rows else None,
            "nonzero_gap_row_count": len(gap_rows)}}
    worst = sorted(rows, key=lambda r: (-r["kl"], r["game_id"], r["boundary_id"], r["observer"]))
    return cell, canonical_jsonl_bytes(worst)


def _report_set_value(experiment, seal, indices, bootstrap_digest, cells):
    game_ids = experiment.manifest["game_ids"]
    def kl(condition, operation):
        return [cells[f"{condition}/{operation}"]["game_scores"][g]["kl"] for g in game_ids]
    stress = {c: paired_summary(kl(c, "all_alive_identifiability_stress"), kl(c, "primary_development_oof"), indices, experiment.config.confidence_level) for c in TEMPORAL_CONDITIONS}
    temporal = paired_summary(kl("implicit", "primary_development_oof"), kl("explicit_day_phase", "primary_development_oof"), indices, experiment.config.confidence_level)
    return {
        "artifact_type": "development_oof_report_set", "schema_version": "classic7_oof_report_set_v1",
        "experiment_digest": experiment.digest, "checkpoint_set_digest": seal["record_digest"],
        "bootstrap_digest": bootstrap_digest, "game_ids": game_ids, "cells": cells,
        "paired_headlines": {"identifiability_stress_penalty": stress, "primary_temporal_information_effect": temporal}}


def aggregate_development_oof(experiment):
    seal = verify_checkpoint_set(experiment)
    indices, bootstrap_digest = load_bootstrap_plan(experiment)
    cells = {}
    for condition in TEMPORAL_CONDITIONS:
        primary_reports, stress_reports = [], []
        for fold in range(5):
            primary = evaluate_primary(experiment, fold, condition)
            stress = evaluate_all_alive(experiment, fold, condition)
            if primary["checkpoint_digest"] != stress["checkpoint_digest"] or primary["prediction_digest"] != stress["prediction_digest"]:
                raise ValueError("paired population reports do not share checkpoint/predictions")
            primary_reports.append(primary)
            stress_reports.append(stress)
        for operation, reports in (("primary_development_oof", primary_reports), ("all_alive_identifiability_stress", stress_reports)):
            cell, worst = _aggregate_cell(experiment, seal, indices, bootstrap_digest, condition, operation, reports)
            directory = experiment.runs_path / "reports" / condition / operation
            cells[f"{condition}/{operation}"] = publish_record(directory / "aggregate_report.json", cell)
            publish_run_bytes(directory / "worst_cases.jsonl", worst)
    return publish_record(experiment.runs_path / "reports" / "oof_report_set.json",
        _report_set_value(experiment, seal, indices, bootstrap_digest, cells))


def _validate_report(path, expected):
    actual = read_record(path)
    if actual != record_with_digest(expected):
        raise ValueError("report semantic/provenance mismatch")
    return actual


def validate_existing_evaluation(experiment):
    """Read-only audit of existing outputs, including incomplete pure evaluation."""
    seal = verify_checkpoint_set(experiment)
    reports = {}
    for condition in TEMPORAL_CONDITIONS:
        for fold in range(5):
            lineage = lineage_path(experiment, fold, condition)
            if (lineage / "prediction_manifest.json").exists() or (lineage / "held_out_predictions.jsonl").exists():
                open_predictions(experiment, fold, condition)
            for operation, build in (("primary_development_oof", primary_fold_report_value),
                                     ("all_alive_identifiability_stress", all_alive_fold_report_value)):
                path = lineage / operation / "fold_report.json"
                if path.exists():
                    reports[condition, operation, fold] = _validate_report(path, build(experiment, fold, condition))
    indices, bootstrap_digest = load_bootstrap_plan(experiment)
    cells = {}
    for condition in TEMPORAL_CONDITIONS:
        for operation in ("primary_development_oof", "all_alive_identifiability_stress"):
            directory = experiment.runs_path / "reports" / condition / operation
            path, worst_path = directory / "aggregate_report.json", directory / "worst_cases.jsonl"
            if not path.exists() and not worst_path.exists():
                continue
            if not path.exists() or any((condition, operation, f) not in reports for f in range(5)):
                raise ValueError("aggregate report is missing its parent reports")
            expected, worst = _aggregate_cell(experiment, seal, indices, bootstrap_digest, condition, operation,
                [reports[condition, operation, f] for f in range(5)])
            cells[f"{condition}/{operation}"] = _validate_report(path, expected)
            if worst_path.exists() and worst_path.read_bytes() != worst:
                raise ValueError("worst-case report payload mismatch")
    path = experiment.runs_path / "reports" / "oof_report_set.json"
    if path.exists():
        if len(cells) != 4:
            raise ValueError("OOF report set is missing aggregate cells")
        _validate_report(path, _report_set_value(experiment, seal, indices, bootstrap_digest, cells))


def run_development_oof(experiment):
    reopened = open_experiment(experiment.path)
    if reopened.digest != experiment.digest:
        raise ValueError("experiment identity changed")
    if not (experiment.runs_path / "checkpoint_set_manifest.json").exists():
        for condition in TEMPORAL_CONDITIONS:
            for fold in range(5):
                train_primary_fold(experiment, fold, condition)
        seal_checkpoint_set(experiment)
    verify_checkpoint_set(experiment)
    for condition in TEMPORAL_CONDITIONS:
        for fold in range(5):
            predict_held_out_fold(experiment, fold, condition)
    return aggregate_development_oof(experiment)
