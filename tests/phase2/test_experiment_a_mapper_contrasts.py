"""Minimal immutable CV artifacts; no training, OOF inference, or mapper fitting."""

from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
import math
from pathlib import Path
import shutil

import numpy as np
import pytest

from scripts import experiment_a_mapper_contrasts as analysis
from werewolf.artifact_io import (
    ArtifactConflictError, canonical_json_bytes, canonical_jsonl_bytes, publish_artifact, verify_artifact,
)
from werewolf.phase2_mapper import MODEL_SPECS, OOFProvenance, MapperPrediction
from werewolf.tom import backbone_evaluation as evaluation

ANALYSIS_SOURCE = analysis._analysis_source


def digest(value):
    return sha256(canonical_json_bytes(value)).hexdigest()


def make_oof(root, pair="b" * 64, seal_tag="default", evaluation_revision=None):
    source = {"source_revision": analysis.EXECUTION_REVISION, "implementation_digest": "c" * 64}
    contract = publish_artifact(root / "contract", manifest_fields={
        "artifact_type": "backbone_oof_evaluation_contract", "schema_version": evaluation.PAIRED_VERSION,
        "preparation": {"manifest_digest": pair}, "training_source": source,
        "evaluation_source": {**source, "source_revision": evaluation_revision or analysis.EXECUTION_REVISION},
        "training_revision": analysis.EXECUTION_REVISION, "training_seal_digest": "d" * 64,
    }, files={})
    descriptors = []
    for temporal in ("explicit_day_phase", "implicit"):
        for fold in range(5):
            prediction = publish_artifact(evaluation.prediction_path(root, "qwen3", fold, temporal, paired=True),
                manifest_fields={"artifact_type": "backbone_oof_fold_predictions",
                    "schema_version": evaluation.PAIRED_PREDICTION_VERSION,
                    "architecture": "qwen3", "fold": fold, "temporal_condition": temporal,
                    "evaluation_digest": contract.manifest_digest, "training_seal_digest": "d" * 64}, files={})
            descriptors.append({"architecture": "qwen3", "fold": fold,
                                "temporal_condition": temporal, "prediction_digest": prediction.manifest_digest})
    seal = publish_artifact(root / "seal", manifest_fields={
        "artifact_type": "backbone_oof_evaluation_report", "schema_version": evaluation.PAIRED_REPORT_VERSION,
        "preparation": {"manifest_digest": pair}, "training_source": source,
        "evaluation_source": {**source, "source_revision": evaluation_revision or analysis.EXECUTION_REVISION},
        "evaluation_digest": contract.manifest_digest, "paired_experiment_digest": pair,
        "training_seal_digest": "d" * 64, "predictions": descriptors,
        "synthetic_seal_tag": seal_tag}, files={})
    return {condition: OOFProvenance.from_evaluation_root(root, paired_experiment_digest=pair,
        temporal_condition=temporal, q_representation=representation, expected_seal_digest=seal.manifest_digest)
        for condition, (temporal, representation) in analysis.CONDITIONS.items()}


def make_mapper(root, condition, oof, evaluation_root):
    games = [f"game-{fold}-{suffix}" for fold in range(5) for suffix in ("a", "b")]
    folds = {}
    for fold in range(5):
        held = [g for g in games if g.startswith(f"game-{fold}-")]
        train = sorted(set(games) - set(held))
        folds[str(fold)] = {"held_out_game_ids": held, "held_out_game_digest": digest(held),
            "training_game_ids": train, "training_game_digest": digest(train), "held_out_candidate_rows": 3}
    fits = []
    for fold in range(5):
        fit = {"model_spec": MODEL_SPECS[3].record(), "mapper_fold": fold,
               "training_game_ids": folds[str(fold)]["training_game_ids"],
               "training_game_digest": folds[str(fold)]["training_game_digest"],
               "oof_q": oof.record(), "experiment_a": oof.experiment_a}
        fit["model_digest"] = digest(fit)
        fits.append(fit)
    probabilities = {"A0": (.8, .2, .4), "A1": (.6, .4, .7), "A2": (.7, .3, .2)}[condition]
    rows = []
    for fold in range(5):
        for index, (suffix, seat, label) in enumerate((("a", "player3", True),
                ("a", "player4", False), ("b", "player5", True))):
            p = probabilities[index]
            rows.append(asdict(MapperPrediction(f"game-{fold}-{suffix}", "pre", "player1", "speech", seat,
                label, fold, MODEL_SPECS[3].name, math.log(p / (1 - p)), p, fold,
                digest([fold, suffix]), oof.prediction_digest_by_fold[fold], oof.evaluation_seal_digest,
                fits[fold]["model_digest"])))
    counts = {"games": 10, "candidate_rows": 15, "candidate_pre": 10, "theta_positive_rows": 10,
              "post_day_states": 10, "skipped_no_candidate_pk_pre": 0}
    fields = {"artifact_type": "phase2_mapper_development_study", "schema_version": analysis.EXPERIMENT_A_VERSION,
        "source": {"commit": analysis.EXECUTION_REVISION}, "experiment_a": oof.experiment_a,
        "inputs": {"publication_path": str(root.parent / "publication"),
            "publication_digest": analysis.offline.PUBLICATION_DIGEST,
            "role_sidecar_digest": analysis.offline.ROLE_SIDECAR_DIGEST,
            "fold_manifest_digest": analysis.offline.FOLD_MANIFEST_DIGEST,
            "oof_evaluation_path": str(evaluation_root), "oof": oof.record()},
        "protocol": {"model_specs": [MODEL_SPECS[3].record()],
                     "feature_contract_version": analysis.FEATURE_CONTRACT_VERSION, "bootstrap": None},
        "selection": {"primary_model": MODEL_SPECS[3].name, "stopped_at": None,
                      "development_supported_alternatives": [], "method": "prespecified_fixed_M3"},
        "metric_contract": analysis.METRIC_CONTRACT, "folds": folds, "population": counts,
        "fit_model_digests": {f"{MODEL_SPECS[3].name}/fold-{i}": f["model_digest"] for i, f in enumerate(fits)},
        "candidate_identity_digest": candidate_digest(rows)}
    files = {f"cross_fitted_predictions/{MODEL_SPECS[3].name}.jsonl": canonical_jsonl_bytes(rows),
             "population.json": canonical_json_bytes({"counts": counts, "folds": folds}),
             "fit_manifests.json": canonical_json_bytes(fits)}
    return publish_artifact(root, manifest_fields=fields, files=files)


def candidate_digest(rows):
    return digest(sorted([r[k] for k in ("game_id", "boundary_id", "acting_wolf", "phase", "candidate_j",
                                         "mapper_fold", "theta_ac")] for r in rows))


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    root = tmp_path / "artifacts"
    root.mkdir()
    profile = tmp_path / "storage.json"
    profile.write_bytes(canonical_json_bytes({"artifact_root": str(root)}))
    oof = make_oof(root / "oof")
    artifacts = {condition: make_mapper(root / condition, condition, oof[condition], root / "oof")
                 for condition in analysis.CONDITIONS}
    monkeypatch.setattr(analysis, "REPLICATES", 48)
    monkeypatch.setattr(analysis, "_analysis_source", lambda: {
        "source_revision": "a" * 40, "source_sha256": {"consumer": "f" * 64},
        "implementation_digest": "f" * 64, "bootstrap_implementation_digest": "e" * 64})
    return profile, artifacts


def run(inputs, name="report"):
    profile, artifacts = inputs
    return analysis.analyze(profile, {k: v.path for k, v in artifacts.items()},
                            {k: v.manifest_digest for k, v in artifacts.items()}, name)


def load_panel(inputs):
    profile, artifacts = inputs
    root = analysis._storage_root(profile)
    loaded = {name: analysis._load(a.path, a.manifest_digest, name, root) for name, a in artifacts.items()}
    return analysis._aligned_panel(loaded)


def test_deterministic_immutable_report_and_separate_sources(inputs):
    first, second = run(inputs), run(inputs, "second-report")
    assert first.manifest_digest == second.manifest_digest == run(inputs).manifest_digest
    for name in ("statistics.json", "report.md", "manifest.json"):
        assert (first.path / name).read_bytes() == (second.path / name).read_bytes()
    verified = verify_artifact(first.path, expected_artifact_type="phase2_experiment_a_mapper_contrasts",
                               expected_schema_version=analysis.VERSION)
    m = verified.manifest
    assert m["execution_source_revision"] == analysis.EXECUTION_REVISION
    assert m["analysis_source"]["source_revision"] == "a" * 40 != analysis.EXECUTION_REVISION
    assert m["q_prediction_digests"]["A0"] == m["q_prediction_digests"]["A2"]
    assert m["q_prediction_digests"]["A0"] != m["q_prediction_digests"]["A1"]
    assert m["statistics"]["bootstrap"]["shared_across_conditions"] is True
    assert m["metric_clipping"]["epsilon"] == 1e-15
    assert set(m["statistics"]["conditions"]) == {"A0", "A1", "A2"}
    assert set(m["statistics"]["contrasts"]) == {"A1", "A2"}
    assert set(m["file_table"]) == {"statistics.json", "report.md"}


def test_manual_metrics_and_contrast_direction(inputs):
    report = run(inputs).manifest["statistics"]
    expected = {"A0": (.8, .4), "A1": (.6, .7), "A2": (.7, .2)}
    for condition, (a, b) in expected.items():
        values = (-math.log(a), -math.log(b), (1 - a) ** 2, (1 - b) ** 2)
        la, lb, ba, bb = values
        manual = ((2 * la + lb) / 3, (2 * ba + bb) / 3, (la + lb) / 2, (ba + bb) / 2)
        for metric, value in zip(analysis.METRICS, manual, strict=True):
            assert report["conditions"][condition][metric]["point"] == pytest.approx(value)
    for condition in ("A1", "A2"):
        contrast = report["contrasts"][condition]
        assert contrast["direction"] == f"{condition} - A0"
        for metric in analysis.METRICS:
            assert contrast["metrics"][metric]["point"] == pytest.approx(
                report["conditions"][condition][metric]["point"] - report["conditions"]["A0"][metric]["point"])


def test_one_shared_fold_stratified_game_draw_and_percentile_ci(inputs, monkeypatch):
    panel = load_panel(inputs)
    draws, metrics = [], []
    draw, weighted = analysis._draw_game_multiplicities, analysis._weighted_metrics
    def capture_draw(panel, rng):
        value = draw(panel, rng)
        draws.append(value.copy())
        assert sum(value) == len(panel.game_ids) == 10
        assert all(sum(value[indices]) == len(indices) == 2 for indices in panel.fold_game_indices)
        return value
    def capture_metrics(panel, weights):
        value = weighted(panel, weights)
        metrics.append(value.copy())
        return value
    monkeypatch.setattr(analysis, "_draw_game_multiplicities", capture_draw)
    monkeypatch.setattr(analysis, "_weighted_metrics", capture_metrics)
    result = analysis._bootstrap(panel)
    assert len(draws) == analysis.REPLICATES and len(metrics) == analysis.REPLICATES + 1
    assert any(any(weight == 2 for weight in draw) for draw in draws)
    assert result["bootstrap"]["shared_draw_digest"] == sha256(
        b"".join(draw.astype("<i8").tobytes() for draw in draws)).hexdigest()
    samples = np.stack(metrics[1:])
    for condition, index in (("A1", 1), ("A2", 2)):
        for j, metric in enumerate(analysis.METRICS):
            expected = np.quantile(samples[:, index, j] - samples[:, 0, j], [.025, .975], method="linear")
            np.testing.assert_allclose(result["contrasts"][condition]["metrics"][metric]["ci95"], expected)
    # Unequal candidate counts: resampling a whole game changes the pooled denominator.
    weights = np.zeros(10, dtype=np.int64)
    weights[0], weights[1] = 2, 1
    value = weighted(panel, weights)[0]
    assert value[0] == pytest.approx((-4 * math.log(.8) - math.log(.4)) / 5)
    assert value[2] == pytest.approx((-2 * math.log(.8) - math.log(.4)) / 3)


def test_output_records_verified_condition_and_evaluation_identities(inputs):
    report = run(inputs)
    manifest = json.loads((report.path / "manifest.json").read_bytes())
    artifacts = inputs[1]
    expected = {"A0": ("explicit_day_phase", "continuous", "M3_r2_additive_logistic"),
                "A1": ("implicit", "continuous", "M3_r2_additive_logistic"),
                "A2": ("explicit_day_phase", "hard_top1", "M3_r2_additive_logistic")}
    keys = ("temporal_condition", "q_representation", "mapper_spec")
    assert set(manifest["conditions"]) == set(expected)
    for condition, values in expected.items():
        identity = manifest["conditions"][condition]
        artifact = artifacts[condition]
        assert tuple(identity[key] for key in keys) == values
        assert identity == {**{key: artifact.manifest["experiment_a"][key] for key in keys},
                            "mapper_artifact_digest": artifact.manifest_digest}
    base_inputs = artifacts["A0"].manifest["inputs"]
    assert manifest["evaluation_ownership"] == {
        "resolved_root": str(Path(base_inputs["oof_evaluation_path"]).resolve(strict=True)),
        "path_scope": "local_execution_provenance"}
    assert manifest["common_oof_seal_digest"] == base_inputs["oof"]["evaluation_seal_digest"]
    assert manifest["oof_evaluation_contract_digest"] == base_inputs["oof"]["evaluation_contract_digest"]


def test_full_frozen_bootstrap_budget(inputs, monkeypatch):
    monkeypatch.setattr(analysis, "REPLICATES", 100_000)
    result = run(inputs).manifest["statistics"]["bootstrap"]
    assert result["replicates"] == 100_000 and result["seed"] == 20260929
    assert result["confidence"] == .95 and result["interval_method"] == "percentile_linear"


def mutate_artifact(inputs, target, mutation):
    profile, artifacts = inputs
    original = artifacts[target]
    fields = {k: deepcopy(v) for k, v in original.manifest.items() if k not in ("file_table", "manifest_digest")}
    files = {name: (original.path / name).read_bytes() for name in original.manifest["file_table"]}
    path = f"cross_fitted_predictions/{MODEL_SPECS[3].name}.jsonl"
    rows = [json.loads(line) for line in files[path].splitlines()]
    mutation(fields, rows)
    fields["candidate_identity_digest"] = candidate_digest(rows)
    fields["population"]["candidate_rows"] = len(rows)
    fields["population"]["theta_positive_rows"] = sum(r["theta_ac"] for r in rows)
    fields["population"]["candidate_pre"] = len({tuple(r[k] for k in
        ("game_id", "boundary_id", "acting_wolf", "phase")) for r in rows})
    for fold in range(5):
        fields["folds"][str(fold)]["held_out_candidate_rows"] = sum(r["mapper_fold"] == fold for r in rows)
    files[path] = canonical_jsonl_bytes(rows)
    files["population.json"] = canonical_json_bytes({"counts": fields["population"], "folds": fields["folds"]})
    replacement = publish_artifact(original.path.parent / f"{target}-changed", manifest_fields=fields, files=files)
    return profile, {**artifacts, target: replacement}


def extra_candidate(fields, rows):
    extra = deepcopy(rows[0])
    extra["boundary_id"] = "extra-pre"
    rows.append(extra)


@pytest.mark.parametrize("target,mutation", [
    ("A1", lambda m, r: r.pop()),
    ("A1", extra_candidate),
    ("A1", lambda m, r: r.append(deepcopy(r[0]))),
    ("A1", lambda m, r: r[0].update(theta_ac=False)),
    ("A1", lambda m, r: r[0].update(mapper_fold=1, game_oof_fold=1)),
    ("A1", lambda m, r: r[0].update(prefix_digest="foreign-prefix")),
    ("A1", lambda m, r: m["inputs"].update(publication_digest="0" * 64)),
    ("A1", lambda m, r: m["experiment_a"].update(paired_experiment_digest="0" * 64)),
    ("A1", lambda m, r: m["inputs"]["oof"].update(evaluation_seal_digest="0" * 64)),
    ("A2", lambda m, r: m["inputs"]["oof"]["prediction_digest_by_fold"].update({"0": "0" * 64})),
    ("A1", lambda m, r: m["experiment_a"].update(temporal_condition="explicit_day_phase")),
    ("A2", lambda m, r: m["experiment_a"].update(q_representation="continuous")),
    ("A1", lambda m, r: m["protocol"]["model_specs"][0].update(name="M2_r2_linear_logistic")),
    ("A1", lambda m, r: r[0].update(model_spec="M0_context_logistic")),
    ("A1", lambda m, r: m["source"].update(commit="a" * 40)),
    ("A1", lambda m, r: m["metric_contract"].update(primary="pooled_brier")),
    ("A1", lambda m, r: r[0].update(raw_probability=float("inf"))),
])
def test_fail_closed_before_bootstrap(inputs, monkeypatch, target, mutation):
    # Non-finite JSON is already rejected by the canonical publisher.
    with pytest.raises(ValueError):
        changed = mutate_artifact(inputs, target, mutation)
        monkeypatch.setattr(analysis, "_bootstrap", lambda *_: pytest.fail("invalid inputs reached bootstrap"))
        run(changed)
    assert not (inputs[1]["A0"].path.parent / "report").exists()


def test_same_candidate_set_with_reordered_rows_is_not_dropped(inputs):
    changed = mutate_artifact(inputs, "A1", lambda m, r: r.reverse())
    assert run(inputs).manifest["statistics"] == run(changed, "reordered").manifest["statistics"]


@pytest.mark.parametrize("pair,seal_tag,revision", [
    ("b" * 64, "other-seal", None),
    ("f" * 64, "other-preparation", None),
    ("b" * 64, "wrong-source", "a" * 40),
])
def test_coherent_foreign_oof_parent_or_execution_source_rejected(inputs, pair, seal_tag, revision):
    profile, artifacts = inputs
    root = artifacts["A0"].path.parent
    foreign = root / "foreign-oof"
    oof = make_oof(foreign, pair, seal_tag, revision)
    other = make_mapper(root / "foreign-A1", "A1", oof["A1"], foreign)
    with pytest.raises(ValueError, match="shared|OOF execution source"):
        run((profile, {**artifacts, "A1": other}))


@pytest.mark.parametrize("condition", ("A1", "A2"))
def test_byte_identical_copied_evaluation_root_rejected(inputs, monkeypatch, condition):
    profile, artifacts = inputs
    original = Path(artifacts["A0"].manifest["inputs"]["oof_evaluation_path"])
    copied = original.parent / "copied-oof"
    shutil.copytree(original, copied)
    original_bytes = {str(p.relative_to(original)): p.read_bytes()
                      for p in original.rglob("*") if p.is_file()}
    copied_bytes = {str(p.relative_to(copied)): p.read_bytes()
                    for p in copied.rglob("*") if p.is_file()}
    assert original_bytes == copied_bytes
    assert original.resolve(strict=True) != copied.resolve(strict=True)
    identity = artifacts[condition].manifest["experiment_a"]
    oof = OOFProvenance.from_evaluation_root(copied,
        paired_experiment_digest=identity["paired_experiment_digest"],
        temporal_condition=identity["temporal_condition"], q_representation=identity["q_representation"],
        expected_seal_digest=artifacts[condition].manifest["inputs"]["oof"]["evaluation_seal_digest"])
    assert oof.record() == artifacts[condition].manifest["inputs"]["oof"]
    other = make_mapper(original.parent / f"copied-{condition}", condition, oof, copied)
    loaded = {name: analysis._load(a.path, a.manifest_digest, name, original.parent)
              for name, a in {**artifacts, condition: other}.items()}
    assert loaded[condition][2].evaluation_seal_digest == loaded["A0"][2].evaluation_seal_digest
    assert loaded[condition][2].evaluation_contract_digest == loaded["A0"][2].evaluation_contract_digest
    if condition == "A2":
        assert loaded[condition][2].prediction_digest_by_fold == loaded["A0"][2].prediction_digest_by_fold
    monkeypatch.setattr(analysis, "_bootstrap", lambda *_: pytest.fail("copied root reached bootstrap"))
    with pytest.raises(ValueError, match="shared resolved evaluation root mismatch"):
        run((profile, {**artifacts, condition: other}))
    assert not (original.parent / "report").exists()


def test_metric_clipping_preserves_unclipped_brier(inputs):
    changed = mutate_artifact(inputs, "A1", lambda m, r:
        r[0].update(raw_probability=0., raw_score=-1000.))
    report = run(changed).manifest["statistics"]
    assert math.isfinite(report["conditions"]["A1"]["pooled_log_loss"]["point"])
    expected = (1. + 4 * .16 + 5 * .16 + 5 * .09) / 15
    assert report["conditions"]["A1"]["pooled_brier"]["point"] == pytest.approx(expected)


def test_analysis_source_change_during_computation_rejects_publication(inputs, monkeypatch):
    source = analysis._analysis_source()
    values = iter((source, {**source, "source_revision": "f" * 40}))
    monkeypatch.setattr(analysis, "_analysis_source", lambda: next(values))
    with pytest.raises(ValueError, match="source changed"):
        run(inputs)
    assert not (inputs[1]["A0"].path.parent / "report").exists()


def test_trust_anchor_and_immutable_destination(inputs):
    profile, artifacts = inputs
    with pytest.raises(ValueError, match="mapper digest"):
        analysis.analyze(profile, {k: v.path for k, v in artifacts.items()},
            {k: "0" * 64 for k in artifacts}, "invalid")
    run(inputs)
    changed = mutate_artifact(inputs, "A1", lambda m, r: r[0].update(raw_probability=.9))
    with pytest.raises(ArtifactConflictError):
        run(changed)
    with pytest.raises(ValueError, match="immutable inputs"):
        run(inputs, "A0/inside")
    with pytest.raises(ValueError, match="immutable inputs"):
        run(inputs, "publication/inside")


def test_clean_analysis_inventory_and_dirty_publish_rejection(inputs, monkeypatch):
    # Exercise attestation without making commits in this project or a temporary repository.
    root = Path(analysis.__file__).resolve().parents[1]
    calls = []
    def git(_root, *args):
        calls.append(args)
        if args == ("rev-parse", "--show-toplevel"):
            return str(root).encode()
        if args == ("rev-parse", "HEAD"):
            return b"a" * 40
        if args[0] == "status":
            return b""
        if args[0] == "show":
            return (root / args[1].split(":", 1)[1]).read_bytes()
        pytest.fail(str(args))
    monkeypatch.setattr(analysis, "_git", git)
    source = ANALYSIS_SOURCE()
    assert source["source_revision"] == "a" * 40
    assert set(source["source_sha256"]) == set(analysis.SOURCE_FILES)
    assert source["implementation_digest"] == digest(source["source_sha256"])
    assert any("--untracked-files=all" in call for call in calls)
    monkeypatch.setattr(analysis, "_git", lambda _root, *args:
        b"?? unreviewed.py\n" if args[0] == "status" else git(_root, *args))
    monkeypatch.setattr(analysis, "_analysis_source", ANALYSIS_SOURCE)
    with pytest.raises(ValueError, match="must be clean"):
        run(inputs)
    assert not (inputs[1]["A0"].path.parent / "report").exists()
