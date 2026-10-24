"""Explicitly synthetic OOF envelopes; never represent formal model inference."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from scripts.inspect_observer_dtrs_samples import main
from werewolf.artifact_io import (
    ArtifactValidationError, canonical_json_bytes, canonical_jsonl_bytes,
    publish_artifact, sha256_bytes,
)
from werewolf.observer_dtrs.features import observer_features
from werewolf.observer_dtrs.offline import build_observer_samples
from werewolf.observer_dtrs.oof import load_qwen3_oof
from werewolf.structured_history import plan_structured_history


@pytest.fixture
def oof_factory(tmp_path, publication_factory):
    counter = 0

    def create(*, publication=None, mutate=None):
        nonlocal counter
        counter += 1
        root = tmp_path / f"synthetic-oof-{counter}"
        publication = publication or publication_factory(name=f"pub-{counter}")
        public = publication.public
        view = public.public_view
        games = set(view.game_ids)
        folds = {f.fold_index: set(f.game_ids) for f in view.fold_manifest.folds}

        def edit(stage, value, fold=None):
            if mutate:
                mutate(stage, value, fold)
            return value

        def write(path, kind, version, fields, files=None):
            return publish_artifact(path, manifest_fields={"artifact_type": kind,
                "schema_version": version, **fields}, files=files or {})

        prep_folds = [{"fold": f, "held_out_game_ids": sorted(ids),
                       "training_game_ids": sorted(games - ids)} for f, ids in folds.items()]
        prep_fields = edit("preparation", {"protocol_inputs": {
            "publication_digest": public.manifest_digest,
            "fold_manifest_digest": view.fold_manifest.manifest_digest,
            "development_game_set_digest": view.development_game_set_digest}, "folds": prep_folds})
        prep = write(root / "prepared", "backbone_tom_study", "classic7_backbone_study_v1", prep_fields)
        preparation = {"manifest_digest": prep.manifest_digest}
        source = {"source_revision": "a" * 40, "implementation_digest": "b" * 64}
        train = write(root / "training/contract", "backbone_execution", "classic7_backbone_execution_v1",
            {"engineering_only": False, "prepared_path": str(prep.path),
             "preparation": preparation, "execution_source": source})
        terminals = {}
        for fold in range(5):
            terminals[fold] = write(root / "training/runs/qwen3" / str(fold) / "terminal",
                "backbone_terminal", "classic7_backbone_terminal_v1", edit("terminal", {
                    "binding": {"architecture": "qwen3", "fold": fold,
                        "temporal_condition": "explicit_day_phase", "engineering_only": False,
                        "execution_digest": train.manifest_digest, "execution_source": source,
                        "preparation": preparation},
                    "checkpoint_ancestry": [{"digest": sha256_bytes(f"synthetic-{fold}".encode())}]}, fold))
        ts = write(root / "training/seal", "backbone_execution_seal", "classic7_backbone_all_twenty_v1",
            {"execution_digest": train.manifest_digest, "preparation": preparation,
             "execution_source": source, "terminals": [{"architecture": "qwen3", "fold": f,
                "temporal_condition": "explicit_day_phase", "terminal_digest": t.manifest_digest}
                 for f, t in terminals.items()]})
        contract = write(root / "evaluation/contract", "backbone_oof_evaluation_contract",
            "classic7_backbone_oof_evaluation_v1", {"preparation": preparation,
            "training_execution_digest": train.manifest_digest, "training_seal_digest": ts.manifest_digest,
            "training_path": str(root / "training"), "training_source": source,
            "training_revision": source["source_revision"],
            "evaluation_source": {"source_revision": "c" * 40, "implementation_digest": "d" * 64}})
        descriptors, bindings = [], {}
        for fold in range(5):
            checkpoint = terminals[fold].manifest["checkpoint_ancestry"][-1]["digest"]
            rows = []
            for g in view.games:
                if g.game_id not in folds[fold]:
                    continue
                for p in g.authoritative_pre_prefixes:
                    for observer in range(7):
                        values = [0 if j == observer else (j + 1) for j in range(7)]
                        values = [v / sum(values) for v in values]
                        rows.append(edit("row", {"game_id": g.game_id, "boundary_id": p.boundary_id,
                            "prefix_digest": p.prefix_digest, "plan_digest": plan_structured_history(p).plan_digest,
                            "observer": observer, "probability": values,
                            # Ignored legacy target/log/mask fields, deliberately synthetic.
                            "q": [0] * 7, "label_observed": True, "observer_alive": True,
                            "non_self_log_probability": [-99] * 6, "checkpoint_digest": checkpoint,
                            "temporal_condition": "explicit_day_phase"}, fold))
            edit("rows", rows, fold)
            bindings[str(fold)] = sha256_bytes(canonical_json_bytes(["synthetic-primary", fold]))
            artifact = write(root / "evaluation/folds/qwen3" / str(fold), "backbone_oof_fold_predictions",
                "classic7_backbone_oof_predictions_v1", edit("prediction_manifest", {
                    "evaluation_digest": contract.manifest_digest, "training_seal_digest": ts.manifest_digest,
                    "architecture": "qwen3", "fold": fold, "temporal_condition": "explicit_day_phase",
                    "terminal_digest": terminals[fold].manifest_digest, "checkpoint_digest": checkpoint,
                    "row_count": len(rows), "primary_row_identity_digest": bindings[str(fold)]}, fold),
                {"predictions.jsonl": canonical_jsonl_bytes(rows)})
            descriptors.append({"architecture": "qwen3", "fold": fold,
                "prediction_digest": artifact.manifest_digest, "terminal_digest": terminals[fold].manifest_digest})
        seal_fields = {k: contract.manifest[k] for k in ("preparation", "training_execution_digest",
                       "training_seal_digest", "training_source", "evaluation_source")}
        seal = write(root / "evaluation/seal", "backbone_oof_evaluation_report",
            "classic7_backbone_oof_report_v1", edit("seal", {**seal_fields,
                "evaluation_digest": contract.manifest_digest, "predictions": descriptors,
                "primary_row_bindings": bindings}))
        samples = build_observer_samples(public.path)
        return SimpleNamespace(publication=public.path, evaluation=root / "evaluation",
            seal_digest=seal.manifest_digest, samples=samples, contexts=tuple(s.context for s in samples))

    return create


def load(fixture, **kwargs):
    return load_qwen3_oof(fixture.publication, fixture.evaluation,
        kwargs.pop("contexts", fixture.contexts),
        expected_seal_digest=kwargs.pop("seal", fixture.seal_digest), **kwargs)


@pytest.mark.parametrize("phase,dead", (("discussion", ()), ("pk_discussion", ()),
                                      ("pk_discussion", ("player2", "player7"))))
def test_real_artifact_io_identity_coverage_and_features(oof_factory, publication_factory, phase, dead):
    publication = publication_factory(phase=phase, dead=dead)
    fixture = oof_factory(publication=publication)
    predictions = load(fixture)
    assert len(predictions) == len(fixture.samples)
    for sample, prediction in zip(fixture.samples, predictions, strict=True):
        assert prediction.context == sample.context
        assert prediction.source.fold_index == sample.source.fold_index
        assert prediction.source.publication_digest == sample.source.publication_digest
        assert prediction.source.evaluation_seal_digest == fixture.seal_digest
        assert prediction.source.architecture == "qwen3"
        assert prediction.probability.observer_id == sample.context.observer_id
        features = observer_features(sample.context, prediction.probability)
        assert features.nW == (1 if dead else 2) and features.phase == phase
        assert (features.dW is None) == bool(dead)
        assert "q" not in prediction.source.to_record()
    assert load(fixture, contexts=fixture.contexts[:2]) == predictions[:2]


@pytest.mark.parametrize("field,value", (("game_id", "wrong"), ("boundary_id", "wrong"),
    ("prefix_digest", "f" * 64), ("plan_digest", "f" * 64), ("checkpoint_digest", "f" * 64),
    ("temporal_condition", "none"), ("observer", True), ("observer", 7),
    ("probability", [-1] * 7)))
def test_resealed_but_semantically_wrong_prediction_fails(oof_factory, field, value):
    def mutate(stage, row, fold):
        if stage == "row" and fold == 0:
            row[field] = value
    fixture = oof_factory(mutate=mutate)
    with pytest.raises(ArtifactValidationError):
        load(fixture)


@pytest.mark.parametrize("mode", ("missing", "duplicate", "fold_swap", "train_leakage",
                                  "publication", "model", "duplicate_lineage", "terminal"))
def test_resealed_source_contract_errors_fail_closed(oof_factory, mode):
    def mutate(stage, value, fold):
        if stage == "rows" and fold == 0:
            if mode == "missing":
                value.pop()
            if mode == "duplicate":
                value.append(dict(value[0]))
        if stage == "preparation":
            if mode == "fold_swap":
                value["folds"][0]["held_out_game_ids"], value["folds"][1]["held_out_game_ids"] = (
                    value["folds"][1]["held_out_game_ids"], value["folds"][0]["held_out_game_ids"])
            if mode == "train_leakage":
                value["folds"][0]["training_game_ids"] += value["folds"][0]["held_out_game_ids"]
            if mode == "publication":
                value["protocol_inputs"]["publication_digest"] = "f" * 64
        if stage == "prediction_manifest" and fold == 0 and mode == "model":
            value["architecture"] = "gpt2"
        if stage == "seal" and mode == "duplicate_lineage":
            value["predictions"].append(dict(value["predictions"][0]))
        if stage == "terminal" and fold == 0 and mode == "terminal":
            value["binding"]["fold"] = 1
    fixture = oof_factory(mutate=mutate)
    with pytest.raises(ArtifactValidationError):
        load(fixture)


def test_unsealed_payload_tamper_and_external_trust_anchor(oof_factory):
    fixture = oof_factory()
    for anchor in ("f" * 64, "", None):
        with pytest.raises(ArtifactValidationError):
            load(fixture, seal=anchor)
    payload = fixture.evaluation / "folds/qwen3/0/predictions.jsonl"
    raw = payload.read_bytes()
    payload.write_bytes(raw.replace(b'"probability":', b'"probabilitx":', 1))
    with pytest.raises(ArtifactValidationError, match="digest mismatch"):
        load(fixture)


def test_requested_identity_and_supervision_envelope_rejected(oof_factory):
    fixture = oof_factory()
    with pytest.raises(ArtifactValidationError, match="duplicate requested"):
        load(fixture, contexts=fixture.contexts[:1] * 2)
    for field, value in (("step_index", 99), ("day", 99), ("prefix_digest", "f" * 64),
                         ("phase", "pk_discussion"), ("boundary_id", "unknown")):
        with pytest.raises(ArtifactValidationError):
            load(fixture, contexts=(replace(fixture.contexts[0], **{field: value}),))
    with pytest.raises(TypeError):
        load(fixture, contexts=fixture.samples)


def test_labels_targets_logs_and_legacy_masks_never_change_features(oof_factory, publication_factory):
    left_pub = publication_factory(name="positive", supports={"player3": ("player1",)})
    right_pub = publication_factory(name="negative", supports={"player3": ("player4",)})

    def mutate(stage, value, fold):
        if stage == "row":
            value.update(q=None, label_observed=False, observer_alive=False,
                         non_self_log_probability=[123] * 6)
    left, right = oof_factory(publication=left_pub), oof_factory(publication=right_pub, mutate=mutate)
    assert left.contexts == right.contexts
    assert left.samples[0].supervision.event_label_y != right.samples[0].supervision.event_label_y
    a = [observer_features(p.context, p.probability) for p in load(left)]
    b = [observer_features(p.context, p.probability) for p in load(right)]
    assert a == b


def test_readonly_inspector_oof_path(oof_factory, capsys):
    fixture = oof_factory()
    assert main(["--publication", str(fixture.publication), "--oof-root", str(fixture.evaluation),
                 "--expected-oof-seal-digest", fixture.seal_digest]) == 0
    import json
    report = json.loads(capsys.readouterr().out)
    assert report["oof"]["feature_count"] == 25
    assert report["oof"]["fold_sample_counts"] == {str(i): 5 for i in range(5)}
    with pytest.raises(SystemExit):
        main(["--publication", str(fixture.publication), "--oof-root", str(fixture.evaluation)])
