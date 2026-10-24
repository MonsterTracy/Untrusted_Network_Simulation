"""Synthetic paired Qwen3 checks only: two engineering steps, never formal OOF."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import json
from types import SimpleNamespace as NS

import pytest
import torch

from tests.development_publication.test_development_publication import _publication
from tests.tom.test_paper_study_execution import rewritten
from werewolf.artifact_io import canonical_json_bytes, verify_artifact
from werewolf.tom import backbone_study as p, backbone_execution as e, backbone_evaluation as v
from werewolf.tom.dataset import PublicTensors
from werewolf.tom.experiment import ExperimentConfig, runtime_provenance


def _paired_engineering_process(root, expected, architecture, fold, interrupt, connection, temporal_condition):
    """Real spawned worker; substitute only synthetic publication/source facts."""
    metadata = json.loads((Path(root) / "contract/manifest.json").read_bytes())
    snapshot = json.loads((Path(metadata["prepared_path"]) / "manifest.json").read_bytes())
    design = p._frozen_design()
    for key in ("publication_id", "game_count", "reference_protocol"):
        design[key] = snapshot["protocol_inputs"]["design"][key]
    p._frozen_design = lambda: deepcopy(design)
    e.attest_source = lambda: dict(metadata["execution_source"])
    e._process_entry(root, expected, architecture, fold, interrupt, connection, temporal_condition)


@pytest.fixture(scope="module")
def paired(tmp_path_factory):
    root = tmp_path_factory.mktemp("paired-qwen3-synthetic")
    _, handles = _publication(root / "publication")
    design = p._frozen_design()
    design["publication_id"] = handles.public.public_view.publication_id
    design["game_count"] = len(handles.public.public_view.game_ids)
    design["reference_protocol"]["device"] = "cpu"
    source = {"source_revision": "a" * 40,
              "implementation_digest": runtime_provenance(
                  ExperimentConfig(**design["reference_protocol"]))["implementation_digest"]}
    config = root / "design.json"
    config.write_bytes(canonical_json_bytes(design))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(p, "_frozen_design", lambda: deepcopy(design))
        patch.setattr(p, "attest_source", lambda: dict(source))
        patch.setattr(e, "attest_source", lambda: dict(source))
        artifact = p.prepare_study(config, handles.public.path, root / "prepared", paired_temporal=True)
        execution = e.create_execution(artifact.path, artifact.manifest_digest, engineering=True)
        yield artifact, execution


def assert_same_rng(left, right):
    assert e._same(left[1], right[1])
    assert left[0].keys() == right[0].keys()
    for key in left[0]:
        assert left[0][key].array.tobytes() == right[0][key].array.tobytes()


def test_exact_initial_tensors_rng_order_targets_and_only_temporal_input_changes(paired):
    artifact, execution = paired
    assert artifact.manifest["schema_version"] == p.PAIRED_STUDY_VERSION
    assert len(artifact.manifest["initializations"]) == 5
    assert {r["temporal_condition"] for r in artifact.manifest["initializations"]} == {"shared"}
    assert len(e.study_lineages(execution.snapshot)) == 10
    explicit, implicit = (execution.for_condition(t) for t in p.PAIRED_CONDITIONS)
    left, right = e._binding(explicit, "qwen3", 0), e._binding(implicit, "qwen3", 0)
    assert {k for k in left if not e._same(left[k], right[k])} == {"temporal_condition"}
    assert left["training_rng"] == right["training_rng"]
    assert "temporal_condition" not in left["training_rng"]
    assert e._lineage(explicit, "qwen3", 0) != e._lineage(implicit, "qwen3", 0)
    assert e._same(explicit.schedule(0), implicit.schedule(0))

    model_a, opt_a = e._model_optimizer(explicit, "qwen3", 0)
    rng_a = e.training._rng_capture()
    model_b, opt_b = e._model_optimizer(implicit, "qwen3", 0)
    assert_same_rng(rng_a, e.training._rng_capture())
    assert e._same(opt_a.state_dict(), opt_b.state_dict())
    assert all(torch.equal(value, model_b.state_dict()[key])
               for key, value in model_a.state_dict().items())
    assert model_a.temporal.explicit_enabled and not model_b.temporal.explicit_enabled
    assert model_a.temporal.artifact_digest == model_b.temporal.artifact_digest

    games_a, masks_a = e._materialize(explicit, 0)
    games_b, masks_b = e._materialize(implicit, 0)
    assert list(games_a) == list(games_b) and masks_a == masks_b
    for game in games_a:
        for a, b in zip(games_a[game], games_b[game], strict=True):
            assert (a.boundary_id, a.prefix_digest, a.plan_digest) == (b.boundary_id, b.prefix_digest, b.plan_digest)
            for key, tensor in a.public.kwargs().items():
                assert torch.equal(tensor, b.public.kwargs()[key])
            for key in ("q", "label_observed", "observer_alive"):
                assert torch.equal(getattr(a, key), getattr(b, key))
    sample = next(iter(games_a.values()))[-1]
    inputs = PublicTensors.stack([sample.public]).kwargs()
    saved = {k: value.clone() for k, value in inputs.items()}
    seen, streams = [], []
    for model, condition in ((model_a, explicit), (model_b, implicit)):
        def capture(_module, _args, kwargs):
            seen.append((kwargs["inputs_embeds"].detach().clone(), kwargs["attention_mask"].clone()))
        hook = model.transformer.register_forward_pre_hook(capture, with_kwargs=True)
        try:
            e._reset_rng(e._binding(condition, "qwen3", 0))
            with torch.no_grad():
                model(**inputs)
            streams.append(e.training._rng_capture())
        finally:
            hook.remove()
    assert_same_rng(*streams)
    assert all(torch.equal(value, saved[k]) for k, value in inputs.items())
    torch.testing.assert_close(seen[0][0] - seen[1][0],
        model_a.temporal.code(inputs["day_ids"], inputs["phase_ids"]), rtol=1e-5, atol=2e-8)
    assert torch.equal(seen[0][1], seen[1][1])
    assert torch.count_nonzero(model_b.temporal.code(inputs["day_ids"], inputs["phase_ids"])) == 0



def test_paired_rng_ignores_provenance_and_changes_only_with_randomization_controls(paired):
    artifact, execution = paired
    binding = e._binding(execution, "qwen3", 0)
    manifest = deepcopy(artifact.manifest)
    manifest["study_digest"] = "b" * 64
    for key in ("publication_digest", "fold_manifest_digest", "primary_digest"):
        manifest["protocol_inputs"][key] = "c" * 64
    manifest["protocol_inputs"]["source"] = {
        "source_revision": "d" * 40, "implementation_digest": "e" * 64}
    snapshot = NS(path=artifact.path, manifest=manifest, manifest_digest="f" * 64)
    contract = deepcopy(execution.manifest)
    contract["preparation"] = e.snapshot_identity(snapshot)
    contract["execution_source"] = manifest["protocol_inputs"]["source"]
    changed = replace(execution, snapshot=snapshot, artifact=NS(
        path=execution.artifact.path.parent / "other-output/contract",
        manifest=contract, manifest_digest="1" * 64))
    other = e._binding(changed, "qwen3", 0)
    assert not e._same(binding, other)
    assert binding["training_rng"] == other["training_rng"]
    assert set(binding["training_rng"]) == {"version", "protocol_seed", "architecture", "fold"}
    e._reset_rng(binding)
    expected = e.training._rng_capture()
    e._reset_rng(other)
    assert_same_rng(expected, e.training._rng_capture())
    for altered in (e._binding(execution, "qwen3", 1), e._binding(replace(execution,
            config=replace(execution.config, rng_seed=execution.config.rng_seed + 1)), "qwen3", 0)):
        assert binding["training_rng"] != altered["training_rng"]
        e._reset_rng(altered)
        assert not e._same(expected[1], e.training._rng_capture()[1])
    for key, value in e.snapshot_identity(artifact).items():
        wrong = {**e.snapshot_identity(artifact), key: "0" * len(value)}
        with pytest.raises(ValueError, match="frozen preparation identity mismatch"):
            e.validate_frozen_snapshot(artifact.path, wrong)


def test_initial_tensor_bytes_are_independent_of_preparation_provenance(paired, tmp_path, monkeypatch):
    artifact, _ = paired
    config = tmp_path / "design.json"
    config.write_bytes(canonical_json_bytes(p._frozen_design()))
    source = {**e.attest_source(), "source_revision": "b" * 40}
    monkeypatch.setattr(p, "attest_source", lambda: dict(source))
    changed = p.prepare_study(config, artifact.manifest["publication_path"],
                             tmp_path / "prepared", paired_temporal=True)
    assert changed.manifest_digest != artifact.manifest_digest
    assert changed.manifest["study_digest"] != artifact.manifest["study_digest"]
    for fold in range(5):
        relative = f"initial/qwen3/{fold}/tensors.bin"
        assert (artifact.path / relative).read_bytes() == (changed.path / relative).read_bytes()
        assert changed.manifest["initializations"][fold]["state_digest"] != artifact.manifest["initializations"][fold]["state_digest"]
    with pytest.raises(ValueError, match="frozen preparation identity mismatch"):
        e.validate_frozen_snapshot(changed.path, e.snapshot_identity(artifact))


def test_two_engineering_steps_recovery_and_prediction_identity(paired, tmp_path, monkeypatch):
    _, execution = paired
    predictions = []
    owner = v.HistoricalTraining(execution, NS(manifest_digest="e" * 64))
    contract = NS(path=tmp_path / "contract", manifest_digest="d" * 64)
    dataset, masks = v._held_out(owner, 0)
    for condition in p.PAIRED_CONDITIONS:
        selected = execution.for_condition(condition)
        if condition == "implicit":
            with pytest.raises(InterruptedError):
                e._train(selected, "qwen3", 0, interrupt_after=1)
        e._train(selected, "qwen3", 0)
        terminal = e.validate_terminal(selected, "qwen3", 0)
        model, optimizer = e._model_optimizer(selected, "qwen3", 0)
        e._recovery_chain(selected, "qwen3", 0, model, optimizer)
        selected_owner = replace(owner, execution=selected)
        monkeypatch.setattr(v, "open_evaluation", lambda _root, o=selected_owner: (contract, o))
        monkeypatch.setattr(v, "validate_historical_terminal", lambda *_args: (terminal, model))
        artifact = v.evaluate_fold(tmp_path, "qwen3", 0, temporal_condition=condition)
        v._validated_fold(selected_owner, contract, "qwen3", 0, terminal, dataset, masks)
        v._verify_model_predictions(selected_owner, artifact, terminal, model, dataset, masks)
        assert artifact.manifest["temporal_condition"] == condition
        assert artifact.manifest["schema_version"] == v.PAIRED_PREDICTION_VERSION
        predictions.append(artifact)
    paths = [e._lineage(execution.for_condition(t), "qwen3", 0) / "recovery/2" for t in p.PAIRED_CONDITIONS]
    assert (paths[0] / "rng.bin").read_bytes() == (paths[1] / "rng.bin").read_bytes()
    manifests = [json.loads((path / "manifest.json").read_bytes()) for path in paths]
    assert e._same(manifests[0]["rng_scalars"], manifests[1]["rng_scalars"])
    assert predictions[0].path != predictions[1].path
    assert predictions[0].manifest_digest != predictions[1].manifest_digest
    implicit = execution.for_condition("implicit")
    model, optimizer = e._model_optimizer(implicit, "qwen3", 0)
    manifest = e._lineage(implicit, "qwen3", 0) / "recovery/1/manifest.json"
    with rewritten(manifest, lambda m: m["binding"].update(temporal_condition="explicit_day_phase"), "manifest_digest"):
        with pytest.raises(ValueError):
            e._read_checkpoint(implicit, "qwen3", 0, 1, None, [], model, optimizer)


def test_paired_schema_source_gates_and_legacy_contract(paired, monkeypatch):
    artifact, execution = paired
    assert e.open_execution(execution.root).artifact.manifest_digest == execution.artifact.manifest_digest
    assert p.open_study(artifact.path).manifest_digest == artifact.manifest_digest
    assert p.PAIRED_STUDY_VERSION == "classic7_qwen3_paired_temporal_study_v2"
    assert e.PAIRED_RNG_VERSION == "qwen3_paired_training_rng_v2"
    with pytest.raises(ValueError, match="unknown backbone study schema"):
        p._design("classic7_qwen3_paired_temporal_study_v1")
    with rewritten(execution.artifact.path / "manifest.json",
                   lambda m: m.update(training_rng_version="qwen3_paired_training_rng_v1"), "manifest_digest"):
        with pytest.raises(ValueError, match="execution schema/lineage/control mismatch"):
            e.open_execution(execution.root)
    with pytest.raises(ValueError):
        e._lineage(execution, "qwen2", 0)
    with pytest.raises(ValueError):
        execution.for_condition("other")
    owner = v.HistoricalTraining(execution, NS())
    v._evaluation_source(owner)
    source = e.attest_source()
    monkeypatch.setattr(e, "attest_source", lambda: {**source, "source_revision": "b" * 40})
    with pytest.raises(ValueError, match="paired training/inference source"):
        v._evaluation_source(owner)
    assert p.STUDY_VERSION == "classic7_backbone_study_v1"
    assert p.INITIAL_VERSION == "classic7_backbone_initial_v1"
    assert e.RNG_VERSION == "backbone_training_rng_v1"
    assert len(e.LINEAGES) == 20 and {r[2] for r in e.LINEAGES} == {"explicit_day_phase"}
    assert v.TRAINING_REVISION == "ebf98b7e3249f78d3f1fa1fd46c20affe0486913"
    with pytest.raises(ValueError):
        verify_artifact(artifact.path, expected_artifact_type="backbone_tom_study",
                        expected_schema_version=p.STUDY_VERSION)


def test_public_runner_passes_condition_into_real_spawned_worker(paired, monkeypatch):
    _, execution = paired
    monkeypatch.setattr(e, "_process_entry", _paired_engineering_process)
    result = e.run_lineage(execution.root, "qwen3", 1, temporal_condition="implicit")
    terminal = e.validate_terminal(execution.for_condition("implicit"), "qwen3", 1)
    assert result["terminal_digest"] == terminal.manifest_digest
    assert terminal.manifest["binding"]["temporal_condition"] == "implicit"
    assert not e._lineage(execution, "qwen3", 1).exists()
