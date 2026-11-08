from pathlib import Path

import pytest

from tests.tom.test_experiment import prepared_experiment


def _wrong_tensor_container(payload, payload_name):
    import numpy as np

    from werewolf.artifact_io import TensorValue
    from werewolf.artifact_io.tensor_state import encode_tensor_container

    container, rewritten = encode_tensor_container(
        {
            "wrong_graph.weight": TensorValue(
                np.frombuffer(payload, dtype="<f4").copy(),
                True,
            )
        },
        payload_path=payload_name,
    )
    assert rewritten == payload
    return container


def _reinterpret_model_payload(path, payload_name):
    """Replace only tensor graph metadata while preserving canonical raw bytes."""
    import json
    import shutil

    from werewolf.artifact_io import publish_artifact

    manifest = json.loads((path / "manifest.json").read_bytes())
    payload = (path / payload_name).read_bytes()
    container = _wrong_tensor_container(payload, payload_name)
    fields = {
        key: value
        for key, value in manifest.items()
        if key not in {"manifest_digest", "file_table", "tensor_container"}
    }
    shutil.rmtree(path)
    return publish_artifact(
        path,
        manifest_fields={**fields, "tensor_container": container},
        files={payload_name: payload},
    )


def _reinterpret_final_recovery(experiment, terminal, *, wrong_terminal):
    """Rebind terminal provenance to a validly hashed wrong recovery graph."""
    import json
    import shutil

    from werewolf.artifact_io import publish_artifact

    point = experiment.fold(0)["optimizer_steps"]
    recovery_path = terminal.path.parent / "recovery" / str(point)
    recovery_manifest = json.loads((recovery_path / "manifest.json").read_bytes())
    recovery_files = {
        name: (recovery_path / name).read_bytes()
        for name in recovery_manifest["file_table"]
    }
    containers = dict(recovery_manifest["containers"])
    containers["model_tensors"] = _wrong_tensor_container(
        recovery_files["model_tensors.bin"],
        "model_tensors.bin",
    )
    recovery_fields = {
        key: value
        for key, value in recovery_manifest.items()
        if key not in {"manifest_digest", "file_table", "containers"}
    }
    shutil.rmtree(recovery_path)
    recovery = publish_artifact(
        recovery_path,
        manifest_fields={**recovery_fields, "containers": containers},
        files=recovery_files,
    )

    terminal_manifest = json.loads((terminal.path / "manifest.json").read_bytes())
    terminal_payload = (terminal.path / "tensors.bin").read_bytes()
    terminal_provenance = dict(terminal_manifest["provenance"])
    terminal_provenance["last_recovery_digest"] = recovery.manifest_digest
    terminal_fields = {
        key: value
        for key, value in terminal_manifest.items()
        if key not in {"manifest_digest", "file_table", "provenance", "tensor_container"}
    }
    container = (
        _wrong_tensor_container(terminal_payload, "tensors.bin")
        if wrong_terminal
        else terminal_manifest["tensor_container"]
    )
    shutil.rmtree(terminal.path)
    return publish_artifact(
        terminal.path,
        manifest_fields={
            **terminal_fields,
            "provenance": terminal_provenance,
            "tensor_container": container,
        },
        files={"tensors.bin": terminal_payload},
    )


@pytest.fixture(scope="module")
def all_ten_unsealed_experiment(tmp_path_factory):
    from werewolf.tom.experiment import TEMPORAL_CONDITIONS
    from werewolf.tom.training import train_primary_fold

    experiment, _ = prepared_experiment(tmp_path_factory.mktemp("typed-seal"))
    for condition in TEMPORAL_CONDITIONS:
        for fold in range(5):
            train_primary_fold(experiment, fold, condition)
    return experiment


def _clone_unsealed_experiment(source, tmp_path):
    import shutil

    from werewolf.tom.experiment import open_experiment

    path = tmp_path / "experiments" / "experiment"
    shutil.copytree(source.path, path)
    clone = open_experiment(path)
    shutil.copytree(source.runs_path, clone.runs_path)
    return clone


def test_fixed_terminal_primary_training_and_preseal_evaluation_gate(tmp_path):
    from werewolf.tom.training import train_primary_fold
    from werewolf.tom.evaluation import predict_held_out_fold

    experiment, _ = prepared_experiment(tmp_path)
    checkpoint = train_primary_fold(experiment, 0, "implicit")
    provenance = checkpoint.manifest["provenance"]
    assert provenance["terminal_step"] == 7
    assert provenance["paired_initial_state_digest"] == experiment.fold(0)["paired_initial_state_digest"]
    assert not list(experiment.runs_path.rglob("best.pt"))
    with pytest.raises(ValueError, match="seal"):
        predict_held_out_fold(experiment, 0, "implicit")


def test_verify_terminal_rejects_self_consistent_wrong_qwen_graph(tmp_path):
    from werewolf.tom.training import train_primary_fold, verify_terminal

    experiment, _ = prepared_experiment(tmp_path)
    terminal = train_primary_fold(experiment, 0, "implicit")
    _reinterpret_model_payload(terminal.path, "tensors.bin")

    with pytest.raises(ValueError, match="model state key"):
        verify_terminal(experiment, 0, "implicit")


def test_training_rejects_changed_environment_and_all_alive_condition(tmp_path):
    from werewolf.tom.training import train_primary_fold

    experiment, _ = prepared_experiment(tmp_path)
    with pytest.raises(ValueError, match="temporal"):
        train_primary_fold(experiment, 0, "all_alive")


def test_actual_training_worker_never_opens_outer_games_or_stress_masks(tmp_path, monkeypatch):
    from werewolf.tom.training import _train_worker
    experiment, _ = prepared_experiment(tmp_path)
    held_out = set(experiment.fold(0)["held_out_game_ids"])
    read = Path.read_bytes

    def gated(path):
        # The typed population boundary validates membership against role
        # truth; the worker itself still receives only boolean eligibility.
        assert not any(part.startswith("held_out_") for part in path.parts)
        if "public" in path.parts and "games" in path.parts:
            assert path.stem not in held_out
        return read(path)

    monkeypatch.setattr(Path, "read_bytes", gated)
    assert _train_worker(str(experiment.path), experiment.digest, 0, "implicit").manifest["provenance"]["terminal_step"] == 7


@pytest.mark.parametrize("violation", ["failure", "rng", "recovery_missing", "recovery_corrupt", "recovery_schema"])
def test_seal_rejects_incomplete_or_inconsistent_terminal_evidence(tmp_path, violation):
    from werewolf.tom.training import train_primary_fold, seal_checkpoint_set
    from werewolf.tom.run_records import publish_record, read_record
    from werewolf.artifact_io import canonical_json_bytes, sha256_bytes
    import json

    experiment, _ = prepared_experiment(tmp_path)
    terminal = train_primary_fold(experiment, 0, "implicit")
    lineage = terminal.path.parent
    if violation == "failure":
        publish_record(lineage / "failure.json", {"error": "injected failure"})
    elif violation == "rng":
        path = lineage / "training_manifest.json"
        value = read_record(path)
        del value["record_digest"]
        value["initial_rng_digest"] = "0" * 64
        value["record_digest"] = sha256_bytes(canonical_json_bytes(value))
        path.write_bytes(canonical_json_bytes(value))
    elif violation == "recovery_missing":
        (lineage / "recovery" / "7").rename(lineage / "recovery" / ".lost-7")
    elif violation == "recovery_corrupt":
        (lineage / "recovery" / "7" / "rng_state.bin").write_bytes(b"corrupt")
    else:
        path = lineage / "recovery" / "7" / "manifest.json"
        value = json.loads(path.read_bytes())
        del value["manifest_digest"]
        value["scheduler_state"] = {"name": "adaptive", "step": 7}
        value["manifest_digest"] = sha256_bytes(canonical_json_bytes(value))
        path.write_bytes(canonical_json_bytes(value))
    with pytest.raises(ValueError, match="failed|RNG|recovery|mismatch"):
        seal_checkpoint_set(experiment)
    assert not (experiment.runs_path / "checkpoint_set_manifest.json").exists()


def test_seal_rejects_self_consistent_wrong_terminal_graph(
    tmp_path,
    all_ten_unsealed_experiment,
):
    from werewolf.tom.training import seal_checkpoint_set, verify_terminal

    experiment = _clone_unsealed_experiment(all_ten_unsealed_experiment, tmp_path)
    terminal = verify_terminal(experiment, 0, "implicit")
    _reinterpret_model_payload(terminal.path, "tensors.bin")

    with pytest.raises(ValueError, match="model state key"):
        seal_checkpoint_set(experiment)
    assert not (experiment.runs_path / "checkpoint_set_manifest.json").exists()


def test_seal_rejects_same_wrong_graph_in_terminal_and_final_recovery(
    tmp_path,
    all_ten_unsealed_experiment,
):
    from werewolf.tom.training import seal_checkpoint_set, verify_terminal

    experiment = _clone_unsealed_experiment(all_ten_unsealed_experiment, tmp_path)
    terminal = verify_terminal(experiment, 0, "implicit")
    _reinterpret_final_recovery(experiment, terminal, wrong_terminal=True)

    with pytest.raises(ValueError, match="model state key"):
        seal_checkpoint_set(experiment)
    assert not (experiment.runs_path / "checkpoint_set_manifest.json").exists()


def test_seal_rejects_wrong_final_recovery_graph_with_unchanged_model_bytes(
    tmp_path,
    all_ten_unsealed_experiment,
):
    from werewolf.tom.training import seal_checkpoint_set, verify_terminal

    experiment = _clone_unsealed_experiment(all_ten_unsealed_experiment, tmp_path)
    terminal = verify_terminal(experiment, 0, "implicit")
    _reinterpret_final_recovery(experiment, terminal, wrong_terminal=False)

    with pytest.raises(ValueError, match="model state key"):
        seal_checkpoint_set(experiment)
    assert not (experiment.runs_path / "checkpoint_set_manifest.json").exists()


def test_all_ten_fixed_qwen_terminal_and_recovery_states_seal(
    tmp_path,
    all_ten_unsealed_experiment,
):
    from werewolf.tom.training import seal_checkpoint_set, verify_checkpoint_set

    experiment = _clone_unsealed_experiment(all_ten_unsealed_experiment, tmp_path)
    seal = seal_checkpoint_set(experiment)
    assert verify_checkpoint_set(experiment) == seal


def test_training_gate_traverses_the_result_tree_once(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from werewolf.tom.training import _training_gate

    (tmp_path/'implicit/0/recovery/1').mkdir(parents=True)
    (tmp_path/'implicit/0/recovery/1/state.bin').write_bytes(b'fixture')
    patterns = []
    original = Path.rglob
    def observe(path, pattern):
        patterns.append(pattern)
        return original(path, pattern)
    monkeypatch.setattr(Path, 'rglob', observe)
    _training_gate(SimpleNamespace(runs_path=tmp_path))
    assert len(patterns) == 1


@pytest.mark.parametrize('name', ('held_out_predictions.jsonl', 'fold_report.json'))
@pytest.mark.parametrize('kind', ('file', 'directory', 'symlink', 'dangling_symlink'))
def test_training_gate_preserves_entry_rules_without_reading_them(tmp_path, monkeypatch, name, kind):
    from types import SimpleNamespace
    from werewolf.tom.training import _training_gate

    root = tmp_path/'runs'
    parent = root/'implicit/4/nested'
    parent.mkdir(parents=True)
    path = parent/name
    if kind == 'directory':
        path.mkdir()
    elif kind in ('symlink', 'dangling_symlink'):
        target = tmp_path/'private-result'
        if kind == 'symlink':
            target.write_bytes(b'not a model input')
        path.symlink_to(target)
    else:
        path.write_bytes(b'not a model input')
    # 支持的 Python 版本对精确 glob 的悬空链接语义不同，保持原判定。
    forbidden = bool(list(root.rglob(name)))
    monkeypatch.setattr(Path, 'read_bytes', lambda path: pytest.fail('gate opened held-out content'))
    if forbidden:
        with pytest.raises(ValueError, match='held-out access'):
            _training_gate(SimpleNamespace(runs_path=root))
    else:
        _training_gate(SimpleNamespace(runs_path=root))


@pytest.mark.parametrize('name', ('held_out_predictions.jsonl', 'fold_report.json'))
def test_training_stops_before_next_optimizer_step_when_held_out_appears(tmp_path, monkeypatch, name):
    import torch
    from werewolf.tom import training

    experiment, _ = prepared_experiment(tmp_path)
    steps = []
    original = torch.optim.AdamW.step
    def step(optimizer, *args, **kwargs):
        result = original(optimizer, *args, **kwargs)
        steps.append(1)
        target = experiment.runs_path/'explicit_day_phase/4'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b'forbidden result appeared during training')
        return result
    monkeypatch.setattr(torch.optim.AdamW, 'step', step)
    with pytest.raises(ValueError, match='held-out access'):
        training._train_worker(str(experiment.path), experiment.digest, 0, 'implicit')
    assert len(steps) == 1
    assert not (training.lineage_path(experiment, 0, 'implicit')/'terminal_checkpoint').exists()


@pytest.mark.parametrize('name', ('held_out_predictions.jsonl', 'fold_report.json'))
def test_resumed_worker_rechecks_held_out_before_reusing_terminal(tmp_path, monkeypatch, name):
    from werewolf.tom import training

    experiment, _ = prepared_experiment(tmp_path)
    training._train_worker(str(experiment.path), experiment.digest, 0, 'implicit')
    path = experiment.runs_path/'explicit_day_phase/4'/name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'forbidden result appeared between calls')
    monkeypatch.setattr(training, 'build_model', lambda *args: pytest.fail('resume bypassed held-out gate'))
    with pytest.raises(ValueError, match='held-out access'):
        training._train_worker(str(experiment.path), experiment.digest, 0, 'implicit')


def test_paper_gate_keeps_checking_both_families_on_every_call(tmp_path):
    from types import SimpleNamespace
    from werewolf.tom import paper_study_execution

    full, agnostic = tmp_path/'full', tmp_path/'agnostic'
    full.mkdir()
    agnostic.mkdir()
    study = SimpleNamespace(seal_path=tmp_path/'study-seal.json',
        full=SimpleNamespace(runs_path=full), agnostic=SimpleNamespace(runs_path=agnostic))
    paper_study_execution._training_gate(study)
    (agnostic/'fold_report.json').write_bytes(b'forbidden')
    with pytest.raises(ValueError, match='held-out access'):
        paper_study_execution._training_gate(study)
    (agnostic/'fold_report.json').unlink()
    study.seal_path.write_bytes(b'sealed')
    with pytest.raises(ValueError, match='study seal'):
        paper_study_execution._training_gate(study)


def test_training_gate_preserves_optimizer_rng_and_result_bytes(tmp_path, monkeypatch):
    import shutil
    from werewolf.tom import training

    experiment, _ = prepared_experiment(tmp_path)
    # 仅在干净的合成 Fixture 中使用空 gate 作为数值对照，不增加生产执行路径。
    with monkeypatch.context() as control:
        control.setattr(training, '_training_gate', lambda experiment: None)
        training._train_worker(str(experiment.path), experiment.digest, 0, 'implicit')
    def records():
        return {str(p.relative_to(experiment.runs_path)): p.read_bytes()
                for p in experiment.runs_path.rglob('*') if p.is_file()}
    expected = records()
    assert experiment.runs_path.is_relative_to(tmp_path)
    shutil.rmtree(experiment.runs_path)
    training._train_worker(str(experiment.path), experiment.digest, 0, 'implicit')
    assert records() == expected
