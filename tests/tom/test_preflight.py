from dataclasses import asdict

import pytest

from tests.tom.test_experiment import prepared_experiment, experiment_config


@pytest.mark.parametrize("violation", ["mask_role", "mask_boolean", "mask_parent", "bootstrap_version", "bootstrap_field", "bootstrap_range"])
def test_preflight_rejects_semantic_corruption_even_with_valid_outer_hashes(tmp_path, violation):
    from werewolf.artifact_io import canonical_json_bytes, publish_artifact, sha256_bytes
    from werewolf.cli import validate_artifact
    from werewolf.tom.experiment import read_json

    experiment, _ = prepared_experiment(tmp_path)
    files = {p: experiment.file(p) for p in experiment.manifest["file_table"]}
    if violation.startswith("mask"):
        relative = next(p for p in files if "/held_out_primary/" in p)
        value = read_json(files[relative])
        if violation == "mask_role":
            value["role"] = "Werewolf"
        elif violation == "mask_boolean":
            next(iter(value["rows"].values()))[0] = 1
        else:
            value["parent_digest"] = "0" * 64
    else:
        relative = "bootstrap/game_cluster_bootstrap_plan.manifest.json"
        value = read_json(files[relative])
        if violation == "bootstrap_version":
            value["schema_version"] = "other"
        elif violation == "bootstrap_field":
            value["scope"] = "all_alive"
        else:
            import numpy as np
            payload = "bootstrap/game_cluster_bootstrap_indices.bin"
            data = np.full(value["shape"], 5, dtype="<i4").tobytes()
            files[payload] = data
            value["digest"] = sha256_bytes(data)
    files[relative] = canonical_json_bytes(value)
    fields = {k: v for k, v in experiment.manifest.items() if k not in {"manifest_digest", "file_table"}}
    destination = tmp_path / "corrupt"
    publish_artifact(destination, manifest_name="experiment_manifest.json", manifest_fields=fields, files=files)
    with pytest.raises(ValueError, match="eligibility|bootstrap"):
        validate_artifact(destination)


def test_cli_prepares_and_semantically_validates_same_experiment(tmp_path, capsys):
    from werewolf.artifact_io import canonical_json_bytes
    from werewolf.cli import main
    from tests.development_publication.test_development_publication import _publication
    _, handles = _publication(tmp_path)
    protocol = tmp_path / "protocol.json"
    protocol.write_bytes(canonical_json_bytes(asdict(experiment_config(handles.public.public_view.max_structured_token_count))))
    destination = tmp_path / "experiments" / "cli"
    assert main(["prepare-experiment", "--publication", str(handles.public.path), "--protocol", str(protocol), "--destination", str(destination)]) == 0
    identity = capsys.readouterr().out.strip()
    assert len(identity) == 64
    assert main(["validate-artifact", str(destination)]) == 0
    assert capsys.readouterr().out.strip() == identity


def test_runtime_provenance_binds_observed_source_before_training_or_evaluation(tmp_path, monkeypatch):
    from werewolf.tom import experiment as module
    from werewolf.tom.training import _train_worker
    experiment, _ = prepared_experiment(tmp_path)
    observed = module.runtime_provenance(experiment.config)
    assert len(observed["implementation_digest"]) == 64
    assert observed["implementation_digest"] != experiment.config.source_revision
    monkeypatch.setattr(module, "runtime_provenance", lambda config: {**observed, "implementation_digest": "0" * 64})
    with pytest.raises(ValueError, match="runtime"):
        _train_worker(str(experiment.path), experiment.digest, 0, "implicit")
    assert not experiment.runs_path.exists()


def test_cli_preflight_cannot_open_held_out_records_during_training(tmp_path, monkeypatch):
    from pathlib import Path
    from werewolf.cli import validate_artifact
    from werewolf.tom.run_records import publish_record
    experiment, _ = prepared_experiment(tmp_path)
    publish_record(experiment.runs_path / "implicit" / "0" / "training_manifest.json", {"started": True})
    read = Path.read_bytes

    def no_held_out_access(path):
        assert not ("public" in path.parts and "games" in path.parts)
        assert not any(p.startswith("held_out_") for p in path.parts)
        return read(path)

    monkeypatch.setattr(Path, "read_bytes", no_held_out_access)
    with pytest.raises(ValueError, match="seal"):
        validate_artifact(experiment.path)


@pytest.mark.parametrize("partition", ["training_primary", "held_out_primary"])
@pytest.mark.parametrize("membership", ["include_wolves", "omit_non_wolf"])
def test_primary_rejects_wrong_boolean_membership_with_self_consistent_digests(tmp_path, partition, membership):
    from werewolf.artifact_io import canonical_json_bytes, publish_artifact
    from werewolf.cli import validate_artifact
    from werewolf.tom.experiment import open_experiment, read_json
    from werewolf.tom.population import load_training_primary, load_held_out_primary

    experiment, handles = prepared_experiment(tmp_path)
    files = {p: experiment.file(p) for p in experiment.manifest["file_table"]}
    relative = next(p for p in files if f"/{partition}/" in p)
    value = read_json(files[relative])
    game = handles.public.public_view.load_game(value["game_id"])
    if membership == "include_wolves":
        value["rows"] = {p.boundary_id: [f"player{i}" in p.alive_observer_ids for i in range(1, 8)]
                         for p in game.authoritative_pre_prefixes}
    else:
        row = next(iter(value["rows"].values()))
        row[row.index(True)] = False
    files[relative] = canonical_json_bytes(value)
    fields = {k: v for k, v in experiment.manifest.items() if k not in {"manifest_digest", "file_table"}}
    destination = tmp_path / "wrong-membership"
    publish_artifact(destination, manifest_name="experiment_manifest.json", manifest_fields=fields, files=files)
    wrong = open_experiment(destination)
    loader = load_training_primary if partition == "training_primary" else load_held_out_primary
    with pytest.raises(ValueError, match="Primary membership"):
        loader(wrong, int(relative.split("/")[1]), game)
    with pytest.raises(ValueError, match="Primary membership"):
        validate_artifact(destination)
