"""Prepare/read/score only. No optimizer step or new model training is run."""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.development_publication.test_canonical_split_v2 import _collection
from tests.development_publication.test_development_publication import _publication, _republish
from tests.tom.test_experiment import experiment_config
from werewolf.artifact_io import ArtifactValidationError, canonical_json_bytes
from werewolf.canonical_split import publish_canonical_split
from werewolf.development_publication import open_verified_collection
from werewolf.development_publication_v2 import publish_development_v2, open_development_publication_v2
from werewolf.tom.development_input import open_development_input
from werewolf.tom.dataset import CanonicalToMDataset, ExperimentCapacity
from werewolf.tom.experiment import prepare_experiment, validate_experiment_preflight, open_experiment
from werewolf.tom.population import select_primary_population, load_training_primary, load_held_out_primary
from werewolf.tom.protocol import load_bootstrap_plan


@pytest.fixture(scope="module")
def v2(tmp_path_factory):
    root = tmp_path_factory.mktemp("v2-training-input")
    protocol, path, replay = _collection(root, target=12, completed=12)
    collection = open_verified_collection(path, replay_executor=replay)
    split = publish_canonical_split(collection, protocol, root / "split", split_id="split")
    publication = publish_development_v2(collection, split, root / "dev", publication_id="dev").public
    return root, collection, split, publication


@pytest.fixture(scope="module")
def experiment(v2):
    root, _, split, publication = v2
    read = open_development_input(publication.path, expected_publication_digest=publication.manifest_digest,
                                  expected_split_digest=split.digest)
    config = experiment_config(read.public_view.max_structured_token_count)
    return prepare_experiment(read, config, root / "experiment")


def test_preparation_folds_population_bootstrap_and_exact_target_semantics(v2, experiment):
    _, _, split, publication = v2
    assert open_experiment(experiment.path).digest == experiment.digest
    validate_experiment_preflight(experiment)
    indices, _ = load_bootstrap_plan(experiment)
    assert indices.shape == (20, 10)
    test_ids = {m["game_id"] for m in split.members("independent_test")}
    assert not set(experiment.manifest["game_ids"]) & test_ids
    all_held_out = []
    for index in range(5):
        fold = experiment.fold(index)
        train, held = fold["training_game_ids"], fold["held_out_game_ids"]
        assert len(train) == 8 and len(held) == 2 and not set(train) & set(held)
        assert not (set(train) | set(held)) & test_ids
        all_held_out.extend(held)
        data = open_development_input(publication.path, expected_publication_digest=publication.manifest_digest, game_ids=train)
        dataset = CanonicalToMDataset(data.public_view, train, ExperimentCapacity(experiment.config.max_seq_len))
        for sample in dataset:
            assert sample.game_id in train and sample.game_id not in held
            assert torch.equal(sample.q.diagonal(), torch.zeros(7))
            assert torch.allclose(sample.q.sum(-1), torch.ones(7))
            game = data.public_view.load_game(sample.game_id)
            prefix = next(p for p in game.authoritative_pre_prefixes if p.boundary_id == sample.boundary_id)
            observations = {o.observer_id: o for o in game.belief_observations}
            for seat in range(7):
                support = observations[f"player{seat + 1}"].suspicion_support
                expected_support = list(support) or [f"player{p + 1}" for p in range(7) if p != seat]
                expected = torch.zeros(7)
                for player in expected_support:
                    expected[int(player[6:]) - 1] = 1 / len(expected_support)
                assert torch.allclose(sample.q[seat], expected)
            assert sample.prefix_digest == prefix.prefix_digest
            assert not {"q", "role_assignment", "seed", "game_id"} & set(sample.public.kwargs())
        for game in data.public_view.games:
            mask = load_training_primary(experiment, index, game)["rows"]
            assert mask == select_primary_population(data).rows[game.game_id]
        held_data = open_development_input(publication.path, expected_publication_digest=publication.manifest_digest, game_ids=held)
        for game in held_data.public_view.games:
            assert load_held_out_primary(experiment, index, game)["rows"] == select_primary_population(held_data).rows[game.game_id]
    assert sorted(all_held_out) == experiment.manifest["game_ids"]


def test_worker_reader_does_not_open_test_raw_collection_or_held_out(v2, experiment, monkeypatch):
    _, collection, split, publication = v2
    train = experiment.fold(0)["training_game_ids"]
    original = Path.open
    def guarded(path, *args, **kwargs):
        assert not path.is_relative_to(collection.path)
        if path.parent == publication.path / "public/games":
            assert path.stem in train
        assert "restricted" not in path.parts
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    data = open_development_input(publication.path, expected_publication_digest=publication.manifest_digest, game_ids=train)
    assert set(data.public_view.game_ids) == set(train)


def test_wrong_anchor_and_test_artifact_fail_before_payload_open(v2, monkeypatch):
    _, _, split, publication = v2
    original = Path.open
    def guarded(path, *args, **kwargs):
        assert "games" not in path.parts and "restricted" not in path.parts
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    with pytest.raises(ArtifactValidationError, match="trusted Publication"):
        open_development_input(publication.path)
    with pytest.raises(ArtifactValidationError, match="trusted Publication"):
        open_development_input(publication.path, expected_publication_digest="0" * 64)
    with pytest.raises(ArtifactValidationError, match="trusted Split"):
        open_development_input(publication.path, expected_publication_digest=publication.manifest_digest, expected_split_digest="0" * 64)
    with pytest.raises(ArtifactValidationError, match="not a training input"):
        open_development_input(split.artifact.path, expected_publication_digest=split.digest)


@pytest.mark.parametrize("extra", ["restricted/test_labels.json", "lineage/test_seeds.json", "extra-directory"])
def test_v2_rejects_unlisted_files_and_directories(v2, tmp_path, extra):
    _, _, split, publication = v2
    destination = tmp_path / "dev"
    _republish(publication.path, destination)
    path = destination / extra
    if extra == "extra-directory":
        path.mkdir()
    else:
        path.write_bytes(b"private Test data")
    with pytest.raises(ArtifactValidationError, match="exactly match"):
        open_development_publication_v2(destination, expected_split_digest=split.digest)


def test_v2_role_bytes_are_gated_and_linked_lineage_is_rejected(v2, tmp_path, monkeypatch):
    _, _, split, publication = v2
    original = Path.open
    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", lambda path, *a, **kw: pytest.fail("public reader opened roles")
                      if "restricted" in path.parts else original(path, *a, **kw))
        open_development_input(publication.path, expected_publication_digest=publication.manifest_digest)
    destination = tmp_path / "dev"
    _republish(publication.path, destination)
    target = destination / "lineage/development_membership.json"
    target.unlink()
    target.symlink_to(publication.path / "lineage/development_membership.json")
    with pytest.raises(ArtifactValidationError, match="symbolic link"):
        open_development_input(destination, expected_publication_digest=publication.manifest_digest)


def test_legacy_reader_and_score_functions_are_reused(tmp_path):
    _, handles = _publication(tmp_path)
    assert open_development_input(handles.public.path).manifest_digest == handles.public.manifest_digest


@pytest.mark.parametrize("command", ["prepare-experiment-v2", "prepare-final-experiment-v2"])
def test_explicit_v2_cli_prepares_existing_experiment_types(v2, monkeypatch, capsys, command):
    from werewolf.cli import main
    from werewolf.tom import final_experiment
    root, collection, split, publication = v2
    profile, protocol = root / "storage.json", root / "protocol.json"
    profile.write_bytes(canonical_json_bytes({"artifact_root": str(root)}))
    config = experiment_config(publication.public_view.max_structured_token_count)
    protocol.write_bytes(canonical_json_bytes(asdict(config)))
    monkeypatch.setattr(final_experiment, "actual_clean_revision", lambda: "a" * 40)
    original = Path.open
    def development_only(path, *a, **kw):
        assert not path.is_relative_to(collection.path) and not path.is_relative_to(split.artifact.path)
        return original(path, *a, **kw)
    monkeypatch.setattr(Path, "open", development_only)
    destination = root / "experiments" / command / "experiments/experiment"
    assert main(["--storage-profile", str(profile), command, "--publication", str(publication.path),
        "--publication-digest", publication.manifest_digest, "--split-digest", split.digest,
        "--protocol", str(protocol), "--destination", command]) == 0
    digest = capsys.readouterr().out.strip()
    opened = (final_experiment.open_final_experiment(destination) if command == "prepare-final-experiment-v2"
              else open_experiment(destination))
    assert opened.digest == digest
    assert opened.manifest["game_ids"] == sorted(publication.public_view.game_ids)
    if command == "prepare-final-experiment-v2":
        inputs, primary = final_experiment.final_training_inputs(opened)
        assert inputs.manifest_digest == publication.manifest_digest
        assert set(primary.rows) == set(publication.public_view.game_ids)
    assert not list((destination.parent.parent / "runs").glob("*"))


def test_v2_primary_rows_use_existing_kl_tv_top1_nu_and_paired_game_bootstrap(v2, experiment):
    import math
    from werewolf.tom.evaluation import score_prediction_rows
    from werewolf.tom.backbone_evaluation import _top1_nu, _accumulate_top1_nu
    from werewolf.tom.scoring import game_macro_summary, paired_summary
    _, _, split, publication = v2
    data = CanonicalToMDataset(publication.public_view, publication.public_view.game_ids,
                              ExperimentCapacity(experiment.config.max_seq_len))
    primary = select_primary_population(publication)
    rows = []
    for sample in data:
        for observer in range(7):
            if primary.rows[sample.game_id][sample.boundary_id][observer] and sample.label_observed[observer]:
                rows.append(dict(game_id=sample.game_id, boundary_id=sample.boundary_id, observer=observer,
                    prefix_digest=sample.prefix_digest, q=sample.q[observer].tolist(),
                    non_self_log_probability=[-math.log(6)] * 6))
    ids = experiment.manifest["game_ids"]
    scores, games = score_prediction_rows(rows, ids)
    assert set(games) == set(ids) and not set(games) & {m["game_id"] for m in split.members("independent_test")}
    assert all(score["kl"] == pytest.approx(score["uniform_kl"]) for score in scores)
    assert all(0 <= score["total_variation"] <= 1 for score in scores)
    counts = {}
    excluded = _accumulate_top1_nu(counts, rows, scores)
    nu = _top1_nu(counts, excluded, len(rows), len(ids))
    assert nu["excluded_s6_row_count"] + nu["valid_nu_row_count"] == len(rows)
    indices, _ = load_bootstrap_plan(experiment)
    values = [games[game_id]["kl"] for game_id in ids]
    assert game_macro_summary(values, indices, .95)["sampling_unit"] == "game"
    assert paired_summary(values, values, indices, .95)["interval"] == [0., 0.]


def test_qwen3_v2_prepared_study_and_execution_read_without_training(v2, tmp_path, monkeypatch):
    from werewolf.tom import backbone_study as study, backbone_execution as execution, backbone_evaluation as evaluation
    _, _, split, publication = v2
    base = study._frozen_design()
    base["reference_protocol"]["device"] = "cpu"
    monkeypatch.setattr(study, "_frozen_design", lambda: deepcopy(base))
    design = {**base, "schema_version": study.STUDY_V2_VERSION, "publication_id": "dev", "game_count": 10}
    config_path = tmp_path / "design.json"
    config_path.write_bytes(canonical_json_bytes(design))
    from werewolf.tom.experiment import ExperimentConfig, runtime_provenance
    config = ExperimentConfig(**design["reference_protocol"])
    source = {"source_revision": "a" * 40, "implementation_digest": runtime_provenance(config)["implementation_digest"]}
    monkeypatch.setattr(study, "attest_source", lambda: source)
    monkeypatch.setattr(execution, "attest_source", lambda: source)
    artifact = study.prepare_study(config_path, publication.path, tmp_path / "study",
        publication_digest=publication.manifest_digest, split_digest=split.digest)
    assert study.open_study(artifact.path).manifest_digest == artifact.manifest_digest
    snapshot = execution.validate_frozen_snapshot(artifact.path, execution.snapshot_identity(artifact))
    assert execution.formal_optimizer_steps(snapshot) == 168
    prepared_execution = execution.create_execution(artifact.path, artifact.manifest_digest)
    reopened = execution.open_execution(prepared_execution.root)
    assert reopened.manifest["expected_optimizer_steps"] == 168
    assert not (reopened.root / "runs").exists()
    owner = SimpleNamespace(execution=reopened)
    data, masks = evaluation._held_out(owner, 0)
    assert {s.game_id for s in data} == set(snapshot.manifest["folds"][0]["held_out_game_ids"])
    assert set(masks) <= set(publication.public_view.game_ids)
    training_data, training_masks = execution._materialize(reopened, 0)
    assert set(training_data) == set(training_masks) == set(snapshot.manifest["folds"][0]["training_game_ids"])
    assert not set(training_data) & {s.game_id for s in data}
    from werewolf.cli import main
    profile = tmp_path / "cli-storage.json"
    profile.write_bytes(canonical_json_bytes({"artifact_root": str(v2[0])}))
    destination = v2[0] / "cli-backbone-study"
    assert main(["--storage-profile", str(profile), "prepare-backbone-tom-study-v2",
        "--publication", str(publication.path), "--publication-digest", publication.manifest_digest,
        "--split-digest", split.digest, "--config", str(config_path), "--destination", str(destination)]) == 0
    assert study.open_study(destination).manifest["schema_version"] == study.STUDY_V2_VERSION


@pytest.mark.parametrize("change", ["root_link", "ancestor_link", "manifest_link", "parent_traversal"])
def test_adapter_rejects_illegal_paths_before_any_payload(v2, tmp_path, monkeypatch, change):
    _, _, split, publication = v2
    if change == "root_link":
        path = tmp_path / "linked"
        path.symlink_to(publication.path, target_is_directory=True)
    elif change == "ancestor_link":
        ancestor = tmp_path / "linked-parent"
        ancestor.symlink_to(publication.path.parent, target_is_directory=True)
        path = ancestor / publication.path.name
    elif change == "manifest_link":
        path = tmp_path / "dev"
        _republish(publication.path, path)
        (path / "manifest.json").unlink()
        (path / "manifest.json").symlink_to(publication.path / "manifest.json")
    else:
        path = publication.path / ".." / publication.path.name
    monkeypatch.setattr(Path, "open", lambda *a, **kw: pytest.fail("illegal path opened a file"))
    with pytest.raises(ArtifactValidationError):
        open_development_input(path, expected_publication_digest=publication.manifest_digest,
                               expected_split_digest=split.digest)


@pytest.mark.parametrize("record", [[], {"artifact_type": "development_publication_v2", "schema_version": "unknown"},
    {"artifact_type": "final_evaluation_publication", "schema_version": "classic7_final_evaluation_publication_v1"}])
def test_unknown_or_test_schema_rejects_without_payload(tmp_path, monkeypatch, record):
    path = tmp_path / "not-development"
    path.mkdir()
    (path / "manifest.json").write_bytes(canonical_json_bytes(record))
    original = Path.open
    def manifest_only(file, *args, **kwargs):
        assert file == path / "manifest.json"
        return original(file, *args, **kwargs)
    monkeypatch.setattr(Path, "open", manifest_only)
    with pytest.raises(ArtifactValidationError):
        open_development_input(path, expected_publication_digest="0" * 64)


def test_v1_rejects_split_identity_and_v2_rejects_test_member(v2, tmp_path):
    _, handles = _publication(tmp_path)
    with pytest.raises(ArtifactValidationError, match="V1.*no Split"):
        open_development_input(handles.public.path, expected_split_digest="a" * 64)
    _, _, split, publication = v2
    member = split.members("independent_test")[0]["game_id"]  # Synthetic fixture only.
    with pytest.raises(ArtifactValidationError, match="outside Development"):
        open_development_input(publication.path, expected_publication_digest=publication.manifest_digest,
                               game_ids=[member])


@pytest.mark.parametrize("command", ["prepare-experiment-v2", "prepare-final-experiment-v2", "prepare-backbone-tom-study-v2"])
@pytest.mark.parametrize("missing", ["--publication-digest", "--split-digest"])
def test_v2_cli_requires_both_trust_anchors(command, missing):
    from werewolf.cli import build_parser
    args = [command, "--publication", "not-opened", "--destination", "not-created",
            "--config" if command == "prepare-backbone-tom-study-v2" else "--protocol", "not-opened"]
    for flag in ("--publication-digest", "--split-digest"):
        if flag != missing:
            args += [flag, "a" * 64]
    with pytest.raises(SystemExit):
        build_parser().parse_args(args)


@pytest.mark.parametrize("change", ["learning_rate", "fold_count", "temporal_conditions", "initialization"])
def test_v2_design_cannot_change_frozen_numerical_controls(change):
    from werewolf.tom import backbone_study as study
    design = {**study._frozen_design(), "schema_version": study.STUDY_V2_VERSION,
              "publication_id": "synthetic-dev", "game_count": 10}
    if change == "learning_rate":
        design["reference_protocol"]["learning_rate"] *= 2
    elif change == "fold_count":
        design[change] = 4
    else:
        design[change] = []
    with pytest.raises(ValueError, match="frozen design"):
        study.validated_design(design, study.STUDY_V2_VERSION)


@pytest.mark.parametrize("kind", ["publication_only", "split_only", "paired_v2"])
def test_v2_preparation_rejects_unregistered_contract_before_source_or_data(tmp_path, monkeypatch, kind):
    from werewolf.tom import backbone_study as study
    config = tmp_path / "config.json"
    config.write_bytes(canonical_json_bytes(study._frozen_design()))
    monkeypatch.setattr(study, "attest_source", lambda: pytest.fail("invalid contract reached source/data"))
    kwargs = {"publication_digest": "a" * 64, "split_digest": "b" * 64}
    if kind == "publication_only":
        kwargs.pop("split_digest")
    elif kind == "split_only":
        kwargs.pop("publication_digest")
    else:
        kwargs["paired_temporal"] = True
    with pytest.raises(ValueError, match="requires trusted|not a registered"):
        study.prepare_study(config, tmp_path / "absent", tmp_path / "output", **kwargs)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("schema", ["classic7_backbone_study_v1", "classic7_qwen3_paired_temporal_study_v2"])
def test_legacy_and_paired_formal_budgets_remain_fixed(schema):
    from werewolf.tom.backbone_execution import formal_optimizer_steps
    snapshot = SimpleNamespace(manifest={"schema_version": schema, "folds": [{"optimizer_steps": 168}] * 5})
    with pytest.raises(ValueError, match="remains 25200"):
        formal_optimizer_steps(snapshot)
    snapshot.manifest["folds"] = [{"optimizer_steps": 25200}] * 5
    assert formal_optimizer_steps(snapshot) == 25200
