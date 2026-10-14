"""Synthetic source-to-publication tests, with real immutable artifact I/O."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tests.observer_dtrs.test_oof import oof_factory
from tests.development_publication.test_development_publication import _republish
from werewolf.artifact_io import ArtifactConflictError, ArtifactValidationError, canonical_json_bytes, canonical_jsonl_bytes
from werewolf.observer_dtrs.dataset import open_dataset
from werewolf.observer_dtrs.dataset_offline import (
    audit_dataset_sources, open_supervision, publish_dataset, verify_dataset,
)
from werewolf.observer_dtrs.features import observer_features


BASE = "d8397ce56c65d6d46b24a6b53f0b71431c04267b"


@pytest.fixture
def published(tmp_path, oof_factory, publication_factory):
    def create(*, phase="discussion", dead=(), actor="player1"):
        pub = publication_factory(phase=phase, dead=dead, actor=actor,
            supports={"player3": (), "player4": ("player2",), "player5": ("player1",),
                      "player6": ("player7",), "player7": ("player6",)})
        fixture = oof_factory(publication=pub)
        artifact = publish_dataset(tmp_path / "dataset", dataset_id="controlled-observer-v1",
            publication_path=fixture.publication, evaluation_path=fixture.evaluation,
            expected_seal_digest=fixture.seal_digest, base_revision=BASE)
        return artifact, fixture
    return create


def republish(artifact, destination, *, filename=None, edit=None, manifest_edit=None):
    files = {}
    if filename:
        raw = (artifact.path / filename).read_bytes()
        value = [json.loads(line) for line in raw.splitlines()] if filename.endswith(".jsonl") else json.loads(raw)
        edit(value)
        files[filename] = canonical_jsonl_bytes(value) if filename.endswith(".jsonl") else canonical_json_bytes(value)
    fields = deepcopy(artifact.manifest)
    if manifest_edit:
        manifest_edit(fields)
    return _republish(artifact.path, destination, files=files,
        manifest_updates={k: v for k, v in fields.items() if k not in {"manifest_digest", "file_table"}})


@pytest.mark.parametrize("phase,dead,actor", (("discussion", (), "player1"),
    ("pk_discussion", (), "player1"), ("pk_discussion", ("player2", "player7"), "player1"),
    ("discussion", (), "player3")))
def test_publish_read_full_audit_and_exact_source_rebuild(published, phase, dead, actor):
    artifact, fixture = published(phase=phase, dead=dead, actor=actor)
    view = open_dataset(artifact.path, expected_digest=artifact.manifest_digest)
    offline = open_supervision(view)
    assert len(view.rows) == len(offline) == len(fixture.samples)
    audit = verify_dataset(artifact.path, expected_digest=artifact.manifest_digest)
    assert audit["overall"]["positive"] + audit["overall"]["negative"] == len(offline)
    assert len(audit["by_game"]) == 5
    assert audit["overall"]["successful_empty"] == (0 if actor == "player3" else 5)
    for original, row in zip(fixture.samples, offline, strict=True):
        assert original.context == row.legal.context
        assert original.supervision == row.supervision and original.source == row.sample_source
        assert observer_features(row.legal.context, row.legal.probability) == row.legal.features
        with pytest.raises(TypeError):
            observer_features(row, row.legal.probability)
    report = audit_dataset_sources(artifact.path, publication_path=fixture.publication,
        evaluation_path=fixture.evaluation, expected_seal_digest=fixture.seal_digest,
        expected_digest=artifact.manifest_digest)
    assert report["source_reconstruction"] == "exact_match"
    again = publish_dataset(artifact.path, dataset_id="controlled-observer-v1",
        publication_path=fixture.publication, evaluation_path=fixture.evaluation,
        expected_seal_digest=fixture.seal_digest, base_revision=BASE)
    assert again.manifest_digest == artifact.manifest_digest
    with pytest.raises(ArtifactConflictError):
        publish_dataset(artifact.path, dataset_id="different-identity", publication_path=fixture.publication,
            evaluation_path=fixture.evaluation, expected_seal_digest=fixture.seal_digest, base_revision=BASE)


def test_partition_does_not_open_other_games_or_restricted_payloads(published, monkeypatch):
    artifact, _ = published()
    game = artifact.manifest["games"][2]["game_id"]
    original = Path.read_bytes
    accessed = []

    def guarded(path):
        relative = path.relative_to(artifact.path).as_posix()
        accessed.append(relative)
        assert relative in {"manifest.json", f"legal/games/{game}.jsonl"}
        return original(path)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_bytes", guarded)
        view = open_dataset(artifact.path, game_ids=(game,))
    assert {r.context.game_id for r in view.rows} == {game}
    assert f"legal/games/{game}.jsonl" in accessed
    offline = open_supervision(view)
    assert {r.legal.context.game_id for r in offline} == {game}
    for ids in ((), (game, game), ("unknown",)):
        with pytest.raises(ArtifactValidationError):
            open_dataset(artifact.path, game_ids=ids)
    with pytest.raises(TypeError):
        open_supervision(artifact.path)


@pytest.mark.parametrize("part", ("legal/games", "restricted/supervision", "restricted/provenance"))
@pytest.mark.parametrize("mode", ("missing", "duplicate", "identity", "semantic"))
def test_rehashed_payload_corruption_detected(published, tmp_path, part, mode):
    artifact, _ = published()
    game = artifact.manifest["games"][0]["game_id"]

    def edit(rows):
        if mode == "missing":
            rows.pop()
        elif mode == "duplicate":
            rows.append(deepcopy(rows[0]))
        elif mode == "identity":
            rows[0]["sample_id"] = "f" * 64
        elif part == "legal/games":
            rows[0]["features"]["r"] = 0.99
        elif part == "restricted/supervision":
            rows[0]["supervision"]["event_label_y"] = 0
        else:
            rows[0]["sample_source"]["fold_index"] = (rows[0]["sample_source"]["fold_index"] + 1) % 5
    bad = republish(artifact, tmp_path / "bad", filename=f"{part}/{game}.jsonl", edit=edit)
    with pytest.raises(ArtifactValidationError):
        verify_dataset(bad.path)


@pytest.mark.parametrize("filename", ("restricted/audit.json", "restricted/implementation_sources.json",
                                      "restricted/upstream_manifests.json"))
def test_rehashed_audit_or_source_metadata_corruption(published, tmp_path, filename):
    artifact, _ = published()

    def edit(value):
        if filename.endswith("audit.json"):
            value["overall"]["positive"] += 1
        elif filename.endswith("implementation_sources.json"):
            value["base_revision"] = "f" * 40
        else:
            value["publication"]["role_sidecar_digest"] = "f" * 64
    bad = republish(artifact, tmp_path / "bad", filename=filename, edit=edit)
    with pytest.raises(ArtifactValidationError):
        verify_dataset(bad.path)


def test_truncated_changed_bytes_and_immutable_identity(published):
    artifact, _ = published()
    with pytest.raises(ArtifactValidationError, match="trust anchor"):
        open_dataset(artifact.path, expected_digest="f" * 64)
    path = artifact.path / "restricted/audit.json"
    path.write_bytes(path.read_bytes()[:-1])
    # The legal interface deliberately never reads the restricted partition.
    assert len(open_dataset(artifact.path).rows) == 25
    with pytest.raises(ArtifactValidationError, match="size mismatch"):
        verify_dataset(artifact.path)


def test_manifest_cross_game_fold_tamper_is_detected(published, tmp_path):
    artifact, _ = published()
    bad = republish(artifact, tmp_path / "bad", manifest_edit=lambda m:
        m["games"][0].update(fold_index=(m["games"][0]["fold_index"] + 1) % 5))
    with pytest.raises(ArtifactValidationError):
        verify_dataset(bad.path)


def test_locally_consistent_counterfactual_q_requires_source_rebuild(published, tmp_path):
    artifact, fixture = published()
    game = artifact.manifest["games"][0]["game_id"]
    view = open_dataset(artifact.path)
    original_row = view.rows[0]
    values = tuple(v * 0.5 + (0 if i == 2 else 1 / 12)
                   for i, v in enumerate(original_row.probability.values))
    from werewolf.observer_dtrs.probabilities import ProbabilityRow
    q = ProbabilityRow(game, original_row.context.boundary_id, original_row.context.prefix_digest,
                        original_row.context.observer_id, values)

    def edit(rows):
        rows[0]["probability"] = values
        rows[0]["features"] = observer_features(original_row.context, q).to_record()
    bad = republish(artifact, tmp_path / "counterfactual", filename=f"legal/games/{game}.jsonl", edit=edit)
    assert verify_dataset(bad.path)["overall"]["sample_count"] == 25
    with pytest.raises(ArtifactValidationError, match="source reconstruction differs"):
        audit_dataset_sources(bad.path, publication_path=fixture.publication,
            evaluation_path=fixture.evaluation, expected_seal_digest=fixture.seal_digest)


def test_legal_reader_imports_no_supervision_or_model_modules():
    code = "import sys,json;from werewolf.observer_dtrs.dataset import open_dataset;print(json.dumps(sorted(sys.modules)))"
    result = subprocess.run([sys.executable, "-B", "-c", code], check=True,
        capture_output=True, text=True, cwd=Path(__file__).resolve().parents[2])
    modules = json.loads(result.stdout)
    assert not any(m.startswith(("werewolf.observer_dtrs.dataset_offline", "werewolf.observer_dtrs.labels",
        "werewolf.observer_dtrs.offline", "werewolf.observer_dtrs.oof", "werewolf.development_publication",
        "werewolf.phase2", "werewolf.tom", "torch", "sklearn")) for m in modules)


def test_invalid_upstream_is_not_published(oof_factory, tmp_path):
    def corrupt(stage, value, fold):
        if stage == "row" and fold == 0:
            value["prefix_digest"] = "f" * 64
    fixture = oof_factory(mutate=corrupt)
    destination = tmp_path / "must-not-exist"
    with pytest.raises(ArtifactValidationError):
        publish_dataset(destination, dataset_id="controlled", publication_path=fixture.publication,
            evaluation_path=fixture.evaluation, expected_seal_digest=fixture.seal_digest, base_revision=BASE)
    assert not destination.exists()
    with pytest.raises(ArtifactValidationError, match="overlaps"):
        publish_dataset(fixture.publication / "wrong", dataset_id="controlled", publication_path=fixture.publication,
            evaluation_path=fixture.evaluation, expected_seal_digest=fixture.seal_digest, base_revision=BASE)


def test_cli_storage_guard_and_independent_verify(published, tmp_path, capsys):
    from scripts.manage_observer_dtrs_dataset import main
    artifact, fixture = published()
    assert main(["verify", "--dataset", str(artifact.path),
                 "--expected-dataset-digest", artifact.manifest_digest]) == 0
    assert json.loads(capsys.readouterr().out)["overall"]["sample_count"] == 25
    profile = tmp_path / "storage.json"
    profile.write_text(json.dumps({"artifact_root": str(tmp_path / "allowed")}))
    with pytest.raises(SystemExit):
        main(["publish", "--destination", str(tmp_path / "outside"), "--dataset-id", "controlled",
            "--base-revision", BASE, "--publication", str(fixture.publication),
            "--oof-root", str(fixture.evaluation), "--expected-oof-seal-digest", fixture.seal_digest,
            "--storage-profile", str(profile)])
