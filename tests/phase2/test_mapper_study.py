"""Synthetic runner checks; no sealed development publication is read."""

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_phase2_mapper_study as study
from werewolf.artifact_io import verify_artifact
from werewolf.phase2_offline import CandidateRow, OOF_SEAL_DIGEST, R2Input, ResolvedOutcome
from werewolf.phase2_mapper import OOFProvenance


def synthetic_rows():
    rows = []
    for fold in range(5):
        game = f"game-{fold}"
        for index in range(4):
            phase = "speech_pk" if fold % 2 else "speech"
            z = R2Input(.1 + .08 * index + .01 * fold,
                        -.3 + .1 * index - .02 * fold,
                        .02 * (index + fold), .15 * index + .03 * fold,
                        phase, 2 + fold % 3, 2 + fold % 4, index % 3)
            rows.append(CandidateRow(game, f"boundary-{fold}", "player1",
                phase, f"player{index + 3}", fold, f"prefix-{fold}", (2, 5),
                (index + fold) % 3 == 0, z, ResolvedOutcome.NO_EXILE))
    return tuple(rows)


def configure_synthetic(monkeypatch, tmp_path):
    rows = synthetic_rows()
    layer = SimpleNamespace(candidates=rows, post_days=tuple(range(5)),
                            pre_count=5, skipped_no_candidate_pk_pre_count=0)
    oof = OOFProvenance(OOF_SEAL_DIGEST, "a" * 64,
        {fold: sha256(f"prediction-{fold}".encode()).hexdigest() for fold in range(5)})
    monkeypatch.setattr(study, "attest_source", lambda: {
        "commit": "a" * 40, "branch": "twd/mainline", "tracked_clean": True,
        "dirty": False, "status_porcelain": [],
        "source_sha256": {path: "b" * 64 for path in study.SOURCE_FILES},
    })
    monkeypatch.setattr(study.OOFProvenance, "from_evaluation_root",
                        classmethod(lambda cls, root, **_kwargs: oof))
    monkeypatch.setattr(study.offline, "build_development_layer",
                        lambda publication, evaluation, **_kwargs: layer)
    monkeypatch.setattr(study.offline, "EXPECTED_GAMES", 5)
    monkeypatch.setattr(study.offline, "EXPECTED_PRE", 5)
    monkeypatch.setattr(study.offline, "EXPECTED_CANDIDATES", 20)
    monkeypatch.setattr(study.offline, "EXPECTED_POST_DAYS", 5)
    monkeypatch.setattr(study.offline, "EXPECTED_SKIPPED_NO_CANDIDATE_PK_PRE", 0)
    monkeypatch.setattr(study, "EXPECTED_POSITIVES", sum(row.theta_ac for row in rows))
    monkeypatch.setattr(study, "DEFAULT_REPLICATES", 12)
    root = tmp_path / "artifacts"
    root.mkdir()
    profile = tmp_path / "storage.json"
    profile.write_text(json.dumps({"artifact_root": str(root)}), encoding="utf-8")
    return profile, layer, oof


def test_formal_study_orchestrates_frozen_functions_and_publishes_artifact(monkeypatch, tmp_path):
    profile, layer, _ = configure_synthetic(monkeypatch, tmp_path)
    artifact = study.run_study(profile)
    verified = verify_artifact(artifact.path,
        expected_artifact_type="phase2_mapper_development_study",
        expected_schema_version=study.STUDY_VERSION)
    manifest = verified.manifest
    assert manifest["source"]["commit"] == "a" * 40
    assert manifest["population"]["candidate_rows"] == 20
    assert manifest["population"]["theta_positive_rows"] == sum(
        row.theta_ac for row in layer.candidates)
    assert manifest["protocol"]["bootstrap"]["replicates"] == 12
    assert len(manifest["fit_model_digests"]) == 20
    assert len(manifest["folds"]) == 5
    assert manifest["inputs"]["oof"]["evaluation_seal_digest"] == OOF_SEAL_DIGEST
    for spec in study.MODEL_SPECS:
        lines = (artifact.path / "cross_fitted_predictions" /
                 f"{spec.name}.jsonl").read_text().splitlines()
        assert len(lines) == 20
        assert len({tuple(json.loads(line)[field] for field in
                    ("game_id", "boundary_id", "acting_wolf", "phase", "candidate_j"))
                    for line in lines}) == 20
    assert set(json.loads((artifact.path / "metrics.json").read_text())) == set(study.MODEL_ORDER)
    assert len(json.loads((artifact.path / "bootstrap.json").read_text())["pairs"]) == 5
    assert "Primary selected specification" in (artifact.path / "report.md").read_text()
    with pytest.raises(study.MapperStudyError, match="already exists"):
        study.run_study(profile)


@pytest.mark.parametrize("owner,constant", [
    ("offline", "EXPECTED_GAMES"),
    ("offline", "EXPECTED_PRE"),
    ("offline", "EXPECTED_CANDIDATES"),
    ("study", "EXPECTED_POSITIVES"),
    ("offline", "EXPECTED_POST_DAYS"),
    ("offline", "EXPECTED_SKIPPED_NO_CANDIDATE_PK_PRE"),
])
def test_all_six_counts_fail_before_any_mapper_fit(monkeypatch, tmp_path, owner, constant):
    profile, layer, _ = configure_synthetic(monkeypatch, tmp_path)
    module = study if owner == "study" else study.offline
    monkeypatch.setattr(module, constant, getattr(module, constant) + 1)
    monkeypatch.setattr(study, "run_mapper_cv",
                        lambda *_args: pytest.fail("mapper fit started before count check"))
    with pytest.raises(study.MapperStudyError, match="formal population mismatch before fitting"):
        study.run_study(profile)
    assert not (tmp_path / "artifacts" / study.OUTPUT_RELATIVE).exists()


def test_duplicate_prediction_or_training_leak_fails_runner_audit(monkeypatch, tmp_path):
    _, layer, oof = configure_synthetic(monkeypatch, tmp_path)
    population = study.validate_population(layer, oof)
    result = study.run_mapper_cv(layer.candidates, oof)
    repeated = replace(result, predictions=(*result.predictions, result.predictions[0]))
    with pytest.raises(study.MapperStudyError, match="duplicate"):
        study.validate_cv(repeated, layer.candidates, population)
    altered_manifest = dict(result.fit_manifests[0])
    altered_manifest["training_game_ids"] = ["game-0", *altered_manifest["training_game_ids"]]
    leaked = replace(result, fit_manifests=(altered_manifest, *result.fit_manifests[1:]))
    with pytest.raises(study.MapperStudyError, match="held-out game"):
        study.validate_cv(leaked, layer.candidates, population)


def test_preflight_does_not_read_data_or_train(monkeypatch, tmp_path, capsys):
    profile, _, _ = configure_synthetic(monkeypatch, tmp_path)
    monkeypatch.setattr(study.offline, "build_development_layer",
                        lambda *_args: pytest.fail("preflight read publication"))
    monkeypatch.setattr(study, "run_mapper_cv",
                        lambda *_args: pytest.fail("preflight fit mapper"))
    assert study.main(["--storage-profile", str(profile), "--preflight"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["destination"].endswith(str(study.OUTPUT_RELATIVE))
    assert not Path(printed["destination"]).exists()


def test_cli_help_is_available_without_data(capsys):
    with pytest.raises(SystemExit) as result:
        study.main(["--help"])
    assert result.value.code == 0
    assert "--storage-profile" in capsys.readouterr().out


def test_attestation_rejects_tracked_dirty_source(monkeypatch):
    root = Path(study.__file__).resolve().parents[1]

    def fake_git(_root, *arguments):
        if arguments == ("rev-parse", "--show-toplevel"):
            return f"{root}\n".encode()
        if arguments == ("rev-parse", "HEAD"):
            return ("a" * 40 + "\n").encode()
        if arguments == ("branch", "--show-current"):
            return b"twd/mainline\n"
        if arguments[0] == "status" and "--untracked-files=no" in arguments:
            return b" M werewolf/phase2_mapper.py\n"
        pytest.fail("source attestation proceeded after dirty tracked file")

    monkeypatch.setattr(study, "_git", fake_git)
    with pytest.raises(study.MapperStudyError, match="tracked Git files must be clean"):
        study.attest_source()


def test_attestation_records_untracked_note_without_blocking(monkeypatch):
    root = Path(study.__file__).resolve().parents[1]
    note = "?? docs/research/phase2-wolf-pt3wd-review-2026-09-13.md"

    def fake_git(_root, *arguments):
        if arguments == ("rev-parse", "--show-toplevel"):
            return f"{root}\n".encode()
        if arguments == ("rev-parse", "HEAD"):
            return ("a" * 40 + "\n").encode()
        if arguments == ("branch", "--show-current"):
            return b"twd/mainline\n"
        if arguments[0] == "status":
            return b"" if "--untracked-files=no" in arguments else (note + "\n").encode()
        if arguments[0] == "ls-files":
            return b""
        if arguments[0] == "show":
            return (root / arguments[1].removeprefix("HEAD:")).read_bytes()
        pytest.fail(f"unexpected Git query: {arguments}")

    monkeypatch.setattr(study, "_git", fake_git)
    source = study.attest_source()
    assert source["tracked_clean"] and source["dirty"]
    assert source["status_porcelain"] == [note]
    assert set(source["source_sha256"]) == set(study.SOURCE_FILES)
