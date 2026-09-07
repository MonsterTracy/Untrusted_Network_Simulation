from pathlib import Path

import pytest

from tests.tom.test_experiment import prepared_experiment


def test_all_ten_seal_then_four_cells_and_paired_reports(tmp_path, monkeypatch):
    from werewolf.tom.reporting import run_development_oof
    from werewolf.tom.training import train_primary_fold

    experiment, _ = prepared_experiment(tmp_path)
    read = Path.read_bytes
    def gated(path):
        if "public" in path.parts and path.parent.name == "games":
            assert (experiment.runs_path / "checkpoint_set_manifest.json").is_file()
        return read(path)
    monkeypatch.setattr(Path, "read_bytes", gated)
    result = run_development_oof(experiment)
    assert len(result["cells"]) == 4
    assert set(result["paired_headlines"]) == {"identifiability_stress_penalty", "primary_temporal_information_effect"}
    for cell in result["cells"].values():
        assert cell["headline"]["aggregation"] == "game_macro"
        assert cell["headline"]["sampling_unit"] == "game"
        assert len(cell["game_scores"]) == 5
        assert "interval" not in cell["observer_row_weighted_diagnostic"]
    with pytest.raises(ValueError, match="seal"):
        train_primary_fold(experiment, 0, "implicit")
    assert run_development_oof(experiment) == result


@pytest.fixture(scope="module")
def completed_experiment(tmp_path_factory):
    from werewolf.tom.reporting import run_development_oof
    experiment, _ = prepared_experiment(tmp_path_factory.mktemp("completed-oof-validation"))
    run_development_oof(experiment)
    return experiment


@pytest.mark.parametrize("relative", [
    "implicit/0/held_out_predictions.jsonl",
    "implicit/0/primary_development_oof/fold_report.json",
    "implicit/0/all_alive_identifiability_stress/fold_report.json",
    "reports/implicit/primary_development_oof/aggregate_report.json",
    "reports/implicit/all_alive_identifiability_stress/aggregate_report.json",
    "reports/oof_report_set.json",
    "reports/implicit/primary_development_oof/worst_cases.jsonl",
])
def test_cli_validates_existing_evaluation_payloads_without_writing(completed_experiment, monkeypatch, relative):
    from werewolf.cli import validate_artifact
    experiment = completed_experiment
    before = {p: p.read_bytes() for p in experiment.runs_path.rglob("*") if p.is_file()}
    assert validate_artifact(experiment.path) == experiment.digest
    assert {p: p.read_bytes() for p in experiment.runs_path.rglob("*") if p.is_file()} == before
    target = experiment.runs_path / relative
    assert target in before
    read = Path.read_bytes
    with monkeypatch.context() as corruption:
        corruption.setattr(Path, "read_bytes", lambda path: b"corrupt" if path == target else read(path))
        with pytest.raises(ValueError):
            validate_artifact(experiment.path)


@pytest.mark.parametrize("relative", [
    "implicit/0/primary_development_oof/fold_report.json",
    "reports/implicit/primary_development_oof/aggregate_report.json",
    "reports/oof_report_set.json",
])
@pytest.mark.parametrize("violation", ["parent", "schema", "field", "score"])
def test_cli_rejects_rehashed_report_semantic_corruption(completed_experiment, monkeypatch, relative, violation):
    from werewolf.artifact_io import canonical_json_bytes
    from werewolf.cli import validate_artifact
    from werewolf.tom.run_records import read_record, record_with_digest
    experiment = completed_experiment
    target = experiment.runs_path / relative
    value = read_record(target)
    del value["record_digest"]
    if violation == "parent":
        value["checkpoint_set_digest"] = "0" * 64
    elif violation == "schema":
        value["schema_version"] = "other"
    elif violation == "field":
        value["scope"] = "all_alive"
    elif "game_scores" in value:
        next(iter(value["game_scores"].values()))["kl"] += 1
    else:
        value["paired_headlines"] = {}
    payload = canonical_json_bytes(record_with_digest(value))
    read = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda path: payload if path == target else read(path))
    with pytest.raises(ValueError, match="report semantic/provenance"):
        validate_artifact(experiment.path)


def test_cli_validates_partial_evaluation_without_creating_missing_outputs(completed_experiment, monkeypatch):
    from werewolf.cli import validate_artifact
    experiment = completed_experiment
    outputs = {p for p in experiment.runs_path.rglob("*") if p.is_file() and (
        "reports" in p.parts or p.name in {"prediction_manifest.json", "held_out_predictions.jsonl", "fold_report.json"})}
    visible = {experiment.runs_path / "implicit" / "0" / name for name in (
        "prediction_manifest.json", "held_out_predictions.jsonl", "primary_development_oof/fold_report.json")}
    hidden = outputs - visible
    exists, read = Path.exists, Path.read_bytes
    before = {p: p.read_bytes() for p in outputs}
    def only_existing(path):
        assert path not in hidden, "validation tried to read an unpublished output"
        return read(path)
    with monkeypatch.context() as partial:
        partial.setattr(Path, "exists", lambda path: False if path in hidden else exists(path))
        partial.setattr(Path, "read_bytes", only_existing)
        assert validate_artifact(experiment.path) == experiment.digest
    assert {p: p.read_bytes() for p in outputs} == before


def test_cli_rejects_aggregate_with_missing_parent_report(completed_experiment, monkeypatch):
    from werewolf.cli import validate_artifact
    experiment = completed_experiment
    missing = experiment.runs_path / "implicit" / "0" / "primary_development_oof" / "fold_report.json"
    exists = Path.exists
    monkeypatch.setattr(Path, "exists", lambda path: False if path == missing else exists(path))
    with pytest.raises(ValueError, match="missing its parent reports"):
        validate_artifact(experiment.path)
