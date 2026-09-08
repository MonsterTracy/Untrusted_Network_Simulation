"""Deployment routing tests; no collection, training or formal artifact creation."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from werewolf.cli import _artifact_path, _experiment_path, _storage_root, main


@pytest.fixture
def storage(tmp_path, monkeypatch):
    root = (tmp_path / "artifacts").resolve()
    root.mkdir()
    profile = tmp_path / "storage.json"
    profile.write_text(json.dumps({"artifact_root": str(root)}))
    monkeypatch.setenv("UNS_STORAGE_PROFILE", str(profile))
    return root, profile


def test_exact_identity_layout_retains_experiment_owned_runs(storage):
    from werewolf.tom.experiment import VerifiedExperiment
    root, profile = storage
    assert _storage_root(profile) == root
    assert _artifact_path(root, "collection-a", "canonical") == root / "canonical/collection-a"
    assert _artifact_path(root, "publication-b", "publications") == root / "publications/publication-b"
    path = _artifact_path(root, "experiment-c", "experiments")
    experiment = VerifiedExperiment(SimpleNamespace(path=path, manifest_digest="d" * 64), None)
    assert path == root / "experiments/experiment-c/experiments/experiment"
    assert experiment.runs_path == root / "experiments/experiment-c/runs" / ("d" * 64)
    assert _artifact_path(root, path.relative_to(root)) == path
    assert _artifact_path(root, path) == path
    assert _experiment_path(root, "experiment-c") == path
    assert _experiment_path(root, path) == path
    with pytest.raises(ValueError, match="layout"):
        _experiment_path(root, "experiments/loose")


@pytest.mark.parametrize("value", [None, {}, {"artifact_root": "relative"},
    {"artifact_root": "/nonexistent/uns-storage"}, {"artifact_root": 4},
    {"artifact_root": "/", "learning_rate": 1}])
def test_profile_rejects_missing_invalid_or_scientific_fields(tmp_path, value):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(value))
    with pytest.raises((ValueError, TypeError)):
        _storage_root(path)
    with pytest.raises(ValueError, match="required"):
        _storage_root(None)


def test_paths_reject_escape_symlinks_and_missing_profile(storage, tmp_path, monkeypatch):
    root, _ = storage
    for value in ("../outside", root.parent / "outside", ".", root):
        with pytest.raises(ValueError):
            _artifact_path(root, value)
    (root / "link").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic"):
        _artifact_path(root, "link/new-artifact")
    with pytest.raises(ValueError, match="symbolic"):
        _storage_root_for_link(root, tmp_path)
    monkeypatch.delenv("UNS_STORAGE_PROFILE")
    with pytest.raises(ValueError, match="required"):
        main(["validate-artifact", "publications/a"])


def _storage_root_for_link(root, tmp_path):
    profile = tmp_path / "link-profile.json"
    profile.write_text(json.dumps({"artifact_root": str(root / "link")}))
    return _storage_root(profile)


def test_pending_formal_protocol_stops_before_publication_access(storage, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("publication must not be opened")
    monkeypatch.setattr("werewolf.development_publication.open_publication", forbidden)
    protocol = Path(__file__).resolve().parents[2] / "configs/formal/development-experiment-v1/protocol.json"
    with pytest.raises(ValueError, match="max_seq_len"):
        main(["prepare-experiment", "--publication", "p", "--protocol", str(protocol), "--destination", "e"])
    assert not (storage[0] / "experiments").exists()


def test_oof_requires_explicit_resume_and_propagates_failure(storage, monkeypatch):
    # Load production imports before replacing their entry points, so other
    # tests cannot retain the fake through a module-level imported binding.
    import werewolf.tom.reporting
    root, _ = storage
    path = _artifact_path(root, "e", "experiments")
    runs = root / "experiments/e/runs/digest"
    experiment = SimpleNamespace(path=path, runs_path=runs,
        manifest={"publication_path": str(root / "publications/p")})
    opened, called = [], []
    def open_exact(value):
        opened.append(value)
        return experiment
    def operation(value):
        called.append(value)
        return {"record_digest": "result"}
    monkeypatch.setattr("werewolf.tom.experiment.open_experiment", open_exact)
    monkeypatch.setattr("werewolf.tom.reporting.run_development_oof", operation)
    args = ["run-development-oof", "--experiment", "e"]
    with pytest.raises(ValueError, match="existing run records"):
        main(args + ["--resume"])
    assert not called
    assert main(args) == 0
    runs.mkdir(parents=True)
    (runs / "training-record").write_text("started")
    with pytest.raises(ValueError, match="explicit --resume"):
        main(args)
    assert len(called) == 1
    assert main(args + ["--resume"]) == 0
    assert opened == [path] * 4
    def fail(value):
        raise ValueError("lineage failed closed")
    monkeypatch.setattr("werewolf.tom.reporting.run_development_oof", fail)
    with pytest.raises(ValueError, match="lineage failed closed"):
        main(args + ["--resume"])
    experiment.manifest["publication_path"] = str(root.parent / "outside")
    with pytest.raises(ValueError, match="inside artifact root"):
        main(args + ["--resume"])


def test_collection_resume_gate_precedes_backend_access(storage):
    root, _ = storage
    destination = root / "canonical/c"
    args = ["collect", "--plan", "missing", "--runtime-config", "missing",
        "--call-limit", "1", "--destination", "c"]
    with pytest.raises(ValueError, match="existing run records"):
        main(args + ["--resume"])
    destination.mkdir(parents=True)
    (destination / "collection_plan.json").write_text("existing")
    with pytest.raises(ValueError, match="explicit --resume"):
        main(args)


def test_single_console_owner():
    from unittest.mock import patch
    import runpy
    root = Path(__file__).resolve().parents[2]
    with patch("setuptools.setup") as setup:
        runpy.run_path(str(root / "setup.py"), run_name="__test__")
    assert setup.call_args.kwargs["entry_points"]["console_scripts"] == ["uns=werewolf.cli:main"]


@pytest.mark.parametrize("game_count", [5, 17, 61])
def test_schedule_budget_remains_derived_for_variable_sizes(game_count):
    from collections import Counter
    from math import ceil
    from werewolf.tom.protocol import training_schedule
    games = [f"game-{i}" for i in range(game_count)]
    schedule = training_schedule(0, 0, games, 2, 3)
    assert schedule["optimizer_steps"] == 7 * 2 * ceil(game_count / 3)
    counts = Counter((r["game_id"], r["shift"]) for batch in schedule["batches"] for r in batch)
    assert counts == Counter({(game, shift): 2 for game in games for shift in range(7)})
