"""Crash fault injection lives in fresh test workers, never production switches."""

import multiprocessing
import os
import shutil
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.development_publication.test_development_publication import _publication
from tests.tom.test_experiment import experiment_config


def _crash_at(path, digest, moment):
    from werewolf.tom import training
    from werewolf.artifact_io import canonical
    original = training._publish_recovery
    publish = canonical._publish_directory_noreplace
    fsync = canonical._write_fsynced

    def recovery(target, *args):
        if target.name == "2" and moment == "after_optimizer":
            os._exit(73)
        result = original(target, *args)
        if target.name == "2" and moment == "after_durable":
            os._exit(73)
        return result

    def publish_hook(source, target):
        if target.parent.name == "recovery" and target.name == "2" and moment == "before_publish":
            os._exit(73)
        publish(source, target)
        if target.parent.name == "recovery" and target.name == "2" and moment == "after_publish":
            os._exit(73)

    def file_hook(target, data):
        fsync(target, data)
        if target.parent.name.startswith(".2.staging-") and moment == "partial_stage":
            os._exit(73)

    with patch.object(training, "_publish_recovery", recovery), patch.object(canonical, "_publish_directory_noreplace", publish_hook), patch.object(canonical, "_write_fsynced", file_hook):
        training._train_worker(path, digest, 0, "implicit")


def _prepared(tmp_path):
    from werewolf.tom.experiment import prepare_experiment
    _, handles = _publication(tmp_path)
    config = replace(experiment_config(handles.public.public_view.max_structured_token_count), recovery_cadence=1)
    return prepare_experiment(handles.public, config, tmp_path / "baseline" / "experiments" / "experiment")


@pytest.mark.parametrize("moment", ["after_optimizer", "partial_stage", "before_publish", "after_publish", "after_durable"])
def test_process_crash_resumes_identical_optimizer_rng_cursor_and_model(tmp_path, moment):
    from werewolf.tom.experiment import open_experiment
    from werewolf.tom.training import train_primary_fold
    baseline = _prepared(tmp_path)
    clone_path = tmp_path / "interrupted" / "experiments" / "experiment"
    shutil.copytree(baseline.path, clone_path)
    interrupted = open_experiment(clone_path)
    expected = train_primary_fold(baseline, 0, "implicit")
    process = multiprocessing.get_context("spawn").Process(target=_crash_at, args=(str(clone_path), interrupted.digest, moment))
    process.start()
    process.join(45)
    if process.is_alive():
        process.kill()
        process.join()
        pytest.fail("crash fixture failed to reach the declared fault point")
    assert process.exitcode == 73
    actual = train_primary_fold(interrupted, 0, "implicit")
    assert actual.manifest_digest == expected.manifest_digest
    for file in (baseline.runs_path / "implicit" / "0").rglob("*"):
        if file.is_file():
            assert file.read_bytes() == (interrupted.runs_path / "implicit" / "0" / file.relative_to(baseline.runs_path / "implicit" / "0")).read_bytes()


def test_corrupt_recovery_cannot_select_earlier_point_or_rerun(tmp_path):
    from werewolf.tom.training import train_primary_fold
    experiment = _prepared(tmp_path)
    process = multiprocessing.get_context("spawn").Process(target=_crash_at, args=(str(experiment.path), experiment.digest, "after_durable"))
    process.start()
    process.join(45)
    assert process.exitcode == 73
    payload = experiment.runs_path / "implicit" / "0" / "recovery" / "2" / "rng_state.bin"
    payload.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="mismatch"):
        train_primary_fold(experiment, 0, "implicit")
    with pytest.raises(ValueError, match="permanently failed"):
        train_primary_fold(experiment, 0, "implicit")
