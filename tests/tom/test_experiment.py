from dataclasses import replace

import pytest

from tests.development_publication.test_development_publication import _publication


def experiment_config(capacity):
    from werewolf.tom.experiment import ExperimentConfig
    return ExperimentConfig(max_seq_len=capacity, learning_rate=.001, weight_decay=0.,
        adam_betas=(.9, .999), adam_eps=1e-8, game_batch_size=4, rotation_cycles=1,
        initialization_seed=101, rng_seed=202, schedule_seed=303, bootstrap_seed=404,
        bootstrap_replicates=20, confidence_level=.95, recovery_cadence=7,
        device="cpu", torch_num_threads=1, source_revision="deterministic-test-source-v1",
        optimizer="adamw", scheduler="constant", deterministic_algorithms=True)


def prepared_experiment(tmp_path):
    from werewolf.tom.experiment import prepare_experiment
    _, handles = _publication(tmp_path)
    config = experiment_config(handles.public.public_view.max_structured_token_count)
    return prepare_experiment(handles.public, config, tmp_path / "experiments" / "experiment"), handles


def test_preparation_binds_five_folds_two_lineages_and_fixed_protocol(tmp_path):
    from werewolf.tom.experiment import prepare_experiment, open_experiment
    _, handles = _publication(tmp_path)
    config = experiment_config(handles.public.public_view.max_structured_token_count)
    with pytest.raises(ValueError, match="capacity"):
        prepare_experiment(handles.public, replace(config, max_seq_len=1), tmp_path / "bad")
    experiment = prepare_experiment(handles.public, config, tmp_path / "experiment")
    reopened = open_experiment(experiment.path)
    assert experiment.digest == reopened.digest
    assert experiment.manifest["temporal_conditions"] == ["implicit", "explicit_day_phase"]
    assert len(experiment.manifest["folds"]) == 5
    for fold in experiment.manifest["folds"]:
        assert fold["optimizer_steps"] == 7
        assert len(fold["training_game_ids"]) == 4
        assert not set(fold["training_game_ids"]) & set(fold["held_out_game_ids"])
        assert "paired_initial_state_digest" in fold
    assert experiment.manifest["recovery_policy"] == "deterministic_step_boundary_resume_v1"
    assert not any("training_all_alive" in p for p in experiment.manifest["file_table"])
