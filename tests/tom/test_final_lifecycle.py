"""Synthetic invariants; no formal data or external inference backend."""

from dataclasses import replace
import inspect
import math
from pathlib import Path

import pytest
import torch

from tests.tom.test_experiment import experiment_config
from tests.development_publication.test_development_publication import _publication
from tests.canonical_collection.test_final_capacity import sparse_pre


def prepare(tmp_path, monkeypatch):
    import werewolf.tom.final_experiment as module
    monkeypatch.setattr(module, "actual_clean_revision", lambda: "a" * 40)
    _, handles = _publication(tmp_path / "development")
    config = replace(experiment_config(4), game_batch_size=5)
    experiment = module.prepare_final_experiment(handles.public, config, tmp_path / "final" / "experiments" / "experiment")
    return experiment, handles


def final_collection(root, count=2, transform_evidence=None):
    from tests.canonical_collection.test_game_bundle import _plan, _fixture
    from werewolf.canonical_collection import (initialize_attempt_ledger, construct_attempt_claim,
        publish_ledger_record, publish_canonical_game_bundle, construct_attempt_terminal, TerminalOutcome)
    from werewolf.development_publication import open_verified_collection
    plan = _plan(collection_id="independent-final", ordered_seed_pool=tuple(range(1000, 1000+count)), target_canonical_success_count=count)
    ledger = initialize_attempt_ledger(root, plan)
    for ordinal in range(count):
        claim = construct_attempt_claim(plan, ordinal=ordinal, attempt_id=f"final-{ordinal}", claim_timestamp_utc=f"2026-09-04T01:00:{ordinal:02d}Z")
        publish_ledger_record(ledger, plan, claim)
        game_id = f"final-game-{ordinal}"
        _, _, evidence, executor = _fixture(plan, claim, game_id=game_id)
        if transform_evidence is not None:
            evidence = transform_evidence(evidence)
        bundle = publish_canonical_game_bundle(root / "games" / game_id, plan=plan, claim=claim, evidence=evidence, replay_executor=executor)
        publish_ledger_record(ledger, plan, construct_attempt_terminal(plan, claim, outcome=TerminalOutcome.CANONICAL_SUCCESS,
            canonical_game_bundle_id=game_id, canonical_game_bundle_digest=bundle.manifest_digest,
            terminal_timestamp_utc=f"2026-09-04T01:01:{ordinal:02d}Z"))
    return open_verified_collection(root, replay_executor=executor)


def test_final_preparation_has_no_partition_and_paired_controls(tmp_path, monkeypatch):
    from werewolf.tom.final_experiment import prepare_final_experiment, open_final_experiment
    from werewolf.tom.temporal import TemporalCodeProvider
    from werewolf.tom.state import load_model_state
    from werewolf.tom.training import build_model
    experiment, handles = prepare(tmp_path, monkeypatch)
    m = experiment.manifest
    assert m["game_ids"] == sorted(handles.public.public_view.game_ids)
    assert not {"folds", "held_out_game_ids", "validation"} & set(m)
    assert list(inspect.signature(prepare_final_experiment).parameters) == ["publication", "config", "destination"]
    assert m["optimizer_steps"] == 7 * experiment.config.rotation_cycles * math.ceil(5 / experiment.config.game_batch_size)
    schedule = experiment.schedule()
    for game in m["game_ids"]:
        assert sorted(r["shift"] for b in schedule["batches"] for r in b if r["game_id"] == game) == list(range(7))
    provider_a = TemporalCodeProvider.implicit(experiment.path / "temporal", m["protocol_digest"])
    provider_b = TemporalCodeProvider.explicit(experiment.path / "temporal", m["protocol_digest"])
    assert provider_a.artifact_digest == provider_b.artifact_digest == m["temporal_artifact_digest"]
    assert provider_a.day.tobytes() == provider_b.day.tobytes()
    assert len(provider_a.day) == experiment.config.max_seq_len // 4 + 1
    assert not provider_a.code(torch.tensor([1]), torch.tensor([1])).any()
    states = []
    for c in m["temporal_conditions"]:
        model = build_model(experiment, c)
        load_model_state(experiment.path / "initial_state", model, m["paired_initial_state_digest"])
        states.append(model.state_dict())
    assert all(torch.equal(states[0][k], states[1][k]) for k in states[0])
    assert open_final_experiment(experiment.path).digest == experiment.digest


def test_final_content_cannot_control_preparation(tmp_path, monkeypatch):
    from werewolf.tom.final_experiment import prepare_final_experiment
    from werewolf.tom.protocol import final_training_schedule
    experiment, handles = prepare(tmp_path, monkeypatch)
    # Arbitrary final bytes, including identities/statistics/private labels, have no input edge.
    final_path = tmp_path / "unknown-final.json"
    final_path.write_text('{"labels": [1], "roles": [2], "day": 9999}')
    original_read = Path.read_bytes
    def forbid_final(path):
        if path == final_path:
            raise AssertionError("final content opened during preparation")
        return original_read(path)
    monkeypatch.setattr(Path, "read_bytes", forbid_final)
    second = prepare_final_experiment(handles.public, experiment.config, tmp_path / "second" / "experiments" / "experiment")
    assert second.digest == experiment.digest
    assert second.schedule() == experiment.schedule()
    assert (second.path / "initial_state/tensors.bin").read_bytes() == (experiment.path / "initial_state/tensors.bin").read_bytes()
    ids = experiment.manifest["game_ids"]
    assert final_training_schedule(303, ids, 1, 5) != final_training_schedule(303, [*ids[:-1], "different-training-game"], 1, 5)


def test_public_tensorization_is_label_free_and_matches_dataset(tmp_path, monkeypatch):
    from werewolf.tom.dataset import tensorize_public_pre, CanonicalToMDataset, ExperimentCapacity
    _, handles = prepare(tmp_path, monkeypatch)
    view = handles.public.public_view
    dataset = CanonicalToMDataset(view, view.game_ids, ExperimentCapacity(4))
    for sample in dataset:
        prefix = view.load_game(sample.game_id).authoritative_pre_prefixes[0]
        _, public = tensorize_public_pre(prefix, ExperimentCapacity(4))
        assert all(torch.equal(value, sample.public.kwargs()[key]) for key, value in public.kwargs().items())
    # Links identify unavailable observations; no supervision records are supplied.
    plan, public = tensorize_public_pre(sparse_pre(1), ExperimentCapacity(4))
    assert plan.token_count == 4 and public.attention_mask.all()


def test_unsealed_evaluation_does_not_open_final_path(tmp_path, monkeypatch):
    from werewolf.tom.final_evaluation import run_final_evaluation, SealedFinalPredictor
    experiment, _ = prepare(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="seal required"):
        run_final_evaluation(experiment, tmp_path / "does-not-exist")
    with pytest.raises(ValueError, match="seal required"):
        SealedFinalPredictor(experiment, "implicit")


def test_paired_final_fit_seal_inference_and_independent_report(tmp_path, monkeypatch):
    from werewolf.tom.final_training import _final_worker, seal_final_models, verify_final_seal, train_final_condition
    from werewolf.tom.final_evaluation import (publish_final_evaluation, run_final_evaluation,
        validate_final_evaluation, SealedFinalPredictor)
    from werewolf.tom.evaluation import score_prediction_rows
    from werewolf.tom.experiment import read_json
    experiment, _ = prepare(tmp_path, monkeypatch)
    initial = (experiment.path / "initial_state/tensors.bin").read_bytes()
    terminals = [_final_worker(experiment.path, experiment.digest, c, False) for c in experiment.manifest["temporal_conditions"]]
    assert len({t.manifest["provenance"]["initial_rng_digest"] for t in terminals}) == 1
    assert all(t.manifest["provenance"]["terminal_step"] == 7 for t in terminals)
    seal = seal_final_models(experiment)
    assert verify_final_seal(experiment) == seal
    with pytest.raises(ValueError, match="permanently closes"):
        train_final_condition(experiment, "implicit", resume=True)
    predictor = SealedFinalPredictor(experiment, "implicit")
    original_forward = predictor.model.forward
    def public_only(**kwargs):
        assert set(kwargs) == {"event_ids", "source_ids", "action_ids", "target_ids", "day_ids", "phase_ids", "attention_mask"}
        return original_forward(**kwargs)
    monkeypatch.setattr(predictor.model, "forward", public_only)
    p = predictor.predict(sparse_pre(1), 0)
    assert p.shape == (7,) and p[0] == 0 and torch.isclose(p.sum(), torch.tensor(1.))
    collection = final_collection(tmp_path / "independent")
    publication = publish_final_evaluation(experiment, collection, tmp_path / "final-publication", publication_id="final-publication")
    from tests.tom.test_content_independence import _changed_evidence
    from werewolf.tom.final_experiment import prepare_final_experiment
    from werewolf.development_publication import open_publication
    terminal_bytes = [(t.path / "tensors.bin").read_bytes() for t in terminals]
    for change in ("belief", "role", "private"):
        alternate_collection = final_collection(tmp_path / f"independent-{change}",
            transform_evidence=lambda e: _changed_evidence(e, change))
        alternate = publish_final_evaluation(experiment, alternate_collection,
            tmp_path / f"final-{change}", publication_id=f"final-{change}")
        assert alternate.manifest_digest != publication.manifest_digest
        again = prepare_final_experiment(open_publication(experiment.manifest["publication_path"]),
            experiment.config, tmp_path / f"reprepare-{change}/experiments/experiment")
        assert again.digest == experiment.digest
        assert again.schedule() == experiment.schedule()
        assert (again.path / "initial_state/tensors.bin").read_bytes() == initial
        assert [(t.path / "tensors.bin").read_bytes() for t in terminals] == terminal_bytes
    report = run_final_evaluation(experiment, publication.path)
    assert report == validate_final_evaluation(experiment)
    assert report["game_ids"] == sorted(publication.public_view.game_ids)
    assert set(report["cells"]) == {f"{c}/{p}" for c in experiment.manifest["temporal_conditions"] for p in ("primary", "all_alive_identifiability_stress")}
    rows = [read_json(line) for line in (experiment.runs_path / "final_evaluation/implicit_predictions.jsonl").read_bytes().splitlines()]
    selected = [r for r in rows if r["observer_alive"] and r["label_observed"]]
    _, games = score_prediction_rows(selected, report["game_ids"])
    assert games == report["cells"]["implicit/all_alive_identifiability_stress"]["game_scores"]
    assert (experiment.path / "initial_state/tensors.bin").read_bytes() == initial
    with pytest.raises(ValueError, match="already consumed"):
        run_final_evaluation(experiment, publication.path)
    assert run_final_evaluation(experiment, publication.path, resume=True) == report
    # Any final payload mutation remains fail-closed; frozen model artifacts stay untouched.
    game_file = publication.path / f"public/games/{report['game_ids'][0]}.json"
    game_file.write_bytes(game_file.read_bytes() + b" ")
    with pytest.raises(ValueError):
        validate_final_evaluation(experiment)
    assert (experiment.path / "initial_state/tensors.bin").read_bytes() == initial


def test_interruption_recovery_matches_uninterrupted_terminal(tmp_path, monkeypatch):
    import werewolf.tom.final_training as training
    from werewolf.tom.final_experiment import prepare_final_experiment
    experiment, handles = prepare(tmp_path, monkeypatch)
    config = replace(experiment.config, recovery_cadence=3)
    a = prepare_final_experiment(handles.public, config, tmp_path / "a/experiments/experiment")
    b = prepare_final_experiment(handles.public, config, tmp_path / "b/experiments/experiment")
    terminal_a = training._final_worker(a.path, a.digest, "implicit", False)
    publish = training._publish_point
    class Interrupted(BaseException):
        pass
    def interrupt(*args, **kwargs):
        result = publish(*args, **kwargs)
        raise Interrupted()
    monkeypatch.setattr(training, "_publish_point", interrupt)
    with pytest.raises(Interrupted):
        training._final_worker(b.path, b.digest, "implicit", False)
    monkeypatch.setattr(training, "_publish_point", publish)
    with pytest.raises(ValueError, match="explicit resume"):
        training._final_worker(b.path, b.digest, "implicit", False)
    terminal_b = training._final_worker(b.path, b.digest, "implicit", True)
    assert terminal_a.manifest_digest == terminal_b.manifest_digest
    assert (terminal_a.path / "tensors.bin").read_bytes() == (terminal_b.path / "tensors.bin").read_bytes()


def test_temporal_table_uses_full_derived_capacity(tmp_path, monkeypatch):
    from werewolf.tom.final_experiment import prepare_final_experiment
    from werewolf.tom.temporal import TemporalCodeProvider
    experiment, handles = prepare(tmp_path, monkeypatch)
    larger = prepare_final_experiment(handles.public, replace(experiment.config, max_seq_len=1024), tmp_path / "larger/experiments/experiment")
    m = larger.manifest
    assert m["protocol_inputs"]["capacity"]["derived_day_capacity"] == 256
    a = TemporalCodeProvider.implicit(larger.path / "temporal", m["protocol_digest"])
    b = TemporalCodeProvider.explicit(larger.path / "temporal", m["protocol_digest"])
    assert a.day.shape == b.day.shape == (257, 128)
    assert a.artifact_digest == b.artifact_digest
    assert torch.isfinite(b.code(torch.tensor([256]), torch.tensor([1]))).all()
    for provider in (a, b):
        with pytest.raises(ValueError, match="out-of-range"):
            provider.code(torch.tensor([257]), torch.tensor([1]))


def test_final_cli_rejects_manual_day_and_training_overrides():
    from werewolf.cli import build_parser
    parser = build_parser()
    base = ["prepare-final-experiment", "--publication", "p", "--protocol", "c", "--destination", "e"]
    for flags in (["--max-day", "256"], ["--learning-rate", ".1"], ["--fold", "0"]):
        with pytest.raises(SystemExit):
            parser.parse_args(base + flags)


def test_final_preflight_capacity_failure_precedes_all_forward(tmp_path, monkeypatch):
    import werewolf.tom.final_evaluation as evaluation
    from werewolf.tom.final_training import _final_worker, seal_final_models
    experiment, _ = prepare(tmp_path, monkeypatch)
    for condition in experiment.manifest["temporal_conditions"]:
        _final_worker(experiment.path, experiment.digest, condition, False)
    seal_final_models(experiment)
    publication = evaluation.publish_final_evaluation(experiment, final_collection(tmp_path / "independent"),
        tmp_path / "final-publication", publication_id="final-publication")
    last = publication.public_view.game_ids[-1]
    original = evaluation.validate_final_pre
    def capacity_failure(prefix, capacity):
        if prefix.game_id == last:
            # Exercise the real overflow validator rather than accepting a truncated prefix.
            return original(sparse_pre(2), capacity)
        return original(prefix, capacity)
    def forbidden(*args, **kwargs):
        raise AssertionError("inference before full-set capacity validation")
    monkeypatch.setattr(evaluation, "validate_final_pre", capacity_failure)
    monkeypatch.setattr(evaluation, "SealedFinalPredictor", forbidden)
    with pytest.raises(ValueError, match="sequence capacity"):
        evaluation.run_final_evaluation(experiment, publication.path)
    assert (experiment.runs_path / "final_evaluation/failure.json").exists()
    assert not (experiment.runs_path / "final_evaluation/report.json").exists()
    with pytest.raises(ValueError, match="failed closed"):
        evaluation.run_final_evaluation(experiment, publication.path, resume=True)


def test_actual_source_identity_requires_clean_head(monkeypatch):
    import werewolf.tom.final_experiment as module
    def dirty(command, **kwargs):
        assert command[3] == "status"
        return " M werewolf/tom/model.py\n"
    monkeypatch.setattr(module.subprocess, "check_output", dirty)
    with pytest.raises(ValueError, match="clean implementation"):
        module.actual_clean_revision()
