"""Synthetic invariants; no formal data or external inference backend."""

from dataclasses import replace
import inspect
import math
from pathlib import Path

import pytest
import torch

from tests.tom.test_experiment import experiment_config
from tests.development_publication.test_development_publication import _publication, _republish
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


@pytest.fixture
def independent_test_case(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from tests.development_publication import test_canonical_split_v2 as synthetic
    from werewolf.artifact_io import canonical_json_bytes
    from werewolf.canonical_collection import construct_belief_observation
    from werewolf.canonical_split import publish_canonical_split
    from werewolf.development_publication import open_verified_collection
    from werewolf.structured_history import STRUCTURED_TOKEN_TYPES
    from werewolf.tom import final_evaluation as evaluation, makto
    from werewolf.tom.dataset import ExperimentCapacity
    from werewolf.tom.qwen3_large import graph_identity

    original_fixture = synthetic._fixture

    def with_empty_and_nonempty_reports(*args, game_id):
        plan, claim, evidence, executor = original_fixture(*args, game_id=game_id)
        observations = []
        for observation in evidence.belief_observations:
            support = observation.suspicion_support
            if observation.observer_id not in ("player1", "player2"):
                if game_id == "game-001" or observation.observer_id == "player3":
                    support = ()
                elif observation.observer_id == "player4":
                    support = ("player1", "player5")
            values = observation.to_record()
            for field in ("schema_version", "observation_digest", "label_observed"):
                values.pop(field)
            values.update(status=observation.status, suspicion_support=support)
            observations.append(construct_belief_observation(**values))
        return plan, claim, replace(evidence, belief_observations=tuple(observations)), executor

    monkeypatch.setattr(synthetic, "_fixture", with_empty_and_nonempty_reports)
    protocol, path, replay = synthetic._collection(tmp_path, target=7, completed=7)
    collection = open_verified_collection(path, replay_executor=replay)
    split = publish_canonical_split(collection, protocol, tmp_path / "split", split_id="split")
    games = sorted(member["game_id"] for member in split.members("independent_test"))
    assert games == ["game-001", "game-005"]

    class MockEpoch35(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0
            self.capacity = ExperimentCapacity(1024)
            self.graph = graph_identity()
            self.temporal = SimpleNamespace(day=range(6), artifact_digest=evaluation.TEST_TEMPORAL_DIGEST)
            logits = torch.zeros(7, 7)
            for observer in range(7):
                logits[observer, observer] = -torch.inf
                logits[observer, (observer + 1) % 7] = 2.
            self.logp = logits.log_softmax(-1)

        def forward(self, **public):
            self.calls += 1
            assert not self.training and torch.is_inference_mode_enabled()
            assert set(public) == {"event_ids", "source_ids", "action_ids", "target_ids", "day_ids", "phase_ids", "attention_mask"}
            events = public["event_ids"][public["attention_mask"]]
            assert (STRUCTURED_TOKEN_TYPES.index("public_speech") + 1) not in events
            assert events[-1].item() == STRUCTURED_TOKEN_TYPES.index("turn_start") + 1
            return self.logp.unsqueeze(0)

    model = MockEpoch35().eval()
    source = {"source_revision": "a" * 40, "implementation_digest": "b" * 64}
    monkeypatch.setattr(evaluation, "TEST_SPLIT_DIGEST", split.digest)
    monkeypatch.setattr(evaluation, "TEST_GAME_COUNT", len(games))
    monkeypatch.setattr(evaluation, "attest_source", lambda: source)
    monkeypatch.setattr(makto, "load_epoch35_model", lambda *_: model)
    profile = tmp_path / "storage.json"
    profile.write_bytes(canonical_json_bytes({"artifact_root": str(tmp_path)}))
    return SimpleNamespace(root=tmp_path, collection=path, split=split, games=games, model=model, source=source,
                           cli=["--storage-profile", str(profile), "independent-test"],
                           inputs=["--split", str(split.artifact.path), "--collection", str(path)])


def test_independent_test_cli_seals_public_predictions_before_gold_and_scores(independent_test_case, monkeypatch, capsys):
    import json

    from werewolf.artifact_io import sha256_bytes
    from werewolf.cli import main
    from werewolf.tom.evaluation import score_prediction_rows, uniform_prediction_rows
    from werewolf.tom.makto import EPOCH35_DIGEST
    from werewolf.tom.protocol import bootstrap_indices
    from werewolf.tom.scoring import game_macro_summary
    from werewolf.tom_llm import common

    case = independent_test_case
    pred_path, report_path = case.root / "predictions", case.root / "analysis"
    original = Path.open
    gold_files, aligned = [], []
    valid = common.valid_distribution
    scoring = False

    def track_distribution(row):
        valid(row)
        aligned.append(row)

    def guarded(path, *args, **kwargs):
        if path.is_relative_to(case.collection):
            _, game_id, *parts = path.relative_to(case.collection).parts
            assert game_id in case.games
            if "/".join(parts) not in {"manifest.json", "public/authoritative_pre_prefixes.jsonl"}:
                assert scoring and len(aligned) == 14 and (pred_path / "manifest.json").is_file()
                assert "/".join(parts) in {"public/belief_observations.jsonl", "private/private_replay_evidence.json"}
                gold_files.append((game_id, path.name))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(common, "valid_distribution", track_distribution)
    monkeypatch.setattr(Path, "open", guarded)
    assert main([*case.cli, "predict", *case.inputs, "--checkpoint", "checkpoint", "--temporal", "temporal",
                 "--destination", str(pred_path)]) == 0
    digest = capsys.readouterr().out.strip()
    assert case.model.calls == 2 and not gold_files
    rows = [json.loads(line) for line in (pred_path / "predictions.jsonl").read_bytes().splitlines()]
    assert [(r["game_id"], r["observer"]) for r in rows] == [(g, i) for g in case.games for i in range(7)]
    for row in rows:
        assert not {"q", "role_assignment", "label_observed", "suspicion_support", "seed", "public_events"} & row.keys()
        assert row["observer_alive"]
        seat = row["observer"]
        assert row["probability"] == case.model.logp[seat].exp().tolist()
        assert len(row["non_self_log_probability"]) == 6 and row["probability"][seat] == 0.

    scoring = True
    assert main([*case.cli, "score", *case.inputs, "--predictions", str(pred_path), "--prediction-digest", digest,
                 "--destination", str(report_path)]) == 0
    assert len(gold_files) == 4 and case.model.calls == 2
    report = json.loads((report_path / "analysis.json").read_bytes())
    population = [json.loads(line) for line in (report_path / "population.jsonl").read_bytes().splitlines()]
    assert report["coverage"] == {
        "game_count": 2, "pre_count": 2, "public_prediction_row_count": 14, "public_alive_row_count": 14,
        "primary_row_count": 10, "primary_game_count": 2, "nu_row_count": 4, "nu_game_count": 1,
    }
    assert report["split_manifest_digest"] == case.split.digest
    assert report["independent_test_identity_digest"] == case.split.manifest["independent_test_identity_digest"]
    assert report["prediction_seal_digest"] == digest and report["checkpoint_digest"] == EPOCH35_DIGEST
    assert report["prediction_source"] == report["scoring_source"] == case.source
    assert report["nu_game_ids"] == ["game-005"]
    assert all(r["observer"] >= 2 for r in population)
    assert {r["observer"] for r in population if r["nu"]} == {3, 4, 5, 6}
    primary = []
    for row in rows:
        seat = row["observer"]
        if seat < 2:
            continue
        if row["game_id"] == "game-001" or seat == 2:
            q = [0. if j == seat else 1/6 for j in range(7)]
        elif seat == 3:
            q = [.5, 0., 0., 0., .5, 0., 0.]
        else:
            q = [0., 0., 1., 0., 0., 0., 0.]
        primary.append({**row, "q": torch.tensor(q, dtype=torch.float32).tolist()})
    seed = 2196221715671680944
    indices = bootstrap_indices(case.games, 10000, seed)
    assert sha256_bytes(indices.tobytes()) == report["bootstrap"]["indices_digest"]
    nu_indices = bootstrap_indices(["game-005"], 10000, seed)
    assert sha256_bytes(nu_indices.tobytes()) == report["bootstrap"]["nu_indices_digest"]
    for method, inputs in (("epoch35", primary), ("uniform_non_self", uniform_prediction_rows(primary))):
        scores, per_game = score_prediction_rows(inputs, case.games)
        for key, metric in (("kl", "kl"), ("tv", "total_variation")):
            assert report["cells"][method][key] == game_macro_summary([per_game[g][metric] for g in case.games], indices, .95)
        hit_rate = sum(s["argmax_in_target_support"] for r, s in zip(inputs, scores, strict=True)
                       if sum(x > 0 for x in r["q"]) < 6) / 4
        assert report["cells"][method]["top1_nu"] == game_macro_summary([hit_rate], nu_indices, .95)


def test_independent_test_cli_rejects_split_before_opening_payload(independent_test_case, monkeypatch):
    from werewolf.artifact_io import ArtifactValidationError
    from werewolf.cli import main

    case = independent_test_case
    candidate = _republish(case.split.artifact.path, case.root / "rejected" / "split",
                           files={"simulated_supervision.json": b'{"synthetic":true}'})
    manifest_path = candidate.path / "manifest.json"
    destination = case.root / "predictions"
    original = Path.open
    opened = []

    def guarded(path, *args, **kwargs):
        if path.is_relative_to(candidate.path):
            opened.append(path)
            assert path == manifest_path, f"Independent Test predict opened a Split payload: {path.name}"
        assert not path.is_relative_to(case.collection), "Independent Test predict opened the Collection"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    with pytest.raises(ArtifactValidationError, match="version/inventory mismatch"):
        main([*case.cli, "predict", "--split", str(candidate.path), "--collection", str(case.collection),
              "--checkpoint", "checkpoint", "--temporal", "temporal", "--destination", str(destination)])
    assert opened == [manifest_path]
    assert case.model.calls == 0 and not destination.exists()


@pytest.mark.parametrize("fault", ["seal", "missing", "duplicate", "plan", "self_probability"])
def test_independent_test_score_rejects_invalid_predictions_before_gold(independent_test_case, monkeypatch, fault):
    import json

    from werewolf.artifact_io import canonical_jsonl_bytes, publish_artifact
    from werewolf.tom import final_evaluation as evaluation

    case = independent_test_case
    split = case.split.artifact.path
    pred = evaluation.predict_independent_test(split, case.collection, "checkpoint", "temporal", case.root / "predictions")
    expected_digest = pred.manifest_digest
    if fault == "seal":
        expected_digest = "0" * 64
    else:
        rows = [json.loads(line) for line in (pred.path / "predictions.jsonl").read_bytes().splitlines()]
        if fault == "missing":
            rows.pop()
        elif fault == "duplicate":
            rows[1] = rows[0]
        elif fault == "plan":
            rows[0]["plan_digest"] = "0" * 64
        else:
            rows[0]["probability"][0] = .25
        fields = {k: v for k, v in pred.manifest.items() if k not in {"file_table", "manifest_digest"}}
        pred = publish_artifact(case.root / "misaligned", manifest_fields=fields,
                                files={"predictions.jsonl": canonical_jsonl_bytes(rows)})
        expected_digest = pred.manifest_digest
    original = Path.open

    def guarded(path, *args, **kwargs):
        if path.name in {"belief_observations.jsonl", "private_replay_evidence.json"}:
            pytest.fail("invalid predictions opened Independent Test Gold")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    destination = case.root / "analysis"
    with pytest.raises(ValueError, match="Seal mismatch|misaligned|non-self simplex"):
        evaluation.score_independent_test(split, case.collection, pred.path, expected_digest, destination)
    assert not destination.exists() and case.model.calls == 2


@pytest.mark.parametrize("fault", ["missing_bundle", "capacity", "temporal"])
def test_independent_test_fails_before_forward_without_full_usable_pres(independent_test_case, fault):
    from werewolf.tom import final_evaluation as evaluation
    from werewolf.tom.dataset import ExperimentCapacity

    case = independent_test_case
    if fault == "missing_bundle":
        (case.collection / "games" / case.games[-1] / "manifest.json").unlink()
    elif fault == "capacity":
        case.model.capacity = ExperimentCapacity(3)
    else:
        case.model.temporal.day = range(1)
    destination = case.root / "predictions"
    with pytest.raises(ValueError, match="manifest is missing|exceeds model capacity|exceeds frozen Temporal"):
        evaluation.predict_independent_test(case.split.artifact.path, case.collection,
                                           "checkpoint", "temporal", destination)
    assert case.model.calls == 0 and not destination.exists()
