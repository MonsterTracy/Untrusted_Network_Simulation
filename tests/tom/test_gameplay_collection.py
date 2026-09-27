"""Focused contract tests for the frozen paired gameplay collection."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import tom_gameplay_collection as collection
from tests.canonical_collection.test_game_bundle import _fixture
from werewolf.artifact_io import ArtifactConflictError
from werewolf.canonical_collection.attempt_ledger import (
    CollectionSeedPoolExhausted,
    TerminalOutcome,
    construct_attempt_claim,
    initialize_attempt_ledger,
    publish_ledger_record,
    validate_attempt_ledger,
)
from werewolf.canonical_collection.trajectory_evidence import (
    construct_canonical_game_evidence,
    construct_private_replay_evidence,
)


def _record():
    return collection._experiment_record(
        "a" * 40, {"served_model_name": "fixture-model",
                   "runtime_config_sha256": "b" * 64}, 32
    )


def _freeze(root, record):
    return collection._publish_record(
        root / "experiment_plan", "tom_gameplay_experiment_plan", record
    )


def _successful(arm):
    return {"arm": arm, "status": "success",
            "roles": {"player1": "Werewolf"}}


def _failed(arm, stage="runtime", category="BackendFailure"):
    return {"arm": arm, "status": "failure", "roles": None,
            "failure_stage": stage, "failure_type": category}


def test_fixed_identity_pool_digest_and_even_odd_order():
    record = _record()
    assert record["experiment_id"] == "paper-tom-gameplay-ablation-v1"
    assert record["target_valid_pairs"] == 40
    assert len(record["ordered_seed_pool"]) == 100
    assert record["ordered_seed_pool"] == list(
        collection.operator.derive_seed_pool(collection.EXPERIMENT_ID, 100)
    )
    assert record["candidate_seed_pool_digest"] == collection._digest(
        record["ordered_seed_pool"]
    )
    assert collection._order(0) == (collection.NOTOM, collection.TOM)
    assert collection._order(1) == (collection.TOM, collection.NOTOM)
    assert collection._order(2) == (collection.NOTOM, collection.TOM)


def test_pair_inclusion_and_fail_fast_contract():
    record = _record()
    order = collection._order(0)
    valid = collection._pair_record(
        record, 0, order,
        {collection.NOTOM: _successful(collection.NOTOM),
         collection.TOM: _successful(collection.TOM)},
    )
    assert valid["status"] == "completed_valid"
    assert valid["roles_equal"] is True
    first_failure = collection._pair_record(
        record, 0, order,
        {collection.NOTOM: _failed(collection.NOTOM),
         collection.TOM: None},
    )
    assert first_failure["status"] == "invalid"
    assert first_failure["arms"][collection.TOM]["status"] == "not_run"
    second_failure = collection._pair_record(
        record, 0, order,
        {collection.NOTOM: _successful(collection.NOTOM),
         collection.TOM: _failed(collection.TOM)},
    )
    assert second_failure["failure_arm"] == collection.TOM
    with pytest.raises(ValueError, match="fail-fast"):
        collection._pair_record(
            record, 0, order,
            {collection.NOTOM: _failed(collection.NOTOM),
             collection.TOM: _successful(collection.TOM)},
        )
    with pytest.raises(ValueError, match="terminal second"):
        collection._pair_record(
            record, 0, order,
            {collection.NOTOM: _successful(collection.NOTOM),
             collection.TOM: None},
        )
    mismatched = collection._pair_record(
        record, 0, order,
        {collection.NOTOM: _successful(collection.NOTOM),
         collection.TOM: {"arm": collection.TOM, "status": "success",
                          "roles": {"player1": "Villager"}}},
    )
    assert mismatched["status"] == "invalid"
    assert mismatched["failure_type"] == "initial_roles_mismatch"


def test_candidate_progression_quota_and_final_manifest(tmp_path, monkeypatch):
    root = tmp_path / "ablation"
    record = _record()
    _freeze(root, record)
    calls = []

    def execute_arm(**kwargs):
        ordinal, arm = kwargs["ordinal"], kwargs["arm"]
        calls.append((ordinal, arm))
        if ordinal == 0:
            return _failed(arm)
        return _successful(arm)

    monkeypatch.setattr(collection, "_execute_arm", execute_arm)
    digest = collection.collect_pairs(
        root=root, record=record, base_factory=None, replay_executor=None,
        predictor_checkout=Path("/unused"), fit_path=Path("/unused"),
    )
    final, artifact = collection._read_record(root / "final", "tom_gameplay_final")
    assert artifact.manifest_digest == digest
    assert final["attempted_candidate_count"] == 41
    assert final["completed_valid_count"] == 40
    assert [item["candidate_ordinal"] for item in final["included_pairs"]] == list(
        range(1, 41)
    )
    assert final["invalid_pairs"][0]["candidate_ordinal"] == 0
    assert calls[0] == (0, collection.NOTOM)
    assert (0, collection.TOM) not in calls
    assert calls[1:3] == [(1, collection.TOM), (1, collection.NOTOM)]
    assert len(calls) == 81
    assert not collection._pair_path(root, 41).exists()


def test_exhausted_frozen_pool_fails_closed(tmp_path, monkeypatch):
    root = tmp_path / "ablation"
    record = _record()
    _freeze(root, record)
    calls = []

    def fail_first(**kwargs):
        calls.append(kwargs["ordinal"])
        return _failed(kwargs["arm"])

    monkeypatch.setattr(collection, "_execute_arm", fail_first)
    with pytest.raises(CollectionSeedPoolExhausted, match="100 frozen"):
        collection.collect_pairs(
            root=root, record=record, base_factory=None, replay_executor=None,
            predictor_checkout=Path("/unused"), fit_path=Path("/unused"),
        )
    assert calls == list(range(100))
    assert not (root / "final").exists()


def test_containment_and_zero_treatment_are_protocol_valid():
    prefix = SimpleNamespace(
        boundary_id="pre-1", prefix_digest="a" * 64,
        current_speaker="player1",
        public_temporal_state=SimpleNamespace(phase=collection.PublicPhase.DISCUSSION),
    )
    action = SimpleNamespace(
        boundary_id="pre-1", actor_id="player1", action_type="public_speech"
    )
    payload = SimpleNamespace(to_value=lambda: {
        "args": [], "kwargs": {"messages": [{"content": collection.TOM_HEADER}]}
    })
    call = SimpleNamespace(
        boundary_id="pre-1", observer_id="player1", private_payload=payload
    )
    private = SimpleNamespace(role_assignment=(("player1", "Werewolf"),))
    evidence = SimpleNamespace(
        game_id="game-1", authoritative_pre_prefixes=(prefix,),
        private_replay_evidence=private, submitted_gameplay_actions=(action,),
        backend_call_evidence=(call,),
    )
    entry = {
        "game_id": "game-1", "boundary_id": "pre-1",
        "prefix_digest": "a" * 64, "observer_ids": ["player3"],
        "payload_digest": "b" * 64,
        "fit_digest": collection.FIT_DIGEST,
        "seal_digest": collection.SEAL_DIGEST,
    }
    collection._validate_treatment(evidence, collection.TOM, [entry], 1)
    with pytest.raises(ValueError, match="NoToM backend"):
        collection._validate_treatment(evidence, collection.NOTOM, [], None)
    with pytest.raises(ValueError, match="request count"):
        collection._validate_treatment(evidence, collection.TOM, [entry], 2)
    with pytest.raises(ValueError, match="unknown treatment boundary"):
        collection._validate_treatment(
            evidence, collection.TOM, [{**entry, "boundary_id": "pre-2"}], 1
        )
    leaked = SimpleNamespace(**{**vars(call), "boundary_id": "unrelated"})
    with pytest.raises(ValueError, match="untreated boundary"):
        collection._validate_treatment(
            SimpleNamespace(**{**vars(evidence), "backend_call_evidence": (leaked,)}),
            collection.TOM, [entry], 1,
        )
    nonwolf = SimpleNamespace(**{**vars(private), "role_assignment": (
        ("player1", "Villager"),
    )})
    with pytest.raises(ValueError, match="wolf public speech"):
        collection._validate_treatment(
            SimpleNamespace(**{**vars(evidence), "private_replay_evidence": nonwolf}),
            collection.TOM, [entry], 1,
        )
    quiet = SimpleNamespace(**{**vars(evidence), "backend_call_evidence": ()})
    collection._validate_treatment(quiet, collection.NOTOM, [], None)
    collection._validate_treatment(quiet, collection.TOM, [], 0)


def _evidence_with_winner(plan, claim):
    _, _, original, replay = _fixture(plan, claim)
    prior = original.private_replay_evidence
    private = construct_private_replay_evidence(
        game_id=original.game_id, seed=claim.seed,
        role_assignment=dict(prior.role_assignment),
        runtime_configuration=prior.runtime_configuration,
        initial_runtime_state=prior.initial_runtime_state,
        replay_inputs={"winner": "Werewolf"},
        expected_public_event_digest=prior.expected_public_event_digest,
    )
    evidence = construct_canonical_game_evidence(
        game_id=original.game_id,
        public_event_stream=original.public_event_stream,
        authoritative_pre_prefixes=original.authoritative_pre_prefixes,
        belief_observations=original.belief_observations,
        speech_annotations_v1=original.speech_annotations_v1,
        call_budget_summary=original.call_budget_summary,
        submitted_gameplay_actions=original.submitted_gameplay_actions,
        backend_call_evidence=original.backend_call_evidence,
        private_replay_evidence=private,
    )
    return evidence, replay


class _FixtureBaseFactory:
    def __init__(self):
        self.calls = []
        self.replay = None

    def __call__(self, *, plan, claim):
        self.calls.append((plan.collection_id, claim.seed))
        evidence, replay = _evidence_with_winner(plan, claim)
        self.replay = replay
        recorder = SimpleNamespace(
            complete_evidence=lambda: evidence,
            failure_from_exception=lambda error, **_kwargs: error,
        )
        return SimpleNamespace(
            env=object(), agents=[], roles=["Werewolf"] * 7,
            recorder=recorder, call_audit=object(), replay_executor=replay,
        )


class _FakePredictor:
    _next_request = 0
    instances = 0

    def __init__(self, **_kwargs):
        type(self).instances += 1

    def __enter__(self):
        return self

    def __exit__(self, *_errors):
        pass


def test_tom_wrapper_uses_real_collector_ledger_and_bundle(
    tmp_path, monkeypatch,
):
    root = tmp_path / "ablation"
    record = _record()
    _freeze(root, record)
    monkeypatch.setattr(collection, "run_ablation", lambda *_args, **_kwargs: "Werewolf win")
    monkeypatch.setattr(collection, "Qwen3GameplayPredictorClient", _FakePredictor)
    _FakePredictor.instances = 0
    base = _FixtureBaseFactory()
    plan = collection._arm_plan(record, 1, collection.TOM)
    claim = construct_attempt_claim(
        plan, ordinal=0, attempt_id="fixture-claim",
        claim_timestamp_utc="2026-09-27T00:00:00Z",
    )
    _, replay = _evidence_with_winner(plan, claim)
    status = collection._execute_arm(
        root=root, record=record, ordinal=1, arm=collection.TOM,
        base_factory=base, replay_executor=replay,
        predictor_checkout=Path("/fake-checkout"), fit_path=Path("/fake-fit"),
    )
    assert status["status"] == "success"
    assert status["treatment_count"] == 0
    assert _FakePredictor.instances == 1
    path = collection._arm_path(root, 1, collection.TOM)
    ledger = validate_attempt_ledger(path / "attempt_ledger", plan)
    assert len(ledger.claims) == len(ledger.terminals) == 1
    assert ledger.terminals[0].outcome is TerminalOutcome.CANONICAL_SUCCESS
    assert (path / "games" / "game-000" / "manifest.json").is_file()
    assert (path / "private_treatment_audit" / "manifest.json").is_file()
    again = collection._execute_arm(
        root=root, record=record, ordinal=1, arm=collection.TOM,
        base_factory=base, replay_executor=replay,
        predictor_checkout=Path("/fake-checkout"), fit_path=Path("/fake-fit"),
    )
    assert again == status
    assert len(base.calls) == 1


def test_failed_arm_publishes_canonical_failure_terminal(tmp_path, monkeypatch):
    root = tmp_path / "ablation"
    record = _record()
    _freeze(root, record)
    monkeypatch.setattr(
        collection, "run_ablation",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("fixture backend failure")),
    )
    base = _FixtureBaseFactory()
    plan = collection._arm_plan(record, 0, collection.NOTOM)
    claim = construct_attempt_claim(
        plan, ordinal=0, attempt_id="fixture-claim",
        claim_timestamp_utc="2026-09-27T00:00:00Z",
    )
    _, replay = _evidence_with_winner(plan, claim)
    status = collection._execute_arm(
        root=root, record=record, ordinal=0, arm=collection.NOTOM,
        base_factory=base, replay_executor=replay,
        predictor_checkout=Path("/unused"), fit_path=Path("/unused"),
    )
    assert status["status"] == "failure"
    assert status["failure_type"] == "RuntimeError"
    path = collection._arm_path(root, 0, collection.NOTOM)
    ledger = validate_attempt_ledger(path / "attempt_ledger", plan)
    assert ledger.terminals[0].outcome is TerminalOutcome.CANONICAL_FAILURE
    assert len(base.calls) == 1


def test_open_claim_recovers_as_interrupted_without_rerun(tmp_path):
    root = tmp_path / "ablation"
    record = _record()
    _freeze(root, record)
    plan = collection._arm_plan(record, 0, collection.NOTOM)
    path = collection._arm_path(root, 0, collection.NOTOM)
    ledger = initialize_attempt_ledger(path, plan)
    claim = construct_attempt_claim(
        plan, ordinal=0, attempt_id="claim-000",
        claim_timestamp_utc="2026-09-27T00:00:00Z",
    )
    publish_ledger_record(ledger, plan, claim)
    class ForbiddenFactory:
        def __call__(self, **_kwargs):
            raise AssertionError("interrupted game must not restart")
    status = collection._execute_arm(
        root=root, record=record, ordinal=0, arm=collection.NOTOM,
        base_factory=ForbiddenFactory(), replay_executor=None,
        predictor_checkout=Path("/unused"), fit_path=Path("/unused"),
    )
    assert status["terminal_outcome"] == "interrupted_failure"
    assert status["failure_stage"] == "interrupted"
    assert status["status"] == "failure"


def test_pair_record_is_immutable(tmp_path):
    record = _record()
    path = tmp_path / "pair"
    first = {"status": "invalid", "candidate_ordinal": 0}
    collection._publish_record(path, "tom_gameplay_pair", first)
    collection._publish_record(path, "tom_gameplay_pair", first)
    with pytest.raises(ArtifactConflictError):
        collection._publish_record(
            path, "tom_gameplay_pair", {**first, "status": "completed_valid"}
        )


@pytest.mark.parametrize("missing_pair_record", [False, True])
def test_resume_keeps_durable_arms_and_finishes_pair(
    tmp_path, monkeypatch, missing_pair_record,
):
    root = tmp_path / "ablation"
    record = _record()
    _freeze(root, record)
    monkeypatch.setattr(collection, "TARGET_PAIRS", 1)
    monkeypatch.setattr(
        collection, "run_ablation", lambda *_args, **_kwargs: "Werewolf win"
    )
    monkeypatch.setattr(collection, "Qwen3GameplayPredictorClient", _FakePredictor)
    base = _FixtureBaseFactory()
    replay = _evidence_with_winner(
        collection._arm_plan(record, 0, collection.NOTOM),
        construct_attempt_claim(
            collection._arm_plan(record, 0, collection.NOTOM),
            ordinal=0, attempt_id="fixture-claim",
            claim_timestamp_utc="2026-09-27T00:00:00Z",
        ),
    )[1]
    common = dict(
        root=root, record=record, ordinal=0, base_factory=base,
        replay_executor=replay, predictor_checkout=Path("/fake-checkout"),
        fit_path=Path("/fake-fit"),
    )
    first = collection._execute_arm(arm=collection.NOTOM, **common)
    assert first["status"] == "success"
    if missing_pair_record:
        second = collection._execute_arm(arm=collection.TOM, **common)
        assert second["status"] == "success"
    calls_before_resume = len(base.calls)
    collection.collect_pairs(
        root=root, record=record, base_factory=base, replay_executor=replay,
        predictor_checkout=Path("/fake-checkout"), fit_path=Path("/fake-fit"),
    )
    assert len(base.calls) == calls_before_resume + (0 if missing_pair_record else 1)
    pair, _ = collection._read_record(
        collection._pair_path(root, 0), "tom_gameplay_pair"
    )
    assert pair["status"] == "completed_valid"
    assert pair["roles_equal"] is True
    assert pair["seed"] == record["ordered_seed_pool"][0]
    assert len(base.calls) == 2
    collection.collect_pairs(
        root=root, record=record, base_factory=base, replay_executor=replay,
        predictor_checkout=Path("/fake-checkout"), fit_path=Path("/fake-fit"),
    )
    assert len(base.calls) == 2


def test_initial_root_and_resume_plan_identity_gates(tmp_path, monkeypatch):
    record = _record()
    root = tmp_path / "gameplay_ablations" / collection.EXPERIMENT_ID
    root.mkdir(parents=True)
    monkeypatch.setattr(collection.operator, "clean_head", lambda: "a" * 40)
    monkeypatch.setattr(collection.operator, "live_preflight", lambda *_args: None)
    monkeypatch.setattr(collection.cli, "_runtime", lambda *_args: {"fixture": True})
    provenance = {
        **record["runtime_provenance"],
        "runtime_config_sha256": collection._digest({"fixture": True}),
    }
    monkeypatch.setattr(
        collection.operator, "inspect_inputs",
        lambda: (provenance, "http://127.0.0.1:8000/v1"),
    )
    monkeypatch.setattr(collection.cli, "_storage_root", lambda *_args: tmp_path)
    monkeypatch.setattr(
        collection.operator, "read_json", lambda *_args: {
            "collection_id": "fixture", "target_games": 1,
            "seed_pool_size": 1, "call_limit": 32,
        },
    )
    with pytest.raises(ValueError, match="requires --resume"):
        collection.main(["--predictor-checkout", "/unused", "--fit-path", "/unused"])
    with pytest.raises(ValueError, match="existing frozen experiment plan"):
        collection.main([
            "--predictor-checkout", "/unused", "--fit-path", "/unused", "--resume",
        ])
    _freeze(root, collection._experiment_record("a" * 40, provenance, 32))
    monkeypatch.setattr(
        collection.operator, "derive_seed_pool",
        lambda *_args: tuple(range(100)),
    )
    with pytest.raises(ValueError, match="seed pool, or provenance mismatch"):
        collection.main([
            "--predictor-checkout", "/unused", "--fit-path", "/unused", "--resume",
        ])
    monkeypatch.setattr(
        collection.operator, "derive_seed_pool",
        lambda *_args: tuple(record["ordered_seed_pool"]),
    )
    changed = {**provenance, "served_model_name": "different-fixture-model"}
    monkeypatch.setattr(
        collection.operator, "inspect_inputs",
        lambda: (changed, "http://127.0.0.1:8000/v1"),
    )
    with pytest.raises(ValueError, match="seed pool, or provenance mismatch"):
        collection.main([
            "--predictor-checkout", "/unused", "--fit-path", "/unused", "--resume",
        ])
