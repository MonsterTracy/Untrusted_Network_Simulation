"""Focused contract tests for the frozen paired gameplay collection."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import tom_gameplay_collection as collection
from scripts import analyze_tom_gameplay_ablation as analysis
from scripts.tom_gameplay_ablation import prediction_request
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


def _epoch35_record(monkeypatch, name):
    from scripts.run_werewolf_baseline import _scenarios

    baseline = {"default_profile": "local_qwen", "scenarios": _scenarios("local_qwen")}
    baseline["plan_sha256"] = collection._digest(baseline)
    with pytest.raises(ValueError, match="frozen scenario source"):
        collection._epoch35_record("a" * 40, {"served_model_name": "qwen35-9b"}, 32,
                                   experiment_id=name, baseline_plan=baseline, device="cpu")
    monkeypatch.setattr(collection, "BASELINE_PLAN_SHA256", baseline["plan_sha256"])
    return collection._epoch35_record("a" * 40, {"served_model_name": "qwen35-9b"}, 32,
                                      experiment_id=name, baseline_plan=baseline, device="cpu")


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


def test_epoch35_attempts_all_scenarios_and_reports_both_populations(tmp_path, monkeypatch):
    from tests.tom.test_analyze_gameplay_ablation import _arm
    from werewolf.canonical_collection.public_history import PLAYER_IDS

    root = tmp_path / "ablation"
    record = _epoch35_record(monkeypatch, root.name)
    _freeze(root, record)
    issued = []

    def execute_arm(**kwargs):
        ordinal, arm = kwargs["ordinal"], kwargs["arm"]
        issued.append((ordinal, arm))
        status = _arm(record, ordinal, arm, failure=ordinal == 0 and arm == collection.NOTOM,
                      winner="Werewolf" if arm == collection.TOM else "Villager")
        for name in ("predictor_fit_digest", "predictor_seal_digest"):
            del status[name]
        status.update(collection._predictor_fields(record, arm))
        status.update(attempt_id="fixture", failure_evidence_digest=None)
        if status["status"] == "success":
            scene = record["baseline_plan"]["scenarios"][ordinal]
            status["roles"] = dict(zip(PLAYER_IDS, scene["roles"], strict=True))
        else:
            status["failure_evidence_digest"] = "e" * 64
        return status

    monkeypatch.setattr(collection, "_execute_arm", execute_arm)
    digest = collection.collect_pairs(root=root, record=record, base_factory=None, replay_executor=None)
    assert issued == [(ordinal, arm) for ordinal in range(210) for arm in collection._order(ordinal)]

    # 只替换抽样次数，分析入口仍核验同一套 Plan、Final 与 210 个 Pair 封存。
    effect = analysis._paired_ratio_effects
    monkeypatch.setattr(analysis, "_paired_ratio_effects", lambda rows: effect(rows, replicates=8))
    result = analysis.analyze_epoch35_collection(root, expected_final_digest=digest)
    assert result["attempted_tasks"] == 420
    assert result["all_valid_games"][collection.TOM]["WWR"]["denominator"] == 210
    assert result["all_valid_games"][collection.NOTOM]["WWR"]["denominator"] == 209
    assert result["paired_valid_games"][collection.TOM]["WWR"]["denominator"] == 209
    assert result["paired"]["N"] == result["paired"]["n01"] == 209
    assert result["paired"]["metrics"]["WWR"]["bootstrap_95_ci"] == [1.0, 1.0]
    assert result["paired"]["metrics"]["GMR"]["delta"] == "UNKNOWN"
    assert result["games"][0]["paired_valid"] is False
    with pytest.raises(ValueError, match="digest mismatch"):
        analysis.analyze_epoch35_collection(root, expected_final_digest="0" * 64)
    arguments = ["--collection-root", str(root), "--final-digest", digest]
    with pytest.raises(ValueError, match="outside historical evidence"):
        analysis.main([*arguments, "--output-dir", str(root.parent / collection.EXPERIMENT_ID)])
    assert analysis.main(arguments) == 0
    assert (root.with_name("ablation-analysis") / "analysis.json").is_file()


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


def _evidence_with_winner(plan, claim, *, game_id="game-000"):
    _, _, original, replay = _fixture(plan, claim, game_id=game_id)
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

    def __call__(self, *, plan, claim, roles=None, record_requests=False):
        self.calls.append((plan.collection_id, claim.seed))
        game_id = f"{plan.collection_id}-game-000000-seed-{claim.seed}" if record_requests else "game-000"
        evidence, replay = _evidence_with_winner(plan, claim, game_id=game_id)
        self.replay = replay
        recorder = SimpleNamespace(
            complete_evidence=lambda: evidence,
            failure_from_exception=lambda error, **_kwargs: error,
        )
        return SimpleNamespace(
            env=object(), agents=[], roles=roles if roles is not None else ["Werewolf"] * 7,
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


@pytest.mark.parametrize("epoch35", (False, True))
def test_tom_wrapper_uses_real_collector_ledger_and_bundle(
    tmp_path, monkeypatch, epoch35,
):
    root = tmp_path / "ablation"
    record = _epoch35_record(monkeypatch, root.name) if epoch35 else _record()
    ordinal = 0 if epoch35 else 1
    _freeze(root, record)
    monkeypatch.setattr(collection, "run_ablation", lambda *_args, **_kwargs: "Werewolf win")
    monkeypatch.setattr(collection, "Qwen3GameplayPredictorClient", _FakePredictor)
    _FakePredictor.instances = 0
    base = _FixtureBaseFactory()
    plan = collection._arm_plan(record, ordinal, collection.TOM)
    claim = construct_attempt_claim(
        plan, ordinal=0, attempt_id="fixture-claim",
        claim_timestamp_utc="2026-09-27T00:00:00Z",
    )
    _, replay = _evidence_with_winner(plan, claim)
    predictor = SimpleNamespace(identity=record["predictor"]) if epoch35 else None
    args = dict(
        root=root, record=record, ordinal=ordinal, arm=collection.TOM,
        base_factory=base, replay_executor=replay,
        predictor_checkout=Path("/fake-checkout"), fit_path=Path("/fake-fit"),
        epoch35_predictor=predictor,
    )
    status = collection._execute_arm(**args)
    assert status["status"] == "success"
    assert status["treatment_count"] == 0
    assert _FakePredictor.instances == (0 if epoch35 else 1)
    path = collection._arm_path(root, ordinal, collection.TOM)
    ledger = validate_attempt_ledger(path / "attempt_ledger", plan)
    assert len(ledger.claims) == len(ledger.terminals) == 1
    assert ledger.terminals[0].outcome is TerminalOutcome.CANONICAL_SUCCESS
    assert (path / "games" / status["game_id"] / "manifest.json").is_file()
    assert (path / "private_treatment_audit" / "manifest.json").is_file()
    again = collection._execute_arm(**args)
    assert again == status
    assert len(base.calls) == 1
    if epoch35:
        pair = {"candidate_ordinal": ordinal, "seed": claim.seed, "arms": {collection.TOM: status}}
        scored = analysis._coverage_game(root, record, pair, collection.TOM)
        assert scored["vote_counts"]["wolf_participations"] == 2
        assert scored["tom_integration"]["A"] == 0
        assert scored["cost"]["game_elapsed_seconds"] >= 0


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


@pytest.mark.parametrize("interrupted", (False, True))
def test_epoch35_preserves_failed_prediction_and_does_not_redispatch(tmp_path, monkeypatch, interrupted):
    from werewolf.canonical_collection.collector import CanonicalAttemptFailure
    from werewolf.canonical_collection.failure_evidence import CanonicalFailureStage

    root = tmp_path / "ablation"
    record = _epoch35_record(monkeypatch, root.name)
    _freeze(root, record)
    predictor = SimpleNamespace(identity=record["predictor"])
    invocations = []

    def make_runtime(*, plan, claim, roles, record_requests):
        assert roles == record["baseline_plan"]["scenarios"][0]["roles"] and record_requests is True
        game_id = f"{plan.collection_id}-game-000000-seed-{claim.seed}"
        prefix = _fixture(plan, claim, game_id=game_id)[2].authoritative_pre_prefixes[0]
        invocations.append(prefix)

        def failure(error, **_kwargs):
            return CanonicalAttemptFailure(game_id=game_id, stage=CanonicalFailureStage.RUNTIME,
                error_category=type(error).__name__, error_message="fixture failure", retry_exhausted=False,
                partial_payload={"authoritative_pre_prefixes": [prefix.to_record()], "backend_calls": []})

        return SimpleNamespace(env=None, agents=[SimpleNamespace(decision_method=None)] * 7, roles=roles,
                               recorder=SimpleNamespace(game_id=game_id, failure_from_exception=failure),
                               call_audit=None, replay_executor=None)

    def fail(*_args, **kwargs):
        assert kwargs["predictor"] is predictor
        status = {"status": "unknown"} if interrupted else {"status": "failed", "error_type": "ValueError"}
        kwargs["prediction_attempts"].append({**prediction_request(invocations[-1]), **status})
        if interrupted:
            raise KeyboardInterrupt
        raise ValueError("fixture forecast failed")

    monkeypatch.setattr(collection, "run_ablation", fail)
    args = dict(root=root, record=record, ordinal=0, arm=collection.TOM,
                base_factory=make_runtime, replay_executor=None, epoch35_predictor=predictor)
    if interrupted:
        with pytest.raises(KeyboardInterrupt):
            collection._execute_arm(**args)
    status = collection._execute_arm(**args)
    assert status["terminal_outcome"] == ("interrupted_failure" if interrupted else "canonical_failure")
    assert collection._execute_arm(**args) == status and len(invocations) == 1
    pair = {"candidate_ordinal": 0, "seed": record["ordered_seed_pool"][0], "arms": {collection.TOM: status}}
    evidence = analysis._failed_epoch35_game(root, record, pair, collection.TOM)
    assert evidence["tom_integration"]["A"] == 1
    assert evidence["tom_integration"]["P"] == ("UNKNOWN" if interrupted else 0)
    assert evidence["vote_counts"] == "UNKNOWN"


@pytest.mark.parametrize("epoch35", (False, True))
def test_open_claim_recovers_as_interrupted_without_rerun(tmp_path, monkeypatch, epoch35):
    root = tmp_path / "ablation"
    record = _epoch35_record(monkeypatch, root.name) if epoch35 else _record()
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


def test_epoch35_cli_binds_baseline_checkpoint_and_uses_existing_collector(tmp_path, monkeypatch):
    from unittest.mock import Mock
    from werewolf.artifact_io import canonical_json_bytes
    from werewolf.canonical_collection.game_bundle import FullGameReplayExecutor

    root = tmp_path.resolve()
    archive = root / "gameplay_ablations/ablation"
    plan = _epoch35_record(monkeypatch, archive.name)
    config = collection.cli._runtime(collection.operator.RUNTIME)
    provenance = {**plan["runtime_provenance"], "runtime_config_sha256": collection._digest(config)}
    baseline = root / "werewolf-baseline/formal/scenarios.json"
    baseline.parent.mkdir(parents=True)
    before = canonical_json_bytes(plan["baseline_plan"])
    baseline.write_bytes(before)
    checkpoint_path, temporal_path = root / "training/checkpoint", root / "temporal/code"
    monkeypatch.setattr(collection.operator, "clean_head", lambda: "a" * 40)
    monkeypatch.setattr(collection.operator, "inspect_inputs", lambda: (provenance, "http://unused/v1"))
    monkeypatch.setattr(collection.operator, "live_preflight", Mock())
    monkeypatch.setattr(collection.cli, "_storage_root", lambda *_args: root)
    monkeypatch.setattr(collection.operator, "read_json", lambda *_args: {
        "collection_id": "fixture", "target_games": 1, "seed_pool_size": 1, "call_limit": 32,
    })
    monkeypatch.setattr(collection, "load_named_backends", Mock(return_value={}))
    predictor = SimpleNamespace(identity=plan["predictor"])
    load = Mock(return_value=predictor)
    monkeypatch.setattr(collection, "Epoch35GameplayPredictor", load)
    collect = Mock(return_value="1" * 64)
    monkeypatch.setattr(collection, "collect_pairs", collect)
    args = ["--baseline-plan", str(baseline), "--checkpoint", str(checkpoint_path),
            "--temporal", str(temporal_path), "--destination", archive.name]
    assert collection.main(args) == 0
    load.assert_called_once_with(checkpoint_path, temporal_path, "cpu")
    assert collect.call_args.kwargs["root"] == archive
    assert collect.call_args.kwargs["epoch35_predictor"] is predictor
    assert isinstance(collect.call_args.kwargs["replay_executor"], FullGameReplayExecutor)
    assert baseline.read_bytes() == before
    stored, _ = collection._read_record(archive / "experiment_plan", "tom_gameplay_experiment_plan")
    assert stored["baseline_plan"] == plan["baseline_plan"]
    with pytest.raises(ValueError, match="seed pool, or provenance mismatch"):
        collection.main([*args, "--device", "cuda", "--resume"])
    assert load.call_count == collect.call_count == 1
