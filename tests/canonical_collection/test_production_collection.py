from __future__ import annotations

from dataclasses import dataclass
import inspect
from pathlib import Path

import pytest

from tests.canonical_collection.test_game_bundle import _fixture, _plan
from werewolf.canonical_collection import (
    AttemptTerminal,
    CanonicalFailureStage,
    CollectionSeedPoolExhausted,
    TerminalOutcome,
    recover_attempt_ledger,
    validate_attempt_ledger,
    validate_canonical_failure_evidence,
    validate_canonical_game_bundle,
)
from werewolf.canonical_collection.collector import (
    CanonicalAttemptFailure,
    CanonicalGameProduct,
    collect,
)


@dataclass
class _Runtime:
    product: CanonicalGameProduct

    def run(self) -> CanonicalGameProduct:
        return self.product


class _Timestamps:
    def __init__(self) -> None:
        self._index = 0

    def __call__(self) -> str:
        self._index += 1
        return f"2026-09-03T01:{self._index:02d}:00Z"


def _success_factory(plan, calls, collection_directory):
    def factory(*, plan, claim):
        claim_path = (
            collection_directory
            / "attempt_ledger"
            / f"{claim.ordinal}-{claim.seed}.claim.json"
        )
        assert claim_path.is_file()
        calls.append(claim.seed)
        _, _, evidence, replay = _fixture(plan, claim)
        return _Runtime(CanonicalGameProduct(evidence, replay))

    return factory


def test_collect_publishes_claim_then_verified_bundle_then_success_terminal(
    tmp_path,
):
    plan = _plan(target_canonical_success_count=1)
    calls: list[int] = []

    result = collect(
        plan=plan,
        runtime_factory=_success_factory(plan, calls, tmp_path),
        destination=tmp_path,
        timestamp_utc=_Timestamps(),
    )

    state = validate_attempt_ledger(tmp_path / "attempt_ledger", plan)
    assert calls == [101]
    assert result.ledger_state == state
    assert state.target_reached is True
    assert state.canonical_success_count == 1
    assert len(state.claims) == len(state.terminals) == 1
    terminal = state.terminals[0]
    assert terminal.outcome is TerminalOutcome.CANONICAL_SUCCESS
    bundle = validate_canonical_game_bundle(
        tmp_path / "games" / terminal.canonical_game_bundle_id,
        plan=plan,
        claim=state.claims[0],
        replay_executor=_fixture(plan, state.claims[0])[3],
    )
    assert bundle.manifest_digest == terminal.canonical_game_bundle_digest


def test_runtime_factory_is_never_called_before_durable_claim(tmp_path):
    plan = _plan(target_canonical_success_count=1)

    def factory(*, plan, claim):
        claim_path = (
            tmp_path
            / "attempt_ledger"
            / f"{claim.ordinal}-{claim.seed}.claim.json"
        )
        assert claim_path.is_file()
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        collect(
            plan=plan,
            runtime_factory=factory,
            destination=tmp_path,
            timestamp_utc=_Timestamps(),
        )


def test_normal_failure_publishes_bound_evidence_and_exactly_one_terminal(
    tmp_path,
):
    plan = _plan(
        ordered_seed_pool=(101,),
        target_canonical_success_count=1,
    )

    def factory(*, plan, claim):
        del plan, claim

        class FailingRuntime:
            def run(self):
                raise CanonicalAttemptFailure(
                    game_id="game-000",
                    stage=CanonicalFailureStage.GAMEPLAY_ACTION,
                    error_category="GameplayGenerationExhausted",
                    error_message="bounded generation exhausted",
                    retry_exhausted=True,
                )

        return FailingRuntime()

    with pytest.raises(CollectionSeedPoolExhausted):
        collect(
            plan=plan,
            runtime_factory=factory,
            destination=tmp_path,
            timestamp_utc=_Timestamps(),
        )

    state = validate_attempt_ledger(tmp_path / "attempt_ledger", plan)
    assert len(state.claims) == len(state.terminals) == 1
    terminal = state.terminals[0]
    assert terminal.outcome is TerminalOutcome.CANONICAL_FAILURE
    verified = validate_canonical_failure_evidence(
        tmp_path / "attempts" / state.claims[0].attempt_id / "failure_evidence.json",
        plan=plan,
        claim=state.claims[0],
    )
    assert verified.file_sha256 == terminal.failure_evidence_digest
    assert verified.evidence.stage is CanonicalFailureStage.GAMEPLAY_ACTION


def test_interrupted_claim_is_closed_and_resume_starts_next_seed(tmp_path):
    plan = _plan(
        ordered_seed_pool=(101, 202),
        target_canonical_success_count=1,
    )
    calls: list[int] = []

    def interrupted_factory(*, plan, claim):
        del plan
        calls.append(claim.seed)
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        collect(
            plan=plan,
            runtime_factory=interrupted_factory,
            destination=tmp_path,
            timestamp_utc=_Timestamps(),
        )

    collect(
        plan=plan,
        runtime_factory=_success_factory(plan, calls, tmp_path),
        destination=tmp_path,
        timestamp_utc=_Timestamps(),
    )

    state = validate_attempt_ledger(tmp_path / "attempt_ledger", plan)
    assert calls == [101, 202]
    assert [terminal.outcome for terminal in state.terminals] == [
        TerminalOutcome.INTERRUPTED_FAILURE,
        TerminalOutcome.CANONICAL_SUCCESS,
    ]
    assert [claim.seed for claim in state.claims] == [101, 202]


def test_target_success_stops_without_claiming_another_seed(tmp_path):
    plan = _plan(
        ordered_seed_pool=(101, 202, 303),
        target_canonical_success_count=1,
    )
    calls: list[int] = []
    factory = _success_factory(plan, calls, tmp_path)

    first = collect(
        plan=plan,
        runtime_factory=factory,
        destination=tmp_path,
        timestamp_utc=_Timestamps(),
    )
    second = collect(
        plan=plan,
        runtime_factory=factory,
        destination=tmp_path,
        timestamp_utc=_Timestamps(),
    )

    assert first.ledger_state == second.ledger_state
    assert calls == [101]


def test_crash_after_bundle_publication_leaves_claim_for_interrupted_closure(
    tmp_path,
    monkeypatch,
):
    plan = _plan(target_canonical_success_count=1)
    calls: list[int] = []
    from werewolf.canonical_collection import collector as collector_module

    publish_record = collector_module.publish_ledger_record

    def crash_before_success_terminal(directory, plan, record):
        if isinstance(record, AttemptTerminal):
            raise RuntimeError("simulated process crash before terminal publish")
        return publish_record(directory, plan, record)

    monkeypatch.setattr(
        collector_module,
        "publish_ledger_record",
        crash_before_success_terminal,
    )

    with pytest.raises(RuntimeError, match="simulated process crash"):
        collect(
            plan=plan,
            runtime_factory=_success_factory(plan, calls, tmp_path),
            destination=tmp_path,
            timestamp_utc=_Timestamps(),
        )

    open_state = validate_attempt_ledger(tmp_path / "attempt_ledger", plan)
    assert open_state.open_claim is not None
    assert open_state.terminals == ()
    assert (tmp_path / "games" / "game-000" / "manifest.json").is_file()

    recovered = recover_attempt_ledger(
        tmp_path / "attempt_ledger",
        plan,
        interruption_timestamp_utc="2026-09-03T02:00:00Z",
    )
    assert recovered.terminals[0].outcome is TerminalOutcome.INTERRUPTED_FAILURE


def test_superseded_collection_entry_points_and_configs_are_absent():
    import importlib.util

    import run_random

    parameters = inspect.signature(run_random.eval).parameters
    assert "sample_collector" not in parameters
    assert "trajectory_recorder" not in parameters
    assert "allow_gameplay_fallback" not in parameters
    assert not hasattr(run_random, "build_twd_tom_sample_collector")
    repository = Path(__file__).resolve().parents[2]
    assert not (repository / "werewolf" / "models" / "twd_tom" / "__init__.py").exists()
    assert not (repository / "script" / "twd_tom" / "collect_canonical_trajectories.py").exists()
    assert not (repository / "configs" / "twd_tom_server_qwen35_9b.yaml").exists()
    assert not (
        repository / "configs" / "twd_tom_server_qwen35_9b_canonical_60.yaml"
    ).exists()
