"""The sole plan-closed executable Canonical Collection orchestrator."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from werewolf.canonical_collection.attempt_ledger import (
    AttemptClaim,
    AttemptLedgerState,
    CollectionPlan,
    CollectionSeedPoolExhausted,
    TerminalOutcome,
    construct_attempt_claim,
    construct_attempt_terminal,
    initialize_attempt_ledger,
    next_planned_attempt,
    publish_ledger_record,
    recover_attempt_ledger,
    validate_attempt_ledger,
    validate_collection_plan,
)
from werewolf.canonical_collection.failure_evidence import (
    CanonicalFailureAttempt,
    CanonicalFailureStage,
    construct_canonical_failure_attempt,
    construct_canonical_failure_evidence,
    construct_canonical_partial_evidence,
    publish_canonical_failure_evidence,
    publish_canonical_partial_evidence,
)
from werewolf.canonical_collection.game_bundle import (
    DeterministicReplayExecutor,
    publish_canonical_game_bundle,
)
from werewolf.canonical_collection.trajectory_evidence import (
    CanonicalGameEvidence,
)


@dataclass(frozen=True)
class CanonicalGameProduct:
    """One completed runtime result awaiting durable Bundle publication."""

    evidence: CanonicalGameEvidence
    replay_executor: DeterministicReplayExecutor


class CanonicalGameRuntime(Protocol):
    def run(self) -> CanonicalGameProduct: ...


class RuntimeFactory(Protocol):
    def __call__(
        self,
        *,
        plan: CollectionPlan,
        claim: AttemptClaim,
    ) -> CanonicalGameRuntime: ...


class CanonicalAttemptFailure(RuntimeError):
    """A normal, terminal attempt failure with complete causal context."""

    def __init__(
        self,
        *,
        game_id: str,
        stage: CanonicalFailureStage,
        error_category: str,
        error_message: str,
        retry_exhausted: bool,
        boundary_id: str | None = None,
        observer_id: str | None = None,
        day: int | None = None,
        phase: str | None = None,
        attempt_evidence: Sequence[CanonicalFailureAttempt] = (),
        partial_payload: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(error_message)
        self.game_id = game_id
        self.stage = stage
        self.error_category = error_category
        self.error_message = error_message
        self.retry_exhausted = retry_exhausted
        self.boundary_id = boundary_id
        self.observer_id = observer_id
        self.day = day
        self.phase = phase
        self.attempt_evidence = tuple(attempt_evidence)
        self.partial_payload = (
            None if partial_payload is None else dict(partial_payload)
        )


@dataclass(frozen=True)
class CollectionResult:
    """Non-authoritative convenience view of the validated Attempt Ledger."""

    collection_directory: Path
    ledger_state: AttemptLedgerState


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _attempt_id(ordinal: int, seed: int) -> str:
    return f"attempt-{ordinal:06d}-seed-{seed}"


def _runtime_failure_attempt(
    plan: CollectionPlan,
    failure: CanonicalAttemptFailure,
) -> CanonicalFailureAttempt:
    return construct_canonical_failure_attempt(
        attempt_index=1,
        call_id=f"{failure.game_id}-failure-01",
        backend_identity=plan.backend_identity,
        model_identity=plan.model_identity,
        parser_identity=plan.parser_identity,
        prompt_identity=plan.prompt_identity,
        retry_policy_identity=plan.retry_policy_identity,
        call_budget_identity=plan.call_budget_identity,
        error_category=failure.error_category,
        error_message=failure.error_message,
        response_digest=None,
    )


def _publish_failure(
    *,
    collection_directory: Path,
    ledger_directory: Path,
    plan: CollectionPlan,
    claim: AttemptClaim,
    failure: CanonicalAttemptFailure,
    terminal_timestamp_utc: str,
) -> None:
    partials = ()
    if failure.partial_payload is not None:
        partial = construct_canonical_partial_evidence(
            plan=plan,
            claim=claim,
            evidence_id="runtime-state",
            evidence_type="canonical_collection_runtime_state",
            payload=failure.partial_payload,
        )
        partials = (
            publish_canonical_partial_evidence(
                collection_directory,
                plan=plan,
                claim=claim,
                evidence=partial,
            ),
        )
    attempts = failure.attempt_evidence or (
        _runtime_failure_attempt(plan, failure),
    )
    evidence = construct_canonical_failure_evidence(
        plan=plan,
        claim=claim,
        game_id=failure.game_id,
        stage=failure.stage,
        error_category=failure.error_category,
        boundary_id=failure.boundary_id,
        observer_id=failure.observer_id,
        day=failure.day,
        phase=failure.phase,
        retry_exhausted=failure.retry_exhausted,
        attempt_evidence=attempts,
        partial_evidence=partials,
    )
    verified = publish_canonical_failure_evidence(
        collection_directory,
        plan=plan,
        claim=claim,
        evidence=evidence,
    )
    terminal = construct_attempt_terminal(
        plan,
        claim,
        outcome=TerminalOutcome.CANONICAL_FAILURE,
        failure_evidence_digest=verified.file_sha256,
        partial_evidence=tuple(item.ledger_reference for item in partials),
        terminal_timestamp_utc=terminal_timestamp_utc,
    )
    publish_ledger_record(ledger_directory, plan, terminal)


def collect(
    *,
    plan: CollectionPlan,
    runtime_factory: RuntimeFactory,
    destination: Path | str,
    timestamp_utc: Callable[[], str] = _utc_now,
) -> CollectionResult:
    """Execute or resume exactly one immutable production Collection Plan.

    The runtime factory is deliberately invoked only after the claim record's
    durable no-replace publication has returned successfully.
    """

    validate_collection_plan(plan)
    if not callable(runtime_factory):
        raise TypeError("runtime_factory must be callable")
    if not callable(timestamp_utc):
        raise TypeError("timestamp_utc must be callable")
    collection_directory = Path(destination)
    ledger_directory = initialize_attempt_ledger(collection_directory, plan)
    state = recover_attempt_ledger(
        ledger_directory,
        plan,
        interruption_timestamp_utc=(
            timestamp_utc()
            if validate_attempt_ledger(ledger_directory, plan).open_claim
            else None
        ),
    )

    while not state.target_reached:
        planned = next_planned_attempt(plan, state)
        claim = construct_attempt_claim(
            plan,
            ordinal=planned.ordinal,
            attempt_id=_attempt_id(planned.ordinal, planned.seed),
            claim_timestamp_utc=timestamp_utc(),
        )
        publish_ledger_record(ledger_directory, plan, claim)

        try:
            runtime = runtime_factory(plan=plan, claim=claim)
            if runtime is None or not callable(getattr(runtime, "run", None)):
                raise TypeError("runtime_factory must return a runtime with run()")
            product = runtime.run()
            if not isinstance(product, CanonicalGameProduct):
                raise TypeError("runtime must return CanonicalGameProduct")
        except CanonicalAttemptFailure as failure:
            _publish_failure(
                collection_directory=collection_directory,
                ledger_directory=ledger_directory,
                plan=plan,
                claim=claim,
                failure=failure,
                terminal_timestamp_utc=timestamp_utc(),
            )
        except Exception as error:
            failure = CanonicalAttemptFailure(
                game_id=(
                    f"{plan.collection_id}-game-{claim.ordinal:06d}"
                    f"-seed-{claim.seed}"
                ),
                stage=CanonicalFailureStage.RUNTIME,
                error_category=type(error).__name__,
                error_message=str(error) or type(error).__name__,
                retry_exhausted=False,
            )
            _publish_failure(
                collection_directory=collection_directory,
                ledger_directory=ledger_directory,
                plan=plan,
                claim=claim,
                failure=failure,
                terminal_timestamp_utc=timestamp_utc(),
            )
        else:
            # Artifact publication is deliberately outside the runtime-error
            # conversion block. A crash while publishing the immutable Bundle
            # or its success terminal leaves an open durable claim; resume then
            # closes it as interrupted_failure and never reruns the seed.
            bundle = publish_canonical_game_bundle(
                collection_directory / "games" / product.evidence.game_id,
                plan=plan,
                claim=claim,
                evidence=product.evidence,
                replay_executor=product.replay_executor,
            )
            terminal = construct_attempt_terminal(
                plan,
                claim,
                outcome=TerminalOutcome.CANONICAL_SUCCESS,
                canonical_game_bundle_id=product.evidence.game_id,
                canonical_game_bundle_digest=bundle.manifest_digest,
                terminal_timestamp_utc=timestamp_utc(),
            )
            publish_ledger_record(ledger_directory, plan, terminal)
        state = validate_attempt_ledger(ledger_directory, plan)

    return CollectionResult(
        collection_directory=collection_directory,
        ledger_state=state,
    )


__all__ = [
    "CanonicalAttemptFailure",
    "CanonicalGameProduct",
    "CanonicalGameRuntime",
    "CollectionResult",
    "RuntimeFactory",
    "collect",
]
