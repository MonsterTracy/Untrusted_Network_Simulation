import errno
from dataclasses import replace

import pytest

from werewolf.artifact_io import (
    canonical_json_bytes,
    publish_artifact,
    sha256_bytes,
)
from werewolf.canonical_collection import attempt_ledger as ledger_module
from werewolf.canonical_collection.attempt_ledger import (
    CollectionSeedPoolExhausted,
    CollectionTargetReached,
    InjectedLedgerCrash,
    LedgerDurabilityUnsupportedError,
    LedgerPublishStep,
    LedgerRecordConflictError,
    LedgerValidationError,
    PartialEvidenceReference,
    TerminalOutcome,
    construct_attempt_claim,
    construct_attempt_terminal,
    construct_collection_plan,
    initialize_attempt_ledger,
    next_planned_attempt,
    preflight_attempt_ledger,
    publish_ledger_record,
    recover_attempt_ledger,
    validate_attempt_claim,
    validate_attempt_ledger,
    validate_attempt_terminal,
    validate_collection_plan,
)


def _plan(**overrides):
    arguments = {
        "collection_id": "collection-001",
        "ordered_seed_pool": (101, 202, 303),
        "target_canonical_success_count": 2,
        "runtime_identity": "classic7-runtime-v1",
        "agent_identity": "canonical-playing-agent-v1",
        "backend_identity": "deterministic-fixture-backend-v1",
        "model_identity": "fixture-model-v1",
        "parser_identity": "v1-parser-v1",
        "prompt_identity": "v1-prompt-v1",
        "retry_policy_identity": "bounded-retries-v1",
        "call_budget_identity": "fixture-call-budget-v1",
        "public_event_schema_version": "classic7_public_event_history_v1",
        "pre_prefix_schema_version": "classic7_authoritative_pre_prefix_v1",
        "belief_observation_schema_version": "belief-observation-v1",
        "v1_annotation_schema_version": "classic7_speech_annotation_v1",
        "bundle_schema_version": "classic7-canonical-game-bundle-v1",
        "source_revision": "2187ba4d2965d2485e57177a75b75aea5c6bf1bb",
        "environment_provenance": {
            "implementation": "phase1-commit3",
            "python": "3.12",
        },
    }
    arguments.update(overrides)
    return construct_collection_plan(**arguments)


def test_collection_plan_freezes_one_production_identity():
    plan = _plan()

    assert validate_collection_plan(plan) is plan
    assert plan.ordered_seed_pool == (101, 202, 303)
    assert plan.target_canonical_success_count == 2
    assert plan.to_record()["environment_provenance"] == {
        "implementation": "phase1-commit3",
        "python": "3.12",
    }
    assert "mode" not in plan.to_record()
    assert plan.runtime_provenance_digest == (
        "bd9c50c5a0ceaffd0535925b532f995f4c08ad85d1e4d7ec39f5b59f568303be"
    )
    assert plan.plan_digest == (
        "dee50f26eda0499fa344617397f7fd71f980692d53a893fe83cee12b78364c62"
    )


def test_nested_collection_ancestors_are_durable_before_first_claim(tmp_path, monkeypatch):
    from werewolf.artifact_io import canonical
    synced = []
    sync = canonical._fsync_directory

    def record(path):
        sync(path)
        synced.append(path)

    monkeypatch.setattr(canonical, "_fsync_directory", record)
    root = tmp_path / "new" / "collection"
    initialize_attempt_ledger(root, _plan())
    assert root in synced and root.parent in synced and tmp_path in synced


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"ordered_seed_pool": (101, 101)}, "unique"),
        ({"ordered_seed_pool": ()}, "non-empty"),
        ({"target_canonical_success_count": 0}, "positive"),
        ({"target_canonical_success_count": 4}, "seed pool"),
        ({"environment_provenance": {}}, "environment provenance"),
    ],
)
def test_collection_plan_rejects_invalid_seed_target_or_provenance(
    overrides,
    match,
):
    with pytest.raises((TypeError, ValueError), match=match):
        _plan(**overrides)


def test_collection_plan_validation_rejects_frozen_provenance_drift():
    plan = _plan()

    with pytest.raises(ValueError, match="runtime provenance digest"):
        validate_collection_plan(
            replace(plan, runtime_identity="changed-runtime")
        )


def _claim(plan=None, *, ordinal=0, attempt_id="attempt-000"):
    plan = _plan() if plan is None else plan
    return construct_attempt_claim(
        plan,
        ordinal=ordinal,
        attempt_id=attempt_id,
        claim_timestamp_utc="2026-09-03T01:02:03Z",
    )


def test_claim_binds_plan_seed_attempt_and_runtime_provenance():
    plan = _plan()
    claim = _claim(plan)

    assert validate_attempt_claim(claim, plan) is claim
    assert claim.plan_digest == plan.plan_digest
    assert claim.ordinal == 0
    assert claim.seed == 101
    assert claim.attempt_id == "attempt-000"
    assert claim.runtime_provenance_digest == plan.runtime_provenance_digest
    assert len(claim.record_digest) == 64


@pytest.mark.parametrize("attempt_id", ["../outside", "nested/attempt", "bad\\id"])
def test_claim_rejects_attempt_id_that_is_not_one_safe_path_component(attempt_id):
    with pytest.raises(ValueError, match="safe path component"):
        _claim(attempt_id=attempt_id)


@pytest.mark.parametrize(
    ("outcome", "fields"),
    [
        (
            TerminalOutcome.CANONICAL_SUCCESS,
            {
                "canonical_game_bundle_id": "game-000",
                "canonical_game_bundle_digest": "a" * 64,
            },
        ),
        (
            TerminalOutcome.CANONICAL_FAILURE,
            {
                "failure_evidence_digest": "b" * 64,
                "partial_evidence": (
                    PartialEvidenceReference(
                        relative_path=(
                            "attempts/attempt-000/partial_evidence/calls.jsonl"
                        ),
                        sha256="c" * 64,
                    ),
                ),
            },
        ),
        (
            TerminalOutcome.INTERRUPTED_FAILURE,
            {"partial_evidence": ()},
        ),
    ],
)
def test_terminal_outcomes_have_one_frozen_evidence_shape(outcome, fields):
    plan = _plan()
    terminal = construct_attempt_terminal(
        plan,
        _claim(plan),
        outcome=outcome,
        terminal_timestamp_utc="2026-09-03T01:03:00Z",
        **fields,
    )

    assert validate_attempt_terminal(terminal, plan, _claim(plan)) is terminal
    assert terminal.outcome is outcome
    assert len(terminal.record_digest) == 64


def test_terminal_rejects_success_without_bundle_digest():
    plan = _plan()

    with pytest.raises(ValueError, match="bundle digest"):
        construct_attempt_terminal(
            plan,
            _claim(plan),
            outcome=TerminalOutcome.CANONICAL_SUCCESS,
            terminal_timestamp_utc="2026-09-03T01:03:00Z",
        )


def _ledger(tmp_path, plan):
    return initialize_attempt_ledger(tmp_path, plan)


def _write_failure_evidence(ledger_directory, claim, data=b"failure evidence"):
    path = (
        ledger_directory.parent
        / "attempts"
        / claim.attempt_id
        / "failure_evidence.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return sha256_bytes(data)


def _publish_test_bundle(ledger_directory, plan, game_id="game-000"):
    return publish_artifact(
        ledger_directory.parent / "games" / game_id,
        manifest_fields={
            "artifact_type": "canonical_game_bundle",
            "schema_version": plan.bundle_schema_version,
            "artifact_identity": game_id,
        },
        files={"public/fixture.json": canonical_json_bytes({"fixture": True})},
    )


def _terminal(
    plan,
    claim,
    ledger_directory,
    *,
    outcome=TerminalOutcome.CANONICAL_FAILURE,
):
    if outcome is TerminalOutcome.CANONICAL_SUCCESS:
        bundle = _publish_test_bundle(ledger_directory, plan)
        fields = {
            "canonical_game_bundle_id": "game-000",
            "canonical_game_bundle_digest": bundle.manifest_digest,
        }
    else:
        fields = {
            "failure_evidence_digest": _write_failure_evidence(
                ledger_directory,
                claim,
            )
        }
    return construct_attempt_terminal(
        plan,
        claim,
        outcome=outcome,
        terminal_timestamp_utc="2026-09-03T01:03:00Z",
        **fields,
    )


def test_claim_and_terminal_publish_as_one_append_only_ledger(tmp_path):
    plan = _plan()
    ledger_directory = _ledger(tmp_path, plan)
    claim = _claim(plan)

    claim_path = publish_ledger_record(ledger_directory, plan, claim)
    terminal_path = publish_ledger_record(
        ledger_directory,
        plan,
        _terminal(plan, claim, ledger_directory),
    )
    state = validate_attempt_ledger(ledger_directory, plan)

    assert claim_path.name == "0-101.claim.json"
    assert terminal_path.name == "0-101.terminal.json"
    assert state.claims == (claim,)
    assert state.terminals[0].outcome is TerminalOutcome.CANONICAL_FAILURE
    assert state.canonical_success_count == 0
    assert state.open_claim is None
    assert next_planned_attempt(plan, state).ordinal == 1
    assert next_planned_attempt(plan, state).seed == 202
    assert not tuple((ledger_directory / ".staging").iterdir())


@pytest.mark.parametrize(
    ("crash_after", "claim_is_authoritative"),
    [
        (LedgerPublishStep.STAGING_CREATED, False),
        (LedgerPublishStep.COMPLETE_WRITE, False),
        (LedgerPublishStep.FILE_FSYNCED, False),
        (LedgerPublishStep.HARD_LINK_PUBLISHED, True),
        (LedgerPublishStep.LEDGER_DIRECTORY_FSYNCED, True),
    ],
)
def test_recovery_observes_no_record_or_one_complete_record_at_every_crash_point(
    tmp_path,
    crash_after,
    claim_is_authoritative,
):
    plan = _plan()
    ledger_directory = _ledger(tmp_path, plan)

    with pytest.raises(InjectedLedgerCrash, match=crash_after.value):
        publish_ledger_record(
            ledger_directory,
            plan,
            _claim(plan),
            crash_after=crash_after,
        )

    assert tuple((ledger_directory / ".staging").iterdir())
    state = recover_attempt_ledger(
        ledger_directory,
        plan,
        interruption_timestamp_utc="2026-09-03T01:04:00Z",
    )

    assert not tuple((ledger_directory / ".staging").iterdir())
    if claim_is_authoritative:
        assert len(state.claims) == 1
        assert len(state.terminals) == 1
        assert state.terminals[0].outcome is TerminalOutcome.INTERRUPTED_FAILURE
        assert next_planned_attempt(plan, state).ordinal == 1
    else:
        assert state.claims == ()
        assert state.terminals == ()
        assert next_planned_attempt(plan, state).ordinal == 0


def _write_forged_record(ledger_directory, record):
    preflight_attempt_ledger(ledger_directory)
    record_type = (
        "claim"
        if record.to_record()["record_type"] == "claim"
        else "terminal"
    )
    path = ledger_directory / (
        f"{record.ordinal}-{record.seed}.{record_type}.json"
    )
    path.write_bytes(canonical_json_bytes(record.to_record()))
    return path


def test_preexisting_final_path_is_never_overwritten_or_reused(tmp_path):
    plan = _plan()
    ledger_directory = _ledger(tmp_path, plan)
    final_path = ledger_directory / "0-101.claim.json"
    final_path.write_bytes(b"preexisting")

    with pytest.raises(LedgerRecordConflictError):
        publish_ledger_record(ledger_directory, plan, _claim(plan))

    assert final_path.read_bytes() == b"preexisting"


def test_corrupt_final_record_fails_closed(tmp_path):
    plan = _plan()
    ledger_directory = _ledger(tmp_path, plan)
    path = publish_ledger_record(ledger_directory, plan, _claim(plan))
    corrupted = bytearray(path.read_bytes())
    corrupted[-1] ^= 1
    path.write_bytes(corrupted)

    with pytest.raises(LedgerValidationError, match="invalid ledger record"):
        validate_attempt_ledger(ledger_directory, plan)


def test_out_of_order_claim_gap_fails_closed(tmp_path):
    plan = _plan()
    ledger_directory = _ledger(tmp_path, plan)
    _write_forged_record(
        ledger_directory,
        _claim(plan, ordinal=1, attempt_id="attempt-001"),
    )

    with pytest.raises(LedgerValidationError, match="out-of-order"):
        validate_attempt_ledger(ledger_directory, plan)


def test_terminal_without_claim_fails_closed(tmp_path):
    plan = _plan()
    ledger_directory = _ledger(tmp_path, plan)
    claim = _claim(plan)
    _write_forged_record(
        ledger_directory,
        _terminal(plan, claim, ledger_directory),
    )

    with pytest.raises(LedgerValidationError, match="without a claim"):
        validate_attempt_ledger(ledger_directory, plan)


def test_duplicate_attempt_identity_is_rejected_before_second_claim(tmp_path):
    plan = _plan()
    ledger_directory = _ledger(tmp_path, plan)
    claim = _claim(plan)
    publish_ledger_record(ledger_directory, plan, claim)
    publish_ledger_record(
        ledger_directory,
        plan,
        _terminal(plan, claim, ledger_directory),
    )

    with pytest.raises(LedgerValidationError, match="already claimed"):
        publish_ledger_record(
            ledger_directory,
            plan,
            _claim(plan, ordinal=1, attempt_id=claim.attempt_id),
        )


def test_existing_ledger_rejects_collection_plan_provenance_drift(tmp_path):
    plan = _plan()
    ledger_directory = _ledger(tmp_path, plan)
    publish_ledger_record(ledger_directory, plan, _claim(plan))
    changed_plan = _plan(backend_identity="different-backend")

    with pytest.raises(LedgerRecordConflictError, match="different Collection Plan"):
        validate_attempt_ledger(ledger_directory, changed_plan)


def test_target_success_stops_every_future_claim(tmp_path):
    plan = _plan(target_canonical_success_count=1)
    ledger_directory = _ledger(tmp_path, plan)
    claim = _claim(plan)
    publish_ledger_record(ledger_directory, plan, claim)
    publish_ledger_record(
        ledger_directory,
        plan,
        _terminal(
            plan,
            claim,
            ledger_directory,
            outcome=TerminalOutcome.CANONICAL_SUCCESS,
        ),
    )
    state = validate_attempt_ledger(ledger_directory, plan)

    with pytest.raises(CollectionTargetReached):
        next_planned_attempt(plan, state)
    with pytest.raises(CollectionTargetReached):
        publish_ledger_record(
            ledger_directory,
            plan,
            _claim(plan, ordinal=1, attempt_id="attempt-001"),
        )


def test_seed_pool_exhaustion_fails_before_target(tmp_path):
    plan = _plan(
        ordered_seed_pool=(101, 202),
        target_canonical_success_count=2,
    )
    ledger_directory = _ledger(tmp_path, plan)
    for ordinal in range(2):
        claim = _claim(plan, ordinal=ordinal, attempt_id=f"attempt-{ordinal:03d}")
        publish_ledger_record(ledger_directory, plan, claim)
        publish_ledger_record(
            ledger_directory,
            plan,
            _terminal(plan, claim, ledger_directory),
        )

    with pytest.raises(CollectionSeedPoolExhausted):
        next_planned_attempt(
            plan,
            validate_attempt_ledger(ledger_directory, plan),
        )


def test_interrupted_claim_is_closed_once_and_cannot_be_rerun(tmp_path):
    plan = _plan()
    ledger_directory = _ledger(tmp_path, plan)
    claim = _claim(plan)
    publish_ledger_record(ledger_directory, plan, claim)

    first = recover_attempt_ledger(
        ledger_directory,
        plan,
        interruption_timestamp_utc="2026-09-03T01:04:00Z",
    )
    terminal_path = ledger_directory / "0-101.terminal.json"
    terminal_bytes = terminal_path.read_bytes()
    second = recover_attempt_ledger(
        ledger_directory,
        plan,
        interruption_timestamp_utc="2026-09-03T01:05:00Z",
    )

    assert first == second
    assert first.terminals[0].failure_evidence_digest is None
    assert first.terminals[0].partial_evidence == ()
    assert terminal_path.read_bytes() == terminal_bytes
    with pytest.raises(LedgerRecordConflictError):
        publish_ledger_record(ledger_directory, plan, claim)


@pytest.mark.parametrize("unsupported_operation", ["hard_link", "file_fsync"])
def test_filesystem_without_required_durability_fails_preflight(
    tmp_path,
    monkeypatch,
    unsupported_operation,
):
    if unsupported_operation == "hard_link":
        def unsupported_link(*args, **kwargs):
            raise OSError(errno.EOPNOTSUPP, "hard links unsupported")

        monkeypatch.setattr(ledger_module.os, "link", unsupported_link)
    else:
        def unsupported_fsync(descriptor):
            raise OSError(errno.EOPNOTSUPP, "fsync unsupported")

        monkeypatch.setattr(ledger_module.os, "fsync", unsupported_fsync)

    with pytest.raises(LedgerDurabilityUnsupportedError):
        preflight_attempt_ledger(tmp_path / "attempt_ledger")


def test_ledger_initialization_durably_binds_the_exact_collection_plan(tmp_path):
    plan = _plan()

    ledger_directory = initialize_attempt_ledger(tmp_path, plan)

    assert ledger_directory == tmp_path / "attempt_ledger"
    assert (tmp_path / "collection_plan.json").read_bytes() == (
        canonical_json_bytes(plan.to_record())
    )
    with pytest.raises(LedgerRecordConflictError):
        initialize_attempt_ledger(
            tmp_path,
            _plan(backend_identity="provenance-drift"),
        )


def test_terminal_publication_rejects_nonexistent_failure_evidence(tmp_path):
    plan = _plan()
    ledger_directory = initialize_attempt_ledger(tmp_path, plan)
    claim = _claim(plan)
    publish_ledger_record(ledger_directory, plan, claim)

    with pytest.raises(LedgerValidationError, match="failure evidence"):
        publish_ledger_record(
            ledger_directory,
            plan,
            construct_attempt_terminal(
                plan,
                claim,
                outcome=TerminalOutcome.CANONICAL_FAILURE,
                terminal_timestamp_utc="2026-09-03T01:03:00Z",
                failure_evidence_digest="b" * 64,
            ),
        )


def test_ledger_validation_rejects_changed_referenced_failure_evidence(tmp_path):
    plan = _plan()
    ledger_directory = initialize_attempt_ledger(tmp_path, plan)
    claim = _claim(plan)
    publish_ledger_record(ledger_directory, plan, claim)
    terminal = _terminal(plan, claim, ledger_directory)
    publish_ledger_record(ledger_directory, plan, terminal)

    failure_path = (
        tmp_path / "attempts" / claim.attempt_id / "failure_evidence.json"
    )
    failure_path.write_bytes(b"changed failure evidence")

    with pytest.raises(LedgerValidationError, match="failure evidence digest"):
        validate_attempt_ledger(ledger_directory, plan)


def test_ledger_validation_rejects_changed_referenced_game_bundle(tmp_path):
    plan = _plan(target_canonical_success_count=1)
    ledger_directory = initialize_attempt_ledger(tmp_path, plan)
    claim = _claim(plan)
    publish_ledger_record(ledger_directory, plan, claim)
    terminal = _terminal(
        plan,
        claim,
        ledger_directory,
        outcome=TerminalOutcome.CANONICAL_SUCCESS,
    )
    publish_ledger_record(ledger_directory, plan, terminal)

    bundle_payload = tmp_path / "games" / "game-000" / "public" / "fixture.json"
    bundle_payload.write_bytes(b"changed bundle")

    with pytest.raises(LedgerValidationError, match="bundle is missing or invalid"):
        validate_attempt_ledger(ledger_directory, plan)


def test_recovery_derives_all_partial_evidence_instead_of_trusting_digests(
    tmp_path,
):
    plan = _plan()
    ledger_directory = initialize_attempt_ledger(tmp_path, plan)
    claim = _claim(plan)
    publish_ledger_record(ledger_directory, plan, claim)
    partial_directory = (
        tmp_path / "attempts" / claim.attempt_id / "partial_evidence"
    )
    partial_directory.mkdir(parents=True)
    (partial_directory / "calls.jsonl").write_bytes(b"recoverable-calls")
    (partial_directory / "runtime.json").write_bytes(b"recoverable-runtime")

    state = recover_attempt_ledger(
        ledger_directory,
        plan,
        interruption_timestamp_utc="2026-09-03T01:04:00Z",
    )

    assert tuple(
        (reference.relative_path, reference.sha256)
        for reference in state.terminals[0].partial_evidence
    ) == (
        (
            "attempts/attempt-000/partial_evidence/calls.jsonl",
            sha256_bytes(b"recoverable-calls"),
        ),
        (
            "attempts/attempt-000/partial_evidence/runtime.json",
            sha256_bytes(b"recoverable-runtime"),
        ),
    )


def test_recovery_rejects_symlinked_attempt_evidence_ancestor(tmp_path):
    plan = _plan()
    ledger_directory = initialize_attempt_ledger(tmp_path, plan)
    claim = _claim(plan)
    publish_ledger_record(ledger_directory, plan, claim)
    outside = tmp_path / "outside"
    outside.mkdir()
    attempts = tmp_path / "attempts"
    attempts.symlink_to(outside, target_is_directory=True)

    with pytest.raises(LedgerValidationError, match="attempts directory"):
        recover_attempt_ledger(
            ledger_directory,
            plan,
            interruption_timestamp_utc="2026-09-03T01:04:00Z",
        )


def test_evidence_directory_tree_is_durable_before_terminal_publish(
    tmp_path,
    monkeypatch,
):
    plan = _plan()
    ledger_directory = initialize_attempt_ledger(tmp_path, plan)
    claim = _claim(plan)
    publish_ledger_record(ledger_directory, plan, claim)
    nested_directory = (
        tmp_path
        / "attempts"
        / claim.attempt_id
        / "partial_evidence"
        / "nested"
    )
    nested_directory.mkdir(parents=True)
    (nested_directory / "calls.jsonl").write_bytes(b"recoverable-calls")
    synced_directories = set()
    real_fsync_directory = ledger_module._fsync_directory
    real_link = ledger_module.os.link

    def observe_fsync(path):
        synced_directories.add(path)
        real_fsync_directory(path)

    def assert_barrier_before_terminal_link(source, destination, **kwargs):
        if destination.name.endswith(".terminal.json"):
            assert {
                nested_directory,
                nested_directory.parent,
                nested_directory.parent.parent,
                tmp_path / "attempts",
                tmp_path,
            }.issubset(synced_directories)
        return real_link(source, destination, **kwargs)

    monkeypatch.setattr(ledger_module, "_fsync_directory", observe_fsync)
    monkeypatch.setattr(ledger_module.os, "link", assert_barrier_before_terminal_link)

    recover_attempt_ledger(
        ledger_directory,
        plan,
        interruption_timestamp_utc="2026-09-03T01:04:00Z",
    )


def test_existing_collection_plan_reuse_completes_parent_durability_barrier(
    tmp_path,
    monkeypatch,
):
    plan = _plan()
    initialize_attempt_ledger(tmp_path, plan)
    synced_directories = []
    real_fsync_directory = ledger_module._fsync_directory

    def observe_fsync(path):
        synced_directories.append(path)
        real_fsync_directory(path)

    monkeypatch.setattr(ledger_module, "_fsync_directory", observe_fsync)

    initialize_attempt_ledger(tmp_path, plan)

    assert tmp_path in synced_directories


def test_preflight_exercises_the_cross_directory_hard_link(tmp_path, monkeypatch):
    real_link = ledger_module.os.link

    def reject_only_cross_directory(source, destination, **kwargs):
        if destination.parent.name == "attempt_ledger":
            raise OSError(errno.EOPNOTSUPP, "cross-directory link unsupported")
        return real_link(source, destination, **kwargs)

    monkeypatch.setattr(ledger_module.os, "link", reject_only_cross_directory)

    with pytest.raises(LedgerDurabilityUnsupportedError):
        preflight_attempt_ledger(tmp_path / "attempt_ledger")
