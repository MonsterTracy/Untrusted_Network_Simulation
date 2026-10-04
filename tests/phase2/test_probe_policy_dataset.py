"""Whole-strategy ledger datasets retain failed branches and complete ITT evidence."""

import pytest

from test_probe_policy_ledger import LiteralRecord, literal_snapshots, public_opportunity
from werewolf.artifact_io import canonical_json_bytes, sha256_bytes
from werewolf.phase2_online_ledger import OnlinePilotAssignmentLedgerV1
from werewolf.phase2_online_plan import (
    Phase2OnlineProbePilotPlanV1, ProbeStrategy, assign_probe_strategy, select_probe_candidate,
)
from werewolf.phase2_pilot_dataset import (
    PROBE_ONLINE_VERSION, Phase2OnlineConsequenceDatasetV1, PilotDatasetError,
    analyze_phase2_online_support_record, build_phase2_online_dataset_from_ledger,
    require_online_dataset_access,
)


def test_optional_lifecycle_snapshots_preserve_legacy_dataset_shape():
    terminal = Phase2OnlineConsequenceDatasetV1((), (), (), (), {"source_commit": "9" * 40})
    assert "strategy_stages" not in terminal.to_record()
    snapshot = {"game_id": "game-1", "assignment_id": "a" * 64,
                "record": {"lifecycle": ["ASSIGNED", "T1_ATTEMPTED"]}}
    probe = Phase2OnlineConsequenceDatasetV1(
        (), (), (), (), {"source_commit": "9" * 40}, strategy_stages=(snapshot,))
    assert probe.to_record()["strategy_stages"] == [snapshot]


def completed_ledger(tmp_path, strategy=ProbeStrategy.PROBE_THEN_REDIRECT, *, purpose="qualification",
                     stop_after_execution=False):
    opportunity = public_opportunity()
    for seed in range(100):
        plan = Phase2OnlineProbePilotPlanV1("probe-dataset", seed, 1, 2, campaign_purpose=purpose)
        selection = select_probe_candidate(plan, opportunity.legal_context)
        assignment = assign_probe_strategy(plan, selection, opportunity)
        if assignment.strategy is strategy:
            break
    else:
        raise AssertionError("fixture needs each strategy")
    ledger = OnlinePilotAssignmentLedgerV1(
        tmp_path / "events.jsonl", plan=plan, source_commit="9" * 40)
    ledger.start_game("game-1")
    ledger.persist_assignment(assignment)
    for raw in literal_snapshots(assignment):
        if raw["day_consequence"] is not None:
            raw["day_consequence"]["Y"] = "target_j_exiled"
        record = LiteralRecord(assignment, raw)
        ledger.persist_strategy_stage(record)
        event = raw["lifecycle"][-1]
        if event == "EXECUTION_RECORDED":
            ledger.persist_execution(record)
            if stop_after_execution:
                break
        elif event == "DAY_CONSEQUENCE_RECORDED":
            ledger.persist_consequence(record)
        elif event == "GAME_RESULT_RECORDED":
            ledger.persist_game_result("game-1", "Werewolf")
    return ledger


@pytest.mark.parametrize("strategy", list(ProbeStrategy))
def test_dataset_keeps_language_invalid_regime_outcomes_and_all_lifecycle_snapshots(tmp_path, strategy):
    ledger = completed_ledger(tmp_path, strategy)
    dataset = build_phase2_online_dataset_from_ledger(ledger, allow_synthetic=True)
    support = analyze_phase2_online_support_record(dataset.to_record())

    assert dataset.manifest["schema_version"] == PROBE_ONLINE_VERSION
    assert dataset.executions[0]["success"] is False
    assert len(dataset.consequences) == len(dataset.assignments) == 1
    assert dataset.consequences[0]["day_consequence"]["l_ref"] == 0.4
    assert list(dataset.strategy_stages) == ledger.snapshot()["games"]["game-1"]["STRATEGY_STAGE_HISTORY"]
    assert support["assigned_by_strategy"] == {strategy.value: 1}
    assert support["itt_assignment_count"] == support["itt_consequence_count"] == 1
    assert support["itt_outcome_complete"] is True
    assert support["consequence_missing"] == 0
    assert support["estimator_eligible"] is False


def test_incomplete_outcome_keeps_original_assignment_and_reports_incomplete_itt(tmp_path):
    ledger = completed_ledger(tmp_path, stop_after_execution=True)
    dataset = build_phase2_online_dataset_from_ledger(ledger, allow_synthetic=True)
    support = analyze_phase2_online_support_record(dataset.to_record())
    assert len(dataset.assignments) == 1
    assert dataset.consequences == ()
    assert support["itt_assignment_count"] == 1
    assert support["itt_consequence_count"] == 0
    assert support["itt_outcome_complete"] is False
    assert support["consequence_missing"] == 1
    assert dataset.manifest["sealable"] is False


@pytest.mark.parametrize("purpose", ["qualification", "pilot"])
def test_probe_datasets_are_audit_only_even_with_explicit_complete_formal_budget(tmp_path, purpose):
    ledger = completed_ledger(tmp_path, purpose=purpose)
    dataset = build_phase2_online_dataset_from_ledger(ledger, allow_synthetic=True)
    assert dataset.manifest["target_assignment_count"] == 1
    assert dataset.manifest["estimator_eligible"] is False
    assert require_online_dataset_access((dataset,), audit_only=True).estimator_eligible is False
    with pytest.raises(PilotDatasetError, match="Probe strategy analysis is unsupported"):
        require_online_dataset_access((dataset,))


def test_production_builder_requires_proof_for_both_execution_and_consequence(tmp_path):
    ledger = completed_ledger(tmp_path)
    with pytest.raises(PilotDatasetError, match="canonical execution verifier"):
        build_phase2_online_dataset_from_ledger(ledger)
    verified = []

    def execution_only(game_id, stage):
        verified.append(game_id)
        assert "EXECUTION" in stage and "CONSEQUENCE" in stage
        return len(verified) == 1

    with pytest.raises(PilotDatasetError, match="consequence"):
        build_phase2_online_dataset_from_ledger(ledger, execution_verifier=execution_only)
    assert verified == ["game-1", "game-1"]


def reseal(payload):
    manifest = payload["manifest"]
    manifest["tables_digest"] = sha256_bytes(canonical_json_bytes([
        payload["assignments"], payload["executions"], payload["consequences"],
        payload["backend_calls"], payload.get("strategy_stages", [])]))
    manifest["manifest_digest"] = sha256_bytes(canonical_json_bytes({
        key: value for key, value in manifest.items() if key != "manifest_digest"}))


def test_support_rejects_lifecycle_history_gap_even_after_digest_rewrite(tmp_path):
    payload = build_phase2_online_dataset_from_ledger(
        completed_ledger(tmp_path), allow_synthetic=True).to_record()
    payload["strategy_stages"] = payload["strategy_stages"][1:]
    payload["manifest"]["strategy_stage_count"] -= 1
    reseal(payload)
    with pytest.raises(PilotDatasetError, match="initial assignment"):
        analyze_phase2_online_support_record(payload)


def test_support_rejects_execution_that_differs_from_lifecycle_even_after_digest_rewrite(tmp_path):
    payload = build_phase2_online_dataset_from_ledger(
        completed_ledger(tmp_path), allow_synthetic=True).to_record()
    payload["executions"][0]["failure_reason"] = "ALTERED"
    reseal(payload)
    with pytest.raises(PilotDatasetError, match="lifecycle evidence"):
        analyze_phase2_online_support_record(payload)


def test_support_rejects_false_itt_complete_claim_even_after_digest_rewrite(tmp_path):
    payload = build_phase2_online_dataset_from_ledger(
        completed_ledger(tmp_path, stop_after_execution=True), allow_synthetic=True).to_record()
    payload["manifest"]["itt_outcome_complete"] = True
    reseal(payload)
    with pytest.raises(PilotDatasetError, match="ITT audit counts"):
        analyze_phase2_online_support_record(payload)
