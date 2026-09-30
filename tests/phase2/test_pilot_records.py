"""Synthetic pilot records test schema behavior, never claim real intervention."""

from dataclasses import replace
import json
import subprocess
import sys

import pytest

from tests.phase2.test_decision_opportunity import opportunity
from tests.phase2.test_intervention_risk import values
from werewolf.artifact_io import canonical_json_bytes, sha256_bytes
from werewolf.phase2_actions import Action
from werewolf.phase2_outcome import extract_phase2_day_outcome
from werewolf.phase2_pilot_dataset import (
    PilotDatasetError, analyze_phase2_consequence_support,
    analyze_phase2_consequence_support_record,
    build_phase2_consequence_dataset,
)
from werewolf.phase2_pilot_records import (
    Phase2ActionConsequenceRecordV1, Phase2ProbeSequenceRecordV1,
    PilotRecordError, PublicProbeObservationV1,
    record_probe_day_outcome, record_probe_execution, record_probe_observation,
    record_probe_reconsideration,
)
from werewolf.phase2_treatment import build_phase2_treatment


def treatment(opp, action):
    return build_phase2_treatment(opp, action, assignment_source="paired_branch",
                                  assignment_probability=1, randomization_key="pilot-x")


def branch_id(opp, action):
    return sha256_bytes(canonical_json_bytes([
        "checkpoint-1", treatment(opp, action).treatment_id, 123]))


def terminal(opp, action):
    return Phase2ActionConsequenceRecordV1(
        opp, treatment(opp, action), "checkpoint-1", branch_id(opp, action), 123,
        "execution-proof-synthetic", "language-audit-synthetic", True,
        extract_phase2_day_outcome(opp, "player2", values()),
        "reference-full-synthetic", False)


def probe(opp):
    return Phase2ProbeSequenceRecordV1(
        opp, treatment(opp, Action.PROBE), "checkpoint-1", branch_id(opp, Action.PROBE), 123,
        "execution-proof-synthetic")


def test_consequence_record_separates_runtime_and_offline_audit():
    opp = opportunity()
    row = terminal(opp, Action.PUSH)
    record = row.to_record()
    assert record["runtime_safe"]["p_tilde_at_pre"] == opp.p_tilde_j
    assert "theta_ac" not in record["runtime_safe"]
    assert record["offline_audit"]["theta_ac"] is False
    assert record["offline_audit"]["Y"] == "target_j_exiled"
    assert record["offline_audit"]["s_plus"] == [2, 4]
    with pytest.raises(PilotRecordError):
        replace(row, language_execution_valid=False)


def test_probe_sequence_has_separate_stages_and_no_vote_leakage():
    opp = opportunity()
    initial = probe(opp)
    assert initial.to_record()["T0"]["request"]["request_type"] == "CURRENT_SUSPICION_BASIS"
    t1 = record_probe_execution(initial, True)
    events = tuple(PublicProbeObservationV1(f"e{n}", speaker, "公开发言")
                   for n, speaker in enumerate(opp.observation_window.expected_speakers))
    t2 = record_probe_observation(t1, reached=True, events=events)
    assert t2.response_opportunity_reached and t2.public_observation_realized
    assert t2.information_gain is None
    t3 = record_probe_reconsideration(t2, reached=True,
        legal_targets=opp.legal_context.legal_targets, q_digest="updated-q-synthetic",
        p_panel=tuple((row.candidate_j, row.p_tilde) for row in opp.evidence_panel),
        actions=opp.legal_actions)
    assert t3.original_j_still_legal
    assert t3.to_record()["T3"]["updated_q_digest"] == "updated-q-synthetic"
    t4 = record_probe_day_outcome(t3, extract_phase2_day_outcome(opp, None, values()))
    assert t4.to_record()["T4"]["Y"] == "no_exile"
    with pytest.raises(PilotRecordError):
        PublicProbeObservationV1("vote-event", "player2", "投票", "vote_result")
    with pytest.raises(PilotRecordError):
        record_probe_observation(initial, reached=True, events=events)


def test_probe_continuation_not_reached_and_no_guaranteed_response():
    opp = opportunity()
    t1 = record_probe_execution(probe(opp), False)
    t2 = record_probe_observation(t1, reached=False, events=())
    t3 = record_probe_reconsideration(t2, reached=False)
    assert not t3.reconsideration_reached
    assert not t3.public_observation_realized
    assert t3.updated_q_digest is None
    with pytest.raises(PilotRecordError):
        replace(t3, information_gain=True)


def test_dataset_requires_execution_verification_and_reports_only_support():
    opp = opportunity()
    p, n, b = terminal(opp, Action.PUSH), terminal(opp, Action.REDIRECT), probe(opp)
    with pytest.raises(PilotDatasetError, match="verified real intervention"):
        build_phase2_consequence_dataset((p, n), (b,), source_commit="abc",
            pilot_artifact_digest="pilot", checkpoint_format="full-pre-v1")
    dataset = build_phase2_consequence_dataset((p, n), (b,), source_commit="abc",
        pilot_artifact_digest="pilot", checkpoint_format="full-pre-v1", allow_synthetic=True)
    assert dataset.manifest["synthetic_audit_only"]
    assert dataset.manifest["observational_vote_intent_used"] is False
    support = analyze_phase2_consequence_support(dataset)
    assert support["action_counts"] == {"PROBE": 1, "PUSH": 1, "REDIRECT": 1}
    assert support["paired_opportunities"]["P_vs_N_vs_B"] == 1
    assert support["candidate_p_tilde"]["PUSH"]["median"] == opp.p_tilde_j
    assert support["terminal_Y_counts"]["('PUSH', 'target_j_exiled')"] == 1
    with pytest.raises(PilotDatasetError):
        build_phase2_consequence_dataset((p, p), (), source_commit="abc",
            pilot_artifact_digest="pilot", checkpoint_format="full-pre-v1", allow_synthetic=True)


def test_serialized_support_rejects_tampering_and_cli_reports(tmp_path):
    opp = opportunity()
    dataset = build_phase2_consequence_dataset((terminal(opp, Action.PUSH),), (),
        source_commit="abc", pilot_artifact_digest="pilot", checkpoint_format="full-pre-v1",
        allow_synthetic=True)
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(dataset.to_record()), encoding="utf-8")
    output = subprocess.check_output((sys.executable,
        "scripts/analyze_phase2_consequence_support.py", str(path)), text=True)
    assert json.loads(output)["action_counts"] == {"PUSH": 1}
    corrupted = dataset.to_record()
    corrupted["terminal"][0]["offline_audit"]["l_ref"] = 0
    with pytest.raises(PilotDatasetError, match="digest mismatch"):
        analyze_phase2_consequence_support_record(corrupted)
