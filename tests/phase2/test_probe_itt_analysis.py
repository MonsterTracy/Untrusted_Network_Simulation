"""Fictional immutable formal artifacts: no gameplay or empirical Probe effect.

The full 200-row fixture uses genuine typed assignment reconstruction and real
canonical PRE constructors. Generic artifact publication here certifies fixture
integrity only, never a real production experiment.
"""

from copy import deepcopy
from dataclasses import replace
import inspect
import math
from pathlib import Path
import statistics
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import phase2_probe_itt_analysis as analysis
from tests.phase2.test_probe_policy_ledger import literal_snapshots
from tests.phase2.test_probe_policy_plan import public_opportunity
from tests.phase2.test_probe_policy_runner import canonical_pre
from werewolf.artifact_io import canonical_json_bytes, canonical_jsonl_bytes, publish_artifact, read_artifact_file
from werewolf.phase2_actions import Action, context_from_pre
from werewolf.phase2_online_plan import Phase2OnlineProbePilotPlanV1, assign_probe_strategy, select_probe_candidate
from werewolf.phase2_online_records import Phase2OnlineProbeRecordV1, Phase2StrategyStageExecutionV1
from werewolf.phase2_pilot_dataset import PROBE_ONLINE_VERSION, analyze_phase2_online_support_record
from werewolf.phase2_treatment import build_phase2_treatment


def successful_probe_trace(assignment):
    record = Phase2OnlineProbeRecordV1(assignment)
    trace = [record.to_record()]

    def advance(event, **fields):
        nonlocal record
        record = replace(record, lifecycle=(*record.lifecycle, event), **fields)
        trace.append(record.to_record())

    def execution(opportunity, treatment, stage):
        return Phase2StrategyStageExecutionV1(opportunity, treatment, "1" * 64, 1, (),
            canonical_link=SimpleNamespace(canonical_event_id=f"{stage}-speech",
                                           canonical_event_digest="2" * 64, public_text_digest="3" * 64))

    advance("T1_ATTEMPTED")
    advance("T1_COMMITTED", t1_execution=execution(assignment.opportunity, assignment.treatment, "T1"))
    advance("T3_SCHEDULED")
    from werewolf.phase2_pilot_records import PublicProbeObservationV1
    record = replace(record, observations=(PublicProbeObservationV1("reply-speech", "player2", "我怀疑 player6。"),))
    advance("T3_REACHED")
    initial = assignment.opportunity
    context = context_from_pre(canonical_pre(phase=initial.legal_context.phase, actor="player5",
                                            game_id=initial.legal_context.game_id, boundary="pre-3"),
                               initial.legal_context.known_wolves)
    opportunity = replace(initial, legal_context=context, current_position=2,
                          remaining_speakers=len(context.public_speaker_queue) - 3,
                          legal_actions=(Action.PUSH, Action.REDIRECT),
                          probe_request_type=None, continuation_actor=None, observation_window=None)
    treatment = build_phase2_treatment(opportunity, Action.REDIRECT, assignment_source="strategy_continuation",
        assignment_probability=1, randomization_key=analysis.digest([
            "phase2_probe_continuation_v1", assignment.digest(), opportunity.digest()]))
    advance("T3_PREPARED", t3_opportunity=opportunity, t3_treatment=treatment)
    advance("T3_COMMITTED", t3_execution=execution(opportunity, treatment, "T3"))
    advance("EXECUTION_RECORDED")
    # Remaining consequence/game snapshots are attached as literal serialized
    # values below, so their synthetic loss is visibly independent of gameplay.
    return trace


def fixture_payload():
    # This is fictional planning metadata, not a newly frozen formal cap.
    plan = Phase2OnlineProbePilotPlanV1(analysis.FORMAL_NAME, analysis.FORMAL_SEED, 200, 400,
                                       campaign_purpose="pilot")
    assignments, executions, consequences, snapshots = [], [], [], []
    for number in range(200):
        phase = "speech_pk" if number % 5 == 0 else "speech"
        game_id = f"fictional-formal-game-{number:03d}"
        prefix = canonical_pre(phase=phase, game_id=game_id)
        opportunity = replace(public_opportunity(phase=phase, game_id=game_id),
                              legal_context=context_from_pre(prefix, frozenset(("player1", "player5"))))
        assigned = assign_probe_strategy(plan, select_probe_candidate(plan, opportunity.legal_context), opportunity)
        assignments.append(assigned.to_record())
        # Both valid and invalid policies remain randomized ITT members.
        probe = assigned.strategy.value == "PROBE_THEN_REDIRECT"
        invalid = number % 3 == 0
        trace = successful_probe_trace(assigned) if probe and not invalid else literal_snapshots(assigned, invalid=invalid)
        loss = .8 if invalid else .2
        day = {"exiled_player": "player2", "Y": "target_j_exiled", "s_plus": [2, 2],
               "v_ref": 1 - loss, "l_ref": loss, "reference_artifact_digest": "4" * 64}
        if probe and not invalid:
            raw = deepcopy(trace[-1])
            raw["lifecycle"].append("DAY_CONSEQUENCE_RECORDED")
            raw["day_consequence"] = deepcopy(day)
            trace.append(deepcopy(raw))
            raw["lifecycle"].append("GAME_RESULT_RECORDED")
            raw["offline_audit"]["final_game_result"] = "Werewolf"
            trace.append(raw)
        for raw in trace:
            if raw["day_consequence"] is not None:
                raw["day_consequence"] = deepcopy(day)
            snapshots.append({"game_id": game_id, "assignment_id": assigned.digest(), "record": raw})
        last = trace[-1]
        executions.append({"game_id": game_id, "assignment_digest": assigned.digest(),
                           "theta_audit_label": None, **last["execution"]})
        consequences.append({"game_id": game_id, "assignment_digest": assigned.digest(),
                             "theta_audit_label": None, "day_consequence": day,
                             "final_game_result": "Werewolf"})
    tables = [assignments, executions, consequences, [], snapshots]
    m = {"schema_version": PROBE_ONLINE_VERSION, "source_commit": "a" * 40,
         "pilot_plan_digest": plan.digest(), "probe_plan": plan.to_record(),
         "campaign_purpose": "pilot", "estimator_eligible": False,
         "target_assignment_count": 200, "campaign_status": "READY_TO_SEAL", "sealable": True,
         "assignment_semantics": "online_randomized_whole_probe_strategy",
         "observational_vote_intent_used": False, "synthetic_audit_only": False,
         "games_seen": 200, "eligible_opportunities": 200, "assignment_count": 200,
         "execution_count": 200, "consequence_count": 200, "missing_execution_count": 0,
         "successful_execution_without_day_result": 0, "ledger_digest": "b" * 64,
         "tables_digest": analysis.digest(tables), "strategy_stage_count": len(snapshots),
         "itt_assignment_count": 200, "itt_consequence_count": 200, "itt_outcome_complete": True}
    m["manifest_digest"] = analysis.digest(m)
    dataset = {"manifest": m, **dict(zip(("assignments", "executions", "consequences",
                                         "backend_calls", "strategy_stages"), tables))}
    support = analyze_phase2_online_support_record(dataset)
    fields = {"artifact_type": "phase2_online_probe_pilot", "schema_version": "phase2_online_probe_pilot_v1",
              "study_name": analysis.FORMAL_NAME, "campaign_purpose": "pilot",
              "target_assignment_count": 200, "completion_status": "COMPLETE",
              "ledger_digest": "b" * 64, "pilot_plan_digest": plan.digest(),
              "q_source_digest": "c" * 64, "mapper_artifact_digest": "e" * 64,
              "reference_artifact_digest": "4" * 64, "smoke_v3_manifest_digest": "5" * 64,
              "source_provenance": {"commit": "a" * 40, "branch": "twd/mainline",
                                    "tracked_worktree_clean": True, "staged_tracked_changes": False,
                                    "source_sha256": {"werewolf/phase2_online_runner.py": "6" * 64}},
              "server_run_provenance": {"inputs_digest": "7" * 64, "game_plan_digest": "8" * 64,
                                        "probe_policy_protocol_sha256": analysis.PROTOCOL_DIGEST,
                                        "probe_campaign_profile_digest": "9" * 64,
                                        "work_directory": "/nonexistent/frozen-formal-work"}}
    files = {"pilot_plan.json": canonical_json_bytes(plan.to_record()),
             **{name: canonical_jsonl_bytes(rows) for name, rows in zip(analysis.TABLE_FILES, tables)},
             "metrics/support.json": canonical_json_bytes(support), "report.md": b"Synthetic fixture only.\n"}
    return fields, files


@pytest.fixture(scope="module")
def payload():
    return fixture_payload()


@pytest.fixture
def formal(tmp_path, payload):
    fields, files = payload
    return publish_artifact(tmp_path / analysis.FORMAL_NAME, manifest_fields=fields, files=files)


def test_full_formal_join_keeps_all_invalid_and_cancelled_assignments(formal):
    _, rows = analysis.load_formal(formal.path, formal.manifest_digest)
    assert len(rows) == 200
    assert {row["phase"] for row in rows} == {"speech", "speech_pk"}
    probe = [row for row in rows if row["assigned_strategy"] == "PROBE_THEN_REDIRECT"]
    assert any(not row["execution_success"] and row["T3_cancelled"] for row in probe)
    assert any(row["execution_success"] and row["T3_committed"] for row in probe)
    summary = analysis.primary_itt(rows)
    expected_means = {arm: statistics.fmean(row["L_ref"] for row in rows if row["assigned_strategy"] == arm)
                      for arm in analysis.ARMS}
    expected_variances = {arm: statistics.variance(row["L_ref"] for row in rows if row["assigned_strategy"] == arm)
                          for arm in analysis.ARMS}
    expected_se = math.sqrt(sum(expected_variances[arm] / summary["arm_counts"][arm]
                                for arm in analysis.ARMS))
    difference = expected_means["PROBE_THEN_REDIRECT"] - expected_means["IMMEDIATE_REDIRECT"]
    assert summary["arm_means"] == expected_means
    assert summary["arm_sample_variances_ddof1"] == expected_variances
    assert summary["raw_difference"] == difference
    assert summary["neyman_standard_error"] == expected_se
    assert summary["normal_95_percent_ci"] == pytest.approx(
        [difference - 1.959963984540054 * expected_se, difference + 1.959963984540054 * expected_se])
    assert not summary["post_treatment_filter"]
    assert summary["supplementary_fisher_test"] is None
    secondary = analysis.secondary_descriptive(rows)
    assert secondary["PROBE_THEN_REDIRECT"]["T3_cancelled"] == sum(row["T3_cancelled"] for row in probe)
    assert secondary["PROBE_THEN_REDIRECT"]["T3_committed"] == sum(row["T3_committed"] for row in probe)


def test_post_treatment_filtering_has_no_api_or_cli_knob(formal, tmp_path):
    assert "execution_success" not in inspect.signature(analysis.analyze).parameters
    assert "filter" not in inspect.signature(analysis.primary_itt).parameters
    with pytest.raises(SystemExit) as error:
        analysis.main(["--formal", str(formal.path), "--expected-formal-manifest-digest", formal.manifest_digest,
                       "--destination", str(tmp_path / analysis.NAME), "--execution-success-only"])
    assert error.value.code == 2
    _, rows = analysis.load_formal(formal.path, formal.manifest_digest)
    with pytest.raises(ValueError, match="every one of 200"):
        analysis.primary_itt(tuple(row for row in rows if row["execution_success"]))


def test_one_member_arm_keeps_point_estimate_but_has_no_invented_variance_or_ci():
    rows = tuple({"assignment_id": str(i), "assigned_strategy": analysis.ARMS[i == 0],
                  "L_ref": i / 200} for i in range(200))
    summary = analysis.primary_itt(rows)
    assert summary["arm_counts"]["PROBE_THEN_REDIRECT"] == 1
    assert summary["arm_sample_variances_ddof1"]["PROBE_THEN_REDIRECT"] is None
    assert summary["raw_difference"] is not None
    assert summary["neyman_standard_error"] is None
    assert summary["normal_95_percent_ci"] is None


@pytest.mark.parametrize("mutation", [
    "qualification", "incomplete", "wrongtarget", "protocol", "source_dirty", "source_staged",
    "plan_digest", "q_digest", "mapper_digest", "reference_digest", "support",
    "missing_consequence", "duplicate_execution", "foreign_consequence", "altered_assignment",
    "duplicate_snapshot", "backend_payload", "loss_bridge", "theta_label",
])
def test_fail_closed_on_foreign_incomplete_or_tampered_formal_input(tmp_path, payload, mutation):
    fields, files = deepcopy(payload)
    if mutation == "qualification":
        fields.update(campaign_purpose="qualification", study_name="paper-phase2-online-probe-qualification-v2")
    elif mutation == "incomplete":
        fields["completion_status"] = "INCOMPLETE"
    elif mutation == "wrongtarget":
        fields["target_assignment_count"] = 20
    elif mutation == "protocol":
        fields["server_run_provenance"]["probe_policy_protocol_sha256"] = "0" * 64
    elif mutation in ("source_dirty", "source_staged"):
        fields["source_provenance"]["tracked_worktree_clean" if mutation == "source_dirty" else "staged_tracked_changes"] = mutation != "source_dirty"
    elif mutation in ("plan_digest", "q_digest", "mapper_digest", "reference_digest"):
        fields[{"plan_digest": "pilot_plan_digest", "q_digest": "q_source_digest",
                "mapper_digest": "mapper_artifact_digest", "reference_digest": "reference_artifact_digest"}[mutation]] = "0" * 64
    elif mutation == "support":
        support = analysis._strict_json(files["metrics/support.json"])
        support["games_assigned"] -= 1
        files["metrics/support.json"] = canonical_json_bytes(support)
    else:
        name = {"missing_consequence": "consequences.jsonl", "duplicate_execution": "executions.jsonl",
                "foreign_consequence": "consequences.jsonl", "altered_assignment": "assignments.jsonl",
                "duplicate_snapshot": "strategy_stages.jsonl", "backend_payload": "backend_calls.jsonl",
                "loss_bridge": "consequences.jsonl", "theta_label": "executions.jsonl"}[mutation]
        table = [analysis._strict_json(line) for line in files[name].splitlines()]
        if mutation == "missing_consequence":
            table.pop()
        elif mutation in ("duplicate_execution", "duplicate_snapshot"):
            table[-1] = deepcopy(table[0])
        elif mutation == "foreign_consequence":
            table[0]["game_id"] = "foreign-game"
        elif mutation == "altered_assignment":
            table[0]["assignment_seed"] += 1
        elif mutation == "backend_payload":
            table.append({"assignment_id": "x"})
        elif mutation == "theta_label":
            table[0]["theta_audit_label"] = True
        else:
            table[0]["day_consequence"]["v_ref"] = .11
        files[name] = canonical_jsonl_bytes(table)
    artifact = publish_artifact(tmp_path / f"input-{mutation}", manifest_fields=fields, files=files)
    with pytest.raises((ValueError, KeyError, TypeError)):
        analysis.load_formal(artifact.path, artifact.manifest_digest)


def test_expected_digest_is_mandatory_and_exact(formal):
    for value in (None, "", "0" * 64):
        with pytest.raises(ValueError, match="digest"):
            analysis.load_formal(formal.path, value)


def test_deterministic_exclusive_publication_freezes_source_once(formal, tmp_path, monkeypatch):
    captured = analysis.source_provenance()
    count = []
    monkeypatch.setattr(analysis, "source_provenance", lambda: count.append(1) or deepcopy(captured))
    outputs = [tmp_path / name / analysis.NAME for name in ("first", "second")]
    before = {name: read_artifact_file(formal, name) for name in formal.manifest["file_table"]}
    result = [analysis.analyze(formal.path, output, expected_manifest_digest=formal.manifest_digest)
              for output in outputs]
    assert count == [1, 1]
    assert result[0]["manifest_digest"] == result[1]["manifest_digest"]
    assert (outputs[0] / "manifest.json").read_bytes() == (outputs[1] / "manifest.json").read_bytes()
    assert before == {name: read_artifact_file(formal, name) for name in before}
    with pytest.raises(ValueError, match="absent"):
        analysis.analyze(formal.path, outputs[0], expected_manifest_digest=formal.manifest_digest)
    with pytest.raises(ValueError, match="disjoint"):
        analysis.analyze(formal.path, formal.path / analysis.NAME,
                         expected_manifest_digest=formal.manifest_digest)


def test_source_drift_prevents_publication(formal, tmp_path, monkeypatch):
    frozen = analysis.source_provenance()
    monkeypatch.setattr(analysis, "source_provenance", lambda: frozen)
    monkeypatch.setattr(analysis, "_source_files", lambda: {})
    destination = tmp_path / analysis.NAME
    with pytest.raises(ValueError, match="source changed"):
        analysis.analyze(formal.path, destination, expected_manifest_digest=formal.manifest_digest)
    assert not destination.exists()


def test_analysis_import_has_no_gameplay_model_or_server_assembly():
    result = subprocess.run([sys.executable, "-c", "import sys; import scripts.phase2_probe_itt_analysis; "
        "assert not any(name in sys.modules for name in ('run_random', 'werewolf.phase2_online_runner', "
        "'werewolf.phase2_online_server', 'transformers', 'torch', 'openai', 'gymnasium'))"],
        cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
