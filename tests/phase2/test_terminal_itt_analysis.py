"""ITT joins/recovery with immutable synthetic evidence and real recorded losses.

Only artifact-level constants are rebound to temporary test artifact digests.
The production digest pins are unchanged. No gameplay or model calls occur.
"""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import phase2_terminal_itt_analysis as itt
from werewolf.artifact_io import canonical_json_bytes, canonical_jsonl_bytes, publish_artifact
from werewolf.canonical_collection.attempt_ledger import construct_attempt_claim, construct_collection_plan
from werewolf.canonical_collection.failure_evidence import construct_canonical_partial_evidence
from werewolf.canonical_collection.pre import construct_authoritative_pre_prefix
from werewolf.canonical_collection.public_history import PLAYER_IDS, freeze_public_event_history
from werewolf.phase2_offline import ReferenceTables, ValueEstimate, ValueTable


FIXTURE = Path(__file__).parent / "fixtures/phase2-terminal-itt-recorded-arm-losses.json"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def tree_bytes(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    observed = json.loads(FIXTURE.read_bytes())
    assert observed["source_formal_manifest_digest"] == itt.FORMAL_DIGEST
    assert len(observed["rows"]) == 119
    formal, work = tmp_path / itt.FORMAL_NAME, tmp_path / "work"
    assignments, executions, consequences = [], [], []
    event_start = [
        {"event_id": "event-000000", "event_index": 0, "event_type": "phase_change", "day": 0, "phase": "night"},
        {"event_id": "event-000001", "event_index": 1, "event_type": "death_announcement", "dead_players": []},
        {"event_id": "event-000002", "event_index": 2, "event_type": "phase_change", "day": 1, "phase": "discussion"},
        {"event_id": "event-000003", "event_index": 3, "event_type": "turn_start", "speaker": "player2"}]
    boundary = itt.FAILED_GAME + "-pre-000003"
    prefix = construct_authoritative_pre_prefix(game_id=itt.FAILED_GAME, boundary_id=boundary,
        step_index=3, report_trigger_id=boundary + "-trigger", current_speaker="player2",
        alive_observer_ids=PLAYER_IDS, public_event_history=freeze_public_event_history(event_start),
        v1_annotations=(), belief_observation_ids_by_observer={p: boundary + "-" + p for p in PLAYER_IDS})
    values = ReferenceTables({"development": 0}, {}, ValueTable(None, 1, {
        (2, 4): ValueEstimate(.765, "empirical", None, 1, 200, 100, 153, .765)}))
    monkeypatch.setattr(itt, "REFERENCE_DIGEST", itt.reference_tables_digest(values))
    for source in [*observed["rows"], {"game_id": itt.FAILED_GAME, "assigned_action": "PUSH", "L_ref": .235}]:
        failed = source["game_id"] == itt.FAILED_GAME
        assignment = {"assigned_action": source["assigned_action"], "assignment_probability": .5,
            "treatment": {"assignment_source": "randomized_pilot", "requested_action": source["assigned_action"]},
            "opportunity": {"identity": {"game_id": source["game_id"], "acting_wolf": "player2",
                "candidate_j": "player4", "phase": "speech", "boundary_id": boundary,
                "prefix_digest": prefix.prefix_digest}, "s_pre": [2, 5],
                "public_legal": {"known_wolves": ["player2", "player3"], "alive": list(PLAYER_IDS)},
                "evidence": {"p_tilde_j": .1}}}
        key = itt.digest(assignment)
        execution = {"game_id": source["game_id"], "assignment_digest": key, "success": not failed,
            "failure_reason": "PRIVATE_FACT_CLAIM" if failed else None,
            "canonical_commit_success": not failed, "canonical_event_id": None if failed else "event-000004",
            "canonical_event_digest": None if failed else "a" * 64, "theta_audit_label": None}
        assignments.append(assignment)
        executions.append(execution)
        if not failed:
            consequences.append({"game_id": source["game_id"], "assignment_digest": key,
                "day_consequence": {"Y": "other_nonwolf_exiled", "s_plus": [2, 4],
                    "v_ref": 1 - source["L_ref"], "l_ref": source["L_ref"],
                    "reference_artifact_digest": itt.REFERENCE_DIGEST}})
    seeds = list(range(240))
    seeds[75] = 7714212207331013538
    identities = {k: "synthetic-fixture" for k in ("runtime_identity", "agent_identity", "backend_identity",
        "model_identity", "parser_identity", "prompt_identity", "retry_policy_identity", "call_budget_identity",
        "public_event_schema_version", "pre_prefix_schema_version", "belief_observation_schema_version",
        "v1_annotation_schema_version", "bundle_schema_version")}
    game = construct_collection_plan(collection_id=itt.FORMAL_NAME, ordered_seed_pool=seeds,
        target_canonical_success_count=240, source_revision=itt.FORMAL_SOURCE,
        environment_provenance={"scope": "synthetic-test-fixture"}, **identities)
    claim = construct_attempt_claim(plan=game, ordinal=75,
        attempt_id=f"attempt-000075-seed-{seeds[75]}", claim_timestamp_utc="2026-10-03T00:00:00Z")
    write(work / "claims" / f"{itt.FAILED_GAME}.json", {"record_type": "claim", **claim.to_record()})
    events = event_start + [
        {"event_id": "event-000004", "event_index": 4, "event_type": "public_speech", "speaker": "player2", "raw_text": "test"},
        {"event_id": "event-000005", "event_index": 5, "event_type": "phase_change", "day": 1, "phase": "vote"},
        {"event_id": "event-000006", "event_index": 6, "event_type": "vote_result",
         "votes": [{"voter": p, "target": "player6"} for p in PLAYER_IDS]},
        {"event_id": "event-000007", "event_index": 7, "event_type": "exile_result", "exiled_players": ["player6"]},
        {"event_id": "event-000008", "event_index": 8, "event_type": "phase_change", "day": 1, "phase": "night"}]
    evidence_pins = {}
    for stage in ("execution", "game-result"):
        evidence = construct_canonical_partial_evidence(plan=game, claim=claim,
            evidence_id=f"phase2-{stage}", evidence_type="phase2_online_canonical_stage_v1", payload={
                "game_id": itt.FAILED_GAME, "stage": stage,
                "phase2_record": {"assignment": assignments[-1],
                    "execution": {k: v for k, v in executions[-1].items()
                                  if k not in ("game_id", "assignment_digest", "theta_audit_label")},
                    "offline_audit": {"theta_ac": None}, "day_consequence": None},
                "canonical_runtime": {"authoritative_pre_prefixes": [prefix.to_record()],
                    "public_events": event_start if stage == "execution" else events}})
        path = work / "attempts" / claim.attempt_id / "partial_evidence" / f"phase2-{stage}.json"
        write(path, evidence.to_record())
        sha = itt.sha256_bytes(path.read_bytes())
        monkeypatch.setattr(itt, "EXECUTION_SHA" if stage == "execution" else "GAME_RESULT_SHA", sha)
        if stage == "execution":
            evidence_pins[path.relative_to(work).as_posix()] = sha
    bound = {"source_commit": itt.FORMAL_SOURCE, "reference_tables_digest": itt.REFERENCE_DIGEST,
        "canonical_game_plan": game.to_record(), "paths": {"work_directory": str(work),
            "destination": str(formal), "publication": str(tmp_path / "publication"),
            "evaluation_root": str(tmp_path / "evaluation")}}
    bound["inputs_digest"] = itt.digest(bound)
    write(work / "run_inputs.json", bound)
    artifact = publish_artifact(formal, manifest_fields={
        "artifact_type": "phase2_online_terminal_pilot", "schema_version": "phase2_online_terminal_pilot_v1",
        "study_name": itt.FORMAL_NAME, "campaign_purpose": "pilot", "completion_status": "COMPLETE",
        "target_assignment_count": 120, "source_provenance": {"commit": itt.FORMAL_SOURCE},
        "reference_artifact_digest": itt.REFERENCE_DIGEST, "synthetic_test_fixture": True,
        "server_run_provenance": {"work_directory": str(work), "inputs_digest": bound["inputs_digest"],
            "game_plan_digest": game.plan_digest, "canonical_stage_evidence": evidence_pins}}, files={
        "assignments.jsonl": canonical_jsonl_bytes(assignments), "executions.jsonl": canonical_jsonl_bytes(executions),
        "consequences.jsonl": canonical_jsonl_bytes(consequences)})
    monkeypatch.setattr(itt, "FORMAL_DIGEST", artifact.manifest_digest)
    monkeypatch.setattr(itt, "build_development_layer", lambda *_: SimpleNamespace(values=values))
    return SimpleNamespace(formal=formal, work=work, artifact=artifact, game=game, claim=claim,
        assignments=assignments, executions=executions, consequences=consequences, values=values,
        destination=tmp_path / itt.NAME)


def recovery(setup):
    return itt.recover(setup.assignments[-1], setup.executions[-1], setup.artifact,
                       setup.work, setup.game, setup.values)


def test_full_analysis_exact_120_itt_and_real_arm_loss_numerics_no_input_mutation(setup):
    before = tree_bytes(setup.formal), tree_bytes(setup.work)
    result = itt.analyze(setup.formal, setup.destination)
    summary = result["support"]
    assert summary["N_total"] == 120 and summary["action_counts"] == {"PUSH": 58, "REDIRECT": 62}
    assert summary["recovery_count"] == 1 and summary["execution_failure"] == 1
    assert summary["raw_ITT_mean_L_ref"] == pytest.approx(itt.EXPECTED_MEANS, abs=1e-12)
    assert summary["raw_PUSH_minus_REDIRECT"] == pytest.approx(.07223827446680861, abs=1e-12)
    rows = [json.loads(l) for l in (setup.destination / "itt_rows.jsonl").read_bytes().splitlines()]
    assert len(rows) == 120 and len({r["assignment_id"] for r in rows}) == 120
    failed = next(r for r in rows if r["game_id"] == itt.FAILED_GAME)
    assert failed["execution_success"] is False and failed["execution_failure_reason"] == "PRIVATE_FACT_CLAIM"
    assert failed["L_ref"] == pytest.approx(.235)
    assert failed["outcome_source"] == "recovered_frozen_canonical_trajectory"
    by_id = {r["assignment_id"]: r for r in rows}
    for recorded in setup.consequences:
        row = by_id[recorded["assignment_digest"]]
        day = recorded["day_consequence"]
        assert [row[k] for k in ("Y", "S_plus", "V_ref", "L_ref")] == [day[k] for k in ("Y", "s_plus", "v_ref", "l_ref")]
        assert row["outcome_source"] == "formal_consequence"
    assert before == (tree_bytes(setup.formal), tree_bytes(setup.work))
    assert (setup.destination / "manifest.json").is_file()
    with pytest.raises(ValueError, match="already exists"):
        itt.analyze(setup.formal, setup.destination)


def test_formal_manifest_digest_mismatch(setup, monkeypatch):
    monkeypatch.setattr(itt, "FORMAL_DIGEST", "0" * 64)
    with pytest.raises(ValueError, match="manifest digest"):
        itt.load_formal(setup.formal)


@pytest.mark.parametrize("fault", ("duplicate_assignment", "duplicate_execution", "unknown_consequence", "wrong_game", "wrong_missing", "missing_execution"))
def test_identity_and_joins_fail_closed(setup, fault):
    a, e, c = deepcopy((setup.assignments, setup.executions, setup.consequences))
    if fault == "duplicate_assignment":
        a[0] = deepcopy(a[1])
    elif fault == "duplicate_execution":
        e[0] = deepcopy(e[1])
    elif fault == "unknown_consequence":
        c[0]["assignment_digest"] = "f" * 64
    elif fault == "wrong_game":
        e[0]["game_id"] = "wrong"
    elif fault == "wrong_missing":
        c[0]["assignment_digest"] = e[-1]["assignment_digest"]
        c[0]["game_id"] = itt.FAILED_GAME
    else:
        e.pop()
    with pytest.raises(ValueError):
        itt.exact_joins(a, e, c)


@pytest.mark.parametrize("fault", ("candidate", "wolves", "actor", "action", "failure", "commit", "state"))
def test_only_exact_known_failure_can_recover(setup, fault):
    a, e = deepcopy((setup.assignments[-1], setup.executions[-1]))
    if fault == "candidate":
        a["opportunity"]["identity"]["candidate_j"] = "player5"
    elif fault == "wolves":
        a["opportunity"]["public_legal"]["known_wolves"] = ["player2", "player4"]
    elif fault == "actor":
        a["opportunity"]["identity"]["acting_wolf"] = "player3"
    elif fault == "action":
        a["assigned_action"] = "REDIRECT"
    elif fault == "failure":
        e["failure_reason"] = "OTHER"
    elif fault == "commit":
        e["canonical_commit_success"] = True
    else:
        a["opportunity"]["s_pre"] = [2, 4]
    with pytest.raises(ValueError, match="recovery conditions"):
        itt.recover(a, e, setup.artifact, setup.work, setup.game, setup.values)


def test_immutable_evidence_tamper_fails_even_with_recomputed_record_digest(setup):
    path = setup.work / "attempts" / setup.claim.attempt_id / "partial_evidence/phase2-game-result.json"
    row = json.loads(path.read_bytes())
    row["payload"]["canonical_runtime"]["public_events"][7]["exiled_players"] = ["player5"]
    row["record_digest"] = itt.digest({k: v for k, v in row.items() if k != "record_digest"})
    write(path, row)
    with pytest.raises(ValueError, match="SHA mismatch"):
        recovery(setup)


def test_wrong_exile_fails_after_valid_test_evidence_pin(setup, monkeypatch):
    path = setup.work / "attempts" / setup.claim.attempt_id / "partial_evidence/phase2-game-result.json"
    row = json.loads(path.read_bytes())
    row["payload"]["canonical_runtime"]["public_events"][7]["exiled_players"] = ["player5"]
    row["record_digest"] = itt.digest({k: v for k, v in row.items() if k != "record_digest"})
    write(path, row)
    monkeypatch.setattr(itt, "GAME_RESULT_SHA", itt.sha256_bytes(path.read_bytes()))
    with pytest.raises(ValueError, match="wrong exile"):
        recovery(setup)


def test_wrong_reference_digest_and_mapping_fail(setup, monkeypatch):
    table = setup.values.v_ref_pub_full
    cell = replace(table.cells[(2, 4)], value=.5, empirical_value=.5)
    wrong = replace(setup.values, v_ref_pub_full=replace(table, cells={(2, 4): cell}))
    with pytest.raises(ValueError, match="reference table digest"):
        itt.recover(setup.assignments[-1], setup.executions[-1], setup.artifact, setup.work, setup.game, wrong)
    monkeypatch.setattr(itt, "REFERENCE_DIGEST", itt.reference_tables_digest(wrong))
    with pytest.raises(ValueError, match="reference mapping"):
        itt.recover(setup.assignments[-1], setup.executions[-1], setup.artifact, setup.work, setup.game, wrong)


def test_numeric_disagreement_fails_without_adjusting_data(setup):
    indexed, executions, consequences = itt.exact_joins(setup.assignments, setup.executions, setup.consequences)
    day, _ = recovery(setup)
    rows = itt.itt_rows(indexed, executions, consequences, day)
    rows[0]["L_ref"] += .01
    with pytest.raises(ValueError, match="numerical support mismatch"):
        itt.support(rows)


def test_output_cannot_be_inside_frozen_evidence(setup):
    for protected in (setup.formal, setup.work):
        with pytest.raises(ValueError, match="disjoint"):
            itt.analyze(setup.formal, protected / itt.NAME)
