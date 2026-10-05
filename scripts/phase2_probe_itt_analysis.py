"""All-assignment Probe Policy V1 ITT analysis of one completed formal artifact.

No gameplay, inference, fitting, outcome recovery, or post-treatment selection.
Qualification artifacts cannot enter this independent analysis publication.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys

from werewolf.artifact_io import (
    canonical_json_bytes, canonical_jsonl_bytes, publish_artifact,
    read_artifact_file, sha256_bytes, verify_artifact,
)
from werewolf.phase2_offline import _transition, classify_outcome
from werewolf.phase2_online_plan import Phase2OnlineProbePilotPlanV1, probe_assignment_from_record
from werewolf.phase2_pilot_dataset import PROBE_ONLINE_VERSION, analyze_phase2_online_support_record


FORMAL_NAME = "paper-phase2-online-probe-pilot-v1"
FORMAL_SEED = 6449283966833280907
TARGET = 200
NAME = "paper-phase2-probe-itt-analysis-v1"
VERSION = "phase2_probe_itt_analysis_v1"
PROTOCOL = "docs/research/phase2-probe-policy-protocol-v1.md"
PROTOCOL_DIGEST = "96eb5d35c82f35803dcdb953e611ea0a9e41001e40178cd517f9cde1430e416b"
ARMS = ("IMMEDIATE_REDIRECT", "PROBE_THEN_REDIRECT")
Z975 = 1.959963984540054
ROOT = Path(__file__).resolve().parents[1]
TABLE_FILES = ("assignments.jsonl", "executions.jsonl", "consequences.jsonl",
               "backend_calls.jsonl", "strategy_stages.jsonl")
SOURCE_FILES = (
    "scripts/phase2_probe_itt_analysis.py", "werewolf/artifact_io/canonical.py",
    "werewolf/phase2_online_plan.py", "werewolf/phase2_online_records.py",
    "werewolf/phase2_pilot_dataset.py", "werewolf/phase2_offline.py",
    "werewolf/phase2_actions.py", "werewolf/phase2_decision_opportunity.py",
    "werewolf/phase2_treatment.py", "werewolf/phase2_pilot_records.py",
    "werewolf/phase2_outcome.py", "werewolf/phase2_backend_audit.py",
    "werewolf/canonical_collection/pre.py", "werewolf/canonical_collection/public_history.py",
    "werewolf/development_publication.py", PROTOCOL,
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return sha256_bytes(canonical_json_bytes(value))


def _sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _strict_json(data):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"non-finite JSON value: {value}")

    return json.loads(data, object_pairs_hook=unique, parse_constant=invalid)


def _json(artifact, name):
    data = read_artifact_file(artifact, name)
    value = _strict_json(data)
    require(canonical_json_bytes(value) == data, f"noncanonical {name}")
    return value


def _jsonl(artifact, name):
    data = read_artifact_file(artifact, name)
    rows = [_strict_json(line) for line in data.splitlines()]
    require(canonical_jsonl_bytes(rows) == data, f"noncanonical {name}")
    return rows


def _source_files():
    return {name: sha256_bytes((ROOT / name).read_bytes()) for name in SOURCE_FILES}


def source_provenance():
    """Freeze once before input analysis, with no clock or output-directory field."""
    files = _source_files()
    require(files[PROTOCOL] == PROTOCOL_DIGEST, "frozen Probe protocol digest mismatch")

    def git(*args):
        return subprocess.check_output(["git", "-C", str(ROOT), *args]).decode().strip()

    return {"commit": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
            "files": files,
            "tracked_source_diff_digest": sha256_bytes(git(
                "diff", "HEAD", "--binary", "--", *SOURCE_FILES).encode()),
            "environment": {"python": platform.python_version(), "system": platform.system(),
                            "machine": platform.machine()}}


def load_formal(path, expected_manifest_digest):
    """Read only sealed formal payloads; historical work paths stay metadata."""
    require(_sha256(expected_manifest_digest), "expected formal manifest digest is required")
    artifact = verify_artifact(path, expected_artifact_type="phase2_online_probe_pilot",
                               expected_schema_version="phase2_online_probe_pilot_v1")
    m = artifact.manifest
    require(artifact.manifest_digest == expected_manifest_digest, "formal manifest digest mismatch")
    require(m["study_name"] == FORMAL_NAME and m["campaign_purpose"] == "pilot"
            and m["completion_status"] == "COMPLETE" and m["target_assignment_count"] == TARGET,
            "only completed 200-assignment formal Probe input is allowed")
    source = m["source_provenance"]
    require(isinstance(source, dict) and isinstance(source.get("commit"), str)
            and len(source["commit"]) == 40
            and all(c in "0123456789abcdef" for c in source["commit"])
            and source.get("tracked_worktree_clean") is True
            and source.get("staged_tracked_changes") is False
            and isinstance(source.get("source_sha256"), dict) and source["source_sha256"]
            and all(_sha256(value) for value in source["source_sha256"].values()),
            "formal execution source provenance is not clean and pinned")
    run = m["server_run_provenance"]
    require(isinstance(run, dict) and run.get("probe_policy_protocol_sha256") == PROTOCOL_DIGEST
            and all(_sha256(run.get(name)) for name in (
                "probe_campaign_profile_digest", "inputs_digest", "game_plan_digest"))
            and all(_sha256(m.get(name)) for name in (
                "ledger_digest", "pilot_plan_digest", "q_source_digest",
                "mapper_artifact_digest", "reference_artifact_digest", "smoke_v3_manifest_digest")),
            "formal protocol/plan/frozen artifact provenance mismatch")
    raw_plan = _json(artifact, "pilot_plan.json")
    plan = Phase2OnlineProbePilotPlanV1(
        raw_plan["pilot_id"], raw_plan["assignment_seed"], raw_plan["target_assignment_count"],
        raw_plan["max_games_attempted"], raw_plan["campaign_purpose"],
        raw_plan["candidate_selection_rule"], raw_plan["selection_status"])
    require(raw_plan == plan.to_record() and plan.digest() == m["pilot_plan_digest"]
            and plan.pilot_id == FORMAL_NAME and plan.campaign_purpose == "pilot"
            and plan.target_assignment_count == TARGET and plan.assignment_seed == FORMAL_SEED,
            "formal randomized policy plan mismatch")
    tables = [_jsonl(artifact, name) for name in TABLE_FILES]
    assignments, executions, consequences, calls, snapshots = tables
    require(len(assignments) == len(executions) == len(consequences) == TARGET,
            "INCOMPLETE ITT: every one of 200 assignments requires execution and outcome")
    sealed_support = _json(artifact, "metrics/support.json")
    require(isinstance(sealed_support, dict) and sealed_support.get("campaign_status") == "READY_TO_SEAL"
            and sealed_support.get("sealable") is True
            and sealed_support.get("itt_outcome_complete") is True,
            "INCOMPLETE ITT: formal support is not complete and sealable")
    # The publication contains support, rather than the intermediate dataset
    # manifest. Reconstruct only its in-memory validation envelope; none of its
    # synthetic digests are presented as an independently sealed source digest.
    manifest = {
        "schema_version": PROBE_ONLINE_VERSION, "source_commit": source["commit"],
        "pilot_plan_digest": plan.digest(), "probe_plan": raw_plan,
        "campaign_purpose": "pilot", "estimator_eligible": False,
        "target_assignment_count": TARGET, "campaign_status": "READY_TO_SEAL", "sealable": True,
        "assignment_semantics": "online_randomized_whole_probe_strategy",
        "observational_vote_intent_used": False, "synthetic_audit_only": False,
        "games_seen": sealed_support["games_seen"],
        "eligible_opportunities": sealed_support["eligible_opportunities"],
        "assignment_count": TARGET, "execution_count": TARGET, "consequence_count": TARGET,
        "missing_execution_count": 0, "successful_execution_without_day_result": 0,
        "ledger_digest": m["ledger_digest"], "tables_digest": digest(tables),
        "strategy_stage_count": len(snapshots), "itt_assignment_count": TARGET,
        "itt_consequence_count": TARGET, "itt_outcome_complete": True,
    }
    manifest["manifest_digest"] = digest(manifest)
    payload = {"manifest": manifest, **dict(zip((
        "assignments", "executions", "consequences", "backend_calls", "strategy_stages"), tables))}
    recomputed = analyze_phase2_online_support_record(payload)
    require(recomputed == sealed_support, "sealed support differs from validated lifecycle tables")
    latest = {snapshot["game_id"]: snapshot["record"] for snapshot in snapshots}
    require(set(latest) == {row["opportunity"]["identity"]["game_id"] for row in assignments}
            and all(raw["lifecycle"][-1] in ("DAY_CONSEQUENCE_RECORDED", "GAME_RESULT_RECORDED")
                    for raw in latest.values()),
            "INCOMPLETE ITT: assigned lifecycle lacks day consequence endpoint")
    expected_calls = [call for assigned in assignments for stage in ("T1", "T3")
                      if latest[assigned["opportunity"]["identity"]["game_id"]]["stages"][stage]
                      for call in latest[assigned["opportunity"]["identity"]["game_id"]]["stages"][stage]["backend_calls"]]
    require(calls == expected_calls, "backend payload differs from complete strategy stage evidence")
    consequences_by_game = {row["game_id"]: row for row in consequences}
    rows = []
    for assigned in assignments:
        typed = probe_assignment_from_record(assigned, plan)
        op, c = typed.opportunity, typed.opportunity.legal_context
        raw = latest[c.game_id]
        consequence = consequences_by_game[c.game_id]
        day = consequence["day_consequence"]
        category = classify_outcome(day["exiled_player"], acting_wolf=c.acting_wolf,
                                    candidate_j=op.candidate_j, wolves=c.known_wolves)
        require(op.q_source_digest == m["q_source_digest"]
                and op.mapper_artifact_digest == m["mapper_artifact_digest"]
                and day["reference_artifact_digest"] == m["reference_artifact_digest"]
                and (day["exiled_player"] is None or day["exiled_player"] in c.alive)
                and day["Y"] == category.value and day["s_plus"] == list(_transition(op.s_pre, category))
                and type(day["v_ref"]) in (int, float) and math.isfinite(day["v_ref"])
                and 0 <= day["v_ref"] <= 1
                and math.isclose(day["l_ref"] + day["v_ref"], 1, rel_tol=0, abs_tol=1e-9),
                "assignment/outcome frozen source or T0 loss semantics differ")
        rows.append({"assignment_id": typed.digest(), "game_id": c.game_id,
                     "assigned_strategy": typed.strategy.value, "L_ref": day["l_ref"],
                     "Y": day["Y"], "S_pre": list(op.s_pre), "candidate_j": op.candidate_j,
                     "phase": c.phase, "execution_success": raw["execution"]["success"],
                     "T1_success": raw["stages"]["T1"]["success"],
                     "T3_reached": "T3_REACHED" in raw["lifecycle"],
                     "T3_committed": "T3_COMMITTED" in raw["lifecycle"],
                     "T3_cancelled": "T3_CANCELLED" in raw["lifecycle"]})
    return artifact, tuple(sorted(rows, key=lambda row: row["assignment_id"]))


def primary_itt(rows):
    """Raw frozen estimand; no filtering or model-selection arguments exist."""
    require(len(rows) == TARGET and len({r["assignment_id"] for r in rows}) == TARGET
            and all(r["assigned_strategy"] in ARMS and type(r["L_ref"]) in (int, float)
                    and math.isfinite(r["L_ref"]) and 0 <= r["L_ref"] <= 1 for r in rows),
            "primary ITT requires every one of 200 unique assignments")
    values = {arm: [row["L_ref"] for row in rows if row["assigned_strategy"] == arm] for arm in ARMS}
    means = {arm: statistics.fmean(v) if v else None for arm, v in values.items()}
    variances = {arm: statistics.variance(v) if len(v) >= 2 else None for arm, v in values.items()}
    difference = means[ARMS[1]] - means[ARMS[0]] if all(values.values()) else None
    eligible_ci = all(len(v) >= 2 for v in values.values())
    se = math.sqrt(sum(variances[arm] / len(values[arm]) for arm in ARMS)) if eligible_ci else None
    return {"primary_population": "all_200_randomized_assignments", "row_count": TARGET,
            "primary_estimand": "E[L_ref | assignment=PROBE_THEN_REDIRECT] - E[L_ref | assignment=IMMEDIATE_REDIRECT]",
            "lower_is_better": True, "post_treatment_filter": False,
            "arm_counts": {arm: len(v) for arm, v in values.items()}, "arm_means": means,
            "arm_sample_variances_ddof1": variances, "raw_difference": difference,
            "neyman_standard_error": se,
            "normal_95_percent_ci": [difference - Z975 * se, difference + Z975 * se] if eligible_ci else None,
            "normal_critical_value": Z975,
            "uncertainty_status": "AVAILABLE" if eligible_ci else "ARM_HAS_FEWER_THAN_TWO_OUTCOMES",
            "supplementary_fisher_test": None,
            "fisher_omission_reason": "Probe Protocol V1 freezes Neyman normal CI; no Monte Carlo test protocol is registered."}


def secondary_descriptive(rows):
    """Process counts only: never feed primary selection or effect estimation."""
    return {arm: {"N": sum(row["assigned_strategy"] == arm for row in rows),
                  **{field: sum(row["assigned_strategy"] == arm and row[field] for row in rows)
                     for field in ("execution_success", "T1_success", "T3_reached", "T3_committed", "T3_cancelled")}}
            for arm in ARMS}


def analyze(formal, destination, *, expected_manifest_digest):
    source = source_provenance()
    artifact, rows = load_formal(formal, expected_manifest_digest)
    destination = Path(destination)
    require(destination.name == NAME, "analysis requires its independent fixed artifact name")
    for protected in (artifact.path, Path(artifact.manifest["server_run_provenance"]["work_directory"])):
        a, b = destination.resolve(), protected.resolve()
        require(a != b and not a.is_relative_to(b) and not b.is_relative_to(a),
                "analysis output must be disjoint from frozen evidence")
    require(not os.path.lexists(destination), "analysis destination must be absent")
    summary = primary_itt(rows)
    secondary = secondary_descriptive(rows)
    again = verify_artifact(formal, expected_artifact_type="phase2_online_probe_pilot",
                            expected_schema_version="phase2_online_probe_pilot_v1")
    require(again.manifest_digest == artifact.manifest_digest, "formal input changed during analysis")
    require(_source_files() == source["files"], "analysis source changed during execution")
    report = ("# Formal Probe policy ITT\n\n"
              "Every randomized assignment is included, including language-invalid and cancelled T3 paths.\n\n"
              f"Arm counts: {summary['arm_counts']}. Arm means L_ref: {summary['arm_means']}.\n\n"
              f"Probe-policy minus Immediate-Redirect: {summary['raw_difference']}; "
              f"Neyman SE: {summary['neyman_standard_error']}; "
              f"normal 95% CI: {summary['normal_95_percent_ci']}.\n\n"
              "Negative differences favor Probe policy. Process counts are secondary descriptive only. "
              "No conditional model, lambda, router, or effect-based selection is fitted. "
              "No supplementary Fisher simulation is run because its simulation protocol is not preregistered.\n")
    m = artifact.manifest
    published = publish_artifact(destination, manifest_fields={
        "artifact_type": "phase2_probe_itt_analysis", "schema_version": VERSION, "study_name": NAME,
        "analysis_only": True, "row_count": TARGET, "primary_population": summary["primary_population"],
        "source_formal_manifest_digest": artifact.manifest_digest,
        "source_formal_commit": m["source_provenance"]["commit"],
        "source_formal_file_table": m["file_table"], "pilot_plan_digest": m["pilot_plan_digest"],
        "source_run_inputs_digest": m["server_run_provenance"]["inputs_digest"],
        "canonical_game_plan_digest": m["server_run_provenance"]["game_plan_digest"],
        "probe_policy_protocol_sha256": PROTOCOL_DIGEST,
        "probe_campaign_profile_digest": m["server_run_provenance"]["probe_campaign_profile_digest"],
        "reference_artifact_digest": m["reference_artifact_digest"], "analysis_source_provenance": source,
    }, files={"itt_rows.jsonl": canonical_jsonl_bytes(rows), "primary_itt.json": canonical_json_bytes(summary),
              "secondary_descriptive.json": canonical_json_bytes(secondary), "report.md": report.encode()})
    return {"destination": str(published.path), "manifest_digest": published.manifest_digest,
            "primary_itt": summary}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal", type=Path, required=True)
    parser.add_argument("--expected-formal-manifest-digest", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(analyze(args.formal, args.destination,
              expected_manifest_digest=args.expected_formal_manifest_digest), sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        print(f"Probe ITT analysis failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
