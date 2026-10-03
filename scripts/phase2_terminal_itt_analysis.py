"""Read-only formal Pilot-T ITT audit; publish only to a separate analysis artifact.

The primary population is every randomized assignment, including failed language
execution. This script never runs gameplay, inference, randomization or fitting.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
import sys

from werewolf.artifact_io import (
    canonical_json_bytes, canonical_jsonl_bytes, publish_artifact,
    read_artifact_file, sha256_bytes, verify_artifact,
)
from werewolf.canonical_collection.attempt_ledger import _claim_from_record, collection_plan_from_record
from werewolf.canonical_collection.failure_evidence import validate_canonical_partial_evidence
from werewolf.canonical_collection.game_bundle import _prefix_from_record
from werewolf.canonical_collection.public_history import freeze_public_event_history
from werewolf.phase2_offline import build_development_layer, classify_outcome
from werewolf.phase2_outcome import reference_tables_digest


FORMAL_NAME = "paper-phase2-online-terminal-pilot-v1"
FORMAL_DIGEST = "ec1cc8730aefe8fce57ed83a3c84ca9231ae645ea6c9047cd588f8022d2223c2"
FORMAL_SOURCE = "05c217c98dab74e919e525f76c55ca3a1e12867e"
REFERENCE_DIGEST = "1bc2fa51ea771e90389cb511bc0c61a442d862c872a121d61037aca6308f7c2c"
NAME = "paper-phase2-terminal-itt-analysis-v1"
VERSION = "phase2_terminal_itt_analysis_v1"
RECOVERY_RULE = "phase2_terminal_itt_known_failed_day1_exile_v1"
FAILED_GAME = FORMAL_NAME + "-game-000075-seed-7714212207331013538"
# The game-result file is independently pinned by this analysis rule. Unlike the
# execution file, its SHA is not present in the original formal manifest.
GAME_RESULT_SHA = "680dd0198213502d805d51687306a97dd8acc9fd046e7759a2a3d17894d6b5f8"
EXECUTION_SHA = "887c613fc1eefb7747bce45d6d4bb55e3e1634c0080057b7123a9d5447a33829"
EXPECTED_MEANS = {"PUSH": 0.31184333758288774, "REDIRECT": 0.23960506311607913}
EXPECTED_DIFFERENCE = 0.07223827446680861
ROOT = Path("/data/yuxiao/Untrusted_Network_Simulation/paper-studies")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def strict_json(data):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError(f"non-finite JSON value: {value}")
    return json.loads(data, object_pairs_hook=unique, parse_constant=invalid)


def read_json(path):
    path = Path(path)
    require(not any(p.is_symlink() for p in (path, *path.parents)), "input symlink prohibited")
    return strict_json(path.read_bytes())


def digest(value):
    return sha256_bytes(canonical_json_bytes(value))


def load_formal(path):
    artifact = verify_artifact(path, expected_artifact_type="phase2_online_terminal_pilot",
                               expected_schema_version="phase2_online_terminal_pilot_v1")
    m = artifact.manifest
    require(artifact.manifest_digest == FORMAL_DIGEST, "formal manifest digest mismatch")
    require(m["study_name"] == FORMAL_NAME and m["campaign_purpose"] == "pilot"
            and m["completion_status"] == "COMPLETE" and m["target_assignment_count"] == 120
            and m["source_provenance"]["commit"] == FORMAL_SOURCE
            and m["reference_artifact_digest"] == REFERENCE_DIGEST,
            "formal population/source/reference mismatch")
    tables = [tuple(strict_json(line) for line in read_artifact_file(artifact, name).splitlines())
              for name in ("assignments.jsonl", "executions.jsonl", "consequences.jsonl")]
    return artifact, tables


def exact_joins(assignments, executions, consequences):
    require(len(assignments) == len(executions) == 120 and len(consequences) == 119,
            "expected 120 assignments/executions and 119 formal consequences")
    indexed = {}
    games = set()
    for assignment in assignments:
        identity = assignment["opportunity"]["identity"]
        game = identity["game_id"]
        key = digest(assignment)
        require(game not in games and key not in indexed, "assignment identity must be unique")
        require(assignment["assigned_action"] in ("PUSH", "REDIRECT")
                and assignment["assignment_probability"] == 0.5
                and assignment["treatment"]["assignment_source"] == "randomized_pilot"
                and assignment["treatment"]["requested_action"] == assignment["assigned_action"],
                "assignment is not a formal randomized P/N record")
        games.add(game)
        indexed[key] = assignment
    joined = []
    for table in (executions, consequences):
        rows = {}
        for row in table:
            key = row["assignment_digest"]
            require(key in indexed and key not in rows, "execution/consequence join is not exact")
            require(row["game_id"] == indexed[key]["opportunity"]["identity"]["game_id"],
                    "joined game identity differs")
            rows[key] = row
        joined.append(rows)
    execution_map, consequence_map = joined
    require(set(execution_map) == set(indexed), "execution identities do not cover assignments")
    missing = set(indexed) - set(consequence_map)
    require(len(missing) == 1 and indexed[next(iter(missing))]["opportunity"]["identity"]["game_id"] == FAILED_GAME,
            "recovery is allowed only for the known failed assignment")
    for key, execution in execution_map.items():
        if key in consequence_map:
            require(execution["success"] is True and execution["canonical_commit_success"] is True
                    and execution["failure_reason"] is None,
                    "formal consequence must join a successful execution")
    return indexed, execution_map, consequence_map


def bound_inputs(artifact):
    manifest = artifact.manifest
    work = Path(manifest["server_run_provenance"]["work_directory"])
    bound = read_json(work / "run_inputs.json")
    require(bound["inputs_digest"] == digest({k: v for k, v in bound.items() if k != "inputs_digest"})
            and bound["inputs_digest"] == manifest["server_run_provenance"]["inputs_digest"]
            and bound["source_commit"] == FORMAL_SOURCE
            and bound["reference_tables_digest"] == REFERENCE_DIGEST
            and Path(bound["paths"]["work_directory"]).resolve() == work.resolve()
            and Path(bound["paths"]["destination"]).resolve() == artifact.path.resolve(),
            "frozen run_inputs lineage differs")
    game_plan = collection_plan_from_record(bound["canonical_game_plan"])
    require(game_plan.plan_digest == manifest["server_run_provenance"]["game_plan_digest"]
            and game_plan.source_revision == FORMAL_SOURCE and game_plan.collection_id == FORMAL_NAME,
            "canonical game plan binding differs")
    return work, bound, game_plan


def recover(assignment, execution, artifact, work, game_plan, values):
    opportunity = assignment["opportunity"]
    identity, legal = opportunity["identity"], opportunity["public_legal"]
    require(identity["game_id"] == FAILED_GAME and assignment["assigned_action"] == "PUSH"
            and identity["candidate_j"] == "player4" and identity["acting_wolf"] == "player2"
            and identity["phase"] == "speech" and opportunity["s_pre"] == [2, 5]
            and legal["known_wolves"] == ["player2", "player3"]
            and execution["success"] is False and execution["failure_reason"] == "PRIVATE_FACT_CLAIM"
            and execution["canonical_commit_success"] is False
            and execution["canonical_event_id"] is None and execution["canonical_event_digest"] is None,
            "known failure recovery conditions differ")
    require(reference_tables_digest(values) == REFERENCE_DIGEST, "frozen reference table digest differs")
    claim = _claim_from_record(read_json(work / "claims" / f"{FAILED_GAME}.json"), game_plan)
    require(claim.ordinal == 75 and claim.seed == 7714212207331013538,
            "recovery claim identity differs")
    evidence = {}
    # Published executions add join IDs and an audit label to the unchanged
    # nested execution body in the canonical stage record.
    execution_body = {k: v for k, v in execution.items()
                      if k not in ("game_id", "assignment_digest", "theta_audit_label")}
    for stage, expected_sha in (("execution", EXECUTION_SHA), ("game-result", GAME_RESULT_SHA)):
        path = work / "attempts" / claim.attempt_id / "partial_evidence" / f"phase2-{stage}.json"
        verified = validate_canonical_partial_evidence(path, plan=game_plan, claim=claim,
                                                       collection_directory=work)
        require(verified.file_sha256 == expected_sha, "immutable canonical evidence SHA mismatch")
        if stage == "execution":
            require(artifact.manifest["server_run_provenance"]["canonical_stage_evidence"][
                path.relative_to(work).as_posix()] == verified.file_sha256,
                "execution evidence differs from formal manifest pin")
        payload = verified.evidence.payload.to_value()
        require(payload["game_id"] == FAILED_GAME and payload["stage"] == stage
                and payload["phase2_record"]["assignment"] == assignment
                and payload["phase2_record"]["execution"] == execution_body
                and payload["phase2_record"]["offline_audit"]["theta_ac"] == execution["theta_audit_label"]
                and payload["phase2_record"]["day_consequence"] is None,
                "partial evidence record differs from formal assignment/execution")
        evidence[stage] = verified, payload["canonical_runtime"]
    canonical = evidence["game-result"][1]
    prefixes = [p for p in canonical["authoritative_pre_prefixes"] if p["boundary_id"] == identity["boundary_id"]]
    require(len(prefixes) == 1, "recovery PRE must be unique")
    prefix = _prefix_from_record(prefixes[0])
    history = freeze_public_event_history(canonical["public_events"])
    records = history.to_records()
    require(prefix.game_id == FAILED_GAME and prefix.prefix_digest == identity["prefix_digest"]
            and prefix.current_speaker == "player2" and prefix.public_temporal_state.day == 1
            and set(prefix.alive_observer_ids) == set(legal["alive"])
            and len(set(legal["alive"]) & set(legal["known_wolves"])) == 2
            and len(set(legal["alive"]) - set(legal["known_wolves"])) == 5
            and records[:len(prefix.public_event_history.events)] == prefix.public_event_history.to_records()
            and freeze_public_event_history(evidence["execution"][1]["public_events"]).to_records() ==
                prefix.public_event_history.to_records(), "frozen canonical PRE/trajectory differs")
    exiles = [e for e in history.events if e.event_type == "exile_result" and e.temporal_state.day == 1]
    votes = [e for e in history.events if e.event_type == "vote_result" and e.temporal_state.day == 1]
    require(len(exiles) == len(votes) == 1, "day-1 vote/exile must be unique; no heuristic resolution")
    exile, vote = exiles[0], votes[0]
    require(exile.affected_players == ("player6",) and vote.event_index < exile.event_index
            and exile.event_index >= len(prefix.public_event_history.events)
            and exile.temporal_state.phase.value == "vote"
            and exile.event_index + 1 < len(history.events)
            and history.events[exile.event_index + 1].event_type == "phase_change"
            and history.events[exile.event_index + 1].temporal_state.phase.value == "night",
            "wrong exile or unresolved/intermediate day result")
    require("player6" not in legal["known_wolves"] and "player6" != identity["candidate_j"],
            "exile is not another non-wolf")
    outcome = classify_outcome("player6", acting_wolf="player2", candidate_j="player4",
                               wolves=frozenset(legal["known_wolves"]))
    loss = values.deployment_loss(tuple(opportunity["s_pre"]), outcome)
    require(outcome.value == "other_nonwolf_exiled" and loss.s_post == (2, 4)
            and math.isclose(loss.value.value, 0.765, rel_tol=0, abs_tol=1e-12)
            and math.isclose(loss.loss, 0.235, rel_tol=0, abs_tol=1e-12), "wrong frozen reference mapping")
    day = {"Y": outcome.value, "s_plus": list(loss.s_post), "v_ref": loss.value.value,
           "l_ref": loss.loss, "reference_artifact_digest": REFERENCE_DIGEST}
    audit = {"recovery_rule_version": RECOVERY_RULE, "game_id": FAILED_GAME,
        "assignment_id": digest(assignment), "source_formal_manifest_digest": artifact.manifest_digest,
        "reference_table_digest": REFERENCE_DIGEST, "claim_record_digest": claim.record_digest,
        "known_wolves": legal["known_wolves"], "known_wolves_source": "formal_assignment_public_legal",
        "boundary_id": identity["boundary_id"], "prefix_digest": prefix.prefix_digest,
        "relevant_event_ids": [vote.event_id, exile.event_id],
        "relevant_events": [records[vote.event_index], records[exile.event_index]],
        "canonical_partial_evidence": {stage: {"path": str(v.path), "file_sha256": v.file_sha256,
            "record_digest": v.evidence.record_digest,
            "formal_manifest_bound": stage == "execution"} for stage, (v, _) in evidence.items()},
        "recovered_outcome": day}
    return day, audit


def itt_rows(indexed, executions, consequences, recovered):
    rows = []
    for key, assignment in indexed.items():
        opportunity = assignment["opportunity"]
        identity = opportunity["identity"]
        execution = executions[key]
        observed = key in consequences
        day = consequences[key]["day_consequence"] if observed else recovered
        p = opportunity["evidence"]["p_tilde_j"]
        require(type(p) in (int, float) and math.isfinite(p) and 0 <= p <= 1, "invalid frozen p_tilde")
        require(day["reference_artifact_digest"] == REFERENCE_DIGEST
                and day["Y"] in {"target_j_exiled", "other_nonwolf_exiled", "acting_wolf_exiled",
                                 "teammate_wolf_exiled", "no_exile"}
                and all(type(day[k]) in (int, float) and math.isfinite(day[k]) and 0 <= day[k] <= 1
                        for k in ("v_ref", "l_ref"))
                and math.isclose(day["v_ref"] + day["l_ref"], 1, rel_tol=0, abs_tol=1e-12),
                "invalid recorded consequence/reference")
        rows.append({"game_id": identity["game_id"], "assignment_id": key,
            "assigned_action": assignment["assigned_action"],
            "assignment_probability": assignment["assignment_probability"],
            "candidate_j": identity["candidate_j"], "acting_wolf": identity["acting_wolf"],
            "p_tilde_j": p, "S_pre": opportunity["s_pre"], "execution_success": execution["success"],
            "execution_failure_reason": execution["failure_reason"],
            "Y": day["Y"], "S_plus": day["s_plus"], "V_ref": day["v_ref"], "L_ref": day["l_ref"],
            "outcome_source": ("formal_consequence" if observed else "recovered_frozen_canonical_trajectory")})
    return rows


def support(rows):
    counts = Counter(r["assigned_action"] for r in rows)
    means = {arm: statistics.mean(r["L_ref"] for r in rows if r["assigned_action"] == arm)
             for arm in ("PUSH", "REDIRECT")}
    difference = means["PUSH"] - means["REDIRECT"]
    require(len(rows) == 120 and counts == {"PUSH": 58, "REDIRECT": 62}
            and sum(not r["execution_success"] for r in rows) == 1
            and sum(r["outcome_source"] == "recovered_frozen_canonical_trajectory" for r in rows) == 1,
            "ITT population/support count mismatch")
    require(all(math.isclose(means[a], EXPECTED_MEANS[a], rel_tol=0, abs_tol=1e-12) for a in means)
            and math.isclose(difference, EXPECTED_DIFFERENCE, rel_tol=0, abs_tol=1e-12),
            "raw ITT numerical support mismatch; never adjust the data")
    arms = {}
    for arm in means:
        subset = [r for r in rows if r["assigned_action"] == arm]
        ps = [r["p_tilde_j"] for r in subset]
        arms[arm] = {"N": len(subset), "execution_success": sum(r["execution_success"] for r in subset),
            "execution_failure": sum(not r["execution_success"] for r in subset),
            "outcome_categories": dict(Counter(r["Y"] for r in subset)),
            "S_pre_support": dict(Counter(str(r["S_pre"]) for r in subset)),
            "p_tilde_support": {"min": min(ps), "max": max(ps), "mean": statistics.mean(ps),
                "median": statistics.median(ps),
                "deciles_descriptive": dict(Counter(str(min(9, int(p * 10))) for p in ps))},
            "mean_L_ref": means[arm]}
    return {"schema_version": VERSION, "primary_population": "all_120_randomized_assignments",
        "primary_estimand": "E[L_ref | assignment=PUSH] - E[L_ref | assignment=REDIRECT]",
        "execution_success_filter": False, "N_total": 120, "action_counts": dict(counts),
        "execution_success": 119, "execution_failure": 1, "recovery_count": 1,
        "formal_consequence_count": 119, "by_randomized_action": arms,
        "raw_ITT_mean_L_ref": means, "raw_PUSH_minus_REDIRECT": difference, "fitted_lambda": None}


def analyze(formal, destination):
    analysis_source_sha = sha256_bytes(Path(__file__).read_bytes())
    destination = Path(destination)
    artifact, tables = load_formal(formal)
    indexed, executions, consequences = exact_joins(*tables)
    work, bound, game_plan = bound_inputs(artifact)
    require(destination.name == NAME, "analysis destination must use its independent fixed name")
    for protected in (artifact.path, work, Path(bound["paths"]["publication"]),
                      Path(bound["paths"]["evaluation_root"])):
        a, b = destination.resolve(), protected.resolve()
        require(a != b and not a.is_relative_to(b) and not b.is_relative_to(a),
                "analysis output must be disjoint from frozen evidence")
    require(not destination.exists() and not destination.is_symlink(), "analysis destination already exists")
    print("Verifying frozen development reference tables (read-only; no inference or fitting)",
          file=sys.stderr, flush=True)
    values = build_development_layer(bound["paths"]["publication"], bound["paths"]["evaluation_root"]).values
    failed_key = next(k for k in indexed if k not in consequences)
    recovered, audit = recover(indexed[failed_key], executions[failed_key], artifact, work, game_plan, values)
    rows = itt_rows(indexed, executions, consequences, recovered)
    summary = support(rows)
    report = ("# Formal Pilot-T ITT support/audit\n\n"
        "Primary population: all 120 randomized assignments, with no execution-success filter.\n\n"
        "119 outcomes are copied from formal consequences; one is deterministically recovered "
        "from the independently pinned frozen game-result trajectory. The failed execution remains failed.\n\n"
        f"PUSH N=58, mean L_ref={summary['raw_ITT_mean_L_ref']['PUSH']!r}; "
        f"REDIRECT N=62, mean L_ref={summary['raw_ITT_mean_L_ref']['REDIRECT']!r}.\n\n"
        f"Raw PUSH minus REDIRECT={summary['raw_PUSH_minus_REDIRECT']!r}.\n\n"
        "These are raw descriptive ITT arm summaries, without an uncertainty interval or "
        "conditional terminal estimator. No lambda, router, Probe or policy is fitted.\n")
    # Re-verify all source files immediately before independent publication.
    again = verify_artifact(formal, expected_artifact_type="phase2_online_terminal_pilot",
                            expected_schema_version="phase2_online_terminal_pilot_v1")
    require(again.manifest_digest == artifact.manifest_digest, "formal artifact changed during analysis")
    for proof in audit["canonical_partial_evidence"].values():
        require(sha256_bytes(Path(proof["path"]).read_bytes()) == proof["file_sha256"],
                "canonical evidence changed during analysis")
    require(sha256_bytes(Path(__file__).read_bytes()) == analysis_source_sha,
            "analysis source changed during execution")
    published = publish_artifact(destination, manifest_fields={
        "artifact_type": "phase2_terminal_itt_analysis", "schema_version": VERSION, "study_name": NAME,
        "analysis_only": True, "primary_population": summary["primary_population"],
        "source_formal_manifest_digest": artifact.manifest_digest, "source_formal_path": str(artifact.path),
        "source_formal_commit": FORMAL_SOURCE, "reference_table_digest": REFERENCE_DIGEST,
        "analysis_source_sha256": analysis_source_sha,
        "source_run_inputs_digest": bound["inputs_digest"], "canonical_game_plan_digest": game_plan.plan_digest,
        "recovery_rule_version": RECOVERY_RULE, "row_count": 120,
        "recovery_count": 1, "fitted_lambda": None}, files={
            "itt_rows.jsonl": canonical_jsonl_bytes(rows), "recovery_audit.json": canonical_json_bytes(audit),
            "support.json": canonical_json_bytes(summary), "report.md": report.encode("utf-8")})
    return {"destination": str(published.path), "manifest_digest": published.manifest_digest, "support": summary}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal", type=Path, default=ROOT / FORMAL_NAME)
    parser.add_argument("--destination", type=Path, default=ROOT / NAME)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(analyze(args.formal, args.destination), sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"ITT analysis failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
