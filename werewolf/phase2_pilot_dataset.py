"""Validated in-memory pilot dataset and descriptive support, without fitting."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import statistics

from werewolf.artifact_io import canonical_json_bytes, sha256_bytes
from werewolf.phase2_pilot_records import (
    Phase2ActionConsequenceRecordV1, Phase2ProbeSequenceRecordV1,
)
from werewolf.phase2_online_records import (
    Phase2OnlineInterventionRecordV1, validate_probe_lifecycle_record,
)
from werewolf.phase2_online_plan import (
    Phase2OnlineProbePilotPlanV1, probe_assignment_from_record,
)
from werewolf.phase2_offline import ResolvedOutcome


VERSION = "phase2_consequence_dataset_v1"
ONLINE_VERSION = "phase2_online_terminal_consequence_dataset_v1"
PROBE_ONLINE_VERSION = "phase2_online_probe_consequence_dataset_v1"


class PilotDatasetError(ValueError):
    """Only unique, provenanced controlled-intervention records are accepted."""


@dataclass(frozen=True)
class Phase2ConsequenceDatasetV1:
    terminal: tuple[Phase2ActionConsequenceRecordV1, ...]
    probe: tuple[Phase2ProbeSequenceRecordV1, ...]
    manifest: dict

    def to_record(self) -> dict:
        return {"manifest": self.manifest,
                "terminal": [row.to_record() for row in self.terminal],
                "probe": [row.to_record() for row in self.probe]}


@dataclass(frozen=True)
class Phase2OnlineConsequenceDatasetV1:
    assignments: tuple[dict, ...]
    executions: tuple[dict, ...]
    consequences: tuple[dict, ...]
    backend_calls: tuple[dict, ...]
    manifest: dict
    strategy_stages: tuple[dict, ...] = ()

    def to_record(self) -> dict:
        record = {"manifest": self.manifest,
                "assignments": list(self.assignments),
                "executions": list(self.executions),
                "consequences": list(self.consequences),
                "backend_calls": list(self.backend_calls)}
        if self.strategy_stages:
            record["strategy_stages"] = list(self.strategy_stages)
        return record


@dataclass(frozen=True)
class OnlineDatasetAccessV1:
    """Validated read access; audit-only access cannot feed an estimator."""

    datasets: tuple[Phase2OnlineConsequenceDatasetV1, ...]
    estimator_eligible: bool


def require_online_dataset_access(
        datasets: tuple[Phase2OnlineConsequenceDatasetV1, ...], *,
        audit_only: bool = False) -> OnlineDatasetAccessV1:
    """Fail closed on qualification, synthetic, or mixed estimation inputs.

    This is a purpose/provenance gate, not an estimator or a scientific
    assessment of whether the formal pilot has adequate consequence support.
    """
    if (not isinstance(datasets, tuple) or not datasets
            or type(audit_only) is not bool
            or any(not isinstance(dataset, Phase2OnlineConsequenceDatasetV1)
                   for dataset in datasets)):
        raise PilotDatasetError("validated online datasets are required")
    for dataset in datasets:
        analyze_phase2_online_support_record(dataset.to_record())
    if audit_only:
        return OnlineDatasetAccessV1(datasets, estimator_eligible=False)
    if any(dataset.manifest.get("schema_version") == PROBE_ONLINE_VERSION for dataset in datasets):
        raise PilotDatasetError("Probe strategy analysis is unsupported; audit-only access is required")
    if len(datasets) != 1:
        raise PilotDatasetError("estimation requires one formal campaign dataset")
    manifest = datasets[0].manifest
    if (manifest.get("campaign_purpose") != "pilot"
            or manifest.get("estimator_eligible") is not True
            or manifest.get("synthetic_audit_only") is not False
            or manifest.get("target_assignment_count") != 120
            or manifest.get("assignment_count") != 120
            or manifest.get("campaign_status") != "READY_TO_SEAL"
            or manifest.get("sealable") is not True):
        raise PilotDatasetError("qualification or incomplete campaign cannot enter estimation")
    return OnlineDatasetAccessV1(datasets, estimator_eligible=True)


def build_phase2_consequence_dataset(
        terminal: tuple[Phase2ActionConsequenceRecordV1, ...] = (),
        probe: tuple[Phase2ProbeSequenceRecordV1, ...] = (), *,
        source_commit: str, pilot_artifact_digest: str | None = None,
        checkpoint_format: str | None = None, execution_verifier=None,
        allow_synthetic: bool = False,
        online_records: tuple[Phase2OnlineInterventionRecordV1, ...] | None = None,
        online_pilot_plan_digest: str | None = None,
        games_seen: int | None = None,
        eligible_opportunities: int | None = None,
        ) -> Phase2ConsequenceDatasetV1 | Phase2OnlineConsequenceDatasetV1:
    """Reject natural gameplay vote_intent and any duplicate branch identity."""
    if online_records is not None:
        if terminal or probe or checkpoint_format is not None:
            raise PilotDatasetError("online assignments cannot mix with replay branches")
        return build_phase2_online_consequence_dataset(
            online_records, source_commit=source_commit,
            pilot_plan_digest=online_pilot_plan_digest,
            execution_verifier=execution_verifier, allow_synthetic=allow_synthetic,
            games_seen=games_seen, eligible_opportunities=eligible_opportunities)
    rows = (*terminal, *probe)
    if not allow_synthetic:
        if not callable(execution_verifier) or any(
                execution_verifier(row) is not True for row in rows):
            raise PilotDatasetError("verified real intervention execution is required")
    if (not rows or any(not isinstance(row, (Phase2ActionConsequenceRecordV1,
                                            Phase2ProbeSequenceRecordV1)) for row in rows)
            or not all(isinstance(value, str) and value for value in (
                source_commit, pilot_artifact_digest, checkpoint_format))
            or len({row.branch_identity for row in rows}) != len(rows)
            or len({(row.checkpoint_identity, row.treatment.action,
                     row.opportunity.digest()) for row in rows}) != len(rows)
            or any(row.treatment.assignment_source not in ("paired_branch", "randomized_pilot")
                   for row in rows)):
        raise PilotDatasetError("dataset requires unique controlled branch records and provenance")
    if any(not isinstance(row, Phase2ActionConsequenceRecordV1) for row in terminal) or any(
            not isinstance(row, Phase2ProbeSequenceRecordV1) for row in probe):
        raise PilotDatasetError("terminal and Probe records must be separated")
    identities = sorted((row.opportunity.identity, row.treatment.action.value,
                         row.checkpoint_identity, row.branch_identity) for row in rows)
    digests = sorted(row.digest() for row in rows)
    manifest = {"schema_version": VERSION, "source_commit": source_commit,
                "pilot_artifact_digest": pilot_artifact_digest,
                "checkpoint_format": checkpoint_format,
                "assignment_semantics": "controlled_intervention_only",
                "synthetic_audit_only": allow_synthetic,
                "observational_vote_intent_used": False,
                "terminal_count": len(terminal), "probe_count": len(probe),
                "identity_digest": sha256_bytes(canonical_json_bytes(identities)),
                "record_digest": sha256_bytes(canonical_json_bytes(digests))}
    manifest["manifest_digest"] = sha256_bytes(canonical_json_bytes(manifest))
    return Phase2ConsequenceDatasetV1(terminal, probe, manifest)


def build_phase2_online_consequence_dataset(
        records: tuple[Phase2OnlineInterventionRecordV1, ...], *,
        source_commit: str, pilot_plan_digest: str,
        execution_verifier=None, allow_synthetic: bool = False,
        games_seen: int | None = None,
        eligible_opportunities: int | None = None,
        ) -> Phase2OnlineConsequenceDatasetV1:
    """Keep every randomized assignment, including language noncompliance."""
    if (not records or any(not isinstance(row, Phase2OnlineInterventionRecordV1)
                           or row.execution_success is None for row in records)
            or not source_commit or not pilot_plan_digest
            or any(row.assignment.selection.plan_digest != pilot_plan_digest
                   for row in records)
            or len({row.game_id for row in records}) != len(records)
            or len({(row.game_id, row.assignment.opportunity.identity,
                     row.assignment.treatment.treatment_id) for row in records}) != len(records)
            or type(games_seen) is not int or games_seen < len(records)
            or type(eligible_opportunities) is not int
            or eligible_opportunities < len(records)):
        raise PilotDatasetError("online Pilot-T requires one completed assignment per game")
    if not allow_synthetic and (not callable(execution_verifier)
                                or any(execution_verifier(row) is not True for row in records)):
        raise PilotDatasetError("verified real online execution is required")
    assignments = tuple(row.assignment.to_record() for row in records)
    executions = tuple({"game_id": row.game_id,
                        "assignment_digest": row.assignment.digest(),
                        "theta_audit_label": row.theta_audit_label,
                        **row.to_record()["execution"]} for row in records)
    consequences = tuple({"game_id": row.game_id,
                          "assignment_digest": row.assignment.digest(),
                          "day_consequence": row.to_record()["day_consequence"],
                          "theta_audit_label": row.theta_audit_label,
                          "final_game_result": row.final_game_result}
                         for row in records if row.execution_success
                         and row.day_outcome is not None)
    backend_calls = tuple(call.to_record() for row in records
                          for call in row.backend_calls)
    manifest = {"schema_version": ONLINE_VERSION,
                "source_commit": source_commit,
                "pilot_plan_digest": pilot_plan_digest,
                "assignment_semantics": "online_randomized_single_path_do_a",
                "observational_vote_intent_used": False,
                "synthetic_audit_only": allow_synthetic,
                "games_seen": games_seen,
                "eligible_opportunities": eligible_opportunities,
                "assignment_count": len(assignments),
                "execution_count": len(executions),
                "consequence_count": len(consequences),
                "successful_execution_without_day_result": sum(
                    row.execution_success and row.day_outcome is None for row in records),
                "tables_digest": sha256_bytes(canonical_json_bytes([
                    assignments, executions, consequences, backend_calls]))}
    manifest["manifest_digest"] = sha256_bytes(canonical_json_bytes(manifest))
    return Phase2OnlineConsequenceDatasetV1(
        assignments, executions, consequences, backend_calls, manifest)


def build_phase2_online_dataset_from_ledger(ledger, *, execution_verifier=None,
                                             allow_synthetic: bool = False,
                                             ) -> Phase2OnlineConsequenceDatasetV1:
    """Recover all randomized assignments after a crash without rerunning games."""
    from werewolf.phase2_online_ledger import OnlinePilotAssignmentLedgerV1
    if not isinstance(ledger, OnlinePilotAssignmentLedgerV1):
        raise PilotDatasetError("validated durable assignment ledger required")
    snapshot = ledger.snapshot()
    is_probe = isinstance(ledger.plan, Phase2OnlineProbePilotPlanV1)
    if not allow_synthetic and not callable(execution_verifier):
        raise PilotDatasetError("canonical execution verifier required")
    assignments, executions, consequences, backend_calls, strategy_stages = [], [], [], [], []
    for game_id, stage in snapshot["games"].items():
        if "ASSIGNMENT" not in stage:
            continue
        assignment = stage["ASSIGNMENT"]
        assigned = assignment["assignment"]
        assigned_digest = assignment["assignment_id"]
        if assigned_digest != sha256_bytes(canonical_json_bytes(assigned)):
            raise PilotDatasetError("ledger assignment digest differs")
        assignments.append(assigned)
        if is_probe:
            strategy_stages.extend(stage.get("STRATEGY_STAGE_HISTORY", ()))
        calls = [item["call"] for item in stage.get("BACKEND_CALL", ())]
        backend_calls.extend(calls)
        if "EXECUTION" in stage:
            execution = stage["EXECUTION"]
            if (execution["assignment_id"] != assigned_digest
                    or execution["backend_calls"] != calls
                    or not allow_synthetic and execution_verifier(game_id, stage) is not True):
                raise PilotDatasetError("ledger execution lacks canonical proof")
            executions.append({"game_id": game_id,
                               "assignment_digest": assigned_digest,
                               "theta_audit_label": None,
                               **execution["execution"]})
        if "CONSEQUENCE" in stage:
            consequence = stage["CONSEQUENCE"]
            if (consequence["assignment_id"] != assigned_digest
                    or is_probe and not allow_synthetic
                    and execution_verifier(game_id, stage) is not True):
                raise PilotDatasetError("ledger consequence disagrees with assignment")
            consequences.append({"game_id": game_id,
                                 "assignment_digest": assigned_digest,
                                 "day_consequence": consequence["day_consequence"],
                                 "theta_audit_label": consequence["theta_audit_label"],
                                 "final_game_result": (
                                     stage["GAME_RESULT"]["winner"]
                                     if "GAME_RESULT" in stage else
                                     stage["STRATEGY_STAGE"]["record"]["offline_audit"]["final_game_result"]
                                     if is_probe else consequence["final_game_result"])})
    assignments = tuple(assignments)
    executions = tuple(executions)
    consequences = tuple(consequences)
    backend_calls = tuple(backend_calls)
    strategy_stages = tuple(strategy_stages)
    tables = [assignments, executions, consequences, backend_calls]
    if is_probe:
        tables.append(strategy_stages)
    consequence_games = {row["game_id"] for row in consequences}
    manifest = {"schema_version": PROBE_ONLINE_VERSION if is_probe else ONLINE_VERSION,
                "source_commit": ledger.source_commit,
                "pilot_plan_digest": ledger.plan.digest(),
                "campaign_purpose": ledger.plan.campaign_purpose,
                "estimator_eligible": not is_probe and ledger.plan.campaign_purpose == "pilot",
                "target_assignment_count": ledger.plan.target_assignment_count,
                "campaign_status": ledger.status(),
                "sealable": ledger.sealable(),
                "assignment_semantics": ("online_randomized_whole_probe_strategy" if is_probe
                                         else "online_randomized_single_path_do_a"),
                "observational_vote_intent_used": False,
                "synthetic_audit_only": allow_synthetic,
                "games_seen": snapshot["games_attempted"],
                "eligible_opportunities": len(assignments),
                "assignment_count": len(assignments),
                "execution_count": len(executions),
                "missing_execution_count": len(assignments) - len(executions),
                "consequence_count": len(consequences),
                "successful_execution_without_day_result": sum(
                    row["success"] is True and row["game_id"] not in consequence_games
                    for row in executions) if is_probe else sum(
                    row["success"] is True for row in executions) - len(consequences),
                "ledger_digest": snapshot["events"][-1]["digest"],
                "tables_digest": sha256_bytes(canonical_json_bytes(tables))}
    if is_probe:
        manifest.update({"probe_plan": ledger.plan.to_record(),
                         "strategy_stage_count": len(strategy_stages),
                         "itt_assignment_count": len(assignments),
                         "itt_consequence_count": len(consequences),
                         "itt_outcome_complete": bool(assignments)
                         and len(assignments) == len(consequences)})
    manifest["manifest_digest"] = sha256_bytes(canonical_json_bytes(manifest))
    dataset = Phase2OnlineConsequenceDatasetV1(
        assignments, executions, consequences, backend_calls, manifest, strategy_stages)
    analyze_phase2_online_support_record(dataset.to_record())
    return dataset


def analyze_phase2_consequence_support(dataset: Phase2ConsequenceDatasetV1) -> dict:
    """Counts and observed loss distribution only; no smoothing or selection."""
    if not isinstance(dataset, Phase2ConsequenceDatasetV1):
        raise PilotDatasetError("validated pilot dataset required")
    return analyze_phase2_consequence_support_record(dataset.to_record())


def analyze_phase2_online_support_record(payload: dict) -> dict:
    """Descriptive randomized assignment, compliance, and consequence support."""
    try:
        manifest = payload["manifest"]
        assignments = payload["assignments"]
        executions = payload["executions"]
        consequences = payload["consequences"]
        backend_calls = payload["backend_calls"]
        is_probe = manifest["schema_version"] == PROBE_ONLINE_VERSION
        strategy_stages = payload.get("strategy_stages", [])
        tables = [assignments, executions, consequences, backend_calls]
        if is_probe:
            tables.append(strategy_stages)
        if (manifest["schema_version"] not in (ONLINE_VERSION, PROBE_ONLINE_VERSION)
                or manifest["assignment_semantics"] != (
                    "online_randomized_whole_probe_strategy" if is_probe else
                    "online_randomized_single_path_do_a")
                or not is_probe and strategy_stages
                or manifest["observational_vote_intent_used"] is not False
                or manifest["manifest_digest"] != sha256_bytes(canonical_json_bytes({
                    key: value for key, value in manifest.items() if key != "manifest_digest"}))
                or manifest["tables_digest"] != sha256_bytes(canonical_json_bytes(tables))
                or (len(assignments), len(executions), len(consequences)) != (
                    manifest["assignment_count"], manifest["execution_count"],
                    manifest["consequence_count"])):
            raise PilotDatasetError("online dataset manifest or table digest mismatch")
        if is_probe:
            frozen = manifest["probe_plan"]
            plan = Phase2OnlineProbePilotPlanV1(
                frozen["pilot_id"], frozen["assignment_seed"], frozen["target_assignment_count"],
                frozen["max_games_attempted"], frozen["campaign_purpose"],
                frozen["candidate_selection_rule"], frozen["selection_status"])
            if frozen != plan.to_record() or plan.digest() != manifest["pilot_plan_digest"]:
                raise PilotDatasetError("Probe dataset frozen plan differs")
        by_assignment = {}
        for row in assignments:
            identity = row["opportunity"]["identity"]
            game_id = identity["game_id"]
            key = sha256_bytes(canonical_json_bytes(row))
            if is_probe:
                probe_assignment_from_record(row, plan)
                if game_id in by_assignment:
                    raise PilotDatasetError("duplicate Probe assignment")
            elif (game_id in by_assignment or row["schema_version"] !=
                    "phase2_online_terminal_assignment_v1"
                    or row["assigned_action"] not in ("PUSH", "REDIRECT")
                    or row["assigned_action"] not in row["legal_action_set"]
                    or not {"PUSH", "REDIRECT"}.issubset(row["legal_action_set"])
                    or row["treatment"]["assignment_source"] != "randomized_pilot"
                    or row["selection"]["plan_digest"] != manifest["pilot_plan_digest"]
                    or row["selection"]["candidate_j"] not in
                       row["selection"]["candidate_pool"]
                    or len(row["selection"]["candidate_pool"]) < 2
                    or row["selection"]["candidate_selection_probability"] !=
                       1 / len(row["selection"]["candidate_pool"])
                    or row["selection"]["candidate_selection_seed"] != row["assignment_seed"]
                    or row["assignment_probability"] != 0.5
                    or row["assignment_probability"] != row["treatment"]["assignment_probability"]
                    or row["treatment"]["observational_vote_intent_is_treatment"] is not False):
                raise PilotDatasetError("illegal or duplicate online assignment")
            by_assignment[game_id] = (key, row)
        latest_strategy = {}
        if is_probe:
            for snapshot in strategy_stages:
                game_id = snapshot["game_id"]
                raw = snapshot["record"]
                validate_probe_lifecycle_record(raw)
                if (game_id not in by_assignment
                        or snapshot["assignment_id"] != by_assignment[game_id][0]
                        or raw["assignment"] != by_assignment[game_id][1]):
                    raise PilotDatasetError("Probe lifecycle snapshot differs from assignment")
                previous = latest_strategy.get(game_id)
                if previous is None:
                    if raw["lifecycle"] != ["ASSIGNED"]:
                        raise PilotDatasetError("Probe lifecycle history lacks initial assignment")
                else:
                    if (raw["lifecycle"][:-1] != previous["lifecycle"]
                            or len(raw["lifecycle"]) != len(previous["lifecycle"]) + 1
                            or raw["observations"][:len(previous["observations"])] != previous["observations"]):
                        raise PilotDatasetError("Probe lifecycle history is not append-only")
                    for name in ("T1", "T3"):
                        if (previous["stages"][name] is not None
                                and raw["stages"][name] != previous["stages"][name]):
                            raise PilotDatasetError("Probe lifecycle stage evidence changed")
                    for name in ("opportunity", "treatment"):
                        if (previous["continuation"][name] is not None
                                and raw["continuation"][name] != previous["continuation"][name]):
                            raise PilotDatasetError("Probe lifecycle continuation changed")
                latest_strategy[game_id] = raw
        by_execution = {}
        for row in executions:
            game_id = row["game_id"]
            if (game_id in by_execution or game_id not in by_assignment
                    or row["assignment_digest"] != by_assignment[game_id][0]
                    or type(row["success"]) is not bool
                    or type(row["theta_audit_label"]) not in (bool, type(None))
                    or row["canonical_commit_success"] is not row["success"]):
                raise PilotDatasetError("online execution disagrees with assignment")
            if is_probe and (game_id not in latest_strategy or row != {
                    "game_id": game_id, "assignment_digest": by_assignment[game_id][0],
                    "theta_audit_label": None, **latest_strategy[game_id]["execution"]}):
                raise PilotDatasetError("Probe execution differs from lifecycle evidence")
            by_execution[game_id] = row
        if (not set(by_execution) <= set(by_assignment)
                or manifest.get("missing_execution_count", 0) !=
                   len(by_assignment) - len(by_execution)):
            raise PilotDatasetError("assignment/execution counts disagree")
        by_consequence = {}
        for row in consequences:
            game_id = row["game_id"]
            day = row["day_consequence"]
            if (game_id in by_consequence or game_id not in by_assignment
                    or game_id not in by_execution
                    or not is_probe and by_execution[game_id]["success"] is not True
                    or row["assignment_digest"] != by_assignment[game_id][0]
                    or day["Y"] not in {category.value for category in ResolvedOutcome}
                    or type(day["l_ref"]) not in (int, float)
                    or not math.isfinite(day["l_ref"]) or not 0 <= day["l_ref"] <= 1):
                raise PilotDatasetError("consequence is not a successful terminal intervention")
            if is_probe and (game_id not in latest_strategy
                    or day != latest_strategy[game_id]["day_consequence"]
                    or row["theta_audit_label"] is not None
                    or row["final_game_result"] != latest_strategy[game_id]["offline_audit"]["final_game_result"]):
                raise PilotDatasetError("Probe consequence differs from lifecycle evidence")
            by_consequence[game_id] = row
        completed_executions = (set(by_execution) if is_probe else
                                {game_id for game_id, row in by_execution.items() if row["success"]})
        expected_success_missing = (sum(row["success"] and game_id not in by_consequence
                                        for game_id, row in by_execution.items()) if is_probe else
                                    sum(row["success"] for row in by_execution.values()) - len(consequences))
        if (not set(by_consequence) <= completed_executions
                or manifest["successful_execution_without_day_result"] !=
                   expected_success_missing):
            raise PilotDatasetError("incomplete day consequences disagree with execution")
        if (type(manifest["games_seen"]) is not int
                or manifest["games_seen"] < len(assignments)
                or type(manifest["eligible_opportunities"]) is not int
                or manifest["eligible_opportunities"] < len(assignments)):
            raise PilotDatasetError("invalid pilot opportunity counters")
        if (manifest.get("campaign_purpose") is not None and
                (manifest["campaign_purpose"] not in ("qualification", "pilot")
                 or manifest.get("target_assignment_count") != (
                    plan.target_assignment_count if is_probe else
                    (10 if manifest["campaign_purpose"] == "qualification" else 120))
                 or manifest.get("estimator_eligible") is not
                    (not is_probe and manifest["campaign_purpose"] == "pilot"))):
            raise PilotDatasetError("campaign purpose or estimator exclusion differs")
        if is_probe and (manifest["campaign_purpose"] != plan.campaign_purpose
                or manifest["strategy_stage_count"] != len(strategy_stages)
                or manifest["itt_assignment_count"] != len(assignments)
                or manifest["itt_consequence_count"] != len(consequences)
                or manifest["itt_outcome_complete"] is not
                   (bool(assignments) and len(assignments) == len(consequences))):
            raise PilotDatasetError("Probe ITT audit counts differ")
    except (KeyError, TypeError, AttributeError, IndexError, ValueError) as error:
        if isinstance(error, PilotDatasetError):
            raise
        raise PilotDatasetError("malformed online pilot dataset") from error
    arm = "assigned_strategy" if is_probe else "assigned_action"
    assigned = Counter(row[arm] for row in assignments)
    success = Counter(by_assignment[game_id][1][arm]
                      for game_id, execution in by_execution.items() if execution["success"])
    by_phase = Counter((row["opportunity"]["identity"]["phase"], row[arm])
                       for row in assignments)
    by_s_pre = Counter((tuple(row["opportunity"]["s_pre"]), row[arm])
                       for row in assignments)
    by_candidate = Counter((row["opportunity"]["identity"]["candidate_j"],
                            row[arm]) for row in assignments)
    by_pool_size = Counter(len(row["selection"].get("candidate_pool", ()))
                           for row in assignments)
    candidate_frequency = Counter(row["selection"]["candidate_j"]
                                  for row in assignments)
    failure_reasons = Counter(row["failure_reason"] for row in executions
                              if row["success"] is False)
    by_theta = Counter((str(row["theta_audit_label"]),
                        by_assignment[row["game_id"]][1][arm])
                       for row in executions)
    by_p_bin = Counter((min(9, int(row["opportunity"]["evidence"]["p_tilde_j"] * 10)),
                        row[arm]) for row in assignments)
    y = Counter((by_assignment[row["game_id"]][1][arm],
                 row["day_consequence"]["Y"]) for row in consequences)
    loss = defaultdict(list)
    for row in consequences:
        loss[by_assignment[row["game_id"]][1][arm]].append(
            row["day_consequence"]["l_ref"])
    support = {"schema_version": ("phase2_online_probe_support_v1" if is_probe else
                                  "phase2_online_terminal_support_v1"),
            "games_seen": manifest["games_seen"],
            "eligible_opportunities": manifest["eligible_opportunities"],
            "games_assigned": len(assignments),
            "campaign_purpose": manifest.get("campaign_purpose"),
            "estimator_eligible": manifest.get("estimator_eligible"),
            "target_assignment_count": manifest.get("target_assignment_count"),
            "campaign_status": manifest.get("campaign_status"),
            "sealable": manifest.get("sealable"),
            "assigned_by_action": dict(sorted(assigned.items())),
            "execution_success_by_action": dict(sorted(success.items())),
            "execution_failure_reasons": dict(sorted(failure_reasons.items())),
            "execution_missing": len(assignments) - len(executions),
            "consequence_missing": len(assignments) - len(consequences),
            "candidate_pool_sizes": {str(size): count for size, count in
                                     sorted(by_pool_size.items())},
            "candidate_selection_frequencies": dict(sorted(candidate_frequency.items())),
            "successful_execution_without_day_result": manifest[
                "successful_execution_without_day_result"],
            "execution_rate_by_action": {action: success[action] / count
                                         for action, count in sorted(assigned.items())},
            "by_phase": {str(key): count for key, count in sorted(by_phase.items())},
            "by_s_pre": {str(key): count for key, count in sorted(by_s_pre.items())},
            "by_candidate": {str(key): count for key, count in sorted(by_candidate.items())},
            "by_theta_audit": {str(key): count for key, count in sorted(by_theta.items())},
            "by_p_tilde_decile_descriptive": {str(key): count for key, count in sorted(by_p_bin.items())},
            "Y_by_action": {str(key): count for key, count in sorted(y.items())},
            "L_ref_by_action": {action: {"count": len(values),
                                  "min": min(values), "median": statistics.median(values),
                                  "max": max(values), "mean": statistics.fmean(values)}
                                for action, values in sorted(loss.items())},
            "fitted_lambda": None}
    if is_probe:
        for name in ("assigned_by_action", "execution_success_by_action", "execution_rate_by_action",
                     "Y_by_action", "L_ref_by_action"):
            support[name.replace("action", "strategy")] = support.pop(name)
        support.pop("fitted_lambda")
        support.update({"strategy_stage_count": len(strategy_stages),
                        "itt_assignment_count": len(assignments),
                        "itt_consequence_count": len(consequences),
                        "itt_outcome_complete": manifest["itt_outcome_complete"]})
    return support


def analyze_phase2_consequence_support_record(payload: dict) -> dict:
    """Validate serialized dataset digests before reporting observed support."""
    try:
        manifest = payload["manifest"]
        terminal, probe = payload["terminal"], payload["probe"]
        rows = (*terminal, *probe)
        if (manifest["schema_version"] != VERSION or not rows
                or type(manifest["synthetic_audit_only"]) is not bool
                or manifest["manifest_digest"] != sha256_bytes(canonical_json_bytes({
                    key: value for key, value in manifest.items() if key != "manifest_digest"}))
                or manifest["assignment_semantics"] != "controlled_intervention_only"
                or manifest["observational_vote_intent_used"] is not False
                or (manifest["terminal_count"], manifest["probe_count"]) != (len(terminal), len(probe))):
            raise PilotDatasetError("invalid controlled pilot manifest")
        normalized = []
        identities = []
        for record in terminal:
            if record["schema_version"] != "phase2_action_consequence_v1":
                raise PilotDatasetError("invalid terminal record schema")
        for record in probe:
            if record["schema_version"] != "phase2_probe_sequence_v1":
                raise PilotDatasetError("invalid Probe record schema")
        for record in rows:
            is_probe = "T0" in record
            runtime = record["T0"] if is_probe else record["runtime_safe"]
            audit = record["audit"] if is_probe else record["offline_audit"]
            opportunity, treatment = runtime["opportunity"], runtime["treatment"]
            identity = opportunity["identity"]
            identity_tuple = tuple(identity[key] for key in (
                "game_id", "boundary_id", "prefix_digest", "acting_wolf", "phase", "candidate_j"))
            action = treatment["requested_action"]
            opportunity_digest = sha256_bytes(canonical_json_bytes(opportunity))
            p_tilde = runtime["p_tilde_before"] if is_probe else runtime["p_tilde_at_pre"]
            terminal_valid = is_probe or (
                audit["Y"] in {category.value for category in ResolvedOutcome}
                and type(audit["l_ref"]) in (int, float)
                and math.isfinite(audit["l_ref"]) and 0 <= audit["l_ref"] <= 1)
            if (treatment["assignment_source"] not in ("paired_branch", "randomized_pilot")
                    or treatment["observational_vote_intent_is_treatment"] is not False
                    or treatment["opportunity_digest"] != opportunity_digest
                    or treatment["opportunity_identity"] != list(identity_tuple)
                    or treatment["candidate_j"] != identity["candidate_j"]
                    or audit["branch_identity"] != sha256_bytes(canonical_json_bytes([
                        audit["checkpoint_identity"], treatment["treatment_id"], audit["branch_seed"]]))
                    or action not in opportunity["legal_actions"]
                    or type(p_tilde) not in (int, float) or not math.isfinite(p_tilde)
                    or not 0 <= p_tilde <= 1
                    or (is_probe and action != "PROBE")
                    or (not is_probe and action not in ("PUSH", "REDIRECT"))):
                raise PilotDatasetError("record is not a legal controlled intervention")
            if not terminal_valid:
                raise PilotDatasetError("invalid terminal outcome or reference loss")
            identities.append((identity_tuple, action, audit["checkpoint_identity"],
                               audit["branch_identity"]))
            normalized.append({"action": action, "phase": identity["phase"],
                "theta": "unknown" if is_probe or audit["theta_ac"] is None else str(audit["theta_ac"]).lower(),
                "s_pre": str(tuple(opportunity["s_pre"])),
                "action_set": "/".join(opportunity["legal_actions"]),
                "candidate": identity["candidate_j"],
                "p_tilde": p_tilde,
                "opportunity_digest": opportunity_digest,
                "checkpoint": audit["checkpoint_identity"],
                "Y": None if is_probe else audit["Y"],
                "loss": None if is_probe else audit["l_ref"],
                "probe_status": record if is_probe else None})
        digests = sorted(sha256_bytes(canonical_json_bytes(row)) for row in rows)
        if (len(set(identities)) != len(identities)
                or len({identity[3] for identity in identities}) != len(identities)
                or manifest["identity_digest"] != sha256_bytes(canonical_json_bytes(sorted(identities)))
                or manifest["record_digest"] != sha256_bytes(canonical_json_bytes(digests))):
            raise PilotDatasetError("dataset identity or record digest mismatch")
    except (KeyError, TypeError, AttributeError, IndexError) as error:
        raise PilotDatasetError("malformed pilot dataset record") from error
    by_action = Counter(row["action"] for row in normalized)
    by_phase = Counter((row["phase"], row["action"]) for row in normalized)
    by_theta = Counter((row["theta"], row["action"]) for row in normalized)
    by_s_pre = Counter((row["s_pre"], row["action"]) for row in normalized)
    by_action_set = Counter((row["action_set"], row["action"]) for row in normalized)
    by_candidate = Counter((row["candidate"], row["action"]) for row in normalized)
    y = Counter((row["action"], row["Y"]) for row in normalized if row["Y"] is not None)
    losses = defaultdict(list)
    candidate_probabilities = defaultdict(list)
    for row in normalized:
        candidate_probabilities[row["action"]].append(row["p_tilde"])
        if row["loss"] is not None:
            losses[row["action"]].append(row["loss"])
    groups = defaultdict(set)
    for row in normalized:
        groups[(row["opportunity_digest"], row["checkpoint"])].add(row["action"])
    def pairs(*names):
        return sum(all(name in actions for name in names) for actions in groups.values())
    probe_rows = [row["probe_status"] for row in normalized if row["probe_status"] is not None]
    return {"schema_version": "phase2_consequence_support_v1",
            "dataset_identity_digest": manifest["identity_digest"],
            "action_counts": dict(sorted(by_action.items())),
            "by_phase": {str(key): count for key, count in sorted(by_phase.items())},
            "by_theta_audit": {str(key): count for key, count in sorted(by_theta.items())},
            "by_s_pre": {str(key): count for key, count in sorted(by_s_pre.items())},
            "by_action_set": {str(key): count for key, count in sorted(by_action_set.items())},
            "by_candidate_seat": {str(key): count for key, count in sorted(by_candidate.items())},
            "candidate_p_tilde": {action: {"count": len(values), "min": min(values),
                "median": statistics.median(values), "max": max(values),
                "mean": statistics.fmean(values)}
                for action, values in sorted(candidate_probabilities.items())},
            "terminal_Y_counts": {str(key): count for key, count in sorted(y.items())},
            "terminal_L_ref": {action: {"count": len(values), "min": min(values),
                "median": statistics.median(values), "max": max(values),
                "mean": statistics.fmean(values)} for action, values in sorted(losses.items())},
            "probe": {"request_executed": sum(row["T1"]["request_executed"] is True for row in probe_rows),
                      "response_opportunity_reached": sum(row["T2"]["response_opportunity_reached"] is True
                                                          for row in probe_rows),
                      "public_observation_realized": sum(row["T2"]["public_observation_realized"] is True
                                                         for row in probe_rows),
                      "reconsideration_reached": sum(row["T3"]["reconsideration_reached"] is True
                                                     for row in probe_rows),
                      "original_j_still_legal": sum(row["T3"]["original_j_still_legal"] is True
                                                    for row in probe_rows)},
            "paired_opportunities": {"P_vs_N": pairs("PUSH", "REDIRECT"),
                                     "P_vs_B": pairs("PUSH", "PROBE"),
                                     "N_vs_B": pairs("REDIRECT", "PROBE"),
                                     "P_vs_N_vs_B": pairs("PUSH", "REDIRECT", "PROBE")}}
