"""Frozen 58-case real-LLM Phase-2 language execution smoke study.

Selection reads current public PREs and the acting wolf's lawful team only.
No gameplay continuation, outcome, report, or final-predictor inference occurs.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.phase2_language_realization import realize_verify_action
from werewolf import phase2_offline as offline
from werewolf.artifact_io import canonical_json_bytes, canonical_jsonl_bytes, publish_artifact, sha256_bytes
from werewolf.canonical_collection.public_history import PLAYER_IDS
from werewolf.development_publication import open_publication, open_role_sidecar
from werewolf.phase2_actions import (
    Action, ActionContextV1, CONTRACT_VERSION, context_from_pre,
    probe_continuation, probe_plan, push_plan, redirect_plan,
)
from werewolf.phase2_language import (
    LANGUAGE_VERSION, Phase2LanguageActorV1, Phase2SemanticPerceiverV1,
    public_language_context_from_pre,
)
from werewolf.phase2_language_audit import LANGUAGE_AUDIT_VERSION


NAME = "paper-phase2-language-smoke-v3"
VERSION = "phase2_language_smoke_v3"
SEED = "phase2-language-smoke-v1:20260929"
EXPECTED_CASE_SELECTION_DIGEST = "9ab58cb73fdee18fee63795ee71df37e5ff00c2c320939c2f48221c3167e192e"
# Reserve the entire scarce PK Probe stratum before all other draws.
LAYOUT = (("speech_pk", Action.PROBE, 8), ("speech", Action.PROBE, 10),
          ("speech_pk", Action.REDIRECT, 10), ("speech", Action.REDIRECT, 10),
          ("speech_pk", Action.PUSH, 10), ("speech", Action.PUSH, 10))
PUBLICATION = Path("/data/yuxiao/Untrusted_Network_Simulation/publications/paper-development-qwen35-9b-1500-v1")
EVALUATION = Path("/data/yuxiao/Untrusted_Network_Simulation/paper-studies/evaluations/0b8c1d7220aae6fcf577038f8d930c97d6453b7f45616ed0a19c817e7a80b956")
MAPPER = Path("/data/yuxiao/Untrusted_Network_Simulation/paper-studies/mapper-final/paper-phase2-mapper-final-v1")
MAPPER_DIGEST = "8ab529972a5722e61e0a81ec37f089677d1276c21cb83921aa842b2ce33995c3"
REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE_FILES = ("scripts/run_phase2_language_smoke.py", "scripts/phase2_language_realization.py",
                "werewolf/phase2_actions.py", "werewolf/phase2_language.py",
                "werewolf/phase2_language_audit.py", "werewolf/phase2_mapper_runtime.py",
                "werewolf/phase2_offline.py")


class SmokeStudyError(ValueError):
    """Selection, lineage, or execution differs from the preregistered smoke study."""


@dataclass(frozen=True)
class Case:
    context: ActionContextV1
    candidate_j: str
    action: Action
    pre: object | None = None

    @property
    def identity(self) -> tuple[str, str, str, str, str, str]:
        c = self.context
        return c.game_id, c.boundary_id, c.acting_wolf, c.phase, self.candidate_j, self.action.value

    @property
    def pre_identity(self) -> tuple[str, str]:
        return self.context.game_id, self.context.boundary_id

    def to_record(self) -> dict:
        c = self.context
        return {"game_id": c.game_id, "boundary_id": c.boundary_id,
                "acting_wolf": c.acting_wolf, "phase": c.phase,
                "candidate_j": self.candidate_j, "action": self.action.value,
                "prefix_digest": c.prefix_digest, "public_history_digest": c.public_history_digest}


def collect_eligible(publication, sidecar) -> tuple[Case, ...]:
    """Inspect current PRE and wolf-team membership; never read game suffixes."""
    if (publication.manifest_digest != offline.PUBLICATION_DIGEST
            or sidecar.sidecar_digest != offline.ROLE_SIDECAR_DIGEST
            or len(publication.public_view.games) != offline.EXPECTED_GAMES):
        raise SmokeStudyError("development publication or role sidecar lineage mismatch")
    teams = {game.game_id: frozenset(p for p, role in game.role_assignment
                                     if role == "Werewolf") for game in sidecar.games}
    cases = []
    candidate_pre = candidate_rows = 0
    for game in publication.public_view.games:
        wolves = teams[game.game_id]
        for pre in game.authoritative_pre_prefixes:
            if pre.public_temporal_state.phase.value not in ("discussion", "pk_discussion"):
                continue
            if pre.current_speaker not in wolves or pre.current_speaker not in pre.alive_observer_ids:
                continue
            context = context_from_pre(pre, wolves)
            if not context.legal_targets:
                if context.phase != "speech_pk":
                    raise SmokeStudyError("ordinary wolf PRE lacks legal candidate")
                continue
            candidate_pre += 1
            candidate_rows += len(context.legal_targets)
            for j in context.legal_targets:
                cases.append(Case(context, j, Action.PUSH, pre))
                if len(context.legal_targets) > 1:
                    cases.append(Case(context, j, Action.REDIRECT, pre))
                if probe_continuation(context, j) is not None:
                    cases.append(Case(context, j, Action.PROBE, pre))
    if (candidate_pre, candidate_rows) != (offline.EXPECTED_PRE, offline.EXPECTED_CANDIDATES):
        raise SmokeStudyError(f"candidate population mismatch: {candidate_pre} PRE, {candidate_rows} rows")
    return tuple(cases)


def select_cases(eligible: tuple[Case, ...]) -> tuple[Case, ...]:
    """SHA-256 seed rank within fixed strata; one unique PRE globally."""
    seen_identity = set()
    buckets: dict[tuple[str, Action], list[Case]] = {(p, a): [] for p, a, _ in LAYOUT}
    for case in eligible:
        if case.identity in seen_identity:
            raise SmokeStudyError("duplicate eligible action/candidate identity")
        seen_identity.add(case.identity)
        if (case.context.phase, case.action) in buckets:
            buckets[(case.context.phase, case.action)].append(case)
    chosen, used_pre = [], set()
    for phase, action, quota in LAYOUT:
        pool = buckets[(phase, action)]
        if phase == "speech_pk" and action is Action.PROBE and len(pool) != quota:
            raise SmokeStudyError(f"PK Probe population changed: expected all {quota}, found {len(pool)}")
        pool.sort(key=lambda c: (sha256_bytes(canonical_json_bytes([SEED, *c.identity])), c.identity))
        available = [c for c in pool if c.pre_identity not in used_pre]
        if len({c.pre_identity for c in available}) < quota:
            raise SmokeStudyError(f"insufficient unique PRE for {phase}/{action.value}: "
                                  f"need {quota}, have {len({c.pre_identity for c in available})}; "
                                  "sampling definition unchanged")
        for case in available:
            if case.pre_identity in used_pre:
                continue
            chosen.append(case)
            used_pre.add(case.pre_identity)
            if sum(c.context.phase == phase and c.action is action for c in chosen) == quota:
                break
    if len(chosen) != 58 or len(used_pre) != 58:
        raise SmokeStudyError("58 unique PRE selection invariant failed")
    return tuple(chosen)


def selection_digest(cases: tuple[Case, ...]) -> str:
    return sha256_bytes(canonical_json_bytes([case.to_record() for case in cases]))


def selected_oof_q(publication, cases: tuple[Case, ...], evaluation: Path | str) -> dict[tuple[str, str], tuple[tuple[float, ...], ...]]:
    """Use frozen sealed reader for all seven observer rows at selected PREs."""
    folds = offline.fold_assignment(publication.public_view.fold_manifest)
    opportunities = tuple(SimpleNamespace(game_id=c.context.game_id,
        boundary_id=c.context.boundary_id, prefix_digest=c.context.prefix_digest,
        fold=folds[c.context.game_id], alive=PLAYER_IDS, wolves=frozenset())
        for c in cases if c.action is Action.REDIRECT)
    probabilities = offline.load_oof_probabilities(publication, evaluation, opportunities)
    result = {}
    for case in cases:
        if case.action is not Action.REDIRECT:
            continue
        key = case.pre_identity
        result[key] = tuple(probabilities[(key[0], key[1], observer)] for observer in PLAYER_IDS)
    return result


def redirect_probabilities(case: Case, q, mapper) -> dict[str, float]:
    c = case.context
    return {k: mapper.infer(q, alive=c.alive, known_wolves=c.known_wolves,
                            acting_wolf=c.acting_wolf, candidate_j=k, phase=c.phase,
                            public_speaker_queue=c.public_speaker_queue,
                            competition=c.competition)
            for k in c.legal_targets if k != case.candidate_j}


def make_plan(case: Case, q_by_pre, mapper):
    c, j = case.context, case.candidate_j
    if case.action is Action.PUSH:
        return push_plan(c, j)
    if case.action is Action.PROBE:
        return probe_plan(c, j)
    q = q_by_pre[case.pre_identity]
    return redirect_plan(c, j, redirect_probabilities(case, q, mapper))


def _metrics(rows: list[dict]) -> dict:
    n = len(rows)
    first = sum(bool(row["audit"]["attempts"][0]["language_execution_valid"]) for row in rows)
    repaired = sum(len(row["audit"]["attempts"]) == 2 for row in rows)
    repair_success = sum(len(row["audit"]["attempts"]) == 2 and row["audit"]["language_execution_valid"]
                         for row in rows)
    final = sum(bool(row["audit"]["language_execution_valid"]) for row in rows)
    return {"case_count": n, "first_pass_valid": first,
            "first_pass_valid_rate": first / n if n else None,
            "repair_attempted": repaired, "repair_success": repair_success,
            "final_valid": final, "final_valid_rate": final / n if n else None,
            "final_invalid": n - final}


def aggregate(rows: list[dict], *, plan_isolation_verified: bool) -> dict:
    reasons = Counter(a["language_invalid_reason"] for row in rows for a in row["audit"]["attempts"]
                      if not a["language_execution_valid"])
    final_reasons = Counter(row["audit"]["language_invalid_reason"] for row in rows
                            if not row["audit"]["language_execution_valid"])
    parsed = [a["perceived_semantics"] for row in rows for a in row["audit"]["attempts"]
              if a["perceived_semantics"] is not None]
    identities = sorted({p["action_identity"] for p in parsed if p["action_identity"] is not None})
    extra = Counter()
    for row in rows:
        perceived = row["audit"]["attempts"][-1]["perceived_semantics"]
        if perceived is None:
            continue
        expected = [] if row["action"] == "PROBE" else [row["plan"]["commitment_target"]]
        if set(perceived["commitment_targets"]) - set(expected):
            extra[row["action"]] += 1
    overall = _metrics(rows)
    actions = {a.value: _metrics([row for row in rows if row["action"] == a.value]) for a in Action}
    phases = {p: _metrics([row for row in rows if row["phase"] == p]) for p in ("speech", "speech_pk")}
    # A systematic final extra commitment is >=30% within an action.
    systematic = {a: extra[a] >= math.ceil(.30 * actions[a]["case_count"])
                  for a in actions}
    gate_checks = {"overall_final_valid_at_least_0_90": overall["final_valid_rate"] is not None and overall["final_valid_rate"] >= .90,
                   **{f"{a.lower()}_final_valid_at_least_0_80": actions[a]["final_valid_rate"] is not None and actions[a]["final_valid_rate"] >= .80 for a in actions},
                   "no_schema_wide_parse_failure": bool(parsed),
                   "no_perception_one_action_collapse": len(identities) >= 2,
                   "no_requested_plan_leakage": plan_isolation_verified,
                   "no_systematic_illegal_extra_commitment": not any(systematic.values())}
    return {"overall": overall, "by_action": actions, "by_phase": phases,
            "invalid_reasons_all_attempts": dict(sorted(reasons.items())),
            "invalid_reasons_final": dict(sorted(final_reasons.items())),
            "parsed_attempt_count": len(parsed), "perceived_action_identities": identities,
            "final_extra_commitment_by_action": dict(sorted(extra.items())),
            "systematic_extra_commitment_by_action": systematic,
            "gate": {"passed": all(gate_checks.values()), "checks": gate_checks,
                     "phase_rates_are_diagnostic_only": True,
                     "extra_commitment_rule": "final-perception mismatch in >=30% of cases within any action"}}


def execute(cases: tuple[Case, ...], *, q_by_pre, mapper, actor, perceiver) -> tuple[list[dict], dict]:
    """Call the frozen pipeline once per selected case; no fallback sampling."""
    if len(cases) != 58:
        raise SmokeStudyError("execution requires the fixed 58-case selection")
    parameters = tuple(inspect.signature(perceiver.perceive).parameters.values())
    plan_isolation_verified = (len(parameters) == 2 and all(p.kind in (
        inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for p in parameters))
    if not plan_isolation_verified:
        raise SmokeStudyError("perceiver interface must accept only text and public context")
    rows = []
    for case in cases:
        if case.pre is None:
            raise SmokeStudyError("selected case lacks its public PRE")
        plan = make_plan(case, q_by_pre, mapper)
        public = public_language_context_from_pre(case.pre, case.context)
        result = realize_verify_action(plan, case.context, public, actor=actor, perceiver=perceiver)
        if not result.audit.structured_execution_valid or len(result.audit.attempts) not in (1, 2):
            raise SmokeStudyError("frozen language pipeline audit invariant failed")
        rows.append({**case.to_record(), "plan": plan.to_record(), "audit": result.audit.to_record()})
    return rows, aggregate(rows, plan_isolation_verified=plan_isolation_verified)


def source_provenance() -> dict:
    """Freeze a clean tracked source snapshot; unrelated untracked files are allowed."""
    for label, command in (("tracked working tree", ("git", "diff", "--quiet", "--exit-code", "--")),
                           ("staged tracked changes", ("git", "diff", "--cached", "--quiet", "--exit-code", "--"))):
        result = subprocess.run(command, cwd=REPOSITORY, capture_output=True, text=True, check=False)
        if result.returncode == 1:
            raise SmokeStudyError(f"formal smoke requires clean {label}")
        if result.returncode != 0:
            raise SmokeStudyError(f"cannot verify {label}: {result.stderr.strip()}")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=REPOSITORY, text=True).strip()
    return {"commit": head, "branch": branch, "tracked_worktree_clean": True,
            "staged_tracked_changes": False,
            "source_sha256": {name: sha256_bytes((REPOSITORY / name).read_bytes()) for name in SOURCE_FILES}}


def publish_smoke(destination: Path, *, cases: tuple[Case, ...], rows: list[dict], metrics: dict,
                  model: dict, mapper, source: dict) -> str:
    if os.path.lexists(destination):
        raise SmokeStudyError(f"smoke destination already exists: {destination}")
    selected = [case.to_record() for case in cases]
    manifest = {"artifact_type": "phase2_language_execution_smoke",
                "schema_version": VERSION, "study_name": NAME,
                "source": source, "model_deployment": model,
                "publication_digest": offline.PUBLICATION_DIGEST,
                "role_sidecar_digest_legality_only": offline.ROLE_SIDECAR_DIGEST,
                "oof_seal_digest_redirect_only": offline.OOF_SEAL_DIGEST,
                "action_contract_version": CONTRACT_VERSION,
                "language_semantic_version": LANGUAGE_VERSION,
                "language_audit_version": LANGUAGE_AUDIT_VERSION,
                "final_mapper_manifest_digest": mapper.artifact_digest,
                "final_mapper_model_digest": mapper.model_digest,
                "case_selection": {"seed": SEED,
                                   "rule": "SHA256(canonical JSON [seed, identity]); reserve PK Probe first; unique PRE globally",
                                   "layout": [{"phase": p, "action": a.value, "count": n} for p, a, n in LAYOUT],
                                   "digest": selection_digest(cases), "count": len(cases)},
                "semantic_boundary": {"future_outcome_in_selection": False,
                                      "future_outcome_in_model_prompt": False,
                                      "role_sidecar_in_model_prompt": False,
                                      "suspicion_support_in_model_prompt": False,
                                      "perceiver_receives_requested_plan": False,
                                      "hidden_team_used_for_legality_only": True}}
    artifact = publish_artifact(destination, manifest_fields=manifest,
        files={"selected_cases.json": canonical_json_bytes(selected),
               "case_executions.jsonl": canonical_jsonl_bytes(rows),
               "metrics.json": canonical_json_bytes(metrics)})
    return artifact.manifest_digest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        "Requires the frozen 1500-game publication, sealed Qwen3 OOF evaluation, "
        "full M3 mapper, configs/server.json, local Qwen3.5-9B vLLM at "
        "127.0.0.1:8000, and the configured client/vLLM environments. "
        "See docs/research/phase2-language-smoke-v3-runbook.md."))
    parser.add_argument("--publication", type=Path, default=PUBLICATION)
    parser.add_argument("--evaluation-root", type=Path, default=EVALUATION)
    parser.add_argument("--mapper", type=Path, default=MAPPER)
    parser.add_argument("--storage-profile", type=Path, default=REPOSITORY / "configs/server.json")
    parser.add_argument("--preflight", action="store_true", help="validate and print selected cases; no LLM call or artifact")
    args = parser.parse_args(argv)
    frozen_source = None if args.preflight else source_provenance()
    publication = open_publication(args.publication)
    sidecar = open_role_sidecar(publication)
    cases = select_cases(collect_eligible(publication, sidecar))
    selected_digest = selection_digest(cases)
    if selected_digest != EXPECTED_CASE_SELECTION_DIGEST:
        raise SmokeStudyError("smoke-v3 cases differ from the sealed smoke-v1/v2 selection")
    from werewolf.phase2_mapper_runtime import load_runtime_mapper
    runtime_mapper = load_runtime_mapper(args.mapper, expected_manifest_digest=MAPPER_DIGEST)
    q_by_pre = selected_oof_q(publication, cases, args.evaluation_root)
    # Construct every plan before a first LLM call; any Q/mapper error cannot create a partial run.
    for case in cases:
        make_plan(case, q_by_pre, runtime_mapper)
    print(json.dumps({"case_count": len(cases), "case_selection_digest": selected_digest,
                      "layout": [{"phase": p, "action": a.value, "count": n} for p, a, n in LAYOUT],
                      "mapper_digest": runtime_mapper.artifact_digest}, sort_keys=True))
    if args.preflight:
        return 0
    from scripts import collect_games as operator
    from werewolf.backends.factory import load_named_backends
    import yaml
    model, base_url = operator.inspect_inputs()
    operator.live_preflight(base_url, model["served_model_name"])
    config = yaml.safe_load(operator.RUNTIME.read_bytes())
    backend = load_named_backends(config, env_file=None, max_retries=0)["local_qwen"]
    actor = Phase2LanguageActorV1(backend, model["served_model_name"])
    perceiver = Phase2SemanticPerceiverV1(backend, model["served_model_name"])
    if type(perceiver) is not Phase2SemanticPerceiverV1:
        raise SmokeStudyError("untrusted perceiver may receive plan")
    profile = json.loads(args.storage_profile.read_bytes())
    if set(profile) != {"artifact_root"} or not isinstance(profile["artifact_root"], str):
        raise SmokeStudyError("invalid storage profile")
    destination = Path(profile["artifact_root"]) / "paper-studies" / "language-smoke" / NAME
    if os.path.lexists(destination):
        raise SmokeStudyError(f"smoke destination already exists: {destination}")
    rows, metrics = execute(cases, q_by_pre=q_by_pre, mapper=runtime_mapper,
                            actor=actor, perceiver=perceiver)
    digest = publish_smoke(destination, cases=cases, rows=rows, metrics=metrics,
                           model=model, mapper=runtime_mapper, source=frozen_source)
    print(json.dumps({"artifact": str(destination), "manifest_digest": digest,
                      "metrics": metrics}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
