"""Shared canonical server assembly for terminal and Probe strategy pilots."""

from __future__ import annotations

from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

import yaml

from werewolf.artifact_io import canonical_json_bytes, sha256_bytes, verify_artifact
from werewolf.artifact_io.canonical import ensure_durable_directory
from werewolf.backends.factory import load_named_backends
from werewolf.canonical_collection.attempt_ledger import (
    _claim_from_record, _publish_bytes_noreplace, collection_plan_from_record,
    construct_attempt_claim,
)
from werewolf.canonical_collection.failure_evidence import (
    construct_canonical_partial_evidence, publish_canonical_partial_evidence,
    validate_canonical_partial_evidence,
)
from werewolf.canonical_collection.game_bundle import (
    _backend_call_from_record, _prefix_from_record,
)
from werewolf.canonical_collection.production_runtime import Classic7RuntimeFactory
from werewolf.canonical_collection.public_history import freeze_public_event_history
from werewolf.phase2_mapper_runtime import load_runtime_mapper
from werewolf.phase2_offline import build_development_layer, classify_outcome
from werewolf.phase2_online_ledger import OnlinePilotAssignmentLedgerV1
from werewolf.phase2_online_plan import (
    PROBE_PLAN_VERSION, Phase2OnlineProbePilotPlanV1, Phase2OnlineTerminalPilotPlanV1,
    probe_opportunity_from_record,
)
from werewolf.phase2_online_preflight import (
    FrozenArtifactRequirement, _smoke_v3_gate, assess_pilot_preflight,
    freeze_online_source_provenance, verify_probe_plan_provenance,
)
from werewolf.phase2_online_runner import (
    ARTIFACT_NAME, QUALIFICATION_NAME, PROBE_ARTIFACT_NAME, PROBE_QUALIFICATION_NAME,
    OnlinePilotRuntimeBundleV1,
    OnlineTerminalPilotRunnerV1, build_online_dataset, publish_online_pilot,
    run_online_campaign,
)
from werewolf.phase2_outcome import reference_tables_digest
from werewolf.phase2_pilot_dataset import analyze_phase2_online_support_record
from werewolf.runtime_config import normalize_runtime_config
from scripts.qwen3_gameplay_predictor import Qwen3GameplayPredictorClient, FIT_DIGEST, SEAL_DIGEST


INPUT_VERSION = "phase2_online_server_inputs_v1"


def _read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(Path(path).read_bytes(), object_pairs_hook=unique)


def _publish_json(path, record):
    """Reuse the canonical durable no-replace primitive for running inputs/claims."""
    ensure_durable_directory(path.parent)
    staging = path.parent / ".staging"
    ensure_durable_directory(staging)
    _publish_bytes_noreplace(staging_directory=staging, final_path=path,
                           durable_directory=path.parent, data=canonical_json_bytes(record))


def load_online_plan(path, purpose):
    raw = _read_json(path)
    if raw.get("schema_version") == PROBE_PLAN_VERSION:
        fields = ("pilot_id", "assignment_seed", "campaign_purpose", "target_assignment_count",
                  "candidate_selection_rule", "selection_status", "max_games_attempted")
        plan = Phase2OnlineProbePilotPlanV1(**{name: raw[name] for name in fields})
    else:
        fields = ("pilot_id", "assignment_seed", "campaign_purpose", "push_probability",
                  "candidate_selection_rule", "selection_status", "max_games_attempted")
        plan = Phase2OnlineTerminalPilotPlanV1(**{name: raw[name] for name in fields})
    if raw != plan.to_record() or plan.campaign_purpose != purpose:
        raise ValueError("campaign purpose or frozen Pilot-T plan differs")
    for identity in (plan.pilot_id,):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", identity) or ".." in identity:
            raise ValueError("campaign id must be one safe path component")
    return plan


def _require_package_source(repo):
    if Path(__file__).resolve().parents[1] != repo:
        raise ValueError("installed Pilot-T package must originate in --repo")


def _verify_smoke_source(repo, artifact):
    """Additional CLI lineage check; never relax the existing smoke gate."""
    from scripts.run_phase2_language_smoke import SOURCE_FILES
    source = artifact.manifest["source"]
    if (not re.fullmatch(r"[0-9a-f]{40}", source["commit"])
            or source.get("tracked_worktree_clean") is not True
            or source.get("staged_tracked_changes") is not False
            or any(source["source_sha256"].get(name) !=
                   sha256_bytes((repo / name).read_bytes()) for name in SOURCE_FILES)):
        raise ValueError("Smoke-V3 source bytes differ; rerun Smoke-V3 on the current source")
    return source


def load_server_inputs(args):
    """Read and pin all inputs before a worker or any language call is started."""
    if os.path.lexists(args.destination):
        raise FileExistsError("Pilot-T artifact destination already exists")
    args.repo = args.repo.resolve()
    _require_package_source(args.repo)
    q_python = shutil.which(args.q_python)
    if q_python is None:
        raise ValueError("Q worker Python executable is unavailable")
    args.q_python = str(Path(q_python).resolve())
    args.destination = args.destination.resolve()
    args.work_directory = args.work_directory.resolve()
    if (args.destination.is_relative_to(args.work_directory)
            or args.work_directory.is_relative_to(args.destination)):
        raise ValueError("work directory and formal destination must be disjoint")
    plan = load_online_plan(args.plan, args.campaign_purpose)
    if isinstance(plan, Phase2OnlineProbePilotPlanV1):
        expected_name = (PROBE_QUALIFICATION_NAME if args.campaign_purpose == "qualification"
                         else PROBE_ARTIFACT_NAME)
    else:
        expected_name = QUALIFICATION_NAME if args.campaign_purpose == "qualification" else ARTIFACT_NAME
    if args.destination.name != expected_name:
        raise ValueError("qualification and pilot require their distinct fixed artifact names")
    source = freeze_online_source_provenance(args.repo)
    if source is None or source["commit"] != args.source_commit:
        raise ValueError("tracked source/index must be clean at the preregistered source commit")
    game_plan = collection_plan_from_record(_read_json(args.game_plan))
    if isinstance(plan, Phase2OnlineProbePilotPlanV1):
        purpose = "qualification" if plan.campaign_purpose == "qualification" else "formal"
        profile = _read_json(args.repo / f"configs/phase2/online-probe-{purpose}-v1.json")
        verify_probe_plan_provenance(args.repo, plan, game_plan, profile)
        # Raw server CLI and operator CLI share the same admission/overlap checks.
        from scripts.phase2_online_campaign import probe_inputs, canonical_plan, checked_seed_overlaps
        pins, qualified, _, excluded = probe_inputs(profile)
        if game_plan != canonical_plan(plan, args.source_commit, qualified, profile=profile):
            raise ValueError("Probe canonical plan differs from frozen runtime/profile")
        if any(Path(profile[key]).resolve() != Path(getattr(args, key)).resolve()
               for key in ("plan", "game_plan", "work_directory", "destination")):
            raise ValueError("Probe server paths differ from frozen profile")
        for key in ("runtime_config", "deployment_config", "publication", "evaluation_root",
                    "mapper", "smoke_v3", "q_checkout", "q_fit"):
            if Path(getattr(args, key)).resolve() != Path(qualified["paths"][key]).resolve():
                raise ValueError(f"Probe runtime path differs from qualification: {key}")
        for key in ("mapper_manifest_digest", "reference_tables_digest", "smoke_v3_manifest_digest", "q_python"):
            if getattr(args, key) != qualified[key]:
                raise ValueError(f"Probe runtime pin differs from qualification: {key}")
        checked_seed_overlaps(game_plan, pins, qualified, excluded)
    if (game_plan.source_revision != args.source_commit
            or game_plan.collection_id != plan.pilot_id
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", game_plan.collection_id)
            or ".." in game_plan.collection_id
            or len(game_plan.ordered_seed_pool) < plan.target_assignment_count
            or plan.max_games_attempted is None
            or plan.max_games_attempted > len(game_plan.ordered_seed_pool)):
        raise ValueError("canonical game plan source/seed pool or frozen safety cap differs")
    config = normalize_runtime_config(yaml.safe_load(args.runtime_config.read_bytes()))
    deployment_bytes = args.deployment_config.read_bytes()
    deployment = yaml.safe_load(deployment_bytes)
    provenance = dict(game_plan.environment_provenance)
    config_digest = sha256_bytes(canonical_json_bytes(config))
    if (provenance.get("runtime_config_sha256") != config_digest
            or provenance.get("serve_config_sha256") != sha256_bytes(deployment_bytes)):
        raise ValueError("runtime/deployment config differs from frozen canonical game plan")
    call_limit = int(provenance["configured_call_limit"])
    if call_limit <= 0 or game_plan.call_budget_identity != f"classic7-backend-dispatch-cap-{call_limit}-v1":
        raise ValueError("canonical game plan call budget differs")
    from scripts.collect_games import plan_fields
    extra = {key: value for key, value in provenance.items() if key not in (
        "configured_call_limit", "gameplay_prompt_profile", "seed_pool_size", "target_success_count")}
    expected_fields = plan_fields({"collection_id": game_plan.collection_id,
        "target_games": game_plan.target_canonical_success_count,
        "seed_pool_size": len(game_plan.ordered_seed_pool), "call_limit": call_limit},
        args.source_commit, extra)
    if any((dict(game_plan.environment_provenance) if key == "environment_provenance" else
            getattr(game_plan, key)) != value for key, value in expected_fields.items()):
        raise ValueError("canonical runtime/prompt/retry provenance differs from production plan")
    served = deployment["served-model-name"]
    base_url = f"http://{deployment['host']}:{deployment['port']}/v1"
    if (deployment["host"] != "127.0.0.1"
            or deployment["language-model-only"] is not True
            or deployment["default-chat-template-kwargs"]["enable_thinking"] is not False
            or deployment["max-num-seqs"] != 1 or deployment["enforce-eager"] is not True
            or served != game_plan.model_identity
            or config["parser"]["model"] != served
            or any(profile["model"] != served for profile in config["agent_config"]["all_candidates"])
            or any(backend["base_url"] != base_url or backend["default_model"] != served
                   for backend in config["backends"].values())):
        raise ValueError("runtime/model identities do not match the frozen loopback deployment")
    # Reference values are rebuilt by the existing sealed-data loader, never refitted.
    mapper = load_runtime_mapper(args.mapper, expected_manifest_digest=args.mapper_manifest_digest)
    reference = build_development_layer(args.publication, args.evaluation_root).values
    if reference_tables_digest(reference) != args.reference_tables_digest:
        raise ValueError("preregistered reference table digest mismatch")
    if not _smoke_v3_gate(args.smoke_v3, args.smoke_v3_manifest_digest,
                          args.mapper_manifest_digest):
        raise ValueError("Smoke-V3 manifest, mapper lineage, selection, or gate mismatch")
    from scripts.run_phase2_language_smoke import VERSION as SMOKE_VERSION
    smoke_artifact = verify_artifact(args.smoke_v3,
        expected_artifact_type="phase2_language_execution_smoke", expected_schema_version=SMOKE_VERSION)
    if smoke_artifact.manifest_digest != args.smoke_v3_manifest_digest:
        raise ValueError("Smoke-V3 manifest changed during input validation")
    smoke_source = _verify_smoke_source(args.repo, smoke_artifact)
    bound = {
        "schema_version": INPUT_VERSION, "pilot_plan": plan.to_record(),
        "canonical_game_plan": game_plan.to_record(), "source_commit": args.source_commit,
        "runtime_config_sha256": config_digest,
        "deployment_config_sha256": sha256_bytes(deployment_bytes),
        "mapper_manifest_digest": args.mapper_manifest_digest,
        "reference_tables_digest": args.reference_tables_digest,
        "smoke_v3_manifest_digest": args.smoke_v3_manifest_digest,
        "smoke_v3_source": smoke_source,
        "q_fit_digest": FIT_DIGEST, "q_seal_digest": SEAL_DIGEST,
        "paths": {name: str(getattr(args, name).resolve()) for name in (
            "repo", "runtime_config", "deployment_config", "publication", "evaluation_root",
            "mapper", "smoke_v3", "q_checkout", "q_fit", "work_directory", "destination")},
        "q_python": str(Path(args.q_python).resolve()),
    }
    bound["inputs_digest"] = sha256_bytes(canonical_json_bytes(bound))
    if args.resume:
        if (not args.work_directory.is_dir()
                or not (args.work_directory / "assignment-ledger.jsonl").is_file()
                or _read_json(args.work_directory / "run_inputs.json") != bound):
            raise ValueError("--resume requires the exact existing campaign inputs and ledger")
    elif os.path.lexists(args.work_directory):
        raise FileExistsError("existing campaign work directory requires explicit --resume")
    return plan, game_plan, config, call_limit, mapper, reference, bound


class ServerRuntimeFactory:
    """Compose the canonical factory, Pilot-T object and the sole full preflight."""

    def __init__(self, *, args, plan, game_plan, config, call_limit, mapper, reference,
                 predictor, backends, ledger, work_directory):
        self.args, self.plan, self.game_plan = args, plan, game_plan
        self.config, self.mapper, self.reference = config, mapper, reference
        self.predictor, self.ledger, self.work_directory = predictor, ledger, work_directory
        self.canonical = Classic7RuntimeFactory(
            runtime_config=config, backends=backends, configured_call_limit=call_limit)
        self.game_ids = tuple(f"{game_plan.collection_id}-game-{ordinal:06d}-seed-{seed}"
                              for ordinal, seed in enumerate(game_plan.ordered_seed_pool))
        self.evidence_digests = {}

    def _claim(self, game_id, *, persist):
        ordinal = self.game_ids.index(game_id)
        path = self.work_directory / "claims" / f"{game_id}.json"
        if path.is_file():
            return _claim_from_record(_read_json(path), self.game_plan)
        claim = construct_attempt_claim(
            self.game_plan, ordinal=ordinal,
            attempt_id=f"attempt-{ordinal:06d}-seed-{self.game_plan.ordered_seed_pool[ordinal]}",
            claim_timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"))
        if persist:
            _publish_json(path, {"record_type": "claim", **claim.to_record()})
        return claim

    def __call__(self, game_id, *, persist_claim=True, publication_only=False):
        claim = self._claim(game_id, persist=persist_claim)
        runtime = self.canonical(plan=self.game_plan, claim=claim)
        def record_evidence(stage, record):
            self.persist_stage_evidence(game_id, claim, runtime.recorder, stage, record)
        pilot = OnlineTerminalPilotRunnerV1(
            plan=self.plan, predictor=self.predictor, mapper=self.mapper,
            backend=runtime.env.speech_perceiver.backend,
            model_name=self.config["parser"]["model"], reference_tables=self.reference,
            reference_artifact_digest=self.args.reference_tables_digest,
            ledger=self.ledger, record_evidence=record_evidence)
        from werewolf.phase2_mapper_final import FINAL_VERSION
        from scripts.run_phase2_language_smoke import VERSION as SMOKE_VERSION
        preflight = assess_pilot_preflight(self.args.repo,
            frozen_artifacts=(FrozenArtifactRequirement(self.args.mapper,
                "phase2_full_development_oof_mapper", FINAL_VERSION, self.args.mapper_manifest_digest),
                FrozenArtifactRequirement(self.args.smoke_v3,
                    "phase2_language_execution_smoke", SMOKE_VERSION, self.args.smoke_v3_manifest_digest)),
            expected_source_commit=self.args.source_commit,
            expected_reference_tables_digest=self.args.reference_tables_digest,
            mapper_artifact=self.args.mapper, mapper_manifest_digest=self.args.mapper_manifest_digest,
            mapper_runtime=self.mapper, smoke_v3_artifact=self.args.smoke_v3,
            smoke_v3_manifest_digest=self.args.smoke_v3_manifest_digest,
            online_plan=self.plan, predictor=self.predictor, backend=pilot.backend,
            call_audit=runtime.call_audit, ledger=self.ledger, env=runtime.env,
            recorder=runtime.recorder, reference_tables=self.reference,
            reference_artifact_digest=self.args.reference_tables_digest,
            destination=self.args.destination, publication_only=publication_only)
        return OnlinePilotRuntimeBundleV1(runtime.env, tuple(runtime.agents), tuple(runtime.roles),
                                         runtime.recorder, runtime.call_audit, pilot, preflight)

    def persist_stage_evidence(self, game_id, claim, recorder, stage, record):
        evidence = construct_canonical_partial_evidence(
            plan=self.game_plan, claim=claim, evidence_id=f"phase2-{stage}",
            evidence_type="phase2_online_canonical_stage_v1",
            payload={"game_id": game_id, "stage": stage,
                     "phase2_record": record.to_record(),
                     "canonical_runtime": recorder._failure_partial_payload()})
        publish_canonical_partial_evidence(self.work_directory,
            plan=self.game_plan, claim=claim, evidence=evidence)

    def _stage_evidence(self, game_id, stage):
        claim = self._claim(game_id, persist=False)
        path = self.work_directory / "attempts" / claim.attempt_id / "partial_evidence" / f"phase2-{stage}.json"
        verified = validate_canonical_partial_evidence(path, plan=self.game_plan,
            claim=claim, collection_directory=self.work_directory)
        self.evidence_digests[path.relative_to(self.work_directory).as_posix()] = verified.file_sha256
        payload = verified.evidence.payload.to_value()
        if payload["game_id"] != game_id or payload["stage"] != stage:
            raise ValueError("canonical stage evidence game/stage mismatch")
        return payload

    def verify_execution(self, game_id, stages):
        """Validate durable canonical calls, committed text and actual day outcome."""
        if "assigned_strategy" in stages["ASSIGNMENT"]["assignment"]:
            return self._verify_probe_execution(game_id, stages)
        proof = self._stage_evidence(game_id, "execution")
        assigned = stages["ASSIGNMENT"]["assignment"]
        execution = stages["EXECUTION"]["execution"]
        record, canonical = proof["phase2_record"], proof["canonical_runtime"]
        if record["assignment"] != assigned or record["execution"] != execution:
            raise ValueError("canonical execution proof differs from ledger")
        identity = assigned["opportunity"]["identity"]
        prefix = next(_prefix_from_record(row) for row in canonical["authoritative_pre_prefixes"]
                      if row["boundary_id"] == identity["boundary_id"])
        history = freeze_public_event_history(canonical["public_events"])
        if (prefix.game_id != game_id or prefix.prefix_digest != identity["prefix_digest"]
                or prefix.current_speaker != identity["acting_wolf"]
                or history.to_records()[:len(prefix.public_event_history.events)] !=
                   prefix.public_event_history.to_records()):
            raise ValueError("canonical execution PRE differs")
        calls = stages["EXECUTION"]["backend_calls"]
        if sha256_bytes(canonical_json_bytes(calls)) != execution["backend_call_audit_digest"]:
            raise ValueError("execution backend audit digest differs")
        canonical_calls = {row["call_id"]: _backend_call_from_record(row)
                           for row in canonical["backend_calls"]}
        for call in calls:
            actual = canonical_calls[call["canonical_call_id"]]
            private = actual.private_payload.to_value()
            if (actual.boundary_id != identity["boundary_id"]
                    or actual.observer_id != identity["acting_wolf"]
                    or call["request_digest"] != sha256_bytes(canonical_json_bytes(private["kwargs"]))
                    or call["response_digest"] != (None if private["response"] is None else
                        sha256_bytes(canonical_json_bytes(private["response"])))):
                raise ValueError("Phase-2 backend sidecar differs from canonical dispatch")
        if execution["success"]:
            event = next(row for row in history.to_records()
                         if row["event_id"] == execution["canonical_event_id"])
            # The commit link hashes the original env event, before history
            # decoding adds its derived day/phase fields.
            committed_event = {key: event[key] for key in (
                "event_id", "event_index", "event_type", "speaker", "raw_text")}
            if (event["event_type"] != "public_speech"
                    or event["speaker"] != identity["acting_wolf"]
                    or event["event_index"] != len(prefix.public_event_history.events)
                    or sha256_bytes(canonical_json_bytes(committed_event)) != execution["canonical_event_digest"]
                    or sha256_bytes(event["raw_text"].encode("utf-8")) != execution["generated_text_digest"]):
                raise ValueError("canonical committed speech differs from execution")
        elif history.to_records() != prefix.public_event_history.to_records():
            raise ValueError("failed execution cannot claim a new public event")
        if "CONSEQUENCE" in stages:
            proof = self._stage_evidence(game_id, "consequence")
            day = stages["CONSEQUENCE"]["day_consequence"]
            if (proof["phase2_record"]["assignment"] != assigned
                    or proof["phase2_record"]["day_consequence"] != day):
                raise ValueError("canonical day proof differs from ledger")
            events = freeze_public_event_history(proof["canonical_runtime"]["public_events"]).events
            exiles = [event for event in events if event.event_type == "exile_result"
                      and event.temporal_state.day == prefix.public_temporal_state.day]
            if not exiles or len(exiles[-1].affected_players) > 1:
                raise ValueError("canonical day resolution is missing")
            if (exiles[-1].event_index + 1 < len(events)
                    and events[exiles[-1].event_index + 1].event_type == "phase_change"
                    and events[exiles[-1].event_index + 1].temporal_state.phase.value == "pk_discussion"):
                raise ValueError("intermediate PK tie is not a day consequence")
            exile = exiles[-1].affected_players[0] if exiles[-1].affected_players else None
            legal = assigned["opportunity"]["public_legal"]
            outcome = classify_outcome(exile, acting_wolf=identity["acting_wolf"],
                candidate_j=identity["candidate_j"], wolves=frozenset(legal["known_wolves"]))
            loss = self.reference.deployment_loss(tuple(assigned["opportunity"]["s_pre"]), outcome)
            expected = {"exiled_player": exile, "Y": outcome.value, "s_plus": list(loss.s_post),
                        "v_ref": loss.value.value, "l_ref": loss.loss,
                        "reference_artifact_digest": self.args.reference_tables_digest}
            if day != expected or not any(event.event_type == "vote_result"
                    and event.temporal_state.day == prefix.public_temporal_state.day
                    and event.event_index < exiles[-1].event_index for event in events):
                raise ValueError("canonical vote/reference consequence differs")
        return True

    @staticmethod
    def _probe_pre(game_id, opportunity, canonical):
        from werewolf.phase2_actions import context_from_pre

        parsed = probe_opportunity_from_record(opportunity)
        identity = opportunity["identity"]
        prefix = next((_prefix_from_record(row) for row in canonical["authoritative_pre_prefixes"]
                       if row["boundary_id"] == identity["boundary_id"]), None)
        if prefix is None:
            raise ValueError("canonical Probe stage PRE is missing")
        context = context_from_pre(prefix, parsed.legal_context.known_wolves)
        history = freeze_public_event_history(canonical["public_events"])
        if (context != parsed.legal_context or context.game_id != game_id
                or history.to_records()[:len(prefix.public_event_history.events)] !=
                   prefix.public_event_history.to_records()):
            raise ValueError("canonical Probe stage PRE differs")
        return prefix, history

    @staticmethod
    def _verify_probe_stage(game_id, assignment_id, stage, canonical, *, invalid_at_pre=False):
        from werewolf.canonical_collection.trajectory_evidence import BackendCallPurpose

        prefix, history = ServerRuntimeFactory._probe_pre(game_id, stage["opportunity"], canonical)
        identity = stage["opportunity"]["identity"]
        treatment = stage["treatment"]
        calls = stage["backend_calls"]
        expected_actors = [("realization", 1)] + ([("repair", 2)] if stage["attempt_count"] == 2 else [])
        if ([(call["role"], call["attempt_index"]) for call in calls
             if call["role"] in ("realization", "repair")] != expected_actors
                or any(call["role"] not in ("realization", "repair", "perception") for call in calls)
                or (stage["success"] and not any(call["role"] == "perception"
                    and call["attempt_index"] == stage["attempt_count"]
                    and call["response_digest"] is not None and call["error_category"] is None
                    for call in calls))):
            raise ValueError("Probe stage lacks required language dispatch evidence")
        current_attempt = 0
        for call in calls:
            if call["role"] in ("realization", "repair"):
                current_attempt += 1
            if (call["attempt_index"] != current_attempt
                    or call["perception_public_only"] is not (call["role"] == "perception")):
                raise ValueError("Probe language attempt/public perception order differs")
        canonical_calls = {row["call_id"]: _backend_call_from_record(row)
                           for row in canonical["backend_calls"]}
        for call in calls:
            actual = canonical_calls[call["canonical_call_id"]]
            private = actual.private_payload.to_value()
            if (call["game_id"] != game_id or call["assignment_id"] != assignment_id
                    or call["treatment_id"] != treatment["treatment_id"]
                    or call["opportunity_digest"] != treatment["opportunity_digest"]
                    or call["boundary_id"] != identity["boundary_id"]
                    or call["prefix_digest"] != identity["prefix_digest"]
                    or actual.boundary_id != identity["boundary_id"]
                    or actual.observer_id != identity["acting_wolf"]
                    or actual.purpose != BackendCallPurpose.RUNTIME
                    or actual.operation_id != f"phase2-{treatment['treatment_id'][:16]}-{call['sequence']:03d}"
                    or actual.backend_identity != call["backend_identity"]
                    or actual.model_identity != call["model_identity"]
                    or call["request_digest"] != sha256_bytes(canonical_json_bytes(private["kwargs"]))
                    or call["response_digest"] != (None if private["response"] is None else
                        sha256_bytes(canonical_json_bytes(private["response"])))):
                raise ValueError("Probe stage sidecar differs from canonical dispatch")
        if stage["success"]:
            event = next((row for row in history.to_records()
                          if row["event_id"] == stage["canonical_event_id"]), None)
            if event is None:
                raise ValueError("Probe canonical committed speech is missing")
            committed = {key: event[key] for key in (
                "event_id", "event_index", "event_type", "speaker", "raw_text")}
            if (event["event_type"] != "public_speech"
                    or event["speaker"] != identity["acting_wolf"]
                    or event["event_index"] != len(prefix.public_event_history.events)
                    or sha256_bytes(canonical_json_bytes(committed)) != stage["canonical_event_digest"]
                    or sha256_bytes(event["raw_text"].encode("utf-8")) != stage["generated_text_digest"]):
                raise ValueError("Probe canonical committed speech differs")
        elif invalid_at_pre and history.to_records() != prefix.public_event_history.to_records():
            raise ValueError("invalid Probe stage cannot claim a new public event")
        return prefix, history

    def _verify_probe_execution(self, game_id, stages):
        from werewolf.phase2_online_records import validate_probe_lifecycle_record

        assigned = stages["ASSIGNMENT"]["assignment"]
        assignment_id = stages["ASSIGNMENT"]["assignment_id"]
        snapshots = stages.get("STRATEGY_STAGE_HISTORY", [])
        if (not snapshots or snapshots[0]["record"]["lifecycle"] != ["ASSIGNED"]
                or stages.get("STRATEGY_STAGE") != snapshots[-1]):
            raise ValueError("Probe lifecycle history is missing")
        final_stages = snapshots[-1]["record"]["stages"]
        previous = []
        for snapshot in snapshots:
            record = snapshot["record"]
            validate_probe_lifecycle_record(record)
            lifecycle = record["lifecycle"]
            if (snapshot["assignment_id"] != assignment_id or snapshot["game_id"] != game_id
                    or record["assignment"] != assigned
                    or len(lifecycle) != len(previous) + 1 or lifecycle[:-1] != previous):
                raise ValueError("Probe lifecycle assignment/order differs")
            previous = lifecycle
            event = lifecycle[-1]
            if event == "ASSIGNED":
                continue
            name = f"strategy-{len(lifecycle):02d}-{event.lower()}"
            proof = self._stage_evidence(game_id, name)
            if proof["phase2_record"] != record:
                raise ValueError("canonical Probe lifecycle proof differs from ledger")
            canonical = proof["canonical_runtime"]
            self._probe_pre(game_id, assigned["opportunity"], canonical)
            if event in ("T1_ATTEMPTED", "T3_PREPARED"):
                stage_name = "T1" if event == "T1_ATTEMPTED" else "T3"
                stage = final_stages[stage_name]
                actual_ids = {row["call_id"] for row in canonical["backend_calls"]}
                if stage is not None and any(call["canonical_call_id"] in actual_ids
                        for call in stage["backend_calls"]):
                    raise ValueError("Probe language call precedes lifecycle writeahead")
            if record["continuation"]["opportunity"] is not None:
                self._probe_pre(game_id, record["continuation"]["opportunity"], canonical)
            for stage_name, stage in record["stages"].items():
                if stage is not None:
                    self._verify_probe_stage(game_id, assignment_id, stage, canonical,
                        invalid_at_pre=event == f"{stage_name}_LANGUAGE_INVALID")
            if record["observations"]:
                prefix, history = self._probe_pre(game_id, assigned["opportunity"], canonical)
                speech = [row for row in history.to_records()[len(prefix.public_event_history.events) + 1:]
                          if row["event_type"] == "public_speech"]
                if len(speech) < len(record["observations"]) or any(
                        observed["event_id"] != actual["event_id"]
                        or observed["speaker"] != actual["speaker"]
                        or observed["text"] != actual["raw_text"]
                        for observed, actual in zip(record["observations"], speech)):
                    raise ValueError("Probe observations differ from canonical public window")
        proof = self._stage_evidence(game_id, "execution")
        executed = next((snapshot["record"] for snapshot in snapshots
                         if snapshot["record"]["lifecycle"][-1] == "EXECUTION_RECORDED"), None)
        if (executed is None or proof["phase2_record"] != executed
                or executed["execution"] != stages["EXECUTION"]["execution"]):
            raise ValueError("canonical Probe execution proof differs from ledger")
        calls = [call for stage in executed["stages"].values() if stage is not None
                 for call in stage["backend_calls"]]
        if (calls != stages["EXECUTION"]["backend_calls"]
                or calls != [row["call"] for row in stages.get("BACKEND_CALL", [])]
                or [call["sequence"] for call in calls] != list(range(1, len(calls) + 1))
                or len({call["canonical_call_id"] for call in calls}) != len(calls)
                or sha256_bytes(canonical_json_bytes(calls)) != executed["execution"]["backend_call_audit_digest"]):
            raise ValueError("Probe execution backend evidence differs")
        for stage in executed["stages"].values():
            if stage is not None:
                self._verify_probe_stage(game_id, assignment_id, stage, proof["canonical_runtime"],
                    invalid_at_pre=stage["success"] is False)
        if "CONSEQUENCE" in stages:
            proof = self._stage_evidence(game_id, "consequence")
            day_record = next((snapshot["record"] for snapshot in snapshots
                               if snapshot["record"]["lifecycle"][-1] == "DAY_CONSEQUENCE_RECORDED"), None)
            day = stages["CONSEQUENCE"]["day_consequence"]
            if (day_record is None or proof["phase2_record"] != day_record
                    or day_record["day_consequence"] != day):
                raise ValueError("canonical Probe day proof differs from ledger")
            prefix, history = self._probe_pre(game_id, assigned["opportunity"], proof["canonical_runtime"])
            events = history.events
            exiles = [event for event in events if event.event_type == "exile_result"
                      and event.temporal_state.day == prefix.public_temporal_state.day]
            if not exiles or len(exiles[-1].affected_players) > 1:
                raise ValueError("canonical Probe day resolution is missing")
            if (exiles[-1].event_index + 1 < len(events)
                    and events[exiles[-1].event_index + 1].event_type == "phase_change"
                    and events[exiles[-1].event_index + 1].temporal_state.phase.value == "pk_discussion"):
                raise ValueError("intermediate PK tie is not a Probe day consequence")
            exile = exiles[-1].affected_players[0] if exiles[-1].affected_players else None
            identity, legal = assigned["opportunity"]["identity"], assigned["opportunity"]["public_legal"]
            outcome = classify_outcome(exile, acting_wolf=identity["acting_wolf"],
                candidate_j=identity["candidate_j"], wolves=frozenset(legal["known_wolves"]))
            loss = self.reference.deployment_loss(tuple(assigned["opportunity"]["s_pre"]), outcome)
            expected = {"exiled_player": exile, "Y": outcome.value, "s_plus": list(loss.s_post),
                        "v_ref": loss.value.value, "l_ref": loss.loss,
                        "reference_artifact_digest": self.args.reference_tables_digest}
            if day != expected or not any(event.event_type == "vote_result"
                    and event.temporal_state.day == prefix.public_temporal_state.day
                    and event.event_index < exiles[-1].event_index for event in events):
                raise ValueError("canonical Probe vote/reference consequence differs")
        if "GAME_RESULT" in stages:
            proof = self._stage_evidence(game_id, "game-result")
            if (proof["phase2_record"] != snapshots[-1]["record"]
                    or proof["phase2_record"]["offline_audit"]["final_game_result"] != stages["GAME_RESULT"]["winner"]):
                raise ValueError("canonical Probe final game audit differs")
        return True


def execute_server_campaign(args) -> int:
    plan, game_plan, config, call_limit, mapper, reference, bound = load_server_inputs(args)
    with ExitStack() as stack:
        # Startup verifies the pinned worker checkout/fit/seal; predict is not called by preflight.
        predictor = stack.enter_context(Qwen3GameplayPredictorClient(
            checkout=args.q_checkout, fit_path=args.q_fit, python_executable=args.q_python))
        backends = load_named_backends(config, env_file=None, max_retries=0)
        for backend in backends.values():
            stack.callback(backend.client.close)
        def factory_at(work):
            ledger = OnlinePilotAssignmentLedgerV1(work / "assignment-ledger.jsonl",
                plan=plan, source_commit=args.source_commit)
            factory = ServerRuntimeFactory(args=args, plan=plan, game_plan=game_plan,
                config=config, call_limit=call_limit, mapper=mapper, reference=reference,
                predictor=predictor, backends=backends, ledger=ledger, work_directory=work)
            return ledger, factory
        # Always certify a full runtime before creating/changing the durable campaign.
        with tempfile.TemporaryDirectory(prefix="phase2-online-preflight-") as temp:
            work = Path(temp)
            if args.resume:
                shutil.copyfile(args.work_directory / "assignment-ledger.jsonl",
                                work / "assignment-ledger.jsonl")
            ledger, factory = factory_at(work)
            ledger.mark_interrupted_on_resume()
            publication_only = ledger.status() == "READY_TO_SEAL"
            game_id = next((game for game in factory.game_ids
                            if game not in ledger.snapshot()["games"]), factory.game_ids[0])
            preview = factory(game_id, persist_claim=False, publication_only=publication_only)
            print(json.dumps({"preflight": preview.preflight.to_record(),
                              "campaign_purpose": plan.campaign_purpose,
                              "target_assignment_count": plan.target_assignment_count}, sort_keys=True), flush=True)
            if not preview.preflight.ready:
                return 1
            if args.preflight_only:
                return 0
        if not args.resume:
            args.work_directory.mkdir(parents=True, exist_ok=False, mode=0o700)
            _publish_json(args.work_directory / "run_inputs.json", bound)
        ledger, factory = factory_at(args.work_directory)
        ledger.mark_interrupted_on_resume()
        game_id = next((game for game in factory.game_ids
                        if game not in ledger.snapshot()["games"]), factory.game_ids[0])
        first = factory(game_id, persist_claim=False,
                        publication_only=ledger.status() == "READY_TO_SEAL")
        if not first.preflight.ready:
            return 1
        if ledger.status() == "RUNNING":
            result = run_online_campaign(factory.game_ids, factory, ledger=ledger)
        else:
            result = {"status": ledger.status(), "assignment_count": ledger.snapshot()["assignment_count"]}
        if not ledger.sealable():
            print(json.dumps({"campaign": result, "published": False}, sort_keys=True), flush=True)
            return 1
        # The ledger is authoritative even when earlier games ran in another process.
        dataset = build_online_dataset(first.pilot, source_commit=args.source_commit,
                                       execution_verifier=factory.verify_execution)
        support = analyze_phase2_online_support_record(dataset.to_record())
        digest = publish_online_pilot(args.destination, pilot=first.pilot, dataset=dataset,
            preflight=first.preflight, server_run_provenance={
                "inputs_digest": bound["inputs_digest"], "game_plan_digest": game_plan.plan_digest,
                "runtime_config_sha256": bound["runtime_config_sha256"],
                "deployment_config_sha256": bound["deployment_config_sha256"],
                "smoke_v3_source": bound["smoke_v3_source"],
                "work_directory": str(args.work_directory),
                "canonical_stage_evidence": dict(sorted(factory.evidence_digests.items()))})
        print(json.dumps({"artifact": str(args.destination), "manifest_digest": digest,
                          "campaign_purpose": plan.campaign_purpose, "support": support},
                         sort_keys=True), flush=True)
        return 0
