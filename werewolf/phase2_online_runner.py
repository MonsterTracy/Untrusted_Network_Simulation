"""Explicit opt-in, single-path terminal and Probe pilots; no checkpoint replay.

Production orchestration shared by the server CLI and scripted tests.
Default gameplay does not import or enable this module.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import inspect
from pathlib import Path
import os

from run_random import eval as run_canonical_game
from werewolf.phase2_execution import prepare_phase2_intervention_speech
from werewolf.artifact_io import (
    canonical_json_bytes, canonical_jsonl_bytes, publish_artifact, sha256_bytes,
)
from werewolf.phase2_actions import Action, context_from_pre
from werewolf.phase2_backend_audit import Phase2BackendCallAuditV1
from werewolf.phase2_canonical_commit import commit_phase2_verified_speech
from werewolf.phase2_decision_opportunity import build_phase2_decision_opportunity
from werewolf.phase2_intervention import assess_phase2_checkpoint_closure
from werewolf.phase2_language import (
    Phase2LanguageActorV1, Phase2SemanticPerceiverV1,
    public_language_context_from_pre,
)
from werewolf.phase2_online_plan import (
    OnlinePilotGameStateV1, Phase2OnlineTerminalPilotPlanV1,
    select_candidate, select_probe_candidate, Phase2OnlineProbePilotPlanV1, ProbeStrategy,
)
from werewolf.phase2_online_ledger import OnlinePilotAssignmentLedgerV1
from werewolf.phase2_online_records import (
    Phase2OnlineInterventionRecordV1, record_online_day_outcome,
    record_online_execution, Phase2OnlineProbeRecordV1, Phase2StrategyStageExecutionV1,
    validate_probe_lifecycle_record,
)
from werewolf.phase2_offline import phase_context
from werewolf.phase2_outcome import (
    extract_phase2_day_outcome, reference_tables_digest,
)
from werewolf.phase2_pilot_dataset import (
    Phase2OnlineConsequenceDatasetV1, analyze_phase2_online_support_record,
    build_phase2_consequence_dataset, build_phase2_online_dataset_from_ledger,
)
from werewolf.speech.validation import normalize_player
from werewolf.phase2_pilot_records import PublicProbeObservationV1
from werewolf.phase2_treatment import build_phase2_treatment


ARTIFACT_NAME = "paper-phase2-online-terminal-pilot-v1"
QUALIFICATION_NAME = "paper-phase2-online-terminal-qualification-v1"
ARTIFACT_VERSION = "phase2_online_terminal_pilot_v1"
PROBE_ARTIFACT_NAME = "paper-phase2-online-probe-pilot-v1"
PROBE_QUALIFICATION_NAME = "paper-phase2-online-probe-qualification-v3"


def _legal_wolf_team(observation, actor: str) -> frozenset[str]:
    if observation.get("identity") != "Werewolf":
        raise ValueError("Pilot-T requires the current living wolf's legal observation")
    records = [log.content.get("wolf_team") for log in observation["game_log"]
               if log.event == "werewolf_team_info"]
    if len(records) != 1 or not isinstance(records[0], list):
        raise ValueError("wolf team is not in current legal observation")
    team = frozenset(normalize_player(seat) for seat in records[0])
    if len(team) != 2 or actor not in team:
        raise ValueError("current legal wolf team differs")
    return team


class OnlineTerminalPilotRunnerV1:
    """One randomized strategy per game on the original canonical PRE path."""

    def __init__(self, *, plan: Phase2OnlineTerminalPilotPlanV1 | Phase2OnlineProbePilotPlanV1,
                 predictor, mapper,
                 backend, model_name: str, reference_tables,
                 reference_artifact_digest: str,
                 ledger: OnlinePilotAssignmentLedgerV1 | None = None,
                 record_evidence=None):
        if (not isinstance(plan, (Phase2OnlineTerminalPilotPlanV1, Phase2OnlineProbePilotPlanV1))
                or not callable(getattr(predictor, "predict", None))
                or not isinstance(getattr(predictor, "seal_digest", None), str)
                or not callable(getattr(mapper, "infer", None))
                or not isinstance(getattr(mapper, "artifact_digest", None), str)
                or not callable(getattr(backend, "chat_with_metadata", None))
                or not model_name
                or reference_artifact_digest != reference_tables_digest(reference_tables)
                or (isinstance(plan, Phase2OnlineProbePilotPlanV1) and ledger is None)
                or (record_evidence is not None and not callable(record_evidence))
                or (ledger is not None and
                    (not isinstance(ledger, OnlinePilotAssignmentLedgerV1)
                     or ledger.plan.digest() != plan.digest()))):
            raise ValueError("Pilot-T requires frozen Q/mapper/language/reference inputs")
        self.plan = plan
        self.predictor = predictor
        self.mapper = mapper
        self.backend = backend
        self.model_name = model_name
        self.reference_tables = reference_tables
        self.reference_artifact_digest = reference_artifact_digest
        self.ledger = ledger
        self.record_evidence = record_evidence
        self.state = OnlinePilotGameStateV1()
        if ledger is not None:
            ledger.mark_interrupted_on_resume()
            previous = ledger.snapshot()["games"]
            self.state.games_seen.update(previous)
            self.state.assignments.update({game_id: stage["ASSIGNMENT"]
                                           for game_id, stage in previous.items()
                                           if "ASSIGNMENT" in stage})
        self.records: dict[str, Phase2OnlineInterventionRecordV1] = {}
        self._active_game_id: str | None = None
        self._last_public_event_count = 0

    def start_game(self, game_id: str) -> None:
        if not game_id or self._active_game_id is not None:
            raise ValueError("Pilot-T game is already active")
        if self.ledger is not None:
            if self.ledger.status() != "RUNNING":
                raise ValueError("Pilot-T target or safety cap reached")
            self.ledger.start_game(game_id)
        self._active_game_id = game_id
        self._last_public_event_count = 0
        self.state.mark_seen(game_id)

    def handle_pre(self, *, env, recorder, call_audit, observation, handoff):
        game_id = self._active_game_id
        if game_id is None or recorder.game_id != game_id:
            raise ValueError("online runner game/recorder identity mismatch")
        if isinstance(self.plan, Phase2OnlineProbePilotPlanV1):
            return self._handle_probe_pre(env=env, recorder=recorder, call_audit=call_audit,
                                          observation=observation, handoff=handoff)
        if game_id in self.state.assignments or observation.get("identity") != "Werewolf":
            return None
        pending = recorder._pending
        prefix = pending.prefix
        actor = prefix.current_speaker
        if (handoff.boundary_id != prefix.boundary_id
                or handoff.prefix_digest != prefix.prefix_digest
                or handoff.observer_id != actor
                or normalize_player(observation["current_act_idx"]) != actor):
            raise ValueError("online intervention PRE handoff mismatch")
        known_wolves = _legal_wolf_team(observation, actor)
        context = context_from_pre(prefix, known_wolves)
        selection = select_candidate(self.plan, context)
        if selection is None:
            return None
        q = self.predictor.predict(prefix)
        opportunity = build_phase2_decision_opportunity(
            context, selection.candidate_j, q, self.mapper,
            q_source_digest=self.predictor.seal_digest)
        assignment = self.state.try_assign(
            self.plan, selection, opportunity,
            persist=self.ledger.persist_assignment if self.ledger is not None else None)
        if assignment is None:
            return None
        self.records[game_id] = Phase2OnlineInterventionRecordV1(assignment)
        treatment = assignment.treatment
        backend_audit = Phase2BackendCallAuditV1(
            opportunity, treatment, call_audit,
            pilot_id=self.plan.pilot_id, assignment_id=assignment.digest(),
            record_sink=(self.ledger.persist_backend_call
                         if self.ledger is not None else None))
        actor_model = Phase2LanguageActorV1(self.backend, self.model_name,
                                            call_audit=backend_audit)
        perceiver_model = Phase2SemanticPerceiverV1(self.backend, self.model_name,
                                                    call_audit=backend_audit)
        public = public_language_context_from_pre(prefix, context)
        verified = None
        audit_digest = None
        public_event_count_before = len(env.public_events)
        try:
            prepared = prepare_phase2_intervention_speech(
                opportunity, treatment, public, actor=actor_model,
                perceiver=perceiver_model)
            verified = prepared.verified
            audit_digest = sha256_bytes(verified.language_audit.canonical_bytes())
            calls = tuple(backend_audit.records)
            if not verified.success:
                self.records[game_id] = record_online_execution(
                    self.records[game_id], language_audit_digest=audit_digest,
                    attempt_count=verified.attempt_count, backend_calls=calls,
                    failure_reason=verified.failure_reason)
                if self.ledger is not None:
                    if self.record_evidence is not None:
                        self.record_evidence("execution", self.records[game_id])
                    self.ledger.persist_execution(self.records[game_id])
                return None  # baseline speech, explicitly recorded as noncompliance
            closure = assess_phase2_checkpoint_closure(opportunity, env=env, recorder=recorder)
            result, link = commit_phase2_verified_speech(
                env=env, recorder=recorder, opportunity=opportunity,
                treatment=treatment, verified=verified, closure=closure)
            self.records[game_id] = record_online_execution(
                self.records[game_id], language_audit_digest=audit_digest,
                attempt_count=verified.attempt_count, backend_calls=calls,
                canonical_link=link)
            if self.ledger is not None:
                if self.record_evidence is not None:
                    self.record_evidence("execution", self.records[game_id])
                self.ledger.persist_execution(self.records[game_id])
            return result
        except Exception as error:
            if (self.records[game_id].execution_success is None
                    and len(env.public_events) == public_event_count_before):
                self.records[game_id] = record_online_execution(
                    self.records[game_id], language_audit_digest=audit_digest,
                    attempt_count=(verified.attempt_count if verified is not None else
                                   min(2, len([call for call in backend_audit.records
                                               if call.role in ("realization", "repair")]))),
                    backend_calls=tuple(backend_audit.records),
                    failure_reason=f"LANGUAGE_OR_COMMIT_{type(error).__name__}")
                if self.ledger is not None:
                    if self.record_evidence is not None:
                        self.record_evidence("execution", self.records[game_id])
                    self.ledger.persist_execution(self.records[game_id])
            # A public event may have been committed before a downstream error.
            # Its execution state is unknown, not a proven language failure.
            raise

    def _probe_stage(self, event, **changes):
        game_id = self._active_game_id
        record = replace(self.records[game_id],
                         lifecycle=self.records[game_id].lifecycle + (event,), **changes)
        validate_probe_lifecycle_record(record.to_record())
        if self.ledger is not None:
            if self.record_evidence is not None:
                self.record_evidence(f"strategy-{len(record.lifecycle):02d}-{event.lower()}", record)
            self.ledger.persist_strategy_stage(record)
        self.records[game_id] = record
        return record

    def _probe_structural_failure(self, error, selection=None):
        record = self.records.get(self._active_game_id)
        if record is None and self.ledger is not None:
            self.ledger.persist_preparation_failure(
                self._active_game_id, f"{type(error).__name__}: {error}", selection)
            return
        if record is not None and record.lifecycle[-1] != "STRUCTURAL_FAILURE":
            self._probe_stage("STRUCTURAL_FAILURE",
                              structural_failure_reason=f"{type(error).__name__}: {error}")

    def _execute_probe_stage(self, *, env, recorder, call_audit, prefix, opportunity,
                             treatment, stage):
        record = self.records[self._active_game_id]
        audit = Phase2BackendCallAuditV1(
            opportunity, treatment, call_audit, pilot_id=self.plan.pilot_id,
            assignment_id=record.assignment.digest(),
            sequence_offset=len(record.backend_calls),
            record_sink=self.ledger.persist_backend_call if self.ledger else None)
        public = public_language_context_from_pre(prefix, opportunity.legal_context)
        prepared = prepare_phase2_intervention_speech(
            opportunity, treatment, public,
            actor=Phase2LanguageActorV1(self.backend, self.model_name, call_audit=audit),
            perceiver=Phase2SemanticPerceiverV1(self.backend, self.model_name, call_audit=audit))
        verified = prepared.verified
        if (verified.attempt_count not in (1, 2)
                or verified.language_audit.structured_execution_valid is not True):
            raise ValueError("structural preparation failure is not language invalid")
        link, result = None, None
        if verified.success:
            closure = assess_phase2_checkpoint_closure(opportunity, env=env, recorder=recorder)
            result, link = commit_phase2_verified_speech(
                env=env, recorder=recorder, opportunity=opportunity,
                treatment=treatment, verified=verified, closure=closure)
        execution = Phase2StrategyStageExecutionV1(
            opportunity, treatment, sha256_bytes(verified.language_audit.canonical_bytes()),
            verified.attempt_count, tuple(audit.records), link,
            None if link else verified.failure_reason)
        self._probe_stage(f"{stage}_COMMITTED" if link else f"{stage}_LANGUAGE_INVALID",
                          **{f"{stage.lower()}_execution": execution})
        if stage == "T1" and record.assignment.strategy is ProbeStrategy.PROBE_THEN_REDIRECT:
            self._probe_stage("T3_SCHEDULED" if link else "T3_CANCELLED")
            if link:
                return result
        record = self._probe_stage("EXECUTION_RECORDED")
        if self.ledger is not None:
            if self.record_evidence is not None:
                self.record_evidence("execution", record)
            self.ledger.persist_execution(record)
        return result  # final invalid executes the same PRE's baseline speech

    def _handle_probe_pre(self, *, env, recorder, call_audit, observation, handoff):
        game_id = self._active_game_id
        selection = None
        try:
            record = self.records.get(game_id)
            if record is not None:
                if "STRUCTURAL_FAILURE" in record.lifecycle:
                    raise ValueError("structurally failed strategy cannot continue")
                if record.lifecycle[-1] != "T3_SCHEDULED":
                    return None
            elif game_id in self.state.assignments:
                raise ValueError("interrupted assignment cannot resume gameplay")
            prefix = recorder._pending.prefix
            actor = prefix.current_speaker
            if (prefix.game_id != game_id or handoff.boundary_id != prefix.boundary_id or
                    handoff.prefix_digest != prefix.prefix_digest or
                    handoff.observer_id != actor or
                    normalize_player(observation["current_act_idx"]) != actor):
                raise ValueError("Probe PRE handoff mismatch")
            if record is not None:
                initial = record.assignment.opportunity
                if (phase_context(prefix)[0] != initial.legal_context.phase or
                        prefix.boundary_id == initial.legal_context.boundary_id):
                    raise ValueError("Probe continuation phase/boundary mismatch")
                expected = initial.observation_window.expected_speakers
                if actor != initial.continuation_actor:
                    if len(record.observations) >= len(expected) or actor != expected[len(record.observations)]:
                        raise ValueError("Probe continuation speaker/timing mismatch")
                    return None
                if len(record.observations) != len(expected):
                    raise ValueError("Probe continuation observation window incomplete")
                self._probe_stage("T3_REACHED")
            if observation.get("identity") != "Werewolf":
                if record is not None:
                    raise ValueError("designated continuation actor is not a living wolf")
                return None
            context = context_from_pre(prefix, _legal_wolf_team(observation, actor))
            if record is None:
                selection = select_probe_candidate(self.plan, context)
                if selection is None:
                    return None
                candidate = selection.candidate_j
            else:
                initial_context = record.assignment.opportunity.legal_context
                if (context.alive != initial_context.alive or
                        context.competition != initial_context.competition or
                        context.known_wolves != initial_context.known_wolves or
                        context.public_speaker_queue != initial_context.public_speaker_queue):
                    raise ValueError("Probe continuation legal state changed within speech")
                candidate = record.assignment.opportunity.candidate_j
                if candidate not in context.legal_targets:
                    raise ValueError("initial Probe candidate is no longer legal")
            opportunity = build_phase2_decision_opportunity(
                context, candidate, self.predictor.predict(prefix), self.mapper,
                q_source_digest=self.predictor.seal_digest)
            if record is None:
                assignment = self.state.try_assign(
                    self.plan, selection, opportunity,
                    persist=self.ledger.persist_assignment if self.ledger else None)
                if assignment is None:
                    return None
                self.records[game_id] = Phase2OnlineProbeRecordV1(assignment)
                if self.ledger:
                    self.ledger.persist_strategy_stage(self.records[game_id])
                self._last_public_event_count = len(env.public_events)
                self._probe_stage("T1_ATTEMPTED")
                treatment = assignment.treatment
                stage = "T1"
            else:
                key = sha256_bytes(canonical_json_bytes([
                    "phase2_probe_continuation_v1", record.assignment.digest(), opportunity.digest()]))
                treatment = build_phase2_treatment(
                    opportunity, Action.REDIRECT, assignment_source="strategy_continuation",
                    assignment_probability=1, randomization_key=key)
                self._probe_stage("T3_PREPARED", t3_opportunity=opportunity, t3_treatment=treatment)
                stage = "T3"
            return self._execute_probe_stage(
                env=env, recorder=recorder, call_audit=call_audit, prefix=prefix,
                opportunity=opportunity, treatment=treatment, stage=stage)
        except Exception as error:
            self._probe_structural_failure(error, selection)
            raise

    def _after_probe_step(self, env, new_events, done):
        record = self.records.get(self._active_game_id)
        if record is None:
            return
        try:
            if record.lifecycle[-1] == "T3_SCHEDULED":
                expected = record.assignment.opportunity.observation_window.expected_speakers
                for event in new_events:
                    if (event["event_type"] == "public_speech" and
                            event.get("event_id") == record.t1_execution.canonical_link.canonical_event_id
                            and not record.observations):
                        continue  # initial committed Probe, outside its observation window
                    if event["event_type"] == "public_speech":
                        index = len(record.observations)
                        if index >= len(expected) or event["speaker"] != expected[index]:
                            raise ValueError("unexpected speech in Probe observation window")
                        record = replace(record, observations=record.observations + (
                            PublicProbeObservationV1(event["event_id"], event["speaker"], event["raw_text"]),))
                        self.records[self._active_game_id] = record
                if env.phase != record.assignment.opportunity.legal_context.phase or done:
                    raise ValueError("designated Probe continuation PRE became unavailable")
            if "EXECUTION_RECORDED" not in record.lifecycle or record.day_outcome is not None:
                return
            for event in new_events:
                if event["event_type"] != "exile_result" or env.phase == "speech_pk":
                    continue
                expelled = event["exiled_players"]
                if not isinstance(expelled, list) or len(expelled) > 1:
                    raise ValueError("invalid canonical day exile")
                outcome = extract_phase2_day_outcome(record.assignment.opportunity,
                            expelled[0] if expelled else None, self.reference_tables)
                record = self._probe_stage("DAY_CONSEQUENCE_RECORDED", day_outcome=outcome,
                                          reference_artifact_digest=self.reference_artifact_digest)
                if self.ledger:
                    if self.record_evidence:
                        self.record_evidence("consequence", record)
                    self.ledger.persist_consequence(record)
                break
        except Exception as error:
            self._probe_structural_failure(error)
            raise

    def after_step(self, *, env, done: bool, info):
        del info
        game_id = self._active_game_id
        if game_id is None:
            raise ValueError("Pilot-T step has no active game")
        new_events = env.public_events[self._last_public_event_count:]
        self._last_public_event_count = len(env.public_events)
        if isinstance(self.plan, Phase2OnlineProbePilotPlanV1):
            return self._after_probe_step(env, new_events, done)
        record = self.records.get(game_id)
        if record is None or record.execution_success is not True or record.day_outcome is not None:
            return
        for event in new_events:
            if event["event_type"] != "exile_result":
                continue
            # Ordinary vote ties open speech_pk; that event is not day resolution.
            if env.phase == "speech_pk":
                continue
            expelled = event["exiled_players"]
            if not isinstance(expelled, list) or len(expelled) > 1:
                raise ValueError("canonical day outcome has multiple exiles")
            outcome = extract_phase2_day_outcome(
                record.assignment.opportunity,
                expelled[0] if expelled else None, self.reference_tables)
            self.records[game_id] = record_online_day_outcome(
                record, outcome,
                reference_artifact_digest=self.reference_artifact_digest)
            if self.ledger is not None:
                if self.record_evidence is not None:
                    self.record_evidence("consequence", self.records[game_id])
                self.ledger.persist_consequence(self.records[game_id])
            break

    def after_game(self, *, env, winner: str):
        del env
        game_id = self._active_game_id
        if game_id is None:
            raise ValueError("Pilot-T game was not started")
        record = self.records.get(game_id)
        if record is not None:
            if isinstance(record, Phase2OnlineProbeRecordV1):
                if record.day_outcome is None:
                    error = ValueError("assigned Probe game ended without T0 day consequence")
                    self._probe_structural_failure(error)
                    raise error
                self._probe_stage("GAME_RESULT_RECORDED", final_game_result=winner)
            else:
                self.records[game_id] = replace(record, final_game_result=winner)
            if self.ledger is not None:
                if self.record_evidence is not None:
                    self.record_evidence("game-result", self.records[game_id])
                self.ledger.persist_game_result(game_id, winner)
        self._active_game_id = None


def require_online_gameplay_hook() -> None:
    """Validate the actual production callable without starting a game."""
    signature = inspect.signature(run_canonical_game)
    hook = signature.parameters.get("online_pilot")
    if hook is None or hook.kind is not inspect.Parameter.KEYWORD_ONLY:
        raise TypeError("production gameplay requires an explicit online_pilot PRE hook")
    signature.bind(None, (), (), canonical_recorder=None, call_audit=None,
                   online_pilot=None)


def run_online_game(env, agents, roles, *, recorder, call_audit,
                    pilot: OnlineTerminalPilotRunnerV1, preflight):
    """The canonical loop is unchanged unless this explicit hook is supplied."""
    if not isinstance(pilot, OnlineTerminalPilotRunnerV1):
        raise TypeError("explicit online Pilot-T runner required")
    from werewolf.phase2_online_preflight import PilotPreflightV1
    if (not isinstance(preflight, PilotPreflightV1)
            or preflight.online_randomized_pilot_ready is not True
            or preflight.publication_only
            or not isinstance(pilot.ledger, OnlinePilotAssignmentLedgerV1)
            or preflight.online_plan_digest != pilot.plan.digest()
            or preflight.source_commit_pin != pilot.ledger.source_commit
            or preflight.reference_tables_digest_pin != pilot.reference_artifact_digest
            or preflight.runtime_token !=
               (id(env), id(recorder), id(call_audit), id(pilot.backend),
                id(pilot.predictor), id(pilot.mapper), id(pilot.reference_tables),
                id(pilot.ledger))):
        raise ValueError("online Pilot-T preflight does not bind this plan/runtime")
    require_online_gameplay_hook()
    pilot.start_game(recorder.game_id)
    return run_canonical_game(env, agents, roles, canonical_recorder=recorder,
                              call_audit=call_audit, online_pilot=pilot)


@dataclass(frozen=True)
class OnlinePilotRuntimeBundleV1:
    """Server-supplied canonical game runtime; construction must make no model call."""

    env: object
    agents: tuple
    roles: tuple
    recorder: object
    call_audit: object
    pilot: OnlineTerminalPilotRunnerV1
    preflight: object


def run_online_campaign(game_ids: tuple[str, ...], runtime_factory, *,
                        ledger: OnlinePilotAssignmentLedgerV1) -> dict:
    """Use new game IDs until the frozen assignment target or safety cap.

    The caller pre-registers game IDs and supplies a fresh canonical runtime
    and passed preflight per game. An exception stops the campaign with the
    durable journal intact; no game is replayed and no arm is reassigned.
    """
    if (not isinstance(ledger, OnlinePilotAssignmentLedgerV1)
            or not callable(runtime_factory)
            or not isinstance(game_ids, tuple)
            or any(not isinstance(game_id, str) or not game_id for game_id in game_ids)
            or len(set(game_ids)) != len(game_ids)):
        raise ValueError("campaign requires unique pre-registered game IDs")
    for game_id in game_ids:
        if ledger.status() != "RUNNING":
            break
        if game_id in ledger.snapshot()["games"]:
            continue  # prior attempt is not replayable without a simulator checkpoint
        bundle = runtime_factory(game_id)
        if (not isinstance(bundle, OnlinePilotRuntimeBundleV1)
                or bundle.recorder.game_id != game_id
                or bundle.pilot.ledger is not ledger
                or bundle.pilot.plan.digest() != ledger.plan.digest()):
            raise ValueError("server runtime does not bind this campaign/game")
        run_online_game(bundle.env, bundle.agents, bundle.roles,
                        recorder=bundle.recorder, call_audit=bundle.call_audit,
                        pilot=bundle.pilot, preflight=bundle.preflight)
    snapshot = ledger.snapshot()
    return {"campaign_purpose": ledger.plan.campaign_purpose,
            "target_assignment_count": ledger.plan.target_assignment_count,
            "games_attempted": snapshot["games_attempted"],
            "assignment_count": snapshot["assignment_count"],
            "status": ledger.status(), "sealable": ledger.sealable()}


def build_online_dataset(pilot: OnlineTerminalPilotRunnerV1, *,
                         source_commit: str,
                         execution_verifier, allow_synthetic=False):
    if pilot.ledger is not None:
        if source_commit != pilot.ledger.source_commit:
            raise ValueError("dataset source differs from assignment ledger")
        return build_phase2_online_dataset_from_ledger(
            pilot.ledger, execution_verifier=execution_verifier,
            allow_synthetic=allow_synthetic)
    return build_phase2_consequence_dataset(
        source_commit=source_commit,
        online_pilot_plan_digest=pilot.plan.digest(),
        online_records=tuple(pilot.records.values()),
        games_seen=len(pilot.state.games_seen),
        eligible_opportunities=pilot.state.eligible_opportunities,
        execution_verifier=execution_verifier, allow_synthetic=allow_synthetic)


def publish_online_pilot(destination: Path, *, pilot, dataset, preflight,
                         server_run_provenance: dict | None = None) -> str:
    """Future server-only immutable publication; never called by default."""
    from werewolf.phase2_online_preflight import PilotPreflightV1
    if os.path.lexists(destination):
        raise FileExistsError("online pilot destination must be absent")
    is_probe = isinstance(pilot.plan, Phase2OnlineProbePilotPlanV1)
    expected_name = ((PROBE_QUALIFICATION_NAME if pilot.plan.campaign_purpose == "qualification"
                      else PROBE_ARTIFACT_NAME) if is_probe else
                     (QUALIFICATION_NAME if pilot.plan.campaign_purpose == "qualification"
                      else ARTIFACT_NAME))
    if destination.name != expected_name:
        raise ValueError("qualification and formal pilot require distinct artifact names")
    if is_probe:
        pins = server_run_provenance
        protocol = Path(__file__).resolve().parents[1] / "docs/research/phase2-probe-policy-protocol-v1.md"
        if (not isinstance(pins, dict)
                or pins.get("probe_policy_protocol_sha256") != sha256_bytes(protocol.read_bytes())
                or any(not isinstance(pins.get(key), str) or len(pins[key]) != 64
                       or any(c not in "0123456789abcdef" for c in pins[key]) for key in (
                           "probe_policy_protocol_sha256", "probe_campaign_profile_digest"))):
            raise ValueError("Probe publication requires bound protocol/profile provenance")
    if not isinstance(preflight, PilotPreflightV1) or not preflight.ready:
        raise ValueError("formal online pilot requires a passed frozen preflight")
    source_provenance = preflight.source_provenance
    smoke_v3_manifest_digest = preflight.smoke_v3_manifest_digest
    if (not isinstance(dataset, Phase2OnlineConsequenceDatasetV1)
            or not isinstance(pilot.ledger, OnlinePilotAssignmentLedgerV1)
            or pilot.ledger.status() != "READY_TO_SEAL"
            or not pilot.ledger.sealable()
            or dataset.manifest.get("ledger_digest") !=
               pilot.ledger.snapshot()["events"][-1]["digest"]
            or dataset.manifest["assignment_count"] != pilot.plan.target_assignment_count
            or dataset.manifest.get("target_assignment_count") !=
               pilot.plan.target_assignment_count
            or dataset.manifest.get("campaign_purpose") != pilot.plan.campaign_purpose
            or dataset.manifest.get("estimator_eligible") !=
               (not is_probe and pilot.plan.campaign_purpose == "pilot")
            or pilot.plan.selection_status != "FROZEN"
            or preflight.online_plan_digest != pilot.plan.digest()
            or preflight.source_commit_pin != dataset.manifest["source_commit"]
            or preflight.reference_tables_digest_pin != pilot.reference_artifact_digest
            or preflight.destination != str(destination)
            or preflight.runtime_token is None
            or preflight.runtime_token[3:] !=
               (id(pilot.backend), id(pilot.predictor), id(pilot.mapper),
                id(pilot.reference_tables), id(pilot.ledger))
            or dataset.manifest["synthetic_audit_only"]
            or dataset.manifest["pilot_plan_digest"] != pilot.plan.digest()
            or not isinstance(source_provenance, dict)
            or set(source_provenance) != {"commit", "branch", "tracked_worktree_clean",
                                         "staged_tracked_changes", "source_sha256"}
            or source_provenance.get("commit") != dataset.manifest["source_commit"]
            or source_provenance.get("tracked_worktree_clean") is not True
            or source_provenance.get("staged_tracked_changes") is not False
            or not isinstance(smoke_v3_manifest_digest, str)
            or len(smoke_v3_manifest_digest) != 64):
        raise ValueError("formal online pilot requires frozen plan and real execution proof")
    support = analyze_phase2_online_support_record(dataset.to_record())
    report = (("# Online Probe policy support\n\n" if is_probe else "# Online Pilot-T support\n\n")
              + "Assignment, execution, and consequence counts are descriptive. "
              "No lambda or policy effect is fitted.\n")
    artifact = publish_artifact(destination, manifest_fields={
        "artifact_type": "phase2_online_probe_pilot" if is_probe else "phase2_online_terminal_pilot",
        "schema_version": "phase2_online_probe_pilot_v1" if is_probe else ARTIFACT_VERSION,
        "study_name": expected_name,
        "campaign_purpose": pilot.plan.campaign_purpose,
        "target_assignment_count": pilot.plan.target_assignment_count,
        "completion_status": "COMPLETE",
        "ledger_digest": dataset.manifest["ledger_digest"],
        "pilot_plan_digest": pilot.plan.digest(),
        "source_provenance": source_provenance,
        "smoke_v3_manifest_digest": smoke_v3_manifest_digest,
        "mapper_artifact_digest": pilot.mapper.artifact_digest,
        "q_source_digest": pilot.predictor.seal_digest,
        "reference_artifact_digest": pilot.reference_artifact_digest,
        **({"server_run_provenance": server_run_provenance}
           if server_run_provenance is not None else {}),
    }, files={
        "pilot_plan.json": canonical_json_bytes(pilot.plan.to_record()),
        "assignments.jsonl": canonical_jsonl_bytes(dataset.assignments),
        "executions.jsonl": canonical_jsonl_bytes(dataset.executions),
        "consequences.jsonl": canonical_jsonl_bytes(dataset.consequences),
        "backend_calls.jsonl": canonical_jsonl_bytes(dataset.backend_calls),
        **({"strategy_stages.jsonl": canonical_jsonl_bytes(dataset.strategy_stages)} if is_probe else {}),
        "metrics/support.json": canonical_json_bytes(support),
        "report.md": report.encode("utf-8"),
    })
    return artifact.manifest_digest
