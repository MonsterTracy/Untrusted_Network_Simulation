"""Canonical Collection authority at the Classic7 Game Runtime boundary."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from werewolf.artifact_io import sha256_bytes
from werewolf.canonical_collection.attempt_ledger import AttemptClaim, CollectionPlan
from werewolf.canonical_collection.call_audit import CanonicalCallAudit
from werewolf.canonical_collection.collector import CanonicalAttemptFailure
from werewolf.canonical_collection.failure_evidence import (
    CanonicalFailureStage,
    construct_canonical_failure_attempt,
)
from werewolf.canonical_collection.game_bundle import DeterministicReplayExecutor
from werewolf.canonical_collection.pre import (
    AuthoritativePREPrefix,
    SpeakerPREBeliefHandoff,
    construct_authoritative_pre_prefix,
    construct_speaker_pre_belief_handoff,
)
from werewolf.canonical_collection.public_history import (
    PLAYER_IDS,
    freeze_public_event_history,
)
from werewolf.canonical_collection.trajectory_evidence import (
    BeliefObservation,
    BeliefObservationStatus,
    CanonicalGameEvidence,
    construct_belief_observation,
    construct_canonical_game_evidence,
    construct_private_replay_evidence,
    construct_submitted_gameplay_action,
)
from werewolf.envs.werewolf_text_env_v0 import V1SpeechPerceptionExhausted
from werewolf.models.twd_tom.schema import normalize_player


_EXTERNAL_AGENT_FIELDS = {
    "backend",
    "handler",
    "logger",
    "tokenizer",
    "strategy",
    "matcher",
}


def _snapshot_agent_state(agent) -> dict[str, Any]:
    return {
        name: deepcopy(value)
        for name, value in vars(agent).items()
        if name not in _EXTERNAL_AGENT_FIELDS
    }


def _response_digest(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("belief response must be text or None")
    return sha256_bytes(value.encode("utf-8"))


class PlayingAgentBeliefObservationCollector:
    """Collect each linked alive observer exactly once on one frozen prefix."""

    def __init__(self, *, plan: CollectionPlan, reporter, agents) -> None:
        if reporter is None or not callable(getattr(reporter, "report", None)):
            raise TypeError("reporter must provide report()")
        if not isinstance(agents, (list, tuple)) or len(agents) != 7:
            raise ValueError("agents must contain exactly seven playing agents")
        self.plan = plan
        self.reporter = reporter
        self.agents = tuple(agents)

    def _failure_attempts(self, records):
        return tuple(
            construct_canonical_failure_attempt(
                attempt_index=record["attempt_index"],
                call_id=record["call_id"],
                backend_identity=self.plan.backend_identity,
                model_identity=self.plan.model_identity,
                parser_identity=self.plan.parser_identity,
                prompt_identity=self.plan.prompt_identity,
                retry_policy_identity=self.plan.retry_policy_identity,
                call_budget_identity=self.plan.call_budget_identity,
                error_category=record["error_category"] or record["status"],
                error_message=record["error_message"] or "belief report failed",
                response_digest=_response_digest(record["raw_response"]),
            )
            for record in records
        )

    def collect(
        self,
        prefix: AuthoritativePREPrefix,
        *,
        env,
        attempt_id: str,
    ) -> tuple[BeliefObservation, ...]:
        observations: list[BeliefObservation] = []
        for link in prefix.belief_observation_links:
            observer = link.observer_id
            seat = int(observer.removeprefix("player"))
            agent = self.agents[seat - 1]
            backend_id = getattr(agent, "backend_id", None)
            if not isinstance(backend_id, str) or not backend_id:
                raise ValueError(f"{observer} has no agent_backend_id")
            state_before = _snapshot_agent_state(agent)
            result = self.reporter.report(
                agent=agent,
                observation=env.get_observation_for(seat),
                observer_id=observer,
                pre_prefix=prefix,
                observation_id=link.observation_id,
                agent_backend_id=backend_id,
                known_werewolves=env.get_twd_tom_hard_knowledge_for(seat)[0],
                known_non_werewolves=env.get_twd_tom_hard_knowledge_for(seat)[1],
            )
            state_after = _snapshot_agent_state(agent)
            self.reporter.record_agent_state(
                observer_id=observer,
                state_before=state_before,
                state_after=state_after,
            )
            if state_after != state_before:
                raise RuntimeError(
                    f"readonly belief report mutated {observer} agent state"
                )
            if not isinstance(result, dict) or result.get("observer") != observer:
                raise TypeError("belief reporter returned an invalid observer result")
            attempts = tuple(result.get("generation_attempts", ()))
            if result.get("status") != "ok":
                raise CanonicalAttemptFailure(
                    game_id=prefix.game_id,
                    stage=CanonicalFailureStage.BELIEF_OBSERVATION,
                    error_category=result.get("status") or "belief_report_error",
                    error_message=result.get("error") or "belief report failed",
                    retry_exhausted=True,
                    boundary_id=prefix.boundary_id,
                    observer_id=observer,
                    day=prefix.public_temporal_state.day,
                    phase=prefix.public_temporal_state.phase.value,
                    attempt_evidence=self._failure_attempts(attempts),
                    partial_payload={"pre_prefix": prefix.to_record()},
                )
            observation_attempts = tuple(
                {
                    "attempt_index": record["attempt_index"],
                    "call_id": record["call_id"],
                    "status": (
                        "success" if record["status"] == "ok" else "error"
                    ),
                    "response_digest": _response_digest(record["raw_response"]),
                    "error_category": record["error_category"],
                    "error_message": record["error_message"],
                }
                for record in attempts
            )
            observations.append(
                construct_belief_observation(
                    game_id=prefix.game_id,
                    attempt_id=attempt_id,
                    boundary_id=prefix.boundary_id,
                    prefix_digest=prefix.prefix_digest,
                    observation_id=link.observation_id,
                    observer_id=observer,
                    observer_alive=True,
                    day=prefix.public_temporal_state.day,
                    phase=prefix.public_temporal_state.phase.value,
                    status=BeliefObservationStatus.SUCCESS,
                    suspicion_support=result["suspected_werewolves"],
                    attempts=observation_attempts,
                )
            )
        return tuple(observations)


@dataclass
class _PendingAction:
    actor_id: str
    event_count_before: int
    raw_action: Any | None = None
    prefix: AuthoritativePREPrefix | None = None
    handoff: SpeakerPREBeliefHandoff | None = None


class CanonicalGameRecorder:
    """Sole constructor of PRE, observation, handoff and submitted-action evidence."""

    def __init__(
        self,
        *,
        plan: CollectionPlan,
        claim: AttemptClaim,
        game_id: str,
        belief_collector: PlayingAgentBeliefObservationCollector,
        call_audit: CanonicalCallAudit,
        runtime_configuration: dict[str, Any],
    ) -> None:
        self.plan = plan
        self.claim = claim
        self.game_id = game_id
        self.belief_collector = belief_collector
        self.call_audit = call_audit
        self.runtime_configuration = deepcopy(runtime_configuration)
        self.prefixes: list[AuthoritativePREPrefix] = []
        self.observations: list[BeliefObservation] = []
        self.submitted_gameplay_actions = []
        self._raw_actions: list[Any] = []
        self._pending: _PendingAction | None = None
        self._roles: tuple[str, ...] | None = None
        self._env = None
        self._winner: str | None = None

    def start(self, env, *, roles) -> None:
        if self._env is not None:
            raise RuntimeError("Canonical Game Recorder already started")
        if len(roles) != 7:
            raise ValueError("Classic7 runtime requires seven roles")
        freeze_public_event_history(env.public_events)
        self._env = env
        self._roles = tuple(roles)

    def before_agent_act(
        self,
        env,
        *,
        step_idx: int,
        acting_player_id: int,
        delivered_observation,
        speech_kind: str | None,
    ) -> SpeakerPREBeliefHandoff | None:
        del step_idx, delivered_observation
        if self._pending is not None:
            raise RuntimeError("previous runtime action has not been committed")
        actor = normalize_player(acting_player_id)
        self._pending = _PendingAction(
            actor_id=actor,
            event_count_before=len(env.public_events),
        )
        if speech_kind is None:
            return None
        history = freeze_public_event_history(env.public_events)
        terminal = history.events[-1]
        boundary_id = f"{self.game_id}-pre-{terminal.event_index:06d}"
        alive = tuple(
            player
            for player, is_alive in zip(PLAYER_IDS, env.alive, strict=True)
            if is_alive == 1
        )
        observation_ids = {
            observer: f"{boundary_id}-belief-{observer}"
            for observer in alive
        }
        prefix = construct_authoritative_pre_prefix(
            game_id=self.game_id,
            boundary_id=boundary_id,
            step_index=terminal.event_index,
            report_trigger_id=f"{boundary_id}-trigger",
            current_speaker=actor,
            alive_observer_ids=alive,
            public_event_history=history,
            v1_annotations=tuple(env.speech_annotations),
            belief_observation_ids_by_observer=observation_ids,
        )
        self.prefixes.append(prefix)
        self._pending.prefix = prefix
        observations = self.belief_collector.collect(
            prefix,
            env=env,
            attempt_id=self.claim.attempt_id,
        )
        speaker_observation = next(
            item for item in observations if item.observer_id == actor
        )
        handoff = construct_speaker_pre_belief_handoff(
            prefix,
            observation_id=speaker_observation.observation_id,
            observation_digest=speaker_observation.observation_digest,
            observer_id=actor,
            observation_status=speaker_observation.status.value,
            suspicion_support=speaker_observation.suspicion_support,
        )
        self.observations.extend(observations)
        self._pending.handoff = handoff
        return handoff

    def after_agent_act(self, action) -> None:
        if self._pending is None or self._pending.raw_action is not None:
            raise RuntimeError("runtime action has no unique pending slot")
        self._pending.raw_action = deepcopy(action)

    def after_env_step(self, env, *, observation_after, terminal_after) -> None:
        del observation_after, terminal_after
        if self._pending is None or self._pending.raw_action is None:
            raise RuntimeError("environment step has no submitted action")
        pending = self._pending
        appended = env.public_events[pending.event_count_before :]
        if pending.handoff is not None:
            speech_events = [
                event for event in appended if event["event_type"] == "public_speech"
            ]
            if len(speech_events) != 1:
                raise ValueError("submitted speech must produce one public_speech")
            resulting_ids = (speech_events[0]["event_id"],)
            payload = pending.raw_action[1]
            action_type = "public_speech"
            step_index = pending.prefix.step_index
            boundary_id = pending.prefix.boundary_id
        else:
            resulting_ids = tuple(event["event_id"] for event in appended)
            payload = list(pending.raw_action)
            action_type = str(pending.raw_action[0])
            step_index = len(self.submitted_gameplay_actions)
            boundary_id = None
        self.submitted_gameplay_actions.append(
            construct_submitted_gameplay_action(
                action_id=f"{self.game_id}-action-{len(self.submitted_gameplay_actions):06d}",
                step_index=step_index,
                actor_id=pending.actor_id,
                action_type=action_type,
                action_payload=payload,
                resulting_public_event_ids=resulting_ids,
                boundary_id=boundary_id,
                speaker_pre_belief_handoff=pending.handoff,
                fallback_used=False,
            )
        )
        self._raw_actions.append(deepcopy(pending.raw_action))
        self._pending = None

    def finish(self, env, *, winner: str) -> None:
        self._env = env
        self._winner = winner

    def complete_evidence(self) -> CanonicalGameEvidence:
        if self._env is None or self._roles is None or self._winner is None:
            raise RuntimeError("game is not complete")
        if self._pending is not None:
            raise RuntimeError("game ended with an uncommitted action")
        history = freeze_public_event_history(self._env.public_events)
        private = construct_private_replay_evidence(
            game_id=self.game_id,
            seed=self.claim.seed,
            role_assignment=dict(zip(PLAYER_IDS, self._roles, strict=True)),
            runtime_configuration=self.runtime_configuration,
            initial_runtime_state={"day": 0, "phase": "night"},
            replay_inputs={
                "submitted_actions": self._raw_actions,
                "winner": self._winner,
            },
            expected_public_event_digest=history.digest,
        )
        return construct_canonical_game_evidence(
            game_id=self.game_id,
            public_event_stream=history,
            authoritative_pre_prefixes=tuple(self.prefixes),
            belief_observations=tuple(self.observations),
            speech_annotations_v1=tuple(self._env.speech_annotations),
            call_budget_summary=self.call_audit.summary(),
            submitted_gameplay_actions=tuple(self.submitted_gameplay_actions),
            backend_call_evidence=self.call_audit.records,
            private_replay_evidence=private,
        )

    def _failure_partial_payload(self) -> dict[str, Any]:
        pending_action = (
            None
            if self._pending is None or self._pending.raw_action is None
            else deepcopy(self._pending.raw_action)
        )
        return {
            "public_events": (
                [] if self._env is None else deepcopy(self._env.public_events)
            ),
            "authoritative_pre_prefixes": [
                item.to_record() for item in self.prefixes
            ],
            "belief_observations": [
                item.to_record() for item in self.observations
            ],
            "speech_annotations_v1": (
                []
                if self._env is None
                else [item.to_record() for item in self._env.speech_annotations]
            ),
            "submitted_actions": deepcopy(self._raw_actions),
            "pending_submitted_action": pending_action,
            "backend_calls": [
                item.to_record() for item in self.call_audit.records
            ],
        }

    def failure_from_exception(
        self,
        error: Exception,
        *,
        default_stage: CanonicalFailureStage = CanonicalFailureStage.GAMEPLAY_ACTION,
    ) -> CanonicalAttemptFailure:
        if isinstance(error, CanonicalAttemptFailure):
            partial_payload = self._failure_partial_payload()
            if error.partial_payload is not None:
                partial_payload.update(error.partial_payload)
            return CanonicalAttemptFailure(
                game_id=error.game_id,
                stage=error.stage,
                error_category=error.error_category,
                error_message=error.error_message,
                retry_exhausted=error.retry_exhausted,
                boundary_id=error.boundary_id,
                observer_id=error.observer_id,
                day=error.day,
                phase=error.phase,
                attempt_evidence=error.attempt_evidence,
                partial_payload=partial_payload,
            )
        pending_prefix = None if self._pending is None else self._pending.prefix
        if isinstance(error, V1SpeechPerceptionExhausted):
            annotation = error.annotation
            attempts = tuple(
                construct_canonical_failure_attempt(
                    attempt_index=item.attempt_index,
                    call_id=item.call_id,
                    backend_identity=self.plan.backend_identity,
                    model_identity=self.plan.model_identity,
                    parser_identity=self.plan.parser_identity,
                    prompt_identity=self.plan.prompt_identity,
                    retry_policy_identity=self.plan.retry_policy_identity,
                    call_budget_identity=self.plan.call_budget_identity,
                    error_category=item.error_category,
                    error_message=item.error_message,
                    response_digest=_response_digest(item.raw_response),
                )
                for item in annotation.attempts
            )
            stage = CanonicalFailureStage.SPEECH_PERCEPTION
            retry_exhausted = True
            category = annotation.attempts[-1].error_category
            message = annotation.attempts[-1].error_message
        else:
            attempts = ()
            stage = default_stage
            retry_exhausted = "Exhausted" in type(error).__name__
            category = type(error).__name__
            message = str(error) or category
        return CanonicalAttemptFailure(
            game_id=self.game_id,
            stage=stage,
            error_category=category,
            error_message=message,
            retry_exhausted=retry_exhausted,
            boundary_id=(
                None if pending_prefix is None else pending_prefix.boundary_id
            ),
            observer_id=(
                None if self._pending is None else self._pending.actor_id
            ),
            day=(
                None
                if pending_prefix is None
                else pending_prefix.public_temporal_state.day
            ),
            phase=(
                None
                if pending_prefix is None
                else pending_prefix.public_temporal_state.phase.value
            ),
            attempt_evidence=attempts,
            partial_payload=self._failure_partial_payload(),
        )


def make_classic7_replay_executor(
    *,
    identity: str,
    environment_factory,
) -> DeterministicReplayExecutor:
    """Replay submitted runtime actions without any external backend calls."""

    if not callable(environment_factory):
        raise TypeError("environment_factory must be callable")

    def execute(private_replay_evidence, submitted_actions):
        replay_inputs = private_replay_evidence.replay_inputs.to_value()
        raw_actions = replay_inputs.get("submitted_actions")
        if not isinstance(raw_actions, list):
            raise ValueError("Private Replay Evidence lacks submitted_actions")
        if len(raw_actions) != len(submitted_actions):
            raise ValueError("replay action count disagrees with submitted evidence")
        env = environment_factory(private_replay_evidence.seed)
        roles = [
            role for _, role in private_replay_evidence.role_assignment
        ]
        env.reset(roles=roles)
        for raw_action in raw_actions:
            if not isinstance(raw_action, list):
                raise TypeError("replay action must be a JSON array")
            env.step(tuple(raw_action))
        return list(env.public_events)

    return DeterministicReplayExecutor(identity=identity, execute=execute)


__all__ = [
    "CanonicalGameRecorder",
    "PlayingAgentBeliefObservationCollector",
    "make_classic7_replay_executor",
]
