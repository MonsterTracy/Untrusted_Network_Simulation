"""Production RuntimeFactory for the sole Canonical Collection path."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping

from run_random import build_runtime, eval as run_game
from werewolf.canonical_collection.attempt_ledger import AttemptClaim, CollectionPlan
from werewolf.canonical_collection.call_audit import CanonicalCallAudit, audited_backends
from werewolf.canonical_collection.collector import (
    CanonicalAttemptFailure,
    CanonicalGameProduct,
)
from werewolf.canonical_collection.failure_evidence import CanonicalFailureStage
from werewolf.canonical_collection.runtime import (
    CanonicalGameRecorder,
    PlayingAgentBeliefObservationCollector,
    make_classic7_replay_executor,
)
from werewolf.envs.werewolf_text_env_v0 import WerewolfTextEnvV0
from werewolf.runtime_config import normalize_runtime_config
from werewolf.speech.private_belief_perceiver import PlayingAgentBeliefReporter
from werewolf.speech.speech_perceiver import SpeechParseAuditResult


class _ReplaySpeechPerceiver:
    """Replay public speech boundaries without re-invoking semantic perception."""

    model_name = "canonical-replay-no-backend"
    backend_id = "canonical-replay-no-backend"

    def parse_with_audit(self, *, speaker, speech, day, phase):
        del speaker, speech, day, phase
        return SpeechParseAuditResult(
            normalized_actions=[],
            raw_response="NONE",
            parse_status="ok",
            error_type=None,
            error_message=None,
            generation_attempts=(
                {
                    "generation_attempt": 1,
                    "status": "ok",
                    "raw_response": "NONE",
                    "error_type": None,
                    "error_message": None,
                },
            ),
        )


@dataclass
class _RecordedRuntime:
    env: Any
    agents: list[Any]
    roles: list[str]
    recorder: CanonicalGameRecorder
    call_audit: CanonicalCallAudit
    replay_executor: Any

    def run(self) -> CanonicalGameProduct:
        try:
            run_game(
                self.env,
                self.agents,
                self.roles,
                canonical_recorder=self.recorder,
                call_audit=self.call_audit,
            )
            evidence = self.recorder.complete_evidence()
        except CanonicalAttemptFailure:
            raise
        except Exception as error:
            raise self.recorder.failure_from_exception(
                error,
                default_stage=CanonicalFailureStage.RUNTIME,
            ) from error
        return CanonicalGameProduct(
            evidence=evidence,
            replay_executor=self.replay_executor,
        )


class Classic7RuntimeFactory:
    """Build one external-backend runtime only after a durable attempt claim."""

    def __init__(
        self,
        *,
        runtime_config: Mapping[str, Any],
        backends: Mapping[str, Any],
        configured_call_limit: int,
    ) -> None:
        self._raw_config = deepcopy(dict(runtime_config))
        self._normalized = normalize_runtime_config(self._raw_config)
        self._backends = dict(backends)
        if isinstance(configured_call_limit, bool) or not isinstance(
            configured_call_limit,
            int,
        ) or configured_call_limit <= 0:
            raise ValueError("configured_call_limit must be a positive integer")
        self._configured_call_limit = configured_call_limit

    def __call__(
        self,
        *,
        plan: CollectionPlan,
        claim: AttemptClaim,
    ) -> _RecordedRuntime:
        game_id = (
            f"{plan.collection_id}-game-{claim.ordinal:06d}-seed-{claim.seed}"
        )
        audit = CanonicalCallAudit(
            plan=plan,
            configured_call_limit=self._configured_call_limit,
        )
        wrapped_backends = audited_backends(self._backends, audit)
        env, agents, roles, _profiles = build_runtime(
            deepcopy(self._raw_config),
            log_save_path=None,
            random_seed=claim.seed,
            backends=wrapped_backends,
        )
        belief_collector = PlayingAgentBeliefObservationCollector(
            plan=plan,
            reporter=PlayingAgentBeliefReporter(audit_hook=audit),
            agents=agents,
        )
        recorder = CanonicalGameRecorder(
            plan=plan,
            claim=claim,
            game_id=game_id,
            belief_collector=belief_collector,
            call_audit=audit,
            runtime_configuration=self._normalized,
        )
        replay = classic7_replay_executor(self._normalized)
        return _RecordedRuntime(
            env=env,
            agents=agents,
            roles=roles,
            recorder=recorder,
            call_audit=audit,
            replay_executor=replay,
        )


def classic7_replay_executor(runtime_config):
    env_config = deepcopy(normalize_runtime_config(runtime_config)["env_config"])
    env_config["log_save_path"] = None

    def replay_environment(seed):
        return WerewolfTextEnvV0(**env_config, speech_perceiver=_ReplaySpeechPerceiver(), random_seed=seed)

    return make_classic7_replay_executor(
        identity="classic7-runtime-action-replay-v1", environment_factory=replay_environment)


__all__ = ["Classic7RuntimeFactory", "classic7_replay_executor"]
