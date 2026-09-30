"""Fail-closed intervention planning and a versioned live-state closure audit."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import random

from werewolf.artifact_io import canonical_json_bytes, sha256_bytes
from werewolf.phase2_decision_opportunity import Phase2DecisionOpportunityV1
from werewolf.phase2_treatment import Phase2TreatmentV1


class CheckpointUnavailable(RuntimeError):
    """A public PRE is insufficient to clone a complete intervention state."""


def _stable_local_value(value):
    """Exact local-state fingerprint input, never a restore codec."""
    if value is None or type(value) in (bool, int, float, str):
        return value
    if isinstance(value, Enum):
        return {"enum": type(value).__qualname__, "value": _stable_local_value(value.value)}
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, random.Random):
        return {"random_state": _stable_local_value(value.getstate())}
    if isinstance(value, (list, tuple)):
        return {"type": type(value).__name__, "items": [_stable_local_value(v) for v in value]}
    if isinstance(value, (set, frozenset)):
        items = [_stable_local_value(v) for v in value]
        return {"type": type(value).__name__, "items": sorted(
            items, key=lambda item: canonical_json_bytes(item))}
    if isinstance(value, dict):
        items = [(_stable_local_value(k), _stable_local_value(v))
                 for k, v in value.items()]
        return {"mapping": sorted(items, key=lambda pair: canonical_json_bytes(pair[0]))}
    if is_dataclass(value) and not isinstance(value, type):
        return {"class": type(value).__qualname__,
                "fields": {field.name: _stable_local_value(getattr(value, field.name))
                           for field in fields(value)}}
    if type(value).__module__ == "werewolf.helper.log_utils" and hasattr(value, "__dict__"):
        return {"class": type(value).__qualname__,
                "fields": _stable_local_value(vars(value))}
    raise CheckpointUnavailable(f"unsupported local state type: {type(value).__name__}")


def local_env_state_digest(env) -> str:
    """Detect env drift; explicitly excludes the external speech perceiver."""
    values = {key: value for key, value in vars(env).items()
              if key != "speech_perceiver"}
    return sha256_bytes(canonical_json_bytes(_stable_local_value(values)))


CHECKPOINT_CLOSURE_VERSION = "phase2_checkpoint_closure_v1"
CHECKPOINT_BLOCKERS = (
    "CANONICAL_RECORDER_AND_CALL_AUDIT_RESTORE_UNVERIFIED",
    "AGENT_AND_BACKEND_STATE_RESTORE_UNVERIFIED",
    "PROCESS_RNG_AND_EXTERNAL_GENERATION_NOT_PAIRED",
    "LIVE_PRE_CLONE_AND_RESTORE_EQUIVALENCE_UNVERIFIED",
)


@dataclass(frozen=True)
class Phase2CheckpointClosureV1:
    """An identity-bound audit of the missing closure, never a checkpoint."""

    game_id: str
    boundary_id: str
    prefix_digest: str
    acting_wolf: str
    phase: str
    public_history_digest: str
    local_env_digest: str
    blockers: tuple[str, ...] = CHECKPOINT_BLOCKERS
    executable: bool = False

    def __post_init__(self):
        if (not all(isinstance(value, str) and value for value in (
                self.game_id, self.boundary_id, self.prefix_digest,
                self.acting_wolf, self.phase, self.public_history_digest,
                self.local_env_digest))
                or self.phase not in ("speech", "speech_pk")
                or self.blockers != CHECKPOINT_BLOCKERS or self.executable):
            raise CheckpointUnavailable("incomplete Phase-2 checkpoint closure")

    @property
    def identity(self) -> tuple[str, str, str, str, str]:
        return (self.game_id, self.boundary_id, self.prefix_digest,
                self.acting_wolf, self.phase)

    def to_record(self) -> dict:
        return {"schema_version": CHECKPOINT_CLOSURE_VERSION,
                "identity": list(self.identity),
                "public_history_digest": self.public_history_digest,
                "local_env_digest": self.local_env_digest,
                "blockers": list(self.blockers), "executable": False}

    def digest(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_record()))


def assess_phase2_checkpoint_closure(opportunity: Phase2DecisionOpportunityV1,
                                     *, env, recorder) -> Phase2CheckpointClosureV1:
    """Check current live PRE identity before declaring restore unavailable.

    This deliberately records no private state bytes: no complete state codec
    or backend replay contract has passed the required equivalence tests.
    """
    if not isinstance(opportunity, Phase2DecisionOpportunityV1):
        raise CheckpointUnavailable("current Phase-2 opportunity is required")
    context = opportunity.legal_context
    pending = getattr(recorder, "_pending", None)
    prefix = getattr(pending, "prefix", None)
    handoff = getattr(pending, "handoff", None)
    if (getattr(recorder, "_env", None) is not env or prefix is None
            or getattr(pending, "raw_action", None) is not None
            or handoff is None
            or (prefix.game_id, prefix.boundary_id, prefix.prefix_digest,
                prefix.current_speaker) !=
               (context.game_id, context.boundary_id, context.prefix_digest,
                context.acting_wolf)
            or (getattr(handoff, "boundary_id", None),
                getattr(handoff, "prefix_digest", None),
                getattr(handoff, "observer_id", None)) !=
               (context.boundary_id, context.prefix_digest, context.acting_wolf)
            or getattr(env, "phase", None) != context.phase
            or getattr(env, "current_act_idx", None) != int(context.acting_wolf[-1]) - 1
            or getattr(getattr(prefix, "public_event_history", None), "digest", None)
               != context.public_history_digest):
        raise CheckpointUnavailable("live recorder/environment PRE differs from opportunity")
    return Phase2CheckpointClosureV1(
        context.game_id, context.boundary_id, context.prefix_digest,
        context.acting_wolf, context.phase, context.public_history_digest,
        local_env_state_digest(env))


def validate_treatment_checkpoint_binding(opportunity: Phase2DecisionOpportunityV1,
                                          treatment: Phase2TreatmentV1,
                                          closure: Phase2CheckpointClosureV1) -> None:
    """Validate PRE lineage only; this does not make the arm executable."""
    if (not isinstance(opportunity, Phase2DecisionOpportunityV1)
            or not isinstance(treatment, Phase2TreatmentV1)
            or not isinstance(closure, Phase2CheckpointClosureV1)
            or treatment.opportunity_digest != opportunity.digest()
            or treatment.opportunity_identity != opportunity.identity
            or closure.identity != opportunity.identity[:5]
            or closure.public_history_digest != opportunity.legal_context.public_history_digest):
        raise CheckpointUnavailable("treatment/opportunity/checkpoint PRE mismatch")


@dataclass(frozen=True)
class PlannedBranchV1:
    checkpoint_identity: str
    treatment_id: str
    opportunity_digest: str
    downstream_seed: int
    branch_identity: str
    executable: bool = False

    def __post_init__(self):
        if (not self.checkpoint_identity or not self.treatment_id or
                type(self.downstream_seed) is not int or self.downstream_seed < 0 or
                self.executable or self.branch_identity != sha256_bytes(canonical_json_bytes([
                    self.checkpoint_identity, self.treatment_id, self.downstream_seed]))):
            raise CheckpointUnavailable("branch lacks a verified full-state checkpoint")


def capture_intervention_checkpoint(*_args, **_kwargs):
    raise CheckpointUnavailable(
        "complete PRE clone/restore of environment, agents, recorder, backend and RNG is unimplemented")


def restore_intervention_checkpoint(*_args, **_kwargs):
    raise CheckpointUnavailable("no verified full-state checkpoint can be restored")


def clone_phase2_checkpoint(*_args, **_kwargs):
    raise CheckpointUnavailable("no verified full-state checkpoint can be cloned")


def run_intervention_branch(*_args, **_kwargs):
    raise CheckpointUnavailable("intervention branches cannot run from a public PRE or plan")


def plan_paired_branches(checkpoint_identity: str, treatments: tuple[Phase2TreatmentV1, ...],
                         *, downstream_seed: int) -> tuple[PlannedBranchV1, ...]:
    """Record deterministic intent only; no branch outcome may be fabricated."""
    if (not isinstance(checkpoint_identity, str) or not checkpoint_identity
            or not treatments or type(downstream_seed) is not int or downstream_seed < 0
            or any(not isinstance(treatment, Phase2TreatmentV1)
                   or treatment.assignment_source != "paired_branch" for treatment in treatments)
            or len({treatment.treatment_id for treatment in treatments}) != len(treatments)
            or len({treatment.opportunity_digest for treatment in treatments}) != 1
            or len({treatment.action for treatment in treatments}) != len(treatments)):
        raise CheckpointUnavailable("paired branches require one opportunity and unique controlled arms")
    return tuple(PlannedBranchV1(
        checkpoint_identity, treatment.treatment_id, treatment.opportunity_digest,
        downstream_seed, sha256_bytes(canonical_json_bytes([
            checkpoint_identity, treatment.treatment_id, downstream_seed])))
        for treatment in treatments)
