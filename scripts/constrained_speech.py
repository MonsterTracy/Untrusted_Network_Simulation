"""Phase 4A public gate and injectable provider; no ToM execution or policy."""
from dataclasses import dataclass
from typing import Callable

from scripts.speech_planning import SpeechPlan, generate_candidates
from scripts.counterfactual_tom import derive_public_phase_speaker_order
from scripts.verified_speech_commit import realize_and_commit
from werewolf.canonical_collection.pre import AuthoritativePREPrefix
from werewolf.canonical_collection.public_history import PLAYER_IDS


@dataclass(frozen=True)
class PublicOpportunity:
    parent: AuthoritativePREPrefix
    speaker: str
    candidates: tuple[SpeechPlan, ...]


PlanProvider = Callable[[PublicOpportunity], SpeechPlan]


@dataclass(frozen=True)
class PublicEligibility:
    eligible: bool
    reason: str
    opportunity: PublicOpportunity


def public_eligibility(parent):
    """Shared C0/C1 public gate; malformed PRE raises, never means untreated."""
    order = derive_public_phase_speaker_order(parent)
    candidates = generate_candidates(parent)
    if not candidates:
        raise ValueError("empty public candidate set")
    opportunity = PublicOpportunity(parent, parent.current_speaker, candidates)
    if order[-1] == parent.current_speaker:
        return PublicEligibility(False, "LAST_SAME_PHASE_SPEAKER", opportunity)
    return PublicEligibility(True, "SAME_PHASE_NEXT_SPEAKER", opportunity)


@dataclass(frozen=True)
class SpeechTreatment:
    """Private runner assignment; public eligibility never receives these fields."""
    players: tuple[str, ...] = ()
    roles: tuple[str, ...] = ("Werewolf",)

    def __post_init__(self):
        if (type(self.players) is not tuple or type(self.roles) is not tuple
                or any(p not in PLAYER_IDS for p in self.players)
                or any(r not in ('Werewolf', 'Villager', 'Seer', 'Witch') for r in self.roles)
                or len(set(self.players)) != len(self.players)
                or len(set(self.roles)) != len(self.roles)):
            raise ValueError("invalid explicit speech treatment assignment")

    def assigned(self, speaker, private_role):
        return speaker in self.players or private_role in self.roles


@dataclass(frozen=True)
class SpeechTrace:
    game_id: str
    boundary_id: str
    parent_digest: str
    speaker: str
    day: int
    phase: str
    treatment_assigned: bool
    public_eligible: bool | None
    eligibility_reason: str
    planning_mode: str
    selected_plan: SpeechPlan | None
    status: str
    error_type: str | None


class ConstrainedExecutionFailure(RuntimeError):
    """Terminal eligible/provider/public-validation failure, never untreated."""


def handle_speech(*, env, actor, recorder, provider, assigned, trace):
    """Return committed env step result, or None only for declared untreated.

    Called only in constrained mode after the real PRE/handoff is collected.
    The provider receives no env, Actor, assignment or private roles.
    """
    parent = recorder._pending.prefix
    eligibility = None
    selected = None
    def record(status, error=None):
        trace.append(SpeechTrace(parent.game_id, parent.boundary_id, parent.prefix_digest,
            parent.current_speaker, parent.public_temporal_state.day,
            parent.public_temporal_state.phase.value, assigned,
            None if eligibility is None else eligibility.eligible,
            "PUBLIC_VALIDATION_FAILED" if eligibility is None else eligibility.reason,
            "constrained", selected, status, None if error is None else type(error).__name__))
    try:
        eligibility = public_eligibility(parent)
        if not assigned:
            record("UNTREATED_NOT_ASSIGNED")
            return None
        if not eligibility.eligible:
            record("INELIGIBLE_FOR_CONSTRAINED_TREATMENT")
            return None
        candidate = provider(eligibility.opportunity)
        if not isinstance(candidate, SpeechPlan) or candidate not in eligibility.opportunity.candidates:
            raise ValueError("provider must return one generated SpeechPlan")
        selected = candidate
        result = realize_and_commit(selected.public_payload(), env=env, actor=actor, recorder=recorder)
    except Exception as error:
        record("CONSTRAINED_EXECUTION_FAILED", error)
        raise ConstrainedExecutionFailure("constrained speech execution failed") from error
    record("VERIFIED_COMMIT_SUCCEEDED")
    return result
