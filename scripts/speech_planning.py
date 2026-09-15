"""Phase 2 decision primitives. No gameplay, LLM, or artifact-writing path."""

from dataclasses import dataclass
import math

from scripts.counterfactual_tom import (
    CounterfactualSpeechAction, PlanningAction, ConsumerProvenance,
    derive_public_phase_speaker_order, UnsupportedOpportunityError,
)
from scripts.suspicion_objectives import WolfSuspicionMass, wolf_suspicion_mass
from werewolf.canonical_collection.public_history import PLAYER_IDS
from scripts.speech_versions import CANDIDATE_ORDER_VERSION


SELECTOR_VERSION = "alive_conditional_argmin_exact_tie_v1"
# Frozen Phase 1 enum declaration order, then canonical player seat order.
ACTION_ORDER = tuple(PlanningAction)


@dataclass(frozen=True, slots=True)
class SpeechPlan:
    action: PlanningAction
    target: str | None = None

    def __post_init__(self):
        if not isinstance(self.action, PlanningAction):
            raise ValueError("SpeechPlan requires a V1 PlanningAction")
        if self.action in (PlanningAction.SELF_DEFEND, PlanningAction.NO_COMMITMENT):
            if self.target is not None:
                raise ValueError("targetless SpeechPlan requires target=None")
        elif self.target not in PLAYER_IDS:
            raise ValueError("targeted SpeechPlan requires a canonical player")

    def public_payload(self):
        """Only the selected public intent; no score or private state."""
        return {"action": self.action.value, "target": self.target}

    def for_speaker(self, speaker):
        """Delegate semantic representation to Phase 1 (including self rules)."""
        return CounterfactualSpeechAction(speaker, self.action, self.target)


def _order_key(plan):
    return (ACTION_ORDER.index(plan.action),
            -1 if plan.target is None else PLAYER_IDS.index(plan.target))


def generate_candidates(parent):
    """Public-only candidates, including on a last-speaker PRE.

    Generation and evaluability are distinct: the final speaker still has
    SELF_DEFEND/NO_COMMITMENT, but Phase 1 cannot evaluate that opportunity.
    SUPPORT/OPPOSE refer to any prior speech on the same day, across phases.
    """
    derive_public_phase_speaker_order(parent)  # validates real public evidence
    targets = tuple(p for p in PLAYER_IDS
                    if p in parent.alive_observer_ids and p != parent.current_speaker)
    spoken = {e.speaker for e in parent.public_event_history.events
              if e.event_type == "public_speech"
              and e.temporal_state.day == parent.public_temporal_state.day}
    plans = []
    for action in ACTION_ORDER:
        if action in (PlanningAction.SELF_DEFEND, PlanningAction.NO_COMMITMENT):
            plans.append(SpeechPlan(action))
        else:
            for target in targets:
                if action in (PlanningAction.SUPPORT, PlanningAction.OPPOSE) and target not in spoken:
                    continue
                plans.append(SpeechPlan(action, target))
    return tuple(plans)


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    plan: SpeechPlan
    parent_prefix_digest: str
    consumer_provenance: ConsumerProvenance
    suspicion: WolfSuspicionMass
    current_suspicion: WolfSuspicionMass | None
    delta: float | None
    candidate_order_version: str = CANDIDATE_ORDER_VERSION

    @property
    def primary_score(self):
        return self.suspicion.alive_conditional_team_mass


def evaluate_candidates(parent, candidates, consumer, *, alive_wolves, current_predictor=None):
    """Evaluate the complete public candidate tuple with an injected Phase 1 consumer.

    Production callers supply CounterfactualToMConsumer, already seal-validated.
    Test doubles may implement its probabilities/provenance interface. Wolf IDs
    flow only to the peripheral objective, never to either inference call.
    Optional current_predictor is the existing real-PRE SealedFinalPredictor;
    its experiment/seal/checkpoint/condition must match the consumer exactly.
    No arbitrary baseline matrix is accepted. No records are written.
    """
    expected = generate_candidates(parent)
    if not isinstance(candidates, tuple) or candidates != expected:
        raise ValueError("evaluation requires the complete canonical candidate tuple")
    order = derive_public_phase_speaker_order(parent)
    if parent.current_speaker == order[-1]:
        raise UnsupportedOpportunityError("current speaker is the last phase speaker")
    # Validate peripheral population before spending inference calls.
    if (not isinstance(alive_wolves, tuple) or not alive_wolves
            or any(p not in parent.alive_observer_ids for p in alive_wolves)
            or len(set(alive_wolves)) != len(alive_wolves)
            or parent.current_speaker not in alive_wolves):
        raise ValueError("alive_wolves must be unique living wolves including the acting speaker")
    if set(alive_wolves) == set(parent.alive_observer_ids):
        raise ValueError("empty O_t")
    provenance = consumer.provenance
    if not isinstance(provenance, ConsumerProvenance):
        raise TypeError("consumer must expose Phase 1 ConsumerProvenance")

    def score(probabilities):
        return wolf_suspicion_mass(probabilities, alive_players=parent.alive_observer_ids,
                                   alive_wolves=alive_wolves, self_player=parent.current_speaker)

    baseline = None
    if current_predictor is not None:
        from werewolf.tom.final_evaluation import SealedFinalPredictor
        if not isinstance(current_predictor, SealedFinalPredictor):
            raise TypeError("current baseline requires SealedFinalPredictor")
        if (current_predictor.experiment.digest != provenance.parent_experiment_digest
                or current_predictor.seal["record_digest"] != provenance.parent_final_seal_digest
                or current_predictor.checkpoint.manifest_digest != provenance.checkpoint_digest
                or current_predictor.condition != provenance.condition):
            raise ValueError("current predictor provenance differs from candidate consumer")
        baseline = score(current_predictor.log_probabilities(parent).exp())
    results = []
    for plan in candidates:
        probabilities = consumer.probabilities(parent, plan.for_speaker(parent.current_speaker))
        suspicion = score(probabilities)
        delta = None if baseline is None else baseline.alive_conditional_team_mass - suspicion.alive_conditional_team_mass
        results.append(CandidateEvaluation(plan, parent.prefix_digest, provenance,
                                           suspicion, baseline, delta))
    return tuple(results)


def select_minimum_suspicion(candidates, evaluations):
    """Replaceable decision layer: exact argmin, canonical-order exact ties.

    No evaluator or private context is called here. A future No-ToM decision
    function can consume the same candidates without invoking evaluation.
    """
    if (not isinstance(candidates, tuple) or not candidates
            or any(not isinstance(p, SpeechPlan) for p in candidates)
            or len(set(candidates)) != len(candidates)
            or candidates != tuple(sorted(candidates, key=_order_key))):
        raise ValueError("selector requires unique candidates in canonical order")
    if (not isinstance(evaluations, tuple) or len(evaluations) != len(candidates)
            or any(not isinstance(e, CandidateEvaluation) for e in evaluations)):
        raise ValueError("selector requires complete candidate evaluations")
    by_plan = {e.plan: e for e in evaluations}
    if len(by_plan) != len(candidates) or set(by_plan) != set(candidates):
        raise ValueError("evaluation coverage mismatch")
    first = evaluations[0]
    for e in evaluations:
        if (e.parent_prefix_digest != first.parent_prefix_digest
                or e.consumer_provenance != first.consumer_provenance
                or e.current_suspicion != first.current_suspicion
                or e.candidate_order_version != CANDIDATE_ORDER_VERSION):
            raise ValueError("mixed evaluation provenance")
        if not math.isfinite(e.primary_score) or not 0 <= e.primary_score <= 1:
            raise ValueError("invalid Alive-Conditional Wolf Suspicion Mass")
    # Python min keeps the first exact minimum. No tolerance, weights or RNG.
    return min(candidates, key=lambda plan: by_plan[plan].primary_score)
