"""Research-only contextual DTRS and public speech intents; no runtime effects.

The event is an observer's effective suspicion support intersecting live wolves,
NOT the observer's role or an action-success probability. Costs and the speech
mapping are normative hypotheses. No fitting, labels, artifact I/O or env step.
"""
from dataclasses import dataclass
from fractions import Fraction

from scripts.speech_planning import PlanningAction, SpeechPlan, generate_candidates
from scripts.speech_realization import PublicRealizationContext
from werewolf.agents.prompt_template_v0 import PublicClaim
from werewolf.canonical_collection.pre import AuthoritativePREPrefix
from werewolf.canonical_collection.public_history import PLAYER_IDS
from werewolf.observer_dtrs.context import ObserverContext, contexts_for_pre
from werewolf.observer_dtrs.d2_protocol import FrozenProtocol
from werewolf.observer_dtrs.decision import Decision, LossSpec, Region, TiePolicy, decide
from werewolf.observer_dtrs.estimation import FittedModel, validate_model
from werewolf.observer_dtrs.features import ObserverFeatures, observer_features
from werewolf.observer_dtrs.probabilities import ProbabilityMatrix, ProbabilityRow


POLICY_VERSION = "observer_dtrs_contextual_speech_research_v1"
_MISS_MULTIPLIER = Fraction(2)
_DISCUSSION_DEFERRAL = Fraction(1, 4)
_PK_DEFERRAL = Fraction(2, 5)


def contextual_loss(context: ObserverContext, *, tie_policy: TiePolicy) -> LossSpec:
    """Predeclared prototype rule using only public phase and legal live-team count.

    A missed observer-level threat costs twice as much in PK and twice as much
    with one living wolf. Deferral costs 1/4 of each error in discussion, 2/5 in
    PK. These choices are illustrative utilities, not fitted/validated effects.
    """
    if not isinstance(context, ObserverContext):
        raise TypeError("contextual loss requires a legal ObserverContext")
    miss = _MISS_MULTIPLIER ** (int(context.phase == "pk_discussion")
                                + int(context.wolf_count == 1))
    fraction = _PK_DEFERRAL if context.phase == "pk_discussion" else _DISCUSSION_DEFERRAL
    return LossSpec(c_fp=1, c_fn=miss, b1=fraction * miss, b0=fraction,
                    tie_policy=tie_policy)


@dataclass(frozen=True)
class ContextualSpeechProposal:
    """Private audit plus a separate existing two-field public intent transport.

    A proposal does not authorize canonical collection, env mutation or a vote.
    The caller explicitly selects ONE observer; no observer aggregation is implied.
    """
    context: ObserverContext
    features: ObserverFeatures
    model_digest: str
    estimator_level: str
    loss: LossSpec
    decision: Decision
    plan: SpeechPlan
    mapping_reason: str
    public_context: PublicRealizationContext

    def public_payload(self) -> dict:
        return self.plan.public_payload()

    def audit_record(self) -> dict:
        """Private: never pass this record to the language Actor or public logs."""
        return {"schema_version": POLICY_VERSION, "scope": "research_prototype_only",
                "context": self.context.to_record(), "features": self.features.to_record(),
                "estimator": {"model_digest": self.model_digest, "level": self.estimator_level},
                "loss": self.loss.to_record(), "decision": self.decision.to_record(),
                "speech_intent": self.public_payload(), "mapping_reason": self.mapping_reason}


def estimate_event(context, prediction, *, model, protocol, loss):
    """Shared D2 -> D3 seam for prototype and opt-in verified speech adapters."""
    features = observer_features(context, prediction)
    if not isinstance(model, FittedModel) or not isinstance(protocol, FrozenProtocol):
        raise TypeError("require an existing D2 FittedModel and frozen D2 protocol")
    validate_model(model, protocol)
    if context.game_id in model.record["source"]["game_sizes"]:
        raise ValueError("current game belongs to the estimator training partition")
    estimate = model.predict_detail(features)
    decision = decide(estimate["probability"], loss)
    return features, estimate, decision


def propose_speech(pre: AuthoritativePREPrefix, *, known_wolf_ids: tuple[str, ...],
                   observer_id: str, prediction: ProbabilityRow | ProbabilityMatrix,
                   model: FittedModel, protocol: FrozenProtocol,
                   tie_policy: TiePolicy) -> ContextualSpeechProposal:
    """Legal PRE + identity-bound Q -> D2 event estimate -> contextual risk -> intent.

    known_wolf_ids must come from the acting wolf's own legal observation, never
    a role sidecar. The in-memory model is validated against D2; this is NOT a
    formal artifact loader or ToM source/seal admission. Use synthetic inputs for
    this prototype; frozen experiments retain their original source/entry points.
    """
    contexts = contexts_for_pre(pre, known_wolf_ids)
    context = next((c for c in contexts if c.observer_id == observer_id), None)
    if context is None:
        raise ValueError("require a living wolf speaker and living non-wolf observer")
    if context.wolf_count >= len(context.alive_player_ids) - context.wolf_count:
        raise ValueError("game has already reached the Classic7 wolf victory condition")
    # This also validates the public cyclic speaking order and real PK ballot.
    candidates = generate_candidates(pre)
    loss = contextual_loss(context, tie_policy=tie_policy)
    features, estimate, decision = estimate_event(
        context, prediction, model=model, protocol=protocol, loss=loss)

    if decision.region is Region.POS:
        plan = SpeechPlan(PlanningAction.SELF_DEFEND)
        reason = "POS_SELF_DEFENSE_HYPOTHESIS"
    elif decision.region is Region.BND:
        # OPPOSE is a public critique/question, not a role accusation or Probe.
        # Existing eligibility requires the observer to have spoken today.
        review = SpeechPlan(PlanningAction.OPPOSE, observer_id)
        if review in candidates:
            plan, reason = review, "BND_PUBLIC_BASIS_REVIEW"
        else:
            plan = SpeechPlan(PlanningAction.NO_COMMITMENT)
            reason = "BND_NO_CURRENT_DAY_CLAIM"
    else:
        plan = SpeechPlan(PlanningAction.NO_COMMITMENT)
        reason = "NEG_NO_COMMITMENT_HYPOTHESIS"
    if plan not in candidates:
        raise ValueError("selected intent is not legal at this public PRE")
    claims = tuple(PublicClaim(
        claim_id=e.event_id, time=f"{e.temporal_state.day}_{e.temporal_state.phase.value}",
        event="speech_pk" if e.temporal_state.phase.value == "pk_discussion" else "speech",
        speaker=PLAYER_IDS.index(e.speaker) + 1, raw_text=e.raw_text,
    ) for e in pre.public_event_history.events
        if e.event_type == "public_speech" and e.temporal_state.day == context.day)
    public = PublicRealizationContext(context.acting_wolf, context.day, context.phase, claims)
    return ContextualSpeechProposal(context, features, model.digest, estimate["level"],
                                    loss, decision, plan, reason, public)
