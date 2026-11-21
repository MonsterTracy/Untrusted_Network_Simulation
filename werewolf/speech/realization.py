"""Phase 3A: public payload -> existing Actor -> one perception -> verification.

No planner/ToM imports, cognition, environment commit, or semantic retry.
"""
from dataclasses import dataclass
from contextlib import nullcontext

from werewolf.agents.gpt_agent import GPTAgent
from werewolf.agents.prompt_template_v0 import DiscussionAct, PublicClaim
from werewolf.canonical_collection.public_history import PLAYER_IDS
from werewolf.canonical_collection.speech import V1SpeechAction
from werewolf.speech.speech_perceiver import SpeechParseAuditResult, SpeechPerceiver


_ACTIONS = {
    "ACCUSE_WOLF": "point_as_werewolf",
    "CLEAR": "point_as_non_werewolf",
    "SUPPORT": "support",
    "OPPOSE": "oppose",
    "SELF_DEFEND": "point_as_non_werewolf",
    "NO_COMMITMENT": "no_commitment",
}


def expected_action(payload, speaker):
    """Accept only the two-field SpeechPlan.public_payload() transport."""
    if type(payload) is not dict or set(payload) != {"action", "target"}:
        raise ValueError("expected action/target public payload only")
    action, target = payload["action"], payload["target"]
    if type(action) is not str or action not in _ACTIONS:
        raise ValueError("unsupported V1 plan action")
    if speaker not in PLAYER_IDS:
        raise ValueError("expected canonical speaker")
    if action in ("SELF_DEFEND", "NO_COMMITMENT"):
        if target is not None:
            raise ValueError("targetless plan requires None")
        target = speaker if action == "SELF_DEFEND" else None
    elif target not in PLAYER_IDS or target == speaker:
        raise ValueError("expected canonical non-self target")
    return V1SpeechAction(speaker, _ACTIONS[action], target)


def actor_intent(payload, speaker):
    expected = expected_action(payload, speaker)
    target = None if expected.object is None else PLAYER_IDS.index(expected.object) + 1
    return (DiscussionAct(expected.action, target),)


@dataclass(frozen=True)
class PublicRealizationContext:
    speaker: str
    day: int
    phase: str
    claims: tuple[PublicClaim, ...] = ()

    def __post_init__(self):
        if self.speaker not in PLAYER_IDS or type(self.day) is not int or self.day < 0:
            raise ValueError("invalid public speaker/day")
        if self.phase not in ("discussion", "pk_discussion"):
            raise ValueError("unsupported public phase")
        if type(self.claims) is not tuple or any(not isinstance(c, PublicClaim) for c in self.claims):
            raise TypeError("claims must be a public PublicClaim tuple")

    def observation(self):
        phase = "speech_pk" if self.phase == "pk_discussion" else "speech"
        return {"current_act_idx": PLAYER_IDS.index(self.speaker) + 1,
                "day": self.day, "phase": f"{self.day}_{phase}"}


class SemanticVerificationError(ValueError):
    def __init__(self, perception):
        super().__init__("speech perception does not exactly match selected plan")
        self.perception = perception


def verify_semantics(payload, speaker, perception):
    """Inspect without modifying/repairing; return the identical audit result."""
    expected = expected_action(payload, speaker)
    if not isinstance(perception, SpeechParseAuditResult):
        raise TypeError("verification requires SpeechParseAuditResult")
    if (perception.parse_status != "ok" or perception.error_type is not None
            or perception.error_message is not None
            or perception.normalized_actions != [expected.to_record()]):
        raise SemanticVerificationError(perception)
    return perception


@dataclass(frozen=True)
class VerifiedRealization:
    speech: str
    expected: V1SpeechAction
    perception: SpeechParseAuditResult


def realize_parse_verify(payload, context, *, actor, perceiver, actor_context=None, perception_context=None):
    """Return speech and the same perception for a future commit adapter.

    Context must contain authentic public claims. Candidate eligibility belongs
    to Phase 2. This result is not canonical evidence or an env commit permit.
    """
    if not isinstance(context, PublicRealizationContext):
        raise TypeError("expected PublicRealizationContext")
    if not isinstance(actor, GPTAgent) or not isinstance(perceiver, SpeechPerceiver):
        raise TypeError("expected existing GPTAgent and SpeechPerceiver")
    expected = expected_action(payload, context.speaker)
    intent = actor_intent(payload, context.speaker)
    temperature, max_tokens = actor._request_limits()
    with nullcontext() if actor_context is None else actor_context:
        speech = actor._generate_public_speech(
            context.observation(), discussion_acts=intent, claim_catalog=context.claims,
            temperature=temperature, max_tokens=max_tokens)
    with nullcontext() if perception_context is None else perception_context:
        perception = perceiver.parse_with_audit(
            speaker=PLAYER_IDS.index(context.speaker) + 1, speech=speech,
            day=context.day, phase=context.phase)
    verify_semantics(payload, context.speaker, perception)
    return VerifiedRealization(speech, expected, perception)
