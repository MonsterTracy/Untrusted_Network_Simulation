"""Independent Phase-2 speech semantics and language-execution verification.

This sidecar neither extends canonical V1 speech actions nor publishes a game
event. The perceiver interface receives public text/context, never a plan.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from werewolf.canonical_collection.pre import validate_authoritative_pre_prefix
from werewolf.canonical_collection.public_history import PLAYER_IDS
from werewolf.phase2_actions import (
    Action, ActionContextV1, InformationRequestV1, NO_STANCE,
    Phase2SemanticPlanV1, REQUEST_TYPE, verify_plan,
)


LANGUAGE_VERSION = "phase2_speech_semantic_v1"
OTHER_REQUEST = "OTHER"


class Phase2LanguageError(ValueError):
    """A generated speech or its independent perception violates the contract."""


@dataclass(frozen=True)
class PublicSpeechTurnV1:
    speaker: str
    text: str

    def __post_init__(self) -> None:
        if self.speaker not in PLAYER_IDS or not isinstance(self.text, str) or not self.text.strip():
            raise Phase2LanguageError("invalid public speech turn")

    def to_record(self) -> dict:
        return {"speaker": self.speaker, "text": self.text}


@dataclass(frozen=True)
class PublicLanguageContextV1:
    """Only current public state and already published speech may reach models."""

    phase: str
    speaker: str
    alive: tuple[str, ...]
    competition: tuple[str, ...]
    public_history_digest: str
    prior_public_speeches: tuple[PublicSpeechTurnV1, ...] = ()

    def __post_init__(self) -> None:
        if self.phase not in ("speech", "speech_pk") or self.speaker not in self.alive:
            raise Phase2LanguageError("invalid public speech opportunity")
        if (not isinstance(self.public_history_digest, str) or not self.public_history_digest
                or not isinstance(self.prior_public_speeches, tuple)
                or any(not isinstance(turn, PublicSpeechTurnV1)
                       for turn in self.prior_public_speeches)):
            raise Phase2LanguageError("invalid public history")
        for values in (self.alive, self.competition):
            if (not isinstance(values, tuple) or not values
                    or values != tuple(p for p in PLAYER_IDS if p in values)):
                raise Phase2LanguageError("public targets require canonical seat order")
        if self.phase == "speech" and self.competition != self.alive:
            raise Phase2LanguageError("ordinary competition must equal alive players")
        if not set(self.competition) <= set(self.alive):
            raise Phase2LanguageError("competition contains a dead player")

    def to_record(self) -> dict:
        return {"phase": self.phase, "speaker": self.speaker,
                "alive": list(self.alive), "competition": list(self.competition),
                "public_history_digest": self.public_history_digest,
                "prior_public_speeches": [turn.to_record() for turn in self.prior_public_speeches]}


def public_language_context_from_pre(pre, legal_context: ActionContextV1) -> PublicLanguageContextV1:
    """Build model-visible context from this PRE's event prefix, never later events."""
    validate_authoritative_pre_prefix(pre)
    if (pre.game_id != legal_context.game_id or pre.boundary_id != legal_context.boundary_id
            or pre.prefix_digest != legal_context.prefix_digest
            or pre.public_event_history.digest != legal_context.public_history_digest
            or pre.current_speaker != legal_context.acting_wolf):
        raise Phase2LanguageError("public language context differs from action PRE")
    turns = tuple(PublicSpeechTurnV1(event.speaker, event.raw_text)
                  for event in pre.public_event_history.events
                  if event.event_type == "public_speech")
    return PublicLanguageContextV1(legal_context.phase, legal_context.acting_wolf,
                                   legal_context.alive, legal_context.competition,
                                   legal_context.public_history_digest, turns)


@dataclass(frozen=True)
class Phase2SpeechSemanticV1:
    """Independent multi-label perception; plural fields reveal conflicting claims."""

    speaker: str
    phase: str
    commitment_targets: tuple[str, ...]
    rejected_targets: tuple[str, ...]
    vote_intent_targets: tuple[str, ...]
    information_requests: tuple[InformationRequestV1, ...]
    abstain_intent: bool = False
    private_fact_claim: bool = False

    def __post_init__(self) -> None:
        if self.speaker not in PLAYER_IDS or self.phase not in ("speech", "speech_pk"):
            raise Phase2LanguageError("invalid perceived speaker or phase")
        for name, values in (("commitment_targets", self.commitment_targets),
                             ("rejected_targets", self.rejected_targets),
                             ("vote_intent_targets", self.vote_intent_targets)):
            if (not isinstance(values, tuple) or len(values) != len(set(values))
                    or any(value not in PLAYER_IDS for value in values)):
                raise Phase2LanguageError(f"invalid perceived {name}")
        if (not isinstance(self.information_requests, tuple)
                or any(not isinstance(value, InformationRequestV1)
                       or value.target_j not in PLAYER_IDS
                       or value.addressee_j not in PLAYER_IDS
                       or value.request_type not in (REQUEST_TYPE, OTHER_REQUEST)
                       for value in self.information_requests)
                or type(self.abstain_intent) is not bool
                or type(self.private_fact_claim) is not bool):
            raise Phase2LanguageError("invalid perceived request or flags")

    @property
    def action_identity(self) -> str | None:
        if self.private_fact_claim or self.abstain_intent:
            return None
        if (len(self.commitment_targets) == len(self.vote_intent_targets) == 1
                and self.commitment_targets == self.vote_intent_targets
                and not self.information_requests):
            if not self.rejected_targets:
                return Action.PUSH.value
            if (len(self.rejected_targets) == 1
                    and self.rejected_targets[0] != self.commitment_targets[0]):
                return Action.REDIRECT.value
        if (not self.commitment_targets and not self.rejected_targets
                and not self.vote_intent_targets and len(self.information_requests) == 1
                and self.information_requests[0].request_type == REQUEST_TYPE):
            return Action.PROBE.value
        return None

    def to_record(self) -> dict:
        return {"schema_version": LANGUAGE_VERSION, "speaker": self.speaker,
                "phase": self.phase, "action_identity": self.action_identity,
                "commitment_targets": list(self.commitment_targets),
                "rejected_targets": list(self.rejected_targets),
                "vote_intent_targets": list(self.vote_intent_targets),
                "information_requests": [item.to_record() for item in self.information_requests],
                "abstain_intent": self.abstain_intent,
                "private_fact_claim": self.private_fact_claim}


def parse_perceived_semantics(raw: str) -> Phase2SpeechSemanticV1:
    """Strictly decode the independent perceiver's JSON transport."""
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as error:
        raise Phase2LanguageError("PERCEPTION_JSON_INVALID") from error
    fields = {"speaker", "phase", "commitment_targets", "rejected_targets",
              "vote_intent_targets", "information_requests", "abstain_intent",
              "private_fact_claim"}
    if not isinstance(value, dict) or set(value) != fields:
        raise Phase2LanguageError("PERCEPTION_FIELDS_INVALID")
    for name in ("commitment_targets", "rejected_targets", "vote_intent_targets"):
        if not isinstance(value[name], list):
            raise Phase2LanguageError("PERCEPTION_FIELDS_INVALID")
    requests = value["information_requests"]
    if not isinstance(requests, list) or any(not isinstance(r, dict)
            or set(r) != {"target_j", "addressee_j", "request_type"} for r in requests):
        raise Phase2LanguageError("PERCEPTION_REQUEST_INVALID")
    try:
        return Phase2SpeechSemanticV1(
            value["speaker"], value["phase"],
            tuple(value["commitment_targets"]), tuple(value["rejected_targets"]),
            tuple(value["vote_intent_targets"]),
            tuple(InformationRequestV1(**r) for r in requests),
            value["abstain_intent"], value["private_fact_claim"])
    except (TypeError, ValueError) as error:
        raise Phase2LanguageError("PERCEPTION_SEMANTICS_INVALID") from error


@dataclass(frozen=True)
class LanguageVerificationV1:
    valid: bool
    invalid_reason: str | None


def verify_language_execution(requested_plan: Phase2SemanticPlanV1,
                              perceived_semantics: Phase2SpeechSemanticV1,
                              legal_context: ActionContextV1) -> LanguageVerificationV1:
    """Compare independent text perception to the unchanged structured plan."""
    if not isinstance(perceived_semantics, Phase2SpeechSemanticV1):
        raise TypeError("expected independent Phase2SpeechSemanticV1")
    structured = verify_plan(legal_context, requested_plan)
    if not structured.valid:
        return LanguageVerificationV1(False, "STRUCTURED_PLAN_INVALID:" + structured.invalid_reason)
    perceived = perceived_semantics
    if perceived.speaker != legal_context.acting_wolf or perceived.phase != legal_context.phase:
        return LanguageVerificationV1(False, "PERCEIVED_OPPORTUNITY_MISMATCH")
    if perceived.private_fact_claim:
        return LanguageVerificationV1(False, "PRIVATE_FACT_CLAIM")
    if perceived.abstain_intent:
        return LanguageVerificationV1(False, "UNREQUESTED_ABSTAIN")
    j = requested_plan.candidate_j
    if requested_plan.action is Action.PUSH:
        if perceived.commitment_targets != (j,):
            return LanguageVerificationV1(False, "PUSH_COMMITMENT_MISMATCH")
        if perceived.vote_intent_targets != (j,):
            return LanguageVerificationV1(False, "PUSH_VOTE_INTENT_MISMATCH")
        if perceived.rejected_targets or perceived.information_requests:
            return LanguageVerificationV1(False, "PUSH_EXTRA_SEMANTICS")
    elif requested_plan.action is Action.REDIRECT:
        k = requested_plan.redirect_target
        if perceived.rejected_targets != (j,):
            return LanguageVerificationV1(False, "REDIRECT_REJECTION_MISMATCH")
        if perceived.commitment_targets != (k,):
            return LanguageVerificationV1(False, "REDIRECT_COMMITMENT_MISMATCH")
        if perceived.vote_intent_targets != (k,):
            return LanguageVerificationV1(False, "REDIRECT_VOTE_INTENT_MISMATCH")
        if perceived.information_requests:
            return LanguageVerificationV1(False, "REDIRECT_REQUEST_CONFLICT")
    elif requested_plan.action is Action.PROBE:
        if perceived.information_requests != (requested_plan.information_request,):
            return LanguageVerificationV1(False, "PROBE_REQUEST_MISMATCH")
        if perceived.commitment_targets or perceived.vote_intent_targets:
            return LanguageVerificationV1(False, "PROBE_COMMITMENT_CONFLICT")
        if perceived.rejected_targets:
            return LanguageVerificationV1(False, "PROBE_REJECTION_CONFLICT")
    else:
        return LanguageVerificationV1(False, "UNKNOWN_ACTION")
    if perceived.action_identity != requested_plan.action.value:
        return LanguageVerificationV1(False, "PERCEIVED_ACTION_MISMATCH")
    return LanguageVerificationV1(True, None)


def _public_block(context: PublicLanguageContextV1) -> str:
    history = "\n".join(f"{turn.speaker}: {turn.text}"
                        for turn in context.prior_public_speeches) or "（无此前公开发言）"
    return (f"阶段：{context.phase}\n当前发言者：{context.speaker}\n"
            f"存活玩家：{', '.join(context.alive)}\n"
            f"本阶段公开竞争目标：{', '.join(context.competition)}\n"
            f"此前已公开发言：\n{history}")


def build_realization_prompt(plan: Phase2SemanticPlanV1,
                             context: PublicLanguageContextV1,
                             failure_reason: str | None = None) -> str:
    """Tell the actor the frozen semantic plan, not any hidden-role assertion."""
    if plan.phase != context.phase or plan.acting_wolf != context.speaker:
        raise Phase2LanguageError("actor context does not match plan")
    if failure_reason is not None and (not isinstance(failure_reason, str) or not failure_reason):
        raise Phase2LanguageError("repair reason must be non-empty text")
    if plan.action is Action.PUSH:
        instruction = f"明确推动{plan.candidate_j}作为本轮主要投票/放逐目标。不得再推动另一主要目标。"
    elif plan.action is Action.REDIRECT:
        instruction = (f"明确说明本轮不以{plan.candidate_j}为主要放逐目标，"
                       f"并明确转而推动{plan.redirect_target}。拒绝前者不等于断言其是好人。")
    elif plan.action is Action.PROBE:
        instruction = (f"直接向{plan.candidate_j}提问：其当前主要怀疑或投票判断是什么，"
                       "依据哪些已经公开的信息。不要推动任何人的放逐票，也不要弃票表态。")
    else:
        raise Phase2LanguageError("unknown Phase-2 action")
    repair = "" if failure_reason is None else (
        f"\n上次独立验证未通过，原因代码：{failure_reason}。"
        "仅重新表达同一冻结计划；不可改变动作或目标，也不可引用上次发言。")
    return ("你只负责把已冻结的 Phase-2 公开策略语义写成一段自然的中文狼人杀发言。\n"
            + _public_block(context) + "\n冻结语义：" + instruction + repair +
            "\n只输出1至4句公开发言，玩家用player1至player7指称。"
            "不得声称掌握尚未公开的身份、夜间信息或未来结果；"
            "不得增加与冻结语义冲突的承诺、信息请求或角色断言。"
            "不要输出结构化字段、JSON或解释。")


def build_perception_prompt(text: str, context: PublicLanguageContextV1) -> str:
    """Independent parser input intentionally has no plan or requested action."""
    if not isinstance(text, str) or not text.strip():
        raise Phase2LanguageError("speech text must be non-empty")
    return ("只根据以下公开发言及公开阶段信息，独立提取 Phase-2 言语语义。"
            "你不知道生成者原计划，不能猜测其期望动作。"
            "历史只用于判断信息是否已经公开，不可把历史发言的动作算作本轮动作。\n"
            + _public_block(context) +
            "\n待解析的本轮公开发言：\n" + text +
            "\ncommitment_targets：所有被明确推动为本轮主要投票/放逐目标的玩家；"
            "即使出现互相冲突的多个目标，也须全部列出。"
            "rejected_targets：所有被明确表示本轮不作为主要放逐目标的玩家；"
            "不要把这解释为好人身份断言。"
            "vote_intent_targets：所有被明确提出要投票的玩家；没有则为空。"
            "information_requests：每个明确公开提问的主题玩家、被问玩家和请求类型；"
            "只有要求被问者说明其当前主要怀疑/投票判断及公开依据时，"
            f"类型才是{REQUEST_TYPE}，其他问题为{OTHER_REQUEST}。"
            "abstain_intent 仅在明确表示弃票时为 true。"
            "private_fact_claim 仅在将尚未公开的身份、夜间信息或未来结果声称为已知事实时为 true。"
            "猜测某人身份本身不是私密事实。只输出规定 JSON，不推断未明确表达的语义。")


def perception_response_format() -> dict[str, Any]:
    # Duplicate rejection stays in parse_perceived_semantics; strict provider
    # JSON-Schema subsets do not consistently support uniqueItems.
    seat_array = {"type": "array", "items": {"type": "string", "enum": list(PLAYER_IDS)}}
    return {"type": "json_schema", "json_schema": {
        "name": LANGUAGE_VERSION, "strict": True,
        "schema": {"type": "object", "additionalProperties": False,
                   "required": ["speaker", "phase", "commitment_targets", "rejected_targets",
                                "vote_intent_targets", "information_requests", "abstain_intent",
                                "private_fact_claim"],
                   "properties": {
                       "speaker": {"type": "string", "enum": list(PLAYER_IDS)},
                       "phase": {"type": "string", "enum": ["speech", "speech_pk"]},
                       "commitment_targets": seat_array,
                       "rejected_targets": seat_array,
                       "vote_intent_targets": seat_array,
                       "information_requests": {"type": "array", "items": {
                           "type": "object", "additionalProperties": False,
                           "required": ["target_j", "addressee_j", "request_type"],
                           "properties": {"target_j": {"type": "string", "enum": list(PLAYER_IDS)},
                                          "addressee_j": {"type": "string", "enum": list(PLAYER_IDS)},
                                          "request_type": {"type": "string",
                                                           "enum": [REQUEST_TYPE, OTHER_REQUEST]}}}},
                       "abstain_intent": {"type": "boolean"},
                       "private_fact_claim": {"type": "boolean"}}}}}


class Phase2LanguageActorV1:
    """Existing chat backend transport with a separate Phase-2 actor prompt."""

    def __init__(self, backend, model_name: str, *, temperature: float = 0.0,
                 max_tokens: int = 512):
        if not callable(getattr(backend, "chat_with_metadata", None)) or not model_name:
            raise Phase2LanguageError("actor requires chat backend and model")
        self.backend, self.model_name = backend, model_name
        self.temperature, self.max_tokens = temperature, max_tokens

    def realize(self, plan: Phase2SemanticPlanV1, context: PublicLanguageContextV1,
                *, failure_reason: str | None = None) -> str:
        prompt = build_realization_prompt(plan, context, failure_reason)
        content, metadata = self.backend.chat_with_metadata(
            messages=[{"role": "user", "content": prompt}], model=self.model_name,
            temperature=self.temperature, max_tokens=self.max_tokens)
        if not isinstance(content, str) or not content.strip() or (
                isinstance(metadata, dict) and metadata.get("finish_reason") == "length"):
            raise Phase2LanguageError("REALIZATION_INVALID_OR_TRUNCATED")
        return content.strip()


class Phase2SemanticPerceiverV1:
    """No method accepts a requested plan; model sees only public context/text."""

    def __init__(self, backend, model_name: str, *, max_tokens: int = 384):
        if (not callable(getattr(backend, "chat_with_metadata", None))
                or getattr(backend, "supports_json_schema", False) is not True
                or not model_name):
            raise Phase2LanguageError("perceiver requires JSON-schema chat backend and model")
        self.backend, self.model_name, self.max_tokens = backend, model_name, max_tokens

    def perceive(self, text: str, context: PublicLanguageContextV1) -> Phase2SpeechSemanticV1:
        prompt = build_perception_prompt(text, context)
        content, metadata = self.backend.chat_with_metadata(
            messages=[{"role": "user", "content": prompt}], model=self.model_name,
            temperature=0.0, max_tokens=self.max_tokens,
            response_format=perception_response_format())
        if isinstance(metadata, dict) and metadata.get("finish_reason") == "length":
            raise Phase2LanguageError("PERCEPTION_TRUNCATED")
        return parse_perceived_semantics(content)
