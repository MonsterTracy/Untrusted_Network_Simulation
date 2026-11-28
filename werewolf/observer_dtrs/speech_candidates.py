"""从合法狼人的发言 PRE 生成可直接提交的候选文本。"""

from dataclasses import dataclass
import json

from werewolf.agents.llm_agent import GameplayActionValidationError, validate_gameplay_public_speech
from werewolf.canonical_collection.pre import AuthoritativePREPrefix
from werewolf.observer_dtrs.context import contexts_for_pre
from werewolf.observer_dtrs.probabilities import ProbabilityMatrix
from werewolf.observer_dtrs.report_state_decision import EVENTS, NEG, POS
from werewolf.strategy.counterfactual_tom import derive_public_phase_speaker_order


SPEECH_FUNCTIONS = ("INFORM", "ADVOCATE", "INQUIRE")
MAX_CANDIDATES = 4
SPEECH_CANDIDATE_PROMPT = """你是当前发言的狼人。请根据下方合法 PRE，为狼队准备本回合可以直接公开说出的不同发言。
先写一个 topic，指出当前最需要处理的问题，不在议题中预设判断或行动结论。
INFORM 提供、整理或澄清相关信息。区分已公开的事实与玩家曾作出的主张。
ADVOCATE 提出或维护某种公开判断、立场或行动倡议。
INQUIRE 围绕具体未知提出定向问题，允许回答支持不同方向。只询问 replyable_observers 中的一人，在 text 中直接称呼该席位，并用 addressee 指明同一席位。列表为空时不准备 INQUIRE。其他功能的 addressee 为 null。
wolf_suspicion_mass 的行对应 observer_ids，列对应 wolf_ids，values 为模型给出的原始相对嫌疑质量。它们不是报告命中概率，接近 0.5 不表示认知最不确定。可以参考这些信息确定议题和探询对象，不能把预测当成已知的私人报告或公开事实。
每种适用功能优先写一份完整发言。所有功能合计最多补充一份，补充可属于任一功能。最终有序 candidates 共 1 至 4 份，可以省去不适用的功能。候选须改变所提供的信息、公开判断、行动建议或请求的信息，避免仅调整措辞。
text 只写要公开说出的正文，避免暴露真实队伍身份、模型数值或内部字段。每份候选另填私有的 required_premises 和 public_evidence_refs。
required_premises 列出给出这份发言时采用的全部确定性预测前提。每项用 observer_id、event、direction 表达。speaker_hit 表示该观察者的成功原始嫌疑报告包含当前说话的狼人，team_hit 表示包含至少一名当前存活狼人，POS 表示命中，NEG 表示未命中。保留两种可能、开放提问和独立公开依据无需声明确定方向。依赖狼队命中的预测不能用来指定是哪一个狼人。不要把预测前提说成已观测到的私人事实。
public_evidence_refs 仅列出作为依据使用的当前公开事件 ID。候选中至少有一份能在不采用任何确定性预测前提时成立，其 required_premises 为空。最终只输出包含 topic 和 candidates 的 JSON，候选须包括上述五个字段。"""


@dataclass(frozen=True)
class SpeechCandidate:
    function: str
    text: str
    addressee: str | None


@dataclass(frozen=True)
class PredictionPremise:
    observer_id: str
    event: str
    direction: str


@dataclass(frozen=True)
class DeclaredSpeechCandidate(SpeechCandidate):
    required_premises: tuple[PredictionPremise, ...]
    public_evidence_refs: tuple[str, ...]


@dataclass(frozen=True)
class SpeechCandidates:
    pre: AuthoritativePREPrefix
    topic: str
    candidates: tuple[SpeechCandidate, ...]


def build_candidate_context(pre, known_wolf_ids, q):
    """调用者从当前行动者的合法视角提供队伍和同一 PRE 的冻结 Q。"""
    observers = contexts_for_pre(pre, known_wolf_ids)
    if not observers:
        raise ValueError("候选生成要求有存活非狼观察者的合法狼人发言 PRE")
    if not isinstance(q, ProbabilityMatrix) or (q.game_id, q.boundary_id, q.prefix_digest) != (
        pre.game_id, pre.boundary_id, pre.prefix_digest
    ):
        raise ValueError("候选生成要求与当前 PRE 匹配的冻结 ProbabilityMatrix")

    order = derive_public_phase_speaker_order(pre)
    remaining = order[order.index(pre.current_speaker) + 1:]
    observer_ids = tuple(context.observer_id for context in observers)
    wolf_ids = observers[0].alive_wolf_ids
    columns = tuple(q.seat_ids.index(wolf) for wolf in wolf_ids)
    return {
        "speaker": pre.current_speaker,
        "day": pre.public_temporal_state.day,
        "phase": pre.public_temporal_state.phase.value,
        "alive_players": pre.alive_observer_ids,
        "known_wolf_team": known_wolf_ids,
        "public_events": pre.public_event_history.to_records(),
        "remaining_speakers": remaining,
        "replyable_observers": tuple(seat for seat in remaining if seat in observer_ids),
        "wolf_suspicion_mass": {
            "observer_ids": observer_ids,
            "wolf_ids": wolf_ids,
            "values": [[q.values[q.seat_ids.index(observer)][col] for col in columns]
                       for observer in observer_ids],
        },
    }


def basis_response_properties(observer_ids, event_ids):
    premise = {"observer_id": {"type": "string", "enum": list(observer_ids)},
               "event": {"type": "string", "enum": list(EVENTS)},
               "direction": {"type": "string", "enum": [POS, NEG]}}
    return {
        "required_premises": {"type": "array", "maxItems": 2 * len(observer_ids),
            "items": {"type": "object", "additionalProperties": False, "required": list(premise), "properties": premise}},
        "public_evidence_refs": {"type": "array", "maxItems": len(event_ids),
                                 "items": {"type": "string", "enum": list(event_ids)}},
    }


def parse_candidate_basis(row, observer_ids, event_ids, wolf_count):
    """校验显式前提的身份、合取逻辑和公开引用。"""
    required, refs = row["required_premises"], row["public_evidence_refs"]
    if not isinstance(required, list) or not isinstance(refs, list):
        raise GameplayActionValidationError("候选依据字段须为列表")
    premises, signed = [], {}
    for item in required:
        if (not isinstance(item, dict) or set(item) != {"observer_id", "event", "direction"}
                or item["observer_id"] not in observer_ids or item["event"] not in EVENTS
                or item["direction"] not in (POS, NEG)):
            raise GameplayActionValidationError("预测前提需要合格观察者、合法事件及确定方向")
        premise = PredictionPremise(**item)
        if (premise.observer_id, premise.event) in signed:
            raise GameplayActionValidationError("同一观察者事件的预测前提重复或矛盾")
        signed[premise.observer_id, premise.event] = premise.direction
        premises.append(premise)
    for seat in observer_ids:
        speaker, team = (signed.get((seat, event)) for event in EVENTS)
        if ((speaker, team) == (POS, NEG) or
                (wolf_count == 1 and speaker is not None and team is not None and speaker != team)):
            raise GameplayActionValidationError("预测前提违反自身与狼队事件的包含关系")
    if any(not isinstance(ref, str) or ref not in event_ids for ref in refs) or len(set(refs)) != len(refs):
        raise GameplayActionValidationError("公开证据引用必须唯一并指向当前 PRE 公开事件")
    return tuple(premises), tuple(refs)


def candidate_response_properties(context):
    replyable_observers = context["replyable_observers"]
    functions = SPEECH_FUNCTIONS if replyable_observers else SPEECH_FUNCTIONS[:2]
    properties = {"function": {"type": "string", "enum": list(functions)}, "text": {"type": "string"},
                  "addressee": {"type": ["string", "null"], "enum": [*replyable_observers, None]},
                  **basis_response_properties(context["wolf_suspicion_mass"]["observer_ids"],
                                              [event["event_id"] for event in context["public_events"]])}
    return {
        "topic": {"type": "string"},
        "candidates": {
            "type": "array",
            "minItems": 1,
            "maxItems": MAX_CANDIDATES,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": list(properties),
                "properties": properties,
            },
        },
    }


def parse_speech_candidates(content, *, pre, context, archived=False):
    """新生成要求声明完备，旧归档按原样读入且没有声明语义。"""
    try:
        payload = json.loads(content)
    except (TypeError, json.JSONDecodeError) as error:
        raise GameplayActionValidationError("发言候选响应必须为 JSON") from error
    if not isinstance(payload, dict) or set(payload) != {"topic", "candidates"}:
        raise GameplayActionValidationError("发言候选响应须包含 topic 和 candidates")
    topic, rows = payload["topic"], payload["candidates"]
    if not isinstance(topic, str) or not topic.strip():
        raise GameplayActionValidationError("主要议题不能为空")
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_CANDIDATES:
        raise GameplayActionValidationError("须生成 1 至 4 份发言候选")

    base_fields = {"function", "text", "addressee"}
    old_archive = archived and all(isinstance(row, dict) and set(row) == base_fields for row in rows)
    fields = base_fields if old_archive else base_fields | {"required_premises", "public_evidence_refs"}
    replyable_observers = context["replyable_observers"]
    observers = context["wolf_suspicion_mass"]["observer_ids"]
    wolves = context["wolf_suspicion_mass"]["wolf_ids"]
    events = {event["event_id"] for event in context["public_events"]}
    candidates, texts = [], set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != fields:
            raise GameplayActionValidationError("候选须使用完整且统一的发言与依据字段")
        function, speech, addressee = row["function"], row["text"], row["addressee"]
        if function not in SPEECH_FUNCTIONS:
            raise GameplayActionValidationError("未知的发言功能")
        required_players = ()
        if function == "INQUIRE":
            if addressee not in replyable_observers:
                raise GameplayActionValidationError("INQUIRE 须指定下一次投票前可以回复的非狼观察者")
            required_players = (int(addressee.removeprefix("player")),)
        elif addressee is not None:
            raise GameplayActionValidationError("仅 INQUIRE 可指定受问者")

        validate_gameplay_public_speech(
            speech,
            player_id=int(pre.current_speaker.removeprefix("player")),
            phase=pre.public_temporal_state.phase.value,
            required_player_ids=required_players,
        )
        # 忽略首尾空白判重，保留通过校验的原文。
        stripped = speech.strip()
        if stripped in texts:
            raise GameplayActionValidationError("发言候选文本重复")
        texts.add(stripped)
        if old_archive:
            candidates.append(SpeechCandidate(function, speech, addressee))
        else:
            basis = parse_candidate_basis(row, observers, events, len(wolves))
            candidates.append(DeclaredSpeechCandidate(function, speech, addressee, *basis))

    if len(candidates) - len({candidate.function for candidate in candidates}) > 1:
        raise GameplayActionValidationError("基础功能覆盖之外至多补充一份候选")
    if not old_archive and all(candidate.required_premises for candidate in candidates):
        raise GameplayActionValidationError("每批须有一份声明无确定性预测前提的候选")
    return SpeechCandidates(pre=pre, topic=topic, candidates=tuple(candidates))
