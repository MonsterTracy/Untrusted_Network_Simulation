"""从合法狼人的发言 PRE 生成可直接提交的候选文本。"""

from dataclasses import dataclass
import json

from werewolf.agents.llm_agent import GameplayActionValidationError, validate_gameplay_public_speech
from werewolf.canonical_collection.pre import AuthoritativePREPrefix
from werewolf.observer_dtrs.context import contexts_for_pre
from werewolf.observer_dtrs.probabilities import ProbabilityMatrix
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
text 只写要公开说出的正文，避免暴露真实队伍身份、模型数值或内部字段。仅输出包含 topic 和 candidates 的 JSON，每份候选包含 function、text 和 addressee。"""


@dataclass(frozen=True)
class SpeechCandidate:
    function: str
    text: str
    addressee: str | None


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


def candidate_response_properties(replyable_observers):
    functions = SPEECH_FUNCTIONS if replyable_observers else SPEECH_FUNCTIONS[:2]
    return {
        "topic": {"type": "string"},
        "candidates": {
            "type": "array",
            "minItems": 1,
            "maxItems": MAX_CANDIDATES,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["function", "text", "addressee"],
                "properties": {
                    "function": {"type": "string", "enum": list(functions)},
                    "text": {"type": "string"},
                    "addressee": {"type": ["string", "null"], "enum": [*replyable_observers, None]},
                },
            },
        },
    }


def parse_speech_candidates(content, *, pre, replyable_observers):
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

    candidates, texts = [], set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"function", "text", "addressee"}:
            raise GameplayActionValidationError("每份候选须包含 function、text 和 addressee")
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
        candidates.append(SpeechCandidate(function, speech, addressee))

    if len(candidates) - len({candidate.function for candidate in candidates}) > 1:
        raise GameplayActionValidationError("基础功能覆盖之外至多补充一份候选")
    return SpeechCandidates(pre=pre, topic=topic, candidates=tuple(candidates))
