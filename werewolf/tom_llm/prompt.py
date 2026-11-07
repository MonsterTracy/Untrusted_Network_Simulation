"""Lossless semantic-token rendering; audit identities never enter messages."""

from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment, StrictUndefined

from werewolf.artifact_io import canonical_json_bytes, sha256_bytes
from werewolf.canonical_collection.public_history import PLAYER_IDS, PUBLIC_PHASES
from werewolf.canonical_collection.speech import V1_ACTIONS
from werewolf.structured_history import StructuredTokenPlan, STRUCTURED_TOKEN_TYPES, STRUCTURED_TOKEN_PLANNER_VERSION
from werewolf.tom_llm.common import digest, require, require_digest, observer_seat

ASSETS = Path(__file__).with_name("assets")
PROMPT_VERSION = "classic7_llm_public_plan_prompt_v1"
SET_PROMPT_VERSION = "classic7_llm_suspicion_set_prompt_v2"
TEMPLATE_VERSION = "qwen3_single_turn_nonthinking_v1"
COLUMNS = ("token_type", "source", "action", "target", "day", "phase")


def asset_bytes(name):
    return (ASSETS / name).read_bytes()


def validate_plan(plan):
    require(type(plan) is StructuredTokenPlan, "only StructuredTokenPlan is accepted")
    require(plan.planner_version == STRUCTURED_TOKEN_PLANNER_VERSION, "planner version mismatch")
    require_digest(plan.prefix_digest)
    require_digest(plan.plan_digest)
    record = plan.to_record()
    record.pop("plan_digest")
    require(digest(record) == plan.plan_digest, "plan digest mismatch")
    require(type(plan.token_count) is int and plan.token_count == len(plan.tokens)
            and 1 <= plan.token_count <= 1024, "complete plan exceeds frozen capacity or is empty")
    require(plan.tokens[0].token_type == "phase_change" and plan.tokens[-1].token_type == "turn_start",
            "complete PRE must start with public phase and end with turn_start")
    for index, token in enumerate(plan.tokens):
        require(token.token_index == index and token.token_type in STRUCTURED_TOKEN_TYPES
                and token.source in (None, *PLAYER_IDS) and token.target in (None, *PLAYER_IDS)
                and token.action in (None, *V1_ACTIONS) and type(token.day) is int and token.day >= 0
                and token.phase in PUBLIC_PHASES, "invalid public semantic token")
    state = None
    for token in plan.tokens:
        if token.token_type == "phase_change":
            state = (token.day, token.phase)
        require((token.day, token.phase) == state, "public temporal state must propagate causally")
        if token.token_type in {"turn_start", "public_speech"}:
            require(token.source in PLAYER_IDS and token.action is None and token.target is None,
                    "speech marker reference mismatch")
        elif token.token_type == "speech_action":
            require(token.source in PLAYER_IDS and token.action in V1_ACTIONS
                    and (token.target is None if token.action in {"abstain_intent", "no_commitment"}
                         else token.target in PLAYER_IDS), "speech action references mismatch")
        elif token.token_type == "vote":
            require(token.source in PLAYER_IDS and token.action is None, "vote references mismatch")
        elif token.token_type in {"exiled_player", "dead_player"}:
            require(token.source is None and token.action is None and token.target in PLAYER_IDS,
                    "result player references mismatch")
        else:
            require(token.source is None and token.action is None and token.target is None,
                    "public result/phase marker references mismatch")
    require(plan.tokens[-1].source in PLAYER_IDS, "terminal turn_start requires ordinary seat reference")
    return plan


def public_table(plan):
    validate_plan(plan)
    return "type|source|action|target|day|phase\n" + "\n".join(
        "|".join("-" if getattr(token, field) is None else str(getattr(token, field)) for field in COLUMNS)
        for token in plan.tokens)


def schema(observer):
    observer_seat(observer)
    seats = [p for j, p in enumerate(PLAYER_IDS) if j != observer]
    return {"type": "object", "additionalProperties": False, "required": ["probabilities"],
            "properties": {"probabilities": {"type": "object", "additionalProperties": False,
                "required": seats, "properties": {p: {"type": "number", "minimum": 0, "maximum": 1} for p in seats}}}}


@dataclass(frozen=True)
class Prompt:
    observer: int
    messages: tuple
    rendered: str
    table: str
    schema: dict
    prompt_digest: str


def render_prompt(plan, observer):
    observer_seat(observer)
    table = public_table(plan)
    output_schema = schema(observer)
    user = (table + "\n\n观察者：" + PLAYER_IDS[observer]
            + "\n只输出以下 schema 对应的概率 JSON，六值和为一：\n"
            + canonical_json_bytes(output_schema).decode("utf-8"))
    messages = ({"role": "system", "content": asset_bytes("system.txt").decode("utf-8")},
                {"role": "user", "content": user})
    template = Environment(undefined=StrictUndefined, autoescape=False).from_string(
        asset_bytes("chat_template.jinja").decode("utf-8"))
    rendered = template.render(messages=messages)
    return Prompt(observer, messages, rendered, table, output_schema,
                  sha256_bytes(rendered.encode("utf-8")))


def set_schema(observer):
    """集合的结构约束；重复、身份等错误仍由应用层分别登记。"""
    observer_seat(observer)
    seats = [p for j, p in enumerate(PLAYER_IDS) if j != observer]
    return {"type": "object", "additionalProperties": False, "required": ["suspected_players"],
            "properties": {"suspected_players": {"type": "array", "minItems": 0, "maxItems": 6,
                "uniqueItems": True, "items": {"type": "string", "enum": seats}}}}


def set_system_text():
    """保留原规则和事件词典，只替换预测目标及输出要求。"""
    _, marker, rules = asset_bytes("system.txt").decode("utf-8").partition("规则背景：")
    require(marker, "frozen rules section missing")
    return ("你负责根据截至当前 PRE 的公开信息，预测指定观察者此时会报告怀疑哪些其他玩家。\n"
            "预测观察者的报告集合；不要输出你自己怀疑谁、真实狼人身份、投票或发言策略。\n"
            "观察者可能拥有你看不到的私人认知；不要编造私人信息。\n"
            "候选为观察者之外的六席，死亡目标仍可入选。允许空集合，不要求固定大小或两名狼人。\n"
            "只返回含 suspected_players 数组的严格 JSON，不返回解释、概率或额外字段。\n"
            + marker + rules)


def render_set_prompt(plan, observer):
    """集合协议继续使用同一公开 PRE 表和模型原生消息角色。"""
    table = public_table(plan)
    output_schema = set_schema(observer)
    user = (table + "\n\n观察者：" + PLAYER_IDS[observer]
            + "\n预测该观察者当前会报告的怀疑对象集合。只输出以下 schema 对应的 JSON：\n"
            + canonical_json_bytes(output_schema).decode("utf-8"))
    messages = ({"role": "system", "content": set_system_text()}, {"role": "user", "content": user})
    template = Environment(undefined=StrictUndefined, autoescape=False).from_string(
        asset_bytes("chat_template.jinja").decode("utf-8"))
    rendered = template.render(messages=messages)
    return Prompt(observer, messages, rendered, table, output_schema,
                  sha256_bytes(rendered.encode("utf-8")))
