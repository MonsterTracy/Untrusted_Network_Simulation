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
