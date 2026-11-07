"""A single fixed parser/normalization/fallback policy, independent of labels."""

from dataclasses import asdict, dataclass
import math
from fractions import Fraction

from werewolf.artifact_io import canonical_json_bytes
from werewolf.canonical_collection.public_history import PLAYER_IDS
from werewolf.tom_llm.common import observer_seat, strict_json, require, valid_distribution, digest, require_digest

EPSILON = 1e-6
SUM_TOLERANCE = .001
OUTPUT_VERSION = "classic7_llm_non_self_output_v1"
SET_OUTPUT_VERSION = "classic7_llm_suspicion_set_output_v2"
PROJECTION_VERSION = "classic7_uniform_reference_strict_projection_v1"
SET_FAILURES = {"length", "invalid_finish", "empty_or_refusal", "invalid_json", "invalid_schema",
                "self_suspicion", "invalid_player", "duplicate_player"}


@dataclass(frozen=True)
class Prediction:
    observer: int
    probability: list
    non_self_log_probability: list
    status: str
    failure_type: str | None
    raw_probability: list | None
    raw_sum: float | None
    normalized_for_rounding: bool
    raw_zero_count: int

    def to_record(self):
        return asdict(self)


def _valid_sum(values):
    difference = abs(math.fsum(values) - 1)
    return difference <= SUM_TOLERANCE or math.isclose(difference, SUM_TOLERANCE, rel_tol=0, abs_tol=2*math.ulp(1.))


def _prediction(observer, raw=None, *, failure=None):
    values = [1 / 6] * 6 if raw is None else raw
    total = math.fsum(values)
    normalized = [value / total for value in values]
    predicted = normalized if failure else [(1 - EPSILON) * p + EPSILON / 6 for p in normalized]
    p = [0.] * 7
    for seat, value in zip((j for j in range(7) if j != observer), predicted, strict=True):
        p[seat] = value
    result = Prediction(observer, p, [math.log(v) for v in predicted],
        "fallback_uniform" if failure else "valid", failure, raw,
        None if raw is None else total, raw is not None and total != 1.,
        0 if raw is None else sum(v == 0 for v in raw))
    valid_distribution(result.to_record())
    return result


def parse_output(content, observer, *, finish_reason="stop"):
    observer_seat(observer)
    if finish_reason != "stop":
        return _prediction(observer, failure="length" if finish_reason == "length" else "invalid_finish")
    if content is None or content == "":
        return _prediction(observer, failure="empty_or_refusal")
    if type(content) is not str:
        return _prediction(observer, failure="invalid_json")
    try:
        value = strict_json(content)
    except (ValueError, TypeError):
        return _prediction(observer, failure="invalid_json")
    seats = [p for j, p in enumerate(PLAYER_IDS) if j != observer]
    if (type(value) is not dict or set(value) != {"probabilities"}
            or type(value["probabilities"]) is not dict or set(value["probabilities"]) != set(seats)):
        return _prediction(observer, failure="invalid_seats_or_fields")
    raw = [value["probabilities"][p] for p in seats]
    if not all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in raw):
        return _prediction(observer, failure="invalid_probability")
    if not _valid_sum(raw):
        return _prediction(observer, failure="invalid_sum")
    return _prediction(observer, [float(v) for v in raw])


def validate_prediction(record):
    if type(record) is dict and record.get("schema_version") == SET_OUTPUT_VERSION:
        return validate_set_prediction(record)
    require(type(record) is dict and set(record) == set(Prediction.__dataclass_fields__), "prediction schema mismatch")
    observer_seat(record["observer"])
    valid_distribution(record)
    failure = record["failure_type"]
    if record["status"] == "fallback_uniform":
        require(failure in {"length", "invalid_finish", "empty_or_refusal", "invalid_json",
                "invalid_seats_or_fields", "invalid_probability", "invalid_sum"}, "invalid fallback reason")
        expected = _prediction(record["observer"], failure=failure).to_record()
    else:
        require(record["status"] == "valid" and failure is None and type(record["raw_probability"]) is list,
                "invalid prediction status")
        raw = record["raw_probability"]
        require(len(raw) == 6 and all(type(v) in (float, int) and math.isfinite(v) and 0 <= v <= 1 for v in raw)
                and _valid_sum(raw), "invalid raw distribution")
        expected = _prediction(record["observer"], raw).to_record()
    require(record == expected, "prediction normalization/fallback policy mismatch")
    return record


@dataclass(frozen=True)
class SetPrediction:
    distribution: Prediction
    suspected_players: list | None
    raw_response_digest: str

    def to_record(self):
        return {**self.distribution.to_record(), "schema_version": SET_OUTPUT_VERSION,
                "suspected_players": self.suspected_players,
                "raw_response_digest": self.raw_response_digest}


def _set_prediction(observer, players, response_digest, *, failure=None):
    """非法集合保持 None；仅补充分布评价使用既有失败均匀策略。"""
    if failure:
        return SetPrediction(_prediction(observer, failure=failure), None, response_digest)
    raw = [1 / 6] * 6
    if players:
        raw = [1 / len(players) if p in players else 0.
               for j, p in enumerate(PLAYER_IDS) if j != observer]
    return SetPrediction(_prediction(observer, raw), list(players), response_digest)


def parse_set_output(content, observer, *, finish_reason="stop"):
    """严格集合解析，不修补非法成员，不把失败转换成合法空报告。"""
    observer_seat(observer)
    response_digest = digest({"content": content, "finish_reason": finish_reason})

    def invalid(reason):
        return _set_prediction(observer, None, response_digest, failure=reason)

    if finish_reason != "stop":
        return invalid("length" if finish_reason == "length" else "invalid_finish")
    if content is None or content == "":
        return invalid("empty_or_refusal")
    try:
        value = strict_json(content) if type(content) is str else None
    except (ValueError, TypeError):
        return invalid("invalid_json")
    if type(content) is not str:
        return invalid("invalid_json")
    if (type(value) is not dict or set(value) != {"suspected_players"}
            or type(value["suspected_players"]) is not list
            or len(value["suspected_players"]) > 6
            or not all(type(p) is str for p in value["suspected_players"])):
        return invalid("invalid_schema")
    players = value["suspected_players"]
    if PLAYER_IDS[observer] in players:
        return invalid("self_suspicion")
    if not set(players) <= set(PLAYER_IDS):
        return invalid("invalid_player")
    if len(set(players)) != len(players):
        return invalid("duplicate_player")
    return _set_prediction(observer, sorted(players, key=PLAYER_IDS.index), response_digest)


def validate_set_prediction(record):
    fields = set(Prediction.__dataclass_fields__) | {"schema_version", "suspected_players", "raw_response_digest"}
    require(set(record) == fields and record["schema_version"] == SET_OUTPUT_VERSION,
            "set prediction schema mismatch")
    observer = observer_seat(record["observer"])
    require_digest(record["raw_response_digest"])
    if record["status"] == "valid":
        players = record["suspected_players"]
        require(type(players) is list and all(type(p) is str and p in PLAYER_IDS for p in players)
                and len(set(players)) == len(players) and PLAYER_IDS[observer] not in players
                and players == sorted(players, key=PLAYER_IDS.index), "invalid canonical prediction set")
        expected = _set_prediction(observer, players, record["raw_response_digest"])
    else:
        require(record["status"] == "fallback_uniform" and record["suspected_players"] is None
                and record["failure_type"] in SET_FAILURES, "invalid set failure state")
        expected = _set_prediction(observer, None, record["raw_response_digest"], failure=record["failure_type"])
    require(record == expected.to_record(), "set conversion/failure policy drift")
    return record


def parse_task_output(content, task, *, finish_reason="stop"):
    """解析器由已验证任务的协议选择，旧响应保持原解析语义。"""
    from werewolf.tom_llm.task import task_version, SET_VERSION
    parser = parse_set_output if task_version(task) == SET_VERSION else parse_output
    return parser(content, task["observer"], finish_reason=finish_reason)


def probability_to_set(probability, observer, rule, *, formal_game_ids):
    """仅消费预测概率；开发来源未核准前，本接口只接受显式 Mock 规则。"""
    observer_seat(observer)
    fields = {"schema_version", "scope", "threshold", "comparison", "development_game_ids", "development_source_digest"}
    require(type(rule) is dict and set(rule) == fields
            and rule["schema_version"] == "classic7_probability_to_set_development_v2"
            and rule["scope"] == "ENGINEERING_ONLY" and rule["comparison"] == ">=",
            "probability-to-set rule requires an explicit versioned engineering/development identity")
    threshold = rule["threshold"]
    require(type(threshold) in (int, float) and math.isfinite(threshold) and 0 <= threshold <= 1,
            "invalid probability-to-set threshold")
    development = rule["development_game_ids"]
    require(type(development) is list and development and all(type(g) is str and g for g in development)
            and len(set(development)) == len(development) and formal_game_ids
            and not set(development) & set(formal_game_ids),
            "development games must be disjoint from the complete formal Selection")
    require_digest(rule["development_source_digest"])
    require(type(probability) is list and len(probability) == 7
            and all(type(p) in (int, float) and math.isfinite(p) and 0 <= p <= 1 for p in probability)
            and probability[observer] == 0 and math.isclose(math.fsum(probability), 1, abs_tol=1e-6),
            "probability-to-set requires a non-self probability prediction")
    return [player for j, player in enumerate(PLAYER_IDS) if j != observer and probability[j] >= threshold]


def fixed_projection_rule():
    """已批准的无校准读出；规则身份包含数值解释，不继承旧 Mock 阈值。"""
    return {'schema_version':PROJECTION_VERSION, 'input_field':'sealed probability',
            'comparison':'6*p_j > sum_non_self(p)', 'number_semantics':'canonical_json_decimal_rational',
            'self':'excluded', 'equality':'excluded', 'ordering':list(PLAYER_IDS),
            'calibration':'none', 'scope':'uniform_reference_deterministic_projection_not_f1_optimal'}


def uniform_reference_projection(probability, observer):
    """只读取概率与观察者；经既有 canonical 验证的数值按十进制有理数比较。"""
    observer_seat(observer)
    require(type(probability) is list and len(probability)==7
            and all(type(p) in (int,float) and math.isfinite(p) and 0<=p<=1 for p in probability)
            and probability[observer]==0
            and math.isclose(math.fsum(probability),1,rel_tol=0,abs_tol=1e-6),
            'fixed projection requires a valid non-self probability')
    values = [Fraction(canonical_json_bytes(p).decode('ascii')) for p in probability]
    total = sum(values)
    return [player for j,player in enumerate(PLAYER_IDS) if j!=observer and 6*values[j]>total]
