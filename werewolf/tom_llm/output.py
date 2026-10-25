"""A single fixed parser/normalization/fallback policy, independent of labels."""

from dataclasses import asdict, dataclass
import math

from werewolf.canonical_collection.public_history import PLAYER_IDS
from werewolf.tom_llm.common import observer_seat, strict_json, require, valid_distribution

EPSILON = 1e-6
SUM_TOLERANCE = .001
OUTPUT_VERSION = "classic7_llm_non_self_output_v1"


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
