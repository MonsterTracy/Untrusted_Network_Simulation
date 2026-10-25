"""Small shared validation helpers for the new artifact family."""

import json
import math
import re

from werewolf.artifact_io import canonical_json_bytes, sha256_bytes


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return sha256_bytes(canonical_json_bytes(value))


def require_digest(value):
    require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
            "invalid SHA-256 identity")
    return value


def strict_json(raw):
    def pairs(values):
        result = {}
        for key, value in values:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    def constant(value):
        raise ValueError("nonfinite JSON constant")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def observer_seat(observer):
    require(type(observer) is int and 0 <= observer < 7, "observer must be zero-based seat")
    return observer


def valid_distribution(row):
    observer = observer_seat(row["observer"])
    p, logs = row["probability"], row["non_self_log_probability"]
    require(type(p) is list and len(p) == 7 and type(logs) is list and len(logs) == 6,
            "seven-seat probability and six log probabilities required")
    require(all(type(v) in (float, int) and math.isfinite(v) for v in p + logs)
            and p[observer] == 0 and all(v >= 0 for v in p)
            and math.isclose(math.fsum(p), 1, rel_tol=0, abs_tol=1e-6), "invalid non-self simplex")
    require(all(math.isclose(p[j], math.exp(logp), rel_tol=1e-6, abs_tol=1e-7)
                for j, logp in zip((j for j in range(7) if j != observer), logs, strict=True)),
            "probability/log-probability mismatch")


def identity(row):
    return row["game_id"], row["boundary_id"], row["prefix_digest"], observer_seat(row["observer"])
