"""Capacity derived solely from the versioned complete public PRE contract."""

from werewolf.canonical_collection.pre import AUTHORITATIVE_PRE_PREFIX_SCHEMA_VERSION
from werewolf.canonical_collection.public_history import PUBLIC_EVENT_SCHEMA_VERSION
from werewolf.structured_history import STRUCTURED_TOKEN_PLANNER_VERSION, plan_structured_history

DERIVATION_VERSION = "classic7_validator_day_capacity_floor_sequence_over_four_v1"


def derived_capacity(max_seq_len):
    if type(max_seq_len) is not int or max_seq_len < 4:
        raise ValueError("final capacity must represent at least one legal PRE")
    versions = (PUBLIC_EVENT_SCHEMA_VERSION, AUTHORITATIVE_PRE_PREFIX_SCHEMA_VERSION,
                STRUCTURED_TOKEN_PLANNER_VERSION)
    if versions != ("classic7_public_event_history_v1", "classic7_authoritative_pre_prefix_v1",
                    "classic7_structured_token_planner_v1"):
        raise ValueError("day-capacity derivation requires reviewed public semantics")
    return {"derivation_version": DERIVATION_VERSION, "max_seq_len": max_seq_len,
            "derived_day_capacity": max_seq_len // 4, "formula": "floor(max_seq_len/4)",
            "public_event_version": versions[0], "pre_version": versions[1],
            "structured_token_planner_version": versions[2], "prefix_policy": "full_prefix_no_truncation"}


def validate_final_pre(prefix, capacity):
    if capacity != derived_capacity(capacity["max_seq_len"]):
        raise ValueError("derived capacity identity mismatch")
    plan = plan_structured_history(prefix)
    if plan.token_count > capacity["max_seq_len"]:
        raise ValueError("complete PRE prefix exceeds frozen sequence capacity")
    if any(t.day > capacity["derived_day_capacity"] for t in plan.tokens):
        raise ValueError("PRE exceeds frozen day capacity")
    return plan
