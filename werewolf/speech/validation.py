"""Runtime-only player validation and lawful observer-private knowledge closure."""

import re
from itertools import combinations
from typing import Any

from werewolf.canonical_collection.public_history import PLAYER_IDS

PLAYER_ORDER = {p: i for i, p in enumerate(PLAYER_IDS)}
LABEL_PROMPT_VERSION = "classic7_pre_speech_player_suspicion_prompt_v6"


def canonicalize_player_set(values: Any, *, field_name: str) -> list[str]:
    """Validate and order one duplicate-free set of canonical player IDs."""

    if not isinstance(values, list):
        raise TypeError(f"{field_name} must be a list")
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            raise TypeError(f"{field_name} entries must be strings")
        if value not in PLAYER_IDS:
            raise ValueError(
                f"{field_name} entries must be canonical player1...player7 names"
            )
        if value in seen:
            raise ValueError(f"duplicate player in {field_name}: {value}")
        seen.add(value)
        normalized.append(value)
    return sorted(normalized, key=PLAYER_ORDER.__getitem__)


def normalize_player(value: Any) -> str:
    """Convert a player reference into canonical form such as ``player3``.

    Accepted examples:
        3
        "3"
        "player3"
        "Player_3"
        "player 3"
    """

    if isinstance(value, bool):
        raise TypeError("boolean values are not valid player references")

    if isinstance(value, int):
        player_id = value
    elif isinstance(value, str):
        compact = re.sub(r"[\s_-]+", "", value.strip().lower())
        match = re.fullmatch(r"(?:player)?([1-7])", compact)

        if match is None:
            raise ValueError(f"invalid player reference: {value!r}")

        player_id = int(match.group(1))
    else:
        raise TypeError(
            "player reference must be an integer or string; "
            f"got {type(value).__name__}"
        )

    if not 1 <= player_id <= 7:
        raise ValueError(
            f"player ID must be in [1, {7}], got {player_id}"
        )

    return f"player{player_id}"


def validate_player_suspicion(
    suspected_werewolves: Any,
    known_werewolves: Any,
    known_non_werewolves: Any,
    *,
    observer_id: Any,
) -> list[str]:
    """Validate one soft suspicion support against legal hard knowledge."""

    observer = normalize_player(observer_id)
    suspected = canonicalize_player_set(
        suspected_werewolves,
        field_name="suspected_werewolves",
    )
    known_wolves = canonicalize_player_set(
        known_werewolves,
        field_name="known_werewolves",
    )
    known_non_wolves = canonicalize_player_set(
        known_non_werewolves,
        field_name="known_non_werewolves",
    )
    if set(known_wolves) & set(known_non_wolves):
        raise ValueError("hard knowledge sets must be disjoint")
    if observer in suspected:
        raise ValueError("suspected_werewolves cannot contain the observer")

    missing = sorted(
        (set(known_wolves) - {observer}) - set(suspected),
        key=PLAYER_ORDER.__getitem__,
    )
    if missing:
        raise ValueError(
            "suspected_werewolves must include all known Werewolves other "
            f"than the observer; missing={missing}"
        )

    forbidden_non_wolves = sorted(
        set(suspected) & set(known_non_wolves),
        key=PLAYER_ORDER.__getitem__,
    )
    if forbidden_non_wolves:
        raise ValueError(
            "suspected_werewolves cannot contain known non-Werewolves; "
            f"forbidden={forbidden_non_wolves}"
        )
    return suspected


def close_hard_knowledge(
    known_werewolves: Any,
    known_non_werewolves: Any,
) -> tuple[list[str], list[str]]:
    """Apply only the deterministic consequence of exactly two Werewolves."""

    known_wolves = canonicalize_player_set(
        known_werewolves,
        field_name="known_werewolves",
    )
    known_non_wolves = canonicalize_player_set(
        known_non_werewolves,
        field_name="known_non_werewolves",
    )
    if set(known_wolves) & set(known_non_wolves):
        raise ValueError("hard knowledge sets must be disjoint")
    hard_support = [
        pair
        for pair in combinations(PLAYER_IDS, 2)
        if set(known_wolves).issubset(pair)
        and set(pair).isdisjoint(known_non_wolves)
    ]
    if not hard_support:
        raise ValueError("hard knowledge has no legal two-Werewolf pair")
    closed_wolves = [
        player
        for player in PLAYER_IDS
        if all(player in pair for pair in hard_support)
    ]
    closed_non_wolves = [
        player
        for player in PLAYER_IDS
        if all(player not in pair for pair in hard_support)
    ]
    return closed_wolves, closed_non_wolves
