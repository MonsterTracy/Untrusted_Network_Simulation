"""Current public PRE state and the acting wolf's legally known team only."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re

from werewolf.canonical_collection.pre import (
    AuthoritativePREPrefix, validate_authoritative_pre_prefix,
)
from werewolf.canonical_collection.public_history import PLAYER_IDS


CONTEXT_VERSION = "observer_dtrs_context_v1"
SPEECH_PHASES = ("discussion", "pk_discussion")


def _seats(values: tuple[str, ...]) -> None:
    if (not isinstance(values, tuple) or any(p not in PLAYER_IDS for p in values)
            or values != tuple(p for p in PLAYER_IDS if p in values)):
        raise ValueError("player IDs must be unique canonical seat-ordered tuples")


@dataclass(frozen=True)
class ObserverContext:
    """Legal context, with no reports, supervision or publication reader handle."""

    game_id: str
    boundary_id: str
    prefix_digest: str
    step_index: int  # Copy the canonical field; never a runtime loop counter.
    acting_wolf: str
    observer_id: str
    day: int
    phase: str
    alive_player_ids: tuple[str, ...]
    known_wolf_ids: tuple[str, ...]

    def __post_init__(self):
        _seats(self.alive_player_ids)
        _seats(self.known_wolf_ids)
        if len(self.known_wolf_ids) != 2:
            raise ValueError("Classic7 legal wolf team must contain two seats")
        if (self.acting_wolf not in self.alive_wolf_ids
                or self.observer_id not in self.alive_player_ids
                or self.observer_id in self.known_wolf_ids):
            raise ValueError("context requires a living wolf and living non-wolf observer")
        if self.phase not in SPEECH_PHASES:
            raise ValueError("context requires a canonical speech phase")
        if any(type(v) is not int or v < 0 for v in (self.day, self.step_index)):
            raise ValueError("day and canonical step_index must be nonnegative integers")
        if any(not isinstance(v, str) or not v.strip() for v in (self.game_id, self.boundary_id)):
            raise ValueError("game and boundary identities must be nonempty strings")
        if not isinstance(self.prefix_digest, str) or re.fullmatch(r"[0-9a-f]{64}", self.prefix_digest) is None:
            raise ValueError("prefix_digest must be a lowercase SHA-256")

    @property
    def alive_wolf_ids(self) -> tuple[str, ...]:
        return tuple(p for p in self.known_wolf_ids if p in self.alive_player_ids)

    @property
    def wolf_count(self) -> int:
        return len(self.alive_wolf_ids)

    def to_record(self) -> dict:
        return {"schema_version": CONTEXT_VERSION, **asdict(self),
                "alive_wolf_ids": self.alive_wolf_ids, "wolf_count": self.wolf_count}


def contexts_for_pre(prefix: AuthoritativePREPrefix,
                     known_wolf_ids: tuple[str, ...]) -> tuple[ObserverContext, ...]:
    """Enumerate A\\W for a wolf speech PRE, without Phase2 candidate filters.

    Online callers must obtain known_wolf_ids from the acting wolf's legal
    observation. Only the restricted offline builder derives them from sidecar.
    """
    validate_authoritative_pre_prefix(prefix)
    _seats(known_wolf_ids)
    if len(known_wolf_ids) != 2:
        raise ValueError("Classic7 legal wolf team must contain two seats")
    state = prefix.public_temporal_state
    if state.phase.value not in SPEECH_PHASES or prefix.current_speaker not in known_wolf_ids:
        return ()
    return tuple(ObserverContext(
        game_id=prefix.game_id, boundary_id=prefix.boundary_id,
        prefix_digest=prefix.prefix_digest, step_index=prefix.step_index,
        acting_wolf=prefix.current_speaker, observer_id=observer,
        day=state.day, phase=state.phase.value,
        alive_player_ids=prefix.alive_observer_ids, known_wolf_ids=known_wolf_ids,
    ) for observer in prefix.alive_observer_ids if observer not in known_wolf_ids)
