"""Identity-bound probability mass in absolute Classic7 seat order.

These value objects validate numerical/identity contracts, not model provenance.
Offline provenance is checked by oof; an online caller must supply frozen-model
inference from its legal public PRE. Neither path accepts targets or log-Q.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
import re

from werewolf.canonical_collection.public_history import PLAYER_IDS


def _identity(game_id, boundary_id, prefix_digest):
    if any(not isinstance(v, str) or not v.strip() for v in (game_id, boundary_id)):
        raise ValueError("Q requires game and PRE identities")
    if not isinstance(prefix_digest, str) or re.fullmatch(r"[0-9a-f]{64}", prefix_digest) is None:
        raise ValueError("Q requires a lowercase SHA-256 prefix_digest")


@dataclass(frozen=True)
class ProbabilityRow:
    game_id: str
    boundary_id: str
    prefix_digest: str
    observer_id: str
    values: tuple[float, ...]
    seat_ids: tuple[str, ...] = PLAYER_IDS

    def __post_init__(self):
        _identity(self.game_id, self.boundary_id, self.prefix_digest)
        if self.seat_ids != PLAYER_IDS or self.observer_id not in PLAYER_IDS:
            raise ValueError("Q axes must be absolute player1...player7 seats")
        if len(self.values) != 7 or any(isinstance(v, bool) or not isinstance(v, Real)
                                      for v in self.values):
            raise ValueError("Q must contain seven numeric probability masses")
        values = tuple(float(v) for v in self.values)
        if (any(not math.isfinite(v) or not 0 <= v <= 1 for v in values)
                or values[PLAYER_IDS.index(self.observer_id)] != 0
                or not math.isclose(math.fsum(values), 1, rel_tol=0, abs_tol=1e-6)):
            raise ValueError("Q must be finite, nonnegative, zero-self and sum to one")
        # Freeze caller-owned lists; never clip, exponentiate or renormalize.
        object.__setattr__(self, "values", values)


@dataclass(frozen=True)
class ProbabilityMatrix:
    """Future online input: all seven observer rows, including dead seats."""

    game_id: str
    boundary_id: str
    prefix_digest: str
    values: tuple[tuple[float, ...], ...]
    seat_ids: tuple[str, ...] = PLAYER_IDS

    def __post_init__(self):
        _identity(self.game_id, self.boundary_id, self.prefix_digest)
        if self.seat_ids != PLAYER_IDS or len(self.values) != 7:
            raise ValueError("Q matrix must have seven absolute observer/target seats")
        rows = tuple(ProbabilityRow(self.game_id, self.boundary_id, self.prefix_digest,
                                   observer, row, self.seat_ids).values
                     for observer, row in zip(PLAYER_IDS, self.values, strict=True))
        object.__setattr__(self, "values", rows)

    def row(self, observer_id: str) -> ProbabilityRow:
        if observer_id not in PLAYER_IDS:
            raise ValueError("unknown observer seat")
        return ProbabilityRow(self.game_id, self.boundary_id, self.prefix_digest,
                              observer_id, self.values[PLAYER_IDS.index(observer_id)])
