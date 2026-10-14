"""Legal Observer-DTRS condition attributes; no supervision or offline imports."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from werewolf.canonical_collection.public_history import PLAYER_IDS
from werewolf.observer_dtrs.context import ObserverContext
from werewolf.observer_dtrs.probabilities import ProbabilityMatrix, ProbabilityRow


FEATURE_VERSION = "observer_dtrs_conditions_v1"


@dataclass(frozen=True)
class ObserverFeatures:
    s: float
    m: float
    r: float
    dW: float | None
    phase: str
    nW: int

    def to_record(self) -> dict:
        return {"schema_version": FEATURE_VERSION, **asdict(self)}


def observer_features(context: ObserverContext,
                      prediction: ProbabilityRow | ProbabilityMatrix) -> ObserverFeatures:
    """Use the same seat lookup for historical rows and future online matrices."""
    if not isinstance(context, ObserverContext):
        raise TypeError("features require only an ObserverContext, never an offline sample")
    if isinstance(prediction, ProbabilityMatrix):
        prediction = prediction.row(context.observer_id)
    if not isinstance(prediction, ProbabilityRow):
        raise TypeError("features require a validated probability row or matrix")
    if ((prediction.game_id, prediction.boundary_id, prediction.prefix_digest, prediction.observer_id)
            != (context.game_id, context.boundary_id, context.prefix_digest, context.observer_id)):
        raise ValueError("Q game/PRE/prefix/observer identity mismatch")

    def mass(seat):
        return prediction.values[PLAYER_IDS.index(seat)]

    s = mass(context.acting_wolf)
    m = math.fsum(mass(k) for k in context.alive_wolf_ids if k != context.acting_wolf)
    dW = (abs(mass(context.alive_wolf_ids[0]) - mass(context.alive_wolf_ids[1]))
          if context.wolf_count == 2 else None)
    return ObserverFeatures(s=s, m=m, r=s + m, dW=dW,
                            phase=context.phase, nW=context.wolf_count)
