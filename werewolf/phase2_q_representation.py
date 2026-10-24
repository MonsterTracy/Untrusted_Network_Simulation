"""Competition-aware representation of frozen non-self suspicion Q."""
import math
import numpy as np
from werewolf.canonical_collection.public_history import PLAYER_IDS

REPRESENTATION_VERSION = "classic7_competition_q_representation_v1"
REPRESENTATIONS = ("continuous", "hard_top1")


def condition_id(temporal_condition, q_representation):
    conditions = {("explicit_day_phase", "continuous"): "A0",
                  ("implicit", "continuous"): "A1",
                  ("explicit_day_phase", "hard_top1"): "A2"}
    try:
        return conditions[(temporal_condition, q_representation)]
    except (KeyError, TypeError) as error:
        raise ValueError("outside frozen Experiment A conditions") from error


def transform_q(q, competition, *, representation="continuous"):
    """Retain absolute 7x7 coordinates; harden once per PRE, never per candidate."""
    if representation not in REPRESENTATIONS:
        raise ValueError("unknown Q representation")
    matrix = np.asarray(q)
    if matrix.shape != (7, 7) or matrix.dtype.kind not in "fiu":
        raise ValueError("Q must be a numeric 7x7 matrix")
    matrix = matrix.astype(np.float64, copy=True)
    if (not np.isfinite(matrix).all() or np.any(matrix < 0)
            or np.any(np.diag(matrix) != 0)
            or any(abs(math.fsum(row) - 1) > 1e-6 for row in matrix)):
        raise ValueError("Q violates non-self simplex")
    competition = tuple(competition)
    if (len(competition) < 2 or len(set(competition)) != len(competition)
            or not set(competition) <= set(PLAYER_IDS)):
        raise ValueError("invalid current competition")
    if representation == "continuous":
        return matrix
    result = np.zeros((7, 7), dtype=np.float64)
    for observer in range(7):
        targets = tuple(j for j, seat in enumerate(PLAYER_IDS)
                        if seat in competition and j != observer)
        winner = max(targets, key=lambda j: matrix[observer, j])
        result[observer, winner] = 1.0
    return result
