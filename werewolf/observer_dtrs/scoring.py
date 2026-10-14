"""Game-equal proper scores and paired, inner-only parameter selection."""
from collections import defaultdict
import math


def score_predictions(rows, predictions, *, epsilon=1e-12):
    """Explicit offline evaluation; no fitting or probability output clipping."""
    rows = tuple(rows)
    if (not rows or set(predictions) != {r.sample_id for r in rows}
            or len({r.sample_id for r in rows}) != len(rows) or not 0 < epsilon < .5):
        raise ValueError("evaluation requires exact unique prediction coverage")
    grouped = defaultdict(list)
    for r in rows:
        p = predictions[r.sample_id]
        if isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("invalid predicted probability")
        q = min(1 - epsilon, max(epsilon, p))
        grouped[r.game_id].append({"brier": (p - r.y) ** 2,
            "log_loss": -(r.y * math.log(q) + (1 - r.y) * math.log1p(-q)),
            "bias": p - r.y, "clipped": p != q,
            "wrong_endpoint": (p == 0 and r.y == 1) or (p == 1 and r.y == 0)})
    by_game = {g: {"rows": len(records), **{metric: math.fsum(r[metric] for r in records) / len(records)
               for metric in ("brier", "log_loss", "bias")}} for g, records in sorted(grouped.items())}
    return {"by_game": by_game,
        "game_equal": {metric: math.fsum(v[metric] for v in by_game.values()) / len(by_game)
                       for metric in ("brier", "log_loss", "bias")},
        "record_equal": {metric: math.fsum(r[metric] for values in grouped.values() for r in values) / len(rows)
                         for metric in ("brier", "log_loss", "bias")},
        "clipped_rows": sum(r["clipped"] for values in grouped.values() for r in values),
        "wrong_endpoint_rows": sum(r["wrong_endpoint"] for values in grouped.values() for r in values)}


def paired_difference(first, second):
    """Second minus first on exactly the same held-out games and row counts."""
    a, b = first["by_game"], second["by_game"]
    if set(a) != set(b) or any(a[g]["rows"] != b[g]["rows"] for g in a):
        raise ValueError("model comparison requires identical evaluation games/population")
    return {metric: math.fsum(b[g][metric] - a[g][metric] for g in a) / len(a)
            for metric in ("brier", "log_loss", "bias")}


def select_parameters(candidates, *, protocol, model_id, weighting="game_equal", game_sizes=None):
    """Precomputed INNER validation game Brier only; formal runner is D2b-2.

    candidates maps (requested_bins, lambda) to {game_id: mean_game_Brier}.
    Every frozen candidate and identical game coverage are required.
    """
    p = protocol.record
    if model_id not in p["primary_models"]:
        raise ValueError("unknown primary model")
    grid = {(b, l) for b in ((1,) if model_id == "OBS-EQ0" else p["histogram"]["requested_bin_counts"])
            for l in p["smoothing"]["lambda_grid_game_equivalent_mass"]}
    if set(candidates) != grid:
        raise ValueError("incomplete or expanded frozen candidate grid")
    games = set(next(iter(candidates.values())))
    if not games or any(set(v) != games or any(not math.isfinite(x) or not 0 <= x <= 1 for x in v.values())
                        for v in candidates.values()):
        raise ValueError("unpaired inner validation games or invalid Brier")
    if weighting == "game_equal":
        weights = {g: 1 for g in games}
    elif (weighting == "record_equal" and game_sizes is not None and set(game_sizes) == games
          and all(type(n) is int and n > 0 for n in game_sizes.values())):
        weights = game_sizes
    else:
        raise ValueError("record-equal selection requires exact validation game sizes")
    losses = {key: math.fsum(weights[g] * v[g] for g in sorted(games)) / sum(weights.values())
              for key, v in candidates.items()}
    best = min(losses.values())
    eligible = [key for key, loss in losses.items() if loss <= best + p["selection"]["tie_absolute_tolerance"]]
    return min(eligible, key=lambda key: (key[0], -key[1]))
