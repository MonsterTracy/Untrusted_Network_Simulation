"""Prespecified, development-only comparison of cross-fitted Phase-2 mapper predictions.

Bootstrap replicates resample held-out games within mapper folds. They never refit a
mapper, estimate a deployment calibrator, or use candidate rows as sampling units.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from werewolf.phase2_mapper import (
    FOLDS, MODEL_SPECS, PROBABILITY_EPSILON, MapperPrediction,
)


DEFAULT_REPLICATES = 100_000
DEFAULT_SEED = 20260929
CI_LEVEL = 0.95
MODEL_ORDER = tuple(spec.name for spec in MODEL_SPECS)
ADJACENT_PAIRS = tuple(zip(MODEL_ORDER[1:], MODEL_ORDER[:-1], strict=True))
RESCUE_PAIRS = ((MODEL_ORDER[2], MODEL_ORDER[0]),
                (MODEL_ORDER[3], MODEL_ORDER[0]))
ALL_PAIRS = ADJACENT_PAIRS + RESCUE_PAIRS


class MapperSelectionError(ValueError):
    """Predictions or a comparison violate the frozen selection protocol."""


@dataclass(frozen=True)
class ModelPointMetrics:
    candidate_count: int
    game_count: int
    pooled_log_loss: float
    pooled_brier: float
    game_macro_log_loss: float
    game_macro_brier: float


@dataclass(frozen=True)
class PairComparison:
    complex_model: str
    simple_model: str
    role: str
    delta_pooled_log_loss: float
    ci_lower: float
    ci_upper: float
    delta_pooled_brier: float
    delta_game_macro_log_loss: float
    delta_game_macro_brier: float
    development_supported: bool
    ci_pooled_brier: tuple[float, float] | None = None
    ci_game_macro_log_loss: tuple[float, float] | None = None
    ci_game_macro_brier: tuple[float, float] | None = None


@dataclass(frozen=True)
class BootstrapComparison:
    points: dict[str, ModelPointMetrics]
    pairs: tuple[PairComparison, ...]
    replicates: int
    seed: int
    ci_level: float

    def pair(self, complex_model: str, simple_model: str) -> PairComparison:
        for pair in self.pairs:
            if (pair.complex_model, pair.simple_model) == (complex_model, simple_model):
                return pair
        raise MapperSelectionError("comparison pair absent from bootstrap result")


@dataclass(frozen=True)
class SelectionDecision:
    primary_model: str
    stopped_at: tuple[str, str] | None
    development_supported_alternatives: tuple[str, ...]
    comparison: BootstrapComparison


@dataclass(frozen=True)
class _GamePanel:
    game_ids: tuple[str, ...]
    folds: np.ndarray
    fold_game_indices: tuple[np.ndarray, ...]
    candidate_counts: np.ndarray
    log_loss_sums: np.ndarray  # shape: model x game
    brier_sums: np.ndarray


def _prepare_panel(predictions: tuple[MapperPrediction, ...]) -> _GamePanel:
    if not predictions:
        raise MapperSelectionError("empty cross-fitted prediction population")
    by_model: dict[str, dict[tuple[str, str, str, str, str], MapperPrediction]] = {
        name: {} for name in MODEL_ORDER}
    for row in predictions:
        if row.model_spec not in by_model:
            raise MapperSelectionError("unknown model in comparison predictions")
        model_rows = by_model[row.model_spec]
        if row.identity in model_rows:
            raise MapperSelectionError("duplicate candidate identity within model")
        if (type(row.theta_ac) is not bool or row.mapper_fold not in FOLDS
                or row.game_oof_fold != row.mapper_fold
                or not np.isfinite(row.raw_probability)
                or not 0 <= row.raw_probability <= 1
                or not np.isfinite(row.raw_score)):
            raise MapperSelectionError("invalid cross-fitted prediction row")
        model_rows[row.identity] = row
    identities = set(by_model[MODEL_ORDER[0]])
    if not identities or any(set(rows) != identities for rows in by_model.values()):
        raise MapperSelectionError("candidate identity mismatch across models")
    ordered_ids = sorted(identities)
    reference = by_model[MODEL_ORDER[0]]
    game_fold: dict[str, int] = {}
    for identity in ordered_ids:
        base = reference[identity]
        previous = game_fold.setdefault(base.game_id, base.mapper_fold)
        if previous != base.mapper_fold:
            raise MapperSelectionError("one game spans mapper folds")
        for name in MODEL_ORDER[1:]:
            other = by_model[name][identity]
            if (other.theta_ac != base.theta_ac
                    or other.mapper_fold != base.mapper_fold
                    or other.prefix_digest != base.prefix_digest
                    or other.qwen3_prediction_digest != base.qwen3_prediction_digest
                    or other.evaluation_seal_digest != base.evaluation_seal_digest):
                raise MapperSelectionError("candidate label, fold, or provenance mismatch")
    games = tuple(sorted(game_fold))
    folds = np.asarray([game_fold[game] for game in games], dtype=np.int64)
    if set(folds) != set(FOLDS):
        raise MapperSelectionError("all five mapper folds require held-out games")
    game_index = {game: index for index, game in enumerate(games)}
    row_games = np.asarray([game_index[identity[0]] for identity in ordered_ids],
                           dtype=np.int64)
    counts = np.bincount(row_games, minlength=len(games)).astype(np.int64)
    y = np.asarray([int(reference[identity].theta_ac) for identity in ordered_ids])
    log_sums, brier_sums = [], []
    for name in MODEL_ORDER:
        p = np.asarray([by_model[name][identity].raw_probability
                        for identity in ordered_ids], dtype=np.float64)
        safe_p = np.clip(p, PROBABILITY_EPSILON, 1 - PROBABILITY_EPSILON)
        log_losses = -np.where(y == 1, np.log(safe_p), np.log1p(-safe_p))
        log_sums.append(np.bincount(row_games, weights=log_losses,
                                    minlength=len(games)))
        brier_sums.append(np.bincount(row_games, weights=(p - y) ** 2,
                                      minlength=len(games)))
    fold_indices = tuple(np.flatnonzero(folds == fold) for fold in FOLDS)
    return _GamePanel(games, folds, fold_indices, counts,
                      np.asarray(log_sums), np.asarray(brier_sums))


def _draw_game_multiplicities(panel: _GamePanel,
                              rng: np.random.Generator) -> np.ndarray:
    """One cluster draw: each fold contributes its original number of games."""
    weights = np.zeros(len(panel.game_ids), dtype=np.int64)
    for indices in panel.fold_game_indices:
        drawn = rng.choice(indices, size=len(indices), replace=True)
        np.add.at(weights, drawn, 1)
    return weights


def _weighted_metrics(panel: _GamePanel, multiplicities: np.ndarray) -> np.ndarray:
    """Return model x (pooled LL, pooled Brier, macro LL, macro Brier)."""
    weights = np.asarray(multiplicities, dtype=np.float64)
    if (weights.shape != (len(panel.game_ids),) or not np.isfinite(weights).all()
            or np.any(weights < 0) or weights.sum() <= 0):
        raise MapperSelectionError("invalid game multiplicities")
    row_total = float(np.dot(weights, panel.candidate_counts))
    game_total = float(weights.sum())
    return np.column_stack((
        panel.log_loss_sums @ weights / row_total,
        panel.brier_sums @ weights / row_total,
        (panel.log_loss_sums / panel.candidate_counts) @ weights / game_total,
        (panel.brier_sums / panel.candidate_counts) @ weights / game_total,
    ))


def percentile_interval(samples: np.ndarray, *, level: float = CI_LEVEL) -> tuple[float, float]:
    samples = np.asarray(samples, dtype=np.float64)
    if (samples.ndim != 1 or len(samples) == 0 or not np.isfinite(samples).all()
            or not 0 < level < 1):
        raise MapperSelectionError("invalid percentile interval input")
    tail = (1 - level) / 2
    lower, upper = np.quantile(samples, (tail, 1 - tail), method="linear")
    return float(lower), float(upper)


def passes_promotion(ci_upper: float, delta_pooled_brier: float,
                     delta_game_macro_log_loss: float) -> bool:
    """A complex model advances only when all three prespecified gates pass."""
    values = (ci_upper, delta_pooled_brier, delta_game_macro_log_loss)
    if not all(np.isfinite(value) for value in values):
        raise MapperSelectionError("non-finite promotion input")
    return ci_upper < 0 and delta_pooled_brier <= 0 and delta_game_macro_log_loss <= 0


def bootstrap_compare(predictions: tuple[MapperPrediction, ...], *,
                      replicates: int = DEFAULT_REPLICATES,
                      seed: int = DEFAULT_SEED) -> BootstrapComparison:
    """Compare fixed CV predictions with one shared game draw for every model."""
    if (type(replicates) is not int or replicates < 1
            or type(seed) is not int or seed < 0):
        raise MapperSelectionError("invalid bootstrap replicates or seed")
    panel = _prepare_panel(predictions)
    points_array = _weighted_metrics(panel, np.ones(len(panel.game_ids)))
    points = {name: ModelPointMetrics(int(panel.candidate_counts.sum()),
        len(panel.game_ids), *map(float, points_array[index]))
        for index, name in enumerate(MODEL_ORDER)}
    pair_indices = tuple((MODEL_ORDER.index(complex_model), MODEL_ORDER.index(simple_model))
                         for complex_model, simple_model in ALL_PAIRS)
    deltas = np.empty((replicates, len(ALL_PAIRS), 4), dtype=np.float64)
    rng = np.random.default_rng(seed)
    for replicate in range(replicates):
        metrics = _weighted_metrics(panel, _draw_game_multiplicities(panel, rng))
        deltas[replicate] = [metrics[b] - metrics[a] for b, a in pair_indices]
    pairs = []
    for index, ((complex_model, simple_model), (b, a)) in enumerate(
            zip(ALL_PAIRS, pair_indices, strict=True)):
        lower, upper = percentile_interval(deltas[:, index, 0])
        difference = points_array[b] - points_array[a]
        pairs.append(PairComparison(complex_model, simple_model,
            "adjacent" if index < len(ADJACENT_PAIRS) else "rescue",
            float(difference[0]), lower, upper, float(difference[1]),
            float(difference[2]), float(difference[3]),
            passes_promotion(upper, float(difference[1]), float(difference[2])),
            percentile_interval(deltas[:, index, 1]),
            percentile_interval(deltas[:, index, 2]),
            percentile_interval(deltas[:, index, 3])))
    return BootstrapComparison(points, tuple(pairs), replicates, seed, CI_LEVEL)


def select_from_comparison(comparison: BootstrapComparison) -> SelectionDecision:
    """Apply the fixed adjacent path; rescue comparisons cannot change primary."""
    if (set(comparison.points) != set(MODEL_ORDER)
            or {(p.complex_model, p.simple_model) for p in comparison.pairs}
            != set(ALL_PAIRS)
            or len(comparison.pairs) != len(ALL_PAIRS)):
        raise MapperSelectionError("incomplete prespecified comparison")
    for pair in comparison.pairs:
        expected_role = ("adjacent" if (pair.complex_model, pair.simple_model)
                         in ADJACENT_PAIRS else "rescue")
        if (pair.role != expected_role
                or pair.development_supported != passes_promotion(
                    pair.ci_upper, pair.delta_pooled_brier,
                    pair.delta_game_macro_log_loss)):
            raise MapperSelectionError("inconsistent prespecified promotion result")
    selected = MODEL_ORDER[0]
    stopped_at = None
    for complex_model, simple_model in ADJACENT_PAIRS:
        pair = comparison.pair(complex_model, simple_model)
        if pair.development_supported:
            selected = complex_model
        else:
            stopped_at = (complex_model, simple_model)
            break
    alternatives = ()
    if stopped_at is not None:
        alternatives = tuple(complex_model for complex_model, simple_model in RESCUE_PAIRS
            if complex_model != selected
            and comparison.pair(complex_model, simple_model).development_supported)
    return SelectionDecision(selected, stopped_at, alternatives, comparison)


def compare_and_select(predictions: tuple[MapperPrediction, ...], *,
                       replicates: int = DEFAULT_REPLICATES,
                       seed: int = DEFAULT_SEED) -> SelectionDecision:
    return select_from_comparison(bootstrap_compare(predictions,
                                   replicates=replicates, seed=seed))
