"""Development-only Z -> P(Theta_AC=1 | Z) mapper protocol.

No formal study, deployment calibrator, or gameplay policy is run on import.
The input boundary is the verified offline candidate population.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from werewolf.artifact_io import verify_artifact
from werewolf import phase2_offline as offline
from werewolf.phase2_q_representation import REPRESENTATION_VERSION, condition_id


FEATURE_CONTRACT_VERSION = "phase2_compact_r2_mapper_v1"
CONTEXT_NUMERIC = ("audience_size", "competition_size", "remaining_speakers_before_vote")
EVIDENCE = ("mu", "delta", "sigma", "h")
FOLDS = tuple(range(5))
PROBABILITY_EPSILON = 1e-15  # Numerical metric guard, never a label/prevalence adjustment.


class MapperProtocolError(ValueError):
    """A mapper input or provenance violates the frozen development protocol."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: object) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


@dataclass(frozen=True)
class ModelSpec:
    name: str
    evidence_features: tuple[str, ...]
    additive: bool
    l2_inverse_strength: float = 100.0  # Fixed C; likelihood remains unweighted.
    max_iter: int = 1000
    tolerance: float = 1e-8
    knot_quantiles: tuple[float, ...] = ()

    def record(self) -> dict:
        return asdict(self)

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.record())


MODEL_SPECS = (
    ModelSpec("M0_context_logistic", (), False),
    ModelSpec("M1_r1_logistic", ("mu",), False),
    ModelSpec("M2_r2_linear_logistic", EVIDENCE, False),
    ModelSpec("M3_r2_additive_logistic", EVIDENCE, True,
              knot_quantiles=(1 / 3, 2 / 3)),
)


@dataclass(frozen=True)
class OOFProvenance:
    evaluation_seal_digest: str
    evaluation_contract_digest: str
    prediction_digest_by_fold: dict[int, str]
    paired_experiment_digest: str | None = None
    temporal_condition: str = "explicit_day_phase"
    q_representation: str = "continuous"
    source_revision: str | None = None

    @property
    def experiment_a(self):
        if self.paired_experiment_digest is None:
            return None
        return {"condition": condition_id(self.temporal_condition, self.q_representation),
                "paired_experiment_digest": self.paired_experiment_digest,
                "architecture": "qwen3", "temporal_condition": self.temporal_condition,
                "q_representation": self.q_representation,
                "representation_version": REPRESENTATION_VERSION,
                "mapper_spec": MODEL_SPECS[3].name, "source_revision": self.source_revision,
                "publication_digest": offline.PUBLICATION_DIGEST,
                "fold_manifest_digest": offline.FOLD_MANIFEST_DIGEST}

    @property
    def identity_digest(self):
        return _digest(self.record())

    def __post_init__(self) -> None:
        if ((self.paired_experiment_digest is None and self.evaluation_seal_digest != offline.OOF_SEAL_DIGEST)
                or set(self.prediction_digest_by_fold) != set(FOLDS)
                or any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
                       for value in (self.evaluation_seal_digest, self.evaluation_contract_digest,
                                     *self.prediction_digest_by_fold.values()))):
            raise MapperProtocolError("invalid sealed Qwen3 OOF provenance")
        if self.paired_experiment_digest is None:
            if (self.temporal_condition != "explicit_day_phase" or self.q_representation != "continuous"
                    or self.source_revision is not None):
                raise MapperProtocolError("historical OOF identity cannot become Experiment A")
        else:
            condition_id(self.temporal_condition, self.q_representation)
            if (re.fullmatch(r"[0-9a-f]{64}", self.paired_experiment_digest) is None
                    or not isinstance(self.source_revision, str)
                    or re.fullmatch(r"[0-9a-f]{40}", self.source_revision) is None):
                raise MapperProtocolError("invalid paired experiment/source identity")

    @classmethod
    def from_evaluation_root(cls, root: Path | str, *, paired_experiment_digest=None,
                             temporal_condition="explicit_day_phase", q_representation="continuous",
                             expected_seal_digest=None) -> OOFProvenance:
        root = Path(root)
        paired = paired_experiment_digest is not None
        if paired:
            from werewolf.tom import backbone_evaluation as evaluation
            condition_id(temporal_condition, q_representation)
            if not isinstance(expected_seal_digest, str) or re.fullmatch(r"[0-9a-f]{64}", expected_seal_digest) is None:
                raise MapperProtocolError("paired OOF requires an explicit seal trust anchor")
        elif temporal_condition != "explicit_day_phase" or q_representation != "continuous" or expected_seal_digest not in (None, offline.OOF_SEAL_DIGEST):
            raise MapperProtocolError("historical OOF representation/condition mismatch")
        contract = verify_artifact(root / "contract",
            expected_artifact_type="backbone_oof_evaluation_contract",
            expected_schema_version=evaluation.PAIRED_VERSION if paired else "classic7_backbone_oof_evaluation_v1")
        seal = verify_artifact(root / "seal",
            expected_artifact_type="backbone_oof_evaluation_report",
            expected_schema_version=evaluation.PAIRED_REPORT_VERSION if paired else "classic7_backbone_oof_report_v1")
        if (seal.manifest_digest != (expected_seal_digest if paired else offline.OOF_SEAL_DIGEST)
                or seal.manifest["evaluation_digest"] != contract.manifest_digest):
            raise MapperProtocolError("unexpected OOF evaluation seal")
        if paired and (seal.manifest["paired_experiment_digest"] != paired_experiment_digest
                       or contract.manifest["preparation"]["manifest_digest"] != paired_experiment_digest
                       or seal.manifest["preparation"] != contract.manifest["preparation"]
                       or seal.manifest["training_source"] != contract.manifest["training_source"]
                       or seal.manifest["training_source"]["source_revision"] != contract.manifest["training_revision"]
                       or seal.manifest["training_seal_digest"] != contract.manifest["training_seal_digest"]):
            raise MapperProtocolError("paired OOF parent/source mismatch")
        descriptors = [row for row in seal.manifest["predictions"]
                       if row["architecture"] == "qwen3" and (not paired or row["temporal_condition"] == temporal_condition)]
        predictions = {row["fold"]: row["prediction_digest"] for row in descriptors}
        if len(descriptors) != 5 or set(predictions) != set(FOLDS):
            raise MapperProtocolError("Qwen3 OOF fold manifest incomplete or duplicated")
        for fold in FOLDS:
            path = (evaluation.prediction_path(root, "qwen3", fold, temporal_condition, paired=True)
                    if paired else root / "folds" / "qwen3" / str(fold))
            artifact = verify_artifact(path,
                expected_artifact_type="backbone_oof_fold_predictions",
                expected_schema_version=evaluation.PAIRED_PREDICTION_VERSION if paired else "classic7_backbone_oof_predictions_v1")
            if (artifact.manifest_digest != predictions[fold]
                    or artifact.manifest["evaluation_digest"] != contract.manifest_digest
                    or artifact.manifest["architecture"] != "qwen3"
                    or artifact.manifest["fold"] != fold
                    or (paired and (artifact.manifest["temporal_condition"] != temporal_condition
                                    or artifact.manifest["training_seal_digest"] != seal.manifest["training_seal_digest"]))):
                raise MapperProtocolError("Qwen3 OOF fold prediction lineage mismatch")
        return cls(seal.manifest_digest, seal.manifest["evaluation_digest"], predictions,
                   paired_experiment_digest, temporal_condition, q_representation,
                   contract.manifest["training_revision"] if paired else None)

    def record(self) -> dict:
        return {"evaluation_seal_digest": self.evaluation_seal_digest,
                "evaluation_contract_digest": self.evaluation_contract_digest,
                "prediction_digest_by_fold": {str(f): self.prediction_digest_by_fold[f]
                                              for f in FOLDS},
                **({"experiment_a": self.experiment_a} if self.experiment_a is not None else {})}


@dataclass(frozen=True)
class FoldPreprocessor:
    """Fold-fitted numeric scaling and, for M3 only, additive hinge knots."""

    spec: ModelSpec
    means: dict[str, float]
    scales: dict[str, float]
    knots: dict[str, tuple[float, ...]]
    training_game_digest: str

    @classmethod
    def fit(cls, rows: tuple[offline.CandidateRow, ...], spec: ModelSpec) -> FoldPreprocessor:
        if not rows or spec not in MODEL_SPECS:
            raise MapperProtocolError("empty training rows or unknown fixed ModelSpec")
        names = CONTEXT_NUMERIC + spec.evidence_features
        matrix = _numeric_matrix(rows, names)
        means_array = matrix.mean(axis=0)
        scales_array = matrix.std(axis=0, ddof=0)
        scales_array[scales_array == 0] = 1.0
        means = {name: float(means_array[i]) for i, name in enumerate(names)}
        scales = {name: float(scales_array[i]) for i, name in enumerate(names)}
        knots = {}
        if spec.additive:
            for name in spec.evidence_features:
                column = (matrix[:, names.index(name)] - means[name]) / scales[name]
                knots[name] = tuple(float(k) for k in np.quantile(
                    column, spec.knot_quantiles, method="linear"))
        return cls(spec, means, scales, knots,
                   _digest(sorted({row.game_id for row in rows})))

    def columns(self) -> tuple[str, ...]:
        names = ["phase_pk", *CONTEXT_NUMERIC]
        for name in self.spec.evidence_features:
            names.append(name)
            if self.spec.additive:
                names.extend(f"{name}_hinge_{i}" for i in range(len(self.spec.knot_quantiles)))
        return tuple(names)

    def transform(self, rows: tuple[offline.CandidateRow, ...]) -> np.ndarray:
        if not rows:
            raise MapperProtocolError("cannot transform empty mapper rows")
        names = CONTEXT_NUMERIC + self.spec.evidence_features
        raw = _numeric_matrix(rows, names)
        scaled = {name: (raw[:, i] - self.means[name]) / self.scales[name]
                  for i, name in enumerate(names)}
        columns = [np.array([_phase_code(row.z.phase) for row in rows], dtype=np.float64)]
        columns.extend(scaled[name] for name in CONTEXT_NUMERIC)
        for name in self.spec.evidence_features:
            columns.append(scaled[name])
            if self.spec.additive:
                columns.extend(np.maximum(0.0, scaled[name] - knot)
                               for knot in self.knots[name])
        result = np.column_stack(columns).astype(np.float64, copy=False)
        if result.shape[1] != len(self.columns()) or not np.isfinite(result).all():
            raise MapperProtocolError("invalid fold-fitted design matrix")
        return result

    def record(self) -> dict:
        return {"columns": self.columns(), "means": self.means, "scales": self.scales,
                "knots": self.knots, "training_game_digest": self.training_game_digest}


def _phase_code(phase: str) -> float:
    if phase == "speech":
        return 0.0
    if phase == "speech_pk":
        return 1.0
    raise MapperProtocolError("unknown mapper phase")


def _numeric_matrix(rows: tuple[offline.CandidateRow, ...], names: tuple[str, ...]) -> np.ndarray:
    # Explicit whitelist: no S_pre, outcome, report truth, vote, or value can enter Z.
    data = np.asarray([[getattr(row.z, name) for name in names] for row in rows],
                      dtype=np.float64)
    if data.shape != (len(rows), len(names)) or not np.isfinite(data).all():
        raise MapperProtocolError("invalid numeric Z feature")
    for row in rows:
        _phase_code(row.z.phase)
        if row.phase != row.z.phase:
            raise MapperProtocolError("candidate and Z phases differ")
    return data


def validate_rows(rows: tuple[offline.CandidateRow, ...],
                  oof: OOFProvenance) -> dict[str, int]:
    if not rows or not isinstance(oof, OOFProvenance):
        raise MapperProtocolError("missing candidate rows or OOF provenance")
    identities, game_folds = set(), {}
    for row in rows:
        if not isinstance(row, offline.CandidateRow) or type(row.theta_ac) is not bool:
            raise MapperProtocolError("invalid mapper candidate row")
        identity = (row.game_id, row.boundary_id, row.acting_wolf, row.phase, row.candidate_j)
        if identity in identities or row.fold not in FOLDS or not row.prefix_digest:
            raise MapperProtocolError("duplicate identity or invalid row provenance")
        identities.add(identity)
        if row.game_id in game_folds and game_folds[row.game_id] != row.fold:
            raise MapperProtocolError("one game spans mapper folds")
        game_folds[row.game_id] = row.fold
        if row.fold not in oof.prediction_digest_by_fold:
            raise MapperProtocolError("candidate has no held-out OOF fold")
        if row.evidence_identity_digest != (oof.identity_digest if oof.experiment_a is not None else None):
            raise MapperProtocolError("candidate Q representation/temporal identity mismatch")
        _numeric_matrix((row,), CONTEXT_NUMERIC + EVIDENCE)
    return game_folds


def split_game_fold(rows: tuple[offline.CandidateRow, ...], fold: int,
                    oof: OOFProvenance) -> tuple[tuple[offline.CandidateRow, ...],
                                                 tuple[offline.CandidateRow, ...]]:
    validate_rows(rows, oof)
    if fold not in FOLDS:
        raise MapperProtocolError("unknown mapper fold")
    train = tuple(row for row in rows if row.fold != fold)
    held_out = tuple(row for row in rows if row.fold == fold)
    if not train or not held_out or {r.game_id for r in train} & {r.game_id for r in held_out}:
        raise MapperProtocolError("empty or game-overlapping mapper fold")
    return train, held_out


@dataclass(frozen=True)
class MapperPrediction:
    game_id: str
    boundary_id: str
    acting_wolf: str
    phase: str
    candidate_j: str
    theta_ac: bool  # Evaluation-only.
    mapper_fold: int
    model_spec: str
    raw_score: float
    raw_probability: float
    game_oof_fold: int
    prefix_digest: str
    qwen3_prediction_digest: str
    evaluation_seal_digest: str
    model_digest: str

    @property
    def identity(self) -> tuple[str, str, str, str, str]:
        return (self.game_id, self.boundary_id, self.acting_wolf,
                self.phase, self.candidate_j)


@dataclass(frozen=True)
class FittedMapper:
    spec: ModelSpec
    preprocessor: FoldPreprocessor
    estimator: LogisticRegression
    manifest: dict
    mapper_fold: int | None

    def predict_held_out(self, rows: tuple[offline.CandidateRow, ...],
                         oof: OOFProvenance) -> tuple[MapperPrediction, ...]:
        if (self.mapper_fold is None or not rows
                or any(row.fold != self.mapper_fold for row in rows)
                or self.manifest["oof_q"] != oof.record()):
            raise MapperProtocolError("prediction is not on this held-out mapper fold")
        validate_rows(rows, oof)
        x = self.preprocessor.transform(rows)
        scores = self.estimator.decision_function(x)
        probabilities = self.estimator.predict_proba(x)[:, 1]
        return tuple(MapperPrediction(row.game_id, row.boundary_id, row.acting_wolf,
            row.phase, row.candidate_j, row.theta_ac, self.mapper_fold, self.spec.name,
            float(score), float(probability), row.fold, row.prefix_digest,
            oof.prediction_digest_by_fold[row.fold], oof.evaluation_seal_digest,
            self.manifest["model_digest"])
            for row, score, probability in zip(rows, scores, probabilities, strict=True))


def _fit_mapper(rows: tuple[offline.CandidateRow, ...], spec: ModelSpec,
                oof: OOFProvenance, mapper_fold: int | None) -> FittedMapper:
    game_folds = validate_rows(rows, oof)
    if spec not in MODEL_SPECS or (mapper_fold is not None and mapper_fold not in FOLDS):
        raise MapperProtocolError("unknown fixed mapper spec/fold")
    if mapper_fold is not None and mapper_fold in set(game_folds.values()):
        raise MapperProtocolError("held-out mapper fold entered training data")
    if oof.experiment_a is not None and spec != MODEL_SPECS[3]:
        raise MapperProtocolError("Experiment A requires fixed M3")
    y = np.asarray([int(row.theta_ac) for row in rows], dtype=np.int64)
    if len(np.unique(y)) != 2:
        raise MapperProtocolError("binary likelihood requires both training classes")
    preprocessor = FoldPreprocessor.fit(rows, spec)
    x = preprocessor.transform(rows)
    estimator = LogisticRegression(C=spec.l2_inverse_strength,
        solver="lbfgs", fit_intercept=True, class_weight=None,
        max_iter=spec.max_iter, tol=spec.tolerance)
    estimator.fit(x, y)  # No sample weights or class resampling.
    if estimator.n_iter_[0] >= spec.max_iter:
        raise MapperProtocolError("mapper logistic optimization did not converge")
    game_ids = sorted(game_folds)
    manifest = {
        "model_spec": spec.record(),
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "offline_contract": {
            "publication_digest": offline.PUBLICATION_DIGEST,
            "role_sidecar_digest": offline.ROLE_SIDECAR_DIGEST,
            "fold_manifest_digest": offline.FOLD_MANIFEST_DIGEST,
            "source_digest": sha256(Path(offline.__file__).read_bytes()).hexdigest(),
        },
        "oof_q": oof.record(),
        "training_scope": "full_development" if mapper_fold is None else "mapper_cv_fold",
        "mapper_fold": mapper_fold,
        "training_game_ids": game_ids,
        "training_game_digest": _digest(game_ids),
        "training_game_folds": {game: game_folds[game] for game in game_ids},
        "training_candidate_count": len(rows),
        "preprocessing": preprocessor.record(),
        "likelihood": "unweighted_binary_negative_log_likelihood",
        "regularizer": {"type": "L2_nonintercept", "C": spec.l2_inverse_strength},
        "coefficient": estimator.coef_[0].tolist(),
        "intercept": float(estimator.intercept_[0]),
        **({"experiment_a": oof.experiment_a} if oof.experiment_a is not None else {}),
    }
    manifest["model_digest"] = _digest(manifest)
    return FittedMapper(spec, preprocessor, estimator, manifest, mapper_fold)


def fit_full_development(rows: tuple[offline.CandidateRow, ...], spec: ModelSpec,
                         oof: OOFProvenance) -> FittedMapper:
    """Future final-fit interface; caller must explicitly invoke it after model choice."""
    games = validate_rows(rows, oof)
    if len(games) != offline.EXPECTED_GAMES or len(rows) != offline.EXPECTED_CANDIDATES:
        raise MapperProtocolError("full fit requires the complete fixed development population")
    return _fit_mapper(rows, spec, oof, None)


@dataclass(frozen=True)
class ReliabilityBin:
    lower: float
    upper: float
    candidate_count: int
    mean_probability: float | None
    observed_rate: float | None


@dataclass(frozen=True)
class MetricSummary:
    candidate_count: int
    game_count: int
    pooled_log_loss: float
    pooled_brier: float
    game_macro_log_loss: float
    game_macro_brier: float
    auroc: float | None
    auprc: float | None
    calibration_intercept: float | None
    calibration_slope: float | None
    reliability_bins: tuple[ReliabilityBin, ...]


def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    clipped = np.clip(p, PROBABILITY_EPSILON, 1 - PROBABILITY_EPSILON)
    return float(-np.mean(y * np.log(clipped) + (1 - y) * np.log1p(-clipped)))


def _calibration_diagnostics(y: np.ndarray, raw_scores: np.ndarray) -> tuple[float | None, float | None]:
    if len(np.unique(y)) != 2 or np.ptp(raw_scores) <= 1e-12:
        return None, None
    if (max(raw_scores[y == 0]) < min(raw_scores[y == 1])
            or max(raw_scores[y == 1]) < min(raw_scores[y == 0])):
        return None, None  # Unpenalized intercept/slope are unidentified under separation.
    evaluator = LogisticRegression(C=np.inf, solver="lbfgs", fit_intercept=True,
                                   class_weight=None, max_iter=1000, tol=1e-8)
    with warnings.catch_warnings():
        # sklearn 1.8 warns that its unpenalized setting ignores C=np.inf;
        # this is exactly the development-only diagnostic fit requested here.
        warnings.filterwarnings("ignore", message="Setting penalty=None will ignore the C",
                                category=UserWarning)
        evaluator.fit(raw_scores.reshape(-1, 1), y)
    return float(evaluator.intercept_[0]), float(evaluator.coef_[0, 0])


def evaluate_predictions(predictions: tuple[MapperPrediction, ...], *,
                         reliability_bin_count: int = 10) -> MetricSummary:
    if not predictions or type(reliability_bin_count) is not int or reliability_bin_count < 2:
        raise MapperProtocolError("invalid development evaluation input")
    if len({p.identity for p in predictions}) != len(predictions):
        raise MapperProtocolError("duplicate development prediction identity")
    y = np.asarray([int(pred.theta_ac) for pred in predictions], dtype=np.int64)
    p = np.asarray([pred.raw_probability for pred in predictions], dtype=np.float64)
    raw_scores = np.asarray([pred.raw_score for pred in predictions], dtype=np.float64)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise MapperProtocolError("prediction probability outside [0,1]")
    if not np.isfinite(raw_scores).all():
        raise MapperProtocolError("non-finite raw mapper score")
    games = sorted({pred.game_id for pred in predictions})
    macro_log, macro_brier = [], []
    for game in games:
        mask = np.asarray([pred.game_id == game for pred in predictions])
        macro_log.append(_log_loss(y[mask], p[mask]))
        macro_brier.append(float(np.mean((p[mask] - y[mask]) ** 2)))
    bins = []
    indices = np.minimum((p * reliability_bin_count).astype(int), reliability_bin_count - 1)
    for index in range(reliability_bin_count):
        mask = indices == index
        bins.append(ReliabilityBin(index / reliability_bin_count,
            (index + 1) / reliability_bin_count, int(mask.sum()),
            float(p[mask].mean()) if mask.any() else None,
            float(y[mask].mean()) if mask.any() else None))
    intercept, slope = _calibration_diagnostics(y, raw_scores)
    has_both = len(np.unique(y)) == 2
    return MetricSummary(len(predictions), len(games), _log_loss(y, p),
        float(np.mean((p - y) ** 2)), float(np.mean(macro_log)),
        float(np.mean(macro_brier)),
        float(roc_auc_score(y, p)) if has_both else None,
        float(average_precision_score(y, p)) if has_both else None,
        intercept, slope, tuple(bins))


@dataclass(frozen=True)
class MapperCVResult:
    predictions: tuple[MapperPrediction, ...]
    metrics_by_spec: dict[str, MetricSummary]
    fit_manifests: tuple[dict, ...]


def run_mapper_cv(rows: tuple[offline.CandidateRow, ...],
                  oof: OOFProvenance) -> MapperCVResult:
    """Five-fold game CV harness; intentionally never called on import."""
    game_folds = validate_rows(rows, oof)
    if set(game_folds.values()) != set(FOLDS):
        raise MapperProtocolError("mapper CV requires all five development folds")
    predictions, manifests = [], []
    metrics = {}
    for spec in ((MODEL_SPECS[3],) if oof.experiment_a is not None else MODEL_SPECS):
        spec_predictions = []
        for fold in FOLDS:
            train, held_out = split_game_fold(rows, fold, oof)
            fitted = _fit_mapper(train, spec, oof, fold)
            spec_predictions.extend(fitted.predict_held_out(held_out, oof))
            manifests.append(fitted.manifest)
        if len(spec_predictions) != len(rows):
            raise MapperProtocolError("mapper CV prediction coverage mismatch")
        spec_predictions = tuple(spec_predictions)
        metrics[spec.name] = evaluate_predictions(spec_predictions)
        predictions.extend(spec_predictions)
    return MapperCVResult(tuple(predictions), metrics, tuple(manifests))
