"""在已冻结的联合认知概率上拟合并预测历史序贯认知改善。"""

from dataclasses import dataclass
import json

import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.feature_extraction.text import TfidfVectorizer

from werewolf.artifact_io import canonical_json_bytes, open_artifact_envelope, read_artifact_file, sha256_bytes
from werewolf.canonical_collection.public_history import PLAYER_IDS
from werewolf.observer_dtrs import report_state_probability as probability
from werewolf.strategy.counterfactual_tom import derive_public_phase_speaker_order


ARTIFACT_TYPE = "observer_dtrs_cognitive_value"
METHODS = ("C0", "C1")
HEADS = ("redecision_opportunity", "target_stability", "cognitive_gain")
NUMERIC_FEATURES = (
    *probability.FEATURES,
    *(f"q_mass_{seat}" for seat in PLAYER_IDS),
    *(f"{kind}_{seat}" for kind in ("actor", "observer", "alive", "alive_wolf") for seat in PLAYER_IDS),
    "step_index", "earlier_speakers", "later_speakers", "observer_can_reply", "report_q0", "report_q1", "report_q2",
)
REPLY_COLUMN = NUMERIC_FEATURES.index("observer_can_reply")
PK_COLUMN = NUMERIC_FEATURES.index("pk")
TEXT_CONFIG = {"analyzer": "char", "ngram_range": (2, 3), "min_df": 2, "max_features": 2048,
               "lowercase": False, "norm": "l2", "smooth_idf": True, "sublinear_tf": False}
ACTION_INTERACTIONS = ("observer_can_reply", "ps", "pT", "pk")
REGULARIZATION = 0.01
SOLVER_OPTIONS = {"maxiter": 1000, "gtol": 1e-8, "ftol": 1e-11}
SPECIFICATION = {
    "numeric_features": NUMERIC_FEATURES,
    "numeric_scaling": "game_pre_observer_weighted_training_mean_and_sd",
    "history": "canonical_public_event_fields_before_action_without_event_ids_or_indices",
    "text": {"history": TEXT_CONFIG, "action": TEXT_CONFIG, "idf_document_unit": "one_per_pre"},
    "action_interactions": ACTION_INTERACTIONS,
    "features_by_method": {"C0": ["numeric", "history"], "C1": ["numeric", "history", "action", "action_interactions"]},
    "regularization": REGULARIZATION,
    "loss": "conditional_weighted_mean_bernoulli_cross_entropy_plus_half_lambda_l2_slopes",
    "intercept_penalty": False,
    "gain_transform": "Z=(G_total+1)/2",
    "solver": {"method": "L-BFGS-B", **SOLVER_OPTIONS},
    "base_weight": "equal_game_equal_pre_equal_initial_observer",
}


@dataclass(frozen=True, slots=True)
class InputRow:
    """单个观察者的决策前输入，当前发言正文单独保存。"""

    pre_id: tuple[str, str]
    history: str
    action: str
    numeric: tuple[float, ...]


def pre_inputs(pre, panel, q_values):
    """共用合法 PRE 表示，供历史训练和候选推理使用。"""
    order = derive_public_phase_speaker_order(pre)
    position = order.index(pre.current_speaker)
    later = order[position + 1:]
    history = canonical_json_bytes([
        {key: value for key, value in event.to_record().items() if key not in ("event_id", "event_index")}
        for event in pre.public_event_history.events
    ]).decode("utf-8")
    numeric = {}
    for context, features in panel:
        observer = context.observer_id
        memberships = ((context.acting_wolf,), (observer,), context.alive_player_ids, context.alive_wolf_ids)
        numeric[observer] = (
            *features, *q_values[PLAYER_IDS.index(observer)],
            *(float(seat in members) for members in memberships for seat in PLAYER_IDS),
            float(context.step_index), float(position), float(len(later)), float(observer in later),
        )
    return history, numeric


def joint_values(record):
    return tuple(record[f"q{k}"] for k in range(3))


def fit_transforms(rows, weights):
    """历史及行动的词表各按每 PRE 一个训练文档拟合。"""
    numeric = np.asarray([row.numeric for row in rows])
    mean = np.average(numeric, weights=weights, axis=0)
    sd = np.sqrt(np.average((numeric - mean) ** 2, weights=weights, axis=0))
    sd[np.ptp(numeric, axis=0) == 0] = 0
    transforms = {"numeric": {"mean": mean.tolist(), "std": sd.tolist()}}
    pres = {row.pre_id: row for row in rows}
    for field in ("history", "action"):
        vectorizer = TfidfVectorizer(**TEXT_CONFIG).fit(getattr(row, field) for row in pres.values())
        transforms[field] = {"vocabulary": {token: int(index) for token, index in vectorizer.vocabulary_.items()},
                             "idf": vectorizer.idf_.tolist()}
    return transforms


def _text_matrix(rows, name, transform):
    vectorizer = TfidfVectorizer(**TEXT_CONFIG, vocabulary=transform["vocabulary"])
    vectorizer.idf_ = np.asarray(transform["idf"], dtype=float)
    documents = {}
    indices = [documents.setdefault(text, len(documents)) for text in (getattr(row, name) for row in rows)]
    return vectorizer.transform(list(documents))[indices]


def design_matrix(rows, transforms, method):
    """两组共用历史和数值特征，C1 额外使用当前正文及交互。"""
    if method not in METHODS:
        raise ValueError(f"未知认知改善预测方法：{method}")
    numeric = np.asarray([row.numeric for row in rows], dtype=float).reshape(-1, len(NUMERIC_FEATURES))
    matrix = [sparse.csr_matrix(probability.standardize(numeric, transforms["numeric"])),
              _text_matrix(rows, "history", transforms["history"])]
    if method == "C1":
        action = _text_matrix(rows, "action", transforms["action"])
        matrix.append(action)
        q1, q2 = numeric[:, -2], numeric[:, -1]
        gates = (numeric[:, REPLY_COLUMN], q2, q1 + q2, numeric[:, PK_COLUMN])
        for gate in gates:
            interaction = action.multiply(gate[:, None]).tocsr()
            interaction.eliminate_zeros()
            matrix.append(interaction)
    return sparse.hstack(matrix, format="csr")


def fit_head(design, target, weights):
    """二元概率和分数响应共用带 L2 的 Bernoulli 准似然。"""
    valid = np.isfinite(target)
    target, weights = target[valid], weights[valid]
    mass = float(weights.sum())
    support = {"rows": len(target), "weight": mass, "min": float(target.min()) if len(target) else None,
               "max": float(target.max()) if len(target) else None,
               "mean": float(weights @ target / mass) if mass > 0 else None}
    if not len(target):
        return {"status": "no_support", "support": support}
    if support["min"] == support["max"]:
        return {"status": "no_variation", "support": support}

    x = design[valid]
    weights = weights / mass

    def objective(coefficients):
        intercept, slopes = coefficients[0], coefficients[1:]
        logits = intercept + x @ slopes
        nll = target * np.logaddexp(0, -logits) + (1 - target) * np.logaddexp(0, logits)
        residual = weights * (expit(logits) - target)
        value = weights @ nll + (REGULARIZATION / 2) * (slopes @ slopes)
        gradient = np.r_[residual.sum(), x.T @ residual + REGULARIZATION * slopes]
        return float(value), gradient

    result = minimize(objective, np.zeros(x.shape[1] + 1), jac=True, method="L-BFGS-B", options=SOLVER_OPTIONS)
    if not result.success or not np.isfinite(result.x).all() or not np.isfinite(result.fun):
        raise RuntimeError(f"主动认知价值模型拟合未收敛：{result.message}")
    return {"status": "fitted", "support": support, "intercept": float(result.x[0]),
            "coefficients": result.x[1:].tolist(), "iterations": result.nit}


def predict_heads(design, heads):
    estimates = {}
    for name in HEADS:
        head = heads[name]
        if head["status"] == "fitted":
            score = expit(head["intercept"] + design @ np.asarray(head["coefficients"]))
            estimates[name] = 2 * score - 1 if name == "cognitive_gain" else score
        else:
            estimates[name] = np.full(design.shape[0], np.nan)
    return estimates


def observer_prediction(estimates, index):
    return {f"predicted_{name}": float(values[index]) if np.isfinite(values[index]) else None
            for name, values in estimates.items()}


@dataclass(frozen=True)
class CognitiveValuePredictor:
    """已联结第一层来源的 C1 预测器，仅消费当前 PRE 和有序候选。"""

    report_model: dict
    transforms: dict
    heads: dict

    def predict_candidates(self, batch, known_wolf_ids, matrix):
        pre = batch.pre
        panel = probability.pre_features(pre, known_wolf_ids, matrix)
        current = probability.predict_observers(panel, self.report_model)
        history, numeric = pre_inputs(pre, panel, matrix.values)
        pre_id = pre.game_id, pre.boundary_id
        inputs = {seat: (*values, *joint_values(current[seat])) for seat, values in numeric.items()}
        rows = [InputRow(pre_id, history, candidate.text, values)
                for candidate in batch.candidates for values in inputs.values()]
        estimates = predict_heads(design_matrix(rows, self.transforms, "C1"), self.heads)
        first = panel[0][0]
        return {
            "game_id": first.game_id, "boundary_id": first.boundary_id, "prefix_digest": first.prefix_digest,
            "step_index": first.step_index, "acting_wolf": first.acting_wolf, "alive_wolf_ids": first.alive_wolf_ids,
            "head_statuses": {name: self.heads[name]["status"] for name in HEADS},
            "candidates": [
                {"index": index, "text": candidate.text, "text_sha256": sha256_bytes(candidate.text.encode("utf-8")),
                 "observers": {seat: observer_prediction(estimates, index * len(inputs) + offset)
                               for offset, seat in enumerate(inputs)}}
                for index, candidate in enumerate(batch.candidates)
            ],
        }


def load_predictor(path, probability_path):
    archive = open_artifact_envelope(path, expected_artifact_type=ARTIFACT_TYPE, expected_schema_version=ARTIFACT_TYPE)
    source = open_artifact_envelope(probability_path, expected_artifact_type=probability.ARTIFACT_TYPE,
                                    expected_schema_version=probability.ARTIFACT_TYPE)
    if source.manifest_digest != archive.manifest["probability_source"]["manifest_digest"]:
        raise ValueError("认知价值模型与第一层冻结概率来源不符")
    saved = json.loads(read_artifact_file(archive, "models.json"))
    if canonical_json_bytes(saved["specification"]) != canonical_json_bytes(SPECIFICATION):
        raise ValueError("认知价值模型与固定特征规格不符")
    return CognitiveValuePredictor(probability.load_development_model(probability_path), saved["transforms"],
                                   saved["models"]["C1"])
