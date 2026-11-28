"""同一合法 PRE 上的三状态原始报告概率。"""

import json
from math import fsum, log

import numpy as np

from werewolf.artifact_io import open_artifact_envelope, read_artifact_file
from werewolf.canonical_collection.public_history import PLAYER_IDS
from werewolf.observer_dtrs.context import contexts_for_pre
from werewolf.observer_dtrs.probabilities import ProbabilityMatrix


ARTIFACT_TYPE = "observer_dtrs_report_state_probability"
CONTEXT_ONLY = "context_only"
HIERARCHICAL = "hierarchical"
MULTINOMIAL = "multinomial"
FEATURES = ("s", "m", "s_squared", "m_squared", "s_times_m", "entropy", "uniform_tv",
            "day", "pk", "alive", "dual", "dual_times_tv", "dual_times_s")
CONTEXT_COLUMNS = tuple(range(7, 11))
DUAL_COLUMN = FEATURES.index("dual")
METHOD_FEATURES = {
    CONTEXT_ONLY: tuple(FEATURES[i] for i in CONTEXT_COLUMNS),
    HIERARCHICAL: FEATURES,
    MULTINOMIAL: FEATURES,
}


def observer_report_features(context, q_row):
    """以指定行动狼人为目标，对真实或假设 Q 行使用同一组特征。"""
    s = q_row[PLAYER_IDS.index(context.acting_wolf)]
    m = fsum(q_row[PLAYER_IDS.index(seat)] for seat in context.alive_wolf_ids if seat != context.acting_wolf)
    nonself = [mass for seat, mass in zip(PLAYER_IDS, q_row, strict=True) if seat != context.observer_id]
    entropy = -fsum(mass * log(mass) for mass in nonself if mass) / log(6)
    distance = .5 * fsum(abs(mass - 1 / 6) for mass in nonself)
    dual = float(context.wolf_count == 2)
    return (s, m, s * s, m * m, s * m, entropy, distance,
            float(context.day), float(context.phase == "pk_discussion"),
            float(len(context.alive_player_ids)), dual, dual * distance, dual * s)


def pre_features(pre, known_wolf_ids, matrix):
    """取合法存活非狼观察者，保留 Q 全行的分布特征。"""
    if not isinstance(matrix, ProbabilityMatrix):
        raise TypeError("需要当前 PRE 的冻结 ProbabilityMatrix")
    contexts = contexts_for_pre(pre, known_wolf_ids)
    if not contexts:
        raise ValueError("联合报告概率需要有合格观察者的狼人发言 PRE")
    if (matrix.game_id, matrix.boundary_id, matrix.prefix_digest) != (pre.game_id, pre.boundary_id, pre.prefix_digest):
        raise ValueError("Q game/PRE/prefix/observer identity mismatch")
    return [(ctx, observer_report_features(ctx, matrix.values[PLAYER_IDS.index(ctx.observer_id)])) for ctx in contexts]


def model_design(features, method):
    return features[:, CONTEXT_COLUMNS] if method == CONTEXT_ONLY else features


def standardize(x, parameters):
    scale = np.asarray(parameters["std"])
    return np.divide(x - parameters["mean"], scale, out=np.zeros_like(x), where=scale > 0)


def _linear(x, head):
    return standardize(x, head["standardization"]) @ np.asarray(head["coefficients"]) + head["intercept"]


def masked_log_softmax(relative_logits, dual):
    """E0 为参考 logit，E1 在单狼时不进入归一化。"""
    logits = np.column_stack((np.zeros(len(dual)), relative_logits))
    logits[~dual, 1] = -np.inf
    shifted = logits - logits.max(axis=1, keepdims=True)
    return shifted - np.logaddexp.reduce(shifted, axis=1, keepdims=True)


def log_probabilities(model, features):
    method = model["method"]
    if tuple(model["features"]) != METHOD_FEATURES[method]:
        raise ValueError("联合报告概率模型的特征顺序不匹配")
    x = model_design(features, method)
    dual = features[:, DUAL_COLUMN] == 1
    heads = model["heads"]
    if method == MULTINOMIAL:
        return masked_log_softmax(_linear(x, heads["state_logits"]), dual)

    speaker = _linear(x, heads["speaker"])[:, 0]
    teammate = _linear(x[dual], heads["teammate_given_no_speaker"])[:, 0]
    log_no_speaker = -np.logaddexp(0, speaker)
    log_e0 = log_no_speaker.copy()
    log_e0[dual] -= np.logaddexp(0, teammate)
    log_e1 = np.full(len(x), -np.inf)
    log_e1[dual] = log_no_speaker[dual] - np.logaddexp(0, -teammate)
    log_e2 = -np.logaddexp(0, -speaker)
    return np.column_stack((log_e0, log_e1, log_e2))


def probabilities(log_q, features):
    q = np.exp(log_q)
    if (q.shape != (len(features), 3) or not np.isfinite(q).all() or (q < 0).any()
            or not np.allclose(q.sum(axis=1), 1, rtol=0, atol=1e-12)
            or (q[features[:, DUAL_COLUMN] == 0, 1] != 0).any()):
        raise ValueError("联合概率必须归一化，且单狼的 E1 概率必须为零")
    return q


def probability_record(q):
    q0, q1, q2 = map(float, q)
    return {"q0": q0, "q1": q1, "q2": q2, "ps": q2, "pT": q1 + q2}


def predict_observers(panel, model):
    features = np.asarray([row for _, row in panel])
    q = probabilities(log_probabilities(model, features), features)
    return {context.observer_id: probability_record(row) for (context, _), row in zip(panel, q, strict=True)}


def predict_pre(pre, known_wolf_ids, matrix, model):
    """推理只接收权威 PRE、发言狼人已知队伍、冻结 Q 和模型参数。"""
    panel = pre_features(pre, known_wolf_ids, matrix)
    first = panel[0][0]
    return {
        "game_id": first.game_id, "boundary_id": first.boundary_id,
        "prefix_digest": first.prefix_digest, "step_index": first.step_index,
        "acting_wolf": first.acting_wolf, "alive_wolf_ids": first.alive_wolf_ids,
        "observers": predict_observers(panel, model),
    }


def load_development_model(path):
    """通过既有 Artifact 读取器载入开发阶段选定的模型。"""
    artifact = open_artifact_envelope(path, expected_artifact_type=ARTIFACT_TYPE,
                                     expected_schema_version=ARTIFACT_TYPE)
    saved = json.loads(read_artifact_file(artifact, "models.json"))
    return saved["models"][saved["selected_method"]]
