"""独立的发言后果预测，使用父 PRE 上可推导的公开调度。"""

from dataclasses import asdict, dataclass, replace
from math import fsum

import torch

from werewolf.artifact_io import canonical_json_bytes, sha256_bytes
from werewolf.canonical_collection.public_history import PLAYER_IDS
from werewolf.canonical_collection.speech import V1SpeechAction
from werewolf.observer_dtrs import report_state_probability as probability
from werewolf.observer_dtrs.final_tom_capsule import CapsulePredictor
from werewolf.observer_dtrs.probabilities import ProbabilityMatrix
from werewolf.strategy.counterfactual_tom import derive_public_phase_speaker_order
from werewolf.structured_history import StructuredTokenDescriptor, StructuredTokenPlan
from werewolf.tom.evaluation import predict_structured_observers
from werewolf.tom.final_capacity import derived_capacity, validate_final_pre


CAPACITY = derived_capacity(1024)
VOTE_PHASE = {"discussion": "vote", "pk_discussion": "pk_vote"}
PARSER_PHASE = {"discussion": "speech", "pk_discussion": "speech_pk"}
MODEL_FIELDS = ("token_type", "source", "action", "target", "day", "phase")


class ConsequenceCapacityError(ValueError):
    """完整的假设输入超过冻结容量。"""


@dataclass(frozen=True, slots=True)
class ConsequencePlan:
    game_id: str
    parent_boundary_id: str
    parent_prefix_digest: str
    text_sha256: str
    token_plan: StructuredTokenPlan
    model_input_digest: str

    @property
    def marker(self):
        return self.token_plan.tokens[-1]


def _model_tokens(plan):
    return [tuple(getattr(t, field) for field in MODEL_FIELDS) for t in plan.tokens]


def model_input_differences(left, right):
    """逐字段比较冻结张量化的来源，固定容量下长度相同即掩码相同。"""
    a, b = _model_tokens(left), _model_tokens(right)
    if left.token_count != right.token_count:
        return ("token_count", "attention_mask")
    return tuple(field for i, field in enumerate(MODEL_FIELDS) if any(x[i] != y[i] for x, y in zip(a, b, strict=True)))


def build_consequence_plan(pre, text, actions):
    """复用真实前缀，追加仅供预测的发言与原生确定性调度描述符。"""
    parent = validate_final_pre(pre, CAPACITY)
    speaker = pre.current_speaker
    if not isinstance(text, str) or not text.strip():
        raise ValueError("假设发言需要完整原文")
    actions = tuple(actions)
    if any(not isinstance(action, V1SpeechAction) or action.subject != speaker for action in actions):
        raise ValueError("V1 候选动作必须属于当前发言人")
    if parent.token_count + len(actions) + 2 > CAPACITY["max_seq_len"]:
        raise ConsequenceCapacityError("完整发言后果超过冻结序列容量")

    order = derive_public_phase_speaker_order(pre)
    index = order.index(speaker) + 1
    day, phase = pre.public_temporal_state.day, pre.public_temporal_state.phase.value
    event_index, start = len(pre.public_event_history.events), parent.token_count
    text_sha256 = sha256_bytes(text.encode())
    scenario = sha256_bytes(canonical_json_bytes({
        "parent_prefix_digest": pre.prefix_digest, "hypothetical_speech_sha256": text_sha256,
        "actions": [action.to_record() for action in actions],
    }))
    speech_id = f"hypothetical:{scenario}:speech"
    speech = StructuredTokenDescriptor(start, "public_speech", speaker, None, None, day, phase, speech_id, event_index, 0)
    semantic = tuple(StructuredTokenDescriptor(start + i, "speech_action", action.subject, action.action, action.object,
                                              day, phase, speech_id, event_index, i)
                     for i, action in enumerate(actions, start=1))
    next_speaker = order[index] if index < len(order) else None
    marker = StructuredTokenDescriptor(start + 1 + len(actions), "turn_start" if next_speaker else "phase_change",
                                       next_speaker, None, None, day, phase if next_speaker else VOTE_PHASE[phase],
                                       f"hypothetical:{scenario}:schedule", event_index + 1, 0)
    tokens = (*parent.tokens, speech, *semantic, marker)
    plan = StructuredTokenPlan(parent.planner_version, scenario, tokens, len(tokens), "")
    record = plan.to_record()
    del record["plan_digest"]
    plan = replace(plan, plan_digest=sha256_bytes(canonical_json_bytes(record)))
    return ConsequencePlan(pre.game_id, pre.boundary_id, pre.prefix_digest, text_sha256, plan,
                           sha256_bytes(canonical_json_bytes(_model_tokens(plan))))


@dataclass(frozen=True)
class Forecast:
    matrix: ProbabilityMatrix
    log_q: torch.Tensor


class SpeechConsequencePredictor(CapsulePredictor):
    """在通过原 Capsule 来源校验的冻结图上运行独立假设计划。"""

    def forecast(self, plan: ConsequencePlan):
        values, nonself_log_q = predict_structured_observers(plan.token_plan, self._model, self._device)
        digest = plan.token_plan.prefix_digest
        matrix = ProbabilityMatrix(plan.game_id, f"hypothetical:{digest}", digest, values)
        nonself = ~torch.eye(7, dtype=torch.bool)
        log_q = torch.full((7, 7), -torch.inf)
        log_q[nonself] = torch.tensor(nonself_log_q).flatten()
        return Forecast(matrix, log_q)


@dataclass(frozen=True)
class Exposure:
    observers: dict
    mean_team_probability: float
    mean_team_mass: float


def _exposure(features, model):
    panel = probability.predict_observers(features, model)
    for context, row in features:
        panel[context.observer_id]["team_mass"] = row[0] + row[1]
    return Exposure(panel, fsum(row["pT"] for row in panel.values()) / len(panel),
                    fsum(row["team_mass"] for row in panel.values()) / len(panel))


def current_exposure(pre, known_wolf_ids, matrix, model):
    panel = probability.pre_features(pre, known_wolf_ids, matrix)
    return tuple(context for context, _ in panel), _exposure(panel, model)


def consequence_exposure(contexts, plan, matrix, model):
    first = contexts[0]
    if ((plan.game_id, plan.parent_boundary_id, plan.parent_prefix_digest)
            != (first.game_id, first.boundary_id, first.prefix_digest)
            or (matrix.game_id, matrix.prefix_digest) != (plan.game_id, plan.token_plan.prefix_digest)):
        raise ValueError("假设 Q 与父 PRE 或候选计划不一致")
    panel = [(ctx, probability.observer_report_features(ctx, matrix.values[PLAYER_IDS.index(ctx.observer_id)]))
             for ctx in contexts]
    return _exposure(panel, model)


def social_values(current, updated):
    return {"V_soc": 1 - updated.mean_team_probability,
            "G_soc": current.mean_team_probability - updated.mean_team_probability,
            "V_Q": 1 - updated.mean_team_mass, "G_Q": current.mean_team_mass - updated.mean_team_mass}


def forecast_record(plan, forecast, updated, current):
    return {"plan_digest": plan.token_plan.plan_digest, "model_input_digest": plan.model_input_digest,
            "hypothetical_boundary_digest": plan.token_plan.prefix_digest,
            "token_count": plan.token_plan.token_count, "marker": plan.marker.to_record(),
            "q": forecast.matrix.values, "exposure": asdict(updated), **social_values(current, updated)}


def score_candidates(batch, known_wolf_ids, matrix, model, *, perceiver, predictor):
    """解析既有候选，重复输入共享同一预测，候选身份分别保留。"""
    pre = batch.pre
    contexts, current = current_exposure(pre, known_wolf_ids, matrix, model)
    cache, records = {}, []
    for i, candidate in enumerate(batch.candidates):
        audit = perceiver.parse_with_audit(speaker=PLAYER_IDS.index(pre.current_speaker) + 1,
                                          speech=candidate.text, day=pre.public_temporal_state.day,
                                          phase=PARSER_PHASE[pre.public_temporal_state.phase.value])
        record = {"index": i, **asdict(candidate), "text_sha256": sha256_bytes(candidate.text.encode()),
                  "parse": asdict(audit), "prediction": None}
        records.append(record)
        if audit.parse_status != "ok":
            record.update(v1_status="parser_error", prediction_status="no_valid_parse")
            continue
        actions = tuple(V1SpeechAction(*row) for row in audit.normalized_actions)
        record["v1_status"] = "ok" if actions else "no_action"
        try:
            plan = build_consequence_plan(pre, candidate.text, actions)
        except ConsequenceCapacityError:
            record["prediction_status"] = "capacity_exceeded"
            continue
        if plan.model_input_digest not in cache:
            cache[plan.model_input_digest] = predictor.forecast(plan)
        forecasts = cache[plan.model_input_digest]
        # 重用的是数值，另一个原文仍属于其自己的假设边界。
        if forecasts.matrix.prefix_digest != plan.token_plan.prefix_digest:
            digest = plan.token_plan.prefix_digest
            forecasts = replace(forecasts, matrix=ProbabilityMatrix(plan.game_id, f"hypothetical:{digest}", digest,
                                                                     forecasts.matrix.values))
        updated = consequence_exposure(contexts, plan, forecasts.matrix, model)
        record.update(prediction_status="predicted", prediction=forecast_record(plan, forecasts, updated, current))

    return {"game_id": pre.game_id, "boundary_id": pre.boundary_id, "prefix_digest": pre.prefix_digest,
            "step_index": pre.step_index, "acting_wolf": pre.current_speaker,
            "alive_wolf_ids": contexts[0].alive_wolf_ids, "observer_ids": tuple(current.observers),
            "topic": batch.topic, "current_exposure": asdict(current), "candidates": records}
