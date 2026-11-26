"""Wolf speech-cognition ablation on the existing canonical game loop."""

from werewolf.artifact_io import canonical_json_bytes, sha256_bytes
from werewolf.canonical_collection.public_history import PLAYER_IDS
from werewolf.speech.validation import normalize_player
from werewolf.game_runtime import eval as run_game


def epoch35_identity(device):
    from werewolf.tom.final_evaluation import TEST_TEMPORAL_DIGEST
    from werewolf.tom.makto import EPOCH35_DIGEST
    from werewolf.tom.qwen3_large import graph_identity

    if device not in {"cpu", "cuda"}:
        raise ValueError("Epoch 35 device must be cpu or cuda")
    return {"checkpoint_digest": EPOCH35_DIGEST, "temporal_artifact_digest": TEST_TEMPORAL_DIGEST,
            "graph_digest": graph_identity()["graph_digest"], "device": device}


class Epoch35GameplayPredictor:
    """在本进程复用已冻结权重，每个合法公开 PRE 只预测一次七席矩阵。"""

    def __init__(self, checkpoint, temporal, device):
        from werewolf.tom.makto import load_epoch35_model

        self.identity = epoch35_identity(device)
        self.model = load_epoch35_model(checkpoint, temporal)
        if self.model.temporal.artifact_digest != self.identity["temporal_artifact_digest"]:
            raise ValueError("Epoch 35 gameplay Temporal identity mismatch")
        self.model.to(device).eval()

    def predict(self, prefix):
        from werewolf.structured_history import plan_structured_history
        from werewolf.tom.evaluation import predict_structured_observers

        plan = plan_structured_history(prefix)
        if plan.token_count > self.model.capacity.max_seq_len:
            raise ValueError(f"Epoch 35 PRE exceeds model capacity: {prefix.boundary_id}")
        if prefix.maximum_public_day >= len(self.model.temporal.day):
            raise ValueError(f"Epoch 35 PRE exceeds frozen Temporal: {prefix.boundary_id}")
        probabilities, _ = predict_structured_observers(plan, self.model, self.identity["device"])
        return probabilities


def eligible_observers(prefix, observation):
    """Use only the acting wolf's delivered legal view and the public PRE."""
    if observation.get("identity") != "Werewolf":
        raise ValueError("ToM rows require a wolf observation")
    actor = normalize_player(observation["current_act_idx"])
    if actor != prefix.current_speaker:
        raise ValueError("ToM speaker differs from public PRE")
    alive = tuple(normalize_player(seat) for seat in
                  observation["authoritative_public_state"]["alive_players"])
    if len(set(alive)) != len(alive) or set(alive) != set(prefix.alive_observer_ids):
        raise ValueError("legal public alive seats differ from PRE")
    if actor not in alive:
        raise ValueError("dead wolf cannot receive ToM")
    team_records = [log.content.get("wolf_team") for log in observation["game_log"]
                    if log.event == "werewolf_team_info"]
    if len(team_records) != 1 or not isinstance(team_records[0], list):
        raise ValueError("wolf observation must contain one legal team record")
    team = tuple(normalize_player(seat) for seat in team_records[0])
    if len(set(team)) != 2 or actor not in team:
        raise ValueError("invalid legally known wolf team")
    return tuple(seat for seat in PLAYER_IDS if seat in alive and seat not in team)


def tom_payload(prefix, observation, probabilities):
    import math

    if len(probabilities) != 7 or any(len(row) != 7 for row in probabilities):
        raise ValueError("ToM predictor must return a full 7x7 matrix")
    for observer, row in enumerate(probabilities):
        if (any(type(value) is not float or not math.isfinite(value) or
                not 0.0 <= value <= 1.0 for value in row)
                or row[observer] != 0.0
                or abs(sum(row) - 1.0) > 1e-6):
            raise ValueError("ToM predictor returned an invalid distribution")
    rows = eligible_observers(prefix, observation)
    return {"target_ids": list(PLAYER_IDS), "rows": [
        {"observer_id": observer,
         "probabilities": list(probabilities[PLAYER_IDS.index(observer)])}
        for observer in rows]}


def prediction_request(prefix):
    identity = {"game_id": prefix.game_id, "boundary_id": prefix.boundary_id,
                "prefix_digest": prefix.prefix_digest, "current_speaker": prefix.current_speaker}
    return {**identity, "request_id": sha256_bytes(canonical_json_bytes(identity))}


def run_ablation(env, agents, roles, *, recorder, call_audit, arm,
                 predictor=None, treatment_audit=None, prediction_attempts=None):
    """Run one game; the two arms share the production action loop."""
    if arm not in {"Wolf+ToM", "Wolf-NoToM"}:
        raise ValueError("unknown gameplay ablation arm")
    if arm == "Wolf-NoToM":
        if predictor is not None:
            raise ValueError("NoToM must not receive a predictor")
        return run_game(env, agents, roles, canonical_recorder=recorder,
                        call_audit=call_audit)
    if predictor is None or treatment_audit is None:
        raise ValueError("+ToM requires an existing predictor and private treatment audit")

    def wolf_speech_tom(prefix, observation):
        # 同一 PRE 的认知重试复用一次预测，失败直接向原采集生命周期传播。
        eligible_observers(prefix, observation)
        boundary = {
            "game_id": prefix.game_id,
            "boundary_id": prefix.boundary_id,
            "prefix_digest": prefix.prefix_digest,
        }
        if prediction_attempts is not None:
            attempt = {**prediction_request(prefix), "status": "unknown"}
            prediction_attempts.append(attempt)
        try:
            payload = tom_payload(prefix, observation, predictor.predict(prefix))
        except Exception as error:
            if prediction_attempts is not None:
                attempt.update(status="failed", error_type=type(error).__name__)
            raise
        wire = canonical_json_bytes(payload)
        if prediction_attempts is None:
            binding = {"fit_digest": predictor.fit_digest, "seal_digest": predictor.seal_digest}
        else:
            binding = {"predictor": predictor.identity, "request_id": attempt["request_id"]}
        treatment_audit.append({
            **boundary,
            "observer_ids": [row["observer_id"] for row in payload["rows"]],
            "payload_digest": sha256_bytes(wire),
            **binding,
        })
        if prediction_attempts is not None:
            attempt["status"] = "succeeded"
        return wire.decode("utf-8")

    return run_game(env, agents, roles, canonical_recorder=recorder,
                    call_audit=call_audit, wolf_speech_tom=wolf_speech_tom)
