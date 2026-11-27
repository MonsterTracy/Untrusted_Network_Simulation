"""Classic7 环境装配、Agent 分配与唯一游戏执行循环。"""

from __future__ import annotations

from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass, replace
from math import isfinite
import os
from pathlib import Path
import random
from re import fullmatch
import subprocess
from copy import deepcopy
from time import perf_counter
from typing import Any


from werewolf.agents.gpt_agent import GPTAgent
from werewolf.artifact_io import canonical_json_bytes, sha256_bytes
from werewolf.artifact_io.canonical import require_sha256
from werewolf.backends import (
    OpenAICompatibleBackend,
    load_named_backends,
    resolve_backend,
)
from werewolf.backends.openai_compatible import model_request_context, model_request_details
from werewolf.envs.werewolf_text_env_v0 import V1SpeechPerceptionExhausted, WerewolfTextEnvV0
from werewolf.canonical_collection.pre import construct_speaker_pre_belief_handoff
from werewolf.canonical_collection.public_history import (
    PLAYER_IDS, PublicEventHistory, PublicPhase, freeze_public_event_history,
)
from werewolf.canonical_collection.runtime import (
    classic7_native_winner,
    freeze_classic7_speech_pre,
    replay_classic7_actions,
)
from werewolf.speech.private_belief_perceiver import PlayingAgentBeliefReporter, STATUS_OK
from werewolf.speech.speech_perceiver import SpeechPerceiver
from werewolf.speech.validation import normalize_player
from werewolf.runtime_config import MULTIMIND_METHOD, normalize_runtime_config


PUBLIC_SPEECH_RUNTIME_PHASES = frozenset({"speech", "speech_pk"})
_GAMEPLAY_SOURCE_FILES = (
    "werewolf/game_runtime.py",
    "werewolf/runtime_config.py",
    "werewolf/agents/gpt_agent.py",
    "werewolf/agents/llm_agent.py",
    "werewolf/agents/prompt_template_v0.py",
    "werewolf/backends/factory.py",
    "werewolf/backends/openai_compatible.py",
    "werewolf/canonical_collection/production_runtime.py",
    "werewolf/canonical_collection/runtime.py",
    "werewolf/envs/werewolf_text_env_v0.py",
    "werewolf/speech/private_belief_perceiver.py",
    "werewolf/speech/speech_perceiver.py",
)
_MULTIMIND_SOURCE_FILES = (
    "werewolf/tom/multimind.py",
    "werewolf/strategy/multimind_planning.py",
    "werewolf/strategy/speech_planning.py",
)

UNKNOWN = "UNKNOWN"
_VOTE_FIELDS = (
    "wolf_participations", "wolf_votes_received", "wolf_votes_from_villagers",
    "wolf_votes_from_teammates", "wolf_eligible_rounds", "villager_valid_votes",
    "wolves_exiled", "villagers_exiled", "day1_wolf_exile_games",
    "wolf_ballots", "villager_ballots", "wolf_abstentions", "villager_abstentions",
    "normal_vote_rounds", "pk_vote_rounds",
)


@dataclass(frozen=True)
class GameplayOnlyResult:
    """单局内存结果，包含可独立重放的原生动作。"""

    seed: int
    roles: tuple[str, ...]
    seat_profiles: tuple[str, ...]
    decision_methods: tuple[str | None, ...]
    source_identity: dict[str, Any]
    submitted_actions: tuple[tuple[str, int | str], ...] = ()
    terminal_reached: bool = False
    winner: str | None = None
    replay_winner: str | None = None
    public_event_digest: str | None = None
    failure_type: str | None = None
    failure_stage: str | None = None
    failure_phase: str | None = None
    failure_message: str | None = None
    failure_cause_type: str | None = None
    speech_parser_attempts: int = 0
    completed_steps: int = 0
    elapsed_seconds: float | None = None
    request_timeout_seconds: float | None = None
    request_records: tuple[dict[str, Any], ...] = ()
    speech_results: tuple[dict[str, Any], ...] = ()
    model_requests: int | str = "UNKNOWN"
    retries: int | str = "UNKNOWN"
    token_usage: dict[str, int | str] | str = "UNKNOWN"


def _gameplay_source_identity(*, include_file_digests=False, multimind=False):
    repo = Path(__file__).resolve().parents[1]
    identity = {"commit": "UNKNOWN", "tracked_source_diff_sha256": "UNKNOWN"}
    try:
        identity["commit"] = subprocess.check_output(
            ["git", "--no-optional-locks", "rev-parse", "HEAD"], cwd=repo,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        diff = subprocess.check_output(
            ["git", "--no-optional-locks", "diff", "--no-ext-diff", "--no-textconv",
             "--binary", "HEAD", "--", "werewolf"], cwd=repo, stderr=subprocess.DEVNULL,
        )
        identity["tracked_source_diff_sha256"] = sha256_bytes(diff)
    except (OSError, subprocess.CalledProcessError):
        pass
    if include_file_digests:
        files = {}
        source_files = _GAMEPLAY_SOURCE_FILES + (_MULTIMIND_SOURCE_FILES if multimind else ())
        for name in source_files:
            try:
                files[name] = sha256_bytes((repo / name).read_bytes())
            except OSError:
                files[name] = "UNKNOWN"
        identity["file_sha256"] = files
    return identity


def _copy_gameplay_source_identity(snapshot):
    """核对结果所需的身份字段，并与调用方持有的快照隔离。"""
    if not isinstance(snapshot, dict):
        raise TypeError("source_identity must be a dictionary")
    commit = snapshot.get("commit")
    if not isinstance(commit, str) or not fullmatch(r"[0-9a-f]{40}(?:[0-9a-f]{24})?", commit):
        raise ValueError("source_identity.commit must be a Git object ID")
    branch = snapshot.get("branch")
    if not isinstance(branch, str) or not branch.strip():
        raise ValueError("source_identity.branch must be a non-empty string")
    require_sha256(snapshot.get("tracked_source_diff_sha256"), "source_identity.tracked_source_diff_sha256")
    files = snapshot.get("file_sha256")
    if not isinstance(files, dict) or not files:
        raise ValueError("source_identity.file_sha256 must be a non-empty dictionary")
    for path, digest in files.items():
        if not isinstance(path, str) or not path.strip():
            raise ValueError("source_identity.file_sha256 requires non-empty path keys")
        require_sha256(digest, f"source_identity.file_sha256[{path!r}]")
    return deepcopy(snapshot)


def _gameplay_speaker_handoff(env, agent, observation):
    seat = observation["current_act_idx"]
    actor = f"player{seat}"
    prefix = freeze_classic7_speech_pre(env, game_id="gameplay-only", actor=actor)
    observation_id = prefix.speaker_pre_belief_observation_id
    known_wolves, known_non_wolves = env.observer_private_knowledge(seat)
    report = PlayingAgentBeliefReporter().report(
        agent=agent, observation=observation, observer_id=actor, pre_prefix=prefix,
        observation_id=observation_id, agent_backend_id=getattr(agent, "backend_id", "UNKNOWN"),
        known_werewolves=known_wolves, known_non_werewolves=known_non_wolves,
    )
    if report["status"] != STATUS_OK:
        raise RuntimeError(f"speaker PRE report {report['status']}: {report['error']}")
    # 摘要绑定实际通过验证的报告，只用于本次发言交接。
    return construct_speaker_pre_belief_handoff(
        prefix, observation_id=observation_id, observer_id=actor,
        observation_digest=sha256_bytes(canonical_json_bytes(report)),
        observation_status="success", suspicion_support=report["suspected_werewolves"],
    )


def _is_last_public_speaker(history, alive):
    """只从已发生的公开发言和普通投票确认当前发言是否末位。"""
    state = history.current_state
    turn_count = sum(e.event_type == "turn_start" and e.temporal_state == state for e in history.events)
    if state.phase is PublicPhase.DISCUSSION:
        return turn_count == len(alive)
    ballot = next(e for e in reversed(history.events)
                  if e.event_type == "vote_result" and e.temporal_state.phase is PublicPhase.VOTE
                  and e.temporal_state.day == state.day)
    counts = list(Counter(v.target for v in ballot.votes if v.target is not None).values())
    return turn_count == counts.count(max(counts))


def _gameplay_multimind_speech(env, agent, observation, reasoner, evidence, check_interrupt):
    from werewolf.strategy.multimind_planning import plan_speech
    from werewolf.tom.multimind import encode_history

    seat = observation["current_act_idx"]
    speaker = normalize_player(seat)
    if (observation.get("observer_id") != seat or observation.get("identity") != "Werewolf"
            or observation.get("valid_action") != (env.phase, -1)):
        raise ValueError("MultiMind requires the current wolf's legal speech observation")
    public = freeze_public_event_history(env.public_events)
    turn = public.events[-1]
    if turn.event_type != "turn_start" or turn.speaker != speaker:
        raise ValueError("MultiMind speech does not match the current public turn")
    alive = tuple(normalize_player(p) for p in observation["authoritative_public_state"]["alive_players"])
    evidence.update(boundary_event_id=turn.event_id, day=turn.temporal_state.day,
                    last_same_phase_speaker=_is_last_public_speaker(public, alive))
    history = encode_history(public, tuple(env.speech_annotations))
    known_wolves, _ = env.observer_private_knowledge(seat)
    seed = int(sha256_bytes(canonical_json_bytes([env.random_seed, speaker, turn.event_id]))[:16], 16)
    temperature, max_tokens = agent._request_limits()

    evidence["active_stage"] = "multimind_belief"
    belief = agent._generate_belief(observation, temperature=temperature, max_tokens=max_tokens)
    evidence["active_stage"] = "multimind_planner"
    if check_interrupt is not None:
        check_interrupt()
    evidence.update(planner_attempted=True, reasoner_prediction_succeeded=None,
                    completed_simulations=None, candidate_score_calls=None, reasoner_forward_calls=None,
                    planner_seed=seed)
    failed_search = {}
    started = perf_counter()
    try:
        plan = plan_speech(history, speaker, alive_players=alive,
                           known_wolves=known_wolves, reasoner=reasoner, seed=seed,
                           failure_evidence=failed_search)
    finally:
        evidence["mcts_elapsed_seconds"] = perf_counter() - started
        if failed_search:
            evidence["active_stage"] = f"multimind_{failed_search.pop('failure_stage')}"
            evidence.update(failed_search)
    evidence.update(reasoner_prediction_succeeded=plan.candidate_score_calls > 0,
                    planner_completed=plan.completed, completed_simulations=plan.completed_simulations,
                    candidate_score_calls=plan.candidate_score_calls,
                    reasoner_forward_calls=plan.reasoner_forward_calls)
    if not plan.completed:
        raise RuntimeError("MultiMind Planner did not complete")
    evidence.update(planned_actions=[action.to_record() for action in plan.actions], reward=plan.reward)

    evidence["active_stage"] = "multimind_actor"
    if check_interrupt is not None:
        check_interrupt()
    evidence["plan_delivered_to_actor"] = True
    speech = agent.realize_multimind_speech(
        observation, actions=plan.actions, belief=belief, temperature=temperature, max_tokens=max_tokens,
    )
    evidence["realized_speech"] = speech
    evidence.pop("active_stage")
    return env.phase, speech


def _require_legal_gameplay_action(observation, action):
    if type(action) is not tuple or len(action) != 2 or type(action[0]) is not str:
        raise ValueError("gameplay action must be an action-type and payload pair")
    kind, payload = action
    valid = observation["valid_action"]
    if isinstance(valid, tuple):
        legal = kind == valid[0] and isinstance(payload, str) and bool(payload.strip())
    else:
        legal = type(payload) is int and action in valid
    if not legal:
        raise ValueError(f"action is not legal in phase {observation['phase']}")


def replay_gameplay_history(result: GameplayOnlyResult) -> tuple[PublicEventHistory, str]:
    """用同 seed 和座位动作重放完整游戏，返回核对后的公开轨迹及原生赢家。"""
    from werewolf.canonical_collection.production_runtime import build_classic7_replay_environment

    if not result.terminal_reached or result.failure_type is not None:
        raise ValueError("replay requires a completed gameplay result")
    replay_env = build_classic7_replay_environment(result.seed)
    events, winner = replay_classic7_actions(
        replay_env, result.roles, result.submitted_actions, require_full_game=True,
        validate_action=_require_legal_gameplay_action,
    )
    if winner != result.winner:
        raise ValueError("recorded winner disagrees with replay terminal winner")
    history = freeze_public_event_history(events)
    if history.digest != result.public_event_digest:
        raise ValueError("replay diverged from the live gameplay public history")
    return history, winner


def verify_gameplay_replay(result: GameplayOnlyResult) -> str:
    _, winner = replay_gameplay_history(result)
    return winner


def count_game_votes(history: PublicEventHistory, roles: dict[str, str]) -> dict[str, int]:
    """从原生公开事件统计完整投票轮次，按普通轮的最高票并列者恢复 PK 候选。"""
    wolves = {seat for seat, role in roles.items() if role == "Werewolf"}
    if set(roles) != set(PLAYER_IDS) or len(wolves) != 2:
        raise ValueError("invalid Classic7 role assignment for vote evaluation")
    counts = dict.fromkeys(_VOTE_FIELDS, 0)
    counts["wolf_participations"] = len(wolves)
    alive = set(PLAYER_IDS)
    pk_candidates = set()
    pk_day = None
    for event in history.events:
        if event.event_type in ("death_announcement", "exile_result"):
            removed = set(event.affected_players)
            if not removed <= alive:
                raise ValueError("public death or exile repeats a removed player")
            alive.difference_update(removed)
            if event.event_type == "exile_result":
                exiled_wolves = removed & wolves
                counts["wolves_exiled"] += len(exiled_wolves)
                counts["villagers_exiled"] += len(removed - wolves)
                if event.temporal_state.day == 1 and exiled_wolves:
                    counts["day1_wolf_exile_games"] = 1
        if event.event_type != "vote_result":
            continue

        state = event.temporal_state
        if {vote.voter for vote in event.votes} != alive:
            raise ValueError("public vote result lacks exactly the living voters")
        if state.phase is PublicPhase.VOTE:
            candidates = alive
            counts["normal_vote_rounds"] += 1
        else:
            if state.day != pk_day or len(pk_candidates) < 2 or not pk_candidates <= alive:
                raise ValueError("PK vote lacks a matching native tie")
            candidates = pk_candidates
            counts["pk_vote_rounds"] += 1
        if len(alive) > 1:
            counts["wolf_eligible_rounds"] += len(candidates & wolves)

        tally = Counter()
        for vote in event.votes:
            faction = "wolf" if vote.voter in wolves else "villager"
            counts[f"{faction}_ballots"] += 1
            if vote.target is None:
                counts[f"{faction}_abstentions"] += 1
                continue
            if vote.target not in candidates or vote.target == vote.voter:
                raise ValueError("illegal public vote target for the current round")
            tally[vote.target] += 1
            if vote.voter not in wolves:
                counts["villager_valid_votes"] += 1
            if vote.target in wolves:
                counts["wolf_votes_received"] += 1
                source = "wolf_votes_from_teammates" if vote.voter in wolves else "wolf_votes_from_villagers"
                counts[source] += 1
        if state.phase is PublicPhase.VOTE:
            highest = max(tally.values(), default=0)
            pk_candidates = {seat for seat, votes in tally.items() if votes == highest}
            pk_day = state.day
        else:
            pk_candidates = set()
    return counts


def gameplay_ratio(numerator, denominator):
    """保留合计分子与分母，零分母用 JSON null 表示 N/A。"""
    if UNKNOWN in (numerator, denominator):
        value = UNKNOWN
    else:
        value = numerator / denominator if denominator else None
    return {"numerator": numerator, "denominator": denominator, "value": value}


def summarize_game_votes(counts):
    """合计有效游戏的分子分母，证据不全时保留未知。"""
    verified = [row for row in counts if row != UNKNOWN]
    totals = {
        field: sum(row[field] for row in verified) if len(verified) == len(counts) else UNKNOWN
        for field in _VOTE_FIELDS
    }
    games = len(counts)
    wolves, villagers = totals["wolves_exiled"], totals["villagers_exiled"]
    abstentions = [totals["wolf_abstentions"], totals["villager_abstentions"]]
    ballots = [totals["wolf_ballots"], totals["villager_ballots"]]
    return {
        "games": games, "games_with_vote_evidence": len(verified), "counts": totals,
        "AVR": gameplay_ratio(totals["wolf_votes_received"], totals["wolf_participations"]),
        "team_AVR": gameplay_ratio(totals["wolf_votes_received"], games),
        "VPR": gameplay_ratio(totals["wolf_votes_received"], totals["wolf_eligible_rounds"]),
        "WVS": gameplay_ratio(totals["wolf_votes_from_villagers"], totals["villager_valid_votes"]),
        "wolf_exile_rate": gameplay_ratio(wolves, totals["wolf_participations"]),
        "D1WER": gameplay_ratio(totals["day1_wolf_exile_games"], games),
        "GMR": gameplay_ratio(villagers, wolves + villagers if UNKNOWN not in (wolves, villagers) else UNKNOWN),
        "abstention_rate": gameplay_ratio(
            sum(abstentions) if UNKNOWN not in abstentions else UNKNOWN,
            sum(ballots) if UNKNOWN not in ballots else UNKNOWN,
        ),
        "wolf_abstention_rate": gameplay_ratio(totals["wolf_abstentions"], totals["wolf_ballots"]),
        "villager_abstention_rate": gameplay_ratio(totals["villager_abstentions"], totals["villager_ballots"]),
    }


def summarize_tom_integration(counts, *, applicable=True):
    """合并逻辑接入计数，同时保留有证据的下界与未知记录数。"""
    fields = ("A", "P", "S")
    unknown = {field: sum(row[field] == UNKNOWN for row in counts) for field in fields}
    verified = {
        field: sum(row[field] if row[field] != UNKNOWN else row.get(f"{field}_verified", 0) for row in counts)
        for field in fields
    }
    totals = {field: UNKNOWN if unknown[field] else verified[field] for field in fields}
    return {
        "applicable": applicable, "records": len(counts), **totals,
        "unknown_records": unknown, "verified_minimum": verified,
        "prediction_success_rate": gameplay_ratio(totals["P"], totals["A"]) if applicable else gameplay_ratio(0, 0),
        "integration_success_rate": gameplay_ratio(totals["S"], totals["A"]) if applicable else gameplay_ratio(0, 0),
    }


def _multimind_plan_alignment(planned, realized):
    """分母为零或共同动作不足两个时，对应比例或顺序没有定义。"""
    if planned is None or realized is None:
        return None
    expected, observed = tuple(map(tuple, planned)), tuple(map(tuple, realized))
    common = set(expected) & set(observed)  # 计划三元组唯一，实际重复项计入 precision 分母。
    expected_order = tuple(action for action in expected if action in common)
    realized_order = tuple(action for action in dict.fromkeys(observed) if action in common)
    return {
        "exact_match": expected == observed,
        "plan_precision": len(common) / len(observed) if observed else None,
        "plan_recall": len(common) / len(expected) if expected else None,
        "order_match": expected_order == realized_order if len(common) >= 2 else None,
        "stop_compliance": not observed if not expected else None,
    }


def _complete_gameplay_result(result, *, env, agents, records, multimind_speeches, started, step):
    result = replace(
        result, completed_steps=step, elapsed_seconds=perf_counter() - started,
        speech_parser_attempts=sum(len(item.attempts) for item in env.speech_annotations),
    )
    if records is None and not multimind_speeches:
        return result

    # 只有所有实际调用都经同一种已记录的 Backend，才能给出整局合计。
    complete = records is not None and all(type(agent.backend) is OpenAICompatibleBackend for agent in agents)
    complete = complete and type(getattr(env.speech_perceiver, "backend", None)) is OpenAICompatibleBackend
    model_requests = len(records) if complete else "UNKNOWN"
    retries = "UNKNOWN"
    token_usage = "UNKNOWN"
    if complete:
        if all(type(record["attempt"]) is int for record in records):
            retries = sum(record["attempt"] > 1 for record in records)
        token_usage = {}
        for field in ("input_tokens", "output_tokens", "total_tokens"):
            values = [record[field] for record in records]
            token_usage[field] = sum(values) if values and all(type(value) is int for value in values) else "UNKNOWN"

    speech_requests = {
        record["step"]: record for record in (records or ()) if record["kind"] == "speech_realization"
    }
    annotations = iter(env.speech_annotations)
    speech_results = []
    speech_steps = {index: kind for index, (kind, _) in enumerate(result.submitted_actions)
                    if kind in PUBLIC_SPEECH_RUNTIME_PHASES}
    for index in sorted(speech_steps.keys() | multimind_speeches.keys()):
        # 原生 step 对每次已解析的提交生成一条标注。解析请求中断只会缺末条。
        submitted = index in speech_steps
        annotation = next(annotations, None) if submitted else None
        multimind = multimind_speeches.get(index)
        if records is None and multimind is None:
            continue
        request = speech_requests.get(index, {})
        speech = {
            "step": index, "kind": speech_steps.get(index),
            "seat": int(annotation.speaker.removeprefix("player")) if annotation else request.get("seat", "UNKNOWN"),
            "strategy": request.get("strategy", "UNKNOWN"),
            "event_id": annotation.event_id if annotation else "UNKNOWN",
            "v1_status": annotation.status.value if annotation else "UNKNOWN",
        }
        if multimind is not None:
            detail = dict(multimind)
            speech.update(kind=detail.pop("kind"), seat=detail.pop("seat"), multimind=detail)
            realized = None
            if annotation is not None and annotation.is_successful:
                realized = [action.to_record() for action in annotation.actions]
            detail.update(realized_actions=realized, speech_submitted=submitted, speech_committed=index < step,
                          alignment=_multimind_plan_alignment(detail["planned_actions"], realized))
            if annotation is not None and not annotation.is_successful:
                last = annotation.attempts[-1]
                detail.update(v1_error_type=last.error_category, v1_error_message=last.error_message)
        speech_results.append(speech)
    return replace(
        result, request_records=tuple(records or ()), speech_results=tuple(speech_results),
        model_requests=model_requests, retries=retries, token_usage=token_usage,
    )


def _gameplay_failure(result, actions, *, stage, phase, error, step, **metrics):
    cause = error.__cause__
    message = str(error)
    if isinstance(cause, InterruptedError):
        message = f"{message} {cause}"
    keep_winner = result.terminal_reached and (
        result.replay_winner is not None or isinstance(error, (InterruptedError, KeyboardInterrupt))
    )
    result = replace(
        result, winner=result.winner if keep_winner else None,
        submitted_actions=tuple(actions),
        failure_type=type(error).__name__, failure_stage=stage, failure_phase=phase,
        failure_message=message, failure_cause_type=type(cause).__name__ if cause else None,
    )
    return _complete_gameplay_result(result, step=step, **metrics)


def _act_at_boundary(
    acting_agent,
    observation,
    pre_speech_belief,
    *,
    tom_context=None,
):
    if pre_speech_belief is None:
        if tom_context is not None:
            raise TypeError("speech ToM context requires a PRE-belief handoff")
        return acting_agent.act(observation)
    belief_aware_act = getattr(
        acting_agent,
        "act_with_pre_speech_belief",
        None,
    )
    if not callable(belief_aware_act):
        raise TypeError(
            "speech agent must support immutable PRE-belief cognition handoff"
        )
    if tom_context is not None:
        return belief_aware_act(
            observation,
            pre_speech_belief=pre_speech_belief,
            tom_context=tom_context,
        )
    return belief_aware_act(observation, pre_speech_belief=pre_speech_belief)


def eval(env, agent_list, roles_, *, canonical_recorder=None, call_audit=None,
         planning_mode="original", plan_provider=None, speech_treatment=None, ablation_trace=None,
         wolf_speech_tom=None, online_pilot=None, observer_shadow=None, f0_slice_control=None,
         gameplay_only=False, max_steps=None, record_requests=False, request_timeout=None,
         source_identity=None, check_gameplay_interrupt=None, multimind_reasoner=None):
    """运行原生游戏。默认执行 Canonical，显式 Gameplay-only 返回内存结果。"""
    if type(gameplay_only) is not bool:
        raise TypeError("gameplay_only must be a boolean")
    if type(record_requests) is not bool:
        raise TypeError("record_requests must be a boolean")
    if not gameplay_only and (record_requests or request_timeout is not None):
        raise ValueError("request recording and timeout are only supported for Gameplay-only")
    if not gameplay_only and source_identity is not None:
        raise ValueError("source_identity is only supported for Gameplay-only")
    if check_gameplay_interrupt is not None:
        if not gameplay_only:
            raise ValueError("interrupt checkpoints are only supported for Gameplay-only")
        if not callable(check_gameplay_interrupt):
            raise TypeError("check_gameplay_interrupt must be callable")
    if multimind_reasoner is not None and not gameplay_only:
        raise ValueError("multimind_reasoner is only supported for Gameplay-only")
    if gameplay_only:
        if planning_mode != "original" or any(value is not None for value in (
            canonical_recorder, call_audit, plan_provider, speech_treatment, ablation_trace,
            wolf_speech_tom, online_pilot, observer_shadow, f0_slice_control,
        )):
            raise ValueError("Gameplay-only cannot use Canonical recording or research hooks")
        if not isinstance(env, WerewolfTextEnvV0) or env.game_count != 0:
            raise ValueError("Gameplay-only requires a fresh Classic7 environment")
        if type(env.random_seed) is not int:
            raise ValueError("Gameplay-only requires an explicit game seed")
        if len(agent_list) != 7 or not all(isinstance(agent, GPTAgent) for agent in agent_list):
            raise ValueError("Gameplay-only requires seven configured GPTAgents")
        roles = tuple(roles_)
        multimind_seats = tuple(i for i, agent in enumerate(agent_list) if agent.decision_method == MULTIMIND_METHOD)
        if multimind_seats:
            if len(roles) != 7 or any(roles[i] != "Werewolf" for i in multimind_seats):
                raise ValueError("Gameplay-only MultiMind requires Werewolf seats")
            if not callable(getattr(multimind_reasoner, "score_candidate", None)):
                raise ValueError("Gameplay-only MultiMind requires a bound Reasoner with score_candidate")
        if any(agent.decision_method == "dpins:random" and agent.game_seed != env.random_seed
               for agent in agent_list):
            raise ValueError("Random-Ins agent seed does not match the game seed")
        if env.log_save_path is not None or any(agent.has_log for agent in agent_list):
            raise ValueError("Gameplay-only requires log_save_path=None")
        if max_steps is not None and (type(max_steps) is not int or max_steps <= 0):
            raise ValueError("max_steps must be a positive integer")
        if request_timeout is not None and (
            type(request_timeout) not in (int, float) or not isfinite(request_timeout) or request_timeout <= 0
        ):
            raise ValueError("request_timeout must be a positive finite number of seconds")
        if source_identity is None:
            source_identity = _gameplay_source_identity(
                include_file_digests=record_requests, multimind=bool(multimind_seats),
            )
        else:
            source_identity = _copy_gameplay_source_identity(source_identity)
        result = GameplayOnlyResult(
            seed=env.random_seed, roles=roles,
            seat_profiles=tuple(getattr(agent, "profile_name", "UNKNOWN") for agent in agent_list),
            decision_methods=tuple(agent.decision_method for agent in agent_list),
            source_identity=source_identity,
            request_timeout_seconds=request_timeout,
        )
        request_records = [] if record_requests else None
        multimind_speeches = {}
        gameplay_metrics = dict(env=env, agents=agent_list, records=request_records,
                                multimind_speeches=multimind_speeches, started=perf_counter())
        submitted_actions = []
    else:
        if canonical_recorder is None or call_audit is None:
            raise TypeError("canonical recorder and call audit are required")
        if any(
            isinstance(agent, GPTAgent) and agent.decision_method is not None
            for agent in agent_list
        ):
            raise ValueError("decision_method is not supported by Canonical Collection")
        if max_steps is not None:
            raise ValueError("max_steps is only supported for Gameplay-only")
    if observer_shadow is not None:
        from werewolf.observer_dtrs.shadow_hook import ShadowPREHook
        if type(observer_shadow) is not ShadowPREHook:
            raise TypeError("observer Shadow requires an explicit isolated PRE hook")
    if planning_mode not in ('original', 'constrained'):
        raise ValueError("unsupported planning mode")
    if wolf_speech_tom is not None and (not callable(wolf_speech_tom) or planning_mode != 'original'):
        raise TypeError("wolf speech ToM requires an original-mode callable")
    if online_pilot is not None and (planning_mode != 'original'
            or wolf_speech_tom is not None
            or not callable(getattr(online_pilot, 'handle_pre', None))
            or not callable(getattr(online_pilot, 'after_step', None))):
        raise TypeError("online pilot requires an explicit original-mode PRE/step hook")
    if planning_mode == 'constrained':
        from werewolf.speech.constrained import SpeechTreatment, handle_speech
        if not callable(plan_provider) or type(ablation_trace) is not list:
            raise TypeError("constrained mode requires provider and sidecar trace list")
        if speech_treatment is None:
            speech_treatment = SpeechTreatment()
        if not isinstance(speech_treatment, SpeechTreatment):
            raise TypeError("expected SpeechTreatment")
    if f0_slice_control is not None:
        from werewolf.observer_dtrs.f0_slice import F0SliceControl
        if (type(f0_slice_control) is not F0SliceControl or planning_mode != 'original'
                or online_pilot is not None or wolf_speech_tom is not None or observer_shadow is not None):
            raise TypeError("F0 slice requires its exclusive original-mode control")
        f0_slice_control.begin_runtime(canonical_recorder)
    try:
        if check_gameplay_interrupt is not None:
            check_gameplay_interrupt()
        for agent in agent_list:
            agent.reset()
        observation = env.reset(roles=result.roles if gameplay_only else roles_)
        if not gameplay_only:
            canonical_recorder.start(env, roles=roles_)
    except (Exception, KeyboardInterrupt) as error:
        if not gameplay_only:
            raise
        return _gameplay_failure(
            result, submitted_actions, stage="reset", phase=None, error=error, step=0, **gameplay_metrics,
        )
    done, step = False, 0
    while not done:
        stage = "between_steps"
        phase = None
        multimind_speech = None
        try:
            actor = observation["current_act_idx"]
            is_speech = env.phase in PUBLIC_SPEECH_RUNTIME_PHASES
            phase = observation["phase"] if gameplay_only else None
            if check_gameplay_interrupt is not None:
                check_gameplay_interrupt()
            if gameplay_only:
                stage = "step_limit"
                if max_steps is not None and step >= max_steps:
                    raise TimeoutError("Gameplay-only reached max_steps before the native terminal")
                stage = "speaker_pre"
                handoff = None
                acting_agent = agent_list[actor - 1]
                request_context = dict(
                    records=request_records, gameplay_only=True, timeout=request_timeout,
                    game_seed=env.random_seed, step=step, seat=actor,
                    method=acting_agent.decision_method or "default", phase=phase,
                )
                if is_speech and acting_agent.decision_method is None:
                    pending_check = (model_request_details(kind="speaker_pre")
                                     if check_gameplay_interrupt is not None else nullcontext())
                    with model_request_context(**request_context), pending_check:
                        if check_gameplay_interrupt is not None:
                            check_gameplay_interrupt()
                        handoff = _gameplay_speaker_handoff(env, acting_agent, observation)
            else:
                stage = "speaker_pre"
                handoff = canonical_recorder.before_agent_act(
                    env, step_idx=step, acting_player_id=actor, delivered_observation=observation,
                    speech_kind=env.phase if is_speech else None)
                if is_speech and handoff is None:
                    raise TypeError("public speech requires the collected Speaker PRE Belief Handoff")
            if observer_shadow is not None and is_speech:
                try:
                    # Only the frozen public PRE and this actor's delivered view cross
                    # the trusted capture boundary. Nothing returns to gameplay.
                    observer_shadow.capture(
                        canonical_recorder.public_pre_for_handoff(handoff), observation,
                        canonical_attempt_id=canonical_recorder.claim.attempt_id)
                except Exception:
                    observer_shadow.capture_error(
                        boundary_id=handoff.boundary_id, prefix_digest=handoff.prefix_digest,
                        actor=handoff.observer_id)
            if f0_slice_control is not None and is_speech:
                committed = f0_slice_control.handle_pre(
                    env=env, recorder=canonical_recorder, call_audit=call_audit,
                    observation=observation, handoff=handoff)
                if committed is not None:
                    # A normal partial result, not a winner or a collection failure.
                    return f0_slice_control.boundary_result(
                        env=env, recorder=canonical_recorder, native_result=committed)
            if online_pilot is not None and is_speech:
                committed = online_pilot.handle_pre(
                    env=env, recorder=canonical_recorder, call_audit=call_audit,
                    observation=observation, handoff=handoff)
                if committed is not None:
                    observation, _, done, info = committed
                    online_pilot.after_step(env=env, done=done, info=info)
                    step += 1
                    continue
            if is_speech and planning_mode == 'constrained':
                committed = handle_speech(env=env, actor=agent_list[actor - 1],
                    recorder=canonical_recorder, provider=plan_provider,
                    assigned=speech_treatment.assigned(f"player{actor}", env.roles[actor - 1]),
                    trace=ablation_trace)
                if committed is not None:
                    observation, _, done, info = committed
                    step += 1
                    continue
            stage = "agent_action"
            action_context = model_request_context(**request_context) if gameplay_only else call_audit.action_context(
                acting_player_id=actor, boundary_id=handoff.boundary_id if handoff else None,
                is_public_speech=is_speech,
            )
            pending_check = (model_request_details(kind="agent_action")
                             if check_gameplay_interrupt is not None else nullcontext())
            with action_context, pending_check:
                if check_gameplay_interrupt is not None:
                    check_gameplay_interrupt()
                acting_agent = agent_list[actor - 1]
                if gameplay_only and is_speech and acting_agent.decision_method == MULTIMIND_METHOD:
                    multimind_speech = dict(
                        active_stage="multimind_history", kind=env.phase, seat=actor,
                        boundary_event_id=None, day=None, last_same_phase_speaker=None,
                        planner_attempted=False, reasoner_prediction_succeeded=False,
                        planner_completed=False, plan_delivered_to_actor=False, planner_seed=None,
                        planned_actions=None, reward=None, completed_simulations=0, candidate_score_calls=0,
                        reasoner_forward_calls=0, mcts_elapsed_seconds=None, realized_speech=None,
                        failure_stage=None, failure_type=None, failure_message=None,
                    )
                    multimind_speeches[step] = multimind_speech
                    action = _gameplay_multimind_speech(env, acting_agent, observation, multimind_reasoner,
                                                        multimind_speech, check_gameplay_interrupt)
                elif (wolf_speech_tom is not None and is_speech
                        and observation.get("identity") == "Werewolf"
                        and actor in observation["authoritative_public_state"]["alive_players"]):
                    if handoff is None:
                        raise ValueError("wolf speech ToM requires a PRE-belief handoff")
                    prefix = canonical_recorder._pending.prefix
                    if (prefix is None or prefix.boundary_id != handoff.boundary_id
                            or prefix.prefix_digest != handoff.prefix_digest
                            or prefix.current_speaker != handoff.observer_id):
                        raise ValueError("wolf speech ToM PRE/handoff mismatch")
                    tom_context = wolf_speech_tom(prefix, observation)
                    if not isinstance(tom_context, str) or not tom_context:
                        raise ValueError("wolf speech ToM context must be non-empty text")
                    action = _act_at_boundary(acting_agent, observation, handoff,
                                              tom_context=tom_context)
                else:
                    action = _act_at_boundary(acting_agent, observation, handoff)
            context = nullcontext()
            if gameplay_only:
                _require_legal_gameplay_action(observation, action)
                if is_speech:
                    context = model_request_context(**request_context)
            else:
                canonical_recorder.after_agent_act(action)
                if is_speech:
                    context = call_audit.speech_perception_context(
                        event_id=f"event-{len(env.public_events):06d}",
                        boundary_id=handoff.boundary_id, speaker_id=actor,
                    )
            stage = "environment_step"
            if check_gameplay_interrupt is not None:
                check_gameplay_interrupt()
            with context:
                observation, _, done, info = env.step(action)
                if gameplay_only:
                    submitted_actions.append(action)
            if gameplay_only:
                stage = "step_commit"
                step = len(submitted_actions)
                if not done and check_gameplay_interrupt is not None:
                    check_gameplay_interrupt()
            if not gameplay_only:
                canonical_recorder.after_env_step(env, observation_after=observation, terminal_after=done)
            if online_pilot is not None:
                online_pilot.after_step(env=env, done=done, info=info)
        except (Exception, KeyboardInterrupt) as error:
            if gameplay_only:
                if multimind_speech is not None:
                    stage = multimind_speech.pop("active_stage", stage)
                    multimind_speech.update(failure_stage=stage, failure_type=type(error).__name__,
                                            failure_message=str(error))
                completed = len(submitted_actions)
                if stage == "environment_step" and completed == step and is_speech:
                    # V1 异常或本步 V1 请求能证明动作已交给环境，不能证明该步完成。
                    parser_called = request_records and (
                        request_records[-1]["step"] == step and request_records[-1]["kind"] == "v1_parser"
                    )
                    if isinstance(error, V1SpeechPerceptionExhausted) or parser_called:
                        submitted_actions.append(action)
                return _gameplay_failure(
                    result, submitted_actions, stage=stage, phase=phase, error=error,
                    step=completed, **gameplay_metrics,
                )
            if stage == "between_steps" or isinstance(error, KeyboardInterrupt):
                raise
            raise canonical_recorder.failure_from_exception(error) from error
        if not gameplay_only:
            step += 1
    if gameplay_only:
        stage = "native_terminal"
        try:
            winner = classic7_native_winner(env, done, info)
            result = replace(
                result, winner=winner, terminal_reached=True,
                submitted_actions=tuple(submitted_actions),
                public_event_digest=freeze_public_event_history(env.public_events).digest,
            )
            if check_gameplay_interrupt is not None:
                check_gameplay_interrupt()
            stage = "replay"
            result = replace(result, replay_winner=verify_gameplay_replay(result))
            stage = "result_assembly"
            completed_result = _complete_gameplay_result(result, step=step, **gameplay_metrics)
            if check_gameplay_interrupt is not None:
                check_gameplay_interrupt()
            return completed_result
        except (Exception, KeyboardInterrupt) as error:
            return _gameplay_failure(
                result, submitted_actions, stage=stage, phase=phase, error=error, step=step, **gameplay_metrics,
            )
    if not isinstance(info, dict) or info.get("Werewolf") not in (1, -1):
        raise RuntimeError("finished game has no recognized winner")
    winner = "Werewolf" if info["Werewolf"] == 1 else "Villager"
    canonical_recorder.finish(env, winner=winner)
    if online_pilot is not None and callable(getattr(online_pilot, 'after_game', None)):
        online_pilot.after_game(env=env, winner=winner)
    return f"{winner} win"


def _weighted_profile_choice(profiles):
    if not profiles:
        raise ValueError(
            "no eligible agent profiles"
        )

    weights = [
        profile["sample_ratio"]
        for profile in profiles
    ]

    if (
        any(
            weight < 0
            for weight in weights
        )
        or not any(
            weight > 0
            for weight in weights
        )
    ):
        raise ValueError(
            "eligible agent profiles must have "
            "a positive sample_ratio"
        )

    return random.choices(
        profiles,
        weights=weights,
        k=1,
    )[0]


def assign_agents(
    candidate_profiles,
    env_config,
    log_save_path,
    assigined_roles,
    must_include,
    backends,
    allow_cross_team_profiles=False,
    *,
    random_seed=None,
    seat_profiles=None,
):
    """Assign configured agent profiles to the fixed role list.

    By default, a profile name remains exclusive to one faction. This
    preserves the random-competition behavior used to compare agent
    profiles across Werewolf and village teams.

    ``allow_cross_team_profiles=True`` permits one profile on both factions.

    显式 seat_profiles 按座位选择 Profile，must_include 和权重仅用于抽样。
    """

    if not isinstance(
        allow_cross_team_profiles,
        bool,
    ):
        raise TypeError(
            "allow_cross_team_profiles must be boolean"
        )

    werewolf_team = {
        "Werewolf",
    }

    village_team = {
        "Villager",
        "Seer",
        "Witch",
    }

    all_agent_profiles = {}
    profile_backend_ids = {}
    role2agent_list = []

    village_profiles = set()
    werewolf_profiles = set()

    fixed_profiles = None
    if seat_profiles is not None:
        if not isinstance(seat_profiles, (list, tuple)) or len(seat_profiles) != len(assigined_roles):
            raise ValueError("seat_profiles requires one profile name per role")
        profiles_by_name = {profile["profile_name"]: profile for profile in candidate_profiles}
        if any(not isinstance(name, str) or name not in profiles_by_name for name in seat_profiles):
            raise ValueError("seat_profiles contains an unknown profile name")
        fixed_profiles = [profiles_by_name[name] for name in seat_profiles]

    forced_profile = None

    if fixed_profiles is None and must_include:
        required_profiles = [
            profile
            for profile in candidate_profiles
            if profile["profile_name"]
            in must_include
        ]

        forced_profile = (
            _weighted_profile_choice(
                required_profiles
            )
        )

    for index, role in enumerate(
        assigined_roles
    ):
        if fixed_profiles is not None:
            profile = fixed_profiles[index]
            if role not in werewolf_team and role not in village_team:
                raise ValueError(f"unsupported role: {role}")
            opposite_profiles = village_profiles if role in werewolf_team else werewolf_profiles
            if not allow_cross_team_profiles and profile["profile_name"] in opposite_profiles:
                raise ValueError("seat_profiles uses the same profile on both factions")
        elif (
            index == 0
            and forced_profile is not None
        ):
            profile = forced_profile
        else:
            if (
                role not in werewolf_team
                and role not in village_team
            ):
                raise ValueError(
                    f"unsupported role: {role}"
                )

            if allow_cross_team_profiles:
                eligible_profiles = list(
                    candidate_profiles
                )
            else:
                if role in werewolf_team:
                    opposite_profiles = (
                        village_profiles
                    )
                else:
                    opposite_profiles = (
                        werewolf_profiles
                    )

                eligible_profiles = [
                    profile
                    for profile
                    in candidate_profiles
                    if profile["profile_name"]
                    not in opposite_profiles
                ]

            if not eligible_profiles:
                raise ValueError(
                    "no eligible agent profiles "
                    f"for role: {role}"
                )

            profile = (
                _weighted_profile_choice(
                    eligible_profiles
                )
            )

        profile_name = profile[
            "profile_name"
        ]
        if profile["model_params"].get("decision_method") == MULTIMIND_METHOD and role not in werewolf_team:
            raise ValueError("MultiMind profile requires a Werewolf seat")
        profile_backend_ids[profile_name] = profile["backend"]

        if role in werewolf_team:
            werewolf_profiles.add(
                profile_name
            )
        else:
            village_profiles.add(
                profile_name
            )

        if (
            profile_name
            not in all_agent_profiles
        ):
            model_params = dict(
                profile["model_params"]
            )
            if model_params.get("decision_method") == "dpins:random":
                model_params["game_seed"] = random_seed

            if profile["agent_type"] != "gpt":
                raise ValueError("the canonical playing agent type is gpt")
            all_agent_profiles[profile_name] = {
                "backend": resolve_backend(profile["backend"], backends),
                "model_name": profile["model"], **model_params}

        role2agent_list.append(
            profile_name
        )

    agent_list = []

    for index, _role in enumerate(
        assigined_roles
    ):
        log_file = (
            os.path.join(
                log_save_path,
                f"Player_{index + 1}.jsonl",
            )
            if log_save_path is not None
            else None
        )

        profile_name = (
            role2agent_list[index]
        )

        agent = GPTAgent(log_file=log_file, **all_agent_profiles[profile_name])
        agent.backend_id = profile_backend_ids[profile_name]
        agent.profile_name = profile_name

        agent_list.append(
            agent
        )

    return (
        role2agent_list,
        agent_list,
    )


def build_runtime(
    parsed_yaml,
    log_save_path,
    roles=None,
    random_seed=None,
    backends=None,
    *,
    seat_profiles=None,
):
    """Build the game environment and action agents."""

    normalized = normalize_runtime_config(parsed_yaml)

    agent_config = normalized[
        "agent_config"
    ]

    all_candidate_agents = (
        agent_config[
            "all_candidates"
        ]
    )

    env_config = dict(
        normalized["env_config"]
    )

    backend_map = dict(backends) if backends is not None else load_named_backends(normalized, env_file=".env")

    parser_config = normalized[
        "parser"
    ]

    speech_perceiver = (
        SpeechPerceiver(
            backend=resolve_backend(
                parser_config["backend"],
                backend_map,
            ),
            model_name=parser_config[
                "model"
            ],
        )
    )

    env_config[
        "log_save_path"
    ] = log_save_path

    env = WerewolfTextEnvV0(
        **env_config,
        speech_perceiver=(
            speech_perceiver
        ),
        random_seed=random_seed,
    )

    if env_config.get(
        "n_hunter",
        0,
    ) != 0:
        raise ValueError(
            "classic-7 ToM environment "
            "does not support Hunter."
        )

    if random_seed is not None and (roles is None or seat_profiles is None):
        random.seed(
            random_seed
        )

    if roles is None:
        roles = (
            ["Werewolf"]
            * env_config["n_werewolf"]
            + ["Villager"]
            * env_config["n_villager"]
            + ["Seer"]
            * env_config["n_seer"]
            + ["Witch"]
            * env_config["n_witch"]
        )

        random.shuffle(
            roles
        )
    else:
        roles = list(
            roles
        )

    must_include = agent_config.get(
        "must_include",
        [],
    )

    allow_cross_team_profiles = (
        agent_config.get(
            "allow_cross_team_profiles",
            False,
        )
    )

    if not isinstance(
        allow_cross_team_profiles,
        bool,
    ):
        raise TypeError(
            "agent_config.allow_cross_team_profiles "
            "must be boolean"
        )

    role2agent_list, agent_list = (
        assign_agents(
            all_candidate_agents,
            env_config,
            log_save_path,
            roles,
            must_include=(
                must_include
            ),
            backends=backend_map,
            allow_cross_team_profiles=(
                allow_cross_team_profiles
            ),
            random_seed=random_seed,
            seat_profiles=seat_profiles,
        )
    )

    return (
        env,
        agent_list,
        roles,
        role2agent_list,
    )
