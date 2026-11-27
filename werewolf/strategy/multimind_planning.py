"""MultiMind-Text 的单次发言 MCTS。"""

from __future__ import annotations

from collections.abc import Collection
from contextlib import nullcontext
from dataclasses import dataclass, field
from math import log, sqrt
from random import Random
from time import perf_counter
from typing import Protocol

import torch

from werewolf.canonical_collection.public_history import PLAYER_IDS
from werewolf.canonical_collection.speech import V1SpeechAction
from werewolf.strategy.speech_planning import PlanningAction, _ordered_candidates
from werewolf.tom.multimind import MultiMindHistory, MultiMindReasoner


DEFAULT_SIMULATIONS = 500
MAX_ACTIONS = 3
UCT_C = 1.414
_STOP = None
_ROLE_CLAIMS = ("point_as_werewolf", "point_as_villager", "point_as_seer", "point_as_witch")
_STANCE = {PlanningAction.SUPPORT: "support", PlanningAction.OPPOSE: "oppose"}
_Speech = tuple[V1SpeechAction, ...]


class _CandidateScorer(Protocol):
    def score_candidate(self, history: MultiMindHistory, speaker: str, actions: _Speech) -> torch.Tensor:
        ...


@dataclass(frozen=True)
class MultiMindPlan:
    actions: _Speech
    reward: float
    completed_simulations: int
    candidate_score_calls: int
    reasoner_forward_calls: int | None
    elapsed_seconds: float
    completed: bool
    terminal_evaluations: tuple[tuple[_Speech, float], ...]


@dataclass
class _Node:
    actions: _Speech = ()
    stopped: bool = False
    visits: int = 0
    total_reward: float = 0.0
    children: dict[V1SpeechAction | None, _Node] = field(default_factory=dict)


def _public_actions(history, speaker, alive):
    day = history.tokens[-1].day
    spoken_today = {token.subject for token in history.tokens
                    if token.predicate == "SPEECH_END" and token.day == day}
    plans = _ordered_candidates(speaker, alive, spoken_today)
    # 与原公开规划共用存活、非自身目标；支持与反对还须有当天发言。
    targets = tuple(plan.target for plan in plans if plan.action is PlanningAction.ACCUSE_WOLF)
    claims = tuple(V1SpeechAction(speaker, predicate, target)
                   for predicate in _ROLE_CLAIMS for target in targets)
    stances = tuple(V1SpeechAction(speaker, _STANCE[plan.action], plan.target)
                    for plan in plans if plan.action in _STANCE)
    return claims + stances


def _legal_extensions(actions, stopped, candidates):
    if stopped or len(actions) == MAX_ACTIONS:
        return ()
    # V1 去除完全相同的三元组，保留不同（包括互相矛盾）的命题及其顺序。
    return (_STOP, *(candidate for candidate in candidates if candidate not in actions))


def _uct(child, parent_visits):
    if child.visits == 0:
        return float("inf")
    return child.total_reward / child.visits + UCT_C * sqrt(log(max(1, parent_visits)) / child.visits)


def _select_and_expand(root, candidates, rng):
    node = root
    path = [root]
    while options := _legal_extensions(node.actions, node.stopped, candidates):
        unexpanded = [action for action in options if action not in node.children]
        if unexpanded:
            action = rng.choice(unexpanded)
            stopped = action is _STOP
            child = _Node(node.actions if stopped else (*node.actions, action), stopped=stopped)
            node.children[action] = child
            path.append(child)
            break
        scores = [(_uct(child, node.visits), child) for child in node.children.values()]
        highest = max(score for score, _ in scores)
        node = rng.choice([child for score, child in scores if score == highest])
        path.append(node)
    return path


def _rollout(node, candidates, rng):
    actions = node.actions
    while options := _legal_extensions(actions, node.stopped, candidates):
        action = rng.choice(options)
        if action is _STOP:
            break
        actions = (*actions, action)
    return actions


def _backpropagate(path, reward):
    for node in reversed(path):
        node.visits += 1
        node.total_reward += reward


def _terminal_reward(probabilities, observer_rows, speaker_column):
    shape = (len(PLAYER_IDS), len(PLAYER_IDS))
    if (not isinstance(probabilities, torch.Tensor) or probabilities.shape != shape
            or not probabilities.is_floating_point()):
        raise ValueError("MultiMind scorer must return a floating [7, 7] probability matrix")
    mass = probabilities[observer_rows, speaker_column]
    if not (torch.isfinite(mass) & (mass >= 0) & (mass <= 1)).all():
        raise ValueError("MultiMind scorer returned invalid suspicion mass for the reward")
    return -mass.sum(dtype=torch.float64).item()


@torch.inference_mode()
def plan_speech(
    history: MultiMindHistory,
    speaker: str,
    *,
    alive_players: Collection[str],
    known_wolves: Collection[str],
    reasoner: _CandidateScorer,
    seed: int,
    budget: int = DEFAULT_SIMULATIONS,
    failure_evidence: dict | None = None,
) -> MultiMindPlan:
    """搜索一次发言，真实 Reasoner 由调用者设为 eval，搜索异常可保留已观测计数。"""
    if type(budget) is not int or budget < 1 or type(seed) is not int:
        raise ValueError("MultiMind requires a positive simulation budget and an integer seed")
    if not isinstance(history, MultiMindHistory) or not history.tokens:
        raise ValueError("MultiMind requires a frozen public speech history")
    turn = history.tokens[-1]
    if (turn.predicate != "turn_start" or turn.subject != speaker
            or turn.phase not in ("discussion", "pk_discussion")):
        raise ValueError("MultiMind speaker must match the current public speech turn")
    alive = frozenset(alive_players)
    eliminated = {token.target for token in history.tokens
                  if token.predicate in ("exiled_player", "dead_player")}
    if alive != set(PLAYER_IDS) - eliminated or speaker not in alive:
        raise ValueError("MultiMind alive players must match public eliminations and include the speaker")
    wolves = frozenset(known_wolves)
    if speaker not in wolves or len(wolves) > 2 or not wolves <= set(PLAYER_IDS):
        raise ValueError("MultiMind known wolves must be Classic7 seats including the acting wolf")
    observer_rows = [index for index, player in enumerate(PLAYER_IDS) if player in alive - wolves]
    speaker_column = PLAYER_IDS.index(speaker)
    candidates = _public_actions(history, speaker, alive)

    root = _Node()
    rng = Random(seed)
    evaluations = []
    forward_calls = 0
    score_calls = 0
    scoring = False

    def count_forward(*_args):
        nonlocal forward_calls
        forward_calls += 1

    observable = isinstance(reasoner, MultiMindReasoner)
    forward_hook = reasoner.register_forward_pre_hook(count_forward) if observable else nullcontext()
    started = perf_counter()
    try:
        with forward_hook:
            for _ in range(budget):
                path = _select_and_expand(root, candidates, rng)
                actions = _rollout(path[-1], candidates, rng)
                scoring = True
                score_calls += 1
                probabilities = reasoner.score_candidate(history, speaker, actions)
                reward = _terminal_reward(probabilities, observer_rows, speaker_column)
                scoring = False
                evaluations.append((actions, reward))
                _backpropagate(path, reward)
    except (Exception, KeyboardInterrupt):
        if failure_evidence is not None:
            failure_evidence.update(
                failure_stage="reasoner" if scoring else "planner",
                reasoner_prediction_succeeded=bool(evaluations), completed_simulations=root.visits,
                candidate_score_calls=score_calls, reasoner_forward_calls=forward_calls if observable else None,
            )
        raise

    # max 保留首次达到最高奖励的完整终止序列。
    actions, reward = max(evaluations, key=lambda evaluation: evaluation[1])
    return MultiMindPlan(
        actions=actions, reward=reward, completed_simulations=root.visits,
        candidate_score_calls=score_calls, reasoner_forward_calls=forward_calls if observable else None,
        elapsed_seconds=perf_counter() - started, completed=True, terminal_evaluations=tuple(evaluations),
    )
