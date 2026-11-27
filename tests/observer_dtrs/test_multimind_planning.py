"""以冻结公共轨迹、受控矩阵与少量 CPU 前向验证 MultiMind 搜索。"""

from collections import defaultdict
from math import log, sqrt
import random

import pytest
import torch

from tests.observer_dtrs.test_mcts_planning import ManualWitness, P
from werewolf.canonical_collection import (
    V1_SPEECH_PARSER_VERSION, V1_SPEECH_PROMPT_VERSION, V1AnnotationStatus,
    V1PerceptionAttempt, V1SpeechAction, construct_v1_speech_annotation,
    freeze_public_event_history,
)
from werewolf.strategy import multimind_planning as planner
from werewolf.tom.dataset import ExperimentCapacity
from werewolf.tom.multimind import MultiMindReasoner, encode_candidate, encode_history


ROLE_CLAIMS = ("point_as_werewolf", "point_as_villager", "point_as_seer", "point_as_witch")
DEAD = (P[6],)
ALIVE = P[:-1]
TEAM = P[:2]


def _history(witness):
    prefix = witness.pre()
    return encode_history(prefix.public_event_history, prefix.v1_annotations)


def _matrix(masses):
    """第一列单独指定嫌疑，其他非自身座位分配剩余质量。"""
    matrix = torch.zeros((7, 7), dtype=torch.float64)
    matrix[0, 1:] = 1 / 6
    for row, mass in enumerate(masses, start=1):
        matrix[row, [col for col in range(1, 7) if col != row]] = (1 - mass) / 5
        matrix[row, 0] = mass
    return matrix


class ControlledScorer:
    def __init__(self, probabilities):
        self.probabilities = probabilities
        self.calls = []

    def score_candidate(self, history, speaker, actions):
        self.calls.append((history, speaker, actions))
        return self.probabilities(actions)


def _plan(history, reasoner, **overrides):
    options = dict(alive_players=ALIVE, known_wolves=TEAM, reasoner=reasoner, seed=17)
    options.update(overrides)
    return planner.plan_speech(history, history.tokens[-1].subject, **options)


@pytest.mark.parametrize("phase", ["discussion", "pk_discussion"])
@pytest.mark.parametrize("last", [False, True])
def test_public_candidates_and_same_scoring_path_including_last_speaker(phase, last):
    history = _history(ManualWitness(phase, dead=DEAD, last=last))
    expected = {V1SpeechAction(P[0], predicate, target) for predicate in ROLE_CLAIMS for target in ALIVE[1:]}
    if phase == "pk_discussion" or last:
        expected.update(V1SpeechAction(P[0], predicate, target)
                        for predicate in ("support", "oppose") for target in ALIVE[1:])
    assert set(planner._public_actions(history, P[0], ALIVE)) == expected

    matrix = _matrix([0.25] * 6)
    scorer = ControlledScorer(lambda _actions: matrix)
    result = _plan(history, scorer, budget=60)
    assert result.completed
    assert len(scorer.calls) == 60
    for prefix, speaker, actions in scorer.calls:
        assert prefix is history and speaker == P[0] and isinstance(actions, tuple)
        assert len(set(actions)) == len(actions) <= 3 and set(actions) <= expected
    assert any(actions == () for _, _, actions in scorer.calls)
    assert any(len(actions) == 3 for _, _, actions in scorer.calls)


def test_stance_uses_only_prior_completed_speech_that_day_even_if_empty():
    witness = ManualWitness(dead=DEAD, last=True)
    witness.speech(P[0], "no_commitment", None, "上一日公开发言。")
    witness.event("phase_change", day=1, phase="vote")
    witness.event("vote_result", votes=[{"voter": p, "target": None} for p in ALIVE])
    witness.event("exile_result", exiled_players=[])
    witness.event("phase_change", day=1, phase="night")
    witness.event("death_announcement", dead_players=[])
    witness.event("phase_change", day=2, phase="discussion")
    witness.event("turn_start", speaker=P[1])
    witness.event("public_speech", speaker=P[1], raw_text="今天的无语义动作公开发言。")
    attempt = V1PerceptionAttempt(
        attempt_index=1, call_id="manual-empty", backend_id="fixture", model_id="fixture",
        prompt_version=V1_SPEECH_PROMPT_VERSION, parser_version=V1_SPEECH_PARSER_VERSION,
        status=V1AnnotationStatus.NO_ACTION, raw_response="NONE", error_category=None, error_message=None,
    )
    witness.annotations.append(construct_v1_speech_annotation(
        freeze_public_event_history(witness.events), status=V1AnnotationStatus.NO_ACTION,
        actions=(), attempts=(attempt,),
    ))
    witness.event("turn_start", speaker=P[2])
    candidates = planner._public_actions(_history(witness), P[2], ALIVE)
    assert {(a.subject, a.action, a.object) for a in candidates if a.action in ("support", "oppose")} == {
        (P[2], "support", P[1]), (P[2], "oppose", P[1]),
    }
    assert V1SpeechAction(P[2], "point_as_werewolf", P[0]) in candidates


def test_expansion_and_rollout_share_mask_and_keep_conflicting_ordered_actions():
    accuse = V1SpeechAction(P[0], "point_as_werewolf", P[1])
    clear = V1SpeechAction(P[0], "point_as_villager", P[1])
    support = V1SpeechAction(P[0], "support", P[1])
    candidates = (accuse, clear, support)
    node = planner._Node(actions=(accuse,))

    class Choices:
        def __init__(self, *actions):
            self.actions = iter(actions)
            self.options = []

        def choice(self, options):
            self.options.append(tuple(options))
            action = next(self.actions)
            assert action in options
            return action

    expansion = Choices(clear)
    expanded = planner._select_and_expand(node, candidates, expansion)[-1]
    rollout = Choices(clear, support)
    terminal = planner._rollout(node, candidates, rollout)
    assert expansion.options == rollout.options[:1] == [(None, clear, support)]
    assert rollout.options[-1] == (None, support)
    assert expanded.actions == (accuse, clear) and terminal == (accuse, clear, support)
    assert planner._rollout(expanded, candidates, Choices(None)) == (accuse, clear)
    reversed_prefix = planner._Node(actions=(clear,))
    assert planner._rollout(reversed_prefix, candidates, Choices(accuse, None)) == (clear, accuse)


def test_stop_child_never_expands_and_root_stop_is_scored_as_empty():
    root = planner._Node()
    path = planner._select_and_expand(root, (), random.Random(0))
    stopped = path[-1]
    assert root.children[None] is stopped and stopped.stopped and stopped.actions == ()
    planner._backpropagate(path, -0.5)
    assert planner._select_and_expand(root, (), random.Random(1))[-1] is stopped
    action = V1SpeechAction(P[0], "point_as_werewolf", P[1])
    assert planner._select_and_expand(stopped, (action,), None) == [stopped]
    assert planner._rollout(stopped, (action,), None) == () and not stopped.children

    history = _history(ManualWitness(dead=DEAD))
    scorer = ControlledScorer(lambda _actions: _matrix([0.25] * 6))
    result = _plan(history, scorer, seed=31, budget=1)
    assert scorer.calls == [(history, P[0], ())]
    assert result.actions == () and result.terminal_evaluations == (((), -1.0),)


@pytest.mark.parametrize("team, expected", [(TEAM, -1.25), ((P[0], P[6]), -2.125)])
def test_reward_is_raw_suspicion_sum_from_living_nonwolves_to_the_speaker(team, expected):
    history = _history(ManualWitness(dead=DEAD))
    matrix = _matrix([0.875, 0.125, 0.25, 0.375, 0.5, 0.875])
    assert (matrix[2:6, 6] > 0).all()  # 已死亡的目标列保持原质量。
    original = matrix.clone()
    result = _plan(history, ControlledScorer(lambda _actions: matrix), known_wolves=team, budget=3)
    assert result.reward == expected
    assert all(reward == expected for _, reward in result.terminal_evaluations)
    torch.testing.assert_close(matrix, original)


def test_uct_and_backpropagation_update_only_the_selected_path():
    action = V1SpeechAction(P[0], "point_as_werewolf", P[1])
    stopped = planner._Node(stopped=True, visits=10, total_reward=-1)
    explored = planner._Node(actions=(action,), visits=2, total_reward=-0.8)
    root = planner._Node(visits=12, total_reward=-1.8, children={None: stopped, action: explored})
    assert planner._uct(planner._Node(), 0) == float("inf")
    assert planner._uct(explored, 12) == pytest.approx(-0.4 + 1.414 * sqrt(log(12) / 2))
    path = planner._select_and_expand(root, (action,), random.Random(4))
    assert path[1] is explored and path[2].stopped
    planner._backpropagate(path, -0.75)
    assert (root.visits, root.total_reward) == (13, pytest.approx(-2.55))
    assert (explored.visits, explored.total_reward) == (3, pytest.approx(-1.55))
    assert (path[2].visits, path[2].total_reward) == (1, -0.75)
    assert (stopped.visits, stopped.total_reward) == (10, -1)


def test_equal_uct_uses_local_seed():
    action = V1SpeechAction(P[0], "point_as_werewolf", P[1])
    stopped = planner._Node(stopped=True, visits=1, total_reward=-0.5)
    alternative = planner._Node(actions=(action,), visits=1, total_reward=-0.5)
    root = planner._Node(visits=2, total_reward=-1, children={None: stopped, action: alternative})
    chosen = [planner._select_and_expand(root, (action,), random.Random(seed))[1] for seed in (1, 0)]
    assert chosen[0] is stopped and chosen[1] is alternative


def test_default_budget_selects_actual_best_terminal_even_from_low_mean_and_visits():
    history = _history(ManualWitness(dead=DEAD))
    winner = (V1SpeechAction(P[0], "point_as_witch", P[1]),
              V1SpeechAction(P[0], "point_as_seer", P[3]),
              V1SpeechAction(P[0], "point_as_villager", P[4]))
    zero, stop, other = (_matrix([mass] * 6) for mass in (0, 0.125, 0.75))

    def probabilities(actions):
        if actions == winner:
            return zero
        return other if actions else stop

    scorer = ControlledScorer(probabilities)
    result = _plan(history, scorer)
    assert result.completed and result.elapsed_seconds >= 0
    assert result.completed_simulations == result.candidate_score_calls == len(scorer.calls) == 500
    assert result.reasoner_forward_calls is None
    assert (result.actions, result.reward) == (winner, 0.0)
    assert tuple(actions for _, _, actions in scorer.calls) == tuple(a for a, _ in result.terminal_evaluations)
    branches = defaultdict(list)
    for actions, reward in result.terminal_evaluations:
        branches[actions[:1]].append(reward)
    assert len(branches[()]) > len(branches[winner[:1]]) > 1
    assert sum(branches[()]) / len(branches[()]) > sum(branches[winner[:1]]) / len(branches[winner[:1]])


def test_same_local_seed_reproduces_all_evaluations_and_keeps_first_best_tie():
    history = _history(ManualWitness(dead=DEAD))
    scores = _matrix([0.25] * 6)
    scorer = ControlledScorer(lambda _actions: scores)
    state = random.getstate()
    first = _plan(history, scorer, budget=75)
    second = _plan(history, scorer, budget=75, alive_players=tuple(reversed(ALIVE)), known_wolves=set(TEAM))
    third = _plan(history, scorer, budget=75, seed=2)
    assert random.getstate() == state
    assert first.terminal_evaluations == second.terminal_evaluations != third.terminal_evaluations
    assert (first.actions, first.reward) == first.terminal_evaluations[0]


def test_cpu_reasoner_observes_only_original_public_history_and_speech_end(monkeypatch):
    from werewolf.agents.gpt_agent import GPTAgent
    from werewolf.envs.werewolf_text_env_v0 import WerewolfTextEnvV0
    from werewolf.speech.speech_perceiver import SpeechPerceiver

    def forbidden(*_args, **_kwargs):
        pytest.fail("搜索不能访问游戏、私有观察、Parser 或生成 Agent")

    for owner, names in (
        (WerewolfTextEnvV0, ("step", "get_observation_for", "observer_private_knowledge")),
        (SpeechPerceiver, ("parse_strict", "parse_with_audit")),
        (GPTAgent, ("act",)),
    ):
        for name in names:
            monkeypatch.setattr(owner, name, forbidden)

    history = _history(ManualWitness(dead=DEAD))
    before = history.tokens
    capacity = ExperimentCapacity(len(before) + 4)
    reasoner = MultiMindReasoner(capacity, hidden_size=16, num_layers=1, num_heads=2).eval()
    inputs = []

    def capture(_module, _args, kwargs):
        assert torch.is_inference_mode_enabled() and not reasoner.training
        inputs.append(kwargs)

    with reasoner.register_forward_pre_hook(capture, with_kwargs=True):
        result = _plan(history, reasoner, seed=31, budget=5)
    assert result.reasoner_forward_calls == result.candidate_score_calls == len(inputs) == 5
    assert result.terminal_evaluations[0][0] == ()
    for kwargs, (actions, _) in zip(inputs, result.terminal_evaluations, strict=True):
        expected = encode_candidate(history, P[0], actions, capacity)
        assert set(kwargs) == set(expected.kwargs())
        for name, tensor in expected.kwargs().items():
            torch.testing.assert_close(kwargs[name][0], tensor)
    assert history.tokens is before


def test_scoring_failures_propagate_and_no_partial_plan_is_returned():
    history = _history(ManualWitness(dead=DEAD))
    failure = RuntimeError("controlled scoring failure")

    def score(actions):
        if len(scorer.calls) == 2:
            raise failure
        return _matrix([0.25] * 6)

    scorer = ControlledScorer(score)
    with pytest.raises(RuntimeError) as exc:
        _plan(history, scorer, budget=10)
    assert exc.value is failure and len(scorer.calls) == 2

    short = MultiMindReasoner(ExperimentCapacity(len(history.tokens)), hidden_size=16, num_layers=1, num_heads=2)
    with pytest.raises(ValueError, match="requires eval mode"):
        _plan(history, short, budget=1)
    with pytest.raises(ValueError, match="sequence needs .* capacity"):
        _plan(history, short.eval(), budget=1)
    assert not short._forward_hooks


def test_invalid_public_inputs_and_scorer_results_fail_explicitly():
    history = _history(ManualWitness(dead=DEAD))
    scorer = ControlledScorer(lambda _actions: torch.zeros((7, 7)))
    with pytest.raises(ValueError, match="match the current public speech turn"):
        planner.plan_speech(history, P[1], alive_players=ALIVE, known_wolves=TEAM, reasoner=scorer, seed=0)
    with pytest.raises(ValueError, match="match public eliminations"):
        _plan(history, scorer, alive_players=P)
    with pytest.raises(ValueError, match="including the acting wolf"):
        _plan(history, scorer, known_wolves=(P[1],))
    with pytest.raises(ValueError, match="positive simulation budget"):
        _plan(history, scorer, budget=0)
    assert not scorer.calls

    with pytest.raises(ValueError, match=r"floating \[7, 7\]"):
        _plan(history, ControlledScorer(lambda _actions: torch.zeros((7,))), budget=1)
    invalid = _matrix([0, float("nan"), 0, 0, 0, 0])
    with pytest.raises(ValueError, match="invalid suspicion mass"):
        _plan(history, ControlledScorer(lambda _actions: invalid), budget=1)
