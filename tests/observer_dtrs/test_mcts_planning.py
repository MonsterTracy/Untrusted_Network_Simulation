"""固定人工公开树；不调用Parser、模型、真实游戏或科研数据。"""
from collections import Counter
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
import json
import math
import random
import socket
import subprocess
from unittest.mock import patch

import numpy as np
import pytest
import torch

from werewolf.strategy import speech_planning as speech
from werewolf.strategy.counterfactual_tom import CounterfactualToMConsumer
from tests.tom.test_counterfactual_consumer import PublicWitness, P
from werewolf.canonical_collection import (
    construct_v1_speech_annotation, freeze_public_event_history, V1AnnotationStatus,
    V1SpeechAction, V1PerceptionAttempt, V1_SPEECH_PROMPT_VERSION, V1_SPEECH_PARSER_VERSION,
)
from werewolf.strategy.mcts_planning import (
    PlannerState, LegalActionProvider, TransitionModel, ValueModel, MCTSPlanner,
)
from werewolf.strategy import mcts_planning as search
from werewolf.observer_dtrs.online_context import ShadowRequest


class ManualWitness(PublicWitness):
    """只复用公共事件/PRE构造器，所有V1明确为手工夹具，不运行旧夹具Parser。"""
    def __init__(self, phase='discussion', dead=(), last=False):
        self.events, self.annotations = [], []
        self.alive = tuple(p for p in P if p not in dead)
        self.event('phase_change', day=0, phase='night')
        self.event('death_announcement', dead_players=list(dead))
        self.event('phase_change', day=1, phase='discussion')
        if phase == 'pk_discussion':
            for player in self.alive:
                self.event('turn_start', speaker=player)
                self.speech(player, 'no_commitment', None, '人工公开中性发言')
            other = next(p for p in self.alive if p != P[0])
            self.event('phase_change', day=1, phase='vote')
            self.event('vote_result', votes=[{'voter': p, 'target': other if p == P[0]
                       else P[0] if p == other else None} for p in self.alive])
            self.event('exile_result', exiled_players=[])
            self.event('phase_change', day=1, phase='pk_discussion')
            prior = (other,) if last else ()
        else:
            prior = self.alive[1:] if last else ()
        for player in prior:
            self.event('turn_start', speaker=player)
            self.speech(player, 'no_commitment', None, '人工公开中性发言')
        self.event('turn_start', speaker=P[0])

    def speech(self, speaker, action, target, text):
        self.event('public_speech', speaker=speaker, raw_text=text)
        attempt = V1PerceptionAttempt(
            attempt_index=1, call_id=f'manual-{len(self.events)}',
            backend_id='manual-fixture-not-executed', model_id='manual-fixture-not-executed',
            prompt_version=V1_SPEECH_PROMPT_VERSION, parser_version=V1_SPEECH_PARSER_VERSION,
            status=V1AnnotationStatus.OK, raw_response='MANUAL_FIXTURE',
            error_category=None, error_message=None)
        self.annotations.append(construct_v1_speech_annotation(
            freeze_public_event_history(self.events), status=V1AnnotationStatus.OK,
            actions=(V1SpeechAction(speaker, action, target),), attempts=(attempt,)))


@pytest.fixture(autouse=True)
def no_execution(monkeypatch):
    from werewolf.agents.gpt_agent import GPTAgent
    from werewolf.speech.speech_perceiver import SpeechPerceiver
    from werewolf.envs.werewolf_text_env_v0 import WerewolfTextEnvV0
    from werewolf.observer_dtrs import online_model, dataset
    from werewolf.speech import commit_workflow

    def forbidden(*args, **kwargs):
        raise AssertionError('人工搜索不得执行模型、语言、数据reader、游戏或网络')

    for owner, names in (
        (GPTAgent, ('_generate_public_speech', 'act')),
        (SpeechPerceiver, ('parse_with_audit', 'parse_strict')),
        (WerewolfTextEnvV0, ('reset', 'step', 'next_phase', 'step_verified_speech')),
        (CounterfactualToMConsumer, ('__init__', 'probabilities')),
        (torch.nn.Module, ('__call__',)),
        (torch, ('load',)),
        (online_model, ('read_deployment',)),
        (dataset, ('open_dataset',)),
        (commit_workflow, ('realize_and_commit',)),
        (socket.socket, ('connect',)),
        (subprocess, ('Popen',)),
    ):
        for name in names:
            monkeypatch.setattr(owner, name, forbidden)


def action_provider(*, phase='discussion', dead=(), last=False, team=(P[0], P[3])):
    pre = ManualWitness(phase, dead, last).pre()
    request = ShadowRequest(pre, team, 'manual-search', 'manual-attempt', captured_at_ns=0)
    return LegalActionProvider(PlannerState.from_request(request))


def responses(actions, state, label):
    return tuple((p, label) for p in actions.response_speakers(state))


def finite_tree(actions, *, stochastic=True, case='delayed'):
    """事前固定：A短视值.6；B短视值.4、观察响应后可达.9；其余候选始终保留。"""
    root = actions.root
    first = actions.actions(root)[0]
    delayed = speech.SpeechPlan(speech.PlanningAction.NO_COMMITMENT)
    defend = speech.SpeechPlan(speech.PlanningAction.SELF_DEFEND)
    kernel, values = {}, {}
    for action in actions.actions(root):
        labels = ('left', 'right') if stochastic and actions.response_speakers(root) else ('left',)
        rows = tuple((1 / len(labels), responses(actions, root, label)) for label in labels)
        kernel[((), action)] = rows
        for _, reply in rows:
            child = actions.advance(root, action, reply)
            values[child.history] = .6 if action == first else .4 if action == delayed else .2
            if case == 'constant':
                values[child.history] = .5
            if child.actor is None:
                continue
            for second in actions.actions(child):
                final_reply = responses(actions, child, 'final')
                kernel[(child.history, second)] = ((1., final_reply),)
                leaf = actions.advance(child, second, final_reply)
                if case == 'constant':
                    score = .5
                elif case == 'chance':
                    score = float(reply[0][1] == 'right')
                elif action == first:
                    score = .6
                elif action == delayed:
                    matching = defend if reply[0][1] == 'left' else delayed
                    score = .9 if second == matching else .1
                else:
                    score = .2
                values[leaf.history] = score
    return MCTSPlanner(actions,
        TransitionModel(root.root_binding, kernel, 'manual-finite-responses'),
        ValueModel(root.root_binding, values, 'manual-bounded-leaves'))


def exact_oracle(planner, depth=2):
    """验证用Fraction全枚举；不调用搜索器、采样器或其计数器。"""
    def visit(state):
        if state.depth == depth or state.actor is None:
            return Fraction(str(planner.value.values[state.history]))
        return max(evaluate(state, a) for a in planner.actions.actions(state))

    def evaluate(state, action):
        return sum(Fraction(str(p)) * visit(planner.actions.advance(state, action, reply))
                   for p, reply in planner.transition.kernel[(state.history, action)])

    return {a: evaluate(planner.actions.root, a) for a in planner.actions.actions(planner.actions.root)}


def traced_search(planner, *, budget, seed):
    """仅在测试中观察原UCT；不增加模型调用、随机抽样或决策输入。"""
    original_node, original_sample = search._Node, TransitionModel.sample
    original_value, original_choice = ValueModel.evaluate, random.Random.choice
    nodes, candidates, expected = {}, {}, {}
    modes, selections, leaves = Counter(), Counter(), []
    active_path = []
    pending_rollout = False

    def make_node(state):
        assert state.history not in nodes, '不同根动作或公开历史不得合并节点'
        node = original_node(state)
        nodes[state.history] = node
        if state.depth < 2 and state.actor is not None:
            candidates[state.history] = planner.actions.actions(state)
        return node

    def choose(rng, sequence):
        nonlocal pending_rollout
        pending_rollout = True
        return original_choice(rng, sequence)

    def sample(model, state, action, actions, rng):
        nonlocal pending_rollout
        node = nodes[state.history]
        if state.depth == 0:
            active_path.clear()
        if pending_rollout:
            mode = 'rollout'
        elif node.edges[action].visits == 0:
            mode = 'expansion'
            wanted = next(a for a in candidates[state.history]
                          if a not in node.edges or node.edges[a].visits == 0)
            assert action == wanted
        else:
            mode = 'uct'
            wanted = max(candidates[state.history], key=lambda a:
                node.edges[a].total / node.edges[a].visits
                + math.sqrt(2 * math.log(node.visits) / node.edges[a].visits))
            assert action == wanted
        modes[(state.depth, mode)] += 1
        selections[(state.history, mode)] += 1
        if mode != 'rollout':
            active_path.append((state.history, action))
        pending_rollout = False
        return original_sample(model, state, action, actions, rng)

    def evaluate(model, state):
        score = original_value(model, state)
        leaves.append((state.history, score))
        for key in active_path:
            count, total = expected.get(key, (0, 0.))
            expected[key] = count + 1, total + score
        return score

    with patch.object(search, '_Node', make_node), patch.object(TransitionModel, 'sample', sample), \
         patch.object(ValueModel, 'evaluate', evaluate), patch.object(random.Random, 'choice', choose):
        result = planner.plan(budget=budget, seed=seed)

    # 每次真实采样叶值沿实际树路径恰好回传一次；默认rollout不伪造树边访问。
    branch_visits = Counter(history[:1] for history, _ in leaves)
    for history, node in nodes.items():
        if len(history) == 1:
            assert node.visits == branch_visits[history]
            assert 0 <= node.visits - sum(e.visits for e in node.edges.values()) <= 1
        for action, edge in node.edges.items():
            count, total = expected[(history, action)]
            assert edge.visits == count and edge.total == total
            for child_history, child in edge.outcomes.items():
                assert child.state.history == child_history
                assert child_history[:-1] == history and child_history[-1][0] == action
    assert nodes[()].visits == result.completed_simulations == len(leaves)
    assert sum(modes.values()) == result.transition_calls
    assert len(leaves) == result.value_calls
    return result, {'nodes': nodes, 'modes': modes, 'selections': selections, 'leaves': leaves}


def micro_tree(monkeypatch):
    """抽象两根动作、两公开响应、两后续动作；仅测试夹具限制分支数。"""
    actions = action_provider()
    root = actions.root
    first = actions.actions(root)[0]
    delayed = speech.SpeechPlan(speech.PlanningAction.NO_COMMITMENT)
    defend = speech.SpeechPlan(speech.PlanningAction.SELF_DEFEND)
    original_actions = LegalActionProvider.actions

    def abstract_actions(provider, state):
        if provider is not actions:
            return original_actions(provider, state)
        provider.validate(state)
        if state.actor is None:
            return ()
        return (first, delayed) if state.depth == 0 else (defend, delayed)

    # 不修改生产枚举、不删减原完整树；离开monkeypatch作用域即恢复。
    monkeypatch.setattr(LegalActionProvider, 'actions', abstract_actions)
    kernel, values = {}, {}
    for action in (first, delayed):
        kernel[((), action)] = tuple((.5, responses(actions, root, label))
                                     for label in ('left', 'right'))
        for _, reply in kernel[((), action)]:
            child = actions.advance(root, action, reply)
            values[child.history] = .6 if action == first else .4
            for second in (defend, delayed):
                final = responses(actions, child, 'final')
                kernel[(child.history, second)] = ((1., final),)
                leaf = actions.advance(child, second, final)
                if action == first:
                    score = .6
                elif reply[0][1] == 'left':
                    score = 1. if second == defend else .1
                else:
                    score = .1 if second == defend else .8
                values[leaf.history] = score
    return MCTSPlanner(actions, TransitionModel(root.root_binding, kernel, 'abstract-two-actions'),
                       ValueModel(root.root_binding, values, 'abstract-bounded-leaves'))


@pytest.mark.parametrize('phase', ['discussion', 'pk_discussion'])
@pytest.mark.parametrize('dead', [(), (P[3],), (P[1], P[6])])
@pytest.mark.parametrize('last', [False, True])
def test_candidates_keep_original_protocol_order(phase, dead, last):
    actions = action_provider(phase=phase, dead=dead, last=last)
    targets = tuple(p for p in P[1:] if p not in dead)
    # 原协议：先指控/澄清全体非自身存活席位，再支持/反对当日已发言席位。
    spoken_targets = targets if phase == 'pk_discussion' or last else ()
    expected = tuple(speech.SpeechPlan(action, target)
        for action in (speech.PlanningAction.ACCUSE_WOLF, speech.PlanningAction.CLEAR)
        for target in targets)
    expected += tuple(speech.SpeechPlan(action, target)
        for action in (speech.PlanningAction.SUPPORT, speech.PlanningAction.OPPOSE)
        for target in spoken_targets)
    expected += (speech.SpeechPlan(speech.PlanningAction.SELF_DEFEND),
                 speech.SpeechPlan(speech.PlanningAction.NO_COMMITMENT))
    assert actions.actions(actions.root) == expected
    assert tuple(speech.PlanningAction) == (
        speech.PlanningAction.ACCUSE_WOLF, speech.PlanningAction.CLEAR,
        speech.PlanningAction.SUPPORT, speech.PlanningAction.OPPOSE,
        speech.PlanningAction.SELF_DEFEND, speech.PlanningAction.NO_COMMITMENT)


def test_simulated_state_reuses_all_candidates_and_visible_history_only():
    actions = action_provider()
    root = actions.root
    action = actions.actions(root)[0]
    child = actions.advance(root, action, responses(actions, root, 'observed'))
    assert root.origin == 'ROOT_PRE' and root.prefix is not None
    assert child.origin == 'SIMULATED' and child.prefix is None
    assert child.actor == P[3] and child.depth == 1
    assert child.spoken_today == P[:3]
    plans = actions.actions(child)
    assert speech.SpeechPlan(speech.PlanningAction.ACCUSE_WOLF, P[0]) in plans
    assert speech.SpeechPlan(speech.PlanningAction.OPPOSE, P[2]) in plans
    assert speech.SpeechPlan(speech.PlanningAction.OPPOSE, P[4]) not in plans
    assert child == actions.advance(root, action, responses(actions, root, 'observed'))
    with pytest.raises(FrozenInstanceError):
        child.actor = P[4]
    with pytest.raises(TypeError):
        PlannerState.from_request(child)
    for wrong in (replace(child, actor=P[2]), replace(child, prefix=root.prefix),
                  replace(child, root_binding=('wrong', *child.root_binding[1:]))):
        with pytest.raises(ValueError):
            actions.actions(wrong)


def test_one_step_deterministic_uct_equals_complete_greedy():
    planner = finite_tree(action_provider(), stochastic=False)
    count = len(planner.actions.actions(planner.actions.root))
    greedy = planner.greedy(budget=2 * count)
    result = planner.plan(budget=2 * count, seed=11, depth=1)
    assert result.selected_plan == greedy.selected_plan
    assert result.action_statistics == greedy.action_statistics
    assert result.completed_simulations == count
    assert result.transition_calls == result.value_calls == count
    assert result.budget_used == greedy.budget_used == 2 * count
    assert result.leaf_depth_counts == ((1, count),)


def test_delayed_reversal_exact_expectimax_and_independent_oracle():
    planner = finite_tree(action_provider())
    root = planner.actions.root
    expected = exact_oracle(planner)
    result = planner.expectimax(budget=2000)
    greedy = planner.greedy(budget=2000)
    delayed = speech.SpeechPlan(speech.PlanningAction.NO_COMMITMENT)
    assert result.selected_plan == delayed
    assert greedy.selected_plan == planner.actions.actions(root)[0]
    assert expected[delayed] == Fraction(9, 10)
    assert all(mean == pytest.approx(float(expected[a])) for a, _, mean in result.action_statistics)
    assert result.transition_calls == 588 and result.value_calls == 560
    assert result.transition_requests == 574 and result.budget_used == 1148
    assert greedy.transition_calls == greedy.value_calls == 28
    assert greedy.budget_used == 56  # 同上限，不伪称实际调用量相同或填充无意义调用。


def test_random_responses_are_averaged_not_maximized():
    planner = finite_tree(action_provider(), case='chance')
    kernel = dict(planner.transition.kernel)
    for action in planner.actions.actions(planner.actions.root):
        left, right = kernel[((), action)]
        kernel[((), action)] = ((.25, left[1]), (.75, right[1]))
    model = TransitionModel(planner.actions.root.root_binding, kernel, 'manual-quarter-chance')
    planner = replace(planner, transition=model)
    result = planner.expectimax(budget=2000)
    assert all(mean == pytest.approx(.75) for _, _, mean in result.action_statistics)
    assert set(exact_oracle(planner).values()) == {Fraction(3, 4)}
    rng = random.Random(19)
    action = planner.actions.actions(planner.actions.root)[0]
    right_count = sum(model.sample(planner.actions.root, action, planner.actions, rng).history[0][1][0][1]
                      == 'right' for _ in range(2000))
    assert abs(right_count / 2000 - .75) < .04


def test_uct_backpropagates_sampled_chance_values(monkeypatch):
    planner = finite_tree(action_provider(), case='chance')
    kernel = dict(planner.transition.kernel)
    for action in planner.actions.actions(planner.actions.root):
        left, right = kernel[((), action)]
        kernel[((), action)] = ((.25, left[1]), (.75, right[1]))
    planner = replace(planner, transition=TransitionModel(
        planner.actions.root.root_binding, kernel, 'manual-quarter-chance'))
    original_sample, observed = TransitionModel.sample, {}

    def record_sample(model, state, action, actions, rng):
        successor = original_sample(model, state, action, actions, rng)
        if state.depth == 0:
            # 此树的最终人工叶值仅由已采样公开响应决定，与第二动作无关。
            score = float(successor.history[0][1][0][1] == 'right')
            observed.setdefault(action, []).append(score)
        return successor

    monkeypatch.setattr(TransitionModel, 'sample', record_sample)
    result = planner.plan(budget=6000, seed=19)
    assert result.completed_simulations == 2000
    for action, visits, mean in result.action_statistics:
        samples = observed[action]
        assert visits == len(samples)
        assert mean == pytest.approx(sum(samples) / len(samples))
    all_scores = [score for samples in observed.values() for score in samples]
    assert abs(sum(all_scores) / len(all_scores) - .75) < .04
    assert result.transition_calls == 4000 and result.value_calls == 2000


def test_adjacent_controlled_wolves_need_no_fabricated_public_response():
    actions = action_provider(team=(P[0], P[1]))
    root = actions.root
    assert actions.response_speakers(root) == ()
    child = actions.advance(root, actions.actions(root)[0], ())
    assert child.actor == P[1] and child.spoken_today == (P[0],)
    with pytest.raises(ValueError, match='队列'):
        actions.advance(root, actions.actions(root)[0], ((P[2], 'fabricated'),))
    planner = finite_tree(actions, case='constant')
    result = planner.plan(budget=600, seed=19)
    assert result.leaf_depth_counts == ((2, 200),)


@pytest.mark.parametrize('phase,dead,last,depth', [
    ('discussion', (), False, 2), ('discussion', (P[3],), False, 1),
    ('discussion', (), True, 1), ('pk_discussion', (), False, 1),
])
def test_only_next_controlled_wolf_has_second_max(phase, dead, last, depth):
    planner = finite_tree(action_provider(phase=phase, dead=dead, last=last))
    result = planner.plan(budget=600, seed=3)
    assert result.requested_depth == 2
    assert result.leaf_depth_counts == ((depth, result.completed_simulations),)
    assert result.transition_calls == depth * result.completed_simulations
    assert result.value_calls == result.completed_simulations
    assert result.selected_plan in planner.actions.actions(planner.actions.root)


def test_second_max_cannot_see_unobserved_or_duplicate_hidden_outcomes():
    planner = finite_tree(action_provider())
    root = planner.actions.root
    action = planner.actions.actions(root)[-1]
    reply = responses(planner.actions, root, 'same-visible-response')
    with pytest.raises(ValueError, match='隐藏后继'):
        TransitionModel(root.root_binding, {((), action): ((.5, reply), (.5, reply))}, 'hidden-split')
    with pytest.raises(ValueError, match='隐藏状态'):
        TransitionModel(root.root_binding, {((), action): ((1., ((P[1], 'visible', 'private-role'),)),)}, 'hidden-field')
    expected = exact_oracle(planner)
    assert expected[action] == Fraction(9, 10)
    # 不能观察left/right而事先固定第二动作时，最佳期望只有.5。
    left, right = planner.transition.kernel[((), action)]
    children = [planner.actions.advance(root, action, row[1]) for row in (left, right)]
    common = planner.actions.actions(children[0])
    open_loop = max(sum(Fraction(str(p)) * Fraction(str(planner.value.values[
        planner.actions.advance(child, second, planner.transition.kernel[(child.history, second)][0][1]).history]))
        for p, child in zip((.5, .5), children, strict=True)) for second in common)
    assert open_loop == Fraction(1, 2)


@pytest.mark.parametrize('weights', [(0., 0.), (.2, .3), (-.5, 1.5), (float('nan'), 1.), (True, 1.)])
def test_invalid_probability_and_zero_denominator_stop(weights):
    actions = action_provider()
    action = actions.actions(actions.root)[0]
    rows = tuple((p, responses(actions, actions.root, str(i))) for i, p in enumerate(weights))
    with pytest.raises(ValueError):
        TransitionModel(actions.root.root_binding, {((), action): rows}, 'invalid-probability')


@pytest.mark.parametrize('value', [None, -.1, 1.1, float('nan'), float('inf'), True])
def test_invalid_values_are_not_clipped_or_filled(value):
    actions = action_provider()
    with pytest.raises(ValueError):
        ValueModel(actions.root.root_binding, {(): value}, 'invalid-value')


def test_missing_value_missing_transition_and_illegal_response_stop():
    planner = finite_tree(action_provider())
    root = planner.actions.root
    with pytest.raises(ValueError, match='叶值缺失'):
        replace(planner, value=ValueModel(root.root_binding, {}, 'missing')).plan(budget=100, seed=1)
    with pytest.raises(ValueError, match='转移缺失'):
        replace(planner, transition=TransitionModel(root.root_binding, {}, 'missing')).plan(budget=100, seed=1)
    action = planner.actions.actions(root)[0]
    bad = TransitionModel(root.root_binding, {((), action): ((1., ((P[6], 'wrong queue'),)),)}, 'wrong-queue')
    with pytest.raises(ValueError, match='队列'):
        replace(planner, transition=bad).plan(budget=100, seed=1)
    with pytest.raises(ValueError, match='同根'):
        replace(planner, value=ValueModel(('wrong',), {}, 'wrong-root'))


@pytest.mark.parametrize('budget', [0, 1, 2, 3, 41, 42, 43, 600, 2000])
def test_complete_simulation_budget_and_zero_visit_handling(budget):
    planner = finite_tree(action_provider())
    result = planner.plan(budget=budget, seed=5)
    count = len(planner.actions.actions(planner.actions.root))
    assert result.completed_simulations == budget // 3
    assert result.transition_calls == 2 * result.completed_simulations
    assert result.transition_requests == result.transition_calls
    assert result.value_calls == result.completed_simulations
    assert result.budget_used == 3 * result.completed_simulations <= budget
    assert len(result.action_statistics) == count
    if result.completed_simulations < count:
        assert result.selected_plan is None and result.status == 'INCOMPLETE_ROOT_COVERAGE'
    else:
        assert result.status == 'COMPLETE'
        assert result.selected_plan in planner.actions.actions(planner.actions.root)
    for evaluate in (planner.greedy, planner.expectimax):
        other = evaluate(budget=budget)
        assert other.budget_used <= budget
        assert other.transition_requests <= other.transition_calls
        assert other.completed_simulations == 0
        if any(mean is None for _, _, mean in other.action_statistics):
            assert other.selected_plan is None


@pytest.mark.parametrize('budget,seed,depth', [(-1, 0, 2), (True, 0, 2), (10, -1, 2),
    (10, True, 2), (10, 0, 0), (10, 0, 3), (10, 0, True)])
def test_invalid_search_parameters_stop(budget, seed, depth):
    with pytest.raises(ValueError):
        finite_tree(action_provider()).plan(budget=budget, seed=seed, depth=depth)


def test_reproducibility_global_rng_and_no_execution_authority():
    planner = finite_tree(action_provider())
    python_before, numpy_before, torch_before = random.getstate(), np.random.get_state(), torch.get_rng_state()
    result = planner.plan(budget=1200, seed=2026)
    assert result == planner.plan(budget=1200, seed=2026)
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0] and np.array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]
    assert torch.equal(torch_before, torch.get_rng_state())
    assert result.scope == 'ARTIFICIAL_TRANSITION_AND_LEAF_VALUES_ONLY'
    assert result.execution_authorized is False
    with pytest.raises(FrozenInstanceError):
        result.selected_plan = None


def test_exact_ties_use_original_candidate_order_for_all_algorithms():
    planner = finite_tree(action_provider(), case='constant')
    expected = planner.actions.actions(planner.actions.root)[0]
    for result in (planner.greedy(budget=2000), planner.expectimax(budget=2000),
                   planner.plan(budget=2000, seed=13)):
        assert result.selected_plan == expected


@pytest.mark.parametrize('budget', [600, 2000, 9000])
def test_full_tree_trace_preserves_choices_and_checks_both_uct_layers(budget):
    planner = finite_tree(action_provider())
    result, trace = traced_search(planner, budget=budget, seed=17)
    assert result == planner.plan(budget=budget, seed=17)
    plans = planner.actions.actions(planner.actions.root)
    root = trace['nodes'][()]
    assert len(plans) == 14  # 原树为14根动作、28个响应后状态，并非28个根动作。
    assert result.selected_plan == plans[0]  # 保留原有限预算失败，不要求UCT优于贪心。
    assert trace['modes'][(0, 'expansion')] == 14
    assert trace['modes'][(1, 'rollout')] == 14
    assert sum(e.visits for e in root.edges.values()) == result.completed_simulations
    assert root.edges[plans[0]].total == pytest.approx(.6 * root.edges[plans[0]].visits)
    children = root.edges[plans[-1]].outcomes.values()
    if budget == 9000:
        assert trace['modes'][(1, 'uct')] > 0
        assert all(len(child.edges) == 20 for child in children)
        assert any(score == .9 for history, score in trace['leaves'] if history[0][0] == plans[-1])
    else:
        assert all(len(child.edges) < 20 for child in children)
        assert all(score == .1 for history, score in trace['leaves'] if history[0][0] == plans[-1])


def test_micro_tree_fraction_oracle_and_delayed_reversal(monkeypatch):
    planner = micro_tree(monkeypatch)
    plans = planner.actions.actions(planner.actions.root)
    half = Fraction(1, 2)
    expected = {
        plans[0]: half * max(Fraction(3, 5), Fraction(3, 5)) * 2,
        plans[1]: half * max(Fraction(1), Fraction(1, 10))
                  + half * max(Fraction(1, 10), Fraction(4, 5)),
    }
    assert expected == exact_oracle(planner)
    assert tuple(expected.values()) == (Fraction(3, 5), Fraction(9, 10))
    assert max(half * (Fraction(1) + Fraction(1, 10)),
               half * (Fraction(1, 10) + Fraction(4, 5))) == Fraction(11, 20)
    greedy, exact = planner.greedy(budget=100), planner.expectimax(budget=100)
    assert greedy.selected_plan == plans[0] and exact.selected_plan == plans[1]
    assert all(mean == pytest.approx(float(expected[a])) for a, _, mean in exact.action_statistics)
    assert exact.transition_calls == 12 and exact.value_calls == 8
    assert exact.transition_requests == 10 and exact.budget_used == 20


@pytest.mark.parametrize('seed', [0, 1, 2, 3, 17])
@pytest.mark.parametrize('budget', [24, 120, 600, 3000])
def test_micro_tree_fixed_seed_budget_grid(monkeypatch, seed, budget):
    planner = micro_tree(monkeypatch)
    oracle = exact_oracle(planner)
    result, trace = traced_search(planner, budget=budget, seed=seed)
    assert result == planner.plan(budget=budget, seed=seed)
    assert result.completed_simulations == budget // 3
    assert result.transition_calls == 2 * result.value_calls
    assert result.transition_requests == result.transition_calls
    assert result.budget_used == 3 * result.completed_simulations
    if budget == 3000:
        # 固定微型树的数值回归；不是任意树、种子或有限预算的功效保证。
        assert all(abs(mean - float(oracle[a])) < .1 for a, _, mean in result.action_statistics)
    roots = []
    for action, visits, mean in result.action_statistics:
        edge = trace['nodes'][()].edges[action]
        children = []
        for history, child in edge.outcomes.items():
            children.append({'response': history[0][1][0][1], 'visits': child.visits,
                'counts': [child.edges[a].visits if a in child.edges else 0
                           for a in planner.actions.actions(child.state)]})
        roots.append({'action': action.public_payload(), 'visits': visits, 'total': edge.total,
                      'mean': mean, 'oracle': float(oracle[action]), 'children': children})
    print('MICRO_TREE=' + json.dumps({'seed': seed, 'budget': budget, 'roots': roots,
        'selected': result.selected_plan.public_payload(),
        'regret': float(max(oracle.values()) - oracle[result.selected_plan]),
        'transitions': result.transition_calls, 'requests': result.transition_requests,
        'values': result.value_calls, 'used': result.budget_used}, ensure_ascii=False))


def test_fixed_tree_comparison_reports_actual_cost_and_oracle_regret():
    planner = finite_tree(action_provider())
    oracle = exact_oracle(planner)
    optimum = max(oracle.values())
    rows = []
    for budget in (42, 600, 2000, 9000):
        for result in (planner.greedy(budget=budget), planner.expectimax(budget=budget),
                       planner.plan(budget=budget, seed=17)):
            regret = None if result.selected_plan is None else optimum - oracle[result.selected_plan]
            assert regret is None or regret >= 0
            assert result.budget_used <= budget
            rows.append({'algorithm': result.algorithm, 'budget': budget, 'used': result.budget_used,
                'transitions': result.transition_calls, 'transition_requests': result.transition_requests,
                'values': result.value_calls, 'simulations': result.completed_simulations,
                'plan': None if result.selected_plan is None else result.selected_plan.public_payload(),
                'oracle_regret': None if regret is None else float(regret), 'status': result.status})
    # 不要求任何有限预算下UCT必胜；oracle只用于独立验证，不向算法提供答案。
    print('\nMANUAL_TREE_COMPARISON=' + json.dumps(rows, ensure_ascii=False))
