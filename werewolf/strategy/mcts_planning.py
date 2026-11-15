"""人工有限响应树上的短视、Expectimax和UCT；没有真实策略执行资格。

真实PRE只绑定根；后继只记录人工公开响应，绝不构造canonical PRE或未来Q/p。
固定同阶段队列中，已知存活狼队的下一位发言者才有第二次可控决策。
所有回报都是显式人工叶值，不读取D1监督、ToM、D2、D3风险或游戏胜率。
"""
from dataclasses import dataclass, field, replace
import math
import random
from types import MappingProxyType

from werewolf.strategy.counterfactual_tom import derive_public_phase_speaker_order
from werewolf.strategy.speech_planning import SpeechPlan, generate_candidates, _ordered_candidates
from werewolf.canonical_collection.public_history import PLAYER_IDS
from werewolf.observer_dtrs.online_context import ShadowRequest


@dataclass(frozen=True)
class PlannerState:
    root_binding: tuple[str, str, str, str]
    history: tuple
    actor: str | None
    alive_players: tuple[str, ...]
    spoken_today: tuple[str, ...]
    phase_order: tuple[str, ...]
    controlled_wolves: tuple[str, ...]
    prefix: object = field(default=None, repr=False, compare=False, hash=False)

    @classmethod
    def from_request(cls, request):
        """复用合法请求内容；对象类型不证明原始观察真实，也不授予模型资格。"""
        if type(request) is not ShadowRequest:
            raise TypeError('需要原ShadowRequest，不能以模拟历史冒充真实PRE')
        pre = request.prefix
        wolves = request.contexts[0].alive_wolf_ids
        if len(wolves) >= len(pre.alive_observer_ids) - len(wolves):
            raise ValueError('终局没有合法狼人发言')
        spoken = {e.speaker for e in pre.public_event_history.events
                  if e.event_type == 'public_speech'
                  and e.temporal_state.day == pre.public_temporal_state.day}
        return cls((pre.game_id, request.attempt_id, pre.boundary_id, pre.prefix_digest), (),
                   pre.current_speaker, pre.alive_observer_ids,
                   tuple(p for p in PLAYER_IDS if p in spoken),
                   derive_public_phase_speaker_order(pre), wolves, pre)

    @property
    def origin(self):
        return 'ROOT_PRE' if not self.history else 'SIMULATED'

    @property
    def depth(self):
        return len(self.history)


@dataclass(frozen=True)
class LegalActionProvider:
    root: PlannerState

    def __post_init__(self):
        r = self.root
        if (type(r) is not PlannerState or r.history or r.prefix is None
                or r.phase_order != derive_public_phase_speaker_order(r.prefix)
                or r.actor != r.prefix.current_speaker or r.actor not in r.controlled_wolves
                or r.alive_players != r.prefix.alive_observer_ids
                or not 1 <= len(r.controlled_wolves) <= 2
                or len(set(r.controlled_wolves)) != len(r.controlled_wolves)
                or not set(r.controlled_wolves) <= set(r.alive_players)
                or type(r.root_binding) is not tuple or len(r.root_binding) != 4
                or any(type(part) is not str or not part for part in r.root_binding)
                or (r.root_binding[0], r.root_binding[2]) != (r.prefix.game_id, r.prefix.boundary_id)
                or r.root_binding[3] != r.prefix.prefix_digest):
            raise ValueError('需要同一合法根及明确的存活狼队控制范围')
        spoken = {e.speaker for e in r.prefix.public_event_history.events
                  if e.event_type == 'public_speech'
                  and e.temporal_state.day == r.prefix.public_temporal_state.day}
        if r.spoken_today != tuple(p for p in PLAYER_IDS if p in spoken):
            raise ValueError('根公开发言集合不一致')

    def _candidates(self, state):
        if state.actor is None:
            return ()
        if state.depth == 0:
            return generate_candidates(self.root.prefix)
        return _ordered_candidates(state.actor, state.alive_players, state.spoken_today)

    def _window(self, state):
        if state.actor is None or state.depth >= 2:
            raise ValueError('没有下一次可控行动，不能补造MAX节点')
        index = state.phase_order.index(state.actor)
        following = next((i for i in range(index + 1, len(state.phase_order))
                          if state.phase_order[i] in state.controlled_wolves), len(state.phase_order))
        return state.phase_order[index + 1:following], (
            state.phase_order[following] if following < len(state.phase_order) else None)

    def _advance(self, state, action, responses):
        if type(action) is not SpeechPlan or action not in self._candidates(state):
            raise ValueError('动作不是完整原候选中的SpeechPlan')
        speakers, next_actor = self._window(state)
        if (type(responses) is not tuple
                or any(type(r) is not tuple or len(r) != 2 or type(r[1]) is not str
                       or not r[1].strip() for r in responses)
                or tuple(r[0] for r in responses) != speakers):
            raise ValueError('人工公开响应必须准确覆盖同阶段未控制发言队列')
        spoken = set(state.spoken_today) | {state.actor} | set(speakers)
        return replace(state, history=state.history + ((action, responses),), actor=next_actor,
                       spoken_today=tuple(p for p in PLAYER_IDS if p in spoken), prefix=None)

    def validate(self, state):
        """从根和可见历史重建元数据，不接受隐藏世界ID或调用者指定的后继actor。"""
        if type(state) is not PlannerState or type(state.history) is not tuple or state.depth > 2:
            raise ValueError('非法规划状态')
        expected = self.root
        for step in state.history:
            if type(step) is not tuple or len(step) != 2:
                raise ValueError('非法可见响应历史')
            expected = self._advance(expected, *step)
        if state != expected or state.prefix is not expected.prefix:
            raise ValueError('状态不是同一真实根的SIMULATED公开后继')

    def actions(self, state):
        self.validate(state)
        return self._candidates(state)

    def response_speakers(self, state):
        self.validate(state)
        return self._window(state)[0]

    def advance(self, state, action, responses):
        self.validate(state)
        return self._advance(state, action, responses)


@dataclass(frozen=True)
class TransitionModel:
    """人工有限核：(可见历史,动作)映射到(概率,公开响应)；没有隐藏状态参数。"""
    root_binding: tuple
    kernel: object
    fixture_id: str

    def __post_init__(self):
        if not isinstance(self.fixture_id, str) or not self.fixture_id.strip():
            raise ValueError('需要明确人工转移fixture身份')
        table = dict(self.kernel)
        for key, rows in table.items():
            if (type(key) is not tuple or len(key) != 2 or type(key[0]) is not tuple
                    or type(key[1]) is not SpeechPlan or type(rows) is not tuple or not rows
                    or any(type(row) is not tuple or len(row) != 2
                           or type(row[0]) not in (int, float) or not math.isfinite(row[0])
                           or not 0 < row[0] <= 1 or type(row[1]) is not tuple for row in rows)):
                raise ValueError('非法人工概率核；仅登记正概率的有限公开分支')
            if math.fsum(p for p, _ in rows) != 1.0:
                raise ValueError('概率分母必须为1，不归一化或修补零分母')
            for _, responses in rows:
                if any(type(r) is not tuple or len(r) != 2 or not all(type(v) is str for v in r)
                       for r in responses):
                    raise ValueError('只允许玩家与公开响应文本，不接收隐藏状态字段')
            if len({responses for _, responses in rows}) != len(rows):
                raise ValueError('相同公开响应不能伪装成不同隐藏后继')
        object.__setattr__(self, 'kernel', MappingProxyType(table))

    def options(self, state, action, actions):
        actions.validate(state)
        if state.root_binding != self.root_binding or action not in actions.actions(state):
            raise ValueError('转移来源或动作不匹配')
        try:
            return self.kernel[(state.history, action)]
        except KeyError as error:
            raise ValueError('人工转移缺失；不返回默认后继') from error

    def sample(self, state, action, actions, rng):
        if type(rng) is not random.Random:
            raise TypeError('必须传入显式局部Random')
        rows = self.options(state, action, actions)
        draw = rng.random()
        for i, (_, responses) in enumerate(rows):
            if draw < math.fsum(p for p, _ in rows[:i + 1]):
                return actions.advance(state, action, responses)
        raise AssertionError('合法有限概率核应覆盖[0,1)')


@dataclass(frozen=True)
class ValueModel:
    """人工叶值表；不是D3风险、ToM代理、行动收益或胜率。"""
    root_binding: tuple
    values: object
    fixture_id: str

    def __post_init__(self):
        if not isinstance(self.fixture_id, str) or not self.fixture_id.strip():
            raise ValueError('需要明确人工价值fixture身份')
        table = dict(self.values)
        if any(type(h) is not tuple or type(v) not in (int, float)
               or not math.isfinite(v) or not 0 <= v <= 1 for h, v in table.items()):
            raise ValueError('人工叶值必须有限且位于[0,1]，不裁剪或填值')
        object.__setattr__(self, 'values', MappingProxyType(table))

    def evaluate(self, state):
        if state.root_binding != self.root_binding or state.origin != 'SIMULATED':
            raise ValueError('人工叶值必须绑定同根模拟历史')
        try:
            return float(self.values[state.history])
        except KeyError as error:
            raise ValueError('人工叶值缺失；不补零') from error


@dataclass(frozen=True)
class PlannerResult:
    algorithm: str
    root_binding: tuple
    selected_plan: SpeechPlan | None
    status: str
    action_statistics: tuple
    budget: int
    transition_calls: int
    transition_requests: int
    value_calls: int
    completed_simulations: int
    leaf_depth_counts: tuple
    requested_depth: int
    transition_fixture: str
    value_fixture: str

    @property
    def scope(self):
        return 'ARTIFICIAL_TRANSITION_AND_LEAF_VALUES_ONLY'

    @property
    def execution_authorized(self):
        return False

    @property
    def budget_used(self):
        return self.transition_calls + self.value_calls


class _BudgetExhausted(Exception):
    pass


class _Evaluation:
    """一条计数主线：采样后继计1，枚举k后继计k，每次叶值计1；不跨算法缓存。"""
    def __init__(self, planner, budget):
        if type(budget) is not int or budget < 0:
            raise ValueError('预算必须为非负整数')
        self.planner, self.budget = planner, budget
        self.transition_calls = self.transition_requests = self.value_calls = 0
        self.leaf_depths = {}

    @property
    def remaining(self):
        return self.budget - self.transition_calls - self.value_calls

    def reserve(self, count):
        if self.remaining < count:
            raise _BudgetExhausted

    def sample(self, state, action, rng):
        self.reserve(1)
        self.transition_calls += 1
        self.transition_requests += 1
        return self.planner.transition.sample(state, action, self.planner.actions, rng)

    def branches(self, state, action):
        rows = self.planner.transition.options(state, action, self.planner.actions)
        # options只查不可变人工表；真正生成k个后继前，必须有k份预算。
        self.reserve(len(rows))
        self.transition_calls += len(rows)
        self.transition_requests += 1
        return tuple((p, self.planner.actions.advance(state, action, responses)) for p, responses in rows)

    def leaf(self, state):
        self.reserve(1)
        self.value_calls += 1
        value = self.planner.value.evaluate(state)
        self.leaf_depths[state.depth] = self.leaf_depths.get(state.depth, 0) + 1
        return value


@dataclass
class _Edge:
    visits: int = 0
    total: float = 0.
    outcomes: dict = field(default_factory=dict)


@dataclass
class _Node:
    state: PlannerState
    visits: int = 0
    edges: dict = field(default_factory=dict)


@dataclass(frozen=True)
class MCTSPlanner:
    actions: LegalActionProvider
    transition: TransitionModel
    value: ValueModel

    def __post_init__(self):
        if (type(self.actions) is not LegalActionProvider or type(self.transition) is not TransitionModel
                or type(self.value) is not ValueModel
                or self.transition.root_binding != self.actions.root.root_binding
                or self.value.root_binding != self.actions.root.root_binding):
            raise ValueError('只接受同根人工有限转移及人工叶值表')

    def _result(self, algorithm, estimates, calls, depth, simulations=0):
        statistics = tuple((a, *estimates.get(a, (0, None))) for a in self.actions.actions(self.actions.root))
        complete = all(n > 0 and mean is not None for _, n, mean in statistics)
        # max保留第一个精确并列；没有选中的动作仍完整保留，不按样本均值裁剪。
        selected = max(statistics, key=lambda row: row[2])[0] if complete else None
        return PlannerResult(algorithm, self.actions.root.root_binding, selected,
            'COMPLETE' if complete else 'INCOMPLETE_ROOT_COVERAGE', statistics, calls.budget,
            calls.transition_calls, calls.transition_requests, calls.value_calls, simulations,
            tuple(sorted(calls.leaf_depths.items())), depth,
            self.transition.fixture_id, self.value.fixture_id)

    def plan(self, *, budget, seed, depth=2):
        """标准UCT四阶段；中间奖励为0，根按平均回报及原候选顺序选择。"""
        if type(seed) is not int or seed < 0 or type(depth) is not int or depth not in (1, 2):
            raise ValueError('需要非负局部seed及1或2层深度')
        calls = _Evaluation(self, budget)
        root = _Node(self.actions.root)
        rng = random.Random(seed)
        index = root.state.phase_order.index(root.state.actor)
        actual_depth = min(depth, sum(p in root.state.controlled_wolves
                                     for p in root.state.phase_order[index:]))
        simulations = 0
        # 同阶段无死亡/队列变化，完整模拟所需转移数可由根队列精确确定。
        while calls.remaining >= actual_depth + 1:
            node, state = root, root.state
            nodes, edges = [root], []
            while state.depth < depth and state.actor is not None:
                candidates = self.actions.actions(state)
                unvisited = next((a for a in candidates if a not in node.edges), None)
                if unvisited is not None:
                    action = unvisited
                    edge = node.edges.setdefault(action, _Edge())
                else:
                    action = max(candidates, key=lambda a: node.edges[a].total / node.edges[a].visits
                                 + math.sqrt(2 * math.log(node.visits) / node.edges[a].visits))
                    edge = node.edges[action]
                state = calls.sample(state, action, rng)  # chance只能采样，不能MAX
                if state.history not in edge.outcomes:
                    edge.outcomes[state.history] = _Node(state)
                node = edge.outcomes[state.history]
                nodes.append(node)
                edges.append(edge)
                if unvisited is not None:
                    break
            # 新扩展节点后的默认模拟，在全部合法动作上均匀采样。
            while state.depth < depth and state.actor is not None:
                action = rng.choice(self.actions.actions(state))
                state = calls.sample(state, action, rng)
            score = calls.leaf(state)
            for visited in nodes:
                visited.visits += 1
            for edge in edges:
                edge.visits += 1
                edge.total += score
            simulations += 1
        estimates = {a: (e.visits, e.total / e.visits) for a, e in root.edges.items()}
        return self._result('UCT', estimates, calls, depth, simulations)

    def _expectimax(self, *, budget, depth, algorithm):
        calls = _Evaluation(self, budget)

        def action_value(state, action):
            return math.fsum(p * state_value(successor) for p, successor in calls.branches(state, action))

        def state_value(state):
            if state.depth == depth or state.actor is None:
                return calls.leaf(state)
            return max(action_value(state, a) for a in self.actions.actions(state))

        estimates = {}
        try:
            for action in self.actions.actions(self.actions.root):
                estimates[action] = (1, action_value(self.actions.root, action))
        except _BudgetExhausted:
            pass  # 部分根动作不能冒充完整精确比较；保留实际计数且不选计划。
        return self._result(algorithm, estimates, calls, depth)

    def greedy(self, *, budget):
        """同一人工核与叶值的一步期望贪心，不调用额外未计费模型。"""
        return self._expectimax(budget=budget, depth=1, algorithm='GREEDY')

    def expectimax(self, *, budget):
        """同模型深度2精确枚举；预算不足只报告未完成，不称采样近似为精确值。"""
        return self._expectimax(budget=budget, depth=2, algorithm='EXPECTIMAX')
