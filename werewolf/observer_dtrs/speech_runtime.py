"""在已有 pending PRE 上执行显式启用的研究发言。

旧情境入口保留命中任一存活狼的事件。新受控入口使用原始报告包含当前
发言狼的事件。两者都复用公开 V1 证据、Actor、Parser 和原子提交。
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
from enum import Enum
import re
from time import perf_counter
from types import SimpleNamespace
from weakref import WeakSet

from werewolf.strategy.speech_planning import generate_candidates
from werewolf.speech.commit_workflow import realize_and_commit
from werewolf.speech.realization import expected_action
from werewolf.artifact_io import canonical_json_bytes, sha256_bytes
from werewolf.canonical_collection.public_history import PLAYER_IDS, freeze_public_event_history
from werewolf.canonical_collection.trajectory_evidence import BackendCallPurpose
from werewolf.observer_dtrs.context import contexts_for_pre
from werewolf.observer_dtrs.contextual_policy import contextual_loss, estimate_event
from werewolf.observer_dtrs.decision import LossSpec, Region, TiePolicy, candidate_losses, decide, forced_binary
from werewolf.observer_dtrs.probabilities import ProbabilityMatrix
from werewolf.observer_dtrs.estimation import FittedModel


from werewolf.observer_dtrs.speech_evidence import (
    ObserverCognition, ObserverSelection, SelectionMode, SpeechStrategy,
    process_cognition, select_observer, select_speech, strategy_plan,
)


VERSION = 'observer_dtrs_evidence_speech_research_v3_s1a'


class Comparison(str, Enum):
    STATIC_D3 = 'static_d3'
    CONTEXTUAL_LOSS = 'contextual_loss'
    CONTEXTUAL_SPEECH = 'contextual_loss_and_speech'


@dataclass(frozen=True)
class ResearchPolicy:
    static_loss: LossSpec
    reference_strategy: SpeechStrategy
    contextual_strategy: SpeechStrategy

    def __post_init__(self):
        if (not isinstance(self.static_loss, LossSpec)
                or self.static_loss.tie_policy not in (TiePolicy.TERMINAL_NEG_FIRST, TiePolicy.DEFER_FIRST)
                or self.static_loss not in candidate_losses(tie_policy=self.static_loss.tie_policy)):
            raise ValueError('static comparison requires a registered D3 loss and tie policy')
        if any(not isinstance(s, SpeechStrategy) for s in (self.reference_strategy, self.contextual_strategy)):
            raise TypeError('explicit reference and contextual speech strategies required')

    def to_record(self):
        return {'static_loss': self.static_loss.to_record(),
                'reference_strategy': self.reference_strategy.to_record(),
                'contextual_strategy': self.contextual_strategy.to_record()}


@dataclass(frozen=True)
class SyntheticPrediction:
    """Explicit test-only admission, never an online model provenance proof."""
    matrix: ProbabilityMatrix

    def probabilities(self, pre):
        if not isinstance(self.matrix, ProbabilityMatrix) or (
                self.matrix.game_id, self.matrix.boundary_id, self.matrix.prefix_digest) != (
                pre.game_id, pre.boundary_id, pre.prefix_digest):
            raise ValueError('synthetic prediction PRE mismatch')
        return self.matrix, {'scope': 'SYNTHETIC_TEST_ONLY'}


@dataclass(frozen=True)
class SealedQwen3Source:
    """Adapter for an ORIGINAL admitted predictor; no new loader or source bypass.

    Original source/runtime and full seal verification remain mandatory. Mainline
    cannot impersonate the historical Qwen3 checkout. Never invoked by CPU demo.
    """
    predictor: object
    expected_fit_digest: str
    expected_seal_digest: str

    def probabilities(self, pre):
        if any(not isinstance(s, str) or re.fullmatch('[0-9a-f]{64}', s) is None
               for s in (self.expected_fit_digest, self.expected_seal_digest)):
            raise ValueError('explicit frozen fit/seal SHA-256 pins required')
        from werewolf.tom.qwen3_final import SealedQwen3Predictor, _source_gate, verify_seal
        if type(self.predictor) is not SealedQwen3Predictor:
            raise TypeError('original SealedQwen3Predictor required, not arbitrary Q/provider')
        fit = self.predictor.fit
        if fit.digest != self.expected_fit_digest:
            raise ValueError('frozen fit identity mismatch')
        _source_gate(fit)  # Fail before prediction when current source/runtime differs.
        seal = verify_seal(fit)
        if (seal.manifest_digest != self.expected_seal_digest
                or self.predictor.seal.manifest_digest != seal.manifest_digest):
            raise ValueError('frozen seal identity mismatch')
        values = self.predictor.log_probabilities(pre).exp().tolist()  # Original PRE/capacity/simplex validation.
        matrix = ProbabilityMatrix(pre.game_id, pre.boundary_id, pre.prefix_digest,
                                   tuple(tuple(row) for row in values))
        return matrix, {'scope': 'ORIGINAL_SEALED_QWEN3', 'fit_digest': fit.digest,
                        'seal_digest': seal.manifest_digest,
                        'source': fit.manifest['protocol_inputs']['source']}


def _legal_team(pre, observation):
    actor = PLAYER_IDS.index(pre.current_speaker) + 1
    state = observation.get('authoritative_public_state', {})
    phase = 'speech_pk' if pre.public_temporal_state.phase.value == 'pk_discussion' else 'speech'
    if (observation.get('identity') != 'Werewolf' or observation.get('observer_id') != actor
            or observation.get('current_act_idx') != actor
            or observation.get('phase') != f'{pre.public_temporal_state.day}_day_{phase}'
            or state.get('alive_players') != [PLAYER_IDS.index(p)+1 for p in pre.alive_observer_ids]):
        raise ValueError('current acting wolf legal observation does not bind PRE')
    teams = [log.content.get('wolf_team') for log in observation.get('game_log', [])
             if log.event == 'werewolf_team_info']
    if (len(teams) != 1 or not isinstance(teams[0], list) or len(teams[0]) != 2
            or any(type(n) is not int or n not in range(1, 8) for n in teams[0])
            or len(set(teams[0])) != 2 or actor not in teams[0]):
        raise ValueError('one legally observed two-wolf team required')
    return tuple(PLAYER_IDS[n-1] for n in sorted(teams[0]))


@dataclass(frozen=True)
class SpeechExecution:
    audit: dict  # Private research audit; never forwarded to language or canonical public events.
    step_result: object


def execute_pending_speech(*, env, recorder, actor, source, model, protocol,
                           condition: Comparison, policy: ResearchPolicy,
                           selection_mode: SelectionMode = SelectionMode.EXPLICIT_THREE_WAY):
    """One decision -> original Actor/Parser/verified commit at one pending PRE.

    No role table, supervision reader, runner replacement, retry or fallback. In
    this research adapter the estimator must have controlled-fixture provenance;
    formal OOF-to-online transfer is not admitted by an in-memory model digest.
    """
    if (not isinstance(condition, Comparison) or not isinstance(policy, ResearchPolicy)
            or type(selection_mode) is not SelectionMode):
        raise TypeError('explicit comparison, ResearchPolicy and selection mode required')
    pending = recorder._pending
    if (recorder._env is not env or pending is None or pending.prefix is None
            or pending.handoff is None or pending.raw_action is not None):
        raise ValueError('existing untouched pending speech PRE required')
    pre = pending.prefix
    if freeze_public_event_history(env.public_events).digest != pre.public_event_history.digest:
        raise ValueError('stale pending PRE')
    team = _legal_team(pre, env.get_observation())  # ONLY the current actor's filtered view.
    selection = select_observer(pre, team)
    context = next(c for c in contexts_for_pre(pre, team) if c.observer_id == selection.observer_id)
    if context.wolf_count >= len(context.alive_player_ids) - context.wolf_count:
        raise ValueError('cannot speak after the Classic7 wolf victory condition')
    if type(source) not in (SyntheticPrediction, SealedQwen3Source):
        raise TypeError('validated frozen source or explicit synthetic input required')
    if not isinstance(model, FittedModel):
        raise TypeError('existing D2 FittedModel required')
    if model.record['source']['scope'] != 'controlled_fixture':
        raise ValueError('formal estimator online transfer requires a separately admitted source contract')
    matrix, provenance = source.probabilities(pre)
    loss = policy.static_loss if condition is Comparison.STATIC_D3 else contextual_loss(
        context, tie_policy=policy.static_loss.tie_policy)
    features, estimate, decision = estimate_event(context, matrix, model=model, protocol=protocol, loss=loss)
    strategy = policy.contextual_strategy if condition is Comparison.CONTEXTUAL_SPEECH else policy.reference_strategy
    evidence = process_cognition(pre, panel=(ObserverCognition(
        context, features, model.digest, estimate['level'], loss, decision),))
    chosen = select_speech(pre, evidence, candidates=generate_candidates(pre),
                           strategy=strategy, mode=selection_mode)
    plan, reason = chosen.plan, chosen.reason
    audit = {'version': VERSION, 'condition': condition.value, 'selection': asdict(selection),
             'prediction': {**provenance, 'matrix_digest': sha256_bytes(canonical_json_bytes(asdict(matrix)))},
             'policy': policy.to_record(), 'model_digest': model.digest, 'estimator_level': estimate['level'],
             'features': features.to_record(), 'loss': loss.to_record(), 'decision': decision.to_record(),
             'strategy': strategy.to_record(), 'mapping_reason': reason, 'public_payload': plan.public_payload(),
             'cognitive_processing': evidence.to_record(), 'candidate_selection': chosen.to_record(),
             'game_id': pre.game_id, 'boundary_id': pre.boundary_id, 'prefix_digest': pre.prefix_digest}
    if recorder._pending is not pending or freeze_public_event_history(env.public_events).digest != pre.public_event_history.digest:
        raise ValueError('PRE changed during planning')
    result = realize_and_commit(plan.public_payload(), env=env, actor=actor, recorder=recorder)
    return SpeechExecution(audit, result)


class ReportEventArm(str, Enum):
    TOM_ONLY = 'tom_only'
    TWO_WAY = 'tom_2wd'
    THREE_WAY = 'tom_3wd'


_REPORT_EVENT_OPPORTUNITY = 'FIRST_LIVE_WOLF_PRE_WITH_PUBLIC_SELF_ACCUSATION'


def _report_event_panel(pre, contexts, predict, *, m1_parameters, arm):
    from scripts import analyze_observer_dtrs_report_event as study
    from scripts.fit_observer_dtrs_report_event import legal_features

    matrix, provenance = predict(pre)
    if (type(matrix) is not ProbabilityMatrix
            or (matrix.game_id, matrix.boundary_id, matrix.prefix_digest) !=
            (pre.game_id, pre.boundary_id, pre.prefix_digest)):
        raise ValueError('Q 与当前 PRE 不一致')
    # 复用固定特征和校准公式，推理行不携带监督标签。
    rows = tuple(SimpleNamespace(sample_id=c.observer_id, features=x)
                 for c, _, x in legal_features(contexts, matrix))
    probabilities = study.rebuild_probabilities(rows, 'M1', m1_parameters)
    panel = {}
    for seat, probability in probabilities.items():
        if arm is ReportEventArm.THREE_WAY:
            region = decide(probability, study.LOSS).region
            directions = [] if region is Region.BND else [region.value]
        elif arm is ReportEventArm.TWO_WAY:
            region = forced_binary(probability, study.LOSS, tie_policy=study.BINARY_TIE_POLICY)
            directions = [region.value]
        else:
            directions = [Region.POS.value, Region.NEG.value]
        panel[seat] = {'probability': probability, 'adoptable_directions': directions}
    return {
        'model_digest': study.FIT_DIGEST,
        'loss': study.LOSS.to_record(),
        'prediction': {'source': provenance, 'matrix_digest': sha256_bytes(canonical_json_bytes(matrix.values))},
        'panel': panel,
    }


class ControlledReportEventSpeech:
    """在单个 Recorder 的现有 PRE 上验证新事件发言，不提供正式对局入口。

    每个受控 attempt 由调用方装配一个实例。返回 None 表示沿用原策略，
    已尝试的失败会保留审计并停止。Q 回调仅是受控输入，真实来源由其调用端验证。
    """

    _attempted_recorders = WeakSet()

    def __init__(self, recorder, *, arm, assignment_ref, predict, m1_parameters):
        if (type(arm) is not ReportEventArm or type(assignment_ref) is not str or not assignment_ref.strip()
                or not callable(predict)):
            raise ValueError('受控发言需要显式分臂、分配引用及 PRE 预测接口')
        self._recorder = recorder
        self._arm = arm
        self._assignment_ref = assignment_ref
        self._predict = predict
        self._m1_parameters = m1_parameters
        self._audit = None

    @property
    def audit(self):
        return deepcopy(self._audit)

    def execute_pending(self, *, env, actor, handoff):
        return self._execute_pending(env=env, actor=actor, handoff=handoff)

    def _execute_pending(self, *, env, actor, handoff, record_intent=None):
        recorder = self._recorder
        if recorder._env is not env:
            raise ValueError('Recorder 与当前环境不一致')
        if self._audit is not None:
            if self._audit['status'] != 'COMMITTED':
                raise ValueError('受控机会已尝试且未确认成功')
            return None
        if recorder in self._attempted_recorders:
            raise ValueError('此 Recorder 的受控机会已尝试')

        pre = recorder.public_pre_for_handoff(handoff)
        pending = recorder._pending
        if freeze_public_event_history(env.public_events).digest != pre.public_event_digest:
            raise ValueError('pending PRE 已失效')
        observation = env.get_observation()
        if observation.get('identity') != 'Werewolf':
            return None
        team = _legal_team(pre, observation)
        contexts = contexts_for_pre(pre, team)
        if not contexts or contexts[0].wolf_count >= len(contexts):
            return None
        selected = select_observer(pre, team)
        if selected.relation != 'SELF_ACCUSED':
            return None
        candidates = generate_candidates(pre)

        self._attempted_recorders.add(recorder)
        audit = self._audit = {
            'scope': 'CONTROLLED_REPORT_EVENT_SPEECH',
            'status': 'ATTEMPTED',
            'arm': self._arm.value,
            'assignment_ref': self._assignment_ref,
            'game_id': pre.game_id, 'attempt_id': recorder.claim.attempt_id,
            'boundary_id': pre.boundary_id, 'prefix_digest': pre.prefix_digest, 'step_index': pre.step_index,
            'handoff_digest': handoff.handoff_digest, 'actor': pre.current_speaker,
            'selection': asdict(selected), 'opportunity_rule': _REPORT_EVENT_OPPORTUNITY,
        }
        before = len(recorder.submitted_gameplay_actions)
        event_index = len(env.public_events)
        try:
            evidence = _report_event_panel(pre, contexts, self._predict,
                                           m1_parameters=self._m1_parameters, arm=self._arm)
            audit.update(evidence)
            public_events = pre.public_event_history.to_records()
            plan_context = {
                'speaker': pre.current_speaker, 'alive_players': pre.alive_observer_ids,
                'known_wolf_team': team, 'public_events': public_events,
                'report_predictions': evidence['panel'],
                'candidates': [plan.public_payload() for plan in candidates],
            }
            basis_refs = (
                'independent', 'keep_both',
                *(f"public:{event['event_id']}" for event in public_events),
                *(f'prediction:{seat}:{direction}' for seat, row in evidence['panel'].items()
                  for direction in row['adoptable_directions']),
            )
            with recorder.call_audit.action_context(acting_player_id=observation['current_act_idx'],
                    boundary_id=pre.boundary_id, is_public_speech=False):
                index, basis_ref = actor.select_report_event_plan(plan_context, basis_refs=basis_refs)
            payload = candidates[index].public_payload()
            audit.update(candidate_index=index, basis_ref=basis_ref, public_payload=payload)
            if recorder._pending is not pending:
                raise ValueError('发言选择期间 pending PRE 已变化')
            if record_intent is not None:
                record_intent(self.audit)
            step = realize_and_commit(payload, env=env, actor=actor, recorder=recorder)
            if recorder._pending is not None or len(recorder.submitted_gameplay_actions) != before + 1:
                raise ValueError('原生提交数量或 pending 状态无法确认')
            submitted = recorder.submitted_gameplay_actions[-1]
            annotation = env.speech_annotations[-1]
            event = env.public_events[event_index]
            if ((submitted.boundary_id, submitted.step_index, submitted.actor_id) !=
                    (pre.boundary_id, pre.step_index, pre.current_speaker)
                    or submitted.speaker_pre_belief_handoff != handoff
                    or annotation.event_id not in submitted.resulting_public_event_ids
                    or annotation.event_id != event['event_id']
                    or event['event_type'] != 'public_speech'
                    or submitted.action_payload.to_value() != event['raw_text']
                    or annotation.actions != (expected_action(payload, pre.current_speaker),)):
                raise ValueError('原生提交与该次计划的归属不一致')
        except Exception as error:
            audit['status'] = ('FAILED' if recorder._pending is pending
                               and len(recorder.submitted_gameplay_actions) == before
                               and len(env.public_events) == event_index else 'COMMIT_UNKNOWN')
            audit['error_type'] = type(error).__name__
            raise
        audit['commit'] = {'action_id': submitted.action_id, 'event_id': annotation.event_id,
                           'annotation_digest': annotation.annotation_digest}
        audit['status'] = 'COMMITTED'
        return SpeechExecution(self.audit, step)


class ReportEventGamePilot:
    """将固定分臂接到原 Canonical PRE hook，沿用 Recorder 和单次提交。"""

    def __init__(self, *, recorder, agents, predictor, arm, probability, m1_parameters, record_evidence):
        from werewolf.observer_dtrs.final_tom_client import CapsuleGameplayPredictor, REPORT_EVENT_PILOT_SCOPE

        if (type(predictor) is not CapsuleGameplayPredictor or type(arm) is not ReportEventArm
                or len(agents) != len(PLAYER_IDS) or not callable(record_evidence)):
            raise ValueError('整局试运行需要固定分臂、研究专用预测器、原 Agent 和证据写入入口')
        predictor.require_report_event_ready()
        self._recorder = recorder
        self._agents = tuple(agents)
        self._record = record_evidence
        self._scope = REPORT_EVENT_PILOT_SCOPE
        self._q_requests, self._q_successes, self._q_seconds = 0, 0, 0.
        self._started = perf_counter()

        def predict(pre):
            start = perf_counter()
            self._q_requests += 1
            try:
                result = predictor.predict_report_event(pre)
                self._q_successes += 1
                return result
            finally:
                self._q_seconds += perf_counter() - start

        self._assignment = {'scope': self._scope, 'game_id': recorder.game_id,
                            'attempt_id': recorder.claim.attempt_id, 'seed': recorder.claim.seed,
                            'plan_digest': recorder.plan.plan_digest, 'arm': arm.value,
                            'probability': probability, 'opportunity_rule': _REPORT_EVENT_OPPORTUNITY}
        reference = sha256_bytes(canonical_json_bytes(self._assignment))
        self._assignment['assignment_ref'] = reference
        self._record('assignment', dict(self._assignment))
        self._speech = ControlledReportEventSpeech(recorder, arm=arm, assignment_ref=reference,
                                                  predict=predict, m1_parameters=m1_parameters)

    def _record_intent(self, audit):
        self._record('intent', {**audit, 'scope': self._scope})

    def handle_pre(self, *, env, recorder, call_audit, observation, handoff):
        if recorder is not self._recorder or call_audit is not recorder.call_audit:
            raise ValueError('分配与 Canonical Recorder 不匹配')
        try:
            executed = self._speech._execute_pending(env=env, actor=self._agents[observation['current_act_idx'] - 1],
                                                     handoff=handoff, record_intent=self._record_intent)
        except Exception:
            if self._speech.audit is not None:
                self._record_execution(self._speech.audit)
            raise
        if executed is not None:
            self._record_execution(executed.audit)
            return executed.step_result
        return None

    def _record_execution(self, audit):
        boundary = audit.get('boundary_id')
        calls = tuple(call for call in self._recorder.call_audit.records if boundary and call.boundary_id == boundary)
        self._record('execution', {**audit, 'scope': self._scope, 'q_requests': self._q_requests,
                                   'q_successes': self._q_successes, 'q_seconds': self._q_seconds,
                                   'plan_calls': sum(c.purpose is BackendCallPurpose.GAMEPLAY_ACTION for c in calls),
                                   'actor_calls': sum(c.purpose is BackendCallPurpose.SPEECH_GENERATION for c in calls),
                                   'parser_calls': sum(c.purpose is BackendCallPurpose.SPEECH_PERCEPTION for c in calls)})

    def after_step(self, *, env, done, info):
        """保留 Canonical 的普通后续策略。"""

    def after_game(self, *, env, winner):
        if self._speech.audit is None:
            self._record_execution({**self._assignment, 'status': 'NO_COMMON_OPPORTUNITY'})
        # 完整游戏胜者由随后进行的原生 Replay 核验。
        summary = self._recorder.call_audit.summary()
        self._record('runtime', {**self._assignment, 'elapsed_seconds': perf_counter() - self._started,
                                 'backend_calls': summary.used_calls, 'backend_retries': summary.retry_calls})
