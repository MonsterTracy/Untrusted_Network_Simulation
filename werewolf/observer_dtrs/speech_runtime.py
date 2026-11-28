"""通过现有 pending PRE 与原生提交执行三支认知及历史研究发言。"""
from dataclasses import asdict, dataclass
from enum import Enum
import json
import re
from time import perf_counter
from types import SimpleNamespace

from werewolf.strategy.speech_planning import generate_candidates
from werewolf.speech.commit_workflow import commit_public_speech, realize_and_commit
from werewolf.speech.realization import expected_action
from werewolf.artifact_io import canonical_json_bytes, sha256_bytes
from werewolf.canonical_collection.call_audit import AuditedBackend
from werewolf.canonical_collection.public_history import PLAYER_IDS, freeze_public_event_history
from werewolf.canonical_collection.trajectory_evidence import BackendCallPurpose
from werewolf.game_runtime import UNKNOWN, summarize_tom_integration
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


_REPORT_EVENT_OPPORTUNITY = 'ALL_LEGAL_WOLF_SPEECH_PRE'


def _report_event_q_matches(pre, matrix):
    return (type(matrix) is ProbabilityMatrix
            and (matrix.game_id, matrix.boundary_id, matrix.prefix_digest) ==
            (pre.game_id, pre.boundary_id, pre.prefix_digest))


def report_event_probabilities(pre, contexts, matrix, *, m1_parameters):
    from scripts import analyze_observer_dtrs_report_event as study
    from scripts.fit_observer_dtrs_report_event import legal_features

    if not _report_event_q_matches(pre, matrix):
        raise ValueError('Q 与当前 PRE 不一致')
    # 复用固定特征和校准公式，推理行不携带监督标签。
    rows = tuple(SimpleNamespace(sample_id=c.observer_id, features=x)
                 for c, _, x in legal_features(contexts, matrix))
    return study.rebuild_probabilities(rows, 'M1', m1_parameters)


def report_event_directions(probabilities, arm):
    from scripts import analyze_observer_dtrs_report_event as study

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
    return panel


def report_event_plan_input(pre, team, panel, candidates):
    public_events = pre.public_event_history.to_records()
    context = {
        'speaker': pre.current_speaker, 'alive_players': pre.alive_observer_ids,
        'known_wolf_team': team, 'public_events': public_events,
        'report_predictions': panel,
        'candidates': [plan.public_payload() for plan in candidates],
    }
    basis_refs = (
        'independent', 'keep_both',
        *(f"public:{event['event_id']}" for event in public_events),
        *(f'prediction:{seat}:{direction}' for seat, row in panel.items()
          for direction in row['adoptable_directions']),
    )
    return context, basis_refs


def _report_event_panel(pre, contexts, prediction, *, m1_parameters, arm):
    from scripts import analyze_observer_dtrs_report_event as study

    matrix, provenance = prediction
    probabilities = report_event_probabilities(pre, contexts, matrix, m1_parameters=m1_parameters)
    return {
        'model_digest': study.FIT_DIGEST,
        'loss': study.LOSS.to_record(),
        'prediction': {'source': provenance, 'matrix_digest': sha256_bytes(canonical_json_bytes(matrix.values))},
        'panel': report_event_directions(probabilities, arm),
    }


def _report_event_decision_receipts(calls, actor, context):
    """对齐认知后端请求中的完整输入，保留可确认与未确认的真实调用 ID。"""
    confirmed, unconfirmed = [], []
    if context is None:
        return confirmed, unconfirmed
    expected = canonical_json_bytes(context)
    for call in calls:
        if call.purpose is not BackendCallPurpose.GAMEPLAY_ACTION or call.observer_id != actor:
            continue
        request = call.private_payload.to_value()
        try:
            kwargs = request['kwargs']
            response_format = kwargs.get('response_format') or {}
            if response_format.get('json_schema', {}).get('name') != 'report_event_plan':
                continue
            content = kwargs['messages'][0]['content']
            actual = canonical_json_bytes(json.loads(content.rsplit('\n\n', 1)[-1]))
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            unconfirmed.append(call.call_id)
            continue
        if actual == expected:
            receipts = confirmed if request.get('response') is not None else unconfirmed
            receipts.append(call.call_id)
    return confirmed, unconfirmed


class ReportEventGamePolicy:
    """在整局内用同一实验臂处理每个合法狼人发言 PRE。"""

    def __init__(self, *, recorder, agents, predictor, arm, probability, m1_parameters, record_evidence):
        from werewolf.observer_dtrs.final_tom_client import CapsuleGameplayPredictor, REPORT_EVENT_PILOT_SCOPE

        if (type(predictor) is not CapsuleGameplayPredictor or type(arm) is not ReportEventArm
                or len(agents) != len(PLAYER_IDS) or not callable(record_evidence)):
            raise ValueError('整局策略需要固定分臂、研究专用预测器、原 Agent 和证据写入入口')
        predictor.require_report_event_ready()
        self._recorder = recorder
        self._agents = tuple(agents)
        self._predictor = predictor
        self._arm = arm
        self._m1_parameters = m1_parameters
        self._record = record_evidence
        self._q_requests, self._q_successes, self._q_seconds = 0, 0, 0.
        self._tom_integration = []
        self._started = perf_counter()

        self._assignment = {'scope': REPORT_EVENT_PILOT_SCOPE, 'game_id': recorder.game_id,
                            'attempt_id': recorder.claim.attempt_id, 'seed': recorder.claim.seed,
                            'plan_digest': recorder.plan.plan_digest, 'arm': arm.value,
                            'probability': probability, 'opportunity_rule': _REPORT_EVENT_OPPORTUNITY}
        self._assignment['assignment_ref'] = sha256_bytes(canonical_json_bytes(self._assignment))
        self._record('assignment', dict(self._assignment))

    def handle_pre(self, *, env, recorder, call_audit, observation, handoff):
        if recorder is not self._recorder or call_audit is not recorder.call_audit or recorder._env is not env:
            raise ValueError('分配与当前 Canonical Recorder 或环境不匹配')
        if env.phase not in ('speech', 'speech_pk') or observation.get('identity') != 'Werewolf':
            return None

        pre = recorder.public_pre_for_handoff(handoff)
        pending = recorder._pending
        if freeze_public_event_history(env.public_events).digest != pre.public_event_digest:
            raise ValueError('pending PRE 已失效')
        audit = {
            'scope': self._assignment['scope'], 'status': 'ATTEMPTED',
            'arm': self._arm.value, 'assignment_ref': self._assignment['assignment_ref'],
            'game_id': pre.game_id, 'attempt_id': self._assignment['attempt_id'],
            'boundary_id': pre.boundary_id, 'prefix_digest': pre.prefix_digest, 'step_index': pre.step_index,
            'handoff_digest': handoff.handoff_digest, 'acting_wolf': pre.current_speaker,
            'day': pre.public_temporal_state.day, 'phase': pre.public_temporal_state.phase.value,
            'q_requests': 0, 'q_successes': 0, 'q_seconds': 0.,
        }
        before = len(recorder.submitted_gameplay_actions)
        event_index = len(env.public_events)
        legal_prediction = False
        plan_context = None
        try:
            team = _legal_team(pre, observation)
            contexts = contexts_for_pre(pre, team)
            if not contexts:
                raise ValueError('狼人发言 PRE 没有合格的存活非狼观察者')
            candidates = generate_candidates(pre)
            actor = self._agents[observation['current_act_idx'] - 1]
            # 与 Capsule 客户端的 PRE 请求 ID 一致，即使调用未返回也可关联 Owner 日志。
            audit['q_request_id'] = sha256_bytes(canonical_json_bytes({
                'game_id': pre.game_id, 'boundary_id': pre.boundary_id, 'prefix_digest': pre.prefix_digest,
            }))
            started = perf_counter()
            audit['q_requests'] = 1
            try:
                prediction = self._predictor.predict_report_event(pre)
            finally:
                audit['q_seconds'] = perf_counter() - started
            audit['q_successes'] = 1
            matrix, _ = prediction
            legal_prediction = _report_event_q_matches(pre, matrix)
            evidence = _report_event_panel(pre, contexts, prediction,
                                           m1_parameters=self._m1_parameters, arm=self._arm)
            audit.update(evidence)
            plan_context, basis_refs = report_event_plan_input(pre, team, evidence['panel'], candidates)
            with call_audit.action_context(acting_player_id=observation['current_act_idx'],
                    boundary_id=pre.boundary_id, is_public_speech=False):
                index, basis_ref = actor.select_report_event_plan(plan_context, basis_refs=basis_refs)
            payload = candidates[index].public_payload()
            audit.update(candidate_index=index, basis_ref=basis_ref, public_payload=payload)
            if recorder._pending is not pending:
                raise ValueError('发言选择期间 pending PRE 已变化')
            self._record('intent', dict(audit))
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
            self._record_execution(audit, legal_prediction, plan_context)
            raise
        audit['commit'] = {'action_id': submitted.action_id, 'event_id': annotation.event_id,
                           'annotation_digest': annotation.annotation_digest}
        audit['status'] = 'COMMITTED'
        self._record_execution(audit, legal_prediction, plan_context)
        return step

    def _record_execution(self, audit, legal_prediction, plan_context):
        self._q_requests += audit['q_requests']
        self._q_successes += audit['q_successes']
        self._q_seconds += audit['q_seconds']
        calls = tuple(call for call in self._recorder.call_audit.records if call.boundary_id == audit['boundary_id'])
        confirmed, unconfirmed = _report_event_decision_receipts(calls, audit['acting_wolf'], plan_context)
        actor = self._agents[PLAYER_IDS.index(audit['acting_wolf'])]
        backend = getattr(actor, 'backend', None)
        audited = isinstance(backend, AuditedBackend) and backend._audit is self._recorder.call_audit
        if confirmed:
            integrated = 1
        elif plan_context is not None and (unconfirmed or not audited):
            integrated = UNKNOWN
        else:
            integrated = 0
        counts = {'A': audit['q_requests'], 'P': int(legal_prediction), 'S': integrated}
        self._tom_integration.append(counts)
        self._record('execution', {**audit,
                                   'tom_integration': counts,
                                   'decision_call_ids': confirmed,
                                   'unconfirmed_decision_call_ids': unconfirmed,
                                   'plan_calls': sum(c.purpose is BackendCallPurpose.GAMEPLAY_ACTION for c in calls),
                                   'actor_calls': sum(c.purpose is BackendCallPurpose.SPEECH_GENERATION for c in calls),
                                   'parser_calls': sum(c.purpose is BackendCallPurpose.SPEECH_PERCEPTION for c in calls)})

    def after_step(self, *, env, done, info):
        """保留 Canonical 的普通后续策略。"""

    def after_game(self, *, env, winner):
        # 完整游戏胜者由随后进行的原生 Replay 核验。
        summary = self._recorder.call_audit.summary()
        self._record('runtime', {**self._assignment, 'elapsed_seconds': perf_counter() - self._started,
                                 'q_requests': self._q_requests, 'q_successes': self._q_successes,
                                 'q_seconds': self._q_seconds,
                                 'tom_integration': summarize_tom_integration(self._tom_integration),
                                 'backend_calls': summary.used_calls, 'backend_retries': summary.retry_calls})


class SpeechDecisionUnavailable(RuntimeError):
    """当前狼人发言没有可执行的正式选择。"""

    def __init__(self, status):
        self.status = status
        super().__init__(f"ToM+3WD 无法完成发言决策：{status}")


class ThreeWaySpeechPolicy:
    """仅在合法狼人发言接入五层方法，交回原生步结果。"""

    def __init__(self, *, agents, predictor, model, rule, cognitive, delta, record_evidence):
        from werewolf.observer_dtrs.action_selection import parse_delta

        self._delta = parse_delta(delta)
        if any(component is None for component in (predictor, model, rule, cognitive)):
            raise SpeechDecisionUnavailable("frozen_model_missing")
        self._agents = tuple(agents)
        self._predictor = predictor
        self._model = model
        self._rule = rule
        self._cognitive = cognitive
        self._record = record_evidence

    def handle_pre(self, *, env, recorder, call_audit, observation, handoff):
        if env.phase not in ('speech', 'speech_pk') or observation.get('identity') != 'Werewolf':
            return None

        from werewolf.observer_dtrs.action_selection import select_three_way
        from werewolf.observer_dtrs.report_state_decision import decide_pre
        from werewolf.observer_dtrs.report_state_probability import predict_pre
        from werewolf.observer_dtrs.speech_consequences import score_candidates

        if call_audit is not recorder.call_audit or recorder._env is not env:
            raise ValueError('正式三支发言要求当前 Canonical Recorder 与调用记录')
        pre = recorder.public_pre_for_handoff(handoff)
        audit = {'method': ReportEventArm.THREE_WAY.value, 'game_id': pre.game_id, 'attempt_id': recorder.claim.attempt_id,
                 'boundary_id': pre.boundary_id, 'prefix_digest': pre.prefix_digest, 'step_index': pre.step_index,
                 'acting_wolf': pre.current_speaker, 'stage': 'pre', 'delta_exact': str(self._delta)}
        try:
            if freeze_public_event_history(env.public_events).digest != pre.public_event_digest:
                raise ValueError('正式三支发言的 pending PRE 已变化')
            team = _legal_team(pre, observation)
            seat = observation['current_act_idx']
            actor = self._agents[seat - 1]
            audit['stage'] = 'current_final_tom'
            prediction = self._predictor.predict(pre)
            q = ProbabilityMatrix(prediction.game_id, prediction.boundary_id, prediction.prefix_digest,
                                  prediction.probabilities)
            audit['stage'] = 'cognition'
            decisions = decide_pre(predict_pre(pre, team, q, self._model), self._rule)
            audit.update(current_q=q.values, cognition=decisions)

            audit['stage'] = 'candidate_generation'
            with call_audit.action_context(acting_player_id=seat, boundary_id=pre.boundary_id, is_public_speech=True):
                batch = actor.generate_speech_candidates(pre, known_wolf_ids=team, q=q)
            audit.update(topic=batch.topic, generated_candidates=[asdict(candidate) for candidate in batch.candidates])
            audit['stage'] = 'basis_verification'
            with call_audit.context(purpose=BackendCallPurpose.RUNTIME, operation_id=f'{pre.boundary_id}-candidate-bases',
                                    boundary_id=pre.boundary_id, observer_id=pre.current_speaker):
                verification = actor.verify_speech_candidate_bases(batch, known_wolf_ids=team)
            audit['basis_verification'] = verification

            def candidate_perception(index, speech):
                return call_audit.candidate_perception_context(boundary_id=pre.boundary_id, speaker_id=seat,
                                                               candidate_index=index, speech=speech)

            audit['stage'] = 'social_value'
            social = score_candidates(batch, team, q, self._model, perceiver=env.speech_perceiver,
                                      predictor=self._predictor, perception_context=candidate_perception)
            audit['stage'] = 'cognitive_value'
            information = self._cognitive.predict_candidates(batch, team, q)
            audit['stage'] = 'selection'
            decision = select_three_way(batch, decisions, verification, social, information, delta=self._delta)
            audit['decision'] = decision
            selection = decision['selection']
            if selection['status'] != 'selected':
                raise SpeechDecisionUnavailable(selection['status'])
            selected = decision['candidates'][selection['selected_index']]

            audit['stage'] = 'commit'
            step = commit_public_speech(selected['text'], pre=pre, env=env, recorder=recorder, handoff=handoff,
                                        scored_candidate=selected)
            annotation = env.speech_annotations[-1]
            audit['commit'] = {'action_id': recorder.submitted_gameplay_actions[-1].action_id,
                               'event_id': annotation.event_id, 'annotation_digest': annotation.annotation_digest}
            audit['stage'] = 'post_speech_final_tom'
            audit['post_speech'] = self._predictor.observe_after_speech(
                freeze_public_event_history(env.public_events), env.speech_annotations, selected['social_prediction'],
            )
        except Exception as error:
            audit.update(status='failed', error_type=type(error).__name__, error_message=str(error))
            self._record(audit)
            raise
        audit.update(status='committed', stage='complete')
        self._record(audit)
        return step

    def after_step(self, *, env, done, info):
        """公开轨迹由原生环境更新，后续狼人 PRE 会重新读取。"""
