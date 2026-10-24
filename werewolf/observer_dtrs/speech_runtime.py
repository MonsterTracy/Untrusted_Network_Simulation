"""Opt-in research speech at an existing pending PRE, not a campaign runner.

Selection uses public V1 evidence, not Q rankings. Y still concerns ANY live wolf.
Only the wolf's filtered observation supplies team knowledge. Language, perception
and atomic commit remain owned by the existing verified-speech implementation.
"""
from dataclasses import asdict, dataclass
from enum import Enum
import re

from scripts.speech_planning import PlanningAction as A, SpeechPlan, generate_candidates
from scripts.verified_speech_commit import realize_and_commit
from werewolf.artifact_io import canonical_json_bytes, sha256_bytes
from werewolf.canonical_collection.public_history import PLAYER_IDS, freeze_public_event_history
from werewolf.observer_dtrs.context import contexts_for_pre
from werewolf.observer_dtrs.contextual_policy import contextual_loss, estimate_event
from werewolf.observer_dtrs.decision import LossSpec, Region, TiePolicy, candidate_losses
from werewolf.observer_dtrs.probabilities import ProbabilityMatrix
from werewolf.observer_dtrs.estimation import FittedModel


VERSION = 'observer_dtrs_evidence_speech_research_v2'


@dataclass(frozen=True)
class ObserverSelection:
    observer_id: str
    reason: str
    event_id: str | None
    annotation_digest: str | None
    relation: str  # Public claim direction, never the event Y or a role fact.


def select_observer(pre, known_wolf_ids):
    """Latest same-day speech per living non-wolf; direction, recency, seat order.

    Priority: explicit current-actor accusation; teammate accusation; any annotated
    speech; any unparsed/failed speech; no speech. No label, raw-text reparsing,
    Q averaging, predicted probability ranking or past-day resurrection.
    """
    contexts = contexts_for_pre(pre, known_wolf_ids)
    if not contexts:
        raise ValueError('no legal wolf-speech observers')
    generate_candidates(pre)  # Also validate actual public speaking/PK order.
    day = pre.public_temporal_state.day
    annotations = {a.event_id: a for a in pre.v1_annotations}
    ranked = []
    for context in contexts:
        events = [e for e in pre.public_event_history.events if e.event_type == 'public_speech'
                  and e.speaker == context.observer_id and e.temporal_state.day == day]
        last = events[-1] if events else None
        ann = annotations.get(last.event_id) if last else None
        targets = {a.object for a in ann.actions if a.action == 'point_as_werewolf'} if ann and ann.is_successful else set()
        if ann and ann.is_successful:
            targets -= {a.object for a in ann.actions if a.action == 'point_as_non_werewolf'}
        relation = ('SELF_ACCUSED' if context.acting_wolf in targets else
                    'TEAM_ACCUSED' if targets.intersection(context.alive_wolf_ids) else 'NO_EXPLICIT_ACCUSATION')
        priority = (0 if relation == 'SELF_ACCUSED' else 1 if relation == 'TEAM_ACCUSED' else
                    2 if ann and ann.is_successful else 3 if last else 4)
        reason = ('PUBLIC_SELF_ACCUSATION', 'PUBLIC_TEAM_ACCUSATION', 'LATEST_ANNOTATED_SPEECH',
                  'LATEST_UNINTERPRETED_SPEECH', 'CANONICAL_SEAT_NO_EVIDENCE')[priority]
        selection = ObserverSelection(context.observer_id, reason, last.event_id if last else None,
                                      ann.annotation_digest if ann else None, relation)
        ranked.append(((priority, -last.event_index if last else 0, PLAYER_IDS.index(context.observer_id)), selection))
    return min(ranked, key=lambda pair: pair[0])[1]


@dataclass(frozen=True)
class SpeechStrategy:
    """Explicit hypothesis knobs; action eligibility and evidence gates are fixed."""
    positive_self: A
    positive_team: A
    boundary: A
    negative: A

    def __post_init__(self):
        for name in ('positive_self', 'positive_team', 'boundary', 'negative'):
            choices = (A.NO_COMMITMENT, A.OPPOSE, A.SELF_DEFEND) if name == 'positive_self' else (A.NO_COMMITMENT, A.OPPOSE)
            if not isinstance(getattr(self, name), A) or getattr(self, name) not in choices:
                raise ValueError('unsupported evidence-speech hypothesis')

    def to_record(self):
        return {k: v.value for k, v in asdict(self).items()}


def strategy_plan(pre, selection, decision, strategy):
    if not isinstance(strategy, SpeechStrategy):
        raise TypeError('explicit SpeechStrategy required')
    if selection.relation == 'NO_EXPLICIT_ACCUSATION':
        return SpeechPlan(A.NO_COMMITMENT), 'NO_DIRECTIONAL_PUBLIC_EVIDENCE'
    if decision.region is Region.POS:
        action = strategy.positive_self if selection.relation == 'SELF_ACCUSED' else strategy.positive_team
    else:
        action = strategy.boundary if decision.region is Region.BND else strategy.negative
    plan = SpeechPlan(action, selection.observer_id if action is A.OPPOSE else None)
    if plan not in generate_candidates(pre):
        raise ValueError('strategy intent is not eligible at this PRE')
    return plan, 'CONFIGURED_' + decision.region.value + '_' + selection.relation


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
                           condition: Comparison, policy: ResearchPolicy):
    """One decision -> original Actor/Parser/verified commit at one pending PRE.

    No role table, supervision reader, runner replacement, retry or fallback. In
    this research adapter the estimator must have controlled-fixture provenance;
    formal OOF-to-online transfer is not admitted by an in-memory model digest.
    """
    if not isinstance(condition, Comparison) or not isinstance(policy, ResearchPolicy):
        raise TypeError('explicit comparison condition and ResearchPolicy required')
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
    plan, reason = strategy_plan(pre, selection, decision, strategy)
    audit = {'version': VERSION, 'condition': condition.value, 'selection': asdict(selection),
             'prediction': {**provenance, 'matrix_digest': sha256_bytes(canonical_json_bytes(asdict(matrix)))},
             'policy': policy.to_record(), 'model_digest': model.digest, 'estimator_level': estimate['level'],
             'features': features.to_record(), 'loss': loss.to_record(), 'decision': decision.to_record(),
             'strategy': strategy.to_record(), 'mapping_reason': reason, 'public_payload': plan.public_payload(),
             'game_id': pre.game_id, 'boundary_id': pre.boundary_id, 'prefix_digest': pre.prefix_digest}
    if recorder._pending is not pending or freeze_public_event_history(env.public_events).digest != pre.public_event_history.digest:
        raise ValueError('PRE changed during planning')
    result = realize_and_commit(plan.public_payload(), env=env, actor=actor, recorder=recorder)
    return SpeechExecution(audit, result)
