"""Pure cognitive processing G and existing-intent selection pi.

All records are PRIVATE research audit. No label, sidecar, model fit, language
call, env or commit is accepted. Cognitive risks are not speech utilities.
"""
from dataclasses import asdict, dataclass
from enum import Enum
import re

from scripts.speech_planning import PlanningAction as A, SpeechPlan, generate_candidates
from scripts.speech_versions import CANDIDATE_ORDER_VERSION
from werewolf.artifact_io import canonical_json_bytes, sha256_bytes
from werewolf.canonical_collection.public_history import PLAYER_IDS
from werewolf.observer_dtrs.context import ObserverContext, contexts_for_pre
from werewolf.observer_dtrs.decision import Decision, LossSpec, Region, decide
from werewolf.observer_dtrs.estimation import condition
from werewolf.observer_dtrs.features import ObserverFeatures

VERSION = 'observer_dtrs_cognitive_speech_interface_s1a_v1'


@dataclass(frozen=True)
class ObserverSelection:
    observer_id: str
    reason: str
    event_id: str | None
    annotation_digest: str | None
    relation: str  # Public claim direction, never Y or a role fact.


def select_observer(pre, known_wolf_ids):
    """V2 public-only priority: direction, latest same-day speech, seat order."""
    contexts = contexts_for_pre(pre, known_wolf_ids)
    if not contexts:
        raise ValueError('no legal wolf-speech observers')
    generate_candidates(pre)
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
    """The unchanged V2 hypothesis knobs; never a universal meaning of a region."""
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


class CognitiveHandling(str, Enum):
    CONSIDER_HIT = 'CONSIDER_EFFECTIVE_SUPPORT_HIT'
    KEEP_BOTH = 'KEEP_BOTH_EVENT_HYPOTHESES'
    REJECT_HIT_PREMISE = 'DO_NOT_USE_HIT_PREMISE'


class SelectionMode(str, Enum):
    EXPLICIT_THREE_WAY = 'explicit_three_way'
    CONTINUOUS_RISK = 'continuous_risk'


_HANDLING = {Region.POS: CognitiveHandling.CONSIDER_HIT,
             Region.BND: CognitiveHandling.KEEP_BOTH,
             Region.NEG: CognitiveHandling.REJECT_HIT_PREMISE}


@dataclass(frozen=True)
class ObserverCognition:
    context: ObserverContext
    features: ObserverFeatures
    model_digest: str
    estimator_level: str
    loss: LossSpec
    decision: Decision

    def __post_init__(self):
        if (type(self.context) is not ObserverContext or type(self.features) is not ObserverFeatures
                or type(self.loss) is not LossSpec or type(self.decision) is not Decision):
            raise TypeError('typed legal context, features and original D3 risk decision required')
        if condition(self.features) != (self.context.phase, self.context.wolf_count):
            raise ValueError('features/context phase or living-wolf count mismatch')
        if (type(self.model_digest) is not str or re.fullmatch('[0-9a-f]{64}', self.model_digest) is None
                or type(self.estimator_level) is not str or not self.estimator_level):
            raise ValueError('explicit estimator identity required')
        if self.decision != decide(self.decision.probability, self.loss):
            raise ValueError('D3 probability/loss/risks/region binding mismatch')

    def to_record(self):
        return {'context': self.context.to_record(), 'features': self.features.to_record(),
                'model_digest': self.model_digest, 'estimator_level': self.estimator_level,
                'loss': self.loss.to_record(), 'decision': self.decision.to_record()}


@dataclass(frozen=True)
class ProcessedObserver:
    cognition: ObserverCognition
    handling: CognitiveHandling

    def __post_init__(self):
        if (type(self.cognition) is not ObserverCognition
                or self.handling is not _HANDLING[self.cognition.decision.region]):
            raise ValueError('cognitive handling must preserve the original D3 decision')

    def to_record(self):
        return {**self.cognition.to_record(), 'handling': self.handling.value}


@dataclass(frozen=True)
class CognitiveEvidence:
    rows: tuple[ProcessedObserver, ...]
    selected: ObserverSelection
    eligible_observers: tuple[str, ...]

    def to_record(self):
        processed = tuple(r.cognition.context.observer_id for r in self.rows)
        return {'schema_version': VERSION, 'scope': 'PRIVATE_CONTROLLED_RESEARCH_ONLY',
                'panel_coverage': 'ALL_ELIGIBLE' if processed == self.eligible_observers else 'EXPLICIT_SUBSET',
                'eligible_observers': list(self.eligible_observers), 'processed_observers': list(processed),
                'selection': asdict(self.selected), 'rows': [r.to_record() for r in self.rows],
                'meaning': 'effective-support hit, not observer certainty or speech action success'}

    @property
    def digest(self):
        return sha256_bytes(canonical_json_bytes(self.to_record()))


def process_cognition(pre, *, panel):
    """G: bind supplied observer rows and public V1 evidence; choose no speech.

    V2 supplies the publicly selected observer only, preserving one p estimate.
    A complete panel is also accepted; no ranking/averaging or extra inference is
    performed. A partial panel must contain the same public-only selected seat.
    Model identity here is an audit reference, not a formal deployment admission.
    """
    if type(panel) is not tuple or not panel or any(type(r) is not ObserverCognition for r in panel):
        raise TypeError('explicit nonempty tuple of lawful observer cognition required')
    expected = contexts_for_pre(pre, panel[0].context.known_wolf_ids)
    by_id = {c.observer_id: c for c in expected}
    ids = tuple(r.context.observer_id for r in panel)
    if (ids != tuple(p for p in PLAYER_IDS if p in ids)
            or any(by_id.get(r.context.observer_id) != r.context for r in panel)
            or len({r.model_digest for r in panel}) != 1 or len({r.loss.digest for r in panel}) != 1):
        raise ValueError('panel duplicates, PRE/context, estimator or cost mismatch')
    selected = select_observer(pre, panel[0].context.known_wolf_ids)
    if selected.observer_id not in ids:
        raise ValueError('panel omits the V2 public-evidence selected observer')
    return CognitiveEvidence(tuple(ProcessedObserver(r, _HANDLING[r.decision.region]) for r in panel),
                             selected, tuple(c.observer_id for c in expected))


def _configured_plan(selection, branch, strategy, candidates):
    if not isinstance(strategy, SpeechStrategy):
        raise TypeError('explicit SpeechStrategy required')
    if selection.relation == 'NO_EXPLICIT_ACCUSATION':
        return SpeechPlan(A.NO_COMMITMENT), 'NO_DIRECTIONAL_PUBLIC_EVIDENCE'
    if branch is Region.POS:
        action = strategy.positive_self if selection.relation == 'SELF_ACCUSED' else strategy.positive_team
    else:
        action = strategy.boundary if branch is Region.BND else strategy.negative
    plan = SpeechPlan(action, selection.observer_id if action is A.OPPOSE else None)
    if plan not in candidates:
        raise ValueError('strategy intent is not eligible at this PRE')
    return plan, 'CONFIGURED_' + branch.value + '_' + selection.relation


def strategy_plan(pre, selection, decision, strategy):
    """Compatibility adapter for the existing V2 tuple-returning interface."""
    return _configured_plan(selection, decision.region, strategy, generate_candidates(pre))


@dataclass(frozen=True)
class SpeechSelection:
    plan: SpeechPlan
    reason: str
    mode: SelectionMode
    cognitive_branch: Region
    evidence_digest: str
    strategy: SpeechStrategy
    candidates: tuple[SpeechPlan, ...]

    def to_record(self):
        plans = [p.public_payload() for p in self.candidates]
        return {'schema_version': VERSION, 'mode': self.mode.value,
                'cognitive_branch': self.cognitive_branch.value, 'evidence_digest': self.evidence_digest,
                'strategy': self.strategy.to_record(), 'candidates': plans,
                'candidate_order_version': CANDIDATE_ORDER_VERSION,
                'candidates_digest': sha256_bytes(canonical_json_bytes(plans)),
                'selected_plan': self.plan.public_payload(), 'reason': self.reason,
                'selection_basis': 'D3_REGION' if self.mode is SelectionMode.EXPLICIT_THREE_WAY else 'RAW_COGNITIVE_RISK_ARGMIN',
                'speech_utility_model': 'NOT_DEFINED_COGNITIVE_RISKS_ARE_NOT_SPEECH_PAYOFFS'}


def select_speech(pre, evidence, *, candidates, strategy, mode):
    """pi: existing eligible SpeechPlan + explicit mapping; never env/Agent I/O.

    The continuous control minimizes the same already computed cognitive risks
    with the same tie order; it does not consult the region/handling to choose.
    Equality with explicit three-way selection is expected, not evidence of gain.
    """
    if type(evidence) is not CognitiveEvidence or type(mode) is not SelectionMode:
        raise TypeError('processed evidence and an explicit selection mode required')
    if type(candidates) is not tuple or candidates != generate_candidates(pre):
        raise ValueError('complete unchanged canonical candidate tuple required')
    if evidence != process_cognition(pre, panel=tuple(r.cognition for r in evidence.rows)):
        raise ValueError('processed public evidence changed or belongs to another PRE')
    selected = next(r.cognition for r in evidence.rows
                    if r.cognition.context.observer_id == evidence.selected.observer_id)
    if mode is SelectionMode.EXPLICIT_THREE_WAY:
        branch = selected.decision.region
    else:
        risks = selected.decision.risks
        minimum = min(value for _, value in risks.items())
        branch = next(a for a in selected.loss.tie_policy.order if risks[a] == minimum)
    plan, reason = _configured_plan(evidence.selected, branch, strategy, candidates)
    return SpeechSelection(plan, reason, mode, branch, evidence.digest, strategy, candidates)
