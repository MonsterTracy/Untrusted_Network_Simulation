"""S2-P0: prebound cognitive bases and independent artificial action values.

Pure research module. No source reader, model fit/call, environment, language,
supervision or commit. A basis gate never changes game-rule legal candidates.
"""
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
import math
from numbers import Real
import re

from scripts.speech_planning import SpeechPlan, generate_candidates
from scripts.speech_versions import CANDIDATE_ORDER_VERSION
from werewolf.artifact_io import canonical_json_bytes, sha256_bytes
from werewolf.canonical_collection.public_history import PLAYER_IDS
from werewolf.observer_dtrs.decision import Region, decide, forced_binary
from werewolf.observer_dtrs.speech_evidence import CognitiveEvidence, process_cognition

VERSION = 'observer_dtrs_candidate_appraisal_s2_p0_v1'


class BasisKind(str, Enum):
    PUBLIC_ONLY = 'PUBLIC_ONLY'
    ASSUME_HIT = 'ASSUME_EFFECTIVE_SUPPORT_HIT'
    ASSUME_MISS = 'ASSUME_EFFECTIVE_SUPPORT_MISS'
    KEEP_BOTH = 'KEEP_BOTH_HYPOTHESES'


class GateMode(str, Enum):
    RECORD_ONLY = 'record_risk_without_gate'
    THREE_WAY = 'explicit_three_way_basis_gate'
    CONTINUOUS_EQUIVALENCE = 'same_risk_argmin_equivalence_only'
    FORCED_BINARY = 'forced_binary_basis_gate'


JUDGMENT = {BasisKind.ASSUME_HIT: Region.POS, BasisKind.ASSUME_MISS: Region.NEG,
            BasisKind.KEEP_BOTH: Region.BND}


def judgment_risk(probability, loss, judgment):
    """Forced cognitive loss excess, never action utility or causal regret."""
    if type(judgment) is not Region:
        raise TypeError('an explicit cognitive Region required')
    decision = decide(probability, loss)  # Original D3 validates p/loss/ties.
    minimum = min(v for _, v in decision.risks.items())
    excess = decision.risks[judgment] - minimum
    return {'decision': decision.to_record(), 'forced_judgment': judgment.value,
            'forced_risk_exact': str(decision.risks[judgment]), 'minimum_risk_exact': str(minimum),
            'excess_exact': str(excess), 'normalized_excess_exact': str(excess/(loss.c_fp+loss.c_fn)),
            'interpretation': 'COGNITIVE_JUDGMENT_EXCESS_NOT_SPEECH_EFFECT'}


@dataclass(frozen=True)
class PublicReference:
    event_id: str
    annotation_digest: str

    def to_record(self):
        return {'event_id': self.event_id, 'annotation_digest': self.annotation_digest}


@dataclass(frozen=True)
class CandidateBasis:
    basis_id: str
    plan: SpeechPlan
    kind: BasisKind
    observer_id: str | None = None
    public_references: tuple[PublicReference, ...] = ()

    def __post_init__(self):
        if (type(self.basis_id) is not str or re.fullmatch('[A-Za-z0-9_.-]+', self.basis_id) is None
                or type(self.plan) is not SpeechPlan or type(self.kind) is not BasisKind
                or type(self.public_references) is not tuple
                or any(type(r) is not PublicReference for r in self.public_references)):
            raise ValueError('typed, explicit candidate basis required')
        if (self.kind is BasisKind.PUBLIC_ONLY and self.observer_id is not None
                or self.kind is not BasisKind.PUBLIC_ONLY and self.observer_id not in PLAYER_IDS):
            raise ValueError('only cognitive bases require an explicit observer')

    def to_record(self):
        return {'basis_id': self.basis_id, 'plan': self.plan.public_payload(), 'kind': self.kind.value,
                'observer_id': self.observer_id,
                'public_references': [r.to_record() for r in self.public_references]}


def _candidate_digest(candidates):
    return sha256_bytes(canonical_json_bytes([p.public_payload() for p in candidates]))


@dataclass(frozen=True)
class BoundBases:
    game_id: str
    boundary_id: str
    prefix_digest: str
    candidates: tuple[SpeechPlan, ...]
    bases: tuple[CandidateBasis, ...]

    def to_record(self):
        return {'version': VERSION, 'game_id': self.game_id, 'boundary_id': self.boundary_id,
                'prefix_digest': self.prefix_digest, 'candidate_order_version': CANDIDATE_ORDER_VERSION,
                'candidates': [p.public_payload() for p in self.candidates],
                'bases': [b.to_record() for b in self.bases],
                'public_only_without_references': 'declared_public_PRE_only_no_factual_claim_added',
                'meaning': 'predeclared_argument_schema_not_verified_natural_language_or_actual_event_truth'}

    @property
    def digest(self):
        return sha256_bytes(canonical_json_bytes(self.to_record()))


def bind_bases(pre, *, candidates, bases):
    """Bind public PRE/candidate/basis inventory before appraisal; no p/value input.

    An empty basis catalog is valid and will produce NO_ADMISSIBLE_BASIS. A
    public-only PRE basis is explicit, never auto-created as a fallback.
    """
    if type(candidates) is not tuple or candidates != generate_candidates(pre):
        raise ValueError('complete canonical legal candidate tuple required')
    if type(bases) is not tuple or any(type(b) is not CandidateBasis for b in bases):
        raise TypeError('explicit immutable candidate basis tuple required')
    if len({b.basis_id for b in bases}) != len(bases):
        raise ValueError('duplicate basis IDs')
    annotations = {a.event_id: a for a in pre.v1_annotations}
    events = {e.event_id: e for e in pre.public_event_history.events}
    for b in bases:
        if b.plan not in candidates or b.observer_id is not None and b.observer_id not in pre.alive_observer_ids:
            raise ValueError('basis plan or observer is not legal at PRE')
        refs = b.public_references
        if len(set(refs)) != len(refs) or refs != tuple(sorted(refs, key=lambda r: r.event_id)):
            raise ValueError('public references must be unique and canonical')
        for ref in refs:
            ann, event = annotations.get(ref.event_id), events.get(ref.event_id)
            if (ann is None or event is None or event.event_type != 'public_speech'
                    or not ann.is_successful or ann.annotation_digest != ref.annotation_digest):
                raise ValueError('public reference does not bind successful actual V1 evidence')
    ordered = tuple(sorted(bases, key=lambda b: (candidates.index(b.plan), b.basis_id)))
    return BoundBases(pre.game_id, pre.boundary_id, pre.prefix_digest, candidates, ordered)


@dataclass(frozen=True)
class ArtificialActionValues:
    """Complete deterministic preference fixture, not a qualified effect source."""
    game_id: str
    boundary_id: str
    prefix_digest: str
    candidate_digest: str
    fixture_id: str
    scores: tuple[Fraction, ...]

    def __post_init__(self):
        if (type(self.fixture_id) is not str or not self.fixture_id.strip()
                or type(self.scores) is not tuple or not self.scores):
            raise ValueError('explicit complete artificial action score table required')
        exact = []
        for s in self.scores:
            if isinstance(s, bool) or not isinstance(s, Real) or not math.isfinite(float(s)):
                raise ValueError('finite artificial score required, not bool/string/NaN')
            exact.append(Fraction(s))
        object.__setattr__(self, 'scores', tuple(exact))

    def to_record(self):
        return {'version': VERSION, 'scope': 'ARTIFICIAL_PREFERENCE_NOT_MEASURED_SPEECH_EFFECT',
                'game_id': self.game_id, 'boundary_id': self.boundary_id,
                'prefix_digest': self.prefix_digest, 'candidate_digest': self.candidate_digest,
                'fixture_id': self.fixture_id, 'scores_exact': [str(s) for s in self.scores],
                'direction': 'higher_preferred_no_cognitive_risk_added_or_subtracted'}

    @property
    def digest(self):
        return sha256_bytes(canonical_json_bytes(self.to_record()))


def artificial_action_values(pre, *, candidates, scores, fixture_id):
    """Independent value seam: accepts public PRE/candidates/scores, never G/p/Y."""
    if type(candidates) is not tuple or candidates != generate_candidates(pre):
        raise ValueError('complete canonical legal candidate tuple required')
    if type(scores) is not tuple or len(scores) != len(candidates):
        raise ValueError('one artificial value per legal plan required')
    return ArtificialActionValues(pre.game_id, pre.boundary_id, pre.prefix_digest,
                                  _candidate_digest(candidates), fixture_id, scores)


@dataclass(frozen=True)
class CandidateAppraisal:
    status: str
    selected_plan: SpeechPlan | None
    selected_basis_id: str | None
    audit: dict  # Fresh private record; never used as a gameplay envelope.


def appraise_candidates(pre, evidence, catalog, *, mode, values=None):
    """Enumerate all legal plans, inspect prebound bases, then rank admitted plans.

    Gate modes are explicit normative hypotheses. At exact ties the existing
    TiePolicy selects ONE judgment: another zero-excess judgment may be rejected
    by that tie convention. No region maps intrinsically to a speech action.
    """
    if type(mode) is not GateMode or type(evidence) is not CognitiveEvidence or type(catalog) is not BoundBases:
        raise TypeError('validated G, prebound catalog and explicit gate mode required')
    expected = bind_bases(pre, candidates=catalog.candidates, bases=catalog.bases)
    if catalog != expected:
        raise ValueError('basis catalog belongs to another PRE or was changed')
    if evidence != process_cognition(pre, panel=tuple(r.cognition for r in evidence.rows)):
        raise ValueError('G evidence does not bind this PRE')
    context = evidence.rows[0].cognition.context
    if context.wolf_count >= len(context.alive_player_ids) - context.wolf_count:
        raise ValueError('cannot plan after the Classic7 wolf victory condition')
    if values is not None:
        if type(values) is not ArtificialActionValues:
            raise TypeError('only an explicit artificial preference fixture is admitted in P0')
        if ((values.game_id, values.boundary_id, values.prefix_digest, values.candidate_digest)
                != (pre.game_id, pre.boundary_id, pre.prefix_digest, _candidate_digest(catalog.candidates))
                or len(values.scores) != len(catalog.candidates)):
            raise ValueError('action value PRE/candidate identity mismatch')
    cognitions = {r.cognition.context.observer_id: r.cognition for r in evidence.rows}
    by_plan = {p: [] for p in catalog.candidates}
    for b in catalog.bases:
        by_plan[b.plan].append(b)
    reviews, admitted = [], []
    for index, plan in enumerate(catalog.candidates):
        basis_reviews = []
        for basis in by_plan[plan]:
            required = JUDGMENT.get(basis.kind)
            row = cognitions.get(basis.observer_id)
            risk, selected = None, None
            if required is None:
                ok, reason = True, 'PUBLIC_ONLY_NO_COGNITIVE_GATE'
            elif basis.observer_id not in evidence.eligible_observers:
                ok, reason = False, 'BASIS_OBSERVER_OUTSIDE_LIVING_NONWOLF_DOMAIN'
            elif row is None:
                ok, reason = False, 'COGNITIVE_OBSERVER_ROW_UNAVAILABLE_NO_FALLBACK'
            else:
                risk = judgment_risk(row.decision.probability, row.loss, required)
                if mode is GateMode.RECORD_ONLY:
                    ok, reason = True, 'RISK_RECORDED_GATE_DISABLED'
                else:
                    if mode is GateMode.THREE_WAY:
                        selected = row.decision.region
                    elif mode is GateMode.FORCED_BINARY:
                        selected = forced_binary(row.decision.probability, row.loss,
                                                 tie_policy=row.loss.tie_policy)
                    else:
                        minimum = min(v for _, v in row.decision.risks.items())
                        selected = next(r for r in row.loss.tie_policy.order if row.decision.risks[r] == minimum)
                    ok = selected is required
                    reason = ('COGNITIVE_BASIS_ADMITTED' if ok else
                              'BINARY_MECHANISM_EXCLUDES_DEFER_BASIS' if mode is GateMode.FORCED_BINARY and required is Region.BND else
                              'ZERO_EXCESS_ALTERNATIVE_REJECTED_BY_EXPLICIT_TIE' if Fraction(risk['excess_exact']) == 0 else
                              'COGNITIVE_BASIS_NOT_SELECTED_BY_MECHANISM')
            basis_reviews.append({'basis': basis.to_record(), 'admitted': ok, 'reason': reason,
                'required_judgment': required.value if required else None,
                'mechanism_judgment': selected.value if selected else None, 'cognitive_risk': risk})
        accepted = [r['basis']['basis_id'] for r in basis_reviews if r['admitted']]
        if accepted:
            admitted.append((index, plan, accepted))
        reviews.append({'plan': plan.public_payload(), 'game_rule_legal': True,
            'basis_reviews': basis_reviews, 'admitted_basis_ids': accepted, 'assessable': bool(accepted),
            'exclusion': None if accepted else 'NO_DECLARED_BASIS' if not basis_reviews else 'ALL_DECLARED_BASES_REJECTED',
            'artificial_action_score_exact': str(values.scores[index]) if values else None})
    if not admitted:
        status, chosen = 'NO_ADMISSIBLE_BASIS', None
    elif values is None:
        status, chosen = 'ACTION_VALUE_UNAVAILABLE', None
    else:
        # Candidate order is the existing canonical order; basis IDs only break
        # ties between admitted reasons for the SAME plan, never change its score.
        chosen = max(admitted, key=lambda item: values.scores[item[0]])
        status = 'SELECTED_ARTIFICIAL_ONLY'
    audit = {'version': VERSION, 'scope': 'PRIVATE_CONTROLLED_RESEARCH_ONLY',
        'status': status, 'mode': mode.value, 'game_id': pre.game_id,
        'boundary_id': pre.boundary_id, 'prefix_digest': pre.prefix_digest,
        'catalog_digest': catalog.digest, 'G_digest': evidence.digest,
        'value_source': values.to_record() if values else None,
        'value_source_digest': values.digest if values else None,
        'legal_candidate_count': len(catalog.candidates), 'assessable_candidate_count': len(admitted),
        'admitted_basis_count': sum(len(a[2]) for a in admitted), 'candidates': reviews,
        'tie_rule': 'canonical_candidate_order_then_lexicographic_admitted_basis_id',
        'risk_policy': 'selected_judgment_basis_gate; no threshold fitted; forced binary is explicit ablation',
        'risk_is_action_value': False,
        'selected_plan': chosen[1].public_payload() if chosen else None,
        'selected_basis_id': chosen[2][0] if chosen else None}
    return CandidateAppraisal(status, chosen[1] if chosen else None, chosen[2][0] if chosen else None, audit)
