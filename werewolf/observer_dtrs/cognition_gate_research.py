"""Historical cognition/basis diagnostics, not a speech policy or value model.

Only legal dataset rows and a label-free sealed prediction batch cross this
interface. Reader provenance remains the responsibility of the offline caller.
The sole new basis rule verifies a narrow public conversational premise, never
the truth/utility of self-defence or a dependency on the hidden D1 event.
"""
from collections import defaultdict
from dataclasses import asdict, dataclass
from functools import lru_cache
import re

from scripts.speech_planning import PlanningAction, SpeechPlan, generate_candidates
from werewolf.artifact_io import canonical_json_bytes, sha256_bytes
from werewolf.observer_dtrs.candidate_appraisal import (
    BasisKind, CandidateBasis, GateMode, PublicReference, appraise_candidates, bind_bases,
)
from werewolf.observer_dtrs.context import contexts_for_pre
from werewolf.observer_dtrs.d3_protocol import load_protocol
from werewolf.observer_dtrs.dataset import LegalDatasetRow, ObserverDatasetView, sample_id
from werewolf.observer_dtrs.decision import TiePolicy, decide, forced_binary
from werewolf.observer_dtrs.decision_offline import ProbabilityInput
from werewolf.observer_dtrs.features import observer_features
from werewolf.observer_dtrs.probabilities import ProbabilityRow
from werewolf.observer_dtrs.sealed_d2_offline import PredictionBatch, SampleOrigin
from werewolf.observer_dtrs.speech_evidence import ObserverCognition, process_cognition, select_observer

VERSION = 'historical_cognition_gate_diagnostic_v1'
D3_DIGEST = '0ebd6b97669c35e1d9201f88152da8919d87586864a751f3fd688edd3a11c4ab'
PUBLIC_RULE = 'latest_same_day_eligible_observer_explicit_self_accusation_reply_cue_v1'
ARMS = tuple(GateMode)


def digest(value):
    return sha256_bytes(canonical_json_bytes(value))


@dataclass(frozen=True)
class HistoricalObserver:
    legal: LegalDatasetRow
    probability: float
    origin: SampleOrigin
    prediction_reference: str  # Outer prediction artifact, not a parameter hash.

    def __post_init__(self):
        if type(self.legal) is not LegalDatasetRow or type(self.origin) is not SampleOrigin:
            raise TypeError('typed legal row and sealed origin required')
        c=self.legal.context
        ProbabilityInput(self.legal.sample_id,c.game_id,self.probability)
        if (self.legal.sample_id!=sample_id(c)
                or observer_features(c,self.legal.probability)!=self.legal.features
                or any(type(x) is not int or x not in range(5)
                       for x in (self.origin.tom_fold,self.origin.outer_fold))
                or type(self.origin.nW) is not int
                or re.fullmatch('[0-9a-f]{64}',self.origin.block_digest or '') is None
                or (self.origin.sample_id,self.origin.game_id,self.origin.phase,self.origin.nW)
                    !=(self.legal.sample_id,c.game_id,c.phase,c.wolf_count)
                or re.fullmatch('[0-9a-f]{64}',self.prediction_reference or '') is None):
            raise ValueError('observer/Q/p/PRE/reference mismatch')


def bind_inputs(dataset, batch):
    """Strict sample-ID join of already verified legal readers; no targets handle.

    Does not authenticate arbitrary manually constructed objects as formal data.
    Formal callers must use open_dataset/read_formal_d2 with external anchors.
    """
    if type(dataset) is not ObserverDatasetView or type(batch) is not PredictionBatch:
        raise TypeError('legal dataset view and label-free PredictionBatch required')
    identity = batch.identity
    run = identity['run']
    if (identity['model_id'] != 'OBS-EQ1' or identity['weighting'] != 'game_equal'
            or run['dataset_digest'] != dataset.manifest_digest
            or identity['run_identity_digest'] != digest(run)):
        raise ValueError('dataset/model/weighting/run source mismatch')
    legal = {r.sample_id: r for r in dataset.rows}
    probabilities = {r.sample_id: r for r in batch.rows}
    origins = {r.sample_id: r for r in batch.origins}
    if (len(legal) != len(dataset.rows) or len(probabilities) != len(batch.rows)
            or len(origins) != len(batch.origins) or set(legal) != set(probabilities)
            or set(legal) != set(origins)):
        raise ValueError('duplicate/missing sample IDs')
    if (identity['predictions_digest'] != digest(sorted(
            [r.sample_id, r.game_id, float(r.probability)] for r in batch.rows))
            or identity['predictions_digest'] != run['prediction_digests']['game_equal/OBS-EQ1']
            or identity['origins_digest'] != digest(sorted(
                (asdict(o) for o in batch.origins), key=lambda o:o['sample_id']))
            or identity['origins_digest'] != run['sample_origins_digest']):
        raise ValueError('probability or origin digest mismatch')
    refs = {k:v['game_equal']['predictions'] for k,v in run['outer_references'].items()}
    if identity['outer_prediction_digest_by_block'] != refs:
        raise ValueError('outer model reference mismatch')
    folds = {g['game_id']:g['fold_index'] for g in dataset.artifact.manifest['games']}
    grouped = defaultdict(list)
    for sid, row in legal.items():
        if type(row) is not LegalDatasetRow or sid != sample_id(row.context):
            raise ValueError('legal row/sample identity mismatch')
        q = row.probability
        validated = ProbabilityRow(q.game_id,q.boundary_id,q.prefix_digest,q.observer_id,q.values,q.seat_ids)
        if observer_features(row.context,validated) != row.features:
            raise ValueError('Q/context/features binding mismatch')
        p, origin, c = probabilities[sid], origins[sid], row.context
        if type(p) is not ProbabilityInput:
            raise TypeError('only scalar ProbabilityInput, never raw supervised rows')
        ProbabilityInput(p.sample_id,p.game_id,p.probability)
        block = f'{origin.tom_fold}-{origin.outer_fold}'
        if (p.game_id != c.game_id or origin.game_id != c.game_id or origin.sample_id != sid
                or (origin.phase,origin.nW) != (c.phase,c.wolf_count)
                or folds[c.game_id] != origin.tom_fold
                or run['block_digests'].get(block) != origin.block_digest):
            raise ValueError('sample/game/PRE/source fold binding mismatch')
        grouped[(c.game_id,c.boundary_id)].append(HistoricalObserver(row,float(p.probability),origin,refs[block]))
    return {k:tuple(sorted(v,key=lambda r:r.legal.context.observer_id)) for k,v in grouped.items()}


def validate_pre_inputs(pre, rows):
    if type(rows) is not tuple or not rows or any(type(r) is not HistoricalObserver for r in rows):
        raise TypeError('nonempty immutable historical observer tuple required')
    expected = contexts_for_pre(pre,rows[0].legal.context.known_wolf_ids)
    if (tuple(r.legal.context for r in rows) != expected
            or len({r.prediction_reference for r in rows}) != 1
            or len({(r.origin.tom_fold,r.origin.outer_fold,r.origin.block_digest) for r in rows}) != 1):
        raise ValueError('PRE/actor/observer population/outer model mismatch')
    return select_observer(pre,rows[0].legal.context.known_wolf_ids)


def public_reply_catalog(pre, rows):
    """Prebind only a public reply cue; all unproved reasons stay ungenerated.

    Uses exactly the latest same-day speech per eligible observer, like S1a's
    evidence selection. A successful V1 must assert self and not also clear self.
    Not an exhaustive catalog of good arguments and not a hidden-belief witness.
    """
    validate_pre_inputs(pre,rows)
    candidates = generate_candidates(pre)
    annotations = {a.event_id:a for a in pre.v1_annotations}
    bases, facts = [], []
    for row in rows:
        observer = row.legal.context.observer_id
        events = [e for e in pre.public_event_history.events if e.event_type=='public_speech'
                  and e.speaker==observer and e.temporal_state.day==pre.public_temporal_state.day]
        last = events[-1] if events else None
        ann = annotations.get(last.event_id) if last else None
        hit = bool(ann and ann.is_successful and any(
            a.action=='point_as_werewolf' and a.object==pre.current_speaker for a in ann.actions))
        clear = bool(ann and ann.is_successful and any(
            a.action=='point_as_non_werewolf' and a.object==pre.current_speaker for a in ann.actions))
        status = ('NO_SAME_DAY_SPEECH' if last is None else 'V1_UNAVAILABLE_OR_FAILED' if not ann or not ann.is_successful
                  else 'CONFLICTING_V1_SELF_CLAIMS' if hit and clear else
                  'VERIFIED_PUBLIC_REPLY_CUE' if hit else 'NO_EXPLICIT_SELF_ACCUSATION')
        fact = {'observer_id':observer,'event_id':last.event_id if last else None,
                'annotation_digest':ann.annotation_digest if ann else None,'status':status}
        facts.append(fact)
        if status=='VERIFIED_PUBLIC_REPLY_CUE':
            bases.append(CandidateBasis('public-reply.'+observer,SpeechPlan(PlanningAction.SELF_DEFEND),
                BasisKind.PUBLIC_ONLY,public_references=(PublicReference(last.event_id,ann.annotation_digest),)))
    return bind_bases(pre,candidates=candidates,bases=tuple(bases)), tuple(facts)


@lru_cache(maxsize=90, typed=True)
def registered_loss(scenario, policy):
    p=load_protocol();p.require_frozen(D3_DIGEST)
    return p.loss(scenario,policy)


@lru_cache(maxsize=100000, typed=True)
def profile(probability, scenario, policy):
    """Reuse original exact D3. Cache identical sealed p/cost inputs only."""
    loss=registered_loss(scenario,policy)
    d=decide(probability,loss)
    smallest=min(v for _,v in d.risks.items())
    continuous=next(a for a in loss.tie_policy.order if d.risks[a]==smallest)
    if continuous is not d.region:
        raise AssertionError('explicit/continuous D3 mismatch')
    binary=forced_binary(probability,loss,tie_policy=policy)
    return d,binary


def audit_pre(pre, rows, *, scenario, policy, catalog=None):
    """Original G/P0 four-arm execution without action values; no selected plan.

    This is the exact callable used by bounded real-PRE differential checks.
    Full-cohort PUBLIC_ONLY cost-invariance can be expanded algebraically, with
    that expansion explicitly distinguished from executing this callable.
    """
    validate_pre_inputs(pre,rows)
    expected,_=public_reply_catalog(pre,rows)
    if catalog is not None and catalog!=expected:
        raise ValueError('catalog/source/semantics mismatch; no invented cognitive bases')
    loss=registered_loss(scenario,policy)
    panel=tuple(ObserverCognition(r.legal.context,r.legal.features,r.prediction_reference,
        'SEALED_OUTER_PREDICTION_REFERENCE_NOT_TRAINING_SUPPORT',loss,profile(r.probability,scenario,policy)[0]) for r in rows)
    full=process_cognition(pre,panel=panel)
    selected=next(r for r in panel if r.context.observer_id==full.selected.observer_id)
    single=process_cognition(pre,panel=(selected,))
    result={}
    for scope,g in [('full',full),('selected',single)]:
        result[scope]={}
        for mode in ARMS:
            appraisal=appraise_candidates(pre,g,expected,mode=mode,values=None)
            if appraisal.selected_plan is not None:
                raise AssertionError('action value unavailable: selected plan forbidden')
            result[scope][mode.value]=appraisal.audit
    return result
