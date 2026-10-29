"""Phase-1 read-only cognition and complete candidate inspection, never a policy.

Current Q/D2 and the legacy B2 proxy have separate provenance. Only already
constructed inputs are consumed; this module loads no ToM, fits no model,
selects no plan, calls no Actor/Parser and publishes no artifact.
"""
from dataclasses import asdict, dataclass
import json
from pathlib import Path

from scripts import counterfactual_tom as cf
from scripts.speech_planning import evaluate_candidates, generate_candidates
from scripts.speech_versions import CANDIDATE_ORDER_VERSION
from werewolf.artifact_io import canonical_json_bytes, sha256_bytes
from werewolf.observer_dtrs.d3_protocol import load_protocol
from werewolf.observer_dtrs.decision import TiePolicy, decide
from werewolf.observer_dtrs.features import observer_features
from werewolf.observer_dtrs.online_context import ShadowRequest, digest
from werewolf.observer_dtrs.online_model import FrozenEventEstimator
from werewolf.observer_dtrs.probabilities import ProbabilityMatrix
from werewolf.observer_dtrs.shadow import QPrediction
from werewolf.structured_history import plan_structured_history
from werewolf.tom.dataset import ExperimentCapacity


VERSION = 'observer_dtrs_planning_research_phase1_v1'
D3_PROTOCOL_DIGEST = '0ebd6b97669c35e1d9201f88152da8919d87586864a751f3fd688edd3a11c4ab'


@dataclass(frozen=True)
class ControlledB2:
    """Explicit artificial consumer admission; provenance strings confer no seal.

    The injected consumer implements the *existing* B2 probabilities/provenance
    interface. Even seal-shaped strings remain CONTROLLED_TEST_ONLY.
    """
    consumer: object
    capacity: ExperimentCapacity
    fixture_id: str

    def __post_init__(self):
        if (not isinstance(self.capacity, ExperimentCapacity)
                or type(self.fixture_id) is not str or not self.fixture_id.strip()
                or type(self.consumer) is cf.CounterfactualToMConsumer
                or not callable(getattr(self.consumer, 'probabilities', None))
                or type(getattr(self.consumer, 'provenance', None)) is not cf.ConsumerProvenance):
            raise ValueError('explicit artificial B2 fixture required')


@dataclass(frozen=True)
class ResearchSnapshot:
    payload: bytes

    @property
    def record(self):
        return json.loads(self.payload)

    @property
    def digest(self):
        return sha256_bytes(self.payload)


def _proxy_binding(proxy, expected_provenance):
    if proxy is None:
        if expected_provenance is not None:
            raise ValueError('proxy anchor supplied without a consumer')
        return None, None, {'scope': 'UNAVAILABLE', 'provenance': None}
    if type(proxy) is ControlledB2:
        if expected_provenance is not None:
            raise ValueError('controlled fixtures cannot claim a sealed B2 anchor')
        return proxy.consumer, proxy.capacity, {
            'scope': 'CONTROLLED_TEST_ONLY_NOT_SEALED_B2',
            'fixture_id': proxy.fixture_id,
            'provenance': proxy.consumer.provenance.to_record()}
    if type(proxy) is not cf.CounterfactualToMConsumer:
        raise TypeError('B2 requires ControlledB2 or the original sealed consumer')
    p = proxy.provenance
    if (type(expected_provenance) is not cf.ConsumerProvenance
            or p != expected_provenance
            or p.consumer_schema_version != cf.CONSUMER_SCHEMA_VERSION
            or p.consumer_implementation_version != cf.CONSUMER_IMPLEMENTATION_VERSION
            or p.counterfactual_builder_version != cf.BUILDER_VERSION
            or p.consumer_source_digest != sha256_bytes(Path(cf.__file__).read_bytes())):
        raise ValueError('sealed B2 consumer/source anchor mismatch')
    # Read only the original consumer's already-loaded capacity and provenance.
    # Its unchanged constructor owns seal/runtime/checkpoint admission.
    return proxy, proxy._capacity, {
        'scope': 'ORIGINAL_SEALED_B2_INTERFACE_NOT_DEPLOYMENT_QUAL',
        'provenance': p.to_record()}


def _proxy_batch(pre, candidates, wolves, proxy, expected_provenance):
    consumer, capacity, source = _proxy_binding(proxy, expected_provenance)

    def unavailable(status, *, error_type=None):
        return {'status': status, 'source': source, 'error_type': error_type,
                'rows': [{'plan': p.public_payload(), 'status': status,
                          'evaluation': None} for p in candidates]}

    if consumer is None:
        return unavailable('PROXY_SOURCE_UNAVAILABLE')
    order = cf.derive_public_phase_speaker_order(pre)
    if pre.current_speaker == order[-1]:
        return unavailable('UNSUPPORTED_LAST_SAME_PHASE_SPEAKER')
    # Official parent/capacity checks and continuation semantics are reused.
    # Preflight the complete tuple before spending any proxy inference calls.
    if plan_structured_history(pre).token_count + 3 > capacity.max_seq_len:
        return unavailable('UNSUPPORTED_COMPLETE_SEQUENCE_CAPACITY')
    continuations = tuple(cf.build_continuation(
        pre, p.for_speaker(pre.current_speaker), capacity=capacity) for p in candidates)
    try:
        evaluations = evaluate_candidates(pre, candidates, consumer, alive_wolves=wolves)
        if (consumer.provenance.to_record() != source['provenance']
                or tuple(e.plan for e in evaluations) != candidates
                or any(e.parent_prefix_digest != pre.prefix_digest
                       or e.consumer_provenance.to_record() != source['provenance']
                       or e.delta is not None or e.current_suspicion is not None
                       for e in evaluations)):
            raise ValueError('proxy batch provenance/coverage changed')
    except Exception as error:
        # The legacy evaluator returns no partial results. Do not rerun it,
        # invent a score, or report unreturned rows as individual successes.
        return unavailable('BATCH_FAILED_NO_RESULTS', error_type=type(error).__name__)
    return {'status': 'EVALUATED', 'source': source, 'error_type': None,
            'rows': [{'plan': p.public_payload(), 'status': 'EVALUATED',
                      'continuation_digest': c.continuation_digest,
                      'next_speaker': c.next_speaker,
                      'evaluation': {'suspicion': asdict(e.suspicion),
                                     'candidate_order_version': e.candidate_order_version,
                                     'current_suspicion': None, 'delta': None}}
                     for p, c, e in zip(candidates, continuations, evaluations, strict=True)]}


def analyze_pre(request, prediction, estimator, *, expected_deployment_digest,
                tie_policy, proxy=None, expected_proxy_provenance=None):
    """Return all legal observer states and all single-plan proxy rows.

    No arbitrary cost_source or caller-supplied LossSpec is accepted: all 45
    costs are reconstructed from the pinned frozen D3 registry. Current Q is
    never used as a B2 baseline, and hypothetical Q is never passed into D2.
    Provider failures are distinct from cognitive BND and candidate legality.
    """
    if (type(request) is not ShadowRequest or type(prediction) is not QPrediction
            or type(prediction.matrix) is not ProbabilityMatrix
            or type(estimator) is not FrozenEventEstimator):
        raise TypeError('legal request, bound QPrediction and frozen estimator required')
    protocol = load_protocol()
    protocol.require_frozen(D3_PROTOCOL_DIGEST)
    if type(tie_policy) is not TiePolicy or tie_policy not in protocol.policies:
        raise ValueError('explicit registered D3 tie policy required')
    pre = request.prefix
    plan = plan_structured_history(pre)
    q = prediction.matrix
    if ((q.game_id, q.boundary_id, q.prefix_digest)
            != (pre.game_id, pre.boundary_id, pre.prefix_digest)
            or prediction.plan_digest != plan.plan_digest
            or prediction.tom != estimator.tom
            or expected_deployment_digest != estimator.manifest_digest):
        raise ValueError('current Q/PRE/plan/ToM/estimator source mismatch')
    if plan.token_count > prediction.tom.max_seq_len:
        raise ValueError('current complete PRE exceeds ToM capacity; no truncation')
    contexts = request.contexts
    wolves = contexts[0].alive_wolf_ids
    if len(wolves) >= len(pre.alive_observer_ids) - len(wolves):
        raise ValueError('no legal speech after Classic7 wolf victory')
    candidates = generate_candidates(pre)
    cognition = []
    for context in contexts:
        features = observer_features(context, q)
        detail = estimator.predict(features)
        decisions = []
        for scenario in range(45):
            loss = protocol.loss(scenario, tie_policy)
            decisions.append({'scenario_id': f'C{scenario:02}',
                              'loss': loss.to_record(),
                              'decision': decide(detail['probability'], loss).to_record()})
        cognition.append({'context': context.to_record(), 'features': features.to_record(),
                          'estimate': detail, 'decisions': decisions})
    batch = _proxy_batch(pre, candidates, wolves, proxy, expected_proxy_provenance)
    record = {'schema_version': VERSION, 'scope': 'READ_ONLY_RESEARCH_NOT_A_SPEECH_POLICY',
              'request': request.identity, 'current_Q': q.values,
              'current_Q_digest': digest(q.values), 'plan_digest': plan.plan_digest,
              'current_source': {'tom': prediction.tom.to_record(),
                  'deployment_scope': estimator.record['scope'],
                  'model_id': estimator.record['model_id'],
                  'weighting': estimator.record['weighting'],
                  'deployment_manifest_digest': estimator.manifest_digest,
                  'estimator_digest': estimator.digest,
                  'source_model_digest': estimator.record['source_model_digest'],
                  'source_evidence_digest': estimator.record['source_evidence_digest'],
                  'selection_rule_digest': estimator.record['selection_rule_digest']},
              'cost_source': {'kind': 'FROZEN_D3_REGISTERED_45',
                  'protocol_digest': protocol.digest, 'tie_policy': tie_policy.value},
              'observer_states': cognition, 'candidate_order_version': CANDIDATE_ORDER_VERSION,
              'candidates': [p.public_payload() for p in candidates],
              'proxy_batch': batch,
              'interpretation': 'B2_SUSPICION_PROXY_NOT_EVENT_PROBABILITY_OR_SPEECH_UTILITY'}
    return ResearchSnapshot(canonical_json_bytes(record))
