"""Standalone controlled Shadow composition. No game loop, agents or target reader."""
from collections import deque
from dataclasses import dataclass
import json
import math
from threading import Lock
import time

from werewolf.artifact_io import canonical_json_bytes
from werewolf.structured_history import plan_structured_history
from werewolf.observer_dtrs.d3_protocol import load_protocol
from werewolf.observer_dtrs.dataset import sample_id
from werewolf.observer_dtrs.decision import decide
from werewolf.observer_dtrs.features import observer_features
from werewolf.observer_dtrs.online_context import ShadowRequest, digest
from werewolf.observer_dtrs.online_model import FrozenEventEstimator, TomIdentity, sha
from werewolf.observer_dtrs.probabilities import ProbabilityMatrix
from werewolf.observer_dtrs.shadow_artifacts import ShadowJournal

VERSION = 'observer_dtrs_controlled_shadow_v1'
PROTOCOL_DIGEST = '0ebd6b97669c35e1d9201f88152da8919d87586864a751f3fd688edd3a11c4ab'


@dataclass(frozen=True)
class QPrediction:
    matrix: ProbabilityMatrix
    tom: TomIdentity
    plan_digest: str


@dataclass(frozen=True)
class ShadowResult:
    payload: bytes

    @property
    def record(self):
        return json.loads(self.payload)

    @property
    def digest(self):
        return digest(self.record)


class ShadowEngine:
    """Inject a trusted cooperative-deadline ToM provider and sealed fixture model.

    predict(prefix, deadline_monotonic=...) runs ONLY in the caller's worker.
    A blocking foreign call cannot be preempted by this synchronous core: D4c
    must provide process isolation/hard deadlines before real service integration.
    """

    def __init__(self, *, tom_provider, probability_provider, expected_deployment_digest,
                 journal_root, timeout_seconds=10., clock=time.monotonic):
        if (type(probability_provider) is not FrozenEventEstimator
                or probability_provider.manifest_digest != expected_deployment_digest
                or not isinstance(getattr(tom_provider, 'identity', None), TomIdentity)
                or tom_provider.identity != probability_provider.tom
                or not callable(getattr(tom_provider, 'predict', None))):
            raise ValueError('explicit matching ToM/probability trust anchors required')
        if (type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds)
                or timeout_seconds <= 0 or not callable(clock)):
            raise ValueError('positive finite worker deadline required')
        self.tom_provider = tom_provider
        self.probability_provider = probability_provider
        self._clock = clock
        self.protocol = load_protocol()
        self.protocol.require_frozen(PROTOCOL_DIGEST)
        self._identity = canonical_json_bytes({'schema_version': VERSION, 'scope': 'controlled_shadow_not_formal_deployment',
            'tom': probability_provider.tom.to_record(), 'deployment_manifest_digest': expected_deployment_digest,
            'inference_digest': probability_provider.digest,
            'probability_source': {k: probability_provider.record[k] for k in
                ('scope', 'model_id', 'weighting', 'source_model_digest', 'source_evidence_digest', 'selection_rule_digest')},
            'loss_protocol_digest': self.protocol.digest, 'policies': [p.value for p in self.protocol.policies],
            'cost_scenarios': 45, 'timeout_seconds': timeout_seconds})
        self.journal = ShadowJournal(journal_root, engine_identity=self.identity)

    @property
    def identity(self):
        return json.loads(self._identity)

    def _binding(self, request):
        if type(request) is not ShadowRequest:
            raise TypeError('only an explicit legal ShadowRequest is accepted')
        if (self.tom_provider.identity != TomIdentity(**self.identity['tom'])
                or self.probability_provider.digest != self.identity['inference_digest']
                or self.probability_provider.manifest_digest != self.identity['deployment_manifest_digest']
                or self.protocol.digest != self.identity['loss_protocol_digest']):
            raise ValueError('Shadow provider/model identity changed')
        plan = plan_structured_history(request.prefix)
        if plan.token_count > self.probability_provider.tom.max_seq_len:
            raise ValueError('full PRE exceeds frozen ToM capacity; no truncation')
        return plan

    def _rows(self, request, matrix):
        rows = []
        for context in request.contexts:
            features = observer_features(context, matrix)
            detail = self.probability_provider.predict(features)
            decisions = []
            for policy in self.protocol.policies:
                for scenario in range(45):
                    d = decide(detail['probability'], self.protocol.loss(scenario, policy))
                    decisions.append({'scenario_id': f'C{scenario:02}', **d.to_record()})
            rows.append({'sample_id': sample_id(context), 'observer_id': context.observer_id,
                'features': features.to_record(), **detail, 'decisions': decisions})
        return rows

    def _validate_record(self, request, plan, record):
        if (set(record) != {'schema_version', 'status', 'scope', 'request', 'engine_digest',
                           'tom', 'plan_digest', 'Q', 'Q_digest', 'rows', 'timing'}
                or record['schema_version'] != VERSION or record['status'] != 'SUCCEEDED'
                or record['scope'] != self.identity['scope']
                or canonical_json_bytes(record['request']) != canonical_json_bytes(request.identity)
                or record['engine_digest'] != digest(self.identity)
                or record['tom'] != self.identity['tom'] or record['plan_digest'] != plan.plan_digest
                or record['Q_digest'] != digest(record['Q'])):
            raise ValueError('Shadow record identity/coverage mismatch')
        timing = record['timing']
        if (set(timing) != {'captured_at_ns', 'completed_at_ns', 'inference_elapsed_seconds'}
                or any(type(timing[k]) is not int or timing[k] < 0 for k in ('captured_at_ns', 'completed_at_ns'))
                or type(timing['inference_elapsed_seconds']) not in (int, float)
                or not math.isfinite(timing['inference_elapsed_seconds']) or timing['inference_elapsed_seconds'] < 0):
            raise ValueError('invalid Shadow timing metadata')
        q = ProbabilityMatrix(request.prefix.game_id, request.prefix.boundary_id,
                              request.prefix.prefix_digest, record['Q'])
        if canonical_json_bytes(record['rows']) != canonical_json_bytes(self._rows(request, q)):
            raise ValueError('Shadow record probability/feature/decision mismatch')

    def infer(self, request):
        plan = self._binding(request)
        with self.journal.lock():
            stored = self.journal.begin(request)
            if stored is not None:
                self._validate_record(request, plan, stored)
                return ShadowResult(canonical_json_bytes(stored))
            stage = 'ToM'
            try:
                started = self._clock()
                deadline = started + self.identity['timeout_seconds']
                prediction = self.tom_provider.predict(request.prefix, deadline_monotonic=deadline)
                elapsed = self._clock()-started
                if elapsed > self.identity['timeout_seconds']:
                    raise TimeoutError('Shadow cooperative inference deadline exceeded')
                if (type(prediction) is not QPrediction or prediction.tom != self.probability_provider.tom
                        or prediction.plan_digest != plan.plan_digest or type(prediction.matrix) is not ProbabilityMatrix
                        or (prediction.matrix.game_id, prediction.matrix.boundary_id, prediction.matrix.prefix_digest)
                        != (request.prefix.game_id, request.prefix.boundary_id, request.prefix.prefix_digest)):
                    raise ValueError('ToM response identity mismatch')
                stage = 'composition'
                q = prediction.matrix
                record = {'schema_version': VERSION, 'status': 'SUCCEEDED', 'scope': self.identity['scope'],
                    'request': request.identity, 'engine_digest': digest(self.identity), 'tom': prediction.tom.to_record(),
                    'plan_digest': plan.plan_digest, 'Q': q.values, 'Q_digest': digest(q.values),
                    'rows': self._rows(request, q), 'timing': {'captured_at_ns': request.captured_at_ns,
                    'completed_at_ns': time.time_ns(), 'inference_elapsed_seconds': elapsed}}
                self._validate_record(request, plan, record)
            except Exception as error:
                self.journal.fail(request, 'INFERENCE_TIMEOUT' if isinstance(error, TimeoutError) else 'INFERENCE_FAILED', stage)
                raise
            # An uncertain/disk-failed publication is never overwritten or reclassified.
            self.journal.finish(request, record)
            return ShadowResult(canonical_json_bytes(record))

    def inspect(self, requests, *, expected_result_digests=None):
        """Explicit expected population; verify/recompute sealed results without ToM calls."""
        with self.journal.lock():
            return self._inspect_locked(requests, expected_result_digests)

    def _inspect_locked(self, requests, expected_result_digests):
        requests = tuple(requests)
        if len({r.digest for r in requests}) != len(requests):
            raise ValueError('duplicate expected Shadow requests')
        counts = self.journal._inventory()
        if set(counts['request_digests']) != {r.digest for r in requests}:
            raise ValueError('Shadow request population incomplete')
        results, failures = 0, 0
        result_digests = {}
        for request in requests:
            plan = self._binding(request)
            kind, record = self.journal._read_terminal(request)
            if kind == 'results':
                self._validate_record(request, plan, record)
                result_digests[request.digest] = digest(record)
                results += 1
            elif kind == 'failures':
                if (set(record) != {'status', 'code', 'stage', 'request_digest', 'engine_digest', 'rows'}
                        or record['status'] != 'FAILED' or record['rows'] != []
                        or record['request_digest'] != request.digest
                        or record['engine_digest'] != digest(self.identity)):
                    raise ValueError('invalid technical failure record')
                failures += 1
        if expected_result_digests is not None:
            for h in expected_result_digests.values():
                sha(h)
            if expected_result_digests != result_digests:
                raise ValueError('Shadow external result anchors mismatch')
        return {**counts, 'verified_results': results, 'verified_failures': failures,
                'result_digests': result_digests, 'external_result_anchors_verified': expected_result_digests is not None,
                'scope': self.identity['scope'], 'engine_digest': digest(self.identity)}


class ShadowQueue:
    """Bounded nonblocking submission; manual external-worker pump, no game callbacks.

    Queue membership and submission receipts are transient engineering state,
    not a durable coverage ledger. D4c must retain receipts separately.
    """

    def __init__(self, max_pending):
        if type(max_pending) is not int or max_pending < 1:
            raise ValueError('positive queue bound required')
        self.max_pending = max_pending
        self._pending = deque()
        self._active = set()
        self._lock = Lock()

    def submit(self, request):
        if type(request) is not ShadowRequest:
            raise TypeError('legal ShadowRequest required')
        with self._lock:
            if request.digest in self._active:
                status = 'DUPLICATE'
            elif len(self._active) >= self.max_pending:
                status = 'SKIPPED_BACKPRESSURE'
            else:
                self._active.add(request.digest)
                self._pending.append(request)
                status = 'QUEUED'
        return {'request_digest': request.digest, 'status': status}

    def work_one(self, engine):
        with self._lock:
            if not self._pending:
                return None
            request = self._pending.popleft()
        try:
            result = engine.infer(request)
            return {'request_digest': request.digest, 'status': 'SUCCEEDED', 'result_digest': result.digest}
        except Exception:
            # No exception payload is copied to public/game state or to an untrusted log.
            return {'request_digest': request.digest, 'status': 'FAILED'}
        finally:
            with self._lock:
                self._active.remove(request.digest)
