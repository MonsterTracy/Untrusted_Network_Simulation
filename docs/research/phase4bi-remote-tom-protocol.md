# Phase 4B-I: frozen-ToM JSONL client protocol

Gameplay baseline: c9f92b9a05c91fed330bada994e817ee42e2ef55.
Worker source baseline (future work only):
8886af9505036b55662d54ae9a108b09121e8c71.
NOT RUN: tests have not been executed. No worker/model/subprocess transport was
started or implemented; no frozen branch/worktree was created or modified.

## Transport and versions

REMOTE_TOM_PROTOCOL_VERSION = remote_tom_plan_v1.
JsonLineTransport.request(line: str) -> str exchanges exactly one newline-terminated
JSON object per call. The client uses raw JSON lines so duplicate keys remain
observable and are rejected at every nesting level. Multiple lines, non-objects,
malformed JSON, NaN/Infinity and unknown response fields are rejected. Worker
stdout must contain only protocol JSONL; stderr must be separate.

Only the client abstraction is implemented. A future persistent subprocess
adapter must use explicit argv, shell=False, bounded read/write waits, separate
stderr draining and explicit exit/EOF errors. No retries, automatic reconnect,
shared memory, pickle, sys.path/PYTHONPATH changes or cross-worktree imports.
An arbitrary injected transport must uphold the bounded-wait contract; this
synchronous provider does not itself interrupt a transport that blocks forever.

CANDIDATE_ORDER_VERSION and QUEUE_RULE_VERSION now live in a lightweight
scripts/speech_versions.py. Phase 1/2 re-export the same exact values through
imports. No ontology, ordering or eligibility change. The remote client imports
neither speech_planning/evaluator nor counterfactual_tom/consumer/model modules.

## Required request fields

protocol_version, request_id, request_digest, game_id, boundary_id,
parent_digest, parent_pre, speaker, day, phase, candidates,
candidate_order_version, public_queue_rule_version, alive_wolves, condition,
expected_worker_identity.

Candidates are the complete Phase 4A tuple serialized in its supplied canonical
order as action/target dictionaries. The client does not reimplement generation
or eligibility; the trusted input is an already gated PublicOpportunity. It
checks nonempty tuple, duplicate candidates, valid parent and matching speaker.
The future worker must regenerate and compare the full canonical tuple, not
trust a caller-declared ordering or subset.

alive_wolves contains only objective-required living wolf seats, sorted by
canonical seat order. The client requires a nonempty, unique living subset,
including the acting speaker, and at least one living non-wolf observer. The
researcher/runner is responsible for supplying lawful membership; it cannot be
inferred from public PRE. No complete private role table is transmitted.

## PRE representation

parent_pre is the unchanged AuthoritativePREPrefix.to_record() after validation
by validate_authoritative_pre_prefix. It includes schema/game/boundary/step/
trigger/speaker/day/phase, alive_observer_ids, public_events and their schema/
digest, V1 annotations with actual attempts and their schema/digest, maximum
public day, belief observation links, speaker observation ID and prefix_digest.
No boundary/provenance fields are removed. Private belief contents or role truth
are not part of this representation.

This is the existing canonical record, not a second PRE semantics. In 4B-II the
worker must reconstruct existing public-history, annotation and PRE value types
from that full record, call their official binding/validation functions and
require exact canonical round-trip equality, including all supplied digests and
observation links. Prefix validity is not proof that an untrusted sender really
ran the game: authentic opportunity provenance remains the trusted runner's
responsibility, as in existing Phase 1.

## Digest and deterministic request identity

1. Build all request fields except request_id and request_digest.
2. request_id = "rtom-" + SHA256(canonical_json_bytes(those fields)).
3. Add request_id; request_digest = SHA256(canonical_json_bytes(the resulting object)).
4. Add request_digest and serialize the JSONL request.

Thus all decision-relevant fields, including full PRE, its digest, ordered
candidates, version strings, private alive_wolves, condition and expected worker
identity are bound. Identical semantic inputs produce identical IDs/digests;
repeated identical requests are not assigned fresh random IDs. Python hash()
is never used. The worker must recompute both bindings before inference.

Both request_id and request_digest are planner-private metadata: the small
wolf-seat domain permits enumeration even when only a digest is exposed. They
must not enter SpeechPlan, Actor payloads, canonical public events, game logs
or agent-visible/public SpeechTrace. Allowed locations are RPC request/response,
worker-private logs and planner-private research sidecars. Existing public
game_id/boundary_id provide correlation without these private bindings; no new
public correlation field is needed. Equal alive_wolves sets, regardless of
supplier order, produce identical canonical requests and bindings. Duplicate
or invalid IDs are rejected rather than silently deduplicated.

## Worker identity

WorkerIdentity requires planner_baseline_commit and worker_commit (each 40
lowercase hex), runtime_digest,
source_digest, experiment_digest, seal_digest, checkpoint_digest (each 64
lowercase hex), and condition (implicit or explicit_day_phase). All fields are
required configuration; no unknown/latest/None placeholders are supported.
Tests use explicitly artificial fixture digests, not claimed runtime values.

planner_baseline_commit identifies the source lineage baseline:
8886af9505036b55662d54ae9a108b09121e8c71. worker_commit must identify the actual
Phase 4B-II worker worktree HEAD after the worker adapter has been committed.
Do not claim the baseline hash is that later HEAD. Both fields are required,
covered by request_id/request_digest, and compared in the complete response
worker_identity. Configuration validates hash syntax; the future worker must
verify actual Git HEAD and baseline lineage. Fixture worker_commit is synthetic
and deliberately different from the baseline.

runtime_digest identifies the canonical frozen runtime inputs; source_digest
binds the actual worker source/adapter release to be agreed in 4B-II. Model
experiment/seal/checkpoint digests must come from verified artifacts. The actual
worker must independently measure/verify these identities, never merely echo
client expectations. This local IPC protocol is an integrity/binding contract,
not cryptographic authentication against a malicious executable.

## Response

Success has exactly: protocol_version, request_id, request_digest, status="ok",
selected_index, selected_plan, worker_identity. Index must be a JSON integer
(not bool), in range; the returned plan must exactly match that indexed request
candidate. The provider returns the corresponding ORIGINAL candidate object.

Error has exactly: protocol_version, request_id, request_digest, status="error",
error_code, error_message, worker_identity. Strings must be nonempty and all
bindings/identity must match. Missing identity is rejected as malformed rather
than accepted as a successful handshake. Unavailable/failed startup may instead
raise a transport error. Any worker error is terminal. Worker messages are not
copied into gameplay exceptions/public traces.

Unknown fields, including scores/matrices/diagnostics, are rejected. No candidate
substitution, local selector, fallback, retry or silent repair exists.

## Provider and information boundaries

RemoteToMPlanProvider holds transport, complete WorkerIdentity and an explicit
alive_wolves_for(public_opportunity) private objective-state supplier. It has no
env/Actor/perceiver/recorder argument or field. The supplier is an integration
seam for lawful current membership, not an implemented game-state collector.
Phase 4A already treats provider errors as eligible execution failure and does
not call Original Agent afterward.

The outbound wire and transport are planner-private and must not be copied into
public logs. Provider returns only a SpeechPlan; Phase 4A trace/Actor/public
canonical evidence are unchanged. No execution-metadata sidecar is added in this
phase. Future worker diagnostics stay on a worker-private sidecar.

## Validation

Fake transport tests cover normal/PK requests, deterministic digests, candidate
order/full PRE/private state binding, valid canonical-object return, all response
bindings/worker identity fields, malformed/duplicate JSON, missing/unknown fields,
worker errors, bad indices/plans, timeout/unavailability/exit exceptions, no retry,
Phase 4A failure propagation, original/nonassigned/last-speaker zero transport
calls, and fresh-process absence of model/evaluator imports.

From the server repository root:

```sh
python -m pytest -q tests/tom/test_remote_tom_provider.py
python -m pytest -q tests/tom/test_constrained_speech.py
python -m pytest -q tests/tom/test_verified_speech_commit.py tests/tom/test_speech_realization.py tests/tom/test_speech_planning.py tests/tom/test_counterfactual_consumer.py tests/tom/test_suspicion_objectives.py
python -m pytest -q
```

## Phase 4B-II exact plan (not performed)

1. Create an independent worker branch/worktree FROM
   8886af9505036b55662d54ae9a108b09121e8c71, never from current gameplay HEAD.
2. Add only worker entrypoint, protocol adapter, worker tests/docs. Do not modify
   werewolf/**, run_random.py or Phase 1/2 model semantics; do not cherry-pick
   gameplay integration commits. Check protected source diffs before loading.
3. Commit the worker adapter and configure worker_commit from its actual HEAD,
   retaining planner_baseline_commit=8886af9505036b55662d54ae9a108b09121e8c71.
   Configure/measure actual runtime/source/experiment/seal/checkpoint identities.
   Retain all existing seal/runtime/checkpoint validation; reject incompatible
   artifacts instead of disabling checks.
4. Implement strict request framing/schema/binding validation, official PRE
   reconstruction and canonical round-trip checks. Verify versions, condition,
   public order/eligibility and regenerated candidate tuple BEFORE model inference.
5. Use existing Phase 1 consumer and Phase 2 evaluation/argmin unchanged. Private
   alive_wolves flows only into the existing objective. Return one bound candidate
   or explicit error; never return model matrices/score tables by default.
6. Implement and test persistent JSONL lifecycle separately with explicit argv,
   deadlines, clean EOF/exit handling, separate stderr and no retry/reconnect.
7. Run worker-side semantic/tensor/seal regressions and end-to-end process tests
   on the server. No worker implementation, launch, handshake or IPC infrastructure
   is included in this Phase 4B-I patch.
