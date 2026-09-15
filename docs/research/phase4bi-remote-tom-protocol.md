# Frozen-ToM JSONL adapter

Development stays on ablation/tom-gameplay. The model executes in a server
checkout detached at 8886af9505036b55662d54ae9a108b09121e8c71 with two copied
adapter files. No other checkout or Git state was changed for this implementation.
Tests: NOT RUN. No model or real worker was launched locally.

## Minimal files and responsibilities

* scripts/remote_tom_protocol.py: shared strict JSON, request/response schema,
  identity and canonical bindings. It uses stdlib plus the existing frozen
  artifact_io canonical JSON/SHA helpers; no planner/Actor/env imports.
* scripts/remote_tom_worker.py: frozen-checkout bootstrap, official PRE reader,
  validation, existing planner dispatch and sequential stdin/stdout loop.
* scripts/remote_tom_provider.py: existing Phase 4A callable boundary, public PRE
  encoding, private objective supplier and injected JsonLineTransport.

The shared protocol is necessary because client and worker must not maintain
separate binding/schema implementations. Worker dependencies only allow model-free
unit tests to spy on existing planner calls; there is no service hierarchy.
No transport framework, reconnect, retries, heuristic or fallback is added.
JsonLineTransport remains the existing injected client transport contract: one
newline-terminated JSON object in/out, bounded waits and explicit timeout/EOF/
process failures. A process launcher is not added by this worker patch.

## Final identity

WorkerIdentity requires exactly:

* planner_baseline_commit: 8886af9505036b55662d54ae9a108b09121e8c71;
* adapter_digest: actual two deployed adapter source files;
* runtime_digest: SHA256 of canonical JSON of the verified experiment runtime record;
* source_digest: that verified runtime record's implementation_digest;
* experiment_digest: verified final experiment manifest digest;
* seal_digest: original consumer provenance parent_final_seal_digest;
* checkpoint_digest: original consumer provenance checkpoint_digest;
* condition: implicit or explicit_day_phase.

The baseline is a 40-character lowercase Git SHA; digest fields are 64-character
lowercase SHA-256 values. Required fields cannot be None/unknown/latest. The
worker measures/verifies its own identity; CLI accepts only experiment path and
condition, not user-declared digests.

adapter_digest = SHA256 of the concatenation, in this exact order:

1. scripts/remote_tom_protocol.py
2. scripts/remote_tom_worker.py

For each file append UTF-8 relative filename, NUL, ASCII decimal byte length,
colon, then raw file bytes. Runtime identity covers frozen model execution;
adapter identity covers only the RPC code. No additional repository hash is
introduced. Protocol parsing verifies expected identity by exact dictionary
comparison, including absence of unknown fields.

## Bootstrap

Run with python -m scripts.remote_tom_worker from the frozen checkout root.
Bootstrap checks that cwd is the adapter checkout, actual HEAD equals the fixed
baseline, HEAD is detached, and protected files have no tracked diff against
HEAD. Original runtime validation additionally checks all runtime Python source
content and numeric environment.

open_final_experiment validates the experiment artifact/config/capacity and
initial-state records. open_publication verifies the development publication;
its digest, publication ID and game IDs must match the experiment's recorded
publication. CounterfactualToMConsumer is constructed ONCE; its unchanged
SealedFinalPredictor executes verify_final_seal, validate_runtime,
verify_final_terminal and load_model_state. No training entrypoint or
actual_clean_revision training requirement is bypassed/called to accommodate
adapters. Adding the two adapter files does not change the original runtime
source inventory (werewolf/**/*.py and run_random.py).

Only after successful bootstrap is the measured identity logged to stderr and
service started. Any bootstrap failure prints a diagnostic to stderr and exits
nonzero. Artifact paths/publication paths and the numeric runtime must match the
sealed experiment; errors are not repaired. Bootstrap/model initialization is
not repeated per request.

## Request and PRE

Version remains remote_tom_plan_v1; incompatible old field shapes are rejected,
with no compatibility fields. Required request keys:
protocol_version, request_id, request_digest, game_id, boundary_id, parent_digest,
parent_pre, speaker, day, phase, candidates, candidate_order_version,
public_queue_rule_version, alive_wolves, condition, expected_worker_identity.

parent_pre is the existing AuthoritativePREPrefix.to_record(), with all event,
annotation/attempt, observation-link, boundary and digest data. The pinned official
werewolf.canonical_collection.game_bundle._prefix_from_record is used directly:
it reconstructs history/annotations using canonical constructors, reconstructs
the PRE, and requires record equality. The worker additionally invokes
validate_authoritative_pre_prefix and exact canonical-byte round-trip comparison.
The internal helper's use is deliberate and pinned to this baseline. No fake PRE,
publication or perception evidence is built by the adapter.

Request binding uses the existing canonical JSON/SHA-256 contract:
1. Build the complete record except request_id/request_digest.
2. request_id = "rtom-" + SHA256 of those bytes.
3. Add request_id, then request_digest = SHA256 of the resulting canonical bytes.
4. Send with request_digest and one newline.

Both ID and digest are PLANNER-PRIVATE: alive wolf seats are enumerable. They
may appear only in RPC or planner-private logs/sidecars, never Actor/SpeechPlan,
public events/game logs or agent-visible SpeechTrace. Existing public game and
boundary IDs remain the public correlation mechanism.

alive_wolves is a unique nonempty canonical seat-ordered list. Duplicate,
invalid or unsorted wire IDs fail. Provider canonicalizes supplier ordering.
The worker checks living membership, acting-speaker membership and a nonempty
living non-wolf population. No complete role table is sent; the trusted objective
supplier is responsible for lawful membership.

## Before any candidate evaluation or model forward

Worker.handle performs, in order:

1. Shared exact schema, request ID/digest, protocol and expected identity checks;
2. fixed baseline/adapter/condition and candidate-order/queue-rule versions;
3. canonical wolf IDs and official PRE reconstruction/validation;
4. game/boundary/digest/speaker/day/phase agreement and living objective population;
5. frozen public phase-order derivation, reject final same-phase speaker;
6. existing capacity validation, including room for three continuation tokens;
7. local generate_candidates and exact ordered incoming candidate equality.

Only then: evaluate_candidates(parent, candidates, existing_consumer,
alive_wolves=...) -> select_minimum_suspicion(candidates, evaluations).
The worker does not implement tensorization, objective, probability interpretation,
argmin or tie-breaking. It does not request a baseline prediction/Delta.

## Responses and loop

Success keys exactly: protocol_version, request_id, request_digest, status=ok,
selected_candidate_index, selected_plan, worker_identity. Provider checks binding,
identity, non-bool integer index/range and exact indexed plan, then returns the
original candidate object. No matrices/scores/alive_wolves/diagnostics are returned.

Error keys: protocol_version, status=error, error_code, error_message,
worker_identity after bootstrap, and request_id only if shared binding validation
succeeded. The provider rejects worker errors even when no trusted request ID
is available. Error messages are generic, not a copy of private request content.
Unknown fields and duplicate JSON keys at any depth are rejected.

One input line produces one output line and flush. Invalid request/JSON/planning
errors do not end the loop; the next request is handled independently. Bootstrap
failure never enters the loop. Planner/import prints are redirected to stderr;
stdout contains protocol JSON only. No recovery decision is made by gameplay:
provider errors still terminate eligible execution via unchanged Phase 4A logic.

## Static boundaries and tests

No protected source changes: werewolf/**, run_random.py, Phase 1/2 planner and
objective, eligibility, Actor, verifier and verified commit remain unchanged.
No sys.path/PYTHONPATH modification, cross-checkout imports, pickle, HTTP/gRPC,
threads or asynchronous server framework. The provider imports only the protocol
and existing public PRE encoding helpers, not worker/planner/model code.

Tests use a fake consumer with real Phase 2 evaluator/selector, spies, official
PRE reconstruction and in-memory StringIO JSONL transport. They cover both phases,
invalid request zero evaluation/forward, candidate regeneration, identity/adapter/
digest/PRE/order/population tampering, response agreement, error then success,
stdout isolation, adapter content hashing, bootstrap failure exit and provider
round-trip. Existing Phase 4A failure/original-mode tests remain in the regression
suite. No real model, worker process or frozen checkout is required by new tests.

```sh
python -m pytest -q tests/tom/test_remote_tom_worker.py tests/tom/test_remote_tom_provider.py
python -m pytest -q tests/tom/test_constrained_speech.py
python -m pytest -q tests/tom/test_verified_speech_commit.py tests/tom/test_speech_realization.py tests/tom/test_speech_planning.py tests/tom/test_counterfactual_consumer.py tests/tom/test_suspicion_objectives.py
python -m pytest -q
```

## Server deployment outline (not executed)

Prepare the existing/planned server worker directory as a detached checkout of
8886af9505036b55662d54ae9a108b09121e8c71. Do not cherry-pick gameplay changes.
Copy ONLY these two current adapter files into its scripts directory:
remote_tom_worker.py and remote_tom_protocol.py. No speech_versions.py or current
Phase 1/2 files are required. Verify their adapter_digest, retaining all frozen
protected source bytes unchanged.

```sh
# From the server gameplay repository; frozen directory must already be prepared.
cp scripts/remote_tom_worker.py scripts/remote_tom_protocol.py /home/dell/yuxiao/Untrusted_Network_Simulation-worker/scripts/
cd /home/dell/yuxiao/Untrusted_Network_Simulation-worker
# Use the exact Python/numeric environment of the sealed experiment.
python -m scripts.remote_tom_worker --experiment /absolute/path/to/verified-final-experiment --condition implicit 2>worker-private.log
```

stdin/stdout are the private JSONL pipe. Obtain measured identity from bootstrap
stderr and configure the gameplay provider with that exact identity; do not
invent digests. The injected transport must own process lifecycle/deadlines and
keep stderr separate. Real model compatibility and deployment remain to be
verified on the server. The adapter does not silently repair a mismatched seal,
runtime, publication path, checkpoint or request.
