# Phase 3B: single-perception verified speech commit

Baseline: d8d01b7a67099a1a2a992b522fb51fecd37e3fd3.
Validation: NOT RUN. No pytest, model inference or external LLM service was run.

## Existing path and first mutation

run_random.eval prepares canonical PRE/observer reports/handoff through
before_agent_act before invoking the Agent. The recorder first mutates its
pending slot there, and then its prefixes/observations. after_agent_act stores
the submitted raw action. env.step -> next_phase parses the speech, appends the
public_speech event, builds the canonical annotation and marks actual perception
attempts, appends the game log, then advances speech queue or enters vote.
Appending public_speech is the first lasting gameplay mutation in that speech
branch; Original parse failure can leave an error annotation/public event.
after_env_step creates submitted-action evidence and clears the pending slot.

## Explicit constrained path

scripts.verified_speech_commit.realize_and_commit requires an already collected,
authentic pending PRE/handoff. It does not collect beliefs, create substitute
observations or select a plan. The transaction starts with that pending state.
The caller supplies only SpeechPlan.public_payload(), env, existing Actor and
recorder. Public context comes from the existing public claim catalog builder.

The existing Phase 3A realization runs in the recorder's action audit context;
its sole parse_with_audit invocation runs in the existing perception audit
context. This optional context manager is the only Phase 3A API extension.
Existing Actor quality retries and parser-internal bounded retries remain;
semantic mismatch never triggers retry, another candidate or fallback.

## Envelope

VerifiedSpeechCommit binds exact speech, expected canonical V1 action (including
speaker), day, runtime phase, public history digest, parser backend/model
identity and the complete successful SpeechParseAuditResult. canonical_json_bytes
and sha256_bytes bind these fields to immutable bytes and a digest. This is an
intervention binding, not a replacement canonical annotation schema or digest.

Commit checks both the live nested audit content and all envelope fields against
that snapshot/digest, then materializes an isolated audit from the snapshot.
It checks current speaker/day/phase/history/parser identity. Mutation of speech,
actions or attempt content is rejected; stale/replayed opportunity is rejected.
No second parse is involved. This binding detects post-verification mutation;
it is not an authentication boundary against arbitrary trusted Python code
constructing a new envelope and digest from scratch.

## Transaction and canonical semantics

Env stages only speech-mutated lists (public events, annotations, logs, speech
queue, vote queue) on a shallow environment copy. Phase/current actor assignments
stay on that copy. It reuses next_phase's existing annotation constructor and
queue/phase rules, injecting the isolated verified audit and deferring semantic
attempt marking. No game RNG is consumed by the normal speech branch or Actor/
parser path; Actor/parser do not receive an env RNG reference.

CanonicalGameRecorder.commit_verified_speech checks pending PRE identity, stages
the env step, and invokes existing after_agent_act/after_env_step on a recorder
copy with independent pending/submitted/raw-action values. Only after both
constructions succeed are semantic call statuses marked and env/recorder state
published. No speech/annotation/submitted-action evidence is published on a
semantic, binding or staged-construction failure. The existing pending PRE and
its observer evidence remain unchanged, rather than being fabricated or erased.

Actual Actor/perception call-budget and backend-call audit records are retained
on failed attempts; those real external calls are not rolled back. Backend
semantic-status marking is an existing external callback and is not a general
transactional I/O system. The state-publication guarantee concerns the listed
gameplay and recorder speech-evidence values, under synchronous single-threaded
execution, not crash recovery or concurrent env mutation.

Default env.step and Original Agent need no SpeechPlan. Direct
step_verified_speech provides explicit env-only commit; callers using canonical
recording must use the recorder wrapper. run_random remains unchanged.

## Tests

New tests use real env/recorder and existing scripted perception/observer
fixtures. They cover successful one-perception commits and exact text/actions;
wrong action/target, extra action, parser failure and NONE in normal/PK phases;
full listed state plus RNG preservation; nested/text/boundary tampering;
recorder-construction failure; all six semantic mappings across Phase 1
continuation, SpeechPlan, Phase 3A expected action and committed annotation;
normal/PK default-versus-verified annotation/log/queue/progression equivalence,
including the final speaker; and envelope field exclusion of planner diagnostics.

## Provenance and next phase

Changed frozen-runtime sources: werewolf/envs/werewolf_text_env_v0.py,
werewolf/canonical_collection/runtime.py, and new
werewolf/speech/verified_commit.py. Old sealed runtime provenance will no longer
match this checkout. No validation is disabled and no ToM consumer is imported
into gameplay/Actor. Phase 1 imports occur only in semantic equivalence tests.

Further work must wire an explicit constrained mode into a runner, preserve
real PRE/report collection and correct audit contexts, and implement frozen-ToM
runtime separation/transport if planning is later connected. None of planner,
selector, vote control, 3WD, MCTS, IPC or a gameplay campaign is implemented here.
Server validation remains required before proceeding.

```sh
# From server repository root
python -m pytest -q tests/tom/test_verified_speech_commit.py
python -m pytest -q tests/tom/test_speech_realization.py
python -m pytest -q tests/tom/test_speech_planning.py
python -m pytest -q tests/tom/test_counterfactual_consumer.py tests/tom/test_suspicion_objectives.py
python -m pytest -q tests/agents/test_agent_backend.py tests/canonical_collection/test_runtime_game_evidence.py
python -m pytest -q
```
