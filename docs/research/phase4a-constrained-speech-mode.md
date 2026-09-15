# Phase 4A constrained speech opportunity wiring

Base: e00e40606c17dce4299d42fd94db262483b3c619.
Tests: NOT RUN. No model, IPC, external service, commit or push.

## Runner insertion and usage

run_random.eval identifies speech using the existing speech/speech_pk set.
before_agent_act still collects real PRE, alive observer reports and handoff.
The new insertion is immediately after handoff validation and before the normal
Agent action. Night/vote/skill turns continue through the existing code.

Default planning_mode="original" executes the existing lifecycle. It does not
import the constrained helper, generate candidates, pre-parse speech or create
trace entries. An explicit original mode behaves identically.

Programmatic constrained invocation:

```python
trace = []
result = eval(env, agents, roles,
    canonical_recorder=recorder, call_audit=audit,
    planning_mode="constrained", plan_provider=provider,
    speech_treatment=SpeechTreatment(roles=("Werewolf",)),
    ablation_trace=trace)
```

SpeechTreatment is defined in scripts.constrained_speech. Constrained mode
requires a callable provider and a caller-owned list for the sidecar trace.
There is no shipped provider policy, CLI campaign or config-file integration.

## Assignment and public eligibility

SpeechTreatment contains explicit canonical player IDs and/or role names;
assignment is their union. In constrained mode only, omitted treatment defaults
to roles=("Werewolf",). For seat-only treatment set roles=(). The runner evaluates
assignment using its lawful current role state. No role data goes to the gate,
provider, SpeechPlan or Actor.

public_eligibility(parent) reuses the unchanged Phase 1 public cyclic-order rule
and Phase 2 canonical candidate generator. Invalid PRE/rule evidence raises an
error; it is never classified as a routine untreated opportunity. A nonempty
candidate tuple plus a same-phase next speaker is eligible. The final speaker
is explicitly LAST_SAME_PHASE_SPEAKER, even though targetless candidates exist.
This is the common C0/C1 population, independent of model/provider outputs.

Provider boundary: Callable[[PublicOpportunity], SpeechPlan]. The frozen
opportunity holds parent PRE, public speaker and generated candidate tuple;
identity/day/phase are available in PRE. A returned value must be a SpeechPlan
present in that tuple. No env, Actor, assignment, roles, scores or probabilities
are passed. Future private planner state requires a separately designed channel.

## Four paths

* Original mode: existing generation -> env.step -> recorder lifecycle.
* Constrained, assigned and publicly eligible: provider once -> Phase 3B
  realization/perception/verification/commit once. Runner consumes the returned
  step result, increments its step and continues. It never calls Agent/env.step
  again for this opportunity.
* Nonassigned or last same-phase speaker: provider zero calls; record a declared
  untreated reason, then execute the existing original speech path.
* Invalid public evidence, provider error/invalid plan or eligible execution
  failure: append failure trace, raise ConstrainedExecutionFailure and let the
  existing canonical runner produce attempt failure. No retry, fallback,
  alternate candidate or unverified env.step follows.

Untreated trace statuses describe routing decisions, not verified-commit success.
An untreated Original-path failure remains an ordinary canonical attempt failure.

## Sidecar trace

SpeechTrace fields: game_id, boundary_id, parent_digest, speaker, day, phase,
treatment_assigned, public_eligible (None if public validation failed),
eligibility_reason, planning_mode, selected_plan (optional), status, error_type.
Statuses are UNTREATED_NOT_ASSIGNED, INELIGIBLE_FOR_CONSTRAINED_TREATMENT,
VERIFIED_COMMIT_SUCCEEDED and CONSTRAINED_EXECUTION_FAILED. Only exception class
names are recorded, not potentially private messages. The caller retains the
list on failure. No trace data is inserted in canonical/public game evidence.

## Import and provenance isolation

Phase 2 imports Phase 1 to share canonical plans/order. Phase 1 previously
imported SealedFinalPredictor at module import time. Its import is now local to
CounterfactualToMConsumer.__init__; all consumer validation and inference logic
is unchanged. Phase 4A never constructs that consumer or invokes evaluation.
This minimal import change avoids duplicating public rules or changing their
semantics. A fresh-process test asserts importing runner and constrained helper
does not load werewolf.tom.final_evaluation. Numeric/tensor modules may still be
imported transitively; this is not a model load or frozen-worktree import.

No werewolf/** source changes are made in Phase 4A. run_random.py changes, and
the Phase 1 consumer source digest changes due to the local import. The existing
runtime remains different from older sealed checkouts. No validation, q,
population, training, OOF/final lifecycle, env or Phase 3B internals change.

## Tests / server commands

Tests use real env/recorder, actual parser and scripted audited backends.
They cover normal/PK eligibility and execution, shared public eligibility,
private assignment separation, successful single commit, provider/semantic
failure, wrong provider return, default/nonassigned original paths, final-speaker
untreated progression, no double call, vote/night bypass, trace separation,
default-versus-explicit original canonical equality, and sealed import isolation.
All existing regression tests remain unchanged. Run from server repo root:

```sh
python -m pytest -q tests/tom/test_constrained_speech.py
python -m pytest -q tests/tom/test_verified_speech_commit.py
python -m pytest -q tests/tom/test_speech_realization.py tests/tom/test_speech_planning.py tests/tom/test_counterfactual_consumer.py tests/tom/test_suspicion_objectives.py
python -m pytest -q tests/agents/test_agent_backend.py tests/canonical_collection/test_runtime_game_evidence.py
python -m pytest -q
```

## Phase 4B minimum interface (not implemented)

An explicit process request should bind request/opportunity ID, parent PRE digest
and serialized public PRE, canonical candidates/order version and requested
frozen runtime/model identity. Any lawful private objective state must be a
separate planner-only field/channel and never forwarded to Actor. Response should
bind that request/digest and return one candidate plan or an explicit error.
Gameplay must check membership/identity and fail closed on mismatch or worker
failure; the public eligibility gate remains unchanged. Transport, worker launch,
timeouts, replay/audit bindings and private sidecar results require a separate
Phase 4B design. No cross-worktree in-process import is used here.
