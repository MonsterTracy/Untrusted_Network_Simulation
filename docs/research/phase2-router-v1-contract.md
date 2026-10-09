# Phase-2 Router-v1 execution contract

Router-v1 and its matched control use **the first Formal Probe-eligible PRE**
in each game. Both require the existing `probe_candidate_pool(context)` to be
nonempty. Earlier terminal-only opportunities do not consume the intervention.
The research policy is `PROBE_THEN_REDIRECT`; control is `IMMEDIATE_REDIRECT`.
Neither selects Push. No opportunity leaves the original agent policy intact.

## Entry and evidence

Construct `Phase2RouterPolicyV1(strategy, candidate_sampling_id, candidate_seed,
source_revision)` and pass it as `plan` to `OnlineTerminalPilotRunnerV1`, along
with the existing frozen predictor, mapper, language backend and reference
inputs. A synchronous `record_evidence(kind, evidence)` sink is mandatory. The
sink must durably acknowledge each write or raise; evidence objects expose
`to_record()`, while game-start/preparation-failure evidence is a dictionary.
The caller owns the immutable run/source binding and append-only evidence
destination; this hook does not implement campaign preparation or resume.

Call `start_game(recorder.game_id)`, then install the runner in the existing
`run_random.eval(..., canonical_recorder=..., call_audit=..., online_pilot=runner)`
canonical path. No second gameplay loop exists. `handle_pre` returns a committed
canonical step, or `None` to allow baseline action generation. `None` after
language invalid is **not** NOT_APPLICABLE: the assignment and invalid lifecycle
remain recorded. Structural errors preserve failure evidence and raise.

The initial assignment and T1-attempt record precede language/backend execution.
Backend calls and every lifecycle transition use the same evidence sink. Each
game has at most one initial assignment, including failures; a used game ID
cannot restart in the same executor. Randomized pilot ledgers/preflight/CLI do
not admit Router policies. Experiment admission, clean source/artifact pins,
campaign budgets and durable restart validation belong to the next experiment
design, and are not satisfied by passing a randomized-pilot preflight.

## Policy and candidate provenance

Policy identities are `phase2-router-v1` and
`phase2-router-v1-matched-control`. Initial treatment source is
`deterministic_policy`, probability 1. The policy record and assignment digest
bind strategy, source revision and common candidate sampling identity. Historical
randomized assignment records retain their original schema and probabilities.

`select_probe_candidate` retains the existing pool and uniform SHA256 rejection
sampling rule. Router sampling keys bind the common sampling digest and seed,
game ID, boundary ID, PRE digest and ordered pool. The policy arm is excluded
from the sampling digest; its identity is included in the deterministic treatment
binding. In Router selection records the legacy field `plan_digest` denotes
this candidate sampling digest, explicitly identified by the policy record.
The legacy treatment field `randomization_key` holds the deterministic binding
hash; it is not treatment randomization. Backend sidecars retain their existing
`pilot_id` field name and carry the explicit Router policy ID there.

Identical **complete** sampling inputs give identical candidate selections.
Equal seeds alone do not give identical PREs or candidates across runs. This
implements matching of eligibility and sampling semantics, **not strict paired
blocks or shared counterfactual PREs**. Such an experiment requires a separately
defined shared pre-treatment identity/state mechanism and is currently blocked;
no replay or state copying is introduced.

## Continuation and failures

The existing Probe executor/verifier/canonical commit are reused. Only a valid,
committed T1 schedules T3. Invalid T1 cancels T3 and returns to baseline while
retaining the assignment and its day consequence. At T3 the original candidate,
phase, actor and observation schedule stay fixed. Current real PRE, Q, mapper
panel and Redirect target are recomputed. The treatment uses the existing
`strategy_continuation` parent binding. Frozen T1 proof may be revalidated; there
is no new initial policy decision or assignment at T3. T3 invalid and structural
failures retain their existing semantics. Baseline voting remains unchanged.

No score threshold, alpha/beta, M2 gate, risk model, fallback or new action
definition participates in this contract. The policy hook is locally testable;
a gameplay experiment still needs its own pre-registration and admission.
