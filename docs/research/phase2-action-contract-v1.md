# Phase-2 Action Contract V1

Status: structured action semantics and offline availability only. This contract does not execute speech, infer Q, call an LLM, alter a ballot, or route a Three-Way Decision. The frozen mapper remains the source of uncalibrated `p_tilde`, not calibrated `p_hat`.

## Current decision population

The acting player is an alive wolf at a real `speech` or `speech_pk` PRE. `known_wolves` is exactly the wolf team legally known to that actor. `J` is the canonical-seat-ordered alive non-wolves in the competition set: all alive seats in ordinary speech, or the public normal-vote maximum-tie set in PK. No bussing. Context construction uses only the current PRE's public events and the actor's known wolf team. It validates the cyclic public speaker order; later votes, exile, response content, belief reports, and outcomes are absent.

## Structured actions

| Action | commitment | rejected | redirect | information request | vote intent |
| --- | --- | --- | --- | --- | --- |
| `PUSH(j)` | `j` | null | null | null | `j` |
| `REDIRECT(j,k)` | `k` | `j` | `k` | null | `k` |
| `PROBE(j,r)` | null | null | null | direct request `r` to `j` | `NO_STANCE` |

`PUSH` requires `j in J`. `REDIRECT` requires distinct `j,k in J`. Its first-version selector is `argmax_{k in J\{j}} p_tilde(t,k)` with exact ties broken by canonical seat order. The selector requires one finite probability for every alternative; it may also receive the current candidate's probability, but no illegal target. It does not call the mapper. No alternative means unavailable.

`PROBE` has the sole request type `CURRENT_SUSPICION_BASIS`: publicly ask `j` to explain their current principal suspicion or vote judgment and its public basis. Both `target_j` and `addressee_j` equal `j`. `NO_STANCE` without this request is not Probe. Natural-language realization and verification remain future work.

## Same-phase Probe continuation

Let `q` be the current phase's public speaker order. V1 Probe is available only if `pos(w) < pos(j) < pos(w_next)`, where `w_next` is the first living wolf speaker after `j` in `q`. `continuation_actor = w_next`. The expected observation window contains only the public speech turns strictly after `w` and strictly before `w_next`, including `j`; the real event IDs and content are unknown at planning time. Both ordinary and PK use this rule. A second current-wolf turn, a future normal-vote tie, or a later phase cannot create V1 eligibility.

The request's structured legality is separate from its eventual execution. A future execution audit must separately represent `request_executed`, `response_opportunity_reached`, `public_observation_realized`, `reconsideration_executed`, and `information_gain`. An informative answer is not guaranteed and is not required for request execution. No future outcome is consulted to determine availability.

## Audit boundary

`Phase2SemanticPlanV1` and `Phase2ActionAuditV1` serialize deterministically. The current verifier checks only fields and legality, so `execution_valid` means **structured plan valid**, not language execution success. Fields for generated text, perceived actions, real request occurrence, observed public event IDs, and reconsideration remain null. Future execution code must not reinterpret this boolean as a successful intervention. Actual votes remain a separate downstream decision.

## Read-only availability

`scripts/analyze_phase2_action_availability.py --publication PATH` opens the frozen development publication and role sidecar, checks their digests, visits current wolf speech PREs, and prints JSON to stdout. It computes `{P}`, `{P,N}`, `{P,B}`, `{P,N,B}` counts, phase breakdowns, Probe PRE and row rates, teammate continuation, and the candidate's distances between wolf positions. It reads no sealed Q because availability only needs the existence of a legal alternative; `p_tilde` is required later to select the Redirect target. It creates no formal artifact. The frozen server publication path is the command's default; actual 1500-game counts require running it on a host with that publication.

## Deferred before language realization

The V1 speech ontology has no explicit rejection or information-request action. A Phase-2 language actor and semantic verifier must therefore check that Redirect rejects `j` while committing to `k`, and that Probe visibly requests information without an exile commitment. These extensions must stay isolated from the completed NoToM/+ToM gameplay and V1 canonical annotation contract.
