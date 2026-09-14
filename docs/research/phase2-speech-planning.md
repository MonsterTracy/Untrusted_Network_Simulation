# Phase 2: pure speech planning and ToM decision primitives

Base: `2192b7b4ae83d756b071c653889ce2f8ff147f02`, branch
`ablation/tom-gameplay`. Implementation: `scripts/speech_planning.py`.
No gameplay, Actor, verifier, LLM prompt, vote control or 3WD integration.

## Public plans and candidate ordering

`SpeechPlan(action: PlanningAction, target: str | None)` is a frozen, hashable
dataclass. It stores neither speaker role nor probabilities, score, private
context or prompt text. `public_payload()` returns only action and target.
`for_speaker(public_speaker)` constructs the existing Phase 1 semantic action;
Phase 1 alone implements ontology-to-token mapping and tensorization.

`generate_candidates(parent)` accepts only real public PRE evidence. It uses
Phase 1 public-order derivation to validate the public state; it never accepts
wolf membership. Targets are living non-self seats. SUPPORT and OPPOSE require
at least one real prior public speech on the current day, across phases. A
normal-discussion speech therefore remains eligible in same-day PK; an older
day alone does not qualify. A possible wolf teammate is not filtered out.

`speech_plan_action_then_seat_v1` freezes this ordering:

1. ACCUSE_WOLF, targets in player1...player7 order;
2. CLEAR, same target order;
3. SUPPORT, same order among eligible previously spoken targets;
4. OPPOSE, same eligibility/order;
5. SELF_DEFEND, target=None;
6. NO_COMMITMENT, target=None.

Generation always includes the two targetless plans for a valid discussion PRE,
even at the final speaker. This does not make the opportunity evaluable: Phase 1
cannot evaluate a continuation past the last speaker, and evaluation explicitly
rejects it without fallback or model calls.

## Evaluation and information separation

`evaluate_candidates(parent, candidates, consumer, *, alive_wolves,
current_predictor=None)` accepts the complete generated canonical tuple. It
rejects incomplete or reordered candidate sets, malformed peripheral population,
and unsupported opportunities. Production supplies the existing, already
seal-validated CounterfactualToMConsumer; tests may inject its narrow
probabilities/provenance interface. This dependency seam is not a new sealed
model loader. No model is constructed or loaded in this module.

For each plan it calls only:

```text
consumer.probabilities(real_parent, plan.for_speaker(parent.current_speaker))
```

The private living-wolf tuple is used for peripheral population validation and
the existing `wolf_suspicion_mass` objective only. It is never included in a
plan, candidate generation, hypothetical history, or inference argument. Primary
observer rows are selected by the Phase 1 objective as alive players minus alive
wolves; no role truth is supplied to the encoder. There is no objective or
tensorization formula duplicated in Phase 2.

Each immutable CandidateEvaluation contains the plan, parent digest, Phase 1
consumer provenance, ordering version, Phase 1 WolfSuspicionMass diagnostics,
optional current-state diagnostics, and optional Delta. `primary_score` is
Alive-Conditional Wolf Suspicion Mass. Diagnostics include raw team/self,
conditional self and per-alive-wolf raw/conditional mass.

Current baseline is optional to avoid reaching into a consumer's private loaded
predictor. If supplied, it must be an existing SealedFinalPredictor matching the
consumer's experiment, final seal, checkpoint and temporal condition. Its normal
real-PRE log_probabilities method is called once. Delta is current conditional
team mass minus candidate conditional team mass. Without this predictor, both
baseline and Delta are None, never an invented zero. No arbitrary baseline
matrix or stale baseline record is accepted.

These records are planner-private: per-wolf diagnostics reveal membership.
Only the selected SpeechPlan's public payload is suitable for a future Actor.
This stage neither serializes nor publishes records. A future intervention
manifest should bind this module's source, its ordering/selector versions, the
Phase 1 consumer provenance and objective implementation; the existing model
seal does not cover the new scripts layer.

## Selector seam

`select_minimum_suspicion(candidates, evaluations)` checks exact candidate
coverage, canonical order, common provenance and finite bounded primary scores.
It chooses the smallest primary score, with the first canonical candidate winning
an exact tie. Evaluation records can arrive in any order; tie-breaking never
depends on that order. No tolerance, weights, RNG or LLM are involved.

This function does not call the evaluator or generator. A future No-ToM decision
function can consume the same public candidate tuple without evaluating ToM.
A future 3WD decision function can replace this selector and consume the existing
evaluations. Neither policy is implemented now. No hierarchy or gameplay
orchestrator is introduced for those future consumers.

## Validation status and server commands

**NOT RUN**: no local pytest, CI or model execution in this change. Phase 1 tests
are unchanged. New tests include schema/mapping, frozen order, eligibility,
same-day PK reuse, previous-day exclusion, private-input isolation, consumer
delegation, exact/near ties, malformed evaluation rejection, final-speaker
rejection, selector interchangeability and synthetic sealed integration with
baseline provenance/Delta and artifact byte preservation.

After synchronizing these uncommitted changes and activating the server project
environment:

```sh
cd /home/dell/yuxiao/Untrusted_Network_Simulation

# Phase 2 targeted tests
python -m pytest -q tests/tom/test_speech_planning.py

# Phase 1 regression tests
python -m pytest -q \
  tests/development_publication/test_development_publication.py \
  tests/tom/test_counterfactual_consumer.py \
  tests/tom/test_suspicion_objectives.py \
  tests/tom/test_dataset.py tests/tom/test_model.py \
  tests/tom/test_final_lifecycle.py \
  tests/canonical_collection/test_structured_token_plan.py \
  tests/canonical_collection/test_pre_prefix.py \
  tests/canonical_collection/test_final_capacity.py

# Full suite
python -m pytest -q
```

## Limitations

Scores are suspicion-mass predictions, not win/vote probabilities or causal
effects. Argmin can exploit prediction error under hypothetical histories.
Delta includes progression to the next public turn boundary. Candidate ordering
introduces a deliberate action/seat bias on exact ties. Candidate inference is
sequential and costs one Phase 1 call per plan; an optional baseline adds one
real-PRE call. No caching, batching, retry or silent repair is added. Authentic
PRE provenance and lawful wolf membership remain caller obligations; full
server validation is pending. No files under werewolf or run_random.py change.
