# Phase 1: Counterfactual ToM Consumer V1

This is an independent intervention consumer, not an extension of the original
`SealedFinalPredictor.log_probabilities(real_pre)` contract. Implementation lives
in `scripts/counterfactual_tom.py`; peripheral scoring lives separately in
`scripts/suspicion_objectives.py`. No gameplay, Actor, verifier, selector,
candidate enumeration, campaign, or deployment change is included.

## Input and opportunity contract

`CounterfactualSpeechAction(speaker, PlanningAction, target)` supports ACCUSE_WOLF,
CLEAR, SUPPORT, OPPOSE, SELF_DEFEND and NO_COMMITMENT only. Targeted actions require
a living non-self target. SUPPORT/OPPOSE additionally require a prior public
speech on the current day (including an earlier discussion phase that day).
SELF_DEFEND has no planning target and compiles to a self-directed
`point_as_non_werewolf`. NO_COMMITMENT compiles to an explicit `no_commitment`
token, not an absent action. Wolf membership is neither accepted nor used.

The parent must be a genuine real `AuthoritativePREPrefix` supplied by the caller.
Official validation checks its structure and digest; a digest alone does not
prove an externally supplied object's historical authenticity. The original
public-history/planner validators remain authoritative and unchanged.

V1 supports only a same-day, same-discussion-phase continuation with a known next
speaker. It appends exactly public_speech, speech_action, next turn_start semantic
descriptors. No canonical event, annotation, raw text, perception attempt or belief
observation link is manufactured. The immutable continuation carries schema,
implementation/builder versions, parent/queue digests, speaker/action/target,
next speaker, day/phase, token count, descriptors and its own digest.

### Public queue attestation (remaining integration obligation)

`PublicQueueEvidence` contains the parent PRE digest, day, phase, complete phase
speaker order and an auditable `source_reference`. The order includes previously
started speakers, the current speaker and remaining speakers. Consumer checks:

- parent/day/phase binding, canonical unique living seats;
- complete living membership in ordinary discussion;
- exact agreement with turn_start events since the latest phase_change;
- requested next speaker is the next queue entry and differs from current;
- an exhausted queue raises `UnsupportedOpportunityError`.

PK membership/order is caller-attested; consumer does not derive or authenticate
the public tie-resolution queue from a canonical PRE. Nor can it prove an
unobserved normal-discussion suffix from a digest/reference string. A test
explicitly demonstrates that two otherwise consistent caller-attested future
orders cannot be distinguished by this interface. This is not a claimed proof.

Before gameplay integration, the runtime must bind a public phase order to a
public announcement or a documented deterministic rule plus its public inputs.
Retain that evidence and its reference in a separate intervention audit record.
If the order is known only from a private runtime queue, that is insufficient:
establish its public provenance first or reject the opportunity. Never pass env,
full observations, role assignments or a private queue dump to this consumer.
Final speakers and cross-vote/night continuations are unsupported; no fallback.

## Tensorization and inference

`tensorize_counterfactual` accepts a real parent, one action, explicit next speaker,
queue evidence and `ExperimentCapacity`. It returns continuation metadata and
fresh `PublicTensors`. The real prefix comes from `plan_structured_history` and
official `tensorize_public_pre`; only three appended tokens are encoded locally
using official vocabulary constants. All seven tensor fields and padding are
identical to the corresponding real successor PRE. Capacity includes the frozen
day bound and requires parent_count + 3 <= max_seq_len; no truncation is allowed.

Returned tensor storage is mutable, like official PublicTensors, but is fresh
per invocation and never shared with the real parent or future calls. The
continuation metadata is frozen. Inference accepts neither arbitrary tensors nor
a caller-fabricated continuation, so modified exported tensors cannot be fed back
through the supported consumer interface.

`CounterfactualToMConsumer(experiment, condition)` constructs the unchanged
SealedFinalPredictor, executing original seal, runtime, terminal-checkpoint
verification and model loading. It preserves caller CPU RNG during construction.
Then its own hypothetical method passes only the seven public tensor fields to
the already loaded eval-mode model under inference_mode. Results are CPU [7,7]
log probabilities on the unchanged fixed non-self simplex. `probabilities` is
their exponential. Dead columns are not masked in the model.

This explicit access to the verified predictor's model is the new consumer's
contract, not permission to relax the old predictor's input contract. No
validate_runtime bypass, manifest patch, re-seal or artifact publication occurs.

## Independent provenance

`consumer.provenance.to_record()` returns consumer schema/implementation version,
SHA-256 of this consumer source file (which includes the builder), parent
experiment and final-seal digests, condition, checkpoint digest and builder
version. It performs no writes. The caller should retain this alongside the
continuation digest and the public queue evidence in a future intervention
manifest. The peripheral objective source is separate and is not covered by the
inference consumer's source digest; future scoring provenance must bind it too.

The old runtime digest covers `werewolf/**/*.py` and `run_random.py`. None of
those files is changed. Loading real sealed artifacts still requires the exact
original runtime environment; successful synthetic CPU tests do not claim a
production CUDA seal is loadable on this Mac.

## Peripheral objective

`wolf_suspicion_mass(B, alive_players=..., alive_wolves=..., self_player=...)`
accepts probabilities, not logs, and canonical player-ID tuples. It validates the
fixed non-self simplex and returns Alive-Conditional Wolf Suspicion Mass plus
raw team/self and per-living-wolf raw/conditional diagnostics. For each alive
non-wolf observer it first divides wolf mass by that observer's total alive
non-self mass, then averages observers. It never alters B or the ToM target/model.
Empty observer/wolf sets and zero denominators fail; no epsilon or repair occurs.
These are suspicion-mass scores, not win or vote probabilities.

## Tests

Run with an environment providing the existing project's torch/transformers and
pytest dependencies. On macOS use a resolved temporary path because canonical
artifact validation intentionally rejects symlink ancestors:

```sh
TMPDIR=/private/tmp PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  tests/tom/test_counterfactual_consumer.py tests/tom/test_suspicion_objectives.py
```

The oracle uses explicit synthetic public event fixtures and real SpeechPerceiver
parsing against a scripted test backend. Its six mappings are independent of the
consumer mapping. Twelve tensor cases cover all six actions in discussion and
pk_discussion. Synthetic sealed lifecycle tests compare all six successor outputs
in both phases under both temporal conditions, using real gate checks, real
checkpoint loading and the real model. Only the test clean-checkout revision
provider is isolated, following existing lifecycle-test conventions. Tiny local
fixture training/sealing writes exclusively to pytest temporary directories; no
existing experiment/model seal is loaded, trained or rewritten.

Additional tests cover gate failures and actual disposable artifact corruption,
public-only model kwargs, capacity, eligibility, immutable metadata, canonical
writer rejection, repeatability, non-contamination and objective invariants.
Scripted parser outputs test tensor/contract equivalence, not natural-language
perception accuracy or counterfactual scientific validity.

### Local verification, 2026-09-14

Base HEAD: `33d580cdf0700f6e38ae5b14648fcfdaa9be2b9a`.
Python: `/Users/name_yuxiao/anaconda3/envs/3wd/bin/python`.
The following completed with **88 passed in 50.85s**:

```sh
TMPDIR=/private/tmp PYTHONDONTWRITEBYTECODE=1 \
  /Users/name_yuxiao/anaconda3/envs/3wd/bin/python -m pytest -q -p no:cacheprovider \
  tests/tom/test_counterfactual_consumer.py tests/tom/test_suspicion_objectives.py \
  tests/tom/test_dataset.py tests/tom/test_model.py tests/tom/test_final_lifecycle.py \
  tests/canonical_collection/test_structured_token_plan.py \
  tests/canonical_collection/test_pre_prefix.py \
  tests/canonical_collection/test_final_capacity.py
```

`git diff -- werewolf run_random.py` was empty. A before/after SHA-256 inventory
of all 57 protected Python files also matched exactly (including file set).
Original untracked research material was left untouched. No formal artifacts,
Agent, environment, prompt, recorder, selector or gameplay wiring were changed.
