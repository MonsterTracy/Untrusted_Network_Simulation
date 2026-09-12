# Paired final-fit lifecycle

Development OOF remains available for protocol qualification and development
analysis. Final Fit uses all Development Publication games without validation
partition or model selection. The scientific decision is
[ADR 0022](adr/0022-seal-paired-final-models-before-independent-evaluation.md).

```text
frozen ExperimentConfig + Development Publication
  → final_experiment.prepare_final_experiment
      → derived validator capacity + shared immutable temporal table
      → all-game seven-shift schedule + paired initial state + Primary masks
  → final_training (shared training.fit_steps)
      → implicit terminal + explicit terminal + recovery chains
  → final_model_seal.json
  → independent Final Evaluation Publication (same public record codec)
  → immutable final_evaluation_consumption.json
  → whole-set capacity validation
  → SealedFinalPredictor (shared public tensorization/model)
  → shared scoring / game-macro summaries / paired bootstrap
  → immutable final_evaluation/report.json
```

`werewolf/cli.py` owns every command. With the existing explicit storage profile:

```sh
uns prepare-final-experiment --publication DEV_ID --protocol configs/formal/development-experiment-v1/protocol.json --destination FINAL_ID
uns run-final-fit --experiment FINAL_ID
uns seal-final-models --experiment FINAL_ID
uns publish-final-evaluation --experiment FINAL_ID --collection FINAL_COLLECTION_ID --runtime-config configs/runtime/local-qwen35-9b.yaml --destination FINAL_PUBLICATION_ID
uns run-final-evaluation --experiment FINAL_ID --publication FINAL_PUBLICATION_ID
uns validate-artifact experiments/FINAL_ID/experiments/experiment
uns validate-artifact publications/FINAL_PUBLICATION_ID --experiment FINAL_ID
```

Use `--resume` explicitly for interrupted `run-final-fit` or
`run-final-evaluation`. A training exception is not resumable. A completed seal
rejects all further training calls. A completed evaluation resume validates its
stored result; it does not run model selection or generate another report.
An incomplete or failed evaluation does not validate as a complete experiment.
The final source collection must be reserved for independent evaluation; do not
inspect it during development. The publication command verifies the model seal
before replaying that collection. Low-level artifact codecs are not access
control boundaries.

All outputs use the existing artifact-root layout: `canonical/`, `publications/`
and `experiments/ID/`. Immutable final inputs occupy
`experiments/ID/experiments/experiment/final_experiment_manifest.json`; training
records, recovery, model seal, consumption and reports live in the owning
`runs/<experiment-digest>/`. No automatic artifact discovery is performed.
The frozen formal protocol is reused without alteration. Its `source_revision`
is the declared implementation baseline; final provenance additionally records
and requires the actual clean preparation HEAD and runtime implementation hash.

`SealedFinalPredictor(experiment, condition).predict(prefix, observer)` provides
label-free inference. `prefix` is a validated Authoritative PRE carrying its
complete public event history and V1 annotations; `observer` is a zero-based
seat. Observation-link identities are part of the PRE contract but observation
payloads, belief labels and role truth are not needed. The output has seven
positions, exact zero at self, and a unit sum over the other six positions.
There is no online gameplay integration. Both input capacity checks apply.

## Validator-derived capacity

The public validator permits sparse histories, unlike the runtime. To reach a
Day-d discussion PRE, it requires initial Night0, d death announcements,
d discussion phase changes, d-1 vote phase changes, d-1 night phase changes,
and the current turn start. Each costs at least one structured token:

`1 + d + d + (d-1) + (d-1) + 1 = 4d`.

A sparse history attains equality without speech annotations or vote payloads.
Thus `day <= floor(L/4)` whenever the entire validated PRE fits L. Day256 has
1024 tokens and Day257 has 1028. The final immutable table covers 0..256 for
L=1024. Sequence capacity is still independently checked; the day bound cannot
make an oversized history valid. The derivation refuses different contract
versions until the rule has been reviewed.

## Current runtime theorem (not protocol control)

This theorem depends on the current `WerewolfTextEnvV0` implementation, its
parity terminal rule, mandatory events and planner v1. Its regression tests use
the actual runtime transitions, not a synthetic replacement engine. Any future
runtime change requires rechecking this proof against its implementation hash.

Let a_i be the number alive at the first discussion PRE of day i. For a PRE to
exist, at least one wolf and strictly more nonwolves remain; hence a_i >= 3.
There is no resurrection. Night0 kills at most two distinct players (one wolf
team target and one poison target), so a_1 >= 5. A no-kill Night0 is legal.

For the first PRE on day d, with no optional PK/actions, the token count is:

`T(d) = 6d + 5 - a_d + 3 * sum(a_i, i=1..d-1)`.

The terms count all phase changes, turn starts/speeches, vote-result headers and
one ballot per living voter including abstention, exile-result headers, death
announcements, and one token for every removed player. In particular each
removal costs one token now; it can only reduce the three-per-player cost of
later completed days. Stable three-alive days cost 15, including an empty exile
result and an empty death announcement. PK adds events/ballots/speeches and
cannot remove more than the single exile available without PK. Extra V1 actions
add tokens. Later same-day PREs add tokens to the cumulative first PRE.

- Day1: `T=11-a_1`, so the minimum is 4, reached by no kill and no poison.
- Day2: `T=17-a_2+3a_1 >= 17+2a_1 >= 27`. Reach 5 alive on Night0,
  then abstain and no-kill; no further removal cost is incurred.
- d>=3: `a_1>=5` and `a_i>=a_d` for i>=2 give
  `T >= 6d+20+(3d-7)a_d >= 15d-1`. Optional stages cannot lower this.

The last lower bound is attainable. Assign player1/2 wolves, player3 Seer,
player4 Witch and player5/6/7 villagers. Night0 wolves kill player5 and Witch
poisons player2. Five players remain, with one wolf. Day1 exiles player6
(the candidate abstains); four remain. Night1 wolves kill player7 and Witch
passes. Day2 has player1/3/4 alive, with one wolf and two nonwolves, so no terminal
check fires. Its first PRE costs 29, not the global Day2 minimum. Subsequently
all votes abstain, the wolf chooses no-kill, Witch passes and Seer chooses legal
remaining checks, then pass after exhaustion. From Day2 onward the state is
stable and costs 15 tokens per full day, attaining `15d-1` for all d>=3.
Three alive cannot occur on Day1; it is reachable on Day2. Delaying deaths,
using poison differently or paying for PK cannot improve the lower bound.

Consequently the exact first-PRE minima are `4`, `27`, and `15d-1` for d>=3.
For L=1024 the tight reachable runtime maximum day is 68:
its three normal PREs have 1019, 1021 and 1023 tokens; the earliest Day69 PRE
has 1034. This implementation theorem is **never** stored as temporal capacity
or used to reduce the formal 0..256 table.
