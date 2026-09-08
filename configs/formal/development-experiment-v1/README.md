# Development Experiment Protocol v1

Status: **PENDING RESEARCHER VALUES — NOT FROZEN — NOT RUNNABLE**.

This directory is the designated source for the first formal development OOF
protocol. It is not an example or a test configuration. No formal experiment
has been prepared from it. The null entries in `protocol.json` are unresolved
researcher decisions, not defaults. They must not be populated from fixtures,
examples, historical runs, or the old reference repository.

## Contract and source identity

`protocol.json` uses exactly the existing `ExperimentConfig` fields. This
directory name identifies Development Experiment Protocol v1; it does not add
a second JSON schema. The resulting experiment uses `classic7_experiment_v2`.

The recorded source revision is
`c2449428362af8a4e25bee6d61c82cb336350a61`, the independently committed P1 fix.
It identifies the production implementation, not this configuration document's
own commit. Experiment preparation additionally records actual production
source hashes, dependency versions and runtime settings. If the implementation
changes before freezing, record its actual reviewed source revision instead;
never use a placeholder revision or suppress runtime validation.

The source fixes five whole-game folds, both `implicit` and
`explicit_day_phase`, the current model graph, AdamW with constant LR, Primary
alive-nonwolf supervision, current targets, game-balanced loss, seven cyclic
seat shifts, complete rotation cycles, terminal checkpoints, all-ten seal,
and fail-closed validation. These are not additional user-selectable JSON
fields. The three fixed config values are already present in the JSON.

## Required researcher decisions

All sixteen null fields must be explicitly supplied:

- `learning_rate`, `weight_decay`, `adam_betas`, `adam_eps`
- `rotation_cycles`, `game_batch_size`
- `initialization_seed`, `rng_seed`, `schedule_seed`
- `max_seq_len`
- `bootstrap_seed`, `bootstrap_replicates`, `confidence_level`
- `device`, `torch_num_threads`, `recovery_cadence`

Capacity and numerical settings must be declared independently of held-out
content. Publication statistics validate the capacity declaration; they do not
select or enlarge it. Overflow rejects preparation. Budget is the existing
`7 * rotation_cycles * ceil(training_game_count / game_batch_size)` per fold.

## Validation and freeze

The existing `prepare-experiment --protocol` entry point constructs
`ExperimentConfig` before opening the publication. Missing constructor fields,
unknown fields, null values, invalid values, or unsupported fixed settings
raise an error; no defaults are inserted and no experiment is published.
The configuration alone can be checked without reading publication data:

```sh
python -B -c 'import json; from pathlib import Path; from werewolf.tom.experiment import ExperimentConfig; ExperimentConfig(**json.loads(Path("configs/formal/development-experiment-v1/protocol.json").read_text()))'
```

This command must fail while any required value is unresolved. A successful
configuration check is necessary but does not replace publication, capacity,
runtime, or experiment preflight validation.

After researcher values are supplied and validated, record the decision and
configuration commit, change this status to frozen, and preserve that exact
configuration as the v1 source. Do not edit a frozen protocol in place to adapt
to outcomes or preparation failures. No experiment preparation or training is
authorized merely by completing this pending document.

Use the existing `classic7-tom prepare-experiment` operation with this JSON as
`--protocol`, a verified development publication as `--publication`, and a new
artifact destination as `--destination`. All three arguments are explicit.
The artifact records the full config under `protocol_inputs.config`, together
with publication/fold provenance, protocol digest, temporal tables, schedules,
initial states, population masks and runtime provenance. Retain the frozen
config commit alongside the resulting experiment manifest identity.

Publication, sidecar and complete protocol digests bind provenance only.
Fold membership uses stable game identities and its frozen rule; numerical
training controls use the explicit seeds, fold and training identities.
This document does not introduce final fit or inference.
