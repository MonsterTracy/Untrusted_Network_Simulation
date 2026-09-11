# Development Experiment Protocol v1

Status: **FROZEN — DEVELOPMENT EXPERIMENT PROTOCOL V1**.

This directory is the designated source for the first formal development OOF
protocol. It is not an example or a test configuration. All sixteen
researcher-selected values in `protocol.json` are now frozen. This freeze does
not use held-out outcomes, metrics, or historical training results for selection.

## Contract and source identity

`protocol.json` uses exactly the existing `ExperimentConfig` fields. This
directory name identifies Development Experiment Protocol v1; it does not add
a second JSON schema. The resulting experiment uses `classic7_experiment_v2`.

The recorded source revision is
`927c4cd7576be04792d5729999bb254b22c27626`.
It identifies the production scientific implementation, not this protocol/README
configuration commit. Experiment preparation additionally records actual
production source hashes, dependency versions and runtime settings; runtime
validation remains mandatory.

The source fixes five whole-game folds, both `implicit` and
`explicit_day_phase`, the current model graph, AdamW with constant LR, Primary
alive-nonwolf supervision, current targets, game-balanced loss, seven cyclic
seat shifts, complete rotation cycles, terminal checkpoints, all-ten seal,
and fail-closed validation. These are not additional user-selectable JSON
fields. The three fixed config values remain unchanged in the JSON.

## Frozen researcher decisions

The sixteen fields are explicitly supplied in `protocol.json`:

- `learning_rate`, `weight_decay`, `adam_betas`, `adam_eps`
- `rotation_cycles`, `game_batch_size`
- `initialization_seed`, `rng_seed`, `schedule_seed`
- `max_seq_len`
- `bootstrap_seed`, `bootstrap_replicates`, `confidence_level`
- `device`, `torch_num_threads`, `recovery_cadence`

The researcher declared `max_seq_len=1024` after RTX 3090 engineering capacity
calibration and before opening the publication. The completed publication's
observed maximum is 192 structured tokens: `192 <= 1024` validates the declared
capacity. This is capacity validation, not post-hoc selection; the observed
maximum does not select or enlarge capacity. Overflow remains fail closed.

Capacity-validation context supplied by the researcher:

- Publication digest: `c528294c292790af07d8200cacef3b143c329b73b231c53fa6031e37b12b0b3d`
- Game count: 300; maximum observed day: 4; maximum structured token count: 192.

This context is documentation only. The experiment artifact binds the publication
digest; no publication field is added to the protocol JSON.

The four seeds are deterministic researcher decisions frozen before formal
training, independently of publication contents. For each field name
`initialization_seed`, `rng_seed`, `schedule_seed`, and `bootstrap_seed`, use
UTF-8 bytes of `classic7-development-experiment-v1:` followed by the field name,
compute SHA-256, read the first eight digest bytes as a big-endian integer, and
apply `& ((1 << 63) - 1)`. The resulting values are respectively:

- `463146641602440340`
- `8223107198166168390`
- `4747664282435381788`
- `2196221715671680944`

Game IDs, labels, fold contents, outcomes, held-out metrics, and historical
training results do not select or change these seeds.

Execution declares `device="cuda"` and `torch_num_threads=1`.
`recovery_cadence=1000` is an operational recovery policy, not learning-budget
selection. Budget remains the existing
`7 * rotation_cycles * ceil(training_game_count / game_batch_size)` per fold.
For this 300-game, five-fold publication, each fold has 240 training games;
three rotation cycles and batch size one give `7 * 3 * 240 = 5040` optimizer
steps. Existing recovery logic therefore yields steps 1000, 2000, 3000, 4000,
5000, and terminal step 5040. This document adds no recovery logic.

## Validation and use

The existing `prepare-experiment --protocol` entry point constructs
`ExperimentConfig` before opening the publication. Missing constructor fields,
unknown fields, null values, invalid values, or unsupported fixed settings
raise an error; no defaults are inserted and no experiment is published.
The configuration alone can be checked without reading publication data:

```sh
python -B -c 'import json; from pathlib import Path; from werewolf.tom.experiment import ExperimentConfig; ExperimentConfig(**json.loads(Path("configs/formal/development-experiment-v1/protocol.json").read_text()))'
```

A successful configuration check is necessary but does not replace publication,
capacity, runtime, or experiment preflight validation. Preserve this exact
configuration as the v1 source. Do not edit a frozen protocol in place to adapt
to outcomes or preparation failures. This freeze does not itself execute
experiment preparation or training.

Use the existing `uns prepare-experiment` operation with this JSON as
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

CLI execution additionally requires an explicit deployment storage profile.
That profile is separate from this scientific JSON; see
[server execution](../../../docs/server-execution.md).
