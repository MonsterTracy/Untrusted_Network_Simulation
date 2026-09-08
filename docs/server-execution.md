# Server execution and storage

## Synthetic capacity check

`uns capacity-check` is a separate engineering operation owned by the same CLI.
It does not load a storage profile, protocol, publication, belief, role map or
temporal artifact, and does not write files. All four sizing arguments are
required:

```sh
uns capacity-check --pre-count-per-game PRE_COUNT --max-seq-len LENGTH --game-batch-size GAMES --device cuda
```

Use `--device cpu` for shape/execution validation only. Output is marked
`ENGINEERING CAPACITY CHECK — NOT A SCIENTIFIC RUN`. The real model runs two
fixed forward/loss/backward/AdamW steps with synthetic full-length inputs,
non-self targets, masks and 128+128 temporal lookup shapes. Each game retains
its graph until the shared game-balanced loss is backpropagated. Scratch AdamW
settings are engineering constants, not formal protocol defaults or choices.
The second step includes already allocated Adam moments. No loss or accuracy
is reported; CUDA reports peak allocated/reserved bytes for this invocation
(including any preexisting allocator state), not a guaranteed production peak.

Sizes are neither searched nor changed. Unavailable CUDA, invalid input, OOM
or execution failure propagates directly. The check cannot update Formal
Development Protocol v1 and must not participate in model selection. It models
training allocation shapes, not complete dataset host-memory or recovery I/O.

Code checkout: `/home/dell/yuxiao/Untrusted_Network_Simulation`.
Deployment profile: `configs/server.json`, containing only
`artifact_root: /data/yuxiao/Untrusted_Network_Simulation` (JSON syntax).
This is not an ExperimentConfig and contains no scientific hyperparameters.

Install the sole console entry in the selected server Python environment:

```sh
cd /home/dell/yuxiao/Untrusted_Network_Simulation
python -m pip install -e '.[tom]'
mkdir -p /data/yuxiao/Untrusted_Network_Simulation
export UNS_STORAGE_PROFILE="$PWD/configs/server.json"
uns --help
```

`uns --storage-profile /absolute/profile.json ...` explicitly selects a profile
instead of the environment variable. There is no implicit profile, directory
search, repository fallback, or latest selection. The root must already exist,
be absolute and contain no symlink components. Invalid/missing profiles fail.
The environment must satisfy existing runtime/durability requirements.

## Identity and layout

Single-component artifact names resolve deterministically by operation:

```text
artifact_root/
  canonical/COLLECTION_ID/                 # plan, ledger, bundles, private evidence
  publications/PUBLICATION_ID/            # public records, folds, role sidecar
  experiments/EXPERIMENT_ID/
    experiments/experiment/               # immutable experiment envelope
      experiment_manifest.json
      temporal/                           # immutable day/phase artifacts
      folds/                              # masks, schedules, initial states
    runs/EXPERIMENT_MANIFEST_DIGEST/       # existing mutable run contract
      implicit/0..4/                      # recovery, terminal, predictions, reports
      explicit_day_phase/0..4/
      checkpoint_set_manifest.json
      reports/
```

The nested `experiments/experiment` preserves the existing path contract:
`VerifiedExperiment.runs_path = experiment.path.parent.parent / runs / digest`.
No global checkpoint/report directory is introduced. Experiment directory names
are explicit locators; manifests and digests remain authoritative identities.
No identity is selected by modification time or substituted on failure.

Multi-component relative artifact paths resolve under the root; absolute
artifact paths must also be inside it. `..`, symlink traversal, root-as-artifact,
and out-of-root paths are rejected. `validate-artifact` takes an explicit
root-relative or contained absolute path because it supports multiple types.
Experiment paths must use the displayed experiment-specific layout even when
spelled as an absolute path; an unscoped global `runs` layout is rejected.
OOF execution also checks the recorded publication location and derived run
directory against the selected root. Preparation stores the resolved absolute
publication path in the experiment artifact, never in the scientific protocol.
Relocating an existing experiment's bound publication is not an automatic
migration; plan the server storage locations before artifact preparation.

## Formal commands

`COLLECTION_ID`, `PUBLICATION_ID`, `EXPERIMENT_ID`, plan/runtime paths and
`DECLARED_CALL_LIMIT` below are explicit operator-supplied values, not defaults.

```sh
uns collect --plan PLAN.json --runtime-config RUNTIME.yaml --call-limit DECLARED_CALL_LIMIT --destination COLLECTION_ID
uns publish-development --collection COLLECTION_ID --runtime-config RUNTIME.yaml --destination PUBLICATION_ID
uns prepare-experiment --publication PUBLICATION_ID --protocol configs/formal/development-experiment-v1/protocol.json --destination EXPERIMENT_ID
uns validate-artifact experiments/EXPERIMENT_ID/experiments/experiment
uns run-development-oof --experiment EXPERIMENT_ID
```

The pending formal protocol still contains null researcher values. Preparation
must fail until it is explicitly frozen; these commands do not authorize filling
those values or starting an experiment. Collection uses its existing immutable
plan and runtime binding. No game-count, model, population, LR or budget CLI
overrides are added. Counts follow the plan/publication; fold partitions and
optimizer steps derive from existing rules.

`run-development-oof` already performs both conditions across five folds,
verifies and seals all ten terminal checkpoints, then predicts and reports.
It stops on exceptions. No separate train/report wrapper or duplicate runner
is needed. `validate-artifact` performs existing preflight and, when sealed,
validates existing evaluation artifacts without generating missing reports.

## Explicit continuation

Nonempty existing collection/run directories require `--resume`. Conversely,
`--resume` rejects a location with no existing records. Continuation uses the
exact selected artifact, immutable protocol, deterministic recovery chain and
existing fail-closed validators. It does not clear failures, select an earlier
checkpoint, grant a replacement run, or resume training after the seal.

```sh
uns collect --plan PLAN.json --runtime-config RUNTIME.yaml --call-limit DECLARED_CALL_LIMIT --destination COLLECTION_ID --resume
uns run-development-oof --experiment EXPERIMENT_ID --resume
```

The latter also explicitly continues unfinished prediction/report publication
after a verified seal. No training or formal artifact creation is part of
installing this profile. Server filesystem permissions, capacity and runtime
availability must be checked on the server itself.
