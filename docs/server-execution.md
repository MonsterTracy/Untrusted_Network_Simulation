# Server execution and storage

## Local Qwen service

The scientific client and external base-model server use separate environments:

- Client: `/data/yuxiao/envs/untrusted-network-simulation`.
- vLLM: `/data/yuxiao/envs/untrusted-network-simulation-vllm`.
- External model: `/data/yuxiao/models/qwen3.5-9b`.

The external model directory must contain the explicitly selected complete
Qwen3.5-9B checkpoint and tokenizer files. It is outside the scientific artifact
root. Do not use symlinks or substitute a model identifier on path failure.
No model download, service startup, or scientific run is automatic.

The server's existing environment was installed with Python user-site packages
visible, so pip accepted dependencies outside the environment. Disabling the
user site afterwards hides those dependencies; it does not install the missing
ones. Remove that incomplete environment and rebuild it in full. Do not repair
it by installing individual missing packages. The dependency tree comes solely
from `vllm==0.27.0`, not a manually maintained dependency list.

Stop the vLLM process and leave its environment first (`conda deactivate` if it
is active). Only for the known damaged environment, remove this exact prefix:

```sh
rm -rf -- /data/yuxiao/envs/untrusted-network-simulation-vllm
```

Do not change or clean `/home/dell/.local`, `/home/dell/ENTER`, or other server
directories. Do not reuse the old inference environment. Direct prefix removal
avoids `conda env remove` updating a user-level environment registry outside
the allowed directories; leave any old registry entry untouched.

Create the standalone environment from the source checkout. These exports
apply before Conda starts its pip subprocess, including on a fresh install:

The environment YAML uses `conda-forge` plus `nodefaults` to exclude configured
default channels when creating this environment. Its Conda dependencies are
Python and pip; vLLM and its dependency tree are installed through pip. No
Anaconda defaults channel or ToS acceptance is required by this specification.
This file-local channel policy does not change any global Conda configuration
and does not apply to separate `conda env remove` commands. The exact-prefix
removal above invokes no Conda channels.
See the [official environment YAML specification](https://conda.org/learn/specifications/exchange/environment-yml/).

```sh
cd /home/dell/yuxiao/Untrusted_Network_Simulation
export CONDA_PKGS_DIRS=/data/yuxiao/cache/conda/pkgs
export CONDA_ENVS_PATH=/data/yuxiao/envs
export CONDA_REGISTER_ENVS=false
export CONDA_NUMBER_CHANNEL_NOTICES=0
export XDG_CACHE_HOME=/data/yuxiao/cache
export XDG_DATA_HOME=/data/yuxiao/share
export XDG_STATE_HOME=/data/yuxiao/state
export PIP_CACHE_DIR=/data/yuxiao/cache/pip
export TMPDIR=/data/yuxiao/tmp
mkdir -p "$CONDA_PKGS_DIRS" "$CONDA_ENVS_PATH" "$XDG_CACHE_HOME" "$XDG_DATA_HOME" "$XDG_STATE_HOME" "$PIP_CACHE_DIR" "$TMPDIR"
PYTHONNOUSERSITE=1 conda env create --prefix /data/yuxiao/envs/untrusted-network-simulation-vllm --file configs/environments/vllm.yaml
conda activate /data/yuxiao/envs/untrusted-network-simulation-vllm
python -c 'import os, site, sys; assert os.environ.get("PYTHONNOUSERSITE") == "1"; assert site.ENABLE_USER_SITE is False; assert site.getusersitepackages() not in sys.path; print(sys.executable)'
python -m pip check
nvidia-smi
python -c 'import torch, vllm; print(vllm.__version__, torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
```

The YAML `variables` entry preserves `PYTHONNOUSERSITE=1` on activation. It is
not assumed to isolate the pip installation phase; the creation command must
also set it explicitly. Both the isolated `pip check` and actual imports must
pass before this environment is considered usable. A failed check means stop,
not disable isolation or fill dependencies one at a time.

The cache/temp exports keep installation downloads and working data under
`/data/yuxiao`; disabling registration avoids writing `~/.conda/environments.txt`,
and disabling channel notices avoids their user-level cache. Conda may briefly
create its generated requirements file beside the YAML in the source checkout,
which is also inside the allowed tree. No global Conda configuration is edited.
Keep these explicit cache settings in shells used for subsequent installation
or service operation. They do not move or clean any existing user directories.

The environment pins **vLLM 0.27.0** and Python 3.12. Its own pip dependencies
provide the matching Torch stack; do not install the main environment's Torch
into it. The Linux wheel requires glibc >= 2.28 and a driver compatible with
its CUDA build. Actual driver support and RTX 3090 memory headroom must be
verified on the server before using the service.

Version evidence reviewed for this configuration:

- [Official Qwen3.5-9B recipe](https://recipes.vllm.ai/Qwen/Qwen3.5-9B)
  requires vLLM >= 0.17.0.
- [v0.27.0 model registry](https://github.com/vllm-project/vllm/blob/v0.27.0/vllm/model_executor/models/registry.py)
  includes Qwen3.5 dense conditional generation.
- [v0.27.0 release metadata](https://pypi.org/pypi/vllm/0.27.0/json)
  pins torch 2.13.0 and torchaudio 2.11.0.
  [TorchAudio's compatibility contract](https://docs.pytorch.org/audio/stable/installation.html)
  explicitly supports Torch 2.11 and later with TorchAudio 2.11.
- [Native YAML configuration](https://docs.vllm.ai/en/v0.27.0/configuration/serve_args/)
  uses the long CLI argument names. No project launcher is needed.

Ordinary service operation is:

```sh
conda activate /data/yuxiao/envs/untrusted-network-simulation-vllm
vllm serve --config configs/deployment/qwen35-9b.yaml
```

Run from the source checkout. The declared deployment uses one GPU, BF16,
text-only loading, eager execution, one concurrent sequence, 90% GPU memory
budget, an 8192-token context and disabled thinking. These are explicit initial
deployment settings, not a demonstrated 3090 capacity result. The context is
the external model's tokenizer capacity, unrelated to ToM `max_seq_len`.
Oversized requests or OOM must fail; do not truncate, resize, or switch backend
automatically. The runtime declares gameplay temperature 0.7 and output budget
2048; parser `model_params` is empty because runtime assembly does not forward
those parameters. Existing perception/belief request contracts remain in force.

The service listens only on loopback without authentication. The served model
name matches every client reference. In another terminal:

```sh
cd /home/dell/yuxiao/Untrusted_Network_Simulation
conda activate /data/yuxiao/envs/untrusted-network-simulation
export UNS_STORAGE_PROFILE="$PWD/configs/server.json"
curl --fail http://127.0.0.1:8000/v1/models
uns --help
```

Select `configs/runtime/local-qwen35-9b.yaml` explicitly with the existing
`--runtime-config` argument when an authorized collection/publication is run.
`uns` remains the only scientific CLI. It does not start or discover vLLM.

The Collection Plan must bind the normalized runtime config digest and exact
call limit as before. Separately record the serve-config digest, installed
vLLM/runtime versions, and external model revision/file identity in the existing
backend/environment provenance before collection. The client cannot attest
remote weights or settings from an endpoint/model alias. Changing them requires
a newly declared plan, not resuming a bound collection. Never override serve
flags outside the recorded configuration. Stop vLLM explicitly before ToM GPU
capacity checks or training on the same 3090; they need their own memory budget.

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
