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
it by installing individual missing packages. Rebuild from the environment
specification, which retains `vllm==0.27.0` and pins its CUDA 13.0 JIT toolchain
as a coherent set. Do not downgrade individual packages in the existing prefix.

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

The clean-create invocation also sets `CONDA_NO_PLUGINS=true` for that command
only. Conda disables external plugins, including `conda-anaconda-tos`, while
retaining built-in plugins. This is separate from the YAML's `nodefaults`
channel policy: do not accept the defaults ToS or edit global Conda settings.
Other external plugins, including external solvers, are also disabled. The
creation command explicitly selects the built-in `classic` solver; a creation
failure must be reported rather than silently changing the solver.
See [Conda plugin configuration](https://docs.conda.io/projects/conda/en/stable/configuration.html)
and the [Anaconda ToS plugin](https://github.com/anaconda/conda-anaconda-tos).

Use `env` to pass the build environment directly to the executable identified
by the initialized shell's `CONDA_EXE`, bypassing the `conda` shell function.
`CONDA_EXE` must identify the existing Conda executable; do not replace or
modify that installation. `PYTHONNOUSERSITE=1` must reach both the Conda process
and its pip subprocess, not just the later activated environment.

```sh
cd /home/dell/yuxiao/Untrusted_Network_Simulation
export CONDA_PKGS_DIRS=/data/yuxiao/cache/conda-pkgs
export CONDA_ENVS_PATH=/data/yuxiao/envs
export CONDA_REGISTER_ENVS=false
export CONDA_NUMBER_CHANNEL_NOTICES=0
export XDG_CACHE_HOME=/data/yuxiao/cache/xdg
export XDG_DATA_HOME=/data/yuxiao/share
export XDG_STATE_HOME=/data/yuxiao/state
export PIP_CACHE_DIR=/data/yuxiao/cache/pip
export TMPDIR=/data/yuxiao/tmp/vllm-build
mkdir -p /data/yuxiao/conda-home "$CONDA_PKGS_DIRS" "$CONDA_ENVS_PATH" "$XDG_CACHE_HOME" "$XDG_DATA_HOME" "$XDG_STATE_HOME" "$PIP_CACHE_DIR" "$TMPDIR"
env \
  HOME=/data/yuxiao/conda-home \
  XDG_CACHE_HOME=/data/yuxiao/cache/xdg \
  CONDA_PKGS_DIRS=/data/yuxiao/cache/conda-pkgs \
  PIP_CACHE_DIR=/data/yuxiao/cache/pip \
  TMPDIR=/data/yuxiao/tmp/vllm-build \
  PYTHONNOUSERSITE=1 \
  CONDA_NO_PLUGINS=true \
  CONDA_REGISTER_ENVS=false \
  CONDA_NUMBER_CHANNEL_NOTICES=0 \
  "$CONDA_EXE" env create \
  --solver classic \
  --prefix /data/yuxiao/envs/untrusted-network-simulation-vllm \
  --file configs/environments/vllm.yaml
conda activate /data/yuxiao/envs/untrusted-network-simulation-vllm
python -c 'import os, site, sys; assert os.environ.get("PYTHONNOUSERSITE") == "1"; assert site.ENABLE_USER_SITE is False; assert site.getusersitepackages() not in sys.path; print(sys.executable)'
python -m pip check
python -m pip show \
  flashinfer-python \
  flashinfer-jit-cache \
  flashinfer-cubin
python -m pip show \
  cuda-toolkit \
  nvidia-cuda-nvcc \
  nvidia-cuda-crt \
  nvidia-nvvm \
  nvidia-cuda-runtime \
  nvidia-cuda-nvrtc
echo "$CUDA_HOME"
test -x "$CUDA_HOME/bin/nvcc"
"$CUDA_HOME/bin/nvcc" --version
nvidia-smi
python -c 'import torch, vllm; print(vllm.__version__, torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
```

The CUDA discovery check must point to
`/data/yuxiao/envs/untrusted-network-simulation-vllm/lib/python3.12/site-packages/nvidia/cu13`.
Updating the repository YAML alone does not update an already-created
environment's activation variables. Stop if the activated value is missing or
different. These checks verify discovery, not successful JIT compilation.

The FlashInfer package check must show `flashinfer-python` 0.6.16.post3 and
`flashinfer-jit-cache` 0.6.16.post3+cu130. `flashinfer-cubin` must remain absent;
the corresponding package-not-found warning from `pip show` is expected.
The environment uses the exact x86_64, cp39-abi3, manylinux_2_28 wheel URL and
SHA-256 listed in the [official CUDA 13.0 JIT-cache index](https://flashinfer.ai/whl/cu130/flashinfer-jit-cache/).
This package-specific direct reference leaves the existing PyPI index policy
unchanged. It targets this Linux x86_64 server; Python 3.12 and glibc 2.39
satisfy the wheel tags. Do not install `flashinfer-cubin` or upgrade the core
FlashInfer package to satisfy this check.

The package versions must be `cuda-toolkit` 13.0.3.0 (equivalently 13.0.3),
`nvidia-cuda-nvcc`, `nvidia-cuda-crt`, and `nvidia-nvvm` 13.0.88,
`nvidia-cuda-runtime` 13.0.96, and `nvidia-cuda-nvrtc` 13.0.88.
`nvcc --version` must report CUDA 13.0, V13.0.88. These match the
[CUDA 13.0.3 component table](https://pypi.org/project/cuda-toolkit/13.0.3/)
and Torch 2.13.0's CUDA toolkit requirement. The explicit compiler component
pins prevent unconstrained dependencies from selecting CUDA 13.3 components.
Stop on a mismatch or failed `pip check`; do not repair the prefix piecemeal.

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
  uses the long CLI argument names; the operator start script only activates
  the environment and execs this native configuration.

Before starting the service, verify the activated environment and workspace:

```sh
conda activate /data/yuxiao/envs/untrusted-network-simulation-vllm
echo "$FLASHINFER_WORKSPACE_BASE"
test "$FLASHINFER_WORKSPACE_BASE" = /data/yuxiao/cache/flashinfer || exit 1
mkdir -p /data/yuxiao/cache/flashinfer
./scripts/start_vllm.sh
```

FlashInfer 0.6.16.post3 appends `.cache/flashinfer` to
`FLASHINFER_WORKSPACE_BASE`, so its JIT workspace is under
`/data/yuxiao/cache/flashinfer/.cache/flashinfer/<version>/<architecture>/`.
The variable must be set before starting the service. Leave historical caches
under `/home/dell/.cache` untouched; this service must not use them.
See the [versioned FlashInfer workspace implementation](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.16.post3/flashinfer/jit/env.py).

The JIT-cache acceptance criterion is actual vLLM startup on this SM86 GPU:
the sampling module must load an available prebuilt artifact from the installed
cache, without attempting a local Ninja link of `sampling.so`. Package presence
or `python -m flashinfer show-config` alone is insufficient. A missing or
unloadable prebuilt artifact can still lead upstream FlashInfer to attempt JIT;
that does not satisfy this deployment check.
The show-config message `No supported CUDA architectures found for major
versions [9, 10, 11, 12]` concerns modules restricted to SM90+, not FlashInfer
as a whole. It does not invalidate detection of `{(8, '6')}`. Keep the real
SM86 architecture; do not add a fake architecture to suppress this message.

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

## Operator collection workflow

The recommended collection entry is `scripts/collect_games.py`. It freezes a
Collection Plan and delegates to the existing `uns collect` implementation;
it does not own a second collector or change ledger/recovery semantics.
Run from `/home/dell/yuxiao/Untrusted_Network_Simulation` with the main
environment installed from this checkout. Commit and review all source changes
before collection: both new and resumed campaigns require a clean actual Git
HEAD. No revision or provenance digest is copied from a calibration run.

Prepare the operator file once:

```sh
mkdir -p /data/yuxiao/Untrusted_Network_Simulation/operator
cp configs/operator/collection_campaign.example.json /data/yuxiao/Untrusted_Network_Simulation/operator/collection_campaign.json
```

Edit only `collection_id`, `target_games`, `seed_pool_size`, and `call_limit`
in that external JSON. The example specifies 300 successes, 450 planned seeds,
and a call limit of 1000. The call limit is an operational fail-closed backend
dispatch cap, not a theoretical Classic7 call upper bound. Changing 300 to 500
requires a new collection_id (and a sufficient seed pool); never alter an
already frozen campaign in place. Unknown fields and invalid values fail.
`--campaign PATH` selects another explicit operator file; no discovery occurs.

The first `development-qwen35-9b-300-v1` campaign froze incompatible V1 model
and prompt identities at source `f8109750d3a4ea47a9d03c49e44f19e0c9134a88`
and already consumed a durable claim. Preserve its Plan and failed/interrupted
evidence. Do not resume it with corrected code, edit its Plan, or delete it to
reuse the identity. After the identity fix is committed, use
`development-qwen35-9b-300-v2` on the new clean HEAD to freeze a new Plan.

Initialize Conda in the operator shell so `CONDA_EXE` identifies the existing
Conda executable. Start the foreground service in one terminal:

```sh
./scripts/start_vllm.sh
```

The script activates the vLLM environment and execs the authoritative deployment
YAML without copying model options. No restart, background daemon, or endpoint
substitution is performed. In another terminal, activate the main environment:

```sh
conda activate /data/yuxiao/envs/untrusted-network-simulation
python scripts/collect_games.py
```

After interruption, explicitly resume:

```sh
python scripts/collect_games.py --resume
```

Paths are resolved under the artifact root in `configs/server.json`:

- Operator parameters: `operator/collection_campaign.json` (default).
- Frozen Plan: `plans/<collection_id>.json`.
- Canonical destination: `canonical/<collection_id>`.

New execution requires both Plan and destination to be absent. An existing
Plan always rejects ordinary execution, even if its destination is absent.
Resume requires both the original Plan and existing run records. A Plan frozen
before a crash but lacking a destination requires explicit operator handling;
the script does not repair, overwrite, delete, or restart it. Resume compares
campaign fields, source HEAD, schema identities, and all recomputed provenance
with the original Plan. It never regenerates the ordered seed pool. Existing
collector policy closes an interrupted claim and advances through the frozen
pool; it does not retry that game seed.

Seeds use zero-based ordinal order. Each is the big-endian integer from the
first eight bytes of SHA-256 of
`classic7-canonical-collection-v1:<collection_id>:<ordinal>`, masked to 63 bits.
The rule is content-independent, and duplicate seeds or any seed in
900000001..900000015 fail closed, without skipping or re-rolling. Those 15
engineering calibration seeds are permanently excluded from formal campaigns.

Preflight recomputes normalized runtime SHA-256, deployment file SHA-256, and
a complete model manifest digest. The model directory specified by deployment
YAML must contain `HF_REVISION`, a plain UTF-8 file with the exact 40-character
HF commit. There is no inferred revision or fallback. The manifest contains
every regular file (including HF_REVISION), recursively sorted by relative
POSIX path, as records with `path`, `size_bytes`, and `sha256`; its digest uses
the project's canonical JSON. Symlinks are rejected. Keep the external model
directory immutable during preflight and collection. A previously computed
digest using another manifest format is not substituted for this calculation.

The Plan also records serving model name, model path, client Python/OpenAI
versions, serving Python/vLLM/Torch/CUDA/FlashInfer versions, seed rule identity,
pool size, target, and exact call limit. Serving versions are queried from the
vLLM environment's Python. Missing software or revision evidence fails.
`model_identity` is the served model name used by the runtime parser;
`prompt_identity` is exactly the V1 speech perception prompt version.
HF revision, model manifest digest/rule, and the separate
`gameplay_prompt_profile` remain bound in `environment_provenance`.
Health and model-list requests use the declared loopback endpoint with
environment proxies disabled. These requests prove service availability and
served-name agreement, not remote weight bytes or process launch arguments:
the operator must run this checkout's start script against the audited model
and environment, and leave them unchanged during the campaign.

The Plan is durably published without replacement using the existing ledger
publication primitive, then reloaded through canonical Plan validation before
collection. Console output prints the frozen identity, counts, digests, paths,
revision and mode, without hidden/private game data. All actual dispatch limits
passed to `uns collect` come from the validated frozen Plan.

vLLM and ToM training must not run concurrently on the single RTX 3090.
Stop the service explicitly before training. This workflow performs no training
and does not fill the pending Formal Development Protocol.

## Advanced/reference commands

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

## Paired final lifecycle

All-development final fit, terminal model sealing and independent final
evaluation use the same storage profile and `uns` ownership. See the
[final lifecycle commands and contracts](final-lifecycle.md). Final preparation
requires a clean Git checkout and records its actual HEAD. No new numerical CLI
overrides are accepted, and final data cannot supply temporal capacity.
