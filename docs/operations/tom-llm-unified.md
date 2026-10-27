# ToM LLM operations

`uns tom-llm` is the operator interface. It reuses the existing ModelSpec,
Selection, Cohort, budget, probe, Runner, Prediction Seal and analysis modules.
The older scripts remain valid for direct/diagnostic use. No Selection publishing,
model download, environment installation, tmux, automatic GPU startup or unknown
request replay is provided.

## Configuration and stages

Copy `configs/paper/tom-llm-operations.example.yaml` outside the checkout as
`tom-llm.yaml`. Fill actual paths and externally verified manifest digests once.
Placeholders are deliberately not usable identities. JSON experiment configs
continue to work with the original scripts. The frozen model registry supplies
official revisions, templates, decoding and termination parameters; Deployment
supplies ports 18581/18582/18583, device 0 and the dedicated GPU UUID. These are
shown/verified rather than duplicated as freely editable YAML sampling values.
Changing those frozen values is a new protocol, not an operations setting.

All commands below are **future Ubuntu manual operations after review and separate
authorization**. They have not been executed by this refactor. Use a committed
checkout and controller dependencies (including the optional ToM/LLM
dependencies). Models may share the explicitly configured read-only
Python3.12/vLLM prefix `/data/yuxiao/envs/untrusted-network-simulation-tom-baseline`.
The controller calls a prefix's own CPU interpreter for environment inspection.
The original synthetic client also runs in that prefix. No environment is edited.

```bash
uns tom-llm --config tom-llm.yaml status
uns tom-llm --config tom-llm.yaml prepare --stage cohort
uns tom-llm --config tom-llm.yaml prepare --stage model --models yi
```

`status` reads manifests/availability, historical Admission and GPU occupancy; it
does not hash every weight or start anything. File presence is not physical
Admission. `prepare cohort` reads the configured pinned Cohort or publishes one
to `output_root/cohort` using the existing formally published Selection. It
cannot publish/reselect games. `prepare model` requires complete local official
weights, tokenizer assets and environment. It verifies/publishes model and
environment receipts, native suite and full-150 budget, then a NEW Deployment.
Existing pinned `assets`, `environment`, `suite`, `budget` references can be reused.
It prints path/digest pairs; optionally persist them with `--save-config`. A missing digest on
an existing input is an error, never an inferred external approval. A preexisting
Deployment stops preparation before hashing or writing. Interrupted preparation
is preserved; it is not overwritten or automatically cleaned/resumed.

### Shared Conda environment: manual installation and CPU acceptance

The supplied YAML uses one `environment_prefix` via YAML anchors for all three
models. A prefix must be inside the historical baseline root, or be a direct,
non-symlink child of `/data/yuxiao/envs/`. The parent directory itself, nested
external prefixes and `..` paths are rejected. This exception is for a read-only
environment asset only: models, caches, temporary files, logs, candidates and
results still use separate paths inside `/data/yuxiao/tom-llm-reasoner-b2`.
No Conda activation/discovery, registration or automatic installation is added.
The selected interpreter is always the prefix's own `bin/python`.

The new prefix **has no GPU qualification**, even if the wheels match the old
environment. Keep Candidate-v4 and its Admission/config as historical evidence.
Use a NEW per-model candidate root (the example uses `candidate-shared-env-v1`)
and new environment receipts. Do not pin old `environment`, `deployment` or
`admission` references into the new config. Existing official weights, tokenizer,
Suite, common Cohort and Budget may be reused after their existing strict checks;
do not move them, republish Selection, or create a different request population.
Each model needs its own two-round physical Qualification and Admission on this
prefix before a new Permit. Changing the YAML prefix alone cannot retarget an old
Deployment or Permit: prepare-permit, start, qualify and run reject the mismatch.
Historical reads, logs and an explicit owned-service stop remain available.

`configs/paper/tom-llm-environment-linux-py312-v1.txt` pins 213 official binary
wheel URLs and SHA-256 values for CPython 3.12/Linux x86_64/glibc >= 2.35. Core
versions are vLLM 0.27.0, Torch 2.13.0, Transformers 5.14.1, Tokenizers 0.22.2,
Triton 3.7.1, FlashInfer 0.6.16.post3 and TVM-FFI 0.1.11. CUDA toolkit 13.0.3.0,
NVCC/CRT/NVRTC 13.0.88, runtime 13.0.96 and the official cu130 FlashInfer JIT
cache are included. These match the previously verified release combination;
they do not copy its environment identity or RECORD findings. vLLM's official
[CUDA requirements](https://github.com/vllm-project/vllm/blob/v0.27.0/requirements/cuda.txt)
and [published metadata](https://pypi.org/pypi/vllm/0.27.0/json) specify the
Torch/FlashInfer/TVM dependencies. Do not independently upgrade Torch,
Transformers, CUDA or install a second cu12 family. The driver is NOT installed
by this lock. Driver/3090/JIT compatibility still requires physical qualification.

The lock was resolved on Mac for the Linux/Python3.12 target using uv 0.11.8,
without overrides; wheels were selected by compatible tags and physical hashes.
This is dependency-resolution evidence, **not** an installed Linux `pip check`
or GPU PASS. The 198 historical wheel references are retained, and the remaining
controller/report/test dependencies are also pinned. The precompiled cu130 JIT
wheel retains its separate official release SHA. Regenerating dependencies is a
reviewed lock update, not a step to repeat during each deployment.

The following commands are for the **user's Ubuntu terminal only**, after
reviewing the new source commit. They install packages only into the already
created, unqualified prefix; they neither create a Conda environment nor modify
`/home/dell/.conda/environments.txt`. Replace SOURCE with the independently
deployed checkout, never Candidate-v4's source. Stop on any error. If this prefix
already serves a task or has a receipt/qualification, do not install into it.
If it already contains packages beyond the newly created Conda Python/pip
bootstrap, first compare `pip freeze --all` with the lock and stop on unexplained
differences; do not use this sequence as an environment repair/cleanup procedure.

```bash
set -euo pipefail
ENV_PREFIX=/data/yuxiao/envs/untrusted-network-simulation-tom-baseline
SOURCE=/REPLACE/INDEPENDENT/COMMITTED/CHECKOUT
PREP=/data/yuxiao/tom-llm-reasoner-b2/deployment-prep/shared-env-v1
test -x "$ENV_PREFIX/bin/python"
test ! -e "$PREP"
test -f "$SOURCE/configs/paper/tom-llm-environment-linux-py312-v1.txt"
"$ENV_PREFIX/bin/python" -I -B - "$ENV_PREFIX" <<'PY'
from pathlib import Path
import platform,sys
p=Path(sys.argv[1])
assert p.is_absolute() and p.parent==Path('/data/yuxiao/envs')
assert not any(x.is_symlink() for x in (p,*p.parents))
assert Path(sys.prefix).resolve()==p and sys.version_info[:2]==(3,12)
assert Path(sys.executable).resolve().is_relative_to(p)
assert sys.platform=='linux' and platform.machine()=='x86_64'
libc,version=platform.libc_ver()
assert libc=='glibc' and tuple(map(int,version.split('.'))) >= (2,35)
print(sys.version,libc,version)
PY
mkdir -p "$(dirname "$PREP")"
mkdir "$PREP"
export PIP_CACHE_DIR="$PREP/pip-cache" TMPDIR="$PREP/tmp"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=''
mkdir "$TMPDIR"
cd "$SOURCE"
git rev-parse HEAD > "$PREP/source-revision.txt"
sha256sum configs/paper/tom-llm-environment-linux-py312-v1.txt > "$PREP/lock.sha256"
"$ENV_PREFIX/bin/python" -I -B -m pip freeze --all > "$PREP/pip-before.txt"
"$ENV_PREFIX/bin/python" -I -B -m pip --isolated install --prefix "$ENV_PREFIX" \
  --cache-dir "$PIP_CACHE_DIR" --require-hashes --only-binary=:all: \
  --report "$PREP/install-report.json" -r configs/paper/tom-llm-environment-linux-py312-v1.txt
"$ENV_PREFIX/bin/python" -I -B -m pip --isolated install --prefix "$ENV_PREFIX" \
  --cache-dir "$PIP_CACHE_DIR" --no-deps --no-build-isolation \
  --report "$PREP/project-install-report.json" -e .
"$ENV_PREFIX/bin/python" -I -B -m pip check > "$PREP/pip-check.txt"
cat "$PREP/pip-check.txt"
"$ENV_PREFIX/bin/python" -I -B -m pip freeze --all > "$PREP/pip-freeze.txt"
"$ENV_PREFIX/bin/python" -I -B -m pip inspect > "$PREP/pip-inspect.json"
conda list --explicit -p "$ENV_PREFIX" > "$PREP/conda-explicit.txt"
```

Expected `pip check`: `No broken requirements found.` A failure stops acceptance;
do not use `--no-deps` for the locked dependencies, override metadata, edit RECORD
or reinstall an old qualified environment. The application's separate `--no-deps`
install only exposes `uns` after the complete dependency lock has been installed;
the following `pip check` checks the application's requirements too. The editable
install is necessary: runtime Source verification must resolve the committed Git
checkout, not an untracked copy of the package under site-packages. Installation
is rerunnable with identical inputs in an unqualified prefix, but each attempt
uses a NEW PREP directory and preserves earlier reports. Inspect the source Git
status after packaging; do not erase an unexpected change. No command imports
vLLM/Torch or executes a forward pass here.

Before `prepare --stage model`, perform the existing environment audit on this
new prefix and pin its actual report SHA as `prior_audit_digest`. Historical nine
RECORD deviations remain in their original reports; do not present that old SHA
as an audit of this prefix. Any new deviations must be disclosed for review.
`capture_environment` still requires pip check, pinned vLLM seam bytes, package
METADATA/RECORD and critical-file hashes; a shared prefix does not waive them.
After CPU acceptance, keep the prefix unchanged across all three models. An
installation change invalidates the corresponding physical binding. Sharing an
environment is not sharing model Qualification.

With a separately reviewed YAML and its actual new audit digest:

```bash
"$ENV_PREFIX/bin/uns" tom-llm --config /data/yuxiao/tom-llm-reasoner-b2/tom-llm.yaml status
"$ENV_PREFIX/bin/uns" tom-llm --config /data/yuxiao/tom-llm-reasoner-b2/tom-llm.yaml prepare --stage model --models qwen3 --save-config
```

Repeat model preparation separately for Yi/Mistral when their full local assets
exist. It publishes only new per-model CPU receipts and Deployment; no service
is started. The 76-call two-instance Qualification commands below require a
separate GPU authorization for each new model/environment binding.

### CPU inputs before weight acquisition

Tokenizer, Suite and Budget no longer require complete weights, model prefixes,
GPU UUID or a Deployment. Tokenizer/Suite use official small `tokenizer_assets`;
Suite output uses a new `cpu_root` (default `output_root/cpu/MODEL`). Budget also
requires the pinned common Cohort:

```bash
uns tom-llm --config tom-llm.yaml prepare --stage tokenizer --models yi mistral
uns tom-llm --config tom-llm.yaml prepare --stage suite --models yi mistral --save-config
uns tom-llm --config tom-llm.yaml prepare --stage budget --models yi mistral --save-config
```

Tokenizer checks pinned local asset bytes and the existing native text/token-ID
parity on the synthetic suite. Suite publishes the unchanged 12 cases/38-call
schedule. Budget calls the existing full-150 publisher and reader: all 10,714
public tasks in the real Cohort, no Primary filtering, truncation or per-model
intersection. It prints request count, row identity, total/p50/p95/p99/max input
tokens and the fixed 8192/512 context/output limit. `--games 10` is rejected for
budget; N only controls technical runs. These CPU results grant no GPU Admission.
Mac tests use a synthetic full-150 population, not the actual server 10,714 rows.

Existing pinned Suite/Budget references are strictly read back and reused. For
fresh computation choose a NEW `cpu_root` and remove only the references in the
mutable operations config; do not delete/overwrite the old Artifacts. New CPU
outputs cannot be written inside a sealed Artifact or a configured old candidate.
Later `prepare --stage model` reuses the same pinned CPU references.

`--save-config` is optional and explicit for `prepare cohort/suite/budget/model`
and `admit`. It stores returned path/digest pairs, not PID or approval evidence.
Externally supplied historical records still need their verified digest once.
The file is atomically re-serialized (YAML comments/layout are not preserved);
on Ubuntu it must be outside the checkout and inside the baseline root. A detected
intervening edit refuses replacement and preserves generated Artifacts. Without
this flag the config is never edited. Run/Permit commands already derive their
references from the owning immutable run; no manual digest/PID repetition is needed.

## Services and qualification

For a model requiring new qualification, put its new Deployment reference in
YAML. Start/stop are explicit actions with `--execute-live`:

```bash
uns tom-llm --config tom-llm.yaml serve start --model yi --execute-live
uns tom-llm --config tom-llm.yaml serve status --model yi
uns tom-llm --config tom-llm.yaml qualify --model yi --round 0 --execute-live
uns tom-llm --config tom-llm.yaml serve stop --model yi --execute-live
uns tom-llm --config tom-llm.yaml serve start --model yi --execute-live
uns tom-llm --config tom-llm.yaml qualify --model yi --round 1 --execute-live
uns tom-llm --config tom-llm.yaml serve stop --model yi --execute-live
uns tom-llm --config tom-llm.yaml admit --models yi --save-config
uns tom-llm --config tom-llm.yaml serve logs --model yi --lines 40
```

Each round is the unchanged 38-call schedule (two independent instances, total
76 synthetic calls). PID and process identity are discovered from the tool's
own launch state; they are not YAML fields. Existing round destinations are
refused before dispatch. Failure/unknown artifacts stop the round with no retry.
The synthetic client stdout/traceback is retained under `service-control/client-*`;
service stdout is `service-control/stdout-*`; engine events retain their existing
immutable process-lifetime paths. Check `serve logs` and the reported client path
on errors. Neither command replays requests.

The manager uses a new session and passes the existing single-GPU lock FD to the
child, so the lock survives the CLI's exit. It checks an idle GPU and free port
before launch. Duplicate launches refuse. Stop uses Linux pidfd plus recorded
boot ID/start ticks/UID/argv/executable/cwd/session identity; it cannot adopt a
manually launched service or signal a reused PID. It sends SIGTERM to the verified
parent; vLLM must shut down its workers. It never force-kills a group or other jobs.
A startup timeout preserves the process/log for explicit inspection/stop. If
workers remain after an abnormal exit, the next idle-GPU check refuses startup;
the tool does not clean up unowned workers. Missing pidfd support refuses launch
before any GPU check/start. CPU tests exercise real POSIX lock-FD inheritance
with an isolated Python sleep child, plus injected spawn failure, early exit,
readiness timeout, PID reuse races and stop timeout (one SIGTERM, no force kill).
Linux pidfd behavior and vLLM bootstrap/worker shutdown remain unverified here.

Runtime `service-control/current.json` is mutable ownership state, replaced only
on another explicit launch after the old instance has exited. It is not research
evidence or Admission. Sealed Deployment, Round, Admission and prediction files
are never overwritten. The same manager implements existing `--manage-services`.

Before `Popen`, the manager exclusively creates and fsyncs
`service-control/launch.pending.jsonl` and its directory. This intent contains the
anchor, exact command, launch token/time and stdout location. The subsequent
`pid_hint` is diagnostic only, NOT authority to capture/stop that PID. Only after
the complete verified process identity is atomically written and fsynced as
`current.json` is ownership committed and the journal archived as
`launch-TOKEN.jsonl`. Readiness is checked afterwards.

Any pending intent means `launch_incomplete`, even if a complete-looking
`current.json` exists or the hinted PID appears absent. Status/log inspection is
allowed; capture, stop and another launch refuse. Failures/interrupts after spawn
do not send a signal or retry. Even a failed `Popen` leaves the fence: absence of
a recorded PID cannot prove that no child was created during an interruption.
On storage/identity failure preserve the intent, stdout and state/temp files.
There is no automatic reset/adoption command. A human must correlate the launch
time/token/command with actual boot/start/executable/cwd and GPU/port/lock state;
do not rely on PID existence or `kill -0`, and do not kill an uncertain PID. If
ownership remains uncertain, keep the fence and stop. Only after independently
confirming no surviving launch and preserving the evidence may the operator
archive the pending marker and explicitly start again. This manual recovery is
not authorization to resend any failed/unknown model request. A readiness timeout
after a completed ownership commit retains a known service that can be explicitly
stopped by the normal pidfd checks.

## Technical and formal runs

Ten games is a recommended integration example, not a prerequisite. Any legal
technical N uses the existing deterministic whole-game rule. Formal means the
full frozen 150 games (currently 10,714 public observer tasks per model).

The following run examples require an Admission for the configured environment.
Keep the original Candidate-v4 Deployment, Admission and budget unchanged. Its
qualification binds the old prefix; use it for offline history, not to authorize
the new shared prefix. Publish new Run/Permit outputs only after the new physical
qualification has passed. Do not migrate an old/incomplete Run to this prefix.

```bash
uns tom-llm --config tom-llm.yaml prepare --stage run --run-id qwen-tech --models qwen3 --games 10 --mode technical
uns tom-llm --config tom-llm.yaml prepare --stage permit --run-id qwen-tech --models qwen3
uns tom-llm --config tom-llm.yaml run --run-id qwen-tech --models qwen3 --execute-live --manage-services
uns tom-llm --config tom-llm.yaml report --run-id qwen-tech
```

For the final shared three-model run, first obtain all three tokenizer budgets,
then prepare **one** run listing the three models. Issue permits as Admissions
become available; run one model at a time on the 3090:

```bash
uns tom-llm --config tom-llm.yaml prepare --stage run --run-id baseline-150 --models qwen3 yi mistral --games 150 --mode formal
uns tom-llm --config tom-llm.yaml prepare --stage permit --run-id baseline-150 --models qwen3 yi mistral
uns tom-llm --config tom-llm.yaml run --run-id baseline-150 --models qwen3 --execute-live --manage-services
uns tom-llm --config tom-llm.yaml run --run-id baseline-150 --models yi mistral --execute-live --manage-services
uns tom-llm --config tom-llm.yaml status --run-id baseline-150
uns tom-llm --config tom-llm.yaml report --run-id baseline-150
```

Without `--manage-services`, first explicitly `serve start --purpose run --run-id
baseline-150 --model qwen3 --execute-live`; run that single model, then explicitly
`serve stop` with the same purpose/run/model. No PID or Permit digest must be
retyped. `--resume` preserves the existing Runner's rule: only confirmed-not-
executed attempts or accepted durable responses can progress; unknown execution
stays quarantined without resend. Do not migrate an incomplete run to a changed
Source. Report requires every planned model's complete seal and retains all
fallback rows and predeclared metrics. It never publishes a success-only subset.

Reports reuse KL/TV/Top-1_NU, left-minus-right game-paired bootstrap, CSV/JSON,
Markdown/HTML and SVG output. Model preparation, startup and synthetic timing
are not substituted for formal efficiency measurements.

## Source and Admission lifecycle

* Historical `open_admission` validates the v1 immutable Manifest, Deployment,
  76 durable requests/replies/events and two-instance report. It does not inspect
  a running PID, weights or installed environment. Candidate-v4 is not rewritten.
  This supports existing v1 records with the same frozen ModelSpec/template/
  recipe semantics. W5/Mock domains remain rejected. Actual server Candidate-v4
  bytes were not available during Mac tests; compatibility tests use synthetic
  artifacts of that exact format, never claim physical success.
* Historical Deployment/Cohort/Prediction/report reads do not demand current
  HEAD equality. Digests, ordered coverage, task/Selection/Reference/ModelSpec
  and all durable ownership chains still validate. Analysis records execution
  Source and actual analysis Source separately. Native tokenizer assets remain
  required to reconstruct real predictions; no server/old GPU process is needed.
* New qualification services still require their Deployment's exact current
  committed Source. New runs, Permits, real service startup and execution still
  bind exact current Source. Old Permits cannot resume real requests on new HEAD.
* Reuse of historical physical qualification compares the entire qualified and
  current Source inventories except the closed `deployment.CONTROL_FILES` list
  (CLI, source/contract/evidence readers, service orchestration and analysis).
  All other files—including Prompt, task, template, ModelSpec, output parsing,
  probe, SamplingParams validation, shared research/data code and recipe—must
  match. Unknown added runtime files also fail. Compatibility policy v2 compares
  the ENTIRE Runner AST, including `read_predictions`, `collect_rows` and `status`:
  these are also called from online sealing. Only the exact reviewed additions
  of `current_source=False` to `open_run(root)` in those three functions and
  `physical=False` to `open_permit(root,key,digest_anchor)` in `collect_rows` are
  normalized. No function/body is omitted. Changes to result verification, row
  coverage, recovery, dispatch or sealing reject inheritance. The comparison
  requires the qualified Git blob and verifies its SHA; unavailable historical
  source refuses inheritance. Existing Admission bytes are never rewritten to
  install the new policy; only a newly issued Permit records it.
* Control changes require independent review, CPU regression and a new committed
  Source/Run/Permit; they do not grant formal execution or empirical accuracy.
  The new Permit records the compatibility policy, both Source digests and the
  compared inference inventory digest. Changed inference code requires a new
  candidate and two-round GPU qualification, with separate user authorization.
  Physical model SHA, exact prefix environment (with its own audit evidence),
  full budget and active GPU/driver/kernel/process/input evidence
  are revalidated independently. No recorded deviation is repaired or waived.

Yi and Mistral use the same operations with their own candidate and independent
ModelSpec, native suite, budget, service port, rounds, Admission and Permit. No
Qwen physical result is reused for those models. Their explicitly selected prefix
may be shared read-only. All server actions remain manual.

## Supported, proposed and still unverified

Supported now: YAML/JSON operations, optional explicit reference persistence,
small-asset Tokenizer/Suite/full-150 Budget before weights, physical CPU model/
environment preparation, owned service control, unchanged two-round qualification,
offline Admission verification, strict new execution and sealed reports. GPU
operations still require explicit authorization and `--execute-live`.

Not implemented: model/tokenizer download or environment installation. Missing
assets fail clearly; no command downloads implicitly. The smallest proposed
acquisition addition is one explicitly authorized operation over the existing
HF Hub client and frozen manifests, not a new downloader:

1. Read repo ID and FULL revision from `ModelSpec`; read allowed filenames, sizes
   and hashes from its CPU asset table or `physical.asset_spec`. No `main`, custom
   revision, quantized replacement or arbitrary file patterns are accepted.
2. Require separate user download authorization and an explicit future
   `--execute-download` operation (NOT available in this version). Download through
   `hf_hub_download` into a dedicated staging/cache inside the baseline root.
3. Export only manifest-listed regular files to a NEW final directory. Do not
   export Hub cache metadata/symlinks or overwrite existing assets. For full
   weights create the existing `HF_REVISION` marker; the small-asset inventory
   does not accept that extra marker.
4. Reuse `verify_cpu_assets`/native parity for small assets, or `verify_model`
   followed by the existing physical CPU receipt publisher for full assets.
   Partial/mismatched downloads remain staged and are not admitted or repaired.

The current pinned sources (also available in status) are:

| Key | Official repository | Frozen revision |
|---|---|---|
| qwen3 | Qwen/Qwen3-8B | b968826d9c46dd6066d109eabc6255188de91218 |
| yi | 01-ai/Yi-1.5-9B-Chat-16K | 2f8753bed95bd4fede3fd64f8305428620b4b212 |
| mistral | mistralai/Mistral-7B-Instruct-v0.3 | c170c708c41dac9275d15a8fff4eca08d52bab71 |

After review/commit, the next **manual Linux engineering checks**, with separate
GPU authorization where needed, are:

* Re-run CPU operation tests on Linux/Python3.12 to check real pidfd support and
  real process identity. Confirm a second controller cannot acquire the GPU lock
  while the owned child holds its inherited FD, and can acquire it after exit.
* For an authorized NEW vLLM instance, check launch-state argv/PID/boot/start ticks,
  audit FD, port ownership and GPU occupancy. Verify vLLM retains the inherited
  lock through bootstrap. `ready` means owned startup/listener evidence; it is not
  a guarantee that generation or structured output will succeed.
* Explicitly stop that owned instance. Verify API/engine workers have exited,
  port closed and GPU released. If they remain, report/retain state; do not force
  kill or signal unrelated processes. Failed/unknown generation stays blocked
  without resend. Fault injection must never touch collection jobs or sealed
  Qwen rounds.
* Read the original Qwen Candidate-v4 Admission OFFLINE and check v2 Source
  compatibility without retargeting its environment. This is not qualification
  for the new prefix. Do not re-run Qwen's sealed 76 calls in their old destinations.
  Actual v4 bytes have not been read by these Mac tests. All three models still
  need their own physical assets, environment and two-round GPU evidence before
  formal execution; no Qwen evidence can authorize them.
