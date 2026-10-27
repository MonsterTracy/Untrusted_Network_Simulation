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
checkout and existing controller dependencies (including the optional ToM/LLM
dependencies); each model retains its own immutable Python3.12/vLLM prefix.
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

For an already-qualified Qwen, pin the original Candidate-v4 Deployment,
Admission and budget in YAML. Do **not** prepare over it or re-run its rounds.

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
  Physical model SHA, prefix environment (including all nine disclosed RECORD
  deviations), full budget and active GPU/driver/kernel/process/input evidence
  are revalidated independently. No recorded deviation is repaired or waived.

Yi and Mistral use the same operations with their own candidate, prefix,
ModelSpec, native suite, budget, service port, rounds, Admission and Permit. No
Qwen physical result is reused for those models. All server actions remain manual.

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
* Read the original Qwen Candidate-v4 Admission OFFLINE and check v2 compatibility
  for the new committed Source. Do not re-run Qwen's sealed 76 calls for these
  checks. Actual v4 bytes have not been read by these Mac tests. Yi/Mistral still
  need their own physical assets, environment and two-round GPU evidence before
  formal execution; no Qwen evidence can authorize them.
