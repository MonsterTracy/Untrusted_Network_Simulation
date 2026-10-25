# Multimodel physical Admission and GPU Qualification tools (local implementation)

This document prepares a later, explicitly authorized manual run. Codex performs Mac development only; every Ubuntu command is executed manually by the user. No physical model or GPU is qualified by the local CPU/Mock tests. No formal inference runner is introduced.

## Boundaries and artifact chain

Source v2 -> physical model/environment receipts + native suite + common 150-game CPU budget -> per-model Deployment Candidate -> two fresh physical Probe rounds -> new physical Admission. Each arrow verifies external manifest digests and actual source bytes. There is no upgrade from W5, an earlier Mock contract, or a dirty-source execution.

The old Qwen Policy, Wire v2, native template, PRE, parser/fallback, Selection, Reference and scoring are unchanged. The W5 Qwen synthetic suite is copied byte-for-byte to qualification_assets/qwen3-suite-v1.json, SHA256 f1052efa861e458606d59b33793649fc81772d13958aad3c38ddffb7acf0f98f. Its 12 cases, native input_ids, wires, 2 warmups + 36 measurements per instance, two instances and 76 total calls remain exact. Only the surrounding Source/Deployment/Admission identities are new. Yi/Mistral use those same public synthetic tasks with independently computed native token IDs. No old Candidate qualification is inherited.

ModelSpec revisions remain:

- Qwen/Qwen3-8B: b968826d9c46dd6066d109eabc6255188de91218
- 01-ai/Yi-1.5-9B-Chat-16K: 2f8753bed95bd4fede3fd64f8305428620b4b212
- mistralai/Mistral-7B-Instruct-v0.3: c170c708c41dac9275d15a8fff4eca08d52bab71

The new physical asset specification is built from official revision API metadata with LFS SHA256/byte sizes and Git blob OIDs for small files. It binds every required shard, index, tokenizer/config/generation file, and LICENSE (Qwen) or the license-bearing README (Yi/Mistral). Physical verification streams files, records SHA256 for every file, requires HF_REVISION and exact index/shard inventory, and never loads tensors. No downloader is provided.

The environment receipt uses its own isolated Python 3.12 interpreter, package versions, METADATA/RECORD hashes, pinned vLLM 0.27.0 critical source files and critical Torch/FlashInfer/TVM-FFI/OpenAI source. It runs pip check without installation or GPU imports. This is NOT a new full installed-binary/RECORD audit: the earlier strict nine RECORD failures remain unchanged and must be supplied by their external prior-audit digest. No exception is repaired, waived or relabeled as a strict PASS. Missing packages/source mismatch halt preparation.

Source v2 re-runs Git status over the entire extended runtime file list, including both new configs, all new runtime Python modules, CLI, templates and Qwen suite. Legacy development_source.py is untouched. Tests/docs belong to delivery evidence rather than the execution inventory. After a future commit, rebuild Source and all dependent artifacts; byte equality alone does not preserve HEAD/tree/status identity.

## Mac: review and publish later, not in this phase

1. Review the actual diff and local test evidence in codex/tom-multimodel-baseline. No additional worktree is needed.
2. Only after separate authorization, commit/publish the approved new implementation to GitHub. Record the exact new commit/tree and Source v2 digest. The current base 95ee58b is NOT a commit containing these uncommitted qualification additions.
3. Keep W3, W5 and old Candidate artifacts unchanged. Do not re-sign old responses under a new identity.
4. Transfer/clone only that approved source on Ubuntu. A deployment bundle contains no .git, weights, environment, or dataset. Runtime Source validation requires an actual Git checkout at the approved commit, not an untracked copy presented as a clean checkout.

## Ubuntu: manual CPU preparation only

Prerequisites: a separately approved exact source commit, existing frozen V1 publication/OOF seal/Selection and canonical provenance, approved model acquisition (if absent), an isolated unchanged environment, and an empty per-model candidate directory. No model download, environment installation, registry edit or cleanup is authorized by this document.

Use independent roots:

| model key | root | loopback port |
|---|---|---:|
| qwen3 | /data/yuxiao/tom-llm-reasoner-b2/multimodel/qwen3/candidate-v1 | 18581 |
| yi | /data/yuxiao/tom-llm-reasoner-b2/multimodel/yi/candidate-v1 | 18582 |
| mistral | /data/yuxiao/tom-llm-reasoner-b2/multimodel/mistral/candidate-v1 | 18583 |

All writes stay inside /data/yuxiao/tom-llm-reasoner-b2. Old W5 directories and services are not deployment targets. Approved existing Qwen model/environment directories may be reused READ-ONLY after complete new binding. Different models have separate deployment, cache, trace, response and admission roots; a shared environment is allowed only as an explicit immutable read-only physical asset with separate per-model receipts/bindings. Never clone/reinstall Conda automatically.

In the commands below, replace every uppercase placeholder using reviewed absolute paths/digests. TOOL is the approved checkout's scripts/tom_multimodel_admission.py; PY is the chosen isolated environment/bin/python. Do not point TOOL at W5. Every publisher uses a NEW destination and fails on conflict; never delete/overwrite an old artifact to retry.

1. **Source and resource checks (read-only).** Inspect branch/HEAD/status and compare the approved runtime manifest. Read GPU/process/disk status manually; hashing shards can create substantial disk I/O, so schedule CPU checks without disturbing collection. Do not stop other jobs. Physical receipt commands do not allocate a GPU.

2. **Model and environment receipts (write only new receipt directories).**

```bash
"$PY" -B "$TOOL" assets --model qwen3 --directory "$MODEL_DIRECTORY" --output "$ROOT/receipts/model"
"$PY" -B "$TOOL" environment --prefix "$ENV_PREFIX" --prior-audit-digest "$APPROVED_PRIOR_AUDIT_DIGEST" --output "$ROOT/receipts/environment"
```

Repeat for yi/mistral with their own root/model. Expected output is a new artifact path and manifest_digest, not GPU Admission. Wrong revision, index/shards, symlink, extra physical files, missing package, dependency failure or source drift stops. Do not change an existing environment to make the command pass. Model hashing is streamed but reads all shards; its time is preparation overhead, not inference latency.

3. **Native tokenizer suite (CPU; writes a new suite).** Supply a separate small-asset directory containing exactly the ModelSpec whitelist, copied without symlinks from already verified assets. Full weight directories are intentionally rejected by the small-tokenizer loader; do not point --tokenizer-assets at a weight directory. No auto-download occurs.

```bash
"$PY" -B "$TOOL" suite --model qwen3 --tokenizer-assets "$SMALL_ASSETS" --output "$ROOT/receipts/suite"
```

Expected: 12 official native token sequences, exact Qwen compatibility where applicable. Any ID/template mismatch stops; do not edit the frozen expected suite. Actual small assets are required; a fake counter cannot build this artifact.

4. **One shared public 150-game cohort (CPU; new artifact).** Reuse the existing Selection Artifact and its reviewed expected/provenance JSON; do not call a new sampling procedure. The prepared root is the historical backbone_tom_study artifact referenced by that provenance. Build all public-alive tasks before opening prediction rows; Reference is used only for strict identity/coverage validation, never for Prompt content. No scoring is performed.

```bash
"$PY" -B "$TOOL" cohort --publication "$PUBLICATION" --evaluation "$FROZEN_OOF_EVALUATION" \
  --expected-reference "$OOF_SEAL" --selection "$EXISTING_SELECTION" \
  --expected-selection-json "$SELECTION_EXPECTED_JSON" --selection-provenance-json "$SELECTION_PROVENANCE_JSON" \
  --prepared "$FROZEN_PREPARED_ROOT" --output "$NEW_SHARED_COHORT"
```

Expected: existing same 150 games, public-alive rows and frozen OOF/Selection identity. Missing/duplicate/foreign rows fail, no intersection or Primary-based request filtering. The request file contains no labels/q/private roles/raw speech.

5. **All-row CPU budgets and matrix (CPU; new per-model budget).**

```bash
"$PY" -B "$TOOL" budget --model qwen3 --cohort "$NEW_SHARED_COHORT" --expected-cohort "$COHORT_MANIFEST" \
  --tokenizer-assets "$SMALL_ASSETS" --output "$ROOT/receipts/budget"
# Repeat for yi and mistral, using exactly the same cohort/digest.
"$PY" -B "$TOOL" inspect-budgets \
  --qwen3 "$QWEN_BUDGET" --expected-qwen3 "$QWEN_BUDGET_MANIFEST" \
  --yi "$YI_BUDGET" --expected-yi "$YI_BUDGET_MANIFEST" \
  --mistral "$MISTRAL_BUDGET" --expected-mistral "$MISTRAL_BUDGET_MANIFEST"
```

Expected: identical cohort/Selection/Reference/row identity/count, with separate p50/p95/p99/max/total token counts. Every input+512 must fit8192. Overflow stops the whole model; never truncate PRE, omit requests, enlarge Qwen context or select a shorter subset. Historical 10,714 requests is context, not a hardcoded substitute for actual validated coverage.

6. **Deployment Candidate (CPU; new artifact).** Values below are the just-verified physical receipts. Explicit GPU UUID is recorded, not silently copied from W5. No HTTP request or GPU allocation occurs.

```bash
"$PY" -B "$TOOL" deploy --model qwen3 --gpu-uuid "$REVIEWED_GPU_UUID" \
  --model-directory "$MODEL_DIRECTORY" --environment-prefix "$ENV_PREFIX" \
  --assets "$ROOT/receipts/model" --expected-assets "$MODEL_RECEIPT_MANIFEST" \
  --environment "$ROOT/receipts/environment" --expected-environment "$ENV_RECEIPT_MANIFEST" \
  --suite "$ROOT/receipts/suite" --expected-suite "$SUITE_MANIFEST" \
  --budget "$ROOT/receipts/budget" --expected-budget "$BUDGET_MANIFEST" --output "$ROOT/deployment"
```

The receipt rechecks actual model and environment bytes. Source can still be a candidate during local development, but physical startup requires committed clean runtime Source. Profiles bind BF16, no quantization/reasoning parser,8192/512,T0,seed,one sequence, fixed native template, xgrammar backend, prefix caching off, generation_config auto, no request-level template and no proxies/retries. Qwen Wire v2 is untouched. Auto tokenizer mode is only accepted if its actual final input_ids match the approved native suite; no fallback to another tokenizer/template is allowed.

CUDA/TVM-FFI/FlashInfer/Torch/Numba/Triton/vLLM/HF/XDG caches and TMPDIR are scoped to the model root. PYTHONDONTWRITEBYTECODE=1 avoids writing into reused source/environment trees. The environment prefix itself is not modified.

## Ubuntu: later GPU window, separate explicit authorization required

Do not execute this section merely because CPU checks passed or the GPU is idle. The user first reviews CPU results and explicitly authorizes a dedicated GPU window. No existing collection service may be stopped/restarted/retuned. The launcher refuses a non-idle or mismatched single RTX3090/24GiB window.

Run only one model at a time. In a foreground terminal, launch the approved env/bin/python with exactly:

```text
ENV_PREFIX/bin/python -m werewolf.tom_llm.gpu_probe --deployment ROOT/deployment \
  --expected-deployment DEPLOYMENT_MANIFEST --execute-synthetic-gpu-qualification
```

Before launching, manually supply precisely deployment.json profile.environment_requirements to this child process; remove proxy variables and undeclared VLLM_* overrides from that child environment. Work in the approved source checkout under ROOT/source/Untrusted_Network_Simulation. Do not use a request-level chat template. The fixed template file at ROOT/template/chat_template.jinja must match the candidate artifact; copying it is a reviewed NEW-path deployment step, never overwriting an old template.

The launcher rechecks Source, model, environment, cache config, template and idle GPU BEFORE vLLM import/start. It validates actual parsed serve arguments and instruments the pinned vLLM final EngineCoreRequest seam. Unsupported argument names/private API/JSON backend fail closed; CPU tests do not prove vLLM0.27, FlashInfer/JIT or RTX3090 compatibility. Do not modify the protocol to bypass a failure.

In a second terminal, using the actual new frontend PID:

```bash
"$PY" -B "$TOOL" round --deployment "$ROOT/deployment" --expected-deployment "$DEPLOYMENT_MANIFEST" \
  --pid "$OWN_FRONTEND_PID" --run 0 --output "$ROOT/qualification/round0" --execute-synthetic-gpu-qualification
```

Expected: exactly38 dispatches, startup+76 engine events, durable raw replies/calls and a round seal. Check the service-owned loopback socket AND open audit-log descriptor. Successful rounds must preserve final input_ids, effective SamplingParams/JSON Schema, usage/token agreement, memory samples and repeated output reproducibility.

After round0, the user shuts down ONLY this dedicated new service instance using its own foreground terminal. Keep all evidence. Launch a second fresh instance with the same approved deployment, collect run1 into a new round1 path. Two different process lifetimes and exactly76 total calls are required. Do not touch W5 or Qwen3.5 services. A failed/unknown request consumes its ordinal; there is no automatic retry/resume, even for a confirmed-not-executed qualification call. Never overwrite a failed round or re-sign its replies. Stop, retain evidence and review before any new attempt.

Yi requires effective EOS union {2,7}, while Wire explicitly adds stop7; the engine may normalize implied primary EOS out of the explicit stop list, but the final all-stop set must match. Mistral requires native HF-equivalent [INST]/system/BOS IDs and EOS2. Qwen requires the original exact Non-thinking input suffix and no think tokens in outputs. Schema constrains keys/numeric ranges; sum-to-one remains the old parser's semantic check, and unchanged fallback statistics are retained. Stop reasons are captured without inventing missing engine evidence.

CPU/HTTP/guard/monitor overhead is included in qualification client-wall timings. Diagnostic token throughput and sampled whole-device memory are NOT formal warm-model efficiency or allocator high-water marks. A fair final cost experiment needs separate approved timing/TTFT/GPU measurements and the same observer/PRE/game coverage.

Finally, while the second instance remains active:

```bash
"$PY" -B "$TOOL" admit --deployment "$ROOT/deployment" --expected-deployment "$DEPLOYMENT_MANIFEST" \
  --round0 "$ROOT/qualification/round0" --expected-round0 "$ROUND0_SEAL" \
  --round1 "$ROOT/qualification/round1" --expected-round1 "$ROUND1_SEAL" --output "$ROOT/qualification/admission-v1"
"$PY" -B "$TOOL" inspect-admission --model qwen3 --admission "$ROOT/qualification/admission-v1" \
  --expected-admission "$ADMISSION_MANIFEST" --expected-deployment "$DEPLOYMENT_MANIFEST"
```

Expected only after actual physical success: a NEW synthetic-only physical Admission, whose reader revalidates raw durable chains, both rounds, CPU identities and the active second instance. Mac and Mock cannot publish it. W5/mock/other-model artifact domains are rejected. Formal execution remains unauthorized and unimplemented; do not send the150-game cohort to this synthetic-only service.

## What remains NOT_RUN after local tests

Full physical model/environment/CPU150 admission on Ubuntu; actual budget totals; real engine input_ids/arguments/JSON backend; JIT/kernels/OOM/memory/throughput; two real restarts and76 calls per model; physical Admission publication. Full weight acquisition or environment changes, if needed, require separate approval. The existing nine RECORD deviations remain disclosed, unmodified and outside this implementation's repair scope.

Cache and log parents must not be symlinks. PATH is fixed to ENV_PREFIX/bin:/usr/bin:/bin. Undeclared CUDA/Torch/FlashInfer/TVM/Triton/Numba/NCCL/CUBLAS/HF/Transformers backend variables and loader/proxy overrides are rejected; do not silently delete a required setting to bypass the gate. Submit a new reviewed profile if the real environment needs one. GPU receipts record actual driver and kernel identities and require both restarts to agree. Compiler/JIT execution compatibility remains an actual GPU test, not an assertion from package metadata.

After validating the CPU Deployment and before authorizing a GPU launch, the user must prepare all declared cache/TMPDIR directories and the fixed template at NEW owned paths. The launcher requires the directories to exist and be writable: a nonexistent TMPDIR could otherwise make Python/JIT fall back to system /tmp. This local implementation does not create server directories. Example manual Ubuntu layout preparation, after replacing the placeholders and reviewing the Deployment digest:

```bash
"$PY" -B - "$ROOT/deployment" "$DEPLOYMENT_MANIFEST" <<'PY'
import sys
from pathlib import Path
from werewolf.tom_llm.deployment import open_deployment, validate_cache_paths, CACHE_VARIABLES
artifact, record = open_deployment(sys.argv[1], sys.argv[2], physical_recheck=True)
p = record['profile']
validate_cache_paths(p)
for key in CACHE_VARIABLES:
    Path(p['environment_requirements'][key]).mkdir(parents=True, exist_ok=True)
validate_cache_paths(p, require_existing=True)
target = Path(p['argv_options']['chat-template'])
assert not any(v.is_symlink() for v in (target, *target.parents))
target.parent.mkdir(parents=True, exist_ok=True)
with target.open('xb') as stream:  # fail if already present; do not overwrite
    stream.write((artifact.path/'chat_template.jinja').read_bytes())
print('layout prepared; no GPU or service started')
PY
```

If this step encounters an existing template, conflicting artifact, permissions problem or symlink, stop and compare identities. Do not remove an existing path to force preparation to pass.
