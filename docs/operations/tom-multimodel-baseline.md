# ToM multimodel Baseline experiment system

Mac develops and tests; the user manually deploys and operates Ubuntu. These commands are usage documentation, not server/GPU authorization. No model is admitted by local tests. The old 76-call Qwen synthetic qualification, all ModelSpecs, public PRE, Wire v2, parser/fallback, Primary and scoring implementations remain unchanged.

## Design and identities

`experiment.py` owns immutable run contracts and Source v3. `live_admission.py` binds an execution Permit to the complete Run, Source, ModelSpec, common Cohort, sealed physical qualification, CPU budget, native assets and service configuration. It records no human-approval evidence. `live_service.py` validates final native input_ids/effective sampling for each real request; it is separate from the frozen synthetic service. `live_runner.py` owns serial HTTP calls, durable attempts/replies/acceptance/results and seals. `baseline_analysis.py` reopens complete seals, the frozen OOF Reference, and produces all predeclared metrics/reports. `baseline_cli.py` connects those modules.

Source v3 extends Source v2 with both CLI scripts and the fixed experiment policy, and rechecks Git status across the whole runtime inventory. After a startup-format source repair, regenerate Source-bound Cohort/Budget/Deployment/Run/Permit with the same Selection and request identities; do not patch old Artifacts. Existing weights, environment and source-independent receipts/Suite are reusable after strict validation. It is software identity, not GPU authorization. Source v2 code is unchanged, but its inventory includes all werewolf Python modules: adding these modules also changes its digest. **The old f3ca87cb… Source is not the identity of the new working tree.** After an authorized commit, rebuild Source, Cohort, budgets and Deployment/qualification/permits under the committed bytes. Never re-sign old responses or reuse a W5 Candidate/Mock seal. Model/Suite physical assets may be reused read-only after exact ModelSpec verification; source-independent Selection/Expected/Provenance identities do not change.

The common cohort is the existing public-only engineering artifact, not an inference result. Preparation either strictly reopens a supplied cohort digest or uses the existing exporter. It verifies the two original historical evidence files by byte SHA and requires an already published Selection. It never publishes Selection. Formal mode accepts exactly all150 games, 30/fold. Technical/fake mode selects N whole games deterministically from that fixed Selection with `tom_llm_technical_subset_v1`, seed20261026; no labels, errors, model outputs or prompt lengths enter this selection. N is a game count (1–150), not request count; all alive observers and full legal PREs are retained. The separate technical rule does not alter the frozen150 list.

The old require_live_admission stub and Mock APIs remain blocked and unchanged; only the new explicit permit/Runner path supports live execution. Formal results require exact Source v3, ModelSpec, Selection, Reference, ordered task identities, per-model permit/session ownership and complete sealed replies. Synthetic Admission remains synthetic-only; a new execution Permit binds its sealed qualification evidence to the actual experiment. Creating a Permit is CPU preparation and does not authorize or start GPU work. Server operations require the user's separate authorization; actual execution additionally requires `--execute-live`.

## Environment and preparation

Use a Git checkout of the newly reviewed/committed source and the existing isolated Python3.12/vLLM0.27 environment and pinned critical source files. Tests use pytest, Torch, Transformers, Jinja2, openai/httpx2 (legacy interfaces), and Matplotlib for reports. The control/analysis environment may be separate from each immutable model prefix. Revalidation invokes each model prefix's own Python3.12 CPU child, calls the unchanged capture_environment, and compares its entire result; it never installs into that prefix. Install/report dependencies only in a separately authorized independent CPU environment when the frozen model environment lacks Matplotlib. The real Runner itself uses standard-library HTTP: one connection per attempt, no SDK/proxy/redirect/hidden retry. No automatic model download or environment installation is provided. Tokenizers require exact existing official small assets; full weight directories are rejected by that loader.

Start from `configs/paper/tom-llm-experiment.example.json`. Verify actual server paths and replace the two handoff placeholders and all three tokenizer asset locations. Verify the actual published Selection/Expected/Provenance paths. Reuse an existing approved Selection read-only; preparation never republishes it and must fail if it is missing. An existing cohort can be configured as `{"path":"ABSOLUTE_PATH","digest":"REVIEWED_MANIFEST_SHA256"}`. Existing per-model budgets can likewise be supplied as `models.KEY.budget`. An absent budget is generated using every row of the full150 cohort for live modes. Fake mode may omit physical tokenizer budgets and cannot gain physical qualification.

The CLI never edits this input file. Its full JSON and byte SHA are sealed into the run contract. Do not put credentials into it. All Ubuntu live output and caches stay inside `/data/yuxiao/tom-llm-reasoner-b2`; existing environment/model/source paths are read-only.

After separate CPU-artifact authorization:

```bash
PY=/data/yuxiao/tom-llm-reasoner-b2/env/bin/python
OUT=/data/yuxiao/tom-llm-reasoner-b2/experiments
$PY -B scripts/run_tom_multimodel_baseline.py --action prepare \
  --config /APPROVED/experiment.json --models qwen3 yi mistral \
  --games 150 --mode formal --run-id baseline-150 --output-root "$OUT"
$PY -B scripts/run_tom_multimodel_baseline.py --action status --run-id baseline-150 --output-root "$OUT"
```

Preparation performs strict Reference validation but no scoring or model call. Public tasks are constructed by the existing exporter before prediction rows are opened. Reference/Primary labels are used only in existing identity/coverage validation and subsequent sealed-result analysis, never in Prompt construction. Preparation is a publisher: use a new run directory; failures leave evidence and must be reviewed, not deleted/replaced to retry.

## Physical qualification and execution binding

1. Review new committed Source v2/v3 identities, the actually published Selection, model/environment receipts, native suites and all-row150 budgets. Qwen/Yi/Mistral must each fit input+512<=8192. No truncation, model-dependent game filtering, quantization or replacement revision.
2. Build each physical Deployment with the existing `scripts/tom_multimodel_admission.py deploy`, binding this common cohort/budget. Follow `tom-multimodel-qualification-v1.md` for an independently authorized idle3090 window, two distinct instances and76 synthetic calls per model. GPU compatibility, EOS/schema, FlashInfer/JIT, memory, input_ids and restart behavior remain actual GPU tests. The Qwen76 schedule is untouched.
3. Prepare the desired formal150 or technicalN Run, then create one execution Permit per model. Both modes use the same sealed qualification and identity checks. Deployment roots come from the existing `deploy --output NEW_ROOT/deployment`; use a new root for this source/configuration repair, preserving old candidates. No previous technical Run is required, and neither historical qualification process needs to remain alive.

After separate authorization for CPU artifact preparation:

```bash
$PY -B scripts/run_tom_multimodel_baseline.py --action permit --run-id baseline-150 --output-root "$OUT" \
  --models qwen3 --qualification /ACTUAL/QWEN/ADMISSION --expected-qualification QUALIFICATION_MANIFEST \
  --budget /ACTUAL/QUALIFIED/QWEN/BUDGET --tokenizer-assets /ACTUAL/QWEN/SMALL_ASSETS
```

Repeat for Yi/Mistral using each model's own qualification, exact qualified budget and native assets. The budget must equal the one in that model's qualified Deployment and share the Run's common Cohort. Record each returned Permit manifest digest. No Permit can be published from Mac/Mock or dirty runtime Source. Publication and strict readback revalidate sealed physical qualification, model bytes, environment, Source, ModelSpec, common Cohort/Budget and fixed service parameters. Driver/kernel identities remain bound to qualification and are checked when the new live instance starts and at live-instance guards. The new service independently validates its current process, environment, socket and log ownership, plus the idle-GPU gate before launch. It never stops discovered collection services.

There is no external human-approval file or software assertion of human approval. Permit creation binds provenance; `--execute-live` distinguishes real execution. The user's authorization for server operations is confirmed independently before any actual operation. Earlier development Permits have a different Source/content identity and cannot be reused or upgraded.

## Optional technical test (10 games recommended)

Technical mode accepts any deterministic whole-game subset of the frozen Selection with 1<=N<=150. Ten games are a recommended engineering check, not a condition for issuing a formal Permit or running formal150.

```bash
$PY -B scripts/run_tom_multimodel_baseline.py --action prepare \
  --config /APPROVED/experiment.json --models qwen3 yi mistral --mode technical --games 10 \
  --run-id baseline-tech10 --output-root "$OUT"
```

Issue technical Permits with the same command as above using `--run-id baseline-tech10`. Run with its own Permit digests and `--run-id baseline-tech10 --games 10 --mode technical`. Technical results are engineering results; Prompt and decoding are identical to formal. Each Run's Permit binds its own mode/game/task identities and cannot be reused by another Run.

A single-model Qwen technical Run can be prepared with `--models qwen3` before the other models' assets exist. The three-model formal Run needs all three models' physical assets and budgets. To reuse an existing common Cohort/budget, supply its exact path/digest in the Run config, including the actual qualified budget rather than a newly generated different-path payload. Physical qualification and budget checks remain necessary even when no technical Run exists.

## Real execution on a single3090

This is a GPU operation requiring separate user authorization. Never run merely because GPU is idle. The managed path launches models sequentially, verifies current GPU/process/cache/service identity, and terminates only the process group it created. The shared GPU lock coordinates this system's runs; the idle-GPU gate also refuses other existing compute processes. It never searches for or kills collection processes.

```bash
$PY -B scripts/run_tom_multimodel_baseline.py --run-id baseline-150 --output-root "$OUT" \
  --models qwen3 yi mistral --games 150 --mode formal --execute-live --manage-services \
  --permit qwen3=QWEN_PERMIT_MANIFEST --permit yi=YI_PERMIT_MANIFEST --permit mistral=MISTRAL_PERMIT_MANIFEST
```

The Permit stores `vllm.yaml` in the same native format as Qualification. Its serializer uses explicit native negative switches for declared false booleans; the exact parsed profile is still checked, including prefix caching off and the single served-model name. Caches/TMPDIR are created only in that model/run's new `service/` tree. Only declared cache variables are relocated; PATH and CUDA_HOME keep the qualified environment prefix. Service configs/templates are immutable permit files. The service invokes the actual environment Python with `-m werewolf.tom_llm.live_service`, pinned runtime argv and scoped child environment. vLLM imports/startup occur only after permit/source/physical/cache/idle-GPU gates. Each request's final engine input_ids and effective SamplingParams/schema/EOS are logged and matched to its durable raw HTTP reply/usage. New service startup refuses unresolved requests before GPU loading.

For user-managed foreground services, use the profile/environment in the permit and the exact command:

```text
ENV/bin/python -m werewolf.tom_llm.live_service --run ABSOLUTE_RUN_ROOT \
  --model qwen3 --expected-permit QWEN_PERMIT_MANIFEST --execute-live
```

Then run the experiment CLI for only one model with `--execute-live --permit qwen3=... --service-pid qwen3=ACTUAL_PID`, without `--manage-services`. The process argv, executable, cwd, loopback socket, log FD, GPU/driver and environment must match. The second model cannot start until the GPU window is free. Managed mode is simpler and avoids manually reconstituting the environment. Neither path uses the old synthetic38-call client.

## Recovery and failure semantics

Every request has immutable dispatch, raw reply, acceptance and parsed-result artifacts. Prediction Seals also bind each attempt outcome and every existing Session/completion artifact, including measured timing; strict readers reject replaced or newly added timing artifacts. Raw reply is stored before identity/proof/parsing checks. Legal uniform prediction and failure-uniform fallback remain distinct. Malformed model content is parsed exactly once using the old fallback policy; no semantic repair or extra generation. Missing/wrong usage, physical identity drift and network uncertainty are infrastructure failures, not distributions.

```bash
$PY -B scripts/run_tom_multimodel_baseline.py --action status --run-id baseline-150 --output-root "$OUT"
# Only after reviewing status, use the same execution arguments plus:
# --resume
```

Before any resumed network request, the entire existing durable prefix/results are validated. A fully accepted saved reply can be converted/sealed locally without repeating its request. An explicit pre-connect failure or429 rejection can use the second and final transport attempt. Two confirmed rejections exhaust the budget and leave the run incomplete. A post-connect timeout, unaccepted reply, rejected reply, crash after dispatch, unknown outcome or partial/corrupted artifact blocks automatic recovery. Keep evidence and stop for review; no blind resend, omission, forced overwrite or fabricated fallback. A saved reply without its acceptance artifact is conservatively blocked. Termination during publication can leave staging; generic artifact I/O retains its existing behavior and no retry bypasses it.

Different models/revisions/permits/services/run IDs cannot share response state. A run/model lock prevents concurrent dispatch; no finished model/run is silently re-executed. All planned models must be completely sealed before a run seal or analysis is published. Request/attempt/session data remain available even when no success seal exists.

## Analysis and paper artifacts

```bash
$PY -B scripts/analyze_tom_multimodel_baseline.py --run-id baseline-150 --output-root "$OUT" \
  --expected-run-seal ACTUAL_COMPLETE_RUN_SEAL
```

The analyzer reopens the run seal, every model seal, original request/reply/acceptance chains, ModelSpec/permit/source and exact Reference before scoring. All models share Selection, public-alive requests and Primary/NU evaluation rows. It reuses `evaluation.compare`, `paired_extras`, `score_prediction_rows`, `game_macro_summary` and `paired_summary` unchanged. It reports KL/TV/Top-1_NU and LLM−Observer-ToM as well as every LLM pair's left−right difference. Bootstrap is the existing game-level10000-replicate,95% percentile-linear protocol, with the same NU-game restriction and fixed seed. Lower KL/TV and higher Top-1_NU are better. Uniform is retained as a secondary reference. No complete-case or performance-based model/sample selection.

`analysis/` is a new immutable artifact containing `analysis.json`, `metrics.csv`, `paired.csv`, `performance.csv`, `report.md`, `report.html`, `metrics.svg`, `efficiency.svg`. An existing destination is rejected; use a new `--report-output` for a deliberately separate regeneration. Analysis never invents values for an incomplete run. Fake/technical reports and plots are explicitly marked ENGINEERING ONLY; formal_result is true only for a complete formal experiment with the live evidence chain.

Efficiency reports include all public-alive request/attempt counts, semantic validity/fallback reasons, legal uniform count, input/output tokens, measured per-attempt client wall (including request conversion, HTTP/service and confirmed rejections), and completed request-loop wall including per-row validation/artifact I/O. Interrupted sessions are marked as lacking complete session timing, not zero-cost work. Client timings are nonstreaming, including connection/service/queue time; final sealing and analysis are outside request-loop wall; TTFT, allocator high-water and Observer-ToM inference time are not measured and remain null. Synthetic qualification timings are never substituted. Startup/preparation/physical hashing overhead is outside per-attempt HTTP time and should be separately measured in a later approved fair-cost experiment. The report cannot claim pure GPU throughput or an Observer-ToM speedup without that measurement. The comparison is method-level, including pretraining/posttraining, explicit rules and Qwen family relation, not a pure architecture comparison.

## CPU/fake-service tests

The local test suite uses only synthetic publication/OOF fixtures and a real loopback HTTP fake server; native-tokenizer tests use provided official small assets offline. It performs no model forward. Provide `UNS_LLM_TOKENIZER_ASSETS`, `UNS_LLM_YI_ASSETS`, `UNS_LLM_MISTRAL_ASSETS` to existing verified small directories. Missing native assets remain NOT_RUN/skipped; never claim physical qualification.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -B -m pytest tests/tom_llm -q -p no:cacheprovider --basetemp=/tmp/NEW_OWNED_TEST_DIRECTORY \
  tests/canonical_collection/test_structured_token_plan.py \
  tests/tom/test_cli.py tests/tom/test_dataset.py tests/tom/test_paper_tom_metrics.py \
  tests/tom/test_paper_top1_nu.py tests/tom/test_population.py tests/tom/test_protocol.py
```

Use an environment containing pytest, Jinja2, Torch, Transformers/tokenizers, the existing SDK/httpx2 dependencies and Matplotlib. Tests cover real local HTTP, exact wire bodies, three-model sealing/reporting, parser/fallback reuse, paired directions, timeout/accepted-reply recovery, cross-model/source/tamper/coverage rejection, native final input_ids/params/EOS, CLI and no CPU-issued physical authorization. CPU evidence cannot establish actual vLLM0.27 private-seam compatibility,3090 BF16 capacity, kernels/JIT, physical restarts or any model's empirical accuracy.
