# Optional LLM Reasoner integration

This is a selectively integrated baseline interface, not an admitted deployment
or a formal experiment result. It is never selected by the default game runtime.

## Available code

`werewolf.tom_llm.prompt` renders only the complete public structured token plan
and one alive observer's seat. Raw speech, supervision, private roles, previous
LLM responses and future speech are not prompt inputs. The fixed Qwen3-8B policy
and two prompt resources are preserved from the original research implementation.
Suspicion mass is observer-relative, not a wolf-role posterior.

`output` validates exactly six non-self seats, finite nonnegative mass and the
registered sum tolerance. It preserves the original epsilon smoothing and
distinguishes valid uniform output from a recorded semantic failure's uniform
fallback. Transport exhaustion, uncertain in-flight outcome or infrastructure
failure does not fabricate a prediction. The durable Mock runner retains at
most two transport dispatches and one semantic response, with immutable attempts,
response recovery, seals and source/row ownership checks.

`transport` accepts only explicitly injected clients with fixed model/tokenizer/
template declarations, disabled SDK retries and a loopback endpoint. Identity is
checked around a dispatch. `server_fixed` additionally binds explicit synthetic
service evidence and the original server-fixed wire surface. Neither interface
constructs a network client, starts a server or proves physical weights/runtime
identity. The runner accepts only `MockBackend`; injected provider seams are not
a live execution factory or formal admission.

`reference` reads explicitly pinned historical V1 Qwen3 OOF seals and checks
Publication/folds, terminal/checkpoint, PRE/observer/target and Primary identities.
It performs no model inference and does not turn the historical source into a
new execution qualification. This reader does not claim V2 or Experiment A
Reference support. `evaluation` uses exact public-alive row joins, existing
game-macro KL/TV/NU scoring and paired bootstrap mathematics. The fixed
full1500/subset150 selection is not a new V2 cohort or a Test selection rule.
Mock comparison artifacts remain engineering-only, not paper results.

## Separate source and artifact domains

The old `BASE_COMMIT=d8397ce56c65d6d46b24a6b53f0b71431c04267b`
and `study.source_identity()` guard are retained. That guard rejects Mainline;
it is not bypassed or updated to accept the new HEAD. W5 and its Candidate bytes,
branch and historical source inventory remain the original source.

New Mock studies use `classic7_llm_mainline_development_source_v1` and explicit
Mainline Mock study/attempt/result/seal/execution/service/comparison versions.
The new source contract requires ancestry from the S4 integration base and binds
actual HEAD/tree, runtime file/asset/config SHA inventory and whether those source
files are uncommitted. Dirty development is recorded as such, never represented
as a clean historical revision. Source or commit changes invalidate a prepared
Mock run; there is no legacy artifact upgrade or alternate-digest fallback.
This contract grants only `MOCK_ONLY_NOT_CANDIDATE` status.

Candidate/admission, physical-model manifests, vLLM probes, GPU/qualification
tools, deployment profiles and old operator kits are intentionally not installed
in this integration. New physical execution requires a separately reviewed
source/deployment/asset/environment/qualification contract. Passing CPU tests
does not qualify the old Candidate or any new deployment.

## Explicit CLI and tests

The only additional shared CLI command is offline validation:

```sh
python -m werewolf.cli check-llm-reasoner-config \
  --config configs/paper/tom-llm-reasoner-v1.json
```

The optional `llm-reasoner` extra adds `Jinja2>=3.1,<4`; setup packages only the
two prompt resources. Scientific commands still use an explicit source checkout
and reviewed configurations. No model, environment, credential or Candidate
payload is distributed. Default CLI parsing does not import Jinja2 or inference
dependencies. Existing ToM/Mapper extras and all previous commands are preserved.

```sh
python -m pytest tests/tom_llm -q
```

Tests use artificial PREs, distributions, services and sealed References. The
optional official-tokenizer parity tests remain NOT_RUN unless an independently
approved local asset directory is explicitly supplied; no assets are downloaded
or inferred. They are not a model or GPU qualification. The old physical
admission suite is held with W5 rather than represented as a new Mainline PASS.
