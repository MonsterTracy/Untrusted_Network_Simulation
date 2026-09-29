# Phase-2 language execution smoke V1

This runner tests **language execution only** on 58 real development PRE contexts. It does not publish a game event, use a later vote/exile or private suspicion report, run the final ToM predictor, or measure game effectiveness. The action and language contracts remain unchanged.

## Frozen case selection

The input is the verified `paper-development-qwen35-9b-1500-v1` publication (manifest SHA-256 `396ea0fa3f3f03dd1dfce8b25c0162ca489b992321d0c9e6f772eb2b0c4a2acd`) and its restricted role sidecar (digest `276b6ee182bfb07639ae3ab99a5c8037b52ffb99b37f5c60f7f26a229178a2da`). The sidecar is used only to recover the acting wolf's lawful two-wolf team and legal target set. Eligibility reads each **current** authoritative PRE's alive set, phase, current speaker, public tie and queue. It must reproduce 5,761 candidate PREs and 21,638 candidate rows before sampling. No future public game suffix, `suspicion_support`, or outcome participates in selection.

A case identity is `(game_id, boundary_id, acting_wolf, phase, candidate_j, action)`. Each stratum is sorted by `SHA256(canonical JSON ["phase2-language-smoke-v1:20260929", *identity])`, with identity as a tie break. At most one case is retained from any `(game_id, boundary_id)` across all strata. The fixed draw order and quotas are:

| Order | Phase | Action | Cases |
|---:|---|---|---:|
| 1 | speech_pk | Probe | 8 (all eligible PK Probe rows) |
| 2 | speech | Probe | 10 |
| 3 | speech_pk | Redirect | 10 |
| 4 | speech | Redirect | 10 |
| 5 | speech_pk | Push | 10 |
| 6 | speech | Push | 10 |

If PK Probe is no longer exactly eight or any stratum cannot supply its quota after the global unique-PRE rule, the runner fails **before any model call**. It never replaces or augments cases using language results. The ordered identities, prefix digests, and selection digest are stored in the artifact.

For Redirect only, the runner reads seven observer rows at each selected PRE from the verified sealed Qwen3 OOF evaluation. It reconstructs the full 7×7 Q matrix and calls the **frozen full-development M3 runtime mapper** once for each legal alternative. The Action Contract's selector then takes maximum `p_tilde`, with canonical seat-order tie break. The expected mapper manifest digest is `8ab529972a5722e61e0a81ec37f089677d1276c21cb83921aa842b2ce33995c3`. This use of development OOF Q constructs a legal semantic plan; it is not a new mapper evaluation. The final Qwen3 predictor is not called. Probe uses the frozen `pos(w) < pos(j) < pos(teammate)` continuation and `CURRENT_SUSPICION_BASIS`; the teammate does not act in this study.

## Server prerequisites and commands

Use the existing separate client and vLLM environments. The source checkout must contain this runner and the completed Phase-2 language files. The publication, OOF evaluation, frozen mapper, local model files, deployment YAML, and client runtime YAML must already be present at the paths recorded in the code and existing server configuration. The client needs `openai`, `PyYAML`, NumPy, SciPy, and scikit-learn. The deployment is the existing Qwen3.5-9B local, text-only, thinking-disabled vLLM service (`qwen35-9b`, `127.0.0.1:8000`); no new HTTP client or model download is introduced. Follow [server execution](/Users/name_yuxiao/Desktop/VscodeProjects/Untrusted_Network_Simulation/docs/server-execution.md) for the documented service environment checks.

In terminal 1:

```bash
cd /home/dell/yuxiao/Untrusted_Network_Simulation
source "$(conda info --base)/etc/profile.d/conda.sh"
export FLASHINFER_WORKSPACE_BASE=/data/yuxiao/cache/flashinfer
mkdir -p "$FLASHINFER_WORKSPACE_BASE"
./scripts/start_vllm.sh
```

In terminal 2:

```bash
cd /home/dell/yuxiao/Untrusted_Network_Simulation
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/yuxiao/envs/untrusted-network-simulation
export PYTHONNOUSERSITE=1
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/v1/models
python scripts/run_phase2_language_smoke.py --help
python scripts/run_phase2_language_smoke.py --preflight \
  --publication /data/yuxiao/Untrusted_Network_Simulation/publications/paper-development-qwen35-9b-1500-v1 \
  --evaluation-root /data/yuxiao/Untrusted_Network_Simulation/paper-studies/evaluations/0b8c1d7220aae6fcf577038f8d930c97d6453b7f45616ed0a19c817e7a80b956 \
  --mapper /data/yuxiao/Untrusted_Network_Simulation/paper-studies/mapper-final/paper-phase2-mapper-final-v1
python scripts/run_phase2_language_smoke.py \
  --publication /data/yuxiao/Untrusted_Network_Simulation/publications/paper-development-qwen35-9b-1500-v1 \
  --evaluation-root /data/yuxiao/Untrusted_Network_Simulation/paper-studies/evaluations/0b8c1d7220aae6fcf577038f8d930c97d6453b7f45616ed0a19c817e7a80b956 \
  --mapper /data/yuxiao/Untrusted_Network_Simulation/paper-studies/mapper-final/paper-phase2-mapper-final-v1 \
  --storage-profile configs/server.json
```

`--preflight` validates the publication, selection, OOF Q, and mapper; it prints the selection digest and makes **no LLM calls or artifact**. The full command reuses `collect_games.inspect_inputs`/`live_preflight` and `load_named_backends` for the configured client/deployment identity. It uses the existing `Phase2LanguageActorV1`, `Phase2SemanticPerceiverV1`, and `realize_verify_action`. The perceiver gets only generated text and public PRE context. A failed first attempt receives exactly one frozen repair attempt with the same plan. Backend/transport failure aborts the study rather than being counted as a semantic invalidity or silently retried. No incomplete artifact is published.

## Metrics, gate, and artifact

Overall, by action, and by phase, the artifact reports case count, first-pass valid count/rate, repair attempts/successes, and final valid/invalid count/rate. Invalid-reason frequencies are reported over all failed attempts and over final failures. Phase-specific rates are descriptive only. The hard gate requires at least 53/58 final valid overall, at least 16/20 Push, 16/20 Redirect, and 15/18 Probe final valid. It also fails if no semantics are parsed, all parsed actions have one action identity, or at least 30% of the final perceptions for any action contain a commitment target outside the requested target set. Requested-plan isolation is a structural gate: the runner instantiates only the frozen independent perceiver class, whose `perceive` method receives text and public context, and records that interface boundary. These rates concern language execution, **not gameplay effectiveness**.

The immutable canonical artifact is:

```text
/data/yuxiao/Untrusted_Network_Simulation/paper-studies/language-smoke/paper-phase2-language-smoke-v1
```

It contains `manifest.json`, `selected_cases.json`, `case_executions.jsonl`, and `metrics.json`. The manifest records source commit and exact source hashes (including local worktree status), model/deployment identity, publication/OOF/mapper lineage, contract versions, selection rule/digest, and semantic-boundary declarations. Case executions contain requested plans, every generated text and digest, independent perceived semantics, verification reasons, and repair audit. The restricted sidecar itself, role assignment, future outcome, and canonical private reports are not serialized. A destination that already exists is rejected before model calls; use a separate reviewed study identity for any later rerun rather than overwriting this one.

Bring back the terminal's selection digest, printed aggregate metrics and gate, artifact path and manifest digest, plus `manifest.json`, `selected_cases.json`, `metrics.json`, and `case_executions.jsonl` for review. A failed gate permits only prompt, JSON/schema transport, or parser/serialization repairs within the language layer. It does not authorize changes to the Action Contract, Probe eligibility, Redirect selector, or mapper.
