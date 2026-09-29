# Phase-2 language execution smoke V2

This is a rerun of the same 58 development PRE cases under [Language Execution V1.1](/Users/name_yuxiao/Desktop/VscodeProjects/Untrusted_Network_Simulation/docs/research/phase2-language-execution-v1-1.md). All selection rules, the `phase2-language-smoke-v1:20260929` seed, per-action/phase quotas, unique-PRE constraint, sealed Q/mapper Redirect construction, retry limit, metrics, and gate remain as documented in the [smoke-v1 runbook](/Users/name_yuxiao/Desktop/VscodeProjects/Untrusted_Network_Simulation/docs/research/phase2-language-smoke-runbook.md).

The runner now requires this exact **ordered selection digest** before loading the mapper or calling the model:

```text
9ab58cb73fdee18fee63795ee71df37e5ff00c2c320939c2f48221c3167e192e
```

A mismatch is a hard error; there is no resampling. Smoke V2 publishes to a new immutable destination and never overwrites smoke V1:

```text
/data/yuxiao/Untrusted_Network_Simulation/paper-studies/language-smoke/paper-phase2-language-smoke-v2
```

On the configured server, after updating the source checkout and starting the existing Qwen3.5-9B service using the smoke-v1 runbook, activate the existing client environment and run:

```bash
cd /home/dell/yuxiao/Untrusted_Network_Simulation
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/yuxiao/envs/untrusted-network-simulation
export PYTHONNOUSERSITE=1
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

`--preflight` makes no LLM call or artifact. The formal command checks tracked worktree and index cleanliness and freezes source provenance before study inputs or model calls; untracked files do not block it. The v2 artifact records `phase2_speech_semantic_v1_1` and `phase2_language_smoke_v2`, selected cases/digest, individual attempt audits, and the unchanged gate metrics. Bring back the v2 `manifest.json`, `selected_cases.json`, `case_executions.jsonl`, `metrics.json`, and terminal digest/gate output for side-by-side review with v1. A V2 gate pass is an engineering language-execution result, not gameplay effectiveness evidence.
