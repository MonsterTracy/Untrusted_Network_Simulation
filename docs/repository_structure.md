# Phase-1 module responsibilities

This document covers the Phase-1 implementation and paired final lifecycle,
not every module in the current project. See the [README](../README.md) for the
current development entry and implemented research-module navigation, including
TomBaseline, Observer-DTRS and later research extensions.

| Modules | Phase-1 responsibility |
| --- | --- |
| run_random.py | Runtime assembly, profile assignment and required canonical recorder loop; no CLI |
| werewolf/envs | Classic7 mechanics and authoritative public events |
| werewolf/agents | Playing-agent cognition, constrained gameplay transport, role-legal observation prompts |
| werewolf/backends + runtime_config.py | One named OpenAI-compatible transport/configuration contract |
| werewolf/speech | V1 semantic perception and playing-agent PRE reports/validation |
| werewolf/helper | Per-call runtime audit logging |
| werewolf/canonical_collection | Ledger, immutable PRE/V1/evidence contracts, bundle/failure publication and deterministic replay |
| werewolf/artifact_io | Canonical JSON/raw tensor containers and atomic immutable artifact publication |
| werewolf/structured_history.py | One pure semantic token planner |
| werewolf/development_publication.py | Plan-closed publication, whole-game folds and restricted sidecar |
| werewolf/tom/dataset.py | Population/private-blind tensorization, target conversion and cyclic permutation |
| werewolf/tom/population.py | Named eligibility construction/loaders; role truth terminates here |
| werewolf/tom/model.py + temporal.py | Public Qwen2 graph and non-trainable canonical temporal artifacts |
| werewolf/tom/experiment.py + protocol.py + state.py | Frozen protocol, schedules, bootstrap indices and canonical state serialization |
| werewolf/tom/training.py + run_records.py | Primary fixed-budget training, recovery, terminal states and checkpoint-set seal |
| werewolf/tom/evaluation.py + scoring.py + reporting.py | Pure held-out prediction, named metrics and paired OOF reporting |
| werewolf/final_publication.py | Independent, seal-bound final publication without folds |
| werewolf/tom/final_capacity.py + final_experiment.py | Validator-derived frozen capacity and all-development final input identity |
| werewolf/tom/final_training.py | Paired final orchestration over shared optimizer steps, deterministic recovery and two-model seal |
| werewolf/tom/final_evaluation.py | Sealed label-free prediction, immutable final consumption and shared-math reporting |
| werewolf/cli.py | Development and final lifecycle commands; also available as python -m werewolf.cli |
| Package __init__.py files | Package boundaries and current explicit exports |

The gameplay discussion intent transport is not a ToM action selector or V2
perception path: it belongs only to the playing agent that generates raw speech.
Call-audit fields named fallback_used/fallback_action_count remain rejection
evidence: canonical validation requires false/zero. They expose no fallback
execution branch.

The following records Phase-1 reconstruction history; it is not an instruction
to remove retained research modules, branches or worktrees.

The former script/twd_tom runners, models/twd_tom schemas/tensorizers/models,
standalone materializers, agent registry aliases, random rollout CLI, historical
backend/schema conversion and obsolete server/evaluation documents are removed.
No archive or compatibility namespace replaces them.
