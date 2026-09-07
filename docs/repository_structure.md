# Live module responsibilities

Every production Python module belongs to one of these current responsibilities:

| Modules | Current justification |
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
| werewolf/cli.py | Five formal commands; also available as python -m werewolf.cli |
| Package __init__.py files | Package boundaries and current explicit exports |

The gameplay discussion intent transport is not a ToM action selector or V2
perception path: it belongs only to the playing agent that generates raw speech.
Call-audit fields named fallback_used/fallback_action_count remain rejection
evidence: canonical validation requires false/zero. They expose no fallback
execution branch.

The former script/twd_tom runners, models/twd_tom schemas/tensorizers/models,
standalone materializers, agent registry aliases, random rollout CLI, historical
backend/schema conversion and obsolete server/evaluation documents are removed.
No archive or compatibility namespace replaces them.
