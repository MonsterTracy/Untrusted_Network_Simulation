# Project module ownership and Phase-1 responsibilities

This map locates current implementations and operator entrypoints. See the
[research overview](research-overview.md) for study status and evidence, and the
[README](../README.md) for installation and execution commands.

## Current ownership

| Responsibility | Existing implementation and scope |
| --- | --- |
| Game Runtime | [game_runtime.py](../werewolf/game_runtime.py), [envs](../werewolf/envs/), [agents](../werewolf/agents/), [backends](../werewolf/backends/) own Classic7 assembly, legal observations and authoritative execution. Root [run_random.py](../run_random.py) retains thin exports. |
| Canonical Collection | [canonical_collection](../werewolf/canonical_collection/), [Publication](../werewolf/development_publication.py), [V2 Publication](../werewolf/development_publication_v2.py) and [artifact_io](../werewolf/artifact_io/) own generated data, provenance and immutable evidence. |
| Observer-ToM | [tom](../werewolf/tom/) owns observer suspicion prediction; [tom_llm](../werewolf/tom_llm/) owns the corresponding general-model Baseline comparison. |
| Observer-DTRS | [observer_dtrs](../werewolf/observer_dtrs/) owns conditional event estimation, cognitive risks and three-way judgments. Its evidence does not itself define language-action utility. |
| Strategy | [strategy](../werewolf/strategy/) owns reusable candidate and planning research implementations. The retained MCTS prototype is not the current paper's strategy method. |
| Speech | [Constrained speech](../werewolf/speech/constrained.py) checks public eligibility. [Realization](../werewolf/speech/realization.py) verifies one V1 speech action. The [commit workflow](../werewolf/speech/commit_workflow.py) uses the [canonical speech binding](../werewolf/speech/verified_commit.py) and the existing recorder transaction. |
| Phase-2 | [interventions](../werewolf/interventions/) owns shared actions and evidence, [Mapper](../werewolf/interventions/mapper/), [Language](../werewolf/interventions/language/), [Online](../werewolf/interventions/online/), and Terminal statistics. Gameplay and shared modules sit at the package root. See the [Phase-2 navigation](research/phase2-current-status.md) for execution entrypoints and Source inventories. |
| Evaluation | [ToM reporting](../werewolf/tom/reporting.py), [Baseline analysis](../werewolf/tom_llm/baseline_analysis.py), [Terminal statistics](../werewolf/interventions/terminal_estimator.py) and [Gameplay results](../werewolf/interventions/gameplay.py) retain their distinct populations and endpoints. |
| scripts | [scripts](../scripts/) supplies operator and analysis commands, experiment setup and the study-specific support modules listed below. |

## Operator and research entrypoints

Under `scripts/`, 50 Python files have executable entry guards, four provide
callable or compatibility code, and two initialize packages. The Shell launcher
is counted separately. These counts describe source files. The single installed
console entry is `uns=werewolf.cli:main`. The module command
`python -m werewolf.cli` uses the same implementation.

| Responsibility | Representative implementation and usage boundary |
| --- | --- |
| Collection and service operation | [collect_games.py](../scripts/collect_games.py) freezes inputs and coordinates the existing `uns collect` path. [Read-only inspection](../scripts/inspect_canonical_collection_readonly.py) examines collection metadata. [start_vllm.sh](../scripts/start_vllm.sh) activates the registered serving environment and execs the deployment YAML; its server paths are not portable installation defaults. See [data operations](research/data-acquisition-preprocessing.md) and [deployment](server-execution.md#local-qwen-service). |
| ToM training and inference lifecycle | CLI Development/OOF and [Final Fit](qwen3-final-fit.md) commands own scientific execution. [Capsule compilation](../scripts/build_final_tom_capsule.py), [CPU qualification](../scripts/qualify_final_tom_capsule.py) and [Resident serving](../scripts/serve_final_tom_capsule.py) have separate responsibilities and explicit identities; the service script is an Owner-launched process, not a user-facing Gameplay Runner. |
| DTRS offline tools | [Dataset management](../scripts/manage_observer_dtrs_dataset.py), [D2 protocol/split management](../scripts/manage_observer_dtrs_d2.py), [nested validation](../scripts/run_observer_dtrs_validation.py) and diagnostic/inspection tools consume the documented offline populations. See [D2](research/observer-dtrs-v1-d2a-protocol.md) and [D3](research/observer-dtrs-v1-d3a-protocol.md). |
| Phase-2 operations | [Mapper study](../scripts/run_phase2_mapper_study.py), [Mapper final fit](../scripts/run_phase2_mapper_final_fit.py), [language Smoke](../scripts/run_phase2_language_smoke.py), [Online campaign](../scripts/phase2_online_campaign.py) and [Gameplay campaign](../scripts/phase2_gameplay_campaign.py) retain source/configuration and artifact requirements. The [Phase-2 navigation](research/phase2-current-status.md) distinguishes Pilot and Gameplay endpoints. |
| Historical statistics | [OOF metrics](../scripts/paper_tom_metrics.py), [Backbone contrasts](../scripts/paper_backbone_contrasts.py), [Experiment A analysis](../scripts/experiment_a_mapper_contrasts.py), [Terminal ITT](../scripts/phase2_terminal_itt_analysis.py), [Probe ITT](../scripts/phase2_probe_itt_analysis.py) and [Gameplay analysis](../scripts/analyze_tom_gameplay_ablation.py) read distinct evidence and do not imply interchangeable populations or outcomes. |
| Controlled research demonstrations | [Synthetic comparison](../scripts/compare_observer_dtrs_synthetic.py), [S1a comparison](../scripts/compare_observer_dtrs_s1a.py), [candidate appraisal](../scripts/compare_observer_dtrs_appraisal.py) and controlled check tools use artificial inputs. See the [research scope](research/observer-dtrs-contextual-speech-prototype.md); these are not formal strategy-effect evidence. |
| Development diagnostics | [Contextual DTRS demo](../scripts/diagnostics/dtrs_contextual_demo.py) prints decisions for built-in synthetic cases. [Phase-2 pilot support](../scripts/diagnostics/phase2_pilot_support.py) summarizes an existing dataset JSON. Both print to stdout and can be run as modules in `scripts.diagnostics`. |
| Result display | [Matrix tables](../scripts/show_tom_game_matrices.py) and [single-game visualization](../scripts/visualize_tom_single_game.py) consume existing exports through the shared [review reader](../werewolf/tom/single_game_review.py). The [table guide](tom-game-matrices-cli.md) and [visualization guide](tom-single-game-visualization.md) explain which evidence each command can display. |
| Historical transport adapter | [remote_tom_worker.py](../scripts/remote_tom_worker.py) checks a fixed detached planner Source and its adapter bytes before loading. The [Remote-ToM protocol](research/phase4bi-remote-tom-protocol.md) describes this historical deployment design, not a current-HEAD quick start. |

Non-CLI files under `scripts/` have the following current import roles.

| Files | Role and consumers |
| --- | --- |
| [__init__.py](../scripts/__init__.py) | Installable operator-package boundary; importing it starts no runtime. |
| [diagnostics/__init__.py](../scripts/diagnostics/__init__.py) | Package marker for the local diagnostic commands. |
| [phase2_integration_seam.py](../scripts/phase2_integration_seam.py) | Thin exports listed in the Pilot Source inventory. [language/execution.py](../werewolf/interventions/language/execution.py) owns speech preparation and calls the verifier. |
| [remote_tom_protocol.py](../scripts/remote_tom_protocol.py), [remote_tom_provider.py](../scripts/remote_tom_provider.py) | Shared wire validation and injected Gameplay-side provider; neither is a model-service launcher. |
| [tom_gameplay_ablation.py](../scripts/tom_gameplay_ablation.py) | Callable experiment integration used by the existing paired collection/smoke operators; preserves their original scientific contract. |

## Installation and checkout scope

See the [README installation table](../README.md#install-and-execute) for base and
optional dependencies. `setup.py` declares Python >=3.10; CI selects 3.12, while
the existing serving and Baseline environment specifications describe particular
Linux/Python/CUDA combinations, not universal client-environment compatibility.
The vLLM environment remains separate from the scientific client.

These instructions target an editable source checkout. Configurations, protocols
and source inventories are read from repository paths; installation does not
provide model weights, Publications, sealed predictions or external services.
Formal tools require their actual pinned inputs and execution Source. Historical
documents retain their own dates and do not authorize a new run on current HEAD.

`find_packages()` currently includes `scripts` and some `tests` packages as well
as `werewolf`, including Strategy. The three artificial comparison scripts above
reuse `tests.observer_dtrs.test_speech_runtime` fixtures and are explicitly
repository/development tools. Excluding tests from packaging without addressing
those consumers would change their behavior; no such packaging change is made.
Root `run_random` remains declared in `py_modules` for its thin exports and
Source inventories. No standalone wheel support is established by this map.

## Phase-1 responsibilities

This table retains the Phase-1 implementation and paired final lifecycle scope;
the Runtime location reflects the completed ownership move. It is not a complete
inventory of later research, nor a replacement for historical Source identities.

| Modules | Phase-1 responsibility |
| --- | --- |
| werewolf/game_runtime.py; root run_random.py | Package-owned Runtime assembly, profile assignment and required canonical recorder loop; root file is a thin export, not another loop or CLI |
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
