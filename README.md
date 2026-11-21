# Classic7 observer-conditioned ToM

Data Acquisition & Preprocessing: [technical lifecycle and operation](docs/research/data-acquisition-preprocessing.md) · [historical branch registry](docs/BRANCHES.md).

This repository owns canonical scientific game generation and the public-only
Theory-of-Mind mainline for seven-player, multi-day text Werewolf:
two Werewolves, three Villagers, one Seer and one Witch, including Night0.

The research goal extends from observer cognition to DTRS-guided strategy,
verified language/actions and full-game evaluation. Module connections and
research effects require their own evidence; the implemented paths are described
in the [research overview](docs/research-overview.md).

`CONTEXT.md`, ADRs 0001–0022 and
[the Phase-1 specification](docs/specs/phase-1-classic7-tom-mainline.md)
remain authoritative for the Phase-1 scientific definitions and contracts.
Later research modules use their explicitly approved protocols; a research draft
or an implementation alone does not establish a formal method.

## Development and documentation

Development follows `twd/mainline`. See [AGENTS.md](AGENTS.md) and the
[development guidelines](docs/development-guidelines.md) for collaboration,
checkout locations and research-asset protection. Historical Source identities
remain attached to their original experiments.

| Document role | Entry and scope |
|---|---|
| Current navigation | This README links implemented research modules; it is not a live experiment-status register. |
| Research architecture | [Research overview](docs/research-overview.md): module responsibilities, data flow and evidence boundaries. |
| Current TomBaseline operation | [Unified operations](docs/operations/tom-llm-unified.md); `uns tom-llm` is the recommended formal operator entry. |
| Frozen scientific contracts | `CONTEXT.md`, approved ADRs and the protocol explicitly bound to each experiment. |
| Historical provenance | [Branch/worktree registry](docs/BRANCHES.md), read at its stated snapshot date; completed runs retain their original Source and Artifact identities. |
| Research documents | Module-specific protocols, drafts and implementation/evidence notes have different authority; use their stated approval, date and evidence limits. |

TomBaseline uses only `uns tom-llm` for preparation, execution, recovery and
three-run analysis. Superseded script entrypoints and the two-round physical
qualification pipeline have been removed. Historical artifacts retain their
original Source and protocol identities.

## Implemented research modules

Functional integration baseline before this documentation update (2026-10-24):
`29fba64f64b3ec310d3f12ecd4247ff46879ec54`. This is a historical snapshot,
not a permanently current HEAD. `twd/mainline` is the ongoing research development
line; `main` remains the stable-release branch. See the
[registry](docs/BRANCHES.md) for retained sources and pending work.

| Module | Implemented source and reference |
|---|---|
| Game Runtime | Classic7 environment, agents, backends and the authoritative game loop: [game_runtime.py](werewolf/game_runtime.py), [environment](werewolf/envs/), [agents](werewolf/agents/), [backends](werewolf/backends/) |
| Canonical Data | Collection/evidence audit, whole-game split, V1/V2 publication and release: [Data lifecycle](docs/research/data-acquisition-preprocessing.md) |
| Observer-conditioned ToM | Public-only non-self suspicion modeling, training/recovery, OOF and sealed Final Fit lifecycles: [ToM contract](docs/tom_contract.md), [Qwen3 Final Fit](docs/qwen3-final-fit.md) |
| ToM LLM Baselines | Qwen3/Yi/Mistral via `uns tom-llm`: one shared set-task execution path, full-input CPU budget, owned service, durable recovery and sealed paired reports: [operations and current limits](docs/operations/tom-llm-unified.md), [YAML example](configs/paper/tom-llm-operations.example.yaml) |
| Observer-DTRS D1–D4 | PRE features and conditional event estimation, `LossSpec`, conditional risks, derived thresholds and POS/BND/NEG; Capsule/Broker/Resident Service/Shadow/QUAL engineering: [D2](docs/research/observer-dtrs-v1-d2a-protocol.md), [D3](docs/research/observer-dtrs-v1-d3a-protocol.md), [D4](docs/research/observer-dtrs-v1-d4c2b2-resident-service.md) |
| Strategy | Reusable candidate plans, counterfactual continuations and research scoring: [werewolf/strategy](werewolf/strategy/). The finite-response-tree UCT prototype is retained research, not the current paper's planning method. |
| Phase-2 | Action/opportunity contracts, Mapper, verified language, canonical commit and Terminal/Probe/fixed-Router experiments: [module navigation and evidence limits](docs/research/phase2-current-status.md) |
| Contextual speech research | Opt-in contextual losses and strategy intents through existing Actor/Parser/verified speech, with three-condition synthetic comparisons: [prototype and evidence limits](docs/research/observer-dtrs-contextual-speech-prototype.md) |
| Experiment A | Paired Qwen3 temporal conditions, continuous/competition-top1 Q and fixed-M3 development contrasts: [temporal contract](docs/adr/0005-isolate-direct-day-phase-features-in-temporal-ablation.md), [paired preparation](werewolf/tom/backbone_study.py), [mapper runner](scripts/run_phase2_mapper_study.py), [statistical consumer](scripts/experiment_a_mapper_contrasts.py) |
| Evaluation and entrypoints | ToM/Baseline metrics, ITT and full-game outcomes answer different research questions. The [module and entrypoint map](docs/repository_structure.md) locates operator commands and shared implementations. |

Strategy and Game Runtime ownership consolidation is complete. Unified Final-D2
EQ1/EQ0 candidates have been fitted, published and read back; applicability to
Final-Q still awaits Development migration evaluation, which has not run.
See the [research status](docs/research-overview.md#4-各研究阶段的当前状态)
for evidence and historical Sources. Existing gameplay code does not establish
completed strategy research or improved full-game win rates.

These are source capabilities, not a claim that a frozen real model or formal
experiment is admitted on this checkout. Runtime/source, data partition and
artifact guards still apply. Synthetic CPU/controlled-generator tests establish
contracts and transport behavior, not LLM quality, information gain or win-rate
improvement. The speech prototype is not a frozen experiment or default game
policy. Experiment A retains Execution Source
`39f4ea25906808bf1ed28b05fdfb63865cc3ffb3` and Analysis Source
`13ca742d7d98e41d5f68e5cb98ad5901f017115c` for the original study.

The ToM LLM entry prepares the frozen set cohort and full-input tokenizer budget,
then runs each model on one owned service instance with durable recovery.
It reuses existing assets and environments; download and installation are not
implemented. CPU/Mock tests do not establish completion of the 10,714-request
formal run. See [the operations guide](docs/operations/tom-llm-unified.md).

Mapper/Experiment A tools and their tests additionally use the `mapper` optional
dependency group in [setup.py](setup.py). The default CLI and collection flow
below remain unchanged.

## Canonical data and Phase-1 ToM path

This is the offline data/model lifecycle. Opt-in strategy experiments reuse the
game Runtime and Canonical evidence path; their execution and endpoints are
described in the [Phase-2 navigation](docs/research/phase2-current-status.md).

```text
Game Runtime → Canonical Collection → immutable Game Bundles + Attempt Ledger
→ Development Publication + whole-game folds + restricted Role Sidecar
→ public-only Dataset → named eligibility operations → immutable Experiment
→ paired Primary training → all-ten checkpoint seal
→ held-out Primary / All-Alive evaluation → game-macro paired-bootstrap reports
```

Collection alone constructs complete cumulative PRE prefixes, ending in
`turn_start(current_speaker)` before that speech, and V1 speech annotations.
Successful empty belief reports are observed uniform distributions over the
six non-self seats. Failed alive-observer reports exclude the whole game.

Primary Development OOF predicts realized beliefs of alive non-wolf observers;
Seer/Witch private cognition can affect those labels. Role truth ends at the
Population Selector; neither Dataset nor model receives it. All-Alive
Identifiability Stress is evaluation-only population expansion on the same
Primary-trained checkpoints, not a second training population.

## Install and execute

[setup.py](setup.py) declares Python **>=3.10**. The [Linux CI configuration](.github/workflows/ci.yml)
selects Python 3.12; local CLI/help and small CPU contract checks have also run
on Python 3.12. This is not a claim that every supported Python version or
GPU environment has passed the full suite.

Run editable installation from the repository root in the intended Python
environment. Base installation provides the CLI and common runtime dependencies:

```sh
python -m pip install -e .
uns --help
```

`python -m werewolf.cli --help` uses the same CLI. Optional groups are additive:

| Group | Declared dependencies and purpose |
|---|---|
| `tom` | PyTorch and Transformers for ToM training, prediction and tokenizer/model tools; does not include model weights. |
| `mapper` | scikit-learn for Mapper fitting and related analysis/tests. |
| `llm-reasoner` | Jinja2 for TomBaseline templates and Matplotlib for plots. This group does not install a language-model serving environment. |
| `dev` | pytest for repository tests. |

For example, the existing ToM development installation is:

```sh
python -m pip install -e ".[tom,dev]"
```

CI installs the combined `tom,mapper,llm-reasoner,dev` groups. Select the groups
needed by the task; installation alone does not provision Publications, model
weights, a Capsule, credentials or external services. These are source-checkout
instructions, not verification of a new installation or a standalone wheel.

The following existing lifecycle recipe requires a real Publication, an explicit
storage profile and the selected protocol; it is not a self-contained quick start:

```sh
export UNS_STORAGE_PROFILE="$PWD/configs/server.json"
uns prepare-experiment --publication DEV_ID --protocol configs/formal/development-experiment-v1/protocol.json --destination EXP_ID
uns validate-artifact experiments/EXP_ID/experiments/experiment
uns run-development-oof --experiment EXP_ID
```

The formal protocol is frozen; preparation still requires a valid real publication.
See [server execution and storage](docs/server-execution.md) for installation,
collection, identity resolution and explicit resume. No server paths belong
in the scientific protocol. See [configuration](configs/README.md).

Real collection and language execution need a configured backend service;
credentials are supplied through environment variables as shown by
[.env.example](.env.example). ToM training follows the protocol's explicit device
and runtime. Live vLLM Baselines require their registered Linux/GPU environment
and model weights. The serving environment is separate from the scientific
client; see the existing
[environment and deployment instructions](docs/server-execution.md#local-qwen-service).
Final-ToM Capsule CPU inference uses matching assets and runtime. The qualified
resident client also verifies qualification evidence and execution scope.

## Entrypoint navigation

| Task | Representative entry and documentation |
|---|---|
| Formal CLI | `uns` / `python -m werewolf.cli`; `uns tom-llm` uses the [current Baseline operations](docs/operations/tom-llm-unified.md). |
| Game collection and runtime | [collect_games.py](scripts/collect_games.py) prepares/operates collection through `uns collect`; see the [data lifecycle](docs/research/data-acquisition-preprocessing.md). [start_vllm.sh](scripts/start_vllm.sh) is a server-specific service launcher, not a general local quick start. |
| ToM model lifecycle | `uns run-development-oof`, `uns run-qwen3-final-fit`; see [Final Fit](docs/qwen3-final-fit.md). [Capsule build](scripts/build_final_tom_capsule.py) and [CPU qualification](scripts/qualify_final_tom_capsule.py) have explicit source/input requirements. |
| DTRS offline analysis and validation | [D2 protocol tools](scripts/manage_observer_dtrs_d2.py), [nested validation](scripts/run_observer_dtrs_validation.py); see the [D2 protocol](docs/research/observer-dtrs-v1-d2a-protocol.md). |
| Phase-2 experiment operation | [Online campaign](scripts/phase2_online_campaign.py), [Gameplay campaign](scripts/phase2_gameplay_campaign.py); see [Phase-2 navigation](docs/research/phase2-current-status.md). Pilot day-end outcomes and full-game outcomes are distinct. |
| Historical research and statistics | [Backbone contrasts](scripts/paper_backbone_contrasts.py), Terminal/Probe ITT and paired Gameplay analyses retain their own Sources, populations and endpoints; use their linked research protocols. |
| Artificial fixtures and demonstrations | [Synthetic speech comparison](scripts/compare_observer_dtrs_synthetic.py); see its [controlled scope](docs/research/observer-dtrs-contextual-speech-prototype.md). These are not formal effect or win-rate experiments. |
| Visualization and display | [Matrix display guide](docs/tom-game-matrices-cli.md) and [single-game visualization guide](docs/tom-single-game-visualization.md) consume existing exports; they do not run model inference. |

Some `scripts` files are callable libraries, not CLIs. The
[detailed entry and library map](docs/repository_structure.md#operator-and-research-entrypoints)
also identifies checkout-only and historical tools. Run operators from the source
checkout; frozen entrypoints require their original Git Source and evidence,
not unconditional execution on the current HEAD. Root `run_random.py` is a thin
Runtime export, not another game command.

No network calls or large-scale collection are part of the test suite.
Formal training runs in fresh processes, uses complete seven-shift cycles and
only deterministic step-boundary recovery. Once all ten checkpoints are sealed,
training/recovery is permanently closed for that experiment identity.
Held-out evaluation never selects a checkpoint.

## Validation

Synthetic training-capacity validation is available through `uns capacity-check`;
see [server execution](docs/server-execution.md#synthetic-capacity-check).
It is engineering-only, reads no scientific artifacts and changes no protocol.

```sh
python -m pytest -q tests/phase2/test_outcome_and_execution.py
```

This small CPU check covers reference-loss and verified-speech contracts with
controlled inputs; it needs no model weights or external service. The broader
repository checks below require the applicable optional dependencies and do
not establish scientific performance or runtime qualification:

```sh
python -m pytest -q
python -m compileall -q run_random.py werewolf tests
git diff --check
```

[Architecture](docs/architecture.md), [collection contract](docs/collection_contract.md),
[ToM contract](docs/tom_contract.md), and
[module ownership](docs/repository_structure.md) distinguish current ownership
from the retained Phase-1 contracts. Use the research-module table above for
later extensions; the Phase-1 architecture is not a complete current inventory.
In the default canonical-collection path, game agents generate canonical evidence;
trained ToM outputs do not control gameplay. The opt-in research prototypes above
do not change default collection, language or game behavior. Phase-1 acceptance
is contract-complete small-scale execution, not model quality or a paper-level
scientific conclusion.

## Final fit and independent evaluation

The [paired final lifecycle](docs/final-lifecycle.md) trains both conditions on
all development games with the frozen cycle budget, seals their terminal
checkpoints, and only then consumes an independent final publication.
Development OOF remains available for qualification and analysis.
