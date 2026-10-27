# Classic7 observer-conditioned ToM

Data Acquisition & Preprocessing: [technical lifecycle and operation](docs/research/data-acquisition-preprocessing.md) · [current branch registry](docs/BRANCHES.md).

This repository owns canonical scientific game generation and the public-only
Theory-of-Mind mainline for seven-player, multi-day text Werewolf:
two Werewolves, three Villagers, one Seer and one Witch, including Night0.

`CONTEXT.md`, ADRs 0001–0022 and
[the Phase-1 specification](docs/specs/phase-1-classic7-tom-mainline.md)
are authoritative. This repository owns the current scientific mainline.

## Implemented research modules

Functional integration baseline before this documentation update (2026-10-24):
`29fba64f64b3ec310d3f12ecd4247ff46879ec54`. This is a historical snapshot,
not a permanently current HEAD. `twd/mainline` is the ongoing research development
line; `main` remains the stable-release branch. See the
[registry](docs/BRANCHES.md) for retained sources and pending work.

| Module | Implemented source and reference |
|---|---|
| Canonical Data | Collection/evidence audit, whole-game split, V1/V2 publication and release: [Data lifecycle](docs/research/data-acquisition-preprocessing.md) |
| Observer-conditioned ToM | Public-only non-self suspicion modeling, training/recovery, OOF and sealed Final Fit lifecycles: [ToM contract](docs/tom_contract.md), [Qwen3 Final Fit](docs/qwen3-final-fit.md) |
| ToM LLM Baselines | Qwen3/Yi/Mistral via `uns tom-llm`: independent CPU Tokenizer/Suite/full-150 Budget, guarded service/qualification/execution and sealed paired reports: [operations and current limits](docs/operations/tom-llm-unified.md), [YAML example](configs/paper/tom-llm-operations.example.yaml) |
| Observer-DTRS D1–D4 | PRE features and conditional event estimation, `LossSpec`, conditional risks, derived thresholds and POS/BND/NEG; Capsule/Broker/Resident Service/Shadow/QUAL engineering: [D2](docs/research/observer-dtrs-v1-d2a-protocol.md), [D3](docs/research/observer-dtrs-v1-d3a-protocol.md), [D4](docs/research/observer-dtrs-v1-d4c2b2-resident-service.md) |
| Contextual speech research | Opt-in contextual losses and strategy intents through existing Actor/Parser/verified speech, with three-condition synthetic comparisons: [prototype and evidence limits](docs/research/observer-dtrs-contextual-speech-prototype.md) |
| Experiment A | Paired Qwen3 temporal conditions, continuous/competition-top1 Q and fixed-M3 development contrasts: [temporal contract](docs/adr/0005-isolate-direct-day-phase-features-in-temporal-ablation.md), [paired preparation](werewolf/tom/backbone_study.py), [mapper runner](scripts/run_phase2_mapper_study.py), [statistical consumer](scripts/experiment_a_mapper_contrasts.py) |

These are source capabilities, not a claim that a frozen real model or formal
experiment is admitted on this checkout. Runtime/source, data partition and
artifact guards still apply. Synthetic CPU/controlled-generator tests establish
contracts and transport behavior, not LLM quality, information gain or win-rate
improvement. The speech prototype is not a frozen experiment or default game
policy. Experiment A retains Execution Source
`39f4ea25906808bf1ed28b05fdfb63865cc3ffb3` and Analysis Source
`13ca742d7d98e41d5f68e5cb98ad5901f017115c` for the original study.

The ToM LLM unified entry supports CPU Tokenizer/Suite/full-150 context checks
before weights or a model environment are available, then explicit service,
qualification and formal-experiment operations. Asset download and environment
installation are **not implemented**; the pinned-revision acquisition proposal
and remaining Linux/vLLM pidfd, lock-FD and worker-shutdown checks are listed in
[the operations guide](docs/operations/tom-llm-unified.md). CPU/Mock tests do not
establish physical Admission or completion of the 10,714-request formal run.

Mapper/Experiment A tools and their tests additionally use the `mapper` optional
dependency group in [setup.py](setup.py). The default CLI and collection flow
below remain unchanged.

## One executable path

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

```sh
python -m pip install -e ".[tom,dev]"
uns --help
export UNS_STORAGE_PROFILE="$PWD/configs/server.json"
uns prepare-experiment --publication DEV_ID --protocol configs/formal/development-experiment-v1/protocol.json --destination EXP_ID
uns validate-artifact experiments/EXP_ID/experiments/experiment
uns run-development-oof --experiment EXP_ID
```

The formal protocol is frozen; preparation still requires a valid real publication.
See [server execution and storage](docs/server-execution.md) for installation,
collection, identity resolution and explicit resume. No server paths belong
in the scientific protocol. See [configuration](configs/README.md).

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
python -m pytest -q
python -m compileall -q run_random.py werewolf tests
git diff --check
```

[Architecture](docs/architecture.md), [collection contract](docs/collection_contract.md),
[ToM contract](docs/tom_contract.md), and
[module ownership](docs/repository_structure.md) describe the live implementation.
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
