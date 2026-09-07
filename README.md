# Classic7 observer-conditioned ToM

This repository owns canonical scientific game generation and the public-only
Theory-of-Mind mainline for seven-player, multi-day text Werewolf:
two Werewolves, three Villagers, one Seer and one Witch, including Night0.

`CONTEXT.md`, ADRs 0001–0021 and
[the Phase-1 specification](docs/specs/phase-1-classic7-tom-mainline.md)
are authoritative. Historical artifacts belong to the old repository.

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
classic7-tom --help
classic7-tom collect --plan plan.json --runtime-config runtime.yaml --call-limit 10000 --destination artifacts/collection
classic7-tom publish-development --collection artifacts/collection --runtime-config runtime.yaml --destination artifacts/publications/development
classic7-tom prepare-experiment --publication artifacts/publications/development --protocol protocol.json --destination artifacts/experiments/experiment
classic7-tom run-development-oof --experiment artifacts/experiments/experiment
classic7-tom validate-artifact artifacts/experiments/experiment
```

The call limit above is an example runtime value, not a scientific constant.
The Collection Plan must bind its actual value and the normalized runtime
configuration digest in `environment_provenance`. Plan records are constructed
with `construct_collection_plan`; every ExperimentConfig field must be
explicitly declared. See [configuration](configs/README.md).

No network calls or large-scale collection are part of the test suite.
Formal training runs in fresh processes, uses complete seven-shift cycles and
only deterministic step-boundary recovery. Once all ten checkpoints are sealed,
training/recovery is permanently closed for that experiment identity.
Held-out evaluation never selects a checkpoint.

## Validation

```sh
conda run -n 3wd python -m pytest -q
conda run -n 3wd python -m compileall -q run_random.py werewolf tests
git diff --check
```

[Architecture](docs/architecture.md), [collection contract](docs/collection_contract.md),
[ToM contract](docs/twd_tom_contract.md), and
[module ownership](docs/repository_structure.md) describe the live implementation.
Game agents exist only to generate canonical evidence; trained ToM outputs never
control gameplay. Phase-1 acceptance is contract-complete small-scale execution,
not model quality or a paper-level scientific conclusion.
