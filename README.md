# Classic7 observer-conditioned ToM

This repository owns canonical scientific game generation and the public-only
Theory-of-Mind mainline for seven-player, multi-day text Werewolf:
two Werewolves, three Villagers, one Seer and one Witch, including Night0.

`CONTEXT.md`, ADRs 0001–0021 and
[the Phase-1 specification](docs/specs/phase-1-classic7-tom-mainline.md)
are authoritative. This repository owns the current scientific mainline.

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
Game agents exist only to generate canonical evidence; trained ToM outputs never
control gameplay. Phase-1 acceptance is contract-complete small-scale execution,
not model quality or a paper-level scientific conclusion.

## Final fit and independent evaluation

The [paired final lifecycle](docs/final-lifecycle.md) trains both conditions on
all development games with the frozen cycle budget, seals their terminal
checkpoints, and only then consumes an independent final publication.
Development OOF remains available for qualification and analysis.
