# Phase-2 mapper development study: server runbook

The committed `scripts/run_phase2_mapper_study.py` is the sole formal entry point. It uses the frozen publication, role sidecar, five game folds, and Qwen3 OOF seal. It requires the source branch `twd/mainline`, a clean tracked worktree, and all Phase-2 implementation/protocol files present in HEAD. Untracked notes are recorded in provenance but do not block the run. The destination is immutable; a second run with the same study name fails before fitting.

The script first verifies Git identity, the storage profile, and the unused destination. It then verifies the OOF provenance and builds the frozen offline layer. Before fitting any mapper, it checks 1,500 games, 5,761 candidate PREs, 21,638 candidate rows, 5,412 positive `Theta_AC` rows, 3,317 post-day states, and 74 skipped no-candidate PK PREs. It fits only M0–M3 in the existing five whole-game folds, audits all 21,638 held-out predictions per model, computes the frozen metrics, performs 100,000 paired fold-stratified game-cluster bootstrap replicates with seed `20260929`, and applies the frozen promotion/rescue rules. The runner never refits within a bootstrap replicate.

After the user commits/pushes these files and updates the server checkout to that commit, run:

```bash
cd /home/dell/yuxiao/Untrusted_Network_Simulation
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/yuxiao/envs/untrusted-network-simulation
git status
git rev-parse HEAD
git branch --show-current
python -c 'import numpy, scipy, sklearn; print("numpy", numpy.__version__, "scipy", scipy.__version__, "sklearn", sklearn.__version__)'
python scripts/run_phase2_mapper_study.py --storage-profile configs/server.json --preflight
python scripts/run_phase2_mapper_study.py --storage-profile configs/server.json
```

If the environment is missing the optional mapper dependency, install only that dependency, then rerun the import check:

```bash
python -m pip install --no-deps 'scikit-learn>=1.4,<2.0'
```

The immutable output path is:

```text
/data/yuxiao/Untrusted_Network_Simulation/paper-studies/mapper-development/paper-phase2-mapper-development-v1
```

The runner prints the path and manifest digest after successful publication. Inspect and validate the result with:

```bash
STUDY=/data/yuxiao/Untrusted_Network_Simulation/paper-studies/mapper-development/paper-phase2-mapper-development-v1
sed -n '1,160p' "$STUDY/report.md"
python -m json.tool "$STUDY/selection.json"
python -m json.tool "$STUDY/metrics.json"
python - <<'PY'
from pathlib import Path
from werewolf.artifact_io import verify_artifact
root = Path('/data/yuxiao/Untrusted_Network_Simulation/paper-studies/mapper-development/paper-phase2-mapper-development-v1')
artifact = verify_artifact(root, expected_artifact_type='phase2_mapper_development_study', expected_schema_version='phase2_mapper_development_study_v1')
print('manifest_digest:', artifact.manifest_digest)
print('source_commit:', artifact.manifest['source']['commit'])
print('population:', artifact.manifest['population'])
print('selection:', artifact.manifest['selection'])
PY
```

Bring back the printed manifest digest, `report.md`, `manifest.json`, `metrics.json`, `bootstrap.json`, and `selection.json`. Retain `fit_manifests.json` and all four `cross_fitted_predictions/*.jsonl` files in the immutable study directory for audit. This is development model comparison, not independent calibration or final testing.
