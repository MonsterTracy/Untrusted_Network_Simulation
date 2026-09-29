# Selected M3 full-development OOF mapper

The selected development study is the immutable artifact `paper-phase2-mapper-development-v1` with manifest digest `7c44ea21e202d481a91c4e8d0a8695020fee90d42eff7030e62f2dab8a9e546c`, source commit `94fa47bfc127822b90e3f6ad908362bc9ab1e9a9`, and primary specification `M3_r2_additive_logistic`. The final-fit runner checks these identities, the full promotion path, the fixed population, and unchanged frozen offline/mapper source hashes before fitting. It then uses only the 21,638 development candidate rows derived from sealed Qwen3 OOF Q, with 5,412 positive `Theta_AC` labels. All preprocessing, knots, and coefficients are fit on these full development rows via the existing `fit_full_development` function. There is no new model comparison or hyperparameter search.

The final artifact is an **uncalibrated full-development mapper trained on OOF-Q features**. Runtime inference returns `p_tilde`; it is not `p_hat`, an independently validated posterior, or a deployment-calibrated probability. Runtime Q comes from the frozen final Qwen3 predictor at a current PRE, not from in-sample development Q. This runbook does not attach the mapper to gameplay.

The runtime API validates a full 7×7 non-self Q simplex, uses the acting wolf's known two-wolf team to form alive non-wolf observer panels, and derives remaining speakers from the supplied public queue. It calls the frozen `r2_input` and `FoldPreprocessor.transform` implementations directly. The frozen entropy `m=1` convention is retained and tested; for a legal current wolf-speaker PK, that case is unreachable because the acting wolf is necessarily in the PK competition set.

After committing and pushing the new final-fit/runtime files and updating the server checkout to that commit, execute:

```bash
cd /home/dell/yuxiao/Untrusted_Network_Simulation
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/yuxiao/envs/untrusted-network-simulation
git status
git rev-parse HEAD
git branch --show-current
python -c 'import numpy, scipy, sklearn; print("numpy", numpy.__version__, "scipy", scipy.__version__, "sklearn", sklearn.__version__)'
python scripts/run_phase2_mapper_final_fit.py --storage-profile configs/server.json --preflight
python scripts/run_phase2_mapper_final_fit.py --storage-profile configs/server.json
```

If `scikit-learn` alone is absent while NumPy/SciPy are present, install only the mapper optional dependency with `python -m pip install --no-deps 'scikit-learn>=1.4,<2.0'`, then rerun the import check.

The runner will publish one immutable artifact at:

```text
/data/yuxiao/Untrusted_Network_Simulation/paper-studies/mapper-final/paper-phase2-mapper-final-v1
```

Inspect and verify after a successful fit:

```bash
FINAL=/data/yuxiao/Untrusted_Network_Simulation/paper-studies/mapper-final/paper-phase2-mapper-final-v1
python -m json.tool "$FINAL/training_summary.json"
python - <<'PY'
from pathlib import Path
from werewolf.artifact_io import verify_artifact
from werewolf.phase2_mapper_runtime import load_runtime_mapper
root = Path('/data/yuxiao/Untrusted_Network_Simulation/paper-studies/mapper-final/paper-phase2-mapper-final-v1')
artifact = verify_artifact(root, expected_artifact_type='phase2_full_development_oof_mapper', expected_schema_version='phase2_full_development_oof_mapper_v1')
runtime = load_runtime_mapper(root, expected_manifest_digest=artifact.manifest_digest)
print('artifact_path:', artifact.path)
print('manifest_digest:', artifact.manifest_digest)
print('model_digest:', runtime.model_digest)
print('training:', artifact.manifest['training'])
print('selected_study:', artifact.manifest['selected_development_study'])
PY
```

Bring back the artifact path, printed manifest and model digests, `manifest.json`, `model.json`, and `training_summary.json`. Keep the immutable directory intact. An independent calibration set is required before any future `p_hat` is defined.
