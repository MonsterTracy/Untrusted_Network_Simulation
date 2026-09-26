# Qwen3 all-development final fit

This is a separate, single-terminal lifecycle for the completed backbone study.
It trains the full observer-conditioned Qwen3 graph with explicit day/phase codes
on the Primary rows of all 1500 development games. It does not use a fold's
training or held-out partition. The Qwen3 fold-0 **untrained initial tensor
artifact** is the sole initialization source; its fold number is provenance only.

The immutable `qwen3_final_manifest.json` binds the exact prepared backbone
study and initial-state digests, development publication and Primary sidecar,
Qwen3 graph, clean source revision, implementation/runtime identity, a derived
day capacity of 256, and a new all-development seven-shift schedule. The
schedule contains 3 cycles × 7 rotations × 1500 games at one game per batch:
31500 optimizer steps. AdamW, constant learning rate, and game-balanced cross
entropy use the frozen backbone reference controls. No validation partition,
checkpoint selection, early stop, or gameplay outcome enters this identity.

`runs/<fit-digest>/recovery/<step>/` contains canonical model, optimizer, RNG
and loss artifacts at each 1000-step boundary and at step 31500. Recovery
requires an explicit `--resume` and verifies the complete preceding chain.
`runs/<fit-digest>/terminal/` binds only the exact step-31500 recovery and its
ancestry. `runs/<fit-digest>/seal/` binds that one terminal and closes training.
No command searches for a latest checkpoint or substitutes another lineage.

With `UNS_STORAGE_PROFILE=configs/server.json`, supply the exact prepared-study
path and manifest digest from the completed backbone study:

```sh
uns prepare-qwen3-final-fit \
  --study "$PREPARED_STUDY_PATH" \
  --study-digest PREPARED_STUDY_MANIFEST_DIGEST \
  --publication "$DEVELOPMENT_PUBLICATION_PATH" \
  --destination QWEN3_FINAL_ID
uns run-qwen3-final-fit --experiment QWEN3_FINAL_ID
uns seal-qwen3-final-fit --experiment QWEN3_FINAL_ID
uns validate-qwen3-final-fit --experiment QWEN3_FINAL_ID
```

Set `PREPARED_STUDY_PATH` and `DEVELOPMENT_PUBLICATION_PATH` to the exact existing
absolute paths under the artifact root; the digest token is also a placeholder,
not an automatic discovery rule.
If a run was
interrupted after a complete recovery point, repeat `run-qwen3-final-fit` with
`--resume`. A semantic mismatch fails closed.

Python consumers open the fit explicitly and construct
`SealedQwen3Predictor(open_fit(path))`. `log_probabilities(prefix)` returns a
7×7 tensor; `predict(prefix, observer)` returns one seven-seat probability
vector. The predictor requires the seal and exact terminal, validates the full
public PRE and capacity. Opening the fit validates the training publication and
Primary sidecar; model forward reads no belief label or role truth.
It has no gameplay integration. The older paired final fit, backbone OOF,
evaluation and contrasts retain their own identities and contracts.
