"""Bind the selected M3 full-development fit to the verified mapper study."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from werewolf.artifact_io import (
    canonical_json_bytes, publish_artifact, sha256_bytes, verify_artifact,
)
from werewolf import phase2_mapper as mapper
from werewolf import phase2_mapper_selection as selection
from werewolf import phase2_offline as offline


FINAL_NAME = "paper-phase2-mapper-final-v1"
FINAL_VERSION = "phase2_full_development_oof_mapper_v1"
STUDY_PATH = Path("/data/yuxiao/Untrusted_Network_Simulation/paper-studies/"
                  "mapper-development/paper-phase2-mapper-development-v1")
STUDY_DIGEST = "7c44ea21e202d481a91c4e8d0a8695020fee90d42eff7030e62f2dab8a9e546c"
STUDY_SOURCE_COMMIT = "94fa47bfc127822b90e3f6ad908362bc9ab1e9a9"
STUDY_VERSION = "phase2_mapper_development_study_v1"
SELECTED_SPEC = mapper.MODEL_SPECS[3]
EXPECTED_POSITIVES = 5412


class FinalMapperError(ValueError):
    """The selected study, full OOF population, or final mapper is inconsistent."""


def verify_selected_study(path: Path | str = STUDY_PATH):
    artifact = verify_artifact(path, expected_artifact_type="phase2_mapper_development_study",
                               expected_schema_version=STUDY_VERSION)
    manifest = artifact.manifest
    if (artifact.manifest_digest != STUDY_DIGEST
            or manifest["source"]["commit"] != STUDY_SOURCE_COMMIT
            or manifest["selection"]["primary_model"] != SELECTED_SPEC.name
            or manifest["selection"]["stopped_at"] is not None
            or manifest["population"] != {
                "games": offline.EXPECTED_GAMES,
                "candidate_pre": offline.EXPECTED_PRE,
                "candidate_rows": offline.EXPECTED_CANDIDATES,
                "theta_positive_rows": EXPECTED_POSITIVES,
                "post_day_states": offline.EXPECTED_POST_DAYS,
                "skipped_no_candidate_pk_pre": offline.EXPECTED_SKIPPED_NO_CANDIDATE_PK_PRE,
            }):
        raise FinalMapperError("selected development study digest, selection, or population mismatch")
    protocol = manifest["protocol"]
    if (protocol["feature_contract_version"] != mapper.FEATURE_CONTRACT_VERSION
            or canonical_json_bytes(protocol["model_specs"]) != canonical_json_bytes(
                [spec.record() for spec in mapper.MODEL_SPECS])
            or protocol["offline_contract_source_sha256"] != sha256_bytes(
                Path(offline.__file__).read_bytes())
            or protocol["mapper_source_sha256"] != sha256_bytes(
                Path(mapper.__file__).read_bytes())
            or protocol["selection_source_sha256"] != sha256_bytes(
                Path(selection.__file__).read_bytes())
            or manifest["inputs"]["publication_digest"] != offline.PUBLICATION_DIGEST
            or manifest["inputs"]["role_sidecar_digest"] != offline.ROLE_SIDECAR_DIGEST
            or manifest["inputs"]["fold_manifest_digest"] != offline.FOLD_MANIFEST_DIGEST
            or manifest["inputs"]["oof"]["evaluation_seal_digest"] != offline.OOF_SEAL_DIGEST):
        raise FinalMapperError("frozen study protocol or input lineage mismatch")
    selection_file = json.loads((artifact.path / "selection.json").read_bytes())
    if selection_file != manifest["selection"]:
        raise FinalMapperError("selected study result file differs from manifest")
    bootstrap = json.loads((artifact.path / "bootstrap.json").read_bytes())
    adjacent = {(pair["complex_model"], pair["simple_model"]): pair
                for pair in bootstrap["pairs"] if pair["role"] == "adjacent"}
    expected = tuple(zip((spec.name for spec in mapper.MODEL_SPECS[1:]),
                         (spec.name for spec in mapper.MODEL_SPECS[:-1]), strict=True))
    if set(adjacent) != set(expected) or any(not adjacent[pair]["development_supported"]
                                           for pair in expected):
        raise FinalMapperError("M3 was not supported along the full promotion path")
    return artifact


def candidate_identity_digest(rows: tuple[offline.CandidateRow, ...]) -> str:
    identities = sorted((row.game_id, row.boundary_id, row.acting_wolf,
                         row.phase, row.candidate_j) for row in rows)
    if len(identities) != len(set(identities)):
        raise FinalMapperError("duplicate full-development candidate identity")
    return sha256_bytes(canonical_json_bytes(identities))


def validate_study_candidate_population(study, rows: tuple[offline.CandidateRow, ...]) -> str:
    """Match every OOF-Q feature/label row to the selected study's held-out rows."""
    expected = {(row.game_id, row.boundary_id, row.acting_wolf,
                 row.phase, row.candidate_j): row.theta_ac for row in rows}
    if len(expected) != len(rows):
        raise FinalMapperError("duplicate full-development candidate identity")
    path = study.path / f"cross_fitted_predictions/{SELECTED_SPEC.name}.jsonl"
    seen = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line)
            identity = tuple(value[field] for field in
                             ("game_id", "boundary_id", "acting_wolf", "phase", "candidate_j"))
            if identity in seen or type(value["theta_ac"]) is not bool:
                raise FinalMapperError("duplicate or invalid selected study prediction identity")
            seen[identity] = value["theta_ac"]
    if seen != expected or len(seen) != offline.EXPECTED_CANDIDATES:
        raise FinalMapperError("full OOF candidate population differs from selected study")
    return candidate_identity_digest(rows)


def publish_full_mapper(destination: Path | str, *, fitted: mapper.FittedMapper,
                        rows: tuple[offline.CandidateRow, ...], study,
                        source: dict, candidate_digest: str):
    """Publish one immutable canonical model, never a calibrated posterior."""
    model = fitted.manifest
    game_ids = sorted({row.game_id for row in rows})
    if (fitted.spec != SELECTED_SPEC or fitted.mapper_fold is not None
            or study.manifest_digest != STUDY_DIGEST
            or model["training_scope"] != "full_development"
            or model["model_spec"] != SELECTED_SPEC.record()
            or model["feature_contract_version"] != mapper.FEATURE_CONTRACT_VERSION
            or model["oof_q"]["evaluation_seal_digest"] != offline.OOF_SEAL_DIGEST
            or model["training_candidate_count"] != offline.EXPECTED_CANDIDATES
            or len(model["training_game_ids"]) != offline.EXPECTED_GAMES
            or model["training_game_ids"] != game_ids
            or model["training_game_digest"] != sha256_bytes(canonical_json_bytes(game_ids))
            or sum(row.theta_ac for row in rows) != EXPECTED_POSITIVES
            or candidate_digest != candidate_identity_digest(rows)):
        raise FinalMapperError("fitted M3 does not cover the frozen full-development population")
    unhashed = dict(model)
    digest = unhashed.pop("model_digest")
    if digest != sha256_bytes(canonical_json_bytes(unhashed)):
        raise FinalMapperError("fitted model digest does not match canonical serialization")
    design = fitted.preprocessor.transform(rows)
    scores = fitted.estimator.decision_function(design)
    probabilities = fitted.estimator.predict_proba(design)[:, 1]
    labels = np.asarray([int(row.theta_ac) for row in rows], dtype=np.int64)
    if len(scores) != len(rows) or not np.isfinite(scores).all():
        raise FinalMapperError("invalid full-development fitted scores")
    summary = {"training_data_semantics": "full-development mapper trained on OOF-Q features",
        "training_scope": "all_1500_development_games",
        "candidate_rows": len(rows), "theta_positive_rows": int(labels.sum()),
        "unweighted_training_log_loss": mapper._log_loss(labels, probabilities),
        "optimizer_iterations": int(fitted.estimator.n_iter_[0]),
        "likelihood": model["likelihood"], "regularizer": model["regularizer"],
        "calibration": "none", "probability_name": "p_tilde"}
    model_bytes = canonical_json_bytes(model)
    return publish_artifact(destination, manifest_fields={
        "artifact_type": "phase2_full_development_oof_mapper",
        "schema_version": FINAL_VERSION,
        "artifact_name": FINAL_NAME,
        "semantics": "full-development mapper trained on OOF-Q features",
        "probability_name": "p_tilde",
        "calibration": "none",
        "source": source,
        "selected_development_study": {"path": str(study.path),
                                       "manifest_digest": study.manifest_digest,
                                       "source_commit": STUDY_SOURCE_COMMIT},
        "inputs": {"publication_digest": offline.PUBLICATION_DIGEST,
                   "role_sidecar_digest": offline.ROLE_SIDECAR_DIGEST,
                   "fold_manifest_digest": offline.FOLD_MANIFEST_DIGEST,
                   "qwen3_oof_evaluation_seal_digest": offline.OOF_SEAL_DIGEST},
        "training": {"game_digest": model["training_game_digest"],
                     "candidate_identity_digest": candidate_digest,
                     "candidate_rows": len(rows),
                     "theta_positive_rows": int(labels.sum()),
                     "objective": summary},
        "model": {"spec": SELECTED_SPEC.record(),
                  "feature_contract_version": mapper.FEATURE_CONTRACT_VERSION,
                  "feature_order": model["preprocessing"]["columns"],
                  "preprocessing": model["preprocessing"],
                  "coefficient": model["coefficient"],
                  "intercept": model["intercept"],
                  "model_digest": digest},
    }, files={"model.json": model_bytes,
              "training_summary.json": canonical_json_bytes(summary)})
