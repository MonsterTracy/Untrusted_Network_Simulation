"""Pure Q -> frozen R2/X -> uncalibrated p_tilde inference for the selected M3."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np

from werewolf.artifact_io import canonical_json_bytes, sha256_bytes, verify_artifact
from werewolf.canonical_collection.public_history import PLAYER_IDS
from werewolf import phase2_mapper as mapper
from werewolf import phase2_offline as offline
from werewolf.phase2_mapper_final import (
    EXPECTED_POSITIVES, FINAL_VERSION, SELECTED_SPEC, STUDY_DIGEST,
)


class RuntimeMapperError(ValueError):
    """An untrusted final model, public state, or Q matrix is invalid."""


@dataclass(frozen=True)
class RuntimeInference:
    p_tilde: float
    raw_score: float
    z: offline.R2Input
    observer_ids: tuple[str, ...]
    competition: tuple[str, ...]
    feature_order: tuple[str, ...]
    design_values: tuple[float, ...]


@dataclass(frozen=True)
class _FeatureRow:
    phase: str
    z: offline.R2Input


def validate_final_q(q) -> np.ndarray:
    """Validate all seven absolute-seat observer rows, including dead rows."""
    try:
        array = np.asarray(q)
    except (TypeError, ValueError) as error:
        raise RuntimeMapperError("final Q is not a numeric 7x7 matrix") from error
    if array.shape != (7, 7) or array.dtype.kind not in "fiu":
        raise RuntimeMapperError("final Q must be numeric 7x7")
    array = array.astype(np.float64)
    if (not np.isfinite(array).all() or np.any(array < 0)
            or not np.all(np.diag(array) == 0.0)
            or any(abs(math.fsum(array[index]) - 1.0) > 1e-6
                   for index in range(7))):
        raise RuntimeMapperError("final Q violates the frozen non-self simplex")
    return array


def build_runtime_z(q, *, alive: tuple[str, ...], known_wolves: frozenset[str],
                    acting_wolf: str, candidate_j: str, phase: str,
                    public_speaker_queue: tuple[str, ...],
                    competition: tuple[str, ...]) -> offline.R2Input:
    """Derive R2 only from final Q and current wolf-visible/public state."""
    matrix = validate_final_q(q)
    alive = tuple(alive)
    wolves = frozenset(known_wolves)
    competition = tuple(competition)
    queue = tuple(public_speaker_queue)
    if (not alive or len(alive) != len(set(alive))
            or not set(alive) <= set(PLAYER_IDS)
            or len(wolves) != 2 or not wolves <= set(PLAYER_IDS)
            or acting_wolf not in wolves or acting_wolf not in alive
            or phase not in ("speech", "speech_pk")
            or len(competition) < 2 or len(competition) != len(set(competition))
            or not set(competition) <= set(alive)
            or (phase == "speech" and set(competition) != set(alive))
            or len(queue) != len(set(queue))
            or set(queue) != (set(alive) if phase == "speech" else set(competition))
            or acting_wolf not in queue):
        raise RuntimeMapperError("illegal current wolf-visible/public competition or queue")
    nonwolves = tuple(player for player in PLAYER_IDS
                      if player in alive and player not in wolves)
    q_by_observer = {observer: tuple(float(value) for value in
                     matrix[PLAYER_IDS.index(observer)]) for observer in nonwolves}
    remaining = len(queue) - queue.index(acting_wolf) - 1
    try:
        return offline.r2_input(alive, wolves, competition, candidate_j,
                                q_by_observer, phase=phase, remaining=remaining)
    except offline.Phase2DataError as error:
        raise RuntimeMapperError(str(error)) from error


def _open_sigmoid(score: float) -> float:
    if not math.isfinite(score):
        raise RuntimeMapperError("non-finite mapper raw score")
    if score >= 0:
        value = 1.0 / (1.0 + math.exp(-score))
    else:
        exponential = math.exp(score)
        value = exponential / (1.0 + exponential)
    return min(math.nextafter(1.0, 0.0), max(math.nextafter(0.0, 1.0), value))


@dataclass(frozen=True)
class RuntimeMapper:
    artifact_digest: str
    model_digest: str
    preprocessor: mapper.FoldPreprocessor
    coefficient: tuple[float, ...]
    intercept: float

    def infer(self, q, *, alive: tuple[str, ...], known_wolves: frozenset[str],
              acting_wolf: str, candidate_j: str, phase: str,
              public_speaker_queue: tuple[str, ...],
              competition: tuple[str, ...], audit: bool = False) -> float | RuntimeInference:
        z = build_runtime_z(q, alive=alive, known_wolves=known_wolves,
                            acting_wolf=acting_wolf, candidate_j=candidate_j,
                            phase=phase, public_speaker_queue=public_speaker_queue,
                            competition=competition)
        design = self.preprocessor.transform((_FeatureRow(z.phase, z),))[0]
        score = math.fsum(float(weight) * float(value)
                          for weight, value in zip(self.coefficient, design, strict=True))
        score += self.intercept
        p_tilde = _open_sigmoid(score)
        if not audit:
            return p_tilde
        observers = tuple(player for player in PLAYER_IDS
                          if player in alive and player not in known_wolves and player != candidate_j)
        return RuntimeInference(p_tilde, score, z, observers, tuple(competition),
                                self.preprocessor.columns(), tuple(map(float, design)))


def load_runtime_mapper(path: Path | str, *,
                        expected_manifest_digest: str | None = None) -> RuntimeMapper:
    artifact = verify_artifact(path,
        expected_artifact_type="phase2_full_development_oof_mapper",
        expected_schema_version=FINAL_VERSION)
    manifest = artifact.manifest
    if (expected_manifest_digest is not None
            and artifact.manifest_digest != expected_manifest_digest):
        raise RuntimeMapperError("final mapper artifact digest mismatch")
    if (manifest["semantics"] != "full-development mapper trained on OOF-Q features"
            or manifest["probability_name"] != "p_tilde" or manifest["calibration"] != "none"
            or manifest["selected_development_study"]["manifest_digest"] != STUDY_DIGEST
            or manifest["model"]["feature_contract_version"] != mapper.FEATURE_CONTRACT_VERSION
            or canonical_json_bytes(manifest["model"]["spec"]) != canonical_json_bytes(
                SELECTED_SPEC.record())
            or manifest["inputs"]["qwen3_oof_evaluation_seal_digest"] != offline.OOF_SEAL_DIGEST):
        raise RuntimeMapperError("final mapper semantics or study binding mismatch")
    sources = manifest["source"]["source_sha256"]
    for relative, path in (("werewolf/phase2_offline.py", Path(offline.__file__)),
                           ("werewolf/phase2_mapper.py", Path(mapper.__file__)),
                           ("werewolf/phase2_mapper_runtime.py", Path(__file__))):
        if sources.get(relative) != sha256_bytes(path.read_bytes()):
            raise RuntimeMapperError(f"runtime feature implementation differs from artifact: {relative}")
    model = json.loads((artifact.path / "model.json").read_bytes())
    unhashed = dict(model)
    model_digest = unhashed.pop("model_digest")
    if (model_digest != sha256_bytes(canonical_json_bytes(unhashed))
            or model_digest != manifest["model"]["model_digest"]
            or model["training_scope"] != "full_development"
            or manifest["training"]["game_digest"] != model["training_game_digest"]
            or model["feature_contract_version"] != mapper.FEATURE_CONTRACT_VERSION
            or canonical_json_bytes(model["model_spec"]) != canonical_json_bytes(
                SELECTED_SPEC.record())
            or manifest["training"]["candidate_rows"] != offline.EXPECTED_CANDIDATES
            or manifest["training"]["theta_positive_rows"] != EXPECTED_POSITIVES):
        raise RuntimeMapperError("final M3 model lineage or full-development population mismatch")
    preprocessing = model["preprocessing"]
    restored = mapper.FoldPreprocessor(SELECTED_SPEC, preprocessing["means"],
        preprocessing["scales"],
        {name: tuple(values) for name, values in preprocessing["knots"].items()},
        preprocessing["training_game_digest"])
    coefficients = tuple(model["coefficient"])
    if (tuple(preprocessing["columns"]) != restored.columns()
            or canonical_json_bytes(manifest["model"]["preprocessing"]) != canonical_json_bytes(preprocessing)
            or manifest["model"]["feature_order"] != preprocessing["columns"]
            or manifest["model"]["coefficient"] != model["coefficient"]
            or manifest["model"]["intercept"] != model["intercept"]
            or len(coefficients) != len(restored.columns())
            or not all(math.isfinite(value) for value in coefficients)
            or not math.isfinite(model["intercept"])
            or set(restored.means) != set(mapper.CONTEXT_NUMERIC + mapper.EVIDENCE)
            or set(restored.scales) != set(restored.means)
            or set(restored.knots) != set(mapper.EVIDENCE)
            or any(not math.isfinite(value) for value in restored.means.values())
            or any(not math.isfinite(value) or value <= 0 for value in restored.scales.values())
            or any(len(values) != 2 or not all(math.isfinite(value) for value in values)
                   or values[0] > values[1] for values in restored.knots.values())):
        raise RuntimeMapperError("invalid frozen feature order, preprocessing, or coefficients")
    return RuntimeMapper(artifact.manifest_digest, model_digest,
                         restored, coefficients, float(model["intercept"]))
