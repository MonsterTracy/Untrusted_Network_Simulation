"""Read sealed historical Qwen3 OOF v1 without importing training or Phase2.

The caller pins the evaluation seal digest. Prediction payloads are fully
verified; training/preparation manifests are metadata-only trust links (no
weights, optimizer state, bootstrap draws or model replay are opened).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re

from werewolf.artifact_io import (
    ArtifactValidationError, canonical_json_bytes, open_artifact_envelope,
    read_artifact_file,
)
from werewolf.canonical_collection.public_history import PLAYER_IDS
from werewolf.development_publication import open_publication
from werewolf.observer_dtrs.context import ObserverContext
from werewolf.observer_dtrs.probabilities import ProbabilityRow
from werewolf.structured_history import plan_structured_history


OOF_SOURCE_VERSION = "observer_dtrs_qwen3_oof_source_v1"
CONDITION = "explicit_day_phase"
_ROW_FIELDS = {"game_id", "boundary_id", "observer", "prefix_digest", "plan_digest",
               "probability", "non_self_log_probability", "q", "label_observed",
               "observer_alive", "checkpoint_digest", "temporal_condition"}


@dataclass(frozen=True)
class OOFSource:
    publication_digest: str
    fold_manifest_digest: str
    evaluation_digest: str
    evaluation_seal_digest: str
    preparation_digest: str
    training_execution_digest: str
    training_seal_digest: str
    training_revision: str
    evaluation_revision: str
    architecture: str
    temporal_condition: str
    fold_index: int
    terminal_digest: str
    checkpoint_digest: str
    prediction_artifact_digest: str
    prediction_payload_sha256: str

    def to_record(self) -> dict:
        return {"schema_version": OOF_SOURCE_VERSION, **asdict(self)}


@dataclass(frozen=True)
class OOFObserverPrediction:
    """Offline provenance envelope; only .probability goes to legal features."""

    context: ObserverContext
    probability: ProbabilityRow
    source: OOFSource


@dataclass(frozen=True)
class Qwen3OOFBundle:
    """Offline coverage evidence, including folds with zero requested samples."""

    predictions: tuple[OOFObserverPrediction, ...]
    sources: tuple[OOFSource, ...]
    publication_manifest: dict
    game_folds: tuple[tuple[str, int], ...]
    fold_prediction_counts: tuple[int, ...]


def _require(condition, message):
    if not condition:
        raise ArtifactValidationError(message)


def _metadata(path, artifact_type, version, digest=None):
    artifact = open_artifact_envelope(path, expected_artifact_type=artifact_type,
                                      expected_schema_version=version)
    if digest is not None:
        _require(artifact.manifest_digest == digest, "OOF artifact identity mismatch")
    return artifact


def _lineage(rows, fold):
    matches = [r for r in rows if r["architecture"] == "qwen3" and r["fold"] == fold]
    _require(len(matches) == 1 and type(matches[0]["fold"]) is int,
             "OOF missing/duplicate model-fold lineage")
    return matches[0]


def load_qwen3_oof(publication_path, evaluation_path, contexts, *,
                   expected_seal_digest: str) -> tuple[OOFObserverPrediction, ...]:
    return load_qwen3_oof_bundle(publication_path, evaluation_path, contexts,
        expected_seal_digest=expected_seal_digest).predictions


def load_qwen3_oof_bundle(publication_path, evaluation_path, contexts, *,
                          expected_seal_digest: str) -> Qwen3OOFBundle:
    """Require complete 7-row/PRE OOF coverage, then return requested contexts.

    No label mask, true report, q target or old Primary/candidate population is
    consulted. contexts may be a whole-game subset; validation still covers all
    five folds and all publication PREs. Paths in the historical evaluation
    contract identify the original sealed training/preparation directories.
    """
    if not isinstance(expected_seal_digest, str) or re.fullmatch(r"[0-9a-f]{64}", expected_seal_digest) is None:
        raise ArtifactValidationError("an explicit evaluation seal SHA-256 trust anchor is required")
    contexts = tuple(contexts)
    if any(not isinstance(c, ObserverContext) for c in contexts):
        raise TypeError("OOF adapter accepts ObserverContext, never offline samples/supervision")
    keys = [(c.game_id, c.boundary_id, c.observer_id) for c in contexts]
    _require(len(keys) == len(set(keys)), "duplicate requested observer context")
    try:
        return _load(publication_path, evaluation_path, contexts, keys, expected_seal_digest)
    except ArtifactValidationError:
        raise
    except (OSError, KeyError, TypeError, ValueError, IndexError) as error:
        raise ArtifactValidationError(f"invalid Qwen3 OOF source: {error}") from error


def _load(publication_path, evaluation_path, contexts, keys, seal_digest):
    publication = open_publication(publication_path)
    view = publication.public_view
    root = Path(evaluation_path)
    seal = _metadata(root / "seal", "backbone_oof_evaluation_report",
                     "classic7_backbone_oof_report_v1", seal_digest)
    sm = seal.manifest
    contract = _metadata(root / "contract", "backbone_oof_evaluation_contract",
                         "classic7_backbone_oof_evaluation_v1", sm["evaluation_digest"])
    cm = contract.manifest
    for key in ("preparation", "training_execution_digest", "training_seal_digest",
                "training_source", "evaluation_source"):
        _require(sm[key] == cm[key], "OOF evaluation/seal provenance mismatch")
    training_root = Path(cm["training_path"])
    training = _metadata(training_root / "contract", "backbone_execution",
                         "classic7_backbone_execution_v1", cm["training_execution_digest"])
    tm = training.manifest
    train_seal = _metadata(training_root / "seal", "backbone_execution_seal",
                           "classic7_backbone_all_twenty_v1", cm["training_seal_digest"])
    ts = train_seal.manifest
    _require(tm["engineering_only"] is False and tm["preparation"] == cm["preparation"]
             and tm["execution_source"] == cm["training_source"]
             and cm["training_revision"] == cm["training_source"]["source_revision"]
             and ts["execution_digest"] == training.manifest_digest
             and ts["preparation"] == tm["preparation"]
             and ts["execution_source"] == tm["execution_source"],
             "OOF training identity mismatch")
    preparation = _metadata(tm["prepared_path"], "backbone_tom_study",
                            "classic7_backbone_study_v1", cm["preparation"]["manifest_digest"])
    pm = preparation.manifest
    protocol = pm["protocol_inputs"]
    _require(protocol["publication_digest"] == publication.manifest_digest
             and protocol["fold_manifest_digest"] == view.fold_manifest.manifest_digest
             and protocol["development_game_set_digest"] == view.development_game_set_digest,
             "OOF publication/fold identity mismatch")

    folds = {f.fold_index: set(f.game_ids) for f in view.fold_manifest.folds}
    _require(set(folds) == set(range(5)) and len(pm["folds"]) == 5,
             "OOF requires the canonical five-fold publication")
    game_fold = {g: f for f, ids in folds.items() for g in ids}
    all_games = set(game_fold)
    for fold, held_out in folds.items():
        records = [r for r in pm["folds"] if r["fold"] == fold]
        _require(len(records) == 1, "OOF missing/duplicate preparation fold")
        record = records[0]
        held, trained = record["held_out_game_ids"], record["training_game_ids"]
        _require(type(record["fold"]) is int and len(held) == len(set(held))
                 and len(trained) == len(set(trained)) and set(held) == held_out
                 and set(trained) == all_games - held_out,
                 "OOF training/held-out game partition mismatch")

    prefixes = {(g.game_id, p.boundary_id): p for g in view.games
                for p in g.authoritative_pre_prefixes}
    plans = {k: plan_structured_history(p).plan_digest for k, p in prefixes.items()}
    for c in contexts:
        p = prefixes.get((c.game_id, c.boundary_id))
        _require(p is not None, "OOF requested game/PRE missing from publication")
        _require((c.prefix_digest, c.step_index, c.acting_wolf, c.day, c.phase, c.alive_player_ids)
                 == (p.prefix_digest, p.step_index, p.current_speaker,
                     p.public_temporal_state.day, p.public_temporal_state.phase.value, p.alive_observer_ids),
                 "OOF requested context/PRE identity mismatch")

    wanted = set(keys)
    selected = {}
    sources, counts = [], []
    for fold in range(5):
        descriptor = _lineage(sm["predictions"], fold)
        terminal_ref = _lineage(ts["terminals"], fold)
        _require(terminal_ref["temporal_condition"] == CONDITION
                 and terminal_ref["terminal_digest"] == descriptor["terminal_digest"],
                 "OOF sealed terminal identity mismatch")
        terminal = _metadata(training_root / "runs" / "qwen3" / str(fold) / "terminal",
                             "backbone_terminal", "classic7_backbone_terminal_v1",
                             descriptor["terminal_digest"])
        binding = terminal.manifest["binding"]
        _require(binding["architecture"] == "qwen3" and type(binding["fold"]) is int
                 and binding["fold"] == fold and binding["temporal_condition"] == CONDITION
                 and binding["engineering_only"] is False
                 and binding["execution_digest"] == training.manifest_digest
                 and binding["execution_source"] == tm["execution_source"]
                 and binding["preparation"] == tm["preparation"],
                 "OOF terminal model/fold binding mismatch")
        checkpoint = terminal.manifest["checkpoint_ancestry"][-1]["digest"]
        _require(isinstance(checkpoint, str) and re.fullmatch(r"[0-9a-f]{64}", checkpoint) is not None,
                 "OOF invalid checkpoint identity")
        artifact = _metadata(root / "folds" / "qwen3" / str(fold), "backbone_oof_fold_predictions",
                             "classic7_backbone_oof_predictions_v1", descriptor["prediction_digest"])
        am = artifact.manifest
        expected = {"evaluation_digest": contract.manifest_digest,
                    "training_seal_digest": train_seal.manifest_digest,
                    "architecture": "qwen3", "fold": fold, "temporal_condition": CONDITION,
                    "terminal_digest": terminal.manifest_digest, "checkpoint_digest": checkpoint}
        _require(all(am[k] == v for k, v in expected.items()) and type(am["fold"]) is int
                 and set(am["file_table"]) == {"predictions.jsonl"}
                 and am["primary_row_identity_digest"] == sm["primary_row_bindings"][str(fold)],
                 "OOF prediction artifact model/fold provenance mismatch")
        source = OOFSource(publication.manifest_digest, view.fold_manifest.manifest_digest,
            contract.manifest_digest, seal.manifest_digest, preparation.manifest_digest,
            training.manifest_digest, train_seal.manifest_digest, cm["training_revision"],
            cm["evaluation_source"]["source_revision"], "qwen3", CONDITION, fold,
            terminal.manifest_digest, checkpoint, artifact.manifest_digest,
            am["file_table"]["predictions.jsonl"]["sha256"])
        raw = read_artifact_file(artifact, "predictions.jsonl")
        _require(raw.endswith(b"\n"), "OOF truncated prediction payload")
        seen = set()
        for line in raw.splitlines():
            row = json.loads(line)
            _require(line == canonical_json_bytes(row) and set(row) == _ROW_FIELDS,
                     "OOF prediction row schema/serialization mismatch")
            observer = row["observer"]
            _require(type(observer) is int and 0 <= observer < 7, "OOF invalid observer index")
            pre_key = row["game_id"], row["boundary_id"]
            p = prefixes.get(pre_key)
            _require(p is not None and game_fold[row["game_id"]] == fold,
                     "OOF game/PRE/fold mismatch")
            _require(row["prefix_digest"] == p.prefix_digest and row["plan_digest"] == plans[pre_key]
                     and row["checkpoint_digest"] == checkpoint and row["temporal_condition"] == CONDITION,
                     "OOF row PRE/model identity mismatch")
            key = (*pre_key, PLAYER_IDS[observer])
            _require(key not in seen, "duplicate OOF observer prediction")
            seen.add(key)
            probability = ProbabilityRow(*pre_key, p.prefix_digest, PLAYER_IDS[observer], row["probability"])
            if key in wanted:
                selected[key] = probability, source
            # Deliberately discard q, labels, alive masks and logs. They neither
            # define this population nor enter the feature interface.
        expected_count = sum(7 for game_id, _ in prefixes if game_id in folds[fold])
        _require(type(am["row_count"]) is int and len(seen) == am["row_count"] == expected_count,
                 "OOF missing observer/PRE predictions")
        sources.append(source)
        counts.append(len(seen))
    _require(set(selected) == wanted, "OOF missing requested observer prediction")
    predictions = tuple(OOFObserverPrediction(c, *selected[key])
                        for c, key in zip(contexts, keys, strict=True))
    return Qwen3OOFBundle(predictions, tuple(sources), publication.manifest,
        tuple((g, game_fold[g]) for g in view.game_ids), tuple(counts))
