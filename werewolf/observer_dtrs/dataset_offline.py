"""Explicit restricted dataset construction, supervision access and source audit."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, fields
import importlib.metadata
import json
from pathlib import Path
import platform
import re

from werewolf.artifact_io import (
    ArtifactValidationError, canonical_json_bytes, canonical_jsonl_bytes,
    open_artifact_envelope, publish_artifact, read_artifact_file, sha256_bytes,
)
from werewolf.canonical_collection.public_history import PLAYER_IDS
from werewolf.development_publication import _validate_fold_manifest_record
from werewolf.observer_dtrs.context import CONTEXT_VERSION
from werewolf.observer_dtrs.dataset import (
    ACCESS_POLICY, DATASET_TYPE, DATASET_VERSION, LegalDatasetRow, ObserverDatasetView,
    _check, _digest, _equal, _json, _jsonl, open_dataset, sample_id,
)
from werewolf.observer_dtrs.features import FEATURE_VERSION, observer_features
from werewolf.observer_dtrs.labels import LABEL_VERSION, ObserverLabel
from werewolf.observer_dtrs.offline import SAMPLE_VERSION, SampleSource, build_observer_samples
from werewolf.observer_dtrs.oof import OOF_SOURCE_VERSION, OOFSource, load_qwen3_oof_bundle


AUDIT_VERSION = "observer_dtrs_dataset_coverage_v1"


@dataclass(frozen=True)
class OfflineDatasetRow:
    legal: LegalDatasetRow
    supervision: ObserverLabel
    sample_source: SampleSource
    q_source: OOFSource


def implementation_snapshot(base_revision):
    """Archive exact uncommitted Python sources, not just the Git base revision."""
    _check(isinstance(base_revision, str) and re.fullmatch(r"[0-9a-f]{40}", base_revision) is not None,
           "dataset requires an explicit Git base revision")
    root = Path(__file__).resolve().parents[2]
    paths = sorted([*(root / "werewolf").rglob("*.py"),
                    root / "scripts/manage_observer_dtrs_dataset.py", root / "setup.py"])
    _check(all(not p.is_symlink() for p in paths), "implementation source cannot be a symlink")
    return {"base_revision": base_revision,
            "files": {p.relative_to(root).as_posix(): p.read_text(encoding="utf-8") for p in paths}}


def _counts(rows):
    return {"sample_count": len(rows),
        "game_count": len({r.legal.context.game_id for r in rows}),
        "wolf_pre_count": len({(r.legal.context.game_id, r.legal.context.boundary_id) for r in rows}),
        "positive": sum(r.supervision.event_label_y == 1 for r in rows),
        "negative": sum(r.supervision.event_label_y == 0 for r in rows),
        "successful_empty": sum(not r.supervision.raw_suspicion_support for r in rows),
        "successful_nonempty": sum(bool(r.supervision.raw_suspicion_support) for r in rows)}


def coverage_audit(rows, games, coverage):
    groups = {
        "by_game": [(g["game_id"], lambda r, g=g: r.legal.context.game_id == g["game_id"]) for g in games],
        "by_phase": [(p, lambda r, p=p: r.legal.context.phase == p) for p in ("discussion", "pk_discussion")],
        "by_nW": [(str(n), lambda r, n=n: r.legal.features.nW == n) for n in (1, 2)],
        "by_fold": [(str(f), lambda r, f=f: r.sample_source.fold_index == f) for f in range(5)],
        "by_report": [("empty", lambda r: not r.supervision.raw_suspicion_support),
                      ("nonempty", lambda r: bool(r.supervision.raw_suspicion_support))],
        "by_phase_nW": [(f"{p}/{n}", lambda r, p=p, n=n: r.legal.context.phase == p and r.legal.features.nW == n)
                         for p in ("discussion", "pk_discussion") for n in (1, 2)],
    }
    audit = {"schema_version": AUDIT_VERSION, "source_coverage": coverage,
             "overall": _counts(rows), "zero_sample_games": [g["game_id"] for g in games if not g["row_count"]]}
    for name, predicates in groups.items():
        audit[name] = {key: _counts([r for r in rows if predicate(r)]) for key, predicate in predicates}
    return audit


def _capsules(publication_path, evaluation_path, publication_manifest):
    """Copy verified metadata only; never copy sidecar roles or model payloads."""
    root = Path(evaluation_path)
    contract = _json((root / "contract/manifest.json").read_bytes())
    training = Path(contract["training_path"])
    tc = _json((training / "contract/manifest.json").read_bytes())
    paths = {"publication": Path(publication_path) / "manifest.json",
        "evaluation_contract": root / "contract/manifest.json", "evaluation_seal": root / "seal/manifest.json",
        "training_contract": training / "contract/manifest.json", "training_seal": training / "seal/manifest.json",
        "preparation": Path(tc["prepared_path"]) / "manifest.json"}
    for f in range(5):
        paths[f"prediction/{f}"] = root / f"folds/qwen3/{f}/manifest.json"
        paths[f"terminal/{f}"] = training / f"runs/qwen3/{f}/terminal/manifest.json"
    result = {key: _json(path.read_bytes()) for key, path in paths.items()}
    _check(_equal(result["publication"], publication_manifest), "source publication changed during build")
    result["fold_manifest"] = _json((Path(publication_path) / "public/development_fold_manifest.json").read_bytes())
    return result


def _assemble(publication_path, evaluation_path, expected_seal_digest):
    samples = build_observer_samples(publication_path)
    bundle = load_qwen3_oof_bundle(publication_path, evaluation_path,
        (s.context for s in samples), expected_seal_digest=expected_seal_digest)
    pm = bundle.publication_manifest
    publication_origin = {k: pm[k] for k in ("publication_id", "collection_identity_digest",
        "development_game_set_digest", "role_sidecar_digest", "manifest_digest")}
    publication_origin["fold_manifest_digest"] = pm["development_fold_manifest_digest"]
    origins = {"publication": publication_origin, "oof": [s.to_record() for s in bundle.sources]}
    rows, seen = [], set()
    for sample, prediction in zip(samples, bundle.predictions, strict=True):
        _check(sample.context == prediction.context, "dataset sample/Q context join mismatch")
        source = sample.source
        _check(all(getattr(source, k) == publication_origin[k] for k in (
            "publication_id", "collection_identity_digest", "development_game_set_digest",
            "role_sidecar_digest", "fold_manifest_digest"))
            and source.publication_digest == publication_origin["manifest_digest"]
            and prediction.source == bundle.sources[source.fold_index], "dataset source join mismatch")
        legal = LegalDatasetRow(sample_id(sample.context), sample.context, prediction.probability,
                                observer_features(sample.context, prediction.probability))
        _check(legal.sample_id not in seen, "duplicate dataset sample join")
        seen.add(legal.sample_id)
        rows.append(OfflineDatasetRow(legal, sample.supervision, source, prediction.source))
    by_game = defaultdict(list)
    for row in rows:
        by_game[row.legal.context.game_id].append(row)
    files, games = {}, []
    for game_id, fold in bundle.game_folds:
        selected = by_game[game_id]
        games.append({"game_id": game_id, "fold_index": fold, "row_count": len(selected),
            "pre_count": len({r.legal.context.boundary_id for r in selected}),
            "sample_ids_digest": sha256_bytes(canonical_json_bytes([r.legal.sample_id for r in selected]))})
        files[f"legal/games/{game_id}.jsonl"] = canonical_jsonl_bytes(r.legal.to_record() for r in selected)
        files[f"restricted/supervision/{game_id}.jsonl"] = canonical_jsonl_bytes(
            {"sample_id": r.legal.sample_id, "supervision": r.supervision.to_record()} for r in selected)
        files[f"restricted/provenance/{game_id}.jsonl"] = canonical_jsonl_bytes(
            {"sample_id": r.legal.sample_id, "sample_source": asdict(r.sample_source),
             "q_source": r.q_source.to_record()} for r in selected)
    coverage = {"game_count": len(games), "canonical_pre_count": pm["pre_boundary_count"],
        "oof_prediction_row_count": sum(bundle.fold_prediction_counts),
        "oof_fold_prediction_counts": bundle.fold_prediction_counts,
        "sample_count": len(rows), "wolf_pre_count": sum(g["pre_count"] for g in games)}
    files["restricted/audit.json"] = canonical_json_bytes(coverage_audit(rows, games, coverage))
    files["restricted/upstream_manifests.json"] = canonical_json_bytes(
        _capsules(publication_path, evaluation_path, pm))
    return origins, games, coverage, files


def publish_dataset(destination, *, dataset_id, publication_path, evaluation_path,
                    expected_seal_digest, base_revision):
    """Build the complete source population and atomically publish without overwrite."""
    _check(isinstance(dataset_id, str) and re.fullmatch(r"[A-Za-z0-9_.-]+", dataset_id) is not None,
           "dataset_id must be a safe artifact identity")
    target = Path(destination).resolve()
    _check(all(not target.is_relative_to(Path(p).resolve()) and not Path(p).resolve().is_relative_to(target)
               for p in (publication_path, evaluation_path)), "dataset destination overlaps upstream artifacts")
    source = implementation_snapshot(base_revision)
    origins, games, coverage, payloads = _assemble(publication_path, evaluation_path, expected_seal_digest)
    _check(_equal(source, implementation_snapshot(base_revision)), "implementation changed during dataset build")
    payloads["restricted/implementation_sources.json"] = canonical_json_bytes(source)
    versions = {"context": CONTEXT_VERSION, "sample": SAMPLE_VERSION, "label": LABEL_VERSION,
                "features": FEATURE_VERSION, "q_source": OOF_SOURCE_VERSION, "audit": AUDIT_VERSION}
    artifact = publish_artifact(destination, manifest_fields={
        "artifact_type": DATASET_TYPE, "schema_version": DATASET_VERSION, "artifact_identity": dataset_id,
        "access_policy": ACCESS_POLICY, "versions": versions, "origins": origins, "games": games,
        "coverage": coverage, "implementation": {"base_revision": base_revision,
            "snapshot_digest": sha256_bytes(canonical_json_bytes(source))},
        "runtime": {"python": platform.python_version(),
            "pydantic": importlib.metadata.version("pydantic")},
        "source_locations": {"publication": str(Path(publication_path).resolve()),
                             "evaluation": str(Path(evaluation_path).resolve())}}, files=payloads)
    # Read back every partition and audit before returning success.
    verify_dataset(artifact.path, expected_digest=artifact.manifest_digest)
    return artifact


def _label(context, record):
    raw = dict(record)
    _check(raw.pop("label_construction_version") == LABEL_VERSION, "dataset label version mismatch")
    _check(set(raw) == {f.name for f in fields(ObserverLabel)}, "dataset supervision schema mismatch")
    for key in ("raw_suspicion_support", "effective_suspicion_support"):
        support = raw[key]
        _check(isinstance(support, list) and support == [p for p in PLAYER_IDS if p in support]
               and context.observer_id not in support, "dataset invalid suspicion support")
        raw[key] = tuple(support)
    label = ObserverLabel(**raw)
    _check(label.observation_status == "success" and label.label_observed is True
           and type(label.event_label_y) is int, "dataset requires successful observed supervision")
    effective = label.raw_suspicion_support or tuple(p for p in PLAYER_IDS if p != context.observer_id)
    _check(label.effective_suspicion_support == effective
           and label.event_label_y == int(bool(set(effective) & set(context.alive_wolf_ids))),
           "dataset label/effective support mismatch")
    _digest(label.observation_digest)
    _check(isinstance(label.observation_id, str) and bool(label.observation_id)
           and (label.attempt_id is None or isinstance(label.attempt_id, str)), "dataset report identity missing")
    return label


def open_supervision(view: ObserverDatasetView) -> tuple[OfflineDatasetRow, ...]:
    """Explicit offline opt-in, limited to the legal view's selected whole games."""
    if not isinstance(view, ObserverDatasetView):
        raise TypeError("supervision requires an opened legal dataset view")
    try:
        return _supervision(view)
    except ArtifactValidationError:
        raise
    except (OSError, KeyError, TypeError, ValueError, IndexError) as error:
        raise ArtifactValidationError(f"invalid dataset supervision: {error}") from error


def _supervision(view):
    m = view.artifact.manifest
    _check(m["versions"]["sample"] == SAMPLE_VERSION and m["versions"]["label"] == LABEL_VERSION
           and m["versions"]["q_source"] == OOF_SOURCE_VERSION, "dataset offline version mismatch")
    origins = m["origins"]["publication"]
    by_game = defaultdict(list)
    for row in view.rows:
        by_game[row.context.game_id].append(row)
    games = {g["game_id"]: g for g in m["games"]}
    result = []
    for game_id in view.game_ids:
        labels = _jsonl(read_artifact_file(view.artifact, f"restricted/supervision/{game_id}.jsonl"))
        provenance = _jsonl(read_artifact_file(view.artifact, f"restricted/provenance/{game_id}.jsonl"))
        _check(len(labels) == len(provenance) == len(by_game[game_id]), "dataset restricted coverage mismatch")
        for legal, label, proof in zip(by_game[game_id], labels, provenance, strict=True):
            _check(set(label) == {"sample_id", "supervision"}
                   and set(proof) == {"sample_id", "sample_source", "q_source"}
                   and legal.sample_id == label["sample_id"] == proof["sample_id"],
                   "dataset supervision/provenance identity mismatch")
            source = SampleSource(**proof["sample_source"])
            _check(type(source.fold_index) is int and source.fold_index == games[game_id]["fold_index"]
                   and source.publication_digest == origins["manifest_digest"]
                   and all(getattr(source, k) == origins[k] for k in ("publication_id", "collection_identity_digest",
                       "development_game_set_digest", "role_sidecar_digest", "fold_manifest_digest")),
                   "dataset sample source/fold mismatch")
            for key, value in asdict(source).items():
                if key.endswith("digest"):
                    _digest(value)
            _check(_equal(proof["q_source"], m["origins"]["oof"][source.fold_index]), "dataset Q source/fold mismatch")
            q = dict(proof["q_source"])
            _check(q.pop("schema_version") == OOF_SOURCE_VERSION, "dataset Q source version mismatch")
            result.append(OfflineDatasetRow(legal, _label(legal.context, label["supervision"]), source, OOFSource(**q)))
    return tuple(result)


def _verify_capsules(m, capsules, rows):
    _check(_equal(m["versions"], {"context": CONTEXT_VERSION, "sample": SAMPLE_VERSION, "label": LABEL_VERSION,
        "features": FEATURE_VERSION, "q_source": OOF_SOURCE_VERSION, "audit": AUDIT_VERSION}),
        "dataset version set mismatch")
    for key, record in capsules.items():
        if key != "fold_manifest":
            _check(sha256_bytes(canonical_json_bytes({k: v for k, v in record.items() if k != "manifest_digest"}))
                   == record["manifest_digest"], "dataset upstream manifest digest mismatch")
    pm = capsules["publication"]
    origin = m["origins"]["publication"]
    _check(all(pm[k] == origin[k] for k in ("manifest_digest", "publication_id", "collection_identity_digest",
               "development_game_set_digest", "role_sidecar_digest"))
           and pm["game_count"] == m["coverage"]["game_count"]
           and pm["pre_boundary_count"] == m["coverage"]["canonical_pre_count"],
           "dataset publication capsule mismatch")
    original_games = [(g["game_id"], g["bundle_digest"]) for g in pm["development_games"]]
    fold = _validate_fold_manifest_record(capsules["fold_manifest"], original_games, origin["development_game_set_digest"])
    _check(fold.manifest_digest == origin["fold_manifest_digest"], "dataset fold capsule mismatch")
    fold_map = {g: f.fold_index for f in fold.folds for g in f.game_ids}
    _check([(g["game_id"], g["fold_index"]) for g in m["games"]]
           == [(g, fold_map[g]) for g, _ in original_games], "dataset cross-game fold partition mismatch")
    game_sources = {g["game_id"]: g for g in pm["development_games"]}
    for row in rows:
        g = game_sources[row.legal.context.game_id]
        _check(row.sample_source.bundle_digest == g["bundle_digest"]
               and row.sample_source.published_game_record_digest == g["published_record_digest"],
               "dataset game source capsule mismatch")
    contract, seal = capsules["evaluation_contract"], capsules["evaluation_seal"]
    tc, ts, preparation = (capsules[k] for k in ("training_contract", "training_seal", "preparation"))
    _check(contract["training_execution_digest"] == tc["manifest_digest"] == ts["execution_digest"]
           and contract["training_seal_digest"] == ts["manifest_digest"]
           and contract["preparation"] == tc["preparation"] == ts["preparation"]
           and tc["preparation"]["manifest_digest"] == preparation["manifest_digest"]
           and contract["training_source"] == tc["execution_source"] == ts["execution_source"],
           "dataset OOF training capsule chain mismatch")
    for key in ("preparation", "training_execution_digest", "training_seal_digest", "training_source", "evaluation_source"):
        _check(_equal(contract[key], seal[key]), "dataset evaluation capsule chain mismatch")
    protocol = preparation["protocol_inputs"]
    _check(protocol["publication_digest"] == pm["manifest_digest"]
           and protocol["fold_manifest_digest"] == fold.manifest_digest
           and protocol["development_game_set_digest"] == origin["development_game_set_digest"],
           "dataset preparation publication capsule mismatch")
    _check(len(m["origins"]["oof"]) == 5, "dataset missing OOF source fold")
    all_game_ids = set(game_sources)
    for f, source in enumerate(m["origins"]["oof"]):
        am, terminal = capsules[f"prediction/{f}"], capsules[f"terminal/{f}"]
        expected_source = OOFSource(pm["manifest_digest"], fold.manifest_digest,
            contract["manifest_digest"], seal["manifest_digest"], preparation["manifest_digest"],
            tc["manifest_digest"], ts["manifest_digest"], contract["training_revision"],
            contract["evaluation_source"]["source_revision"], "qwen3", "explicit_day_phase", f,
            terminal["manifest_digest"], terminal["checkpoint_ancestry"][-1]["digest"],
            am["manifest_digest"], am["file_table"]["predictions.jsonl"]["sha256"])
        _check(_equal(source, expected_source.to_record()), "dataset OOF source record mismatch")
        held = {g for g in all_game_ids if fold_map[g] == f}
        prep_folds = [r for r in preparation["folds"] if r["fold"] == f]
        _check(len(prep_folds) == 1 and set(prep_folds[0]["held_out_game_ids"]) == held
               and set(prep_folds[0]["training_game_ids"]) == all_game_ids - held,
               "dataset training/held-out capsule mismatch")
        binding = terminal["binding"]
        _check(contract["manifest_digest"] == seal["evaluation_digest"] == am["evaluation_digest"]
               and am["training_seal_digest"] == ts["manifest_digest"]
               and am["architecture"] == binding["architecture"] == "qwen3"
               and am["temporal_condition"] == binding["temporal_condition"] == "explicit_day_phase"
               and type(am["fold"]) is int and type(binding["fold"]) is int
               and am["fold"] == binding["fold"] == f
               and binding["execution_digest"] == tc["manifest_digest"]
               and binding["execution_source"] == tc["execution_source"]
               and binding["preparation"] == tc["preparation"] and binding["engineering_only"] is False
               and am["terminal_digest"] == terminal["manifest_digest"]
               and am["checkpoint_digest"] == source["checkpoint_digest"]
               and am["row_count"] == m["coverage"]["oof_fold_prediction_counts"][f],
               "dataset OOF capsule identity mismatch")
        descriptors = [r for r in seal["predictions"] if r["architecture"] == "qwen3" and r["fold"] == f]
        _check(len(descriptors) == 1 and descriptors[0]["prediction_digest"] == am["manifest_digest"],
               "dataset sealed prediction descriptor mismatch")


def verify_dataset(path, *, expected_digest=None):
    """Verify every byte, local semantics, source capsules and full audit; no source I/O."""
    try:
        view = open_dataset(path, expected_digest=expected_digest)
        rows = open_supervision(view)
        m = view.artifact.manifest
        capsules = _json(read_artifact_file(view.artifact, "restricted/upstream_manifests.json"))
        _verify_capsules(m, capsules, rows)
        source = _json(read_artifact_file(view.artifact, "restricted/implementation_sources.json"))
        _check(source["base_revision"] == m["implementation"]["base_revision"]
               and sha256_bytes(canonical_json_bytes(source)) == m["implementation"]["snapshot_digest"],
               "dataset implementation snapshot mismatch")
        audit = _json(read_artifact_file(view.artifact, "restricted/audit.json"))
        _check(_equal(audit, coverage_audit(rows, m["games"], m["coverage"])), "dataset coverage audit mismatch")
        return audit
    except ArtifactValidationError:
        raise
    except (OSError, KeyError, TypeError, ValueError, IndexError) as error:
        raise ArtifactValidationError(f"invalid dataset audit: {error}") from error


def audit_dataset_sources(path, *, publication_path, evaluation_path, expected_seal_digest,
                          expected_digest=None):
    """Rebuild all D1a/D1b records from original sources and compare complete bytes."""
    audit = verify_dataset(path, expected_digest=expected_digest)
    view = open_dataset(path, expected_digest=expected_digest)
    origins, games, coverage, payloads = _assemble(publication_path, evaluation_path, expected_seal_digest)
    m = view.artifact.manifest
    _check(_equal(origins, m["origins"]) and _equal(games, m["games"])
           and _equal(coverage, m["coverage"]), "dataset source reconstruction population/origin mismatch")
    for name, data in payloads.items():
        _check(read_artifact_file(view.artifact, name) == data, f"dataset source reconstruction differs: {name}")
    return {"manifest_digest": view.manifest_digest, "source_reconstruction": "exact_match",
            "audit": audit}
