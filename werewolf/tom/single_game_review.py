"""读取并校验单局 ToM 导出及其绑定的公开 PRE 快照。"""
from __future__ import annotations

import csv
from dataclasses import dataclass
import math
import os
from pathlib import Path
import re

import numpy as np

from werewolf.artifact_io import canonical_json_bytes, canonical_jsonl_bytes, sha256_bytes, verify_artifact
from werewolf.artifact_io.canonical import _load_canonical_json
from werewolf.canonical_collection.game_bundle import _history_from_records, _prefix_from_record
from werewolf.canonical_collection.pre import validate_authoritative_pre_prefix
from werewolf.canonical_collection.public_history import PLAYER_IDS
from werewolf.observer_dtrs.final_tom_audit import _Pins

EXPORT_TYPE = "tom_one_game_all_pre_matrices"
EXPORT_VERSION = "tom_one_game_all_pre_export_v1"
ROW_SUM_ATOL = 1e-5  # Exact convention in the supplied export script.
EXPORT_FOLDER = "tom-single-game-matrices-demo-v1"
CSV_COLUMNS = (
    "game_id", "boundary_id", "step_index", "day", "phase", "current_speaker",
    "observer", "target", "observer_alive", "target_alive", "value",
)
METADATA_FIELDS = (
    "game_id", "boundary_id", "prefix_digest", "step_index", "day", "phase",
    "current_speaker", "alive_observer_ids",
)
_PINS = _Pins()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_jsonl(payload):
    rows = [_load_canonical_json(line) for line in payload.splitlines()]
    require(bool(rows) and canonical_jsonl_bytes(rows) == payload, "noncanonical/empty JSONL")
    return rows


def validate_matrix(values):
    require(isinstance(values, list) and len(values) == 7
            and all(isinstance(row, list) and len(row) == 7 for row in values), "Q shape must be 7x7")
    require(all(type(v) is float and math.isfinite(v) and 0 <= v <= 1
                for row in values for v in row), "Q cells must be finite float probabilities")
    require(all(values[i][i] == 0.0 for i in range(7)), "Q self target must be exactly zero")
    require(all(abs(math.fsum(row) - 1) <= ROW_SUM_ATOL for row in values), "Q row sum mismatch")
    result = np.array(values, dtype=np.float64)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class Review:
    root: Path
    manifest: dict
    summary: dict
    bundle_manifest: dict
    records: tuple
    prefixes: tuple
    events: tuple
    q: np.ndarray
    alive: np.ndarray
    input_files: dict


def _review_files(root, export_dir=None, canonical_dir=None):
    export = Path(export_dir).resolve(strict=True) if export_dir is not None else root / EXPORT_FOLDER
    canonical = Path(canonical_dir).resolve(strict=True) if canonical_dir is not None else root / "canonical_public"
    files = {f"export/{name}": export / name for name in (
        "manifest.json", "summary.json", "all_tom_matrices.jsonl", "tom_cells.csv")}
    files["canonical/manifest.json"] = canonical / "manifest.json"
    for name in ("authoritative_pre_prefixes.jsonl", "public_event_stream.jsonl"):
        candidates = [canonical / "public" / name, canonical / name]
        present = [p for p in candidates if os.path.lexists(p)]
        require(len(present) == 1, f"missing/ambiguous canonical public layout: {name}")
        files[f"canonical/{name}"] = present[0]
    if (root / "export_all_tom_from_canonical.py").is_file():
        files["export_script"] = root / "export_all_tom_from_canonical.py"
    return files


def input_hashes(root, *, export_dir=None, canonical_dir=None):
    result = {}
    for path in _review_files(root, export_dir, canonical_dir).values():
        name = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
        require(not path.is_symlink() and path.is_file(), f"input must be a regular file: {name}")
        payload = path.read_bytes()
        result[name] = {"byte_size": len(payload), "sha256": sha256_bytes(payload)}
    return result


def load_review(input_dir, *, export_dir=None, canonical_dir=None):
    root = Path(input_dir).expanduser().resolve(strict=True)
    files = _review_files(root, export_dir, canonical_dir)
    hashes = input_hashes(root, export_dir=export_dir, canonical_dir=canonical_dir)
    artifact = verify_artifact(files["export/manifest.json"].parent,
        expected_artifact_type=EXPORT_TYPE, expected_schema_version=EXPORT_VERSION)
    require(set(artifact.manifest["file_table"]) == {
        "summary.json", "all_tom_matrices.jsonl", "tom_cells.csv"}, "export inventory mismatch")
    summary = _load_canonical_json(files["export/summary.json"].read_bytes())
    records = read_jsonl(files["export/all_tom_matrices.jsonl"].read_bytes())
    bundle = _load_canonical_json(files["canonical/manifest.json"].read_bytes())
    require(bundle["artifact_type"] == "canonical_game_bundle"
            and bundle["schema_version"] == "classic7-canonical-game-bundle-v1", "canonical manifest schema")
    require(bundle["manifest_digest"] == sha256_bytes(canonical_json_bytes(
        {k: v for k, v in bundle.items() if k != "manifest_digest"})), "canonical manifest digest")
    require(bundle["canonical_eligibility"] is True, "source manifest lacks canonical eligibility")
    public_payloads = {}
    for name in ("authoritative_pre_prefixes.jsonl", "public_event_stream.jsonl"):
        payload = files[f"canonical/{name}"].read_bytes()
        entry = bundle["file_table"][f"public/{name}"]
        require(entry == {"byte_size": len(payload), "sha256": sha256_bytes(payload)},
                f"canonical public file digest mismatch: {name}")
        public_payloads[name] = payload
    raw_prefixes = read_jsonl(public_payloads["authoritative_pre_prefixes.jsonl"])
    prefixes = tuple(validate_authoritative_pre_prefix(_prefix_from_record(p)) for p in raw_prefixes)
    events = tuple(read_jsonl(public_payloads["public_event_stream.jsonl"]))
    require(_history_from_records(events).to_records() == list(events), "public event round-trip")
    n = len(records)
    require(len(prefixes) == n and n > 0, "matrix/PRE coverage mismatch")
    require(summary["schema_version"] == EXPORT_VERSION
            and summary["scope"] == "all_authoritative_public_speech_pre_boundaries", "summary scope/schema")
    for name, expected in (("num_pre", n), ("num_matrices", n), ("num_cells", n * 49)):
        require(type(summary[name]) is int and summary[name] == expected, f"summary {name} mismatch")
    require(summary["shape"] == [7, 7] and artifact.manifest["num_pre"] == n, "manifest shape/count")
    for name, expected in (("fit_digest", _PINS.fit), ("seal_digest", _PINS.seal)):
        require(summary[name] == artifact.manifest[name] == expected, f"frozen {name} mismatch")
    require(summary["predictor_source_revision"] == _PINS.revision, "frozen predictor source mismatch")
    require(re.fullmatch(r"[0-9a-f]{40}", bundle["source_revision"]) is not None
            and summary["gameplay_source_revision"] == bundle["source_revision"], "gameplay source mismatch")
    require(summary["source_bundle_manifest_digest"] ==
            artifact.manifest["source_bundle_manifest_digest"] == bundle["manifest_digest"],
            "export/canonical manifest lineage mismatch")
    require(summary["game_id"] == artifact.manifest["game_id"] == bundle["game_id"], "game identity mismatch")
    require(type(bundle["seed"]) is int and bundle["seed"] >= 0
            and bundle["game_id"].endswith(f"-seed-{bundle['seed']}"), "canonical seed/game identity")
    steps = [r["step_index"] for r in records]
    require(steps == sorted(set(steps)), "PRE step order must be strictly increasing")
    require(len({r["boundary_id"] for r in records}) == n, "duplicate boundary")
    matrices = []
    for record, prefix in zip(records, prefixes):
        require(set(record) == {*METADATA_FIELDS, "observer_ids", "target_ids", "q"}, "matrix record fields")
        expected = prefix.to_record()
        require(canonical_json_bytes({k: record[k] for k in METADATA_FIELDS}) ==
                canonical_json_bytes({k: expected[k] for k in METADATA_FIELDS}), "matrix/PRE identity mismatch")
        require(prefix.game_id == summary["game_id"], "PRE game identity")
        require(record["observer_ids"] == record["target_ids"] == list(PLAYER_IDS), "seat mapping mismatch")
        require(record["phase"] in ("discussion", "pk_discussion"), "not a public speech PRE")
        history = prefix.public_event_history.to_records()
        require(history == list(events[:len(history)]), "PRE is not the exact public event prefix")
        alive = set(PLAYER_IDS)
        for event in history:
            alive.difference_update(event.get("dead_players", ()))
            alive.difference_update(event.get("exiled_players", ()))
        require(tuple(p for p in PLAYER_IDS if p in alive) == prefix.alive_observer_ids, "public alive mismatch")
        matrices.append(validate_matrix(record["q"]))
    endpoints = [p.public_event_history.events[-1].event_index for p in prefixes]
    require(endpoints == sorted(set(endpoints)), "PRE public endpoints must strictly increase")
    with files["export/tom_cells.csv"].open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames == list(CSV_COLUMNS), "CSV columns mismatch")
        cells = list(reader)
    require(len(cells) == n * 49, "CSV cell count mismatch")
    index = 0
    for record in records:
        for i, observer in enumerate(PLAYER_IDS):
            for j, target in enumerate(PLAYER_IDS):
                cell = cells[index]
                expected = {name: str(record[name]) for name in CSV_COLUMNS[:6]}
                expected.update(observer=observer, target=target,
                    observer_alive=str(observer in record["alive_observer_ids"]),
                    target_alive=str(target in record["alive_observer_ids"]))
                require(set(cell) == set(CSV_COLUMNS)
                        and all(cell[name] == value for name, value in expected.items())
                        and float(cell["value"]) == record["q"][i][j], "CSV/JSONL consistency mismatch")
                index += 1
    q = np.stack(matrices)
    alive = np.array([[p in r["alive_observer_ids"] for p in PLAYER_IDS] for r in records], dtype=bool)
    q.setflags(write=False)
    alive.setflags(write=False)
    require(input_hashes(root, export_dir=export_dir, canonical_dir=canonical_dir) == hashes, "inputs changed during validation")
    return Review(root, artifact.manifest, summary, bundle, tuple(records), prefixes, events, q, alive, hashes)


def check_output_path(input_dir, output_dir):
    root = Path(input_dir).expanduser().resolve(strict=True)
    raw = Path(output_dir).expanduser().absolute()
    require(not raw.is_symlink(), "output must not be a symlink")
    destination = raw.resolve()
    require(not destination.is_relative_to(root) and not root.is_relative_to(destination),
            "output must be outside and disjoint from the input tree")
    if os.path.lexists(destination):
        raise FileExistsError(f"refusing to overwrite output: {destination}")
    return destination
