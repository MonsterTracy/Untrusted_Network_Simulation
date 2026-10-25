"""Read-only sealed Qwen3 OOF adapter. Never constructs or calls a model."""

from dataclasses import dataclass
from pathlib import Path

from scripts import paper_backbone_contrasts as historical
from werewolf.artifact_io import verify_artifact, canonical_jsonl_bytes
from werewolf.development_publication import open_publication
from werewolf.tom.dataset import CanonicalToMDataset, ExperimentCapacity
from werewolf.tom.population import select_primary_population
from werewolf.canonical_collection.public_history import PUBLIC_PHASES
from werewolf.tom_llm.common import digest, identity, require, strict_json, valid_distribution
from werewolf.tom_llm.selection import validate_folds


@dataclass(frozen=True)
class Reference:
    seal_digest: str
    publication_digest: str
    fold_manifest_digest: str
    population_metadata: dict
    folds: list
    rows: tuple
    primary_rows: tuple
    alive_keys: frozenset
    game_ids: tuple
    bootstrap_indices: object
    bootstrap_digest: str
    prediction_digests: dict

    def selected(self, game_ids):
        games = tuple(sorted(game_ids))
        require(games and len(set(games)) == len(games) and set(games) <= set(self.game_ids),
                "reference selection has duplicate/missing/foreign games")
        rows = tuple(row for row in self.primary_rows if row["game_id"] in set(games))
        require({row["game_id"] for row in rows} == set(games), "game has zero effective Primary rows")
        return rows


def open_reference(publication_path, evaluation_root, *, expected_seal_digest):
    """External seal required; historical source gates and re-inference are not used."""
    root = Path(evaluation_root).resolve(strict=True)
    report, prepared, training_root, train_seal, primary = historical._inputs(root, expected_seal_digest)
    games, fold_games, indices, bootstrap_digest = historical._bootstrap(report, prepared.manifest["folds"])
    require(validate_folds(prepared.manifest["folds"]) == games, "reference game/fold mismatch")
    publication = open_publication(publication_path)
    inputs = prepared.manifest["protocol_inputs"]
    require(publication.manifest_digest == inputs["publication_digest"]
            and publication.public_view.fold_manifest.manifest_digest == inputs["fold_manifest_digest"],
            "reference publication/fold binding mismatch")
    population = select_primary_population(publication)
    require(population.metadata == inputs["primary_identity"] and population.rows == primary,
            "reference Primary/selector mismatch")
    actual_folds = {fold.fold_index: set(fold.game_ids) for fold in publication.public_view.fold_manifest.folds}
    require(actual_folds == {i: set(ids) for i, ids in enumerate(fold_games)}, "reference publication fold mismatch")
    require(tuple(sorted(publication.public_view.game_ids)) == tuple(games), "reference publication game coverage mismatch")
    descriptors = {(r["architecture"], r["fold"]): r for r in report.manifest["predictions"]}
    terminals = {(r["architecture"], r["fold"]): r for r in train_seal.manifest["terminals"]}
    require(len(descriptors) == len(report.manifest["predictions"])
            and len(terminals) == len(train_seal.manifest["terminals"]), "duplicate lineage descriptor")
    expected_lineages = {(a, f) for a in historical.ARCHITECTURES for f in range(5)}
    require(set(descriptors) == set(terminals) == expected_lineages
            and all(t["temporal_condition"] == "explicit_day_phase" for t in train_seal.manifest["terminals"]),
            "reference lineage inventory mismatch")
    all_rows, primary_rows, alive_keys, prediction_digests = [], [], set(), {}
    for fold, ids in enumerate(fold_games):
        key = "qwen3", fold
        require(key in descriptors and key in terminals, "missing Qwen3 fold descriptor")
        descriptor = descriptors[key]
        terminal = verify_artifact(training_root / "runs/qwen3" / str(fold) / "terminal",
            expected_artifact_type="backbone_terminal", expected_schema_version=historical.TERMINAL_VERSION)
        artifact = verify_artifact(root / "folds/qwen3" / str(fold),
            expected_artifact_type="backbone_oof_fold_predictions", expected_schema_version=historical.PREDICTION_VERSION)
        m = artifact.manifest
        require(terminal.manifest_digest == descriptor["terminal_digest"] == terminals[key]["terminal_digest"]
                and artifact.manifest_digest == descriptor["prediction_digest"]
                and m["architecture"] == "qwen3" and type(m["fold"]) is int and m["fold"] == fold
                and m["temporal_condition"] == "explicit_day_phase"
                and m["evaluation_digest"] == report.manifest["evaluation_digest"]
                and m["training_seal_digest"] == report.manifest["training_seal_digest"]
                and m["terminal_digest"] == terminal.manifest_digest
                and m["checkpoint_digest"] == terminal.manifest["checkpoint_ancestry"][-1]["digest"]
                and set(m["file_table"]) == {"predictions.jsonl"}, "reference fold/terminal lineage mismatch")
        raw = (artifact.path / "predictions.jsonl").read_bytes()
        rows = [strict_json(line) for line in raw.splitlines()]
        dataset = CanonicalToMDataset(publication.public_view, ids, ExperimentCapacity(1024))
        require(raw == canonical_jsonl_bytes(rows) and len(rows) == m["row_count"] == len(dataset) * 7,
                "reference prediction coverage/serialization mismatch")
        selected_ids = []
        for sample, batch in zip(dataset, (rows[i:i+7] for i in range(0, len(rows), 7)), strict=True):
            for observer, row in enumerate(batch):
                semantic = {"game_id": sample.game_id, "boundary_id": sample.boundary_id, "observer": observer,
                    "prefix_digest": sample.prefix_digest, "plan_digest": sample.plan_digest,
                    "q": sample.q[observer].tolist(), "label_observed": bool(sample.label_observed[observer]),
                    "observer_alive": bool(sample.observer_alive[observer]),
                    "checkpoint_digest": m["checkpoint_digest"], "temporal_condition": "explicit_day_phase"}
                require(set(row) == historical.PREDICTION_FIELDS and type(row["observer"]) is int
                        and type(row["label_observed"]) is bool and type(row["observer_alive"]) is bool
                        and all(type(v) in (int, float) for v in row["q"])
                        and all(row[k] == v for k, v in semantic.items()), "reference PRE/observer/q semantics mismatch")
                valid_distribution(row)
                entry = {**row, "fold": fold, "canonical_shift": 0,
                         "primary_eligible": primary[sample.game_id][sample.boundary_id][observer],
                         "reference_prediction_digest": artifact.manifest_digest,
                         "public_token_count": sample.token_count,
                         "public_phase": PUBLIC_PHASES[int(sample.public.phase_ids[sample.token_count - 1])]}
                all_rows.append(entry)
                if row["observer_alive"]:
                    alive_keys.add(identity(entry))
                if row["label_observed"] and entry["primary_eligible"]:
                    require(row["observer_alive"], "dead Primary observer")
                    primary_rows.append(entry)
                    selected_ids.append([sample.game_id, sample.boundary_id, observer])
        require(digest(selected_ids) == m["primary_row_identity_digest"]
                == report.manifest["primary_row_bindings"][str(fold)], "reference Primary row identity mismatch")
        prediction_digests[str(fold)] = artifact.manifest_digest
    require(len({identity(r) for r in all_rows}) == len(all_rows), "duplicate reference row")
    return Reference(report.manifest_digest, publication.manifest_digest,
        publication.public_view.fold_manifest.manifest_digest, population.metadata, prepared.manifest["folds"],
        tuple(all_rows), tuple(primary_rows), frozenset(alive_keys), tuple(games), indices,
        bootstrap_digest, prediction_digests)
