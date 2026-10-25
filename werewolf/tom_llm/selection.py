"""Selection depends only on frozen whole-game fold identities."""

from werewolf.tom_llm.common import digest, require
from werewolf.tom_llm.policy import POLICY


def validate_folds(folds):
    require(type(folds) in (list, tuple) and len(folds) == 5, "exactly five folds required")
    games = []
    for index, row in enumerate(folds):
        require(type(row) is dict and row.get("fold") == index and type(row["fold"]) is int,
                "fold indices must be canonical 0..4")
        held = row["held_out_game_ids"]
        require(type(held) is list and held and all(type(g) is str and g for g in held)
                and len(set(held)) == len(held), "invalid held-out games")
        games.extend(held)
    require(len(set(games)) == len(games), "game overlap across folds")
    all_games = set(games)
    for row in folds:
        train = row["training_game_ids"]
        require(type(train) is list and len(set(train)) == len(train)
                and set(train) == all_games - set(row["held_out_game_ids"]), "training/held-out isolation mismatch")
    return sorted(games)


def select_games(folds, *, scope):
    games = validate_folds(folds)
    require(scope in ("full1500", "subset150"), "unknown selection scope")
    require(len(games) == 1500 and all(len(row["held_out_game_ids"]) == 300 for row in folds),
            "selection requires the original 1500-game partition")
    settings = POLICY["subset"]
    records = []
    for row in folds:
        selected = row["held_out_game_ids"]
        if scope == "subset150":
            selected = sorted(selected, key=lambda g: (digest([settings["version"], settings["seed"], g]), g))[:30]
        records.append({"fold": row["fold"], "game_ids": sorted(selected)})
    result = {"scope": scope, "rule": dict(settings), "folds": records,
              "game_ids": sorted(g for row in records for g in row["game_ids"]), "parent_folds_digest": digest(folds)}
    return {**result, "selection_digest": digest(result)}


def validate_selection(selection, folds):
    require(selection == select_games(folds, scope=selection["scope"]), "selection identity/rule mismatch")
    return selection


def publish_selection(destination, folds, *, scope):
    """Freeze identities before probabilities/labels are inspected; never overwrite."""
    from pathlib import Path
    from werewolf.artifact_io import publish_artifact, canonical_json_bytes
    selection = select_games(folds, scope=scope)
    require(not Path(destination).exists(), "selection destination already exists")
    return publish_artifact(destination, manifest_fields={
        "artifact_type": "llm_game_selection", "schema_version": "classic7_llm_game_selection_v1",
        "scope": scope, "selection_digest": selection["selection_digest"],
        "parent_folds_digest": selection["parent_folds_digest"],
    }, files={"selection.json": canonical_json_bytes(selection)})
