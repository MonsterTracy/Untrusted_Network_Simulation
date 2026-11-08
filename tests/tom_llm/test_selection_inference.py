from copy import deepcopy
import pytest
from tests.canonical_collection.test_structured_token_plan import _prefix
from werewolf.tom_llm.selection import select_games,validate_selection

def folds():
    games = [f"game-{i:04d}" for i in range(1500)]
    return [{"fold": f, "held_out_game_ids": games[f*300:(f+1)*300],
             "training_game_ids": [g for i, g in enumerate(games) if i//300 != f]} for f in range(5)]


def test_fixed_thirty_whole_games_per_fold_and_order_invariance():
    partition = folds()
    subset = select_games(partition, scope="subset150")
    assert len(subset["game_ids"]) == 150
    assert all(len(r["game_ids"]) == 30 for r in subset["folds"])
    assert validate_selection(subset, partition) == subset
    reversed_partition = deepcopy(partition)
    for row in reversed_partition:
        row["held_out_game_ids"].reverse()
        row["training_game_ids"].reverse()
    second = select_games(reversed_partition, scope="subset150")
    assert second["game_ids"] == subset["game_ids"]
    assert second["parent_folds_digest"] != subset["parent_folds_digest"]
    assert len(select_games(partition, scope="full1500")["game_ids"]) == 1500
    subset["game_ids"][0] = "foreign"
    with pytest.raises(ValueError):
        validate_selection(subset, partition)


@pytest.mark.parametrize("change", ["overlap", "leak", "omit", "fold", "too_small"])
def test_fold_and_selection_corruption_rejected(change):
    partition = folds()
    if change == "overlap":
        partition[1]["held_out_game_ids"][0] = partition[0]["held_out_game_ids"][0]
    elif change == "leak":
        partition[0]["training_game_ids"].append(partition[0]["held_out_game_ids"][0])
    elif change == "omit":
        partition[0]["training_game_ids"].pop()
    elif change == "fold":
        partition[0]["fold"] = True
    else:
        partition[0]["held_out_game_ids"].pop()
    with pytest.raises(ValueError):
        select_games(partition, scope="subset150")


def test_selection_artifact_is_immutable(tmp_path):
    from werewolf.tom_llm.selection import publish_selection
    partition=folds()
    artifact=publish_selection(tmp_path/"selection",partition,scope="subset150")
    assert artifact.manifest["scope"]=="subset150"
    assert artifact.manifest["selection_digest"]==select_games(partition,scope="subset150")["selection_digest"]
    with pytest.raises(ValueError,match="already exists"):
        publish_selection(tmp_path/"selection",partition,scope="subset150")
