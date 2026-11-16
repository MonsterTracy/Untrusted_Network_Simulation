"""Human-checkable D2 mathematics; all rows below are synthetic."""
from collections import Counter
from copy import deepcopy
from dataclasses import replace
import json
import math

import pytest

from werewolf.artifact_io import ArtifactConflictError, ArtifactValidationError, canonical_json_bytes
from werewolf.observer_dtrs.d2_protocol import frozen_protocol
from werewolf.observer_dtrs.estimation import (
    FittedModel, TrainingRow, cell_stats, condition, fit_model, validate_model,
)
from werewolf.observer_dtrs.estimation_offline import publish_model, read_model, implementation_digest
from werewolf.observer_dtrs.features import ObserverFeatures
from werewolf.observer_dtrs.scoring import paired_difference, score_predictions, select_parameters


@pytest.fixture
def protocol():
    return frozen_protocol()


def features(r=.3, phase="discussion", nW=2):
    return ObserverFeatures(s=r if nW == 1 else r / 2, m=0 if nW == 1 else r / 2,
                            r=r, dW=None if nW == 1 else 0, phase=phase, nW=nW)


def row(game, y, r=.3, *, sample=None, phase="discussion", nW=2):
    return TrainingRow(sample or game, game, features(r, phase, nW), y)


def fit(rows, protocol, *, model="OBS-EQ1", bins=3, smoothing=1, weighting="game_equal"):
    rows = tuple(rows)
    source = {"scope": "controlled_fixture", "protocol_digest": protocol.digest,
        "dataset_digest": None, "split_manifest_digest": None,
        "tom_fold": None, "outer_fold": None, "inner_fold": None,
        "game_sizes": dict(Counter(r.game_id for r in rows)),
        "origins": {"fixture": "hand_checkable_not_formal_data"}, "implementation_digest": implementation_digest()}
    return fit_model(rows, protocol=protocol, model_id=model, bins=bins, smoothing=smoothing,
                     source=source, weighting=weighting)


def test_exact_weighted_counts_kish_and_hierarchical_formula(protocol):
    rows = [row("large", 1, sample=f"l{j}") for j in range(9)] + [row("small", 0)]
    m = fit(rows, protocol, model="OBS-EQ0", bins=1)
    root = m.record["root"]
    assert root["stats"]["mass"] == pytest.approx(2)
    assert root["stats"]["positive_mass"] == pytest.approx(1)
    assert root["stats"]["kish_games"] == pytest.approx(2)
    assert root["probability"] == pytest.approx(.5)
    assert m.predict(features()) == pytest.approx(.5)
    rec = fit(rows, protocol, model="OBS-EQ0", bins=1, weighting="record_equal")
    assert rec.record["root"]["stats"]["mass"] == pytest.approx(2)
    assert rec.record["root"]["stats"]["positive_mass"] == pytest.approx(1.8)
    assert rec.record["root"]["probability"] == pytest.approx(2.3 / 3)
    # Full-game weights do not get re-normalized inside a subgroup/cell.
    stats = cell_stats([rows[0], rows[-1]], {**{f"l{j}": 1 / 9 for j in range(9)}, "small": 1})
    assert stats.mass == pytest.approx(10 / 9)
    assert stats.positive_mass == pytest.approx(1 / 9)
    assert stats.kish_games == pytest.approx((10 / 9) ** 2 / (1 + 1 / 81))
    assert cell_stats([], {}).kish_games == 0
    validate_model(rec, protocol)


def test_root_prior_and_exact_two_level_shrinkage(protocol):
    rows = [row("a", 1, nW=1), row("b", 0, phase="pk_discussion", nW=2)]
    m = fit(rows, protocol, model="OBS-EQ0", bins=1)
    assert m.record["root"]["probability"] == .5
    assert m.record["wolf_nodes"]["1"]["probability"] == .75
    assert m.predict(features(nW=1)) == .875
    assert m.predict(features(phase="pk_discussion", nW=2)) == .125
    validate_model(m, protocol)


@pytest.mark.parametrize("smoothing", (.25, 1, 4))
def test_eq1_one_bin_exact_eq0_predictions_no_extra_shrinkage(protocol, smoothing):
    rows = [row(f"g{j}", j % 2, .1 + j / 100, phase="pk_discussion" if j % 3 else "discussion",
                nW=1 if j % 4 else 2) for j in range(40)]
    m0 = fit(rows, protocol, model="OBS-EQ0", bins=1, smoothing=smoothing)
    m1 = fit(rows, protocol, bins=1, smoothing=smoothing)
    assert m0.record["state_nodes"] == m1.record["state_nodes"]
    for phase in ("discussion", "pk_discussion"):
        for nW in (1, 2):
            assert m0.predict(features(.9, phase, nW)) == m1.predict(features(.9, phase, nW))


def test_quantiles_ties_boundary_endpoints_and_single_class_leaves(protocol):
    rows = [row(f"g{j:02}", int(j >= 10), (.1, .4, .8)[j // 10]) for j in range(30)]
    m = fit(rows, protocol)
    state = m.record["state_nodes"]["discussion/2"]
    assert state["edges"] == [.1, .4]
    assert [leaf["stats"]["games"] for leaf in state["leaves"]] == [10, 10, 10]
    assert [leaf["stats"]["positive_mass"] for leaf in state["leaves"]] == [0, 10, 10]
    assert 0 < m.predict(features(.1)) < m.predict(features(math.nextafter(.1, 1))) < 1
    assert m.predict(features(0)) == m.predict(features(.1))
    assert m.predict(features(1)) == m.predict(features(.8))
    # All-equal r never gets artificially split, including endpoint atoms.
    constant = fit([replace(r, features=features(1)) for r in rows], protocol, bins=5)
    assert constant.record["state_nodes"]["discussion/2"]["edges"] == []
    validate_model(m, protocol)


def test_sparse_merge_x_only_deterministic_and_one_leaf_is_parent(protocol):
    rows = [row(f"g{j:02}", j % 2, (.1, .4, .8)[j // 6]) for j in range(18)]
    m = fit(rows, protocol)
    state = m.record["state_nodes"]["discussion/2"]
    assert state["edges"] == []
    assert state["leaves"] == []
    eq0 = fit(rows, protocol, model="OBS-EQ0", bins=1)
    assert m.predict(features(.9)) == eq0.predict(features(.9))
    changed = fit([replace(r, y=1 - r.y) for r in rows], protocol)
    assert changed.record["state_nodes"]["discussion/2"]["edges"] == state["edges"]


def test_kish_gate_prevents_many_low_mass_games_looking_like_strong_support(protocol):
    # 10 distinct games in a low-r cell, but mass is dominated by one game.
    rows = [row("dominant", 1, .1)]
    for g in range(9):
        rows += [row(f"long{g}", 0, .1 if j == 0 else .8, sample=f"long{g}-{j}") for j in range(100)]
    stats = cell_stats([r for r in rows if r.features.r == .1],
                       {r.sample_id: 1 if r.game_id == "dominant" else .01 for r in rows})
    assert stats.games == 10 and stats.kish_games < 5
    m = fit(rows, protocol, bins=5)
    assert m.record["state_nodes"]["discussion/2"]["edges"] == []


def test_missing_stratum_and_wolf_state_use_nearest_observed_parent(protocol):
    m = fit([row("a", 1, nW=1)], protocol)
    assert m.predict_detail(features(phase="pk_discussion", nW=1))["level"] == "nW"
    assert m.predict_detail(features(nW=2))["level"] == "global"
    assert 0 < m.predict(features(nW=2)) < 1
    with pytest.raises(ValueError):
        fit([], protocol)


def test_external_labels_and_covariates_cannot_enter_training_or_prediction(protocol):
    train = [row(f"train{j}", j % 2, (.1, .4, .8)[j // 10]) for j in range(30)]
    held_out = [row("test", 0, .2), row("validation", 1, .99)]
    first = fit(train, protocol)
    first_prediction = first.predict(held_out[0].features)
    for test_y in (0, 1):
        changed = [replace(held_out[0], y=test_y), replace(held_out[1], features=features(.001))]
        # The same training partition, not a re-hashed/corrupted source artifact.
        refit = fit(train, protocol)
        assert refit.payload == first.payload
        assert refit.predict(changed[0].features) == first_prediction
        refit.predict(changed[1].features)
        assert refit.payload == first.payload
    with pytest.raises(TypeError):
        first.predict(held_out[0])


@pytest.mark.parametrize("bad", (
    features(float("nan")), features(-.1), features(1.1),
    replace(features(), phase="speech"), replace(features(), nW=True),
    replace(features(nW=1), dW=0), replace(features(), dW=None),
    replace(features(), r=.8),
))
def test_invalid_or_runtime_attributes_rejected(bad):
    with pytest.raises(ValueError):
        condition(bad)


def test_prediction_validation_no_renormalization_of_q_sum_tolerance(protocol):
    # D1 permits total probability mass within 1e-6 of one; do not alter r.
    m = fit([row("g", 1)], protocol)
    f = ObserverFeatures(.5000002, .5000002, 1.0000004, 0, "discussion", 2)
    assert 0 < m.predict(f) < 1


def test_game_scores_pairing_and_extreme_logloss(protocol):
    rows = [row("large", 1, sample=f"l{j}") for j in range(9)] + [row("small", 0)]
    pred = {r.sample_id: 0. for r in rows}
    scores = score_predictions(rows, pred)
    assert scores["game_equal"]["brier"] == .5
    assert scores["record_equal"]["brier"] == .9
    assert scores["wrong_endpoint_rows"] == 9
    assert scores["clipped_rows"] == 10
    assert scores["game_equal"]["log_loss"] == pytest.approx((-math.log(1e-12) - math.log1p(-1e-12)) / 2)
    other = score_predictions(rows, {r.sample_id: .5 for r in rows})
    assert paired_difference(scores, other)["brier"] == -.25
    with pytest.raises(ValueError):
        paired_difference(scores, score_predictions(rows[:-1], {r.sample_id: .5 for r in rows[:-1]}))
    for value in (float("nan"), -.1, 1.1, True):
        with pytest.raises(ValueError):
            score_predictions(rows, {r.sample_id: value for r in rows})


def test_inner_selection_complete_paired_grids_and_frozen_ties(protocol):
    grid = {(b, l): {"a": .2, "b": .3} for b in (1, 3, 5) for l in (.25, 1, 4)}
    assert select_parameters(grid, protocol=protocol, model_id="OBS-EQ1") == (1, 4)
    eq0 = {key: value for key, value in grid.items() if key[0] == 1}
    assert select_parameters(eq0, protocol=protocol, model_id="OBS-EQ0") == (1, 4)
    grid[(3, .25)] = {"a": .1, "b": .2}
    assert select_parameters(grid, protocol=protocol, model_id="OBS-EQ1") == (3, .25)
    grid[(5, 4)] = {"different": .2}
    with pytest.raises(ValueError):
        select_parameters(grid, protocol=protocol, model_id="OBS-EQ1")


def test_model_artifact_round_trip_source_configuration_and_damage(protocol, tmp_path):
    m = fit([row(f"g{j}", j % 2) for j in range(20)], protocol)
    a = publish_model(tmp_path / "model", m, protocol=protocol)
    loaded = read_model(a.path, expected_manifest_digest=a.manifest_digest, protocol=protocol)
    assert loaded.payload == m.payload
    assert loaded.record["source"]["scope"] == "controlled_fixture"
    assert loaded.record["parameters"]["weighting"] == "game_equal"
    assert publish_model(a.path, m, protocol=protocol).manifest_digest == a.manifest_digest
    changed = fit([row(f"g{j}", 1) for j in range(20)], protocol)
    with pytest.raises(ArtifactConflictError):
        publish_model(a.path, changed, protocol=protocol)
    for edit in (lambda x: x["root"].update(probability=.99),
                 lambda x: x["state_nodes"]["discussion/2"]["stats"].update(kish_games=999),
                 lambda x: x["parameters"].update(requested_bins=99)):
        bad = deepcopy(m.record)
        edit(bad)
        with pytest.raises(ValueError):
            validate_model(FittedModel(canonical_json_bytes(bad)), protocol)
    (a.path / "model.json").write_bytes(b"corrupt")
    with pytest.raises(ArtifactValidationError):
        read_model(a.path, expected_manifest_digest=a.manifest_digest, protocol=protocol)


def test_sparse_merging_stops_at_supported_leaves(protocol):
    # Five requested 8-game leaves: merge leftmost failing, lower-mass neighbor.
    rows = [row(f"g{j:02}", j % 2, (j + 1) / 100) for j in range(40)]
    m = fit(rows, protocol, bins=5)
    node = m.record["state_nodes"]["discussion/2"]
    assert node["edges"] == [.16]
    assert [leaf["stats"]["games"] for leaf in node["leaves"]] == [16, 24]
    assert m.payload == fit(reversed(rows), protocol, bins=5).payload


def test_game_weights_change_training_quantiles_without_rescaling_within_state(protocol):
    # 30 short low-r games vs 30 long high-r games; game mass is 30 vs 30.
    rows = [row(f"low{j:02}", 0, .1) for j in range(30)]
    rows += [row(f"high{j:02}", 1, .8, sample=f"high{j:02}-{h}") for j in range(30) for h in range(10)]
    game = fit(rows, protocol)
    record = fit(rows, protocol, weighting="record_equal")
    assert game.record["state_nodes"]["discussion/2"]["edges"] == [.1]
    assert record.record["state_nodes"]["discussion/2"]["edges"] == []
    assert game.record["root"]["stats"]["mass"] == pytest.approx(60)
    assert record.record["root"]["stats"]["mass"] == pytest.approx(60)


def test_incomplete_training_games_and_out_of_protocol_parameters_rejected(protocol):
    rows = [row("g", 1), row("h", 0)]
    m = fit(rows, protocol)
    source = deepcopy(m.record["source"])
    source["game_sizes"]["g"] += 1
    with pytest.raises(ValueError):
        fit_model(rows, protocol=protocol, model_id="OBS-EQ0", bins=1, smoothing=1, source=source)
    for options in ({"bins": 2}, {"smoothing": 0}, {"model": "Phase2-M0"}, {"model": "OBS-EQ0", "bins": 3}):
        with pytest.raises(ValueError):
            fit(rows, protocol, **options)


def test_single_class_extreme_training_is_finite_without_output_clip(protocol):
    for y in (0, 1):
        m = fit([row(f"g{j}", y) for j in range(50)], protocol)
        p = m.predict(features())
        assert 0 < p < 1
        assert score_predictions([row("test", 1-y)], {"test": p})["clipped_rows"] == 0


def test_record_sensitivity_selects_for_its_own_measure_on_same_games(protocol):
    grid = {(1, .25): {"big": .1, "small": .9},
            (1, 1): {"big": .4, "small": .2}, (1, 4): {"big": .8, "small": .8}}
    assert select_parameters(grid, protocol=protocol, model_id="OBS-EQ0") == (1, 1)
    assert select_parameters(grid, protocol=protocol, model_id="OBS-EQ0", weighting="record_equal",
                             game_sizes={"big": 9, "small": 1}) == (1, .25)
    with pytest.raises(ValueError):
        select_parameters(grid, protocol=protocol, model_id="OBS-EQ0", weighting="record_equal")


def test_controlled_acceptance_cli_runs_without_any_formal_data(tmp_path):
    import subprocess
    import sys
    result = subprocess.run([sys.executable, '-B', 'scripts/check_observer_dtrs_estimator.py',
                             '--destination', str(tmp_path / 'controlled-cli')],
                            text=True, capture_output=True, check=True)
    report = json.loads(result.stdout)
    assert report['scope'] == 'controlled_fixture_not_performance_validation'
    assert report['predictions'] == {'controlled-a': .875, 'controlled-b': .125}
    assert report['scores']['game_equal']['brier'] == .015625


@pytest.fixture
def final_selections(protocol):
    """75个人工游戏只用于评分归集，不读取或拟合正式数据。"""
    from werewolf.observer_dtrs.validation_data import controlled_fixture
    from werewolf.observer_dtrs.splits import partition_games
    data = controlled_fixture(protocol)
    selections = []
    for k in range(5):
        for j in range(5):
            train, _ = partition_games(data.split, tom_fold=k, outer_fold=j)
            entries = []
            for name in ("OBS-EQ1", "OBS-EQ0"):
                for b in ((1, 3, 5) if name == "OBS-EQ1" else (1,)):
                    for l in (.25, 1, 4):
                        base = .1 if (name, b, l) == ("OBS-EQ1", 3, 1) else .3
                        entries.append({"model": name, "parameters": {"bins": b, "smoothing": l},
                            "games": {g: base + j / 100 + int(g.rsplit('-', 1)[1]) / 10000 for g in train}})
            # 历史LR及局部最佳参数故意与本轮全局选择相冲突。
            entries.append({"model": "OBS-LR-r", "parameters": {"tau": 1}, "games": {g: 0 for g in train}})
            selections.append(({"tom_fold": k, "outer_fold": j, "weighting": "game_equal"},
                {"candidate_game_brier": entries, "selected": {"OBS-EQ1": {"bins": 5, "smoothing": 4}},
                 "candidate_artifact_digests": ["a" * 64]}))
    return data, selections


def test_final_inner_scores_average_four_per_game_and_ignore_record_order(protocol, final_selections):
    from werewolf.observer_dtrs.estimation_offline import aggregate_inner_scores
    data, selections = final_selections
    table = aggregate_inner_scores(selections, split_record=data.split, game_folds=data.folds, protocol=protocol)
    game = sorted(data.folds)[0]
    values = [r["candidate_game_brier"][0]["games"][game]
              for _, r in selections if game in r["candidate_game_brier"][0]["games"]]
    assert len(values) == 4
    assert table["OBS-EQ1"][1, .25][game] == math.fsum(values) / 4
    assert set(table) == {"OBS-EQ0", "OBS-EQ1"}
    assert len(table["OBS-EQ0"]) == 3 and len(table["OBS-EQ1"]) == 9
    assert select_parameters(table["OBS-EQ1"], protocol=protocol, model_id="OBS-EQ1") == (3, 1)
    assert select_parameters(table["OBS-EQ0"], protocol=protocol, model_id="OBS-EQ0") == (1, 4)
    changed = deepcopy(list(reversed(selections)))
    for _, record in changed:
        record["candidate_game_brier"].reverse()
        for entry in record["candidate_game_brier"]:
            entry["games"] = dict(reversed(list(entry["games"].items())))
    assert aggregate_inner_scores(changed, split_record=data.split, game_folds=dict(reversed(list(data.folds.items()))),
                                   protocol=protocol) == table


@pytest.mark.parametrize("bad", ("missing_block", "duplicate_block", "record_equal", "missing_candidate",
    "duplicate_candidate", "missing_game", "extra_game", "wrong_source", "invalid_parameter",
    "nan", "inf", "negative", "above_one", "boolean", "text"))
def test_final_inner_score_rejections(protocol, final_selections, bad):
    from werewolf.observer_dtrs.estimation_offline import aggregate_inner_scores
    data, selections = final_selections
    identity, record = selections[0]
    entries = record["candidate_game_brier"]
    games = entries[0]["games"]
    if bad == "missing_block": selections.pop()
    elif bad == "duplicate_block": selections.append(deepcopy(selections[0]))
    elif bad == "record_equal": identity["weighting"] = "record_equal"
    elif bad == "missing_candidate": entries.pop(0)
    elif bad == "duplicate_candidate": entries.append(deepcopy(entries[0]))
    elif bad == "missing_game": games.pop(next(iter(games)))
    elif bad == "extra_game": games["outside"] = .2
    elif bad == "wrong_source": identity["tom_fold"] = 1
    elif bad == "invalid_parameter": entries[0]["parameters"]["bins"] = True
    else:
        games[next(iter(games))] = {"nan": float("nan"), "inf": float("inf"), "negative": -.1,
                                  "above_one": 1.1, "boolean": True, "text": "0.2"}[bad]
    with pytest.raises(ValueError):
        aggregate_inner_scores(selections, split_record=data.split, game_folds=data.folds, protocol=protocol)


def test_final_inner_scores_reject_five_appearances(protocol, final_selections):
    from werewolf.observer_dtrs.estimation_offline import aggregate_inner_scores
    data, selections = final_selections
    game = sorted(data.folds)[0]
    data.split["folds"][0]["outer_assignment"][game] = None
    for identity, record in selections:
        if identity["tom_fold"] == 0:
            for entry in record["candidate_game_brier"]:
                entry["games"][game] = .2
    with pytest.raises(ValueError, match="exactly four"):
        aggregate_inner_scores(selections, split_record=data.split, game_folds=data.folds, protocol=protocol)


def _selection_run(tmp_path, protocol, data, selections, *, bad=None):
    """仅发布小型人工引用链；不需要内层模型、外层成绩或Bootstrap。"""
    from werewolf.observer_dtrs.validation_artifacts import digest, publish_unit
    from werewolf.observer_dtrs.splits import partition_games
    manifest = {"manifest_digest": protocol.record["dataset_digest"], "origins": {"fixture": "synthetic"},
        "games": [{"game_id": g, "fold_index": data.folds[g], "row_count": n} for g, n in data.sizes.items()]}
    c = {"scope": "formal_development_validation", "protocol_digest": protocol.digest,
         "dataset_digest": manifest["manifest_digest"], "origins": manifest["origins"],
         "game_sizes": data.sizes, "game_folds": data.folds, "split_record": data.split, "split_digest": "b" * 64}
    if bad == "fixture_scope": c["scope"] = "controlled_fixture"
    contract = publish_unit(tmp_path / "contract", "validation_contract", {"run_identity": digest(c)}, c)
    blocks = {}
    for identity, record in selections:
        k, j = identity["tom_fold"], identity["outer_fold"]
        common = {"contract_digest": contract.manifest_digest, "tom_fold": k, "outer_fold": j}
        base = tmp_path / "blocks" / f"{k}-{j}"
        sid = {**common, "weighting": "record_equal" if bad == "weighting" else "game_equal",
               "candidate_artifact_digests": record["candidate_artifact_digests"]}
        selection = publish_unit(base / "game_equal" / "selection", "validation_selection", sid, record)
        train, test = partition_games(data.split, tom_fold=k, outer_fold=j)
        block = publish_unit(base / "complete", "validation_block_complete", common,
            {"status": "PAIRED_BLOCK_COMPLETE", "train_games": train, "test_games": test,
             "references": {"game_equal": {"selection": selection.manifest_digest}}})
        blocks[f"{k}-{j}"] = block.manifest_digest
    complete = publish_unit(tmp_path / "complete", "validation_complete",
        {"contract_digest": contract.manifest_digest, "block_digests": blocks},
        {"status": "ALL_BLOCKS_COMPLETE", "block_digests": blocks, "summary": {"not_used_for_selection": True}})
    return manifest, complete.manifest_digest


def test_final_selection_reads_pinned_chain_and_retains_original_ties(tmp_path, protocol, final_selections):
    from werewolf.observer_dtrs.estimation_offline import select_final_parameters
    data, selections = final_selections
    for _, record in selections:
        for entry in record["candidate_game_brier"]:
            entry["games"] = {g: .2 for g in entry["games"]}
            if entry["model"] == "OBS-EQ1" and entry["parameters"] == {"bins": 3, "smoothing": .25}:
                entry["games"] = {g: .2 - 5e-9 for g in entry["games"]}
    manifest, complete = _selection_run(tmp_path, protocol, data, selections)
    result = select_final_parameters(tmp_path, expected_complete_digest=complete, expected_split_digest="b" * 64,
                                    dataset_manifest=manifest, split_record=data.split, protocol=protocol)
    assert result["selected_parameters"] == {name: {"bins": 1, "smoothing": 4} for name in ("OBS-EQ1", "OBS-EQ0")}
    assert len(result["selection_digests"]) == 25
    assert result["complete_digest"] == complete
    assert len(result["aggregated_scores_digest"]) == 64
    with pytest.raises(ValueError, match="identity"):
        select_final_parameters(tmp_path, expected_complete_digest="0" * 64, expected_split_digest="b" * 64,
                                dataset_manifest=manifest, split_record=data.split, protocol=protocol)
    path = tmp_path / "blocks/0-0/game_equal/selection/record.json"
    path.write_bytes(b"damaged")
    with pytest.raises(ArtifactValidationError):
        select_final_parameters(tmp_path, expected_complete_digest=complete, expected_split_digest="b" * 64,
                                dataset_manifest=manifest, split_record=data.split, protocol=protocol)


@pytest.mark.parametrize("bad", ("fixture_scope", "weighting"))
def test_final_selection_rejects_wrong_run_scope_or_weighting(tmp_path, protocol, final_selections, bad):
    from werewolf.observer_dtrs.estimation_offline import select_final_parameters
    data, selections = final_selections
    manifest, complete = _selection_run(tmp_path, protocol, data, selections, bad=bad)
    with pytest.raises(ValueError):
        select_final_parameters(tmp_path, expected_complete_digest=complete, expected_split_digest="b" * 64,
                                dataset_manifest=manifest, split_record=data.split, protocol=protocol)


def _final_source_fixture(protocol):
    """1500局来源字段的人工元数据夹具，不含真实Q、标签或拟合产物。"""
    from werewolf.observer_dtrs.estimation import FINAL_SELECTION_RULE
    from werewolf.artifact_io import sha256_bytes
    ids = [f"synthetic-{k}-{j:03}" for k in range(5) for j in range(300)]
    pins = ("evaluation_digest", "evaluation_seal_digest", "preparation_digest", "training_execution_digest",
            "training_seal_digest", "terminal_digest", "checkpoint_digest", "prediction_artifact_digest", "prediction_payload_sha256")
    selection = {"rule": FINAL_SELECTION_RULE, "run_path": "/synthetic",
        "contract_digest": "c" * 64, "complete_digest": "d" * 64, "aggregated_scores_digest": "e" * 64,
        "selection_digests": {f"{k}-{j}": sha256_bytes(f"{k}-{j}".encode()) for k in range(5) for j in range(5)},
        "selected_parameters": {name: {"bins": 1, "smoothing": 4} for name in ("OBS-EQ1", "OBS-EQ0")}}
    return {"scope": "final_development_oof", "protocol_digest": protocol.digest,
        "dataset_digest": protocol.record["dataset_digest"], "split_manifest_digest": "b" * 64,
        "tom_fold": None, "outer_fold": None, "inner_fold": None,
        "game_sizes": {g: 16 if i < 185 else 15 for i, g in enumerate(ids)},
        "game_folds": {g: int(g.split('-')[1]) for g in ids},
        "origins": {"publication": {"manifest_digest": "a" * 64, "fold_manifest_digest": "f" * 64},
                    "oof": [{"fold_index": k, "architecture": "qwen3", "temporal_condition": "explicit_day_phase",
                             "publication_digest": "a" * 64, "fold_manifest_digest": "f" * 64,
                             **{p: str(k) * 64 for p in pins}} for k in range(5)]},
        "selection": selection, "fit_revision": "0" * 40, "implementation_digest": implementation_digest()}


def test_final_source_is_explicit_and_cannot_claim_historical_partitions(protocol):
    from werewolf.observer_dtrs.estimation import _validate_source
    source = _final_source_fixture(protocol)
    _validate_source(source, protocol)
    assert len(source["game_folds"]) == 1500 and sum(source["game_sizes"].values()) == 22685
    assert [s["fold_index"] for s in source["origins"]["oof"]] == list(range(5))
    for change in (lambda s: s.update(tom_fold=0), lambda s: s["game_sizes"].pop(next(iter(s["game_sizes"]))),
                   lambda s: s["origins"]["oof"].pop(), lambda s: s["origins"]["oof"][0].pop("checkpoint_digest"),
                   lambda s: s["origins"]["oof"][0].update(publication_digest="1" * 64),
                   lambda s: s.update(scope="formal_training_partition")):
        bad = deepcopy(source)
        change(bad)
        with pytest.raises(ValueError): _validate_source(bad, protocol)
    # 既有160/240局正式分区仍沿用原规则，不能被全量来源分支替代。
    historical = {k: v for k, v in source.items() if k not in {"game_folds", "selection", "fit_revision"}}
    historical.update(scope="formal_training_partition", tom_fold=0, outer_fold=0)
    for count, inner in ((240, None), (160, 0)):
        historical.update(game_sizes={f"g{i}": 1 for i in range(count)}, inner_fold=inner)
        _validate_source(historical, protocol)
        historical["game_sizes"]["extra"] = 1
        with pytest.raises(ValueError): _validate_source(historical, protocol)
    with pytest.raises(ValueError, match="all rows"):
        fit_model([row("g", 1)], protocol=protocol, model_id="OBS-EQ1", bins=1, smoothing=4, source=source)


def test_final_model_rejects_parameters_or_weighting_not_selected(protocol):
    original = fit([row("g", 1)], protocol)
    for parameters in ({"requested_bins": 3, "lambda": 4, "weighting": "game_equal"},
                       {"requested_bins": 1, "lambda": 1, "weighting": "game_equal"},
                       {"requested_bins": 1, "lambda": 4, "weighting": "record_equal"}):
        record = deepcopy(original.record)
        record.update(source=_final_source_fixture(protocol), parameters=parameters)
        with pytest.raises(ValueError, match="differ from its inner selection"):
            validate_model(FittedModel(canonical_json_bytes(record)), protocol)


def test_final_fit_adapter_selects_first_and_reuses_existing_kernel(protocol, monkeypatch):
    from types import SimpleNamespace
    from werewolf.observer_dtrs import estimation_offline as module
    source = _final_source_fixture(protocol)
    games = [{"game_id": g, "row_count": n, "fold_index": source["game_folds"][g]} for g, n in source["game_sizes"].items()]
    m = {"manifest_digest": source["dataset_digest"], "games": games, "origins": source["origins"]}
    monkeypatch.setattr(module, "source_manifest", lambda *a: m)
    monkeypatch.setattr(module, "read_splits", lambda *a, **kw: {})
    monkeypatch.setattr(module, "open_dataset", lambda *a, **kw: object())
    calls = []
    def select(*args, **kwargs):
        calls.append("select")
        return source["selection"]
    monkeypatch.setattr(module, "select_final_parameters", select)
    game = next(iter(source["game_sizes"]))
    def labels(view):
        calls.append("labels")
        return [SimpleNamespace(legal=SimpleNamespace(sample_id=f"s{j}", context=SimpleNamespace(game_id=game), features=features()),
                                supervision=SimpleNamespace(event_label_y=j), q_source=SimpleNamespace(fold_index=0)) for j in (0, 1)]
    monkeypatch.setattr(module, "open_supervision", labels)
    def kernel(rows, **kwargs):
        calls.append(kwargs["model_id"])
        assert rows == tuple(row(game, j, sample=f"s{j}") for j in (0, 1))
        assert kwargs["weighting"] == "game_equal" and kwargs["bins"] == 1 and kwargs["smoothing"] == 4
        assert kwargs["source"]["origins"] == source["origins"]
        assert kwargs["source"]["tom_fold"] is None
        return kwargs["model_id"]
    monkeypatch.setattr(module, "fit_model", kernel)
    result = module.fit_final_models("dataset", split_path="split", expected_split_digest="b" * 64,
                                     run_path="run", expected_complete_digest="d" * 64, protocol=protocol)
    assert result == {name: name for name in ("OBS-EQ1", "OBS-EQ0")}
    assert calls == ["select", "labels", "OBS-EQ1", "OBS-EQ0"]
    def wrong_source(view):
        rows = labels(view)
        rows[0].q_source.fold_index = 1
        return rows
    monkeypatch.setattr(module, "open_supervision", wrong_source)
    with pytest.raises(ValueError, match="OOF source/game mismatch"):
        module.fit_final_models("dataset", split_path="split", expected_split_digest="b" * 64,
                                run_path="run", expected_complete_digest="d" * 64, protocol=protocol)
    assert calls.count("OBS-EQ1") == calls.count("OBS-EQ0") == 1
