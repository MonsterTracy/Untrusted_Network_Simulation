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
