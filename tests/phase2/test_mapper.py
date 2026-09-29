"""Synthetic mapper-protocol checks; never reads the 1,500-game publication."""

from dataclasses import replace
from hashlib import sha256
import math
import os
from pathlib import Path

import numpy as np
import pytest

from werewolf.phase2_offline import CandidateRow, OOF_SEAL_DIGEST, R2Input, ResolvedOutcome
from werewolf.phase2_mapper import (
    CONTEXT_NUMERIC, EVIDENCE, FEATURE_CONTRACT_VERSION, MODEL_SPECS,
    FoldPreprocessor, MapperPrediction, MapperProtocolError, OOFProvenance,
    evaluate_predictions, fit_full_development, run_mapper_cv,
    split_game_fold, validate_rows, _fit_mapper,
)


def provenance():
    return OOFProvenance(OOF_SEAL_DIGEST, "a" * 64,
        {fold: sha256(f"prediction-{fold}".encode()).hexdigest() for fold in range(5)})


def rows():
    result = []
    for fold in range(5):
        game = f"game-{fold}"
        for index, candidate in enumerate(("player3", "player4", "player5", "player6")):
            phase = "speech_pk" if fold % 2 else "speech"
            z = R2Input(.1 + .08 * index + .01 * fold,
                        -.3 + .1 * index - .02 * fold,
                        .02 * (index + fold), .15 * index + .03 * fold,
                        phase, 2 + fold % 3, 2 + fold % 4, index % 3)
            result.append(CandidateRow(game, f"boundary-{fold}", "player1",
                phase, candidate, fold, f"prefix-{fold}", (2, 5),
                (index + fold) % 3 == 0, z, ResolvedOutcome.NO_EXILE))
    return tuple(result)


def test_four_specs_select_only_frozen_z_features_and_are_deterministic():
    assert len(MODEL_SPECS) == 4
    assert tuple(spec.evidence_features for spec in MODEL_SPECS) == (
        (), ("mu",), EVIDENCE, EVIDENCE)
    assert all(spec.canonical_bytes() == type(spec)(**dict(reversed(
        list(spec.record().items())))).canonical_bytes() for spec in MODEL_SPECS)
    expected = (("phase_pk", *CONTEXT_NUMERIC),
                ("phase_pk", *CONTEXT_NUMERIC, "mu"),
                ("phase_pk", *CONTEXT_NUMERIC, *EVIDENCE))
    for spec, columns in zip(MODEL_SPECS[:3], expected):
        assert FoldPreprocessor.fit(rows(), spec).columns() == columns
    additive = FoldPreprocessor.fit(rows(), MODEL_SPECS[3])
    assert len(additive.columns()) == 16
    assert all("*" not in name and ":" not in name for name in additive.columns())
    assert set(additive.knots) == set(EVIDENCE)
    base = rows()[0]
    changed = replace(base, z=replace(base.z, mu=base.z.mu + .1))
    original_design, changed_design = additive.transform((base, changed))
    differing = {name for name, a, b in zip(additive.columns(), original_design,
                                              changed_design) if not np.isclose(a, b)}
    assert differing <= {"mu", "mu_hinge_0", "mu_hinge_1"}
    assert "mu" in differing


def test_evaluation_metadata_cannot_enter_design_matrix():
    sample = rows()[0]
    altered = replace(sample, s_pre=(1, 5), theta_ac=not sample.theta_ac,
                      resolved_outcome=ResolvedOutcome.TARGET_J_EXILED)
    for spec in MODEL_SPECS:
        fitted = FoldPreprocessor.fit(rows(), spec)
        np.testing.assert_array_equal(fitted.transform((sample,)),
                                      fitted.transform((altered,)))
        assert all(name not in fitted.columns() for name in
                   ("s_pre", "s_post", "resolved_outcome", "theta_ac", "v_ref"))


def test_entire_game_is_one_fold_and_candidate_identity_unique():
    population, oof = rows(), provenance()
    assert len(validate_rows(population, oof)) == 5
    for fold in range(5):
        train, held_out = split_game_fold(population, fold, oof)
        assert {r.game_id for r in train}.isdisjoint({r.game_id for r in held_out})
        assert {r.fold for r in held_out} == {fold}
    with pytest.raises(MapperProtocolError, match="duplicate identity"):
        validate_rows(population + (population[0],), oof)
    with pytest.raises(MapperProtocolError, match="one game spans"):
        validate_rows(population + (replace(population[0], candidate_j="player7", fold=1),), oof)


def test_preprocessing_fits_only_training_games_and_knots():
    population, oof = rows(), provenance()
    train, held_out = split_game_fold(population, 0, oof)
    extreme = replace(held_out[0], z=replace(held_out[0].z, mu=1000.0))
    fitted = FoldPreprocessor.fit(train, MODEL_SPECS[3])
    assert fitted.means["mu"] == pytest.approx(np.mean([r.z.mu for r in train]))
    assert fitted.means["mu"] != pytest.approx(np.mean([r.z.mu for r in train] + [1000.0]))
    expected = np.quantile([(r.z.mu - fitted.means["mu"]) / fitted.scales["mu"]
                            for r in train], [1 / 3, 2 / 3], method="linear")
    np.testing.assert_allclose(fitted.knots["mu"], expected)
    assert np.isfinite(fitted.transform((extreme,))).all()


def test_four_models_fit_predict_without_rebalancing_and_keep_oof_lineage():
    population, oof = rows(), provenance()
    train, _ = split_game_fold(population, 0, oof)
    assert _fit_mapper(train, MODEL_SPECS[0], oof, 0).estimator.class_weight is None
    result = run_mapper_cv(population, oof)
    assert len(result.predictions) == 4 * len(population)
    assert len(result.fit_manifests) == 20
    assert set(result.metrics_by_spec) == {spec.name for spec in MODEL_SPECS}
    for manifest in result.fit_manifests:
        assert manifest["likelihood"] == "unweighted_binary_negative_log_likelihood"
        assert manifest["regularizer"] == {"type": "L2_nonintercept", "C": 100.0}
        assert manifest["feature_contract_version"] == FEATURE_CONTRACT_VERSION
        assert manifest["training_scope"] == "mapper_cv_fold"
        assert manifest["mapper_fold"] not in set(manifest["training_game_folds"].values())
        assert manifest["preprocessing"]["training_game_digest"] == manifest["training_game_digest"]
        assert manifest["oof_q"] == oof.record()
        assert len(manifest["model_digest"]) == 64
    for prediction in result.predictions:
        assert prediction.mapper_fold == prediction.game_oof_fold
        assert prediction.qwen3_prediction_digest == oof.prediction_digest_by_fold[prediction.game_oof_fold]
        assert prediction.evaluation_seal_digest == OOF_SEAL_DIGEST
        assert math.isfinite(prediction.raw_score)
        assert 0 < prediction.raw_probability < 1


def prediction(game, candidate, label, probability):
    return MapperPrediction(game, "b", "player1", "speech", candidate,
        label, 0, MODEL_SPECS[0].name, math.log(probability / (1 - probability)),
        probability, 0, "prefix", "b" * 64, OOF_SEAL_DIGEST, "c" * 64)


def test_pooled_and_game_macro_losses_are_distinct_and_correct():
    examples = (prediction("g0", "player3", False, .2),
                prediction("g0", "player4", True, .8),
                prediction("g1", "player3", True, .1))
    metrics = evaluate_predictions(examples)
    easy, hard = -math.log(.8), -math.log(.1)
    assert metrics.pooled_log_loss == pytest.approx((2 * easy + hard) / 3)
    assert metrics.game_macro_log_loss == pytest.approx((easy + hard) / 2)
    assert metrics.pooled_brier == pytest.approx((.04 + .04 + .81) / 3)
    assert metrics.game_macro_brier == pytest.approx((.04 + .81) / 2)
    assert metrics.candidate_count == 3 and metrics.game_count == 2
    assert sum(bin_.candidate_count for bin_ in metrics.reliability_bins) == 3


def test_calibration_diagnostics_have_no_deployment_calibrator():
    varied = (prediction("g0", "player3", False, .2),
              prediction("g0", "player4", True, .4),
              prediction("g1", "player3", False, .6),
              prediction("g1", "player4", True, .8))
    metrics = evaluate_predictions(varied)
    assert math.isfinite(metrics.calibration_intercept)
    assert math.isfinite(metrics.calibration_slope)
    assert not hasattr(metrics, "calibrator")
    constant = (prediction("g0", "player3", False, .5),
                prediction("g1", "player3", True, .5))
    assert evaluate_predictions(constant).calibration_slope is None
    one_class = (prediction("g0", "player3", False, .2),
                 prediction("g1", "player3", False, .8))
    assert evaluate_predictions(one_class).calibration_intercept is None
    assert evaluate_predictions(one_class).auroc is None
    separated = (prediction("g0", "player3", False, .2),
                 prediction("g1", "player3", True, .8))
    assert evaluate_predictions(separated).calibration_slope is None


def test_calibration_diagnostic_receives_saved_raw_score_not_clipped_logit(monkeypatch):
    import werewolf.phase2_mapper as mapper

    seen = []

    def capture(_y, raw_scores):
        seen.extend(raw_scores.tolist())
        return None, None

    monkeypatch.setattr(mapper, "_calibration_diagnostics", capture)
    examples = (replace(prediction("g0", "player3", False, .2),
                        raw_probability=0., raw_score=-1000.),
                replace(prediction("g1", "player3", True, .8),
                        raw_probability=1., raw_score=1000.))
    assert evaluate_predictions(examples).calibration_slope is None
    assert seen == [-1000., 1000.]


def test_full_fit_and_unsealed_provenance_are_distinct_from_development_cv():
    population, oof = rows(), provenance()
    train, held_out = split_game_fold(population, 0, oof)
    fitted = _fit_mapper(train, MODEL_SPECS[0], oof, 0)
    with pytest.raises(MapperProtocolError, match="held-out mapper fold"):
        fitted.predict_held_out(train[:1], oof)
    altered_oof = OOFProvenance(OOF_SEAL_DIGEST, "d" * 64,
                                oof.prediction_digest_by_fold)
    with pytest.raises(MapperProtocolError, match="held-out mapper fold"):
        fitted.predict_held_out(held_out, altered_oof)
    with pytest.raises(MapperProtocolError, match="complete fixed development"):
        fit_full_development(population, MODEL_SPECS[0], oof)
    with pytest.raises(MapperProtocolError, match="sealed Qwen3"):
        OOFProvenance("0" * 64, "a" * 64, oof.prediction_digest_by_fold)
    with pytest.raises(MapperProtocolError, match="five development folds"):
        run_mapper_cv(tuple(r for r in population if r.fold != 4), oof)


@pytest.mark.skipif(not os.environ.get("PHASE2_OOF_ROOT"),
                    reason="sealed OOF root is not a quick-test fixture")
def test_sealed_oof_provenance_regression():
    resolved = OOFProvenance.from_evaluation_root(Path(os.environ["PHASE2_OOF_ROOT"]))
    assert resolved.evaluation_seal_digest == OOF_SEAL_DIGEST
    assert set(resolved.prediction_digest_by_fold) == set(range(5))
