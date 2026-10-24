"""Experiment A evidence/identity checks using synthetic data, never formal OOF."""
from dataclasses import replace
from hashlib import sha256
import math
from types import SimpleNamespace as NS

import numpy as np
import pytest

from tests.phase2.test_mapper import rows as mapper_rows, provenance as historical_provenance
from tests.phase2.test_mapper_study import configure_synthetic
from scripts import run_phase2_mapper_study as study
from werewolf.artifact_io import canonical_jsonl_bytes, publish_artifact
from werewolf.canonical_collection.public_history import PLAYER_IDS as P
from werewolf import phase2_offline as offline, phase2_mapper as mapper
from werewolf.phase2_q_representation import transform_q, condition_id, REPRESENTATION_VERSION
from werewolf.tom import backbone_evaluation as evaluation

PAIR = "b" * 64


def continuous_q():
    q = np.ones((7, 7), dtype=np.float64) / 6
    np.fill_diagonal(q, 0)
    return q


def paired_provenance(temporal="explicit_day_phase", representation="continuous"):
    return mapper.OOFProvenance("e" * 64, "c" * 64,
        {fold: sha256(f"{temporal}-{fold}".encode()).hexdigest() for fold in range(5)},
        PAIR, temporal, representation, "a" * 40)


def test_full_numeric_shape_simplex_competition_self_and_exact_canonical_ties():
    q = continuous_q()
    original = q.copy()
    # Deliberately unordered C includes a wolf and differs from Redirect J.
    competition = (P[4], P[1], P[2])
    hard = transform_q(q, competition, representation="hard_top1")
    assert hard.shape == (7, 7) and hard.dtype.kind in "fiu"
    np.testing.assert_array_equal(hard.sum(axis=1), np.ones(7))
    np.testing.assert_array_equal(np.diag(hard), np.zeros(7))
    np.testing.assert_array_equal(hard[:, [0, 3, 5, 6]], np.zeros((7, 4)))
    # Observer outside C chooses canonical player2 (a wolf); inside C excludes self.
    assert hard[0, 1] == hard[2, 1] == hard[4, 1] == 1
    assert hard[1, 2] == 1
    np.testing.assert_array_equal(q, original)
    continuous = transform_q(q, competition)
    np.testing.assert_array_equal(continuous, q)
    assert not np.shares_memory(continuous, q)
    # This is exact equality, not an approximate tie or rounded threshold.
    q[0, 1] -= 1e-10
    q[0, 2] += 1e-10
    assert transform_q(q, competition, representation="hard_top1")[0, 2] == 1


@pytest.mark.parametrize("bad", [
    np.ones((7, 6)), np.zeros((7, 7)), np.ones((7, 7), dtype=bool),
    np.full((7, 7), np.nan), np.full((7, 7), -1),
])
def test_reject_invalid_q(bad):
    with pytest.raises(ValueError):
        transform_q(bad, P, representation="hard_top1")


@pytest.mark.parametrize("competition", [(), (P[0],), (P[0], P[0]), (P[0], "unknown")])
def test_reject_invalid_competition(competition):
    with pytest.raises(ValueError):
        transform_q(continuous_q(), competition, representation="hard_top1")


def opportunity(game="g", fold=0):
    alive = P[:5]
    return offline._Opportunity(game, "b", "prefix", P[0], "speech", 1,
        alive, frozenset(P[:2]), alive, P[2:5], 4,
        {i: frozenset(set(P[2:5]) - {i}) for i in P[2:5]}, fold)


def test_shared_pre_transform_single_r2_math_and_candidate_population(monkeypatch):
    op = opportunity()
    q = continuous_q()
    # Make binary shares nonconstant, with a wolf competing for top-1.
    for observer, winner in ((2, 3), (3, 0), (4, 2)):
        q[observer] = 0
        q[observer, winner] = .7
        for target in range(7):
            if target not in (observer, winner):
                q[observer, target] = .06
    probabilities = {("g", "b", p): tuple(q[i]) for i, p in enumerate(P)}
    post = (offline.PostDayState("g", 1, 2, 2, True, True, P[2], 9, 0),)
    calls, r2_calls = [], []
    original_transform, original_r2 = offline.transform_q, offline.r2_input
    def transform(*args, **kwargs):
        result = original_transform(*args, **kwargs)
        calls.append(result.copy())
        return result
    def r2(*args, **kwargs):
        r2_calls.append(args)
        return original_r2(*args, **kwargs)
    monkeypatch.setattr(offline, "transform_q", transform)
    monkeypatch.setattr(offline, "r2_input", r2)
    continuous = offline.build_candidate_rows((op,), probabilities, post, evidence_identity_digest="a" * 64)
    calls.clear(); r2_calls.clear()
    hard = offline.build_candidate_rows((op,), probabilities, post,
        q_representation="hard_top1", evidence_identity_digest="b" * 64)
    assert len(calls) == 1 and len(r2_calls) == len(op.candidates) == len(hard) == len(continuous)
    assert all(args[4] is r2_calls[0][4] for args in r2_calls)
    identity = lambda r: (r.game_id, r.boundary_id, r.acting_wolf, r.phase, r.candidate_j,
                          r.fold, r.theta_ac, r.s_pre, r.resolved_outcome)
    assert [identity(r) for r in continuous] == [identity(r) for r in hard]
    matrix = calls[0]
    nonwolves = set(op.alive) - op.wolves
    mu = {target: np.mean([matrix[P.index(i), P.index(target)]
                           for i in nonwolves if i != target]) for target in op.competition}
    for row in hard:
        assert row.z.mu == pytest.approx(mu[row.candidate_j])
        assert row.z.delta == pytest.approx(mu[row.candidate_j] -
            max(value for target, value in mu.items() if target != row.candidate_j))
        assert row.z.sigma == pytest.approx(math.sqrt(row.z.mu * (1 - row.z.mu)))
        assert row.z.h == 0
    with pytest.raises(offline.Phase2DataError, match="evidence identity"):
        offline.build_candidate_rows((op,), probabilities, post, q_representation="hard_top1")
    with pytest.raises(KeyError):
        offline.build_candidate_rows((op,), {k: v for k, v in probabilities.items() if k[2] != P[0]},
                                    post, q_representation="hard_top1", evidence_identity_digest="b" * 64)


def synthetic_oof(root, *, bad_fold_condition=False, bad_row_condition=False):
    preparation = {"manifest_digest": PAIR}
    source = {"source_revision": "a" * 40, "implementation_digest": "a" * 64}
    contract = publish_artifact(root / "contract", manifest_fields={
        "artifact_type": "backbone_oof_evaluation_contract", "schema_version": evaluation.PAIRED_VERSION,
        "preparation": preparation, "training_source": source, "training_revision": "a" * 40,
        "training_seal_digest": "d" * 64,
    }, files={})
    descriptors = []
    ops = tuple(opportunity(f"g{fold}", fold) for fold in range(5))
    for temporal in ("explicit_day_phase", "implicit"):
        for fold in range(5):
            q = continuous_q()
            rows = [{"game_id": f"g{fold}", "boundary_id": "b", "observer": i,
                     "prefix_digest": "prefix", "checkpoint_digest": "f" * 64,
                     "temporal_condition": "implicit" if bad_row_condition and temporal == "explicit_day_phase" else temporal,
                     "probability": q[i].tolist()} for i in range(7)]
            artifact = publish_artifact(evaluation.prediction_path(root, "qwen3", fold, temporal, paired=True),
                manifest_fields={"artifact_type": "backbone_oof_fold_predictions",
                    "schema_version": evaluation.PAIRED_PREDICTION_VERSION,
                    "architecture": "qwen3", "fold": fold, "evaluation_digest": contract.manifest_digest,
                    "training_seal_digest": "d" * 64,
                    "temporal_condition": "implicit" if bad_fold_condition and temporal == "explicit_day_phase" else temporal,
                    "checkpoint_digest": "f" * 64, "row_count": len(rows)},
                files={"predictions.jsonl": canonical_jsonl_bytes(rows)})
            descriptors.append({"architecture": "qwen3", "fold": fold, "temporal_condition": temporal,
                                "prediction_digest": artifact.manifest_digest})
    seal = publish_artifact(root / "seal", manifest_fields={
        "artifact_type": "backbone_oof_evaluation_report", "schema_version": evaluation.PAIRED_REPORT_VERSION,
        "evaluation_digest": contract.manifest_digest, "preparation": preparation, "training_source": source,
        "training_seal_digest": "d" * 64, "paired_experiment_digest": PAIR, "predictions": descriptors,
    }, files={})
    folds = NS(fold_count=5, folds=tuple(NS(fold_index=f, game_ids=(f"g{f}",)) for f in range(5)))
    return seal, NS(public_view=NS(fold_manifest=folds)), ops


def test_loader_full_seven_observers_a0_a2_reuse_and_strict_identity(tmp_path):
    seal, publication, ops = synthetic_oof(tmp_path)
    def read(temporal="explicit_day_phase", representation="continuous", **overrides):
        return mapper.OOFProvenance.from_evaluation_root(tmp_path,
            **{"paired_experiment_digest": PAIR, "temporal_condition": temporal,
               "q_representation": representation, "expected_seal_digest": seal.manifest_digest, **overrides})
    a0, a1, a2 = read(), read("implicit"), read(representation="hard_top1")
    assert a0.prediction_digest_by_fold == a2.prediction_digest_by_fold
    assert a0.prediction_digest_by_fold != a1.prediction_digest_by_fold
    assert len({a0.identity_digest, a1.identity_digest, a2.identity_digest}) == 3
    assert [o.experiment_a["condition"] for o in (a0, a1, a2)] == ["A0", "A1", "A2"]
    q = offline.load_oof_probabilities(publication, tmp_path, ops, oof=a2)
    assert len(q) == len(ops) * 7
    assert all((op.game_id, "b", i) in q for op in ops for i in P)
    with pytest.raises(mapper.MapperProtocolError, match="parent/source"):
        read(paired_experiment_digest="0" * 64)
    with pytest.raises(mapper.MapperProtocolError, match="seal"):
        read(expected_seal_digest="0" * 64)
    with pytest.raises(ValueError):
        read("implicit", "hard_top1")
    with pytest.raises(ValueError):
        mapper.OOFProvenance.from_evaluation_root(tmp_path)
    for other in (a1, a2):
        population = tuple(replace(r, evidence_identity_digest=a0.identity_digest) for r in mapper_rows())
        with pytest.raises(mapper.MapperProtocolError, match="identity"):
            mapper.validate_rows(population, other)


def test_loader_rejects_coherent_fold_and_row_temporal_mismatches(tmp_path):
    for folder, bad_fold, bad_row in (("fold", True, False), ("row", False, True)):
        root = tmp_path / folder
        seal, publication, ops = synthetic_oof(root, bad_fold_condition=bad_fold, bad_row_condition=bad_row)
        if bad_fold:
            with pytest.raises(mapper.MapperProtocolError, match="lineage"):
                mapper.OOFProvenance.from_evaluation_root(root, paired_experiment_digest=PAIR,
                                                         expected_seal_digest=seal.manifest_digest)
        else:
            oof = mapper.OOFProvenance.from_evaluation_root(root, paired_experiment_digest=PAIR,
                                                          expected_seal_digest=seal.manifest_digest)
            with pytest.raises(offline.Phase2DataError, match="row lineage"):
                offline.load_oof_probabilities(publication, root, ops, oof=oof)


def test_fixed_m3_h_zero_and_historical_identity():
    oof = paired_provenance(representation="hard_top1")
    population = tuple(replace(r, evidence_identity_digest=oof.identity_digest,
        z=replace(r.z, h=0., sigma=math.sqrt(r.z.mu * (1 - r.z.mu)))) for r in mapper_rows())
    result = mapper.run_mapper_cv(population, oof)
    assert len(result.fit_manifests) == 5 and len(result.predictions) == len(population)
    assert set(result.metrics_by_spec) == {mapper.MODEL_SPECS[3].name}
    for manifest in result.fit_manifests:
        assert manifest["experiment_a"] == oof.experiment_a
        assert manifest["preprocessing"]["means"]["h"] == 0
        assert manifest["preprocessing"]["scales"]["h"] == 1
        assert manifest["preprocessing"]["knots"]["h"] == (0., 0.)
        assert "h" in mapper.FoldPreprocessor.fit(population, mapper.MODEL_SPECS[3]).columns()
    with pytest.raises(mapper.MapperProtocolError, match="fixed M3"):
        mapper._fit_mapper(tuple(r for r in population if r.fold != 0), mapper.MODEL_SPECS[2], oof, 0)
    old = historical_provenance()
    assert old.experiment_a is None
    assert set(old.record()) == {"evaluation_seal_digest", "evaluation_contract_digest", "prediction_digest_by_fold"}
    mapper.validate_rows(mapper_rows(), old)
    with pytest.raises(mapper.MapperProtocolError):
        replace(old, q_representation="hard_top1")
    assert condition_id("explicit_day_phase", "hard_top1") == "A2"
    assert oof.experiment_a["representation_version"] == REPRESENTATION_VERSION

    train, held_out = mapper.split_game_fold(population, 0, oof)
    fitted = mapper._fit_mapper(train, mapper.MODEL_SPECS[3], oof, 0)
    a0 = paired_provenance()
    mixed_rows = tuple(replace(r, evidence_identity_digest=a0.identity_digest) for r in held_out)
    with pytest.raises(mapper.MapperProtocolError, match="identity"):
        fitted.predict_held_out(mixed_rows, oof)
    with pytest.raises(mapper.MapperProtocolError, match="held-out"):
        fitted.predict_held_out(held_out, a0)


def test_existing_runner_publishes_fixed_m3_condition_artifact_without_bootstrap(monkeypatch, tmp_path):
    profile, old_layer, _ = configure_synthetic(monkeypatch, tmp_path)
    oof = paired_provenance(representation="hard_top1")
    layer = NS(**{**vars(old_layer), "candidates": tuple(replace(r,
        evidence_identity_digest=oof.identity_digest, z=replace(r.z, h=0.)) for r in old_layer.candidates)})
    monkeypatch.setattr(study, "attest_source", lambda **_kwargs: {
        "commit": "a" * 40, "branch": "research/main-experiments",
        "source_sha256": {path: "b" * 64 for path in study.SOURCE_FILES}})
    monkeypatch.setattr(study.OOFProvenance, "from_evaluation_root", classmethod(lambda cls, _root, **_kwargs: oof))
    monkeypatch.setattr(study.offline, "build_development_layer", lambda *_args, **_kwargs: layer)
    monkeypatch.setattr(study, "bootstrap_compare", lambda *_a, **_k: pytest.fail("Experiment A ran bootstrap"))
    monkeypatch.setattr(study, "select_from_comparison", lambda *_a: pytest.fail("Experiment A selected a model"))
    artifact = study.run_study(profile, evaluation_root=tmp_path / "synthetic-oof",
        paired_experiment_digest=PAIR, expected_seal_digest=oof.evaluation_seal_digest, q_representation="hard_top1")
    assert artifact.manifest["schema_version"] == study.EXPERIMENT_A_VERSION
    assert artifact.manifest["experiment_a"] == oof.experiment_a
    assert artifact.manifest["selection"]["method"] == "prespecified_fixed_M3"
    assert artifact.manifest["selection"]["primary_model"] == mapper.MODEL_SPECS[3].name
    assert len(artifact.manifest["fit_model_digests"]) == 5
    assert not (artifact.path / "bootstrap.json").exists()
    assert artifact.path.name.endswith("-a2")
    assert "fixed M3" in (artifact.path / "report.md").read_text()
    for representation, temporal in (("hard_top1", "explicit_day_phase"), ("continuous", "implicit")):
        with pytest.raises(study.MapperStudyError, match="frozen"):
            study.run_study(profile, q_representation=representation, temporal_condition=temporal)
