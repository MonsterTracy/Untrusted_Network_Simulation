"""Synthetic selected-M3 final fit and pure runtime inference checks."""

from dataclasses import replace
from hashlib import sha256
import inspect
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import run_phase2_mapper_final_fit as runner
from scripts import run_phase2_mapper_study as development
from werewolf import phase2_offline as offline
from werewolf import phase2_mapper_final as final
from werewolf import phase2_mapper_runtime as runtime
from werewolf.artifact_io import (
    canonical_json_bytes, canonical_jsonl_bytes, publish_artifact, sha256_bytes,
    verify_artifact,
)
from werewolf.canonical_collection.public_history import PLAYER_IDS
from werewolf.phase2_mapper import MODEL_SPECS, OOFProvenance, fit_full_development


def rows():
    result = []
    for fold in range(5):
        game = f"game-{fold}"
        for index in range(4):
            phase = "speech_pk" if fold % 2 else "speech"
            z = offline.R2Input(.1 + .08 * index + .01 * fold,
                -.3 + .1 * index - .02 * fold, .02 * (index + fold),
                .15 * index + .03 * fold, phase, 2 + fold % 3,
                2 + fold % 4, index % 3)
            result.append(offline.CandidateRow(game, f"boundary-{fold}", "player1",
                phase, f"player{index + 3}", fold, f"prefix-{fold}", (2, 5),
                (index + fold) % 3 == 0, z, offline.ResolvedOutcome.NO_EXILE))
    return tuple(result)


def oof_provenance():
    return OOFProvenance(offline.OOF_SEAL_DIGEST, "a" * 64,
        {fold: sha256(f"prediction-{fold}".encode()).hexdigest()
         for fold in range(5)})


def patch_synthetic_counts(monkeypatch):
    examples = rows()
    for name, count in (("EXPECTED_GAMES", 5), ("EXPECTED_PRE", 5),
                        ("EXPECTED_CANDIDATES", 20), ("EXPECTED_POST_DAYS", 5),
                        ("EXPECTED_SKIPPED_NO_CANDIDATE_PK_PRE", 0)):
        monkeypatch.setattr(offline, name, count)
    positives = sum(row.theta_ac for row in examples)
    monkeypatch.setattr(final, "EXPECTED_POSITIVES", positives)
    monkeypatch.setattr(runtime, "EXPECTED_POSITIVES", positives)
    monkeypatch.setattr(development, "EXPECTED_POSITIVES", positives)
    return examples


def synthetic_study(tmp_path, monkeypatch, *, selected=None, positive_offset=0):
    examples = patch_synthetic_counts(monkeypatch)
    selection = {"primary_model": selected or final.SELECTED_SPEC.name,
                 "stopped_at": None, "development_supported_alternatives": []}
    population = {"games": 5, "candidate_pre": 5, "candidate_rows": 20,
        "theta_positive_rows": sum(row.theta_ac for row in examples) + positive_offset,
        "post_day_states": 5, "skipped_no_candidate_pk_pre": 0}
    adjacent = [{"complex_model": MODEL_SPECS[i].name,
                 "simple_model": MODEL_SPECS[i - 1].name,
                 "role": "adjacent", "development_supported": True}
                for i in range(1, 4)]
    prediction_rows = [{"game_id": row.game_id, "boundary_id": row.boundary_id,
        "acting_wolf": row.acting_wolf, "phase": row.phase,
        "candidate_j": row.candidate_j, "theta_ac": row.theta_ac}
        for row in examples]
    study = publish_artifact(tmp_path / "selected-study", manifest_fields={
        "artifact_type": "phase2_mapper_development_study",
        "schema_version": final.STUDY_VERSION,
        "source": {"commit": final.STUDY_SOURCE_COMMIT},
        "selection": selection, "population": population,
        "protocol": {"feature_contract_version": final.mapper.FEATURE_CONTRACT_VERSION,
            "model_specs": [spec.record() for spec in MODEL_SPECS],
            "offline_contract_source_sha256": sha256_bytes(Path(offline.__file__).read_bytes()),
            "mapper_source_sha256": sha256_bytes(Path(final.mapper.__file__).read_bytes()),
            "selection_source_sha256": sha256_bytes(Path(final.selection.__file__).read_bytes())},
        "inputs": {"publication_digest": offline.PUBLICATION_DIGEST,
                   "role_sidecar_digest": offline.ROLE_SIDECAR_DIGEST,
                   "fold_manifest_digest": offline.FOLD_MANIFEST_DIGEST,
                   "oof": {"evaluation_seal_digest": offline.OOF_SEAL_DIGEST}},
    }, files={"selection.json": canonical_json_bytes(selection),
              "bootstrap.json": canonical_json_bytes({"pairs": adjacent}),
              f"cross_fitted_predictions/{final.SELECTED_SPEC.name}.jsonl":
                  canonical_jsonl_bytes(prediction_rows)})
    return study, examples


def uniform_q():
    q = np.full((7, 7), 1 / 6, dtype=np.float64)
    np.fill_diagonal(q, 0.0)
    return q


def source_record():
    return {"commit": "b" * 40, "source_sha256": {
        "werewolf/phase2_offline.py": sha256_bytes(Path(offline.__file__).read_bytes()),
        "werewolf/phase2_mapper.py": sha256_bytes(Path(final.mapper.__file__).read_bytes()),
        "werewolf/phase2_mapper_runtime.py": sha256_bytes(Path(runtime.__file__).read_bytes()),
    }}


def test_selected_study_digest_mismatch_fails(monkeypatch, tmp_path):
    study, _ = synthetic_study(tmp_path, monkeypatch)
    with pytest.raises(final.FinalMapperError, match="digest"):
        final.verify_selected_study(study.path)


def test_selected_model_mismatch_fails(monkeypatch, tmp_path):
    study, _ = synthetic_study(tmp_path, monkeypatch, selected=MODEL_SPECS[2].name)
    monkeypatch.setattr(final, "STUDY_DIGEST", study.manifest_digest)
    with pytest.raises(final.FinalMapperError, match="selection"):
        final.verify_selected_study(study.path)


def test_selected_population_mismatch_fails(monkeypatch, tmp_path):
    study, _ = synthetic_study(tmp_path, monkeypatch, positive_offset=1)
    monkeypatch.setattr(final, "STUDY_DIGEST", study.manifest_digest)
    with pytest.raises(final.FinalMapperError, match="population"):
        final.verify_selected_study(study.path)


def test_frozen_study_candidate_identity_and_label_must_match(monkeypatch, tmp_path):
    study, examples = synthetic_study(tmp_path, monkeypatch)
    assert final.validate_study_candidate_population(study, examples) == \
        final.candidate_identity_digest(examples)
    changed = (replace(examples[0], theta_ac=not examples[0].theta_ac), *examples[1:])
    with pytest.raises(final.FinalMapperError, match="differs"):
        final.validate_study_candidate_population(study, changed)


def test_full_fit_and_model_serialization_are_deterministic(monkeypatch, tmp_path):
    study, examples = synthetic_study(tmp_path, monkeypatch)
    monkeypatch.setattr(final, "STUDY_DIGEST", study.manifest_digest)
    provenance = oof_provenance()
    first = fit_full_development(examples, final.SELECTED_SPEC, provenance)
    second = fit_full_development(examples, final.SELECTED_SPEC, provenance)
    assert first.manifest == second.manifest
    assert first.preprocessor.columns() == second.preprocessor.columns()
    assert len(first.preprocessor.columns()) == 16
    assert first.preprocessor.knots == second.preprocessor.knots
    assert len(first.manifest["training_game_ids"]) == 5
    assert first.manifest["training_candidate_count"] == 20
    assert first.manifest["training_scope"] == "full_development"
    assert first.manifest["model_spec"] == final.SELECTED_SPEC.record()
    source = source_record()
    identity = final.candidate_identity_digest(examples)
    artifact1 = final.publish_full_mapper(tmp_path / "final-a", fitted=first,
        rows=examples, study=study, source=source, candidate_digest=identity)
    artifact2 = final.publish_full_mapper(tmp_path / "final-b", fitted=second,
        rows=examples, study=study, source=source, candidate_digest=identity)
    assert artifact1.manifest_digest == artifact2.manifest_digest
    assert (artifact1.path / "model.json").read_bytes() == (artifact2.path / "model.json").read_bytes()
    assert artifact1.manifest["training"]["candidate_identity_digest"] == identity
    assert artifact1.manifest["training"]["theta_positive_rows"] == sum(
        row.theta_ac for row in examples)
    assert artifact1.manifest["model"]["feature_order"] == list(first.preprocessor.columns())
    assert artifact1.manifest["model"]["preprocessing"]["knots"] == \
        json.loads((artifact1.path / "model.json").read_text())["preprocessing"]["knots"]
    assert artifact1.manifest["calibration"] == "none"
    assert artifact1.manifest["probability_name"] == "p_tilde"
    verify_artifact(artifact1.path,
        expected_artifact_type="phase2_full_development_oof_mapper",
        expected_schema_version=final.FINAL_VERSION)


def test_runner_passes_all_21638_rows_without_final_q_feature_path(monkeypatch):
    sentinel = tuple(range(21638))
    study = SimpleNamespace(manifest_digest=final.STUDY_DIGEST, path=Path("/tmp/study"),
                            manifest={"inputs": {"oof": oof_provenance().record()}})
    monkeypatch.setattr(runner, "preflight", lambda profile: (Path("/tmp/final"), {}, study))
    monkeypatch.setattr(runner.OOFProvenance, "from_evaluation_root",
                        classmethod(lambda cls, root: oof_provenance()))
    monkeypatch.setattr(runner.offline, "build_development_layer",
                        lambda publication, evaluation: SimpleNamespace(candidates=sentinel))
    monkeypatch.setattr(runner.development, "validate_population", lambda layer, oof: None)
    monkeypatch.setattr(runner.final, "validate_study_candidate_population",
                        lambda selected_study, candidate_rows: "a" * 64)

    def capture_fit(candidate_rows, spec, oof):
        assert candidate_rows is sentinel and len(candidate_rows) == 21638
        assert spec is final.SELECTED_SPEC
        return SimpleNamespace(manifest={"oof_q": oof.record()})

    monkeypatch.setattr(runner, "fit_full_development", capture_fit)
    monkeypatch.setattr(runner.final, "publish_full_mapper", lambda *args, **kwargs: kwargs)
    output = runner.run_final_fit("unused-profile")
    assert output["rows"] is sentinel
    assert output["candidate_digest"] == "a" * 64


def test_final_preflight_is_read_only_and_rejects_existing_destination(monkeypatch, tmp_path):
    study, _ = synthetic_study(tmp_path, monkeypatch)
    monkeypatch.setattr(final, "STUDY_DIGEST", study.manifest_digest)
    monkeypatch.setattr(final, "STUDY_PATH", study.path)
    monkeypatch.setattr(runner, "attest_final_source",
                        lambda: {"commit": "b" * 40, "source_sha256": {}})
    monkeypatch.setattr(runner.offline, "build_development_layer",
                        lambda *_args: pytest.fail("preflight read the full publication"))
    root = tmp_path / "storage"
    root.mkdir()
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"artifact_root": str(root)}))
    destination, source, selected = runner.preflight(profile)
    assert source["commit"] == "b" * 40
    assert selected.manifest_digest == study.manifest_digest
    assert not destination.exists()
    destination.parent.mkdir(parents=True)
    destination.mkdir()
    with pytest.raises(final.FinalMapperError, match="already exists"):
        runner.preflight(profile)


def test_runtime_q_schema_validation():
    valid = uniform_q()
    np.testing.assert_array_equal(runtime.validate_final_q(valid), valid)
    invalid = []
    invalid.append(valid[:6])
    diagonal = valid.copy(); diagonal[0, 0] = .1; invalid.append(diagonal)
    negative = valid.copy(); negative[0, 1] = -.1; invalid.append(negative)
    nan = valid.copy(); nan[0, 1] = float("nan"); invalid.append(nan)
    wrong_sum = valid.copy(); wrong_sum[0, 1] = 0.; invalid.append(wrong_sum)
    invalid.append(np.ones((7, 7), dtype=bool))
    for q in invalid:
        with pytest.raises(runtime.RuntimeMapperError):
            runtime.validate_final_q(q)


def test_runtime_observer_mask_raw_mu_delta_and_population_sigma():
    q = uniform_q()
    # Alive non-wolf observers are p3, p4, p5; p3 cannot observe itself.
    for observer in ("player4", "player5"):
        i = PLAYER_IDS.index(observer)
        q[i] = 0.0
        q[i, PLAYER_IDS.index("player3")] = .1
        q[i, PLAYER_IDS.index("player6")] = .5  # Dead target retains raw Q mass.
        other = [seat for seat in range(7) if seat not in
                 (i, PLAYER_IDS.index("player3"), PLAYER_IDS.index("player6"))]
        for seat in other:
            q[i, seat] = .4 / len(other)
    alive = PLAYER_IDS[:5]
    z = runtime.build_runtime_z(q, alive=alive,
        known_wolves=frozenset(("player1", "player2")), acting_wolf="player1",
        candidate_j="player3", phase="speech", public_speaker_queue=alive,
        competition=alive)
    assert z.audience_size == 2
    assert z.mu == pytest.approx(.1)  # No alive-target renormalization.
    nonwolves = {"player3", "player4", "player5"}
    means = {target: np.mean([q[PLAYER_IDS.index(observer), PLAYER_IDS.index(target)]
             for observer in nonwolves - {target}]) for target in alive}
    assert z.delta == pytest.approx(means["player3"] - max(
        value for target, value in means.items() if target != "player3"))
    assert z.sigma == pytest.approx(0.)
    assert z.remaining_speakers_before_vote == 4


def test_runtime_sigma_single_observer_and_frozen_entropy_m_one():
    q = uniform_q()
    alive = PLAYER_IDS[:4]
    wolves = frozenset(("player1", "player2"))
    z = runtime.build_runtime_z(q, alive=alive, known_wolves=wolves,
        acting_wolf="player1", candidate_j="player3", phase="speech",
        public_speaker_queue=alive, competition=alive)
    assert z.audience_size == 1 and z.sigma == 0.
    # The frozen R2 helper defines m=1 as H=0. Under current legal wolf-speaker
    # PK rules this edge cannot arise in runtime, because the wolf joins C.
    q_by_observer = {player: tuple(map(float, q[PLAYER_IDS.index(player)]))
                     for player in ("player3", "player4")}
    edge = offline.r2_input(alive, wolves, ("player3", "player4"), "player3",
                            q_by_observer, phase="speech_pk", remaining=0)
    assert edge.h == 0.


def test_runtime_pk_entropy_conditions_on_public_competition():
    q = uniform_q()
    # Place substantial mass on a dead target; entropy must renormalize only
    # inside the public tie set, while mu/delta continue to use raw masses.
    for observer, selected in (("player4", {"player1": .05, "player3": .15}),
                               ("player5", {"player1": .05, "player3": .15,
                                            "player4": .1})):
        index = PLAYER_IDS.index(observer)
        q[index] = 0.0
        q[index, PLAYER_IDS.index("player6")] = .5
        for target, mass in selected.items():
            q[index, PLAYER_IDS.index(target)] = mass
        unused = [seat for seat in range(7) if seat != index and q[index, seat] == 0]
        remainder = (1.0 - q[index].sum()) / len(unused)
        for seat in unused:
            q[index, seat] = remainder
    alive = PLAYER_IDS[:5]
    competition = ("player1", "player3", "player4")
    z = runtime.build_runtime_z(q, alive=alive,
        known_wolves=frozenset(("player1", "player2")), acting_wolf="player1",
        candidate_j="player3", phase="speech_pk",
        public_speaker_queue=competition, competition=competition)
    h4 = -sum(p * math.log(p) for p in (.25, .75)) / math.log(2)
    h5 = -sum(p * math.log(p) for p in (1 / 6, .5, 1 / 3)) / math.log(3)
    assert z.h == pytest.approx((h4 + h5) / 2)
    assert z.mu == pytest.approx(.15)
    assert z.remaining_speakers_before_vote == 2


def test_runtime_feature_parity_and_uncalibrated_p_tilde(monkeypatch, tmp_path):
    study, examples = synthetic_study(tmp_path, monkeypatch)
    monkeypatch.setattr(final, "STUDY_DIGEST", study.manifest_digest)
    fitted = fit_full_development(examples, final.SELECTED_SPEC, oof_provenance())
    artifact = final.publish_full_mapper(tmp_path / "final", fitted=fitted,
        rows=examples, study=study, source=source_record(),
        candidate_digest=final.candidate_identity_digest(examples))
    monkeypatch.setattr(runtime, "STUDY_DIGEST", study.manifest_digest)
    monkeypatch.setattr(final.mapper, "_fit_mapper",
                        lambda *_args, **_kwargs: pytest.fail("runtime refit a mapper"))
    loaded = runtime.load_runtime_mapper(artifact.path,
        expected_manifest_digest=artifact.manifest_digest)
    alive = PLAYER_IDS[:5]
    wolves = frozenset(("player1", "player2"))
    q = uniform_q()
    audit = loaded.infer(q, alive=alive, known_wolves=wolves,
        acting_wolf="player1", candidate_j="player3", phase="speech",
        public_speaker_queue=alive, competition=alive, audit=True)
    by_observer = {player: tuple(map(float, q[PLAYER_IDS.index(player)]))
                   for player in ("player3", "player4", "player5")}
    frozen = offline.r2_input(alive, wolves, alive, "player3", by_observer,
                              phase="speech", remaining=4)
    assert audit.z == frozen
    assert audit.observer_ids == ("player4", "player5")
    assert audit.feature_order == fitted.preprocessor.columns()
    np.testing.assert_allclose(audit.design_values,
        fitted.preprocessor.transform((SimpleNamespace(phase="speech", z=frozen),))[0])
    expected_score = float(fitted.estimator.decision_function(
        np.asarray(audit.design_values).reshape(1, -1))[0])
    assert audit.raw_score == pytest.approx(expected_score)
    assert 0 < audit.p_tilde < 1
    assert loaded.infer(q, alive=alive, known_wolves=wolves,
        acting_wolf="player1", candidate_j="player3", phase="speech",
        public_speaker_queue=alive, competition=alive) == audit.p_tilde
    assert not {"suspicion_support", "resolved_outcome", "v_ref", "y"} & set(
        inspect.signature(loaded.infer).parameters)


def test_runtime_rejects_illegal_queue_and_candidate():
    q = uniform_q()
    alive = PLAYER_IDS[:5]
    kwargs = dict(alive=alive, known_wolves=frozenset(("player1", "player2")),
        acting_wolf="player1", candidate_j="player3", phase="speech",
        public_speaker_queue=alive, competition=alive)
    with pytest.raises(runtime.RuntimeMapperError, match="competition or queue"):
        runtime.build_runtime_z(q, **{**kwargs, "public_speaker_queue": alive[:-1]})
    with pytest.raises(runtime.RuntimeMapperError):
        runtime.build_runtime_z(q, **{**kwargs, "candidate_j": "player2"})
