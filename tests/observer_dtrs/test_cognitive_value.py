"""主动认知价值的标签、固定规格和只读预测使用合成证据验证。"""

from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from scripts import fit_observer_dtrs_cognitive_value as study
from tests.observer_dtrs.test_sequential import ATTEMPT, TEAM, pairs_for, trajectory
from tests.observer_dtrs.test_speech_consequences import fixture_game, followup_reports, parent_matrix
from tests.tom.test_counterfactual_consumer import P
from werewolf.artifact_io import canonical_json_bytes, canonical_jsonl_bytes, publish_artifact
from werewolf.observer_dtrs import cognitive_value as cognitive
from werewolf.observer_dtrs import report_state_probability as probability
from werewolf.observer_dtrs.speech_candidates import SpeechCandidate, SpeechCandidates


Q_T = (.2, .2, .6)
Q_U = (.7, .1, .2)


def paired_rows():
    game, _ = trajectory()
    return [p for p in pairs_for(game) if p.start.acting_wolf == P[0] and p.start.observer_id == P[2]]


def sample(context, state=0, report_status="success"):
    features = (.3, .2, .09, .04, .06, .95, .05, float(context.day), float(context.phase == "pk_discussion"),
                float(len(context.alive_player_ids)), float(context.wolf_count == 2), .05, .3)
    return study.probability_study.Sample(context, ATTEMPT, features, state, report_status, state == 0, ())


def test_e4_gain_uses_first_followup_even_when_it_fails_and_start_truth_is_absent():
    first, second, last = paired_rows()
    s = [sample(first.start, None, "missing"), sample(second.start), sample(last.start, 2)]
    anchors = {item.sample_id[:2]: study.Anchor("历史", {P[2]: (1.,)}, "正文", "event", "sha") for item in s}
    joint = {item.sample_id: q for item, q in zip(s, (Q_T, Q_U, (.5, 0., .5)), strict=True)}
    trials = study.build_trials((first, second, last), s, anchors, joint, trajectory_complete=True)
    assert trials[0].outcome["O"] is trials[0].outcome["K"] is trials[0].outcome["M"] is True
    assert trials[0].outcome["successful_empty"] and trials[0].outcome["E_u"] == 0
    assert trials[0].outcome["carry_brier"] == pytest.approx(.52)
    assert trials[0].outcome["updated_brier"] == pytest.approx(.07)
    assert trials[0].outcome["G_total"] == pytest.approx(.45)
    assert trials[1].outcome["K"] is False and trials[1].outcome["M"] is True
    assert trials[1].outcome["G_total"] is None

    s[1] = replace(s[1], state=None, report_status="error", successful_empty=False)
    failed = study.build_trials((first, second, last), s, anchors, joint, trajectory_complete=True)[0]
    assert failed.pair.followup == second.start and failed.outcome["O"] is True
    assert failed.outcome["M"] is False and failed.outcome["report_status"] == "error"
    assert failed.outcome["G_total"] is None
    missing_q = study.label_pair(first, Q_T, None, sample(second.start), trajectory_complete=True)
    assert missing_q["M"] is False and missing_q["report_status"] == "success"


def test_incomplete_trajectory_has_unknown_opportunity_and_keeps_end_evidence():
    _, _, last = paired_rows()
    assert study.label_pair(last, Q_T, None, None, trajectory_complete=False)["O"] is None
    assert study.label_pair(last, Q_T, None, None, trajectory_complete=True)["O"] is False
    ended = replace(last, wolf_death=99, observer_death=99, terminal=99)
    assert study.label_pair(ended, Q_T, None, None, trajectory_complete=False)["O"] is None
    label = study.label_pair(ended, Q_T, None, None, trajectory_complete=True)
    assert label["O"] is False and label["K"] is label["M"] is label["G_total"] is None
    assert ended.wolf_death == ended.observer_death == ended.terminal


def test_base_weights_equalize_games_then_pres_then_observers():
    ids = [("long", "a", "i1"), ("long", "a", "i2"), ("long", "a", "i3"), ("long", "b", "i1"), ("short", "a", "i1")]
    rows = [SimpleNamespace(game_id=key[0], sample_id=key) for key in ids]
    assert study.base_weights(rows) == pytest.approx([1/6, 1/6, 1/6, .5, 1])


def rows_for_text():
    first = [0.] * len(cognitive.NUMERIC_FEATURES)
    first[cognitive.NUMERIC_FEATURES.index("pk")] = 1.
    first[cognitive.NUMERIC_FEATURES.index("observer_can_reply")] = 1.
    first[-3:] = Q_T
    second = first.copy()
    second[-3:] = Q_U
    return [cognitive.InputRow(("game", str(i)), "共享历史公开投票", "询问公开理由" if i % 2 == 0 else "审查公开理由",
                               tuple(first if i % 2 == 0 else second)) for i in range(4)]


def test_c0_uses_pre_history_only_and_c1_has_declared_text_interactions():
    rows = rows_for_text()
    transforms = cognitive.fit_transforms(rows, np.ones(len(rows)))
    assert "询问" not in transforms["history"]["vocabulary"] and "询问" in transforms["action"]["vocabulary"]
    original, alternative = rows[0], replace(rows[0], action=rows[1].action)
    c0 = cognitive.design_matrix([original, alternative], transforms, "C0").toarray()
    c1 = cognitive.design_matrix([original, alternative], transforms, "C1").toarray()
    assert c0[0] == pytest.approx(c0[1])
    assert c1[:, :c0.shape[1]] == pytest.approx(c0) and not np.array_equal(c1[0], c1[1])
    action, reply, ps, team, pk = np.split(c1[0, c0.shape[1]:], 5)
    assert reply == pytest.approx(action) and pk == pytest.approx(action)
    assert ps == pytest.approx(.6 * action) and team == pytest.approx(.8 * action)


def test_fractional_response_endpoints_and_unsupported_binary_head():
    design = sparse.csr_matrix([[-2.], [-1.], [1.], [2.]])
    weights = np.array([1., 2., 2., 1.])
    fitted = cognitive.fit_head(design, np.array([0., .25, .75, 1.]), weights)
    constant = cognitive.fit_head(design, np.ones(4), weights)
    unavailable = cognitive.fit_head(design, np.full(4, np.nan), weights)
    estimates = cognitive.predict_heads(design, dict(zip(cognitive.HEADS, (unavailable, constant, fitted), strict=True)))
    assert fitted["status"] == "fitted" and np.all(np.diff(estimates["cognitive_gain"]) > 0)
    assert max(abs(estimates["cognitive_gain"])) < 1
    assert constant == {"status": "no_variation", "support": {"rows": 4, "weight": 6., "min": 1., "max": 1., "mean": 1.}}
    assert cognitive.observer_prediction(estimates, 0)["predicted_target_stability"] is None
    assert unavailable["status"] == "no_support"


def test_oof_keeps_joint_probabilities_after_reusing_source_binding():
    first, _, _ = paired_rows()
    partitions, records, selections = {}, [], {}
    for fold in range(4):
        s = sample(replace(first.start, game_id=f"fold{fold}"))
        c = s.context
        partitions[fold] = [s]
        selections[str(fold)] = {"selection_folds": tuple(i for i in range(4) if i != fold),
                                 "selected_method": probability.HIERARCHICAL,
                                 "best_lambda": {probability.HIERARCHICAL: .01}}
        records.append({"game_id": c.game_id, "boundary_id": c.boundary_id, "observer_id": c.observer_id,
                        "attempt_id": s.attempt_id, "fold": fold, "prefix_digest": c.prefix_digest,
                        "step_index": c.step_index, "acting_wolf": c.acting_wolf, "alive_wolf_ids": c.alive_wolf_ids,
                        "method": probability.HIERARCHICAL, "lambda": .01, **probability.probability_record(Q_T)})
    joint, _ = study.read_oof(partitions, records, selections)
    assert joint[partitions[3][0].sample_id] == Q_T
    records[2]["attempt_id"] = "other"
    with pytest.raises(ValueError, match="折外概率与 PRE"):
        study.read_oof(partitions, records, selections)


def test_development_hooks_bind_actual_speech_and_raw_report_without_new_inference(monkeypatch):
    pre, _, game = fixture_game(single=True)
    game.game_id, game.bundle_digest = pre.game_id, "bundle"
    reports = followup_reports(pre, pre)
    archive_q = {**{name: getattr(pre, name) for name in study.development.PRE_IDENTITY}, "q": parent_matrix(pre).values}

    def load(folds, *, build_samples, visit_game):
        assert tuple(folds) == (0, 1, 2, 3, 4)
        visit_game(4, game, TEAM, publication_digest="publication", attempt_id="attempt")
        return {4: build_samples(pre, TEAM, reports, archive_q, attempt_id="attempt")}, {"publication_digest": "publication"}

    monkeypatch.setattr(study.development, "load_development", load)
    partitions, pairs, anchors, source = study.load_data()
    anchor = anchors[(pre.game_id, pre.boundary_id)]
    assert anchor.speech == "3号是狼人。" and anchor.speech not in anchor.history
    assert [s.state for s in partitions[4]] == [0, 2, 2, None, 2]
    assert len(pairs[4]) == 5 and source == {"publication_digest": "publication"}


def fixture_trials(prefix, *, novel=False):
    first, _, _ = paired_rows()
    trials = []
    for i in range(8):
        ctx = replace(first.start, game_id=f"{prefix}-{i}")
        o = (True, True, True, False, None, True, True, False)[i]
        k = (True, True, False, None, None, True, True, None)[i]
        pair = replace(first, start=ctx, followup=replace(first.followup, game_id=ctx.game_id) if o else None)
        text = "询问公开理由" if i % 2 == 0 else "审查公开理由"
        history = "留出专用龙龟" if novel else "共享历史公开投票"
        inputs = rows_for_text()[i % 4]
        inputs = replace(inputs, pre_id=(ctx.game_id, ctx.boundary_id), history=history,
                         action="留出玄鸟" if novel else text)
        anchor = study.Anchor(inputs.history, {ctx.observer_id: inputs.numeric[:-3]}, inputs.action, "event", "sha")
        # 非法或缺少标签时保留机会，G 只留在三个条件均满足的行。
        label = study.label_pair(pair, Q_T, Q_U, sample(pair.followup, 0 if i % 2 == 0 else 2) if o else None,
                                 trajectory_complete=o is not None)
        if k is False:
            label.update(K=False, G_total=None, carry_brier=None, updated_brier=None)
        trials.append(study.Trial(pair, anchor, inputs, label))
    return trials


def test_held_labels_and_vocabulary_do_not_change_training_and_scoring_population_is_shared():
    train, held = fixture_trials("train"), fixture_trials("held", novel=True)
    saved, y_hat = study.fit_and_predict(train, held)
    changed_held = [replace(row, outcome={**row.outcome, "O": False, "K": None, "G_total": None}) for row in held]
    other, _ = study.fit_and_predict(train, changed_held)
    assert canonical_json_bytes(saved) == canonical_json_bytes(other)
    assert "龙龟" not in saved["transforms"]["history"]["vocabulary"]
    assert "玄鸟" not in saved["transforms"]["action"]["vocabulary"]
    report = study.summarize(held, y_hat, replicates=40)["overall"]
    assert report["population"]["opportunity"]["unknown"]["observer_rows"] == 1
    assert report["population"]["target_stability"]["changed"]["observer_rows"] == 1
    for head in cognitive.HEADS:
        a, b = (report["models"][m][head] for m in cognitive.METHODS)
        assert a["scored_support"] == b["scored_support"]
    a, b = (report["models"][m]["cognitive_gain"]["mse"] for m in cognitive.METHODS)
    assert report["G_mse_C0_minus_C1"]["value"] == pytest.approx(a - b)
    assert report["G_mse_C0_minus_C1"]["replicates"] == 40
    assert report["models"]["C0"]["target_stability"]["calibration"]["bins"]


def report_model():
    width = len(probability.FEATURES)
    scaler = {"mean": [0.] * width, "std": [1.] * width}
    heads = {name: {"intercept": [bias], "coefficients": [[.25]] + [[0.]] * (width - 1), "standardization": scaler}
             for name, bias in (("speaker", -1.), ("teammate_given_no_speaker", -.5))}
    return {"method": probability.HIERARCHICAL, "features": probability.FEATURES, "lambda": .01, "heads": heads}


def test_saved_predictor_uses_current_public_history_and_exact_candidate_identity(tmp_path):
    model = report_model()
    p_artifact = publish_artifact(tmp_path / "probability", manifest_fields={
        "artifact_type": probability.ARTIFACT_TYPE, "schema_version": probability.ARTIFACT_TYPE,
    }, files={"models.json": canonical_json_bytes({"selected_method": model["method"], "models": {model["method"]: model}})})
    train, held = fixture_trials("train"), fixture_trials("held")
    saved, _ = study.fit_and_predict(train, held)
    path = tmp_path / "cognitive"
    publish_artifact(path, manifest_fields={"artifact_type": cognitive.ARTIFACT_TYPE,
        "schema_version": cognitive.ARTIFACT_TYPE, "probability_source": {"manifest_digest": p_artifact.manifest_digest}},
        files={"models.json": canonical_json_bytes(saved)})
    predictor = cognitive.load_predictor(path, p_artifact.path)
    pre, _, _ = fixture_game(single=True)
    matrix = parent_matrix(pre)
    panel = probability.pre_features(pre, TEAM, matrix)
    history, context = cognitive.pre_inputs(pre, panel, matrix.values)
    assert "3号是狼人" not in history and "turn_start" in history
    assert all(values[-1] == 1 for values in context.values())
    texts = ("询问公开理由", "审查公开理由")
    batch = SpeechCandidates(pre, "主题", (SpeechCandidate("INQUIRE", texts[0], P[2]), SpeechCandidate("INFORM", texts[1], None)))
    result = predictor.predict_candidates(batch, TEAM, matrix)
    assert result["prefix_digest"] == pre.prefix_digest and result["alive_wolf_ids"] == (P[0],)
    assert [r["text"] for r in result["candidates"]] == list(texts)
    assert set(result["candidates"][0]["observers"]) == set(context)
    assert result["candidates"][0]["observers"] != result["candidates"][1]["observers"]
    changed_tags = replace(batch, candidates=(SpeechCandidate("INFORM", texts[0], None), batch.candidates[1]))
    assert result == predictor.predict_candidates(changed_tags, TEAM, matrix)
    with pytest.raises(ValueError, match="identity mismatch"):
        predictor.predict_candidates(batch, TEAM, replace(matrix, boundary_id="wrong-pre"))


def test_optional_stage_scores_use_e4_report_even_without_immediate_supervision(tmp_path):
    trial = fixture_trials("held")[0]
    c = trial.pair.start
    target = {c.observer_id: probability.probability_record((.25, .25, .5))}
    row = {"game_id": c.game_id, "boundary_id": c.boundary_id, "attempt_id": trial.pair.attempt_id,
           "prefix_digest": c.prefix_digest, "step_index": c.step_index, "acting_wolf": c.acting_wolf,
           "alive_wolf_ids": c.alive_wolf_ids, "speech_event_id": trial.anchor.speech_event_id,
           "speech_text": trial.anchor.speech, "status": "predicted", "report_boundary": "immediate_vote_report_uncollected",
           "current_exposure": {"observers": {c.observer_id: probability.probability_record(Q_T)}},
           "prediction": {"exposure": {"observers": target}}}
    p_source = {"manifest_digest": "model", "method": probability.HIERARCHICAL, "lambda": .01}
    path = tmp_path / "post"
    publish_artifact(path, manifest_fields={"artifact_type": study.speech_replay.ARTIFACT_TYPE,
        "schema_version": study.speech_replay.ARTIFACT_TYPE, "mode": "factual",
        "probability_source": {"probability_manifest_digest": "model", "method": probability.HIERARCHICAL, "lambda": .01}},
        files={"records.jsonl": canonical_jsonl_bytes([row])})
    stages, results = study.decomposition(path, [trial], p_source)
    record = stages[trial.sample_id]
    assert record["G_self"] + record["G_external"] == pytest.approx(trial.outcome["G_total"])
    assert results["support"]["games"] == 1
    with pytest.raises(ValueError, match="同一冻结概率模型"):
        study.decomposition(path, [trial], {**p_source, "manifest_digest": "other"})


def test_cli_publishes_models_labels_and_fold4_evaluation_from_frozen_artifact(tmp_path, monkeypatch):
    template = paired_rows()
    partitions, pairs, anchors, oof, outer = {}, {}, {}, [], {}
    for fold in range(5):
        contexts = [replace(pair.start, game_id=f"game-{fold}") for pair in template]
        partitions[fold] = [sample(c, 0 if fold % 2 == 0 else 2) for c in contexts]
        pairs[fold] = [replace(pair, start=c, followup=contexts[i + 1] if i < 2 else None)
                       for i, (pair, c) in enumerate(zip(template, contexts, strict=True))]
        for i, s in enumerate(partitions[fold]):
            history = "先前公开投票"
            text = "审查公开理由" if i == 1 else "询问公开理由"
            anchors[s.sample_id[:2]] = study.Anchor(history, {s.context.observer_id: rows_for_text()[i].numeric[:-3]},
                                                     text, f"speech-{i}", "digest")
            c = s.context
            if fold < 4:
                oof.append({"fold": fold, "game_id": c.game_id, "boundary_id": c.boundary_id, "observer_id": c.observer_id,
                            "prefix_digest": c.prefix_digest, "step_index": c.step_index, "acting_wolf": c.acting_wolf,
                            "alive_wolf_ids": c.alive_wolf_ids, "attempt_id": s.attempt_id, "method": probability.HIERARCHICAL,
                            "lambda": .01, **probability.probability_record((Q_T, Q_U, (.5, 0., .5))[i])})
        if fold < 4:
            outer[str(fold)] = {"selection_folds": tuple(i for i in range(4) if i != fold),
                                "selected_method": probability.HIERARCHICAL, "best_lambda": {probability.HIERARCHICAL: .01}}

    lineage = {"publication_digest": "publication", "fold_manifest_digest": "folds", "q_manifest_digest": "Q"}
    model = report_model()
    p = publish_artifact(tmp_path / "probability-artifact", manifest_fields={**lineage,
        "artifact_type": probability.ARTIFACT_TYPE, "schema_version": probability.ARTIFACT_TYPE,
    }, files={
        "models.json": canonical_json_bytes({"selected_method": model["method"], "models": {model["method"]: model}}),
        "results.json": canonical_json_bytes({"selection": outer["0"], "outer_selection": outer}),
        "oof_predictions.jsonl": canonical_jsonl_bytes(oof),
    })
    monkeypatch.setattr(study, "load_data", lambda: (partitions, pairs, anchors, lineage))
    monkeypatch.setattr(study, "BOOTSTRAP_REPLICATES", 16)
    output = tmp_path / "cognitive-value"
    monkeypatch.setattr(study, "DESTINATION", output)
    study.main(["--probability-artifact", str(p.path)])
    summary = json.loads((output / "results.json").read_bytes())
    labels = [json.loads(line) for line in (output / "labels.jsonl").read_bytes().splitlines()]
    assert summary["probability_source"]["manifest_digest"] == p.manifest_digest
    assert summary["fit_folds"] == [0, 1, 2, 3] and summary["evaluation_fold"] == 4 and len(labels) == 15
    assert summary["fit_population"]["actual_actions"] == 12 and summary["stage_decomposition"]["status"] == "not_requested"
    assert labels[0]["q_t"] == list(Q_T) and labels[0]["predictions"] is None
    assert labels[-3]["predictions"]["C1"]["predicted_cognitive_gain"] is not None
    assert labels[-3]["q_t"] != list(Q_T) and labels[-3]["G_total"] == pytest.approx(0.)
    assert set(cognitive.load_predictor(output, p.path).heads) == set(cognitive.HEADS)
