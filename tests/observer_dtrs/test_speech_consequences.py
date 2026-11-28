"""合成公开边界上的发言后果、事实监督与候选表示测试。"""

from dataclasses import replace
import json
from math import fsum, log
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import evaluate_observer_dtrs_speech_consequences as study
from tests.tom.test_counterfactual_consumer import P, PublicWitness
from werewolf.artifact_io import canonical_jsonl_bytes
from werewolf.canonical_collection import (BeliefObservationStatus, construct_belief_observation,
                                           freeze_public_event_history)
from werewolf.canonical_collection.speech import V1SpeechAction
from werewolf.observer_dtrs import report_state_probability as probability
from werewolf.observer_dtrs import speech_consequences as consequences
from werewolf.observer_dtrs.context import contexts_for_pre
from werewolf.observer_dtrs.features import observer_features
from werewolf.observer_dtrs.probabilities import ProbabilityMatrix
from werewolf.observer_dtrs.speech_candidates import SpeechCandidate, SpeechCandidates
from werewolf.speech.speech_perceiver import SpeechParseAuditResult
from werewolf.structured_history import plan_structured_history
from werewolf.tom.dataset import ExperimentCapacity, tensorize_structured_plan


TEAM = P[:2]
CAPACITY = ExperimentCapacity(1024)


def fixture_game(phase="discussion", *, last=False, single=False):
    witness = PublicWitness(phase, alive=tuple(p for p in P if not single or p != TEAM[1]))
    if single:
        witness.events = witness.events[:3]
        witness.annotations = []
        witness.event("turn_start", speaker=P[0])
    prefixes = [witness.pre()]

    def speak():
        speaker = witness.events[-1]["speaker"]
        witness.speech(speaker, "point_as_werewolf", P[2], "3号是狼人。")

    if last:
        speak()
        witness.event("turn_start", speaker=P[1])
        prefixes.append(witness.pre())
    pre = prefixes[-1]
    speak()
    if last:
        witness.event("phase_change", day=1, phase=consequences.VOTE_PHASE[phase])
        followup = None
    else:
        witness.event("turn_start", speaker=P[2] if single else P[1])
        followup = witness.pre()
        prefixes.append(followup)
    return pre, followup, SimpleNamespace(public_event_stream=freeze_public_event_history(witness.events),
                                         authoritative_pre_prefixes=tuple(prefixes), speech_annotations_v1=tuple(witness.annotations))


@pytest.fixture
def model():
    width = len(probability.FEATURES)
    scaler = {"mean": [0.] * width, "std": [1.] * width}
    return {"method": probability.HIERARCHICAL, "features": probability.FEATURES, "lambda": .01,
            "heads": {"speaker": {"intercept": [-2.], "coefficients": [[5.], *([[0.]] * (width - 1))],
                                  "standardization": scaler},
                      "teammate_given_no_speaker": {"intercept": [-1.], "coefficients": [[0.]] * width,
                                                     "standardization": scaler}}}


def parent_matrix(pre):
    values = tuple(tuple(0 if i == j else .35 if j == (i + 1) % 7 else .13 for j in range(7)) for i in range(7))
    return ProbabilityMatrix(pre.game_id, pre.boundary_id, pre.prefix_digest, values)


@pytest.fixture
def predictor():
    class ControlledGraph:
        capacity = CAPACITY
        calls = 0

        def __call__(self, *, target_ids, attention_mask, **inputs):
            self.calls += 1
            target = target_ids[0, int(attention_mask[0].sum()) - 2]
            boost = 4 * (torch.arange(1, 8) == target)
            logits = boost.float().expand(7, 7).masked_fill(torch.eye(7, dtype=torch.bool), -torch.inf)
            return logits.log_softmax(-1).unsqueeze(0)

    result = consequences.SpeechConsequencePredictor.__new__(consequences.SpeechConsequencePredictor)
    result._model, result._device = ControlledGraph(), torch.device("cpu")
    return result


@pytest.mark.parametrize(("phase", "last", "marker", "post_phase"), (
    ("discussion", False, "turn_start", "discussion"), ("discussion", True, "phase_change", "vote"),
    ("pk_discussion", False, "turn_start", "pk_discussion"), ("pk_discussion", True, "phase_change", "pk_vote"),
))
def test_four_boundaries_match_observed_model_fields(phase, last, marker, post_phase):
    pre, following, game = fixture_game(phase, last=last)
    replay = study.ReplayGame(game)
    speech, annotation = replay.realized_speech(pre)
    plan = consequences.build_consequence_plan(pre, speech.raw_text, annotation.actions)
    assert plan.marker.token_type == marker and plan.marker.phase == post_phase
    assert plan.marker.source == (None if last else P[1])
    actual, _, differences = replay.observed_boundary(pre, plan, speech, annotation)
    assert actual == following and differences == ()
    parent = plan_structured_history(pre)
    assert plan.token_plan.tokens[:parent.token_count] == parent.tokens
    if following is not None:
        hypothetical = tensorize_structured_plan(plan.token_plan, CAPACITY).kwargs()
        observed = tensorize_structured_plan(plan_structured_history(following), CAPACITY).kwargs()
        assert all(torch.equal(hypothetical[key], observed[key]) for key in observed)


def test_independent_hypothetical_inputs_and_same_v1_equivalence(predictor, model):
    pre, _, _ = fixture_game()
    original = pre.to_record()
    actions = (V1SpeechAction(pre.current_speaker, "support", P[2]),
               V1SpeechAction(pre.current_speaker, "vote_intent", P[3]))
    first = consequences.build_consequence_plan(pre, "发言一", actions)
    second = consequences.build_consequence_plan(pre, "发言二", actions)
    empty = consequences.build_consequence_plan(pre, "请你再解释一下", ())
    assert first.token_plan.prefix_digest != second.token_plan.prefix_digest
    assert first.model_input_digest == second.model_input_digest != empty.model_input_digest
    assert empty.token_plan.token_count + len(actions) == first.token_plan.token_count
    assert pre.to_record() == original
    a, b = predictor.forecast(first), predictor.forecast(second)
    assert a.matrix.values == b.matrix.values and torch.equal(a.log_q, b.log_q)
    ctxs, _ = consequences.current_exposure(pre, TEAM, parent_matrix(pre), model)
    with pytest.raises(ValueError, match="候选计划不一致"):
        consequences.consequence_exposure(ctxs, second, a.matrix, model)


@pytest.mark.parametrize("phase", ("discussion", "pk_discussion"))
def test_real_pre_features_and_predictions_are_bit_equivalent_to_legacy(phase, model):
    pre, _, _ = fixture_game(phase)
    matrix = parent_matrix(pre)
    old_features = []
    for ctx in contexts_for_pre(pre, TEAM):
        row = matrix.row(ctx.observer_id)
        s, m = observer_features(ctx, row).s, observer_features(ctx, row).m
        nonself = tuple(v for p, v in zip(row.seat_ids, row.values, strict=True) if p != ctx.observer_id)
        h = -fsum(v * log(v) for v in nonself if v) / log(6)
        d = .5 * fsum(abs(v - 1 / 6) for v in nonself)
        dual = float(ctx.wolf_count == 2)
        old_features.append((ctx, (s, m, s*s, m*m, s*m, h, d, float(ctx.day), float(phase == "pk_discussion"),
                                   float(len(ctx.alive_player_ids)), dual, dual*d, dual*s)))
    assert probability.pre_features(pre, TEAM, matrix) == old_features
    x = np.asarray([f for _, f in old_features])
    legacy = probability.probabilities(probability.log_probabilities(model, x), x)
    assert probability.predict_pre(pre, TEAM, matrix, model)["observers"] == {
        ctx.observer_id: probability.probability_record(q) for (ctx, _), q in zip(old_features, legacy, strict=True)}


def test_original_actor_and_pk_features_remain_fixed_at_vote(model, predictor):
    pre, _, _ = fixture_game("pk_discussion", last=True)
    matrix = parent_matrix(pre)
    contexts, current = consequences.current_exposure(pre, TEAM, matrix, model)
    actions = (V1SpeechAction(pre.current_speaker, "point_as_non_werewolf", pre.current_speaker),)
    plan = consequences.build_consequence_plan(pre, "我不是狼。", actions)
    forecast = predictor.forecast(plan)
    assert plan.marker.phase == "pk_vote"
    updated = consequences.consequence_exposure(contexts, plan, forecast.matrix, model)
    expected = []
    for ctx in contexts:
        x = probability.observer_report_features(ctx, forecast.matrix.row(ctx.observer_id).values)
        assert x[8] == 1 and x[0] == forecast.matrix.row(ctx.observer_id).values[1]
        assert updated.observers[ctx.observer_id]["team_mass"] == x[0] + x[1]
        expected.append(updated.observers[ctx.observer_id]["pT"])
    values = consequences.social_values(current, updated)
    assert values["V_soc"] == 1 - fsum(expected) / len(contexts)
    assert values["G_soc"] == pytest.approx(values["V_soc"] - (1 - current.mean_team_probability))
    assert values["V_Q"] == 1 - updated.mean_team_mass
    assert values["G_Q"] == current.mean_team_mass - updated.mean_team_mass


def followup_reports(pre, followup, *, hit=None):
    contexts = contexts_for_pre(pre, TEAM)
    links = {link.observer_id: link.observation_id for link in followup.belief_observation_links}
    reports = {}
    for i, context in enumerate(contexts):
        failed = i == 3
        status = BeliefObservationStatus.ERROR if failed else BeliefObservationStatus.SUCCESS
        sid = links[context.observer_id]
        support = () if failed or i == 0 else (hit or pre.current_speaker,)
        reports[sid] = construct_belief_observation(
            game_id=pre.game_id, attempt_id="attempt", boundary_id=followup.boundary_id, prefix_digest=followup.prefix_digest,
            observation_id=sid, observer_id=context.observer_id, observer_alive=True, day=1,
            phase=followup.public_temporal_state.phase.value, status=status, suspicion_support=support,
            attempts=({"attempt_index": 1, "call_id": f"call-{i}", "status": status.value,
                       "response_digest": None if failed else "1" * 64, "error_category": "failed" if failed else None,
                       "error_message": "failed" if failed else None},),
        )
    return reports


def test_factual_report_scoring_preserves_actor_ahead_of_nonwolf(model, predictor):
    pre, followup, game = fixture_game(single=True)
    data = study.replay_speech(study.ReplayGame(game), pre, TEAM, followup_reports(pre, followup), parent_matrix(pre),
                               model, attempt_id="attempt", predictor=predictor).data
    assert data["input_verified"] and data["next_speaker_faction"] == "non_wolf"
    assert data["acting_wolf"] == TEAM[0] and data["alive_wolf_ids"] == (TEAM[0],)
    empty, hit, _, failed, _ = data["reports"]
    assert empty["successful_empty"] and empty["state"] == 0
    assert hit["state"] == 2 and failed["status"] == "error" and "state" not in failed
    q = parent_matrix(pre).row(empty["observer_id"]).values
    expected_kl = sum(log((1 / 6) / p) / 6 for p in q if p)
    assert empty["carry"]["scores"]["q_kl"] == pytest.approx(expected_kl)
    for row in (empty, hit):
        for mode in ("carry", "updated"):
            probabilities = row[mode]["probabilities"]
            assert probabilities["q1"] == 0 and probabilities["ps"] == probabilities["pT"]
            assert sum(probabilities[f"q{k}"] for k in range(3)) == pytest.approx(1)
    summary = study.summarize_factual([data], replicates=5)
    assert summary["population"]["overall"]["successful_reports"] == 4
    assert summary["scores"]["next_speaker/non_wolf"]["rows"] == 4
    assert summary["scores"]["overall"]["metrics"]["team_brier"]["paired_bootstrap"]["valid_replicates"] == 5
    assert summary["scores"]["overall"]["calibration"]["pT"]["updated"]["bins"]


def test_factual_teammate_hit_remains_e1_when_teammate_speaks_next(model, predictor):
    pre, followup, game = fixture_game("pk_discussion")
    reports = followup_reports(pre, followup, hit=followup.current_speaker)
    data = study.replay_speech(study.ReplayGame(game), pre, TEAM, reports, parent_matrix(pre), model,
                               attempt_id="attempt", predictor=predictor).data
    assert data["next_speaker_faction"] == "wolf" and data["acting_wolf"] == TEAM[0]
    hit = data["reports"][1]
    assert (hit["state"], hit["ys"], hit["yT"]) == (1, 0, 1)
    assert hit["updated"]["probabilities"]["ps"] < hit["updated"]["probabilities"]["pT"]


@pytest.mark.parametrize("phase", ("discussion", "pk_discussion"))
def test_last_speech_predicts_and_keeps_missing_labels_explicit(phase, model, predictor):
    pre, _, game = fixture_game(phase, last=True)
    data = study.replay_speech(study.ReplayGame(game), pre, TEAM, {}, parent_matrix(pre), model,
                               attempt_id="attempt", predictor=predictor).data
    assert data["status"] == "predicted" and data["input_verified"]
    assert data["report_boundary"] == "immediate_vote_report_uncollected" and data["reports"] == []
    assert data["prediction"]["V_soc"] is not None
    summary = study.summarize_factual([data], replicates=5)
    assert summary["population"]["overall"]["immediate_reports_uncollected_observers"] == 5
    assert summary["scores"] == {}


def test_candidate_none_failure_and_collisions_are_separate(model, predictor):
    pre, _, _ = fixture_game()
    candidates = (SpeechCandidate("INFORM", "前情请回顾。", None), SpeechCandidate("INQUIRE", "请3号具体解释。", P[2]),
                  SpeechCandidate("ADVOCATE", "我投3号。", None), SpeechCandidate("ADVOCATE", "解析异常。", None))
    calls = []

    def parse(**request):
        calls.append(request)
        speech = request["speech"]
        if speech == candidates[3].text:
            return SpeechParseAuditResult([], None, "parser_error", "Error", "unparsed")
        actions = [[pre.current_speaker, "vote_intent", P[2]]] if speech == candidates[2].text else []
        return SpeechParseAuditResult(actions, "parsed", "ok", None, None)

    result = consequences.score_candidates(SpeechCandidates(pre, "topic", candidates), TEAM, parent_matrix(pre), model,
                                           perceiver=SimpleNamespace(parse_with_audit=parse), predictor=predictor)
    assert all(call["phase"] == "speech" and call["speaker"] == 1 for call in calls)
    a, b, c, failed = result["candidates"]
    assert [row["v1_status"] for row in (a, b, c, failed)] == ["no_action", "no_action", "ok", "parser_error"]
    assert a["prediction"]["plan_digest"] != b["prediction"]["plan_digest"]
    assert a["prediction"]["q"] == b["prediction"]["q"] and a["prediction"]["V_soc"] == b["prediction"]["V_soc"]
    assert failed["prediction"] is None and predictor._model.calls == 2
    summary = study.summarize_candidates([{**result, "status": "evaluated"}])
    assert summary["overall"]["parse_success_rate"] == .75
    assert summary["overall"]["no_action_rate_given_success"] == pytest.approx(2/3)
    assert summary["overall"]["v1_pair_collision_rate"] == pytest.approx(1/3)
    assert summary["overall"]["pres_two_model_inputs_rate"] == 1
    assert summary["semantic_compression_examples"]["INFORM/INQUIRE"]["shared_normalized_actions"] == []


def test_candidate_cli_requires_already_generated_jsonl(tmp_path, monkeypatch):
    destination = tmp_path / "candidate-consequences"
    monkeypatch.setitem(study.DESTINATIONS, "candidates", destination)
    path = tmp_path / "candidates.jsonl"
    args = ["candidates", "--capsule", "mock", "--capsule-digest", "unused", "--runtime-config", "mock",
            "--candidate-jsonl", str(path)]
    with pytest.raises(FileNotFoundError, match="既有固定样本"):
        study.main(args)
    plan = {"type": "sample", "selected": [{"game_id": "g", "boundary_id": "b"}]}
    path.write_bytes(canonical_jsonl_bytes([plan, {"type": "settings"}]))
    with pytest.raises(ValueError, match="数量不符"):
        study.read_candidate_study(path)
    row = {"type": "pre", **plan["selected"][0], "status": "input_error", "candidates": []}
    path.write_bytes(canonical_jsonl_bytes([plan, {"type": "settings"}, row]))
    assert study.read_candidate_study(path)[:3] == (plan, {"type": "settings"}, [row])


def test_candidate_cli_reuses_fixed_sources_and_preserves_original_text(monkeypatch, tmp_path, model, predictor):
    pre, _, _ = fixture_game(single=True)
    q = parent_matrix(pre)
    selected = study.candidate_study.opportunity(pre, TEAM, fold=0, attempt_id="attempt")
    member = {"game_id": pre.game_id, "bundle_digest": "bundle", "attempt_id": "attempt"}
    source = {"publication_digest": study.frozen.PUBLICATION_DIGEST, "split_digest": study.frozen.SPLIT_DIGEST,
              "fold_manifest_digest": "folds", "role_sidecar_digest": "roles", "q_manifest_digest": study.frozen.Q_DIGEST}
    publication = SimpleNamespace(manifest_digest=study.frozen.PUBLICATION_DIGEST, manifest={
        "split_manifest_digest": study.frozen.SPLIT_DIGEST, "development_fold_manifest_digest": "folds",
        "role_sidecar_digest": "roles", "development_games": [member],
    })
    game = SimpleNamespace(game_id=pre.game_id, bundle_digest="bundle", authoritative_pre_prefixes=(pre,))
    fold = SimpleNamespace(fold_index=0, game_ids=(pre.game_id,))
    view = SimpleNamespace(game_ids=(pre.game_id,), games=(game,), fold_manifest=SimpleNamespace(folds=(fold,)))
    roles = SimpleNamespace(game_id=pre.game_id, bundle_digest="bundle",
                            role_assignment=tuple((seat, "Werewolf" if seat in TEAM else "Villager") for seat in P))
    q_record = {**{key: selected[key] for key in study.development.PRE_IDENTITY}, "q": q.values}
    original = "  请3号解释之前的判断。\n"
    archived = {"type": "pre", **selected, "status": "ok", "topic": "判断依据",
                "input": study.candidate_study.build_candidate_context(pre, TEAM, q),
                "candidates": [{"function": "INQUIRE", "text": original, "addressee": P[2]}]}
    path = tmp_path / "candidates.jsonl"
    path.write_bytes(canonical_jsonl_bytes([{"type": "sample", "folds": (0, 1, 2, 3), "sources": source,
                                           "selected": [selected]}, {"type": "settings"}, archived]))
    monkeypatch.setitem(study.DESTINATIONS, "candidates", tmp_path / "candidate-consequences")
    monkeypatch.setattr(study, "open_artifact_envelope", lambda *args, **kwargs: publication)
    monkeypatch.setattr(study, "load_probability_model", lambda *args: (model, {"probability_manifest_digest": "prob"}))
    monkeypatch.setattr(study.candidate_study, "open_games", lambda ids: SimpleNamespace(public_view=view))
    monkeypatch.setattr(study, "open_role_sidecar", lambda batch: SimpleNamespace(games=(roles,)))
    monkeypatch.setattr(study.development, "_frozen_q", lambda *args: {study.candidate_study.pre_key(selected): q_record})
    monkeypatch.setattr(study.consequences, "SpeechConsequencePredictor", lambda *args, **kwargs: SimpleNamespace(
        identity={"capsule_digest": "capsule"}, forecast=predictor.forecast,
    ))
    seen = []

    def parse(**request):
        seen.append(request["speech"])
        return SpeechParseAuditResult([], "NONE", "ok", None, None)

    monkeypatch.setattr(study, "load_perceiver", lambda path: (SimpleNamespace(parse_with_audit=parse), {"model": "mock"}))
    monkeypatch.setattr(study.torch, "set_num_threads", lambda *args: None)
    study.main(["candidates", "--candidate-jsonl", str(path), "--runtime-config", "mock", "--capsule", "mock",
                "--capsule-digest", "mock"])
    results = json.loads((tmp_path / "candidate-consequences" / "results.json").read_bytes())
    record = json.loads((tmp_path / "candidate-consequences" / "records.jsonl").read_bytes())
    assert results["by_function"]["INQUIRE"]["no_action_rate_given_success"] == 1
    assert record["topic"] == archived["topic"] and record["candidates"][0]["text"] == original
    assert seen == [original]


def test_factual_cli_reuses_loader_and_publishes_real_cut_separately(monkeypatch, tmp_path, model, predictor):
    pre, _, game = fixture_game("pk_discussion", last=True)
    publication = SimpleNamespace(manifest_digest=study.frozen.PUBLICATION_DIGEST,
                                  manifest={"split_manifest_digest": study.frozen.SPLIT_DIGEST})
    monkeypatch.setitem(study.DESTINATIONS, "factual", tmp_path / "replay")
    monkeypatch.setattr(study, "open_artifact_envelope", lambda *args, **kwargs: publication)
    monkeypatch.setattr(study, "load_probability_model", lambda *args: (model, {"probability_manifest_digest": "prob"}))
    monkeypatch.setattr(study.consequences, "SpeechConsequencePredictor", lambda *args, **kwargs: SimpleNamespace(
        identity={"origin": {"publication_digest": "frozen_training"}, "capsule_digest": "capsule"},
        forecast=predictor.forecast,
    ))

    def loader(folds, *, build_samples, visit_game):
        assert folds == (4,)
        visit_game(4, game, TEAM, publication_digest="development", attempt_id="attempt")
        q = {**{key: getattr(pre, key) for key in study.development.PRE_IDENTITY}, "q": parent_matrix(pre).values}
        rows = build_samples(pre, TEAM, {}, q, attempt_id="attempt")
        return {4: rows}, {"publication_digest": "development"}

    monkeypatch.setattr(study.development, "load_development", loader)
    # 避免在共享测试进程中改变全局线程设置。
    monkeypatch.setattr(study.torch, "set_num_threads", lambda *args: None)
    study.main(["factual", "--capsule", "mock", "--capsule-digest", "mock"])
    results = json.loads((tmp_path / "replay" / "results.json").read_bytes())
    row = json.loads((tmp_path / "replay" / "records.jsonl").read_bytes())
    assert results["primary_metric"] == "team_brier" and results["population"]["overall"]["predicted_pres"] == 1
    assert results["final_tom_training_population"]["games"] == 1500
    assert row["prediction"]["marker"]["phase"] == "pk_vote" and row["reports"] == []
