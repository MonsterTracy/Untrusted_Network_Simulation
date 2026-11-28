"""显式依据、三臂共用评价和精确容忍量用合成 PRE 验证。"""

from dataclasses import asdict, replace
from fractions import Fraction
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts import evaluate_observer_dtrs_action_selection as study
from tests.observer_dtrs.test_cognitive_value import report_model
from tests.observer_dtrs.test_speech_candidates import P, candidate, discussion_pre, frozen_q, mock_agent, request_context
from tests.observer_dtrs.test_speech_consequences import predictor
from werewolf.agents.llm_agent import GameplayActionValidationError
from werewolf.artifact_io import canonical_jsonl_bytes, sha256_bytes
from werewolf.observer_dtrs import action_selection as select
from werewolf.observer_dtrs.report_state_decision import FrozenSharedRule, Thresholds
from werewolf.observer_dtrs.report_state_probability import probability_record
from werewolf.observer_dtrs.speech_candidates import build_candidate_context, parse_speech_candidates
from werewolf.speech.speech_perceiver import SpeechParseAuditResult


TEAM = P[:2]
RULE = FrozenSharedRule(Thresholds.shared(.2, .75), "rule", "probability")


def premise(event="speaker_hit", direction="POS", observer=P[2]):
    return {"observer_id": observer, "event": event, "direction": direction}


def payload(pre, *, declared=True):
    public_id = next(event.event_id for event in pre.public_event_history.events if event.event_type == "public_speech")
    rows = [candidate("INFORM", "核对上一位玩家的公开表态。", evidence=(public_id,)),
            candidate("ADVOCATE", "建议回应这一轮对狼队的怀疑。", premises=(premise("team_hit"),)),
            candidate("ADVOCATE", "我想直接回应3号对我的指认。", premises=(premise(),)),
            candidate("INQUIRE", "3号，你最关注哪条公开线索？", P[2], premises=(premise(direction="NEG"),))]
    if not declared:
        rows = [{key: row[key] for key in ("function", "text", "addressee")} for row in rows]
    return {"topic": "公开讨论依据", "candidates": rows}


def parse(pre, rows, *, archived=False, team=TEAM):
    context = build_candidate_context(pre, team, frozen_q(pre))
    return parse_speech_candidates(json.dumps(rows), pre=pre, context=context, archived=archived)


def audit_rows(batch):
    return [{"index": i, "status": "identified", "reason": "已核对原文及公开事件",
             "required_premises": [asdict(p) for p in c.required_premises] if hasattr(c, "required_premises") else [],
             "public_evidence_refs": list(getattr(c, "public_evidence_refs", ()))} for i, c in enumerate(batch.candidates)]


def verification(batch, rows=None):
    context = select.basis_verification_context(batch, TEAM)
    return select.parse_basis_verification(json.dumps({"candidates": audit_rows(batch) if rows is None else rows}), batch, context)


def test_new_generation_validates_declarations_before_accepting_batch():
    pre = discussion_pre()
    source = payload(pre)
    agent, chat = mock_agent(source["candidates"])
    batch = agent.generate_speech_candidates(pre, known_wolf_ids=TEAM, q=frozen_q(pre))
    assert select.has_declarations(batch) and not batch.candidates[0].required_premises
    assert batch.candidates[1].required_premises[0].event == "team_hit"
    assert batch.candidates[0].text == source["candidates"][0]["text"]
    assert "report_predictions" not in request_context(chat)
    chat.assert_called_once()

    # 不合约的批次只能报错，不能再调用生成器寻找替代发言。
    bad, bad_call = mock_agent(source["candidates"][1:])
    with pytest.raises(GameplayActionValidationError, match="声明无确定性预测前提"):
        bad.generate_speech_candidates(pre, known_wolf_ids=TEAM, q=frozen_q(pre))
    bad_call.assert_called_once()


@pytest.mark.parametrize(("override", "message"), (
    ({"required_premises": [premise(), premise()]}, "重复或矛盾"),
    ({"required_premises": [premise(), premise("team_hit", "NEG")]}, "包含关系"),
    ({"required_premises": [premise(observer=P[1])]}, "合格观察者"),
    ({"required_premises": [premise("specific_teammate")]}, "合法事件"),
    ({"public_evidence_refs": ["future-event"]}, "当前 PRE 公开事件"),
))
def test_generation_rejects_invalid_premise_logic_and_public_sources(override, message):
    pre = discussion_pre()
    source = payload(pre)
    source["candidates"][1].update(override)
    with pytest.raises(GameplayActionValidationError, match=message):
        parse(pre, source)


def test_single_wolf_disallows_other_only_hit_combination_and_archives_stay_undeclared():
    pre = discussion_pre(alive=(P[0], *P[2:]))
    source = payload(pre)
    source["candidates"][1]["required_premises"] = [premise(direction="NEG"), premise("team_hit")]
    with pytest.raises(GameplayActionValidationError, match="包含关系"):
        parse(pre, source)
    archived = payload(pre, declared=False)
    with pytest.raises(GameplayActionValidationError, match="依据字段"):
        parse(pre, archived)
    batch = parse(pre, archived, archived=True)
    assert not select.has_declarations(batch)
    assert asdict(batch.candidates[0]) == archived["candidates"][0]
    verdict = verification(batch)
    assert verdict["origin"] == "retrospective" and verdict["candidates"][0]["status"] == "no_predictive_premise"
    with pytest.raises(ValueError, match="原始依据声明"):
        select.compare_actions(batch, {}, RULE, verdict, {}, {}, delta=0)


def test_one_blind_audit_records_omissions_and_uncertainty_in_original_order():
    pre = discussion_pre()
    batch = parse(pre, payload(pre))
    checked = audit_rows(batch)
    checked[2]["required_premises"].append(premise("team_hit"))
    checked[3].update(status="undetermined", reason="提问背后的预设无法确认")
    agent, chat = mock_agent([], raw=json.dumps({"candidates": list(reversed(checked))}))
    result = study.verify_once(batch, TEAM, agent)
    assert [item["status"] for item in result["candidates"]] == [
        "no_predictive_premise", "predictive_premise", "inconsistent", "undetermined"]
    assert result["candidates"][2]["omitted_premises"] == [premise("team_hit")]
    assert result["candidates"][3]["required_premises"] is None
    assert set(request_context(chat)) == {"speaker", "observer_ids", "alive_wolf_ids", "public_events", "topic", "candidates"}
    assert result["candidates"][0]["text_sha256"] == sha256_bytes(batch.candidates[0].text.encode())
    chat.assert_called_once()


def test_audit_distinguishes_declared_inconsistency_from_unusable_inference():
    pre = discussion_pre()
    batch = parse(pre, payload(pre))
    checked = audit_rows(batch)
    checked[0].update(status="inconsistent", public_evidence_refs=["future-event"])
    checked[1].update(public_evidence_refs=["future-event"])
    first, second, *_ = verification(batch, checked)["candidates"]
    assert first["status"] == "inconsistent" and second["status"] == "undetermined"
    assert first["public_evidence_refs"] is None and second["public_evidence_refs"] is None


def test_audit_failure_retains_unresolved_rows_and_does_not_repeat():
    pre = discussion_pre()
    batch = parse(pre, payload(pre))
    agent, chat = mock_agent([], raw=json.dumps({"candidates": audit_rows(batch)[:1]}))
    result = study.verify_once(batch, TEAM, agent)
    assert {row["status"] for row in result["candidates"]} == {"undetermined"}
    assert all(row["required_premises"] is None for row in result["candidates"])
    assert result["error"]["type"] == "GameplayActionValidationError"
    chat.assert_called_once()


def forecast(gain, *, opportunity=.5, stability=.5):
    return {"predicted_redecision_opportunity": opportunity, "predicted_target_stability": stability,
            "predicted_cognitive_gain": gain}


def test_information_keeps_fixed_denominator_and_uses_only_proven_structural_probability():
    observers = P[2:4]
    values = {P[2]: forecast(.6, stability=None), P[3]: forecast(-.4, stability=None)}
    single = select.aggregate_information(observers, values, 1)
    assert single["V_info"] == pytest.approx(.05)
    assert single["observers"][P[2]]["pi_K_predicted"] is None
    assert single["observers"][P[2]]["K_source"] == "single_wolf_structure"
    assert select.aggregate_information(observers, values, 2)["V_info"] is None
    assert select.aggregate_information(observers, {P[2]: values[P[2]]}, 1)["V_info"] is None
    missing_gain = {**values, P[3]: forecast(None, opportunity=0., stability=None)}
    assert select.aggregate_information(observers, missing_gain, 1)["V_info"] is None


def test_exact_breakpoints_tie_rules_and_failures_within_band():
    scores = [{"V_soc": s, "V_info": i} for s, i in ((.9, .1), (.8, .4), (.8, .4), (.7, None))]
    before = select.select_with_sensitivity(scores, range(4), ".09999999999999999999")
    at = select.select_with_sensitivity(scores, range(4), ".1", shared_best=Fraction(19, 20))
    assert before["selected_index"] == 0 and at["selected_index"] == 1 and at["band_indices"] == [0, 1, 2]
    assert at["used_delta_exact"] == "1/10" and at["gap_to_unfiltered_best"] == pytest.approx(.15)
    assert [(row["delta_from_exact"], row["status"]) for row in at["sensitivity"]] == [
        ("0", "selected"), ("1/10", "selected"), ("1/5", "info_value_missing_in_band")]
    assert select.select_with_sensitivity(scores, range(4), ".2")["missing_info_indices"] == [3]
    scores[0]["V_info"] = .4
    assert select.select_with_sensitivity(scores, range(4), ".1")["selected_index"] == 0
    scores[3]["V_soc"] = None
    missing = select.select_with_sensitivity(scores, range(4), 0)
    assert missing["status"] == "social_value_missing" and missing["V_soc_star"] is None and not missing["sensitivity"]
    assert select.select_with_sensitivity(scores, [], 0)["status"] == "no_certified_feasible_candidate"
    with pytest.raises(ValueError, match="delta"):
        select.select_with_sensitivity(scores, range(4), "-.1")


def comparison_inputs(batch):
    pre = batch.pre
    identity = {**{key: getattr(pre, key) for key in select.PRE_FIELDS}, "acting_wolf": pre.current_speaker, "alive_wolf_ids": TEAM}
    observer_ids = P[2:]
    panel = {**identity, "observers": {seat: probability_record((.3, .2, .5)) for seat in observer_ids}}
    social, info = [], []
    for i, c in enumerate(batch.candidates):
        key = {"index": i, "text_sha256": sha256_bytes(c.text.encode())}
        social.append({**key, "prediction": {"V_soc": (.8, .88, .9, .78)[i]}, "parse": {"normalized_actions": []},
                       "v1_status": "no_action", "prediction_status": "predicted"})
        info.append({**key, "observers": {seat: forecast((.1, .4, .6, .2)[i]) for seat in observer_ids}})
    return panel, {**identity, "observer_ids": observer_ids, "candidates": social}, {**identity, "head_statuses": {}, "candidates": info}


def test_three_arms_change_only_permissions_and_report_set_nesting():
    pre = discussion_pre()
    batch = parse(pre, payload(pre))
    panel, social, info = comparison_inputs(batch)
    out = select.compare_actions(batch, panel, RULE, verification(batch), social, info, delta=0)
    assert [row["feasible_indices"] for row in out["arms"].values()] == [[0, 1, 2, 3], [0, 1, 3], [0]]
    assert [row["selected_index"] for row in out["arms"].values()] == [2, 1, 0]
    assert out["feasible_nesting"] and all(out["selection_differences"].values())
    assert out["arms"]["tom_3wd"]["gap_to_unfiltered_best"] == pytest.approx(.1)
    assert out["unfiltered_V_soc_star"] == .9
    assert out["candidates"][0]["v1_parse"]["normalized_actions"] == []
    changed_declaration = replace(batch.candidates[0], public_evidence_refs=())
    changed_batch = replace(batch, candidates=(changed_declaration, *batch.candidates[1:]))
    with pytest.raises(ValueError, match="核验与原始声明"):
        select.compare_actions(changed_batch, panel, RULE, verification(batch), social, info, delta=0)
    info["candidates"][0]["text_sha256"] = "different"
    with pytest.raises(ValueError, match="原文摘要"):
        select.compare_actions(batch, panel, RULE, verification(batch), social, info, delta=0)


def test_common_social_references_keep_population_and_missingness_separate():
    pre = discussion_pre()
    batch = parse(pre, payload(pre))
    panel, social, info = comparison_inputs(batch)
    checked = audit_rows(batch)
    checked[2].update(status="undetermined")
    verdict = verification(batch, checked)
    out = select.compare_actions(batch, panel, RULE, verdict, social, info, delta=0)
    assert out["unfiltered_V_soc_star"] == .9
    assert out["basis_certified_V_soc_star"] == .88
    assert out["arms"]["tom_3wd"]["gap_to_basis_certified_best"] == pytest.approx(.08)
    social["candidates"][2]["prediction"] = None
    out = select.compare_actions(batch, panel, RULE, verdict, social, info, delta=0)
    assert out["unfiltered_V_soc_star"] is None
    assert out["unfiltered_missing_social_indices"] == [2]
    assert out["arms"]["tom_3wd"]["gap_to_basis_certified_best"] == pytest.approx(.08)


def test_missing_independent_certification_preserves_reasons_without_substituting():
    pre = discussion_pre()
    batch = parse(pre, payload(pre))
    panel, social, info = comparison_inputs(batch)
    checked = audit_rows(batch)
    checked[0].update(status="undetermined")
    out = select.compare_actions(batch, panel, RULE, verification(batch, checked), social, info, delta=0)
    assert out["status"] == "independent_candidate_not_certified"
    assert out["arms"]["tom_only"]["feasible_indices"] == [1, 2, 3]
    assert out["arms"]["tom_3wd"]["status"] == "no_certified_feasible_candidate"
    assert all(row["selected_index"] is None for row in out["arms"].values())


def test_real_offline_value_path_binds_original_text_to_v1_and_calls_each_model_once_per_batch(predictor):
    pre = discussion_pre()
    batch = parse(pre, payload(pre))
    _, _, info = comparison_inputs(batch)
    information = Mock(return_value=info)
    perception = Mock(return_value=SpeechParseAuditResult([], "NONE", "ok", None, None))
    out = study.compare_pre(batch, TEAM, frozen_q(pre), verification(batch), model=report_model(), rule=RULE,
        cognitive=SimpleNamespace(predict_candidates=information), perceiver=SimpleNamespace(parse_with_audit=perception),
        predictor=predictor, delta=0)
    information.assert_called_once()
    assert [call.kwargs["speech"] for call in perception.call_args_list] == [c.text for c in batch.candidates]
    assert [row["text"] for row in out["candidates"]] == [c.text for c in batch.candidates]
    assert {row["v1_status"] for row in out["candidates"]} == {"no_action"}
    assert len({row["social_prediction"]["model_input_digest"] for row in out["candidates"]}) == 1
    assert predictor._model.calls == 1


@pytest.mark.parametrize("mode", ["diagnose", "compare"])
def test_offline_cli_publishes_diagnostics_or_paired_evidence_without_mutating_source(mode, tmp_path, monkeypatch):
    pre = discussion_pre()
    q = frozen_q(pre)
    generation = study.generation
    selected = generation.opportunity(pre, TEAM, fold=0, attempt_id="attempt")
    archived = {"type": "pre", **selected, "status": "ok", "input": build_candidate_context(pre, TEAM, q),
                **payload(pre, declared=mode == "compare")}
    source_path = tmp_path / "source.jsonl"
    source = canonical_jsonl_bytes([{"type": "sample", "selected": [selected], "sources": {"publication_digest": "frozen"}},
                                   {"type": "settings"}, archived])
    source_path.write_bytes(source)
    published = SimpleNamespace(manifest_digest=study.replay.frozen.PUBLICATION_DIGEST,
                                manifest={"split_manifest_digest": study.replay.frozen.SPLIT_DIGEST})
    q_record = {**{name: getattr(pre, name) for name in select.PRE_FIELDS}, "q": q.values}
    monkeypatch.setattr(study, "open_artifact_envelope", lambda *args, **kwargs: published)
    monkeypatch.setattr(study.replay, "load_candidate_inputs", lambda *args: {generation.pre_key(selected): (pre, TEAM, q_record)})
    agent = SimpleNamespace(verify_speech_candidate_bases=Mock(side_effect=lambda b, **_: verification(b)))
    agent_settings = {name: None for name in ("runtime_config_sha256", "backend", "model", "temperature", "max_tokens",
                                               "max_retries", "chat_template_kwargs")}
    monkeypatch.setattr(generation, "load_agent", lambda *args: (agent, agent_settings))

    def compare(batch, team, matrix, verdict):
        panel, social, info = comparison_inputs(batch)
        return select.compare_actions(batch, panel, RULE, verdict, social, info, delta=0)

    evaluate = Mock(return_value=(compare, {"delta_exact": "0"}))
    monkeypatch.setattr(study, "_comparison", evaluate)
    output = tmp_path / mode
    monkeypatch.setitem(study.DESTINATIONS, mode, output)
    args = [mode, "--candidate-jsonl", str(source_path), "--runtime-config", "mock"]
    if mode == "compare":
        args += ["--delta", "0", "--capsule", "mock", "--capsule-digest", "mock"]
    study.main(args)
    record = json.loads((output / "records.jsonl").read_bytes())
    summary = json.loads((output / "results.json").read_bytes())
    assert source_path.read_bytes() == source and summary["pres"] == 1
    agent.verify_speech_candidate_bases.assert_called_once()
    if mode == "compare":
        assert record["arms"]["tom_3wd"]["selected_index"] == 0 and summary["paired_pres"] == 1
    else:
        evaluate.assert_not_called()
        assert record["status"] == "basis_diagnostic" and "arms" not in record and summary["paired_pres"] == 0
