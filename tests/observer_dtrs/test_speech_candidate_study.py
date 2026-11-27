import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts import analyze_observer_dtrs_speech_candidates as study
from tests.observer_dtrs.test_speech_candidates import P, PublicWitness, candidate, discussion_pre, frozen_q
from werewolf.agents.gpt_agent import GPTAgent
from werewolf.backends.openai_compatible import OpenAICompatibleBackend
from werewolf.observer_dtrs.speech_candidates import candidate_response_properties


def sample_row(game, boundary, replies):
    return {"game_id": game, "boundary_id": boundary, "prefix_digest": "c" * 64, "step_index": 1,
            "attempt_id": "attempt", "fold": 0, "position": "discussion_middle", "alive_wolves": 2,
            "replyable_observer_count": replies}


def test_sample_prioritizes_distinct_games_and_covers_reply_groups_stably():
    rows = [sample_row("a", "one", 0), sample_row("a", "two", 1), sample_row("b", "three", 1),
            sample_row("b", "four", 2), sample_row("c", "five", 4), sample_row("c", "six", 5)]
    strata, selected = study.select_samples(rows, "a" * 64)
    assert (strata, selected) == study.select_samples(reversed(rows), "a" * 64)
    assert len(strata) == 10
    assert {row["game_id"] for row in selected} == {"a", "b", "c"}
    assert {min(row["replyable_observer_count"], 2) for row in selected} == {0, 1, 2}
    populated, = [row for row in strata if row["available_pres"]]
    assert populated == {"position": "discussion_middle", "alive_wolves": 2, "available_pres": 6,
                         "available_games": 3, "replyable_observer_counts": {0: 1, 1: 2, 2: 1, 4: 1, 5: 1},
                         "selected_pres": 3}


def test_sample_counts_sparse_strata_and_keeps_useful_same_game_opportunities():
    rows = [sample_row("a", "one", 0), sample_row("a", "two", 1), sample_row("a", "three", 2),
            sample_row("b", "four", 1), {**sample_row("c", "five", 0), "position": "pk_last"}]
    strata, selected = study.select_samples(rows, "a" * 64)
    ordinary = [row for row in selected if row["position"] == "discussion_middle"]
    assert len(ordinary) == 3 and {row["game_id"] for row in ordinary} == {"a", "b"}
    assert {row["replyable_observer_count"] for row in ordinary} == {0, 1, 2}
    assert len(selected) == 4
    assert sum(row["available_pres"] for row in strata) == 5
    assert sum(row["available_games"] == 0 for row in strata) == 8


@pytest.mark.parametrize("phase,last,prior,expected,count", [
    ("discussion", False, (), "discussion_first", 5),
    ("discussion", False, (P[-1],), "discussion_middle", 4),
    ("discussion", False, P[1:], "discussion_last", 0),
    ("pk_discussion", False, (), "pk_nonlast", 1),
    ("pk_discussion", True, (), "pk_last", 0),
])
def test_strata_follow_existing_public_speech_order(phase, last, prior, expected, count):
    if phase == "discussion":
        pre = discussion_pre(prior=prior)
        team = P[:2]
    else:
        witness = PublicWitness(phase)
        if last:
            witness.speech(P[0], "no_commitment", None, "我暂时不作明确表态。")
            witness.event("turn_start", speaker=P[1])
        pre = witness.pre()
        team = (pre.current_speaker, P[3])
    row = study.opportunity(pre, team, fold=1, attempt_id="attempt")
    assert (row["position"], row["alive_wolves"], row["replyable_observer_count"]) == (expected, 2, count)


def test_load_sample_binds_fold_members_roles_and_q_without_reading_reports(monkeypatch):
    pre = discussion_pre(alive=(P[0], *P[2:]), prior=(P[-1],))
    game = SimpleNamespace(game_id=pre.game_id, bundle_digest="bundle", authoritative_pre_prefixes=(pre,))
    member = {"game_id": pre.game_id, "bundle_digest": "bundle", "attempt_id": "attempt"}
    fold4 = {"game_id": "fold4-game"}
    manifest = {"split_manifest_digest": study.frozen.SPLIT_DIGEST, "development_games": [member, fold4],
                "development_fold_manifest_digest": "fold", "role_sidecar_digest": "roles"}
    publication = SimpleNamespace(manifest=manifest, manifest_digest=study.frozen.PUBLICATION_DIGEST)
    monkeypatch.setattr(study, "open_artifact_envelope", Mock(return_value=publication))
    monkeypatch.setattr(study, "read_artifact_file", Mock(return_value=json.dumps({"folds": [
        {"fold_index": 0, "held_out_games": [member]}, {"fold_index": 4, "held_out_games": [fold4]},
    ]})))
    open_games = Mock(return_value=SimpleNamespace(public_view=SimpleNamespace(games=(game,), game_ids=(pre.game_id,))))
    monkeypatch.setattr(study, "open_games", open_games)
    role_game = SimpleNamespace(game_id=pre.game_id, bundle_digest="bundle",
                                role_assignment=tuple((seat, "Werewolf" if seat in P[:2] else "Villager") for seat in P))
    monkeypatch.setattr(study, "open_role_sidecar", Mock(return_value=SimpleNamespace(games=(role_game,))))
    identity = {name: getattr(pre, name) for name in study.PRE_IDENTITY}
    key = (pre.game_id, pre.boundary_id)
    frozen_matrix = frozen_q(pre)
    q_loader = Mock(return_value={key: {**identity, "q": frozen_matrix.values}})
    monkeypatch.setattr(study, "_frozen_q", q_loader)

    plan, inputs = study.load_sample()

    open_games.assert_called_once_with((pre.game_id,))
    q_loader.assert_called_once_with(manifest, {pre.game_id})
    assert plan["q_coverage"] == {"matched": 1, "missing": 0, "invalid": 0, "unmatched_q_records": 0}
    row, = plan["selected"]
    assert row == study.opportunity(pre, P[:2], fold=0, attempt_id="attempt")
    assert row["alive_wolves"] == 1 and row["replyable_observer_count"] == 4
    assert inputs == {key: (P[:2], frozen_matrix, None)}

    q_loader.return_value = {key: {**identity, "step_index": pre.step_index + 1, "q": frozen_matrix.values}}
    invalid_plan, invalid_inputs = study.load_sample()
    assert invalid_plan["q_coverage"]["invalid"] == 1
    assert invalid_plan["selected"] == [row]
    assert invalid_inputs[key][1] is None and "身份不符" in invalid_inputs[key][2]["message"]


def mock_backend(content, finish_reason="stop"):
    client = Mock()
    client.max_retries = 0
    create = client.chat.completions.create
    create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish_reason)],
        usage={"prompt_tokens": 101, "completion_tokens": 25, "total_tokens": 126},
    )
    return OpenAICompatibleBackend(client=client, supports_json_schema=True), create


def fixture_sample():
    pre = discussion_pre()
    row = study.opportunity(pre, P[:2], fold=2, attempt_id="attempt")
    return pre, row, frozen_q(pre)


def test_run_emits_fixed_sample_before_one_request_and_reconstructs_its_exact_payload(monkeypatch, capsys):
    pre, row, q = fixture_sample()
    key = study.pre_key(row)
    plan = {"type": "sample", "selected": [row], "request_limit": 1}
    monkeypatch.setattr(study, "load_sample", lambda: (plan, {key: (P[:2], q, None)}))
    monkeypatch.setattr(study, "load_selected_pres", lambda selected: {key: pre})
    # 语义质量留给人工，结构合法但空泛的陈述也应原样输出。
    payload = {"topic": "核对转述", "candidates": [candidate("INFORM", "  我想说几句。\n")]}
    backend, create = mock_backend(json.dumps(payload))
    load_backends = Mock(return_value={"local_qwen": backend})
    monkeypatch.setattr(study, "load_named_backends", load_backends)
    before_request = []
    reply = create.return_value

    def create_after_plan(**kwargs):
        before_request.extend(json.loads(line) for line in capsys.readouterr().out.splitlines())
        return reply

    create.side_effect = create_after_plan
    study.main(["--runtime-config", str(study.ROOT / "configs/runtime/local-qwen35-9b.yaml")])

    create.assert_called_once()
    assert load_backends.call_args.kwargs == {"env_file": None, "max_retries": 0}
    assert before_request[0] == plan
    settings = before_request[1]
    assert settings["type"] == "settings" and settings["semantic_quality"] == "human_review_pending"
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "ok" and result["candidates"] == payload["candidates"]
    assert result["phase_speaker_order"] == [P[-1], *P[:-1]]
    assert result["input"]["public_events"] == pre.public_event_history.to_records()
    assert result["input"]["known_wolf_team"] == list(P[:2])
    assert result["request"]["finish_reason"] == "stop" and result["request"]["total_tokens"] == 126
    request = create.call_args.kwargs
    assert request["messages"] == [{"role": "user", "content": settings["prompt_prefix"] + "\n\n"
                                    + json.dumps(result["input"], ensure_ascii=False)}]
    expected_properties = candidate_response_properties(result["input"]["replyable_observers"])
    assert request["response_format"]["json_schema"]["schema"]["properties"] == expected_properties
    assert request["model"] == settings["model"] == "qwen35-9b"
    assert request["temperature"] == settings["temperature"] == 0.7
    assert request["max_tokens"] == settings["max_tokens"] == 2048
    assert request["extra_body"] == {"chat_template_kwargs": settings["chat_template_kwargs"]}


@pytest.mark.parametrize("content,fail,expected", [
    ("{", False, "validation_error"),
    (None, False, "validation_error"),
    (None, True, "backend_error"),
])
def test_rejected_responses_and_backend_errors_preserve_the_sample_and_telemetry(content, fail, expected):
    pre, row, q = fixture_sample()
    backend, create = mock_backend(content)
    if fail:
        create.side_effect = RuntimeError("mock dispatch failure")
    agent = GPTAgent(backend=backend, model_name="qwen35-9b")
    result = study.generate_record(row, pre, P[:2], q, None, agent)
    create.assert_called_once()
    assert result["status"] == expected and result["error"]["message"]
    assert result["candidates"] == [] and result["boundary_id"] == pre.boundary_id
    assert result["request"]["finish_reason"] == ("UNKNOWN" if fail else "stop")


def test_input_error_records_available_context_without_calling_backend():
    pre, row, _ = fixture_sample()
    backend, create = mock_backend(None)
    result = study.generate_record(row, pre, P[:2], None, {"type": "ValueError", "message": "缺少 Q"},
                                   GPTAgent(backend=backend, model_name="qwen35-9b"))
    create.assert_not_called()
    assert result["status"] == "input_error" and result["request"] is None
    assert result["input"]["wolf_suspicion_mass"]["values"] is None
    assert result["input"]["public_events"] == pre.public_event_history.to_records()


def test_sample_only_never_initializes_a_backend(monkeypatch, capsys):
    plan = {"type": "sample", "selected": [], "request_limit": 0}
    monkeypatch.setattr(study, "load_sample", lambda: (plan, {}))
    load_agent = Mock(side_effect=AssertionError("sample-only reached the language backend"))
    monkeypatch.setattr(study, "load_agent", load_agent)
    study.main(["--sample-only"])
    assert json.loads(capsys.readouterr().out) == plan
    load_agent.assert_not_called()
