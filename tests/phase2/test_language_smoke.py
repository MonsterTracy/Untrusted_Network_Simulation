"""Offline smoke-runner tests; no real model, game, or server is used."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from scripts import run_phase2_language_smoke as smoke
from werewolf.artifact_io import verify_artifact
from werewolf.phase2_actions import Action, ActionContextV1, InformationRequestV1
from werewolf.phase2_language import Phase2SpeechSemanticV1, PublicLanguageContextV1


ALIVE = tuple(f"player{i}" for i in range(1, 8))
WOLVES = frozenset(("player1", "player5"))


def context(n, phase, *, probe=False):
    if phase == "speech":
        competition = queue = ALIVE
    elif probe:
        competition = queue = ("player1", "player2", "player5")
    else:
        competition = queue = ("player1", "player2", "player3")
    return ActionContextV1(f"g{n:04d}", f"b{n:04d}", f"prefix{n}", f"history{n}",
                           phase, "player1", ALIVE, WOLVES, competition, queue)


def eligible_cases():
    cases = []
    for n in range(45):
        c = context(n, "speech")
        for j in c.legal_targets:
            cases.extend((smoke.Case(c, j, Action.PUSH, object()),
                          smoke.Case(c, j, Action.REDIRECT, object())))
            if smoke.probe_continuation(c, j) is not None:
                cases.append(smoke.Case(c, j, Action.PROBE, object()))
    for n in range(100, 108):
        c = context(n, "speech_pk", probe=True)
        cases.append(smoke.Case(c, "player2", Action.PROBE, object()))
        cases.append(smoke.Case(c, "player2", Action.PUSH, object()))
    for n in range(200, 250):
        c = context(n, "speech_pk")
        for j in c.legal_targets:
            cases.extend((smoke.Case(c, j, Action.PUSH, object()),
                          smoke.Case(c, j, Action.REDIRECT, object())))
    return tuple(cases)


def test_deterministic_58_case_layout_and_pk_probe_exhaustion():
    eligible = eligible_cases()
    first = smoke.select_cases(eligible)
    second = smoke.select_cases(tuple(reversed(eligible)))
    assert [c.identity for c in first] == [c.identity for c in second]
    assert smoke.selection_digest(first) == smoke.selection_digest(second)
    assert len(first) == len({c.pre_identity for c in first}) == 58
    assert [(p, a.value, sum(c.context.phase == p and c.action is a for c in first))
            for p, a, _ in smoke.LAYOUT] == [(p, a.value, n) for p, a, n in smoke.LAYOUT]
    pk_probe = [c for c in first if c.context.phase == "speech_pk" and c.action is Action.PROBE]
    assert {c.pre_identity for c in pk_probe} == {c.pre_identity for c in eligible if c.action is Action.PROBE and c.context.phase == "speech_pk"}
    assert all(smoke.probe_continuation(c.context, c.candidate_j) is not None for c in pk_probe)


def test_unique_pre_shortage_fails_without_resampling():
    eligible = [c for c in eligible_cases() if c.context.phase == "speech_pk" and c.action is Action.PROBE]
    with pytest.raises(smoke.SmokeStudyError, match="insufficient unique PRE"):
        smoke.select_cases(tuple(eligible))


def test_pk_probe_population_change_is_not_silently_subsampled():
    eligible = list(eligible_cases())
    c = context(999, "speech_pk", probe=True)
    eligible.append(smoke.Case(c, "player2", Action.PROBE, object()))
    with pytest.raises(smoke.SmokeStudyError, match="PK Probe population changed"):
        smoke.select_cases(tuple(eligible))


def test_collect_eligible_reads_no_future_or_private_report(monkeypatch):
    c = context(1, "speech")
    pre = SimpleNamespace(public_temporal_state=SimpleNamespace(phase=SimpleNamespace(value="discussion")),
                          current_speaker="player1", alive_observer_ids=ALIVE)

    class Game:
        game_id = "g0001"
        authoritative_pre_prefixes = (pre,)

        @property
        def public_event_stream(self):
            raise AssertionError("future public stream accessed")

        @property
        def belief_observations(self):
            raise AssertionError("private reports accessed")

    pub = SimpleNamespace(manifest_digest=smoke.offline.PUBLICATION_DIGEST,
                          public_view=SimpleNamespace(games=(Game(),)))
    sidecar = SimpleNamespace(sidecar_digest=smoke.offline.ROLE_SIDECAR_DIGEST,
                              games=(SimpleNamespace(game_id="g0001",
                                  role_assignment=tuple((p, "Werewolf" if p in WOLVES else "Villager") for p in ALIVE)),))
    monkeypatch.setattr(smoke.offline, "EXPECTED_GAMES", 1)
    monkeypatch.setattr(smoke.offline, "EXPECTED_PRE", 1)
    monkeypatch.setattr(smoke.offline, "EXPECTED_CANDIDATES", 5)
    monkeypatch.setattr(smoke, "context_from_pre", lambda actual, wolves: c)
    cases = smoke.collect_eligible(pub, sidecar)
    assert len(cases) >= 5
    assert all(case.pre is pre for case in cases)


@dataclass
class Mapper:
    artifact_digest: str = smoke.MAPPER_DIGEST
    model_digest: str = "m" * 64

    def infer(self, q, *, candidate_j, **kwargs):
        return .9 if candidate_j == "player3" else .2


def test_redirect_uses_frozen_mapper_and_canonical_tie_break():
    c = context(200, "speech")
    case = smoke.Case(c, "player2", Action.REDIRECT)
    called = []
    class TiedMapper:
        def infer(self, q, **kwargs):
            called.append(kwargs["candidate_j"])
            return .5
    plan = smoke.make_plan(case, {case.pre_identity: "sealed-q"}, TiedMapper())
    assert called == ["player3", "player4", "player6", "player7"]
    assert plan.redirect_target == "player3"


def test_oof_reader_requests_seven_rows_only_for_selected_redirect(monkeypatch):
    selected = smoke.select_cases(eligible_cases())
    redirect = tuple(c for c in selected if c.action is Action.REDIRECT)
    seen = {}
    monkeypatch.setattr(smoke.offline, "fold_assignment", lambda manifest: {c.context.game_id: 0 for c in selected})
    def fake_load(publication, evaluation, opportunities):
        seen["opportunities"] = opportunities
        return {(o.game_id, o.boundary_id, p): tuple(0 if j == i else 1/6 for j in range(7))
                for o in opportunities for i, p in enumerate(smoke.PLAYER_IDS)}
    monkeypatch.setattr(smoke.offline, "load_oof_probabilities", fake_load)
    publication = SimpleNamespace(public_view=SimpleNamespace(fold_manifest=object()))
    matrices = smoke.selected_oof_q(publication, selected, "/sealed")
    assert len(seen["opportunities"]) == len(redirect) == 20
    assert all(o.alive == smoke.PLAYER_IDS and o.wolves == frozenset() for o in seen["opportunities"])
    assert len(matrices) == 20
    assert all(len(q) == 7 and all(len(row) == 7 for row in q) for q in matrices.values())


class Actor:
    def __init__(self):
        self.calls = []

    def realize(self, plan, public, *, failure_reason=None):
        self.calls.append((plan, public, failure_reason))
        if plan.action is Action.PROBE:
            return f"PROBE:{plan.candidate_j}"
        if plan.action is Action.REDIRECT:
            return f"REDIRECT:{plan.candidate_j}:{plan.redirect_target}"
        return f"PUSH:{plan.candidate_j}"


class Perceiver:
    def __init__(self):
        self.calls = []

    def perceive(self, text, public):
        self.calls.append((text, public))
        parts = text.split(":")
        if parts[0] == "PROBE":
            return Phase2SpeechSemanticV1(public.speaker, public.phase, (), (), (),
                (InformationRequestV1(parts[1], parts[1]),))
        if parts[0] == "REDIRECT":
            return Phase2SpeechSemanticV1(public.speaker, public.phase, (parts[2],),
                                           (parts[1],), (parts[2],), ())
        return Phase2SpeechSemanticV1(public.speaker, public.phase, (parts[1],), (), (parts[1],), ())


def test_mock_end_to_end_without_plan_leakage(monkeypatch):
    cases = smoke.select_cases(eligible_cases())
    def public_from_pre(pre, c):
        return PublicLanguageContextV1(c.phase, c.acting_wolf, c.alive,
                                       c.competition, c.public_history_digest)
    monkeypatch.setattr(smoke, "public_language_context_from_pre", public_from_pre)
    actor, perceiver = Actor(), Perceiver()
    q = {c.pre_identity: [[0.0] * 7] * 7 for c in cases if c.action is Action.REDIRECT}
    rows, metrics = smoke.execute(cases, q_by_pre=q, mapper=Mapper(),
                                  actor=actor, perceiver=perceiver)
    assert len(rows) == len(actor.calls) == len(perceiver.calls) == 58
    assert metrics["gate"]["passed"]
    assert metrics["overall"]["first_pass_valid"] == metrics["overall"]["final_valid"] == 58
    assert all(len(call) == 2 and isinstance(call[1], PublicLanguageContextV1) for call in perceiver.calls)
    assert metrics["invalid_reasons_all_attempts"] == {}


def test_mock_end_to_end_has_exactly_one_repair(monkeypatch):
    cases = smoke.select_cases(eligible_cases())
    monkeypatch.setattr(smoke, "public_language_context_from_pre",
        lambda pre, c: PublicLanguageContextV1(c.phase, c.acting_wolf, c.alive,
                                               c.competition, c.public_history_digest))
    class BadPerceiver:
        def __init__(self):
            self.calls = 0
        def perceive(self, text, public):
            self.calls += 1
            return Phase2SpeechSemanticV1(public.speaker, public.phase, (), (), (), ())
    actor, perceiver = Actor(), BadPerceiver()
    q = {c.pre_identity: [[0.0] * 7] * 7 for c in cases if c.action is Action.REDIRECT}
    rows, metrics = smoke.execute(cases, q_by_pre=q, mapper=Mapper(),
                                  actor=actor, perceiver=perceiver)
    assert len(actor.calls) == perceiver.calls == 116
    assert all(len(row["audit"]["attempts"]) == 2 for row in rows)
    assert metrics["overall"]["repair_attempted"] == 58
    assert metrics["overall"]["repair_success"] == 0
    assert not metrics["gate"]["passed"]


def row(action, *, first=True, final=True, reason="PERCEPTION_JSON_INVALID", commitment=()):
    attempts = [{"language_execution_valid": first, "language_invalid_reason": None if first else reason,
                 "perceived_semantics": {"action_identity": action, "commitment_targets": list(commitment)} if first else None}]
    if not first:
        attempts.append({"language_execution_valid": final,
                         "language_invalid_reason": None if final else reason,
                         "perceived_semantics": {"action_identity": action, "commitment_targets": list(commitment)} if final else None})
    return {"action": action, "phase": "speech", "plan": {"commitment_target": None if action == "PROBE" else "player2"},
            "audit": {"attempts": attempts, "language_execution_valid": final,
                      "language_invalid_reason": None if final else reason}}


def test_metric_reasons_repair_and_gate_failure():
    rows = [row("PUSH", commitment=("player2",)) for _ in range(20)]
    rows += [row("REDIRECT", first=False, final=True, commitment=("player2",)) for _ in range(20)]
    rows += [row("PROBE", first=False, final=False) for _ in range(18)]
    result = smoke.aggregate(rows, plan_isolation_verified=True)
    assert result["overall"]["case_count"] == 58
    assert result["overall"]["repair_attempted"] == 38
    assert result["overall"]["repair_success"] == 20
    assert result["overall"]["final_valid"] == 40
    assert result["invalid_reasons_all_attempts"] == {"PERCEPTION_JSON_INVALID": 56}
    assert result["invalid_reasons_final"] == {"PERCEPTION_JSON_INVALID": 18}
    assert not result["gate"]["passed"]
    assert result["gate"]["checks"]["no_schema_wide_parse_failure"]


def test_schema_wide_failure_and_one_action_collapse():
    failed = smoke.aggregate([row("PUSH", first=False, final=False) for _ in range(58)], plan_isolation_verified=True)
    assert not failed["gate"]["checks"]["no_schema_wide_parse_failure"]
    collapsed = smoke.aggregate([row("PUSH", commitment=("player2",)) for _ in range(58)], plan_isolation_verified=True)
    assert not collapsed["gate"]["checks"]["no_perception_one_action_collapse"]
    leaked = smoke.aggregate([row("PUSH", commitment=("player2",))], plan_isolation_verified=False)
    assert not leaked["gate"]["checks"]["no_requested_plan_leakage"]


def test_systematic_extra_commitment_fails_gate():
    rows = [row("PUSH", commitment=("player2",)) for _ in range(14)]
    rows += [row("PUSH", commitment=("player2", "player3")) for _ in range(6)]
    rows += [row("REDIRECT", commitment=("player2",)) for _ in range(20)]
    rows += [row("PROBE") for _ in range(18)]
    result = smoke.aggregate(rows, plan_isolation_verified=True)
    assert result["final_extra_commitment_by_action"]["PUSH"] == 6
    assert not result["gate"]["checks"]["no_systematic_illegal_extra_commitment"]


def test_artifact_canonical_serialization_and_no_overwrite(tmp_path, monkeypatch):
    cases = smoke.select_cases(eligible_cases())
    monkeypatch.setattr(smoke, "source_provenance", lambda: {"commit": "a" * 40})
    rows = [{**case.to_record(), "plan": {}, "audit": {"attempts": []}} for case in cases]
    metrics = {"overall": {"case_count": 58}}
    model = {"served_model_name": "qwen35-9b", "hf_revision": "b" * 40}
    first = tmp_path / "one"
    second = tmp_path / "two"
    d1 = smoke.publish_smoke(first, cases=cases, rows=rows, metrics=metrics, model=model, mapper=Mapper())
    d2 = smoke.publish_smoke(second, cases=cases, rows=rows, metrics=metrics, model=model, mapper=Mapper())
    assert d1 == d2
    assert verify_artifact(first, expected_artifact_type="phase2_language_execution_smoke",
                           expected_schema_version=smoke.VERSION).manifest_digest == d1
    with pytest.raises(smoke.SmokeStudyError, match="already exists"):
        smoke.publish_smoke(first, cases=cases, rows=rows, metrics=metrics, model=model, mapper=Mapper())
