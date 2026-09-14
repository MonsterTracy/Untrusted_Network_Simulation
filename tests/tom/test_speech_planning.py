"""Phase 2 public candidate/decision tests; no gameplay or remote LLM calls."""

from dataclasses import FrozenInstanceError, fields, replace
import inspect
from unittest.mock import Mock

import pytest
import torch

from scripts import speech_planning as planning
from scripts.counterfactual_tom import (
    PlanningAction as A, ConsumerProvenance, CounterfactualToMConsumer,
    UnsupportedOpportunityError, tensorize_counterfactual,
)
from scripts.suspicion_objectives import wolf_suspicion_mass
from tests.tom.test_counterfactual_consumer import PublicWitness, P, CAPACITY, sealed
from werewolf.artifact_io import canonical_json_bytes


def uniform():
    b = torch.full((7, 7), 1 / 6, dtype=torch.float64)
    b.fill_diagonal_(0)
    return b


class RecordingConsumer:
    """Explicit test double for the Phase 1 interface; no seal claim."""
    provenance = ConsumerProvenance("test", "test", "test", "test", "test", "test", "test", "test")

    def __init__(self):
        self.calls = []
        self.inputs = []

    def probabilities(self, parent, candidate):
        self.calls.append((parent.prefix_digest, candidate))
        _, tensors = tensorize_counterfactual(parent, candidate, capacity=CAPACITY)
        self.inputs.append(tensors)
        return uniform()


@pytest.mark.parametrize("action", tuple(A))
def test_plan_schema_immutable_comparable_and_phase1_mapping(action):
    target = None if action in (A.SELF_DEFEND, A.NO_COMMITMENT) else P[2]
    plan = planning.SpeechPlan(action, target)
    assert plan == planning.SpeechPlan(action, target)
    assert hash(plan) == hash(planning.SpeechPlan(action, target))
    assert {f.name for f in fields(plan)} == {"action", "target"}
    assert plan.public_payload() == {"action": action.value, "target": target}
    with pytest.raises(FrozenInstanceError):
        plan.target = P[3]
    continuation, _ = tensorize_counterfactual(PublicWitness().pre(), plan.for_speaker(P[0]), capacity=CAPACITY)
    expected = {A.ACCUSE_WOLF: "point_as_werewolf", A.CLEAR: "point_as_non_werewolf",
                A.SUPPORT: "support", A.OPPOSE: "oppose", A.SELF_DEFEND: "point_as_non_werewolf",
                A.NO_COMMITMENT: "no_commitment"}
    assert continuation.tokens[-2].action == expected[action]
    assert continuation.tokens[-2].target == (P[0] if action == A.SELF_DEFEND else target)


@pytest.mark.parametrize("action,target", [("ACCUSE_WOLF", P[2]), ("vote_intent", P[2]),
    (A.CLEAR, None), (A.CLEAR, "player8"), (A.SELF_DEFEND, P[0]), (A.NO_COMMITMENT, P[1])])
def test_invalid_plan(action, target):
    with pytest.raises(ValueError):
        planning.SpeechPlan(action, target)


def test_schema_excludes_private_scores_and_public_candidate_order_is_frozen():
    with pytest.raises(TypeError):
        planning.SpeechPlan(A.CLEAR, P[2], score=.1)
    with pytest.raises(ValueError):
        planning.SpeechPlan(A.CLEAR, P[0]).for_speaker(P[0])
    p = PublicWitness(alive=P[:-1]).pre()
    candidates = planning.generate_candidates(p)
    assert candidates == planning.generate_candidates(p)
    assert planning.ACTION_ORDER == (A.ACCUSE_WOLF, A.CLEAR, A.SUPPORT, A.OPPOSE, A.SELF_DEFEND, A.NO_COMMITMENT)
    expected = tuple(planning.SpeechPlan(a, target)
                     for a in (A.ACCUSE_WOLF, A.CLEAR, A.SUPPORT, A.OPPOSE)
                     for target in (P[1:-1] if a in (A.ACCUSE_WOLF, A.CLEAR) else P[2:-1]))
    assert candidates == expected + (planning.SpeechPlan(A.SELF_DEFEND), planning.SpeechPlan(A.NO_COMMITMENT))
    assert all(c.target not in (P[0], P[6]) for c in candidates)


def test_generation_has_no_role_truth_or_teammate_input():
    p = PublicWitness().pre()
    assert tuple(inspect.signature(planning.generate_candidates).parameters) == ("parent",)
    for field in ("wolf_team", "role_truth", "env", "probabilities"):
        with pytest.raises(TypeError):
            planning.generate_candidates(p, **{field: object()})
    candidates = planning.generate_candidates(p)
    # Every possible living teammate remains accus-able and clear-able.
    for teammate in P[1:]:
        assert planning.SpeechPlan(A.ACCUSE_WOLF, teammate) in candidates
        assert planning.SpeechPlan(A.CLEAR, teammate) in candidates


def test_same_day_normal_speech_is_eligible_in_pk_even_outside_pk_candidates():
    p = PublicWitness("pk_discussion").pre()
    candidates = planning.generate_candidates(p)
    for target in P[1:]:
        assert planning.SpeechPlan(A.SUPPORT, target) in candidates
        assert planning.SpeechPlan(A.OPPOSE, target) in candidates


def test_previous_day_does_not_supply_today_support():
    w = PublicWitness()
    for player in P[:2]:
        if w.events[-1].get("speaker") != player:
            w.event("turn_start", speaker=player)
        w.speech(player, "no_commitment", None, "我暂不表态。")
    w.event("phase_change", day=1, phase="vote")
    w.event("vote_result", votes=[{"voter": p, "target": None} for p in P])
    w.event("exile_result", exiled_players=[])
    w.event("phase_change", day=1, phase="night")
    w.event("death_announcement", dead_players=[])
    w.event("phase_change", day=2, phase="discussion")
    w.event("turn_start", speaker=P[0])
    candidates = planning.generate_candidates(w.pre())
    assert not any(c.action in (A.SUPPORT, A.OPPOSE) for c in candidates)
    assert candidates[-2:] == (planning.SpeechPlan(A.SELF_DEFEND), planning.SpeechPlan(A.NO_COMMITMENT))


def test_evaluator_reuses_phase1_without_private_inputs_or_parent_mutation():
    p = PublicWitness().pre()
    before = canonical_json_bytes(p.to_record())
    candidates = planning.generate_candidates(p)
    consumers = [RecordingConsumer(), RecordingConsumer()]
    results = []
    for consumer, wolves in zip(consumers, ((P[0], P[4]), (P[0], P[2]))):
        rows = planning.evaluate_candidates(p, candidates, consumer, alive_wolves=wolves)
        results.append(rows)
        assert len(consumer.calls) == len(candidates)
        for plan, row, (_, action) in zip(candidates, rows, consumer.calls):
            assert action == plan.for_speaker(P[0])
            assert row.plan == plan
            assert row.suspicion == wolf_suspicion_mass(uniform(), alive_players=P,
                                                        alive_wolves=wolves, self_player=P[0])
            assert row.delta is None and row.current_suspicion is None
            assert row.parent_prefix_digest == p.prefix_digest
        planning.select_minimum_suspicion(candidates, rows)
    assert consumers[0].calls == consumers[1].calls
    for left, right in zip(consumers[0].inputs, consumers[1].inputs):
        assert all(torch.equal(value, right.kwargs()[key]) for key, value in left.kwargs().items())
    assert results[0][0].suspicion.per_alive_wolf_raw_mass != results[1][0].suspicion.per_alive_wolf_raw_mass
    assert canonical_json_bytes(p.to_record()) == before


def test_selector_exact_minimum_tie_and_permuted_evaluation_order():
    p = PublicWitness().pre()
    candidates = planning.generate_candidates(p)
    rows = planning.evaluate_candidates(p, candidates, RecordingConsumer(), alive_wolves=(P[0], P[4]))
    assert planning.select_minimum_suspicion(candidates, rows) == candidates[0]
    def with_score(row, value):
        return replace(row, suspicion=replace(row.suspicion, alive_conditional_team_mass=value))
    changed = tuple(with_score(row, .1 if i in (2, 4) else .5) for i, row in enumerate(rows))
    for _ in range(3):
        assert planning.select_minimum_suspicion(candidates, changed[::-1]) == candidates[2]
    nearly_tied = list(changed)
    nearly_tied[4] = with_score(changed[4], .1 - 1e-12)
    assert planning.select_minimum_suspicion(candidates, tuple(nearly_tied)) == candidates[4]


def test_selector_rejects_incomplete_nonfinite_or_mixed_records():
    p = PublicWitness().pre()
    candidates = planning.generate_candidates(p)
    rows = planning.evaluate_candidates(p, candidates, RecordingConsumer(), alive_wolves=(P[0],))
    for bad_candidates, bad_rows in ((candidates[::-1], rows), (candidates, rows[:-1]),
            (candidates, (rows[0],) * len(rows)),
            (candidates, (replace(rows[0], parent_prefix_digest="wrong"),) + rows[1:]),
            (candidates, (replace(rows[0], suspicion=replace(rows[0].suspicion,
                              alive_conditional_team_mass=float("nan"))),) + rows[1:])):
        with pytest.raises(ValueError):
            planning.select_minimum_suspicion(bad_candidates, bad_rows)


@pytest.mark.parametrize("case", ["missing", "duplicate", "extra", "unknown"])
def test_selector_requires_candidate_evaluation_bijection(case):
    p = PublicWitness().pre()
    candidates = planning.generate_candidates(p)
    rows = planning.evaluate_candidates(p, candidates, RecordingConsumer(), alive_wolves=(P[0],))
    unknown = replace(rows[0], plan=planning.SpeechPlan(A.CLEAR, P[0]))
    assert unknown.plan not in candidates
    bad_rows = {
        "missing": rows[:-1],
        "duplicate": rows[:-1] + (rows[0],),
        "extra": rows + (unknown,),
        "unknown": rows[:-1] + (unknown,),
    }[case]
    with pytest.raises(ValueError, match="complete candidate evaluations|evaluation coverage mismatch"):
        planning.select_minimum_suspicion(candidates, bad_rows)


def test_selector_uses_conditional_score_not_raw_mass():
    p = PublicWitness().pre()
    candidates = planning.generate_candidates(p)
    rows = planning.evaluate_candidates(p, candidates, RecordingConsumer(), alive_wolves=(P[0],))
    changed = tuple(replace(row, suspicion=replace(row.suspicion,
        alive_conditional_team_mass=.1 if i == 1 else .5,
        raw_team_mass=.001 if i == 0 else .1)) for i, row in enumerate(rows))
    assert planning.select_minimum_suspicion(candidates, changed) == candidates[1]


def test_evaluator_propagates_failure_without_partial_results_or_fallback():
    p = PublicWitness().pre()
    candidates = planning.generate_candidates(p)
    before = canonical_json_bytes(p.to_record())
    class FailingConsumer(RecordingConsumer):
        def probabilities(self, parent, candidate):
            if self.calls:
                raise ValueError("explicit inference failure")
            return super().probabilities(parent, candidate)
    consumer = FailingConsumer()
    with pytest.raises(ValueError, match="explicit inference failure"):
        planning.evaluate_candidates(p, candidates, consumer, alive_wolves=(P[0],))
    assert len(consumer.calls) == 1
    assert canonical_json_bytes(p.to_record()) == before


@pytest.mark.parametrize("phase", ["discussion", "pk_discussion"])
@pytest.mark.parametrize("with_baseline", [False, True])
def test_final_speaker_generation_is_separate_from_evaluability(monkeypatch, phase, with_baseline):
    from werewolf.tom.final_evaluation import SealedFinalPredictor
    w = PublicWitness(phase)
    w.speech(P[0], "no_commitment", None, "我暂不表态。")
    w.event("turn_start", speaker=P[1])
    p = w.pre()
    candidates = planning.generate_candidates(p)
    assert candidates[-2:] == (planning.SpeechPlan(A.SELF_DEFEND), planning.SpeechPlan(A.NO_COMMITMENT))
    consumer = RecordingConsumer()
    probability_call = Mock(wraps=consumer.probabilities)
    monkeypatch.setattr(consumer, "probabilities", probability_call)
    predictor = Mock(spec=SealedFinalPredictor)
    evaluation_record = Mock(wraps=planning.CandidateEvaluation)
    selector = Mock(wraps=planning.select_minimum_suspicion)
    monkeypatch.setattr(planning, "CandidateEvaluation", evaluation_record)
    monkeypatch.setattr(planning, "select_minimum_suspicion", selector)
    not_returned = object()
    result = not_returned
    with pytest.raises(UnsupportedOpportunityError):
        result = planning.evaluate_candidates(
            p, candidates, consumer, alive_wolves=(P[1],),
            current_predictor=predictor if with_baseline else None)
    assert consumer.calls == []
    probability_call.assert_not_called()
    predictor.log_probabilities.assert_not_called()
    evaluation_record.assert_not_called()
    selector.assert_not_called()
    assert result is not_returned


def test_invalid_population_and_candidate_set_fail_before_calls():
    p = PublicWitness().pre()
    candidates = planning.generate_candidates(p)
    consumer = RecordingConsumer()
    for wolves in ((), (P[1],), (P[0], P[0]), P):
        with pytest.raises(ValueError):
            planning.evaluate_candidates(p, candidates, consumer, alive_wolves=wolves)
    with pytest.raises(ValueError):
        planning.evaluate_candidates(p, candidates[:-1], consumer, alive_wolves=(P[0],))
    assert consumer.calls == []


def test_future_selector_can_consume_unchanged_candidates_without_tom():
    p = PublicWitness().pre()
    candidates = planning.generate_candidates(p)
    # Test-only interface witness, not a shipped No-ToM policy.
    def external_decision(plans):
        return plans[-1]
    assert external_decision(candidates) == planning.SpeechPlan(A.NO_COMMITMENT)
    assert planning.generate_candidates(p) == candidates


@pytest.mark.parametrize("condition", ["implicit", "explicit_day_phase"])
def test_real_phase1_consumer_baseline_delta_and_artifact_noncontamination(sealed, condition):
    from werewolf.tom.final_evaluation import SealedFinalPredictor
    before = {path: path.read_bytes() for root in (sealed.path, sealed.runs_path)
              for path in root.rglob("*") if path.is_file()}
    p = PublicWitness().pre()
    candidates = planning.generate_candidates(p)
    consumer = CounterfactualToMConsumer(sealed, condition)
    predictor = SealedFinalPredictor(sealed, condition)
    baseline = wolf_suspicion_mass(predictor.log_probabilities(p).exp(), alive_players=P,
                                   alive_wolves=(P[0], P[4]), self_player=P[0])
    rows = planning.evaluate_candidates(p, candidates, consumer,
                                        alive_wolves=(P[0], P[4]), current_predictor=predictor)
    for row in rows:
        direct = wolf_suspicion_mass(consumer.probabilities(p, row.plan.for_speaker(P[0])),
                                    alive_players=P, alive_wolves=(P[0], P[4]), self_player=P[0])
        assert row.suspicion == direct
        assert row.current_suspicion == baseline
        assert row.delta == baseline.alive_conditional_team_mass - direct.alive_conditional_team_mass
    assert planning.select_minimum_suspicion(candidates, rows) in candidates
    other = "explicit_day_phase" if condition == "implicit" else "implicit"
    with pytest.raises(ValueError, match="provenance differs"):
        planning.evaluate_candidates(p, candidates, consumer, alive_wolves=(P[0],),
                                    current_predictor=SealedFinalPredictor(sealed, other))
    assert {path: path.read_bytes() for root in (sealed.path, sealed.runs_path)
            for path in root.rglob("*") if path.is_file()} == before
