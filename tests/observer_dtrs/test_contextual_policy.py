"""Synthetic PRE -> genuine D2 fit -> contextual risk -> Actor/parser, CPU only.

Scripted language backends check transport/protocol, not real LLM competence.
No real publication, sidecar, held-out labels, frozen checkpoint or env mutation.
"""
from dataclasses import replace
from fractions import Fraction as F
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.speech_planning import PlanningAction as A, generate_candidates
from scripts.speech_realization import realize_parse_verify, SemanticVerificationError
from tests.observer_dtrs.conftest import prefix_for
from tests.observer_dtrs.test_estimation import fit, row
from tests.tom.test_counterfactual_consumer import PublicWitness, P
from tests.tom.test_speech_realization import ActorBackend, parser
from werewolf.agents.gpt_agent import GPTAgent
from werewolf.artifact_io import canonical_json_bytes
from werewolf.observer_dtrs.context import contexts_for_pre
from werewolf.observer_dtrs.contextual_policy import (
    POLICY_VERSION, contextual_loss, propose_speech,
)
from werewolf.observer_dtrs.d2_protocol import frozen_protocol
from werewolf.observer_dtrs.decision import Region, TiePolicy, decide, thresholds
from werewolf.observer_dtrs.estimation import FittedModel
from werewolf.observer_dtrs.probabilities import ProbabilityMatrix, ProbabilityRow


TEAM = P[:2]  # Explicitly synthetic legal wolf-team knowledge, not a role sidecar.
POLICY = TiePolicy.TERMINAL_NEG_FIRST


def fitted(positive_middle=5):
    protocol = frozen_protocol()
    rows = [row(f"synthetic-train-{j}", int(j >= 20 or 10 <= j < 10 + positive_middle),
                (.1, .4, .8)[j // 10], sample=f"train-{j}-{phase}-{nW}", phase=phase, nW=nW)
            for j in range(30) for phase in ("discussion", "pk_discussion") for nW in (1, 2)]
    return protocol, fit(rows, protocol, model="OBS-EQ1", bins=3)


@pytest.fixture
def estimator():
    return fitted()


def q_row(pre, r=.4, observer=P[2]):
    values = [0.] * 7
    living_team = [p for p in TEAM if p in pre.alive_observer_ids]
    for p in living_team:
        values[P.index(p)] = r / len(living_team)
    values[P.index(next(p for p in P if p not in (*TEAM, observer)))] = 1 - r
    return ProbabilityRow(pre.game_id, pre.boundary_id, pre.prefix_digest, observer, tuple(values))


def proposal(pre, estimator, *, r=.4, observer=P[2], **changes):
    protocol, model = estimator
    args = dict(known_wolf_ids=TEAM, observer_id=observer, prediction=q_row(pre, r, observer),
                model=model, protocol=protocol, tie_policy=POLICY)
    args.update(changes)
    return propose_speech(pre, **args)


@pytest.mark.parametrize("phase,dead,c_fn,beta,alpha", [
    ("discussion", (), 1, F(1, 4), F(3, 4)),
    ("discussion", (P[1],), 2, F(1, 7), F(3, 5)),
    ("pk_discussion", (), 2, F(1, 4), F(3, 7)),
    ("pk_discussion", (P[1],), 4, F(1, 7), F(3, 11)),
])
@pytest.mark.parametrize("policy", list(TiePolicy))
def test_contextual_risks_and_thresholds_match_independent_oracle(phase, dead, c_fn, beta, alpha, policy):
    ctx = contexts_for_pre(prefix_for(phase=phase, dead=dead), TEAM)[0]
    loss = contextual_loss(ctx, tie_policy=policy)
    k = F(2, 5) if phase == "pk_discussion" else F(1, 4)
    assert (loss.c_fp, loss.c_fn, loss.b0, loss.b1) == (1, c_fn, k, k * c_fn)
    t = thresholds(loss)
    assert (t.beta, t.alpha, t.gamma, t.regime) == (beta, alpha, F(1, 1 + c_fn), "NONDEGENERATE")
    for p in {F(j, 100) for j in range(101)} | {beta, alpha}:
        risks = {Region.POS: 1-p, Region.BND: k*(1-p)+k*c_fn*p, Region.NEG: c_fn*p}
        d = decide(p, loss)
        assert d.region == next(a for a in policy.order if risks[a] == min(risks.values()))
        assert dict(d.risks.items()) == risks
    assert decide(beta, loss).minimizing_actions == (Region.BND, Region.NEG)
    assert decide(alpha, loss).minimizing_actions == (Region.POS, Region.BND)


def test_loss_depends_only_on_declared_context_not_identity_or_clock():
    ctx = contexts_for_pre(prefix_for(), TEAM)[0]
    changed = replace(ctx, game_id="unrelated-public-id", boundary_id="other-boundary",
                      prefix_digest="e"*64, day=9, step_index=99, observer_id=P[3])
    assert contextual_loss(ctx, tie_policy=POLICY) == contextual_loss(changed, tie_policy=POLICY)
    with pytest.raises(TypeError):
        contextual_loss({"phase": "discussion"}, tie_policy=POLICY)
    with pytest.raises(ValueError, match="TiePolicy"):
        contextual_loss(ctx, tie_policy="implicit-default")


@pytest.mark.parametrize("r,region,action", [
    (.1, Region.NEG, A.NO_COMMITMENT), (.4, Region.BND, A.OPPOSE), (.8, Region.POS, A.SELF_DEFEND),
])
def test_real_estimator_drives_region_then_legal_intent_without_mutation(estimator, r, region, action):
    pre = PublicWitness().pre()
    before = canonical_json_bytes(pre.to_record())
    model_before = estimator[1].payload
    q = q_row(pre, r)
    result = proposal(pre, estimator, r=r, prediction=q)
    assert result.decision.region is region and result.plan.action is action
    assert float(result.decision.probability) == estimator[1].predict(result.features)
    assert float(result.decision.probability) != pytest.approx(result.features.r)
    assert result.plan in generate_candidates(pre)
    assert result.public_payload() == {"action": action.value, "target": P[2] if action is A.OPPOSE else None}
    assert result.audit_record()["schema_version"] == POLICY_VERSION
    assert result.audit_record()["estimator"]["model_digest"] == estimator[1].digest
    assert result.audit_record()["loss"] == result.loss.to_record()
    assert result.audit_record()["decision"] == result.decision.to_record()
    assert result.audit_record() == proposal(pre, estimator, r=r).audit_record()
    assert canonical_json_bytes(pre.to_record()) == before and estimator[1].payload == model_before
    assert result.features.r == r and q == q_row(pre, r)
    payload = result.public_payload()
    payload["action"] = "ALTERED_RETURN_VALUE"
    assert result.public_payload()["action"] == action.value


def test_same_event_estimate_different_legal_phase_changes_loss_and_region(estimator):
    normal = proposal(PublicWitness().pre(), estimator)
    pk = proposal(PublicWitness("pk_discussion").pre(), estimator)
    assert normal.decision.probability == pk.decision.probability == F(1, 2)
    assert normal.model_digest == pk.model_digest
    assert normal.decision.region is Region.BND and pk.decision.region is Region.POS
    assert normal.loss.c_fn == 1 and pk.loss.c_fn == 2
    assert normal.decision.thresholds.alpha > pk.decision.thresholds.alpha


def test_matrix_and_row_use_same_observer_estimate(estimator):
    pre = PublicWitness().pre()
    rows = tuple(q_row(pre, observer=p).values if p not in TEAM else
                 tuple(0. if q == p else 1/6 for q in P) for p in P)
    matrix = ProbabilityMatrix(pre.game_id, pre.boundary_id, pre.prefix_digest, rows)
    assert proposal(pre, estimator, prediction=matrix) == proposal(pre, estimator)


def test_bnd_without_prior_claim_keeps_region_and_declines_opposition(estimator):
    pre = prefix_for()  # Legal first speech, no observer has spoken yet.
    result = proposal(pre, estimator)
    assert result.decision.region is Region.BND
    assert result.plan.action is A.NO_COMMITMENT and result.mapping_reason == "BND_NO_CURRENT_DAY_CLAIM"
    assert result.public_context.claims == ()


def test_pk_non_candidate_can_be_discussed_but_no_reply_or_vote_is_assumed():
    pre = PublicWitness("pk_discussion").pre()  # Only P1/P2/P3 are PK speakers.
    result = proposal(pre, fitted(2), observer=P[3])
    assert result.decision.region is Region.BND and result.plan.action is A.OPPOSE
    assert result.plan.target == P[3] and result.plan in generate_candidates(pre)
    assert result.public_context.phase == "pk_discussion"
    assert all(c.event in ("speech", "speech_pk") for c in result.public_context.claims)
    assert set(result.public_payload()) == {"action", "target"}


def test_lone_wolf_and_last_speaker_are_supported_without_future_turn_assumption(estimator):
    pre = PublicWitness(alive=tuple(p for p in P if p != P[1])).pre()
    result = proposal(pre, estimator)
    assert result.context.wolf_count == 1 and result.loss.c_fn == 2
    assert result.plan in generate_candidates(pre)


@pytest.mark.parametrize("field,value", [("game_id", "other-game"), ("boundary_id", "other-pre"),
    ("prefix_digest", "b"*64), ("observer_id", P[4])])
def test_misaligned_q_fails_before_decision(estimator, field, value):
    pre = PublicWitness().pre()
    with pytest.raises(ValueError, match="identity mismatch"):
        proposal(pre, estimator, prediction=replace(q_row(pre), **{field: value}))


@pytest.mark.parametrize("bad", [.5, [0, .2, 0, .8, 0, 0, 0], {"probability": .8}])
def test_raw_probabilities_and_rows_cannot_bypass_estimation(estimator, bad):
    pre = PublicWitness().pre()
    with pytest.raises(TypeError):
        proposal(pre, estimator, prediction=bad)
    with pytest.raises(TypeError):
        proposal(pre, estimator, model=bad)


def test_corrupted_estimator_and_bad_context_fail_closed(estimator):
    pre = PublicWitness().pre()
    record = estimator[1].record
    record["root"]["probability"] = .99
    with pytest.raises(ValueError, match="shrinkage"):
        proposal(pre, estimator, model=FittedModel(canonical_json_bytes(record)))
    with pytest.raises(ValueError, match="living wolf"):
        proposal(pre, estimator, observer_id=TEAM[1])
    with pytest.raises(ValueError, match="digest mismatch"):
        proposal(replace(pre, prefix_digest="0"*64), estimator)
    with pytest.raises(ValueError, match="victory condition"):
        proposal(PublicWitness(alive=P[:4]).pre(), estimator)
    with pytest.raises(ValueError, match="living wolf"):
        proposal(prefix_for(actor=P[2]), estimator, observer=P[3])


def test_current_game_cannot_reuse_its_training_labels(estimator):
    pre = PublicWitness().pre()
    protocol = estimator[0]
    trained_here = fit([row(pre.game_id, 1)], protocol, model="OBS-EQ0", bins=1)
    with pytest.raises(ValueError, match="training partition"):
        proposal(pre, estimator, model=trained_here)


@pytest.mark.parametrize("forbidden", ["role_truth", "roles", "event_label_y", "test_labels", "future_events"])
def test_hidden_or_future_information_not_in_interface(estimator, forbidden):
    with pytest.raises(TypeError, match="unexpected keyword"):
        proposal(PublicWitness().pre(), estimator, **{forbidden: object()})


@pytest.mark.parametrize("phase,r,semantic,obj,text", [
    ("discussion", .1, "no_commitment", None, "我暂时不作明确表态。"),
    ("discussion", .4, "oppose", P[2], "我质疑player3此前发言的依据。"),
    ("discussion", .8, "point_as_non_werewolf", P[0], "我认为player1不是狼人。"),
    ("pk_discussion", .4, "point_as_non_werewolf", P[0], "我认为player1不是狼人。"),
])
def test_cpu_closed_loop_reuses_real_actor_parser_and_public_only_transport(estimator, phase, r, semantic, obj, text):
    pre = PublicWitness(phase).pre()
    result = proposal(pre, estimator, r=r)
    backend = ActorBackend(text)
    actor = GPTAgent(backend=backend, model_name="synthetic-language-actor")
    perceiver = parser(f"player1 | {semantic} | {obj or 'NONE'}")
    realized = realize_parse_verify(result.public_payload(), result.public_context, actor=actor, perceiver=perceiver)
    assert realized.speech == text and realized.expected.to_record() == [P[0], semantic, obj]
    assert len(backend.calls) == len(perceiver.backend.calls) == 1
    request = json.dumps(backend.calls, ensure_ascii=False)
    for private in (result.model_digest, result.context.prefix_digest,
                    "known_wolf_ids", "wolf_count", "loss_digest", "probability", POLICY_VERSION):
        assert private not in request
    assert "synthetic-language-actor" not in str(pre.to_record())
    # Genuine language/parser disagreement is rejected; no silent strategy change.
    with pytest.raises(SemanticVerificationError):
        realize_parse_verify(result.public_payload(), result.public_context, actor=actor,
                            perceiver=parser("player1 | point_as_werewolf | player4"))


def test_entry_imports_no_restricted_supervision_and_runs_no_service():
    root = Path(__file__).resolve().parents[2]
    code = "import sys; import werewolf.observer_dtrs.contextual_policy; " + \
        "assert not set(('werewolf.observer_dtrs.labels', 'werewolf.observer_dtrs.dataset_offline', " + \
        "'werewolf.observer_dtrs.estimation_offline', 'werewolf.phase2_gameplay_server')) & set(sys.modules)"
    subprocess.run([sys.executable, "-B", "-c", code], cwd=root, check=True)


def test_synthetic_command_is_real_entrypoint_without_formal_inputs():
    root = Path(__file__).resolve().parents[2]
    run = subprocess.run([sys.executable, "-B", "-m", "scripts.check_observer_dtrs_contextual_policy"],
                         cwd=root, capture_output=True, text=True, check=True)
    output = json.loads(run.stdout)
    assert output["scope"] == "synthetic_only_not_formal_or_gameplay"
    examples = output["examples"]
    assert [r["decision"]["region"] for r in examples] == ["NEG", "BND", "POS"]
    assert [r["speech_intent"]["action"] for r in examples] == ["NO_COMMITMENT", "NO_COMMITMENT", "SELF_DEFEND"]
    assert examples[1]["mapping_reason"] == "BND_NO_CURRENT_DAY_CLAIM"
    assert all(r["scope"] == "research_prototype_only" for r in examples)
