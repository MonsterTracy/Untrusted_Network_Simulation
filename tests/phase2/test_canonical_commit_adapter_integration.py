"""Real env/recorder transaction tests; scripted backends, no live LLM."""

from copy import deepcopy
from dataclasses import replace

import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("openai")
pytest.importorskip("torch")

from tests.canonical_collection.test_final_capacity import _night
from tests.canonical_collection.test_runtime_game_evidence import _runtime, ROLES
from tests.phase2.test_decision_opportunity import Mapper, q_matrix
from tests.phase2.test_language import ControlledActor, ControlledPerceiver
from werewolf.phase2_actions import Action, InformationRequestV1
from werewolf.phase2_canonical_commit import (
    Phase2CanonicalCommitError, commit_phase2_verified_speech,
)
from werewolf.phase2_decision_opportunity import build_phase2_decision_opportunity_from_pre
from werewolf.phase2_intervention import assess_phase2_checkpoint_closure
from werewolf.phase2_language import (
    Phase2SpeechSemanticV1, public_language_context_from_pre,
)
from werewolf.phase2_treatment import build_phase2_treatment
from werewolf.phase2_verified_speech import verified_phase2_speech


def _prepare(action):
    _, _, env, _, _, recorder, backend = _runtime()
    env.reset(roles=ROLES)
    # Synthetic fixture: player2 speaks first, player1 is the later teammate.
    env._rng.seed(6)
    _night(env, 0)
    assert env.current_act_idx == 1 and env.phase == "speech"
    recorder.start(env, roles=ROLES)
    recorder.before_agent_act(env, step_idx=0, acting_player_id=2,
        delivered_observation=env.get_observation(), speech_kind="speech")
    prefix = recorder._pending.prefix
    opp = build_phase2_decision_opportunity_from_pre(
        prefix, frozenset(("player1", "player2")), "player3", q_matrix(), Mapper(),
        q_source_digest="b" * 64)
    arm = build_phase2_treatment(opp, action, assignment_source="paired_branch",
                                 assignment_probability=1.0, randomization_key="fixture")
    closure = assess_phase2_checkpoint_closure(opp, env=env, recorder=recorder)
    public = public_language_context_from_pre(prefix, opp.legal_context)
    k = arm.plan.redirect_target
    if action is Action.PUSH:
        text = "本轮我明确投player3。"
        semantic = Phase2SpeechSemanticV1("player2", "speech", ("player3",), (),
                                           ("player3",), ())
        parser_text = "player2 | vote_intent | player3"
    elif action is Action.REDIRECT:
        text = f"本轮不以player3为主要放逐目标，转而投{k}。"
        semantic = Phase2SpeechSemanticV1("player2", "speech", (k,), ("player3",),
                                           (k,), ())
        parser_text = f"player2 | oppose | player3\nplayer2 | vote_intent | {k}"
    else:
        text = "player3，你当前主要怀疑谁？依据哪些公开信息？"
        semantic = Phase2SpeechSemanticV1("player2", "speech", (), (), (),
                                           (InformationRequestV1("player3", "player3"),))
        parser_text = "NONE"
    verified = verified_phase2_speech(
        arm.plan, opp.legal_context, public,
        actor=ControlledActor([text]), perceiver=ControlledPerceiver([semantic]))
    assert verified.success
    backend.perception_response = parser_text
    return env, recorder, opp, arm, verified, closure


@pytest.mark.parametrize("action", list(Action))
def test_one_exact_canonical_event_and_no_actual_vote_mutation(action):
    env, recorder, opp, arm, verified, closure = _prepare(action)
    before_votes = deepcopy(env.vote_target)
    before_events = len(env.public_events)
    _, link = commit_phase2_verified_speech(
        env=env, recorder=recorder, opportunity=opp, treatment=arm,
        verified=verified, closure=closure)
    events = env.public_events[before_events:]
    assert len([e for e in events if e["event_type"] == "public_speech"]) == 1
    assert events[0]["raw_text"] == verified.public_text
    assert env.vote_target == before_votes
    assert recorder._pending is None
    assert link.canonical_event_id == events[0]["event_id"]
    assert link.treatment_id == arm.treatment_id
    assert link.checkpoint_closure_digest == closure.digest()
    assert env.speech_annotations[-1].event_id == events[0]["event_id"]
    with pytest.raises((ValueError, Phase2CanonicalCommitError)):
        commit_phase2_verified_speech(
            env=env, recorder=recorder, opportunity=opp, treatment=arm,
            verified=verified, closure=closure)
    assert len([e for e in env.public_events if e["event_type"] == "public_speech"]) == 1


def test_invalid_audit_and_pre_drift_publish_no_speech():
    env, recorder, opp, arm, verified, closure = _prepare(Action.PUSH)
    before = deepcopy(env.public_events)
    public = public_language_context_from_pre(recorder._pending.prefix, opp.legal_context)
    wrong = Phase2SpeechSemanticV1("player2", "speech", ("player4",), (),
                                   ("player4",), ())
    failed = verified_phase2_speech(
        arm.plan, opp.legal_context, public,
        actor=ControlledActor(["本轮投player4。", "仍投player4。"]),
        perceiver=ControlledPerceiver([wrong, wrong]))
    assert not failed.success
    with pytest.raises(Phase2CanonicalCommitError):
        commit_phase2_verified_speech(
            env=env, recorder=recorder, opportunity=opp, treatment=arm,
            verified=failed, closure=closure)
    assert env.public_events == before
    bad = replace(verified, language_audit=replace(
        verified.language_audit, generated_text_digest="0" * 64))
    with pytest.raises(Phase2CanonicalCommitError):
        commit_phase2_verified_speech(
            env=env, recorder=recorder, opportunity=opp, treatment=arm,
            verified=bad, closure=closure)
    assert env.public_events == before
    env.alive[6] = 0  # Private live-state drift without a new public event.
    with pytest.raises(Phase2CanonicalCommitError, match="drift"):
        commit_phase2_verified_speech(
            env=env, recorder=recorder, opportunity=opp, treatment=arm,
            verified=verified, closure=closure)
    assert not any(e["event_type"] == "public_speech" for e in env.public_events)
