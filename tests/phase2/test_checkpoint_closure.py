"""The present live PRE is identifiable, but no causal clone is certified."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from tests.phase2.test_decision_opportunity import opportunity
from werewolf.phase2_actions import Action
from werewolf.phase2_intervention import (
    CHECKPOINT_BLOCKERS, CheckpointUnavailable,
    assess_phase2_checkpoint_closure, capture_intervention_checkpoint,
    clone_phase2_checkpoint, plan_paired_branches,
    restore_intervention_checkpoint, run_intervention_branch,
    validate_treatment_checkpoint_binding,
)
from werewolf.phase2_treatment import build_phase2_treatment


def live_slot(opp):
    context = opp.legal_context
    prefix = SimpleNamespace(
        game_id=context.game_id, boundary_id=context.boundary_id,
        prefix_digest=context.prefix_digest, current_speaker=context.acting_wolf,
        public_event_history=SimpleNamespace(digest=context.public_history_digest))
    env = SimpleNamespace(phase=context.phase,
                          current_act_idx=int(context.acting_wolf[-1]) - 1)
    handoff = SimpleNamespace(boundary_id=context.boundary_id,
        prefix_digest=context.prefix_digest, observer_id=context.acting_wolf)
    recorder = SimpleNamespace(_env=env, _pending=SimpleNamespace(
        prefix=prefix, handoff=handoff, raw_action=None))
    return env, recorder


def test_closure_assessment_is_deterministic_but_never_executable():
    opp = opportunity()
    env, recorder = live_slot(opp)
    closure = assess_phase2_checkpoint_closure(opp, env=env, recorder=recorder)
    assert closure.digest() == assess_phase2_checkpoint_closure(
        opp, env=env, recorder=recorder).digest()
    assert closure.identity == opp.identity[:5]
    assert closure.blockers == CHECKPOINT_BLOCKERS
    assert closure.executable is False
    assert closure.to_record()["schema_version"] == "phase2_checkpoint_closure_v1"
    env.private_state = {"night_target": "player7"}
    assert assess_phase2_checkpoint_closure(
        opp, env=env, recorder=recorder).local_env_digest != closure.local_env_digest
    arm = build_phase2_treatment(opp, Action.PUSH,
                                 assignment_source="paired_branch",
                                 assignment_probability=1, randomization_key="paired")
    validate_treatment_checkpoint_binding(opp, arm, closure)
    with pytest.raises(CheckpointUnavailable, match="mismatch"):
        validate_treatment_checkpoint_binding(opp, arm,
            replace(closure, boundary_id="another-boundary"))
    with pytest.raises(CheckpointUnavailable, match="mismatch"):
        validate_treatment_checkpoint_binding(opp, arm,
            replace(closure, prefix_digest="another-prefix"))
    with pytest.raises(CheckpointUnavailable, match="mismatch"):
        validate_treatment_checkpoint_binding(opp, arm,
            replace(closure, acting_wolf="player5"))


def test_live_pre_drift_is_rejected_and_all_checkpoint_entry_points_fail_closed():
    opp = opportunity()
    env, recorder = live_slot(opp)
    env.phase = "vote"
    with pytest.raises(CheckpointUnavailable, match="PRE differs"):
        assess_phase2_checkpoint_closure(opp, env=env, recorder=recorder)
    env.phase = opp.legal_context.phase
    recorder._pending.handoff.observer_id = "player5"
    with pytest.raises(CheckpointUnavailable, match="PRE differs"):
        assess_phase2_checkpoint_closure(opp, env=env, recorder=recorder)
    for operation in (capture_intervention_checkpoint,
                      clone_phase2_checkpoint, restore_intervention_checkpoint,
                      run_intervention_branch):
        with pytest.raises(CheckpointUnavailable):
            operation()


def test_paired_branch_identity_is_deterministic_and_arm_specific():
    opp = opportunity()
    arms = tuple(build_phase2_treatment(
        opp, action, assignment_source="paired_branch",
        assignment_probability=1, randomization_key="paired")
        for action in (Action.PUSH, Action.REDIRECT, Action.PROBE))
    first = plan_paired_branches("checkpoint-id", arms, downstream_seed=123)
    second = plan_paired_branches("checkpoint-id", arms, downstream_seed=123)
    assert first == second
    assert len({arm.branch_identity for arm in first}) == len(arms)
    assert all(not arm.executable for arm in first)
