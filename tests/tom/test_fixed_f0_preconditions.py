"""Phase-2I opt-in boundary regression, scripted local partial PRE only."""
from unittest.mock import Mock
import pytest
from scripts.speech_planning import PlanningAction as A, SpeechPlan, generate_candidates
from scripts.speech_realization import expected_action
from scripts.verified_speech_commit import PreCommitAudit, capture_precommit_binding, realize_and_commit
from tests.tom.test_precommit_audit import setup
from tests.tom.test_verified_speech_commit import actor_for, response_for, snapshot


def test_opt_in_rejects_shape_valid_but_ineligible_plan_before_actor(tmp_path):
    env, rec, backend, _, _, owner = setup(tmp_path)
    try:
        pre = rec.public_pre_for_handoff(rec._pending.handoff)
        candidates = generate_candidates(pre)
        plan = next(SpeechPlan(A.OPPOSE, p) for p in pre.alive_observer_ids
                    if p != pre.current_speaker and SpeechPlan(A.OPPOSE, p) not in candidates)
        payload = plan.public_payload()
        expected = expected_action(payload, pre.current_speaker)  # Shape is VALID.
        actor, _ = actor_for(expected)
        backend.perception_response = response_for(expected)
        binding = capture_precommit_binding(pre, attempt_id=rec.claim.attempt_id,
            handoff_digest=rec._pending.handoff.handoff_digest, ledger_id='controlled-precommit',
            assignment_ref=owner.assignment_ref, owner_epoch='CONTROLLED_EPOCH_NOT_A_REAL_PERMISSION',
            payload=payload)
        native = Mock(wraps=rec.commit_verified_speech)
        rec.commit_verified_speech = native
        callback = Mock(side_effect=ValueError('stop after language if reached'))
        before = snapshot(env, rec)
        audit = PreCommitAudit(binding, callback)
        with pytest.raises(ValueError, match='not a legal candidate'):
            realize_and_commit(payload, env=env, actor=actor, recorder=rec, precommit_audit=audit)
        with pytest.raises(ValueError, match='consumed'):
            realize_and_commit(payload, env=env, actor=actor, recorder=rec, precommit_audit=audit)
        assert actor.backend.calls == [], 'ineligible audited plan reached Actor before rejection'
        callback.assert_not_called()
        native.assert_not_called()
        assert snapshot(env, rec) == before
    finally:
        owner.close()


@pytest.mark.parametrize('phase', ['speech', 'speech_pk'])
@pytest.mark.parametrize('action', [A.SELF_DEFEND, A.NO_COMMITMENT])
def test_fixed_plan_reuses_frozen_PRE_records_intent_then_one_native(phase, action, tmp_path, monkeypatch):
    import torch
    from scripts import counterfactual_tom as cf
    env, rec, backend, actor, payload, owner = setup(tmp_path, phase, action)
    try:
        reports, prefixes = tuple(rec.observations), tuple(rec.prefixes)
        history_before = len(env.public_events)
        original_generate = actor._generate_public_speech
        def generate(*args, **kwargs):
            assert [r['kind'] for r in owner.store.records] == ['OPPORTUNITY', 'ASSIGNMENT', 'EXECUTION']
            assert owner.store.records[1]['payload']['assignment_source'] == 'INJECTED_FIXED_NOT_RANDOMIZED'
            assert owner.store.inspect().assignments[0].commit_status == 'NO_ATTEMPT'
            return original_generate(*args, **kwargs)
        monkeypatch.setattr(actor, '_generate_public_speech', generate)
        original_native = rec.commit_verified_speech
        def native(*args, **kwargs):
            assert owner.store.inspect().assignments[0].commit_status == 'AWAITING_EXTERNAL_READBACK'
            assert owner.store.records[-1]['payload']['status'] == 'ATTEMPT_RESERVED'
            return original_native(*args, **kwargs)
        commit = Mock(side_effect=native)
        monkeypatch.setattr(rec, 'commit_verified_speech', commit)
        monkeypatch.setattr(rec, 'before_agent_act', Mock(side_effect=AssertionError('PRE/reports must not be collected again')))
        monkeypatch.setattr(torch.nn.Module, '__call__', Mock(side_effect=AssertionError('ToM is NOT_USED')))
        monkeypatch.setattr(cf.CounterfactualToMConsumer, '__init__', Mock(side_effect=AssertionError('B2 is NOT_USED')))
        audit = PreCommitAudit(owner.binding, owner.observe)
        result = realize_and_commit(payload, env=env, actor=actor, recorder=rec, precommit_audit=audit)
        assert isinstance(result, tuple) and len(result) == 4
        assert commit.call_count == 1 and len(actor.backend.calls) == owner.calls == 1
        assert tuple(rec.observations) == reports and tuple(rec.prefixes) == prefixes
        rec.before_agent_act.assert_not_called()
        assert rec._pending is None and len(rec.submitted_gameplay_actions) == 1
        native_action = rec.submitted_gameplay_actions[0]
        assert native_action.boundary_id == prefixes[-1].boundary_id
        assert native_action.speaker_pre_belief_handoff.prefix_digest == prefixes[-1].prefix_digest
        assert len(native_action.resulting_public_event_ids) == 1
        speech = env.public_events[history_before]
        assert speech['event_id'] == native_action.resulting_public_event_ids[0]
        assert speech['raw_text'] == native_action.action_payload.to_value()
        assert [a.to_record() for a in env.speech_annotations[-1].actions] == [
            expected_action(payload, prefixes[-1].current_speaker).to_record()]
        with pytest.raises(ValueError):
            realize_and_commit(payload, env=env, actor=actor, recorder=rec, precommit_audit=audit)
        assert commit.call_count == 1 and len(actor.backend.calls) == 1
    finally:
        owner.close()


@pytest.mark.parametrize('phase', ['speech', 'speech_pk'])
def test_other_legal_fixed_plan_does_not_replace_assignment(phase, tmp_path):
    env, rec, backend, actor, payload, owner = setup(tmp_path, phase, A.NO_COMMITMENT)
    try:
        alternative = SpeechPlan(A.SELF_DEFEND).public_payload()
        assert alternative in [p.public_payload() for p in generate_candidates(rec._pending.prefix)]
        native = Mock(wraps=rec.commit_verified_speech)
        rec.commit_verified_speech = native
        with pytest.raises(ValueError, match='binding mismatch'):
            realize_and_commit(alternative, env=env, actor=actor, recorder=rec,
                precommit_audit=PreCommitAudit(owner.binding, owner.observe))
        assert actor.backend.calls == [] and owner.calls == 0
        native.assert_not_called()
        assert payload == SpeechPlan(A.NO_COMMITMENT).public_payload()
    finally:
        owner.close()


@pytest.mark.parametrize('phase', ['speech', 'speech_pk'])
def test_complete_multi_action_response_is_not_single_plan_compliance(phase, tmp_path):
    from scripts.speech_realization import SemanticVerificationError
    env, rec, backend, actor, payload, owner = setup(tmp_path, phase, A.SELF_DEFEND)
    try:
        speaker = rec._pending.prefix.current_speaker
        target = next(p for p in rec._pending.prefix.alive_observer_ids if p != speaker)
        expected = expected_action(payload, speaker)
        backend.perception_response = response_for(expected) + '\n' + f'{speaker} | point_as_werewolf | {target}'
        native = Mock(wraps=rec.commit_verified_speech)
        rec.commit_verified_speech = native
        before = snapshot(env, rec)
        with pytest.raises(SemanticVerificationError) as failure:
            realize_and_commit(payload, env=env, actor=actor, recorder=rec,
                precommit_audit=PreCommitAudit(owner.binding, owner.observe))
        assert len(failure.value.perception.normalized_actions) == 2
        assert len(actor.backend.calls) == 1 and owner.calls == 0
        native.assert_not_called()
        assert snapshot(env, rec) == before
        assert owner.store.inspect().assignments[0].commit_status == 'NO_ATTEMPT'
    finally:
        owner.close()


@pytest.mark.parametrize('bad_result', [None, (), {'unconfirmed': True}])
def test_unconfirmed_native_return_never_looks_like_unassigned_hook(bad_result, tmp_path):
    env, rec, _, actor, payload, owner = setup(tmp_path)
    native = Mock(return_value=bad_result)
    rec.commit_verified_speech = native
    audit = PreCommitAudit(owner.binding, owner.observe)
    try:
        with pytest.raises(ValueError, match='native.*unconfirmed'):
            realize_and_commit(payload, env=env, actor=actor, recorder=rec, precommit_audit=audit)
        assert native.call_count == 1
        assert owner.store.inspect().assignments[0].commit_status == 'AWAITING_EXTERNAL_READBACK'
        calls = len(actor.backend.calls)
        with pytest.raises(ValueError, match='consumed'):
            realize_and_commit(payload, env=env, actor=actor, recorder=rec, precommit_audit=audit)
        assert native.call_count == 1 and len(actor.backend.calls) == calls
        assert owner.store.inspect().assignments[0].commit_status == 'AWAITING_EXTERNAL_READBACK'
    finally:
        owner.close()


def test_one_boundary_is_not_a_complete_game_or_typed_native_failure_archive(tmp_path):
    env, rec, _, actor, payload, owner = setup(tmp_path)
    try:
        realize_and_commit(payload, env=env, actor=actor, recorder=rec,
            precommit_audit=PreCommitAudit(owner.binding, owner.observe))
        assert len(rec.submitted_gameplay_actions) == 1
        with pytest.raises(RuntimeError, match='game is not complete'):
            rec.complete_evidence()
        # Exercise the REAL failure wrapper, unlike the old one-boundary Hook
        # fixture. A failure partial is useful evidence, not an F0 success seal.
        failure = rec.failure_from_exception(RuntimeError('CONTROLLED_FIXED_BOUNDARY_STOP'))
        partial = failure.partial_payload
        assert partial['submitted_actions'] == rec._raw_actions and len(partial['submitted_actions']) == 1
        assert partial['public_events'] == env.public_events
        assert partial['authoritative_pre_prefixes'] == [p.to_record() for p in rec.prefixes]
        assert partial['speech_annotations_v1'] == [a.to_record() for a in env.speech_annotations]
        assert 'submitted_gameplay_actions' not in partial
        assert 'assignment_ref' not in partial and 'attempt_token' not in partial
        assert rec._winner is None  # Never invent a game terminal for the slice.
    finally:
        owner.close()
