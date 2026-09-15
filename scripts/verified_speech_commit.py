"""Phase 3B orchestration at an already collected PRE; no planner imports."""
from scripts.speech_realization import PublicRealizationContext, realize_parse_verify
from werewolf.agents.prompt_template_v0 import build_public_claim_catalog
from werewolf.canonical_collection.public_history import freeze_public_event_history
from werewolf.speech.verified_commit import bind_verified_speech


def realize_and_commit(payload, *, env, actor, recorder):
    """Realize once, perceive once, verify then transactionally commit.

    before_agent_act must already have collected the real PRE/handoff. This
    function never creates fake belief observations or changes their population.
    On semantic failure that existing pending slot remains unchanged.
    """
    pending = recorder._pending
    if (recorder._env is not env or pending is None or pending.handoff is None
            or pending.prefix is None or pending.raw_action is not None
            or env.phase not in ('speech', 'speech_pk')):
        raise ValueError("constrained realization requires an existing pending speech PRE")
    history_digest = freeze_public_event_history(env.public_events).digest
    speaker = f"player{env.current_act_idx + 1}"
    if (pending.actor_id != speaker or pending.event_count_before != len(env.public_events)
            or pending.prefix.public_event_history.digest != history_digest):
        raise ValueError("pending PRE does not match current opportunity")
    context = PublicRealizationContext(speaker, env.day,
        'discussion' if env.phase == 'speech' else 'pk_discussion',
        build_public_claim_catalog(env.get_observation()))
    realized = realize_parse_verify(payload, context, actor=actor, perceiver=env.speech_perceiver,
        actor_context=recorder.call_audit.action_context(acting_player_id=env.current_act_idx + 1,
            boundary_id=pending.handoff.boundary_id, is_public_speech=True),
        perception_context=recorder.call_audit.speech_perception_context(
            event_id=f"event-{len(env.public_events):06d}",
            boundary_id=pending.handoff.boundary_id, speaker_id=env.current_act_idx + 1))
    envelope = bind_verified_speech(speech=realized.speech, expected=realized.expected,
        perception=realized.perception, day=context.day, phase=env.phase,
        public_history_digest=history_digest, perceiver=env.speech_perceiver)
    return recorder.commit_verified_speech(env, envelope)
