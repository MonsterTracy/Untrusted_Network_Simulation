"""Validator capacity and the separately proved current-runtime length theorem."""

import pytest

from werewolf.canonical_collection import construct_authoritative_pre_prefix, freeze_public_event_history
from werewolf.structured_history import plan_structured_history
from werewolf.tom.final_capacity import derived_capacity, validate_final_pre


def sparse_pre(day):
    events = []
    def event(kind, **values):
        events.append(dict(event_id=f"e{len(events)}", event_index=len(events), event_type=kind, **values))
    event("phase_change", day=0, phase="night")
    for current in range(1, day + 1):
        event("death_announcement", dead_players=[])
        event("phase_change", day=current, phase="discussion")
        if current < day:
            event("phase_change", day=current, phase="vote")
            event("phase_change", day=current, phase="night")
    event("turn_start", speaker="player1")
    return construct_authoritative_pre_prefix(game_id="sparse", boundary_id="b", step_index=len(events)-1,
        report_trigger_id="r", current_speaker="player1", alive_observer_ids=("player1",),
        public_event_history=freeze_public_event_history(events), v1_annotations=(),
        belief_observation_ids_by_observer={"player1": "unloaded-observation"})


def test_validator_day_capacity_boundary():
    capacity = derived_capacity(1024)
    assert capacity["derived_day_capacity"] == 256
    assert plan_structured_history(sparse_pre(256)).token_count == 1024
    assert plan_structured_history(sparse_pre(257)).token_count == 1028
    validate_final_pre(sparse_pre(256), capacity)
    with pytest.raises(ValueError, match="sequence capacity"):
        validate_final_pre(sparse_pre(257), capacity)
    assert "publication" not in repr(capacity)
    assert "runtime" not in capacity["derivation_version"]


@pytest.mark.parametrize("length", [4, 5, 7, 8, 1024, 1028])
def test_capacity_is_integer_derivation_only(length):
    capacity = derived_capacity(length)
    assert capacity["derived_day_capacity"] == length // 4
    validate_final_pre(sparse_pre(length // 4), capacity)
    with pytest.raises(ValueError):
        validate_final_pre(sparse_pre(length // 4 + 1), capacity)


def _runtime():
    from werewolf.envs.werewolf_text_env_v0 import WerewolfTextEnvV0
    from werewolf.speech.speech_perceiver import SpeechParseAuditResult
    class NoActionParser:
        def parse_with_audit(self, **kwargs):
            return SpeechParseAuditResult(normalized_actions=[], raw_response="NONE", parse_status="ok",
                error_type=None, error_message=None, generation_attempts=({"generation_attempt": 1,
                    "status": "ok", "raw_response": "NONE", "error_type": None, "error_message": None},))
    env = WerewolfTextEnvV0(log_save_path=None, random_seed=0, speech_perceiver=NoActionParser())
    env.reset(roles=["Werewolf", "Werewolf", "Seer", "Witch", "Villager", "Villager", "Villager"])
    return env


def _runtime_prefix(env):
    alive = tuple(f"player{i+1}" for i, value in enumerate(env.alive) if value)
    return construct_authoritative_pre_prefix(game_id="runtime-witness", boundary_id=f"b{len(env.public_events)}",
        step_index=len(env.public_events)-1, report_trigger_id="r", current_speaker=f"player{env.current_act_idx+1}",
        alive_observer_ids=alive, public_event_history=freeze_public_event_history(env.public_events),
        v1_annotations=env.speech_annotations, belief_observation_ids_by_observer={p: f"o-{p}" for p in alive})


def _step(env, action):
    _, _, done, _ = env.step(action)
    assert not done, "constructive witness must remain nonterminal"


def _night(env, kill, poison=False):
    while env.phase == "skill_wolf":
        _step(env, ("kill", kill))
    if env.phase == "skill_seer":
        _step(env, env.get_observation()["valid_action"][0])
    if env.phase == "skill_witch":
        _step(env, ("witch_poison", 2) if poison else ("witch_pass", 0))
    assert env.phase == "speech"


def _day(env, exile=0):
    while env.phase == "speech":
        _step(env, ("speech", ""))
    while env.phase == "vote":
        target = exile if env.current_act_idx + 1 != exile else 0
        _step(env, ("vote", target))
    assert env.phase == "skill_wolf"  # no PK, no terminal


def test_runtime_day_one_and_global_day_two_minima():
    env = _runtime()
    _night(env, 0)
    assert plan_structured_history(_runtime_prefix(env)).token_count == 4
    env = _runtime()
    _night(env, 5, poison=True)
    assert sum(env.alive) == 5
    _day(env)
    _night(env, 0)
    assert env.day == 2 and sum(env.alive) == 5
    assert plan_structured_history(_runtime_prefix(env)).token_count == 27


def test_runtime_tight_constructive_witness_and_boundary():
    env = _runtime()
    _night(env, 5, poison=True)  # remove one nonwolf and one wolf on Night0
    _day(env, exile=6)
    assert sum(env.alive) == 4
    _night(env, 7)  # Day2: one wolf, Seer, Witch; poison was already spent
    assert env.day == 2 and sum(env.alive) == 3
    assert not env.is_done()[1]
    assert plan_structured_history(_runtime_prefix(env)).token_count == 29
    counts = {}
    for day in range(2, 70):
        assert env.day == day and sum(env.alive) == 3
        counts[day] = []
        while env.phase == "speech":
            counts[day].append(plan_structured_history(_runtime_prefix(env)).token_count)
            _step(env, ("speech", ""))
        while env.phase == "vote":
            _step(env, ("vote", 0))
        if day < 69:
            _night(env, 0)
    assert all(counts[d][0] == 15*d-1 for d in range(3, 70))
    assert all(counts[d+1][0]-counts[d][0] == 15 for d in range(2, 69))
    assert counts[68] == [1019, 1021, 1023]
    assert counts[69][0] == 1034
    assert min(counts[69]) > 1024
    assert derived_capacity(1024)["derived_day_capacity"] == 256  # theorem is not control
