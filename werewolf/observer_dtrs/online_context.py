"""Allowlisted actor observation -> immutable legal Shadow request. No env access."""
from dataclasses import dataclass, field
import time

from werewolf.artifact_io import canonical_json_bytes, sha256_bytes
from werewolf.canonical_collection.pre import AuthoritativePREPrefix, validate_authoritative_pre_prefix
from werewolf.canonical_collection.public_history import PLAYER_IDS
from werewolf.observer_dtrs.context import contexts_for_pre
from werewolf.observer_dtrs.dataset import sample_id

VERSION = 'observer_dtrs_shadow_request_v1'


def digest(value):
    return sha256_bytes(canonical_json_bytes(value))


def seat(value):
    if type(value) is int and 1 <= value <= 7:
        return PLAYER_IDS[value-1]
    if type(value) is str and value in PLAYER_IDS:
        return value
    raise ValueError('invalid absolute Classic7 seat')


@dataclass(frozen=True)
class ShadowRequest:
    prefix: AuthoritativePREPrefix
    known_wolf_ids: tuple[str, ...]
    shadow_run_id: str
    attempt_id: str
    captured_at_ns: int = field(default_factory=time.time_ns, compare=False)

    def __post_init__(self):
        validate_authoritative_pre_prefix(self.prefix)
        if (any(type(v) is not str or not v.strip() for v in (self.shadow_run_id, self.attempt_id))
                or type(self.captured_at_ns) is not int or self.captured_at_ns < 0
                or not self.contexts):
            raise ValueError('Shadow requires an eligible living wolf speech PRE and request identity')

    @property
    def contexts(self):
        return contexts_for_pre(self.prefix, self.known_wolf_ids)

    @property
    def identity(self):
        return {'schema_version': VERSION, 'shadow_run_id': self.shadow_run_id,
                'attempt_id': self.attempt_id, 'game_id': self.prefix.game_id,
                'boundary_id': self.prefix.boundary_id, 'prefix_digest': self.prefix.prefix_digest,
                'step_index': self.prefix.step_index, 'acting_wolf': self.prefix.current_speaker,
                'legal_source': 'delivered_actor_observation_v1',
                'contexts': [c.to_record() for c in self.contexts],
                'sample_ids': [sample_id(c) for c in self.contexts]}

    @property
    def digest(self):
        # Capture time is observational metadata, never an idempotency discriminator.
        return digest(self.identity)


def build_shadow_request(prefix, acting_observation, *, shadow_run_id, attempt_id):
    """Read only actor identity, public state and the one legal wolf-team message.

    The caller must deliver the actor's filtered observation. This adapter cannot
    attest an arbitrary caller's honesty; it never obtains other players' views.
    """
    validate_authoritative_pre_prefix(prefix)
    o = acting_observation
    if not isinstance(o, dict) or o.get('identity') != 'Werewolf':
        raise ValueError('current actor must have a legal Werewolf observation')
    actor = seat(o['current_act_idx'])
    if seat(o['observer_id']) != actor or actor != prefix.current_speaker:
        raise ValueError('actor/observer/PRE identity mismatch')
    public = o['authoritative_public_state']
    alive = tuple(seat(v) for v in public['alive_players'])
    state = prefix.public_temporal_state
    runtime = {'discussion': 'speech', 'pk_discussion': 'speech_pk'}.get(state.phase.value)
    if (runtime is None or public['phase'] != runtime or public['day'] != state.day
            or type(public['day']) is not int or public['day_or_night'] != 'day'
            or o['phase'] != f'{state.day}_day_{runtime}'
            or len(set(alive)) != len(alive) or set(alive) != set(prefix.alive_observer_ids)
            or actor not in alive):
        raise ValueError('public alive/day/phase does not match canonical PRE')
    teams = [log.content.get('wolf_team') for log in o['game_log']
             if log.event == 'werewolf_team_info']
    if len(teams) != 1 or type(teams[0]) is not list:
        raise ValueError('exactly one legal wolf team message is required')
    team = tuple(seat(v) for v in teams[0])
    if len(team) != 2 or len(set(team)) != 2 or actor not in team:
        raise ValueError('invalid legal wolf team')
    return ShadowRequest(prefix, tuple(p for p in PLAYER_IDS if p in team), shadow_run_id, attempt_id)
