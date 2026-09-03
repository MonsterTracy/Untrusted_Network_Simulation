"""Coarse observer-visible Classic7 public history."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from werewolf.artifact_io import canonical_json_bytes, sha256_bytes


PUBLIC_EVENT_SCHEMA_VERSION = "classic7_public_event_history_v1"
PUBLIC_PHASES = (
    "night",
    "discussion",
    "vote",
    "pk_discussion",
    "pk_vote",
)
PUBLIC_EVENT_TYPES = (
    "phase_change",
    "turn_start",
    "public_speech",
    "vote_result",
    "exile_result",
    "death_announcement",
)
PLAYER_IDS = tuple(f"player{seat}" for seat in range(1, 8))
_PLAYER_ORDER = {player_id: seat for seat, player_id in enumerate(PLAYER_IDS)}

_INPUT_FIELDS = {
    "phase_change": frozenset(
        {"event_id", "event_index", "event_type", "day", "phase"}
    ),
    "turn_start": frozenset(
        {"event_id", "event_index", "event_type", "speaker"}
    ),
    "public_speech": frozenset(
        {"event_id", "event_index", "event_type", "speaker", "raw_text"}
    ),
    "vote_result": frozenset(
        {"event_id", "event_index", "event_type", "votes"}
    ),
    "exile_result": frozenset(
        {"event_id", "event_index", "event_type", "exiled_players"}
    ),
    "death_announcement": frozenset(
        {"event_id", "event_index", "event_type", "dead_players"}
    ),
}


class PublicPhase(str, Enum):
    NIGHT = "night"
    DISCUSSION = "discussion"
    VOTE = "vote"
    PK_DISCUSSION = "pk_discussion"
    PK_VOTE = "pk_vote"


@dataclass(frozen=True)
class PublicTemporalState:
    day: int
    phase: PublicPhase

    def to_record(self) -> dict[str, Any]:
        return {"day": self.day, "phase": self.phase.value}


@dataclass(frozen=True)
class PublicVote:
    voter: str
    target: str | None

    def to_record(self) -> dict[str, Any]:
        return {"voter": self.voter, "target": self.target}


@dataclass(frozen=True)
class FrozenPublicEvent:
    event_id: str
    event_index: int
    event_type: str
    temporal_state: PublicTemporalState
    speaker: str | None = None
    raw_text: str | None = None
    votes: tuple[PublicVote, ...] = ()
    affected_players: tuple[str, ...] = ()

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "event_id": self.event_id,
            "event_index": self.event_index,
            "event_type": self.event_type,
            **self.temporal_state.to_record(),
        }
        if self.event_type in {"turn_start", "public_speech"}:
            record["speaker"] = self.speaker
        if self.event_type == "public_speech":
            record["raw_text"] = self.raw_text
        elif self.event_type == "vote_result":
            record["votes"] = [vote.to_record() for vote in self.votes]
        elif self.event_type == "exile_result":
            record["exiled_players"] = list(self.affected_players)
        elif self.event_type == "death_announcement":
            record["dead_players"] = list(self.affected_players)
        return record


@dataclass(frozen=True)
class PublicEventHistory:
    events: tuple[FrozenPublicEvent, ...]
    digest: str
    current_state: PublicTemporalState
    max_public_day: int

    def to_records(self) -> list[dict[str, Any]]:
        return [event.to_record() for event in self.events]


def _sequence(value: Any, field_name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{field_name} must be a sequence")
    return value


def _player(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or value not in PLAYER_IDS:
        raise ValueError(
            f"{field_name} must be an exact player1...player7 reference"
        )
    return value


def _ordered_players(value: Any, field_name: str) -> tuple[str, ...]:
    players = tuple(
        _player(player, field_name)
        for player in _sequence(value, field_name)
    )
    if len(players) != len(set(players)):
        raise ValueError(f"{field_name} contains a duplicate player")
    if players != tuple(sorted(players, key=_PLAYER_ORDER.__getitem__)):
        raise ValueError(f"{field_name} must use canonical seat order")
    return players


def _nonnegative_integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _phase(value: Any) -> PublicPhase:
    if not isinstance(value, str) or value not in PUBLIC_PHASES:
        raise ValueError(
            "Public Phase must be one of: " + ", ".join(PUBLIC_PHASES)
        )
    return PublicPhase(value)


def _validate_transition(
    previous: PublicTemporalState,
    current: PublicTemporalState,
) -> None:
    day = previous.day
    allowed = {
        PublicPhase.NIGHT: PublicTemporalState(
            day=day + 1,
            phase=PublicPhase.DISCUSSION,
        ),
        PublicPhase.DISCUSSION: PublicTemporalState(
            day=day,
            phase=PublicPhase.VOTE,
        ),
        PublicPhase.PK_DISCUSSION: PublicTemporalState(
            day=day,
            phase=PublicPhase.PK_VOTE,
        ),
    }
    if previous.phase is PublicPhase.VOTE:
        choices = {
            PublicTemporalState(day=day, phase=PublicPhase.PK_DISCUSSION),
            PublicTemporalState(day=day, phase=PublicPhase.NIGHT),
        }
    elif previous.phase is PublicPhase.PK_VOTE:
        choices = {PublicTemporalState(day=day, phase=PublicPhase.NIGHT)}
    elif previous.phase in allowed:
        choices = {allowed[previous.phase]}
    else:
        choices = set()
    if current not in choices:
        raise ValueError(
            "invalid causal Public Phase transition: "
            f"({previous.day}, {previous.phase.value}) -> "
            f"({current.day}, {current.phase.value})"
        )


def _validate_event_identity(
    event: Mapping[str, Any],
    expected_index: int,
    seen_ids: set[str],
) -> tuple[str, str]:
    event_type = event.get("event_type")
    if event_type not in PUBLIC_EVENT_TYPES:
        raise ValueError(f"unsupported public event type: {event_type!r}")
    if set(event) != _INPUT_FIELDS[event_type]:
        raise ValueError(f"{event_type} public event has an invalid field set")
    event_id = event.get("event_id")
    if not isinstance(event_id, str) or not event_id.strip():
        raise ValueError("public event_id must be non-empty text")
    if event_id in seen_ids:
        raise ValueError(f"duplicate public event_id: {event_id}")
    event_index = event.get("event_index")
    if (
        isinstance(event_index, bool)
        or not isinstance(event_index, int)
        or event_index != expected_index
    ):
        raise ValueError(
            f"public event_index must be continuous from zero; "
            f"expected {expected_index}"
        )
    seen_ids.add(event_id)
    return event_id, event_type


def _votes(value: Any) -> tuple[PublicVote, ...]:
    votes: list[PublicVote] = []
    seen_voters: set[str] = set()
    for raw_vote in _sequence(value, "votes"):
        if not isinstance(raw_vote, Mapping) or set(raw_vote) != {
            "voter",
            "target",
        }:
            raise ValueError("each public vote requires voter and target")
        voter = _player(raw_vote.get("voter"), "vote voter")
        target_value = raw_vote.get("target")
        target = (
            None
            if target_value is None
            else _player(target_value, "vote target")
        )
        if voter in seen_voters:
            raise ValueError(f"duplicate public voter: {voter}")
        seen_voters.add(voter)
        votes.append(PublicVote(voter=voter, target=target))
    if tuple(vote.voter for vote in votes) != tuple(
        sorted(seen_voters, key=_PLAYER_ORDER.__getitem__)
    ):
        raise ValueError("public votes must use canonical voter order")
    return tuple(votes)


def _validate_event_phase(event_type: str, state: PublicTemporalState) -> None:
    phase = state.phase
    if event_type in {"turn_start", "public_speech"}:
        if phase not in {PublicPhase.DISCUSSION, PublicPhase.PK_DISCUSSION}:
            raise ValueError(f"{event_type} is not public in phase {phase.value}")
    elif event_type in {"vote_result", "exile_result"}:
        if phase not in {PublicPhase.VOTE, PublicPhase.PK_VOTE}:
            raise ValueError(f"{event_type} is not public in phase {phase.value}")
    elif event_type == "death_announcement" and phase is not PublicPhase.NIGHT:
        raise ValueError("death_announcement must inherit the night phase")


def freeze_public_event_history(events: Any) -> PublicEventHistory:
    """Validate and freeze one complete cumulative public-event sequence."""

    raw_events = _sequence(events, "public events")
    if not raw_events:
        raise ValueError(
            "public history must begin with phase_change(day=0, phase=night)"
        )
    first = raw_events[0]
    if not isinstance(first, Mapping):
        raise TypeError("each public event must be a mapping")
    if first.get("event_type") != "phase_change":
        raise ValueError(
            "ordinary public event cannot receive state from a future phase_change"
        )

    frozen_events: list[FrozenPublicEvent] = []
    state: PublicTemporalState | None = None
    seen_ids: set[str] = set()
    pending_speaker: str | None = None
    state_event_types: set[str] = set()
    for expected_index, raw_event in enumerate(raw_events):
        if not isinstance(raw_event, Mapping):
            raise TypeError("each public event must be a mapping")
        event_id, event_type = _validate_event_identity(
            raw_event,
            expected_index,
            seen_ids,
        )

        if event_type == "phase_change":
            day = _nonnegative_integer(raw_event.get("day"), "public day")
            next_state = PublicTemporalState(
                day=day,
                phase=_phase(raw_event.get("phase")),
            )
            if state is None:
                if next_state != PublicTemporalState(0, PublicPhase.NIGHT):
                    raise ValueError(
                        "initial public temporal state must be "
                        "day=0, phase=night"
                    )
            else:
                _validate_transition(state, next_state)
                if (
                    state.phase is PublicPhase.NIGHT
                    and "death_announcement" not in state_event_types
                ):
                    raise ValueError(
                        "night must publish a death_announcement before "
                        "the next discussion phase"
                    )
            if pending_speaker is not None:
                raise ValueError("turn_start must be followed by its public_speech")
            state = next_state
            state_event_types = set()
            frozen_events.append(
                FrozenPublicEvent(
                    event_id=event_id,
                    event_index=expected_index,
                    event_type=event_type,
                    temporal_state=state,
                )
            )
            continue

        if state is None:
            raise ValueError(
                "ordinary public event cannot receive state from a future phase_change"
            )
        _validate_event_phase(event_type, state)

        speaker: str | None = None
        raw_text: str | None = None
        votes: tuple[PublicVote, ...] = ()
        affected_players: tuple[str, ...] = ()
        if event_type == "turn_start":
            if pending_speaker is not None:
                raise ValueError("turn_start cannot replace a pending speaker")
            speaker = _player(raw_event.get("speaker"), "turn_start speaker")
            pending_speaker = speaker
        elif event_type == "public_speech":
            speaker = _player(raw_event.get("speaker"), "public_speech speaker")
            if pending_speaker != speaker:
                raise ValueError(
                    "public_speech must immediately follow its matching turn_start"
                )
            raw_text_value = raw_event.get("raw_text")
            if not isinstance(raw_text_value, str):
                raise TypeError("public_speech raw_text must be text")
            raw_text = raw_text_value
            pending_speaker = None
        elif pending_speaker is not None:
            raise ValueError("turn_start must be followed by its public_speech")
        elif event_type == "vote_result":
            votes = _votes(raw_event.get("votes"))
        elif event_type == "exile_result":
            affected_players = _ordered_players(
                raw_event.get("exiled_players"),
                "exiled_players",
            )
        else:
            affected_players = _ordered_players(
                raw_event.get("dead_players"),
                "dead_players",
            )

        frozen_events.append(
            FrozenPublicEvent(
                event_id=event_id,
                event_index=expected_index,
                event_type=event_type,
                temporal_state=state,
                speaker=speaker,
                raw_text=raw_text,
                votes=votes,
                affected_players=affected_players,
            )
        )
        state_event_types.add(event_type)

    assert state is not None
    records = [event.to_record() for event in frozen_events]
    return PublicEventHistory(
        events=tuple(frozen_events),
        digest=sha256_bytes(canonical_json_bytes(records)),
        current_state=state,
        max_public_day=max(event.temporal_state.day for event in frozen_events),
    )


def _validate_frozen_event_payload(event: FrozenPublicEvent) -> None:
    if not isinstance(event.temporal_state, PublicTemporalState):
        raise TypeError("public event temporal_state must be PublicTemporalState")
    _nonnegative_integer(event.temporal_state.day, "public day")
    if not isinstance(event.temporal_state.phase, PublicPhase):
        raise ValueError("public event phase must be a frozen PublicPhase")

    if event.event_type == "phase_change":
        if (
            event.speaker is not None
            or event.raw_text is not None
            or event.votes
            or event.affected_players
        ):
            raise ValueError("phase_change has invalid frozen payload")
        return

    _validate_event_phase(event.event_type, event.temporal_state)
    if event.event_type == "turn_start":
        _player(event.speaker, "turn_start speaker")
        if event.raw_text is not None or event.votes or event.affected_players:
            raise ValueError("turn_start has invalid frozen payload")
    elif event.event_type == "public_speech":
        _player(event.speaker, "public_speech speaker")
        if not isinstance(event.raw_text, str):
            raise TypeError("public_speech raw_text must be text")
        if event.votes or event.affected_players:
            raise ValueError("public_speech has invalid frozen payload")
    elif event.event_type == "vote_result":
        if event.speaker is not None or event.raw_text is not None:
            raise ValueError("vote_result has invalid frozen payload")
        if event.affected_players:
            raise ValueError("vote_result has invalid frozen payload")
        if not isinstance(event.votes, tuple) or any(
            not isinstance(vote, PublicVote) for vote in event.votes
        ):
            raise TypeError("vote_result votes must be frozen PublicVote values")
        if _votes([vote.to_record() for vote in event.votes]) != event.votes:
            raise ValueError("vote_result has non-canonical frozen votes")
    else:
        if event.speaker is not None or event.raw_text is not None or event.votes:
            raise ValueError(f"{event.event_type} has invalid frozen payload")
        if not isinstance(event.affected_players, tuple):
            raise TypeError("affected players must be a frozen tuple")
        field_name = (
            "exiled_players"
            if event.event_type == "exile_result"
            else "dead_players"
        )
        if (
            _ordered_players(event.affected_players, field_name)
            != event.affected_players
        ):
            raise ValueError(f"{field_name} is not canonical")


def validate_public_event_history(
    history: PublicEventHistory,
) -> PublicEventHistory:
    """Defensively validate a frozen history without repairing its state."""

    if not isinstance(history, PublicEventHistory):
        raise TypeError("history must be a PublicEventHistory")
    if not isinstance(history.events, tuple) or not history.events:
        raise ValueError(
            "public history must begin with phase_change(day=0, phase=night)"
        )

    state: PublicTemporalState | None = None
    seen_ids: set[str] = set()
    pending_speaker: str | None = None
    state_event_types: set[str] = set()
    for expected_index, event in enumerate(history.events):
        if not isinstance(event, FrozenPublicEvent):
            raise TypeError("public history must contain FrozenPublicEvent values")
        if event.event_type not in PUBLIC_EVENT_TYPES:
            raise ValueError(f"unsupported public event type: {event.event_type!r}")
        if not isinstance(event.event_id, str) or not event.event_id.strip():
            raise ValueError("public event_id must be non-empty text")
        if event.event_id in seen_ids:
            raise ValueError(f"duplicate public event_id: {event.event_id}")
        if event.event_index != expected_index:
            raise ValueError(
                "public event_index must be continuous from zero; "
                f"expected {expected_index}"
            )
        seen_ids.add(event.event_id)
        _validate_frozen_event_payload(event)

        if event.event_type == "phase_change":
            next_state = event.temporal_state
            if state is None:
                if next_state != PublicTemporalState(0, PublicPhase.NIGHT):
                    raise ValueError(
                        "initial public temporal state must be "
                        "day=0, phase=night"
                    )
            else:
                _validate_transition(state, next_state)
                if (
                    state.phase is PublicPhase.NIGHT
                    and "death_announcement" not in state_event_types
                ):
                    raise ValueError(
                        "night must publish a death_announcement before "
                        "the next discussion phase"
                    )
            if pending_speaker is not None:
                raise ValueError("turn_start must be followed by its public_speech")
            state = next_state
            state_event_types = set()
            continue

        if state is None:
            raise ValueError(
                "ordinary public event cannot receive state from a future phase_change"
            )
        if event.temporal_state != state:
            raise ValueError(
                "ordinary public event temporal state is not causally inherited"
            )
        if event.event_type == "turn_start":
            if pending_speaker is not None:
                raise ValueError("turn_start cannot replace a pending speaker")
            pending_speaker = event.speaker
        elif event.event_type == "public_speech":
            if pending_speaker != event.speaker:
                raise ValueError(
                    "public_speech must immediately follow its matching turn_start"
                )
            pending_speaker = None
        elif pending_speaker is not None:
            raise ValueError("turn_start must be followed by its public_speech")
        state_event_types.add(event.event_type)

    assert state is not None
    if history.current_state != state:
        raise ValueError("public event history current_state mismatch")
    derived_max_public_day = max(
        event.temporal_state.day for event in history.events
    )
    if history.max_public_day != derived_max_public_day:
        raise ValueError("public event history max_public_day mismatch")
    expected_digest = sha256_bytes(canonical_json_bytes(history.to_records()))
    if history.digest != expected_digest:
        raise ValueError("public event history digest mismatch")
    return history
