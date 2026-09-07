# Public speech and temporal events

Canonical Collection owns V1 Speech Perception and immutable
speech_annotations[].actions. The only statuses are ok, no_action and error.
no_action is a successful empty semantic observation; exhausted error excludes
the game. Dataset and Publication never reparse speech.

An Authoritative PRE Prefix contains all earlier public events and their frozen
annotations, ending in turn_start(current_speaker). That speaker's subsequent
public_speech boundary and derived speech_action tokens are excluded.

The initial public event is phase_change(day=0, phase=night). Each phase_change
belongs to the new state it enters; subsequent tokens inherit the latest
already-observed state. Speech boundary and action tokens share one state.
There is no future-state backfill or private scheduler phase exposure.

The Structured Token Planner is pure and population-blind. It emits ordered
semantic descriptors, causal state, token count and digest; it never emits
numeric model tensors. Publication and Dataset use this same planner.
