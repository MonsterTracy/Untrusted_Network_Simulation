"""Legacy raw-sample constants retained only until Dataset cutover (Commit 7).

Canonical Collection no longer constructs this schema. The Authoritative PRE
Prefix and Belief Observation contracts own production collection semantics.
"""

SAMPLE_SCHEMA_VERSION = "classic7_pre_speech_player_suspicion_v7"
REPORT_TRIGGERS = frozenset({"pre_public_speech", "pre_public_speech_pk"})
SAMPLE_FIELDS = frozenset(
    {
        "schema_version",
        "game_id",
        "step_idx",
        "phase",
        "speaker_id",
        "report_trigger",
        "public_event_schema_version",
        "public_events",
        "public_event_digest",
        "speech_annotation_schema_version",
        "speech_action_ontology_version",
        "speech_annotations",
        "speech_annotation_digest",
        "structured_input_digest",
        "observer_ids",
        "suspected_werewolves",
        "known_werewolves",
        "known_non_werewolves",
        "belief_status",
        "belief_errors",
        "label_cutoff_step_idx",
        "public_action_count",
        "label_prompt_version",
        "label_provenance",
        "agent_backend_ids",
    }
)

__all__ = ["REPORT_TRIGGERS", "SAMPLE_FIELDS", "SAMPLE_SCHEMA_VERSION"]
