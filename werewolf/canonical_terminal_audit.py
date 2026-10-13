"""Independent native terminal audit, without changing Canonical qualification.

Runs no backend. Rejections do not rewrite, delete or reclassify source games or
Ledger records. Complete-Collection reports remain in custody: winners are Test
labels. This audit is an additional release check, not a new replay identity.
"""

from copy import deepcopy
from pathlib import Path

from werewolf.artifact_io import (
    ArtifactValidationError, canonical_json_bytes, publish_artifact, sha256_bytes, verify_artifact,
)
from werewolf.canonical_collection.game_bundle import VerifiedCanonicalGameBundle
from werewolf.canonical_collection.public_history import PLAYER_IDS, freeze_public_event_history
from werewolf.development_publication import _role_assignment, _validate_verified_collection, _load_canonical_json
from werewolf.runtime_config import normalize_runtime_config

AUDIT_VERSION = "classic7_native_terminal_audit_v1"
AUDIT_TYPE = "canonical_collection_terminal_audit"


def _audit_sources_digest():
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "werewolf").rglob("*.py")) + [root / "run_random.py"]
    return sha256_bytes(canonical_json_bytes({p.relative_to(root).as_posix(): sha256_bytes(p.read_bytes()) for p in paths}))


def audit_terminal_bundle(bundle, *, runtime_config):
    """Replay recorded actions against the real engine and require its terminal."""
    if not isinstance(bundle, VerifiedCanonicalGameBundle):
        raise TypeError("terminal audit requires a verified Canonical Bundle")
    from werewolf.envs.werewolf_text_env_v0 import WerewolfTextEnvV0
    from werewolf.canonical_collection.production_runtime import _ReplaySpeechPerceiver
    config = normalize_runtime_config(runtime_config)
    private = bundle.private_replay_evidence
    if private.initial_runtime_state.to_value() != {"day": 0, "phase": "night"}:
        raise ArtifactValidationError("terminal audit initial runtime state mismatch")
    if canonical_json_bytes(private.runtime_configuration.to_value()) != canonical_json_bytes(config):
        raise ArtifactValidationError("terminal audit runtime configuration mismatch")
    roles = _role_assignment(private.role_assignment)
    replay = private.replay_inputs.to_value()
    raw_actions = replay.get("submitted_actions")
    if not isinstance(raw_actions, list) or len(raw_actions) != len(bundle.submitted_gameplay_actions):
        raise ArtifactValidationError("terminal audit raw/submitted action coverage mismatch")
    declared_winner = replay.get("winner")
    if declared_winner not in {"Werewolf", "Villager"}:
        raise ArtifactValidationError("terminal audit requires a recognized recorded winner")
    env_config = deepcopy(config["env_config"])
    env_config["log_save_path"] = None
    try:
        env = WerewolfTextEnvV0(**env_config, speech_perceiver=_ReplaySpeechPerceiver(), random_seed=private.seed)
        env.reset(roles=[role for _, role in roles])
        done, info = False, {}
        for index, (raw, action) in enumerate(zip(raw_actions, bundle.submitted_gameplay_actions, strict=True)):
            if done or env.phase == "end_game":
                raise ArtifactValidationError("terminal audit rejects action after native terminal")
            if (not isinstance(raw, list) or len(raw) != 2 or action.actor_id != PLAYER_IDS[env.current_act_idx]):
                raise ArtifactValidationError("terminal audit action/actor/phase identity mismatch")
            speech = env.phase in {"speech", "speech_pk"}
            if speech:
                if raw[0] != env.phase or not isinstance(raw[1], str):
                    raise ArtifactValidationError("terminal audit speech phase/payload mismatch")
            elif canonical_json_bytes(raw) not in {
                    canonical_json_bytes(list(a)) for a in env.get_observation()["valid_action"]}:
                raise ArtifactValidationError("terminal audit illegal native action")
            # Speech PRE indexes public events; other actions index submissions.
            # Preserve that existing distinction rather than renumbering PRE.
            expected_step = env.public_events[-1]["event_index"] if speech else index
            if action.step_index != expected_step:
                raise ArtifactValidationError("terminal audit action/PRE step identity mismatch")
            expected_type = "public_speech" if speech else raw[0]
            payload = raw[1] if expected_type == "public_speech" else raw
            if action.action_type != expected_type or canonical_json_bytes(action.action_payload.to_value()) != canonical_json_bytes(payload):
                raise ArtifactValidationError("terminal audit submitted action payload mismatch")
            before = len(env.public_events)
            _, _, done, info = env.step(tuple(raw))
            appended = env.public_events[before:]
            resulting = tuple(event["event_id"] for event in appended
                              if expected_type != "public_speech" or event["event_type"] == "public_speech")
            if resulting != action.resulting_public_event_ids:
                raise ArtifactValidationError("terminal audit action/public outcome mismatch")
        if not done or env.phase != "end_game":
            raise ArtifactValidationError("terminal audit requires a complete native terminal")
        wolf_count = sum(role == "Werewolf" and env.alive[i] == 1 for i, role in enumerate(env.roles))
        non_wolf_count = sum(role != "Werewolf" and env.alive[i] == 1 for i, role in enumerate(env.roles))
        winner = "Villager" if wolf_count == 0 else "Werewolf" if wolf_count >= non_wolf_count else None
        if winner != declared_winner or info.get("Werewolf") != (1 if winner == "Werewolf" else -1):
            raise ArtifactValidationError("terminal audit native/recorded winner mismatch")
        history = freeze_public_event_history(env.public_events)
        if (history.to_records() != bundle.public_event_stream.to_records()
                or history.digest != bundle.public_event_stream.digest or history.digest != private.expected_public_event_digest):
            raise ArtifactValidationError("terminal audit authoritative public trajectory mismatch")
    except ArtifactValidationError:
        raise
    except (TypeError, ValueError, KeyError, IndexError, AssertionError) as error:
        raise ArtifactValidationError(f"invalid native terminal replay: {error}") from error
    return {"schema_version": AUDIT_VERSION, "game_id": bundle.game_id,
            "bundle_digest": bundle.manifest_digest, "private_replay_evidence_digest": private.evidence_digest,
            "public_event_digest": history.digest, "action_digests": [a.action_digest for a in bundle.submitted_gameplay_actions],
            "terminal_step": len(raw_actions), "native_phase": env.phase, "winner": winner,
            "terminal_alive_player_ids": [PLAYER_IDS[i] for i in range(7) if env.alive[i] == 1]}


def publish_collection_terminal_audit(collection, destination, *, runtime_config):
    """All successes or no report; failure never emits a selected subset."""
    _validate_verified_collection(collection)
    config = normalize_runtime_config(runtime_config)
    rows = [audit_terminal_bundle(bundle, runtime_config=config) for bundle in collection.games]
    return publish_artifact(destination, manifest_fields={
        "artifact_type": AUDIT_TYPE, "schema_version": AUDIT_VERSION,
        "collection_plan_digest": collection.plan.plan_digest,
        "collection_identity_digest": collection.collection_identity_digest,
        "game_count": len(rows), "runtime_config_digest": sha256_bytes(canonical_json_bytes(config)),
        "audit_implementation_digest": _audit_sources_digest(),
        "audited_bundle_set_digest": sha256_bytes(canonical_json_bytes([[b.game_id, b.manifest_digest] for b in collection.games])),
    }, files={"restricted/terminal_audit.json": canonical_json_bytes({"games": rows})})


def verify_collection_terminal_audit(path, collection, *, runtime_config):
    """Re-audit native outcomes and compare the complete immutable report."""
    _validate_verified_collection(collection)
    artifact = verify_artifact(path, expected_artifact_type=AUDIT_TYPE, expected_schema_version=AUDIT_VERSION)
    config = normalize_runtime_config(runtime_config)
    expected = {"artifact_type": AUDIT_TYPE, "schema_version": AUDIT_VERSION,
        "collection_plan_digest": collection.plan.plan_digest, "collection_identity_digest": collection.collection_identity_digest,
        "game_count": len(collection.games), "runtime_config_digest": sha256_bytes(canonical_json_bytes(config)),
        "audit_implementation_digest": _audit_sources_digest(),
        "audited_bundle_set_digest": sha256_bytes(canonical_json_bytes([[b.game_id, b.manifest_digest] for b in collection.games]))}
    if (canonical_json_bytes({k: v for k, v in artifact.manifest.items() if k not in {"file_table", "manifest_digest"}}) != canonical_json_bytes(expected)
            or set(artifact.manifest["file_table"]) != {"restricted/terminal_audit.json"}):
        raise ArtifactValidationError("terminal audit collection/source binding mismatch")
    rows = [audit_terminal_bundle(bundle, runtime_config=config) for bundle in collection.games]
    if _load_canonical_json(artifact.path / "restricted/terminal_audit.json") != {"games": rows}:
        raise ArtifactValidationError("terminal audit semantic outcome mismatch")
    return artifact
