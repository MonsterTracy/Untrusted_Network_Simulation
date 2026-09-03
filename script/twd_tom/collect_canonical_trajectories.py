"""Validators for the superseded batch artifacts pending Publication cutover.

This module intentionally has no executable collection entry point. Production
collection is owned exclusively by :mod:`werewolf.canonical_collection`.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from werewolf.models.twd_tom.public_events import (
    PUBLIC_EVENT_SCHEMA_VERSION,
    normalize_public_events,
    public_event_digest,
    public_speech_actions,
    structured_input_digest,
)
from werewolf.models.twd_tom.samples import (
    SAMPLE_FIELDS,
    SAMPLE_SCHEMA_VERSION,
)
from werewolf.models.twd_tom.speech_annotations import (
    SPEECH_ACTION_ONTOLOGY_VERSION,
    SPEECH_ANNOTATION_SCHEMA_VERSION,
    SPEECH_PARSER_PROMPT_VERSION,
    STATUS_ERROR,
    normalize_speech_annotations,
    speech_annotation_digest,
)
from werewolf.models.twd_tom.schema import (
    LABEL_PROMPT_VERSION,
    LABEL_PROVENANCE,
)
from werewolf.speech.speech_perceiver import (
    SPEECH_PARSER_GENERATION_MAX_ATTEMPTS,
)
from werewolf.trajectory import canonical_digest


BATCH_PLAN_SCHEMA_VERSION = "classic7_canonical_gameplay_batch_plan_v8"
GAME_SUMMARY_SCHEMA_VERSION = "classic7_canonical_gameplay_game_summary_v7"
BATCH_SUMMARY_SCHEMA_VERSION = "classic7_canonical_gameplay_batch_summary_v8"
TRAJECTORY_SCHEMA_VERSION = "classic7_game_interaction_trajectory_v2"
OBSERVATION_SCHEMA_VERSION = "classic7_agent_observation_v3"
OBSERVER_VIEW_PROVENANCE_SCHEMA_VERSION = "classic7_observer_view_provenance_v2"
SIMULATOR_BASELINE = "classic7-witch-parity-v1"
PRE_PUBLIC_SPEECH = "PRE_PUBLIC_SPEECH"
POST_PUBLIC_SPEECH = "POST_PUBLIC_SPEECH"
CANONICAL_COLLECTION_MODE = "canonical"
BELIEF_SNAPSHOTS_FILENAME = "belief_snapshots.jsonl"

REPO_ROOT = Path(__file__).resolve().parents[2]
_GIT_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

_TRAJECTORY_FIELDS = frozenset(
    {
        "schema_version",
        "game_id",
        "run_id",
        "source_commit",
        "simulator_baseline",
        "environment_seed",
        "runtime_config",
        "runtime_config_digest",
        "players",
        "public_event_schema_version",
        "observation_schema_version",
        "initial_public_events",
        "transitions",
        "termination",
        "public_event_digest",
        "trajectory_digest",
    }
)
_PROVENANCE_FIELDS = frozenset(
    {
        "schema_version",
        "game_id",
        "run_id",
        "source_commit",
        "simulator_baseline",
        "observation_schema_version",
        "trajectory_digest",
        "boundaries",
        "artifact_digest",
    }
)
_TRANSITION_FIELDS = frozenset(
    {
        "step_idx",
        "phase_before",
        "acting_player_id",
        "delivered_observation",
        "delivered_observation_digest",
        "submitted_action",
        "public_event_count_before",
        "public_events_appended",
        "phase_after",
        "alive_players_after",
        "terminal_after",
    }
)
_BOUNDARY_FIELDS = frozenset(
    {
        "boundary_id",
        "boundary_type",
        "step_idx",
        "speech_kind",
        "speaker_id",
        "speech_event_idx",
        "public_event_count_at_materialization",
        "public_event_digest_at_materialization",
        "observer_views",
        "boundary_digest",
    }
)
_OBSERVER_VIEW_FIELDS = frozenset(
    {"observer_id", "observation", "observation_digest"}
)
_FORBIDDEN_BELIEF_ARTIFACT_KEYS = frozenset(
    {
        "observation",
        "private_observation",
        "delivered_observation",
        "role",
        "roles",
        "true_role",
        "true_roles",
        "winner",
    }
)


def _positive_integer(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _nonnegative_integer(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value



def _sha256(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"required file not found: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required JSON artifact not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON artifact must be an object: {path}")
    return value


def _load_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"required JSONL artifact not found: {path}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            raise ValueError(f"blank JSONL line at {path}:{line_number}")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"JSONL record must be an object: {path}:{line_number}")
        records.append(value)
    return records


def _validate_embedded_digest(
    value: Mapping[str, Any],
    *,
    digest_field: str,
    artifact_name: str,
) -> str:
    payload = dict(value)
    digest = payload.pop(digest_field, None)
    if not isinstance(digest, str) or not digest:
        raise ValueError(f"{artifact_name} has no {digest_field}")
    if digest != canonical_digest(payload):
        raise ValueError(f"{artifact_name} {digest_field} mismatch")
    return digest


def validate_canonical_belief_batch(
    canonical_root: str | Path,
) -> dict[str, Any]:
    """Verify the successful batch summary chain that owns belief snapshots."""

    root = Path(canonical_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"canonical root not found: {root}")
    if (root / "batch_failure.json").exists():
        raise ValueError("canonical batch contains batch_failure.json")

    plan_path = root / "plan.json"
    plan = _load_json_object(plan_path)
    if plan.get("schema_version") != BATCH_PLAN_SCHEMA_VERSION:
        raise ValueError("canonical batch plan schema version mismatch")
    plan_digest = _validate_embedded_digest(
        plan,
        digest_field="plan_digest",
        artifact_name="canonical batch plan",
    )
    if plan.get("collection_mode") != CANONICAL_COLLECTION_MODE:
        raise ValueError("canonical batch plan is not in canonical mode")
    if plan.get("canonical_eligible") is not True:
        raise ValueError("canonical batch plan is not canonical-eligible")

    summary_path = root / "summary.json"
    summary = _load_json_object(summary_path)
    if summary.get("schema_version") != BATCH_SUMMARY_SCHEMA_VERSION:
        raise ValueError("canonical batch summary schema version mismatch")
    summary_digest = _validate_embedded_digest(
        summary,
        digest_field="summary_digest",
        artifact_name="canonical batch summary",
    )
    if summary.get("plan_digest") != plan_digest:
        raise ValueError("canonical batch summary does not match plan")
    for field_name in ("batch_code_commit", "run_id"):
        if summary.get(field_name) != plan.get(field_name):
            raise ValueError(
                f"canonical batch summary {field_name} does not match plan"
            )
    if summary.get("collection_mode") != CANONICAL_COLLECTION_MODE:
        raise ValueError("canonical batch summary is not in canonical mode")
    if summary.get("canonical_eligible") is not True:
        raise ValueError("canonical batch summary is not canonical-eligible")
    if summary.get("total_gameplay_fallback_count") != 0:
        raise ValueError("canonical batch contains gameplay fallback actions")
    if summary.get("total_missing_pre_belief_snapshot_count") != 0:
        raise ValueError("canonical batch contains missing PRE belief snapshots")
    if summary.get("total_label_snapshot_failure_count") != 0:
        raise ValueError("canonical batch contains failed label snapshots")
    if summary.get("total_speech_annotation_error_count") != 0:
        raise ValueError("canonical batch contains failed speech annotations")

    planned_count = _positive_integer(
        summary.get("planned_game_count"),
        field_name="planned_game_count",
    )
    target_count = _positive_integer(
        summary.get("target_game_count"),
        field_name="target_game_count",
    )
    completed_count = _positive_integer(
        summary.get("completed_game_count"),
        field_name="completed_game_count",
    )
    failed_count = _nonnegative_integer(
        summary.get("failed_game_count"),
        field_name="failed_game_count",
    )
    attempted_count = _positive_integer(
        summary.get("attempted_game_count"),
        field_name="attempted_game_count",
    )
    if target_count > planned_count:
        raise ValueError("canonical batch target exceeds its seed plan")
    if completed_count != target_count or summary.get("target_reached") is not True:
        raise ValueError("canonical batch did not reach its successful-game target")
    if attempted_count != completed_count + failed_count:
        raise ValueError("canonical batch attempted-game count mismatch")
    if plan.get("planned_game_count") != planned_count:
        raise ValueError("canonical batch plan and summary game counts differ")
    if plan.get("target_game_count") != target_count:
        raise ValueError("canonical batch plan and summary targets differ")

    planned_seeds = plan.get("seeds")
    if (
        not isinstance(planned_seeds, list)
        or len(planned_seeds) != planned_count
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in planned_seeds)
        or len(set(planned_seeds)) != planned_count
    ):
        raise ValueError("canonical batch plan has invalid seeds")
    if summary.get("seeds") != planned_seeds:
        raise ValueError("canonical batch summary seed plan mismatch")

    completed_seeds = summary.get("completed_seeds")
    failed_seeds = summary.get("failed_seeds")
    attempted_seeds = summary.get("attempted_seeds")
    unattempted_seeds = summary.get("unattempted_seeds")
    seed_lists = {
        "completed_seeds": (completed_seeds, completed_count),
        "failed_seeds": (failed_seeds, failed_count),
        "attempted_seeds": (attempted_seeds, attempted_count),
        "unattempted_seeds": (
            unattempted_seeds,
            planned_count - attempted_count,
        ),
    }
    for field_name, (values, expected_count) in seed_lists.items():
        if (
            not isinstance(values, list)
            or len(values) != expected_count
            or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in values)
            or len(set(values)) != len(values)
        ):
            raise ValueError(f"canonical batch {field_name} is invalid")
    completed_seed_set = set(completed_seeds)
    failed_seed_set = set(failed_seeds)
    unattempted_seed_set = set(unattempted_seeds)
    if completed_seed_set & failed_seed_set:
        raise ValueError("canonical batch has both success and failure for one seed")
    if (completed_seed_set | failed_seed_set) & unattempted_seed_set:
        raise ValueError("canonical batch attempted/unattempted seed sets overlap")
    if completed_seed_set | failed_seed_set | unattempted_seed_set != set(planned_seeds):
        raise ValueError("canonical batch seed outcomes do not partition the plan")
    expected_attempted = [
        seed for seed in planned_seeds if seed in completed_seed_set | failed_seed_set
    ]
    if attempted_seeds != expected_attempted:
        raise ValueError("canonical batch attempted seeds are not in plan order")
    if completed_seeds != [seed for seed in planned_seeds if seed in completed_seed_set]:
        raise ValueError("canonical batch completed seeds are not in plan order")
    if failed_seeds != [seed for seed in planned_seeds if seed in failed_seed_set]:
        raise ValueError("canonical batch failed seeds are not in plan order")
    if unattempted_seeds != [
        seed for seed in planned_seeds if seed in unattempted_seed_set
    ]:
        raise ValueError("canonical batch unattempted seeds are not in plan order")

    game_ids = summary.get("game_ids")
    if not isinstance(game_ids, list) or len(game_ids) != completed_count:
        raise ValueError("canonical batch summary game_ids count mismatch")
    if any(not isinstance(game_id, str) or not game_id.strip() for game_id in game_ids):
        raise ValueError("canonical batch game_ids must be non-empty text")
    if len(set(game_ids)) != len(game_ids):
        raise ValueError("canonical batch game_ids must be unique")
    summary_digests = summary.get("game_summary_digests")
    if (
        not isinstance(summary_digests, Mapping)
        or set(summary_digests) != set(game_ids)
    ):
        raise ValueError("canonical batch game summary digest set mismatch")

    failure_digests = summary.get("failure_digests")
    if not isinstance(failure_digests, Mapping) or len(failure_digests) != failed_count:
        raise ValueError("canonical batch failure digest count mismatch")

    games_root = root / "games"
    if not games_root.is_dir():
        raise FileNotFoundError(f"canonical games directory not found: {games_root}")
    game_directories = sorted(path for path in games_root.iterdir() if path.is_dir())
    if len(game_directories) != completed_count:
        raise ValueError("canonical game directory count mismatch")

    verified_by_id: dict[str, dict[str, Any]] = {}
    total_snapshot_count = 0
    total_report_count = 0
    for game_dir in game_directories:
        game_summary_path = game_dir / "summary.json"
        game_summary = _load_json_object(game_summary_path)
        if game_summary.get("schema_version") != GAME_SUMMARY_SCHEMA_VERSION:
            raise ValueError("canonical game summary schema version mismatch")
        game_summary_digest = _validate_embedded_digest(
            game_summary,
            digest_field="summary_digest",
            artifact_name=f"canonical game summary {game_dir.name}",
        )
        game_id = game_summary.get("game_id")
        if not isinstance(game_id, str) or not game_id.strip():
            raise ValueError("canonical game summary has no valid game_id")
        if game_id in verified_by_id:
            raise ValueError(f"duplicate canonical game_id: {game_id}")
        if summary_digests.get(game_id) != game_summary_digest:
            raise ValueError(f"canonical game summary digest mismatch: {game_id}")
        if game_summary.get("collection_mode") != CANONICAL_COLLECTION_MODE:
            raise ValueError(f"canonical game is not in canonical mode: {game_id}")
        if game_summary.get("canonical_eligible") is not True:
            raise ValueError(f"canonical game is not canonical-eligible: {game_id}")
        environment_seed = game_summary.get("environment_seed")
        if environment_seed not in completed_seed_set:
            raise ValueError(f"canonical game has unplanned completed seed: {game_id}")
        seed_position = planned_seeds.index(environment_seed) + 1
        expected_directory_name = f"game_{seed_position:04d}_seed_{environment_seed}"
        if game_dir.name != expected_directory_name:
            raise ValueError(f"canonical game directory/seed mismatch: {game_id}")
        call_audit = game_summary.get("call_audit")
        if not isinstance(call_audit, Mapping):
            raise ValueError(f"canonical game has no call audit: {game_id}")
        if call_audit.get("gameplay_fallback_count") != 0:
            raise ValueError(f"canonical game contains gameplay fallback: {game_id}")
        if call_audit.get("label_snapshot_failure_count") != 0:
            raise ValueError(f"canonical game contains label failure: {game_id}")
        if game_summary.get("belief_snapshot_complete") is not True:
            raise ValueError(f"canonical game has incomplete PRE labels: {game_id}")
        if game_summary.get("belief_snapshot_missing_pre_boundary_count") != 0:
            raise ValueError(f"canonical game has missing PRE labels: {game_id}")
        if game_summary.get("speech_annotation_error_count") != 0:
            raise ValueError(
                f"canonical game contains failed speech annotations: {game_id}"
            )

        belief_path = game_dir / BELIEF_SNAPSHOTS_FILENAME
        belief_sha256 = _sha256(belief_path)
        if game_summary.get("belief_snapshots_sha256") != belief_sha256:
            raise ValueError(f"canonical belief snapshot SHA-256 mismatch: {game_id}")
        snapshot_count = _positive_integer(
            game_summary.get("belief_snapshot_count"),
            field_name="belief_snapshot_count",
        )
        report_count = _positive_integer(
            game_summary.get("belief_report_count"),
            field_name="belief_report_count",
        )
        total_snapshot_count += snapshot_count
        total_report_count += report_count
        verified_by_id[game_id] = {
            "game_id": game_id,
            "game_summary_digest": game_summary_digest,
            "relative_path": str(belief_path.relative_to(root)),
            "belief_snapshots_sha256": belief_sha256,
            "belief_snapshot_count": snapshot_count,
        }

    if set(verified_by_id) != set(game_ids):
        raise ValueError("canonical game directories do not match batch summary")
    if summary.get("total_belief_snapshot_count") != total_snapshot_count:
        raise ValueError("canonical batch belief snapshot total mismatch")
    if summary.get("total_belief_report_count") != total_report_count:
        raise ValueError("canonical batch belief report total mismatch")

    failures_root = root / "failures"
    failure_directories = (
        sorted(path for path in failures_root.iterdir() if path.is_dir())
        if failures_root.is_dir()
        else []
    )
    if len(failure_directories) != failed_count:
        raise ValueError("canonical failed-game directory count mismatch")
    verified_failure_ids: set[str] = set()
    run_id = plan.get("run_id")
    commit = plan.get("batch_code_commit")
    if not isinstance(run_id, str) or not isinstance(commit, str):
        raise ValueError("canonical batch plan has invalid provenance")
    for failure_dir in failure_directories:
        failure = _load_json_object(failure_dir / "failure.json")
        failed_seed = failure.get("failed_seed")
        if failed_seed not in failed_seed_set:
            raise ValueError("canonical failure has unplanned failed seed")
        seed_position = planned_seeds.index(failed_seed) + 1
        expected_directory_name = f"game_{seed_position:04d}_seed_{failed_seed}"
        expected_game_id = _game_id(run_id, seed_position, failed_seed)
        if failure_dir.name != expected_directory_name:
            raise ValueError("canonical failure directory/seed mismatch")
        failure = _load_failure_record(
            failure_dir / "failure.json",
            run_id=run_id,
            commit=commit,
            collection_mode=CANONICAL_COLLECTION_MODE,
            expected_seed=failed_seed,
            expected_game_id=expected_game_id,
        )
        if failure_digests.get(expected_game_id) != failure["failure_digest"]:
            raise ValueError("canonical failed-game digest mismatch")
        verified_failure_ids.add(expected_game_id)
    if verified_failure_ids != set(failure_digests):
        raise ValueError("canonical failure directories do not match batch summary")

    return {
        "canonical_root": str(root),
        "plan_digest": plan_digest,
        "batch_summary_digest": summary_digest,
        "batch_summary_sha256": _sha256(summary_path),
        "game_ids": list(game_ids),
        "games": [verified_by_id[game_id] for game_id in game_ids],
        "failed_seeds": list(failed_seeds),
    }


def _reject_private_belief_artifact_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        forbidden = _FORBIDDEN_BELIEF_ARTIFACT_KEYS & set(value)
        if forbidden:
            raise ValueError(
                "belief snapshot artifact contains forbidden private/truth fields: "
                f"{sorted(forbidden)}"
            )
        for item in value.values():
            _reject_private_belief_artifact_keys(item)
    elif isinstance(value, list):
        for item in value:
            _reject_private_belief_artifact_keys(item)



def _read_code_provenance(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    try:
        top_level = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--short", "--untracked-files=all"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"canonical batch collection requires a readable Git worktree: {root}"
        ) from exc
    if Path(top_level).resolve() != root:
        raise RuntimeError("canonical batch collection must resolve the repository root")
    if _GIT_SHA_PATTERN.fullmatch(commit) is None:
        raise RuntimeError("batch code commit must be a lowercase 40-character Git SHA")
    if dirty:
        raise RuntimeError(
            "canonical batch collection requires a clean Git worktree; dirty files:\n"
            + "\n".join(dirty)
        )
    return {"batch_code_commit": commit, "git_worktree_clean": True}



def _game_id(run_id: str, game_number: int, seed: int) -> str:
    return f"{run_id}_game_{game_number:04d}_seed_{seed}"



def _expected_winner(
    players: Sequence[Mapping[str, Any]], final_alive_players: Sequence[int]
) -> str | None:
    alive = set(final_alive_players)
    alive_roles = [
        player["role"] for player in players if player["player_id"] in alive
    ]
    wolf_count = sum(role == "Werewolf" for role in alive_roles)
    non_wolf_count = len(alive_roles) - wolf_count
    if wolf_count == 0:
        return "Villager"
    if wolf_count >= non_wolf_count:
        return "Werewolf"
    return None


def validate_complete_game_artifacts(
    trajectory_path: str | Path,
    observer_views_path: str | Path,
    *,
    expected_game_id: str,
    expected_run_id: str,
    expected_seed: int,
    expected_source_commit: str,
) -> dict[str, Any]:
    """Strictly validate one completed recorder A/C0 pair and return a summary."""

    trajectory_path = Path(trajectory_path)
    observer_views_path = Path(observer_views_path)
    trajectory = _load_json_object(trajectory_path)
    provenance = _load_json_object(observer_views_path)

    if set(trajectory) != _TRAJECTORY_FIELDS:
        raise ValueError("trajectory top-level fields do not match contract")
    if set(provenance) != _PROVENANCE_FIELDS:
        raise ValueError("observer-view top-level fields do not match contract")

    expected_identity = {
        "game_id": expected_game_id,
        "run_id": expected_run_id,
        "source_commit": expected_source_commit,
    }
    for field_name, expected_value in expected_identity.items():
        if trajectory[field_name] != expected_value:
            raise ValueError(f"trajectory {field_name} mismatch")
        if provenance[field_name] != expected_value:
            raise ValueError(f"observer-view {field_name} mismatch")
    if trajectory["environment_seed"] != expected_seed:
        raise ValueError("trajectory environment_seed mismatch")
    if trajectory["schema_version"] != TRAJECTORY_SCHEMA_VERSION:
        raise ValueError("trajectory schema version mismatch")
    if provenance["schema_version"] != OBSERVER_VIEW_PROVENANCE_SCHEMA_VERSION:
        raise ValueError("observer-view schema version mismatch")
    if trajectory["simulator_baseline"] != SIMULATOR_BASELINE:
        raise ValueError("trajectory simulator baseline mismatch")
    if provenance["simulator_baseline"] != SIMULATOR_BASELINE:
        raise ValueError("observer-view simulator baseline mismatch")
    if trajectory["public_event_schema_version"] != PUBLIC_EVENT_SCHEMA_VERSION:
        raise ValueError("trajectory public-event schema mismatch")
    if trajectory["observation_schema_version"] != OBSERVATION_SCHEMA_VERSION:
        raise ValueError("trajectory observation schema mismatch")
    if provenance["observation_schema_version"] != OBSERVATION_SCHEMA_VERSION:
        raise ValueError("observer-view observation schema mismatch")

    runtime_config = trajectory["runtime_config"]
    if not isinstance(runtime_config, Mapping):
        raise TypeError("trajectory runtime_config must be a mapping")
    if trajectory["runtime_config_digest"] != canonical_digest(runtime_config):
        raise ValueError("trajectory runtime_config_digest mismatch")

    trajectory_payload = deepcopy(trajectory)
    recorded_trajectory_digest = trajectory_payload.pop("trajectory_digest")
    if recorded_trajectory_digest != canonical_digest(trajectory_payload):
        raise ValueError("trajectory_digest mismatch")
    if provenance["trajectory_digest"] != recorded_trajectory_digest:
        raise ValueError("observer-view trajectory_digest mismatch")

    provenance_payload = deepcopy(provenance)
    recorded_artifact_digest = provenance_payload.pop("artifact_digest")
    if recorded_artifact_digest != canonical_digest(provenance_payload):
        raise ValueError("observer-view artifact_digest mismatch")

    initial_events = normalize_public_events(trajectory["initial_public_events"])
    reconstructed = list(initial_events)
    transitions = trajectory["transitions"]
    if not isinstance(transitions, list) or not transitions:
        raise ValueError("complete trajectory must contain transitions")
    terminal_indices = []
    speech_steps: list[tuple[int, Mapping[str, Any], list[Mapping[str, Any]]]] = []
    for expected_step, transition in enumerate(transitions):
        if not isinstance(transition, Mapping) or set(transition) != _TRANSITION_FIELDS:
            raise ValueError("trajectory transition fields do not match contract")
        if transition["step_idx"] != expected_step:
            raise ValueError("trajectory step_idx is not contiguous")
        observation = transition["delivered_observation"]
        if not isinstance(observation, Mapping):
            raise TypeError("delivered_observation must be a mapping")
        if observation.get("current_act_idx") != transition["acting_player_id"]:
            raise ValueError("delivered observation actor mismatch")
        if observation.get("phase") != transition["phase_before"]:
            raise ValueError("delivered observation phase mismatch")
        if transition["delivered_observation_digest"] != canonical_digest(observation):
            raise ValueError("delivered_observation_digest mismatch")
        if transition["public_event_count_before"] != len(reconstructed):
            raise ValueError("public_event_count_before mismatch")
        appended = transition["public_events_appended"]
        if not isinstance(appended, list):
            raise TypeError("public_events_appended must be a list")
        action = transition["submitted_action"]
        if (
            isinstance(action, list)
            and len(action) == 2
            and action[0] in {"speech", "speech_pk"}
        ):
            speeches = [
                event
                for event in appended
                if isinstance(event, Mapping)
                and event.get("event_type") == "public_speech"
            ]
            if len(speeches) != 1:
                raise ValueError("speech step must append exactly one public_speech")
            speech = speeches[0]
            if speech.get("speaker") != f"player{transition['acting_player_id']}":
                raise ValueError("committed public speech speaker mismatch")
            content = action[1]
            if not isinstance(content, str):
                raise TypeError("speech action content must be text")
            if content != speech.get("raw_text"):
                raise ValueError("submitted and committed speech raw_text differ")
            speech_steps.append((expected_step, transition, speeches))
        reconstructed = normalize_public_events([*reconstructed, *appended])
        alive = transition["alive_players_after"]
        if (
            not isinstance(alive, list)
            or not alive
            or any(isinstance(player_id, bool) or not isinstance(player_id, int) for player_id in alive)
            or any(not 1 <= player_id <= 7 for player_id in alive)
            or len(set(alive)) != len(alive)
            or alive != sorted(alive)
        ):
            raise ValueError(
                "alive_players_after must be unique ascending Classic-7 player IDs"
            )
        if transition["terminal_after"] is True:
            terminal_indices.append(expected_step)
        elif transition["terminal_after"] is not False:
            raise TypeError("terminal_after must be boolean")

    reconstructed = normalize_public_events(reconstructed)
    if trajectory["public_event_digest"] != public_event_digest(reconstructed):
        raise ValueError("trajectory public_event_digest mismatch")
    if terminal_indices != [len(transitions) - 1]:
        raise ValueError("complete trajectory requires exactly one final terminal step")

    termination = trajectory["termination"]
    if not isinstance(termination, Mapping) or set(termination) != {
        "completion_status",
        "termination_kind",
        "winner",
        "final_alive_players",
    }:
        raise ValueError("complete termination fields do not match contract")
    if termination["completion_status"] != "COMPLETE":
        raise ValueError("production corpus requires COMPLETE games")
    if termination["termination_kind"] != "normal_game_end":
        raise ValueError("production corpus requires normal_game_end")
    if termination["winner"] not in {"Werewolf", "Villager"}:
        raise ValueError("trajectory winner is invalid")
    if transitions[-1]["alive_players_after"] != termination["final_alive_players"]:
        raise ValueError("final alive players disagree with final transition")

    players = trajectory["players"]
    if not isinstance(players, list) or len(players) != 7:
        raise ValueError("trajectory must contain seven player metadata entries")
    expected_player_fields = {
        "player_id", "role", "profile_name", "backend_id", "model_name"
    }
    for player_id, player in enumerate(players, start=1):
        if not isinstance(player, Mapping) or set(player) != expected_player_fields:
            raise ValueError("player metadata fields do not match contract")
        if player["player_id"] != player_id:
            raise ValueError("player metadata must use ascending IDs")
    mechanically_expected = _expected_winner(players, termination["final_alive_players"])
    if mechanically_expected != termination["winner"]:
        raise ValueError("trajectory winner is not mechanically valid")

    boundaries = provenance["boundaries"]
    if not isinstance(boundaries, list):
        raise TypeError("observer-view boundaries must be a list")
    if len(boundaries) != 2 * len(speech_steps):
        raise ValueError(
            "observer-view provenance must contain exactly PRE+POST for each speech step"
        )
    by_key: dict[tuple[int, str], Mapping[str, Any]] = {}
    observer_view_count = 0
    for boundary in boundaries:
        if not isinstance(boundary, Mapping) or set(boundary) != _BOUNDARY_FIELDS:
            raise ValueError("observer-view boundary fields do not match contract")
        boundary_payload = deepcopy(dict(boundary))
        recorded_boundary_digest = boundary_payload.pop("boundary_digest")
        if recorded_boundary_digest != canonical_digest(boundary_payload):
            raise ValueError("boundary_digest mismatch")
        step_idx = boundary["step_idx"]
        boundary_type = boundary["boundary_type"]
        if (
            isinstance(step_idx, bool)
            or not isinstance(step_idx, int)
            or not 0 <= step_idx < len(transitions)
        ):
            raise ValueError("boundary step_idx is invalid")
        if boundary_type not in {PRE_PUBLIC_SPEECH, POST_PUBLIC_SPEECH}:
            raise ValueError("unsupported observer-view boundary type")
        key = (step_idx, boundary_type)
        if key in by_key:
            raise ValueError("duplicate observer-view boundary")
        by_key[key] = boundary
        views = boundary["observer_views"]
        if not isinstance(views, list) or not views:
            raise ValueError("boundary observer_views cannot be empty")
        observer_ids = []
        for view in views:
            if not isinstance(view, Mapping) or set(view) != _OBSERVER_VIEW_FIELDS:
                raise ValueError("observer-view fields do not match contract")
            observer_id = view["observer_id"]
            if (
                isinstance(observer_id, bool)
                or not isinstance(observer_id, int)
                or not 1 <= observer_id <= 7
            ):
                raise ValueError("observer_id must be a Classic-7 player ID")
            observer_ids.append(observer_id)
            observation = view["observation"]
            if not isinstance(observation, Mapping):
                raise TypeError("observer observation must be a mapping")
            if observation.get("observer_id") != observer_id:
                raise ValueError("observer observation identity mismatch")
            if view["observation_digest"] != canonical_digest(observation):
                raise ValueError("observer observation digest mismatch")
            observer_view_count += 1
        if observer_ids != sorted(observer_ids) or len(set(observer_ids)) != len(observer_ids):
            raise ValueError("boundary observer IDs must be unique ascending IDs")

    for step_idx, transition, speech_events in speech_steps:
        action_kind = transition["submitted_action"][0]
        pre = by_key.get((step_idx, PRE_PUBLIC_SPEECH))
        post = by_key.get((step_idx, POST_PUBLIC_SPEECH))
        if pre is None or post is None:
            raise ValueError("speech step is missing PRE/POST observer-view boundaries")
        for boundary in (pre, post):
            if boundary["speech_kind"] != action_kind:
                raise ValueError("boundary speech kind mismatch")
            if boundary["speaker_id"] != transition["acting_player_id"]:
                raise ValueError("boundary speaker mismatch")
            expected_boundary_id = (
                f"{expected_game_id}:step_{step_idx:06d}:{boundary['boundary_type']}"
            )
            if boundary["boundary_id"] != expected_boundary_id:
                raise ValueError("boundary_id mismatch")

        if pre["speech_event_idx"] is not None:
            raise ValueError("PRE boundary speech_event_idx must be null")
        if pre["public_event_count_at_materialization"] != transition["public_event_count_before"]:
            raise ValueError("PRE public event count mismatch")
        pre_count = transition["public_event_count_before"]
        if pre["public_event_digest_at_materialization"] != public_event_digest(
            reconstructed[:pre_count]
        ):
            raise ValueError("PRE public event digest mismatch")
        pre_views = {int(view["observer_id"]): view for view in pre["observer_views"]}
        actor = int(transition["acting_player_id"])
        if actor not in pre_views:
            raise ValueError("PRE boundary is missing acting-player view")
        if pre_views[actor]["observation"] != transition["delivered_observation"]:
            raise ValueError("PRE acting-player observation differs from delivered observation")
        alive_before = (
            list(range(1, 8))
            if step_idx == 0
            else transitions[step_idx - 1]["alive_players_after"]
        )
        if sorted(pre_views) != alive_before:
            raise ValueError("PRE observer set differs from alive players before speech")
        for view in pre_views.values():
            if view["observation"].get("current_act_idx") != transition["acting_player_id"]:
                raise ValueError("PRE observer view current actor mismatch")
            if view["observation"].get("phase") != transition["phase_before"]:
                raise ValueError("PRE observer view phase mismatch")

        post_count = transition["public_event_count_before"] + len(
            transition["public_events_appended"]
        )
        if post["public_event_count_at_materialization"] != post_count:
            raise ValueError("POST public event count mismatch")
        if post["public_event_digest_at_materialization"] != public_event_digest(
            reconstructed[:post_count]
        ):
            raise ValueError("POST public event digest mismatch")
        if post["speech_event_idx"] != speech_events[0]["event_idx"]:
            raise ValueError("POST speech_event_idx mismatch")
        post_views = {int(view["observer_id"]): view for view in post["observer_views"]}
        if sorted(post_views) != transition["alive_players_after"]:
            raise ValueError("POST observer set differs from alive players after speech")

    return {
        "game_id": expected_game_id,
        "run_id": expected_run_id,
        "environment_seed": expected_seed,
        "completion_status": "COMPLETE",
        "winner": termination["winner"],
        "transition_count": len(transitions),
        "speech_transition_count": len(speech_steps),
        "boundary_count": len(boundaries),
        "public_event_count": len(reconstructed),
        "pre_public_speech_boundary_count": len(speech_steps),
        "post_public_speech_boundary_count": len(speech_steps),
        "observer_view_count": observer_view_count,
        "runtime_config_digest": trajectory["runtime_config_digest"],
        "trajectory_digest": recorded_trajectory_digest,
        "observer_view_artifact_digest": recorded_artifact_digest,
        "trajectory_sha256": _sha256(trajectory_path),
        "observer_views_sha256": _sha256(observer_views_path),
    }


def validate_speech_annotation_artifact(
    speech_annotations_path: str | Path,
    trajectory_path: str | Path,
    *,
    require_success: bool = True,
) -> dict[str, Any]:
    """Bind every parser annotation and enforce the selected mode policy."""

    speech_annotations_path = Path(speech_annotations_path)
    if not isinstance(require_success, bool):
        raise TypeError("require_success must be boolean")
    trajectory = _load_json_object(Path(trajectory_path))
    public_events = list(trajectory["initial_public_events"])
    for transition in trajectory["transitions"]:
        public_events.extend(transition["public_events_appended"])
    public_events = normalize_public_events(public_events)
    annotations = normalize_speech_annotations(
        _load_jsonl_objects(speech_annotations_path),
        public_events=public_events,
        require_complete=True,
    )
    missing_attempts = [
        annotation["event_idx"]
        for annotation in annotations
        if not annotation["generation_attempts"]
    ]
    if missing_attempts:
        raise ValueError(
            "collected speech annotations require parser generation attempts; "
            f"event_indices={missing_attempts}"
        )
    excess_attempts = [
        annotation["event_idx"]
        for annotation in annotations
        if len(annotation["generation_attempts"])
        > SPEECH_PARSER_GENERATION_MAX_ATTEMPTS
    ]
    if excess_attempts:
        raise ValueError(
            "speech parser generation attempt limit exceeded; "
            f"event_indices={excess_attempts}"
        )
    failed = [
        annotation["event_idx"]
        for annotation in annotations
        if annotation["status"] == STATUS_ERROR
    ]
    if require_success and failed:
        raise ValueError(
            "canonical speech annotations require successful parsing; "
            f"error_event_indices={failed}"
        )
    return {
        "speech_annotation_schema_version": SPEECH_ANNOTATION_SCHEMA_VERSION,
        "speech_action_ontology_version": SPEECH_ACTION_ONTOLOGY_VERSION,
        "speech_parser_prompt_version": SPEECH_PARSER_PROMPT_VERSION,
        "speech_annotation_count": len(annotations),
        "speech_annotation_action_count": sum(
            len(annotation["actions"]) for annotation in annotations
        ),
        "speech_annotation_error_count": len(failed),
        "speech_annotation_error_event_indices": failed,
        "speech_parser_generation_attempt_count": sum(
            len(annotation["generation_attempts"])
            for annotation in annotations
        ),
        "speech_no_action_count": sum(
            annotation["status"] == "no_action" for annotation in annotations
        ),
        "speech_annotation_digest": speech_annotation_digest(annotations),
        "speech_annotations_sha256": _sha256(speech_annotations_path),
    }


def validate_belief_snapshot_artifact(
    belief_snapshots_path: str | Path,
    observer_views_path: str | Path,
    speech_annotations_path: str | Path,
    *,
    expected_game_id: str,
    require_complete: bool = True,
) -> dict[str, Any]:
    """Validate raw self-reports against the canonical PRE-speech cutoffs."""

    belief_snapshots_path = Path(belief_snapshots_path)
    provenance = _load_json_object(Path(observer_views_path))
    canonical_annotations = _load_jsonl_objects(Path(speech_annotations_path))
    records = _load_jsonl_objects(belief_snapshots_path)
    pre_boundaries = {
        boundary["step_idx"]: boundary
        for boundary in provenance.get("boundaries", [])
        if isinstance(boundary, Mapping)
        and boundary.get("boundary_type") == PRE_PUBLIC_SPEECH
    }
    if not isinstance(require_complete, bool):
        raise TypeError("require_complete must be boolean")
    if require_complete and len(records) != len(pre_boundaries):
        raise ValueError(
            "belief snapshot count must equal PRE_PUBLIC_SPEECH boundary count"
        )

    seen_steps: set[int] = set()
    for record in records:
        _reject_private_belief_artifact_keys(record)
        if set(record) != SAMPLE_FIELDS:
            raise ValueError("belief snapshot fields do not match raw sample contract")
        if record["schema_version"] != SAMPLE_SCHEMA_VERSION:
            raise ValueError("belief snapshot schema version mismatch")
        if record["game_id"] != expected_game_id:
            raise ValueError("belief snapshot game_id mismatch")
        if record["label_prompt_version"] != LABEL_PROMPT_VERSION:
            raise ValueError("belief snapshot label prompt version mismatch")
        if record["label_provenance"] != LABEL_PROVENANCE:
            raise ValueError("belief snapshot label provenance mismatch")
        if record["public_event_schema_version"] != PUBLIC_EVENT_SCHEMA_VERSION:
            raise ValueError("belief snapshot public-event schema mismatch")
        if (
            record["speech_annotation_schema_version"]
            != SPEECH_ANNOTATION_SCHEMA_VERSION
        ):
            raise ValueError("belief snapshot speech-annotation schema mismatch")
        if (
            record["speech_action_ontology_version"]
            != SPEECH_ACTION_ONTOLOGY_VERSION
        ):
            raise ValueError("belief snapshot speech-action ontology mismatch")

        step_idx = record["step_idx"]
        if isinstance(step_idx, bool) or not isinstance(step_idx, int):
            raise TypeError("belief snapshot step_idx must be an integer")
        if step_idx in seen_steps:
            raise ValueError("duplicate belief snapshot step_idx")
        seen_steps.add(step_idx)
        if record["label_cutoff_step_idx"] != step_idx:
            raise ValueError("belief snapshot cutoff must equal its PRE-speech step")
        boundary = pre_boundaries.get(step_idx)
        if boundary is None:
            raise ValueError("belief snapshot has no matching PRE-speech boundary")

        expected_trigger = {
            "speech": "pre_public_speech",
            "speech_pk": "pre_public_speech_pk",
        }.get(boundary["speech_kind"])
        if record["report_trigger"] != expected_trigger:
            raise ValueError("belief snapshot report trigger mismatch")
        if record["speaker_id"] != boundary["speaker_id"]:
            raise ValueError("belief snapshot speaker mismatch")
        boundary_views = boundary["observer_views"]
        expected_observer_ids = [view["observer_id"] for view in boundary_views]
        if record["observer_ids"] != expected_observer_ids:
            raise ValueError("belief snapshot observer identities mismatch")
        expected_subjects = {f"player{player_id}" for player_id in expected_observer_ids}
        for field_name in (
            "suspected_werewolves",
            "known_werewolves",
            "known_non_werewolves",
            "belief_status",
            "belief_errors",
            "agent_backend_ids",
        ):
            value = record[field_name]
            if not isinstance(value, Mapping) or set(value) != expected_subjects:
                raise ValueError(f"belief snapshot {field_name} observer set mismatch")
        failed_reports = {
            subject: status
            for subject, status in record["belief_status"].items()
            if status != "ok"
        }
        if failed_reports:
            raise ValueError(
                "canonical belief snapshot requires status=ok for every "
                f"alive observer; failures={failed_reports}"
            )
        if any(
            error is not None
            for error in record["belief_errors"].values()
        ):
            raise ValueError(
                "successful canonical belief reports must have null errors"
            )
        if any(
            not isinstance(suspected, list)
            for suspected in record["suspected_werewolves"].values()
        ):
            raise TypeError(
                "successful canonical suspected_werewolves rows must be lists"
            )

        phases = {view["observation"].get("phase") for view in boundary_views}
        if phases != {record["phase"]}:
            raise ValueError("belief snapshot phase differs from PRE observer views")
        public_events = normalize_public_events(record["public_events"])
        speech_annotations = normalize_speech_annotations(
            record["speech_annotations"],
            public_events=public_events,
            require_complete=True,
        )
        public_speech_event_indices = {
            event["event_idx"]
            for event in public_events
            if event["event_type"] == "public_speech"
        }
        expected_speech_annotations = normalize_speech_annotations(
            [
                annotation
                for annotation in canonical_annotations
                if annotation.get("event_idx") in public_speech_event_indices
            ],
            public_events=public_events,
            require_complete=True,
        )
        if speech_annotations != expected_speech_annotations:
            raise ValueError(
                "belief snapshot speech annotations differ from canonical sidecar"
            )
        if len(public_events) != boundary["public_event_count_at_materialization"]:
            raise ValueError("belief snapshot public cutoff count mismatch")
        if record["public_event_digest"] != public_event_digest(public_events):
            raise ValueError("belief snapshot public event digest mismatch")
        if record["public_event_digest"] != (
            boundary["public_event_digest_at_materialization"]
        ):
            raise ValueError("belief snapshot public cutoff differs from PRE boundary")
        if record["speech_annotation_digest"] != speech_annotation_digest(
            speech_annotations
        ):
            raise ValueError("belief snapshot speech annotation digest mismatch")
        if record["structured_input_digest"] != structured_input_digest(
            public_events,
            speech_annotations,
        ):
            raise ValueError("belief snapshot structured input digest mismatch")
        if record["public_action_count"] != len(
            public_speech_actions(public_events, speech_annotations)
        ):
            raise ValueError("belief snapshot public action count mismatch")

    if require_complete and seen_steps != set(pre_boundaries):
        raise ValueError("belief snapshots do not cover every PRE-speech boundary")
    missing_steps = sorted(set(pre_boundaries) - seen_steps)
    return {
        "belief_snapshot_count": len(records),
        "belief_report_count": sum(
            len(record["observer_ids"])
            for record in records
        ),
        "belief_snapshot_complete": not missing_steps,
        "belief_snapshot_missing_pre_boundary_count": len(missing_steps),
        "belief_snapshot_missing_pre_step_indices": missing_steps,
        "belief_snapshots_sha256": _sha256(belief_snapshots_path),
    }



__all__ = [
    "BATCH_SUMMARY_SCHEMA_VERSION",
    "CANONICAL_COLLECTION_MODE",
    "GAME_SUMMARY_SCHEMA_VERSION",
    "_read_code_provenance",
    "validate_belief_snapshot_artifact",
    "validate_canonical_belief_batch",
    "validate_complete_game_artifacts",
    "validate_speech_annotation_artifact",
]
