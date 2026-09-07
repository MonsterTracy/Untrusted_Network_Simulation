from __future__ import annotations

import builtins
import hashlib
import importlib
import inspect
import json
import sys
from pathlib import Path

import pytest

from tests.canonical_collection.test_game_bundle import _fixture, _plan
from werewolf.artifact_io import (
    ArtifactValidationError,
    canonical_json_bytes,
    publish_artifact,
)
from werewolf.canonical_collection import (
    CanonicalFailureStage,
    TerminalOutcome,
    construct_attempt_claim,
    construct_attempt_terminal,
    construct_canonical_failure_attempt,
    construct_canonical_failure_evidence,
    initialize_attempt_ledger,
    publish_canonical_game_bundle,
    publish_canonical_failure_evidence,
    publish_ledger_record,
)
from werewolf.structured_history import plan_structured_history


def _completed_collection(
    root: Path,
    *,
    game_count: int = 5,
    target_count: int | None = None,
):
    target = game_count if target_count is None else target_count
    plan = _plan(
        collection_id="collection-publication",
        ordered_seed_pool=tuple(
            100 + index for index in range(max(target, game_count))
        ),
        target_canonical_success_count=target,
    )
    ledger = initialize_attempt_ledger(root, plan)
    executor = None
    for ordinal in range(game_count):
        claim = construct_attempt_claim(
            plan,
            ordinal=ordinal,
            attempt_id=f"attempt-{ordinal:03d}",
            claim_timestamp_utc=f"2026-09-04T00:00:{ordinal:02d}Z",
        )
        publish_ledger_record(ledger, plan, claim)
        game_id = f"game-{ordinal:03d}"
        _, _, evidence, executor = _fixture(
            plan,
            claim,
            game_id=game_id,
        )
        bundle = publish_canonical_game_bundle(
            root / "games" / game_id,
            plan=plan,
            claim=claim,
            evidence=evidence,
            replay_executor=executor,
        )
        publish_ledger_record(
            ledger,
            plan,
            construct_attempt_terminal(
                plan,
                claim,
                outcome=TerminalOutcome.CANONICAL_SUCCESS,
                canonical_game_bundle_id=game_id,
                canonical_game_bundle_digest=bundle.manifest_digest,
                terminal_timestamp_utc=f"2026-09-04T00:01:{ordinal:02d}Z",
            ),
        )
    assert executor is not None
    return plan, executor


def _completed_collection_after_failure(root: Path):
    plan = _plan(
        collection_id="collection-publication-with-failure",
        ordered_seed_pool=tuple(range(200, 206)),
        target_canonical_success_count=5,
    )
    ledger = initialize_attempt_ledger(root, plan)
    failed_claim = construct_attempt_claim(
        plan,
        ordinal=0,
        attempt_id="attempt-000",
        claim_timestamp_utc="2026-09-04T01:00:00Z",
    )
    publish_ledger_record(ledger, plan, failed_claim)
    failure = construct_canonical_failure_evidence(
        plan=plan,
        claim=failed_claim,
        game_id="failed-game-000",
        stage=CanonicalFailureStage.BELIEF_OBSERVATION,
        error_category="bounded_attempts_exhausted",
        boundary_id="boundary-003",
        observer_id="player4",
        day=1,
        phase="discussion",
        retry_exhausted=True,
        attempt_evidence=(
            construct_canonical_failure_attempt(
                attempt_index=1,
                call_id="failure-call-001",
                backend_identity=plan.backend_identity,
                model_identity=plan.model_identity,
                parser_identity=plan.parser_identity,
                prompt_identity=plan.prompt_identity,
                retry_policy_identity=plan.retry_policy_identity,
                call_budget_identity=plan.call_budget_identity,
                error_category="bounded_attempts_exhausted",
                error_message="fixture exhausted",
            ),
        ),
        partial_evidence=(),
    )
    verified_failure = publish_canonical_failure_evidence(
        root,
        plan=plan,
        claim=failed_claim,
        evidence=failure,
    )
    publish_ledger_record(
        ledger,
        plan,
        construct_attempt_terminal(
            plan,
            failed_claim,
            outcome=TerminalOutcome.CANONICAL_FAILURE,
            failure_evidence_digest=verified_failure.file_sha256,
            terminal_timestamp_utc="2026-09-04T01:01:00Z",
        ),
    )
    executor = None
    for ordinal in range(1, 6):
        claim = construct_attempt_claim(
            plan,
            ordinal=ordinal,
            attempt_id=f"attempt-{ordinal:03d}",
            claim_timestamp_utc=f"2026-09-04T01:02:{ordinal:02d}Z",
        )
        publish_ledger_record(ledger, plan, claim)
        game_id = f"game-{ordinal:03d}"
        _, _, evidence, executor = _fixture(plan, claim, game_id=game_id)
        bundle = publish_canonical_game_bundle(
            root / "games" / game_id,
            plan=plan,
            claim=claim,
            evidence=evidence,
            replay_executor=executor,
        )
        publish_ledger_record(
            ledger,
            plan,
            construct_attempt_terminal(
                plan,
                claim,
                outcome=TerminalOutcome.CANONICAL_SUCCESS,
                canonical_game_bundle_id=game_id,
                canonical_game_bundle_digest=bundle.manifest_digest,
                terminal_timestamp_utc=f"2026-09-04T01:03:{ordinal:02d}Z",
            ),
        )
    assert executor is not None
    return plan, executor


def _publication(tmp_path: Path):
    from werewolf.development_publication import (
        open_verified_collection,
        publish_development,
    )

    collection_path = tmp_path / "collection"
    plan, executor = _completed_collection(collection_path)
    collection = open_verified_collection(
        collection_path,
        replay_executor=executor,
    )
    handles = publish_development(
        collection,
        tmp_path / "development-fixture-v1",
        publication_id="development-fixture-v1",
    )
    return collection, handles


def _republish(source: Path, destination: Path, *, manifest_updates=None, files=None):
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    manifest_fields = {
        key: value
        for key, value in manifest.items()
        if key not in {"file_table", "manifest_digest"}
    }
    manifest_fields.update(manifest_updates or {})
    payloads = {
        relative_path: (source / relative_path).read_bytes()
        for relative_path in manifest["file_table"]
    }
    payloads.update(files or {})
    return publish_artifact(
        destination,
        manifest_fields=manifest_fields,
        files=payloads,
    )


def test_publication_is_plan_closed_and_has_no_subset_interface(tmp_path):
    from werewolf.development_publication import (
        open_publication,
        open_verified_collection,
        publish_development,
    )

    assert "game_ids" not in inspect.signature(publish_development).parameters
    collection_path = tmp_path / "collection"
    _plan_value, executor = _completed_collection(collection_path)
    collection = open_verified_collection(
        collection_path,
        replay_executor=executor,
    )
    handles = publish_development(
        collection,
        tmp_path / "development-fixture-v1",
        publication_id="development-fixture-v1",
    )
    reopened = open_publication(handles.public.path)

    expected = tuple(bundle.game_id for bundle in collection.games)
    assert reopened.public_view.game_ids == expected
    assert reopened.public_view.game_ids == tuple(
        game.game_id for game in reopened.public_view.games
    )
    assert len(reopened.public_view.game_ids) == 5
    assert not hasattr(reopened.public_view, "role_assignments")
    assert not hasattr(reopened.public_view, "private_replay_evidence")
    for source, published in zip(
        collection.games,
        reopened.public_view.games,
        strict=True,
    ):
        assert [item.to_record() for item in published.authoritative_pre_prefixes] == [
            item.to_record() for item in source.authoritative_pre_prefixes
        ]
        assert [item.to_record() for item in published.belief_observations] == [
            item.to_record() for item in source.belief_observations
        ]
        assert [item.to_record() for item in published.speech_annotations_v1] == [
            item.to_record() for item in source.speech_annotations_v1
        ]


def test_publication_rejects_incomplete_collection(tmp_path):
    from werewolf.development_publication import open_verified_collection

    collection_path = tmp_path / "collection"
    _plan_value, executor = _completed_collection(
        collection_path,
        game_count=4,
        target_count=5,
    )

    with pytest.raises(ArtifactValidationError, match="target"):
        open_verified_collection(collection_path, replay_executor=executor)


def test_publication_reports_every_complete_case_exclusion(tmp_path):
    from werewolf.development_publication import (
        open_verified_collection,
        publish_development,
    )

    collection_path = tmp_path / "collection"
    _plan_value, executor = _completed_collection_after_failure(collection_path)
    collection = open_verified_collection(
        collection_path,
        replay_executor=executor,
    )
    handles = publish_development(
        collection,
        tmp_path / "development-with-failure-v1",
        publication_id="development-with-failure-v1",
    )

    (exclusion,) = handles.public.public_view.complete_case_exclusions
    assert exclusion.terminal_outcome == "canonical_failure"
    assert exclusion.game_id == "failed-game-000"
    assert exclusion.failure_stage == "belief_observation"
    assert exclusion.error_category == "bounded_attempts_exhausted"
    assert exclusion.boundary_id == "boundary-003"
    assert exclusion.observer_id == "player4"
    assert exclusion.day == 1
    assert exclusion.phase == "discussion"
    assert exclusion.retry_exhausted is True


def test_five_fold_assignment_is_deterministic_balanced_and_whole_game(tmp_path):
    from werewolf.development_publication import assign_development_folds

    collection, handles = _publication(tmp_path)
    public = handles.public.public_view
    manifest = public.fold_manifest
    expected_ranked = sorted(
        (
            hashlib.sha256(
                (
                    manifest.assignment_version
                    + public.development_game_set_digest
                    + game.game_id
                    + game.bundle_digest
                ).encode("utf-8")
            ).hexdigest(),
            game.game_id,
            game.bundle_digest,
        )
        for game in public.games
    )

    assert tuple(item.ranking_digest for item in manifest.ranked_games) == tuple(
        item[0] for item in expected_ranked
    )
    assert manifest == assign_development_folds(
        tuple((game.game_id, game.bundle_digest) for game in public.games),
        development_game_set_digest=public.development_game_set_digest,
    )
    fold_games = [set(fold.game_ids) for fold in manifest.folds]
    assert max(map(len, fold_games)) - min(map(len, fold_games)) <= 1
    assert set.union(*fold_games) == set(public.game_ids)
    assert sum(len(items) for items in fold_games) == len(public.game_ids)
    assert all(
        len([fold for fold in manifest.folds if game_id in fold.game_ids]) == 1
        for game_id in public.game_ids
    )
    assert tuple(bundle.game_id for bundle in collection.games) == public.game_ids
    changed_pairs = list(
        (game.game_id, game.bundle_digest) for game in public.games
    )
    changed_pairs[0] = (changed_pairs[0][0], "f" * 64)
    with pytest.raises(ArtifactValidationError, match="game-set digest"):
        assign_development_folds(
            changed_pairs,
            development_game_set_digest=public.development_game_set_digest,
        )
    with pytest.raises(ArtifactValidationError, match="game-set digest"):
        assign_development_folds(
            tuple((game.game_id, game.bundle_digest) for game in public.games),
            development_game_set_digest="0" * 64,
        )
    assert "boundary" not in json.dumps(manifest.to_record())


def test_partition_reader_never_opens_unselected_game_records(tmp_path, monkeypatch):
    from werewolf.development_publication import open_publication

    _, handles = _publication(tmp_path)
    view = handles.public.public_view
    held_out = set(view.fold_manifest.folds[0].game_ids)
    training = tuple(g for g in view.game_ids if g not in held_out)
    read = Path.read_bytes
    def guarded(path):
        assert path.stem not in held_out
        assert "restricted" not in path.parts
        return read(path)
    monkeypatch.setattr(Path, "read_bytes", guarded)
    partition = open_publication(handles.public.path, game_ids=training)
    assert partition.public_view.game_ids == training
    assert partition.manifest_digest == handles.public.manifest_digest


def test_publication_statistics_use_the_single_count_only_planner(tmp_path):
    _collection, handles = _publication(tmp_path)
    public = handles.public.public_view
    expected_max_tokens = max(
        plan_structured_history(prefix).token_count
        for game in public.games
        for prefix in game.authoritative_pre_prefixes
    )
    expected_max_day = max(
        event.temporal_state.day
        for game in public.games
        for event in game.public_event_stream.events
    )

    assert public.max_structured_token_count == expected_max_tokens
    assert public.max_observed_day == expected_max_day
    assert public.structured_token_planner_version == (
        "classic7_structured_token_planner_v1"
    )


def test_role_sidecar_is_restricted_complete_and_public_records_are_private_free(
    tmp_path,
):
    from werewolf.development_publication import open_role_sidecar

    _collection, handles = _publication(tmp_path)
    roles = open_role_sidecar(handles.public)
    assert roles.sidecar_digest == handles.restricted.sidecar_digest
    assert tuple(game.game_id for game in roles.games) == (
        handles.public.public_view.game_ids
    )
    for game in roles.games:
        counts = {role: 0 for role in ("Werewolf", "Villager", "Seer", "Witch")}
        for _player, role in game.role_assignment:
            counts[role] += 1
        assert counts == {"Werewolf": 2, "Villager": 3, "Seer": 1, "Witch": 1}

    for path in (handles.public.path / "public" / "games").glob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        forbidden = {
            "actual_role",
            "backend_call_evidence",
            "hard_knowledge",
            "private_replay_evidence",
            "role_assignment",
            "seer_check",
            "teammate_knowledge",
            "wolf_teammates",
        }

        def keys(value):
            if isinstance(value, dict):
                return set(value).union(*(keys(item) for item in value.values()))
            if isinstance(value, list):
                return set().union(*(keys(item) for item in value))
            return set()

        assert not forbidden.intersection(keys(record))
    assert "role_assignment" not in json.dumps(handles.public.manifest)


def test_role_sidecar_rejects_parent_drift_and_invalid_role_multiplicity(tmp_path):
    from werewolf.development_publication import open_publication, open_role_sidecar

    _collection, handles = _publication(tmp_path)
    source = handles.public.path

    drifted_path = tmp_path / "drifted" / handles.public.public_view.publication_id
    _republish(
        source,
        drifted_path,
        manifest_updates={"collection_identity_digest": "0" * 64},
    )
    drifted_public = open_publication(drifted_path)
    with pytest.raises(ArtifactValidationError, match="lineage mismatch"):
        open_role_sidecar(drifted_public)

    sidecar_path = source / "restricted" / "role_sidecar.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["games"][0]["role_assignment"]["player1"] = "Villager"
    payload = dict(sidecar)
    payload.pop("sidecar_digest")
    sidecar["sidecar_digest"] = hashlib.sha256(
        canonical_json_bytes(payload)
    ).hexdigest()
    invalid_path = tmp_path / "invalid" / handles.public.public_view.publication_id
    _republish(
        source,
        invalid_path,
        manifest_updates={"role_sidecar_digest": sidecar["sidecar_digest"]},
        files={
            "restricted/role_sidecar.json": canonical_json_bytes(sidecar),
        },
    )
    invalid_public = open_publication(invalid_path)
    with pytest.raises(ArtifactValidationError, match="multiplicities"):
        open_role_sidecar(invalid_public)


def test_publication_validation_fails_on_child_byte_corruption(tmp_path):
    from werewolf.development_publication import open_publication

    _collection, handles = _publication(tmp_path)
    game_path = next((handles.public.path / "public" / "games").glob("*.json"))
    game_path.write_bytes(game_path.read_bytes() + b"corruption")

    with pytest.raises(ArtifactValidationError, match="file size mismatch"):
        open_publication(handles.public.path)


def test_publication_rejects_schema_and_private_field_drift(tmp_path):
    from werewolf.development_publication import open_publication

    _collection, handles = _publication(tmp_path)
    source = handles.public.path
    schema_mutation = tmp_path / "schema-mutation"
    _republish(
        source,
        schema_mutation,
        manifest_updates={
            "artifact_identity": "schema-mutation",
            "publication_id": "schema-mutation",
            "schema_version": "unsupported-publication-schema",
        },
    )
    with pytest.raises(ArtifactValidationError, match="schema_version"):
        open_publication(schema_mutation)

    manifest = handles.public.manifest
    game_id = manifest["development_games"][0]["game_id"]
    relative_path = f"public/games/{game_id}.json"
    game_record = json.loads((source / relative_path).read_text(encoding="utf-8"))
    game_record["call_budget_summary"]["role_assignment"] = {
        "player1": "Werewolf"
    }
    private_mutation = tmp_path / "private-mutation"
    _republish(
        source,
        private_mutation,
        manifest_updates={
            "artifact_identity": "private-mutation",
            "publication_id": "private-mutation",
        },
        files={relative_path: canonical_json_bytes(game_record)},
    )
    with pytest.raises(ArtifactValidationError, match="private field leaked"):
        open_publication(private_mutation)


def test_public_view_opens_when_restricted_sidecar_is_physically_unavailable(
    tmp_path,
):
    from werewolf.development_publication import open_publication, open_role_sidecar

    _collection, handles = _publication(tmp_path)
    handles.restricted.path.unlink()

    reopened = open_publication(handles.public.path)
    assert reopened.public_view.game_ids == handles.public.public_view.game_ids
    with pytest.raises((ArtifactValidationError, FileNotFoundError)):
        open_role_sidecar(reopened)


def test_publication_module_does_not_import_dataset(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if "dataset" in name.lower():
            raise AssertionError(f"forbidden Dataset import: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    sys.modules.pop("werewolf.development_publication", None)
    module = importlib.import_module("werewolf.development_publication")
    assert module.STRUCTURED_TOKEN_PLANNER_VERSION == (
        "classic7_structured_token_planner_v1"
    )
