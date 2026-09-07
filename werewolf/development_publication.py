"""Plan-closed Development Publication and restricted Role Sidecar."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any

from werewolf.artifact_io import (
    ArtifactValidationError,
    canonical_json_bytes,
    publish_artifact,
    sha256_bytes,
)
from werewolf.canonical_collection import (
    AttemptLedgerState,
    BeliefObservation,
    CallBudgetSummary,
    CanonicalGameBundlePublicView,
    CanonicalFailureEvidence,
    CollectionPlan,
    DeterministicReplayExecutor,
    ParserSummary,
    TerminalOutcome,
    V1SpeechAnnotation,
    VerifiedCanonicalGameBundle,
    load_collection_plan,
    validate_attempt_ledger,
    validate_canonical_failure_evidence,
    validate_canonical_game_bundle,
    validate_canonical_game_bundle_public_records,
)
from werewolf.canonical_collection.pre import AuthoritativePREPrefix
from werewolf.canonical_collection.public_history import (
    PLAYER_IDS,
    PublicEventHistory,
)
from werewolf.structured_history import (
    STRUCTURED_TOKEN_PLANNER_VERSION,
    plan_structured_history,
)


DEVELOPMENT_PUBLICATION_ARTIFACT_TYPE = "development_publication"
DEVELOPMENT_PUBLICATION_SCHEMA_VERSION = "classic7_development_publication_v1"
PUBLISHED_GAME_SCHEMA_VERSION = "classic7_published_game_v1"
DEVELOPMENT_FOLD_MANIFEST_SCHEMA_VERSION = (
    "classic7_development_fold_manifest_v1"
)
FOLD_ASSIGNMENT_VERSION = "classic7_sha256_rank_round_robin_5fold_v1"
ROLE_SIDECAR_SCHEMA_VERSION = "classic7_role_sidecar_v1"
FOLD_COUNT = 5

_SAFE_IDENTITY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CLASSIC7_ROLE_COUNTS = {
    "Werewolf": 2,
    "Villager": 3,
    "Seer": 1,
    "Witch": 1,
}
_PUBLICATION_DOMAIN_FIELDS = frozenset(
    {
        "artifact_type",
        "schema_version",
        "artifact_identity",
        "publication_id",
        "collection_id",
        "collection_plan_digest",
        "collection_identity_digest",
        "development_game_set_digest",
        "development_games",
        "game_count",
        "attempt_count",
        "canonical_success_count",
        "canonical_failure_count",
        "interrupted_failure_count",
        "complete_case_exclusions",
        "pre_boundary_count",
        "belief_observation_count",
        "public_event_count",
        "v1_annotation_count",
        "max_observed_day",
        "max_structured_token_count",
        "structured_token_planner_version",
        "fold_assignment_version",
        "development_fold_manifest_digest",
        "role_sidecar_digest",
        "source_revision",
    }
)


@dataclass(frozen=True)
class CollectionExclusion:
    ordinal: int
    seed: int
    attempt_id: str
    claim_record_digest: str
    terminal_outcome: str
    terminal_record_digest: str
    failure_evidence_digest: str | None
    game_id: str | None
    failure_stage: str | None
    error_category: str | None
    boundary_id: str | None
    observer_id: str | None
    day: int | None
    phase: str | None
    retry_exhausted: bool | None

    def to_record(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "seed": self.seed,
            "attempt_id": self.attempt_id,
            "claim_record_digest": self.claim_record_digest,
            "terminal_outcome": self.terminal_outcome,
            "terminal_record_digest": self.terminal_record_digest,
            "failure_evidence_digest": self.failure_evidence_digest,
            "game_id": self.game_id,
            "failure_stage": self.failure_stage,
            "error_category": self.error_category,
            "boundary_id": self.boundary_id,
            "observer_id": self.observer_id,
            "day": self.day,
            "phase": self.phase,
            "retry_exhausted": self.retry_exhausted,
        }


@dataclass(frozen=True)
class VerifiedCollection:
    path: Path
    plan: CollectionPlan
    ledger_state: AttemptLedgerState
    games: tuple[VerifiedCanonicalGameBundle, ...]
    exclusions: tuple[CollectionExclusion, ...]
    collection_identity_digest: str


@dataclass(frozen=True)
class PublishedGame:
    schema_version: str
    game_id: str
    bundle_digest: str
    public_event_stream: PublicEventHistory
    authoritative_pre_prefixes: tuple[AuthoritativePREPrefix, ...]
    belief_observations: tuple[BeliefObservation, ...]
    speech_annotations_v1: tuple[V1SpeechAnnotation, ...]
    call_budget_summary: CallBudgetSummary
    parser_summary: ParserSummary
    record_digest: str

    def to_record(self) -> dict[str, Any]:
        return {
            **_published_game_without_digest(self),
            "record_digest": self.record_digest,
        }


@dataclass(frozen=True)
class RankedFoldGame:
    ranking_digest: str
    game_id: str
    bundle_digest: str
    fold_index: int

    def to_record(self) -> dict[str, Any]:
        return {
            "ranking_digest": self.ranking_digest,
            "game_id": self.game_id,
            "bundle_digest": self.bundle_digest,
            "fold_index": self.fold_index,
        }


@dataclass(frozen=True)
class DevelopmentFold:
    fold_index: int
    game_ids: tuple[str, ...]
    bundle_digests: tuple[str, ...]

    def to_record(self) -> dict[str, Any]:
        return {
            "fold_index": self.fold_index,
            "held_out_games": [
                {"game_id": game_id, "bundle_digest": bundle_digest}
                for game_id, bundle_digest in zip(
                    self.game_ids,
                    self.bundle_digests,
                    strict=True,
                )
            ],
        }


@dataclass(frozen=True)
class DevelopmentFoldManifest:
    schema_version: str
    assignment_version: str
    fold_count: int
    development_game_set_digest: str
    ranked_games: tuple[RankedFoldGame, ...]
    ranking_order_digest: str
    folds: tuple[DevelopmentFold, ...]
    manifest_digest: str

    def to_record(self) -> dict[str, Any]:
        return {
            **_fold_manifest_without_digest(self),
            "manifest_digest": self.manifest_digest,
        }


@dataclass(frozen=True)
class PublicationPublicView:
    publication_id: str
    collection_id: str
    collection_plan_digest: str
    collection_identity_digest: str
    development_game_set_digest: str
    games: tuple[PublishedGame, ...]
    fold_manifest: DevelopmentFoldManifest
    complete_case_exclusions: tuple[CollectionExclusion, ...]
    max_observed_day: int
    max_structured_token_count: int
    structured_token_planner_version: str

    @property
    def game_ids(self) -> tuple[str, ...]:
        return tuple(game.game_id for game in self.games)

    def load_game(self, game_id: str) -> PublishedGame:
        for game in self.games:
            if game.game_id == game_id:
                return game
        raise ArtifactValidationError("game is outside publication partition")


@dataclass(frozen=True)
class VerifiedDevelopmentPublication:
    path: Path
    manifest: dict[str, Any]
    manifest_digest: str
    public_view: PublicationPublicView


@dataclass(frozen=True)
class RoleSidecarGame:
    game_id: str
    bundle_digest: str
    role_assignment: tuple[tuple[str, str], ...]

    def to_record(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "bundle_digest": self.bundle_digest,
            "role_assignment": dict(self.role_assignment),
        }


@dataclass(frozen=True)
class VerifiedRoleSidecar:
    path: Path
    publication_id: str
    collection_identity_digest: str
    development_game_set_digest: str
    games: tuple[RoleSidecarGame, ...]
    sidecar_digest: str


@dataclass(frozen=True)
class DevelopmentPublicationHandles:
    public: VerifiedDevelopmentPublication
    restricted: VerifiedRoleSidecar


def _require_digest(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ArtifactValidationError(f"{field_name} must be a lowercase SHA-256")
    return value


def _load_canonical_json(path: Path) -> dict[str, Any]:
    try:
        data = path.read_bytes()
        value = json.loads(data.decode("utf-8"))
        if not isinstance(value, dict) or canonical_json_bytes(value) != data:
            raise ValueError("JSON bytes are not canonical")
    except (OSError, UnicodeError, TypeError, ValueError) as error:
        raise ArtifactValidationError(f"invalid canonical JSON: {path.name}") from error
    return value


def _collection_identity(
    plan: CollectionPlan,
    state: AttemptLedgerState,
) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "schema_version": "classic7_verified_collection_identity_v1",
                "collection_id": plan.collection_id,
                "collection_plan_digest": plan.plan_digest,
                "claim_record_digests": [item.record_digest for item in state.claims],
                "terminal_record_digests": [
                    item.record_digest for item in state.terminals
                ],
            }
        )
    )


def _validate_verified_collection(collection: VerifiedCollection) -> None:
    if collection.plan.plan_digest != collection.ledger_state.plan_digest:
        raise ArtifactValidationError("verified collection plan/ledger mismatch")
    if (
        not collection.ledger_state.target_reached
        or collection.ledger_state.open_claim is not None
        or collection.ledger_state.canonical_success_count
        != collection.plan.target_canonical_success_count
    ):
        raise ArtifactValidationError("verified collection target is incomplete")
    if collection.collection_identity_digest != _collection_identity(
        collection.plan,
        collection.ledger_state,
    ):
        raise ArtifactValidationError("verified collection identity mismatch")
    expected = tuple(
        (
            terminal.canonical_game_bundle_id,
            terminal.canonical_game_bundle_digest,
        )
        for terminal in collection.ledger_state.terminals
        if terminal.outcome is TerminalOutcome.CANONICAL_SUCCESS
    )
    actual = tuple(
        (bundle.game_id, bundle.manifest_digest) for bundle in collection.games
    )
    if actual != expected:
        raise ArtifactValidationError("verified collection game set was substituted")
    expected_root = (collection.path / "games").resolve()
    if any(
        bundle.path.parent.resolve() != expected_root
        for bundle in collection.games
    ):
        raise ArtifactValidationError("verified collection bundle path is not bound")


def _exclusion(
    terminal: Any,
    failure: CanonicalFailureEvidence | None,
    claim_record_digest: str,
) -> CollectionExclusion:
    return CollectionExclusion(
        ordinal=terminal.ordinal,
        seed=terminal.seed,
        attempt_id=terminal.attempt_id,
        claim_record_digest=claim_record_digest,
        terminal_outcome=terminal.outcome.value,
        terminal_record_digest=terminal.record_digest,
        failure_evidence_digest=terminal.failure_evidence_digest,
        game_id=None if failure is None else failure.game_id,
        failure_stage=None if failure is None else failure.stage.value,
        error_category=None if failure is None else failure.error_category,
        boundary_id=None if failure is None else failure.boundary_id,
        observer_id=None if failure is None else failure.observer_id,
        day=None if failure is None else failure.day,
        phase=None if failure is None else failure.phase,
        retry_exhausted=None if failure is None else failure.retry_exhausted,
    )


def _exclusion_from_record(value: Mapping[str, Any]) -> CollectionExclusion:
    required = {
        "ordinal",
        "seed",
        "attempt_id",
        "claim_record_digest",
        "terminal_outcome",
        "terminal_record_digest",
        "failure_evidence_digest",
        "game_id",
        "failure_stage",
        "error_category",
        "boundary_id",
        "observer_id",
        "day",
        "phase",
        "retry_exhausted",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ArtifactValidationError("complete-case exclusion field set mismatch")
    for field in ("ordinal", "seed"):
        if (
            isinstance(value[field], bool)
            or not isinstance(value[field], int)
            or value[field] < 0
        ):
            raise ArtifactValidationError(f"complete-case exclusion {field} is invalid")
    attempt_id = value["attempt_id"]
    if not isinstance(attempt_id, str) or _SAFE_IDENTITY.fullmatch(attempt_id) is None:
        raise ArtifactValidationError("complete-case exclusion attempt_id is invalid")
    _require_digest(value["terminal_record_digest"], "terminal_record_digest")
    _require_digest(value["claim_record_digest"], "claim_record_digest")
    outcome = value["terminal_outcome"]
    if outcome == TerminalOutcome.CANONICAL_FAILURE.value:
        _require_digest(value["failure_evidence_digest"], "failure_evidence_digest")
        if not all(
            isinstance(value[field], str) and value[field]
            for field in ("game_id", "failure_stage", "error_category")
        ) or not isinstance(value["retry_exhausted"], bool):
            raise ArtifactValidationError("canonical failure exclusion is incomplete")
    elif outcome == TerminalOutcome.INTERRUPTED_FAILURE.value:
        if any(
            value[field] is not None
            for field in (
                "failure_evidence_digest",
                "game_id",
                "failure_stage",
                "error_category",
                "boundary_id",
                "observer_id",
                "day",
                "phase",
                "retry_exhausted",
            )
        ):
            raise ArtifactValidationError(
                "interrupted failure exclusion contains invented evidence"
            )
    else:
        raise ArtifactValidationError("complete-case exclusion outcome is invalid")
    exclusion = CollectionExclusion(
        ordinal=value["ordinal"],
        seed=value["seed"],
        attempt_id=attempt_id,
        claim_record_digest=value["claim_record_digest"],
        terminal_outcome=outcome,
        terminal_record_digest=value["terminal_record_digest"],
        failure_evidence_digest=value["failure_evidence_digest"],
        game_id=value["game_id"],
        failure_stage=value["failure_stage"],
        error_category=value["error_category"],
        boundary_id=value["boundary_id"],
        observer_id=value["observer_id"],
        day=value["day"],
        phase=value["phase"],
        retry_exhausted=value["retry_exhausted"],
    )
    if exclusion.to_record() != value:
        raise ArtifactValidationError("complete-case exclusion is not canonical")
    return exclusion


def open_verified_collection(
    path: Path | str,
    *,
    replay_executor: DeterministicReplayExecutor,
) -> VerifiedCollection:
    """Open exactly the completed ledger-defined Collection Plan."""

    collection_path = Path(path)
    try:
        plan = load_collection_plan(collection_path)
        state = validate_attempt_ledger(collection_path / "attempt_ledger", plan)
    except (TypeError, ValueError) as error:
        raise ArtifactValidationError(f"invalid collection: {error}") from error
    if not state.target_reached or state.open_claim is not None:
        raise ArtifactValidationError("collection target is not durably complete")
    if state.canonical_success_count != plan.target_canonical_success_count:
        raise ArtifactValidationError("collection target success count mismatch")
    claims = {claim.ordinal: claim for claim in state.claims}
    games: list[VerifiedCanonicalGameBundle] = []
    exclusions: list[CollectionExclusion] = []
    for terminal in state.terminals:
        claim = claims[terminal.ordinal]
        if terminal.outcome is TerminalOutcome.CANONICAL_SUCCESS:
            assert terminal.canonical_game_bundle_id is not None
            assert terminal.canonical_game_bundle_digest is not None
            bundle = validate_canonical_game_bundle(
                collection_path / "games" / terminal.canonical_game_bundle_id,
                plan=plan,
                claim=claim,
                replay_executor=replay_executor,
            )
            if bundle.manifest_digest != terminal.canonical_game_bundle_digest:
                raise ArtifactValidationError(
                    "ledger terminal disagrees with Canonical Game Bundle"
                )
            games.append(bundle)
            continue
        failure = None
        if terminal.outcome is TerminalOutcome.CANONICAL_FAILURE:
            verified_failure = validate_canonical_failure_evidence(
                collection_path
                / "attempts"
                / terminal.attempt_id
                / "failure_evidence.json",
                plan=plan,
                claim=claim,
            )
            if verified_failure.file_sha256 != terminal.failure_evidence_digest:
                raise ArtifactValidationError(
                    "ledger terminal disagrees with failure evidence"
                )
            failure = verified_failure.evidence
        exclusions.append(_exclusion(terminal, failure, claim.record_digest))
    if len(games) != plan.target_canonical_success_count:
        raise ArtifactValidationError("Development Game Set is not plan-closed")
    return VerifiedCollection(
        path=collection_path,
        plan=plan,
        ledger_state=state,
        games=tuple(games),
        exclusions=tuple(exclusions),
        collection_identity_digest=_collection_identity(plan, state),
    )


def _development_game_set_digest(
    games: Sequence[tuple[str, str]],
) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            [
                {"game_id": game_id, "bundle_digest": bundle_digest}
                for game_id, bundle_digest in games
            ]
        )
    )


def _fold_manifest_without_digest(
    manifest: DevelopmentFoldManifest,
) -> dict[str, Any]:
    return {
        "schema_version": manifest.schema_version,
        "assignment_version": manifest.assignment_version,
        "fold_count": manifest.fold_count,
        "development_game_set_digest": manifest.development_game_set_digest,
        "ranked_games": [item.to_record() for item in manifest.ranked_games],
        "ranking_order_digest": manifest.ranking_order_digest,
        "folds": [fold.to_record() for fold in manifest.folds],
    }


def assign_development_folds(
    games: Sequence[tuple[str, str]],
    *,
    development_game_set_digest: str,
) -> DevelopmentFoldManifest:
    """Assign exact whole games to the frozen deterministic five folds."""

    frozen = tuple(games)
    _require_digest(development_game_set_digest, "development_game_set_digest")
    if len(frozen) < FOLD_COUNT:
        raise ArtifactValidationError(
            "Development Publication requires at least 5 games"
        )
    if len({game_id for game_id, _ in frozen}) != len(frozen):
        raise ArtifactValidationError("Development Game Set has duplicate game IDs")
    if development_game_set_digest != _development_game_set_digest(frozen):
        raise ArtifactValidationError("Development game-set digest does not bind ordered games")
    ranked_values: list[tuple[str, str, str]] = []
    for game_id, bundle_digest in frozen:
        if not isinstance(game_id, str) or _SAFE_IDENTITY.fullmatch(game_id) is None:
            raise ArtifactValidationError("invalid Development Game Set game ID")
        _require_digest(bundle_digest, "bundle_digest")
        ranking_digest = sha256_bytes(
            (
                FOLD_ASSIGNMENT_VERSION
                + development_game_set_digest
                + game_id
                + bundle_digest
            ).encode("utf-8")
        )
        ranked_values.append((ranking_digest, game_id, bundle_digest))
    ranked_values.sort(key=lambda item: (item[0], item[1]))
    ranked = tuple(
        RankedFoldGame(
            ranking_digest=ranking_digest,
            game_id=game_id,
            bundle_digest=bundle_digest,
            fold_index=index % FOLD_COUNT,
        )
        for index, (ranking_digest, game_id, bundle_digest) in enumerate(
            ranked_values
        )
    )
    ranking_order_digest = sha256_bytes(
        canonical_json_bytes([item.to_record() for item in ranked])
    )
    folds = tuple(
        DevelopmentFold(
            fold_index=fold_index,
            game_ids=tuple(
                item.game_id for item in ranked if item.fold_index == fold_index
            ),
            bundle_digests=tuple(
                item.bundle_digest
                for item in ranked
                if item.fold_index == fold_index
            ),
        )
        for fold_index in range(FOLD_COUNT)
    )
    provisional = DevelopmentFoldManifest(
        schema_version=DEVELOPMENT_FOLD_MANIFEST_SCHEMA_VERSION,
        assignment_version=FOLD_ASSIGNMENT_VERSION,
        fold_count=FOLD_COUNT,
        development_game_set_digest=development_game_set_digest,
        ranked_games=ranked,
        ranking_order_digest=ranking_order_digest,
        folds=folds,
        manifest_digest="",
    )
    return replace(
        provisional,
        manifest_digest=sha256_bytes(
            canonical_json_bytes(_fold_manifest_without_digest(provisional))
        ),
    )


def _validate_fold_manifest_record(
    value: Mapping[str, Any],
    games: Sequence[tuple[str, str]],
    development_game_set_digest: str,
) -> DevelopmentFoldManifest:
    expected = assign_development_folds(
        games,
        development_game_set_digest=development_game_set_digest,
    )
    if value != expected.to_record():
        raise ArtifactValidationError("Development Fold Manifest is not canonical")
    return expected


def _published_game_without_digest(game: PublishedGame) -> dict[str, Any]:
    return {
        "schema_version": game.schema_version,
        "game_id": game.game_id,
        "bundle_digest": game.bundle_digest,
        "public_event_stream": game.public_event_stream.to_records(),
        "authoritative_pre_prefixes": [
            prefix.to_record() for prefix in game.authoritative_pre_prefixes
        ],
        "belief_observations": [
            item.to_record() for item in game.belief_observations
        ],
        "speech_annotations_v1": [
            item.to_record() for item in game.speech_annotations_v1
        ],
        "call_budget_summary": game.call_budget_summary.to_record(),
        "parser_summary": game.parser_summary.to_record(),
    }


def _published_game_from_bundle(bundle: VerifiedCanonicalGameBundle) -> PublishedGame:
    provisional = PublishedGame(
        schema_version=PUBLISHED_GAME_SCHEMA_VERSION,
        game_id=bundle.game_id,
        bundle_digest=bundle.manifest_digest,
        public_event_stream=bundle.public_event_stream,
        authoritative_pre_prefixes=bundle.authoritative_pre_prefixes,
        belief_observations=bundle.belief_observations,
        speech_annotations_v1=bundle.speech_annotations_v1,
        call_budget_summary=bundle.call_budget_summary,
        parser_summary=bundle.parser_summary,
        record_digest="",
    )
    return replace(
        provisional,
        record_digest=sha256_bytes(
            canonical_json_bytes(_published_game_without_digest(provisional))
        ),
    )


def _published_game_from_record(value: Mapping[str, Any]) -> PublishedGame:
    required = {
        "schema_version",
        "game_id",
        "bundle_digest",
        "public_event_stream",
        "authoritative_pre_prefixes",
        "belief_observations",
        "speech_annotations_v1",
        "call_budget_summary",
        "parser_summary",
        "record_digest",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ArtifactValidationError("published game has an invalid field set")
    if value["schema_version"] != PUBLISHED_GAME_SCHEMA_VERSION:
        raise ArtifactValidationError("unsupported published game schema")
    game_id = value["game_id"]
    if not isinstance(game_id, str) or _SAFE_IDENTITY.fullmatch(game_id) is None:
        raise ArtifactValidationError("published game has an invalid identity")
    bundle_digest = _require_digest(value["bundle_digest"], "bundle_digest")
    public: CanonicalGameBundlePublicView = (
        validate_canonical_game_bundle_public_records(
            game_id=game_id,
            public_event_records=value["public_event_stream"],
            authoritative_pre_prefix_records=value["authoritative_pre_prefixes"],
            belief_observation_records=value["belief_observations"],
            speech_annotation_records=value["speech_annotations_v1"],
            call_budget_record=value["call_budget_summary"],
            parser_summary_record=value["parser_summary"],
        )
    )
    provisional = PublishedGame(
        schema_version=PUBLISHED_GAME_SCHEMA_VERSION,
        game_id=game_id,
        bundle_digest=bundle_digest,
        public_event_stream=public.public_event_stream,
        authoritative_pre_prefixes=public.authoritative_pre_prefixes,
        belief_observations=public.belief_observations,
        speech_annotations_v1=public.speech_annotations_v1,
        call_budget_summary=public.call_budget_summary,
        parser_summary=public.parser_summary,
        record_digest="",
    )
    result = replace(
        provisional,
        record_digest=sha256_bytes(
            canonical_json_bytes(_published_game_without_digest(provisional))
        ),
    )
    if result.to_record() != value:
        raise ArtifactValidationError("published game record is not canonical")
    return result


def _role_assignment(value: Sequence[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
    roles = tuple(value)
    if tuple(player for player, _ in roles) != PLAYER_IDS:
        raise ArtifactValidationError("Role Sidecar must cover exact Classic7 seats")
    if Counter(role for _, role in roles) != Counter(_CLASSIC7_ROLE_COUNTS):
        raise ArtifactValidationError(
            "Role Sidecar has invalid Classic7 multiplicities"
        )
    return roles


def _role_sidecar_record(
    *,
    publication_id: str,
    collection_identity_digest: str,
    development_game_set_digest: str,
    games: Sequence[RoleSidecarGame],
) -> dict[str, Any]:
    without_digest = {
        "schema_version": ROLE_SIDECAR_SCHEMA_VERSION,
        "publication_id": publication_id,
        "collection_identity_digest": collection_identity_digest,
        "development_game_set_digest": development_game_set_digest,
        "games": [game.to_record() for game in games],
    }
    return {
        **without_digest,
        "sidecar_digest": sha256_bytes(canonical_json_bytes(without_digest)),
    }


def _publication_statistics(games: Sequence[PublishedGame]) -> dict[str, int]:
    plans = [
        plan_structured_history(prefix)
        for game in games
        for prefix in game.authoritative_pre_prefixes
    ]
    return {
        "pre_boundary_count": sum(
            len(game.authoritative_pre_prefixes) for game in games
        ),
        "belief_observation_count": sum(
            len(game.belief_observations) for game in games
        ),
        "public_event_count": sum(
            len(game.public_event_stream.events) for game in games
        ),
        "v1_annotation_count": sum(
            len(game.speech_annotations_v1) for game in games
        ),
        "max_observed_day": max(
            event.temporal_state.day
            for game in games
            for event in game.public_event_stream.events
        ),
        "max_structured_token_count": max(plan.token_count for plan in plans),
    }


def publish_development(
    collection: VerifiedCollection,
    destination: Path | str,
    *,
    publication_id: str,
) -> DevelopmentPublicationHandles:
    """Publish the exact ledger-defined Development Game Set."""

    if not isinstance(collection, VerifiedCollection):
        raise TypeError("collection must be a VerifiedCollection")
    _validate_verified_collection(collection)
    if _SAFE_IDENTITY.fullmatch(publication_id) is None:
        raise ValueError("publication_id must be one safe path component")
    destination = Path(destination)
    if destination.name != publication_id:
        raise ValueError("publication destination must end with publication_id")
    if len(collection.games) < FOLD_COUNT:
        raise ArtifactValidationError(
            "Development Publication requires at least 5 games"
        )
    if len(collection.games) != collection.plan.target_canonical_success_count:
        raise ArtifactValidationError("Development Game Set is not plan-closed")

    games = tuple(_published_game_from_bundle(bundle) for bundle in collection.games)
    game_pairs = tuple((game.game_id, game.bundle_digest) for game in games)
    game_set_digest = _development_game_set_digest(game_pairs)
    folds = assign_development_folds(
        game_pairs,
        development_game_set_digest=game_set_digest,
    )
    sidecar_games = tuple(
        RoleSidecarGame(
            game_id=bundle.game_id,
            bundle_digest=bundle.manifest_digest,
            role_assignment=_role_assignment(
                bundle.private_replay_evidence.role_assignment
            ),
        )
        for bundle in collection.games
    )
    sidecar_record = _role_sidecar_record(
        publication_id=publication_id,
        collection_identity_digest=collection.collection_identity_digest,
        development_game_set_digest=game_set_digest,
        games=sidecar_games,
    )
    statistics = _publication_statistics(games)
    outcome_counts = Counter(
        terminal.outcome for terminal in collection.ledger_state.terminals
    )
    claims = {claim.ordinal: claim for claim in collection.ledger_state.claims}
    success_terminals = tuple(
        terminal
        for terminal in collection.ledger_state.terminals
        if terminal.outcome is TerminalOutcome.CANONICAL_SUCCESS
    )
    development_games = []
    for game, terminal in zip(games, success_terminals, strict=True):
        claim = claims[terminal.ordinal]
        development_games.append(
            {
                "ordinal": terminal.ordinal,
                "seed": terminal.seed,
                "attempt_id": terminal.attempt_id,
                "claim_record_digest": claim.record_digest,
                "terminal_record_digest": terminal.record_digest,
                "game_id": game.game_id,
                "bundle_digest": game.bundle_digest,
                "published_record_digest": game.record_digest,
            }
        )
    files = {
        **{
            f"public/games/{game.game_id}.json": canonical_json_bytes(
                game.to_record()
            )
            for game in games
        },
        "public/development_fold_manifest.json": canonical_json_bytes(
            folds.to_record()
        ),
        "restricted/role_sidecar.json": canonical_json_bytes(sidecar_record),
    }
    publish_artifact(
        destination,
        manifest_fields={
            "artifact_type": DEVELOPMENT_PUBLICATION_ARTIFACT_TYPE,
            "schema_version": DEVELOPMENT_PUBLICATION_SCHEMA_VERSION,
            "artifact_identity": publication_id,
            "publication_id": publication_id,
            "collection_id": collection.plan.collection_id,
            "collection_plan_digest": collection.plan.plan_digest,
            "collection_identity_digest": collection.collection_identity_digest,
            "development_game_set_digest": game_set_digest,
            "development_games": development_games,
            "game_count": len(games),
            "attempt_count": len(collection.ledger_state.claims),
            "canonical_success_count": outcome_counts[
                TerminalOutcome.CANONICAL_SUCCESS
            ],
            "canonical_failure_count": outcome_counts[
                TerminalOutcome.CANONICAL_FAILURE
            ],
            "interrupted_failure_count": outcome_counts[
                TerminalOutcome.INTERRUPTED_FAILURE
            ],
            "complete_case_exclusions": [
                item.to_record() for item in collection.exclusions
            ],
            **statistics,
            "structured_token_planner_version": STRUCTURED_TOKEN_PLANNER_VERSION,
            "fold_assignment_version": FOLD_ASSIGNMENT_VERSION,
            "development_fold_manifest_digest": folds.manifest_digest,
            "role_sidecar_digest": sidecar_record["sidecar_digest"],
            "source_revision": collection.plan.source_revision,
        },
        files=files,
    )
    public = open_publication(destination)
    restricted = open_role_sidecar(public)
    return DevelopmentPublicationHandles(public=public, restricted=restricted)


def _validate_manifest_shape(path: Path, manifest: Mapping[str, Any]) -> None:
    domain_fields = set(manifest) - {"file_table", "manifest_digest"}
    if domain_fields != _PUBLICATION_DOMAIN_FIELDS:
        raise ArtifactValidationError("Development Publication manifest field mismatch")
    expected = {
        "artifact_type": DEVELOPMENT_PUBLICATION_ARTIFACT_TYPE,
        "schema_version": DEVELOPMENT_PUBLICATION_SCHEMA_VERSION,
        "structured_token_planner_version": STRUCTURED_TOKEN_PLANNER_VERSION,
        "fold_assignment_version": FOLD_ASSIGNMENT_VERSION,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ArtifactValidationError(f"Development Publication {field} mismatch")
    if manifest.get("artifact_identity") != manifest.get("publication_id"):
        raise ArtifactValidationError("publication artifact identity mismatch")
    if path.name != manifest.get("publication_id"):
        raise ArtifactValidationError("publication path identity mismatch")
    if _SAFE_IDENTITY.fullmatch(manifest["publication_id"]) is None:
        raise ArtifactValidationError("publication identity is invalid")
    for field in ("collection_id", "source_revision"):
        if not isinstance(manifest.get(field), str) or not manifest[field].strip():
            raise ArtifactValidationError(
                f"Development Publication {field} is invalid"
            )
    for field in (
        "collection_plan_digest",
        "collection_identity_digest",
        "development_game_set_digest",
        "development_fold_manifest_digest",
        "role_sidecar_digest",
    ):
        _require_digest(manifest.get(field), field)


def _open_public_envelope(path: Path | str, game_ids=None) -> tuple[Path, dict[str, Any], str]:
    """Validate the publication envelope and public files without role bytes."""

    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        raise ArtifactValidationError("Development Publication is missing or invalid")
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ArtifactValidationError("Development Publication manifest is missing")
    manifest = _load_canonical_json(manifest_path)
    if "manifest_digest" not in manifest or "file_table" not in manifest:
        raise ArtifactValidationError("Development Publication envelope is incomplete")
    without_digest = dict(manifest)
    manifest_digest = without_digest.pop("manifest_digest")
    _require_digest(manifest_digest, "manifest_digest")
    if sha256_bytes(canonical_json_bytes(without_digest)) != manifest_digest:
        raise ArtifactValidationError(
            "Development Publication manifest digest mismatch"
        )
    file_table = manifest.get("file_table")
    if not isinstance(file_table, Mapping):
        raise ArtifactValidationError("Development Publication file table is invalid")
    for relative_path, entry in file_table.items():
        relative = PurePosixPath(relative_path)
        if (
            not isinstance(relative_path, str)
            or relative.is_absolute()
            or relative.as_posix() != relative_path
            or any(part in {"", ".", ".."} for part in relative.parts)
            or not isinstance(entry, Mapping)
            or set(entry) != {"byte_size", "sha256"}
            or isinstance(entry["byte_size"], bool)
            or not isinstance(entry["byte_size"], int)
            or entry["byte_size"] < 0
        ):
            raise ArtifactValidationError(
                "Development Publication file table entry is invalid"
            )
        _require_digest(entry["sha256"], "file sha256")
        if not relative_path.startswith("public/"):
            continue
        if game_ids is not None and relative_path.startswith("public/games/") and relative.stem not in game_ids:
            continue
        child = root / relative_path
        if child.is_symlink() or not child.is_file():
            raise ArtifactValidationError(f"public file is missing: {relative_path}")
        data = child.read_bytes()
        if len(data) != entry["byte_size"]:
            raise ArtifactValidationError(f"file size mismatch: {relative_path}")
        if sha256_bytes(data) != entry["sha256"]:
            raise ArtifactValidationError(f"file digest mismatch: {relative_path}")
    actual_public_files = {
        path.relative_to(root).as_posix()
        for path in (root / "public").rglob("*")
        if path.is_file()
    }
    expected_public_files = {
        relative_path
        for relative_path in file_table
        if relative_path.startswith("public/")
    }
    if actual_public_files != expected_public_files:
        raise ArtifactValidationError("Development Publication public files differ")
    return root, manifest, manifest_digest


def open_publication(path: Path | str, *, game_ids=None) -> VerifiedDevelopmentPublication:
    """Open the Dataset-facing public view without parsing Role Sidecar roles."""

    if game_ids is not None:
        game_ids = tuple(game_ids)
        if not game_ids or len(set(game_ids)) != len(game_ids):
            raise ArtifactValidationError("publication partition must be nonempty and unique")
    root, manifest, manifest_digest = _open_public_envelope(path, game_ids)
    _validate_manifest_shape(root, manifest)
    listed_games = manifest.get("development_games")
    if not isinstance(listed_games, list) or len(listed_games) < FOLD_COUNT:
        raise ArtifactValidationError("Development Game Set is missing or too small")
    if game_ids is not None and not set(game_ids) <= {item["game_id"] for item in listed_games}:
        raise ArtifactValidationError("publication partition has unknown game")
    expected_files = {
        "public/development_fold_manifest.json",
        "restricted/role_sidecar.json",
    }
    expected_files.update(
        f"public/games/{item.get('game_id')}.json"
        for item in listed_games
        if isinstance(item, Mapping)
    )
    if set(manifest["file_table"]) != expected_files:
        raise ArtifactValidationError("Development Publication file partition mismatch")

    games: list[PublishedGame] = []
    for item in listed_games:
        if not isinstance(item, Mapping) or set(item) != {
            "ordinal",
            "seed",
            "attempt_id",
            "claim_record_digest",
            "terminal_record_digest",
            "game_id",
            "bundle_digest",
            "published_record_digest",
        }:
            raise ArtifactValidationError("Development Game Set entry is invalid")
        for field in ("ordinal", "seed"):
            if (
                isinstance(item[field], bool)
                or not isinstance(item[field], int)
                or item[field] < 0
            ):
                raise ArtifactValidationError(
                    f"Development Game Set {field} is invalid"
                )
        if (
            not isinstance(item["attempt_id"], str)
            or _SAFE_IDENTITY.fullmatch(item["attempt_id"]) is None
        ):
            raise ArtifactValidationError("Development Game Set attempt ID is invalid")
        _require_digest(item["claim_record_digest"], "claim_record_digest")
        _require_digest(item["terminal_record_digest"], "terminal_record_digest")
        _require_digest(item["published_record_digest"], "published_record_digest")
        _require_digest(item["bundle_digest"], "bundle_digest")
        if not isinstance(item["game_id"], str) or _SAFE_IDENTITY.fullmatch(item["game_id"]) is None:
            raise ArtifactValidationError("invalid game identity")
        if game_ids is not None and item["game_id"] not in game_ids:
            continue
        game = _published_game_from_record(
            _load_canonical_json(
                root / "public" / "games" / f"{item['game_id']}.json"
            )
        )
        if (
            game.game_id != item["game_id"]
            or game.bundle_digest != item["bundle_digest"]
            or game.record_digest != item["published_record_digest"]
        ):
            raise ArtifactValidationError("published game lineage mismatch")
        games.append(game)
    if len({item["game_id"] for item in listed_games}) != len(listed_games):
        raise ArtifactValidationError("Development Game Set has duplicate game IDs")
    game_pairs = tuple((item["game_id"], item["bundle_digest"]) for item in listed_games)
    game_set_digest = _development_game_set_digest(game_pairs)
    if game_set_digest != manifest["development_game_set_digest"]:
        raise ArtifactValidationError("Development Game Set digest mismatch")
    fold_manifest = _validate_fold_manifest_record(
        _load_canonical_json(
            root / "public" / "development_fold_manifest.json"
        ),
        game_pairs,
        game_set_digest,
    )
    if fold_manifest.manifest_digest != manifest["development_fold_manifest_digest"]:
        raise ArtifactValidationError("Development Fold Manifest digest mismatch")
    statistics = _publication_statistics(games)
    for field, value in statistics.items():
        if (game_ids is None and manifest.get(field) != value) or manifest.get(field, -1) < value:
            raise ArtifactValidationError(f"Development Publication {field} mismatch")
    if manifest.get("game_count") != len(listed_games):
        raise ArtifactValidationError("Development Publication game count mismatch")
    if manifest.get("canonical_success_count") != len(listed_games):
        raise ArtifactValidationError("Development Publication success count mismatch")
    for field in (
        "canonical_failure_count",
        "interrupted_failure_count",
        "attempt_count",
        "pre_boundary_count",
        "belief_observation_count",
        "public_event_count",
        "v1_annotation_count",
        "max_observed_day",
        "max_structured_token_count",
    ):
        if isinstance(manifest.get(field), bool) or not isinstance(
            manifest.get(field), int
        ) or manifest[field] < 0:
            raise ArtifactValidationError(f"Development Publication {field} is invalid")
    exclusion_records = manifest.get("complete_case_exclusions")
    if not isinstance(exclusion_records, list):
        raise ArtifactValidationError("complete-case exclusions must be a list")
    exclusions = tuple(_exclusion_from_record(item) for item in exclusion_records)
    if len(exclusions) != (
        manifest["canonical_failure_count"] + manifest["interrupted_failure_count"]
    ):
        raise ArtifactValidationError("complete-case exclusion count mismatch")
    if len({item.ordinal for item in exclusions}) != len(exclusions):
        raise ArtifactValidationError("complete-case exclusions contain duplicates")
    if sum(
        item.terminal_outcome == TerminalOutcome.CANONICAL_FAILURE.value
        for item in exclusions
    ) != manifest["canonical_failure_count"]:
        raise ArtifactValidationError("canonical failure exclusion count mismatch")
    if sum(
        item.terminal_outcome == TerminalOutcome.INTERRUPTED_FAILURE.value
        for item in exclusions
    ) != manifest["interrupted_failure_count"]:
        raise ArtifactValidationError("interrupted failure exclusion count mismatch")
    success_ordinals = [item["ordinal"] for item in listed_games]
    exclusion_ordinals = [item.ordinal for item in exclusions]
    if success_ordinals != sorted(success_ordinals):
        raise ArtifactValidationError(
            "Development Game Set is not in Attempt Ledger order"
        )
    all_ordinals = sorted(success_ordinals + exclusion_ordinals)
    if all_ordinals != list(range(manifest["attempt_count"])):
        raise ArtifactValidationError(
            "publication ledger attempt sequence is not closed"
        )
    if max(success_ordinals) != manifest["attempt_count"] - 1:
        raise ArtifactValidationError(
            "publication target was not reached on final attempt"
        )
    return VerifiedDevelopmentPublication(
        path=root,
        manifest=manifest,
        manifest_digest=manifest_digest,
        public_view=PublicationPublicView(
            publication_id=manifest["publication_id"],
            collection_id=manifest["collection_id"],
            collection_plan_digest=manifest["collection_plan_digest"],
            collection_identity_digest=manifest["collection_identity_digest"],
            development_game_set_digest=game_set_digest,
            games=tuple(games),
            fold_manifest=fold_manifest,
            complete_case_exclusions=exclusions,
            max_observed_day=manifest["max_observed_day"],
            max_structured_token_count=manifest["max_structured_token_count"],
            structured_token_planner_version=STRUCTURED_TOKEN_PLANNER_VERSION,
        ),
    )


def open_role_sidecar(
    publication: VerifiedDevelopmentPublication,
) -> VerifiedRoleSidecar:
    """Open the restricted role truth through its only publication boundary."""

    if not isinstance(publication, VerifiedDevelopmentPublication):
        raise TypeError("publication must be a VerifiedDevelopmentPublication")
    value = _load_canonical_json(
        publication.path / "restricted" / "role_sidecar.json"
    )
    required = {
        "schema_version",
        "publication_id",
        "collection_identity_digest",
        "development_game_set_digest",
        "games",
        "sidecar_digest",
    }
    if set(value) != required or value["schema_version"] != ROLE_SIDECAR_SCHEMA_VERSION:
        raise ArtifactValidationError("Role Sidecar schema or field set mismatch")
    without_digest = dict(value)
    sidecar_digest = without_digest.pop("sidecar_digest")
    _require_digest(sidecar_digest, "sidecar_digest")
    if sha256_bytes(canonical_json_bytes(without_digest)) != sidecar_digest:
        raise ArtifactValidationError("Role Sidecar digest mismatch")
    manifest = publication.manifest
    public = publication.public_view
    if (
        value["publication_id"] != public.publication_id
        or value["collection_identity_digest"] != public.collection_identity_digest
        or value["development_game_set_digest"] != public.development_game_set_digest
        or sidecar_digest != manifest["role_sidecar_digest"]
    ):
        raise ArtifactValidationError("Role Sidecar publication lineage mismatch")
    entry = manifest["file_table"]["restricted/role_sidecar.json"]
    data = (publication.path / "restricted" / "role_sidecar.json").read_bytes()
    if len(data) != entry["byte_size"] or sha256_bytes(data) != entry["sha256"]:
        raise ArtifactValidationError("Role Sidecar file digest mismatch")
    raw_games = value["games"]
    if not isinstance(raw_games, list):
        raise ArtifactValidationError("Role Sidecar games must be a list")
    # Role Sidecar covers the full publication even when the public reader is
    # partitioned. Validate its lineage against manifest metadata, never by
    # opening the other partitions' public records.
    listed_games = manifest["development_games"]
    if len(raw_games) != len(listed_games):
        raise ArtifactValidationError("Role Sidecar game coverage mismatch")
    games: list[RoleSidecarGame] = []
    for raw, listed_game in zip(raw_games, listed_games, strict=True):
        if not isinstance(raw, Mapping) or set(raw) != {
            "game_id",
            "bundle_digest",
            "role_assignment",
        }:
            raise ArtifactValidationError("Role Sidecar game entry is invalid")
        role_map = raw["role_assignment"]
        if not isinstance(role_map, Mapping) or set(role_map) != set(PLAYER_IDS):
            raise ArtifactValidationError(
                "Role Sidecar must cover exact Classic7 seats"
            )
        roles = _role_assignment(
            tuple((player, role_map[player]) for player in PLAYER_IDS)
        )
        game = RoleSidecarGame(
            game_id=raw["game_id"],
            bundle_digest=_require_digest(raw["bundle_digest"], "bundle_digest"),
            role_assignment=roles,
        )
        if (
            game.game_id != listed_game["game_id"]
            or game.bundle_digest != listed_game["bundle_digest"]
        ):
            raise ArtifactValidationError("Role Sidecar game lineage mismatch")
        games.append(game)
    return VerifiedRoleSidecar(
        path=publication.path / "restricted" / "role_sidecar.json",
        publication_id=public.publication_id,
        collection_identity_digest=public.collection_identity_digest,
        development_game_set_digest=public.development_game_set_digest,
        games=tuple(games),
        sidecar_digest=sidecar_digest,
    )


__all__ = [
    "DEVELOPMENT_FOLD_MANIFEST_SCHEMA_VERSION",
    "DEVELOPMENT_PUBLICATION_ARTIFACT_TYPE",
    "DEVELOPMENT_PUBLICATION_SCHEMA_VERSION",
    "DevelopmentFold",
    "DevelopmentFoldManifest",
    "DevelopmentPublicationHandles",
    "FOLD_ASSIGNMENT_VERSION",
    "FOLD_COUNT",
    "PUBLISHED_GAME_SCHEMA_VERSION",
    "PublicationPublicView",
    "PublishedGame",
    "ROLE_SIDECAR_SCHEMA_VERSION",
    "RankedFoldGame",
    "RoleSidecarGame",
    "STRUCTURED_TOKEN_PLANNER_VERSION",
    "VerifiedCollection",
    "VerifiedDevelopmentPublication",
    "VerifiedRoleSidecar",
    "assign_development_folds",
    "open_publication",
    "open_role_sidecar",
    "open_verified_collection",
    "publish_development",
]
