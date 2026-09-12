"""Independent, plan-closed Final Evaluation Publication; no fold assignment."""

from dataclasses import dataclass
from pathlib import Path

from werewolf.artifact_io import canonical_json_bytes, publish_artifact, verify_artifact, sha256_bytes
from werewolf.development_publication import (PublicGameView, PublishedGame, VerifiedCollection,
    _validate_verified_collection, _published_game_from_bundle, _published_game_from_record,
    _publication_statistics, _load_canonical_json, _role_assignment, _SAFE_IDENTITY, _require_digest)
from werewolf.canonical_collection.attempt_ledger import TerminalOutcome
from werewolf.structured_history import STRUCTURED_TOKEN_PLANNER_VERSION

FINAL_PUBLICATION_VERSION = "classic7_final_evaluation_publication_v1"


@dataclass(frozen=True)
class FinalPublicView(PublicGameView):
    publication_id: str
    games: tuple[PublishedGame, ...]
    max_observed_day: int
    max_structured_token_count: int
    structured_token_planner_version: str


@dataclass(frozen=True)
class VerifiedFinalPublication:
    path: Path
    manifest: dict
    manifest_digest: str
    public_view: FinalPublicView


def publish_final_publication(collection, destination, *, publication_id, model_seal_digest):
    """The caller verifies the model seal before opening the source collection."""
    if not isinstance(collection, VerifiedCollection):
        raise TypeError("final publication requires a verified collection")
    _validate_verified_collection(collection)
    _require_digest(model_seal_digest, "model_seal_digest")
    if not _SAFE_IDENTITY.fullmatch(publication_id) or Path(destination).name != publication_id:
        raise ValueError("invalid final publication identity")
    if len(collection.games) != collection.plan.target_canonical_success_count or not collection.games:
        raise ValueError("final collection must be plan-closed")
    games = tuple(_published_game_from_bundle(g) for g in collection.games)
    successes = [t for t in collection.ledger_state.terminals if t.outcome is TerminalOutcome.CANONICAL_SUCCESS]
    entries = [{"game_id": g.game_id, "bundle_digest": g.bundle_digest, "record_digest": g.record_digest,
                "seed": t.seed, "ordinal": t.ordinal, "terminal_record_digest": t.record_digest}
               for g, t in zip(games, successes, strict=True)]
    roles = [{"game_id": g.game_id, "bundle_digest": g.manifest_digest,
              "role_assignment": dict(_role_assignment(g.private_replay_evidence.role_assignment))}
             for g in collection.games]
    sidecar = {"schema_version": "classic7_final_role_sidecar_v1", "publication_id": publication_id,
               "collection_identity_digest": collection.collection_identity_digest, "games": roles}
    files = {f"public/games/{g.game_id}.json": canonical_json_bytes(g.to_record()) for g in games}
    files["restricted/role_sidecar.json"] = canonical_json_bytes(sidecar)
    files["collection_plan.json"] = canonical_json_bytes(collection.plan.to_record())
    files["attempt_ledger.json"] = canonical_json_bytes({
        "claims": [r.to_record() for r in collection.ledger_state.claims],
        "terminals": [r.to_record() for r in collection.ledger_state.terminals]})
    publish_artifact(destination, manifest_fields={
        "artifact_type": "final_evaluation_publication", "schema_version": FINAL_PUBLICATION_VERSION,
        "publication_id": publication_id, "model_seal_digest": model_seal_digest,
        "collection_id": collection.plan.collection_id, "collection_plan_digest": collection.plan.plan_digest,
        "collection_identity_digest": collection.collection_identity_digest, "games": entries,
        "exclusions": [e.to_record() for e in collection.exclusions],
        "role_sidecar_digest": sha256_bytes(files["restricted/role_sidecar.json"]),
        "structured_token_planner_version": STRUCTURED_TOKEN_PLANNER_VERSION,
        "statistics": _publication_statistics(games)}, files=files)
    return open_final_publication(destination, model_seal_digest=model_seal_digest)


def open_final_publication(path, *, model_seal_digest):
    _require_digest(model_seal_digest, "model_seal_digest")
    artifact = verify_artifact(path, expected_artifact_type="final_evaluation_publication",
                              expected_schema_version=FINAL_PUBLICATION_VERSION)
    m = artifact.manifest
    expected = {"artifact_type", "schema_version", "publication_id", "model_seal_digest", "collection_id",
        "collection_plan_digest", "collection_identity_digest", "games", "exclusions", "role_sidecar_digest",
        "structured_token_planner_version", "statistics", "file_table", "manifest_digest"}
    if set(m) != expected or m["model_seal_digest"] != model_seal_digest:
        raise ValueError("final publication seal/schema mismatch")
    if (not _SAFE_IDENTITY.fullmatch(m["publication_id"]) or artifact.path.name != m["publication_id"]
        or m["structured_token_planner_version"] != STRUCTURED_TOKEN_PLANNER_VERSION):
        raise ValueError("final publication identity/planner mismatch")
    entries = m["games"]
    ids = [r["game_id"] for r in entries]
    if not ids or len(set(ids)) != len(ids) or any(not _SAFE_IDENTITY.fullmatch(g) for g in ids):
        raise ValueError("invalid final game coverage")
    expected_files = {"restricted/role_sidecar.json", "collection_plan.json", "attempt_ledger.json"}
    expected_files.update(f"public/games/{g}.json" for g in ids)
    if set(m["file_table"]) != expected_files:
        raise ValueError("final publication file inventory mismatch")
    from werewolf.canonical_collection.attempt_ledger import (
        collection_plan_from_record, _claim_from_record, _terminal_from_record, _state_from_records)
    from werewolf.development_publication import _collection_identity, _exclusion_from_record
    plan = collection_plan_from_record(_load_canonical_json(artifact.path / "collection_plan.json"))
    ledger = _load_canonical_json(artifact.path / "attempt_ledger.json")
    if set(ledger) != {"claims", "terminals"}:
        raise ValueError("invalid final ledger fields")
    claims = tuple(_claim_from_record(r, plan) for r in ledger["claims"])
    terminals = tuple(_terminal_from_record(r, plan) for r in ledger["terminals"])
    state = _state_from_records(plan, claims, terminals)
    if state.open_claim is not None or not state.target_reached or claims != state.claims or terminals != state.terminals:
        raise ValueError("final publication ledger must be closed and ordered")
    if (plan.plan_digest != m["collection_plan_digest"] or plan.collection_id != m["collection_id"]
        or _collection_identity(plan, state) != m["collection_identity_digest"]):
        raise ValueError("final collection lineage mismatch")
    successes = [t for t in terminals if t.outcome is TerminalOutcome.CANONICAL_SUCCESS]
    if len(successes) != plan.target_canonical_success_count or len(entries) != len(successes) or successes[-1] != terminals[-1]:
        raise ValueError("final publication is not plan-closed")
    exclusions = [_exclusion_from_record(e) for e in m["exclusions"]]
    failures = [t for t in terminals if t.outcome is not TerminalOutcome.CANONICAL_SUCCESS]
    if [e.ordinal for e in exclusions] != [t.ordinal for t in failures]:
        raise ValueError("final exclusions coverage mismatch")
    for exclusion, terminal in zip(exclusions, failures, strict=True):
        if (exclusion.terminal_record_digest != terminal.record_digest or exclusion.seed != terminal.seed
            or exclusion.attempt_id != terminal.attempt_id or exclusion.terminal_outcome != terminal.outcome.value
            or exclusion.failure_evidence_digest != terminal.failure_evidence_digest
            or exclusion.claim_record_digest != claims[terminal.ordinal].record_digest):
            raise ValueError("final exclusion lineage mismatch")
    games = []
    for item, terminal in zip(entries, successes, strict=True):
        game = _published_game_from_record(_load_canonical_json(artifact.path / f"public/games/{item['game_id']}.json"))
        if item != {"game_id": game.game_id, "bundle_digest": game.bundle_digest, "record_digest": game.record_digest,
                    "seed": terminal.seed, "ordinal": terminal.ordinal, "terminal_record_digest": terminal.record_digest}:
            raise ValueError("final published game identity mismatch")
        if terminal.canonical_game_bundle_id != game.game_id or terminal.canonical_game_bundle_digest != game.bundle_digest:
            raise ValueError("final bundle/terminal mismatch")
        games.append(game)
    if _publication_statistics(games) != m["statistics"]:
        raise ValueError("final publication statistics mismatch")
    return VerifiedFinalPublication(artifact.path, m, artifact.manifest_digest,
        FinalPublicView(m["publication_id"], tuple(games), m["statistics"]["max_observed_day"],
                        m["statistics"]["max_structured_token_count"], STRUCTURED_TOKEN_PLANNER_VERSION))


def final_role_assignments(publication):
    """Restricted reader; called only by Population Selection."""
    m = publication.manifest
    path = publication.path / "restricted/role_sidecar.json"
    if sha256_bytes(path.read_bytes()) != m["role_sidecar_digest"]:
        raise ValueError("final sidecar digest mismatch")
    sidecar = _load_canonical_json(path)
    if set(sidecar) != {"schema_version", "publication_id", "collection_identity_digest", "games"} or any(
        sidecar[k] != m[k] for k in ("publication_id", "collection_identity_digest")) or sidecar["schema_version"] != "classic7_final_role_sidecar_v1":
        raise ValueError("final sidecar parent mismatch")
    if len(sidecar["games"]) != len(m["games"]):
        raise ValueError("final sidecar coverage mismatch")
    assignments = {}
    for row, game in zip(sidecar["games"], m["games"], strict=True):
        if set(row) != {"game_id", "bundle_digest", "role_assignment"} or any(row[k] != game[k] for k in ("game_id", "bundle_digest")):
            raise ValueError("final sidecar game mismatch")
        assignments[row["game_id"]] = dict(_role_assignment(tuple(row["role_assignment"].items())))
    return assignments
