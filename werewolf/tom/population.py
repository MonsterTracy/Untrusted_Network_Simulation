"""Role truth terminates here; downstream supervision sees only booleans."""

from dataclasses import dataclass
import json

from werewolf.artifact_io import canonical_json_bytes, sha256_bytes

import torch

from werewolf.development_publication import open_publication, open_role_sidecar
from werewolf.canonical_collection.public_history import PLAYER_IDS

PRIMARY_SELECTOR_VERSION = "classic7_primary_population_v1"
ALL_ALIVE_VERSION = "classic7_all_alive_eligibility_v1"

def _decode_eligibility(experiment, game, payload, *, primary):
    value = json.loads(payload)
    if canonical_json_bytes(value) != payload:
        raise ValueError("noncanonical eligibility bytes")
    expected_fields = {"schema_version", "artifact_type", "population_identity", "population_selector_version",
        "publication_id", "parent_digest", "game_id", "bundle_digest", "row_identity", "row_identity_digest", "rows"}
    if primary:
        expected_fields.add("role_sidecar_digest")
    identity = [{"boundary_id": p.boundary_id, "prefix_digest": p.prefix_digest, "observer_ids": list(range(7))}
                for p in game.authoritative_pre_prefixes]
    if (set(value) != expected_fields or value["schema_version"] != "classic7_eligibility_v1"
        or value["parent_digest"] != experiment.manifest["protocol_digest"] or value["bundle_digest"] != game.bundle_digest
        or value["publication_id"] != experiment.manifest["publication_id"]
        or value["game_id"] != game.game_id or value["row_identity"] != identity
        or value["row_identity_digest"] != sha256_bytes(canonical_json_bytes(identity))):
        raise ValueError("eligibility identity/schema mismatch")
    expected = ("primary_observer_eligibility", "non_wolf_alive", PRIMARY_SELECTOR_VERSION) if primary else (
        "all_alive_observer_eligibility", "all_alive", ALL_ALIVE_VERSION)
    if (value["artifact_type"], value["population_identity"], value["population_selector_version"]) != expected:
        raise ValueError("eligibility operation mismatch")
    if primary and value["role_sidecar_digest"] != experiment.manifest["primary_sidecar_digest"]:
        raise ValueError("Primary provenance mismatch")
    if set(value["rows"]) != {p.boundary_id for p in game.authoritative_pre_prefixes}:
        raise ValueError("eligibility row coverage mismatch")
    for row in value["rows"].values():
        if len(row) != 7 or any(type(v) is not bool for v in row):
            raise ValueError("eligibility must contain boolean rows only")
    for prefix in game.authoritative_pre_prefixes:
        alive = [p in prefix.alive_observer_ids for p in PLAYER_IDS]
        row = value["rows"][prefix.boundary_id]
        if any(eligible and not living for eligible, living in zip(row, alive)) or (not primary and row != alive):
            raise ValueError("eligibility public alive-state mismatch")
    if primary:
        # Only this typed population boundary may reopen role truth. Limit
        # public reads to this game, including when called by a training worker.
        publication = open_publication(experiment.manifest["publication_path"], game_ids=[game.game_id])
        if publication.manifest_digest != experiment.manifest["publication_digest"]:
            raise ValueError("Primary publication provenance mismatch")
        expected_primary = select_primary_population(publication)
        if (expected_primary.metadata["role_sidecar_digest"] != value["role_sidecar_digest"]
            or value["rows"] != expected_primary.rows[game.game_id]):
            raise ValueError("Primary membership does not match Population Selector")
    return value


def load_training_primary(experiment, fold, game):
    return _decode_eligibility(experiment, game,
        experiment.file(f"folds/{fold}/population/training_primary/{game.game_id}.json"), primary=True)


def load_held_out_primary(experiment, fold, game):
    return _decode_eligibility(experiment, game,
        experiment.file(f"folds/{fold}/population/held_out_primary/{game.game_id}.json"), primary=True)


def load_held_out_all_alive(experiment, fold, game):
    return _decode_eligibility(experiment, game,
        experiment.file(f"folds/{fold}/population/held_out_all_alive/{game.game_id}.json"), primary=False)


@dataclass(frozen=True)
class ObserverEligibility:
    rows: dict
    metadata: dict

    def mask(self, game_id, boundary_id):
        values = self.rows[game_id][boundary_id]
        if len(values) != 7 or any(type(v) is not bool for v in values):
            raise ValueError("eligibility must contain seven booleans")
        return torch.tensor(values, dtype=torch.bool)


def select_primary_population(publication):
    sidecar = open_role_sidecar(publication)
    assignments = {g.game_id: dict(g.role_assignment) for g in sidecar.games}
    return _primary_membership(publication.public_view, assignments, sidecar.sidecar_digest)


def _primary_membership(public_view, assignments, sidecar_digest):
    rows = {}
    for game in public_view.games:
        rows[game.game_id] = {
            p.boundary_id: [seat in p.alive_observer_ids and assignments[game.game_id][seat] != "Werewolf" for seat in PLAYER_IDS]
            for p in game.authoritative_pre_prefixes}
    return ObserverEligibility(rows, {
        "artifact_type": "primary_observer_eligibility",
        "population_identity": "non_wolf_alive",
        "population_selector_version": PRIMARY_SELECTOR_VERSION,
        "publication_id": public_view.publication_id,
        "role_sidecar_digest": sidecar_digest})


def build_all_alive_eligibility(public_view):
    rows = {
        game.game_id: {p.boundary_id: [seat in p.alive_observer_ids for seat in PLAYER_IDS]
                       for p in game.authoritative_pre_prefixes}
        for game in public_view.games}
    return ObserverEligibility(rows, {
        "artifact_type": "all_alive_observer_eligibility",
        "population_identity": "all_alive",
        "population_selector_version": ALL_ALIVE_VERSION,
        "publication_id": public_view.publication_id})


def select_final_primary_population(publication):
    from werewolf.final_publication import final_role_assignments
    return _primary_membership(publication.public_view, final_role_assignments(publication),
                               publication.manifest["role_sidecar_digest"])
