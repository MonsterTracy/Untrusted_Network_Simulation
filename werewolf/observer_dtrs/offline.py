"""Controlled offline publication reader; contains labels and restricted lineage."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from werewolf.artifact_io import ArtifactValidationError
from werewolf.development_publication import open_publication, open_role_sidecar
from werewolf.observer_dtrs.context import ObserverContext, contexts_for_pre
from werewolf.observer_dtrs.labels import ObserverLabel, label_from_observation


SAMPLE_VERSION = "observer_dtrs_offline_sample_v1"


@dataclass(frozen=True)
class SampleSource:
    publication_id: str
    publication_digest: str
    collection_identity_digest: str
    development_game_set_digest: str
    role_sidecar_digest: str
    fold_manifest_digest: str
    bundle_digest: str
    published_game_record_digest: str
    fold_index: int


@dataclass(frozen=True)
class ObserverSample:
    """Offline envelope. Only .context is suitable for future legal features."""

    context: ObserverContext
    supervision: ObserverLabel
    source: SampleSource

    @property
    def identity(self) -> tuple[str, str, str, str]:
        c = self.context
        return c.game_id, c.boundary_id, c.acting_wolf, c.observer_id

    def to_record(self) -> dict:
        return {"schema_version": SAMPLE_VERSION, "context": self.context.to_record(),
                "supervision": self.supervision.to_record(), "source": asdict(self.source)}


def build_observer_samples(publication_path: Path | str, *,
                           game_ids=None) -> tuple[ObserverSample, ...]:
    """Read verified publication/sidecar and emit every eligible observer row.

    game_ids, when supplied, is an explicit whole-game reader partition. Every
    selected game's canonical reports are validated before any rows are built,
    including reports outside the eventual wolf/non-wolf sample population.
    """
    publication = open_publication(publication_path, game_ids=game_ids)
    sidecar = open_role_sidecar(publication)
    view = publication.public_view
    roles = {(g.game_id, g.bundle_digest): g.role_assignment for g in sidecar.games}
    folds = {g: fold.fold_index for fold in view.fold_manifest.folds for g in fold.game_ids}
    samples = []
    for game in view.games:
        # The existing sidecar reader verifies publication, exact seats, role
        # multiplicities, game coverage and (game_id, bundle_digest) binding.
        known_wolves = tuple(p for p, role in roles[game.game_id, game.bundle_digest]
                             if role == "Werewolf")
        source = SampleSource(view.publication_id, publication.manifest_digest,
            view.collection_identity_digest, view.development_game_set_digest,
            sidecar.sidecar_digest, view.fold_manifest.manifest_digest,
            game.bundle_digest, game.record_digest, folds[game.game_id])
        observations = {o.observation_id: o for o in game.belief_observations}
        for prefix in game.authoritative_pre_prefixes:
            links = {link.observer_id: link.observation_id for link in prefix.belief_observation_links}
            for context in contexts_for_pre(prefix, known_wolves):
                observation = observations.get(links[context.observer_id])
                label = label_from_observation(context, observation)
                if not label.label_observed:
                    raise ArtifactValidationError("formal publication has missing observer supervision")
                samples.append(ObserverSample(context, label, source))
    return tuple(samples)
