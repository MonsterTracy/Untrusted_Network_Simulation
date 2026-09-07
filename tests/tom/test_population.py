from pathlib import Path
from unittest.mock import patch

import torch

from tests.development_publication.test_development_publication import _publication


def test_population_boundary_and_all_alive_without_sidecar(tmp_path):
    from werewolf.tom.population import select_primary_population, build_all_alive_eligibility

    _, handles = _publication(tmp_path)
    primary = select_primary_population(handles.public)
    for game in handles.restricted.games:
        roles = dict(game.role_assignment)
        public_game = handles.public.public_view.load_game(game.game_id)
        for prefix in public_game.authoritative_pre_prefixes:
            mask = primary.mask(game.game_id, prefix.boundary_id)
            assert mask.tolist() == [f"player{i}" in prefix.alive_observer_ids and roles[f"player{i}"] != "Werewolf" for i in range(1, 8)]
    read_bytes = Path.read_bytes
    def public_only(path):
        assert "restricted" not in path.parts
        return read_bytes(path)
    with patch.object(Path, "read_bytes", public_only):
        stress = build_all_alive_eligibility(handles.public.public_view)
    assert "role_sidecar_digest" not in stress.metadata
    assert primary.metadata["role_sidecar_digest"] == handles.restricted.sidecar_digest
    assert all(isinstance(b, bool) for game in primary.rows.values() for row in game.values() for b in row)


def test_typed_population_validation_keeps_public_reads_partitioned_and_stress_private_blind(tmp_path, monkeypatch):
    from tests.tom.test_experiment import prepared_experiment
    from werewolf.tom.population import load_training_primary, load_held_out_all_alive
    experiment, handles = prepared_experiment(tmp_path)
    game = handles.public.public_view.load_game(experiment.fold(0)["training_game_ids"][0])
    read = Path.read_bytes
    def one_game(path):
        if path.parent.name == "games":
            assert path.stem == game.game_id
        return read(path)
    with monkeypatch.context() as guard:
        guard.setattr(Path, "read_bytes", one_game)
        mask = load_training_primary(experiment, 0, game)
    assert all(type(b) is bool for row in mask["rows"].values() for b in row)
    held_out = handles.public.public_view.load_game(experiment.fold(0)["held_out_game_ids"][0])
    def no_private(path):
        assert "restricted" not in path.parts
        return read(path)
    with monkeypatch.context() as guard:
        guard.setattr(Path, "read_bytes", no_private)
        stress = load_held_out_all_alive(experiment, 0, held_out)
    assert "role_sidecar_digest" not in stress
