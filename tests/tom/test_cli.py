import pytest


def test_five_scientific_commands_and_isolated_capacity_check():
    from werewolf.cli import build_parser
    parser = build_parser()
    commands = parser._subparsers._group_actions[0].choices
    assert set(commands) == {"collect", "publish-development", "prepare-experiment", "run-development-oof", "validate-artifact", "capacity-check"}
    for command in commands.values():
        options = {option for action in command._actions for option in action.option_strings}
        assert not options & {"--scope", "--population", "--pilot", "--private", "--resume-step", "--test", "--best", "--backbone"}
    with pytest.raises(SystemExit):
        parser.parse_args(["run-development-oof", "--scope", "all_alive"])


@pytest.mark.parametrize("extra", ["extra.json", "restricted/extra.json"])
def test_publication_validation_rejects_unlisted_files_without_private_public_reads(tmp_path, monkeypatch, extra):
    from pathlib import Path
    from werewolf.cli import validate_artifact
    from werewolf.development_publication import open_publication
    from tests.development_publication.test_development_publication import _publication

    _, handles = _publication(tmp_path)
    assert validate_artifact(handles.public.path) == handles.public.manifest_digest
    read = Path.read_bytes
    def public_only(path):
        assert "restricted" not in path.parts
        return read(path)
    with monkeypatch.context() as guard:
        guard.setattr(Path, "read_bytes", public_only)
        assert open_publication(handles.public.path).manifest_digest == handles.public.manifest_digest
    (handles.public.path / extra).write_bytes(b"{}")
    with pytest.raises(ValueError, match="file|inventory"):
        validate_artifact(handles.public.path)


def test_standalone_model_state_validation_requires_fixed_qwen2_graph(tmp_path):
    import torch
    from werewolf.cli import validate_artifact
    from werewolf.tom.state import publish_model_state
    from tests.tom.test_experiment import prepared_experiment

    experiment, _ = prepared_experiment(tmp_path)
    initial = experiment.path / "folds" / "0" / "initial_state"
    assert validate_artifact(initial) == experiment.fold(0)["paired_initial_state_digest"]
    wrong = publish_model_state(tmp_path / "non-qwen", torch.nn.Linear(2, 2), {"purpose": "initial"})
    with pytest.raises(ValueError, match="model state key"):
        validate_artifact(wrong.artifact.path)
