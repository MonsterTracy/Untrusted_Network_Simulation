import ast
import inspect
from pathlib import Path

import pytest
import torch

from tests.tom.test_experiment import prepared_experiment, experiment_config


def test_no_hidden_protocol_fields_or_incompatible_parent(tmp_path):
    from dataclasses import asdict
    from werewolf.artifact_io import publish_artifact
    from werewolf.tom.experiment import ExperimentConfig, open_experiment
    experiment, _ = prepared_experiment(tmp_path)
    config = asdict(experiment_config(10))
    del config["rotation_cycles"]
    with pytest.raises(TypeError):
        ExperimentConfig(**config)
    fields = {k: v for k, v in experiment.manifest.items() if k not in {"manifest_digest", "file_table"}}
    files = {p: experiment.file(p) for p in experiment.manifest["file_table"]}
    fields["publication_digest"] = "0" * 64
    publish_artifact(tmp_path / "wrong-parent", manifest_name="experiment_manifest.json", manifest_fields=fields, files=files)
    with pytest.raises(ValueError, match="parent"):
        open_experiment(tmp_path / "wrong-parent")


def test_off_diagonal_readout_can_memorize_controlled_public_fixture(tmp_path):
    from werewolf.tom.dataset import CanonicalToMDataset, PublicTensors, ExperimentCapacity
    from werewolf.tom.training import build_model
    from werewolf.tom.scoring import game_balanced_cross_entropy
    experiment, handles = prepared_experiment(tmp_path)
    torch.manual_seed(1234)
    model = build_model(experiment, "implicit")
    sample = CanonicalToMDataset(handles.public.public_view, handles.public.public_view.game_ids[:1], ExperimentCapacity(experiment.config.max_seq_len))[0]
    public = PublicTensors.stack([sample.public]).kwargs()
    targets = torch.zeros(1, 7, 7)
    for observer in range(7):
        targets[0, observer, (observer + 1) % 7] = 1
    mask = torch.ones(1, 7, dtype=torch.bool)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003)
    model.eval()  # test assertion, not a scientific training lineage
    initial = game_balanced_cross_entropy([(model(**public), targets, mask)]).item()
    for _ in range(150):
        optimizer.zero_grad()
        loss = game_balanced_cross_entropy([(model(**public), targets, mask)])
        loss.backward()
        optimizer.step()
    assert loss.item() < initial / 4
    bad = dict(public, day_ids=public["day_ids"][:, :1])
    with pytest.raises(ValueError, match="shape"):
        model(**bad)


def test_forbidden_paths_and_semantic_dependency_boundaries():
    import werewolf.tom.dataset as dataset
    import werewolf.tom.model as model
    import werewolf.tom.training as training
    import run_random
    root = Path(__file__).resolve().parents[2]
    assert not (root / "werewolf/models/twd_tom/dataset.py").exists()
    assert not list((root / "script").rglob("*.py"))
    assert not hasattr(run_random, "main_cli")
    assert inspect.signature(run_random.eval).parameters["canonical_recorder"].default is inspect.Parameter.empty
    for module in (dataset, model):
        text = inspect.getsource(module)
        assert not any(word in text for word in ("RoleSidecar", "open_role_sidecar", "non_wolf_alive", "all_alive", "private_conditioning"))
    imports = [n for n in ast.walk(ast.parse(inspect.getsource(training))) if isinstance(n, ast.ImportFrom)]
    assert not any(a.name in {"load_held_out_primary", "load_held_out_all_alive", "open_role_sidecar"} for n in imports for a in n.names)


def _production_import_closure(root):
    # Phase-1 used only uns; Phase-2 runbooks also use executable scripts.
    # Helpers are transit nodes, never roots or an exclusion allowlist.
    paths = (list((root / "werewolf").rglob("*.py")) + [root / "run_random.py"]
             + list((root / "scripts").rglob("*.py")))
    modules = {".".join(p.relative_to(root).with_suffix("").parts).removesuffix(".__init__"): p for p in paths}
    production = {name for name in modules if name == "run_random" or name == "werewolf"
                  or name.startswith("werewolf.")}
    edges, entrypoints = {}, {"werewolf.cli"}
    main_guard = ast.dump(ast.parse("__name__ == '__main__'", mode="eval").body)
    for name, path in modules.items():
        tree = ast.parse(path.read_text())
        if name.startswith("scripts.") and any(
                isinstance(node, ast.If) and ast.dump(node.test) == main_guard
                for node in tree.body):
            entrypoints.add(name)
        dependencies = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                dependencies.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    package = name if path.name == "__init__.py" else name.rpartition(".")[0]
                    prefix = package.split(".")[:len(package.split(".")) - node.level + 1]
                    base = ".".join(prefix + ([base] if base else []))
                dependencies.add(base)
                dependencies.update(f"{base}.{a.name}" for a in node.names)
        expanded = {".".join(d.split(".")[:i]) for d in dependencies for i in range(1, len(d.split(".")) + 1)}
        edges[name] = expanded & modules.keys()
    reached, pending = set(), list(entrypoints)
    while pending:
        name = pending.pop()
        if name not in reached:
            reached.add(name)
            pending.extend(edges[name] - reached)
    return production, reached, entrypoints


def test_all_production_modules_belong_to_current_cli_import_closure():
    """No production orphans across uns and standalone executable script roots."""
    root = Path(__file__).resolve().parents[2]
    production, reached, _ = _production_import_closure(root)
    assert production - reached == set()


@pytest.mark.parametrize("script", (
    "from werewolf import analysis\n",
    "def helper():\n    if __name__ == '__main__':\n        from werewolf import analysis\n",
    "if __name__ != '__main__':\n    from werewolf import analysis\n",
))
def test_non_entrypoint_script_does_not_hide_production_orphan(tmp_path, script):
    (tmp_path / "werewolf").mkdir()
    (tmp_path / "scripts").mkdir()
    for name in ("werewolf/__init__.py", "werewolf/cli.py", "werewolf/analysis.py", "run_random.py"):
        (tmp_path / name).write_text("", encoding="utf-8")
    (tmp_path / "scripts/helper.py").write_text(script, encoding="utf-8")
    production, reached, entrypoints = _production_import_closure(tmp_path)
    assert entrypoints == {"werewolf.cli"}
    assert "werewolf.analysis" in production - reached


def test_executable_script_reaches_analysis_without_exempting_orphans(tmp_path):
    (tmp_path / "werewolf").mkdir()
    (tmp_path / "scripts").mkdir()
    for name in ("werewolf/__init__.py", "werewolf/cli.py", "werewolf/analysis.py", "run_random.py"):
        (tmp_path / name).write_text("", encoding="utf-8")
    (tmp_path / "scripts/analysis.py").write_text(
        "from werewolf import analysis\ndef main():\n    pass\n"
        "if __name__ == '__main__':\n    main()\n", encoding="utf-8")
    production, reached, entrypoints = _production_import_closure(tmp_path)
    assert "scripts.analysis" in entrypoints and "werewolf.analysis" in reached
    assert production - reached == {"run_random"}


def test_terminal_estimator_is_reachable_from_its_real_command_entrypoint():
    root = Path(__file__).resolve().parents[2]
    production, reached, entrypoints = _production_import_closure(root)
    assert "scripts.phase2_terminal_estimator" in entrypoints
    estimator_modules = {"werewolf.phase2_terminal_estimator", "werewolf.phase2_terminal_estimator_io"}
    assert estimator_modules <= production & reached


@pytest.mark.parametrize("field,value", [("backend", {}), ("pilot", True), ("scope", "all_alive"), ("shadow", {})])
def test_runtime_rejects_obsolete_configuration(field, value):
    from tests.runtime.test_runtime_config import new_config
    from werewolf.runtime_config import normalize_runtime_config
    config = new_config()
    config[field] = value
    with pytest.raises(ValueError, match="unsupported"):
        normalize_runtime_config(config)
