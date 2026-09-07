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


def test_all_production_modules_belong_to_current_cli_import_closure():
    root = Path(__file__).resolve().parents[2]
    paths = list((root / "werewolf").rglob("*.py")) + [root / "run_random.py"]
    modules = {".".join(p.relative_to(root).with_suffix("").parts).removesuffix(".__init__"): p for p in paths}
    edges = {}
    for name, path in modules.items():
        dependencies = set()
        for node in ast.walk(ast.parse(path.read_text())):
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
    reached, pending = set(), ["werewolf.cli"]
    while pending:
        name = pending.pop()
        if name not in reached:
            reached.add(name)
            pending.extend(edges[name] - reached)
    assert modules.keys() - reached == set()


@pytest.mark.parametrize("field,value", [("backend", {}), ("pilot", True), ("scope", "all_alive"), ("shadow", {})])
def test_runtime_rejects_obsolete_configuration(field, value):
    from tests.runtime.test_runtime_config import new_config
    from werewolf.runtime_config import normalize_runtime_config
    config = new_config()
    config[field] = value
    with pytest.raises(ValueError, match="unsupported"):
        normalize_runtime_config(config)
