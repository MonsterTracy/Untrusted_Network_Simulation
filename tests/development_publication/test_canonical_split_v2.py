"""Independent synthetic collections only: no network, language model or training."""

from dataclasses import replace
import json
from pathlib import Path

import pytest

from tests.canonical_collection.test_game_bundle import _fixture, _plan
from tests.development_publication.test_development_publication import _republish, _publication
from werewolf.artifact_io import ArtifactConflictError, ArtifactValidationError, canonical_json_bytes
from werewolf.canonical_collection import (
    construct_attempt_claim, construct_attempt_terminal, initialize_attempt_ledger,
    publish_ledger_record, publish_canonical_game_bundle, TerminalOutcome,
)
from werewolf.canonical_split import (
    PROTOCOL_BINDING, publish_split_protocol, publish_canonical_split,
    open_canonical_split, open_split_protocol,
)
from werewolf.development_publication import open_verified_collection, open_publication, publish_development, open_role_sidecar
from werewolf.development_publication_v2 import publish_development_v2, open_development_publication_v2


def _collection(root, *, target=10, completed=10, failure=False, seed=20261012, test_count=2):
    protocol = publish_split_protocol(root / "protocol", protocol_id="protocol", collection_id="synthetic-split",
                                     development_count=target - test_count, test_count=test_count, split_seed=seed)
    plan = _plan(collection_id="synthetic-split", ordered_seed_pool=tuple(range(100, 100 + target + 2)),
                 target_canonical_success_count=target,
                 environment_provenance={"fixture": "split-v2", PROTOCOL_BINDING: protocol.digest})
    collection_path = root / "collection"
    ledger = initialize_attempt_ledger(collection_path, plan)
    executor = None
    for ordinal in range(completed + int(failure)):
        claim = construct_attempt_claim(plan, ordinal=ordinal, attempt_id=f"attempt-{ordinal:03d}",
                                        claim_timestamp_utc=f"2026-10-12T00:00:{ordinal:02d}Z")
        publish_ledger_record(ledger, plan, claim)
        if failure and ordinal == 0:
            terminal = construct_attempt_terminal(plan, claim, outcome=TerminalOutcome.INTERRUPTED_FAILURE,
                                                  terminal_timestamp_utc="2026-10-12T00:01:00Z")
        else:
            game_id = f"game-{ordinal:03d}"
            _, _, evidence, executor = _fixture(plan, claim, game_id=game_id)
            bundle = publish_canonical_game_bundle(collection_path / "games" / game_id,
                plan=plan, claim=claim, evidence=evidence, replay_executor=executor)
            terminal = construct_attempt_terminal(plan, claim, outcome=TerminalOutcome.CANONICAL_SUCCESS,
                canonical_game_bundle_id=game_id, canonical_game_bundle_digest=bundle.manifest_digest,
                terminal_timestamp_utc=f"2026-10-12T00:01:{ordinal:02d}Z")
        publish_ledger_record(ledger, plan, terminal)
    return protocol, collection_path, executor


@pytest.fixture
def completed(tmp_path):
    protocol, path, executor = _collection(tmp_path, failure=True)
    collection = open_verified_collection(path, replay_executor=executor)
    split = publish_canonical_split(collection, protocol, tmp_path / "split", split_id="split")
    return collection, protocol, split, executor


def test_split_exact_disjoint_cover_and_repeat(completed):
    collection, protocol, split, _ = completed
    dev = {g["game_id"] for g in split.members("development")}
    test = {g["game_id"] for g in split.members("independent_test")}
    assert len(dev) == 8 and len(test) == 2
    assert not dev & test and dev | test == {g.game_id for g in collection.games}
    assert len(split.ledger_state.claims) == 11
    assert publish_canonical_split(collection, protocol, split.artifact.path, split_id="split").digest == split.digest
    assert open_canonical_split(split.artifact.path, expected_digest=split.digest).manifest == split.manifest


def test_filesystem_order_does_not_affect_split(completed, monkeypatch):
    collection, protocol, split, executor = completed
    original = Path.iterdir
    monkeypatch.setattr(Path, "iterdir", lambda path: iter(reversed(list(original(path)))))
    reopened = open_verified_collection(collection.path, replay_executor=executor)
    assert publish_canonical_split(reopened, protocol, split.artifact.path, split_id="split").digest == split.digest
    assert open_canonical_split(split.artifact.path, expected_digest=split.digest).manifest == split.manifest


def test_metadata_order_independence_of_assignment(completed):
    from werewolf.canonical_split import _split_fields
    collection, protocol, split, _ = completed
    reverse = replace(collection.ledger_state, terminals=tuple(reversed(collection.ledger_state.terminals)))
    a = _split_fields("split", protocol.manifest, collection.plan, collection.ledger_state)
    b = _split_fields("split", protocol.manifest, collection.plan, reverse)
    assert a[1] == b[1]
    # Changing content-derived bundle provenance cannot affect ranking or membership.
    altered = replace(collection.ledger_state, terminals=tuple(
        replace(t, canonical_game_bundle_digest="0" * 64) if t.outcome is TerminalOutcome.CANONICAL_SUCCESS else t
        for t in collection.ledger_state.terminals))
    changed = _split_fields("split", protocol.manifest, collection.plan, altered)
    assert [(g["game_id"], g["ranking_digest"], g["partition"]) for g in a[1]] == [
        (g["game_id"], g["ranking_digest"], g["partition"]) for g in changed[1]]
    with pytest.raises(ArtifactValidationError, match="substituted"):
        publish_canonical_split(replace(collection, games=tuple(reversed(collection.games))), protocol,
                                split.artifact.path, split_id="split")


@pytest.mark.parametrize("mutation", ["membership", "bundle_digest", "collection_identity", "protocol"])
def test_rehashed_json_cannot_change_membership_or_bindings(completed, tmp_path, mutation):
    _, _, split, _ = completed
    updates = {}
    files = {}
    if mutation in {"membership", "bundle_digest"}:
        rows = json.loads(json.dumps(split.ranked_games))
        if mutation == "membership":
            rows[0]["partition"], rows[-1]["partition"] = rows[-1]["partition"], rows[0]["partition"]
        else:
            rows[0]["bundle_digest"] = "0" * 64
        files["split_membership.json"] = canonical_json_bytes({"ranked_games": rows})
    elif mutation == "collection_identity":
        updates["collection_identity_digest"] = "0" * 64
    else:
        protocol = json.loads((split.artifact.path / "split_protocol.json").read_bytes())
        protocol["split_seed"] += 1
        from werewolf.artifact_io import sha256_bytes
        protocol["manifest_digest"] = sha256_bytes(canonical_json_bytes({k: v for k, v in protocol.items() if k != "manifest_digest"}))
        files["split_protocol.json"] = canonical_json_bytes(protocol)
    path = tmp_path / mutation / "split"
    forged = _republish(split.artifact.path, path, manifest_updates=updates, files=files)
    with pytest.raises(ArtifactValidationError, match="binding|bind"):
        open_canonical_split(path, expected_digest=forged.manifest_digest)


def test_incomplete_collection_rejected(tmp_path):
    _, path, executor = _collection(tmp_path, completed=9)
    with pytest.raises(ArtifactValidationError, match="complete"):
        open_verified_collection(path, replay_executor=executor)


def test_publication_only_projects_development_and_dataset(completed, tmp_path, monkeypatch):
    import werewolf.development_publication_v2 as v2
    from werewolf.tom.dataset import CanonicalToMDataset, ExperimentCapacity
    collection, _, split, _ = completed
    test_ids = {g["game_id"] for g in split.members("independent_test")}
    projection = v2._published_game_from_bundle
    def guarded(bundle):
        assert bundle.game_id not in test_ids
        return projection(bundle)
    monkeypatch.setattr(v2, "_published_game_from_bundle", guarded)
    handles = publish_development_v2(collection, split, tmp_path / "dev-v2", publication_id="dev-v2")
    view = handles.public.public_view
    assert len(view.game_ids) == 8 and not set(view.game_ids) & test_ids
    assert {g.game_id for g in handles.restricted.games} == set(view.game_ids)
    assert {g for f in view.fold_manifest.folds for g in f.game_ids} == set(view.game_ids)
    assert sorted(len(f.game_ids) for f in view.fold_manifest.folds) == [1, 1, 2, 2, 2]
    dataset = CanonicalToMDataset(view, view.game_ids, capacity=ExperimentCapacity(view.max_structured_token_count))
    assert {dataset[i].game_id for i in range(len(dataset))} == set(view.game_ids)
    for game_id in test_ids:
        with pytest.raises(ArtifactValidationError, match="outside"):
            view.load_game(game_id)
        with pytest.raises(ArtifactValidationError, match="outside"):
            open_development_publication_v2(handles.public.path, expected_split_digest=split.digest, game_ids=[game_id])
    assert {p.stem for p in (handles.public.path / "public/games").iterdir()} == set(view.game_ids)
    assert not (handles.public.path / "lineage/collection_plan.json").exists()
    lineage = b"".join(p.read_bytes() for p in (handles.public.path / "lineage").iterdir())
    assert all(game_id.encode() not in lineage for game_id in test_ids)
    assert b'"ordered_seed_pool"' not in lineage and b'"terminals"' not in lineage
    for member in split.members("independent_test"):
        assert canonical_json_bytes({"seed": member["seed"]})[1:-1] not in lineage
    from werewolf.tom.population import select_primary_population
    assert set(select_primary_population(handles.public).rows) == set(view.game_ids)
    with pytest.raises(ValueError, match="publication partition"):
        CanonicalToMDataset(view, list(test_ids), ExperimentCapacity(view.max_structured_token_count))
    with pytest.raises(ArtifactValidationError):
        open_publication(handles.public.path)


def test_development_reader_does_not_open_raw_collection_or_other_fold(completed, tmp_path, monkeypatch):
    collection, _, split, _ = completed
    handles = publish_development_v2(collection, split, tmp_path / "dev-v2", publication_id="dev-v2")
    chosen = handles.public.public_view.game_ids[:2]
    original = Path.open
    def guarded(path, *args, **kwargs):
        assert not path.is_relative_to(collection.path), "reader touched raw Collection/Test"
        if path.parent == handles.public.path / "public/games":
            assert path.stem in chosen, "reader touched another fold"
        assert "restricted" not in path.parts, "public reader touched role truth"
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    read = open_development_publication_v2(handles.public.path, expected_split_digest=split.digest, game_ids=chosen)
    assert read.public_view.game_ids == chosen


def test_v1_compatibility_and_no_old_path_for_split_bound_collection(completed, tmp_path):
    from werewolf.final_publication import publish_final_publication
    collection, _, _, _ = completed
    with pytest.raises(ArtifactValidationError, match="V2"):
        publish_development(collection, tmp_path / "v1", publication_id="v1")
    with pytest.raises(ArtifactValidationError, match="sealed"):
        publish_final_publication(collection, tmp_path / "final-v1", publication_id="final-v1", model_seal_digest="a" * 64)
    _, handles = _publication(tmp_path / "legacy")
    assert open_publication(handles.public.path).manifest_digest == handles.public.manifest_digest
    assert open_role_sidecar(handles.public).sidecar_digest == handles.restricted.sidecar_digest


def test_immutable_destinations(completed, tmp_path):
    collection, protocol, split, _ = completed
    before = (protocol.artifact.path / "manifest.json").read_bytes()
    with pytest.raises(ArtifactConflictError):
        publish_split_protocol(protocol.artifact.path, protocol_id="protocol", collection_id="synthetic-split",
                               development_count=8, test_count=2, split_seed=999)
    assert (protocol.artifact.path / "manifest.json").read_bytes() == before
    destination = tmp_path / "dev-v2"
    first = publish_development_v2(collection, split, destination, publication_id="dev-v2")
    assert publish_development_v2(collection, split, destination, publication_id="dev-v2").public.manifest_digest == first.public.manifest_digest
    with pytest.raises(ArtifactConflictError):
        _republish(destination, destination, manifest_updates={"source_revision": "changed"})
    with pytest.raises(ArtifactConflictError):
        _republish(split.artifact.path, split.artifact.path, manifest_updates={"development_count": 7})


def test_trusted_digests_required(completed):
    _, protocol, split, _ = completed
    with pytest.raises(ArtifactValidationError, match="trusted"):
        open_split_protocol(protocol.artifact.path, expected_digest="0" * 64)
    with pytest.raises(ArtifactValidationError, match="trusted"):
        open_canonical_split(split.artifact.path, expected_digest="0" * 64)
    from werewolf.cli import validate_artifact
    with pytest.raises(ValueError, match="expected-digest"):
        validate_artifact(split.artifact.path)
    assert validate_artifact(split.artifact.path, expected_digest=split.digest) == split.digest


def test_no_test_label_consumer_or_seal_bypass(completed, monkeypatch, tmp_path):
    from werewolf.tom import final_evaluation
    from types import SimpleNamespace
    collection, _, split, _ = completed
    def no_seal(_):
        raise ValueError("model seal missing")
    monkeypatch.setattr(final_evaluation, "verify_final_seal", no_seal)
    monkeypatch.setattr(final_evaluation, "publish_final_publication",
                        lambda *a, **kw: pytest.fail("publish before seal"))
    with pytest.raises(ValueError, match="seal missing"):
        final_evaluation.publish_final_evaluation(SimpleNamespace(), collection, tmp_path / "test", publication_id="test")
    monkeypatch.setattr(final_evaluation, "open_artifact_envelope",
                        lambda *a, **kw: pytest.fail("consume before seal"))
    with pytest.raises(ValueError, match="seal missing"):
        final_evaluation._consume(SimpleNamespace(), split.artifact.path, False)
    assert set(split.manifest["file_table"]) == {"split_protocol.json", "collection_plan.json", "attempt_ledger.json", "split_membership.json"}


@pytest.mark.parametrize("mutation", ["membership", "test_member", "extra_test_file", "source_revision"])
def test_development_rehashed_tampering_rejected(completed, tmp_path, mutation):
    collection, _, split, _ = completed
    handles = publish_development_v2(collection, split, tmp_path / "dev-v2", publication_id="dev-v2")
    updates, files = {}, {}
    if mutation in {"membership", "test_member"}:
        value = json.loads((handles.public.path / "lineage/development_membership.json").read_bytes())
        if mutation == "membership":
            value["members"][0]["ranking_digest"] = "0" * 64
        else:
            value["members"][0] = split.members("independent_test")[0]
        files["lineage/development_membership.json"] = canonical_json_bytes(value)
    elif mutation == "extra_test_file":
        game_id = split.members("independent_test")[0]["game_id"]
        files[f"public/games/{game_id}.json"] = b"{}"
    else:
        updates["source_revision"] = "0" * 40
    destination = tmp_path / mutation / "dev-v2"
    _republish(handles.public.path, destination, manifest_updates=updates, files=files)
    with pytest.raises(ArtifactValidationError):
        open_development_publication_v2(destination, expected_split_digest=split.digest)


def test_bundle_and_collection_substitution_rejected(completed):
    collection, protocol, split, _ = completed
    with pytest.raises(ArtifactValidationError, match="identity"):
        publish_canonical_split(replace(collection, collection_identity_digest="0" * 64), protocol,
                                split.artifact.path, split_id="split")
    altered = replace(collection.games[0], artifact=replace(collection.games[0].artifact, manifest_digest="0" * 64))
    with pytest.raises(ArtifactValidationError, match="substituted"):
        publish_canonical_split(replace(collection, games=(altered, *collection.games[1:])), protocol,
                                split.artifact.path, split_id="split")


def test_verified_wrapper_cannot_substitute_plan_fields_or_ledger_summary(completed):
    collection, protocol, split, _ = completed
    invalid = [replace(collection, plan=replace(collection.plan, source_revision="changed")),
               replace(collection, ledger_state=replace(collection.ledger_state, canonical_success_count=999))]
    for wrapper in invalid:
        with pytest.raises(ArtifactValidationError, match="Plan/Ledger|Ledger state"):
            publish_canonical_split(wrapper, protocol, split.artifact.path, split_id="split")


@pytest.mark.parametrize('development,test_count,sizes', [
    (5, 1, [1] * 5), (8, 2, [2, 2, 2, 1, 1]), (11, 4, [3, 2, 2, 2, 2]),
    (4800, 1000, [960] * 5),
])
def test_parameterized_protocol_counts_and_five_folds(tmp_path, development, test_count, sizes):
    from werewolf.development_publication import assign_development_folds, _development_game_set_digest
    protocol = publish_split_protocol(tmp_path / "formal-counts-fixture", protocol_id="formal-counts-fixture",
        collection_id="metadata-only-fixture", development_count=development, test_count=test_count, split_seed=101)
    assert protocol.manifest["target_success_count"] == development + test_count
    # Metadata only: no large acquisition or simulation is executed.
    pairs = tuple((f"synthetic-{i:04d}", f"{i:064x}") for i in range(development))
    folds = assign_development_folds(pairs, development_game_set_digest=_development_game_set_digest(pairs))
    assert [len(f.game_ids) for f in folds.folds] == sizes


@pytest.mark.parametrize('target,test_count', [(7, 2), (8, 1), (13, 4)])
def test_different_complete_collection_sizes_publish_only_development(tmp_path, target, test_count):
    protocol, path, executor = _collection(tmp_path, target=target, completed=target, test_count=test_count, seed=37)
    collection = open_verified_collection(path, replay_executor=executor)
    split = publish_canonical_split(collection, protocol, tmp_path / 'split', split_id='split')
    dev = {g['game_id'] for g in split.members('development')}
    test = {g['game_id'] for g in split.members('independent_test')}
    assert len(dev) == target - test_count and len(test) == test_count
    assert not dev & test and dev | test == {g.game_id for g in collection.games}
    handles = publish_development_v2(collection, split, tmp_path / 'dev', publication_id='dev')
    assert set(handles.public.public_view.game_ids) == dev
    assert handles.public.manifest['collection_success_count'] == target
    assert len(handles.restricted.games) == len(dev)
    fold_ids = [g for f in handles.public.public_view.fold_manifest.folds for g in f.game_ids]
    assert len(fold_ids) == len(set(fold_ids)) == len(dev) and set(fold_ids) == dev


def test_development_projection_does_not_open_test_bundles(completed, tmp_path, monkeypatch):
    collection, _, split, _ = completed
    test_ids = {g['game_id'] for g in split.members('independent_test')}
    roots = [g.path for g in collection.games if g.game_id in test_ids]
    original = Path.open
    def guarded(path, *args, **kwargs):
        assert not any(path.is_relative_to(root) for root in roots), 'Development projection opened Test evidence'
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded)
    handles = publish_development_v2(collection, split, tmp_path / 'dev-only', publication_id='dev-only')
    assert not test_ids & set(handles.public.public_view.game_ids)


def test_cli_synthetic_end_to_end(completed, tmp_path, monkeypatch, capsys):
    from werewolf import cli
    from werewolf.canonical_collection import production_runtime
    collection, protocol, _, executor = completed
    profile = tmp_path / "storage.json"
    profile.write_text(json.dumps({"artifact_root": str(tmp_path.resolve())}))
    prefix = ["--storage-profile", str(profile)]
    monkeypatch.setattr(cli, "_runtime", lambda _: {})
    monkeypatch.setattr(production_runtime, "classic7_replay_executor", lambda _: executor)
    assert cli.main(prefix + ["publish-canonical-split", "--collection", str(collection.path),
        "--runtime-config", "synthetic.yaml", "--split-protocol", str(protocol.artifact.path),
        "--split-protocol-digest", protocol.digest, "--destination", "cli-split"]) == 0
    digest = capsys.readouterr().out.strip()
    split_path = tmp_path / "splits/cli-split"
    split = open_canonical_split(split_path, expected_digest=digest)
    assert cli.main(prefix + ["publish-development-v2", "--collection", str(collection.path),
        "--runtime-config", "synthetic.yaml", "--split", "cli-split", "--split-digest", digest,
        "--destination", "cli-dev"]) == 0
    publication_digest = capsys.readouterr().out.strip()
    path = tmp_path / "publications/cli-dev"
    assert cli.main(prefix + ["validate-artifact", str(path), "--expected-digest", digest]) == 0
    assert capsys.readouterr().out.strip() == publication_digest
    assert cli.main(prefix + ["prepare-canonical-split-protocol", "--collection-id", "new-fixture",
        "--development-count", "8", "--test-count", "2", "--split-seed", "101", "--destination", "new-protocol"]) == 0
    protocol_digest = capsys.readouterr().out.strip()
    assert cli.main(prefix + ["validate-artifact", "split_protocols/new-protocol", "--expected-digest", protocol_digest]) == 0
    assert capsys.readouterr().out.strip() == protocol_digest
    assert len(split.members("independent_test")) == 2


def test_collect_requires_frozen_protocol_before_backend_access(completed, tmp_path, monkeypatch):
    from werewolf import cli, backends
    collection, _, _, _ = completed
    profile = tmp_path / "storage.json"
    profile.write_text(json.dumps({"artifact_root": str(tmp_path.resolve())}))
    monkeypatch.setattr(cli, "_runtime", lambda _: {})
    monkeypatch.setattr(backends, "load_named_backends", lambda *a, **k: pytest.fail("backend accessed before split preflight"))
    with pytest.raises(ValueError, match="split-bound collection requires"):
        cli.main(["--storage-profile", str(profile), "collect", "--plan", str(collection.path / "collection_plan.json"),
                  "--runtime-config", "synthetic.yaml", "--call-limit", "1000", "--destination", "new-collection"])


def test_cli_parameter_file_and_ambiguous_config_rejected(tmp_path, capsys):
    from werewolf import cli
    profile = tmp_path / "storage.json"
    profile.write_text(json.dumps({"artifact_root": str(tmp_path.resolve())}))
    config = tmp_path / "parameters.json"
    config.write_text(json.dumps(dict(collection_id="config-fixture", development_count=8, test_count=2, split_seed=101)))
    prefix = ["--storage-profile", str(profile), "prepare-canonical-split-protocol", "--config", str(config)]
    assert cli.main(prefix + ["--destination", "config-protocol"]) == 0
    digest = capsys.readouterr().out.strip()
    protocol = open_split_protocol(tmp_path / "split_protocols/config-protocol", expected_digest=digest)
    assert protocol.manifest["split_seed"] == 101
    with pytest.raises(ValueError, match="combined"):
        cli.main(prefix + ["--destination", "ambiguous", "--split-seed", "102"])
    config.write_text('{"collection_id":"x","development_count":8,"test_count":2,"split_seed":101,"split_seed":102}')
    with pytest.raises(ValueError, match="duplicate"):
        cli.main(prefix + ["--destination", "duplicate"])
    assert not (tmp_path / "split_protocols/ambiguous").exists()
    assert not (tmp_path / "split_protocols/duplicate").exists()
