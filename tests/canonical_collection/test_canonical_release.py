"""Synthetic native games; no external inference, training, or server access."""

import json

import pytest

from tests.canonical_collection.test_game_bundle import _plan
from tests.canonical_collection.test_terminal_audit import _NativeAgent, _native_game
from tests.runtime.test_runtime_config import new_config
from tests.development_publication.test_canonical_split_v2 import _collection
from tests.development_publication.test_development_publication import _republish
from werewolf.artifact_io import ArtifactConflictError, ArtifactValidationError, canonical_json_bytes, sha256_bytes
from werewolf.canonical_collection import construct_attempt_claim
from werewolf.canonical_collection.production_runtime import classic7_replay_executor
from werewolf.canonical_split import PROTOCOL_BINDING, publish_split_protocol, open_canonical_split
from werewolf.canonical_release import publish_canonical_release, verify_canonical_release, ACCESS_POLICY
from werewolf.development_publication import open_verified_collection
from werewolf.development_publication_v2 import open_development_publication_v2
from werewolf.runtime_config import normalize_runtime_config


@pytest.fixture
def native_split_collection(tmp_path):
    protocol = publish_split_protocol(tmp_path / "split_protocols/protocol", protocol_id="protocol",
        collection_id="native-split", development_count=5, test_count=1, split_seed=20261012)
    plan = _plan(collection_id="native-split", ordered_seed_pool=tuple(range(100, 106)),
        target_canonical_success_count=6, environment_provenance={"fixture": "release", PROTOCOL_BINDING: protocol.digest})
    config = normalize_runtime_config(new_config())
    path = tmp_path / "canonical/native-split"
    for ordinal in range(6):
        claim = construct_attempt_claim(plan, ordinal=ordinal, attempt_id=f"attempt-{ordinal:03d}",
            claim_timestamp_utc="2026-10-12T00:00:00Z")
        _native_game(path, _NativeAgent, plan=plan, claim=claim, config=config, game_id=f"native-{ordinal:03d}")
    collection = open_verified_collection(path, replay_executor=classic7_replay_executor(config))
    return tmp_path, collection, protocol, config


def test_native_release_is_repeatable_and_keeps_test_out_of_development(native_split_collection, monkeypatch):
    import werewolf.canonical_release as release_module
    projected = []
    project = release_module._published_game_from_bundle
    def observe_projection(bundle):
        projected.append(bundle.game_id)
        return project(bundle)
    monkeypatch.setattr(release_module, '_published_game_from_bundle', observe_projection)
    root, collection, protocol, config = native_split_collection
    destination = root / "releases/release-v1"
    original = {p: p.read_bytes() for p in collection.path.rglob('*') if p.is_file()}
    receipt = publish_canonical_release(collection, protocol, destination, runtime_config=config, artifact_root=root)
    assert receipt.manifest['test_access_policy'] == ACCESS_POLICY
    assert receipt.manifest['success_count'] == 6
    assert receipt.manifest['development_count'] == 5 and receipt.manifest['test_count'] == 1
    assert verify_canonical_release(destination, expected_digest=receipt.manifest_digest, artifact_root=root).manifest_digest == receipt.manifest_digest
    assert publish_canonical_release(collection, protocol, destination, runtime_config=config, artifact_root=root).manifest_digest == receipt.manifest_digest
    split = open_canonical_split(root / 'splits/release-v1', expected_digest=receipt.manifest['split_manifest_digest'])
    development = open_development_publication_v2(root / 'publications/release-v1-development', expected_split_digest=split.digest)
    test_ids = {m['game_id'] for m in split.members('independent_test')}
    assert projected and not test_ids & set(projected)
    assert not test_ids & set(development.public_view.game_ids)
    assert [len(f.game_ids) for f in development.public_view.fold_manifest.folds] == [1] * 5
    for game_id in test_ids:
        with pytest.raises(ArtifactValidationError, match='outside'):
            development.public_view.load_game(game_id)
        assert game_id.encode() not in (destination / 'release.json').read_bytes()
    assert 'winner' not in receipt.manifest and 'role_assignment' not in receipt.manifest
    assert all(p.read_bytes() == data for p, data in original.items())
    with pytest.raises(ArtifactValidationError):
        open_development_publication_v2(destination, expected_split_digest=split.digest)
    with pytest.raises(ArtifactValidationError, match='trusted digest'):
        verify_canonical_release(destination, expected_digest='0' * 64, artifact_root=root)


def test_partial_game_rejected_before_split_or_development(tmp_path):
    protocol, path, executor = _collection(tmp_path)
    collection = open_verified_collection(path, replay_executor=executor)
    with pytest.raises(ArtifactValidationError):
        publish_canonical_release(collection, protocol, tmp_path / 'releases/rejected',
            runtime_config=normalize_runtime_config(new_config()), artifact_root=tmp_path)
    assert not (tmp_path / 'releases/rejected').exists()
    assert not (tmp_path / 'splits/rejected').exists()
    assert not (tmp_path / 'publications/rejected-development').exists()


@pytest.mark.parametrize('change', ['parent_digest', 'path_escape', 'audit_digest'])
def test_rehashed_receipt_cannot_change_verified_parents(native_split_collection, change):
    root, collection, protocol, config = native_split_collection
    receipt = publish_canonical_release(collection, protocol, root / 'releases/release-v1', runtime_config=config, artifact_root=root)
    updates, files = {}, {}
    if change == 'parent_digest':
        updates['development_publication_digest'] = '0' * 64
    else:
        record = json.loads((receipt.path / 'release.json').read_bytes())
        record['collection_path' if change == 'path_escape' else 'terminal_audit_digest'] = '../outside' if change == 'path_escape' else '0' * 64
        files['release.json'] = canonical_json_bytes(record)
    isolated = root / 'forged/releases/release-v1'
    forged = _republish(receipt.path, isolated, manifest_updates=updates, files=files)
    # Reuse exactly the valid parents, retaining their bytes and source paths.
    from shutil import copytree
    for directory in ('canonical', 'split_protocols', 'terminal_audits', 'splits', 'publications'):
        copytree(root / directory, root / 'forged' / directory)
    with pytest.raises(ArtifactValidationError):
        verify_canonical_release(isolated, expected_digest=forged.manifest_digest, artifact_root=root / 'forged')


def test_receipt_never_overwrites_conflicting_destination(native_split_collection):
    from werewolf.artifact_io import publish_artifact
    root, collection, protocol, config = native_split_collection
    destination = root / 'releases/release-v1'
    publish_artifact(destination, manifest_fields={'artifact_type': 'sentinel', 'schema_version': 'fixture-v1'}, files={'keep': b'preserve'})
    with pytest.raises((ArtifactConflictError, ArtifactValidationError)):
        publish_canonical_release(collection, protocol, destination, runtime_config=config, artifact_root=root)
    assert (destination / 'keep').read_bytes() == b'preserve'


@pytest.mark.parametrize('change', ['roles', 'public_record'])
def test_rehashed_release_rejects_development_content_different_from_source(native_split_collection, change):
    from shutil import copytree
    from werewolf.development_publication import open_role_sidecar
    root, collection, protocol, config = native_split_collection
    receipt = publish_canonical_release(collection, protocol, root / 'releases/release-v1',
                                        runtime_config=config, artifact_root=root)
    source = root / 'publications/release-v1-development'
    updates, files = {}, {}
    if change == 'roles':
        sidecar = json.loads((source / 'restricted/role_sidecar.json').read_bytes())
        roles = sidecar['games'][0]['role_assignment']
        # Preserve legal role counts while changing authoritative seat truth.
        left = next(iter(roles))
        right = next(player for player in roles if roles[player] != roles[left])
        roles[left], roles[right] = roles[right], roles[left]
        sidecar['sidecar_digest'] = sha256_bytes(canonical_json_bytes(
            {k: v for k, v in sidecar.items() if k != 'sidecar_digest'}))
        updates['role_sidecar_digest'] = sidecar['sidecar_digest']
        files['restricted/role_sidecar.json'] = canonical_json_bytes(sidecar)
    else:
        entries = json.loads((source / 'manifest.json').read_bytes())['development_games']
        game_path = f"public/games/{entries[0]['game_id']}.json"
        game = json.loads((source / game_path).read_bytes())
        budget = game['call_budget_summary']
        budget['configured_call_limit'] += 1
        budget['summary_digest'] = sha256_bytes(canonical_json_bytes(
            {k: v for k, v in budget.items() if k != 'summary_digest'}))
        game['record_digest'] = sha256_bytes(canonical_json_bytes(
            {k: v for k, v in game.items() if k != 'record_digest'}))
        entries[0]['published_record_digest'] = game['record_digest']
        updates['development_games'] = entries
        files[game_path] = canonical_json_bytes(game)
    isolated = root / 'forged'
    development = _republish(source, isolated / 'publications/release-v1-development',
                             manifest_updates=updates, files=files)
    # Portable readers still accept internally consistent, validly rehashed
    # records. Source-side release verification must also check Canonical truth.
    view = open_development_publication_v2(development.path,
                                          expected_split_digest=receipt.manifest['split_manifest_digest'])
    open_role_sidecar(view)
    for directory in ('canonical', 'split_protocols', 'terminal_audits', 'splits'):
        copytree(root / directory, isolated / directory)
    forged = _republish(receipt.path, isolated / 'releases/release-v1',
        manifest_updates={'development_publication_digest': development.manifest_digest})
    with pytest.raises(ArtifactValidationError, match='Canonical source'):
        verify_canonical_release(forged.path, expected_digest=forged.manifest_digest, artifact_root=isolated)


def test_cli_release_and_validation_use_trusted_receipt_digest(native_split_collection, capsys):
    import yaml
    from werewolf.cli import main
    root, collection, protocol, config = native_split_collection
    profile = root / 'storage.json';profile.write_bytes(canonical_json_bytes({'artifact_root': str(root)}))
    runtime = root / 'runtime.yaml';runtime.write_text(yaml.safe_dump(config))
    prefix = ['--storage-profile', str(profile)]
    assert main(prefix + ['release-canonical-development-v2', '--collection', str(collection.path),
        '--runtime-config', str(runtime), '--split-protocol', str(protocol.artifact.path),
        '--split-protocol-digest', protocol.digest, '--destination', 'release-v1']) == 0
    digest = capsys.readouterr().out.strip()
    assert main(prefix + ['validate-artifact', str(root / 'releases/release-v1'), '--expected-digest', digest]) == 0
    assert capsys.readouterr().out.strip() == digest
    with pytest.raises(ValueError, match='expected-digest'):
        main(prefix + ['validate-artifact', str(root / 'releases/release-v1')])
