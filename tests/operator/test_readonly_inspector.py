"""No server contact: synthetic metadata and subprocess mocks only."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import inspect_canonical_collection_readonly as snapshot
from tests.development_publication.test_canonical_split_v2 import _collection


def test_ledger_snapshot_reads_only_metadata_and_never_writes(tmp_path, monkeypatch):
    from werewolf.canonical_collection import load_collection_plan
    _, path, _ = _collection(tmp_path)
    plan = load_collection_plan(path)
    opened = []
    original = Path.open
    def read_only(p, mode='r', *args, **kwargs):
        assert mode in ('r', 'rb')
        assert 'games' not in p.parts
        opened.append(p)
        return original(p, mode, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', read_only)
    result = snapshot._ledger_summary(path, plan.to_record())
    assert result['claims'] == result['terminals'] == 10
    assert result['outcomes'] == {'canonical_success': 10}
    assert result['open_claims'] == 0 and result['target_reached_in_metadata']
    assert opened


@pytest.mark.parametrize('recompute', [False, True])
def test_metadata_inspector_rejects_modified_seed_even_after_rehash(tmp_path, recompute):
    import hashlib
    from werewolf.canonical_collection import load_collection_plan
    _, path, _ = _collection(tmp_path)
    plan = load_collection_plan(path)
    claim = sorted((path/'attempt_ledger').glob('*.claim.json'))[0]
    value = snapshot._json(claim); value['seed'] += 1
    if recompute:
        value['record_digest'] = hashlib.sha256(snapshot._canonical({k:v for k,v in value.items() if k != 'record_digest'})).hexdigest()
    claim.write_bytes(snapshot._canonical(value))
    with pytest.raises(ValueError, match='mismatch'):
        snapshot._ledger_summary(path, plan.to_record())


def test_seed_inventory_matches_proposed_parameters_and_detects_overlap(tmp_path):
    first = dict(collection_id='candidate-a', target_games=7, seed_pool_size=11, call_limit=37)
    second = dict(collection_id='candidate-b', target_games=19, seed_pool_size=23, call_limit=211)
    path = tmp_path/'canonical/old/collection_plan.json'; path.parent.mkdir(parents=True)
    path.write_bytes(snapshot._canonical({'ordered_seed_pool': [snapshot._seeds(first)[0]]}))
    inventory = snapshot._seed_inventory([tmp_path], [first, second], [snapshot._seeds(second)[0]])
    assert inventory['historical_plan_count'] == 1
    assert inventory['pools']['candidate-a']['historical_overlap'] == 1
    assert inventory['pools']['candidate-b']['historical_overlap'] == 0
    assert inventory['pools']['candidate-b']['excluded_overlap'] == 1
    assert inventory['pairwise_candidate_overlap'] == [{'left': 'candidate-a', 'right': 'candidate-b', 'overlap': 0}]
    assert inventory['pools']['candidate-a']['size'] == 11
    assert inventory['pools']['candidate-b']['size'] == 23


def test_snapshot_main_uses_explicit_paths_and_trusted_reference(tmp_path, monkeypatch, capsys):
    from werewolf.canonical_collection import load_collection_plan
    _, collection, _ = _collection(tmp_path)
    plan = load_collection_plan(collection)
    candidate = tmp_path / 'candidate.json'
    candidate.write_text(json.dumps(dict(collection_id='independent-candidate', target_games=8, seed_pool_size=17, call_limit=55)))
    monkeypatch.setattr(snapshot, '_command', lambda argv: {'exit_code': 0, 'stdout': 'fixture', 'stderr': ''})
    args = ['--repo', str(tmp_path), '--data-root', str(tmp_path), '--collection', str(collection),
            '--plan-digest', plan.plan_digest, '--campaign', str(candidate), '--history-root', str(tmp_path), '--service-port', '9123']
    assert snapshot.main(args) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['reference_plan']['collection_id'] == plan.collection_id
    assert report['reference_ledger_metadata']['outcomes']['canonical_success'] == 10
    assert report['seed_inventory']['pools']['independent-candidate']['size'] == 17
    assert report['deep_bundle_replay_performed'] is False
    assert snapshot.main([*args[:args.index('--plan-digest')], '--plan-digest', '0' * 64, *args[args.index('--campaign'):]]) == 2
    assert json.loads(capsys.readouterr().out)['errors']


@pytest.mark.parametrize('extra', [[], ['--collection', '/reference'], ['--service-port', '0'], ['--campaign', '/candidate.json']])
def test_snapshot_rejects_missing_or_invalid_explicit_inputs(extra):
    args = [] if not extra else ['--repo', '/repository', '--data-root', '/artifacts', *extra]
    with pytest.raises(SystemExit) as error:
        snapshot.main(args)
    assert error.value.code == 2


@pytest.mark.parametrize('change', [{'target_games': True}, {'seed_pool_size': 1}, {'call_limit': 0}, {'collection_id': '../unsafe'}])
def test_candidate_parameters_reject_invalid_values(change):
    candidate = dict(collection_id='candidate', target_games=7, seed_pool_size=11, call_limit=37)
    with pytest.raises(ValueError):
        snapshot._campaign({**candidate, **change})


def test_commands_disable_git_locks_and_use_no_shell_or_network(monkeypatch):
    observed = []
    def run(argv, **kwargs):
        assert kwargs['env']['GIT_OPTIONAL_LOCKS'] == '0'
        assert 'shell' not in kwargs
        assert argv[0] not in ('ssh', 'scp', 'curl', 'wget')
        observed.append(argv)
        return SimpleNamespace(returncode=0, stdout='ok\n', stderr='')
    monkeypatch.setattr(snapshot.subprocess, 'run', run)
    assert snapshot._command(['git', '-C', '/local-fixture', 'status', '--porcelain'])['stdout'] == 'ok'
    assert observed


def test_duplicate_or_nonfinite_metadata_is_rejected(tmp_path):
    path = tmp_path/'bad.json'
    for value in (b'{"seed":1,"seed":2}', b'{"seed":NaN}'):
        path.write_bytes(value)
        with pytest.raises(ValueError): snapshot._json(path)
