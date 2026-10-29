"""Data-only migration boundary and synthetic CLI lifecycle, without Torch."""
import os
from pathlib import Path
import subprocess
import sys


def test_data_cli_end_to_end_without_training_imports(tmp_path):
    code = r'''
import importlib.abc
from pathlib import Path
import sys
class NoTraining(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'werewolf.tom' or fullname.startswith(('werewolf.tom.', 'torch', 'transformers')):
            raise AssertionError('unexpected training dependency: ' + fullname)
sys.meta_path.insert(0, NoTraining())
from tests.development_publication.test_canonical_split_v2 import _collection
from werewolf import cli
from werewolf.canonical_collection import production_runtime
from werewolf.development_publication_v2 import open_development_publication_v2
from werewolf.canonical_split import open_canonical_split
from werewolf import canonical_terminal_audit, canonical_release, final_publication
from scripts import collect_games
root = Path(sys.argv[1])
protocol, path, executor = _collection(root, target=7, completed=7, test_count=2, seed=37)
native_executor = production_runtime.classic7_replay_executor
cli._runtime = lambda _: {}
production_runtime.classic7_replay_executor = lambda _: executor
profile = root / 'storage.json'
profile.write_text(__import__('json').dumps({'artifact_root': str(root)}))
base = ['--storage-profile', str(profile)]
assert cli.main(base + ['publish-canonical-split', '--collection', str(path), '--runtime-config', 'fixture.yaml',
    '--split-protocol', str(protocol.artifact.path), '--split-protocol-digest', protocol.digest, '--destination', 'split']) == 0
split = open_canonical_split(root / 'splits/split', expected_digest=__import__('json').loads((root/'splits/split/manifest.json').read_text())['manifest_digest'])
assert cli.main(base + ['publish-development-v2', '--collection', str(path), '--runtime-config', 'fixture.yaml',
    '--split', str(split.artifact.path), '--split-digest', split.digest, '--destination', 'dev']) == 0
assert cli.main(base + ['validate-artifact', str(root/'publications/dev'), '--expected-digest', split.digest]) == 0
pub = open_development_publication_v2(root/'publications/dev', expected_split_digest=split.digest)
assert pub.manifest['game_count'] == 5 and pub.manifest['collection_success_count'] == 7
# Native terminal audit and the controlled release path also run without any
# training import, using local deterministic Fixture agents only.
production_runtime.classic7_replay_executor = native_executor
from tests.canonical_collection.test_canonical_release import native_split_collection
native_root, native_collection, native_protocol, config = native_split_collection.__wrapped__(root / 'native')
profile.write_text(__import__('json').dumps({'artifact_root': str(native_root)}))
cli._runtime = lambda _: config
assert cli.main(base + ['release-canonical-development-v2', '--collection', str(native_collection.path),
    '--runtime-config', 'fixture.yaml', '--split-protocol', str(native_protocol.artifact.path),
    '--split-protocol-digest', native_protocol.digest, '--destination', 'native-release']) == 0
receipt_path = native_root / 'releases/native-release'
receipt_digest = __import__('json').loads((receipt_path/'manifest.json').read_bytes())['manifest_digest']
assert cli.main(base + ['validate-artifact', str(receipt_path), '--expected-digest', receipt_digest]) == 0
assert 'werewolf.tom.development_input' not in sys.modules
'''
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', TMPDIR='/private/tmp')
    result = subprocess.run([sys.executable, '-B', '-c', code, str(tmp_path)],
                            cwd=Path(__file__).resolve().parents[2], env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_integrated_v2_training_and_data_commands_are_registered():
    from werewolf.cli import build_parser
    import argparse
    commands = next(a for a in build_parser()._actions if isinstance(a, argparse._SubParsersAction)).choices
    assert {'prepare-experiment-v2', 'prepare-final-experiment-v2', 'prepare-backbone-tom-study-v2'} <= commands.keys()
    assert {'prepare-canonical-split-protocol', 'publish-canonical-split', 'publish-development-v2',
            'audit-canonical-terminals', 'release-canonical-development-v2'} <= commands.keys()
