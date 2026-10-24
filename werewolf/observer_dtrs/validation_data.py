"""Whole-game legal and explicit offline target partitions for nested execution."""
from collections import Counter
from dataclasses import dataclass

from werewolf.artifact_io import canonical_json_bytes, sha256_bytes
from werewolf.observer_dtrs.dataset import open_dataset
from werewolf.observer_dtrs.dataset_offline import open_supervision
from werewolf.observer_dtrs.estimation import TrainingRow
from werewolf.observer_dtrs.features import ObserverFeatures
from werewolf.observer_dtrs.splits import source_manifest, read_splits, game_partition


@dataclass(frozen=True)
class PredictionInput:
    sample_id: str
    game_id: str
    features: ObserverFeatures

    def record(self):
        return [self.sample_id, self.game_id, self.features.to_record()]


class FormalData:
    scope = 'formal_development_validation'

    def __init__(self, path, *, split_path, split_digest, protocol):
        self.path, self.protocol = path, protocol
        self.manifest = source_manifest(path, protocol)
        self.split = read_splits(split_path, dataset_manifest=self.manifest, protocol=protocol,
                                 expected_manifest_digest=split_digest)
        self.split_digest = split_digest
        self.dataset_digest = self.manifest['manifest_digest']
        self.sizes = {g['game_id']: g['row_count'] for g in self.manifest['games']}
        self.folds = {g['game_id']: g['fold_index'] for g in self.manifest['games']}
        if any(n <= 0 for n in self.sizes.values()):
            raise ValueError('frozen eligible game population unexpectedly contains zero samples')
        self.origins = self.manifest['origins']
        self.views = {}

    def _view(self, games):
        key=tuple(sorted(games))
        if key not in self.views:
            self.views[key]=open_dataset(self.path,expected_digest=self.dataset_digest,game_ids=key)
        return self.views[key]

    def legal(self, games):
        view = self._view(games)
        return tuple(PredictionInput(r.sample_id, r.context.game_id, r.features) for r in view.rows)

    def targets(self, games):
        view = self._view(games)
        rows = open_supervision(view)
        if any(r.q_source.fold_index != self.folds[r.legal.context.game_id] for r in rows):
            raise ValueError('ToM prediction source fold mismatch')
        return {r.legal.sample_id: {'y': r.supervision.event_label_y,
            'successful_empty': not r.supervision.raw_suspicion_support} for r in rows}


class ControlledData:
    """Small explicit fixture, never an alternate formal population or dataset."""
    scope = 'controlled_fixture'

    def __init__(self, rows, targets, *, protocol):
        self.rows = tuple(rows)
        self.target_values = targets
        if len({r.sample_id for r in rows}) != len(rows) or set(targets) != {r.sample_id for r in rows}:
            raise ValueError('controlled fixture coverage mismatch')
        self.sizes = dict(Counter(r.game_id for r in rows))
        self.protocol = protocol
        games = sorted(self.sizes)
        # Game IDs carry source-fold membership in the fixture, independently of Y.
        self.folds = {g: int(g.split('-')[1]) for g in games}
        self.origins = {'fixture': 'controlled_not_formal_ToM_or_publication'}
        self.dataset_digest = sha256_bytes(canonical_json_bytes([[r.record(), targets[r.sample_id]] for r in rows]))
        folds = []
        for k in range(5):
            ids = [g for g in games if self.folds[g] == k]
            if not ids or len(ids) % 15:
                raise ValueError('controlled fold needs a positive multiple of 15 games')
            outer = game_partition(ids, protocol=protocol, tom_fold=k, folds=5)
            nested = [{'outer_fold': j, 'inner_assignment': game_partition(
                [g for g in ids if outer[g] != j], protocol=protocol, tom_fold=k, folds=3, parent_outer=j)} for j in range(5)]
            folds.append({'tom_fold': k, 'outer_assignment': outer, 'nested': nested})
        self.split = {'folds': folds, 'coverage': {'games': len(games), 'samples': len(rows)},
                      'scope': self.scope}
        self.split_digest = sha256_bytes(canonical_json_bytes(self.split))

    def legal(self, games):
        ids = set(games)
        return tuple(r for r in self.rows if r.game_id in ids)

    def targets(self, games):
        return {r.sample_id: self.target_values[r.sample_id] for r in self.legal(games)}


def controlled_fixture(protocol):
    rows, targets = [], {}
    for k in range(5):
        for g in range(15):
            game = f'fixture-{k}-{g:02}'
            for j in range(1 + (g % 3)):
                sid = f'{game}-{j}'
                nW = 1 + ((g + j) % 2)
                phase = 'pk_discussion' if (g + j) % 4 == 0 else 'discussion'
                r = (.1, .3, .6, .9)[(g + j) % 4]
                f = ObserverFeatures(r if nW == 1 else r / 2, 0 if nW == 1 else r / 2,
                                     r, None if nW == 1 else 0, phase, nW)
                rows.append(PredictionInput(sid, game, f))
                targets[sid] = {'y': int((g + 2 * j + k) % 5 != 0),
                                'successful_empty': (g + 2 * j + k) % 5 == 1}
    return ControlledData(rows, targets, protocol=protocol)


def training_rows(data, games):
    legal = data.legal(games)
    targets = data.targets(games)
    if set(targets) != {r.sample_id for r in legal} or len(legal) != sum(data.sizes[g] for g in games):
        raise ValueError('training partition has missing/duplicate supervision')
    return tuple(TrainingRow(r.sample_id, r.game_id, r.features, targets[r.sample_id]['y']) for r in legal)


def fit_source(data, games, *, tom_fold, outer_fold, inner_fold, implementation_digest):
    if any(data.folds[g] != tom_fold for g in games):
        raise ValueError('cannot mix ToM held-out source folds')
    formal = data.scope == 'formal_development_validation'
    return {'scope': 'formal_training_partition' if formal else 'controlled_fixture',
        'protocol_digest': data.protocol.digest,
        'dataset_digest': data.dataset_digest if formal else None,
        'split_manifest_digest': data.split_digest if formal else None,
        'tom_fold': tom_fold if formal else None, 'outer_fold': outer_fold if formal else None,
        'inner_fold': inner_fold if formal else None,
        'game_sizes': {g: data.sizes[g] for g in games}, 'origins': data.origins,
        'implementation_digest': implementation_digest}
