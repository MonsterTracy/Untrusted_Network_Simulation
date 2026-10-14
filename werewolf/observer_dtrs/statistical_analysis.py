"""Fixed-prediction calibration and paired whole-game bootstrap, never refits."""
from bisect import bisect_right
from collections import Counter, defaultdict
import math

import numpy as np

from werewolf.artifact_io import canonical_json_bytes, sha256_bytes

MODELS = ('OBS-EQ0', 'OBS-EQ1', 'OBS-LR-r', 'OBS-EF')
WEIGHTINGS = ('game_equal', 'record_equal')
VERSION = 'observer_dtrs_fixed_prediction_analysis_v1'


def bootstrap_plan(blocks, *, protocol, formal):
    ids = [g for key in sorted(blocks) for g in sorted(blocks[key])]
    if (set(blocks) != {(k, j) for k in range(5) for j in range(5)} or len(set(ids)) != len(ids)
            or any(not v for v in blocks.values()) or (formal and any(len(v) != 60 for v in blocks.values()))):
        raise ValueError('bootstrap requires complete unique source/outer game blocks')
    spec = protocol.record['uncertainty']
    rng = np.random.Generator(np.random.PCG64(spec['seed']))
    indices = np.empty((spec['replicates'], len(ids)), dtype=np.int64)
    for replicate in range(spec['replicates']):
        offset = 0
        for key in sorted(blocks):
            size = len(blocks[key])
            indices[replicate, offset:offset + size] = offset + rng.integers(0, size, size=size, dtype=np.int64)
            offset += size
    return ids, indices


def _interval(values, *, valid_min=.95):
    valid = values[np.isfinite(values)]
    if len(valid) / len(values) < valid_min:
        return None
    return np.quantile(valid, [.025, .975], method='linear').tolist()


def analyze(rows, *, blocks, protocol, scope):
    rows = tuple(rows)
    if (not rows or len({r['sample_id'] for r in rows}) != len(rows)
            or any(type(r['y']) is not int or r['y'] not in (0, 1)
                or type(r['successful_empty']) is not bool or (r['successful_empty'] and r['y'] != 1)
                or set(r['predictions']) != set(WEIGHTINGS)
                or any(set(r['predictions'][w]) != set(MODELS) for w in WEIGHTINGS) for r in rows)):
        raise ValueError('analysis population/target/prediction coverage mismatch')
    order, draws = bootstrap_plan(blocks, protocol=protocol, formal=scope == 'formal_development_validation')
    if set(order) != {r['game_id'] for r in rows}:
        raise ValueError('analysis incomplete held-out games')
    index = {g: j for j, g in enumerate(order)}
    sizes = Counter(r['game_id'] for r in rows)
    epsilon = protocol.record['evaluation']['log_loss_score_epsilon']
    edges = protocol.record['evaluation']['calibration_bin_edges']
    groups = {'overall': lambda r: True,
        **{f'{p}/{n}': (lambda r, p=p, n=n: r['phase'] == p and r['nW'] == n)
           for p in ('discussion', 'pk_discussion') for n in (1, 2)},
        'successful_empty': lambda r: r['successful_empty'], 'nonempty': lambda r: not r['successful_empty']}
    summary = {'schema_version': VERSION, 'scope': scope, 'research_scope': protocol.record['research_scope'],
        'known_source_limitations': protocol.record['known_source_limitations'],
        'protocol_digest': protocol.digest, 'rows': len(rows), 'games': len(order),
        'game_order': order, 'numpy_version': np.__version__,
        'bootstrap': {**protocol.record['uncertainty'],
            'indices_sha256': sha256_bytes(draws.astype('<i8').tobytes()),
            'game_order_digest': sha256_bytes(canonical_json_bytes(order))},
        'models': {}, 'comparisons': {}}
    overall = {}
    for weighting in WEIGHTINGS:
        summary['models'][weighting] = {}
        overall[weighting] = {}
        for model in MODELS:
            probabilities = [r['predictions'][weighting][model] for r in rows]
            if any(isinstance(p, bool) or not isinstance(p, (float, int)) or not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
                raise ValueError('invalid analysis probability')
            stats = np.zeros((len(order), 4), dtype=np.float64)  # weighted BS, LL, bias, mass
            values = []
            for r, p in zip(rows, probabilities, strict=True):
                w = 1 / sizes[r['game_id']] if weighting == 'game_equal' else 1.
                q = min(1 - epsilon, max(epsilon, p))
                v = np.array([(p-r['y'])**2, -(r['y']*math.log(q)+(1-r['y'])*math.log1p(-q)), p-r['y'], 1.]) * w
                values.append(v)
                stats[index[r['game_id']]] += v
            overall[weighting][model] = stats
            result = {'groups': {}, 'calibration': [],
                'clipped_rows': sum(p != min(1-epsilon, max(epsilon, p)) for p in probabilities),
                'wrong_endpoint_rows': sum((p == 0 and r['y'] == 1) or (p == 1 and r['y'] == 0)
                    for r,p in zip(rows,probabilities,strict=True))}
            for name, keep in groups.items():
                chosen = [j for j, r in enumerate(rows) if keep(r)]
                matrix = np.zeros_like(stats)
                for j in chosen:
                    matrix[index[rows[j]['game_id']]] += values[j]
                total = matrix.sum(axis=0)
                boot = matrix[draws].sum(axis=1)
                with np.errstate(divide='ignore', invalid='ignore'):
                    boot = boot[:,:3] / boot[:,3,None]
                result['groups'][name] = {'rows': len(chosen), 'games': len({rows[j]['game_id'] for j in chosen}),
                    'positive': sum(rows[j]['y'] for j in chosen), 'mass': float(total[3]),
                    **{metric: (float(total[c]/total[3]) if total[3] else None)
                       for c, metric in enumerate(('brier','log_loss','bias'))},
                    'intervals': {metric: _interval(boot[:,c]) for c,metric in enumerate(('brier','log_loss','bias'))}}
            calibration_sum = 0.
            for b in range(10):
                chosen = [j for j,p in enumerate(probabilities) if min(9,bisect_right(edges[1:-1],p)) == b]
                matrix = np.zeros((len(order), 3))  # weighted prediction, Y, mass
                for j in chosen:
                    r,p=rows[j],probabilities[j]
                    w=1/sizes[r['game_id']] if weighting == 'game_equal' else 1.
                    matrix[index[r['game_id']]] += [w*p,w*r['y'],w]
                total=matrix.sum(axis=0)
                games=len({rows[j]['game_id'] for j in chosen})
                point={'bin':b,'rows':len(chosen),'games':games,'mass':float(total[2]),
                    'mean_prediction':float(total[0]/total[2]) if total[2] else None,
                    'observed_rate':float(total[1]/total[2]) if total[2] else None,
                    'intervals':None}
                if total[2]:
                    calibration_sum += abs(total[0]-total[1])
                if games >= protocol.record['evaluation']['calibration_interval_min_games']:
                    boot=matrix[draws].sum(axis=1)
                    with np.errstate(divide='ignore',invalid='ignore'):
                        boot=boot[:,:2]/boot[:,2,None]
                    point['intervals']={'mean_prediction':_interval(boot[:,0]),'observed_rate':_interval(boot[:,1])}
                result['calibration'].append(point)
            result['ece_descriptive']=float(calibration_sum/stats[:,3].sum())
            result['by_tom_fold']={str(k): _point_scores(stats[[index[g] for (f,j),ids in blocks.items() if f == k for g in ids]]) for k in range(5)}
            summary['models'][weighting][model]=result
    for weighting in WEIGHTINGS:
        summary['comparisons'][weighting]={}
        first=overall[weighting]['OBS-EQ0']
        for model in MODELS[1:]:
            second=overall[weighting][model]
            a,b=first.sum(axis=0),second.sum(axis=0)
            boot_a,boot_b=first[draws].sum(axis=1),second[draws].sum(axis=1)
            delta=boot_b[:,:3]/boot_b[:,3,None]-boot_a[:,:3]/boot_a[:,3,None]
            summary['comparisons'][weighting][model+'_minus_OBS-EQ0']={
                'primary':model == 'OBS-EQ1',
                'point':{metric:float(b[c]/b[3]-a[c]/a[3]) for c,metric in enumerate(('brier','log_loss','bias'))},
                'intervals':{metric:_interval(delta[:,c]) for c,metric in enumerate(('brier','log_loss','bias'))}}
    return summary, draws.astype('<i8').tobytes()


def _point_scores(matrix):
    total=matrix.sum(axis=0)
    return {metric:float(total[c]/total[3]) if total[3] else None
            for c,metric in enumerate(('brier','log_loss','bias'))}
