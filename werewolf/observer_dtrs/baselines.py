"""Frozen continuous-r Logistic and EQ1-bound empirical frequency ablation."""
from bisect import bisect_left
from collections import Counter
import math

import numpy as np

from werewolf.artifact_io import canonical_json_bytes, sha256_bytes
from werewolf.observer_dtrs.estimation import condition, STATES, train_edges, cell_stats

VERSION = 'observer_dtrs_secondary_models_v1'
SOLVER = {'algorithm': 'deterministic_damped_newton_float64', 'max_iterations': 200,
          'gradient_tolerance': 1e-9, 'armijo': 1e-4, 'max_backtracks': 50,
          'roundoff_rule': '32eps_objective_tolerance_only_if_decrement_tiny_and_gradient_strictly_improves'}


def design(features, mean, std):
    state = condition(features)
    z = (features.r - mean) / std if std else 0.
    return [1., *[float(state == s) for s in STATES[1:]],
            z if features.nW == 1 else 0., z if features.nW == 2 else 0.]


def sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    result = np.empty_like(z)
    positive = z >= 0
    result[positive] = 1 / (1 + np.exp(-z[positive]))
    e = np.exp(z[~positive])
    result[~positive] = e / (1 + e)
    return result


def objective(beta, X, y, weights, tau):
    """weights sum to one, matching sum(u*loss)/G; global intercept unpenalized."""
    z = X @ beta
    p = sigmoid(z)
    penalty = np.array([0., *([tau] * (X.shape[1] - 1))])
    loss = float(weights @ (np.logaddexp(0, z) - y * z) + .5 * (penalty * beta) @ beta)
    gradient = X.T @ (weights * (p - y)) + penalty * beta
    hessian = X.T @ ((weights * p * (1 - p))[:, None] * X) + np.diag(penalty)
    return loss, gradient, hessian


def fit_logistic(rows, *, protocol, tau, weighting, source):
    rows = tuple(sorted(rows, key=lambda r: (r.game_id, r.sample_id)))
    if not rows or tau not in protocol.record['secondary_logistic']['tau_grid'] or weighting not in ('game_equal', 'record_equal'):
        raise ValueError('invalid frozen Logistic request')
    sizes = Counter(r.game_id for r in rows)
    if dict(sizes) != source['game_sizes'] or len({r.sample_id for r in rows}) != len(rows):
        raise ValueError('Logistic requires exact complete training games')
    u = np.array([1 / sizes[r.game_id] if weighting == 'game_equal' else len(sizes) / len(rows) for r in rows])
    weights = u / len(sizes)
    r = np.array([r.features.r for r in rows])
    mean = float(weights @ r)
    variance = float(weights @ ((r - mean) ** 2))
    std = math.sqrt(variance) if len(set(r)) > 1 else 0.
    y = np.array([r.y for r in rows], dtype=np.float64)
    result = {'schema_version': VERSION, 'model_id': 'OBS-LR-r', 'protocol_digest': protocol.digest,
        'parameters': {'tau': tau, 'weighting': weighting}, 'source': source,
        'training_fingerprint': sha256_bytes(canonical_json_bytes([
            [r.sample_id, r.game_id, r.features.to_record(), r.y] for r in rows])),
        'mean_r': mean, 'std_r': std, 'solver': SOLVER}
    if len(set(y)) == 1:
        result.update(constant=float((u @ y + .5) / (len(sizes) + 1)),
                      coefficients=None, convergence={'reason': 'single_class_root_constant', 'iterations': 0})
        return result
    X = np.array([design(r.features, mean, std) for r in rows])
    beta = np.zeros(6)
    converged = False
    for iteration in range(SOLVER['max_iterations'] + 1):
        loss, gradient, hessian = objective(beta, X, y, weights, tau)
        norm = float(np.max(np.abs(gradient)))
        if norm <= SOLVER['gradient_tolerance']:
            converged = True
            break
        if iteration == SOLVER['max_iterations']:
            break
        direction = np.linalg.solve(hessian, gradient)
        descent = float(gradient @ direction)
        step = 1.
        for _ in range(SOLVER['max_backtracks']):
            candidate = beta - step * direction
            new_loss,new_gradient,_=objective(candidate, X, y, weights, tau)
            tolerance=32*np.finfo(np.float64).eps*max(1.,abs(loss))
            roundoff=(step*descent<=tolerance and new_loss<=loss+tolerance
                      and float(np.max(np.abs(new_gradient)))<norm)
            if new_loss <= loss - SOLVER['armijo'] * step * descent or roundoff:
                beta = candidate
                break
            step /= 2
        else:
            raise RuntimeError('Logistic line search failed; no fallback/model substitution')
    if not converged or not np.isfinite(beta).all():
        raise RuntimeError('Logistic did not converge; stop frozen experiment')
    result.update(constant=None, coefficients=beta.tolist(), convergence={
        'reason': 'gradient_tolerance', 'iterations': iteration, 'gradient_inf': norm, 'objective': loss})
    return result


def empirical_model(eq1):
    """No new tuning: unshrunk S/E on the selected EQ1 training partition/bins."""
    record = eq1.record
    if record['model_id'] != 'OBS-EQ1':
        raise ValueError('OBS-EF is bound to selected EQ1, never Phase2')
    record['schema_version'] = VERSION
    record['model_id'] = 'OBS-EF'
    record['parent_EQ1_digest'] = eq1.digest
    record['ablation_rule'] = 'same_selected_EQ1_bins_weights_unsmoothed_frequency_nearest_observed_parent'
    nodes = [record['root'], *record['wolf_nodes'].values()]
    for state in record['state_nodes'].values():
        nodes += [state, *state['leaves']]
    for node in nodes:
        s = node['stats']
        node['probability'] = s['positive_mass'] / s['mass']
    return record


def predict_many(model, features):
    """Batch legal prediction; deserialize once, never receive supervised rows."""
    m = model.record if hasattr(model, 'record') else model
    result = []
    if m['model_id'] == 'OBS-LR-r':
        for f in features:
            condition(f)
            p = m['constant'] if m['coefficients'] is None else float(sigmoid(
                np.array([design(f, m['mean_r'], m['std_r'])]) @ np.array(m['coefficients']))[0])
            result.append({'probability': p, 'level': 'single_class_root' if m['coefficients'] is None else 'logistic'})
    else:
        for f in features:
            phase, nW = condition(f)
            node, level = m['root'], 'global'
            if str(nW) in m['wolf_nodes']:
                node, level = m['wolf_nodes'][str(nW)], 'nW'
            state = m['state_nodes'].get(f'{phase}/{nW}')
            if state:
                node, level = state, 'phase_nW'
                if state['leaves']:
                    node, level = state['leaves'][bisect_left(state['edges'], f.r)], 'r_bin'
            result.append({'probability': node['probability'], 'level': level})
    if any(not math.isfinite(r['probability']) or not 0 <= r['probability'] <= 1 for r in result):
        raise ValueError('invalid predicted event probability')
    return result


def bin_audit(model, rows, protocol):
    """Record initial/final edges and deterministic X-only merge decisions."""
    m = model.record
    sizes = Counter(r.game_id for r in rows)
    weights = {r.sample_id: 1 / sizes[r.game_id] if m['parameters']['weighting'] == 'game_equal'
               else len(sizes) / len(rows) for r in rows}
    audits = {}
    for key, state in m['state_nodes'].items():
        group = [r for r in rows if f'{r.features.phase}/{r.features.nW}' == key]
        spec = protocol.record['histogram']
        initial = list(train_edges(group, weights, m['parameters']['requested_bins'],
                        {**spec, 'min_distinct_games': 0, 'min_kish_game_mass_effective_support': 0}))
        edges, trace = initial[:], []
        while edges and edges != state['edges']:
            cells = [[r for r in group if bisect_left(edges, r.features.r) == j] for j in range(len(edges) + 1)]
            stats = [cell_stats(c, weights) for c in cells]
            bad = next(j for j, s in enumerate(stats) if s.games < spec['min_distinct_games'] or
                       s.kish_games + spec['support_float_tolerance'] < spec['min_kish_game_mass_effective_support'])
            delete = 0 if bad == 0 else bad - 1 if bad == len(cells) - 1 else (
                bad - 1 if stats[bad - 1].mass <= stats[bad + 1].mass else bad)
            trace.append({'failing_bin': bad, 'games': stats[bad].games, 'kish': stats[bad].kish_games,
                          'removed_edge': edges.pop(delete)})
        if edges != state['edges']:
            raise ValueError('bin merge audit does not reconstruct fitted edges')
        audits[key] = {'initial_edges': initial, 'final_edges': edges, 'merges': trace}
    return audits
