"""Fixed artificial candidate-basis experiment; not real speech-effect research.

Reuses S1a synthetic PRE/Q/D2 fixtures. No new ToM, LLM, gameplay execution,
real value model, label reader or external input. All scores are preferences.
"""
import argparse
from collections import Counter
from dataclasses import asdict
from fractions import Fraction as F
import json
from pathlib import Path
import platform
import subprocess

from scripts.compare_observer_dtrs_synthetic import scenarios, synthetic_model
from scripts.speech_planning import generate_candidates
from tests.observer_dtrs.test_speech_runtime import EVIDENCE, POLICY, runtime, source
from werewolf.artifact_io import canonical_json_bytes, sha256_bytes
from werewolf.observer_dtrs.context import contexts_for_pre
from werewolf.observer_dtrs.contextual_policy import contextual_loss, estimate_event
from werewolf.observer_dtrs.decision import decide
from werewolf.observer_dtrs.speech_evidence import (
    ObserverCognition, SelectionMode, process_cognition, select_observer, select_speech,
)
from werewolf.observer_dtrs.candidate_appraisal import (
    BasisKind as B, GateMode as G, PublicReference, CandidateBasis,
    bind_bases, artificial_action_values, appraise_candidates,
)

VERSION = 'observer_dtrs_s2_p0_artificial_paired_appraisal_v1'
COSTS = ('static', 'contextual')
CATALOGS = ('public_only', 'cyclic_cognitive', 'same_action_multiple_bases', 'defer_only')
MODES = ('S1a', *(m.value for m in G))
CAVEAT = ('Artificial preferences and predeclared basis schemas only, not measured speech utility. '
    'Selected-judgment gating is a normative hypothesis. Explicit 3WD and the same continuous '
    'risk argmin are equivalent. S1a ignores this new basis/value policy and is a legacy reference, '
    'not an isolated gating ablation. No outcome, regret-as-game-payoff or win-rate claim.')


def catalogs_for_pre(pre, candidates, selection):
    """Public PRE/seat/candidate-only declaration; no p, Q, risk or selected plan."""
    refs = ((PublicReference(selection.event_id, selection.annotation_digest),)
            if selection.event_id and selection.annotation_digest else ())
    observer = selection.observer_id
    kinds = (B.ASSUME_HIT, B.ASSUME_MISS, B.KEEP_BOTH)
    result = {}
    for name in CATALOGS:
        bases = []
        for index, plan in enumerate(candidates):
            if name == 'public_only' or name == 'same_action_multiple_bases' and index > 0:
                selected_kinds = (B.PUBLIC_ONLY,)
            elif name == 'same_action_multiple_bases':
                selected_kinds = kinds
            elif name == 'defer_only':
                selected_kinds = (B.KEEP_BOTH,)
            else:
                selected_kinds = (kinds[index % 3],)
            for kind in selected_kinds:
                bases.append(CandidateBasis(f'{name}.{index:02}.{kind.name}', plan, kind,
                    None if kind is B.PUBLIC_ONLY else observer, refs))
        result[name] = bind_bases(pre, candidates=candidates, bases=tuple(bases))
    return result


def compare_scene(scene, protocol, model, *, capture_bindings=None):
    # Shared, deterministic fixture setup; no evaluated plan is executed.
    env, recorder, _ = runtime(scene['phase'], scene['target'])
    pre = recorder._pending.prefix
    selection = select_observer(pre, ('player1', 'player2'))
    candidates = generate_candidates(pre)
    catalogs = catalogs_for_pre(pre, candidates, selection)
    values = artificial_action_values(pre, candidates=candidates,
        scores=tuple(F(len(candidates)-j) for j in range(len(candidates))),
        fixture_id='canonical-order-descending-artificial-preference-v1')
    if capture_bindings is not None:
        capture_bindings(dict(scene_id=scene['scene_id'], phase='BOUND_BEFORE_Q_P_APPRAISAL',
            catalogs={n:c.to_record() for n,c in catalogs.items()}, values=values.to_record()))
    context = next(c for c in contexts_for_pre(pre, ('player1', 'player2')) if c.observer_id == selection.observer_id)
    prediction, _ = source(pre, scene['q_r']).probabilities(pre)
    features, estimate, _ = estimate_event(context, prediction, model=model, protocol=protocol, loss=POLICY.static_loss)
    pairing = dict(prefix_digest=pre.prefix_digest,
        q_digest=sha256_bytes(canonical_json_bytes(asdict(prediction))), features=features.to_record(),
        probability=estimate['probability'], model_digest=model.digest, selected_observer=selection.observer_id,
        candidates_digest=values.candidate_digest, action_value_digest=values.digest,
        legacy_strategy=EVIDENCE.to_record(), shared_preparation_budget=dict(
            artificial_Q_materializations=1, D2_predictions=1, artificial_score_entries=len(candidates),
            inference_calls_real_models=0, evaluated_plan_language_calls=0, evaluated_plan_commits=0))
    rows = []
    for cost in COSTS:
        loss = POLICY.static_loss if cost == 'static' else contextual_loss(context, tie_policy=POLICY.static_loss.tie_policy)
        cognition = ObserverCognition(context, features, model.digest, estimate['level'], loss, decide(estimate['probability'], loss))
        evidence = process_cognition(pre, panel=(cognition,))
        for name, catalog in catalogs.items():
            for mode in MODES:
                if mode == 'S1a':
                    chosen = select_speech(pre, evidence, candidates=candidates,
                        strategy=EVIDENCE, mode=SelectionMode.EXPLICIT_THREE_WAY)
                    audit = chosen.to_record()
                    status, plan, basis, count, basis_count = ('LEGACY_S1A_REFERENCE', chosen.plan.public_payload(),
                        None, None, None)
                else:
                    chosen = appraise_candidates(pre, evidence, catalog, mode=G(mode), values=values)
                    audit = chosen.audit
                    status, plan, basis = chosen.status, audit['selected_plan'], audit['selected_basis_id']
                    count, basis_count = audit['assessable_candidate_count'], audit['admitted_basis_count']
                rows.append(dict(scene_id=scene['scene_id'], cost=cost, catalog=name, mode=mode,
                    pairing={**pairing, 'catalog_digest':catalog.digest}, status=status,
                    selected_plan=plan, selected_basis_id=basis,
                    legal_candidates=len(candidates), assessable_candidates=count, admitted_bases=basis_count,
                    cognition=cognition.to_record(), audit=audit))
    return rows


def summarize(rows):
    expected={(s['scene_id'],c,n,m) for s in scenarios() for c in COSTS for n in CATALOGS for m in MODES}
    actual=[(r['scene_id'],r['cost'],r['catalog'],r['mode']) for r in rows]
    if len(actual)!=len(expected) or set(actual)!=expected:
        raise ValueError('missing/duplicate controlled comparison cell')
    indexed={key:r for key,r in zip(actual,rows,strict=True)}
    for scene in scenarios():
        for catalog in CATALOGS:
            group=[indexed[scene['scene_id'],c,catalog,m] for c in COSTS for m in MODES]
            if len({sha256_bytes(canonical_json_bytes(r['pairing'])) for r in group})!=1:
                raise ValueError('unpaired PRE/Q/p/candidates/value/basis/budget')
    arms=[]
    for catalog in CATALOGS:
        for cost in COSTS:
            for mode in MODES:
                group=[indexed[s['scene_id'],cost,catalog,mode] for s in scenarios()]
                arms.append(dict(catalog=catalog,cost=cost,mode=mode,rows=len(group),
                    statuses=dict(Counter(r['status'] for r in group)),
                    selected_intents=dict(Counter(r['selected_plan']['action'] if r['selected_plan'] else 'NONE' for r in group)),
                    legal_candidates_total=sum(r['legal_candidates'] for r in group),
                    assessable_candidates_total=sum(r['assessable_candidates'] for r in group) if mode!='S1a' else None,
                    admitted_bases_total=sum(r['admitted_bases'] for r in group) if mode!='S1a' else None))
    contrasts=[]
    mode_pairs=[(G.RECORD_ONLY.value,G.THREE_WAY.value),
                (G.THREE_WAY.value,G.CONTINUOUS_EQUIVALENCE.value),
                (G.THREE_WAY.value,G.FORCED_BINARY.value),('S1a',G.THREE_WAY.value)]
    for catalog in CATALOGS:
        comparisons=[((cost,left),(cost,right)) for cost in COSTS for left,right in mode_pairs]
        comparisons += [(('static',mode),('contextual',mode)) for mode in MODES]
        for left,right in comparisons:
            pairs=[(indexed[s['scene_id'],left[0],catalog,left[1]],indexed[s['scene_id'],right[0],catalog,right[1]]) for s in scenarios()]
            contrasts.append(dict(catalog=catalog,left=list(left),right=list(right),paired_scenes=30,
                selected_plan_changes=sum(a['selected_plan']!=b['selected_plan'] for a,b in pairs),
                selected_action_type_changes=sum((a['selected_plan']['action'] if a['selected_plan'] else None)
                    != (b['selected_plan']['action'] if b['selected_plan'] else None) for a,b in pairs),
                target_only_changes=sum(a['selected_plan'] is not None and b['selected_plan'] is not None
                    and a['selected_plan']['action']==b['selected_plan']['action']
                    and a['selected_plan']['target']!=b['selected_plan']['target'] for a,b in pairs),
                no_plan_status_changes=sum((a['selected_plan'] is None)!=(b['selected_plan'] is None) for a,b in pairs),
                selected_basis_changes=(None if 'S1a' in (left[1],right[1]) else
                    sum(a['selected_basis_id']!=b['selected_basis_id'] for a,b in pairs)),
                assessable_count_changes=(None if 'S1a' in (left[1],right[1]) else
                    sum(a['assessable_candidates']!=b['assessable_candidates'] for a,b in pairs))))
    return dict(version=VERSION,scope='ARTIFICIAL_ONLY_NOT_G1_G2_OR_GAME_EFFECT',status='PASS',
        rows=len(rows),shared_scenes=30,catalog_scenarios=120,arms_per_catalog_scenario=10,
        complete_grid=True,pairing_ok=True,arms=arms,contrasts=contrasts,caveat=CAVEAT)


def run(output):
    output=Path(output)
    if (not output.is_absolute() or output!=output.resolve() or not output.parent.is_dir()
            or any((p/'.git').exists() for p in (output,*output.parents))):
        raise ValueError('new canonical absolute output outside Git required')
    output.mkdir()
    protocol,model=synthetic_model()  # One artificial D2 fixture fit, no ToM training.
    root=Path(__file__).resolve().parents[1]
    paths=['scripts/compare_observer_dtrs_appraisal.py','werewolf/observer_dtrs/candidate_appraisal.py',
        'werewolf/observer_dtrs/speech_evidence.py','werewolf/observer_dtrs/contextual_policy.py',
        'werewolf/observer_dtrs/decision.py','scripts/speech_planning.py','scripts/compare_observer_dtrs_synthetic.py',
        'tests/observer_dtrs/test_speech_runtime.py']
    plan=dict(version=VERSION,scope='ARTIFICIAL_ONLY',scenarios=scenarios(),costs=COSTS,catalogs=CATALOGS,modes=MODES,
        shared_s1a_strategy=EVIDENCE.to_record(),static_loss=POLICY.static_loss.to_record(),
        contextual_rule='existing contextual_loss unchanged',preference_rule='score=N-canonical_candidate_index; no cognitive-risk mixing',
        basis_rules='public_all; cyclic_HIT/MISS/BOTH; top_plan_three_bases_and_other_public; all_DEFER',
        gate_rule='explicit selected judgment including original tie order; public bases ungated',
        no_true_value_source=True,model_digest=model.digest,protocol_digest=protocol.digest,caveat=CAVEAT,
        source=dict(base_head=subprocess.check_output(['git','--no-optional-locks','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
            files={p:sha256_bytes((root/p).read_bytes()) for p in paths}))
    (output/'plan.json').write_bytes(canonical_json_bytes(plan))
    rows=[]
    with (output/'bindings-before-appraisal.jsonl').open('x') as bound_log, (output/'rows.jsonl').open('x') as log:
        def capture(value):
            bound_log.write(json.dumps(value,ensure_ascii=False,sort_keys=True)+'\n');bound_log.flush()
        for scene in plan['scenarios']:
            for row in compare_scene(scene,protocol,model,capture_bindings=capture):
                rows.append(row);log.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n');log.flush()
    summary=summarize(rows)
    (output/'summary.json').write_bytes(canonical_json_bytes(summary))
    (output/'execution.json').write_bytes(canonical_json_bytes(dict(python=platform.python_version(),
        platform=platform.platform(),only_fixture_initialization_and_pure_selection=True)))
    files={p.name:dict(bytes=p.stat().st_size,sha256=sha256_bytes(p.read_bytes())) for p in output.iterdir()}
    (output/'manifest.json').write_bytes(canonical_json_bytes(dict(version=VERSION,files=files)))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    result=run(parser.parse_args().output)
    print(json.dumps(dict(status=result['status'],rows=result['rows'],scope=result['scope']),sort_keys=True))
