from copy import deepcopy
from dataclasses import replace
import math

import numpy as np
import pytest
import torch

from tests.tom_llm.conftest import predictions_for, empty_and_full_reports
from werewolf.tom_llm.reference import open_reference
from werewolf.tom_llm.evaluation import compare
from werewolf.tom_llm.output import parse_output
from werewolf.tom_llm.common import identity
from werewolf.tom.scoring import paired_summary


@pytest.mark.parametrize('sealed_reference',[empty_and_full_reports],indirect=True)
def test_approved_sets_include_derived_baselines_tiers_and_common_denominators(sealed_reference):
    from collections import Counter
    from werewolf.tom_llm.reference import raw_set_labels,projection_source
    from werewolf.tom_llm.evaluation import compare_sets
    ref,pub,*_=sealed_reference
    source=projection_source(ref,engineering_only=True)
    assert source['status']=='ENGINEERING_ONLY_NOT_FORMAL_OOF'
    with pytest.raises(ValueError,match='fixed historical OOF'):
        projection_source(ref,engineering_only=False)
    labels=raw_set_labels(ref,pub.path,ref.game_ids)
    predictions={'qwen3':_set_rows(ref),'yi':_set_rows(ref,invalid=True),'mistral':_set_rows(ref)}
    result=compare_sets(ref,predictions,ref.game_ids,labels,engineering_only=True)
    full=result['populations']['full_primary_raw'];nu=result['populations']['primary_comparable_nu']
    assert result['comparison_complete'] is True
    assert len(full['paired'])==10 and len(nu['paired'])==15
    assert Counter(p['tier'] for p in full['paired'])=={'simple_baseline':6,'exploratory':4}
    assert Counter(p['tier'] for p in nu['paired'])=={'key_method':3,'simple_baseline':8,'exploratory':4}
    assert {c['row_count'] for c in full['cells'].values()}=={len(ref.primary_rows)}
    assert full['cells']['yi']['metrics']['valid_output_rate']['point']==0
    assert nu['cells']['Always Empty']['metrics']['set_f1']['point']==0
    assert full['cells']['Always All-Six']['metrics']['valid_output_rate']['point']==1
    assert all(v['model_run_seal'] is None for v in result['derived_methods'].values())
    for name,group in result['populations'].items():
        for cell in group['cells'].values():
            assert cell['metrics']['set_f1']['point']==pytest.approx(
                np.mean([s['set_f1'] for s in cell['game_scores'].values()]))
        for pair in group['paired']:
            left=group['cells'][pair['left']]['metrics']['set_f1']
            right=group['cells'][pair['right']]['metrics']['set_f1']
            assert pair['metrics']['set_f1']['point']==pytest.approx(left['point']-right['point'])
    from werewolf.tom_llm.common import digest
    from werewolf.tom.protocol import bootstrap_indices
    from werewolf.tom_llm.policy import POLICY
    from werewolf.artifact_io import sha256_bytes
    for group in result['populations'].values():
        expected=bootstrap_indices(group['game_ids'],10000,POLICY['bootstrap']['seed'])
        assert group['bootstrap_indices_digest']==sha256_bytes(expected.tobytes())
    with pytest.raises(TypeError):
        compare_sets(ref,predictions,ref.game_ids,labels,engineering_only=True,probability_to_set_rule={})


@pytest.mark.parametrize('change',['canonical','architecture','checkpoint','prediction'])
def test_projection_source_rejects_unbound_oof(sealed_reference,change):
    from werewolf.tom_llm.reference import projection_source
    ref,*_=sealed_reference
    changed=deepcopy(ref.provenance)
    if change=='canonical':changed['canonical_jsonl_verified']=False
    elif change=='architecture':changed['architecture']='qwen2'
    elif change=='checkpoint':changed['checkpoint_digests']['0']='a'*64
    else:
        ref=replace(ref,prediction_digests={**ref.prediction_digests,'0':'a'*64})
    with pytest.raises(ValueError,match='OOF'):
        projection_source(replace(ref,provenance=changed),engineering_only=True)


@pytest.mark.parametrize('change',[None,'training_source','evaluation_source','training_seal',
                                  'checkpoint','prediction','temporal','reference'])
def test_fixed_oof_contract_with_mock_identity_adapter(sealed_reference,monkeypatch,change):
    """隔离测试固定合同的每项身份；不把人工行声明为真实封存 OOF。"""
    from werewolf.tom_llm.reference import fixed_oof_contract,projection_source
    from werewolf.tom_llm import multimodel_evaluation
    from werewolf.tom_llm.common import digest
    fixed=fixed_oof_contract();ref,*_=sealed_reference
    provenance={k:deepcopy(v) for k,v in fixed.items() if k!='reference'}
    provenance['canonical_jsonl_verified']=True
    binding=deepcopy(fixed['reference'])
    ref=replace(ref,prediction_digests=deepcopy(binding['prediction_digests']),provenance=provenance)
    if change in ('training_source','evaluation_source'):
        provenance[change]['source_revision']='a'*40
    elif change=='training_seal':provenance['training_seal_digest']='a'*64
    elif change=='checkpoint':provenance['checkpoint_digests']['0']='a'*64
    elif change=='prediction':ref.prediction_digests['0']=binding['prediction_digests']['0']='a'*64
    elif change=='temporal':provenance['temporal_condition']='none'
    elif change=='reference':binding['seal_digest']='a'*64
    rows=tuple({**r,'checkpoint_digest':provenance['checkpoint_digests'][str(r['fold'])],
                'reference_prediction_digest':ref.prediction_digests[str(r['fold'])]} for r in ref.rows)
    ref=replace(ref,rows=rows)
    monkeypatch.setattr(multimodel_evaluation,'reference_binding',lambda actual:binding)
    if change is None:
        report=projection_source(ref,engineering_only=False)
        assert report['actual']==fixed and report['actual_digest']==digest(fixed)
        assert report['status']=='FIXED_HISTORICAL_OOF_VERIFIED'
    else:
        with pytest.raises(ValueError,match='OOF'):projection_source(ref,engineering_only=False)


@pytest.mark.parametrize('players,gold,f1,jaccard,exact', [
    ([],[],1.,1.,1.), ([],['player2'],0.,0.,0.), (['player2'],[],0.,0.,0.),
    (['player2'],['player2'],1.,1.,1.),
    (['player2','player3'],['player3','player4'],.5,1/3,0.)])
def test_set_metric_definitions(players,gold,f1,jaccard,exact):
    import json
    from werewolf.tom_llm.output import parse_set_output
    from werewolf.tom_llm.evaluation import set_scores
    prediction = parse_set_output(json.dumps({'suspected_players':players}),0).to_record()
    scores = set_scores(prediction,gold)
    assert scores['set_f1'] == f1 and scores['jaccard'] == jaccard
    assert scores['exact_set_match'] == exact and scores['valid_output_rate'] == 1
    if players and gold:
        assert scores['set_precision'] == len(set(players)&set(gold))/len(players)
        assert scores['set_recall'] == len(set(players)&set(gold))/len(gold)
    failed = parse_set_output('invalid',0).to_record()
    assert set(set_scores(failed,gold).values()) == {0.}


@pytest.mark.parametrize('sealed_reference', [empty_and_full_reports], indirect=True)
def test_raw_empty_and_full_sets_have_distinct_gold_identity(sealed_reference):
    from werewolf.tom_llm.reference import raw_set_labels
    ref,publication,*_ = sealed_reference
    labels = raw_set_labels(ref,publication.path,ref.game_ids)
    raw = {identity(r):r for r in labels['labels']}
    empty = next(r for r in ref.primary_rows if raw[identity(r)]['suspected_players'] == [])
    full = next(r for r in ref.primary_rows if len(raw[identity(r)]['suspected_players']) == 6)
    assert sum(q > 0 for q in empty['q']) == sum(q > 0 for q in full['q']) == 6
    assert raw[identity(empty)]['observation_digest'] != raw[identity(full)]['observation_digest']
    assert labels['publication_digest'] == ref.publication_digest


def _set_rows(reference, *, invalid=False):
    from werewolf.tom_llm.output import parse_set_output
    rows = predictions_for(reference)
    for row in rows:
        content = 'invalid' if invalid else '{"suspected_players":[]}'
        row['prediction'] = parse_set_output(content,row['observer']).to_record()
    return rows


def test_set_comparison_common_rows_failure_denominators_and_game_bootstrap(sealed_reference):
    from werewolf.tom_llm.reference import raw_set_labels
    from werewolf.tom_llm.evaluation import compare_sets
    from werewolf.tom_llm.common import digest
    from werewolf.tom.protocol import bootstrap_indices
    from werewolf.tom_llm.policy import POLICY
    from werewolf.artifact_io import sha256_bytes
    ref,publication,*_ = sealed_reference
    labels = raw_set_labels(ref,publication.path,ref.game_ids)
    predictions = {'qwen3':_set_rows(ref), 'yi':_set_rows(ref,invalid=True), 'mistral':_set_rows(ref)}
    result = compare_sets(ref,predictions,ref.game_ids,labels,engineering_only=True)
    nu = result['populations']['primary_comparable_nu']
    assert len(nu['paired']) == 15 and set(nu['cells']) == {'qwen3','yi','mistral','Observer-ToM','Always Empty','Always All-Six'}
    assert {c['row_count'] for c in nu['cells'].values()} == {nu['row_count']}
    assert nu['cells']['yi']['metrics']['valid_output_rate']['point'] == 0
    assert nu['cells']['qwen3']['metrics']['valid_output_rate']['point'] == 1
    indices = bootstrap_indices(nu['game_ids'],10000,POLICY['bootstrap']['seed'])
    assert nu['bootstrap_indices_digest'] == sha256_bytes(indices.tobytes())
    for cell in nu['cells'].values():
        assert cell['metrics']['set_f1']['point'] == pytest.approx(np.mean([v['set_f1'] for v in cell['game_scores'].values()]))
    # 仅验证不可恢复原集合的接口统计；不把改造的人工资料当作真实来源证据。
    missing = deepcopy(labels)
    missing['labels'][0].update(status='not_recoverable',suspected_players=None)
    missing['labels_digest'] = digest(missing['labels'])
    partial = compare_sets(ref,predictions,ref.game_ids,missing,engineering_only=True)
    assert partial['unobservable_or_unrecoverable_rows'] == 1
    assert partial['populations']['full_primary_raw']['row_count'] == len(ref.primary_rows)-1
    predictions['yi'].pop()
    with pytest.raises(ValueError,match='missing'):
        compare_sets(ref,predictions,ref.game_ids,labels,engineering_only=True)


def test_reference_reads_full_semantics_without_model_inference(sealed_reference, monkeypatch):
    ref, publication, root, report = sealed_reference
    before = {str(p):p.read_bytes() for p in root.parent.parent.rglob("*") if p.is_file()}
    monkeypatch.setattr(torch.nn.Module, "_call_impl", lambda *_a, **_k:pytest.fail("model inference forbidden"))
    loaded = open_reference(publication.path, root, expected_seal_digest=report.manifest_digest)
    assert loaded.primary_rows == ref.primary_rows
    assert loaded.game_ids == tuple(sorted(publication.public_view.game_ids))
    assert all(r["observer_alive"] and r["label_observed"] and r["primary_eligible"] for r in loaded.primary_rows)
    assert set(loaded.prediction_digests) == set(map(str, range(5)))
    assert before == {str(p):p.read_bytes() for p in root.parent.parent.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="anchor"):
        open_reference(publication.path, root, expected_seal_digest="0"*64)


@pytest.mark.parametrize("change", ["payload", "manifest", "missing_fold", "bootstrap"])
def test_reference_artifact_corruption_fails_closed(sealed_reference, change):
    ref, publication, root, report = sealed_reference
    if change == "payload":
        target = root/"folds/qwen3/0/predictions.jsonl"
        target.write_bytes(target.read_bytes()+b"{}\n")
    elif change == "manifest":
        target = root/"folds/qwen3/0/manifest.json"
        target.write_bytes(target.read_bytes().replace(b'"fold":0',b'"fold":1'))
    elif change == "missing_fold":
        (root/"folds/qwen3/0/predictions.jsonl").unlink()
    else:
        (root/"seal/bootstrap_indices.bin").write_bytes(b"bad")
    with pytest.raises((ValueError, OSError)):
        open_reference(publication.path, root, expected_seal_digest=report.manifest_digest)


@pytest.mark.parametrize("change", ["missing", "duplicate", "extra", "prefix", "plan", "fold", "observer", "shift", "temporal", "probability"])
def test_exact_row_join_never_intersects(sealed_reference, change):
    ref, *_ = sealed_reference
    rows = predictions_for(ref)
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows.append(deepcopy(rows[0]))
    elif change == "extra":
        extra = deepcopy(rows[0]); extra["boundary_id"]="foreign"; rows.append(extra)
    elif change == "prefix":
        rows[0]["prefix_digest"]="0"*64
    elif change == "plan":
        rows[0]["plan_digest"]="0"*64
    elif change == "fold":
        rows[0]["fold"]=(rows[0]["fold"]+1)%5
    elif change == "observer":
        rows[0]["observer"]=True
    elif change == "shift":
        rows[0]["canonical_shift"]=1
    elif change == "temporal":
        rows[0]["temporal_condition"]="implicit"
    else:
        rows[0]["prediction"]["probability"][rows[0]["observer"]]=.1
    with pytest.raises(ValueError):
        compare(ref, rows, ref.game_ids)


def test_paired_zero_macro_and_failure_rows_are_kept(sealed_reference):
    ref, *_ = sealed_reference
    result = compare(ref, predictions_for(ref), ref.game_ids)
    assert result["Delta_KL"]["point"] == pytest.approx(0, abs=1e-12)
    assert result["cells"]["llm"]["row_count"] == len(ref.primary_rows)
    assert result["bootstrap_indices_digest"] == ref.bootstrap_digest
    failed = compare(ref, predictions_for(ref, failure=True), ref.game_ids)
    assert failed["failures"]["primary_row_failure_rate"] == 1
    assert failed["cells"]["llm"]["row_count"] == len(ref.primary_rows)
    assert failed["Delta_KL"]["point"] == pytest.approx(0, abs=1e-12)
    assert result["failures"]["legal_uniform_rows"] == len(ref.primary_rows)
    assert failed["failures"]["legal_uniform_rows"] == 0


def test_unequal_game_sizes_and_difference_sign_use_shared_scoring(sealed_reference):
    ref, *_ = sealed_reference
    rows = predictions_for(ref)
    # Make two held-out games have different error distributions; independently compute exact game deltas.
    for row in rows:
        if row["game_id"] == ref.game_ids[0]:
            observer=row["observer"]
            seats=[j for j in range(7) if j!=observer]
            import json
            values={f"player{j+1}":float(j==seats[-1]) for j in seats}
            row["prediction"]=parse_output(json.dumps({"probabilities":values}),observer).to_record()
    result=compare(ref,rows,ref.game_ids)
    llm=[result["cells"]["llm"]["game_scores"][g]["kl"] for g in ref.game_ids]
    qwen=[result["cells"]["qwen3_oof"]["game_scores"][g]["kl"] for g in ref.game_ids]
    assert result["Delta_KL"] == paired_summary(llm,qwen,ref.bootstrap_indices,.95)
    assert result["Delta_KL"]["point"] > 0
    assert result["cells"]["llm"]["kl"]["point"] == pytest.approx(np.mean(llm))
    # Exercise subset draw construction, with no reuse/slicing of the full draw matrix.
    selected=ref.game_ids[:2]
    subrows=[row for row in rows if row["game_id"] in selected]
    subset=compare(ref,subrows,selected)
    assert subset["cells"]["llm"]["game_count"] == 2
    assert subset["bootstrap_indices_digest"] != ref.bootstrap_digest


def test_top1_nu_game_coverage_no_uniform_game_zero_fill(sealed_reference):
    ref, *_ = sealed_reference
    # Use a synthetic all-uniform target reference; NU is absent, never encoded as accuracy=0.
    changed=[]
    for row in ref.rows:
        changed.append({**row,"q":[0. if j==row["observer"] else 1/6 for j in range(7)]})
    primary=tuple(r for r in changed if r["label_observed"] and r["primary_eligible"])
    uniform_ref=replace(ref,rows=tuple(changed),primary_rows=primary)
    result=compare(uniform_ref,predictions_for(uniform_ref),uniform_ref.game_ids)
    assert result["nu_game_ids"] == [] and result["nu_bootstrap_indices_digest"] is None
    assert all(c["top1_nu"] is None and c["nu_game_count"]==0 for c in result["cells"].values())


def test_shared_row_score_argmax_tie_is_preserved(sealed_reference):
    ref, *_=sealed_reference
    result=compare(ref,predictions_for(ref),ref.game_ids)
    expected={}
    for row in ref.primary_rows:
        if sum(q>0 for q in row["q"]) < 6:
            target=next(j for j in range(7) if j!=row["observer"])
            expected.setdefault(row["game_id"],[]).append(int(row["q"][target]>0))
    point=np.mean([np.mean(v) for v in expected.values()]) if expected else None
    if point is not None:
        assert result["cells"]["llm"]["top1_nu"]["point"] == pytest.approx(point)


def test_wrong_canonical_targets_and_fold_masks_are_rejected(sealed_reference,monkeypatch):
    ref,publication,root,report=sealed_reference
    from werewolf.tom_llm import reference as adapter
    original=adapter.CanonicalToMDataset
    def changed(*args,**kwargs):
        data=original(*args,**kwargs)
        sample=data.samples[0]
        q=sample.q.clone(); q[0,1]=.123
        data.samples[0]=replace(sample,q=q)
        return data
    monkeypatch.setattr(adapter,"CanonicalToMDataset",changed)
    with pytest.raises(ValueError,match="semantics"):
        open_reference(publication.path,root,expected_seal_digest=report.manifest_digest)


def test_game_macro_is_not_observer_row_weighted(sealed_reference):
    ref,*_=sealed_reference
    extra=[]
    for row in ref.rows:
        if row["game_id"]==ref.game_ids[0]:
            for count in range(4):
                extra.append({**row,"boundary_id":row["boundary_id"]+f"-extra-{count}"})
    rows=ref.rows+tuple(extra)
    primary=tuple(r for r in rows if r["label_observed"] and r["primary_eligible"])
    unbalanced=replace(ref,rows=rows,primary_rows=primary,
                       alive_keys=frozenset(identity(r) for r in rows if r["observer_alive"]))
    predictions=predictions_for(unbalanced)
    for r in predictions:
        if r["game_id"]==ref.game_ids[0]:
            import json
            seats=[j for j in range(7) if j!=r["observer"]]
            content=json.dumps({"probabilities":{f"player{j+1}":int(j==seats[-1]) for j in seats}})
            r["prediction"]=parse_output(content,r["observer"]).to_record()
    result=compare(unbalanced,predictions,unbalanced.game_ids)
    scores=result["cells"]["llm"]["game_scores"]
    macro=np.mean([s["kl"] for s in scores.values()])
    weighted=sum(s["kl"]*s["row_count"] for s in scores.values())/sum(s["row_count"] for s in scores.values())
    assert result["cells"]["llm"]["kl"]["point"]==pytest.approx(macro)
    assert macro!=pytest.approx(weighted)




def test_injected_supervision_cannot_override_reference(sealed_reference):
    ref,*_=sealed_reference
    for field,value in (("q",[0,1,0,0,0,0,0]),("observer_alive",False),("primary_eligible",True)):
        rows=predictions_for(ref); rows[0][field]=value
        with pytest.raises(ValueError,match="supervision"):
            compare(ref,rows,ref.game_ids)


def test_partial_nu_game_set_never_zero_fills_absent_games(sealed_reference):
    ref,*_=sealed_reference
    changed=[]
    for row in ref.rows:
        if row["game_id"] == ref.game_ids[0]:
            row={**row,"q":[0. if j==row["observer"] else 1/6 for j in range(7)]}
        changed.append(row)
    primary=tuple(r for r in changed if r["label_observed"] and r["primary_eligible"])
    partial=replace(ref,rows=tuple(changed),primary_rows=primary)
    result=compare(partial,predictions_for(partial),partial.game_ids)
    assert ref.game_ids[0] not in result["nu_game_ids"]
    assert result["nu_game_ids"]
    for cell in result["cells"].values():
        assert cell["nu_game_count"] == len(result["nu_game_ids"])


def _mock_epoch35():
    from types import SimpleNamespace

    from werewolf.tom.dataset import ExperimentCapacity

    class MockEpoch35(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0
            self.capacity = ExperimentCapacity(1024)
            self.temporal = SimpleNamespace(day=range(6), artifact_digest="d" * 64)
            self.graph = {"architecture": "synthetic"}
            logits = torch.zeros(7, 7)
            seats = torch.arange(7)
            logits[seats, seats] = -torch.inf
            logits[seats, (seats + 1) % 7] = 2
            self.logp = logits.log_softmax(dim=-1)

        def forward(self, **public):
            assert torch.is_inference_mode_enabled() and not self.training
            assert set(public) == {"event_ids", "source_ids", "action_ids", "target_ids",
                                   "day_ids", "phase_ids", "attention_mask"}
            assert public["attention_mask"].any()
            self.calls += 1
            return self.logp.unsqueeze(0)

    return MockEpoch35().eval()


def test_epoch35_v1_comparison_cli_preserves_seals_and_pairs_scores(sealed_reference, tmp_path, monkeypatch, capsys):
    import json

    from werewolf.artifact_io import canonical_json_bytes, canonical_jsonl_bytes, publish_artifact, sha256_bytes
    from werewolf.cli import main
    from werewolf.tom import makto
    from werewolf.tom.protocol import bootstrap_indices
    from werewolf.tom_llm import baseline_analysis as analysis, operations
    from werewolf.tom_llm.common import digest
    from werewolf.tom_llm.multimodel import COHORT_VERSION
    from werewolf.tom_llm.multimodel_evaluation import reference_binding
    from werewolf.tom_llm.policy import POLICY
    from werewolf.tom_llm.task import VERSION, public_tasks

    ref, publication, evaluation, _ = sealed_reference
    prefixes = [p for game in publication.public_view.games for p in game.authoritative_pre_prefixes]
    tasks = public_tasks(prefixes, {g: fold["fold"] for fold in ref.folds for g in fold["held_out_game_ids"]})
    # 将重复 PRE 的不同观察者分开，检验前向复用和原始请求顺序。
    tasks = tasks[::2] + tasks[1::2]
    key_digest = digest([identity(t) for t in tasks])
    task_digests = [digest(t) for t in tasks]
    reference = reference_binding(ref)
    selection = {"game_ids": list(ref.game_ids)}
    cohort = publish_artifact(tmp_path / "cohort", manifest_fields={
        "artifact_type": "llm_multimodel_public_cohort", "schema_version": COHORT_VERSION,
        "task": {"schema_version": VERSION}, "reference": reference, "selection": selection,
        "requests_digest": digest(tasks), "row_identity_digest": key_digest,
    }, files={"requests.jsonl": canonical_jsonl_bytes(tasks)})

    def record(path, kind, body):
        payload = canonical_json_bytes(body)
        return publish_artifact(path, manifest_fields={"artifact_type": kind,
            "schema_version": analysis.V1_EXPERIMENT, "record_digest": sha256_bytes(payload)},
            files={"record.json": payload})

    paths, seals = {}, {}
    model_spec_digest = "b" * 64
    historical_source = {"source_revision": analysis.V1_SOURCE_REVISION}
    for key in analysis.V1_RUN_SEALS:
        root = tmp_path / key
        paths[key] = root
        contract = record(root / "contract", "mm_baseline_experiment", {
            "source": historical_source, "protocol": {"schema_version": analysis.V1_EXPERIMENT},
            "run_id": key, "mode": "formal", "models": {key: model_spec_digest},
            "cohort_root": str(cohort.path), "cohort_digest": cohort.manifest_digest, "selection": selection,
            "reference": reference, "publication": str(publication.path), "evaluation": str(evaluation),
            "games": list(ref.game_ids), "game_count": len(ref.game_ids), "request_count": len(tasks),
            "row_identity_digest": key_digest, "task_digests": task_digests,
        })
        indexed = {identity(row): row for row in predictions_for(ref, failure=key == "yi")}
        rows = [{**indexed[identity(t)], "task_digest": td} for t, td in zip(tasks, task_digests, strict=True)]
        prediction = record(root / "models" / key / "seal", "mm_baseline_predictions", {
            "run_digest": contract.manifest_digest, "model": key, "mode": "formal", "formal_result": True,
            "model_spec_digest": model_spec_digest, "task_digests": task_digests, "rows": rows,
        })
        seals[key] = record(root / "seal", "mm_baseline_run_seal", {
            "mode": "formal", "source": historical_source, "run_digest": contract.manifest_digest,
            "models": {key: prediction.manifest_digest},
        }).manifest_digest

    bootstrap = bootstrap_indices(ref.game_ids, 10000, POLICY["bootstrap"]["seed"])
    nu_count = sum(sum(q > 0 for q in row["q"]) < 6 for row in ref.primary_rows)
    monkeypatch.setattr(analysis, "V1_RUN_SEALS", seals)
    monkeypatch.setattr(analysis, "V1_COHORT_DIGEST", cohort.manifest_digest)
    monkeypatch.setattr(analysis, "V1_PRE_COUNT", len(prefixes))
    monkeypatch.setattr(analysis, "V1_PRIMARY_COUNT", len(ref.primary_rows))
    monkeypatch.setattr(analysis, "V1_NU_COUNT", nu_count)
    monkeypatch.setattr(analysis, "V1_PRIMARY_ROWS_DIGEST", digest([identity(r) for r in ref.primary_rows]))
    monkeypatch.setattr(analysis, "V1_BOOTSTRAP_DIGEST", sha256_bytes(bootstrap.tobytes()))
    # 人工封存仅替换正式人口的规模检查，保留真实文件校验和行对齐。
    monkeypatch.setattr(analysis, "require_formal_cohort", lambda *_: None)
    current_source = {"source_revision": "c" * 40}
    monkeypatch.setattr(analysis, "source_identity", lambda: current_source)
    monkeypatch.setattr(operations, "load_config", lambda *_: pytest.fail("LLM service config must stay closed"))
    model = _mock_epoch35()
    loaded = []

    def load_model(checkpoint, temporal):
        loaded.append((checkpoint, temporal))
        return model

    def open_after_prediction(*args, **kwargs):
        assert model.calls == len(prefixes)
        return open_reference(*args, **kwargs)

    monkeypatch.setattr(makto, "load_epoch35_model", load_model)
    monkeypatch.setattr(analysis, "open_reference", open_after_prediction)
    source_files = [p for p in tmp_path.rglob("*") if p.is_file()]
    before = {p: p.read_bytes() for p in source_files}
    destination = tmp_path / "paired"
    args = ["tom-llm", "compare-epoch35", "--checkpoint", str(tmp_path / "checkpoint"),
            "--temporal", str(tmp_path / "temporal"), "--output", str(destination)]
    for key, path in paths.items():
        args.extend(["--sealed-run", f"{key}={path}"])
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["path"] == str(destination)
    assert loaded == [(tmp_path / "checkpoint", tmp_path / "temporal")]
    assert {p: p.read_bytes() for p in source_files} == before
    assert set(p.name for p in destination.iterdir()) == {"manifest.json", "predictions.jsonl", "analysis.json"}

    predicted = [json.loads(line) for line in (destination / "predictions.jsonl").read_bytes().splitlines()]
    assert [identity(row) for row in predicted] == [identity(t) for t in tasks]
    assert [row["task_digest"] for row in predicted] == task_digests
    for row in predicted:
        seat = row["observer"]
        assert set(row) == {*analysis.ROW_FIELDS, *analysis.PROBABILITY_FIELDS, "task_digest",
                            "canonical_shift", "checkpoint_digest", "temporal_condition"}
        assert row["checkpoint_digest"] == makto.EPOCH35_DIGEST
        assert row["probability"] == model.logp[seat].exp().tolist()
        assert row["non_self_log_probability"] == [model.logp[seat, j].item() for j in range(7) if j != seat]

    result = json.loads((destination / "analysis.json").read_bytes())
    assert result["prediction_source"] == result["analysis_source"] == current_source
    assert result["v1_runs"]["yi"]["execution_source"] == historical_source
    assert result["historical_qwen3_oof"] == ref.provenance
    assert result["failure_counts"]["yi"]["primary_failure_types"] == {"invalid_json": len(ref.primary_rows)}
    assert result["failure_counts"]["yi"]["public_alive_failure_types"] == {"invalid_json": len(tasks)}
    assert set(result["cells"]) == {"epoch35", "uniform_non_self", "historical_qwen3_oof", "qwen3", "yi", "mistral"}
    expected = compare(ref, predictions_for(ref, failure=True), ref.game_ids)
    assert result["cells"]["yi"]["row_count"] == expected["cells"]["llm"]["row_count"] == len(ref.primary_rows)
    assert result["cells"]["yi"]["nu_row_count"] == nu_count
    for metric in ("kl", "tv", "top1_nu"):
        uniform = result["cells"]["uniform_non_self"][metric]["point"]
        assert result["cells"]["yi"][metric]["point"] == pytest.approx(uniform)
        assert uniform == pytest.approx(expected["cells"]["uniform_non_self"][metric]["point"])
    large = result["cells"]["epoch35"]
    yi = result["cells"]["yi"]
    for metric, game_metric in (("kl", "kl"), ("tv", "total_variation")):
        assert result["paired"]["yi"][metric] == paired_summary(
            [yi["game_scores"][g][game_metric] for g in ref.game_ids],
            [large["game_scores"][g][game_metric] for g in ref.game_ids], bootstrap, .95)
    assert result["paired"]["yi"]["top1_nu"] == paired_summary(
        [large["nu_game_scores"][g] for g in ref.game_ids], [yi["nu_game_scores"][g] for g in ref.game_ids], bootstrap, .95)

    with pytest.raises(ValueError, match="output exists"):
        main(args)
    with pytest.raises(ValueError, match="inside frozen evidence"):
        analysis.analyze_epoch35_v1(paths, tmp_path / "checkpoint", tmp_path / "temporal", paths["yi"] / "extra")
    with pytest.raises(ValueError, match="seal/inventory mismatch"):
        analysis._read_v1_comparison_inputs({**paths, "yi": paths["qwen3"]})
    prediction_file = paths["yi"] / "models" / "yi" / "seal" / "record.json"
    prediction_file.write_bytes(before[prediction_file] + b"\n")
    with pytest.raises(ValueError, match="(file|payload)"):
        analysis._read_v1_comparison_inputs(paths)


def _synthetic_public_plan():
    from werewolf.structured_history import STRUCTURED_TOKEN_PLANNER_VERSION, StructuredTokenDescriptor
    from werewolf.tom_llm.common import digest

    plan = {"planner_version": STRUCTURED_TOKEN_PLANNER_VERSION, "prefix_digest": "a" * 64, "token_count": 2,
            "tokens": [StructuredTokenDescriptor(0, "phase_change", None, None, None, 1, "discussion", "phase", 0, 0).to_record(),
                       StructuredTokenDescriptor(1, "turn_start", "player1", None, None, 1, "discussion", "turn", 1, 0).to_record()]}
    return {**plan, "plan_digest": digest(plan)}


@pytest.mark.parametrize("fault", ["label", "conflicting_plan", "temporal"])
def test_epoch35_public_predictor_rejects_inconsistent_or_nonpublic_inputs(monkeypatch, fault):
    from werewolf.tom_llm import baseline_analysis as analysis
    from werewolf.tom_llm.common import digest

    plan = _synthetic_public_plan()
    task = {"game_id": "game", "boundary_id": "pre", "observer": 0, "fold": 0,
            "prefix_digest": plan["prefix_digest"], "plan_digest": plan["plan_digest"],
            "public_plan": plan, "messages": [], "response_schema": {}}
    tasks = [task, {**task, "observer": 1}]
    model = _mock_epoch35()
    monkeypatch.setattr(analysis, "V1_PRE_COUNT", 1)
    if fault == "label":
        tasks[1]["q"] = [0.] * 7
    elif fault == "conflicting_plan":
        tasks[1]["public_plan"] = {**plan, "token_count": 3}
    else:
        model.temporal.day = range(1)
    with pytest.raises(ValueError, match="public tasks|inconsistent public plan|Temporal mismatch"):
        analysis._predict_epoch35(tasks, [digest(t) for t in tasks], model, "cpu")
    assert model.calls == 0


def test_epoch35_full_shape_keeps_every_public_and_primary_identity(monkeypatch):
    from types import SimpleNamespace

    from werewolf.artifact_io import sha256_bytes
    from werewolf.tom_llm import baseline_analysis as analysis
    from werewolf.tom_llm.common import digest

    base_plan = _synthetic_public_plan()
    games = [f"game-{i:03}" for i in range(150)]
    tasks, refs, primary = [], [], []
    for i in range(1886):
        game, boundary = games[i % 150], f"pre-{i}"
        public_plan = {**base_plan, "prefix_digest": digest([game, boundary])}
        public_plan.pop("plan_digest")
        public_plan["plan_digest"] = digest(public_plan)
        for seat in range(5 if i < 602 else 6):
            task = {"game_id": game, "boundary_id": boundary, "observer": seat, "fold": (i % 150) // 30,
                    "prefix_digest": public_plan["prefix_digest"], "plan_digest": public_plan["plan_digest"],
                    "public_plan": public_plan, "messages": [], "response_schema": {}}
            row = {k: task[k] for k in analysis.ROW_FIELDS}
            q = [0. if k == seat else 1/6 for k in range(7)]
            if seat < (2 if i < 156 else 3):
                q = [float(k == (seat + 1) % 7) for k in range(7)]
            ref = {**row, "observer_alive": True, "q": q, **analysis.uniform_prediction_rows([row])[0]}
            refs.append(ref)
            tasks.append(task)
            if seat < (3 if i < 64 else 4):
                primary.append(ref)
    model = _mock_epoch35()
    predictions = analysis._predict_epoch35(tasks, [digest(t) for t in tasks], model, "cpu")
    assert model.calls == 1886
    assert len(predictions) == len({identity(t) for t in tasks}) == 10714
    assert [identity(r) for r in predictions] == [identity(t) for t in tasks]

    # 缩短合成测试抽样，真实 10,000 次抽样由上面的配对测试覆盖。
    indices = np.arange(150, dtype="<i4").reshape(1, 150)
    monkeypatch.setattr(analysis, "bootstrap_indices", lambda *_: indices)
    monkeypatch.setattr(analysis, "V1_BOOTSTRAP_DIGEST", sha256_bytes(indices.tobytes()))
    monkeypatch.setattr(analysis, "V1_PRIMARY_ROWS_DIGEST", digest([identity(r) for r in primary]))
    reference = SimpleNamespace(rows=refs, selected=lambda _: primary)
    passed = []
    score = analysis.probability_cell

    def score_aligned(rows, *args):
        assert [identity(r) for r in rows] == [identity(r) for r in primary]
        passed.append(len(rows))
        return score(rows, *args)

    monkeypatch.setattr(analysis, "probability_cell", score_aligned)
    v1_rows = {key: [{"prediction": {**row, "status": "valid"}} for row in predictions] for key in analysis.V1_RUN_SEALS}
    result = analysis._score_epoch35_v1(reference, games, tasks, predictions, v1_rows)
    assert passed == [7480] * 6 and result["nu_row_count"] == 5502
    assert result["nu_game_ids"] == games
    with pytest.raises(ValueError, match="ordered prediction coverage"):
        analysis._score_epoch35_v1(reference, games, tasks, predictions[:-1], v1_rows)
