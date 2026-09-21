"""Synthetic sealed-layout witnesses only; no training or real study access."""
from copy import deepcopy

import pytest
from scripts import paper_top1_nu as m
from tests.tom.test_paper_tom_metrics import prediction
from werewolf.artifact_io import publish_artifact, canonical_json_bytes, canonical_jsonl_bytes, sha256_bytes
from werewolf.tom.run_records import publish_record, record_with_digest


def hit(row):
    return m.score_prediction_rows([row],[row['game_id']])[0][0]['argmax_in_target_support']


@pytest.mark.parametrize('support,top,expected', [((1,),1,True),((1,),2,False),
    ((1,2),1,True),((1,2),2,True),((1,2,3),1,True),((1,2,3),2,True),((1,2,3),3,True)])
def test_formal_support_hit(support,top,expected):
    p=[0]+[.1]*6;p[top]=.5
    assert hit(prediction(support=support,p=p)) is expected


def test_exact_tie_and_nu_game_macro():
    assert hit(prediction(support=(1,))) is True
    assert hit(prediction(support=(2,))) is False
    rows=[prediction(game='a'),prediction(game='a'),prediction(game='b',support=(2,)),
          prediction(game='c',support=tuple(range(1,7)))]
    result=m.summarize(rows,[hit(r) for r in rows])
    assert result==dict(top1_nu=.5,total_primary_row_count=4,valid_nu_row_count=3,
        excluded_s6_row_count=1,total_primary_game_count=3,valid_nu_game_count=2)
    assert result['top1_nu']!=2/3
    assert m.summarize(rows[-1:],[True])['top1_nu'] is None


@pytest.fixture
def sealed(tmp_path):
    root=tmp_path/'study'
    expected=[(a,c,f) for a in m.FAMILIES for c in m.TEMPORAL_CONDITIONS for f in range(5)]
    full_digest='a'*64;base_digest='b'*64;ten_digest='c'*64
    execution=publish_artifact(root/'execution',manifest_fields={
        'artifact_type':'paper_study_execution','schema_version':'classic7_paper_study_execution_v1',
        'full_experiment_digest':full_digest,'agnostic_experiment_digest':base_digest,
        'expected_lineages':expected},files={})
    seal=publish_record(root/'study_seal.json',{'study_digest':execution.manifest_digest,
        'checkpoints':[dict(model_family=a,temporal_condition=c,fold=f,
             study_digest=execution.manifest_digest,checkpoint_digest=f'{a}/{c}/{f}') for a,c,f in expected]})
    games=[f'g{f}' for f in range(5)]
    formal_cells={}; bindings={}; files={}; descriptions={};cells={}
    for c in m.TEMPORAL_CONDITIONS:
        folds=[]
        for f in range(5):
            key=f'{c}/{f}'
            rows=[prediction(game=games[f],observer=i,support=((i+1)%7,)) for i in range(7)]
            for row in rows:
                row['temporal_condition']=c;row['checkpoint_digest']=f'{m.FAMILIES[0]}/{key}'
            p=canonical_jsonl_bytes(rows)
            descriptor=publish_record(root/f'runs/{full_digest}/{key}/prediction_manifest.json',{
                'experiment_digest':full_digest,'checkpoint_set_digest':ten_digest,'checkpoint_digest':rows[0]['checkpoint_digest'],
                'prediction_digest':sha256_bytes(p),'fold':f,'temporal_condition':c,'canonical_shift':0,'row_count':len(rows)})
            (root/f'runs/{full_digest}/{key}/held_out_predictions.jsonl').write_bytes(p)
            scores,game_scores=m.score_prediction_rows(rows,[games[f]])
            ids=sha256_bytes(canonical_json_bytes([list(m.identity(r)) for r in rows]))
            report=publish_record(root/f'runs/{full_digest}/{key}/{m.PRIMARY}/fold_report.json',{
                **{k:descriptor[k] for k in ('experiment_digest','checkpoint_set_digest','checkpoint_digest','prediction_digest','fold','temporal_condition')},
                'row_scores':scores,'game_scores':game_scores,'row_identity_digest':ids,
                'population_identity':'non_wolf_alive','population_selector_version':'classic7_primary_population_v1'})
            folds.append(report['record_digest'])
            base=deepcopy(rows)
            for row in base: row['checkpoint_digest']=f'{m.FAMILIES[1]}/{key}'
            payload=canonical_jsonl_bytes(base)
            files[f'agnostic/{key}/held_out_predictions.jsonl']=payload
            descriptions[key]={'full_prediction_digest':descriptor['record_digest'],
                'full_primary_report_digest':report['record_digest'],'agnostic':record_with_digest({
                'experiment_digest':base_digest,'checkpoint_set_digest':seal['record_digest'],
                'checkpoint_digest':base[0]['checkpoint_digest'],'fold':f,'temporal_condition':c,
                'canonical_shift':0,'row_count':len(base),'prediction_digest':sha256_bytes(payload),'primary_row_identity_digest':ids})}
            masks={m.identity(r):True for r in rows}
            bindings[key]=m._pair_rows(rows,base,masks,masks,fold=f,condition=c,game_ids=[games[f]])
        formal_cells[f'{c}/{m.PRIMARY}']={'fold_report_digests':folds}
        for a in m.FAMILIES: cells[f'{a}/{c}']={'row_count':35,'game_count':5}
    formal=publish_record(root/f'runs/{full_digest}/reports/oof_report_set.json',{
        'experiment_digest':full_digest,'checkpoint_set_digest':ten_digest,'game_ids':games,'cells':formal_cells})
    evaluation=publish_artifact(root/'evaluation',manifest_fields={
        'artifact_type':'paper_study_evaluation','schema_version':'classic7_paper_study_evaluation_v1',
        'study_digest':execution.manifest_digest,'study_seal_digest':seal['record_digest'],
        'full_report_set_digest':formal['record_digest'],'full_checkpoint_set_digest':ten_digest,
        'bootstrap':{'game_ids':games},'predictions':descriptions,'cells':cells,
        'row_identity_binding_digest':sha256_bytes(canonical_json_bytes(bindings))},files=files)
    pins=dict(expected_study=execution.manifest_digest,expected_seal=seal['record_digest'],expected_evaluation=evaluation.manifest_digest)
    return root,pins


def test_read_only_deterministic_four_cells(sealed,tmp_path,monkeypatch):
    import torch
    from werewolf.tom import training, evaluation, paper_study_evaluation, protocol
    def forbidden(*a,**kw): pytest.fail('training/inference/bootstrap forbidden')
    monkeypatch.setattr(training,'load_model_state',forbidden)
    monkeypatch.setattr(evaluation,'predict_held_out_fold',forbidden)
    monkeypatch.setattr(paper_study_evaluation,'evaluate_study',forbidden)
    monkeypatch.setattr(protocol,'load_bootstrap_plan',forbidden)
    monkeypatch.setattr(torch.nn.Module,'_call_impl',forbidden)
    root,pins=sealed
    before={str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    result=m.analyze(root,tmp_path/'one.json',**pins)
    m.analyze(root,tmp_path/'two.json',**pins)
    assert (tmp_path/'one.json').read_bytes()==(tmp_path/'two.json').read_bytes()
    assert before=={str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    assert result['status']=='post_hoc_supplementary_diagnostic'
    assert len(result['cells'])==4
    for cell in result['cells'].values():
        assert cell['valid_nu_row_count']+cell['excluded_s6_row_count']==cell['total_primary_row_count']
    with pytest.raises(ValueError,match='outside'): m.analyze(root,root/'result.json',**pins)
    (tmp_path/'elsewhere').mkdir()
    with pytest.raises(ValueError,match='outside'):
        m.analyze(root,tmp_path/'elsewhere/../study/result.json',**pins)
    with pytest.raises(ValueError,match='already exists'):
        m.analyze(root,tmp_path/'one.json',**pins)


@pytest.mark.parametrize('change',['missing','duplicate','q','condition','identity','digest'])
def test_prediction_corruption_rejected(sealed,tmp_path,change):
    root,pins=sealed
    if change=='digest': pins['expected_evaluation']='0'*64
    else:
        path=root/'runs'/('a'*64)/'implicit/0/held_out_predictions.jsonl'
        rows=[m.strict_json(line) for line in path.read_bytes().splitlines()]
        if change=='missing': rows.pop()
        elif change=='duplicate': rows.append(rows[0])
        elif change=='q': rows[0]['q']=[0]*7
        elif change=='condition': rows[0]['temporal_condition']='bad'
        else: rows[0]['boundary_id']='bad'
        path.write_bytes(canonical_jsonl_bytes(rows))
    with pytest.raises(ValueError): m.analyze(root,tmp_path/'out.json',**pins)
    assert not (tmp_path/'out.json').exists()


@pytest.mark.parametrize('change',['missing','duplicate','q','condition','identity'])
def test_pairing_rejects_rebound_semantic_mismatch(tmp_path,change):
    full=[prediction(observer=i,support=((i+1)%7,)) for i in range(7)]
    base=deepcopy(full)
    if change=='missing': base.pop()
    elif change=='duplicate': base.append(base[0])
    elif change=='q': base[0]['q']=[0,0,1,0,0,0,0]
    elif change=='condition': base[0]['temporal_condition']='explicit_day_phase'
    else: base[0]['boundary_id']='other-boundary'
    payload=canonical_jsonl_bytes(base)
    (tmp_path/'predictions.jsonl').write_bytes(payload)
    descriptor=dict(prediction_digest=sha256_bytes(payload),row_count=len(base),
                    checkpoint_digest=full[0]['checkpoint_digest'])
    with pytest.raises(ValueError):
        rows,_=m.predictions(m.Inputs(tmp_path),'predictions.jsonl',descriptor,'implicit')
        m._pair_rows(full,rows,{m.identity(r):True for r in full},
            {m.identity(r):True for r in rows},fold=0,condition='implicit',
            game_ids=[full[0]['game_id']])
