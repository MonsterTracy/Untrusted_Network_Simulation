from copy import deepcopy
from dataclasses import replace
import math

import numpy as np
import pytest
import torch

from tests.tom_llm.conftest import predictions_for
from werewolf.tom_llm.reference import open_reference
from werewolf.tom_llm.evaluation import compare
from werewolf.tom_llm.output import parse_output
from werewolf.tom_llm.common import identity
from werewolf.tom.scoring import paired_summary


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


def test_sealed_mock_to_reference_analysis_end_to_end(sealed_reference,tmp_path):
    ref,publication,*_=sealed_reference
    from werewolf.tom_llm.study import public_requests,prepare_mock_study
    from werewolf.tom_llm.inference import MockBackend,run_mock_study,seal_mock_study
    from werewolf.tom_llm.evaluation import analyze_mock
    from tests.tom_llm.test_selection_inference import uniform
    import json
    prefixes=[p for g in publication.public_view.games for p in g.authoritative_pre_prefixes]
    membership={g:f["fold"] for f in ref.folds for g in f["held_out_game_ids"]}
    requests=public_requests(prefixes,membership)
    root=tmp_path/"llm-mock"
    prepare_mock_study(root,requests,tokenizer_identity="synthetic:counter-v1",count_tokens=lambda _:100)
    backend=MockBackend(uniform)
    run_mock_study(root,backend); seal=seal_mock_study(root)
    result=analyze_mock(ref,root,ref.game_ids,tmp_path/"analysis")
    data=json.loads((result.path/"comparison.json").read_bytes())
    assert result.manifest["engineering_only"] is True
    assert result.manifest["llm_prediction_seal_digest"]==seal.manifest_digest
    assert data["Delta_KL"]["point"]==pytest.approx(0,abs=1e-12)
    assert data["failures"]["scheduled_public_alive_rows"]==len(backend.calls)
    assert data["failures"]["primary_rows"]==len(ref.primary_rows)
    with pytest.raises(ValueError,match="outside"):
        analyze_mock(ref,root,ref.game_ids,root/"bad-analysis")


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
