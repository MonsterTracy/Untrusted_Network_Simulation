"""Unified preparation, admission, serial execution, recovery, and analysis CLI."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from werewolf.tom_llm.common import require, strict_json
from werewolf.tom_llm.experiment import prepare_run, open_run, safe_path, choose_games
from werewolf.tom_llm.model_spec import ModelSpec
from werewolf.tom_llm.live_runner import (HttpTransport,LiveTransport,run_model,run_lock,status,
    pending_tasks,finalize_received,read_predictions,seal_run)


def keyed(values):
    result={}
    for value in values or []:
        k,v=value.split('=',1);require(k not in result,'duplicate keyed argument');result[k]=v
    return result


def prepare(args,root):
    from werewolf.artifact_io import verify_artifact,sha256_bytes
    from werewolf.tom_llm.qualification_inputs import export_cpu_cohort,publish_budget,open_budget
    from werewolf.tom_llm.multimodel import read_cohort
    from werewolf.tom_llm.reference import open_reference
    from werewolf.tom_llm.multimodel_evaluation import reference_binding,validate_public_coverage
    require(args.config and not root.exists(),'prepare needs config and NEW run directory')
    raw=safe_path(args.config).read_bytes();c=strict_json(raw)
    models=args.models or list(c['models']);games=args.games if args.games is not None else 150
    mode=args.mode or 'technical'
    expected=strict_json(safe_path(c['expected_selection_json']).read_bytes())
    provenance=strict_json(safe_path(c['selection_provenance_json']).read_bytes())
    for name,field in [('pre_reference_evidence','pre_reference_selection_evidence_digest'),('reconciliation_evidence','reconciliation_evidence_digest')]:
        require(sha256_bytes(safe_path(c[name]).read_bytes())==provenance[field],'historical evidence bytes drift')
    # No Selection publisher exists in this CLI. An unpublished Selection must stop here.
    from werewolf.tom_llm.selection_binding import open_selection_binding
    prepared=verify_artifact(c['prepared'],expected_artifact_type='backbone_tom_study',expected_schema_version='classic7_backbone_study_v1')
    binding=open_selection_binding(c['selection'],prepared.manifest['folds'],expected=expected,provenance=provenance,prepared_manifest=prepared.manifest)
    selection_artifact=verify_artifact(c['selection'],expected_artifact_type='llm_game_selection',expected_schema_version='classic7_llm_game_selection_v1')
    choose_games(strict_json((selection_artifact.path/'selection.json').read_bytes()),games,mode)
    for key in models:ModelSpec(key).record();require(key in c['models'],'model config missing')
    if mode!='fake' and sys.platform=='linux':
        from werewolf.tom_llm.deployment import BASE
        require(root.is_relative_to(BASE),'Ubuntu writes must remain inside independent b2 root')
    if c.get('cohort'):
        cohort,tasks=read_cohort(c['cohort']['path'],c['cohort']['digest'])
    else:
        cohort=export_cpu_cohort(root/'cohort',publication_root=c['publication'],evaluation_root=c['evaluation'],
            expected_reference=c['expected_reference'],selection_root=c['selection'],expected_selection=expected,
            selection_provenance=provenance,prepared_root=c['prepared'])
        cohort,tasks=read_cohort(cohort.path,cohort.manifest_digest)
    selected=strict_json((selection_artifact.path/'selection.json').read_bytes())
    require(cohort.manifest['selection']=={**binding,'game_ids':selected['game_ids']},'configured Selection/Prepared/provenance differs from cohort')
    ref=open_reference(c['publication'],c['evaluation'],expected_seal_digest=c['expected_reference'])
    require(cohort.manifest['reference']==reference_binding(ref),'configured reference differs from cohort')
    validate_public_coverage(tasks,ref,cohort.manifest['selection']['game_ids'])
    budgets={}
    for key in models:
        item=c['models'][key]
        if item.get('budget'):
            budget=open_budget(item['budget']['path'],ModelSpec(key),item['budget']['digest'])
            require(budget['cohort_digest']==cohort.manifest_digest,'budget belongs to another cohort')
            budgets[key]=item['budget']
        elif mode!='fake':
            b=publish_budget(root/'budgets'/key,cohort.path,cohort.manifest_digest,ModelSpec(key),item['tokenizer_assets'])
            budgets[key]={'path':str(b.path),'digest':b.manifest_digest}
    return prepare_run(root,run_id=args.run_id,models=models,games=games,mode=mode,
        cohort_root=cohort.path,expected_cohort=cohort.manifest_digest,publication=c['publication'],evaluation=c['evaluation'],
        expected_reference=c['expected_reference'],configuration={'input_config':c,'input_sha256':sha256_bytes(raw),'budgets':budgets})


def owned_service(root,key,expected):
    """Only this Popen's new process group may be terminated. No discovery/kill of other jobs."""
    from contextlib import contextmanager
    from werewolf.tom_llm.live_admission import open_permit,capture_live_instance
    from werewolf.tom_llm.deployment import CACHE_VARIABLES,validate_cache_paths,BASE
    @contextmanager
    def context():
        permit,r=open_permit(root,key,expected);p=r['profile'];service=safe_path(p['root'])
        service.mkdir(parents=True,exist_ok=True)
        with (BASE/'baseline-gpu.lock').open('a+b') as gpu_lock:
            fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            for var in CACHE_VARIABLES:safe_path(p['environment_requirements'][var]).mkdir(parents=True,exist_ok=True)
            validate_cache_paths(p,require_existing=True)
            env={k:os.environ[k] for k in ('HOME','USER','LANG','LC_ALL') if k in os.environ}
            env.update(p['environment_requirements'])
            cmd=[p['environment_prefix']+'/bin/python','-m','werewolf.tom_llm.live_service','--run',str(Path(root).absolute()),
                 '--model',key,'--expected-permit',expected,'--execute-live']
            from werewolf.tom_llm.model_spec import ROOT
            with (service/('stdout-'+uuid.uuid4().hex+'.log')).open('xb') as log:
                proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:
                    deadline=time.monotonic()+900
                    while True:
                        require(proc.poll() is None,'owned service exited; retain stdout log')
                        try:
                            instance=capture_live_instance(permit,r,proc.pid)
                            with Path(instance['engine_log']).open('rb') as audit:startup=strict_json(audit.readline())
                            require(startup['kind']=='startup' and startup['permit_digest']==expected,'live engine startup not ready')
                            break
                        except (OSError,ValueError,KeyError):
                            require(time.monotonic()<deadline,'service startup timeout; no model request sent');time.sleep(.25)
                    yield proc.pid
                finally:
                    if proc.poll() is None:
                        require(os.getpgid(proc.pid)==proc.pid,'owned process-group identity changed')
                        os.killpg(proc.pid,signal.SIGTERM)
                        proc.wait(timeout=60)  # No force kill or retry. Retain process/logs on failure.
    return context()


def execute(args,root):
    _,run,tasks=open_run(root)
    require(args.games is None or args.games==run['game_count'],'games differ from prepared contract')
    require(args.mode is None or args.mode==run['mode'],'mode differs from prepared contract')
    models=args.models or run['model_order'];require(len(set(models))==len(models) and set(models)<=set(run['models']),'duplicate/unplanned models')
    permits=keyed(args.permit);endpoints=keyed(args.fake_endpoint);pids=keyed(args.service_pid)
    if run['mode']!='fake':
        require(args.execute_live and sys.platform=='linux','explicit --execute-live on Ubuntu required')
        require(args.manage_services or len(models)==1,'external services require one model at a time')
        require(not endpoints,'fake endpoint cannot authorize a live run')
    else:require(not args.execute_live and not args.manage_services and not pids,'fake mode has no GPU/service-management capability')
    with run_lock(root):
        if (Path(root)/'seal').exists():
            for key in run['models']:read_predictions(root,key)
            return status(root)
        for key in models:
            model=Path(root)/'models'/key
            if (model/'seal').exists():read_predictions(root,key);continue
            require(args.resume or not (model/'requests').exists(),'existing requests require explicit --resume')
            pending=pending_tasks(root,key,tasks)
            if not pending:finalize_received(root,key);continue
            if run['mode']=='fake':
                require(key in endpoints,'fake model needs explicit --fake-endpoint key=http://127.0.0.1:PORT')
                run_model(root,key,HttpTransport(endpoints[key]));continue
            require(key in permits,'explicit --permit key=MANIFEST required')
            if args.manage_services:
                with owned_service(root,key,permits[key]) as pid:run_model(root,key,LiveTransport(root,key,permits[key],pid))
            else:
                require(key in pids,'explicit --service-pid key=PID required')
                run_model(root,key,LiveTransport(root,key,permits[key],int(pids[key])))
        if all((Path(root)/'models'/key/'seal').exists() for key in run['models']):
            return seal_run(root)
        return status(root)


def main(argv=None, *, analysis=False):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action',choices=['prepare','permit','run','status','analyze'],default='analyze' if analysis else 'run')
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--output-root',default='/data/yuxiao/tom-llm-reasoner-b2/experiments')
    parser.add_argument('--models',nargs='+',choices=['qwen3','yi','mistral'])
    parser.add_argument('--games',type=int);parser.add_argument('--mode',choices=['formal','technical','fake'])
    parser.add_argument('--config');parser.add_argument('--resume',action='store_true')
    parser.add_argument('--execute-live',action='store_true');parser.add_argument('--manage-services',action='store_true')
    parser.add_argument('--permit',action='append',help='model=expected permit manifest digest')
    parser.add_argument('--fake-endpoint',action='append');parser.add_argument('--service-pid',action='append')
    parser.add_argument('--qualification');parser.add_argument('--expected-qualification')
    parser.add_argument('--budget')
    parser.add_argument('--tokenizer-assets');parser.add_argument('--expected-run-seal');parser.add_argument('--report-output')
    a=parser.parse_args(argv)
    require(__import__('re').fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}',a.run_id) is not None,'invalid run-id')
    root=safe_path(a.output_root)/a.run_id
    if a.action=='prepare':result=prepare(a,root)
    elif a.action=='status':result=status(root)
    elif a.action=='run':result=execute(a,root)
    elif a.action=='permit':
        from werewolf.tom_llm.live_admission import publish_permit
        require(a.models and len(a.models)==1,'permit is issued per-model')
        require(all((a.qualification,a.expected_qualification,a.budget,a.tokenizer_assets)),'permit requires qualification, budget and tokenizer assets')
        result=publish_permit(root,a.models[0],qualification=a.qualification,expected_qualification=a.expected_qualification,
            tokenizer_assets=a.tokenizer_assets,budget_root=a.budget)
    else:
        from werewolf.tom_llm.baseline_analysis import analyze
        result=analyze(root,a.report_output or root/'analysis',expected_run_seal=a.expected_run_seal)
    if hasattr(result,'manifest_digest'):result={'path':str(result.path),'manifest_digest':result.manifest_digest}
    print(json.dumps(result,ensure_ascii=False,sort_keys=True))
    return result
