"""Manual CPU/qualification entry points. GPU calls require a separate explicit flag."""

import argparse
from pathlib import Path
import sys

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    def model(p):p.add_argument('--model',choices=['qwen3','yi','mistral'],required=True)
    def output(p):p.add_argument('--output',required=True)
    def upstream(p,name):
        p.add_argument('--'+name,required=True);p.add_argument('--expected-'+name,required=True)
    p=sub.add_parser('cohort');output(p)
    for n in ('publication','evaluation','expected-reference','selection','expected-selection-json','selection-provenance-json','prepared'):
        p.add_argument('--'+n,required=True)
    p=sub.add_parser('assets');model(p);output(p);p.add_argument('--directory',required=True)
    p=sub.add_parser('environment');output(p);p.add_argument('--prefix',required=True);p.add_argument('--prior-audit-digest',required=True)
    p=sub.add_parser('suite');model(p);output(p);p.add_argument('--tokenizer-assets',required=True)
    p=sub.add_parser('budget');model(p);output(p);p.add_argument('--tokenizer-assets',required=True);upstream(p,'cohort')
    p=sub.add_parser('deploy');model(p);output(p);p.add_argument('--gpu-uuid',required=True)
    p.add_argument('--model-directory',required=True);p.add_argument('--environment-prefix',required=True)
    for n in ('assets','environment','suite','budget'):upstream(p,n)
    p=sub.add_parser('bundle');output(p);upstream(p,'deployment')
    p=sub.add_parser('start');upstream(p,'deployment');p.add_argument('--execute-synthetic-gpu-qualification',action='store_true')
    p=sub.add_parser('round');output(p);upstream(p,'deployment');p.add_argument('--pid',type=int,required=True)
    p.add_argument('--run',type=int,choices=[0,1],required=True);p.add_argument('--execute-synthetic-gpu-qualification',action='store_true')
    p=sub.add_parser('admit');output(p);upstream(p,'deployment')
    for n in ('round0','round1'):upstream(p,n)
    p=sub.add_parser('inspect-admission');model(p);upstream(p,'admission');p.add_argument('--expected-deployment',required=True)
    p=sub.add_parser('inspect-budgets')
    for n in ('qwen3','yi','mistral'):upstream(p,n)
    args=parser.parse_args()
    if sys.platform=='linux' and hasattr(args,'output'):
        target=Path(args.output).resolve()
        if not target.is_relative_to(Path('/data/yuxiao/tom-llm-reasoner-b2')):
            parser.error('Ubuntu writes must stay inside the independent b2 directory')
    if args.command=='inspect-budgets':
        from werewolf.tom_llm.qualification_inputs import open_budget,budget_matrix
        from werewolf.tom_llm.model_spec import ModelSpec
        import json
        print(json.dumps(budget_matrix({key:open_budget(getattr(args,key),ModelSpec(key),getattr(args,'expected_'+key))
              for key in ('qwen3','yi','mistral')}),sort_keys=True))
        return
    from werewolf.tom_llm.model_spec import ModelSpec
    if args.command=='cohort':
        from werewolf.tom_llm.qualification_inputs import export_cpu_cohort
        from werewolf.tom_llm.common import strict_json
        result=export_cpu_cohort(args.output,publication_root=args.publication,evaluation_root=args.evaluation,
            expected_reference=args.expected_reference,selection_root=args.selection,
            expected_selection=strict_json(Path(args.expected_selection_json).read_bytes()),
            selection_provenance=strict_json(Path(args.selection_provenance_json).read_bytes()),prepared_root=args.prepared)
    elif args.command=='assets':
        from werewolf.tom_llm.physical import verify_model,publish_receipt
        result=publish_receipt(args.output,'model',verify_model(ModelSpec(args.model),args.directory))
    elif args.command=='environment':
        from werewolf.tom_llm.physical import capture_environment,publish_receipt
        result=publish_receipt(args.output,'environment',capture_environment(args.prefix,prior_audit_digest=args.prior_audit_digest))
    elif args.command=='suite':
        from werewolf.tom_llm.qualification_inputs import publish_suite
        result=publish_suite(args.output,ModelSpec(args.model),args.tokenizer_assets)
    elif args.command=='budget':
        from werewolf.tom_llm.qualification_inputs import publish_budget
        result=publish_budget(args.output,args.cohort,args.expected_cohort,ModelSpec(args.model),args.tokenizer_assets)
    elif args.command=='deploy':
        from werewolf.tom_llm.deployment import prepare_deployment
        result=prepare_deployment(args.output,ModelSpec(args.model),gpu_uuid=args.gpu_uuid,
            model_root=args.model_directory,environment_root=args.environment_prefix,
            model_receipt=args.assets,expected_model=args.expected_assets,
            environment_receipt=args.environment,expected_environment=args.expected_environment,
            suite_root=args.suite,expected_suite=args.expected_suite,budget_root=args.budget,expected_budget=args.expected_budget)
    elif args.command=='bundle':
        from werewolf.tom_llm.deployment import publish_bundle
        result=publish_bundle(args.output,args.deployment,args.expected_deployment)
    elif args.command=='start':
        from werewolf.tom_llm.gpu_probe import start_deployment
        return start_deployment(args.deployment,args.expected_deployment,execute=args.execute_synthetic_gpu_qualification)
    elif args.command=='round':
        from werewolf.tom_llm.candidate_client import collect_physical_round
        result=collect_physical_round(args.output,args.deployment,args.expected_deployment,pid=args.pid,run=args.run,
                                      execute=args.execute_synthetic_gpu_qualification)
    elif args.command=='admit':
        from werewolf.tom_llm.physical_admission import publish_admission
        result=publish_admission(args.output,args.deployment,args.expected_deployment,
            round_roots=[args.round0,args.round1],expected_rounds=[args.expected_round0,args.expected_round1])
    else:
        from werewolf.tom_llm.physical_admission import open_admission
        result,_=open_admission(args.admission,expected_manifest=args.expected_admission,
            expected_deployment=args.expected_deployment,model_spec=ModelSpec(args.model))
    import json
    print(json.dumps({'path':str(result.path),'manifest_digest':result.manifest_digest,
        'artifact_type':result.manifest['artifact_type'],'formal_execution_authorized':False},sort_keys=True))


if __name__=='__main__':main()
