"""Lightweight uns registration; model/service dependencies load only on dispatch."""

def register(commands):
    register_operations(commands)
    check = commands.add_parser("check-llm-reasoner-config", help="validate frozen LLM policy; no model/backend is loaded")
    check.add_argument("--config", required=True)


def check_config(path):
    from werewolf.tom_llm.common import digest
    from werewolf.tom_llm.policy import load_policy, policy_binding
    config = load_policy(path)
    return {"status": "valid", "config_digest": digest(config), "binding": policy_binding(),
            "execution": "local_validation_only", "formal_readiness": False}


def register_operations(commands):
    p=commands.add_parser('tom-llm',help='configured ToM LLM preparation, services, qualification and experiments')
    p.add_argument('--config',default='tom-llm.yaml',help='operations YAML (legacy experiment JSON also accepted)')
    sub=p.add_subparsers(dest='operation',required=True)
    def models(p):p.add_argument('--models',nargs='+',choices=['qwen3','yi','mistral'])
    def run(p):p.add_argument('--run-id')
    p=sub.add_parser('status');models(p);run(p)
    p=sub.add_parser('prepare');models(p);run(p)
    p.add_argument('--stage',choices=['cohort','tokenizer','suite','budget','model','run','permit'],default='run')
    p.add_argument('--games',type=int,default=150);p.add_argument('--mode',choices=['technical','formal'],default='technical')
    p.add_argument('--save-config',action='store_true',help='explicitly store generated Artifact references in the operations YAML/JSON')
    p=sub.add_parser('serve');p.add_argument('service_action',choices=['start','stop','status','logs'])
    p.add_argument('--model',required=True,choices=['qwen3','yi','mistral']);run(p)
    p.add_argument('--purpose',choices=['qualification','run'],default='qualification')
    p.add_argument('--execute-live',action='store_true');p.add_argument('--lines',type=int,default=30)
    p=sub.add_parser('qualify');p.add_argument('--model',required=True,choices=['qwen3','yi','mistral'])
    p.add_argument('--round',type=int,choices=[0,1],required=True);p.add_argument('--execute-live',action='store_true')
    p=sub.add_parser('admit');models(p)
    p.add_argument('--save-config',action='store_true')
    p=sub.add_parser('run');models(p);run(p)
    p.add_argument('--execute-live',action='store_true');p.add_argument('--manage-services',action='store_true')
    p.add_argument('--resume',action='store_true')
    p=sub.add_parser('report');run(p);p.add_argument('--output');p.add_argument('--expected-run-seal')
