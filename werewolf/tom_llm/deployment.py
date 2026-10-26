"""Independent per-model deployment candidates with externally anchored CPU inputs."""

import os
from pathlib import Path
import subprocess
import sys

from werewolf.artifact_io import publish_artifact, verify_artifact, canonical_json_bytes, sha256_bytes
from werewolf.tom_llm.common import digest, require, require_digest, strict_json
from werewolf.tom_llm.model_spec import ModelSpec, ROOT, template_bytes
from werewolf.tom_llm.multimodel import source_identity
from werewolf.tom_llm.physical import read_receipt, verify_model, capture_environment
from werewolf.tom_llm.qualification_inputs import open_suite, open_budget

VERSION = 'classic7_multimodel_deployment_v1'
BASE = Path('/data/yuxiao/tom-llm-reasoner-b2')
RECIPE = ROOT/'configs/paper/tom-llm-qualification-recipe-v1.json'
RECIPE_SHA = '8b664c3cceede49cc9083c64cb2a4207e8e5853352ddb51501bec20712a97aaa'


def recipe():
    raw = RECIPE.read_bytes()
    require(sha256_bytes(raw)==RECIPE_SHA, 'qualification deployment recipe drift')
    return strict_json(raw)


def vllm_config_bytes(options):
    """Canonical flow-style YAML; vLLM drops false values, so use explicit negation."""
    return canonical_json_bytes({('no-'+key if value is False else key):
        (True if value is False else value) for key,value in options.items()})


def profile(spec, *, gpu_uuid, candidate_root, model_directory=None, environment_prefix=None):
    require(type(spec) is ModelSpec and type(gpu_uuid) is str and gpu_uuid.startswith('GPU-'), 'explicit model/GPU identity required')
    row = spec.record(); root = Path(candidate_root)
    require(root.is_absolute() and root.is_relative_to(BASE/'multimodel'/spec.key)
            and root != BASE/'multimodel'/spec.key and '..' not in root.parts
            and not any(p.is_symlink() for p in (root,*root.parents)), 'isolated candidate root required')
    model = Path(model_directory) if model_directory else root/'model'
    env = Path(environment_prefix) if environment_prefix else root/'env'
    require(model.is_absolute() and env.is_absolute() and model.is_relative_to(BASE) and env.is_relative_to(BASE)
            and '..' not in model.parts and '..' not in env.parts, 'physical paths must remain inside independent base')
    cache = root/'cache'
    variables = {k:str(cache/n) for k,n in {'HF_HOME':'huggingface','XDG_CACHE_HOME':'xdg',
        'TORCH_HOME':'torch','TRITON_CACHE_DIR':'triton','VLLM_CACHE_ROOT':'vllm','NUMBA_CACHE_DIR':'numba',
        'FLASHINFER_WORKSPACE_BASE':'flashinfer','TVM_FFI_CACHE_DIR':'tvm-ffi',
        'CUDA_CACHE_PATH':'cuda','TORCH_EXTENSIONS_DIR':'torch-extensions'}.items()}
    variables.update({'TMPDIR':str(root/'tmp'),'PYTHONPATH':str(root/'source/Untrusted_Network_Simulation'),
        'PATH':str(env/'bin')+':/usr/bin:/bin','CUDA_VISIBLE_DEVICES':'0','PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
        'VLLM_NO_USAGE_STATS':'1','DO_NOT_TRACK':'1','CUDA_HOME':str(env/'lib/python3.12/site-packages/nvidia/cu13')})
    args = {'model':str(model),'tokenizer':str(model),'revision':row['revision'],
        'tokenizer-revision':row['tokenizer_revision'],'tokenizer-mode':'auto','served-model-name':[row['model']],
        'dtype':'bfloat16','generation-config':'auto','host':'127.0.0.1','port':{'qwen3':18581,'yi':18582,'mistral':18583}[spec.key],
        'max-model-len':8192,'max-num-seqs':1,'gpu-memory-utilization':.9,'seed':20261014,
        'enable-prefix-caching':False,'trust-request-chat-template':False,'chat-template-content-format':'string',
        'chat-template':str(root/'deployment/chat_template.jinja'),'structured-outputs-config':{'backend':'xgrammar'}}
    return {'schema_version':VERSION+'_profile','key':spec.key,'model_spec_digest':spec.digest,'root':str(root),
        'environment_prefix':str(env),'environment_requirements':variables,'argv_options':args,
        'gpu':{'uuid':gpu_uuid,'name':'NVIDIA GeForce RTX 3090','memory_total_mib':24576},
        'engine_log_root':str(root/'qualification/engine'),'frontend':recipe()['frontend'],
        'recipe_sha256':RECIPE_SHA,'sdk_retries':0,'http_retries':0,'proxy':False,'qualification_only':True}



CACHE_VARIABLES = ('HF_HOME','XDG_CACHE_HOME','TORCH_HOME','TRITON_CACHE_DIR','VLLM_CACHE_ROOT',
    'NUMBA_CACHE_DIR','FLASHINFER_WORKSPACE_BASE','TVM_FFI_CACHE_DIR','CUDA_CACHE_PATH','TORCH_EXTENSIONS_DIR','TMPDIR')
BACKEND_PREFIXES = ('VLLM_','CUDA_','TORCH_','PYTORCH_','FLASHINFER_','TVM_','TRITON_','NUMBA_','NCCL_','CUBLAS_','HF_','TRANSFORMERS_')
FORBIDDEN_ENV = ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy','LD_PRELOAD','LD_LIBRARY_PATH')


def validate_cache_paths(p,*,require_existing=False):
    root=Path(p['root'])
    for key in CACHE_VARIABLES:
        path=Path(p['environment_requirements'][key])
        require(path.is_relative_to(root) and not any(q.is_symlink() for q in (path,*path.parents)),
                'cache path escaped model root or uses symlink: '+key)
        if require_existing:
            require(path.is_dir() and os.access(path,os.W_OK|os.X_OK),'scoped writable cache/TMPDIR missing: '+key)


def validate_process_environment(p,environment):
    validate_cache_paths(p)
    require(all(environment.get(k)==v for k,v in p['environment_requirements'].items()),'isolated process environment drift')
    require(not any(k in environment for k in FORBIDDEN_ENV),'proxies/loader overrides forbidden')
    actual={k:v for k,v in environment.items() if k.startswith(BACKEND_PREFIXES)}
    expected={k:v for k,v in p['environment_requirements'].items() if k.startswith(BACKEND_PREFIXES)}
    require(actual==expected,'undeclared CUDA/JIT/backend environment override')


def verify_cpu_binding(record,spec):
    require(record['model_spec']==spec.record() and record['source']==source_identity(), 'deployment source/ModelSpec drift')
    p = record['profile']
    validate_cache_paths(p)
    require(p==profile(spec,gpu_uuid=p['gpu']['uuid'],candidate_root=p['root'],model_directory=p['argv_options']['model'],
                      environment_prefix=p['environment_prefix']), 'deployment profile/parameters drift')
    require(record['model']==verify_model(spec,record['model']['directory']), 'physical model changed after receipt')
    require(record['environment']==capture_environment(p['environment_prefix'],
            prior_audit_digest=record['environment']['prior_audit_digest']), 'physical environment changed after receipt')
    require(record['model']['directory']==p['argv_options']['model']
            and record['environment']['prefix']==p['environment_prefix'], 'deployment physical path mismatch')


def prepare_deployment(destination,spec,*,gpu_uuid,model_root,model_receipt,expected_model,
                       environment_root,environment_receipt,expected_environment,
                       suite_root,expected_suite,budget_root,expected_budget):
    destination = Path(destination).absolute()
    require(destination.name=='deployment' and not destination.exists(), 'new ROOT/deployment destination required')
    p = profile(spec,gpu_uuid=gpu_uuid,candidate_root=destination.parent,
        model_directory=model_root,environment_prefix=environment_root)
    model = read_receipt(model_receipt,'model',expected_model)
    environment = read_receipt(environment_receipt,'environment',expected_environment)
    suite = open_suite(suite_root,spec,expected_suite); budget = open_budget(budget_root,spec,expected_budget)
    record = {'schema_version':VERSION,'key':spec.key,'model_spec':spec.record(),'source':source_identity(),
        'profile':p,'model':model,'environment':environment,'suite':suite,'budget':budget,
        'upstream_receipts':{'model':expected_model,'environment':expected_environment,'suite':expected_suite,'budget':expected_budget},
        'purpose':'synthetic_gpu_qualification_only','physical_gpu_admission':False}
    verify_cpu_binding(record,spec)
    return publish_artifact(destination,manifest_fields={'artifact_type':'mm_deployment_candidate','schema_version':VERSION,
        'deployment_digest':digest(record),'model_spec_digest':spec.digest,'gpu_qualification_passed':False},
        files={'deployment.json':canonical_json_bytes(record),'vllm.yaml':vllm_config_bytes(p['argv_options']),
               'chat_template.jinja':template_bytes(spec)})


def open_deployment(root,expected_digest,*,physical_recheck=False):
    require_digest(expected_digest)
    a = verify_artifact(root,expected_artifact_type='mm_deployment_candidate',expected_schema_version=VERSION)
    require(a.manifest_digest==expected_digest and set(a.manifest['file_table'])=={'deployment.json','vllm.yaml','chat_template.jinja'},
            'new deployment artifact identity/inventory required; no W5/mock upgrade')
    r = strict_json((a.path/'deployment.json').read_bytes());spec = ModelSpec(r['key'])
    require(r['schema_version']==VERSION and r['model_spec']==spec.record() and r['source']==source_identity()
            and digest(r)==a.manifest['deployment_digest'] and a.manifest['model_spec_digest']==spec.digest
            and r['purpose']=='synthetic_gpu_qualification_only' and r['physical_gpu_admission'] is False,
            'deployment execution identity/domain drift')
    p=r['profile']
    require(p==profile(spec,gpu_uuid=p['gpu']['uuid'],candidate_root=p['root'],model_directory=p['argv_options']['model'],
                      environment_prefix=p['environment_prefix']) and (a.path/'chat_template.jinja').read_bytes()==template_bytes(spec)
            and (a.path/'vllm.yaml').read_bytes()==vllm_config_bytes(p['argv_options']), 'deployment template/config drift')
    from werewolf.tom_llm.qualification_inputs import validate_suite
    validate_suite(r['suite'],spec)
    if physical_recheck:verify_cpu_binding(r,spec)
    return a,r


def publish_bundle(destination,root,expected_digest):
    a,r = open_deployment(root,expected_digest)
    files = {name:(a.path/name).read_bytes() for name in a.manifest['file_table']}
    for name,wanted in r['source']['file_digests'].items():
        raw=(ROOT/name).read_bytes();require(sha256_bytes(raw)==wanted,'bundle source drift')
        files['source/'+name]=raw
    return publish_artifact(destination,manifest_fields={'artifact_type':'mm_deployment_bundle','schema_version':VERSION,
        'source':r['source'],'candidate_manifest_digest':a.manifest_digest,'key':r['key'],
        'physical_gpu_admission':False,'qualification':'CANDIDATE_ONLY'},files=files)


def gpu_snapshot():
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.total','--format=csv,noheader,nounits'],text=True).strip().splitlines()
    require(len(rows)==1,'single dedicated GPU required')
    uuid,name,size=[x.strip() for x in rows[0].split(',')]
    return {'uuid':uuid,'name':name,'memory_total_mib':int(size)}


def driver_version():
    values=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader,nounits'],text=True).strip().splitlines()
    require(len(values)==1 and values[0].strip(),'physical NVIDIA driver identity absent')
    return values[0].strip()


def require_idle_gpu(p):
    require(gpu_snapshot()==p['gpu'],'GPU physical identity drift')
    row=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
    free,util=[int(x.strip()) for x in row.split(',')]
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    require(free>=22528 and util==0 and not processes,'GPU not idle; never stop or alter existing jobs')


def capture_instance(deployment,record,pid):
    """Linux /proc and listener ownership; reads no secret environment values into records."""
    require(sys.platform=='linux' and type(pid) is int and pid>0,'physical Linux service PID required')
    require(record['source']['source_inventory_clean'],'physical service requires reviewed committed Source')
    p=record['profile'];proc=Path('/proc');process=proc/str(pid)
    require(Path(p['argv_options']['chat-template']).read_bytes()==(deployment.path/'chat_template.jinja').read_bytes(), 'installed service template drift')
    expected=[p['environment_prefix']+'/bin/python','-m','werewolf.tom_llm.gpu_probe','--deployment',str(deployment.path),
              '--expected-deployment',deployment.manifest_digest,'--execute-synthetic-gpu-qualification']
    argv=[s.decode() for s in (process/'cmdline').read_bytes().split(b'\0') if s]
    require(argv==expected and (process/'exe').resolve()==Path(expected[0]).resolve(),'service argv/executable drift')
    env=dict(s.split(b'=',1) for s in (process/'environ').read_bytes().split(b'\0') if b'=' in s)
    env={k.decode():v.decode() for k,v in env.items()}
    validate_process_environment(p,env)
    process_life={'pid':pid,'boot_id':(proc/'sys/kernel/random/boot_id').read_text().strip(),
        'start_ticks':(process/'stat').read_text().rsplit(')',1)[1].split()[19]}
    expected_log=str(Path(p['engine_log_root'])/digest(process_life)/'engine-events.jsonl')
    inodes=set();owned_log=False
    for fd in (process/'fd').iterdir():
        try:
            target=os.readlink(fd)
            if target.startswith('socket:['):inodes.add(target[8:-1])
            if target==expected_log:owned_log=True
        except OSError:pass
    require(owned_log,'engine audit log is not open in the declared service process')
    port=p['argv_options']['port']
    require(any(v[1]==f'0100007F:{port:04X}' and v[3]=='0A' and v[9] in inodes
        for line in (proc/'net/tcp').read_text().splitlines()[1:] if (v:=line.split())), 'loopback listener not owned by declared service')
    require(gpu_snapshot()==p['gpu'],'physical service GPU drift')
    result={'pid':pid,'boot_id':(proc/'sys/kernel/random/boot_id').read_text().strip(),
            'start_ticks':(process/'stat').read_text().rsplit(')',1)[1].split()[19],
            'deployment_manifest_digest':deployment.manifest_digest,'gpu':p['gpu'],'argv':argv,
            'driver_version':driver_version(),'kernel_release':os.uname().release,
            'environment_requirements':p['environment_requirements']}
    return {**result,'instance_digest':digest(result)}
