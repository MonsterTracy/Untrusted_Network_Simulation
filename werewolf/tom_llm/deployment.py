"""固定 vLLM 启动配置及单 GPU 空闲检查，不再发布 Deployment/Admission。"""

import os
from pathlib import Path
import subprocess

from werewolf.artifact_io import canonical_json_bytes
from werewolf.tom_llm.common import require
from werewolf.tom_llm.experiment import safe_path
from werewolf.tom_llm.model_spec import ROOT

BASE = Path('/data/yuxiao/tom-llm-reasoner-b2')
EXPERIMENT_BASE = Path('/data/yuxiao/TomBaseline-data')
MODEL_BASE = Path('/data/yuxiao/models')
ENV_BASE = Path('/data/yuxiao/envs')
GPU_LOCK = BASE/'baseline-gpu.lock'


def baseline_root(path):
    path = safe_path(path)
    base = next((b for b in (BASE,EXPERIMENT_BASE) if path.is_relative_to(b)),None)
    require(base is not None and path!=base,'output outside baseline roots')
    return base


def vllm_config_bytes(options):
    # vLLM 的 YAML 需要显式 no-* 才能保留 False。
    return canonical_json_bytes({('no-'+k if v is False else k):(True if v is False else v)
                                 for k,v in options.items()})


def service_environment(spec, *, root, environment_prefix, cache_base, gpu_uuid=None):
    """CPU 准备与 GPU 服务复用同一离线缓存布局；未选 GPU 时显式隐藏设备。"""
    root = safe_path(root);prefix = safe_path(environment_prefix);base = safe_path(cache_base)
    require(base in (BASE,EXPERIMENT_BASE),'registered service cache root required')
    cache = base/'runtime-cache'/spec.key
    ipc = base/'runtime-ipc'/spec.key
    require(len(os.fsencode(ipc))+1+36<=107,'RPC socket path exceeds Linux limit')
    variables = {key:str(cache/name) for key,name in {
        'HF_HOME':'huggingface','XDG_CACHE_HOME':'xdg','TORCH_HOME':'torch','TRITON_CACHE_DIR':'triton',
        'VLLM_CACHE_ROOT':'vllm','NUMBA_CACHE_DIR':'numba','FLASHINFER_WORKSPACE_BASE':'flashinfer',
        'TVM_FFI_CACHE_DIR':'tvm-ffi','CUDA_CACHE_PATH':'cuda','TORCH_EXTENSIONS_DIR':'torch-extensions'}.items()}
    variables.update(TMPDIR=str(root/'runtime/tmp'),VLLM_RPC_BASE_PATH=str(ipc),
        PYTHONPATH=str(ROOT),PATH=str(prefix/'bin')+':/usr/bin:/bin',CUDA_VISIBLE_DEVICES=gpu_uuid or '',
        PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        VLLM_NO_USAGE_STATS='1',DO_NOT_TRACK='1',
        CUDA_HOME=str(prefix/'lib/python3.12/site-packages/nvidia/cu13'))
    return variables


def profile(spec, *, root, model_directory, environment_prefix, gpu_uuid, cache_base=None):
    root = safe_path(root)
    if cache_base is None:
        base = baseline_root(root)
    else:
        # 外部标注只复用服务，不扩大 TomBaseline 正式 Run 的输出范围。
        require(root.is_relative_to('/data/yuxiao/Untrusted_Network_Simulation/tom_sft'),
                'external annotation work root required')
        base = safe_path(cache_base)
    model = safe_path(model_directory);prefix = safe_path(environment_prefix)
    require(model.is_dir() and (model.parent==MODEL_BASE or model.is_relative_to(BASE)), 'model path missing/outside registry roots')
    require(prefix.is_dir() and (prefix.parent==ENV_BASE or prefix.is_relative_to(BASE))
            and (prefix/'bin/python').is_file(),'environment interpreter missing/outside roots')
    require(type(gpu_uuid) is str and gpu_uuid.startswith('GPU-'),'explicit GPU UUID required')
    row = spec.record()
    variables = service_environment(spec,root=root,environment_prefix=prefix,cache_base=base,gpu_uuid=gpu_uuid)
    options = {'model':str(model),'tokenizer':str(model),'revision':row['revision'],
        'tokenizer-revision':row['tokenizer_revision'],'tokenizer-mode':'auto','served-model-name':[row['model']],
        'dtype':'bfloat16','generation-config':'auto','host':'127.0.0.1',
        'port':{'qwen3':18581,'yi':18582,'mistral':18583}[spec.key],
        'max-model-len':8192,'max-num-seqs':1,'gpu-memory-utilization':.9,'seed':20261014,
        'enable-prefix-caching':False,'trust-request-chat-template':False,'chat-template-content-format':'string',
        'chat-template':str(root/'runtime/chat_template.jinja'),'structured-outputs-config':{'backend':'xgrammar'}}
    return {'root':str(root),'environment_prefix':str(prefix),'gpu_uuid':gpu_uuid,
            'argv_options':options,'environment':variables,'cache_root':str(base/'runtime-cache'/spec.key),
            'ipc_root':variables['VLLM_RPC_BASE_PATH']}


def require_idle_gpu(runtime):
    """仅查询配置指定的设备；绝不停止其他进程。"""
    gpu = runtime['gpu_uuid']
    command = ['nvidia-smi','--id='+gpu]
    text = subprocess.check_output(command+['--query-gpu=uuid,name,memory.total,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'],text=True,timeout=10).strip()
    rows = text.splitlines()
    require(len(rows)==1,'one selected GPU required')
    uuid,name,total,free,util = [x.strip() for x in rows[0].split(',')]
    processes = subprocess.check_output(command+['--query-compute-apps=pid',
        '--format=csv,noheader,nounits'],text=True,timeout=10).strip()
    require(uuid==gpu and name=='NVIDIA GeForce RTX 3090' and int(total)==24576,
            'selected GPU identity mismatch')
    require(int(free)>=22528 and int(util)==0 and not processes,'selected GPU busy; no process was modified')
    return {'uuid':uuid,'name':name,'memory_total_mib':int(total)}
