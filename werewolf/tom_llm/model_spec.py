"""固定模型注册表、原生模板、Wire 和真实 CPU Tokenizer 校验。"""

from dataclasses import dataclass, field
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

from jinja2 import Environment, StrictUndefined

from werewolf.artifact_io import sha256_bytes,canonical_json_bytes
from werewolf.tom_llm.common import digest, require, strict_json
from werewolf.tom_llm.policy import policy_binding
from werewolf.tom_llm.task import validate_task, task_version, SET_VERSION

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / 'configs/paper/tom-llm-models-v1.json'
REGISTRY_SHA256 = 'c50b75a8d9580434e9fc327dca0a866c09fa3503f61707b4ffe5ab3db73c97cf'
SET_WIRE_SCHEMA_PROJECTION = {
    'schema_version':'classic7_suspicion_set_xgrammar_wire_projection_v1',
    'backend':'xgrammar','vllm_version':'0.27.0',
    'removed_keywords':['/properties/suspected_players/uniqueItems'],
    'duplicate_policy':'application_reject_duplicate_player_without_repair',
}


def registry():
    policy_binding()  # Preserve the legacy task/resources as independent trust anchors.
    raw = REGISTRY.read_bytes()
    require(sha256_bytes(raw) == REGISTRY_SHA256, 'model registry identity drift')
    rows = strict_json(raw)
    require(set(rows) == {'qwen3', 'yi', 'mistral'}, 'unauthorized model inventory')
    return rows


@dataclass(frozen=True)
class ModelSpec:
    key: str
    _record: dict = field(init=False,repr=False,compare=False)
    _template: bytes = field(init=False,repr=False,compare=False)

    def __post_init__(self):
        rows = registry()
        require(self.key in rows, 'unauthorized model identity')
        row = rows[self.key]
        object.__setattr__(self,'_template',template_bytes(self,row=row))
        object.__setattr__(self,'_record',row)

    def record(self):
        return deepcopy(self._record)

    @property
    def digest(self):
        return digest(self.record())


def template_bytes(spec, *, row=None):
    require(type(spec) is ModelSpec, 'explicit ModelSpec required')
    if row is None:
        return spec._template
    p = Path(__file__).parent / row['chat_template']['path']
    require(p.is_file() and not any(x.is_symlink() for x in (p, *p.parents)), 'model template symlink/missing')
    raw = p.read_bytes()
    require(sha256_bytes(raw) == row['chat_template']['sha256'], 'model template drift')
    return raw


def render(request, spec):
    """Same validated task messages/schema; only native model serialization differs."""
    validate_task(request)
    row = spec.record()
    tokens = row['tokenizer']
    def refuse(message):
        raise ValueError(message)
    env = Environment(undefined=StrictUndefined, autoescape=False)
    env.globals['raise_exception'] = refuse
    text = env.from_string(template_bytes(spec).decode()).render(
        messages=request['messages'], bos_token=tokens['bos_token'], eos_token=tokens['eos_token'],
        add_generation_prompt=True, **row['chat_template']['kwargs'])
    return {'model_spec_digest': spec.digest, 'task_digest': digest(request),
            'text': text, 'text_sha256': sha256_bytes(text.encode())}


def wire_schema(request):
    """仅投影后端不支持的唯一性约束，原任务、Prompt 和应用层校验不变。"""
    validate_task(request)
    require(task_version(request)==SET_VERSION,'only suspicion-set tasks are executable')
    schema=deepcopy(request['response_schema'])
    del schema['properties']['suspected_players']['uniqueItems']
    return schema


def wire(request, spec):
    schema=wire_schema(request)
    row = spec.record()
    d = row['decoding']
    extra = {'chat_template_kwargs': dict(row['chat_template']['kwargs']), 'top_k': d['top_k'], 'min_p': d['min_p']}
    if row['termination']['extra_stop_token_ids']:
        extra['stop_token_ids'] = row['termination']['extra_stop_token_ids']
    return {'model': row['model'], 'messages': request['messages'], 'temperature': d['temperature'],
            'top_p': d['top_p'], 'n': d['n'], 'max_tokens': d['max_tokens'], 'seed': d['seed'],
            'response_format': {'type': 'json_schema', 'json_schema': {
                'name': 'non_self_suspicion_set',
                'strict': True, 'schema': schema}},
            'extra_body': extra}


def verify_cpu_assets(spec, directory):
    """Only locally supplied, exact official small assets; never downloads or loads weights."""
    row = spec.record()
    root = Path(directory)
    require(root.is_dir() and not any(x.is_symlink() for x in (root, *root.parents)), 'CPU asset directory missing/symlink')
    # AutoTokenizer can implicitly consume extra local template/config files. Refuse them.
    audit_only = {'official-api.json', 'snapshot.json', 'LICENSE', 'README.md'}
    require({p.name for p in root.iterdir()} <= set(row['assets']) | audit_only,
            'unbound CPU asset inventory')
    for name, expected in row['assets'].items():
        p = root / name
        require(p.is_file() and not p.is_symlink(), 'CPU asset missing/symlink: ' + name)
        raw = p.read_bytes()
        require(len(raw) == expected['bytes'] and sha256_bytes(raw) == expected['sha256'], 'CPU asset identity drift: ' + name)
    return root


def load_cpu_tokenizer(spec, directory):
    root = verify_cpu_assets(spec, directory)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    expected = spec.record()['tokenizer']
    require(tokenizer.bos_token_id == expected['bos_token_id']
            and tokenizer.eos_token_id == expected['eos_token_id'], 'native special token identity drift')
    return tokenizer




def check_cpu_parity(requests, spec, directory):
    """Compare native official chat template to this adapter, including every token ID."""
    tokenizer = load_cpu_tokenizer(spec, directory)
    spec_row = spec.record()
    lengths = []
    states = [sha256(b'[') for _ in range(3)]
    for ordinal,request in enumerate(requests):
        ours = render(request, spec)['text']
        kwargs = {'add_generation_prompt': True, **spec_row['chat_template']['kwargs']}
        official = tokenizer.apply_chat_template(request['messages'], tokenize=False, **kwargs)
        actual = tokenizer.encode(ours, add_special_tokens=False)
        expected = tokenizer.apply_chat_template(request['messages'], tokenize=True, return_dict=False, **kwargs)
        require(ours == official and actual == expected, 'native template/text/input_ids mismatch')
        lengths.append(len(actual))
        # 摘要与完整 canonical JSON 数组一致，但不保留全量文本和 token IDs。
        for state,value in zip(states,(ours,actual,wire(request,spec)),strict=True):
            if ordinal:
                state.update(b',')
            state.update(canonical_json_bytes(value))
    for state in states:
        state.update(b']')
    require(lengths and all(0<n<=7680 for n in lengths),'complete PRE exceeds context; no truncation')
    counts = {'request_count':len(lengths),'input_token_counts':lengths,'input_tokens_total':sum(lengths),
              'maximum_input_tokens':max(lengths)}
    result = {'status': 'CPU_PARITY_PASS', 'model_spec_digest': spec.digest, 'weights_verified': False,
            'model': spec_row['model'], 'revision': spec_row['revision'],
            'text_sequence_digest':states[0].hexdigest(),'input_ids_sequence_digest':states[1].hexdigest(),
            **counts}
    require(all(task_version(r)==SET_VERSION for r in requests),'only set tasks may enter CPU parity')
    result.update(wire_schema_projection=deepcopy(SET_WIRE_SCHEMA_PROJECTION),
                  wire_sequence_digest=states[2].hexdigest())
    return result
