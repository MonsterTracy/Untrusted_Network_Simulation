"""Closed, versioned CPU/Mock model registry. No model or service admission."""

from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment, StrictUndefined

from werewolf.artifact_io import sha256_bytes
from werewolf.tom_llm.common import digest, require, strict_json
from werewolf.tom_llm.policy import policy_binding
from werewolf.tom_llm.task import validate_task, to_legacy
from werewolf.tom_llm.server_fixed import wire_request_v2

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / 'configs/paper/tom-llm-models-v1.json'
REGISTRY_SHA256 = 'c50b75a8d9580434e9fc327dca0a866c09fa3503f61707b4ffe5ab3db73c97cf'


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

    def record(self):
        rows = registry()
        require(self.key in rows, 'unauthorized model identity')
        row = rows[self.key]
        template_bytes(self, row=row)
        return row

    @property
    def digest(self):
        return digest(self.record())


def template_bytes(spec, *, row=None):
    require(type(spec) is ModelSpec, 'explicit ModelSpec required')
    row = registry()[spec.key] if row is None else row
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


def wire(request, spec):
    validate_task(request)
    row = spec.record()
    if spec.key == 'qwen3':
        return wire_request_v2(to_legacy(request))  # Exact original Wire v2 bytes, no extra fields.
    d = row['decoding']
    extra = {'chat_template_kwargs': {}, 'top_k': d['top_k'], 'min_p': d['min_p']}
    if row['termination']['extra_stop_token_ids']:
        extra['stop_token_ids'] = row['termination']['extra_stop_token_ids']
    return {'model': row['model'], 'messages': request['messages'], 'temperature': d['temperature'],
            'top_p': d['top_p'], 'n': d['n'], 'max_tokens': d['max_tokens'], 'seed': d['seed'],
            'response_format': {'type': 'json_schema', 'json_schema': {
                'name': 'non_self_suspicion', 'strict': True, 'schema': request['response_schema']}},
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


def token_preflight(requests, spec, *, count_tokens):
    counts = [count_tokens(render(r, spec)['text']) for r in requests]
    row = spec.record()
    require(counts and all(type(n) is int and 0 < n <= row['context_length'] - row['decoding']['max_tokens']
                          for n in counts), 'complete PRE exceeds context; truncation forbidden')
    return {'request_count': len(counts), 'input_token_counts': counts, 'input_tokens_total': sum(counts),
            'maximum_input_tokens': max(counts)}


def check_cpu_parity(requests, spec, directory):
    """Compare native official chat template to this adapter, including every token ID."""
    tokenizer = load_cpu_tokenizer(spec, directory)
    spec_row = spec.record()
    texts, token_ids = [], []
    for request in requests:
        ours = render(request, spec)['text']
        kwargs = {'add_generation_prompt': True, **spec_row['chat_template']['kwargs']}
        official = tokenizer.apply_chat_template(request['messages'], tokenize=False, **kwargs)
        actual = tokenizer.encode(ours, add_special_tokens=False)
        expected = tokenizer.apply_chat_template(request['messages'], tokenize=True, return_dict=False, **kwargs)
        require(ours == official and actual == expected, 'native template/text/input_ids mismatch')
        texts.append(ours)
        token_ids.append(actual)
        if spec.key == 'qwen3':
            require(wire(request, spec) == wire_request_v2(to_legacy(request)), 'frozen Qwen wire compatibility failure')
            require(ours == to_legacy(request)['rendered_prompt'] and actual[-7:] == [151644,77091,198,151667,271,151668,271],
                    'frozen Qwen text/token compatibility failure')
    counts = token_preflight(requests, spec, count_tokens=lambda t: len(tokenizer.encode(t, add_special_tokens=False)))
    return {'status': 'CPU_PARITY_PASS', 'model_spec_digest': spec.digest, 'weights_verified': False,
            'gpu_admission': False, 'model': spec_row['model'], 'revision': spec_row['revision'],
            'text_sequence_digest': digest(texts), 'input_ids_sequence_digest': digest(token_ids),
            'legacy_qwen_compatibility': spec.key == 'qwen3', **counts}
