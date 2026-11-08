"""共用模型适配与原生模板；不构建第二套 Mock Runner。"""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import pytest

from tests.canonical_collection.test_structured_token_plan import _prefix
from werewolf.artifact_io import canonical_json_bytes
from werewolf.tom_llm import model_spec as ms
from werewolf.tom_llm.common import digest
from werewolf.tom_llm.model_spec import ModelSpec,render,wire,check_cpu_parity
from werewolf.tom_llm.study import make_request
from werewolf.tom_llm.task import from_legacy,SET_VERSION
from werewolf.tom_llm.output import parse_set_output

KEYS = ('qwen3','yi','mistral')


@pytest.fixture
def tasks():
    prefix = _prefix()
    observers = [int(p.removeprefix('player'))-1 for p in prefix.alive_observer_ids]
    return [from_legacy(make_request(prefix,o,fold=0),version=SET_VERSION) for o in observers]


@pytest.mark.parametrize('key',KEYS)
def test_native_wire_preserves_scientific_task(tasks,key):
    spec = ModelSpec(key)
    for task in tasks:
        original = canonical_json_bytes(task)
        request = wire(task,spec)
        text = render(task,spec)['text']
        expected = deepcopy(task['response_schema'])
        assert expected['properties']['suspected_players'].pop('uniqueItems') is True
        assert request['response_format']['json_schema']['schema']==expected
        assert request['messages']==task['messages'] and canonical_json_bytes(task)==original
        for keyword in ('uniqueItems','contains','minContains','maxContains','multipleOf','patternProperties','propertyNames'):
            assert ('"'+keyword+'":').encode() not in canonical_json_bytes(expected)
        assert '"uniqueItems":true' in text
        assert all(m['content'] in text for m in task['messages'])
        assert request['temperature']==0 and request['max_tokens']==512 and request['n']==1
        assert request['seed']==20261014
        if key=='qwen3':
            assert request['extra_body']['chat_template_kwargs']=={'enable_thinking':False}
        elif key=='yi':
            assert request['extra_body']['stop_token_ids']==[7] and text.endswith('<|im_start|>assistant\n')
        else:
            assert text.startswith('<s>[INST] ') and text.endswith('[/INST]')


@pytest.mark.parametrize('key',KEYS)
def test_wire_mock_parity_uses_complete_actual_tasks_and_same_schema(tasks,key,monkeypatch):
    spec = ModelSpec(key)
    texts = {digest(t['messages']):render(t,spec)['text'] for t in tasks}
    class Tokenizer:
        def encode(self,text,**kw):
            return list(bytes.fromhex(digest(text)))
        def apply_chat_template(self,messages,*,tokenize,**kw):
            value = texts[digest(messages)]
            return self.encode(value) if tokenize else value
    monkeypatch.setattr(ms,'load_cpu_tokenizer',lambda *a:Tokenizer())
    result = check_cpu_parity(tasks,spec,'synthetic')
    assert result['request_count']==len(tasks) and result['input_token_counts']==[32]*len(tasks)
    assert result['input_ids_sequence_digest']==digest([Tokenizer().encode(render(t,spec)['text']) for t in tasks])
    assert result['wire_sequence_digest']==digest([wire(t,spec) for t in tasks])
    assert result['wire_schema_projection']==ms.SET_WIRE_SCHEMA_PROJECTION
    assert result['weights_verified'] is False


@pytest.mark.parametrize('key',KEYS)
def test_official_assets_missing_never_pass_parity(tmp_path,tasks,key):
    with pytest.raises(ValueError,match='missing'):
        check_cpu_parity(tasks,ModelSpec(key),tmp_path/'absent')


def test_old_probability_task_not_executable_on_new_mainline():
    task = from_legacy(make_request(_prefix(),0,fold=0))
    with pytest.raises(ValueError,match='only suspicion-set'):
        wire(task,ModelSpec('qwen3'))


def test_registry_and_native_template_are_read_once_per_instance(tasks,monkeypatch):
    spec = ModelSpec('qwen3')
    monkeypatch.setattr(ms,'registry',lambda:pytest.fail('registry reread per request'))
    original = ms.template_bytes
    def template(s, *, row=None):
        assert row is None
        return original(s)
    monkeypatch.setattr(ms,'template_bytes',template)
    for task in tasks:
        wire(task,spec)
        render(task,spec)


def test_duplicates_invalid_empty_and_all_six_stay_legal(tasks):
    for task in tasks:
        players = task['response_schema']['properties']['suspected_players']['items']['enum']
        for output in ([],players[:1],players):
            assert parse_set_output(json.dumps({'suspected_players':output}),task['observer']).to_record()['status']=='valid'
        bad = parse_set_output(json.dumps({'suspected_players':[players[0]]*2}),task['observer']).to_record()
        assert bad['failure_type']=='duplicate_player' and bad['suspected_players'] is None


@pytest.mark.parametrize('count',(7680,7681))
def test_native_cpu_budget_checks_full_prompt_without_truncation(tasks,monkeypatch,count):
    spec = ModelSpec('qwen3')
    task = tasks[0]
    text = render(task,spec)['text']
    class Tokenizer:
        def encode(self,value,**kw):
            assert value==text
            return [1]*count
        def apply_chat_template(self,messages,*,tokenize,**kw):
            assert messages==task['messages']
            return [1]*count if tokenize else text
    monkeypatch.setattr(ms,'load_cpu_tokenizer',lambda *a:Tokenizer())
    if count==7680:
        assert check_cpu_parity([task],spec,'synthetic')['maximum_input_tokens']==count
    else:
        with pytest.raises(ValueError,match='no truncation'):
            check_cpu_parity([task],spec,'synthetic')


def test_frozen_task_prompt_assets_and_scoring_bytes_unchanged():
    root = Path(__file__).resolve().parents[2]
    paths = ['task.py','prompt.py','policy.py','reference.py']
    for name in paths:
        relative = 'werewolf/tom_llm/'+name
        assert (root/relative).read_bytes()==subprocess.check_output(['git','show','bacc351:'+relative],cwd=root)
    for relative in ('werewolf/tom/scoring.py','werewolf/tom/evaluation.py'):
        assert (root/relative).read_bytes()==subprocess.check_output(['git','show','bacc351:'+relative],cwd=root)
    import ast
    path = root/'werewolf/tom_llm/output.py'
    before = ast.parse(subprocess.check_output(['git','show','bacc351:werewolf/tom_llm/output.py'],cwd=root))
    previous = {n.name:ast.dump(n) for n in before.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)):
            assert ast.dump(node)==previous[node.name]
