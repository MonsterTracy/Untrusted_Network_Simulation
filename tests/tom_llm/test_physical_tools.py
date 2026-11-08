"""CPU/Mock 验证轻量资产和 owned-process 边界；不启动 vLLM 或真实 GPU。"""

from contextlib import contextmanager
from copy import deepcopy
import os
from pathlib import Path
import signal
from types import SimpleNamespace
import pytest

from werewolf.artifact_io import canonical_json_bytes
from werewolf.tom_llm.common import digest
from werewolf.tom_llm import physical,service_manager as sm,deployment as dep
from werewolf.tom_llm.model_spec import ModelSpec


def test_weight_verification_reads_only_small_index_revision_and_stats(tmp_path,monkeypatch):
    root = tmp_path/'model'
    root.mkdir()
    spec = ModelSpec('qwen3')
    revision = spec.record()['revision']
    (root/'HF_REVISION').write_text(revision+'\n')
    (root/'model.safetensors.index.json').write_text('{"weight_map":{"one":"one.safetensors"}}')
    with (root/'one.safetensors').open('wb') as stream:
        stream.truncate(1024**3)
    manifest = tmp_path/'assets.json'
    manifest.write_bytes(canonical_json_bytes({'qwen3':{'revision':revision,'files':{
        'one.safetensors':{'bytes':1024**3,'sha256':'a'*64}}}}))
    monkeypatch.setattr(physical,'ASSET_PATH',manifest)
    original = Path.read_bytes
    def no_weight_bytes(path):
        assert path.suffix!='.safetensors'
        return original(path)
    monkeypatch.setattr(Path,'read_bytes',no_weight_bytes)
    first = physical.verify_model(spec,root)
    assert first['full_weight_sha_verified'] is first['weights_loaded'] is False
    assert first==physical.verify_model(spec,root)
    os.utime(root/'one.safetensors',ns=(1,1))
    assert physical.verify_model(spec,root)!=first


@pytest.mark.parametrize('processes,free,util',[('42',24000,0),('',100,0),('',24000,10)])
def test_busy_selected_gpu_never_signals(processes,free,util,monkeypatch):
    calls = []
    def command(args,**kw):
        calls.append(args)
        assert '--id=GPU-selected' in args
        if '--query-compute-apps=pid' in args:
            return processes
        return f'GPU-selected, NVIDIA GeForce RTX 3090, 24576, {free}, {util}'
    monkeypatch.setattr(dep.subprocess,'check_output',command)
    with pytest.raises(ValueError,match='busy'):
        dep.require_idle_gpu({'gpu_uuid':'GPU-selected'})
    assert len(calls)==2


def manager_for(run,tmp_path):
    prefix = tmp_path/'env'
    (prefix/'bin').mkdir(parents=True)
    (prefix/'bin/python').touch()
    run.record['mode'] = 'formal'
    run.record['runtime'] = {'environment_prefix':str(prefix),'argv_options':{'port':18581}}
    manager = sm.Service(run)
    manager.directory.mkdir(parents=True)
    ident = {'pid':12345,'boot_id':'boot','start_ticks':'7','uid':os.getuid(),'cwd':str(sm.ROOT),
             'executable':str(prefix/'bin/python'),'pgid':12345,'argv':manager.command()}
    state = {'run_digest':run.artifact.manifest_digest,'argv':manager.command(),'log':'owned.log','identity':ident}
    state['state_digest'] = digest(state)
    (manager.directory/'current.json').write_bytes(canonical_json_bytes(state))
    return manager,ident


def test_status_and_stop_have_no_heavy_validation_and_signal_only_owned_pid(make_run,tmp_path,monkeypatch):
    manager,ident = manager_for(make_run(),tmp_path)
    current = [ident]
    monkeypatch.setattr(physical,'verify_model',lambda *a:pytest.fail('weight scan'))
    monkeypatch.setattr(manager,'identity',lambda pid:current[0])
    assert manager.status()['state']=='running'
    signals = []
    @contextmanager
    def fd(pid):
        assert pid==ident['pid']
        def send(sig):
            signals.append(sig)
            current[0] = None
        yield send
    monkeypatch.setattr(sm,'_pidfd',fd)
    monkeypatch.setattr(sm.sys,'platform','linux')
    assert manager.stop(execute=True)['state']=='exited'
    assert signals==[signal.SIGTERM]


@pytest.mark.parametrize('change',('pid_reused','pending','state_tampered'))
def test_nonowned_or_unknown_launch_never_signals(make_run,tmp_path,monkeypatch,change):
    manager,ident = manager_for(make_run(),tmp_path)
    monkeypatch.setattr(sm.sys,'platform','linux')
    monkeypatch.setattr(sm,'_pidfd',lambda *a:pytest.fail('unsafe signal'))
    if change=='pending':
        manager.pending_launch.write_text('{}')
        assert manager.status()['state']=='launch_incomplete'
    elif change=='state_tampered':
        path = manager.directory/'current.json'
        path.write_text(path.read_text().replace('owned.log','other.log'))
    else:
        monkeypatch.setattr(manager,'identity',lambda _:dict(ident,start_ticks='99'))
        assert manager.status()['state']=='pid_reused'
    with pytest.raises(ValueError):
        manager.stop(execute=True)


def test_direct_service_command_has_one_entrypoint_and_no_qualification(make_run,tmp_path):
    manager,_ = manager_for(make_run(),tmp_path)
    command = manager.command()
    assert command[1:3]==['-m','vllm.entrypoints.openai.api_server']
    assert not any('qualification' in part or 'gpu_probe' in part or 'live_service' in part for part in command)


def test_mock_service_launch_reuses_instance_and_holds_shared_gpu_lock(make_run,tmp_path,monkeypatch):
    import fcntl
    manager,ident = manager_for(make_run(),tmp_path)
    (manager.directory/'current.json').unlink()
    runtime = manager.runtime
    runtime.update(cache_root=str(tmp_path/'cache'),ipc_root=str(tmp_path/'ipc'),
                   environment={'TMPDIR':str(tmp_path/'temp')})
    runtime['argv_options']['model'] = 'not-loaded'
    runtime['argv_options']['port'] = 0
    manager.run.record['model_evidence'] = {'mock':'stat'}
    monkeypatch.setattr(physical,'verify_model',lambda *a:{'mock':'stat'})
    monkeypatch.setattr(sm.sys,'platform','linux')
    monkeypatch.setattr(sm,'GPU_LOCK',tmp_path/'gpu.lock')
    monkeypatch.setattr(sm,'require_idle_gpu',lambda *a:None)
    monkeypatch.setattr(sm.subprocess,'check_output',lambda *a,**kw:'0.27.0\n')
    current = [ident]
    monkeypatch.setattr(manager,'identity',lambda _:current[0])
    monkeypatch.setattr(manager,'ready',lambda:{'ready':True})
    signals = []
    @contextmanager
    def fd(pid):
        def send(sig):
            if sig:
                assert pid==ident['pid']
                signals.append(sig)
                current[0] = None
        yield send
    monkeypatch.setattr(sm,'_pidfd',fd)
    launched = []
    def spawn(command,**kw):
        launched.append(command)
        assert command==manager.command() and kw['start_new_session']
        assert len(kw['pass_fds'])==1
        with (tmp_path/'gpu.lock').open('a+b') as other:
            with pytest.raises(BlockingIOError):
                fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert manager.pending_launch.exists()
        return SimpleNamespace(pid=ident['pid'],wait=lambda **kw:0)
    monkeypatch.setattr(sm.subprocess,'Popen',spawn)
    assert manager.start(execute=True)=={'ready':True}
    assert not manager.pending_launch.exists()
    assert manager.start(execute=True)=={'ready':True}
    assert len(launched)==1
    assert manager.stop(execute=True)['state']=='exited' and signals==[signal.SIGTERM]


def test_native_runtime_same_decoding_template_and_cache_boundaries(monkeypatch):
    from tempfile import TemporaryDirectory
    with TemporaryDirectory(dir='/private/tmp',prefix='tb-') as temporary:
        _runtime_paths(Path(temporary),monkeypatch)


def _runtime_paths(tmp_path,monkeypatch):
    root = tmp_path/'outputs'/'one'
    model = tmp_path/'models'/'model'
    prefix = tmp_path/'envs'/'conda'
    model.mkdir(parents=True)
    (prefix/'bin').mkdir(parents=True)
    (prefix/'bin/python').touch()
    monkeypatch.setattr(dep,'EXPERIMENT_BASE',tmp_path/'outputs')
    monkeypatch.setattr(dep,'MODEL_BASE',model.parent)
    monkeypatch.setattr(dep,'ENV_BASE',prefix.parent)
    for key in ('qwen3','yi','mistral'):
        profile = dep.profile(ModelSpec(key),root=root,model_directory=model,environment_prefix=prefix,gpu_uuid='GPU-example')
        options = profile['argv_options']
        assert options['max-model-len']==8192 and options['max-num-seqs']==1 and options['seed']==20261014
        assert options['structured-outputs-config']=={'backend':'xgrammar'}
        assert options['chat-template']==str(root/'runtime/chat_template.jinja')
        assert profile['environment']['CUDA_VISIBLE_DEVICES']=='GPU-example'
        assert profile['environment']['HF_HUB_OFFLINE']=='1'
        encoded = dep.vllm_config_bytes(options)
        assert b'"no-enable-prefix-caching":true' in encoded
    assert not root.exists()


@pytest.fixture
def listener_fixture(tmp_path,monkeypatch):
    """只模拟当前 owned PID 的 /proc，不模拟整个 Linux/GPU 环境。"""
    import json
    root = tmp_path/'proc'
    process = root/'12345'
    (process/'fd').mkdir(parents=True)
    (process/'net').mkdir()
    ident = {'pid':12345,'boot_id':'boot','start_ticks':'7','uid':os.getuid()}
    spec = ModelSpec('qwen3')
    run = SimpleNamespace(root=tmp_path,spec=spec,
        record={'mode':'formal','runtime':{'argv_options':{'port':18581}}})
    manager = sm.Service(run)
    monkeypatch.setattr(sm,'_PROC',root,raising=False)
    monkeypatch.setattr(sm.sys,'platform','linux')
    monkeypatch.setattr(manager,'_state',lambda:{'identity':ident,'log':'not-opened.log'})
    monkeypatch.setattr(manager,'identity',lambda pid:dict(ident))
    monkeypatch.setattr(physical,'verify_model',lambda *a:pytest.fail('weights in ownership check'))
    def install(*,owned=('41','42'),listener='41',extra=(),peer='42'):
        for path in (process/'fd').iterdir():
            path.unlink()
        for i,inode in enumerate(owned):
            (process/'fd'/str(i)).symlink_to('socket:['+inode+']')
        def row(local,remote,state,inode):
            return f'0: {local} {remote} {state} 0:0 0:0 0 {os.getuid()} 0 {inode}\n'
        lines = [row('0100007F:4895','00000000:0000','0A',listener)]
        lines += [row('0100007F:4895','00000000:0000','0A',inode) for inode in extra]
        if peer is not None:
            lines += [row('0100007F:4895','0100007F:9C40','01',peer)]
        (process/'net/tcp').write_text('header\n'+''.join(lines))
        (process/'net/tcp6').write_text('header\n')
    install()
    calls = []
    class Socket:
        def getsockname(self):return ('127.0.0.1',40000)
        def getpeername(self):return ('127.0.0.1',18581)
        def dup(self):return self
        def close(self):pass
    class Connection:
        def __init__(self,*args,**kw):self.sock=Socket()
        def connect(self):pass
        def request(self,*args,**kw):calls.append(args)
        def getresponse(self):
            return SimpleNamespace(status=200,read=lambda:json.dumps({'data':[{'id':spec.record()['model']}]}).encode())
        def close(self):pass
    monkeypatch.setattr('http.client.HTTPConnection',Connection)
    return manager,ident,install,calls


def test_wrong_listener_with_correct_model_name_is_rejected(listener_fixture):
    manager,ident,install,calls = listener_fixture
    install(listener='99')
    with pytest.raises(RuntimeError,match='listener'):
        manager.ready()
    assert calls==[]


def test_owned_listener_and_connected_socket_allow_ready(listener_fixture):
    manager,ident,install,calls = listener_fixture
    binding = manager.guard()
    assert binding=={'identity':ident,'listener_inode':'41'}
    assert manager.ready()['ready'] is True
    assert calls==[('GET','/v1/models')]


@pytest.mark.parametrize('change',('multiple','replaced','lifecycle','missing'))
def test_listener_binding_and_lifecycle_cannot_change(listener_fixture,monkeypatch,change):
    manager,ident,install,calls = listener_fixture
    binding = manager.guard()
    if change=='multiple':
        install(extra=('99',))
    elif change=='replaced':
        install(owned=('43','42'),listener='43')
    elif change=='missing':
        install(listener='99',owned=())
    else:
        monkeypatch.setattr(manager,'identity',lambda _:dict(ident,start_ticks='8'))
    with pytest.raises(RuntimeError,match='listener'):
        manager.guard(expected=binding)


def test_connected_peer_must_be_held_by_owned_pid(listener_fixture,monkeypatch):
    manager,ident,install,calls = listener_fixture
    install(peer='99')  # 监听属于 owned PID，实际连接却由其他进程持有。
    ticks = iter(range(10))
    monkeypatch.setattr(sm.time,'monotonic',lambda:next(ticks))
    monkeypatch.setattr(sm.time,'sleep',lambda _:None)
    with pytest.raises(RuntimeError,match='connected socket'):
        manager.ready()
    assert calls==[]


def test_accept_queue_waits_for_owned_fd_without_http_retry(listener_fixture,monkeypatch):
    manager,ident,install,calls = listener_fixture
    install(peer=None)
    waits = []
    def accept(_):
        waits.append('accept')
        install()
    monkeypatch.setattr(sm.time,'sleep',accept)
    assert manager.ready()['ready'] is True
    assert waits==['accept'] and calls==[('GET','/v1/models')]


def test_ipv6_wildcard_cannot_hide_an_unowned_listener(listener_fixture):
    manager,ident,install,calls = listener_fixture
    process = sm._PROC/str(ident['pid'])
    (process/'net/tcp6').write_text('header\n0: '+'0'*32+':4895 '+'0'*32+
                                   f':0000 0A 0:0 0:0 0 {os.getuid()} 0 99\n')
    with pytest.raises(RuntimeError,match='listener'):
        manager.ready()
    assert calls==[]


@pytest.mark.parametrize('failure',('permission','pid_reused'))
def test_ownership_evidence_unavailable_is_terminal(listener_fixture,monkeypatch,failure):
    manager,ident,install,calls = listener_fixture
    if failure=='permission':
        monkeypatch.setattr(sm.os,'readlink',lambda *_:(_ for _ in ()).throw(PermissionError('denied')))
    else:
        identities = iter((dict(ident),dict(ident,start_ticks='8')))
        monkeypatch.setattr(manager,'identity',lambda _:next(identities))
    with pytest.raises(RuntimeError):
        manager.ready()
    assert calls==[]
