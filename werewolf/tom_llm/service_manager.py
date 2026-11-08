"""单一 vLLM 服务的持久所有权；状态和停止不读取模型、Cohort 或 Source 库存。"""

from contextlib import contextmanager
import ctypes
import fcntl
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import uuid

from werewolf.artifact_io import canonical_json_bytes
from werewolf.artifact_io.canonical import _fsync_directory,ensure_durable_directory
from werewolf.tom_llm.common import digest,require,strict_json
from werewolf.tom_llm.experiment import safe_path
from werewolf.tom_llm.model_spec import ROOT,template_bytes
from werewolf.tom_llm.deployment import GPU_LOCK,require_idle_gpu,vllm_config_bytes

_PROC = Path('/proc')

@contextmanager
def _pidfd(pid):
    """Linux x86-64 LP64 pidfd, independent of optional CPython bindings."""
    require(sys.platform=='linux' and os.uname().machine=='x86_64'
            and ctypes.sizeof(ctypes.c_void_p)==ctypes.sizeof(ctypes.c_long)==8,
            'pidfd syscalls require Linux x86-64 LP64')
    require(type(pid) is int and 0<pid<2**31,'positive pid_t required')
    # Linux v6.14 arch/x86/entry/syscalls/syscall_64.tbl: 434=open, 424=send.
    # https://man7.org/linux/man-pages/man2/pidfd_open.2.html
    # https://man7.org/linux/man-pages/man2/pidfd_send_signal.2.html
    syscall=ctypes.CDLL(None,use_errno=True).syscall
    syscall.restype=ctypes.c_long
    syscall.argtypes=(ctypes.c_long,)  # Variadic arguments carry explicit C types.
    def call(number,*args):
        result=syscall(ctypes.c_long(number),*args)
        if result==-1:
            error=ctypes.get_errno()
            raise OSError(error,'pidfd syscall '+str(number)+': '+os.strerror(error))
        return result
    fd=call(434,ctypes.c_int(pid),ctypes.c_uint(0))  # Kernel sets close-on-exec.
    try:
        def send(sig):return call(424,ctypes.c_int(fd),ctypes.c_int(sig),ctypes.c_void_p(),ctypes.c_uint(0))
        yield send
    finally:os.close(fd)


class Service:
    def __init__(self, run):
        require(run.record['mode']=='formal' and run.record['runtime'],'real Run runtime required')
        self.run = run
        self.runtime = run.record['runtime']
        self.directory = safe_path(run.root/'runtime/service-control')
        self.pending_launch = self.directory/'launch.pending.json'
        self.child = None

    def command(self):
        return [self.runtime['environment_prefix']+'/bin/python','-m','vllm.entrypoints.openai.api_server',
                '--config',str(self.run.root/'runtime/vllm.yaml')]

    def identity(self, pid):
        proc = Path('/proc')/str(pid)
        try:
            stat = (proc/'stat').read_text().rsplit(')',1)[1].split()
            if stat[0]=='Z':
                return None
            return {'pid':pid,'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                    'start_ticks':stat[19],'uid':proc.stat().st_uid,'cwd':str((proc/'cwd').resolve()),
                    'executable':str((proc/'exe').resolve()),'pgid':os.getpgid(pid),
                    'argv':[v.decode() for v in (proc/'cmdline').read_bytes().split(b'\0') if v]}
        except FileNotFoundError:
            return None

    def _state(self):
        require(not self.pending_launch.exists(),'incomplete launch ownership; no automatic adoption/stop')
        path = safe_path(self.directory/'current.json')
        if not path.exists():
            return None
        state = strict_json(path.read_bytes())
        wanted = state.pop('state_digest')
        require(digest(state)==wanted and state['run_digest']==self.run.artifact.manifest_digest
                and state['argv']==self.command(),'service ownership record drift')
        ident = state['identity']
        require(ident['argv']==self.command() and ident['uid']==os.getuid()
                and ident['cwd']==str(ROOT) and ident['pgid']==ident['pid']
                and ident['executable']==str(Path(self.command()[0]).resolve()),'stored process ownership drift')
        return state

    def status(self):
        if self.pending_launch.exists():
            return {'state':'launch_incomplete','ready':False}
        state = self._state()
        if state is None:
            return {'state':'not_started','ready':False}
        current = self.identity(state['identity']['pid'])
        return {'state':'exited' if current is None else 'running' if current==state['identity'] else 'pid_reused',
                'pid':state['identity']['pid'],'log':state['log'],'ready':False}

    def _socket_view(self, pid):
        """只读取 owned PID 的 FD 和所在网络命名空间的目标端口。"""
        process = _PROC/str(pid)
        inodes = set()
        rows = []
        try:
            for fd in (process/'fd').iterdir():
                try:
                    target = os.readlink(fd)
                except FileNotFoundError:
                    continue  # FD 在遍历期间正常关闭；随后仍须取得完整归属证明。
                if target.startswith('socket:[') and target.endswith(']'):
                    inodes.add(target[8:-1])
            for table in ('tcp','tcp6'):
                try:
                    text = (process/'net'/table).read_text()
                except FileNotFoundError:
                    if table=='tcp6':
                        continue  # 内核可能禁用 IPv6。
                    raise
                for line in text.splitlines()[1:]:
                    v = line.split()
                    if v and int(v[1].split(':')[1],16)==self.runtime['argv_options']['port']:
                        rows.append((table,v[1],v[2],v[3],int(v[7]),v[9]))
        except (OSError,ValueError,IndexError) as error:
            raise RuntimeError('cannot prove owned listener from /proc') from error
        return inodes,rows

    def guard(self, connection=None, expected=None):
        """绑定生命周期、唯一监听 inode，并在发送前证明已连接 socket 归属。"""
        require(sys.platform=='linux','listener ownership requires Linux /proc')
        port = self.runtime['argv_options']['port']
        local = f'0100007F:{port:04X}'
        deadline = time.monotonic()+1
        while True:
            state = self._state()
            if not state or self.identity(state['identity']['pid'])!=state['identity']:
                raise RuntimeError('owned listener process exited/changed')
            ident = state['identity']
            inodes,rows = self._socket_view(ident['pid'])
            addresses = {'0100007F','00000000','00000000000000000000000000000000',
                         '0000000000000000FFFF00000100007F'}
            listeners = [r for r in rows if r[3]=='0A' and r[1].split(':')[0] in addresses]
            if not listeners and expected is None:
                raise ValueError('owned listener not ready')
            if (len(listeners)!=1 or listeners[0][:2]!=('tcp',local)
                    or listeners[0][4]!=ident['uid'] or listeners[0][5] not in inodes):
                raise RuntimeError('listener is missing, ambiguous or not owned by recorded PID')
            binding = {'identity':ident,'listener_inode':listeners[0][5]}
            if expected is not None and binding!=expected:
                raise RuntimeError('owned listener binding changed')
            if connection is None:
                break
            source = connection.getsockname()
            if connection.getpeername()!=('127.0.0.1',port) or source[0]!='127.0.0.1':
                raise RuntimeError('connection endpoint differs from owned listener')
            remote = f'0100007F:{source[1]:04X}'
            peers = [r for r in rows if r[:4]==('tcp',local,remote,'01')]
            if len(peers)==1 and peers[0][4]==ident['uid'] and peers[0][5] in inodes:
                break
            if time.monotonic()>=deadline:
                raise RuntimeError('connected socket not owned by recorded PID')
            # connect 可以先于服务 accept；仅短暂等待内核证据，不发送或重试 HTTP。
            time.sleep(.005)
        if self.identity(ident['pid'])!=ident:
            raise RuntimeError('owned listener lifecycle changed during check')
        return binding

    def ready(self):
        from http.client import HTTPConnection
        binding = self.guard()
        connection = HTTPConnection('127.0.0.1',self.runtime['argv_options']['port'],timeout=2)
        pinned = None
        try:
            connection.connect()
            connection.auto_open = 0
            pinned = connection.sock.dup()
            endpoints = (pinned.getsockname(),pinned.getpeername())
            self.guard(pinned,binding)
            connection.request('GET','/v1/models')
            reply = connection.getresponse()
            require(reply.status==200,'service not ready')
            data = strict_json(reply.read())
            self.guard(expected=binding)
            require((pinned.getsockname(),pinned.getpeername())==endpoints,'ready connection changed')
            require(self.run.spec.record()['model'] in [r['id'] for r in data['data']],'served model mismatch')
        finally:
            connection.close()
            if pinned is not None:
                pinned.close()
        return {**self.status(),'ready':True}

    def start(self, *, execute=False, timeout=1800):
        require(execute and sys.platform=='linux','explicit --execute-live on Ubuntu required')
        from werewolf.tom_llm.physical import verify_model
        require(verify_model(self.run.spec,self.runtime['argv_options']['model'])==self.run.record['model_evidence'],
                'model revision/stat changed since preparation')
        with _pidfd(os.getpid()) as send:
            send(0)
        state = self.status()['state']
        if state=='running':
            return self.ready()
        require(state in ('not_started','exited'),'unresolved launch/PID ownership; no adoption or signal')
        ensure_durable_directory(self.directory)
        runtime = self.run.root/'runtime'
        for name,raw in [('chat_template.jinja',template_bytes(self.run.spec)),
                         ('vllm.yaml',vllm_config_bytes(self.runtime['argv_options']))]:
            path = safe_path(runtime/name)
            if path.exists():
                require(path.read_bytes()==raw,'runtime config drift')
            else:
                with path.open('xb') as stream:
                    stream.write(raw);stream.flush();os.fsync(stream.fileno())
        for path in [self.runtime['cache_root'],self.runtime['ipc_root'],self.runtime['environment']['TMPDIR'],
                     *[v for k,v in self.runtime['environment'].items() if k.endswith(('_HOME','_DIR','_ROOT','_PATH')) and k not in ('PYTHONPATH','PATH','CUDA_HOME')]]:
            target = safe_path(path)
            require(not any((p/'manifest.json').exists() for p in (target,*target.parents)),'cache inside sealed evidence')
            target.mkdir(parents=True,exist_ok=True)
        prefix = self.runtime['environment_prefix']
        version = subprocess.check_output([prefix+'/bin/python','-I','-B','-c',
            "import importlib.metadata as m;print(m.version('vllm'))"],
            env={'PATH':prefix+'/bin:/usr/bin:/bin','CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'},
            text=True,timeout=30).strip()
        require(version=='0.27.0','registered vLLM version required')
        with safe_path(GPU_LOCK).open('a+b') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            require_idle_gpu(self.runtime)
            with socket.socket() as sock:
                sock.bind(('127.0.0.1',self.runtime['argv_options']['port']))
            token = uuid.uuid4().hex
            log = self.directory/('stdout-'+token+'.log')
            env = {k:os.environ[k] for k in ('HOME','USER','LANG','LC_ALL') if k in os.environ}
            env.update(self.runtime['environment'])
            intent = {'run_digest':self.run.artifact.manifest_digest,'argv':self.command(),'log':str(log)}
            with self.pending_launch.open('xb') as stream:
                stream.write(canonical_json_bytes(intent));stream.flush();os.fsync(stream.fileno())
            _fsync_directory(self.directory)
            with log.open('xb') as output:
                self.child = subprocess.Popen(self.command(),cwd=ROOT,env=env,stdout=output,
                    stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lock.fileno(),))
            ident = self.identity(self.child.pid)
            require(ident and ident['argv']==self.command() and ident['pgid']==ident['pid']
                    and ident['cwd']==str(ROOT) and ident['uid']==os.getuid()
                    and ident['executable']==str(Path(self.command()[0]).resolve()),'launch ownership unknown; retain pending evidence')
            state = {**intent,'identity':ident}
            state['state_digest'] = digest(state)
            temp = self.directory/('state-'+token+'.tmp')
            with temp.open('xb') as stream:
                stream.write(canonical_json_bytes(state));stream.flush();os.fsync(stream.fileno())
            os.replace(temp,self.directory/'current.json')
            os.rename(self.pending_launch,self.directory/('launch-'+token+'.json'))
            _fsync_directory(self.directory)
        deadline = time.monotonic()+timeout
        while True:
            require(self.identity(ident['pid'])==ident,'owned service exited before ready; inspect '+str(log))
            try:
                return self.ready()
            except (OSError,ValueError,KeyError):
                require(time.monotonic()<deadline,'startup timeout; service retained for explicit owned stop')
                time.sleep(.5)

    def stop(self, *, execute=False, timeout=60):
        require(execute and sys.platform=='linux','explicit --execute-live on Ubuntu required')
        state = self._state()
        require(state is not None,'no owned service; no signal sent')
        ident = state['identity'];current = self.identity(ident['pid'])
        if current is None:
            return self.status()
        require(current==ident,'PID identity changed; no signal sent')
        with _pidfd(ident['pid']) as send:
            require(self.identity(ident['pid'])==ident,'PID identity changed; no signal sent')
            send(signal.SIGTERM)
        deadline = time.monotonic()+timeout
        while self.identity(ident['pid'])==ident:
            require(time.monotonic()<deadline,'owned service still exiting; no force kill')
            time.sleep(.25)
        if self.child is not None:
            self.child.wait(timeout=1)
        return self.status()
