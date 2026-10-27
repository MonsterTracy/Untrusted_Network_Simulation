"""Persistent ownership for the existing qualification and live vLLM entry points.

Only launch state is mutable; Deployment, Permit and engine evidence are untouched.
No model requests, reconnect, force kill, tmux or third-party process adoption.
"""

from contextlib import contextmanager
import fcntl
import datetime
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
from werewolf.tom_llm.common import digest, require, strict_json
from werewolf.tom_llm.experiment import safe_path
from werewolf.tom_llm.model_spec import ROOT


class Service:
    def __init__(self, root, key, expected, *, qualification=False):
        self.root=safe_path(root);self.key=key;self.expected=expected
        self.qualification=qualification;self.child=None

    def binding(self, *, physical=False):
        if self.qualification:
            from werewolf.tom_llm.deployment import open_deployment
            a,r=open_deployment(self.root,self.expected,physical_recheck=physical,current_source=physical)
            require(r['key']==self.key,'service model mismatch')
            cmd=[r['profile']['environment_prefix']+'/bin/python','-m','werewolf.tom_llm.gpu_probe',
                '--deployment',str(a.path),'--expected-deployment',a.manifest_digest,'--execute-synthetic-gpu-qualification']
        else:
            from werewolf.tom_llm.live_admission import open_permit
            a,r=open_permit(self.root,self.key,self.expected,physical=physical)
            cmd=[r['profile']['environment_prefix']+'/bin/python','-m','werewolf.tom_llm.live_service',
                '--run',str(self.root),'--model',self.key,'--expected-permit',a.manifest_digest,'--execute-live']
        return a,r,cmd

    @property
    def directory(self):
        return safe_path(self.binding()[1]['profile']['root'])/'service-control'

    @property
    def pending_launch(self):
        return safe_path(self.directory/'launch.pending.jsonl')

    def identity(self, pid):
        """PID reuse is distinguished from ownership; no signals here."""
        proc=Path('/proc')/str(pid)
        try:
            stat=(proc/'stat').read_text().rsplit(')',1)[1].split()
            if stat[0]=='Z':return None
            return {'pid':pid,'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                'start_ticks':stat[19],'uid':proc.stat().st_uid,'cwd':str((proc/'cwd').resolve()),
                'executable':str((proc/'exe').resolve()),'pgid':os.getpgid(pid),
                'argv':[v.decode() for v in (proc/'cmdline').read_bytes().split(b'\0') if v]}
        except FileNotFoundError:return None

    def _state(self):
        require(not self.pending_launch.exists(),
                'incomplete launch ownership; no automatic restart or stop; inspect '+str(self.pending_launch))
        path=safe_path(self.directory/'current.json')
        if not path.exists():return None
        r=strict_json(path.read_bytes())
        require(r.pop('state_digest')==digest(r),'service state digest drift')
        _,_,cmd=self.binding()
        require(r['anchor']==self.expected and r['model']==self.key and r['argv']==cmd
            and r['qualification']==self.qualification,'service state ownership drift')
        identity=r['identity']
        require(identity['argv']==cmd and identity['uid']==os.getuid() and identity['cwd']==str(ROOT)
            and identity['executable']==str(Path(cmd[0]).resolve()) and identity['pgid']==identity['pid'],
            'stored service process identity drift')
        require(not self.pending_launch.exists(),'incomplete launch ownership; no automatic restart or stop; inspect '+str(self.pending_launch))
        return r

    def status(self):
        _,record,_=self.binding()
        result={'model':self.key,'purpose':'qualification' if self.qualification else 'run',
            'anchor':self.expected,'port':record['profile']['argv_options']['port'],'gpu':record['profile']['gpu']}
        if self.pending_launch.exists():
            return {**result,'state':'launch_incomplete','ready':False,'pending_launch':str(self.pending_launch)}
        r=self._state()
        if r is None:return {**result,'state':'not_started','ready':False}
        identity=self.identity(r['identity']['pid'])
        return {**result,'state':'exited' if identity is None else 'running' if identity==r['identity'] else 'pid_reused',
            'pid':r['identity']['pid'],'log':r['log'],'ready':False}

    def capture(self):
        r=self._state();require(r is not None,'tool has not started this service')
        require(self.identity(r['identity']['pid'])==r['identity'],'owned service exited or PID identity changed')
        a,record,_=self.binding()
        if self.qualification:
            from werewolf.tom_llm.deployment import capture_instance
            instance=capture_instance(a,record,r['identity']['pid'])
            life={k:instance[k] for k in ('pid','boot_id','start_ticks')}
            log=Path(record['profile']['engine_log_root'])/digest(life)/'engine-events.jsonl'
        else:
            from werewolf.tom_llm.live_admission import capture_live_instance
            instance=capture_live_instance(a,record,r['identity']['pid']);log=Path(instance['engine_log'])
        with log.open('rb') as stream:startup=strict_json(stream.readline())
        require(startup['kind']=='startup','engine startup evidence missing')
        require(startup.get('deployment_digest' if self.qualification else 'permit_digest')==
            (digest(record) if self.qualification else a.manifest_digest),'engine startup anchor drift')
        return instance

    def start(self, *, execute=False, timeout=900):
        require(execute and sys.platform=='linux','explicit --execute-live on Ubuntu required')
        require(hasattr(os,'pidfd_open') and hasattr(signal,'pidfd_send_signal'),
                'Linux pidfd support required before service launch')
        from werewolf.tom_llm.deployment import BASE,CACHE_VARIABLES,require_idle_gpu,validate_cache_paths
        require(not self.pending_launch.exists(), 'incomplete launch ownership; no restart; inspect '+str(self.pending_launch))
        a,record,cmd=self.binding(physical=True);p=record['profile']
        require(safe_path(p['root']).is_relative_to(BASE),'service outside independent baseline root')
        require(self.status()['state']!='running','owned service already running')
        ensure_durable_directory(self.directory)
        with safe_path(BASE/'baseline-gpu.lock').open('a+b') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            require_idle_gpu(p)
            with socket.socket() as sock:sock.bind(('127.0.0.1',p['argv_options']['port']))
            for var in CACHE_VARIABLES:safe_path(p['environment_requirements'][var]).mkdir(parents=True,exist_ok=True)
            validate_cache_paths(p,require_existing=True)
            env={k:os.environ[k] for k in ('HOME','USER','LANG','LC_ALL') if k in os.environ}
            env.update(p['environment_requirements'])
            token=uuid.uuid4().hex
            log=self.directory/('stdout-'+token+'.log')
            intent={'kind':'launch_intent','launch_token':token,'model':self.key,'qualification':self.qualification,
                'anchor':a.manifest_digest,'argv':cmd,'log':str(log),
                'launched_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
            with log.open('xb') as output:
                # Durable intent precedes Popen. An interruption at ANY later
                # point leaves a blocker, even if no PID could be recorded.
                fd=os.open(self.pending_launch,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600)
                with os.fdopen(fd,'wb') as journal:
                    def append(event):
                        journal.write(canonical_json_bytes(event)+b'\n');journal.flush();os.fsync(journal.fileno())
                    append(intent);_fsync_directory(self.directory)
                    try:
                        # Only the GPU lock is inherited, not the intent FD.
                        self.child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT,
                            start_new_session=True,pass_fds=(lock.fileno(),))
                        append({'kind':'popen_returned','pid_hint':self.child.pid})  # Diagnostic only; NOT ownership.
                        require(self.child.poll() is None,'owned service exited')
                        identity=self.identity(self.child.pid)
                        require(identity and identity['argv']==cmd and identity['pgid']==self.child.pid
                            and identity['uid']==os.getuid() and identity['cwd']==str(ROOT)
                            and identity['executable']==str(Path(cmd[0]).resolve()),'new owned process identity unavailable')
                        state={**{k:intent[k] for k in ('launch_token','model','qualification','anchor','argv','log','launched_utc')},
                            'identity':identity}
                        state['state_digest']=digest(state)
                        temp=self.directory/('state-'+token+'.tmp')
                        fd=os.open(temp,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
                        with os.fdopen(fd,'wb') as stream:
                            stream.write(canonical_json_bytes(state));stream.flush();os.fsync(stream.fileno())
                        os.replace(temp,self.directory/'current.json');_fsync_directory(self.directory)
                        append({'kind':'ownership_committed','state_digest':state['state_digest']})
                    except Exception as error:
                        raise ValueError('launch incomplete; no retry or automatic signal; inspect '+str(self.pending_launch)
                            +' and '+str(log)+'; '+type(error).__name__+': '+str(error)) from error
                # Archive only AFTER the complete identity is durably committed.
                os.rename(self.pending_launch,self.directory/('launch-'+token+'.jsonl'))
                _fsync_directory(self.directory)
        deadline=time.monotonic()+timeout
        while True:
            require(self.child.poll() is None,'owned service exited; inspect '+str(log))
            try:
                instance=self.capture()
                return {**self.status(),'ready':True,'instance':instance}
            except (OSError,ValueError,KeyError):
                require(time.monotonic()<deadline,'startup timeout; retain service and log for explicit stop: '+str(log))
                time.sleep(.25)

    def stop(self, *, execute=False, timeout=60):
        require(execute and sys.platform=='linux','explicit --execute-live on Ubuntu required')
        r=self._state();require(r is not None,'no owned launch state; refusing process adoption')
        pid=r['identity']['pid'];current=self.identity(pid)
        if current is None:return self.status()
        require(current==r['identity'] and current['pgid']==pid,'PID/group identity changed; no signal sent')
        # pidfd prevents signalling an unrelated process that reused this PID.
        fd=os.pidfd_open(pid)
        try:
            require(self.identity(pid)==r['identity'],'PID identity changed; no signal sent')
            signal.pidfd_send_signal(fd,signal.SIGTERM)
        finally:os.close(fd)
        deadline=time.monotonic()+timeout
        while self.identity(pid)==r['identity']:
            require(time.monotonic()<deadline,'owned service did not exit; no force kill; inspect '+r['log'])
            time.sleep(.25)
        if self.child is not None:self.child.wait(timeout=1)
        return self.status()

    def logs(self, lines=30):
        require(type(lines) is int and 0<lines<=1000,'log tail must be 1..1000 lines')
        from collections import deque
        if self.pending_launch.exists():
            with self.pending_launch.open('rb') as stream:r=strict_json(stream.readline())
        else:r=self._state()
        require(r is not None,'no owned service log')
        path=safe_path(r['log']);require(path.parent==self.directory and path.name.startswith('stdout-'),'log escaped service directory')
        with path.open(errors='replace') as stream:return ''.join(deque(stream,maxlen=lines))


@contextmanager
def owned_service(root,key,expected):
    service=Service(root,key,expected)
    state=service.start(execute=True)
    try:yield state['pid']
    finally:service.stop(execute=True)
