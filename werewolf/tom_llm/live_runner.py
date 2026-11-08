"""单一请求执行器：顺序发送、追加日志、未知结果停止及结束封存。"""

from collections import Counter
from contextlib import contextmanager
import fcntl
import http.client
import math
import os
from pathlib import Path
import time
from urllib.parse import urlparse

from werewolf.artifact_io import canonical_json_bytes,canonical_jsonl_bytes,publish_artifact,verify_artifact,sha256_bytes
from werewolf.artifact_io.canonical import _fsync_directory
from werewolf.tom_llm.common import digest,require,strict_json
from werewolf.tom_llm.experiment import VERSION,open_run,safe_path
from werewolf.tom_llm.multimodel import source_identity
from werewolf.tom_llm.model_spec import wire
from werewolf.tom_llm.output import parse_task_output,validate_set_prediction
from werewolf.tom_llm.provider_semantics import NotExecuted,OutcomeUnknown


class HttpTransport:
    """正式与 Mock 共用传输；不使用 SDK、代理、重定向或自动重试。"""
    def __init__(self, endpoint, *, timeout=120, alive_check=None):
        u = urlparse(endpoint)
        require(u.scheme=='http' and u.hostname=='127.0.0.1' and u.port and u.path in ('','/')
                and not u.query and not u.fragment and not u.username,'loopback endpoint required')
        self.port = u.port
        self.timeout = timeout
        self.alive_check = alive_check

    def send(self, request, task_id):
        binding = None
        if self.alive_check:
            try:
                binding = self.alive_check()
                require(binding is not None,'listener ownership proof missing')
            except Exception as error:
                raise NotExecuted('owned listener unavailable before send: '+str(error)) from error
        c = http.client.HTTPConnection('127.0.0.1',self.port,timeout=self.timeout)
        pinned = None
        try:
            try:
                c.connect()
            except OSError as error:
                raise NotExecuted('connect failed before transmission') from error
            c.auto_open = 0  # 禁止 http.client 在验证后换连接。
            if self.alive_check:
                try:
                    pinned = c.sock.dup()
                    endpoints = (pinned.getsockname(),pinned.getpeername())
                    self.alive_check(pinned,binding)
                except Exception as error:
                    raise NotExecuted('connected listener unverified before send: '+str(error)) from error
            body = {k:v for k,v in request.items() if k!='extra_body'}
            require(not set(body)&set(request['extra_body']),'wire extra-body collision')
            body.update(request['extra_body'])
            try:
                c.request('POST','/v1/chat/completions',body=canonical_json_bytes(body),
                          headers={'Content-Type':'application/json','X-Request-Id':task_id})
                reply = c.getresponse();raw = reply.read(8*1024**2+1)
                if self.alive_check:
                    self.alive_check(None,binding)
                    require((pinned.getsockname(),pinned.getpeername())==endpoints,
                            'response connection changed')
                require(len(raw)<=8*1024**2,'response too large')
                if reply.status!=200:
                    raise OutcomeUnknown('HTTP status '+str(reply.status))
                return raw.decode('utf-8')
            except OutcomeUnknown:
                raise
            except Exception as error:
                raise OutcomeUnknown('request may have reached model; no automatic resend: '+str(error)) from error
        finally:
            c.close()
            if pinned is not None:
                pinned.close()


def prediction_row(raw, task, spec, elapsed):
    """保留模型非法输出；基础设施错误不补造预测。"""
    body = strict_json(raw)
    require(body['model']==spec.record()['model'] and len(body['choices'])==1,'reply model/choices mismatch')
    choice = body['choices'][0];message = choice['message']
    content = None if message.get('refusal') is not None else message.get('content')
    prediction = parse_task_output(content,task,finish_reason=choice['finish_reason']).to_record()
    usage = {a:body['usage'][b] for a,b in
             [('input_tokens','prompt_tokens'),('output_tokens','completion_tokens'),('total_tokens','total_tokens')]}
    require(all(type(v) is int and v>=0 for v in usage.values()) and usage['input_tokens']>0
            and usage['input_tokens']<=7680 and usage['output_tokens']<=512
            and usage['total_tokens']==usage['input_tokens']+usage['output_tokens'],'token usage invalid')
    fields = ('game_id','boundary_id','prefix_digest','plan_digest','observer','fold')
    return {**{k:task[k] for k in fields},'canonical_shift':0,'temporal_condition':'explicit_day_phase',
            'task_digest':digest(task),'prediction':prediction,
            'usage':usage,'elapsed_seconds':elapsed,'reply_id':body.get('id')}


class Journal:
    """每进程只读一次历史，之后在内存维护状态；每条事件追加并 fsync。"""
    def __init__(self, run):
        self.run = run
        self.path = safe_path(run.root/'requests.jsonl')
        self.ids = run.record['task_digests']
        self.rows = []
        self.current = 'pending'
        self.received = None
        self.attempt = 0
        self.attempts = 0
        self.outcomes = Counter()
        self.sequence = 0
        self.previous = '0'*64
        raw = self.path.read_bytes() if self.path.exists() else b''
        require(not raw or raw.endswith(b'\n'),'partial journal tail; stop without automatic repair/resend')
        for line in raw.splitlines():
            event = strict_json(line)
            self._apply(event,replay=True)

    def _apply(self, event, *, replay=False):
        fields = {'sequence','previous','run_digest','task_id','kind','attempt','payload','event_digest'}
        require(set(event)==fields and event['event_digest']==digest({k:v for k,v in event.items() if k!='event_digest'})
                and event['sequence']==self.sequence and event['previous']==self.previous
                and event['run_digest']==self.run.artifact.manifest_digest,'journal ownership/order/hash drift')
        require(len(self.rows)<len(self.ids) and event['task_id']==self.ids[len(self.rows)],
                'journal task coverage/order drift')
        kind = event['kind'];payload = event['payload']
        if kind=='dispatch':
            require(self.current in ('pending','not_executed') and event['attempt']==self.attempt+1,
                    'unknown/received/completed task cannot be dispatched again')
            require(set(payload)=={'wire_digest'},'dispatch wire identity missing')
            if replay and self.run.tasks is not None:
                require(payload['wire_digest']==digest(wire(self.run.tasks[len(self.rows)],self.run.spec)),
                        'dispatched wire differs from task/model')
            self.current = 'dispatched';self.attempt = event['attempt'];self.attempts += 1
        else:
            require(event['attempt']==self.attempt,'attempt identity drift')
            if kind in ('not_executed','unknown'):
                require(self.current=='dispatched' and type(payload.get('error')) is str,'failure state drift')
                self.current = kind;self.outcomes[kind] += 1
            elif kind=='received':
                require(self.current=='dispatched' and type(payload.get('body')) is str
                        and type(payload.get('elapsed_seconds')) in (int,float)
                        and math.isfinite(payload['elapsed_seconds']) and payload['elapsed_seconds']>=0,'response state drift')
                self.current = 'received';self.received = payload;self.outcomes['received'] += 1
            elif kind=='completed':
                require(self.current=='received' and payload['task_digest']==event['task_id'],'completion without receipt')
                validate_set_prediction(payload['prediction'])
                require(payload['usage']['input_tokens']==self.run.record['budget']['input_token_counts'][len(self.rows)],
                        'actual input tokens differ from native budget')
                if replay and self.run.tasks is not None:
                    task = self.run.tasks[len(self.rows)]
                    require(payload==prediction_row(self.received['body'],task,self.run.spec,self.received['elapsed_seconds']),
                            'completion differs from durable response')
                self.rows.append(payload);self.current = 'pending';self.attempt = 0;self.received = None
            else:
                raise ValueError('unknown journal event')
        self.previous = event['event_digest'];self.sequence += 1

    def append(self, kind, payload):
        event = {'sequence':self.sequence,'previous':self.previous,'run_digest':self.run.artifact.manifest_digest,
                 'task_id':self.ids[len(self.rows)],'kind':kind,
                 'attempt':self.attempt+1 if kind=='dispatch' else self.attempt,'payload':payload}
        event['event_digest'] = digest(event)
        # 先验证状态转换，再追加；写失败时整个进程停止，不能继续依赖内存状态。
        self._apply(event)
        raw = canonical_json_bytes(event)+b'\n'
        fd = os.open(self.path,os.O_CREAT|os.O_APPEND|os.O_WRONLY|os.O_NOFOLLOW,0o600)
        try:
            while raw:
                n = os.write(fd,raw)
                require(n>0,'journal write failed')
                raw = raw[n:]
            os.fsync(fd)
        finally:
            os.close(fd)
        if self.sequence==1:
            _fsync_directory(self.path.parent)

    def summary(self):
        invalid = Counter(r['prediction']['failure_type'] for r in self.rows if r['prediction']['status']!='valid')
        done = len(self.rows)
        return {'completed_requests':done,'request_count':len(self.ids),'valid_predictions':done-sum(invalid.values()),
                'invalid_predictions':sum(invalid.values()),
                'valid_rate':None if not done else (done-sum(invalid.values()))/done,
                'failure_reasons':dict(invalid),'attempts':self.attempts,'transport_outcomes':dict(self.outcomes),
                'state':self.current,'blocked':self.current in ('dispatched','unknown'),
                'elapsed_seconds':sum(r['elapsed_seconds'] for r in self.rows),'output_directory':str(self.run.root)}


@contextmanager
def run_lock(root):
    path = safe_path(root)
    require((path/'contract').is_dir(),'missing Run contract')
    with safe_path(path/'execution.lock').open('a+b') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream,fcntl.LOCK_UN)


def recover_received(run, journal):
    """离线恢复已经持久化的响应，不启动模型、不发送请求。"""
    require(not journal.summary()['blocked'],'unknown result; no automatic resend')
    if journal.current=='received':
        task = run.tasks[len(journal.rows)]
        received = journal.received
        row = prediction_row(received['body'],task,run.spec,received['elapsed_seconds'])
        journal.append('completed',row)


def run_model(run, transport, *, resume=False, journal=None, after_received=None, after_completed=None):
    """三模型只走本函数；恢复已收到的响应时只重解析，不重新发送。"""
    require(run.tasks is not None,'loaded public tasks required')
    journal = journal if journal is not None else Journal(run)
    require(resume or journal.sequence==0,'existing requests require explicit --resume')
    require(not journal.summary()['blocked'],'unknown result; no automatic resend')
    for task in run.tasks[len(journal.rows):]:
        task_id = run.record['task_digests'][len(journal.rows)]
        if journal.current=='received':
            received = journal.received
        else:
            request = wire(task,run.spec)
            journal.append('dispatch',{'wire_digest':digest(request)})
            start = time.perf_counter()
            try:
                raw = transport.send(request,task_id)
            except NotExecuted as error:
                journal.append('not_executed',{'error':str(error)})
                raise
            except Exception as error:
                journal.append('unknown',{'error':str(error)})
                raise OutcomeUnknown('execution stopped; unknown result; no resend') from error
            received = {'body':raw,'elapsed_seconds':time.perf_counter()-start}
            journal.append('received',received)
            if after_received:
                after_received(task)
        row = prediction_row(received['body'],task,run.spec,received['elapsed_seconds'])
        journal.append('completed',row)
        if after_completed:
            after_completed(task)
    if run.record['mode']=='formal':
        require(source_identity()==run.record['source'],'Source changed during Run; cannot seal')
    return seal_run(run,journal)


def seal_run(run, journal):
    require(len(journal.rows)==run.record['request_count'] and journal.current=='pending','incomplete Run')
    persisted = Journal(run)
    require(persisted.previous==journal.previous and persisted.rows==journal.rows,'journal changed during execution')
    rows = journal.rows
    require([r['task_digest'] for r in rows]==run.record['task_digests'],'prediction coverage drift')
    return publish_artifact(run.root/'seal',manifest_fields={'artifact_type':'tom_baseline_predictions',
        'schema_version':VERSION,'run_digest':run.artifact.manifest_digest,'model':run.key,
        'journal_sha256':sha256_bytes(journal.path.read_bytes()),'rows_digest':digest(rows),
        'summary':journal.summary(),'mode':run.record['mode']},
        files={'predictions.jsonl':canonical_jsonl_bytes(rows)})


def read_predictions(run, expected=None):
    seal = verify_artifact(run.root/'seal',expected_artifact_type='tom_baseline_predictions',expected_schema_version=VERSION)
    m = seal.manifest
    require((expected is None or seal.manifest_digest==expected)
            and m['run_digest']==run.artifact.manifest_digest and m['model']==run.key
            and m['mode']==run.record['mode'] and set(m['file_table'])=={'predictions.jsonl'},
            'prediction seal identity drift')
    require(m['journal_sha256']==sha256_bytes((run.root/'requests.jsonl').read_bytes()),'sealed journal changed')
    journal = Journal(run)
    rows = [strict_json(line) for line in (seal.path/'predictions.jsonl').read_bytes().splitlines()]
    require(digest(rows)==m['rows_digest'] and [r['task_digest'] for r in rows]==run.record['task_digests'],
            'sealed prediction coverage drift')
    require(rows==journal.rows and m['summary']==journal.summary(),'Seal/journal results differ')
    for row,task in zip(rows,run.tasks,strict=True):
        require(all(row[k]==task[k] for k in ('game_id','boundary_id','prefix_digest','plan_digest','observer','fold')),
                'prediction scientific row mismatch')
        validate_set_prediction(row['prediction'])
    return seal,rows


def status(run):
    summary = Journal(run).summary()
    return {'run_id':run.record['run_id'],'model':run.key,'mode':run.record['mode'],
            'seal_present':(run.root/'seal/manifest.json').is_file(),**summary}
