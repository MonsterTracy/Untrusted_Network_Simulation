"""单一执行器的持久化、恢复及跨 Run 评价 CPU/Mock 回归。"""

from copy import deepcopy
import json
from http.server import BaseHTTPRequestHandler,HTTPServer
import threading
import pytest

from tests.tom_llm.conftest import Replies
from werewolf.artifact_io import canonical_json_bytes
from werewolf.tom_llm.common import digest
from werewolf.tom_llm.experiment import open_run
from werewolf.tom_llm.live_runner import Journal,HttpTransport,run_model,read_predictions,run_lock,status
from werewolf.tom_llm.provider_semantics import NotExecuted,OutcomeUnknown


@pytest.mark.parametrize('key',('qwen3','yi','mistral'))
def test_three_models_use_one_runner_and_only_contract_seal_artifacts(make_run,key):
    run = make_run(key)
    replies = Replies()
    seal = run_model(run,replies)
    _,rows = read_predictions(run,seal.manifest_digest)
    assert [task for _,task in replies.calls]==run.record['task_digests']
    assert len(rows)==run.record['request_count'] and all(r['prediction']['status']=='valid' for r in rows)
    assert {p.parent.name for p in run.root.rglob('manifest.json')}=={'contract','seal'}
    assert Journal(open_run(run.root)).summary()['completed_requests']==len(rows)
    assert status(open_run(run.root,with_tasks=False))['seal_present']
    assert not any(x in run.record for x in ('admission','permit','suite','qualification'))


@pytest.mark.parametrize('content',('bad JSON','{"suspected_players":["player2","player2"]}','{"suspected_players":["player9"]}'))
def test_invalid_outputs_count_without_retry_or_dropping(make_run,content):
    run = make_run()
    replies = Replies(content)
    run_model(run,replies)
    _,rows = read_predictions(run)
    assert len(replies.calls)==len(run.tasks)==len(rows)
    assert any(r['prediction']['status']!='valid' for r in rows)
    assert all(r['prediction']['suspected_players'] is None for r in rows)


@pytest.mark.parametrize('point',('after_received','after_completed'))
def test_crash_after_persistence_resumes_without_duplicate_send(make_run,point):
    run = make_run()
    replies = Replies()
    def crash(_):
        raise RuntimeError('simulated disconnect')
    with pytest.raises(RuntimeError):
        run_model(run,replies,**{point:crash})
    with pytest.raises(ValueError,match='explicit'):
        run_model(open_run(run.root),replies)
    run_model(open_run(run.root),replies,resume=True)
    assert len(replies.calls)==len(run.tasks)
    read_predictions(run)


@pytest.mark.parametrize('failure',('unknown','dispatched','interrupt'))
def test_unknown_or_interrupted_dispatch_never_resends(make_run,failure):
    run = make_run()
    replies = Replies(failure=OutcomeUnknown('HTTP500'))
    if failure=='dispatched':
        from werewolf.tom_llm.model_spec import wire
        Journal(run).append('dispatch',{'wire_digest':digest(wire(run.tasks[0],run.spec))})
    else:
        if failure=='interrupt':
            replies.failure = KeyboardInterrupt()
        with pytest.raises((OutcomeUnknown,KeyboardInterrupt)):
            run_model(run,replies)
    calls = len(replies.calls)
    with pytest.raises(ValueError,match='unknown'):
        run_model(open_run(run.root),replies,resume=True)
    assert len(replies.calls)==calls and not (run.root/'seal').exists()


def test_not_executed_requires_explicit_resume_not_automatic_retry(make_run):
    run = make_run()
    replies = Replies(failure=NotExecuted('connect refused'))
    with pytest.raises(NotExecuted):
        run_model(run,replies)
    assert len(replies.calls)==1 and Journal(run).current=='not_executed'
    replies.failure = None
    run_model(open_run(run.root),replies,resume=True)
    assert len(replies.calls)==len(run.tasks)+1
    assert replies.calls[0]==replies.calls[1]


def test_partial_tail_and_reordered_events_fail_closed(make_run):
    run = make_run()
    journal = Journal(run)
    journal.append('dispatch',{'wire_digest':'a'*64})
    original = journal.path.read_bytes()
    journal.path.write_bytes(original[:-1])
    with pytest.raises(ValueError,match='partial'):
        Journal(run)
    journal.path.write_bytes(original)
    event = json.loads(original)
    event['task_id'] = 'b'*64
    event['event_digest'] = digest({k:v for k,v in event.items() if k!='event_digest'})
    journal.path.write_bytes(canonical_json_bytes(event)+b'\n')
    with pytest.raises(ValueError,match='coverage'):
        Journal(run)


def test_completed_payload_must_match_raw_response_even_with_rehashed_chain(make_run):
    run = make_run()
    def crash(_):
        raise RuntimeError()
    with pytest.raises(RuntimeError):
        run_model(run,Replies(),after_completed=crash)
    path = run.root/'requests.jsonl'
    events = [json.loads(s) for s in path.read_bytes().splitlines()]
    events[-1]['payload']['usage']['output_tokens'] += 1
    events[-1]['event_digest'] = digest({k:v for k,v in events[-1].items() if k!='event_digest'})
    path.write_bytes(b''.join(canonical_json_bytes(e)+b'\n' for e in events))
    with pytest.raises(ValueError,match='durable'):
        Journal(run)


def test_exclusive_run_lock(make_run):
    run = make_run()
    with run_lock(run.root):
        with pytest.raises(BlockingIOError):
            with run_lock(run.root):
                pass


@pytest.mark.parametrize('guarded',(False,True))
def test_loopback_real_http_wire_uses_same_runner_no_retry(make_run,guarded):
    run = make_run()
    seen = []
    replies = Replies()
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            task_id = self.headers['X-Request-Id']
            seen.append((self.path,request,task_id))
            raw = replies.send(request,task_id).encode()
            self.send_response(200)
            self.send_header('Content-Length',str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        def log_message(self,*args):
            pass
    server = HTTPServer(('127.0.0.1',0),Handler)
    thread = threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    checks = []
    def guard(connection=None,expected=None):
        checks.append('pre' if expected is None else 'connected' if connection is not None else 'response')
        if connection is not None:
            assert connection.getpeername()==('127.0.0.1',server.server_port)
        return {'owned':'same-instance'}
    try:
        run_model(run,HttpTransport('http://127.0.0.1:'+str(server.server_port),
                                   alive_check=guard if guarded else None))
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert len(seen)==len(run.tasks)
    assert checks==(['pre','connected','response']*len(run.tasks) if guarded else [])
    for path,request,task_id in seen:
        assert path=='/v1/chat/completions' and task_id in run.record['task_digests']
        assert 'extra_body' not in request and request['top_k']==-1
        assert 'uniqueItems' not in request['response_format']['json_schema']['schema']['properties']['suspected_players']
    read_predictions(run)


def test_no_per_request_model_source_cohort_or_artifact_verification(make_run,monkeypatch):
    import werewolf.tom_llm.live_runner as lr
    import werewolf.tom_llm.physical as physical
    import werewolf.tom_llm.multimodel as mm
    run = make_run()
    monkeypatch.setattr(physical,'verify_model',lambda *a:pytest.fail('per-row assets'))
    monkeypatch.setattr(mm,'read_cohort',lambda *a:pytest.fail('per-row Cohort'))
    monkeypatch.setattr(lr,'source_identity',lambda:pytest.fail('Mock cannot claim Source qualification'))
    publish = lr.publish_artifact
    calls = []
    def once(*a,**kw):
        calls.append(kw['manifest_fields']['artifact_type'])
        return publish(*a,**kw)
    monkeypatch.setattr(lr,'publish_artifact',once)
    run_model(run,Replies())
    assert calls==['tom_baseline_predictions']


@pytest.mark.parametrize('code',(429,500))
def test_http_error_preserves_status_and_never_retries(make_run,code):
    run = make_run()
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(self.rfile.read(int(self.headers['Content-Length'])))
            self.send_response(code)
            self.send_header('Content-Length','0')
            self.end_headers()
        def log_message(self,*args):
            pass
    server = HTTPServer(('127.0.0.1',0),Handler)
    thread = threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    transport = HttpTransport('http://127.0.0.1:'+str(server.server_port))
    try:
        with pytest.raises(OutcomeUnknown):
            run_model(run,transport)
        journal = Journal(run)
        assert journal.current=='unknown' and len(requests)==1
        assert json.loads(journal.path.read_bytes().splitlines()[-1])['payload']['error']=='HTTP status '+str(code)
        with pytest.raises(ValueError,match='unknown'):
            run_model(run,transport,resume=True)
        assert len(requests)==1
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize('failure',('before_connect','after_connect','after_response','after_response_not_executed'))
def test_ownership_change_stops_without_accepting_or_retrying(make_run,monkeypatch,failure):
    import werewolf.tom_llm.live_runner as lr
    run = make_run()
    sent = []
    checks = []
    sockets = []
    class Socket:
        def __init__(self):self.closed=False;sockets.append(self)
        def getsockname(self):return ('127.0.0.1',40000)
        def getpeername(self):return ('127.0.0.1',18581)
        def dup(self):return Socket()
        def close(self):self.closed=True
    class Connection:
        def __init__(self,*args,**kw):self.sock=Socket()
        def connect(self):checks.append('connect')
        def request(self,method,path,*,body,headers):
            assert self.auto_open==0
            sent.append((json.loads(body),headers['X-Request-Id']))
        def getresponse(self):
            body = Replies().send(*sent[-1]).encode()
            return type('Response',(),{'status':200,'read':lambda self,_:body})()
        def close(self):self.sock.close()
    def ownership(connection=None,expected=None):
        phase = 'before_connect' if expected is None else 'after_connect' if connection is not None else 'after_response'
        checks.append(phase)
        if phase==failure or phase=='after_response' and failure=='after_response_not_executed':
            exception = NotExecuted if failure=='after_response_not_executed' else RuntimeError
            raise exception('listener ownership changed')
        return {'owned':'same-instance'}
    monkeypatch.setattr(lr.http.client,'HTTPConnection',Connection)
    transport = HttpTransport('http://127.0.0.1:18581',alive_check=ownership)
    sent_response = failure.startswith('after_response')
    error = OutcomeUnknown if sent_response else NotExecuted
    with pytest.raises(error):
        run_model(run,transport)
    journal = Journal(run)
    assert len(sent)==(1 if sent_response else 0)
    assert not journal.rows and not (run.root/'seal').exists()
    assert journal.current==('unknown' if sent_response else 'not_executed')
    assert 'ownership' in json.loads(journal.path.read_bytes().splitlines()[-1])['payload']['error']
    assert all(sock.closed for sock in sockets)
    if sent_response:
        with pytest.raises(ValueError,match='unknown'):
            run_model(open_run(run.root),transport,resume=True)
        assert len(sent)==1
    if failure=='before_connect':
        assert 'connect' not in checks


def test_guarded_http_keeps_original_mock_results(make_run,monkeypatch):
    import werewolf.tom_llm.live_runner as lr
    run = make_run()
    replies = Replies()
    stages = []
    class Socket:
        def dup(self):return Socket()
        def getsockname(self):return ('127.0.0.1',40000)
        def getpeername(self):return ('127.0.0.1',18581)
        def close(self):pass
    class Connection:
        def __init__(self,*a,**kw):self.sock=Socket()
        def connect(self):pass
        def request(self,method,path,*,body,headers):
            self.body = replies.send(json.loads(body),headers['X-Request-Id']).encode()
        def getresponse(self):
            return type('Response',(),{'status':200,'read':lambda _,limit:self.body})()
        def close(self):pass
    def guard(connection=None,expected=None):
        stages.append('pre' if expected is None else 'connected' if connection is not None else 'response')
        return {'owned':'same-instance'}
    monkeypatch.setattr(lr.http.client,'HTTPConnection',Connection)
    run_model(run,HttpTransport('http://127.0.0.1:18581',alive_check=guard))
    _,rows = read_predictions(run)
    assert len(replies.calls)==len(run.tasks) and len(rows)==len(run.tasks)
    assert stages==['pre','connected','response']*len(run.tasks)


def test_actual_token_mismatch_stops_with_raw_receipt_and_no_retry(make_run):
    run = make_run(count=101)
    replies = Replies()
    with pytest.raises(ValueError,match='native budget'):
        run_model(run,replies)
    assert len(replies.calls)==1 and Journal(run).current=='received'
    with pytest.raises(ValueError,match='native budget'):
        run_model(open_run(run.root),replies,resume=True)
    assert len(replies.calls)==1 and not (run.root/'seal').exists()


def test_log_dispatch_is_durable_before_send(make_run,monkeypatch):
    import werewolf.tom_llm.live_runner as lr
    run = make_run()
    calls = []
    original = lr._fsync_directory
    def sync(directory):
        calls.append('directory_fsync')
        return original(directory)
    monkeypatch.setattr(lr,'_fsync_directory',sync)
    class StopAfterDispatch:
        def send(self,request,task_id):
            assert calls==['directory_fsync']
            assert Journal(run).current=='dispatched'
            raise OutcomeUnknown('stop')
    with pytest.raises(OutcomeUnknown):
        run_model(run,StopAfterDispatch())


def test_cross_run_report_same_populations_bootstrap_and_immutable_outputs(make_run,tmp_path,monkeypatch):
    from werewolf.tom_llm.baseline_analysis import analyze_cross_runs
    import werewolf.tom_llm.baseline_analysis as analysis
    import werewolf.tom_llm.experiment as experiment
    roots = {}
    for key in ('qwen3','yi','mistral'):
        run = make_run(key)
        seal = run_model(run,Replies())
        roots[key] = {'path':str(run.root),'digest':seal.manifest_digest}
    read = experiment.read_cohort
    opened = []
    def once(*args):
        opened.append(args)
        return read(*args)
    monkeypatch.setattr(experiment,'read_cohort',once)
    report = analyze_cross_runs(roots,tmp_path/'report')
    assert len(opened)==1
    result = json.loads((report.path/'analysis.json').read_bytes())
    assert result['formal_result'] is False
    populations = result['sets']['populations']
    assert len(populations['full_primary_raw']['paired'])==10
    assert len(populations['primary_comparable_nu']['paired'])==15
    assert populations['full_primary_raw']['bootstrap_replicates']==10000
    for pop in populations.values():
        assert len({c['row_count'] for c in pop['cells'].values()})<=1
    assert set(result['sets']['derived_methods'])=={'Always Empty','Always All-Six','Observer-ToM'}
    assert all(v['model_run_seal'] is None for v in result['sets']['derived_methods'].values())
    assert set(report.manifest['file_table'])=={'analysis.json','metrics.csv','paired.csv','supplementary.csv','performance.csv','report.md'}
    with pytest.raises(ValueError,match='exists'):
        analyze_cross_runs(roots,tmp_path/'report')
    original = analysis.open_run
    for field in ('source','mode','cohort_digest','task_digests','selection','reference'):
        def changed(root,**kw):
            run = original(root,**kw)
            if run.key=='yi':
                run.record[field] = 'changed'
            return run
        monkeypatch.setattr(analysis,'open_run',changed)
        with pytest.raises(ValueError,match='cross-run '+field):
            analyze_cross_runs(roots,tmp_path/('mismatch-'+field))
    monkeypatch.setattr(analysis,'open_run',original)
    roots['yi']['digest'] = roots['qwen3']['digest']
    with pytest.raises(ValueError,match='identity'):
        analyze_cross_runs(roots,tmp_path/'wrong')
