"""Local B2-A contracts. All clients/services are synthetic; no network or GPU."""

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil

from jinja2 import Environment, StrictUndefined
import pytest

from tests.canonical_collection.test_structured_token_plan import _prefix
from tests.tom_llm.test_selection_inference import uniform, folds
from tests.tom_llm.test_transport import client, reply
from werewolf.artifact_io import publish_artifact, canonical_json_bytes, sha256_bytes
from werewolf.tom_llm.common import digest, identity
from werewolf.tom_llm.study import make_request, wire_request, prepare_mock_study, open_study
from werewolf.tom_llm.server_fixed import (WIRE_CONFIG, WIRE_VERSION, SERVICE_VERSION,
    ServerFixedService, wire_binding, wire_request_v2, InjectedServerFixedTransport)
from werewolf.tom_llm.execution import ServerFixedExecution, prepare_mock_study_v2, open_study_v2
from werewolf.tom_llm.inference import (MockBackend, Reply, TransportFailure, InfrastructureFailure,
    run_mock_study, seal_mock_study, open_mock_predictions)
from werewolf.tom_llm.prompt import asset_bytes
from werewolf.tom_llm.selection import select_games, publish_selection
from werewolf.tom_llm.selection_binding import open_selection_binding, validate_selection_source


def service_artifact(root, *, startup=None, template=None, changes=None):
    service = {"service_instance_id": "synthetic:isolated-instance-1", "endpoint": "http://127.0.0.1:8000/v1",
        "runtime": {"vllm": "0.27.0", "python": "synthetic:3.12", "torch": "synthetic:2.9"},
        "weights_manifest_sha256": "a" * 64, "template_origin": "server_fixed",
        "template_source": WIRE_CONFIG["template_source"], "wire_config_digest": digest(WIRE_CONFIG),
        "tokenizer_asset_set_digest": WIRE_CONFIG["tokenizer_asset_set_digest"]}
    service.update(changes or {})
    artifact = publish_artifact(root, manifest_fields={"artifact_type": "llm_mock_serving_identity",
        "schema_version": SERVICE_VERSION, "engineering_only": True}, files={
        "service.json": canonical_json_bytes(service),
        "startup.json": canonical_json_bytes(WIRE_CONFIG["server"] if startup is None else startup),
        "chat_template.jinja": asset_bytes("chat_template.jinja") if template is None else template})
    return ServerFixedService(artifact.path, artifact.manifest_digest)


def prepared_v2(tmp_path, *, requests=None, service=None):
    service = service or service_artifact(tmp_path / "service")
    execution = ServerFixedExecution(service)
    requests = requests or [make_request(_prefix(), 0, fold=0)]
    root = tmp_path / "study-v2"
    prepare_mock_study_v2(root, requests, execution=execution,
                         tokenizer_identity="synthetic:counter-v1", count_tokens=lambda _:100)
    return root, execution, requests


def all_alive_prefix():
    from werewolf.canonical_collection import construct_authoritative_pre_prefix, freeze_public_event_history
    alive = tuple(f"player{i+1}" for i in range(7))
    events = [{"event_id":"e0", "event_index":0, "event_type":"phase_change", "day":0, "phase":"night"},
              {"event_id":"e1", "event_index":1, "event_type":"death_announcement", "dead_players":[]},
              {"event_id":"e2", "event_index":2, "event_type":"phase_change", "day":1, "phase":"discussion"},
              {"event_id":"e3", "event_index":3, "event_type":"turn_start", "speaker":"player1"}]
    return construct_authoritative_pre_prefix(game_id="synthetic-all-alive", boundary_id="b3", step_index=3,
        report_trigger_id="pre-synthetic-b3", current_speaker="player1", alive_observer_ids=alive,
        public_event_history=freeze_public_event_history(events), v1_annotations=(),
        belief_observation_ids_by_observer={seat:f"belief-{seat}" for seat in alive})


PARITY_CASES = [(all_alive_prefix, i) for i in range(7)] + [(_prefix, i) for i in (0, 1, 3, 4)]


@pytest.mark.parametrize("prefix,observer", PARITY_CASES)
def test_only_request_template_is_removed_and_rendered_text_identical(tmp_path, prefix, observer):
    request = make_request(prefix(), observer, fold=0)
    v1 = wire_request(request)
    v2 = wire_request_v2(request)
    old = deepcopy(v1)
    template = old["extra_body"].pop("chat_template")
    assert v2 == old and digest(v2) != digest(v1)
    assert "chat_template" not in v2["extra_body"]
    service = service_artifact(tmp_path / "service")
    server_template = (service.root / "chat_template.jinja").read_text()
    env = Environment(undefined=StrictUndefined, autoescape=False)
    a = env.from_string(template).render(messages=v1["messages"], **v1["extra_body"]["chat_template_kwargs"])
    b = env.from_string(server_template).render(messages=v2["messages"], **v2["extra_body"]["chat_template_kwargs"])
    assert a.encode() == b.encode() == request["rendered_prompt"].encode()
    assert b.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


@pytest.fixture(scope="module")
def real_tokenizer():
    path = os.environ.get("UNS_LLM_TOKENIZER_ASSETS")
    if not path:
        pytest.skip("explicit offline B1 tokenizer directory required for optional qualification")
    # Verify pinned B1 file identities before an offline, tokenizer-only load.
    root = Path(path)
    hashes = {
        "config.json": "f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30",
        "generation_config.json": "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2",
        "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
        "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
        "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
        "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
        "LICENSE": "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e"}
    assert {n:sha256_bytes((root/n).read_bytes()) for n in hashes} == hashes
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False)


@pytest.mark.parametrize("prefix,observer", PARITY_CASES)
def test_real_offline_token_ids_v1_v2_and_official_template_equal(tmp_path, prefix, observer, real_tokenizer):
    request = make_request(prefix(), observer, fold=0)
    old, new = wire_request(request), wire_request_v2(request)
    service = service_artifact(tmp_path / "service")
    templates = [old["extra_body"]["chat_template"], (service.root/"chat_template.jinja").read_text(), None]
    texts, ids = [], []
    for template in templates:
        kwargs = {"chat_template": template, "add_generation_prompt": True, "enable_thinking": False}
        texts.append(real_tokenizer.apply_chat_template(new["messages"], tokenize=False, **kwargs))
        ids.append(real_tokenizer.apply_chat_template(new["messages"], tokenize=True, return_dict=False, **kwargs))
    direct = real_tokenizer.encode(request["rendered_prompt"], add_special_tokens=False)
    assert texts == [request["rendered_prompt"]] * 3
    assert ids == [direct] * 3
    assert direct[-7:] == [151644, 77091, 198, 151667, 271, 151668, 271]


@pytest.mark.parametrize("field,value", [
    ("trust_request_chat_template", True), ("trust_request_chat_template", 0),
    ("dtype", "float16"), ("revision", "0"*40), ("tokenizer_revision", "0"*40),
    ("max_model_len", 4096), ("generation_config", "vllm"), ("generation_config_sha256", "0"*64),
    ("served_model_name", "alias"), ("chat_template_content_format", "openai"),
    ("add_generation_prompt", False), ("add_special_tokens", True), ("continue_final_message", True),
    ("reasoning_parser", "qwen3"), ("chat_template", "other.jinja"), ("quantization", "awq")])
def test_validly_sealed_but_drifting_startup_refused(tmp_path, field, value):
    startup = deepcopy(WIRE_CONFIG["server"]); startup[field] = value
    service = service_artifact(tmp_path/"service", startup=startup)
    with pytest.raises(ValueError, match="configuration drift"):
        service.identity()


@pytest.mark.parametrize("changes", [
    {"template_origin":"request"}, {"template_source":"other.jinja"},
    {"wire_config_digest":"0"*64}, {"tokenizer_asset_set_digest":"0"*64},
    {"weights_manifest_sha256":"invalid"}, {"runtime":{}},
    {"endpoint":"https://example.invalid/v1"}, {"service_instance_id":"actual-unverified"},
    {"unexpected":True}])
def test_origin_asset_or_service_schema_drift_refused(tmp_path, changes):
    service=service_artifact(tmp_path/"service", changes=changes)
    with pytest.raises(ValueError): service.identity()


def test_template_bytes_checked_even_with_new_valid_manifest(tmp_path):
    service = service_artifact(tmp_path/"service", template=asset_bytes("chat_template.jinja")+b" ")
    with pytest.raises(ValueError, match="template bytes drift"): service.identity()


def test_wire_config_file_drift_refused_without_changing_real_file(tmp_path, monkeypatch):
    from werewolf.tom_llm import server_fixed
    config=deepcopy(WIRE_CONFIG); config["server"]["trust_request_chat_template"]=True
    path=tmp_path/"config.json"; path.write_bytes(canonical_json_bytes(config))
    monkeypatch.setattr(server_fixed, "CONFIG_PATH", path)
    with pytest.raises(ValueError, match="configuration drift"): wire_binding()


@pytest.mark.parametrize("change", ["template", "nested_template", "thinking", "max_tokens", "schema", "system", "seed", "extra"])
def test_injected_transport_refuses_wire_drift_before_call(tmp_path, change):
    calls=[]; fake=client(lambda **k: calls.append(k) or reply("{}"))
    transport=InjectedServerFixedTransport(fake, service=service_artifact(tmp_path/"service"))
    wire=wire_request_v2(make_request(_prefix(),0,fold=0))
    if change=="template": wire["extra_body"]["chat_template"]="bad"
    elif change=="nested_template": wire["extra_body"]["chat_template_kwargs"]["chat_template"]="bad"
    elif change=="thinking": wire["extra_body"]["chat_template_kwargs"]["enable_thinking"]=True
    elif change=="max_tokens": wire["max_tokens"]=513
    elif change=="schema": wire["response_format"]["json_schema"]["strict"]=False
    elif change=="system": wire["messages"][0]["content"]="other"
    elif change=="seed": wire["seed"]+=1
    else: wire["extra_body"]["repetition_penalty"]=1.1
    with pytest.raises(ValueError): transport.chat(wire)
    assert calls==[]


def test_injected_transport_uses_exact_v2_and_rechecks_endpoint_sdk_and_evidence(tmp_path):
    calls=[]; fake=client(lambda **k: calls.append(k) or reply("{}"))
    service=service_artifact(tmp_path/"service")
    transport=InjectedServerFixedTransport(fake, service=service)
    request=make_request(_prefix(),0,fold=0)
    transport.chat_public(request)
    assert calls==[wire_request_v2(request)]
    fake.base_url="http://127.0.0.1:9000/v1"
    with pytest.raises(ValueError,match="endpoint"): transport.chat_public(request)
    fake.base_url="http://127.0.0.1:8000/v1"; fake.max_retries=1
    with pytest.raises(ValueError,match="retry"): transport.chat_public(request)
    fake.max_retries=0; (service.root/"chat_template.jinja").write_bytes(b"drift")
    with pytest.raises(ValueError): transport.chat_public(request)
    assert len(calls)==1


@pytest.mark.parametrize("failure", [False, True])
def test_v2_durable_lifecycle_and_uniform_vs_fallback_unchanged(tmp_path, failure):
    root, execution, requests=prepared_v2(tmp_path)
    backend=MockBackend((lambda _:Reply("invalid", "stop", {})) if failure else uniform)
    run_mock_study(root,backend,execution=execution)
    run_mock_study(root,backend,execution=execution)
    assert len(backend.calls)==1 and "chat_template" not in backend.calls[0]["extra_body"]
    seal=seal_mock_study(root,execution=execution)
    assert seal.manifest["schema_version"]=="classic7_llm_mainline_mock_seal_v2"
    _,rows=open_mock_predictions(root,execution=execution)
    row=rows[0]
    assert row["prediction"]["status"]==("fallback_uniform" if failure else "valid")
    assert row["wire_version"]==WIRE_VERSION and row["service_digest"]==execution.service.expected_digest
    assert row["request_digest"]==digest(wire_request_v2(requests[0]))
    assert row["request_digest"]!=digest(wire_request(requests[0]))
    assert row["execution_digest"]==digest(execution.identity())
    with pytest.raises(ValueError): run_mock_study(root,backend,execution=execution)


@pytest.mark.parametrize("source_version,target_version", [(1,2),(2,1)])
@pytest.mark.parametrize("terminal", [True, False])
def test_copied_old_or_new_response_cannot_cross_versions(tmp_path, source_version, target_version, terminal):
    request=make_request(_prefix(),0,fold=0)
    root2,execution,_=prepared_v2(tmp_path)
    root1=tmp_path/"study-v1"
    prepare_mock_study(root1,[request],tokenizer_identity="synthetic:counter-v1",count_tokens=lambda _:100)
    roots={1:root1,2:root2}; kwargs={1:{},2:{"execution":execution}}
    if terminal:
        run_mock_study(roots[source_version],MockBackend(uniform),**kwargs[source_version])
    else:
        def crash(): raise RuntimeError("durable attempt only")
        with pytest.raises(RuntimeError):
            run_mock_study(roots[source_version],MockBackend(uniform),after_response=crash,**kwargs[source_version])
    shutil.copytree(roots[source_version]/"requests",roots[target_version]/"requests")
    backend=MockBackend(uniform)
    with pytest.raises(ValueError): run_mock_study(roots[target_version],backend,**kwargs[target_version])
    assert backend.calls==[]


def test_explicit_reader_version_required_and_other_service_cannot_resume(tmp_path):
    root,execution,_=prepared_v2(tmp_path)
    run_mock_study(root,MockBackend(uniform),execution=execution)
    with pytest.raises(ValueError): open_study(root)
    with pytest.raises(ValueError): open_mock_predictions(root)
    other=ServerFixedExecution(service_artifact(tmp_path/"service2", changes={"service_instance_id":"synthetic:other"}))
    backend=MockBackend(uniform)
    with pytest.raises(ValueError,match="drift"): run_mock_study(root,backend,execution=other)
    assert backend.calls==[]
    root1=tmp_path/"study-v1"; request=make_request(_prefix(),0,fold=0)
    prepare_mock_study(root1,[request],tokenizer_identity="synthetic:c",count_tokens=lambda _:100)
    with pytest.raises(ValueError): open_study_v2(root1,execution=execution)


def test_v2_source_drift_refused_before_dispatch(tmp_path, monkeypatch):
    root,execution,_=prepared_v2(tmp_path)
    from werewolf.tom_llm import execution as module
    source=module.source_identity()
    monkeypatch.setattr(module,"source_identity",lambda:{**source,"implementation_digest":"0"*64})
    backend=MockBackend(uniform)
    with pytest.raises(ValueError,match="drift"): run_mock_study(root,backend,execution=execution)
    assert backend.calls==[]


def test_service_drift_during_call_quarantines_inflight_and_never_falls_back(tmp_path):
    root,execution,_=prepared_v2(tmp_path)
    template=execution.service.root/"chat_template.jinja"; original=template.read_bytes()
    def changed(request):
        template.write_bytes(b"drift")
        return uniform(request)
    backend=MockBackend(changed)
    with pytest.raises(ValueError): run_mock_study(root,backend,execution=execution)
    assert len(backend.calls)==1 and not list(root.rglob("result.json"))
    template.write_bytes(original)
    with pytest.raises(InfrastructureFailure,match="in-flight"):
        run_mock_study(root,backend,execution=execution)
    assert len(backend.calls)==1


def test_v2_crash_recovery_and_exact_two_transport_attempt_cap(tmp_path):
    root,execution,_=prepared_v2(tmp_path)
    calls=[]
    def respond(request):
        calls.append(request)
        if len(calls)==1: raise TransportFailure()
        return uniform(request)
    backend=MockBackend(respond)
    def crash(): raise RuntimeError("after durable response")
    with pytest.raises(RuntimeError): run_mock_study(root,backend,execution=execution,after_response=crash)
    run_mock_study(root,backend,execution=execution)
    assert len(calls)==2 and calls[0]==calls[1]
    seal_mock_study(root,execution=execution)
    assert len(open_mock_predictions(root,execution=execution)[1][0]["attempt_digests"])==2


def selection_inputs(partition):
    selection=select_games(partition,scope="subset150")
    expected={"parent_folds_digest":digest(partition),"selection_digest":selection["selection_digest"],
              "game_ids_digest":digest(selection["game_ids"]),"selection_payload_sha256":digest(selection)}
    provenance={key:digest(key) for key in ["publication_digest","fold_manifest_digest","prepared_manifest_digest",
        "pre_reference_selection_evidence_digest","reconciliation_evidence_digest"]}
    prepared={"artifact_type":"backbone_tom_study", "schema_version":"synthetic:prepared-v1",
              "protocol_inputs":{k:provenance[k] for k in ("publication_digest", "fold_manifest_digest")},
              "folds":partition}
    prepared["manifest_digest"]=digest(prepared)
    provenance["prepared_manifest_digest"]=prepared["manifest_digest"]
    return selection,expected,provenance,prepared


def test_selection_existing_seal_binding_and_no_overwrite(tmp_path):
    partition=folds(); selection,expected,provenance,prepared=selection_inputs(partition)
    artifact=publish_selection(tmp_path/"selection",partition,scope="subset150")
    bound=open_selection_binding(artifact.path,partition,expected=expected,provenance=provenance,prepared_manifest=prepared)
    assert bound["selection_artifact_manifest_digest"]==artifact.manifest_digest
    assert bound["selection_digest"]==selection["selection_digest"] and bound["rule"]==selection["rule"]
    with pytest.raises(ValueError): publish_selection(artifact.path,partition,scope="subset150")


@pytest.mark.parametrize("change", ["parent", "games", "fold", "rule", "expected", "provenance", "prepared"])
def test_selection_rejects_drift_without_intersection(change):
    partition=folds(); selection,expected,provenance,prepared=selection_inputs(partition)
    if change=="parent": partition[0]["schedule_digest"]="0"*64
    elif change=="games": selection["game_ids"][0]="foreign"
    elif change=="fold": selection["folds"][0]["game_ids"].pop()
    elif change=="rule": selection["rule"]["seed"]+=1
    elif change=="expected": expected["selection_digest"]="0"*64
    elif change=="prepared": prepared["protocol_inputs"]["publication_digest"]="0"*64
    else: provenance["prepared_manifest_digest"]="0"*64
    with pytest.raises(ValueError): validate_selection_source(selection,partition,expected=expected,provenance=provenance,prepared_manifest=prepared)


def test_v2_exhausted_transport_does_not_generate_uniform_and_live_backend_rejected(tmp_path):
    root,execution,_=prepared_v2(tmp_path)
    with pytest.raises(ValueError,match="only local"):
        run_mock_study(root,object(),execution=execution)
    def lost(_): raise TransportFailure()
    backend=MockBackend(lost)
    for _ in range(2):
        with pytest.raises(InfrastructureFailure,match="budget exhausted"):
            run_mock_study(root,backend,execution=execution)
    assert len(backend.calls)==2 and not list(root.rglob("result.json"))
    with pytest.raises((ValueError, OSError)): seal_mock_study(root,execution=execution)


def test_resealed_service_cannot_replace_external_expected_digest(tmp_path):
    original=service_artifact(tmp_path/"original")
    replacement=service_artifact(tmp_path/"replacement",changes={"service_instance_id":"synthetic:replacement"})
    with pytest.raises(ValueError,match="identity drift"):
        ServerFixedService(replacement.root, original.expected_digest).identity()


def test_dispatch_timing_excludes_post_response_identity_check(tmp_path, monkeypatch):
    from werewolf.tom_llm import inference
    root,execution,_=prepared_v2(tmp_path)
    clock=[0.0]; original=inference._check_execution
    def checked(contract, bound):
        original(contract, bound)
        clock[0]+=10.0
    def respond(request):
        clock[0]+=1.0
        return uniform(request)
    monkeypatch.setattr(inference, "_check_execution", checked)
    monkeypatch.setattr(inference.time, "perf_counter", lambda:clock[0])
    run_mock_study(root,MockBackend(respond),execution=execution)
    seal_mock_study(root,execution=execution)
    rows=open_mock_predictions(root,execution=execution)[1]
    assert rows[0]["dispatch_elapsed_seconds"]==[1.0]
