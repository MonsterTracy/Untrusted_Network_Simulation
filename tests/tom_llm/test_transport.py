from types import SimpleNamespace
from copy import deepcopy

import openai
import pytest

from tests.canonical_collection.test_structured_token_plan import _prefix
from werewolf.tom_llm.policy import POLICY, ASSET_DIGESTS
from werewolf.tom_llm.study import make_request, wire_request
from werewolf.tom_llm.transport import InjectedChatTransport
from werewolf.tom_llm.inference import InfrastructureFailure, TransportFailure
from werewolf.tom_llm.output import parse_output


def identity():
    return {"model":POLICY["model"],"revision":POLICY["model_revision"],"dtype":"bfloat16",
            "tokenizer_revision":POLICY["model_revision"],
            "chat_template_sha256":ASSET_DIGESTS["chat_template.jinja"],"weights_manifest_sha256":"a"*64}


def client(create):
    return SimpleNamespace(max_retries=0,base_url="http://127.0.0.1:8000/v1",
                           chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def reply(content=None,refusal=None,reason="stop"):
    return SimpleNamespace(model=POLICY["model"],
        choices=[SimpleNamespace(message=SimpleNamespace(content=content,refusal=refusal),finish_reason=reason)],
        usage=SimpleNamespace(prompt_tokens=100,completion_tokens=5,total_tokens=105))


def test_injected_client_receives_exact_frozen_request_without_factory():
    calls=[]
    def create(**request):
        calls.append(request); return reply("{}",reason="length")
    transport=InjectedChatTransport(client(create),attestation=identity())
    request=wire_request(make_request(_prefix(),0,fold=0))
    result=transport.chat(request)
    assert calls == [request]
    assert result.usage=={"input_tokens":100,"output_tokens":5,"total_tokens":105}
    assert parse_output(result.content,0,finish_reason=result.finish_reason).failure_type=="length"


@pytest.mark.parametrize("change",["sdk","endpoint","revision","dtype","template"])
def test_provider_identity_and_hidden_retries_are_rejected(change):
    fake=client(lambda **_:reply()); attestation=identity()
    if change=="sdk": fake.max_retries=2
    elif change=="endpoint": fake.base_url="https://remote.example/v1"
    elif change=="revision": attestation["revision"]="0"*40
    elif change=="dtype": attestation["dtype"]="float16"
    else: attestation["chat_template_sha256"]="0"*64
    with pytest.raises(ValueError):
        InjectedChatTransport(fake,attestation=attestation)


def test_refusal_becomes_auditable_semantic_failure_not_transport_retry():
    transport=InjectedChatTransport(client(lambda **_:reply(refusal="refused")),attestation=identity())
    result=transport.chat(wire_request(make_request(_prefix(),0,fold=0)))
    assert parse_output(result.content,0,finish_reason=result.finish_reason).failure_type=="empty_or_refusal"


def test_connection_error_is_explicit_transport_and_service_failure_pauses():
    def lost(**_):
        raise openai.APIConnectionError(message="synthetic connection loss",request=object())
    request=wire_request(make_request(_prefix(),0,fold=0))
    with pytest.raises(TransportFailure):
        InjectedChatTransport(client(lost),attestation=identity()).chat(request)
    with pytest.raises(InfrastructureFailure,match="model/choice"):
        InjectedChatTransport(client(lambda **_:SimpleNamespace(model=POLICY["model"])),
                              attestation=identity()).chat(request)
    def broken(**_): raise RuntimeError("synthetic OOM")
    with pytest.raises(InfrastructureFailure):
        InjectedChatTransport(client(broken),attestation=identity()).chat(request)
