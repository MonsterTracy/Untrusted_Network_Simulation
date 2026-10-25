"""New Mock source domain; old Candidate/Study identities are never upgraded."""

from pathlib import Path
import importlib.util
import subprocess
from unittest.mock import patch
import runpy

import pytest

from tests.canonical_collection.test_structured_token_plan import _prefix
from tests.tom_llm.test_selection_inference import prepared, uniform
from tests.tom_llm.test_transport import client, identity, reply
from tests.tom_llm.test_server_fixed import service_artifact
from werewolf.artifact_io import publish_artifact
from werewolf.tom_llm import study, development_source
from werewolf.tom_llm.inference import MockBackend, run_mock_study, verify_result
from werewolf.tom_llm.policy import BASE_COMMIT
from werewolf.tom_llm.server_fixed import InjectedServerFixedTransport, ServerFixedService, SERVICE_VERSION, wire_request_v2
from werewolf.tom_llm.transport import InjectedChatTransport


def test_legacy_head_guard_is_preserved_and_rejects_mainline():
    assert BASE_COMMIT == "d8397ce56c65d6d46b24a6b53f0b71431c04267b"
    with pytest.raises(ValueError, match="base commit mismatch"):
        study.source_identity()


def test_development_source_records_actual_checkout_and_inventory():
    source = development_source.development_source_identity()
    root = Path(development_source.__file__).resolve().parents[2]
    actual = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    assert source["source_revision"] == actual
    assert source["schema_version"] == "classic7_llm_mainline_development_source_v1"
    assert source["integration_base_commit"] == "24950337cd307d4bc900cf41b5193907cde505ae"
    assert source["qualification"] == "MOCK_ONLY_NOT_CANDIDATE" and source["engineering_only"] is True
    assert "werewolf/tom_llm/development_source.py" in source["file_digests"]
    assert "werewolf/cli.py" in source["file_digests"]
    assert type(source["source_inventory_clean"]) is bool
    assert not {"base_commit", "admission_digest", "candidate_digest"} & set(source)


@pytest.mark.parametrize("version", ["classic7_llm_engineering_study_v1", "classic7_llm_engineering_study_v2"])
def test_legacy_mock_schema_is_not_accepted_as_new_study(tmp_path, version):
    root, _ = prepared(tmp_path)
    artifact, _ = study.open_study(root)
    fields = {k: v for k, v in artifact.manifest.items() if k not in {"manifest_digest", "file_table"}}
    fields["schema_version"] = version
    legacy = tmp_path / "legacy"
    publish_artifact(legacy / "contract", manifest_fields=fields,
                    files={"requests.jsonl": (artifact.path / "requests.jsonl").read_bytes()})
    backend = MockBackend(uniform)
    with pytest.raises(ValueError):
        run_mock_study(legacy, backend)
    assert backend.calls == []


def test_new_schema_does_not_accept_legacy_source_binding(tmp_path):
    root, _ = prepared(tmp_path)
    artifact, _ = study.open_study(root)
    fields = {k: v for k, v in artifact.manifest.items() if k not in {"manifest_digest", "file_table"}}
    fields["source"] = {"base_commit": BASE_COMMIT, "engineering_only": True,
                         "implementation_digest": "a" * 64, "file_digests": {}}
    wrong = tmp_path / "wrong-source"
    publish_artifact(wrong / "contract", manifest_fields=fields,
                    files={"requests.jsonl": (artifact.path / "requests.jsonl").read_bytes()})
    backend = MockBackend(uniform)
    with pytest.raises(ValueError, match="source drift"):
        run_mock_study(wrong, backend)
    assert backend.calls == []


def test_runtime_inventory_change_blocks_mock_resume(tmp_path, monkeypatch):
    root, _ = prepared(tmp_path)
    target = Path(study.__file__)
    original = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda p: original(p) + b"\n# test-only drift\n"
                        if p == target else original(p))
    backend = MockBackend(uniform)
    with pytest.raises(ValueError, match="source drift"):
        run_mock_study(root, backend)
    assert backend.calls == []


def test_candidate_and_physical_qualification_are_not_installed():
    for name in ("admission", "candidate", "qualification", "probe"):
        assert importlib.util.find_spec("werewolf.tom_llm." + name) is None


def test_setup_keeps_existing_extras_and_only_prompt_resources():
    root = Path(study.__file__).resolve().parents[2]
    with patch("setuptools.setup") as setup:
        runpy.run_path(str(root / "setup.py"), run_name="__test__")
    config = setup.call_args.kwargs
    assert config["entry_points"]["console_scripts"] == ["uns=werewolf.cli:main"]
    assert set(config["extras_require"]) == {"tom", "mapper", "dev", "llm-reasoner"}
    assert config["extras_require"]["llm-reasoner"] == ["Jinja2>=3.1,<4"]
    assert config["package_data"] == {"werewolf.tom_llm": ["assets/*.txt", "assets/*.jinja"]}


@pytest.mark.parametrize("during", [False, True])
def test_injected_transport_rejects_endpoint_drift_before_return(during):
    calls = []
    fake = client(lambda **kw: reply())
    def create(**kw):
        calls.append(kw)
        fake.base_url = "https://untrusted.invalid/v1"
        return reply("{}")
    fake.chat.completions.create = create
    transport = InjectedChatTransport(fake, attestation=identity())
    if not during:
        fake.base_url = "https://untrusted.invalid/v1"
    request = study.wire_request(study.make_request(_prefix(), 0, fold=0))
    with pytest.raises(ValueError, match="endpoint or retry drift"):
        transport.chat(request)
    assert len(calls) == int(during)


def test_server_fixed_service_is_checked_after_injected_response(tmp_path):
    service = service_artifact(tmp_path / "service")
    def changed(**kwargs):
        (service.root / "chat_template.jinja").write_bytes(b"synthetic drift")
        return reply("{}")
    transport = InjectedServerFixedTransport(client(changed), service=service)
    with pytest.raises(ValueError):
        transport.chat(wire_request_v2(study.make_request(_prefix(), 0, fold=0)))


def test_legacy_result_schema_is_not_accepted_by_mainline_reader(tmp_path):
    root, request = prepared(tmp_path)
    run_mock_study(root, MockBackend(uniform))
    contract, _ = study.open_study(root)
    result = next(root.glob('requests/*/result'))
    artifact, _ = verify_result(result, contract, request)
    fields = {k: v for k, v in artifact.manifest.items() if k not in {'manifest_digest', 'file_table'}}
    fields['schema_version'] = 'classic7_llm_mock_result_v1'
    legacy = publish_artifact(tmp_path / 'legacy-result', manifest_fields=fields,
                              files={'result.json': (result / 'result.json').read_bytes()})
    with pytest.raises(ValueError):
        verify_result(legacy.path, contract, request)


def test_legacy_service_schema_is_not_a_new_mock_admission(tmp_path):
    service = service_artifact(tmp_path / 'new-service')
    from werewolf.artifact_io import verify_artifact
    artifact = verify_artifact(service.root, expected_artifact_type='llm_mock_serving_identity',
                               expected_schema_version=SERVICE_VERSION)
    fields = {k: v for k, v in artifact.manifest.items() if k not in {'manifest_digest', 'file_table'}}
    fields['schema_version'] = 'classic7_llm_server_fixed_service_v1'
    legacy = publish_artifact(tmp_path / 'legacy-service', manifest_fields=fields,
        files={name: (service.root / name).read_bytes() for name in artifact.manifest['file_table']})
    with pytest.raises(ValueError):
        ServerFixedService(legacy.path, legacy.manifest_digest).identity()


def test_default_cli_parser_does_not_import_optional_llm_or_model_dependencies():
    import sys
    code = '''import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'jinja2', 'torch', 'transformers', 'openai'}:
        raise AssertionError('default parser imported optional dependency: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from werewolf.cli import build_parser
p = build_parser()
assert p.parse_args(['prepare-backbone-tom-study', '--config', 'c', '--publication', 'p',
                     '--destination', 'd', '--paired-temporal']).paired_temporal is True
assert p.parse_args(['check-llm-reasoner-config', '--config', 'c']).command == 'check-llm-reasoner-config'
'''
    result = subprocess.run([sys.executable, '-B', '-s', '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
