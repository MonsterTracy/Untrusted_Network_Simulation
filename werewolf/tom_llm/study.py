"""Immutable public request contracts and engineering-only preparation."""

from pathlib import Path
import subprocess

from werewolf.artifact_io import canonical_jsonl_bytes, publish_artifact, verify_artifact, sha256_bytes
from werewolf.structured_history import plan_structured_history, StructuredTokenPlan, StructuredTokenDescriptor
from werewolf.tom_llm.common import digest, identity, require, strict_json
from werewolf.tom_llm.policy import BASE_COMMIT, POLICY, policy_binding
from werewolf.tom_llm.prompt import render_prompt, asset_bytes
from werewolf.tom_llm.development_source import development_source_identity

VERSION = "classic7_llm_mainline_engineering_study_v1"


def source_identity():
    root = Path(__file__).resolve().parents[2]
    paths = sorted((root / "werewolf").rglob("*.py")) + sorted((root / "werewolf/tom_llm/assets").iterdir())
    paths += [root / "configs/paper/tom-llm-reasoner-v1.json", root / "setup.py",
              root / "scripts/paper_backbone_contrasts.py"]
    files = {p.relative_to(root).as_posix(): sha256_bytes(p.read_bytes()) for p in paths if p.is_file()}
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    require(head == BASE_COMMIT, "engineering worktree base commit mismatch")
    return {"base_commit": head, "implementation_digest": digest(files), "file_digests": files, "engineering_only": True}


def make_request(prefix, observer, *, fold):
    require(type(fold) is int and 0 <= fold < 5, "invalid request fold")
    plan = plan_structured_history(prefix)
    require(observer in [int(p.removeprefix("player")) - 1 for p in prefix.alive_observer_ids],
            "dispatch must use a public alive observer")
    prompt = render_prompt(plan, observer)
    return {"game_id": prefix.game_id, "boundary_id": prefix.boundary_id, "prefix_digest": plan.prefix_digest,
            "plan_digest": plan.plan_digest, "observer": observer, "fold": fold, "public_plan": plan.to_record(),
            "messages": list(prompt.messages), "rendered_prompt": prompt.rendered,
            "prompt_digest": prompt.prompt_digest, "response_schema": prompt.schema}


def public_requests(prefixes, fold_of_game):
    requests = [make_request(prefix, int(seat.removeprefix("player")) - 1, fold=fold_of_game[prefix.game_id])
                for prefix in prefixes for seat in prefix.alive_observer_ids]
    require(requests and len({identity(r) for r in requests}) == len(requests), "duplicate or empty public requests")
    return requests


def validate_request(request):
    fields = {"game_id", "boundary_id", "prefix_digest", "plan_digest", "observer", "fold",
              "public_plan", "messages", "rendered_prompt", "prompt_digest", "response_schema"}
    require(type(request) is dict and set(request) == fields, "request schema forbids labels/private fields")
    require(type(request["game_id"]) is str and request["game_id"] and type(request["boundary_id"]) is str
            and request["boundary_id"] and type(request["fold"]) is int and 0 <= request["fold"] < 5, "invalid request identity")
    value = dict(request["public_plan"])
    value["tokens"] = tuple(StructuredTokenDescriptor(**t) for t in value["tokens"])
    plan = StructuredTokenPlan(**value)
    prompt = render_prompt(plan, request["observer"])
    require(request["prefix_digest"] == plan.prefix_digest and request["plan_digest"] == plan.plan_digest
            and request["messages"] == list(prompt.messages) and request["rendered_prompt"] == prompt.rendered
            and request["response_schema"] == prompt.schema and request["prompt_digest"] == prompt.prompt_digest,
            "request differs from public plan/frozen renderer")
    return request


def wire_request(request):
    validate_request(request)
    policy_binding()
    decoding = POLICY["decoding"]
    return {"model": POLICY["model"], "messages": request["messages"], "temperature": decoding["temperature"],
        "top_p": decoding["top_p"], "n": 1, "max_tokens": decoding["max_tokens"], "seed": decoding["seed"],
        "response_format": {"type": "json_schema", "json_schema": {"name": "non_self_suspicion",
            "strict": True, "schema": request["response_schema"]}},
        "extra_body": {"chat_template": asset_bytes("chat_template.jinja").decode("utf-8"),
            "chat_template_kwargs": {"enable_thinking": False}, "top_k": -1, "min_p": 0}}


def context_preflight(requests, count_tokens):
    counts = [count_tokens(validate_request(r)["rendered_prompt"]) for r in requests]
    require(counts and all(type(n) is int and 0 < n <= POLICY["context_length"] - POLICY["decoding"]["max_tokens"]
                for n in counts), "full prompt plus output exceeds frozen context; no truncation permitted")
    return {"input_token_counts": counts, "request_count": len(counts), "input_tokens_total": sum(counts),
            "maximum_input_tokens": max(counts)}


def prepare_mock_study(destination, requests, *, tokenizer_identity, count_tokens):
    """No live factory exists in phase two: these artifacts cannot be formal results."""
    require(type(tokenizer_identity) is str and tokenizer_identity.startswith("synthetic:"), "mock tokenizer required")
    require(requests and len({identity(r) for r in requests}) == len(requests), "duplicate or empty public request set")
    fields = {"artifact_type": "llm_engineering_study", "schema_version": VERSION,
              "engineering_only": True, "policy": policy_binding(), "source": development_source_identity(),
              "tokenizer_identity": tokenizer_identity, "preflight": context_preflight(requests, count_tokens),
              "row_identity_digest": digest([identity(r) for r in requests]), "request_count": len(requests)}
    return publish_artifact(Path(destination) / "contract", manifest_fields=fields,
                            files={"requests.jsonl": canonical_jsonl_bytes(requests)})


def open_study(root):
    artifact = verify_artifact(Path(root) / "contract", expected_artifact_type="llm_engineering_study",
                               expected_schema_version=VERSION)
    m = artifact.manifest
    require(m["engineering_only"] is True and m["policy"] == policy_binding() and m["source"] == development_source_identity(),
            "study policy/source drift")
    rows = [strict_json(line) for line in (artifact.path / "requests.jsonl").read_bytes().splitlines()]
    for row in rows:
        validate_request(row)
    require(len(rows) == m["request_count"] and len({identity(r) for r in rows}) == len(rows)
            and digest([identity(r) for r in rows]) == m["row_identity_digest"], "study request coverage mismatch")
    return artifact, rows
