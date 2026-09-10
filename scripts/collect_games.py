"""Operator preflight and Plan freezing; collection remains owned by uns."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import openai
import werewolf
import yaml

from werewolf import cli
from werewolf.agents.gpt_agent import GAMEPLAY_GENERATION_MAX_ATTEMPTS
from werewolf.agents.prompt_template_v0 import STRICT_CLASSIC7_GAMEPLAY_PROMPT_PROFILE
from werewolf.artifact_io import canonical_json_bytes, sha256_bytes
from werewolf.artifact_io.canonical import ensure_durable_directory
from werewolf.canonical_collection.attempt_ledger import (
    _publish_bytes_noreplace,
    collection_plan_from_record,
    construct_collection_plan,
    validate_collection_plan,
)
from werewolf.canonical_collection.game_bundle import CANONICAL_GAME_BUNDLE_SCHEMA_VERSION
from werewolf.canonical_collection.pre import AUTHORITATIVE_PRE_PREFIX_SCHEMA_VERSION
from werewolf.canonical_collection.public_history import PUBLIC_EVENT_SCHEMA_VERSION
from werewolf.canonical_collection.speech import (
    V1_ANNOTATION_SCHEMA_VERSION,
    V1_SPEECH_PARSER_VERSION,
    V1_SPEECH_PROMPT_VERSION,
)
from werewolf.canonical_collection.trajectory_evidence import BELIEF_OBSERVATION_SCHEMA_VERSION
from werewolf.runtime_config import normalize_runtime_config
from werewolf.speech.private_belief_perceiver import LABEL_GENERATION_MAX_ATTEMPTS
from werewolf.speech.speech_perceiver import SPEECH_PARSER_GENERATION_MAX_ATTEMPTS


REPOSITORY = Path(__file__).resolve().parents[1]
STORAGE = REPOSITORY / "configs/server.json"
RUNTIME = REPOSITORY / "configs/runtime/local-qwen35-9b.yaml"
DEPLOYMENT = REPOSITORY / "configs/deployment/qwen35-9b.yaml"
VLLM_ENV = Path("/data/yuxiao/envs/untrusted-network-simulation-vllm")
SEED_RULE = "classic7-canonical-collection-v1:sha256-first8-big-endian-low63:ordinal0"
CALIBRATION_SEEDS = range(900000001, 900000016)


def read_json(path):
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(Path(path).read_bytes(), object_pairs_hook=unique_keys)


def validate_campaign(value):
    fields = {"collection_id", "target_games", "seed_pool_size", "call_limit"}
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("campaign requires exactly collection_id, target_games, seed_pool_size, call_limit")
    identity = value["collection_id"]
    if (not isinstance(identity, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", identity)
            or ".." in identity):
        raise ValueError("collection_id must be a safe filesystem identifier")
    for field in ("target_games", "seed_pool_size", "call_limit"):
        if type(value[field]) is not int or value[field] <= 0:
            raise ValueError(f"{field} must be a positive integer")
    if value["seed_pool_size"] < value["target_games"]:
        raise ValueError("seed_pool_size must be >= target_games")
    return dict(value)


def derive_seed_pool(collection_id, size):
    seeds = tuple(
        int.from_bytes(hashlib.sha256(
            f"classic7-canonical-collection-v1:{collection_id}:{ordinal}".encode("utf-8")
        ).digest()[:8], "big") & ((1 << 63) - 1)
        for ordinal in range(size)
    )
    if len(set(seeds)) != size or any(seed in CALIBRATION_SEEDS for seed in seeds):
        raise ValueError("seed collision or forbidden calibration seed; no skip or re-roll")
    return seeds


def clean_head():
    def git(*args):
        return subprocess.check_output(["git", "-C", str(REPOSITORY), *args], text=True).strip()

    if git("status", "--porcelain", "--untracked-files=all"):
        raise ValueError("collection requires a clean Git checkout")
    if Path(werewolf.__file__).resolve().parent != REPOSITORY / "werewolf":
        raise ValueError("installed werewolf must be the current source checkout")
    return git("rev-parse", "HEAD")


def model_manifest(model):
    if not model.is_absolute() or ".." in model.parts:
        raise ValueError("model directory must be an explicit absolute path")
    if any(path.is_symlink() for path in (model, *model.parents)) or not model.is_dir():
        raise ValueError("model directory must exist without symlinks")
    records = []
    for path in sorted(model.rglob("*"), key=lambda p: p.relative_to(model).as_posix()):
        if path.is_symlink():
            raise ValueError("model manifest rejects symlinks")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("model manifest requires regular files")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
            after = os.fstat(handle.fileno())
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("model file changed while hashing")
        records.append({"path": path.relative_to(model).as_posix(),
                        "size_bytes": before.st_size, "sha256": digest.hexdigest()})
    if not records:
        raise ValueError("empty model directory")
    revision_bytes = (model / "HF_REVISION").read_bytes()
    revision = revision_bytes.decode("utf-8").strip()
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("model/HF_REVISION must contain an exact HF commit")
    if not any(record["path"] == "HF_REVISION" and record["sha256"] == sha256_bytes(revision_bytes)
               for record in records):
        raise ValueError("HF_REVISION changed while hashing the model")
    return sha256_bytes(canonical_json_bytes(records)), revision


def software_versions():
    # Query the serving environment, not the main environment's Torch stack.
    probe = (
        "import json,platform,torch; from importlib.metadata import version; "
        "print(json.dumps({'python':platform.python_version(),"
        "'vllm':version('vllm'),'torch':torch.__version__,'torch_cuda':torch.version.cuda,"
        "'flashinfer-python':version('flashinfer-python'),"
        "'flashinfer-jit-cache':version('flashinfer-jit-cache')}))"
    )
    env = dict(os.environ, HOME="/data/yuxiao/conda-home", PYTHONNOUSERSITE="1",
               XDG_CACHE_HOME="/data/yuxiao/cache/xdg",
               TMPDIR="/data/yuxiao/tmp/vllm-build")
    serving = json.loads(subprocess.check_output(
        [str(VLLM_ENV / "bin/python"), "-c", probe], env=env, text=True))
    if any(not isinstance(value, str) or not value for value in serving.values()):
        raise ValueError("serving software version probe is incomplete")
    return {"client_python": platform.python_version(), "client_openai": version("openai"),
            **{f"serving_{key}": value for key, value in serving.items()}}


def live_preflight(base_url, served_model):
    with openai.DefaultHttpx2Client(trust_env=False, follow_redirects=False) as client:
        health = client.get(base_url.removesuffix("/v1") + "/health")
        health.raise_for_status()
        response = client.get(base_url + "/models")
        response.raise_for_status()
        if served_model not in {item["id"] for item in response.json()["data"]}:
            raise ValueError("vLLM /v1/models does not expose the configured model")


def inspect_inputs():
    runtime = normalize_runtime_config(yaml.safe_load(RUNTIME.read_bytes()))
    serve_bytes = DEPLOYMENT.read_bytes()
    deployment = yaml.safe_load(serve_bytes)
    if (deployment["language-model-only"] is not True
            or deployment["default-chat-template-kwargs"]["enable_thinking"] is not False
            or deployment["max-num-seqs"] != 1 or deployment["enforce-eager"] is not True):
        raise ValueError("deployment must retain text-only, thinking disabled, single-sequence eager execution")
    base_url = f"http://{deployment['host']}:{deployment['port']}/v1"
    if deployment["host"] != "127.0.0.1":
        raise ValueError("campaign requires the explicit loopback vLLM deployment")
    served = deployment["served-model-name"]
    if not isinstance(served, str) or not served:
        raise ValueError("served-model-name must be one explicit name")
    if any(backend["base_url"] != base_url or backend.get("default_model") != served
           for backend in runtime["backends"].values()):
        raise ValueError("runtime and deployment backend identities differ")
    if (runtime["parser"]["model"] != served
            or any(profile["model"] != served for profile in runtime["agent_config"]["all_candidates"])):
        raise ValueError("runtime and deployment model identities differ")
    manifest, revision = model_manifest(Path(deployment["model"]))
    provenance = {
        "runtime_config_sha256": sha256_bytes(canonical_json_bytes(runtime)),
        "serve_config_sha256": sha256_bytes(serve_bytes),
        "model_manifest_sha256": manifest,
        "model_manifest_rule": "sorted-relative-path-size_bytes-sha256-canonical-json-v1",
        "hf_revision": revision, "served_model_name": served,
        "model_path": deployment["model"], "seed_rule_identity": SEED_RULE,
        **software_versions(),
    }
    return provenance, base_url


def plan_fields(campaign, head, provenance):
    environment = {**provenance, "configured_call_limit": str(campaign["call_limit"]),
                   "gameplay_prompt_profile": STRICT_CLASSIC7_GAMEPLAY_PROMPT_PROFILE,
                   "seed_pool_size": str(campaign["seed_pool_size"]),
                   "target_success_count": str(campaign["target_games"])}
    return dict(
        collection_id=campaign["collection_id"],
        target_canonical_success_count=campaign["target_games"],
        runtime_identity="classic7-canonical-runtime-v1",
        agent_identity="classic7-gpt-pre-belief-handoff-v1",
        backend_identity="openai-compatible-loopback-vllm-v1",
        model_identity=provenance["served_model_name"],
        parser_identity=V1_SPEECH_PARSER_VERSION,
        prompt_identity=V1_SPEECH_PROMPT_VERSION,
        retry_policy_identity=(f"classic7-transport0-gameplay{GAMEPLAY_GENERATION_MAX_ATTEMPTS}"
                               f"-belief{LABEL_GENERATION_MAX_ATTEMPTS}"
                               f"-parser{SPEECH_PARSER_GENERATION_MAX_ATTEMPTS}-v1"),
        call_budget_identity=f"classic7-backend-dispatch-cap-{campaign['call_limit']}-v1",
        public_event_schema_version=PUBLIC_EVENT_SCHEMA_VERSION,
        pre_prefix_schema_version=AUTHORITATIVE_PRE_PREFIX_SCHEMA_VERSION,
        belief_observation_schema_version=BELIEF_OBSERVATION_SCHEMA_VERSION,
        v1_annotation_schema_version=V1_ANNOTATION_SCHEMA_VERSION,
        bundle_schema_version=CANONICAL_GAME_BUNDLE_SCHEMA_VERSION,
        source_revision=head, environment_provenance=environment,
    )


def load_plan(path):
    plan = collection_plan_from_record(read_json(path))
    return validate_collection_plan(plan)


def publish_plan(path, plan):
    # Reuse the exact atomic hard-link/fsync primitive that publishes ledger Plans.
    staging = path.parent / ".staging"
    ensure_durable_directory(staging)
    _publish_bytes_noreplace(staging_directory=staging, final_path=path,
                           durable_directory=path.parent, data=canonical_json_bytes(plan.to_record()))
    loaded = load_plan(path)
    if loaded != plan:
        raise ValueError("published Plan differs after reload")
    return loaded


def run_campaign(campaign_path=None, *, resume=False):
    root = cli._storage_root(STORAGE)
    campaign_path = campaign_path or root / "operator/collection_campaign.json"
    campaign = validate_campaign(read_json(campaign_path))
    identity = campaign["collection_id"]
    plan_path = cli._artifact_path(root, f"plans/{identity}.json")
    destination = cli._artifact_path(root, identity, "canonical")
    if resume:
        if not plan_path.is_file() or not destination.is_dir():
            raise ValueError("--resume requires the existing Plan and canonical destination")
        cli._require_resume(destination, True)
        plan = load_plan(plan_path)
    elif plan_path.exists() or destination.exists():
        raise ValueError("existing Plan/destination: no implicit restart; --resume requires both")
    head = clean_head()
    provenance, base_url = inspect_inputs()
    fields = plan_fields(campaign, head, provenance)
    if resume:
        for field, expected in fields.items():
            actual = dict(plan.environment_provenance) if field == "environment_provenance" else getattr(plan, field)
            if actual != expected:
                raise ValueError(f"frozen Plan mismatch: {field}")
        if len(plan.ordered_seed_pool) != campaign["seed_pool_size"]:
            raise ValueError("frozen seed pool size mismatch")
        if any(seed in CALIBRATION_SEEDS for seed in plan.ordered_seed_pool):
            raise ValueError("frozen Plan contains calibration seeds")
    live_preflight(base_url, provenance["served_model_name"])
    if clean_head() != head:
        raise ValueError("source revision changed during preflight")
    if not resume:
        plan = construct_collection_plan(
            ordered_seed_pool=derive_seed_pool(identity, campaign["seed_pool_size"]), **fields)
        plan = publish_plan(plan_path, validate_collection_plan(plan))
    print("USING EXISTING FROZEN COLLECTION PLAN" if resume else "COLLECTION PLAN FROZEN", flush=True)
    print("COLLECTION PREFLIGHT PASS", flush=True)
    bound = dict(plan.environment_provenance)
    print(json.dumps({"collection_id": plan.collection_id,
                      "target_canonical_success_count": plan.target_canonical_success_count,
                      "ordered_seed_pool_size": len(plan.ordered_seed_pool),
                      "configured_call_limit": int(bound["configured_call_limit"]),
                      "source_revision": plan.source_revision,
                      **{key: bound[key] for key in ("runtime_config_sha256", "serve_config_sha256",
                          "model_manifest_sha256", "hf_revision")},
                      "plan_digest": plan.plan_digest, "plan_path": str(plan_path),
                      "canonical_destination": str(destination), "mode": "resume" if resume else "new"},
                     sort_keys=True), flush=True)
    args = ["--storage-profile", str(STORAGE), "collect", "--plan", str(plan_path),
            "--runtime-config", str(RUNTIME), "--call-limit", bound["configured_call_limit"],
            "--destination", plan.collection_id]
    if resume:
        args.append("--resume")
    return cli.main(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    return run_campaign(args.campaign, resume=args.resume)


if __name__ == "__main__":
    sys.exit(main())
