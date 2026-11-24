"""Router adapter to the existing production loop and canonical bundle publisher."""
from __future__ import annotations

from contextlib import ExitStack
import inspect
import os
from pathlib import Path
import subprocess

from werewolf.artifact_io import canonical_json_bytes, sha256_bytes, verify_artifact
from werewolf.canonical_collection.attempt_ledger import collection_plan_from_record
from werewolf.canonical_collection.failure_evidence import (
    construct_canonical_partial_evidence, publish_canonical_partial_evidence,
    validate_canonical_partial_evidence,
)
from werewolf.canonical_collection.game_bundle import _raw_public_event, publish_canonical_game_bundle
from werewolf.canonical_collection.public_history import freeze_public_event_history
from werewolf.interventions.actions import context_from_pre
from werewolf.interventions.gameplay import (
    GameplayLedger, digest, freeze_source, policy_for, publish_campaign, recover,
    REQUIRED_PILOT_EXCLUSIONS, TOM_GAMEPLAY_SEED_COUNT, TOM_GAMEPLAY_SEED_SOURCE,
    collection_plan_inputs, qualification_source_matches, read_qualification_plan, tom_seed_exclusion,
    safe_path, verify_results, bind_campaign_profile, write_once,
)
from werewolf.interventions.online.plan import probe_candidate_pool, router_assignment_from_record
from werewolf.interventions.online.records import validate_probe_lifecycle_record


def execute_report_event_games(*, plan, reference_collection, assignment_seed,
                               destination, runtime_factory, predictor):
    """在 Canonical Collection 中按游戏预先分配三臂，各局最多干预一次。"""
    from collections import Counter
    from random import Random
    from scripts import analyze_observer_dtrs_report_event as study
    from werewolf.canonical_collection.attempt_ledger import (
        CollectionSeedPoolExhausted, load_collection_plan, validate_attempt_ledger, validate_collection_plan,
    )
    from werewolf.canonical_collection.collector import CollectionResult, collect
    from werewolf.canonical_collection.production_runtime import Classic7RuntimeFactory, classic7_full_game_replay_executor
    from werewolf.observer_dtrs.final_tom_client import CapsuleGameplayPredictor, REPORT_EVENT_PILOT_SCOPE
    from werewolf.observer_dtrs.speech_runtime import ReportEventArm, ReportEventGamePilot

    validate_collection_plan(plan)
    dest = safe_path(destination)
    arm_order = tuple(ReportEventArm)
    game_count = len(plan.ordered_seed_pool)
    if (type(runtime_factory) is not Classic7RuntimeFactory or runtime_factory.f0_control is not None
            or type(predictor) is not CapsuleGameplayPredictor or type(assignment_seed) is not int
            or assignment_seed < 0 or not len(arm_order) <= game_count <= predictor.request_limit
            or plan.target_canonical_success_count != game_count
            or dest.exists()):
        raise ValueError('需要新的 Canonical 目录、研究专用预测器及覆盖三臂且符合 Q 请求预算的固定种子计划')
    prior = load_collection_plan(safe_path(reference_collection))
    if (prior.collection_id != 'paper-canonical-qwen35-9b-5800-v1'
            or prior.plan_digest != 'ccc63353fe626ba5b76b06174a94272874a650f9a12b404de7c98c646b1e6bbe'
            or set(prior.ordered_seed_pool).intersection(plan.ordered_seed_pool)
            or plan.collection_id == prior.collection_id):
        raise ValueError('试运行须排除 5800 局 Development/Test 的计划种子')
    predictor.require_report_event_ready()
    m1_parameters = study.open_fitted_models()['models']['M1']
    shuffled_seeds = list(plan.ordered_seed_pool)
    Random(assignment_seed).shuffle(shuffled_seeds)
    arms = {seed: arm_order[index % len(arm_order)] for index, seed in enumerate(shuffled_seeds)}
    arm_counts = Counter(arms.values())
    # 随机排列令每个种子对各臂可交换，边际分配概率为该臂的名额比例。
    probabilities = {arm: count / game_count for arm, count in arm_counts.items()}
    write_once(dest / 'assignments.json', {
        'scope': REPORT_EVENT_PILOT_SCOPE, 'plan_digest': plan.plan_digest,
        'reference_plan_digest': prior.plan_digest, 'assignment_seed': assignment_seed,
        'predictor_identity': predictor.identity, 'm1_manifest_digest': study.FIT_DIGEST,
        'allocations': [{'seed': seed, 'arm': arms[seed].value, 'probability': probabilities[arms[seed]]}
                        for seed in plan.ordered_seed_pool],
    })

    def with_pilot(*, plan, claim):
        runtime = runtime_factory(plan=plan, claim=claim)
        evidence = dest / 'speech-evidence' / runtime.recorder.game_id

        def record(stage, row):
            write_once(evidence / f'{stage}.json', row)

        arm = arms[claim.seed]
        runtime.online_pilot = ReportEventGamePilot(
            recorder=runtime.recorder, agents=runtime.agents, predictor=predictor, arm=arm,
            probability=probabilities[arm], m1_parameters=m1_parameters, record_evidence=record,
        )
        runtime.replay_executor = classic7_full_game_replay_executor(runtime.recorder.runtime_configuration)
        return runtime

    try:
        return collect(plan=plan, runtime_factory=with_pilot, destination=dest)
    except CollectionSeedPoolExhausted:
        # 已分配种子用尽，失败保留在原 Attempt Ledger 中。
        return CollectionResult(dest, validate_attempt_ledger(dest / 'attempt_ledger', plan))


def verify_router(ledger, allocation, history, bundle):
    """Re-use the frozen lifecycle and canonical language proof validators."""
    from werewolf.interventions.online.server import ServerRuntimeFactory
    game_plan = collection_plan_from_record(ledger.plan["game_plan"])
    from werewolf.canonical_collection.attempt_ledger import _claim_from_record
    claim = _claim_from_record(history[0]["payload"]["claim"], game_plan)
    game_id = allocation["game_id"]
    policy = policy_for(ledger.plan, allocation)
    wolves = frozenset(p for p, r in bundle.private_replay_evidence.role_assignment if r == "Werewolf")
    first = next((pre for pre in bundle.authoritative_pre_prefixes
                  if pre.current_speaker in wolves and pre.current_speaker in pre.alive_observer_ids
                  and probe_candidate_pool(context_from_pre(pre, wolves))), None)
    audit = [r["payload"] for r in history if r["kind"] == "AUDIT"]
    records, calls, writeahead, assignment = [], [], {}, None
    full = {"public_events": [_raw_public_event(e) for e in bundle.public_event_stream.to_records()],
            "authoritative_pre_prefixes": [p.to_record() for p in bundle.authoritative_pre_prefixes],
            "backend_calls": [c.to_record() for c in bundle.backend_call_evidence]}
    previous = []
    for entry in audit:
        proof = validate_canonical_partial_evidence(safe_path(ledger.work / entry["proof"]["path"]),
            plan=game_plan, claim=claim, collection_directory=ledger.work)
        payload = proof.evidence.payload.to_value()
        if payload["game_id"] != game_id or payload["kind"] != entry["kind"] or proof.file_sha256 != entry["proof"]["sha256"]:
            raise ValueError("Router durable proof binding mismatch")
        record, canonical = payload["record"], payload["canonical_runtime"]
        events = canonical["public_events"]
        if events and bundle.public_event_stream.to_records()[:len(events)] != freeze_public_event_history(events).to_records():
            raise ValueError("Router stage is not a prefix of the full game")
        kind = entry["kind"]
        if kind == "game-start":
            if record != {"game_id": game_id, "policy": policy.to_record(), "policy_digest": policy.digest()}:
                raise ValueError("game policy differs from randomized allocation")
        elif kind == "assignment":
            if assignment is not None or first is None:
                raise ValueError("duplicate/illegal initial Router intervention")
            assigned = router_assignment_from_record(record)
            c = assigned.opportunity.legal_context
            if (assigned.policy != policy or c.boundary_id != first.boundary_id or
                    c.prefix_digest != first.prefix_digest or c.game_id != game_id or c.known_wolves != wolves):
                raise ValueError("Router did not use the first shared eligible PRE")
            assignment = record
        elif kind == "backend-call":
            if not records or records[-1]["lifecycle"][-1] not in ("T1_ATTEMPTED", "T3_PREPARED"):
                raise ValueError("language dispatch before durable assignment/stage")
            if record["pilot_id"] != policy.policy_id or record["sequence"] != len(calls) + 1:
                raise ValueError("Router backend policy/sequence mismatch")
            calls.append(record)
        elif kind.startswith("strategy-"):
            validate_probe_lifecycle_record(record)
            lifecycle = record["lifecycle"]
            if assignment is None or record["assignment"] != assignment or lifecycle != previous + [lifecycle[-1]]:
                raise ValueError("Router lifecycle is not append-only")
            if kind != f"strategy-{len(lifecycle):02d}-{lifecycle[-1].lower()}":
                raise ValueError("Router stage name mismatch")
            event = lifecycle[-1]
            if event in ("T1_ATTEMPTED", "T3_PREPARED"):
                writeahead["T1" if event == "T1_ATTEMPTED" else "T3"] = {
                    c["call_id"] for c in canonical["backend_calls"]}
            if record["continuation"]["opportunity"] is not None:
                ServerRuntimeFactory._probe_pre(game_id, record["continuation"]["opportunity"], canonical)
            for name, stage in record["stages"].items():
                if stage is not None:
                    ServerRuntimeFactory._verify_probe_stage(game_id, digest(assignment), stage, canonical,
                        invalid_at_pre=event == f"{name}_LANGUAGE_INVALID")
            if record["observations"]:
                prefix, public = ServerRuntimeFactory._probe_pre(game_id, assignment["opportunity"], canonical)
                actual = [e for e in public.to_records()[len(prefix.public_event_history.events) + 1:]
                          if e["event_type"] == "public_speech"]
                if any(o != {"event_id": e["event_id"], "event_type": "public_speech", "speaker": e["speaker"], "text": e["raw_text"]}
                       for o, e in zip(record["observations"], actual)) or len(actual) < len(record["observations"]):
                    raise ValueError("Router observation window differs from canonical game")
            records.append(record)
            previous = lifecycle
        else:
            raise ValueError("failed/unrecognized Router evidence cannot be published as success")
    if not audit or audit[0]["kind"] != "game-start" or sum(e["kind"] == "game-start" for e in audit) != 1:
        raise ValueError("missing unique Router game-start audit")
    canonical_ids = {c.call_id for c in bundle.backend_call_evidence if c.operation_id.startswith("phase2-")}
    if canonical_ids != {c["canonical_call_id"] for c in calls}:
        raise ValueError("unrecorded or missing Phase-2 canonical calls")
    if assignment is None:
        if first is not None or len(audit) != 1:
            raise ValueError("eligible game cannot be called NOT_APPLICABLE")
        return {"triggered": False, "t1_committed": False, "t3_committed": False}
    if not records or previous[-1] != "GAME_RESULT_RECORDED":
        raise ValueError("Router game has no complete policy lifecycle")
    final = records[-1]
    if final["offline_audit"]["final_game_result"] != bundle.private_replay_evidence.replay_inputs.to_value()["winner"]:
        raise ValueError("Router and full canonical winner differ")
    if calls != [c for stage in final["stages"].values() if stage is not None for c in stage["backend_calls"]]:
        raise ValueError("Router dispatch journal differs from final lifecycle")
    for name, stage in final["stages"].items():
        if stage is not None:
            if name not in writeahead or any(c["canonical_call_id"] in writeahead[name] for c in stage["backend_calls"]):
                raise ValueError("language call precedes stage writeahead")
            ServerRuntimeFactory._verify_probe_stage(game_id, digest(assignment), stage, full)
    return {"triggered": True, "t1_committed": "T1_COMMITTED" in previous,
            "t3_committed": "T3_COMMITTED" in previous}


def execute_games(ledger, *, runtime_factory, router_factory, run_game, replay_executor,
                  source_check, resume=False):
    """Single canonical loop; claim and allocation are durable before construction."""
    game_plan = collection_plan_from_record(ledger.plan["game_plan"])
    if inspect.signature(run_game).parameters["online_pilot"].kind is not inspect.Parameter.KEYWORD_ONLY:
        raise TypeError("production gameplay has no keyword-only Router hook")
    with ledger.lock():
        source_check()
        if resume:
            recover(ledger, replay_executor, verify_router)
        else:
            ledger.initialize()
        for allocation in ledger.plan["allocations"]:
            history = ledger.read()[1][allocation["game_id"]]
            if history:
                if history[-1]["kind"] != "RESULT":
                    raise ValueError("incomplete started game cannot continue campaign")
                continue
            source_check()
            claim = ledger.start(allocation)
            runtime = None
            try:
                runtime = runtime_factory(plan=game_plan, claim=claim)
                def sink(kind, record):
                    raw = record.to_record() if hasattr(record, "to_record") else record
                    seq = len(ledger.read()[0]) + 1
                    proof = construct_canonical_partial_evidence(plan=game_plan, claim=claim,
                        evidence_id=f"router-{seq:06d}", evidence_type="phase2_router_gameplay_stage_v1",
                        payload={"game_id": allocation["game_id"], "kind": kind, "record": raw,
                                 "canonical_runtime": runtime.recorder._failure_partial_payload()})
                    verified = publish_canonical_partial_evidence(ledger.work, plan=game_plan, claim=claim, evidence=proof)
                    ledger._append("AUDIT", {"game_id": allocation["game_id"], "kind": kind,
                        "proof": {"path": str(verified.path.relative_to(ledger.work)), "sha256": verified.file_sha256}})
                router = router_factory(policy_for(ledger.plan, allocation), runtime, sink)
                router.start_game(allocation["game_id"])
                run_game(runtime.env, runtime.agents, runtime.roles,
                    canonical_recorder=runtime.recorder, call_audit=runtime.call_audit, online_pilot=router)
                product = runtime.recorder.complete_evidence()
                publish_canonical_game_bundle(ledger.work / "games" / allocation["game_id"],
                    plan=game_plan, claim=claim, evidence=product, replay_executor=replay_executor)
                source_check()
                result = dict((a["game_id"], r) for a, r in verify_results(ledger, replay_executor, verify_router))[allocation["game_id"]]
                ledger._append("RESULT", result)
            except Exception as error:
                # A published full bundle can be indexed on explicit resume only.
                partial = runtime.recorder._failure_partial_payload() if runtime is not None else {}
                proof = construct_canonical_partial_evidence(plan=game_plan, claim=claim,
                    evidence_id="gameplay-failure", evidence_type="phase2_router_gameplay_failure_v1",
                    payload={"error_type": type(error).__name__, "error": str(error), "canonical_runtime": partial})
                verified = publish_canonical_partial_evidence(ledger.work, plan=game_plan, claim=claim, evidence=proof)
                ledger._append("FAILURE", {"game_id": allocation["game_id"], "error_type": type(error).__name__,
                    "error": str(error), "evidence_path": str(verified.path.relative_to(ledger.work)),
                    "evidence_sha256": verified.file_sha256})
                raise
        return ledger.status()


def verify_q_static(paths, bound):
    """Read pinned source/manifest evidence, without importing a ToM model."""
    from scripts.qwen3_gameplay_predictor import SOURCE_REVISION, FIT_DIGEST, SEAL_DIGEST, TERMINAL_DIGEST, MODEL_DIGEST
    checkout = paths["q_checkout"]
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    def git(*args):
        return subprocess.check_output(["git", "-C", str(checkout), *args], env=env, text=True).strip()
    if (safe_path(git("rev-parse", "--show-toplevel")) != safe_path(checkout) or git("rev-parse", "HEAD") != SOURCE_REVISION or
            git("status", "--porcelain=v1", "--untracked-files=all", "--ignore-submodules=none")):
        raise ValueError("static Q checkout identity/cleanliness mismatch")
    fit = verify_artifact(paths["q_fit"], manifest_name="qwen3_final_manifest.json",
        expected_artifact_type="qwen3_all_development_final", expected_schema_version="classic7_qwen3_all_development_final_v1")
    runs = fit.path.parent.parent / "runs" / fit.manifest_digest
    terminal = verify_artifact(runs / "terminal", expected_artifact_type="qwen3_final_terminal",
        expected_schema_version="classic7_qwen3_final_terminal_v1")
    seal = verify_artifact(runs / "seal", expected_artifact_type="qwen3_final_seal",
        expected_schema_version="classic7_qwen3_final_seal_v1")
    if (fit.manifest_digest != FIT_DIGEST or terminal.manifest_digest != TERMINAL_DIGEST or seal.manifest_digest != SEAL_DIGEST or
            (bound["q_fit_digest"], bound["q_seal_digest"]) != (FIT_DIGEST, SEAL_DIGEST) or
            terminal.manifest["model_digest"] != MODEL_DIGEST or seal.manifest["terminal_digest"] != TERMINAL_DIGEST or
            terminal.manifest["binding"]["fit_digest"] != FIT_DIGEST or seal.manifest["binding"] != terminal.manifest["binding"]):
        raise ValueError("static Q fit/terminal/seal linkage mismatch")


def verify_runtime_service(plan, paths):
    """Explicit runtime-only software/service checks; no language generation."""
    import yaml
    from scripts.collect_games import software_versions, live_preflight
    serve = yaml.safe_load(paths["deployment_config"].read_bytes())
    provenance = dict(collection_plan_from_record(plan["game_plan"]).environment_provenance)
    if any(provenance[key] != value for key, value in software_versions().items()):
        raise ValueError("qualified serving software provenance mismatch")
    live_preflight(f"http://{serve['host']}:{serve['port']}/v1", serve["served-model-name"])


def server_inputs(plan, repo):
    """Static source/artifact/config checks: no worker, GPU or service access."""
    import yaml
    from scripts.qwen3_gameplay_predictor import FIT_DIGEST, SEAL_DIGEST
    from scripts.collect_games import derive_seed_pool, plan_fields, model_manifest
    from werewolf.interventions.online.preflight import _smoke_v3_gate, qualification_inputs
    from werewolf.interventions.online.server import _verify_smoke_source
    from werewolf.interventions.mapper.runtime import load_runtime_mapper
    from werewolf.interventions.offline import build_development_layer
    from werewolf.interventions.outcome import reference_tables_digest
    from werewolf.runtime_config import normalize_runtime_config
    bound = plan["runtime_inputs"]
    if freeze_source(repo) != plan["source"] or (bound["q_fit_digest"], bound["q_seal_digest"]) != (FIT_DIGEST, SEAL_DIGEST):
        raise ValueError("gameplay source or Q pin mismatch")
    _, qualified, parent = qualification_inputs()
    if parent.manifest_digest != bound["qualified_runtime_manifest_digest"]:
        raise ValueError("qualified runtime provenance mismatch")
    for key in ("q_python", "q_fit_digest", "q_seal_digest", "mapper_manifest_digest", "reference_tables_digest",
                "smoke_v3_manifest_digest", "runtime_config_sha256", "deployment_config_sha256"):
        if bound[key] != qualified[key]:
            raise ValueError(f"qualified runtime pin drift: {key}")
    for key in ("runtime_config", "deployment_config", "publication", "evaluation_root", "mapper", "smoke_v3", "q_checkout", "q_fit"):
        if safe_path(bound["paths"][key]) != safe_path(qualified["paths"][key]):
            raise ValueError(f"qualified runtime path drift: {key}")
    expected_tom = tom_seed_exclusion(derive_seed_pool(TOM_GAMEPLAY_SEED_SOURCE, TOM_GAMEPLAY_SEED_COUNT))
    if bound["additional_seed_exclusions"] != expected_tom:
        raise ValueError("ToM planned seed exclusion mismatch")
    paths = {k: safe_path(v) for k, v in bound["paths"].items()}
    verify_q_static(paths, bound)
    config = normalize_runtime_config(yaml.safe_load(paths["runtime_config"].read_bytes()))
    serve_bytes = paths["deployment_config"].read_bytes()
    serve = yaml.safe_load(serve_bytes)
    if sha256_bytes(canonical_json_bytes(config)) != bound["runtime_config_sha256"] or sha256_bytes(serve_bytes) != bound["deployment_config_sha256"]:
        raise ValueError("runtime/deployment drift")
    base = f"http://{serve['host']}:{serve['port']}/v1"
    if (serve["host"] != "127.0.0.1" or serve["language-model-only"] is not True or
            serve["default-chat-template-kwargs"]["enable_thinking"] is not False or
            serve["max-num-seqs"] != 1 or serve["enforce-eager"] is not True or
            any(b["base_url"] != base or b["default_model"] != serve["served-model-name"] for b in config["backends"].values()) or
            config["parser"]["model"] != serve["served-model-name"] or
            any(p["model"] != serve["served-model-name"] for p in config["agent_config"]["all_candidates"])):
        raise ValueError("qualified loopback gameplay configuration mismatch")
    game = collection_plan_from_record(plan["game_plan"])
    if (game.ordered_seed_pool != derive_seed_pool(game.collection_id, plan["randomized_games"]) or
            not REQUIRED_PILOT_EXCLUSIONS <= {raw["collection_id"] for raw in plan["exclusions"]} or
            qualified["canonical_game_plan"] not in plan["exclusions"]):
        raise ValueError("gameplay plan or historical planned exclusions mismatch")
    provenance = dict(game.environment_provenance)
    limit = int(provenance["configured_call_limit"])
    campaign, extra = collection_plan_inputs(game.collection_id, plan["randomized_games"], limit, provenance)
    expected = plan_fields(campaign, plan["source"]["commit"], extra)
    if (provenance.get("runtime_config_sha256") != bound["runtime_config_sha256"] or
            provenance.get("serve_config_sha256") != bound["deployment_config_sha256"] or
            game.model_identity != serve["served-model-name"] or limit <= 0 or
            any((dict(game.environment_provenance) if k == "environment_provenance" else getattr(game, k)) != v for k, v in expected.items())):
        raise ValueError("production canonical plan provenance mismatch")
    model_digest, revision = model_manifest(safe_path(serve["model"]))
    if (provenance["model_manifest_sha256"] != model_digest or provenance["hf_revision"] != revision or
            provenance["model_path"] != serve["model"] or provenance["served_model_name"] != serve["served-model-name"]):
        raise ValueError("qualified serving model provenance mismatch")
    mapper = load_runtime_mapper(paths["mapper"], expected_manifest_digest=bound["mapper_manifest_digest"])
    reference = build_development_layer(paths["publication"], paths["evaluation_root"]).values
    if reference_tables_digest(reference) != bound["reference_tables_digest"]:
        raise ValueError("reference lineage mismatch")
    smoke = _smoke_v3_gate(paths["smoke_v3"], bound["smoke_v3_manifest_digest"], bound["mapper_manifest_digest"])
    if smoke is None:
        raise ValueError("Smoke-V3 gate failed")
    _verify_smoke_source(repo, smoke)
    return paths, config, limit, mapper, reference


def execute_server(plan, *, repo, work, destination, preflight_only=False, runtime_check_only=False, allow_gpu=False, resume=False,
                   predictor=None, public_pre=None):
    if predictor is not None or public_pre is not None:
        # 显式单 PRE 接口检查，先于游戏、语言、Ledger 构建；不冒充完整预验。
        from werewolf.interventions.online.preflight import assess_capsule_q_interface
        if (not runtime_check_only or preflight_only or allow_gpu or resume
                or Path(__file__).resolve().parents[2] != safe_path(repo)):
            raise ValueError('controlled capsule check requires runtime_check_only without gameplay/GPU/resume')
        return assess_capsule_q_interface(predictor, public_pre)
    if preflight_only and (runtime_check_only or allow_gpu):
        raise ValueError("static preflight cannot request GPU/runtime check")
    if not preflight_only and allow_gpu is not True:
        raise ValueError("runtime check/run/resume requires explicit GPU permission")
    bind_campaign_profile(plan, repo, work=work, destination=destination)
    from werewolf.canonical_collection.production_runtime import Classic7RuntimeFactory, classic7_full_game_replay_executor, run_game
    from werewolf.interventions.online.runner import OnlineTerminalPilotRunnerV1
    paths, config, limit, mapper, reference = server_inputs(plan, repo)
    if Path(__file__).resolve().parents[2] != safe_path(repo):
        raise ValueError("gameplay imports must originate in the pinned checkout")
    ledger = GameplayLedger(work, plan)
    replay = classic7_full_game_replay_executor(config)
    inspect.signature(run_game).bind(None, (), (), canonical_recorder=None, call_audit=None, online_pilot=None)
    if plan["purpose"] == "formal":
        from werewolf.interventions.gameplay import validate_publication
        admission = plan["qualification"]
        qplan = read_qualification_plan(admission["path"])
        qualified = validate_publication(admission["path"], admission["manifest_digest"], replay, verify_router,
                                         expected_plan=qplan)
        if (qualified.manifest["purpose"] != "qualification" or qualified.manifest["qualification_gate_passed"] is not True or
                qplan["game_plan"] not in plan["exclusions"] or
                not qualification_source_matches(qplan["source"], plan["source"])):
            raise ValueError("formal qualification/source admission mismatch")
    if resume:
        durable = {a["game_id"] for a, _ in verify_results(ledger, replay, verify_router)}
        histories = ledger.read()[1]
        if (preflight_only or runtime_check_only) and any(g not in durable or histories[g][-1]["kind"] == "INTERRUPTED"
                                                       for g in ledger.status()["unresolved_games"]):
            raise ValueError("started game lacks full endpoint; resume blocked")
        if not preflight_only and not runtime_check_only:
            with ledger.lock():
                recover(ledger, replay, verify_router)
    elif os.path.lexists(work):
        raise FileExistsError("existing work requires explicit resume")
    if os.path.lexists(destination):
        raise FileExistsError("immutable destination already exists; validate it instead")
    if preflight_only:
        return {"status": "STATIC_PREFLIGHT_PASSED", "static_preflight_passed": True,
                "runtime_check_passed": None, "q_worker_started": False, "gameplay_run": False,
                "source": plan["source"], "allocations": len(plan["allocations"])}
    from scripts.qwen3_gameplay_predictor import Qwen3GameplayPredictorClient
    from werewolf.backends.factory import load_named_backends
    verify_runtime_service(plan, paths)
    with ExitStack() as stack:
        predictor = stack.enter_context(Qwen3GameplayPredictorClient(checkout=paths["q_checkout"],
            fit_path=paths["q_fit"], python_executable=plan["runtime_inputs"]["q_python"]))
        backends = load_named_backends(config, env_file=None, max_retries=0)
        for backend in backends.values():
            stack.callback(backend.client.close)
        if runtime_check_only:
            if freeze_source(repo) != plan["source"]:
                raise ValueError("runtime check source drift")
            return {"status": "RUNTIME_CHECK_PASSED", "static_preflight_passed": True,
                    "runtime_check_passed": True, "q_worker_started": True, "gameplay_run": False,
                    "inference_run": False, "source": plan["source"]}
        factory = Classic7RuntimeFactory(runtime_config=config, backends=backends, configured_call_limit=limit)
        def router_factory(policy, runtime, sink):
            return OnlineTerminalPilotRunnerV1(plan=policy, predictor=predictor, mapper=mapper,
                backend=runtime.env.speech_perceiver.backend, model_name=config["parser"]["model"],
                reference_tables=reference, reference_artifact_digest=plan["runtime_inputs"]["reference_tables_digest"],
                record_evidence=sink)
        def source_check():
            if freeze_source(repo) != plan["source"]:
                raise ValueError("execution source drift; no further gameplay/publication")
        execute_games(ledger, runtime_factory=factory, router_factory=router_factory,
                      run_game=run_game, replay_executor=replay, source_check=source_check, resume=resume)
        source_check()
        artifact = publish_campaign(ledger, destination, replay, verify_router)
        return {"status": "COMPLETE", "manifest_digest": artifact.manifest_digest}
