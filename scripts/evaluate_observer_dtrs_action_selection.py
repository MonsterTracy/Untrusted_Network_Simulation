"""三臂离线配对、候选依据诊断及正式 ToM+3WD 在线执行。"""

import argparse
from collections import Counter
from dataclasses import asdict
from functools import partial
from itertools import combinations
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from scripts import analyze_observer_dtrs_report_state_risk as risk_study  # noqa: E402
from scripts import analyze_observer_dtrs_speech_candidates as generation  # noqa: E402
from scripts import evaluate_observer_dtrs_speech_consequences as replay  # noqa: E402
from scripts import fit_observer_dtrs_cognitive_value as cognitive_study  # noqa: E402
from scripts import fit_observer_dtrs_report_state as probability_study  # noqa: E402
from werewolf.agents.llm_agent import GameplayActionValidationError  # noqa: E402
from werewolf.artifact_io import canonical_json_bytes, canonical_jsonl_bytes, open_artifact_envelope, publish_artifact, sha256_bytes  # noqa: E402
from werewolf.backends import BackendError  # noqa: E402
from werewolf.backends.openai_compatible import model_request_context  # noqa: E402
from werewolf.observer_dtrs import action_selection, cognitive_value, report_state_decision, report_state_probability  # noqa: E402
from werewolf.observer_dtrs import speech_consequences  # noqa: E402


ARTIFACT_TYPE = "observer_dtrs_action_selection"
DESTINATIONS = {"compare": replay.frozen.DATA / "dtrs" / "speech-action-selection",
                "diagnose": replay.frozen.DATA / "dtrs" / "speech-basis-diagnostic"}


def verify_once(batch, team, agent):
    pre = batch.pre
    requests = []
    try:
        with model_request_context(requests, step=pre.step_index, seat=int(pre.current_speaker.removeprefix("player")),
                                   method="speech_candidate_bases", phase=pre.public_temporal_state.phase.value):
            verdict = agent.verify_speech_candidate_bases(batch, known_wolf_ids=team)
    except (BackendError, GameplayActionValidationError) as error:
        verdict = action_selection.undetermined_bases(batch, str(error))
        verdict["error"] = generation.error_detail(error)
    if requests:
        fields = ("status", "response_received", "finish_reason", "sdk_retries", "input_tokens", "output_tokens")
        verdict["request"] = {name: requests[0][name] for name in fields}
    return verdict


def compare_pre(batch, team, q, verdict, *, model, rule, cognitive, perceiver, predictor, delta):
    panel = report_state_probability.predict_pre(batch.pre, team, q, model)
    social = speech_consequences.score_candidates(batch, team, q, model, perceiver=perceiver, predictor=predictor)
    information = cognitive.predict_candidates(batch, team, q)
    return action_selection.compare_actions(batch, panel, rule, verdict, social, information, delta=delta)


def evaluate_records(generated, inputs, agent, *, compare=None):
    rows = []
    for row in generated:
        identity = {key: row[key] for key in (*action_selection.PRE_FIELDS, "attempt_id", "fold", "position", "alive_wolves")}
        if row["status"] != "ok":
            rows.append({**identity, "status": row["status"], "error": row["error"], "candidate_count": 0, "candidates": []})
            continue
        batch, team, q = replay.bind_archived_candidates(row, inputs)
        if compare is not None and not action_selection.has_declarations(batch):
            rows.append({**identity, "status": "declarations_missing", "candidate_count": len(batch.candidates),
                         "candidates": [asdict(candidate) for candidate in batch.candidates]})
            continue
        verdict = verify_once(batch, team, agent)
        if compare is None:
            rows.append({**identity, "status": "basis_diagnostic", "topic": batch.topic,
                         "candidate_count": len(batch.candidates), "candidates": [asdict(c) for c in batch.candidates],
                         "candidate_basis_verification": verdict})
        else:
            rows.append({**identity, **compare(batch, team, q, verdict)})
    return rows


def summarize(rows):
    verdicts = [basis for row in rows for basis in row.get("candidate_basis_verification", {}).get("candidates", ())]
    paired = [row for row in rows if "arms" in row]
    counts = Counter(basis["status"] for basis in verdicts)
    result = {"pres": len(rows), "shared_candidates": sum(row["candidate_count"] for row in rows),
              "pre_statuses": dict(Counter(row["status"] for row in rows)), "audited_candidates": len(verdicts),
              "determinate_candidates": len(verdicts) - counts["undetermined"],
              "basis_statuses": dict(counts), "undetermined_rate": replay.ratio(counts["undetermined"], len(verdicts)),
              "paired_pres": len(paired), "arms": {}, "paired_selection_differences": {}}
    for name in action_selection.ARMS:
        outcomes = Counter(row["arms"][name]["status"] for row in paired)
        result["arms"][name] = {"outcomes": dict(outcomes),
            "feasible_candidates": sum(row["arms"][name]["feasible_count"] for row in paired),
            "no_solution_rate": replay.ratio(len(paired) - outcomes["selected"], len(paired)),
            "no_certified_candidate_rate": replay.ratio(outcomes["no_certified_feasible_candidate"], len(paired))}
    for a, b in combinations(action_selection.ARMS, 2):
        key = f"{a}/{b}"
        both = [row["selection_differences"][key] for row in paired if row["selection_differences"][key] is not None]
        result["paired_selection_differences"][key] = {"joint_selected_pres": len(both), "different_pres": sum(both),
                                                       "rate": replay.ratio(sum(both), len(both))}
    return result


def _decision_models(args, publication):
    model, source = replay.load_probability_model(args.probability_artifact, publication)
    rule = report_state_decision.load_shared_rule(args.risk_artifact,
                                                  probability_manifest_digest=source["probability_manifest_digest"])
    active = cognitive_value.load_predictor(args.cognitive_artifact, args.probability_artifact)
    archive = open_artifact_envelope(args.cognitive_artifact, expected_artifact_type=cognitive_value.ARTIFACT_TYPE,
                                     expected_schema_version=cognitive_value.ARTIFACT_TYPE)
    torch.set_num_threads(1)
    predictor = speech_consequences.SpeechConsequencePredictor(args.capsule, expected_manifest_digest=args.capsule_digest,
                                                               expected_scope="PINNED_FINAL_INFERENCE_UNQUALIFIED", device="cpu")
    return {"model": model, "rule": rule, "cognitive": active, "predictor": predictor, "delta": args.delta}, {
        "probability_source": source, "risk_manifest_digest": rule.rule_manifest_digest,
        "cognitive_manifest_digest": archive.manifest_digest, "final_tom": predictor.identity, "delta_exact": str(args.delta),
    }


def _comparison(args, publication):
    components, sources = _decision_models(args, publication)
    perceiver, parser_identity = replay.load_perceiver(args.runtime_config)
    return partial(compare_pre, perceiver=perceiver, **components), {**sources, "candidate_parser": parser_identity}


def _frozen_publication():
    frozen = replay.frozen
    publication = open_artifact_envelope(frozen.DEVELOPMENT, expected_artifact_type=frozen.PUBLICATION_TYPE,
                                         expected_schema_version=frozen.PUBLICATION_VERSION)
    if publication.manifest_digest != frozen.PUBLICATION_DIGEST or publication.manifest["split_manifest_digest"] != frozen.SPLIT_DIGEST:
        raise ValueError("冻结 Development 或 Split 不匹配")
    return publication


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    compare = commands.add_parser("compare", help="评价有原始声明的新候选")
    diagnose = commands.add_parser("diagnose", help="只做既有候选依据诊断")
    online = commands.add_parser("online", help="以正式 ToM+3WD 执行固定种子的 Classic7 运行")
    for command in (compare, diagnose):
        command.add_argument("--candidate-jsonl", type=Path, required=True)
        command.add_argument("--runtime-config", type=Path, required=True)
    for command in (compare, online):
        command.add_argument("--delta", type=action_selection.parse_delta, required=True)
        command.add_argument("--capsule", type=Path, required=True)
        command.add_argument("--capsule-digest", required=True)
        command.add_argument("--probability-artifact", type=Path, default=probability_study.DESTINATION)
        command.add_argument("--risk-artifact", type=Path, default=risk_study.DESTINATION)
        command.add_argument("--cognitive-artifact", type=Path, default=cognitive_study.DESTINATION)
    online.add_argument("--campaign", type=Path, required=True, help="使用现有 Classic7 campaign JSON")
    online.add_argument("--storage-profile", type=Path, help="默认使用现有服务器科研数据目录")
    args = parser.parse_args(argv)

    if args.mode == "online":
        from scripts import collect_games
        from werewolf.observer_dtrs.speech_runtime import ReportEventArm, ThreeWaySpeechPolicy

        components, sources = _decision_models(args, _frozen_publication())
        method = {"method": ReportEventArm.THREE_WAY.value, "thresholds": components["rule"].thresholds.to_record(), **sources}
        return collect_games.run_campaign(args.campaign, storage_profile=args.storage_profile,
                                          online_policy=(partial(ThreeWaySpeechPolicy, **components), method))

    destination = DESTINATIONS[args.mode]
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"已有候选行动评价产物：{destination}")
    plan, candidate_settings, generated, candidate_digest = replay.read_candidate_study(args.candidate_jsonl)
    publication = _frozen_publication()
    inputs = replay.load_candidate_inputs(plan, publication)
    agent, agent_settings = generation.load_agent(args.runtime_config)
    selector, models = _comparison(args, publication) if args.mode == "compare" else (None, {})
    rows = evaluate_records(generated, inputs, agent, compare=selector)

    sources = {**plan["sources"], "candidate_jsonl_sha256": candidate_digest, **models}
    verification_settings = {key: agent_settings[key] for key in ("runtime_config_sha256", "backend", "model", "temperature",
                                                                  "max_tokens", "max_retries", "chat_template_kwargs")}
    verification_settings.update(prompt_prefix=action_selection.BASIS_VERIFICATION_PROMPT, response_schema="speech_candidate_bases")
    results = {"scope": "fixed_pre_paired_prediction" if selector is not None else "retrospective_basis_diagnostic",
               "sources": sources, "generation_settings": candidate_settings, "basis_verification_settings": verification_settings,
               **summarize(rows)}
    archive = publish_artifact(destination, manifest_fields={
        "artifact_type": ARTIFACT_TYPE, "schema_version": ARTIFACT_TYPE, "artifact_identity": destination.name,
        "mode": args.mode, **sources,
        "source_sha256": {str(path.relative_to(ROOT)): sha256_bytes(path.read_bytes()) for path in (
            Path(__file__).resolve(), Path(action_selection.__file__).resolve(),
            ROOT / "werewolf/observer_dtrs/speech_candidates.py", ROOT / "werewolf/agents/gpt_agent.py")},
    }, files={"results.json": canonical_json_bytes(results), "records.jsonl": canonical_jsonl_bytes(rows)})
    print(json.dumps({"path": str(destination), "manifest_digest": archive.manifest_digest, "pres": len(rows)}), flush=True)


if __name__ == "__main__":
    sys.exit(main())
