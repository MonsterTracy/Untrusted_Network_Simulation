"""Game-randomized Router campaigns; full canonical winners are the endpoint.

This ledger is deliberately independent of the day-consequence Pilot ledger.
No model, policy selection, replacement game or missing-outcome estimator lives
here. All allocations are published before a runtime can be constructed.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess

from werewolf.artifact_io import (
    canonical_json_bytes, canonical_jsonl_bytes, publish_artifact,
    read_artifact_file, sha256_bytes, verify_artifact,
)
from werewolf.artifact_io.canonical import (
    _load_canonical_json, _fsync_directory, ensure_durable_directory,
    _reject_duplicate_keys, _reject_json_constant,
)
from werewolf.canonical_collection.attempt_ledger import (
    _claim_from_record, _publish_bytes_noreplace, collection_plan_from_record,
    construct_attempt_claim,
)
from werewolf.canonical_collection.game_bundle import CANONICAL_GAME_BUNDLE_SCHEMA_VERSION, validate_canonical_game_bundle
from werewolf.phase2_online_plan import Phase2RouterPolicyV1, ProbeStrategy


VERSION = "phase2_router_gameplay_v1"
NAMES = {p: f"paper-phase2-router-gameplay-{p}-v1" for p in ("qualification", "formal")}
ALLOCATION_RULE = "sha256-ranked-complete-balanced-game-allocation-v1"


def digest(value):
    return sha256_bytes(canonical_json_bytes(value))


def derived_seed(identity, domain):
    return int.from_bytes(bytes.fromhex(sha256_bytes(f"{identity}:{domain}".encode()))[:8], "big") & ((1 << 63) - 1)


def safe_path(path):
    path = Path(path).absolute()
    if ".." in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("campaign paths must not contain symlinks or parent traversal")
    return path


def validate_profile(profile, *, ready=False):
    fields = {"schema_version", "purpose", "campaign_id", "randomized_games", "planning_status",
              "qualification", "qualification_gate", "paths", "excluded_game_plans"}
    if (not isinstance(profile, dict) or set(profile) != fields or profile["schema_version"] != VERSION or
            profile["purpose"] not in NAMES or profile["campaign_id"] != NAMES[profile["purpose"]] or
            profile["planning_status"] not in ("TENTATIVE", "UNFROZEN", "FROZEN")):
        raise ValueError("invalid gameplay profile")
    n = profile["randomized_games"]
    if ready and (type(n) is not int or n < 2 or n % 2 or profile["planning_status"] != "FROZEN"):
        raise ValueError("campaign budget/sample size is unfrozen; no preparation")
    paths = profile["paths"]
    if not isinstance(paths, dict) or set(paths) != {"plan", "work", "destination"}:
        raise ValueError("profile requires exact plan/work/destination paths")
    if ready and any(value is None for value in paths.values()):
        raise ValueError("campaign paths are unverified/unfrozen; no preparation")
    actual = [safe_path(value) for value in paths.values() if value is not None]
    if any(str(value) != raw for raw, value in zip([p for p in paths.values() if p is not None], actual)):
        raise ValueError("profile paths must be canonical absolute paths")
    if any(a == b or a.is_relative_to(b) or b.is_relative_to(a) for i, a in enumerate(actual) for b in actual[i + 1:]):
        raise ValueError("profile plan/work/destination paths must be disjoint")
    pins = profile["excluded_game_plans"]
    if ready and pins is None:
        raise ValueError("historical exclusion pointers are unverified/unfrozen; no preparation")
    if pins is not None:
        if not isinstance(pins, list) or any(not isinstance(p, dict) or set(p) != {"path", "plan_digest"} or
                not isinstance(p["path"], str) or str(safe_path(p["path"])) != p["path"] or
                not isinstance(p["plan_digest"], str) or not re.fullmatch(r"[0-9a-f]{64}", p["plan_digest"]) for p in pins):
            raise ValueError("invalid historical canonical plan pointers")
        if len({p["path"] for p in pins}) != len(pins) or len({p["plan_digest"] for p in pins}) != len(pins):
            raise ValueError("duplicate historical plan pointers")
    return profile


def load_campaign_profile(repo, purpose):
    path = safe_path(repo) / f"configs/phase2/router-gameplay-{purpose}-v1.json"
    profile = validate_profile(json.loads(safe_path(path).read_bytes(), object_pairs_hook=_reject_duplicate_keys,
                                        parse_constant=_reject_json_constant))
    if profile["purpose"] != purpose or (purpose == "qualification" and (
            profile["randomized_games"] != 20 or profile["qualification_gate"] != "full-games-and-both-mechanisms-v1")):
        raise ValueError("canonical qualification requires 20 games and the frozen mechanism gate")
    return profile


def bind_campaign_profile(plan, repo, *, work, destination, prepared_path=None):
    """One source-frozen path tuple per campaign, shared by prepare and execution."""
    profile = validate_profile(load_campaign_profile(repo, plan["purpose"]), ready=True)
    if plan["campaign_profile"] != profile:
        raise ValueError("run plan differs from the source-frozen campaign profile")
    paths = profile["paths"]
    if (safe_path(work) != safe_path(paths["work"]) or safe_path(destination) != safe_path(paths["destination"]) or
            prepared_path is not None and safe_path(prepared_path) != safe_path(paths["plan"])):
        raise ValueError("campaign cannot change its frozen plan/work/destination paths")
    return profile


def write_once(path, value):
    path = safe_path(path)
    ensure_durable_directory(path.parent)
    staging = path.parent / ".staging"
    ensure_durable_directory(staging)
    _publish_bytes_noreplace(staging_directory=staging, final_path=path,
                            durable_directory=path.parent, data=canonical_json_bytes(value))


def freeze_source(repo):
    """Freeze actual tracked production bytes, before any gameplay/backend call."""
    repo = safe_path(repo)
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args]).decode().strip()
    if git("branch", "--show-current") != "twd/mainline" or git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("gameplay requires clean tracked source/index on twd/mainline")
    required = ("werewolf/phase2_gameplay.py", "werewolf/phase2_gameplay_server.py",
                "scripts/phase2_gameplay_campaign.py", "docs/research/phase2-router-v1-contract.md",
                "configs/phase2/router-gameplay-qualification-v1.json", "configs/phase2/router-gameplay-formal-v1.json")
    names = git("ls-files").splitlines()
    if not set(required) <= set(names):
        raise ValueError("gameplay and Router source must be committed before preparation")
    selected = [n for n in names if n == "run_random.py" or
                n.startswith(("werewolf/", "scripts/", "configs/")) and n.endswith((".py", ".json", ".yaml")) or n in required]
    return {"commit": git("rev-parse", "HEAD"), "branch": "twd/mainline",
            "tracked_worktree_clean": True, "staged_tracked_changes": False,
            "source_sha256": {n: sha256_bytes((repo / n).read_bytes()) for n in selected}}


def allocations(game_plan, candidate_seed):
    n = len(game_plan.ordered_seed_pool)
    seed = derived_seed(game_plan.collection_id, "allocation")
    order = sorted(range(n), key=lambda i: (digest([ALLOCATION_RULE, seed, i]), i))
    probe = set(order[:n // 2])
    return [{"ordinal": i, "seed": s,
             "game_id": f"{game_plan.collection_id}-game-{i:06d}-seed-{s}",
             "assigned_policy": (ProbeStrategy.PROBE_THEN_REDIRECT.value if i in probe
                                 else ProbeStrategy.IMMEDIATE_REDIRECT.value),
             "allocation_probability": 0.5,
             "candidate_seed": candidate_seed}
            for i, s in enumerate(game_plan.ordered_seed_pool)]


def make_plan(*, purpose, game_plan, source, runtime_inputs, exclusions, campaign_profile,
              qualification=None, qualification_gate=None):
    n = len(game_plan.ordered_seed_pool)
    if purpose not in NAMES or game_plan.collection_id != NAMES[purpose] or n < 2 or n % 2:
        raise ValueError("explicit even game count and independent gameplay identity required")
    if game_plan.source_revision != source["commit"] or game_plan.target_canonical_success_count != n:
        raise ValueError("game plan/source/count mismatch")
    candidate_seed = derived_seed(game_plan.collection_id, "candidate")
    plan = {"schema_version": VERSION, "campaign_id": game_plan.collection_id,
            "purpose": purpose, "randomized_games": n, "allocation_rule": ALLOCATION_RULE,
            "allocation_seed": derived_seed(game_plan.collection_id, "allocation"),
            "candidate_sampling_id": game_plan.collection_id + ":candidate",
            "candidate_seed": candidate_seed, "source": source,
            "game_plan": game_plan.to_record(), "runtime_inputs": runtime_inputs,
            "exclusions": exclusions, "qualification": qualification,
            "campaign_profile": campaign_profile,
            "qualification_gate": qualification_gate,
            "allocations": allocations(game_plan, candidate_seed)}
    plan["plan_digest"] = digest(plan)
    return validate_plan(plan)


def validate_plan(plan):
    fields = {"schema_version", "campaign_id", "purpose", "randomized_games", "allocation_rule",
              "allocation_seed", "candidate_sampling_id", "candidate_seed", "source", "game_plan",
              "runtime_inputs", "exclusions", "qualification", "qualification_gate", "campaign_profile", "allocations", "plan_digest"}
    if set(plan) != fields or plan["schema_version"] != VERSION or plan["purpose"] not in NAMES:
        raise ValueError("invalid gameplay plan schema")
    if plan["plan_digest"] != digest({k: v for k, v in plan.items() if k != "plan_digest"}):
        raise ValueError("gameplay plan digest mismatch")
    profile = validate_profile(plan["campaign_profile"], ready=True)
    if any(profile[key] != plan[key] for key in ("purpose", "campaign_id", "randomized_games", "qualification", "qualification_gate")):
        raise ValueError("campaign profile/plan binding mismatch")
    if not {p["plan_digest"] for p in profile["excluded_game_plans"]} <= {p["plan_digest"] for p in plan["exclusions"]}:
        raise ValueError("frozen historical plan exclusion missing")
    game = collection_plan_from_record(plan["game_plan"])
    n = plan["randomized_games"]
    source = plan["source"]
    if (type(n) is not int or n < 2 or n % 2 or len(game.ordered_seed_pool) != n or
            game.target_canonical_success_count != n or game.collection_id != NAMES[plan["purpose"]] or
            plan["campaign_id"] != game.collection_id or game.source_revision != source["commit"] or
            source["branch"] != "twd/mainline" or source["tracked_worktree_clean"] is not True or
            source["staged_tracked_changes"] is not False or not source["source_sha256"] or
            not re.fullmatch(r"[0-9a-f]{40}", source["commit"])):
        raise ValueError("gameplay source/population binding mismatch")
    if (plan["allocation_rule"] != ALLOCATION_RULE or
            plan["allocation_seed"] != derived_seed(game.collection_id, "allocation") or
            plan["candidate_seed"] != derived_seed(game.collection_id, "candidate") or
            plan["candidate_sampling_id"] != game.collection_id + ":candidate" or
            plan["allocations"] != allocations(game, plan["candidate_seed"])):
        raise ValueError("game allocations do not reproduce")
    for raw in plan["exclusions"]:
        old = collection_plan_from_record(raw)
        if set(game.ordered_seed_pool) & set(old.ordered_seed_pool):
            raise ValueError("planned seed pool overlap; no reroll/replacement")
    extra_seeds = plan["runtime_inputs"].get("additional_seed_exclusions", {}).get("seeds", [])
    if set(game.ordered_seed_pool) & (set(extra_seeds) | set(range(900000001, 900000016))):
        raise ValueError("ToM/calibration seed overlap")
    if plan["qualification_gate"] not in (None, "full-games-and-both-mechanisms-v1"):
        raise ValueError("unreviewed qualification gate")
    if plan["purpose"] == "formal" and (plan["qualification"] is None or plan["qualification_gate"] is None):
        raise ValueError("formal preparation requires reviewed qualification admission")
    return plan


def policy_for(plan, allocation):
    return Phase2RouterPolicyV1(ProbeStrategy(allocation["assigned_policy"]),
        plan["candidate_sampling_id"], plan["candidate_seed"], plan["source"]["commit"])


class GameplayLedger:
    """Append-only durable game journal; reading never creates or repairs it."""
    def __init__(self, work, plan):
        self.work, self.plan = safe_path(work), validate_plan(plan)
        self.path = self.work / "game-ledger.jsonl"

    @contextmanager
    def lock(self):
        ensure_durable_directory(self.work)
        path = self.work / "campaign.lock"
        fd = os.open(safe_path(path), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield
        finally:
            os.close(fd)

    def initialize(self):
        if self.path.exists():
            raise FileExistsError("existing gameplay ledger requires resume")
        write_once(self.work / "run_inputs.json", self.plan)
        self._append("ALLOCATED", {"plan_digest": self.plan["plan_digest"],
                                  "allocations": self.plan["allocations"]}, initialize=True)

    def read(self):
        if _load_canonical_json(safe_path(self.work / "run_inputs.json").read_bytes()) != self.plan:
            raise ValueError("immutable run_inputs/source mismatch")
        data = safe_path(self.path).read_bytes()
        if not data or not data.endswith(b"\n"):
            raise ValueError("incomplete gameplay ledger")
        rows = [_load_canonical_json(line) for line in data.splitlines()]
        return self._validate_rows(rows)

    def _validate_rows(self, rows):
        previous = None
        games = {a["game_id"]: [] for a in self.plan["allocations"]}
        for seq, row in enumerate(rows, 1):
            if (set(row) != {"sequence", "previous_digest", "kind", "payload", "digest"} or
                    row["sequence"] != seq or row["previous_digest"] != previous or
                    row["digest"] != digest({k: v for k, v in row.items() if k != "digest"})):
                raise ValueError("game ledger hash chain mismatch")
            previous = row["digest"]
            if seq == 1:
                if row["kind"] != "ALLOCATED" or row["payload"] != {
                    "plan_digest": self.plan["plan_digest"], "allocations": self.plan["allocations"]}:
                    raise ValueError("game ledger allocation binding mismatch")
                continue
            payload, kind = row["payload"], row["kind"]
            game_id = payload.get("game_id")
            if game_id not in games:
                raise ValueError("unplanned game in ledger")
            history = games[game_id]
            if kind == "STARTED":
                if history or any(h and h[-1]["kind"] not in ("RESULT",) for h in games.values()):
                    raise ValueError("duplicate start or unresolved game; cannot continue")
                claim = _claim_from_record(payload["claim"], collection_plan_from_record(self.plan["game_plan"]))
                if claim.ordinal != next(a["ordinal"] for a in self.plan["allocations"] if a["game_id"] == game_id):
                    raise ValueError("canonical claim allocation mismatch")
                if game_id != next(g for g, h in games.items() if not h):
                    raise ValueError("games must start in the frozen plan order")
            elif not history or history[-1]["kind"] in ("RESULT", "INTERRUPTED"):
                raise ValueError("invalid game transition")
            elif kind == "AUDIT":
                proof = payload["proof"]
                if not re.fullmatch(r"attempts/[^/]+/partial_evidence/router-[0-9]+\.json", proof["path"]):
                    raise ValueError("invalid canonical proof path")
                if sha256_bytes(safe_path(self.work / proof["path"]).read_bytes()) != proof["sha256"]:
                    raise ValueError("canonical stage proof mismatch")
            elif kind == "RESULT":
                if payload["winner"] not in ("Werewolf", "Villager") or payload["bundle_path"] != "games/" + game_id:
                    raise ValueError("invalid final result")
            elif kind == "FAILURE":
                if not re.fullmatch(r"attempts/[^/]+/partial_evidence/gameplay-failure\.json", payload["evidence_path"]):
                    raise ValueError("invalid failure evidence path")
                if sha256_bytes(safe_path(self.work / payload["evidence_path"]).read_bytes()) != payload["evidence_sha256"]:
                    raise ValueError("failure evidence hash mismatch")
            elif kind not in ("FAILURE", "INTERRUPTED"):
                raise ValueError("unknown game ledger event")
            history.append(row)
        return rows, games

    def _append(self, kind, payload, *, initialize=False):
        rows = [] if initialize else self.read()[0]
        row = {"sequence": len(rows) + 1, "previous_digest": rows[-1]["digest"] if rows else None,
               "kind": kind, "payload": payload}
        row["digest"] = digest(row)
        self._validate_rows(rows + [row])
        fd = os.open(safe_path(self.path), os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW |
                     (os.O_CREAT | os.O_EXCL if initialize else 0), 0o600)
        try:
            block = canonical_json_bytes(row) + b"\n"
            while block:
                block = block[os.write(fd, block):]
            os.fsync(fd)
        finally:
            os.close(fd)
        if initialize:
            _fsync_directory(self.work)
        self.read()

    def start(self, allocation):
        if allocation not in self.plan["allocations"]:
            raise ValueError("unplanned allocation")
        game = collection_plan_from_record(self.plan["game_plan"])
        claim = construct_attempt_claim(game, ordinal=allocation["ordinal"],
            attempt_id=f"attempt-{allocation['ordinal']:06d}-seed-{allocation['seed']}",
            claim_timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"))
        self._append("STARTED", {"game_id": allocation["game_id"],
                                "claim": {"record_type": "claim", **claim.to_record()}})
        return claim

    def status(self):
        _, games = self.read()
        complete = sum(bool(h) and h[-1]["kind"] == "RESULT" for h in games.values())
        unresolved = [g for g, h in games.items() if h and h[-1]["kind"] != "RESULT"]
        return {"status": "INCOMPLETE" if unresolved else "COMPLETE" if complete == len(games) else "RUNNING",
                "randomized_games": len(games), "complete_games": complete,
                "unresolved_games": unresolved}


def verify_results(ledger, replay_executor, verify_router):
    """Validate every full bundle, including games with zero interventions."""
    _, games = ledger.read()
    game_plan = collection_plan_from_record(ledger.plan["game_plan"])
    results = []
    for allocation in ledger.plan["allocations"]:
        game_id = allocation["game_id"]
        history = games[game_id]
        if not history:
            continue
        claim = _claim_from_record(history[0]["payload"]["claim"], game_plan)
        result = history[-1]["payload"] if history[-1]["kind"] == "RESULT" else None
        path = ledger.work / "games" / game_id
        if not path.exists():
            if result is not None:
                raise ValueError("recorded winner has no complete canonical bundle")
            continue
        bundle = validate_canonical_game_bundle(safe_path(path), plan=game_plan,
                                                claim=claim, replay_executor=replay_executor)
        winner = bundle.private_replay_evidence.replay_inputs.to_value().get("winner")
        if winner not in ("Werewolf", "Villager"):
            raise ValueError("canonical bundle has no final winner")
        process = verify_router(ledger, allocation, history, bundle)
        expected = {"game_id": game_id, "winner": winner, "bundle_path": "games/" + game_id,
                    "bundle_digest": bundle.manifest_digest, "router_process": process}
        if result is not None and result != expected:
            raise ValueError("winner/index/canonical evidence mismatch")
        results.append((allocation, expected))
    return results


def recover(ledger, replay_executor, verify_router):
    """Only reconstruct an index from already durable evidence; never replay play."""
    verified = dict((a["game_id"], r) for a, r in verify_results(ledger, replay_executor, verify_router))
    _, games = ledger.read()
    for game_id, history in games.items():
        if not history or history[-1]["kind"] == "RESULT":
            continue
        if history[-1]["kind"] == "INTERRUPTED":
            raise ValueError("started game is permanently incomplete; no replay/replacement")
        if game_id in verified:
            ledger._append("RESULT", verified[game_id])
        else:
            ledger._append("INTERRUPTED", {"game_id": game_id, "reason": "STARTED_WITHOUT_COMPLETE_WINNER_EVIDENCE"})
            raise ValueError("missing whole-game endpoint; campaign INCOMPLETE")


def publish_campaign(ledger, destination, replay_executor, verify_router):
    verified = verify_results(ledger, replay_executor, verify_router)
    rows, _ = ledger.read()
    if ledger.status()["status"] != "COMPLETE" or len(verified) != ledger.plan["randomized_games"]:
        raise ValueError("all randomized games require full winners; cannot publish partial ITT")
    outcomes = [dict(a, **r, wolf_win=int(r["winner"] == "Werewolf")) for a, r in verified]
    gate = qualification_diagnostics(outcomes)
    files = {"run_plan.json": canonical_json_bytes(ledger.plan), "run_inputs.json": canonical_json_bytes(ledger.plan),
             "game_plan.json": canonical_json_bytes(ledger.plan["game_plan"]),
             "allocations.jsonl": canonical_jsonl_bytes(ledger.plan["allocations"]),
             "outcomes.jsonl": canonical_jsonl_bytes(outcomes), "game-ledger.jsonl": canonical_jsonl_bytes(rows),
             "qualification_diagnostics.json": canonical_json_bytes(gate)}
    for allocation, _ in verified:
        game_id = allocation["game_id"]
        artifact = verify_artifact(ledger.work / "games" / game_id,
            expected_artifact_type="canonical_game_bundle", expected_schema_version=CANONICAL_GAME_BUNDLE_SCHEMA_VERSION)
        for path in ["manifest.json", *artifact.manifest["file_table"]]:
            files["games/" + game_id + "/" + path] = (artifact.path / path).read_bytes()
    for row in rows:
        if row["kind"] == "AUDIT":
            path = row["payload"]["proof"]["path"]
            files[path] = (ledger.work / path).read_bytes()
        elif row["kind"] == "FAILURE":
            path = row["payload"]["evidence_path"]
            files[path] = (ledger.work / path).read_bytes()
    return publish_artifact(safe_path(destination), manifest_fields={
        "artifact_type": "phase2_router_gameplay", "schema_version": VERSION,
        "study_name": ledger.plan["campaign_id"], "purpose": ledger.plan["purpose"],
        "plan_digest": ledger.plan["plan_digest"], "source": ledger.plan["source"],
        "completion_status": "COMPLETE", "primary_population": "all_randomized_games",
        "primary_endpoint": "full_canonical_wolf_win", "randomized_games": len(outcomes),
        "ledger_digest": rows[-1]["digest"], "qualification_gate": ledger.plan["qualification_gate"],
        "qualification_gate_passed": all(gate.values()) if ledger.plan["qualification_gate"] else None}, files=files)


def qualification_diagnostics(outcomes):
    return {"complete_full_games": True,
            "immediate_redirect_committed": any(r["assigned_policy"] == ProbeStrategy.IMMEDIATE_REDIRECT.value
                and r["router_process"]["t1_committed"] for r in outcomes),
            "probe_then_redirect_committed": any(r["assigned_policy"] == ProbeStrategy.PROBE_THEN_REDIRECT.value
                and r["router_process"]["t3_committed"] for r in outcomes)}


def validate_publication(path, expected_digest, replay_executor, verify_router):
    artifact = verify_artifact(safe_path(path), expected_artifact_type="phase2_router_gameplay", expected_schema_version=VERSION)
    if artifact.manifest_digest != expected_digest:
        raise ValueError("publication digest mismatch")
    plan = validate_plan(_load_canonical_json(read_artifact_file(artifact, "run_plan.json")))
    # The publication includes a byte-identical immutable run_inputs copy.
    ledger = GameplayLedger(artifact.path, plan)
    verified = verify_results(ledger, replay_executor, verify_router)
    rows, _ = ledger.read()
    expected = canonical_jsonl_bytes(dict(a, **r, wolf_win=int(r["winner"] == "Werewolf")) for a, r in verified)
    diagnostics = qualification_diagnostics([dict(a, **r) for a, r in verified])
    m = artifact.manifest
    if (ledger.status()["status"] != "COMPLETE" or read_artifact_file(artifact, "outcomes.jsonl") != expected or
            read_artifact_file(artifact, "allocations.jsonl") != canonical_jsonl_bytes(plan["allocations"]) or
            _load_canonical_json(read_artifact_file(artifact, "game_plan.json")) != plan["game_plan"] or
            m["plan_digest"] != plan["plan_digest"] or m["source"] != plan["source"] or
            m["study_name"] != plan["campaign_id"] or m["purpose"] != plan["purpose"] or
            m["randomized_games"] != plan["randomized_games"] or m["ledger_digest"] != rows[-1]["digest"] or
            m["completion_status"] != "COMPLETE" or m["primary_population"] != "all_randomized_games" or
            m["primary_endpoint"] != "full_canonical_wolf_win" or
            _load_canonical_json(read_artifact_file(artifact, "qualification_diagnostics.json")) != diagnostics or
            m["qualification_gate"] != plan["qualification_gate"] or
            m["qualification_gate_passed"] != (all(diagnostics.values()) if plan["qualification_gate"] else None)):
        raise ValueError("publication population/endpoint/provenance mismatch")
    return artifact
