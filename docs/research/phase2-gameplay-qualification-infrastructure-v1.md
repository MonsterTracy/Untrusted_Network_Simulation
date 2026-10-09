# Router gameplay infrastructure V1

Status: local engineering implementation. No gameplay, qualification, formal
plan or result has been generated. Router policy and candidate semantics remain
those of `phase2-router-v1-contract.md`; this document adds evidence machinery.

The primary population is **every allocated game**, including games without a
Probe-eligible PRE, T1 language invalid and T3 cancelled. The primary endpoint
requires a complete canonical game bundle and its verified final Werewolf or
Villager winner. DAY_CONSEQUENCE / L_ref and Router execution are process
evidence; they cannot complete this campaign's endpoint. No statistical model,
missing-outcome imputation or outcome-dependent sampling is implemented.

## Plans and identities

- Qualification identity: `paper-phase2-router-gameplay-qualification-v1`.
- Formal identity: `paper-phase2-router-gameplay-formal-v1`.
- Qualification budget and mechanism gate are FROZEN: 20 randomized games,
  exactly 10 per arm, `full-games-and-both-mechanisms-v1`. This does not certify
  server admission. The qualification profile binds six historical canonical
  plan pointers and one reserved plan/work/destination tuple. The operator's
  server `collection_plan_from_record()` audit reported six production digests
  verified, 1320 planned/unique seeds and zero historical pairwise overlap.
  Those results are the pinning authority; no historical JSON was reconstructed
  on this Mac. New-qualification overlap and output-path availability remain
  server checks, not facts established by filling the profile.
  Formal N and qualification admission remain null / UNFROZEN and cannot prepare.
- Game seeds reuse the production `derive_seed_pool(campaign_id, N)` rule.
  There are exactly N planned games, no oversampling/replacement pool.
- Allocation seed is low63(first8_big_endian(SHA256(UTF8(id + ':allocation')))).
  Rank ordinals by SHA256 canonical JSON `[allocation_rule, seed, ordinal]`;
  assign the first N/2 to ProbeThenRedirect, the rest to ImmediateRedirect.
  Run in frozen ordinal order, yielding a mixed, balanced allocation. The
  probability .5 is game-policy allocation; Router treatment probability 1
  remains deterministic. This is unpaired complete randomization implemented
  by an audited pseudorandom rule, not independent per-game Bernoulli draws.
- Candidate seed uses the same derivation with domain `candidate`. Both arms
  share the candidate sampling identity. Game/PRE identity and prefix remain
  in the existing sampler; equal seed alone does not imply paired PREs or j.
- Source is frozen once during prepare and copied to the publication. Execution
  only checks for drift; it cannot replace provenance after the games finish.
  Tracked/index changes reject; unrelated untracked research is allowed. The
  new executable, core files, profiles and Router contract must be committed.

Prepare verifies the existing qualified Terminal runtime pins and actual
excluded canonical game plans. Exclude full planned pools for development,
Terminal qualification/formal, Probe V1/V2/V3/formal and the frozen 100-candidate
ToM gameplay pool, plus calibration seeds. No skip/reroll on overlap. Formal
also excludes this gameplay qualification's full pool and requires its verified
publication and reviewed mechanism gate. Its source commit may differ, but
runtime Python/Router contract bytes must match successful qualification.

Only the canonical tracked profile may prepare a campaign. After operator review,
freeze its three absolute, disjoint paths and `excluded_game_plans` list of exact
`{path, plan_digest}` pointers. CLI paths and exclusions must equal that profile;
the whole profile is included in the run-plan digest. Open-plan and server entry
independently bind the same profile/path tuple. A copied profile, relocated plan
or changed work/destination cannot restart the same frozen campaign. Existing
plan/work/publication paths reject preparation. An interrupted campaign cannot
be given replacement directories; a new engineering qualification requires a
separately reviewed identity and source. No new identity is defined here.

The reserved qualification paths are **new outputs**, not assertions that these
directories exist or have passed server checks:

```text
plan: /data/yuxiao/Untrusted_Network_Simulation/plans/phase2-router-gameplay-qualification-v1-plan
work: /data/yuxiao/Untrusted_Network_Simulation/paper-studies/phase2-router-gameplay-qualification-v1-work
destination: /data/yuxiao/Untrusted_Network_Simulation/paper-studies/paper-phase2-router-gameplay-qualification-v1
```

Historical pin inventory (all paths are under the authoritative `plans/`):

| File | Planned seeds | Production digest |
| --- | ---: | --- |
| paper-phase2-online-terminal-qualification-v1-canonical-game-plan.json | 40 | a6b575bc72bfab802be3019e740c4f98c9d5bb9254ca47a2d6c1571ea3be9661 |
| phase2-online-terminal-pilot-v1-game-plan.json | 240 | add3fe53054d9b89b6c6570b585bce13070f010dcf2e28a40a4671fcd713c64b |
| phase2-online-probe-qualification-v1-game-plan.json | 80 | 4719b85665febcb47479a594f13c00940a6636061ebe9410c4b68391be4bda6e |
| phase2-online-probe-qualification-v2-game-plan.json | 80 | ef8e2d0d239f3e0d98ef84ed44c51d3cbceb80b4ddacee5b52fff6a7ac0bd151 |
| phase2-online-probe-qualification-v3-game-plan.json | 80 | 97ecfabf670d05d799fb39819ca4e78275416d004cdb936eb455ec18f40a1aca |
| phase2-online-probe-pilot-v1-game-plan.json | 800 | 7f69c01059a350fdc0bef9f21febe70e36b67901f3e3e6200a9a05f6c3c3aa57 |

The new 20-seed pool uses the existing production
`derive_seed_pool(campaign_id, randomized_games)`. On the server, `prepare` reads
each real plan through `collection_plan_from_record()`, compares its production
digest to the profile and retains its complete `ordered_seed_pool`. Existing
`seed_overlaps` checks development/Terminal qualification/calibration; the
ToM 100-seed exclusion is unchanged. Before publishing the immutable plan,
`make_plan -> validate_plan` intersects the new seeds with every complete
historical pool and rejects any overlap. Static preflight rechecks the embedded
exclusions and their required runtime lineage. The explicitly pinned Terminal
qualification is also retained by the existing qualified-runtime binding; this
does not change the union or permit dropping seeds.

## Durable execution and recovery

The prepared immutable artifact contains run_plan, independent game_plan and
disjoint operator paths. At run start, immutable run_inputs and an fsynced
hash-chain ledger record the **entire allocation table before constructing a
game runtime**. A campaign-wide exclusive lock prevents concurrent execution.

`ALLOCATED -> STARTED -> AUDIT* -> full bundle publication/validation -> RESULT`

STARTED binds a real canonical attempt claim before runtime construction.
Router callbacks persist immutable canonical partial evidence; ledger rows bind
its file hash. The existing language/canonical validators verify dispatch,
writeahead stages, commitment, observation windows and full-history linkage.
All games use the existing `Classic7RuntimeFactory` and `run_random.eval`
original-mode loop. No second loop or ToM text-injection ablation is introduced.

Recovery is fail closed:

- Not started: retain allocation and use the same next planned game.
- Started with a verified durable full bundle but no RESULT: validate existing
  evidence and append only the result index. No gameplay or language is rerun.
- Started without full winner evidence: append INTERRUPTED; campaign remains
  INCOMPLETE. Do not replay, replace, rerandomize or start another game.
- Day consequence alone is insufficient. Language invalid is an audited policy
  branch; a structural/technical failure is never NOT_APPLICABLE.
- Hash/source/plan/input drift, symlink paths or malformed/truncated records
  reject. Exceptions retain canonical partial evidence. Hard process termination
  is detected from STARTED on explicit resume; status/preflight never repair it.
- Canonical deterministic **action replay validation** uses recorded actions
  without agent/backend calls. It is evidence verification, not re-execution of
  an assigned game to obtain a new outcome.

Publication includes every allocation and result, the journal, all complete
canonical bundle bytes and Router partial proofs. Validation needs no historical
work directory: it recomputes allocations, joins exactly N games, revalidates
bundles/winners and policy proofs, and rejects filtered or rehashed bad tables.
Immutable destinations cannot be overwritten. No partial ITT completion claim.

## CLI

After independent review/freeze and a clean source commit, operator preparation
uses an explicit exclusions JSON list, identical to the frozen profile, of `{path, plan_digest}` pointers to the
Terminal qualification/formal and Probe V1/V2/V3/formal canonical plans. Additional known
development/ToM/calibration exclusions are verified by
the preparation path. No digest is fabricated.

`SOURCE_COMMIT_READY` means the local source/config/contracts/tests can be
submitted as one reviewed revision, not that a clean commit was already created
or deployed. `SERVER_STATIC_PREFLIGHT_READY` requires the deployed source,
paths, exclusions and artifact bindings to pass the actual server check. Neither
status certifies GPU/runtime or qualification success.

After deploying the new clean revision, run this read-only check first. Replace
the source placeholder with that exact revision. The script writes no plan,
ledger or experimental artifact and starts no Q worker:

```bash
export PHASE2_SOURCE="<NEW_CLEAN_COMMIT>"
cd /data/yuxiao/Untrusted_Network_Simulation
PYTHONDONTWRITEBYTECODE=1 python - <<'PY'
import json, os
from types import SimpleNamespace
from scripts.collect_games import derive_seed_pool
from scripts.phase2_gameplay_campaign import ROOT, read_json, check_paths
from scripts.phase2_online_campaign import qualification_inputs, checked_seed_overlaps
from werewolf.canonical_collection.attempt_ledger import collection_plan_from_record
from werewolf.phase2_gameplay import freeze_source, load_campaign_profile, validate_profile, safe_path

source = freeze_source(ROOT)
if source["commit"] != os.environ["PHASE2_SOURCE"]:
    raise ValueError("deployed source differs from the explicit new clean commit")
p = validate_profile(load_campaign_profile(ROOT, "qualification"), ready=True)
old = [collection_plan_from_record(read_json(pin["path"])) for pin in p["excluded_game_plans"]]
if any(plan.plan_digest != pin["plan_digest"] for plan, pin in zip(old, p["excluded_game_plans"])):
    raise ValueError("historical production digest mismatch")
union = set().union(*(set(plan.ordered_seed_pool) for plan in old))
if len(old) != 6 or sum(len(plan.ordered_seed_pool) for plan in old) != 1320 or len(union) != 1320:
    raise ValueError("historical full-pool inventory differs from the reviewed evidence")
seeds = derive_seed_pool(p["campaign_id"], p["randomized_games"])
pins, bound, _ = qualification_inputs()
overlap = checked_seed_overlaps(SimpleNamespace(ordered_seed_pool=seeds), pins, bound, old)
overlap["tom_gameplay_100"] = len(set(seeds) & set(derive_seed_pool("paper-tom-gameplay-ablation-v1", 100)))
if any(overlap.values()):
    raise ValueError("seed overlap; no skip/reroll")
paths = check_paths(*(p["paths"][key] for key in ("plan", "work", "destination")))
if any(os.path.lexists(path) or not path.parent.is_dir() or not os.access(path.parent, os.W_OK | os.X_OK) for path in paths):
    raise ValueError("reserved outputs are occupied or parents are unavailable")
for directory in (safe_path(ROOT / "plans"), safe_path(ROOT / "paper-studies")):
    for pattern in ("*/run_plan.json", "*/run_inputs.json", "*/manifest.json"):
        for path in directory.glob(pattern):
            value = read_json(path)
            if p["campaign_id"] in (value.get("campaign_id"), value.get("study_name"), value.get("collection_id")):
                raise ValueError(f"existing campaign evidence; do not restart: {path}")
if freeze_source(ROOT) != source:
    raise ValueError("source changed during read-only audit")
print(json.dumps({"historical_seed_union": len(union), "new_seed_count": len(seeds),
    "actual_full_pool_overlap": overlap, "reserved_paths_checked": p["paths"],
    "server_static_preflight_passed": False}, sort_keys=True))
PY
```

After success, export the pinned list to a temporary CLI input. `prepare` then
writes a new immutable plan artifact, with no gameplay or assignment ledger;
`preflight` is read-only. No `--allow-gpu` is used:

```bash
PHASE2_ROOT=/data/yuxiao/Untrusted_Network_Simulation
PHASE2_PLAN="$PHASE2_ROOT/plans/phase2-router-gameplay-qualification-v1-plan"
PHASE2_WORK="$PHASE2_ROOT/paper-studies/phase2-router-gameplay-qualification-v1-work"
PHASE2_DEST="$PHASE2_ROOT/paper-studies/paper-phase2-router-gameplay-qualification-v1"
PHASE2_EXCLUSIONS=$(mktemp /tmp/phase2-router-gameplay-exclusions.XXXXXX)
PYTHONDONTWRITEBYTECODE=1 python - "$PHASE2_EXCLUSIONS" <<'PY'
import json, sys
from scripts.phase2_gameplay_campaign import ROOT
from werewolf.phase2_gameplay import load_campaign_profile
with open(sys.argv[1], "w") as file:
    json.dump(load_campaign_profile(ROOT, "qualification")["excluded_game_plans"], file)
PY
python -m scripts.phase2_gameplay_campaign prepare --purpose qualification \
  --source-commit "$PHASE2_SOURCE" --exclusions "$PHASE2_EXCLUSIONS" \
  --plan "$PHASE2_PLAN" --work-directory "$PHASE2_WORK" --destination "$PHASE2_DEST"
python -m scripts.phase2_gameplay_campaign preflight --purpose qualification \
  --plan "$PHASE2_PLAN" --plan-manifest-digest "<PRINTED_PLAN_MANIFEST_DIGEST>"
```

The following modes remain outside static admission and need separate runtime
authorization; these are command shapes, not instructions to start them now:

```bash
# Only later, with explicit permission to load models/GPU and access the service:
python -m scripts.phase2_gameplay_campaign runtime-check --allow-gpu --purpose qualification \
  --plan <new-plan-artifact> --plan-manifest-digest <printed-plan-manifest>
python -m scripts.phase2_gameplay_campaign run --allow-gpu --purpose qualification \
  --plan <new-plan-artifact> --plan-manifest-digest <printed-plan-manifest>
python -m scripts.phase2_gameplay_campaign preflight --resume-check \
  --plan <new-plan-artifact> --plan-manifest-digest <printed-plan-manifest>
python -m scripts.phase2_gameplay_campaign resume --allow-gpu \
  --plan <new-plan-artifact> --plan-manifest-digest <printed-plan-manifest>
python -m scripts.phase2_gameplay_campaign validate \
  --plan <new-plan-artifact> --plan-manifest-digest <printed-plan-manifest> \
  --publication-manifest-digest <printed-publication-manifest>
```

Static `preflight` validates source/config/Smoke/reference bindings, serving model
file hashes/HF revision and the pinned Q checkout/fit/terminal/seal manifest chain.
It does not start a Q worker, restore a tensor model, move anything to GPU,
construct backend clients or contact the inference service. Success is
`STATIC_PREFLIGHT_PASSED`, `runtime_check_passed=null`, `q_worker_started=false`.
No durable campaign work is created. Resume preflight checks existing full
evidence without modifying the ledger. `--resume-check` is accepted only by
`preflight`; `run --resume-check` rejects during argument validation.

Explicit `runtime-check --allow-gpu` additionally verifies serving software and
health/model endpoints, starts the unchanged sealed Q worker (including its
original fit/device/model checks) and loads backends from the **complete** runtime
config with `env_file=None, max_retries=0`. Success is `RUNTIME_CHECK_PASSED`,
`runtime_check_passed=true`, `q_worker_started=true`. This establishes startup
readiness; it does not call prediction, language generation or gameplay
(`inference_run=false`), and cannot substitute for mechanism qualification.
Both run and resume require `--allow-gpu` and repeat these checks. Missing full
endpoints reject before GPU startup; only actual resume can durably mark them
INTERRUPTED. `status` reads without running models. Static pass is not runtime pass.

The frozen qualification gate requires 20/20 verified complete canonical winners,
the reproducible 10/10 allocation, at least one Immediate Redirect canonical
success and at least one complete Probe→T3 Redirect canonical success. Canonical
publication/validation proves completeness and allocation before recomputing the
mechanism checks. Winner direction, win rate and L_ref never enter the gate.
Full games with inadequate mechanism coverage can publish evidence with
`qualification_gate_passed=false`; they cannot admit Formal. Missing winners
remain INCOMPLETE and cannot publish a completed ITT artifact. Formal admission
stays disabled until N and successful qualification lineage are separately frozen.

Local tests use real canonical types and durable I/O, actual production eval
definitions, scripted dependencies and explicitly synthetic outcomes. They do
not substitute for native server qualification or establish gameplay efficacy.
