# Phase-2 Online Pilot-T：执行前 runbook

**状态：服务器 qualification 已完成并通过人工审阅；正式 120-assignment Pilot-T 尚未冻结或启动。** 已核验 qualification manifest digest 为 `7f0154666b6df570b403a6718c7bd3927b4893c6ca383fe07065b518f589dec3`，source 为 `4d0d78a76e1ab51d038208f4ce1b4e982cdd32c3`。下文长命令保留为底层 CLI 说明和历史 qualification 操作记录；不得据此重跑已完成 qualification。主 protocol 是自然游戏中一次性在线随机 P/N 干预。完整 PRE checkpoint/restore 只用于未来可选的 paired-counterfactual replay；它不再是在线因果识别的前置条件。

## 方法和冻结顺序

1. 先预注册 `expected_source_commit` 和 `expected_reference_tables_digest`（由已核验的 development reference table 计算），以及最终 Qwen3 predictor seal/fit、最终 M3 mapper manifest、Action Contract、Language V1.2、模型部署及 runtime 配置。preflight 对预注册 commit 和 reference 内容摘要逐项比对，且 Q 必须是已完成 pinned worker 握手、仍在运行的 `Qwen3GameplayPredictorClient`；仅提供 `seal_digest` 字符串的替身不能通过。tracked source 与 index 必须 clean；无关 untracked 文件不阻断。正式 artifact 目录必须不存在。
2. 在服务器以同一 selection digest `9ab58cb73fdee18fee63795ee71df37e5ff00c2c320939c2f48221c3167e192e` 和原 gate 完成真实 58-case Smoke-V3。preflight 必须读取、验证不可覆盖 canonical smoke-v3 artifact 及 `gate.passed=true`；Smoke-V1/V2 fixture 回归不能替代这一步。
3. 人工审阅 `Phase2OnlineTerminalPilotPlanV1` 的已冻结规则、纳入游戏范围、seed、服务器部署与止损上限。每局只考虑第一个 P/N 双合法 wolf speech/speech_pk PRE，在其 $J_{PN}$ 中均匀抽 $j$；记录候选池、$1/|J_{PN}|$、selection seed/key，再以独立域分隔随机键按 P/N 各 0.5 分配。Probe 合法性不影响 Pilot-T eligibility，Pilot-T 不分配 Probe。
4. 先运行独立的 `paper-phase2-online-terminal-qualification-v1`，固定 10 次 **assignment**，审阅 runner、审计、commit、账本、日终和恢复。qualification 永不进入未来 λ 数据集。审阅通过后再运行 `paper-phase2-online-terminal-pilot-v1`，固定 120 次 **assignment**。语言/commit 失败仍计数；`max_games_attempted` 仅防无限运行，触顶未达标为 `INCOMPLETE`。每局至多一次 assignment。

随机 treatment assignment 先于语言生成和 outcome，实施单路径 `do(a)`。不能把自然历史 `vote_intent` 视为 treatment，也不能用 θ 真值、未来投票/放逐、语言成功率或终局结果选择机会或重抽动作。

## 单局执行

运行 canonical game 至当前存活狼 speech/speech_pk PRE；正常收集 speaker handoff。先从合法狼队视角与公开 PRE 构造 $J_{PN}$，均匀选 (j)，再调用冻结 Q、R2/M3 得到完整 p-panel 并随机分配 P/N。`OnlinePilotGameStateV1.try_assign` 将 immutable assignment 写入 `OnlinePilotAssignmentLedgerV1` 的 append-only hash-chain JSONL，文件和新建目录 `fsync` 成功后才允许第一笔 Phase-2 backend 调用。账本先存 pilot/source、game、候选池/概率/key、legal actions、treatment 概率/key；execution/backend calls 与 consequence 各占后续独立事件，不覆盖 assignment。

重启时校验全部账本哈希和 plan/source，恢复已用的 game ID 与 assignment 计数。没有完整 PRE checkpoint 时，已有 assignment 但缺 execution、或成功 execution 缺 consequence 的旧 game 标记 `INTERRUPTED`；旧 game ID 不再重演或重新抽样，后续只能使用新 game ID。10/120 次计数始终以落盘 assignment 为准。

正式 CLI 读取预先冻结的 canonical `CollectionPlan`，以其有序 seed pool 派生 game ID，由显式 opt-in 的 `run_online_campaign` 顺序运行。`ServerRuntimeFactory` 复用 `Classic7RuntimeFactory`，为每局建立 canonical env/agents/recorder/call audit、冻结 predictor/mapper、同一账本和该局通过的 preflight；异常使 campaign 停止，账本供下次从**新** game ID 继续。该 API 不在默认 gameplay 入口自动启用。

语言 actor 在已选 P/N 和目标下生成，独立 perceiver 只看 generated text 与 trusted public context，原 verifier 最多 repair 一次。每次 realization、repair、perception 都在返回前逐笔 `fsync` 到独立 `BACKEND_CALL` 账本事件，记录 pilot/assignment/treatment、role、sequence、attempt、model/backend、request/response digest、PRE 与 canonical call ID；不得覆盖 assignment，也不得冒充 V1 `SPEECH_PERCEPTION` annotation。perception 请求必须逐字等于仅由文本/公开 context 构造的 prompt。sidecar 不保存 secrets。

若语言最终 invalid，记录 `execution_success=false` 和失败原因；**不换成另一 treatment、不重新随机、不删除 assignment**。游戏可按原 baseline speech 继续，但这次不是成功的 P/N 执行，不进入 successful-treatment consequence table。若 canonical commit 报错则中止该实验 game，按 public event 是否已变化区分明确失败与执行状态未定；不得产生伪 outcome。成功时复用 `commit_phase2_verified_speech` 与现有 recorder staged commit，核对逐字文本、Phase-2 audit、canonical V1 annotation 和 event digest；speech vote intent 不直接改 actual vote。
若 commit 路径报错但 public event 数已变化，不能证明 treatment 未执行：此时 execution 留空，恢复时标为 `INTERRUPTED`，不得把它归入明确的 language noncompliance。只有确认没有新增 public event 才记录明确执行失败。

当前 intervention speech 完成后，后续每一步回到原 baseline policy。Pilot-T 是 **one-shot treatment**，并非整局 TWD policy 实验。游戏在原 `vote/vote_pk` 流程结算后，以 `exile_result`（普通 tie 后进入 PK 的中间事件除外）确定日终 Y，复用 `extract_phase2_day_outcome`/`ReferenceTables` 得到 $S_+$、$V_{ref}$、$L_{ref}$。最终狼队胜负可作下游描述性审计，不能代替日终 consequence。

## 数据、support 与发布

`Phase2OnlineInterventionRecordV1` 分清 assignment、execution、day consequence。恢复后从 durable ledger 构造三张独立表：**所有**分配、有证据的执行状态、仅成功执行且有真实日终结果的 consequence；另有 backend_calls sidecar。缺失 execution 不能假写为失败；同一 game 重复写入完全相同内容可幂等识别，冲突则 fatal。真实数据默认要求 canonical execution verifier，synthetic 测试必须显式标记。

当前 `reference_artifact_digest` 参数在 online 路径中必须等于 `reference_tables_digest(reference_tables)`，即加载后 reference table 的确定性**内容摘要**；preflight 还要求它等于独立预注册的 `expected_reference_tables_digest`。预注册摘要的权威来源须由服务器装配者人工核验；相同内容摘要本身不证明上游 publication lineage。正式发布从运行前通过的 `PilotPreflightV1` 直接取冻结 source provenance 和 Smoke-V3 manifest digest，不在语言执行后重新读取 Git。

只读 support 分开列 qualification/formal purpose、games seen/eligible/assigned、P/N 分配、候选池大小和被选席位、执行率/失败原因/缺失数、consequence 缺失数、phase、$S_{pre}$、描述性 p decile、离线 θ 审计层、各臂 Y 和 $L_{ref}$。不拟合 λ，不选择 ITT/per-protocol estimator，不自动平滑或 pooling。θ 只在 collection 后离线联结，绝不进入随机分配或 runtime prompt。

qualification 与 formal artifact 名分别固定为 `paper-phase2-online-terminal-qualification-v1`、`paper-phase2-online-terminal-pilot-v1`，目的地不可覆盖。运行中账本状态 `RUNNING`；安全上限触顶而 assignment 未达标为 `INCOMPLETE`；达到 10/120 且各 assignment 的执行/中断阶段已核验，才能进入 `READY_TO_SEAL`，不可覆盖发布后 manifest 才写 `COMPLETE`。规划文件：`manifest.json`、`pilot_plan.json`、`assignments.jsonl`、`executions.jsonl`、`consequences.jsonl`、`backend_calls.jsonl`、`metrics/support.json`、`report.md`。本轮未生成任何正式 pilot artifact。source、Q/mapper、reference、Smoke-V3 gate、账本摘要和 canonical event/treatment lineage 均须写入 manifest。

`COMPLETE` 仅表示预注册 assignment 数达到且账本/manifest 完整，不表示 qualification 的工程质量自动通过。若 10 次均失败或中断，必须在人工审阅中否决进入 120 次 Pilot-T。

## Preflight 与后续阶段

`phase2_intervention_preflight.py` 只是 static capability diagnostic；不能用其有限参数认证真实 qualification。正式 runner 装配 runtime 后调用同一个 `assess_pilot_preflight` / `PilotPreflightV1`，分开 `online_randomized_pilot_ready` 和 `paired_branch_replay_ready`。Online Pilot-T 要求 clean tracked source、验证过的 frozen artifacts、运行时 Q/mapper lineage、冻结 V1 plan、Smoke-V3 canonical gate、现有 canonical commit/bound backend audit、持久化账本、reference table 及空目的地。checkpoint/branch replay 可以继续为 false，**不能因此使 online ready 失败**。CLI 的新增源码和已有 tracked 改动未提交时，正式预检应失败；无关 untracked 研究文档不阻断。完整 checkpoint 的剩余状态闭包及 fail-closed API 见 [checkpoint/commit contract](phase2-checkpoint-and-commit-contract.md)。

## 正式服务器入口与输入

在待执行 checkout 使用服务器 client 环境安装 editable package（含 mapper/ToM 依赖）：`python -m pip install -e '.[mapper,tom]'`。实际执行包必须来自 `--repo`；不得用另一个 checkout 的 installed package 执行当前 source。推荐 module invocation；direct script invocation 也支持，不修改 `sys.path`。两种 `--help` 在解析参数后立即退出，只依赖标准库，不创建 ledger/artifact、不启动 worker、不连接 gameplay。

CLI 接受以下基础设施参数，不提供策略、propensity、threshold 或 assignment-count 覆盖：

| 参数 | 冻结输入/含义 |
|---|---|
| `--campaign-purpose` | `qualification`（10 assignments）或 `pilot`（120）；不自动串联 |
| `--plan` | `Phase2OnlineTerminalPilotPlanV1.to_record()` 的完整 JSON；purpose 必须匹配 |
| `--game-plan` | canonical `CollectionPlan.to_record()` 完整 JSON，含有序 seeds、runtime provenance 和 source pin |
| `--runtime-config` / `--deployment-config` | 已冻结 runtime/loopback deployment YAML，与 game-plan 中摘要和模型身份一致 |
| `--publication` / `--evaluation-root` | sealed 1500-game publication / OOF；复用 offline loader 只读重建 reference tables，无训练 |
| `--mapper` / `--mapper-manifest-digest` | sealed final mapper 与独立预注册摘要 |
| `--reference-tables-digest` | 独立预注册的 reference table 内容摘要 |
| `--smoke-v3` / `--smoke-v3-manifest-digest` | 已存在不可覆盖 Smoke-V3 artifact 与 manifest 摘要 |
| `--q-checkout` / `--q-fit` / `--q-python` | pinned final-Q checkout、sealed fit 和 worker Python；最后一项默认当前 Python |
| `--source-commit` / `--repo` | 本次 CLI commit 后冻结的 clean HEAD / 当前 checkout；repo 默认入口所在 checkout |
| `--work-directory` | 独立、持久、受限的运行目录；首次必须不存在 |
| `--destination` | 固定名称的正式 artifact 目的地；必须不存在，与 work directory 不嵌套 |
| `--resume` | 显式恢复完全相同 inputs 和 ledger；无隐式重启 |
| `--preflight-only` | 完整装配及 Q handshake，使用临时 ledger；不开始 game/assignment/语言处理，不发布 |

先由研究者冻结两份 purpose 独立的 Pilot-T plan / canonical game plan，不在 CLI 中生成研究参数。canonical `collection_id` 必须等于对应 `pilot_id`；其 `source_revision` 必须等于提交 CLI 后的 `--source-commit`。Pilot-T plan 的 `max_games_attempted` 必须为显式有限上限，且不超过有序 seed pool；pool 至少覆盖固定 assignment count。canonical plan 复用 `scripts.collect_games.plan_fields` 的生产 metadata、prompt/retry identities 和 call budget，配置摘要也须由实际配置计算。canonical `target_canonical_success_count` 是该 plan 的 metadata，**不改变** Pilot-T 的 10/120 assignment 停止条件。不得借此改变纳入规则或重新选择 seeds。

正式目录仅保存冻结表、摘要与 support；运行目录另保留 append-only `assignment-ledger.jsonl`、不可覆盖 `run_inputs.json`、canonical claims，以及每次 execution/consequence 的 canonical partial evidence。证据包含真实 PRE、public event 和 canonical backend request/response；按受限数据保留，不作为公开 prompt。发布前 verifier 从这些持久证据核对实际 commit、audit 和投票日终，失败则不发布。其文件摘要进入 manifest 的 `server_run_provenance`，因此恢复后不依赖先前进程的内存。

## 服务器命令顺序

以下是实际 argparse 参数模板；`/absolute/...` 和摘要变量须替换为操作者已核验的绝对路径/预注册值，不代表新增研究默认值。当前报告的 Smoke-V3 manifest 为 `22ed81165668c3cc92f05882efca62ce07bf819d98744d662462b40c6c6327f8`，来源 HEAD 为 `da0cf3a5b00ba73c33ef5227eec1789358876923`。

**Step 1：核验已通过的 Smoke-V3。** 原 preflight 校验 manifest/gate、58-case digest、mapper 和 Language V1.2，未要求 Smoke commit 等于 pilot HEAD，也未核对 Smoke source-file digests。正式 CLI 在原 gate 外增加全部七个 Smoke `SOURCE_FILES` 的逐字摘要比对，并保留 clean-source 要求：CLI commit 本身不强制重跑；任何这些文件的字节变化都必须重跑。不得通过更改 guard 复用不一致 artifact。若必须重跑，先依 runbook 使用新的、尚不存在的 storage destination；以下 full smoke 命令会调用真实 LLM，本轮不执行：

```bash
python -m scripts.run_phase2_language_smoke \
  --publication /absolute/development-publication \
  --evaluation-root /absolute/sealed-oof-evaluation \
  --mapper /absolute/paper-phase2-mapper-final-v1 \
  --storage-profile /absolute/smoke-storage-profile.json
```

加 `--preflight` 只验证 selection 与 frozen plans，不能替代真实 Smoke gate。不可覆盖已有 `paper-phase2-language-smoke-v3`。

**Step 2：完整 online preflight。** 在项目根目录执行。先设置 `PILOT_SOURCE_COMMIT` 为提交本轮 CLI 后人工核验的 HEAD、`REFERENCE_TABLES_DIGEST` 为预注册 table 摘要。以下数组复用于 qualification 和未来独立 pilot：

```bash
COMMON=(
  --repo /absolute/Untrusted_Network_Simulation
  --source-commit "$PILOT_SOURCE_COMMIT"
  --runtime-config /absolute/Untrusted_Network_Simulation/configs/runtime/local-qwen35-9b.yaml
  --deployment-config /absolute/Untrusted_Network_Simulation/configs/deployment/qwen35-9b.yaml
  --publication /absolute/development-publication
  --evaluation-root /absolute/sealed-oof-evaluation
  --mapper /absolute/paper-phase2-mapper-final-v1
  --mapper-manifest-digest 8ab529972a5722e61e0a81ec37f089677d1276c21cb83921aa842b2ce33995c3
  --reference-tables-digest "$REFERENCE_TABLES_DIGEST"
  --smoke-v3 /absolute/paper-phase2-language-smoke-v3
  --smoke-v3-manifest-digest 22ed81165668c3cc92f05882efca62ce07bf819d98744d662462b40c6c6327f8
  --q-checkout /absolute/pinned-final-q-checkout
  --q-fit /absolute/sealed-final-q-fit
  --q-python /absolute/final-q-environment/bin/python
)
QUALIFICATION=(
  --campaign-purpose qualification
  --plan /absolute/qualification-pilot-plan.json
  --game-plan /absolute/qualification-canonical-game-plan.json
  --work-directory /absolute/qualification-work
  --destination /absolute/publications/paper-phase2-online-terminal-qualification-v1
)
python -m scripts.run_phase2_online_intervention_pilot \
  "${COMMON[@]}" "${QUALIFICATION[@]}" --preflight-only
```

若重跑 Smoke，替换上面的 manifest pin 为已核验的新摘要。预检先读取/比对 plan、source、配置、mapper、reference、Smoke；再启动并验证 Q worker，构造真实 canonical runtime 和临时 ledger，调用唯一 full preflight。Q 握手不进行 prediction；不调用语言 backend。失败 exit nonzero；不创建正式 work directory、assignment 或 publication。

**Step 3：独立 10-assignment qualification。** 仅在 Step 2 passed 后移除 `--preflight-only`；runner 自身仍重新完整预检：

```bash
python -m scripts.run_phase2_online_intervention_pilot \
  "${COMMON[@]}" "${QUALIFICATION[@]}"
```

运行中的输入/ledger 已存在时必须明确恢复：

```bash
python -m scripts.run_phase2_online_intervention_pilot \
  "${COMMON[@]}" "${QUALIFICATION[@]}" --resume --preflight-only
python -m scripts.run_phase2_online_intervention_pilot \
  "${COMMON[@]}" "${QUALIFICATION[@]}" --resume
```

恢复必须保持 inputs、source、configs、pins、paths、seed pool 与 ledger 完全一致，不重抽旧 assignment、不重演旧 game。缺 execution/day consequence 的旧 assignment 按现有 ledger 记 `INTERRUPTED`，以新 game ID 继续。如果计数已达标、进程仅在 publication 前中断，runner 只验证/发布，不启动新 game；该 publication-only preflight 不能用于 gameplay。完整目的地一旦存在，含 `--resume` 也拒绝覆盖。

### Production gameplay hook 与 source 变更

正式入口实际调用 `werewolf.phase2_online_runner.run_online_game` → `run_random.eval`，使用显式 `online_pilot` keyword-only hook。这个 hook 必须进入提交的 `run_random.py`；本地测试读取未提交 hook 不能证明服务器 clean HEAD 可执行。full preflight 与 `run_online_game` 启动 game 前均检查实际 callable 的参数绑定；缺少 hook 返回 `PRODUCTION_GAMEPLAY_HOOK_UNAVAILABLE`，不能先写 `GAME_STARTED` 再发现 signature mismatch。

该 hook 在 canonical recorder 收集 PRE/handoff 后、baseline action 之前调用 `handle_pre`；成功 canonical commit 消耗当前 speech step，否则继续 baseline。每次实际 env step 后调用 `after_step` 观察日终，finish 后调用 `after_game`。`plan_provider` 只属于既有 constrained planning 路径；language preparation seam 本身不运行 gameplay，不能替代此注入点。默认无 hook 的执行行为不变。

若运行后需修改 tracked source，旧 `run_inputs.json` / ledger 的 source pin 不能替换，旧 canonical claims 也不能重写。必须保留旧 work directory；修复提交后另行冻结新 canonical game plan（新 `source_revision`、必然变化的 plan digest），并使用新 work directory。即使 assignment 为 0，已有 `GAME_STARTED` / claim 仍是应保留的工程失败证据；原 source 下的 resume 规则不能用于跨 source 重启。

本次旧运行没有 assignment，修复后的独立 work directory 可保留原 Pilot plan 的完整冻结内容，重冻结同一 `collection_id` 的 canonical game plan 并显式记录这次零 assignment 工程重启；CLI 不强制要求更换 campaign identity。若研究者选择用新 identity 区分两次运行，则 Pilot plan 的 `pilot_id` 与 canonical `collection_id` 必须同步，并保留原 assignment seed、0.5/0.5、selection rule、10 assignments、安全上限和完整有序 seed pool。新 identity 会改变随机键的域分隔，不能声称未来逐笔抽样结果与旧 identity 相同。不要在旧 campaign 中重写已 claimed game；新的 canonical plan/source 是独立运行，不能把旧 claim 移入新 ledger。这些 plans 由操作者在修复 commit 后冻结，不覆盖旧 plans/evidence。

更新 `COMMON` 的 `--source-commit`、`QUALIFICATION` 的两个 plan 路径和独立 work directory 后，先重新运行 `--preflight-only`；此步骤不启动 qualification。启动前保留 `CUBLAS_WORKSPACE_CONFIG=:4096:8`，它是 sealed Q runtime 的要求。Smoke 仍由原 gate 与全部 `SOURCE_FILES` 字节摘要验证；`run_random.py` 和 online runner/preflight 不在 Smoke 的七个 source files 中，因此仅这些文件的修复不要求重跑 Smoke，也不能省略其摘要检查。

**Step 4：人工审阅 qualification artifact 和 evidence。** `COMPLETE` 不自动通过工程审查；人工确认 backend audit、canonical commit、实际 vote/exile、失败/中断、support 和恢复证据。qualification 不进入 λ 数据集。

**Step 5：未来另行启动 120-assignment Pilot-T。** 仅在人工通过后，用独立的预注册 plans / seeds / work / destination 执行；qualification 不会自动启动此步骤：

```bash
PILOT=(
  --campaign-purpose pilot
  --plan /absolute/formal-pilot-plan.json
  --game-plan /absolute/formal-canonical-game-plan.json
  --work-directory /absolute/formal-pilot-work
  --destination /absolute/publications/paper-phase2-online-terminal-pilot-v1
)
python -m scripts.run_phase2_online_intervention_pilot \
  "${COMMON[@]}" "${PILOT[@]}" --preflight-only
python -m scripts.run_phase2_online_intervention_pilot \
  "${COMMON[@]}" "${PILOT[@]}"
```

Terminal Estimator 数据加载经过 `require_online_dataset_access`：默认仅接受已完成的单一 formal 120-assignment terminal campaign；qualification、synthetic、混合 campaign 全部拒绝。`audit_only=True` 可读这些数据，但返回对象的 `estimator_eligible` 恒为 false。此保护只约束数据用途，不代表 formal support 已经通过科学审阅。

Probe 的当前研究定义以 [Probe Policy Protocol V1](phase2-probe-policy-protocol-v1.md) 为准。经典 λ 不再是开发门槛；不把旧 M2 运输到 T3，也不拟合 observation kernel、information gain 或 Probe value。

### Probe Policy V1 本地执行边界

复用同一 server CLI、canonical PRE hook、账本及提交路径。`Phase2OnlineProbePilotPlanV1` 必须显式提供 N、assignment seed 和 games cap；没有正式默认数值。编排 CLI 的默认 terminal profile 仍只用于 Pilot-T；Probe 必须显式指定 `--policy probe`。

每局首个共同合法 PRE 从 `E_t` 均匀选 j，再以独立随机域分配 `IMMEDIATE_REDIRECT` 或 `PROBE_THEN_REDIRECT`。账本先写完整策略及指定队友/观察窗口，T1 后端调用前记录 `T1_ATTEMPTED`。Probe 规范提交成功才记录 `T3_SCHEDULED`；最终 language invalid 记录 `T3_CANCELLED`，当前及后续恢复 baseline，assignment 留在 ITT。T3 到达后保留 j，使用该队友当前完整面板重新算 k1；`strategy_continuation` 的概率为 1，随机键绑定父 assignment 与 T3 opportunity，不再次随机化。

每次 lifecycle 变化均写不可覆盖的 canonical partial evidence，随后追加策略快照；T3 plan 在后端调用前落盘。backend sequence 跨 T1/T3 连续。结构、PRE、Q 或 commit 异常记录 `STRUCTURAL_FAILURE` 并停止，不能转成 language invalid。

完整 assignment 需要初始合法 Q/面板及 k0；若在落盘前构建失败，账本保存已选 j 的 selection 和 `PREPARATION_FAILURE`，campaign 永久为 `INCOMPLETE`、不可封存。恢复不能跳过这一错误继续凑齐样本；不能把 Q 质量变成隐含入组条件。真实 PRE/运维诊断仍由既有 canonical failure evidence 保存。

`strategy_stages.jsonl` 保存全部阶段证据；day consequence 使用 T0 的 j/acting wolf/S_pre，包含两个阶段的 language invalid 分支。缺 execution 或真实 day outcome 的 assignment 阻止完整发布。已完成真实 day `CONSEQUENCE` 后缺 `GAME_RESULT` 属于 post-endpoint tail interruption：保留全部 canonical evidence，不补造 result、不重放旧游戏或 T3；同源、同输入的合法恢复可继续未用的新 seed。final game result 只作 secondary，缺失不否定已经完成的 T0 day consequence / `L_ref` endpoint。

当前 qualification / formal 的独立目的地分别为 `paper-phase2-online-probe-qualification-v3` / `paper-phase2-online-probe-pilot-v1`。失败 V1/V2 保留原路径和全部证据，永久停止，不 resume。Probe dataset 不进入旧 Terminal Estimator 的 operational access。正式采集前仍须另行冻结两套计划、游戏池及版本摘要，满足 clean-source、canonical runtime、Q/mapper/reference 和 Smoke-V3 guards；本次不执行任何服务器或模型调用。

## 正式 campaign 短命令

`scripts.phase2_online_campaign` 只装配配置、环境、冻结 plans 和既有 server CLI；不执行任何新增方法逻辑。提交 wrapper 和人工预注册 profile 后，在服务器同一 clean checkout、已安装的 client 环境使用：

```bash
python -m scripts.phase2_online_campaign status qualification
python -m scripts.phase2_online_campaign prepare formal
python -m scripts.phase2_online_campaign preflight formal
python -m scripts.phase2_online_campaign run formal
python -m scripts.phase2_online_campaign status formal
# 仅在存在同源、同输入的合法 work evidence 时显式恢复：
python -m scripts.phase2_online_campaign resume formal
```

本轮没有执行这些正式 prepare/run/resume，也没有生成正式 plans。qualification 只允许 `status`，永远 `estimator_eligible=false`；wrapper 不修改或迁移它。其只读来源是 `/data/yuxiao/Untrusted_Network_Simulation/paper-studies/paper-phase2-online-terminal-qualification-v1` 和原 `phase2-online-terminal-qualification-v1-work-runtimefix`，不是早期失败的 work directory。

正式 profile 为 `configs/phase2/online-terminal-pilot-formal-v1.json`。字段为 schema、purpose、pilot ID、assignment seed、安全上限、target、estimator population flag 和四个绝对输出路径。purpose 固定 `pilot`，target 与现有 schema 核对为 120；不提供 assignment 数或概率覆盖。正式最后两项参数已由研究者预注册：

- `assignment_seed = 730553457560960336`。冻结推导规则为 `low63(first8_big_endian(SHA256("paper-phase2-online-terminal-pilot-v1:assignment")))`：取 SHA256 原始 digest 的前 8 bytes，按 big-endian 解读为整数，再与 `(1 << 63) - 1` 做 bitwise AND；结果必须严格等于上述值，不允许改变 seed。
- `max_games_attempted = 240`。预注册规则为 `2 * target_assignment_count = 2 * 120 = 240`；它仅是 safety cap，120 assignments 才是正式目标样本量。达到 120 assignments 即停止继续采集；若先达到 240 games 而 assignments <120，状态必须为 `INCOMPLETE`，不允许自动增加 cap。

seed pool 长度等于这个安全上限。冻结参数的 profile 必须与 wrapper、pins 先提交，属于 preregistered clean HEAD；再冻结 plans。不能在 prepare 后修改 profile 或 source 然后复用原 plans。本次只记录 preregistration 参数，未运行 `prepare formal` 或生成正式 plans。

runtime pins 集中在 `configs/phase2/online-terminal-runtime-pins-v1.json`。它绑定已完成 qualification artifact manifest、原 `run_inputs.json` 内容摘要、development CollectionPlan 摘要和 publication manifest 摘要。wrapper 校验 artifact 全部文件、run_inputs 自身摘要与 manifest 的绑定、plan/source/mapper/reference/Smoke/Q lineage，再从原 run_inputs 读取 publication、evaluation root、mapper、reference、Smoke-V3、Q checkout/fit/python、runtime/deployment 路径和 canonical environment provenance；不复制这些 artifacts。qualification inputs digest 为 `c4d81b3f59fa1c767023a10d94ba1f7571358af8f8cddd43f67fcc2fea93bb6b`。development frozen plan 摘要为 `3bb599ac9504b981c8f8ec7f434f60cdc7ff654ce568da7ed6d05a0af6682a46`，完整池为 2,250 seeds，目标为 1,500 canonical successes。

`prepare` 使用 production `derive_seed_pool` / `plan_fields` / canonical plan validator，以当前 clean HEAD 冻结两份 immutable plans；排除全部 qualification、development 预声明池和 calibration seeds，碰撞则失败，不 skip/reroll。任何输出 plan/work/destination 已存在即拒绝；路径不得覆盖或嵌入 qualification evidence。两个文件均 exclusive durable create，若中途失败，保留已写出的文件，人工审查，不自动删掉或重建。tracked worktree 与 staged index 必须 clean，无关 untracked 研究文档允许，但 wrapper/profile/pins 自身必须已提交。

执行前自动设置未定义的 `CUBLAS_WORKSPACE_CONFIG=:4096:8`；已定义为其他值（含空字符串）立即失败。status 不修改这个环境变量。`preflight` 直接调用既有 full server preflight，输出 source、plan digests、destination 状态、reference digest 和完整 ready/blockers/Q/Smoke/mapper lineage 检查；exit code 适合 shell 检查。Q 仅启动验证握手，不 prediction；仅使用临时 ledger，不创建 durable work、game、assignment 或语言调用。ready=false 或任何输入错误均非零退出。

`run` 不生成 plans、不覆盖已有 destination、不隐式 resume。`resume` 必须有原 work/run_inputs/ledger，原 source/config/pins/plans/path 必须完全一致；继续走底层 loader 和账本恢复，不能重抽旧 assignment。所有运行命令都将 frozen canonical plan 与 qualification runtime provenance 的 production plan 重建结果逐项核对，再委托既有 parser / `execute_server_campaign`；不拼接 shell runner 命令、不建立第二套 gameplay loop。

## Probe qualification / formal 准备

本地 PRE contract 修复基准为 `8bb1ccc6ed5fe2c6f1e572dfee09dd0c1c39d4ff`。源码和 profile 必须已提交且 tracked worktree/index clean；无关 untracked 文档不阻断。prepare 要求操作者显式给出实际执行 commit，canonical game plan 与后续 preflight 必须精确匹配该 clean HEAD。Qualification V3 必须绑定本轮 endpoint 恢复边界修复之后的新 clean commit，不能沿用 V2 的旧 source pin。qualification 后冻结 formal cap 时，formal 应绑定后续新的 clean commit；两次 campaign 的 source commit 可以不同，formal admission 仍须满足 qualification runtime 源文件字节一致的既有 guard。

当前 CLI qualification profile 为 `configs/phase2/online-probe-qualification-v3.json`，formal profile 为 `online-probe-formal-v1.json`。原 Qualification V1/V2 profiles 和原 work/ledger/plans/canonical partial evidence 永久保留不改、不删除、不覆盖、不 resume、不 replay，不用于 Probe effect estimator 或 cap feasibility 规划。按服务器失败报告，V1 有 2 assignments、1 consequence、1 STRUCTURAL_FAILURE 与 unfinished assigned game，状态 INCOMPLETE；V2 在 source `5359829a48432b9c510bb395578d870ceebe211c` 下已形成 8 assignments，其 canonical game-plan digest 为 `ef8e2d0d239f3e0d98ef84ed44c51d3cbceb80b4ddacee5b52fff6a7ac0bd151`，旧 campaign 永久停止。本轮只为新 V3 修复 endpoint 后的恢复边界，不回写或修补旧 V2 ledger。

V3 是独立 identity `paper-phase2-online-probe-qualification-v3`；plan/game-plan 为 `plans/phase2-online-probe-qualification-v3{,-game-plan}.json`，work 为 `paper-studies/phase2-online-probe-qualification-v3-work`，destination 为 `paper-studies/paper-phase2-online-probe-qualification-v3`，均位于原服务器项目目录下。Probe Policy Protocol V1 与策略语义保持不变。此次修复的是 primary endpoint 完成后的恢复边界，target/cap 无须改变：

| Campaign | Randomized assignment target | Max games attempted | Assignment seed | Planning status |
|---|---:|---:|---:|---|
| qualification V3 | 20 | 80 | 6288282358363007265 | FROZEN |
| formal | 200 | 800 | 6449283966833280907 | FROZEN |

allocation 保持现有每次 assignment 独立的 1:1 随机化（两策略概率各 .5），不强制最终恰为 10/10 或 100/100。qualification 仅验证 execution/mechanism，`estimator_eligible=false`，不进入 Probe effect estimator。达到 assignment target 即停止；cap 用完而 target 未满为 `INCOMPLETE`。不得自动扩 cap、更换 seed、补样本、rerandomize 或 replay 已受控 game。V3 已在 clean source `b9a88afe8673fe95c41f4b8279581f2c7baa15f1` 下完成；本次只冻结 formal profile，正式 prepare 仍须等待本轮改动提交为新 clean source，并通过既有 admission/preflight。

服务器已报告 V1 source 为 `8bb1ccc6ed5fe2c6f1e572dfee09dd0c1c39d4ff`，canonical game-plan digest 为 `4719b85665febcb47479a594f13c00940a6636061ebe9410c4b68391be4bda6e`。V1 有 2 assignments（两策略各 1）、1 consequence、1 `STRUCTURAL_FAILURE` 和 unfinished assignment，状态为 `INCOMPLETE`。这是 canonical PRE consumer 错用不存在的 `.phase` 导致的工程 qualification failure；修复直接使用既有 `phase_context(prefix)`，不改变 Probe protocol。V2 已形成的 8 assignments 和中断证据完整保留。独立 V3 identity/source 用于 endpoint 恢复边界修复后的机制验收，不依据 outcome 改 target、cap 或 treatment allocation，因此不是 outcome-based rerandomization。

种子与 provenance：

- assignment seed 沿用已有 identity 推导 convention：`low63(first8_big_endian(SHA256(UTF8(pilot_id + ":assignment"))))`，即 `int.from_bytes(hashlib.sha256((pilot_id + ":assignment").encode("utf-8")).digest()[:8], "big") & ((1 << 63) - 1)`。V3 输入为 `paper-phase2-online-probe-qualification-v3:assignment`，结果 `6288282358363007265`；V1/V2 的历史 seeds `4325493448326950820` / `766378239883399222` 保留不变。formal 输入与 seed 仍为 `paper-phase2-online-probe-pilot-v1:assignment` / `6449283966833280907`。不得在同一 campaign 内更换 identity 或 seed。
- gameplay seed pool 沿用 `collect_games.derive_seed_pool(pilot_id, cap)` 的有序、域分隔规则，区别于 assignment seed；qualification/formal 使用不同 identity。候选选择与策略随机化继续使用已有分离随机域。
- 复用 `online-terminal-runtime-pins-v1.json` 指向的已验证 runtime、Q、mapper、reference、Smoke-V3、development publication。Smoke 复用仍由原字节 digest guard 决定；本轮未改 Smoke SOURCE_FILES 的内容。
- `excluded_game_plans` 绑定旧 Terminal formal canonical plan digest `add3fe53054d9b89b6c6570b585bce13070f010dcf2e28a40a4671fcd713c64b`。该值读取自既有只读本地缓存中冻结 Terminal formal manifest 的 `server_run_provenance.game_plan_digest`；manifest 内容哈希核验为 `ec1cc8730aefe8fce57ed83a3c84ca9231ae645ea6c9047cd588f8022d2223c2`，本轮未读取服务器实际 plan。prepare/preflight 仍必须读取真实 canonical plan 并核验此 digest。校验全部 seed pool 与 Terminal qualification、Terminal formal、1500-game development、calibration 不相交；formal 还必须与 Probe qualification 不相交。发现碰撞直接停止，不 skip/reroll。
- V3 与 formal 的 `excluded_game_plans` 已绑定上述 V1/V2 digests；prepare/preflight 必须读取服务器原 V1/V2 canonical plans、核验 digests，并排除其完整 planned seed pools，包含已用和未用 seeds；不能只排除已形成的 2/8 assignments 或成功游戏。旧 V1/V2 输出路径受到既有 path-overlap guard 保护。V3/V1、V3/V2、V1/V2 各 80 seeds 按真实 production derivation 静态计算交集均为 0；同一 derivation 根据冻结 Terminal formal profile 的 identity/cap 推导出 240-seed planned pool，V3 与该静态推导池的交集也为 0。本地没有真实 Terminal formal canonical plan，这些静态计算均不替代读取真实 server plans 并核验 digests，不声称已核验实际服务器池。formal 另排除已完成 Qualification V3 的完整 pool。
- 新 campaign 的 plan digest 只由最终 canonical plan 内容生成，不手工伪造或预填；canonical plan 额外绑定 Probe protocol 内容 SHA256 和当前 profile 的 canonical JSON digest。preflight 再绑定核验；protocol/profile 漂移在 Q worker、backend、gameplay 前拒绝。publication 将这两个已验证的冻结 digest 写入 `server_run_provenance`，使后续分析不依赖历史 work 路径。
- formal 的 `probe_qualification` 只指向 Qualification V3 artifact 与原 run_inputs，已冻结下列确切 digests；V1/V2 不可用于 formal admission，不读取 latest。formal 四个输出路径不得覆盖任一 qualification/formal 既有 evidence 路径。`probe_inputs` 经 qualification admission 后，直接将 V3 `canonical_game_plan` 加入现有 exclusion 集合，排除全部 80 planned seeds，不仅是实际运行的 30 games；不另加重复 exclusion。

据冻结的服务器交接，V3 为 `COMPLETE`，20 assignments / 20 consequences、30 games seen，mechanism gate passed。本轮不连接服务器、不修改历史 evidence；prepare/admission 仍须从真实 artifact、原 `run_inputs` 和 durable ledger 重新核验，不能用交接文本代替 guard。

| V3 provenance | Frozen digest |
|---|---|
| manifest | `927dddc2d48d42bca2cb48034b748803e9a0086b0154aa977f26fd021f1ab069` |
| run inputs | `3c7e6f3c7065aa7b345c9478ffbe3b1c1407a4e5b8c20459538bcda413df5aca` |
| pilot plan | `aa4d99ccd0ec6535f715a468335eb5f7e34e9e886ece51b033fc8afb49f195c0` |
| canonical game plan | `97ecfabf670d05d799fb39819ca4e78275416d004cdb936eb455ec18f40a1aca` |
| ledger | `a91d6cbf5683e4e925088902603e27b0aec4255398f4cb318d5d5eebc6d990e7` |

原 `run_inputs.pilot_plan` 不含 `plan_digest` 字段；pilot-plan digest 的权威来源是 sealed manifest / pilot-plan file，并由现有 `plan.digest()` 核验。不得为只读打印结果补写历史 `run_inputs`。

Qualification hard gates：全部 assignment 的 execution 与 primary endpoint（真实原始 T0 day consequence / `L_ref`）完整且账本可封存；至少一次 Immediate Redirect 规范提交成功；至少一条 Probe T1→预定 T3 Redirect 规范提交完整成功。完整 `GAME_RESULT` 仅为 secondary，不作为 qualification hard gate，也不能因缺失而否定有效 `L_ref`。candidate、same-j、k1 重算、语言失败取消/基线分支、公开观察、vote/consequence linkage 与规范调用证明由现有严格 ledger/lifecycle、server verifier 及确定性测试核验。
不以 L_ref、答案质量或 execution-success rate 设通过阈值。T1-invalid、T3-invalid、speech_pk 可能在真实 qualification 中零次出现，它们由确定性 regression 验证，不强求真实失败。不可覆盖 artifact 的 `COMPLETE` 只表示采集证据完整；mechanism gate 失败仍不能进入 formal，wrapper 返回失败并保留 artifact。

Probe resume 不恢复游戏状态：

| 中断位置 | 恢复语义 |
|---|---|
| assignment 后、T1 前 | 保留 assignment；`INTERRUPTED` / `INCOMPLETE`，不继续新游戏凑样本 |
| T1 committed / T3 scheduled / T3 prepared | 同上，不重放 T1/T3，不改队友/j，不重新随机化 |
| T3 committed 或 execution 后、日结局前 | 同上，不插补结局 |
| 真实 day consequence 后、game-result 前 | post-endpoint tail interruption；保留 canonical evidence，标记尾部中断，可继续未用的新 seed；不补造 result、不重放旧游戏/T3 |
| 全部 game-result 已落盘、publication 前 | 可只恢复验核与不可覆盖封存，不调用受控语言、不运行游戏 |
| 完整零 assignment 游戏后 | 仅可继续未用的新 game ID，不重放旧游戏 |

`status` 只读，不写 START/INTERRUPTED；报告 unfinished assignment、post-endpoint tail interruption、lifecycle counts 与 qualification gate。primary endpoint 前的 `PREPARATION_FAILURE`、structural failure、assigned interruption 都保持 `INCOMPLETE`、不可继续新游戏、不可封存，不能隐藏成 language invalid 或静默丢弃。primary endpoint 后的 tail interruption 完整保留，不以 secondary game-result 缺失阻断已完成的 ITT endpoint。

在参数、evidence digest 与执行 source pin 全部冻结后，未来服务器命令如下；本轮未执行：

```bash
python -m scripts.phase2_online_campaign status qualification --policy probe
python -m scripts.phase2_online_campaign prepare qualification --policy probe --source-commit <frozen-execution-commit>
python -m scripts.phase2_online_campaign preflight qualification --policy probe
python -m scripts.phase2_online_campaign run qualification --policy probe
python -m scripts.phase2_online_campaign status qualification --policy probe
```

这些命令只操作 V3。仅在上述表格允许的 V3 endpoint、完整游戏或 publication 边界，才使用 `resume qualification --policy probe`；primary endpoint 前的 unfinished assignment 不可恢复采集，V1/V2 永不 resume。formal 用同一入口将 `qualification` 换为 `formal`，必须先通过上述 qualification gate 并完成后续 profile 冻结；没有自动串联两次 campaign。preflight 只使用临时 ledger 和 pinned Q handshake，既有 loader 继续核验模型路径、部署、依赖、hook、source、artifact、输入和目标路径；不调用 predict，不运行 game/语言，不创建 durable campaign work。

### Formal precision planning 与 operational cap 冻结

Primary estimand 保持全部 randomized assignments 的完整 assignment-policy ITT：`E[L_ref | Probe-policy] - E[L_ref | Immediate-Redirect-policy]`，lower is better，不按 execution success 删除 assignments。

正式规划目标冻结为 95% CI half-width `0.05`。planning variance 取冻结 Terminal Pilot 两臂中较大的 sample variance，冻结数值为 `0.030782462625115643`；使用 `z_0.975 = 1.959963984540054` 与 approximate balanced allocation：

`N ≈ 4 * z_0.975² * variance / half_width² = 189.19930011830033`。

因此预注册 formal target 为 **200 randomized assignments**；该平衡方差代理在 N=200 时 half-width 约为 `0.04863117571557113`。这是实验前的 precision planning，依赖历史方差运输与近似平衡假设，不保证 Probe 实际 CI 达到该半宽。不得使用 qualification 的 treatment effect、arm means、p-value、L_ref difference 或 treatment direction 重新调整 formal N。

历史 variance 来源为完整 120-row ITT；`itt_rows.jsonl` SHA256 为 `7ebb3b4848712ea40142ef6a1bebe1ef9ebbf1eb6e1ff1c0513d673cdfb86779`，manifest digest 为 `58a8611a0c9f5605f6cbdf5a7a9ea9f0c74fc69e637760e507f073e298cfaf8c`。不得用缺少既有冻结 recovery 的 119-row fixture 代替该 planning population。

qualification 的 20/80 是预注册机制覆盖目标与预算，不是 effect-estimation 样本量。旧 first-P/N PRE 中有 88/120 存在 E_t、194 个合法候选，这不是新策略首个 E_t 的逐局入组率；不能运输旧 Terminal 的成功率来冻结 formal cap。

formal max-games cap 冻结为 **800**。沿用 qualification 预先设计的 `20 target / cap 80`，保留相同 `80 / 20 = 4 games / assignment` operational safety ratio：`200 * 4 = 800`。V3 的 30 games / 20 assignments 仅为运行可行性记录，不用于效果估计或按臂调整预算。cap 只作安全上限，200 assignments 才是目标样本量；达到目标即停止，先达到 cap 而 assignments <200 则为 `INCOMPLETE`。

不得依据 qualification arm means、L_ref difference、Y distribution、execution success difference、significance、游戏胜负或 apparent policy effect 调整 formal N、seed、cap 或 policy。不得自动补样本、扩大 cap 或 rerandomize。

formal profile 已绑定 V3 manifest/input digests，并标为 `FROZEN`。随后人工审阅并提交为新的 clean source，再 prepare/preflight/run formal；运行时源码字节必须继续匹配成功 V3 qualification。当前冻结不生成真实 plans，不执行 prepare 或服务器调用，也不降低 COMPLETE/sealable、mechanism gate、protocol、source/runtime、population/plan 或 ledger integrity guards。

### Formal Probe primary ITT 分析

`scripts.phase2_probe_itt_analysis` 只读取一个已完成、不可覆盖的 `paper-phase2-online-probe-pilot-v1` artifact，强制正式 identity/seed、200 assignments、显式预期 manifest digest、干净执行源码及 protocol/profile pins，重新核验表联结、生命周期、support 和 T0 consequence。qualification、缺失结局、结构失败或混合输入拒绝；不读取历史 work 文件，不补结局。

Primary 保留全部 randomized assignments，包括 language invalid 与 T3 cancelled：输出 arm counts、arm means、Probe-policy minus Immediate-Redirect 原始差值、sample variance（ddof=1）、Neyman SE 与冻结的正态 95% CI（z=1.959963984540054）。任何臂少于两个结局不产生 CI。不存在 execution/T3 筛选参数；这些过程量只作 secondary descriptive。协议未预注册 Fisher Monte Carlo seed/重复次数，本实现不增加模拟检验，不拟合 conditional Probe model。

未来 formal 完成后才可运行：

```bash
python -m scripts.phase2_probe_itt_analysis \
  --formal /data/yuxiao/Untrusted_Network_Simulation/paper-studies/paper-phase2-online-probe-pilot-v1 \
  --expected-formal-manifest-digest <verified-formal-manifest-digest> \
  --destination /data/yuxiao/Untrusted_Network_Simulation/paper-studies/paper-phase2-probe-itt-analysis-v1
```

分析独立发布 `itt_rows.jsonl`、primary/secondary JSON、短报告及确定性 manifest；不修改 formal/ITT artifact。provenance 在分析开始采集一次，绑定源 manifest/file table、plan/run-input/reference/protocol/profile digest 与 analysis 源码内容，发布前核验输入/源码未漂移。目的地必须与 formal/work evidence 分离且不存在。本地测试只有明确的 synthetic artifact，不宣称真实 Probe effect。

`status` 只读校验账本哈希链和输入绑定；缺 work/destination 返回 `NOT_STARTED`，已存在但缺失/损坏的 evidence 报错，不创建 START、不标记中断。输出 games、assignment、PUSH/REDIRECT、execution/failure、consequence、backend calls、remaining target、source、paths、terminal digest；最终 artifact 存在时还核对 artifact/input/ledger 绑定后显示 `COMPLETE`。backend calls 是 Phase-2 专用账本调用数；不冒充整个 baseline gameplay 的调用总数。

本 wrapper 需要新的 source commit；正式 canonical plan 在它和 profile 的最终提交之后才冻结。已经完成的 qualification 保持原 source、digest 和 evidence，不因新增 wrapper 重跑。Smoke-V3 七个 `SOURCE_FILES` 没有修改时，可由现有逐文件 digest guard 验证复用；不更改 gate，也不豁免检查。
