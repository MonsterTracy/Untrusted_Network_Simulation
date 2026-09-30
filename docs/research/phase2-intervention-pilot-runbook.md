# Phase-2 Online Pilot-T：执行前 runbook

**状态：待服务器 qualification；当前不得运行真实实验。** 主 protocol 是自然游戏中一次性在线随机 P/N 干预。完整 PRE checkpoint/restore 只用于未来可选的 paired-counterfactual replay；它不再是在线因果识别的前置条件。

## 方法和冻结顺序

1. 先预注册 `expected_source_commit` 和 `expected_reference_tables_digest`（由已核验的 development reference table 计算），以及最终 Qwen3 predictor seal/fit、最终 M3 mapper manifest、Action Contract、Language V1.2、模型部署及 runtime 配置。preflight 对预注册 commit 和 reference 内容摘要逐项比对，且 Q 必须是已完成 pinned worker 握手、仍在运行的 `Qwen3GameplayPredictorClient`；仅提供 `seal_digest` 字符串的替身不能通过。tracked source 与 index 必须 clean；无关 untracked 文件不阻断。正式 artifact 目录必须不存在。
2. 在服务器以同一 selection digest `9ab58cb73fdee18fee63795ee71df37e5ff00c2c320939c2f48221c3167e192e` 和原 gate 完成真实 58-case Smoke-V3。preflight 必须读取、验证不可覆盖 canonical smoke-v3 artifact 及 `gate.passed=true`；Smoke-V1/V2 fixture 回归不能替代这一步。
3. 人工审阅 `Phase2OnlineTerminalPilotPlanV1` 的已冻结规则、纳入游戏范围、seed、服务器部署与止损上限。每局只考虑第一个 P/N 双合法 wolf speech/speech_pk PRE，在其 $J_{PN}$ 中均匀抽 $j$；记录候选池、$1/|J_{PN}|$、selection seed/key，再以独立域分隔随机键按 P/N 各 0.5 分配。Probe 合法性不影响 Pilot-T eligibility，Pilot-T 不分配 Probe。
4. 先运行独立的 `paper-phase2-online-terminal-qualification-v1`，固定 10 次 **assignment**，审阅 runner、审计、commit、账本、日终和恢复。qualification 永不进入未来 λ 数据集。审阅通过后再运行 `paper-phase2-online-terminal-pilot-v1`，固定 120 次 **assignment**。语言/commit 失败仍计数；`max_games_attempted` 仅防无限运行，触顶未达标为 `INCOMPLETE`。每局至多一次 assignment。

随机 treatment assignment 先于语言生成和 outcome，实施单路径 `do(a)`。不能把自然历史 `vote_intent` 视为 treatment，也不能用 θ 真值、未来投票/放逐、语言成功率或终局结果选择机会或重抽动作。

## 单局执行

运行 canonical game 至当前存活狼 speech/speech_pk PRE；正常收集 speaker handoff。先从合法狼队视角与公开 PRE 构造 $J_{PN}$，均匀选 (j)，再调用冻结 Q、R2/M3 得到完整 p-panel 并随机分配 P/N。`OnlinePilotGameStateV1.try_assign` 将 immutable assignment 写入 `OnlinePilotAssignmentLedgerV1` 的 append-only hash-chain JSONL，文件和新建目录 `fsync` 成功后才允许第一笔 Phase-2 backend 调用。账本先存 pilot/source、game、候选池/概率/key、legal actions、treatment 概率/key；execution/backend calls 与 consequence 各占后续独立事件，不覆盖 assignment。

重启时校验全部账本哈希和 plan/source，恢复已用的 game ID 与 assignment 计数。没有完整 PRE checkpoint 时，已有 assignment 但缺 execution、或成功 execution 缺 consequence 的旧 game 标记 `INTERRUPTED`；旧 game ID 不再重演或重新抽样，后续只能使用新 game ID。10/120 次计数始终以落盘 assignment 为准。

服务器装配须提供预先确定的 game ID 序列与无模型调用的 `runtime_factory(game_id)`，由显式 opt-in 的 `run_online_campaign` 顺序运行。factory 为每局建立 canonical env/agents/recorder/call audit、冻结 predictor/mapper、同一账本和该局通过的 preflight；异常使 campaign 停止，账本供下次从**新** game ID 继续。该 API 不在默认 gameplay 入口自动启用，也不包含本轮服务器部署配置。

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

`phase2_intervention_preflight.py` 分开 `online_randomized_pilot_ready` 和 `paired_branch_replay_ready`。Online Pilot-T 要求 clean tracked source、验证过的 frozen artifacts、运行时 Q/mapper lineage、冻结 V1 plan、Smoke-V3 canonical gate、现有 canonical commit/bound backend audit、持久化账本、reference table 及空目的地。checkpoint/branch replay 可以继续为 false，**不能因此使 online ready 失败**。当前 Smoke-V3 尚未运行、source 仍未提交，因此正式 preflight 应失败。完整 checkpoint 的剩余状态闭包及 fail-closed API 见 [checkpoint/commit contract](phase2-checkpoint-and-commit-contract.md)。

未来 consequence / λ / production-loss 数据加载统一经过 `require_online_dataset_access`：默认仅接受已完成的单一 formal 120-assignment campaign；qualification、synthetic、混合 campaign 全部拒绝。`audit_only=True` 可读这些数据，但返回对象的 `estimator_eligible` 恒为 false。此保护只约束数据用途，不代表 formal support 已经通过科学审阅。

Probe Pilot 只保留 T0–T3/T4 sequence 接口：公开 request → 观察窗 → 队友 continuation PRE → 新 Q/p-panel。须待 Pilot-T support 人工审阅后再规划，不拟合 observation kernel、information gain、R_B 或自动 continuation action。完整依赖顺序：Smoke-V3 → Online Pilot-T → support review → terminal consequence estimator → λ(P/N) → Probe Pilot → Probe value model → production ThreeWayRouter → gameplay integration → paired win-rate experiment。
