# Phase-2 Three-Way Decision：当前工程状态

当前已冻结 ToM Qwen3 final predictor、R2/M3 full-development mapper、Audience Competitive State θ 标签、Action Contract V1、reference continuation-value table。语言执行已有 V1.1 真实 smoke-v1 fixture 回归；V1.2 已用 SHA256 为 `33480c195d6907797a337d3e8f8f1ba4756b97eaa32bf32cb4345974056d0d5d` 的真实 smoke-v2 7 个 final-invalid case（14 段原文）回放：9 段可归于 perception false negative 的发言恢复合法，5 段真实违规仍 invalid。smoke-v3 runner 保持相同 58 个 case、selection digest 和 gate，但尚未真实运行 V3 LLM。V2 的服务器结果为 overall 51/58、Push 20/20、Probe 17/18、Redirect 14/20；离线 fixture 回归不是新的服务器 smoke。

本轮新增 runtime-safe `Phase2DecisionOpportunityV1`、受控 `Phase2TreatmentV1`、verified speech 准备接口、outcome/reference-loss bridge、Push/Redirect terminal risk 纯数学核、Probe 序贯价值接口、non-production 三支比较核、T0–T4 Probe 记录、consequence record、pilot dataset 与只读 support analyzer。Online Pilot-T V1 已冻结“首个双合法 PRE → 均匀候选抽样 → 独立 0.5/0.5 P/N 分配”，qualification 固定 10 次、正式 pilot 固定 120 次 assignment。新增 `fsync` append-only 阶段账本、fail-closed interrupted 恢复、每局至多一次分配、backend 调用审计、opt-in 单路径 hook，以及 assignment/execution/consequence 三表。所有 synthetic 数值只用于测试；没有 fitted λ、Probe kernel、continuation value 或生产 router。`run_random.py` 默认路径和 actual vote 语义未改。

**🔴 当前仍不能运行服务器 qualification。** 真实 Smoke-V3 canonical artifact/gate 尚不存在；当前工作树还未形成 clean tracked source，本机缺原生生产依赖，scripted backend 的完整 canonical 集成验证需在服务器复跑。正式 Q/mapper/reference lineage、运行时配置与 artifact destination 也须由服务器 preflight 核验。完整 PRE checkpoint/restore **不再是 Online Pilot-T blocker**，只保留为未来可选的 paired replay；`Phase2CheckpointClosureV1` 仍 `executable=false`。Phase-2 backend 调用在独立 sidecar 留 hash，并在 canonical call audit 的 `RUNTIME` 类别留证，不混入 V1 speech annotation。详见 [checkpoint/commit contract](phase2-checkpoint-and-commit-contract.md)和[Online Pilot-T runbook](phase2-intervention-pilot-runbook.md)。

依赖关系：

```text
Smoke V3 gate
       ↓
10-assignment qualification: plumbing / audit / ledger / outcome / resume review
       ↓
120-assignment Online Pilot-T: uniform candidate j → randomized P/N → one speech/game
       ↓
assignment + execution/noncompliance + day Y/L_ref
       ↓
support review → terminal consequence estimator → λ(P/N)
       ↓
Probe Pilot → Probe value model
       ↓
production ThreeWayRouter → gameplay integration
       ↓
paired win-rate experiment

Optional separate track: complete PRE checkpoint/restore → paired branch replay
```

下一阶段须人工审阅已版本化的 opportunity/candidate 规则、seed、游戏范围和服务器装配，在服务器以原生依赖重跑 canonical 集成测试并运行 Smoke-V3；通过 gate 后先执行 10-assignment qualification，审阅通过后才运行 120-assignment Pilot-T。qualification 记录不进入 λ 数据集。pilot support 回来后才能决定 consequence estimator、λ 和后续 Probe pilot。不能因当前代码中存在数学比较核，就把它称为已可部署的 ThreeWayRouter。
