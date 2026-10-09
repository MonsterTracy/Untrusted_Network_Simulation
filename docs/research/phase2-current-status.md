# Phase-2：当前代码与研究责任

本文是导航，不是新的实验协议。核对基准为 `twd/mainline`、HEAD `57fd2d716eff8ac83477a17298e160c04b7e98b5` 及本地 Router-v1 未提交实现。`research/main-experiments` 在该 HEAD 之上还有两个 paired mapper/development 研究提交；本工作区没有合并它们。

## 当前路径

```text
canonical Public PRE + acting wolf 的合法狼队视角
  → frozen observer-specific Q
  → R2 / frozen M3 mapper score panel
  → opportunity / candidate selection / treatment
  → verified language → canonical commit
  → baseline gameplay / actual vote
  → day consequence / L_ref（Terminal 与 Probe ITT）
```

无 opt-in hook 时执行原 agent。Probe 在有效 T1 commit 后，按冻结窗口到指定队友 T3，固定原 j、重算 Q/panel/Redirect target；最终 LANGUAGE invalid 恢复该 PRE 的基线分支并保留 assignment。结构异常停止并保留证据，不能当作正常不适用。DAY_CONSEQUENCE 是现有 pilot endpoint；完整 GAME_RESULT 是不同的结局，不能互相替代。

| 研究责任 | 当前实现 / 权威入口 | 证据边界 |
| --- | --- | --- |
| Phase-1 ToM | `werewolf/tom/`、`scripts/qwen3_gameplay_predictor.py` | 正式任务、sealed predictor 与 worker lineage 保留；Q 是 observer-specific 相对 suspicion mass，不是身份或胜率概率。 |
| 社会状态 | `phase2_offline.py`、`phase2_mapper*.py`；[offline contract](phase2-offline-data-contract.md) | 1500-game development、canonical θ 标签、R2、OOF 与 M3 final-fit/验证路径保留；运行时 p_tilde 是 score，没有外部 calibration。 |
| Action / Language | [Action Contract](phase2-action-contract-v1.md)、`phase2_language*.py`、`phase2_verified_speech.py` | Push、Redirect、Probe 定义和 verifier 不变；Smoke-V3 仍按原 gate/source digest 核验。 |
| Terminal Pilot / Estimator | `phase2_online_*.py`、`phase2_terminal_estimator*.py`；[Estimator Protocol](phase2-terminal-estimator-protocol-v1.md) | 冻结协议记录全部 120 assignments 的 ITT；conditional M0/M1/M2 是 reduced-form risk，support gate 不等于 runtime eligibility，也不能把系数称为 classic λ。 |
| Probe qualification / formal | [Probe Policy Protocol](phase2-probe-policy-protocol-v1.md)、`configs/phase2/online-probe-*.json`、`scripts/phase2_probe_itt_analysis.py` | V1/V2 失败 evidence 保留，formal profile 绑定 V3 已完成 qualification 的摘要；formal target=200、cap=800。策略比较是 PROBE→REDIRECT 与 Immediate Redirect 的 assignment-policy ITT。 |
| Router-v1 / matched control | [Router contract](phase2-router-v1-contract.md)、[Gameplay infrastructure](phase2-gameplay-qualification-infrastructure-v1.md)、`Phase2RouterPolicyV1`、`test_router_v1.py` | 本地 hook 与独立 gameplay campaign/ledger/publication 校验已实现；首个 Probe-eligible PRE、同一候选规则、每局至多一次初始干预，不主动选 Push。Qualification 固定 20 游戏、10/arm 与 mechanism gate；六份历史 production plan pins 已按服务器校验结果绑定，三个新输出路径已预约冻结。新 20-seed pool 与历史完整池的交集、路径实际可用性及 artifact/source 绑定仍须部署后由生产逻辑验证。Formal N/admission 未冻结；SOURCE_COMMIT_READY 不等于 SERVER_STATIC_PREFLIGHT_READY，更不等于 Qualification 或胜率实验完成。 |

用户交接报告 Terminal 正式 estimator fit 与 Formal Probe 200-assignment ITT 已完成。本地协议/profile 提供相应身份及约束；本次整理没有连接服务器、重新核验远端 artifact 或计算实验结果。源码存在、测试通过、profile 冻结与正式 artifact 完成是不同证据层次。

## 历史实现与保留范围

- [早期三支方法 V1](phase2-three-way-decision-method-v1.md) 是历史方案。其 λ → Probe kernel/value → argmin router 依赖顺序已不代表当前执行路径；旧 risk/probe-value 模块在此前提交已删除，本轮未重新实现。
- 本轮删除 `phase2_intervention.py` 中只有 synthetic tests 调用的 `PlannedBranchV1`、paired planning 与四个始终抛异常的 checkpoint/replay 占位 API。没有可复现的 branch experiment 使用这些接口。
- 同文件的 `Phase2CheckpointClosureV1`、live PRE 核验、环境漂移指纹及 canonical commit binding 保留，schema/digest 不变；完整 checkpoint/replay 不可用，preflight 仍明确返回 `paired_branch_replay_ready=false`。
- `scripts/phase2_integration_seam.py` 的历史导出仍列入 pilot source attestation，保留以维持该契约；实际 runner 从包内 `phase2_execution` 导入。
- `speech_planning`、constrained speech、remote ToM adapter、backbone/paired temporal study 与 zero/explicit baselines 仍承担论文消融或复现责任。它们不等于当前 Phase-2 Router，不能只按“旧 Planner”名称删除。
- assignment schema 的 `paired_branch` 来源值、旧 pilot 命名及审计字段保留；删除未实现执行器不意味着可以改写历史记录格式。冻结协议、配置、科学结果与 failed qualification evidence 均未清理。

## 下一步的真实边界

继续理论工作时，应明确区分 θ、mapper score、行动风险和最终狼队胜率。当前 Probe 是固定 PROBE→REDIRECT regime，Router-v1 是该 regime 与 Immediate Redirect 的 evidence-contracted policy；它们尚不构成已识别六损失或 α/β 的完整 DTRS。

任何独立全局胜率实验都须先冻结自己的游戏级分配、seeds、source/run-plan、完整 GAME_RESULT 持久化与恢复、qualification 和不可覆盖 publication 验证。相同 candidate seed 不保证两次运行具有相同 PRE；当前没有严格 paired counterfactual replay。本文不增加风险模型、阈值、fallback 或新的执行路径。
