# Phase-2 language smoke V3

V3 主要调整 Redirect realization 与通用 perceiver 的否定作用域，并更新 artifact 版本；通用校正同样接受现有 Push/Probe 回归测试。`language_semantic_version=phase2_speech_semantic_v1_2`；language audit record schema 仍为 V1。Runner 写入新的不可覆盖 `paper-phase2-language-smoke-v3`，复用 V1/V2 相同 seed、58 个 selected identities、Action Contract、Redirect selector、mapper、一次 repair、metrics 和 gate。固定 selection digest：

```text
9ab58cb73fdee18fee63795ee71df37e5ff00c2c320939c2f48221c3167e192e
```

预期目标路径为 `/data/yuxiao/Untrusted_Network_Simulation/paper-studies/language-smoke/paper-phase2-language-smoke-v3`。沿用 [V2 runbook](/Users/name_yuxiao/Desktop/VscodeProjects/Untrusted_Network_Simulation/docs/research/phase2-language-smoke-v2-runbook.md) 的配置和命令，仅源代码与目标 artifact 名称由 V3 runner 决定。先运行 `python scripts/run_phase2_language_smoke.py --preflight`；它不调用 LLM、不产 artifact。正式命令仍用同一脚本与服务器配置；runner 在首次 LLM 前冻结 source provenance，要求 tracked source/index clean，untracked 文件不阻断，并在 publish 时复用该对象。

原 gate 不变：overall ≥ 0.90，Push/Redirect/Probe 各 ≥ 0.80，且无 schema-wide parse failure、perception action collapse、requested-plan leakage 或 systematic illegal extra commitment。当前本地**没有运行真实 V3 LLM**，也没有 smoke-v2 `case_executions.jsonl` 原文，因此本地只能针对用户提供的 V2 汇总和通用否定句做回归；不能宣称逐条修复 V2 的四个 Redirect rejection failure。拿回 V3 `manifest.json`、`selected_cases.json`、`case_executions.jsonl`、`metrics.json`、终端 digest/gate 输出后，应按 V1/V2/V3 相同 case identity 做逐例审阅。语言 gate 不能替代 intervention pilot 或 gameplay 效果。
