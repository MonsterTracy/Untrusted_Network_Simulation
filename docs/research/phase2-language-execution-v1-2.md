# Phase-2 Language Execution V1.2：Redirect 否定作用域

用户提供的真实 smoke-v2 汇总为 overall 51/58、Push 20/20、Probe 17/18、Redirect 14/20；Redirect 唯一未过 gate，最终 4 个 `REDIRECT_REJECTION_MISMATCH`。这支持优先审查 Redirect，但本地未取得 V2 `case_executions.jsonl` 原文，不能将这四例逐条归因于否定作用域。以下修复针对明示的语义规则与 synthetic 回归；真实 smoke-v3 负责检验外部表现。

V1.2 的 perceiver 仍只接收 generated text 与 trusted public context，完全不接收 requested action、(j)、(k) 或 Probe type。确定性校正层对“不投 X”“不以 X 为主要放逐目标”“反对将票投向 X”“绝不能投 X”“X 不是当前主要处理目标”只补 `rejected_targets=X`，并去除由该**否定短语自身**造成的 `commitment_targets` / `vote_intent_targets=X` 误报。若另一独立正向短语确实又推动 X，冲突语义保留，交给原 verifier 判 invalid。“投 Y”“推动放逐 Y”“票转向 Y”“主要处理 Y”补正向承诺与投票意图；“暂时不投”“弃票”“不投任何人”补弃票意图，和正向承诺同时出现时保留冲突。模型的 private-fact flag 与其它不确定语义不因 requested plan 被清除。

Redirect actor 的冻结要求在初始及一次 repair 中重复：本轮不以 (j) 为主要放逐目标、明确转向并把票投给 (k)，不得增加第三目标、弃票、Probe 或私密身份断言；建议一到两句自然发言。`verify_language_execution`、Action Contract、Probe eligibility、Redirect selector、mapper、58-case selection 和 smoke gate 均未改。`LANGUAGE_VERSION` 升为 `phase2_speech_semantic_v1_2`；audit record 字段结构未改。测试覆盖五类否定句、正向转向、同一目标前后矛盾、弃票与正向承诺矛盾、冻结 repair prompt、真实 smoke-v1 fixture 重放，以及 V3 版本/selection lock。
