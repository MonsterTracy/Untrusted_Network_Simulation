# Terminal Estimator Implementation V1

状态：LOCAL IMPLEMENTATION / FORMAL FIT DEFERRED。此文说明独立 analysis pipeline；统计规则仍以冻结的 [Protocol V1](phase2-terminal-estimator-protocol-v1.md) 为准，原 Support Review 不改写。所有本轮模型拟合均使用明确的 synthetic rows，尚未读取或拟合服务器正式 120-row feature dataset。

## 实现与输入

| 文件 | 职责 |
| --- | --- |
| `werewolf/phase2_terminal_estimator.py` | 全 population validation、raw ITT/Neyman、Fisher supplement、固定 OLS/HC1、数值诊断、support gate、独立 audit 与 supported prediction |
| `werewolf/phase2_terminal_estimator_io.py` | sealed formal/ITT lineage、逐 assignment join、source snapshot、不可覆盖 publication、带外 digest 核验后的 estimator loader |
| `scripts/phase2_terminal_estimator.py` | 显式三个路径的 CLI；无服务器默认路径、无 digest override、无 gameplay/runtime 调用 |

正式输入仅为显式给定的 Formal Pilot-T artifact 和 ITT artifact。复用现有 canonical envelope / JSON / SHA256 conventions，以及 ITT analysis 的 `load_formal`、`exact_joins`、`itt_rows`。不调用历史 `analyze`、`bound_inputs` 或 `recover`；manifest / recovery audit 中的 work 与 canonical evidence 路径仅是 provenance 文本，绝不沿路径重新读取或恢复 outcome。

验证包括冻结 formal / ITT manifest digest、ITT JSONL SHA256、source / reference / canonical game plan / run inputs binding；120 个唯一 randomized assignments、58 PUSH / 62 REDIRECT、120 executions / 119 formal consequences 与唯一 sealed recovered outcome；全部 speech、null theta audit、finite loss / score。每个 ITT row 必须与正式 assignment / execution / consequence 或 sealed recovery 完全一致，不跳过坏行。冻结 protocol 与 support review 的文件 SHA256 也必须一致。

统计 API 按 ASCII `assignment_id` 排序并复制输入。所有模型使用完整 120 行、等权 OLS、identity link。额外字段留作 audit，不进入 design。

## 统计与预测接口

`primary_itt(rows)` 独立返回两臂 N / means / sample variances、raw PUSH-minus-REDIRECT、Neyman variance / SE / normal 95% interval。它不依赖 conditional features，也不按 success / state / score 筛选。完整 protocol 的随机化 reference 假设与正态近似局限必须一并解释。

`fisher_sharp_null(rows)` 是补充 sharp-null Monte Carlo test；正式 CLI 强制冻结 seed=5027505451237408553、B=100000、PCG64、ASCII row order、58-of-120、tie tolerance=1e-12 和 plus-one rule。单元测试可显式传小 B 验证重现性；正式 adapter 不提供配置变更入口。Fisher 不参与 model eligibility 或替代 primary。

| 模型 | 固定 columns | k |
| --- | --- | --- |
| M0 | intercept, I_A | 2 |
| M1 | M0 + state_14, state_23, state_24 | 5 |
| M2 | M1 + p_tilde, I_A:p_tilde | 7 |

PUSH=1、REDIRECT=0；state reference=(2,5)。无 category 自动排序、score transformation、model selection 或 fallback。`fit_models(rows)` 输出三个完整 JSON records：系数、raw HC1 covariance、SE、normal coefficient intervals、n/k/df、SVD rank / tolerance / conditioning、leverage / residual、各 sanity checks 和失败原因。

HC1 与 protocol 的矩阵公式一致；不切换 HC2/HC3。允许 covariance 的数值零 eigenvalue；singleton 的有限或零 SE 不意味着其风险已确定。supported-domain mean 检查覆盖每个 state/arm 的两端点；variance 检查另外覆盖区间内 quadratic stationary point。raw risk / CI / covariance 不 clip、不覆盖。微小负 projected variance 只在计算 sqrt 时按明确 numerical-roundoff convention 处理，raw variance 保留。

`support_gate(action, S_pre, p_tilde, *, lineage_verified, diagnostics_passed)` 返回 machine-readable eligibility，不返回风险。只有 (2,4)/(2,5) 且 score 位于 protocol 冻结的**闭区间**内可继续；(2,3) 返回 AUDIT_ONLY，(1,4)/未知 state 返回 UNSUPPORTED，interval 外返回 UNSUPPORTED_EXTRAPOLATION。NaN/inf、缺失、invalid action、lineage 或 sanity 失败均拒绝，不加 epsilon 扩张 score domain。

`predict_risk` 必须通过 gate 和重新核验的模型数值检查；失败的 risk / variance / SE / interval 均为 null。M0/M1 同样要求全局 M2 sanity PASS，不能替补失败的 M2。`audit_prediction` 单独标记 audit_only / supported=false，允许观察已编码稀疏 state 的 raw working prediction，不能提升为 operational 结果。

推荐 artifact 消费入口：`load_estimator(path, expected_digest=独立记录的发布digest)`，随后 `verified.predict(action, S_pre, score)` 或明确的 `verified.audit(...)`。Loader 验证完整 envelope、冻结 parent / protocol / support / analysis-source lineage 和 design coding；每次 prediction/audit 都核验加载时的 model payload hashes，拒绝包括小幅 coefficient drift 在内的内存修改，supported prediction 另重新检查 M2。低层纯数学 API 的 lineage flag 不是 artifact 验证替代品。适用范围仅为本 formal speech、相同 opportunity / treatment execution；此库未接入其他 runtime 或 policy。

## Artifact 与 fail closed

固定 schema：`phase2_terminal_estimator_v1`；study name：`paper-phase2-terminal-estimator-v1`。

输出 files：manifest.json、primary_itt.json、fisher_supplement.json、models/m0.json、models/m1.json、models/m2.json、support_gate.json、diagnostics.json、report.md。

Manifest 记录 source commit/branch、analysis 与依赖 helper / protocol / support 文件 SHA256、tracked diff digest、Python/NumPy/system/machine versions、两个 parent digest、ITT payload digest、reference digest、N/counts、完整 model specs / coding / HC1 / gate version 与 Fisher configuration。Canonical file table 记录每个 payload 的 SHA256 与 byte size，manifest digest 排除其自身字段。无当前时间或 destination / input 路径进入 digest-critical content。相同 inputs/source/environment 下输出字节与 digest 可复现；跨 NumPy/BLAS/平台不承诺 bitwise 相同，应锁定服务器分析环境。

Source provenance 在 computation 前采集一次。Publication 前仅做 source drift 检查和两输入 envelope 的完整复核，直接使用预冻结对象。Output 必须与两个 inputs 分离，且目标不存在。为保证**内容相同也不可复用**，writer 复用现有 canonical/durable primitives 与 atomic no-replace rename，但不进入通用 writer 的 idempotent-reuse 分支。写入/fsync 后先验证完整 staging envelope，再 atomic promotion；损坏的 staging 不会成为正式 destination。并发竞争同样失败；只清理本次 staging，绝不删除 existing destination。

CLI status：

| code | 含义 |
| --- | --- |
| 0 | SANITY_PASSED_FOR_REVIEW；仅可进入后续理论审阅，不代表授权部署 |
| 1 | input / lineage / source / publication 失败；不得当作成功运行 |
| 2 | FAIL_CLOSED；保留完整 primary 与三模型诊断 artifact，但停止 operational 推进 |

M2 rank / numerical sanity 失败不删行、不删列、不 ridge，不调整 interval、不改模型复杂度，不从 M0/M1 择优。输出明确 `classic_lambda_identified=false`；无经典 action-loss lambda、action penalties、Probe、router 或阈值。

Promotion 前失败不留下 destination。若 atomic promotion 已成功而最终 parent-directory fsync/复核报错，可能留下完整、已校验的目录；保留并人工核验其 bytes/digest 与持久化状态，不删除或用同名自动重试。CLI 非零不能被解释为部署授权。

## 本地验证与未来一次正式运行

`tests/phase2/test_terminal_estimator.py` 测试固定数学和 gate，包括独立 Fraction 手算的 Neyman、M0 normal equations / HC1、显式 dummy / interaction vectors、M2 affine mean / quadratic variance，以及 frozen PCG64 one-hot reference。超大整数不能转换为有限 float64 时明确拒绝，gate/audit 不泄漏 OverflowError。`test_terminal_estimator_io.py` 用 tmp_path synthetic envelopes 验证 lineage / joins / CLI / determinism / hashes / exclusive race / immutability，另覆盖 write/promotion failure、corrupt staging 与 manifest/model/gate/primary payload tamper。`terminal_estimator_test_support.py` 的 features、losses、identity 均明确为 synthetic，不代表正式 fit。现有 119-row recorded arm-loss fixture 加上已记录的唯一 recovered loss 仅用于 primary arithmetic regression，绝不据此伪造正式 S_pre / score 数据。

完整本地验收、真实 clean-HEAD full-suite 比较与后续阶段边界见 [Local Acceptance V1](phase2-terminal-estimator-local-acceptance-v1.md)。验收不改写冻结 protocol/support review。

正式 coefficients、HC1 SE/CI、经完整 lineage 核验的 primary Neyman CI、Fisher 结果及正式 estimator digest 均留待服务器运行。用户审阅、自行 commit/push、同步服务器并锁定分析环境后，从 repository root 执行下列一次命令；本轮**未执行**：

```bash
python -m scripts.phase2_terminal_estimator \
  --formal paper-studies/paper-phase2-online-terminal-pilot-v1 \
  --itt paper-studies/paper-phase2-terminal-itt-analysis-v1 \
  --destination paper-studies/paper-phase2-terminal-estimator-v1
```

目标已存在时拒绝；不要删除或覆盖已有正式 evidence。保存 stdout digest 与 exit code，审阅 primary、三个 models、sanity/support diagnostics。PASS 只授予下一阶段理论审阅资格；FAIL 先审阅理论，不自动重跑或升级模型。
