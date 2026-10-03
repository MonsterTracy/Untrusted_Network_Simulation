# Phase-2 Terminal Estimator Local Acceptance V1

状态：LOCAL ACCEPTANCE READY / SERVER FORMAL FIT PENDING。审阅对象是未提交的 estimator implementation，比较基点是 `5a6f2a657d63bb09791e052cfc51de47a2557556` / `twd/mainline`。本轮没有连接服务器、访问正式 artifact、训练 Q/mapper、调用 LLM/gameplay、运行正式 estimator、stage、commit 或 push。

## Source state 与保护边界

开始时 tracked/staged 均 clean；有 9 个 untracked 文件：protocol、implementation 文档、estimator 的 3 个 production/CLI 文件及 3 个 test/helper 文件，另有受保护的 wolf review。最终增加本验收文档及 regression audit 的 Markdown/JSON。新代码尚不属于 HEAD；未来 analysis source revision 必须是人工验收后实际提交的 commit，不能把当前 HEAD 冒充新增代码的已提交 revision。

以下文件逐字节保持不变：

| 文件 | SHA256 |
| --- | --- |
| `phase2-terminal-estimator-protocol-v1.md` | `d6dd3dc34b49f5ea6479a1e13e024008b5c0ea6539e6be3e540b9a6bfb3cfde0` |
| `phase2-terminal-estimator-support-review-v1.md` | `e82d789352cb14b0f368597bf9a844da8d8b3d373325ff494baa817d3088ea8d` |
| `phase2-wolf-pt3wd-review-2026-09-13.md` | `88a30495f24f491872abf12fad63cda01d2e2ee3eeb65d03e33aeb6d544da38d` |

后续 import-closure 专项验收仅修改已有 tracked `tests/tom/test_boundaries.py`；没有修改已有 production source/config、formal/ITT、runtime pins、randomization、Action/Language Contract、Q、mapper、reference tables、实际 vote 或 gameplay。Smoke runner 的 7 个 `SOURCE_FILES` 与 HEAD bytes 全部一致；新增 estimator 文件不在该集合中。不因本轮分析代码的新增而重跑 Smoke，但未来复用仍须执行原 digest guards。

## 源码审阅结论与修复

Standards 审阅未发现需修改的项目规范问题；Spec/数学与 publication 审阅发现并修复两项真实问题：

1. 超大整数转 float64 泄漏 `OverflowError`。现在以 `ValueError` 拒绝；score gate、audit 和模型诊断保持 fail closed。新增 loss/score/coefficient/covariance regression。
2. writer 原在 promotion 后才验证 envelope，损坏的 staging 可能留下目标目录。现在写入/fsync 后先验证 staging，再 atomic no-replace promotion；corrupt-staging regression 在修复前确实失败，修复后通过。

没有扩大模型或更改 gate。复用 canonical JSON、SHA256、manifest、durable directory 与 atomic rename primitives；独立 exclusive writer 保留，是因为通用 writer 的 identical-artifact reuse 与本协议“目标存在即拒绝”不一致。没有新增 framework、debug/test-only production branch 或不必要 dependency。

## 数学 specification 与独立验证

以冻结 [Protocol V1](phase2-terminal-estimator-protocol-v1.md) 为准。全部 120 randomized rows、58 PUSH/62 REDIRECT，包括 language failure 和稀疏 state。Primary 单独计算 raw arm means / PUSH-minus-REDIRECT；不按 execution_success、state、score 或模型状态筛选。

| 项目 | 实现与独立验证 |
| --- | --- |
| Neyman | arm sample variance 的 ddof=1；`s_P²/58+s_N²/62`；正态临界值 1.959963984540054。Fraction 手算两点损失的 mean/variance/SE/CI |
| Fisher | PCG64、ASCII assignment order、58-of-120 permutation、B=100000、冻结 seed 5027505451237408553、ties 与 plus-one。one-hot reference 的 11 draws 固定 tail=3、p=4/12；没有 bootstrap outcome |
| Coding | PUSH=1、REDIRECT=0；state reference=(2,5)，dummy 顺序 (1,4)/(2,3)/(2,4)。逐 state/arm 显式 vector 对照 |
| M0/M1/M2 | 参数数 2/5/7；完整 120 行等权 OLS；M2 只追加原 score 和 A:score。无模型选择、regularization 或自动 fallback |
| OLS/HC1 | SVD full-rank 判断与 solve；df=120-k；sandwich 修正 120/(120-k)。M0 2×2 normal equations 和 HC1 的独立手算 coefficient/covariance/SE/CI |
| Prediction | 原 coefficient 与 covariance；手算 M2 affine mean / quadratic variance；load 后 model bytes hash 防止 mutable drift；raw risk/CI 不 clip |

Neyman 是依冻结 randomization reference 前提的正态近似；Fisher 是 supplementary sharp-null test。HC1 为 working-mean pointwise uncertainty，不能补回 singleton 的信息，也不是个体 outcome 区间。Synthetic PASS 不预判正式 M2。

## 输入、artifact、CLI 与 support gate

三个文件分别承担 pure statistics (`werewolf/phase2_terminal_estimator.py`)、sealed I/O (`werewolf/phase2_terminal_estimator_io.py`) 和唯一 CLI (`scripts/phase2_terminal_estimator.py`)；接口细节见 [Implementation V1](phase2-terminal-estimator-implementation-v1.md)。

| 固定输入 | Pin |
| --- | --- |
| Formal manifest | `ec1cc8730aefe8fce57ed83a3c84ca9231ae645ea6c9047cd588f8022d2223c2` |
| ITT manifest | `58a8611a0c9f5605f6cbdf5a7a9ea9f0c74fc69e637760e507f073e298cfaf8c` |
| ITT rows SHA256 | `7ebb3b4848712ea40142ef6a1bebe1ef9ebbf1eb6e1ff1c0513d673cdfb86779` |
| Formal execution source | `05c217c98dab74e919e525f76c55ca3a1e12867e` |
| Reference table | `1bc2fa51ea771e90389cb511bc0c61a442d862c872a121d61037aca6308f7c2c` |

验证所有 file-table bytes、原 assignment/arm/pre-features、唯一 assignment/game IDs、120 executions/119 consequences 和已 sealed recovery 的 exact join；不沿历史 work/evidence 路径读取或重新恢复结果。缺失、重复、非法值或 lineage 不符停止全分析。输入 byte immutability 有测试。

Artifact type/schema：`phase2_terminal_estimator` / `phase2_terminal_estimator_v1`。Payload 为 primary、Fisher、M0/M1/M2、support、diagnostics、report，另有 canonical manifest。Provenance 在计算前生成一次，publication 前检查漂移但不重建对象；包含 source commit/branch/files/tracked-diff、环境、parents、protocol/support digest、coding、HC1 与 Fisher 规则。Digest-critical content 不含 wall-clock、mtime、临时目录、input/destination path。不同 tmp paths 的 identical-source/environment 运行有 byte/digest 对照。跨 NumPy/BLAS/platform 不承诺 bitwise 一致；正式服务器须记录并锁定环境。

目标独占创建；既有目标即使内容相同也拒绝；竞争、write/promotion failure 或 corrupt staging 不覆盖 evidence，只清理本次 staging。若 promotion 后最终 fsync/复核失败，保留可能已完整发布的目录并人工核验，不同名重试或删除。

| Gate state | Domain |
| --- | --- |
| SUPPORTED | (2,4)，score ∈ [0.0077301424406220195, 0.6566524289627977] |
| SUPPORTED | (2,5)，score ∈ [0.00521034560968036, 0.5252706354667532] |
| AUDIT_ONLY | (2,3)；不输出 operational risk |
| UNSUPPORTED | (1,4)、未知 state |
| UNSUPPORTED_EXTRAPOLATION | supported state 的闭区间外；nextafter 边界外也拒绝 |

Lineage、finite [0,1] score、state、interval、全局 M2 sanity 全部通过才可返回 conditional estimate；M0/M1 同样不能绕过。Sparse audit 有独立接口和 `supported=false`，不是 operational fallback。Range intersection 只是必要 support，不保证细 bins 重叠或精度。

CLI exit 0=`SANITY_PASSED_FOR_REVIEW`；exit 2=`FAIL_CLOSED`，可保留明确不可部署的 primary/三模型诊断 artifact；input/source/publication failure 为 exit 1。不存在 pseudo COMPLETE 或自动部署。即使 exit 0，`analysis_only=true`、`operational_deployment_authorized=false`、`classic_lambda_identified=false` 仍成立。

## 最终本地测试与 full-suite baseline

| 验证 | 结果 |
| --- | --- |
| Estimator math / I/O | 111 + 27 = 138 passed |
| Estimator + ITT | 158 passed，含 20 ITT tests |
| Phase-2 + artifact I/O | 395 passed / 6 skipped |
| Clean HEAD full suite | 615 passed / 69 failed / 74 errors / 6 skipped / 66 subtests passed |
| 最终 worktree full suite | 753 passed / 69 failed / 74 errors / 6 skipped / 66 subtests passed |

Full suite **没有全绿**。同一 Python 3.12.14 / NumPy 2.3.5 / pytest 9.0.3 / torch 2.10.0 / macOS arm64 环境，对本地 clean-HEAD clone 与最终 worktree 实跑；完整 143 个 `(nodeid, stage, category)` 失败/error events 集合相同，新增/消失均 0；138 个新增 tests 均通过。142 events 来自缺失 transformers/dotenv/gymnasium/openai/httpx2；1 event 是 HEAD 已存在的 legacy CLI import-closure assertion。

以上 753-passed 比较属于首次验收。仅有相同失败 node IDs **不足以排除 integration regression**；该 assertion 当时的 missing-module payload 从 baseline 28 增加为 current 30，包含两个独立 estimator modules。随后专项验收审阅了原契约并作最小测试修正，详见下节与更新的 [Regression Audit](phase2-terminal-estimator-local-regression-audit-v1.md) 及其 JSON；旧 CLI/core 不变。

Compile、tracked diff 与全部本批 untracked 文件 whitespace 检查通过。受保护文件未编辑或 stage。

## Import-closure 最终专项验收

原测试来自 Phase-1 commit `0d4b970`。`docs/specs/phase-1-classic7-tom-mainline.md` 的 Current module ownership 明确写“Every production module is reachable from the sole uns CLI import graph”；测试对所有 `werewolf/*.py` 与 `run_random.py` 做静态 AST import 可达性检查，唯一根为 `werewolf.cli`，根本没有扫描 `scripts`。它防止 orphan modules，不验证实际 backend 执行。

现有 Phase-2 runbooks 明确使用独立 script 命令：mapper final runbook 的 `python scripts/run_phase2_mapper_final_fit.py`、language smoke V3 runbook 的脚本命令、intervention pilot runbook 的 `python -m scripts.phase2_online_campaign` / `scripts.run_phase2_online_intervention_pilot`。它们在 HEAD 已存在；`tests/tom/test_cli.py` 同时固定 uns 的 ToM lifecycle command 集合。Estimator 新命令按当前独立分析约定提供 `main`/`__main__`，实际 import 为 `scripts.phase2_terminal_estimator → werewolf.phase2_terminal_estimator_io → werewolf.phase2_terminal_estimator`。无需把分析模块导入 gameplay 或改变 uns 的正式 command 集合。

判断为用户问题的情况 B：旧单根测试已不覆盖当前 Phase-2 的执行入口。28→30 是旧断言下真实新增违例，上一轮不能仅靠失败 node 相同将其排除；源码审阅后确认它属于测试入口契约滞后，没有发现 estimator 运行时接入缺失。

最小修正仅在 `tests/tom/test_boundaries.py`：scripts 作为 AST transit nodes；根集合为 uns 与具有**模块顶层、精确 `if __name__ == '__main__'` guard**的 scripts。无 module allowlist、无 production exclusion；所有原 production modules 继续必须可达。普通 helper、嵌套 guard、`!=` guard 不能成为根。保留旧 test node 名称，以保持失败对照稳定，docstring 明示扩展后的 contract。

同一 corrected policy 对实际 clean-HEAD 与 current 计算：missing 均为 `{werewolf.phase2_pilot_execution, werewolf.phase2_probe_value, werewolf.phase2_risk}`，current-minus-baseline 与 baseline-minus-current 均为空。单独原 assertion **仍失败**于这三个既有模块，不做本轮无关修复或豁免。五个新 focused tests 通过，证明新 root convention 不隐藏 orphan，并核验真实 estimator entrypoint。

专项修正后再次实跑 full suite：clean HEAD 为 615 passed / 69 failed / 74 errors / 6 skipped，current 为 **758 passed / 69 failed / 74 errors / 6 skipped**；新增的五个 closure tests 全部通过，143 个失败/error 的 node/stage/category 集合仍相同。Estimator + ITT 重跑 158 passed；Phase-2 + artifact I/O 重跑 395 passed / 6 skipped。旧 615→753 与原 policy 28→30 的完整证据保存在 regression audit JSON 历史中，未用新规则覆盖删除。

可在上述限定范围声明“Terminal Estimator 本地阶段无新增 integration regression”；不能声明 repository 无遗留 integration issue、full suite 全绿或正式 estimator 已 PASS。

## Phase-2 主线的只读状态审计

以下为本地源码/文档/pins 与用户交接的交叉核对，**不是本轮重新读取远端结果的证明**。Formal、ITT、Smoke-V3 和已完成 gameplay 的 empirical completion 依据用户交接；本地冻结身份与其一致。

| 阶段 | 当前边界与实际 repository 证据 |
| --- | --- |
| ToM final | sealed Qwen3 身份冻结，`docs/qwen3-final-fit.md`、`tom-gameplay-ablation-protocol.md`、`werewolf/tom/qwen3_final.py`；fit 87c2485a…、seal fcbb81ee… |
| ToM gameplay ablation | 冻结配对 protocol/runner；40 完成配对是交接状态，本轮不重算 win-rate。`tom-gameplay-ablation-protocol.md` |
| R2 / mapper | compact R2 runtime、固定 M3 full-development OOF mapper；score 仍为 p_tilde。`phase2-mapper-final-runbook.md`、`werewolf/phase2_mapper_runtime.py`；final manifest 8ab52997… |
| Action Contract | 三动作/eligibility/Redirect selector 冻结；`phase2-action-contract-v1.md`、`werewolf/phase2_actions.py` |
| Language V1.2 / Smoke-V3 | semantic version 与原 gate/58-case digest 冻结；`phase2-language-execution-v1-2.md`、`phase2-language-smoke-v3-runbook.md`、`scripts/run_phase2_language_smoke.py`；Smoke pin 22ed8116… |
| Online Pilot-T | canonical production intervention、ledger/backend/pins 机制已实现；`phase2-intervention-pilot-runbook.md`、online runner/server、campaign CLI |
| Formal120 / ITT120 | formal profile 目标 120、seed 730553457560960336、cap 240；120-row ITT pins 如上。`configs/phase2/online-terminal-pilot-formal-v1.json`、runtime pins、ITT script/support review |
| Support Review / Protocol | frozen source digests 如上；score intersections、model/uncertainty rules 已冻结 |
| Estimator implementation | 本地源码、独立数学/lineage/publication tests 与 CLI 验收完成；正式 fit 尚未运行 |

`phase2-current-status.md` 是较早阶段的历史快照，其中“Smoke未运行/qualification blocked/后续lambda”等内容不代表本交接的最新阶段。不改写历史 frozen 资料；当前 estimator 阶段以本验收文档、冻结 Protocol 和 Support Review 为准。已有 `phase2_risk.py` / `phase2_probe_value.py` 的 synthetic/protocol arithmetic 不能当作真实 lambda 或 Probe value；production 入口仍明确 fail closed。

下一依赖链：**server formal estimator fit → result review → terminal risk freeze → classic lambda identifiability review → Probe evidence → Probe value → final ThreeWayRouter → gameplay integration → paired final experiment → paper integration**。

本轮没有生产 classic lambda、posterior calibration、Probe pilot/value、router、thresholds 或最终 experiment。M2 仅估计 reduced-form working risk；p_tilde 非 calibrated posterior、theta 全 null、端点与结构条件尚未建立，不能把 intercept/slope 命名 lambda。

## 人工验收后的最短流程与结果清单

本地开发达到正式 estimator fit 前的合理边界；剩余真实 empirical gate 只能由正式 inputs 运行回答，不能由 synthetic PASS 替代。

人工验收 → 显式 add 本批 11 个 estimator/docs/tests 文件和 tracked import-closure test 修改（排除 wolf review）→ 用户 commit/push → server pull 到该新 commit → 在服务器依赖环境运行 estimator/ITT/Phase-2/artifact I/O tests，并记录 git status/HEAD/branch 与环境 → 从 repository root 执行一次：

```bash
python -m scripts.phase2_terminal_estimator \
  --formal paper-studies/paper-phase2-online-terminal-pilot-v1 \
  --itt paper-studies/paper-phase2-terminal-itt-analysis-v1 \
  --destination paper-studies/paper-phase2-terminal-estimator-v1
```

此命令本轮未执行。目标必须不存在；保留原 formal/ITT/runtime pins。新增 analysis commit 与历史 formal execution source 是两种 lineage，不重写历史 source pin。

正式运行后最少返回/核验：

- CLI exit code、summary status、manifest digest；校验全部 file-table bytes。
- Primary：N=120、58/62、arm means、raw difference、Neyman SE/95% CI；核对已冻结 raw means/difference，保留 uncertainty reference assumptions。
- Fisher：seed/B/PRNG/order/tail/plus-one p；独立 supplement，不作 M2 gate。
- M0/M1/M2：n=120、k=2/5/7、rank/df、columns、coefficients、HC1 covariance/SE/CI、sanity reasons、supported-domain predictions；无 clipping/fallback。
- Support：两闭区间、(2,3) audit-only、(1,4)/unknown/interval 外拒绝；loader 验证独立记录的 manifest digest。
- Lineage：实际新 analysis source commit/files、formal/ITT/reference/protocol/support pins、环境；`analysis_only=true` 与无 deployment authorization。

正式 coefficients、HC1 SE/CI、完整 lineage 下的新 Neyman uncertainty、Fisher 结果、正式 estimator digest 与 M2 PASS/FAIL 当前均尚不存在。若 FAIL_CLOSED，停止并审阅理论；不得加复杂模型、扩域、删稀疏行、clip 或同名重跑。即使 PASS，下一步仅是结果/理论审阅。
