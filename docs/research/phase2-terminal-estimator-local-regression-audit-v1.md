# Terminal Estimator Local Regression Audit V1

状态：本地回归审计完成；**full suite 未通过**。此文档不包含正式 estimator 拟合结果，不替代服务器测试或 empirical gate。

## 比较方法

基线来自本地隔离 clone：`git clone --local --no-hardlinks --branch twd/mainline`，checkout `/tmp/phase2-clean-head-baseline-gadxl67p`，HEAD `5a6f2a657d63bb09791e052cfc51de47a2557556`，branch `twd/mainline`；初始 tracked/staged working tree clean，运行后 `git status --short` 仍为空。未使用“排除新增测试”冒充 clean-HEAD baseline，也没有网络/remote 操作。

基线与当前 checkout 使用完全相同 Python、PYTHONPATH、pytest 命令与只读 report collector：

```text
/Users/name_yuxiao/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m pytest tests --continue-on-collection-errors -q -p phase2_full_suite_audit
```

精确环境变量、完整命令、checkout source fingerprint、日志 SHA256、所有 failed/error node IDs 与阶段/category 均保存在 [phase2-terminal-estimator-local-regression-audit-v1.json](phase2-terminal-estimator-local-regression-audit-v1.json)。环境为 Python 3.12.14 / NumPy 2.3.5 / pytest 9.0.3 / macOS arm64；使用已有本地 dependency cache，无安装、shim 或测试排除。

## 实际结果

| 运行 | Pytest terminal summary | Exit |
| --- | --- | --- |
| 真正 clean HEAD | 615 passed / 69 failed / 74 errors / 6 skipped | 1 |
| 当前完整工作树 | 758 passed / 69 failed / 74 errors / 6 skipped | 1 |

失败/error 共 143 → 143 个 report events；完整 `(node_id, stage, category)` 集合相同。新增 0、消失 0；passed 增加 143。Pytest 还单独报告 subtests；不把 terminal summary 相加当作独立 observation population。

这里的 142 个 dependency failure **是 report events，不是 142 个 distinct tests**；它们包含 collection、setup 与 call 阶段。完整阶段计数及 skipped node IDs / 原因也保存在 JSON。

Dependency events 为 collection=35、setup=39、call=68；另外一个 call event 是既有 import-closure assertion。

| 原因类别 | Clean HEAD events | Current events |
| --- | --- | --- |
| baseline_cli_import_closure_assertion | 1 | 1 |
| missing_dependency:dotenv | 17 | 17 |
| missing_dependency:gymnasium | 8 | 8 |
| missing_dependency:httpx2 | 1 | 1 |
| missing_dependency:openai | 4 | 4 |
| missing_dependency:transformers | 112 | 112 |

除既有 import-closure assertion 外，剩余失败/error 均在 traceback 中明确报告缺少 `transformers`、`dotenv`、`gymnasium`、`openai` 或 `httpx2`。这属于当前本地环境的缺失依赖证据，不推断完整服务器环境也会失败。

两次运行的 6 个 skipped node / 原因相同：

| Node ID | Stage | Reason |
| --- | --- | --- |
| `tests/phase2/test_canonical_commit_adapter_integration.py` | collect | Skipped: could not import 'gymnasium': No module named 'gymnasium' |
| `tests/phase2/test_online_campaign.py` | collect | Skipped: could not import 'gymnasium': No module named 'gymnasium' |
| `tests/phase2/test_online_runner_integration.py` | collect | Skipped: could not import 'gymnasium': No module named 'gymnasium' |
| `tests/phase2/test_online_server.py` | collect | Skipped: could not import 'gymnasium': No module named 'gymnasium' |
| `tests/phase2/test_mapper.py::test_sealed_oof_provenance_regression` | setup | Skipped: sealed OOF root is not a quick-test fixture |
| `tests/phase2/test_offline.py::test_sealed_development_population_regression` | setup | Skipped: sealed development inputs are not local test fixtures |

## 已有断言的精确边界

`tests/tom/test_boundaries.py::test_all_production_modules_belong_to_current_cli_import_closure` 在真正 clean HEAD 已失败。原 policy 仅把 `werewolf.cli` 作为 root；对两份源码应用同一原 policy 得到 baseline=28、current=30 个未覆盖模块，额外项正是两个 estimator 模块。本次没有删除旧证据：JSON 的 `prior_original_policy_audit` 保留此前 615→753 passed 比较及全部失败/skip记录。

本轮明确修改 tracked test `tests/tom/test_boundaries.py` 的 root 发现规则：仍保留 `werewolf.cli`，并加入具有 top-level 精确 `__name__ == '__main__'` guard 的脚本；其他 scripts 只作 transit nodes。没有 production 模块豁免、排除名单、硬编码入口名单，也未把所有脚本无条件当 root。新增 5 个 focused checks 覆盖非入口脚本不得掩盖孤立模块、真实脚本可达性与 estimator CLI。Production estimator/CLI/runtime source 均未因这次 policy 修正而修改。

为区分测试规则变化与源码回归，把**当前同一个 `_production_import_closure` probe** 以 AST 只读方式应用于隔离 baseline 和 current：

| 同一 closure policy | Clean HEAD源码未可达数 | Current源码未可达数 | 新增 |
| --- | --- | --- | --- |
| 原 `werewolf.cli` 单入口 policy | 28 | 30 | 2 estimator modules |
| 修正后的真实可执行入口 policy | 3 | 3 | 0 |

修正后的双方剩余模块完全相同：`werewolf.phase2_pilot_execution`、`werewolf.phase2_probe_value`、`werewolf.phase2_risk`；baseline/current roots 和完整 module sets 均保存于 JSON。当前 full-suite 的原 assertion node 仍因这三个既有 orphan 而失败，没有伪装通过。

真实 clean-HEAD pytest 仍使用其原测试源码，current 使用显式修正后的测试源码；所以 tracked fingerprint 不再相同，唯一 tracked diff 正是此测试。结论是“相同入口规则下无新增 production orphan，也无新增失败 node”，不是“pytest assertion payload 逐字相同”。

## 结论与限制

本轮 Terminal Estimator 变更在相同本地环境下没有增加 failed/error node IDs；所有既有失败都有真实 clean-HEAD 对照。缺失依赖仍使相应路径无法在此环境验收，既有 CLI closure assertion 仍未通过，故不得报告 `full suite passed`。

没有连接服务器、访问服务器数据目录、运行真实 LLM/gameplay、拟合正式数据或 commit/push。未来服务器需在正式依赖环境执行相应 tests；本地 synthetic tests 与这份比较都不能预判真实 M2 sanity PASS。
