# Phase-2D：研究侧 Action Evidence Ledger 最小受控实现

状态：**CONTROLLED FIXED FIXTURE ONLY / NOT AN EXECUTABLE G2 CONTRACT**。
当前实际W3基点：`twd/mainline@ec008ef50dd10f2c678aa2d7e7439d31e333b80d`。
该HEAD较Phase-2C推进两项既有ToM运维提交；本轮没有合并分支。原12项S1a/P0/Phase-1成果保持字节不变。
不改变D1–D3、SpeechPlan、Actor/Parser、Recorder、默认Agent或研究目标。

## 已审阅的提交边界

- `realize_parse_verify`仅进行原Actor生成、Parser解析和完整单项语义核验，不获得env/recorder提交句柄；生成/解析可能消耗调用预算并失败。
- `realize_and_commit`在已有pending PRE/handoff上调用上述原流程，再绑定原envelope并调用`commit_verified_speech`；没有新建PRE或重复采集报告。
- Recorder先复制/stage环境及提交证据，正常校验错误发生时不发布原环境；随后先`env._publish_verified_speech(staged_env)`，再更新recorder字段。这不是env、recorder、研究journal、远端执行器之间的跨进程/跨存储事务。
- 因而pending未清、日志缺失、异常或超时均不能单独证明“尚未提交”。本轮不修改该核心，也不补第二个提交器。
- 原Shadow `ReceiptLedger`是capture/outcome专用的单writer持久化模块。本轮不改变它的状态或复用其身份冒充行动记录；生产持久化层尚未实现。

## 最小接口

实现：`werewolf/observer_dtrs/action_evidence_ledger.py`。

```python
new_records = extend_ledger(records, ledger_id=..., operation_id=...,
                            kind=..., payload=...)
view = validate_ledger(records, ledger_id=..., expected_head=...)
```

两个接口均为纯函数，无文件、网络、RNG、Agent、env或commit调用。`extend_ledger`只返回前瞻记录副本；它不写盘、不发放可执行permit、不调用动作。相同operation_id和完整内容重试返回原历史；同ID异内容拒绝。
`LedgerView`不可变，分别保存执行、提交、日终证据状态。返回的记录dict是副本，调用者改写后再次验证会触发摘要失败；不声称普通dict物理不可变。

每条记录绑定version/scope/ledger_id/operation_id/连续sequence/previous digest/payload/record digest。
OPPORTUNITY绑定protocol_ref、game/attempt/boundary/prefix、canonical step、actor/day/phase、公共历史、capture摘要和完整候选快照。
ASSIGNMENT用具体opportunity record digest作前驱，绑定所注入候选ID、支持集、treatment_ref；没有重新抽签函数。
EXECUTION绑定assignment和唯一STARTED记录；提交意图绑定VERIFIED记录；外部读回绑定reservation/token及完整同一PRE身份。
DAY_WITNESS绑定assignment及game/day，MISSING升级OBSERVED须明确引用被补充的旧记录。

候选快照中的`plan/expected_v1`由可信捕获者提供；人工fixture复用原`generate_candidates/expected_action`生成。纯validator只验证记录及V1自洽，**不证明捕获者确实读取了合法PRE、候选完整、原始语义映射或外部见证真实**。这些须由正式reader及独立准入验证，不能靠任意digest字符串取得资格。
本轮scope仅允许`INJECTED_FIXED_NOT_RANDOMIZED`。可选的精确分数概率声明只检验正值与和为1；夹具实际为None，既不是真实随机概率，也不构成IPS资格。

## 允许状态转换

| 记录 | 必要前驱/条件 | 记录后的意义 |
|---|---|---|
| OPPORTUNITY | 新的(game,boundary)及opportunity ID | 捕获记录；不等于游戏机会准入 |
| ASSIGNMENT | 已有opportunity；每个PRE至多一项assignment | 原assignment永久保留；不允许失败重抽 |
| EXECUTION STARTED | assignment；尚无执行、提交或日终记录 | 开始证据；不自动执行Actor |
| EXECUTION VERIFIED/FAILED | 同一STARTED digest；终态一次 | VERIFIED必须完整actions等于捕获的单项预期；多动作不拆分 |
| COMMIT ATTEMPT_RESERVED | VERIFIED；此前无提交意图；日终仍PENDING | 单一历史中一次意图；只有外部持久化并确认后，受控driver才可能调用假对象 |
| COMMIT_UNKNOWN | 已有reservation | 等待严格外部读回，停止新增执行、提交和回退 |
| COMMITTED/NOT_COMMITTED | 同一reservation/token和PRE的外部读回；后者可用于尚无意图的执行后故障 | 记录提供者声明的读回；不重新授权提交 |
| DAY MISSING | 已有assignment | 技术缺失，席位必须None，不填零/空放逐 |
| DAY OBSERVED | 已有assignment；PENDING或明确补充MISSING | 公开事实；[]才是观察到无放逐；不改变提交未知状态 |

执行、提交、日终是**三个证据维度**，不是新的游戏状态机。Actor失败的assignment仍可获得独立日终见证；记录到达次序可能晚于事实发生次序。日终见证不能自动证明某计划已执行。已有日终记录后，不允许新增执行STARTED或提交意图。
同一game/day多个PRE共享同一公开见证digest；冲突见证拒绝。所有assignment均保留，不按成功、可观测或依从性过滤，不能把PRE引用数当独立随机试验数。

## 持久化与严格一次边界

本模块不提供生产writer。独立人工`tests/observer_dtrs/action_evidence_fixture.py`在pytest临时目录独占创建journal，逐条canonical JSONL写入并fsync；先确认意图写入再调用独立FakeWorld。FakeWorld刻意没有去重，以暴露ledger的能力边界。
fixture是单writer测试夹具，没有真实进程间锁或真实owner协议，不能用于生产文件。恢复需测试拥有者明确提供预期head；不自动截断、修复、选择有效前缀或覆盖旧文件。

**最多一条记录中的提交意图 != 外部严格一次提交。**

- 有效前缀在没有带外head时仍可能通过；有效fork也可以各有一次意图。测试明确展示两个人工直接调用可发生，view始终`external_exactly_once_proven=False`。
- head参数只检查与调用者提供值相等，不认证该值来源、不证明fsync/电源故障后的持久化。正式高水位/外部锚点、writer排他及环境fencing尚未实现。
- 提交已发生、COMMITTED日志写入失败时，耐久链仍可能只含ATTEMPT_RESERVED。不能把没记录到成功当作未提交；恢复先读回，再补同一attempt的COMMITTED及`had_persistence_failure=True`，不重做发言。
- 失败的写入本身无法同时成为该次失败journal中的耐久事件。人工case另存operational failure证据；没有虚构一条已落盘的失败记录。
- 截断半帧、未知head、不同PRE/token读回、重复/冲突operation或前驱错误均停止，不自动修复。

F0/F1、真实目标、机会选择、比较动作、概率和继续策略均未冻结；本轮不实现任何回退。读回NOT_COMMITTED也不会自动调用原Agent。人工fixture中固定NO_COMMITMENT仅是注入stub，不是推荐动作、失败兜底或正式比较方案。

## 已执行受控验收

最终72项通过：46项本模块测试、26项原Actor/Parser脚本化backend回归，0失败/错误/跳过。
另生成12个人工故障证据case：正常、Actor失败、Parser失败、多动作不依从、两类COMMIT_UNKNOWN、提交成功但日志失败、半帧、重复分配、共享日终、执行失败但日终可见、有效fork能力限制。
测试不调用真实模型或env/recorder，也不运行游戏。故障case使用新实例重放和人工异常，并未测试真实OS进程崩溃、断电或跨进程环境事务。

开发中最小修正了一处**本轮新fixture**的幂等回执：重试早期operation应返回该原记录，而不是当前head；新增回归后通过。外部case报告脚本首轮缩进错误退出1，保留失败日志，修正后完整运行；验证器和原有源码未受影响。各中间测试日志保留，不将通过数累加。

剩余真实集成风险：独立捕获/读回真实性、完整候选准入、带外head的持久保管、并发writer/fork、迟到Actor/远端执行、env发布与Recorder更新间的崩溃窗口、研究日志失败后的外部协调，以及全assignment结果观察者。正式运行前仍须独立审阅；当前状态是**受控合同与纯validator验收通过，真实集成及Action Scorer未准入**。
