# Phase-2I：首次固定SpeechPlan执行链的最小修补与真实F0阻断

**最终状态：F0_BLOCKED_WITH_EXACT_GAPS。** 本轮完成两项已复现的C类前置/未知返回修补及定向CPU验收；没有真实可启动的单次F0入口，没有REAL_F0_PASS。`ToM_D2_D3 = NOT_USED`：未运行模型、读取权重或调用认知规划；不表示依赖图完全不import相关Python包。

## 1　实时基准与本轮修改

W3：`/Users/name_yuxiao/.codex/worktrees/data-acquisition-preprocessing/Untrusted_Network_Simulation`，`twd/mainline@ec008ef50dd10f2c678aa2d7e7439d31e333b80d`。开始621项文件、26项未提交成果；index为空。2026-10-29 16:21（Asia/Shanghai）只读GitHub确认远端仍e5818d46fa1d02869e2c0f381a14f2b5ea5ff4a5。没有fetch、同步、切换、merge或新worktree。会话列表未发现另一活跃W3任务，文件持续核验未发现其他写入；不声称已建立OS排他所有权。

仅改C类`scripts/verified_speech_commit.py`，相对**本轮开始的工作字节**+10/-1；不能把相对HEAD的全部预提交审计历史diff归为本轮新增。原C文件完整前像已保存。另新增：

- `tests/tom/test_fixed_f0_preconditions.py`，13项人工CPU测试；
- 本文。

原26项中只有上述C文件按授权做必要增量；其余620项原文件及所有A/B/H成果不变。不改Actor/Parser、Recorder/env、run_random、Ledger/Store/2G状态机或D1–D4。没有新增owner服务、提交器、F0 CLI、回退、重试或随机分配。

## 2　真正修复的两个局部边界

### 2.1　当前PRE候选资格前移

旧审计路径只先检查绑定摘要；格式合法但当前目标尚未发言、因而不属于合法候选的OPPOSE，仍可先调用Actor，再被审计回调拒绝。本轮以原人工PRE及脚本Actor实际复现：1 failed，失败点为`actor.backend.calls != []`，native未提交。这是缺少审计opt-in入口前置条件，不意味着原Actor应该承担候选生成职责。

修补在原`PreCommitAudit._claim`后、Actor前，lazy复用原`generate_candidates`进行完整公共payload membership检查。非法输入也消耗原审计对象，不允许同对象重试；不裁剪或重排候选。默认关闭路径不执行该检查，仍与原oracle相同。

该lazy导入会传递import `counterfactual_tom`及torch；没有读取权重或运行ToM，实际依赖版本必须进入未来F0源码/环境身份，不能宣称F0完全没有相关import。

### 2.2　未确认native返回不能冒充“未分配”

旧审计编排会把异常的native返回None原样透传。原original-mode Hook把None解释为未介入，若未来控制层直接返回该值，存在隐式原Agent路径风险。本轮第二个red测试实际证明`DID NOT RAISE`。

修补仅对审计opt-in结果要求原生四元tuple；None、空tuple或其他未确认类型显式抛出`PreCommitAuditError`。它发生于native调用之后，**不证明未提交**，不能重试、回退或写成NOT_COMMITTED。合法四元tuple同样不是提交成功/归属证明；正例另核验原action/event/V1。

原Actor重试、Parser、bind和唯一native调用不变。没有把本地ACK变成执行许可。

## 3　按真实调用链核对首次F0的9步

|步骤|已有真实代码|本轮证据及未完成项|
|---|---|---|
|原生claim先于backend|collector.collect:223–237及RuntimeFactory|原durable no-replace claim可复用；不是全局native能力owner证明|
|原Agent到首次合格狼PRE|run_random.eval:88–99、111–119|original-mode hook在收集后、Agent前存在；生产工厂未绑定F0控制层|
|复用PRE/handoff|runtime.before_agent_act:228、public_pre_for_handoff:292|原报告只采集一次；CPU正例报告数、PRE及handoff保持；不得重置/补造|
|固定合法计划|原SpeechPlan/generate_candidates；本轮前置检查|SELF_DEFEND/NO_COMMITMENT人工正例；另一合法计划不能替换已绑定assignment；真实计划尚未由研究者指定|
|assignment/intent落盘|现有Store/PreCommitAudit|CPU证明ASSIGNMENT/STARTED先于Actor，VERIFIED/ATTEMPT_RESERVED及fsync ACK先于native；receipt仍execution_permission=False；生产保管主体缺失|
|原Actor/Parser/native|realize_parse_verify→bind→CanonicalGameRecorder.commit_verified_speech|CPU精确单动作与native一次；真实后端身份/参数/预算尚无本轮证据|
|action/event/V1/PRE见证|runtime.after_env_step:345、2G联结|内存可严格核验；2G仍无native assignment token，归属UNKNOWN；不是外部持久读回|
|指定边界停止和保存|现有run_random及collector不足|没有F0切片完成分支；complete_evidence要求winner，失败路径会继续后续claim，见下文|
|未知即停|原异常传播＋本轮None防护|单个调用/人工Hook可停止，不能证明所有持native能力执行者停止；全部assignment终态捕获仍需真实控制层|

`run_random.eval(..., online_pilot=...)`是Python扩展点，不是已具备F0资格的手动入口。`Classic7RuntimeFactory._RecordedRuntime.run:63–83`只调用正常run_game，无F0参数；`uns collect`也无固定计划、F0停止或资格ID选项。本轮没有修改这条生产收集路径。

## 4　不是“再加一个CLI参数”就能修好的具体阻断

### G1：真实控制入口与正常前序轨迹

旧`FixedOwnerFixture`在第65行要求`not recorder.submitted_gameplay_actions`且`len(recorder.prefixes)==1`；它在night/人工前序发言后才start Recorder，并替换reset/start/before_agent_act/失败包装器。真实生产Recorder在游戏启动即start，首个狼人PRE前通常已有night动作和其他玩家发言记录。**不能复制该fixture当真实runner或删除断言后改名生产owner。**

下一次必要集成应复用原original Hook，绑定真实已存在claim/PRE及前序记录基线，不重新采集报告；固定计划由批准配置提供，只消费该assignment一次。相关生产工厂参数传递和停止/封存合同尚未实现。这里确实需要一个经审阅的最小可信控制层，而不是第二个Actor/提交器。

### G2：停止边界、封存与collector行为

原`CanonicalGameRecorder.commit_verified_speech:328–343`先stage验证，后publish env，再更新Recorder；发布后异常可能留下env/Recorder不一致。`complete_evidence:388–416`要求完整游戏winner。本轮新增测试直接使用**原失败包装器**证明：一步native之后complete_evidence拒绝；failure partial保留public_events、PRE、V1及raw submitted_actions，但不直接保存typed SubmittedGameplayAction集合或assignment/token。

`run_random:157–158`把停止异常包装为CanonicalAttemptFailure；`collector.collect:243–297`发布失败后继续直到成功目标。故`target_success_count=1`不等于一次F0，也不保证第一失败后不开始下一局。不得用虚假winner封存，或用普通collect命令让“资格停止”悄悄变成失败采集及下局执行。

需在真实控制入口中事前明确单attempt停止结果、目标PRE/action/event/V1证据的原生读回/封存和失败终态。应复用原recorder与failure evidence，不另写提交核心；若需要新的生产F0结果合同或原生来源扩展，必须单独审阅。当前未实施。

### G3：唯一native能力owner与停止资格

PreCommitAudit只防同对象/同PID线程重复。Store锁只排除合作文件writer。原生claim和空新目录不能证明其他进程/线程没有env/native能力。人工迟到执行者反例本轮再次通过：调用方超时后持有原native方法/envelope的线程仍可提交一次。

普通进程运行可以作为候选部署方式；本轮不要求重建systemd。但必须事前明确唯一owner、哪些线程/进程能持native能力、如何停止本任务执行者及保留失败证据。PID消失、客户端超时或日志缺失不能单独替代该证明；远端语言生成器只持文本生成能力时，也不能误当native执行者或据此kill共享vLLM。

### G4/G5：现行合同保留的外部读回／独立高水位

旧F0签审已区分：对首次全新不可恢复native调用，这两项不是逻辑上的普遍最低条件；但当前待审方案明确保留为额外门槛，且waiver_approved=false。当前任务没有批准缩窄保证范围。因此它们在**现行F0合同下仍阻止执行**，不能因放弃systemd或不做F1自动删除。

2G当前没有submission→assignment正例契约，返回NATIVE_SCHEMA_HAS_NO_ASSIGNMENT_TOKEN；Store外部readback只接受人工provider声明，现有同机两个目录不能构成真实独立保管。没有发现可以直接复用的真实保管/读回服务，不新增模拟服务补齐。不需要完整日终或终局结果来做单边界F0，但全部assignment失败/未知/缺失必须保留。

### G6：源码、实际服务、预算及权限

当前代码未提交，且W3落后远端；没有可用于真实F0的完整已批准源码OID。Actor/Parser、PRE报告和到达首狼PRE的原Agent均会使用真实backend，不能只预算目标一次发言。须绑定实际服务模型身份、配置/重试/调用预算、全部前序调用、唯一输出/claim和证据保管权限。

Final-ToM、Final-CAL、旧B2资格以及此前rc226**不是本次NOT_USED固定计划F0的准入前提**。本轮不以旧隔离失败增加新的F0门槛，也不启动任何语言服务或模型。

## 5　CPU实际结果与故障证据

最终：**172 passed，0 failed，0 skipped，0 deselected，pytest 4.69秒**。

|模块/fixture|通过|
|---|---:|
|本轮fixed_f0_preconditions|13|
|原项目外单owner Hook fixture，字节未改|21|
|原PreCommitAudit|31|
|原Verified Commit|26|
|原Speech realization|26|
|原Phase-2G纯联结器|55|

没有运行G/π对照、ToM/D2/D3数学全套或全仓库测试。人工Actor/Parser与局部env状态变更不是新游戏采集/完整游戏，也不是服务器执行。没有模型forward；ToM/B2被新正例拒绝调用。主测试进程TCP连接尝试0；spawn进程仅做原Store锁竞争。Python级网络/模型拦截不是OS沙箱。

13项新增覆盖：非法候选Actor前拒绝；两计划×普通/PK；assignment不能改为另一个合法计划；完整多动作V1拒绝；三种未确认native返回停止；真实Recorder包装单边界不等于完整封存。

原21项复验保留15份case evidence：Actor/Parser/审计拒绝/ACK不确定均native调用0；ACK已落但丢失仍AWAITING_EXTERNAL_READBACK；native未知/None调用1且不重试；成功后日志失败观察到1次原生提交但研究状态仍未知；合作owner竞争、旧attempt/owner拒绝；迟到持native能力反例仍可发生。人工fixture的“0个记录”不能推断现实未提交。

首次测试收集因本地默认cwd错误导入Desktop旧版本，未运行测试；随后显式W3 cwd重跑得到真实red。该收集错误、两个red和所有绿灯日志全部保留，不把它算模型或生产失败。最初9项修补验收通过；最终以13新增＋159相关=172为准，不累加中间通过数。

运行环境Python3.12.13、pytest9.1.1、NumPy2.5.1、Torch2.12.1，Mac CPU；禁字节码/cache，临时目录在项目外。精确命令、JUnit、导入与网络记录在外部证据包。

## 6　源码身份及Mainline兼容

本轮受控执行身份（不是正式OID）：`ce4b668ce6fd0a1cf5920878fafc20f5d10c8cded5342f66a19a8ac92a8a1c87`。实际加载70项目模块、67不同项目文件，全部逐字节指纹核对W3，包括修改后的C入口。旧619文件CPU身份不能代表本轮新字节。

仅检查11项直接入口/依赖：ec008与e581 committed blob中10项一致，Broker不同。没有改Broker，也没有重新运行其服务；未来必须保留e581内部请求ID摘要修补。Baseline、Source Identity实现均不修改；其资格是各自作用域的，不等于覆盖新的F0编排。尤其现有D4快照枚举不含本脚本，不能拿旧D4摘要冒充完整F0身份。

原26项仍保留；只有明确授权的C脚本有上述增量。新增测试、本文，故最终623文件、28项pending。原620文件SHA/大小/权限不变；C完整前像及增量diff保留。index原始字节、stash、HEAD及worktree引用不变。未commit/push/merge。

## 7　精确提交/手动运行边界

当前**不能提供可执行的真实单次F0命令**，标记`NO_SINGLE_F0_COMMAND_AVAILABLE`。有函数接缝不等于有满足合同的CLI；不提供虚构的`--f0`/`--fixed-plan`/`--attempt-id`参数，禁止以uns collect代替。

本轮C增量为脚本＋新测试＋本文3路径；本脚本依赖尚未提交的Ledger/Store，因此不能只提交脚本使服务端缺模块。后续可审阅的C基础闭包11路径：脚本、Ledger/Store实现及对应2测试、人工fixture、旧PreCommitAudit测试、新测试、2D/2E文档、本文；2G的实现/测试/文档3路径单独保留为离线联结组。**不把A/B/H混进C提交**；精确清单见外部staging-plan.json。没有执行git add。

源码应先按文件级diff保留最新e581修复后固定，不能直接把落后W3整体推到远端。服务器更新仅由用户手动、在工作区干净且人工确认不占用后进行；独立的SERVER-MANUAL.zh.md只给真实存在的只读检查与后续有条件源码同步步骤。当前没有F0执行命令、有效资格ID、指定真实计划或运行许可。

## 8　下一项最小必要工作与停止

下一项不是更多Store/人工规划测试，而是**审定并实现原original-mode接缝上的单attempt F0控制和切片停止/封存合同**，绑定真实原生claim及现有完整前序轨迹，复用原realize/bind/native并保全所有assignment。与此同时必须保持现行读回/高水位/停止门槛，或由研究者单独明确批准缩窄的首次不可恢复保证范围；Codex不自行降级。

真实服务、控制主体、停止方法、来源与权限满足后，才可能给出已存在入口的精确手动命令。F0将来通过也只支持固定计划的真实执行资格，不证明行动收益、两臂效果或三域表示增量。本轮不开始随机试验、CAL/TEST、模型训练或胜率实验。
