# Phase-2J：单attempt F0控制与原生执行切片封存

结论：**CONTROLLED_F0_SLICE_CPU_VERIFIED**，同时 **F0_EXECUTION_BLOCKED_BY_CONTRACT**。本轮已实现单attempt Python入口、原生产生命周期上的固定计划控制、正常单边界返回及不可覆盖切片封存；未运行真实F0，未取得真实执行许可，不报告MANUAL_F0_READY或REAL_F0_PASS。

`ToM_D2_D3=NOT_USED`。没有权重加载、ToM/D2/D3计算、模型服务、SSH、真实游戏采集、随机行动或CAL/TEST。传递依赖会import torch，不等于运行模型。原systemd隔离不再作为本次F0前提，亦未开发其他隔离控制器。

## 1. 实时基准与改动范围

W3：`/Users/name_yuxiao/.codex/worktrees/data-acquisition-preprocessing/Untrusted_Network_Simulation`；本地`HEAD=ec008ef50dd10f2c678aa2d7e7439d31e333b80d`，分支`twd/mainline`。开始623个文件、28项未提交内容，index为空。会话快照未发现另一个活跃W3任务；持续文件核对未发现并行写入，不声称OS排他所有权。

GitHub只读核对的Mainline仍为`e5818d46fa1d02869e2c0f381a14f2b5ea5ff4a5`。首次沙箱网络读取被拒，随后经只读网络权限执行成功；没有fetch或更新本地引用。

本轮仅涉及6路径：

|路径|修改目的|
|---|---|
|`run_random.py`|新增默认None的`f0_slice_control`参数；独占original-mode接缝，正常返回切片，不以异常伪装成功|
|`werewolf/canonical_collection/production_runtime.py`|RuntimeFactory显式绑定F0控制；新增`run_f0_slice()`，普通`run()`拒绝F0模式|
|`werewolf/canonical_collection/collector.py`|在初始化/claim/循环前拒绝F0 factory及已有F0目录，防误用成功目标采集器|
|`werewolf/observer_dtrs/f0_slice.py`（新增）|复用原claim、Ledger/Store、PreCommitAudit和原native链；单attempt编排及受限切片封存|
|`tests/observer_dtrs/test_f0_slice.py`（新增）|28项人工CPU生命周期、身份、故障及停止测试|
|本文（新增）|实现、验收、来源范围与剩余合同|

未修改`CanonicalGameRecorder`、env、Actor、Parser、`realize_and_commit`、Ledger/Store/2G状态机、attempt ledger或冻结D1–D4。原28项未提交成果均保持字节、大小和权限不变；原623文件中仅上述3个原已跟踪入口按授权改变，其余620文件不变。最终626文件、34项pending，index、stash、HEAD及worktree引用不变。没有git add、commit、push、merge、reset或clean。

## 2. 实际入口与原生命周期

新增的实际Python入口为`werewolf.observer_dtrs.f0_slice.run_f0_attempt`，其Interface是：

```python
run_f0_attempt(factory=..., plan=..., destination=..., attempt_id=..., timestamp_utc=...)
```

`factory`为显式注入`F0SliceControl(F0SliceConfig(...))`的原`Classic7RuntimeFactory`。当前唯一接受的scope为`CONTROLLED_CPU_ONLY_NOT_REAL_F0_AUTHORIZATION`；`REAL_F0`等声明会被拒绝。**这不是已合格的服务器真实F0命令或新CLI。** 不能把人工scope用于真实服务而宣称合格。

配置固定study/qualification ID、一个原SpeechPlan、事前声明的eligible phases；计划只允许SELF_DEFEND(None)或NO_COMMITMENT(None)。默认phase人口为discussion和pk_discussion；PK人工验收单独事前指定pk_discussion，以测试该原生边界，不能解释为默认首狼PRE会等到PK。

实际顺序：

1. 控制对象核验当前合作PID/thread及一次性消费；输出目录必须全新、独占创建，不接受resume。
2. 用原artifact_io封存受控合同标记，用原`initialize_attempt_ledger`、`construct_attempt_claim`、`publish_ledger_record`持久发布ordinal=0的claim；只用第一个seed，不调用`collect`、`next_planned_attempt`或`recover_attempt_ledger`。
3. 原RuntimeFactory创建env/agents/CallAudit/Recorder。核对返回Recorder的plan及claim严格等于已发布对象。控制器只能绑定尚未start、无PRE/action的Recorder。
4. 原`run_random.eval`执行agent.reset→env.reset→**原Recorder.start**，然后原night、非目标发言等前序。未分配的PRE返回None，继续原Agent。
5. 每个发言PRE仍先走原`before_agent_act`。控制层从`public_pre_for_handoff`读取已冻结PRE；从当前actor合法observation判断存活狼人，在事前声明phase内只锁定第一个机会。所有其他观察者报告均由原collector一次采集，不重置、不删前序。
6. 原`generate_candidates`验证固定计划；原Store落盘OPPORTUNITY/ASSIGNMENT/STARTED，才调用原Actor。
7. 原Actor→Parser→语义核验→bind→原PreCommitAudit回调；原Store记录VERIFIED及ATTEMPT_RESERVED并确认fsync ACK；receipt始终`execution_permission=False`。
8. 原`recorder.commit_verified_speech`调用一次。成功返回后直接构造`F0SliceProduct`并从eval返回，不进入下一次Agent、报告或游戏动作。
9. 原Runtime包装器的`run_f0_slice()`处理F0结果/失败；外层关闭Store，使用原artifact_io对`slice/`不可覆盖封存后返回。封存失败抛出`F0SliceSealError`，异常携带已捕获的不可变product，不能重新执行。

对应源码：`run_random.py:54`、`:118`；`production_runtime.py:65`、`:80`、`:104`；`f0_slice.py:110`、`:121`、`:162`、`:184`、`:282`。准确AST定位及完整diff见配套证据。

## 3. 单边界结果与原生证据

`F0SliceProduct`不是`CanonicalGameProduct`，不会调用`finish`或降低`complete_evidence()`条件。成功受控切片状态为`BOUNDARY_RETURNED`；失败为`STOPPED_FAILURE`，不充分返回为`STOPPED_UNCONFIRMED`，矛盾证据为`STOPPED_CONFLICT`，没有符合资格机会时为`NO_ELIGIBLE_PRE`。本轮正例未完成游戏，winner为None，原完整游戏reader仍拒绝。

切片封存：`destination/slice/manifest.json`＋`restricted-slice.json`；使用既有canonical JSON、manifest/file_table、SHA-256、fsync及不可覆盖发布。不是完整canonical bundle，不生成canonical_success或虚假winner。原生claim有意保持open，没有新增/伪造AttemptTerminal枚举。F0标记阻止普通collect恢复该目录；**不得对其人工调用通用recover/collect**。

保存真实来源对象：study/qualification/config；完整原claim；game、boundary、prefix、canonical step、actor、phase/day；原PRE/handoff；固定计划及候选来源版本；assignment/intent ACK；原审计envelope和完整V1结果；完整前序typed native actions、公共事件、PRE和V1；目标action/event/V1；backend call IDs和预算；源码摘要、错误类别、局部停止状态。没有伪造native assignment token。

env事件/V1与typed action分别观察：若env已经publish、Recorder更新前失败，公开事件和V1仍记OBSERVED_LOCAL，而typed action为UNKNOWN。若缺失或重复/错身份，则保存全部原记录，返回UNKNOWN/CONFLICT，不修复、不重试。完整多动作不会被抽取为单计划依从。

assignment ACK丢失不会变成“未分配”：目标锁定状态保留，`assignment_persistence=UNKNOWN_AFTER_TARGET_LOCK`。封存原Store contract/journal/anchor的原始字节、大小及SHA（包括可能的未确认后缀），不重append、不重新确认。它们只是封存时的同机字节，**不是第二套Ledger，不是独立高水位，不是外部读回，也不是持久化真实提交证明**。

完整native action内嵌handoff，可能含当前发言者的私有报告支持；因此切片为**受限证据**。顶层资格目录及artifact目录为独占私有目录，不能将真实切片直接放入普通分享包或在线模型。本轮证据包只有人工数据，没有真实角色、报告、标签或权重；未把受限载荷加入SpeechPlan的公共两个字段。

Phase-2G接口针对完成态、已验证bundle；不能给本切片伪造完整bundle以强行调用。这里仅捕获原生现场对象，不重新实现2G完整来源验证。2G及其UNKNOWN归属合同原样保留并完成回归。

## 4. 已分配后的错误和停止

|观测|切片中的含义|禁止操作|
|---|---|---|
|Actor/Parser/语义失败，或ACK未确认|停止，保留已有assignment/Store；未走通native调用前路径|None回退、换计划、重抽、重采PRE|
|ATTEMPT_RESERVED后异常/None/超时|UNKNOWN_AFTER_INTENT，不等于NOT_COMMITTED|重新提交、F1、创建下一attempt|
|env事件已出现但Recorder缺action|public event/V1局部OBSERVED，typed action UNKNOWN|把缺日志当未执行|
|原生成功、最终研究封存失败|异常携带捕获product，保留Store与原现场|重跑整个入口|
|当前eval正常返回|只证明该合作调用栈不继续执行下一动作|据此宣称所有native能力执行者已停止|

新控制对象仅有同PID/thread合作约束；Store锁仅排除合作writer。原迟到native人工反例本轮在新生命周期复验：切片封存后，持原native方法的线程仍可追加一次提交。因此`all_native_capable_executors_stopped=UNKNOWN`，G3未通过。测试最终显式释放/join自己创建的人工线程，不影响共享进程。只有文本生成权限的远端请求与持env/native能力执行者不同，不能据此终止共享语言服务。

## 5. 实际CPU验收

最终**286 passed，0 failed，0 errors，0 skipped**，pytest8.67秒（外层10.07秒）：

|范围|数量|
|---|---:|
|本轮新增F0切片测试|28|
|项目外任务开始时eval原函数对照|1|
|Phase-2I fixed preconditions|13|
|PreCommitAudit|31|
|Verified Commit|26|
|Speech realization|26|
|Ledger / Store / 2G|46 / 44 / 55|
|原production collection / runtime evidence / profiles|7 / 6 / 3|

没有运行5835 PRE、D2/D3评分、G/π矩阵、全量CI或完整真实游戏。旧回归有自身人工终局fixture，不属于真实游戏执行；本轮新增正例使用原WerewolfTextEnvV0，不使用旧首句伪winner的_OneSpeechEnvironment。

本轮人工装配仅替换RuntimeFactory的build_runtime依赖，注入人工backend及脚本baseline Agent；**没有替换start/reset/before_agent_act/失败包装器，没有在night之后启动Recorder**。目标语言实现调用继承的原GPTAgent Actor，Parser和env/Recorder提交均为原实现。真实服务连接/后端语言质量未验证。

实际四个正例：

|预登记phase|固定计划|目标前原生动作|包含目标的PRE数|原报告观测总数|
|---|---|---:|---:|---:|
|discussion|SELF_DEFEND|8|5|35|
|discussion|NO_COMMITMENT|8|5|35|
|pk_discussion|SELF_DEFEND|19|9|63|
|pk_discussion|NO_COMMITMENT|19|9|63|

每个正例各1个factory/runtime、1次Recorder.start、1次目标Actor、1次native、1个claim；完整前序保留。原native本身可发布下一turn_start，这是目标提交的原生结果；没有后续Agent动作或下一个PRE报告采集。PK正例是事前指定的phase边界测试，不是按效果筛选人口。

默认关闭对照：编译本轮开始时原eval作oracle，以相同人工输入运行到首狼PRE前停止。事件、完整V1、PRE、报告、actions、Agent轨迹、请求、call budget/records、env RNG逐项相等；Python/NumPy RNG均未改变。该oracle及前像在项目外证据包，未写第二条生产执行链。

故障覆盖：Actor超时、Parser失败、完整多动作、intent ACK拒绝/丢失、assignment落盘ACK丢失、None/超时native、publish后返回丢失、env publish与Recorder更新之间故障、最终封存失败、重复控制/目录/旧claim、跨线程owner、native错actor/step、重复action冲突、迟到持native能力线程。失败后没有顶层再提交、原Agent隐式回退或第二claim。

首次运行1项失败是人工Agent缺少backend_id，被原合法来源检查正确拒绝；只修正fixture字段。之后17项初验、283项中间回归与最终286项分别留证，不累加数量。第一轮XML保留；没有把它改写为通过。

精确命令和退出码见execution.json/run_tests.py；JUnit和日志见scoped.xml/scoped.log。运行Python3.12.13，实际版本/导入见runtime-dependencies.json。主测试进程TCP尝试0；这不是OS沙箱。实际80模块、77不同项目文件的本轮受控源码指纹为`78cf460b63cdd3e9f66afb69945da59fc5ea590583d27c8f500e67b7e0605ea3`，不等于正式OID或生产资格。

## 6. 严格合同与供研究者决定的缩窄方案

**A　当前严格合同。** G1/G2的控制/正常停止已在上述受控原生产生命周期证明；真实执行仍缺：G3的native能力唯一owner及迟到执行者停止资格；G4 attempt-aware真实外部读回；G5独立可信高水位保管；完整已固定源码/依赖与真实Actor/Parser/原Agent/报告服务、预算、输出及数据权限。当前scope明确只接受人工CPU。封存同机原字节、PID/thread字段、Store receipt和观察到的公开事件均不能替代这些条件。不提供真实F0启动命令。

**B　未采用的研究者选项。** 若以后明确批准“首次全新、不可恢复、无F1”，可把结论限定为一次事前固定计划的局部执行/公开事件/V1证据；不宣称历史未回滚、外部严格一次提交、恢复安全、assignment原生token归属或所有执行者已被fence。仍至少需要唯一实际native能力owner、无旧未知执行者、新真实claim、合法PRE/固定计划、真实服务及全部前序预算、本地assignment/intent确认、未知即停、实际停止方式和受限证据封存。是否缩窄、是否另行开发合格真实入口必须独立审批，本轮没有自动采用B。

## 7. Mainline兼容及后续精确提交边界

对ec008与已核验e581的提交blob，run_random、RuntimeFactory、Recorder、collector、attempt ledger及原verified_speech_commit基线一致；本轮3入口增量不替换远端其他文件。Broker、Python3.12 CI和setup.py在e581与本地不同，本轮均未修改；Linux py312环境清单相同。未来整合必须保留远端修补，不能用落后W3整体覆盖。

建议待审提交次序：原C基础闭包（含尚未提交Ledger/Store/PreCommitAudit及Phase-2I）→独立2G只读联结组（如需）→本轮J六路径。A/B/H继续独立，不混合提交。精确清单、各文件SHA及增量diff在配套staging-plan.json/source.diff。没有执行暂存或提交；原HEAD不能代表当前新增运行源码，源码固定后须重新核验。

## 8. 最终十项回答与下一步

1. 已有完整的**受控CPU单attempt Python入口**；没有已批准真实服务CLI或真实执行许可。
2. 使用原生产Recorder启动及前序生命周期，原记录保留。
3. 已有正常单边界返回和独立partial artifact，不伪装winner/完整游戏成功。
4. 已保存原claim/PRE/handoff、计划/ACK、原envelope/V1、action/event和前序；外部归属、独立保管、全执行者停止仍UNKNOWN/NOT_QUALIFIED。
5. 已证明本入口不调用普通collector、不循环、不恢复，正反例均只有一个claim；不宣称任何任意外部调用者都无法另启进程。
6. 合作对象/线程/新目录约束已验证；真实owner fencing及迟到能力停止未解决。
7. G4/G5仍无合格真实证据/实现。
8. 如选择缩窄保证范围，必须研究者另行审定；本轮未放宽。
9. 六文件增量、286项实际通过、原成果保护均有证据。
10. 直接入口基线兼容；远端Broker/CI/setup.py差异保留待后续安全整合。

下一步只需研究者审阅本切片及严格/缩窄保证范围，并决定唯一native能力owner与真实证据门槛如何满足。**本轮到此停止**，不继续扩展Store/Hook/审批包装、不启动真实F0或正式行动实验。将来F0通过也不证明SpeechPlan行动效用、两臂因果效果或D3表示增量。
