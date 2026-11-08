# Phase-2G：attempt-aware 原生证据纯联结器

状态：**CONTROLLED_PARTIAL_LINK_ACCEPTED / REAL_LINK_NOT_RUN / NOT_EXECUTION_AUTHORITY**。
源码基点：W3 `twd/mainline@ec008ef50dd10f2c678aa2d7e7439d31e333b80d`。
本轮不改Phase-2D/E、D1–D3、Phase-1、S1a/P0、Actor/Parser、Recorder或默认Agent。

## 输入与边界

```python
link_native_evidence(
    records, ledger_id=..., assignment_ref=..., plan=..., claim=...,
    bundle=None, expected_bundle_digest=None, attempt_terminal=None,
) -> NativeActionLink
```

`records`仍由原`validate_ledger`验证；当前只准入原固定注入研究合同，未新增正式随机assignment schema。
`bundle`必须是原授权reader返回的`VerifiedCanonicalGameBundle`，其artifact须为原`VerifiedArtifact`。
调用方先取得合法权限及可信来源，必要时在独立许可环境中运行原生reader；本函数不打开目录、不调用reader/回放/恢复。
`expected_bundle_digest`须由调用方可信保管提供；自己重新计算一个摘要不自动成为独立锚点。
Python对象类型和digest不能证明任意调用者确实运行过原reader；这是明确的可信输入合同，不是OS/密码学认证。
人工测试手工构造原生值对象及manifest，**不是**真实完整archive的资格验证。

先绑定plan/claim/manifest的game、attempt、collection、ordinal、seed、源码和runtime来源，严格拒绝额外manifest字段。
对公共history、PRE、V1、submitted actions四个选定分区重算原canonical JSONL字节数/SHA，对照manifest。
其余分区的全树验证仍由原reader负责，不另建一套bundle读取/发布体系。

## 独立结果

每项为`OBSERVED / UNKNOWN / CONFLICT`，附reason及规范JSON证据；缺失不是否定。

| 项目 | OBSERVED的含义 | UNKNOWN/CONFLICT边界 |
|---|---|---|
| submission | 同一可信attempt容器中找到唯一同PRE原生提交，handoff/action/event/text严格一致 | 缺action为UNKNOWN，绝不推断未提交；错PRE、重复及不一致为CONFLICT |
| assignment_link | 当前无可准入的正例 | 原生action无assignment/token字段，保持UNKNOWN；研究侧相反提交声明/执行文本/V1可产生CONFLICT |
| semantics | 已观察完整V1及其分类 | 注释缺失为UNKNOWN；错绑定/摘要为CONFLICT；多动作、空动作、单动作不依从并不抹除提交 |
| day | 原`_resolved_exiles`验证了目标日最终普通投票/必要PK结算 | 未完成投票/PK为UNKNOWN，重复或矛盾结算为CONFLICT；第一次平票空放逐不算日终 |
| completion | 当前原生记录不足，不提供OBSERVED分支 | 可核验CANONICAL_SUCCESS与bundle联结，但它不是完整终局证明；错terminal/replay来源为CONFLICT |

`semantics.classification`：EXACT_SINGLE、MULTI_ACTION、EMPTY_ACTIONS、SINGLE_MISMATCH、PARSE_ERROR。
分类仅评价完整语义依从，不是行动效果。原候选生成和expected_action只用于验证已声明计划合法且语义映射一致，不选择任何动作。

即使submission与day均OBSERVED，assignment仍可能UNKNOWN；研究Store中的COMMITTED/token不会被反向合成为原生token。
COMMIT_UNKNOWN不因看见同文本而获得新的执行许可。研究侧明确NOT_COMMITTED与已观察原生提交矛盾时，只标记联结冲突。
同一尝试可有共享日终见证，这不是独立随机化样本数。

## 终局的明确未完成项

`DeterministicReplayResult`只有executor身份与公共history摘要；原Canonical bundle允许部分轨迹。
`CANONICAL_SUCCESS`只证明这份canonical证据合法完成发布，不必然证明完整游戏终止或winner已核验。
把普通executor换成一个同名FullGameReplayExecutor对象或加verified标志不能补足历史证据。
本轮因此仅核验attempt terminal/claim/bundle/replay记录的联结，返回：
`UNKNOWN / CANONICAL_SUCCESS_NOT_FULL_GAME_TERMINAL_PROOF`。
证据中保留`canonical_attempt_closed=True`，不输出winner，不读取private replay、角色、BeliefObservation或backend私有载荷。
完整终局OBSERVED仍需原`phase2_gameplay.verify_results`等既有完整终局验证路径的真实、可审计来源与授权；本轮不执行它，不新造证明服务。

## 证据保全与无副作用

输出保留研究完整输入/record refs、plan/claim输入摘要、opportunity/assignment/PRE身份、调用方预期bundle摘要、
实际manifest与四分区字节摘要、action/event/handoff/annotation refs、日终及terminal输入摘要。
不改写任何输入。输出不复制真实报告、嫌疑集合、角色或私有终局内容；完整原始输入应仍由原来源保管，不由本函数另发数据集。
允许将已授权SubmittedGameplayAction的handoff用于内部hash/身份验证；不会读取对应BeliefObservation正文或把handoff嫌疑内容导出。

所有输出固定：`submission_permission=False`、`F1_permission=False`、`external_exactly_once_proven=False`、`replay_executed=False`。
没有commit、env、fallback、writer、随机化、行动评分或外部调用入口。
已有attempt ledger的恢复函数有副作用，未调用；本轮不扩展高水位或生产provider。

## 受控验收与资格限制

人工原生值对象由现有手工`test_game_bundle._fixture`构造，不调用其中executor；V1其他情景直接用原构造器手写，不调用Parser。
新增测试对Actor生成、Parser、bundle reader/publisher、replay和attempt recovery入口设置拒绝；I/O隔离另有专门检查。
覆盖跨attempt、错PRE/来源、缺token/action/annotation、COMMIT_UNKNOWN、研究COMMITTED矛盾、完整V1、PK未完成/最终空放逐、
attempt完成态仍未知完整终局、重复读取、输入不可变以及受限载荷不可访问。

首轮41通过/5失败发生在新人工fixture：4项原生V1版本字符串非法、1项错actor用例先违反原Ledger自洽。
只修正新fixture；原规则、来源准入、未知语义未放宽。各轮日志分别保留，不累计通过数。
最终测试数和命令以项目外验收JSON/JUnit为准；旧Actor/Parser回归或会调用Parser的Ledger夹具本轮不运行。

真实来源核验：NOT_RUN。本轮未取得明确准入的真实完整native action与研究assignment共同来源，不打开本地受限private分区、不访问服务器。
真实提交归属、完整终局正例、独立可信高水位、运行期联合读回/停止证明及F1准入均未完成；不能用人工测试或同PRE一致性替代。

## 2026-10-30 增量：SpeechPlan日终狼队放逐事件合同

开发基点为 Mac `twd/mainline@ebb2bd29a45f2ebee012b5dfe676d4e9b26c7bff`。
本节追加离线事实评价，不改写上面的历史验收，也不改变原 `NativeActionLink`、五个Finding、
`link_native_evidence()`或`_day()`的语义。不修改D1–D3、O/Z/T、G/π、执行路径及冻结Artifact。

定义合法狼人发言PRE τ的事件：

\[
E=\mathbf1\{X^{final}_{g,d}\cap W_\tau\ne\varnothing\}.
\]

`W_τ`是该PRE时行动者合法已知且仍存活的狼人，不按日终存活状态重新计算；
`X_final`是完整普通投票及必要PK结束后的最终公开放逐集合。
即使E已知，它也只是一项事实结果，不是SpeechPlan成功、行动价值、因果效果或终局胜率。

### 附加纯离线接口与当前阻断

`evaluate_day_wolf_exile()`接收原联结器的同一组参数，另可接`request`与`acting_observation`，返回独立的`Finding`。
其evidence保留原`native_link`全部五项结果、较严格的`public_day`、`PRE_wolf_team`以及`E`。
不调用模型、Actor、Parser、env、reader、回放、writer或恢复，不生成随机分配。

**当前状态：日终一致性检查已实现；真实狼队来源资格不足，确定E的构造保持阻断。**
当前接口没有E=0/1的准入分支：E始终为None，存在可检测矛盾时为CONFLICT，否则为UNKNOWN。
这不是证明事件没有发生，也不是用缺失补零。没有可由调用者设置的`trusted=True`、资格摘要或人工provider来解除阻断。
若未来取得真实来源材料，仍需单独审定其消费合同；本轮没有修改其他模块建立来源证明系统。

### 两份日终见证必须同时完整、一致

`_day()`复用`_resolved_exiles()`，核对普通/PK投票与最终放逐的一致性，并区分开放日终。
它没有覆盖所有公共合法性条件。例如，缺少部分存活投票人、投票含自身目标、
PK投票指向普通平票集合外或缺少PK发言时，旧Finding仍可能为OBSERVED。

附加评价原样调用`public_behavior_research.public_day_witness()`，复用其存活投票人覆盖、合法目标、
PK阶段/候选/实际发言者及公开淘汰检查；没有复制解析器或修改该函数。
之后将game/attempt、公共history与原生bundle来源绑定，核对两份见证的day、最终event_id、phase、
放逐席位、是否最终无人放逐及完整同日事件摘要。公共辅助函数本身没有game字段，不能独立充当game来源证明。

- 两份均完整且上述字段一致：附加`public_day=OBSERVED`，只确认公共事实。
- 一份完整而另一份不完整，或事件身份/放逐集合/公开来源不一致：CONFLICT，不选较宽松结果。
- 两份均未完成、没有结算或普通平票后PK未结束：UNKNOWN；临时空放逐不作最终零事件。
- 原生来源已CONFLICT：保留CONFLICT；缺少来源/外部bundle锚则保持UNKNOWN。

`shared_day_fact_digest`绑定game、attempt、day及最终事件身份。同日多个PRE可以共享该事实，
它不是assignment ID或独立随机实验单位，接口不计算试验样本数。

### 合法狼队的内容绑定不等于来源认证

原`build_shadow_request()`只读取过滤观察中的actor/observer、Werewolf身份、公共日/阶段/存活状态，
以及唯一的`werewolf_team_info.wolf_team`消息；原`contexts_for_pre()`在PRE存活集合中取交集。
本接口复用该构造器，核对native PRE与request的game、attempt、boundary、prefix、step、actor、
day/phase、完整公共前缀和捕获摘要，并重建request核对完整identity。

这些检查可以拒绝错身份、错PRE、错attempt、观察与request队伍不一致等输入。
但`ShadowRequest`可以直接构造，`ObserverContext`只校验字段，`legal_source`是固定声明，摘要只验证内容一致性。
`build_shadow_request()`自己的合同也明确不能证明任意调用者提供观察的真实性。
当前获准离线接口没有可验证的“原始执行路径→该attempt/PRE的行动者过滤观察”来源凭据及其消费合同。
因此即使内容检查通过，`PRE_wolf_team`仍为`UNKNOWN / ACTOR_OBSERVATION_ORIGIN_UNVERIFIED`。
`claimed_PRE_alive_wolf_ids`仅记录请求中未经来源认证的PRE声明，不是已确认狼队，不用于生成确定E。
不读取角色sidecar、私有回放、BeliefObservation正文或后续放逐来补全角色身份。

理论上，只有可信原始观察来源、全部PRE绑定及两份完整日终见证均成立，才可按定义给E=0/1；
当前来源条件未满足，不能通过人工测试、同机摘要或新增声明将此理论前提冒充已获资格。

### 独立行动证据及验收范围

结果不以submission、assignment归属或单动作依从成功筛选日终事实。
可以有完整公共日终，而assignment、submission或完整终局仍UNKNOWN；
多动作、Parser错误和执行失败不被解释为行动成功。原始五项Finding原样保留，
`randomized_assignment_inferred=False`、`continuation_verified=False`，不产生ITT或反事实行动值。

本轮人工CPU测试覆盖当前actor/队友/非狼放逐、完整正常/PK空放逐、PK未结束、两份见证分歧、
错PRE/attempt/观察、任意席位或自称可信来源、语义不依从、执行失败和共享日终。
所有人工观察只验证接口内容绑定，**不构成真实来源资格；测试没有确定E的正例**。
真实历史研究、真实来源资格、GPU、F0和行动实验均为`NOT_RUN`。

实际CPU验收使用已有`/Users/name_yuxiao/anaconda3/envs/3wd/bin/python`（Python3.12.13），
禁用GPU可见性、字节码和pytest缓存，最终命令为：

```sh
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' /Users/name_yuxiao/anaconda3/envs/3wd/bin/python -B -m pytest -q -p no:cacheprovider tests/observer_dtrs/test_native_action_link.py tests/observer_dtrs/test_public_behavior_research.py --tb=short
```

最终 **178 passed、0 failed、0 errors、0 skipped**（1.76秒），其中原生联结/新增评价103项，公共行为直接回归75项。
开发首轮36 failed/55 passed源于新夹具把规范事件记录交给原始输入构造器；改用原`_history_from_records()`解码。
次轮1 failed/90 passed源于新PK缺发言夹具留下未结束的turn_start；删除对应人工turn/speech对后通过。
未放宽生产规则、断言或原有用例；这些中间计数不与最终通过数累加。
静态AST对照确认原模块11项函数/类及原测试30项定义未改变，本文历史内容完整保留；`git diff --check`通过。
