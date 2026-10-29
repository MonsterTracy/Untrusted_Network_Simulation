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
