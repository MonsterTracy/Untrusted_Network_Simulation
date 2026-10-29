# Phase-2H：G/π认知—发言研究闭环与Mainline兼容验收

状态分别为：**MATHEMATICAL_EQUIVALENCE / CONTROLLED_CPU_VERIFIED / RESEARCH_HYPOTHESIS / REAL_ACTION_EFFECT_NOT_VALIDATED**。前两项成立，不推出后两项已验证。没有生产策略、模型部署、F0或随机行动实验准入。

## 1　基准、实施范围与实际新增

2026-10-29，W3实际为`twd/mainline@ec008ef50dd10f2c678aa2d7e7439d31e333b80d`，不是远端e581。开始时619项文件、24项未提交成果（2 tracked、22 untracked），index为空。会话列表未见另一活跃W3任务，间隔取样与测试前原文件摘要一致；这只是检查期间未发现并行写入，不是操作系统排他锁或永久无写入保证。

原有Phase-1、S1a/P0已经具备目标接口，没有证实必须修改生产实现的缺陷。因此**仅新增**：

- `tests/observer_dtrs/test_g_pi_phase2h.py`：19项跨接口受控验收。
- 本文：机制、数值结果、反例、兼容与整合边界。

不修改原24项成果，不改D1–D4、Broker/Resident、Actor/Parser/Recorder、默认Agent、Source Identity或任何正式数据／模型。没有新快照类型、生产adapter、候选生成器、评分器、门控、奖励或兜底。

## 2　实际数据流与完整认知输入

原`planning_research.analyze_pre`复用`ShadowRequest`、`QPrediction`、`FrozenEventEstimator`、`observer_features`、冻结`D3Protocol.loss`及`decide`：

\[
O_t=A_t\setminus W_t,\qquad
\mathcal C_t=(Q_t,\{F_i,p_i,R_i,D_i^*,z_i\}_{i\in O_t}).
\]

Q保持绝对七席位、自列零及死亡列质量。`F_i=(s,m,r,dW,phase,nW)`，单狼m=0且dW不适用。p是现有只读估计表给出的D1事件概率，不能把r或Q替代p。D3风险为

\[
R_P=c_{FP}(1-p),\quad R_B=b_1p+b_0(1-p),\quad R_N=c_{FN}p.
\]

保留全部Bayes最优集合与明确TiePolicy选中的单一区域；精确并列不靠浮点容差吸附。Phase-1从原冻结摘要`0ebd6b97669c35e1d9201f88152da8919d87586864a751f3fd688edd3a11c4ab`重建45成本，不接受任意cost_source字符串。

本轮测试侧将快照与原typed context、features、模型摘要、成本及完整decision逐字段／canonical字节核对，再交给现有`process_cognition`；没有新增生产反序列化或绕过来源入口。独立JSON往返的tuple/list表示差异用canonical完整字节比较，不丢身份字段。

**支持诊断限制**：当前`FrozenEventEstimator.predict`只有probability和lookup level；global/nW/phase_nW/r_bin表示预测路径，不是Kish、训练游戏数或完整节点支持。快照保留现有层级诊断，完整训练支持未包含，亦未通过读取训练审计补齐。缺支持不转成BND。

## 3　G与π的实际职责

`process_cognition(pre,panel=...)`可接全体合法观察者，也可显式接子集；子集必须包含按合法公开V1优先规则选中的座位。全体面板为ALL_ELIGIBLE，旧V2只供一行时为EXPLICIT_SUBSET。

`select_speech`仍只消费公开规则选中的**一个**观察者，没有实现全观察者聚合或优化。它要求原完整候选tuple与显式`SpeechStrategy`。区域和完整连续风险两模式在相同TiePolicy下，再经过同一原型映射；没有获得行动价值函数。

G/P0的超额风险
\[
\Delta_i(d)=R_i(d)-\min_zR_i(z)
\]
仅是强制认知判断的额外规范性损失。不能解释为发言效用、因果后悔或放逐概率。

以下保留为旧研究假设，**没有修成新的理论默认值**：

- selected-region依据门控：另一零超额风险的Bayes并列判断仍可能被拒绝。
- KEEP_BOTH绑定BND：静态暂缓判断不等于某行动在两情景均适用。
- PUBLIC_ONLY可不带事件引用；有引用仅验证实际公开V1来源身份，不证明候选主张得到充分支持。
- POS/BND/NEG经过配置选择SELF_DEFEND、OPPOSE或NO_COMMITMENT，没有真实行动效果依据。
- 原`contextual_loss`的误判倍数及暂缓比例是已有规范性假设，不是测得的狼人发言效用。

## 4　π层四种资格分开

|层次|现有来源|实际含义|
|---|---|---|
|LEGAL|原generate_candidates|完整六类单动作候选与原顺序，含末位和合法队友目标|
|EVALUABLE|Phase-1代理来源、同阶段后继、原容量预检|旧B2接口在限定输入下可进行局部嫌疑代理评价，不是行动准入|
|APPRAISED|完整批次返回及来源/覆盖核验|确有该代理结果；人工consumer仍只能标CONTROLLED_TEST_ONLY_NOT_SEALED_B2|
|ACTION_VALUE_AVAILABLE|当前不存在合格真实行动价值源|即使B2结果齐全，P0无values时仍ACTION_VALUE_UNAVAILABLE，不选择计划|

P0的assessable只是**依据准入**，不是B2可评价；人工values只用于选择机制测试，不使ACTION_VALUE_AVAILABLE成立。完整候选始终保留，不因末位、容量、来源缺失或批次失败删行、补零、重试或返回默认NO_COMMITMENT。

Phase-1当前Q与B2分别记录身份；current_suspicion/delta均为空。没有跨来源差值，没有把假想后继Q送入当前D2。本轮没有读取或运行正式B2 Final，原Final及ExperimentCapacity仍未因此准入。

## 5　实际CPU机制对照与完整结果

使用既有30场景定义（普通/PK×三种公开指控方向×五个Q质量值），换用既有人工public witness，无env执行；D2为**手声明只读EQ1表**，五个输出为1/8、1/4、1/2、3/4、7/8，未拟合真实或人工训练集。所有臂共用同PRE/Q/全体p/候选/目录/旧人工偏好与准备预算。

对照沿用原4目录、2成本条件、5模式，共**1200受控单元**；不是1200随机分配或真实观测。静态C22只是旧对照的示例，未指定为正式唯一主矩阵；上下文成本单列，未冒称45登记成本之一。原45组均在每个快照保存并核对。

- 6750条逐观察者×成本记录：原完整连续风险、Bayes最优集合与显式D3完全一致。
- 60次S1a双模式选择：认知分支与计划完全一致。
- 240组P0显式门控 vs 相同风险argmin：计划、依据、可评议数的差异均0。
- 被评价的纯研究调用期无Actor、Parser、env.step、commit或真实模型调用；fixture准备阶段仅调用已有scripted V1 Parser，不属于行动执行或真实模型调用。研究调用期阻断语言/Parser/torch.nn.Module与非配置I/O；Python/NumPy/Torch RNG不变。该调用期唯一读取为原D3冻结配置。

下表每格分母为30场景，记为“计划变化／所选依据变化”；S/C分别为静态／上下文成本。完整行与所有对照保留项目外results，不挑选有利场景。

|既有目录|记录风险→三支门控 S/C|三支→连续argmin S/C|三支→强制二支 S/C|静态→上下文（三支）|
|---|---|---|---|---|
|public_only|0/0；0/0|0/0；0/0|0/0；0/0|0/0|
|cyclic_cognitive|18/18；15/15|0/0；0/0|6/6；3/3|3/3|
|same_action_multiple_bases|0/18；0/15|0/0；0/0|0/6；0/3|0/3|
|defer_only|24/24；27/27|0/0；0/0|6/6；3/3|3/3|

门控对cyclic_cognitive在30/30场景减少可评议候选。defer_only有24/30（静态）、27/30（上下文）变成NO_ADMISSIBLE_BASIS，属于准入规则造成的无结果，**不是效益改善**。S1a从静态改上下文只在1/30场景改变计划；它不使用P0目录/偏好，因此S1a对P0不是隔离了门控唯一差异的因果消融。

### 可复核反例

1. `discussion-self_accused-q0`，p=1/8、NEG，cyclic目录：关闭门控选择ACCUSE_WOLF(player2)，门控后选择ACCUSE_WOLF(player3)，可评议数24→8。变化来自预置HIT/MISS依据与旧人工偏好，不证明对队友/非队友的实际行动优劣。
2. 同场景same_action_multiple_bases：计划同为ACCUSE_WOLF(player2)，理由由ASSUME_HIT变ASSUME_MISS。认知解释变化不等于行动变化。
3. 同场景defer_only：门控后无计划，完整24个合法候选仍在；没有自动NO_COMMITMENT。
4. `pk_discussion-self_accused-q2`，p=1/2：静态C22为BND/OPPOSE(player3)，原上下文成本为POS/SELF_DEFEND。变化是成本与配置映射的组合效应，不是新认知信息或已识别收益。
5. C22、p=1/4精确边界：NEG与BND同为Bayes最优、BND超额风险0；terminal-first仍拒绝KEEP_BOTH，defer-first允许。保留原规则与两者区别，不静默改成全最优集合准入。
6. 公开V1指控player1，但为ACCUSE_WOLF(player4)提供同一引用，bind_bases可验证来源而不会验证推论充分性；无真实value仍不选择。这证明PUBLIC_ONLY不是行动论证验证器。

## 6　每PRE重新绑定与来源边界

人工同局相邻合法狼人PRE（player1后player2）使用同一EQ1估计表；Q的r从.1变.9，全部观察者p从1/8变7/8，C22区域NEG→POS。旧Q、旧G、旧候选目录或旧PRE身份被拒绝；重复调用一致。未把过去概率当未来预测器，也不是序贯后验、因果响应或整局动态规划验收。

Final/OOF身份改名与fixture估计包不兼容时明确失败；没有放宽现有reader。当前Final-ToM/CAL独立来源、真实权重资格和行动监督仍缺，不能用本轮表或旧D2外层模型替代。

## 7　实际测试与未验证项目

Python3.12.13、PyTorch2.12.1、NumPy2.5.1、pytest9.1.1，Mac CPU；禁GPU可见性、字节码与pytest cache，临时文件在项目外。

|本轮执行|通过|失败|跳过|主动排除|pytest时间|
|---|---:|---:|---:|---:|---:|
|原有定向基线|395|0|0|14|25.02秒|
|新增最终验收|19|0|0|0|1.82秒|
|最终合并定向回归|420|0|0|14|25.59秒|

420包括19新增及6项原D4 wire framing测试；没有启动Resident/Broker服务。14项排除是原counterfactual/sealed fixture生命周期，明确没有作为通过。原Actor/Parser/提交及预审计兼容回归仅用人工后端／受控env fixture，不是整局游戏或真实F0。

保留首次新增测试的17失败/2通过记录：测试helper把JSON list与原typed tuple直接比较，生产输入/输出无误，改为canonical全字节核对。第二次18通过/1失败因旧候选集合在PRE身份检查之前已被正确拒绝；分别建立旧候选与旧身份前提，不放宽异常正则。最终19通过，均只修新测试，不改变生产行为。机制runner首次同一helper断言失败也保留。

主命令使用`/Users/name_yuxiao/anaconda3/envs/3wd/bin/python -B -m pytest -q -p no:cacheprovider`，精确路径、排除表达式、退出码与JUnit在项目外证据包。机制runner同样只在项目外输出，不是新的生产入口。

## 8　Mainline文件级兼容与Source Identity

2026-10-29 15:05（Asia/Shanghai）GitHub只读实查：远端`twd/mainline`仍为e5818d46fa1d02869e2c0f381a14f2b5ea5ff4a5。Run40829081627仍in_progress/conclusion=null；同SHA Run40821435932为success。不是把历史3308等CI计数当本轮通过，也不继承远端CI到W3未提交代码。

对指定入口及直接本地导入49路径，ec008与e581的committed blob：43相同，1变更，5两端均未提交。43相同不代表W3两个tracked研究修改已经进入远端。完整矩阵在compatibility.json。

唯一相关生产差异是`final_tom_broker.py:144–147`：内部owner请求ID由token16+':'+rid变为token16+':'+digest(rid)，内部长度81，保护原公开128字符边界。W3未修改此文件，也未执行最新版请求ID整套集成测试；只验了两端字节相同的wire framing。本轮研究路径不调用Broker，不能把局部兼容写成全部最新Mainline通过。

未来整合必须保留远端Broker补丁、其111/112/128/129回归、Python3.12 CI及matplotlib依赖；**不使用旧W3整文件覆盖远端**，不在本轮fetch/merge/rebase。S1a runtime与precommit审计两份已有tracked diff继续分别审阅。

现有QUAL源码inventory为202文件，摘要`21c92263f8f3816447ac3704094a207b3d015b1e4d8dde56c8a5ea2635c25949`；本轮测试/文档不在这些runtime inventory内，原生产字节/大小/权限均未变。D2 formal source guard在当前已有未提交runtime下按预期拒绝，未绕过。不能只用HEAD代表完整W3；证据包列出所有相关实际工作字节。将来提交A/B新增运行源码或整合e581会改变源码身份，旧Baseline/Final资格不自动继承。

## 9　精确整合分组（仅建议，不暂存、不提交）

A：纯完整快照与候选检查的CPU合同可考虑独立审阅为3路径：

```
werewolf/observer_dtrs/planning_research.py
tests/observer_dtrs/test_planning_research.py
docs/research/observer-dtrs-dynamic-planning-phase1.md
```

B：S1a/P0人工规则仍为研究假设；整体提交必须保留明确原型标签。9路径：

```
werewolf/observer_dtrs/speech_evidence.py
werewolf/observer_dtrs/candidate_appraisal.py
werewolf/observer_dtrs/speech_runtime.py
tests/observer_dtrs/test_speech_evidence.py
tests/observer_dtrs/test_candidate_appraisal.py
scripts/compare_observer_dtrs_s1a.py
scripts/compare_observer_dtrs_appraisal.py
docs/research/observer-dtrs-phase-s1a.md
docs/research/observer-dtrs-phase-s2-p0.md
```

A/B就绪后才考虑本轮2路径，不可孤立提交测试后声称主线可运行：

```
tests/observer_dtrs/test_g_pi_phase2h.py
docs/research/observer-dtrs-g-pi-phase2h.md
```

G身份绑定／数学委托是技术能力，G→动作映射和P0门控是B类科学假设；不能按整个speech_evidence.py“pytest通过”混为已验证行动机制。

C：Phase-2D～2G及预提交观察点12项原成果全部保留，本轮不建议混入A/B：Ledger/Store/Native link三个实现、对应三测试、人工fixture、三个阶段文档、`scripts/verified_speech_commit.py`、`tests/tom/test_precommit_audit.py`。它们不具有真实执行、回退或严格一次提交许可。

D：历史来源、真实数据与受限实验、独立审阅证据在项目外，只保留引用和摘要；不向Git加入数据/监督/模型或旧资格包。本轮具体分组与路径见integration-groups.json；这些是待用户批准的精确清单，不是git add .授权。

## 10　下一项尚缺的研究证据与停止

D3确实提供了风险敏感的认知判断；本轮未证明它改变了**有依据的真实SpeechPlan行动评价**。所观察差异完全可由相同连续p/风险加同一人工mapping/gate复现。还缺：在合法来源与固定执行机制下，候选计划与实际完整执行、独立后继／日终结果严格对应的行动价值证据，以及为什么某认知依据对该行动有效的明确可检验机制。

旧B2还需原Final、condition、seal/checkpoint与真实ExperimentCapacity准入、合法同阶段后继、数值及科学代理有效性；不能覆盖末位或冒充收益。随机行动价值学习需另审定支持集、已知分配概率、固定Actor/Parser/继续策略、ITT与失败结果合同、数据和执行资格；本轮未实现或冻结。

历史多动作发言不得拆成单计划监督。未来研究必须先说明单计划的完整语义执行与多动作表达空间的关系，之后才能选择扩大行动集或进行受控比较；本轮六类候选不变。距离每PRE实际动态重规划，最关键的科学缺口仍是**合格的行动评价目标与候选—执行—结果证据**；Final/CAL来源和在线资格另为工程前提。

Phase-2H以受控机制验收结束，不进入Mainline合并、模型运行、F0或真实行动／胜率实验。
