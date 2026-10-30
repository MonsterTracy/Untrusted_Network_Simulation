# 真实PRE认知面板与候选依据诊断（研究切片）

状态：`REAL_HISTORICAL_DIAGNOSTICS_COMPLETED`；`NO_GO_FOR_FORMAL_G_PI_SPEECH_POLICY`。
这是历史开发数据上的事后研究，不是新独立TEST、在线策略或行动效果实验。

## 实现范围

新增模块 `werewolf/observer_dtrs/cognition_gate_research.py` 只做三件事：

1. 将 `open_dataset` 验证的合法七列Q/条件属性，与 `read_formal_d2(...).batch('OBS-EQ1','game_equal')` 的无标签概率按sample_id严格联结，保留来源折、外层块及预测artifact引用。
2. 用冻结PRE中最近一次同日的、存活非狼观察者的成功V1，识别指向当前speaker的显式狼人指控；同一注释同时将其清白化时保守返回冲突。
3. 复用原D3、G和P0进行风险/准入检查，所有调用不提供ActionValues，因此不选择SpeechPlan。

原始Q仍为绝对七席位概率质量，不是D1事件概率；保留死亡目标质量和自列零。历史D1仅提供本任务所需22685个合法观察者行，不能冒称每个PRE都有完整7×7矩阵。

`HistoricalObserver`只含合法行、事件p、封存origin和outer预测artifact引用，不含Y、报告、sidecar或训练支持。后一个引用不是模型参数SHA，也不是在线部署许可。

`bind_inputs`是已验证reader之间的联结器，不认证调用者手工拼造的对象。正式来源必须由原reader与外部锚核验；人工单元测试明确为受控fixture。

## 新增公开线索规则的有限语义

`latest_same_day_eligible_observer_explicit_self_accusation_reply_cue_v1`仅生成：

```text
PUBLIC_ONLY / SELF_DEFEND(None)
前提：观察者i最近同日公开发言的成功V1明确point_as_werewolf(current_speaker)
证据：真实event_id + annotation_digest + PRE公共前缀
用途：可定位某项公开指控的回应语境
```

它不证明“我不是狼人”是真话，不证明自辩合理或有效，也不证明隐藏D1事件Y成立。该规则是本轮公开回应前提的窄研究合同，不能称所有发言理由的完整生成器。

不凭来源存在就给任意候选添加PUBLIC_ONLY；不自动添加NO_COMMITMENT兜底；不自动生成ASSUME_HIT、ASSUME_MISS或KEEP_BOTH。它们与具体SpeechPlan的独立必要依赖尚未证明，保持UNSUPPORTED。

所有原合法候选仍保留。`NO_ADMISSIBLE_BASIS`只表示此有限目录无合格条目，不表示没有合法或合理发言。没有行动值时返回`ACTION_VALUE_UNAVAILABLE`，`selected_plan=None`不是“动作不变”。

## 四臂与完整面板

- A0：原P0 RECORD_ONLY；A1：原FORCED_BINARY；A2：原THREE_WAY；A3：原CONTINUOUS_EQUIVALENCE。
- 共用同一PRE、Q/p、成本、TiePolicy、完整候选和事前公开目录，不拟合阈值或理由。
- PUBLIC_ONLY不依赖认知判断，因此本目录的全45成本准入表可代数展开；CLI明确区分这一展开与848次原G/P0有界调用交叉核验。
- A2/A3使用同一三风险及相同精确并列顺序，数学上等价；不是两种独立的信息源。
- 原G支持完整面板，原S1a π仍只消费公开规则选中的一个观察者。这里不引入概率均值、观察者权重、投票或新的聚合政策。
- 原P0 selected-region门控与KEEP_BOTH语义保持不变，未擅自采用P1尚待审定的完整Bayes集合门控。

## 实际结果与统计口径

固定1500局、5835 PRE、22685观察者行、96200候选；完整PRE集合身份摘要：
`2e151e0ac08d794368f32782eab4943b9c23d56640e3a22c9df006a68c30b464`。

799 PRE有980条上述公开线索；5036 PRE无此目录条目。首次机会1500 PRE中76个有88条线索。
四臂、两种面板在此目录上的准入集合/依据身份差异为0；所有真实计划选择未执行，动作/目标差异UNKNOWN。

3058/5835 PRE的p不完全相同；首次机会为651/1500。所有PRE的r存在严格数值差异，2777 PRE仍有相同p。
完整45成本下，主策略的游戏等权预测区域异质PRE率为0.39%–36.29%。这不是实际私人认知分歧或策略收益。

每个PRE的指标包括：至少两种区域、选中观察者与其余观察者不一致的比例、BND覆盖等。前者与“选中观察者和至少一人不同”在完整面板中恒等，不能作两条独立证据。

区间复用原D2 2000次来源折×外层块分层整局索引：先计算每局PRE指标均值，再游戏等权。首机会每局一PRE。阶段/nW/座位为描述性统计，无人口的first-PK保留为空。
这是固定预测的点式探索性区间，不含上游重训、选参、目录选择或多成本确证校正；不能用于择优成本。

## 复现及证据层级

入口 `scripts/analyze_observer_dtrs_cognition_gates.py`，CPU-only，完整显式参数见后文。
固定已准入来源及摘要；不覆盖现有目录，不访问服务器，不加载模型或运行Actor/Parser。

脚本先保存analysis-plan，再输出全人口、90个成本/并列组合的表与派生npz、公开依据证据和有界原接口核验。npz是本轮派生判定，不是原正式D3逐样本artifact。
D2原reader会内部校验包含Y的封存混合行；之后只提取label-free batch并释放reader。未open D1 supervision，不把targets或未来结果传入研究模块。

新增161项CPU测试及原接口196项回归全部通过；有界848次原P0交叉核验零差异。真实统计不是这些人工测试的产物。

## 研究与执行边界

现有D1事件是狼队命中事件，双狼时不等于怀疑当前speaker；单狼时也仍是有效支持的操作化事件，不等于公开指控或真实心理确信。成功空报告的冻结规则不变。

下一项最上游科学决定仍是：具体候选理由何时独立地需要Y或暂缓Y判断，以及何种共同选择准则可以使用这些理由。之后才是独立行动结果监督和政策比较，不能用人工偏好代替。

本轮不冻结政策/损失主矩阵/随机化，不解除Final-ToM/CAL来源、G3/G4/G5或真实F0门槛。
GitHub cbb5 的Run40868711936已查得failure；本轮357项本地目标测试通过不等于主线CI通过，未在本轮修复CI。

## Mainline可复现入口（路径修复）

代码不再绑定个人Mac目录，输入路径必须显式给出；只接受原已封存来源，所有数据/协议/分割/公共PRE/抽样摘要保持不变。数据不随Git提交。

```sh
python -B scripts/analyze_observer_dtrs_cognition_gates.py \
  --dataset "$D1_DATASET" --d2-run "$D2_RUN" \
  --public-pre-file "$PUBLIC_PRE_FILE" --output "$NEW_OUTPUT"
```

变量分别指正式D1目录、完整D2 run目录、已准入公共PRE JSONL和全新独立输出目录。输出父目录须存在且规范化，输出不得位于源码/输入目录内，已有输出拒绝覆盖。原161/196测试与历史数字是旧执行证据；本次路径修复不重跑1500局统计，不将新源码身份冒称旧封存身份。正式策略和F0资格不变。
