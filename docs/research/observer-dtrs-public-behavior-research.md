# ToM认知表示与后续公开指控：历史条件预测研究

状态：`HISTORICAL_PUBLIC_BEHAVIOR_PREDICTION_COMPLETED`。新目标只作离线预测，不是D1事件、行动收益或独立TEST；G/π与真实F0继续NO-GO。

## 输入与目标

复用原D1合法reader、上一轮`cognition_gate_research.bind_inputs`、D2完整来源校验与冻结45成本；固定1500局/5835 PRE/22685观察者行。原626文件及上一轮4个新文件均不修改。

Z是：当前狼的实际公共发言之后，观察者在同日最终普通投票/必要PK完整结算前的后续公开发言，是否曾由成功V1明确point_as_werewolf(current_actor)。需要原公共流与PRE前缀、紧接speaker事件、注释与日终截止严格绑定。不存在后续发言机会不补0；V1缺失/失败/同条指控并清白化保持UNKNOWN。不同时间先指控后清白化仍为曾指控，单列反转。测量对象是封存V1记录，未重新核对原始自然语言准确性。

实际可判定10835行/1454局，正1469/负9366；11848行NO_FUTURE_SPEECH，2行V1_CONFLICT。覆盖只属有后续机会且可解析的条件人口，不代表所有PRE均可部署新Z预测。

## 代码切片

- `public_behavior_research.py`：`public_features(pre, observer)`只接冻结PRE；`legal_features(context)`明确区分合法队友信息与纯公开历史；`observe_after_speech`独立接收未来公共见证只生成离线目标。新目标fit只复用原`baselines.objective/sigmoid/SOLVER`，不是重训或改名D2。
- `analyze_observer_dtrs_public_behavior.py`：先保存全部无标签特征与身份SHA，再打开完整未来公共归档。原始文件/manifest/前缀完整验证；按原25个来源折×外层块拟合相同CPU映射。
- `test_public_behavior_research.py`：人工CPU边界与数学验证；不得将其解释为真实行动效果。

当前环境无sklearn/scipy，不安装依赖。固定L2 Logistic C=1，均值权重目标tau=1/N，截距不罚；原NumPy确定性Newton、训练内游戏等权标准化，无超参选择/测试校准。稳定logaddexp评分，不裁剪概率。

## 主要结果

| 表示 | 游戏等权Brier | Log Loss |
|---|---:|---:|
| 指定公开历史M0 | 0.115943378 | 0.392153467 |
| M0+s | 0.114921426 | 0.386563287 |
| M0+r | 0.115530978 | 0.390063807 |
| M0+s+r | 0.115055802 | 0.387006186 |
| M0+原事件p | 0.115844916 | 0.391812134 |
| M0+C22区域 | 0.115831280 | 0.391655363 |

s−M0：Brier差−0.001021952，95%固定预测整局区间[−0.002053037,0.000081198]；Log Loss差−0.005590180，[−0.008913457,−0.002126670]。这是相对指定有限公开特征基线的小幅关联证据，不是完整公共历史之外的新统计信息。

s−r的Log Loss差−0.003500520，[−0.005699643,−0.001403153]；Brier仍不确定。单狼时s=r，混合人口训练导致的单狼子组预测差不能被解释为两种输入在单狼时携带不同信息。

原p不是Z后验，所有分数均来自新的训练内Z映射。p和C22相对M0的区间均跨0。C22−p Brier差−0.000013636，[−0.000192756,0.000156333]；无批准非劣界，不能声称等效/可接受无损压缩。

45成本全部保留：M3−M0的Brier点差21负/24非负，范围[−0.000336041,0.000335767]，45个点式区间均跨0，不择优成本。C22完整风险与区域仍为确定性关系，区域编码不是新增信息。

## 解释与限制

M0是37个事前声明的结构化公共特征，不包括任意文本的全部信息。另有L0合法狼队上下文控制与p+同C22区域的连续丰富控制；主要方向相同，不能把r/p含队友信息的差值全部归因于ToM。

3276 PRE至少有两个可判定观察者。C22完整合法面板异质时，行为混合为71/382；面板同质时为765/2894。前者平均可观测2.57人，后者2.93人，存在机会数/阶段等差异；这是描述性反例，不是因果或面板策略效果。

2000次按来源折×原外层块分层的整局配对bootstrap，只使用1454有结果游戏，新索引SHA `a007a96f47e2a863633f55bced7699ac8e607739686361982179c6e2a3340b82`，不冒称原1500局抽样字节相同。区间条件于固定预测，不含上游重训/新映射再拟合；45成本相关、事后开发研究，非同时确证。

虽新Z映射train/test整局隔离，训练行的固定D2 OOF p可能曾使用新Z测试游戏的原D1监督；历史ToM全publication摘要影响RNG的来源限制也保留。因此不能声称端到端完全无泄漏/独立泛化。

## 验收与下一步判断

60项新增测试与253项直接回归最终全部通过；首次人工fixture对象类型错误已修正并保留失败日志。真实CPU分析86秒，58表示×25块=1450拟合；无私有D1监督读取、无模型服务/游戏/服务器。

**值得继续把原ToM的s作为行为预测候选进行独立验证；当前不支持把旧狼队事件p或三域升级为行为评分核心。** 有限/不确定增量不等于证明无任何关联。行为预测也不识别SpeechPlan效用，不解除G/π、F0、Final-ToM/CAL或行动实验的资格限制。本轮不执行后续实验。

## Mainline可复现入口与公共验证依赖

不再从个人结果目录加载Python。原`public_speech_day_audit.py`中仅`parse_public_game`和`public_day_witness`两函数迁入本文件组的`public_behavior_research.py`，函数验证逻辑保持不变；其历史来源文件SHA仍记录为`50996ec5b66277fdd46565dbbbfc5c2933097838ef5e6faff2b312b70b695b61`，这是来源元数据，不冒称新模块字节SHA。

解析复用仓库原`_history_from_records`、`_annotation_from_record`、`validate_v1_annotation_binding`；日终复用`_resolved_exiles`、`_vote_winner`并保留存活投票者覆盖、目标资格、PK阶段/发言、淘汰一致性等附加检查。二者仅消费公共事件/V1；不会调用角色sidecar、私有认知报告或模型权重读取。未迁入无关的`recover_speech`，也未新建第二套算法。

```sh
python -B scripts/analyze_observer_dtrs_public_behavior.py \
  --dataset "$D1_DATASET" --d2-run "$D2_RUN" \
  --public-pre-file "$PUBLIC_PRE_FILE" --public-archive "$PUBLIC_ARCHIVE" \
  --plan "$ANALYSIS_PLAN" --output "$NEW_OUTPUT"
```

PUBLIC_ARCHIVE为原已准入公共event/V1 ZIP，不是Python目录；ANALYSIS_PLAN是已审阅计划，必须由研究者提供。所有原来源摘要不变，路径参数不是重新选数据或放宽身份验证。干净checkout具有完整代码依赖，但真实分析仍需获准数据/计划；本次只进行人工CPU与入口验收，不重新发布历史结果。输出必须全新，父目录规范化且在源码及输入外。
