# Observer-DTRS v1 D3b 第一轮：纯决策器与受控离线评价

日期：2026-10-15。状态：**IMPLEMENTED_CONTROLLED_ACCEPTANCE_NOT_FORMAL_PROTOCOL**。
在本地 three-way、HEAD `3ff578f97b9d4c3620a15a67357ddabd848d10fc` 实施；未提交、
未 push、未合并、未操作 stash，没有任何服务器访问或正式 45 场景真实数据实验。
D0 事件、D1c、冻结 D2/ToM/Mapper/概率模型和既有结果全部不变。

## 1. 文件与职责

- `werewolf/observer_dtrs/decision.py`：LossSpec、显式 TiePolicy、条件风险、完整最优
  集合、直接 argmin、阈值/退化诊断、概率/成本情景稳定性、45 个规范性候选。
- `werewolf/observer_dtrs/decision_offline.py`：最小合法 ProbabilityInput、显式 Y 评分、
  完整游戏权重、子组审计、固定行动/强制二支对照、无标签稳定性聚合。
- `tests/observer_dtrs/test_decision.py`、`test_decision_offline.py`：数学及访问边界测试。
- `scripts/check_observer_dtrs_decisions.py`：仅两个人工游戏/四条概率的受控验收入口。
- `observer-dtrs-v1-d3b-controlled-acceptance.json`：受控输出及五个代码/测试文件指纹。
- 本文及 `observer-dtrs-v1-d3c-download-plan.md`：实现契约与下一阶段材料清单。

没有修改旧 D3a 数学工具/验收记录，也没有修改 D0/D3a 原文来掩盖并列约定差异。

## 2. 并列规则：明确保留尚未解决的历史差异

重新读取用户提供的 D0 原文：
`/Users/name_yuxiao/.codex/attachments/39addd02-7062-4eb8-9885-d68c3b8dd867/已粘贴的文本.txt`
第七节明确写出 POS: p≥alpha，BND: beta<p<alpha，NEG: p≤beta，并规定
“阈值处采用终端判断优先于暂缓判断”。D3a §3 则建议 B 优先，仍为未冻结草案。

两者在非退化的精确边界不同。实现**没有默认策略**：构造 LossSpec 时必须显式
提供 TiePolicy；字符串或省略均拒绝。策略和成本一起进入 loss digest 与每条结果。

| 策略 | 完整优先序 | beta 处 | alpha 处 | 三者同时并列 |
| --- | --- | --- | --- | --- |
| TERMINAL_POS_FIRST | POS→NEG→BND | NEG | POS | POS |
| TERMINAL_NEG_FIRST | NEG→POS→BND | NEG | POS | NEG |
| DEFER_FIRST | BND→NEG→POS | BND | BND | BND |

前两者都符合 D0 **非退化边界**约定，但 D0 没有指定退化 P/N 同时最优时谁优先；
所以不能把某一完整优先序冒充 D0 已确认的唯一规则。DEFER_FIRST 实现 D3a 建议。
强制二支对照另外要求显式 binary_tie_policy，过滤掉 BND 后使用其 P/N 次序；
受控例子显式选 N 优先，符合 D3a 对该对照的拟议约定。

正式实验前仍须确认完整并列政策，尤其退化 P/N 并列。这不阻碍兼容的通用实现，
但不构成对正式损失、主矩阵或实验协议的冻结。

## 3. 输入、数值与输出契约

```python
spec = LossSpec(c_fp=1, c_fn=1, b1=Fraction(1,4), b0=Fraction(1,4),
                tie_policy=TiePolicy.TERMINAL_POS_FIRST)
decision = decide(event_probability, spec)
```

纯模块仅使用 Python 标准库，无监督、支持、ToM、D1 reader 或 Phase2 导入。
`decide(probability, loss)` 只有这两个参数。NaN/无穷、bool/字符串、越界 p、
非正错误成本、负暂缓成本和违反列序的成本都报错；真实 p 从不裁剪或重归一化。
参数有限性限制为可以转换成有限 float 的实数；保留其实际精确有理值进行计算。

支持正错误成本和**弱列序** `0≤b0≤c_fp, 0≤b1≤c_fn`。D0/D3a 正常场景仍是
严格内部；等号只作为显式退化诊断，未加入 45 候选。高于对应错误成本的 b 被
认定超出当前规范性输入域，即使一般仿射风险也能机械算出，仍不静默接收。

输出 Decision 为不可变结构，保存 p、loss digest、策略、三项 ConditionalRisks、
按 POS/BND/NEG 固定顺序排列的全部最优行动、所选域、风险间隔、阈值和适用状态。
风险间隔是三个行动的有重数排序中第二小值减最小值，任何最优并列时为零。
只有非退化状态给两阈值距离；其他状态返回 None，不虚构边界。

**精确并列的数值含义**：输入 float 按其二进制值转为 Fraction，而非按打印小数
猜测或引入误差带。内部三项原始风险、阈值、最优集合与间隔均用精确有理数。
`to_record()` 同时给浮点展示和 `*_exact` 字段，判定依据是后者；极小差异可能
在浮点展示中消失，不能由展示值重新选域。`thresholds.alpha/beta` 的 Fraction
可精确传回；把一个非二进制可表示交点先舍入成 float，可能已经不是数学精确交点。
没有阈值吸附、近并列容差或人为拓宽 BND。

`spec.scaled(k)` 在有理数层面正比例缩放，保证区间和行动不变；若调用者先自行
浮点乘法再构造另一矩阵，可能已改变精确比例，不能把该输入变化当作理论反例。
内部有理运算还避免 `c_fp+c_fn` 在两个有限大成本下中间溢出。此轮未以真实
22685×45 样本场景计时，不从受控案例推断正式吞吐量。

## 4. 非退化与退化处理

直接 argmin 始终决定结果，绝不由阈值捷径替代。保存
alpha=(c_fp−b0)/(c_fp−b0+b1)、beta=b0/(b0+c_fn−b1)、gamma=c_fp/(c_fp+c_fn)。

| regime | 说明 |
| --- | --- |
| NONDEGENERATE | b0,b1 严格内部且 b0/c_fp+b1/c_fn<1 |
| COLLAPSED_BND | 严格内部比值和=1；BND 仅并列点 |
| BND_DOMINATED | 比值和>1；BND 不最优 |
| ZERO_DEFERRAL | b0=b1=0；内部 BND 最优，端点并列 |
| NEG_INTERVAL_COLLAPSED | b0=0、其余非重合情况 |
| POS_INTERVAL_COLLAPSED | b1=0、其余非重合情况 |
| IDENTICAL_NEG_BND | b0=0、b1=c_fn；两风险线重合，beta 为 None |
| IDENTICAL_POS_BND | b0=c_fp、b1=0；两风险线重合，alpha 为 None |

“比值和=1 时 BND 只有一个点”的结论需要严格列序前提，不能应用到整条风险线
重合的弱序边界。这是将 D3a 推导的适用范围显式落实，不改变正常候选的定义。

## 5. 离线评价与稳定性

`ProbabilityInput(sample_id, game_id, probability)` 只承载标识和事件概率，不声称
已经验证模型来源。`evaluate(rows, targets, *, loss, game_sizes, weighting,
binary_tie_policy, subgroup_ids=None)` 显式另接收受限整数 Y；无真实报告/角色接口。
固定完整游戏大小、拒绝缺失/重复/跨游戏人口，才能计算主权重 1/n_g 或记录权重 1。
子组先验证全人口再选择 ID，仍使用原 n_g。空子组、零决定覆盖和无某标签类的
条件比率输出 None，不删除游戏或填零。

输出全人口实际六损失、模型自身预测最小风险（非独立效果证据）、三域行数/
游戏数/质量/比例/事件率/损失贡献、决定覆盖、联合 FP/FN、条件 FP/FN、POS/NEG
错误率、选择性错误率。三种固定行动分别列出，不根据测试 Y 选“最好常数”；
强制二支另记完整评分与显式并列规则。逐游戏报告保留质量；对子组不能简单
平均条件游戏均值改变测度。决策记录不含 Y；离线统计本身仍是受限研究输出。

`evaluate_stability` 不接收 targets。概率半径是外生敏感性范围，与 [0,1] 相交，
不是将原 p 裁剪；枚举交点、端点和分段内部的可达选择域及全部最优集合，区分
所选规则稳定与最优集合是否多值。返回的区间最大 regret 取端点，假设 eta 在该
区间；它不是从 Y 估计出的真实 Bayes regret，也不是置信上界。
`cost_stability` 只评价调用者声明的合法矩阵，要求并列策略相同，逐个记录是否翻域，
不优化代价或选择赢家；非法扰动报错，不投影回非退化区。低支持不进入任何选域。

45 候选保持 rho∈{1/4,1/2,1,2,4}、kappa∈{.2,.5,.8}、h∈{.25,.5,.75}，
c_fp+c_fn=1；`candidate_losses(tie_policy=...)` 必须显式指定策略，不指定主矩阵。

## 6. 手工例子与实际验收

两个人工游戏：A 一条 (p=.9,Y=0)；B 三条 (.1,1)、(.5,1)、(.9,1)。
示例 c_fp=c_fn=1、b0=b1=.25，显式终端优先，强制二支显式 N 优先。
它是演示单位，不是正式主损失。

- 游戏等权实际风险 17/24≈.7083333；记录等权 9/16=.5625。
- 主测度 POS/BND/NEG 比例 2/3、1/6、1/6，决定覆盖 5/6；选择性错误率 .8。
- 固定 P/N 风险均 .5，固定 B 为 .25，强制二支为 5/6。
- 反转所有 Y，纯决策记录与预测最小风险完全不变，实际损失改变。
- 子组仅 A 与 B 第一条时质量 4/3、POS 比例 .75；不能按子组内每局等权改成 .5。

受控 CLI：

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/name_yuxiao/anaconda3/envs/3wd/bin/python -B \
  scripts/check_observer_dtrs_decisions.py
```

本轮已用 `--output docs/research/observer-dtrs-v1-d3b-controlled-acceptance.json` 首次保存。
该参数使用 exclusive create，不覆盖历史验收；再次核对可省略 --output。
文件保存两局例子、三种策略的边界、稳定性及五个实现/测试文件的实际 SHA-256，
不伪造 ToM/正式来源。

实际测试：本地 Python 3.12.13，针对性 **47 passed，0.50s**；相关回归
**399 passed，1 deselected，64.82s**。

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/name_yuxiao/anaconda3/envs/3wd/bin/python -B -m pytest -q \
  tests/observer_dtrs/test_decision.py tests/observer_dtrs/test_decision_offline.py \
  -p no:cacheprovider --maxfail=3

PYTHONDONTWRITEBYTECODE=1 /Users/name_yuxiao/anaconda3/envs/3wd/bin/python -B -m pytest -q \
  tests/observer_dtrs tests/canonical_collection tests/development_publication \
  tests/tom/test_dataset.py tests/tom/test_boundaries.py \
  -k 'not off_diagonal_readout_can_memorize_controlled_public_fixture' \
  -p no:cacheprovider --maxfail=3
```

覆盖三策略/45 候选独立风险 oracle、精确边界与 nextafter、弱序退化、正比例缩放、
NaN/Inf/越界、NumPy 标量和大成本、监督反事实、单类别/空决定、子组权重、
固定/二支对照、概率/成本扰动、D3a 的 regret 界与区间端点最大值。未声称全仓库
所有测试通过；仍排除此前 ToM 小模型训练案例。

## 7. D3c 接口和就绪结论

D3c 适配器应先验证 complete→contract→25 块→外层产物→D1c 的身份，之后对
每个模型/权重分别构造上述 ProbabilityInput，并显式另开监督。须保留 model_id、
weighting、source model/outer artifact digest、ToM/outer fold、dataset/protocol/split
锚点和样本上下文的外部 batch 绑定；当前内存评价器**不是来源验证器**。
四模型名称固定 OBS-EQ0/EQ1/LR-r/EF，不从总体均值或校准箱生成任何伪逐样本 p。

下载范围与原文件布局见 [D3c 手动材料清单](observer-dtrs-v1-d3c-download-plan.md)。
纯决策与受控评价已具备进入真实数据适配开发的条件；真实数据分析还需所选层级
的手动材料、来源验证和最终并列策略。当前不冻结 D3 正式执行协议，不启动真实
45 场景比较，不将同一 D2 开发数据称新独立测试。

新增 werewolf 生产模块尚未提交，既有正式 D2 source guard 会按其设计拒绝当前
生产源码状态；这是本轮不提交边界下的预期保护，不修改 guard、不绕过它，也不
用旧 OID 为新 D3 生产源码背书。未来正式运行前需固定新的 D3 源码 OID，同时
保留历史 D2 来源 OID 和所有 RNG/开发验证限制。
