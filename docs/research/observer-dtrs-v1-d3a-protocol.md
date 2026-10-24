# Observer-DTRS v1 D3a：六损失、三域与决策风险协议草案

状态：**PROPOSED_NOT_FROZEN_NOT_RUN**。日期：2026-10-15（Asia/Shanghai）。
本文件是理论和执行前协议建议，不是已冻结的成本数值，也不是正式三支实验结果。
仅新增文档与受控数学验证，不改变 D0–D2、冻结 ToM、概率估计器、数据或旧 Phase2。
原始文献支持范围见 [sources](observer-dtrs-v1-d3a-sources.md)，
D2 结果与证据缺口见 [D2 archive](observer-dtrs-v1-d2-results.md)。

## 1. 研究对象和认知行动

仍对合法样本 $x=(g,t,w,i)$ 判断
$Y_x=1\{\widetilde S_{t,i}\cap W_t\ne\varnothing\}$。
成功空报告保持六名非自身玩家的有效集合；不能为改善分类结果改变事件或标签。
主输入为 D2 外层的游戏等权 OBS-EQ1 事件概率 $p_x$，不是原始 r、Q、log-Q、
狼人身份概率或旧 Theta_AC。p 条件于其固定 ToM 来源、训练分箱与抽样测度。

P/POS：接受“该观察者有效嫌疑集合含存活狼人”的命题；
B/BND：暂缓肯定或否定；N/NEG：否定。三者都不是攻击、Probe、发言或忽略。
当前没有未来观测分布、等待时长或动态策略模型，B 的损失只是一次静态认知
行动的机会成本。不能称其为已证明最优的动态延期或将其解释为未来信息增益。

## 2. 损失的规范性假设

记 $a=c_{\mathrm{FP}}>0$、$c=c_{\mathrm{FN}}>0$，采用用户确定的形式：

\[
\Lambda=\begin{array}{c|cc}&Y=1&Y=0\\\hline
P&0&a\\ B&b_1&b_0\\ N&c&0\end{array}.
\]

a 是错误接受该命题的认知损失；c 是错误否定的认知损失。它们不是已测得的
胜率、狼队攻击收益或错误狼人身份判断的代价。$b_1$ 是命题为真仍暂缓的成本，
$b_0$ 是命题为假仍暂缓的成本；状态依赖允许二者不相等。

这一静态 Bayes 框架假设：有限非负、状态条件下确定且可比较的损失；线性期望
效用/风险中性；同一矩阵适用于事先指定的人口和测度。正确判断损失取零是当前
规范性基准。若情景要求非零正确判断成本、动态观测或行动外部性，须另立协议，
不能事后借用本矩阵声称已经建模。

通常要求 $0\le b_0<a$、$0\le b_1<c$，即暂缓分别优于错误断言。
**三个概率域都有正长度**还需：

\[
0<b_0<a,\quad 0<b_1<c,\quad
u+v<1,\qquad u=b_0/a,\ v=b_1/c.
\]

逐列排序本身不够。正长度也不保证实际预测在三域中都有非零样本；不得因某域
人数少而反向调节代价。零 b、等号或越界情形只作明确退化/对照，不静默修补。

## 3. 阈值与 Bayes 规则的严格推导

\[
R_P(p)=a(1-p),\quad R_B(p)=b_1p+b_0(1-p),\quad R_N(p)=cp.
\]

在上述严格逐列约束下，三条交线给出：

\[
\begin{aligned}
R_P\le R_B&\iff p\ge\alpha=\frac{a-b_0}{a-b_0+b_1},\\
R_B\le R_N&\iff p\ge\beta=\frac{b_0}{b_0+c-b_1},\\
R_P\le R_N&\iff p\ge\gamma=\frac{a}{a+c}.
\end{aligned}
\]

因此一般成对最小风险条件为：

\[
P:\ p\ge\max(\alpha,\gamma),\quad
B:\ \beta\le p\le\alpha,\quad
N:\ p\le\min(\beta,\gamma).
\]

这些是可并列的最优行动集合，而非自动互斥的标签。具体有

\[
\beta<\gamma<\alpha\iff b_0c+b_1a<ac\iff u+v<1.
\]

在非退化情形，拟议确定性约定为 **精确并列优先 B，其次 N，再次 P**：

\[
\delta_\Lambda(p)=\begin{cases}
NEG,&p<\beta,\\ BND,&\beta\le p\le\alpha,\\ POS,&p>\alpha.
\end{cases}
\]

端点选择是规范性约定，须在执行前确认，不是文献唯一规定。
当 $u+v=1$，$\alpha=\beta=\gamma$，B 只在该点并列最优，没有正宽 BND。
当 $u+v>1$，B 永不最优，规则退化为以 $\gamma$ 区分 N/P；不能错误地使用
颠倒的 alpha/beta 构造重叠 BND。$b_0=0$ 时 NEG 的正宽区间消失；$b_1=0$
时 POS 的正宽区间消失；两者都零时 B 在内部处处最优。

正确实现始终以三风险 **直接 argmin** 为规范定义；阈值只是满足适用条件的等价
捷径。NaN、无穷、负成本和越界 p 明确报错，不能裁剪真实 p 来掩盖契约错误。
浮点执行保存原始风险、全部计算上精确并列最优者及次小风险间隔；近并列容差
仅作诊断，不人为拓宽 BND，不把低支持或概率不稳定改判为 BND。
本轮工具使用 Fraction 精确有理数，独立核对成对交点、阈值分类与直接风险比较。

以上阈值可由 Yao 的一般六损失规则代入得到；非退化条件与规则依据见
[Yao 原论文 §3](https://www2.cs.uregina.ca/~yyao/PAPERS/jrs07.pdf)。

## 4. 如何设定损失而不反向优化结果

损失必须先回答“哪种认知错误更严重、暂缓分别有多大机会成本”，不能从测试集
求出使准确率、BND 比例、显著性或某模型排名最好看的矩阵。高事件基率不是
a/c 的设定理由；类不平衡已经在 p 中，不能不加解释再次按类频率加权。

建议先向领域研究者独立征询 cFP/cFN 的相对严重性，以及暂缓在两个真状态下
相对错误断言的成本。不向设定者展示按候选成本划分后的分组人数/成绩以引导选择。
若暂无外部依据，报告一组明确的规范性场景，不声称存在唯一“真实最优损失”。
可在看 D3 分配和标签风险之前确认一个锚点；若不能论证，则所有场景均作探索，
不制造单一主检验。

正比例缩放不改变行动，因此可统一令 $a+c=1$：

\[
\rho=a/c,\quad a=\rho/(1+\rho),\ c=1/(1+\rho),\quad
u=\kappa h,\ v=\kappa(1-h),\ b_0=ua,\ b_1=vc.
\]

| 参数 | 拟议场景范围，尚未冻结 | 含义 |
| --- | --- | --- |
| rho | 1/4、1/2、1、2、4 | FP/FN 相对成本 |
| kappa=u+v | .2、.5、.8 | 暂缓相对错误成本；越小一般越宽容 B |
| h=u/(u+v) | .25、.5、.75 | 暂缓的状态成本不对称性 |

共 45 个非退化场景，先声明全部展示、无最优场景搜索。对称示例 rho=1、kappa=.5、
h=.5 给 $a=c=.5,b_0=b_1=.125,\beta=.25,\alpha=.75$；**只是人工示例，不是
已接受的正式主损失**。kappa=1、1.2 和零暂缓成本另作数学退化对照，不混入主场景。
同一场景下所有模型使用相同损失；跨场景报告归一化风险，不能把“B 很便宜”
引起的低风险解释为更好的概率模型。若以后需要 phase/nW 特定损失，应先论证
新的规范性原因并另行登记，不能用子组结果选择成本。

## 5. 四种不同现象与支持审计

| 现象 | 定义及允许的用途 | 不能作的推断 |
| --- | --- | --- |
| BND | 给定 p、给定损失下 B 最小风险 | 不等于概率估计置信区间宽 |
| 低支持粒 | 外层训练节点的 distinct games、Kish、mass、回退路径 | 不得用测试标签重估支持或转 BND |
| 概率估计误差/不确定性 | 未知真概率与 p 的差异；如需区间须另建训练内程序 | 现有收缩不是精确 Beta 后验；Kish 不是独立伯努利样本量 |
| 决策不稳定 | p 或成本在预定扰动集变化时 argmin 改变 | 不等于实际错误或语义 BND |

支持从对应 `(ToM fold, outer fold, weighting, model_digest)` 的**训练模型**读取。
当前 EQ `predict_many` 只给 probability/level；D3b 审计适配器可结合合法 phase/nW/r、
外层模型的 edges/nodes 恢复命中叶和回退节点，记录 rows、games、mass、kish_games。
不能从全体 D1 数据重新分箱或重计节点。监督相关 positive_mass/正负游戏数放在
受限离线审计；区域接口本身只接收 p 与损失，不接收这些统计。
LR 无同义的 r_bin 支持，记为 NA，不能虚构 EQ 风格的置信度。

外层训练最小叶支持仍沿用 D2 的 games≥10、Kish≥5，不修改模型。D3 只描述
实际使用节点及层级；父层回退不保证概率可靠。任何额外支持区间或重拟合都不是
本轮 D3a 的隐含授权，须单独设计训练内验证。

## 6. 概率误差、风险后悔值与稳定性

令共同信息与抽样测度下的真事件概率为 eta。给定矩阵，真正 Bayes 行动为
$a^*(\eta)=\arg\min_a R_a(\eta)$，plug-in 行动为 $\widehat a(p)=\arg\min_aR_a(p)$。
本项目的三动作推导：三风险斜率分别为 $-a,b_1-b_0,c$，在逐列约束下跨度为
$D=a+c$。由 $R_{\widehat a}(p)\le R_{a^*}(p)$ 有

\[
0\le\operatorname{Regret}(p,\eta;\Lambda)
=R_{\widehat a(p)}(\eta)-R_{a^*(\eta)}(\eta)
\le D|p-\eta|.
\]

积分后 $E[Regret]\le D\sqrt{E[(p-\eta)^2]}$。这是真概率下、同一信息目标的界；
不能把单条 Y 当 eta、不能把样本风险差当已识别的 Bayes regret。总体 Brier
包含不可约条件方差，也不决定误差是否集中在某个损失阈值附近。
因此 LR 的较低 D2 Brier 不保证所有矩阵下风险更低；EQ1 的可解释粒也不保证更优。
相同 p 若只相对粗化信息校准，其最优性也只相对该信息层级，而非完整历史的真后验。

计划的**受控**诊断：

- 非退化时记录 $d_p=\min(|p-\beta|,|p-\alpha|)$ 和最小/次小风险间隔。
  若真 eta 已知落在半径 epsilon 内，且 $d_p>epsilon$，则这一概率扰动不会翻域；
  但没有概率区间覆盖保证时不能据此声称统计置信保证。
  对任意适用的逐列有序矩阵，唯一最优行动相对其余行动的最小风险差若
  $\Delta(p)>(a+c)epsilon$，也足以保证行动不变；因为每个成对风险差的斜率
  绝对值不超过 a+c。精确并列时 Delta=0，不能据此提供唯一行动稳定保证。
- epsilon∈{0,.01,.025,.05} 为外生误差半径示例；考察
  $I=[p-epsilon,p+epsilon]\cap[0,1]$ 上可达行动集合，含端点并列。
  它们是敏感性范围，不是估计出的 95% 置信区间。区域翻转率与可能风险变化独立报告。
- 若仅把 I 当假设集，可报告
  $\sup_{\eta\in I}[R_{\delta(p)}(\eta)-\min_aR_a(\eta)]$。
  固定行动下此差是若干仿射函数的最大值，故在闭区间端点取得最大值；这是情景
  上界，不是已测得的真实 regret。需改变行动为 minimax 时另立协议，不暗中替代 Bayes 规则。
- 另外检验规范性成本的 ±5%、±10% 扰动，以及预定 45 场景的转移矩阵；超出
  非退化约束的情况标明退化，不投影回“好看”的三域。跨矩阵差异不是同一效用下
  的模型优劣比较。临界附近的小概率误差可能大量翻转但成本差很小，须联合报告。

## 7. D3 实验对象、指标及配对比较

使用 D2 已封存外层预测，不重新训练、重新校准或按 D3 风险重选 EQ1/LR。
同一 1500 局/22685 条人口、完整游戏与来源折身份。EQ1 为解释性主机制，EQ0 为
主比较；LR 为必报次要概率基线，EF 为端点/平滑消融。D2 点值更好不允许隐藏 LR。
旧 Phase2 M0–M3 不进入此比较。

每个事先给定矩阵首先报告**含 B 成本的全人口实际损失**：

\[
\widehat L_M(\Lambda)=\frac1G\sum_g\frac1{n_g}\sum_j
\Lambda_{\delta_\Lambda(p^M_{gj}),Y_{gj}}.
\]

主比较 $\widehat L_{EQ1}-\widehat L_{EQ0}$；次要为 LR−EQ1、EF−EQ1。
这是模型间实际损失差，不称为对未知真 Bayes 的 regret。对矩阵未定的情况只报告
风险曲面，不汇报“挑出的最佳损失”或最显著的区间作为主结果。

| 指标组 | 必须报告 |
| --- | --- |
| 总体风险 | 实际损失、三项行动损失贡献、配对差；同时列模型内部预测最小风险，但不把它当独立效果证据 |
| 错误断言 | P(a=P,Y=0)、P(a=N,Y=1)；可补 P(Y=0\|POS)、P(Y=1\|NEG)，明确分母 |
| 覆盖与延期 | POS/BND/NEG 质量与原始行数/游戏数、决定覆盖 1−BND、BND 内事件率 |
| 选择性表现 | 仅 POS/NEG 上错误率与覆盖一起报告；零分母为 NA；不能删 B 后只报准确率 |
| 概率质量 | D2 Brier/LL、固定箱校准；阈值附近仅作预定窗口诊断，不在窗口内重校准 |
| 稳定与支持 | 风险间隔、阈值距离、概率/成本扰动翻转率、EQ1↔LR 三域转移、节点支持和回退层级 |

始终包含同一矩阵下的三种固定行动 P、N、B，以及**同一 p 的强制二支规则**
（阈值 gamma、精确并列选 N）作对照。列出三个固定行动的实际风险可以诊断便宜 B
造成的表观优势；若按测试 Y 选其中最小者，只能标为事后最佳常数下界对照，不能
称可部署基线。拟议主表直接列三者，不作这种选择。
三支的预测最小风险不大于强制二支的预测最小风险是数学性质，不能作为经验优越性证明。

游戏等权为主；记录等权敏感性使用 D2 独立的 record_equal 外层模型与预测。
子组含 phase×nW、五个 ToM 来源折、成功空/非空（后者仅离线审计），均保持
整局 $1/n_g$ 基础权重再按子组质量归一化。单类别、小支持和空行动组不删游戏。

统计：复用正式 D2 的完整游戏顺序和 2000×1500 bootstrap 索引及其摘要，维持
25 个 (ToM fold, outer fold) 分层。模型、矩阵及指标共享同一抽样，以整局损失和
计数重算比率。区间仅为固定预测/规则的病例组成稳定性，不覆盖重新选损失、
上游重训、下游选参和 CV 重叠依赖。多场景区间是逐点描述，**不是同时置信带**；
45 个区间中挑显著结果不构成预登记检验。若之后需要“全成本范围一致占优”结论，
须事先登记相应整体统计和多重比较控制，不能直接根据逐点区间宣布。

本轮不计算任何正式三域人数、真实样本决策风险或新置信区间。
用户现已提供 complete 的 record/manifest/索引，本地完整载荷校验及索引复现通过；
这不包含逐样本外层模型/预测/块及受限 targets，后者仍是正式 D3 执行阻碍。
不能由 D2 总体均值或校准箱均值重构逐样本决策。

## 8. EQ1 与 LR 可能出现的差异

EQ1 对当前训练粒分段常数，阈值可能使整个粒一起翻域；不同外层的同号 bin
不是同一粒。层级回退可将很多样本推向同一 p。LR 在状态内连续随 r 变化，
可产生更细的阈值穿越，也可能在少量阈值附近校准不足。
二者不同并不等于任一个错误。须保持相同损失、相同样本，报告转移质量、实际
配对损失及支持。不能为 EQ1/LR 分别选择损失以使各自最好，再称为公平对比。
尤其不能因 EQ1 的 BND 多就称其更审慎可靠，也不能因 LR 的 BND 少就称其更果断有效。

## 9. D3b 输入输出与访问边界（待实现）

建议新建独立 `werewolf/observer_dtrs/decision.py`（本轮不创建），不用旧 Router。

```text
LossSpec:
  schema/id/version, cFP,cFN,b1,b0, scale_normalization,
  normative_rationale, status, tie_policy, canonical_digest

ProbabilityDecisionInput:
  sample_id, game_id, canonical_PRE_identity, acting_wolf, observer,
  event_probability, probability_model_id/model_digest, weighting,
  prediction_artifact_digest, source_fold/outer_fold (offline provenance)

decide(probability: finite [0,1], loss: validated LossSpec) -> DecisionCore:
  raw_risks[P,B,N], minimizing_actions, chosen_region,
  alpha/beta/gamma or applicability_status,
  risk_gap, threshold_distance (nondegenerate only), tie/near_tie diagnostics

DecisionRecord = source identity + DecisionCore + loss digest
audit_support(legal_features, verified_outer_model) -> separate training_support_record
score_offline(decision_records, explicitly_opened_targets, game_population) -> risk_statistics
```

纯区域函数不接受 Y、真实报告、空报告标记、训练 target、Theta_AC、角色 sidecar
或支持记录；把同一合法 p 的监督换掉，区域输出必须不变。统计接口才接收受限 Y。
已验证角色知识仅用于此前合法狼人上下文；D3 不能重新读取 sidecar 来修正 p。
来源包装器严格联结 D1 sample_id 与 canonical PRE，不把 runtime 循环编号当 step_index；
七座位/Q 解释继续留在 D1b，不在 decision 内重算。
支持审计元数据不参与纯区域选择，不允许低支持自动进入 BND。未来若要支持门控、
概率区间决策或主动查询策略，另立模型 ID 和规范性协议。

正式适配器应核对完整 D2 锚点、scope、来源 HEAD、25 块/样本唯一覆盖、模型及权重；
把 DecisionRecord 与受限评分/支持审计分开发布，沿用现有 artifact 机制。D3 新源码
在未来正式执行前单独固定 OID；仍保留 D2 原模型来源 OID，不能重写历史 contract。
新的 D3 contract/summary 必须继承历史 ToM 全量 publication 摘要→训练 RNG
依赖、架构选择独立性未确证、固定预测 bootstrap 边界和后验开发分析状态；
不能因为 D3 只读取 p 就省略上游来源限制。

D3b 最小验收：损失/概率拒绝非法值，完整/退化阈值及端点并列，正比例缩放不变，
监督反事实隔离，缺失重复/错折/错模型预测拒绝，训练支持来自正确外层节点，整局
权重与子组分母，受控风险/翻转/区间 regret 上界，artifact 损坏与恢复不覆盖。

## 10. D3 的验证范围和进入 D3b 的条件

D2 外层预测的拟合隔离并不能使看过结果后的 D3 设计自动成为新的独立测试。
在这些游戏上选择成本、模型或解释窗口后再评价，均属后验开发研究。可以在
现在锁定矩阵和规则后做可复现的条件分析，但必须明确 exploratory/posthoc；
不能改写为预登记于 D2 之前的检验。要验证固定完整管线的独立泛化，须使用未参与
当前历史训练哈希/架构选择/损失设计的新游戏，或覆盖所有选择的完整独立外层。
当前也无从据此推断发言动作、策略价值或胜率改善。

进入 D3b 前需确认：

1. 是否接受上述规范性场景参数化、B 优先的精确并列约定；是否有独立依据指定
   主矩阵。若无，建议先不设唯一主矩阵，保留对称矩阵仅作说明，完整展示场景曲面。
2. 是否接受 EQ1 主机制、EQ0 主比较、LR 必报次比较，以及全人口实际损失主指标；
   全部当前 D3 统计标为固定历史产物下的后验开发分析。
3. complete 的 JSON/manifest/索引已到位并完成其本地字节验证；仍需用户手动提供
   contract 与外层模型/预测/受限 targets 的本地位置。到位后校验来源和完整样本身份，
   再登记 D3 执行配置；本轮禁止主动服务器访问，不会替代用户传输。

数学原型已足以指导 D3b 的纯决策层实现；损失规范、完整结果来源与正式执行配置
尚未冻结，因此当前不具备直接启动正式 D3 比较实验的条件。

## 11. 本轮实际受控验收

本地 Python 3.12.13，`/Users/name_yuxiao/anaconda3/envs/3wd/bin/python`。
数学工具为 `scripts/check_observer_dtrs_decision_theory.py`，不导入生产决策模块、
不接收正式数据路径、不拟合任何模型。受控结果见
[controlled-validation JSON](observer-dtrs-v1-d3a-controlled-validation.json)。

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/name_yuxiao/anaconda3/envs/3wd/bin/python -B \
  scripts/check_observer_dtrs_decision_theory.py

PYTHONDONTWRITEBYTECODE=1 /Users/name_yuxiao/anaconda3/envs/3wd/bin/python -B -m pytest -q \
  tests/observer_dtrs/test_decision_theory.py -p no:cacheprovider --maxfail=1

PYTHONDONTWRITEBYTECODE=1 /Users/name_yuxiao/anaconda3/envs/3wd/bin/python -B -m pytest -q \
  tests/observer_dtrs tests/canonical_collection tests/development_publication \
  tests/tom/test_dataset.py tests/tom/test_boundaries.py \
  -k 'not off_diagonal_readout_can_memorize_controlled_public_fixture' \
  -p no:cacheprovider --maxfail=3
```

实际结果：新增测试 **20 passed，0.88s**；相关回归 **352 passed，1 deselected，64.34s**。
排除项仍是此前排除的 ToM 小模型训练测试，不声称全仓库所有测试通过。
45 个规范性场景、5 个退化对照，完成 5136 次阈值/argmin/缩放核对、22050 次
regret 界核对、4200 次区间情景核对。有限网格验证不能代替一般数学证明。

精确反例：对称示例损失下，真 eta=.74；q=.76 的期望 Brier 比 q=.70 更低，
但前者选择 POS、真实风险 .13，后者选择 BND、真实风险 .125；对应 regret 为
.005 与 0。它不是正式 EQ1/LR 的观测结果，而是对“Brier 排名推出任何成本下风险
排名”的反例。

另外实调 `source_snapshot(formal=True, expected_revision=3ff578f...)` 成功，
117 个生产源文件满足已提交 guard，冻结 D2 protocol payload 摘要保持
`c76940b446ffaa3dc9e873620fe775a3b64db8544d0c5190a49778a2e48a6c20`。
HEAD、已有 tracked 文件、暂存区和原 stash 均未变化。只新增本文等阶段文件，
未 commit/push，未访问服务器，也未执行正式 D3 分类、风险比较或新 D2 实验。

## 12. 完整 D2 summary 到位后的设计核对

下述数值来自本地已通过 `verify_artifact` 的正式 complete，详见
[D2 报告](observer-dtrs-v1-d2-results.md)和[完整表格](observer-dtrs-v1-d2-results-tables.md)。
contract/块来源尚未完整回放，不能把 summary 字节校验描述为全流程独立复核。
本节没有变更前述 45 场景、并列规则建议或任何 D2 冻结参数，也未冻结主损失。

1. 两种测度和五个来源折中，EQ1 的 Brier 点值均优于 EQ0；但成功空报告组的
   EQ1 Brier .02710596 高于 EQ0 .02390903。事件及空报告定义保持不变，不因该
   子组结果改标签、选损失或对在线输入按不可见报告状态分流。
2. LR 的游戏等权 Brier .09041031、LL .30505714 均低于 EQ1 的 .09586629、
   .31379228，但十箱 ECE .02604078 高于 EQ1 .00900910。D3 必须同时呈现概率
   评分、描述性校准与给定损失下的实现风险，不能依据其中一项剔除不利的比较模型。
   现有 JSON 只存 LR−EQ0 区间；LR−EQ1 区间仍需逐游戏数据，不能作区间相减。
3. EF 有 15 条（记录等权 13 条）精确错误端点；未裁剪 LL 为无穷。D3 风险仍须
   使用原始 [0,1] 概率，有限六损失可正常评价 p=0/1，不沿用 LL 裁剪来移动决策。
4. EQ1 的 [.1,.2) 校准箱仅 7 个测试游戏，预测均值 .17698、正例率 .45971；
   [.5,.6) 为 .52250 对 .46759。它们提示关注成本阈值附近的概率误差，但固定十箱
   不能定位精确阈值穿越，不能用整箱均值替代逐样本 p。7 个测试游戏也不是
   训练信息粒支持不足的证据；后者仍须读取对应外层节点。
5. 全局 bias 近零掩盖成功空/非空组的相反偏差，ECE 又可能受高概率大质量箱主导。
   这支持保留阈值距离、风险间隔和支持的分开诊断，不支持把 BND 改为低支持或
   “校准差”的替代标签，也不为任何具体 cFP/cFN 或 b0/b1 提供规范性数值依据。

剩余数据需求缩小为 contract 与逐游戏/逐样本外层材料，而非重复索取已到位的
complete 三个文件。仍不主动读取服务器，不启动正式 D3 分类或发言策略实验。
