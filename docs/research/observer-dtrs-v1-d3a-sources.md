# Observer-DTRS v1 D3a：原始理论来源核对

状态：研究来源记录；2026-10-15；不冻结损失参数，不改变 D0–D2。

本记录核对原作者论文及期刊全文。下文明确区分文献已有结果与本项目的代入或推导；没有访问服务器。来源提供决策论依据，不替代本项目的数据证据。

## 1. DTRS 六损失、阈值及附加条件

Yiyu Yao，*Decision-theoretic rough set models*，RSKT 2007，pp. 1–12。[作者主页完整 PDF](https://www2.cs.uregina.ca/~yyao/PAPERS/jrs07.pdf)。已核验全文，重点为 §3.1–3.2、式 (10)–(14)、规则 (P1)–(B2)。

原文先按后验概率计算行动的条件期望损失，再取最小风险行动；并列最小值需要额外规则。六损失的逐列排序之外，还需式 (14) 的额外约束才能获得有序的两阈值。原文并不把“存在三个行动”当成“三个区域必然非退化”。这些是原文支持的范围。

以下是本项目的符号代入。令 $a=c_{\mathrm{FP}}>0$、$c=c_{\mathrm{FN}}>0$，并采用已提出的零正确判定损失矩阵。则

\[
\alpha=\frac{a-b_0}{a-b_0+b_1},\qquad
\beta=\frac{b_0}{c-b_1+b_0},\qquad
\gamma=\frac{a}{a+c}.
\]

规范性逐列约束为 $0\le b_1<c$、$0\le b_0<a$。在此条件下，

\[
\alpha>\beta
\iff(a-b_0)(c-b_1)>b_0b_1
\iff \frac{b_1}{c}+\frac{b_0}{a}<1.
\]

这时 $\beta<\gamma<\alpha$，BND 在概率轴上具有正宽度。若还要求 POS、NEG 均有正宽度的概率区间，应增加 $b_1>0$、$b_0>0$，从而 $0<\beta<\alpha<1$。这不保证实际预测样本在三个区域都有质量；后者是数据分布问题。

当比值之和等于 1 时，三条风险线在 $\gamma$ 相交，BND 至多是并列点；大于 1 时 B 从不严格最优。代码应以三风险 argmin 为通用定义，不能在未检查条件时机械套用两阈值。本文不擅自选定边界并列规则。

## 2. 静态拒绝与动态延期不能混同

Peter L. Bartlett、Marten H. Wegkamp，*Classification with a Reject Option using a Hinge Loss*，JMLR 9 (2008)，1823–1840。[期刊全文 PDF](https://jmlr.org/papers/volume9/bartlett08a/bartlett08a.pdf)。已核验全文，重点为 §1、式 (1)–(2)、§2 及 §4。

该文对称情形的错误成本为 1、拒绝成本为预先给定的 $d$；按 $\min\{\eta,1-\eta,d\}$ 选择行动，拒绝区间为 $[d,1-d]$。原文明确指出成本应由任务考虑预先确定，且阈值附近的概率质量影响学习风险。这里是预设成本下的静态拒绝决策，不是最优后续观测计划。

本项目的 $a=c=1,b_1=b_0=d$ 是上述结构的特例；严格三域要求 $0<d<1/2$。这只是结构核对，**不是建议冻结 $d$**。由于尚无下一次观测、转移机制、预算或后续价值函数，BND 只能解释为独立认知行动，不能宣称已求解动态延期、Probe 或狼队发言优化。

## 3. 概率评分、校准与具体决策风险

Tilmann Gneiting、Adrian E. Raftery，*Strictly Proper Scoring Rules, Prediction, and Estimation*，JASA 102 (2007)，359–378。[作者主页完整 PDF](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf)。已核验全文，重点为 §1、§3、Example 1–2。

原文区分概率校准与预测集中程度，说明严格恰当评分的真实分布最优性，并讨论 Brier 与对数评分。本文使用“损失越小越好”的方向；原文主要采用奖励方向。概率评分与某一给定行动损失不是同一个评价目标。

本项目因此保留以下解释边界：D2 较低的总体 Brier 或 Log Loss 不能单独证明在每个六损失矩阵下都获得更低决策风险。校准 $E[Y\mid q]=q$ 至多直接支持以 $q$ 为全部信息的行动优化，不意味着 $q=P(Y=1\mid X)$ 或已达到使用所有合法信息 $X$ 的 Bayes 风险。比较 EQ1/LR-r 的三支行动时应对同一固定矩阵直接计算配对实现损失，同时保留概率评价。

## 4. Plug-in regret：文献边界与本项目推导

Mark D. Reid、Robert C. Williamson，*Information, Divergence and Risk for Binary Experiments*，JMLR 12 (2011)，731–817。[期刊全文 PDF](https://jmlr.org/papers/volume12/reid11a/reid11a.pdf)。已核验完整正文，重点为 §4.4、§7.1、Theorem 25、Lemma 26、Corollary 28。原文将恰当概率损失的 regret 表示为 Bregman 散度，并给出成本敏感二元行动的 surrogate regret 界。其二动作定理不能未经证明当成该项目三动作定理。

下面给出本项目直接证明，**不归因于上述论文的原定理**。固定一个损失矩阵和同一目标分布；令真实条件概率为 $\eta(x)$，估计为 $q(x)$，$a(q)$ 为按 $q$ 取三风险最小值的行动，$a^*(\eta)$ 为按真实概率取最小值的行动。在第 1 节逐列约束下，三风险的斜率分别为 $-a,b_1-b_0,c$，最大跨度为 $D=a+c$。由于

\[
R_{a(q)}(q)-R_{a^*(\eta)}(q)\le0,
\]

对仿射风险之差展开可得

\[
0\le R_{a(q)}(\eta)-R_{a^*(\eta)}(\eta)
\le D\lvert q-\eta\rvert.
\]

在同一概率测度（例如既定游戏等权目标）下积分并用 Cauchy–Schwarz 不等式：

\[
\operatorname{Regret}\le D E|q-\eta|
\le D\sqrt{E(q-\eta)^2}.
\]

这里 $E(q-\eta)^2$ 是总体超额 Brier，而非拿每条实现标签 $y$ 代替未知 $\eta$。结果控制一个模型相对同一信息集合 oracle 的风险上界；上界较小不推出两个模型的实际风险排序，不提供数据集外独立泛化或置信区间。固定矩阵、已定义的 $\eta$ 和同一评价测度都是前提。

若 $|q-\eta|\le\epsilon$，且 $q$ 距两阈值均大于 $\epsilon$，在相同并列规则下区域不变。这是概率轴上的确定性证书，不是经验证的概率置信区间。支持量、Kish 有效支持、未见类回退、估计概率区间、损失扰动与 BND 应分别记录；不能用低支持自动替换 Bayes 行动，除非另立明确的安全策略协议。

## 5. 使用限制

本文核对的文献支持静态成本敏感判断的决策论形式；不能确定本项目六损失的规范性数值，也不能证明三域会改善游戏决策或胜率。成本比例、BND 行动价值与损失扰动范围仍需研究负责人确认。Chow (1970) 是上述拒绝分类文献引用的历史来源，但本轮未从官方或作者站点取得并独立核对其完整正文，故未把具体公式归因于该原文。
