# ToM–DTRS 动态发言规划：Phase-1 受控研究快照

状态：实现及人工 CPU 验收；不是已冻结的行动策略或真实模型资格证明。

## 实施来源与保护范围

本轮在 Downloads、Desktop、Codex attachments 和本地研究结果目录的有限深度
ZIP 清查中，没有找到包含 `planning_research.py` / `test_planning_research.py`
的独立补丁。因此按已审定接口自行实现，不声称使用了外部补丁。

W3 基点：`twd/mainline@7b07bb5ee2040e14991c167b0cbd64bcfab8b849`。
仅新增本文件、`werewolf/observer_dtrs/planning_research.py` 和对应测试。
既有九项 S1a/P0 成果、D1–D3、模型、Actor/Parser、recorder、提交核心均不修改。
源码身份由实际 HEAD 加新增文件字节摘要共同说明；未提交文件不冒称已属于 HEAD。

## 输入与输出

`analyze_pre(request, prediction, estimator, *, expected_deployment_digest,
tie_policy, proxy=None, expected_proxy_provenance=None)` 返回不可变字节快照。

- `request` 复用合法 `ShadowRequest`：冻结公共 PRE 和当前狼人的合法队友信息。
  必须由可信控制层使用原合法 observation 构造；数据类型本身不能证明调用者诚实。
- `prediction` 复用 `QPrediction`：完整 7×7、绝对席位的概率质量，绑定 PRE、
  public token plan 及 ToM 身份。加载器若输出 log-Q，应由原适配器仅指数转换一次。
- `estimator` 是已有只读 `FrozenEventEstimator`；其 ToM 必须与当前 Q 完全匹配，
  manifest 必须等于显式外部锚点。现有 reader 的 fixture / OOF / final 边界不变。
- 全体合法存活非狼观察者逐一复用 `observer_features`、原估计器、`decide`。
  输出当前 Q、条件属性、p、估计层级、每个成本的三风险、完整 Bayes 最优集合和
  显式并列规则选中的区域。死亡目标质量保留，单狼 m=0、dW=None。
- 完整候选仅来自 `generate_candidates`，包括同伴目标及末位发言合法计划。
  返回每个候选的 B2 结果或明确不可评价状态，不输出选中的计划。

## 成本来源不是任意字符串

入口不接受 `cost_source` 字符串或自填 LossSpec。使用原 D3 loader，要求协议冻结且
canonical digest 为
`0ebd6b97669c35e1d9201f88152da8919d87586864a751f3fd688edd3a11c4ab`。
全部 C00–C44 经原 `protocol.loss` 生成，明确允许正式主并列与敏感性并列策略。
每条记录保留原成本及摘要。不会把 S1a 上下文成本冒称为正式登记成本。

## 当前认知与 B2 分离

受控 consumer 必须显式包装为 `ControlledB2(consumer, capacity, fixture_id)`。
即使 consumer 的 provenance 使用与真实摘要相同形状的字符串，输出仍强制标注
`CONTROLLED_TEST_ONLY_NOT_SEALED_B2`。普通 duck-typed consumer 或仿制子类不能进入
真实封存路径。受控包装不能同时提供真实封存锚点。

真实路径只接受原 `CounterfactualToMConsumer` 的精确类型和独立的预期 provenance，
检查原版本、builder 和源码摘要；原构造器负责原 seal/runtime/checkpoint 检查。
本轮没有构造真实 consumer，也没有取得其真实数值或部署资格证据。
此路径不是 OS 沙箱，不能防止进程内恶意伪造 Python 对象；可信加载器仍是前提。

B2 复用原完整批次评价和外围 `wolf_suspicion_mass`，没有新增奖励：
它仍是预测嫌疑质量代理，不是 D2 的事件概率、放逐风险、因果效应或发言收益。
其存活条件化仅用于外围指标，不改变当前 Q 或 D1–D3 的原始特征。

当前 Q 与候选代理分别保存来源。入口不传 `current_predictor`，所有
`current_suspicion`、`delta` 均为 None；即使矩阵结构相同也不计算跨来源差值。
假想延续使用独立 continuation digest，不能伪装真实 PRE 或送入当前 D2。

## 能力与失败契约

- 无 proxy：完整候选标记 `PROXY_SOURCE_UNAVAILABLE`，不是零分或默认计划。
- 同阶段末位：保留完整候选与当前认知，零代理调用，标记不支持后继。
- parent token count + 3 超容量：完整候选不可评价，不截断、不调用模型。
- 其他非法 PRE、来源冲突、无资格输入：明确抛错，不用默认 Q / p / BND 补齐。
- 原批次评价异常或批次来源改变：全部代理结果为空，标记
  `BATCH_FAILED_NO_RESULTS`，记录错误类型；不发布未返回的部分结果，不重试。
  当前认知快照仍有效，但该 PRE 的候选评价不完整，不能据此选择行动。

旧 B2 仅支持同日同阶段的公开可推导下一位发言者。它不是整局环境模拟器。
本模块不调用 Actor、Parser、游戏 step、选择器或 Shadow 日志，不缓存游戏状态，
不拟合模型，不读监督/sidecar，不写 artifact。外部可信 provider 的运行资格仍须另验。

## CPU 验收与限制

新增测试使用人工公共事件 witness、手填概率部署表及人工 B2 张量；不执行整局游戏、
不加载真实权重、不训练 ToM、不拟合正式 D2、不采集 CAL/TEST。
覆盖单/双狼、普通/PK、死亡列、全部候选、末位、容量、D3 精确并列、任意成本
声明拒绝、来源错配、伪装封存 consumer 拒绝、整批失败、原代理数值一致、RNG/
PRE/Q 不变，以及语音实现/提交不可达。验收统计由项目外机器记录和日志提供。

建议受控回归（排除旧 sealed fixture 中会训练/封存人工模型的测试）：

```sh
PYTHONDONTWRITEBYTECODE=1 TMPDIR=/private/tmp \
  /Users/name_yuxiao/anaconda3/envs/3wd/bin/python -m pytest -q -p no:cacheprovider \
  tests/observer_dtrs/test_planning_research.py
```

通过仅证明组合、数值委托及隔离契约；真实封存 B2、Final-ToM/CAL 兼容、反事实科学
有效性、发言选择效果和胜率均没有由本轮证明。第一阶段不是搜索器或行动优化算法。

## 本轮实际执行记录

- 新增测试首轮 23 passed / 1 failed；失败由测试在 RNG 快照后导入旧 Actor
  模块造成。仅将测试准备导入移到 RNG 快照之前，生产实现不变。
- 修复后新增测试：24 passed，0 failed，0 skipped，2.47 秒。
- 相关回归（含上述 24 项）：402 passed，0 failed，0 skipped，16 deselected，
  22.91 秒。16 项主动排除会训练/封存或加载旧人工模型的生命周期测试，未将其
  冒称为本轮通过。其余测试使用人工数据/后端，未运行整局游戏或真实模型。
- Python 3.12.13，PyTorch 2.12.1，CPU；禁用字节码和 pytest cache，测试临时
  产物在 `/private/tmp`，没有写入正式数据、模型或实验目录。
- AST 解析、Git diff whitespace 与新文件尾随空白检查通过。
- 完整命令、JUnit、初次失败和重跑日志、源码 diff、全部原有文件及九项成果的
  前后 SHA-256 对照保存在本轮项目外验收目录；未 commit/push/merge。
