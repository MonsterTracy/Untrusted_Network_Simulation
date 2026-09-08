# Phase-1 implementation report

审阅基线：`c8db59756ee5a02b8a79fee161d65254fa5fca19`（已完成 Commit 5）。
本报告涵盖该基线之后的完整最终文件状态。实施和审阅阶段曾包含早期 staged
WP6 candidate、后续 unstaged 修改及新文件；最终提交以完整审阅状态为准。

CONTEXT.md、ADR 0001–0021 和 approved specification 未修改。
它们在本次任务开始时已是 untracked authority documents，不计入以下实施文件清单。
当时状态（WP6–15 implementation 完成时）：尚未创建该阶段的 Git commit，
也没有重写 Commit 1–5。

历史状态记录（Commit B final staging 前）：Commit A 当时已提交为
`1e5fb6d9f13ef295f333d8c92120ee2591d83459`，message 为
`Complete the canonical Classic7 ToM mainline cutover`。
此时 staging area 为空，working tree 仅剩预期 35 个 Commit B
authority/docs/CI changes（包括 documentation-only deletions）。
以上仅记录当时的暂存前核验时点，不代表最终 repository state。

最终提交状态：WP6–15 functional implementation、authority docs、CI 和
documentation 已共同形成一个 Phase-1 mainline commit：
`Complete the Phase-1 Classic7 ToM mainline`。
其直接 parent 为上述 `c8db597`；此前 functional/docs 两个提交属于历史审阅过程，
不再作为当前主线的两个独立提交。Commit 1–5 及更早历史保持不变。

## 1. Final architecture / dataflow

```text
Classic7 Runtime（Night0、固定 2/3/1/1 角色；只生成研究数据）
→ Canonical Collection（唯一 PRE / observation / handoff / V1 owner）
→ immutable Game Bundles（public / audit / private）+ durable Attempt Ledger
→ Development Publication（全部 plan-eligible successes）
  ├─ public games + complete PRE/V1/belief records
  ├─ deterministic whole-game 5-fold manifest
  └─ restricted Role Sidecar → Primary Population Selector → boolean eligibility
→ CanonicalToMDataset（唯一 full-prefix tensorization / q conversion）
→ immutable Experiment（capacity、temporal bytes、masks、schedule、initial states）
→ paired Implicit / Explicit Primary training（5 folds × 2）
→ all-ten terminal checkpoint-set seal
→ held-out shift-0 predictions
→ Primary / All-Alive named evaluations（2 × 2 cells）
→ game-macro KL + paired game-cluster bootstrap reports
```

Collection 保留唯一 authoritative construction；Publication 只验证/冻结；
Dataset 消费同一 Structured Token Planner。All-Alive 只评价对应的 Primary
checkpoint，不另行训练。Primary targets 是 realized non-wolf cognition，
可以包含合法的 Seer/Witch private-state effects。

## 2. Files added / modified / deleted

相对于基线：新增 **38**、修改 **39**、删除 **79** 个文件，
包括本报告。逐文件清单见附录；迁移到当前目录的测试按删除旧文件/新增新文件计数。

主要新增为 `werewolf/development_publication.py`、`werewolf/tom/`、
`werewolf/cli.py`、独立 current speech validation，以及相应 contract/E2E tests。
公共 artifact I/O、runtime wiring、Bundle PRE coverage 验证和当前文档同步修改。

## 3. Major legacy paths removed

- 删除整个 `script/twd_tom/` Python runner 集合，包括三个 obsolete materializers；
  没有 archive、wrapper、alias 或 migration namespace。
- 删除 `werewolf/models/twd_tom/` 中旧 Dataset/dense Dataset/schema/scope/private/
  GPT-2/V2/checkpoint/inference/loss/metrics 实现，以及它们的旧 tests。
- 删除 V2/shadow 配置、final-fit/sealed runners 与旧 server/sealed 文档。
- 删除旧 agent registry、RandomAgent/抽象旧 Agent、独立 random rollout CLI、
  旧单 backend 配置转换/API 和旧 gameplay prompt 分支。
- 固定 Classic7 Witch 规则；删除不能属于该环境的 Guard 执行分支。
- 仍有当前用途的 private-runtime knowledge、public events、agent assignment 与
  V1 parser cases 测试迁移到当前 runtime/collection/speech 目录，不保留旧 ToM consumer。

删除内容是受 Git 跟踪的源码/测试/配置/文档，可从上述基线恢复。
没有读取、迁移或删除历史 sealed trajectories、speech、belief 或研究 artifacts。

## 4. Remaining executable entry points

唯一 root CLI 是 `uns`，与 `python -m werewolf.cli` 指向同一 implementation：

| Command | Responsibility |
| --- | --- |
| collect | 先 durable claim，再调用生产 runtime；唯一 canonical collection |
| publish-development | 从 verified completed ledger 发布完整 Development Game Set |
| prepare-experiment | 冻结 paired experiment 所需全部 protocol/artifacts |
| run-development-oof | 两条训练 lineage、十个 terminal seal、四个评价 cells |
| validate-artifact | owner-aware semantic/provenance validation；无 repair |

`run_random.py` 仅保留 runtime assembly/recorded game loop，不再是 CLI；
其 game loop 要求 canonical recorder 和 call audit。没有 All-Alive training、
generic scope、private-conditioned 或 historical evaluation entry point。

## 5. Information-boundary verification

| Boundary | Verification |
| --- | --- |
| PRE/V1 | 保留完整 cross-day prefix、terminal turn_start；Bundle/publication 检查覆盖与顺序；无下游裁剪/重解析 |
| Public/private | Publication public view 不打开 restricted sidecar；Dataset/model 不导入 Role Sidecar/private evidence |
| Role semantics | 实际角色判断只在 Primary selector；typed population loader 复用 selector 核验 membership，角色值不离开该边界；下游只获得 boolean masks 与 provenance digests |
| Supervision/model | eligibility/observed 控制 CE/metrics，不进入 forward；输出 diagonal 为结构性零 |
| All-Alive | 公共 alive mask；不读 sidecar；无训练入口；共享 Primary checkpoint/prediction digest |
| Outer-fold isolation | 实际 training worker 的 file-read spy 禁止 held-out games、stress masks；typed population validation 可读取 Role Sidecar，但 public records 严格限于请求的 game；Dataset/model/trainer 不新增 role input |
| Preflight | 仅在训练前，或验证 all-ten seal 后允许完整记录审计；训练途中 CLI validation 在读取前拒绝 |
| Paired training | 同一 canonical trainable bytes、schedule、rotation、RNG/optimizer protocol；相同 fixed step budget |
| Recovery/seal | source/runtime identity、immutable recovery chain、log/RNG binding、failure marker、十个 terminal 均验证 |
| Evaluation | seal-first；shift 0；不读取 optimizer/recovery state；无 TTA、selection、retraining |
| Gameplay | gameplay/runtime/speech 没有 trained ToM import 或 prediction/control feedback |

## 6. Fold game-set digest finding

已修复 `assign_development_folds`：在排名之前，独立对传入的、保持 ledger 顺序的
`(game_id, bundle_digest)` 序列计算 Development Game Set digest，
要求与声明值精确一致，而不是只把未核验的 digest 当作 hash 输入。

负例位于 `test_five_fold_assignment_is_deterministic_balanced_and_whole_game`：

- 改变某个 bundle digest、沿用旧 game-set digest → fail closed；
- 不改变 games，但传入错误 game-set digest → fail closed。

原 WP6 candidate 的 PRE coverage/order 修复保留于 Bundle/publication validation，
没有用修复、补全或重新构造 PRE prefix 替代。

### Final Working-Tree Review findings 修复范围

本轮只修改已确认的四项 findings，不重做 WP6–15，也不改变科学决策。

- **P1 Primary membership**：原 loader 只验证 boolean/alive/provenance，
  无法拒绝重算 digest 后错误选入狼人的 mask。现在两个 Primary loader 均在
  `population.py` 内打开对应 game 的 public partition，复用唯一
  `select_primary_population`，精确比较 rows 和 Sidecar identity。
  `open_role_sidecar` 以完整 publication manifest 的 game identities 验证 truth
  coverage，不通过打开其他 public partitions 来验证。错纳狼人和漏选非狼人
  都必须拒绝；All-Alive loader 不调用这一角色验证。
- **P2 已存在 evaluation artifacts**：原 CLI 只验证 experiment 和 checkpoint
  seal。现在 seal 后复用 `open_predictions`，并通过 writer/validator 共享的
  fold-score、aggregate 和 paired-report 计算核对已有 payload。
  `validate_existing_evaluation` 只读，不调用 publisher，不创建缺失的纯评价输出；
  已存在的 aggregate/report-set 必须有完整 parent reports。
- **P2 Publication inventory**：完整 CLI validation 先复用公共
  `open_artifact_envelope` 验证全部文件 inventory，拒绝 root/restricted extras，
  再执行原 public/sidecar typed readers。Dataset-facing public reader 未改为
  private reader，仍允许 restricted sidecar 物理不可用。
- **P2 standalone model state**：CLI 不再只调用底层 tensor-container validator，
  而是用唯一 Qwen2 trainable graph 复用 `load_model_state` 的 key/shape/dtype/
  finite-value/schema/byte-restore checks。仅用于 state 校验的对象不加载或生成
  temporal table；缺少 verified temporal provider 时 forward 显式拒绝。
  正式 experiment 仍只有原先两个 temporal conditions。

原 `no restricted read anywhere in the worker call stack` 测试限制已改为精确的
typed population boundary：允许唯一 population owner 验证角色，同时保留
held-out public-game / stress-mask 禁读，以及 All-Alive、Dataset/model 的 private
隔离检查。没有把 role input 添加到 trainer 或 representation。

### Terminal checkpoint seal typed-state finding

最终完整 working-tree review 发现 terminal checkpoint 和 final recovery 的 model
state 在 seal 前只经过通用 tensor-container validation；因此一套自洽但不属于固定
Qwen2 graph 的 tensor metadata 可能被错误 seal。现在：

- `verify_terminal` 复用 `state.validate_model_state`，在检查 terminal provenance、
  training log 和 recovery binding 之前，先把 terminal tensors 恢复到唯一固定的
  `ObserverConditionedToM` trainable graph；
- final recovery 的 decoded model tensors 在参与 terminal-equivalence 证明前，复用
  `state.validate_model_tensors` 和同一个 `restore_model_tensors` typed gate；
- 原有 recovery-chain provenance、terminal ↔ final recovery raw tensor-byte equality、
  fixed-step 和 all-ten seal checks 均保留，没有由 typed validation 替代；
- inference 不再是首个发现 wrong graph 的位置，wrong terminal/recovery 在 seal
  boundary fail closed。

该修复只修改 model-state owner、seal consumer 及其公开 seam tests，没有增加另一个
graph/schema validator、兼容分支或 experiment condition。

## 7. Tests and exact results

最终完整环境命令：

```sh
PYTHONDONTWRITEBYTECODE=1 conda run --no-capture-output -n 3wd python -B -m pytest -p no:cacheprovider -q --tb=short
```

结果：**468 passed, 169 subtests passed in 1130.52s (0:18:50)**，exit 0；无 skipped/failed。
相对第一次 Final Working-Tree Review 时的 434 项共新增 34 个 cases；terminal
typed-state finding 本轮新增 5 个 cases，未删除测试。

最终套件包含：

- Ledger 全部 crash points、顺序/唯一性、无覆盖、interrupted consumption、
  unsupported durability；补充新建 collection/run ancestors 的 fsync 验证。
- Bundle/publication 的 PRE coverage/order、digest、role/privacy、complete-case、
  plan closure、fold isolation/statistics 验证。
- full-prefix Dataset、empty/nonempty/unobserved q、rotation inverse/mask metamorphism。
- 七 shift 完整覆盖、round-independent offset、partial-batch coefficient、
  unequal-row-count game-balanced CE、zero-row fail。
- Walsh/Day-Code canonical bytes、capacity/range、metadata/parent/schema/forbidden fields、
  parameter parity、非 self simplex、canonical initial tensor graph/state。
- experiment preflight、CLI prepare/validate、source drift、held-out access gate。
- 五类真实 test-worker process crash：optimizer 后、partial staging、
  publish 前、publish 后、durable publication 后；恢复结果逐文件与 uninterrupted
  deterministic reference 一致。损坏 latest recovery 禁止回退/重跑。
- invalid terminal evidence：failure marker、RNG drift、缺失/损坏 recovery、
  recovery scheduler/schema drift、wrong-graph terminal、wrong-graph final recovery，
  以及 terminal/recovery 同时使用同一错误 graph；全部禁止 checkpoint-set seal。
- 独立五局、跨多日真实 runtime loop + deterministic external adapters 的完整 E2E：
  Collection → Publication → 两条训练 lineage → 十个 checkpoints → 四个 reports；
  显式断言 claim/bundle/publication/fold/sidecar/experiment/checkpoint/prediction/report digest chain。
- paired bootstrap 的手算区间、负差值、same-row Uniform reference、report naming。
- 实际 CLI import closure、forbidden legacy paths 与 current runtime regressions。

过程中单独完成的定向结果（已被最终 suite 覆盖，不额外计入总数）：

| Group | Exact result |
| --- | --- |
| Primary membership / preflight / population / Publication（首轮） | 26 passed in 16.37s |
| CLI / Primary preflight / population / training / Publication | 37 passed in 36.08s |
| standalone state / typed state / model | 11 passed in 3.72s |
| OOF payload corruption / CLI / population | 14 passed in 230.55s |
| rehashed report corruption / partial output / missing parent | 14 passed, 8 deselected in 155.77s |
| terminal/final-recovery typed graph seal cases | 5 passed, 8 deselected in 30.65s |
| training / recovery / checkpoint seal / standalone typed state | 29 passed in 95.54s |

新增负例先分别暴露了错误 Primary membership（4 个）、Publication extras（2 个）、
非 Qwen2 standalone state（1 个）和被忽略的 evaluation payload（7 个），
随后在同一公开接口转为通过。另覆盖 report 重算 digest 后的 parent/schema/
forbidden-field/score 错误、部分纯评价输出、缺失 report parent、只读验证不写文件。
terminal finding 的负例进一步保持合法 terminal 的 raw tensor bytes 不变，只把
terminal/recovery tensor metadata 重解释为自洽 wrong graph，从而独立证明通用
container/digest validation 不能替代 fixed-Qwen2 typed validation；修复前两条 tracer
tests 分别以 `DID NOT RAISE` 暴露 terminal 和 final-recovery seal 缺口，修复后全部通过。
一次中间 OOF 测试因运行期间 source fingerprint 变化被既有 provenance gate
拒绝，未作为验收结果；随后在稳定源码上重跑通过。上述最终 full suite 期间
实现源码保持不变。这些中间 red 结果不是最终 acceptance。
测试使用 deterministic fixtures/外部 fakes，不调用付费
backend，不以模型性能作为 Phase-1 scientific conclusion。

## 8. Compile / diff checks

内存 `compile(source_bytes, path, "exec")` 检查全部 **100** 个当前 Python files，
不写 bytecode；结果通过。以下命令均 exit 0：

```sh
git diff --check
git diff --cached --check
git diff HEAD --check
PYTHONDONTWRITEBYTECODE=1 conda run --no-capture-output -n 3wd python -B -m werewolf.cli --help
```

`3wd` 的 pytest、torch、transformers 完整可用。mypy/pyright 未安装，
没有将 compile validation 描述为静态 type checking。CUDA/跨硬件 bitwise
identity 未作为本次验证结论；frozen contract 本来不要求跨硬件最终 checkpoint 一致。

## 9. Anti-stacking / dead-code / reachability audit

- 当前 scientific transformations 均有唯一 owner，见 architecture/ownership 文档；
  Collection construction 与 Publication/Dataset defensive validation 未混为重复 transformation。
- fixed-Qwen2 key/shape/dtype/finite/state-restore semantics 仍唯一位于
  `werewolf/tom/state.py`；terminal 与 final recovery 只复用该 typed gate。
- Publication 无 Dataset import；Dataset 无 population/role/private import；
  actual-role string comparison 在 ToM 下只存在于 Primary selector。
- 对全部现存 production Python modules 做 AST import-closure test：
  **51** 个模块均从唯一 CLI 的当前 import graph 可达。目录级用途逐项写入
  `docs/repository_structure.md`，无只有历史用途的非测试模块留下。
- CodeGraph 用于现有 entrypoint 调用关系审计；其索引未包含新 untracked
  `werewolf/tom/` 文件，因此新增模块的闭包由实际 source AST/test 验证，
  不把索引缺失当作不可达证据。
- 对 live source/config/entry points 搜索旧 materializers、models.twd_tom、
  V2/shadow、private_conditioning、generic scope、pilot、best.pt、
  final-fit/sealed runners：没有 live implementation/import/entry point。
- `fallback_used` / `fallback_action_count` 仅为保留的拒绝证据：
  canonical 必须 false/zero；没有 fallback 执行分支。
- `OpenAICompatibleBackend` 是当前 backend wire protocol，
  `CustomLoggerAdapter` 是日志上下文，不是历史 API compatibility layer。
- 当前 checkpoint-set seal 是 development training/evaluation barrier，
  不是已删除的 historical final/sealed evaluation lineage。
- 前一次 implementation report 的“没有 remaining actionable finding”被
  Final Working-Tree Review 的四项 findings 纠正；本报告以修复后的测试和
  scoped re-review 为依据，不把此前的完成声明当作验证证据。
- 本轮 scoped re-review：Standards **0 actionable findings**；Spec
  **0 actionable findings**。该结果只覆盖本轮修复，不替代下一次完整
  Final Working-Tree Review，也不是 staging/commit 授权。

## 10. Remaining spec deviations

前一轮四项 findings 与随后发现的 terminal checkpoint seal typed-state finding 均已
关闭。定向测试和完整套件未发现修复范围内的 remaining spec deviation。最终完整
working-tree acceptance 仍交由下一次 Final Working-Tree Review 判断。没有恢复 legacy consumer，
没有新增实验条件、fallback、compatibility namespace 或 generic population switch。
已冻结科学/架构文件未修改。

前一轮四项 findings 修改 8 个 implementation files、5 个 test files 和本报告；
本轮 terminal finding 只追加修改 `state.py`、`training.py`、`test_training.py` 和本报告。
Dataset、Collection/PRE constructor 和 scientific protocol 未修改。
当时状态（Final Staging 之前的历史审阅）：HEAD 为上述 Commit 5，
early candidate staged entries digest 为
`7aa1700bd80f33cc3e836d97f659daefc1bcdd2372568a3d0067962e3a44125d`，
该轮修复没有执行 staging、reset、restore 或 commit。以上是历史审阅状态，
不代表最终 HEAD 或 index；最终提交状态见报告开头。

## 11. Contradictions

未发现使 frozen design 无法实现的 specification/repository contradiction。
terminal seal 问题是 fixed model-state semantic validation gate 的实现遗漏，已复用既有
typed state contract 修复；它不要求修改 scientific semantics 或 artifact lineage。

## 12. Historical staging boundaries and final commit

历史审阅阶段曾采用以下两个 staging/commit 边界；它们不构成最终主线的提交拆分：

1. **Complete the canonical Classic7 ToM mainline cutover**：
   合并所有相互依赖的 production source、exports、setup CLI、producer/consumer
   replacements/deletions 和对应测试。Publication/新 ToM consumers 与旧 materializers/
   runners/models 的删除不能拆成制造临时 compatibility 状态的提交。
   artifact/recovery/ledger 修复与相应 acceptance tests 也放在这一完整 functional boundary。
2. **Document and validate the current Phase-1 execution path**：
   README、configs/current contract/architecture/module inventory、旧文档删除、
   本报告和 CI compile-path 更新；纯文档/CI closure，不引入另一个 implementation path。

以上记录当时根据实际相依 diff 确定的审阅边界。
当时状态（Commit A staging 前）：index 混有之前 staged candidate，
最终工作树还有 unstaged/untracked 新实现，因此不能直接提交当时的 index。
随后曾按审阅后的最终文件状态分别完成 Commit A 与 Commit B。
最终两者合并为一个 `Complete the Phase-1 Classic7 ToM mainline` 提交，
统一包含 functional code/tests、authority/spec、CI 和 documentation。
合并只修正本报告的 Git-history 表述，不改变其余最终文件内容或审阅职责边界。

## Appendix: complete file inventory

### Added (38)

- `docs/phase-1-implementation-report.md`
- `tests/canonical_collection/test_public_event_environment.py`
- `tests/development_publication/test_development_publication.py`
- `tests/runtime/test_agent_assignment.py`
- `tests/runtime/test_fixed_roles.py`
- `tests/runtime/test_observer_private_knowledge.py`
- `tests/speech/test_speech_perceiver_cases.py`
- `tests/tom/test_boundaries.py`
- `tests/tom/test_cli.py`
- `tests/tom/test_dataset.py`
- `tests/tom/test_experiment.py`
- `tests/tom/test_full_mainline.py`
- `tests/tom/test_model.py`
- `tests/tom/test_oof.py`
- `tests/tom/test_population.py`
- `tests/tom/test_preflight.py`
- `tests/tom/test_protocol.py`
- `tests/tom/test_recovery.py`
- `tests/tom/test_run_records.py`
- `tests/tom/test_state.py`
- `tests/tom/test_temporal.py`
- `tests/tom/test_training.py`
- `werewolf/cli.py`
- `werewolf/development_publication.py`
- `werewolf/speech/validation.py`
- `werewolf/tom/__init__.py`
- `werewolf/tom/dataset.py`
- `werewolf/tom/evaluation.py`
- `werewolf/tom/experiment.py`
- `werewolf/tom/model.py`
- `werewolf/tom/population.py`
- `werewolf/tom/protocol.py`
- `werewolf/tom/reporting.py`
- `werewolf/tom/run_records.py`
- `werewolf/tom/scoring.py`
- `werewolf/tom/state.py`
- `werewolf/tom/temporal.py`
- `werewolf/tom/training.py`

### Modified (39)

- `.github/workflows/ci.yml`
- `configs/README.md`
- `docs/architecture.md`
- `docs/collection_contract.md`
- `docs/public_speech_event_contract.md`
- `docs/repository_structure.md`
- `docs/tom_contract.md`
- `README.md`
- `run_random.py`
- `setup.py`
- `tests/agents/test_agent_backend.py`
- `tests/agents/test_backends.py`
- `tests/agents/test_named_backends.py`
- `tests/artifact_io/test_canonical_artifacts.py`
- `tests/canonical_collection/test_attempt_ledger.py`
- `tests/canonical_collection/test_game_bundle.py`
- `tests/canonical_collection/test_production_collection.py`
- `tests/runtime/test_runtime_config.py`
- `tests/speech/test_private_belief_perceiver.py`
- `tests/speech/test_speech_perceiver.py`
- `werewolf/agents/__init__.py`
- `werewolf/agents/gpt_agent.py`
- `werewolf/agents/llm_agent.py`
- `werewolf/agents/prompt_template_v0.py`
- `werewolf/artifact_io/__init__.py`
- `werewolf/artifact_io/canonical.py`
- `werewolf/artifact_io/tensor_state.py`
- `werewolf/backends/__init__.py`
- `werewolf/backends/factory.py`
- `werewolf/canonical_collection/__init__.py`
- `werewolf/canonical_collection/attempt_ledger.py`
- `werewolf/canonical_collection/call_audit.py`
- `werewolf/canonical_collection/game_bundle.py`
- `werewolf/canonical_collection/production_runtime.py`
- `werewolf/canonical_collection/runtime.py`
- `werewolf/envs/werewolf_text_env_v0.py`
- `werewolf/runtime_config.py`
- `werewolf/speech/private_belief_perceiver.py`
- `werewolf/speech/speech_perceiver.py`

### Deleted (79)

- `configs/twd_tom_shadow_deepseek_v4_flash.yaml`
- `docs/sealed_evaluation_protocol.md`
- `docs/server_deployment_checklist.md`
- `script/twd_tom/__init__.py`
- `script/twd_tom/audit_belief_label_repeatability.py`
- `script/twd_tom/audit_canonical_belief_data.py`
- `script/twd_tom/audit_dense_belief_dataset.py`
- `script/twd_tom/audit_shadow_speech_parser.py`
- `script/twd_tom/collect_canonical_trajectories.py`
- `script/twd_tom/eval.py`
- `script/twd_tom/export_belief_worst_cases.py`
- `script/twd_tom/materialize_canonical_belief_dataset.py`
- `script/twd_tom/materialize_development_folds.py`
- `script/twd_tom/materialize_role_sidecar.py`
- `script/twd_tom/run_annotation_v2_ablation.py`
- `script/twd_tom/run_development_oof.py`
- `script/twd_tom/run_final_fit.py`
- `script/twd_tom/run_memorization_sanity.py`
- `script/twd_tom/run_non_wolf_oof_diagnostic.py`
- `script/twd_tom/run_sealed_eval.py`
- `script/twd_tom/train.py`
- `tests/runtime/test_runtime_backend_wiring.py`
- `tests/speech/test_speech_perceiver_pilot_cases.py`
- `tests/twd_tom/__init__.py`
- `tests/twd_tom/conftest.py`
- `tests/twd_tom/public_event_fixtures.py`
- `tests/twd_tom/README.md`
- `tests/twd_tom/test_annotation_v2_ablation.py`
- `tests/twd_tom/test_annotation_v2.py`
- `tests/twd_tom/test_audit_canonical_belief_data.py`
- `tests/twd_tom/test_audit_dense_belief_dataset.py`
- `tests/twd_tom/test_audit_shadow_speech_parser.py`
- `tests/twd_tom/test_belief_labels.py`
- `tests/twd_tom/test_eval_subjective_tom.py`
- `tests/twd_tom/test_export_belief_worst_cases.py`
- `tests/twd_tom/test_final_fit.py`
- `tests/twd_tom/test_hard_knowledge.py`
- `tests/twd_tom/test_materialize_canonical_belief_dataset.py`
- `tests/twd_tom/test_materialize_development_folds.py`
- `tests/twd_tom/test_memorization_sanity.py`
- `tests/twd_tom/test_public_event_environment.py`
- `tests/twd_tom/test_public_events.py`
- `tests/twd_tom/test_run_development_oof.py`
- `tests/twd_tom/test_run_non_wolf_oof_diagnostic.py`
- `tests/twd_tom/test_run_random_agent_assignment.py`
- `tests/twd_tom/test_schema.py`
- `tests/twd_tom/test_sealed_eval.py`
- `tests/twd_tom/test_speech_annotations.py`
- `tests/twd_tom/test_supervision.py`
- `tests/twd_tom/test_train_lr_scheduler.py`
- `tests/twd_tom/test_train_subjective_tom.py`
- `tests/twd_tom/test_twd_tom_action_features.py`
- `tests/twd_tom/test_twd_tom_baselines.py`
- `tests/twd_tom/test_twd_tom_belief_backbone.py`
- `tests/twd_tom/test_twd_tom_dataset.py`
- `tests/twd_tom/test_twd_tom_inference.py`
- `tests/twd_tom/test_twd_tom_losses.py`
- `tests/twd_tom/test_twd_tom_metrics.py`
- `werewolf/agents/base_agent.py`
- `werewolf/models/__init__.py`
- `werewolf/models/twd_tom/__init__.py`
- `werewolf/models/twd_tom/action_features.py`
- `werewolf/models/twd_tom/annotation_v2.py`
- `werewolf/models/twd_tom/baselines.py`
- `werewolf/models/twd_tom/belief_backbone.py`
- `werewolf/models/twd_tom/belief_labels.py`
- `werewolf/models/twd_tom/checkpoint.py`
- `werewolf/models/twd_tom/dataset.py`
- `werewolf/models/twd_tom/dense_dataset.py`
- `werewolf/models/twd_tom/inference.py`
- `werewolf/models/twd_tom/losses.py`
- `werewolf/models/twd_tom/metrics.py`
- `werewolf/models/twd_tom/public_events.py`
- `werewolf/models/twd_tom/samples.py`
- `werewolf/models/twd_tom/schema.py`
- `werewolf/models/twd_tom/speech_annotations.py`
- `werewolf/models/twd_tom/supervision.py`
- `werewolf/registry.py`
- `werewolf/trajectory.py`
