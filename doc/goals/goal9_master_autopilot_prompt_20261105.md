# Goal9 主自动执行提示词：P → A/B/C → V

更新日期：2026-11-05

> **2026-11-06 V 计划最终修正：** 本文件中所有 `V0→V1→V1b/V2→V3` 内容均已过期。
> 当前主臂是 `V-0D` 纯路径输入，`V-3D` 只作可选诊断控制。权威计划见
> `doc/goals/goal9_v0d_direct_pure_path_plan_20261106.md`；实现提示词见
> `doc/goals/goal9_v0d_non_destructive_prompt_20261106.md`。发生冲突时，以新计划为准。

用途：把本文件作为长期运行 goal 的附件，粘贴下面的“启动句”，让执行 agent 从服务器
真实断点开始，自动完成 P0/P1/P2/P3、A/B/C、S15 冻结、V-0D 主臂（可选 V-3D 诊断），直到形成
可审计的正结果、负结果和最终报告。固定顺序是 **P → ABC → V**；不要改成 P → V →
ABC，也不要在 S15 冻结前启动 V 正式训练。

> 本文件只负责编排。训练预算、环境数、步数、seed、评估 tag、门禁阈值、消融要求和
> 产物目录，一律以 `goal9_odarrl_control_reproduction.md` 的 14.4.18/14.4.19 和
> `goal9_pc_server_validation_plan.md` 的 S15/S16 为准。若本文件与协议冲突，采用更
> 保守、不会放宽科学门禁的解释；只有冲突会改变科学结论且无法保守解释时才暂停询问。

## 启动句（复制这一段给 Goal）

```text
你是 Goal9 的长期实验执行 agent，不是单轮问答助手，也不是只写计划的文档助手。
立即完整读取本文件、goal9_odarrl_control_reproduction.md（重点 14.4.18/14.4.19）、
goal9_pc_server_validation_plan.md（重点 S15/S16 和“6. 当前执行顺序”）、
goal9_auto_run_prompt_20261105.md 和 goal9_execution_state.md。先从服务器 artifacts、
manifest、summary、日志、PID 和 queue.lock 审计真实断点，再实际执行最靠前的未完成
合法动作。不要只回复计划，不要停在“代码已实现/任务已排队/进程正在运行”。

固定顺序是 P → ABC → V，直到完成 S15 冻结、V-0D 主臂（R-BC-0D → R-Z0-0D →
R-PPO-0D）、3×50（如门禁允许）和 CPU PVS 64-seed（如仅有一个通过候选），最后写完整正/负结果报告。已完成且
可审计的 P0/P1/P2/P3、A screen、B10 smoke/screen 不得重跑；B5 已正常结束但门禁 fail，
不得重跑或加预算救结果。A2 只有相关性，不是因果胜出臂。B10 是效率非劣候选，不能写成
性能优于 A0。C 只有协议规定的“时间尺度有因果影响且 DADE+MLP 仍有明确瓶颈”证据成立时
才启动，否则写 C not_triggered。随后按协议冻结唯一 S15 基线并写 decision，再进入 V。

当前 V 代码可能仍是 WIP：先审计，不要假设能跑。若 ALOS-free 环境、固定 u_ref/删除 u_d、
路径奖励、非零初速/初 RPM、command-dim/history-path schema、runner、summarizer、
latency benchmark 或测试缺失，先把实现补齐并做本地 pytest/ruff/1-iteration CPU smoke；
修到契约通过后同步服务器，再做 V0 smoke。V0 fail 就停止 V；V1 fail 就保留 ALOS 外置；
V1b/V2 默认不跑；V3 只对唯一通过候选做正式复核。

严格使用协议固定的 seed、预算、门禁、环境数、步数和现有统一入口；严禁换 seed、改阈值、
追加训练步数、调 reward 权重、增加第二个科学变量或覆盖历史 artifacts 来救失败结果。
长任务必须用 nohup/tmux 持久化，记录 host/PID/GPU/CPU/log/lock/output/resume command，
并持续写回 goal9_execution_state.md。除协议列出的硬阻塞外不要询问用户：普通 smoke 失败、
基础设施重试、科学负结果、C/V1b/V2 不触发、SSH 中断和缺失代码都必须自动处理。
只要还有合法动作就继续执行；全部可执行分支结束后才输出最终报告。
```

## 唯一权威来源

开始前必须完整阅读，并按下面的优先级使用：

| 内容 | 权威来源 |
|---|---|
| P0/P1/P2/P3 和 A/B/C 的科学配方、预算、门禁 | `doc/goals/goal9_odarrl_control_reproduction.md` 的 14.4.18 |
| V0/V1/V1b/V2/V3 的科学配方、预算、门禁 | `doc/goals/goal9_odarrl_control_reproduction.md` 的 14.4.19 |
| 服务器执行顺序、S15/S16 阶段边界 | `doc/goals/goal9_pc_server_validation_plan.md` 的 S15、S16 和“6. 当前执行顺序” |
| 逐阶段操作循环、产物和恢复规则 | `doc/goals/goal9_auto_run_prompt_20261105.md` |
| 跨轮次实时状态 | `doc/goals/goal9_execution_state.md` |
| P1 路径观测负结果 | `doc/goals/goal9_p1_path_observation_report.md` |
| A/B/C 已完成阶段的 summary 和 manifest | `artifacts/lowlevel/pc_abc_temporal_20261105/` |
| P0–P3 已完成阶段的 summary 和 manifest | `artifacts/lowlevel/pc_abc_prework_20261105/` |
| V 阶段产物 | `artifacts/lowlevel/pc_alos_free_v_20261105/` |

如果文档、旧提示词和实际 artifacts 不一致，以 artifacts、manifest、summary、日志和
服务器进程为准；文档只用于解释门禁和减少重复劳动。

## 当前断点快照

以下只是 2026-11-05 的交接快照，用于避免重复劳动。每次接手都必须重新核对服务器。

- **P0 已完成。** 正式产物在
  `artifacts/lowlevel/pc_abc_prework_20261105/P0_a_screen/`；基线为 `A0_stride1`。
  不要重跑，不要扩大预算。
- **P1 已完成负结果。** `stage_decision=no_path_attribution`，`path-zero` 没有可解释
  退化；路径分支停止。不要重跑 P1，不要继续加路径特征、改路径 scale 或换评估 seed。
- **P1_curriculum 未运行是门禁主动跳过**，不是 curriculum 训练失败。不要把它写成 fail，
  也不要为了补跑而启动。
- **P2 已完成，结论 `no_signal`。** 保留 raw-history `A0_stride1`，停止 encoded-history
  DADE 重构；不要加层、加 head 或换编码器挑结果。
- **P3 已正式记录 `not_applicable`。** 原因是 P1 `no_path_attribution` 且 P2b 使用
  `path_dim=0`。不要为了跑满阶段而添加路径奖励或扫描 `w_path_pc`。
- **A screen 已完成。** `selected_arm=A0_stride1`、`selection_status=baseline_no_signal`。
  `A2_stride10` 在 default 上 success `+0.05`、3D RMSE 改善 `0.4271 m`，但
  `history-zero`/`history-reverse` 都没有可解释退化，因此只能登记相关性，不能扩 A 的
  3×32，也不能冻结为 S15 因果胜出臂。
- **B10 smoke 已完成。** 服务器产物在
  `artifacts/lowlevel/pc_abc_temporal_20261105/smoke/B10/`；`run_manifest.json` 为
  `status=completed`、`training_audit_passed=true`，`report/summary.json` 为
  `pass_non_inferior`，独立 latency 测试 p95 约 `0.879 ms`。这只是 smoke，不是正式
  科学结论。接手时先核对 manifest；如果最新 runner 的 latency 审计还没有落盘，只补跑
  latency 审计，不重训。
- **B10 正式 screen 已完成并通过非劣/效率门禁。** 产物在
  `artifacts/lowlevel/pc_abc_temporal_20261105/B10/`；manifest 为 `status=completed`、
  `training_audit_passed=true`，summary 为 `pass_non_inferior`。default success/strict
  降幅为 `0.0`、RMSE ratio `1.0014`、timeout 增幅 `0.0`、saturation 增幅 `0.0009`；
  latency gate 通过，p95 `0.8907 ms`、compute ratio `0.1186`。这是 B 组效率候选，
  不能写成相对 A0 的性能提升。
- **B5 正式 screen 已正常结束，但科学门禁 fail。** 产物在
  `artifacts/lowlevel/pc_abc_temporal_20261105/B5/`；manifest 为 `status=completed`、
  `training_audit_passed=true`，summary 为 `fail_non_inferior`。default success/strict
  均下降 `0.05`、RMSE ratio `1.1153`、timeout 增加 `0.05`、saturation 变化
  `-0.0037`；strict125 也下降 `0.05`。按协议停止 5 Hz 降频分支，不重跑、不增加训练步数。
- **当前没有活跃的 A/B/C/V runner。** 下一步不是再训练 B，而是写 B 阶段判定、完成 C
  触发审计、冻结唯一 S15 基线并启动 V 实现；V 正式训练仍不得早于 S15 冻结。
- **V 组代码、配置、runner、summarizer 和测试仍不完整。** 仓库里可能存在 command-dim/
  history-path 的 WIP 改动，但在完整单测、schema 审计和 CPU smoke 通过前不得视为可用。
  V0/V1 正式训练必须在 S15 冻结之后才能启动。

## 固定执行顺序

```text
P0（复用）
  → P1（已完成负结果，停止路径分支）
  → P2（已完成 no_signal，保留 raw-history）
  → P3（已正式记录 not_applicable）
  → A（复用 screen；不扩 3×32）
  → B10（已完成，pass_non_inferior + latency gate 通过）
  → B5（已完成，fail_non_inferior；停止 5 Hz 分支）
  → C（仅在 A/B 证据成立时启动；当前默认 not_triggered）
  → 冻结唯一 S15 基线
  → V-0D 主臂：R-BC-0D（state + 8D path -> action，command_dim=0）
  → R-BC-3D 与 R-BC-0D 闭环对照
  → R-Z0-0D 零残差等价
  → R-PPO-0D screen（1×20）
  → 可选 V-3D 诊断控制（仅 0D 失败或需要归因时启动）
  → V-0D 正式比较（3×50 + CPU PVS 64-seed）
  → 最终报告
```

约束：

- P 是 S15 前置门禁；A/B/C 是 S15 主体；V 是 S16。
- S16 只能在 S15 报告完成并冻结唯一基线后启动。V 的代码、配置和 BC 审计可以提前准备，
  但 V0/V1 正式训练不得提前。
- 一次只改变一个科学变量。A 只改历史 stride；B 只改 actor 决策频率；C 只改 actor
  信息通路；V0 只移除 ALOS heading/pitch 制导；V1 只在 V0 通过后删除 `u_d`。
- 不得为了利用空闲 GPU 跳过依赖门禁，不得把不同阶段混在同一轮比较中。

## Goal 级自动状态机（每轮照此推进）

每次接手先建立一份只读 `audit.json` 或等价状态快照：服务器 host、GPU/磁盘、
`queue.lock`、活跃 PID、已完成 manifest/summary、缺失实现和当前唯一允许动作。随后按
下面的优先级循环，直到没有合法动作：

```text
1. 若任一预注册 run 仍活跃：轮询/接管，不重启，不启动依赖它的下一阶段。
2. 若 P0–P3/A 的证据缺失或不可审计：只做只读校验；必要时按同一命令修复基础设施，
   不扩大预算、不换 seed、不改门禁。
3. 若 B 阶段尚未写 decision/report：从现有 manifest/summary 生成 B 阶段判定；
   B10 保持 efficiency candidate，B5 记为 fail_non_inferior。
4. 若 C 尚未写触发决策：逐条检查协议谓词；不满足就写 C_not_triggered；满足才实现并
   跑一次预注册的有界 C，禁止先跑再看结果补理由。
5. 若 S15 尚未冻结：按本文和协议冻结唯一基线，写配置/checkpoint/schema/seed/tag 和
   selection reason；没有唯一基线前禁止启动 V 正式训练。
6. 若 V 实现闸门未通过：实现缺失能力、修复已知风险、补测试、跑本地 CPU smoke。
   这是当前最先可做的工作之一，可与服务器只读盘点并行，但不得提前正式训练。
7. 若 V0 未完成：服务器 preflight → V0 smoke → V0 1×20 → 门禁。V0 fail 即停止 V。
8. 若 V0 通过而 V1 未完成：V1 smoke → V1 1×20 → 门禁。V1 fail 即保留 ALOS 外置。
9. 若 V1 通过：按证据决定 V1b/V2 是否触发；不触发就写 skipped 及理由。
10. 若存在唯一通过候选而 V3 未完成：3×32 → CPU PVS 64-seed → 最终报告。
11. 全部可执行分支结束：更新协议、阶段报告、最终 evidence 和 state；停止询问。
```

每次状态转移都必须写回 `goal9_execution_state.md`，至少包含 `current_stage`、
`completed_nodes`、`running_jobs`、`failed_or_skipped`、`next_action` 和可复制的
`resume_command`。不能只在聊天里说“正在跑”，也不能用“代码已实现”代替 smoke 和
manifest。

## 逐阶段执行规则

### P0/P1/P2/P3：复用，不重跑

- P0、P1、P2 的正式产物直接复用；P3 已记录 `not_applicable`。
- 只有 summary、training audit、逐 seed 记录缺失，或有新的契约失败证据时，才允许按
  同一预注册 smoke/screen 重跑；不得扩大预算、换 seed 或改阈值。
- P1 路径分支和 P2 DADE-z 重构已经停止。后续 A/B/C/V 使用 raw-history `A0_stride1`
  作为无信号时的降级基线。

### A：复用 screen，不扩 3×32

- A screen 已完成，`selected_arm=A0_stride1`。
- A2 的 default 指标只能作为相关性；没有 `history-zero`/`history-reverse` 可解释退化，
  不得写成时间感受野的因果收益，不得据此启动 A 的 3×32。
- 不要重跑 A0/A1/A2，不要改历史 stride、路径 scale 或评估 seed。

### B10/B5：审计并关闭，不重跑

- B10 smoke 和正式 screen 均已完成。正式 screen 为 `pass_non_inferior`，latency gate
  通过；它是 S15 的效率候选，不是相对 A0 的性能胜出结论。
- B5 smoke 与正式 screen 均已完成。B5 正式结果为 `fail_non_inferior`：default/strict125
  success 都下降 `0.05`，RMSE ratio `1.1153`，timeout 增加 `0.05`。按协议停止 5 Hz
  降频分支。
- 不重跑 B10/B5，不补训练 seed，不增加训练步数，不换评估 seed，不修改门禁。只补写
  stage decision、B 阶段报告和 execution state；若发现 manifest/summary 缺字段，做只读
  校验或同口径重算，不重训。
- 报告必须把 B10 的效率收益与 B5 的负结果分开，不能把 B5 的失败解释成 B10 无效，也不能
  把 B10 的非劣写成性能提升。

### C：条件启动，默认不强行跑

- 只有 A/B 证明“时间尺度/历史采样确实有影响，且 `DADE + MLP` 仍有明确通路瓶颈”时
  才启动 C。
- 当前 A screen 的历史消融没有可解释因果信号；B10 只是效率非劣，B5 又失败。除非已有
  可审计 artifact 明确满足上述瓶颈条件，否则写 `C_not_triggered` 及逐条判断依据，
  不启动 C 训练。不要因为“有空闲 GPU”或“想跑满组”而启动 C。
- C 输入必须包含当前状态 token、命令 token 和历史 token，不能只喂历史而丢掉当前指令。
- 先对 BC/最佳基线动作做监督蒸馏，再做一次有界微调；参数规模和 p95 时延与 DADE+MLP
  同量级。
- 无稳定收益立即判 fail，不继续加层、加 head 或换 seed。
- 如果 A/B 证据不成立，正式记录 `C not triggered` 及依据，然后进入 S15 冻结；不要为了
  跑满阶段强行启动 C。

### S15 冻结

- 逐 seed 汇总后，按协议冻结唯一基线：最佳 A/B/C；C 失败则最佳 A/B 的 DADE+MLP；
  A/B 都无可解释信号则回退 `A0_stride1`。
- 采用最保守的现有证据解释：A0 是 `selected_arm`，A1/A2 无历史归因，B10 只是效率
  非劣，B5 fail。默认冻结 `A0_stride1` 为唯一 S15 科学基线，并把 B10 记录为
  `efficiency_candidate`，不得把 B10 与 A0 混成两个同时使用的 V 基线。只有在现有
  协议工具或已冻结的 selection artifact 明确把 B10 选为“最佳 A/B”时，才允许把 B10
  作为唯一 S16 基线；此时仍须写 `selection_reason=efficiency_non_inferior`，并明确
  它不代表性能提升。
- 记录配置、checkpoint、schema、归一化、训练 seed、评估 seed、tags 和评估口径。
- 冻结后不再根据 V 的结果反向修改 S15。
- 写阶段 decision/report，并把 `goal9_execution_state.md` 更新到“S15 frozen”。

### V：S16（历史，已被 V-0D 主臂取代）

> 以下 V0/V1/V1b/V2/V3 详细内容只保留为历史。当前执行必须使用
> `doc/goals/goal9_v0d_direct_pure_path_plan_20261106.md` 和
> `doc/goals/goal9_v0d_non_destructive_prompt_20261106.md`，不要再按下面旧的
> ALOS-free V0/V1 顺序启动训练。

#### V 实现闸门：先审计 WIP，再谈训练

当前仓库可能已有 `actor_command_dim`、`critic_command_dim`、`history_path_dim` 等 WIP
改动，但不得因为“代码已存在”就直接开训。先完成以下契约，否则任何 V smoke 都无效：

- 修正 `PCNormalization.normalize_command`，只校验 0/1/3 维命令的形状，不得把已经
  归一化的 `PCCommand.normalized()` 再除一次 command scale。
- `path-zero` 必须同时作用于当前 path 和 encoded-history 里的 path；不能只清当前帧。
- `pc_schema_hash` 对非法命令维度必须抛明确错误；actor/critic command dim、
  history path dim、path block 和 `active_mask` 必须进入 metadata 与 hash。
- 训练、评估、latency benchmark、checkpoint 校验和 summarizer 必须完整传递新布局；
  默认 3 维命令 + raw history + `path_dim=0` 的旧 schema 必须保持兼容。
- 实现 ALOS-free 环境模式：V0 actor 只看 `z_t=E(o_t)` 和固定 `u_ref`，不看
  `psi_d/theta_d`；V1 只删除 `u_d`，actor 始终输出 `[rudder, stern, rpm]`。
- 路径特征只能由 `sensor_output.p_nav` 投影并以 block/`active_mask` 进入 `o_t`，再编码
  为 `z_t`；DADE 历史使用同一套含路径的 `z_t`，不得存在 raw `path_t` 旁路。
- 实现路径误差 + 进度/时间为主、速度平滑优先于动作平滑、能耗次级、安全/包线/TIMEOUT
  硬约束的奖励；material saturation 只报告，不作为门禁；不加过冲惩罚，不硬编码
  曲率→速度，不要求终端减速。
- reset/eval 必须实际出现非零初始纵荡速度、非零初始 RPM、路径/横向/垂向/航向/俯仰
  偏差、低速/中速、曲率段和恢复状态；初始状态是环境变量，不是 actor 特权观测。
- 至少补测试：ALOS `psi_d/theta_d/u_d` 不进 actor；path 只经 encoder 进 `z_t`；
  path-zero 清当前和历史；V0/V1 schema 只差 `u_d`；非零初速/初 RPM 真被采样；
  actor 输出始终三维执行器动作；default schema 兼容；非法 command dim 明确失败。
- 本地必须通过相关 pytest、ruff、schema/import smoke 和 1-iteration CPU smoke。之后同步
  服务器并跑服务器 smoke；本地通过不等于正式实验通过。

#### V0：ALOS-free + 固定速度参考

- actor 输入是 `z_t = E(o_t)`，`o_t` 由机器人状态和路径特征组成；固定 `u_ref` 作为
  显式输入；输出 `[rudder, stern, rpm]`。
- 路径投影只能来自 `sensor_output.p_nav`；不使用 ALOS 生成的 `psi_d/theta_d`。
- BC 可来自 ALOS teacher 或中低速轨迹，但学生训练与评估观测必须一致。
- 训练和评估必须覆盖非零初始纵荡速度、非零起步 RPM、路径/横向/垂向/航向/俯仰偏差
  起步、低速和中速、曲率段及恢复状态。初始状态属于环境，不作为 actor 特权观测。
- 奖励以路径误差、进度和任务时间为主；速度平滑优先于动作平滑，能耗尽量优化；安全、
  timeout 和速度包线是硬约束。material saturation 只报告执行器使用风险，不作为
  通过/否决门禁。不加过冲惩罚，不要求终端减速。
- 先 smoke（1 训练 seed × 8 评估 seed），再 1×20 screen；相对最佳 ALOS-external S15
  基线，success 或 strict125 至少 `+0.05`，或 3D RMSE 至少改善 `0.10 m`；
  time-to-goal 不超过 `1.05×`；safety 增幅不超过 `0.02`，timeout 增幅不超过
  `0.05`；material saturation 只报告，不作为硬门禁。
- V0 未通过则停止 V 分支，不启动 V1。

#### V1：删除显式速度参考

- 在 V0 通过后启动；`o_t` 不含 `u_d`，actor 仍输出 `[rudder, stern, rpm]`。
- 路径切向、曲率、前视和剩余距离进入 `o_t` 并编码进 `z_t`，但不在奖励中硬编码
  曲率-速度映射；由策略自己学习转弯减速。
- BC 可用 ALOS 轨迹或中低速轨迹；教师速度行为通过 RPM 标签隐式蒸馏，后续再做有界
  PPO 微调。
- 速度包线固定 `u_min/u_max`、RPM 上限；material saturation 继续测量并报告。
- 与 V0 同 seed 配对；V1 通过至少要求 success/strict 降幅不超过 `0.02`，3D RMSE
  `<=1.05×`，timeout/safety 增幅不超过 `0.02`；同时记录速度标准差、`Δu` p95、
  RPM²/能耗和转弯段速度-曲率关系。material saturation 只报告，不作为硬门禁。
- V1 primary 指标失败则记录负结果并保留 ALOS 外置方案。

#### V1b：可选曲率软约束

- 只有 V1 在转弯段明显不减速、且不是路径精度/时间目标导致的合理高速时才启动。
- 只允许加一个很小的曲率软约束：
  `r_turn = -w_turn * relu(u - u_turn_max(kappa))`。
- `w_turn` 必须小，不得压过路径精度和时间目标；与 V1 同 seed 配对，检查转弯段速度、
  success/strict、RMSE、time-to-goal 和 saturation。
- 如果只是降低速度但 primary 指标退化，判 fail，停止该分支。

#### V2：可选 speed head，默认不启动

- 只有 V1 出现严重速度振荡、速度不可解释、速度包线频繁越界或路径/速度控制互相干扰，
  且复核确认不是奖励或观测问题时，才允许测试。
- `actor -> [rudder, stern, u_cmd] -> speed controller -> rpm`。
- 启动前必须写清触发证据和为何不是奖励/观测缺陷；V2 是独立 arm，不能与 V1/V1b
  同时改动。默认不启动。

#### V3：正式比较

- 只对通过的唯一候选跑 3 个训练 seed × 32 个评估 seed，再做 CPU PVS 64-seed。
- 至少比较 `ALOS + PD/PI` teacher、V0、V1；V1b/V2 只有触发证据成立时才加入。
- 主指标：success、strict125、横向/垂向 RMSE、time-to-goal、timeout 和 safety；
  信息性风险指标：material saturation；次级指标：速度标准差、`Δu` p95、RPM²/能耗、
  动作变化率、转弯段速度-曲率关系和过冲诊断。
- 不根据评估 seed 反向挑 checkpoint；不为了让结果好看而换 seed、改阈值或增加预算。

## 每个阶段的自动循环

每个阶段都按下面循环执行，完成后自动进入下一个允许阶段，不等待人工逐步确认：

1. **盘点。** 检查代码、配置、runner、summarizer、测试、artifacts、服务器目录、PID、
   锁、日志和 GPU。已完成且可审计的结果直接复用；同一阶段已有任务时先等待/接管。
2. **补齐实现。** 缺少配置、入口、汇总器、latency benchmark、消融入口或单测时先实现。
   正式训练不得用临时 Python 命令绕过统一入口。
3. **本地验证。** 运行相关 PC 单测、ruff、schema/import smoke 和必要的 1-iteration
   CPU smoke；修到契约通过为止。
4. **服务器 preflight。** 同步代码和配置；确认 Python/venv、GPU、磁盘、目录和已有
   队列。服务器目录不是 Git 仓库，Git 操作只在本机。
5. **契约 smoke。** 按协议运行 1 个训练 seed × 8 个评估 seed；检查物理时长、动作
   频率/保持、历史 stride、reward 时间尺度、无特权泄漏、finite audit、schema/hash。
6. **低预算筛查。** 按协议运行 1×20（固定评估 seed `1000–1019`）或等价阶梯；逐 seed
   配对，不换 seed 挑结果。
7. **门禁判定。** 从协议读取阈值，写出 `summary.json`、`summary.md`、manifest 和判定
   依据。pass 才进入下一阶段；fail 写负结果并按分支规则继续或停止该分支。
8. **正式比较。** 只有协议允许且筛查有可解释信号时才扩到 3 个训练 seed × 32 个评估
   seed；不因结果不好而增加训练步数、预算或第二个变量来救结果。
9. **最终复核。** 只对唯一通过候选运行 CPU PVS 64-seed；不根据评估 seed 反向挑
   checkpoint。
10. **记录与交接。** 更新协议文档、阶段报告、产物路径、偏离说明和
    `goal9_execution_state.md`；输出阶段摘要后继续，不等待用户说“继续”。

## 长任务、轮询与断点恢复

### 启动与记录

- 服务器参考：
  `ssh -p 43127 yhx@152.136.183.51`
  仓库：`/data/yhx/underwater-robotics/src/plan_p1/settled_局部规划器_反应式`
  Python：`/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python`
- 启动前必须检查：
  `nvidia-smi`、`ps -ef | grep -E "run_pc_abc|run_pc_alos_free|train_batch_lowlevel_ppo|evaluate_pc_ppo" | grep -v grep`、
  `df -h /data`、输出目录的 `queue.lock` 和 `run_manifest.json`。
- 长任务用 `nohup` 或 `tmux` 托管，命令、PID、日志、锁、GPU、CPU 绑核、输出目录和
  开始时间写入 `goal9_execution_state.md`。
- 同一个输出目录已有活跃 runner 或 `queue.lock` 时，只等待/接管，禁止启动第二个。

### 轮询

- 训练/评估运行期间，每 60–120 秒检查一次即可；不要高频空转。
- 等待时只做不依赖该结果且不冲突的工作：补 V 的配置、测试、BC 数据审计、报告模板、
  只读代码审计。当前可优先做 V 的实现闸门。
- 不得一边等待某个正式 arm 一边启动依赖它的下一阶段；不得在 S15 冻结前启动 V 正式
  训练。

### 失败分类

| 类型 | 处理 |
|---|---|
| 普通 smoke 失败 | 修实现，重跑同一 smoke，不扩大 seed 或预算 |
| 科学负结果 | 写负结果，按协议停止该分支或使用规定降级基线 |
| GPU 显存不足/排队 | 换独立 GPU、等待或按协议重试，保持配置不变 |
| SSH 短暂中断 | 重连后检查 PID 和日志；进程仍活着就继续等待，不重启 |
| runner 退出但无 final manifest | 先判定基础设施失败；同一命令最多重试 2 次 |
| 锁存在但 PID 已死 | 确认无活跃进程且产物不完整后，将旧锁重命名保存，再按同一配置重试 |
| 配置/runner/测试缺失 | 先补齐并验证，再执行；不要用临时脚本绕过统一入口 |

基础设施失败最多重试两次；科学负结果不重跑，不换 seed、不改阈值、不加训练步数、
不调 reward 权重。

可以使用 subagent 并行做互不重叠的实现、只读审计、测试和报告草稿；主 agent 必须统一
检查产物、执行门禁、决定状态转移并写最终报告。subagent 不得自行修改 seed、阈值、预算、
实验顺序或启动正式 run。

## 硬门禁速查

> 下面是执行时的速查表；若与协议正文冲突，以协议正文为准。

| 阶段 | 通过条件 | 失败处理 |
|---|---|---|
| P1 | `path-zero` 出现可解释退化 | 已失败：停止路径分支 |
| P2 | success/strict `+0.05` 或 RMSE `-0.10 m`；`history-zero` 可解释退化 | 已失败：保留 raw-history |
| A | 相对 A0 达阈值且历史消融可解释 | 已判无因果信号：复用 A0，不扩 3×32 |
| B10/B5 | success/strict 降幅 ≤0.02；RMSE ≤1.05×；timeout ≤0.02；p95 低于决策周期；效率收益可复现；saturation 仅报告 | B10 已通过；B5 已 fail：停止 5 Hz 分支 |
| C | 达到 A 组同等级性能阈值，且不靠更多参数/训练/seed | 默认 not_triggered；触发后 fail 就停止堆层 |
| V0 | success/strict `+0.05` 或 RMSE `-0.10 m`；time-to-goal ≤1.05×；safety/timeout 不退化；saturation 仅报告 | 停止 V，不启动 V1 |
| V1 | success/strict 降幅 ≤0.02；RMSE ≤1.05×；time-to-goal ≤1.05×；timeout/safety ≤0.02；saturation 仅报告 | 保留 ALOS 外置，写负结果 |
| V1b | 只修转弯减速，primary 指标不退化 | 停止该分支 |
| V2 | 仅在 V1 严重速度异常且有书面证据时启动 | 默认跳过 |
| V3 | 唯一候选完成 3×32 和 CPU PVS 64-seed，且稳定优于或非劣于 ALOS teacher | 保留 ALOS 外置 |

## 硬约束

- 一次只改变一个科学变量；同组 arm 必须同 BC/教师数据、同训练物理秒数、同更新次数、
  同评估 seed 集和同硬件条件。
- actor 只能使用可部署观测；真实海流、故障标签、仿真真值和 critic 特权量不得进入
  部署路径。
- 路径特征必须作为 `o_t` 的一部分进入 encoder 得到 `z_t`；禁止 raw `path_t` 绕过
  encoder 直接进入 actor 顶层或 DADE。
- 当前协议使用时间步 token；不要擅自引入 block 级 token、RoPE 或 speed head。
- BC 标签是执行器动作 `[rudder, stern, rpm]`，不是中间制导量。
- 任何 Torch 结果都只是预筛；正式结论必须经过协议规定的 CPU PVS 复核。
- 不换评估 seed、不改阈值、不加训练步数、不加预算、不调 reward 权重来抢救 fail 的
  arm；所有偏离必须写入报告。
- 不覆盖历史 artifacts、checkpoint、日志或已有 run 目录；续跑只在 runner 明确支持且
  产物不完整时进行。
- 不使用 `git add -A`；不自动 commit/push；不提交 artifacts、checkpoint、日志或凭据。
- 每个 run 记录 GPU 型号/显存、torch/CUDA、配置 hash、checkpoint hash、训练 seed、
  评估 seed、评估 tag 和 wall-clock。

## 产物与文档更新

每个阶段至少保留：

- `summary.json`、`summary.md`；
- `training_manifest.json` 或等价训练 manifest；
- 配置 hash、checkpoint hash、schema hash；
- 训练 seed、评估 seed、评估 tag 和逐 seed 配对记录；
- finite audit、无特权泄漏审计和必要的消融记录；
- p50/p95 latency、运行环境、资源绑定和 wall-clock；
- `run_manifest.json`、`stage_decision.json` 和失败/重试记录。

每完成一个阶段，更新：

- `doc/goals/goal9_odarrl_control_reproduction.md`
- `doc/goals/goal9_pc_server_validation_plan.md`
- 对应阶段报告；A/B/C 和 V 完成后分别新增报告。
- `doc/goals/goal9_execution_state.md`

`goal9_execution_state.md` 至少维护以下字段：

```text
updated_at:
current_stage:
completed_nodes:
running_jobs:
  - host:
    pid:
    stage:
    command:
    log:
    lock:
    output:
    gpu:
failed_or_skipped:
next_action:
resume_commands:
```

它不能替代 `summary.json` 或阶段报告；只是跨轮次状态账本。

## 完成标准与最终交付

只有以下条件同时成立，才可宣告 Goal9 可执行部分完成：

- P、A/B/C 已按门禁结束并写入报告；
- S15 已冻结唯一基线；
- V0/V1 及允许启动的 V1b/V2 已按门禁结束；
- 唯一通过的 V 候选已完成 3×32 和 CPU PVS 64-seed，或已明确记录 V 分支失败；
- 所有未执行、跳过、负结果和基础设施问题均已写入最终报告。

最终必须给出：

- 实际完成和未完成的阶段；
- 每个阶段的正结果、负结果、判定依据和协议门禁；
- S15 冻结的唯一基线配置、checkpoint、schema 和评估口径；
- S16 候选以及 V0/V1/V1b/V2/V3 的执行状态；
- 是否真正完成 3×32 和 CPU PVS 64-seed；
- 所有产物目录和报告路径；
- 所有偏离、重试、基础设施问题和未执行项；
- 哪些结论可以进入论文，哪些只能作为负结果、相关性或待验证项；
- 最新 `CURRENT_STATE`：已完成、运行中 PID/队列、下一阶段和阻塞原因。

## 不要做的事

- 不要把“已写计划”“代码已实现”“本地 smoke 通过”“任务已排队”或“进程正在运行”
  写成实验完成。
- 不要为了跑满阶段而强行启动 C、V1b 或 V2。
- 不要因为 A2 的 default 指标好看就改写为时间感受野因果收益。
- 不要因为 B10/B5 失败就增加训练步数、换评估 seed 或改门禁。
- 不要在没有 S15 冻结的情况下启动 V0/V1 正式训练。
- 不要在没有 final manifest 和 summary 的情况下进入下一阶段。
- 不要在同一个输出目录已有活跃 runner 或锁时重复启动。
- 不要只给计划或只汇报状态；当前阶段可执行就立即执行，直到真正阻塞。
