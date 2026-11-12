# Goal9 全自动执行 Goal 提示词（P → A/B/C → V）

更新时间：2026-11-05

> **2026-11-06 V 计划最终修正：** 本文件中所有 `V0→V1→V1b/V2→V3` 内容均已过期。
> 当前主臂是 `V-0D` 纯路径输入，`V-3D` 只作可选诊断控制。权威计划见
> `doc/goals/goal9_v0d_direct_pure_path_plan_20261106.md`；实现提示词见
> `doc/goals/goal9_v0d_non_destructive_prompt_20261106.md`。发生冲突时，以新计划为准。

> 当前首选入口是 `doc/goals/goal9_run_all_pvabc_prompt_20261105.md`；本文件保留为
> 启动摘要和协议导航。若两份文件的当前断点不一致，以服务器 artifacts、日志、PID 和
> `goal9_execution_state.md` 的实际证据为准。

## 启动句（复制给 Goal）

你是长期运行的实验执行 agent。立即完整读取本文件、`goal9_odarrl_control_reproduction.md`、
`goal9_pc_server_validation_plan.md`、`goal9_auto_run_prompt_20261105.md` 和
`goal9_execution_state.md`，然后从服务器 artifacts、日志、PID 和 queue.lock 的真实断点
继续执行 Goal9。不要只回复计划，不要停在“已实现/已排队/正在运行”；自动完成 P 的审计、
A/B/C 的可执行分支、S15 冻结和 V-0D 主臂（可选 V-3D 诊断），直到形成可审计的最终报告。P0–P3 和
A screen 已完成的部分不得重跑；A2 只有相关性、没有历史归因，不能当作因果胜出臂。
B10 已完成并记为效率非劣候选，B5 已完成并记为负结果，二者都不重跑。当前下一步是写
B 阶段 decision、完成 C 触发判定、冻结唯一 S15 基线，然后补齐并执行 S16 的 V-0D 主臂。
严格按协议使用固定 seed、预算、门禁和现有 runner，不换 seed、不改阈值、不增加训练步数，
不重复启动同一 run。长任务用 nohup/tmux 持久化，并把 PID、锁、日志、断点和下一动作持续
写回 `goal9_execution_state.md`。最终必须给出正结果、负结果、未触发分支、产物路径和
是否真正完成 3×32 与 CPU PVS 64-seed。

你是一个长期运行的实验执行 agent，不是单轮问答助手，也不是只写计划的文档助手。你的任务是
从当前真实断点开始，实际执行并审计 Goal9 的 S15（P0–P3、A/B/C）和 S16（V-0D 主臂），直到所有
可执行分支结束，并形成可审计的正结果、负结果和最终报告。只有所有可执行实验、消融、正式复核
和报告都结束，才算完成；不要把“已写计划”“代码已实现”“本地 smoke 通过”“任务已排队”或
“进程正在运行”写成实验完成。

## 工作区与唯一协议

工作区：

`/home/bubble/水下机器人/settled_局部规划器_反应式`

开始前必须完整阅读，并以这些文件为唯一科学依据：

- `doc/goals/goal9_odarrl_control_reproduction.md`
- `doc/goals/goal9_pc_server_validation_plan.md`
- `doc/goals/goal9_auto_run_prompt_20261105.md`
- `doc/goals/goal9_execution_state.md`
- 已完成的阶段报告，尤其是 `doc/goals/goal9_p1_path_observation_report.md`

其中主文档的 `14.4.18`、`14.4.19`，服务器文档的 `S15`、`S16` 和 `6. 当前执行顺序`
优先。训练预算、环境数、步数、seed、评估 tag、门禁阈值、消融要求、产物目录和停机规则，
全部以协议文档为准。本提示词只负责编排。若提示词、旧快照和协议冲突，采用更保守、不会放宽
科学门禁的解释；只有冲突无法保守解释且会改变科学结论时才暂停询问。

服务器参考信息：

- SSH：`ssh -p 43127 yhx@152.136.183.51`
- 仓库：`/data/yhx/underwater-robotics/src/plan_p1/settled_局部规划器_反应式`
- Python：`/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python`

## 当前断点（只用于减少重复劳动）

每次接手都必须以代码、配置、artifacts、summary、日志、锁和服务器进程的实际状态为准。
本快照过期时，不要机械照搬。

- P0 已完成并复用，基线为 `A0_stride1`；不要重跑。
- P1 已完成正式 screen，`stage_decision=no_path_attribution`，路径分支停止；不要重跑，
  不要继续增加路径特征、改路径 scale 或换评估 seed 救结果。
- P1_curriculum screen 未运行是门禁主动跳过，不是 curriculum 训练失败。
- P2 已完成正式 screen，结论 `no_signal`；保留 raw-history `A0_stride1`，停止 DADE-z
  重构，不重跑、不加层、不加 head、不换编码器挑结果。
- P3 已正式记录为 `not_applicable`，原因是 P1 `no_path_attribution` 且 P2b 使用
  `path_dim=0`。不要为了跑满阶段而添加路径奖励或扫描 `w_path_pc`。
- 正式 A 组 smoke 和 A screen 均已完成。A screen 于 2026-11-05 18:30:55 CST 正常结束，
  `status=completed`、`training_audit_passed=true`，当前没有 A/B/C/V 训练或评估进程。
  证据位于 `artifacts/lowlevel/pc_abc_temporal_20261105/A_screen/report/summary.{json,md}`。
  `selected_arm=A0_stride1`、`selection_status=baseline_no_signal`；`A1_stride5` 为
  `no_signal`；`A2_stride10` 在 default 上 success `+0.05`、3D RMSE 改善 `0.4271 m`，
  但 `history-zero`/`history-reverse` 都没有可解释退化。因此 A2 只能登记为相关性，不能
  写成时间感受野的因果收益，也不能据此再扩 A 的 3×32 或冻结为 S15 因果胜出臂。
- B10/B5 均已完成。B10 正式 screen 为 `pass_non_inferior`，只能登记为
  `efficiency_candidate`，不能写成相对 A0 的性能提升；B5 正式 screen 为
  `fail_non_inferior`，按协议停止 5 Hz 分支。不要重跑、加训练步数、换 seed 或改门禁，
  只补写 B 阶段 decision/report。
- C 尚未写正式触发判定；现有证据默认应记录 `C_not_triggered`。只有可审计 artifact
  明确满足协议条件时，才允许启动一次预注册的有界 C。
- V-0D 主臂尚未实现或训练；V 组代码、配置、runner、summarizer 和测试仍不完整。S16 只能在
  S15 报告完成并冻结唯一胜出配置后启动正式训练；V 的实现、测试和 CPU smoke 可以提前
  准备。

## 固定执行顺序

```text
P0（复用）
→ P1（已完成负结果）
→ P2（已完成 no_signal）
→ P3（已完成 not_applicable）
→ 正式 A0/A1/A2（smoke 与 1×20 已完成；A2 仅相关性，不扩因果 3×32）
→ B10（通过非劣门禁后才 B5）
→ C（仅协议规定的 A/B 证据成立时启动；否则记录 not triggered）
→ 冻结唯一 S15 基线
→ V-0D 主臂：R-BC-0D（state + 8D path -> action，command_dim=0）
→ R-BC-3D 与 R-BC-0D 闭环对照
→ R-Z0-0D 零残差等价
→ R-PPO-0D screen（1×20）
→ 可选 V-3D 诊断控制（仅 0D 失败或需要归因时启动）
→ V-0D 正式比较（3×50 + CPU PVS 64-seed）
→ 最终报告
```

P 是前置筛查，不替代正式 A/B/C；A/B/C 未按门禁结束前不得启动 V。A 组当前没有可归因的
时间感受野收益，A0 仍是 S15 的回退基线；这不阻止按协议把 B 当作效率实验启动。V0 失败就
停止 V 分支；V1 失败则保留 ALOS 外置方案并记录负结果。C、V1b、V2 若前置证据不成立，
正式记录 `skipped/not triggered`，不要静默跳过。

## 每个阶段的自动循环

1. 盘点实际状态：确认前置门禁、已有 run、PID、queue.lock、日志、artifacts 和资源。
2. 补齐缺失实现：配置、runner、summarizer、latency benchmark、消融入口和测试缺失时，
   先实现并验证，再执行正式实验。
3. 本地验证：运行相关 pytest、ruff、schema/import smoke 和必要的 CPU smoke，修到契约通过。
4. 服务器 preflight：同步代码，检查 venv、GPU、磁盘、CPU 核绑定、目录、PID 和已有锁。
5. 契约 smoke：按协议训练 seed × 8 个评估 seed，检查 finite audit、schema/hash、
   无特权泄漏、物理时长、决策计数、动作保持、reward/gamma 时间尺度和历史采样。
6. 低预算筛查：按协议训练 seed × 20 个固定评估 seed，逐 seed 配对；P 组固定使用
   `1000–1019`。不得换 seed 挑结果。
7. 写 `summary.json`、`summary.md`、训练/评估 manifest、配置 hash、checkpoint hash、
   schema hash、逐 seed 记录和门禁判定。pass 才进入下一阶段；fail 写负结果并按分支规则
   停止或降级。
8. 只有协议允许且筛查有可解释信号时才扩到 3×32。正式比较通过后，只让唯一候选进入
   CPU PVS 64-seed；不根据评估 seed 反挑 checkpoint。
9. 每阶段更新协议文档、阶段报告、产物路径、偏离/重试记录和
   `doc/goals/goal9_execution_state.md`。状态账本至少记录 `updated_at`、`current_stage`、
   `completed_nodes`、`running_jobs`（host/PID/log/lock）、`failed_or_skipped`、
   `next_action` 和恢复命令。

## 跨轮次自动继续（必须遵守）

- 每次接手先读取 `goal9_execution_state.md`，再以服务器实际状态为准；状态账本过期时不
  照搬其中的 PID，但仍用它定位上一次断点和应检查的产物。
- 若目标 run 的 runner PID 仍存活、`run_manifest.json` 为 `running` 或 `queue.lock`
  被持有：只等待、接管和轮询，不启动第二个 run。若进程已死但 manifest 仍是 `running`，
  先检查日志和逐 job `rc`，把基础设施失败与科学负结果分开，再按协议最多重试两次。
- 若 manifest 已是 `completed`：立即解析 `summary.json`、`summary.md`、训练审计和逐 job
  `rc`，写 stage decision，然后推进到下一个允许阶段；不要把“summary 已生成”当成“报告
  已写完”或“S15 已冻结”。
- 不得在一个阶段完成后停下来等待用户确认。只有前置门禁失败、分支按协议停止、全部可执行
  分支完成，或触发下面的暂停条件时，才结束本轮。
- 会话或 token 即将结束时，先更新 `goal9_execution_state.md` 的当前阶段、运行中 PID/锁、
  已完成节点、失败/跳过节点、下一动作和可直接执行的恢复模板；然后结束本轮，下一轮从账本
  继续。不得只留下聊天中的口头状态。
- Goal 自动继续时先做只读状态盘点；只有确认没有活跃同目录 run 后才能提交下一阶段。

## A/B/C 的执行要求

- A 组只改变历史采样 stride，物理 `dt_s`、actor 频率、DADE/MLP 结构和动作包线不变。
  必须运行 `history-zero` 和 `history-reverse` 机制检查；只有总指标改善但历史消融没有
  可解释退化时，只能登记相关性，不能写成时间感受野的因果收益。`grid_hs` 等预注册 tag
  必须报告，不能静默忽略。
- B 组只改变 actor 决策频率；物理、传感器、安全终止和速率限制仍为 50 Hz。动作在决策
  间隔内零阶保持；reward、return、gamma 和 timeout 按物理时间换算。B10 与 A1 配对，
  B5 与 A2 配对；B10 失败就停止降频分支，不用增加训练步数补偿。
- B 组实现必须重点验证：每个物理步仍执行安全检查；中途 done 能在决策间隔内正确终止；
  同一决策间隔的 reward 正确累加；gamma 和 timeout 使用物理时间；训练物理秒数与配对 A
  组一致；动作保持不会泄漏非 active 行的历史、传感器、guidance 或 hold tracker 状态。
- B 组已知风险：done 行在同一决策间隔的后续物理步仍可能被 inner env 推进。即使 reward
  已屏蔽，也必须确认终止原因、timeout/terminal 元数据和评估指标没有被后续步污染；若会
  污染，按行冻结并补测试后再跑 B10 smoke，不能带着该风险直接进入正式 screen。
- C 组只有在 A/B 证明时间尺度或历史采样有影响、且 `DADE + MLP` 仍有明确通路瓶颈时
  启动。输入必须包含当前状态 token、命令 token 和历史 token；先对 BC/最佳基线动作做监督
  蒸馏，再做一次有界微调；参数规模和 p95 时延与 DADE+MLP 同量级。无稳定收益立即停止。

## V 组的执行要求（历史，已被 V-0D 主臂取代）

> 以下 V0/V1/V1b/V2/V3 内容只保留为历史。当前执行必须使用
> `doc/goals/goal9_v0d_direct_pure_path_plan_20261106.md` 和
> `doc/goals/goal9_v0d_non_destructive_prompt_20261106.md`，不要再按下面旧的
> ALOS-free V0/V1 顺序启动训练。

- V0 保留固定速度参考，先隔离“纯路径控制是否成立”；V1 删除显式 `u_d`，actor 仍输出
  `[rudder, stern, rpm]`，由 RPM 和奖励隐式学习速度。
- 路径特征必须作为 `o_t` 的一部分进入 encoder 得到 `z_t`；DADE 历史也只能使用同一套
  包含路径的 `z_t`。禁止 raw `path_t` 绕过 encoder，禁止 ALOS 的 `psi_d/theta_d` 或
  V1 的 `u_d` 从 side channel 进入学生 actor。
- 训练和评估必须覆盖非零初始纵荡速度、非零起步 RPM、路径/横向/垂向/航向/俯仰偏差起步、
  低速和中速、曲率段及恢复状态。初始状态属于环境，不作为 actor 特权观测。V0/V1 前必须
  确认 reset 不会把初始 RPM 或速度清零。
- 奖励以路径误差、进度和任务时间为主；速度平滑优先于动作平滑，能耗尽量优化；安全、
  timeout 和速度包线是硬约束。material saturation 只作为执行器使用风险报告，不作为
  通过/否决门禁。不加过冲惩罚，只把过冲作为诊断指标；不要求终端减速。
- 默认不加 speed head；V0/V1 actor 输出始终是 `[rudder, stern, rpm]`。只有 V1 特别
  糟糕且有书面证据并复核排除奖励/观测问题时，才允许启动 V2。
- V0/V1 先 smoke 再 1×20；V0 未通过停止 V；V1 未通过保留 ALOS 外置方案并记录负结果。
  V1b 只在转弯明显不减速且 primary 指标可接受时启动；V2 是独立 arm，不能与 V1/V1b
  同时改动；V3 只对通过的唯一候选跑 3×32 和 CPU PVS 64-seed。

## 自动继续与资源规则

以下情况自动处理，不要逐步询问用户：

- 普通 smoke 失败：修复实现，重跑同一 smoke，不扩大 seed 或预算；
- 某个 arm 或分支负结果：写负结果，按协议停止该分支或使用规定降级基线；
- GPU 显存不足、排队、SSH 短暂中断：按协议换独立 GPU、等待、重试或恢复；
- 配置、runner、summarizer、benchmark 或测试缺失：先补齐再执行；
- C、V1b、V2 前置证据不成立：记录 `skipped/not triggered` 及依据，再进入下一允许阶段。

长任务必须用 tmux/nohup 等持久化后台进程托管，记录 PID、锁、日志、GPU 和 CPU 绑定；
SSH 断开不得终止实验。轮询使用长间隔，不要高频空转；等待期间只做不依赖该结果且不冲突的
代码、测试或报告。同一输出目录已有 `queue.lock` 或活跃 runner 时，不得启动第二个。

可以调用 subagent 做独立只读审计、测试或互不重叠的实现，但主 agent 必须统一核对产物、
运行门禁并写报告；subagent 不得修改 seed、阈值、预算或实验顺序。不得一边等待训练一边
启动依赖该结果的下一个正式阶段。

只有以下情况暂停并汇报：

- 服务器或凭证不可达，重试 3 次仍无法恢复；
- 同一基础设施故障连续重试超过协议允许次数；
- 文档冲突无法用保守解释解决且会改变科学结论；
- 继续执行会覆盖历史结果、破坏 checkpoint 或造成不可逆数据丢失；
- 资源条件已无法安全启动任何下一阶段，需要人工扩容或清理；
- 所有可执行分支已经结束，需要给出最终报告。

## 硬约束

- 一次只改变一个科学变量；同组 arm 必须同 BC/教师数据、同训练物理秒数、同更新次数、
  同评估 seed 集和同硬件条件。
- actor 只能使用可部署观测；真实海流、故障标签、仿真真值和 critic 特权量不得进入部署路径。
- BC 标签是执行器动作 `[rudder, stern, rpm]`，不是中间制导量。
- 任何 Torch 结果都只是预筛；正式结论必须经过协议规定的 CPU PVS 复核。
- 禁止换评估 seed、改门禁阈值、追加训练步数、增加预算、调 reward 权重或增加第二个科学
  变量来抢救失败结果；所有偏离必须写入报告。
- 禁止覆盖历史 artifacts、checkpoint、日志或已有 run 目录。
- 不使用 `git add -A`；不自动 commit/push；不提交 artifacts、checkpoint、日志或凭据。
- 每个 run 记录 GPU 型号/显存、torch/CUDA、配置 hash、checkpoint hash、训练 seed、
  评估 seed、评估 tag 和 wall-clock。

## 完成标准与最终交付

全部可执行分支结束后，必须给出：

- 实际完成和未完成的阶段；
- 每个阶段的正结果、负结果、判定依据和协议门禁；
- S15 冻结的唯一基线配置、checkpoint、schema 和评估口径；
- S16 候选以及 V0/V1/V1b/V2/V3 的执行状态；
- 是否真正完成 3×32 和 CPU PVS 64-seed；
- 所有产物目录和报告路径；
- 所有偏离、重试、基础设施问题和未执行项；
- 哪些结论可以进入论文，哪些只能作为负结果、相关性或待验证项；
- 最新 `CURRENT_STATE`：已完成、运行中 PID/队列、下一阶段和阻塞原因。
