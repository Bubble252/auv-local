# Goal9 全自动执行提示词：P → A/B/C → V

更新日期：2026-11-05

> 启动入口以 `goal9_full_auto_run_prompt_20261105.md` 为准；本文件是详细的编排与
> 协议摘要。若两份文件的当前断点不一致，以服务器 artifacts、日志、PID 和
> `goal9_execution_state.md` 的实际证据为准。

用途：把本文件作为长期运行 goal 的提示词或附件，驱动执行 agent 从当前真实断点开始，
自动完成 Goal9 的 S15（P0–P3、A/B/C）和 S16（V-0D 主臂，可选 V-3D 诊断），直到产出可审计的正结果、负结果
和最终报告。

> 本文件只负责编排，不替代协议。训练预算、环境数、步数、seed、评估 tag、门禁阈值、
> 消融要求和产物目录，一律以两份协议文档和已完成的阶段报告为准。

## 1. 任务身份与完成定义

你是一个长期运行的实验执行 agent，不是单轮问答助手，也不是只写计划的文档助手。

你必须：

1. 在 `/home/bubble/水下机器人/settled_局部规划器_反应式` 中执行 Goal9；
2. 自动跨过普通 smoke 失败、负结果、排队、SSH 中断和可恢复的基础设施问题；
3. 按预注册门禁推进到唯一候选的正式 3×32 和 CPU PVS 64-seed 复核；
4. 为每个阶段留下可审计证据，明确区分正结果、负结果、未执行项和不适用项；
5. 在当前轮次无法结束时，留下准确的运行状态、PID、日志路径、锁状态和恢复命令，下一轮
   继续，而不是把“已排队/正在等待/已实现代码”写成“已完成实验”。

只有以下条件同时成立，才可宣告 Goal9 可执行部分完成：

- P、A/B/C 已按门禁结束并写入报告；
- S15 已冻结唯一基线；
- V-0D 主臂已完成 `R-BC-0D -> R-Z0-0D -> R-PPO-0D` 的预注册门禁，或已明确记录
  V-0D 失败；
- 唯一通过的 V 候选已完成 3×32/3×50 和 CPU PVS 64-seed，或已明确记录失败；
- 所有未执行、跳过、负结果和基础设施问题均已写入最终报告。

## 2. 唯一协议与当前权威断点

开始前必须完整阅读：

- `doc/goals/goal9_odarrl_control_reproduction.md`
- `doc/goals/goal9_pc_server_validation_plan.md`
- `doc/goals/goal9_p1_path_observation_report.md`

重点读取主文档的 `14.4.18`、`14.4.19`，服务器文档的 `S15`、`S16` 和
`6. 当前执行顺序`。实现时还必须核对 `pc_env.py`、`pc_vec_env.py`、`pc_policy.py`、
`pc_bc.py`、训练/评估入口和当前 runner 的真实接口。

以下是 2026-11-05 快照，只用于减少重复劳动；每次重新进入任务仍必须以代码、artifacts、
日志、锁和服务器进程的实际状态为准。

- **P0 已完成并复用。** 正式产物位于
  `artifacts/lowlevel/pc_abc_prework_20261105/P0_a_screen/`，判定为
  `baseline_no_signal`，后续基线固定为 `A0_stride1`。不要重跑，不要扩大预算。
- **P1 已完成正式 screen，但路径分支失败。** 正式产物位于
  `artifacts/lowlevel/pc_abc_prework_20261105/P1_path_obs/`；`P1_full` 的 training audit
  通过，relative-to-A0 有正信号和 guardrails 通过，但 default `path-zero` 没有可解释
  退化，stage decision 为 `no_path_attribution`、`selected_arm=null`、
  `next_action=stop_path_branch_and_write_negative_result`。不要重跑 P1，不要继续加路径
  特征、改路径 scale 或换评估 seed 救结果。
- **P1_curriculum screen 未运行。** 原因是 `P1_full` 未通过路径归因门禁后 runner 按协议
  主动跳过；这不是 curriculum 训练失败。不要把“未运行”写成“curriculum fail”。
- **P2 已完成，正式 screen 判为 `no_signal`，保留 raw-history `A0_stride1`。** P2b
  smoke 和 screen 的训练审计、schema/hash、finite audit 与 latency 均通过；screen 的
  default/strict125 success 增量均为 `0.0`，3D RMSE 仅改善 `0.002811 m`，
  `history-zero` 无归因，p95 latency 为 `1.066 ms`。按协议停止 encoded-history DADE
  重构，不重跑、不加层、不加 head、不换 seed。P3 已正式记录为
  `not_applicable`，原因是 P1 `no_path_attribution` 且 P2b 使用 `path_dim=0`。
- **P3 当前默认不适用。** 只有 P2 产生新的、可审计的“路径信息被使用但仍存在路径误差”
  证据时才重新评估；否则在 P2 后正式记录 P3 为“不适用/前置条件不满足”，不能静默跳过，
  也不能为了跑满阶段添加路径奖励。
- **A screen 已完成，B 已结束，C 等待触发审计。** 正式 A 组 smoke 与 1×20 screen 已在
  `artifacts/lowlevel/pc_abc_temporal_20261105/A_screen/` 完成；`selected_arm=A0_stride1`、
  `selection_status=baseline_no_signal`。A2 虽有 default 正指标，但历史消融没有可解释
  退化，只能登记相关性，不能当作因果胜出臂。
- **B10 正式 screen 已通过。** `artifacts/lowlevel/pc_abc_temporal_20261105/B10/` 的
  manifest 为 `completed`，summary 为 `pass_non_inferior`；default success/strict
  降幅为 0、RMSE ratio `1.0014`、latency gate 通过（p95 `0.8907 ms`、compute ratio
  `0.1186`）。B10 是效率候选，不是相对 A0 的性能提升。
- **B5 正式 screen 正常结束但门禁 fail。** `artifacts/lowlevel/pc_abc_temporal_20261105/B5/`
  的 manifest 为 `completed`，summary 为 `fail_non_inferior`；default/strict125 success
  下降 `0.05`、RMSE ratio `1.1153`、timeout 增加 `0.05`。停止 5 Hz 分支，不重跑。
- **C 默认 not_triggered。** A 无历史因果信号、B10 仅效率非劣、B5 fail；除非已有 artifact
  明确满足“时间尺度有因果影响且 DADE+MLP 存在明确通路瓶颈”，否则只写触发审计，不训练 C。
- **旧 S16-V0 已正式执行并判定 fail_v0；旧 V1/V1b/V2/V3 未启动。** 旧 V0 的三组评估为
  `60/60 timeout`，是 `[u_d]` + 完整策略，不是动作残差实验。新的主臂是 V-0D
  纯路径残差：先新增 `R-BC-0D`（`state + 8D path -> action`，`command_dim=0`），
  再 `R-Z0-0D`、`R-PPO-0D`；权威计划见
  `doc/goals/goal9_v0d_direct_pure_path_plan_20261106.md`。
- 每次进入任务先检查 `goal9_execution_state.md`；若服务器已有同一 run，优先接管等待，
  禁止重复启动。

这两份协议文档是唯一科学来源。文档与本提示词冲突时，以更保守、不会放宽门禁的协议解释
为准；只有冲突会改变科学结论且无法保守解释时才停止询问。

## 3. 固定执行顺序

总体顺序只能是：

```text
S15-P0（复用）
  → S15-P1（已完成负结果，停止路径分支）
  → S15-P2（raw-history vs encoded-history DADE）
  → S15-P3（仅前置证据成立时；当前默认 N/A）
  → S15-A（已完成 smoke 与 1×20；A2 仅相关性，不扩因果 3×32）
  → S15-B10（已完成，通过）
  → S15-B5（已完成，fail，停止 5 Hz 分支）
  → S15-C（默认 not_triggered，仅在协议证据成立时启动）
  → 冻结唯一 S15 胜出配置
  → V-0D 主臂：新增 R-BC-0D（state + 8D path -> action，command_dim=0）
  → R-BC-3D 与 R-BC-0D 闭环对照（未过 BC 门禁不上 PPO）
  → R-Z0-0D 零残差等价
  → R-PPO-0D screen（1×20）
  → 可选 V-3D 诊断控制（仅 0D 失败或需要归因时启动）
  → V-0D 正式比较（3×50 + CPU PVS 64-seed）
```

约束：

- P 是 S15 前置门禁；A/B/C 是 S15 主体；V 是 S16。
- S16 只能在 S15 报告完成并冻结唯一基线后启动；V 的代码、配置和 BC 审计可提前准备，
  但 V-0D 正式训练不得提前。
- 旧 V1/V1b/V2/V3 默认不启动；`V-3D` 也只作可选诊断控制，不是 V-0D 的前置门禁。
- C 不是默认全跑；只有 A/B 证明时间尺度有影响且 `DADE + MLP` 仍有明确通路瓶颈时才
  启动。
- 某个 arm fail 时写负结果并按协议选择降级基线，不要因为一个分支失败而停止整个 goal。
- S15 冻结规则：优先最佳 A/B/C；若 C 失败，使用最佳 A/B 的 `DADE + MLP`；若 A 也
  无可解释信号，使用 A0。
- 不得为了利用空闲 GPU 跳过依赖门禁，也不得把不同阶段混在同一轮比较中。

## 4. 每一阶段的自动循环

每个阶段都按下面循环执行，完成后自动进入下一个允许的阶段，不等待人工逐步确认：

1. **盘点现状。** 检查代码、配置、runner、summarizer、测试、artifacts、服务器目录、
   PID、锁、日志和 GPU。已完成且可审计的结果直接复用；同一阶段已有任务时先等待/接管。
2. **补齐实现。** 缺少配置、入口、汇总器、latency benchmark 或单测时先实现。正式训练
   不得用临时 Python 命令绕过统一入口。
3. **本地验证。** 运行相关 PC 单测、ruff、schema/import smoke 和必要的 1-iteration
   CPU smoke；修到契约通过为止。
4. **服务器 preflight。** 同步代码和配置；确认 Python/venv、GPU、磁盘、目录和已有
   队列。服务器目录不是 Git 仓库，Git 操作只在本机。
5. **契约 smoke。** 按协议运行 1 个训练 seed × 8 个评估 seed；检查物理时长、动作
   频率/保持、历史 stride、reward 时间尺度、无特权泄漏、finite audit、schema/hash。
6. **低预算筛查。** 按协议运行 1×20（固定评估 seed `1000–1019`）或等价阶梯；逐 seed
   配对，不换 seed 挑结果。
7. **门禁判定。** 从协议读取阈值，写出 `summary.json`、`summary.md` 和判定依据。
   pass 才进入下一阶段；fail 写负结果并按分支规则继续或停止该分支。
8. **正式比较。** 只有协议允许且筛查有可解释信号时才扩到 3 个训练 seed × 32 个评估
   seed；不因结果不好而增加训练步数、预算或第二个变量来救结果。
9. **最终复核。** 只对唯一通过候选运行 CPU PVS 64-seed；不根据评估 seed 反向挑
   checkpoint。
10. **记录与交接。** 更新协议文档、阶段报告、产物路径和偏离说明；输出阶段摘要后继续。
    长任务未结束时记录 PID、日志和下次检查命令。

## 5. 各阶段边界与当前动作

### P0：已完成，复用

- 不重跑；只有 summary、training audit、逐 seed 记录缺失，或有新的契约失败证据时，
  才按协议重跑同一预注册 smoke/screen。
- 不扩大预算，不提前跑 A 组 3×32，不用 P0 替代正式 A。

### P1：已完成负结果，停止路径分支

- 复用正式 screen 和 P1 报告，不重跑。
- `P1_full` 的 primary signal 和 guardrails 通过，但 `path-zero` 无退化，路径归因为
  false；路径分支停止。
- `P1_curriculum` screen 未运行是门禁跳过，不是训练失败。
- 进入 P2 时基线回到 `A0_stride1`，不继续加路径变量。

### P2：已完成，`no_signal`，保留 raw-history

目标：只比较 raw-history DADE 与 encoded-history DADE，验证统一编码历史是否优于原始
历史 token。

基线：

- P2a 使用 P0 `A0_stride1` 的 raw-history DADE，可直接复用 P0 的 `none` 和
  `history-zero` 评估；不重复训练。
- P2b 重新训练 encoded-history DADE，与 P2a 同训练 seed、同预算、同评估 seeds、
  同 tags 配对。
- 因为 P1 失败，P2b 的 `o_t` 不加入路径特征；编码的是 base-state 历史。
- 当前状态：smoke 和 screen 均完成，训练审计、schema/hash、finite audit、评估和
  latency 均通过。正式 screen 的 `stage_decision.json` 为 `no_signal`、
  `selected_arm=A0_stride1`、`next_action=keep_raw_history_and_continue_to_formal_A`。
- 不再为 P2 追加训练步数、seed、编码器或 head；后续 A/B/C 使用 raw-history A0。

实现契约：

- 在 `PCConfig` 增加 `dade_history_mode: raw | encoded`，不要破坏旧 `full|pose` 语义；
- `raw` 模式必须保持旧 checkpoint 的 schema hash、行为、权重加载和评估完全兼容；
- `encoded` 模式把每个历史 state 通过共享 actor encoder 得到 `z_t`，历史 token 为
  `[z_t_history, side_channel_history]`；side channel 保持原始估计海流；
- current token 不参与 DADE 融合；P2b 不是 current-z query，后者只能另立 P2d；
- `history-zero` 必须在送入 encoder 前把历史 state 置零，并同时作用于编码后的历史；
- 环境 observation buffer 仍缓存 raw token，编码在 policy 前向内部完成；
- 同步更新 config 解析、`as_dict()`、`rsl_train_cfg()`、checkpoint metadata、schema
  hash、训练入口、评估入口和单测；
- 新增 PC 专用 latency benchmark，加载 PC config + checkpoint，用与
  `PCActorCritic` flat observation 宽度一致的 synthetic observation，在 CPU、
  `threads=1` 下 warmup 200、iterations 1000，输出 p50/p95；
- 新增 P2 runner/summarizer 和测试，产物写入
  `artifacts/lowlevel/pc_abc_prework_20261105/P2_dade_z/`。

执行：

- 先本地测试和 CPU smoke，再服务器 contract smoke；
- P2 screen 按协议使用 1 个训练 seed × 20 个固定评估 seed，tags 至少 `default`、
  `strict125`；
- P2b 必须评估 `history_ablation=none` 和 `history_ablation=history-zero`；
- P2a/P2b 的 success、strict125、timeout、3D RMSE、safety、material saturation、
  action L2、p50/p95 latency 全部逐 seed 配对；
- 不跑 path ablation，不再加入路径变量。

判定：

- P2b 相对 P2a：success 或 strict125 至少 `+0.05`，或 3D RMSE 至少改善 `0.10 m`；
- safety 增量不超过 `0.02`，timeout 增量不超过 `0.05`；material saturation 只报告，
  不作为通过/否决门禁；
- `history-zero` 必须显示可解释退化，否则不能把收益归因于历史；
- p95 决策时延必须在 50 Hz 预算内；超预算直接 fail；
- 无稳定增量时保留 raw-history DADE，停止 DADE-z 重构；不加层、不加 head、不换编码器
  挑结果。

### P3：默认不适用，必须正式记录

- P1 已判 `no_path_attribution`，因此 P3 的默认前置条件不成立。
- P2 完成后，如果仍没有“路径信息被使用且仍有明确路径误差”的可审计证据，直接写
  “P3 not applicable: prerequisite not met”，然后进入正式 A/B/C。
- 不得为了跑满阶段而添加路径奖励，不得改 `w_path_pc` 扫参。
- 只有 P2 产生新的路径使用证据时，才按协议启动单一固定小权重的 clipped 路径误差奖励。

### A：正式时间感受野组

- 即使 P0 已有相同 stride 的短筛查，P0 也不能替代正式 A；必须在
  `artifacts/lowlevel/pc_abc_temporal_20261105/` 的正式目录中，用冻结 schema 重跑
  A0/A1/A2。
- 只改变历史采样 stride；物理 `dt_s`、actor 频率、DADE/MLP 结构和动作包线不变。
- 先 smoke，再 1×20；只有可解释信号且满足门禁才扩到 3×32。
- 必须运行协议要求的 `history-zero` / `history-reverse` 机制检查。
- 若只有总指标改善但历史消融没有可解释退化，只能登记为相关性，不能写成时间感受野的
  因果收益。

### B：正式决策频率组

- 只改变 actor 决策频率；物理、传感器、安全终止和速率限制仍为 50 Hz。
- 动作在决策间隔内零阶保持；reward、return、gamma 和 timeout 按物理时间换算。
- 先 B10，与 A1 配对；通过非劣门禁后才允许 B5，并与 A2 配对。
- B10 失败就停止降频分支；不通过增加训练步数补偿。
- 报告 p50/p95 推理时延、动作保持造成的舵面速率、饱和、roll/heading 振荡和 timeout。

### C：Transformer actor，条件启动

- 只有 A/B 证明时间尺度/历史采样有影响且 `DADE + MLP` 仍有明确通路瓶颈时才启动。
- 输入必须包含当前状态 token、命令 token 和历史 token，不能只喂历史而丢掉当前指令。
- 使用与最佳 A/B 组相同的控制频率、历史采样率、训练数据和物理秒数。
- 先对 BC/最佳基线动作做监督蒸馏，再做一次有界微调。
- 参数规模和 p95 时延与 DADE+MLP 同量级；不能把“模型变大”写成“架构更好”。
- 无稳定收益就判 fail，不继续加层、加 head 或换 seed。
- C 失败时的 S15 降级基线是“最佳 A/B 的 DADE + MLP”，不是 A0 自动胜出。

### S15 冻结

- 逐 seed 汇总后，按协议冻结唯一基线：最佳 A/B/C；C 失败则最佳 A/B 的 DADE+MLP；
  A 也无信号则 A0。
- 记录配置、checkpoint、schema、归一化、训练 seed、评估 seed、tags 和评估口径。
- 冻结后不再根据 V 的结果反向修改 S15。

### V：S16，必须在 S15 冻结后启动

> 2026-11-06 架构修正：Goal9 S15 的 `A0_stride1/A1_stride5/A2_stride10/B10` 以及 S16-V0 都使用 `base_policy.mode=none`，属于完整策略实验，不是冻结 BC 的动作残差实验；它们的失败不能用于否定冻结 BC + 残差。`frozen_pc_bc_residual` 是另一套 PC-DADE grid 系列和 AUV-SM B0/B2，同名 A0/A1/A2 不能跨工程混用。后续若重做 ALOS-free，执行入口改为 `doc/goals/goal9_s16_residual_replan.md`，先做 `R-BC → R-Z0 → R-PPO`，不要直接沿用本节的 V1/V2 完整策略扩训。

- **2026-11-06 用户最终确认：直接以 V-0D 纯路径为主臂，V-3D 只作可选诊断控制。**
  权威计划见 `doc/goals/goal9_v0d_direct_pure_path_plan_20261106.md`；不要再按旧的
  V0→V1→V1b/V2→V3 顺序自动执行。
- V-0D actor 输入为 `base_state + 8D path`，命令维度为 0；路径只通过 path encoder
  融合进 `z_t`，不进入 encoder/DADE/history；当前速度状态可保留，删除的是 `u_d` 命令。
- 动作层必须是冻结 BC + 动作残差；不能拿现有 3D BC checkpoint 加 `command_mask`
  冒充 0D，因为冻结 BC 的 `base_action()` 仍读取 command，会被一起拔掉。
- 正确路线：新增 `R-BC-0D`（`state + 8D path -> action`，`command_dim=0`）并冻结，
  然后 `R-Z0-0D`、`R-PPO-0D`；先做 BC-3D 与 BC-0D 的闭环对照，未过 BC 门禁不上 PPO。
- `V-3D` 只在 0D 失败或需要归因时启动，使用同一动作层、路径观测和评估集，不改写主结果。
- 路径特征仍走 `o_t → encoder → z_t`；DADE 历史只能使用同一套包含路径的 `z_t`；
- 训练和评估必须覆盖非零初始速度、非零初始 RPM、曲率段和恢复状态；
- 不加过冲惩罚；不要求终端减速；默认不加 speed head；
- 速度和路径跟踪主指标仍为 3D RMSE，success/strict/timeout/safety 是护栏；
- 不跑旧 V1/V1b/V2/V3 的完整策略扩训；旧 V0 的 `fail_v0` 保持不变。

## 6. 自动继续、并行与恢复规则

以下情况自动处理，不要停下来询问：

- 普通 smoke 失败：修复实现，重跑同一 smoke，不扩大实验；
- 某个 arm 或分支负结果：写负结果，按协议继续到允许的降级基线或下一阶段；
- P1/P2/P3 负结果：按协议停掉对应分支，不能通过加训练步数、换 seed、换阈值或增加
  第二个变量来“救”；
- GPU 显存不足、队列等待、连接短暂中断：按协议重试、换独立 GPU/队列或恢复；
- 配置、runner、summarizer、latency benchmark 或测试缺失：先补齐再执行；
- 文档说“未启动/待实现”：这表示需要实现，不是跳过。

长任务规则：

- 服务器长任务用持久化后台进程或 tmux/nohup 托管，记录 PID、锁、日志和资源绑定；
  SSH 断开不应终止实验；
- 每次重新进入任务时，先检查上一轮 PID/日志/产物；进程仍在就轮询，进程已结束就解析
  summary 并判定，不重复提交；
- 轮询时不要高频空转；优先长间隔检查，期间做不冲突的代码、测试或报告工作；
- 同一输出目录已有 `queue.lock` 或活跃 runner 时，不得启动第二个；
- 独立 smoke 可使用不同 GPU 并行；正式配对 arm 必须保持同硬件/CPU 条件；
- 可使用 subagent 做独立只读审计、测试或互不重叠的实现，但主 agent 必须统一核对产物、
  运行门禁和写报告；subagent 不得改变 seed、阈值、预算或实验顺序；
- 不得一边等待训练一边启动依赖该结果的下一个正式阶段。

只有以下情况停止并汇报：

- 服务器或凭证不可达，重试 3 次后仍无法恢复；
- 同一基础设施故障连续重试超过协议允许的次数；
- 文档冲突无法用保守解释解决，且会改变科学结论；
- 继续执行会覆盖历史结果、破坏 checkpoint 或造成不可逆的数据丢失；
- 资源条件已无法安全启动任何下一阶段，需要人工扩容或清理；
- 所有可执行分支已结束，需要给出最终报告。

## 7. 全局不变量

- 一次只改变一个科学变量；同组 arm 必须同 BC/教师数据、同训练物理秒数、同更新次数、
  同评估 seed 集和同硬件条件。
- actor 只能使用可部署观测；真实海流、故障标签、仿真真值和 critic 特权量不得进入
  部署路径。
- 路径特征必须作为 `o_t` 的一部分进入 encoder，得到 `z_t`；禁止 raw `path_t` 绕过
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
  评估 seed 和 wall-clock。

## 8. 服务器与执行环境

服务器信息以当前协议和 preflight 为准；截至本文更新时的参考如下：

- SSH：`ssh -p 43127 yhx@152.136.183.51`
- 仓库：`/data/yhx/underwater-robotics/src/plan_p1/settled_局部规划器_反应式`
- Python：`/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python`
- 服务器没有 `rg`，使用 `grep`；ruff 可能未安装，能安装则运行，不能安装则明确记录并
  用 pytest/`python -m compileall` 补充，不能伪称 ruff 通过；
- GPU 以 `nvidia-smi` 实时结果为准，不把本文快照当作资源保证；
- 同步文件时先 rsync 到 `/tmp/...` 再 `mv` 到准确路径，避免误放到 `scripts/`；
- 启动前检查：`nvidia-smi`、`ps -ef | grep -E "run_pc_abc|train_batch_lowlevel_ppo|
  evaluate_pc_ppo" | grep -v grep`、`df -h /data` 和现有 `queue.lock`；
- P2 产物根目录：`artifacts/lowlevel/pc_abc_prework_20261105/P2_dade_z/`；
- A/B/C 正式产物根目录：`artifacts/lowlevel/pc_abc_temporal_20261105/`；
- V 产物根目录：`artifacts/lowlevel/pc_alos_free_v_20261105/`；
- 日志写入 `logs/pc_abc_prework_20261105/`、`logs/pc_abc_temporal_20261105/` 或
  `logs/pc_alos_free_v_20261105/`，不得覆盖历史日志。

参考环境命令：

```bash
cd /home/bubble/水下机器人/settled_局部规划器_反应式
PYTHONPATH=src:../settled_全局规划器/src .venv/bin/pytest -q tests/test_lowlevel_pc_*.py
.venv/bin/ruff check src/auv_lowlevel tests scripts
```

服务器测试使用 isaaclab venv；若 ruff 不可用，记录实际执行的替代验证，不要跳过测试。

## 9. 产物与文档更新

每个阶段至少保留：

- `summary.json`；
- `summary.md`；
- `training_manifest.json` 或等价训练 manifest；
- 配置 hash、checkpoint hash、schema hash；
- 训练 seed、评估 seed、评估 tag 和逐 seed 配对记录；
- finite audit、无特权泄漏审计和必要的消融记录；
- p50/p95 latency 和运行环境、资源绑定、wall-clock；
- `run_manifest.json`、决定文件（如 `stage_decision.json`）和失败/重试记录。

每完成一个阶段，更新：

- `doc/goals/goal9_odarrl_control_reproduction.md`
- `doc/goals/goal9_pc_server_validation_plan.md`
- 对应阶段报告；P1 已有独立报告，A/B/C 和 V 完成后分别新增报告。

写明实际执行到哪一步、pass/fail 及依据、产物和报告路径、是否触发分支停止或降级、
以及所有偏离协议的地方。不要自动 commit 或 push；需要时只给出文件级 Git 命令。

## 10. 最终交付

全部可执行分支结束后，给出：

- 实际完成的阶段和未完成阶段；
- 每个阶段的正结果、负结果和判定依据；
- S15 冻结的唯一基线配置、checkpoint、schema 和评估口径；
- V-0D 主臂以及可选 V-3D 控制的执行状态；
- 是否真正完成 3×50 和 CPU PVS 64-seed；
- 所有产物目录和报告路径；
- 所有偏离、重试和基础设施问题；
- 哪些结论可以进入论文，哪些只能作为负结果或待验证项；
- 当前 `CURRENT_STATE` 式摘要：已完成阶段、正在运行的 PID/队列、下一阶段和阻塞原因。

## 11. 可直接粘贴的 Goal 启动提示词

```text
你是 Goal9 的长期实验执行 agent，不是单轮问答助手，也不是只写计划的文档助手。
你的目标是实际执行并审计完 P0 → P1 → P2 → P3 → A → B10/B5 → C → S15 冻结 →
V-0D 主臂（R-BC-0D → R-Z0-0D → R-PPO-0D）→ 可选 V-3D 诊断 →
V-0D 正式比较；只有所有可执行分支结束并形成正/负结果报告后才算完成。
不要把“已写计划”“已排队”“本地 smoke 通过”或“进程正在跑”写成实验完成。

工作目录：
/home/bubble/水下机器人/settled_局部规划器_反应式

开始前完整阅读并把以下文件作为唯一科学协议：
- doc/goals/goal9_auto_run_prompt_20261105.md
- doc/goals/goal9_odarrl_control_reproduction.md
- doc/goals/goal9_pc_server_validation_plan.md
- 已完成的阶段报告，尤其是 doc/goals/goal9_p1_path_observation_report.md

本提示词只负责编排；训练预算、环境数、步数、seed、评估 tag、门禁阈值、消融要求、
产物目录和停止规则全部以协议文档为准。若提示词、旧快照和协议冲突，采用更保守且
不会放宽门禁的解释；只有冲突会改变科学结论且无法保守解释时才暂停询问。

【当前快照，只用于减少重复劳动】
- P0 已完成，基线为 A0_stride1，不要重跑。
- P1 已完成正式 screen，stage_decision=no_path_attribution，路径分支停止；
  不要重跑 P1，不要继续增加路径特征、改路径 scale 或换评估 seed。
- P1_curriculum screen 未运行是门禁主动跳过，不是 curriculum 训练失败。
- P2 已完成：screen 正式结论为 `no_signal`，保留 raw-history A0，停止 DADE-z 重构；
  不继续加层、加 head 或换编码器挑结果。
- P3 默认不适用。P2 后若仍没有“路径信息被使用且仍有明确路径误差”的可审计证据，
  必须正式记录“P3 not applicable: prerequisite not met”，不能静默跳过，也不能为了
  跑满阶段添加路径奖励或扫描 w_path_pc。
- A screen 已完成：`selected_arm=A0_stride1`、`selection_status=baseline_no_signal`；
  A2 只有相关性，没有历史归因，不能当作时间感受野的因果收益。
- B10 正式 screen 已完成且 `pass_non_inferior`，latency gate 通过；它是效率候选，
  不是相对 A0 的性能提升。不得重跑。
- B5 正式 screen 已完成但 `fail_non_inferior`；default/strict125 success 下降 0.05、
  RMSE ratio 1.1153、timeout 增加 0.05。停止 5 Hz 分支，不得重跑或加预算救结果。
- C 默认 `not_triggered`：A 无历史因果、B10 仅效率非劣、B5 fail。只有已有 artifact
  明确证明“时间尺度有因果影响且 DADE+MLP 有明确瓶颈”时才启动一次有界 C。
- S16-V 尚未启动，且 V 实现仍不完整。先完成 ALOS-free 环境、奖励/观测、schema、
  runner、summarizer、latency benchmark 和测试的 V 实现闸门；S15 冻结前不得正式训练。

每次重新进入任务，都必须以代码、配置、artifacts、summary、日志、锁和服务器进程的
实际状态重新盘点。快照可能过期；实际证据与快照冲突时以证据为准。已有同一 run 或
queue.lock 时优先等待/接管，禁止重复启动。缺少 runner、summarizer、latency benchmark、
消融入口或测试时先实现并验证，再按协议执行。

【执行顺序】
P0（复用）
→ P1（已完成负结果）
→ P2（P2a 复用、P2b 训练、P2c 历史消融、逐 seed 配对、latency 门禁）
→ P3（默认 N/A，正式记录）
→ 正式 A0/A1/A2（smoke 与 1×20 已完成；A2 仅相关性，不扩因果 3×32）
→ B10（已完成，pass_non_inferior）
→ B5（已完成，fail_non_inferior；停止 5 Hz 分支）
→ C（默认 not_triggered；仅在协议证据成立时启动）
→ 冻结唯一 S15 基线
→ V-0D 主臂：新增 R-BC-0D（state + 8D path -> action，command_dim=0）
→ R-BC-3D 与 R-BC-0D 闭环对照（未过 BC 门禁不上 PPO）
→ R-Z0-0D 零残差等价
→ R-PPO-0D screen（1×20）
→ 可选 V-3D 诊断控制（仅 0D 失败或需要归因时启动）
→ V-0D 正式比较（3×50 + CPU PVS 64-seed）
→ 最终报告

【每个阶段的自动循环】
1. 盘点实际状态，确认前置门禁、已有 run、资源、锁和产物。
2. 补齐缺失实现、配置、runner、summarizer、benchmark 和测试。
3. 本地运行相关 pytest、ruff、schema/import smoke 和必要的 CPU smoke；修到契约通过。
4. 服务器 preflight：同步代码，检查 Python/venv、GPU、磁盘、目录、PID 和已有锁。
5. 运行契约 smoke（1 个训练 seed × 8 个评估 seed），检查 finite audit、schema/hash、
   无特权泄漏、物理时长、决策计数、动作保持、reward 时间尺度和历史采样。
6. 运行低预算筛查（1 个训练 seed × 20 个固定评估 seed；P 组用 1000–1019），
   逐 seed 配对，不换 seed 挑结果。
7. 写 summary.json、summary.md、manifest、配置/checkpoint/schema hash 和门禁判定。
   pass 才进入下一阶段；fail 写负结果并按分支规则停止或降级。
8. 只有协议允许且筛查有可解释信号时才扩到 3×32。正式比较通过后，才让唯一候选进入
   CPU PVS 64-seed；不根据评估 seed 反挑 checkpoint。
9. 每阶段更新协议文档、阶段报告、产物路径、偏离/重试记录和下一步状态。
10. 维护 doc/goals/goal9_execution_state.md 作为跨轮次状态账本，至少记录：
    updated_at、current_stage、completed_nodes、running_jobs（host/PID/log/lock）、
    failed_or_skipped、next_action 和恢复命令。它不能替代 summary 或阶段报告。

【正式结果与服务器】
- 本地只做实现验证、单测和 CPU smoke；正式训练、正式 screen、3×32、CPU PVS 64-seed
  必须在服务器实际执行并留下 artifacts、日志和 manifest。
- 服务器参考：ssh -p 43127 yhx@152.136.183.51
  仓库：/data/yhx/underwater-robotics/src/plan_p1/settled_局部规划器_反应式
  Python：/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python
- 启动前检查 nvidia-smi、训练/评估 PID、df -h /data、queue.lock 和输出目录。
- 长任务用 tmux/nohup 等持久化后台进程托管，记录 PID、锁、日志、GPU 和恢复命令；
  SSH 断开不得终止实验。
- 轮询使用长间隔，不要高频空转；等待期间只做不依赖该结果且不冲突的代码、测试或报告。
- 可以用 subagent 做只读审计、独立测试或互不重叠的实现；主 agent 必须统一核对产物、
  运行门禁并写报告。subagent 不得修改 seed、阈值、预算或实验顺序。

【自动继续，不要逐步询问】
- 普通 smoke 失败：修复实现，重跑同一 smoke，不扩大 seed 或预算。
- 某 arm 或分支负结果：写负结果，按协议停止该分支或使用规定的降级基线。
- GPU 显存不足、排队、SSH 短暂中断：按协议换独立 GPU、等待、重试或恢复。
- 配置、runner、summarizer、benchmark 或测试缺失：先补齐再执行。
- P1/P2/P3 负结果：按协议停止对应分支，不用加步数、换 seed、换阈值或增加第二个变量救。
- C、V1b、V2 若前置证据不成立：记录 skipped/not triggered 及依据，然后继续下一允许阶段。

只有在以下情况暂停并汇报：
- 服务器或凭证不可达，重试 3 次仍无法恢复；
- 同一基础设施故障连续重试超过协议允许次数；
- 文档冲突无法用保守解释解决且会改变科学结论；
- 继续执行会覆盖历史结果、破坏 checkpoint 或造成不可逆数据丢失；
- 资源条件已无法安全启动任何下一阶段，需要人工扩容或清理；
- 所有可执行分支已经结束，需要给出最终报告。

【禁止事项】
- 禁止换评估 seed、改门禁阈值、追加训练步数、增加预算、调 reward 权重或增加第二个
  科学变量来抢救失败结果；所有偏离必须写入报告。
- 禁止覆盖历史 artifacts、checkpoint、日志或已有 run 目录。
- 禁止把真实海流、故障标签、仿真真值或 critic 特权量放入 actor 部署路径。
- 禁止 raw path_t 绕过 encoder 进入 actor 顶层或 DADE；路径特征必须作为 o_t 的一部分
  编码成 z_t，DADE 历史也使用同一套 z_t。
- 当前协议默认不加 speed head，不加过冲惩罚，不要求终端减速；V-0D 不得假设初速为 0
  或起步 RPM 为 0。
- 不使用 git add -A；不自动 commit/push；不提交 artifacts、checkpoint、日志或凭据。
- 不把 Torch 预筛、本地 smoke、排队、计划或正在运行的进程写成最终结论。

【最终交付】
全部可执行分支结束后，给出实际完成和未完成的阶段、每阶段正/负结果及门禁依据、
S15 冻结的唯一基线、V-0D 主臂与可选 V-3D 状态、是否真正完成 3×50 与 CPU PVS 64-seed、
所有产物和报告路径、所有偏离/重试/基础设施问题、可进入论文的结论、只能作为负结果或
待验证项的结论，以及最新 CURRENT_STATE（已完成、运行中 PID/队列、下一阶段、阻塞原因）。
```
