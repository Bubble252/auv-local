# Goal136：训练环境多样化、课程学习与压力测评计划

**日期**：2026-11-11<br>
**状态**：`P0_PHASE_A1_COMPLETED / C_LINE_S0_S1_S2_S3_IMPLEMENTED / CURRICULUM_STRATIFIED_GATE_REPAIRED_20261111 / C0_DIRECT_COMMAND_OOD_FROZEN_BC / GOAL143_C0_FAILED / PIVOT_TO_GOAL144 / HELDOUT_CLOSED`<br>
**结果**：`doc/goals/goal136_result_20261111.md`<br>
**上游**：Goal134 `SINGLE_SEED_ONLY / SAC_STRESS_HARM_AT_HIGH_STRESS`；Goal135 `authority-regularized SAC`；Goal86 `PPO_RESIDUAL_COLLAPSE`<br>
**性质**：定义训练分布与测评矩阵；本轮已完成 S0–S4 接线、1M pilot 启动和 C0
控制语义诊断。只新增 C0/C1 控制语义的诊断/修复开关，不改
reward/terminal/success/radius，不换 PPO/SAC 主算法。

## 0. 优先级与依赖

**当前优先级队列（2026-11-11 更新）**

1. `Goal139`：mixed-horizon SAC + PPO credit horizon（已完成并记录裁决）。
2. 本文件 `Goal136 C-line`：Phase A1 与 S0/S1/S2/S3 已完成；S3 的 PPO/SAC 训练循环
   hook、每 100k 冻结 dev 评估、promotion/rollback、stage-mix ramp、C0/C1 按行 direct
   command 与 C4–C6 stage-aware current alpha 均已接入并 smoke 通过。S4 的 6 个 1M
   pilot 已启动；PPO `C-full` 在 C0 连续 100k/200k/300k 只有 `0.10` success，
   判 `C0_DIRECT_COMMAND_OOD_FROZEN_BC`。同一 checkpoint 在关闭 direct command、
   保留 ALOS 后恢复到 `0.90`，因此 C-line 暂停在控制语义修复，转入 Goal143。
3. `Goal140`：全程纵向跟踪奖励；P2，必须等 C-line 课程有裁决后再启动，不得插队。

本文件此前写的 `P0_COMPLETED` 只表示 Phase A1 D-line 完成，不表示完整 C-line 完成。
C-line 的路径族、噪声 scale、actor current encoder 与训练期 alpha ramp 已经接通；
stage sampler、per-episode manifest、PPO/SAC 每 100k 冻结 dev 评估与 promotion/rollback
已接入，并已用极小 smoke 验证 checkpoint、评估 summary、决策日志和 state 文件都会落盘。
per-stage control 语义已补齐：同一 batch 内 C0/C1 行按 per-row stage mask 使用 direct
command，C2–C6 行走 ALOS；C4–C6 的 current encoder alpha 按 stage 进入点线性 ramp。
S4 的 1M pilot 已启动，但 C0 出现 direct-command/BC 先验不兼容；因此仍不能把
D-line/Goal138/Goal139 的静态混合结果写成完整阶段式课程结论。正式课程效果必须等
Goal143 的 C0/C1 ALOS 修复和随后的 screen 裁决。

2026-11-11 追加审计发现：状态机虽然存在，但 `StageGate.stratified_default_success_min`
所需的分层通过率从未由 evaluator 填充，导致“总体已过阈值、仍永远不晋级”。该接线已
修复，并规定缺分层时显式报错；旧 Goal144 `S_s86201` 课程日志判 `INVALID`，新跑使用
`S_s86201_gatefix1`。因此“课程机制已实现”不等于“课程已严格跑通”：目前仍没有任何
实验臂完成连续两次过 Gate 后的真实晋级。

| 项目 | 冻结值 |
|---|---|
| 优先级 | Goal139 已完成；完整 C-line 现为当前 `P1`；Goal140 为后续 `P2`，不得插队 |
| 启动条件 | 已满足：Goal135 已裁决 `AUTHORITY_SAC_HARM`，Goal136 已完成 60 SAC + 90 PPO dev40 评估 |
| 第一执行块 | Phase A D-line：`N0 / D1 / D2`，先 SAC，PPO 独立同协议跟进 |
| 第二执行块 | Phase A OOD 测评：`current20 / combined20 / geom_hi / current_fast` |
| 第三执行块 | C-line 课程：`C-full / C-hard / C-no-current`，只在 D-line 可解释后启动 |
| 并行边界 | Goal135 的 authority 正则、PPO/SAC 改进继续进行；不得把两个 Goal 的样本合并成一个 Gate |
| 不改变 | reward、terminal、success、goal radius、dev/held-out split |

优先级提升不改变本文件已有的预注册 Gate。Goal136 的正式裁决已记录：
SAC `DIVERSITY_SAFETY_STOP`，PPO `PPO_RESIDUAL_COLLAPSE_NOT_DIVERSITY`；
不扩 8 seeds、不报 formal。该结果不能靠换 seed 或放宽阈值改写。

后续“神经网络方法是否能在不同压力测评下产生优势”仍保持高优先级主线：
Goal138 已检验 near-terminal sample density，Goal139 继续 mixed-horizon 与 PPO
credit horizon。PPO/SAC 的改进工作不因 Goal136 的一次 null 停止。

### 0.0 ALOS 洋流补偿范式冻结（2026-11-12）

后续所有 ALOS-based 训练/评价只登记两种洋流处理方式，不再把 crab 当成默认：

| 范式 | `compensate_current` | `current_crab_enabled` | 洋流来源 | 角色 |
|---|---:|---:|---|---|
| `ALOS-NONE` | `false` | `false` | 无 | 无补偿对照 |
| `ALOS-COMP` | `true` | `false` | 传感器估计 | 默认主基线 |

`ALOS-COMP` 只能使用 estimated current；true-current feedforward 仍禁止。历史
`ALOS-CRAB-EST` 结果（含 Goal173）保留原标签，不改写、不混表。canonical 定义见
`configs/lowlevel/eval/pc_standing_eval_suite_v1.json`，预注册见
`doc/goals/goal174_alos_current_compensation_paradigm_20261112.md`。

### 0.1 D-line Phase A1：先做可执行最小闭环
### 0.2 C-line 落实状态（2026-11-11 审计）

“课程学习”在本项目中包含两条不同证据线，不能混为一谈：

| 组件 | 状态 | 证据 / 缺口 |
|---|---|---|
| Goal136 Phase A1 D-line：N0/D1/D2 环境多样化 | `[x] 已完成` | 60 SAC + 90 PPO dev40；判 `DIVERSITY_SAFETY_STOP / PPO_RESIDUAL_COLLAPSE_NOT_DIVERSITY` |
| Goal138 T15/T10 近终端起点课程 | `[x] 已完成` | 6 SAC + 6 PPO 1M；判 `TERMINAL_CURRICULUM_TRADEOFF / TERMINAL_SAFETY_STOP / PPO_NO_TERMINAL_DENSITY_EFFECT` |
| Goal139 fixed full-path/T15 mixed-horizon | `[x] 已完成` | 6 SAC + 9 PPO 1M；判 `MIXED_SINGLE_SEED_ONLY / MIXED_SAFETY_STOP / PPO_NO_CREDIT_HORIZON_EFFECT` |
| S0：P0–P4 路径族与混合 path bank | `[x] 已完成` | `generate_curriculum_path` / `generate_mixed_path_bank` / `generate_pc_path_bank`；默认仍逐位走旧正弦 bank；长度、采样间距、曲率、混合比例、XY 投影无自交均有回归 |
| S1：`noise_scale` 与噪声课程 range | `[x] 接口已完成` | `SensorConfig.noise_scale` / `noise_scale_range` 已接入逐 episode 参数；默认 `1.0` 不额外消耗 RNG，scale 只乘零均值噪声。S3 的 manifest 记录仍待完成 |
| S2：actor current encoder / alpha ramp | `[x] 已完成` | PPO/SAC current encoder 零初始化；`alpha=0` 时 current 输入无作用；PPO 关闭 encoder 时 history side channel 置零；训练期按累计环境 transitions 线性 ramp，PPO 与 SAC 均已接通并写 manifest/iteration record |
| S3：stage sampler、promotion/rollback 状态机 | `[x] 已实现；分层 Gate 与 per-stage 预算已修复` | PPO/SAC 均已在训练循环中按 interval 调 `CurriculumTrainerHook`；smoke 已生成 `curriculum_checkpoints/`、`curriculum_evals/`、`curriculum_decisions.jsonl`、`curriculum_state.json` 与逐 episode manifest。C0/C1 direct-command 已按 per-row stage mask 切换，C4–C6 current alpha 已按 stage 进入点 ramp。2026-11-11 修复 `stratified_default_success` 数据通路与回滚后 per-stage 预算；gatefix2 实测 C0 晋级、C1 回滚、C0 anchor 到 600k 才 hard-cap |
| S4：冻结 C-line 配置与 1M pilot | `[~] pilot 已启动；C0 控制语义 blocker + Gate 接线 blocker` | 6 个 1M pilot 已启动。PPO `C-full` 在 C0 的 100k/200k/300k dev20 均为 `0.10/0.10`、plane miss `0.90`；100k `CONTINUE`，200k/300k `CURRICULUM_STALLED`。同一 100k checkpoint 的 `zero-residual` 在 direct command 下 `0/20`，关闭 direct、保留 ALOS 后 `18/20`。Goal144 旧 `S_s86201` 100k dev40 已达 `0.925` 但分层 Gate 未接线，日志判 `INVALID`；修复后用 `S_s86201_gatefix1` 重跑 |
| S5–S8：screen、8 seeds、formal、结果裁决 | `[ ] 未开始` | 必须在 S0–S4 完成后按预注册执行 |

因此当前只能声明：

```text
已严格落实 = D-line 环境多样化 + Goal138 近终端起点课程 + Goal139 静态混合诊断
             + S0 路径族 + S1 noise_scale + S2 actor current encoder/alpha ramp
             + S3 PPO/SAC trainer hook + 每 100k frozen-dev eval + promotion/rollback
               + stage-mix ramp + C0/C1 per-row direct-command
               + C4–C6 stage-aware current alpha（机制级，不是正式效果证据）
已启动未裁决 = S4 的 6 个 1M pilot（C-full/C-hard/C-no-current × PPO/SAC）
已阻断 = C0/C1 随机 direct command 与冻结 BC 先验不兼容；C-full 无法离开 C0
       → 预注册 Goal143 的 C0/C1 ALOS 控制修复
未开始 = S5–S8 screen/formal/裁决
```

下一步顺序固定为：保留现有 pilot 作为失败证据；先完成 Goal143 的 C0/C1 ALOS
控制修复和 1M/dev40 Gate，再回到 C-line screen。C-line 裁决前不启动 Goal140，
也不把 D-line、Goal138 或失败的 direct-command pilot 写成“完整课程学习已经完成”。

### 0.3 C0/C1 direct-command blocker（2026-11-11）

证据与反事实：

| 臂 | 控制语义 | default success | plane miss |
|---|---|---:|---:|
| `zero-residual` | C0 随机 direct command | `0/20` | `1.00` |
| `zero-residual` | C0 ALOS（关闭 direct） | `18/20` | `0.10` |
| `policy` | C0 随机 direct command（100k checkpoint） | `2/20` | `0.90` |

C0/C1 的 direct command 会在路径切线附近随机采样 heading `±30°/±60°`、
pitch `±5°`、speed `1.0–1.5 m/s`。冻结 BC checkpoint 的
`actor_current_encoder=encoded`、`path_encoder_alpha=1.0`，训练基座是 ALOS
path-tracking；因此 direct command 是 BC 先验的 OOD 输入。失败发生在动作/命令
参数化层，不是 PPO/SAC 优化器、current encoder alpha、reward 或 terminal。

裁决：不修改本文件已冻结的 C0 Gate，不换 seed、不加预算；预注册
`doc/goals/goal143_curriculum_alos_control_repair_20261111.md`，只把 C0/C1
控制语义改为 ALOS，保留 C0–C6 的深度、路径、噪声、洋流和 stage mix。
详细 blocker：
`research/blockers/2026-11-11-goal136-c0-direct-command-bc-mismatch.md`。

截至 2026-11-11 17:53 的 pilot 状态：课程状态机已真实执行
`CONTINUE / ROLLBACK / STAGE_BUDGET_EXHAUSTED / CURRICULUM_STALLED`，但没有
任何一条臂完成过“连续两次通过 Gate 后晋级”。`C-full` 与
`C-no-current_retry1` 都停在 C0（default/strict `0.10`、plane miss `0.90`）；
`C-hard` 的配置 `initial_stage=C6`，只观察到 C6→C5→C4→C3 的连续回退。
因此本轮 6 个 pilot 只能证明课程接线、回退和停滞裁决有效，不能写成课程收益。


完整 `S0–S4` 需要新增 P0–P4 路径族、current encoder 和课程状态机；S1 的
`noise_scale` 接口已完成，但尚未混入旧 Gate。为了尽快回答“环境多样化是否有用”，
Phase A1 继续只使用当前 PC batch env 已经支持的轴，不把尚未整链验证的接口混进
既有 Gate：

| 项目 | N0 | D1 | D2 |
|---|---|---|---|
| 训练基座 | Goal134 nominal strict，与 Goal126 相同 | 同 N0 | 同 N0 |
| 算法 | SAC | SAC | SAC |
| authority 正则 | 关闭 | 关闭 | 关闭 |
| 训练 seed | `86201–86203` | `86201–86203` | `86201–86203` |
| 预算 | `1M` / seed | `1M` / seed | `1M` / seed |
| path bank | `32` | `48` | `64` |
| 基础横向/深度幅值 | `1.5/0.4 m` | `1.7/0.5 m` | `2.0/0.6 m` |
| 洋流速度 | `0.25–0.45 m/s` | `0.25–0.55 m/s` | `0.25–0.675 m/s` |
| 洋流周期 / 幅值 | `8 s / 0.1 m/s` | `10 s / 0.15 m/s` | `6 s / 0.2 m/s` |
| 剪切倍率 | `1.0×` | `1.25×` | `1.5×` |
| plant 五通道 | `1.0` | `0.9–1.1` | `0.8–1.2` |
| 起始 xy/z/yaw 抖动 | `0.5/0.5/0.087` | `0.75/0.75/0.12` | `1.0/1.0/0.15` |

Phase A1 明确**不改变**：command speed（固定 `1.2 m/s`）、path length（固定 `35 m`）、
reward、terminal、success、goal radius、sensor profile 和 held-out。`command_speed_range`
与 sheared current 的组合、`noise_scale`、`P0–P4` 路径族、深度/长度课程都标记为
`DEFERRED_AFTER_A1`，等 A1 结果证明 diversity 有信号后再实现。

Phase A1 的采样口径固定为 **joint-wide**：D1/D2 每局从登记的宽分布联合采样，不额外
插入 nominal episode。这样做的原因是当前 batch env 没有 per-episode anchor 开关；
若声称“30% nominal anchor”而代码实际是整局宽分布，会把 Gate 的解释力污染。uniform
联合采样仍覆盖 nominal 区间（例如 D1 海流 `0.25–0.55 m/s` 中 `0.25–0.45 m/s` 占
`2/3`，plant 与起始抖动也以 nominal 为中心），但它不是严格的 30% anchor。

显式 anchor 混采登记为 `DEFERRED_A2_AFTER_A1`：只有 A1 出现
`DIVERSITY_TRADEOFF`，或 stress 有信号但 nominal default 掉超过 `1/40` 时才实现。
A2 必须使用同一批训练 seed、同一 1M 预算，并单独预注册，不能拿 A2 的结果替换 A1。

Phase A1 评估固定为：

```text
D1/D2 × 3 seeds × 5 levels × 2 scenarios × 40 dev seeds = 60 runs
levels = nominal, current15, current20, plant20, combined20
dev seeds = 20–59
```

N0 复用 Goal126 `86201–86203` 的 Goal134 同协议结果；zero 复用 Goal134 结果。所有
比较按训练 seed 配对，不重训 N0，不打开 held-out。

Phase A1 Gate：

| 观察 | 判读 |
|---|---|
| `>=2/3` training seeds 在至少两个 level 上 strict `>= N0 +2`，且 default `>= N0` | `DIVERSITY_SIGNAL`，扩 8 seeds |
| 有上述 stress 增益，但 nominal default 比 N0 低超过 1/40，或 path RMSE > `1.05×N0` | `DIVERSITY_TRADEOFF`，不扩 formal |
| 只有 `1/3` seed 达标 | `SINGLE_SEED_ONLY` |
| timeout/safety 比同 level N0 多超过 `2/120` episodes | `DIVERSITY_SAFETY_STOP` |
| 有 success 信号但 residual authority > `1.5×N0` | `AUTHORITY_STOP`，转 Goal135 |
| 没有 seed 在至少两个 level 达标 | `NO_DIVERSITY_EFFECT` |

PPO 的 D1/D2 使用同一环境配置和评测矩阵，但单独训练、单独判断，不能与 SAC 合并。

## Research Contract

```yaml
autoresearch:
  required: true
  triggers:
    - gate_fail_twice
    - no_metric_movement_after_3_fixes
    - heldout_collapse
    - seed_noise_ambiguous
  ledger: research/LEDGER.jsonl
  blockers_dir: research/blockers/
  reading_dir: research/paper_reading/
  max_outer_loops: 3
  check_command: python3 ../scripts/check_research_ledger.py --goal Goal136 --goal-doc doc/goals/goal136_environment_diversity_training_eval_plan_20261111.md --strict
```

## 1. 目的与边界

本计划回答两个严格分开的问题：

1. **D-line**：训练分布不够丰富，是否解释了 SAC 在压力下的单 seed 优势和跨 seed harm？
2. **C-line**：在最终训练分布相同的条件下，按深度、路径、噪声、洋流分阶段晋级，是否比直接训练最终混合分布更容易得到跨 seed 的稳定收益？

D-line 继续使用已有的 `D1/D2` 一次性混合分布；C-line 是新增的阶段式课程，不是把
`D2` 改个名字。两条线必须分开训练、分开报告，不能把结果合并成一个 Gate。

边界：

- 不追求“把环境调到神经网络赢”；训练分布和测评档位测量前冻结。
- 不训练在 `current20/combined20` 这些最终 OOD 档位上，避免测试泄漏。
- 不在同一 screen 里同时改 diversity 和 authority；authority 只作为后续交互臂。
- PPO 与 SAC 分开训练、分开 judging、分开报告；不能把两种算法的结果合并成一个 Gate。
- success 主口径仍是 `p_nav tight`，`eta` 只作诊断；`success_frame=both` 表示两套都记录。
- 课程只改变训练分布、采样顺序和可观测性，不改变 reward、terminal、success 或成功半径。
- 课程目标是论文证据，不是把公开测试集调成最好看的结果；晋级阈值和测评 split 在训练前冻结。
- **500 m 只作为仿真外推/OOD 档位**：当前 `plant.bounds_m` 的 z 范围是 `0–200 m`，仓库里也没有 REMUS100 的耐压/壳体深度模型。C6 的 `400–500 m` 必须在仿真里显式扩展 bounds，不能据此宣称真实平台可下潜 500 m。
- 深度课程表示“episode 起始深度/路径所在深度”，不表示 AUV 必须在 50 s 内从水面下潜到 500 m；若要做真实长距离下潜，必须另立长时域 Goal。
- actor 只能读取可部署的 `current_estimate_body`，不得读取 `current_ned_mps` 或 oracle 真值；oracle current 仅允许进入 critic、分析和诊断。

## 2. 训练环境计划

### 2.1 固定基座

| 项目 | 固定值 |
|---|---|
| 训练基座配置 | Goal125 `B15` / Goal126 SAC P-A 配置 |
| 载体与控制 | REMUS100，3D ALOS + PD/PI，50 Hz，`dt=0.02 s` |
| 训练 success radius | `1.25 m`，保持终端压力 |
| 最大步数 | `2500`，约 `50 s` |
| actor 架构 | SAC 用现有 `z_t64 + zero-DADE64 + command3`；PPO 用现有 PC PPO actor；不加 DADE |
| reward / terminal | 与 Goal126 完全一致，不因 diversity 改 reward |
| residual authority | Phase A 固定 Goal126 设置；Phase B 才与 Goal135 的 rho 组合 |
| held-out | `100–149` 保持关闭 |

### 2.2 训练随机化轴

`D1` 是温和混合分布，`D2` 是更宽但仍可训练的混合分布。**Phase A1 只做 joint-wide
联合采样，不插入显式 nominal anchor**；显式 anchor 版本登记为 A2，见 §0.1。原因是
当前实现没有 per-episode nominal/wide 两套参数切换，先把可执行、可审计的 A1 跑清楚，
再根据 nominal 退化证据决定是否值得增加 anchor 采样器。

| 轴 | nominal | D1 温和范围 | D2 宽范围 | 主要检验 |
|---|---:|---:|---:|---|
| 路径长度 | `35 m` | `35–45 m` | `35–50 m` | 累计漂移与长时信用分配 |
| 巡航速度 | `1.2 m/s` | `1.05–1.30 m/s` | `1.00–1.40 m/s` | 终端速度与 authority 需求 |
| 初始深度 | `10 m` | `9–11 m` | `8–12 m` | 深度泛化 |
| 深度幅值 | `0.4 m` | `0.2–0.6 m` | `0.1–0.8 m` | 垂直跟踪与终端几何 |
| 横向幅值 | `1.5 m` | `1.0–2.2 m` | `0.8–2.8 m` | 曲率与横向跟踪 |
| 起始 xy 抖动 | `0.5 m` | `0.5–0.75 m` | `0.5–1.0 m` | 初始状态鲁棒性 |
| 起始 z 抖动 | `0.5 m` | `0.5–0.75 m` | `0.5–1.0 m` | 深度初始误差 |
| 起始 yaw 抖动 | `0.087 rad` | `0.087–0.12 rad` | `0.087–0.15 rad` | 航向初始误差 |
| 海流速度 | `0.25–0.45 m/s` | `0.25–0.55 m/s` | `0.25–0.675 m/s` | 中等到 1.5× 海流 |
| 海流周期 | `8 s` | `5–10 s` | `3–12 s` | 变化频率/记忆需求 |
| 海流速度波动 | `0.1 m/s` | `0.05–0.15 m/s` | `0.03–0.20 m/s` | 非定常扰动 |
| 剪切强度 | 基线 `1.0×` | `0.75–1.25×` | `0.5–1.5×` | 空间非均匀扰动 |
| plant 五通道 | 全 `1.0` | `0.9–1.1` | `0.85–1.15` | 温和 plant mismatch |
| path bank | `32` | `64` | `128` | 路径几何覆盖 |
| 传感器噪声 | `unbiased_noise_v1` 固定 | 暂不随机 | Phase 2 才能加 scale | 观测鲁棒性 |

说明：

- 传感器噪声 scale 当前没有独立配置字段；不要把“噪声更大”写进第一批训练，除非先实现并验证无偏噪声缩放。
- 海流最大值 `0.675 m/s` 只到 Goal134 的 `current15`，不把 `current20` 作为训练档。
- 路径长度先不拉到 `>50 m`，否则在 50 秒上限下会混入 timeout 机制。

### 2.3 训练臂与预算

| arm | 算法 | 训练分布 | 训练 seeds | screen 预算 | 是否进入 formal | 说明 |
|---|---|---|---:|---:|---|---|
| `N0-SAC` | SAC | 现有 nominal B15 | `86201–86208` | 已有 Goal126/134 结果 | 只作对照 | 不重训 |
| `D1-SAC` | SAC | nominal 70% + D1 温和 30% | `86201–86203` | `1M` / seed | 有方向才扩 8 seeds | 先验证 diversity 是否改变 seed 方差 |
| `D2-SAC` | SAC | nominal 30% + D2 宽分布 70% | `86201–86203` | `1M` / seed | 有方向才扩 8 seeds | 检验更宽分布是否更好或过难 |
| `D2+AR-SAC` | SAC | D2 + Goal135 选出的 rho | `86201–86203` | `1M` / seed | 仅 Phase B | 不同时改 diversity 和 authority |
| `N0-PPO` | PPO | 现有 nominal B15 | `86201–86203` | 复用 Goal125 B15 契约；本计划补 `1M` 三 seed | 只作对照 | 先确认 PPO 残差是否仍接近零 |
| `D1-PPO` | PPO | nominal 70% + D1 温和 30% | `86201–86203` | `1M` / seed | 有方向才扩 8 seeds | 检验多样性是否让 PPO 离开零残差 |
| `D2-PPO` | PPO | nominal 30% + D2 宽分布 70% | `86201–86203` | `1M` / seed | 有方向才扩 8 seeds | 检验更宽分布是否会让 PPO 过难 |

| 算法 | 1M screen 预算 | 5M formal 预算 | 并行设置 |
|---|---:|---:|---|
| SAC | `256` envs × `1,000,192` transitions；batch `256`；UTD `0.125` | 同配置扩到 `5M` | 8 张 GPU 按 run 分配 |
| PPO | `1000` envs × `25` steps × `40` iterations = `1M`；保留 Goal125 B15 的 4 epochs / 32 minibatches / scale 1.5 / bias `(0,-0.25,0)` | 同配置扩到 `5M` | 8 张 GPU 按 run 分配 |

规则：

- Phase A 在 SAC 内比较 `N0-SAC / D1-SAC / D2-SAC`，在 PPO 内比较 `N0-PPO / D1-PPO / D2-PPO`；两个算法分别回答“多样性本身是否有效”。
- Phase B 只在 Goal135 选出 rho 后运行 `D2+AR-SAC`，回答“多样性与 authority 正则是否互补”。PPO 不加入 AR 臂。
- 训练 seed 必须与 Goal126 对齐；不能因为某个 seed 结果不好而换 seed。
- screen 只扩到 8 seeds；5M formal 只在 screen 通过后启动。

### 2.4 C-line：课程学习臂

C-line 与上面的 `D1/D2` 分开登记。`D1/D2` 回答“把一根混合分布直接交给策略是否有用”；
C-line 回答“同一最终分布，如果按能力逐级解锁，是否比直接硬训更容易学”。因此至少要
保留两个容量匹配的对照臂：

| arm | 训练分布 | 训练顺序 | 作用 |
|---|---|---|---|
| `N0` | 现有 nominal B15 | 固定，不晋级 | 传统控制器上的零残差基准 |
| `C-full` | C0→C6，最终阶段混合同 `C-hard` | 按能力晋级 | 课程学习主臂 |
| `C-hard` | 直接训练 C6 的最终混合分布 | 不晋级 | 隔离“课程顺序”本身的价值 |
| `C-no-current` | 同 `C-full`，但 actor current encoder 全程关闭 | 同 `C-full` | 检验显式洋流观测是否真的有用 |
| `C-oracle-current` | 仅诊断，actor 接真值 current | 不进入主文 | actor 利用洋流的上界，不可部署 |

规则：

- `C-full` 与 `C-hard` 必须使用相同的最终分布和相同训练预算；否则“课程有效”会与“多训练了”混淆。
- `C-no-current` 不是“无洋流环境”，而是“物理洋流仍在、actor 不看 current estimate”；它用于回答显式观测是否有增益。
- `C-oracle-current` 只能用于诊断 actor 是否存在可利用信息，不能出现在部署结果或主结论中。
- C-line 的 screen 先做 SAC；PPO 只有在 SAC 的课程信号可解释后才单独复制同样的阶梯，不合并样本。

速度课程单独版本化：REMUS100 最大速度为 `2.5 m/s`。standing eval 的速度轴固定为
`0.9 / 1.2 / 1.4 / 1.8 / 2.2 / 2.5 m/s`；未来课程必须从 C0 的
`1.0–1.2 m/s` 逐级抬升，到 C6 覆盖 `0.9–2.5 m/s`（见
`configs/lowlevel/eval/pc_standing_eval_suite_v1.json` 的 `speed_curriculum`）。
本轮 Goal136/Goal143/Goal153 已冻结的 `speed_range_mps=(1.10–1.40)` 不得静默改写；
启用高速课程必须新开预注册 Goal，并与同预算的旧速度课程对照。强洋流叠加
`2.5 m/s` 时不可达/饱和必须作为结果报告，不能删除该档。
训练侧 `CurriculumStageSpec.speed_range_mps` 也由代码硬校验 `<=2.5 m/s`，新课程
即使误写 `2.6` 也会在构造阶段直接报错，不会静默采样到超包络速度。

速度课程 v2 的固定阶梯如下。它只允许由新的预注册 Goal 激活；当前代码中的
`default_curriculum_stages()` 仍是 Goal136/143/153 冻结的 `1.10–1.40 m/s` 版本，
本轮不得为了覆盖高速档而静默改写旧课程。

可执行入口已经独立加入 `high_speed_curriculum_stages()`：它只替换
`speed_range_mps`，深度、路径、噪声、洋流、终端和 Gate 全部沿用 v1；因此新
预注册 Goal 可以直接显式调用 v2，而 `default_curriculum_stages()` 和历史 Gate
保持冻结。两者的分离由单测锁定，避免未来把 `2.5 m/s` 端点无意混入旧课程。

| stage | 速度范围（m/s） | 说明 |
|---|---:|---|
| `C0` | `1.00–1.20` | 低速接通数据、奖励和终端语义 |
| `C1` | `1.10–1.40` | 小幅提速，保持路径跟踪稳定 |
| `C2` | `1.20–1.60` | 中速折线路径 |
| `C3` | `1.30–1.80` | 中高速与更强噪声/洋流组合 |
| `C4` | `1.40–2.00` | 高速、显式洋流观测 |
| `C5` | `1.50–2.25` | 接近包络上界，强洋流与剪切 |
| `C6` | `0.90–2.50` | 覆盖完整评测速度轴，含 `2.5 m/s` 上界 |

速度课程约束：

- 阶段上界必须单调不降；`C6` 必须同时覆盖 `0.9` 和 `2.5 m/s`，不能只保留高速档。
- **用户确认（2026-11-12）**：`2.5 m/s` 是 REMUS100 的绝对速度上界；`C6` 的
  `2.5` 端点只有在海流和速度波动均为零/可忽略时才可采样，不能与强流独立组合。
  当前 v1 训练仍保持 `1.2 m/s`（C1 `1.15–1.25`），不得静默改写。
- 任何阶段写入 `>2.5 m/s` 都在构造阶段报错；standing eval 的 `speed_profiles_mps`
  也不得包含 `>2.5` 的档位。
- `speed_range_mps` 表示期望**地速**。使用 `reference.command_speed_range_mps`
  或 domain cell 的配置会在 parse 阶段强制
  `v_cmd + v_current + v_variation <= 2.5`；v2 的 curriculum sampler 已按最坏情况
  `v_current + v_variation` 收缩电流项，逐 episode 显式实现同样的联合约束。因此
  C6 的 `0.90–2.50` 是评测轴包络，`2.5` 端点只允许与零/弱流组合，不能和
  `0.40–0.675 m/s` 强流独立采样。若要把 `2.5` 与强流组合，只能作为 OOD
  压力档并报告不可达/饱和。
- 高速课程必须与同预算的旧速度课程对照；不能把“课程顺序 + 速度提升”混成一个收益。
- 强洋流叠加 `2.5 m/s` 时，不可达或控制饱和必须作为正式结果报告，不能删除该档或换 seed。

### 2.5 课程阶梯：深度、路径、噪声、洋流

所有阶段继续使用 50 Hz、`dt=0.02 s`、`max_episode_steps=2500`、default 2.0 m 成功半径
和 strict 1.25 m 诊断半径。路径长度先保持在 35–50 m；深度阶段改变的是路径所在深度和
起始深度，不是强制 50 s 内完成大深度下潜。

| stage | 起始/路径深度 | 路径族 | 噪声档 | 物理洋流 | actor current | 速度/长度 | 目的 |
|---|---|---:|---:|---|---:|---:|---|
| `C0` | `10 ± 0.5 m` | `P0` 直线 | `N0` clean | `0.15–0.25 m/s` 常值，无剪切 | off | `1.2 m/s` / `35 m` | 接通数据、奖励和终端语义 |
| `C1` | `10–30 m` | `P0` 70% + `P1` 30% | `N1` `0.25–0.50×` | `0.20–0.35 m/s` 常值 | off | `1.15–1.25` / `35–40 m` | 学会小幅横向/深度跟踪 |
| `C2` | `30–80 m` | `P1` 50% + `P2` 50% | `N2` `0.50–1.00×` | `0.25–0.45 m/s` 常值 | off | `1.10–1.30` / `35–45 m` | 引入折线和更完整噪声 |
| `C3` | `80–150 m` | `P2` 50% + `P3` 50% | `N3` `1.00–2.00×` | `0.30–0.55 m/s` 常值 | off | `1.10–1.35` / `40–50 m` | 中等深水、不可见洋流下隐式适应 |
| `C4` | `150–250 m` | `P3` 60% + `P2` 40% | `N3` `1.00–2.00×` | `0.30–0.60 m/s` 常值 | on，`alpha≈0→1` | `1.10–1.35` / `40–50 m` | 首次显式读取 current estimate，先隔离常值洋流 |
| `C5` | `250–400 m` | `P3` 70% + `P4` 30% | `N4` `2.00–3.00×` | `0.40–0.675 m/s`，慢变化 `12–20 s`，轻度剪切 | on，真实估计延迟/噪声 | `1.10–1.40` / `40–50 m` | 学会对缓变洋流做记忆和预判 |
| `C6` | `400–500 m`（仅仿真） | `P4` 70% + `P3` 30% | `N4` `2.00–3.00×` | `0.40–0.675 m/s`，快变化 `4–8 s`，中度剪切 | on，估计退化/短暂失效 | `1.20–1.40` / `40–50 m` | 快速反复洋流和深水外推压力 |

速度边界：上表的 `1.20–1.40 m/s` 是 Goal136/143/153 冻结的 **v1 课程**，不是平台
速度上限。REMUS100 硬上限为 `2.5 m/s`；未来 v2 课程按 §2.4 从 C0 `1.00–1.20`
逐级抬升到 C6 `0.90–2.50 m/s`。v1 结果不得被静默改写成高速结论；启用 v2 必须
新开预注册 Goal，并与同预算的 v1 课程对照。强洋流叠加 `2.5 m/s` 时，不可达、
饱和和 timeout 都要报告，不能删除该档。

深度说明：

- `C0–C3` 的深度范围都在当前 z bounds `0–200 m` 内；`C4` 的 `150–250 m`、`C5` 的
  `250–400 m` 和 `C6` 的 `400–500 m` 需要显式修改仿真 `plant.bounds_m` 与 termination
  depth limits，且只能出现在 C-line 配置中。
- 500 m 不是“真实可部署课程”。在没有载体耐压模型、浮力随深度变化、压力传感器和真实
  深海动力学之前，`C6` 只能报成 `SIM_ONLY_OOD`。
- 若后续要研究从水面下潜到 500 m，需要单独增加 episode 时长、速度下限和长时域成功
  口径；本 Goal 不做这种隐式改动。

### 2.6 路径族定义

仓库当前 `path.py` 只有 `sinusoidal_3d_arclength` 及其正弦 bank；直线、折线和一般
不规则曲线需要新增生成器。课程路径必须保持 `length_m`、`sample_spacing_m` 和
`ReferencePath` 的弧长约定不变，否则 path RMSE 和 progress 不可比。

| 路径族 | 几何定义 | 约束 | 用途 |
|---|---|---|---|
| `P0` 直线 | 沿起始切向的等长直线，横向/深度振荡为 0 | 不改变起点、终点和长度 | C0，排除曲率混杂 |
| `P1` 缓正弦 | 横向幅值 `0.5–1.0 m`，`0.5–1.0` 周期；深度幅值 `0.1–0.2 m` | 曲率平滑，起始切向连续 | C1，低曲率跟踪 |
| `P2` 正弦/折线 | 正弦横向 `1.0–2.0 m`，`1–2` 周期；或 3–4 个航点、转角 `10–20°`、最短段 `8 m` | 折线不出现瞬时 90° 转向 | C2，曲率和不连续切向 |
| `P3` 有界不规则 | 5–7 个航点插值，转角 `≤30°`，曲率 `≤0.08 1/m`，最短段 `5 m` | 不自交、不倒退、终点可达 | C3–C5，一般非规则路径 |
| `P4` 高难不规则 | 6–8 个航点，交替转向，转角 `≤40°`，曲率 `≤0.10 1/m`，最短段 `4 m` | 仍满足控制可达性；越界直接判 INVALID | C6，压力路径 |

实现要点：

- 新增 `straight_3d`、`polyline_3d`、`bounded_curvature_3d` 三个 generator，并在
  `ReferenceConfig`/config parser 中显式记录 `generator` 或 `path_family`。
- 每个 episode 的 path family、waypoint seed、长度、曲率和最大转角写入 manifest；
  evaluator 只按 manifest 分层，不能根据结果反推路径类型。
- 路径 bank 需要支持“本 stage 混合多个 family”，而不是只对正弦参数做随机扰动。
- 生成器单测至少覆盖：零长度、重复点、非有限值、长度守恒、最大曲率/转角、确定性 seed。

### 2.7 噪声档与 current estimate 质量

当前 `unbiased_noise_v1` 只有 profile，没有独立的噪声 scale。课程需要先把“无偏噪声大小”
做成单独轴；不能通过换成 `realistic_stress` 同时引入 bias、延迟、outage，从而污染解释。

| 档位 | 零均值噪声 scale | 其他传感器项 | 用途 |
|---|---:|---|---|
| `N0` | `0×`（等价 clean） | 无 bias、无 delay、无 outage | C0 |
| `N1` | `0.25–0.50×` | 固定低延迟，不随机 | C1 |
| `N2` | `0.50–1.00×` | 固定 `unbiased_noise_v1` 的非噪声项 | C2 |
| `N3` | `1.00–2.00×` | 固定延迟，不叠加 outage | C3–C4 |
| `N4` | `2.00–3.00×` | 固定延迟；current estimate 另按右表退化 | C5–C6 |

实现要求：

1. 在 `SensorConfig` 增加 `noise_scale` 和课程用 `noise_scale_range`，默认 `1.0`；只乘零均值
   dvl/IMU/USBL/nav/current-estimate 噪声，不乘 bias、drift、scale、delay 或 outage。
2. `noise_scale=0` 必须与 `clean` 在允许误差内一致；同一 seed 的噪声方差随 scale 单调增加。
3. 噪声 scale 逐 episode 采样并写入 manifest；训练 manifest 与评估 manifest 都必须可追溯。

当前 `actor_current_encoder_alpha` 的校验要求严格大于 0；实现零初始化平滑 ramp 时必须
先允许 `alpha=0`（或定义等价的无贡献 epsilon），否则“从 C3 无 current 平滑进入 C4”无法
按本计划执行。

`actor current` 的质量单独按阶段退化，避免把“噪声轴”和“洋流观测轴”混在一起：

| stage | current estimate | 延迟/失效 | actor 可见性 |
|---|---|---|---|
| C0–C3 | 不接 actor current encoder | 估计器仍可给 guidance/critic 用 | `current_encoder=off` |
| C4 | 低噪声、无 dropout | 年龄 `≤0.2 s` | encoder 开启，零初始化后 ramp |
| C5 | 中等噪声 | 年龄 `0.2–0.5 s`，短暂 dropout `≤1%` | 真实估计，不允许 oracle |
| C6 | 高噪声、退化 | 年龄 `0.5–1.0 s`，dropout `≤5%` | 真实估计；失效时用 zero/mask，不喂真值 |

Goal17 已经在 BC 上看到“current encoder 单独开启常常有害，只有与 crab 同时开启才出现
交互收益，但没有净收益”。因此 C4–C6 必须保留 current-on/current-off 对照；如果显式
current 在常值/无洋流层反而伤害、且在快变洋流层没有补偿，就把该分支判为
`CURRENT_ENCODER_REJECTED`，不能把负结果包装成课程收益。

### 2.8 每阶段配比与平滑过渡

每个 stage 不只用当前难度；保留前一阶段和 C0 anchor，避免灾难遗忘；在进入下一阶段前
提前少量 preview，使分布变化连续。preview 不能用于本阶段晋级判定。

| stage | 当前 stage | 上一 stage | C0 nominal anchor | 下一 stage preview |
|---|---:|---:|---:|---:|
| C0 | 90% | 0% | 0% | 10% C1 |
| C1 | 70% | 20% C0 | 0% | 10% C2 |
| C2 | 70% | 15% C1 | 10% | 5% C3 |
| C3 | 70% | 15% C2 | 10% | 5% C4 |
| C4 | 70% | 15% C3 | 10% | 5% C5 |
| C5 | 70% | 15% C4 | 10% | 5% C6 |
| C6 | 80% | 15% C5 | 5% | 0% |

平滑规则：

- stage 切换不瞬时生效；在 `100k` environment transitions 内用线性或 cosine ramp 从
  旧配比过渡到新配比，且记录每个 episode 的实际 stage mix。
- preview 只在该 stage 完成至少 `50%` 的软预算后启用；若 preview 触发 safety/timeout
  护栏，立即关闭 preview，不降低主 stage 阈值。
- stage 之间先切一个主要轴（C4 才开启 current encoder），不同时把深度、噪声和洋流
  全部跳到最高档。
- 每个 episode 必须能复现固定的 `(stage, depth, path_family, noise_scale, current_level,
  current_estimator_level)`；不允许只在 batch 统计上匹配。

### 2.9 晋级、回退与早停

每个 stage 每 `100k` environment transitions 做一次冻结 dev 评估。晋级评估至少包含
20 个 dev seeds × 2 个 path variants × default/strict 两套口径，共至少 80 个 episode；
不能只用训练集的 running success 晋级。

| stage | default success ≥ | strict success ≥ | timeout ≤ | plane-miss ≤ | safety | 额外条件 |
|---|---:|---:|---:|---:|---:|---|
| C0 | `0.90` | `0.70` | `0.10` | `0.10` | `0` | 所有直线起点 bucket 均 ≥`0.80` default |
| C1 | `0.90` | `0.65` | `0.10` | `0.10` | `0` | P0/P1 各层都不低于总阈值 `-0.10` |
| C2 | `0.85` | `0.60` | `0.12` | `0.15` | `0` | P1/P2 各层都不低于 `0.55` default |
| C3 | `0.80` | `0.55` | `0.15` | `0.20` | `0` | 深度分层最低 default ≥`0.55` |
| C4 | `0.80` | `0.55` | `0.15` | `0.20` | `0` | current-on 相对 off 在常值层不下降 >`0.05` |
| C5 | `0.75` | `0.50` | `0.18` | `0.20` | `≤0.01` | 慢变/剪切层最低 default ≥`0.50` |
| C6 | `0.70` | `0.45` | `0.20` | `0.25` | `≤0.01` | 只用于 `SIM_ONLY_OOD`，不用于真实部署声明 |

晋级需要满足全部条件，并连续两次评估通过（hysteresis）；若只通过一次，继续本 stage，
不做“单次好运晋级”。上一 stage anchor 的 default success 相对晋级时下降不得超过
`0.05`，否则先处理遗忘。

| 观察 | 判读 | 动作 |
|---|---|---|
| 连续两次 gate 低于阈值 `-0.10` | 当前难度过高 | 回退到上一 stage，补 `200k` anchor；每个 stage 最多回退一次 |
| safety 或 timeout 连续两次超护栏 | 课程不安全 | 立即回退；不通过放宽阈值救结果 |
| 连续两次相对上次提升 `<0.01` 且仍低于阈值 | `CURRICULUM_STALLED` | 停止该 stage，保留上一 stage checkpoint；报告阻塞 |
| current encoder 在常值层退化 >`0.05`，快变层无补偿 | `CURRENT_ENCODER_REJECTED` | 关闭 current branch，继续后续 stage，不把负结果算成课程收益 |
| 到 stage 最大预算仍未过 gate | `STAGE_BUDGET_EXHAUSTED` | 不降低阈值、不换 seed；记录后停止或交人工裁决 |

预算按 environment transitions 计，SAC 与 PPO 分开执行；每条曲线只在自己的算法内比较。

| stage | soft budget | hard cap | 说明 |
|---|---:|---:|---|
| C0 | `0.25M` | `0.40M` | 先接通契约 |
| C1 | `0.35M` | `0.55M` | 低曲率 |
| C2 | `0.45M` | `0.65M` | 折线+噪声 |
| C3 | `0.55M` | `0.75M` | 不可见洋流 |
| C4 | `0.55M` | `0.75M` | 常值洋流+current encoder |
| C5 | `0.65M` | `0.90M` | 慢变洋流 |
| C6 | `0.70M` | `1.00M` | 快变洋流+500 m 仿真外推 |

总 hard cap 为 `5.0M`；晋级可提前结束，因此实际预算通常小于 5M。1M screen 只用于接通
和方向检查，必须按同一 stage 顺序压缩预算，不能把 1M pilot 的结果当作课程有效性的
正式证据。正式比较必须把 `C-full`、`C-hard` 和 `N0` 放在同一预算、同一训练 seed 集上。

最终 checkpoint 选择规则在训练前冻结：在最后三次 stage 评估中，先最大化最差分层
default success，再比较 strict success，最后比较 residual L2；不能取“最后一步”作为默认
选择，也不能看 formal 结果再回头挑 checkpoint。

## 3. 测评计划

### 3.1 测评档位

所有档位使用同一 reward、terminal 和 controller；只改变环境或成功半径。`default` 主指标固定 `2.0 m`，`strict` 次级指标固定 `1.25 m`。

| level | 路径长度 | 速度 | 深度幅值 | 横向幅值 | 海流 | plant | 定位 |
|---|---:|---:|---:|---:|---|---|---|
| `nominal` | `35 m` | `1.2 m/s` | `0.4 m` | `1.5 m` | 基线 `0.25–0.45 m/s`，8 s | `1.0` | 主基准/天花板检查 |
| `current15` | `35 m` | `1.2` | `0.4` | `1.5` | `1.5×` 速度/剪切/波动 | `1.0` | 已知可区分档 |
| `current20` | `35 m` | `1.2` | `0.4` | `1.5` | `2.0×` | `1.0` | OOD，不用于训练 |
| `plant20` | `35 m` | `1.2` | `0.4` | `1.5` | 基线 | `0.8–1.2` | 温和 plant OOD |
| `geom_mild` | `45 m` | `1.3` | `0.6` | `2.2` | 基线 | `1.0` | 几何外推 |
| `geom_hi` | `50 m` | `1.4` | `0.8` | `2.8` | 基线 | `1.0` | 几何上限/OOD 边缘 |
| `current_fast` | `35 m` | `1.2` | `0.4` | `1.5` | `current15` + 周期 `4 s` | `1.0` | 频率/记忆检验 |
| `combined20` | `35 m` | `1.2` | `0.4` | `1.5` | `2.0×` | `0.8–1.2` | 强压力 OOD |
| `noise_hi` | `35 m` | `1.2` | `0.4` | `1.5` | 基线 | `1.0` | 仅 Phase 2，需先实现无偏噪声 scale |

### 3.2 测评矩阵

`success` 主指标为 `p_nav tight`；同时记录 `eta tight ever/at_done`、timeout、plane-miss、safety、path RMSE、residual L2、action max 和 material saturation。

| 阶段 | policies | levels | scenarios | dev seeds | 用途 |
|---|---|---|---:|---:|---|
| Phase A screen - SAC | `zero / N0-SAC / D1-SAC / D2-SAC` | `nominal, current15, plant20, geom_mild` | `default, strict` | `20–39` | 先看 SAC 方向，不烧 formal |
| Phase A OOD - SAC | `zero / N0-SAC / D1-SAC / D2-SAC` | `current20, combined20, geom_hi, current_fast` | `default, strict` | `20–39` | 只看 SAC 泛化和崩溃风险 |
| Phase A screen - PPO | `zero / N0-PPO / D1-PPO / D2-PPO` | `nominal, current15, plant20, geom_mild` | `default, strict` | `20–39` | 先看 PPO 是否离开零残差 |
| Phase A OOD - PPO | `zero / N0-PPO / D1-PPO / D2-PPO` | `current20, combined20, geom_hi, current_fast` | `default, strict` | `20–39` | 只看 PPO 泛化和崩溃风险 |
| Phase A formal - SAC | 通过 screen 的 SAC arm | 同 screen + OOD | `default, strict` | `20–59` | 8 个 SAC 训练 seed、40 dev seeds |
| Phase A formal - PPO | 通过 screen 的 PPO arm | 同 screen + OOD | `default, strict` | `20–59` | 8 个 PPO 训练 seed、40 dev seeds |
| Phase B - SAC | `D2+AR-SAC` vs `N0-SAC`/`R0-SAC` | 同 formal | `default, strict` | `20–59` | 仅在 Goal135 rho 确定后 |

### 3.3 判读 Gate

| 检查 | 通过条件 | 失败含义 |
|---|---|---|
| Validity | nominal 下 zero 复现 `strict/default=24/33`；配置 SHA 固定；held-out 为 0 | `INVALID`，先修实现 |
| Stress 可区分 | 至少一个压力档 zero strict 下降 `>=2/40`，且不在地板以下 | 压力太弱/太强，先调档位 |
| Diversity 主效果 | `>=2/3` screen seeds 或 `>=4/8` formal seeds 在至少两个档位 strict `+2`，且 default 不掉 | diversity 不是主因 |
| 非单 seed | 优势不能只来自 86201；至少两个不同训练 seed 同向 | `SINGLE_SEED_ONLY` |
| 安全护栏 | timeout/safety 不恶化超过 `2 pp`；path RMSE 不恶化超过 `5%` | 停止该 arm |
| 动作护栏 | residual RMS 不超过 N0 的 `1.5×`；action max 不持续贴近 1.0 | authority 仍过大，转 Goal135 |
| 算法隔离 | PPO 与 SAC 的 Gate 分别在算法内计算；不能把两者样本合并 | 混合报告无效，按算法重算 |
| PPO 残差活性 | 至少一个档位的 gate-active residual L2 `>=1%`；否则判 `PPO_RESIDUAL_COLLAPSE` | 环境多样性不能解决 PPO 探索/信用分配 |
| OOD 泛化 | current20/combined20 允许下降，但不得出现 safety/timeout 爆发 | 过拟合压力分布 |

### 3.4 停止规则

| 观察 | 判读 | 下一步 |
|---|---|---|
| D1/D2 只在 86201 有优势 | `DIVERSITY_NOT_CAUSAL` | 停止 diversity 线，回到 authority/reward/terminal |
| D1/D2 多 seed 在 current15/plant20 改善，但 nominal 退化 | `DIVERSITY_TRADEOFF` | 不扩 formal；收紧随机化范围或加 nominal anchor |
| D1/D2 多 seed 在 nominal 与压力同时改善 | `DIVERSITY_SIGNAL` | 扩 8 seeds/5M formal，再决定是否加 authority 正则 |
| 只有加 AR 后才有跨 seed 优势 | `AUTHORITY_INTERACTION` | 进入 Phase B 独立 Goal，不把 diversity 单独写成收益 |
| PPO 在 D1/D2 下仍接近零残差 | `PPO_RESIDUAL_COLLAPSE_NOT_DIVERSITY` | 停止 PPO diversity 线，转 exploration/credit assignment/reward |
| 所有档位都无方向 | `NO_DIVERSITY_EFFECT` | 停止环境多样化假设，转 reward/terminal/observation |

### 3.5 C-line 专用测评矩阵

测评不能只重复训练分布。主轴采用“单轴阶梯”，每个 suite 只改变一个因素；交叉 suite
只放预先登记的角落场景，避免把全部笛卡尔积都跑一遍后选择性报告。所有 primary 指标仍是
default 2.0 m 的 `p_nav tight` success；strict 1.25 m 只作诊断，timeout/plane-miss/safety
单独报告。

| suite | levels | 采样要求 | 主要回答 |
|---|---|---|---|
| `E0_nominal` | 现有 nominal：35 m、1.2 m/s、深度 10 m、N2、基线洋流 | 20–39；formal 可扩到 20–59 | 是否牺牲 nominal/复现零残差 parity |
| `E1_depth` | `10, 30, 80, 150, 250, 400, 500 m` | 20–39 × P2/P3 各一层；500 m 单列 OOD | 深度泛化是否连续，哪里开始崩 |
| `E2_path` | `P0, P1, P2, P3, P4` | 深度 10 m、N2、基线洋流；20–39 | 路径曲率/折线/不规则是否真的改善 |
| `E3_noise` | `N0, N1, N2, N3, N4` | P2、深度 10 m、基线洋流；20–39 | 噪声 scale 的退化曲线是否平滑 |
| `E4_current` | 无洋流诊断、`0.3` 常值、`0.6` 常值、慢变、快变、剪切 | P2/P3、N2/N3；20–39 | 常值到反复变化洋流的退化与恢复 |
| `E5_current_obs` | actor current off / estimate / oracle | 常值、慢变、快变各一层；20–39 | 显式洋流观测是否有真实增益 |
| `E6_cross_corner` | 浅水+快变+高噪声+不规则；150 m+快变；250 m+快变；500 m+快变 | 只跑预注册 corner，不扩全乘积 | 轴间交互和致命组合 |
| `E7_stress_long` | 45/50 m 路径、1.3/1.4 m/s、50 s 边界 | 只作 OOD，不混入 formal 主表 | 长路径、速度与 timeout 风险 |

补充规则：

- `E1_depth` 的 `400/500 m` 和 `E6` 的深水角落只作为 `SIM_ONLY_OOD`；结果不能写成
  真实海试或真实耐压能力。
- `E4_current` 的“无洋流”是诊断层，不是最终部署分布；如果某个策略只在无洋流下好，
  不能据此通过课程 Gate。
- `E5_current_obs` 的 oracle-current 只用于上界诊断；oracle 优于 estimate 说明观测有信息，
  不说明 estimate 足够好，也不说明可部署。
- 所有 evaluator 必须从同一份冻结 manifest 读取 stage、path family、noise scale、
  current level 和 estimator level；禁止根据 policy 输出反推测试难度。

### 3.6 测评指标与论文判据

每个 episode 至少记录以下表；聚合时同时给出总体、按 depth/path/noise/current 分层的结果。

| 类别 | 指标 | 用途 |
|---|---|---|
| 主结果 | `p_nav` tight success（default 2.0 m）、episode 数、Wilson 95% CI | 论文主指标 |
| 诊断成功 | `eta` tight ever/at_done、success margin、final goal distance | 区分真实到达与评估 frame 差异 |
| 失败模式 | timeout、plane-miss、depth-limit、safety、NaN/flew-out | 判断是“更难”还是“坏了” |
| 跟踪 | path 3D/horizontal/vertical RMSE、progress、heading/pitch tracking | 解释成功或失败发生在哪一层 |
| 终端 | 进入 10/5/2 m 的时间、入口角、终端速度、终端 residual | 解释 terminal capture |
| 控制 | residual L2、action max、material saturation、rate clipping | 防止 authority 爆炸 |
| 洋流观测 | estimate age/valid、estimate error、actor current sensitivity、mask/shuffle 消融 | 证明 actor 是否真的用了 current |
| 统计 | paired Δsuccess、bootstrap 95% CI、win/loss、跨 seed 方向 | 防止单 seed/单 episode 结论 |

课程论文 Gate（SAC 与 PPO 各自单独计算）：

| 检查 | 通过条件 | 失败判读 |
|---|---|---|
| `CURRICULUM_MAIN` | `C-full` 相对 `N0` 在至少两个 depth/path/noise/current 分层上 default `+2/40` 或 strict `+2/40`，且 nominal default 不下降超过 `2 pp` | 课程不是主因 |
| `CURRICULUM_ORDER` | 同预算下 `C-full` 相对 `C-hard` 在至少两个压力分层同向改善 | 收益来自更多数据/更宽分布，不是顺序 |
| `MULTI_SEED` | formal 至少 `4/8` 个训练 seed 同向；不能只由 86201 提供优势 | `SINGLE_SEED_ONLY` |
| `CURRENT_USE` | 快变洋流层 current-estimate 优于 current-off，且常值层不退化 >`2/40` | `CURRENT_ENCODER_REJECTED` |
| `DEPTH_TRANSFER` | 深度阶梯上不是单点偶然；至少两个相邻深度层同向，500 m 单列 OOD | 深度课程不成立 |
| `NOISE_MONOTONIC` | 随 noise scale 增加，success 可下降；若 `C-full` 在高噪声层明显优于 `N0` 且不出现 safety 爆发，记为鲁棒性收益 | 噪声课程无效或过载 |
| `SAFETY` | timeout/safety 不恶化超过 `2 pp`；path RMSE 不恶化超过 `5%` | 停止该 arm，不降阈值救结果 |
| `CONTRACT` | reward/terminal/success/radius/SHA 与冻结配置一致；held-out `100–149` 为 0 次使用 | `INVALID`，先修复 |

报告顺序固定为：`E0` 有效性 → `E1–E4` 单轴 → `E5` 洋流观测消融 → `E6/E7` OOD →
`C-full` vs `C-hard` 课程顺序 → 跨 seed 统计。任何 `INVALID` 不能写成 `FAIL`，任何
`SIM_ONLY_OOD` 不能写成真实闭环结论。

## 4. 运行前置与执行顺序

### 4.1 冻结与基线

- [ ] 冻结本文件、Goal125/126/134 的 reward/terminal/success/radius 和 config SHA；课程只新增分布轴。
- [ ] 确认 `plant.bounds_m` 的当前 z 上界；只有 C-line 的 C4/C5/C6 配置可以分别扩展到 250/400/500 m，`N0` 与所有 nominal 配置不得改。
- [ ] 确认 held-out `100–149` 关闭；课程晋级只读 `20–39`，formal 如需扩展也只到 `20–59`。
- [ ] 先复现 `N0` 的 nominal strict/default parity；不复现就判 `INVALID`，不启动课程。

### 4.2 实现与单测

- [x] S0：在 `src/auv_lowlevel/path.py` 增加 `P0–P4` 生成器和混合 path bank，保留弧长/采样间距契约。
  - 验证：`python3 -m pytest tests/test_lowlevel_path.py tests/test_lowlevel_curriculum_paths.py -q`
  - 检查：长度守恒、最大曲率/转角、无自交、确定性 seed、重复点/零长度报错。
  - 2026-11-11 落实：`reference.path_family` 与 `reference.path_family_mix` 均可用；默认不配置时逐位复用旧正弦 bank；P2/P3/P4 的 XY 投影自交检查覆盖 20 seeds。
- [x] S1：在 `src/auv_lowlevel/config.py`/`sensors.py` 增加 `noise_scale` 与课程 range；只缩放零均值噪声。
  - 验证：`PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_lowlevel_curriculum_noise.py tests/test_pc_goal85_unbiased_contract.py`
  - 检查：`noise_scale=0` 等价 clean；同 seed 方差随 scale 单调；bias/delay/outage 不被误缩放。
  - 2026-11-11 落实：固定 `noise_scale` 与逐 episode `noise_scale_range` 均可用；默认配置逐位保持旧噪声序列；`reset()` 与 `reset_rows()` 都可复现同一 seed 的 scale。
- [x] S2：接通 `actor_current_encoder_enabled`/`alpha`，只给 actor 喂 `current_estimate_body`；增加 zero-init 和 alpha ramp；禁止 oracle current 进入 actor。
  - 验证：`PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_lowlevel_curriculum_current.py tests/test_lowlevel_curriculum_schedule.py`
  - 检查：`current_contract='hidden'` 时真值不泄漏；alpha=0 时动作与关闭分支一致；mask/shuffle 消融可运行。
  - 2026-11-11 落实：PPO 使用 `pc.actor_current_encoder_alpha_start/_end/_ramp_transitions`；SAC 使用 `policy.current_encoder_alpha_start/_end/_ramp_transitions`。默认不配置 ramp，旧配置行为不变；PPO 在每次 update 后、SAC 在 callback 中按累计 transitions 更新 alpha，并写入 iteration record / manifest。
  - 2026-11-11 smoke：SAC `256` transitions（ramp `256`）最终 alpha=`1.0`；PPO `8` transitions（ramp `8`）最终 alpha=`1.0`，两侧 manifest 均记录 schedule 和 final alpha。
- [x] S3：在低层 PC env reset 路径增加 stage sampler、stage mix、promotion/rollback 状态机和 per-episode manifest。
  - 验证：`python3 -m pytest tests/test_lowlevel_curriculum_sampler.py tests/test_lowlevel_curriculum_manifest.py -q`
  - 检查：同 seed 可复现；每个 episode 记录 depth/path/noise/current/estimator/stage；promotion 不在训练 batch 统计上触发。
  - 2026-11-11 落实：`CurriculumStageSampler`、逐 episode manifest、`reset()/reset_indices()` 注入已完成；PPO 与 SAC 均已接入 `CurriculumTrainerHook`，每 N transitions 保存 checkpoint、跑 default/strict125 冻结 dev 评估、写 `curriculum_decisions.jsonl` / `curriculum_state.json`，并在 promotion/rollback 后切换或 ramp stage mix。
  - 2026-11-11 smoke：PPO `64` transitions（interval `32`）生成 `C0_000000064.pt`、default/strict125 summary 和 `CONTINUE` 决策；SAC `2048` transitions（interval `1024`）生成两个 `.zip` checkpoint、两组评估和 `CONTINUE -> CURRICULUM_STALLED` 决策。smoke 只验证接线，不是课程效果证据。
  - 2026-11-11 per-stage control：同一 batch 内 C0/C1 行按 per-row stage mask 使用 direct command，C2–C6 行走 ALOS；C4–C6 current encoder alpha 按 stage 进入点 ramp。回归 `tests/test_pc_goal136_curriculum_configs.py::test_mixed_curriculum_direct_commands_are_per_row` 与 `tests/test_lowlevel_curriculum_trainer.py::test_stage_aware_current_encoder_alpha` 通过。
- [~] S4：生成冻结配置和 1M pilot 配置；禁止手工改 stage 阈值。
  - 验证：`python3 scripts/build_goal136_curriculum_configs.py`
  - 检查：C0–C6 的 config SHA、bounds、success radius、reward/terminal 全部写入 manifest。
  - 2026-11-11 落实：已生成 `configs/lowlevel/goal136/curriculum/C0..C6.json`、`C-full/C-hard/C-no-current` 的 PPO/SAC pilot 配置和 `curriculum_manifest.json`。C4/C5/C6 的 z bounds 分别扩到 `260/410/510 m`，C0/C1 `hold_bonus=true`，其余 stage 为 false。6 个 1M pilot 已启动；C-full 在 C0 停滞，已判 `C0_DIRECT_COMMAND_OOD_FROZEN_BC` 并转入 Goal143。

### 4.2a D-line Phase A1 落地状态

- [x] 生成 N0/D1/D2 环境配置和 9 个 SAC training config：
  `python3 scripts/build_pc_goal136_dline_configs.py`
- [x] 路径 bank 支持 64 条宽几何路径；累计弧长容差固定为 `1 mm`，不改变既有路径数值。
- [x] 本地测试：`tests/test_pc_goal136_dline_configs.py`、`tests/test_pc_goal136_dline_analysis.py`、`tests/test_batch_path.py`。
- [x] 本地 2-env reset/step smoke：N0/D1/D2 均 finite，path bank/current/plant 随机化实际生效。
- [x] A1 采样口径冻结为 `joint_wide`；显式 nominal anchor 登记为 `DEFERRED_A2_AFTER_A1`，不在本轮配置中伪称已实现。
- [x] SAC 评估 runner：`scripts/run_goal136_dline_eval.sh`。
- [x] SAC 分析器：`scripts/analyze_pc_goal136_dline.py`。
- [x] SAC 训练 runner：`scripts/run_goal136_dline_sac_train.sh`（D1/D2 × 3 seeds）。
- [x] PPO 独立同协议配置：`scripts/build_pc_goal136_dline_ppo_configs.py`，N0/D1/D2 × 3 seeds，每 run `1000 env × 25 steps × 40 iterations = 1M`；基座为本地冻结的 Goal125 B15 配置，保留 frozen-BC、bias `(0,-0.25,0)`、scale `1.5`、4 epochs / 32 minibatches。
- [x] PPO 训练 runner：`scripts/run_goal136_dline_ppo_train.sh`；基座固定为 `artifacts/lowlevel/goal86/pa_residual_bc_path12_cmd3_current_20261110/pc_bc_actor.pt` 的 frozen-BC residual。
- [x] PPO 评估 runner：`scripts/run_goal136_dline_ppo_eval.sh`；N0-PPO 与 D1/D2 共用 5×2×40 矩阵。
- [x] PPO 分析器：`scripts/analyze_pc_goal136_dline_ppo.py`；只与同算法 N0-PPO 比较，不与 SAC 合并。
- [x] Goal135 完成后启动 D1/D2 六个 1M SAC screen，并同时启动 N0/D1/D2 九个 1M PPO screen。
- [x] 跑 60 个 SAC + 90 个 PPO 的 5-level × 2-scenario × 40-seed 评估；全部 finite。
- [x] 按 Phase A1 Gate 写 result/decision：SAC 判 `DIVERSITY_SAFETY_STOP`，PPO 判 `PPO_RESIDUAL_COLLAPSE_NOT_DIVERSITY`；不扩 8 seeds、不报 formal。

C-line 状态：`S0_S1_S2_S3_IMPLEMENTED_SMOKE_VERIFIED / S4_PILOTS_RUNNING / C0_DIRECT_COMMAND_OOD_FROZEN_BC / PIVOT_TO_GOAL143`。
S0 路径族、S1 噪声 scale、S2 current encoder/alpha ramp、S3 训练闭环与 per-stage control
均已完成并通过回归；S4 的 6 个 1M pilot 已启动。C-full 在 C0 停滞，反事实已证明
direct command 是主因；因此不再把“1M pilot 跑完”当作课程收益，而是先做 Goal143
的 C0/C1 ALOS 修复。A2 显式 anchor 也未触发。

2026-11-11 本轮回归：`119 passed`，覆盖 S0/S1/S2、Goal85/86/136/139 相关配置与路径契约：

```bash
cd settled_局部规划器_反应式
PYTHONPATH=src .venv/bin/python -m pytest -q \
  tests/test_lowlevel_curriculum_schedule.py \
  tests/test_lowlevel_curriculum_current.py \
  tests/test_lowlevel_curriculum_paths.py \
  tests/test_lowlevel_curriculum_noise.py \
  tests/test_pc_sac_features.py \
  tests/test_pc_goal86_ppo_config.py \
  tests/test_pc_goal86_ppo_path_prior.py \
  tests/test_pc_goal85_unbiased_contract.py \
  tests/test_pc_goal136_dline_configs.py \
  tests/test_pc_goal136_dline_analysis.py \
  tests/test_lowlevel_pc_sensors.py \
  tests/test_batch_path.py
```

启动前依赖检查：

- N0/zero 对照位于 `artifacts/lowlevel/goal134_frozen_sac_stress_20261111/<level>/<scenario>/<S_s86201..86203|zero>/episodes.json`。
- Goal136 评估必须与该目录同机运行；分析器默认 `--n0-root` 指向此路径。
- 若 Goal135 失败，先写 blocker/decision，再启动 D-line；不得用 D-line pilot 替代 Goal135 裁决。

### 4.3 Smoke、screen 与 formal

- [x] S4a：启动 6 个 1M pilot（`C-full / C-hard / C-no-current` × PPO/SAC）；只作接线与失败证据，不当正式课程收益。
- [x] S4b：修复 `C-no-current`：保留 frozen-BC current encoder 契约，强制 actor alpha=0；旧语义 PPO 已失败，SAC 已重启为 `retry1`。
- [x] S4c：C0 控制语义反事实：同一 checkpoint 的 direct command `0/20`，ALOS `18/20`；判 `C0_DIRECT_COMMAND_OOD_FROZEN_BC`。
- [x] S4d：执行 Goal143 `C-full-alos / C-no-current-alos` 1M pilot，dev40；两条均停在 C0 `0.85`，未过 Gate。反事实显示 frozen-BC 0.85、pure ALOS 0.925，判 `FROZEN_BC_RESIDUAL_BOTTLENECK`，转入 Goal144。
- [ ] S5：每个算法先跑 1 个 seed 的 1M pilot，只检查接通、episode 长度、timeout、finite 和 stage 晋级日志；不能把 pilot 当正式结果。
  - SAC：`python3 scripts/train_pc_command_residual_sac.py --config configs/lowlevel/goal136/C-full_sac_pilot.json --output artifacts/lowlevel/goal136_pilot/Cfull_s86201 --total-env-transitions 1000000 --device cuda --dtype float64`
  - PPO：`python3 scripts/train_batch_lowlevel_ppo.py --training-config configs/lowlevel/goal136/C-full_ppo_pilot.json --device cuda`
  - 评测：复用 `scripts/evaluate_pc_sac.py` / `scripts/evaluate_pc_ppo.py`，固定 `--seed-start 20 --seed-count 40`。
- [ ] S6：SAC screen 比较 `N0 / C-full / C-hard / C-no-current`；PPO 只有在 SAC 方向可解释后另跑，不合并样本。
  - 预算：1M pilot 只做方向；正式 screen 统一 5M 或明确记录容量匹配关系。
  - 验证：先跑 `default`，再跑 `strict`；先 `E0`，再 `E1–E5`，最后 `E6/E7`。
- [ ] S7：通过 screen 的 arm 扩到 8 个训练 seed、40 个 dev seed；每一步先检查 seed 方向，不因单 seed 不好换 seed。
- [ ] S8：写 result md、更新 `research/LEDGER.jsonl`、运行 checker；失败结果保留，不通过放宽阈值或换 seed 救结果。

### 4.4 运行与 Git 约束

- 训练、评估、分析与课程晋级必须各自产生 manifest；`artifacts/`、checkpoint、日志、缓存不提交。
- 每次实验只 add 本 Goal 新增/修改的文档、配置、脚本和测试；禁止 `git add -A`。
- 每次 commit 前运行 `git diff --cached --check`；发布前运行本文件 Research Contract 的 checker。
- 不自动 commit/push；需要提交时由人工确认后执行：

```bash
git add doc/goals/goal136_environment_diversity_training_eval_plan_20261111.md
git diff --cached --check
git commit -m "docs: add curriculum depth path noise current plan"
```

## 5. Git

只把本计划文档加入版本控制；不提交 `artifacts/`、checkpoint、日志或数据。

```bash
git add doc/goals/goal136_environment_diversity_training_eval_plan_20261111.md
git diff --cached --check
```

正式执行前运行：

```bash
python3 ../scripts/check_research_ledger.py \
  --goal Goal136 \
  --goal-doc doc/goals/goal136_environment_diversity_training_eval_plan_20261111.md \
  --strict
```
