# Goal145：终端捕获 / path 冲突与残差动作权限审计

**日期**：2026-11-11<br>
**状态**：`COMPLETED_EVAL_ONLY / T2_HORIZONTAL_PATH_TRADEOFF / C1_TERMINAL_VERTICAL_DOMINANCE / C1_RESIDUAL_NO_CLOSED_LOOP_EFFECT / HELDOUT_CLOSED`<br>
**上游**：`goal86_t2_3seed_result_20261111.md`、`goal144_result_20261111.md`、`goal140_vertical_tracking_reward_plan_20261111.md`<br>
**优先级**：`P1`。这是 Goal144 C1 瓶颈和 Goal86 T2 path-success 冲突的共同归因；在它给出裁决前，不启动新的奖励训练，也不换算法。<br>
**性质**：只读复用已有 checkpoint 和 dev40 评估，补 eval-only trace。不训练、不改 reward/terminal/success/goal radius、不扫 seed、不打开 held-out。

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
  check_command: .venv/bin/python ../scripts/check_research_ledger.py --goal Goal145 --goal-doc doc/goals/goal145_terminal_capture_path_vertical_audit_20261111.md --strict --require-result
```

`required: true`：本 Goal 是探索型归因。若 trace 不能解释 path-success 冲突，必须写 blocker、至少两个可证伪假设、reading note 和最便宜的下一实验；不得直接换 SAC/REDQ/DroQ。

## 1. 背景与现有证据

### 1.1 已固定的评价契约

- 成功、奖励和 termination 使用真值 `eta`；`p_nav` 只做部署诊断。
- 训练和 dev 评估关闭位置/延迟误差；保留噪声项与执行器动态。
- dev seeds `20–59`，每 run 40 episodes；held-out `100–149` 保持关闭。
- 当前不改变 `eta_tight_at_done`、goal radius、terminal capture 或 checkpoint 选择规则。

### 1.2 Goal86 T2：success 有弱正信号，但 path 退化

Goal86 T2-SAC 3 seeds：

| 指标 | policy | zero-residual | 结论 |
|---|---:|---:|---|
| default success | `28.67/40` | `27/40` | 方向正，pooled CI 跨 0 |
| strict125 success | `15.33/40` | `14/40` | 方向正，pooled CI 跨 0 |
| path 3D RMSE | 比 zero 高 `+0.277 m` | — | pooled CI 不含 0，不能进 formal |

T2-PPO 3 seeds 全部与 zero-residual 打平，说明弱 success 信号来自 SAC 的实际残差，不是动作接口本身。

### 1.3 Goal86 零残差基线的成功/失败几何

对 `t2_ppo_zero_dev40` 的 40 个 episode 只读分解：

| 分组 | n | path vertical RMSE | path horizontal RMSE | path 3D RMSE | final goal distance |
|---|---:|---:|---:|---:|---:|
| success | 27 | `0.8245 m` | `1.4773 m` | `1.7762 m` | `1.9837 m` |
| failure | 13 | `1.3952 m` | `3.1381 m` | `3.5036 m` | `2.5795 m` |

失败组的纵向误差更高，但横向误差的绝对值和增量都更大。仅凭这张表不能把失败归因于纵向，也不能据此启动 Goal140。

### 1.4 Goal144 C1：真正的瓶颈是末段捕获

- PPO command/actuator 两臂 C1 `default/strict = 0.65/0.65`、plane miss `0.35`、timeout/safety `0`。
- PPO command residual 约 `1e-4`，基本塌缩；actuator residual 非零，但 success/plane-miss 与 command 臂逐项相同。
- SAC command residual 主动破坏 C1；SAC actuator residual 只在 C0 有局部收益，C1 仍为 `0.625`。
- 因此 C1 不是“算法不够强”或“还没扫到好 seed”，而是 **动作权限、末段几何或 terminal capture 语义** 尚未被解释。

### 1.5 为什么现在不直接加纵向奖励

Goal140 的 `R1/R2` 只有在以下条件同时成立时才值得开新 reward arm：

1. 失败或 success flip 与纵向误差有稳定、配对的关系；
2. 纵向误差不是横向误差或 terminal geometry 的代理变量；
3. 该关系在至少 2/3 个训练 seed 上同向，且 paired CI 不跨 0；
4. 现有动作权限已经足以表达修正，而不是先被 saturation/rate limit/terminal gate 截断。

当前证据只满足“纵向误差在失败组更高”，不满足 1–4。故本 Goal 先做 trace 审计。

## 2. 可证伪假设

### H145-A：T2 的 path 退化主要由纵向误差主导

机制：SAC actuator residual 改变了 depth/heave 控制，导致 path vertical RMSE 上升；success 翻转来自末段纵向修正，而不是横向路径变好。

成立应看到：

- T2-SAC policy 相对 zero 的 `ΔMSE_vertical` 为正，且占 3D MSE 增量的 `>=50%`；
- 至少 2/3 个 seed 的 vertical RMSE 增量同向；
- success flip 的 episode 在末段 `vertical_error`、`heave_rate_error` 或 final vertical miss 上同步改善。

失败应看到：vertical 增量 CI 跨 0，或 horizontal 解释大部分 3D 增量。

最便宜证伪：复用 3 个 T2-SAC checkpoint，仅对 default/strict 各跑 40 episodes 的 eval-only trace；先算 paired `ΔMSE_vertical` 与 `ΔMSE_horizontal`，不训练。

### H145-B：path-success 冲突来自残差动作权限，而不是 reward

机制：SAC 在 near-goal 产生能翻转 success 的修正，但同一残差在全路径上被过度使用；command residual 在 C1 的方向错误，actuator residual 又受到限幅或通道耦合，导致“近端有效、全程伤 path”。

成立应看到：

- success flip 集中在 near-goal shell，且 flip episode 的 residual 通道/幅值与未 flip episode 有稳定差异；
- command residual 的 heading/pitch/speed 三通道中至少一个方向与 ALOS 修正相反；
- actuator residual 的 rudder/stern/rpm 中至少一个通道接近饱和或 rate clip；
- 仅做 eval-only 的通道 mask/scale/gate 单变量消融即可改变 success/path 交易方向。

失败应看到：无论 residual 通道如何限制，success flip 与 final geometry 都不变；此时应回到 reward/terminal capture 语义，而不是动作参数化。

最便宜证伪：固定 C1 checkpoint `C1_000300000`，对 command/actuator policy 与 zero-residual 各跑 40 episodes trace；只读 channel RMS、saturation、terminal distance、final miss vector；必要时再加一组 eval-only channel mask。

### H145-C：C1 plane miss 是末段几何/terminal gate 问题

机制：C1 的失败不是全程 tracking 差，而是进入 2 m 壳层后仍无法满足 capture sphere / plane entry；这与 Goal144 的 `goal_plane_miss=0.35` 一致。

成立应看到：

- 失败 episode 的 final goal distance 集中在 `1.25–2.5 m`，timeout/safety 为 0；
- 失败与成功的全程 path RMSE 差异较小，但 near-goal 的 final miss vector、closing speed 或 terminal phase 差异大；
- terminal gate 介入距离改变会改变 plane miss，而不需要改 reward。

失败应看到：失败 episode 在远离 goal 时已经严重偏离，或 final miss 分布与 success 大量重叠。

最便宜证伪：只读 C1 trace 的 goal distance、terminal phase、final miss vector 和 closing speed；不训练。

### H145-D：纵向误差不是独立信号

机制：纵向误差与横向误差、路径曲率、初始深度或 terminal geometry 共线；单独加纵向 reward 只会改变轨迹风格，不能提高 success。

成立应看到：控制横向误差、路径族和 terminal geometry 后，纵向误差与 success 的关联消失；或 Goal140 的 `R1/R2` pilot 没有 success 与纵向 RMSE 的同向改善。

失败应看到：纵向误差在控制其他变量后仍独立预测 failure。

最便宜证伪：在 trace 上用分层/配对分析，先把 episode 按 horizontal error 和 path family 分层，再看 vertical error 的判别力。

## 3. 数据、基线与隔离

| 项目 | 规定 |
|---|---|
| T2-SAC 训练 seeds | `86201–86203`，只复用 1M checkpoint |
| T2-PPO 对照 | `86201–86203`，已有 3-seed eval，不重训 |
| C1 checkpoint | `C1_000300000` 主审计；`C1_000400000` 仅复核是否随训练变化 |
| 评估 scenarios | default + strict125 |
| 评估 seeds | dev `20–59`，每 run 40 episodes |
| held-out | `100–149` 关闭 |
| 评估起点 | 已有协议，不改变 |
| success 契约 | Goal85 `eta_tight_at_done`，不改 radius/terminal |
| 训练预算 | `0` transitions；本 Goal 只做 eval/trace 和只读分析 |
| checkpoint 选择 | 固定上述 checkpoint；不按结果选 best checkpoint |

禁止：

- 用 trace 结果反向选择 seed 或 episode；
- 打开 held-out；
- 在同一 Goal 内改 reward、动作接口和算法；
- 以 single-seed 或单 episode 结论启动 Goal140；
- 把 `p_nav` 当作 success 或 reward 证据。

## 4. trace 与分析方案

### 4.1 需要补的 trace

1. **T2-SAC**：3 个 seed × `{policy, zero-residual}` × `{default, strict125}`，共 12 个 eval run；输出 `trace.npz`。
2. **C1-PPO command**：`C1_000300000` × `{policy, zero-command-residual}` × default，必要时补 strict；输出 `.npz`。
3. **C1-PPO actuator**：`C1_000300000` × `{policy, zero-actuator-residual}` × default，必要时补 strict；输出 `.npz`。
4. **C1 复核**：`C1_000400000` 只做 policy/default 的轻量 trace，确认不是单 checkpoint 偶然。

所有 trace 必须记录 `episode_seed`、`eta`、`nu`、`goal_ned`、`goal_distance`、`terminal_kind`、`p_nav`、`requested_action`、`base_action`、`residual_action`、`actual_actuator`、`guidance_terminal_phase`；SAC 额外记录 `path_vertical_error`、`path_horizontal_error`、`path_progress`。

### 4.2 指标定义

- `vertical_rmse`：trace 中 `path_vertical_error` 的均方根；按 episode 计算后再做 paired 统计。
- `horizontal_rmse`：trace 中 `path_horizontal_error` 的均方根。
- `3d_rmse`：`sqrt(vertical_rmse^2 + horizontal_rmse^2)` 的 episode 值。
- `ΔMSE_vertical = mean(v_policy^2 - v_zero^2)`；`ΔMSE_horizontal` 同理。
- `vertical_share = max(0, ΔMSE_vertical) / (max(0, ΔMSE_vertical) + max(0, ΔMSE_horizontal))`。
- `terminal_miss_vector = eta_final - goal_ned`；分别报告 N/E/D 分量和水平/垂直模长。
- `near_goal_shell`：final goal distance 在 `[1.25, 3.0] m`；`capture_shell`：`<1.25 m`。
- `residual_channel_rms`：rudder/stern/rpm 三通道；`channel_saturation_fraction` 和 `rate_clip_fraction` 必须逐通道报告。

### 4.3 配对统计

每个训练 seed 内，以 dev seed 配对 policy 与 zero-residual；报告：

- 每个 seed 的 success 翻转（policy-only / zero-only）；
- 每个 seed 的 vertical/horizontal/3D RMSE 差值；
- pooled mean、median、95% bootstrap CI；
- 只用有 success flip 的 episode 做条件分析，并单独报告没有 flip 的 episode，避免混入路径风格变化。

## 5. 执行计划

- [x] Step 0：冻结本文件、checkpoint 路径、dev seeds、scenario、trace 字段和 Gate；确认远端 GPU 空闲。
- [x] Step 1：只读已有 Goal86 T2/PPO episode 和 summary，生成不依赖 trace 的配对表。
- [x] Step 2：跑 T2-SAC policy/zero trace；先做 default，strict 作为复核。
- [x] Step 3：跑 C1 command/actuator policy/zero trace；分解 terminal miss vector、near-goal shell、residual 通道和 saturation。
- [x] Step 4：按预注册 Gate 裁决；T2 判 `HORIZONTAL_OR_MIXED_PATH_DEGRADATION`，C1 判 `C1_TERMINAL_VERTICAL_DOMINANCE` 且 `C1_RESIDUAL_NO_CLOSED_LOOP_EFFECT`。
- [x] Step 5：写 `goal145_result_20261111.md`、decision、`research/LEDGER.jsonl`，运行 checker。

## 6. 预注册 Gate

### 6.1 T2-SAC 纵向主导 Gate

| check | 阈值 |
|---|---:|
| vertical RMSE 增量方向 | 至少 2/3 个训练 seed 为正 |
| pooled `ΔMSE_vertical` 95% CI | 不跨 0 |
| vertical share of 3D MSE increase | `>=0.50` |
| success flip 条件关联 | flip episodes 的 vertical 改善与 flip 同向，且至少 2/3 seed 可复核 |
| 水平/纵向独立性 | 控制 horizontal RMSE 后，vertical 仍与 success/final geometry 相关 |

全部满足：`VERTICAL_DOMINANT_CONFIRMED`，允许新开 Goal140 `R1/R2` pilot；否则 `VERTICAL_NOT_CONFIRMED`。

### 6.2 C1 动作权限 Gate

| check | 阈值 |
|---|---:|
| C1 失败位置 | `goal_plane_miss` 为主，timeout/safety `<=1/40` |
| final miss 集中度 | `>=50%` 失败的 final distance 在 `1.25–2.5 m` |
| residual 通道 | 至少一个通道 RMS 非零且 saturation/rate clip 可测 |
| gate/通道单变量消融 | 只改一个 eval-only 参数即可改变 plane miss 或 path-success 交易 |

满足：`ACTION_AUTHORITY_CONFIRMED`，下一步做动作参数化/权限修复，不先改 reward。

### 6.3 其他裁决

| 观察 | 判读 | 下一步 |
|---|---|---|
| 横向 MSE 增量占主导 | `HORIZONTAL_DOMINANT` | 查 cross-track/heading/路径曲率；不启动 Goal140 |
| 纵向和横向都不可解释 | `GEOMETRY_OR_TERMINAL_UNRESOLVED` | 转 terminal capture/guidance trace，不换算法 |
| residual 很小且无闭环变化 | `ACTION_COLLAPSE` | 查动作尺度/初始化/信用分配；不把样本效率当唯一原因 |
| success/path 差异小于 seed 噪声 | `SEED_NOISE_AMBIGUOUS` | 写 blocker，扩 seed 只能在新预注册内做 |

## 7. Git

禁止 `git add -A`。只 add 本 Goal 相关文件，提交前运行：

```bash
.venv/bin/python ../scripts/check_research_ledger.py --goal Goal145 --goal-doc doc/goals/goal145_terminal_capture_path_vertical_audit_20261111.md --strict
git diff --cached --check
```

不自动 commit / push；trace、checkpoint、日志和 `artifacts/` 不提交。

## 8. Goal 提示词

```text
你现在执行 Goal145：审计 Goal86 T2 的 path-success 冲突和 Goal144 C1 的 terminal capture 瓶颈，
判断纵向误差是否真的主导，以及问题是否其实是残差动作权限/末段几何。

先读：
- doc/goals/goal145_terminal_capture_path_vertical_audit_20261111.md
- doc/goals/goal86_t2_3seed_result_20261111.md
- doc/goals/goal144_result_20261111.md
- doc/goals/goal140_vertical_tracking_reward_plan_20261111.md

硬约束：
1. 不训练、不换 PPO/SAC、不扫 seed、不打开 held-out；
2. 不改 reward、terminal、success、goal radius、checkpoint 选择规则；
3. 只用 dev seeds 20–59，固定 checkpoint 和 scenario；
4. 先做 T2-SAC 的 vertical/horizontal MSE 分解，再做 C1 terminal miss/residual channel 审计；
5. 只有预注册的 VERTICAL_DOMINANT_CONFIRMED 才允许新开 Goal140 R1/R2 pilot；
6. 若证据指向动作权限，先做 eval-only 单变量消融，不先改 reward；
7. 不自动 commit/push，产物只写 artifacts/，结论写 result md 和 ledger。

执行顺序：
- 先复用已有 summary/episode 生成配对表；
- 再补最少量的 eval-only trace；
- 最后按 Gate 裁决并写 result/blocker/ledger。
```

## 9. 自主迭代契约

本 Goal 遵循 `research/AUTORESEARCH_LOOP.md`：

1. 失败后先归因，不直接换算法或扫 seed；
2. 每次 blocker 至少写两个可证伪 hypothesis；
3. 需要外部方法时先做 paper-search / deep-research 并写 reading note；
4. 先跑最便宜的 falsification，再决定 `continue` / `pivot` / `stop`；
5. `pivot` 时写新的 Goal 文档和 prompt，不修改失败 Gate 的阈值；
6. 不自动 commit/push，不打开 held-out，不改变任务成功定义或主算法族。
