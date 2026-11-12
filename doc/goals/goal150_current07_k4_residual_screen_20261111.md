# Goal150：current07 K4 基座上的 PPO/SAC 残差 screen

**日期**：2026-11-11<br>
**状态**：`COMPLETED / SCREEN_FAIL / PPO_NO_GAIN / SAC_RESIDUAL_HARMFUL / VERTICAL_REWARD_NOT_TRIGGERED / HELDOUT_CLOSED`<br>
**结果**：`goal150_result_20261111.md`<br>
**上游**：`goal149_result_20261111.md`、`goal148_result_20261111.md`、`goal147_result_20261111.md`<br>
**优先级**：`P1`。Goal149 已找到 K4 的 90% 过渡带 `current07`，现在检验残差学习能否把
`36/40` 往上推，而不是继续换控制器或临时改 reward。
**性质**：K4 基座上的 PPO/SAC command-residual screen。1 seed、每臂 1M transitions；
current07 主评估，current05/current10 只做 transfer 诊断；不打开 held-out，不改
success / terminal / goal radius / eval split。

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
  check_command: .venv/bin/python ../scripts/check_research_ledger.py --goal Goal150 --goal-doc doc/goals/goal150_current07_k4_residual_screen_20261111.md --strict --require-result
```

`required: true`：本 Goal 会训练新的 PPO/SAC 残差臂。Gate 记录前不得改 reward、
terminal、success、goal radius、dev split、seed、预算或 checkpoint 选择规则。若
screen 失败，先按 Research Escalation 写 blocker 和至少两个可证伪 hypothesis，再决定
`continue` / `pivot` / `stop`。

## 1. 背景与证据

Goal147 证明 `base_controller.pitch_gain 2 -> 4`（K4）把 C1 nominal dev40 从
`26/40` 提到 `40/40`，且 policy 与 zero-command-residual 完全同分。Goal148 进一步
证明 K4 在 current20 附近先出现执行器饱和。Goal149 的边界阶梯为：

| level | mean current | K4 success | Z success | 判读 |
|---|---:|---:|---:|---|
| nominal | `0.276 m/s` | `40/40` | `26/40` | K4 nominal 已满 |
| current05 | `0.351 m/s` | `39/40` | `18/40` | 仍近满 |
| **current07** | **`0.389 m/s`** | **`36/40`** | `15/40` | **90% 过渡带** |
| current10 | `0.426 m/s` | `27/40` | `11/40` | 已越过 70% 下界 |
| current15 | `0.527 m/s` | `14/40` | `8/40` | 执行器饱和区 |

current07 的 K4 policy/zero 都是 `36/40`，command residual 平均绝对值约 `1.5e-4`。
因此 current07 同时满足两个条件：传统 K4 仍能完成大多数 episode，且保留了
`4/40` 的可改进空间。现在要检验的是“残差是否能把 K4 基座往上推”，不是重新证明
K4 本身。

## 2. 问题与可证伪假设

### H150-A：PPO command residual 能在 current07 超过 K4-zero

成立应看到：至少一个 PPO entropy arm 在 dev40 上比同配置 zero-command-residual
多成功 `>=2/40`，或 `eta_done_distance_m_mean` 改善 `>=0.02 m`，同时
timeout、safety、plane miss / path overshoot 不增加。

失败应看到：三档 entropy 的 success 都不超过 zero 或改善不足；此时判
`PPO_RESIDUAL_NO_SCREEN_GAIN`，不扩 seed、不扫 reward。

最便宜证伪：本 Goal 的 `ppo_ent1e4` 1M 单 seed；若连熵提高也没有动作变化，再检查
`ppo_ent1e3/1e2` 是否只是 residual 数值变大但没有闭环收益。

### H150-B：SAC command residual 能在 current07 超过 K4-zero

成立应看到：SAC 在 dev40 上达到与 H150-A 相同的主指标 Gate，且 residual 确实非零、
动作未出现新的饱和或 safety 回退。

失败应看到：SAC residual 接近零，或一旦非零就伤 success/path；前者是 exploration /
credit 问题，后者是动作参数化或 reward 问题，不能笼统归因于“样本效率不足”。

最便宜证伪：当前 1M、1 seed、256 envs 的 SAC screen；不先扩到 5M，也不先换算法。

### H150-C：current07 的剩余失败主要是纵向捕获，而不是横向路径

成立应看到：失败 episode 在 entry 前已有稳定的
`path_vertical_error` / final vertical error 分离，且残差方向与纵向误差一致但幅度
接近零，或幅度变大后仍未改善 terminal capture。

失败应看到：失败主要是横向、timeout、safety、执行器饱和或 path overshoot；此时
继续修动作参数化/terminal capture，不进入 reward pilot。

最便宜证伪：复用本 Goal 的 rich trace，按成功/失败 seed 比较纵向误差、横向误差、
pitch、执行器饱和和 residual；不训练新的 reward arm。

### H150-D：current05/current10 的迁移方向应与 current07 一致

成立应看到：current07 有收益的 arm 在 current05 不退步，在 current10 至少不出现
`>=3/40` 的 success 损失。

失败应看到：只在 current07 单点有效、迁移到相邻剂量就伤性能；此时不能写成
current 鲁棒性收益，只能保留为单场景 screen。

最便宜证伪：同一 checkpoint 在 current05/current10 各跑一次 dev40；不重新选
checkpoint，不用 transfer 结果反向调 reward。

## 3. 数据、初始化与隔离

| 项目 | 冻结值 |
|---|---|
| controller | K4，`base_controller.pitch_gain=4.0`，其余 controller 字段不变 |
| scenario | C1 + current `0.275–0.50 m/s` + shear scale `0.75` |
| environment | `configs/lowlevel/goal150/env_current07_k4.json` |
| training seed | `86201`，仅 1 seed screen |
| eval seeds | `20–59`，每个 run 40 episodes |
| held-out | `100–149`，关闭 |
| success | Goal85 `eta_tight_at_done` |
| goal radius | `1.25 m`，不改 |
| eval levels | current07 主评估；current05/current10 只做 transfer |
| PPO budget | `1000 envs × 25 steps × 40 iters = 1,000,000` transitions |
| SAC budget | `1,000,192` env transitions |
| checkpoint selection | 唯一 `final` checkpoint；不按 dev 结果挑 best checkpoint |
| actor observation | 现有 PC `z_t` / dual-horizon 接口；不泄露 current 真值 |

初始化说明：SAC profile 使用现有 `pc_z_t` 架构和 BC encoder/zero-residual 初始化；
PPO profile 使用现有 PC actor，不加载 SAC checkpoint。两者是**各自相对 K4-zero 的
残差 screen**，不把 PPO 与 SAC 的差异直接解释为算法优劣，因为初始化并不同构。

禁止：

- 用 current05/current10 的 transfer 结果反向选择 current07 checkpoint；
- 用 held-out 选 arm、threshold 或 reward；
- 在 Gate 记录前改 entropy、reward、terminal、success、radius 或 seed；
- 因为单 seed 没效果就追加 seed/预算救 Gate；
- 把 K4-zero 与 policy 的差异写成神经网络收益，若 residual 实际为零。

## 4. 方法

### 4.1 Arms

| arm | 基座 | 学习内容 | 算法 | 初始化 |
|---|---|---|---|---|
| `K4-zero` | K4 + ALOS | 无 | 传统控制器 | zero command residual |
| `ppo_ent1e4` | K4 + ALOS | 3D command residual | PPO | 现有 PC actor，entropy `1e-4` |
| `ppo_ent1e3` | K4 + ALOS | 3D command residual | PPO | 现有 PC actor，entropy `1e-3` |
| `ppo_ent1e2` | K4 + ALOS | 3D command residual | PPO | 现有 PC actor，entropy `1e-2` |
| `sac` | K4 + ALOS | 3D command residual | SAC | 现有 `pc_z_t` + BC encoder + zero residual |

PPO entropy 三档是最小 exploration screen：之前 PPO 的 residual 长期接近零，先用
一个预注册的熵梯度检验“是否根本没探索”，而不是把失败直接归因于 reward 或算法族。

### 4.2 固定配置

- `configs/lowlevel/goal150/env_current07_k4.json`
- `configs/lowlevel/goal150/ppo_ent1e4.json`
- `configs/lowlevel/goal150/ppo_ent1e3.json`
- `configs/lowlevel/goal150/ppo_ent1e2.json`
- `configs/lowlevel/goal150/sac_current07_k4.json`
- `configs/lowlevel/goal150/profile_manifest.json`

所有 PPO 配置 `initial_residual_zero=false`；SAC 保持现有 `pc_z_t` 与
`zero_residual_init=true`。不接 Goal140 reward，不改 `pc_command_track_v2` 的任何
权重。

## 5. 纵向误差不理想时的预注册分支

用户要求：如果途中发现效果不理想主要来自**纵向误差**，先回看 Goal140 的奖励修缮
方案并重新预注册实验；不能在当前 Gate 内临时改 reward 或扫权重。本节只记录思想，
不是本 Goal 的第二个 arm。

触发条件（必须同时满足）：

1. K4-zero 与 policy 的失败差异主要出现在 entry 前的纵向位置或 heave-rate，而不是
   横向、timeout、safety 或 terminal pitch 权限不足；
2. residual 符号/方向与纵向误差方向一致，但幅度长期接近零，或幅度增加后闭环纵向
   误差仍不降；
3. 当前 Goal150 Gate 已记录，允许开新 Goal，而不是修改本 Gate。

满足后新开一个 reward pilot，首选 Goal140 §4.2 的 potential-based 方案：

```text
R1 = baseline + F_z
F_z = -w_z * Huber(vertical_error)

R2 = baseline + F_z + F_zdot
F_zdot = heave-rate potential

F_predz = -w_predz * Huber(vertical_error + tau_z * heave_rate_error)
```

先 1 seed、200k transitions 的 `R1/R2` pilot，检查纵向 RMSE/p95、strict success、
timeout 和动作饱和；不直接扫 `w_z` / `w_predz`，不打开 held-out，不在本 Goal 中途
替换算法。

不触发的情况：执行器已饱和、terminal pitch 仍跟不上、timeout/safety 主导，或失败
主要是横向/路径误差。此时先修动作参数化、terminal capture 或控制器，不把 reward
当替代品。

## 6. 执行计划

- [x] Step 0：确认远端 GPU/进程空闲；核对 checkpoint、config SHA、dev seeds；
- [x] Step 1：本地跑 profile/config/override 测试，`bash -n` 检查 runner；
- [x] Step 2：远端同步 Goal150 配置、runner、evaluator 和共享代码；
- [x] Step 3：4 个训练 arm 并行，每臂 1M transitions；保留训练日志与 manifest；
- [x] Step 4：current07 主评估 `policy` vs `zero`，每臂 dev40；记录 rich trace；
- [x] Step 5：current05/current10 transfer，仅诊断，不反向选 checkpoint；
- [x] Step 6：按 §7 Gate 裁决；失败不扩 seed，先写 blocker；
- [x] Step 7：写 result、decision、LEDGER、STATE，运行 checker；不自动 commit/push。

运行入口：

```bash
cd /data/yhx/underwater-robotics/src/plan_p1/goal136_curriculum_20261111
GPUS_TRAIN="0 3 4 6" PARALLEL_EVAL=20 \
  bash scripts/run_goal150_current07_residual_screen.sh
```

汇总：

```bash
PYTHONPATH=src /data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python \
  scripts/analyze_pc_goal150_current07_screen.py \
  --root artifacts/lowlevel/goal150_current07_k4_residual_20261111
```

## 7. Gate

主评估：current07，dev40，policy 对同 arm 的 zero-command-residual 配对。

| check | 阈值 |
|---|---|
| 主指标 A | 至少一个学习 arm `success_policy - success_zero >= +2/40` |
| 主指标 B | 或 `eta_done_zero - eta_done_policy >= 0.02 m` |
| success guard | policy 不低于 zero 超过 `1/40` |
| timeout | policy 不高于 zero |
| safety | policy 不高于 zero |
| overshoot | policy `path_overshoot_m_mean` 不高于 zero `+0.02 m` |
| 数值 | 无 NaN / Inf / flew-out；否则 `INVALID` |
| residual 活性 | 通过主指标的 arm 必须有非零 command residual；全零不能算收益 |
| transfer | current05 不退化；current10 不出现 `>=3/40` success 损失 |

裁决规则：

| 观察 | 判读 | 下一步 |
|---|---|---|
| 任一 arm 通过主指标且 guard 全过 | `SCREEN_PASS_NOT_FORMAL` | 另开 3-seed/formal 预注册，仍不直接改 reward |
| 有方向但未过 `+2/40` / `0.02 m` | `SCREEN_INCONCLUSIVE` | 写 blocker，先检查纵向/横向机制，不扩 seed |
| residual 接近零 | `RESIDUAL_COLLAPSE` | 先查 exploration/credit；不先换 REDQ/DroQ |
| residual 非零但 success 下降 | `RESIDUAL_HARMFUL` | 先查动作参数化、reward、terminal capture |
| 纵向误差主导且方向正确但幅度不降 | `VERTICAL_REWARD_TRIGGER` | 按 §5 新开 reward pilot；不改本 Gate |
| 执行器饱和主导 | `SATURATION_LIMIT` | 转 controller/authority，不训练 reward |

本 Goal 只允许得出 screen 结论。1 seed、40 episodes 的结果不能写成 formal 收益。

## 8. Git

禁止 `git add -A`。只 add 本 Goal 的文档、配置、runner、analysis、测试、ledger 和
STATE；不提交 `artifacts/`、checkpoint、日志或缓存。提交前运行
`git diff --cached --check`，不自动 commit/push。

## 9. Goal 提示词

```text
你现在执行 Goal150：current07 K4-based PPO/SAC residual screen。

先读 Goal149 result、Goal148 result、Goal140 §0/§4.2 和本文件。固定 K4 controller、
C1 current07、dev seeds 20–59、goal radius 1.25，训练 1 seed、每臂 1M transitions：
K4+PPO entropy 1e-4/1e-3/1e-2，K4+SAC command residual；K4+zero 作为配对对照。

current07 是主评估；current05/current10 只做 transfer。不得改 reward、terminal、
success、radius、eval split 或 held-out，不得按 dev 结果挑 checkpoint，不得因单 seed
失败扩 seed 或扫权重。

Gate：至少一个学习 arm 多成功 +2/40 或 eta_done 改善 >=0.02 m，且 timeout/safety/
overshoot 不增加、residual 非零；否则先写 blocker 和两个可证伪 hypothesis，再决定
continue/pivot/stop。若明确是纵向误差主导且残差方向正确但幅度不降，另开 Goal140
potential-based reward pilot（F_z/F_zdot/F_predz），不要在当前 Gate 中途改 reward。
```
