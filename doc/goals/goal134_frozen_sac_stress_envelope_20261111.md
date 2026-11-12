# Goal134：冻结 SAC vs zero 的压力包络测评

**日期**：2026-11-11<br>
**状态**：`COMPLETED_EVAL_ONLY / SINGLE_SEED_ONLY / SAC_STRESS_HARM / HELDOUT_CLOSED`<br>
**上游**：Goal133 `GEO_SAFE_PARITY / EVAL_BIAS_FAMILY_STOPPED`；Goal85b `MISMATCH_TOO_SMALL`；Goal126 `SAC_PA_HARM`<br>
**性质**：冻结策略压力测评；不训练、不换算法、不改 success/goal radius/terminal/reward。

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
  check_command: python3 ../scripts/check_research_ledger.py --goal Goal134 --goal-doc doc/goals/goal134_frozen_sac_stress_envelope_20261111.md --strict --require-result
```

- `required: true`：压力结果若没有可复现优势，必须进入 autoresearch 归因。
- Gate 记录前不改 stress 档位、seed、success 定义、checkpoint 集合或判据。

## 1. 背景与证据

Goal133 已停止 eval-only geometry bias 家族：正确符号 dose 单调改善 3 cm
vertical error，但 strict success 只从 24 到 25，default 完全不变。因此固定
bias 已到上限。

Goal85b 已证明 plant randomization 实现正确，但 ±10% drag/thrust/fin/added-mass
只造成约 3 cm path RMSE 退化，没有稳定改变终端捕获。Goal126 又证明当前
1M SAC 在 nominal dev40 上总体 `SAC_PA_HARM`，只有 86201 达到 30/33。

本轮只回答一个问题：

> 当前 nominal 测评是否对 ALOS+BC 太容易，导致神经网络残差的优势被天花板遮住？

如果压力包络下 zero 明显退化、而多个 SAC checkpoint 稳定获益，才值得把压力
分布写进训练型 Goal；否则回到训练信号/算法机制，而不是继续调测评。

## 2. 假设

### H134-A：压力包络放大 SAC 优势

- 机制：higher current / plant mismatch 让 ALOS+BC 的固定 PD/PI 与路径跟踪误差变大，
  SAC 学习残差在多个 checkpoint 上提供额外修正。
- 成立：至少一个 stress level 中 `>=4/8` checkpoint 的 strict success 比 zero
  高 `>=2`，且同 checkpoint default success 不低于 zero。
- 失败：所有 stress level 都没有 `>=4/8`。
- 最便宜证伪：8 checkpoint × 5 stress level 的 eval-only 矩阵。

### H134-B：只有 86201 能在压力下获益

- 机制：1M SAC 训练方差大，只有单个训练 seed 学到可迁移的残差方向。
- 成立：唯一满足 strict `+2` 且 default 不退化的 checkpoint 是 86201。
- 失败：0 个或 `>=2` 个 checkpoint 满足。
- 最便宜证伪：逐 checkpoint 报告，不聚合均值。

### H134-C：压力不足以区分方法

- 机制：current 1.5x/2x 与 plant ±20% 仍没有把 zero 的 strict success 压下 2 个
  seed 以上。
- 成立：zero 在所有 stress level 的 strict success 相比 nominal 下降 `<2`，
  或 default 仍保持 33/40。
- 失败：至少一个 stress level 让 zero strict 下降 `>=2`。
- 最便宜证伪：先跑 zero 的 5 个 stress level；若 zero 不动，不扩到 8 checkpoint。

### H134-D：SAC 在压力下比 zero 更脆弱

- 机制：SAC 残差在 nominal 上过拟合，current/plant 偏移后动作饱和或振荡。
- 成立：至少一个 stress level 中 `>=4/8` checkpoint 的 strict 或 default 比 zero
  低 `>=2`。
- 失败：没有 `>=4/8` 的稳定退化。
- 最便宜证伪：同矩阵的 strict/default paired delta 与 timeout/safety 计数。

## 3. 数据与隔离

| 项目 | 值 |
|---|---|
| policy checkpoint | Goal126 `S_s86201..S_s86208/final.zip` |
| zero control | 同训练配置，`--action-mode zero-residual` |
| checkpoint 选择 | 测量前固定 8 个，不按压力结果换 seed |
| eval seeds | dev `20–59`，40 个 |
| held-out | `100–149` 保持关闭 |
| 训练 | 无 |
| reward / terminal / success / radius | 完全沿用 Goal126/Goal133 |

传感器固定为 Goal85 的 `unbiased_noise_v1`：只保留无偏噪声，不引入 p_nav/eta
位置偏差或延迟。压力只进入真实 current 和真实 plant。Goal134 不增加噪声档；
若 current/plant 被判 `STRESS_NOT_DISCRIMINATIVE`，下一个 Goal 再加无偏噪声 scale。

## 4. 冻结矩阵

### 4.1 Stress levels

| level | current | plant | 说明 |
|---|---|---|---|
| `nominal` | 原始 `[0.25,0.45] m/s` | 全 1.0 | 复现 Goal126 |
| `current15` | speed/shear/time-variation ×1.5 | 全 1.0 | 只加压 current |
| `current20` | speed/shear/time-variation ×2.0 | 全 1.0 | 更强 current |
| `plant20` | 原始 | 五通道 `[0.8,1.2]` | 只加压 plant |
| `combined20` | ×2.0 | 五通道 `[0.8,1.2]` | 组合压力 |

current 缩放只作用于：

- `current.speed_mps` 上下界；
- `current.time_variation.speed_amplitude_mps`；
- `current.shear.velocity_gradient_per_s`。

方向范围 `direction_rad` 和方向波动幅值保持不变，避免把“流速压力”和“方向
分布改变”混在一起。

plant 缩放作用于 Goal85b 已实现的五个通道：

```text
drag_scale, thrust_scale, fin_scale, added_mass_scale, actuator_time_scale
```

五通道共用同一 randomization seed `13420`，不同因子使用不同 hash salt；所有
arm 共用 common random numbers。

### 4.2 Arms 与 run 数

| arm | 数量 |
|---|---:|
| zero-residual | 1 |
| Goal126 SAC checkpoints | 8 |
| **每 stress / scenario** | **9** |

```text
5 stress levels × 2 scenarios × 9 arms = 90 runs
90 runs × 40 episodes = 3600 episodes
```

scenario 固定为：

- `strict`：goal radius `1.25 m`；
- `default`：goal radius `2.00 m`。

### 4.3 固定评估口径

- `--success-frame both`；
- `--residual-scale 1.0`，环境内部 scale 继续为 `2.0`；
- `--device cpu --dtype float64`；
- `--seed-start 20 --seed-count 40`；
- 不启用 geometry bias、learner gate、norm cap 或 channel cap；
- nominal zero 必须复现 strict `24/40`、default `33/40`；
- nominal 86201 必须复现 strict `30/40`、default `33/40`；否则 `INVALID`。

## 5. 指标

主指标：

- strict success（`both`）；
- default success（`both`）；
- 逐 seed paired delta；
- timeout / goal-plane-miss / safety 计数。

解释指标：

- `path_three_d_rmse_m`；
- `terminal_eta_distance_m`、`terminal_p_nav_distance_m`；
- `residual_l2_ratio`、`action_abs_max`、`action_saturation_fraction`；
- `learned_to_bc_residual_action_rms_ratio`。

不把连续指标替代 success，也不改 1.25/2.00 m 阈值。

## 6. Gate

### 6.1 Validity

| check | 要求 |
|---|---|
| run 完整性 | 90/90 `summary.json`，全部 finite |
| nominal parity | zero `24/33`，86201 `30/33` |
| 配置哈希 | 每个 stress config 记录 SHA256 |
| held-out | `100–149` episode 数为 0 |
| 算法 | 无训练、无 PPO/SAC 切换、无 REDQ/DroQ |

任一失败：`INVALID`，只修实现，不解释方法优劣。

### 6.2 Stress discriminative

对 zero，`strict_drop(level) = zero_nominal_strict - zero_level_strict`。

| 观察 | 判读 |
|---|---|
| 至少一个 level `strict_drop >= 2` | `STRESS_DISCRIMINATIVE` |
| 所有 level `strict_drop < 2` | `STRESS_NOT_DISCRIMINATIVE` |

`STRESS_NOT_DISCRIMINATIVE` 时仍报告 SAC delta，但不能用它证明“压力能放大 NN
优势”，下一步先加压力或改压力维度。

### 6.3 SAC advantage

单个 checkpoint 在某个 stress level 记为 `advantaged`，当且仅当：

```text
strict_delta = sac_strict - zero_strict >= 2
default_delta = sac_default - zero_default >= 0
```

| 观察 | 状态 | 下一步 |
|---|---|---|
| `>=4/8` checkpoint advantaged | `STRESS_ADVANTAGE` | 预注册压力分布训练型 Goal135 |
| `2–3/8` checkpoint advantaged | `STRESS_PARTIAL` | 先诊断共同机制，不直接扩 5M |
| 仅 86201 advantaged | `SINGLE_SEED_ONLY` | 不扩训练；查 seed 方差/训练信号 |
| `0/8` advantaged 且 stress discriminative | `STRESS_NO_ADVANTAGE` | 停止“压力自然放大 NN”假设，转训练信号/算法机制 |
| `>=4/8` checkpoint 在 strict 或 default 低 `>=2` | `SAC_STRESS_HARM` | 先查动作饱和与 reward/terminal，不换算法 |

所有状态都保留 nominal 结果；压力结果单独报告，不混入 Goal126 的 nominal
结论，也不重写 `SAC_PA_HARM`。

## 7. 执行计划

- [x] Step 0：写本 Goal，测量前冻结 stress/arms/Gate；
- [x] Step 1：实现 stress config builder、runner、analyzer 和 tests；
- [x] Step 2：本地/远端 smoke：nominal zero + 1 checkpoint × 2 stress；
- [x] Step 3：正式 90 runs × 40 seeds；
- [x] Step 4：Validity、stress sanity、per-checkpoint Gate；
- [x] Step 5：写 result/blocker/decision/LEDGER/STATE；
- [x] Step 6：checker；不自动 commit/push。

### 7.1 便宜失败优先

先跑 zero 的 5 个 stress level × 2 scenarios（10 runs）。若 zero 在所有 level
strict drop `<2`，直接记 `STRESS_NOT_DISCRIMINATIVE`，不启动 8 checkpoint 的
80 个 run；但仍保留 10 个 zero run 作为证据。

## 8. 停止规则

- `STRESS_ADVANTAGE`：只预注册 Goal135 训练型压力分布，不在本 Goal 训练；
- `STRESS_PARTIAL` / `SINGLE_SEED_ONLY`：不扩正式训练，先做训练信号/seed 方差归因；
- `STRESS_NO_ADVANTAGE`：停止“压力环境自然放大 NN 优势”的假设，转 reward/terminal/
  动作参数化/学习信号；
- `SAC_STRESS_HARM`：先查动作饱和与 terminal credit，不换 PPO/SAC；
- 不打开 held-out，不改 success/radius，不换主算法，不自动 commit/push。

## 9. Git

禁止 `git add -A`。只 add 本 Goal 相关文件，提交前运行：

```bash
python3 ../scripts/check_research_ledger.py \
  --goal Goal134 \
  --goal-doc doc/goals/goal134_frozen_sac_stress_envelope_20261111.md \
  --strict --require-result
git diff --cached --check
```

## 10. Goal 提示词

```text
你现在执行 Goal134：冻结 SAC vs zero 的压力包络测评。

先读：
- doc/goals/goal134_frozen_sac_stress_envelope_20261111.md
- doc/goals/goal133_result_20261111.md
- doc/goals/goal85b_result_20261110.md
- doc/goals/goal126_result_20261111.md

硬约束：
1. eval-only；不训练、不换 PPO/SAC、不开 REDQ/DroQ；
2. 不改 reward/terminal/success/goal radius；不打开 held-out；
3. stress 只进入真实 current/plant，传感器保持 unbiased_noise_v1；
4. 先跑 zero 的 10 个 stress run；若 zero 不退化，直接停，不烧 8 checkpoint；
5. 只认跨 checkpoint 的 paired 优势，不认单一 86201 幸存者；
6. 结果写 result/blocker/decision/LEDGER/STATE，跑 checker；不自动 commit/push。
```

## 11. 自主迭代契约

本 Goal 执行 `research/AUTORESEARCH_LOOP.md`：

1. 失败先归因，不直接换算法或扫 seed；
2. blocker 至少写两个可证伪 hypothesis；
3. 需要外部方法时先写 reading note；
4. 先跑最便宜 falsification，再写 `continue` / `pivot` / `stop`；
5. `pivot` 必须新写 Goal 文档，不修改本 Gate 阈值；
6. 不自动 commit/push，不打开 held-out，不改成功定义或主算法族。
