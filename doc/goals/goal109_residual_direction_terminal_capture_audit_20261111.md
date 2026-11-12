# Goal109：B20 residual direction / action parameterization / terminal capture 审计

**日期**：2026-11-11<br>
**状态**：`COMPLETED_AUDIT_ONLY / NO_DIRECTION_SIGNAL / DIRECTION_ONLY_SIGNAL_OBSERVED / HELDOUT_CLOSED`<br>
**上游**：Goal108 `NO_PRIOR_OPTIMIZER_GAIN`；Goal107 terminal reward/credit；Goal104 channel audit<br>
**性质**：只读审计；不训练，不改 reward、terminal、success、goal radius、dev seeds 或 held-out。

## Research Contract

```yaml
autoresearch:
  required: false
  triggers:
    - gate_fail_twice
    - no_metric_movement_after_3_fixes
    - heldout_collapse
    - seed_noise_ambiguous
  ledger: research/LEDGER.jsonl
  blockers_dir: research/blockers/
  reading_dir: research/paper_reading/
  max_outer_loops: 3
  check_command: python ../scripts/check_research_ledger.py --goal Goal109 --goal-doc doc/goals/goal109_residual_direction_terminal_capture_audit_20261111.md --strict --require-result
```

记录前不得修改 base checkpoint、eval scale、seed、scenario、success frame、path 阈值或 Gate 判读规则。

## 1. 背景

Goal108 已经排除“优化器轴自然能从 B20 出发产生收益”：

```text
PPO: 3/3 seeds default 33/40, strict 20/40；increment L2 ratio 0.0060–0.0095
SAC: default 4/5/9, strict 2/2/5；action abs max 约 0.996，并出现 timeout/safety
```

网络审计确认没有 NoisyNet、没有接线错位。SAC 的大残差来自 deterministic mu
本身饱和，不是 eval 噪声。Goal107 又已经证明 terminal reward 和 capture 几何
都能区分 success/failure，但还没有回答：

```text
失败种子的 residual 是方向错、幅度错，还是 terminal capture 几何本身不可达？
```

Goal109 只读复用 Goal107 的 trace，不再训练，避免继续在优化器轴上烧预算。

## 2. 数据与隔离

- 主数据：`artifacts/lowlevel/goal107_terminal_reward_credit_20261111/`
- 主 arm：`P_D3_strict125_C0`、`P_D3_default_C0`（frozen-base，等价 B20 anchor）
- 正对照：`P_D3_strict125_U050`、`P_D3_default_U050`（rpm 0.5）
- scenario：default / strict125；
- seeds：`20–59`；
- success frame：`both`；
- 不新增训练、不打开 held-out、不改任何契约。

## 3. 指标

所有指标在最后 2 s 窗口计算，并复用 Goal107 的 success/failure 标签。

### 3.1 residual 方向与幅度

| 指标 | 定义 | 目的 |
|---|---|---|
| `applied_base_cosine_mean` | `cos(applied_residual, actuator_base_physical)` 的 last2s 均值 | residual 是否在系统性地反对 ALOS/基座控制方向 |
| `applied_base_cosine_p05` | 同上 5% 分位 | 检查少数强反对步是否集中 |
| `residual_norm_mean` | `norm(applied_residual)` last2s 均值 | 失败是否只是幅度更大 |
| `residual_norm_p95` | 同上 95% 分位 | 检查尖峰饱和 |
| `material_saturation_fraction` | `material_saturated` last2s 比例 | 执行器饱和是否与失败共变 |
| `rudder_heading_sign_alignment` | `mean(sign(rudder_residual) * sign(heading_error))` | rudder 与 heading error 的方向关系 |
| `stern_pitch_sign_alignment` | `mean(sign(stern_residual) * sign(pitch_error))` | stern 与 pitch error 的方向关系 |
| `rpm_speed_sign_alignment` | `mean(sign(rpm_residual) * sign(speed_error))` | rpm 与速度误差的方向关系 |

方向指标不预设哪个符号是“正确”；先比较 success/failure 的效应量，再结合
`applied_base_cosine` 判断是方向不一致还是单纯幅度过大。

### 3.2 terminal capture 与闭合

- `final_goal_distance_m`；
- `terminal_horizontal_miss_m`、`terminal_vertical_miss_m`；
- `terminal_abs_heading_error_rad`、`terminal_abs_pitch_error_rad`；
- `last2s_closure_mean_mps`、`last2s_closure_last_mps`；
- `last2s_closure_negative_fraction`；
- 过渡统计：最后 2 s 内进入 `2.0/1.5/1.25/1.0 m` 壳层的步数比例，以及
  `final_distance` 的分位数。该项只作描述，不改变 success 定义。

## 4. 假设

### H109-A：失败 residual 方向与基座/误差方向不一致

- 机制：BC 残差在终端阶段继续追路径，而不是追 goal point，导致闭合速度掉到零。
- 成立：strict C0 failure 的 `applied_base_cosine_mean` 或至少一个通道 sign
  alignment 与 success 的 `abs(Cliff's delta) >= 0.33`，且 residual magnitude /
  saturation 也至少有一项 `>= 0.33`。
- 失败：方向指标效应量都 `< 0.33`，或 success/failure 方向分布几乎相同。

### H109-B：失败主要是 residual 幅度/饱和，而不是方向

- 机制：direction 指标不分离，但 `residual_norm` 或 `material_saturation_fraction`
  在 failure 中显著更大。
- 成立：方向指标全部 `< 0.33`，magnitude/saturation 至少一项 `>= 0.33`。
- 失败：magnitude/saturation 也不分离。

### H109-C：失败由 terminal capture 几何决定，方向不是主因

- 机制：方向/magnitude 都不分离，但 final distance、vertical miss、closure 等
  几何指标分离。
- 成立：方向与 magnitude 指标都 `< 0.33`，至少 2 个几何指标 `>= 0.33`。
- 失败：几何也不分离。

### H109-D：现有 trace 不足以区分

- 机制：方向、幅度、几何都没有达到预注册效应量，或 trace 缺少必要字段。
- 成立：没有任何类别达到 Gate。
- 失败：至少一个类别达到 Gate。

## 5. Gate

### 5.1 `INVALID`

- C0 default/strict 不能复现 `33/20`；
- seed 不是 `20–59`，或 trace/summary 缺失；
- trace 缺 `actuator_residual_applied`、`actuator_base_physical`、`heading_error_rad`、
  `pitch_error_rad`、`speed_error_mps` 或 `material_saturated`；
- 读取 held-out，或修改 reward/terminal/success/goal radius。

### 5.2 `RESIDUAL_DIRECTION_MISMATCH`

strict C0 failure 同时满足：

```text
max(abs(direction_metric_Cliff_delta)) >= 0.33
max(abs(magnitude_or_saturation_Cliff_delta)) >= 0.33
```

此时下一步才允许设计“方向保持的 joint norm / terminal projection”预注册实验，
而不是继续扫 scale。

### 5.3 `RESIDUAL_MAGNITUDE_LIMITED`

方向指标全部 `<0.33`，但 magnitude/saturation 至少一项 `>=0.33`。下一步只审计
joint norm/authority，不先改 reward。

### 5.4 `TERMINAL_GEOMETRY_LIMITED`

方向与 magnitude 都 `<0.33`，至少 2 个 terminal geometry/closure 指标 `>=0.33`。
下一步进入 terminal capture / endpoint-stop 几何或 terminal reward，不改动作方向。

### 5.5 `NO_DIRECTION_SIGNAL`

没有任何类别达到阈值。停止在这批 trace 上继续做方向归因，补 instrumentation 或
另开更接近 terminal 的采样实验。

## 6. 执行计划

- [x] Step 0：写预注册、Research Contract、确定只读数据源；
- [x] Step 1：实现 trace 方向/几何分析器和单测；
- [x] Step 2：在 Goal107 的 C0/U050 trace 上生成 phase0 summary；
- [x] Step 3：按冻结 Gate 判读，写 result/decision/LEDGER/STATE；
- [x] Step 4：运行 checker；
- [x] Step 5：只有 `RESIDUAL_DIRECTION_MISMATCH` 或 `RESIDUAL_MAGNITUDE_LIMITED`
  才另开训练/参数化 Goal；不自动 commit/push。

## 9. 结果

正式输出见：

```text
doc/goals/goal109_result_20261111.md
```

冻结 Gate 输出 `NO_DIRECTION_SIGNAL`，原因是
`direction_separates_but_magnitude_does_not_no_preregistered_branch`。
strict C0 的 `rpm_speed_sign_alignment` 为 `0.3825`，但幅度最大效应量只有
`0.1950`；几何有 6 项强分离。该组合没有落在预注册分支内，不能事后改判。

## 7. 运行命令

```bash
python scripts/analyze_pc_goal109_residual_direction.py \
  --root artifacts/lowlevel/goal107_terminal_reward_credit_20261111 \
  --output artifacts/lowlevel/goal109_residual_direction_20261111/phase0_summary.json \
  --markdown artifacts/lowlevel/goal109_residual_direction_20261111/phase0_summary.md
```

## 8. 不做什么

- 不训练 PPO/SAC/REDQ/DroQ/DADE；
- 不修改 reward、terminal、success、goal radius；
- 不继续扫 residual scale/channel gain/rpm；
- 不打开 held-out `100–149`；
- 不把方向指标的相关性直接写成因果；
- 不用单 seed 或放宽阈值写 PASS。

