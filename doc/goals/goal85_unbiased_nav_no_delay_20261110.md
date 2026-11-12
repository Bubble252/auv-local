# Goal85：去掉 p_nav 系统偏差与观测延迟，建立真值 eta 判分的位置契约

**日期**：2026-11-10<br>
**状态**：`CONTRACT_PASS_FRAME_DIAGNOSTIC_INVALID_EFFECT_NEGATIVE`（结果见 `goal85_result_20261110.md`）<br>
**上游**：Goal84 `FRAME_ACCOUNTED`<br>
**性质**：预注册实现 + eval-only Gate；不训练、不换算法、不打开 held-out。plant mismatch 不在本 Gate 内，但作为强制的 Goal85b 紧随其后。

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
  check_command: python ../scripts/check_research_ledger.py --goal Goal85 --goal-doc doc/goals/goal85_unbiased_nav_no_delay_20261110.md --strict
```

`required: false`：这是冻结的预注册 Gate。Gate 记录前不得改噪声范围、seed、eval
split、held-out、success 定义或 checkpoint 选择规则；Gate 失败后按项目规则补
blocker、hypothesis 和 ledger decision。

## 1. 背景与证据

Goal84 在 20 个已记录 run、800 个 episode 上完成只读重评分：

| 指标 | 数值 |
|---|---:|
| p_nav_tight success | `336/800 = 0.4200` |
| eta_tight_ever | `171/800 = 0.2138` |
| eta_tight_at_done | `151/800 = 0.1888` |
| p_nav-only | `185` |
| net gap | `-20.62 pp`，95% CI `[-29.13, -12.50] pp` |

zero-residual 对照也有约 `-22.5 pp` 的 gap，因此这不是 NoisyNet 或 learned
residual 独有的问题。当前证据指向两个系统性来源：

1. `p_nav` 含 `nav_bias + nav_drift + nav_event`，并且有 `nav_delay`、USBL/DVL/IMU
   延迟；
2. endpoint-stop 开启时，termination、goal distance 和 reward 都直接使用
   `p_nav`，把导航测量偏差带进了 success 判定。

本 Goal 只处理这两个来源。它不回答“动力学参数不准时策略是否鲁棒”；后者由强制的
Goal85b 负责。

## 2. 口径定义

### 2.1 两类位置误差

| 类型 | 定义 | 本 Goal 处理 |
|---|---|---|
| 导航测量位置偏差 | `E[p_nav - eta] != 0`，包括 bias、drift、掉锁漂移和延迟造成的滞后 | 关闭系统项，只保留零均值噪声 |
| 闭环跟踪误差 | `eta - reference` | 保留；这是控制算法要解决的问题 |

### 2.2 新位置契约

```text
training observation:  p_nav = eta + nav_noise,  E[nav_noise] = 0
reward / termination:  eta（真值）
official success:      eta_tight_at_done
diagnostic:            p_nav_tight_at_done
```

`p_nav` 仍然进入 actor observation，但不得进入 reward、termination、plane-miss
guard、checkpoint 选择或正式 success。`p_nav` 只用于部署风险诊断。

### 2.3 延迟边界

本 Goal 关闭所有**传感器/观测延迟**。执行器的一阶动态（舵 `0.1 s`、螺旋桨
`1.0 s`）属于被控对象物理动态，不是通信延迟，保留。`actor_decision_stride` 是
控制降频，不是延迟，本 Goal 固定为 `1`。

## 3. 假设

### H85-A：去掉系统项后 p_nav 位置误差退化为零均值噪声

新 profile 下，`p_nav - eta` 在 10 万步上的均值绝对值 `< 0.05 m`，且没有随时间
增长的漂移；噪声标准差落在预注册范围内。

成立应看到：均值接近 0、趋势接近 0、不同 seed 的均值符号随机。
失败应看到：均值 `>= 0.05 m` 或存在单调漂移。
最便宜证伪：固定 `eta` 轨迹，只跑 sensor suite，统计 `p_nav - eta` 的均值和
分段斜率，不需要训练。

### H85-B：termination/reward 与 p_nav 解耦后，p_nav 不再影响判分

当 `termination.position_source = eta` 时，把 `nav_noise_m` 从 `0` 调到配置上限，
episode 的 terminal kind、goal distance、reward 和 success 只通过 actor action
间接变化；在固定 action 序列下，这些量必须逐 episode 完全一致。

成立应看到：固定 action 序列时，改变 sensor noise 不改变 termination/reward。
失败应看到：p_nav 仍泄漏进 termination、plane-miss 或 reward。
最便宜证伪：构造固定 action 序列，分别用 `nav_noise_m=0` 和上限跑同一 seed，逐
step 比较 terminal/reward 张量。

### H85-C：旧 frame gap 主要由系统项解释，而不是 residual policy

在 `termination.position_source = eta` 且新 profile 无偏时，`eta_tight_at_done`
与 `p_nav_tight_at_done` 的 paired gap 应无系统性方向，95% CI 包含 0；不能再出现
Goal84 那种 `-20 pp` 量级的稳定负 gap。

成立应看到：gap 小且方向不稳定，能被零均值噪声解释。
失败应看到：仍存在稳定的负 gap，或 gap 与噪声量级不匹配。
最便宜证伪：用 Goal84 的 frozen checkpoints 在新 profile 下跑 eval-only 800
episodes，逐 seed 配对；旧 trace 不能直接重评分，因为它们仍携带旧 profile 的
bias/delay。不训练、不选 checkpoint。

## 4. 数据与隔离

| 项目 | 固定值 |
|---|---|
| dev seeds | `20–59` |
| held-out | `100–149` 关闭 |
| 评估规模 | 20 runs / 800 episodes，具体矩阵见 4.1 |
| checkpoint | Goal84 使用的 Goal71/Goal80 frozen checkpoints |
| 训练 | 禁止 |
| 新 rollout | 允许 eval-only rollout，不允许训练型 rollout |
| 算法切换 | 禁止；SAC/REDQ/DroQ 不启动 |
| oracle fallback | 禁止进入 actor、reward、termination 或 checkpoint |

禁止用 `eta` 标签选择 checkpoint、调 reward 或改变 action。`eta` 只允许在
episode 结束后用于离线评分。

### 4.1 冻结实验组（必须重跑，不复用 Goal84 trace）

Goal84 的 trace 是在旧 `realistic_calibrated` + p_nav termination 下生成的，
Goal85 改了 sensor profile 和 termination/reward 后不能直接重评分。正式 Gate
要重跑以下 20 个 run：

| 维度 | 固定值 |
|---|---|
| roots | `baseline`（Goal71 command-rate-limit）、`candidate`（Goal80 terminal-reference） |
| scenarios | `default`、`strict125` |
| policy checkpoints | Goal59 frozen `s240`、`s752`、`s880`、`s1008` 的 `final.pt` |
| zero-residual | 每个 root × scenario 各 1 个 run；不加载 learned residual |
| seeds | `20–59`，每个 run 40 个 |
| runs | policy `2×2×4 = 16`；zero-residual `2×2×1 = 4`；合计 `20` |
| episodes | `20 × 40 = 800` |
| 是否训练 | 否；只加载 frozen checkpoints 做 eval-only rollout |
| checkpoint 选择 | 不重新选择；沿用 Goal84 的 4 个 SHA |

`baseline` 使用 Goal71 的 default/strict125 入口；`candidate` 使用 Goal80 的
default/strict125 入口。`policy` 与 `zero-residual` 必须逐 seed 配对；不能把不同
checkpoint 的 episode 当成独立样本，也不能只跑 strict125 就替代正式 Gate。

### 4.2 目标附近次级评价（不改主 Gate）

本小节只增加解释性输出，不改变 `goal_radius`、`r_tight`、success、termination 或
reward。主 Gate 仍只认 `eta_tight_at_done`；近目标指标不得把 `FAIL` 改成 `PASS`，
也不得用于事后选择半径或 checkpoint。

固定半径阶梯（测量前冻结，不能看完结果再挑半径）：

```text
r_m = 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00 m
```

| 指标 | 定义 | 用途 |
|---|---|---|
| `eta_min_distance_m` | episode 内真实 `eta` 到 goal 的最小距离 | 是否接近过目标 |
| `p_nav_min_distance_m` | 同一轨迹的 `p_nav` 最小距离 | 只作估计器诊断 |
| `eta_done_distance_m` | 终止帧真实距离 | 终端误差 |
| `near_miss_shell_<m>_fraction` | `r_tight < eta_min <= r_tight + m` 的比例，`m=0.25/0.50/1.00` | 判断是否卡在捕获球外壳 |
| `annulus_entry_<r>_fraction` | `eta_min <= r` 的 episode 比例，`r` 取上表阶梯 | 目标附近分布曲线 |
| `conditional_capture_at_r` | 在 `eta_min <= r` 条件下 success 的比例 | 进入近区后的捕获转化率 |
| `radial_closure_speed_at_min_mps` | 最近点附近的 `-d(distance)/dt` | 是接近、悬停还是擦过 |
| `post_min_outward_speed_mps` | 最近点后固定窗口内的最大外离速度 | overshoot / pass-by 诊断 |
| `time_in_near_goal_s` | 在 `[r_tight, r_tight + 0.5 m]` 的停留时间 | 近区稳定性 |
| `plane_cross_lateral_offset_m` | crossing 时垂直航向的横向偏移 | 终端几何 |
| `plane_cross_vertical_offset_m` | crossing 时的垂向偏移 | 终端几何 |
| `radius_sensitivity_curve` | `eta_min_distance_m` 的 CDF 与各半径累计比例 | 一次看整体接近分布 |
| `capture_funnel` | `10→5→3→2→r_tight` 各层的条件通过率 | 定位失败发生在哪一层 |

#### Censoring 规则

`endpoint_stop` 会在成功进入 tight radius 时立即终止，因此成功 episode 的
`eta_min_distance_m` 被截断在 `r_tight` 附近，不能与失败 episode 的自由最小值
直接混算。所有近目标输出必须带：

```text
censored_at_tight = true   # 成功 episode
censored_at_tight = false  # 失败 episode
```

`near_miss_shell_*` 只对失败 episode 解释；`annulus_entry_*` 和
`radius_sensitivity_curve` 必须同时给出“含 censored 成功样本”和“仅失败样本”两套
数值，不能只报其中一套。`success_margin_m = r_tight - eta_done_distance_m` 只作诊断；
Goal74 已证明它不能作为收紧阈值的依据。

#### 可选 capture probe

如果 trace-only 结果显示失败样本大量集中在 `r_tight + 0.5 m` 内，另做 eval-only
`capture_probe`：同一 checkpoint、同一 seed，在首次进入 `r_tight` 后不立即停止，
继续固定 `2.0 s` 或直到 plane guard/timeout，记录无截断的最近距离、过冲和外离速度。
probe 只用于解释近目标分布，不改变主 success、不训练、不选择 checkpoint；其输出
不得写成正式 success。

### 4.3 本 Gate 固定的控制链（不是算法对照）

Goal85 只验证位置/终止契约，不比较控制架构。20 个 run 全部固定为：

| 轴 | Goal85 固定值 | 说明 |
|---|---|---|
| guidance | `pc_control_mode = alos` | 不是 `path_fixed_speed` / `path_learned_speed` |
| residual | `command_residual.enabled = true` | actor 输出 `[psi_d, theta_d, u_d]` 的 3D 指令残差 |
| actuator residual | `residual.scale_rudder/stern/rpm = 0` | 执行器残差关闭 |
| history encoder | Goal59 的 GRU + NoisyNet | 不是 DADE |
| DADE | 不启用 | DADE 是 PPO 架构变量，Goal59 checkpoint 不能直接复用 |
| path tracking | 不包含 | `path_*` 模式不在本 Gate |
| zero-residual | 同一 ALOS + PD/PI，learned residual 置零 | 不是 path tracking，也不是纯开环 |

因此 Goal85 的比较单位是“同一 ALOS + 3D 指令残差策略，在不同 checkpoint /
scenario / action mode 下”，不是“ALOS vs path tracking”“DADE vs no DADE”或
“执行器残差 vs 指令残差”。这些架构轴必须另开 Goal86，不能在 Goal85 的 800
episodes 里混跑。

## 5. 方法

### 5.1 新 sensor profile

新增 `unbiased_noise_v1`，不修改 `clean`、`mild`、`realistic_calibrated` 等历史
profile。建议值如下：

| 项目 | 字段 | 值 |
|---|---|---|
| 位置噪声 | 新增 `nav_noise_m` | `(0.05, 0.20)` m，每轴 1σ，每控制步独立采样 |
| p_nav 偏差 | `nav_bias_m` | `(0.0, 0.0)` |
| p_nav 漂移 | `nav_drift_sigma_m` | `(0.0, 0.0)` |
| DVL 掉锁漂移 | `nav_dvl_outage_drift_mps` | `(0.0, 0.0)` |
| 预标定残余 | `nav_bias_calibration_fraction` | `0.0` |
| USBL 融合 | `nav_usbl_gain` | `0.0`（p_nav 噪声由 `nav_noise_m` 提供，避免双重计数） |
| 所有传感器延迟 | `nav/usbl/dvl/imu/current_est *_delay_s` | `(0.0, 0.0)` |
| DVL 噪声 | `dvl_bottom_noise_mps`, `dvl_water_noise_mps` | `(0.005, 0.02)`, `(0.01, 0.03)` m/s |
| DVL bias/scale | `dvl_*_bias_mps`, `dvl_*_scale` | `0.0`, `1.0` |
| DVL outage | `dvl_*_outage_*` | `0.0` |
| IMU 噪声 | `imu_*_noise_*` | 保留 realistic 量级 |
| IMU bias/drift/declination | `imu_*_bias*`, `imu_drift_rad_s`, `imu_mag_declination_rad` | `0.0` |
| IMU gyro scale | `imu_gyro_scale` | `1.0` |
| current estimate 噪声 | `current_est_noise_mps` | `(0.005, 0.02)` m/s |
| current estimate bias/walk | `current_est_bias_mps`, `current_est_walk_mps` | `(0.005, 0.02)`, `(0.001, 0.005)` m/s |
| current estimate outage | `current_est_outage_*` | `0.0` |
| USBL outlier | `usbl_outlier_prob` | `0.0` |
| current 场 | `current` config | 保持现有 `0.25–0.45 m/s`、shear、time variation |
| actor 真值观测 | `observation.ideal_actor_observations` | `false` |

实现位置：`src/auv_lowlevel/batch/sensors.py:22` 的 `SensorProfile`、
`src/auv_lowlevel/batch/sensors.py:1040` 的 `p_nav_current`，以及
`src/auv_lowlevel/config.py:28` 的 `SUPPORTED_SENSOR_PROFILES`。

### 5.2 termination/reward 解耦

在 `TerminationConfig` 增加：

```text
position_source: legacy | eta | p_nav
```

| 值 | 行为 |
|---|---|
| `legacy` | 保持历史行为：endpoint-stop 开启时用 p_nav，否则用 eta |
| `eta` | 新 Goal85 协议；termination、goal distance、reward 都用 eta |
| `p_nav` | 仅用于显式部署对照；不得作为新 Gate 主口径 |

`legacy` 是为了不改变历史 config 的复现结果。Goal85 新 config 必须显式写
`"position_source": "eta"`。实现入口：`src/auv_lowlevel/batch/pc_env.py:1646`、
`src/auv_lowlevel/batch/pc_env.py:1742`、`src/auv_lowlevel/batch/pc.py:605`。

### 5.3 outcome contract 扩展

在 `src/auv_lowlevel/outcome_contract.py` 增加 `eta_done` frame：

```text
eta_done  -> success_eta_tight_at_done
p_nav_done -> success_pnav_tight_at_done（仅诊断）
```

正式报告必须同时记录 `termination_position_source`。`Goal84` 的
`p_nav`/`eta`/`both` 字段保持向后兼容，但 Goal85 的正式主指标是
`eta_tight_at_done`，不是 `eta_tight_ever`。当 termination 已切到 eta 时，
legacy `goal_terminations` 字段名不得再被解释为 p_nav 终止，报告必须显式写明。

### 5.4 config 与防呆

新增 config 只从 Goal84 的 default/strict125 入口派生，不修改已有的约 99 个
`realistic_calibrated` 历史 config。新 config 必须满足：

```text
sensors.profile == "unbiased_noise_v1"
termination.position_source == "eta"
observation.ideal_actor_observations == false
```

增加测试扫描 Goal85 config，拒绝旧 profile、旧 position source 和 true 真值观测。

### 5.5 plant mismatch 的边界

本 Goal **不实现** plant mismatch。原因是它会把“导航口径”与“模型鲁棒性”两个
变量混在一起。plant mismatch 是强制的后续 Goal85b：先做 drag/thrust/fin
±5–10%，再做 added mass ±5%，训练和评价同分布，不能省略。

## 6. 执行计划

- [x] Step 0：冻结本文件；记录 20 个 checkpoints、config SHA 和 Goal84 评估协议；
- [x] Step 1：实现 `nav_noise_m` 与 `unbiased_noise_v1`，注册 profile；
- [x] Step 2：实现 `termination.position_source`，让 reward/termination/goal distance 走该开关；
- [x] Step 3：扩展 outcome contract 的 `eta_done`/`p_nav_done`，补向后兼容测试；
- [x] Step 4：派生 default/strict125 新 config，加协议防呆测试；
- [x] Step 5：跑 10 万步 sensor bias audit 和固定 action 泄漏测试；
- [x] Step 6：用 frozen checkpoints 跑 eval-only 800 episodes；同时输出 §4.2 的
  近目标次级指标和 `censored_at_tight` 标记；不训练、不选 checkpoint；
- [x] Step 7：Gate 判定；写 result md、decision、LEDGER、STATE；
- [ ] Step 8：Goal85 通过后立即启动 Goal85b，plant mismatch 不得跳过。

验证命令：

```bash
# Step 1-4
../.venv/bin/pytest -q tests/test_pc_goal85_unbiased_contract.py \
  tests/test_pc_terminal_proximity.py
../.venv/bin/ruff check src/auv_lowlevel/batch/sensors.py \
  src/auv_lowlevel/batch/pc_env.py src/auv_lowlevel/batch/pc.py \
  src/auv_lowlevel/outcome_contract.py tests/test_pc_goal85_unbiased_contract.py \
  scripts/analyze_pc_terminal_proximity.py scripts/audit_pc_goal85_unbiased_nav.py

# Step 5（本地或远端，按实际脚本）
../.venv/bin/python scripts/audit_pc_goal85_unbiased_nav.py \
  --steps 100000 --envs 64 \
  --output artifacts/lowlevel/goal85_unbiased_nav_20261110/bias_audit.json

# Step 6（远端；按 4.1 的 20-run matrix 重复，下面是单 run 示例）
# 训练配置固定为 Goal59 的 command-residual 配置；环境配置为 Goal85 新派生 config。
PYTHONPATH=src /data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python \
  scripts/evaluate_pc_command_residual_offpolicy.py \
  --config configs/lowlevel/pc_goal59_offpolicy_g12_20261109.json \
  --environment-config configs/lowlevel/pc_goal85_unbiased_strict125_candidate_20261110.json \
  --checkpoint /data/yhx/underwater-robotics/src/plan_p1/goal59_bounded_terminal_residual_20261109/artifacts/formal_s240/final.pt \
  --output artifacts/lowlevel/goal85_unbiased_nav_20261110/strict125_candidate_s240_policy \
  --seed-start 20 --seed-count 40 --device cpu --dtype float64 \
  --action-mode policy --success-frame eta \
  --trace-output artifacts/lowlevel/goal85_unbiased_nav_20261110/strict125_candidate_s240_policy

# 近目标诊断（20 个 run 完成后逐个 trace 运行）
../.venv/bin/python scripts/analyze_pc_terminal_proximity.py \
  --trace artifacts/lowlevel/goal85_unbiased_nav_20261110/strict125_candidate_s240_policy/trace.npz \
  --output artifacts/lowlevel/goal85_unbiased_nav_20261110/strict125_candidate_s240_proximity.json \
  --markdown artifacts/lowlevel/goal85_unbiased_nav_20261110/strict125_candidate_s240_proximity.md

# Step 7
../.venv/bin/python ../scripts/check_research_ledger.py --goal Goal85 \
  --goal-doc doc/goals/goal85_unbiased_nav_no_delay_20261110.md \
  --strict --require-result
git diff --cached --check
```

## 7. Gate

| check | 阈值 |
|---|---:|
| 所有 `*_delay_steps` | `0` |
| `abs(mean(p_nav - eta))` | `< 0.05 m` |
| `p_nav - eta` 分段漂移 | 无单调趋势；首尾窗口均值差 `< 0.02 m` |
| `nav_noise` 标准差 | 落在 `[0.05, 0.20] m` |
| 固定 action 下 noise 改写 terminal/reward | `0` 差异 |
| 新 config 的 `position_source` | 全部 `eta` |
| 新 config 的 `ideal_actor_observations` | 全部 `false` |
| 新 config 使用旧 profile | `0` |
| 正式 run / episode | `20 / 800`，其中 policy `16`、zero-residual `4` |
| NaN / invalid / out-of-bounds | `0` |
| held-out opened | `false` |
| 历史 config 改动 | `0` |
| 近目标次级指标用于改变 success / Gate | `false` |
| `eta_done` vs `p_nav_done` paired gap | 95% CI 包含 0，且无稳定负方向 |

失败裁决：

| 观察 | 判读 | 下一步 |
|---|---|---|
| 均值偏差 `>=0.05 m` 或有漂移 | `IMPLEMENTATION_BIAS` | 先查 `nav_noise` 是否零均值、是否还有 `nav_event`/calibration 残留；不改算法 |
| 改 noise 后 terminal/reward 变化 | `PNav_LEAK` | 判 `INVALID`，修 termination/reward 解耦，不分析结果 |
| `eta_done` 仍系统性低于 `p_nav_done` | `FRAME_RESIDUAL` | 查 trace 的 done 帧和 goal 半径对齐；不换算法 |
| 新 profile 噪声导致控制崩溃 | `NOISE_TOO_HIGH` | 停下重写预注册噪声范围；不能看结果后微调 |

### 7.1 实际判读（2026-11-10 追加，不修改原 Gate）

原 Gate 的 `eta_done` vs `p_nav_done` success-gap 行在 endpoint-stop 下不可识别：
成功样本的 `eta_done` 被截断在 tight radius 边缘，零均值 `p_nav` 噪声会让约一半
成功样本落到球外。实际结果：

```text
success gap = -24.88 pp (95% CI [-31.00, -18.63] pp)  -> INVALID_METRIC_CENSORING
done-frame distance bias = +0.0157 m (95% CI [+0.0009, +0.0299] m) -> PASS (<0.05 m)
```

因此不能把该行写成 `FRAME_RESIDUAL`。完整证据和取代判据见
`research/blockers/2026-11-10-goal85-frame-gap-censoring.md`。原 success、
termination 和 radius 均未改变。

## 8. Git

禁止 `git add -A`。只 add 本 Goal 的文档、sensor/profile、termination 开关、
outcome contract、测试、新 config、分析脚本和 ledger/STATE；不要提交 artifacts、
checkpoint、日志或缓存。提交前运行 `git diff --cached --check`。用户明确要求前
不 commit / push。

建议提交信息：

```text
eval: add unbiased no-delay navigation contract
```

## 9. Goal 提示词

```text
你现在执行 Goal85：去掉 p_nav 系统偏差与观测延迟，建立真值 eta 判分的位置契约。

先读：
- doc/goals/goal85_unbiased_nav_no_delay_20261110.md
- doc/goals/goal84_result_20261110.md
- doc/goals/goal85b_plant_mismatch_mandatory_20261110.md

硬约束：
1. 不训练、不换算法、不启动 SAC/REDQ/DroQ。
2. 不改 reward 形状、action parameterization、checkpoint、seed 或 held-out。
3. 新 config 必须使用 unbiased_noise_v1、position_source=eta、
   ideal_actor_observations=false。
4. 不改历史 99 个 realistic_calibrated config。
5. plant mismatch 不在本 Gate 做，但 Goal85b 是强制后续，不能取消。
6. 读并执行 `research/AUTORESEARCH_LOOP.md`；Gate 完成后不要停下来等我逐步指令。
7. 如果 Gate FAIL 或出现 hard blocker，按 loop 写 blocker、至少两个可证伪 hypothesis、
   用 `paper-search` / `deep-research` 查论文、下载 PDF/TXT、定位可用代码、写 reading note、跑最便宜 falsification、写 ledger decision；
   不改本 Gate 的阈值或成功定义。
8. 如果 pivot，依据指标自动写新的 Goal/prompt 并在预注册预算内继续；持续迭代到 Gate PASS 或触发停止条件，不要因为一次 FAIL 结束；只有改变任务成功定义、
   打开 held-out、改变主算法族、超过预算或 commit/push 时才问我。不自动 commit/push。

执行顺序：
- 实现 profile 和 termination/reward 解耦；
- 扩展 outcome contract 的 eta_done/p_nav_done；
- 跑 bias audit 和 p_nav 泄漏测试；
- 用 frozen checkpoints 跑 eval-only 800 episodes；
- 按 Gate 判 INVALID / PASS / FAIL；
- 写 result、ledger、STATE，再进入 Goal85b。
```
