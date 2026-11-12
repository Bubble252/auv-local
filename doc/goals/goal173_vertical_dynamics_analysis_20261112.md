# Goal173 纵向误差动力学归因：为什么 rudder-only 不能解决纵向

**日期**：2026-11-12<br>
**状态**：`DIAGNOSTIC_ONLY / HELDOUT_CLOSED / NO_TRAINING_CHANGE`<br>
**数据**：`goal173_rudder_only_residual_20261112/eval`，`default/current10`，dev seeds `20–59`<br>
**结论标签**：`VERTICAL_AUTHORITY_ABSENT / OBSERVABILITY_LIMITED / NOISENET_NOT_DOMINANT`

## 1. 结论先说

纵向没有解决，不是单一原因，而是四个因素叠加：

1. **`P_RUD_075` 在动作空间上没有纵向权限。** 学习器施加的 stern residual 平均只有
   `6.4e-6°`，有效通道增益是 `[1, 1e-6, 1e-6]`。它的纵向统计与
   `zero_alos50` 几乎逐项相同，说明 rudder-only 改善的 9% 是水平/cross-track，
   不是深度。
2. **REMUS100 的深度是间接、欠驱动的二阶通道。** 没有垂直推进器；stern plane
   主要产生 pitch moment，深度靠 `zdot = -u sinθ + ...` 积分出来。舵面直接的
   heave 力与爬升方向相反，且随 `u²` 缩放。舵面权限足够（p95 约 `7.4°`，
   饱和率仅 `1.8%`），但深度环的相位/带宽有限。
3. **控制器看到的 `p_nav` 纵向噪声与真实纵向误差同量级。** 本 cell 的
   `p_nav` 三维误差约 `0.314 m`，其中纵向分量噪声约 `0.197 m`；真实纵向
   误差 RMS 约 `0.204 m`，两者同量级，单步相关系数只有 `0.44`。指导律用
   `p_nav`，评价用 true `eta`，所以真实纵向误差对控制器是“半可观测”的。
4. **全通道 residual 学到的是“补 pitch”，不是“补 depth”。** `P_ALL_075` 的
   有效 stern residual 平均 `5.6°`、p95 `15°`（打到 75% 残差上限），与 pitch error
   相关 `0.60`，与 vertical error 相关 `-0.32`。它把收敛时间从约 `20 s`
   缩短到约 `10 s`，但末端纵向误差从 `0.055 m` 恶化到 `0.168 m`。这是
   奖励/credit assignment 与动作饱和的 tradeoff，不是 NoiseNet。

因此当前最准确的表述是：

> `P_RUD_075` 的收益是横向/整条 3D route 的收益；纵向没有解决是结构性的。
> 即使换成全通道，纵向也主要受“间接 depth 通道 + 外环带宽 + p_nav 观测噪声”
> 限制，而不是舵面不够或 NoiseNet 主导。

## 2. 同协议数值证据

### 2.1 三个 arm 的纵向统计

`default/current10`，40 个 dev seeds：

| arm | vertical bias | vertical RMS | vertical p90 | pitch error p90 | stern p95 | stern 饱和率 | 首次进入 ±5 cm 中位时间 | 末端 abs(e_v) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `P_RUD_075` | `-0.0402 m` | `0.2038 m` | `0.2959 m` | `2.31°` | `7.38°` | `1.79%` | `19.8 s` | `0.060 m` |
| `P_ALL_075` | `+0.0365 m` | `0.1855 m` | `0.3038 m` | `5.59°` | `13.99°` | `7.90%` | `9.96 s` | `0.168 m` |
| `zero_alos50` | `-0.0405 m` | `0.2037 m` | `0.2957 m` | `2.33°` | `7.38°` | `1.83%` | `19.8 s` | `0.055 m` |
| `zero_bc50` | `-0.0120 m` | `0.1823 m` | `0.2854 m` | `3.68°` | `10.52°` | `4.12%` | — | — |

`P_RUD_075` 与 `zero_alos50` 的纵向指标几乎相同；这不是“学得不好”，
而是这个 arm 根本没有 stern/rpm 权限。相对 `zero_bc50`，它的 vertical p90
约恶化 `9.4 mm`；相对 `zero_alos50` 仅差 `0.00004 m`，基本是同一个纵向轨迹。

### 2.2 学习器实际施加的残差

| arm | 有效通道增益 | rudder 施加均值 | stern 施加均值 | stern p95 | rpm 施加均值 | rpm p95 |
|---|---|---:|---:|---:|---:|---:|
| `P_RUD_075` | `[1, 1e-6, 1e-6]` | `7.57°` | `6.4e-6°` | `1.5e-5°` | `0.00017 rpm` | `0.00048 rpm` |
| `P_ALL_075` | `[1, 1, 1]` | `7.34°` | `6.20°` | `14.57°` | `189 rpm` | `480 rpm` |

注意：`P_RUD_075` 的 `actuator_total_physical - actuator_base_physical`
在 stern 上平均约 `0.57°`，这不是学习器动作，而是 base 命令走总命令路径时
被限速后的差值；学习器直接施加的 stern residual 是 `6.4e-6°`。

### 2.3 纵向误差主要是“初始捕获”，不是稳态偏差

`P_RUD_075` / `default/current10`，40 seeds：

| 统计 | 数值 |
|---|---:|
| 初始 true vertical error 均值 / 标准差 | `-0.0438 m` / `0.2790 m` |
| 初始误差 5% / 95% 分位 | `-0.431 m` / `+0.462 m` |
| 单 episode 平均误差与初始误差相关 | `0.933` |
| 误差符号与初始符号一致的时间占比 | `86.9%` |
| 末端误差均值 / 标准差 | `0.0016 m` / `0.0730 m` |
| `abs(e_v) > 0.1 m` 的时间占比 | `64.8%` |
| `abs(e_v) > 0.2 m` 的时间占比 | `37.6%` |

五段 RMS 依次为：

| 路线段 | 0–20% | 20–40% | 40–60% | 60–80% | 80–100% |
|---|---:|---:|---:|---:|---:|
| vertical RMS | `0.234 m` | `0.224 m` | `0.207 m` | `0.172 m` | `0.107 m` |

这说明纵向误差是“从随机初始深度偏差慢慢收敛”的瞬态；控制器最终能把末端
误差压到厘米级，但整段平均仍被前 40% 的捕获过程主导。

### 2.4 p_nav 观测噪声与真实纵向误差同量级

`P_RUD_075`，40 seeds：

| 指标 | 数值 |
|---|---:|
| `p_nav` 三维误差均值 | `0.314 m` |
| `p_nav` 相对 true `eta` 的 z 误差 std | `0.254 m` |
| `p_nav` 纵向误差偏差 | `-0.0003 m`（无系统偏置） |
| `p_nav` 纵向误差噪声 std | `0.197 m` |
| true vertical error RMS | `0.204 m` |
| `p_nav` 与 true vertical error 相关 | `0.444` |
| 逐 episode noise/signal 均值 | `1.63`（聚合 RMS 比约 `0.97`） |

指导律用 `p_nav` 计算 `vertical_error`，而评价用 true `eta` 计算。
因此控制器每步看到的纵向误差被同量级零均值噪声污染；它能靠时间平均
收敛，但单步反馈的有效增益被显著衰减。这是“真实纵向误差动得很慢”的
重要原因，也正好对应之前要求忽略 `p_nav`/`eta` 差异的那类问题。上面的
纵向噪声分量是用 true projection 轴重构的近似值；`p_nav` 三维误差
`0.314 m` 是直接可观测量。

## 3. 动力学解释

### 3.1 stern plane 是 pitch 执行器，不是 depth 执行器

REMUS100 的纵向力和力矩在
`src/auv_lowlevel/batch/dynamics.py:563–587`：

```text
stern_heave = -0.5 * rho * (u^2 + w^2) * S_stern * C_L * delta_s
actuator_force[z] = stern_heave
actuator_force[pitch] = -x_stern * stern_heave
```

`x_stern = -0.8 m`，所以 pitch moment 与 heave 力符号相反。直观地说：
舵面把尾部往下推、让艇艏抬起；但同一个舵面也在把整艇往下推。爬升最终来自
艇体抬头后前向速度的垂直分量：

```text
zdot = -u * sin(theta) + cos(theta) * (w * cos(phi) + v * sin(phi))
```

这不是“给一个 depth 指令就产生一个 depth 力”的直接通道，而是
`stern -> pitch moment -> pitch angle -> zdot -> depth` 的级联积分。

### 3.2 量级：舵面权限足够，但直接 heave 力很小

REMUS100 参数：质量约 `31.0 kg`，重量约 `304 N`，`Iyy ≈ 4.03 kg m²`。
在 `u ≈ 1.3 m/s` 时：

| stern 偏角 | 直接 heave 力 | pitch moment |
|---:|---:|---:|
| `7.4°`（本 cell p95） | `≈ 0.9 N` | `≈ 0.7 Nm` |
| `15°` | `≈ 1.8 N` | `≈ 1.4 Nm` |
| `20°`（满量程） | `≈ 2.4 N` | `≈ 1.9 Nm` |

直接 heave 力不到重量的 `1%`；深度变化主要靠 pitch 角乘以前向速度。
本模型 `weight = buoyancy`，静浮力已经抵消重量，所以真正驱动深度的是
小净力和推力的垂直分量。注意：在小 pitch 角下，推进器推力的垂直分量
也只有约 `0.3–0.8 N`，
所以舵面的反向 heave 力并不是可以忽略的——它会在抬头瞬态里抵消一部分
爬升力，表现为爬升初期的反向力和相位滞后（接近非最小相位）。

### 3.3 控制环：外环 P + 2 m lookahead，没有 depth 积分

指导律在 `src/auv_lowlevel/batch/guidance.py:184–190`：

```text
path_pitch = atan2(-tangent_z, horizontal_norm)
pitch_correction = atan2(vertical_error, vertical_lookahead)
desired_pitch = path_pitch + vertical_gain * pitch_correction
```

本 cell：`vertical_gain = 1`，`vertical_lookahead = 2 m`，
`pitch_gain = 4`，`pitch_rate_gain = 1`。外环没有 depth 积分项；
要维持一个非零 pitch 来爬升，必须存在一个持续的 pitch error 去产生
持续的 stern 偏角。再加上 `p_nav` 噪声和 15°/s 的 command rate limit，
实际纵向环的相位滞后远大于理想的一阶 `u/L` 估算。

### 3.4 pitch 环本身还有一个 0.27 Hz 的欠阻尼模态

同 cell 的 pitch 统计：

| 指标 | 数值 |
|---|---:|
| guidance target pitch RMS | `1.38°` |
| rate-limited command pitch RMS | `0.74°` |
| actual pitch RMS | `1.43°` |
| actual vs target pitch error RMS | `2.00°` |
| actual vs limited command pitch error RMS | `1.36°` |
| actual pitch 频谱主峰 | `0.277 Hz` |
| vertical error 频谱主峰 | `0.031 Hz` |

pitch 环在约 `0.27 Hz`（接近 REMUS100 的俯仰自然频率）有一个欠阻尼模态；
它的垂直位移贡献约厘米级，不是 20 cm 纵向误差的主因。主因是 `0.03–0.05 Hz`
的慢捕获模态。但这个 0.27 Hz 模态说明：当前 PD 增益/阻尼并不理想，
全通道 residual 若继续追 pitch error，会把这个模态放大。

## 4. 为什么 `P_ALL_075` 只换来 tradeoff

`P_ALL_075` 确实有纵向权限，而且用了很大的 stern residual：

- stern residual 平均 `6.20°`，p95 `14.57°`，有效 p95 到 `15°` 上限；
- 与 pitch error 相关 `0.60`，与 vertical error 相关 `-0.32`；
- 收敛时间从 `19.8 s` 降到 `9.96 s`；
- 但 vertical p90 从 `0.2957 m` 变成 `0.3038 m`，末端误差从 `0.055 m`
  变成 `0.168 m`。

这不是“没有权限”，而是“权限用在了 pitch 上，没有形成 depth 的积分/终端
约束”。策略学到的是：看到 pitch error 就加大 stern；这样能更快抬头，
但会在末端过冲。再加上它同时大幅改 rpm（均值 `189 rpm`，p95 `480 rpm`），
速度/推力也参与了这个 tradeoff，不能把全部效果单独归给 stern。

## 5. Noisenet 假设的裁决

本轮没有证据支持“大 residual 被 NoiseNet 主导”：

- `P_RUD_075` 的学习器 stern residual 是 `6.4e-6°`，不是大 residual；
- `P_ALL_075` 的 stern residual 与 pitch error 相关 `0.60`，是结构化的
  确定性响应，不是随机噪声；
- `P_ALL_075` 的 pitch rate std 约 `0.076 rad/s`、heave velocity std 约
  `0.044 m/s`，没有出现噪声主导应有的高频抖动爆炸；
- 纵向误差的慢模态可以用初始偏差、p_nav 噪声和间接 depth 通道解释，
  不需要引入 NoiseNet。

更准确的说法是：**p_nav 观测噪声会让确定性策略的 residual 看起来“抖”，
但 residual 本身仍然是在追 pitch error；这不是 NoiseNet 主导。**

## 6. 对下一步的建议

当前 Goal173 已按用户要求暂停，以下只作为下一个预注册 goal 的输入，
不在本 gate 中途改算法/奖励：

1. **不要把 `P_RUD_075` 的纵向当作 RL 失败。** 它的动作空间没有 stern/rpm，
   纵向本来就只能是 ALOS 的水平；它已经证明 rudder-only 对 cross-track/3D route
   有稳定收益。
2. **先做 observability 对照。** 同一 checkpoint、同一 dev seeds，比较
   `p_nav = eta`（或 nav noise = 0）与当前 `p_nav` 的纵向 RMS/收敛时间。
   如果 oracle 状态下纵向明显改善，优先解决状态估计/噪声，而不是换算法。
3. **单独审计纵向控制结构。** 记录并报告 `e_v`、pitch error、q、w、stern、
   stern 饱和率、从初始偏差到 `|e_v| < 5 cm` 并保持的时间；把“捕获段”
   和“跟踪段”分开统计。
4. **奖励/动作参数化只改纵向通道。** 如果要做 depth 相关 residual，优先考虑
   pitch/depth 的积分或前馈项，而不是继续放大 all-channel stern；`P_ALL_075`
   的 `15°` p95 已经说明当前 scale 过于激进。
5. **评价固化。** 以后同时报告 vertical bias、p50/p90、捕获时间、符号持续率、
   五段和 keypoint 指标；不要只用 success，也不要用整段 RMS 掩盖前 40% 的
   初始捕获瞬态。

## 7. 复现命令

```bash
ssh -p 43127 yhx@152.136.183.51
cd /data/yhx/underwater-robotics/src/plan_p1/goal153_reward_curriculum_20261111

# trace 在 policy_seed86201.npz，不在 trace.npz
python - <<'PY'
import numpy as np
p = "artifacts/lowlevel/goal173_rudder_only_residual_20261112/eval/P_RUD_075/default/current10/policy_seed86201.npz"
d = np.load(p)
print(d.files)
PY
```

关键代码位置：

- `src/auv_lowlevel/batch/dynamics.py:563–587`：stern heave / pitch moment；
- `src/auv_lowlevel/batch/controllers.py:212–256`：pitch PD；
- `src/auv_lowlevel/batch/guidance.py:180–190`：vertical LOS；
- `src/auv_lowlevel/batch/path.py:320–332`：vertical error 定义；
- `scripts/evaluate_pc_ppo.py:3093–3098, 3160–3169`：true `eta` projection 与 trace；
- `src/auv_lowlevel/batch/sensors.py:287–299`：`unbiased_noise_v1` 的 nav noise。

