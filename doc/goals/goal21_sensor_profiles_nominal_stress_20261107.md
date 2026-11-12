# Goal21：传感器误差 profile 拆分（nominal / stress）

日期：2026-11-07

状态：**已实现；待用户验收**

## 1. 背景

已有的 `realistic_calibrated` 在 2026-11-03 主要用于修复导航偏置和 USBL 融合问题，
但其传感器随机范围仍继承旧 `realistic` 的长尾：例如海流估计 `tau=1–10 s`、
延迟 `0–2 s`、更新率下限 `0.1 Hz`，导航延迟上限 `0.5 s`，USBL 噪声上限 `5 m`。
这些值适合鲁棒性压力测试，但不适合作为“实机 actor 输入观测”的默认 nominal 域：

- 10 s 的海流估计时间常数已经比 8 s 海流变化周期更长，会把时变估计抹平；
- 2 s 估计延迟再叠加 `0.6 s` age gate，actor 基本拿不到新鲜估计；
- 0.1 Hz 更新率意味着 10 s 才更新一次，低于任务中的海流变化速度；
- 0.5 s 导航延迟在 2.4 m/s 下对应约 1.2 m 位移，已接近 strict125 半径。

本 Goal 不删除任何字段，而是把同一组字段拆成 nominal 与 stress 两套 profile，并保留
旧行为入口用于历史复现。

## 2. 新 profile 契约

### `realistic_calibrated`：nominal 默认域

用于以后实机 actor 输入观测相关实验，作为新的默认候选。

| 字段 | nominal 范围 | 说明 |
|---|---:|---|
| `current_est_tau_s` | `1.5–3.0 s` | 小于 8 s 海流周期，保留可跟踪的时变成分 |
| `current_est_delay_s` | `0.0–0.3 s` | 与 age gate 兼容 |
| `current_est_hz` | `1.0–3.0 Hz` | 下限不再慢于海流变化 |
| `current_est_noise_mps` | `0.005–0.02 m/s` | 5 cm/s 极端值移入 stress |
| `current_est_bias_mps` | `0.005–0.02 m/s` | 3 cm/s 极端值移入 stress |
| `current_est_outage_duration_s` | `1.0–3.0 s` | 20 s episode 内不再半程失锁 |
| `nav_delay_s` | `0.02–0.15 s` | 控制 strict125 下的滞后位移 |
| `nav_bias_m` | `0.1–0.5 m` | 2 m 偏置移入 stress |
| `usbl_noise_m` | `0.3–1.5 m` | 5 m 单次测量移入 stress |
| `usbl_delay_s` | `0.1–0.3 s` | 0.5 s 移入 stress |
| `dvl_bottom_outage_duration_s` | `1.0–3.0 s` | 10 s 失锁移入 stress |
| `usbl_outlier_jump_m` | `5.0–10.0 m` | 10–25 m 跳变移入 stress |
| `usbl_outlier_prob` | `0.005–0.01` | 3% 概率移入 stress |

保留的标定语义：

- `nav_usbl_gain=0.03`；
- `nav_bias_calibration_fraction=0.8`；
- 其余未列出的 DVL/IMU/current walk/nav drift 字段沿用旧 `realistic` 合理区间。

对应指导层配置使用：

```text
guidance.current_estimate_max_age_s = 0.9
```

原因是 nominal 的 `current_est_hz` 下限为 `1 Hz`；若仍使用 `0.6 s`，1 Hz 估计会天然
频繁超时。只有显式确认估计率始终不低于 2 Hz 时，才允许继续使用 `0.6 s`。

### `realistic_calibrated_legacy`：历史复现

与 2026-11-07 之前的 `realistic_calibrated` 完全同义：

- 旧 `realistic` 全部极值范围；
- `nav_usbl_gain=0.03`；
- `nav_bias_calibration_fraction=0.8`。

历史实验需要严格复现时，应显式把配置改成 `realistic_calibrated_legacy`，不要继续
使用已经重定义的 `realistic_calibrated`。

### `realistic_stress`：旧极端值压力测试

保留旧 `realistic` 的极端范围，包括：

- `current_est_tau_s=1–10 s`、`current_est_delay_s=0–2 s`、
  `current_est_hz=0.1–5 Hz`；
- `nav_delay_s=0.05–0.5 s`、`nav_bias_m=0.2–2.0 m`；
- `usbl_noise_m=0.5–5.0 m`、`usbl_delay_s=0.1–0.5 s`；
- `usbl_outlier_jump_m=10–25 m`、`usbl_outlier_prob=0.01–0.03`；
- `dvl_bottom_outage_duration_s=1–10 s`。

`realistic_stress` 只用于鲁棒性压力测试，不作为 nominal 训练统计或主成功率口径。

### 不变 profile

- `realistic`：原始旧 profile，保持原样；
- `strong`：原有更极端的压力 profile，保持原样；
- `clean`、`mild`：保持原样。

## 3. 新入口配置

- `configs/lowlevel/odarrl_control_pc_c3_random_realistic_v1_nominal_20261107.json`
  - `realistic_calibrated`，goal radius `2.0 m`，`current_estimate_max_age_s=0.9`；
- `configs/lowlevel/odarrl_control_pc_c3_random_realistic_v1_nominal_strict125_20261107.json`
  - `realistic_calibrated`，goal radius `1.25 m`，`current_estimate_max_age_s=0.9`；
- `configs/lowlevel/odarrl_control_pc_c3_random_realistic_v1_stress_20261107.json`
  - `realistic_stress`，goal radius `2.0 m`，`current_estimate_max_age_s=0.6`。

以后新的实机 actor 观测实验优先使用 nominal 配置；压力测试显式使用 stress 配置。

## 4. 运行中任务处理

修改前发现两个 SUPERMIMIC WM 数据导出任务正在使用
`configs/lowlevel/odarrl_control_pc_c3_random_realistic_v1_rewardscale001.json`。
处理原则：

- 不重启、不杀进程、不修改该正在使用的环境配置文件；
- 已启动的 Python 进程在内存中持有旧 profile，文件修改不会回溯影响它；
- 新 profile 和新入口配置只供后续新进程使用；
- 若需要复现旧导出数据，使用 `realistic_calibrated_legacy` 和旧配置的
  `current_estimate_max_age_s=0.6`。

## 5. 验证清单

- [x] `PROFILES` 同时包含 `realistic_calibrated`、`realistic_calibrated_legacy`、
  `realistic_stress`；
- [x] `SUPPORTED_SENSOR_PROFILES` 接受三个新/旧入口；
- [x] nominal profile 的 13 个推荐字段范围被单元测试锁定；
- [x] legacy 与 stress profile 保留旧极端范围；
- [x] 三个入口配置可被 `load_config` 正确加载；
- [x] nominal 默认配置使用 `current_estimate_max_age_s=0.9`；
- [x] 运行传感器、配置和 PC 相关定向测试：`98 passed`；
- [x] 运行 ruff check：`All checks passed!`；
- [ ] `ruff format --check` 仍报告 `sensors.py` 和测试文件中的既有格式漂移；本次只
  新增局部代码，未重排整文件，避免制造无关 diff。

## 6. 后续约束

- 不在 nominal 与 stress 的统计之间混用成功率和 RMSE；
- 新训练、BC 重采和正式评估必须显式记录 `sensors.profile`；
- 若后续实机数据表明 nominal 仍偏乐观，应新增版本号，不要静默修改
  `realistic_calibrated` 的历史语义；
- 任何改变 profile 默认范围的提交都必须同时更新本文件和回归测试。
