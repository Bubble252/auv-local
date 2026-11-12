# Blocker: Goal58 off-policy 命令残差饱和并破坏终端捕获

日期：2026-11-09<br>
Goal：Goal58<br>
状态：`OPEN_PIVOT_TO_BOUNDED_RESIDUAL_AND_TERMINAL_CAPTURE`

## 1. 触发条件

Goal58 的预注册 Gate 在正式 3-seed / 5M transitions 协议下失败：没有任何 seed
达到 pre-goal 3D RMSE 改善 `>=0.10 m`，且 success/plane-miss 护栏同时失败。
这不是 smoke、依赖缺失或配置 typo，因此触发 Research Escalation。

## 2. 现象与证据

正式评估使用 dev `20–59`、CPU/float64、`endpoint_stop`，每个 seed 的
`final.pt` 与同一环境的 forced zero-residual 配对：

| seed | scenario | policy success / RMSE | zero success / RMSE | ΔRMSE | Δsuccess | Δplane-miss |
|---:|---|---:|---:|---:|---:|---:|
| 240 | default | 0.400 / 2.6097 | 0.725 / 2.6431 | +0.0334 | -0.325 | +0.325 |
| 240 | strict125 | 0.075 / 2.6003 | 0.500 / 2.6287 | +0.0284 | -0.425 | +0.425 |
| 280 | default | 0.125 / 2.7058 | 0.725 / 2.6431 | -0.0627 | -0.600 | +0.600 |
| 280 | strict125 | 0.025 / 2.7033 | 0.500 / 2.6287 | -0.0747 | -0.475 | +0.475 |
| 320 | default | 0.100 / 2.6291 | 0.725 / 2.6431 | +0.0140 | -0.625 | +0.625 |
| 320 | strict125 | 0.000 / 2.6272 | 0.500 / 2.6287 | +0.0014 | -0.500 | +0.500 |

timeout/safety 均为 `0`；所有 summary 和训练曲线 finite。residual 不是小：
applied residual abs mean 为 `1.087–1.097`，L2/base ratio 为 `44.6–48.8%`。

## 3. 已排除项

- **不是 zero-residual 基线错误**：修复后的 wrapper 与 provenance 回归通过，
  zero baseline 与 Goal58 修正后的 `2.6431/2.6287 m` 一致；
- **不是限幅吃掉残差**：scale clip 为 `0.0`，rate clip 只有 `5.7–8.3%`；
- **不是 held-out 泄漏**：`100–149` 未打开；
- **不是评估协议变化**：全部 40-seed、default/strict125、CPU/float64、最终 checkpoint；
- **不是 NoisyNet 评估噪声**：deterministic mean 本身在三个维度上 100% 饱和；
- **不是超时或安全终止**：timeout/safety 全为 0；
- **不是 checkpoint 选择**：只使用预注册 final checkpoint。

## 4. 可证伪 hypothesis

### H1：tanh/scale saturation 是充分原因

机制：command residual 输出经 `tanh` 后接近 `±1`，物理 scale
`[0.35 rad, 0.17 rad, 0.60 m/s]` 让 heading/pitch/speed 在每个控制步都得到最大
修正；把 residual 缩小应恢复 B0 行为并保住 success。

可证伪预测：在 S1 上把 eval-time multiplier 从 `1.0` 降到 `0.05` 或 `0.10`，
success 应恢复到 `>=0.695/0.470`，且至少一个场景的 pre-goal RMSE 改善
`>=0.10 m`。

结果：**证伪为充分原因**。`0.05` 的 default success 为 `0.700`，但 RMSE 只改善
`0.0285 m`；`0.10` 的 RMSE 改善 `0.0508 m`，success 降到 `0.650`；`0.50`
的 RMSE 改善约 `0.17 m`，success 却降到 `0.600`、plane-miss 升到 `0.400`。
因此 saturation 是放大器，不是唯一原因。

### H2：residual 的方向服务于 path tracking，而不是 terminal capture

机制：residual 优化 dense command/path 误差，能降低飞行中的 pre-goal path error，
但会在 goal tangent plane 上产生更大的横向/垂直 endpoint offset；plane-miss 是
独立失败通道。

可证伪预测：如果缩小 residual 只改变幅度而不改变方向，则 scale 诊断应出现
“path RMSE 改善但 success 不改善/plane-miss 上升”的区间；若所有 scale 的
RMSE 与 success 同向改善，H2 不成立。

结果：**支持**。scale `0.50` 改善 pre-goal RMSE（aggregate `2.6431→2.4731 m`），
但 success `0.725→0.600`、plane-miss `0.275→0.400`。这与 Goal30/31 的
“path RMSE 不等于 terminal capture”一致。

### H3：NoisyNet exploration 主导残差

机制：训练日志的 stochastic residual 很大，可能是 NoisyNet 噪声把动作推到边界；
评估时关闭噪声应回到小残差。

可证伪预测：deterministic mean action 应显著小于 noisy sample，且 final checkpoint
的 deterministic saturation fraction 低。

结果：**证伪**。1M screen 的 deterministic abs mean 为
`[1.0000, 0.9997, 0.5252]`；三个 final checkpoint 的三个维度 saturation fraction
都为 `1.0`。NoisyNet noise std 约 `0.535×` deterministic norm，但 deterministic
mean 本身已经饱和。final checkpoint 的 NoisyNet 参数也没有膨胀：hidden 层
`softplus(rho)` 均值约 `0.300`，output 层约 `0.100–0.105`，接近初始化
`0.3/0.1`。因此噪声可以影响探索，但不是确定性饱和的原因。

### H4：样本效率不足，换更强 off-policy 算法即可

机制：DDPG/TD3 replay 利用率仍不足，REDQ/DroQ 或 SAC 能解决。

可证伪预测：若样本效率是主因，应有“残差小且不饱和、但训练不够”的证据；或者增加
更新/样本效率后残差应改善方向。当前没有这个证据。

结果：**未支持，暂不执行**。残差已经饱和且规则地破坏 success；应先修动作参数化
和 terminal capture，不能把换算法当成默认下一步。SAC/REDQ/DroQ 只能在新 goal 中
按预注册协议检验。

## 5. 最便宜 falsification

已执行：

1. 三个 final checkpoint 的 deterministic saturation probe（训练 seeds，200 steps）；
2. 1M screen checkpoint 的 deterministic vs NoisyNet probe；
3. S1 final checkpoint 的 eval-time scale sweep：
   `0, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00`，default/strict125 各 40 seeds。

没有新增长训练，没有修改正式 checkpoint，没有打开 held-out。

## 6. Decision

`decision = pivot`。

停止 Goal58 的 off-policy 命令残差当前实现，不继续扫 reward、gamma、seed、scale
或 checkpoint。下一步新开预注册 goal，优先做：

- bounded residual parameterization（限制输出、尺度、符号和速率，避免 `tanh` 饱和）；
- terminal-capture-aware objective/guardrail（goal-plane 法向、横向、垂直误差）。

只有当新 goal 的诊断明确指向优化/探索或样本效率时，才分别考虑 SAC 或 REDQ/DroQ。
当前不启动它们，也不重开 Goal27/Goal58。

证据：

- `doc/goals/goal58_result_20261109.md`；
- `artifacts/formal_s240/eval_*`、`artifacts/formal_s280/eval_*`、`artifacts/formal_s320/eval_*`；
- `artifacts/formal_final_saturation_probe.json`；
- `artifacts/noisynet_probe_screen_s240.json`；
- `artifacts/deterministic_physical_probe_screen_s240.json`；
- `artifacts/code_provenance_20261109.json`。
