# Blocker：terminal command 已对准，但 capture 仍丢失

日期：2026-11-10<br>
关联 Goal：Goal66<br>
状态：`TERMINAL_GEOMETRY_UNRESOLVED`<br>
上游：Goal64 `GEOMETRY_SUPPORTED`、Goal65 `DWELL_NOT_SUFFICIENT`

## 1. 现象

复用 Goal60 baseline 与 Goal65 `_dwell50_m025` candidate trace，只看每个 candidate
episode 最后 51 步和最近 true-`eta` 点：

- default：lost true `9`，retained `61`；lost `p_nav-only = 2/9 = 0.222`；
- strict125：lost true `7`，retained `23`；lost `p_nav-only = 3/7 = 0.429`；
- lost corrected-command LOS cosine 中位数 `0.977/0.982`，已经高度朝向 goal；
- lost speed-command ratio 中位数 `1.000/1.095`，但 retained 更高
  (`1.090/1.127`)，不能区分两组；
- lost 的 `p_nav` 最小距离中位数 `2.003 m / 1.254 m`，仍高于收紧半径
  `1.75 m / 1.00 m`；
- lost 的 plane-miss penalty 只在最后一帧出现，terminal bonus 从未触发。

预注册 H66-A/B/C 均未支持，总状态为 `TERMINAL_GEOMETRY_UNRESOLVED`。

## 2. 已排除项

| 解释 | 状态 | 理由 |
|---|---|---|
| `p_nav` 估计器单独截断 capture | 不充分 | pooled p_nav-only 只有 `0.222/0.429`；strict policy-only `3/6` 是事后边界，不是 Gate |
| terminal command 没朝真实 goal | 不支持 | lost LOS cosine `0.977/0.982`，且 retained - lost 为负 |
| terminal speed command 没下降 | 不充分 | lost 与 retained 都未明显衰减，retained 反而更高 |
| NoisyNet/residual 幅度主导 | 未支持 | 评估 `evaluate=True` 时不采样 NoisyNet 噪声；Goal64 residual L2 lost/retained `0.351/0.352`；zero-residual 仍丢失 5 个 episode |
| reward/terminal label 完全反号 | 未支持 | terminal bonus 只在 retained 出现，plane-miss 只在 lost 终点出现；属于结果伴随信号，未证明因果 |
| dwell 太短 | 已排除 | Goal65 等满 50 步仍不进入收紧球，窗口末端距离更大 |
| 样本效率不足 | 未支持 | 本 Goal 未训练；失败发生在 frozen policy 的终局几何 |

## 3. 新 hypothesis

### H67-A：terminal 修正量超出执行器/动力学可达包络

在最近 true-`eta` 点，若剩余横向/垂直修正所需的速度、转向率或俯仰/深度率超过
REMUS100 在当前速度下的可达包络，那么即使 command 方向正确，也无法在剩余时间内
进入收紧球。

可证伪预测：若 lost 组的所需转向半径、横向加速度和垂直速率都在可达包络内，且
retained 组没有类似的边界贴近，则 H67-A 不成立，应转向 controller tracking。

### H67-B：command 可达但实际状态跟踪滞后

若 command 在可行包络内，但实际 yaw/pitch/depth 状态没有按 controller 的额定
rate 跟上，则失败来自 PD/PI tracking、rate limit、饱和后的恢复或状态估计延迟。

可证伪预测：若从 trace 重建的 actual rate 已跟上 command，且 terminal 状态误差
仍主要由初始几何决定，则 H67-B 不成立，应回到 capture-region 几何。

### H67-C：捕获球与 terminal target 几何不匹配

若水平/垂直 offset 的组合使 `p_nav` 与 true `eta` 在最后窗口内都无法同时满足
半径，且实际状态可达，则问题在 terminal target 选择或 3D capture-region 定义，
而不是 action、reward 或训练算法。

可证伪预测：若存在可达的 command 轨迹，在同一窗口内能同时把 `p_nav` 和 true
`eta` 带入半径，则 H67-C 不成立。

## 4. 排查顺序

按 `数据 → 标签 → 评测 → baseline → 架构 → 文献`：

1. 数据：Goal66 已校验 SHA、seed、finite、dwell 和 trace metadata；
2. 标签：确认 true `eta`、`p_nav`、GOAL 球和 plane guard 的定义没有再次漂移；
3. 评测：只用 dev `20–59` 的最后 51 步，不打开 held-out；
4. baseline：比较 lost/retained 的 terminal reachability envelope；
5. 架构：检查 controller lag、速度/转向/深度 authority 和 actuator saturation；
6. 文献：terminal homing/docking 只用于约束 reachability 与 capture-region 设计。

## 5. 决策

`decision = pivot`：停止 `p_nav` 校准、command 方向、speed profile 和 reward 权重的
单因素微调，进入执行器/动力学 reachability 诊断。下一步必须新预注册 Goal67，
保持 dev `20–59`、frozen checkpoint、CPU/float64、`endpoint_stop` 和 held-out 关闭。

算法切换提醒：当前没有 optimization/exploration 或 sample-efficiency 证据，不启动
SAC/REDQ/DroQ。只有新预注册 Gate 明确证明优化/探索失败，才用 SAC 做同协议对照；
只有明确证明样本效率失败，才考虑 REDQ/DroQ。

## 6. 证据

- `doc/goals/goal66_result_20261110.md`
- `research/paper_reading/2026-11-10-goal66-terminal-reachability.md`
- `artifacts/lowlevel/goal66_terminal_estimator_command_20261110/goal66.json`
- 服务器：`artifacts/goal66_terminal_estimator_command/analysis/goal66.json`
