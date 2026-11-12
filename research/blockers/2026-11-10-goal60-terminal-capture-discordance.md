# Blocker: Goal60 terminal success 与 true capture 系统性不一致

日期：2026-11-10<br>
Goal：Goal60<br>
状态：`OPEN_PIVOT_TO_TERMINAL_CAPTURE_ALIGNMENT`

## 1. 触发条件

Goal59 `FAIL_BOUNDED_RESIDUAL` 后，Goal60 按预注册做了 4 个 Goal59
checkpoint（240/752/880/1008）× default/strict125 × dev 20–59 的只读 trace，
并与 forced zero-residual 配对。H-C 在两个场景都触发，且所有 checkpoint 都
出现 success/capture discordance，因此进入 Research Escalation。

这不是依赖缺失、路径错误或一次 smoke 失败；trace 的 seeds、terminal 行、
finite、checkpoint SHA 和 held-out 隔离都已验证。

## 2. 现象与证据

### 2.1 success/capture 2×2

| scenario | success_capture | success_no_capture | no_success_capture | no_success_no_capture | discordance |
|---|---:|---:|---:|---:|---:|
| default | 54 | 58 | 3 | 45 | 0.381 |
| strict125 | 23 | 36 | 7 | 94 | 0.269 |

- default discordance 95% CI `[0.306, 0.456]`；
- strict125 95% CI `[0.206, 0.344]`；
- success_no_capture 的 true terminal distance 均值：default `2.261 m`，
  strict125 `1.540 m`；
- 同场景 success_capture 的 true distance 均值只有 `1.785 m` / `1.013 m`。

### 2.2 为什么不是动作饱和

- pooled saturation mean `0.833`，240/752/880 接近 1，1008 约 0.333；
- failed-success saturation 中位数差只有 `2.8e-5/default`、`1.1e-5/strict125`；
- saturation 与 terminal distance 的 pooled rho 为 `0.062/0.031`，CI 跨零；
- 752 有中等 rho `0.427/0.491`，但中位数差仍远低于 `0.05`。

因此 saturation 是策略的全局属性，不能解释同一 checkpoint 内的成功/失败分化。

### 2.3 为什么不是样本效率/优化

- residual-to-base ratio 仍为 `0.063/0.065`，不是“残差学不出来”；
- 训练曲线最后 10% 的 residual ratio 变化均值为 `-0.00022`，已平台化；
- 没有证据支持 SAC 的优化/探索分支，也没有证据支持 REDQ/DroQ 的样本效率分支。

### 2.4 reward composition 是弱信号

失败 episode 的 terminal-window 分解：

| scenario | path reward mean | command reward mean | total mean | command>0 | total>0 |
|---|---:|---:|---:|---:|---:|
| default | `+54.77` | `-71.69` | `-187.07` | 0.000 | 0.000 |
| strict125 | `+18.53` | `-69.38` | `-182.67` | 0.000 | 0.000 |

path 正项在失败 episode 中约 `55–60%` 出现，但 command 负项在 100% 失败
episode 中出现，total 也全为负。这不是“path reward 奖励了错误行为”的干净
证据，而是 **path 正项被 command tracking 惩罚淹没**。

## 3. 已排除项

- 不是 trace 数据错误：40 seeds、40 terminal 行、finite、SHA 一致；
- 不是 held-out 泄漏：`100–149` 未打开；
- 不是 checkpoint 选择：只用预注册 `final.pt`；
- 不是 NoisyNet：Goal59 已证明 mean-weight 饱和，sigma 未膨胀；
- 不是动作饱和主导：H-A 的 episode 级判别失败；
- 不是样本效率主导：训练曲线已平台化；
- 不是评估协议变化：dev 20–59、default/strict125、CPU/float64、
  `endpoint_stop` 均固定。

## 4. 可证伪 hypothesis

### H-C1：p_nav 终点偏差导致协议 success 高于 true capture

机制：`success` 使用 `p_nav` 的 goal distance，true capture 使用 `eta`；
p_nav 在终点附近系统性地更靠近 goal，导致 success_no_capture。

可证伪预测：在 success_no_capture episode 中，`p_nav` 到 goal 的距离应显著小于
`eta` 到 goal 的距离；偏差方向应指向 goal。若两者无差异，H-C1 不成立。

### H-C2：终止平面/半径口径与 true capture 不一致

机制：termination 的 goal plane/radius 与评估的 `eta` capture 半径不是同一个
量；即使 p_nav 无偏，success 也可能先于 true capture 触发。

可证伪预测：success_no_capture 的 `terminal_kind`、signed offset 和 true
distance 应显示“先过平面、后进入半径”的顺序；若 success 总在 true capture
之后触发，H-C2 不成立。

### H-B1：terminal reward composition 让 path 正项被 command 负项淹没

机制：terminal 窗口内 path-progress 为正，但 command tracking 惩罚更大，total
reward 仍为负，策略没有收到“继续捕获”的净正信号。

可证伪预测：在 frozen checkpoint 上只读重加权 terminal 窗口的 command/path 项，
应能让 failed episode 的 total reward 转为正或显著提高，同时不改动作；若
total 仍为负或指标不动，H-B1 不成立。

### H-A1：动作饱和是 episode 级失败主因

可证伪预测：saturation 与 terminal distance 正相关且 failed-success 饱和差
`>0.05`。Goal60 已证伪，不再作为主线。

## 5. 排查顺序

按 `数据 → 标签 → 评测 → baseline → 架构 → 文献`：

1. 数据/标签：trace seeds、terminal 行、SHA、finite 已通过；
2. 评测：success 与 capture 的 2×2 已对齐到 episode；
3. baseline：zero-residual 的 success/capture 也保留相同口径；
4. 架构：p_nav/eta 的终点偏差需要只读复算，不改策略；
5. 文献：见
   `research/paper_reading/2026-11-10-goal60-terminal-capture-estimator-reward.md`；
6. 最便宜 falsification：在现有 trace 上比较 p_nav/eta 终点距离和 signed
   offset，不训练。

## 6. 决策

`decision = pivot`。主线从“继续改 residual 参数化/换算法”转到
**terminal capture 对齐与 p_nav 终点偏差**；reward composition 作为第二优先级。
不启动 SAC/REDQ/DroQ，不重开 Goal27，不打开 held-out。

## 7. 证据

- `doc/goals/goal60_result_20261110.md`
- `artifacts/lowlevel/goal60_trace_audit_20261110/goal60_trace_audit.json`
- `artifacts/lowlevel/goal60_trace_audit_20261110/goal60_trace_audit.md`
- 服务器
  `/data/yhx/underwater-robotics/src/plan_p1/goal60_action_reward_terminal_readonly_audit_20261110/artifacts/traces/`
