# Blocker：capture margin 不可用，terminal p_nav 方向偏差成立

日期：2026-11-10<br>
关联：Goal74 / Goal73 `COMMAND_DEMAND_NOT_SUPPORTED`<br>
状态：`MARGIN_UNUSABLE_BIAS_SUPPORTED`<br>
决策：`pivot`，转 terminal bias correction 单因素审计；不换 RL 算法

## 1. 现象

Goal73 发现 false-success 全部是 sphere capture：`p_nav` 刚好在 tight
radius 内，true `eta` 仍在 protocol radius 外。Goal74 只读重分类现有
Goal71 limiter-on episodes，检验“收紧 capture margin”能否作为修正方案。

## 2. 证据

| scenario | margin=0 success/true/false | margin=0.10 true retention | false 组 bias projection | true 组 bias projection |
|---|---:|---:|---:|---:|
| default | `128/97/31` | `0.0` | `+0.5591 m` | `-0.0155 m` |
| strict125 | `58/44/14` | `0.0` | `+0.5513 m` | `+0.0760 m` |

- 重分类 mismatch 为 `0/200`（两个 scenario），success 与 evaluator 的
  `PCTerminalKind.GOAL` 一致；
- `endpoint_stop` 使所有 success 的 terminal `p_nav` 余量只有
  `0.012–0.018 m`，因此任何 `>=0.10 m` 的离线 margin 都会把所有
  success 一起删掉；
- false-success 的 `p_nav-eta` 误差中位 `0.729–0.736 m`，true-success
  组只有 `0.375–0.379 m`；
- bias gap 为 `+0.5745 / +0.4753 m`，两个 scenario 同向。

## 3. 已排除

- 不是 command authority：Goal73 的 heading/pitch command rate 已顶到
  limiter 上下限，且 newly-lost saturation 不高；
- 不是 NoisyNet：正式评估使用 deterministic mean weights；Goal59 已证明
  mean-weight 本身饱和，Goal63/70 的 zero-residual 仍失败；
- 不是简单的“false-success 数量小幅波动”：false-success 占 success 的
  `24.2% / 24.1%`，且偏差是结构化的，不是零均值边界噪声；
- 不是可以靠 post-hoc margin 修掉的问题：margin 会同时清空 true-success。

## 4. 可证伪假设

### H74-A：收紧 margin 能减少 false-success 且保留 true-success

预测：`margin=0.25 m` 相对 `margin=0` 减少至少 2 个 false-success，且
true-success 保留率至少 50%。

结果：**证伪**。false-success 确实降为 0，但 true-success 也降为 0，
retention 为 0。margin 不可用。

### H74-B：`p_nav-eta` 误差沿 goal 方向有系统偏差

预测：false-success 组的方向投影中位数 `>= +0.10 m`，且比 true-success
组高 `>= +0.05 m`，两个 scenario 同向。

结果：**支持**。false 组 `+0.5591/+0.5513 m`，true 组
`-0.0155/+0.0760 m`，gap `+0.5745/+0.4753 m`。

## 5. 最便宜 falsification 与下一步

已执行的最便宜实验是 Goal71 episodes 的只读重分类，不新增 rollout。
下一个最便宜实验是 Goal75 的 **oracle bias correction**：用 true `eta`
计算 `p_nav-eta` 沿 goal 方向的投影并从 `p_nav` 中扣除，重算 true/false
success。若 oracle 能保留大部分 true-success 并消除 false-success，才
预注册可观测的 learnable bias predictor 或 conservative terminal
correction；否则回到动作参数化/reward/terminal reference point。

## 6. 文献与本地记录

- `research/paper_reading/2026-11-10-goal73-terminal-capture-estimator-bias.md`
- `doc/goals/goal74_result_20261110.md`
- `artifacts/lowlevel/goal74_capture_margin_audit_20261110/analysis/goal74.json`

## 7. 决策

`decision = pivot`

不采用 capture margin，不启动 SAC/REDQ/DroQ，不增加 controller
authority。下一步只做 terminal `p_nav` bias correction 的预注册单因素
审计；held-out `100–149` 继续关闭。
