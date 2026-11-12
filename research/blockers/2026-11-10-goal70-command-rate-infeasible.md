# Blocker：crossing 前 command rate 需求超出 shaping envelope

日期：2026-11-10<br>
关联 Goal：Goal70<br>
状态：`COMMAND_RATE_INFEASIBLE`<br>
上游：Goal69 `PRECROSSING_MULTICHANNEL_TRACKING_LAG`

## 1. 现象

Goal70 只读复用 Goal67 的 10 条 rich trace，在 plane crossing 前 1 s 检查
heading/pitch/speed command rate、actual rate、phase lag、执行器 limit 和
terminal guidance。default 与 strict125 都判为 `COMMAND_RATE_INFEASIBLE`。

主证据：

| scenario | group | n | heading cmd p90 | heading exceed | pitch cmd p90 | pitch exceed |
|---|---|---:|---:|---:|---:|---:|
| default | lost | 9 | `1.457` | `0.961` | `0.441` | `0.529` |
| default | retained | 61 | `0.677` | `0.196` | `0.416` | `0.333` |
| strict125 | lost | 7 | `3.489` | `0.667` | `0.900` | `0.902` |
| strict125 | retained | 23 | `0.677` | `0.176` | `0.585` | `0.471` |

`command_rate_limit_enabled=false`：guidance command rate limiter 实际关闭，
配置中的 `0.5236/0.2618/0.25` 限速没有生效。lost 的 heading command rate
P90 在两 scenario 都超过 `1.25 × 0.5236`，且 exceed fraction 高；strict125
的 pitch 也过门槛。

## 2. 已排除项

| 解释 | 状态 | 理由 |
|---|---|---|
| NoisyNet/residual 主导 | 排除 | deterministic eval；residual/base lost 不高于 retained；zero-residual 仍丢 default 4、strict125 1 |
| 执行器 rate saturation | 排除 | `fin_rate_at_limit_fraction=0`；material saturation lost 反而更低 |
| translational command infeasible | 排除 | reachability infeasible fraction 两组都是 0 |
| speed 通道共同恶化 | 排除 | speed error gap 为 `-0.005/-0.111 m/s`，负控通过 |
| phase lead/bandwidth 是 lost 判别因素 | 不追认 | heading corr `0.235/0.328 < 0.50`；pitch gain/lag lost 不比 retained 更差 |
| terminal guidance 额外需求（default） | 不追认 | terminal fraction 两组都是 1；default late heading 在 retained 也上升 |
| 优化/探索失败 | 未证明 | 失败发生在 frozen checkpoint 的 command demand，不是训练曲线 |
| 样本效率失败 | 未证明 | 本 Goal 未训练 |

## 3. 解释

当前最合理的机制是：crossing 前的 command heading 需求本身变化过快，超过已配置
但未启用的 guidance command rate envelope；controller 和执行器没有到 fin rate
limit，闭环也没有低速卡死。这个结论是观测性的，不能直接证明打开 limiter 一定
改善闭环，因此下一步必须是单因素、预注册的 command-rate-limit 对照。

注意：H70-B 的正式否定不等于“controller 没有任何带宽限制”。实际 heading gain
只有 `0.003/0.007`，说明全局响应幅值很弱；但 retained 同样弱，所以它不能解释
这次 lost/retained 的差异。

## 4. 两个可证伪 hypothesis

### H71-A：启用现有 command rate limiter 会降低 tracking error 并改善 true retention

支持：同 checkpoint、同 dev seeds 下，限速 arm 的 heading/pitch tracking error
下降，true retention 或 plane miss 至少一个方向改善，且 timeout/safety 不恶化。

证伪：limiter 开启后 tracking error 不下降，或 true retention 不改善，或收益只
来自更多 false success。

### H71-B：失败来自 controller 带宽，而不是 command demand

支持：即使 command 已被限制在 `0.5236/0.2618 rad/s` 内，actual rate 仍然固定
滞后或幅值不足，并伴随 tracking error 不降。

证伪：限速后 actual rate 能跟上 command，tracking error 随需求下降而下降。

## 5. 最便宜 falsification

复用 frozen Goal59 checkpoint 和 dev `20–59`，只改
`command_rate_limit_enabled`，保持 reward/gamma/seed/controller/checkpoint/eval
split/held-out 不变；跑 default+strict125 的限速 arm，与 Goal67 baseline 逐 seed
比较 true retention、plane miss、tracking error 和 timeout/safety。

## 6. 决策

`decision = continue`：继续同一研究主线，不换 RL 算法，不启动 SAC/REDQ/DroQ。
下一步预注册 Goal71 的单因素 command-rate-limit 闭环 arm。

## 7. 证据

- `doc/goals/goal70_command_feasibility_phase_lead_20261110.md`
- `doc/goals/goal70_result_20261110.md`
- `artifacts/lowlevel/goal70_command_feasibility_phase_lead_20261110/goal70.json`
- `artifacts/lowlevel/goal70_command_feasibility_phase_lead_20261110/goal70.md`
- 服务器：
  `/data/yhx/underwater-robotics/src/plan_p1/goal60_action_reward_terminal_readonly_audit_20261110/artifacts/goal70_command_feasibility_analysis/`
