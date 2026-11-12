# Blocker：pre-crossing tracking lag 是 heading+pitch 多通道，不是 residual/NoisyNet

日期：2026-11-10<br>
关联 Goal：Goal69<br>
状态：`PRECROSSING_MULTICHANNEL_TRACKING_LAG`<br>
上游：Goal67 `TERMINAL_AUTHORITY_LIMITED`、Goal68 `PRECROSSING_TRACKING_LAG`

## 1. 现象

Goal69 只读复用 Goal67 的 10 个 rich trace，主窗口为 crossing 前 1 s，2 s 只作
持续性检查，没有新增 rollout、没有训练、没有打开 held-out。结果：

- default：lost `9`、retained `61`；lost 1 s heading error `0.930 rad`，
  pitch error `0.618 rad`，speed error `0.493 m/s`；
- strict125：lost `7`、retained `23`；heading `0.364 rad`，pitch `0.596 rad`，
  speed `0.169 m/s`；
- command-goal cosine `0.980/0.983`，但 actual-command cosine 只有
  `0.294/0.659`；
- pitch gap（lost-retained）`0.405/0.442 rad`，两个 scenario 都支持 H69-B；
- speed gap 不支持：default `-0.005 m/s`，strict125 `-0.111 m/s`；
- default 的 true eta 在 crossing 前进入 `r_tight` 为 `4/9`，p_nav 为 `0`，
  gap `0.240 m`；strict125 只有 `2/7`，低于 `0.40`。

## 2. 关键解释

失败发生在 frozen controller 的 command-to-velocity tracking，而不是优化或
探索。command 仍然指向 goal，actual 地速也没有低到卡死；但 heading 和 pitch
的实际响应都跟不上 command。pitch 通道在 default/strict125 都独立过门槛，因此
总判定为多通道 tracking lag。

H69-A 的 1 s 原始证据很强（heading error `0.930/0.364 rad`，gap
`0.711/0.199 rad`），但预注册的 2 s 持续性守卫要求 `2s/1s >= 0.75`，实际为
`0.605/0.295`，所以正式判 A=false。1 s 窗口有约 50 行，中位数已经很大，不是
单帧抖动；这里记录为“late heading divergence”的描述性证据，不能追认 A。

预注册正文的 `retained - lost gap` 在 lost 更差时字面为负，和假设语义冲突；
实现按“lost 比 retained 更差的幅度”取 `lost - retained`，阈值未变，并在服务器
正式测量前固定。H69-A 在两种符号下都不通过；H69-B 的正式判定依赖“差距幅度”
解释，这一点已在 result 中显式记录。

lost 的 command pitch 本身很大（`0.875/1.061 rad`），command pitch rate
`0.277/0.521 rad/s`，actual pitch rate `0.164/0.199 rad/s`；material saturation
在 lost 反而更低，scale/rate clip 都是 `0`。因此不能写成执行器绝对饱和，更
像是 command feasibility / command shaping 与 tracking 的组合。

## 3. 已排除项

| 解释 | 状态 | 理由 |
|---|---|---|
| pre-crossing closure authority 不足 | 已排除 | Goal68 deficit 为负，deficit fraction `0` |
| command 方向错 | 排除 | command-goal cosine `0.980/0.983` |
| 低速卡死 | 排除 | actual ground speed `2.043/1.898 m/s` |
| surge tracking 缺口 | 不支持 | speed gap `-0.005/-0.111 m/s` |
| NoisyNet/residual 主导 | 排除 | deterministic eval；residual/base lost `0.109/0.181`，retained `0.200/0.176`；zero-residual 仍丢 4/1 |
| 优化/探索失败 | 未证明 | frozen policy 的 command 已对准 goal，失败在 tracking |
| 样本效率失败 | 未证明 | 本 Goal 未训练；失败发生在 frozen checkpoint 的闭环状态 |

## 4. 下一组可证伪 hypothesis

### H70-A：command rate/feasibility 是主因

若 crossing 前 heading/pitch command rate 系统性超过 actuator rate limit，或
command shaping 在最后 1 s 产生不可跟踪的阶跃，则 tracking lag 主要由 command
feasibility 驱动。证伪：若把 command rate 限制到实际可达范围后 tracking error
不下降，则 command feasibility 不成立。

### H70-B：controller phase lead/bandwidth 是主因

若 command rate 本身可达，但 actual 响应存在固定相位滞后/带宽不足，则 controller
phase lead/bandwidth 是主因。证伪：若 actual rate 已跟上 command rate，只是
幅值比例不同，则带宽解释不成立。

### H70-C：terminal guidance 在最后 1 s 产生额外 heading/pitch 需求

若 default 的 p_nav/terminal guidance 在 crossing 前引入额外 heading/pitch
需求，且 strict125 不出现同幅度需求，则 terminal guidance 是叠加分支。证伪：
若 default/strict125 的 command rate 分布相同，或 p_nav 分支在去掉 tracking
因素后消失，则 C 不成立。

## 5. 最便宜 falsification

复用 Goal67 rich trace，不训练、不新增 rollout：

1. 计算 crossing 前 1–2 s 的 command heading/pitch rate 分布与其分位数；
2. 对照 actuator rate limit、actual yaw/pitch rate 和 material saturation；
3. 用 rate-limited command 做只读重放/近似，比较 tracking error 是否下降；
4. 分开 default 的 p_nav/terminal 分支与 strict125 的非 p_nav 分支；
5. 保持 frozen checkpoint、dev `20–59`、CPU/float64、`endpoint_stop` 和
   held-out `100–149` 关闭。

## 6. 决策

`decision = pivot`：停止把 NoisyNet/residual 作为当前失败线的解释，进入
command feasibility / phase-lead 诊断。算法切换门仍未触发：

- 没有 optimization/exploration 证据，不启动 SAC；
- 没有 sample-efficiency 证据，不启动 REDQ/DroQ；
- 闭环在约 `1.9–2.0 m/s` 运动，不是“residual 大但闭环不动”，不先改动作参数化
  或 reward；
- 只有 Goal70 的新预注册 Gate 明确指向这些条件，才另开对应算法对照。

## 7. 证据

- `doc/goals/goal69_result_20261110.md`
- `artifacts/lowlevel/goal69_precrossing_tracking_channels_20261110/goal69.json`
- `artifacts/lowlevel/goal69_precrossing_tracking_channels_20261110/goal69.md`
- 服务器：
  `/data/yhx/underwater-robotics/src/plan_p1/goal60_action_reward_terminal_readonly_audit_20261110/artifacts/goal69_precrossing_tracking_analysis/`
