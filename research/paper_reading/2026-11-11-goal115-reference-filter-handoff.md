# Goal115 reading note：reference 高频目标、rate limiter 与 plant response 的交接

**日期**：2026-11-11<br>
**关联**：Goal115 `REFERENCE_FILTER_LIMITED`<br>
**性质**：内部 trace 证据 + 串级控制概念对照；不引入新训练算法。

## 1. 观察

Goal115 首次把 guidance 的 raw target 与 limiter 输出拆开。strict C0 的 failure
显示：

- `clip_fraction` 中位 `0.963`，success control `0.931`；
- `limited_at_limit_fraction` failure `0.853`，success control `0.495`；
- `median_tracking_deficit` failure `0.135 rad/s`，success control `0.027 rad/s`；
- `target_rate_p95` 两组都在 `3.8 rad/s` 左右。

解释：raw target 本身在高频抖动，limiter 在两组中都大量工作。failure 的区别
更像是“limiter 输出更长时间贴上限，且 plant 仍跟不上”，而不是“只有 failure
被限幅”。

## 2. 控制含义

在一个串级 reference/attitude/plant 结构中：

1. raw guidance target 可能因路径投影、终点几何、估计状态和接近阶段增益而变化；
2. command rate limiter 把 raw target 变成可执行的 reference；
3. plant 的 pitch rate 和 stern authority 决定 reference 是否真正被跟踪；
4. residual 只能在 plant 可达范围内修正。

所以 `REFERENCE_FILTER_LIMITED` 的含义是“失效链路经过 filter”，不等于
“提高 filter 限幅必然改善”。如果 raw target 是高频噪声，提高限幅可能把噪声
直接灌入控制器；如果 plant 已接近 rate/authority 上限，提高 command limit 也
不会增加实际 pitch rate。

## 3. 对 residual/算法的影响

- Goal115 的 action mode 是 `frozen-base`，PPO 增量为零；
- trace 中的 residual 是 frozen BC residual；
- 没有 `RESIDUAL_DIRECTION_LIMITED`；
- NoisyNet 无参数，Goal108 已排除 NoisyNet 网络主导；
- 因此当前没有证据支持开 SAC 对照，也没有证据支持加 residual scale。

## 4. 下一步 reading 方向

Goal116 先做 eval-only rate-limit sweep，回答“filter 是否因果”：

- 提高限幅是否改善：filter 是瓶颈，且 raw target 的方向信息有价值；
- 降低限幅是否改善：raw target 高频噪声是主要危害；
- 关掉限幅是否更差：limiter 本身是必要的安全/平滑层；
- 四臂无差异：回到 plant response、action parameterization 和 reward/terminal。

只有在 filter sweep 仍不能解释、且 residual 方向正确但闭环不动的证据出现后，
才考虑 reward/terminal 或 SAC 同协议对照。
