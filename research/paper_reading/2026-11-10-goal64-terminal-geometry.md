# Reading note：goal-plane crossing 与 terminal capture 的几何边界

日期：2026-11-10<br>
关联 Goal：Goal64<br>
用途：解释 Goal63 strict125 true capture 丢失，并约束下一步 terminal rule

## 1. 本地证据链

- Goal40/41：protocol success 与 true `eta` capture 存在 discordance；
- Goal42/43：nav age/delay 不能完全解释该偏差；
- Goal59：deterministic residual 已饱和，NoisyNet 不是全局主因；
- Goal60/61：zero-residual 也复现 `p_nav/eta` terminal 偏差；
- Goal62：terminal-row 静态 margin 在首次 crossing 截断，测量退化；
- Goal63：闭环抑制 GOAL 后 default 通过、strict125 true retention 失败；
- Goal64：strict125 的 6 个丢失 true-success 全部为 plane miss，minimum
  `p_nav` 中位数 `1.172 m`，perp offset 中位数 `1.065 m`。

## 2. 方法学要点

“收紧 GOAL 半径”不能只改一个半径。必须把三类边界分开：

1. GOAL 的判定半径；
2. plane-miss guard 的参考半径；
3. success 与 true `eta` capture 的标签。

Goal63/64 固定 plane-miss guard 为原 `r0`，因此 candidate 在进入更紧 GOAL 球
之前仍可能因穿过 goal plane 而终止。Goal64 的 trace 证明这正是 strict125 丢失
true capture 的主要路径，而不是“再飞久一点就能捕获”。

## 3. 为什么不是 residual/NoisyNet

Goal64 的 lost 与 retained 组 last-100-step residual L2 中位数几乎相同
（`0.351` vs `0.352`），material saturation 也没有更高，scale/rate clip 都是
0。两组动作都接近饱和，因此普通“动作饱和”也不能解释丢失组与保留组的差异。

NoisyNet 更没有重新成为解释：Goal59 的 deterministic mean 已饱和，Goal64 的
forced zero-residual 同样以 plane miss 丢失。当前证据不支持用关闭噪声、缩小
噪声、增加 residual 或换 SAC/REDQ/DroQ 来救这个 Gate。

## 4. 与 terminal capture 文献的关系

终端捕获通常同时涉及位置半径、方向/平面穿越和到达后的停留条件。只用球形
半径会在两种情况下产生歧义：

- 估计位置先进入球面，但真实 `eta` 尚未进入；
- 真实轨迹先穿过 goal plane，但仍在球面外侧。

Goal64 的证据说明第二种情况在 strict125 下占主导：横向 offset 约一米时已经
穿过 plane，收紧半径没有机会生效。因此下一轮应测试 plane-aware 或
`eta`-aware 的捕获条件，而不是继续放大残差。

## 5. 下一步边界

- 新预注册 eval-only terminal rule 诊断；
- 候选条件可以同时要求 `p_nav/eta` 半径、perpendicular offset 和短暂 dwell；
- 不重跑 Goal63 margin，不修改正式 reward/controller/checkpoint；
- 不打开 held-out `100–149`；
- 只有明确 optimization/exploration 证据才用 SAC 同协议对照；
- 只有明确 sample efficiency 证据才考虑 REDQ/DroQ。

## 6. 相关项目文档

- `doc/goals/goal63_result_20261110.md`
- `doc/goals/goal64_result_20261110.md`
- `research/blockers/2026-11-10-goal63-strict125-capture-loss.md`
- `research/paper_reading/2026-11-10-goal63-terminal-suppression-partial.md`
- `research/paper_reading/2026-11-10-goal62-terminal-boundary-invalid.md`
