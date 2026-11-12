# Reading note：terminal suppression 的 partial success 与几何边界
日期：2026-11-10<br>
关联 Goal：Goal63<br>
用途：解释 default 通过、strict125 true capture 丢失的机制边界

## 1. 本地证据链

- Goal40/41：protocol success 与 true `eta` capture 存在 discordance；
- Goal42/43：nav age/delay 不能完全解释该偏差；
- Goal59：deterministic residual 仍饱和，NoisyNet 不是全局主因；
- Goal60：H-C 强支持，zero-residual 也复现 terminal 偏差；
- Goal61：success_no_capture 中 `p_nav` closer/direction/order 均 `1.000`；
- Goal62：terminal-row 静态 margin 在首次 crossing 截断，测量退化；
- Goal63：继续 rollout 后，default 通过主 Gate，strict125 true retention
  `0.739`，plane-miss 上升。

## 2. 方法学要点

“抑制 GOAL 终止”并不等于“把 success 标签改小”。必须同时定义：

1. GOAL 的新半径；
2. plane-miss 的参考半径；
3. timeout/safety 是否保持；
4. success 与 true capture 的分母和配对方式。

本次把 GOAL 收到 `r0-0.25 m`，plane-miss 仍以原 `r0` 为 guard。这样正常接近
不会在原半径处被误判 plane-miss，同时 episode 仍有真实的 geometry 风险：如果
轨迹在到达更紧球前穿过 goal plane，仍会终止。

## 3. 为什么 default 通过而 strict125 不通过

strict125 的绝对 margin 同为 `0.25 m`，但相对半径比例更大（20% vs 12.5%）。
在更紧的球内，controller 需要更晚才满足 GOAL，plane crossing 的容错更小。
当前结果中 timeout 不增、plane-miss 增，extra steps 很小，符合“接近几何不够
精确”而不是“飞得不够久”的解释。

这不支持用更多采样、SAC、REDQ/DroQ 或更宽 margin 直接救 PASS；也不支持回头
改 Goal63 的预注册阈值。下一个便宜实验是记录 suppression rollout 的 terminal
前窗口，区分 plane crossing、p_nav bias 和动作饱和。

## 4. 与已有 terminal 诊断的关系

Goal28/30/32 说明 terminal capture 与路径 RMSE 可以解耦；Goal60/61 说明
p_nav/eta 口径错位不是 residual 特有。Goal63 进一步表明：在 default 下，
抑制终止能把部分 false success 转成 true capture，但 strict125 下会损失
true success。因此修复优先级应是 terminal geometry/approach 和标签口径，
而不是继续放大残差或换算法。

## 5. 下一步边界

- 不训练；只读记录 Goal63 candidate 的 terminal 前窗口；
- 统计 plane offset、`p_nav`/`eta` 距离、approach angle、residual/action 饱和、
  plane-miss 前 time-to-plane；
- 若 H64-A/H64-B 指向 geometry，再新开预注册 Goal 设计 dwell/terminal rule；
- 若 H64-B 显示动作饱和是主因，才回到动作参数化；
- 只有明确优化/探索或样本效率证据，才允许 SAC 或 REDQ/DroQ。

## 6. 相关项目文档

- `doc/goals/goal28_result_20261109.md`
- `doc/goals/goal30_result_20261109.md`
- `doc/goals/goal32_result_20261109.md`
- `doc/goals/goal59_result_20261109.md`
- `doc/goals/goal60_result_20261110.md`
- `doc/goals/goal61_result_20261110.md`
- `doc/goals/goal62_result_20261110.md`
- `doc/goals/goal63_result_20261110.md`
- `research/paper_reading/2026-11-10-goal62-terminal-boundary-invalid.md`
