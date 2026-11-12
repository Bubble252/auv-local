# Reading note: terminal success/capture 对齐

日期：2026-11-10<br>
关联 Goal：Goal61<br>
用途：决定是否继续 residual 主线，还是转向 terminal calibration/termination

## 1. 本地证据链

- Goal40/41：protocol success 与 true `eta` capture 存在系统性 discordance；
- Goal42/43：该偏差不能被记录到的 nav age 解释，扣除 delay 后仍有 residual
  calibration bias；
- Goal59：softsign + terminal gate 仍饱和，NoisyNet mean-weight 不是全局主因；
- Goal60：4 个 final checkpoint 上 H-C 强支持，zero-residual 也有同类现象；
- Goal61：在 Goal60 trace 上确认 `p_nav` closer/direction/order 在两个场景都是
  `1.000`，zero-residual 同样复现。

## 2. 方法学要点

在“success 使用 p_nav、capture 使用 eta”的定义下，success_no_capture 的
inside/outside 顺序几乎是标签恒等式。因此不能把 H-C2 的 `1.000` 当作独立的
因果证据；应把重点放在：

- terminal distance gap 的幅度与 bootstrap CI；
- nav error 向量与 goal 方向余弦；
- zero-residual 对照是否同向；
- 修复规则是否只依赖在线可得的量。

## 3. 为什么不转向 SAC/REDQ/DroQ

Goal59/60 的证据不是“residual 接近零、优化不足”或“样本效率不足”：

- 8/8 deterministic mean 已饱和；
- residual 非零，residual-to-base ratio 约 `0.063/0.065`；
- 训练曲线最后 10% 已平台化；
- 失败与 terminal success/capture 语义错位同向。

因此引入 SAC、REDQ 或 DroQ 不会修复标签/终止口径问题。只有新的预注册实验明确
显示优化或样本效率是主因时，才允许启动这些分支。

## 4. 下一项阅读/实验边界

下一 Goal 应优先比较只依赖在线观测的 terminal rule：

- 收紧 p_nav 半径；
- terminal dwell；
- nav uncertainty/age margin；
- success 与 true capture 的语义统一。

先在 frozen checkpoint 和 dev trace 上做离线反事实，不训练、不打开 held-out；
若离线规则没有足够的 false-success/capture-recall 收益，再考虑闭环验证。

## 5. 相关项目文档

- `doc/goals/goal41_result_20261109.md`
- `doc/goals/goal42_result_20261109.md`
- `doc/goals/goal43_result_20261109.md`
- `doc/goals/goal59_result_20261109.md`
- `doc/goals/goal60_result_20261110.md`
- `doc/goals/goal61_result_20261110.md`
