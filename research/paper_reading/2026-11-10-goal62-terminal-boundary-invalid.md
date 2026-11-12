# Reading note：terminal boundary、停止时间与反事实退化

日期：2026-11-10<br>
关联 Goal：Goal62<br>
用途：解释为什么 terminal-row 静态半径反事实不能直接判 FAIL

## 1. 本地证据链

- Goal40/41：protocol success 与 true `eta` capture 存在 discordance；
- Goal42/43：nav age/delay 不能完全解释该偏差；
- Goal59：softsign + terminal gate 仍饱和，deterministic mean 不是 NoisyNet 主导；
- Goal60：4 个 final checkpoint 上 H-C 强支持，zero-residual 也有同类现象；
- Goal61：success_no_capture 中 `p_nav` closer/direction/order 均为 `1.000`，
  中位 gap `-0.200/-0.242 m`，zero-residual 同样复现；
- Goal62：静态收紧半径后 candidate success 全域为 0，说明 trace 在终止边界截断。

## 2. 方法学要点

这是一个**停止时间选择（stopping-time selection）**问题。数据只在事件
“`p_nav` 首次进入原半径”发生时被观测；更紧半径的事件只能在继续 rollout 中
出现。把已停止的 terminal row 重新打标签，相当于把后续未观测事件当作未发生，
会产生结构性偏倚。

因此：

- `candidate_success=0` 是先验可预期的边界退化，不是 radius margin 的闭环效应；
- 需要保留停止后的轨迹，或重新 rollout，才能估计 false-success removal 和
  true-success retention；
- 评估规则必须只依赖在线可得的 `p_nav`/radius，`eta` 只作离线标签；
- 不能直接改 `plant.goal_radius_m`，因为该字段也可能进入 reward/其他语义；
  应在 termination manager 上做 eval-only override。

## 3. 与已有 terminal 诊断的关系

Goal28/30/32 已经表明 terminal capture 与路径 RMSE 可以解耦，不能用 path 指标
替代捕获。Goal61 又表明偏差在 zero-residual 下仍存在，因此不应继续把问题归因
于 residual 策略、NoisyNet 或样本效率。Goal62 的贡献是排除一种便宜的离线测量
方案，而不是否定 terminal calibration 方向。

## 4. 对 Goal63 的要求

Goal63 应评估：

- frozen checkpoint，不训练；
- `default` 和 `strict125`；
- policy 与 forced zero-residual；
- dev `20–59`；
- 抑制 `GOAL` 终止直到更紧 `p_nav` 半径，或 timeout/plane-miss/safety；
- 报告 false success、true capture、success、timeout、plane-miss、extra steps、
  terminal `p_nav`/`eta` distance；
- 多个 scenario/checkpoint 分片并行，但合并时按 seed 对齐。

如果闭环抑制仍不能保留 true capture，再转向 estimator/terminal geometry；
只有明确证明优化/探索或样本效率是主因时，才允许新开 SAC 或 REDQ/DroQ 预注册
Goal。

## 5. 相关项目文档

- `doc/goals/goal28_result_20261109.md`
- `doc/goals/goal30_result_20261109.md`
- `doc/goals/goal32_result_20261109.md`
- `doc/goals/goal59_result_20261109.md`
- `doc/goals/goal60_result_20261110.md`
- `doc/goals/goal61_result_20261110.md`
- `doc/goals/goal62_result_20261110.md`
- `research/paper_reading/2026-11-10-goal61-terminal-alignment.md`
