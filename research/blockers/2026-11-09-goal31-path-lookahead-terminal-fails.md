# Goal31 blocker：路径前视终端控制没有改善 true-state 终端捕获

**日期**：2026-11-09<br>
**Goal**：Goal31<br>
**触发**：预注册效果 Gate 失败；主指标不足且 terminal vertical offset/saturation
护栏失败。属于 required autoresearch 的 `gate_fail` 记录。

## 1. 现象

候选在同一个 20 m 终端窗口内，把 baseline 的“直接指向 goal”替换为“沿参考路径
前视点纯追踪”。控制链有效，40/40 seeds 的 `actual_actuator` 和 `eta` 都改变，
但 40-seed dev formal 没有产生 terminal capture 收益：

- `terminal_eta_capture`：default `0.125 -> 0.125`（delta `0.000`），
  strict125 `0.075 -> 0.100`（delta `+0.025`）；Gate 要求至少一个 scenario
  `+0.05`，未达到。
- terminal horizontal-normal offset paired median：`-0.0112 m` / `-0.0143 m`；
  Gate 要求 `<= -0.05 m`，未达到，CI 都跨零。
- terminal vertical offset paired median：`+0.1061 m` / `+0.2679 m`，
  bootstrap 95% CI 分别为 `[0.0844, 0.4029]` 和 `[0.1364, 0.4434]`，显著变差。
- terminal material saturation paired median：`+0.0396` / `+0.0347`，
  超过 `+0.03` 护栏。
- `success_protocol` 两个 scenario 都为 `-0.025`，`plane_miss` 都为 `+0.025`；
  没有净协议收益。
- 全程 path 3D RMSE 改善约 `-0.115 m`，path vertical RMSE 改善约 `-0.055 m`；
  说明候选确实保留了路径，但没有解决终点平面的 true-state 垂直/横向收敛。

控制链审计：default/strict125 均 40/40 seeds actuator 改变，最大差约
`90.9/91.0 rpm`，`eta` 最大差约 `5.24/5.35 m`。因此不是 Goal29 的
`INVALID_CONTROL_CHAIN`。

## 2. 已排除项

1. **不是窗口问题**：candidate 与 baseline 都使用 20 m
   `terminal_alignment_min_m/max_m`，没有扫窗口。
2. **不是接口问题**：command、actuator、eta 都改变；full trace 的 45 个数组
   默认关闭时与 Goal30 baseline 逐位一致。
3. **不是 reward/PPO/seed 搜索问题**：本 Goal 无训练，checkpoint、seed pool、
   eval split 均冻结。
4. **不是速度问题**：terminal speed ratio paired median 约 `-0.0002/-0.0001`。
5. **不是 timeout/safety**：两者都是 `0`。
6. **不是简单的“更早 goal seeking”**：terminal phase 占比只变化
   `+0.0034/+0.0056`，不是 Goal30 的 82% 路径切割。

## 3. 可证伪假设

### H1：终点平面需要显式 vertical/cross-track 反馈，而不是路径前视点

机制：路径前视点纯追踪优化的是路径投影误差；但 termination 的 plane miss 由
goal 平面的法向/垂直分量决定。前视点落在路径上，终端仍可能沿着路径的垂直相位
穿过平面，无法主动把 `eta` 的垂直误差压到 `goal_radius` 内。

若成立应看到：

- 在同样 20 m 窗口下，加入显式 terminal vertical/cross-track 误差反馈后，
  `vertical_eta_m` 和 `horizontal_normal_eta_m` 至少一个显著下降；
- `terminal_eta_capture` 提升且 protocol/plane miss/safety 不恶化；
- 改善不能由速度、saturation 或 path RMSE 单独解释。

若失败应看到：显式误差反馈也被控制器滞后/权限或状态估计抵消，terminal offset
不改善；此时停止继续改终端参考点。

最便宜证伪：用现有 40-seed trace 做离线误差分解，估计 termination 前
vertical/cross-track 可修正量；然后预注册一个固定参数的 terminal error-feedback
candidate，先跑 seed `41` screen，不训练。

### H2：terminal true-state 垂直误差由导航/状态估计与执行滞后主导

机制：当前 termination 用 `p_nav`，而本 Goal 的主指标用 true `eta`。如果
`p_nav` 与 `eta` 的 terminal offset 差异、以及 PD/PI 的 vertical 响应滞后已经
超过 goal radius，任何 reference 改动都不能单独修复 capture。

若成立应看到：

- terminal `eta - p_nav` 的垂直/横向差值在失败 seed 上接近或超过 miss；
- 相同 candidate 改变 reference 后，`p_nav` 的 plane 指标改善但 `eta` 的
  terminal offset 不变或变差；
- 执行器权限/速率或 sensor/estimator 时延是更直接的解释。

若失败应看到：即使在 `eta/p_nav` 差异较小的 seed 上，显式 reference 反馈仍能
稳定改善 terminal offset。

最便宜证伪：只读现有 Goal31 trace，做 seed-level `eta` vs `p_nav` 分解和
terminal vertical offset 相关分析；不新增 rollout。

### H3：path-lookahead 本身引入了 terminal vertical demand / saturation 代价

机制：前视点沿路径前进，终端会保留路径 vertical profile；当目标平面要求垂直
收敛时，参考点可能产生更大的俯仰/执行器需求，表现为 vertical offset 与
saturation 同时上升。

若成立应看到：

- candidate 的 terminal stern/rudder 绝对均值上升，saturation 上升；
- 去掉前视点、改用 goal-plane 垂直误差反馈后，saturation 回落且 vertical
  offset 改善；
- 或者在前视点中显式注入 vertical goal error 后，改善伴随饱和下降。

若失败应看到：在 saturation 不上升的 seed 上，vertical offset 仍不变，说明不是
前视点垂直需求主导。

最便宜证伪：现有 trace 的 stern/rudder/saturation 按 seed 分组，和
`vertical_eta_m` delta 做相关；不新增训练。

## 4. 研究顺序

按 AGENTS 与 `research/README.md` 的顺序：

1. 数据：核对 seed、trace、termination 时刻和 `eta/p_nav` 对齐；
2. 标签：区分 protocol/ever-capture/terminal capture；
3. 评测：只用 terminal capture + offset + 护栏；
4. baseline：Goal31 B0 的 40-seed formal；
5. 架构：先做 H2/H3 的只读诊断，再决定是否预注册显式 terminal error-feedback；
6. 文献：读 `research/paper_reading/2026-11-09-goal31-terminal-path-vertical-gap.md`。

## 5. Decision

`decision: pivot`

停止“路径前视点纯追踪”主线，不继续扫 lookahead 或窗口。下一步先做 H2/H3 的
只读诊断；只有在诊断给出可执行、可证伪的 terminal error-feedback 或
estimator/lag correction 后，才预注册 Goal32。不得用 Goal31 的 path RMSE 收益
冒充 terminal capture 收益，也不得打开 held-out `100–149`。
