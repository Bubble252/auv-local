# Goal30 decision: PIVOT

日期：2026-11-09<br>
决策：`pivot`，停止 ALOS terminal alignment window 参数搜索。

## 触发

Goal30 在真正执行的 B0 链路上完成 40-seed dev formal。30 m 窗口相对约 35 m 的
参考路径过长，terminal phase 占轨迹约 82%，导致：

- default `success_protocol` `0.725 -> 0.600`，plane miss `0.275 -> 0.400`；
- strict125 `success_protocol` `0.500 -> 0.400`，plane miss `0.500 -> 0.600`；
- terminal eta offset 变差约 `0.10 m`，3D/vertical path RMSE 变差约
  `0.34/0.37 m`；
- terminal speed ratio 不变，saturation 没有升高。

这不是 Goal29 的接口错误，也不是权限/速度问题，而是“更早 goal seeking 破坏路径
跟踪”的机制问题。

## 放弃的路线

- 不再扫 `terminal_alignment_min_m/max_m`；
- 不再用更长窗口、换 seed、调 gain/reward 或放宽 Gate 救结果；
- 不重开 command residual PPO / supervised residual；
- 不打开 held-out `100–149`；
- 不把 `success_eta`（ever-capture）单独当成 terminal success。

## 新路线

`pivot` 到**保留路径跟踪的终端横向/垂直收敛控制**：

1. 保留 20 m 路径跟踪窗口，不把大部分轨迹切换成 goal seeking；
2. 在最后一段加入 cross-track / vertical error feedback 或随距离收缩的
   time-varying lookahead；
3. 显式报告 terminal horizontal/vertical offset、terminal yaw、speed、
   saturation、protocol/eta success 和 plane miss；
4. 先做离线/回放审计，再 1-seed screen，再 40-seed dev formal；
5. 新候选必须是单因子，并先写 Goal31 预注册。

## 证据

- `doc/goals/goal30_result_20261109.md`
- `doc/goals/goal30_screen_result_20261109.md`
- `research/blockers/2026-11-09-goal30-terminal-alignment-worsens-capture.md`
- `research/paper_reading/2026-11-09-goal30-terminal-capture-control.md`
- `artifacts/lowlevel/goal30_executed_terminal_capture_20261109/failure_audit.json`
