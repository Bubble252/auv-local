# Goal29 decision: PIVOT

日期：2026-11-09<br>
决策：`pivot`，停止在 C3 actor 上通过 `--guidance-config` 测试终终端对准窗口。

## 触发

Goal29 的 1-seed paired screen 显示：candidate 提前进入 terminal phase，command
诊断最多变化约 `0.86 rad`，但 `eta`、`nu`、`p_nav` 和 `actual_actuator` 与
baseline 逐位一致。控制链审计确认 C3 actor 直接输出执行器动作；ALOS guidance
只进入 `observation.command`/reward，不进入执行器。因此 screen 为
`INVALID_CONTROL_CHAIN`，不是 H1 被证伪。

## 放弃的路线

- 不在 D3/G26-P2 的 C3 直接执行器策略上继续改 `--guidance-config`；
- 不跑 Goal29 的 40-seed formal；
- 不把 command trace 的 yaw error 变化写成控制或捕获收益；
- 不打开 held-out、不训练 residual、不换 seed 救 screen。

## 新路线

先选择**真正进入执行器**的链路，再重新预注册 terminal capture 实验：

1. B0/`zero-command-residual`：`ALOS -> PD/PI -> actuator`，可直接测试 alignment
   window，但 baseline 不是 D3/G26-P2；
2. 若必须保留 D3/G26-P2，则必须改变 actor 动作空间、增加可执行终端控制头，或重新
   训练终端策略；
3. 任何候选都必须 paired、先 1-seed screen、再 40-seed dev，并同时报告
   `success_protocol`、`success_eta`、horizontal-normal offset、yaw error、
   speed/saturation、timeout 和 safety。

## 下一步的可证伪假设

- H1：在 `ALOS -> PD/PI -> actuator` 链路上，前移终端对准窗口会降低 true-state
  horizontal-normal offset；若 actuator 和 eta 随窗口变化但横向 offset 不变，则
  H1 不成立。
- H2：D3/G26-P2 的终端横向 miss 是 actor 直接动作策略造成的，不能靠只改观测层
  guidance 修复；若存在不改变 actor 语义的可执行 projection 且能降低横向 miss，
  则 H2 不成立。

## 证据

- `doc/goals/goal29_terminal_horizontal_capture_20261109.md`
- `doc/goals/goal29_screen_result_20261109.md`
- `artifacts/lowlevel/goal29_screen_20261109/control_chain_audit/audit.json`
