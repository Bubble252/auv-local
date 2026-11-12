# Goal151 decision：near10_2 gate 通过，进入训练型 Goal152

**日期**：2026-11-11<br>
**决策**：`continue`<br>
**依据**：`doc/goals/goal151_result_20261111.md`

## 观察

- full T2：success 方向为正但 CI 跨 0，path RMSE 稳定 `+0.278 m`。
- near10_2：default `88/78`，strict125 `48/42`；path 相对 zero 为
  `-0.0007 / -0.0013 m`；相对 full 改善约 `0.278 m`，CI 完全为负。
- near10_2 的 2 m 内 residual RMS 为 `1.46`，不是“关掉残差”。
- shell10_3 也过自动 Gate，但 2 m 内 residual 为 0，机制不同，只做备份。
- 纵向 RMSE 未退化，Goal140 reward pilot 不触发。

## 判读

远处 residual 权限是 full T2 path 代价的直接来源；把权限限制到 10→2 m 后，success
收益保留、path 代价消失。该结论是 eval-only 机制结论，训练时启用 gate 是否仍成立，
必须由 Goal152 的 3-seed 1M screen 验证。

## 下一步

预注册 Goal152：

- 主变体 `near10_2`：`residual_terminal_gate_enabled=true`、`enter=10`、`full=2`、
  `learner_terminal_gate_mode=none`；
- 复用 Goal86 T2-SAC 协议：3 seeds `86201–86203`、1M transitions、dev40、
  default/strict125、policy vs zero；
- 不在训练中途改 reward；纵向恶化则按 Goal140 新开 reward pilot。
