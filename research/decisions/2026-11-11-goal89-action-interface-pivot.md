# Decision：Goal89 动作接口修正为 absolute residual-only

日期：2026-11-11<br>
决策：`pivot`（不改变 PPO 主算法族，不改变 Gate 假设）

## 理由

Goal87/88/89 的 frozen-BC + residual arm 使用 actuator wrapper，实际动作是
`ALOS + scale * (BC + residual)`，与预注册的 `BC + residual` 不一致。该问题属于
动作组合语义，不属于网络结构、reward、terminal 或优化器问题。

## 新协议

- `pc.action_interface=absolute`；
- `base_policy.mode=frozen_pc_bc_residual`；
- `base_policy.output_mode=residual`；
- policy 只采样 normalized residual；
- vector env 组合 `frozen_BC + terminal_gate * residual`；
- 评估的 `policy` 模式使用同一组合；`frozen-base` 只输出 BC；
- 仍固定 PPO、`z_t64`、frozen D3 BC、reward、terminal、success、goal radius、
  dev seeds 和 held-out 规则。

## 不做什么

- 不把旧 Goal89 结果写成 Gate FAIL；
- 不换 SAC/REDQ/DroQ；
- 不放宽 path/success 门槛；
- 不打开 held-out；
- 不自动 commit/push。

