# Goal25 残差 PPO 报告

日期：2026-11-08  
状态：**ppo_not_started**

## 结论

Goal25 的 D3 BC 底座没有通过 pre-goal 门禁，因此不满足 residual PPO 的前置条件。
没有执行 zero-residual 对照、1-seed screen 或 3-seed 扩展。

- D3 default pre-goal RMSE：`2.849170 m`，上限 `2.65 m`；
- D3 strict125 pre-goal RMSE：`2.829135 m`，上限 `2.70 m`；
- D3 default success：`0.7250`，通过；
- D3 strict125 success：`0.5000`，通过。

success 通过不能替代 pre-goal 主指标门禁。按合同停止，不扫 lr、residual scale、
reward 权重或 seed。

参考：

- `doc/goals/goal25_d3_repair_report_20261108.md`
- `/dev/shm/pc_v1_goal25_d2_repair_20261108/d3_gate.json`
