# Decision：Goal99 距离调度不支持，转 terminal prior preservation

日期：2026-11-11<br>
关联：Goal99<br>
决定：`pivot`

## 判据

- 三个 schedule 全部与 constant 2.00 逐 seed 相同；
- default D3/BC15 `31/32`，strict 两臂 `18/18`；
- relative to constant 2.00：所有 group gained/lost `0/0`；
- strict 22/29 的 `2–1.25` closure 在 S20 与 constant 2.00 上几乎相同；
- trace 边界错配 0，排除调度未生效；
- checkpoint/网络 parity 通过，排除接线和 NoisyNet。

## 结论

2.00 的 strict 退化发生在进入 2 m 之前。近场退回 1.50 不能恢复 capture，
继续扫 scale 或 switch 距离没有信息增量。

## 新路线

不换 SAC/REDQ/DroQ，先做 Goal100 eval-only terminal prior-preservation：

1. 固定 Goal95 checkpoint，dev40 default+strict125；
2. PPO 全开作为参照；
3. 在 `3 m / 2 m / 1.25 m` 内只关闭或衰减 PPO 增量，保留 frozen BC；
4. 检查 strict 22/29 是否恢复到 frozen 20/40，default 是否保持 31/32；
5. 只有近场 PPO 关闭仍无法恢复时，才把问题转到 BC/ALOS capture 几何或观测。

## 不做什么

- 不继续扫 scale/switch；
- 不训练、不开 held-out、不改 reward/terminal/success/goal radius；
- 不把 schedule 的 default 增益写成修复；
- 不在没有方向正确证据时开 SAC。
