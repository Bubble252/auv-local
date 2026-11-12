# Goal31 H2/H3 decision：停止现有机制线

日期：2026-11-09<br>
Goal：Goal31<br>
Decision：`stop`

## 依据

只读诊断使用 Goal31 冻结的四组 40-seed formal traces，所有完整性检查通过：

- H2-E（导航/状态估计主导）：candidate terminal miss 的 estimator plane share
  median 仅 `0.1000/0.0994`，nav plane error median `0.3290/0.3392 m`，
  estimator-dominated seeds 为 `0/35` 和 `0/36`，Spearman
  `0.0095/0.0548`；H2-E `FALSIFIED`。
- H2-L（内环滞后主导）：pitch error 和 saturation 确实偏高，但 vertical closure
  rate median `0.2053/0.1445 m/s` 高于预注册上限 `0.10 m/s`，pitch/stern 与
  vertical miss 的相关性 `0.0754/-0.3046` 和 `0.0359/-0.1125` 不满足支持条件；
  terminal phase 中位持续约 `10.5/10.8 s`，不是短窗口假象；H2-L `FALSIFIED`。
- H3（path-lookahead 垂直需求/饱和代价）：candidate 的 stern abs median 和
  abs pitch command median 都下降（`-0.0167/-0.0176 rad`、
  `-0.0862/-0.0785 rad`），saturation 仅小幅上升
  （`+0.0396/+0.0347`），且与 vertical abs delta 的相关性为负
  （`-0.0348/-0.2287`）；H3 `FALSIFIED`。

总体为 `NO_SUPPORTED_MECHANISM`。vertical-dominated terminal miss 仍然存在，但
当前 trace 的 estimator、pitch error、stern、saturation 和 2 s closure 代理均不能
单独解释它。

## 停止事项

- 停止 Goal31 的 H2/H3 机制线；
- 不继续扫 path-lookahead、terminal window、reward、gamma/lambda、seed 或
  command/supervised residual；
- 不预注册 Goal32；
- 不打开 held-out `100-149`；
- 不把本次 `NO_SUPPORTED_MECHANISM` 写成 formal PASS/FAIL。

## 下一步

只有形成新的、可证伪且可观测的机制后才恢复：优先考虑直接分析 terminal phase
入口处的垂直状态/可达性，或测量参考 pitch/vertical velocity 与实际响应的频带和
限幅；不得再次只改 terminal reference 参数后直接跑 formal。
