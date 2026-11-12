# Decision：Goal168 归因为 terminal heading lag，开 Goal169 deterministic homing timing 对照

**日期**：2026-11-12<br>
**决策**：`pivot`<br>
**上游**：Goal168 `AUDIT_COMPLETE / TERMINAL_HEADING_LAG_DOMINANT`

## 决策

Goal168 已把 Goal167 的失败机制定位到 terminal homing 的转向时序，而不是
residual 动作幅度、NoiseNet、样本效率或 20 m 处物理 authority。

因此下一步不继续训练 residual，而是开 Goal169：用同一 zero controller 做
deterministic terminal homing timing / blending 对照。

## 理由

- 失败 episode 在 20–15 m 的 terminal phase duty 为 `0.96–1.00`；
- 20 m 入口 authority margin 为 `2.17–33.76`；
- 但命令需要转 `0.74–1.47 rad`，实测 yaw rate 只有 `0.061–0.087 rad/s`；
- 15/18 的 `rotate_time_ratio` 为 `0.87–1.88`，转向时间超过剩余时间；
- 3/18 为 seed 26 的 `path_progress` 5.35 m 跳变，作为次要问题保留。

## 不做什么

- 不换主算法族；
- 不扩 seed；
- 不改 success/goal radius/reward/terminal；
- 不放宽 REMUS100 `2.5 m/s`；
- 不打开 held-out；
- 不把 Goal167 的 A current15 净 +1 写成有效收益。

## 下一步

Goal169 预注册：

1. zero controller deterministic replay；
2. 只改 terminal homing 的切换距离（20→30/40/50 m）或 40→20 m 平滑 blend；
3. 同一 18 个失败 episode + 全部 dev40；
4. Gate：default `+4/120`、strict125 `+2/120`，guards 不退化，所有
   `eta<=2.5 m/s`；
5. 若通过，再以该 guidance 为 base 重新预注册 residual RL 对照。

