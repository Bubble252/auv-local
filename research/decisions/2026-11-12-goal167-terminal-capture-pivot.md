# Decision：Goal167 FAIL 后转向 terminal horizontal capture 审计

**日期**：2026-11-12<br>
**决策**：`pivot`<br>
**上游**：Goal167 `GATE167_B_FAIL_TERMINAL_HORIZONTAL_CAPTURE`

## 决策

停止在 actuator/command residual 参数化上继续加训练预算。两臂在 primary
`current10` 上都是 `Δ=0/120`，且 B 在 full command authority 下仍只改变
final distance 约 `1 cm`。继续扩 seed、换 SAC/PPO/REDQ/DroQ 或加 reward
shaping 都不能回答“终端横向 authority 是否足够”这个前置问题。

下一 goal 采用 `pivot`，不是 `stop`：主线仍是 residual + ALOS + `z_t`，
但先做只读 terminal horizontal capture / homing authority 审计。

## 理由

- A 证明 actuator 通道能产生数百 rpm/0.08 rad 的动作改变，但净终端位移仍只有
  `1.36 cm`；
- B 证明 command 通道在 full authority 下也没有成功翻转；
- 失败全部是 `GOAL_PLANE_MISS`，横向 `2.2–11.15 m`，纵向 `<0.31 m`；
- 11/18 的 A 失败在 gate 完全打开前越平面，但 7/18 在 full gate 下仍失败；
- 因此优先区分“guidance 没收敛”与“物理 authority 不足”，而不是继续
  调 residual scale 或 reward。

## 不做什么

- 不换主算法族；
- 不扩 seed 救 Gate；
- 不改 success/goal radius/reward/terminal；
- 不放宽 REMUS100 `2.5 m/s`；
- 不打开 held-out `100–149`；
- 不把 A current15 的净 +1 写成有效收益。

## 下一步

预注册 Goal168：`terminal horizontal capture authority audit`。

最小输出：

1. 18 个失败 episode 的 required vs available turn-rate/lateral acceleration；
2. ALOS endpoint projection / direct-homing 分支是否生效；
3. 若 authority 足够，预注册 deterministic terminal homing 对照；
4. 若 authority 不足，记录 contract infeasibility，不用放大 radius 救。

