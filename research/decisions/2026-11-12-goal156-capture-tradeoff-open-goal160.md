# Goal156 decision：CAPTURE_TRADEOFF，转动作参数化而不是换算法

**日期**：2026-11-12<br>
**Goal**：Goal156<br>
**决策**：`pivot` 到 Goal160；不换 SAC，不启动 REDQ/DroQ，不扩 seed。

## 证据

- `near10_2` default `84/120` vs zero `81/120`（`+3`），strict `44/120` vs
  `42/120`（`+2`），低于预注册 `+6/+4`；
- path 代价只有 `+0.0017/+0.0016 m`，residual 2–10 m RMS `0.898`，>10 m
  `0.0001`，说明 Gate 没有造成残差塌缩；
- 5M 反而低于 Goal86 1M 的 `+6/+4`，不支持样本效率是主因；
- scale 1.0/1.25/1.5 的 default 都是 `84/120`，不支持的“放大近端 residual”
  H-scale 方案；
- A_screen 4/4 压力档未出现超过 `2/40` 的下降，但 nominal default 只有 `+6`，
  不能替代 formal；
- 失败全部是 `GOAL_PLANE_MISS`，entry 2 m 纵向 p90 `1.917 m`，终端几何仍是
  主瓶颈。

## 解释

5M 训练没有把 residual 学成稳定的 terminal-capture 修正。当前更符合数据的是
actuator residual 的动作参数化和 credit 分配没有把“进入 2 m 后的纵向/横向
误差”变成可复用控制；继续增加预算或换 off-policy 算法不能替代这个归因。

## 决策

1. 保留 `near10_2` 作为部署对照，不改 success/radius/reward/terminal；
2. 预注册 Goal160，比较 actuator residual、command-space residual、
   terminal-constrained residual；
3. 先做 3 seeds × 1M screen；只有新动作参数化在 H3 zero 配对下过 Gate，
   才推进 formal 或更高预算；
4. 只有后续独立证据明确显示样本效率瓶颈，才另开 SAC vs REDQ/DroQ Goal；
5. 继续把路线健康、final-distance bins、2 m entry geometry、vertical/horizontal
   path RMSE、residual channel/bias/saturation 作为固定报告项。

## 下一步

写 Goal160 预注册和 smoke，不直接启动长训练；smoke 通过后再按 3 seeds × 1M
并行训练，避免重演 Goal159 的 direct-command/配置不一致。
