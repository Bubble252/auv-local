# Decision：Goal139 判 fixed mixed-horizon 无效，转 Goal136 C-line

**日期**：2026-11-11<br>
**决策**：`pivot`（不换算法族）<br>
**上游**：`doc/goals/goal139_result_20261111.md`<br>
**Blocker**：`research/blockers/2026-11-11-goal139-mixed-horizon-single-seed.md`

## 裁决

- M50 `MIXED_SINGLE_SEED_ONLY`：只有 86203 在 3 个 level 正向，nominal strict
  比 N0 低 15，未过 nominal guard。
- M75 `MIXED_SAFETY_STOP`：0/3 seeds，combined20 6 个 timeout。
- PPO `NO_CREDIT_HORIZON_EFFECT`：G980/G950 与 G995 几乎相同，residual 约 1%。
- 该结果不能扩 8 seeds、不能报 5M formal、不能改写为 PASS。

## 为什么不下结论换算法

- 残差比例与 N0 同量级，path/authority 没有异常；
- 固定混合起点接线已通过 256 reset census；
- M75 的更高 near-terminal 密度没有带来更好结果，反而触发 timeout；
- 失败更像“课程分布/课程顺序”问题，而不是“算法族不够强”；
- 用户预注册的算法对照条件尚未满足：没有明确证明是优化/探索或样本效率不足。

## 下一步

1. 完成 Goal136 C-line 的 S0–S4：P0–P4 路径族、`noise_scale`、current
   encoder/alpha、stage sampler 与 promotion/rollback、冻结配置与 1M pilot。
2. C-line 仍用 SAC 主臂、PPO 独立同协议；不把两个算法的样本合并。
3. C-line 只有在 1M pilot/screen 通过预注册晋级条件后才扩 8 seeds / 5M。
4. Goal140 纵向 tracking reward 仍排在 C-line 之后。
5. Goal141 的 `GOAL > safety` 契约问题另开 terminal-contract Goal，不混入
   C-line 或 reward 实验。
