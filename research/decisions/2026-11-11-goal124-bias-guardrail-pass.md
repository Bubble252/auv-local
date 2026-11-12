# Decision：Goal124 BIAS_GUARDRAIL_PASS，进入 bias-initialized PPO

**日期**：2026-11-11<br>
**Goal**：Goal124<br>
**decision**：`continue`<br>
**next**：Goal125 bias-initialized PPO（10M × 2 seeds × 4 arms）

## 依据

1. `default_stern-025/030` 都是 33/40，与 `default_zero` 完全相同，default 无伤害。
2. `strict_stern-025/030` 都是 24/40，gained `29/31/33/56`，lost 0。
3. timeout/safety 全 0。
4. Goal125 bridge smoke 已确认 `base_residual_bias=(0,-0.25,0)` 与 Goal124
   const injection 的 dev40 success 集合完全一致；网络初始化 63/63 tensor 一致。

## 裁决

```text
fixed bias 不是场景过拟合
它可以在不伤 default 的情况下作为 PPO base prior
```

## 下一步

Goal125：

- 8 个训练 job：`Z15/B15/B20/BP3` × 2 seeds，10M transitions；
- strict/default dev40；
- 同 seed fixed-bias control；
- Gate：`PPO_BIAS_GAIN` / `BIAS_FIXED_IS_CEILING` / `PPO_BIAS_HARM` / `PPO_BIAS_MIXED`；
- 不打开 held-out，不自动 commit/push。

只有 Goal125 证明 bias prior 存在、PPO 仍不能在其上提升，且动作权限/reward/terminal
capture 已排除后，才允许预注册 SAC 同协议对照。
