# Decision：Goal122 判 GAIN_HARM，先做常数 bias 标定，不直接 SAC

**日期**：2026-11-11<br>
**Goal**：Goal122<br>
**decision**：`continue`<br>
**next**：Goal123 constant stern/rpm bias calibration

## 依据

1. PPO gain sweep 证明动作权限存在：16x/32x 的 Δeta 为 0.313/0.454 m。
2. 但 success 没有改善：16x 20/40（+2/-2）、32x 19/40（+4/-5）、
   64x 13/40（+5/-12）。正式 Gate 为 `GAIN_HARM`。
3. Goal121 的常数 `stern-0.2` 是 22/40（+2/-0），方向净增益；PPO 放大方向
   不是净增益。说明 PPO 学到的 residual 方向没有对齐 terminal capture。
4. 直接开 SAC 不满足用户条件“方向对但探索/样本效率不足”。先用常数 bias
   找到稳定方向，再决定是否把 bias 作为 PPO/SAC 先验。

## 下一步

Goal123：

- eval-only，固定 Goal119 C3 checkpoint、dev40、success frame；
- stern bias `-0.05/-0.10/-0.15/-0.20/-0.25/-0.30`；
- rpm bias `-0.10/-0.20`；
- combined `stern -0.20 + rpm -0.10`；
- Gate：任一 bias `>=23/40` 且 lost<=1 判 `BIAS_IMPROVES_CAPTURE`；
- 找到稳定 bias 后，预注册 bias-initialized PPO 或 bias + RL residual；
- 找不到才考虑 SAC；不打开 held-out，不改 success/goal radius。

不自动 commit/push。
