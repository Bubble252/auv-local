# Decision：Goal123 判 BIAS_IMPROVES_CAPTURE，先做 default guardrail，再 bias-initialized PPO

**日期**：2026-11-11<br>
**Goal**：Goal123<br>
**decision**：`continue`<br>
**next**：Goal124 default-scenario guardrail + bias-initialized PPO design

## 依据

1. `stern-025`/`stern-030` 在 strict125 dev40 达到 24/40，gained
   29/31/33/56，lost 0，timeout/safety 0。
2. 剂量从 -0.05 到 -0.30 单调：21→21→21→22→24→24，不是单 seed 噪声。
3. PPO 的 learned residual 放大到 16x/32x 有权限但净交换/净伤害；常数
   stern bias 是唯一净增益方向。
4. 这是一个动作参数化/终端偏置发现，不是 RL 优化结果；必须先确认 default
   场景不被伤害，再把它作为 PPO 先验。

## 下一步

Goal124：

- eval-only，同一 C3 reward、同一 checkpoint，只把 goal radius 改为 2.0；
- `zero / stern-025 / stern-030` × dev40；
- Gate：default 相对 zero 下降 `<=1` 且 strict 复现 `>=23/40` 判
  `BIAS_GUARDRAIL_PASS`；
- 通过后预注册 **bias-initialized PPO**：base = frozen BC + constant stern
  bias，actor 输出层初始化为该 bias，训练 10M，对照 zero-bias PPO；
- 仍不换算法，不打开 held-out，不改 success 定义。

不自动 commit/push。
