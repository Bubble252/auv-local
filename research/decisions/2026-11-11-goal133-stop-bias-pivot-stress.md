# Decision：停止 eval-only bias，转冻结 SAC vs zero 的压力测评

**日期**：2026-11-11<br>
**Goal**：Goal133<br>
**Decision**：`stop`<br>
**上游**：Goal132 `GEO_NO_EFFECT / correct-sign dose only`

## 触发

Goal133 的 Vm005/Vm010/Vm020 全部 `GEO_SAFE_PARITY`。正确符号让 last-2s
`|vertical|` 单调下降约 3 cm，但 strict success 只增加 1 个 radius 边界 seed，
default 完全不变。

## 判读

固定单通道 geometry bias 已经到达上限：

- 符号正确；
- dose 到 hard cap；
- 没有新增 loss；
- 几何连续改善；
- success 仍低于 `GEO_GAIN`。

继续扫 gain 只会对 seed 49/33 等 radius 边界样本做阈值拟合，不再有科学信息。

## 决策

`stop` eval-only geometry bias 家族。下一个 Goal 不换 PPO/SAC，改为检验用户提出的
关键问题：**当前 nominal 测评是否对 ALOS+BC 太容易，因而看不出神经网络优势。**

Goal134 预注册为冻结策略的压力测评：

```text
arms: zero + 8 个 Goal126 SAC checkpoint
stress: current 1.5x / current 2x / plant ±20% / combined
eval: strict/default dev40，success 定义不变
```

Gate 只认跨训练 seed 的可复现优势，不认单一 86201 幸存者。

## 停止规则

- 任一 stress level 有 ≥4/8 checkpoint strict 提升 ≥2 且 default 不退化：
  `STRESS_ADVANTAGE`，预注册该分布的训练型 Goal；
- 只有 86201 优势：`SINGLE_SEED_ONLY`，不扩训练；
- 全部无优势：`STRESS_NO_ADVANTAGE`，停止“压力环境自然放大 NN”的假设，
  回到训练信号/算法机制；
- 不改 nominal success/radius，不打开 held-out，不换主算法，不自动 commit/push。
