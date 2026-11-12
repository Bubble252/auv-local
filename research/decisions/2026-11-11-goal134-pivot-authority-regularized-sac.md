# Decision：从压力测评 pivot 到 authority-regularized SAC

**日期**：2026-11-11<br>
**Goal**：Goal134<br>
**Decision**：`pivot`<br>
**上游**：Goal134 `SINGLE_SEED_ONLY / SAC_STRESS_HARM`

## 触发

Goal134 的 90-run 压力测评显示：

- 压力有效：zero strict 最多下降 11；
- 没有跨 checkpoint 优势：只有 86201 在 current15/plant20 获益；
- current20/combined20 出现 8/8 checkpoint harm；
- 86201 的学习残差 RMS 最小，86202/86204 的 harm 随 residual scale 降低而
  单调恢复。

## 判读

这不是“SAC 算法族不行”，也不是“压力不够”。更具体的机制是：

```text
1M SAC 在多数 seed 上学到了过大的 residual authority；
高 current / plant mismatch 下，这个 authority 把 frozen BC/ALOS 推坏；
86201 学到的是小残差 + 小幅正确方向。
```

## 决策

`pivot` 到 Goal135：保持 SAC 和现有 z_t/path8+endpoint4 结构，只改训练侧的
authority 参数化/正则：

- 约束 learner residual 的 RMS 或相对 frozen BC 的比例；
- 保持 `mu=0`、`log_std=-3.0`、无 NoisyNet；
- 评价仍用 deterministic mean；
- 不改 reward/terminal/success/goal radius；
- 先 1M × 8 seeds screen，和 zero、86201 及 Goal134 authority dose 对照。

## 为什么不是别的

| 选项 | 不选原因 |
|---|---|
| 换 PPO | 没有证据表明 PPO 在压力下更好；Goal108 PPO 也没有跨 seed 优势 |
| REDQ/DroQ | 没有样本效率证据；当前问题是 authority，不是 Q 值偏差 |
| 继续扫压力 | 压力已足够区分，继续加大只会增加 harm |
| 停止 | 86201 在 scale 0.25 仍有 +2/+2，说明方向信号存在 |

## 停止规则

- Goal135 若 `>=4/8` seed 在 nominal 达到 strict `>=26` 且 default 不退化：
  扩 5M formal；
- 若只有 86201：`SINGLE_SEED_ONLY`，查训练方差/数据覆盖；
- 若全 seed 退化成 zero：`DIRECTION_SIGNAL_INSUFFICIENT`，回到 reward/terminal/state；
- 不改主算法，不打开 held-out，不自动 commit/push。
