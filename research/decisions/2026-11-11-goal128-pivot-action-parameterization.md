# Decision：Goal128 后从 reward/terminal 审计 pivot 到动作参数化

**日期**：2026-11-11<br>
**Goal**：Goal128<br>
**decision**：`pivot`<br>
**next**：Goal129 eval-only 动作参数化 gate/cap screen

## 依据

1. Goal128 的 34 条 trace 全部 finite，success 计数与 Goal126/127 完全一致，
   证明评估链和 scale manifest 有效。
2. 失败几乎全是 `GOAL_PLANE_MISS`；zero strict 16/16、0.125 strict 138/138、
   1.0 strict 160/162。Goal126 的 “TIME_LIMIT” 描述已勘误。
3. 策略没有刷 dense reward：dense/step 更负（1.0 -0.632 vs zero -0.577），
   terminal share 也更低。H128-A `FALSIFIED`。
4. 1.0 下 learner/BC = 1.5–2.3×，近区 raw policy 饱和 24.8%–82.0%，
   pre64 eta +0.317 m；86201 饱和最低（24.8%）且唯一获益（24→30）。
   H128-D `SUPPORTED`。
5. 0.125 下 learner 仅为 BC 的 21%–27%，但仍有 5 个 seed 各丢 1–4 个 episode；
   与厘米级 success margin 一致，说明 terminal capture 是刀口。

## 裁决

```text
不扩 SAC 5M formal
不换 REDQ/DroQ
不继续扫 SAC lr/UTD/entropy/scale
不改 reward/terminal/success/goal radius
不打开 held-out
```

“残差方向对、只是探索/样本效率不足”没有被证明：7/8 seed 的残差把近区几何推坏。
因此不开 SAC 算法对照，先做 eval-only 动作参数化：

- inverse terminal gate：learner 在 10 m 外为 1、3 m 内为 0；
- BC-relative norm cap：`||L|| <= 0.25 ||B||`；
- 两者组合；
- 对照 zero / 0.125 / 1.0 unmodified。

## 下一步 Gate

| 观察 | 判读 | 动作 |
|---|---|---|
| 1.0 + gate/cap ≥25.5 strict、≥33 default、≥4/8 seed 提升 | `ACTION_PARAM_GAIN` | 用该参数化开 1M 训练型 Goal |
| ≥24 strict、≥33 default，且没有 seed 丢超过 1 个 episode | `ACTION_PARAM_SAFE_PARITY` | 只允许训练 screen，不扩 formal |
| 仍低于 zero 或无稳定变化 | `ACTION_PARAM_NO_EFFECT` | 转 terminal capture / approach geometry，不改 reward |

Goal129 不训练；只有 Gate 通过后才讨论训练。失败结果保留，不放宽阈值。
