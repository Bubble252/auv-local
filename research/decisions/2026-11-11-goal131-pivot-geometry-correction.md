# Decision：Goal131 后转 geometry-aware terminal correction

**日期**：2026-11-11<br>
**Goal**：Goal131<br>
**Decision**：`pivot`<br>
**上游**：Goal130 `PARAMETERIZATION_CEILING`

## 触发

Goal131 的只读审计覆盖 2,000 个 episode。strict 与 default 都出现强几何分离：

```text
strict:
  first_2m path3d       Cliff δ = -0.85 ~ -0.92
  last2s |vertical|     Cliff δ = -0.71 ~ -0.81
  last2s eta closure    Cliff δ = +0.77 ~ +0.89

default:
  first_3m vertical     Cliff δ = +0.95 ~ +0.99（zero/C1.0）
  first_3m path3d       Cliff δ = -0.94 ~ -0.97
  last2s eta closure    Cliff δ = +0.84 ~ +0.92
```

成功 episode 的近区闭包更快（约 1.5–1.8 m/s），失败更慢（约 0.5–0.9 m/s）；
失败者的 last-2s 垂直误差约是成功者的 2–3 倍。default 下所有失败都未进入
2 m，说明失败在 2 m 前已经决定；strict 下进入 2 m 后也只有约 70% 成功。

## 判读

这不是 residual 幅值/方向问题，也不是网络结构/NoisyNet：

- Goal129/130 已覆盖 global、shell、channel、direction 四类参数化；
- Goal131 的几何量在 residual 生效前就已分离；
- 继续扫 residual 参数只会重复 `NO_EFFECT`。

## 决策

`pivot` 到 Goal132：**geometry-aware terminal correction**。

优先做最便宜的 eval-only 版本，不改 reward/terminal/success：

1. 垂直修正：在 2–10 m 内，用 `signed_vertical_error_m` 生成小幅度
   stern-plane bias，扫描符号与增益；
2. 闭包修正：在 2–10 m 内给小幅度 rpm bias，扫描增益；
3. 组合：垂直 + 闭包；
4. 先用 zero-residual 与 C1.0/D0.5 两种底座评价，确认修正是否在
   frozen BC 上就有效，再决定是否把它作为 RL residual 的 prior。

## 停止规则

- 若垂直/闭包修正达到 `SAFE_PARITY` 或 `GAIN`：把它固定为新的 prior，
  再开训练型 Goal；
- 若所有 eval-only 修正都无效：写 `stop`，说明当前 success 契约下
  residual 方案已到上限，并向用户报告；
- 不换 PPO/SAC 主算法，不开 REDQ/DroQ，不改 reward/terminal/success/goal
  radius，不打开 held-out，不自动 commit/push。
