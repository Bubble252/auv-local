# Blocker：压力下只有 86201 获益，7/8 checkpoint 退化

**日期**：2026-11-11<br>
**Goal**：Goal134<br>
**触发**：压力包络可区分方法，但没有跨 checkpoint 优势；且多数 checkpoint 在
current20/combined20 达到 `SAC_STRESS_HARM`。

## 现象

Goal134 正式 90 runs 全部 finite，nominal parity 精确：

```text
zero strict/default = 24/33
S_s86201 strict/default = 30/33
```

压力确实生效：

| level | zero strict | drop |
|---|---:|---:|
| current15 | 14/40 | +10 |
| current20 | 16/40 | +8 |
| plant20 | 22/40 | +2 |
| combined20 | 13/40 | +11 |

但 SAC 结果不是跨 seed 优势：

| level | advantaged | harm |
|---|---|---:|
| nominal | 86201 | 7/8 |
| current15 | 86201 | 7/8 |
| current20 | 0/8 | 8/8 |
| plant20 | 86201 | 7/8 |
| combined20 | 0/8 | 8/8 |

唯一 `SINGLE_SEED_ONLY` 信号来自 86201：

```text
current15: strict 16 vs zero 14 (+2), default 29 vs 24 (+5)
plant20:   strict 27 vs zero 22 (+5), default 31 vs 31 (0)
```

86201 在 current20/combined20 也退化：

```text
current20: strict -2, default -1
combined20: strict -4, default +1
```

## 证据：残差幅值与 harm 同向

strict current15 的 learned residual RMS / action abs max：

| arm | success | learned RMS (rudder/stern/rpm) | action max |
|---|---:|---|---:|
| 86201 | 16 | 0.204 / 0.270 / 0.206 | 0.991 |
| 86202 | 8 | 0.280 / 0.269 / 0.279 | 0.997 |
| 86203 | 8 | 0.239 / 0.218 / 0.224 | 0.994 |
| 86204 | 7 | 0.379 / 0.316 / 0.404 | 0.998 |
| 86205 | 12 | 0.254 / 0.270 / 0.260 | 0.997 |
| 86206 | 9 | 0.296 / 0.275 / 0.315 | 0.998 |
| 86207 | 13 | 0.300 / 0.380 / 0.316 | 0.997 |
| 86208 | 12 | 0.295 / 0.365 / 0.293 | 1.000 |

86201 不是“学到了更复杂的修正”，而是**学习残差幅值最小**。其余 checkpoint
的 rpm 学习残差相对 frozen BC 残差达到 2.5–3.3 倍，86204 达到约 3.3–4.0 倍。

## 已排除

- 测评压力不足：zero 在 current15/current20/combined20 分别下降 10/8/11 个
  strict success，压力有效。
- nominal 复现失败：zero 24/33、86201 30/33 精确复现。
- 网络接线/NoisyNet：Goal108/126/132/133 parity；无 `rho/sigma/noise` 参数。
- 评估 scale 重复：Goal126 已修复，本轮 `eval_residual_multiplier=1.0`。
- 传感器 p_nav/eta 偏差或延迟：固定 `unbiased_noise_v1`，未改 profile。

## 可证伪假设

| 假设 | 机制 | 成立时应看到 | 最便宜证伪 |
|---|---|---|---|
| H134-E：harm 来自残差 authority 过大/饱和 | 高 current 下 SAC 残差把执行器推到边界，闭环振荡或 plane-miss | 86202/86204 在 current15 降到 residual-scale 0.5/0.25 后 success 单调回到 zero 附近 | 3 arms × 2 scales × 2 scenarios = 12 runs |
| H134-F：learned direction 本身错误 | 缩放只减弱错误方向，success 仍低于 zero；zero 最好 | 86202/86204 在 0.5/0.25 仍显著低于 zero | 同上 |
| H134-G：86201 优势只是低 authority | 86201 的增益来自幅值小，不是方向对 | 86201 降到 0.5/0.25 后优势消失或回到 zero | 同上 |

## 最便宜 falsification

固定 `current15`，对 `86201/86202/86204` 跑：

```text
residual-scale = 0.5, 0.25
scenario = strict, default
seeds = 20–59
```

判读：

- 若 harm 随 scale 降低单调恢复，且 86201 优势随 scale 降低消失：
  `AUTHORITY_DOMINANT`，下一步预注册 authority/regularization 训练项；
- 若 86202/86204 在低 scale 仍低于 zero：`DIRECTION_BAD`，下一步查训练信号和
  状态/奖励，不换算法；
- 若 86201 在 0.5 仍保持优势：`LOW_AUTHORITY_DIRECTIONAL_GAIN`，值得做
  norm-capped SAC 训练。

## 文献

- `research/paper_reading/2026-11-11-goal134-residual-authority.md`

## 状态

```text
decision = pivot_to_authority_dose
algorithm_switch = forbidden
heldout = closed
```
