# Blocker：Goal130 的 channel cap / direction gate 仍全部 NO_EFFECT

**日期**：2026-11-11<br>
**Goal**：Goal130<br>
**状态**：`OPEN_FOR_TERMINAL_GEOMETRY / PARAMETERIZATION_CEILING / HELDOUT_CLOSED`<br>
**上游**：Goal129 `ACTION_PARAM_NO_EFFECT`、Goal128 `GEOMETRY_LIMITED`

## 现象

同一批 8 个 Goal126 SAC checkpoint，128 条 eval-only trace：

```text
zero   strict 24.000 / default 33.000
M1.0   strict 19.750 / default 28.250
R0.10  strict 20.250 / default 29.625
R0.25  strict 19.750 / default 29.750
R0.50  strict 19.250 / default 29.500
F0.25  strict 21.875 / default 31.625
P0.25  strict 22.750 / default 32.750
D0.0   strict 21.125 / default 31.625
D0.5   strict 23.125 / default 32.375
R0.25D0.0 strict 21.500 / default 31.750
```

八个 arm 全部 `ACTION_PARAM_NO_EFFECT`。

## 证据

1. **rpm-only cap 反而更差**：R0.10/R0.25/R0.50 为 20.25/19.75/19.25，
   低于或等于 M1.0；rpm 残差不是单纯过大，cap 掉了 approach 需要的修正。
2. **fin-only cap 也不够**：F0.25 21.875/31.625；单独压 fin 会损失转向/俯仰。
3. **per-channel 与 global cap 等价**：P0.25 22.75/32.75，与 Goal129 C1.0
   22.75/32.875 基本相同。
4. **方向假设不成立**：近区 `cos(L,B)` 均值约 -0.03，约 51% 控制步为负；
   learner 与 BC 几乎正交，不是“方向对、幅值错”。D0.5 的 23.125 只是
   少做一点错事，不是保留正确方向。
5. **parity 成立**：所有开关关闭时 action maxdiff=0，
   learned_residual maxdiff=0，success/terminal counts 与 Goal129 M1.0 一致。
6. **网络结构不是变量**：同一 checkpoint；Goal108 audit 的 63/63 tensor
   parity、`[z_t64, zero-DADE64, command3]`、`256×128` actor、`mu` 零初始化、
   无 NoisyNet 均未改变。

## 已排除

- 参数化接入错误：parity 精确为 0。
- NoisyNet / 网络结构：无噪声参数，结构对照一致。
- reward farming / timeout：失败仍以 plane-miss 为主；cap 后 timeout 多数为 0。
- 单纯 rpm 幅值：rpm-only cap 反而更差。
- 单纯方向：近区 cos 近 0，方向 gate 只能随机丢弃。

## 可证伪假设

| 假设 | 机制 | 成立时应看到 | 最便宜证伪 |
|---|---|---|---|
| H130-X1：terminal 失败由垂直几何决定 | 近区 vertical error / pitch 在 success/failure 间有最大分离 | first-2m/3m 的 vertical error 效应量大，水平误差不显著 | 只读 trace 几何审计，不训练 |
| H130-X2：terminal 失败由水平/进入顺序决定 | cross-track / heading / 进入 plane 的顺序决定 plane-miss | horizontal error 或 first-entry 顺序有显著分离 | 同 trace，按 first 1/2/3/5/10 m 分层 |
| H130-X3：速度/时间尺度决定 capture | 近区速度过高，擦过 goal plane | terminal speed / closure rate 与失败强相关 | trace 统计 + 现有 controller 速度 scale |
| H130-X4：纯策略残差已无法修复几何 | 所有动作参数化都 ≤ zero | 几何审计显示失败在 residual 生效前已决定 | 几何审计 + 最便宜 approach geometry screen |

## 裁决

```text
decision = pivot
停止动作参数化家族，转 terminal capture / approach geometry 只读审计。
```

不改 reward、terminal、success、goal radius；不打开 held-out；不换算法。
只有几何审计给出明确、可操作的分离量后，才设计新的 eval-only/训练型 Goal。
