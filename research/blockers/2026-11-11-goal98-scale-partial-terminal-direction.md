# Blocker：scale 2.0 改善 default，但 strict 的 terminal closure 退化

日期：2026-11-11<br>
Goal：Goal98<br>
状态：`SCALE_PARTIAL / TERMINAL_DIRECTION_REMAINS_BOTTLENECK / NO_SAC`

## 1. 现象

Goal98 修正 scale 锚点为 1.50 后，2.00 让 default 达到 D3 31/40、
BC15 32/40，但 strict125 只有 18/40，低于 frozen 20。所有 strict 失败都是
`goal_plane_miss`，不是 timeout/safety。

## 2. 证据

### 2.1 基线有效

1.50 no-op 与 no override 都复现 Goal95 final 的 30/19，checkpoint SHA
匹配，评测链没有漂移。

### 2.2 scale 表

| arm/scenario | 0.75 | 1.00 | 1.25 | 1.50 | 1.75 | 2.00 |
|---|---:|---:|---:|---:|---:|---:|
| D3 default | 27 | 29 | 30 | 30 | 30 | 31 |
| BC15 default | 28 | 28 | 29 | 30 | 30 | 32 |
| D3 strict | 15 | 16 | 18 | 19 | 17 | 18 |
| BC15 strict | 15 | 16 | 18 | 19 | 17 | 18 |

### 2.3 seed 翻转

default 2.00 gained `27,49`；D3 额外 lost `59`，BC15 default 无 lost。
strict 2.00 gained `44`，lost `22,29`（两臂一致）。

边界 seed 的终端闭合：

| seed/group | 1.50 2–1.25 closure | 2.00 2–1.25 closure | 结果 |
|---|---:|---:|---|
| strict 22, D3 | 1.028 | 0.668 | success→miss |
| strict 29, D3 | 1.016 | 0.442 | success→miss |
| strict 44, D3 | 0.328 | 2.047 | miss→success |
| strict 22, BC15 | 1.058 | 0.688 | success→miss |
| strict 29, BC15 | 1.363 | 0.587 | success→miss |
| strict 44, BC15 | 0.232 | 2.057 | miss→success |

default gained seed 27 的 `3–2 m` closure 从 0.13 提到 0.66；seed 49 从
0.70 提到 1.22。

## 3. 可证伪 hypothesis

- `H98X-A`：2.00 是全局幅度修复但被 path/risk 限制。否定：strict 反而退化，
  且所有失败都是 plane miss。
- `H98X-B`：2.00 放大的是 seed-dependent 的无效 terminal 方向。支持：22/29
  的 terminal closure 显著下降，27/49/44 显著上升。
- `H98X-C`：NoisyNet 或网络接线导致 strict 退化。否定：无 NoisyNet，
  1.50 no-op 精确复现，checkpoint SHA 一致。
- `H98X-D`：authority/clip 是 strict 退化的主因。部分支持/待审计：全局
  magnitude clip 约 0.72–0.78，但 gained seed 44 的 clip 也高（约 0.92），
  不能只看全局均值；需要逐 seed 的 effective/applied 和 plane-miss 几何审计。

## 4. 裁决

```text
status = SCALE_PARTIAL
qualifying_scale = 2.00
decision = pivot
next = Goal99 terminal residual direction audit
SAC = not yet
```

下一步只读复用 Goal98 的 trace，检查 1.50 与 2.00 在 terminal window 的
residual 方向、effective/applied ratio、clip 和 plane-miss 几何。只有
`DIRECTION_SUPPORTED` 才预注册 direction-aware gate；`AUTHORITY_LIMITED`
才回到动作参数化；`GEOMETRY_LIMITED` 才回到 terminal capture/guidance。

## 5. 证据

- `doc/goals/goal98_result_20261111.md`
- `artifacts/lowlevel/goal98_residual_scale_grid_20261111/phase0_summary.json`
- `artifacts/lowlevel/goal98_residual_scale_grid_20261111/phase0_summary.md`
- `artifacts/lowlevel/goal98_residual_scale_grid_20261111/*/trace.npz`
