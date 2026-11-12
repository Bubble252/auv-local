# Blocker：Goal97 把训练 scale 1.5 误当成 1.0 基线

日期：2026-11-11<br>
Goal：Goal97<br>
状态：`INVALID_SCALE_ANCHOR / NO_GATE_RECORD`

## 1. 现象

Goal97 的 `s=1.00` 评价得到 D3 default 29/40、strict125 16/40；但 Goal95
final 是 30/40、19/40。基线没有复现，必须先查协议，不能继续解释 sweep。

## 2. 证据

### 2.1 配置真实 scale 是 1.5

```text
configs/lowlevel/goal95_20261111/p_d3_ppo.json:  pc.actuator_residual_scale = 1.5
configs/lowlevel/goal95_20261111/p_bc15_ppo.json: pc.actuator_residual_scale = 1.5
```

Goal97 runner 的 `1.00` 是绝对 override，不是相对训练 scale 的乘数。

### 2.2 1.5 是真正的 no-op

同 checkpoint、同 dev40、同 success-frame 的锚点复核：

| arm/scenario | no override | override 1.50 |
|---|---:|---:|
| P_D3 default | 30/40 | 30/40 |
| P_D3 strict125 | 19/40 | 19/40 |
| P_BC15 default | 30/40 | 30/40 |
| P_BC15 strict125 | 19/40 | 19/40 |

这排除了 eval runner、checkpoint 和 success-frame 漂移。

### 2.3 无效 sweep 的单调趋势

| arm/scenario | 0.25 | 0.50 | 0.75 | 1.00 | 1.25 |
|---|---:|---:|---:|---:|---:|
| P_D3 default | 25 | 27 | 27 | 29 | 30 |
| P_D3 strict125 | 16 | 16 | 15 | 16 | 18 |
| P_BC15 default | 25 | 26 | 28 | 28 | 29 |
| P_BC15 strict125 | 15 | 16 | 15 | 16 | 18 |

所有点都低于 1.5；越接近 1.5 越好，不能用于判断“放大是否有效”。

## 3. 可证伪 hypothesis

- `H97X-A`：1.5 no-op 会复现 Goal95 的 30/19。支持；因此错误在 scale
  锚点，不在评测链。
- `H97X-B`：`s=1.25` 的改善来自接近训练 scale，而不是新的有效方向。支持：
  1.25 仍低于 1.5，且 1.5 恢复到 30/19。
- `H97X-C`：NoisyNet 或网络接线导致 1.0 变差。否定：当前 PPO 无 NoisyNet，
  checkpoint SHA 与 Goal95 一致，1.5 no-op 完全复现。

## 4. 裁决

```text
status = INVALID
reason = scale_anchor_assumed_1.0_actual_1.5
gate_recorded = false
decision = pivot
```

不能把无效 sweep 写成 `SCALE_NOT_SUPPORTED`。下一步是 Goal98：以 1.5 为
基线，覆盖 0.75/1.00/1.25/1.50/1.75/2.00，先验证 1.5 复现 30/19，再判定
是否有 scale 能同时达到 frozen 31/20 且 path/risk 不退化。

## 5. 证据

- `doc/goals/goal97_result_20261111.md`
- `artifacts/lowlevel/goal97_prior_preserving_20261111/scale_anchor_check/`
- `artifacts/lowlevel/goal97_prior_preserving_20261111/scale_sweep_invalid_scale_anchor_summary/`
- `artifacts/lowlevel/goal97_prior_preserving_20261111/scale_sweep_invalid_scale_anchor_trace/`
