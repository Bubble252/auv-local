# Goal56 decision：6.0 s early-control 复现，继续做机制实验

**日期**：2026-11-09<br>
**Goal**：Goal56<br>
**Decision**：`continue`

## 1. 决策

Goal56 在 seeds `150-189` 的固定 `[0, 6.0)` 窗口上通过预注册 H1
`early_control` Gate：

```text
family score = 0.442978
family p     = 0.004100
global p     = 0.006299
driver       = early_tracking_heading_error_abs_mean_rad
rho          = 0.508972 (baseline) / 0.442978 (candidate)
```

H2 `early_navigation` 与 H3 success-subset capture 均为 `FALSIFIED`；default
negative control 的 H1/H2/H3 均未 PASS。因此按冻结映射为
`PRETERMINAL_FEATURES_INDEPENDENTLY_REPLICATED / continue`。

## 2. 为什么不是更强结论

- 13 个 early 特征在 baseline 与 candidate 之间逐 seed 完全相同
  （max abs diff `0.0`，40/40）。两 block 同号是同一 predictor、两个 endpoint
  变体，不是两个独立特征样本。
- default baseline heading rho 为 `0.469156`，与 strict125 baseline
  `0.508972` 接近；default candidate 为 `0.259689`，只比 `0.30` 阈值低
  `0.040311`。strict125/default 的差异没有被预注册检验，描述性 Fisher z
  约 `0.904`，不能声称强机制差异。
- H3 capture 失败，说明连续 true-miss 关联不能升级为 true capture 判别。

## 3. 允许的下一步

只允许新的预注册机制实验，优先级如下：

1. **独立 predictor 验证**：在全新 seed block（例如 `190-229`）上只预注册
   heading-error 单特征、单 primary block、固定 effect/CI Gate；
2. **controlled early-heading intervention**：固定 checkpoint/seed，显式扰动
   early heading error 或 initial heading geometry，检验 terminal true miss
   是否随之改变；
3. **instrumentation**：记录 early controller/estimator 内部状态，并冻结
   lead-time、alarm threshold 和 false-alarm 预算。

不授权训练、reward/gamma/seed sweep、controller 修改或 held-out `100-149`。

## 4. 证据

- `doc/goals/goal56_preterminal_fixed_cutoff_replication_20261109.md`
- `doc/goals/goal56_result_20261109.md`
- `research/LEDGER.jsonl`
- `artifacts/lowlevel/goal56_preterminal_fixed_cutoff_replication_20261109/`
