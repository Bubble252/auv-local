# Goal57 decision：heading-error 只在 primary 边界复现，pivot

**日期**：2026-11-09<br>
**Goal**：Goal57<br>
**Decision**：`pivot`

## 1. 决策

Goal57 在全新 seeds `190-229` 上只检验
`early_tracking_heading_error_abs_mean_rad`。预注册结果为：

```text
H1 primary   strict125/baseline  = PASS
H2 secondary strict125/candidate = FALSIFIED
H3 specificity baseline delta    = FALSIFIED
overall = HEADING_ERROR_PRIMARY_ONLY
decision = pivot
```

H1 的 strict125/baseline partial rho 为 `0.348543`，单侧 permutation
`p=0.016998`，valid draws `10000/10000`。但 bootstrap 95% CI
`[-0.010226, 0.642621]` 包含零，rho 只比 `0.30` 阈值高 `0.048543`，所以这是
边界性单点复现，不是稳健机制证据。

## 2. 为什么必须 pivot

- H2 candidate rho 为 `0.181381`，`p=0.149185`，没有同向复现；
- H3 baseline delta 为 `0.075592`，单侧 bootstrap `p=0.248575`，strict125 与
  default baseline 没有分离；default baseline rho `0.272951` 只低于阈值
  `0.027049`；
- 四个 block 的 0-6 s heading feature 逐 seed 完全相同
  （max abs diff `0.0`，40/40），不是独立 predictor 样本；
- 与 Goal56 相比，strict125/baseline `0.508972 -> 0.348543`，
  strict125/candidate `0.442978 -> 0.181381`，effect 明显衰减。

candidate delta 的描述性 `p=0.048695` 不能改变裁决：H2 已失败，CI
`[-0.028112, 0.445158]` 包含零，且 H3 Gate 只看 baseline delta。

## 3. 允许的下一步

只允许 pivot 到新的预注册分析：

1. candidate-specific endpoint 的独立确认；或
2. instrumentation，解释为什么同一个 early predictor 在不同 endpoint 上出现
   `0.349 / 0.181 / 0.273 / -0.001` 的差异。

在看到新的 `HEADING_ERROR_SPECIFIC_REPLICATED` 裁决前，不授权训练、controller
修改、reward/gamma/seed sweep、继续换 seed 扫描 heading-error，或打开 held-out
`100-149`。

## 4. 证据

- `doc/goals/goal57_heading_error_single_feature_replication_20261109.md`
- `doc/goals/goal57_result_20261109.md`
- `research/LEDGER.jsonl`
- `artifacts/lowlevel/goal57_heading_error_replication_20261109/`
