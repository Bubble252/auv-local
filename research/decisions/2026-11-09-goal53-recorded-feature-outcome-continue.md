# Goal53 decision：continue

日期：2026-11-09<br>
Goal：Goal53<br>
决策：`continue`，只允许独立复现或预注册 controlled intervention。

## 触发

Goal53 的预注册只读 audit 完成，输入和 permutation 均有效：

```text
H1 strict125 late-control vs true miss = PASS
H2 strict125 timing vs true miss      = PASS
H3 strict125 success-subset capture   = FALSIFIED
default negative control              = FAIL
overall                               = STRICT125_RECORDED_FEATURE_OUTCOME_RELEVANT
heldout_opened                        = false
valid permutation draws               = 10000/10000
```

## 支持的内容

- strict125 两 block 中，late-window `executor_command_std_norm` 与 true terminal
  miss 同向：baseline rho `0.476185`，candidate rho `0.465204`；
  late_control family score `0.465204`，family p `0.0010`，global p `0.0018`。
- strict125 两 block 中，`guidance_terminal_phase_duration_s` 与 true terminal
  miss 同向：baseline rho `0.358730`，candidate rho `0.339415`；
  timing family score `0.339415`，family p `0.0245`，global p `0.0018`。
- default negative control 的 H1/H2/H3 全部失败（global p `0.3552`）。

## 不支持的内容

- H3 capture discrimination 失败：success 子集内最高 two-sided AUC separation
  只有 `0.625 < 0.65`，family p `0.535`；不能把 H1/H2 写成 success/capture
  discordance 的解释。
- 特征来自 late window（每 seed `226` 行）和 episode-level timing，是
  late/concurrent 变量，不是 pre-terminal predictor；关联可能是反向因果或共同
  原因，不能写成 command variance / terminal duration 导致 miss。
- default 的 H1/H2 分数接近阈值（`0.288977`、`0.279814`），strict125-specific
  只是按预注册 Gate 的标签，不是强机制结论。

## 允许的下一步

1. 独立复现：独立 seed 集或独立 checkpoint，冻结同一特征/Gate；
2. 预注册 controlled intervention：固定 checkpoint、seed 和评估协议，显式改变
   terminal timing/control，检验 true miss 是否改变；
3. 若研究 pre-terminal 预测，另开 Goal，把特征窗口结束点冻结在 terminal phase
   之前。

## 禁止事项

- 不在本 Goal 内追加特征、改窗口、改 covariates、改阈值或换 permutation RNG；
- 不训练、不扫 reward/gamma/seed、不恢复 command-residual PPO/supervised residual；
- 不打开 held-out `100-149`；
- 不把 H3 的失败或任何 exploratory 结果救成 PASS，也不把 global p 当作 H3
  单独通过的证据。

## 证据

- `doc/goals/goal53_recorded_feature_outcome_relevance_audit_20261109.md`
- `doc/goals/goal53_result_20261109.md`
- `artifacts/lowlevel/goal53_recorded_feature_outcome_audit_20261109/`
