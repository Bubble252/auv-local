# Goal54 decision：pre-terminal fixed-window features 不足，stop

日期：2026-11-09<br>
Goal：Goal54<br>
诊断裁决：`PRETERMINAL_FEATURES_NOT_OUTCOME_RELEVANT`<br>
Ledger decision：`stop`

Goal54 在 Goal47 四个 frozen formal trace block 的固定 `0-8.0 s` 窗口上完成
只读审计。窗口内 terminal rows 为 `0`，最早 terminal entry 为 `8.48 s`；
13/13 特征在两个 strict125 block 可评估；10,000/10,000 permutation draws 有效。<br>

```text
strict125 H1 early_control    = FALSIFIED (score 0.353675, global p 0.083292)
strict125 H2 early_navigation = FALSIFIED (score 0.235917, global p 0.083292)
strict125 H3 capture          = FALSIFIED (score 0.692308, global p 0.083292)
default H1                    = PASS
default H3                    = PASS
specific_passes               = []
overall                        = PRETERMINAL_FEATURES_NOT_OUTCOME_RELEVANT
```

H1/H3 的 observed effect 超过阈值，但 family/global permutation 未同时通过；
不能把 effect size 改写成 PASS。default H1/H3 PASS 说明 early-control variance
至少不是 strict125-specific；也不支持把 Goal53 的 late-window association 解释为
pre-terminal early warning。

决定 `stop` 的是“在当前 trace 上用 early fixed-window 低维特征预测 strict125
terminal miss”这条路线，不是宣称 terminal failure 无法预测或底层机制不存在。
不增删特征、不改窗口/covariates/阈值/RNG、不训练、不扫 reward/gamma/seed、不恢复
command-residual PPO/supervised residual、不打开 held-out `100-149`。

后续若要继续，必须新开预注册 goal，并明确选择独立 early-warning 复现、
controlled terminal intervention，或显式记录一个具体 hidden state。不能把
“文献里还有 early-warning 方法”当作绕过冻结协议的理由。

证据：

- `doc/goals/goal54_preterminal_outcome_predictive_audit_20261109.md`
- `doc/goals/goal54_result_20261109.md`
- `research/blockers/2026-11-09-goal54-preterminal-features-null.md`
- `research/paper_reading/2026-11-09-goal54-preterminal-prediction-boundary.md`
- `artifacts/lowlevel/goal54_preterminal_outcome_audit_20261109/`
