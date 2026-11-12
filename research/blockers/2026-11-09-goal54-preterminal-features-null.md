# Goal54 blocker：pre-terminal fixed-window features 未预测 strict125 terminal miss

**日期**：2026-11-09<br>
**Goal**：Goal54<br>
**触发**：Goal53 的 late-window association 无法区分 concurrent/reverse-causal 与
pre-terminal prediction；Goal54 在 terminal phase 前冻结窗口重新检验后，strict125
三族全部失败。<br>
**Ledger decision**：`stop`

## 1. 现象

Goal54 使用 Goal47 四个 formal trace block 的 `0 <= t < 8.0 s` 窗口，计算 13 个
pre-terminal 特征，并以 Goal52 的 `terminal_true_miss_m` 和
`goal_capture_eta` 为 endpoint。窗口内 terminal rows 为 `0`，最早 terminal entry
为 `8.48 s`，因此没有 terminal leakage。

```text
records = 160; blocks = 4; window rows = 400/episode
evaluable features = 13/13
valid permutation draws = 10000/10000

strict125 H1 early_control    score 0.353675, family p 0.041896, global p 0.083292 -> FALSIFIED
strict125 H2 early_navigation score 0.235917, family p 0.229577, global p 0.083292 -> FALSIFIED
strict125 H3 capture          score 0.692308, family p 0.168183, global p 0.083292 -> FALSIFIED
default H1                    score 0.372436, family p 0.014799, global p 0.025497 -> PASS
default H3                    score 0.723958, family p 0.011199, global p 0.025497 -> PASS

overall = PRETERMINAL_FEATURES_NOT_OUTCOME_RELEVANT
```

H1/H3 的 observed effect 超过阈值，但 pre-registered global permutation Gate 未过；
不能把 effect size 救成 PASS。default 的 H1/H3 反而通过，说明该 early-control
variance 信号不是 strict125-specific。

## 2. 已排除项

按 `数据 → 标签 → 评测 → baseline → 架构 → 文献` 检查：

- **数据/哈希**：四个 trace、Goal52 CSV、Goal47 replay gate/diagnostic 全部匹配
  冻结 SHA256；diagnostic `valid=true`，held-out 未打开；
- **标签**：`terminal_true_miss_m` 与 trace 最后一行 `goal_distance_eta_m` 最大
  绝对差 `0.0`；四 block 各 40 seeds；
- **窗口**：每 episode `400` rows，窗口内 terminal rows `0`，最早 terminal entry
  `8.48 s`；没有使用 time `>=8.0 s` 的 row；
- **评测**：10,000 次 permutation 全部有效；family/global p 由冻结 RNG
  `20261131` 生成；
- **baseline/negative control**：default H1/H3 PASS，H2 FAIL；不存在 default 与
  strict125 同时通过的可支持 family；
- **数值**：13/13 特征在两个 strict125 block 可评估，0 NaN/Inf；
- **架构**：没有加载 policy、step environment 或 replay；不存在 controller/config
  改动造成的混淆；
- **文献**：pre-terminal/anomaly prediction 与 terminal docking 是不同问题；
  文献不能证明该 8 s 特征集必须成立，也不能把 null 解释成“所有 early warning
  都不可能”。

## 3. 可证伪 hypotheses

### H1：failure 在 8 s 前没有足够的可观测前兆

- 机制：terminal miss 主要由 terminal phase 内的 closed-loop interaction 决定，
  early tracking/control 与 navigation-consistency 状态不足以预测它。
- 预测：在 strict125 两个 block 中，没有 early family 同时达到效应和 permutation
  Gate；default 可能出现通用难度关联，但不能在 strict125 复现。
- 本轮结果：符合。strict125 H1/H2 均失败，H3 也未通过；default H1/H3 通过。
- 下一步最便宜证伪：独立 seed/checkpoint 上重复同一冻结特征和窗口；若仍 null，
  H1 得到支持。

### H2：pre-terminal 机制需要当前 trace 没有记录的 latent state 或 terminal intervention

- 机制：8 s 前的 observable proxies 不足；真正的机制在 estimator internal state、
  USBL geometric availability、controller mode 或 terminal-phase timing 的闭环反馈中。
- 预测：在不增加新 instrumentation 或 controlled intervention 的情况下，
  继续换 early window/加低维 proxy 不会稳定预测 true miss；只有显式记录或干预
  某个具体状态才会改变结果。
- 本轮结果：支持“当前 13 个 proxies 不足”，但不构成 latent state 存在性的证明。
- 下一步最便宜证伪：先写新的 instrumentation/intervention 预注册，明确一个
  具体 hidden state 和 lead-time Gate；不得在本 Goal 内做 feature sweep。

## 4. 最便宜的 falsification 与结果

Goal54 本身已经是最便宜的 falsification：只读已有 160 条 traces，不训练、不
replay、不加载 policy。确定性复跑后 5 个正式输出 SHA256 完全一致，说明 null 不是
随机性或未固定 RNG 造成的。

## 5. 决策

```text
decision = stop
```

停止“在当前 trace 上用 early fixed-window 低维特征预测 strict125 terminal miss”
的路线。不增删特征、不改 8.0 s 窗口、不改 covariates/阈值/RNG、不训练、不扫
reward/gamma/seed、不恢复 command-residual PPO/supervised residual、不打开
held-out `100-149`。

若未来继续，必须新开预注册 Goal，明确选择：

1. 独立 early-warning 复现（独立 seed/checkpoint、lead time、false-alarm Gate）；
2. controlled intervention（改变 terminal timing/control，检验 true miss）；
3. 显式记录一个具体 hidden estimator/controller state，并冻结 instrumentation
   contract。

## 6. 证据

- `doc/goals/goal54_preterminal_outcome_predictive_audit_20261109.md`
- `doc/goals/goal54_result_20261109.md`
- `research/paper_reading/2026-11-09-goal54-preterminal-prediction-boundary.md`
- `artifacts/lowlevel/goal54_preterminal_outcome_audit_20261109/preterminal_outcome_diagnostic.json`
- `artifacts/lowlevel/goal54_preterminal_outcome_audit_20261109/preterminal_outcome_permutation.csv`
- `artifacts/lowlevel/goal54_preterminal_outcome_audit_20261109/preterminal_outcome_per_seed.csv`
