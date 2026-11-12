# Goal52 decision：sigma 没有真实 outcome relevance，stop sigma-residual line

日期：2026-11-09<br>
Goal：Goal52<br>
诊断裁决：`SIGMA_NOT_OUTCOME_RELEVANT`<br>
Ledger decision：`stop`

Goal52 在冻结的 Goal47 四个 formal trace block 上完成只读 outcome-relevance
审计。输入、terminal identity、sigma consistency、160 条 records 和 10,000 次
joint permutation 均有效，held-out 未打开。

```text
H1 all-seed true miss:
  baseline  rho = 0.2652752523518108
  candidate rho = 0.2469151613036028
  joint p = 0.0240975902409759
  verdict = FAIL (both effect sizes below 0.30)

H2 success-subset true miss:
  baseline  rho = 0.5298479127688591
  candidate rho = 0.26567370456427236
  joint p = 0.0522947705229477
  verdict = FAIL

H3 success-subset capture:
  baseline  AUC = 0.34065934065934067
  candidate AUC = 0.3333333333333333
  joint p = 0.8003199680031997
  verdict = FAIL (direction opposite to the posited mechanism)

overall = STRICT125_SIGMA_NOT_OUTCOME_RELEVANT
```

这不是输入无效，也不是“还没有找到 mediator”。Goal46-51 的 association 是在
`success and not goal_capture_eta` 的条件子集内估计的；Goal52 直接问它能否排序
真实 terminal miss 或 capture，答案是否定的。H1 的 permutation p 虽然小于
`0.05`，但两个 block 都没有达到预注册 effect Gate；H2 只在 baseline success
子集为正，candidate 未复现；H3 的方向与“较大 sigma -> 更差 outcome”相反。
按冻结映射不能拼成 `SUPPORTED`。

因此停止把 `nav_drift_sigma` residual association 当作 task-level mechanism。
不追加 endpoint/covariate/窗口/feature，不改阈值或 RNG，不换 seed，不启动
controlled intervention，不训练，不扫 reward/gamma/seed，不恢复 command-residual
PPO/supervised residual，不打开 held-out `100-149`。default H2 的弱通过只作为
caveat，不能用来救 strict125。

若未来继续 terminal failure 研究，必须新开预注册 goal，选择与 sigma 无关的机制、
独立复现设计或 explicit instrumentation，并先证明真实 outcome relevance，再谈
controller intervention。

证据：

- `doc/goals/goal52_sigma_outcome_relevance_audit_20261109.md`
- `doc/goals/goal52_result_20261109.md`
- `artifacts/lowlevel/goal52_sigma_outcome_relevance_audit_20261109/`
- `research/LEDGER.jsonl`
