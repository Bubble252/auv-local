# Goal51 decision：state proxies 不足，stop retrospective proxy line

日期：2026-11-09<br>
Goal：Goal51<br>
诊断裁决：`UNRECORDED_STATE_PROXIES_INSUFFICIENT`<br>
Ledger decision：`stop`

Goal51 在 Goal47 的四个 formal trace block 上完成只读 Stage A。输入、reference rho、
160 条 records、四 block 和 permutation 均有效；13/14 个冻结代理可评估，coverage
Gate 通过。但 filter-consistency、DVL-calibration 和 actuator-response 三个 family
都没有任何特征在两个 strict125 block 同时满足 `rho <=0.30`、`drop >=0.20`。

```text
strict125 baseline rho  = 0.8336161586462849  (reference error 0.0)
strict125 candidate rho = 0.8857851797221404  (reference error 0.0)
coverage = 13 / 14 (required 10)
valid permutation draws = 9999 / 10000
global p = 1.0
overall = UNRECORDED_STATE_PROXIES_INSUFFICIENT
```

唯一不可评估的 `actuator_best_lag_corr_late` 不是数据丢失，而是所有 seeds 的 late
requested fin norm 都是常数，导致 Pearson correlation 无定义。不能把它写成 lag
mechanism 已被证伪，也不能据此启动 replay。

Stage B 的三个预注册触发条件均未满足：coverage 没有低于 `10/14`；没有
PARTIAL/SUPPORTED 候选依赖精确未记录参数；family/global permutation 和 coverage
均有效。因此不启动 instrumentation replay。

决定 `stop` 的是“在当前 Goal47 trace 上继续用低维 state proxies 解释 strict125
sigma residual”这条路线，不是宣称 association 不存在或底层机制已被完全排除。
不追加特征、不换窗口/阈值、不重跑 Stage B、不恢复 command-residual PPO 或
supervised residual、不扫 reward/gamma/seed、不打开 held-out `100-149`。

后续若要继续，必须新开预注册 goal，并明确选择 controlled intervention、独立的
outcome-relevance 诊断，或显式记录一个具体 hidden state。不能把“文献说还有状态”
当作绕过 Stage B trigger 的理由。

证据：

- `doc/goals/goal51_unrecorded_state_proxy_audit_20261109.md`
- `doc/goals/goal51_phase0_audit_20261109.md`
- `doc/goals/goal51_result_20261109.md`
- `research/blockers/2026-11-09-goal51-state-proxies-insufficient.md`
- `research/paper_reading/2026-11-09-goal51-state-proxy-null.md`
- `artifacts/lowlevel/goal51_unrecorded_state_audit_20261109/`
