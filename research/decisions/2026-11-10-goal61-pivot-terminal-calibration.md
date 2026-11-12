# Goal61 decision: pivot to terminal calibration/termination

日期：2026-11-10<br>
决策：`pivot`

## 触发

Goal61 在 Goal60 trace 上完成只读审计：

- H-C1 支持：success_no_capture 的 p_nav closer fraction 与 direction>0
  在两个 scenario 都是 `1.000`，中位 gap 为 `-0.200/-0.242 m`；
- H-C2 支持：inside/outside 顺序为 `1.000`，但该结果在当前标签定义下接近恒等；
- H-C3 支持：forced zero-residual 也有 `13/13` 个 success_no_capture，
  同样 closer/direction/order 全为 `1.000`。

## 决策

`decision = pivot`。主线从 residual/NoisyNet/算法替换转到 **terminal
calibration/termination**。下一 Goal 只做 frozen-checkpoint 的只读反事实：

1. 收紧 p_nav success 半径；
2. 增加 terminal dwell；
3. 使用在线可得的 nav uncertainty/age margin；
4. 统一 protocol success 与 true eta capture 的判定语义。

先在 dev `20–59` 上冻结候选规则和 Gate，再决定是否值得做闭环验证。reward
composition（H-B1）保持为第二优先级，不与 termination 修复混合。

## 禁止

- 不启动 SAC/REDQ/DroQ；
- 不重开 Goal27；
- 不打开 held-out `100–149`；
- 不把只读诊断写成性能 PASS；
- 不用关闭 NoisyNet、换 seed 或放宽阈值来处理 terminal 语义问题。

## 证据

- `doc/goals/goal61_result_20261110.md`
- `research/blockers/2026-11-10-goal61-terminal-semantics-confirmed.md`
- `artifacts/lowlevel/goal61_terminal_alignment_20261110/goal61_terminal_alignment.json`
