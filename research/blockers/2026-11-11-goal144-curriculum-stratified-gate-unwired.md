# Goal144 blocker：课程分层 Gate 未接线，总体过阈值也永不晋级

**日期**：2026-11-11<br>
**Goal**：Goal144 / Goal136 C-line<br>
**状态**：`CURRICULUM_STRATIFIED_GATE_UNWIRED / RUN_INVALIDATED / ROOT_CAUSE_IDENTIFIED`

## 现象

Goal144 的 `C-full-alos-command` 与 `C-full-alos-actuator` PPO 在 C0
100k dev40 已经达到：

- default success `0.925`
- strict125 success `0.925`
- plane miss `0.075`
- timeout `0`
- safety `0`

这已经超过 C0 Gate 的 `0.90 / 0.70 / 0.10 / 0.10 / 0`，但
`curriculum_decisions.jsonl` 仍写 `CONTINUE / gate not met`。如果继续跑到 1M，
会持续生成“总体指标已过、却永远不晋级”的无效课程日志。

## 根因

不是阈值太高，也不是 PPO 没有效果，而是分层 Gate 的数据通路缺失：

- `default_curriculum_stages()` 的 C0–C6 都给 `StageGate` 设了
  `stratified_default_success_min`（C0 `0.80`）。
- `StageGate.gate_passes()` 在该字段非空时要求
  `StageEvaluation.stratified_default_success` 非空，且所有分层都不低于阈值。
- `parse_eval_summary()` 从未从评测 episode 读取或生成分层，只返回 `None`。
- 因此 `gate_passes()` 永远返回 False；`CONTINUE` 不是“差一点”，而是接线错误。

## 已排除

| 项 | 证据 |
|---|---|
| C0 阈值错误 | 100k default `0.925`、strict `0.925`、plane miss `0.075`，均已过阈值 |
| PPO 优化失败 | 同一 100k 指标已超过 C0 阈值；问题发生在 Gate 判定层 |
| reward / terminal / success / radius | 本轮未改，且评估 summary 的 success/plane-miss 口径正常 |
| dev seed 或 held-out | dev 固定 20–59，held-out 100–149 未打开 |
| curriculum 状态机缺失 | 状态机代码存在，问题只是分层输入永远为空 |

## 可证伪假设

**H144-E：补齐 episode 分层后，C0 会在连续两次评估出现 `CONTINUE -> PROMOTE`。**
最便宜证伪：修复后用新 suffix 重启 command/actuator PPO，检查 C0 100k/200k 的
`curriculum_decisions.jsonl`；若总体过阈值但分层未过，记录具体分层；若仍永不晋级，
检查 `parse_eval_summary()` 是否拿到 `curriculum.stratum`。

**H144-F：分层会暴露总体指标掩盖的局部退化。** 若 C0 两个 start-heading 桶中
有一个低于 `0.80`，则不能晋级，即使总体 `>=0.90`。这不是失败阈值放宽问题，而是
课程要求“每个起点 bucket 都过关”。

## 修复

1. evaluator 为每个 episode 写 `curriculum` 元数据，包含 stage、depth、path family、
   noise、current、seed 与 `stratum`；C0 用初始航向符号两桶，C1/C2/C6 用 path family，
   C3/C4 用深度半区，C5 用 current period 慢/快半区。
2. `parse_eval_summary()` 从 episode 元数据聚合每个 stratum 的 default success；
   stratified Gate 缺少元数据时显式 `ValueError`，不再静默返回 `None`。
3. `combine_eval_summaries()` 接收当前 stage 的 `StageGate`，只对 default-radius
   评估做分层聚合。
4. 加回归测试：缺失分层必须报错；单个分层低于阈值时总体再高也不能过 Gate。

## 处置

- 停止 Goal144 旧 suffix `S_s86201` 的两条 PPO；旧 artifacts 保留，但课程决策日志
  标记为 `INVALID`，不作为晋级证据。
- 修复后用新 suffix `S_s86201_gatefix1` 重启，不覆盖旧目录。
- 保持 C0/C1 阈值、reward、terminal、success、radius、dev seeds、算法不变。
- Goal140 仍为 `P2`，不得提前启动。

## 追加：回滚后的 per-stage 预算错误（2026-11-11）

gatefix1 在 C0→C1→C0 回滚后，500k 直接判 `STAGE_BUDGET_EXHAUSTED`。根因是
`CurriculumPromotionController.observe()` 用**全局** `completed_transitions` 与
当前 stage 的 `hard_cap_transitions` 比较；回滚到 C0 时全局已到 500k，超过 C0
的 400k cap，因而没有执行文档要求的“回退后补 anchor”。

修复：

- controller 维护每个 stage 的累计 consumed transitions；
- promotion/rollback 只切换 stage，不清零该 stage 的累计预算；
- budget 判定使用 `stage_consumed_transitions >= hard_cap_transitions`；
- 回归测试覆盖 `C0 pass -> C1 fail -> rollback C0 -> 500k CONTINUE -> 600k exhausted`。

gatefix2 实测（command 与 actuator 两条臂一致）：

| transitions | stage | decision | default / strict | plane miss | 分层 |
|---:|---|---|---:|---:|---|
| 100k | C0 | `CONTINUE` | `0.925 / 0.925` | `0.075` | `nonneg 0.952 / neg 0.895` |
| 200k | C0 | `PROMOTE -> C1` | `0.925 / 0.925` | `0.075` | 同上 |
| 300k | C1 | `CONTINUE` | `0.65 / 0.65` | `0.35` | `P0 0.633 / P1 0.700` |
| 400k | C1 | `ROLLBACK -> C0` | `0.65 / 0.65` | `0.35` | 同上 |
| 500k | C0 | `CONTINUE` | `0.925 / 0.925` | `0.075` | 同上 |
| 600k | C0 | `STAGE_BUDGET_EXHAUSTED` | `0.925 / 0.925` | `0.075` | 同上 |

这说明分层、滞后晋级、严重失败回滚和 per-stage 预算都已真实执行；C1 是当前
课程瓶颈，不是 Gate 接线问题。gatefix1 在 400k 之后的决策日志不再作为课程证据；
gatefix2 的 100k–600k 决策是当前有效证据。
