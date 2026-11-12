# Blocker：Goal136 环境多样化触发 SAC 安全护栏，PPO 残差近零

**日期**：2026-11-11<br>
**Goal**：Goal136 Phase A1<br>
**状态**：`DIVERSITY_SAFETY_STOP / PPO_RESIDUAL_COLLAPSE_NOT_DIVERSITY`<br>
**触发条件**：D1/D2 SAC 在 combined20/plant20 超过同 level N0 的 timeout/safety 护栏；PPO D1/D2 与 N0 几乎完全相同且 residual authority `<0.01`。

## 现象

- SAC：`D1/D2 × 3 seeds × 1M`，60 个 dev40 评估全部 finite；D1 current20 有 2 个 timeout，combined20 有 3 个 timeout/safety；D2 combined20 有 4 个 timeout/safety。
- D2 虽有 2/3 seed 在部分 stress level 出现 strict +2，但 nominal default 比 N0 低 13，且出现 safety 终止，不能判 diversity signal。
- PPO：N0/D1/D2 在 nominal 的 strict/default 都是 `22/32`；D1/D2 residual/base L2 仅 `0.0075/0.0056`。
- PPO 5×/20×/50× 放大学习残差没有带来 success 增益，50× 出现下降。

## 已排除

- 不是评估契约：SAC 60 runs、PPO 90 runs 全部 finite，success/terminal 契约未改。
- 不是 held-out 泄漏：`100–149` 未打开。
- 不是单一 seed 结论：两个算法均按 3 个训练 seed 评估。
- 不是 PPO 幅度不足：放大同一方向反而下降。
- 不是 NoisyNet：本轮 PPO/SAC 均无 NoisyNet。

## 可证伪 hypotheses

### H136b-A：终端样本密度不足是主因

- 机制：35 m 长 episode 中成功/失败由最后几米决定，但 PPO/SAC 的 batch 里 terminal transition 占比过低。
- 成立：近终端起始课程在同协议 1M screen 中产生跨 seed、可迁移的 stress/default 增益，且不触发 safety。
- 失败：课程仍无 success 移动，或只增加 timeout/safety。
- 最便宜证伪：短时域/近终端起始 curriculum，PPO 与 SAC 分开训练，各 3 seeds、1M。

### H136b-B：PPO 的信用分配已经把可学习方向耗尽

- 机制：现有 reward/path/current 表征下，PPO deterministic residual 的方向本身无闭环价值，不是幅度或环境覆盖问题。
- 成立：即使增加 terminal 样本密度，PPO residual 的 success 贡献仍为 0 或负，且放大后继续 harm。
- 失败：near-terminal 课程使 PPO 在 stress level 上出现 `>=2/3` seed 的 strict +2。
- 最便宜证伪：复用本轮 PPO checkpoint 做 action-only/terminal-only 诊断，再跑 1M near-terminal screen。

### H136b-C：SAC 的 replay 需要 terminal-focused sampling，而不是更宽分布

- 机制：D-line 增加了 current/plant/path 宽度，但没有提高 terminal state 的 replay 密度；timeout 增加说明覆盖更宽只扩大了大残差的机会。
- 成立：SAC 在 near-terminal curriculum 下比 D1/D2 更少 timeout，并在 stress 上出现可迁移增益。
- 失败：SAC 仍 harm，说明不是 replay 密度，而是 reward/action 方向。
- 最便宜证伪：同 checkpoint/同 reward，只改 episode 起始进度或路径长度，1M × 3 seeds。

## 决策

```text
decision = pivot
next = 预注册 Goal138：near-terminal / short-horizon curriculum，PPO 与 SAC 分开 1M screen；
       固定 reward/terminal/success/radius，不打开 held-out，不直接开 REDQ/DroQ。
```
