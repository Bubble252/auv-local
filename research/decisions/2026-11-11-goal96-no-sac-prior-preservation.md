# Decision：Goal96 不开 SAC，先修 PPO 先验保持

日期：2026-11-11<br>
关联：Goal96<br>
决定：`pivot`

## 判据

Goal96 完成了 Phase A/B/C 只读审计：

- 方向：12/12 run `DIRECTION_NOT_SUPPORTED`；
- 权限：survival 最低 `0.9192`，`AUTHORITY_SUFFICIENT`；
- reward：名义 terminal share > 0.80，形式 `REWARD_CREDIT_MISALIGNED`；
  但折扣后 terminal 只占 `1.2%–1.4%`，pre-terminal closure 有分离；
- 先验：default initial 31 → final 30，strict 19 → 19，frozen 31/20；
- Phase C：final 与 initial residual 的 mean cosine `0.907–0.969`，但 L2
  ratio `1.10–1.52`，closure delta `-0.035–0.000`。

因此当前问题是：**PPO 在放大已有的监督残差，没有学出新的 terminal 方向，
也没有提高闭环。** 这不是“方向正确但探索/样本效率不足”。

## 为什么不换算法

用户先前约定：只有明确证明“残差方向对，但探索/样本效率不足”时，才值得开
SAC 同协议对照。Goal96 的证据相反：

1. terminal direction CI 全部跨 0；
2. final residual 方向与 initial 高度一致，说明不是探索没找到方向；
3. final 放大幅度后 closure 没有变快，说明“继续加大探索/样本”不是直接解；
4. frozen BC 比 PPO final 更好，说明更新在消耗先验。

直接开 SAC/REDQ/DroQ 会重复同一个 terminal credit/先验保持问题，只是换一个
算法族；这不满足预注册的换算法条件。

## 新路线

预注册一个 **PPO prior-preserving terminal residual** 实验，仍保持 PPO、
reward semantics、success、goal radius 和 held-out 不变。候选机制只允许从
以下方向中选择，并在 Goal 文档中预先写死判据：

1. 对 terminal window 的 PPO residual L2 相对于 initial/frozen prior 加软约束；
2. 对 final 相对 initial 的方向偏离加约束，避免只放大已有残差；
3. 在 terminal window 对无效 closure 方向的 advantage 做诊断/裁剪；
4. 保持 frozen BC 分支完全冻结，检查 PPO 更新是否破坏了边界 seed 的保持。

最先做的应是最便宜的 1-seed screen；只有 screen 同时改善 success 和
terminal closure，才扩 seed 或进入 formal。若 screen 仍无 movement，再按
Research Escalation 重写 blocker，不允许直接扫 lr/gain/seed。

## 不做什么

- 不开 SAC/REDQ/DroQ；
- 不在 Goal96 中途改算法；
- 不改 reward、terminal、success、goal radius；
- 不打开 held-out `100–149`；
- 不把 Phase B 的名义 terminal share 写成唯一原因；
- 不把 `PPO_DEGRADES_PRIOR` 的单个 seed 翻转夸大成统计显著。

## 证据

- `doc/goals/goal96_result_20261111.md`
- `research/blockers/2026-11-11-goal96-terminal-credit-prior-degradation.md`
- `artifacts/lowlevel/goal96_terminal_credit_audit_20261111/phaseA.json`
- `artifacts/lowlevel/goal96_terminal_credit_audit_20261111/phaseB.json`
- `artifacts/lowlevel/goal96_terminal_credit_audit_20261111/phaseC.json`
