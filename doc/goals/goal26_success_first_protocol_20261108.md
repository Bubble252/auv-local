# Goal26（草案）：Success-First 评价与训练协议

日期：2026-11-08  
状态：**规划草案，未启动**

## 1. 决策

后续优化以 **success rate 为第一目标**，pre-goal 3D RMSE 降为次级指标和模型选择依据。
Goal25 的历史结论保持为 **FAIL**，不回改旧门禁、不把 D3 追认成 PASS；本文件只定义
新的、可在下一次实验前冻结的协议。

## 2. 当前 success 参照

| scenario | 当前最佳 success | 来源 | 当前 pre-goal RMSE |
|---|---:|---|---:|
| default | 0.7250 | D2/D3 | 2.7883 / 2.8492 m |
| strict125 | 0.5000 | D3 | 2.8291 m |

这些数字只作为 success 基线，不代表 D3 已通过旧协议。

## 3. 评价契约

### 主门禁：success-first

- default success 不低于当前最佳参照 `0.725 - 0.03`；
- strict125 success 不低于当前最佳参照 `0.500 - 0.03`；
- safety 增量不超过 `0.02`；
- plane miss、timeout 不得出现明显恶化；建议相对基线恶化不超过 `0.05`；
- 至少一个主场景的 success 提升 `>=0.03`，或总成功 episode 数增加；
- 不得通过放宽终点判定、改变 success 口径或隐藏 timeout 来提升成功率。

### 次级指标：pre-goal

- 仅在通过 success 主门禁的候选之间比较 pre-goal RMSE；
- pre-goal 恶化超过参照 `3%` 时，必须在报告中列为风险；
- 不把 `2.65 / 2.70 m` 继续当作一票否决的硬门禁，除非未来重新预注册；
- 同时保留 success-only 3D RMSE、pre-goal RMSE、full RMSE、plane miss、
  post-goal overshoot 和 terminal longitudinal RMSE，防止 success 掩盖路径质量问题。

### 评估固定项

- dev seeds 固定为 `20–59`；
- held-out `100–149` 只在最后验证时使用一次；
- endpoint protocol 固定为 `endpoint_stop`；
- 默认和 strict125 两套场景都要报告；
- 所有候选使用同一评估矩阵、同一 seed、同一传感器 profile 和同一 success 口径。

## 4. 训练契约

### BC 阶段

- BC 的目标仍是拟合教师动作，不以 pre-goal 指标直接反传；
- 对高 pre-goal 误差的中段样本只做受控加权，不牺牲终点捕获样本；
- 失败 seed 的定向 DAgger 优先于扩大普通教师数据量；
- 每个数据配方先看 success 是否保持，再看 pre-goal 是否改善；
- 不使用“只优化终点附近”的数据配方来刷 success。

### 残差 PPO 阶段

- 奖励必须以终点成功、任务完成和安全为主，路径误差只作为辅助稠密项；
- 可加入 `-lambda_path * 3D路径误差`、progress、yaw、smoothness 和 energy 项；
- `lambda_path` 采用从小到大课程，先确认 success 不掉，再逐步增加；
- 不用 `pre_goal_3d_rmse` 本身直接作为 episode return，因为它是评估统计量，不是
  天然可优化的奖励；
- 如果路径惩罚导致减速、悬停、绕路或终点捕获下降，立即回退该权重；
- PPO 只有在 BC 的 success 主门禁通过后才启动。

## 5. 实验顺序

### Phase 0：只读基线确认

- [ ] 固定 D2/D3 的 success、safety、plane miss、timeout 和 pre-goal 数值；
- [ ] 确认 success 口径为 `p_nav`，并记录评估配置 hash；
- [ ] 检查旧配置没有被覆盖，D2/D3 产物只读；
- [ ] 运行 ledger 检查并记录本 Goal 的 Research Contract。

### Phase 1：success-first 奖励/数据对照

- [ ] A 组：保持当前训练目标，只重新按 success-first 评价；
- [ ] B 组：加入小权重路径误差奖励，验证 success 是否保持；
- [ ] C 组：加入失败 seed 定向 DAgger 或高误差中段样本加权；
- [ ] 各组固定 seed、样本量、网络和评估矩阵；
- [ ] 先做 1-seed screen，通过后扩到 3 个训练 seed；
- [ ] 只有 success 主门禁通过的配方才进入 40-seed dev 评估。

### Phase 2：条件式残差 PPO

- [ ] 选择 success 最好的 BC 底座；
- [ ] 先跑 zero-residual 基线，确认与 frozen base 一致；
- [ ] 只跑一个小权重路径奖励的 PPO screen；
- [ ] success 不下降后，再评估 pre-goal 和完整路径指标；
- [ ] 若 success 下降，停止该奖励权重，不继续扫参。

## 6. Research Contract

```yaml
autoresearch:
  required: true
  triggers:
    - gate_fail_twice
    - no_metric_movement_after_3_fixes
    - heldout_collapse
    - seed_noise_ambiguous
  ledger: research/LEDGER.jsonl
  blockers_dir: research/blockers/
  reading_dir: research/paper_reading/
  max_outer_loops: 3
  check_command: python ../scripts/check_research_ledger.py --goal Goal26 --goal-doc doc/goals/goal26_success_first_protocol_20261108.md --strict
```

- **Research question**：在 success-first 约束下，路径误差奖励或定向数据修正能否在
  不降低成功率的前提下改善 pre-goal 路径精度？
- **Hypothesis H1**：小权重路径奖励和失败状态 DAgger 能保持 success，并让
  pre-goal RMSE 不再恶化。
- **Falsification**：若任一方案 success 低于参照减 `0.03`，或改善只来自单个 seed，
  则该方案失败，停止该权重/数据配方。
- **Primary metric**：default/strict125 的 success、safety、plane miss/timeout。
- **Secondary metric**：pre-goal 3D RMSE、success-only RMSE、full RMSE 和过冲指标。
- **Budget contract**：每阶段最多一次 1-seed screen、一次 3-seed 扩展；不扫 lr、
  reward 权重、DAgger 比例或 seed 追结果。
- **Negative-result contract**：门禁失败即写负结果并停止，不修改历史指标。

## 7. 停止条件

- success 低于参照减 `0.03`；
- safety 或 plane miss/timeout 明显恶化；
- 改善只来自单个 seed；
- 路径奖励导致减速、悬停、绕路或终点捕获下降；
- 需要改变 success 口径或终点协议才能通过；
- 需要 held-out 才能决策；
- 试图通过扫参数、追加 seed 或事后改门禁追结果。

