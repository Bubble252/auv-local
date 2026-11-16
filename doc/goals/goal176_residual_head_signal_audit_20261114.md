# Goal176：Residual head 信号与纵向敏感性诊断

**日期**：2026-11-14<br>
**状态**：`STAGE_A_COMPLETE / PROBE_PASS / RESIDUAL_ACTION_ATTENUATED / PPO_SIGNAL_WEAK_PARTIAL / H176C_NOT_SUPPORTED / HELDOUT_CLOSED`

## Research Contract

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
  check_command: python ../scripts/check_research_ledger.py --goal Goal176 --goal-doc doc/goals/goal176_residual_head_signal_audit_20261114.md --strict
```

- `required: true`：诊断型 goal；若结论进入训练或架构修改，必须沿用
  Goal175 blocker 和 decision，不覆盖失败 Gate。

## 1. 背景与证据

Goal175 的正式结论是 `VERTICAL_GUARD_FAIL`。ALOS-COMP 基座带来 route/cross
收益，但训练后的 `P_RUD_075_COMP` 相对同场景 `zero_bc50` 只有亚毫米级闭环差异。
现有证据留下三个高优先级假设：

1. PPO 从 residual 获得的边际 advantage/credit 太弱，策略停在 BC 附近；
2. residual head 没有把 `z_t` 中的纵向/俯仰表征转成有效动作；
3. `z_t` 本身缺少纵向信息。

本 goal 只做只读诊断，先用现有 6 个 Goal175 A2 checkpoint 和 eval 产物，
不改 reward、termination、success、goal radius，不打开 held-out，不训练
Transformer。

## 2. 假设与证伪

### H176-A：PPO 的 residual 更新/credit 太弱

- 成立预期：训练 KL 远低于 `desired_kl`，residual policy 的 eval return 不高于
  `zero_bc50`，且 actor head 长期只产生近 BC 的小 residual。
- 失败预期：KL 接近目标、return 明确高于 zero baseline，residual 仍无路线作用；
  此时优先检查 reward alignment，而不是“训练信号弱”。
- 最便宜证伪：读取 6 个训练 run 的 `iteration_records.json`，并对同一 eval
  scenario 比较 `P_RUD_*_COMP`、`zero_bc50`、`zero_alos_comp` 的 return。

### H176-B：residual head 不响应纵向/俯仰输入

- 成立预期：`z_t` probe 可预测 `vertical_error / pitch_error / q / w`，但有限差分
  测得的 rudder residual 对 path vertical/pitch 的 Jacobian 接近零，或显著小于
  horizontal reference channel。
- 失败预期：Jacobian 对纵向/俯仰输入有清晰、方向一致的响应；此时不能把动作
  惰性归因于 head 接口。
- 最便宜证伪：在真实 eval actor observation 上对 path vertical、path pitch、
  state q/w 做中心差分，同时以 path cross/heading 作为参考通道。

### H176-C：`z_t` 本身缺少纵向信息

- 成立预期：用 `z_t` 线性回归 `vertical_error / pitch_error / q / w` 的 held-out
  dev-seed R² 很低。
- 失败预期：held-out R² 明显高于随机/常数基线，说明信息仍在 latent。
- 最便宜证伪：现有 checkpoint 的 eval-only linear probe；不训练大模型。

## 3. 数据与隔离

- 数据源：Goal175 A2 checkpoint、`current10 × default` eval 配置和 dev seeds；
- probe/Jacobian 使用 dev seeds `20–27`；`20–25` 训练 probe，`26–27` 测试；
- 不读取或打开 held-out `100–149`；
- 不修改任何训练 checkpoint、config、eval split 或 Gate 阈值。

## 4. 方法

### 4.1 PPO 信号审计

读取 6 个 A2 训练 run 的 `iteration_records.json`，汇总：

- `approx_kl` 与目标 `desired_kl=0.005`；
- clip fraction；
- actor head parameter update norm；
- action std、entropy 和 logged reward 趋势；
- eval `mean_episode_reward` 相对 `zero_bc50` / `zero_alos_comp` 的差值。

不把 mean reward 单独当作因果证据；它只用于判断“是否明显在优化”。

### 4.2 `z_t` linear probe

通过临时包装 eval 的 `_build_policy` 捕获真实 actor observation，复用正常
evaluator 的前向路径计算：

```text
z_t = state encoder + path encoder + current encoder
```

目标为：

- signed vertical error（path block 第 1 列 × 5 m）；
- tangent pitch error（path block 第 3 列 × π/2）；
- pitch rate `q`（actor state 第 7 列）；
- heave velocity `w`（actor state 第 11 列）。

用 ridge linear probe 在 seed `20–25` 训练、`26–27` 测试，固定
`ridge_alpha=1.0`，输入先按训练集标准化，报告 test R² 和 Pearson correlation；
若 R² 受闭环保真影响，只作为敏感性证据，不单独声明因果。probe 使用 evaluator
实际送入 actor 的完整 64-D `z_t`；不把 path 原始列或 command 拼进 probe 输入。

## 4.3 finite-difference Jacobian

对每个采样 observation 做中心差分：

- path vertical error：±0.10 m；
- path tangent pitch error：±0.05 rad；
- state `q`：±0.05 rad/s；
- state `w`：±0.05 m/s；
- reference：path cross-track ±0.10 m、path heading ±0.05 rad。

报告 raw residual head 和 applied residual 的：

- `median |d rudder / d input|`；
- `|vertical| / |cross|`、`|pitch| / |heading|` 比值；
- 导数符号一致性；
- 0.1 m / 0.05 rad 扰动产生的物理 rudder angle 变化。

Jacobian 采样固定为每个 seed 最多 32 个沿轨迹均匀分布的决策点，总计最多 256 个；
ratio 用 rudder 通道的 `median |d action / d physical input|` 计算。

## 5. 执行计划

### 5.1 不可协商的运行契约

| 项目 | 强制要求 | 失败标签 |
|---|---|---|
| 训练 | 本 goal Stage A 不训练 | `INVALID_UNPLANNED_TRAINING` |
| 课程 | 不改变现有 Goal175 训练 provenance | `INVALID_NONCURRICULUM` |
| 评价 | 只读 dev seeds 20–27；不打开 held-out | `INVALID_SEED_SPLIT` |
| 产物 | probe/Jacobian JSON、summary md、命令与 SHA256 | `INVALID_INCOMPLETE_AUDIT` |
| 结论 | 不修改 reward/termination/success/radius；不把诊断写成 Gate pass | `INVALID_CLAIM` |

- [x] Step 0：写入本文档并固定 probe/ Jacobian 参数；
- [x] Step 1：实现 PPO signal audit；
- [x] Step 2：实现真实 eval observation probe/Jacobian；
- [x] Step 3：服务器只读运行 seeds 20–27；
- [x] Step 4：写 result md、更新 ledger/registry；
- [x] Step 5：根据结果决定是否进入 last-layer pilot；本轮记录为下一项最小 pilot，不训练。

## 6. Gate / 判读

| check | 阈值 | 失败标签 |
|---|---|---|
| probe 可预测性 | held-out R² > 0.50 视为“信息清楚”；0.10–0.50 视为“弱/混合”；<0.10 视为“不可用” | `PROBE_POOR` |
| vertical Jacobian | `|d rudder/d vertical|` 至少达到 cross reference 的 10%，且方向一致率 > 0.8 | `VERTICAL_JACOBIAN_INERT` |
| PPO signal | late KL 中位数 < 10% `desired_kl` 且 residual return 不高于 zero_bc50 | `PPO_SIGNAL_WEAK` |

判读规则：

| 观察 | 判读 | 下一步 |
|---|---|---|
| probe 好，vertical Jacobian 惰性 | head/action interface | 进入 head/action interface 修复，不先换 encoder |
| probe 差，Jacobian 也惰性 | latent/表征或输入接线 | 考虑预训练目标、显式纵向 skip 或 DADE |
| probe 好，Jacobian 有响应，但 PPO signal 弱 | credit/优化 | last-layer pilot + reward/advantage 审计 |
| 三者都正常 | 旧 checkpoint 特例 | 重新评估训练预算/seed，不扩 Gate |

## 7. Git

禁止 `git add -A`。只 add 本 goal 相关文件，提交前运行：

```bash
python ../scripts/check_research_ledger.py --goal Goal176 --goal-doc doc/goals/goal176_residual_head_signal_audit_20261114.md --strict
git diff --cached --check
```

本 goal 未明确要求不自动 commit/push；如无用户指示，不执行 commit/push。

## 8. Goal 提示词

```text
你现在执行 Goal176：验证 residual head 的 PPO 信号、z_t 纵向 probe 和
vertical/pitch Jacobian。

先读：
- doc/goals/goal176_residual_head_signal_audit_20261114.md
- doc/goals/goal175_result_20261114.md
- research/blockers/2026-11-14-goal175-vertical-guard-residual-inert.md

硬约束：
1. 不训练 Transformer；
2. 不改 reward / termination / success / goal radius；
3. 不打开 held-out；
4. 只使用 dev seeds 20–27；不扩 seed 救 Gate；
5. 先跑只读 PPO signal audit、z_t linear probe 和 finite-difference Jacobian；
6. 不把诊断结论写成 Goal175 Gate PASS。

验收：
- 产出 JSON + Markdown；
- 报告 probe test R²、vertical/cross 和 pitch/heading Jacobian 比值、KL/return 证据；
- 根据结果选择 continue / pivot / stop，并写 ledger。
```

## 9. 自主迭代契约

执行 `research/AUTORESEARCH_LOOP.md`。若 probe/Jacobian 结果仍无法区分
“head 不可用”与“credit 不可用”，先进入 last-layer-only 最便宜 pilot，不直接
训练 Transformer；若触发 hard blocker，按根 `AGENTS.md` 写 blocker、至少两个
可证伪 hypothesis 和 reading note。
