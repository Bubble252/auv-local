# Goal100：终端 PPO 先验保持与近场衰减

**日期**：2026-11-11<br>
**状态**：`PROTOCOL_DRAFTED_BEFORE_MEASUREMENT / EVAL_ONLY / NO_TRAINING`<br>
**上游**：Goal99 `SCHEDULE_NOT_SUPPORTED / TERMINAL_DEGRADATION_PRECEDES_2M / NO_SAC`<br>
**性质**：冻结的预注册 eval-only Gate；不训练、不改 reward/terminal/success/goal radius，不打开 held-out。

## Research Contract

```yaml
autoresearch:
  required: false
  triggers:
    - gate_fail_twice
    - no_metric_movement_after_3_fixes
    - heldout_collapse
    - seed_noise_ambiguous
  ledger: research/LEDGER.jsonl
  blockers_dir: research/blockers/
  reading_dir: research/paper_reading/
  max_outer_loops: 3
  check_command: python ../scripts/check_research_ledger.py --goal Goal100 --goal-doc doc/goals/goal100_terminal_ppo_prior_preservation_20261111.md --strict
```

本 Goal 记录前不得修改 profile 集合、衰减距离、dev seed、success frame、scenario、
checkpoint、trace 字段或 Gate 阈值。失败后先补 blocker/decision，再决定是否进入
Research Loop。

## 1. 背景与证据

Goal99 已证明：远场 2.00、近场 1.50/1.75 的距离调度不能修复 strict capture。
三组 schedule 与 constant 2.00 逐 seed 完全相同，strict 仍为 18/40；strict
seed 22/29 的 `2–1.25 m` radial closure 在 S20 与 constant 2.00 上几乎相同，
说明退化发生在进入 2 m 之前。

当前已有关键配对结果：

| profile | residual scale | PPO | default | strict125 |
|---|---:|---|---:|---:|
| `P0_15` policy | 1.50 | 全开 | 30 | 19 |
| `P0_20` policy | 2.00 | 全开 | 31 / 32 | 18 / 18 |
| `B15` frozen-base | 1.50 | 关闭 | 31 / 31 | 20 / 20 |

其中 `P0_20` 的 default 是 D3/BC15 两臂，strict 也是两臂。`B15` 说明：
在 scale 1.50 下，只要关闭 PPO 增量，strict 可以恢复到 frozen BC 的 20/40，
同时 default 保持 31/40。现在缺少两个直接对照：

1. **BC-only scale 2.00**：只放大 frozen BC 残差、关闭 PPO，是否同时得到
   default `>=31` 和 strict `>=20`？
2. **近场 PPO 衰减**：保留 PPO 的远场行为，只在 `3 / 2 / 1.25 m` 内关闭或
   衰减 PPO 增量，是否恢复 strict 而不损失 default？

这两个问题都只需要 eval，不需要训练、SAC 或新 reward。

## 2. 网络结构对照

本 Goal 不改网络。Goal99 已按用户要求与历史代码逐项比对：

- 当前 `pc_policy.py` SHA256 与 Goal89 parity 审计记录完全一致：
  `2e1ebe0c0747b3abf35507c4407a1434dfaaaf8c59970484e122e409acabb1a6`；
- Goal86 P-A BC checkpoint SHA：
  `80024ef5798cfbc9701b974ab14b0eb5ce082daaea9c665978431978d59ff9dd`；
- Goal95 两个 checkpoint 的 16 个冻结浮点张量与 Goal86 `max_abs_diff=0.0`；
- actor `131→256→128→3` ELU，critic `131→256→128→1` ELU；
- state encoder `20→128→128→60`，path encoder `8→64→64`，
  endpoint encoder `4→32→64`，current encoder `3→64→60`；
- `d_z_total=64`、`n_z_reserved=4`、DADE off、无 NoisyNet；
- 评价走确定性 `act_inference()` / `residual_action()`。

因此本 Goal 不允许把结果归因于“网络接错、NoisyNet、容量不足或 checkpoint
错配”。如果 `B15` 不能复现 31/20，先判 `INVALID` 并查评价链，不分析算法。

## 3. 假设

### H100-A：BC-only scale 2.00 已经足够

如果成立：

- `B20` 两臂 default `>=31`；
- `B20` 两臂 strict `>=20`；
- path 3D RMSE 相对 `B15` 恶化 `<=5%`；
- timeout/safety 不增加。

判读：PPO 在当前 checkpoint 上没有必要的闭环贡献；下一阶段应优先保留
frozen BC，并训练 terminal attenuation / prior-preservation，而不是继续放大
或重训 PPO。

### H100-B：近场 PPO 是 strict 退化的直接原因

如果成立：

- 至少一个 `A3/A2/A125` 在两臂达到 default `>=31`、strict `>=20`；
- 近场 attenuation zero fraction 与距离边界一致；
- strict 22/29 的 `2–1.25 m` closure 恢复到 `B15` 附近；
- default 的 gained seed 27/49 不丢。

判读：PPO 的远场增量可以保留，但训练时必须加 terminal attenuation /
prior-preservation；不需要换 SAC。

### H100-C：BC/ALOS 的 terminal capture 几何才是主因

如果成立：

- `B20` 和所有 attenuation profile 都不能同时达到 default `>=31`、
  strict `>=20`；
- 或 strict 恢复但 default 系统性退回 30；
- 失败仍集中在 `goal_plane_miss`，且 closure 与 PPO 开关无关。

判读：问题在 BC/ALOS 的 capture 几何或终末观测，不在 PPO 优化；下一步转
terminal geometry / observation audit，不换算法。

### H100-D：评价链无效

如果成立：

- `B15` 不能复现 31/20；
- checkpoint SHA、seed 集合、success frame 或 trace 字段不匹配；
- attenuation 边界写回与 trace 不一致。

判读：`INVALID`，先修评价链；不把无效结果写成算法 FAIL。

## 4. 数据与隔离

- 训练：不训练；
- eval：dev seeds `20–59`，每个 run 40 episodes；
- scenario：Goal85 unbiased default 与 strict125；
- checkpoint：Goal95 `P_D3/final.pt`、`P_BC15/final.pt`；
- success frame：固定 `both`；
- 固定 checkpoint、config、goal radius、reward、termination 和 eval split；
- held-out `100–149` 禁止打开；
- 复用已有 `P0_15`、`P0_20`、`B15` 结果作为配对参照，只新跑 `B20` 与
  `A3/A2/A125`；
- 所有新 run 必须写 `summary.json` 与 `trace.npz`。

## 5. 预注册 profile 矩阵

| ID | action mode | residual scale | PPO 处理 | full-off 距离 | full-on 距离 |
|---|---|---:|---|---:|---:|
| `P0_15` | policy | 1.50 | checkpoint 原样 | — | — |
| `P0_20` | policy | 2.00 | checkpoint 原样 | — | — |
| `B15` | frozen-base | 1.50 | 全局关闭 | — | — |
| `B20` | frozen-base | 2.00 | 全局关闭 | — | — |
| `A3` | attenuated-residual | 2.00 | 近场衰减 | 3.00 m | 5.00 m |
| `A2` | attenuated-residual | 2.00 | 近场衰减 | 2.00 m | 4.00 m |
| `A125` | attenuated-residual | 2.00 | 近场衰减 | 1.25 m | 3.25 m |

attenuation 定义：

```text
attenuation(d) = 1 - smooth_terminal_gate(d, enter_m, full_m)
PPO_term = checkpoint_gate * attenuation * residual
```

- `d <= full_m`：attenuation `= 0`，PPO 关闭；
- `d >= enter_m`：attenuation `= 1`，PPO 恢复；
- 中间为 smoothstep；
- checkpoint 原有的 terminal gate 仍保留；本 Goal 只额外乘 attenuation，
  不改训练配置或 wrapper 的 gate 语义。

新跑矩阵：

```text
4 profiles × 2 arms (P_D3, P_BC15) × 2 scenarios
= 16 eval runs
```

每个 run：

- `--seed-start 20 --seed-count 40`；
- `--success-frame both`；
- CPU/float64；
- 单线程 BLAS，8 个进程并行；
- `--actuator-residual-scale-override 2.00`；
- 写 `summary.json/md` 和 `trace.npz`。

## 6. 命令模板

`B20`：

```bash
/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python   scripts/evaluate_pc_ppo.py   --training-config configs/lowlevel/goal95_20261111/p_d3_ppo.json   --checkpoint artifacts/lowlevel/goal95_supervised_residual_20261111/P_D3/final.pt   --environment-config configs/lowlevel/pc_goal85_unbiased_default_candidate_20261110.json   --output artifacts/lowlevel/goal100_terminal_prior_20261111/P_D3_default_B20   --seed-start 20 --seed-count 40 --success-frame both --device cpu   --action-mode frozen-base   --actuator-residual-scale-override 2.00   --trace-output artifacts/lowlevel/goal100_terminal_prior_20261111/P_D3_default_B20/trace.npz
```

`A3`：

```bash
/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python   scripts/evaluate_pc_ppo.py   --training-config configs/lowlevel/goal95_20261111/p_d3_ppo.json   --checkpoint artifacts/lowlevel/goal95_supervised_residual_20261111/P_D3/final.pt   --environment-config configs/lowlevel/pc_goal85_unbiased_default_candidate_20261110.json   --output artifacts/lowlevel/goal100_terminal_prior_20261111/P_D3_default_A3   --seed-start 20 --seed-count 40 --success-frame both --device cpu   --action-mode attenuated-residual   --actuator-residual-scale-override 2.00   --ppo-residual-attenuation-full-m 3.00   --ppo-residual-attenuation-enter-m 5.00   --trace-output artifacts/lowlevel/goal100_terminal_prior_20261111/P_D3_default_A3/trace.npz
```

## 7. 指标

主指标：

- `success`（both-frame）；
- `success_pnav_tight`、`success_eta_tight_ever`、`success_eta_tight_at_done`；
- `timeout`、`safety`、`goal_plane_miss_rate`；
- path 3D/horizontal/vertical RMSE 相对 `B15` 的比值。

机制指标：

- `2–1.25 m` 与 `3–2 m` 的 radial closure；
- strict seed 22/29 是否从 miss 恢复到 success；
- default seed 27/49 是否保持 success；
- attenuation mean / zero fraction / full fraction；
- `actuator_residual_ppo_applied/effective`、magnitude/rate clip；
- 同 seed 的 gained/lost 集合；
- trace 中 attenuation 距离与 profile 边界的一致性。

## 8. Gate

### 8.1 `INVALID`

任一条件成立：

- `B15` 不能复现两臂 default `31/40`、strict `20/40`；
- checkpoint SHA 不匹配；
- seed 集合不是 `20–59`；
- attenuation profile 缺 summary/trace 字段；
- trace 的 attenuation 边界与写入参数不一致；
- NaN/Inf、success 口径改变或读取 held-out。

### 8.2 `BC_SCALE_ONLY_SUPPORTED`

`B20` 同时满足：

- D3 与 BC15 default `>=31`；
- D3 与 BC15 strict `>=20`；
- path 3D RMSE 相对 `B15` 平均恶化 `<=5%`；
- timeout/safety 不增加。

### 8.3 `ATTENUATION_SUPPORTED`

至少一个 `A3/A2/A125` 同时满足：

- 两臂 default `>=31`；
- 两臂 strict `>=20`；
- path 3D RMSE 相对 `B15` 平均恶化 `<=5%`；
- timeout/safety 不增加；
- attenuation 机制指标与 profile 边界一致。

`B20` 也通过时优先记 `BC_SCALE_ONLY_SUPPORTED`，attenuation 只作为机制解释。

### 8.4 `PRIOR_PRESERVATION_PARTIAL`

有 profile 在单 arm 或单 scenario 达到阈值，但不满足 8.2/8.3。只做归因，
不直接 formal。

### 8.5 `PRIOR_PRESERVATION_NOT_SUPPORTED`

所有 profile 都不能同时达到两臂 default `>=31`、strict `>=20`。下一步转
BC/ALOS terminal capture 几何审计，不继续扫 attenuation 距离，不开 SAC。

## 9. 执行计划

- [x] Step 0：确认 Goal99 结论、`B15` 复现和网络 parity；
- [x] Step 1：实现 eval-only `attenuated-residual` 与 trace/summary 字段；
- [x] Step 2：补单测与 1 seed smoke；
- [x] Step 3：远端跑 16 个 dev40 eval；
- [x] Step 4：按冻结 Gate 判断；
- [x] Step 5：补 result/blocker/decision、LEDGER、STATE，并运行 checker。

## 10. 不做什么

- 不训练 PPO/SAC/REDQ/DroQ；
- 不改 reward、terminal、success、goal radius；
- 不重写网络主干，不打开 NoisyNet；
- 不扩 seed，不打开 held-out；
- 不用 `B15` 的旧结果冒充 `B20` 或 attenuation 结果；
- 不把 `PRIOR_PRESERVATION_PARTIAL` 写成 `SUPPORTED`；
- 不在本 Goal 内继续扫 attenuation 距离或 scale。

## 11. Git

禁止 `git add -A`。只 add 本次相关文件，提交前运行：

```bash
python ../scripts/check_research_ledger.py --goal Goal100 --goal-doc doc/goals/goal100_terminal_ppo_prior_preservation_20261111.md --strict
git diff --cached --check
```

不自动 commit/push；不提交 `artifacts/`、checkpoint、日志和 trace。

## 12. Goal 提示词

```text
你现在执行 Goal100：终端 PPO 先验保持与近场衰减。

先读：
- doc/goals/goal100_terminal_ppo_prior_preservation_20261111.md
- doc/goals/goal99_result_20261111.md
- doc/goals/goal89_network_parity_and_gate_scope_audit_20261111.md
- research/AUTORESEARCH_LOOP.md

硬约束：
1. 固定 Goal95 checkpoint、dev seeds 20–59、default+strict125、success frame both；
2. 不训练、不扩 seed、不打开 held-out、不换 SAC/REDQ/DroQ；
3. 不改 reward、terminal、success、goal radius 或网络主干；
4. 先复现 B15 的 31/20，否则判 INVALID；
5. attenuation 只作用于 PPO increment，保留 frozen BC；
6. 失败后先归因，不直接扫距离或改算法。

执行顺序：
- 实现 attenuated-residual、trace 和 summary；
- 单测 + 1 seed smoke；
- 远端跑 B20/A3/A2/A125 共 16 个 dev40 eval；
- 分析 strict 22/29 与 default 27/49 的逐 seed 翻转；
- 按冻结 Gate 判 BC_SCALE_ONLY_SUPPORTED / ATTENUATION_SUPPORTED / PARTIAL / NOT_SUPPORTED / INVALID；
- 写 result、blocker/decision、LEDGER、STATE，并运行 checker。
```

## 13. 自主迭代契约

所有后续动作必须执行 `research/AUTORESEARCH_LOOP.md`：

1. 失败后先归因，不直接换算法或扫 seed；
2. 每次 blocker 至少写两个可证伪 hypothesis；
3. 使用 `paper-search` / `deep-research` 查论文，找到可用代码，写 reading note；
4. 先跑最便宜的 falsification，再决定 `continue` / `pivot` / `stop`；
5. `pivot` 时写新的 Goal 文档和新的 prompt，不修改失败 Gate 的阈值；
6. 持续迭代，直到满足成功标准或触发必须向用户报告的停止条件；
7. 不自动 commit/push，不打开 held-out，不改变任务成功定义或主算法族。

成功标准、停止条件、论文/代码检索要求和自动执行边界见
`research/AUTORESEARCH_LOOP.md`。
