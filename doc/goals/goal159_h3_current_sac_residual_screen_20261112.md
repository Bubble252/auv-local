# Goal159：H3 强基线 + current07–10 的 SAC 残差重训 screen

**日期**：2026-11-12<br>
**状态**：`GOAL159C_COMPLETED / GATE_FAIL_RESIDUAL_ACTIVE_NO_CLOSED_LOOP_GAIN / HELDOUT_CLOSED`<br>
**结果**：[goal159c_result_20261112.md](goal159c_result_20261112.md)<br>
**上游**：Goal157 H3 candidate、Goal158 H3×1M near10_2 配对、Goal150 current07 K4 screen、Goal86 T2-SAC<br>
**性质**：1M SAC screen；不改 reward/terminal/success/radius，不扩 seed 救 Gate，不打开 held-out。

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
  check_command: .venv/bin/python ../scripts/check_research_ledger.py --goal Goal159 --goal-doc doc/goals/goal159_h3_current_sac_residual_screen_20261112.md --strict
```

## 1. 背景与证据

Goal157/158 已确认：

- K2 zero 显著低估传统控制：nominal strict125 `26/40` vs K4 `40/40`；
- H3（`pitch_gain=4`、`heading_gain=3`）在 current10 是 `31/40`，current15 `17/40`；
- 用 H3 zero 配对 Goal86 1M near10_2：current07 `99/120 -> 105/120`（`+6`），
  current10 `93/120 -> 94/120`（`+1`），current15 `51/120 -> 51/120`（`0`）；
- 这说明 1M 残差在中等 current 有一点独立价值，但在 current10/15 基本消失；
- Goal150 的 SAC full residual 在 K4 current07 上有害，说明动作参数化/credit 仍需约束。

因此下一步不是继续放大旧残差，而是在 H3 基线和 current07–10 分布上重新训练，
让 actor 看到“强控制器之后还剩什么误差”。

## 2. 假设

### H159-A：在 H3/current07–10 上重训 1M SAC 能产生独立 success 增益

成立应看到：current10 或 current15 的 pooled policy−H3 zero `>= +4/120`，
至少 `2/3` seeds 非负；final vertical/horizontal 至少一项改善；timeout/safety 不增。

失败应看到：policy 与 zero 相同或更差，或 residual 活跃但闭环指标不动。

最便宜证伪：3 seeds × 1M SAC，H3 + current `[0.275,0.55]` 训练，near10_2 eval，
H3 zero 配对，dev40。

### H159-B：强基线会阻止 Goal150 的横向破坏

成立应看到：policy 的 heading/horizontal 指标不比 zero 明显恶化，failure mode
不增加 MIXED/HORIZONTAL；residual 通道没有出现稳定大 heading 偏置。

失败应看到：heading/horizontal 明显恶化或 rudder saturation 上升，success 下降。
此时转 channel-restricted residual（限制 rudder/heading）或 terminal credit。

### H159-C：1M 不足，不是算法问题

成立应看到：1M policy 的 residual 活跃、方向与误差有正相关，但 success 不移动；
5M 或更长训练可能有效。

失败应看到：1M 已经出现方向性增益；此时不需要等 5M，先做 3-seed/多环境确认。

## 3. 数据与隔离

| 项目 | 值 |
|---|---|
| 算法 | SAC，`pc_z_t` |
| 训练 seeds | `86201–86203` |
| 预算 | 每 seed `1,000,192` transitions（screen） |
| 训练环境 | H3 controller，current `[0.275,0.55] m/s`，shear scale `1.0`，goal radius `2.0 m` |
| eval gate | near10_2，`learner_terminal_gate_mode=none` |
| eval 环境 | H3，strict125 `1.25 m` |
| eval levels | nominal / current07 / current10 / current15 |
| dev seeds | `20–59` |
| held-out | `100–149` 关闭 |
| 评估 | CUDA/float64，trace + reward breakdown |

禁止：用 eval success 选 checkpoint；改 reward/terminal/success/radius；扩 seed 救 Gate；
把 1M screen 写成 formal；用 held-out 调参。

## 4. 方法

- `env_h3_current07_10_train.json`：Goal85 位置契约 + H3 controller + current
  `[0.275,0.55]`，goal radius `2.0`；**curriculum disabled**，与 Goal86 固定环境
  训练协议一致，避免 C0/C1 direct-command 语义污染；
- 训练配置复用 Goal86 T2-SAC 超参数：`num_envs=256`、`batch_size=256`、
  `UTD=0.125`、`learning_starts=20k`、`buffer_size=500k`；
- 训练时 residual authority full，不加 terminal gate；
- 评估时 near10_2 gate（`10 m -> 2 m`），与 H3 zero 同场景配对；
- 记录 success、plane miss、timeout、safety、path vertical/horizontal、
  residual RMS bins、heading/pitch error、crab、saturation、failure mode。

## 5. 执行计划

- [x] Step 0：Goal157/158 证据记录；
- [x] Step 1：写本预注册；
- [x] Step 2：生成旧训练配置并做 smoke；事后审计发现 `pc_stage=C0` 且 direct command 实际启用，判 `INVALID`；
- [x] Step 3a：旧 3 seeds × 1M 训练完成，但训练语义无效，结果作废；
- [x] Step 3b：Goal159c 修正版 `C3/ALOS + direct_commands_active=false`，3 seeds × 1M 完成；
- [x] Step 4：Goal159c 完成后，H3 zero vs policy，4 levels × 40 episodes，配对分析；
- [x] Step 5：按 Gate 裁决：`GATE_FAIL`，进入 blocker，不启动 5M formal；
- [x] Step 6：写 result、blocker、reading note、LEDGER、STATE，跑 checker。

## 6.1 Goal159c 裁决摘要

`current07 +1/120`、`current10 -3/120`、`current15 -2/120`，未达到
`current10/15 >= +4/120` 的 Gate。residual 活跃且动作饱和约 `56%`，但 scale
降低只消除饱和、不恢复 success；channel 限制只有 current07 stern-only `+5`，
current10/15 不复现；mid_only 关闭 terminal residual 反而使 current10 恶化到
`-9`。因此判 `GATE_FAIL_RESIDUAL_ACTIVE_NO_CLOSED_LOOP_GAIN`，下一步转
terminal capture / command-space 动作参数化，不换算法、不扩 seed。

### 5.1 协议完整性记录

第一次启动（2026-11-12 01:16）误继承了 `goal148/env_nominal.json` 的 C0/C1
curriculum 与 direct-command 语义。该批只运行了约 1 分钟，未产生 final.zip，
已全部停止并删除 `train/seed*`，判定为 `INVALID_START`，不进入结果。

第二次启动使用 `curriculum disabled` 的
`configs/lowlevel/goal159/env_h3_current07_10_train.json`，但该配置仍是
`pc_stage=C0` 且 `curriculum_direct_commands_enabled=true`；按
`BatchedPCLowLevelEnv` 的实际控制语义，direct command 仍然启用。因此这一批
也判 `INVALID_TRAIN_DIRECT_COMMAND`，不能进入正式结果。

修正版 `Goal159c` 使用 `pc_stage=C3`、`hold_bonus=false`、无 curriculum，
并在 manifest 中新增 `direct_commands_active`；smoke 已确认该字段为 `false`。
只有 Goal159c 可进入正式配对评估。

## 6. Gate

| check | 阈值 |
|---|---:|
| 主 success | current10 或 current15 的 `Δsuccess >= +4/120` |
| seed 一致性 | 至少 `2/3` seed `Δsuccess >= 0` |
| 横向 guard | horizontal/heading 不比 zero 明显恶化 |
| 纵向 guard | final vertical 不比 zero 差 `+0.05 m` |
| residual 活性 | 2–10 m RMS 可见，>10 m 接近 0 |
| 风险 | timeout/safety 不增加；0 NaN/飞点 |

失败裁决：

| 观察 | 判读 | 下一步 |
|---|---|---|
| 无 success 移动 | 残差/gate 不是瓶颈 | terminal geometry / guidance |
| residual 有方向但幅度不降 | credit/探索不足 | 5M 或 reward pilot |
| 横向恶化 | 动作参数化错 | channel-restricted residual |
| 1M 正增益 | 方向正确 | 5M formal + 多环境 |

## 7. Git

禁止 `git add -A`。只 add 本 Goal 相关文件，提交前运行 `git diff --cached --check`。

## 8. 自主迭代契约

执行 `research/AUTORESEARCH_LOOP.md`；失败先归因，不换算法、不扫 seed；每个 blocker
至少两个可证伪 hypothesis；先跑最便宜 falsification，再决定 `continue` / `pivot` / `stop`；
不自动 commit/push，不打开 held-out，不改成功定义。
