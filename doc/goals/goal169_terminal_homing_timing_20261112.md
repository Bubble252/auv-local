# Goal169：deterministic terminal homing timing / blending 对照

**日期**：2026-11-12<br>
**状态**：`SCREEN_PASS / TERMINAL_HOMING_50M_SELECTED / NO_RL_TRAINING / NO_CONTRACT_CHANGE / HELDOUT_CLOSED`<br>
**上游**：Goal168 `AUDIT_COMPLETE / TERMINAL_HEADING_LAG_DOMINANT`

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
  check_command: .venv/bin/python ../scripts/check_research_ledger.py --goal Goal169 --goal-doc doc/goals/goal169_terminal_homing_timing_20261112.md --strict
```

`required: true`：Goal168 已证明 terminal heading lag 是主导机制；本 goal 只做
zero controller 的 deterministic timing screen，不训练 residual、不改算法族。

## 1. 背景与证据

Goal168 审计 18 个 `GOAL_PLANE_MISS` episode：

- terminal phase 在 20–15 m 的 duty 为 `0.964–0.996`；
- 20 m 入口 authority margin 为 `2.17–33.76`；
- 命令需要转 `0.45–1.47 rad`，实测 yaw 仅 `0.057–0.132 rad/s`；
- 15/18 的 `rotate_time_ratio` 为 `0.87–1.88`；
- 3/18 为 seed 26 的 `path_progress` 5.35 m 跳变。

因此当前 `20 m` 才从路径切线转向 goal bearing，转向时间与剩余时间同量级，
横向误差在最后 10 m 继续长大。最便宜的可证伪问题是：**更早开始 terminal
homing 是否能在同一 zero controller 上减少 goal-plane miss？**

## 2. 可证伪假设

### H169-A：提前 terminal homing 能修复横向 capture

- 成立：30/40/50 m 中至少一个变体在 current10 default 达到 `+4/120`，
  strict125 达到 `+2/120`，且 guards、path、vertical、near bins、速度不退化。
- 失败：所有变体的 success 仍 `Δ=0` 或 guards 明显退化。
- 最便宜 falsification：zero controller deterministic replay 4 个距离 ×
  4 levels × 2 radii × dev40，无训练。

### H169-B：提前 homing 会破坏路径跟踪，需要平滑 blend

- 成立：30/40/50 m 的 success 有改善，但 path3d/vertical/near bins 退化超过
  预注册 guard；40→20 m 平滑 blend 更好。
- 失败：提前 homing 既不改善 success，也不产生 path 代价。

### H169-C：switch distance 不是主因，yaw authority/path family 才是

- 成立：所有 timing 变体都无 success 改善，且失败 episode 仍在 20 m 入口
  有足够 authority、但实际 yaw rate 不足。
- 失败：至少一个 timing 变体通过 Gate169-B。

## 3. 数据与隔离

- zero controller：复用 Goal165 corrected `nu_ground` H3/C3/ALOS zero；
- 变体：`terminal_alignment_min_m = terminal_alignment_max_m = 20/30/40/50 m`；
- 其余 guidance/plant/reward/terminal/success/radius 完全一致；
- dev seeds `20–59`；held-out `100–149` 关闭；
- 不训练、不使用 policy checkpoint、不打开 Goal167 policy；
- 速度硬上限 `2.5 m/s`；课程 v2 不激活。

## 4. 方法

配置：

```text
configs/lowlevel/goal169/env_terminal_alignment_20_eval_nu_ground.json
configs/lowlevel/goal169/env_terminal_alignment_30_eval_nu_ground.json
configs/lowlevel/goal169/env_terminal_alignment_40_eval_nu_ground.json
configs/lowlevel/goal169/env_terminal_alignment_50_eval_nu_ground.json
```

唯一差异是 `guidance.terminal_alignment_min_m/max_m`。评估使用
`scripts/evaluate_pc_sac.py --action-mode zero-residual`，无 checkpoint。

## 5. 执行计划

- [x] Step 0：写 Goal169、Research Contract、冻结约束和 Gate；
- [x] Step 1：生成 4 个配置，校验只有 terminal alignment 距离不同；
- [x] Step 2：4 变体 × 4 levels × 2 radii × dev40 zero replay；
- [x] Step 3：success/path/vertical/near bins/速度审计；
- [x] Step 4：result、ledger、STATE、checker；
- [x] Step 5：根据 Gate169-B 决定 continue/pivot；不自动 commit/push。

## 6. Gate

### Gate169-A：执行完整性

- 32/32 zero runs 有 `summary.json`；
- 无 NaN/Inf；所有 `eta<=2.5 m/s`；
- held-out 未打开；无训练启动。

### Gate169-B：闭环收益

相对 `alignment20` baseline：

- current10 default `>=+4/120`；
- current10 strict125 `>=+2/120`；
- nominal/current07/current15 不退化超过 `2/120`；
- path3d RMSE 不退化超过 `0.05 m`；
- final vertical 不退化超过 `0.05 m`；
- near bins 不被系统性破坏；
- timeout/safety 不高于 baseline。

### Gate169-C：机制

- 记录每个变体的 `GOAL_PLANE_MISS` 数量、final-distance bins、
  entry20 rotate-time ratio；
- 若 40/50 m 改善但 path 退化，优先 40→20 m 平滑 blend；
- 若全部无改善，判 `YAW_AUTHORITY_OR_PATH_FAMILY`，不继续扫距离。

## 7. 运行与验证

```bash
cd settled_局部规划器_反应式
.venv/bin/python -m pytest -q tests/test_lowlevel_curriculum_sampler.py

ssh -p 43127 yhx@152.136.183.51
cd /data/yhx/underwater-robotics/src/plan_p1/goal153_reward_curriculum_20261111
bash scripts/run_goal169_homing_screen.sh
bash scripts/run_goal169_analysis.sh

.venv/bin/python ../scripts/check_research_ledger.py \
  --goal Goal169 \
  --goal-doc doc/goals/goal169_terminal_homing_timing_20261112.md \
  --strict --require-result
```

## 8. 算力与 Git

- 32 个 zero evaluation，CPU/GPU 均可，8 卡并行约 1–2 分钟；
- 不训练、不扩 seed、不改预算；
- 禁止 `git add -A`；只 add 本 goal 相关文件；
- 提交前运行 `git diff --cached --check`；不自动 commit/push。

## 9. 自主迭代契约

执行 `research/AUTORESEARCH_LOOP.md`：先归因，再改最小变量；每次 Gate 失败写
blocker/decision；不换主算法族、不打开 held-out、不改 success/radius/reward/
terminal、不放宽 2.5 m/s。

## 10. Goal 提示词

```text
你现在执行 Goal169：zero controller 下测试 terminal homing 切换距离 20/30/40/50 m。

先读：
- doc/goals/goal169_terminal_homing_timing_20261112.md
- doc/goals/goal168_result_20261112.md
- research/decisions/2026-11-12-goal168-terminal-heading-lag-open-goal169.md

硬约束：
1. 只跑 zero-residual，不训练、不加载 policy checkpoint；
2. 不改 success/radius/reward/terminal，不放宽 2.5 m/s；
3. 不打开 held-out 100–149；
4. 唯一变量是 terminal alignment 20/30/40/50 m；
5. 失败后写 blocker/decision/ledger，再决定 continue/pivot/stop。

执行顺序：
1. 生成 4 个 env config 并校验只差 alignment 字段；
2. 32 个 zero replay；
3. 分析 success/path/vertical/near bins/speed；
4. Gate169-B 裁决；
5. 写 result 和下一步预注册，不自动 commit/push。
```

