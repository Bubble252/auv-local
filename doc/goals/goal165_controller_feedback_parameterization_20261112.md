# Goal165：修复控制器反馈参数化，恢复 2.5 m/s 闭环包线

**日期**：2026-11-12<br>
**状态**：`GATE165_C_FAIL_NO_TERMINAL_TRAINING / SPEED_ENVELOPE_PASS / HELDOUT_CLOSED / NO_FORMAL`<br>
**上游**：Goal164 `GATE164_B_FAIL / ACTION_FEEDBACK_BUG_FOUND`
**结果**：`doc/goals/goal165_result_20261112.md`

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
  check_command: .venv/bin/python ../scripts/check_research_ledger.py --goal Goal165 --goal-doc doc/goals/goal165_controller_feedback_parameterization_20261112.md --strict
```

`required: true`：Goal165 是控制器反馈修复型 goal；每次修改都必须有合成测试、
legacy 回归和 ledger 记录。Gate165-B 通过前不启动 RL 训练。

## 1. 背景与目标

Goal164 证明速度契约正确，但 corrected zero replay 的 `eta` 峰值仍为
`2.558/2.623 m/s`，全部超速在 terminal。审计发现 actor feedback 分支把
`actor_state[:, 6:12]` 直接当成 `nu`，而实际布局是
`[position, attitude, angular_rate, linear_velocity]`。

Goal165 只做最小修复：

- `nu_ground` 下从 actor state 重建正确的 `nu=[u,v,w,p,q,r]`；
- 速度环使用 DVL water-relative surge，而不是 ground surge；
- legacy 配置保留旧分支，保证历史 trace 仍可 bit-compatible；
- 不改 PPO/SAC/REDQ/DroQ，不改 reward/terminal/success/radius。

## 2. 冻结约束

- REMUS100 最大地速硬上限 `2.5 m/s`；
- 课程 v2：C0 `1.00–1.20`、C1 `1.10–1.40`、C2 `1.20–1.60`、C3 `1.30–1.80`、
  C4 `1.40–2.00`、C5 `1.50–2.25`、C6 `0.90–2.50`；
- 每个 episode 必须满足 `speed+current+variation<=2.5`；
- held-out `100–149` 保持关闭；
- 不覆盖 Goal164 的 trace/result 目录。

## 3. 可证伪假设

### H165-A：actor-state 排列错误是 terminal 速度环失真的上游原因

- 成立：修正后 corrected zero replay 的 `eta` 峰值明显下降，且非 terminal
  超速率仍为 0；速度积分/RPM 不再长期偏置。
- 失败：峰值和超速比例在 seed/level 内没有移动。
- 最便宜证伪：合成测试 + current10/current15 各一次 40-episode zero replay。

### H165-B：修正反馈后仍有 terminal 残差，需要单独的 anti-windup/减速策略

- 成立：修正后仍有 episode `eta>2.5`，且 integral cap 比例仍高。
- 失败：所有 episode 满足 `eta<=2.5+1e-9`。
- 最便宜证伪：同一 replay 的 terminal phase、integral、RPM 分箱。

## 4. 方法与隔离

- `BatchedControlContext` 增加可选 `water_surge_mps`；
- `nu_ground` 下控制器使用 `water_surge_mps`，并从 actor state 正确重排 `nu`；
- `legacy_water_nu` 下走旧路径，不改变旧配置输出；
- 只读取 dev seeds 20–59；zero replay 不再复制相同 seed 三次；
- 外部论文/代码：若 Gate165-B 仍失败，再检索 anti-windup/command governor，
  并写 `research/paper_reading/` note。

## 5. 执行计划

- [x] Step 0：写 Goal165、Research Contract 和冻结约束；
- [x] Step 1：加合成测试复现 `actor_state` 排列错误；
- [x] Step 2：实现 `nu_ground` 最小修复与 legacy 分支；
- [x] Step 3：legacy bit-compatible smoke；
- [x] Step 4：corrected zero replay（nominal/current07/current10/current15）；
- [x] Step 5：Gate165-B 裁决；PASS；
- [x] Step 6：重训 Goal162 gain residual（3 seeds × 1M，远程完成）；
- [x] Step 7：result、ledger、STATE、checker；不自动 commit/push。

## 6. Gate

### Gate165-A：动作反馈契约

- 合成测试证明 `controller_nu[:3]` 来自 actor state 的线速度，`controller_nu[3:]`
  来自角速度；
- `nu_ground` 下 speed error 使用 DVL water surge；
- legacy 分支的合成输出与旧实现一致；
- 相关单测通过。

### Gate165-B：corrected zero-only 速度包线

- 数据：dev seeds 20–59，nominal/current07/current10/current15，zero policy；
- 只跑 4 个独立 replay，不把同一 seed block 复制成“三个 seed”；
- 通过：所有 episode `eta<=2.5+1e-9`，非 terminal 超速率为 0；
- 失败：保留真实数值，进入 anti-windup/终端减速的新预注册 goal；不放宽 2.5。

### Gate165-C：gain residual 重训

只有 Gate165-B 通过后才启动：3 seeds × 1M，SAC，z_t，Goal85 位置契约，
同 reward/terminal/success/radius；primary 仍为 current10 default `>=+4/120`，
strict125 `>=+2/120`。

**结果：FAIL。** speed envelope PASS，但 policy 与 zero 在 8 个格子逐 seed
完全同分，Δ 全为 `0/120`。根因是 1M/1024 envs 只有每环境约 976 步，
`completed_episodes=0`，训练从未覆盖 terminal；同时 gain residual 的
requested action 改变只有 `[0.011, 0.022, 0.168]`，闭环位置改变毫米级。
详见 `doc/goals/goal165_result_20261112.md`。

### Gate165-C 后续

不换算法、不扩 seed、不放宽 2.5。预注册 Goal166：只把 `num_envs` 从 1024
改为 256，保持 1M 预算、SAC、z_t、gain residual、reward/terminal/success/radius
不变，使每环境约 3906 步，先证明 terminal 覆盖，再判断 residual 是否还有
authority。

## 7. 运行与验证

```bash
cd settled_局部规划器_反应式
.venv/bin/python -m pytest -q tests/test_pc_goal164_velocity_contract.py \
  tests/test_pc_goal163_speed_semantics.py

ssh -p 43127 yhx@152.136.183.51
cd /data/yhx/underwater-robotics/src/plan_p1/goal153_reward_curriculum_20261111
GPUS="0 1 2 3" PARALLEL=4 bash scripts/run_goal165_replay.sh

.venv/bin/python ../scripts/check_research_ledger.py \
  --goal Goal165 \
  --goal-doc doc/goals/goal165_controller_feedback_parameterization_20261112.md \
  --strict
```

## 8. Git 与算力

- 8×4090 可用；Gate165-B 4 个 replay 并行占 4 张卡；
- Gate165-C 3 个 SAC seed 并行占 3 张卡；
- 禁止 `git add -A`；只 add 本 goal 相关文件；
- 提交前运行 `git diff --cached --check`；不自动 commit/push。

## 9. 自主迭代契约

执行 `research/AUTORESEARCH_LOOP.md`：失败先归因；每次 blocker 至少两个可证伪
hypothesis；需要外部方法时先检索论文/代码并写 reading note；先跑最便宜的
falsification；`pivot` 时新建 goal 文档；不换主算法族，不打开 held-out。

