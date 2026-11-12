# Goal166：修复 1M 训练的 terminal 覆盖，再检验 gain residual

**日期**：2026-11-12<br>
**状态**：`GATE166_A_PASS / GATE166_B_FAIL_ACTION_AUTHORITY_NULL / HELDOUT_CLOSED / NO_FORMAL`<br>
**上游**：Goal165 `GATE165_C_FAIL_NO_TERMINAL_TRAINING`
**结果**：`doc/goals/goal166_result_20261112.md`

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
  check_command: .venv/bin/python ../scripts/check_research_ledger.py --goal Goal166 --goal-doc doc/goals/goal166_terminal_covering_horizon_repair_20261112.md --strict
```

`required: true`：Goal166 是训练协议修复型 goal；必须记录 terminal 覆盖、
paired zero/policy、速度包线和 action authority。训练前不改 reward、terminal、
success、radius、算法族或 held-out。

## 1. 背景与目标

Goal165 已证明 `nu_ground` 控制器反馈修复有效：所有评估 `eta<=2.5 m/s`，
超速为 0。但 3 seeds × 1M 的 gain residual 与 zero 在 8 个格子逐 seed 同分，
Δ 全为 `0/120`。

根因不是 SAC 或 reward，而是训练 horizon：

```text
1,000,448 transitions / 1024 envs = 976 steps per env
corrected zero episode min / median / p90 = 1291 / 1749 / 1996 steps
training manifest completed_episodes = 0, success_episodes = 0
```

训练只覆盖了 episode 的前约 19.5 秒，从未看到 goal 或 goal-plane-miss
terminal。Goal166 只改一个变量：把 `num_envs` 从 `1024` 改为 `256`，
保持 1M 预算，使每环境约 `3906` 步，覆盖 `max_episode_steps=2500`
和实际 zero episode 的 p90/max。

## 2. 冻结约束

- 算法：SAC，`z_t`，H3/C3/ALOS，controller-gain residual；
- 训练预算：3 seeds × 1M，seeds `86201–86203`；
- 唯一训练改动：`training.num_envs: 1024 -> 256`；
- reward、terminal、success、goal radius、observation、action interface 不变；
- REMUS100 地速硬上限 `2.5 m/s`；本 goal 不激活速度课程 v2，训练固定
  `1.2 m/s`、current `0.275–0.55 m/s`、variation `0.1 m/s`，最坏
  `1.2+0.55+0.1=1.85 m/s`；
- 课程 v2 仍只允许通过 `high_speed_curriculum_stages()` 显式启用；
  C6 `0.90–2.50`，每个 episode 满足 `speed+current+variation<=2.5`；
- dev seeds `20–59`；held-out `100–149` 保持关闭；
- 不扩 seed、不换算法、不改 Gate 阈值、不覆盖 Goal165 产物。

## 3. 可证伪假设

### H166-A：horizon 是 terminal 缺失的直接原因

- 成立：3 个 manifest 都出现 `completed_episodes>=256` 且
  `success_episodes>=128`；训练侧 terminal 分布非空。
- 失败：改 `num_envs=256` 后 completed episodes 仍为 0，或远低于 256。
- 最便宜证伪：训练结束后只读 manifest，不跑评估。

### H166-B：补齐 terminal 后 gain residual 能产生可解释收益

- 成立：current10 default `>=+4/120`，strict125 `>=+2/120`；guards 不显著退化；
  signed lateral/vertical 至少一个通道跨 seed 稳定；near-goal requested action
  或 eta 改变明显大于 Goal165。
- 失败：success Δ 仍为 0，或 residual 仍只产生毫米级轨迹改变。
- 最便宜证伪：3 seeds × 1M + paired dev40 评估。

### H166-C：若 horizon 修复后仍无 authority，则瓶颈是动作参数化

- 成立：residual norm 仍大（near goal `>=0.8`），但 requested action 平均差
  仍接近 Goal165 的 `[0.011, 0.022, 0.168]`，或 eta 平均差仍 `<0.01 m`。
- 失败：horizon 修复后 residual 开始产生厘米级以上轨迹改变。
- 结论：进入新的动作参数化 goal，不切换到 PPO/REDQ/DroQ。

## 4. 方法与隔离

- 复制 Goal165 的训练/评估配置到 `configs/lowlevel/goal166/`；
- 只改 `training.num_envs=256`，其余字段逐项保持；
- 复用 Goal165 corrected zero 的 8 个评估格，不重复跑 zero；
- 训练在 3 张 GPU 并行，评估在 3–8 张 GPU 并行；
- 训练后先做只读 manifest 审计，再决定是否启动 policy 评估。

## 5. 执行计划

- [x] Step 0：写 Goal166、Research Contract、冻结约束和 Gate；
- [x] Step 1：生成 Goal166 配置，校验只有 `num_envs` 改变；
- [x] Step 2：3 seeds × 1M 训练，检查 manifest 的 terminal 覆盖；
- [x] Step 3：policy-only paired dev40 评估，复用 Goal165 zero；
- [x] Step 4：success/速度/action-authority/terminal 分布分析；
- [x] Step 5：result、ledger、STATE、checker；不自动 commit/push。

## 6. Gate

### Gate166-A：训练 terminal 覆盖

- 3 个 seed 的 `completed_episodes>=256`；
- 3 个 seed 的 `success_episodes>=128`；
- 无 NaN/Inf；`speed_contract.ok=true`。

**结果：PASS。** completed episodes `591/582/595`，success episodes
`525/516/523`，每环境 `3907` 步。

### Gate166-B：闭环收益

- current10 default policy-zero `>=+4/120`；
- current10 strict125 policy-zero `>=+2/120`；
- nominal/current07/current15 不显著退化；
- 所有 policy/zero 评估 `eta<=2.5 m/s`；
- timeout/safety 不高于 zero；
- path/vertical/entry/near/final bins 不被系统性破坏。

**结果：FAIL。** 8 个格子逐 seed 全部 `Δ=0/120`；速度包线 PASS，
timeout/safety 为 0，path/entry/final 几乎逐位相同。

### Gate166-C：机制与 authority

- signed lateral/vertical 至少一个通道跨 seed 稳定相关；
- near-goal requested action 或 eta 改变明显大于 Goal165；
- residual 不是纯距离 schedule；
- 若 Gate166-B 失败但 Gate166-A 通过，按 H166-C 进入动作参数化 goal。

**结果：H166-C 成立。** near-goal residual norm `1.28`、policy saturation
`0.85–0.98`，但 requested action 平均差仍只有 `[0.012, 0.029, 0.173]`，
eta 平均差 `4.5e-3 m`。瓶颈是动作参数化，不是 SAC 或 terminal 覆盖。

## 7. 运行与验证

```bash
cd settled_局部规划器_反应式
.venv/bin/python -m pytest -q tests/test_lowlevel_curriculum_sampler.py \
  tests/test_pc_goal164_velocity_contract.py \
  tests/test_pc_goal165_controller_feedback.py

ssh -p 43127 yhx@152.136.183.51
cd /data/yhx/underwater-robotics/src/plan_p1/goal153_reward_curriculum_20261111
GPUS="0 1 2" PARALLEL=3 bash scripts/run_goal166_train.sh
SEEDS="86201 86202 86203" MODES="policy" GPUS="0 3 4" PARALLEL=3 bash scripts/run_goal166_eval.sh

.venv/bin/python ../scripts/check_research_ledger.py \
  --goal Goal166 \
  --goal-doc doc/goals/goal166_terminal_covering_horizon_repair_20261112.md \
  --strict --require-result
```

## 8. 算力与 Git

- 8×4090 可用；训练 3 张卡，评估最多 8 张卡；
- 不因为“用满 GPU”而扩 seed 或改预算；
- 禁止 `git add -A`；只 add 本 goal 相关文件；
- 提交前运行 `git diff --cached --check`；不自动 commit/push。

## 9. 自主迭代契约

执行 `research/AUTORESEARCH_LOOP.md`：先归因，再决定是否改动作参数化、
reward 或样本效率；每次 blocker 至少两个可证伪 hypothesis；先跑最便宜的
falsification；不换主算法族，不打开 held-out，不放宽 2.5 m/s。
