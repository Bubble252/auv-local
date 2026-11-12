# Goal164：统一 `nu`/海流速度契约并恢复 2.5 m/s 闭环包线

**日期**：2026-11-12<br>
**状态**：`GATE164_B_FAIL / ACTION_FEEDBACK_BUG_FOUND / PIVOT_TO_GOAL165 / HELDOUT_CLOSED`<br>
**上游**：Goal163 `AUDIT_COMPLETE`、Goal162 `INVALID_SPEED_CONTRACT`

**结果**：`doc/goals/goal164_result_20261112.md`

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
  check_command: .venv/bin/python ../scripts/check_research_ledger.py --goal Goal164 --goal-doc doc/goals/goal164_velocity_contract_repair_20261112.md --strict
```

`required: true`：Goal164 是修复型 goal；每次契约修改都必须有合成测试、旧结果
回归和 ledger 记录。Gate164-B 通过前不启动 RL 训练。

## 1. 背景与目标

Goal163 已证明：`eta_ned` 前向差分与 `nu` 的 MAE 为 `0.000`，本项目的
`nu` 是地速；但 `ground_velocity_body()` 和 `sensors.py` 仍按水速处理，
重复加海流，导致 actor/critic/reward/hold/evaluation 同时偏置。旧评价超限
`7.2%/14.7%`，真实 `eta` 超限只有 `0.31%/0.51%`；真实峰值仍为
`2.546/2.598 m/s`，且全部在 terminal phase。

Goal164 的目标不是让数字好看，而是建立唯一、可验证、可版本化的速度契约，
再在这个契约下判断终端速度环和 RL 方法。

## 2. 冻结的速度契约

```text
nu              = ground velocity
current_body    = R(eta)^T current_ned
water_velocity  = nu - current_body
ground_velocity = nu
```

强约束：

- `eta` 积分只使用 ground velocity；
- 水动力只使用 `nu - current_body`；
- 控制器/ALOS 的 `command_speed_mps` 继续表示 water-relative speed；
- actor/reward 必须显式选择 ground 或 water-relative，不能混用；
- 评估 trace 同时记录 ground 和 water velocity，字段名必须与语义一致；
- REMUS100 地速硬上限保持 `2.5 m/s`，课程 C6 保持 `0.90–2.50`，并继续满足
  `speed + current + variation <= 2.5`。

### 课程学习速度上限（硬约束）

用户已确认 REMUS100 的**最大地速就是 `2.5 m/s`**，不是 2.5 的“均值”或
“软目标”。所有新课程必须显式通过 `high_speed_curriculum_stages()` opt in，
任何阶段都不得把 `speed_range_mps` 上限写到 2.5 以上：

| 阶段 | v2 speed range (m/s) | 课程用途 |
|---|---:|---|
| C0 | `1.00–1.20` | 标称速度与基础跟踪 |
| C1 | `1.10–1.40` | 弱流与轻度噪声 |
| C2 | `1.20–1.60` | 中速与增强流 |
| C3 | `1.30–1.80` | 高速过渡与估计器退化 |
| C4 | `1.40–2.00` | 强流、剪切与长航程 |
| C5 | `1.50–2.25` | 接近包线上沿 |
| C6 | `0.90–2.50` | 完整 standing-eval 速度轴；2.5 为硬端点 |

采样器必须在每个 episode 上同时满足：

```text
command_water_speed + current_speed + current_variation <= 2.5 m/s
```

因此 2.5 端点只能与零/可忽略海流配对；一旦保留海流或速度波动，采样器会先
收缩 current 预算。该约束同时适用于 PPO、SAC、BC+residual、传统 ALOS 和
zero-residual 评价，不能只在训练侧生效。

## 3. 版本化与兼容

新增可选配置 `plant.velocity_contract`：

| 值 | 语义 | 用途 |
|---|---|---|
| `legacy_water_nu` | 旧实现：`nu` 当水速，ground=`nu+current` | 仅用于 bit-compatible replay |
| `nu_ground` | 新契约：`nu` 是地速，water=`nu-current` | 所有新训练/评估 |

- 默认值暂时设为 `legacy_water_nu`，避免历史配置静默改变；
- Goal164 新建的 config 显式写 `"velocity_contract": "nu_ground"`；
- manifest 记录该字段和 config hash；
- 旧 trace/result 目录不覆盖，corrected replay 写到新目录；
- Gate164-A/B 通过后再讨论是否把默认值切到 `nu_ground`。

## 4. 可证伪假设

### H164-A：统一契约后，旧评价的大范围超限消失，只剩 terminal 残留

- 成立：corrected trace 的 `ground=nu`、`water=nu-current`；非 terminal 超速率
  为 0；峰值从 `2.88/3.31` 降到 `2.55/2.60` 附近。
- 失败：修正后仍有大范围非 terminal 超速，说明还有第二个包线/plant 问题。

### H164-B：terminal 残留超速由 speed integral/RPM 回收滞后造成

- 成立：真实超速步 integral 接近 `400 rpm` cap，集中在减速段；条件积分或
  减速前馈能把峰值压回 `2.5` 以内。
- 失败：integral 已回退而地速仍超限，或超速与 terminal 减速无关。

### H164-C：修正速度语义后，gain residual 的相对效果会改变

- 成立：同 SAC/z_t/reward/terminal/seed 协议下，Goal162 gain residual 结果
  不能复现旧方向，或 current10 default 的效应显著移动。
- 失败：修正后效应与旧结果在 seed 噪声内一致，说明旧结论对速度偏置不敏感。

## 5. 执行计划

- [x] Step 0：写 Goal164、Research Contract、速度契约和兼容策略；
- [x] Step 1：新增 `plant.velocity_contract` 解析与 manifest 字段；
- [x] Step 2：实现 ground/water helper 的 `nu_ground` 分支，保留 legacy helper；
- [x] Step 3：修 `sensors.py` 的 DVL bottom/water 生成；
- [x] Step 4：修 PC env 的 actor/critic/speed error/hold/reward/route speed；
- [x] Step 5：修 evaluation trace，增加 water velocity 字段；更新审计脚本；
- [x] Step 6：合成测试 + legacy bit-compatible 回归；
- [x] Step 7：corrected zero-only replay（current10/current15 × 3 seeds）；
- [x] Step 8：Gate164-A/B 裁决；真实 terminal 超速仍在，已预注册 Goal165
  修复动作反馈参数化，再决定 anti-windup；
- [ ] Step 9：速度契约恢复后，重训 Goal162 gain residual（3 seeds × 1M，
  同 SAC/z_t/reward/terminal/success/radius）；由 Goal165 接管；
- [x] Step 10：checker、STATE、结果文档；不自动 commit/push。

## 6. Gate 与验收

### Gate164-A：契约与兼容

- `eta` 前向差分 = `nu`；
- `ground_velocity_body` = `nu`；
- `water_velocity_body` = `nu - current`；
- DVL bottom = ground，DVL water = water-relative；
- 旧 `legacy_water_nu` config 在固定 seed 上的 trace 与旧实现一致（允许新增字段）；
- 新 `nu_ground` config 的 actor/reward/eval 字段全部与契约一致；
- 相关单测通过，manifest 记录 `velocity_contract`。

### Gate164-B：corrected zero-only 速度包线

- 数据：Goal162 dev40 current10/current15 × 3 seeds，zero policy，不训练；
- 主指标：`eta` 峰值、`eta` 超 2.5 比例、非 terminal 超速比例；
- 通过条件：所有 episode `eta <= 2.5 + 1e-9`，且非 terminal 超速率为 0；
- 若失败：保留真实数值，进入 Step 8 的最小 anti-windup/terminal deceleration；
  不放宽 2.5，不换算法，不扩 seed。

### Gate164-C：gain residual 重训

只有 Gate164-B 通过后才启动：

- 3 seeds × 1M transitions，SAC，z_t，同一 reward/terminal/success/radius，
  H3 zero baseline；
- primary：current10 default `>= +4/120`，strict125 `>= +2/120`；
- guards：current07/nominal/current15 不显著退化；
- 速度契约：所有 policy evaluation 满足 `eta <= 2.5`；
- 不打开 held-out `100–149`，不扩 seed 救 Gate。

## 7. 运行与验证

本地单测：

```bash
cd settled_局部规划器_反应式
.venv/bin/python -m pytest -q \
  tests/test_pc_goal163_speed_semantics.py \
  tests/test_lowlevel_pc_env.py \
  tests/test_lowlevel_curriculum_sampler.py
```

远端 corrected replay：

```bash
ssh -p 43127 yhx@152.136.183.51
cd /data/yhx/underwater-robotics/src/plan_p1/goal153_reward_curriculum_20261111
/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python \
  scripts/audit_pc_speed_envelope.py \
  --trace-root artifacts/lowlevel/goal164_velocity_contract_20261112/replay_zero \
  --output artifacts/lowlevel/goal164_velocity_contract_20261112/replay_analysis
```

检查器：

```bash
.venv/bin/python ../scripts/check_research_ledger.py \
  --goal Goal164 \
  --goal-doc doc/goals/goal164_velocity_contract_repair_20261112.md \
  --strict
```

## 8. 算力与 Git

- 8×4090 可用；Step 7 的 6 个 zero replay 可并行；
- Step 9 的 3 个 SAC seed 各占 1 张 4090，同时启动；
- 每个训练继续使用 `num_envs=1024`、`batch_size=512`、
  `update_to_data_ratio=0.125`、`total_env_transitions=1_000_000`；
- 只 add 本 goal 相关文件，禁止 `git add -A`；
- 提交前运行 `git diff --cached --check`；
- 不提交 `artifacts/`、checkpoint、日志、缓存；不自动 commit/push。

