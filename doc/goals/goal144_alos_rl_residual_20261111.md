# Goal144：纯 ALOS 基座上的 PPO/SAC 残差与 terminal capture 对照

**日期**：2026-11-11<br>
**状态**：`PREREGISTERED_P1 / CURRICULUM_GATE_AND_PER_STAGE_BUDGET_REPAIRED / GATEFIX2_C0_PROMOTED_C1_ROLLBACK / PPO_1M_DONE / SAC_600K_EARLY_STOPPED / COMMAND_RESIDUAL_HARM / HELDOUT_CLOSED`<br>
**上游**：`goal143_result_20261111.md`；Goal143 C0 `0.85`、zero-residual frozen-BC `0.85`、pure ALOS `0.925`。<br>
**优先级**：`P1`（Goal136/Goal143 C-line 的同一控制主线）；Goal140 仍为 `P2`，不得插队。<br>
**性质**：只把残差基座从 frozen-BC 换成纯 ALOS，保留 C0/C1 课程、reward、terminal、success、goal radius、dev/held-out 和 PPO/SAC 主算法。

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
  check_command: python3 ../scripts/check_research_ledger.py --goal Goal144 --goal-doc doc/goals/goal144_alos_rl_residual_20261111.md --strict
```

## 1. 为什么现在做这个

Goal143 已证明 C0 的 direct-command 错配只是第一层原因：ALOS 把 C0 从 `0.10`
拉到 `0.85`，但 frozen-BC + 零 PPO 残差仍是 `0.85`，而纯 ALOS 在同 checkpoint、
同 dev40 达到 `0.925`。因此 frozen-BC 残差基座相对纯 ALOS 损失约 `7.5 pp`；
继续在 frozen-BC 上加 PPO 不能回答“RL 残差是否能超过传统方法”。

本 Goal 保留 Goal143 的 frozen-BC 组作为既有失败证据，不改它；新增纯 ALOS 基座上的
PPO/SAC 残差组。主臂是 3D command residual（在 ALOS 输出指令上加残差），对照臂是
actuator residual（在 PD/PI 输出上加小残差）。这也是对 `terminal capture` 和
动作权限审计的最小、可证伪实验。

## 2. 实验矩阵

| arm | 基座 | 残差空间 | PPO | SAC | 用途 |
|---|---|---|---|---|---|
| `C-full-alos-command` | 纯 ALOS | 3D heading/pitch/speed command | 1M pilot | 同协议后续 | 主臂；避免 actuator 限幅 |
| `C-full-alos-actuator` | 纯 ALOS + PD/PI | actuator，scale `0.25` | 1M pilot | 同协议后续 | 动作空间对照 |
| `C-full-alos-command`（zero） | 纯 ALOS | 零残差 | eval | eval | parity 基线，应接近 `0.925` |
| Goal143 frozen-BC + PPO | frozen-BC | actuator `1.5` | 已有 | 已有 | 失败证据，不重跑 |

约束：同一 train seed `86201`、dev seeds `20–59`、held-out `100–149` 关闭；
1M transitions；每 100k 冻结 dev 评估；C0 连续两次过 Gate 才晋级。SAC 只在 PPO
出现非零残差信号、或失败明确指向优化/探索时启动，不和 PPO 样本合并。

## 3. 预注册 Gate

| check | C0 阈值 |
|---|---:|
| default success | `>=0.90` |
| strict125 success | `>=0.70` |
| plane miss | `<=0.10` |
| timeout | `<=0.10` |
| safety | `=0` |
| promotion | 连续 2 次通过 |
| pilot signal vs pure ALOS | `>=0.925` 且 residual 未塌缩；`>=0.95` 才称 pilot 改善 |

若 C0 通过，再跑 C1；C2–C6 保持 Goal136 原语义。1M pilot 只有 1 个 train seed，
不能写成 formal；screen 通过后才扩 3 seeds。若 C0 仍失败，写 `goal144_result`/blocker，
不换算法、不扫 seed、不放宽 Gate、不打开 held-out。

## 4. 可证伪假设

- **H144-A：frozen-BC 是主瓶颈。** 去掉 frozen-BC 后，ALOS+PPO command residual
  应至少保持 `>=0.90`，且 zero-residual parity 与纯 ALOS 接近。若 command/actuator
  都掉回 `0.85` 且残差接近零，则 H144-A 被削弱。
- **H144-B：command residual 优于 actuator residual。** command residual 在 ALOS
  指令层修正，能避开末段 actuator 限幅；若 command 臂的 plane miss 更低且 residual
  clip rate 更低，则支持。若 actuator 反而更好，则问题在 command 参数化。
- **H144-C：末段捕获是几何/权限问题，不是 reward。** 失败应集中在 goal plane
  radius 附近，且 timeout/safety 为 0；若改 reward 前先看到 pure ALOS 也失败，
  才进入 terminal capture/guidance 审计。
- **H144-D：只有优化/探索才需要 SAC 对照。** 若 ALOS+RL 的残差非零、动作未饱和、
  但 success 不动，才开 SAC 同协议归因；否则不把 SAC 当作绕过基座问题的手段。

## 5. 执行计划

- [x] Step 0：冻结 `goal144_manifest.json`、4 个 config SHA、C0–C6 bounds、reward/terminal、预算、dev/held-out。
- [x] Step 1：本地配置验证；确认 command wrapper 与 SAC 一样传递 `command_residual.terminal_gate_*`；跑 command residual 2-env smoke。
- [ ] Step 2：远端 PPO `C-full-alos-command` 与 `C-full-alos-actuator` 各 1M、dev40；旧 `S_s86201` 因分层 Gate 未接线判 `INVALID` 并停止，修复后用 `S_s86201_gatefix1` 重跑；先看 C0 100k/200k Gate。
- [ ] Step 3：若 C0 通过，继续 C1 并做 pure ALOS / zero-residual parity；若失败，写 result/blocker。
- [ ] Step 4：只有 PPO 出现非零残差信号或明确优化/探索证据，才启动 SAC 同协议对照。
- [ ] Step 5：screen 通过后才扩 3 train seeds；不打开 held-out，不把 pilot 当 formal。
- [ ] Step 6：写 `goal144_result_20261111.md`、Ledger、更新 STATE；运行 checker；不自动 commit/push。

## 6. 运行与 Git

```bash
cd settled_局部规划器_反应式
PYTHONPATH="src:../settled_全局规划器/src:$SB3" \
  python scripts/train_batch_lowlevel_ppo.py \
  --training-config configs/lowlevel/goal144/C-full-alos-command_ppo_pilot.json \
  --output artifacts/lowlevel/goal144_alos_rl_20261111/ppo/C-full-alos-command/S_s86201 \
  --curriculum-disable-direct-command \
  --curriculum-eval-interval 100000 \
  --curriculum-eval-seed-count 40
```

```bash
PYTHONPATH="src:../settled_全局规划器/src:$SB3" \
  python scripts/train_pc_command_residual_sac.py \
  --config configs/lowlevel/goal144/C-full-alos-command_sac_pilot.json \
  --output artifacts/lowlevel/goal144_alos_rl_20261111/sac/C-full-alos-command/S_s86201 \
  --curriculum-disable-direct-command \
  --curriculum-eval-interval 100000 \
  --curriculum-eval-seed-count 40
```

禁止 `git add -A`；只 add 本 Goal 相关文件，提交前运行 `git diff --cached --check`。

## 7. Goal 提示词

```text
你现在执行 Goal144：在纯 ALOS 基座上训练 PPO/SAC 残差，检验去掉 frozen-BC 后
C0 是否能过 Gate，并区分 command residual 与 actuator residual。

先读：
- doc/goals/goal144_alos_rl_residual_20261111.md
- doc/goals/goal143_result_20261111.md
- research/blockers/2026-11-11-goal143-c0-frozen-bc-terminal-capture.md

硬约束：
1. PPO/SAC 固定，不换 REDQ/DroQ；
2. 不改 reward、terminal、success、goal radius、dev/held-out；
3. frozen-BC 旧组保留，不重跑、不覆盖；
4. 先跑 command residual PPO 1M + dev40；actuator residual 作为对照；
5. C0 连续两次过 Gate 才晋级；失败写 result/blocker，不扫 seed；
6. 只有明确优化/探索证据才启动 SAC 同协议对照；
7. 不自动 commit/push。
```

## 8. 课程分层 Gate 接线审计（2026-11-11）

Goal144 旧 `S_s86201` PPO 在 C0 100k dev40 已达到 default/strict `0.925`、
plane miss `0.075`、timeout/safety `0`，但 `curriculum_decisions.jsonl` 仍写
`CONTINUE / gate not met`。根因不是模型或阈值，而是
`StageGate.stratified_default_success_min` 被读取、`parse_eval_summary()` 却从未
填充 `StageEvaluation.stratified_default_success`，导致分层条件永远为 False。

已修复：

- evaluator 为每个 episode 写 `curriculum.stratum`；C0 用初始航向符号两桶，
  C1/C2/C6 用 path family，C3/C4 用深度半区，C5 用 current period 慢/快半区。
- `parse_eval_summary()` 聚合每个 stratum 的 default success；stratified Gate
  缺元数据时显式报错，不再静默永不晋级。
- `combine_eval_summaries()` 接收当前 stage 的 `StageGate`，只用 default-radius
  评估做分层。
- 回归测试覆盖“缺失分层必须报错”和“单分层低于阈值则总体再高也不能晋级”。

旧 `S_s86201` artifacts 保留但课程决策日志标 `INVALID`，不作为晋级证据；
新跑使用 `S_s86201_gatefix1`，不覆盖旧 checkpoint。C0 阈值、reward、terminal、
success、goal radius、dev seeds 与 PPO/SAC 算法均未改变。

### 8.1 gatefix2：回滚后 per-stage 预算修复与实测

gatefix1 暴露出第二个接线问题：C1 失败回滚到 C0 后，预算判定仍用全局
`completed_transitions`，导致 500k 直接误判 C0 `STAGE_BUDGET_EXHAUSTED`。已改为
每个 stage 维护累计 consumed transitions，回滚后只恢复该 stage 的剩余预算。

gatefix2（command 与 actuator 两条臂的课程决策一致）：

| transitions | stage | decision | default / strict | plane miss | 分层 |
|---:|---|---|---:|---:|---|
| 100k | C0 | `CONTINUE` | `0.925 / 0.925` | `0.075` | `nonneg 0.952 / neg 0.895` |
| 200k | C0 | `PROMOTE -> C1` | `0.925 / 0.925` | `0.075` | 同上 |
| 300k | C1 | `CONTINUE` | `0.65 / 0.65` | `0.35` | `P0 0.633 / P1 0.700` |
| 400k | C1 | `ROLLBACK -> C0` | `0.65 / 0.65` | `0.35` | 同上 |
| 500k | C0 | `CONTINUE` | `0.925 / 0.925` | `0.075` | 同上 |
| 600k | C0 | `STAGE_BUDGET_EXHAUSTED` | `0.925 / 0.925` | `0.075` | 同上 |

因此当前可以确认：分层 Gate、两次滞后晋级、严重失败回滚、回滚后 anchor 和
per-stage hard cap 都已真实执行。全课程尚未跑通：C1 仍为 `0.65`，是当前瓶颈。

### 8.2 残差活性观察（待 1M 后正式裁决）

gatefix2 的 C0/C1 评估中，command 臂 `command_residual_abs_mean` 约
`8e-5–1.5e-4`，基本为零；actuator 臂 `actuator_residual_abs_mean` 约
`0.08–0.25`，但两条臂的 success/plane-miss 完全相同。这更像
`PPO_RESIDUAL_COLLAPSE / RESIDUAL_NO_CLOSED_LOOP_EFFECT`，而不是样本效率不足。
若 1M 结束仍成立，下一实验应是 SAC 同协议对照，并把“残差是否真正改变动作与
末段几何”作为主诊断；不换 reward、不改 terminal、不扫 seed。

纵向误差若在 1M/后续 C-line screen 中成为主导失败原因，按
`doc/goals/goal140_vertical_tracking_reward_plan_20261111.md` §0/§4.2 的
potential-based vertical/heave/predicted-error reward 另开 reward arm 重跑；
不在当前 Gate 中途改 reward。

