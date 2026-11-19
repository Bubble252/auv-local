# Goal185 blocker：C0 训练崩坏 + 课程预算被全局 iteration 压缩

**日期**：2026-11-19<br>
**Goal**：Goal185（PE2E × BCRES × MOD3–MOD0）<br>
**触发**：四个 arm 的 C0 晋升全部失败；同一 Gate 连续失败超过 2 次；现有
LR 解释不能解释 C1–C5 的训练量异常。

## 现象

- MOD3/MOD1/MOD0 都在 C0 `120 it` 触发
  `StageBudgetExhausted: stage=C0 consumed=120 hard_cap=120`，不是 OOM。
- MOD2 启动后只跑 7 it 就被队列的 `other-growth` 保护 SIGTERM，未真正训练。
- MOD3 的 C0 cross-track：17 it `0.375 m` → 19 it `0.624 m` → 120 it `1.251 m`。
- MOD1：17 it `0.432 m` → 120 it `1.292 m`；MOD0：17 it `0.478 m` → 120 it `0.716 m`。
- C0 晋升阈值是 `0.621 m`；MOD0 只差 15%，属于擦线失败。
- 四臂结果全部 `INVALID`，不能用于模块消融结论。

## 证据

- 队列日志：
  `/data/yhx/underwater-robotics/runs/auv-local/goal185/20261118-goal185-d7ea65d/logs/queue.log`
- MOD 日志：
  `/data/yhx/underwater-robotics/runs/auv-local/goal185/20261118-goal185-d7ea65d/full_wave/mod{3,2,1,0}/pe2e_bcres_mod*_seed86201.log`
- 训练记录：
  `/data/yhx/underwater-robotics/runs/auv-local/goal185/20261118-goal185-d7ea65d/full_wave/mod*/pe2e_bcres_seed86201/iteration_records.jsonl`
- 队列状态：
  `/data/yhx/underwater-robotics/runs/auv-local/goal185/20261118-goal185-d7ea65d/queue_state.json`
- 冻结阈值：
  `/data/yhx/underwater-robotics/runs/auv-local/goal184/20261118-goal184-64d8225/route_gate_calibration_pe2e_stage_matched_v4_100it_fixedlr/thresholds/goal182_route_gate_pe2e_20_3.thresholds.json`
- Goal184 MOD4 对照：
  `/data/yhx/underwater-robotics/runs/auv-local/goal184/20261118-goal184-64d8225/full_wave1/pe2e_bcres_seed86201/iteration_records.jsonl`

关键数字：

- rsl_rl adaptive 规则在每个 minibatch 调 LR，`KL < desired_kl/2` 时 `*1.5`，
  硬上限 `1e-2`。MOD3 的 iteration 1 KL 只有 `0.0018`，iteration 2 LR 已到
  `1e-2`，KL 变成 `0.1028`、clip `0.614`；iteration 6 KL `0.2635`、clip `0.763`。
- 按 LR 分桶：`≤3e-3` 时 KL 最大 `0.061`、clip 最大 `0.44`；
  `5e-3–1e-2` 时 KL 最大 `0.264`、clip 最大 `0.76`。
- 课程预算：C1 实际 22 it / 设计 108；C2 24 / 156；C3 60 / 216；
  C4 36 / 252；C5 84 / 336；C6 317 / 420。
- 代码：`src/auv_lowlevel/curriculum.py:1329` 的
  `eligible = window_full and int(iteration) >= int(min_iterations)` 使用的是
  全局 iteration；而 `src/auv_lowlevel/curriculum_trainer.py:170-177` 的
  `_stage_boundary_target` 使用 `entry + soft_budget`，两者语义不一致。

## 已排除

1. **OOM**：日志 traceback 是 `StageBudgetExhausted`，不是 CUDA OOM；MOD3
   峰值约 19–20 GB，低于 23009 MiB cap。
2. **BC 未加载**：MOD3 iteration 18 的 cross 仍是 `0.375 m`；manifest 的
   `base_checkpoint` 和 `checkpoint_sha256` 都是 v4 `86cab3fe...`。
3. **环境/config/seed 不同**：Goal184 与 Goal185 的
   `environment_config_sha256` 都是 `db4de5ff...`；seed 都是 `86201`。
4. **MOD 外超参变化**：MOD3 与 MOD4 config 的 diff 只有 `name`、`output`、
   `pc.critic_path_dim 30→0`。
5. **评价协议问题**：四臂还没进入正式评价，问题在训练课程晋升。

## 可证伪假设

1. **H185-B1（LR 放大是 C0 崩坏主因）**：若把 adaptive 上限压到 `3e-3`
   并限制每个 iteration 最多放大一次，MOD3 应在 C0 `120 it` 内回到
   cross `≤0.621 m` 并晋升。若仍失败，则 LR 不是主因，或 `3e-3` 仍过高。
2. **H185-B2（全局 iteration 压缩课程预算是 C1–C5 欠训主因）**：若把
   `min_iterations` 改成 stage-local，C1–C5 的实际训练量应达到各自 soft budget
   的 `≥90%`，且 MOD4 的 C3–C5 success 应高于当前 655-it 版本。若训练量没变
   或 success 不升，则预算不是主因。
3. **H185-B3（低 KL 连升机制比 LR 绝对值更关键）**：`1e-2` 本身不是充分条件
   （MOD1/MOD0 在 `1e-2` 稳定），MOD3 崩坏来自低 KL 触发的连续放大。若给 MOD3
   固定 `1e-2` 且不放大仍能稳定，则 H185-B3 被证伪，问题在 LR 绝对值而非
   放大机制。

## Research loop 状态

`RECORDED_PENDING_USER_DECISION`

- 2026-11-19 用户要求先把问题记录到 Goal185 md，待定夺后再改代码。
- 因此本轮暂不跑 falsification、不写 reading note、不写 `LEDGER.jsonl` decision；
  用户选定修复范围后，再按 `research/README.md` 的格式补 experiment 和 decision。

## 待定夺

1. LR 上限 `3e-3` 还是 `1e-3`？是否加每 iteration 放大次数限制或 warmup？
2. 是否修 `min_iterations` 为 stage-local，并重跑 MOD4 干净基线？
3. 队列是否取消 `--stop-on-failure`、降低 `need_mb`、失败后自动重置 pending？
4. C0 hard cap 是否从 `120` 提到 `160`？
5. MOD3/MOD4 的 actor 初始化 RNG 是否要显式配对？
6. MOD2/1/0 复用无 current BC 的质量混杂是否接受？
