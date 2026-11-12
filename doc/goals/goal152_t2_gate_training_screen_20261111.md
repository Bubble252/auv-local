# Goal152：训练时启用 near10_2 残差门控的 3-seed screen

**日期**：2026-11-11<br>
**状态**：`PREREGISTERED / FROZEN_BEFORE_MEASUREMENT / TRAIN_SCREEN / HELDOUT_CLOSED`<br>
**上游**：Goal151 eval-only Gate、Goal86 T2-SAC 3-seed、Goal85 位置契约、Goal140 纵向奖励待命<br>
**优先级**：`P1`。Goal151 已证明“把 residual 权限限制在 10→2 m”能在不训练的情况下保留
T2 success 并收回 path 代价；本 Goal 只验证训练时启用同一 gate 是否仍成立。

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
  check_command: .venv/bin/python ../scripts/check_research_ledger.py --goal Goal152 --goal-doc doc/goals/goal152_t2_gate_training_screen_20261111.md --strict --require-result
```

- `required: true`：本 Goal 是训练型探索；失败后必须先归因，不能在同一 Gate 内换算法、
  改 reward 或扩 seed。
- 本 Goal 的成功标准是 **screen 信号**，不是 formal 收益。5M formal 必须另开预注册。

## 1. 背景与证据

Goal151 在 3 个 Goal86 T2-SAC checkpoint 上做了 eval-only gate 反事实：

| variant | default policy/zero | strict125 policy/zero | default Δpath | strict125 Δpath |
|---|---:|---:|---:|---:|
| full | 85/78 | 46/42 | `+0.2777 m` | `+0.2764 m` |
| near10_2 | 88/78 | 48/42 | `-0.0007 m` | `-0.0013 m` |
| shell10_3 | 86/78 | 46/42 | `-0.0017 m` | `-0.0025 m` |

机制证据：

```text
full/default residual RMS:   <=2m 1.4719, 2-10m 1.3713, >10m 1.3290
near/default residual RMS:   <=2m 1.4611, 2-10m 0.8907, >10m 0.0001
shell/default residual RMS:  <=2m 0.0000, 2-10m 0.8136, >10m 0.0003
```

因此 primary 选 `near10_2`：

```text
residual_terminal_gate_enabled = true
residual_terminal_gate_enter_m = 10.0
residual_terminal_gate_full_m  = 2.0
learner_terminal_gate_mode     = none
```

`shell10_3` 只作为备份，不在本 Goal 训练。Goal151 没有纵向退化，Goal140 reward pilot
仍不触发。

## 2. 假设

### H152-A：训练时启用 near10_2 仍保留 success，并保持 path parity

成立应看到：

- pooled policy - zero `Δsuccess >= +6/120`（default）；
- pooled policy - zero `Δsuccess >= +4/120`（strict125）；
- pooled `Δpath <= +0.05 m`（两个 scenario）；
- 至少 `2/3` 个训练 seed 的 default `Δsuccess >= 0`。

失败应看到：success 跌回 zero，或 path 仍 `+0.15 m` 以上；此时判 `TRAIN_GATE_FAIL`，
不能靠扩 seed 或换 checkpoint 救。

### H152-B：收益不是来自“训练把 residual 学成零”

成立应看到：

- 2 m 内 residual RMS `>= 0.5`；
- 10 m 外 residual RMS `<= 0.05`；
- policy 的 success 高于 zero，且 residual 不是全零。

失败应看到：residual 全零或接近零；此时判 `INVALID_NO_RESIDUAL`，不能把传统控制器
的结果算作 gate 收益。

### H152-C：gate 训练不伤纵向跟踪

成立应看到：

- policy `path_vertical_rmse` 不高于 zero `+0.03 m`；
- `eta_done_distance` 的纵向分量不恶化；
- timeout/safety 不高于 zero。

失败应看到：纵向 RMSE 或 terminal vertical miss 明显恶化。此时不换算法、不改 reward；
先写 blocker，再按 Goal140 的 `F_z / F_zdot / F_predz` 方案另开 reward pilot。

### H152-D：3-seed 信号不是单 seed 偶然

成立应看到：至少 2/3 个训练 seed 的 default `Δsuccess >= 0`，且 pooled CI 不跨 0。

失败应看到：pooled 增益完全由 86203 贡献，另外两个 seed 为负或零；此时判
`SEED_CONCENTRATED`，不 promotion，先查训练分布和 gate 内 credit。

## 3. 数据与隔离

| 项目 | 值 |
|---|---|
| 训练 seeds | `86201–86203` |
| 训练预算 | 每 seed `1,000,000` transitions |
| 训练配置 | 复用 Goal86 T2-SAC，只新增 gate 字段 |
| num_envs | `256` |
| batch_size | `256` |
| buffer_size | `500000` |
| learning_starts | `20000` |
| 算法 | SAC，不改算法族 |
| 评估 | dev seeds `20–59`，每 run 40 episodes |
| scenarios | default、strict125 |
| success | Goal85 `eta_tight_at_done` |
| zero baseline | 复用 Goal145 T2 zero，dev40 同协议 |
| full 参考 | Goal86 T2 3 checkpoints，只作历史对照 |
| held-out | `100–149` 关闭 |
| reward / terminal / success / radius | 不改 |

禁止：

- 在训练中途改 reward、terminal、success 或 gate 曲线；
- 用 validation/dev 结果挑 checkpoint 后再报 formal；
- 为单个 arm 增加 transitions 或 seed；
- 打开 held-out；
- 在同一 Goal 内切换 PPO/REDQ/DroQ 或改动作参数化。

## 4. 执行计划

- [ ] Step 0：核对 Goal151 结果、Goal86 配置、checkpoint SHA 和远端 GPU；
- [ ] Step 1：生成 3 个 Goal152 训练配置，固定 gate 字段和 seed；
- [ ] Step 2：本地静态测试 + 20k transitions smoke，确认 trace 的 `gate`/residual 正常；
- [ ] Step 3：3 seed 并行训练，各 1M transitions，分配 3 张空闲 4090；
- [ ] Step 4：6 个 policy eval run（3 seed × 2 scenario）+ 复用 zero；
- [ ] Step 5：配对汇总 success、path、vertical、residual distance bins；
- [ ] Step 6：按 §5 Gate 裁决并写 result；
- [ ] Step 7：失败则写 blocker、至少两个 hypothesis、decision、LEDGER、STATE，运行 checker。

### 算力利用

- 3 个训练 seed 分配到 3 张 GPU 并行，避免串行等待；
- 训练前先 `nvidia-smi` 和 `pgrep`，不使用他人正在跑满的卡，不 kill 他人进程；
- 6 个 eval run 用 CPU/float64 并行，单 run 约几十秒；
- 不为“用满算力”给任何 arm 偷加 transitions、batch 或 seed；总算力增加只来自并行。

## 5. Gate

按 scenario 分别汇总 3 个 checkpoint 的 pooled 120 对。

| check | 阈值 |
|---|---:|
| default success effect | policy - zero `>= +6/120` |
| strict125 success effect | policy - zero `>= +4/120` |
| path cost | policy - zero `<= +0.05 m`，两个 scenario 都要满足 |
| seed consistency | 至少 `2/3` seed 的 default Δsuccess `>= 0` |
| residual activity | 2 m 内 RMS `>= 0.5`，10 m 外 RMS `<= 0.05` |
| vertical guard | policy vertical RMSE `<= zero + 0.03 m` |
| safety | timeout/safety 不高于 zero |
| 数值 | 无 NaN/Inf；否则 INVALID |

裁决：

| 观察 | 判读 | 下一步 |
|---|---|---|
| 两个 scenario 都过 success/path/vertical | `SCREEN_PASS_READY_FOR_5M_FORMAL` | 另开 5M formal 预注册 |
| success 过、path 不过 | `PATH_TRADEOFF_REMAINS` | 查 gate 内 residual 幅度与 reward credit，不扩 seed |
| path 过、success 不过 | `CAPTURE_TRADEOFF` | 查 terminal capture 与 gate 内 credit，不换算法 |
| 只有 86203 有增益 | `SEED_CONCENTRATED` | 查训练分布/探索，不 promotion |
| residual 全零 | `INVALID_NO_RESIDUAL` | 不把 zero 结果算作 gate 收益 |
| 纵向 RMSE 恶化超过阈值 | `VERTICAL_DOMINANT` | 写 blocker，按 Goal140 另开 reward pilot |

## 6. 纵向误差分支（常驻提醒）

如果本 Goal 出现“残差方向看起来正确，但纵向误差持续不降”或 terminal vertical miss
主导失败，先回看：

`goal140_vertical_tracking_reward_plan_20261111.md`

不要在本 Goal 内临时改 reward 或扫权重。新开预注册后，按以下顺序做：

1. 1 seed、200k transitions 的 `R1 = baseline + F_z` pilot；
2. 通过后再跑 `R2 = R1 + F_zdot` 和 `F_predz`；
3. 保持 success、terminal、radius、gate 和 dev seeds 不变；
4. 只有纵向 RMSE/p95 改善且 strict success 不退化，才扩到 3 seeds。

## 7. Git

禁止 `git add -A`。只 add 本 Goal 相关文件，提交前运行：

```bash
.venv/bin/python ../scripts/check_research_ledger.py \
  --goal Goal152 \
  --goal-doc doc/goals/goal152_t2_gate_training_screen_20261111.md \
  --strict --require-result
git diff --cached --check
```

## 8. Goal 提示词

```text
你现在执行 Goal152：训练时启用 near10_2 残差门控的 3-seed screen。

先读：
- doc/goals/goal152_t2_gate_training_screen_20261111.md
- doc/goals/goal151_result_20261111.md
- doc/goals/goal86_t2_3seed_result_20261111.md
- doc/goals/goal140_vertical_tracking_reward_plan_20261111.md

硬约束：
1. 不换算法、不改 reward/terminal/success/radius、不打开 held-out；
2. gate 固定 enter=10、full=2、mode=none；
3. 3 seeds x 1M transitions，dev40 default/strict125，policy vs zero；
4. 3 张 GPU 并行训练，eval 并行，但不给任何 arm 偷加预算；
5. 失败先归因，不扩 seed；纵向主导才进入 Goal140 的新 reward pilot。
```
