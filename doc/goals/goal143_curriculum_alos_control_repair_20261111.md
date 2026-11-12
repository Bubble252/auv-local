# Goal143：C-line C0/C1 ALOS 控制语义修复与同协议课程对照

**日期**：2026-11-11<br>
**状态**：`C0_GATE_FAILED / FROZEN_BC_RESIDUAL_BOTTLENECK / TERMINAL_CAPTURE_NEAR_RADIUS / PIVOT_TO_GOAL144 / HELDOUT_CLOSED`<br>
**上游**：Goal136 C-line `C0_DIRECT_COMMAND_OOD_FROZEN_BC`；Goal139 mixed-horizon；Goal86 PPO residual collapse<br>
**优先级**：`P1`（Goal136 C-line 的控制语义修复）；Goal140 仍为 `P2`，不得插队。<br>
**性质**：只修 C0/C1 的动作/命令参数化，不改 reward、terminal、success、goal radius、seed 集、held-out 或 PPO/SAC 主算法。

2026-11-11 最终状态：`C-full-alos` 与 `C-no-current-alos` 的 PPO 均跑满
1M，C0 始终 default/strict `0.85`、plane miss `0.15`，未晋级；两条 run
的 manifest 均为 `curriculum_direct_commands_enabled=false`，
`C-no-current-alos` 额外为 `curriculum_force_current_alpha_zero=true`。
同 300k checkpoint dev40：policy `0.85`、zero-residual frozen-BC `0.85`、
pure ALOS `0.925`；因此 frozen-BC 残差是主瓶颈，转入 Goal144。完整证据见
`doc/goals/goal143_result_20261111.md`。

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
  check_command: python3 ../scripts/check_research_ledger.py --goal Goal143 --goal-doc doc/goals/goal143_curriculum_alos_control_repair_20261111.md --strict
```

`required: true`。本 Goal 会训练新的课程 arm，但只改变 C0/C1 的控制语义；
进入正式 screen 前冻结配置、seed、预算、eval split 和 Gate。若 C0 Gate 连续
失败两次，回到 blocker，不允许换 seed、加预算或放宽阈值。

## 1. 背景与证据

Goal136 的 C-line 机制已接通：C0–C6 路径/深度/噪声/洋流、stage sampler、
promotion/rollback、每 100k 冻结 dev 评估、per-episode manifest、PPO/SAC
current encoder alpha ramp 都已实现。但 C0/C1 的随机 direct command 与冻结
BC 先验不兼容：

- C-full PPO 在 C0 的 100k/200k/300k dev20 都是 default `0.10`、strict `0.10`、
  plane miss `0.90`；
- 同一 100k checkpoint、同一 C0 dev20、同一 C0 环境，`zero-residual` 在
  direct command 下 `0/20`，关闭 direct command、保留 ALOS 后 `18/20`；
- 因此 C0 Gate `0.90/0.70` 本身可达，失败发生在 C0/C1 的命令分布，而不是
  PPO 残差、current encoder alpha、优化器、reward 或 terminal。

完整证据见
`research/blockers/2026-11-11-goal136-c0-direct-command-bc-mismatch.md`。

## 2. 研究问题与假设

### H143-A：C0/C1 改回 ALOS 后，冻结 BC 先验能通过 C0 Gate

成立应看到：

- `C-full-alos` 在 C0 的 100k、200k dev40 冻结评估中 default `>=0.90`；
- strict `>=0.70`，plane miss `<=0.10`，timeout `<=0.10`，safety `=0`；
- 连续两次通过后 promotion 到 C1，而不是停在 C0。

失败应看到：

- C0 仍低于 Gate，或 ALOS 与 direct 的差异小于 seed 噪声；
- 此时停止本 arm，回到动作/命令参数化与 BC 数据分布审计，不换算法。

最便宜证伪：复用 Goal136 100k checkpoint 的 eval-only ALOS 反事实（已完成：
`0.00 -> 0.90`）；正式训练仍需独立 1M pilot，不能把 eval-only 当正式结论。

### H143-B：ALOS 课程在相同预算下优于 C-hard 最终混合

成立应看到：

- 1M screen、dev40、3 seeds 下，`C-full-alos` 至少一个压力档
  （nominal/current15/current20/plant20/combined20）的 default success
  比 `C-hard` 高 `>=1/40`；
- 不低于 paired control 的 strict success `-1/40`；
- plane miss、timeout、safety 不出现方向性退化。

失败应看到：ALOS 课程与 C-hard 差异小于 seed 噪声，或在压力档上更差。
此时判 `CURRICULUM_ALOS_NULL_OR_HARM`，不扩 seed，不报 formal。

### H143-C：C-no-current-alos 在洋流压力档更差

成立应看到：`C-no-current-alos` 在 current15/current20/combined20 的
default success 低于 `C-full-alos`，且差异超过 seed 噪声。

失败应看到：两者无差异；此时不能把 current encoder 的收益写成课程结论。

## 3. 数据、预算与隔离

| 项目 | 冻结值 |
|---|---|
| 训练基座 | Goal136 C-line / Goal125 B15 frozen BC residual |
| 算法 | PPO 与 SAC 分开训练、分开报告；不引入 REDQ/DroQ |
| pilot seed | `86201` |
| screen seeds | `86201–86203` |
| pilot 预算 | PPO `1,000,000` env transitions；SAC `1,000,192` |
| formal 预算 | 每 arm `5,000,000`，仅在 screen 通过后启动 |
| dev seeds | `20–59`，每次评估 40 episodes |
| held-out | `100–149` 关闭 |
| 评估起点 | 完整 35 m，`reference.start_progress_m=0` |
| 固定项 | reward、terminal、success、goal radius、sensor profile、stage mix |

禁止：

- 用 held-out 选 arm 或 Gate；
- 用 pilot 单 seed 写正式课程收益；
- 为救 C0 改 Gate 阈值、换 seed、加预算或放宽 plane-miss；
- 在同一 Gate 内更换 PPO/SAC 主算法、reward 或动作接口。

## 4. 实验臂

| arm | C0/C1 控制 | C0–C6 分布 | 用途 |
|---|---|---|---|
| `C-hard` | C6 起跑，ALOS | 最终混合 | 同预算强基线；复用 Goal136 C-hard |
| `C-full-alos` | **ALOS path tracking** | C0→C6 晋级 | 主修复臂 |
| `C-full-direct` | 随机 direct command | C0→C6 | 已有 Goal136 失败证据；不作为正式课程收益 |
| `C-no-current-alos` | ALOS | C0→C6，actor current alpha 固定 0 | 显式 current 观测消融 |

`C-full-alos` 与 `C-full-direct` 只差 C0/C1 控制语义；`C-full-alos` 与
`C-hard` 只差“课程晋级 vs 直接最终混合”。这三组必须分开报告。

## 5. 执行计划

- [ ] Step 0：冻结 Goal143 配置 SHA、C0–C6 bounds、reward/terminal、预算、
  dev/held-out、Gate；写 `goal143_manifest.json`。
- [ ] Step 1：实现/验证 `--curriculum-disable-direct-command` 在 PPO、SAC、
  PPO eval、SAC eval 与 manifest 中的一致性；本地回归通过。
- [ ] Step 2：本地/远端极小 smoke，确认 C0 episode manifest、ALOS command、
  checkpoint 与 eval summary 都落盘。
- [ ] Step 3：跑 `C-full-alos` PPO/SAC 1M pilot，dev40；同时保留 `C-hard`
  与 `C-full-direct` 的已有结果作为对照。
- [ ] Step 4：C0 Gate：default `>=0.90`、strict `>=0.70`、plane miss `<=0.10`、
  timeout `<=0.10`、safety `=0`；连续两次通过才 promotion。
- [ ] Step 5：若 C0 通过，跑到 C2/C3 并检查 promotion/rollback；1M screen
  与 `C-hard` 按 dev40 配对比较。
- [ ] Step 6：只有 screen 通过才跑 5M formal；否则写 result/blocker，不扩 seed。
- [ ] Step 7：写 `goal143_result_20261111.md`、Ledger、更新 STATE；
  运行 checker；不自动 commit/push。

## 6. Gate 与失败裁决

| check | 阈值 |
|---|---:|
| C0 default success | `>=0.90` |
| C0 strict125 success | `>=0.70` |
| C0 plane miss | `<=0.10` |
| C0 timeout | `<=0.10` |
| C0 safety | `=0` |
| promotion | 连续 2 次通过 |
| rollback | 连续 2 次 miss 且不是第一阶 |
| screen 主指标 | dev40 default success，paired vs `C-hard` |
| screen guard | strict `>= control -1/40`；plane/timeout/safety 不退化 |

| 观察 | 判读 | 下一步 |
|---|---|---|
| C0 ALOS 仍 `<0.90` | 不是单纯 direct command | 回到动作/命令参数化与 BC 数据审计 |
| C0 通过但 C2/C3 停滞 | 课程难度/晋级门槛问题 | 只读审计该阶 transition，不扫 seed |
| C-full-alos 与 C-hard 差异小于 seed 噪声 | 课程无收益 | 判 null，不报 formal |
| 压力档更差 | 课程 harm | 保留失败证据，回 Goal136/Goal140 决策 |

## 7. 运行与 Git

训练入口（示例，实际 GPU 在启动前用 `nvidia-smi` 选择）：

```bash
cd settled_局部规划器_反应式
PYTHONPATH="src:../settled_全局规划器/src:$SB3" \
  python scripts/train_batch_lowlevel_ppo.py \
  --training-config configs/lowlevel/goal143/C-full-alos_ppo_pilot.json \
  --output artifacts/lowlevel/goal143_curriculum_alos_20261111/ppo/C-full-alos/S_s86201 \
  --curriculum-disable-direct-command \
  --curriculum-eval-interval 100000 \
  --curriculum-eval-seed-count 40
```

```bash
PYTHONPATH="src:../settled_全局规划器/src:$SB3" \
  python scripts/train_pc_command_residual_sac.py \
  --config configs/lowlevel/goal143/C-full-alos_sac_pilot.json \
  --output artifacts/lowlevel/goal143_curriculum_alos_20261111/sac/C-full-alos/S_s86201 \
  --curriculum-disable-direct-command \
  --curriculum-eval-interval 100000 \
  --curriculum-eval-seed-count 40
```

禁止 `git add -A`。只 add 本次相关文件，提交前运行：

```bash
python3 ../scripts/check_research_ledger.py \
  --goal Goal143 \
  --goal-doc doc/goals/goal143_curriculum_alos_control_repair_20261111.md \
  --strict
git diff --cached --check
```

## 8. Goal 提示词

```text
你现在执行 Goal143：修复 Goal136 C-line 的 C0/C1 控制语义，并用同协议
PPO/SAC 对照验证课程是否真正可晋级。

先读：
- doc/goals/goal143_curriculum_alos_control_repair_20261111.md
- doc/goals/goal136_environment_diversity_training_eval_plan_20261111.md
- research/blockers/2026-11-11-goal136-c0-direct-command-bc-mismatch.md
- research/paper_reading/2026-11-11-goal136-curriculum-control-prior-shift.md

硬约束：
1. 不换 PPO/SAC 主算法，不引入 REDQ/DroQ；
2. 不改 reward、terminal、success、goal radius、dev/held-out；
3. C0/C1 只把随机 direct command 改为 ALOS；C2–C6 保持 Goal136 原语义；
4. 先跑 1M pilot + dev40，C0 Gate 通过后才跑 screen/5M；
5. 失败先归因，写 blocker、至少两个可证伪 hypothesis、reading note 和 LEDGER；
6. 不扫 seed、不放宽 Gate、不覆盖旧 checkpoint/artifacts；
7. 不自动 commit/push。

执行顺序：
- 校验配置 SHA、C0–C6 bounds、reward/terminal 与预算；
- 跑 C-full-alos PPO/SAC 1M pilot；
- 检查 C0 100k/200k dev40 的 success/strict/plane/timeout；
- 若通过，继续 promotion 并与 C-hard 做 dev40 paired screen；
- 若失败，停止并写 goal143_result/blocker。
```

