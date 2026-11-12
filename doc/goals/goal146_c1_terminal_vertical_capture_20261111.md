# Goal146：C1 近终点纵向捕获的 terminal guidance 权限单变量

**日期**：2026-11-11<br>
**状态**：`COMPLETED_EVAL_ONLY / RESULT_RECORDED / NO_SUCCESS_CONVERSION / HELDOUT_CLOSED`<br>
**结果**：`goal146_result_20261111.md`<br>
**上游**：`goal145_result_20261111.md`、`goal144_result_20261111.md`、`goal140_vertical_tracking_reward_plan_20261111.md`<br>
**优先级**：`P1`。Goal145 已把 C1 的 plane miss 定位到 `1.25–2.0 m` 的近终点纵向几何。
**性质**：只复用 `C1_000300000` checkpoint 做 dev40 闭环评估；不训练、不改 reward/terminal/success/goal radius、不扫 seed、不打开 held-out。

## 0. 用户提醒与边界

用户要求：如果途中确认纵向误差是主要限制，要回看 Goal140 的奖励修缮方案并重跑实验。
本轮先把该思想写入 Goal140；Goal146 **不直接加纵向奖励**，因为：

- C1 当前 reward 已经含 `w_approach_vertical=1.0`、`w_approach_pitch=0.5`；
- Goal145 显示失败 episode 的 pitch/speed guidance rate clip 显著高于成功，而舵面
  saturation 为 0；
- Goal145 还显示现有 command/actuator residual 有数值，但对 success 和 terminal
  geometry 没有闭环效果。

因此先问“终端制导权限是否足够”，再决定是否让 reward 去推动策略利用该权限。若权限
单变量本身就能把失败转 success，而 policy 没有额外收益，下一步才是 Goal140 的
potential-based `R1/R2` pilot；若权限没有效果，不能把 reward 当作替代品。

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
  check_command: .venv/bin/python ../scripts/check_research_ledger.py --goal Goal146 --goal-doc doc/goals/goal146_c1_terminal_vertical_capture_20261111.md --strict --require-result
```

`required: false`：这是冻结的 eval-only Gate。Gate 记录前不得改 arms、阈值、seed、
checkpoint 选择规则或 guidance 配置；失败后再进入 blocker/research loop。

## 1. 证据与问题

Goal145 在 `C1_000300000`、`dev40`、`--disable-curriculum-direct-command` 下得到：

| 指标 | 结果 |
|---|---:|
| command policy success | `26/40` |
| command zero-residual success | `26/40` |
| plane miss | `14/40` |
| 失败 final distance | `1.6101 m` |
| 失败 final vertical | `1.5722 m` |
| 失败 final horizontal | `0.3235 m` |
| failure bins `1.25–1.5 / 1.5–2.0 / >=2.0` | `7 / 6 / 1` |
| success pitch rate clipped | `0.811` |
| failure pitch rate clipped | `0.959` |
| success speed rate clipped | `0.120` |
| failure speed rate clipped | `0.454` |
| stern saturation | `0` |

失败 seeds：`[24,25,27,32,33,34,35,36,39,40,41,44,47,49]`。

问题是：在舵面没有饱和的情况下，**把 guidance command rate limiter 放宽，是否能
让 ALOS/BC/PPO 链路真正执行更快的 pitch 或 speed 修正，并把近终点纵向 miss 拉回
1.25 m 捕获半径**？

## 2. 可证伪假设

### H146-A：pitch rate limit 是主要 binding constraint

机制：失败 episode 的 raw pitch target 变化被 `pitch_rate_limit_rad_s=0.2618` rad/s
截断，末段 pitch 修正滞后，导致纵向 miss。

成立应看到：

- `P1`（20 deg/s）相对 `Z` 至少把 `Z` 的 7 个 `1.25–1.5 m` 失败中的 2 个转成 success；
- 失败组 pitch rate clipped fraction 明显下降，且 final vertical 中位数下降；
- 不出现超过 `1/40` 的新增 success loss，timeout/safety 不增加。

失败应看到：success 不增加且 final vertical / pitch clip 基本不变。此时停止继续增大
pitch limit，判 `PITCH_RATE_NOT_BINDING`。

### H146-B：speed rate limit 是主要 binding constraint

机制：失败组 speed command clipped fraction 为 `0.454`，末段速度参考无法及时下降或
调整，导致闭合速度/纵向捕获时序错误。

成立应看到：

- `S1`（`speed_rate_limit_mps2=0.35`）至少把 1 个近终点失败转成 success，或显著降低
  final distance / final vertical；
- speed clipped fraction 下降，且不增加 overshoot、plane miss 或 timeout。

失败应看到：success 和 final geometry 均不动。此时判 `SPEED_RATE_NOT_BINDING`。

### H146-C：权限足够但 policy/zero 的闭环差异会告诉我们瓶颈层级

在同一个 guidance 臂下，`policy` 与 `zero-command-residual` 使用同一 checkpoint 和 seed。

- 若两者都改善：改善来自 guidance/传统控制链，不是 PPO residual；不能把它写成
  “神经网络学会了纵向捕获”。
- 若只有 `policy` 改善：PPO residual 能利用新增权限，下一步检查该收益是否可训练复现。
- 若两者都不改善：当前 terminal guidance 权限不是瓶颈，转查 terminal capture/reward
  或 plant/geometry；不得直接扫 reward 权重。

## 3. 冻结的 arms 与隔离

| arm | guidance 改动 | 目的 |
|---|---|---|
| `Z` | 原 C0 guidance，`pitch=0.2617993878`、`speed=0.25` | 复现 Goal145 基线 |
| `P1` | `pitch_rate_limit_rad_s=0.3490658504`（20 deg/s） | pitch 权限主假设 |
| `P2` | `pitch_rate_limit_rad_s=0.5235987756`（30 deg/s） | 预注册上限剂量，不继续扫 |
| `S1` | `speed_rate_limit_mps2=0.35` | speed 权限主假设 |

所有 profile 由 `scripts/build_goal146_guidance_configs.py` 从 C0 生成，除表中一个字段
外必须逐字段相同；哈希写入 `configs/lowlevel/goal146/guidance_manifest.json`。

固定：

- checkpoint：`artifacts/lowlevel/goal144_alos_rl_20261111/ppo/C-full-alos-command/S_s86201_gatefix2/curriculum_checkpoints/C1_000300000.pt`；
- training config：`configs/lowlevel/goal144/C-full-alos-command_ppo_pilot.json`；
- environment config：`configs/lowlevel/goal136/curriculum/C0.json`；
- stage：`C1`，`--disable-curriculum-direct-command`；
- seeds：`20–59`，共 40 个；
- policy：`policy`；paired control：`zero-command-residual`；
- `--success-frame both`，主指标使用真值 `eta`；
- dtype `float64`，device `cuda`；
- held-out `100–149` 关闭；
- `C1_000400000` 只在 300k 某个 arm 通过 Gate 后作为 guard 复跑，不参与选 checkpoint。

不加入 `PS` 组合臂；只有单变量出现正信号后才允许新开预注册组合臂。

## 4. 执行计划

- [ ] Step 0：确认 4 个 guidance 配置哈希、checkpoint 哈希、C1 配置哈希；
- [ ] Step 1：远端检查 GPU/进程，写入 run manifest；
- [ ] Step 2：跑 `Z/P1/P2/S1 × policy/zero`，每个 40 episodes，保存 summary 和 rich trace；
- [ ] Step 3：审计 success、plane miss、timeout/safety、failure bins、final vertical/horizontal、pitch/speed clip；
- [ ] Step 4：按 §5 Gate 裁决；只在候选 arm 通过后补 400k guard；
- [ ] Step 5：写 result、blocker/ledger、更新 STATE，运行 checker；不自动 commit/push。

运行命令：

```bash
cd /data/yhx/underwater-robotics/src/plan_p1/goal136_curriculum_20261111
bash scripts/run_goal146_terminal_authority.sh
```

分析命令：

```bash
PYTHONPATH=src .venv/bin/python scripts/analyze_pc_goal146_terminal_authority.py \
  --root artifacts/lowlevel/goal146_c1_terminal_authority_20261111
```

## 5. Gate

主指标以 `policy` 对 `Z` 的 paired dev40 结果为准，`zero` 用来解释机制。

| check | 通过阈值 |
|---|---|
| 近终点转化 | 相对 `Z`，7 个 `1.25–1.5 m` 失败中至少 `2` 个在该 seed 转为 success |
| 总 success | 至少 `28/40`，且相对 `Z` 的 loss 不超过 `1/40` |
| plane miss | 不恶化超过 `2/40` |
| timeout / safety | 不增加；不出现 NaN/Inf/flew-out |
| 轨迹 guard | `path_overshoot_m_mean` 不增加超过 `0.05 m`；`eta_done_distance_m_mean` 不增加超过 `0.05 m` |
| 终端几何 | failure final vertical 中位数下降，final horizontal 不系统性恶化 |
| 机制 | 记录 pitch/speed clipped fraction 与 policy-zero 差异 |

预注册优先级：`P1` 是 pitch 主臂，`P2` 只作为上限剂量，`S1` 是独立 speed 假设。
如果多个 arm 同时通过，不能用 best-seed/best-checkpoint 选一个报成功；必须报告全部，
并在新预注册复跑中按同协议确认。

失败裁决：

| 观察 | 判读 | 下一步 |
|---|---|---|
| P1/P2 无 success 转化且 clip 不降 | pitch 不是 binding constraint | 不继续扫 pitch limit；查 speed/terminal capture |
| S1 无 success 转化且 speed clip 不降 | speed 不是 binding constraint | 不继续扫 speed limit；查 capture/reward |
| guidance 改动有效但 policy 与 zero 相同 | 传统制导链是瓶颈，不是 PPO residual | 先修传统 guidance；不宣称神经网络收益 |
| guidance 改动有效且 policy 更好 | 新权限让 residual 可学 | 开最小训练 pilot，仍不换算法 |
| 权限改变但闭环仍完全不动 | 动作/terminal 权限之外还有 capture 语义问题 | 按 Goal140/Goal81 审计 terminal，不盲扫 reward |

## 6. 与 Goal140 的交接

只有同时满足以下条件，才启动 Goal140 的 potential-based vertical reward pilot：

1. Goal146 证明 pitch 或 speed 权限确实改变了 guidance command / clipped fraction；
2. 但仍不能把 `1.25–2.0 m` 的纵向失败转成 success，或 policy 没有利用该权限；
3. 失败与 `vertical_error` / `predicted_vertical_error` 的关联在 paired seeds 上稳定；
4. 不改变 success、terminal、goal radius、dev/held-out 和主算法族。

否则继续修 terminal guidance/动作参数化，不启动 `R1/R2` 权重扫描。

## 7. Git

禁止 `git add -A`。只 add 本 Goal 的文档、配置生成脚本、运行/分析脚本、测试、result 和
ledger 相关文件；提交前运行 `git diff --cached --check`。不自动 commit/push。

## 8. Goal 提示词

```text
你现在执行 Goal146：C1 近终点纵向捕获的 terminal guidance 权限单变量。

先读 goal146_c1_terminal_vertical_capture_20261111.md、goal145_result 和
goal140_vertical_tracking_reward_plan_20261111.md。严格按预注册的 Z/P1/P2/S1
四臂执行，固定 C1_000300000、dev seeds 20–59、policy 与 zero 配对。

不要训练，不要改 reward/terminal/success/goal radius，不要扫 seed，不要打开 held-out，
不要加入组合臂。先检查 guidance 配置哈希和远端 GPU，再并行跑 dev40；保存 trace 和
summary。按 Gate 报告 success 转化、failure bins、final vertical/horizontal、
pitch/speed clipped fraction。若失败，写 blocker、至少两个可证伪假设和 ledger；
若权限足够但 policy 未利用，再写 Goal140 potential reward 的最小预注册 pilot。
```
