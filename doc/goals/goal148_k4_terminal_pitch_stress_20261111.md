# Goal148：K4 终端 pitch 控制器的压力测试与训练接入

**日期**：2026-11-11<br>
**状态**：`COMPLETED_EVAL_ONLY / NOMINAL_K4_PASS / PLANT20_K4_39OF40 / CURRENT_COLLAPSE / GATE_FAIL / HELDOUT_CLOSED`<br>
**上游**：`goal147_result_20261111.md`、`goal136_environment_diversity_training_eval_plan_20261111.md`<br>
**优先级**：`P1`。K4 已在 nominal C1 dev40 上 `40/40`，但还没有证明它在 current/plant 压力下仍然成立。
**性质**：Stage 1 只做 eval-only stress；不改 reward/terminal/success/goal radius，不训练，不打开 held-out。Stage 2 训练只在 Stage 1 Gate 通过后另写预注册。

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
  check_command: .venv/bin/python ../scripts/check_research_ledger.py --goal Goal148 --goal-doc doc/goals/goal148_k4_terminal_pitch_stress_20261111.md --strict --require-result
```

`required: false`：Stage 1 是冻结的 eval-only Gate。记录前不改 arms、stress 剂量、seed、
checkpoint 或 Gate；失败后才进入 blocker/research loop。

## 0. 执行记录（2026-11-11）

第一次 CPU stress 是 `INVALID`：C1 curriculum sampler 覆盖了 environment config 的
`current`，current15/current20 与 nominal 完全相同。该批产物保存在
`artifacts/lowlevel/goal148_k4_terminal_pitch_stress_20261111_cpu_invalid/`，不参与裁决。

修正只增加 eval-only 参数 `--curriculum-current-speed-range-mps` 与
`--curriculum-shear-scale`，只替换 C1 的 current/shear，不改 depth/path/noise/length/
speed/reward/termination/success/goal radius。canonical 结果用 CUDA/float64 重跑，
见 `goal148_result_20261111.md`。

## 1. 证据与问题

Goal147 在固定 `C1_000300000`、nominal C1、dev40 上得到：

```text
Z   policy/zero: 26/40, plane miss 14/40
K4  policy/zero: 40/40, plane miss 0/40
K4 7/7 near-failure conversion, gained 14, lost 0
K4 policy/zero 完全相同，command residual 约 1.5e-4
K4 400k guard: policy/zero 40/40
```

K4 是传统 controller 修复：`base_controller.pitch_gain 2 -> 4`。它在 nominal 上已经
接近天花板，因此 nominal 不能证明 residual 的价值，也不能证明 K4 在压力场景下安全。
本 Goal 先回答两个问题：

1. K4 是否只在 nominal C1 有效，遇到 current/plant mismatch 会过度饱和或崩溃？
2. 在 K4 基座上，policy 与 zero 是否仍然完全同向？如果压力下开始分离，才值得进入
   Stage 2 训练。

## 2. 可证伪假设

### H148-A：K4 在压力场景仍保持 terminal capture

机制：K4 提高 pitch 角度环增益，使 stern 更早跟踪 terminal pitch；如果这个修复不是
对 nominal 过拟合，current15/current20/plant20/combined20 下也应保持高 success。

成立应看到：每个 stress level 的 K4 success `>= 38/40`，且相对 nominal 回退不超过
`2/40`；K4 不低于同场景 Z 超过 `2/40`。

失败应看到：任一 stress level K4 `<= 37/40`，或相对 Z 回退超过 `2/40`，或出现新的
timeout/safety/overshoot。

### H148-B：K4 不会用饱和换 success

机制：更高 pitch gain 可能让 stern 长期顶到 20 deg 上限。若 success 提升但饱和、
overshoot 或 timeout 显著增加，则不能作为训练基座。

成立应看到：timeout/safety 全 0，`path_overshoot_m_mean` 相对 nominal 不增加超过
`0.05 m`，stern 饱和没有在失败组形成新的系统性上限。

失败应看到：success 保持但 overshoot/timeout/safety 增加，或 stern 饱和贯穿全 episode。

### H148-C：K4 基座上 policy/zero 仍然同向

机制：Goal147 的 residual 只有 `1.5e-4`。如果 pressure 下仍完全同向，说明当前 PPO
没有利用 K4 提供的新权限；Stage 2 应先做最小训练 pilot，而不是直接接 reward。

成立应看到：policy/zero 的 success、plane miss、eta_done 差异都在 seed 噪声内。

失败应看到：pressure 下 policy 明显优于或劣于 zero；此时先审计 residual 方向与幅度，
再决定是否训练。

## 3. 冻结 arms 与 stress

Stage 1 固定：

- checkpoint：`C1_000300000.pt`；
- guard checkpoint：`C1_000400000.pt`，只在 nominal 与 combined20 上跑；
- training config：`configs/lowlevel/goal144/C-full-alos-command_ppo_pilot.json`；
- environment base：`configs/lowlevel/goal136/curriculum/C0.json`；
- stage `C1`，`--disable-curriculum-direct-command`；
- guidance：`configs/lowlevel/goal147/guidance_z.json`（K4 与 Z 相同）；
- controller：Z 用原 controller；K4 用 `configs/lowlevel/goal147/controller_k4.json`；
- dev seeds `20–59`，每 run 40 episodes；
- success frame `eta_tight_at_done`；
- **goal radius 固定 1.25 m**。不采用 Goal134 的 `default`（2.0 m），因为那会改变
  success 定义。

Stress 配置由 `scripts/build_goal148_k4_stress_profiles.py` 从 C0 生成，只改
`current` 与 `plant_randomization` 两个 section：

| level | current speed | shear gradient | plant randomization |
|---|---|---|---|
| `nominal` | `0.25–0.45 m/s` | `0.004 / -0.003` | off |
| `current15` | `0.375–0.675 m/s` | `0.006 / -0.0045` | off |
| `current20` | `0.50–0.90 m/s` | `0.008 / -0.006` | off |
| `plant20` | nominal | nominal | on，±20% |
| `combined20` | `0.50–0.90 m/s` | `0.008 / -0.006` | on，±20% |

`plant_randomization` 使用 Goal134 冻结配置：actuator time / added mass / drag /
fin / thrust 各 `[0.8, 1.2]`，seed `13420`。

矩阵：

| arm | checkpoint | levels | modes |
|---|---|---|---|
| `z` | 300k | nominal/current15/current20/plant20/combined20 | policy/zero |
| `k4` | 300k | nominal/current15/current20/plant20/combined20 | policy/zero |
| `k4_400k` | 400k | nominal/combined20 | policy/zero |

共 `2 × 5 × 2 + 1 × 2 × 2 = 24` runs，`960` episodes。

## 4. 执行计划

- [ ] Step 0：生成 stress configs 与 manifest，确认 goal radius 1.25、reward/termination 未变；
- [ ] Step 1：远端检查 GPU/进程与 checkpoint 哈希；
- [ ] Step 2：跑 24 个 eval-only runs，保存 summary 与 rich trace；
- [ ] Step 3：汇总 success、plane miss、eta_done、overshoot、timeout/safety、stern 饱和、policy/zero 差异；
- [ ] Step 4：按 §5 Gate 裁决；失败写 blocker/decision，不救 arm；
- [ ] Step 5：写 result、ledger、STATE，运行 checker；不自动 commit/push。

运行：

```bash
cd /data/yhx/underwater-robotics/src/plan_p1/goal136_curriculum_20261111
bash scripts/run_goal148_k4_stress.sh
```

分析：

```bash
PYTHONPATH=src /data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python \
  scripts/analyze_pc_goal148_k4_stress.py \
  --root artifacts/lowlevel/goal148_k4_terminal_pitch_stress_20261111
```

## 5. Gate

| check | 阈值 |
|---|---|
| nominal K4 | policy/zero 都 `40/40`，plane miss `0` |
| stress K4 success | 每个 level `>= 38/40`，相对 nominal 回退 `<= 2/40` |
| K4 vs Z | 每个 level K4 不低于 Z 超过 `2/40` |
| timeout/safety | 全 0；任何 NaN/Inf/flew-out 直接 FAIL |
| overshoot | `path_overshoot_m_mean` 相对 nominal 增加 `<= 0.05 m` |
| eta_done | 不出现比 nominal 更差的系统性回退；只做机制解释 |
| 400k guard | nominal 与 combined20 的 policy/zero 都保持 `40/40` |
| policy/zero | 记录差异；若 policy 在压力下稳定优于 zero，则作为 Stage 2 的正信号 |

裁决：

| 观察 | 判读 | 下一步 |
|---|---|---|
| 全部 stress 通过 | K4 是可用基座 | 写 Stage 2 预注册：K4-based PPO/SAC pilot vs K4-zero |
| current 通过、plant 失败 | plant mismatch 是下一瓶颈 | 先做 plant adaptation audit，不接 reward |
| combined 失败、单项通过 | 叠加压力超出现有权限 | 降级到单项 stress 或修 terminal capture，不扫 reward |
| nominal 都不稳 | K4 对 checkpoint/eval 过拟合 | 回滚 K4，写 blocker，重审 controller override |
| policy 在 stress 下稳定优于 zero | residual 开始有价值 | 先做 residual 方向/幅度审计，再训练 |
| policy 仍与 zero 相同 | residual 未利用 K4 权限 | Stage 2 做最小 PPO/SAC pilot，不接 reward |

## 6. 与 Goal140 的关系

本 Goal 仍不启动 Goal140 的 potential-based vertical reward。只有 Stage 2 训练后出现
“纵向误差持续、residual 方向正确但幅度/探索不足”时，才按 Goal140 §0 的常驻触发规则
写新的 reward pilot。K4 已证明当前 nominal 瓶颈是 controller，不是 reward。

## 7. Git

禁止 `git add -A`。只 add 本 Goal 的文档、stress 生成器、运行/分析脚本、测试和 ledger；
提交前运行 `git diff --cached --check`。不自动 commit/push。

## 8. Goal 提示词

```text
你现在执行 Goal148：K4 终端 pitch 控制器的 eval-only 压力测试。

先读 goal147 result、goal148 文档、Goal136 stress 设计和 Goal140 §0。固定
C1_000300000、C1 stage、dev seeds 20–59、goal radius 1.25；跑 z/k4 ×
nominal/current15/current20/plant20/combined20 × policy/zero，并补 k4 400k 在
nominal/combined20 的 guard。不训练、不改 reward/terminal/success/goal radius、
不扫 seed、不打开 held-out。

按 Gate 检查每个 stress 的 success、plane miss、eta_done、overshoot、timeout/safety、
stern 饱和和 policy/zero 差异。失败写 blocker/decision；只有 stress 通过后才预注册
K4 基座上的 PPO/SAC pilot，不接 reward。
```
