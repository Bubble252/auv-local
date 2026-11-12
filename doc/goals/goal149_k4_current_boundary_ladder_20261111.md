# Goal149：K4 的 current 可行边界阶梯

**日期**：2026-11-11<br>
**状态**：`PREREGISTERED / FROZEN_BEFORE_MEASUREMENT / EVAL_ONLY / HELDOUT_CLOSED`<br>
**上游**：`goal148_result_20261111.md`、`goal147_result_20261111.md`<br>
**优先级**：`P1`。Goal148 已证明 current15 以后执行器先饱和；现在要找 K4 的过渡带。
**性质**：固定 `C1_000300000` + K4 controller，只做 dev40 eval-only。不训练、不改 reward/terminal/success/goal radius、不扫 seed、不打开 held-out。

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
  check_command: .venv/bin/python ../scripts/check_research_ledger.py --goal Goal149 --goal-doc doc/goals/goal149_k4_current_boundary_ladder_20261111.md --strict --require-result
```

`required: false`：这是冻结的 eval-only boundary audit。记录前不改 levels、剂量、seed、
checkpoint 或 success 定义；失败后再进入 blocker/research loop。

## 1. 证据与问题

Goal148 canonical CUDA dev40：

```text
nominal:    K4 40/40, Z 26/40
plant20:    K4 39/40, Z 26/40
current15:  K4 15/40, Z  8/40
current20:  K4  7/40, Z  6/40
combined20: K4  7/40, Z  5/40
```

current15 时 rudder saturation `0.456`、rpm saturation `0.125`、ground speed >=2.5
占 `13.4%`；current20 时分别为 `0.514/0.145/35.7%`。heading rate clip 只从 `0.504`
升到 `0.549/0.520`，所以不是 heading rate limiter 单独造成。current20 的平均 crab
已达 `26.5 deg`，超过 18 deg 上限，说明 full-compensation 路径在工作但仍失败。

因此需要回答：

1. K4 从 `40/40` 到 `15/40` 的转折点在哪个 current 剂量？
2. 转折点是否与 rudder/rpm/ground-speed 饱和同步？
3. 是否存在一个 `70–90%` success 的区间，适合作为 residual 训练场景？

## 2. 可证伪假设

### H149-A：success 随 current 剂量单调下降，并存在 70–90% 过渡带

成立应看到：nominal `40/40`，在 current05/current10/current15 中至少一个 level 的
K4 success 落在 `28–36/40`，且 current15 仍接近 Goal148 的 `15/40`。

失败应看到：success 从 `40/40` 直接跳到 `<28/40`，没有中间带；此时需要更细的剂量
或承认边界很窄。

### H149-B：饱和指标先于 success 崩塌

成立应看到：在 70–90% success 的 level，rudder saturation 已明显高于 nominal
（例如 `>0.30`），或 ground speed >=2.5 的比例 `>2%`，但 timeout/safety 仍为 0。

失败应看到：success 已下降但饱和指标仍与 nominal 相同；此时更像 guidance/目标几何
问题，而不是执行器边界。

### H149-C：policy/zero 在过渡带仍同向

成立应看到：每个 level 的 policy/zero success 相同或差 1/40 以内，residual 仍约
`1.5e-4`。

失败应看到：policy 在过渡带稳定优于 zero，说明 residual 开始有价值；这应触发
residual 方向/幅度审计，而不是直接训练。

## 3. 冻结 levels

固定：

- checkpoint：`C1_000300000.pt`；
- controller：K4 `pitch_gain=4`；Z 原 controller；
- environment：`configs/lowlevel/goal148/env_nominal.json`（C1 base，无 plant randomization）；
- guidance：`configs/lowlevel/goal147/guidance_z.json`；
- stage `C1`，关闭 direct command；
- seeds `20–59`，40 episodes/run；
- success frame `eta_tight_at_done`，goal radius `1.25 m`；
- policy / zero-command-residual 配对。

Current 剂量通过 eval-only curriculum override 注入：

| level | current range | shear scale | 预期 mean current |
|---|---|---|---:|
| `nominal` | C1 原值 `0.20–0.35` | `0` | `0.276` |
| `current05` | `0.25–0.45 m/s` | `0.5` | `0.350` |
| `current07` | `0.275–0.50 m/s` | `0.75` | `0.388` |
| `current10` | `0.30–0.55 m/s` | `1.0` | `0.425` |
| `current12` | `0.34–0.60 m/s` | `1.25` | `0.470` |
| `current15` | `0.375–0.675 m/s` | `1.5` | `0.525` |

初始矩阵：`4 levels (nominal/current05/current10/current15) × 2 arms × 2 modes = 16 runs`。
如果 success 从 `40/40` 直接跳到 `<28/40`，按 §5 裁决表追加 `current07/current12`
两档细化，共再 `8 runs`。

## 4. 执行计划

- [ ] Step 0：确认 Goal148 canonical nominal K4 `40/40` 与 current15 `15/40` 可复现；
- [ ] Step 1：检查 GPU/进程，写 manifest 与 checkpoint 哈希；
- [ ] Step 2：跑 16 个 eval-only runs，保存 rich trace；
- [ ] Step 3：汇总 success、plane miss、eta_done、rudder/rpm/ground-speed 饱和、policy/zero 差异；
- [ ] Step 4：按 §5 Gate 找过渡带；不救 arm；
- [ ] Step 5：写 result、ledger、STATE，运行 checker；不自动 commit/push。

运行：

```bash
cd /data/yhx/underwater-robotics/src/plan_p1/goal136_curriculum_20261111
DEVICE=cuda GPUS="0 3 4 6 7" PARALLEL=8 \
  bash scripts/run_goal149_current_ladder.sh
```

分析：

```bash
PYTHONPATH=src /data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python \
  scripts/analyze_pc_goal149_current_ladder.py \
  --root artifacts/lowlevel/goal149_k4_current_ladder_20261111
```

## 5. Gate

主指标：K4 policy 的 success vs current 剂量。

| check | 判定 |
|---|---|
| nominal 复现 | K4 policy/zero 都 `40/40` |
| 单调性 | success 随剂量不上升超过 2/40；允许平台 |
| 过渡带 | 至少一个 level 的 K4 success 在 `28–36/40` |
| 配对 | K4 不低于 Z 超过 2/40 |
| 安全 | timeout/safety 全 0；任何 NaN/Inf/flew-out 直接 INVALID |
| 饱和 | 记录 rudder/rpm/ground-speed；过渡带的饱和应介于 nominal 与 current15 之间 |
| policy/zero | 记录差异；若 policy 稳定优于 zero，进入 residual 审计而不是直接训练 |

裁决：

| 观察 | 判读 | 下一步 |
|---|---|---|
| 找到 28–36/40 的 level | 有可用训练带 | 预注册该 level 的 K4-based PPO/SAC pilot vs K4-zero |
| 40/40 直接跳到 <28/40 | 边界很窄 | 在中间再加 current07/current12 两档 |
| success 下降但饱和不变 | guidance/几何问题 | 转 target/heading authority audit，不训练 |
| policy 在过渡带胜出 | residual 有价值 | 先审计 residual 方向/幅度，再训练 |
| 所有 level 都 <28/40 | current 已超包络 | 放弃 current 训练场景，转 plant/其他多样性 |

## 6. 与 Goal140 的关系

本 Goal 不接 Goal140 reward。即使找到过渡带，也先做 K4-based residual pilot；只有
训练后出现“纵向误差持续、residual 方向正确但幅度/探索不足”才按 Goal140 §0 触发
reward pilot。

## 7. Git

禁止 `git add -A`。只 add 本 Goal 的文档、运行/分析脚本、测试和 ledger；提交前运行
`git diff --cached --check`。不自动 commit/push。

## 8. Goal 提示词

```text
你现在执行 Goal149：K4 current 可行边界阶梯。

先读 goal148 result、goal149 文档和 Goal140 §0。固定 C1_000300000 + K4 controller，
跑 nominal/current05/current10/current15 × Z/K4 × policy/zero，dev seeds 20–59，
goal radius 1.25。只改 C1 curriculum current/shear，不训练、不改 reward/terminal/
success/goal radius、不扫 seed、不打开 held-out。

找 K4 success 从 40/40 到 15/40 的转折点，并同时记录 rudder/rpm/ground-speed 饱和。
只有找到 28–36/40 的过渡带，才预注册该 level 的 residual 训练；否则按裁决表继续
边界审计，不接 reward、不换算法。
```
