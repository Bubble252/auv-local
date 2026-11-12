# Goal147：C1 terminal pitch 跟踪权限——减速 vs pitch gain

**日期**：2026-11-11<br>
**状态**：`PREREGISTERED / FROZEN_BEFORE_MEASUREMENT / EVAL_ONLY / HELDOUT_CLOSED`<br>
**上游**：`goal146_result_20261111.md`、`goal140_vertical_tracking_reward_plan_20261111.md`<br>
**优先级**：`P1`。Goal146 已证伪 pitch/speed rate limiter；失败组在 2 m 时已经接近 stern 物理上限。
**性质**：固定 `C1_000300000`，只做 dev40 eval-only。不训练、不改 reward/terminal/success/goal radius、不扫 seed、不打开 held-out。

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
  check_command: .venv/bin/python ../scripts/check_research_ledger.py --goal Goal147 --goal-doc doc/goals/goal147_c1_terminal_pitch_authority_20261111.md --strict --require-result
```

`required: false`：这是冻结的 eval-only Gate。记录前不改 arms、阈值、seed、checkpoint
或 profile；失败后再进入 blocker/research loop。

## 1. 证据

Goal146 在 dev40 上得到：

| 指标 | Z | 结论 |
|---|---:|---|
| success | `26/40` | P1/P2/S1 也全是 `26/40` |
| near failures | `7` 个在 `1.25–1.5 m` | 四个 arm 全部 `0/7` 转化 |
| pitch clip all | `0.613` | P2 降到 `0.407`，但无 success 变化 |
| target-limited pitch gap <=3m | `0.034 rad` | P2 只降到 `0.014 rad` |
| failure `|target pitch|` @2m | `0.782 rad` | 实际 `|pitch|` 只有 `0.289 rad` |
| failure `|pitch error|` @2m | `0.423 rad` | 仍在追赶目标 |
| failure stern @2m | `0.317 rad` | 20 deg 上限 `0.349 rad`，接近饱和 |
| stern saturated <=3m | success `0.085` / failure `0.436` | P2 failure `0.545` |

结论不是“rate limit 不够快”，而是 pitch 目标到得太晚、pitch loop 跟不上，或者 terminal
速度太快导致没有足够时间在 stern 饱和前完成转向。

## 2. 可证伪假设

### H147-A：terminal slowdown 给 pitch loop 更多时间

机制：在 terminal 20 m 内用现有 `terminal_braking_enabled`，从巡航速度逐步降到
`terminal_speed_ratio`，使车辆在每个 goal-distance shell 停留更久，让 actual pitch 追上
target，减少 1.25–2.0 m 的纵向 miss。

成立应看到：

- `B85` 或 `B75` 至少把 Z 的 7 个 `1.25–1.5 m` 失败中的 2 个转成 success；
- failure `|pitch error|@2m` 和 final vertical 中位数下降；
- timeout、plane miss、overshoot 不增加超过 Gate。

失败应看到：success 仍 `26/40`，或者只有更长的 episode 而没有 geometry 改善。

### H147-B：pitch gain 是早期跟踪瓶颈

机制：当前 `pitch_gain=2`、`pitch_rate_gain=1`。失败组在 5 m 的实际 pitch 只有目标的
约 37%，到 2 m 才接近 stern 上限。提高 `pitch_gain` 或 `pitch_rate_gain` 可能让 stern
更早介入，从而增加 2 m 前的 pitch 跟进。

成立应看到：

- `K4` 或 `R2` 至少 2/7 近终点失败转化；
- failure `|pitch error|@2m` 下降；
- stern 饱和增加但不导致新的 plane miss/overshoot。

失败应看到：success 不动，或饱和增加但 terminal error 不变。此时不能继续加 gain。

### H147-C：policy 与 zero 的差异

每个 arm 同时跑 `policy` 和 `zero-command-residual`。若两者都改善，收益来自传统
guidance/controller；若只有 policy 改善，才是 residual 利用了新权限。Goal146 的
residual 约 `1.5e-4`，因此本 Goal 不应预设 policy 会更好。

## 3. 冻结 arms

| arm | guidance | controller | 目的 |
|---|---|---|---|
| `Z` | 原 C0 | `pitch_gain=2`, `pitch_rate_gain=1` | Goal146 基线 |
| `B85` | `terminal_braking_enabled=true`, `terminal_speed_ratio=0.85` | 原 controller | 温和减速 |
| `B75` | `terminal_braking_enabled=true`, `terminal_speed_ratio=0.75` | 原 controller | 减速上限（最小速度比 0.75） |
| `K4` | 原 C0 | `pitch_gain=4` | pitch 角度增益 2x |
| `R2` | 原 C0 | `pitch_rate_gain=2` | pitch rate 增益 2x |

不加入 `B+K` 组合；只有单变量出现正信号后才写新预注册组合臂。

配置由 `scripts/build_goal147_profiles.py` 从 C0 生成，除表中字段外逐字段相同；
哈希在 `configs/lowlevel/goal147/profile_manifest.json`。`--controller-config` 是
eval-only，只允许 `pitch_gain`、`pitch_rate_gain` 两个字段不同。

固定：

- checkpoint：`artifacts/lowlevel/goal144_alos_rl_20261111/ppo/C-full-alos-command/S_s86201_gatefix2/curriculum_checkpoints/C1_000300000.pt`；
- training config：`configs/lowlevel/goal144/C-full-alos-command_ppo_pilot.json`；
- environment config：`configs/lowlevel/goal136/curriculum/C0.json`；
- stage `C1`，`--disable-curriculum-direct-command`；
- dev seeds `20–59`，每 run 40 episodes；
- policy / `zero-command-residual` 配对；
- success frame `both`，主指标 `eta_tight_at_done`；
- `C1_000400000` 只在 300k 候选 arm 通过后做 guard，不参与选 checkpoint。

## 4. 执行计划

- [ ] Step 0：确认 profile 哈希、checkpoint 哈希、controller override contract；
- [ ] Step 1：远端 GPU/进程检查，写 manifest；
- [ ] Step 2：跑 `Z/B85/B75/K4/R2 × policy/zero`，dev40，保存 rich trace；
- [ ] Step 3：分析 success 转化、failure bins、final V/H、pitch error@2m、stern saturation、timeout；
- [ ] Step 4：按 §5 Gate 裁决；候选 arm 过 Gate 后才补 400k guard；
- [ ] Step 5：写 result、ledger、STATE，运行 checker；不自动 commit/push。

运行：

```bash
cd /data/yhx/underwater-robotics/src/plan_p1/goal136_curriculum_20261111
bash scripts/run_goal147_terminal_pitch_authority.sh
```

分析：

```bash
PYTHONPATH=src /data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python \
  scripts/analyze_pc_goal146_terminal_authority.py \
  --root artifacts/lowlevel/goal147_c1_terminal_pitch_authority_20261111 \
  --arms z,b85,b75,k4,r2 --control-arm z
```

## 5. Gate

主指标以 `policy` 对 `Z` 的 paired dev40 结果为准；`zero` 解释机制。

| check | 阈值 |
|---|---|
| 近终点转化 | 相对 Z，7 个 `1.25–1.5 m` 失败中至少 2 个转为 success |
| 总 success | 至少 `28/40`，且相对 Z 的 loss 不超过 1/40 |
| plane miss | 不恶化超过 2/40 |
| timeout/safety | 不增加；无 NaN/Inf/flew-out |
| 轨迹 guard | `path_overshoot_m_mean` 和 `eta_done_distance_m_mean` 不增加超过 0.05 m |
| 机制 | 候选 arm 的 failure `|pitch error|@2m` 和 final vertical 中位数下降 |

失败裁决：

| 观察 | 判读 | 下一步 |
|---|---|---|
| B 臂有效、K 臂无效 | 时间/捕获几何是瓶颈 | 训练 pilot 采用 terminal slowdown，不接 reward |
| K/R 臂有效、B 臂无效 | pitch 跟踪增益是瓶颈 | 训练 pilot 采用 controller gain，不接 reward |
| 两者都无效 | target 过陡或 terminal capture 本身不可达 | 新预注册 audit terminal target / capture geometry |
| 只有 zero 有效 | 传统控制链有效，residual 未利用 | 先修传统基座，不宣称神经网络收益 |
| 权限足够但 policy 不利用 | residual/优化问题 | 才进入 Goal140 reward 或最小训练 pilot |

## 6. 与 Goal140 的关系

Goal147 仍不启动 Goal140 的 potential-based reward。只有满足以下条件才接：

1. B 或 K/R 证明 terminal 权限/时序可以移动近终点几何；
2. 训练后的 policy 没有利用该能力，或 residual 方向正确但幅度/探索不足；
3. paired dev40 上 vertical error 与 failure 的关联稳定；
4. 不改变 success/terminal/goal radius/held-out/主算法族。

否则继续修 action parameterization / terminal target。

## 7. Git

禁止 `git add -A`。只 add 本 Goal 的文档、profile 生成器、运行/分析脚本、测试和 ledger；
提交前运行 `git diff --cached --check`。不自动 commit/push。

## 8. Goal 提示词

```text
你现在执行 Goal147：C1 terminal pitch 跟踪权限的 eval-only 单变量审计。

先读 goal147 文档、goal146 result 和 Goal140。固定 C1_000300000、dev seeds 20–59，
跑 Z/B85/B75/K4/R2 × policy/zero；不训练、不改 reward/terminal/success/goal radius、
不扫 seed、不打开 held-out。先检查 profile 哈希和 GPU，再跑 dev40，保存 trace。

按 Gate 比较 7 个近终点失败的转化、final V/H、pitch error@2m、stern saturation 和
policy/zero 差异。失败就写 blocker/ledger；只有权限被证明足够但 policy 不利用时才接
Goal140 reward，不直接扫 reward 权重。
```
