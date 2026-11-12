# Goal80：沿 goal tangent 前移终端参考点的单因素闭环对照

**日期**：2026-11-10<br>
**状态**：`GATE_COMPLETED_TERMINAL_REFERENCE_HURTS`<br>
**上游**：Goal27 `H3_SR_FAIL_STOP`、Goal31 `PATH_LOOKAHEAD_FAIL`、Goal64 `GEOMETRY_SUPPORTED`、Goal67 `TERMINAL_TRACKING_LAG`、Goal71 `TRACKING_IMPROVES_OUTCOME_UNCHANGED`、Goal79 `OBSERVABLE_FULL_VECTOR_PREDICTOR_NOT_SUPPORTED`<br>
**性质**：冻结的预注册 Gate；只做评估，不训练，不换算法。
**结果**：`doc/goals/goal80_result_20261110.md`。

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
  check_command: python ../scripts/check_research_ledger.py --goal Goal80 --goal-doc doc/goals/goal80_terminal_reference_extension_20261110.md --strict
```

`required: false` 表示这是一个冻结的预注册 Gate。Gate 记录前不得改候选值、阈值、
seed、eval split、checkpoint 选择规则或评估协议；失败后再按 AGENTS 补 blocker、
至少两个可证伪 hypothesis、reading note 和 ledger decision。

## 1. 背景与证据

### 1.1 当前主线已经排除的方向

- Goal27 已停止 command residual PPO、reward 权重、gamma/lambda、seed 搜索和
  H3-SR supervised residual；不能重开。
- Goal31 已证伪“沿参考路径前视点做终端纯追踪”：path RMSE 改善，但 terminal
  vertical offset 显著变差，terminal saturation 上升。因此本 Goal 不再使用
  `terminal_path_preserving_enabled`，也不扫 lookahead。
- Goal30 已证伪 terminal alignment window `20 m -> 30 m`；本 Goal 不扫 window。
- Goal64 证明 strict125 的丢失 true 全部是 `GOAL_PLANE_MISS`，lost 的
  perpendicular offset 中位数约 `1.065 m`，minimum `p_nav` 中位数约 `1.172 m`，
  只比 tight radius `1.00 m` 高约 `0.17 m`；lost/retained 的 residual L2 几乎
  相同。
- Goal67 的 Step 7 已完成 deterministic policy vs forced zero-residual 配对：

  | scenario | policy lost true | zero-residual lost true |
  |---|---:|---:|
  | default | `5/160` | `4/40` |
  | strict125 | `6/160` | `1/40` |

  zero-residual 仍会丢失 true；strict125 的 zero-residual success rate 甚至高于
  policy pooled 值。residual 不是失败的充分原因。
- Goal67 还显示 lost 的 command-goal cosine 约 `0.987`，但 actual-goal cosine
  为负，地速约 `2 m/s`；问题更像 terminal pass-by / tracking lag，而不是
  command 方向完全错。
- Goal71 启用 command rate limiter 后 tracking 明显改善，但 true retention
  没有净改善（default `true_success_delta=-2`，strict125 `+1`）。因此本 Goal
  的 baseline 固定为 Goal71 limiter-on，不再改 limiter。
- Goal79 的 observable full-vector predictor 未过 Gate（false 降幅
  `44.93%/39.39%`），停止 alpha/model/feature/deadband 调参。

### 1.2 NoisyNet 归因：不是当前主因

本轮复核了 Goal59 formal、Goal67 Step 7 和 Goal79 的 checkpoint 分布。结论是：
**当前“大残差”不能解释成 NoisyNet 采样主导。**

- 正式评估走 deterministic mean weights，`sample_noise=False`；训练日志里的
  `residual_abs_mean` 是 stochastic rollout，天然包含噪声，不能直接当作评估
  残差。
- Goal59 formal 的 8/8 checkpoint deterministic saturation 和 raw saturation
  都是 `1.0`；hidden/output sigma 仍在初始化附近（约 `0.300/0.100`），
  `noise/det` 中位数约 `0.019`。只有 s1008 的 `noise/det≈1.03`、
  `noise>det≈0.66`，但它的 mean 仍然饱和。
- Goal79 的剩余 false 中，低噪声的 s240/s752/s880 仍贡献多数；s1008 没有单独
  主导。zero-residual 仍保留 default `12/69`、strict125 `10/33` 的 tight false。
- 历史上真正由 NoisyNet 主导的是 Goal7：hidden sigma `0.5`、rho 继续漂移，
  deterministic eval 成功但 stochastic training rollout 失败，固定 sigma 后
  恢复。当前条件与 Goal7 不同。

因此本 Goal 不关闭/缩小 NoisyNet，也不重扫 residual scale。若未来要做训练期
NoisyNet 对照，必须新建预注册 arm，固定总 transitions、UTD、seed 和 checkpoint
规则。

### 1.3 为什么现在做 terminal reference

Goal64/67/69 的证据把剩余问题指向 terminal approach / termination geometry：
轨迹在横向仍有约 `0.7–1.2 m` 偏差时穿过 goal plane，plane-miss guard 先于
tight radius 触发。Goal71 证明 command feasibility 改善 tracking 但没有转化为
true retention。Goal79 证明可观测 bias correction 不够。剩下的单因素问题是：

> 在同一个 20 m terminal 分支、同一个 limiter、同一个 checkpoint 和同一批
> dev seeds 下，把 terminal reference 沿 goal tangent 前移 1 m，能否让
> pass-by 轨迹更早/更平滑地对齐终点切向，从而恢复 true retention？

这不是算法切换，也不是 residual/reward/gamma 搜索。

### 1.4 离线几何 screen

在正式 Gate 前，用 Goal67 的 rich trace 做了只读几何 screen（不重跑闭环）：
对 terminal phase、距 goal `<8 m` 的 lost 轨迹，比较 baseline 方向
`normalize(goal - p_nav)` 与候选方向
`normalize(goal + 1.0 m * goal_tangent - p_nav)` 的夹角。

| run | 子集 | 1 m 前移后的夹角 median | P90 |
|---|---|---:|---:|
| strict125 policy s752 | true | `3.44°` | `8.96°` |
| strict125 policy s752 | lost | `8.16°` | `27.78°` |
| default zero | true | `4.03°` | `12.19°` |
| default zero | lost | `7.95°` | `25.24°` |
| strict125 zero | true | `4.31°` | `15.86°` |
| strict125 zero | lost | `7.90°` | `28.56°` |

lost 轨迹在 `1–2 m` 距离内的夹角中位数约 `28–31°`，说明 `1.0 m` 不是弱扰动；
同时 true 轨迹的变化更小，具备一定的选择性。screen 只用于选候选值和检查
“扩展是否真的改变终端指令”，不作为 Gate 结果。

## 2. 可证伪假设

### H80-A：1 m goal-tangent 前移能转化为 true retention

在 Goal71 limiter-on 的 `original_lost = Goal60 true and not Goal71 true` 子集上：

```text
recovered_original_lost >= ceil(0.40 * original_lost)
newly_lost <= 1
true_success_delta = candidate_true - baseline_true >= +3 per scenario
false_success_delta <= +2 per scenario
```

同时 terminal perpendicular / vertical offset 不恶化超过 `0.05 m`，plane miss
不恶化超过 `+0.02`，timeout/safety 不恶化超过 `+0.02`，pre-goal RMSE 恶化不超过
`3%`。

成立时应看到：扩展确实改变 terminal command；lost 子集的 true 数增加，且不是
靠增加 false success 或牺牲 safety 换来的。

失败时应看到：true delta 不足，或 newly-lost 过多，或 terminal offset/plane miss
恶化。最便宜的证伪是下面的 seed 20/21 smoke 和 40-seed paired Gate。

### H80-B：扩展改变了几何/tracking，但没有转化为 true retention

支持条件：

```text
extension_effective == true
and H80-A == false
and (
  terminal horizontal offset 改善 >= 0.05 m
  or pre-goal RMSE 改善 >= 3%
  or heading/pitch tracking error 改善 >= 0.05 rad
)
```

这表示 terminal reference 方向是有效的控制旋钮，但 capture/termination 仍然
是瓶颈。此时停止扫 extension 值，转 terminal termination/plane-crossing
shaping 或控制器带宽；不换算法。

### H80-C：前移会重复 Goal31 的“path 改善但 terminal 变差”

支持条件：至少一个 scenario 的 terminal vertical offset 恶化 `>0.05 m`，或
plane miss 恶化 `>+0.02`，或 true_success_delta `<0`。

此时停止 goal-tangent 前移线，不扫 `0.5/2/4 m` 或 terminal window；记录负结果，
转 termination/plane-crossing 或动作参数化。

### H80-D：默认关闭和零扩展严格等价

支持条件：

- `terminal_reference_extension_m=0.0` 与 Goal71 limiter-on 在固定 seed、固定
  checkpoint、固定状态下逐元素一致；
- 非 terminal phase 的 heading/pitch/speed/reference position 完全不变；
- 只有 terminal phase 且 extension `>0` 时才计算新目标。

失败时直接 `INVALID_IMPLEMENTATION`，不得继续 Gate。

## 3. 单因素候选

### 3.1 唯一候选值

```text
terminal_reference_extension_m = 1.0
```

选择理由：Goal64 的 strict125 lost perpendicular offset 中位数约 `1.065 m`，
`1.0 m` 与 terminal 误差同量级；离线 screen 显示它在 lost 子集上产生约 `8°`
中位、`28°` P90 的视线角变化，而在 true 子集上约 `4°/12°`。本 Gate 只测这一个
值，不扫 `0.5/2/4 m`，也不把 1.0 m 解释成调参结果。

### 3.2 几何定义

只在 `robust_terminal_enabled` 的 terminal 分支内，且 extension `>0` 时：

```text
t = unit(path.goal_tangent_ned_for(N))
p_terminal_ref = goal + extension_m * t
terminal_direction = normalize(p_terminal_ref - position)
```

保持以下量不变：

- `capture` mask 仍由 `goal_distance <= alignment_distance` 决定；
- `terminal_ground_speed` 仍由原 goal distance/ratio 决定；
- terminal alignment window 仍为 `20 m`；
- goal radius、plane guard、dwell、reward、gamma、checkpoint、NoisyNet、
  controller gains 全部不变；
- `terminal_path_preserving_enabled` 必须为 `false`；若 extension `>0` 且该
  flag 为 true，配置直接报错；
- extension `=0` 时不执行任何新张量运算，保证与 baseline 逐位一致。

`p_terminal_ref` 与 position 重合时回退到原 `goal_direction`，避免除零。

### 3.3 配置 diff

candidate config 相对 Goal71 limiter-on 只允许：

```text
experiment.name
experiment.output
guidance.terminal_reference_extension_m: 0.0 -> 1.0
```

任何其他递归 diff 直接 `INVALID_IMPLEMENTATION`。

## 4. 数据、隔离与协议

| 项目 | 固定值 |
|---|---|
| checkpoint | Goal59 frozen `s240/s752/s880/s1008` |
| baseline | Goal71 limiter-on `_dwell50_m025_cmdlimit` |
| true-reference | Goal60 true reference，仅用于定义 `original_lost` |
| candidate | Goal80 extension-on `_dwell50_m025_cmdlimit_ext1p0` |
| scenarios | `default`, `strict125` |
| action modes | 4 policy + 1 zero-residual per scenario |
| seeds | dev `20–59`，40 seeds/run；合计 10 runs、400 episodes |
| eval | CPU/float64、`endpoint_stop`、dwell=50、`r_tight = r0 - 0.25 m` |
| default | `r0=2.00 m`，`r_tight=1.75 m`，guard `2.00 m` |
| strict125 | `r0=1.25 m`，`r_tight=1.00 m`，guard `1.25 m` |
| held-out | `100–149` 关闭 |
| training | 无；本 Goal 只加载 frozen checkpoint 做闭环评估 |

配对单位是同一 `scenario/checkpoint/action_mode/seed`。default 与 strict125
分开裁决；一个 scenario 的收益不能掩盖另一个的失败。

## 5. 指标与 Gate

### 5.1 结果指标

- `true = success and terminal_eta_capture`；
- `original_lost = Goal60_true and not Goal71_true`；
- `recovered_original_lost = original_lost and Goal80_true`；
- `newly_lost = Goal71_true and not Goal80_true`；
- `true_success_delta = Goal80_true_count - Goal71_true_count`；
- `false_success = success and not terminal_eta_capture`；
- success、eta capture、plane miss、timeout、safety；
- terminal `horizontal/vertical offset`、`signed_goal_offset`；
- pre-goal 3D RMSE、path 3D RMSE；
- material saturation、action saturation、residual L2 ratio；
- terminal-phase heading/pitch command rate P90 和 exceed fraction。

### 5.2 预注册 Gate

每个 scenario 必须同时满足：

| check | 阈值 | 说明 |
|---|---:|---|
| recovered original-lost | `>= ceil(0.40 * original_lost)` | default 9 -> `>=4`；strict125 7 -> `>=3` |
| newly lost | `<= 1` | 不能靠牺牲已捕获 seed 换取恢复 |
| true_success_delta | `>= +3` | 主效果 |
| false_success_delta | `<= +2` | 不能把 true 换成 false success |
| plane_miss_delta | `<= +0.02` | 全量 paired rate |
| timeout_delta | `<= +0.02` | 全量 paired rate |
| safety_delta | `<= +0.02` | 全量 paired rate |
| pre_goal_3d_rmse | 恶化 `<= 3%` | 同 seed 配对均值 |
| terminal horizontal offset | paired median delta `<= +0.05 m` | 不恶化 |
| terminal vertical offset | paired median delta `<= +0.05 m` | 不恶化 |
| 至少一个 offset 改善 | horizontal 或 vertical 改善 `>= 0.05 m` | 机制方向 |
| material saturation | paired median delta `<= +0.03` | 不能靠更饱和换捕获 |
| heading command rate | original-lost 子集 P90 `<= 0.75 rad/s` | limiter 仍有效 |
| heading exceed fraction | original-lost 子集 `<= 0.20` | limiter 仍有效 |
| extension effective | original-lost 子集 `<8 m` 视线角 median `>= 5°` 或 >=20/40 seeds command 改变 | 防止空扰动 |
| protocol integrity | 10 runs × 40 seeds、SHA、finite、CPU/float64 | 任一失败 -> INVALID |

两个 scenario 都通过才算 H80-A 支持。只通过一个记为
`MIXED_TERMINAL_REFERENCE`，按 strict125 主判定并保留另一个反例。

### 5.3 失败裁决

| 观察 | 判读 | 下一步 |
|---|---|---|
| 两 scenario 均过 Gate | `TERMINAL_REFERENCE_HELPS` | 保留 1 m 参考；下一步可在新预注册 arm 测 plane-crossing shaping 或组合 termination |
| tracking/offset 改善但 true 不动 | `GEOMETRY_MOVES_OUTCOME_UNCHANGED` | 停止 extension 扫参，转 termination/plane-crossing |
| vertical offset 或 plane miss 恶化 | `TERMINAL_REFERENCE_HURTS` | 停止 goal-tangent 前移，保留负结果 |
| true delta 为 0 且机制无变化 | `TERMINAL_REFERENCE_NO_EFFECT` | 检查 extension 是否真正进入 terminal command；不扫值 |
| timeout/safety 恶化 | `TERMINAL_REFERENCE_UNSAFE` | 不追认收益，先查安全 |
| extension `=0` 不等价或只改非 terminal | `INVALID_IMPLEMENTATION` | 只修实现/配置并重跑，不改阈值 |

失败后禁止：扫 extension 值、扫 terminal window、换 seed、加训练预算、改 reward/
gamma、打开 held-out、启动 SAC/REDQ/DroQ 救本 Gate。

## 6. 算法切换提醒

当前没有 optimization/exploration 或 sample-efficiency 证据，因此：

- 不启动 SAC 同协议对照；
- 不启动 REDQ/DroQ；
- 不因为 8×4090 空闲就换算法；
- 不把训练日志的 stochastic residual 当作评估残差。

如果未来出现明确证据，才按下面规则新开预注册 goal：

- 优化/探索失败：同 checkpoint、同 seeds、同评估协议，SAC 只作为对照 arm；
- 样本效率失败：固定总 transitions、UTD、replay 比例，再考虑 REDQ/DroQ；
- 残差大但闭环不动：先查动作参数化、reward 和 terminal capture，不换算法。

本 Goal 的闭环不是“残差不动的静止卡死”：lost 轨迹地速约 `2 m/s`，是 terminal
pass-by。因此按第 3 条先查 terminal capture，而不是换算法。

## 7. 算力计划

### 7.1 本 Goal 是 CPU/float64 评估

- 10 个 run（4 checkpoint × 2 scenario + 2 zero-residual），每个 run 40 envs；
- 正式运行在远端 384 CPU 服务器完成，`MAX_PARALLEL=10`，每进程
  `OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`；
- 远端隔离 venv 已具备 `numpy 2.2.6`、`torch 2.7.0+cpu`、`gymnasium 0.29.1`；
  10 个 run 全部完成，无失败或重试；
- 10 个 run 的自然 run 级并行度就是 10；本次从第一个 summary 到最后一个
  summary 的墙钟约 `3.79 s`，输出约 `166 MB`，0 张 GPU。继续把每个 run
  拆成 seed 分片需要先证明分片与整批逐 seed 逐位等价，本 Gate 不做事后改协议；
- GPU 对这类 CPU/float64 闭环评估没有收益，不占用 4090；
- 结果只在 seed 层合并。

### 7.2 为什么不在本 Goal 加 `num_envs`/batch

Goal80 是 frozen checkpoint 的评估协议，`num_envs=40` 是既有 Goal60/67/71
的随机流和 paired 对照的一部分；静默增加 envs 会改变轨迹和可比性。扩大采样/
批量只适用于训练型 arm，且必须另开预注册。

### 7.3 后续训练型 arm 的算力规则

若 Goal80 失败并进入训练期 plane-crossing shaping，先做单卡吞吐基准：

```text
num_envs = 128 / 256 / 512
batch_size = 256 / 512
固定 total_env_transitions、UTD、replay 比例、网络和 reward
```

记录 `env_transitions/s`、`gradient_steps/s`、GPU 利用率、显存峰值、CPU 负载。
选出稳定且吞吐最高的配置后，再 `8 seeds × 8 GPUs`，一卡一个 seed，
`CUDA_VISIBLE_DEVICES` 固定卡号，禁止多个 seed 共享同一进程的全局 RNG。
扩大采样/批量必须作为独立预注册 arm，不能混进正式 Gate。

## 8. 执行计划

- [x] Step 0：复核 Goal71 baseline、Goal60 true-reference、checkpoint SHA 和
  config diff；确认 held-out 关闭。
- [x] Step 1：实现 `GuidanceConfig.terminal_reference_extension_m`，加入
  dataclass、`__post_init__`、`_parse_guidance` optional 列表和 guidance
  terminal 分支；默认 `0.0`。
- [x] Step 2：加定向测试：默认关闭、零扩展逐位等价、非 terminal 不变、
  1 m 目标在 goal tangent 前方、NaN/负数/与 path-preserving 冲突被拒绝。
- [x] Step 3：生成 Goal80 default/strict125 config，递归 diff 只允许
  `experiment.name`、`experiment.output`、`guidance.terminal_reference_extension_m`。
- [x] Step 4：远端 seed 20 smoke，确认 extension 实际改变 terminal command，
  且 zero-extension 与 Goal71 逐元素一致。
- [x] Step 5：远端跑 10 个 run，`MAX_PARALLEL=10`、单线程 BLAS、rich trace。
- [x] Step 6：写 `scripts/analyze_pc_goal80_terminal_reference.py`，按预注册
  Gate 判 H80-A/B/C/D；输出 JSON/Markdown。
- [x] Step 7：写 `doc/goals/goal80_result_20261110.md`、必要 blocker、
  `research/LEDGER.jsonl`、`research/STATE.md`。
- [x] Step 8：运行 strict checker、定向 pytest、Ruff、diff check；只 add 相关文件。

### 8.1 运行命令（远端）

```bash
cd /data/yhx/underwater-robotics/src/plan_p1/goal80_terminal_reference_20261110/repo

PYTHONPATH=src:/data/yhx/underwater-robotics/src/plan_p1/settled_全局规划器/src \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
MAX_PARALLEL=10 \
OUT_ROOT=/data/yhx/underwater-robotics/src/plan_p1/goal80_terminal_reference_20261110/artifacts/lowlevel/goal80_terminal_reference_20261110 \
CHECKPOINT_ROOT=/data/yhx/underwater-robotics/src/plan_p1/goal59_bounded_terminal_residual_20261109/artifacts \
bash scripts/run_pc_goal80_terminal_reference_20261110.sh

OUT=/data/yhx/underwater-robotics/src/plan_p1/goal80_terminal_reference_20261110/artifacts/lowlevel/goal80_terminal_reference_20261110
.venv/bin/python scripts/analyze_pc_goal80_terminal_reference.py \
  --baseline-root /data/yhx/underwater-robotics/src/plan_p1/goal60_action_reward_terminal_readonly_audit_20261110/artifacts/goal71_command_rate_limit \
  --candidate-root "$OUT" \
  --reference-root /data/yhx/underwater-robotics/src/plan_p1/goal60_action_reward_terminal_readonly_audit_20261110/artifacts/traces \
  --equivalence-root /data/yhx/underwater-robotics/src/plan_p1/goal80_terminal_reference_20261110/smoke_compare \
  --jobs 10 \
  --output "$OUT/analysis/goal80.json" \
  --markdown "$OUT/analysis/goal80.md" \
  --equivalence-output "$OUT/analysis/equivalence.json"
```

### 8.2 验证与提交

```bash
.venv/bin/python ../scripts/check_research_ledger.py \
  --goal Goal80 \
  --goal-doc doc/goals/goal80_terminal_reference_extension_20261110.md \
  --strict
.venv/bin/python -m pytest tests/test_pc_goal80_terminal_reference.py -q
.venv/bin/python -m ruff check src/auv_lowlevel/config.py src/auv_lowlevel/batch/guidance.py \
  tests/test_pc_goal80_terminal_reference.py scripts/analyze_pc_goal80_terminal_reference.py
git diff --check
```

只 add 本次相关文件，例如：

```bash
git add src/auv_lowlevel/config.py src/auv_lowlevel/batch/guidance.py \
  tests/test_pc_goal80_terminal_reference.py \
  configs/lowlevel/pc_goal80_terminal_reference_default_20261110.json \
  configs/lowlevel/pc_goal80_terminal_reference_strict125_20261110.json \
  scripts/run_pc_goal80_terminal_reference_20261110.sh \
  scripts/analyze_pc_goal80_terminal_reference.py \
  doc/goals/goal80_terminal_reference_extension_20261110.md
git diff --cached --check
```

不自动 commit / push；若用户要求，按项目现有风格提交，例如：

```bash
git commit -m "test(goal80): preregister terminal tangent reference extension"
git push origin HEAD
```

## 9. 证据

- `doc/goals/goal27_result_20261109.md`
- `doc/goals/goal27_h3_sr_result_20261109.md`
- `doc/goals/goal31_result_20261109.md`
- `doc/goals/goal64_result_20261110.md`
- `doc/goals/goal67_result_20261110.md`
- `doc/goals/goal69_result_20261110.md`
- `doc/goals/goal71_result_20261110.md`
- `doc/goals/goal79_result_20261110.md`
- `doc/goals/goal80_result_20261110.md`
- `research/decisions/2026-11-10-noisenet-residual-attribution.md`
- `research/decisions/2026-11-10-compute-utilization-protocol.md`
- `research/STATE.md`
- `src/auv_lowlevel/batch/guidance.py`
- `src/auv_lowlevel/config.py`
- `scripts/run_pc_goal71_command_rate_limit_20261110.sh`
- `scripts/analyze_pc_goal71_command_rate_limit.py`
