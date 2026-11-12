# Goal45：fixed calibration residual 与 dynamic residual 审计

**日期**：2026-11-09<br>
**状态**：`RESULT_COMPLETE / DYNAMIC_RESIDUAL_DOMINANT`<br>
**上游**：Goal44 `PERSISTENT_LATE_BIAS`，见
`doc/goals/goal44_result_20261109.md`。<br>
**目的**：只读判断 Goal44 的 late-window persistent error 是否由已知的 fixed
calibration reset residual 解释，还是主要由动态 OU/USBL/outage 分量组成。本 Goal
不 rollout、不加载 policy、不训练、不改 controller/config/seed、不打开 held-out。

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
  check_command: python ../scripts/check_research_ledger.py --goal Goal45 --goal-doc doc/goals/goal45_fixed_calibration_residual_audit_20261109.md --strict
```

Exploratory goal：若 fixed 和 dynamic 两部分都不能解释 late bias，必须进入
autoresearch loop，按“数据 → 标签 → 评测 → baseline → 架构 → 文献”排查，不得转成
controller 参数扫描。

## 1. 背景与证据

Goal44 证明 terminal exact bias 在 `[t_end - 5 s, t_end - 0.5 s]` 中已经持续存在：
45/45 个 discordant seeds 的 late projection 为正，late share median 为
`0.91-1.20`，最后 0.5 秒没有额外放大。

当前 sensor profile 是 `realistic_calibrated`：

```text
nav_bias_calibration_fraction = 0.80
reset event = -0.80 * nav_bias
fixed reset residual = nav_bias + reset_event = 0.20 * nav_bias
```

`nav_bias` 由 `SyntheticSensorSuite.reset(seeds)` 确定性采样，不需要 policy rollout。
本 Goal 用这个已知的 fixed residual 与 Goal44 late error 比较，并检查减去 fixed
residual 后的 dynamic remainder 是否仍 toward goal。

## 2. 固定输入与参数 provenance

只读输入：

```text
configs/lowlevel/pc_goal27_c1_command_residual_default_20261109.json
configs/lowlevel/pc_goal27_c1_command_residual_strict125_20261109.json
src/auv_lowlevel/batch/sensors.py
src/auv_lowlevel/config.py
artifacts/lowlevel/goal42_estimator_calibration_nav_age_audit_20261109/formal/
  {default,strict125}/{baseline,candidate}/trace.npz
artifacts/lowlevel/goal44_terminal_bias_persistence_audit_20261109/
  terminal_bias_persistence_diagnostic.json
  terminal_bias_persistence_per_seed.csv
```

冻结 SHA256：

```text
default env config   6f454e06d3bbc7e2eab8022459e0d20f515edf2288d58db4b3290534795d363f
strict125 env config 4c8e7c53c5e81464bf2fe035167f02323a1bcc2ffe5844fc2aa7b8ffcbf568e1
sensors.py           c1e6f985ae8bc18191e9baee58db59380bee6d8894e636670659338d2af32932
config.py            6ca32634a21fc4937195bac30f1ee9b10936a54a4dc8c53ea0fd2b5715bcc182
Goal44 diagnostic    258553e847bc97ec7532f5136ddbc5bda7e96479cff260224d0045a70c5b8ec9
Goal44 per-seed CSV  2e3eaefc8063e18b95496a2394a732dfc7d815a3cf082542ee3608f21ae3dc0f
```

Goal42 trace SHA256 与 Goal43/44 相同：

```text
default/baseline/trace.npz      fe199b283f248fd94b5c416a4ab6639b3bf43f3f6824672f21966daf2ceb52d8
default/candidate/trace.npz     dc52bed06ad8fdddc52a58bb0fb73dbf9cb4ba049bbcce919344ae6e622e0548
strict125/baseline/trace.npz    aef6238f0996e5843242cb8ef3ac82e3bd966e369e63dcd91a82008644195387
strict125/candidate/trace.npz   b471fe1bd08c1540e10926dbb15a4736997f3787a1f9668831dc42e95b369a02
```

固定 checkpoint SHA256：
`0851496b15ffebbc0a19bafa5bf40c0493c0d9a3f485267b75fde5f811dde5d9`。

- 数据：dev seeds `20-59`，四个 block 各 40 seeds；
- held-out `100-149` 不读取、不打开；
- 不产生 rollout；只允许 `SyntheticSensorSuite.reset(seeds)` 的参数采样，
  禁止调用 `prime` 或 `step`；
- 不加载 checkpoint、不训练、不改 policy/controller/config/termination；
- artifacts 不提交。

## 3. 定义

对每个 scenario config：

```text
suite = SyntheticSensorSuite(config, num_envs=40, device=cpu, dtype=float64)
suite.reset([20, 21, ..., 59])
nav_bias[seed] = suite.parameters.nav_bias[row]
fixed_residual[seed] = (1 - 0.80) * nav_bias[seed]
```

必须验证：

- profile 为 `realistic_calibrated`，sensor seed 为 `81300`，
  calibration fraction 为 `0.80`；
- default 和 strict125 的 per-seed `nav_bias` 完全一致；
- `nav_bias` 三轴相等、finite、非负，符合 sampler 的 `_vector` 语义；
- 不调用 `prime`/`step`，因此没有 policy/controller 轨迹生成。

late-window error 由 Goal42 trace 按 Goal44 定义重算：

```text
queue_age_s = min(nav_age_s over episode)
exact_error(t) = p_nav(t) - eta(t - queue_age_s)
late_error_vec = componentwise_median(exact_error in [t_end - 5 s, t_end - 0.5 s])
u_t = normalize(goal_ned_m[i_end] - eta[i_end, :3])
late_projection_m = dot(late_error_vec, u_t)
fixed_projection_m = dot(fixed_residual, u_t)
dynamic_residual_vec = late_error_vec - fixed_residual
dynamic_projection_m = dot(dynamic_residual_vec, u_t)
```

要求 late projection 与 Goal44 CSV 对齐到 `1e-4 m`。固定门槛为每个 late window
至少 100 rows；所有向量 finite；discordant terminal projection 为正。

## 4. 可证伪 hypotheses

### H-FIXED-RESIDUAL-DOMINANCE：known fixed residual 解释 persistent bias

每个 block 的 discordant seeds 中同时满足：

```text
fraction(dot(late_error_vec, fixed_residual) > 0) >= 0.80
median(||fixed_residual|| / ||late_error_vec||) >= 0.50
median(fixed_projection_m / late_projection_m) >= 0.50
```

- 成立：known `0.2 * nav_bias` reset residual 在方向、norm 和 goal projection 上都
  能解释主要 persistent bias；
- 失败：fixed calibration residual 不是主因，dynamic OU/USBL/outage 分量更强；
- 最便宜证伪：只比较 fixed residual 与 late error 的 norm/projection。

### H-DYNAMIC-RESIDUAL：减去 fixed 后仍 toward goal

每个 block 的 discordant seeds 中同时满足：

```text
fraction(dynamic_projection_m > 1e-6) >= 0.80
median(dynamic_projection_m) >= 0.10 m
```

- 成立：fixed residual 之外仍有实质 dynamic late bias；
- 失败：减去 known fixed 后没有稳定的 dynamic toward-goal 分量；
- 最便宜证伪：只算 `late_error - fixed_residual` 的 terminal-goal projection。

## 5. 总体裁决映射

| 条件 | 裁决 | 允许的下一步 |
|---|---|---|
| 任一输入/hash/seed/parameter/window/CSV 失败 | `INVALID_DIAGNOSTIC` | 修输入后只重跑一次；不解释机制 |
| 两个 H 都 `SUPPORTED` | `MIXED_FIXED_DYNAMIC_RESIDUAL` | 预注册只读 fixed+dynamic 分量分离；不训练 |
| 仅 H-FIXED-RESIDUAL-DOMINANCE `SUPPORTED` | `FIXED_CALIBRATION_RESIDUAL_SUPPORTED` | 预注册只读 calibration profile 审计；不训练 |
| 仅 H-DYNAMIC-RESIDUAL `SUPPORTED` | `DYNAMIC_RESIDUAL_DOMINANT` | 预注册只读 OU/USBL/outage 动态审计；不训练 |
| 至少一个 `PARTIAL` 且无 SUPPORTED | `PARTIAL_COMPONENT_EVIDENCE` | 只补数据/参数验证，不训练 |
| 两个 H 都 `FALSIFIED` | `NO_KNOWN_COMPONENT_SUPPORTED` | 停止单组件解释，回到 estimator 结构 |

任何裁决都不授权训练、controller 修复、参数扫描、新 seed 或 held-out。

## 6. 输出

```text
artifacts/lowlevel/goal45_fixed_calibration_residual_audit_20261109/
  fixed_calibration_residual_diagnostic.json
  fixed_calibration_residual_diagnostic.md
  fixed_calibration_residual_per_seed.csv
```

CSV 至少包含：

```text
scenario,mode,seed,success,goal_capture_eta,discordant,
nav_bias_n,total_fixed_norm_m,late_error_norm_m,fixed_norm_share,
late_projection_m,fixed_projection_m,fixed_projection_share,
dynamic_projection_m,fixed_aligned,dynamic_toward_goal
```

## 7. 执行计划

- [x] Step 0：冻结 config/sensor/code/Goal44 SHA、parameter provenance、两个 H 和映射；
- [x] Step 1：实现 `scripts/audit_pc_goal45_fixed_calibration_residual.py` 和
      focused tests；覆盖 parameter sampling、fixed/dynamic decomposition、verdict；
- [x] Step 2：记录 analyzer/test SHA256，再运行一次性只读 analyzer；输入错误立即停止；
      CSV schema 序列化失败后只补 `window_rows` header，修正后成功完成；
- [x] Step 3：核对 40/40 seeds、45 个 discordant、parameter equality、CSV 对齐、finite；
      160 rows、45 discordant，window 226 rows，default/strict nav_bias 完全一致；
- [x] Step 4：写 `doc/goals/goal45_result_20261109.md`，按冻结映射裁决；
      `H-FIXED-RESIDUAL-DOMINANCE=FALSIFIED`、`H-DYNAMIC-RESIDUAL=SUPPORTED`，
      总体为 `DYNAMIC_RESIDUAL_DOMINANT`；
- [x] Step 5：更新两个 `research/STATE.md`、追加 `research/LEDGER.jsonl`，写 decision；
- [x] Step 6：运行 focused/full pytest、Ruff、strict ledger checker、
      `git diff --cached --check`；
- [x] Step 7：只 add 本 Goal 相关文件；用户未明确要求时不 commit/push。

运行命令：

```bash
cd /home/bubble/水下机器人/settled_局部规划器_反应式

.venv/bin/python -m pytest -q tests/test_pc_goal45_fixed_calibration_residual.py
.venv/bin/ruff check scripts/audit_pc_goal45_fixed_calibration_residual.py \
  tests/test_pc_goal45_fixed_calibration_residual.py

.venv/bin/python scripts/audit_pc_goal45_fixed_calibration_residual.py \
  --formal-root artifacts/lowlevel/goal42_estimator_calibration_nav_age_audit_20261109/formal \
  --goal44-root artifacts/lowlevel/goal44_terminal_bias_persistence_audit_20261109 \
  --output artifacts/lowlevel/goal45_fixed_calibration_residual_audit_20261109
```

Step 2 前必须在本文记录 analyzer/test SHA256。看到结果后不得修改阈值、映射、
parameter provenance 或 late window。

```text
audit_pc_goal45_fixed_calibration_residual.py
  18abc841636833a26223be9b2f5cb178a33d662db57f5f5b3e48b33d148bd2a9
test_pc_goal45_fixed_calibration_residual.py
  3d3f9d69d7a6f471a27514f4dd26be481db7289bcba8bfdc8ccc08380e170abc
```

首次 analyzer 在完成 decomposition 后、写 per-seed CSV 时因 `window_rows` 未列入
CSV schema 而退出。该字段只做 trace 完整性诊断；修正仅将它加入 CSV header，不改
任何 projection、阈值、样本或裁决。修正前没有读取 JSON/MD 的 hypothesis 结果。

## 8. 停止条件

- 任一 input/hash 不匹配，profile/seed/calibration fraction 不符；
- default 与 strict125 的 per-seed `nav_bias` 不一致，或三轴不相等；
- 任一 block 不是 40 seeds，discordant `< 5`、non-discordant `< 10`；
- late window 少于 100 rows、Goal44 CSV 不对齐或 projection 非 finite；
- 需要 `prime`/`step`、policy 加载、replay 或 config 修改才能继续。

## 9. Git 与服务器边界

本 Goal 全部是本地 CPU/read-only 分析和参数采样，不使用服务器。禁止 `git add -A`；
提交前运行：

```bash
.venv/bin/python ../scripts/check_research_ledger.py \
  --goal Goal45 \
  --goal-doc doc/goals/goal45_fixed_calibration_residual_audit_20261109.md \
  --strict
git diff --cached --check
```

不提交 artifacts、trace、checkpoint、日志或数据集。只有用户明确要求才 commit/push。

## 10. Goal 提示词

```text
你现在执行 Goal45：fixed calibration residual 与 dynamic residual 审计。

先读：
- doc/goals/goal45_fixed_calibration_residual_audit_20261109.md
- doc/goals/goal44_result_20261109.md

硬约束：
1. 只用 Goal42 dev traces、Goal44 outputs 和 deterministic sensor parameter sampling；
2. 不 rollout、不加载 policy/checkpoint、不训练、不改 controller/config/termination；
3. 不打开 held-out 100-149，不扫参数；
4. 不调用 prime/step；只允许 reset 后读取 parameters；
5. fixed/dynamic 门槛和裁决映射不得在看到结果后修改。
```
