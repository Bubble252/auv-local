# Goal72：limiter 前后终端几何与残差负控的只读配对审计

**日期**：2026-11-10<br>
**状态**：`MEASURED_LIMITER_GEOMETRY_TRADEOFF`<br>
**上游**：Goal71 `TRACKING_IMPROVES_OUTCOME_UNCHANGED`<br>
**基线**：Goal60 true-reference / Goal67 limiter-off rich trace

**结果**：`doc/goals/goal72_result_20261110.md`

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
  check_command: .venv/bin/python ../scripts/check_research_ledger.py --goal Goal72 --goal-doc doc/goals/goal72_terminal_geometry_limiter_20261110.md --strict
```

## 1. 背景与研究问题

Goal71 已经证明：启用现有 guidance command rate limiter 后，command demand
和 command-side tracking error 明显下降，但 true retention 没有净增加。

当前需要区分两个机制：

1. limiter 改善了 tracking，却把终端轨迹推到了更差的几何位置；
2. tracking error 本身不是 terminal retention 的充分统计量，真正相关的是
   closing、speed/height coupling 或 capture 时序。

本 Goal 只读复用已有 rich trace，不新增 rollout、不训练、不改变任何控制律。

## 2. NoisyNet / 残差问题

历史上确实遇到过“残差很大、闭环不动”的阶段。Goal7 曾定位到 BC 初始化后的
NoisyNet 隐藏层探索噪声；Goal58/59 又正式检查过 residual saturation。

目前不能把 Goal71 的失败归因于 NoisyNet，原因是：

- 正式评估调用 `agent.act_batch(..., evaluate=True)`，`sample_noise=False`，
  使用的是 deterministic mean weights，而不是 NoisyNet 采样；
- Goal59 对同一批 frozen checkpoints 的 probe 显示，s240/s752/s880 的
  noise std 远小于 deterministic action，s1008 的 noise 比 deterministic
  action 更大，但它在 Goal71 default 中的 true 只从 `17` 变 `16`；
- Goal71 的 zero-residual 负控在 default 为 `19 -> 17`，strict125 为
  `11 -> 11`，去掉 residual 仍没有恢复 true retention；
- Goal69 中 lost 的 residual/base 比例不高于 retained。

因此本 Goal 仍要加入一个明确的残差负控，但问题表述不是“残差数值大不大”，
而是“残差幅度是否能区分 newly-lost 与 retained”。如果 residual 不能区分，
就不启动以 NoisyNet/SAC 为核心的算法切换。

## 3. 数据与隔离

只读复用：

| 数据 | 路径 | 用途 |
|---|---|---|
| Goal60 true-reference | `artifacts/traces` | `baseline_true` |
| Goal67 limiter-off | `artifacts/goal67_terminal_reachability` | off 端 rich trace |
| Goal71 limiter-on | `artifacts/goal71_command_rate_limit` | on 端 rich trace |
| Goal65 plane-miss reference | `artifacts/goal65_plane_miss_dwell` | limiter-off 等价性复核 |

固定协议：

| 项目 | 固定值 |
|---|---|
| scenarios | `default`, `strict125` |
| checkpoints | Goal59 frozen `s240/s752/s880/s1008` |
| action modes | `policy` + `zero-residual` |
| seeds | dev `20–59` |
| eval | CPU / float64、`endpoint_stop`、dwell=50 |
| radius | `r_tight = r0 - 0.25 m` |
| held-out | `100–149` 关闭 |
| training | 无 |
| new rollouts | 无 |

配对单位严格为同一 `scenario / checkpoint / action-mode / seed`，不允许跨 seed
或跨 scenario 配对。`default` 是主判定，`strict125` 是一致性复核；两者分开报告。

## 4. 可证伪假设

### H72-A：limiter 的 tracking 改善被终端几何恶化抵消

机制：limiter 降低了 heading/pitch command rate，command-side tracking 变好，
但同时改变接近轨迹，使 `newly_lost` 的终端位置、plane offset 或 crossing 几何
比 limiter-off 更差。

支持条件（测量前冻结）：

1. default 的 `newly_lost = off_true and not on_true` 至少 `5` 个；
2. 在 `newly_lost` 中，至少 `60%` 的 seed 在下列任一几何量上比 off 恶化
   `>= 0.10 m`：
   terminal `eta` distance、terminal `p_nav` distance、terminal
   `abs(signed_offset)`、terminal vertical offset、terminal horizontal offset、
   crossing 时 `eta`/`p_nav` distance；
3. 至少一个几何量的 `newly_lost` 配对差中位数 `>= +0.10 m`；
4. 同一个几何量相对 `both_true` 组的中位数差至少多恶化 `+0.05 m`；
5. 在 `original_lost` 子集上仍能看到 tracking 改善：heading improvement
   `>= 0.10 rad` 或 pitch improvement `>= 0.05 rad`。

证伪：`newly_lost` 的几何没有系统性恶化，或恶化幅度不大于 `both_true`。

### H72-B：tracking error 不是 terminal retention 的充分统计量

机制：limiter 让 heading/pitch tracking error 下降，但 terminal closing、
speed error 或终端几何没有同步改善，因此 true capture 不增加。

支持条件（测量前冻结）：

1. `original_lost` 子集上 heading improvement `>= 0.10 rad` 或 pitch
   improvement `>= 0.05 rad`；
2. 下列至少一项没有实质改善：
   - crossing 前 1 s closing speed P95 的 on-off 改善 `<= 0.05 m/s`；
   - terminal speed error 的 on-off 改善 `<= 0.05 m/s`；
   - terminal `eta` 与 `p_nav` distance 二者的中位改善都 `<= 0.05 m`；
3. `true_success_delta < +3`。

证伪：tracking、closing、terminal speed error 和终端几何同向改善，但 capture
仍不增加；此时下一步应检查 capture label 或未记录状态。

### H72-C：NoisyNet / residual 幅度不是当前失败的主导因素

机制：如果 residual 幅度真的主导，则 limiter-on 的 `newly_lost` 或 `lost`
应该比 retained 有更高的 residual 比例，并且强制 zero-residual 应显著恢复
true retention。

支持“不是主导”的条件（测量前冻结）：

- 不满足以下任一 residual 信号：
  - `newly_lost` 的 `residual_l2_ratio_gate_active` 中位数比 `both_true`
    高 `> 0.05`；
  - `newly_lost` 的 `residual_abs_mean` 中位数比 `both_true` 高 `> 0.02`；
- 或 residual 虽有信号，但 zero-residual 的 pooled true delta `< +3`。

判定为 `RESIDUAL_NOISE_DOMINANT` 必须同时满足“residual 有信号”且
“zero-residual 恢复 `>= +3` true”。单独相关不能判因果。

## 5. 指标与配对口径

### 5.1 outcome

- `true = success and terminal_eta_capture`；
- `original_lost = baseline_true and not off_true`；
- `recovered_original_lost = original_lost and on_true`；
- `newly_lost = off_true and not on_true`；
- `both_true = off_true and on_true`；
- `recovered_all = not off_true and on_true`；
- `newly_lost_all = off_true and not on_true`；
- `true_success_delta = recovered_all - newly_lost_all`；
- false-success = `success and not terminal_eta_capture`。

### 5.2 terminal geometry

逐 seed 从 `episodes.json` 读取：

- `terminal_eta_distance_m`；
- `terminal_p_nav_distance_m`；
- `terminal_signed_goal_offset_m`；
- `terminal_vertical_offset_m`；
- `terminal_horizontal_offset_m`；
- `terminal_kind`、`goal_plane_miss`、`timeout`、`safety`；
- `residual_abs_mean`、`residual_l2_ratio`、
  `residual_l2_ratio_gate_active`、`gate_active_rate`；
- `pre_goal_3d_rmse_m`。

所有距离/offset 的配对差定义为 `on - off`，正值表示 limiter-on 更差。
`abs(signed_offset)` 先取绝对值再配对。

### 5.3 crossing 与 pre-crossing 窗口

- crossing 定义为 trace 中第一个 `signed_offset >= 0` 的 step；
- 若整个 episode 没有 crossing，保留最后一行并记录 `crossed=false`；
- 只有 off/on 两侧都 crossing 时才把 `crossing_*` 字段加入配对几何统计；
- 主窗口为 crossing 前 `1 s`；`2 s` 与 `3 s` 仅作方向一致性检查；
- 窗内至少 `5` 行，否则用最后 `5` 行补齐，并记录实际窗口行数。

窗内计算：

- `eta` / `p_nav` 到 goal 的距离；
- closing speed 的 P95：`-d(eta_distance)/dt`；
- required closure：`(eta_distance_start - r_tight) / remaining_time`；
- closure deficit：required 减 demonstrated，正值表示不足；
- actual ground speed、command speed；
- terminal speed error：`abs(actual ground speed - command speed)`；
- heading/pitch absolute tracking error；
- command heading/pitch rate P90；
- actual yaw/pitch rate；
- actual-goal cosine、command-goal cosine；
- material saturation、fin rate at limit、reachability infeasible fraction。

### 5.4 残差负控

- `policy` 与 `zero-residual` 分开统计；
- `residual_l2_ratio_gate_active` 和 `residual_abs_mean` 按
  `scenario / checkpoint / seed / outcome group` 取中位数；
- 只用同一 checkpoint 的 paired 结果做比较，不跨 checkpoint 聚合 causal claim；
- NoisyNet 本身不重新采样；正式评估的 deterministic 性质由
  `agent.act_batch(..., evaluate=True)` 和 `sample_noise=False` 保证。

## 6. Gate（测量前冻结）

| 观察 | 判读 | 下一步 |
|---|---|---|
| 完整性或配置审计失败 | `INVALID_IMPLEMENTATION` | 只修实现/路径，不改阈值 |
| residual 有信号且 zero-residual 恢复 `>= +3` | `RESIDUAL_NOISE_DOMINANT` | 先审计 NoisyNet/动作参数化，不直接换 SAC |
| default H72-A 支持，strict125 同向 | `LIMITER_GEOMETRY_TRADEOFF` | 终端几何/动作参数化 |
| default H72-A 支持，strict125 反向 | `MIXED_TERMINAL_GEOMETRY` | 保留分 scenario 结论 |
| H72-B 支持且 H72-A 不支持 | `TRACKING_DECOUPLED_FROM_TERMINAL` | 检查 closing/speed-height/capture 口径 |
| tracking 和几何都改善但 true 不增 | `CAPTURE_LABEL_OR_STATE_UNRESOLVED` | 先审计 label/未记录状态 |
| 以上都不成立 | `TERMINAL_GEOMETRY_UNRESOLVED` | 进行下一次预注册，不换算法 |
| timeout/safety 恶化 | `TERMINAL_GEOMETRY_UNSAFE` | 停止收益追认，先查安全/终止 |

`timeout/safety 恶化`固定为 paired rate delta `> +0.02`；不使用事后选择的阈值。

算法切换门保持关闭：

- 没有 optimization/exploration 失败证据，不启动 SAC；
- 没有 sample-efficiency 失败证据，不启动 REDQ/DroQ；
- 不因为 8×4090 空闲而改变 CPU/float64 正式评估协议。

## 7. 执行计划

- [x] Step 0：实现 analyzer、schema 校验和 paired tests；
- [x] Step 1：本地定向测试、Ruff、配置/协议完整性 smoke；
- [x] Step 2：把 analyzer 上传服务器，`--jobs 10` 只读运行；
- [x] Step 3：拉回 JSON/MD，核对 SHA 和逐 scenario Gate；
- [x] Step 4：写 `goal72_result_20261110.md`、blocker（如触发）、LEDGER、STATE；
- [x] Step 5：strict checker、全量 pytest、Ruff、diff check；只 add 相关文件。

  - 全量测试：`.venv/bin/python -m pytest -q` -> `1312 passed in 74.00s`；
  - strict checker：`PASS (goal=Goal72, entries=1)`；
  - Ruff 与 `git diff --check` / `git diff --cached --check` 通过；
  - 之前 `.venv/bin/pytest -q` 的 `ModuleNotFoundError: scripts` 确认只是
    入口路径问题，不是 Goal72 代码问题。

## 8. 算力使用

- 服务器：8 × RTX 4090、384 CPU、约 503 GB RAM；
- 本 Goal 是 CPU/float64 只读分析，不占 GPU；
- 10 个 run × 40 seeds 可并行读取，使用 `--jobs 10`，每进程
  `OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`、
  `NUMEXPR_NUM_THREADS=1`；
- 不通过扩大 batch/num_envs 改变本 Goal 协议；训练型扩样必须另开预注册 arm。

## 9. Git

禁止 `git add -A`。只 add Goal72 相关文件。提交前运行：

```bash
.venv/bin/python ../scripts/check_research_ledger.py \
  --goal Goal72 \
  --goal-doc doc/goals/goal72_terminal_geometry_limiter_20261110.md \
  --strict
git diff --cached --check
```

不自动 commit/push，除非用户明确要求。

## 10. Goal 提示词

```text
你现在执行 Goal72：只读终端几何与残差负控审计。

先读：
- doc/goals/goal72_terminal_geometry_limiter_20261110.md
- doc/goals/goal71_result_20261110.md
- research/blockers/2026-11-10-goal71-command-rate-limit.md

硬约束：
1. 不改 reward、gamma、seed、checkpoint、controller、eval split；
2. 不训练、不新增 rollout、不打开 held-out 100–149；
3. 不因为“残差大”直接换 SAC/REDQ/DroQ；
4. 所有 paired 统计必须按 scenario/checkpoint/action-mode/seed 对齐；
5. 不把相关写成因果，除非 zero-residual 负控支持。

执行顺序：
1. 实现只读 analyzer 与 tests；
2. 本地 smoke 和 Ruff；
3. 服务器 --jobs 10 并行分析；
4. 写 result、ledger、STATE，跑 strict checker。
```
