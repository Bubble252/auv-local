# Goal79：observable full-vector terminal bias predictor

**日期**：2026-11-10<br>
**状态**：`OBSERVABLE_FULL_VECTOR_PREDICTOR_NOT_SUPPORTED`<br>
**上游**：Goal78 `FULL_VECTOR_DEADBAND_SUPPORTED`（oracle）<br>
**结果**：`doc/goals/goal79_result_20261110.md`

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
  check_command: .venv/bin/python ../scripts/check_research_ledger.py --goal Goal79 --goal-doc doc/goals/goal79_observable_full_vector_bias_predictor_20261110.md --strict
```

## 1. 背景与证据

Goal78 证明：如果能看到完整 `p_nav-eta` 误差，使用
`p_corr = p_nav - max(||e|| - 0.25,0) * e/||e||` 可以把 tight false 从
`69→16`（default）和 `33→12`（strict125），同时 tight true retention
保持 `100%/100%`。

但 Goal78 使用真实 `eta`，是 oracle。部署时只有策略可见的 terminal 特征，
因此需要验证：`p_nav-eta` 是否可以从这些特征中稳定预测出来，且预测误差不会
在 correction 中放大成新的 false/true loss。

## 2. 假设与最便宜 falsification

### H79-A：observable full-vector predictor 能保留 true 并降低 false

只用 terminal 可见特征训练固定 ridge 回归，预测
`e = p_nav - eta` 在 goal 局部正交基下的三分量，再做与 Goal78 相同的
`0.25 m` vector deadband。

支持条件（测量前冻结）：default 与 strict125 的 leave-one-seed-out 预测都满足

1. tight false reduction `>=50%`；
2. tight true retention `>=95%`；
3. mismatch / schema `=0`，且 prediction 无 NaN/Inf。

最便宜 falsification：复用 Goal71 terminal snapshots，按 seed 做 LOOCV；
不训练 RL、不新增 rollout。

### H79-B：预测器不需要 `eta` 泄漏特征

机制：只用 `p_nav/goal` 的 goal-frame 几何、path error、base command、
residual/action 统计等可见量，不使用 `eta`、`p_nav-eta` 或它们的任何变换。

支持条件：特征白名单和训练代码审计均不包含 `eta` / `p_nav-eta`；否则记
`INVALID`。这是一个实现正确性假设，不是主指标 Gate。

## 3. 固定特征与模型

### 3.1 特征白名单（每个 terminal episode）

- `signed_goal_m`, `lateral_goal_m`, `vertical_goal_m`（由 `p_nav-goal` 在
  goal 局部基分解）；
- `goal_distance_m`；
- `path_error_3d_m`, `path_horizontal_error_m`, `path_vertical_error_m`；
- `base_command_l2`, `residual_l2`, `action_l2`, `residual_gate`；
- `steps`, `time_s`；
- `base_command` 的三分量、`residual` 的三分量、`action` 的三分量。

禁止输入：`eta_ned`、`p_nav_ned` 原值（目标坐标和 p_nav 几何可由允许量
重建，但不直接喂 6 个绝对坐标）、`p_nav-eta` 及任何 oracle 标签。

### 3.2 标签

```text
e = p_nav - eta
在 goal 局部基下分解为 e_signed, e_lateral, e_vertical
```

标签只在训练/验证时用于计算误差，不能作为推理输入。

### 3.3 固定模型

- 标准化特征（每一折只用训练 seed 拟合 mean/std）；
- 三个独立 ridge 回归，固定 `alpha=1.0`，有截距；
- 不搜模型族、不调 alpha、不做特征选择；
- 预测 `e_hat` 后，用与 Goal78 相同的固定 `b0=0.25 m` vector deadband。

## 4. 数据与隔离

| 项目 | 固定值 |
|---|---|
| candidate | Goal71 limiter-on `_dwell50_m025_cmdlimit` |
| checkpoints | frozen Goal59 `s240/s752/s880/s1008` + `zero` |
| scenarios | `default`, `strict125` |
| seeds | dev `20–59` |
| validation | leave-one-seed-out（同一 seed 的所有 checkpoint 一起留出） |
| held-out | `100–149` 关闭 |
| training | 只训练 ridge 权重，不训练 RL |
| new rollouts | 无 |

禁止把同一 seed 的不同 checkpoint 分到 train/validation 两侧。

## 5. 执行计划

- [x] Step 0：Run Contract、特征白名单、seed-grouped split 审计；
- [x] Step 1：实现 analyzer + 单元测试（特征禁用、LOOCV、deadband）；
- [x] Step 2：本地定向测试、Ruff；
- [x] Step 3：服务器并行 LOOCV，输出 JSON/MD；
- [x] Step 4：按 Gate 判 H79-A/H79-B；
- [x] Step 5：写 result、blocker/ledger、STATE、strict checker、只 add 相关文件。

## 6. Gate

| check | 阈值 |
|---|---:|
| H79-A tight false reduction | 两个 scenario 都 `>=50%` |
| H79-A tight true retention | 两个 scenario 都 `>=95%` |
| prediction finite | 100% |
| mismatch / schema | 0 |
| 特征白名单 | 不含 eta / p_nav-eta |

失败裁决：

| 观察 | 判读 | 下一步 |
|---|---|---|
| 两个 scenario 都过 | 可预注册闭环 learned correction arm | 先做训练吞吐基准，再闭环 |
| 方向对但未过 Gate | 预测误差或 correction 过强 | 回 terminal reference/reward，不扫 alpha |
| 预测器接近零/无效 | 现有 terminal 特征不可预测 bias | 停止 bias correction 线 |
| 特征或 split 泄漏 | `INVALID` | 修实现后重跑，不改 Gate |

## 7. 算力

- 只读 LOOCV，40 个 seed × 2 scenario，固定 ridge，CPU 足够；
- 默认 `--jobs 10`，每进程单线程 BLAS；若单折总耗时超过 `60 s`，按 seed
  分片；
- 0 GPU、不训练 RL、不新增 rollout、不打开 held-out；
- 只有 predictor 通过后才做训练型闭环 arm 的 `num_envs`/batch 吞吐基准；
- SAC/REDQ/DroQ 继续关闭，除非出现独立的 optimization/exploration 或
  sample-efficiency 证据。

## 8. Git

禁止 `git add -A`。只 add 本 Goal 相关文件，提交前运行：

```bash
.venv/bin/python ../scripts/check_research_ledger.py \
  --goal Goal79 \
  --goal-doc doc/goals/goal79_observable_full_vector_bias_predictor_20261110.md \
  --strict
git diff --cached --check
```

不自动 commit/push，除非用户明确要求。
