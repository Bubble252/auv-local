# Goal93：残差方向、动作权限与终端捕获审计

**日期**：2026-11-11<br>
**状态**：`RESULT_RECORDED / ACTION_PARAMETERIZATION_CLIPPING / EVAL_ONLY / NO_ALGORITHM_SWITCH`<br>
**上游**：Goal90 `PASS_RL_GAIN_WEAK`、Goal91 `NOISE_NOT_PRIMARY`、Goal92 `FAIL_NO_UPDATE_SCHEDULE_EFFECT`<br>
**性质**：只读/回放既有 checkpoint 的 eval-only 审计；不训练、不改 reward/terminal/success/goal radius/seed/held-out。

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
  check_command: python ../scripts/check_research_ledger.py --goal Goal93 --goal-doc doc/goals/goal93_residual_direction_and_authority_audit_20261111.md --strict
```

Gate 记录前不得改本文件的阈值、scale 集合、seed 范围、场景或动作模式。Goal93 不训练，
因此任何“调参”都必须作为新的 eval-only 预注册，不得在本 Goal 内静默追加。

## 1. 背景

Goal92 的 6 个 PPO 更新调度 arm 全部没有改变闭环结果：

- default 全部 `31/40`；
- strict125 除 `W0` 为 `19/40` 外全部 `20/40`；
- `C10/C20` 把 PPO 确定性残差放大约 `35%/29%`，但逐 episode gain/loss 为 `0/0`。

同时，Goal92 summary 里有两个之前没有作为主因分析的指标：

| 指标 | Goal92 各 arm |
|---|---:|
| `actuator_residual_magnitude_clip_rate` | 约 `0.689–0.694` |
| `actuator_residual_rate_clip_rate` | 约 `0.490–0.493` |

执行链是：

```text
requested = ALOS + scale * (r_BC + gate * r_PPO)
limited   = clamp(requested, actuator_envelope)
executed  = rate_limit(limited)
effective = executed - ALOS
```

在 `scale=1.25` 下，`r_BC` 不是“小残差”，而是把目标 D3 动作放大 `1.25` 倍；
`r_PPO` 只占 `r_BC` 的约 `6–9%`。如果包线/速率限制已经把 PPO 增量截掉，
那么“残差变大但闭环不动”就不是优化或样本效率问题，而是动作参数化问题。

Goal87 H87-O4 曾经在旧契约下看到 eval-only `2×` 残差放大把 strict125 从
`17/40` 提到 `21/40`，说明“方向可能有信号，但幅度/权限不匹配”。本 Goal 要在
Goal90/92 的 corrected-gate 契约下重新验证这件事，而不是直接换 SAC。

## 2. 本 Goal 要回答的问题

1. `r_PPO` 的 requested 增量有多少真正变成 effective actuator change？
2. `scale=1.25` 是否已经过冲；`scale=1.0` 是否 clip 更少且 success 不降？
3. `r_PPO` 在 gate-active 区域是否与 `r_BC` / 误差下降方向一致？
4. 成功与失败 episode 的残差方向、effective 增量、终端 capture 是否可分？
5. 如果方向对且 effective 增量存在，但闭环仍不改善，才考虑 SAC 同协议对照。

## 3. 可证伪假设

### H93-A：PPO 增量被包线/速率限制截断

- 机制：`scale=1.25` 下 `r_BC` 已接近或超过执行器包线，`r_PPO` 的额外请求
  大部分落在 `clamp` 或 rate limit 之外。
- 成立应看到：gate-active 区域 `effective/applied` L2 生存率中位数 `<0.5`，
  且 `C10` 的生存率不高于 `PG10_3`；`C10 - PG10_3` 的 effective residual 差
  远小于 requested residual 差。
- 失败应看到：`effective/applied` 中位数 `>=0.8`，且 C10 的 effective 增量
  明显大于 PG10_3。
- 最便宜证伪：policy trace 的 `actuator_residual_applied` vs
  `actuator_residual_effective`，40 seeds，default+strict125。

### H93-B：总残差 scale=1.25 过冲，scale=1.0 更合理

- 机制：`r_BC` 的目标是重建 D3 动作；`scale=1.25` 把它放大到 D3 之外，
  造成大量包线饱和。
- 成立应看到：scale=1.0 的 magnitude clip rate 比 1.25 至少低 `30` 个百分点，
  且 strict125 success 不低于 1.25（允许逐 episode 损失 1 个）。
- 失败应看到：scale=1.0 clip 仍高，或 strict125 success 比 1.25 低 `>=2`。
- 最便宜证伪：eval-only scale sweep `1.0/1.25/1.5/1.75/2.0/2.5`。

### H93-C：PPO 方向与 BC 一致，失败来自幅度/信用而不是方向

- 机制：`r_PPO` 在 gate-active 区域与 `r_BC` 正相关；成功 episode 的方向一致性
  高于失败 episode。
- 成立应看到：gate-active `cos(r_PPO, r_BC)` 中位数 `>=0.2`；success 与 failure
  的 cos 差 `>=0.1`，且 seed-cluster bootstrap 95% CI 不跨 0。
- 失败应看到：cos 中位数 `<0.2`，或 success/failure 差 CI 跨 0。
- 最便宜证伪：同一批 trace 的方向统计。

### H93-D：终端捕获仍是毫米级壳层敏感，闭环指标无法区分有效增量

- 机制：成功 episode 的终止余量只有毫米到厘米级，`goal_radius` 球壳附近的
  微小差异就决定 success/plane-miss。
- 成立应看到：success 终止余量中位数 `<0.05 m`；policy 与 frozen-base 的
  终止 eta 距离差 `<0.01 m`，且 success/plane-miss 翻转只发生在壳层附近。
- 失败应看到：终止余量分布明显更宽，或 policy/frozen-base 的终止距离差
  `>=0.05 m`。
- 最便宜证伪：同一批 trace 的 `goal_distance_eta_m`、`terminal_kind` 和
  `signed_goal_tangent_offset_eta_m`。

## 4. 固定契约

| 项目 | 固定值 |
|---|---|
| 基座 | Goal86 P-A residual BC，SHA `80024ef5798cfbc9701b974ab14b0eb5ce082daaea9c665978431978d59ff9dd` |
| 架构 | `z_t64`、path12、command3、current3、`c_hat=0`、state encoder `[128,128]` |
| 动作 | `ALOS + scale * (r_BC + gate_10_3 * r_PPO)` |
| gate | `residual_terminal_gate_enabled=true`，`enter=10 m`，`full=3 m` |
| PPO arms | `PG10_3`（Goal90 seed86201）、`C10`（Goal92 seed86201） |
| 评价 | CPU/float64，dev seeds `20–59`，每臂 40 episodes |
| 场景 | default（goal radius 2.0 m）、strict125（goal radius 1.25 m） |
| 成功语义 | 终止帧 `eta` 进入 goal radius；`p_nav` 仅诊断 |
| held-out | `100–149` 关闭 |
| 训练 | 无；不新增训练型 rollout |

## 5. 实验矩阵

### 5.1 A：总残差 scale sweep

每个 cell 为 40 dev episodes；`scale` 通过 eval-only
`--actuator-residual-scale-override` 设置，不改 checkpoint 或训练配置。

| arm | scale 1.0 | 1.25 | 1.5 | 1.75 | 2.0 | 2.5 |
|---|---:|---:|---:|---:|---:|---:|
| `PG10_3` default | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `PG10_3` strict125 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `C10` default | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `C10` strict125 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

共 `2 arms × 2 scenarios × 6 scales = 24` 个 eval run。

### 5.2 B：policy trace

| arm | scenario | scale | trace |
|---|---|---:|---|
| `PG10_3` | default | 1.0 | `trace.npz` |
| `PG10_3` | strict125 | 1.0 | `trace.npz` |
| `PG10_3` | default | 1.25 | `trace.npz` |
| `PG10_3` | strict125 | 1.25 | `trace.npz` |
| `C10` | default | 1.25 | `trace.npz` |
| `C10` | strict125 | 1.25 | `trace.npz` |

### 5.3 C：frozen-base trace

| 基座 | scenario | trace |
|---|---|---|
| frozen P-A BC | default | `trace.npz` |
| frozen P-A BC | strict125 | `trace.npz` |

### 5.4 并发与算力

- 远端 384 vCPU；eval 单进程单线程，`OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、
  `OPENBLAS_NUM_THREADS=1`；
- 24 个 scale sweep + 8 个 trace run 全部并行，预计数分钟；
- 不使用 GPU 训练；不占用 8 张 4090 做本 Goal。

## 6. 指标定义

### 6.1 闭环指标

- default / strict125 success（`eta_tight_at_done`）；
- plane-miss、timeout、safety；
- path 3D RMSE、pre-goal 3D RMSE；
- 终止帧 `eta` 距离、成功余量、radial closure（最后 2 s `d|goal-eta|/dt`）；
- `actuator_residual_magnitude_clip_rate`、`rate_clip_rate`、material saturation。

### 6.2 动作权限指标

对每个 gate-active step（`frozen_residual_gate > 0.05`）计算：

```text
applied   = actuator_residual_applied      # 请求的物理增量
effective = actuator_residual_effective    # 经包线/速率限制后的实际增量
survival_k = median(effective_k / applied_k)  # 仅在 |applied_k| > eps 的 step
fin_clip   = mean(any(|effective[:2] - applied[:2]| > tol))
loss_k     = ||applied_k - effective_k|| / max(||applied_k||, eps)
```

按 episode 聚合中位数，并分通道（rudder/stern/rpm）报告。`actuator_residual_applied`
的 L2 由 RPM 通道主导（约 `150+`），因此总 L2 生存率会掩盖鳍面截断；主判据必须
同时看 `survival_rudder`、`survival_stern` 和 `fin_clip_step_fraction`。这是
smoke 阶段发现的观测口径问题，不改变实验矩阵或 40-seed 协议。

### 6.3 方向指标

在 gate-active 区域计算：

```text
cos_ppo_bc = cos(r_PPO, r_BC)
ppo_bc_ratio = ||r_PPO|| / max(||r_BC||, eps)
```

并报告：

- `cos_ppo_bc` 的 episode 中位数、IQR；
- success 与 failure 的 `cos_ppo_bc` 差，seed-cluster bootstrap 95% CI；
- 最后 1 s terminal window 的 `cos_ppo_bc`；
- 逐通道符号一致率（只在 `|r_BC_k| > 1e-3` 的 step 统计）。

### 6.4 终端捕获指标

- `terminal_kind` 计数；
- done 帧 `goal_distance_eta_m`、`success_margin_m`；
- `signed_goal_tangent_offset_eta_m` 的 done 分布；
- policy vs frozen-base 的同 seed 终止距离差。

## 7. 预注册判据

以下判据在跑任何 eval 前固定，结果出来后不得移动。

### 7.1 `AUTHORITY_NOT_BINDING`

满足任一：

- scale=1.0 的 magnitude clip rate 比 1.25 低 `>=0.30`，且 strict125 success
  不低于 1.25（允许 -1 episode）；
- scale=1.25/1.5 相对 1.0 的 strict125 paired gain `<=0`。

含义：`scale=1.25` 过冲或动作权限不是瓶颈；优先修动作参数化，不开 SAC。

### 7.2 `AUTHORITY_BINDING_BENEFICIAL`

满足全部：

- scale=1.25 或 1.5 相对 1.0 的 strict125 paired gain `>=2`；
- default success 不低于 31/40；
- magnitude clip rate 不高于 1.25 baseline + `0.05`；
- timeout 不增加。

含义：更多总残差有正作用，但可能受包线限制；下一步优先做 gain/scale 训练型
预注册，而不是 SAC。

### 7.3 `PPO_EFFECTIVE_SURVIVAL_LOW`

满足任一：

- gate-active `fin_clip_step_fraction >= 0.5`；
- gate-active `min(survival_rudder, survival_stern) < 0.5`；
- `C10` 的 `min(survival_rudder, survival_stern)` 不高于 `PG10_3`，且
  `C10 - PG10_3` 的 PPO residual requested L2 差 `>=0.25×`；
- `C10` 与 `PG10_3` 的 `fin_clip_step_fraction` 差 `<0.01`，但 requested
  PPO residual 差 `>=0.25×`。

含义：PPO 增量被执行器截断；换 SAC 不会解决，必须先修动作参数化。

### 7.4 `DIRECTION_POSITIVE`

满足全部：

- gate-active `cos_ppo_bc` episode 中位数 `>=0.2`；
- success 与 failure 的 cos 差 `>=0.1`，seed-cluster bootstrap 95% CI 下界 `>0`；
- 或 terminal window 的 success-failure cos 差 `>=0.1` 且 CI 下界 `>0`。

含义：PPO 方向有信号；若同时 `PPO_EFFECTIVE_SURVIVAL` 不低且闭环不改善，
才支持“信用分配/样本效率不足”。

### 7.5 `TERMINAL_SHELL_SENSITIVE`

满足全部：

- success 终止余量中位数 `<0.05 m`；
- policy vs frozen-base 的 done `goal_distance_eta` 差中位数 `<0.01 m`；
- success/plane-miss 翻转 episode 的 done 距离都落在 goal radius `±0.05 m`。

含义：闭环 success 对壳层毫米级变化极敏感；优先改 terminal capture/评价，
不换算法。

## 8. 决策表

| 证据组合 | 决策 | 下一步 |
|---|---|---|
| `AUTHORITY_NOT_BINDING` 或 `PPO_EFFECTIVE_SURVIVAL_LOW` | `ACTION_PARAMETERIZATION_CLIPPING` | 新开 Goal94：预注册 scale/gain/clip-aware residual 训练；不开 SAC |
| `AUTHORITY_BINDING_BENEFICIAL` 且 `DIRECTION_POSITIVE` | `ACTION_AUTHORITY_GAIN` | 新开 Goal94：预注册更高 gain/scale 训练；不开 SAC |
| `DIRECTION_POSITIVE` 且 survival 不低，但 success 不改善 | `CREDIT_ASSIGNMENT_SAMPLE_INEFFICIENT` | 新开 SAC 同协议对照 Goal，先做 frozen-BC+residual+gate adapter 与 zero-residual parity |
| `DIRECTION_POSITIVE` 不成立 | `DIRECTION_WRONG` | 不换算法；查 reward 形状、terminal capture、动作参数化 |
| `TERMINAL_SHELL_SENSITIVE` 主导 | `TERMINAL_CAPTURE` | 不换算法；先做 terminal capture/评价审计 |

如果多个条件同时成立，优先级：

```text
ACTION_PARAMETERIZATION_CLIPPING
> TERMINAL_CAPTURE
> DIRECTION_WRONG
> CREDIT_ASSIGNMENT_SAMPLE_INEFFICIENT
```

## 9. 执行计划

### Step 0：状态与预注册

- [x] 读 `AGENTS.md`、`research/README.md`、`research/STATE.md`；
- [x] 写本 Goal93 文档；
- [ ] 运行 checker 确认 Research Contract 可解析；
- [ ] 同步到远端。

验证：

```bash
cd settled_局部规划器_反应式
/home/bubble/水下机器人/.venv/bin/python ../scripts/check_research_ledger.py \
  --goal Goal93 \
  --goal-doc doc/goals/goal93_residual_direction_and_authority_audit_20261111.md --strict
```

### Step 1：实现 eval matrix 与分析器

- [ ] `scripts/run_goal93_eval_matrix.sh`
- [ ] `scripts/analyze_pc_goal93_residual_direction.py`
- [ ] `tests/test_pc_goal93_residual_direction.py`
- [ ] 2-seed smoke，确认 trace 字段完整、scale override 生效、JSON 输出可读。

运行：

```bash
bash scripts/run_goal93_eval_matrix.sh
```

### Step 2：跑 40-seed scale sweep 与 trace

- [ ] 24 个 scale sweep eval；
- [ ] 6 个 policy trace；
- [ ] 2 个 frozen-base trace；
- [ ] 检查所有 run 的 `summary.json`、`trace.npz` 和 `success_frame=both`。

### Step 3：分析并裁决

- [ ] 运行分析器生成 `goal93.json` 与 `goal93.md`；
- [ ] 按第 7 节判据输出 `AUTHORITY_*`、`DIRECTION_*`、`TERMINAL_*`；
- [ ] 写 `doc/goals/goal93_result_20261111.md`；
- [ ] 在 `research/LEDGER.jsonl` 写 Goal93 entry，`decision` 只能是
  `continue / pivot / stop`；
- [ ] 更新 `research/STATE.md`。

### Step 4：Git 检查

- [ ] `git diff --check`；
- [ ] 只 `git add` 本 Goal 相关文件，禁止 `git add -A`；
- [ ] `git diff --cached --check`；
- [ ] 不提交 `artifacts/`、checkpoint、日志、`trace.npz`；
- [ ] 用户未明确要求时不 commit/push。

## 10. 证据路径（预期）

- `configs/lowlevel/goal90_20261111/pg10_3.json`
- `configs/lowlevel/goal92_20261111/c10.json`
- `artifacts/lowlevel/goal90_gate_scope_20261111/pg10_3_s86201/final.pt`
- `artifacts/lowlevel/goal92_update_schedule_20261111/c10_s86201/final.pt`
- `artifacts/lowlevel/goal93_direction_authority_20261111/scale_sweep/*/summary.json`
- `artifacts/lowlevel/goal93_direction_authority_20261111/traces/*/trace.npz`
- `artifacts/lowlevel/goal93_direction_authority_20261111/goal93.json`
- `artifacts/lowlevel/goal93_direction_authority_20261111/goal93.md`
