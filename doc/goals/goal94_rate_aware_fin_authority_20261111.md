# Goal94：rate-aware residual 与鳍面权限

**日期**：2026-11-11<br>
**状态**：`RESULT_RECORDED / PHASE1_PARTIAL / PHASE2_RATE_AWARE_FALSIFIED / NO_SAC`<br>
**上游**：Goal93 `ACTION_PARAMETERIZATION_CLIPPING`、Goal90 `PASS_RL_GAIN_WEAK`、Goal92 `FAIL_NO_UPDATE_SCHEDULE_EFFECT`<br>
**性质**：先做 config-only scale screen，再做 rate-aware/fin-gain 动作参数化；不换 PPO/SAC 主算法族。

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
  check_command: python ../scripts/check_research_ledger.py --goal Goal94 --goal-doc doc/goals/goal94_rate_aware_fin_authority_20261111.md --strict
```

Gate 记录前不得改阈值、seed 范围、scale/gain 集合、场景或 held-out。Phase 2 的实现修复
可以迭代，但一旦开始记录 Phase 2 screen，不得根据 dev 结果改确认参数。

## 1. 背景

Goal93 的 40-seed eval-only 结果：

- total residual scale=1.5 把 strict125 从 `17/40`（scale=1.0）提到 `21/40`，
  default `31/40`，clip 只从 `0.691` 增到 `0.734`；
- `C10` 与 `PG10_3` 在 6 个 scale、2 个场景上的 success/path 曲线几乎完全重合；
- gate-active 区域 rudder 的 effective/applied 中位数约为 `0`，鳍面截断 step
  约 `80%`；
- PPO 的鳍面物理权限只有 BC 的 `4–6%`，PPO 主要放大的是 RPM；
- 大残差不是 NoisyNet；`r_BC` 本身把 D3−ALOS 的完整差编码进去，RPM 单位主导 L2；
- terminal success margin 只有 `8–13 mm`，success 对壳层毫米级变化极敏感。

因此当前不是“PPO 更新次数不够”或“探索噪声不够”，而是：

```text
PPO 的转向/俯仰增量在速率限制处被截掉
总残差放大主要放大 BC，不是 PPO
```

Goal94 只修动作参数化，不换算法。

## 2. 假设

### H94-A：scale=1.5 的 eval-only 增益能在训练后保留

- 机制：Goal93 的 scale sweep 显示总残差放大有真实壳层收益；训练时直接使用
  scale=1.5 可能把同样的增益固化到策略。
- 成立应看到：3 seeds 的 strict125 均值 `>=21/40`，default 均值 `>=31/40`，
  path 3D RMSE 相对 scale=1.25 baseline 增加 `<=0.02 m`。
- 失败应看到：strict125 均值 `<=20/40`，或 path 增加 `>0.02 m`。
- 最便宜证伪：Phase 1 `S15`，3 seeds、1M、dev40。

### H94-B：rate-aware residual 能让 PPO 鳍面增量存活

- 机制：先把 BC residual 做速率限制，再把 PPO residual 加到剩余 headroom；
  或把 PPO 参数化为 rate residual。这样 PPO 不再和 BC 的大位置残差一起被
  rate limiter 截断。
- 成立应看到：gate-active `fin_clip_step_fraction <=0.5`，
  `min(survival_rudder, survival_stern) >=0.5`，PPO/BC fin L2 ratio `>=0.15`。
- 失败应看到：fin clip 仍 `>0.5`，或 PPO fin ratio `<0.10`。
- 最便宜证伪：Phase 2 `R15`/`R25`，1 seed screen，再 3 seeds。

### H94-C：只加 fin-specific gain 不足以绕过速率限制

- 机制：如果 rate limiter 是 binding constraint，单纯把 PPO 鳍面 gain 放大
  4–8 倍仍会在 `rate_limit` 处被截断，甚至增加饱和/振荡。
- 成立应看到：`F15` 的 fin clip 仍 `>0.5`，success 不高于 `S15`，或
  `fin_clip` 更高、path 更差。
- 失败应看到：`F15` 把 fin survival 提到 `>=0.5` 且 success 改善。
- 最便宜证伪：Phase 2 `F15`，1 seed screen。

### H94-D：rate-aware + fin gain 仍不改善时，才考虑 SAC

- 机制：如果 PPO 的鳍面增量已经存活（`fin survival>=0.5`、PPO/BC fin
  ratio `>=0.15`），方向也有成功/失败分离，但 success 仍不改善，才说明
  信用分配/样本效率是下一层瓶颈。
- 成立应看到：`RF15` 满足 survival/fin ratio 门槛，但 strict125 仍
  `<=20/40`，且方向统计有分离。
- 失败应看到：survival 仍低；此时继续修动作参数化，不开 SAC。
- 最便宜证伪：Phase 2 `RF15` + 3 seeds。

## 3. 固定契约

| 项目 | 固定值 |
|---|---|
| 基座 | Goal86 P-A residual BC，SHA `80024ef5798cfbc9701b974ab14b0eb5ce082daaea9c665978431978d59ff9dd` |
| 架构 | `z_t64`、path12、command3、current3、state encoder `[128,128]` |
| gate | `residual_terminal_gate_enabled=true`，`enter=10 m`，`full=3 m` |
| PPO | 1000 envs × 25 steps × 40 iterations = 1M |
| 初始化 | `initial_residual_output_std=0.03`，`init_noise_std=0.10` |
| 评价 | CPU/float64，dev seeds `20–59`，default+strict125，`success_frame=both` |
| held-out | `100–149` 关闭 |
| reward/terminal/success/goal radius | 不改 |
| 算法 | 先 PPO；SAC 只在 H94-D 条件下新开 Goal |

## 4. 实验矩阵

### Phase 1：config-only scale transfer（先跑）

| arm | 参数化 | total scale | PPO fin gain | rate-aware | seeds | 预算 |
|---|---|---:|---:|---|---:|---:|
| `S15` | 当前 absolute residual | 1.5 | 1.0 | 否 | 86201/02/03 | 1M |

评价：dev40、default+strict125、policy vs frozen-base 配对。

### Phase 2：rate-aware / fin authority（实现后 screen）

| arm | 参数化 | total scale | PPO fin gain | rate-aware | 预留 PPO 速率 | seeds | 预算 |
|---|---|---:|---:|---|---:|---:|---:|
| `R15_20` | BC 先 rate-limit，再加 PPO | 1.5 | 1.0 | 是 | 20% | 86201 | 1M |
| `R15_30` | BC 先 rate-limit，再加 PPO | 1.5 | 1.0 | 是 | 30% | 86201 | 1M |
| `F15` | 当前 + fin gain | 1.5 | 4.0 | 否 | — | 86201 | 1M |
| `RF15` | rate-aware + fin gain | 1.5 | 4.0 | 是 | 20% | 86201 | 1M |

晋级后才扩到 3 seeds；不因结果好坏给单个 arm 偷跑更多样本。

实现备注（记录于 Phase 2 开跑前）：非 rate-aware 分支最初没有把
`ppo_fin_gain`/`ppo_rpm_gain` 接到动作组合，导致 `F15` 会退化为 `S15`。
Phase 2 前已改为
`base + gate * residual * [fin_gain, fin_gain, rpm_gain]`，并补了
componentwise gain 与 zero-PPO parity 测试；测试结果 52 passed。该修复是
实现修正，不改变本 Goal 的 arm、阈值或评价协议。

### Phase 3：SAC 条件

只有 `RF15` 或后续 arm 同时满足：

1. gate-active `fin_clip_step_fraction <=0.5`；
2. `min(survival_rudder, survival_stern) >=0.5`；
3. PPO/BC fin L2 ratio `>=0.15`；
4. 方向统计有 success/failure 分离（bootstrap CI 下界 >0）；
5. success 仍 `<=20/40`（strict125）；

才新开 SAC 同协议对照 Goal。SAC 必须支持 frozen-BC + residual + terminal gate，
并先做 zero-residual parity；不能直接用 Goal86 T2-SAC 冒充同基座。

## 5. 指标

主指标：

- default / strict125 success；
- path 3D RMSE、pre-goal 3D RMSE；
- plane-miss、timeout、safety；
- `eta_min`、near-miss shell、annulus entry、conditional capture、radius sensitivity。

动作权限：

- `fin_clip_step_fraction`；
- `survival_rudder`、`survival_stern`；
- PPO/BC fin L2 ratio；
- PPO rpm / BC rpm；
- magnitude/rate clip rate、material saturation。

训练过程：

- deterministic PPO residual abs mean；
- `mean_noise_std` 初末值；
- actor 末层范数、approx KL、clip fraction；
- gate active fraction 与远处 `executed == base` 等式。

## 6. 预注册判据

### Phase 1 `S15`

晋级 Phase 2 需全部满足：

- 3-seed strict125 均值 `>=21/40`；
- 3-seed default 均值 `>=31/40`；
- path 3D RMSE 相对 Goal90 scale1.25 三 seed 均值增加 `<=0.02 m`；
- timeout 不增加；
- 无 NaN/Inf、无执行器 material saturation 系统性恶化。

如果 `S15` 失败，不直接开 SAC；先检查 scale 训练/评估一致性和 shell 指标。

### Phase 2 `R*`/`F*`

晋级 3 seeds 需全部满足：

- `fin_clip_step_fraction <=0.5`；
- `min(survival_rudder, survival_stern) >=0.5`；
- PPO/BC fin L2 ratio `>=0.15`；
- strict125 `>=21/40`（1-seed screen 不要求超过 S15，但不能低于 19/40）；
- default `>=31/40`；
- timeout `0`。

### Phase 3 SAC

严格按第 4 节五条；否则不开 SAC。

## 7. 执行计划

### Step 0：预注册与 checker

- [x] 写本 Goal94 文档；
- [x] 运行 checker；
- [x] 同步到远端。

### Step 1：Phase 1 `S15`

- [x] 创建 `configs/lowlevel/goal94_20261111/s15.json`；
- [x] 3 个 seed 在 3 张 4090 并行训练；
- [x] dev40 default+strict125 policy/frozen-base 评价；
- [x] 写 `doc/goals/goal94_s15_result_20261111.md`。

### Step 2：Phase 2 rate-aware / fin gain

- [x] 实现 `step_split`/rate-aware residual，zero-PPO parity 测试；
- [x] fin-specific gain 配置，并修复非 rate-aware 分支漏接线；
- [x] 1-seed screen，按判据决定不扩 3 seeds；
- [x] 写 `doc/goals/goal94_phase2_result_20261111.md`。

### Step 3：裁决

- [x] Phase 2 未通过，不扩 3 seeds；
- [x] rate-aware 未通过，不进入 5M formal；
- [x] PPO/BC fin ratio 在 rate-aware 下仍低；`F15` 虽到 0.15+，但 total fin clip 与 success 未过；不开 SAC；
- [x] H94-D 五条未全部满足，不写 SAC Goal；
- [x] 写 Phase 1/2 result；
- [x] 更新 `research/LEDGER.jsonl` 与 `research/STATE.md`。

## 8. 证据路径（预期）

- `configs/lowlevel/goal94_20261111/s15.json`
- `configs/lowlevel/goal94_20261111/r15_20.json`
- `configs/lowlevel/goal94_20261111/r15_30.json`
- `configs/lowlevel/goal94_20261111/f15.json`
- `configs/lowlevel/goal94_20261111/rf15.json`
- `artifacts/lowlevel/goal94_rate_aware_20261111/S15_s*/final.pt`
- `artifacts/lowlevel/goal94_rate_aware_20261111/eval_dev40/*/summary.json`
- `artifacts/lowlevel/goal94_rate_aware_20261111/traces/*/trace.npz`
