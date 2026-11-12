# Goal172：Goal86 残差方向与奖励接线的 eval-only 归因

**日期**：2026-11-12<br>
**状态**：`PREREGISTERED / EVAL_ONLY / FULL_MATRIX / DIAGNOSTIC_NOT_GATE / NO_TRAINING / HELDOUT_CLOSED`<br>
**上游**：Goal171 `SCREEN_NO_CANDIDATE / FORMAL_GATE_NOT_RUN`；Goal86 `T2_SAC` 弱正信号但 path 退化；Goal85/85b 位置契约与 plant mismatch 审计<br>
**性质**：只复用 Goal171 depthfix checkpoint 做反事实诊断，不训练、不改 reward/terminal/success/radius，不把结果写成 formal。

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
  check_command: .venv/bin/python ../scripts/check_research_ledger.py --goal Goal172 --goal-doc doc/goals/goal172_residual_attribution_20261112.md --strict --require-result
```

## 1. 要回答的问题

Goal171 的 200k screen 没有候选，但同时出现了两个互相竞争的机制解释：

- **方向/幅度问题**：residual 方向可能有用，只是训练出的物理幅度太小；
- **接线/参数化问题**：residual 可能在闭环里根本没有形成有效转向，或 reward 没有进入
  真正的优化信号。

这两者必须先用同一 checkpoint 的反事实评价分开，不能直接开新训练、换 SAC/DroQ 或
扩大预算。

## 2. 可证伪假设

### H172-A：方向存在，但幅度/探索不足

成立条件：同一 checkpoint 在 `2×` 或 `5×` residual scale 下，current10 的整条 3D route
RMSE 至少改善 `>=5%` 且绝对 `>=0.05 m`，并且没有出现执行器饱和、overspeed 或
timeout；改善不能只发生在末段。

失败条件：放大到 `2×/5×` 仍无路线响应；或只有 `10×` 才动且伴随饱和/安全退化。

最便宜证伪：不训练，只放大已训练 residual。

### H172-B：reward 配置没有真正进入优化信号

成立条件：同一 base、同一 seed、零 residual 下，R0 与 R1 的观测/轨迹逐 episode
完全相同，但 R1 的 reward breakdown 出现预期 cross/vertical/progress 项，且训练
manifest 记录的 reward 配置与运行时一致。

失败条件：零 residual 下 R0/R1 的 reward breakdown 也完全相同；或配置显示 R1，
但 trace 里没有对应 reward 项。此时先修接线，不训练。

### H172-C：动作通道权限不均衡

成立条件：在放大后的同一 checkpoint 上，rudder/stern-only 与 rpm-only 的 route
响应显著不同，且 cross-track 的改善主要由转向通道解释。

失败条件：所有通道都无响应，或 rpm-only 也产生同等路线改善。后者说明当前“转向瓶颈”
判断不成立。

### H172-D：SAC vs PPO 的差异是样本效率而不是接口

成立条件：在相同 checkpoint、相同 scale、相同 seeds 下，SAC 的 route 响应明显大于
PPO，且不是由更大执行器幅度或更高速度造成。

失败条件：两者都无响应；或 SAC 的响应完全由饱和/速度包线换取。此时不换算法。

## 3. 诊断矩阵

### 3.0 完整矩阵硬约束

本节不是“只看 current10 / default+strict125”的快速诊断。所有诊断 cell 都必须在
current levels `nominal/current07/current10/current15` 和 radii
`default=2.0/strict125=1.25` 上完整运行；零残差基线沿用 Goal171 同一 dev seeds
`20–59` 的完整矩阵。矩阵由 `scripts/goal172_matrix_contract.py` 冻结，启动前和
结束后分别由 `scripts/check_goal172_protocol.py` 审计；缺任一 cell 直接
`INVALID_INCOMPLETE_EVAL`，不得进入分析或写结论。

旧目录 `goal171_residual_attribution_20261112` 只保留为当前 current10 的取证，
不再作为 Goal172 结论依据。完整诊断写入
`artifacts/lowlevel/goal172_residual_attribution_full_20261112`。

### 3.1 residual scale amplification

固定 dev seeds `20–59`、四个 current levels 和两个 radius：

| arm | algorithm | baseline | 新增只读 scale |
|---|---|---:|---|
| `R1_BC_PPO` | PPO | effective `0.25` | `0.50 / 1.25 / 2.50` |
| `R1_ALOS_PPO` | PPO | effective `0.25` | `0.50 / 1.25 / 2.50` |
| `R1_BC_SAC` | SAC | multiplier `1×` | `2× / 5× / 10×` |
| `R1_ALOS_SAC` | SAC | multiplier `1×` | `2× / 5× / 10×` |

PPO 使用 `--actuator-residual-scale-override`；SAC 使用 `--residual-scale`。两者都
保持同一 checkpoint、同一 environment config、同一 seeds、同一 guidance。

### 3.2 channel attribution

在 effective scale `0.50`（configured `0.25` 的 `2×`）上补，并覆盖全部
current levels 与两个 radius：

```text
rudder-only: 1,0,0
stern-only:  0,1,0
rpm-only:    0,0,1
```

只用 PPO 的 `--actuator-residual-channel-gains`；SAC 若没有等价入口则先不做，
不能把不同接口的结果混在一起。

### 3.3 reward attribution

同一 frozen base、零 residual、全部 current levels、两种 radius：

```text
R0 old-route environment
R1 route-health environment
```

检查：

- 零 residual 的 trajectory/episode 终态是否一致；
- reward breakdown 是否出现 `reward_path_cross`、`reward_path_vertical`、
  `reward_path_progress` 等 R1 项；
- manifest 与环境配置的 reward 字段是否一致。

## 4. 记录与判定

指定产物根：

```text
artifacts/lowlevel/goal172_residual_attribution_full_20261112/
  screen/                 # scale amplification, all levels x radii
  screen_channel/         # channel attribution, all levels x radii
  screen_rudder/          # rudder-only scale sweep, all levels x radii
  reward_attribution/     # R0/R1 zero-residual traces, all levels x radii
  analysis/               # 汇总 JSON/MD
```

每个 run 必须写 `summary.json` 和 `trace.npz`；不完整 run 不进入分析。完整矩阵
共 `4 scale arms × 3 scales × 4 levels × 2 radii + 2 channel arms × 3 channels ×
4 levels × 2 radii + 1 rudder arm × 3 scales × 4 levels × 2 radii + 2 reward
environments × 4 levels × 2 radii = 184` 个诊断 run。
所有结果标记：

```text
DIAGNOSTIC_NOT_GATE
HELDOUT_CLOSED
NO_TRAINING
```

判定规则：

- `2×/5×` 单调改善且无饱和：支持 H172-A，下一步预注册更长训练或 residual warm-start；
- 放大也不动：支持接线/动作参数化问题，先查 reward/channel，不换算法；
- 只有 `10×` 动且饱和：方向可能有但不可用，先查角度约束和 terminal gate；
- R0/R1 零 residual 轨迹不一致：说明环境还有非 reward 差异，必须作为 blocker 处理；
- R0/R1 reward breakdown 相同：说明 reward 没接上，停止训练。

## 5. 不在本轮做的事

- 不训练新 checkpoint；
- 不打开 held-out；
- 不改 reward、termination、success、goal radius；
- 不扩大 seed 数来救单 seed 结果；
- 不把诊断结果写成 Gate PASS/FAIL；
- 不因为单次 scale sweep 就切换到 REDQ/DroQ。
