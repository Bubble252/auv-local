# Goal94 Phase 2 结果：rate-aware 让 PPO 存活，但没有放大 PPO 权限；不满足 SAC 条件

**日期**：2026-11-11<br>
**状态**：`PHASE2_RECORDED / RATE_AWARE_FALSIFIED / FIN_GAIN_INSUFFICIENT / NO_SAC / NETWORK_PARITY_OK`<br>
**上游**：Goal94 Phase 1 `S15`、Goal93 `ACTION_PARAMETERIZATION_CLIPPING`、Goal89 网络结构对照<br>
**性质**：1 seed screen × 1M；dev40；default+strict125；policy/frozen-base 配对；trace 逐通道审计；未打开 held-out。

## 1. 结论

Phase 2 四个 arm 没有把 strict125 稳定推过 `21/40`，同时 rate-aware 的
default/path 变差：

| arm | 参数化 | default policy/frozen | strict125 policy/frozen | strict path | default path |
|---|---|---|---:|---:|---:|---:|
| `S15` | 当前 absolute residual，scale1.5 | 31/31 | 20–21/20 | 2.7168 | 2.7271 |
| `F15` | 当前 + PPO fin gain 4 | 31/31 | 21/20 | 2.7181 | 2.7283 |
| `R15_20` | rate-aware，20% PPO headroom | 30/30 | 20/20 | 2.7754 | 2.7861 |
| `R15_30` | rate-aware，30% PPO headroom | 29/29 | 21/18 | 2.8172 | 2.8264 |
| `RF15` | rate-aware 20% + fin gain 4 | 30/30 | 21/20 | 2.7751 | 2.7855 |

判读：

```text
H94-B = FALSIFIED
H94-C = SUPPORTED
H94-D = NOT_MET
decision = NO_SAC
```

## 2. 逐 arm 结果

### 2.1 `F15`：fin gain 提高了 PPO 比例，但没有提高 success

- default `31/40`，strict125 `21/40`，frozen-base strict `20/40`；
- PPO/BC fin L2 ratio 约 `0.19–0.21`，高于预注册 `0.15`；
- PPO-specific fin clip 约 `0.00–0.0012`，PPO 自身基本不被速率限制；
- 但总 fin clip（BC+PPO，`effective/applied` 任一通道差 >1e-6）约 `0.53–0.54`，
  仍高于预注册 `<=0.5`；
- 相对 `S15` strict 没有增加（21 vs 20–21），default 没有变化，path 只差
  `+0.0013 m`。

解释：`ppo_fin_gain=4` 确实把归一化 PPO 鳍面增量从 BC 的约 `4–6%` 提到
`19–21%`，说明 PPO 输出不是接错或完全零梯度；但闭环 success 没有跟着动，
说明“把 PPO 比例放大”本身不足以产生有效终端策略。`F15` 也证明
`ppo_fin_gain` 的非 rate-aware 分支必须显式接线；本次已在
`BatchTorchPCFrozenBCResidualVecEnv.step()` 中按
`base + gate * residual * [fin_gain, fin_gain, rpm_gain]` 实现，并补了
componentwise gain / zero-PPO parity 测试。

### 2.2 `R15_20` / `R15_30`：post-clamp survival 到 1.0，但 raw rate clip 仍有 27–52%

| arm | post-clamp PPO survival rudder/stern | raw PPO rate clip fin | PPO/BC fin L2 | BC fin clip | total fin clip |
|---|---:|---:|---:|---:|---:|
| `R15_20` default | 1.0 / 1.0 | 0.482 | 0.037 | 0.767 | 0.767 |
| `R15_20` strict | 1.0 / 1.0 | 0.516 | 0.037 | 0.781 | 0.781 |
| `R15_30` default | 1.0 / 1.0 | 0.266 | 0.048 | 0.855 | 0.855 |
| `R15_30` strict | 1.0 / 1.0 | 0.279 | 0.050 | 0.846 | 0.846 |
| `RF15` default | 1.0 / 1.0 | 0.876 | 0.050 | 0.760 | 0.760 |
| `RF15` strict | 1.0 / 1.0 | 0.886 | 0.052 | 0.780 | 0.780 |

这里必须区分两个口径：trace 里的 `actuator_residual_ppo_applied` 已经是
rate clamp 之后的增量，所以 `ppo_effective/applied` 的 post-clamp survival
必然是 `1.0`；这不是“PPO 没有被截断”的证据。用 raw PPO 增量
`gate * action * gain * 1.5 * full_scale` 与 `ppo_applied` 比较，
raw fin rate clip 实际为：

- `R15_20`：`0.482/0.516`；
- `R15_30`：`0.266/0.279`；
- `RF15`：`0.876/0.886`。

也就是说 rate-aware 仍然截掉了 27–89% 的 raw PPO 鳍面增量；同时 PPO 的
归一化输出本身只有约 `0.016–0.020`，BC 约 `0.25–0.26` 物理鳍面 L2，
比例仍只有 `3.7–5.2%`，远低于 `15%`。因此 rate-aware 既没有解决 raw
rate clipping，也没有解决 PPO 输出幅度/方向不足。

同时，给 PPO 预留 20–30% 速率 headroom 会压缩 BC 的可用速率：

- `R15_20` path 比 `S15` 差约 `+0.059 m`；
- `R15_30` path 差约 `+0.099–0.100 m`；
- default success 从 `31` 降到 `30/29`。

这与预注册的 path 阈值 `<=0.02 m` 不符，`H94-B` 判 `FALSIFIED`。

### 2.3 `RF15`：rate-aware + gain 仍没有超过 `F15`

`RF15` 的 PPO/BC fin ratio 约 `0.050–0.052`，只略高于 `R15_20`；`ppo_fin_gain=4`
被 `0.2 * max_fin_step` 的 rate clamp 截住，最大 PPO 鳍面增量仍约
`0.0105 rad/step`，无法把比例推到 `0.15`。success 仍是 default `30`、
strict `21`，path 比 `S15` 差约 `0.058 m`。

## 3. 方向统计

`cos(r_PPO, r_BC)` 的 episode 中位数：

| arm | scenario | success cos | failure cos | success−failure [95% CI] |
|---|---|---:|---:|---:|
| `F15` | default | -0.409 | +0.804 | -1.213 [-1.281, -0.658] |
| `F15` | strict125 | -0.409 | +0.291 | -0.700 [-1.071, -0.085] |
| `R15_20` | default | -0.290 | +0.795 | -1.085 [-1.306, -0.342] |
| `R15_20` | strict125 | -0.285 | +0.378 | -0.663 [-1.090, -0.096] |
| `R15_30` | default | -0.271 | +0.509 | -0.779 [-1.252, -0.400] |
| `R15_30` | strict125 | -0.264 | +0.319 | -0.582 [-0.931, +0.024] |
| `RF15` | default | -0.322 | +0.594 | -0.916 [-1.314, -0.272] |
| `RF15` | strict125 | -0.399 | +0.214 | -0.613 [-1.159, -0.213] |

这说明 success/failure 在 cos 上有分离，但方向是**成功 episode 更负**：
PPO 在成功轨迹里更多抵消 BC，在失败轨迹里更跟随 BC。这不满足预注册的
“bootstrap CI 下界 > 0”的正向方向条件，也不能据此证明 PPO 方向正确。
Goal93 已经指出 `cos(r_PPO, r_BC)` 不是因果最优方向；这里只把它作为
“有分离但符号相反”的诊断记录。

## 4. 与历史网络结构对照

Goal89 已经完成逐张量 parity：

- `d_z_total=64`、`n_z_reserved=4`、path12、command3、current3；
- actor `[256,128]`、critic `[256,128]`、ELU；
- frozen BC 22 个 state entry 与 Goal86 P-A checkpoint `max abs diff=0.0`；
- 当前 Goal94 actor 没有改主干，只新增 rate-aware/gain 的环境侧动作组合。

Goal87 还对照了 Goal80/Goal9 的 `z_t` 主干：state encoder `[128,128]`、
`d_z_total=64`、DADE 关闭、actor `[256,128]` 均同构。历史结论是
`network_wiring_bug=false`、`checkpoint_mismatch=false`、`NoisyNet=false`。

本次 Phase 2 又提供了行为层反证：

- 如果网络接错或 PPO head 零梯度，`F15` 不可能把 PPO/BC fin ratio 从
  `4–6%` 提到 `19–21%`；
- 如果只是 actor 容量不足，`F15` 的 PPO fin 增量不会稳定出现在所有 seed；
- 真正的异常是：PPO 输出幅度和方向都不足以改变 success，而不是特征维度或
  隐藏层宽度错位。

因此本轮继续维持：

```text
network_parity = ok
network_capacity_not_primary = supported
action_parameterization_rate_limit = partially_fixed
remaining_bottleneck = residual magnitude / direction / credit assignment
```

## 5. 预注册裁决

### `H94-A`

`PARTIAL`：scale1.5 三 seed strict 均值 `20.67 < 21`，default `31`，
path `+0.0143 < 0.02`，timeout `0`。相对增益存在，但 Phase 1 阈值未过。

### `H94-B`

`FALSIFIED`：rate-aware 的 post-clamp survival 为 `1.0`，但 raw PPO rate
clip 仍有 `0.266–0.886`，PPO/BC fin ratio 只有 `0.037–0.052 < 0.15`，
且 path/default 变差超过阈值。

### `H94-C`

`SUPPORTED`：只加 fin gain 不能绕过总速率限制；`F15` 总 fin clip 仍
`0.53–0.54 > 0.5`，success 不高于 `S15`。

### `H94-D`

`NOT_MET`：不满足 SAC 五条：

1. 总 `fin_clip_step_fraction <=0.5`：`F15` 为 `0.53–0.54`，rate-aware 为
   `0.76–0.86`；
2. `min(survival) >=0.5`：满足；
3. PPO/BC fin L2 ratio `>=0.15`：只有 `F15` 满足（0.19–0.21），rate-aware 不满足；
4. 方向成功/失败分离且 bootstrap CI 下界 >0：不满足（符号为负，且部分 CI 跨 0）；
5. strict125 仍 `<=20/40`：`F15` 为 `21/40`，不满足。

结论：**不开 SAC**。当前证据不是“残差方向对但样本效率不足”，而是
“残差幅度/方向仍不足，且 rate-aware 有 BC 跟踪代价”。

## 6. 下一步

不换 PPO/SAC 主算法族，先开新的预注册 Goal95，方向是“给残差一个强先验”：

1. 用 Goal86 P-A 的同一批 D3−ALOS 标签，先做 residual 的监督预训练
   （`r_BC` 或 `D3−ALOS` 的 residual actor），再做同协议 PPO 有界微调；
2. 预训练阶段只改 residual actor 初始化，不改 reward/terminal/success/goal
   radius，不打开 held-out；
3. 评价必须同时报告 `success`、`eta_min`、near-miss shell、annulus entry、
   conditional capture、radius sensitivity，以及 PPO/BC fin ratio；
4. 只有监督残差 + PPO 微调后，仍出现“PPO fin ratio>=0.15、方向分离且
   success<=20/40”，才重新评估 SAC/REDQ/DroQ；
5. 不把本次 `F15` 的 `+1/40` 写成稳定收益，也不把 `R15_30` 的 policy−frozen
   `+3` 写成真实改进，因为它的 frozen-base 已经从 20 掉到 18。

## 7. 证据

- `configs/lowlevel/goal94_20261111/{s15,f15,r15_20,r15_30,rf15}.json`
- `artifacts/lowlevel/goal94_rate_aware_20261111/{F15,R15_20,R15_30,RF15}_s86201/final.pt`
- `artifacts/lowlevel/goal94_rate_aware_20261111/eval_dev40_phase2/*/summary.json`
- `artifacts/lowlevel/goal94_rate_aware_20261111/traces_phase2/*/trace.npz`
- `doc/goals/goal89_network_parity_and_gate_scope_audit_20261111.md`
- `doc/goals/goal87_network_history_parity_20261110.md`
