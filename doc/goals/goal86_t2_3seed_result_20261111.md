# Goal86 T2-SAC 3-seed 结果：success 有弱正信号，但 path 显著退化，不能进 5M formal

**日期**：2026-11-11<br>
**预注册**：`doc/goals/goal86_bc_prior_traditional_residual_20261110.md`<br>
**上游**：Goal86 1M screen `T2_ACTUATOR_RESIDUAL = POSITIVE_BUT_NOT_SIGNIFICANT`<br>
**状态**：`T2_3SEED_POSITIVE_BUT_PATH_TRADEOFF / NO_FORMAL / HELDOUT_CLOSED`<br>
**性质**：只读复用已有 1M checkpoint 和 dev40 评估，不重训、不改 reward/terminal/success/action/seed。

## 1. 结论

T2-SAC 的 success 方向在 3 个训练 seed 上都是正的：

```text
default:   28.67/40 vs zero 27/40，平均 +1.67/40
strict125: 15.33/40 vs zero 14/40，平均 +1.33/40
```

但这不是可部署收益：

1. **success 不显著**：default pooled `+5/120`，95% CI `[-0.042,+0.125]`；
   strict pooled `+4/120`，95% CI `[-0.033,+0.100]`，两者都跨 0。
2. **path 显著退化**：pooled path RMSE `+0.277 m`，95% CI `[+0.190,+0.366]`；
   86202/86203 两个 seed 的 path CI 都不含 0。
3. **不是所有 seed 都能同时保住 success 和 path**：86201 path 仅 `+0.014 m`，
   但 success 只 `+1/40`；86202/86203 success 更好，却把 path 推坏约 `0.4 m`。
4. 动作接近饱和（`action_abs_max ≈ 0.998–1.000`），residual L2 约 `0.198–0.283`；
   这不是“残差太小”，而是“残差确实在动，但没有形成可迁移的净收益”。

因此按预注册条件，**不进入 5M formal**。

## 2. 3-seed 配对结果

所有比较使用同一 dev seeds 20–59，逐 seed 配对；括号为 95% bootstrap CI。

### 2.1 default

| train seed | policy success | zero success | Δ | W/L/T | Δ path RMSE | path CI95 | residual L2 |
|---:|---:|---:|---:|---:|---:|---|---:|
| 86201 | 28/40 | 27/40 | +1 | 4/3/33 | +0.014 m | `[-0.052,+0.086]` | 0.256 |
| 86202 | 29/40 | 27/40 | +2 | 5/3/32 | +0.420 m | `[+0.317,+0.522]` | 0.198 |
| 86203 | 29/40 | 27/40 | +2 | 6/4/30 | +0.396 m | `[+0.190,+0.619]` | 0.283 |
| mean | 28.67/40 | 27/40 | +1.67 | 15/10/95 | +0.277 m | `[+0.190,+0.366]` pooled | — |

### 2.2 strict125

| train seed | policy success | zero success | Δ | W/L/T | Δ path RMSE | path CI95 | residual L2 |
|---:|---:|---:|---:|---:|---:|---|---:|
| 86201 | 17/40 | 14/40 | +3 | 4/1/35 | +0.015 m | `[-0.053,+0.088]` | 0.256 |
| 86202 | 15/40 | 14/40 | +1 | 4/3/33 | +0.418 m | `[+0.318,+0.518]` | 0.199 |
| 86203 | 14/40 | 14/40 | 0 | 3/3/34 | +0.393 m | `[+0.188,+0.611]` | 0.283 |
| mean | 15.33/40 | 14/40 | +1.33 | 11/7/102 | +0.275 m | `[+0.190,+0.364]` pooled | — |

## 3. PPO 同协议对照

T2-PPO 的 3 个 seed 与 zero-residual 在 40 个 dev seeds 上全部打平：

| scenario | policy success | zero success | Δ | W/L/T | Δ path |
|---|---:|---:|---:|---:|---:|
| default | 27/40 | 27/40 | 0 | 0/0/40 | `-0.0000 m` |
| strict125 | 14/40 | 14/40 | 0 | 0/0/40 | `-0.0000 m` |

这说明 T2 的弱正信号来自 SAC 的实际残差，不是动作接口本身；PPO 仍是近零残差、
零翻转。

## 4. 假设裁决

| 假设 | 裁决 | 证据 |
|---|---|---|
| H86-T2-3S-A：T2 success 在 3 seeds 上方向一致且显著 | `PARTIAL / GATE_FALSIFIED` | 3/3 default 正、2/3 strict 正；但 pooled CI 跨 0 |
| H86-T2-3S-B：T2 path 没有显著退化 | `FALSIFIED` | pooled `+0.277 m`，CI `[+0.190,+0.366]`；两个 seed 单独显著 |
| H86-T2-3S-C：T2 的 success/path 交易是 seed 噪声，可用更多 seed 救 | `NOT_SUPPORTED_AS_GATE` | path 退化在两个 seed 上稳定且同向；不能靠扩 seed 把负结果改成 PASS |
| H86-T2-3S-D：PPO 与 SAC 对同一 actuator 接口响应不同 | `SUPPORTED` | PPO 全部 0/0/40；SAC 有弱 success 信号但 path 交易 |

## 5. 机制判读

- T2 不是“残差没动”：SAC residual L2 约 `0.20–0.28`，动作接近饱和；
- 也不是“残差太小”：86201 的 residual L2 与 86203 接近，但 success/path 交易不同；
- 更像 **terminal capture 与 path tracking 的冲突**：SAC 学到了一些能翻转 near-goal
  success 的残差，但这些残差在全路径上付出了 `0.4 m` 级 RMSE；
- 该冲突与 Goal127/129/130 的 residual authority/saturation、Goal128 的 reward/terminal
  credit、Goal131/132 的 terminal geometry 诊断方向一致；
- 不能把 T2 的弱 success 信号直接放大成 5M formal，也不能靠扫 scale 或 seed 救 Gate。

## 6. 产物

- `artifacts/lowlevel/goal86/screen/t2_sac_eval_default_dev40/episodes.json`
- `artifacts/lowlevel/goal86/screen/t2_sac_seed86202_eval_default_dev40/episodes.json`
- `artifacts/lowlevel/goal86/screen/t2_sac_seed86203_eval_default_dev40/episodes.json`
- `artifacts/lowlevel/goal86/screen/t2_sac_eval_strict125_dev40/episodes.json`
- `artifacts/lowlevel/goal86/screen/t2_sac_seed86202_eval_strict125_dev40/episodes.json`
- `artifacts/lowlevel/goal86/screen/t2_sac_seed86203_eval_strict125_dev40/episodes.json`
- `artifacts/lowlevel/goal86/screen/t2_ppo_*_dev40/summary.json`
- 3-seed 统计由本轮只读脚本从上述 episodes 计算；不新增训练产物。

## 7. 决策

`pivot`，但不换算法：

- 不进入 5M formal；
- 不继续扫 T2 residual scale 或 seed；
- 保留 T2 的弱正信号为“terminal capture 可被 residual 影响”的证据；
- 下一轮继续使用 Goal85 的 `unbiased_noise_v1`、`position_source=eta`、零延迟和
  Goal86 的 z_t/BC 接口，做 C-line 阶段式课程与 terminal/path 冲突的单独审计；
- Goal140 纵向 tracking reward 继续排在 C-line 之后。
