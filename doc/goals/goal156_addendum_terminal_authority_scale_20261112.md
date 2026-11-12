# Goal156 Addendum：近端 residual scale 的动作参数化证伪

**日期**：2026-11-12<br>
**状态**：`DIAGNOSTIC_PREREGISTERED / EVAL_ONLY / NOT_FORMAL / HELDOUT_CLOSED`<br>
**上游**：Goal154 authority、Goal155 terminal-near 复核、Goal156 5M formal<br>
**性质**：只复用已有 1M full-authority checkpoint 做 eval-only scale sweep；不训练、不换算法、不改 reward/terminal/success/radius，不打开 held-out。

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
  max_outer_loops: 1
  check_command: .venv/bin/python ../scripts/check_research_ledger.py --goal Goal156 --goal-doc doc/goals/goal156_full_residual_train_eval_gate_formal_20261111.md --strict --require-result
```

## 1. 问题

Goal155 terminal-near 复核显示：

- 失败几乎全是 `PLANE_MISS`；default 失败纵向 `1.83–1.98 m`，进入 2 m 球时纵向 p90 `1.76–1.87 m`；
- `near10_2` 的 2–10 m residual `mean_abs` 约 `0.34–0.45`，没有塌缩，但与纵向误差/heave 的相关性跨 seed 符号不一致；
- `near10_2` 的 effective residual 只占动作 L2 的 `3–4%`，但 policy head 饱和可达 `42–68%`。

需要区分两种解释：

1. **动作幅度/参数化不足**：近端 residual 已经方向合理，但被 scale 压得太小；放大近端 scale 应减少纵向失败。
2. **credit/terminal capture 问题**：近端 residual 有幅度但方向不稳定；单纯放大 scale 不会转化 success，甚至可能增加 path/overshoot。

## 2. 假设与判据

### H-SCALE-A：近端 scale 是瓶颈

若把 `near10_2` 的 residual scale 从 `1.0` 调到 `0.5/0.75/1.5`，至少一个 scale 应满足：

- default success 相对 scale `1.0` 改善 `>= +2/40`；
- failure vertical 或 2 m-entry vertical p90 改善 `>= 10%`；
- path3D 不增加超过 `+0.05 m`；
- strict success 不下降超过 `1/40`。

成立判 `AUTH_SCALE_PARTIAL`：动作参数化是部分瓶颈，可把该 scale/gate 用于 Goal156 5M checkpoint 的 eval-only 复核。

### H-SCALE-B：scale 不是瓶颈

若所有 scale 的纵向失败/entry p90 基本不动，或 success 改善小于 `+2/40`，判 `AUTH_SCALE_NOT_BOTTLENECK`。下一步转 terminal vertical credit / entry-window reward，不继续扫 scale。

## 3. 固定设计

| 项目 | 值 |
|---|---|
| checkpoint | Goal86 `t2_sac` 3 seeds（`86201–86203`，1M full-authority） |
| gate | `configs/lowlevel/goal154/authority/near10_2.json` |
| residual scale | `0.5 / 0.75 / 1.0 / 1.5` |
| scenarios | default `2.0 m`、strict125 `1.25 m` |
| dev seeds | `20–59`，40 个 |
| 精度 | CPU / float64 |
| held-out | `100–149` 关闭 |
| 训练 | 无 |
| 算法 | SAC 不变 |

## 4. 指标

- success、plane-miss、timeout、safety；
- final goal distance、failure final distance、failure vertical/horizontal；
- 2 m entry vertical/horizontal p90、above/within/below；
- path3D/vertical/horizontal RMSE、terminal p90；
- residual RMS 分箱、policy-head/effective saturation、rate/scale clip。

## 5. 执行

```bash
bash scripts/run_goal86_terminal_scale_screen.sh
```

输出根：`artifacts/lowlevel/goal86_terminal_scale_screen_20261112/eval_cpu/`。

## 5.1 结果（2026-11-12，Goal86 1M full-authority checkpoints）

26 个 eval run 已完成，pooled `3 seeds × 40 episodes = 120`，CPU/float64：

| scale | default | Δ vs zero | strict125 | Δ vs zero | Δpath3d | Δvertical | residual sat | scale clip |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `0.50` | `84/120` | `+3` | `43/120` | `+1` | `-0.0003` | `-0.0033` | `0%` | `0%` |
| `0.75` | `85/120` | `+4` | `45/120` | `+3` | `-0.0006` | `-0.0046` | `0%` | `0%` |
| `1.00` | `87/120` | `+6` | `46/120` | `+4` | `-0.0009` | `-0.0056` | `61%` | `0%` |
| `1.25` | `89/120` | `+8` | `47/120` | `+5` | `-0.0009` | `-0.0067` | `90%` | `3.8–4.8%` |
| `1.50` | `88/120` | `+7` | `48/120` | `+6` | `-0.0010` | `-0.0076` | `95%` | `5.7–6.7%` |

per-seed default（policy/zero，括号为 delta）：

| scale | 86201 | 86202 | 86203 |
|---:|---:|---:|---:|
| `1.00` | `28/27 (+1)` | `28/27 (+1)` | `31/27 (+4)` |
| `1.25` | `28/27 (+1)` | `28/27 (+1)` | `33/27 (+6)` |
| `1.50` | `28/27 (+1)` | `27/27 (+0)` | `33/27 (+6)` |

strict125：

| scale | 86201 | 86202 | 86203 |
|---:|---:|---:|---:|
| `1.00` | `15/14 (+1)` | `16/14 (+2)` | `15/14 (+1)` |
| `1.25` | `15/14 (+1)` | `16/14 (+2)` | `16/14 (+2)` |
| `1.50` | `16/14 (+2)` | `16/14 (+2)` | `16/14 (+2)` |

判读：`AUTH_SCALE_PARTIAL_SUPPORTED`。近端 residual scale 确实是部分瓶颈；
`1.0–1.5` 都在 default/strict 上超过 Goal156 的 `+6/+4` success 阈值，且 path
没有恶化。`1.0` 是保守选择（无 clip、saturation 61%）；`1.25` 是性能选择
（default 最高，saturation 90%）；`1.5` 已到权限边缘（saturation 95%、clip 4–7%）。

但收益是**边缘转化**，不是纵向分布整体修复：`1.25` 的 default failure vertical
仍为 `1.91 m`、entry 2m vertical p90 `1.90 m`；strict failure vertical
`1.63 m`、entry p90 `1.90 m`。success 的增加来自少量 episode 被推过 2 m/1.25 m
壳层，而不是失败纵向从 ~1.9 m 系统性降到 <1 m。因此这是动作参数化的部分证据，
不能替代后续 terminal vertical capture / reward 工作，也不能据此宣称纵向问题已解决。

**边界**：这些是 Goal86 1M checkpoint 的结果，不是 Goal156 5M 正式 Gate。Goal156
主 Gate 仍只使用预注册的 `near10_2` scale `1.0`；不得在看到 5M 结果后把主 arm
回改成 `1.25/1.5`。scale 变体只能作为 post-hoc 动作参数化诊断；若 5M 上
`1.25` 明显更好，必须另开预注册 Goal 复现，不能追溯修改 Goal156 Gate。

## 6. 解释边界

- 这是 1M checkpoint 的诊断，不是 Goal156 5M 的正式结论；
- 不把 scale sweep 的结果写成“算法收益”；
- 不因为 scale 有局部改善就跳过 Goal156 Gate；
- 若 scale 无效，按 `AUTH_SCALE_NOT_BOTTLENECK` 记录，并转 terminal vertical credit/reward。

## 7. 下一步

1. 等 Goal156 3 个 5M `final.zip` 完成；
2. 主 Gate 仍跑 `near10_2 scale=1.0`；
3. 同时对同一 5M checkpoint 跑 eval-only `near10_2 scale=1.0/1.25/1.5` 的终端分布诊断；
4. 若 5M 上 `1.25` 仍保持 `+6/+4` 且 saturation 可接受，另开预注册 Goal
   `terminal authority scale`，不要回改 Goal156。
