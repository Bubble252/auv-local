# Goal157：K4 基座上的航向增益基线阶梯（eval-only）

**日期**：2026-11-12<br>
**状态**：`PREREGISTERED_BEFORE_MEASUREMENT / EVAL_ONLY / HELDOUT_CLOSED`<br>
**上游**：Goal145/146/147/148/149/150/155、Goal86 1M、Goal156 5M（并行中）<br>
**性质**：只读 eval-only 控制器审计；不训练、不改 reward/terminal/success/radius、不改神经算法、不打开 held-out。

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
  check_command: .venv/bin/python ../scripts/check_research_ledger.py --goal Goal157 --goal-doc doc/goals/goal157_controller_heading_gain_baseline_20261112.md --strict
```

`required: true`：这是假设生成/归因型 eval-only 审计，失败也必须留下 research 记录。

## 1. 背景与证据

Goal147 已证明 `base_controller.pitch_gain: 2 -> 4`（K4）把 C1 nominal strict125
从 `26/40` 提到 `40/40`，policy 与 zero-command-residual 几乎完全相同，说明收益
来自传统控制器，不是 residual。Goal148/149 进一步给出 K4 的 current 边界：

| level | K4 zero strict125 | K2 zero strict125 | K4 heading err | K4 crab | K4 rudder sat |
|---|---:|---:|---:|---:|---:|
| nominal | `40/40` | `26/40` | `0.124` | `7.59°` | `0.265` |
| current05 | `39/40` | `18/40` | `0.165` | `9.79°` | `0.325` |
| current07 | `36/40` | `15/40` | `0.186` | `10.89°` | `0.352` |
| current10 | `27/40` | `11/40` | `0.209` | `12.02°` | `0.377` |
| current15 | `14/40` | `8/40` | `0.269` | `15.20°` | `0.440` |

本轮对 Goal149 K4 zero traces 做了最终 miss 分解：

| level | K4 失败竖向中位数 | K4 失败横向中位数 | 竖向 >1 m 比例 | 横向 >1 m 比例 |
|---|---:|---:|---:|---:|
| nominal | NA（0 失败） | NA | NA | NA |
| current05 | `0.962` | `1.089` | `0.00` | `1.00` |
| current07 | `1.140` | `0.792` | `0.75` | `0.50` |
| current10 | `0.976` | `1.057` | `0.46` | `0.62` |
| current15 | `0.727` | `1.578` | `0.38` | `0.73` |

结论：K2 的失败由纵向主导；K4 修好纵向后，current05–15 的剩余失败转为横向/航向
主导。K4 的 rudder 饱和度只有 `0.33–0.44`，说明航向环仍有增益余量。当前神经
residual 在 Goal150 中一旦输出大幅 heading/pitch 残差就会破坏闭环，说明先建立
一个不被控制器缺陷压低的传统基线，是判断神经方法是否有真实增益的前提。

## 2. 假设

### H157-A：K4 的剩余 current 失败主要由航向环增益不足造成

成立应看到：提高 `heading_gain` 后，current07/current10 的 strict125 success 上升，
横向最终误差和 crab angle 下降，且 rudder 未饱和；纵向指标不退化。

失败应看到：success 不升或下降，横向误差不降，或 heading gain 提高导致振荡、
rudder 饱和、crab 变大。此时剩余瓶颈不是航向增益，转 guidance/current compensation。

最便宜证伪：固定 K4 pitch，`heading_gain ∈ {2,3,4}`，current05/07/10/15，
dev40 strict125，zero-residual eval-only；不训练。

### H157-B：更高航向增益会把失败从横向转移到纵向/执行器饱和

成立应看到：H4 的横向误差下降但纵向误差或 rudder/rpm 饱和上升，success 不增；
失败模式从横向 miss 变为纵向 miss 或饱和。

失败应看到：H4 同时改善横向和 success，且饱和不升；此时 H4 是可保留的传统基线。

最便宜证伪：同一阶梯同时报告 final vertical/horizontal、rudder/stern/rpm saturation、
heading error、crab angle 和 failure mode。

### H157-C：现有 K2 zero 基线低估了传统方法，导致神经增益被高估

成立应看到：K4/H4 zero 在 nominal/current05/current07 接近或达到 success 上限，
Goal86/Goal155 的 neural Δsuccess 在这些档位缩小；神经价值只能在 current10+
或 strict125/高速/长路线中体现。

失败应看到：K4/H4 zero 仍显著低于 neural policy，说明 neural 增益不是基线缺陷。

最便宜证伪：把 Goal86 1M 的 `near10_2` 与本次 K4/H4 zero 做同场景配对，不重训。

## 3. 数据与隔离

| 项目 | 值 |
|---|---|
| checkpoint | Goal144 C1 `C1_000300000.pt`（只用于 zero-residual 兼容；policy 不参与） |
| stage | C1，关闭 direct command |
| dev seeds | `20–59`，40 个 |
| held-out | `100–149` 关闭 |
| current levels | nominal / current05 / current07 / current10 / current15 |
| success | strict125 `1.25 m`，`eta_tight_at_done`；default `2.0 m` 只作辅助 |
| 评估设备 | CUDA、float64 |
| 固定项 | reward、terminal、success、goal radius、sensor、guidance、reference、seed |

禁止：

- 用 held-out 选择增益；
- 在同一阶梯中改 reward/terminal/success/radius；
- 把本审计的 zero 结果写成神经 residual 结果；
- 用 best-on-dev 的增益直接改 Goal156 的冻结 Gate。

## 4. 方法

### 4.1 控制器阶梯

固定 `pitch_gain=4`、`pitch_rate_gain=1`、`heading_rate_gain=0.6`，只改变
`heading_gain`：

| arm | heading_gain | 目的 |
|---|---:|---|
| `H2` | `2` | Goal149 K4 对照，复现 `36/40`、`27/40` |
| `H3` | `3` | 中等航向增益 |
| `H4` | `4` | 与 pitch K4 对称的航向增益 |

所有 arm 使用 zero-command-residual，避免 policy 噪声混淆控制器效果。

### 4.2 压力注入

复用 Goal149 的 eval-only curriculum override：

| level | current range (m/s) | shear |
|---|---|---|
| nominal | C1 原值 `0.20–0.35` | `0` |
| current05 | `0.25–0.45` | `0.5` |
| current07 | `0.275–0.50` | `0.75` |
| current10 | `0.30–0.55` | `1.0` |
| current15 | `0.375–0.675` | `1.5` |

### 4.3 指标

- success、plane miss、timeout、safety；
- final vertical / horizontal miss，2 m 入口纵向 p90；
- heading error、pitch error、crab angle、heading-rate clip；
- rudder/stern/rpm saturation；
- 失败模式分解（竖向/横向/混合）。

## 5. 执行计划

- [x] Step 0：确认 Goal149 traces 可复用，完成 K4 失败竖向/横向分解；
- [x] Step 1：写本预注册文档和固定矩阵；
- [ ] Step 2：生成本轮 `env_h2/h3/h4` 配置并做单场景 smoke；
- [ ] Step 3：跑 `3 arms × 5 levels × 40 seeds` strict125；default 只对 winner 补跑；
- [ ] Step 4：分析 success、final vertical/horizontal、saturation、failure mode；
- [ ] Step 5：若 H3/H4 明确优于 H2，另写新的预注册 Goal 把该控制器作为传统基线；
- [ ] Step 6：写 result、LEDGER、STATE，跑 checker。

## 6. Gate

这不是神经方法 Gate，而是传统基线审计。裁决规则：

| check | 阈值 |
|---|---:|
| H 阶梯有效性 | 至少一个 H3/H4 在 current07 或 current10 比 H2 `+4/40` |
| 不伤纵向 | winner 的 final vertical 不比 H2 差超过 `+0.05 m` |
| 不伤 nominal | winner nominal 不低于 `39/40` |
| 执行器 | winner rudder saturation 不高于 H2 `+0.05`，timeout/safety 全 0 |
| 稳定性 | 0 error / 0 NaN / 0 飞点 |

失败裁决：

| 观察 | 判读 | 下一步 |
|---|---|---|
| H3/H4 无增益 | 航向增益不是瓶颈 | 转 guidance/current compensation 审计 |
| 横向降但 success 不升 | 失败转移到纵向/几何 | 分解 terminal miss，不继续加 gain |
| gain 提高导致饱和/振荡 | 执行器包络不足 | 保留 H2，转速度/舵限幅与 guidance |
| H4 明确更好 | 传统基线被低估 | 新预注册 Goal，用 H4 zero 作为神经对照 |

## 7. Git

禁止 `git add -A`。只 add 本 Goal 相关文件，提交前运行：

```bash
git diff --cached --check
```

## 8. 自主迭代契约

执行 `research/AUTORESEARCH_LOOP.md`：

1. 失败先归因，不换算法、不扫 seed；
2. 每个 blocker 至少两个可证伪 hypothesis；
3. 需要外部方法时先写 reading note；
4. 先跑最便宜 falsification，再决定 `continue` / `pivot` / `stop`；
5. 不自动 commit/push，不打开 held-out，不改成功定义。
