# Goal158：H3 强基线上的 1M near10_2 神经残差配对证伪

**日期**：2026-11-12<br>
**状态**：`PREREGISTERED_BEFORE_MEASUREMENT / EVAL_ONLY_REUSE / HELDOUT_CLOSED`<br>
**上游**：Goal157 H3 candidate、Goal86 T2-SAC 1M、Goal154 near10_2、Goal155/156<br>
**性质**：只读复用已有 checkpoint 的配对 eval；不训练、不改 reward/terminal/success/radius、不扩 seed、不打开 held-out。

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
  check_command: .venv/bin/python ../scripts/check_research_ledger.py --goal Goal158 --goal-doc doc/goals/goal158_h3_strong_baseline_neural_pair_20261112.md --strict
```

## 1. 背景与证据

Goal157 发现：

- K2 zero 在 nominal strict125 只有 `26/40`，K4 zero 是 `40/40`；
- K4 修好纵向后，current10 的失败转为横向/航向主导；
- H3（`pitch_gain=4`、`heading_gain=3`）把 current10 从 `27/40` 提到 `31/40`；
- H3 在 nominal 仍 `40/40`，在 current15 从 `14/40` 提到 `17/40`。

因此之前“neural +19/480”是在 K2 zero 上测得的，不能直接说明神经残差优于传统
方法。现在必须用 H3 zero 作为强基线，重新配对已有 Goal86 1M checkpoint。

## 2. 假设

### H158-A：near10_2 在 H3 上仍有 success 增益

成立应看到：在 current10 或 current15，policy 比 H3 zero 至少 `+2/40`，且不是
单 seed 拉动；final vertical/horizontal 至少一项改善，timeout/safety 不增加。

失败应看到：policy 与 zero 相同或更差，或 residual 活跃但闭环指标不动/变差。

最便宜证伪：3 个 Goal86 1M checkpoint × current07/current10/current15 ×
policy/zero × dev40，复用 H3 配置；不训练。

### H158-B：神经残差在强基线上会破坏横向/航向

成立应看到：policy 的 heading error、rudder saturation 或 horizontal final miss
比 zero 明显上升，success 下降；失败模式从 zero 的 VERTICAL/HORIZONTAL 变为
MIXED/plane-miss。

失败应看到：横向指标不退化，residual 只带来小幅纵向/终端改善。

最便宜证伪：同 H158-A 的 paired traces，按通道和失败模式分解。

### H158-C：K2 zero 上的神经增益主要是“补控制器”

成立应看到：near10_2 相对 K2 zero 的增益，在 H3 zero 上显著缩小；H3 zero 本身
已经吸收了大部分 nominal/current07 的收益。

失败应看到：near10_2 在 H3 上仍保持与 K2 上相近的增益，说明残差有独立价值。

## 3. 数据与隔离

| 项目 | 值 |
|---|---|
| checkpoints | Goal86 T2-SAC 1M：86201/86202/86203 |
| controller | H3：`pitch_gain=4`、`heading_gain=3`、`heading_rate_gain=0.6` |
| stage | C1，关闭 direct command |
| dev seeds | `20–59`，40 个 |
| held-out | `100–149` 关闭 |
| levels | nominal / current07 / current10 / current15 |
| modes | `zero-residual`、`near10_2 policy` |
| success | strict125 `1.25 m`，`eta_tight_at_done` |
| 设备 | CUDA / float64 |

固定：Goal85 位置契约、`unbiased_noise_v1`、reward、terminal、goal radius、path
geometry、seed split。只替换 controller 与 residual action mode。

## 4. 方法

- zero arm：`env_h3.json` + `--action-mode zero-residual`，不加载 checkpoint；
- policy arm：同一 `env_h3.json` + Goal86 checkpoint + `near10_2` 配置，
  `learner_terminal_gate_mode=none`，`residual_scale=1.0`；
- 每个 training seed 独立配对，pooled 3 seeds；
- 记录 success、plane miss、timeout、safety、final vertical/horizontal、
  entry vertical p90、residual RMS 分箱、heading/pitch error、crab、saturation。

## 5. 执行计划

- [x] Step 0：Goal157 记录 H3 candidate；
- [x] Step 1：确认 Goal86 1M checkpoint 与 near10_2 配置可用；
- [ ] Step 2：单 seed/current10 smoke，检查 gate 与 residual 非零；
- [ ] Step 3：跑 `3 checkpoints × 4 levels × 2 modes × 40`；
- [ ] Step 4：配对分析 success、failure mode、residual 通道；
- [ ] Step 5：写 result、LEDGER、STATE；若失败转 channel-restricted residual / terminal credit。

## 6. Gate

这是 screen，不是 formal。裁决：

| check | 阈值 |
|---|---:|
| policy vs H3 zero | current10 或 current15 至少 `+2/40` |
| seed 一致性 | 至少 `2/3` checkpoint 的 Δsuccess `>=0` |
| 横向 guard | heading error / rudder saturation 不显著恶化 |
| 纵向 guard | final vertical 不比 zero 差 `+0.05 m` |
| 风险 | timeout/safety 不增加 |

失败裁决：

| 观察 | 判读 | 下一步 |
|---|---|---|
| policy 与 zero 相同 | 1M 残差被强基线吸收 | 5M/通道限制/terminal credit |
| policy 更差 | 动作参数化或 credit 错 | 限制 heading 通道，先保 zero |
| policy 小幅更好 | 有独立价值 | 新预注册 H3+residual formal |

## 7. Git

禁止 `git add -A`。只 add 本 Goal 相关文件，提交前运行 `git diff --cached --check`。

## 8. 自主迭代契约

执行 `research/AUTORESEARCH_LOOP.md`：失败先归因，不换算法、不扫 seed；每个 blocker
至少两个可证伪 hypothesis；先跑最便宜 falsification，再决定 `continue` / `pivot` /
`stop`；不自动 commit/push，不打开 held-out，不改成功定义。
