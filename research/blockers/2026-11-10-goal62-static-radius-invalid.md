# Blocker：Goal62 static-radius 反事实在终止边界退化

日期：2026-11-10<br>
关联 Goal：Goal62<br>
上游：Goal61 `DIAG_COMPLETE_H_C1_C2_C3_SUPPORTED`<br>
状态：`open`，已 pivot 到 Goal63 闭环终止抑制

## 1. 现象

Goal62 在 Goal60 的 10 个 trace 上，把 success 的 `p_nav` 半径收紧
`0.15/0.25/0.35 m`。结果不是 false success 下降而 true success 保留，而是：

- default policy pooled：baseline success `112`，candidate success `0`；
- strict125 policy pooled：baseline success `59`，candidate success `0`；
- 4 个 policy checkpoint 和 forced zero-residual 全部同向；
- 三个 margin 都得到 `false_success_removed_fraction=1.0`、
  `true_success_retention=0.0`、`candidate_precision=null`。

脚本最初把它写成 `FAIL_STATIC_RADIUS`。复核后改为
`INVALID_COUNTERFACTUAL_DEGENERATE_BOUNDARY`：这是因为 trace 在首次进入原
`p_nav` 半径时结束，不是静态半径假设的有效闭环检验。

## 2. 证据

- `doc/goals/goal62_result_20261110.md`
- `artifacts/lowlevel/goal62_terminal_radius_counterfactual_20261110/goal62_terminal_radius_counterfactual.json`
- `artifacts/lowlevel/goal62_terminal_radius_counterfactual_20261110/goal62_terminal_radius_counterfactual.md`
- `scripts/analyze_pc_goal62_terminal_radius_counterfactual.py`
- `tests/test_pc_goal62_radius_counterfactual.py`
- Goal61：policy success_no_capture 为 `58/36`，zero-residual 为 `13/13`，
  `p_nav` closer/direction 均为 `1.000`

最终脚本 SHA256：
`cb30bb49fd25a2776c208a1b4f22ab957208a285724d6e4761986cd3c558f703`。
最终 JSON SHA256：
`25c12003a45cedf3573d70d65100407edb2a9c4c6a202e061238c34434f25ee6`。

## 3. 已排除项

| 解释 | 状态 | 理由 |
|---|---|---|
| NoisyNet 主导 | 已排除 | Goal59 的 8/8 deterministic mean 饱和；noise/det 中位数 `0.0193`；Goal60/61 的 zero-residual 也复现 SNC |
| residual 太小/样本效率 | 已排除为当前主因 | final residual L2 ratio 约 `0.063/0.065`，训练曲线平台化，zero-residual 仍有终止偏差 |
| 数据损坏 | 已排除 | 40 seeds、40 terminal 行、finite、SHA、device/dtype、policy/zero radius 全部通过 |
| candidate 规则实现为反向 | 已排除 | 更紧半径是 baseline success 的子集，逐 episode 重放 baseline 标签一致 |
| 所有 margin 都是实质 FAIL | 未成立 | trace 在边界截断，闭环后续不可见，指标退化为 0/0 型测量问题 |

## 4. 新 hypothesis

### H63-A：闭环抑制终止能区分 false success 与 true capture

机制：在 frozen checkpoint 上，仅抑制 `GOAL` 终止，直到 `p_nav` 达到
`r0 - 0.25 m`，或发生 timeout/plane-miss/safety。若以前的 false success 只是
“过早按 p_nav 结束”，其中相当一部分 episode 应继续进入更紧半径或最终进入
true `eta` capture。

可证伪预测：两个场景的 policy pooled 至少满足
`false_success_removed_fraction >= 0.50` 且
`true_success_retention >= 0.90`；如果抑制后 false success 仍保留，或 true
success 大量 timeout/plane-miss，则 H63-A 不成立。

最便宜检验：4 个 frozen checkpoint × 2 scenario × dev `20–59` 的 CPU/float64
闭环 eval；zero-residual 同协议对照；不训练。

### H63-B：离线 terminal-row 反事实对停止时间选择有结构性偏倚

机制：Goal60 trace 只在首次穿过原半径时保留一行，继续 rollout 被截断；任何正
margin 都只能删除已观测 success，无法观测后续进入更紧半径的事件。因此该设计
对 margin 的效应估计必然退化。

可证伪预测：若现有 trace 中存在 baseline success 的 terminal `p_nav` 距离明显
小于 `r0 - max_margin`，或某个 margin 的 candidate success 非零，则 H63-B 被
证伪。当前 10 个 run 全部为 `candidate_success=0`，支持该机制，但不能据此推断
闭环规则无效。

最便宜检验：检查 terminal distance 分布和 candidate 集大小；不训练、不 rollout。

### H63-C：若抑制终止后仍不能进入更紧半径，问题在 estimator/geometry

机制：如果闭环继续飞行后，episode 系统性 timeout 或 plane-miss，且 terminal
`p_nav-eta` gap 与方向偏差不收敛，则不是静态标签，而是 `p_nav` 估计器、终端
几何或 controller 无法把真实位置带到目标附近。

可证伪预测：在 H63-A 的抑制 arm 中，若多数原 false-success episode 最终进入
更紧 `p_nav` 或 true capture，则 H63-C 不成立；若大部分仍停在半径外或 plane
miss，则 H63-C 支持。

最便宜检验：复用 Goal63 的 terminal trace，统计 terminal `p_nav`/`eta`
distance、timeout、plane-miss、extra steps；不新增训练。

## 5. 排查顺序与决策

按 `数据 → 标签 → 评测 → baseline → 架构 → 文献`：

1. 数据/标签：Goal60/61 的 40-seed 数据完整性通过；
2. 评测：确认 trace 在首个 p_nav 半径 crossing 停止，离线 margin 不可识别；
3. baseline：zero-residual 与 policy 同向退化；
4. 架构：下一步只做 eval-only 终止抑制，不改 reward/observation/controller；
5. 文献/方法：见
   `research/paper_reading/2026-11-10-goal62-terminal-boundary-invalid.md`；
6. 最便宜 falsification：Goal63 闭环 frozen-checkpoint eval。

决策：`pivot`。不把 Goal62 写成静态半径 FAIL；不重开 Goal27/59；不启动
SAC/REDQ/DroQ；不打开 held-out `100–149`。

## 6. 算力与协议

Goal63 的 40 个 dev seeds 可按 scenario/checkpoint 分片并行，最多 8 个进程，
每片保持相同 seed 集、动作模式、dtype/device 和终止规则，最终在 seed 层合并。
这是评估并行，不改变正式 Gate 的 `num_envs`、`batch_size` 或 UTD。后续若进入
训练分支，只在新的预注册 arm 中扩大 `num_envs`/batch，并保持总 transitions、
UTD 和评估协议可比。
