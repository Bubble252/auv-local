# Goal9 S15-C 触发判定

更新时间：2026-11-05

状态：**`C_not_triggered`**。没有启动 C 训练，没有创建 C checkpoint，也没有用 C 结果
反向选择 S15 基线。

## 1. 协议触发条件

只有同时满足以下两类证据，才允许启动 C：

1. A 或 B 证明时间尺度/历史采样确实有可解释的因果影响；
2. `DADE + MLP` 仍存在明确的信息通路瓶颈，且直接 Transformer actor 有可验证收益空间。

启动后还必须先做 BC 监督蒸馏，再有界微调；C 的参数量、p95 时延、训练数据、物理秒数
和评估 seed 必须与最佳 A/B 组同口径。

## 2. 逐条检查

| 谓词 | 证据 | 判定 |
|---|---|---|
| A 存在可解释的历史因果贡献 | A screen 中 `A2_stride10` 有 default 正指标，但 `history-zero`/`history-reverse` 没有可解释退化；`selected_arm=A0_stride1`、`selection_status=baseline_no_signal` | 不满足 |
| B 证明时间尺度有影响 | B10 只通过非劣效率门禁；B5 为 `fail_non_inferior`，success 下降 `0.05` | 不满足 |
| DADE+MLP 有明确通路瓶颈 | A/B 没有提供可审计的通路瓶颈证据；B10 的收益主要来自降频，不是信息通路 | 不满足 |
| 直接 Transformer 有可验证收益空间 | 没有预注册的 C 前置证据；不能先跑再看结果补理由 | 不满足 |

## 3. 判定

```text
stage = S15-C
decision = C_not_triggered
reason = no_causal_temporal_effect_and_no_auditable_dade_mlp_bottleneck
training_started = false
checkpoint_created = false
next_action = freeze_s15_with_a0_stride1
```

## 4. 证据路径

- A summary：`artifacts/lowlevel/pc_abc_temporal_20261105/A_screen/report/summary.json`
- A manifest：`artifacts/lowlevel/pc_abc_temporal_20261105/A_screen/run_manifest.json`
- B10 summary：`artifacts/lowlevel/pc_abc_temporal_20261105/B10/report/summary.json`
- B5 summary：`artifacts/lowlevel/pc_abc_temporal_20261105/B5/report/summary.json`
- B 阶段报告：`doc/goals/goal9_b_stage_report.md`

不要把 B10 的效率收益写成 C 的通路瓶颈证据，也不要把 A2 的相关性写成历史因果收益。
