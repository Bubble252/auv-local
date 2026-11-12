# Goal27 H3-SR 结果：监督残差仍不能改善闭环

日期：2026-11-09<br>
状态：**H3_SR_FAIL_STOP**<br>
预注册：`doc/goals/goal27_addendum_h3_supervised_residual_20261109.md`

## 1. 结论

在 frozen D3 上，用 40 个独立训练 seed 的 D3-state DAgger/teacher 标签做监督动作
残差，确实学到了非零残差：default/strict125 的 residual/base L2 分别为
`16.14%/16.33%`。但闭环收益不成立：

- 相对同 checkpoint 的 frozen-base，pre-goal 3D RMSE 只改善
  `0.0231/0.0231 m`，远低于 `0.10 m` 门禁；
- success 从 `0.725/0.500` 降到 `0.650/0.350`；
- plane-miss 从 `0.275/0.500` 升到 `0.350/0.650`；
- safety 和 timeout 没有恶化，但 success-first 护栏已经失败；
- default `path-zero` 使 pre-goal RMSE 恶化约 `104.6%`，说明路径输入和残差通路
  确实被使用，但学到的是会放大闭环误差的修正，而不是有效路径修正。

因此 H3-SR（on-policy 样本效率/探索不足是主因，监督残差应能修复）未被支持。
本轮不继续扫学习率、epoch、loss 权重、seed，也不把该 checkpoint warm-start 到 PPO。

## 2. 固定协议

| 项目 | 值 |
|---|---|
| base | `checkpoints/goal26/d3/pc_bc_actor.pt` |
| base SHA256 | `7430b26308b701565a52a92ed31c382ccca9cf1721d183ab856e3fafd9f4ab73` |
| 训练配置 | `configs/lowlevel/pc_goal27_a1_ppo_default_27006.json` |
| 训练环境 | `configs/lowlevel/pc_goal27_a1_reward_v2_default_20261109.json` |
| 训练 seeds | `60–99`（40 env，不碰 dev `20–59`、held-out `100–149`） |
| 采集 | 40 env × 8192 steps，D3 执行动作、ALOS/PC teacher 标注，无动作噪声 |
| 评估 | dev seeds `20–59`，`endpoint_stop`、`realistic_calibrated`、CPU/float64 |
| checkpoint | 监督预训练 `final.pt`，不按 best-epoch 事后挑选 |
| held-out | 未打开 |

## 3. 数据与训练审计

| 项目 | 值 |
|---|---:|
| DAgger samples / episodes | `327,680 / 283` |
| DAgger goals / safety events | `185 / 0` |
| student correction L2 mean / p95 | `0.4173 / 1.0927` |
| dataset SHA256 | `0e182bbaef4571bd1b60c42e95138a663bae8a9c91087951fc277f6e7dcc2fad` |
| train / validation samples | `262,597 / 65,083` |
| train / validation episodes | `226 / 57` |
| best epoch / epochs completed | `9 / 40` |
| frozen BC max abs drift | `0.0` |
| final checkpoint SHA256 | `7549e99edd376c27eef35a5c11c29e1e22888903944f0d2a8cb61bd76212d152` |

标签和预测均非零。归一化标签 abs mean 为
`[1.0255, 0.7191, 0.5871]`，模型预测 abs mean 为
`[0.5190, 0.2245, 0.2627]`；因此失败不是“残差又没有学出来”。

## 4. 40-seed 结果

| scenario | mode | success | pre-goal RMSE (m) | full RMSE (m) | plane miss | timeout | safety | material sat. | residual/base L2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| default | frozen-base | 0.725 | 2.8492 | 2.7634 | 0.275 | 0.0 | 0.0 | 0.1918 | 0 |
| default | policy | 0.650 | 2.8260 | 2.7467 | 0.350 | 0.0 | 0.0 | 0.2021 | 16.14% |
| default | path-zero | 0.000 | 5.7835 | 5.7628 | 0.950 | 0.0 | 0.05 | 0.0657 | 29.86% |
| strict125 | frozen-base | 0.500 | 2.8291 | 2.7520 | 0.500 | 0.0 | 0.0 | 0.1919 | 0 |
| strict125 | policy | 0.350 | 2.8061 | 2.7334 | 0.650 | 0.0 | 0.0 | 0.2020 | 16.33% |

逐 seed 配对：

- default policy − frozen：pre-goal RMSE 均值差 `-0.0077 m`，95% bootstrap CI
  `[-0.0370, 0.0207] m`，20/40 改善、20/40 恶化；success 净变化 `-0.075`，
  4 个 seed 由失败转成功、7 个由成功转失败；
- strict125 policy − frozen：pre-goal RMSE 均值差 `-0.0062 m`，95% bootstrap CI
  `[-0.0375, 0.0244] m`，20/40 改善、20/40 恶化；success 净变化 `-0.150`，
  3 个 seed 由失败转成功、9 个由成功转失败；
- default path-zero − policy：pre-goal RMSE 均值差 `+3.1969 m`，95% CI
  `[+2.6787, +3.7500] m`，39/40 恶化。

## 5. 门禁

| check | 结果 | 判定 |
|---|---|---|
| 主效果：任一场景改善 `>=0.10 m` | default `-0.0231 m`；strict125 `-0.0231 m` | **FAIL** |
| 另一场景 RMSE 恶化不超过 `3%` | 两场景均约改善 `0.8%` | PASS |
| success 不低于 frozen-base `0.025` | default `-0.075`；strict125 `-0.150` | **FAIL** |
| default success `>=0.695` | `0.650` | **FAIL** |
| strict125 success `>=0.470` | `0.350` | **FAIL** |
| safety 恶化不超过 `0.02` | `0.0/0.0` | PASS |
| plane-miss 恶化不超过 `0.05` | `+0.075/+0.150` | **FAIL** |
| timeout 恶化不超过 `0.05` | `0.0/0.0` | PASS |
| 残差非零 | L2 ratio `16.14%/16.33%` | PASS |
| default path-zero 恶化至少 `5%` | `+104.6%`（诊断性，不是候选策略） | PASS |
| finite / NaN / 崩溃 | 5 个 40-seed summary 全部 finite | PASS |

总体：**H3_SR_FAIL_STOP**。

## 6. 解释与下一步

监督标签让残差从近零变成了约 16% 的动作幅度，但没有把动作修正方向对齐到闭环
路径误差；反而在部分 seed 上把 success 变成 plane-miss。这个结果与 Goal9 的
监督残差失败方向一致，但本实验使用了 D3、A1 reward-v2 环境和固定 dev 协议，因此
直接回答本轮 H3-SR 问题。

停止：

- 不继续扫学习率、epoch、output L2、smoothness L2 或训练 seed；
- 不把该 checkpoint warm-start 到 PPO；
- 不把 path-zero 的强退化解释成“残差有效”，它只说明残差依赖路径输入；
- 不把 C1p 或本结果混入可部署收益。

下一主线保持 `pivot`：先预注册 Goal28 的任务级 terminal capture/plane-miss 诊断；
若再测 residual，只能在新的预注册 Gate 下测试 off-policy 或其他机制，不能把
本监督残差失败当作继续加预算的理由。

## 7. 证据

远程隔离目录：

```text
/data/yhx/underwater-robotics/src/plan_p1/goal27_command_residual_20261109/repo/artifacts/h3_sr/
```

关键文件：

```text
artifacts/h3_sr/demos_d3_dagger_default_s60_99/demos.manifest.json
artifacts/h3_sr/pretrain_residual_sr01/training_manifest.json
artifacts/h3_sr/pretrain_residual_sr01/training_result.json
artifacts/h3_sr/pretrain_residual_sr01/final.pt
artifacts/h3_sr/eval_default_40/{policy,frozen_base,path_zero}/summary.json
artifacts/h3_sr/eval_strict125_40/{policy,frozen_base}/summary.json
```

本地文档与 ledger：

```text
doc/goals/goal27_addendum_h3_supervised_residual_20261109.md
doc/goals/goal27_h3_sr_result_20261109.md
research/LEDGER.jsonl
research/STATE.md
```

未使用 held-out `100–149`；未覆盖 Goal24/25/26 checkpoint、配置或 artifacts；
未自动 commit/push。
