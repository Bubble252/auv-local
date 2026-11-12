# Goal9 V-0D 监督残差初始化结果

日期：2026-11-06

结论：**`fail_sr0d`**。监督残差确实被策略使用，而且路径归因很强，但在固定
20-seed 闭环评估中使 3D RMSE 相对同 harness 冻结 BC 恶化约 `1.24%`，未通过
预注册主门禁。按预注册规则停止该分支，不用该 checkpoint warm-start PPO。

## 执行链路

- 预注册文档：`doc/goals/goal9_v0d_supervised_residual_diagnostic_20261106.md`
- 第一次正式运行 v1 已完成训练，但评估器因 checkpoint 缺少 `pc.path.dim/blocks`
  元数据而拒绝加载；v1 原样保留，不进入最终比较。
- 修复 pretrain 的 path checkpoint 元数据后，smoke v3 已通过完整评估器加载。
- 正式结果使用 `R-SR-0D-scale1p0-v2`，训练配置、环境、示范数据、seed、预算和
  GPU 与预注册一致。
- 评估硬件与服务：服务器 GPU5 训练；评估为同一 harness 的 CPU/float64，
  seed `0–19`；三组使用同一个 `final.pt`。

## 训练审计

| 项目 | 值 |
|---|---:|
| manifest status | `completed` |
| epochs completed | `47 / 200`（best epoch `16` 后早停） |
| initial validation loss | `0.0027633749` |
| best validation loss | `0.0024396006` |
| final validation loss | `0.0034395603` |
| 冻结 BC 参数最大漂移 | `0.0` |
| 所有 loss 有限 | `true` |
| 残差标签绝对均值（三通道） | `[0.052119, 0.040131, 0.020966]` |
| 预测残差绝对均值（三通道） | `[0.020459, 0.026266, 0.012416]` |
| 预测残差绝对最大值（三通道） | `[0.165913, 0.389833, 0.150374]` |
| final checkpoint SHA256 | `498bb683b30ad6035fa95ef893bfd5df57bed32adccba8c0882274cfbbc4d349` |

## 20-seed 结果

| 指标 | default SR | frozen-base BC | path-zero SR |
|---|---:|---:|---:|
| 3D RMSE | `12.4160 m` | `12.2633 m` | `21.2390 m` |
| success | `0.15` | `0.15` | `0.00` |
| timeout | `0.85` | `0.85` | `1.00` |
| safety termination | `0.00` | `0.00` | `0.00` |
| material saturation | `0.01194` | `0.01386` | `0.06506` |
| residual action abs mean | `0.022827` | `0.0` | `0.030943` |
| residual action abs p95 | `0.062858` | `0.0` | `0.082090` |
| residual action abs max | `0.334184` | `0.0` | `0.646301` |
| residual / base action L2 | `0.140713` | `0.0` | `0.075826` |

## 预注册门禁

| 门禁 | 结果 | 判定 |
|---|---:|---|
| `SR / BC-0D` 3D RMSE ratio `<= 0.95` | `1.012448` | **失败** |
| `SR / R-BC-3D` ratio `<= 1.05` | `0.986348`（基线 `12.5878 m`） | 通过 |
| path-zero 使 3D RMSE 恶化至少 `5%` | `+71.06%` | 通过 |
| success/timeout/safety 恶化 `<= 0.02` | 三项均 `0.00` | 通过 |
| 冻结 BC 漂移为 `0` | `0.0` | 通过 |
| 所有 loss 有限 | `true` | 通过 |
| `strict125` | evaluator 未输出 | 不可用，未伪造 |

## 逐 seed 配对

- default 相对 frozen-base：
  - 逐 episode 3D RMSE 均值：`11.50854 m` 对 `11.35671 m`；
  - 配对差均值 `+0.151831 m`，95% t 区间 `[+0.030777, +0.272884] m`；
  - `6/20` 改善、`14/20` 恶化。
- path-zero 相对 default：
  - 配对差均值 `+9.014236 m`，95% t 区间 `[+5.263816, +12.764657] m`；
  - `3/20` 改善、`17/20` 恶化；
  - success 减少 `3` 个 episode，timeout 增加 `3` 个 episode。

## 解释

监督初始化没有出现“残差太小、完全没学”的问题：默认策略的残差幅度明显非零，
path-zero 又使 3D RMSE 恶化 `71.06%`，说明路径输入和残差通路都在影响闭环行为。
失败发生在方向和/或分布泛化上：在本轮同分布闭环中，残差修正带来的平均效应是
把路径跟踪变差，而不是改善。

因此本轮只能支持以下结论：

- 监督 teacher-minus-BC 标签本身不足以直接形成可靠的闭环残差；
- 不能通过继续增加监督 epoch、扫学习率或把该 checkpoint 硬塞进 PPO 来追认成功；
- 路径特征并非无效，问题在于当前监督残差没有学到与闭环误差对齐的修正方向。

## 证据文件

结果根：`artifacts/lowlevel/pc_v0d_supervised_residual_20261106/R-SR-0D-scale1p0-v2/`

- comparison：
  `comparison_summary.json`，SHA256
  `0dcfe459a92a8ad64a7eaa2721699edd7b70d361b7c1828d39c0690da4ffdc54`
- default summary：
  `eval_default/summary.json`，SHA256
  `7cae1adcff9d19012bcb89c67003eddc108fad6f78b81b0e0766b8ae48e24a03`
- frozen-base summary：
  `eval_frozen_base/summary.json`，SHA256
  `3602e8ed211cf5f6164701ff67105d7e964a60ea569a982d7bce47d21cf08220`
- path-zero summary：
  `eval_path_zero/summary.json`，SHA256
  `4d63eb415a02329502dde8dc9a6372c74047a53751e35d44b7eba7e4d5173d23`

日志位于 `logs/pc_v0d_20261106/`，前缀为
`pretrain_R-SR-0D-scale1p0-v2`、`eval_R-SR-0D-scale1p0-v2`。

## 下一步

按预注册停止 `R-SR-0D` 分支：不 warm-start PPO，不进入正式
`3 training seeds × 50 evaluation seeds`，不继续扫监督学习率/epoch/output L2。
若继续 Goal9，需要人工审核并另行预注册新的机制性假设。
