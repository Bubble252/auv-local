# 决策：full-vector terminal bias correction 支持，但仍是 oracle

日期：2026-11-10<br>
关联：Goal78 / Goal77 / Goal76 / Goal75<br>
状态：`FULL_VECTOR_DEADBAND_SUPPORTED`<br>
决策：`continue`，先做 observable predictor；不直接部署 oracle，不换 RL 算法

## 1. 结果

Goal78 用完整 `p_nav-eta` 向量 deadband（`b0=0.25 m`）重新评估现有
Goal71 episodes：

| scenario | A1 along-goal false | A2 full-vector false | A1 true retention | A2 true retention |
|---|---:|---:|---:|---:|
| default | 34 | 16 | 100% | 100% |
| strict125 | 17 | 12 | 96% | 100% |

A2 的 tight false 降幅为 `76.81% / 63.64%`，两个 scenario 都通过预注册
`>=50%` 降幅和 `>=95%` true retention Gate。H78-B 也通过：A2 相对 A1 在
default 多降 `26.09pp`，strict125 多降 `15.15pp`，true retention 没有下降。

## 2. 机制

`p_nav-eta` 误差范数中位约 `0.434 m`，但沿-goal 分量中位只有
`0.0655 / 0.0996 m`。Goal75 的 A1 因此对多数样本不触发；A2 使用完整向量
范数，修正中位约 `0.184 / 0.185 m`，继续消掉 lateral/vertical 误差造成的
false-success。

Goal77 的 proper goal-frame lateral 为 `0.3404 / -0.0021 m`，此前 Goal76
的 `1.2364 / 0.7618 m` 是世界水平距离，重复计入 signed offset；该口径已在
Goal76/77 文档中纠正。

## 3. 限制

A2 使用真实 `eta`，是 oracle correction，不能直接部署。它只证明“如果完整
终端估计误差可见，deadband 方向正确”。因此：

- 不把 A2 写成策略或正式结论；
- 不使用 `eta` 作为策略输入；
- 不因 A2 成功就启动 SAC/REDQ/DroQ；
- 不把 held-out `100–149` 打开。

## 4. 下一步

预注册 Goal79：observable full-vector correction predictor。

固定要求：

1. 只使用 terminal 时刻策略可见特征，不使用 `eta` 或 `p_nav-eta` 标签本身；
2. 用 seed-grouped train/validation（建议 leave-one-seed-out），禁止同 seed
   leakage；
3. 预测 `p_nav-eta` 的 goal-frame 三分量，再做与 A2 相同的 `0.25 m` deadband；
4. 在 default/strict125 同时检查 tight true retention `>=95%` 和 false 降幅
   `>=50%`；
5. 不扫模型族/阈值；若线性或固定小模型失败，就停止 bias correction 线，
   回到动作参数化、terminal reference/reward 和 capture 时序。

## 5. 算力

Goal78 为只读分析：10 个 run 级任务、`--jobs 10`、单线程 BLAS，墙钟约
`1.77 s`、CPU 约 `870%`、RSS 约 `36 MB`、0 GPU。Goal79 的 predictor 仍可先
用 CPU/并行完成；只有预测器通过验证并进入闭环闭环 arm 时，才启动 8 卡训练
吞吐基准。

## 6. 证据

- `doc/goals/goal78_full_vector_terminal_bias_correction_20261110.md`
- `doc/goals/goal78_result_20261110.md`
- `scripts/analyze_pc_goal78_full_vector_bias_correction.py`
- `tests/test_pc_goal78_full_vector_bias_correction.py`
- `artifacts/lowlevel/goal78_full_vector_bias_20261110/analysis/goal78.json`
- `artifacts/lowlevel/goal78_full_vector_bias_20261110/analysis/goal78.md`
