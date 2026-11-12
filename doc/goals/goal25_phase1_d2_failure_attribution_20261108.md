# Goal25 Phase 1：D2 相对 D0 的只读失败归因

日期：2026-11-08  
范围：dev seeds `20–59`；未使用 held-out `100–149`  
状态：**完成；D2 的 success 增益真实，但 pre-goal 退化集中在少数后半程 seed**

## 1. 结论

D2 相对 D0 的 success 提升是明确的：

- default：`0.5750 → 0.7250`，增加 `0.1500`；
- strict125：`0.3000 → 0.3750`，增加 `0.0750`。

但 D2 的 pre-goal RMSE 同时变差：

- default：`2.736606 → 2.788330 m`，增加 `1.89%`；
- strict125：`2.720555 → 2.764764 m`，增加 `1.62%`。

逐 seed 配对后，这不是“所有 seed 一起变差”，而是：

- default 有 `8` 个 seed 由失败转为成功，只有 `2` 个 seed 由成功转为失败；
- strict125 有 `9` 个 seed 由失败转为成功，同时有 `6` 个 seed 由成功转为失败；
- default 的 RMSE 退化主要来自 seed `22`，其次为 `46`、`47`、`55`、`56`；
- 路径分段误差显示，D2 的明显恶化集中在 `50–100%` 的后半程，尤其 seed `22` 的
  `50–75%` 和 `75–100%` 两段。

因此 D2 不是简单的“路径跟踪整体变差”，而是用少量后半程轨迹质量换取了更多终点
捕获。D3 的初始化微调应优先修复后半程发散，而不是重新调整起点段。

## 2. 数据来源

- D0 评估：`/dev/shm/pc_v1_data_recipe_20261108/eval_matrix_d0/comparison.json`
- D2 评估：`/dev/shm/pc_v1_data_recipe_20261108/eval_matrix_d2/comparison.json`
- 逐 seed 配对：`/dev/shm/pc_v1_goal25_d2_repair_20261108/audit/phase1_d2_vs_d0_attribution.json`
- 40-seed 路径分段 trace：
  `/dev/shm/pc_v1_goal25_d2_repair_20261108/audit/phase1_trace_attribution.json`
- 评估矩阵：
  `configs/lowlevel/eval/pc_v1_nominal_eval_matrix_teacher_fixed_20261108.json`

## 3. Success 转换

| scenario | both success | D0 fail → D2 success | D0 success → D2 fail | both fail |
|---|---:|---:|---:|---:|
| default | 21 | 8 | 2 | 9 |
| strict125 | 6 | 9 | 6 | 19 |

default 的成功转换明显正向；strict125 虽然净增 `3` 个 seed，但同时损失了 `6` 个原
成功 seed，说明 strict125 更接近“成功与失败互相交换”，而不是稳定增益。

default 的失败转换 seed 为 `22`、`47`；strict125 的失败转换 seed 为
`26`、`27`、`49`、`54`、`55`、`57`。

## 4. Pre-goal 退化最大的 seed

### default

| seed | D0 success | D2 success | D0 pre-goal | D2 pre-goal | delta |
|---:|---|---|---:|---:|---:|
| 22 | true | false | 1.9600 | 3.7584 | +1.7984 |
| 46 | false | false | 2.5525 | 2.8926 | +0.3401 |
| 47 | true | false | 5.5856 | 5.9023 | +0.3168 |
| 55 | true | true | 1.9135 | 2.1184 | +0.2049 |
| 56 | false | true | 3.6505 | 3.8481 | +0.1976 |

### strict125

| seed | D0 success | D2 success | D0 pre-goal | D2 pre-goal | delta |
|---:|---|---|---:|---:|---:|
| 22 | false | false | 1.9599 | 3.7590 | +1.7991 |
| 47 | false | false | 5.4930 | 5.8840 | +0.3911 |
| 46 | false | false | 2.5525 | 2.8926 | +0.3401 |
| 55 | true | false | 1.9134 | 2.1465 | +0.2331 |
| 40 | false | false | 2.4932 | 2.6841 | +0.1909 |

seed `22` 在两个场景下都出现约 `1.8 m` 的退化，是总体 RMSE 变差的最主要原因。

## 5. 路径分段 trace

使用原始 40-seed 同批评估，默认关闭的 `--write-trace` 仅用于本诊断。分段按真实
投影在参考路径上的进度划分，只统计到达终点前的有效步。

### default

| progress | D0 RMSE | D2 RMSE | D2 - D0 |
|---|---:|---:|---:|
| 0–25% | 1.1616 | 1.1466 | -0.0151 |
| 25–50% | 3.2007 | 3.2935 | +0.0929 |
| 50–75% | 3.2718 | 3.3341 | +0.0623 |
| 75–100% | 3.0887 | 3.1442 | +0.0555 |

### strict125

| progress | D0 RMSE | D2 RMSE | D2 - D0 |
|---|---:|---:|---:|
| 0–25% | 1.1616 | 1.1466 | -0.0151 |
| 25–50% | 3.2007 | 3.2935 | +0.0929 |
| 50–75% | 3.2718 | 3.3341 | +0.0623 |
| 75–100% | 3.0119 | 3.0297 | +0.0178 |

D2 在起点 `0–25%` 略好，但从 `25%` 之后开始变差。seed 层面的分段差均值在最后一段
为负，但 pooled RMSE 仍被 seed `22` 的长尾大幅拉高，说明聚合值对单个发散 seed
非常敏感。

## 6. 其余诊断

- default 中 D2 的 material saturation 均值从 `0.1993` 升到 `0.2152`，但失败 seed
  中并没有统一出现饱和升高；seed `22` 饱和率约 `0.41`，seed `47` 仅约 `0.09`。
- default 成功转换 seed 的 final progress 平均下降约 `1.20 m`，但终点捕获率提高，
  说明 D2 更早地进入终点捕获流程，代价是部分路径后段精度。
- D2 的 path-zero 退化仍明显大于 D0，说明路径输入没有被绕过；该结果只作诊断。

## 7. 对 D3 的直接含义

1. D3 不应从零训练，也不应只用 D0 数据；必须保持 D2 的失败恢复样本。
2. 低学习率微调是合理的，因为 D2 的退化是局部、seed 相关的后半程行为，不是整体
   特征崩塌。
3. D3 评估必须保留逐 seed 配对和路径分段 trace；只看 40-seed 聚合 RMSE 会被
   seed `22` 主导。
4. 若 D3 只修复 seed `22` 而损失 default success，仍不能通过门禁；必须同时保留
   成功转换收益。

## 8. 验证

- `--write-trace` 默认关闭，旧评估输出不变；
- 40-seed trace 使用 dev `20–59`，未运行 held-out；
- 新增 trace 单测通过；
- D0/D2 checkpoint 与数据 SHA256 已在 Phase 0 audit 中核对。
