# Goal25 D3 初始化微调报告

日期：2026-11-08  
状态：**FAIL；success 保持/提升，但两个场景的 pre-goal RMSE 均未达到门禁**  
范围：dev seeds `20–59`；未使用 held-out `100–149`  
执行环境：本地工作站，D3 训练使用 `mjlab45` CUDA 环境，评估使用 CPU/float64；
`formal_remote_server=false`

## 1. 结论

D3 验证了“D0 初始化 + D2 数据低学习率微调”可以保住 success，甚至改善 strict125，
但没有修复 pre-goal 精度，因此预注册门禁为 **FAIL**。

| scenario | D2 success | D3 success | success delta | D2 pre-goal | D3 pre-goal | RMSE ratio |
|---|---:|---:|---:|---:|---:|---:|
| default | 0.7250 | 0.7250 | +0.0000 | 2.788330 | 2.849170 | +2.18% |
| strict125 | 0.3750 | 0.5000 | +0.1250 | 2.764764 | 2.829135 | +2.33% |

两个场景都超过 `2.65 m / 2.70 m` 的 pre-goal 上限，且相对 D2 反而变差约 `2.2%`。
按 Goal25 Research Contract，D3 不通过即停止：

- `phase3_stress_status = not_started_by_gate`；
- `phase4_ppo_status = ppo_not_started`；
- 不重跑 S10/S20，不扫描 stress 比例、lr、epoch 或 seed。

## 2. D3 训练

| field | value |
|---|---|
| 初始化 | D0 checkpoint |
| 初始化 SHA256 | `c423b2fc2385db4c49c5cc78d95043e25bae3707cd67a28bc68584997388825b` |
| 数据 | D2 merged，393,216 samples |
| 数据 SHA256 | `fe7feeb6db0e9492212e405b2bd7780965f613188b458633644159c472577d91` |
| fine-tune lr | `2e-4` |
| max epochs / patience | `120 / 20` |
| best epoch | `10` |
| epochs completed | `31` |
| train / validation samples | `332,112 / 61,104` |
| best validation loss | `0.048944328` |
| D3 checkpoint SHA256 | `7430b26308b701565a52a92ed31c382ccca9cf1721d183ab856e3fafd9f4ab73` |

训练 manifest 明确记录 `init_checkpoint_path`、`init_checkpoint_sha256`、
`fine_tune_lr` 和完整结构恢复信息。path encoder alpha 为 `1.0`，连续动作头，8D path，
未启用 current encoder。

## 3. D0 / D2 / D3 对比

### default

| arm | success | pre-goal RMSE | full RMSE | plane miss | safety | saturation |
|---|---:|---:|---:|---:|---:|---:|
| D0 | 0.5750 | 2.736606 | 2.690092 | 0.4250 | 0.0000 | 0.199279 |
| D2 | 0.7250 | 2.788330 | 2.710160 | 0.2750 | 0.0000 | 0.215183 |
| D3 | 0.7250 | 2.849170 | 2.763411 | 0.2750 | 0.0000 | 0.191812 |

### strict125

| arm | success | pre-goal RMSE | full RMSE | plane miss | safety | saturation |
|---|---:|---:|---:|---:|---:|---:|
| D0 | 0.3000 | 2.720555 | 2.672551 | 0.7000 | 0.0000 | 0.198136 |
| D2 | 0.3750 | 2.764764 | 2.693556 | 0.6250 | 0.0000 | 0.213722 |
| D3 | 0.5000 | 2.829135 | 2.752011 | 0.5000 | 0.0000 | 0.191880 |

D3 的 strict125 success 从 D2 的 `0.375` 提升到 `0.500`，plane miss 从 `0.625` 降到
`0.500`，saturation 也下降；但 pre-goal RMSE 仍变差。default success 不变，但 RMSE
同样变差。

## 4. 逐 seed 变化

相对 D2 normal：

| scenario | RMSE 改善 seeds | RMSE 变差 seeds | median RMSE delta | success gain | success loss |
|---|---:|---:|---:|---:|---:|
| default | 16 | 24 | +0.0253 m | 3 | 3 |
| strict125 | 17 | 23 | +0.0216 m | 5 | 0 |

D3 明显修复了 seed `22`（default `-1.5711 m`、strict125 `-1.5756 m`），这也是
Phase 1 中最大的退化 outlier。但 seed `25`、`41`、`27`、`47` 等反而变差，抵消了
收益。说明低学习率微调确实改变了后半程行为，但只完成了局部修复，没有形成整体改进。

## 5. 门禁逐项

| check | default | strict125 |
|---|---:|---:|
| pre-goal <= `2.65 / 2.70 m` | FAIL | FAIL |
| success >= `0.70 / 0.40` | PASS | PASS |
| 相对 D2 success drop <= `0.03` | PASS | PASS |
| safety delta <= `0.02` | PASS | PASS |
| path-zero degradation >= `5%` | PASS | PASS |
| finite + manifest | PASS | PASS |
| **overall** | **FAIL** | **FAIL** |

门禁文件：`/dev/shm/pc_v1_goal25_d2_repair_20261108/d3_gate.json`  
摘要：`/dev/shm/pc_v1_goal25_d2_repair_20261108/d3_gate.md`

## 6. 解释

H1“D0 初始化可以提供更稳定的路径特征，从而降低 pre-goal RMSE”被证伪。更准确的
观察是：

- D0 初始化保留了 D2 的 success 能力，并改善了 strict125 的终点捕获；
- 但它在路径中后段引入了新的偏差，整体 RMSE 反而略差；
- D3 不是简单的欠拟合：best epoch 只有 `10`，但验证损失随后稳定，继续训练没有
  继续改善；
- 该微调轴没有再扫描 lr/epoch 的价值，继续试参会违反预注册合同。

## 7. 产物

- 训练：`/dev/shm/pc_v1_goal25_d2_repair_20261108/bc_d3/`
- 评估：`/dev/shm/pc_v1_goal25_d2_repair_20261108/eval_matrix_d3/`
- 门禁：`/dev/shm/pc_v1_goal25_d2_repair_20261108/d3_gate.json`
- 训练日志：`/dev/shm/pc_v1_goal25_d2_repair_20261108/bc_d3_train.log`
- 评估日志：`/dev/shm/pc_v1_goal25_d2_repair_20261108/eval_matrix_d3.log`
- 评估矩阵：
  `configs/lowlevel/eval/pc_v1_nominal_eval_matrix_goal25_d3_20261108.json`

## 8. 声明

- 未使用 held-out `100–149`；
- 未重跑 S10/S20，未扫描 stress 比例；
- 未启动 stress 诊断或残差 PPO；
- 未覆盖 D0/D2 的旧 checkpoint、数据、manifest 或 evaluation；
- 未使用 `git add -A`。
