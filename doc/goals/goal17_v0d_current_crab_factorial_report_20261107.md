# Goal17 阶段 3–4 报告：encoded-current × ALOS crab 因子实验

日期：2026-11-07

状态：**停止，`FAIL_BC_FACTOR_GATE`。**

协议：`doc/goals/goal17_v0d_current_crab_factorial_plan_20261107.md`

结论先行：

- `T0/T1` 示范已完成，两份数据各 `262,144` transitions，且同 seed 可配对；
- 四格 BC 已完成 `91430/91431/91432` 三个训练 seed 的训练与 dev `20–39` 评估；
- `A11` 的 default RMSE ratio 为 `1.0139`，没有达到冻结的 `<=0.95`；
- `A10` 的 default RMSE ratio 为 `1.2019`，明显变差；
- 按预注册规则停止 BC 因子实验，不进入残差 PPO，不评估 held-out `100–149`；
- `A11` 的 current-zero/current-shuffle 都会显著变差，说明 current encoder 被策略使用，
  但这没有转化为相对 `A00` 的净收益。

## 1. 数据与实验矩阵

教师底座为 R2B baseline，只切换 `current_crab_enabled`；教师筛选选定 `T1`（crab on）。

| 数据集 | 教师 | samples | episodes | safety | current valid | SHA256 |
|---|---|---:|---:|---:|---:|---|
| `demos_T0` | crab off | `262,144` | `216` | `0` | `0.972645` | `4c9faf126a5d5bec8dc6b3f65766cef39215862789a1564b1c7636510eb1d7d2` |
| `demos_T1` | crab on | `262,144` | `219` | `0` | `0.972771` | `5e89cb7d790dcef8f2e9e8de1d1b99a1983a54420a7f21185f75b3b9454bd105` |

采集契约：`64 env × 4096 step`、训练 seed `90000–90063`、CPU、float64、
`path_dim=8`、`actor_command_dim=0`、无动作噪声。

四格定义：

| cell | 示范 | current encoder | ALOS crab |
|---|---|---|---|
| `A00` | `T0` | off | off |
| `A10` | `T0` | encoded | off |
| `A01` | `T1` | off | on |
| `A11` | `T1` | encoded | on |

## 2. BC 训练

冻结配方：state encoder `[128,128]`、actor `[256,128]`、path encoder `[64]`、
current encoder `[64]`、`d_z_total=64`、`n_in_reserved=8`、`n_z_reserved=4`、
batch size `4096`、`max_epochs=300`、patience `30`、lr `1e-3`、weight decay `1e-5`、
validation fraction `0.15`、device CUDA、DADE off。

| cell | train seed | best epoch | checkpoint SHA256 |
|---|---:|---:|---|
| `A00` | `91430` | `91` | `7fad9afd3577b819b783f4ab273dd8de4974e7c0a65abcab714fe364793b968d` |
| `A10` | `91430` | `50` | `2bef0bf239d6f085f6061386d6197faf38c6dba660b05e729f6fd11ac1529096` |
| `A01` | `91430` | `80` | `a95a74fa3ef118118a4d195acf856c9d8142620f082adf4210d738b694254ccf` |
| `A11` | `91430` | `44` | `03034eb31465fc3b56b2c944922d5d4c29a15c298e51bcbb56180eeaba2e796a` |
| `A00` | `91431` | `52` | `ec33cf2f3dc3f840e4cb429a677a63b735fcf46701f64807682de8b7aed653e7` |
| `A10` | `91431` | `44` | `a1e0d497e49160e79f29a7e846eced2e85a9ce842be63010d16862df56e2da7c` |
| `A01` | `91431` | `65` | `6203e7a8e0c57cb18c6c0a3d01f86eb46163e158adc5717f9a772d619da486b7` |
| `A11` | `91431` | `66` | `42e3c49bda3b4bd719a27b364ead852c9332d41137d032de7b404fc767f43a81` |
| `A00` | `91432` | `81` | `2bbd8c31eadc4f0ede2b61716d846b3e7baa29a894ff2b79a9509f84034aa5fd` |
| `A10` | `91432` | `70` | `b502a3a711feab36d53ab2f0cb0f651d74f81a85a5b3402893492082fd126a6e` |
| `A01` | `91432` | `90` | `6d014de281a0274cb352aecb66f2bb5f11494d96b20f14b8aee2084c5d95e8e0` |
| `A11` | `91432` | `54` | `8532a8a3772c85ee66f0085658df74d809bb1f6e4d2b7df6b734affa695e565c` |

## 3. dev default 结果

每个训练 seed 单独与同一训练 seed 的 `A00` 比较；下表为三个训练 seed 的 cell 均值。

| cell | default 3D RMSE | 相对 `A00` | success | timeout | safety | material saturation |
|---|---:|---:|---:|---:|---:|---:|
| `A00` | `8.0067 m` | `1.0000×` | `0.500` | `0.500` | `0` | `0.1471` |
| `A10` | `9.6232 m` | `1.2019×` | `0.450` | `0.550` | `0` | `0.1742` |
| `A01` | `8.7582 m` | `1.0939×` | `0.533` | `0.467` | `0` | `0.1604` |
| `A11` | `8.1176 m` | `1.0139×` | `0.550` | `0.450` | `0` | `0.1587` |

逐训练 seed 的 default RMSE：

| train seed | `A00` | `A10` | `A01` | `A11` |
|---|---:|---:|---:|---:|
| `91430` | `8.0384` | `10.1432` | `9.7369` | `7.7864` |
| `91431` | `8.4857` | `9.3654` | `8.1399` | `8.4986` |
| `91432` | `7.4960` | `9.3610` | `8.3978` | `8.0680` |

`A11` 在 `91430` 上看似改善，但该收益没有跨训练 seed 复现；`91431/91432` 都接近或
略差于各自 `A00`。因此不能把单 seed `0.96864` 的接近门禁解释为稳定改善。

## 4. 配对分析

配对差值使用同一评估 seed 的逐 episode 3D RMSE；95% CI 为均值附近的正态近似，
样本量为 `3 × 20 = 60`。

| 对比 | 均值 ΔRMSE | 95% CI | 判定 |
|---|---:|---:|---|
| `A10 - A00` | `+1.2257 m` | `[+0.0741, +2.3773]` | 显著变差 |
| `A11 - A00` | `-0.2663 m` | `[-1.2012, +0.6686]` | 无显著改善 |

注意：逐 episode RMSE 与 step-weighted 聚合 RMSE 不完全相同；上面的门禁判断使用
`summary.json` 的聚合 RMSE，配对 CI 用于判断差异是否稳定，二者结论一致。

## 5. 因子效应

以下均为 default RMSE 差值，负值才表示改善：

| 对比 | 均值 | `91430` | `91431` | `91432` |
|---|---:|---:|---:|---:|
| `A10 - A00`（current，crab off） | `+1.6165` | `+2.1048` | `+0.8797` | `+1.8651` |
| `A11 - A01`（current，crab on） | `-0.6406` | `-1.9506` | `+0.3587` | `-0.3298` |
| `A01 - A00`（crab，无 current） | `+0.7515` | `+1.6985` | `-0.3458` | `+0.9018` |
| `A11 - A10`（crab，有 current） | `-1.5056` | `-2.3568` | `-0.8669` | `-1.2930` |
| 交互项 | `-2.2571` | `-4.0553` | `-0.5210` | `-2.1949` |

解释：

- current encoder 在没有 crab 时明显有害；
- crab 在没有 current 时平均也有害，且跨训练 seed 不稳定；
- 两者同时开启时存在明显互补交互，`A11` 相对 `A10` 稳定更好；
- 但 `A11` 相对真正 baseline `A00` 仍没有净改善，因此交互结论不能升级为可部署收益。

## 6. current 消融

下表为相对同 cell default 的 3D RMSE 增量，正值表示变差：

| cell | 消融 | 均值 ΔRMSE | 95% CI | 逐训练 seed |
|---|---|---:|---:|---|
| `A10` | current-zero | `+0.6747` | `[-0.3779, +1.7272]` | `+1.0037, +1.5022, -0.4819` |
| `A10` | current-shuffle | `-0.5022` | `[-1.6381, +0.6337]` | `-0.1461, -0.1887, -1.1717` |
| `A11` | current-zero | `+1.5446` | `[+0.3626, +2.7266]` | `+1.3983, +1.8351, +1.4004` |
| `A11` | current-shuffle | `+1.1759` | `[+0.0016, +2.3503]` | `+1.7513, +1.1296, +0.6469` |

`A11` 的 current 分支确实被 actor 有效使用；`A10` 的消融方向不稳定，不能据此宣称
current 单独有效。需要强调：**被使用不等于有收益**。

## 7. strict125 与 path-zero

strict125 软目标（三个训练 seed 聚合）：

| cell | strict125 RMSE | success | timeout |
|---|---:|---:|---:|
| `A00` | `9.8730 m` | `0.283` | `0.717` |
| `A10` | `10.5731 m` | `0.317` | `0.683` |
| `A01` | `10.9282 m` | `0.267` | `0.733` |
| `A11` | `11.0852 m` | `0.233` | `0.767` |

path-zero/default 3D RMSE ratio（三个训练 seed 聚合）：

| cell | ratio | 逐训练 seed |
|---|---:|---|
| `A00` | `1.2116` | `1.0528, 1.2515, 1.3304` |
| `A10` | `1.2047` | `0.9297, 1.5161, 1.1684` |
| `A01` | `1.1342` | `0.8270, 1.3464, 1.2291` |
| `A11` | `1.2550` | `1.0827, 1.2959, 1.3863` |

path-zero 仍有一定恶化，但单 seed 和单 cell 的方向不稳定；它不是本轮通过或否决的主依据。

## 8. 门禁与决策

冻结 default 门禁：

- RMSE ratio `<=0.95`；
- success 绝对下降不超过 `0.05`；
- safety 绝对增加不超过 `0.02`。

| cell | ratio | success Δ | safety Δ | 结果 |
|---|---:|---:|---:|---|
| `A10` | `1.2019` | `-0.050` | `0` | **FAIL** |
| `A11` | `1.0139` | `+0.050` | `0` | **FAIL** |

决策：`stop_goal17_bc_gate_failed`。

- 不冻结任何 BC 作为 PPO 初始化；
- 不进入残差 PPO；
- 不运行 held-out `100–149`；
- 不通过继续扫 seed、learning rate、residual scale 或 PPO 来“救”失败的 BC；
- 若继续研究 current/crab，应另开 Goal，先解决 BC 动作表示或数据分布问题，并重新
  预注册实验与门禁。

## 9. 产物

根目录：

```text
artifacts/lowlevel/pc_v0d_current_crab_factorial_20261107/
```

主要产物：

```text
demos_T0/demos.manifest.json
demos_T1/demos.manifest.json
bc_A00/ ... bc_A11_seed91432/
eval/                         # 训练 seed 91430
eval_seed91431/               # 训练 seed 91431
eval_seed91432/               # 训练 seed 91432
stage4b_multiseed_summary.json
stage5_closure.json
```

日志：

```text
logs/goal17_multiseed_eval/
```

复现评估的核心命令：

```bash
PYTHONPATH=src:../settled_全局规划器/src .venv/bin/python scripts/evaluate_pc_bc.py \
  --config configs/lowlevel/odarrl_control_pc_c3_random_realistic_v0d_teacher_r2b_current.json \
  --checkpoint artifacts/lowlevel/pc_v0d_current_crab_factorial_20261107/bc_A11_seed91431/pc_bc_actor.pt \
  --output artifacts/lowlevel/pc_v0d_current_crab_factorial_20261107/eval_seed91431/A11/default \
  --seed-start 20 --seed-count 20 --device cpu --dtype float64 --path-dim 8
```

strict125 使用 `..._teacher_r2b_current_strict125.json`；path-zero 追加 `--path-zero`；
current 消融追加 `--current-ablation current-zero` 或 `current-shuffle`。

## 10. 限制与非阻塞缺口

- 服务器不可用，阶段 3/4 在本地运行；评估仍固定 CPU、float64、dev `20–39`。
- 教师选择 JSON 位于服务器路径，本地未保留该文件；其 SHA256 已写入采集 manifest。
- crab angle 未落盘，原始 trace 也未保留，本轮记录为“当前产物不可得”。
- 评估配对 CI 使用逐 episode RMSE 与正态近似；聚合 RMSE 是 step-weighted 口径。
- 三个训练 seed 只能说明当前 BC 配方下的稳定性，不代表更大训练预算或不同动作头。

## 11. 完成审计

2026-11-07 重新核对当前工作区后确认：

- `T0/T1` 示范各 `262,144` samples；初始 64 个环境的 seed、state、path、
  current estimate 逐位一致；`safety_events=0`；两份数据集 SHA256 与阶段 3 报告一致；
- 四组配置 SHA256 与计划文档表格一致；
- 12 个 checkpoint 均满足 `z_t_dim=64`、`actor_top_level_input_dim=128`、
  `actor_command_dim=0`、`path_dim=8`，且带 schema hash；
- 所有 Goal17 dev 评估均使用 seed `20–39`，safety termination 均为 `0`；
- 未发现任何 held-out `100–149` 的 Goal17 评估产物；
- 相关测试重新执行：`115 passed`；
- `ruff check`：`All checks passed!`；
- 当前没有 Goal17 teacher/BC/PPO/eval 进程在运行。
- 机器可读收口文件：`stage5_closure.json`，明确记录 `passed_cells=[]`、
  `enter_residual_ppo=false`、`heldout_evaluated=false`。

因此，Goal17 的预注册 BC 因子实验路径已经完整收口；阶段 6 未执行是门禁失败触发的
预期停止条件，不是遗留任务。
