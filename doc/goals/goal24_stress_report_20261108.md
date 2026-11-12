# Goal24 Phase 3 报告：S10/S20 stress 增广

日期：2026-11-08  
状态：**FAIL；S10/S20 均未通过 nominal 与 stress 联合门禁；Phase 4 保持 ppo_not_started**  
范围：dev seeds `20–59`（40 seeds）；未使用 held-out `100–149`

## 1. 结论

本轮只验证一个受控问题：在 D2 targeted DAgger 配方中加入 `10%` 或 `20%` 的
stress 教师数据，能否在不破坏 nominal 精度的前提下提升 `realistic_stress` 表现。

结论是否定的：

- S10 的 nominal pre-goal RMSE 相对 D2 在 default/strict125 分别变差 `+2.85%`、
  `+2.98%`，均超过 `2%` 上限；
- S20 的 nominal pre-goal RMSE 分别变差 `+5.00%`、`+4.94%`，同样失败；
- S10 在 stress 下默认场景 success 下降 `0.05`、strict125 下降 `0.125`，pre-goal
  RMSE 也分别变差 `0.0949 m`、`0.0939 m`；
- S20 只有在 stress/default 有 `+0.025` success，未达到 `+0.05`，strict125 为
  `-0.025`；两个场景的 pre-goal RMSE 都变差约 `0.11 m`；
- safety 没有增加，但不足以抵消 nominal 精度和 stress 收益的失败。

因此 Phase 3 判定为 **FAIL**。按预注册停止规则，不启动 Phase 4 残差 PPO；
`phase4_ppo_status = ppo_not_started`。不扫数据比例、lr、epoch 或 seed。

## 2. 运行环境

- 执行位置：`zhulab3-AS-4125GS-TNRT`，8 × RTX 4090 训练服务器；
- GPU：`CUDA_VISIBLE_DEVICES=1`；
- Python：`/data/yhx/underwater-robotics/envs/conda_envs/oceansim/bin/python`；
- PyTorch：`2.7.0+cu126`；
- 训练设备：CUDA；评估设备：CPU/float64；
- 远端根目录：
  `/data/yhx/underwater-robotics/src/plan_p1/goal24_stress_20261108/`；
- 本地只读证据镜像：
  `/dev/shm/pc_v1_goal24_stress_server_20261108/`；
- 本报告结果是服务器正式结果，不是 local smoke；本地没有对应的正式 S10/S20 运行。

## 3. 数据配方

| arm | nominal teacher | targeted DAgger | stress | 总样本 | stress 占比 | episodes |
|---|---:|---:|---:|---:|---:|---:|
| S10 | 311,296 | 40,960 | 40,960 | 393,216 | 10.4% | 454 |
| S20 | 229,376 | 81,920 | 81,920 | 393,216 | 20.8% | 454 |

数据哈希：

| arm | dataset SHA256 |
|---|---|
| S10 | `74b6afb050f592da9ca03816108959fe8f73362869e5fcc2c5cd4519629b8d71` |
| S20 | `f7427caf85baac5850d82c72799d561f65edb26b04ea8db984ac168120e495d0` |

stress 原始数据：

- default：`ec06f5515806dc18a799eb412f92d077544c51caa03bc49075e916a4e292f6f7`；
- strict125：`fd056c839b64d5b57cc5d54b3fd4db242dcdea8bc32c963eb96ce8ce79aa3812`；
- stress 采集 seeds：default `92000–92009`、strict125 `92100–92109`；
- 与 dev `20–59`、DAgger `200–239`、held-out `100–149` 完全分离。

## 4. 训练

配方与 D2 完全一致：state `128,128`、actor `256,128`、`d_z=64`、path encoder `64`、
batch `4096`、lr `1e-3`、weight decay `1e-5`、max epochs `300`、patience `30`、
seed `0`、8D path、continuous action head。

| arm | best epoch | epochs completed | best validation loss | checkpoint SHA256 |
|---|---:|---:|---:|---|
| S10 | 57 | 88 | 0.037456520 | `a1df2ab1c000528a4f9a702823b986f51117d5e13165d1da0e7bbf931ed0937e` |
| S20 | 78 | 109 | 0.046801154 | `5aa0021fa3ec628d7aaab280bf8c8ebca64dae99952b62d47888a30897e705ea` |

## 5. nominal 评估

参考基线是 Goal24 Phase 2 的 D2。

| arm | scenario | success | pre-goal RMSE | full RMSE | plane miss | safety | saturation |
|---|---|---:|---:|---:|---:|---:|---:|
| D2 | default | 0.7250 | 2.788330 | 2.710160 | 0.2750 | 0.0000 | 0.215183 |
| S10 | default | 0.7000 | 2.867833 | 2.792268 | 0.3000 | 0.0000 | 0.198931 |
| S20 | default | 0.7750 | 2.927846 | 2.816178 | 0.2250 | 0.0000 | 0.204836 |
| D2 | strict125 | 0.3750 | 2.764764 | 2.693556 | 0.6250 | 0.0000 | 0.213722 |
| S10 | strict125 | 0.4250 | 2.847139 | 2.780313 | 0.5750 | 0.0000 | 0.198489 |
| S20 | strict125 | 0.4500 | 2.901345 | 2.799029 | 0.5500 | 0.0000 | 0.203145 |

相对 D2：

| arm | scenario | success delta | pre-goal RMSE delta | 2% 门禁 |
|---|---|---:|---:|---|
| S10 | default | -0.0250 | +2.85% | FAIL |
| S10 | strict125 | +0.0500 | +2.98% | FAIL |
| S20 | default | +0.0500 | +5.00% | FAIL |
| S20 | strict125 | +0.0750 | +4.94% | FAIL |

这是本轮最关键的负结果：S20 确实提升了 success，但代价是 pre-goal RMSE 接近 `5%`
的退化；S10 的参数更保守，仍超过了 `2%` 精度门禁。

## 6. stress 评估

参考基线是 D2 在同一 `realistic_stress` profile 下的 40-seed 评估。

| arm | scenario | success | pre-goal RMSE | full RMSE | plane miss | safety | saturation |
|---|---|---:|---:|---:|---:|---:|---:|
| D2 | default | 0.4750 | 3.122679 | 3.095034 | 0.5250 | 0.0000 | 0.343315 |
| S10 | default | 0.4250 | 3.217551 | 3.193596 | 0.5750 | 0.0000 | 0.301003 |
| S20 | default | 0.5000 | 3.235732 | 3.180174 | 0.5000 | 0.0000 | 0.296420 |
| D2 | strict125 | 0.2750 | 3.112600 | 3.092447 | 0.7250 | 0.0000 | 0.342572 |
| S10 | strict125 | 0.1500 | 3.206523 | 3.180307 | 0.8500 | 0.0000 | 0.300392 |
| S20 | strict125 | 0.2500 | 3.218228 | 3.168537 | 0.7500 | 0.0000 | 0.295691 |

相对 D2 stress：

| arm | scenario | success gain | pre-goal improvement | stress 门禁 |
|---|---|---:|---:|---|
| S10 | default | -0.0500 | -0.0949 m | FAIL |
| S10 | strict125 | -0.1250 | -0.0939 m | FAIL |
| S20 | default | +0.0250 | -0.1131 m | FAIL（success <0.05 且 RMSE 变差） |
| S20 | strict125 | -0.0250 | -0.1056 m | FAIL |

## 7. 门禁逐项判定

预注册门禁：

1. nominal pre-goal RMSE 不比 D2 差超过 `2%`；
2. nominal success 下降不超过 `0.03`；
3. stress success 至少 `+0.05`，或 stress pre-goal RMSE 至少改善 `0.10 m`；
4. safety 增量不超过 `0.02`。

| arm | scenario | nominal RMSE | nominal success | stress gain | safety | overall |
|---|---|---|---|---|---|---|
| S10 | default | FAIL | PASS | FAIL | PASS | FAIL |
| S10 | strict125 | FAIL | PASS | FAIL | PASS | FAIL |
| S20 | default | FAIL | PASS | FAIL | PASS | FAIL |
| S20 | strict125 | FAIL | PASS | FAIL | PASS | FAIL |

- `phase3_stress_status = failed`；
- `phase4_ppo_status = ppo_not_started`；
- 没有触发 held-out，没有扩大 seed，没有扫超参数，没有覆盖旧配置或旧数据。

## 8. path-zero 诊断

path-zero 仍造成明显退化，说明路径输入没有被绕过：

| arm | profile | scenario | normal pre-goal | path-zero pre-goal | delta |
|---|---|---|---:|---:|---:|
| S10 | nominal | default | 2.867833 | 7.408619 | +158.34% |
| S10 | nominal | strict125 | 2.847139 | 7.408619 | +160.21% |
| S20 | nominal | default | 2.927846 | 9.649861 | +229.59% |
| S20 | nominal | strict125 | 2.901345 | 9.646525 | +232.48% |
| S10 | stress | default | 3.217551 | 7.595534 | +136.07% |
| S20 | stress | default | 3.235732 | 9.938308 | +207.14% |

这项只用于确认路径分支依赖，不是本轮的主门禁。

## 9. 解释

stress 数据并非完全无效：S20 在 stress/default 提升了 `0.025` success，并把
stress saturation 从 D2 的约 `0.343` 降到约 `0.296`。但收益没有达到预注册阈值，
且 strict125 出现反向退化。

更关键的是 nominal 代价：S10/S20 都牺牲了约 `3–5%` 的 pre-goal RMSE。对当前
任务，`3D RMSE` 是主目标，单纯提高 success 不能补偿路径精度损失。因此不能把
S20 作为“更好”的候选，也不应通过放宽 2% 门禁来让它进入 PPO。

## 10. 产物

服务器正式产物：

```text
/data/yhx/underwater-robotics/src/plan_p1/goal24_stress_20261108/
  data/
    bc_s10/
    bc_s20/
    eval_nominal_s10/
    eval_nominal_s20/
    eval_stress_s10/
    eval_stress_s20/
    goal24_stress_gate.json
    goal24_stress_gate.md
```

本地只读证据镜像：

```text
/dev/shm/pc_v1_goal24_stress_server_20261108/
  bc_s10/
  bc_s20/
  eval_nominal_s10/
  eval_nominal_s20/
  eval_stress_s10/
  eval_stress_s20/
  goal24_stress_gate.json
  goal24_stress_gate.md
```

评估矩阵哈希：

- nominal matrix：`2d47b99208f615e2c03e797771ad00df42351de8012b9451a6dfcbf9bb39764a`；
- stress matrix：`2d5f0f9109e4ec1c8e6d820dc12e7774318f39dbf484ee297ab5cad609b7f508`。

## 11. 下一步

1. 保留 D2 作为 Goal24 中最好的数据配方，但明确它仍是 `partial_improvement`；
2. 停止当前 stress 比例轴，不再尝试 15%/25%/30% 等未预注册扫描；
3. 不启动残差 PPO；PPO 的前提仍是强 BC 底座，而当前 D2 的 pre-goal RMSE 与
   strict125 success 尚未达标；
4. 若继续研究，应另开 Goal25 的 D2 pre-goal 修复或教师/动作表征假设，而不是在
   本轮结果上事后放宽门禁。
