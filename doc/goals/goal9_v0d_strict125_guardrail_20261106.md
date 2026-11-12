# Goal9 V-0D strict125 护栏对照预注册

日期：2026-11-06

状态：已完成；主门禁失败。结果见
`goal9_v0d_strict125_guardrail_result_20261106.md`。

## 目的

补齐 V-0D 在 `strict125` 目标半径下的护栏对照。此前 default 评估已经看出
`R-PPO-0D` 相对 `R-BC-0D` 没有达到 `<= 0.95` 的 3D RMSE 门禁；本实验不再尝试
通过换 checkpoint、换 seed 或改阈值来“救”负结果，只检查更严格目标半径是否改变
基础结论，并给后续正式对比留下同口径基线。

## 固定条件

- 服务器工程目录：
  `/data/yhx/underwater-robotics/src/plan_p1/settled_局部规划器_反应式`
- 环境配置：
  `configs/lowlevel/odarrl_control_pc_c3_random_realistic_v1_learned_speed_strict125.json`
- 环境配置 SHA256：
  `c47536b7025f60adddedc323050f8f8be58dbf31b566655a8e96c2c087301b15`
- 评估 seed：固定 `0–19`，不得换 seed，不得按结果筛选 seed。
- 计算口径：`--device cpu --dtype float64`。
- 评估 seed、环境随机流、路径、观测归一化和动作接口在组间保持一致。
- 不训练、不选择新 checkpoint、不修改配置、不覆盖任何既有 artifacts、summary 或日志。
- 新产物只写入本预注册指定的新目录；启动前必须检查目录不存在。

`strict125` 与 V-0D 默认环境相比，控制语义只差目标半径：

- `plant.goal_radius_m`：`2.0 -> 1.25`；
- `experiment.name` 和 `experiment.output` 仅为元数据差异，不参与控制。

奖励、观测、归一化、传感器、执行器、残差和终止契约均相同。评估报告必须把
`strict125` 写成环境标签，不得伪造评估器中不存在的 `strict125` 数值字段。

## 固定 checkpoint

| 组别 | checkpoint | SHA256 |
|---|---|---:|
| `R-BC-3D` | `artifacts/lowlevel/pc_bc_c0_c1_c2_c3random_v1_dz64_20261103/pc_bc_actor.pt` | `f56c359648a99e3fc8fde51f2e529af5c9916371450ff0d7a942c0f620d33936` |
| `R-BC-0D` | `artifacts/lowlevel/pc_v0d_bc0d_20261106/pc_bc_actor.pt` | `bddb393175f8c4594746290abdf465ab9064e294b116e20e174d2f96335e4cf0` |
| `R-PPO-0D-screen` | `artifacts/lowlevel/pc_v0d_pure_path_20261106/R-PPO-0D-screen/final.pt` | `eb1fad7fd5791cbdaae6c3d42e2ff05edbd1837e3fcfc71c0a933320a7efe661` |

`R-PPO-0D-screen` 的训练配置：

`configs/lowlevel/odarrl_control_ppo_pc_v0d_bc0d_frozen_residual_server4096_dz64.json`

训练配置 SHA256：

`18fbaa13d6b8c4df632e038b675239b712b53c191f13aa741a364afb1dfd0d90`

checkpoint 在评估前固定；不得根据 strict125 的结果替换为 actor-LR、scale 或监督残差
checkpoint。后述分支如需补充，只能作为单独的诊断表，不能混入主表。

## 评估矩阵

| 组别 | 脚本 | 关键参数 | 输出目录 |
|---|---|---|---|
| `R-BC-3D/default` | `evaluate_pc_bc.py` | `--path-dim 0` | `artifacts/lowlevel/pc_v0d_strict125_gate_20261106/R-BC-3D/default` |
| `R-BC-0D/default` | `evaluate_pc_bc.py` | `--path-dim 8` | `artifacts/lowlevel/pc_v0d_strict125_gate_20261106/R-BC-0D/default` |
| `R-BC-0D/path_zero` | `evaluate_pc_bc.py` | `--path-dim 8 --path-zero` | `artifacts/lowlevel/pc_v0d_strict125_gate_20261106/R-BC-0D/path_zero` |
| `R-PPO/default` | `evaluate_pc_ppo.py` | 默认 `policy` | `artifacts/lowlevel/pc_v0d_strict125_gate_20261106/R-PPO-0D-screen/default` |
| `R-PPO/frozen_base` | `evaluate_pc_ppo.py` | `--action-mode frozen-base` | `artifacts/lowlevel/pc_v0d_strict125_gate_20261106/R-PPO-0D-screen/frozen_base` |
| `R-PPO/path_zero` | `evaluate_pc_ppo.py` | `--path-ablation path-zero` | `artifacts/lowlevel/pc_v0d_strict125_gate_20261106/R-PPO-0D-screen/path_zero` |

`R-PPO/frozen_base` 与 `R-BC-0D/default` 使用同一个冻结 BC checkpoint，但前者走 PPO
同 harness，后者走 BC 专用评估器；两者都必须保留，不能互相替代。

## 服务器命令

先 smoke 一个 seed，确认 strict125 配置能通过环境 override 校验：

```bash
source /data/yhx/underwater-robotics/tools/underwater_8x4090_env.sh
export PYTHONPATH=src:../settled_全局规划器/src
PY=/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python
CFG=configs/lowlevel/odarrl_control_pc_c3_random_realistic_v1_learned_speed_strict125.json
OUT=artifacts/lowlevel/pc_v0d_strict125_gate_20261106_smoke

$PY scripts/evaluate_pc_bc.py \
  --config "$CFG" \
  --checkpoint artifacts/lowlevel/pc_v0d_bc0d_20261106/pc_bc_actor.pt \
  --output "$OUT/R-BC-0D/default" \
  --seed-start 0 --seed-count 1 \
  --device cpu --dtype float64 --path-dim 8

$PY scripts/evaluate_pc_ppo.py \
  --training-config configs/lowlevel/odarrl_control_ppo_pc_v0d_bc0d_frozen_residual_server4096_dz64.json \
  --checkpoint artifacts/lowlevel/pc_v0d_pure_path_20261106/R-PPO-0D-screen/final.pt \
  --environment-config "$CFG" \
  --output "$OUT/R-PPO/default" \
  --seed-start 0 --seed-count 1 \
  --device cpu --dtype float64
```

smoke 通过后，正式 20-seed 评估使用相同命令，只把 `--seed-count` 改为 `20`，并把
输出根目录改为 `artifacts/lowlevel/pc_v0d_strict125_gate_20261106`。正式启动前逐组
执行 `test ! -e <output>`；任何输出目录已存在就停止，不得覆盖。

## 指标与门禁

主指标：

- `path_three_d_rmse_m`（3D RMSE，第一优先级）；
- `success_rate`；
- `timeout_rate`；
- `safety_termination_rate`；
- `path_completion_ratio_mean`。

记录但不等同于主门禁：

- `material_saturation_rate` 只作风险信息，用户已明确其优先级较低；
- 残差幅度、动作饱和和 path-zero 归因只用于解释机制。

严格沿用既有 V-0D 门禁：

- `R-PPO-0D / R-BC-0D` 的 3D RMSE ratio `<= 0.95`；
- `R-PPO-0D / R-BC-3D` 的 3D RMSE ratio `<= 1.05`；
- `path-zero` 相对 default 的 3D RMSE 恶化至少 `5%`；
- success、timeout、safety 的绝对恶化均不超过 `0.02`。

报告必须同时给出逐 seed 配对均值、95% CI、最差 seed 和失败尾部；不得只报总体均值。
如果结果仍失败，结论只能是“该固定 checkpoint 在 strict125 下未通过”，不得据此
反向调参或声称架构已被证伪。

## 完成标准

- [x] smoke 完成，配置与 checkpoint 校验通过；
- [x] 六个正式 20-seed 评估全部完成；
- [x] 记录每个 summary 的路径、seed 范围、checkpoint SHA 和环境 SHA；
- [x] 计算配对 ratio、success/timeout/safety 绝对差和 path-zero 恶化；
- [x] 写入结果文档；负结果必须原样保留；
- [x] 不修改旧 artifacts、旧配置、旧报告和旧 checkpoint。
