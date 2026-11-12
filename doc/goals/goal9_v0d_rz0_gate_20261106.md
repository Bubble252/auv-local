# Goal9 V-0D R-Z0 零残差门禁

日期：2026-11-06

## 结论

`R-Z0-0D` 通过零残差门禁，允许进入 `R-PPO-0D` screen。

本门禁只证明冻结 BC、路径编码器和残差部署链路工作正常，不证明 PPO 已经带来收益。

## 发现并修复的问题

首次 R-Z0 评估在 `policy.base_action()` 处失败：

```text
RuntimeError: frozen BC path encoder has not been loaded
```

根因：冻结 BC 的 path encoder 权重保存在 `model_state_dict`，但标量
`frozen_bc_path_encoder_alpha` 不是参数或 buffer，没有随 checkpoint 恢复。

非破坏性修复：

- 新增 `restore_pc_bc_frozen_path_encoder_alpha()`，从训练配置引用的 BC checkpoint
  校验并恢复 alpha；
- 评估时先校验 BC checkpoint SHA，再恢复 alpha；
- 新 PPO checkpoint 的元数据记录冻结路径编码器的维度、输出维度和 alpha；
- 新增 `--action-mode {policy,frozen-base}`，默认 `policy`，旧行为不变；
- 增加回归测试，targeted suite 在服务器上为 `55 passed`。

受影响文件在服务器上的原版本已备份到：

```text
.server_backup/v0d_alpha_fix_20261106/
```

## 训练配置与预算

- 训练配置：
  `configs/lowlevel/odarrl_control_ppo_pc_v0d_bc0d_frozen_residual_server4096_dz64.json`
- 配置 SHA256：
  `18fbaa13d6b8c4df632e038b675239b712b53c191f13aa741a364afb1dfd0d90`
- R-Z0 初始化：`64 envs × 256 steps × 1 iteration`
- 初始残差：`max |residual| = 0`
- `model_initial.pt` SHA256：
  `b6ac7b3f3d7840783bbc8150b0c286657a70463a83e0a7deb7e701c97ceb6898`

## 等价性证据

20-seed 评估：

```text
artifacts/lowlevel/pc_v0d_bc_gate_20261106/R-Z0-0D/default/
```

| 检查项 | 结果 |
|---|---:|
| `max abs residual_action` | `0.0` |
| `max abs(requested_action - base_action)` | `0.0` |
| 冻结 BC 参数最大漂移 | `0.0` |
| 冻结 BC state entry 数 | `18` |
| 缺失冻结 BC entry | `0` |

同一 checkpoint 上额外运行 1-seed `policy` 与 `frozen-base` 对照：

```text
artifacts/lowlevel/pc_v0d_bc_gate_20261106/R-Z0-0D/policy_smoke/
artifacts/lowlevel/pc_v0d_bc_gate_20261106/R-Z0-0D/frozen_base_smoke/
```

- 113 个 summary 数值指标无差异；
- 所有 trace 数组逐位相同；
- 证明零残差时完整策略与冻结 BC 基线完全等价。

历史 `R-BC-0D` summary 由 `evaluate_pc_bc.py` 生成，R-Z0 由 `evaluate_pc_ppo.py`
生成，两者的随机流不同，因此历史 summary 有约 0.3% 的数值差异。上述同 harness
smoke 排除了策略差异；R-Z0 本身即可作为同 harness 的 BC-0D 基线。

## BC 基座与路径归因

历史 20-seed 门禁仍成立：

| 指标 | R-BC-3D | R-BC-0D |
|---|---:|---:|
| 3D RMSE (m) | 12.5878 | 12.2429 |
| horizontal RMSE (m) | 7.1314 | 7.5953 |
| vertical RMSE (m) | 0.9439 | 0.9734 |
| success | 0.15 | 0.20 |
| timeout | 0.85 | 0.80 |
| safety | 0.00 | 0.00 |
| completion | 0.99417 | 0.99359 |

`R-BC-0D / R-BC-3D` 的 3D RMSE ratio 为 `0.9726`，低于预注册上限 `1.25`。

同 harness path-zero 基线：

```text
artifacts/lowlevel/pc_v0d_bc_gate_20261106/R-BC-0D/path_zero_same_harness/
```

| 指标 | BC-0D | BC-0D + path-zero | 变化 |
|---|---:|---:|---:|
| 3D RMSE (m) | 12.2837 | 20.7088 | `+68.59%` |
| horizontal RMSE (m) | 7.6187 | 19.8541 | `+160.6%` |
| vertical RMSE (m) | 0.9683 | 3.6045 | `+272.2%` |
| success | 0.20 | 0.00 | `-0.20` |
| timeout | 0.80 | 1.00 | `+0.20` |
| completion | 0.9935 | 0.3321 | `-0.6614` |

20 个配对 seed 中 17 个恶化，平均 3D RMSE 恶化 `9.0122 m`。路径输入确实被
BC-0D 使用，满足 path-zero 归因门禁。

## 下一阶段

启动 `R-PPO-0D` screen：

- `4096 envs × 256 steps × 40 iterations`，约 `41.94M` 环境步；
- 训练 seed 固定为配置中的 `91430`；
- checkpoint 预先固定为最终 checkpoint，不使用 screen seed 选模型；
- screen 使用 seed 0–19，与 BC-0D 和 R-Z0 逐 seed 配对；
- 同时评估 default、path-zero 和 `frozen-base` 基线。

