# Goal18 V-0D：Sign/Magnitude 动作头 BC-only 结果报告

日期：2026-11-07

状态：**已执行并收口。判定 `FAIL_RMSE`，停止 Goal18，不进入残差 PPO。**

协议：`doc/goals/goal18_v0d_sign_magnitude_bc_prompt_20261107.md`

## 1. 结论

只用现有 R2B 合并示范替换最后一层动作参数化，**没有**解决当前 BC 问题：

- S0 连续 MSE 复现了同代码 BC 量级：default `8.9129 m`，对历史同配方
  BC artifact `9.0944 m` 的比值为 `0.9800`；
- S1 sign/magnitude default `10.2868 m`，比 S0 恶化 `15.4%`，且高于
  `7.0 m` 主门禁；
- S1 strict125 `11.7268 m`，高于 `10.5 m` 门禁；
- S1 default success `0.50`，低于 `0.60` 护栏；
- 方向分类本身不是主要问题：`|teacher action| >= 0.5` 时 sign accuracy 为
  rudder/stern `0.9501 / 0.9548`；
- 失败集中在幅值和时间一致性：饱和恢复率仅 `0.2689 / 0.1524`，大幅反转率
  `0.6241 / 0.1859 次/秒`，rate-limit 命中率 `71.70%`（教师 `38.69%`）。

因此，**sign/magnitude 动作头不是当前 BC 瓶颈的充分修复**。按预注册规则停止，
没有启动残差 PPO screen。

## 2. 固定条件

- 数据：`artifacts/lowlevel/pc_v0d_r2b_bc_20261106/demos_dagger_merged/demos.npz`；
- 数据 SHA256：`eb21837960f89c5295af4946bd3fb914471602f6cbd15a33f40aa9ecdb2d1387`；
- 样本/episode：`393216 / 310`；
- 教师：R2B，不重采、不加 DAgger、不加 action noise；
- actor 观测：12D base state + 8D path，`command_dim=0`；
- `d_z_total=64`、state encoder `[128,128]`、actor trunk `[256,128]`；
- `batch_size=4096`、`max_epochs=300`、`patience=30`、`lr=1e-3`、
  `validation_fraction=0.15`、`seed=0`；
- 训练 GPU：服务器 `CUDA_VISIBLE_DEVICES=1`；
- 闭环评估：CPU float64、seed `0–19`、20 个 seed；
- 未修改 plant、sensors、current、reward、termination、评估 seed 或既有门禁。

## 3. 训练结果

| run | action head | best epoch | epochs run | val continuous MSE | checkpoint SHA256 |
|---|---|---:|---:|---:|---|
| S0 | continuous | 45 | 76 | 0.127633 | `4397d05b...f9c387` |
| S1 | sign/magnitude | 8 | 39 | 0.157695 | `2cfd63a7...cad7c6` |

S1 的显式总损失为 `CE(sign) + SmoothL1(magnitude，非零样本) + MSE(rpm)`，三项等权；
deadband 固定 `0.05`。S1 的连续动作验证 RMSE 也高于 S0：

| run | rudder normalized RMSE | stern normalized RMSE | RPM normalized RMSE |
|---|---:|---:|---:|
| S0 | 0.4573 | 0.3489 | 0.2282 |
| S1 | 0.5162 | 0.3905 | 0.2327 |

## 4. 闭环结果

| 策略 | 场景 | 3D RMSE (m) | success | timeout | safety |
|---|---|---:|---:|---:|---:|
| S0 continuous | default | 8.9129 | 0.55 | 0.45 | 0.00 |
| S0 continuous | strict125 | 10.1828 | 0.30 | 0.70 | 0.00 |
| S0 continuous | path-zero | 12.8576 | 0.10 | 0.90 | 0.00 |
| S1 sign/magnitude | default | 10.2868 | 0.50 | 0.50 | 0.00 |
| S1 sign/magnitude | strict125 | 11.7268 | 0.30 | 0.70 | 0.00 |
| S1 sign/magnitude | path-zero | 11.9473 | 0.00 | 0.80 | 0.20 |

S1 path-zero 相对自己 default 仍恶化 `16.1%`，说明路径输入没有完全失效；但
path-zero safety 升到 `0.20`，且 S1 的 default/strict125 RMSE 都没有达到门禁。

## 5. 动作表示诊断

阈值：sign accuracy 只统计教师动作绝对值 `>=0.5` 的样本；饱和阈值 `0.99`。

| 指标 | S0 continuous | S1 sign/magnitude | 教师 |
|---|---:|---:|---:|
| sign accuracy：rudder | 0.9835 | 0.9501 | — |
| sign accuracy：stern | 0.9826 | 0.9548 | — |
| 饱和率：rudder | 29.59% | 16.10% | 59.88% |
| 饱和率：stern | 9.10% | 3.22% | 21.13% |
| 饱和恢复率：rudder | 0.4942 | 0.2689 | 1.00 |
| 饱和恢复率：stern | 0.4306 | 0.1524 | 1.00 |
| 大幅反转：rudder (次/秒) | 0.000 | 0.6241 | 0.000 |
| 大幅反转：stern (次/秒) | 0.000 | 0.1859 | 0.000 |
| 任一通道反转 (次/秒) | 0.000 | 0.8095 | 0.000 |
| rate-limit 命中率（任一通道） | 61.34% | 71.70% | 38.69% |

要求是饱和恢复率至少 `0.65 / 0.65`、sign accuracy 至少 `0.90 / 0.90`，且反转与
rate-limit 不得比教师高超过 `0.05`。S1 只有 sign accuracy 单项通过。

## 6. 失败原因判断

本轮结果支持以下判断：

1. 显式 sign 分类确实学到了大部分方向信息，sign accuracy 已达到 `0.95` 左右；
2. 但独立 sigmoid magnitude 没有学会教师的高幅值尾部，饱和恢复反而低于连续 MSE；
3. 在方向不确定的样本上，argmax sign 容易翻转，导致动作反转和 rate-limit 命中率
   显著升高；
4. sign 与 magnitude 分离后，验证集连续动作误差也变差，说明共享 trunk 没有获得
   更好的动作表示，只是把误差从均值化转移到了符号/幅值交界处。

这并不证明“动作表示方向”完全错误，但足以否定当前这个最小 sign/magnitude 实现是
有效修复。不能根据本轮结果调 deadband、权重或 seed 后重试，否则属于结果后调参。

## 7. 决策

预注册判定：

- default RMSE `<=7.0 m`：**FAIL**（`10.2868`）；
- default RMSE `<=0.80 × 12.2396 = 9.7917 m`：**FAIL**；
- strict125 RMSE `<=10.5 m`：**FAIL**（`11.7268`）；
- default success `>=0.60`：**FAIL**（`0.50`）；
- 饱和恢复率 `>=0.65 / 0.65`：**FAIL**（`0.2689 / 0.1524`）；
- sign accuracy `>=0.90 / 0.90`：PASS（`0.9501 / 0.9548`）；
- 反转与 rate-limit 稳定性：**FAIL**。

最终状态：**`FAIL_RMSE`**。决策：**`stop_goal18_bc_gate_failed`**。

不进入残差 PPO screen；Goal17 继续保持暂缓。

## 8. 下一步建议

下一步应另开 Goal，在以下两个方向中**只选一个**先做，不继续在本 Goal 内扫参：

1. **action chunking**：优先处理本轮暴露的时间一致性问题，输出一段动作而不是单步动作；
2. **mixture-density head**：显式建模多峰动作分布，但必须避免把失败再次归因于采样噪声。

基于本轮“方向基本正确、幅值和切换不稳定”的证据，建议先做 action chunking；只有
chunking 仍不能恢复饱和动作时，再做 mixture-density。

## 9. 产物

服务器根目录：

```text
artifacts/lowlevel/pc_v0d_sign_magnitude_bc_20261107/
```

关键文件：

| 文件 | SHA256 |
|---|---|
| `run_manifest.json` | `d16886bb5adfd73e7a8fb1d1aa5a8c9da0f923ed685edb26cd413da85e63a0f2` |
| `stage_decision.json` | `d4e2448b045f2bc312d801bd7e7682de77ab1c69406b07d44f377d72f8459d7c` |
| `stage_decision.md` | `80d3b3a39b4a0a08f81d877a1aa623b8b20d0df43bd383d9f378b937165f79ab` |
| `audit/s0_action_saturation.json` | `7f1a2e5bd4e7650aa0254ca7cea5ac74be5adcf4d0e8b390e378caddffd391d1` |
| `audit/s1_action_saturation.json` | `13da06451bb7b102d6ffbd782ee50d322a9f2a2b4513d7f96e7a32afdcac9284` |
| `s0_continuous/pc_bc_actor.pt` | `4397d05b026634da266e61d0dbc2466a891cfdb08083ab962365e6dedbf9c387` |
| `s1_sign_magnitude/pc_bc_actor.pt` | `2cfd63a7e8c815299a0df924ad989a4442a4fdc848a3ad3b8bdf9c766fcad7c6` |

日志目录：

```text
logs/pc_v0d_sign_magnitude_bc_20261107/
```
