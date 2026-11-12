# Goal85b 结果：plant mismatch 实现正确，但 ±10% 主要改变路径误差而非终端捕获

**日期**：2026-11-10<br>
**状态**：`B1_IMPLEMENTATION_PASS / MISMATCH_TOO_SMALL / B2_NOT_TRIGGERED / B3_IMPLEMENTED`<br>
**预注册**：`doc/goals/goal85b_plant_mismatch_mandatory_20261110.md`<br>
**上游**：Goal85 `CONTRACT_PASS_FRAME_DIAGNOSTIC_INVALID_EFFECT_NEGATIVE`<br>
**远端产物**：`/data/yhx/underwater-robotics/src/plan_p1/goal85b_plant_mismatch_20261110/`

## 1. 结论

plant mismatch 已完整接入真实动力学，且没有污染传感器或终止坐标：

| 检查 | 结果 |
|---|---|
| `m0` 与 Goal85 的 800 个 episode | `0` mismatch，逐 episode 等价 |
| B1 runs / episodes | `60 / 2400`，全部 finite |
| 随机化范围 | 五个因子均在预注册范围内 |
| common random numbers | 所有 arm 同 seed 因子完全一致 |
| `abs(mean(p_nav-eta))` | `0.000765 m < 0.05 m` |
| added mass ±10% | 已进入 6×6 mass matrix，全部闭环稳定 |

B1 的主效应方向一致但统计上不够强：

| 档位 | zero-residual path RMSE Δ | 95% CI 是否含 0 | 判定 |
|---|---:|---|---|
| `m05` | `+0.0140–0.0147 m` | 4 个单元中 1 个不含 0 | detected `1/4` |
| `m10` | `+0.0282–0.0302 m` | 4 个单元全部含 0 | detected `0/4` |

预注册 Gate 要求 `m10 >=3/4` 或 `m05 >=2/4`，因此正式状态是
`MISMATCH_TOO_SMALL`。这不是实现失败；所有 8 个 zero-residual 单元的均值都是正向
路径退化，只是 40 个 seed 的配对分布有长尾，均值 CI 跨零。

## 2. 为什么不是实现接错

`m10` 单因子 zero-residual 诊断（16 runs / 640 episodes）给出：

| 因子 | path RMSE 平均 Δ | 方向 |
|---|---:|---|
| drag `[-10%,+10%]` | `+0.0099–0.0107 m` | 稳定变差 |
| thrust `[-10%,+10%]` | `-0.0020–0.0026 m` | 很小，略改善 |
| fin `[-10%,+10%]` | `+0.0209–0.0213 m` | 稳定变差 |
| added mass `[-10%,+10%]` | `-0.0005–-0.0003 m` | 近零 |

组合 `m10` 的 `+0.028–0.030 m` 与 drag+fin 的贡献一致。因此乘子已经进入对应力项；
当前现象是闭环对 5–10% 参数误差的敏感度本身较小，而不是随机化没有生效。

## 3. 终端捕获

`m10` zero-residual 的 terminal distance 均值变化：

| scenario / root | Δ terminal eta distance | 95% CI |
|---|---:|---|
| default / baseline | `+0.0259 m` | `[-0.0107,+0.0755]` |
| default / candidate | `+0.0305 m` | `[-0.0026,+0.0749]` |
| strict125 / baseline | `+0.0293 m` | `[-0.0126,+0.0811]` |
| strict125 / candidate | `+0.0164 m` | `[-0.0222,+0.0627]` |

方向仍然是变差，但没有形成稳定的 success/plane-miss 移动；因此不能用这批数据支持
“plant mismatch 是神经网络打不过传统方法的主因”。

## 4. B2 与 B3

B2 同分布训练未启动。原因是预注册 B1 Gate 已判 `MISMATCH_TOO_SMALL`：
在 nominal 闭环中，±10% mismatch 只造成约 `3 cm` path RMSE 退化，终端捕获变化
不显著。按协议先检查实现；检查通过后，不用一个低先验的训练组去消耗 formal 预算。

B3 added mass 已实现：

- 标量乘子作用到完整 6×6 added-mass diagonal；
- Coriolis、mass matrix inverse 和 forward-Euler 都按 per-env mass matrix 计算；
- 单元测试覆盖范围、`m0` 等价、因子通道、seed 复现和传感器隔离；
- B1/B1-component 共 3040 个 episode 没有 NaN/Inf 或发散。

## 5. 关键判读

本轮不能支持以下结论：

- 不能支持“随机化没生效”；
- 不能支持“fin/drag 力项接错”；
- 不能支持“plant mismatch 已解释神经网络的终端失败”；
- 不能把 `MISMATCH_TOO_SMALL` 改写成 PASS。

本轮支持：

- 当前 ALOS + PD/PI 闭环对 ±10% drag/thrust/fin/added-mass 有鲁棒性；
- 失配主要压低路径精度，没有稳定改变 near-goal terminal capture；
- 神经网络方法的问题更可能仍在动作参数化、terminal capture、reward 或学习信号，
  而不是 5–10% plant mismatch。

## 6. 复现

```bash
# B1：60 runs × 40 seeds
bash scripts/run_pc_goal85b_b1_matrix_20261110.sh

# B1 Gate
PYTHONPATH=src python scripts/analyze_pc_goal85b_b1.py \
  --root .../goal85b_plant_mismatch_20261110/b1 \
  --output .../b1_gate.json --markdown .../b1_gate.md

# 单因子机制诊断
bash scripts/run_pc_goal85b_component_diagnostic_20261110.sh
PYTHONPATH=src python scripts/analyze_pc_goal85b_components.py \
  --root .../goal85b_plant_mismatch_20261110 \
  --output .../b1_component.json --markdown .../b1_component.md

# 契约测试
PYTHONPATH=src python -m pytest -q tests/test_pc_goal85b_plant_mismatch.py
```

## 7. 产物

- `b1_gate.json` / `b1_gate.md`
- `b1_component.json` / `b1_component.md`
- `b1/{m0,m05,m10}/...`
- `b1_component/{drag_scale,thrust_scale,fin_scale,added_mass_scale}/...`
