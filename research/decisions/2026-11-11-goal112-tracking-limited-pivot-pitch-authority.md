# Goal112 决策：从 terminal credit 转向 pre-entry pitch tracking / authority 审计

**日期**：2026-11-11<br>
**Goal**：Goal112 `TRACKING_LIMITED`<br>
**Decision**：`pivot`<br>
**Next**：Goal113 eval-only pitch tracking / authority / rate 归因审计

## 1. 触发

Goal111 把 strict C0 的 20 个 failure 分成：

```text
APPROACH_DEFICIT                7
PREENTRY_GEOMETRY_BAD          11
ENTRY_GEOMETRY_OK_LATE_FAILURE  2
```

Goal112 只读审计 11 个 `PREENTRY_GEOMETRY_BAD`，区分：

- command 对 direct goal LOS 是否已偏；
- command 对 actual eta yaw/pitch 是否跟得上；
- 还是控制方向对齐但位置偏移更早形成。

结果：

```text
reference involved = 3/11 = 27.3%
tracking involved  = 8/11 = 72.7%
```

互斥分类：

| 分类 | 数量 |
|---|---:|
| `REFERENCE_BAD_ONLY` | 1 |
| `TRACKING_BAD_ONLY` | 6 |
| `MIXED_REFERENCE_AND_TRACKING` | 2 |
| `ALIGNED_CONTROL_POSITION_OFFSET` | 2 |

分通道：

```text
tracking pitch bad   = 7/11
tracking heading bad = 3/11
reference heading bad = 3/11
reference pitch bad   = 0/11
```

## 2. 为什么是 pivot，不是换算法

`TRACKING_LIMITED` 说明失败的主要环节在实际姿态响应，不是 reward 缺失，也不是
terminal 标签本身。Goal109 已经报告 strict C0 的 residual magnitude 没有过
预注册阈值、material saturation 两组都为 0；Goal112 再次确认 11 个 failure 的
material saturation 为 0/11。因此当前没有证据支持：

```text
failure = residual 幅度不够
failure = 需要更大探索 / 更大 residual scale
failure = PPO 信用分配是唯一瓶颈
```

也没有证据支持换 SAC/REDQ/DroQ：

- Goal108 的 SAC 退化发生在 `bc_path_dim=12 + 随机 path encoder + bc_checkpoint=null`
  的非同构初始化下，不能写成“纯 SAC 优化器不如 PPO”；
- 若未来要开 SAC 对照，必须先固定 `bc_path_dim=8 + endpoint4`，并明确是否加载
  BC encoder；
- 当前更便宜的 falsification 是先查 pitch command/rate/authority 是否可达，而不是
  再开一条训练线。

## 3. 网络结构对照

本决策同时复核了用户提出的“是否是网络结构/NoisyNet 问题”：

```text
NoisyNet                    = absent
PPO/SAC state encoder       = 3/3 layers torch.equal
frozen BC tensor parity     = 22/22 torch.equal, max_abs_diff=0.0
PPO zero-increment parity   = true
SAC path init parity        = false (bc_path_dim=12, random path encoder, no BC checkpoint)
```

因此：

- 不能把 tracking pitch 失败归因 NoisyNet；
- 不能把 Goal108 SAC 的退化归因纯优化器；
- 当前问题不是“网络接错”，而是闭环姿态响应/权限/速率尚未被分解。

## 4. 允许的下一步

新开 Goal113，仍然只读：

```text
Goal113 = eval-only pitch tracking / authority / rate 归因审计
```

只分析 strict C0 的 11 个 pre-entry failure，回答：

1. pitch command 本身是否变化过快或不可达；
2. actual pitch response 是否不足，pitch rate 是否长期落后；
3. stern/fin authority、pitch rate、vertical speed、speed error、current 是否限制 tracking；
4. 3.0 m 已坏的 6 个样本是否在 3 m 前就出现稳定 pitch lag；
5. 是姿态控制问题，还是深度/速度/洋流耦合问题。

Goal113 必须保持：

- 不训练；
- 不改 reward/terminal/success/goal radius；
- 不打开 held-out；
- 不先改 residual scale 或动作参数化；
- 先跑最便宜的只读分类。

## 5. 何时才开 SAC 对照

只有同时满足以下条件，才新开预注册 SAC 同协议对照：

```text
pitch command 对 goal 基本合理
actual pitch 的响应方向正确
但训练/探索没有把 residual 推向正确方向，或样本效率明显不足
```

如果 Goal113 指向：

- command rate 不可达；
- fin/actuator authority 不足；
- speed/current 耦合；
- plant mismatch / delay；

则优先做 controller、参数化、plant mismatch 或 reference 修复，不换 RL 算法。

## 6. 证据

- `doc/goals/goal112_preentry_control_cause_20261111.md`
- `doc/goals/goal112_result_20261111.md`
- `artifacts/lowlevel/goal112_preentry_control_cause_20261111/phase0_summary.json`
- `doc/goals/goal111_result_20261111.md`
- `doc/goals/goal109_result_20261111.md`
- `doc/goals/goal108_network_and_noisenet_audit_20261111.md`
- `doc/goals/goal89_network_parity_and_gate_scope_audit_20261111.md`
- `research/paper_reading/2026-11-11-goal112-pitch-tracking-authority.md`
