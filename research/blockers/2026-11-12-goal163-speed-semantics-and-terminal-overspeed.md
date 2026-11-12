# Goal163 blocker：`nu`/海流语义错位导致速度反馈、奖励和评价同时偏置

**日期**：2026-11-12<br>
**状态**：`AUDITED / PIVOT_TO_GOAL164 / NO_RL_TRAINING / HELDOUT_CLOSED`<br>
**上游**：Goal162 `INVALID_SPEED_CONTRACT`、Goal163 只读审计

## 1. 现象

Goal162 的 zero/gain-residual 都报告 current10 峰值 `2.879 m/s`、current15
峰值 `3.308 m/s`，超限 `7.2%/14.7%`。Goal163 用同一批 deterministic replay，
在不加载策略、不训练、不改控制器的情况下重新对齐速度字段，发现：

- `eta_ned` 前向差分与 `nu[:3]` 的 MAE 为 `0.000`；
- 真实地速峰值只有 `2.546/2.598 m/s`；
- 真实超限比例只有 `0.31%/0.51%`；
- 旧 `ground_velocity_body_mps` 与 `eta` 的 MAE 为 `0.286/0.360 m/s`；
- 旧字段的超限比例 `7.20%/14.68%` 比真实值高约 20–30 倍。

## 2. 根因证据

### 2.1 动力学和积分

`src/auv_lowlevel/batch/dynamics.py`：

```python
relative_velocity = nu - current_body
...
position_dot = R @ nu_next[:, :3]
```

这说明 `nu` 是地速，`nu - current_body` 才是水相对速度。`eta` 差分验证
`nu` 就是地速。

### 2.2 速度 helper

`src/auv_lowlevel/batch/geometry.py:153` 的 `ground_velocity_body()` 返回：

```python
nu[:, :3] + world_to_body(current_ned_mps, quat)
```

在 `nu` 已经是地速时，这里重复加了海流。

### 2.3 传感器模型

`src/auv_lowlevel/batch/sensors.py:968`：

```python
water_body = nu[:, :3]
ground_body = water_body + current_body
```

这与动力学相反：它假设 `nu` 是水速。结果是 DVL bottom 比真实地速多一个
current，DVL water 也不是真实的 `nu - current`。

### 2.4 消费者

`BatchedPCLowLevelEnv` 把 `ground_velocity_body_mps` 用于：

- actor state / critic state；
- speed error / hold bonus；
- reward 的 `ground_surge_mps` 和 `route_ground_speed_mps`；
- evaluation trace 和速度包线统计。

ALOS/控制器使用的是水相对速度：`relative_surge = nu - current_body`，
`command_speed_mps` 也是水相对速度。于是学习信号与控制目标存在海流同量级的
系统偏差。这是比“noisenet 主导”更根本的解释。

## 3. 真实超速时序

| level | 真实超速 eps | 非 terminal 超速率 | terminal 超速率 | 首次距离 | 峰值距离 | 峰值到结束 |
|---|---:|---:|---:|---:|---:|---:|
| current10 | 6/120 | 0.0000 | 0.0083 | 6.14 m | 3.31 m | 0.29 s |
| current15 | 12/120 | 0.0000 | 0.0135 | 12.34 m | 3.20 m | 0.00 s |

- current10 的 6 个超速 episode 全部成功，只在最后约 2 s 出现短促超速；
- current15 的 6 个成功 episode 也是短促超速；另外 6 个 `GOAL_PLANE_MISS`
  episode 在距终点 6.6–11.7 m 处终止，超速可持续约 7 s；
- 超速步的 current 不高于正常步；turn rate 只略高，H163-B 不成立。

## 4. 可证伪假设

### H163-A：速度环积分/回收滞后是终端减速段超速的原因

- 成立应看到：超速步的 requested RPM 和 integral 显著高于正常步，且
  integral 接近 `400 rpm` cap。
- 证据：current10/current15 超速步 integral 为 `354.6/389.0`，at-cap 比例
  `50%/75%`；RPM 为 `1523/1409`，显著高于正常步 `1091/1100`。
- 裁决：`PARTIAL`。它解释终端减速段的残留超速，但不能解释旧评价的 7–15% 超限。

### H163-B：强流/转向瞬态使开环包线低估闭环地速

- 成立应看到：超速集中在高 current、高 turn rate 分箱。
- 证据：超速步 current 反而更低，turn rate 只从 `0.097/0.099` 升到
  `0.112/0.139`；非 terminal 超速率为 0。
- 裁决：`WEAK`。

### H163-C：速度定义或 plant backend 有系统错位

- 成立应看到：`ground_velocity_body_mps`、`nu`、`eta` 差分无法互相对齐。
- 证据：`nu` vs `eta` MAE `0.000`；`ground` vs `eta` MAE `0.286/0.360`；
  传感器模型与动力学对 `nu` 的解释相反。
- 裁决：`SUPPORTED`。

## 5. 已排除项

| 检查 | 结论 |
|---|---|
| noisenet / residual | zero 和 policy 的旧超限几乎相同；真实超速与 residual 无关 |
| 课程采样 | C6 为 `0.90–2.50 m/s`，sampler 保证 `speed+current+variation<=2.5`；单测通过 |
| held-out | `100–149` 未打开 |
| 训练预算 | 未新增训练；只 replay Goal162 已有 zero trace |
| seed 选择 | 3 seeds 全部一致；没有换 seed |
| 只读性 | 审计脚本不加载策略、不 step 环境、不改 reward/terminal/success/radius |

## 6. 最便宜证伪实验

已执行：

```bash
cd /data/yhx/underwater-robotics/src/plan_p1/goal153_reward_curriculum_20261111
/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python \
  scripts/audit_pc_speed_envelope.py \
  --trace-root artifacts/lowlevel/goal163_speed_audit_20261112/replay_zero \
  --output artifacts/lowlevel/goal163_speed_audit_20261112/replay_analysis
```

合成测试：

```bash
.venv/bin/python -m pytest -q \
  tests/test_pc_goal163_speed_semantics.py \
  tests/test_lowlevel_curriculum_sampler.py
```

结果：`22 passed, 1 xfailed`。xfail 是故意保留的“正确语义”tripwire：
当前 `ground_velocity_body` 仍返回 `nu+current`，Goal164 修复后会 XPASS，
必须移除 xfail。

## 7. 决策与下一步

`decision = pivot`。不扩 seed、不换 PPO/SAC/REDQ/DroQ、不打开 held-out。
预注册 Goal164：

1. 用版本化配置修正 `nu`/current 语义；
2. 修正 sensors、actor/reward、evaluation 的所有消费者；
3. 保留旧 trace/result，不覆盖历史目录；
4. 先跑 corrected zero-only replay，确认真实地速；
5. 若仍有 terminal 残留超速，再做最小 anti-windup/terminal deceleration 修正；
6. 速度契约恢复后，才重新训练 Goal162 的 gain residual。

