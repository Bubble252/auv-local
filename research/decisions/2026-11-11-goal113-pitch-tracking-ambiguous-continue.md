# Goal113 决策：3 m 主窗口无单一机制，继续做 2 m-anchored mixed-mechanism 审计

**日期**：2026-11-11<br>
**Goal**：Goal113 `PITCH_TRACKING_AMBIGUOUS`<br>
**Decision**：`continue`<br>
**Next**：Goal114 2 m-anchored pitch tracking time-resolved mixed-mechanism 审计

## 1. 触发

Goal112 判 `TRACKING_LIMITED`：

```text
reference involved = 3/11
tracking involved  = 8/11
tracking pitch bad = 7/11 at 2 m entry
material saturation = 0/11
```

Goal113 预注册只读审计 3.0 m 主窗口，想区分：

- command pitch rate 是否不可达；
- stern/fin authority 是否限制；
- 速度/洋流是否耦合；
- 3.0 m 是否已有 pre-entry pitch lag；
- 是否 pitch command 可达、可考虑 SAC。

## 2. 结果

3.0 m 主窗口：

```text
AUTHORITY_LIMITED        = 0/11
COMMAND_RATE_UNREACHABLE = 2/11
SPEED_CURRENT_COUPLED    = 3/11 (mechanism)
PREENTRY_PITCH_LAG       = 0/11
NO_PITCH_LAG             = 8/11
MIXED_OR_AMBIGUOUS       = 2/11
```

没有机制达到 6/11，正式状态为：

```text
PITCH_TRACKING_AMBIGUOUS
reason = no_mechanism_at_least_6_of_11_in_frozen_3m_window
```

但描述层显示：

```text
pitch error >= 0.45:
  3.0 m = 3/11
  2.5 m = 4/11
  2.0 m = 7/11
```

2.0 m 机制计数：

```text
AUTHORITY_LIMITED        = 3/11
COMMAND_RATE_UNREACHABLE = 3/11
SPEED_CURRENT_COUPLED    = 7/11 (mechanism)
MIXED_OR_AMBIGUOUS       = 5/11
```

因此不是“3 m 前已经有稳定 pitch lag”，而是：

```text
pitch lag 主要从 3 m 到 2 m 形成；
到 2 m 时机制已经 mixed，不能压成单一 authority/rate/current 原因。
```

## 3. 为什么是 continue，不是 pivot 到训练或 SAC

- `PITCH_TRACKING_AMBIGUOUS` 表示预注册的 3 m 窗口没有单一机制过半，不是“pitch
  command 可达”；
- 2 m 的 7/11 pitch bad 与 mixed 机制说明还有明确的只读分解空间；
- 没有任何证据支持：
  - residual 幅度不足；
  - material saturation；
  - fin authority 单独不足；
  - 速度/洋流单独主导；
  - 残差方向正确但探索/样本效率不足；
- 因此不换算法、不开 SAC/REDQ/DroQ、不改 reward/terminal/success/goal radius；
- 网络谱系仍然成立：NoisyNet absent、frozen BC 22/22 tensor parity、PPO/SAC
  state encoder 3/3 torch.equal；Goal108 SAC 的 path12 随机初始化仍然不是同构对照。

## 4. 允许的下一步

新开 Goal114，仍然只读：

```text
Goal114 = 2 m-anchored pitch tracking time-resolved mixed-mechanism 审计
```

必须固定：

- 主窗口改为首次进入 2.0 m；2.5/3.0 m 只作 onset 描述；
- 对 3.0→2.0 m 做时间分辨的分段斜率；
- 对 mixed 样本做逐帧 onset 排序：pitch command rate、actual pitch rate、stern norm、
  speed error、current magnitude、pitch error 谁先过阈值；
- 仍不训练、不改动作参数化、不改 reward/terminal/success、不打开 held-out。

Goal114 的 Gate 应明确：

```text
若 rate demand 先于 pitch error，并且 actual pitch rate 长期不足：
  -> PITCH_RATE_DEMAND_LIMITED
若 stern norm 先于 pitch error，并且 material saturation 或 near-limit 持续：
  -> PITCH_AUTHORITY_LIMITED
若 speed/current 先于 pitch error：
  -> PITCH_SPEED_CURRENT_COUPLED
若多个 onset 接近或交替：
  -> MIXED_PITCH_HANDOFF
```

只有 Goal114 明确给出 `PITCH_RATE_DEMAND_LIMITED` 或 `PITCH_COMMAND_REACHABLE`
且证明 residual 方向正确但样本效率不足时，才允许新开 SAC 同协议对照。

## 5. 证据

- `doc/goals/goal113_pitch_tracking_authority_audit_20261111.md`
- `doc/goals/goal113_result_20261111.md`
- `artifacts/lowlevel/goal113_pitch_tracking_authority_20261111/phase0_summary.json`
- `doc/goals/goal112_result_20261111.md`
- `doc/goals/goal109_result_20261111.md`
- `doc/goals/goal108_network_and_noisenet_audit_20261111.md`
- `research/paper_reading/2026-11-11-goal112-pitch-tracking-authority.md`
