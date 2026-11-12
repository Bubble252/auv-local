# Goal115 决策：filter 是失效链路入口，但因果强度待 Goal116 验证

**日期**：2026-11-11<br>
**Goal**：Goal115 `REFERENCE_FILTER_LIMITED`<br>
**决策**：`continue`

## 证据

- strict C0 的 11 个 `PREENTRY_GEOMETRY_BAD` failure：
  `REFERENCE_FILTER_LIMITED=8`、`PLANT_RESPONSE_LIMITED=3`、
  `RESIDUAL_DIRECTION_LIMITED=0`、`MIXED_HANDOFF_CONTINUE=0`。
- 8 个 Goal114 mixed 子集：filter `6/8`、plant `2/8`。
- Goal107 → Goal115 的 40 个 dev seed instrumentation parity 为 `PASS`，
  `atol=1e-10`，没有 success/terminal/command/eta/nu/actuator 差异。
- failure 的 `limited_at_limit_fraction=0.853`，success control 为 `0.495`；
  failure 的 `median_tracking_deficit=0.135 rad/s`，success control 为
  `0.027 rad/s`。
- success control 的 `clip_fraction=0.931`、`target_rate_p95=3.815 rad/s`，说明
  高频 target 与 rate clipping 并非 failure 独有；不能直接推断“抬限幅即可修复”。
- `action_mode=frozen-base`，PPO 增量为零；本 Goal 的 residual 是冻结 BC 残差。
- Goal108 网络审计确认 NoisyNet 参数数为 0，frozen BC/PPO parity 精确。

## 判断

`REFERENCE_FILTER_LIMITED` 是预注册分类结果，不是“已证明 filter 是唯一根因”。
现有证据支持“失效链路经过 command rate limiter”，但 filter、plant response 和
stern saturation 同时存在，必须用 eval-only 因果对照分开。

不允许因此切 SAC/REDQ/DroQ，也不允许直接修改训练 reward/terminal/action 主线。

## 下一步

Goal116：同 checkpoint、同 dev seeds 20–59、strict125 C0、`frozen-base`，
预注册四个 eval-only arms：

```text
R050  7.5 deg/s
R100 15.0 deg/s  (control)
R200 30.0 deg/s
ROFF disabled
```

判读规则：

| 观察 | 判读 | 下一步 |
|---|---|---|
| R200/ROFF 明显改善且安全不退化 | filter 因果瓶颈 | 调 reference/limit 设定 |
| R050 明显改善且安全不退化 | 原始 target 高频噪声主导 | 做 reference smoothing |
| 四臂差异 <= seed 噪声 | filter 非因果主因 | 回 plant/action/reward |
| 任一臂 timeout/safety 增加 | 方向淘汰 | 保留数据，不救指标 |

成功判据在 Goal116 文档中预先冻结；结果出来前不得改 seeds、scenario、checkpoint、
success frame 或阈值。
