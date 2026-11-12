# Decision：Goal132 后只测物理正确符号的 vertical dose

**日期**：2026-11-11<br>
**Goal**：Goal132<br>
**Decision**：`pivot`<br>
**上游**：Goal131 `GEOMETRY_DISCRIMINATOR_FOUND`

## 触发

Goal132 的 16 条 eval 和 parity 全部完成。正号 vertical、rpm closure 和组合
均未提高 success；rpm 把 closure 推高 15% 反而损失 1–2 个 strict success。
唯一 `GEO_SAFE_PARITY` 是负号 vertical（Vm002）。

## 判读

`signed_vertical_error_m` 与 stern-plane actuator 的符号映射与预注册相反：

- 正 gain 使 last-2s `|vertical|` 从 0.887 升到 0.897 m；
- 负 gain 使其降到 0.883 m，并保持 24/33。

因此这不是“垂直方向完全无效”，而是“唯一物理正确方向只测了最小 dose”。
同时，closure 被证明是结果描述，不是可直接优化的充分因果量。

## 决策

`pivot` 到 Goal133：只测正确符号 vertical dose，不混 closure。

```text
Vm005 = -0.05 actuator/m
Vm010 = -0.10 actuator/m
Vm020 = -0.20 actuator/m（受 0.15 hard cap）
```

要求：

- 同 40 dev seeds、strict/default、Goal132 的独立 10→2 m gate；
- 不训练、不改 reward/terminal/success/goal radius；
- 除 success 外记录 last-2s `|vertical|`、near-goal eta 分布和边界样本；
- 若 dose 单调改善几何但 success 不动，则停止 eval-only geometry bias 家族。

## 停止规则

- 任一 dose 达到 `GEO_GAIN`：固定为 prior，再预注册训练型 Goal；
- 只有 `GEO_SAFE_PARITY` 或几何改善：写 `stop`，不再扫 gain；
- 任一 arm `INVALID`：先修字段/parity，不解释指标；
- 不换 PPO/SAC 主算法，不开 REDQ/DroQ，不打开 held-out，不自动 commit/push。
