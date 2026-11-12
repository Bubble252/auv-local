# Goal109 blocker：方向单独过线，缺少预注册分支

**日期**：2026-11-11<br>
**Goal**：Goal109<br>
**类型**：Gate 设计缺口，不是数据缺失。

## 现象

strict C0 的 `rpm_speed_sign_alignment` 达到 `0.3825`，但幅度指标最大只有
`0.1950`；几何指标有 6 项达到或超过 `0.33`。冻结 Gate 没有“方向单独过线”的
分支，因此形式结果只能是 `NO_DIRECTION_SIGNAL`。

## 证据

- `rpm_speed_sign_alignment`：success `0.6634`，failure `-0.3960`，δ `0.3825`
- `residual_norm_mean`：success `282.13`，failure `359.49`，δ `-0.1950`
- `last2s_closure_last_mps`：success `1.4034`，failure `0.1221`，δ `0.9650`
- 6 个 geometry 指标强分离

## 已排除

- 不是 trace 缺字段；所有必需字段存在；
- 不是 C0 不可复现；default 33/40、strict 20/40；
- 不是 material saturation；两组均为 0；
- 不是 NoisyNet；Goal108 网络审计已排除。

## 两个可证伪假设

1. **H109-B1：rpm 方向翻转是 closure stall 的原因。** 在首次进入 2.0 m 后，
   failure 的 rpm residual 与 speed error 提前反向，随后 closure 才跌破阈值。
2. **H109-B2：rpm 方向翻转是 terminal 几何失败的结果。** 在 closure stall 前，
   failure 已出现更大的 vertical/pitch 误差；rpm 符号是后续追不上造成的。

## 最小验证

只读 Goal107 trace，按事件锚点重采样：首次进入 2.0 m、首次 closure <0.2 m/s、
最后一次 closure >0 的时刻；比较事件前后 rpm 符号、vertical/pitch 误差和 closure。
不训练、不改 reward/terminal/success、不打开 held-out。

## 决策

`pivot`：先开 Goal110 eval-only 事件级审计；在方向与几何的先后顺序明确前，
不改动作参数化、不改 reward、不换算法。
