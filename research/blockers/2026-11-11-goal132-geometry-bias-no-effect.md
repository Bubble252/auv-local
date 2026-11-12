# Blocker：eval-only geometry bias 没有移动 success，closure 与 success 解耦

**日期**：2026-11-11<br>
**Goal**：Goal132<br>
**触发**：连续多轮 action/geometry 修复后主指标未移动；预注册 6 个修正 arm 均无 gain。

## 现象

16 条 formal eval、40 个 dev seed、strict/default 两个场景全部完成，parity 为
`true`（action/residual maxdiff=0）。结果：

```text
Z       24 / 33
Vp002   23 / 33
Vp005   23 / 33
Vm002   24 / 33
Sp005   23 / 33
Sp010   22 / 33
VSp     23 / 33
```

所有正增益 arm 都是 `GEO_NO_EFFECT`。`Vm002` 只是 `GEO_SAFE_PARITY`。

## 证据

1. closure 提高但没有 success：Sp005/Sp010 的 strict closure 从 1.257 升到
   1.354/1.447 m/s，success 反而从 24 降到 23/22。
2. 正号 vertical bias 把 last-2s `|vertical|` 从 0.887 推到 0.897 m；
   负号把其降到 0.883 m，说明 actuator 符号映射与预注册正号相反。
3. 所有 strict 损失集中在 seed 31/33，terminal eta 为 1.259–1.306 m，
   只比 1.25 m 边界多 1–6 cm。
4. Vm002 的几何趋势正确，但 dose 太小，success 精确停在 24。

## 已排除

- parity / 执行链接入：action、residual maxdiff 都是 0。
- NoisyNet / 网络结构：Goal108 63/63 tensor parity，无噪声参数。
- closure 幅值不足：Sp010 已把 closure 推高 15%，success 仍下降。
- 大范围轨迹崩溃：strict 损失都是 radius 边界 1–6 cm 的翻转。
- timeout/safety 主导：失败仍以 goal-plane-miss 为主。

## 可证伪假设

| 假设 | 机制 | 成立时应看到 | 最便宜证伪 |
|---|---|---|---|
| H132-X1：正确符号 dose 不足 | `Vm` 方向对，但 gain 0.02 只把垂直误差降 0.004 m | Vm005/Vm010/Vm020 出现单调 vertical 改善，且至少一个 arm strict >=26 | 新预注册 eval-only dose 矩阵 |
| H132-X2：success 指标在 1.25 m 边界过陡 | 1–6 cm 翻转被记成整 seed 损失，掩盖连续几何改善 | 连续误差/近区距离分布改善，但 success 不变或重排 | 同一 trace 的 eta/vertical 分布与 near-radius 曲线 |
| H132-X3：geometry prior 无法替代学习 | 固定 bias 只能改一个方向，不能同时处理进入顺序、速度和姿态 | 大 dose 只移动 vertical，不移动 success | Vm dose 矩阵后写 stop |
| H132-X4：主瓶颈是测评压力不足 | 当前 current/noise/plant mismatch 不足以让 ALOS 失效 | 压力场景中 zero/ALOS gap 才出现 | 独立压力测评 Goal，不改当前 success |

## 最便宜 falsification

先跑 Goal133：只测物理正确符号的 vertical dose（Vm005/Vm010/Vm020），
不混 closure，不改 success/radius。若 dose 呈单调改善但 success 不动，则
H132-X2/X3 得到支持，停止 eval-only geometry bias 家族。

## 文献

- `research/paper_reading/2026-11-11-goal132-terminal-capture-metrics.md`

## 裁决

```text
decision = pivot
停止正号 vertical 与 rpm closure；
只允许一次正确符号 dose-response 证伪，然后停止该家族。
```
