# Blocker：正确符号 dose 单调改善几何，但 success 只在 radius 边界 +1

**日期**：2026-11-11<br>
**Goal**：Goal133<br>
**触发**：连续 geometry 修复后主指标仍低于 GAIN；需要停止 eval-only bias 家族。

## 现象

正确符号 vertical dose 全部无 loss：

```text
Z       24 / 33
Vm005   24 / 33
Vm010   25 / 33
Vm020   25 / 33
```

last-2s `|vertical|` 单调下降：strict `0.887 -> 0.877 -> 0.868 -> 0.858 m`；
default `0.872 -> 0.864 -> 0.857 -> 0.848 m`。但 success 只从 24 到 25。

## 证据

1. 唯一新增 strict success 是 seed 49：terminal eta `1.321 -> 1.248/1.242 m`。
2. Vm010/Vm020 的 strict paired delta 只有 +1/40，CI95 `[0, 0.075]`。
3. default 在所有 dose 下完全不变，33/40。
4. near-radius count：strict Z 为 `24 / 25 / 26 / 27 / 30`（`r/+5/+10/+25/+50 cm`），
   失败集中在捕获球外壳。
5. parity action/residual maxdiff=0，terminal kind 一致。

## 已排除

- 符号错误：负 gain 正确，几何指标单调改善。
- dose 太小：Vm020 已是 0.15 cap，仍只有 1 个边界翻转。
- 大 dose 伤害：Vm020 没有新增 loss。
- 网络结构/NoisyNet：本 Goal 未训练，Goal108 parity 未改变。
- 执行链错误：parity 精确为 0。

## 可证伪假设

| 假设 | 机制 | 成立时应看到 | 最便宜证伪 |
|---|---|---|---|
| H133-X1：nominal 任务对 ALOS+BC 太容易 | 传统闭环已接近上限，NN 残差没有优势空间 | 更高 current/plant mismatch 下 zero 退化，而 SAC 优势出现 | 冻结 SAC vs zero 的压力测评矩阵 |
| H133-X2：SAC 只有稳态 seed 86201 可用 | 1M SAC 高方差，只有单训练 seed 学到可用方向 | 压力测评中 86201 仍最好，其他 seed 无增益 | 8 checkpoint × stress levels |
| H133-X3：评价只看固定 radius 掩盖连续改善 | 1–6 cm 几何变化被二元阈值吞掉 | near-radius 分布改善但 success 不动 | 本 Goal 已支持；后续保留过渡指标 |
| H133-X4：需要训练型多通道 residual | 单通道固定 bias 无法同时修 vertical/closure/进入顺序 | 固定 bias 全部停止后，训练型仍可能提升 | 压力优势成立后再训练 |

## 最便宜 falsification

Goal134 压力测评：同 8 个 Goal126 SAC checkpoint + zero，在 current 1.5x/2x、
plant mismatch ±20%、组合压力下跑 strict/default dev40。如果没有任何 stress
level 出现跨 seed 的 SAC 优势，则停止“压力能体现 NN 优势”的假设，转训练信号本身。

## 文献

- `research/paper_reading/2026-11-11-goal132-terminal-capture-metrics.md`

## 裁决

```text
decision = stop
eval-only geometry bias family stopped
next = frozen SAC vs zero stress-envelope screen
```
