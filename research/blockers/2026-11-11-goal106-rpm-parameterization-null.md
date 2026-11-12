# Blocker：连续三次 rpm/动作参数化修复未移动 strict Gate

**日期**：2026-11-11<br>
**Goal**：Goal104 → Goal105 → Goal106<br>
**触发条件**：`gate_fail_twice` + `no_metric_movement_after_3_fixes`

## 1. 现象

冻结 Gate 始终要求：

```text
default >= 33
strict125 >= 23
path <= 1.05 * C0
timeout == 0
safety == 0
```

历史结果：

- Goal104：`R075/R125` 到 33/21，但各只新增 1 个 strict seed；`U050` 32/23；
- Goal105：`U050` 稳定 32/23，静态 rudder 交互分别得到 33/22 与 32/23；
- Goal106：全部近终端 rpm 门控 strict 20，default 最多 34，strict gained 为空。

## 2. 已排除

| 项目 | 证据 |
|---|---|
| checkpoint/网络接线 | Goal104 parity 22/22，`max_abs_diff=0.0`，无 NoisyNet |
| 评估链 | Goal106 C0 与 Goal105 trace `max_abs_diff=0.0` |
| 路径退化 | Goal106 path ratio 1.0009–1.0032 |
| timeout/safety | 全为 0 |
| 门控未启用 | near gate 10–24% steps 触发，rpm delta 206–282 rpm |
| 单 seed 噪声 | U050 gained 集合 `25/51/56/59` 在 Goal104/105 重现 |

## 3. 可证伪 hypothesis

### H107-A：terminal reward/advantage 被 dense path/speed 项压制

- 机制：策略在 1.25 m 外更关心路径/速度 shaping，缺少把最后几十厘米走完的梯度。
- 成立：strict 失败 seeds 的 terminal 相关 reward/advantage 份额低，且最后一公里
  reward 仍由 path/speed 主导。
- 失败：terminal reward 已是主导项，或失败 seeds 的 terminal advantage 与成功
  seeds 无差异。
- 最便宜证伪：对 C0 strict 40 seeds 复跑一次 `--trace-reward-breakdown`，只做离线
  分解；不训练。

### H107-B：失败来自终端几何与 endpoint-stop capture，而非 reward 权重

- 机制：在 1.25 m 硬壳附近，终端方位/高度偏差与减速时序决定成败；reward 改动
  无法改变该几何。
- 成立：失败 seeds 与 U050 成功 seeds 的 terminal bearing/elevation、plane-miss、
  最后 2 秒 closure 有系统性差异，且 reward 分解无方向性。
- 失败：几何分布在 success/failure 间不可分，或 terminal advantage 明确偏低。
- 最便宜证伪：复用 Goal106/105 trace，只读计算最后 2 m、2 s 的几何与控制量。

### H107-C：BC prior 的终端动作签名固定，评估残差没有足够 terminal correction

- 机制：BC 在最后阶段输出稳定的减速/舵角模式，PPO/scale 残差只做小幅扰动。
- 成立：C0 与 U050 的成功差异主要来自整段速度积分，而不是最后 2 s 的残差动作。
- 失败：失败 seeds 在最后 2 s 有明显但方向错误的 residual correction。
- 最便宜证伪：按 seed 对齐 BC base、applied/effective residual 和 final distance。

## 4. 下一步

进入 Goal107 只读审计；不训练 SAC/PPO，不继续 rpm 扫值，不打开 held-out。
审计完成后在 `research/LEDGER.jsonl` 写 `continue` / `pivot` / `stop`。
