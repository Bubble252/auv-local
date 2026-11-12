# Goal84 决策：冻结双 success 口径，转向 strict125 terminal approach

**日期**：2026-11-10<br>
**Goal**：Goal84<br>
**状态**：`FRAME_ACCOUNTED`<br>
**Decision**：`continue`

## 1. 结果摘要

Goal84 在 20 个已记录 run、800 个 episode 上完成只读重评分，没有训练、新增
rollout、打开 held-out 或切换算法。

关键数字：

| 指标 | 数值 |
|---|---:|
| p_nav_tight success | `336/800 = 0.4200` |
| eta_tight_ever | `171/800 = 0.2138` |
| eta_tight_at_done | `151/800 = 0.1888` |
| p_nav-only | `185` |
| eta-only | `20` |
| net gap | `-20.62 pp`，95% CI `[-29.13, -12.50] pp` |

四个 root × scenario 的 gap 全部为负，说明 p_nav success 稳定高估 true-eta
到达。这不是 Goal83 的重复归因，而是 outcome contract 的正式后果。

## 2. 假设裁决

- H84-A supported：p_nav success 与 terminal GOAL `800/800` 一致；开关不改变
  steps、reward、terminal kind 或 trace；
- H84-B supported：eta gap 非零，方向为负，policy additive false-success 最大
  `0.05`；
- H84-C partial/not confirmed：strict125 heading tracking 大幅改善，但 p_nav
  outcome 下降，eta outcome baseline 持平、candidate 仅 `+3.13 pp`。

## 3. 为什么不换算法

- 当前证据指向 terminal approach 和估计器 frame，不是 NoisyNet 主导；
- p_nav-only 占 `23.1%`，说明 success 口径本身有严重高估；
- policy 与 zero 的 true-eta outcome 都在 `21%` 左右，无法支持“加探索或换 SAC
  就会自然改善”的结论；
- 先用 true-eta 口径查清 terminal authority，才能把算法差异和评估伪影分开。

## 4. 固定规则

```text
training / termination / deployment: p_nav_tight
research primary:                    eta_tight_ever
secondary:                           eta_tight_at_done
protocol label:                      terminal_eta_capture_at_done
```

`--success-frame {p_nav,eta,both}` 默认 `both`。以后不再每轮重开 p_nav/eta
归因；除非出现新的单因素假设，否则直接使用冻结的双口径输出。

## 5. 下一步

预注册 Goal85 的 strict125 terminal approach 审计，只读或最小 eval-only：

1. actuator response：rudder / stern / rpm 的 lag、gain、饱和；
2. terminal approach：closing speed、heading authority、末端转弯半径；
3. true-eta 坐标下的末段收敛与 terminal plane crossing；
4. 与 zero-residual 和 baseline 的同协议对照；
5. 若证明是探索/样本效率，再讨论 SAC；若证明是 authority/parameterization，
   先改动作或 controller，不换算法。

不修改 reward、checkpoint、seed、held-out 或训练预算。只有新 goal 的预注册
Gate 通过后才允许训练型 arm。
