# Goal102 decision：PPO 剂量无窗口，转向 B20 terminal capture 审计

**日期**：2026-11-11<br>
**状态**：`PPO_DOSE_HARM / NO_SAC / PIVOT_TERMINAL_AUDIT`

## 1. 触发

Goal101 已经证明 B20 在同 checkpoint 配对中超过 ALOS（两臂 default `33>27`、
strict `20>14`）。Goal102 进一步检查 Goal95 训练轨迹上的 PPO 剂量-反应：
是否存在更早 checkpoint，既保留 default `33`，又把 strict 提到 `21+`。

## 2. 结果

| dose | P_D3 default/strict | P_BC15 default/strict |
|---|---:|---:|
| `I0` | 32/20 | 32/20 |
| `D25` | 32/19 | 32/19 |
| `D50` | 32/18 | 31/18 |
| `D75` | 32/18 | 32/18 |
| `D100` | 31/18 | 32/18 |

Gate 判 `PPO_DOSE_HARM`：

- 没有 checkpoint 达到 default `>=33` 且 strict `>=21`；
- `I0` 已经不是 B20 的 33/20；
- 最小 PPO 剂量 D25 就把 strict 从 20 降到 19；
- path ratio 约 1.003，退化集中在终端 capture 的离散翻转；
- timeout/safety 全 0。

## 3. 判读

1. 不是样本效率不足：不需要 SAC/REDQ/DroQ。
2. 不是 checkpoint 选择问题：五个剂量没有可用窗口。
3. 不是 path tracking 整体退化：path 几乎不变，success 却下降。
4. 当前 PPO 更新从最小剂量起就破坏 frozen BC 先验；继续加训练预算没有依据。
5. B20 已经是强于 ALOS 的可用基线；剩余 strict 失败是 terminal capture /
   action parameterization 问题，应先审计失败几何和有效动作方向。

## 4. 决策

`decision = pivot`

下一 Goal（Goal103）先做 **B20 strict 失败的动作参数化与 terminal capture
审计**，不立即训练新算法：

- 对 B20 strict 的 20 个失败 seed，检查 goal radius 邻域、plane-miss 的横向/
  纵向分量、final-distance 壳层；
- 检查 actuator saturation、rate clip、terminal gate 是否吞掉有效修正；
- 检查失败是否集中在少数几何模式；
- 只有审计证明“存在可用动作方向，但当前 PPO 目标/约束无法表达”时，才开
  hard-bounded prior-preserving PPO；
- 不改 success/terminal/goal radius，不打开 held-out，不开 SAC/REDQ/DroQ。

## 5. 证据

- `doc/goals/goal102_ppo_dose_response_20261111.md`
- `doc/goals/goal102_result_20261111.md`
- `artifacts/lowlevel/goal102_ppo_dose_response_20261111/phase0_summary.json`
- `artifacts/lowlevel/goal102_ppo_dose_response_20261111/phase0_summary.md`
- `doc/goals/goal101_result_20261111.md`
