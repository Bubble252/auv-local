# 决策：Goal94 后从动作参数化转向 supervised residual pretraining

日期：2026-11-11<br>
Goal：Goal94 → Goal95<br>
决策：`pivot`<br>
算法族：仍为 PPO；不换 SAC/REDQ/DroQ

## 1. 决策

Goal94 已经排除“单纯速率截断”和“单纯 PPO 幅度太小”：

- rate-aware 让 PPO fin survival 到 `1.0`，但 PPO/BC fin ratio 仍只有
  `0.037–0.052`；
- `F15` fin gain 4 把比例提到 `0.19–0.21`，success 仍是 `31/21`；
- rate-aware 还带来 `+0.059/+0.099 m` 的 path 代价和 default `30/29`。

因此继续在环境侧加 gain/headroom 的收益很低。下一步 pivot 到
**supervised residual pretraining + 同协议 PPO 微调**：

```text
先学习 r_sup ≈ D3 − (ALOS + scale * r_BC) 或 D3 − ALOS
再把 residual actor 作为 PPO 初始化
PPO 只在同一 1M/dev40/default+strict125 契约上有界微调
```

这不是换主算法，也不改 reward/terminal/success/goal radius。

## 2. 为什么不是 SAC

预注册的 SAC 五条件没有满足：

1. total fin clip `<=0.5`：`F15` 0.53–0.54，rate-aware 0.76–0.86；
2. min survival `>=0.5`：满足；
3. PPO/BC fin ratio `>=0.15`：只有 `F15` 满足；
4. 方向分离且 bootstrap CI 下界 `>0`：不满足，成功/失败 cos 分离但符号为负；
5. strict125 `<=20/40`：`F15` 为 21。

没有“方向对但样本效率不足”的直接证据；现在开 SAC 会把动作参数化和
信用分配两个问题混在一起。

## 3. 下一步 Gate

Goal95 预注册：

| arm | residual 初始化 | PPO | seeds | 预算 |
|---|---|---|---:|---:|
| `S0` | 当前零初始化 | 1M | 86201 | 1M |
| `P-D3` | 监督 `D3−ALOS` | 1M | 86201 | 1M |
| `P-BC1.5` | 监督 `D3−(ALOS+1.5*r_BC)` | 1M | 86201 | 1M |

先跑 1-seed screen；只有 `P-*` 的 fin ratio `>=0.15`、direction 有分离、
strict `>=21/40` 且 path 不差于 S15 `>0.02 m`，才扩 3 seeds。

## 4. 不允许

- 不在 Goal94 记录后改阈值或把 `F15 +1/40` 写成稳定收益；
- 不直接开 SAC/REDQ/DroQ；
- 不打开 held-out `100–149`；
- 不改 reward/terminal/success/goal radius；
- 不 `git add -A`，不自动 commit/push。

## 5. 证据

- `research/blockers/2026-11-11-goal94-rate-aware-no-authority.md`
- `doc/goals/goal94_phase2_result_20261111.md`
- `doc/goals/goal94_s15_result_20261111.md`
- `research/LEDGER.jsonl`（Goal94 entry）
