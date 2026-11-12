# Blocker：Goal135 authority-regularized SAC 全 arm harm

**日期**：2026-11-11<br>
**状态**：`AUTHORITY_SAC_HARM`<br>
**预注册**：`doc/goals/goal135_authority_regularized_sac_20261111.md`

## 现象

24 个 1M SAC run（3 arms × 8 seeds）和 96 个 dev40 评估全部完成。三个 arm 的
nominal/current15 都低于 zero 参照，且没有达到任何跨 seed 目标：

| arm | nominal strict mean | nominal default mean | current15 strict mean | current15 default mean | target seeds |
|---|---:|---:|---:|---:|---:|
| `R030` | 21.00 | 31.625 | 13.875 | 23.875 | 1/0 |
| `R025` | 20.875 | 31.25 | 13.00 | 23.25 | 0/1 |
| `R020` | 22.125 | 31.125 | 13.75 | 23.875 | 0/1 |

zero 参照：nominal `24/33`，current15 `14/24`（strict/default）。所有 run finite，
timeout/safety 为 0。

## 排除项

- 不是正则未生效：RMS 随 rho 单调，manifest 有统计。
- 不是 NoisyNet：本 Goal 禁用 NoisyNet。
- 不是数值/接线：24 个 run、96 个评估全部 finite。
- 不是 reward/terminal/radius 改变：这些契约完全冻结。
- 不是单个 seed 噪声：三个 arm 的均值都低于 zero，且目标 seed 数不足。

## 可证伪假设

1. **H135-A：residual RMS 是主要 harm 原因，rho=0.20 应把 nominal 恢复到 zero 以上。**
   证据：`FALSIFIED`；R020 nominal strict mean 22.125，default 31.125，均低于 zero。
2. **H135-B：86201 的低 authority 优势可以在 rho=0.20 下复制到其他 seed。**
   证据：`FALSIFIED`；R020 的 86201 为 23/29 和 12/25，低于历史 30/33 和 16/29；
   其他 seed 没有形成 `zero+2` 的跨 level 信号。
3. **H135-C：authority 强度存在可用剂量窗口。**
   证据：`WEAK_ONLY`；R020 的 nominal mean 略高于 R025/R030，但三档全部 harm。
4. **H135-D：剩余失败是探索/样本效率不足，而不是动作/credit 问题。**
   证据：`NOT_ESTABLISHED`；本 screen 没有直接测量探索覆盖；不能据此开 SAC 算法替换。

## 最便宜证伪

在同一 8 个 seed 上做只读 authority/方向分解：比较成功与失败 episode 的 residual
方向余弦、terminal-gate 前后 RMS、以及同 seed 的 zero paired success。该分析不需要
新训练。如果成功/失败 episode 的 authority 分布没有方向差异，则停止“再调 cap/rho”，
转 reward/state/terminal 或 diversity。

## 决策

`pivot`：停止 authority 正则族，不进入 5M formal。先执行 Goal136 D-line，以训练分布
多样性检验“方向和覆盖不足”这一替代解释；PPO 保留独立同协议 screen。
