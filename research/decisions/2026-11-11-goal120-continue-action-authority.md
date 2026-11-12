# Decision：Goal120 判 PPO_OPTIMIZATION_LIMITED，先做 action-authority injection，不直接开 SAC

**日期**：2026-11-11<br>
**Goal**：Goal120<br>
**decision**：`continue`<br>
**next**：Goal121 eval-only action-authority injection

## 依据

1. P0/P1/P2/P3 的 dev40 success 为 `20.0 / 20.0 / 20.5 / 20.5`，全部低于
   预注册的 `24/40`。
2. `approx_kl` 最高 `2.22e-4`，仍低于 `1e-3`；`surrogate` 对 normalized
   advantage 失效，不能作为“无梯度”的独立证据。
3. residual applied 约 `200–250 RPM`，effective/applied 约 `0.94–1.0`，
   1 s 低通仍保留约 `0.89–0.91` 幅度，但四臂 terminal share、2 m margin 和
   final eta 几乎相同。
4. 这满足用户预先设定的“残差很大但闭环不动，先查动作参数化、reward 和
   terminal capture”条件。reward 和 terminal capture 已在 Goal119/117/118/103
   检查；剩余最便宜的是动作权限注入。

## 不直接开 SAC 的理由

SAC 只有在以下条件同时成立时才值得开：

```text
action authority exists
and residual direction is discriminable
and PPO still fails to find it
```

当前只证明了 PPO 优化信号弱，没有证明 residual 通道有足够权限。若通道本身
惰性，SAC 只会换一种方式产生同样无效的 residual。

## 下一步

Goal121：

- 固定 B20 base、Goal119 C3 环境、dev seeds、success frame；
- 注入 `zero / ppo / neg-ppo / ppo-2x / ppo-4x / const-rudder / const-stern /
  const-rpm+ / const-rpm-`；
- 主指标：reachable failure 的终态 eta 变化、success 翻转、terminal miss；
- 若权限存在且方向可分，再预注册 SAC（Goal86 P-A 架构：`bc_path_dim=8`、
  独立 endpoint `4→32→64`、加载 BC encoder、z_t 融合）；
- 若权限惰性，先修动作参数化/终端几何，不换算法。

不自动 commit/push，不打开 held-out。
