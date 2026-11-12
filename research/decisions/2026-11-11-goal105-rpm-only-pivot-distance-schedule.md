# Goal105 decision：静态 rudder 交互失败，pivot 到近终端 rpm 距离门控

**日期**：2026-11-11<br>
**状态**：`RPM_ONLY_SIGNAL / STATIC_INTERACTION_FALSIFIED / PIVOT_DISTANCE_GATED_RPM / NO_TRAINING`

## 1. 触发

Goal104 发现 `U050` 的 strict 方向（23/40）和 `R075/R125` 的弱 rudder 信号。Goal105
预注册 5 个 profile，检验 rpm 降权与 rudder 交互能否同时保住 default 33 和 strict 23。

## 2. 结果

- C0 精确复现 33/20；
- `U050` 复跑 strict 23/40，gained `25/51/56/59` 与 Goal104 完全一致；
- `U050R075` default 33、strict 22；
- `U050R125` default 32、strict 23；
- `U075` default 32、strict 22；
- path RMSE 在 rpm 降权下改善约 2.6%；
- 所有 run timeout/safety 为 0。

Gate：

```text
RPM_ONLY_SIGNAL
```

## 3. 判读

rpm 降权方向真实、可复现，但全局降权损失 default seed `26/33`，rudder 静态交互无法无代价
补回。`U050R075` 把 default 恢复到 33，却把 strict 收益从 23 降到 22。继续扫静态
rudder 或 rpm 常数没有明确机制依据。

## 4. 决策

```text
decision = pivot
```

新开 Goal106，只做 eval-only 的 **近终端 rpm distance schedule**：

- 远处 rpm gain 1.0；
- 接近目标时平滑降到 0.5/0.75；
- 固定 P_D3、scale 2.0、Goal85 unbiased、dev 20-59；
- Gate：default>=33、strict>=23、path<=1.05*C0、timeout/safety=0；
- 失败则停止 rpm 微调，进入 reward/terminal 目标审计；
- 不训练、不打开 held-out、不换算法族。

## 5. 证据

- `doc/goals/goal105_result_20261111.md`
- `artifacts/lowlevel/goal105_channel_interaction_20261111/phase0_summary.json`
- `artifacts/lowlevel/goal105_channel_interaction_20261111/phase0_summary.md`
