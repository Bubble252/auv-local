# Goal108 decision：停止 PPO/SAC 优化器轴，转 terminal capture / residual direction

**日期**：2026-11-11<br>
**状态**：`DECISION_RECORDED / PIVOT_TO_GOAL109`

## 1. 触发

Goal108 预注册的 B20-PPO 与 B20-SAC 3-seed 1M 对照完成。两个 arm 都没有达到
`PASS_PRIOR_INCREMENT`：

```text
PPO: default 33/40, strict125 20/40（3/3 seeds）
SAC: default mean 6.0/40, strict125 mean 3.0/40
```

PPO 的增量 L2 ratio `0.0060–0.0095`，闭环没有移动；SAC 的 applied residual/base
L2 `0.74–1.38`，action 接近饱和，并出现 timeout/safety。网络审计确认没有
NoisyNet，也没有 checkpoint 或维度错位。

## 2. 决策

```text
decision = pivot
```

停止 PPO/SAC 优化器轴，不扩 5M，不引入 REDQ/DroQ/DADE，不继续扫 residual
scale/channel gain/rpm。原因不是“优化器完全无关”，而是本预注册 Gate 已经证明：

1. B20 基座没有被破坏；
2. PPO 零初始化增量太小，无法改变闭环；
3. SAC 的 replay/熵优化在没有残差约束时会放大确定性 mu 到饱和；
4. 继续加预算或换 off-policy 算法会重复同一条失败轴。

## 3. 下一步

新开 Goal109，先做 eval-only / read-only 审计：

- terminal capture 几何与 terminal event 是否把失败集中在最后阶段；
- reward terminal / dense reward 的信用份额是否仍被路径项稀释；
- residual direction 是否在失败种子上与闭合方向相反；
- action parameterization 是否存在“基础残差已大、增量再饱和”的结构性问题。

Goal109 不修改 reward/terminal/success/goal radius，不打开 held-out，不训练新
算法；只有审计给出明确、可证伪的机制后，才允许另开训练型 Goal。

## 4. 证据

- `doc/goals/goal108_result_20261111.md`
- `doc/goals/goal108_network_and_noisenet_audit_20261111.md`
- `research/blockers/2026-11-11-goal108-no-prior-optimizer-gain.md`
- `artifacts/lowlevel/goal108_b20_prior_optimizer_20261111/`

