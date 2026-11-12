# Blocker：Goal120 PPO optimization repair 仍无移动，且大残差没有改变闭环

**日期**：2026-11-11<br>
**Goal**：Goal120 PPO advantage-normalization × reward-scale screen<br>
**状态**：`PPO_OPTIMIZATION_LIMITED / RESIDUAL_CLOSED_LOOP_INERT`<br>
**触发条件**：所有 arm mean `<24/40`，且 `approx_kl` 最高仅 `2.22e-4`；同时
residual applied 约 `200–250 RPM`、effective/applied 约 `0.94–1.0`，但四臂
terminal share 和终态 eta 几乎相同。

## 1. 现象

dev40、`--action-mode policy`：

| arm | success 86201 | success 86202 | mean | KL mean | value loss mean |
|---|---:|---:|---:|---:|---:|
| P0 | 20 | 20 | 20.0 | 1.22e-4 | 0.032 |
| P1 | 20 | 20 | 20.0 | 1.45e-4 | 0.022 |
| P2 | 21 | 20 | 20.5 | 2.22e-4 | 5.11 |
| P3 | 21 | 20 | 20.5 | 8.74e-5 | 4.72 |

- timeout/safety 全 0；
- 唯一翻转是 seed 56，仅 P2/P3 的 86201 训练 seed；
- 5 个可达 failure 的 terminal share 在三位小数上四臂相同；
- 2 m entry margin 仍 `1.51–1.52`。

residual 统计（seed 26）：

```text
applied mean       248.6–250.5 RPM
effective/applied  0.943–0.954
1 s 低通/原信号    0.885–0.903
高频/原信号        0.221–0.223
```

residual 既没有被 clamp 掉，也不是纯高频抖动；但 final eta 只差约 `0.01–0.02 m`。

## 2. 已排除项

- 不是 evaluation 协议：全部用 `--action-mode policy`，P0 复现 Goal119；
- 不是 reward scale 完全没接上：P2/P3 的 KL 和 value loss 都发生了变化；
- 不是训练时长：10M、4 epochs、32 mini-batches、critic warmup=1；
- 不是 residual 幅度为零：applied 约 200–250 RPM；
- 不是 safety/timeout：全 0；
- 不是 held-out 泄漏：`100–149` 关闭；
- 不是 NoisyNet：当前 PPO 无 NoisyNet，eval 是 deterministic mean；
- 不是网络接线：Goal103/108 已做 22/22 frozen-BC 张量 `torch.equal`，
  actor `131→256→128→3`，state encoder `20→128→128→60`。

## 3. 可证伪 hypotheses

### H120b-A：动作权限在该闭环里近乎惰性

- 机制：B20 的 BC 残差已接近/超过 rudder/stern 包线，额外的 PPO 增量被
  执行器/速率限制或闭环反馈抵消；`effective/applied≈1` 只说明动作合成没被
  action clamp，不代表最终执行器/plant 有额外权限。
- 成立时应看到：注入 `2x/4x PPO` 或常量 rudder/stern/rpm residual，reachable
  failure 的终态 eta 仍变化 `<0.05 m`，success 集合不变。
- 失败时应看到：任一注入使终态 eta 移动 `>0.25 m` 或翻转 success。
- 最便宜证伪：Goal121 eval-only action-authority injection，5 个可达 failure
  加 5 个 success，9 个 profile。

### H120b-B：PPO 学到了错误方向，而不是没有权限

- 机制：residual 有权限，但 PPO 的 advantage/terminal credit 让方向错误；
  `neg-PPO` 或 `2x/4x PPO` 应产生不同闭环结果。
- 成立时应看到：`neg-PPO` 的终态 eta 与 `PPO` 明显不同，且 `neg-PPO` 的
  success 更高或 terminal miss 更小。
- 失败时应看到：`PPO` 与 `neg-PPO` 的终态 eta 几乎相同。
- 最便宜证伪：Goal121 的 `ppo / neg-ppo / ppo-2x / ppo-4x` 四臂。

### H120b-C：优化/样本效率不足

- 机制：动作权限存在，方向也可分，但 PPO 的 on-policy 长 episode 信用分配
  无法找到正确 residual；replay 方法能复用 terminal 样本。
- 成立时应看到：Goal121 证明 `2x/4x` 或常量注入能移动终态 eta，但 PPO 的
  learned residual 方向没有 success 增益。
- 失败时应看到：Goal121 证明 residual 通道惰性，则 SAC 也无能为力。
- 最便宜证伪：先做 Goal121；只有 H120b-A 被否定、H120b-B 成立时，才预注册
  SAC 同协议对照。

### H120b-D：surrogate 指标本身失效

- 机制：normalized advantage 零均值使 `mean(-adv*ratio)≈0`，因此
  `surrogate≈1e-4` 不是“没有梯度”。
- 成立时应看到：P1/P3 的 KL 与 P0 同量级，但 surrogate 仍接近 0；用
  `mean(|adv|*|ratio-1|)` 或 explained variance 才能反映更新强度。
- 失败时应看到：归一化后 surrogate 仍随 KL 同向大幅变化。
- 最便宜证伪：Goal121 不改训练，但下一轮 PPO 必须记录替代指标。

## 4. 当前决策

```text
decision = continue
next = Goal121 eval-only action-authority injection
```

按用户规则，残差很大但闭环不动时先查动作参数化，不直接换 SAC。只有 Goal121
证明“权限存在、方向可分，但 PPO 学不到”时，才开 SAC 同协议对照。
不改 success/goal radius，不打开 held-out，不修改 Goal120 的冻结 Gate。
