# Blocker：Goal122 PPO gain sweep 有权限但无 capture，64x 伤害

**日期**：2026-11-11<br>
**Goal**：Goal122<br>
**状态**：`GAIN_HARM / GAIN_AUTHORITY_WITHOUT_CAPTURE`<br>
**触发条件**：16x/32x 明显移动终态 eta，但 success 不升反降；64x 掉到 13/40。

## 1. 现象

```text
ppo-8x   21/40  Δeta 0.174 m
ppo-16x  20/40  Δeta 0.313 m  (+2/-2)
ppo-32x  19/40  Δeta 0.454 m  (+4/-5)
ppo-64x  13/40  Δeta 0.612 m  (+5/-12)
```

## 2. 已排除项

- 不是没有权限：Δeta 随 gain 单调增大；
- 不是幅度太小：16x 已经 0.31 m；
- 不是 zero/ppo 失效：zero 20/40、ppo 20/40 精确复现；
- 不是 safety/timeout：全 0；
- 不是 held-out 泄漏：`100–149` 关闭。

## 3. 可证伪 hypotheses

### H122b-A：PPO 方向被无效/反向分量污染

- 成立时应看到：常数 `stern-` 净增益，而 PPO 放大是 +2/-2、+4/-5；
- 失败时应看到：PPO 放大与常数 bias 一样净增益。
- 当前证据：SUPPORTED。

### H122b-B：终端 capture 需要常数 bias，不需要复杂策略

- 成立时应看到：更细的 stern bias sweep 能稳定 `>=23/40` 且 lost<=1；
- 失败时应看到：所有常数 bias 都 <=22，+2 只是边界噪声。
- 最便宜证伪：Goal123 常数 bias 标定。

### H122b-C：SAC 能学到 capture-aligned 方向

- 成立时应看到：在相同 action authority 下，SAC 的 deterministic residual
  方向与 `stern-` 对齐，并提升 success；
- 失败时应看到：SAC 也学到 +/− 交换或饱和。
- 最便宜证伪：只有 Goal123 找不到稳定 bias 时才开 SAC；不要先开。

## 4. 当前决策

```text
decision = continue
next = Goal123 constant stern/rpm bias calibration
```

不换算法、不改 success/goal radius、不打开 held-out。
