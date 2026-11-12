# Blocker/Lead：Goal123 常数 stern bias 24/40，但还不是正式泛化结果

**日期**：2026-11-11<br>
**Goal**：Goal123<br>
**状态**：`BIAS_IMPROVES_CAPTURE / DIRECTIONAL_DEV40_ONLY`<br>
**触发条件**：Goal122 证明 PPO 方向有权限但无 capture；Goal123 找到净增益
常数方向。

## 1. 现象

```text
zero        20/40
stern-020   22/40  +31,56
stern-025   24/40  +29,31,33,56
stern-030   24/40  +29,31,33,56
```

无 lost、无 timeout/safety；剂量从 -0.05 到 -0.30 单调。

## 2. 已排除项

- 不是 seed 单点：4 个 gained、0 lost；
- 不是幅度无信号：-0.25/-0.30 复现同一 success 集合；
- 不是 safety/timeout：全 0；
- 不是 held-out 泄漏：`100–149` 关闭；
- 不是 PPO 注入分支：zero 和 ppo 都精确复现 Goal120。

## 3. 可证伪 hypotheses

### H123b-A：stern bias 是 default 场景的有害过拟合

- 成立时应看到：default 2.0 m 下 stern-025/030 明显低于 zero。
- 失败时应看到：default 不下降或下降 <=1。
- 最便宜证伪：Goal124 default guardrail，3 臂 × dev40。

### H123b-B：stern bias 是真实的终端参数化修正

- 成立时应看到：default 不下降，strict 的 +4 在重复评估中保持；
- 失败时应看到：default 明显下降，或 strict 只在单次 batch 出现。
- 最便宜证伪：Goal124 + 重复 strict 评估。

### H123b-C：bias-initialized PPO 能在 bias 上继续提升

- 成立时应看到：bias-initialized PPO 相对 fixed bias 再提升 `>=2`；
- 失败时应看到：PPO 仍把 residual 压回零或只做 seed 交换。
- 最便宜证伪：Goal124 通过后，预注册 10M bias-initialized PPO。

## 4. 当前决策

```text
decision = continue
next = Goal124 default guardrail + bias-initialized PPO design
```

不直接 SAC，不打开 held-out，不改 success/goal radius。
