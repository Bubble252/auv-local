# Blocker：Goal103 的 authority 标签触发了，但不是因果判别器

**日期**：2026-11-11<br>
**Goal**：Goal103<br>
**状态**：`AUTHORITY_SIGNATURE_NON_DISCRIMINATIVE / RESIDUAL_PARAMETERIZATION_AUDIT_REQUIRED`

## 1. 现象

B20 strict125 的 20 个失败 seed 在预注册分类下满足
`CONTROL_AUTHORITY_LIMITED`：14/20 有 `radial_excess > 0.25 m` 且
magnitude/rate clip `>= 0.5`。

但进一步配对后：

| outcome | magnitude clip median | rate clip median |
|---|---:|---:|
| failure | 0.746 | 0.500 |
| success | 0.872 | 0.613 |

成功集反而更贴边。因此 clip 只能说明“动作经常在边界”，不能区分成功/失败。

## 2. 诊断语义问题

`material_saturation_rate` 全为 0，不代表没有物理饱和。actuator residual wrapper
在调用环境前已经：

```text
requested = base + residual * scale * full_scale
executed  = clamp(requested, lower, upper)
```

然后把 `executed` 归一化送入环境的 action pipeline。环境看到的已经是边界内动作，
所以 `material_saturated` 不会再次报告这次 clamp。该字段不能用于判断 wrapper 自己
造成的饱和；应使用 `actuator_residual_magnitude_clipped` / `rate_clipped`。

## 3. 已排除项

| 假设 | 结论 | 证据 |
|---|---|---|
| 隐藏层宽度/接线错误 | 排除 | 当前 `pc_policy.py` SHA 与 Goal89 parity 一致；22 个 frozen 张量 `max_abs_diff=0.0` |
| path12 / endpoint4 错位 | 排除 | checkpoint metadata 与 Goal86/89 契约一致 |
| checkpoint 错配 | 排除 | P_D3/P_BC15 SHA 与 Goal100/101/102 冻结值一致 |
| NoisyNet 主导 | 排除 | deterministic mean，当前无 NoisyNet |
| p_nav/eta 延迟主导 | 不支持 | unbiased `nav_age=0`；3/20 pnav-only 只是噪声导致的 ever-capture 边界事件 |

## 4. 可证伪假设

### H103X-A：残差幅值/饱和是剩余失败的主因

如果成立：降低或重分配 residual 包线后，strict 明显提高，且 success/failure
的 clip 差异应变成可区分。

当前证据：不充分。成功集 clip 更高；幅值本身不是判别器。需要通道级干预才能检验。

### H103X-B：通道方向与终端几何是主因

如果成立：分别改变 rudder/stern/rpm 的 scale 或 joint norm 后，某些垂直主导 seed
会翻转，而 horizontal 主导 seed 不会同步翻转。

最便宜证伪：eval-only 通道 scale 小网格，dev seeds `20–59`，default+strict125。

### H103X-C：reward/terminal capture 是主因

如果成立：即使通道方向正确且不再额外贴边，success 仍不动，失败仍集中在同一
plane-crossing 几何。

当前不做：必须先完成 H103X-B 的动作参数化审计，否则无法区分 reward/terminal 与
动作包线。

### H103X-D：网络容量不足

如果成立：在冻结输入/输出契约下增加宽度或深度会改变终端方向；但当前没有任何
“方向可表达但容量不够”的证据。排除优先。

## 5. 决策

`decision = pivot`

不训练、不换算法、不扩 seed。下一步只做 eval-only 的通道级动作参数化审计；
只有该审计证明方向可用但包线/参数化吞掉修正，才考虑 bounded PPO；只有动作侧
不再有可分离效应，才进入 reward/terminal audit。
