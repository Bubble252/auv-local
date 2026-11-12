# Decision：Goal126/127 后继续 SAC P-A，但转向 reward/terminal credit

**日期**：2026-11-11<br>
**Goal**：Goal126 / Goal127<br>
**decision**：`continue`<br>
**next**：Goal128 eval-only reward/terminal credit audit

## 依据

1. Goal126 的网络 parity、init zero-residual 与 terminal gate 已通过；NoisyNet 不存在。
2. 评估重复 scale 已修复；所有剂量 run 的 scale manifest 一致，动作范围正确。
3. 剂量曲线的最佳点仍是 zero-residual：`24/33 > 22.75/32.62`。
4. 只有 86201 是局部 Survivor，不能写成语义上的算法增益。
5. 当前没有样本效率证据支持 REDQ/DroQ；也没有证据支持继续扫 SAC lr/UTD/entropy。

## 裁决

```text
主算法族：保持 SAC / P-A
动作 gain：不继续放大；0.125 只是诊断上限，不是正式配置
新增训练：暂停
下一实验：reward/terminal capture/credit audit
held-out：关闭
```

只有 Goal128 找到可解释、可预注册的 reward/terminal 缺陷后，才允许开新的训练 Goal。
