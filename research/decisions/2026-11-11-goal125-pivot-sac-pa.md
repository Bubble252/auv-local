# Decision：Goal125 后转向 SAC P-A 同协议对照

**日期**：2026-11-11<br>
**Goal**：Goal125<br>
**decision**：`pivot`<br>
**next**：Goal126 SAC P-A 1M screen（固定 stern bias）

## 依据

1. B15/B20/BP3 都停在 fixed-bias control 的 24/40，PPO 没有在其上产生增量。
2. Z15 退回 20/40，证明 fixed bias 本身有效，不是 Gate 噪声。
3. 训练残差只有 base 的 `0.7%–4.0%`，KL 约 `1e-4`；这是优化受限，不是动作幅度过大。
4. 网络/初始化 parity 通过，NoisyNet 不存在，不能继续把失败归因于网络接线。
5. Goal91 已排除低噪声，Goal92 已排除更新调度，Goal120 已判 optimization-limited；
   重复 PPO 参数扫描不再有信息增益。

## 裁决

```text
fixed bias prior = 有效
PPO 在 bias prior 上 = 无增量
SAC eligibility = 满足，但仅限 P-A 架构
```

Goal126 必须满足：

- `bc_path_dim=8`、endpoint `4→32→64`、加载 Goal86 BC encoder；
- 初始 `mu=0`、`log_std=-3.0`，initial zero-residual 精确复现 fixed-bias control；
- 训练动作接口、bias、eval scale、success/goal radius、dev seeds 与 Goal125 一致；
- 先 1M screen；只有 screen 达 Gate 才扩 5M formal；
- 不打开 held-out，不自动 commit/push。
