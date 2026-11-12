# Blocker：Goal128 后，SAC 的失败落在近区动作参数化与 terminal capture

**日期**：2026-11-11<br>
**Goal**：Goal128<br>
**状态**：`OPEN_FOR_ACTION_PARAMETERIZATION / REWARD_NOT_CAUSAL / HELDOUT_CLOSED`<br>
**上游**：Goal126 `SAC_PA_HARM`、Goal127 `SAC_PA_SCALE_INSUFFICIENT`

## 现象

同一批 8 个 1M SAC P-A checkpoint，在 dev40 上：

```text
zero:      strict 24.00 / default 33.00
0.125×:    strict 22.75 / default 32.62
1.0×:      strict 19.75 / default 28.25
```

降低残差只能缓解、不能反转损伤；只有 86201 在 1.0 获益（24→30）。
Goal128 的 34 条 trace 进一步显示：失败几乎全是 `GOAL_PLANE_MISS`，
不是 timeout；策略的 dense/step 和 terminal share 均比 zero 更差。

## 证据

1. **不是 reward farming**：zero strict 的 16 个失败全部 `GOAL_PLANE_MISS`；
   `GOAL_PLANE_MISS` 的 `reward_terminal=0`，一次性 miss penalty 约 -100；
   策略的 dense/step 反而更负（0.125 -0.582、1.0 -0.632 vs zero -0.577）。
2. **不是网络接线/NoisyNet**：PPO 返回 `[z_t, command, c_hat]`，SAC 返回
   `[z_t, zero_dade, command]`；D3 BC 的 `actor_command_dim=0`，128 列映射到
   `[z_t, zero_dade]`；初始化 parity 成立；无 `rho/sigma/noise` 参数。
3. **动作参数化有直接证据**：近区（eta≤10 m）raw policy 在 24.8%–82.0% 的步数
   贴 tanh 边界；1.0 下纯 learner 范数是 frozen BC 的 1.5–2.3 倍；
   pre64 eta 平均变差 +0.317 m（CI 排除 0）。
4. **terminal capture 是刀口**：zero 的 `success_margin_m_mean` 只有
   `0.0088–0.0241 m`，0.125 的小残差也能翻转 1–4 个 episode。

## 已排除

- NoisyNet 主导：当前 SAC 无 NoisyNet 参数；Goal59 旧 DDPG 线已由历史审计排除。
- 特征接线错误：初始化 parity 与 zero-residual control 精确复现。
- dense reward 刷分：策略的 dense 回报更差，不存在“牺牲 terminal 换取 dense 正收益”。
- timeout 拖死：zero/0.125 的失败全部是 plane-miss；1.0 只有 86204 的 3 个 timeout。
- 单纯 gain 太大：0.125 已把 learner 压到 BC 的 21%–27%，仍有 5 个 seed 轻微下降。

## 可证伪假设

| 假设 | 预测 | 最便宜证伪 |
|---|---|---|
| H128-B1：近区 learner 残差是主因 | inverse gate（10 m 外 1、3 m 内 0）应把 1.0 拉回 ≥24，且不损 default | 复用现有 checkpoint 做 eval-only gate |
| H128-B2：残差幅值而非方向是主因 | BC-relative norm cap（α=0.25）应让 7/8 seed 恢复，86201 仍获益 | eval-only norm cap |
| H128-B3：terminal credit 设计是主因 | 即使 gate/cap 保护近区，策略仍无法超过 zero | 上述 gate/cap 若全部 NO_EFFECT，才转 reward/terminal |
| H128-B4：样本效率不足 | 现有 checkpoint 的方向应已正确，只是 1M 不够 | 当前 7/8 seed 方向错误、dense 回报更差，暂不支持 |

## 裁决

```text
decision = pivot
主线从“继续扫 SAC / reward-terminal credit”转为
“eval-only 动作参数化 gate/cap screen → 训练型参数化 Goal”
```

不开 SAC formal、不换 REDQ/DroQ、不改 reward/terminal/success/goal radius、
不打开 held-out。只有 Goal129 的 gate/cap 达到 `ACTION_PARAM_GAIN` 或
`ACTION_PARAM_SAFE_PARITY`，才允许训练。
