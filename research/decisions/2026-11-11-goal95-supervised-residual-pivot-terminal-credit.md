# Decision：从监督残差初始化 pivot 到 terminal capture / credit assignment 审计

日期：2026-11-11<br>
关联：Goal95<br>
决定：`pivot`

## 判据

Goal95 的 1-seed screen 没有达到预注册阈值：

- `P_D3`：default `30/40`、strict125 `19/40`；
- `P_BC15`：default `30/40`、strict125 `19/40`；
- 同 seed S0 是 default `31/40`、strict125 `20/40`；
- frozen BC 是 default `31/40`、strict125 `20/40`；
- residual/base 约 `0.55`，但 success 没有提高；
- PPO final 的 default 比 initial 少 1 个成功。

因此不能继续扩 seed，也不能把失败写成 seed 噪声。监督先验已经安装，但当前
PPO 更新没有把 residual 变成 terminal capture 增量。

## 为什么不直接开 SAC

Goal95 没有提供“方向正确、effective 增量存在，但探索/样本效率不足”的直接证据：

- path-zero 没有恶化 success 或 pre-goal RMSE，path 不是主因；
- residual/base 已经约 `0.55`，不是零解；
- 失败集中在 `3 m -> 2 m` 和 `2 m -> 1.25 m`，更像 terminal capture /
  reward-credit assignment；
- frozen BC 仍高于 PPO final，说明 PPO 可能在破坏先验。

在这些条件下直接换 SAC，可能只是用另一个算法重复同一个 terminal/credit
问题。必须先做只读审计，确认方向、effective survival 和 reward 分项。

## 新路线

Goal96：`terminal_credit_audit`，只读回放 Goal95 的 checkpoint 和 trace。

1. 截取 `10–5 m`、`5–3 m`、`3–2 m`、`2–1.25 m` 窗口；
2. 计算 terminal-window residual 方向、requested/effective/applied、逐通道 clip；
3. 分解 terminal/tracking/progress/action reward，检查 done 行 credit；
4. 比较 initial、final、frozen 的终端漏斗；
5. 按预注册判据决定：
   - `DIRECTION_POSITIVE_BUT_SAMPLE_INEFFICIENT` → 才开 SAC 同协议对照；
   - `TERMINAL_AUTHORITY_OR_DIRECTION` → 先修动作参数化/末端权限；
   - `REWARD_CREDIT_MISALIGNED` → 先修 reward/credit；
   - `PRIOR_BROKEN_BY_PPO` → 先修 PPO 更新/先验保持。

## 不做什么

- 不打开 held-out `100–149`；
- 不改 reward、terminal、success、goal radius；
- 不扩 Goal95 seed；
- 不扫 lr/epoch/gain/seed；
- 不在 Goal96 中途换算法。

## 证据

- `doc/goals/goal95_supervised_residual_pretrain_20261111.md`
- `doc/goals/goal95_result_20261111.md`
- `research/blockers/2026-11-11-goal95-supervised-residual-no-closed-loop-gain.md`
- `artifacts/lowlevel/goal95_supervised_residual_20261111/eval_dev40/*/summary.json`
- `artifacts/lowlevel/goal95_supervised_residual_20261111/traces_dev40/*/trace.npz`
