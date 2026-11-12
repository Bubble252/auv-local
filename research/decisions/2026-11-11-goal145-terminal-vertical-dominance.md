# Decision：Goal145 确认 T2 横向 path 代价，C1 纵向末端瓶颈

**日期**：2026-11-11<br>
**决策**：`continue`（保持 PPO/SAC 主算法与 C-line，不改 success/terminal/goal radius）<br>
**上游**：`doc/goals/goal145_result_20261111.md`<br>
**证据**：`artifacts/lowlevel/goal145_terminal_capture_vertical_audit_20261111/audit_summary.md`<br>

## 裁决

```text
T2-SAC:  HORIZONTAL_OR_MIXED_PATH_DEGRADATION
C1-PPO:  C1_TERMINAL_VERTICAL_DOMINANCE
C1-PPO:  C1_RESIDUAL_NO_CLOSED_LOOP_EFFECT
```

- T2-SAC vertical MSE share 约 `0.04`；path 增量主要来自 horizontal RMSE `+0.316 m`，vertical RMSE `-0.004 m`。
- C1 失败的 final vertical miss `1.572 m`，horizontal `0.324 m`；成功的对应值为 `0.777 m` / `0.829 m`。
- C1 command/actuator 的 policy 与 zero 都 `26/40`，success flip 为 0；400k 仍 `0.65/0.35`。
- C1 失败不是远距离跟踪差，而是 near-goal 的纵向末端捕获；13/14 个失败在 `1.25–2.0 m` 外壳。

## 为什么不是直接扫奖励

当前 C1 reward 已经有 `w_approach_vertical=1.0`、`w_approach_pitch=0.5`、`approach_enter_m=5`、`approach_full_m=2`。在 residual 闭环无效果的前提下，直接扫更大的纵向权重无法区分：

1. reward 尺度不足；
2. pitch/speed guidance 或 residual 权限不足；
3. terminal capture 语义本身不可达。

因此只允许新开一个最小预注册 pilot，不允许在 Goal145/Goal140 内直接扫多个权重。

## 下一步

1. 写 Goal146：C1 terminal vertical capture 的最小可证伪 pilot。
2. 首选 eval-only 单变量：terminal pitch/speed guidance 或 residual 权限；先看能否把 7 个 `1.25–1.5 m` 失败中的至少 2 个转成 success。
3. 若 eval-only 证明“动作权限足够但 reward 不推动策略”，再接入 Goal140 的 potential-based vertical reward；否则不把 reward 当作主修复。
4. 不换 REDQ/DroQ；不打开 held-out；不修改 success/radius；不继续 T2 path 残差。

## 明确不做

- 不用 T2 的弱 success 信号启动纵向 reward；
- 不因 C1 失败直接上 5M formal；
- 不按 best seed / best checkpoint 选结果；
- 不把 `p_nav` 或 residual 幅值单独当成 success 证据。
