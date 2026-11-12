# Goal146 决策：rate limiter 不是 C1 纵向捕获瓶颈，转向 pitch authority / terminal slowdown

**日期**：2026-11-11<br>
**Decision**：`continue`<br>
**Next**：Goal147 eval-only terminal pitch authority pilot

## 1. 证据

Goal146 在固定 `C1_000300000`、dev40、policy/zero 配对的条件下测试了四个冻结 arms：

| arm | 改动 | success | near failures converted | pitch clip | speed clip |
|---|---|---:|---:|---:|---:|
| Z | 原 guidance | 26/40 | — | 0.613 | 0.199 |
| P1 | pitch 20 deg/s | 26/40 | 0/7 | 0.527 | 0.199 |
| P2 | pitch 30 deg/s | 26/40 | 0/7 | 0.407 | 0.199 |
| S1 | speed 0.35 m/s² | 26/40 | 0/7 | 0.612 | 0.148 |

因此两个 rate-limiter 假设都被证伪。`policy` 与 `zero-command-residual` 逐项相同，
residual 平均绝对值约 `1.5e-4`，排除了“residual 很大导致闭环异常”的解释。

更关键的是失败终端几何：

```text
失败组 2 m：|target pitch| = 0.782 rad
             |actual pitch| = 0.289 rad
             |pitch error| = 0.423 rad
             |stern| = 0.317 rad，20 deg 上限为 0.349 rad
成功组 stern 近终点饱和比例 = 0.085
失败组 stern 近终点饱和比例 = 0.436
P2 失败组饱和比例 = 0.545
```

放宽 limiter 让 target 更容易通过，但 target-limited gap 只有 `0.034 -> 0.014 rad`；
真正缺的是**更早的 pitch 响应或更多 terminal 时间**，而不是 command rate。

## 2. 假设

### H147-A：terminal slowdown 给 pitch loop 更多时间

机制：terminal 20 m 内把速度按 `terminal_braking_enabled + terminal_speed_ratio`
从巡航速度降到 0.85/0.75，增加在每个 goal-distance shell 的停留时间，让实际 pitch
有机会追上 target，减少 1.25–2.0 m 的纵向 miss。

成立应看到：至少 2/7 个 Z 的近终点失败转 success；failure `|pitch error|@2m` 下降；
timeout/plane-miss 不增加。

失败应看到：最终 pitch error 和 vertical miss 不变，或 slowdown 让 timeout 增加。

### H147-B：pitch controller gain 是早期跟踪瓶颈

机制：当前 `pitch_gain=2`、`pitch_rate_gain=1` 在 5 m 时输出不足，实际 pitch 只有
目标的约 37%；提高 `pitch_gain` 到 4（或 `pitch_rate_gain` 到 2）可使 stern 更早接近
极限，改善 2 m 前 pitch 跟进。

成立应看到：failure `|pitch error|@2m` 下降，至少 2/7 转化，且 stern 饱和不导致新的
plane miss 或 overshoot。

失败应看到：success 仍 26/40，或者饱和增加但 terminal error 不变。

## 3. 约束

- 只做 eval-only，固定 `C1_000300000`、dev seeds 20–59；
- 每个 arm 必须 policy/zero 配对；
- 先单变量，不把 slowdown 与 gain 同时改；
- 不接 Goal140 reward，不换 PPO/SAC，不扫 seed，不打开 held-out；
- 不改变 success、terminal、goal radius；
- 如果 controller override 需要新代码，只允许显式 `pitch_gain` / `pitch_rate_gain`
  eval-only 字段，并在报告里记录 training contract 未变。

## 4. 下一步

写 `goal147_c1_terminal_pitch_authority_20261111.md`，预注册：

```text
Z  = 原 terminal guidance
B85 = terminal_braking=true, terminal_speed_ratio=0.85
B75 = terminal_braking=true, terminal_speed_ratio=0.75
K4  = base_controller.pitch_gain=4
R2  = base_controller.pitch_rate_gain=2
```

`B85/B75` 用 guidance config 即可；`K4/R2` 需要 evaluator 的显式 eval-only controller
override。先跑四臂，Gate 仍是至少 2/7 近终点失败转化、success loss <=1、plane miss
不恶化超过 2、timeout/safety 不增加。

只有 Goal147 也证明“权限已经足够但 policy/reward 没有利用”时，才启动 Goal140 的
potential-based vertical reward；否则继续修 terminal target / action authority。
