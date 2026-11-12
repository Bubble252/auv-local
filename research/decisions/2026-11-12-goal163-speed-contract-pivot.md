# Goal163 decision：先修速度语义，再谈终端 anti-windup 和 RL 对照

**日期**：2026-11-12<br>
**决策**：`pivot`<br>
**上游**：Goal162 `INVALID_SPEED_CONTRACT`、Goal163 只读审计

## 1. 决策

Goal162 的结果不能继续用于比较 gain residual、PPO/SAC/REDQ/DroQ，也不能扩 seed。
先把速度契约修好：

- `nu` 定义为地速；
- `water_velocity_body = nu - current_body`；
- DVL bottom 返回 `nu`，DVL water 返回 `nu - current_body`；
- actor/reward/evaluation 必须明确自己消费的是 ground 还是 water-relative；
- 控制器继续用 water-relative 跟踪 ALOS 命令，但 reward/观测不能把两者混用。

## 2. 为什么不是 continue

旧评价把 `nu + current` 当作地速，导致 current10/current15 的超限比例被高估
20–30 倍；真实超速只剩 terminal 段 0.31%/0.51%。在这种契约下继续调 RL，
等于用被海流偏置的 reward 和 actor 输入去比较算法，结论不可解释。

同时，真实超速确实存在：

- current10 峰值 `2.546 m/s`，current15 峰值 `2.598 m/s`；
- 超速全部在 terminal phase；
- 超速步 integral 接近 cap，anti-windup 是次要但真实的修复项；
- current15 的 `GOAL_PLANE_MISS` 失败 episode 有约 7 s 的轻微超速。

所以不是“评价 bug 可以忽略”，而是“先修语义，再修终端速度环”。

## 3. 为什么不是 stop

速度语义错误是可定位、可版本化修复的实现问题，不是方法路线失败。修正后仍可
复用同一 SAC、同一 z_t 架构、同一 reward/terminal/success/radius 和同一
训练/评估协议；不需要换算法，也不需要重写整个项目。

## 4. 下一步（Goal164）

1. 新增版本化速度契约（例如 `nu_ground` / legacy），旧配置显式保留 legacy；
2. 修正 `geometry.ground_velocity_body`、`sensors.py`、PC env 的 actor/reward/
   evaluation 消费链；
3. 增加合成测试：`eta` 差分 = `nu`；ground helper = `nu`；
   water helper = `nu - current`；
4. 用同一批 Goal162 zero seed 做 corrected replay，不训练；
5. 若 corrected replay 仍有 terminal 超速，再做最小 speed-loop anti-windup /
   terminal deceleration 修正；仍不换 RL 算法；
6. 速度契约恢复后，重训 Goal162 gain residual，使用同一 3 seeds × 1M 协议。

## 5. 证据

- `doc/goals/goal163_result_20261112.md`
- `research/blockers/2026-11-12-goal163-speed-semantics-and-terminal-overspeed.md`
- `artifacts/lowlevel/goal163_speed_audit_20261112/replay_analysis/speed_audit.{json,md}`
- `tests/test_pc_goal163_speed_semantics.py`

