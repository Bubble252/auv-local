# Goal33 blocker：垂直权限分解没有支持机制

**日期**：2026-11-09<br>
**Goal**：Goal33<br>
**触发**：Goal32 `AUTHORITY_LIMITED_SUPPORTED` 后的三项预注册分解全部失败；现有
authority-magnitude 假设无法解释失败，属于 required autoresearch 的
`hypothesis_unexplained` / `no_metric_movement` 记录。

## 1. 现象

Goal33 使用 Goal31 冻结的四组 40-seed formal traces，无训练、无新 rollout、
held-out 未打开。结果：

- H-SPEED：速度-authority Spearman 为 `-0.1738--0.0358`；高速组减低速组
  authority 仅 `-0.0163-0.0465 m/s`，均低于 `0.05 m/s`；
- H-STERN：vertical-miss 窗口 stern binding 中位数 `0.5208-0.6204`，角度限幅
  很常见；但 shortfall-binding Spearman 为 `-0.4562--0.1211`，高 shortfall 组
  binding 反而低 `0.2270-0.4216`；重建的速率绑定率为 `0`；
- H-EXCESS：入口 excess 与 required closure 的 Spearman 为 `0.9746-0.9891`，
  但 plane-time 相关性为 `-0.4167--0.3370`，未通过预注册 `>= -0.30` 判据。

总体 `NO_SUPPORTED_MECHANISM`。这不是数据无效，而是现有 authority-magnitude
解释不够。

## 2. 已排除项

1. **不是 checkpoint/seed 问题**：40/40 seeds，SHA256 固定，held-out 未开。
2. **不是配置漂移**：四组配置均为 stern `20°`、fin rate `150°/s`、pitch gains
   `2/1`。
3. **不是简单的速度包络**：在这些 traces 的水平速度范围内，速度与 authority
   没有正相关。
4. **不是“更多 stern 饱和 = 更大 shortfall”**：限幅频繁但与 shortfall 反向。
5. **不是单一入口 excess 机制**：excess 与 required closure 极强相关，但时间项
   仍跨过预注册边界；不能改写成 SUPPORTED。
6. **不是 Goal31 lookahead 重跑问题**：candidate/baseline 入口状态在 Goal32 已
   几乎一致；本 Goal 不改终端参考。

## 3. 新的可证伪假设

### H-GLOBAL：终端垂直误差是全程垂直跟踪偏差的延续

机制：垂直误差不是在 20 m 终端段才形成，而是整个 episode 的深度/垂直跟踪偏差
一直存在；终端段只是在最后放大为 plane miss。

若成立应看到：

- 在距 goal 40 m、30 m、20 m 的窗口中，true-state 垂直误差已经很大；
- 20 m 入口误差与 40 m 窗口的垂直误差中位数比值较高；
- 终端段内 sign-matched closure 很小，误差曲线近似平移而不是在 20 m 内生成。

最便宜证伪：只读现有 traces，比较距 goal `40/30/20 m` 三个窗口的垂直误差、
垂直速度和 closed-loop closure；不训练、不新增 rollout。

### H-PHASE：垂直速度的符号/相位错误比幅值不足更关键

机制：控制器虽然在饱和工作区，但实际 vertical velocity 经常与 reduce-error
方向不一致，或在误差反向时恢复太慢；因此 authority 的 p95 幅值不是正确诊断量。

若成立应看到：

- sign-matched closing fraction 在 terminal window 中长期低于 `0.5`；
- 垂直误差符号变化后，actual pitch/vertical velocity 的换向延迟超过 1-2 s；
- 高 shortfall seeds 的低 closing fraction 比低 shortfall seeds 更明显。

最便宜证伪：用现有 trace 计算 sign-matched closing fraction、符号翻转滞后期和
command/actual pitch 的换向时间；不训练。

### H-SLOPE：参考/路径垂直斜率造成持续偏差

机制：路径或 goal tangent 的垂直分量使期望深度持续变化，当前 ALOS-PD 组合的
垂直跟踪带宽不足；终端 miss 只是沿路径累积后的结果。

若成立应看到：

- `goal_tangent_ned.z`、路径垂直斜率或参考 pitch 与垂直误差同向相关；
- 斜率较大的 seeds 在 40 m 窗口已有更大垂直误差；
- 终端段内的 authority shortfall 与斜率相关，而不仅是入口 excess。

最便宜证伪：从现有 trace 的 `goal_tangent_ned`、`command.pitch`、`eta` 和
`nu` 计算斜率/误差/滞后关系；不新增仿真。

## 4. 排查顺序

按 `数据 -> 标签 -> 评测 -> baseline -> 架构 -> 文献`：

1. 复核 Goal31 trace 的 terminal/vertical 标签和 Goal32 entry 定义；
2. 用 40/30/20 m 三个窗口检验 H-GLOBAL；
3. 用 sign-matched closing fraction 和换向滞后检验 H-PHASE；
4. 用 goal tangent/vertical slope 检验 H-SLOPE；
5. 只有单一机制通过后才预注册控制器候选；
6. 不重跑 Goal27/Goal31，不扫 reward/seed/window。

## 5. 当前裁决

`Goal33 decision: NO_SUPPORTED_MECHANISM`。停止 authority-magnitude 线，下一步
必须进入全程垂直/相位/斜率诊断；不得因为 stern 限幅常见就直接改控制器。
