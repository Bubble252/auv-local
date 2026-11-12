# Goal36 blocker：初始垂直偏置可能是 goal-tangent 投影伪差

**日期**：2026-11-09<br>
**Goal**：Goal36 / Goal37 入口<br>
**触发**：Goal36 `NO_SUPPORTED_MECHANISM`，但参考投影中位数 `6.78-6.84 m`、
份额 `75.7%-85.0%`，且它与初始/terminal 绝对误差负相关。现有“初始偏置排序”
假设无法同时解释大份额和反向排序，属于 `hypothesis_unexplained`。

## 1. 现象

Goal35/36 的“初始垂直误差”使用 Goal32 的 **goal tangent** 法向分解：

```text
reference_vertical = -longitudinal * goal_tangent_z
initial_vertical   = world_vertical + reference_vertical
```

Goal36 发现：

- `goal_tangent` 的 pitch 中位数约 `0.2198 rad`，`longitudinal` 约 `-31 m`；
- 因此仅终点切向投影就产生约 `6.8 m` 的初始垂直分量；
- 该分量变化很窄，而 world vertical 变化更宽；两者 signed correlation 为
  `-0.9654--0.9678`，常互相抵消；
- 所以参考份额很大，但绝对参考量与初始/terminal 误差负相关，无法通过
  Goal36 的排序判据。

关键问题：控制器在 episode 起点使用的是**局部 route/path tangent**投影，而不是
终点 `goal_tangent`。如果局部路径坐标系下初始垂直误差远小于 `8 m`，那么
Goal35 的“初始偏置主导”可能是离线坐标框架造成的伪差。

## 2. 已排除项

1. 不是 seed/checkpoint 问题：Goal31 formal 为 40/40 seeds，SHA256 固定，
   held-out 未打开。
2. 不是 baseline/candidate 初始几何不一致：Goal36 的初始几何最大差为 `0.0`。
3. 不是起始 pitch/rate/vertical velocity：Goal36 控制参考投影后 partial
   Spearman 为 `-0.2465--0.0749`，pitch rate 中位数仅 `0.0026 rad/s`。
4. 不是 20 m 交接：Goal35 的 handoff worsening 中位数为 `0`，且不排序 terminal
   miss。
5. 不是简单的“世界垂直差”：H-STATE 的 world-initial rho 仅
   `0.1008-0.1831`。

## 3. 可证伪假设

### H-FRAME：初始偏置主要是终点切向投影伪差

机制：goal tangent 是路径终点切向；在弯曲路径的起点，用终点切向定义“垂直”
会把起点到终点的纵向距离投影到法向，产生一个大的、近常数的虚拟垂直误差。

若成立应看到：

- goal-frame 初始绝对垂直误差中位数仍约 `8 m`；
- local-path-frame 初始绝对垂直误差中位数 `<=1 m`，且绝大多数 seed 在
  capture radius 内；
- 两种框架的初始绝对误差中位数差 `>=5 m`。

若失败应看到：局部路径坐标系下初始误差仍很大，或大多数 seed 仍超出半径。

最便宜证伪：按 `seed % path_bank_size` 重建 Goal31 的 32 条 path，投影 `eta`
第一行到局部 path，比较 goal-frame 与 local-frame 垂直误差；纯只读。

### H-LOCAL-CARRYOVER：局部路径坐标系下 40/20 m 误差仍延续到 terminal

机制：如果全局垂直误差不是坐标框架问题，那么即使改用局部 path 投影，40 m 和
20 m 的局部垂直误差仍应排序 terminal 局部垂直误差。

若成立应看到：

- 局部 `40 m -> terminal` 和 `20 m -> terminal` 的绝对垂直误差 Spearman
  `>=0.50`；
- 40 m、20 m 的误差中位数至少是 terminal 的 `0.50`。

若失败应看到：局部路径坐标系下早期误差很小、不能排序 terminal，说明 terminal
误差主要在 20 m 之后形成。

最便宜证伪：同一只读重投影，在 true distance-to-goal `40/20 m` crossing 和
terminal 行计算局部垂直误差；不新增 rollout。

### H-LOCAL-INITIAL-WEAK：局部初始误差不排序 terminal

机制：如果 H-FRAME 成立，局部初始误差应接近 start jitter 量级，不能解释
terminal 局部垂直 miss。

若成立应看到：

- local initial 与 local terminal 的 Spearman `<=0.30`；
- local initial / local terminal 中位数比 `<=0.25`；
- local initial 超出 capture radius 的比例 `<=0.20`。

若失败应看到：局部初始误差仍然大或仍然排序 terminal，则初始状态确实是问题。

最便宜证伪：与 H-FRAME 共用同一重投影数据。

## 4. 排查顺序

按 `数据 -> 标签 -> 评测 -> baseline -> 架构 -> 文献`：

1. 核对 Goal31 trace 的 goal、goal tangent、route tangent 与 path bank 是否一致；
2. 明确 goal-frame 是终端平面坐标，local-frame 是路径跟踪坐标，不能混用标签；
3. 用 local-frame 重算 initial / 40 m / 20 m / terminal 垂直误差；
4. 比较 Goal35/36 的结论是否来自框架伪差；
5. 只有 local-frame carryover 成立后，才考虑局部垂直控制候选；
6. 文献只用于确认路径跟踪误差应在局部切线/法向坐标中定义，不代替本地数据。

## 5. 文献入口

已有 `research/paper_reading/2026-11-09-goal31-terminal-path-vertical-gap.md`
指出 LOS/ALOS 的 cross-track/vertical-track 指标属于路径局部坐标，而 terminal
capture 属于 goal-plane 指标。Goal37 的新 reading note 将补充
Frenet/Serret-Frenet 局部框架与终点切向投影的区别，并核验至少两篇来源。

## 6. Decision before Goal37

`decision: continue`。不继续 Goal36 的排序阈值，也不改控制器；先执行 Goal37
只读重投影。Goal37 若支持 H-FRAME，则把“初始偏置主导”改判为框架伪差，并重新
评估局部坐标系下误差在何时形成。不得用 Goal36 的负相关直接改控制器。
