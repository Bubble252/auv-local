# Blocker：50 步 plane-miss dwell 不足以恢复 terminal capture

日期：2026-11-10<br>
关联 Goal：Goal65<br>
状态：`DWELL_NOT_SUFFICIENT`<br>
上游：Goal64 `GEOMETRY_SUPPORTED`

## 1. 现象

Goal65 在完全相同的 frozen checkpoint、dev `20–59`、CPU/float64、
`endpoint_stop` 和 `r0 - 0.25 m` candidate GOAL 下，只把 plane-miss 终止延迟
50 步：

- default policy true retention `0.907`，但 precision `0.777`，zero-residual
  retention `0.750`；
- strict125 policy true retention `0.739`，低于预注册的 `0.85`；
- strict125 的 4 个 checkpoint 全部低于 `0.80` retention；
- 两个 scenario 的 false success 相对 Goal63 都没有超过 `+2`，timeout/safety
  增量都是 0；
- strict125 的 extra steps 中位数恰好是 50，说明每个丢失 episode 都等满了窗口，
  但仍以 plane miss 终止。

## 2. 关键证据

逐 seed 的 dwell=0 复现与 Goal63 完全一致，关键字段 0 mismatch。dwell=50 后，
baseline true-success 丢失组在最后 51 步内：

| scenario | lost true | entered tight radius | minimum `p_nav` median | window-end `p_nav` median |
|---|---:|---:|---:|---:|
| default | 5 | 0/5 | `1.974 m` | `2.922 m` |
| strict125 | 6 | 0/6 | `1.370 m` | `2.458 m` |

strict125 的 candidate GOAL 半径是 `1.00 m`；default 是 `1.75 m`。丢失组没有
任何 episode 在 dwell 内进入收紧半径，窗口末端距离反而增加。因此 dwell 不是
短了一点点，而是把同一个未收敛终点整体推迟。

## 3. 已排除项

| 解释 | 状态 | 理由 |
|---|---|---|
| plane-miss 提前一帧截断 | 已排除 | dwell=50 后 strict125 仍 0/6 进入 `1.00 m` |
| dwell 引入 false success | 已排除 | default `23 vs 22`、strict125 `14 vs 14`，都在 `+2` 内 |
| timeout/safety 交换 | 已排除 | 两个 scenario 增量都是 0 |
| residual 幅度/NoisyNet 主导 | 未支持 | dwell 不改变 action/noise；forced zero-residual 也未通过 default Gate |
| 样本效率不足 | 未支持 | 未训练；失败是 terminal 接近未收敛，不是学习曲线 |
| 数据/实现损坏 | 已排除 | dwell=0 逐 seed 完全一致；SHA、finite、seed、radius 校验通过 |

## 4. 新 hypothesis

### H66-A：terminal `p_nav/eta` 偏差或校准误差解释了未进入收紧球

如果 crossing 后控制器已经在朝目标收敛，但 `p_nav` 相对 true `eta` 存在稳定的
terminal bias，则丢失组应表现为 `p_nav` 与 `eta` 的距离方向不一致、偏差随
terminal phase 增大，或同一 seed 的误差在多 checkpoint 间重复。

可证伪预测：若 `p_nav/eta` 偏差在所有丢失 seed 上都小于几何余量，且不能解释
`0/6` 未进入 `1.00 m`，则 H66-A 不成立，应转向 action/terminal command。

### H66-B：terminal action parameterization / command 仍在把载体推离捕获球

如果 crossing 后的 heading、pitch 或 speed command 仍偏向上方/侧向或保持过高
速度，则即使终止规则等待 50 步，载体也会继续离开目标。可证伪预测：丢失组的
terminal command、速度、横向/垂直偏移应系统性不同于 retained 组；若两组 command
分布相同，则 H66-B 不成立。

### H66-C：reward/terminal capture 标签与真实目标存在错配

如果 reward 对 plane crossing 附近的行为给出与 true capture 相反的信号，或者
terminal capture 标签使用了不同于控制目标的几何量，则继续训练只会强化错误行为。
可证伪预测：terminal 前的 reward breakdown 在 lost/retained 组应可区分；若不可
区分，则 reward 不是当前主因。

## 5. 排查顺序

按 `数据 → 标签 → 评测 → baseline → 架构 → 文献`：

1. 数据：Goal65 已通过 dwell=0 逐 seed 等价与 SHA/finite 校验；
2. 标签：检查 `success`、`terminal_eta_capture`、`p_nav/eta` 的定义是否一致；
3. 评测：只在 dev `20–59` 做 terminal-window 对齐，不打开 held-out；
4. baseline：对 lost/retained 比较 `p_nav`、true `eta`、command、reward；
5. 架构：检查 terminal action parameterization、速度/航向限幅与 GOAL 条件；
6. 文献：只用于约束 estimator/calibration 与 terminal capture 设计，不替代本地
   falsification。

## 6. 决策

`decision = pivot`：停止 plane-miss 短 dwell 轴，不扫描更多 dwell，不修改正式
termination 规则。下一步预注册 Goal66，做只读的 terminal estimator/calibration
与 action/command 审计。

算法切换提醒：只有新证据明确指向 optimization/exploration，才新开预注册 Goal
用 SAC 做同协议对照；只有明确指向 sample efficiency，才考虑 REDQ/DroQ。当前没有
这两类证据，不启动算法替换。

## 7. 证据

- `doc/goals/goal65_result_20261110.md`
- `research/paper_reading/2026-11-10-goal65-dwell-terminal-capture.md`
- 服务器：
  `artifacts/goal65_plane_miss_dwell/analysis/goal65.json`
- 服务器 dwell=0 复现：
  `artifacts/goal65_repro_dwell0/default_policy_s240_dwell0_m025/episodes.json`
