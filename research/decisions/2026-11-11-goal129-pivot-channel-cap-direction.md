# Decision：Goal129 后转 channel-aware cap + direction gate，不换算法

**日期**：2026-11-11<br>
**Goal**：Goal129<br>
**Decision**：`pivot`<br>
**上游**：Goal128 `ACTION_PARAMETERIZATION / GEOMETRY_LIMITED`

## 触发

Goal129 的 64 条 eval-only trace 已完成。四个新参数化全部
`ACTION_PARAM_NO_EFFECT`：

```text
G1.0   strict 18.625 / default 27.625
C1.0   strict 22.750 / default 32.875
GC1.0  strict 23.125 / default 32.750
G0.125 strict 23.000 / default 32.750
zero   strict 24.000 / default 33.000
```

parity=True；动作 maxdiff=0、learned residual maxdiff=2.2e-16；失败仍以
`GOAL_PLANE_MISS` 为主。

## 判读

- binary near-goal gate 不是正确粒度：它把 3–10 m 的 approach 修正一起
  关掉，G1.0 甚至比 M1.0 更差，并把 86201 的增益从 30 打回 28。
- global BC-relative cap 是有效方向，但 α=0.25 同时限制 rudder、stern
  plane 和 rpm；rpm 通道的 learner/BC 为 5–8×，其他通道只有 1–4×，
  全局 cap 很可能在压 rpm 的同时过度限制了对 capture 有用的 fin 修正。
- 这不是网络结构回归：所有 arms 共用同一 checkpoint；Goal108 audit 与
  Goal126 配置的 `[z_t64, zero-DADE64, command3]`、`256×128` actor、
  `mu` 零初始化、无 NoisyNet 均未改变。

## 决策

`pivot` 到新的预注册 eval-only Goal130：

```text
channel-aware norm cap（rpm-only / fin-only / per-channel）
+ cap 剂量 α 扫描
+ 方向一致性 gate（cos(L,B) 阈值）
```

这不是换算法，也不训练；仍是 Goal126 的 8 个 SAC checkpoint，仍是 dev
`20–59`、strict/default、success frame both、held-out 关闭。

## 停止规则

- 若任一 arm 达到 `ACTION_PARAM_GAIN`：另开训练型 Goal，只训练该参数化；
- 若任一 arm 达到 `ACTION_PARAM_SAFE_PARITY`：允许 1M screen，不扩 formal；
- 若全部仍 `NO_EFFECT`：按 Goal129 预注册的默认分支转 terminal capture /
  approach geometry，不继续扫同一个 cap 家族，不换 PPO/SAC 主算法；
- 只有出现明确“残差方向对、但探索/样本效率不足”的证据，才另开 SAC 同协议
  对照；REDQ/DroQ 仍需单独预注册。

## 不做什么

- 不改 reward、terminal、success、goal radius；
- 不打开 held-out `100–149`；
- 不换 PPO/SAC 主算法，不重开 DADE；
- 不因失败改 seed、扩预算或移动 Gate 阈值；
- 不自动 commit/push。
