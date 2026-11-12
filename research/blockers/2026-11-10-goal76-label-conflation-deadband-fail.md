# Blocker：标签混淆已确认，但 deadband 仍差 1 个 strict125 false 样本

日期：2026-11-10<br>
关联：Goal76 / Goal75 / Goal74 `MARGIN_UNUSABLE_BIAS_SUPPORTED`<br>
状态：`LABEL_CONFLATION_CONFIRMED_BUT_DEADBAND_FAIL`<br>
决策：`pivot`，转向 terminal plane crossing 与 capture/termination 时序

## 1. 现象

Goal76 把 protocol radius 与 tight radius 标签分开后，deadband `b0=0.25 m`
的 tight true retention 在 default 为 `100%`、strict125 为 `96%`，都过了
预注册 `>=95%`。但 strict125 的 tight false 降幅是 `16/33 = 48.48%`，
比预注册 `>=50%` 少 1 个样本；不能事后放宽阈值。

| scenario | base tight true | deadband tight true | base tight false | deadband tight false | false 降幅 |
|---|---:|---:|---:|---:|---:|
| default | 59 | 59 | 69 | 34 | 50.72% |
| strict125 | 25 | 24 | 33 | 17 | 48.48% |

protocol-only 标签混淆已确认：default `38/97`、strict125 `19/44` 的
protocol true 只在 protocol radius 内。因此 H76-B 成立，H76-A 只通过
default，未通过 strict125 的 false reduction Gate。

## 2. 剩余失败的几何证据

deadband 后剩余 tight false 的中位几何：

| scenario | n | b (m) | p_nav-eta (m) | signed offset (m) | lateral (goal frame, m) | vertical (goal frame, m) |
|---|---:|---:|---:|---:|---:|---:|
| default | 34 | 0.1319 | 0.3252 | -1.2187 | 0.3404 | -0.4242 |
| strict125 | 17 | 0.1456 | 0.2925 | -0.6194 | -0.0021 | 0.0225 |

`b` 已低于 deadband，signed offset 为负：`p_nav` 在 goal plane 前
`0.62–1.22 m` 就停止；proper goal-frame lateral 只有 `0.34 m`（default）
和约 `0 m`（strict125）。早先的 `1.2364/0.7618 m` 是世界坐标水平距离，
重复计入了 signed offset。

继续降低 deadband 只会误伤低 `b` true，不会解决 terminal plane shortfall。

## 3. NoisyNet / 残差排查

正式评估是 deterministic mean，`sample_noise=False`。按 checkpoint 分解
后，deadband 剩余 tight false 中：

| checkpoint | default 剩余 false | strict125 剩余 false | noise/det L2 中位 | residual/action 中位 |
|---|---:|---:|---:|---:|
| s240 | 11 | 3 | 0.0049 | 0.4140 |
| s752 | 6 | 3 | 0.0083 | 0.4163 |
| s880 | 5 | 4 | 0.00049 | 0.4126 |
| s1008 | 7 | 1 | 1.056 | 0.3477 |
| zero-residual | 5 | 6 | — | — |

forced zero-residual 的 `residual_l2=0` 仍有 `5/34`（default）和 `6/17`
（strict125）失败。`s240/s752/s880` 的低噪声 checkpoint 贡献了
default `22/34`、strict125 `10/17` 的剩余失败；只有 s1008 噪声较大，
但失败并未集中在那里。

因此：

- **排除** NoisyNet 采样主导；
- **排除** “residual 必须非零才能解释失败”；
- residual 的量级 `0.35 m` 是 deterministic applied residual，不是噪声；
- 当前最合理机制是 sphere `endpoint_stop` 让 true 与 false 都在 goal plane
  前终止，terminal plane crossing 与 capture/termination 时序不匹配。

## 4. 可证伪假设

### H76-A：tight 标签下 deadband 能同时保留 true 并消除 false

预测：两个 scenario 的 tight false 降幅都 `>=50%`，tight true retention
都 `>=95%`。

结果：**PARTIAL / FAIL**。true retention 过线，strict125 false 降幅
`48.48%`，差 1 个样本。

### H76-B：protocol 标签系统性高估 tight true

预测：`tight_true/protocol_true <=0.70` 且 protocol-only `>=10`。

结果：**SUPPORTED**。default `59/97`、strict125 `25/44`；protocol-only
分别为 `38/19`。

### H76-C：NoisyNet 或 residual 幅度主导剩余失败

预测：失败集中在高噪声 checkpoint，或去掉 residual 后失败显著减少。

结果：**REJECTED**。zero-residual 仍保留 `5/34` 和 `6/17` false；
低噪声 s240/s752/s880 贡献大部分剩余失败。

## 5. 决策

`decision = pivot`

停止沿 goal 方向继续调 deadband、reward、gamma 或 seed；不启动 SAC/REDQ/
DroQ，因为当前没有 optimization/exploration 或 sample-efficiency 证据。

后续 Goal77 已执行并判为 `GEOMETRY_GATE_NOT_SUPPORTED`：plane gate 的 false
降幅 `91.30%/100%`，但 tight true retention 只有 `6.78%/8.00%`，说明 true
也大多停在 goal plane 前。Goal77 固定比较了：

1. plane gate：`signed_offset >= 0`；
2. lateral margin：`abs(lateral_offset) <= 0.75 * r_tight`；
3. vertical margin：`abs(vertical_offset) <= 0.50 * r_tight`；
4. combined gate：plane + lateral + vertical。

离线 gate 无法区分 true/false，因此不再允许事后 gate。Goal78 随后证明
full-vector `p_nav-eta` deadband 可通过只读 Gate，但使用 true `eta`，仍是
oracle。下一步预注册 Goal79：observable full-vector correction predictor；
若失败，再进入 terminal capture/plane-crossing 闭环 arm。

## 6. 证据

- `doc/goals/goal76_terminal_success_label_semantics_20261110.md`
- `doc/goals/goal76_result_20261110.md`
- `scripts/analyze_pc_goal76_label_semantics.py`
- `tests/test_pc_goal76_label_semantics.py`
- `artifacts/lowlevel/goal76_label_semantics_20261110/analysis/goal76.json`
- `artifacts/lowlevel/goal76_label_semantics_20261110/analysis/goal76.md`
- `research/paper_reading/2026-11-10-goal73-terminal-capture-estimator-bias.md`
- 服务器原始 trace：
  `/data/yhx/underwater-robotics/src/plan_p1/goal60_action_reward_terminal_readonly_audit_20261110/artifacts/goal71_command_rate_limit/`
