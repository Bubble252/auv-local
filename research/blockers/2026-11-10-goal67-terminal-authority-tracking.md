# Blocker：lost 组在最近点后外离，authority 与 tracking 未分离

日期：2026-11-10<br>
关联 Goal：Goal67<br>
状态：`TERMINAL_AUTHORITY_LIMITED_WITH_TRACKING_CAVEAT`<br>
上游：Goal65 `DWELL_NOT_SUFFICIENT`、Goal66 `TERMINAL_GEOMETRY_UNRESOLVED`

## 1. 现象

Goal67 用 rich trace 重跑了 Goal65 的 10 个 frozen-checkpoint candidate，40-seed
逐字段等价性全部通过。主要结果：

- default：lost true `9`，retained `61`；eta-outside lost `7`；p_nav-only `2/9`；
- strict125：lost true `7`，retained `23`；eta-outside lost `4`；p_nav-only `3/7`；
- lost command-goal cosine 中位数 `0.987/0.987`，command 已高度朝向 goal；
- lost actual-goal cosine 中位数 `-0.366/-0.419`，实际地速在最近点朝外；
- lost actual ground speed 中位数 `2.119/1.959 m/s`，不是低速卡死；
- eta-outside deficit fraction 为 `1.000/1.000`，deficit 中位数
  `0.847/1.106 m/s`；
- forced zero-residual 仍丢 default `4/40`、strict125 `1/40` 个 true episode。

预注册分类为 default/strict125 均 `TERMINAL_AUTHORITY_LIMITED`，但 A 和 B 同时
支持，且逐 run 大多给出 B。这个结果不能简化成“纯执行器饱和”。

## 2. 关键解释限制

16 个 lost true 中，15 个的 true-eta 最近点位于 final 51 窗口第一帧，之后持续
外离；只有 strict125 s1008/seed23 的最近点在窗口内第 5 帧。因此 H67-A 测到的是
“最近点之后 1 s 没有再闭合”，而不是完整接近过程的 authority 上限。

当前可以确认的是：

1. command 方向不是主因；
2. 载体不是静止，最近点仍有约 `2 m/s` 地速；
3. 失败形态是掠过捕获球后外离，加上 p_nav-only 薄壳；
4. residual/NoisyNet 不是充分原因，zero-residual 仍复现失败；
5. 仍需把时间窗前移到 plane crossing 之前，分离 pre-crossing authority 与
   post-pass recovery。

## 3. 已排除项

| 解释 | 状态 | 理由 |
|---|---|---|
| NoisyNet 噪声主导 | 未支持 | `evaluate=True` 不采样噪声；command 已对准 goal；zero-residual 仍失败 |
| command 方向错 | 排除 | lost command-goal cosine `0.987/0.987` |
| 载体低速卡死 | 排除 | lost actual speed `2.119/1.959 m/s` |
| dwell 太短 | 已排除 | Goal65 dwell=50 仍外离 |
| p_nav 估计器单独解释全部失败 | 不充分 | p_nav-only 仅 `2/9`、`3/7` |
| 优化/探索失败 | 未证明 | frozen policy 的 command 与闭环状态已分离，尚未看到训练不收敛证据 |
| 样本效率失败 | 未证明 | 本 Goal 未训练；失败发生在 frozen checkpoint 的终局轨迹 |

## 4. 下一组可证伪 hypothesis

### H68-A：pre-crossing authority 不足

在 plane crossing 前 1–3 s，若 required closure / required yaw-pitch rate 持续高于
实际展示的 closing / turn rate，则 authority 限制成立。

证伪：若 pre-crossing 的 required rate 在实际展示包络内，且 actual 已跟上 command，
则 H68-A 不成立，应转向 post-pass recovery 或 capture timing。

### H68-B：pre-crossing command 可达但 controller 恢复滞后

若 pre-crossing command 已朝向 goal 且在可行包络内，但 actual yaw/pitch/velocity
在最后 1–3 s 没有按额定 rate 跟随，则 tracking/recovery 滞后成立。

证伪：若 actual rate 与 command 一致，且差异主要由几何近失解释，则 H68-B 不成立。

### H68-C：final-window 时间原点导致 post-pass 被计为 lost

若把评估窗口前移到最近点/plane crossing 之前后，lost 组能进入 `r_tight`，且
terminal capture 分类改变，则当前 final 51 窗口的 post-pass 采样是主要测量伪影。

证伪：若前移窗口后 lost 仍无法进入 `r_tight`，则不能归因于窗口时间原点。

## 5. 最便宜的 falsification experiment

复用 Goal67 已生成的 rich trace，不新增 rollout、不训练：

1. 对每个 lost seed 找到 plane crossing 和 true-eta 最近点；
2. 从 crossing 前 1 s / 2 s / 3 s 起，重复计算 required vs demonstrated
   closure、command/actual cosine、yaw/pitch rate 和 material saturation；
3. 检查窗口前移后 true eta / p_nav 是否进入 `r_tight`；
4. 只按预注册阈值判 H68-A/B/C，不回改 H67 结果。

## 6. 决策

`decision = pivot`：从 final-window authority 审计转到 pre-crossing
authority/recovery/timing 审计。下一步应新预注册 Goal68，保持 frozen checkpoint、
dev `20–59`、CPU/float64、`endpoint_stop` 和 held-out 关闭。

算法切换门仍未触发：

- 没有 optimization/exploration 证据，不启动 SAC；
- 没有 sample-efficiency 证据，不启动 REDQ/DroQ；
- 当前闭环在运动而非卡死，不先改动作参数化、reward 或 terminal capture 定义；
- 只有 Goal68 的证据明确指向上述条件，才另开对应预注册 goal。

## 7. 证据

- `doc/goals/goal67_result_20261110.md`
- `artifacts/lowlevel/goal67_terminal_reachability_20261110/analysis/goal67.json`
- `artifacts/lowlevel/goal67_terminal_reachability_20261110/analysis/goal67.md`
- 服务器：
  `/data/yhx/underwater-robotics/src/plan_p1/goal60_action_reward_terminal_readonly_audit_20261110/artifacts/goal67_terminal_reachability/`
