# Blocker：pre-crossing authority 足够，tracking lag 才是主机制

日期：2026-11-10<br>
关联 Goal：Goal68<br>
状态：`PRECROSSING_TRACKING_LAG`<br>
上游：Goal67 `TERMINAL_AUTHORITY_LIMITED`、Goal66 `TERMINAL_GEOMETRY_UNRESOLVED`

## 1. 现象

Goal68 复用 Goal67 rich trace，把分析窗口从 final 51 步前移到 plane crossing 前
2 s（1 s/3 s 只做方向一致性），没有新增 rollout。结果：

- default：lost `9`，retained `61`；lost deficit median `-0.467 m/s`，retained
  `-0.255 m/s`；deficit fraction `0.000`；
- strict125：lost `7`，retained `23`；lost deficit median `-0.205 m/s`，retained
  `-0.224 m/s`；deficit fraction `0.000`；
- lost command-goal cosine `0.988/0.991`，但 actual-goal cosine 只有
  `0.604/0.856`；
- lost actual ground speed `2.060/1.867 m/s`，不是低速卡死；
- default 的 true eta 在 crossing 前进入收紧球 `4/9`，但 p_nav 从未进入；
  strict125 只有 `2/7`；
- H68-A 证伪，H68-B 在 default/strict125 都支持；default 还支持 H68-C。

## 2. 关键解释

crossing 前 2 s 的 closure 已经高于 required，因此不能把失败归因于“接近段没有
速度/舵权限”。真正稳定的现象是：command 指向 goal，实际地速方向在最后 1–2 s
没有跟上，actual-goal cosine 从 3 s 的 `0.734/0.881` 降到 1 s 的 `0.348/0.764`。

default 还出现 p_nav/capture timing：4 个 lost 的 true eta 已进入 `r_tight`，但
p_nav 在 crossing 前从未进入。这不能解释 strict125 的多数失败，因此只作为叠加机制。

## 3. 已排除项

| 解释 | 状态 | 理由 |
|---|---|---|
| pre-crossing authority 不足 | 证伪 | deficit 中位数为负，deficit fraction 为 0 |
| command 方向错 | 排除 | command-goal cosine `0.988/0.991` |
| 低速卡死 | 排除 | actual speed `2.060/1.867 m/s` |
| NoisyNet/residual 主导 | 未支持 | zero-residual 仍丢 default 4、strict125 1；评估不采样噪声 |
| final-window 时间原点 | 已处理 | 窗口截止 crossing，不再使用 post-pass 行 |
| 优化/探索失败 | 未证明 | frozen policy 的 command 已对准，失败发生在 tracking |
| 样本效率失败 | 未证明 | 本 Goal 未训练；失败在 frozen checkpoint 的闭环状态 |

## 4. 下一组可证伪 hypothesis

### H69-A：controller command-to-velocity tracking lag

若 pre-crossing 的 actual yaw/pitch/surge 对 command 的跟踪误差在最后 1–2 s 持续
增大，且 actual-goal cosine 的下降先于 command-goal cosine 的下降，则 tracking lag
成立。

证伪：若 command 与 actual 的跟踪误差在 crossing 前保持小且稳定，则 lag 不是主因。

### H69-B：最后 1 s 的 heading-rate/authority 不足

若 crossing 前 1 s 的 required heading rate 持续高于 demonstrated yaw-rate p95，
且 lost 与 retained 的差距大于 `0.05 rad/s`，则最后 1 s 的转向 authority 是叠加
机制。

证伪：若 1 s 窗口的 rate deficit 与 retained 无差异，或 actual 已跟上 command，
则 H69-B 不成立。

### H69-C：p_nav/capture timing 独立失败

若 default 的 4 个 true-eta-tight 但 p_nav-not-tight episode 在去掉 tracking 因素后
仍无法被 p_nav 捕获，则 capture timing 是独立分支；若 p_nav 只在 tracking 发散后
才偏离，则 C 不支持。

证伪：若 p_nav 在 true eta 进入收紧球时已经同步进入，或 tracking 改善后 C 消失，
则不能把 p_nav 单独列为主因。

## 5. 最便宜 falsification

复用 Goal67 rich trace，不训练：

1. 计算 pre-crossing 最后 2 s 的 command/actual heading、pitch、surge 跟踪误差；
2. 计算 actual-goal cosine 与 command-goal cosine 的时间导数；
3. 对最后 1 s 计算 required heading rate vs actual yaw rate；
4. 对 default 的 p_nav-only episode 检查 p_nav 是否在 true eta 进球后仍滞后；
5. 只在现有 trace 上判 H69-A/B/C，不新增 rollout。

## 6. 决策

`decision = pivot`：停止 pre-crossing authority 主假设，进入 controller tracking/
recovery 审计。下一步应新预注册 Goal69，保持 frozen checkpoint、dev `20–59`、
CPU/float64、`endpoint_stop` 和 held-out 关闭。

算法切换门仍未触发：

- 没有 optimization/exploration 证据，不启动 SAC；
- 没有 sample-efficiency 证据，不启动 REDQ/DroQ；
- 当前闭环在运动，且 command 已对准 goal，不先改 action parameterization/reward；
- 只有 Goal69 的证据明确指向上述条件，才另开对应预注册 goal。

## 7. 证据

- `doc/goals/goal68_result_20261110.md`
- `artifacts/lowlevel/goal68_precrossing_terminal_20261110/goal68.json`
- `artifacts/lowlevel/goal68_precrossing_terminal_20261110/goal68.md`
- 服务器：
  `/data/yhx/underwater-robotics/src/plan_p1/goal60_action_reward_terminal_readonly_audit_20261110/artifacts/goal68_precrossing_analysis/`
