# Goal37 decision：停止局部 path-frame 诊断线

日期：2026-11-09<br>
Goal：Goal37<br>
诊断裁决：`INVALID_DIAGNOSTIC`<br>
Ledger decision：`stop`

## 依据

Goal37 只读检查 Goal31 四个 40-seed formal blocks。配置来源修正为实际的
`pc_goal27_c1_command_residual_{default,strict125}.json`，`path_bank_size=32`，
checkpoint SHA256 与 dev seeds 均符合预注册要求，held-out 未打开。

四个 block 均出现 `39/40` seeds 的 path provenance 失配：

- 第一行 `eta` 到 `path_bank[seed % 32].start_ned_m` 仍在 start jitter 内：
  `40/40`；
- trace goal 与 seed 对应路径 goal 一致：`1/40`；
- trace goal tangent 与 seed 对应路径一致：`1/40`；
- route tangent 与 seed 对应路径投影一致：`1/40`；
- trace goal 与 `path_bank[0]` goal 一致：`40/40`。

唯一一致的是 seed `32`（`32 % 32 == 0`）。因此同一 episode 的起状态来自
seed 对应路径，而 guidance/projection/logged goal 使用 path 0。这违反 Goal37
预注册的 path/goal/tangent/start 一致性条件，整体必须裁决
`INVALID_DIAGNOSTIC`。

## 不解释机制

H-FRAME、H-LOCAL-CARRYOVER、H-LOCAL-INITIAL-WEAK 均为
`NOT ADJUDICATED`。分析器 JSON 中的 H 字段只是对混合路径数据套用冻结公式后的
原始值，不能写成机制支持或否定，也不能据此设计控制器。

## 停止事项

- 停止 goal-frame 初始偏置与 local-frame carryover 诊断线；
- 不修改 pitch gain、reward、terminal window、lookahead、authority 或 seed；
- 不训练、不新增 rollout、不打开 held-out `100-149`；
- 不预注册控制器候选；
- 不把 `INVALID_DIAGNOSTIC` 写成控制器正式 FAIL 或 PASS。

## 下一步

若未来重开，必须先预注册一次 provenance 一致的 Goal31-style 评估，保证 start
state、guidance path、goal、goal tangent、route tangent 和 trace metadata 来自
同一个 `path_index`。该修复需要新 rollout，Goal37 不授权自动启动。
