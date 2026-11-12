# Goal37 blocker：Goal31 trace 的路径 provenance 不一致

**日期**：2026-11-09<br>
**Goal**：Goal37<br>
**触发**：Goal37 只读重投影在读取机制聚合前发现，Goal31 formal trace 的
`goal_ned_m`、`goal_tangent_ned`、`route_tangent_ned` 与起状态对应的
`path_bank[seed % 32]` 不一致。按 Goal37 冻结规则，整体裁决为
`INVALID_DIAGNOSTIC`。

## 1. 现象

Goal31 formal 的实际 environment configs 是：

```text
default:
  configs/lowlevel/pc_goal27_c1_command_residual_default_20261109.json
strict125:
  configs/lowlevel/pc_goal27_c1_command_residual_strict125_20261109.json
```

两套 config 的 `path_bank_size=32`、`path_bank_seed=41`，因此按 Goal37 预注册规则
重建：

```text
path_bank = generate_sinusoidal_3d_path_bank(config.reference)
expected_path = path_bank[seed % 32]
```

四个 block 的结果完全一致：

| check | 结果 |
|---|---:|
| 40/40 seeds 的第一行 `eta` 到 expected path start 的 jitter 检查 | PASS（40/40） |
| trace goal 与 expected path goal 一致 | 1/40 |
| trace goal tangent 与 expected path goal tangent 一致 | 1/40 |
| 全程 route tangent 与 expected path projection tangent 一致 | 1/40 |
| 检查行 route tangent 与 `path_bank[0]` projection tangent 一致 | 40/40 |
| trace goal 与 `path_bank[0]` goal 一致 | 40/40 |
| trace goal tangent 与 `path_bank[0]` goal tangent 一致 | 40/40 |

唯一一致的 seed 是 `32`，因为 `32 % 32 == 0`。其余 `39/40` seeds 都出现：

- 第一行起状态落在 seed 对应路径的 start jitter 内；
- 但 trace goal、goal tangent、route tangent 和 path progress 都沿 path 0；
- expected path goal 与 trace goal 的最大距离为 `6.8531 m`；
- expected path goal tangent 与 trace goal tangent 的最大差为 `0.8927`；
- start 到 expected path start 的最大距离仅 `0.6821 m`，仍在 `0.8661 m` 上限内。

这不是单纯的 goal 字段日志错误：`route_tangent_ned` 在四个检查行都与 path 0
的投影切向一致，而与 seed 对应路径不一致。因此控制/投影所用 path 与起状态采样
所用 path 在同一 episode 内不一致。

## 2. 证据

- `doc/goals/goal37_local_path_frame_20261109.md`
- `artifacts/lowlevel/goal37_local_path_frame_20261109/local_path_frame_diagnostic.json`
- `artifacts/lowlevel/goal37_local_path_frame_20261109/local_path_frame_diagnostic.md`
- `artifacts/lowlevel/goal37_local_path_frame_20261109/local_path_frame_per_seed.csv`
- `artifacts/lowlevel/goal31_terminal_path_preserving_20261109/formal/{default,strict125}/{baseline,candidate}/{summary.json,trace.npz}`

四个 block 均通过：

- seeds `20-59`，每 block 40 seeds；
- checkpoint SHA256
  `0851496b15ffebbc0a19bafa5bf40c0493c0d9a3f485267b75fde5f811dde5d9`；
- path bank size `32`；
- baseline/candidate 初始几何最大差 `0.0`；
- held-out `100-149` 未打开。

## 3. 可证伪 hypothesis

### H-PATH-MIX：起状态与控制器使用了不同 path

若成立应看到：起状态匹配 `path_bank[seed % 32]`，但 goal、goal tangent 和 route
tangent 匹配 `path_bank[0]`。

当前证据：**SUPPORTED by provenance audit**。四个 block 都是 40/40 start-jitter
匹配 expected path，同时 40/40 goal 匹配 path 0；39/40 seed 的 goal/tangent 与
expected path 不一致。

### H-LOG-ONLY：只有 goal 日志写错，控制器实际沿 expected path

若成立应看到：route tangent 和 path progress 与 expected path 一致，只有
`goal_ned_m` / `goal_tangent_ned` 字段是 path 0。

当前证据：**FALSIFIED**。route tangent 与 expected path 的一致数只有 1/40，
与 path 0 的投影切向一致；path progress 也沿 path 0 投影。

### H-CONFIG-BANK：重建 config 或 path bank 版本错误

若成立应看到：起状态、goal、tangent 和 route tangent 与同一错误 path bank 或
错误 config 一致，或 start 也不匹配。

当前证据：**FALSIFIED**。两个 environment config 的 path bank 参数一致，
起状态 40/40 匹配 seed 对应路径；错误解释无法同时解释“start 匹配 expected path、
route/goal 匹配 path 0”。

## 4. 最便宜的证伪实验

已完成，且是纯只读：

```bash
cd /home/bubble/水下机器人/settled_局部规划器_反应式
.venv/bin/python scripts/analyze_pc_goal37_local_path_frame.py \
  --config-default configs/lowlevel/pc_goal27_c1_command_residual_default_20261109.json \
  --config-strict125 configs/lowlevel/pc_goal27_c1_command_residual_strict125_20261109.json \
  --formal-root artifacts/lowlevel/goal31_terminal_path_preserving_20261109/formal \
  --output artifacts/lowlevel/goal37_local_path_frame_20261109
```

该命令不加载 policy、不 step environment、不训练、不新增 rollout，也不读取 held-out。

## 5. 裁决前状态

Goal37 的 H-FRAME、H-LOCAL-CARRYOVER、H-LOCAL-INITIAL-WEAK 不进行机制裁决。
JSON 中的 H 字段只是按冻结公式算出的诊断值，不能写成支持或否定机制，因为
`goal/tangent/route/start` 的 provenance 已违反预注册一致性要求。

当前 decision：`stop`。不提出控制器候选，不修 inner-loop，不重扫
authority/pitch-gain/reward/window/seed/lookahead，也不打开 held-out。

## 6. 后续边界

若要修复，必须新预注册一次一致的 Goal31-style 评估，明确保证同一 episode 的
start state、guidance path、goal、goal tangent、route tangent 和 trace metadata
来自同一个 `path_index`。这将需要新的 rollout，**不由 Goal37 授权**；用户当前
未要求重跑，因此保持停止。

文献背景见：

- `research/paper_reading/2026-11-09-goal37-local-path-frame.md`
