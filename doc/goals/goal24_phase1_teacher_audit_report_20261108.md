# Goal24 Phase 1 报告：teacher-fixed 一致性审计

日期：2026-11-08  
状态：**PASS**  
范围：仅教师评估，不训练；未使用 held-out `100–149`

## 1. 目标

确认新 nominal teacher-fixed 配置与旧 R2B current control 在 dev `20–59`、同一
`endpoint_stop` 口径下一致，并判断 strict125 教师是否恢复到约 `0.45` 的可用水平。

## 2. 产物与哈希

- 输出根目录：`/dev/shm/pc_v1_teacher_audit_20261108/`
- 审计 manifest：`/dev/shm/pc_v1_teacher_audit_20261108/teacher_audit_manifest.json`
- checkpoint：
  `/dev/shm/pc_v0d_nominal_bc_rebase_20261107/bc_N8/pc_bc_actor.pt`
- checkpoint SHA256：
  `9dd416a121f471c5b1aadafb9d591e8259b9803a8bbdf6650ce14e6b49664e36`
- fixed default config SHA256：
  `882155ec33dba120a4a50c821573878de5549c12638b1772ad518b26648037ef`
- fixed strict125 config SHA256：
  `ca1f0055d21a399196457db3508a06566adf86dcc25bf2883f84b89b676a19e0`
- control default config SHA256：
  `8e4ac934a305f93ae5bc2737a9eec1cf9725fe17adb4e4a9f96f348610464b23`
- control strict125 config SHA256：
  `398dd219e5674ec793a417262af8dbcaf518e6584c815df82557fc9a2bd688ae`

## 3. 评估口径

- dev seeds：`20–59`（40 seeds）
- `endpoint_stop`
- `path_dim=8`
- CPU / `float64`
- teacher 指标全部读取 `summary.json` 的 `teacher` 字段

## 4. 教师结果

| scenario | variant | success | pre-goal 3D RMSE | timeout | safety | plane miss | saturation |
|---|---|---:|---:|---:|---:|---:|---:|
| default | fixed | 0.8250 | 2.660497 | 0.0000 | 0.0000 | 0.1750 | 0.1867 |
| default | control | 0.8250 | 2.660497 | 0.0000 | 0.0000 | 0.1750 | 0.1867 |
| strict125 | fixed | 0.6250 | 2.635537 | 0.0000 | 0.0000 | 0.3750 | 0.1883 |
| strict125 | control | 0.6250 | 2.635537 | 0.0000 | 0.0000 | 0.3750 | 0.1883 |

fixed 与 control 的指标逐字段一致，说明：

- teacher-fixed 配置只显式固定了原本已由旧配置/CLI 提供的控制语义；
- 本次配置修正没有改变教师行为；
- Goal22 的问题不在教师本身，而在于新 BC 配置没有显式继承这些控制开关。

## 5. 配置差异

对比旧 R2B current 配置，teacher-fixed 配置只包含以下预期变化：

- `experiment.name` 和 `experiment.output` 改为新名称/新目录；
- 在 `termination` 中显式加入：
  - `pc_goal_plane_miss_termination_enabled=true`
  - `goal_plane_miss_penalty=0.0`
  - `goal_plane_miss_distance_weight=0.0`

没有改变网络、奖励、路径维数、传感器 profile、crab/current、robust terminal 或
route speed floor。

## 6. 门禁

| scenario | success floor | pre-goal not worse | timeout not worse | safety not worse | overall |
|---|---|---|---|---|---|
| default | PASS (`0.8250 >= 0.80`) | PASS | PASS | PASS | **PASS** |
| strict125 | PASS (`0.6250 >= 0.45`) | PASS | PASS | PASS | **PASS** |

## 7. 判定

Phase 1 **PASS**，允许进入 Phase 2 教师数据采集。strict125 教师 success 为
`0.6250`，明显高于 `0.45` 门禁；没有停止条件被触发。
