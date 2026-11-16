# Goal177 blocker：head-only 权限存在，但 residual 幅度未过 PPO 门槛

**日期**：2026-11-14<br>
**Goal**：Goal177<br>
**触发**：Stage A 的 head-only 审计得到 `GATE_A_FAIL / RESIDUAL_TOO_SMALL_FOR_PPO`；
v1/v2 的审计接线问题已修正，但修正后仍显示 PPO head 的有效动作幅度不足。

## 现象

- Stage A 完整覆盖 `144/144` cells：`128` policy + `16` zero，dev seeds `20–27`，
  `16 arms × 4 levels × 2 radii`。
- 正式 v3 head-only 审计：
  - vertical stern 最大 median share `5.05e-3`；
  - cross rudder 最大 median share `3.66e-3`；
  - forward rpm 最大 median share `2.13e-2`；
  - 三方向 sign consistency 中位数 `1.0`；
  - 最大 head residual/action_std `0.0737`，低于预注册门槛 `0.20`。
- composed authority（冻结 BC + PPO）可达 `0.4–0.65`，但这不能代表 PPO head。
  v2 曾把 composed delta 当作 head delta，得到错误的 `PASS`；正式 v3 已将其
  降级为诊断量。
- `A_S075_RUDDER_ONLY` 与 `zero_bc50` 在 current10/default 的 3D/cross/vertical
  差异只有 `+0.0005 / +0.0008 / +0.00001 m`，说明 native residual 没有形成
  净闭环纠偏。
- 提高 authority 后 `A_S150_RUDDER_ONLY` 改善 cross/3D，但 head residual/std
  仍只有 `0.0069`；`A_S250_ALL` 虽达 `0.0737`，但 vertical/completion 回退，
  strict125 success 降到 6–7/8，magnitude/rate clip 约 `0.84/0.64`。

证据：

- `doc/goals/goal177_result_20261114.md`
- `doc/goals/goal177_residual_head_action_interface_3d_20261114.md`
- `/data/yhx/underwater-robotics/runs/auv-local/goal177/20261114-goal177-stage-a/stage_a/stage_a_summary.json`
- `/data/yhx/underwater-robotics/runs/auv-local/goal177/20261114-goal177-stage-a/stage_a/stage_a_summary.md`
- `/data/yhx/underwater-robotics/runs/auv-local/goal177/20261114-goal177-stage-a/stage_a/stage_a_summary.v1_policy_only.json`
- `/data/yhx/underwater-robotics/runs/auv-local/goal177/20261114-goal177-stage-a/stage_a/stage_a_summary.v2_composed_only.json`

## 已排除

1. **Stage A 未跑完 / cell 缺失**：manifest 有 `144` jobs，summary 有 `144`
   cells，`problems=[]`。
2. **held-out 泄漏**：所有 cell 的 seeds 均为 `20–27`；trace audit 未发现
   `seed >= 100`。
3. **head 完全无方向**：三方向输出随输入变化，sign consistency 中位数 `1.00`。
4. **encoding 没有纵向信息**：Goal176 的 `vertical_error / pitch_error / q / w`
   held-out R² 为 `0.9967–0.99996`，本轮没有推翻该 probe。
5. **纯粹是 wrapper authority 没有打开**：authority sweep 已到 `2.5`，head-only
   share 仍只有约 `0.003–0.021`；继续放大 authority 先让 composed/BC 分支饱和。
6. **稳定性接管失败**：timeout/safety 均为 0，最大地速 `2.196 m/s`，低于
   `2.5 m/s` 契约。

## 可证伪假设

### H177-A1：PPO head 的有效输出幅度被 scale/normalization 压低

预测：保持同一 checkpoint、同一 base 和 reward，只对 head residual 做 eval-only
输出 gain / scale sweep，`head residual/std` 应按剂量单调上升；当它接近 `0.20`
时，闭环 route/cross/vertical 敏感度也应同步上升。若增益很大但 head 仍不移动，
说明问题在 head 参数化或 BC/PPO 方向抵消，而不是尺度。

最便宜证伪：复用现有 Stage A trace 和 checkpoint，先做 head-only 输出 gain
`1x / 4x / 8x` 的 eval-only replay；不训练、不扩 seed、不改 reward。

### H177-A2：冻结 BC residual 与 PPO head residual 在同一状态下相互抵消

预测：在相同 observation 上分别记录 `bc_delta`、`ppo_delta` 和 composed delta；
若 correlation 为负或 PPO delta 在大部分状态落在 BC delta 的反方向，则即使 head
有可见幅度也不会形成净闭环效果。若 head 与 BC 方向稳定同向，则抵消假设不成立，
优先查 PPO credit。

最便宜证伪：对现有 Stage A trace 增加纯离线字段分解；必要时只重跑一个
current10/default cell 记录 `actuator_residual_bc_applied` 和
`actuator_residual_ppo_applied`，不重新训练。

### H177-A3：rate/magnitude limit 在 head 仍很小时截断有效动作

预测：若把 PPO head 单独放大后，rate clipping 在关键状态显著上升，而 route 对
head 的响应仍不增加，则动作限制或通道耦合是下一瓶颈。若 head 放大后 route
按剂量响应且 clip 仍可控，则限制不是主因，应进入短 head-only pilot。

最便宜证伪：在 eval-only gain sweep 中同时记录 magnitude/rate clip 和
head-only Jacobian；无需新训练。

## 当前决策

`decision = continue`：不进入 Stage B，不改变主线算法族；先做 head-only
幅度/抵消/限幅的 eval-only 诊断。`GATE_A_FAIL` 是正式结论，不能用 v2 的
composed `PASS` 覆盖。

## 下一步

1. 写新的最小 goal，固定本 Stage A checkpoint、dev seeds 和四 level/两 radius；
2. 先做 head-only gain/scale 的 eval-only replay，记录 head residual/std、
   BC/PPO 方向相关和 clip；
3. 只有当 head residual/std 接近 `0.20` 且 route 按剂量响应时，才考虑
   head-only / last-layer 1M pilot；
4. 不打开 held-out，不改 reward/termination/success/radius，不先换 Transformer、
   DADE 或 encoder 预训练。
