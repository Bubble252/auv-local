# Goal175 blocker：COMP 路线收益过门，但纵向护栏失败且 residual head 近乎无净作用

**日期**：2026-11-14<br>
**Goal**：Goal175<br>
**触发**：预注册 A2 Gate175 在 `current10 × both radii` 上 cross-track 和整条
route 都通过，但 vertical guard 失败；同时训练后的 `P_RUD_075` residual
相对 `zero_bc50` 几乎没有闭环路线效果。

## 现象

- 有效 root：
  `/data/yhx/underwater-robotics/runs/auv-local/goal175/20261114-goal175-c186c74-a2`
- 训练与评价：`6/6` training runs、`72/72` eval cells；training audit 和
  artifact audit 均 PASS；held-out `100–149` 关闭。
- `P_RUD_075` 在 current10 两 radius：
  - Δ3D route `-0.304 m / -13.4%`；
  - Δcross p90 `-0.439 m / -12.3%`；
  - Δvertical p90 `+0.034 m`；
  - Δterminal vertical `+0.037 / +0.024 m`；
  - success `+14 / +24`，timeout/safety 为 0，speed guard 通过。
- `P_RUD_025` 同样 cross/route 通过，但 vertical p90 `+0.034 m`、terminal
  vertical `+0.043 / +0.021 m`，vertical guard 失败。
- `P_ALL_075` average vertical p90 `+0.009 m`，但 default terminal vertical
  `+0.026 m`，仍使 pair vertical guard 失败。
- 残差归因：`P_RUD_075_COMP` 相对 `zero_bc50` 的 Δ3D 仅 `+0.00037 m`，
  Δcross `+0.00068 m`，Δvertical `-0.00001 m`，success `0`；residual
  mean absolute action 约 `0.0029`，约 base L2 的 `1.0%`。
- ALOS base-only 对照（`zero_alos_comp` vs `zero_alos_none`）在 current10
  也出现 vertical p90 `+0.036–0.037 m`、terminal vertical `+0.023–0.049 m`，
  说明纵向代价主要来自 COMP 基座，而不是 residual 训练副作用。

证据：

- `doc/goals/goal175_result_20261114.md`
- `doc/goals/goal175_a1_result_20261114.md`
- `/data/yhx/underwater-robotics/runs/auv-local/goal175/20261114-goal175-c186c74-a2/a2_summary.json`
- `/data/yhx/underwater-robotics/runs/auv-local/goal175/20261114-goal175-c186c74-a2/run_manifest.json`

## 已排除

1. **训练没完成 / checkpoint 缺失**：6/6 `final.pt`、`training_manifest.json`、
   `curriculum_episodes.jsonl`、`curriculum_state.json` 存在，training audit PASS。
2. **评价不完整 / seed 错**：72/72 summaries/traces，seeds 20–59，artifact audit PASS。
3. **current 泄漏 / crab 残留**：所有 cell crab=0，COMP 使用 estimated current，
   summary/trace 审计通过。
4. **课程覆盖 pressure 轴**：评价没有传 `--curriculum-stage`，
   `curriculum_stage=null`，与 A1 的无效旧 run 不同。
5. **success/speed/timeout/safety 护栏**：均通过；最大地速 `2.053 m/s`，
   overspeed episode `0`。
6. **“只是 residual 全零”**：`P_RUD_075_COMP` residual action 非零，
   mean absolute action 约 `0.0029`；问题是它没有形成有用的闭环纠偏。

## 可证伪假设

### H1：编码器表征里有纵向信息，但 residual head 没把它转成有效动作

预测：用现有 checkpoint 的 `z_t` 线性回归 `vertical_error / pitch_error / q / w`
应有可解释的 held-out 预测力；但对纵向/俯仰输入做小扰动时，residual action 的
Jacobian 近似为零或方向不稳定。若 probe 好而 Jacobian 不敏感，H1 成立。

最便宜证伪：eval-only latent probe + finite-difference Jacobian，不训练、不打开
held-out；直接对现有 6 个 A2 checkpoint 和 `zero_bc50` 比较。若 probe 本身很差，
H1 不成立，转向预训练目标、显式纵向 skip 或 DADE 历史编码。

### H2：表征可用，但 residual 输出尺度/初始化/credit 让动作头学不到纵向纠偏

预测：只放大 residual 输出、只微调最后一层、或只训练 action head 时，
vertical p90 / terminal vertical 会按剂量移动；冻结编码器与微调最后一层不会有
本质差异。若这些便宜干预都不移动，H2 不成立，优先检查 action interface / 观测接线。

最便宜证伪：在现有 checkpoint 上做 residual scale sweep（0.25/0.5/0.75/1.0）
和 last-layer-only 短训练；评价只跑 current10 × 两 radius × dev seeds，不扩 seed。

### H3：纵向代价来自 ALOS-COMP 基座，不是 residual head

预测：即使 residual 为零，`zero_alos_comp` 相对 `zero_alos_none` 仍出现相同的
vertical p90 / terminal vertical 退化；当前 A2 数据已经支持这个方向。
若后续 residual 干预只改变横向、不改变纵向，应把纵向修复放到 ALOS/guidance 层，
而不是继续训练 residual。

最便宜证伪：直接用 `zero_alos_comp` / `zero_alos_none` 的现有 16 个 current10
cell 做配对；无需新训练。

## 当前决策

`decision = pivot`：停止“先换 Transformer actor”的路线；不扩 seed、不改
reward/termination/success/radius、不打开 held-out。先做 eval-only latent probe、
纵向/俯仰 Jacobian 敏感性和 last-layer-only / frozen-encoder 对照。

## 下一步

1. 写 Goal176 预注册：`z_t` probe + residual action Jacobian + last-layer pilot；
2. 先跑最便宜的 read-only probe/Jacobian，不训练；
3. 只有当 probe 与 Jacobian 都支持“表征可用、动作头可学”时，才进入短训练；
4. 若 probe 差，再评估预训练目标、显式纵向 skip 或 DADE；若 probe 好但动作
   不响应，优先修 residual head / action interface。
