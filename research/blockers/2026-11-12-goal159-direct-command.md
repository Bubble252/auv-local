# Goal159 blocker：C0 direct-command 污染训练

**日期**：2026-11-12<br>
**状态**：`INVALID_TRAIN_DIRECT_COMMAND / ROOT_CAUSE_CLEAR / CORRECTED_RUN_STARTED`<br>
**Goal**：Goal159

## 1. 现象

旧 Goal159 的 3 个 1M checkpoint 训练 manifest 显示：

- `pc_stage=C0`
- `curriculum.enabled=false`
- `curriculum_direct_commands_enabled=true`
- `reward.hold_bonus=true`

按 `BatchedPCLowLevelEnv` 的实现，C0/C1 + ALOS + direct flag true 会启用
training-only direct command。因此训练不是沿参考路径跟踪，而是随机 direct command。

## 2. 证据

- 环境配置 SHA：`cdc2718bbc81a3c9934314ff5fd16ed313164765b7d797b45890eec701322c6d`
- `pc_stage=C0`，`pc_control_mode=alos`
- 训练 completed episodes `712 / 718 / 686`，success 仅 `26 / 27 / 23`
- 旧 eval 虽然使用 `--disable-curriculum-direct-command`，但 checkpoint 训练分布错误
- 旧 scale 剂量全部建立在错误 checkpoint 上

## 3. 已排除项

- 不是评测入口 direct-command 泄漏：eval 明确关闭；
- 不是 residual 初始化或 checkpoint 损坏：训练和 eval 均完整落盘；
- 不是 seed 噪声：三批训练都走同一错误语义；
- 不是算法或 reward 结论：本 blocker 判 `INVALID`，不判 `FAIL`。

## 4. 可证伪假设

### H159-INV-1：切换到 C3/ALOS 后 direct command 消失，训练成功率应上升

成立应看到：修正版 smoke manifest 的 `direct_commands_active=false`；训练 episode
success 明显高于旧批的约 3%；评测 checkpoint 的 residual 与路径误差不再完全脱节。

失败应看到：`direct_commands_active` 仍为 true，或训练 success 仍极低；此时继续查
环境控制语义，不进入 eval。

### H159-INV-2：旧 eval 的正增益主要是强基线/评测路径，而不是错误训练 residual

成立应看到：修正版 H3 zero 与旧 checkpoint policy 的差异不能再复现为稳定 current07
增益，或修正版 policy 的增益方向改变。

失败应看到：修正版训练后仍复现同类 current07 增益，且 residual 与误差方向相关；
此时才可把 current07 作为候选 Gain 证据。

## 5. 修复

- 新配置：`configs/lowlevel/goal159/env_h3_current07_10_train_c3.json`
- 新训练配置：`configs/lowlevel/goal159/train_c3_seed86201..86203.json`
- 新 runner：`scripts/run_goal159c_train.sh`
- 训练 manifest 新增 `direct_commands_active`
- smoke 已验证 `direct_commands_active=false`
- Goal159c 3×1M 正在运行

## 6. Decision

`continue`：旧结果作废，按修正协议重跑；不改算法、reward、terminal、success、radius，
不扩 seed，不打开 held-out。
