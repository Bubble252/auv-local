# 任务：BC+DDPG 微调稳定性修复（B2 门）

记录日期：2026-10-29

状态：**B2 门通过（E5，200k）；B3/B4/B5 尚未启动**

工作目录：`/home/bubble/水下机器人/settled_局部规划器_反应式`

关系：本文件接在 `goal6_120m_50hz_bc.md` 之后，只修 BC+RL 微调阶段，不改观测契约、
动作范围、奖励定义、成功判定和 `update_to_data_ratio`。`goal6` 的 PARTIAL 结论继续有效；
在 B2 门重新通过之前，不得启动 B3/B4/B5，也不得声称 120m/50Hz/BC 方案通过。

## 1. 问题背景

`goal6` 的 B2a/B2b 都从 B1 的 BC actor 初始化，BC actor 在 step 0 评估为 6/6；训练 200k
转移后变成 0/6，全部越界，两维动作饱和率 1.0。证据：

| 实验 | batch | 最终 S1/S2 | 最后 20% S1 | 最后 20% S2 | 失败构成 | 两维饱和 |
|---|---:|---|---:|---:|---|---|
| B2a | 128 | 0/3、0/3 | 0/15 | 0/2 | 全部越界 | 1.0 / 1.0 |
| B2b | 256 | 0/3、0/3 | 0/27 | 0/31 | 53 越界、5 碰撞 | 1.0 / 1.0 |

- 两次运行的 `best_evaluation.json` 都指向 `env_transitions=0` 的 BC actor。
- B2a：critic `mean_q` 头段 `0.4899`，单步范围约 `±180`；尾段 `8.4847`；
  最终 `actor_raw_abs_max=23.87`、`tanh_grad_norm=1.6e-5`。
- B2b：critic `mean_q` 头段 `0.5628`、尾段 `6.8532`；最终
  `actor_raw_abs_max=36.17`、`tanh_grad_norm=0.0`。

结论：不是 BC 没学会避障，而是 off-policy 微调在高更新比下把 BC actor 推入饱和区。

## 2. 问题假设

| 编号 | 假设 | 代码依据 | 对应实验 |
|---|---|---|---|
| H1 | critic 尚未收敛时 actor 就开始更新 | `update()` 从第一个梯度步起 `policy_delay=1` 就更新 actor | D1 |
| H2 | BC 锚太弱且衰减太快 | `bc_auxiliary_weight=0.1`，`decay_updates=10000`，相对 Q 量级可忽略 | D2 |
| H3 | actor 学习率相对 critic 过大 | `actor_lr=critic_lr=2e-4`，190k 步 | D3 |
| H4 | 单 critic 无 target smoothing，Q 过估计 | `twin_critics=false`，`target_policy_noise_sigma=0.0` | D4 |
| H5 | actor 更新频率过高，放大错误 Q | `policy_delay=1` | D5 |

## 3. 冻结边界

以下内容在整个 goal7 中不得修改：

- 17 维观测契约、13 扇区语义、120m 归一化；
- 动作范围 `[±0.5 m/s², ±0.2618 rad/s]`；
- 奖励定义与 `reward_audit` 的四条门；
- 成功判定、评估场景、评估 seeds `0,1,2`；
- `update_to_data_ratio=1.0`；
- `gamma=0.99798`、`tau=0.0005`、`history_length=16`、`batch_size=256`；
- B1 的 BC checkpoint 与 `bc_expert_v6_dagger_v2` 数据。

允许修改的只有：`actor_warmup_updates`、`bc_auxiliary_weight/decay_updates`、
`actor_lr`、`twin_critics`、`target_policy_noise_sigma`、`policy_delay`、actor 隐藏层
噪声的 `actor_hidden_sigma_init`/`actor_noise_sigma_trainable`，以及这些量的组合。

## 4. 诊断矩阵 D0–D5

统一预算 **40k transitions**、`num_envs=64`、seed 0、`batch_size=256`、
`learning_starts=10000`、评估每 2k transitions、评估 seeds `0,1,2`。每条约 10 分钟，
8 张 4090 可分两波并行。

| 编号 | 变量 | 值 | 目的 |
|---|---|---|---|
| D0 | 基线 | 原 B2b 配置，eval 每 2k | 定位第一次坏掉的更新步 |
| D1 | actor 冻结 | `actor_warmup_updates=10000` | 验证 critic warmup 是否足够 |
| D2 | BC 锚常驻 | `bc_auxiliary_weight=1.0`、`decay_updates=0` | 验证 BC 锚是否够强 |
| D3 | actor 步长 | `actor_lr=2e-5` | 验证是否 actor 步子太大 |
| D4 | 抑制过估计 | `twin_critics=true`、`target_policy_noise_sigma=0.2` | 验证 Q 过估计 |
| D5 | actor 降频 | `policy_delay=4` | 验证更新频率 |

## 5. 诊断门

每条 D 实验必须同时满足：

- [ ] 最终 6 条评估 S1 `3/3`、S2 `3/3`；
- [ ] 最后 20% episode 中 S1、S2 各至少 1 次成功；
- [ ] 两维动作饱和率均 `<0.5`；
- [ ] 最终 `actor_raw_abs_max < 1.0`；
- [ ] 最终 `tanh_grad_norm > 1e-4`；
- [ ] critic 无非有限值，`mean_q` 尾段与头段同量级（不出现 `±100` 量级）。

任一不满足即该变体失败；全部单变量失败时进入第 7 节的组合迭代。

### 5.1 D0–D8 实际结果（40k）

| 编号 | 配置 | 最终评估 S1/S2 | 探索模式 6 条 | 两维饱和 | `actor_raw_abs_max` | 结论 |
|---|---|---|---|---|---|---|
| D0 | 原 B2 配置 | 0/3、0/3 | - | 1.0 / 1.0 | 11.90 | 复现失败；约 4–6k 梯度步开始退化 |
| D1 | actor 冻结 10k 更新 | 0/3、0/3 | - | 1.0 / 0.0 | 3.98 | 失败；解冻后仍然崩 |
| D2 | BC 锚 1.0 常驻 | 3/3、3/3 | 0/6 | 0.0 / 0.0 | 0.32 | 评估稳定，但探索模式全失败 |
| D3 | `actor_lr=2e-5` | 0/3、0/3 | - | 0.0 / 0.0 | 1.29 | 失败 |
| D4 | twin critics + smoothing | 0/3、0/3 | - | 0.89 / 1.0 | 7.44 | 失败 |
| D5 | `policy_delay=4` | 0/3、0/3 | - | 0.0 / 0.46 | 4.06 | 失败 |
| D7 | 锚 1.0 + 隐藏噪声 0.05 | 3/3、0/3 | 0/6 | 0.0 / 0.0 | 0.23 | S2 退化，失败 |
| D8 | 锚 1000 + 隐藏噪声 0.05 | 3/3、3/3 | 6/6 | 0.0 / 0.0 | 0.42 | 通过 40k 诊断 |

D0 的评估时间线定位了退化起点：`0–14016` 转移仍为 6/6，`16000` 起变为 0/6，
`26048` 起动作饱和接近或达到 1.0；对应约 4–6k 梯度步。

### 5.2 关键发现：探索噪声压过了策略输出

同一条 D2 checkpoint：

- `evaluate=True`（取 NoisyNet 均值）：S1 `3/3`、S2 `3/3`；
- `evaluate=False`（采样噪声）：S1 `0/3`、S2 `0/3`，全部越界；
- 对同一观测采样 200 次，动作均值为 `[-0.481, +0.244]`，几乎贴住
  `[-0.5, +0.2618]` 的动作边界。

原因：隐藏层 `NoisyLinear(sigma_init=0.5)` 远大于输出层均值初始化
`actor_output_weight_init_bound=0.003`，且 BC checkpoint 会把 BC 训练时的 sigma 参数
一起载入。于是训练 rollout 的动作由噪声主导，replay 里充满无效动作；即使评估时策略是
好的，训练 episode 也必然失败。

修复（已实现并测试）：

- 新增 `DDPGConfig.actor_hidden_sigma_init`，单独控制 actor 隐藏层 NoisyNet 的初始 sigma；
- BC 载入后调用 `reset_actor_noise_sigmas()`，只重置 sigma，不覆盖学到的权重均值；
- D7 证明只降噪声不够（锚 1.0 仍会让 S2 退化），D8 把 BC 辅助权重提高到 1000 后同时
  通过评估与探索模式。

进一步在 E3（200k）上做“只改噪声、不重训”的 sigma 扫描，使用同一份均值权重和同样的
6 条评估场景，结果如下：

| 隐藏层 sigma | 探索模式成功 | 动作偏差均值（u / yaw） | 结论 |
|---:|---|---:|---|
| 0.05 | 0/6 | 0.194 / 0.171 | 完全不可用 |
| 0.02 | 0/6 | 0.065 / 0.063 | 仍不可用 |
| 0.01 | 6/6 | 0.012 / 0.017 | 可用 |
| 0.005 | 6/6 | 0.005 / 0.009 | 可用，偏保守 |
| 0.002 | 6/6 | 0.003 / 0.004 | 探索过弱 |

因此 0.05 不是“略大”，而是比可用阈值高一个量级。另一个问题是 sigma 本身可训练：
E3 从 0.05 起步，第二层最大 sigma 在 200k 后漂到 0.16，会重新破坏探索。修复方式是
新增 `actor_noise_sigma_trainable`，在 BC+RL 微调时冻结 actor 的 `rho`，只训练权重均值。

### 5.3 E1/E2：200k 锚定确认（无噪声修复）

| 编号 | 配置 | 最终评估 S1/S2 | 最后 20% 训练 episode | 结论 |
|---|---|---|---|---|
| E1 | 锚 1.0，噪声继承 0.5 | 3/3、3/3 | 0/132（S1 0/90、S2 0/42） | 评估稳定，但训练 episode 全失败 |
| E2 | 锚 1.0，`actor_lr=2e-5`，噪声 0.5 | 3/3、3/3 | 0/133 | 同上 |

E1/E2 证明 BC 锚能把评估稳定在 6/6 到 200k，但高噪声下训练 rollout 仍然全失败；
这两组只作为“评估门”证据，不能通过 B2 门。

### 5.4 E3：200k 最终候选

E3 = D8 配置（锚 1000、隐藏噪声 0.05、batch 256、200k）。门禁以第 5 节和 goal6 的
B2 门同时判定。

| 指标 | 结果 |
|---|---|
| 最终评估 S1/S2 | 3/3、3/3 |
| 最后 20% 训练 episode | 0/64（S1 0/32、S2 0/32） |
| 失败构成 | 全部越界，约 4.5 s 结束 |
| 两维饱和 | 0.0 / 0.0 |
| `actor_raw_abs_max` | 0.4196 |
| `tanh_grad_norm` | ≈0.9 |
| 结束 sigma | 第一层 0.033–0.078；第二层 0.031–0.162 |

结论：确定性评估仍好，但采样噪声下的训练 rollout 全失败；E3 不能通过 B2 门。

### 5.5 D9：sigma 0.01 + 冻结噪声（40k）

D9 在 D8 基础上只改两点：`actor_hidden_sigma_init=0.01`、
`actor_noise_sigma_trainable=false`。

| 指标 | 结果 |
|---|---|
| 最终评估 S1/S2 | 3/3、3/3 |
| 探索模式 6 条 | 6/6 |
| 两维饱和 | 0.0 / 0.0 |
| `actor_raw_abs_max` | 0.4167 |
| `tanh_grad_norm` | 0.899 |
| 动作采样 std | 0.015 / 0.016 |
| 结束 sigma | 两层均为 0.010 |

40k 只有 625 个控制步/环境，不足一个 3000 步 episode，因此不能评价“最后 20% 训练
episode”；该门由 200k 的 E4 判定。

### 5.6 E4：200k 最终策略通过，但训练窗口门失败

E4 在 D9 基础上把预算加到 200k，配置不变（锚 1000、sigma 0.01、冻结噪声）。

| 指标 | 结果 |
|---|---|
| 最终评估 S1/S2 | 3/3、3/3 |
| 最终 checkpoint 探索 rollout（64 环境 × 3 轮） | S1 107/107、S2 85/85 |
| 动作采样 std | 约 0.015 / 0.016 |
| 两维饱和 | 0.0 / 0.0 |
| `actor_raw_abs_max` | 0.4178 |
| `tanh_grad_norm` | 0.888 |
| critic 头/尾 `mean_q` | 0.56 / 0.89，尾段范围 `[-3.44, 4.62]` |
| 结束 sigma | 两层均为 0.010 |
| 正确窗口下的最近 20% episode | 49 条：S1 6/45、S2 0/4 |

这里有两个不同结论，必须分开看：

1. **最终策略已经解决探索失败**：同一 checkpoint 在 64 环境、采样噪声模式下 192/192
   成功，确定性评估 6/6，饱和为 0。
2. **训练期“最近 20% episode”仍然失败**：这 49 条 episode 的完成时间虽然是 166k–199k，
   但它们的起点约在 100k–154k，对应的是训练中段尚未恢复的策略；200k 内没有新的
   episode 能在 160k 之后启动并完成（单条约 60k 转移）。因此该窗口不能代表最终策略。

同时修正了 `scripts/summarize_p1_bc_rl_run.py` 的窗口实现：原实现误把 `rl_step`
（episode 内步数）当作全局训练步，现已改为按完成顺序取最后 20%，并补了回归测试。

### 5.7 E5：低 actor lr + 强锚 + 冻结噪声（200k）

为同时满足“最终策略稳定”和原 B2 的“训练最近 20% episode 有成功”，E5 在 E4 基础上只改
`actor_lr=2e-5`，让 actor 在 200k 内保持贴近 BC，减少中段策略的探索敏感性。

| 指标 | 结果 |
|---|---|
| 最终评估 S1/S2 | 3/3、3/3 |
| 最近 20% episode | 39 条：S1 18/18、S2 21/21 |
| 最近 20% 最小 clearance | 1.2 m |
| 两维饱和 | 0.0 / 0.0 |
| `actor_raw_abs_max` | 0.4279 |
| `tanh_grad_norm` | 0.8936 |
| critic 头/尾 `mean_q` | 0.547 / 1.651，尾段范围 `[-6.43, 7.08]` |
| 结束 sigma | 两层均为 0.010 |
| 最终 checkpoint 探索 rollout（64 环境 × 3 轮） | S1 111/111、S2 81/81 |
| 实际 `update_to_data_ratio` | 1.0 |

E5 同时通过最终评估、训练最近 20% episode 和最终 checkpoint 探索 rollout，因此
`configs/rl/vector_120m50hz/diagnostics/e5_low_lr_frozen_noise_001_200k.json` 是 B2 胜者。
门禁矩阵见 `artifacts/m4/p1/analytic_120m50hz/diagnostics/gate_matrix.csv`，E5 的 9 项全部
为 `pass`。

结论：BC+DDPG 微调阶段的问题已经定位并解决。关键不是 critic 或 reward，而是
NoisyNet 隐藏层探索噪声的幅度与可训练性：0.05 比可用阈值高一个量级，且 `rho` 在训练中
还会继续漂移。把 sigma 固定为 0.01、冻结 actor 噪声参数，并把 actor lr 降到 2e-5，
就能在保持 BC 能力的同时让 RL 稳定更新。

## 6. 组合与 200k 确认

- [x] 从 D0–D5 中选保护措施，形成 D8/D9 组合，仍用 40k 预算验证。
- [x] 组合通过后，用同一配置跑 200k transitions（batch 256），执行 goal6 的 B2 门。
- [x] E5 通过 B2 门；允许启动 B3/B4/B5，B3 的 batch 固定为 256。
- [ ] B3/B4/B5 与 P5 尚未启动，等用户确认下一步。

B2 门与 goal6 一致：固定 6 条解析评估 S1 `>=2/3`、S2 `>=1/3`；最后 20% episode 两个
场景各有成功；`reset_tanh_grad_norm > 1e-4`；两维饱和 `<0.5`；无 NaN/Inf；实际
`update_to_data_ratio=1.0`；`gamma`/`tau` 与 manifest 一致。

这里的“最后 20% episode”按**完成顺序**取最后 20%，与 `vector_trainer` 的
`_recent_scenario_successes` 一致；`episodes.csv` 的 `rl_step` 是 episode 内步数，
不能当成全局训练步使用。此前 summary 脚本误用了该列，已修正并补回归测试。

## 7. 允许的迭代

- 若 D0–D5 全部失败，允许追加 D8+ 组合，但每条仍只相对上一轮改一个维度。
- 若 40k 全部保持 6/6 但 200k 又崩，优先怀疑学习率随步数累积，追加“分段降 lr”实验。
- 若 critic 仍出现 `±100` 量级 Q，先做 reward 尺度与 target 初始化诊断，再改网络结构。
- 任何迭代都必须写回本文件，并保留失败结果，不得只报告成功变体。

## 8. 实现任务

- [x] `DDPGConfig` 增加 `actor_warmup_updates`（默认 0，向后兼容），在 `update()` 中生效。
- [x] `DDPGConfig` 增加 `actor_hidden_sigma_init` 与 `actor_noise_sigma_trainable`。
- [x] `actor_noise_sigma_trainable=false` 时冻结 actor 的 `rho`，权重均值仍正常训练。
- [x] `cuda_graph_update=true` 与 `actor_warmup_updates>0` 同时出现时显式报错。
- [x] 单元测试：warmup 期间 actor 参数不变、critic 参数更新；warmup 后 actor 开始更新。
- [x] 单元测试：冻结噪声时 `weight_rho`/`bias_rho` 不变，`weight_mu` 仍更新。
- [x] 生成 `configs/rl/vector_120m50hz/diagnostics/d0..d5*.json`。
- [x] 新增 `scripts/check_p1_bc_rl_gates.py`，按第 5 节门禁输出逐条 PASS/FAIL。
- [x] 跑完 D0–D5，生成 `training_summary.json` 和 gate 结果。
- [x] 跑完 D7–D9、E1–E5，生成 `training_summary.json` 和 gate 结果。
- [x] 修正 `summarize_p1_bc_rl_run.py` 的“最近 20% episode”窗口，并补回归测试。
- [x] 把结果写回本文件。
- [ ] 同步 `doc/01_技术栈.md`、`doc/02_执行计划.md`。

## 9. 运行方式

```bash
cd "/data/yhx/underwater-robotics/src/plan_p1/settled_局部规划器_反应式"
PY=/data/yhx/underwater-robotics/envs/conda_envs/oceansim/bin/python
export PYTHONPATH=src:/data/yhx/underwater-robotics/src/plan_p1/settled_全局规划器/src

CUDA_VISIBLE_DEVICES=0 "$PY" scripts/train_p1_vector.py \
  --config configs/rl/vector_120m50hz/diagnostics/e5_low_lr_frozen_noise_001_200k.json \
  --bc-checkpoint outputs/m4/p1/analytic_120m50hz/matrix/b1_bc_only_v6_dagger2_w40_30k/actor.pt \
  --demos outputs/m4/p1/analytic_120m50hz/bc_expert_v6_dagger_v2/demos.npz \
  --output outputs/m4/p1/analytic_120m50hz/diagnostics/e5_low_lr_frozen_noise_001_200k \
  --num-envs 64 --total-env-transitions 200000 --seed 0 --device cuda \
  --eval-every 2000 --eval-seeds 0 1 2
```

每个实验独立输出目录；不得覆盖旧结果。大型 `demos.npz`、`actor.pt` 只留服务器。

## 10. Git 检查点

```bash
cd "/home/bubble/水下机器人"
git status --short
git diff --check
git diff --stat
```

建议提交信息：

| 阶段 | 建议提交 |
|---|---|
| 文档 | `docs: add bc ddpg stabilization plan` |
| 代码 | `feat(rl): add actor warmup updates` |
| 配置 | `config: add bc ddpg diagnostic matrix` |
| 结果 | `docs: record bc ddpg diagnostic results` |

只在用户明确要求时执行 `git add/commit/push`；不提交 `outputs/`、checkpoint、
`demos.npz`、大型 CSV。

## 11. 交付物

```text
doc/goals/goal7_bc_ddpg_stabilization.md
configs/rl/vector_120m50hz/diagnostics/d0_baseline.json ... e5_*.json
scripts/check_p1_bc_rl_gates.py
scripts/summarize_p1_bc_rl_run.py
artifacts/m4/p1/analytic_120m50hz/diagnostics/<run>/training_summary.json
artifacts/m4/p1/analytic_120m50hz/diagnostics/gate_matrix.csv
```

## 12. 二维可视化（2026-10-29）

用 E5 最终 checkpoint 生成了俯视 2D 动图，统一使用解析 13 扇区感知、50 Hz、关闭起点抖动。
成功案例：S1 空场、S2 单圆柱（确定性与探索噪声）、专家集单圆柱、S 形双障碍、右侧阻挡。
失败案例：S3 偏移门（未训练场景，零样本碰撞）和 expert_blocking_left 探索噪声碰撞，用于
说明当前策略的边界，不代表 B2 门失败。

入口：`artifacts/m4/p1/analytic_120m50hz/visualizations/gifs/index.html`；
总览：`.../gifs/contact_sheet.png`；生成脚本：`scripts/render_p1_rollout_2d.py` 与
`scripts/build_p1_vis_gallery.py`。
