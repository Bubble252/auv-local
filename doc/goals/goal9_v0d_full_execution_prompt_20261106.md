# Goal9 V-0D 完整执行提示词

记录日期：2026-11-06

用途：作为下一个 Goal 的完整提示词，从 R-BC-0D 真实数据开始，按门禁自动推进到
BC 对照、零残差、PPO screen 和正式 3×50 比较。本文是执行提示词，不是实验结果。

```text
你在 /home/bubble/水下机器人/settled_局部规划器_反应式 工作。

目标：把 Goal9 V-0D 从“接口已实现”推进到“有可审计科学结论”。不要只做 smoke，
也不要完成一个阶段就停下来汇报；按下面的阶梯自动推进，直到通过正式比较或触发
停止条件。

## 开始前必须阅读

1. doc/goals/goal9_v0d_direct_pure_path_plan_20261106.md
2. doc/goals/goal9_v0d_non_destructive_prompt_20261106.md
3. doc/goals/goal9_v0d_implementation_audit_20261106.md
4. doc/goals/goal9_execution_state.md
5. doc/goals/goal9_s16_residual_replan.md

以当前 worktree 和真实文件为准，不要凭对话记忆修改代码。

## 最高优先级约束

- 尽可能非破坏性写代码：新增可选字段、可选参数、新配置、新脚本优先。
- 不删除、不移动、不重写旧配置、旧 checkpoint、旧 artifacts、旧报告和旧测试数据。
- 不改变旧 3D BC、旧 PPO 配置和旧评估口径的默认行为。
- 不把旧 V0 的 fail_v0 改写成 V-0D 的结果。
- 不通过换 seed、加训练步数、看结果后改门槛来抢救负结果。
- 不 git commit、不 git push、不使用 git add -A。
- 正式训练和正式评估优先放服务器；本地只做 unit test、contract check 和短 smoke。
  若服务器不可用，先完成所有本地可验证部分，并明确报告资源阻塞，不要假装已经
  完成正式实验。
- 每完成一个 gate，更新 doc/goals/goal9_execution_state.md，并保留 manifest、
  checkpoint SHA、seed 和配置 SHA。

## 执行位置选择（服务器优先）

- 除 unit test、contract check 和 1–2 iteration 的短 smoke 外，默认在服务器执行；
  只有在服务器排队时间明显更长、或本地能以更短时间完成同一非正式检查时，才在本地跑。
- 一旦某个 screen 或正式 run 已在服务器启动，不允许为了“快一点”中途迁回本地，
  也不允许把本地 128-env 结果与服务器 4096-env 结果合并统计。
- 正式 run 的 manifest 必须能证明：`num_envs=4096`、同一环境配置、同一训练预算、
  同一 seed 集合、同一 checkpoint 选择规则和同一评估协议。
- 服务器资源不足时报告阻塞并等待，不得静默降级环境数、训练步数或评估 seed 后继续
  使用原来的实验标签。

## 训练预算与防过拟合（预注册）

允许比原来的 1M 环境步数训练更充分，但预算必须事先记录，不能看到结果后临时加码。

### 环境数口径（先核对，不要混用）

历史配置不是“全部 4096”，但服务器上的主力残差训练和 B/C 级扩训确实大量使用 4096：

- `configs/lowlevel/*.json` 中可见分布约为：`4096` 60 个、`128` 59 个、`2048` 12 个、
  `64` 3 个、`256` 1 个、`1024` 1 个。
- 服务器主力脚本默认是 `NUM_ENVS=4096`、`NUM_STEPS=256`、`NUM_MINI_BATCHES=128`，
  历史 100 iterations 的一次训练约等于 `104,857,600` 环境步（`4096 × 256 × 100`）。
- Goal9 的 P/V/ABC 多数配置使用 `128` envs，主要用于本地或小规模 screen；不能把
  `128 × 256 × 32 ≈ 1.05M` 误写成与服务器 4096 主力训练同量级。
- 本次 V-0D 保留 `128` 配置作为本地 smoke/contract 检查，另建
  `configs/lowlevel/odarrl_control_ppo_pc_v0d_bc0d_frozen_residual_server4096_dz64.json`
  作为服务器正式配置。不要直接改坏现有 128 配置。

预注册预算如下（除非触发停止条件，不在看到结果后修改）：

| 阶段 | 环境数 | 每环境步数 | iterations | 总环境步数 |
| --- | ---: | ---: | ---: | ---: |
| 本地 smoke/contract | 128 | 256 | 1–2 | 0.03–0.07M |
| 服务器 screen | 4096 | 256 | 40 | 41,943,040 |
| 正式 3×50 | 4096 | 256 | 100 | 104,857,600 |

硬约束：`4096` 只允许在服务器运行；本地机器不尝试 4096。本地正式上限为 128 envs，
资源不足时可降到 64 或更小，并明确标记为 smoke/contract，不能计入 screen 或正式比较。
正式训练、正式评估和 3×50 必须以服务器 manifest 中的 `num_envs=4096` 为准。

`num_envs` 提高代表每轮并行采样更多、梯度方差更低，不等于更少训练；是否过拟合由
独立验证集、best-validation checkpoint、KL/entropy 和闭环 guardrail 控制，不能靠
把环境数降回 128 来“防过拟合”。

- BC-0D 数据预算：至少约 200k transitions；数据不足时优先补数据，不靠无限增加
  epoch 硬拟合。
- BC-0D 训练预算：max_epochs=300、patience=30 作为上限；按 episode 划分训练/验证，
  以验证集动作 RMSE 选择 best checkpoint。若训练 RMSE 继续下降但验证 RMSE 连续
  patience 轮不改善，按过拟合处理并停止。
- PPO screen 预算：服务器 `4096 × 256 × 40`，即约 41.94M 环境步；不得用
  `128 × 256 × 32` 的 1.05M 结果冒充服务器正式 screen。若显存不足，只能先报告
  资源阻塞并记录实际环境数，不得静默降级后仍沿用同一实验标签。
- 正式 3×50：每个训练 seed 使用与 screen 相同的最大预算和同一套早停规则；若中途
  早停，必须记录实际环境步数和早停原因。正式预算为 `4096 × 256 × 100`，即约
  104.86M 环境步；三个训练 seed 必须使用同一预算和同一验证规则。
- 防过拟合监控：BC 记录 train/validation action RMSE gap；PPO 记录 KL、entropy、
  残差幅度、动作饱和、验证 seed 的 3D RMSE/success/timeout。不得用正式 50 个评估
  seed 做 early stopping 或 checkpoint 选择。
- seed 隔离：训练 seed 使用 90000 段；验证/早停 seed 使用独立区间（例如 200000 段，
  10 个）；screen 使用 0～19；正式评估使用 0～49。验证 seed 与正式评估 seed 不得混用。
- 若训练预算增加后验证指标变差、残差持续放大、timeout/safety 恶化，停止加预算，
  回到最后的 best-validation checkpoint，不继续盲目扩训。

## 阶段 0：契约预检

先检查，不训练：

- 确认 R-BC-0D checkpoint 契约是 actor_command_dim=0、path_dim=8；
- 确认 BC 数据中 command 是 (N,0)，actor_path 是 (N,8)；
- 确认路径只经 path encoder 进入 z_t，command 不进入 z_t、history 或 DADE；
- 确认 path-zero 能作用在路径块上；
- 确认初始状态覆盖非零 u/v/w 和非零 RPM，不把零初速当默认；
- 找到并验证现有 R-BC-3D checkpoint，记录路径、metadata 和 SHA256；
- 把 V-0D PPO 配置中的占位 checkpoint_sha256 替换为真实值之前，不允许启动 PPO。

## 阶段 1：采集并训练 R-BC-0D

- 使用现有 ALOS 教师采集 state + 8D path -> action 示范；
- 先按现有 1.2 m/s 左右的 v0d 环境配置采集，覆盖正常转弯、路径恢复和非零初态；
- 单个 shard 建议从 16 env × 4096 step 开始；目标至少约 200k transitions。数据不足
  时增加 shard，不改变观测契约；
- 用 scripts/train_pc_bc.py 训练 R-BC-0D，输出到独立 artifacts 目录，不覆盖旧 BC；
- BC 训练上限为 max_epochs=300、patience=30，按验证集动作 RMSE 选择 best
  checkpoint；检查训练/验证 RMSE gap，避免过拟合；
- 检查 checkpoint metadata、训练/验证 RMSE、路径编码器确实存在且 alpha 正确；
- 训练后计算真实 SHA256，更新 V-0D PPO 配置；
- 如果 BC-0D 明显不收敛，先补数据或做 DAgger，不要直接上 PPO。

## 阶段 2：BC 基座门禁

用相同环境、相同 seed 集合和相同评估口径比较：

- R-BC-3D：现有 3D BC checkpoint，按其原 path_dim 评估；
- R-BC-0D：新 0D + 8D path checkpoint，path_dim=8。

先跑 20 个评估 seed，报告逐 seed 配对结果：

- 3D RMSE；
- success；
- strict125；
- timeout；
- safety；
- path-zero 下的 3D RMSE。

探索门禁：

- R-BC-0D 相对 R-BC-3D 的 3D RMSE ratio <= 1.25；
- timeout 绝对增加 <= 0.10；
- 不能出现全部 timeout；
- 如果失败，只允许补 BC 数据、DAgger、修路径特征或修训练契约；不得用 PPO 抢救。

## 阶段 3：R-Z0-0D 零残差等价

- 使用 R-BC-0D + 固定为零的残差；
- 与 R-BC-0D 使用相同环境、路径、critic 特权和 seed；
- 检查 max|a_final - a_BC0D| <= 1e-10；
- 检查 BC 参数漂移为 0、残差参数全零；
- 检查逐 seed 指标一致；
- 不满足则先修部署路径，不启动 PPO。

## 阶段 4：R-PPO-0D screen

只有 BC 门禁和零残差都通过才启动：

- 1 个训练 seed × 20 个 screen seed；
- 训练预算默认服务器 `4096 envs × 256 steps × 40 iterations`；使用独立验证 seed
  监控 early stopping，不能用 screen 的 20 个 seed 反过来挑 checkpoint；
- 使用独立验证 seed（例如 200000 段，10 个）监控 early stopping；screen 的 20 个
  seed 只用于门禁判断，不能反过来挑 checkpoint；
- 使用冻结 R-BC-0D + 可训练动作残差；
- 不改变 actor_command_dim=0、path_dim=8、pc_path_track 奖励；
- 记录残差幅度、动作饱和、timeout、safety 和冻结 BC 参数漂移。

Screen 目标：

- 相对 R-BC-0D 的 3D RMSE ratio <= 0.95；
- 相对 R-BC-3D 的 3D RMSE ratio <= 1.05；
- path-zero 必须让 3D RMSE 恶化至少 5%；
- success、strict125、timeout、safety 的绝对恶化不超过 0.02；
- 失败时不得换 seed 重挑 checkpoint，只能按失败原因修一次再重跑。

## 阶段 5：正式比较

只有 R-PPO-0D screen 通过才进入：

- 先冻结最终对比矩阵和配置 SHA；启动后不得改奖励、观测、路径编码、DADE、残差
  scale、算法超参、训练预算或评估 seed。
- 除被研究的唯一变量外，R-BC-3D、R-BC-0D、R-PPO-0D 必须使用相同的环境配置、
  路径生成器、初始状态分布、评估 seed、成功/超时定义和 checkpoint 选择规则。
- 不同预算或不同环境数的 run 只能作为诊断，不能混入主对比表，也不能声称是因果比较。
- 评估必须逐 seed 配对；报告全部 3 个训练 seed 和 50 个正式评估 seed 的结果，
  不得只报最好 seed、最好 checkpoint 或删除失败尾部。
- 3 个训练 seed × 50 个正式评估 seed；
- 每个训练 seed 使用正式预算 `4096 envs × 256 steps × 100 iterations`（约 104.86M
  环境步）和同一套早停规则；若提前早停，必须记录实际步数和早停原因；
- 至少比较 R-BC-3D、R-BC-0D、R-PPO-0D；
- 报告每个 seed 的配对结果、均值、95% 置信区间、最差 seed、失败尾部；
- 可选加入 V-3D 诊断控制，但必须标明它不是主结果；
- Torch 门禁通过后再考虑 CPU PVS 64-seed；
- 大型 checkpoint、rollout、event 只留服务器；Git 只保留代码、配置、小型 summary
  和报告。

## 停止条件

遇到以下情况立即停止并报告，不要扩大范围：

- 需要删除或重写旧配置、旧 checkpoint 或旧结果；
- 发现 command 通过 state、history、DADE 或路径块泄漏进 0D actor；
- BC-0D 在允许的补数据/DAgger 后仍过不了门禁；
- 零残差不等价；
- R-PPO-0D 的 timeout 或 safety 明显差于 BC-0D；
- 需要改变预注册指标或门槛才能得到正结果；
- 服务器和本地都没有足够资源完成当前阶段。

## 交付物

最终必须给出：

1. R-BC-0D 数据 manifest、checkpoint 路径和 SHA256；
2. R-BC-3D vs R-BC-0D 的 20-seed 门禁表；
3. R-Z0-0D 等价性证据；
4. R-PPO-0D screen 结果；
5. 通过或不通过正式 3×50 的明确结论；
6. 每个阶段的 seed、配置 SHA、训练预算和产物路径；
7. 更新后的执行状态文档和最终报告。

不要只完成阶段 1 就结束。只有在通过正式比较或触发停止条件时，才结束这个 goal。
```

