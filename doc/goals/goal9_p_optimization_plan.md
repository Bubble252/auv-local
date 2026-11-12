# Goal9 P 组优化与复核计划

更新时间：2026-11-05

状态：**本轮执行完成。X0/X1/X2/X3/X4lr/X4norm 均已产生结果；P1 路径分支仍为
`no_path_attribution`。旧 P0/P1/P2/P3 结果全部冻结，不覆盖、不重写。**

## 1. 为什么需要这个分支

用户的判断是：P1 可能不是路径信息本身无效，而是奖励信号、归一化、训练预算或优化器
设置不足。当前证据支持“需要复核”，但不支持直接降低门禁：

- P1 相对 A0 的 default success delta 为 `+0.05`，但 20 个评估 seed 中只有 seed
  `1005` 发生一次失败到成功的翻转；这不是稳定的多 seed 正收益。
- P1 的 default 3D RMSE 改善为 `0.4308 m`，但 `path-zero` 后只恶化
  `3.13e-05 m`，动作 L2 平均只变化 `4.70e-05`。
- 路径编码器末层按 `path_encoder_alpha=0.01` 进行缩放，且末层零初始化；从
  `model_0` 到 `final`，末层权重范数只从 `0` 增长到约 `0.012`。因此有效路径贡献
  约为 `0.01 × 0.012 = 1.2e-4`，路径支路事实上接近关闭。
- P1 训练使用 `actor_lr=1e-5`、前 5 个 iteration 的 actor learning rate 为 `0`
  （critic warmup）、1 个 learning epoch、128 个 mini-batch、32 个 iteration。
  实际 actor 只获得约 27 个 iteration 的更新机会。
- P1 的 `normalize_advantage_per_mini_batch=false` 并不代表没有优势归一化；在
  rsl_rl 中它会在 `compute_returns` 阶段执行全局 advantage normalization。因此不能
  把“没有归一化”当作既定事实，必须单独验证。

## 2. 目标与不可改变的约束

目标：判断 P1 的负路径归因是否由路径支路未激活、训练预算不足或优化设置造成；若可以
修复，产出一个同时满足性能门禁和路径归因门禁的 P1 配置；若不能，给出可复现的负结果。

不可改变的约束：

- 不覆盖 `artifacts/lowlevel/pc_abc_prework_20261105/` 下的任何历史 P 产物。
- 不修改旧 P1 的训练 seed、评估 seed、门禁阈值或 summary 文件。
- 不同时改变输入结构、奖励、训练预算和优化器；每个 arm 只改一个已声明变量。
- 不通过降低 `path-zero` 的 `0.05` success 或 `0.10 m` RMSE 门禁来“救”结果。
- 新增路径/DADE 分支必须是 additive 的；不得破坏 `pc_command_track` 原行为。
- 所有大训练和评估只在服务器执行；本地只跑单测、ruff 和 CPU smoke。

## 3. 已识别的候选原因

| 编号 | 候选原因 | 当前证据 | 优先级 |
|---|---|---|---|
| H1 | 路径支路有效贡献过小 | `alpha=0.01` × 末层范数约 `0.012`，路径零化动作差 `4.7e-5` | 最高 |
| H2 | actor 更新预算不足 | 5 个 warmup iteration 后只有约 27 次 actor 更新；KL 约 `5e-4` | 高 |
| H3 | PPO 奖励尺度/熵项/优化器设置不匹配 | Adam 对损失常数缩放不敏感，但 entropy、clip 和辅助损失会受影响 | 中 |
| H4 | 路径特征与 `c_t` 冗余 | `c_t` 已含 ALOS heading/pitch/speed 指令；路径特征可能只是重复信息 | 中 |
| H5 | 路径奖励不存在 | P1 使用 `pc_command_track`，没有 path reward；P3 才允许加路径惩罚 | 后续 |

注意：H5 不属于 P1 的原始授权范围。P1 只检验路径输入，不把路径奖励混入；若 H1–H3
修复后仍无路径归因，再讨论 P3。

## 4. 预注册实验臂

训练 seed 固定为 `91430`，评估 seed 固定为 `1000–1019`，环境固定为
`odarrl_control_pc_c3_random_realistic_v1_rewardscale001.json`，基线固定为 P0
`A0_stride1`。除 X2 的预算和 X3 的奖励尺度外，所有训练参数保持与 P1 相同。

### X0：旧 P1 诊断（不训练）

内容：

- 读取旧 P1 final checkpoint；
- 记录路径编码器末层权重范数、`alpha`、有效输出尺度；
- 记录 BC、`model_0`、`model_10`、`model_20`、`model_30`、`final` 的
  actor/path 参数变化；
- 记录路径零化动作差和路径特征统计。

用途：只做机制诊断，不产生新的科学判定。

执行结果（2026-11-05）：旧 P1 路径编码器末层范数从 `model_0` 的 `0` 增长到
`final` 的约 `0.0120`；`alpha=0.01` 时有效输出尺度约 `1.20e-4`。路径零化动作 L2
仅 `4.70e-5`，说明旧 P1 的路径支路事实上接近关闭。

### X1：P1-activation（最低优先改动）

唯一变量：`pc.path_encoder_alpha: 0.01 -> 1.0`，末层继续保持零初始化。

预算：`32 iterations × 128 envs × 256 steps`，训练 seed `91430`，评估
`1000–1019`，tags `default/strict125`，四种路径消融。

判定：

- 机制预检：路径零化动作 L2 应明显高于旧 P1 的 `4.7e-5`；报告中必须记录
  `path_encoder` 末层范数和有效输出尺度。
- 科学判定仍使用原 P1 门禁：primary signal + guardrails + `path-zero` 归因。
- X1 若通过，直接进入 X4 复核；若 primary 通过但归因不通过，进入 X2。

执行结果（2026-11-05）：

- 32 iterations、`1,048,576` 环境步；训练正常完成，所有 losses 有限。
- 路径末层有效输出尺度从 `0` 增长到约 `6.20e-3`，约为旧 P1 的 52 倍。
- default success delta `+0.05`、3D RMSE improvement `+0.4376 m`；guardrails 通过。
- `path-zero` success drop `0.0`、RMSE worsening `+0.00079 m`、动作 L2
  `1.65e-3`；RPM 差约 `1.29 rpm`，但闭环指标没有可检测变化。
- 判定：`no_path_attribution`。`alpha=1` 打开了通路，但 32 轮内路径对闭环行为仍
  没有达到原门禁的因果影响。

### X2：P1-budget（只在 X1 显示路径支路已激活时执行）

唯一变量：训练预算 `32 -> 128 iterations`，其余与 X1 相同。

用途：检验“actor 更新机会不足”这一假设。不能把 X2 的结果与旧 P1 的 32-iteration
预算混为一个实验；必须单独记录 `iterations` 和总环境步数。

判定：仍使用原 P1 门禁，不因增加预算而降低归因阈值。

执行结果（2026-11-05）：

- 128 iterations、`4,194,304` 环境步；训练正常完成。
- 路径末层有效输出尺度约 `1.27e-2`；路径零化动作 L2 `6.10e-3`，RPM 差约
  `7.12 rpm`，相对 X1 继续增大。
- default success delta `0.0`、strict125 success delta `0.0`、3D RMSE improvement
  `-0.0246 m`（略退步）；guardrails 通过。
- `path-zero` success drop `0.0`、RMSE worsening `-0.0085 m`。
- 判定：`no_signal`。4× 训练预算没有把路径影响转化为闭环收益，因此“仅训练步数
  不足”不是充分解释；不继续无上限加预算。

### X3：P1-reward-scale（只在 X1 显示路径支路已激活但优化仍弱时执行）

唯一变量：`reward.pc_ppo_scale: 0.01 -> 1.0`，其余与 X1 相同。

用途：验证“奖励尺度太弱”假设。由于当前代码默认做全局 advantage normalization，
该 arm 的结果若几乎不变，应明确记录为“奖励尺度不是主因”，不要继续扫权重。

执行配置：`configs/lowlevel/odarrl_control_ppo_pc_p1opt_alpha1_rewardscale1_dz64_gamma995_warmup5_critic3e-4.json`；
唯一训练变量为 `reward.pc_ppo_scale: 0.01 -> 1.0`。

### X4：优化器/裁剪诊断（最后执行）

只有在 X1–X3 仍显示路径支路有效但学习没有转化为闭环收益时，才允许单变量比较：

- `normalize_advantage_per_mini_batch=true`（如果确需验证归一化位置）；
- `clip_param` 从 `0.02` 改为 `0.2`；
- `actor_lr` 从 `1e-5` 改为 `1e-4`。

一次只跑一个变量，不把这些设置打包成“调参大礼包”。若没有机制证据，不启动 X4。

当前拆成两个独立 32-iteration 臂：

- `x4lr`：仅 `algorithm.learning_rate_actor: 1e-5 -> 1e-4`；critic LR 保持 `3e-4`。
- `x4norm`：仅 `algorithm.normalize_advantage_per_mini_batch: false -> true`。

`clip_param` 在该轮先不扫；X1/X2 的动作影响持续增大，但闭环收益没有出现，优先检验
actor 更新幅度和优势归一化位置，避免同时改多个优化器参数。

执行结果（2026-11-05）：

- X3：default/strict success delta 均为 `0.00`，3D RMSE 改善 `0.0308 m`，
  `path-zero` success drop `0.00`；奖励尺度不是主因。
- X4lr：default success `+0.10`、strict125 success `+0.25`，3D RMSE 分别约
  `+0.030 m`/`+0.595 m`；说明 actor LR `1e-4` 修复了真实的学习速度瓶颈。
  但 `path-zero` default success drop 仍为 `0.00`，路径归因不通过。原始 material
  saturation 增量 `+0.0868` 保留为风险；按用户最新取舍，不作为本轮硬否决。
- X4norm：与 X1 基本一致，default success `+0.05`、strict `0.00`，
  `path-zero` success drop `0.00`；逐 mini-batch 优势归一化不是主因。
- 综合报告：`doc/goals/goal9_p_optimization_report.md`。

## 5. 机制诊断指标

每个训练 iteration 和最终 checkpoint 至少记录：

- `path_encoder_final_weight_norm`；
- `path_encoder_alpha`；
- `path_encoder_effective_output_scale = alpha × final_weight_norm`；
- `path_encoder_final_weight_grad_norm`（若可得）；
- `actor.0.weight`、`actor.2.weight`、`actor.4.weight`、输出层权重范数；
- 对应参数相对上一层 checkpoint 的更新范数；
- 路径零化动作 L2 的 mean/max，以及路径几何/上下文分块的对应值。

这些指标是机制证据，不替代 success、strict125、3D RMSE、timeout 和 safety 的科学
门禁。material saturation 只作为执行器使用风险记录：超过 `0.02` 时标记为
`elevated`，但不参与 P 组的通过/否决。

## 6. 判定与停止规则

1. X0 必须完成并写入报告，再开始 X1。
2. X1 通过原 P1 门禁：P1-opt 成功；进入 X4 的多 seed 复核。
3. X1 primary 通过但 `path-zero` 不通过：执行 X2；不得直接判 P 失败。
4. X1 primary 不通过且路径零化动作差仍接近旧 P1：先检查机制诊断；若路径支路仍未
   激活，停止并记录负结果，不进入 X2/X3。
5. X2 通过：保留 128-iteration 配置作为 P1 候选，但仍需多 seed 复核。
6. X2 失败：不得继续加预算；报告“在已激活路径支路下，增加预算仍无因果归因”。
7. X3 若相对 X1 没有可解释变化：记录“奖励尺度不是主因”，停止奖励尺度扫描。
8. 任何负结果都保留 artifact；不重跑旧 P1，不替换旧 summary。

## 7. 多 seed 复核

只有在单 seed screen 同时满足 primary、operational guardrails（timeout/safety）和
path attribution 后，才启动：

- 训练 seeds：`91430`、`91431`、`91432`；
- 评估 seeds：`1000–1049`（50 个固定 seed）；
- 报告逐个训练 seed 的 success、strict125、3D RMSE、timeout、safety、作为风险信息的
  material saturation，以及 `path-zero` 的动作差和闭环退化。

多 seed 复核失败时，不能把单 seed 的正结果写成稳定收益；应保留为“单 seed 阳性、多
seed 不稳定”的负结果。

## 8. 产物布局

所有新产物写入：

```text
artifacts/lowlevel/pc_p1opt_20261105/
  X0_diagnosis/
  X1_alpha1_32/
  X2_alpha1_128/
  X3_alpha1_scale1_32/
  X4_optimizer/
  report/
    summary.json
    summary.md
    stage_decision.json
```

历史产物保持不变：

```text
artifacts/lowlevel/pc_abc_prework_20261105/P1_path_obs/
```

## 9. 执行清单

- [x] 完成 X0 checkpoint 诊断；
- [x] 为 X1 生成独立配置，只改 `path_encoder_alpha`；
- [x] 训练记录加入路径编码器范数、有效尺度和 actor 层更新诊断；
- [x] 本地单测、ruff、CPU smoke 通过；
- [x] 服务器 X1 screen（32 × 128 × 256，seed 91430，评估 1000–1019）；
- [x] 服务器 X2 screen（128 × 128 × 256，seed 91430，评估 1000–1019）；
- [x] 服务器 X4lr/X3/X4norm screen；
- [x] 根据 X1/X2/X4 机制/门禁结果写阶段结论；
- [ ] 写入 `summary.json`、`summary.md`、`stage_decision.json` 和 manifest 哈希；
- [ ] 更新 `goal9_execution_state.md`，但不覆盖旧 P1 报告。

## 10. Git 与提交约定

本分支只提交源码、配置、测试和文档，不提交 artifact、checkpoint 或日志。推荐按以下
顺序分小步提交，禁止 `git add -A`：

```bash
git add doc/goals/goal9_p_optimization_plan.md
git commit -m "docs(goal9): add P optimization protocol"

git add src/auv_lowlevel/batch/pc_policy.py src/auv_lowlevel/ppo_config.py
git add tests/test_lowlevel_pc_policy.py tests/test_lowlevel_ppo_config.py
git commit -m "feat(pc): make path branch activation diagnosable"

git add scripts/run_pc_p1_optimization_20261105.py scripts/summarize_pc_abc_p1.py
git add tests/test_pc_p1_optimization_runner.py
git commit -m "feat(goal9): add P optimization runner and diagnostics"
```

推送前先运行：

```bash
PYTHONPATH=src .venv/bin/pytest -q \
  tests/test_lowlevel_pc_policy.py \
  tests/test_lowlevel_ppo_config.py \
  tests/test_pc_abc_p1_runner.py \
  tests/test_pc_abc_p1_summary.py \
  tests/test_pc_p1_optimization_runner.py
.venv/bin/ruff check src/auv_lowlevel scripts tests
```

服务器只同步源码、配置、测试和脚本；训练输出使用服务器本地的
`artifacts/lowlevel/pc_p1opt_20261105/`。

