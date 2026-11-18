# Goal184：传统控制、BC 与 PPO 的 C5 配对诊断

日期：2026-11-18。范围：补充现有 C5 课程边界诊断；不是新 Goal，也不是完整正式评价。

## 主要结果

**BCRES 在本批 C5 场景中的路线误差最低；传统 teacher 的速度 RMSE、材料饱和率更低。BCFT 改善了初始 BC 的速度和饱和，但路线 3D RMSE 尚未改善。**

传统基线是使用特权状态及真值海流的 oracle teacher，不是与学习策略完全同等传感信息的可部署传统基线。PPO 使用训练 seed 86201、iteration 338 / 66,453,504 transitions 的 C5 checkpoint，不是当前最新 C6 权重。

| 指标 | 传统控制（oracle teacher） | 初始 BC | BCFT | BCRES |
|---|---:|---:|---:|---:|
| 路线 3D RMSE ↓，m | 2.5484 | 2.4951 | 2.5528 | 1.7752 |
| 横向 RMSE ↓，m | 1.5075 | 1.4321 | 1.3779 | 0.8150 |
| 垂向 RMSE ↓，m | 2.0547 | 2.0432 | 2.1489 | 1.5770 |
| pre-goal 3D RMSE ↓，m | 2.5585 | 2.4985 | 2.5899 | 1.8905 |
| 速度 RMSE ↓，m/s | 0.3964 | 0.8400 | 0.4492 | 0.4311 |
| 速度 MAE ↓，m/s | 0.3394 | 0.6718 | 0.3276 | 0.3439 |
| 材料饱和率 ↓ | 29.07% | 43.40% | 31.91% | 56.40% |
| completion ↑ | 89.70% | 91.11% | 90.36% | 89.89% |
| success（辅助）↑ | 85.00% | 30.00% | 62.50% | 60.00% |
| timeout ↓ | 0.00% | 0.00% | 0.00% | 0.00% |
| safety 终止 ↓ | 0.00% | 0.00% | 0.00% | 0.00% |
| 穿过终点平面但未捕获目标 ↓ | 15.00% | 70.00% | 37.50% | 40.00% |
| 地速模长 > 2.5 m/s 占比 ↓ | 0.00% | 20.96% | 4.56% | 3.73% |

成功数依次为 34/40、12/40、25/40、24/40；四组 timeout 与 safety 终止均为 0/40。其余终止是 goal-plane miss，因此 timeout=0 不等于全部成功。

## PPO 改善了什么

误差变化按（方法/基线−1）×100% 计算，负值表示改善；饱和率使用百分点差。

| 指标 | BCFT 相对 BC | BCRES 相对 BC | BCFT 相对 teacher | BCRES 相对 teacher |
|---|---:|---:|---:|---:|
| 路线 3D RMSE | +2.31% | -28.85% | +0.17% | -30.34% |
| 横向 RMSE | -3.79% | -43.09% | -8.60% | -45.94% |
| 垂向 RMSE | +5.18% | -22.81% | +4.59% | -23.25% |
| 速度 RMSE | -46.52% | -48.68% | +13.34% | +8.76% |
| 速度 MAE | -51.24% | -48.81% | -3.47% | +1.34% |
| 材料饱和率 | -11.49 个百分点 | +13.01 个百分点 | +2.84 个百分点 | +27.33 个百分点 |

- **BCRES 的路径收益体现在这批配对样本中**：相对初始 BC，33/40 个 seed 的 3D RMSE 更低；相对 teacher，36/40 个更低。它同时带来更多材料速率裁剪，不能据此称为全面优于传统控制。
- **BCFT 的主要收益是速度与执行器负担**：相对初始 BC，速度 RMSE 降低 46.52%，材料饱和率下降 11.49 个百分点；3D RMSE 增加 2.31%。与 teacher 的 3D RMSE 数值接近（+0.17%），不据此排名或宣称等效。
- **传统 teacher 仍有优势**：速度 RMSE、材料饱和率和 success 优于两条 PPO；但 BCFT 的速度 MAE 略低于 teacher，不能说 teacher 在所有速度指标上最优。success 只是辅助指标。
- 两条 PPO 的 completion 都略低于初始 BC。保留完整路线、执行器负担和安全的联合评价，本报告不改变既定训练方案，也不选择正式赢家。

## 比较口径与限制

1. 主表取同一 PPO evaluator 的 summary.metrics。路线 3D/横向/垂向 RMSE 按 episode 等权 MSE 再开方，不混入课程 gate 的 episode RMSE 均值。pre-goal 保留 evaluator 的原有聚合定义，与整条路线指标单列。
2. 速度指标是标量 command speed 与机体前向地速之差，不是 reward 使用的路径速度向量投影目标，也不是地速模长误差。这里 C5 沿用已有 evaluator 的默认采样范围 1.1–1.4 m/s，本批实际约 1.1001–1.3982 m/s，逐 episode 恒定；不能外推到训练中的 1.0–2.2 m/s 分段换速协议。
3. 材料饱和率是执行器 pipeline 的材料裁剪事件率，主要反映两片舵的速率裁剪，不是舵角处于 ±20° 的时间比例，也不是 RPM 满转速率。teacher 内部还包含限幅、限速和积分抗饱和；表中比较的是完整控制链进入同一执行器 pipeline 后的指标，不是所有控制器裁剪前内部请求的统一计数。
4. 场景参数和初态对齐不意味着轨迹或终止时间相同；指标取各自终止前轨迹，应结合 completion 阅读。传感器模型一致，但 teacher 使用真值姿态、角速度和海流等额外信息，信息权限不同。
5. 初始 BC 使用 BCRES 的 model_initial.pt、frozen-base 和初始归一化状态；PPO 使用各自训练后状态。此前已核验初始及 C5 BCRES 的 24 个 frozen_bc 字段逐元素一致。因此对比覆盖整个训练栈，不能将变化全部归因于 actor 权重。
6. 只有一个训练 seed、40 个配对评价 seeds，未计算置信区间或显著性检验，不作统计显著性结论。不使用 held-out，也不宣告 Goal184 完成。

## 背景与执行契约

用户要求在初始 BC、BCFT、BCRES 对照中加入传统控制基线。复用已有两条 PPO 的 C5_066453504 checkpoint 评价和初始 BC 评价，只补传统控制。

技术栈：Batch Torch REMUS100、现有 evaluate_pc_ppo evaluator、CPU float64、40 episodes（seeds 20–59）、C5、nominal start、endpoint-stop、absolute 执行器接口、每仿真步更新控制。传统控制定义为项目现有 BatchedPCTeacher：ALOS 制导 + 真值海流补偿 + REMUS PD/PI；它使用特权状态，应标为 oracle teacher。

evaluate_pc_bc 默认采用训练速度协议及随机初态，与已有 PPO 评价不一致。其 diagnostic_teacher_c5_20261118 输出保留，但不进入主比较。对齐方案是在独立诊断进程中接入既有 teacher.control()，复用 PPO evaluator 的场景采样、初态、终止、trace 和指标聚合；不修改仓库源码或运行中的训练。

- [x] 读取已有 BC/PPO 产物及两种 evaluator，明确协议差异；Git：只读。
- [x] 补充传统控制的协议对齐评价。启动前复用 evaluator 的配置、checkpoint、归一化和初态检查；结束运行 audit_goal182_artifacts.py，核对40个seed逐episode课程参数及指标口径。原始结果保留服务器独立 eval 目录，不提交。
- [x] 汇总四组结果、相对变化及限制，核对文档。Git：本次不 commit/push；如用户后续要求，只显式 add 本文，运行 git diff --cached --check 后 commit/push。

正式协议完整性：本次只有一个 C5 cell、一个训练 seed，未覆盖 arms × levels × radii × zero baselines。若用作 formal 结论，应标记 INVALID_INCOMPLETE_EVAL；本报告仅作为诊断，不判定正式 PASS/FAIL 或训练赢家。

## 实施与核验结果

最初 evaluate_pc_bc 产物中的 teacher success=95% 来自不一致的速度/初态协议，保留在原目录，**未进入主表**。

对齐时用独立适配脚本捕获 PPO evaluator 创建的环境，将策略 act_inference 接到现有 pc_action_from_actuator(teacher.control())，在 rollout 前 reset teacher。采样、初态、终止、动作 pipeline、trace 和聚合沿用 PPO evaluator；teacher 动作不经过神经 actor。适配只作用于独立 CPU 进程，没有修改仓库源码、配置或运行中的训练。

teacher 产物中的 action_mode=policy、checkpoint 和 trainable_scope 来自通用 evaluator 外壳。checkpoint 仅用于已有 schema/layout/归一化检查，不能据此把 teacher 误认为 BCFT。实际控制器身份记录在 controller_provenance.json；teacher 的 residual/base 策略诊断字段不作科学解释，主表仅用环境级指标。

- 四组分别运行 scripts/audit_goal182_artifacts.py --run-dir <目录> --expected-episodes 40，均返回 Goal182/Goal184 artifact audit OK。这是产物有效性检查，不是正式性能 Gate PASS。
- 已逐组核对 seeds、环境路径、C5、nominal start、endpoint-stop、decision stride、absolute 接口、sensor profile、plant randomization、ablation 和 success frame 相同。
- 40 个 episode 的 curriculum 字典、初始速度、初始 RPM 和 plant scale factors 逐项完全一致。
- 数值从四份 summary 提取并计算相对变化，汇总在 comparison.json。没有为本次比较重新选择 PPO checkpoint、调参或修改训练。

## 证据与复现入口

服务器运行根目录 R：

/data/yhx/underwater-robotics/runs/auv-local/goal184/20261118-goal184-64d8225

服务器源码根目录 W：

/data/yhx/underwater-robotics/worktrees/auv-local/goal184/20261118-goal184-64d8225

| 方法 | R 下产物目录（均有 summary.json、episodes.jsonl、trace.npz） |
|---|---|
| 传统 teacher（对齐后） | eval/diagnostic_teacher_matched_c5_20261118 |
| 初始 BC | eval/diagnostic_bc_initial_c5_20261118 |
| BCFT | full_wave1/pe2e_bcft_seed86201/curriculum_evals/C5_066453504/default |
| BCRES | full_wave1/pe2e_bcres_seed86201/curriculum_evals/C5_066453504/default |

对齐 teacher 目录还保存 controller_provenance.json、comparison.json。适配脚本为 R/eval/goal184_teacher_matched.py，SHA256：2d76944a4cbe79cafe66f4d3ead08c90cd322150a445fd6a26f2438dca729ed9。

运行使用服务器既有 Python：/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python；OMP_NUM_THREADS=1、MKL_NUM_THREADS=1；PYTHONPATH 包含 W/src 和既有 auv-core 的 /data/yhx/underwater-robotics/worktrees/auv-core/g04-sync/20261114-g04-sync-0e0bfb1/src。该路径复用已有服务器环境，仅服务本次诊断。

在 W 下执行适配脚本，参数如下；复现时改用新的 output 目录，避免覆盖：

    python "$R/eval/goal184_teacher_matched.py" \
      --training-config configs/lowlevel/goal184/train_formal_mod4_pe2e_bcft_env_empirical_5d_m_seed86201.json \
      --checkpoint "$R/full_wave1/pe2e_bcft_seed86201/model_initial.pt" \
      --output "$R/eval/<新的复现目录>" \
      --seed-start 20 --seed-count 40 --device cpu --dtype float64 \
      --curriculum-stage C5 --action-mode policy
    python scripts/audit_goal182_artifacts.py \
      --run-dir "$R/eval/<新的复现目录>" --expected-episodes 40

环境配置 SHA256：db4de5ff1ed1d5cb9b4b37da6d04df95c82855d61d609cc6e0e5d61512cc9729。

| checkpoint 身份 | SHA256 |
|---|---|
| Teacher（仅外壳检查） | 7e16a8cf2f30d2b9eb73a0ebcc82e37500a1e5ead5f7bc35e469d803de383684 |
| 初始 BC | ffca319842ce21c4490743912c34e3d4f39b7d70a9a5059fbe1a0dddc481d55f |
| BCFT C5 | 41b9e1e682123ada7745cd9a5fd59c919ba18577c721d81b6d93dc64d38faeac |
| BCRES C5 | c7d10704e1d6b0dccb70a910a25e34756ca4a3aca4535073faeb99126443e4f6 |

相关本地入口：

- [此前饱和、速度及 BC/PPO 分析](goal184_rudder_saturation_and_ppo_recipe_review_20261118.md)
- [Goal184 主计划](../goals/goal184_empirical_full_arm_training_20261118.md)
- [环境配置](../../configs/lowlevel/goal184/env_path_fixed_speed_formal_empirical.json)
- [teacher 实现](../../src/auv_lowlevel/batch/pc_residual_env/teacher.py)
- [PPO evaluator](../../src/auv_lowlevel/cli/evaluate_pc_ppo/runner.py)
