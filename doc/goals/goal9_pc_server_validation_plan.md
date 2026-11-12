# PC 实验服务器验证执行计划

记录日期：2026-11-03

更新日期：2026-11-06（S15 已完成并冻结；S16 最终改为 V-0D 纯路径主臂，V-3D 只作
可选诊断控制。旧的 ALOS-free V0/V1/V1b/V2/V3 详细方案保留为历史，不再作为当前执行
入口。P0/P1/P2/P3 只做前置筛查，不替代正式 ABC；V 默认不做 speed head）

关联任务：`doc/goals/goal9_odarrl_control_reproduction.md` 的 PC 纯控制器臂。

本文只约束“把本地 C0–C4 证据迁移到 8×4090 服务器并继续提升模型能力”的执行过程，不改变已经冻结的 PC 观测、动作、奖励和门禁口径。所有新结果必须保留本地基线，不覆盖旧 checkpoint、旧 summary 或失败记录。

> **2026-11-06 优先级更新：** S15 已完成并冻结唯一基线；S16 不再先走
> `V-3D -> V-2D -> V-1D -> V-0D`，而是直接执行 V-0D 纯路径主臂。V-0D 的关键前置是
> 新增同契约的 `R-BC-0D`（`state + 8D path -> action`，`command_dim=0`），不能用现有
> 3D BC checkpoint 加 `command_mask` 冒充。权威计划见
> `doc/goals/goal9_v0d_direct_pure_path_plan_20261106.md`。

## 1. 项目背景

本地已经完成 PC 契约、20° 执行器包线、合成传感器、C0/C1 命令跟踪、C2 ALOS 迁移、C3 随机路径 realistic 环境、PD/PI 教师与 BC 热启动，以及 `d_z=64`、`gamma=0.995`、critic warmup、reward scale 和 no-DADE/DADE 短程配对。当前主要缺口不是再补一个本地 smoke，而是：

- 在真实服务器 GPU 环境复现本地 C 系列结果，排除本机 CPU/内存和依赖差异；
- 用多个训练 seed 和固定评估 seed 集完成正式比较；
- 补齐 C4 强扰动和 H1–H4 高速课程；
- 让 DADE 从“当前帧残差映射”变成真正可利用历史信息的通路，并用消融证明；
- 形成可提交、可复核的服务器证据与最终报告。

## 2. 技术栈与资源

- 服务器：`yhx@152.136.183.51:43127`，8×RTX 4090，单卡 24 GiB；
- 代码目录：`/data/yhx/underwater-robotics/src/plan_p1/settled_局部规划器_反应式`；
- Python：`/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python`；
- 关键依赖：PyTorch 2.7、NumPy 1.26、rsl_rl、pytest、ruff；
- `oceansim` 环境缺少 `rsl_rl`，不能用于 PPO 训练；只把 IsaacLab venv 作为本轮服务器训练解释器；
- 首选 GPU：先使用当前空闲卡做 smoke 与短训练；长训练按实际显存占用分配独立 GPU；
- 数据：本地 `pc_bc_demos_c0_c1_c2_c3random_v1_20261103/demos.npz` 与对应 BC checkpoint 作为服务器复现输入，传输后校验 SHA-256；
- 输出：服务器 `artifacts/lowlevel/pc_server_*` 新目录，禁止覆盖已有目录；大型 checkpoint 和 rollout 不直接进入普通 Git。

## 3. 本轮完成标准

- [ ] 服务器代码与本地 PC 代码哈希一致，PC 单测全部通过；
- [ ] C0/C1 的命令跟踪与 BC-only 门禁在服务器复现，数值与本地同量级；
- [ ] C2 clean/mild 的 no-DADE/DADE 配对复现，记录成功率和三维 RMSE；
- [ ] C3 随机路径 realistic 在服务器完成至少 3 个训练 seed 的 no-DADE/DADE 配对；
- [ ] 固定评估集 `0–49`、`50–99` 和 `2.0 m`/`1.25 m` 两档均报告；
- [ ] C4 强时变、强剪切、延迟/失锁/野值压力集完成基准评估；
- [ ] H1–H3 高速课程完成可达到性标定和 smoke，H4 只做压力评估；
- [x] DADE 完成“显式历史依赖”实验臂并报告消融；三 seed 的 `current-only/history-zero` 动作敏感度仍低于 `0.1%` 门禁，判定为负结果；
- [ ] DADE-pose 全量 multienv 补齐 5 个训练 seed 并在固定评估种子上重新跑历史机制门禁；CPU `94030` 与 GPU `94031/94032` 已启动，监督队列将追加 `94033/94034`、恢复主队列并自动评测；
- [ ] 完成 `128×256×32` 到 `4096×256×40` 的规模放大基准，记录显存、吞吐和单位训练成本；
- [ ] 至少一个主实验臂完成 `>=3` 个训练 seed 的扩大规模配对，之后才允许延长训练；
- [x] S15-P0 按预注册顺序完成 smoke、首轮筛查和判定，结论为 no attributable signal；
- [x] S15-P1 完成 smoke、正式 screen 和 path-zero 消融，结论为 `no_path_attribution`，
      路径分支停止；
- [x] S15-P2/P3 已按预注册顺序完成判定：P2=`no_signal`，保留 raw-history；
      P3=`not_applicable`，前置证据不成立；
- [x] S15-A/B/C 已完成并冻结：A0_stride1 为唯一基线，B10=`pass_non_inferior`
      （效率候选），B5=`fail_non_inferior`，C=`C_not_triggered`；
- [x] 旧 S16-V0 已按历史门禁执行并判定 `fail_v0`；旧 V1/V1b/V2/V3 未启动；
- [ ] S16-V0D 主臂尚未实现或训练；当前只完成规划与预注册；
- [x] 已生成 `doc/goals/goal9_final_report.md`，区分正结果、负结果和未完成项；
- [ ] 按文件粒度提交，不使用 `git add -A`。

当前最终状态（2026-11-05）：

- [x] S15 全部可执行阶段结束，唯一冻结基线为 `A0_stride1`；
- [x] S16-V0 完成 1 个训练 seed × 20 个评估 seed × 3 tags 的正式 screen，
      结果 `fail_v0`；V 分支按预注册规则停止；
- [x] 未达到正式 3×32 + CPU PVS 64-seed，因为没有 V 候选通过前置 screen；
- [ ] Git 提交/推送未自动执行；这是全局约束要求，等待用户确认后按文件粒度处理。

当前 V-0D 状态（2026-11-06）：

- [x] 用户确认直接以 V-0D 纯路径为主臂，V-3D 仅作可选诊断；
- [x] 已记录 3D BC + `command_mask` 不能冒充 0D 的技术边界；
- [ ] 0D BC 数据/训练器扩展；
- [ ] `R-BC-0D`、`R-Z0-0D`、`R-PPO-0D`；
- [ ] 可选 V-3D 控制；
- [ ] 3×50 + CPU PVS 64-seed。

历史快照（2026-11-03）：

- [x] 本地与服务器 PC/PPO 配置测试均通过；当前为 `100 passed`，ruff 通过；
- [x] 修复 `cuda` 与 `cuda:0` 设备比较错误，并补回归测试；
- [x] C0 BC-only 64-seed：hold success `1.0`、safety `0`；
- [x] C1 BC-only 64-seed：hold success `0.984375`、final hold `0.6603`；
- [x] C3 `ck0` 64-seed 服务器复现：success `0.703125`，三维 RMSE `5.1656209 m`；
- [x] C3 五个训练 seed 的 no-DADE/DADE 配对与双半径评测完成，共 `100/100` 个 checkpoint summary；
- [x] C4 压力集完成 5 seed × 2 臂配对，全部为 C3 checkpoint 的显式压力迁移（`stage_override=true`）；
- [x] DADE-head 与 DADE-head+dropout 短程训练完成；`ck31` 双半径评测完成；
- [x] 32-seed 历史机制诊断完成：普通 DADE 与 DADE-head 的动作历史敏感度都低于基线动作的 `0.1%`；
- [x] 新增隔离式 DADE-head：`c_hat` 只经 action head 进入动作，并支持独立 head 学习率；默认旧配置行为不变；
- [x] 隔离式 DADE-head 短程机制门禁完成：`history-zero` 动作敏感度仅约基线动作的 `0.045%`，默认成功率持平、严格半径成功率下降 `0.015625`，判定为负结果，不扩到 B/C 级；
- [x] 修复冻结残差 optimizer 分组缺陷：没有显式 `learning_rate_dade_head` 时，`actor_dade_head` 参数此前未进入任何 optimizer group；已加入 actor group，并补 optimizer 参数计数审计与回归测试。受影响 DADE-head 旧结果保留但作废，不能用于负结论；
- [x] `4096×256` 扩大规模 A 级基准与 GPU 级 `flock` 调度修复完成；
- [x] 评测脚本新增 H0–H4 与 C4 tag 路由；H1/C4 的 64-seed smoke 均正常产出，C4 正确记录 `stage_override=true`；
- [x] B 级四臂正式扩训曾观察到 `10/12` 完成：no-DADE、DADE、DADE-head 三 seed 均完成；`dade_head_dropout05` 的 `seed91430` 完成，`seed91431/91432` 因 GPU6 显存不足被标记为 `blocked_gpu`。其中所有未显式设置 head 学习率的 DADE-head run 已因后续 optimizer 分组缺陷作废，保留目录但不得进入结论；
- [x] B 级缺失的 dropout05 两 seed 已保留旧 `run_status.blocked_gpu.json`，并由 post-queue gapfill 队列重跑；旧失败记录不覆盖；
- [x] 新隔离 head 的短程机制门禁：未通过，不进入 B 级三 seed。
- [x] C 级规模对照：旧 watcher 只做 `sleep`，已停止；no-DADE/DADE 的 `4096×256×100` 三 seed 任务已排入统一扩训队列。
- [x] B 级扩到 5 seed：`seed91433/91434` 的四臂共 8 个训练任务已排在统一队列中；
- [x] `DADE-head + current-frame dropout=1.0` 机制臂已拆成独立 `dade_head_sidechannel10` 配置，避免与旧 `full` dropout 结果混组；三 seed 任务已排入统一队列。
- [x] 新增冻结 BC + 残差策略路径与配置文件：`base_policy.mode=frozen_pc_bc_residual`，独立冻结 BC actor 不进入 optimizer，残差输出零初始化；本地 1-iteration CPU smoke 和闭环评测通过。
- [x] 冻结 BC 残差四臂 × 三 seed 的 `4096×256×100` 训练与固定 seed 评测已排入统一扩训队列；服务器 GPU smoke 已确认冻结审计为 `drift=0`；结果尚未产生，不能提前写正结论。
- [x] 补齐 C2 ALOS 基础复现训练：`c2_nodade/c2_dade × 3 seed × 2048×256×32`，从同一 C0/C1/C2 BC 热启动；训练后自动评测 mild/clean 和 `initial/0/10/20/30/final`。
- [x] 补齐 H3 `2.25–2.4 m/s` 冻结 BC 残差基线：`4096×256×100 × 3 seed`，训练后自动评测 H3 分布内、默认/严格半径和 H4 压力迁移。
- [x] 补齐 teacher/BC-only 基础基线：C2 mild/clean、C3 realistic、H1–H4 与 C4 teacher，以及 C2/C3/H3 BC-only；评测结果独立保存，不使用旧目录覆盖。

## 4. 执行步骤

### S0：冻结本地证据

- [x] 读取 goal9 已有结论，确认 C0–C3、DADE 负结果和 PC-G3 配方；
- [x] 记录本地关键文件 SHA-256：PC 源码、配置、demo 数据集、BC checkpoint；
- [x] 备份服务器即将被覆盖的代码目录；
- [x] 写出服务器同步清单，只传源码、配置、测试和必要数据，不传整个 artifacts。

已记录的输入哈希：

- demo 数据集：`eeac3e63266a1a96451d913f23c5b8df5ee53f5db68a6efbf94367ddedda3ded`；
- `dz64` BC checkpoint：`f56c359648a99e3fc8fde51f2e529af5c9916371450ff0d7a942c0f620d33936`；
- 远端备份：`.server_backup/pc_presync_20261103/code_configs_tests.tar.gz`、`.server_backup/i_bc_smoke_02.tar.gz`、`.server_backup/pc_bc_dz64_20261103.tar.gz`。

Git 记录命令：

```bash
git status --short
git add doc/goals/goal9_pc_server_validation_plan.md
git commit -m "docs(goal9): add PC server validation plan"
git push origin "$(git branch --show-current)"
```

### S1：同步与服务器自检

- [x] 用 `rsync` 同步 `configs/lowlevel`、`src/auv_lowlevel`、`scripts`、`tests` 和关键文档；
- [x] 在服务器校验源码与数据 SHA-256；
- [x] 运行 PC 单测，要求没有 import、schema、动作包线或 checkpoint 失败；
- [x] 做 1 iteration CUDA smoke，记录 GPU 型号、显存、每步耗时和输出 manifest；
- [x] 确认 CUDA 训练结果与本地 CPU smoke 在相同 seed 下没有契约差异。

服务器命令模板：

```bash
PY=/data/yhx/underwater-robotics/envs/conda_envs/oceansim/bin/python
export PYTHONPATH="$PWD/src:$PWD/../settled_全局规划器/src"
export CUDA_VISIBLE_DEVICES=2
$PY -m pytest -q tests/test_lowlevel_pc_*.py
$PY scripts/train_batch_lowlevel_ppo.py --training-config <pc_config> --device cuda --iterations 1 --num-envs 8 --num-steps 16 --output <server_smoke_dir>
```

### S2：C0/C1 复现

- [x] 复现 C0 clean 的 PD/PI teacher 与 BC-only 64-seed 评估；
- [x] 复现 C1 mild 的命令阶跃/斜坡跟踪，报告保持成功率、稳态误差、超调和饱和；
- [ ] 若 BC-only 出现闭环漂移，先做一轮 PC 专用 DAgger，不改主网络和奖励；
- [ ] 用服务器 BC 产物做 PPO warm-start，确认 `ck0` 与 BC-only 指标一致。

通过条件：clean 命令保持成功率 `>=0.95`，稳态误差和超调不超过 teacher 的 `1.2×`；mild 不出现安全终止异常或非有限值。

### S3：C2 ALOS 复现

- [ ] 运行 C2 clean 与 mild 的 teacher、BC-only、no-DADE、DADE；
- [ ] 固定同训练 seed、同 BC、同 env budget，先做 32 iteration 配对；
- [ ] 报告 clean 与 mild 的成功率、三维 RMSE、yaw/speed 跟踪、饱和率；
- [ ] 若 DADE 只与 no-DADE 重合，标记为“无历史收益”，不延长训练掩盖问题。

通过条件：C2 clean 成功率 `>=0.95` 且 RMSE 不超过同包线 teacher 的 `1.2×`；realistic 相对 clean 成功率下降不超过 `0.05`，RMSE 退化不超过 `0.20 m`。

### S4：C3 正式随机路径与多 seed

- [ ] 固定 canonical 配方：`d_z=64`、actor lr `1e-5`、critic lr `3e-4`、`gamma=0.995`、warmup `5`、`pc_ppo_scale=0.01`、`128×256×32`；
- [ ] 训练 seed 至少 `91430/91431/91432`，no-DADE 与 DADE 严格配对；
- [ ] 对 scheduled checkpoint 做 `0–49` 与 `50–99` 固定评估；
- [ ] 同时报告默认 `2.0 m` 与严格 `1.25 m`；
- [ ] 选 checkpoint 的依据只使用训练 seed，不使用评估 seed 反向挑选；
- [ ] 若稳定性和性能冲突，优先保留稳定、无安全退化的 checkpoint。

### S5：C4 与高速课程

- [x] 在 20° 包线上完成 H0 速度可达性、舵效和回转半径标定；
- [ ] 按 H1→H2→H3 逐级 smoke，每级先做命令可达性过滤；
- [ ] H4 `2.5 m/s` 只做固定压力评估，不直接进入长训练；
- [ ] C4 评估强时变流、强剪切、DVL 失锁、USBL 野值和通信延迟；
- [ ] 报告失败尾部、timeout、最大滚转、滚转率和 material saturation；
- [ ] 若高速组主要因 ALOS 地速命令或执行器权限失败，记录为系统边界，不归罪于 DADE。

已有 8-seed teacher 标定结果（`artifacts/lowlevel/calibration_h_speed_20261103/`）：

| 阶段 | 命令速度范围 | 成功率 | 三维 RMSE | speed RMSE | saturation | safety |
|---|---|---:|---:|---:|---:|---:|
| H0 | `1.0–1.8 m/s` | `0.875` | `2.260 m` | `0.422 m/s` | `0.116` | `0` |
| H1 | `1.8–2.0 m/s` | `1.000` | `1.695 m` | `0.432 m/s` | `0.472` | `0` |
| H2 | `2.0–2.25 m/s` | `1.000` | `1.765 m` | `0.335 m/s` | `0.480` | `0` |
| H3 | `2.25–2.4 m/s` | `1.000` | `1.993 m` | `0.282 m/s` | `0.493` | `0` |
| H4 | `2.5 m/s` | `1.000` | `2.188 m` | `0.306 m/s` | `0.510` | `0` |

H1–H3 的 teacher 命令可达性通过，但 saturation 已接近 `0.5`，因此不能只报成功率；后续 PPO smoke 必须同时检查执行器余量和高速下的 material saturation。H4 仅作压力评估。

### S6：让 DADE 真正使用历史

- [x] 复现 `current-only`、`current-zero`、`history-zero`、`history-reverse` 消融；history-required 三 seed 的 `current-only` 约 `0.004%`、`history-zero` 约 `0.06–0.08%`，均未过 `0.1%` 门禁；
- [ ] 对当前 DADE 做法记录首层输入列范数和 `chat` 对动作的贡献；
- [ ] 只选一个主实验臂，不并行堆叠多个未验证技巧；候选优先级如下：
  1. 显式历史门控或历史注意力正则，使 DADE 对历史变化产生可测动作影响；
  2. 100% current-frame dropout 训练，验证历史-only 是否可行；
  3. 海流估计预测辅助目标，并把 DADE 输出接入控制路径而非旁路；
- [ ] 每个候选都必须与 no-DADE、普通 DADE 同 seed、同预算配对；
- [ ] 只有同时满足“性能不降、历史消融可测、跨 seed 稳定”才作为正结果。

已落地的候选 0：`pc.dade_action_head_enabled=true`。

- `actor_dade_head` 直接读取 `c_hat` 并生成动作修正，输出层零初始化，保证 `ck0` 与 BC 行为一致；
- 该路径不依赖顶层 actor 的弱拼接列，给历史特征提供独立的动作梯度通路；
- 另有 `current_frame_dropout_prob=0.5` 实验臂，迫使 DADE 在训练时不能只复述当前帧；
- 普通与隔离式 action head 均已在服务器短程验证；隔离后 `history-zero` 动作敏感度仍仅约 `0.045%`，登记为负结果，不得据此启动 C 级。
- 新增候选 2：`history.current_frame_dropout_prob=1.0`。必须区分两种语义：
  - `current_frame_dropout_mode="full"`：同时清零顶层 actor state、最新历史行，并在 reset 时清零整个窗口；这是旧行为，会连当前控制状态一起隐藏；
  - `current_frame_dropout_mode="side_channel"`：保留顶层 state、命令和较早历史，只隐藏最新历史行的 12:15 维海流侧通道；reset 时隐藏窗口内的海流侧通道。该模式用于检验“策略能否从较早历史推断海流”，不会把当前姿态/速度一起抹掉。
- 新配置使用 `side_channel` 语义，已通过本地 1-iteration CPU smoke；正式三 seed 训练必须在服务器同步代码后重新排队，不能把旧 `full` dropout 的 checkpoint 和新语义混为一组。该臂仍必须通过 `current-only/history-zero` 闭环和 32-seed 动作敏感度门禁，不能仅凭训练 loss 判定成功。

### S7：扩大训练规模

本轮不只复现短程 `128×256×32 = 1,048,576` env-steps，而是分三级放大。所有级别必须同环境、同 reward、同评估 seed 集；只有环境数和总 env-steps 随级别变化。

- [x] **A 级：规模基准。** 在单卡跑 `4096×256×2` smoke，记录峰值 allocated/reserved 显存、collection 吞吐、端到端吞吐、step p50/p95 和首次编译时间；若 OOM，依次测 `2048×256`、`1024×256`，不能静默降级。
- [ ] **B 级：正式放大。** 对 no-DADE、普通 DADE、DADE-head、DADE-head+dropout 四个臂使用 `4096×256×40 = 41,943,040` env-steps；先完成至少 3 个训练 seed，再决定是否补到 5 个。
- [ ] **B 级扩到 5 seed（已排队）。** 在 C 级之后补 `seed91433/91434` 的四臂训练，共新增 8 个 `4096×256×40` run；输出根为 `artifacts/lowlevel/pc_server_scale_B_ext_20261103/`，不覆盖原三 seed 结果。
- [ ] **C 级：候选延长。** 只有 B 级中“成功率不降、严格 `1.25 m` 不降、无 safety 退化、历史消融可测”的臂，才允许延长到 `4096×256×100 = 104,857,600` env-steps；最多保留两个臂，避免把预算摊薄。
- [ ] **已排队的 C 级规模对照。** 无论 DADE 机制门禁是否通过，都对 no-DADE 与普通 DADE 做同 seed、同预算的 `4096×256×100` 对照，用来回答“延长训练是否改变规模效应”；如果 DADE 仍未通过历史机制门禁，这组只登记为规模对照，不作为 DADE 正结果。
- [ ] **dropout=1.0 机制队列（已排队）。** 使用 `dade_head_dropout10` 配置训练 `seed91430/91431/91432`，输出根为 `artifacts/lowlevel/pc_server_scale_B_dropout10_20261103/`；三 seed 完成后必须与普通 DADE-head、no-DADE 做同 checkpoint 双半径及历史消融比较。
- [ ] 每个 B/C 级训练都保存 `training_manifest.json`、`iteration_records.jsonl` 和固定 checkpoint；不许只看 `final.pt`，候选按训练 seed 内的预定 checkpoint 规则选择。
- [ ] 规模对照至少报告：默认 `2.0 m` 成功率、严格 `1.25 m` 成功率、三维 RMSE、timeout、safety、material saturation、speed/yaw 跟踪和训练吞吐。
- [ ] 结果按“同训练 seed 的 no-DADE/DADE 差值”汇总，报告均值、样本标准差、最小/最大值和失败 seed；禁止用评估 seed 反向挑选 checkpoint。
- [ ] 大规模 checkpoint 和完整 rollout 留在服务器，只把小型 `summary.json`、`summary.md`、manifest 和哈希同步回本地。

B 级 checkpoint 选择规则在完整结果产生前固定如下：

- 所有候选都报告 `ck0/10/20/30/39` 的完整默认、严格半径曲线，不只报选中点；
- checkpoint 只在同一个 arm 的多个训练 seed 上聚合选择，绝不使用评估 seed 反向挑选；
- 先过滤 `safety_termination_rate > 0` 或 material saturation 相对 `ck0` 增加超过 `0.01` 的点；
- 剩余点按“跨训练 seed 的默认成功率最小值、严格 `1.25 m` 成功率最小值、双半径成功率均值、双半径三维 RMSE 均值”依次排序，前三项越大越好、最后一项越小越好；
- 选中后所有训练 seed 使用同一个 checkpoint 进入 C 级或最终报告，并补充 `49/50–99/100` 等留出 seeds 复核；
- 如果没有任何点同时满足稳定性和机制门禁，则停止该臂的 C 级扩展，不用延长训练掩盖负结果。

建议的服务器命令模板：

```bash
PY=/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python
export PYTHONPATH="$PWD/src:$PWD/../settled_全局规划器/src"

# A 级：先测吞吐和显存，不进入正式结果
CUDA_VISIBLE_DEVICES=2 "$PY" scripts/train_batch_lowlevel_ppo.py \
  --training-config configs/lowlevel/odarrl_control_ppo_pc_c3_random_realistic_nodade_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json \
  --device cuda --iterations 2 --num-envs 4096 --num-steps 256 \
  --num-mini-batches 128 --seed 91430 \
  --output artifacts/lowlevel/pc_server_scale_smoke_4096_20261103 \
  --warmstart-actor artifacts/lowlevel/pc_bc_c0_c1_c2_c3random_v1_dz64_20261103/pc_bc_actor.pt \
  --record-only

# B 级：每个 arm/seed 单独目录；GPU 按 nvidia-smi 实测再分配
CUDA_VISIBLE_DEVICES=3 "$PY" scripts/train_batch_lowlevel_ppo.py \
  --training-config <arm_config> --device cuda --iterations 40 \
  --num-envs 4096 --num-steps 256 --num-mini-batches 128 \
  --seed <seed> --output artifacts/lowlevel/pc_server_scale_b_<arm>_seed<seed> \
  --warmstart-actor artifacts/lowlevel/pc_bc_c0_c1_c2_c3random_v1_dz64_20261103/pc_bc_actor.pt \
  --record-only
```

正式训练前必须检查：

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
df -h /data
pgrep -af 'train_batch_lowlevel_ppo|evaluate_pc_ppo'
```

A 级实测（`artifacts/lowlevel/pc_server_scale_smoke_4096_dade_head_gpu6_20261103/`）：

- 配置：DADE-head，`4096×256×2`，GPU6；
- 峰值 allocated `15,321.19 MiB`，reserved `16,146 MiB`；
- collection `167,532 env-steps/s`，端到端 `29,412 env-steps/s`；
- step latency p50 `22.83 ms`、p95 `25.18 ms`；
- iteration 1 为 `34.15 s`，iteration 2 为 `34.98 s`，`losses_finite=true`；
- 首次在 GPU3 尝试时，外部进程占用 `9.69 GiB`，本进程 allocated `13.16 GiB` 后还差 `200 MiB` 而 OOM；该失败保存在 `pc_server_scale_smoke_4096_dade_head_20261103/failure_note.json`，不能作为结果；
- 因此 `4096` 正式训练要求至少约 `16.5 GiB` 空闲显存，不能与高占用任务共用显存边缘。

**短程 5-seed 复现结果（`128×256×32`）**

评估使用固定 `0–63` seeds，checkpoint 固定为 `ck31`；配对差的口径均为同训练 seed 的 DADE 减 no-DADE。

| 训练臂 | 半径 | 成功率 | 3D RMSE (m) | 配对成功率差 | 配对 RMSE 差 (m) |
|---|---|---:|---:|---:|---:|
| no-DADE | `2.0 m` | `0.6875 ± 0.0000` | `5.4032 ± 0.0373` | 基线 | 基线 |
| DADE | `2.0 m` | `0.7000 ± 0.0131` | `5.3113 ± 0.0744` | `+0.0125` | `-0.0919` |
| no-DADE | `1.25 m` | `0.3906 ± 0.0156` | `7.1793 ± 0.1625` | 基线 | 基线 |
| DADE | `1.25 m` | `0.4000 ± 0.0086` | `7.1281 ± 0.1014` | `+0.0094` | `-0.0512` |

判定：短程平均方向为正，但逐 seed 的 RMSE 差最大仍达到 `+0.0292 m`（默认）和 `+0.2351 m`（严格），因此只能写成“短程有限平均改善”，不能写成稳定全面优于 no-DADE。产物为 `artifacts/lowlevel/pc_server_c3_5seed_summary_20261103/summary.{json,md}`。

**C4 压力迁移（`ck31`，`stage_override=true`）**

| 训练臂 | 成功率 | 3D RMSE (m) | timeout | saturation | safety |
|---|---:|---:|---:|---:|---:|
| no-DADE | `0.3156 ± 0.0171` | `9.3543 ± 0.0473` | `0.6844` | `0.2562` | `0` |
| DADE | `0.3156 ± 0.0070` | `9.3738 ± 0.0866` | `0.6844` | `0.2567` | `0` |

配对成功率差为 `0.0000`（min `-0.0312`，max `+0.0156`），配对 RMSE 差为 `+0.0195 m`（min `-0.1450`，max `+0.1522`）。这表示 C3 策略迁移到 C4 后 DADE 没有提供压力鲁棒性增益；且这不是 C4 重新训练的结果。产物为 `artifacts/lowlevel/pc_server_c3_5seed_c4_summary_20261103/summary.{json,md}`。

**历史机制诊断（32 seeds × 512 steps）**

| 臂 | base action L2 | base `c_hat` L2 | `current-only` action Δ | `history-zero` action Δ | `history-zero` `c_hat` Δ |
|---|---:|---:|---:|---:|---:|
| 普通 DADE | `0.972964` | `0.551125` | `8.2e-5` | `4.89e-4` | `0.294649` |
| DADE-head | `0.973911` | `0.525999` | `4.2e-5` | `5.71e-4` | `0.276328` |

`c_hat` 对历史消融有明显响应，但传到动作后不足基线动作的 `0.1%`；`history-reverse` 的动作变化仅约 `1e-6`。因此 DADE-head 仍没有证明“真正使用历史”。检查还发现短程 `ck31` 的 head 输出层 L2 仅 `0.00162`，与 `actor` 学习率 `1e-5` 和 27 个有效更新轮次一致。该负结果不能归因于“训练不够长”这一未经检验的解释，必须做隔离通路对照。

**隔离式 DADE-head 门禁**

新增 `pc.dade_action_head_isolated=true` 与可选 `algorithm.learning_rate_dade_head`。隔离模式下：

- actor 主干看到的 `c_hat` 槽位严格置零；
- `c_hat` 只通过 action head 产生动作修正；
- head 参数进入独立 Adam 参数组，warmup 后使用显式学习率；
- 默认不带隔离标志的旧配置仍走原路径，已有 checkpoint 和结果语义不变。

该实验只有在“隔离后 `history-zero` / `current-only` 的动作敏感度显著上升，且默认/严格半径成功率不退化”时才允许扩到 B 级三 seed；否则登记为负结果，不进入 C 级。

**隔离式 DADE-head 短程门禁结果（负结果）**

短程 `ck31` 的双半径闭环结果如下。隔离 head 的默认成功率与普通 DADE-head 持平，严格 `1.25 m` 成功率反而下降 `0.015625`；timeout 上升 `0.015625`，safety 仍为 `0`。

| 模型 | 半径 | 成功率 | 3D RMSE (m) | timeout | material saturation | safety |
|---|---:|---:|---:|---:|---:|---:|
| DADE-head | `2.0 m` | `0.687500` | `5.443713` | `0.312500` | `0.131221` | `0` |
| 隔离 DADE-head | `2.0 m` | `0.687500` | `5.385307` | `0.312500` | `0.130960` | `0` |
| DADE-head | `1.25 m` | `0.406250` | `7.186098` | `0.593750` | `0.117269` | `0` |
| 隔离 DADE-head | `1.25 m` | `0.390625` | `7.148123` | `0.609375` | `0.116284` | `0` |

闭环历史消融与完整历史逐项重合：默认半径 `current-only` / `history-zero` 均为成功率 `0.687500`、RMSE `5.385292` / `5.383556 m`；严格半径均为成功率 `0.390625`、RMSE `7.148092` / `7.145249 m`。因此当前收益仍不是历史通路带来的。

32-seed × 512-step 诊断给出了更直接的否决证据：基线动作 L2 为 `0.981248`；`current-only` 的平均动作变化为 `3.51e-5`（`0.0036%`），`history-zero` 为 `4.45e-4`（`0.0453%`），`history-reverse` 为 `3.56e-7`（`0.000036%`）。虽然 `history-zero` 使 `c_hat` 平均变化 `0.666049`，但动作敏感度仍低于预先设定的 `0.1%` 门禁，且不高于普通 DADE 的 `4.89e-4`。

判定：**隔离 action head 本身不足以建立可测的历史依赖。** 该臂不扩到 B 级三 seed，也不进入 C 级；若继续修复，应改变训练目标或输入通路，例如显式海流预测辅助损失、current-frame 全 dropout 或移除 actor 的当前帧旁路，而不是继续调 head 学习率。

**B 级扩训中间状态**

- no-DADE `seed91430`、`seed91431`、`seed91432` 已完成 `4096×256×40` 并生成 `final.pt`；
- DADE-head `seed91431` 已完成 `4096×256×40`；DADE `seed91430` 与 DADE-head `seed91432` 仍在运行；
- 已完成的 no-DADE seed 正在补齐 `ck0/10/20/30/39` 默认与严格半径评测；
- 两 seed 初步显示 `ck20` 较稳，严格 `1.25 m` 为 `0.4062`、RMSE `7.0548 m`；`ck39` 退化到 `0.3828`、`7.2266 m`；
- 因此最终 checkpoint 不能机械取 `final.pt`，必须使用预定 checkpoint 评测与跨 seed 规则；
- 普通 DADE、DADE-head 与 dropout 仍在训练，缺失队列由 GPU 锁保护，禁止用单 seed 结果提前下结论。

**调度与资源修复**

- 多个驱动进程曾同时看到同一 GPU 空闲并并发启动，存在 OOM 竞态；训练脚本已增加 GPU 级 `flock`，后续任务按卡串行；
- 历史诊断曾在 32 核上创建 255 个 torch 线程，导致主机负载接近 `500` 且 8 分钟无输出；改为 `OMP_NUM_THREADS=4` 后约 2 分钟完成，旧日志保留为失败/中止证据；
- 当前扩训要求单卡至少 `16.5 GiB` 空闲显存；不得抢占其他用户进程，也不得为了并行而降低门禁。

### S8：冻结 PC BC + 残差策略实验组（必须做）

这一组回答的不是“BC 热启动后还能不能继续训练”，而是更严格的问题：**把 PC BC actor 完全冻结，只训练一个小残差策略，能不能在保持 BC 行为先验的同时获得可测的 PPO 改进。** 它与 B 级普通 warm-start 臂必须分开统计，不能把 `ck0` 行为相同误写成“冻结残差有效”。

实现口径：

- 动作严格为 `a = a_bc + residual_scale * delta_theta(obs)`，`residual_scale=0.25`；
- 独立复制一份 BC encoder 和 actor，加载同一个 `pc_bc_actor.pt`，模块 `requires_grad=False`，即使 `policy.train()` 也强制 `eval()`；
- `base_action()` 在 `torch.no_grad()` 下计算，残差 actor 与 DADE head 最后一层零初始化；
- 冻结模块不进入 optimizer；训练开始时验证 `initial_residual_abs_max <= 1e-7`，且总动作与冻结 BC 动作逐元素一致；
- 保存 `model_initial.pt`，训练结束后比较 `model_initial.pt` 与 `final.pt` 中全部 `frozen_bc_*` state entry，要求最大绝对漂移严格为 `0`，并写入 `training_manifest.json` 的 `frozen_bc_audit`；
- BC checkpoint：`artifacts/lowlevel/pc_bc_c0_c1_c2_c3random_v1_dz64_20261103/pc_bc_actor.pt`，SHA-256 `f56c359648a99e3fc8fde51f2e529af5c9916371450ff0d7a942c0f620d33936`。

四个实验臂固定如下，均使用 `4096×256×100 = 104,857,600` env-steps、训练 seed `91430/91431/91432`，不传 `--warmstart-actor`：

| 实验臂 | 冻结 BC | DADE | DADE action head | 历史 dropout | 残差尺度 | 作用 |
|---|---|---|---|---|---:|---|
| `frozen_bc_residual_nodade` | 是 | 否 | 否 | 无 | `0.25` | 残差基线，检验最小改动是否稳定 |
| `frozen_bc_residual_dade` | 是 | 是 | 否 | 无 | `0.25` | DADE 特征经 actor 拼接进入残差 |
| `frozen_bc_residual_dade_head` | 是 | 是 | 是 | 无 | `0.25` | DADE 经独立 action head 进入残差 |
| `frozen_bc_residual_dade_sidechannel10` | 是 | 是 | 是 | `side_channel=1.0` | `0.25` | 只隐藏最新历史行海流侧通道，检验历史推理 |

训练矩阵共 12 个 run，输出根为 `artifacts/lowlevel/pc_server_scale_frozen_bc_residual_20261103/`。每个 run 必须保留 `model_initial.pt`、`model_0/20/40/60/80/99.pt`、`final.pt`、`training_manifest.json`、`iteration_records.jsonl` 和 `run_status.json`。

评测矩阵：

- 固定评估 seed `0–63`，默认 `2.0 m` 与严格 `1.25 m` 两档；
- checkpoint 为 `initial/0/20/40/60/80/99`，`initial` 是“零残差、等价 BC”的强制对照，不能省略；
- 同训练 seed 内以 `frozen_bc_residual_nodade` 为主基线，报告四臂相对基线的成功率和三维 RMSE 配对差；
- 在 `ck99` 上补 `current-only` 与 `history-zero` 闭环消融，输出目录带 ablation 后缀，禁止覆盖完整历史结果；
- 在 `ck99` 上补 H0–H4 与 C4 压力评估，只用于边界检查，不据此反向挑选 checkpoint；
- 与 B 级普通 warm-start 臂比较时，必须明确训练预算不同（100 vs 40 iterations），只能作为趋势对照，不能做同预算因果结论。

判定门禁：

- 初始化门禁：`initial_residual_abs_max <= 1e-7`，总动作与 BC 动作最大误差 `<= 1e-7`；
- 冻结门禁：`frozen_bc_parameters_unchanged=true`、`frozen_bc_parameter_max_abs_drift=0`、`frozen_bc_parameters_require_grad=false`；
- 学习门禁：`ck99` 的残差动作必须非零，且不能长期顶到动作包线；若残差始终近似零，登记为“残差未被训练起来”的负结果；
- 性能门禁：默认和严格半径成功率不得低于 `initial`，safety 不得从 `0` 变正，material saturation 相对 `initial` 增加不得超过 `0.01`；
- 机制门禁：DADE 臂的 `history-zero` / `current-only` 动作敏感度必须达到基线动作 L2 的 `0.1%`；否则即使成功率略升，也只能写“性能结果”，不能写“DADE 历史机制成立”；
- 跨 seed 门禁：三 seed 的配对差必须同号且最小值不跨过退化阈值；否则登记为不稳定或负结果。

训练状态（2026-11-03 17:30）：`frozen_bc_residual_nodade_seed91430` 已完成 100 iterations / `104,857,600` env-steps，manifest 中 `frozen_bc_audit` 为 `frozen_bc_parameters_unchanged=true`、`frozen_bc_parameter_max_abs_drift=0.0`、`frozen_bc_parameters_require_grad=false`；`frozen_bc_residual_dade_seed91430` 与 `frozen_bc_residual_nodade_seed91431` 正在 GPU2/GPU6 运行；`frozen_bc_residual_dade_head_seed91430` 虽已完成，但因 optimizer 分组缺陷已作废，不能计入 DADE 结论。`sidechannel10_seed91430` 及其余 seed 仍在主队列等待。当前只有冻结与初始化门禁结果，尚无性能或 DADE 历史机制正结论。

**Optimizer 分组缺陷（2026-11-03，已修复，旧 DADE-head 结果作废）**

`_configure_optimizer_learning_rates()` 会把 `actor_dade_head.*` 参数单独收集；但旧实现只有在 `learning_rate_dade_head` 存在且不同于 actor 学习率时才创建 `dade_head` 参数组，否则这些参数既不在 `actor` 组，也不在 `critic` 组。检查 `frozen_bc_residual_dade_head_seed91430/model_0..80.pt` 后确认：`actor_dade_head.2.weight` 和 `.bias` 的最大绝对值始终精确为 `0.0`，说明 head 根本没有被优化，而不是“学不出历史依赖”。

影响范围：

- `dade_head`、`dade_head_dropout05`、`frozen_bc_residual_dade_head`、`frozen_bc_residual_dade_sidechannel10` 中未显式配置 `learning_rate_dade_head` 的 run；
- 这些 run 可以保留用于审计“未训练 head 的行为”，但不能用于 DADE 负结论、B/C 对比或 checkpoint 选择；
- no-DADE、普通 DADE（无独立 action head）以及已经显式配置独立 head 学习率的隔离式 DADE-head 不受此分组缺陷影响。

同一轮实现审计还确认两处配置与实验命名不一致：旧 `frozen_bc_residual_dade_head` 启用了 action head，但没有设置 `dade_action_head_isolated=true`；旧 `frozen_bc_residual_dade_sidechannel10` 根本没有设置 `dade_action_head_enabled=true`，所以它是“DADE 特征 + 最新侧通道 dropout”，不是 action-head 臂。B 级 `dade_head` 三个 seed 还跨越了不同的训练脚本版本，其中 seed91431 的 head 虽非零，也不能与另外两个 zero-head run 合并。所有受影响的完成目录已写入 `INVALID_FOR_DADE_CONCLUSIONS.json`，原文件不删除。

修复后，无独立 head 学习率时 head 参数并入 `actor` 组；有独立学习率时仍使用 `dade_head` 组。训练 manifest 新增 `optimizer_param_group_counts` 和 `optimizer_parameter_count`，回归测试 `tests/test_lowlevel_ppo_optimizer.py` 禁止可训练参数遗漏或重复。已同步服务器并通过相关测试。后续正确 DADE-head 重跑必须使用显式 `learning_rate_dade_head=5e-4`，新输出根不能覆盖这些旧目录。

**旧 frozen no-DADE 残差幅值诊断（2026-11-03，初步负结果）**

对 `frozen_bc_residual_nodade_seed91430` 用固定 seeds `0–15` 重跑 `initial` 与 `ck99` 默认半径评测，并在评测器新增残差动作指标：

| checkpoint | success | timeout | 3D RMSE | residual abs mean | residual abs max | residual/base L2 | rudder | stern | RPM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `initial` | `0.625` | `0.375` | `5.64738 m` | `0` | `0` | `0` | `0` | `0` | `0` |
| `ck99` | `0.625` | `0.375` | `5.63210 m` | `0.002323` | `0.004352` | `0.006211` | `0.001337 rad` | `0.000185 rad` | `3.98 RPM` |

残差已经不是精确零，说明残差通路在更新；但它只有 BC 动作 L2 的 `0.62%`，平均舵角修正约 `0.077°`、RPM 修正约 `4`，没有改变成功率和 timeout，3D RMSE 仅改善 `0.0153 m`。在旧配方 `actor_lr=1e-5`、`residual_scale=0.25` 下，该臂登记为“残差未被训练到有效幅度”的初步负结果，不能写成冻结残差提升能力。评测器现在会输出 `residual_action_*`、`residual_to_base_l2_ratio_mean` 和物理量修正，后续候选必须先过这一学习门禁。

统一队列命令（服务器上执行，`setsid` 保证本机待机/断网不影响）：

```bash
cd /data/yhx/underwater-robotics/src/plan_p1/settled_局部规划器_反应式
setsid nohup bash scripts/run_pc_expanded_queue_20261103.sh >/dev/null 2>&1 < /dev/null &
```

队列按以下顺序执行；每个阶段失败只记录 `rc` 并继续下一阶段，不会像旧 watcher 那样因为一个未启动的 C 级条件而死锁：

1. 重跑 B 级 `dade_head_dropout05` 的 `seed91431/91432`；
2. 冻结 BC 残差四臂 × 三 seed × `100` iterations；
3. C 级 no-DADE/DADE × 三 seed × `100` iterations；
4. B 级扩到五 seed（`seed91433/91434`，四臂）；
5. `dade_head_sidechannel10` 三 seed × `40` iterations；
6. 上述各组的固定 seed 双半径评测；
7. 冻结组与 C 组的 `current-only` / `history-zero` 消融；
8. 冻结组、C 组和 side-channel 组的 H0–H4/C4 最终 checkpoint 压力评估。

检查命令：

```bash
tail -f logs/pc_expanded_queue_20261103/queue.log
pgrep -af 'run_pc_expanded_queue|train_batch_lowlevel_ppo|evaluate_pc_ppo'
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv
```

本组结果出来前，只能写“已排队/已实现”，不能写“冻结 BC 残差优于 warm-start”。若三 seed 均未通过性能或机制门禁，保留负结果并停止该方向，不用延长训练掩盖。

### S8b：冻结 BC 残差高容量 corrected 组（必须做）

S8 的第一版配方使用 `actor_lr=1e-5`、`residual_scale=0.25`，固定 seed 审计显示残差只达到 BC 动作 L2 的 `0.62%`，没有改变成功率和 timeout；同时旧 DADE-head 子臂因 optimizer 分组缺陷无效。因此不能把第一版结果当作“冻结 BC 残差不可行”的最终结论，必须补一个高容量、optimizer 已验证的 corrected 组。

本组与 S8 第一版分开输出，不覆盖任何旧目录：

| 实验臂 | DADE 通路 | 历史处理 | 作用 |
|---|---|---|---|
| `frozen_bc_residual_nodade_lr3e-4_scale1p0` | 无 | 无 | 同预算高容量残差基线 |
| `frozen_bc_residual_dade_head_isolated_lr3e-4_scale1p0` | 隔离 DADE action head | 完整历史 | 检验 head 参数确实更新后的 DADE 上限 |
| `frozen_bc_residual_dade_history_only_lr3e-4_scale1p0` | 隔离 DADE action head | pose history + 最新侧通道 dropout `1.0` | 检验残差是否能利用历史海流信息 |

共同口径：

- 冻结 BC 与 S8 相同，BC checkpoint SHA-256 不变，`base_action()` 只在 `no_grad()` 下执行；
- `residual_scale=1.0`、`actor_lr=3e-4`、`critic_lr=3e-4`；DADE-head 臂显式 `learning_rate_dade_head=5e-4` 且 `dade_action_head_isolated=true`；
- `4096×256×100 = 104,857,600` env-steps，训练 seed `91430/91431/91432`；
- 输出根 `artifacts/lowlevel/pc_server_scale_frozen_bc_residual_v2_20261103/`，评测根 `artifacts/lowlevel/pc_server_scale_frozen_bc_residual_v2_eval_20261103/`；
- 启动前确认 GPU4/5 没有计算进程；本组使用 GPU4/5 独立锁，不阻塞已占满的 GPU1 队列，也不抢占其他用户进程。

执行清单：

- [x] 新增三个 corrected 配置并加入训练/评测脚本的 arm 映射；
- [x] 新增独立队列 `scripts/run_pc_frozen_bc_residual_v2_20261103.sh`；
- [x] 本地三臂 1-iteration CPU smoke 通过：`initial_residual_abs_max=0.0`、冻结漂移 `0.0`；head 臂出现 `dade_head` optimizer 组且 head 权重非零；
- [x] 配置与队列已同步服务器，远端配置解析通过；
- [x] 队列已启动（当前 PID `2169497`）；确认 GPU4/5 空闲后，`seed91430` 的 no-DADE 与隔离 DADE-head 已在两卡并行训练，不会覆盖主扩训队列；
- [ ] 三臂 × 三 seed 训练完成；
- [ ] 冻结审计、优化器组计数、head 非零更新门禁通过；
- [ ] 固定 `0–63` seeds 的默认/严格半径评测完成；
- [ ] `current-only` / `history-zero` 机制消融完成；
- [ ] H1/H2/H4/C4 压力评测与配对汇总完成；
- [ ] 根据结果决定是否推广到正式扩训，禁止提前写正结论。

启动命令（服务器，`setsid` 保证本机待机后继续）：

```bash
cd /data/yhx/underwater-robotics/src/plan_p1/settled_局部规划器_反应式
setsid nohup bash scripts/run_pc_frozen_bc_residual_v2_20261103.sh >/dev/null 2>&1 < /dev/null &
```

检查命令：

```bash
tail -f logs/pc_frozen_bc_residual_v2_20261103/queue.log
pgrep -af 'run_pc_frozen_bc_residual_v2|train_batch_lowlevel_ppo'
cat artifacts/lowlevel/pc_server_scale_frozen_bc_residual_v2_20261103/frozen_bc_residual_v2_audit.json
```

本地验证状态（2026-11-03 17:10）：完整 low-level 回归 `252 passed`，ruff 全通过，`git diff --check` 通过。三个 smoke 的 optimizer 组分别为：no-DADE `actor=94018 / critic=94141`，隔离 head `actor=267461 / critic=263613 / dade_head=4355`，history-only `actor=266693 / critic=263613 / dade_head=4355`；两个 head 臂在 1 次更新后的 `actor_dade_head.*` 最大绝对值均为 `0.12497`，证明不再出现旧版 head 完全零更新的问题。服务器 corrected 队列运行中：no-DADE `seed91430` 已完成 100 iterations，隔离 DADE-head 约 57/100，history-only 约 29/100；无 OOM。no-DADE 的 `initial/0/40/60/80/99` 已启动固定 64-seed 双半径早期评测，结果尚未产出。

同轮修复了 `scripts/summarize_pc_server_evals.py` 的 arm 推断：corrected 配置名会被旧通用子串规则误归到 S8 第一版臂，造成跨臂配对和 checkpoint 配对混组。现在三个 corrected 名称及 H1/H2/C4 v2 名称均在通用规则之前精确识别，`tests/test_pc_server_eval_summary.py` 已覆盖；服务器同步后该回归 `9 passed`。旧 summary 和作废 run 标记不改写。

`seed91430` corrected no-DADE 的固定 64-seed 早期结果（单 seed）：

| tag | ck | success | timeout | 3D RMSE (m) | residual/base L2 | Δ RMSE vs initial (m) |
|---|---:|---:|---:|---:|---:|---:|
| default | initial | 0.6875 | 0.3125 | 5.3919 | 0 | 0 |
| default | 20 | 0.6875 | 0.3125 | 5.4014 | 0.002923 | +0.009497 |
| default | 40 | 0.6875 | 0.3125 | 5.3992 | 0.003523 | +0.007286 |
| default | 60 | 0.6875 | 0.3125 | 5.3966 | 0.003316 | +0.004673 |
| default | 80 | 0.6875 | 0.3125 | 5.4006 | 0.005836 | +0.008681 |
| default | 99 | 0.6875 | 0.3125 | 5.3905 | 0.006732 | -0.001393 |
| strict125 | initial | 0.40625 | 0.59375 | 7.1178 | 0 | 0 |
| strict125 | 99 | 0.40625 | 0.59375 | 7.1152 | 0.006427 | -0.002623 |

判定为**单 seed 初步负结果**：残差幅值从零上升到 BC 动作 L2 的约 `0.67%`，但 success/timeout 完全不变，中间 checkpoint RMSE 略差，`ck99` 只有毫米级改善。不能据此关闭实验，因为另外两个训练 seed 和 DADE 臂尚未完成；但也不允许继续用训练 rollout success 代替固定 seed 闭环证据。

### S9：history-only DADE 机制实验（必须做）

S8 的冻结 BC 残差、普通 DADE 和隔离 DADE-head 都暴露出同一个问题：`c_hat` 会随历史变化，但动作几乎不变。根因不是简单的 dropout 或 head 学习率，而是 DADE 的历史 token 仍携带 `u/v/w/p/q/r`，最新一帧状态就能反推当前海流；同时 actor 主干还能从顶层完整状态获得同一信息。继续只调 dropout 或 head 学习率，无法证明动作依赖历史。

S9 的目标是建立一个**没有当前动态状态捷径**的 DADE 通路，并把它接入冻结 BC 残差基线：

- `pc.dade_state_mode="pose"`：actor-DADE 只接收历史 token 的 `x,y,z,phi,theta,psi` 六个位姿列和海流侧通道，完全移除 `p,q,r,u,v,w`；旧配置默认 `full`，不改变已排队实验；
- `history.current_frame_dropout_prob=1.0` 且 `mode="side_channel"`：训练时始终隐藏最新历史行的海流侧通道，迫使 DADE 使用更早历史；
- `aux_current_coef=0.5`、`aux_current_steps=8`：辅助头从遮掉最新侧通道的 pose-history 预测当前侧通道；
- `dade_action_head_isolated=true`、`learning_rate_dade_head=5e-4`：DADE 通过独立动作头进入残差，避免被 actor 主干的 64D 拼接特征淹没；
- actor/critic 仍保留完整状态，保证控制器能力；机制门禁只判断动作是否真的依赖历史，不把主干能力混入 DADE 结论。

实验矩阵如下，先做 `4096×256×40` 的机制 pilot，三 seed 通过门禁后再扩到 `100` iterations：

| 实验臂 | 基础策略 | DADE 状态 | 最新侧通道 | 辅助损失 | 用途 |
|---|---|---|---|---|---|
| `dade_history_only` | no base + BC warm-start | `pose` | 100% 隐藏 | `0.5 × 8 steps` | 无冻结基线下的历史通路 |
| `frozen_bc_residual_dade_history_only` | 冻结 PC BC + `0.25` 残差 | `pose` | 100% 隐藏 | `0.5 × 8 steps` | 冻结 BC 下的历史残差 |

训练输出根：`artifacts/lowlevel/pc_server_scale_history_only_dade_20261103/`。评测与诊断根：`artifacts/lowlevel/pc_server_scale_history_only_dade_eval_20261103/`、`artifacts/lowlevel/pc_server_scale_history_only_dade_diagnostics_20261103/`。

自动门禁：

1. 每个 `ck39` 跑 `32 seed × 512 step` 的 `current-only`、`history-zero` 诊断；
2. `scripts/check_pc_dade_history_gate.py` 要求两种消融的平均动作变化都 `>= 0.1% × base_action_l2`；
3. 无论门禁结果如何，都跑 `32 seed` 的默认 `2.0 m` / 严格 `1.25 m`、`ck0/39`，用于检查学习是否真正改善控制；
4. 只有门禁通过才自动跑 `h1/h2/h4/c4` 的 `ck39` 压力评测；门禁失败只保留性能结果和负结果，不扩大训练预算；
5. 若三 seed 仍无法让动作依赖历史，下一步不是继续调 dropout，而是把残差动作限制为 DADE-head 唯一输出，或改成显式 current estimator 后再进入动作。

队列在服务器上执行（会等待主扩训队列释放锁，不打断当前训练）：

```bash
cd /data/yhx/underwater-robotics/src/plan_p1/settled_局部规划器_反应式
setsid nohup bash scripts/run_pc_history_only_dade_20261103.sh >/dev/null 2>&1 < /dev/null &
```

检查命令：

```bash
tail -f logs/pc_history_only_dade_20261103/queue.log
pgrep -af 'run_pc_history_only_dade|run_pc_expanded_queue|train_batch_lowlevel_ppo'
```

当前状态（2026-11-03 15:48）：`pose` DADE、辅助损失、隔离 action head 和自动门禁脚本已实现；本地 `230 passed`，服务器同 venv `230 passed`，服务器 smoke 完成，队列 PID `2032209` 正在等主队列锁。尚不能写正结论。

Git 追加文件：

```bash
cd /home/bubble/水下机器人
git add \
  'settled_局部规划器_反应式/configs/lowlevel/odarrl_control_ppo_pc_c3_random_realistic_dade_history_only_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json' \
  'settled_局部规划器_反应式/configs/lowlevel/odarrl_control_ppo_pc_c3_random_realistic_frozen_bc_residual_dade_history_only_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json' \
  'settled_局部规划器_反应式/scripts/run_pc_history_only_dade_20261103.sh' \
  'settled_局部规划器_反应式/scripts/check_pc_dade_history_gate.py' \
  'settled_局部规划器_反应式/scripts/diagnose_pc_dade_history.py' \
  'settled_局部规划器_反应式/doc/goals/goal9_pc_server_validation_plan.md'
git commit -m "feat(pc): add history-only DADE mechanism experiment"
git push origin main
```

### S10：H1/H2/C4 硬环境真实训练（补能力，不只做压力评测）

现有 H0–H4/C4 主要是在 C3 checkpoint 上做最终压力评测，这不能回答“模型能否通过训练获得鲁棒性和高速能力”。S10 直接把冻结 BC 残差策略放到目标环境训练，每次 `4096×256×100`、三 seed：

| 实验臂 | 训练环境 | 训练目标 | 主要比较 |
|---|---|---|---|
| `frozen_bc_residual_nodade_h1` | H1 `1.8–2.0 m/s` | 高速、较弱海流 | H1 内分布 vs C3 默认 |
| `frozen_bc_residual_nodade_h2` | H2 `2.0–2.25 m/s` | 更高速度 | H2 内分布 vs H1 |
| `frozen_bc_residual_nodade_c4` | C4 强时变流/强剪切/强传感器 | 强扰动鲁棒性 | C4 内分布 vs C3 默认 |

三者都使用冻结 PC BC + `0.25` 残差、无 DADE，先隔离“环境适配”本身的效果；DADE history-only 只在其机制门禁通过后再叠加。训练输出根为 `artifacts/lowlevel/pc_server_scale_hard_env_finetune_20261103/`。

评测口径：训练分布 `ck0/50/99`，默认 `2.0 m` 与严格 `1.25 m` 的 `ck99`，固定 `32` 个评估 seed。必须同时报告 success、timeout、safety、material saturation、三维 RMSE、speed/yaw RMSE；不能只报成功率。H1/H2 若主要由执行器饱和导致上限，登记为系统边界；C4 若 safety 变正，直接淘汰该 checkpoint。

队列命令（排在 history-only 之后，避免 GPU 冲突）：

```bash
cd /data/yhx/underwater-robotics/src/plan_p1/settled_局部规划器_反应式
setsid nohup bash scripts/run_pc_hard_env_finetune_20261103.sh >/dev/null 2>&1 < /dev/null &
```

当前状态（2026-11-03 16:12）：三套配置完成 JSON/语义解析，本地 ruff 和 `239 passed` 通过，H1 frozen BC 残差 1-iteration CPU smoke 完成；服务器配置解析、`241 passed` 和 smoke 完成。为避免旧汇总把 H1 当成 H2/C4 的基线，等待中的硬环境队列已停止并重启为修正版：评测加入 `initial`，H1/H2/C4 的 `ck0/50/99` 与各自 `initial` 配对，同时保留与 C3 `frozen_bc_residual_nodade` 的同环境对比。新队列 PID `2102665`，仍在等待 history-only 完成。结果为待产出，不能提前把“训练环境更强”写成“能力已提升”。

Git 追加文件：

```bash
cd /home/bubble/水下机器人
git add \
  'settled_局部规划器_反应式/configs/lowlevel/odarrl_control_ppo_pc_h1_speed_1p8_2p0_frozen_bc_residual_nodade_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json' \
  'settled_局部规划器_反应式/configs/lowlevel/odarrl_control_ppo_pc_h2_speed_2p0_2p25_frozen_bc_residual_nodade_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json' \
  'settled_局部规划器_反应式/configs/lowlevel/odarrl_control_ppo_pc_c4_stress_frozen_bc_residual_nodade_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json' \
  'settled_局部规划器_反应式/scripts/run_pc_hard_env_finetune_20261103.sh' \
  'settled_局部规划器_反应式/doc/goals/goal9_pc_server_validation_plan.md'
git commit -m "feat(pc): train frozen residual policies on H1 H2 and C4"
git push origin main
```

### S10b：H1/H2/C4 corrected 高容量冻结残差（新增，必须做）

S10 第一版硬环境训练使用旧配方 `actor_lr=1e-5`、`residual_scale=0.25`。结合 S8 的幅值审计，这很可能再次只得到 `~0.1–1%` 的 BC 动作修正，不能公平回答“残差策略能否获得高速和强扰动能力”。因此新增 S10b，不改写也不删除旧 S10：

| 实验臂 | 训练环境 | 目的 |
|---|---|---|
| `frozen_bc_residual_nodade_h1_lr3e-4_scale1p0` | H1 `1.8–2.0 m/s` | 高速适配 |
| `frozen_bc_residual_nodade_h2_lr3e-4_scale1p0` | H2 `2.0–2.25 m/s` | 更高速度上限 |
| `frozen_bc_residual_nodade_c4_lr3e-4_scale1p0` | C4 强时变流/剪切/传感器扰动 | 强扰动鲁棒性 |

共同协议：冻结同一 PC BC checkpoint；`actor_lr=3e-4`、`critic_lr=3e-4`、`residual_scale=1.0`；无 DADE；三 seed `91430/91431/91432`；`4096×256×100`。配置文件名分别为 `odarrl_control_ppo_pc_{h1_speed_1p8_2p0,h2_speed_2p0_2p25,c4_stress}_frozen_bc_residual_nodade_lr3e-4_scale1p0_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json`，输出根为 `artifacts/lowlevel/pc_server_scale_frozen_bc_residual_hardenv_v2_20261103/`。

评测分两层：

1. 内分布：每个臂只在自己的 H1/H2/C4 环境评测 `initial/0/20/40/60/80/99`，固定 32 seeds，报告 success、timeout、safety、saturation、三维 RMSE、speed/yaw RMSE 和残差物理幅值；
2. 跨环境退化：三个臂统一在 C3 默认 `2.0 m` 和严格 `1.25 m` 下评测 `initial/99`，固定 64 seeds；只有内分布改善且跨环境不明显退化，才可以称为能力提升。

训练后审计必须满足：训练状态 `completed`；`initial_residual_abs_max <= 1e-7`；冻结参数漂移严格为 `0`、`requires_grad=false`；optimizer 仅包含 actor/critic 且两者参数数大于零；无 DADE 臂不得出现 `dade_head` 参数组。若审计失败，队列立即停止，不进入评测。

服务器命令：

```bash
cd /data/yhx/underwater-robotics/src/plan_p1/settled_局部规划器_反应式
setsid nohup bash scripts/run_pc_frozen_bc_residual_hardenv_v2_20261103.sh >/dev/null 2>&1 < /dev/null &
```

当前状态（2026-11-03 17:10）：三套配置完成 JSON/语义解析；三组本地 1-iteration CPU smoke 均通过，`initial_residual_abs_max=0`、冻结漂移 `0`、`requires_grad=false`、optimizer 为 `actor=94018 / critic=94141`。服务器队列 PID `2217790` 已启动并等待 S8b 队列锁；不占用当前训练卡之外的新 GPU。当前只有“已排队/已通过 smoke”，没有任何性能正结论。

Git 追加文件：

```bash
cd /home/bubble/水下机器人
git add \
  'settled_局部规划器_反应式/configs/lowlevel/odarrl_control_ppo_pc_h1_speed_1p8_2p0_frozen_bc_residual_nodade_lr3e-4_scale1p0_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json' \
  'settled_局部规划器_反应式/configs/lowlevel/odarrl_control_ppo_pc_h2_speed_2p0_2p25_frozen_bc_residual_nodade_lr3e-4_scale1p0_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json' \
  'settled_局部规划器_反应式/configs/lowlevel/odarrl_control_ppo_pc_c4_stress_frozen_bc_residual_nodade_lr3e-4_scale1p0_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json' \
  'settled_局部规划器_反应式/scripts/run_pc_frozen_bc_residual_hardenv_v2_20261103.sh' \
  'settled_局部规划器_反应式/scripts/run_pc_server_scale_20261103.sh' \
  'settled_局部规划器_反应式/scripts/eval_pc_server_scale_20261103.sh' \
  'settled_局部规划器_反应式/scripts/summarize_pc_server_evals.py' \
  'settled_局部规划器_反应式/tests/test_pc_server_eval_summary.py' \
  'settled_局部规划器_反应式/doc/goals/goal9_pc_server_validation_plan.md'
git commit -m "feat(pc): add corrected hard-environment frozen residual queue"
git push origin main
```

### S10c：C2/H3 剩余基础实验与自动评测（2026-11-04 已启动）

此前服务器上仍有两个基础缺口：C2 ALOS 只有本地 32-iteration 配对结果，没有服务器三 seed 复现；高速课程只排了 H1/H2/C4，没有排 H3。为避免继续围绕 DADE 调参，本组只补基础能力证据，不新增 DADE 技巧。

共同约束：

- 不覆盖任何旧目录；新产物分别写入 `pc_server_c2_basics_*`、`pc_server_h3_basics_*` 和 `pc_server_remaining_baselines_20261103/`；
- C2 使用 `d_z=32` 的 `pc_bc_c0_c1_c2_v3_20261103/pc_bc_actor.pt`，其 SHA-256 为 `51cd5dec5990c19a799a3aca4f9e73e23e0ba7fade07cddc804c1f8166766efc`；
- H3 使用与 C3/H1/H2 相同的 `d_z=64` 冻结 BC prior，残差零初始化、`actor_lr=3e-4`、`residual_scale=1.0`；
- C2 训练 seed 为 `91205/91206/91207`，no-DADE 与 DADE 严格同 seed 配对；H3 训练 seed 为 `91430/91431/91432`；
- 评测使用固定评估 seed 集；C2 为 `0–63`，H3 与 teacher/BC 压力基线为 `0–31`；
- 新队列使用独立的 `run_pc_server_basics_train_20261103.sh`、`eval_pc_server_basics_20261103.sh` 和 `summarize_pc_server_evals_basics_20261103.py`，避免改写服务器上仍在执行的旧脚本。

训练矩阵：

| 组别 | 实验臂 | 环境 | 预算 |
|---|---|---|---:|
| C2 | `c2_nodade` | ALOS mild，`1.0–1.5 m/s` 命令 | `2048×256×32` |
| C2 | `c2_dade` | 同上，仅保留计划要求的配对 DADE | `2048×256×32` |
| H3 | `frozen_bc_residual_nodade_h3_lr3e-4_scale1p0` | H3 `2.25–2.4 m/s` | `4096×256×100` |

自动评测：

1. C2 两臂评测 `c2` mild 与 `c2_clean` 两个 tag，checkpoint 为 `initial/0/10/20/30/final`；输出 success、timeout、safety、saturation、三维路径 RMSE、speed/yaw RMSE；
2. H3 在自己的分布内评测 `initial/0/20/40/60/80/99`；再在默认 `2.0 m`、严格 `1.25 m` 和 H4 `2.5 m/s` 上评测 `initial/99`，检查跨环境退化；
3. teacher/BC-only 先于训练结果生成基础上界：C2 mild/clean、C3 realistic、H1–H4/C4 teacher，以及 C2/C3/H3 BC-only；这些是后续判断“训练是否真的改善”的对照，不用训练 rollout reward 代替闭环指标；
4. 任一评测 worker 失败时，新评测入口返回非零，队列不会静默汇总不完整结果。

启动命令：

```bash
cd /data/yhx/underwater-robotics/src/plan_p1/settled_局部规划器_反应式
setsid nohup bash scripts/run_pc_remaining_basics_20261103.sh >/dev/null 2>&1 < /dev/null &
```

当前状态（2026-11-04 10:49，队列 PID `3745939`）：

- 队列已持有 `logs/pc_remaining_basics_20261103/queue.lock`，本机待机或 SSH 断开不影响远端执行；
- C2 no-DADE `seed91205` 已在 GPU0 启动，H3 `seed91430` 已在 GPU1 启动；H3 与 C3 早期/history-required 队列共享 GPU1 锁，避免显存超卖；
- teacher/BC-only 基线已全部完成，固定 seed 结果如下：

| 基线 | seeds | success | timeout | 三维路径 RMSE (m) | saturation | safety |
|---|---:|---:|---:|---:|---:|---:|
| C2 mild teacher | `0–63` | `1.0000` | `0.0000` | `1.8852` | `0.1812` | `0` |
| C2 mild BC-only | `0–63` | `1.0000` | `0.0000` | `1.9591` | `0.1696` | `0` |
| C2 clean teacher | `0–63` | `1.0000` | `0.0000` | `1.6231` | `0.0465` | `0` |
| C2 clean BC-only | `0–63` | `1.0000` | `0.0000` | `1.7351` | `0.0440` | `0` |
| C3 realistic teacher | `0–63` | `0.7344` | `0.2656` | `4.6164` | `0.0718` | `0` |
| C3 realistic BC-only | `0–63` | `0.6875` | `0.3125` | `5.3919` | `0.1311` | `0` |
| H1 teacher | `0–31` | `0.9688` | `0.0312` | `2.5725` | `0.4654` | `0` |
| H2 teacher | `0–31` | `1.0000` | `0.0000` | `1.7484` | `0.4819` | `0` |
| H3 teacher | `0–31` | `1.0000` | `0.0000` | `1.7727` | `0.4919` | `0` |
| H4 teacher | `0–31` | `1.0000` | `0.0000` | `2.9402` | `0.4922` | `0` |
| H3 BC-only | `0–31` | `0.8438` | `0.1562` | `9.9935` | `0.3892` | `0` |
| C4 teacher | `0–31` | `0.1875` | `0.8125` | `8.6205` | `0.1731` | `0` |

  H3 BC-only 相对 H3 teacher 的成功率差为 `-0.15625`、三维 RMSE 差为 `+8.2208 m`，说明高速迁移确实是基础缺口；C4 teacher 在强扰动下本身只有 `0.1875` 成功率，后续若残差没有超过该上界，应登记为系统/教师边界而不是 DADE 单独失败；
- H3 配置与训练入口完成本地 1-iteration CPU smoke，`initial_residual_abs_max=0`、冻结漂移 `0`、`requires_grad=false`；
- 目前只有“已启动/基线已出”，没有 C2 PPO 或 H3 残差的性能结论。

检查命令：

```bash
tail -f logs/pc_remaining_basics_20261103/queue.log
pgrep -af 'run_pc_remaining_basics|pc_server_basics_train|train_batch_lowlevel_ppo|evaluate_pc_'
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv
```

Git 提交建议（新增文件，按文件粒度）：

```bash
cd /home/bubble/水下机器人
git add \
  'settled_局部规划器_反应式/configs/lowlevel/odarrl_control_ppo_pc_h3_speed_2p25_2p4_frozen_bc_residual_nodade_lr3e-4_scale1p0_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json' \
  'settled_局部规划器_反应式/scripts/run_pc_remaining_basics_20261103.sh' \
  'settled_局部规划器_反应式/scripts/run_pc_server_basics_train_20261103.sh' \
  'settled_局部规划器_反应式/scripts/eval_pc_server_basics_20261103.sh' \
  'settled_局部规划器_反应式/scripts/summarize_pc_server_evals_basics_20261103.py' \
  'settled_局部规划器_反应式/scripts/summarize_pc_server_evals.py' \
  'settled_局部规划器_反应式/tests/test_pc_server_eval_summary.py' \
  'settled_局部规划器_反应式/doc/goals/goal9_pc_server_validation_plan.md'
git commit -m "feat(pc): queue remaining C2 and H3 baselines"
git push origin main
```

### S10d：五 seed 扩训与自动汇总（2026-11-04 已启动）

S10c 启动后确认原有队列仍健康：C2 no-DADE `seed91205` 已完成 `2048×256×32`，H3 `seed91430` 已完成 `4096×256×100`，对应 DADE 和第二 seed 正在 GPU0/GPU1 执行。原队列在全部训练结束后才统一评测，因此额外启动独立扩训队列，不改写、不重启、不覆盖 S10c。

扩训原则：

- 不新增 DADE 技巧，不继续围绕 DADE 打转；DADE 只保留 S10c 已排好的 C2 同 seed 配对；
- 所有新增训练 seed 写入原有训练根，依靠每 run 的 `JOB_LOCK_DIR` 防止重复写入；
- 新评测写入独立的 `*_eval_ext_20261103` 根，避免与 S10c 的评测和汇总并发覆盖；
- 扩展队列最终会阻塞等待 S10c 的 `queue.lock` 释放，再生成真正的五 seed 联合汇总；
- 远端使用 `setsid nohup`，本机待机、关闭终端或断开 SSH 不影响训练和自动评测。

训练矩阵（新增 seed）：

| 组别 | 实验臂 | 新增训练 seed | 预算 | GPU |
|---|---|---|---:|---|
| C2 | `c2_nodade` / `c2_dade` | `91208/91209` | `2048×256×32` | 2 |
| H3 | `frozen_bc_residual_nodade_h3_lr3e-4_scale1p0` | `91433/91434` | `4096×256×100` | 4/5 |
| C3 | `frozen_bc_residual_nodade_lr3e-4_scale1p0` | `91433/91434` | `4096×256×100` | 2 |
| H1/H2/C4 | 三个 corrected frozen-BC residual no-DADE 臂 | `91433/91434` | `4096×256×100` | 4/5 |

自动评测矩阵：

- C2 扩训 seed：`c2` mild 与 `c2_clean`，`initial/0/10/20/30/final`，评测 seed `0–63`；
- H3 扩训 seed：分布内 `initial/0/20/40/60/80/99`，评测 seed `0–31`；默认 `2.0 m`、严格 `1.25 m`、H4 压力迁移使用 `initial/99`；
- C3 扩训 seed：默认/严格双半径，`initial/0/20/40/60/80/99`，评测 seed `0–63`；
- H1/H2/C4 扩训 seed：各自分布内 `initial/0/20/40/60/80/99`，评测 seed `0–31`；另跑默认/严格 `initial/99`，评测 seed `0–63`；
- 训练后先审计所有冻结 BC run：`status=completed`、初始残差为零、冻结参数最大绝对漂移为零、`requires_grad=false`、actor/critic optimizer 组存在；任一失败则停止扩训队列；
- 扩训先写两 seed 扩展汇总，再等待 S10c 完成后写 `*_summary_all5_20261103/summary.{json,md}`；任何评测或汇总步骤返回非零都会停止队列，不静默接受缺失数据。

启动信息（2026-11-04 11:08）：

- 扩训队列 PID `3779144`，日志 `logs/pc_basics_scale_extension_20261103/queue.log`；
- C2 `seed91208` 已占用 GPU2，H3 `seed91433/91434` 已占用 GPU4/5；
- 脚本：`scripts/run_pc_basics_scale_extension_20261103.sh`；
- 输出：S10c 训练根、`pc_server_*_eval_ext_20261103`、`pc_server_*_summary_ext_20261103`、`pc_server_*_summary_all5_20261103`；
- 独立收尾器 PID `3792998`，日志 `logs/pc_basics_finalize_all5_20261103/queue.log`。它等待 S10c 和扩训队列退出后，重新生成 C2/H3/C3/H1/H2/C4 的五 seed 汇总；H1/H2/C4 必须合并四个旧评测根 `*_h1_eval_*`、`*_h2_eval_*`、`*_c4_eval_*`、`*_cross_eval_*`，不能误用一个不存在的统一根。

本轮同时修复了自动汇总的两个缺陷：

1. C2 直采 PPO 不保存 `model_initial.pt`。评测入口将 C2 的缺失 `initial` 视为可选项；汇总入口在找不到 `initial` 时回退到首个保存 checkpoint `0`，并同时记录 `checkpoint_baseline=0` 与 `checkpoint_baseline_requested=initial`。`0` 只能表述为“首个已保存训练 checkpoint”，不能冒充预训练模型；真正的 warm-start 对照仍是 BC-only。
2. `ck*_seed0_31_history-zero` 一类历史消融目录会被旧解析器误读成训练 seed `0`，且正常结果与消融结果可能被误报为重复。现在训练 seed 从 arm 目录正确读取，普通性能汇总默认只包含 `none`，历史消融需显式请求，不再混入主结果。

本地完整回归为 `585 passed`，ruff 与 `git diff --check` 通过；服务器端新增汇总回归为 `15 passed`。现有 three-seed corrected frozen-residual 结果已经显示负结果：

| 组别 | checkpoint | success | 三维 RMSE (m) | 相对 `initial` 成功率 | 相对 `initial` RMSE |
|---|---:|---:|---:|---:|---:|
| H1 | `99` | `0.8750` | `5.6698` | `0.0000` | `+0.0089 m` |
| H2 | `99` | `0.9063` | `6.7355` | `0.0000` | `+0.0409 m` |
| C4 | `99` | `0.2500` | `8.5402` | `0.0000` | `+0.0470 m` |

因此当前判断仍是：**冻结 BC + PPO 残差在 H1/H2/C4 上没有闭环收益，且 RMSE 略退化。**五 seed 扩展用于确认跨 seed 稳健性和排除偶然性，不是为了寻找最好 checkpoint；若五 seed 仍无收益，下一步应转向教师残差监督/DAgger 或重新定义残差权限，而不是继续增加同配方 PPO 预算。

早鸟评测（仅单训练 seed，不能替代三/五 seed 结论）进一步显示：

- C2 no-DADE `seed91205` 在 mild 上 `ck0→final` 为 success `1.0000→1.0000`、三维 RMSE `1.9509→1.9321 m`；clean 上为 `1.0000→1.0000`、`1.7248→1.7052 m`，闭环稳定且略优于同环境 BC-only；
- H3 frozen residual `seed91430` 在 `ck0→ck99` 上 success `0.8438→0.8125`、timeout `0.1562→0.1875`、三维 RMSE `9.9935→11.1107 m`；`ck20/40` 与 `ck0` 基本重合，`ck60/80/99` 明显退化。该结果说明当前 PPO 并没有在 BC 基础上学到有效残差，反而破坏高速跟踪；必须等待其余 seed 复核，但方向已经很清楚。
- 第二个 seed 复现同一方向：C2 no-DADE `seed91208` 的 mild/clean `ck0→final` RMSE 分别为 `1.9738→1.9443 m`、`1.7538→1.7198 m`，成功率均为 `1.0000`；H3 `seed91431` 的 success `0.8438→0.8125`、RMSE `9.9935→11.1117 m`，与 `seed91430` 一致。

### S10e：冻结 BC + 教师残差监督与自动闭环评测（2026-11-04 新增）

S10d 的早鸟结果说明“同一 PPO 配方继续加预算”没有闭环收益。S10e 因此不再调 DADE，也不先跑 PPO，而是直接监督学习教师残差：

```text
target_residual = (teacher_action - frozen_BC_action) / residual_scale
```

PC 动作是归一化后的 `[rudder, stern, rpm]`，因此标签不需要额外猜测物理尺度。训练时冻结 BC encoder/actor，只优化可训练 `actor_encoder + actor`；输出层零初始化，`initial` checkpoint 与纯 BC 完全一致。

实验矩阵：

| 组别 | 教师轨迹环境 | 训练 seed | 分布内评测 |
|---|---|---:|---|
| C3 | C3 realistic calibrated | `91430/91431/91432` | `default/strict125`，64 seeds |
| H1 | H1 `1.8–2.0 m/s` | 同上 | `h1`，32 seeds |
| H2 | H2 `2.0–2.25 m/s` | 同上 | `h2`，32 seeds |
| H3 | H3 `2.25–2.4 m/s` | 同上 | `h3`，32 seeds |
| C4 | C4 强扰动 | 同上 | `c4`，32 seeds |

流程与门禁：

1. 每个环境采集 `16 × 4096` 条教师状态轨迹，写入独立 `pc_teacher_residual_demos_20261104/`；
2. 每个环境训练三个残差随机 seed，写入 `pc_supervised_residual_20261104/`；
3. 审计 `status=completed`、冻结 BC 参数最大绝对漂移为 `0`、`initial` 与 `final` checkpoint 均存在；
4. 分布内固定 seed 评测 `initial/final`，逐 run 报告 success、timeout、safety、saturation、三维 RMSE、speed/yaw RMSE 与残差物理幅值；
5. H1/H2/H3/C4 额外在 C3 `default/strict125` 上评测 `final`，检查跨环境退化；
6. 每个 arm 自动生成 `summary.{json,md}`，`final` 与同 run 的 `initial` 严格配对。

实现文件：

- `src/auv_lowlevel/batch/pc_residual_pretrain.py`：数据切分、教师残差标签、冻结审计、checkpoint 导出与严格载入；
- `scripts/pretrain_pc_residual.py`：从 PC 训练配置构造与 PPO 完全一致的 actor 架构；
- `scripts/train_batch_lowlevel_ppo.py --warmstart-residual`：监督残差可继续作为 PPO 初值；
- `scripts/run_pc_supervised_residual_20261104.sh`：采集、训练、审计、评测、汇总一条龙队列；
- `tests/test_lowlevel_pc_residual_pretrain.py` 与汇总器回归。

验证状态：本地单测通过；真实 C3 demo 单 epoch smoke 得到 `initial_validation_loss=0.007150`、`best_validation_loss=0.006935`，`frozen_bc_parameter_max_abs_drift=0`；监督 checkpoint 可被 `evaluate_pc_ppo.py` 严格加载并完成 1-seed 闭环；`--warmstart-residual` 1-iteration CPU smoke 通过，manifest 记录非零初始残差 `0.023729`。

S10e 执行结果（2026-11-04 11:31–11:48，服务器）：

- 教师采集 `5/5` 成功，监督训练 `15/15` 完成，冻结 BC 参数漂移全部为 `0`；审计失败数为 `0`；
- 自动评测生成 `84` 个 run-level summary，队列退出码为 `0`，没有静默缺失；所有跨环境检查均加入同 run 的 `initial` 对照；
- 分布内结果：H2 success `0.9062→0.9583`、RMSE `6.6946→4.0056 m`；H3 success `0.8438→0.9792`、RMSE `9.9935→3.3118 m`；H1 success `0.8750→0.8750`、RMSE `5.6609→5.3578 m`；C4 success `0.2500→0.2396`、RMSE `8.4932→8.6479 m`；
- 跨环境结果：H3 的 C3 默认环境 success `0.6875→0.4479`、RMSE `4.9269→12.5591 m`，严格 `1.25 m` 为 `0.4688→0.2812`、`6.5153→13.7405 m`；H1/H2 也出现类似退化；
- 因此当前判定是：**教师残差监督能在训练分布内显著修复高速跟踪，但会明显过拟合训练环境，不能作为跨环境通用收益。** 下一步若继续，应做多环境混合标签或 DAgger，而不是继续增加 DADE 或同配方 PPO 预算。
- 产物：`artifacts/lowlevel/pc_supervised_residual_summary_20261104/index.json` 及五个 arm 的 `summary.{json,md}`；队列日志 `logs/pc_supervised_residual_20261104/queue.log`。

### S10f：显式速度×海流网格的 DADE 配对验证（2026-11-04，已完成，整体 fail）

S10e 和 mixed-domain 的证据都受“训练域来自历史环境拼接”限制。S10f 因此建立一个显式的三格训练域和一个严格留出格：

| 格子 | 命令速度 | 海流速度 | 用途 |
|---|---:|---:|---|
| `grid_lw` | `1.0–1.4 m/s` | `0.10–0.20 m/s` | 训练 |
| `grid_ls` | `1.0–1.4 m/s` | `0.30–0.45 m/s` | 训练 |
| `grid_hw` | `1.8–2.0 m/s` | `0.10–0.20 m/s` | 训练 |
| `grid_hs` | `1.8–2.0 m/s` | `0.30–0.45 m/s` | held-out，零样本评测 |

两臂共享冻结 BC、同一批 `lw/ls/hw` teacher demos、训练 seed `94030–94034`、评估 seed `0–31` 和全部训练预算；唯一差异是 DADE-pose。`10/10` 个训练 run 完成且审计无失败，冻结 BC 参数最大漂移为 `0.0`，`grid_hs` 未泄漏。`220` 个正式闭环 summary 全部 `rc=0`，配对报告为 `artifacts/lowlevel/pc_dade_grid_report_20261104/report.{json,md}`，协议全文见 `doc/goals/goal13_dade_speed_current_grid.md`。

关键结果（DADE / no-DADE）：

| Tag | success | 3D RMSE | timeout | saturation | Gate |
|---|---:|---:|---:|---:|---|
| `grid_lw` | `1.0000 / 0.9812` | `1.88 / 2.20 m` | `0.0000 / 0.0188` | `0.1574 / 0.1837` | pass |
| `grid_ls` | `0.6813 / 0.6625` | `4.79 / 4.82 m` | `0.3187 / 0.3375` | `0.1274 / 0.1431` | pass |
| `grid_hw` | `0.9750 / 0.9875` | `3.48 / 2.52 m` | `0.0250 / 0.0125` | `0.4607 / 0.4797` | **fail**（RMSE 比 `1.523`） |
| `grid_hs` | `0.9000 / 0.8438` | `4.90 / 6.11 m` | `0.1000 / 0.1563` | `0.4382 / 0.4278` | pass（success `+0.0563`，RMSE 比 `0.8220`） |
| default | `0.6813 / 0.6375` | `4.96 / 5.20 m` | `0.3187 / 0.3625` | `0.1171 / 0.1359` | **fail**（success `<0.70`） |
| strict125 | `0.4562 / 0.4062` | `6.36 / 6.63 m` | `0.5437 / 0.5938` | `0.1074 / 0.1249` | **fail**（success `<0.50`，timeout `>0.531`） |
| H1 | `0.9437 / 0.9250` | `3.76 / 4.36 m` | `0.0563 / 0.0750` | `0.4455 / 0.4481` | **fail**（success、saturation） |
| H3 | `1.0000 / 0.9750` | `1.84 / 3.65 m` | `0.0000 / 0.0250` | `0.5098 / 0.5029` | **fail**（saturation） |

H4 压力测试为 `1.0000 / 1.88 m / 0.5061`，C4 为 `0.2500 / 8.27 m / 0.2155 / timeout 0.7500`。**判定：`overall_status=fail`。** `grid_hs` 的 held-out 配对正信号是真实且可复现的，但 `grid_hw` 的 RMSE 退化、default/strict125 未过绝对线、H1/H3/H4 saturation 超线，不能声称泛化或执行器问题已解决。按预注册规则不挑选 seed、不追加同配方调参；下一步转向 actor 实际地速跟踪、闭环状态覆盖、执行器权限/饱和奖励或新的 DAgger 设计。

### S10g：残差权限隔离——只允许 RPM 修正（2026-11-04，已完成，门禁失败）

ALOS-R 和 mixed-domain 的共同失败模式是：控制层已经把地速命令拉高，但 actor 的闭环实际地速仍会在转向/海流组合下跌到巡航速度的 `0.21–0.38×`。现有 full-action 残差同时改动舵角、艉水平舵和 RPM，优化目标会被舵面误差主导，RPM 通道没有获得足够、稳定的学习信号；R1K 的命令速度下限因此不能兑现为实际地速。

本组不再改 DADE、奖励或 ALOS 系数，只做一次执行器权限隔离实验：

```text
action = frozen_BC(x, command) + scale * mask * residual_actor(x, command)
mask = [0, 0, 1]        # rudder=0, stern=0, rpm=1
```

冻结 BC 继续负责航向/俯仰，残差只修螺旋桨；这是权限约束，不是第二个残差网络，也不改变观测维度、动作维度或 ALOS。训练数据仍只使用 `grid_lw/grid_ls/grid_hw` 三格 teacher demos，`grid_hs` 严格留出。对照是现有 no-DADE full-action grid 五 seed 产物，不重训对照臂。

实现与配置：

- `base_policy.residual_action_mask`：长度为 3 的 0/1 掩码，默认 `[1, 1, 1]`，旧配置行为不变；
- 只在 `base_policy.mode=frozen_pc_bc_residual` 时允许设置；至少一个通道为 1；
- 屏蔽通道的监督目标也置零，避免验证损失把不可用通道的动作误差算进早停；
- checkpoint 记录 mask，加载时校验，防止拿 full-action checkpoint 冒充 RPM-only；
- 训练配置：`configs/lowlevel/odarrl_control_ppo_pc_grid_gpu_frozen_bc_residual_nodade_rpm_only_lr3e-4_scale1p0_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json`；
- 队列：`scripts/run_pc_grid_rpm_only_20261104.sh`；
- 汇总/门禁：复用 `scripts/summarize_pc_server_evals_basics_20261103.py` 与新增的配对报告脚本；
- 产物：`artifacts/lowlevel/pc_grid_rpm_only_*_20261104/`，不覆盖已有 DADE/grid 产物；
- 参考臂也使用独立 `pc_grid_rpm_only_reference_eval_20261104` 重评，避免旧 summary 缺少新诊断字段。

执行计划：

- [x] 增加 residual action mask、配置解析、checkpoint 审计和单元测试；
- [x] 本地测试、ruff、旧配置向后兼容测试通过；
- [x] 1 个训练 seed × 8 个评估 seed 的 smoke，确认冻结 BC 漂移为 0、只有 RPM 通道变化；
- [x] 修正 smoke 结果目录隔离和参考臂同 seed 配对；smoke 使用独立的 train/eval/summary/report/reference_eval 根目录；
- [x] 运行 `94030–94034 × 32` 评估 seed，tags 为 `grid_lw grid_ls grid_hw grid_hs default strict125 h1 h2 h3 h4 c4`；
- [x] 与现有 no-DADE full-action grid 基线按同训练 seed、同评估 seed 配对；
- [x] 回填结果，负结果不删除、不挑选 seed。

实现验证（2026-11-04）：

- 本地相关测试 `95 passed`，ruff 通过；服务器同 venv `95 passed`；
- smoke 训练 seed `94030`、评估 seed `0–7`，tags 为 `grid_hs/default/strict125`；
- `audit.json` 的 `failures=[]`，训练 device 为 `cuda`，训练格严格为 `pc_grid_low_weak/pc_grid_low_strong/pc_grid_high_weak`，`pc_grid_high_strong` 未泄漏；
- checkpoint 的 `residual_action_mask=[0,0,1]`，冻结 BC 参数最大漂移为 `0.0`，`prediction_abs_max` 为 `[0.0,0.0,0.2479]`，rudder/stern 残差严格为零；
- treatment 与 reference 都使用评估 seed `0–7` 重算，防止用 32-seed 基线硬比 8-seed smoke；
- 评测输出新增/统一 `ground_route_speed_ratio_episode_min_mean`：先对每个 episode 取路线最低值，再对 episode 求均值；原有全局最低值保留为诊断。硬门禁使用 episode-min 均值，避免单帧转向投影为 0 让所有策略永久不可达。

smoke 指标（仅 1 个训练 seed，不作最终判定）：

| Tag | RPM success / RMSE / timeout / sat | Full success / RMSE / timeout / sat | Route min ep-mean | Gate |
|---|---|---|---|---|
| `grid_hs` | `0.8750 / 4.76 m / 0.1250 / 0.4479` | `0.8750 / 4.51 m / 0.1250 / 0.4408` | `0.5329` | fail |
| default | `0.7500 / 3.48 m / 0.2500 / 0.1350` | `0.7500 / 3.42 m / 0.2500 / 0.1315` | `0.4495` | fail |
| strict125 | `0.3750 / 6.53 m / 0.6250 / 0.1153` | `0.3750 / 6.47 m / 0.6250 / 0.1131` | `0.4137` | fail |

smoke 的结论只是“权限隔离和自动化链路正确”，不是 RPM-only 效果结论。1-seed 下 RPM-only 与 full-action 的成功率/timeout 相同，实际航迹最低速度仍低于 `0.75`，strict125 仍超时；必须等五 seed × 32 评估 seed 的配对报告。

正式结果（5 个训练 seed × 32 个评估 seed）：

| Tag | Role | RPM success | Full success | RPM RMSE | Full RMSE | RPM timeout | RPM sat | Full sat | Route min ep-mean (RPM / Full) | Gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `grid_lw` | in-domain | `1.0000` | `0.9812` | `1.87 m` | `2.20 m` | `0.0000` | `0.1940` | `0.1837` | `0.7195 / 0.7041` | pass |
| `grid_ls` | in-domain | `0.6312` | `0.6625` | `4.93 m` | `4.82 m` | `0.3688` | `0.1528` | `0.1431` | `0.4595 / 0.4717` | fail |
| `grid_hw` | in-domain | `0.9938` | `0.9875` | `2.26 m` | `2.52 m` | `0.0063` | `0.4595` | `0.4797` | `0.6800 / 0.6835` | pass |
| `grid_hs` | held-out | `0.7875` | `0.8438` | `7.62 m` | `6.11 m` | `0.2125` | `0.3986` | `0.4278` | `0.5135 / 0.5588` | fail |
| default | retention | `0.6188` | `0.6375` | `5.36 m` | `5.20 m` | `0.3812` | `0.1409` | `0.1359` | `0.4910 / 0.4989` | fail |
| strict125 | retention | `0.4313` | `0.4062` | `6.45 m` | `6.63 m` | `0.5687` | `0.1312` | `0.1249` | `0.4329 / 0.4373` | fail |
| H1 | robustness | `0.8750` | `0.9250` | `6.18 m` | `4.36 m` | `0.1250` | `0.4225` | `0.4481` | `0.5685 / 0.6100` | fail |
| H2 | diagnostic | `0.8562` | `0.9875` | `9.69 m` | `2.69 m` | `0.1437` | `0.4108` | `0.4992` | `0.5844 / 0.6643` | diagnostic |
| H3 | robustness | `0.5312` | `0.9750` | `18.17 m` | `3.65 m` | `0.4688` | `0.3224` | `0.5029` | `0.4635 / 0.6484` | fail |
| H4 | pressure-only | `0.2375` | `0.9625` | `22.09 m` | `4.32 m` | `0.7188` | `0.3014` | `0.4906` | `0.3018 / 0.6340` | pressure-only |
| C4 | pressure-only | `0.2188` | `0.2062` | `8.89 m` | `8.83 m` | `0.7812` | `0.2390` | `0.2339` | `0.4466 / 0.4323` | pressure-only |

配对差值：

- `grid_lw`：success `+0.0187`、RMSE 比 `0.8646`，是明确的小幅正收益；
- `grid_hw`：success `+0.0063`、RMSE 比 `0.9478`、saturation `-0.0202`，也过训练格门禁；
- `grid_ls`：success `-0.0312`，超过 `0.02` 的下降上限，训练格门禁失败；
- held-out `grid_hs`：success `-0.0563`、RMSE 比 `1.2738`、timeout `+0.0563`，没有迁移收益；
- default：success `-0.0187`、RMSE `+0.16 m`、timeout `+0.0187`，且实际航迹最低速度 `0.4910 < 0.75`；
- strict125：success 虽小升 `+0.0250`、RMSE 比 `0.9736`，但绝对 success `0.4313 < 0.50`、timeout `0.5687 > 0.531`、实际航迹最低速度 `0.4329 < 0.75`，仍失败；
- H1：success `-0.0500`、RMSE 比 `1.5479`；
- H3：success `-0.4437`、RMSE 比 `6.2748`、timeout `+0.4437`；saturation 从 `0.5029` 降到 `0.3224`，但这是以大幅失速/超时为代价，不能算执行器收益；
- H4：success `-0.7250`、RMSE 比 `8.4655`，压力测试灾难退化。

**正式判定：`overall_status=fail`。** RPM-only 的权限隔离确实降低了部分场景的舵面/饱和负担，并在 `grid_lw/grid_hw` 得到稳定局部收益；但它不能稳定兑现实际地速，且在 `grid_hs/H1/H2/H3/H4` 产生大幅轨迹和成功率退化。该臂保留为“只修 RPM 不足以解决闭环地速/跨环境鲁棒性”的负证据，不追加同配方调参。参考臂全部 55 个 run 已用同一新版 evaluator 重评；与旧缓存比较，success/timeout/RMSE/saturation 的最大差值为 `0.0`，排除了环境版本造成的假差异。

产物：

- 训练审计：`artifacts/lowlevel/pc_grid_rpm_only_supervised_residual_20261104/audit.json`；
- treatment 评测：`artifacts/lowlevel/pc_grid_rpm_only_eval_20261104/`；
- 同环境参考重评：`artifacts/lowlevel/pc_grid_rpm_only_reference_eval_20261104/`；
- 汇总：`artifacts/lowlevel/pc_grid_rpm_only_summary_20261104/summary.{json,md}`；
- 配对报告：`artifacts/lowlevel/pc_grid_rpm_only_report_20261104/report.{json,md}`。

预注册门禁：

- 训练格：相对 no-DADE full-action 基线，success 下降不超过 `0.02`，RMSE 不超过 `1.05×`，saturation 增量不超过 `0.05`；
- held-out `grid_hs`：success `>=0.85`、RMSE `<=6.0 m`，相对基线 success 不下降且 RMSE 不超过 `1.00×`；
- default：success `>=0.70`、RMSE `<=5.0 m`；
- strict125：success `>=0.50`、RMSE `<=6.5 m`、timeout `<=0.531`；
- H1：success `>=0.95`、RMSE `<=5.0 m`、saturation `<=0.40`；
- H3：success `>=0.95`、RMSE `<=3.4 m`、saturation `<=0.45`；
- 实际地速：default/strict125 的 episode 最低 `ground_route_speed_ratio` 均值必须 `>=0.75`；该指标是真实 DVL/真值诊断，不由命令速度下限替代；
- H4/C4 只做压力测试，不进入主门禁。

如果 RPM-only 仍不能提高实际地速，下一轮才考虑带地速误差的采样权重或新的奖励/闭环状态覆盖；不回到 DADE 超参搜索，也不放宽 `0.75` 地速线。

Git 提交步骤（用户确认后执行，本代理不自动提交）：

```bash
cd /home/bubble/水下机器人
git add \
  'settled_局部规划器_反应式/src/auv_lowlevel/batch/geometry.py' \
  'settled_局部规划器_反应式/src/auv_lowlevel/batch/guidance.py' \
  'settled_局部规划器_反应式/src/auv_lowlevel/batch/pc_env.py' \
  'settled_局部规划器_反应式/src/auv_lowlevel/batch/pc_vec_env.py' \
  'settled_局部规划器_反应式/src/auv_lowlevel/config.py' \
  'settled_局部规划器_反应式/scripts/evaluate_pc_ppo.py' \
  'settled_局部规划器_反应式/scripts/eval_pc_server_basics_20261103.sh' \
  'settled_局部规划器_反应式/scripts/summarize_pc_server_evals_basics_20261103.py' \
  'settled_局部规划器_反应式/tests/test_batch_alos_r.py' \
  'settled_局部规划器_反应式/tests/test_lowlevel_pc_env.py' \
  'settled_局部规划器_反应式/tests/test_pc_server_eval_summary.py' \
  'settled_局部规划器_反应式/doc/goals/goal9_pc_server_validation_plan.md'
git diff --cached --check
git commit -m "fix(pc): align ALOS-R evaluation and speed diagnostics"

git add \
  'settled_局部规划器_反应式/src/auv_lowlevel/batch/pc_policy.py' \
  'settled_局部规划器_反应式/src/auv_lowlevel/batch/pc_residual_pretrain.py' \
  'settled_局部规划器_反应式/src/auv_lowlevel/ppo_config.py' \
  'settled_局部规划器_反应式/configs/lowlevel/odarrl_control_ppo_pc_grid_gpu_frozen_bc_residual_nodade_rpm_only_lr3e-4_scale1p0_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json' \
  'settled_局部规划器_反应式/scripts/run_pc_grid_rpm_only_20261104.sh' \
  'settled_局部规划器_反应式/scripts/summarize_pc_grid_rpm_only_20261104.py' \
  'settled_局部规划器_反应式/tests/test_lowlevel_ppo_config.py' \
  'settled_局部规划器_反应式/tests/test_lowlevel_pc_residual_pretrain.py' \
  'settled_局部规划器_反应式/tests/test_pc_grid_rpm_only_report.py' \
  'settled_局部规划器_反应式/doc/goals/goal9_pc_server_validation_plan.md'
git diff --cached --check
git commit -m "feat(pc): isolate residual authority to rpm"
git push origin main
```

### S11：history-required DADE（阻断当前动态状态旁路）

history-only pilot 的 `dade_state_mode="pose"` 只移除了 DADE 历史中的 `p/q/r/u/v/w`，但 actor 顶层 `state` 仍含完整当前动态状态。因此 PPO 可以继续走 `z_t` 旁路，历史消融不改变动作，仍不能证明 DADE 使用了历史。S11 的目标不是再调 dropout，而是让“不会用历史”成为可测的控制缺陷：

- 冻结 BC 的 `base_action()` 继续读取完整 `state`，保持 BC 行为先验；
- 可训练残差 actor 的 `state` 只保留 `x,y,z,phi,theta,psi`，`p/q/r/u/v/w` 置零；
- DADE 仍使用 `dade_state_mode="full"`，因此可以从较早的完整历史 token 推断当前动态状态；
- `current_frame_dropout_mode="history"` 在训练中以 `1.0` 概率抹掉最新历史 token，保留更早的 token；
- DADE action head 隔离且零初始化，`c_hat` 只经 action head 进入残差；
- 保留当前估计辅助损失（`0.5 × 8 steps`），继续训练 DADE 表示。

这形成一个明确的瓶颈：如果 DADE 不利用较早历史，残差 actor 看不到当前速度/角速度，动作会直接退化；如果 DADE 真正学会推断历史，动作消融和性能都应出现可测变化。

实现与配置：

- `PCConfig.actor_state_mode` / `PCActorCritic.actor_state_mode`，默认 `"full"`，新臂使用 `"pose"`；
- `history.current_frame_dropout_mode` 新增 `"history"`，只遮最新历史 token，不遮顶层状态；
- 配置：`configs/lowlevel/odarrl_control_ppo_pc_c3_random_realistic_frozen_bc_residual_dade_history_required_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json`；
- 队列：`scripts/run_pc_history_required_dade_20261103.sh`；
- 训练：三 seed、`4096×256×40`，先做机制 pilot；`ck39` 上固定 `32 seed × 512 step` 做 `current-only` / `history-zero` / `history-reverse` 诊断，只有默认两模式全部通过 `0.1%` 门禁才追加 H1/H2/H4/C4；无论门禁是否通过都保留默认/严格半径的 `ck0/39` 和闭环历史消融。

验证状态（2026-11-03 16:12）：本地 `239 passed`、ruff 通过；服务器同 venv `241 passed`；本地和服务器 1-iteration CPU smoke 均通过，`initial_residual_abs_max=0`、`frozen_bc_parameter_max_abs_drift=0`、`frozen_bc_parameters_require_grad=false`。

执行结果（2026-11-04）：三 seed 训练、`ck39` 机制消融和默认/严格半径闭环均完成。`current-only` 动作变化约 `0.004%`，`history-zero` 约 `0.06–0.08%`，全部低于 `0.1%` 门禁；闭环 default 两模式均为成功率 `0.6875`、RMSE `4.922–4.924 m`，strict125 为 `0.4688 / 6.508 m`，与 `ck0` 无实质差异。该臂保留为“显式历史依赖仍未被学会”的负证据，不扩 seed、不再调 dropout/head。报告：`artifacts/lowlevel/pc_server_scale_history_required_dade_gate_20261103/gate.json`、`artifacts/lowlevel/pc_server_scale_history_required_dade_summary_20261103/summary.{json,md}`。

Git 提交建议按文件粒度分两次：

```bash
cd /home/bubble/水下机器人

git add \
  'settled_局部规划器_反应式/src/auv_lowlevel/batch/pc_policy.py' \
  'settled_局部规划器_反应式/src/auv_lowlevel/batch/pc_bc.py' \
  'settled_局部规划器_反应式/src/auv_lowlevel/batch/pc_vec_env.py' \
  'settled_局部规划器_反应式/src/auv_lowlevel/batch/pc_history_ablation.py' \
  'settled_局部规划器_反应式/src/auv_lowlevel/ppo_config.py' \
  'settled_局部规划器_反应式/scripts/train_batch_lowlevel_ppo.py' \
  'settled_局部规划器_反应式/tests/test_lowlevel_pc_bc.py' \
  'settled_局部规划器_反应式/tests/test_lowlevel_ppo_config.py' \
  'settled_局部规划器_反应式/tests/test_lowlevel_pc_vec_env.py' \
  'settled_局部规划器_反应式/tests/test_lowlevel_encoded_policy.py' \
  'settled_局部规划器_反应式/tests/test_lowlevel_pc_policy.py'
git commit -m "feat(pc): add frozen BC residual policy training"

git add \
  'settled_局部规划器_反应式/configs/lowlevel/odarrl_control_ppo_pc_c3_random_realistic_frozen_bc_residual_dade_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json' \
  'settled_局部规划器_反应式/configs/lowlevel/odarrl_control_ppo_pc_c3_random_realistic_frozen_bc_residual_dade_head_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json' \
  'settled_局部规划器_反应式/configs/lowlevel/odarrl_control_ppo_pc_c3_random_realistic_frozen_bc_residual_dade_sidechannel10_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json' \
  'settled_局部规划器_反应式/configs/lowlevel/odarrl_control_ppo_pc_c3_random_realistic_frozen_bc_residual_nodade_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json' \
  'settled_局部规划器_反应式/configs/lowlevel/odarrl_control_ppo_pc_c3_random_realistic_dade_head_sidechannel10_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json' \
  'settled_局部规划器_反应式/scripts/run_pc_server_scale_20261103.sh' \
  'settled_局部规划器_反应式/scripts/eval_pc_server_scale_20261103.sh' \
  'settled_局部规划器_反应式/scripts/summarize_pc_server_evals.py' \
  'settled_局部规划器_反应式/scripts/run_pc_expanded_queue_20261103.sh' \
  'settled_局部规划器_反应式/scripts/run_pc_post_queue_gapfill_20261103.sh' \
  'settled_局部规划器_反应式/doc/goals/goal9_pc_server_validation_plan.md'
git commit -m "chore(pc): queue frozen BC residual and expanded server runs"

git push origin main
```

### S12：ALOS-R 制导改进（2026-11-04，R1K 五 seed 未过；held-out seeds 32–63 复核也未过）

目标是在冻结 no-DADE policy、不改变 actor 输入和动作包线的前提下，只改 ALOS 层。实现包括 robust 终端航向捕获、部署传感器海流蟹行、高速 lookahead 和饱和调速；旧配置默认全部关闭。

S12 使用服务器独立目录 `/data/yhx/underwater-robotics/src/plan_p1/settled_局部规划器_反应式_alos_r`，不覆盖、不重启原 multienv 主队列。固定 no-DADE `seed94030` 的 `final.pt` 和评估 seeds `0–31`：

- R0 当前基线：default `0.7188 / 4.6411 m`，strict125 `0.4688 / 6.5512 m`，H1 `1.0000 / 1.7466 m`，H3 `1.0000 / 1.7735 m`，C4 `0.2188 / 8.4783 m`。
- R1（只开 robust terminal，`terminal_speed_ratio=1.0`）：default `0.3750 / 8.2618 m`，strict125 `0.2188 / 8.7730 m`，H1 `0.3750 / 13.1718 m`，H3 `0.5625 / 8.6590 m`，C4 `0.1562 / 10.2396 m`。
- R1 的 terminal speed ratio 全部为 `1.0`，但没有引入终端减速；strict125 timeout 从 `0.5312` 恶化到 `0.7812`，因此前置门禁失败。
- 诊断确认 PCEnv 的旧 `terminal_speed_mps=0` 实际被车辆最小速度 `1.0 m/s` 钳位，不存在真实“降到 0 后爬行”；R1 的主要问题是 `d_align` 最大 16 m 比旧 terminal capture 的 20 m 更晚，以及终端小减速被取消。把 align 拉到 16 m 可改善，但仍不如 R0。
- 最新速度约束为整条航线命令速度不得低于 episode 巡航速度的 75%。实现新增 `minimum_speed_ratio=0.75`，并在 PCEnv 的速率限制之后再次钳位；评估新增 `guidance_route_speed_ratio_min`。起步螺旋桨和初始纵荡速度允许非零：`initial_propeller_rpm_ratio`、`initial_speed_ratio` 按实际 episode 巡航速度缩放，默认 `0.0/1.0`，旧配置不变。
- 另新增 `ground_speed_ratio_min/mean` 与 `ground_route_speed_ratio_min/mean`，分别统计真实 NED 地速模长比和路径切向投影比。它们只作诊断，不进入硬门禁；R1K default/strict 的命令速度最低比均为 `1.0000`，真实地速最低比仅 `0.3769`，说明命令下限不等于实际地速下限。
- R1B（保留旧 20 m 捕获，终端线性降至 75%）：default `0.5625 / 5.0593 m`，strict125 `0.3750 / 6.2917 m`，H1 `0.8750 / 4.6098 m`，H3 `1.0000 / 1.7654 m`，C4 `0.1562 / 8.6987 m`。所有 tag 的路线最低速度比最低为 `0.7659`，满足 75% 硬约束；但 strict125 timeout `0.6250`、default success `0.5625`，H1/H3 saturation 仍高于门禁。

判定：**R1/R1B fail，R2/R3 不启动五 seed。** R1B 证明 20 m 捕获和 75% 速度下限本身有效，但 strict125/default 的问题不能被终端速度调度解释；下一步应转向当前/路径几何或策略闭环覆盖，而不是继续调终端减速。

起速因子与地速反馈（32 个评估 seed）：

- R1G（地速反馈开、RPM 0、初速 1.10×）：default `0.7812 / 4.7543 m`，strict `0.5000 / 7.3160 m`。
- R1H（反馈开、RPM 75%、初速 1.10×）：default `0.6250 / 6.3082 m`，strict `0.4375 / 7.8910 m`。
- R1I（反馈关、RPM 0、初速 1.10×）：default `0.6875 / 4.8698 m`，strict `0.4062 / 7.0924 m`。
- R1J（反馈关、RPM 75%、初速 1.10×）：default `0.5938 / 5.9776 m`，strict `0.4062 / 7.1809 m`。
- R1K（反馈开、RPM 0、初速 1.00×）：default `0.8125 / 4.5738 m`，strict `0.5000 / 7.4043 m`。R1K 是单 seed 主成功门禁最好的配方；初速 1.10× 和 75% 预置 RPM 都没有收益。

R1K 五训练 seed × 32 评估 seed均值：

| tag | success | timeout | 3D RMSE | saturation | command speed min | ground speed min |
|---|---:|---:|---:|---:|---:|---:|
| default | `0.8000 ± 0.0319` | `0.2000` | `4.8163 m` | `0.2550` | `1.0000` | `0.3769` |
| strict125 | `0.4938 ± 0.0125` | `0.5063` | `7.5770 m` | `0.2392` | `1.0000` | `0.3769` |
| H1 | `0.8875 ± 0.0580` | `0.1125` | `5.3728 m` | `0.4705` | `1.0000` | `0.4692` |
| H3 | `0.9625 ± 0.0459` | `0.0375` | `3.2171 m` | `0.4982` | `1.0000` | `0.6071` |
| H4 | `0.9438 ± 0.0538` | `0.0563` | `4.3161 m` | `0.4962` | `1.0000` | `0.6605` |

R2B（R1K + 蟹行）和 R3B（R2B + 高速 lookahead/饱和调速）8 个评估 seed smoke：R2B default `0.8750 / 4.4119 m`，strict `0.3750 / 8.9109 m`，H1/H3 均 `0.8750`；R3B default `0.7500 / 6.3520 m`，strict `0.6250 / 7.0025 m`，H1/H3 均 `0.7500`，saturation 仍为 `0.4657/0.4818`。两者都没有改善严格的硬环境，不进入 32-seed 或五 seed。

判定：**R1K 跨训练 seed 只有 default 稳定过门禁；strict success `0.4938 < 0.50`、strict RMSE `7.5770 > 6.5 m`、H1 success `0.8875 < 0.95`、H1/H3/H4 saturation 超线，整体仍未通过。R2B/R3B smoke 也失败。** 负结果保留，不把单 seed 的 strict `0.5000` 写成稳定通过。

为避免在已用于筛选的评估 seeds `0–31` 上过拟合，2026-11-04 17:58 又启动了独立的 `32–63` held-out 闭环队列：R0、R1K、R1G 三条并行，均为 5 个 no-DADE 训练 seed × 6 个 tag（`default strict125 h1 h2 h3 h4`）× 32 个评估 seed。队列 PID 分别为 `359152/359366/359542`，产物根为 `artifacts/lowlevel/pc_alos_r_holdout32_{r0,r1k,r1g}_20261104`。`R1G` 与 `R1K` 的差异仅在初速 `1.10×` vs `1.00×`，两者都关闭预旋，用于检验“初始速度非零”的增益是否跨 held-out 评估 seed 稳定。`scripts/run_pc_alos_r_20261104.sh` 已补 `h2` 路由。

`90/90` 个 summary 已完成，配对汇总为 `artifacts/lowlevel/pc_alos_r_holdout32_compare_20261104/summary.{json,md}`。R0 的 held-out default/strict 只有 `0.6562 / 5.545 m`、`0.3937 / 7.750 m`；R1K 为 `0.6937 / 5.994 m`、`0.4000 / 8.831 m`，success 只小幅上升但 RMSE、saturation 变差；R1G 的 `0.7063 / 5.869 m`、`0.3937 / 8.926 m` 同样没有稳定收益。H1/H3/H4 的饱和仍全部超线。**结论是否定的：ALOS-R 在 seeds `0–31` 上的收益没有跨 held-out 复现，停止同配方细调；下一步应改残差策略的闭环状态/奖励或执行器权限，而不是继续调终端、初速或 RPM 系数。**

#### R4 起步预旋与初速校准（2026-11-04，已完成，fail）

逐步 trace 显示最低地速不是单一问题：H1/H3 的部分 run 最低点在 1–3 s，default/strict125 的若干 run 在 7–16 s 的转向段，H1 的错过终点回追段也会出现大航向误差。R4 只隔离起步，不改终端、蟹行、调速或 actor：

- `R4A`：`initial_speed_ratio=1.25`，`initial_propeller_rpm_ratio=0`；
- `R4B`：`initial_speed_ratio=1.25`，`initial_propeller_rpm_ratio=0.75`；
- `R4C`：`initial_speed_ratio=1.25`，`initial_propeller_rpm_ratio=1.0`。

初始速度按实际 episode 巡航速度缩放，并钳位到 REMUS100 最大速度 `2.5 m/s`；H3 在 `2.4 m/s` 巡航时只能保留约 `1.04×` 余量。R4 的 smoke 为 1 个训练 seed × 8 个评估 seed，tags `default strict125 h1 h3`；只有起步最低实际地速改善、success 不回退超过 `0.02`、H1/H3 saturation 不增超过 `0.02`，才扩到 5 seed × 32 seed。实际地速 `>=0.75` 是目标硬约束，报告必须和命令速度分开。命令模板：

```bash
cd /data/yhx/underwater-robotics/src/plan_p1/settled_局部规划器_反应式_alos_r
PY=/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python
ARTIFACT_ROOT="$PWD/artifacts/lowlevel/pc_alos_r_r4_launch_smoke_20261104" \
STAGES="R4A R4B R4C" SEEDS=94030 TAGS="default strict125 h1 h3" \
SEED_COUNT=8 EVAL_WORKERS=4 scripts/run_pc_alos_r_20261104.sh

for stage in R4A R4B R4C; do
  "$PY" scripts/summarize_pc_alos_r.py \
    --eval-root artifacts/lowlevel/pc_alos_r_r4_launch_smoke_20261104/eval \
    --stage "$stage" \
    --output "artifacts/lowlevel/pc_alos_r_r4_launch_smoke_20261104/report/$stage"
done
```

8 个评估 seed 的配对 smoke 已完成：R4A 是唯一通过扩 seed 条件的臂。R4A 相对 R1K 在 default 上 success 相同、RMSE `4.4143→4.0336 m`、实际地速最低比 `0.3737→0.3925`；H1 success `0.8750→1.0000`、RMSE `4.5992→1.6961 m`；H3 RMSE `2.0659→2.0588 m`。但 H1/H3 saturation 仍为 `0.4975/0.5010`，高于 `0.40/0.45`；strict125 仍为 `0.3750 / 8.8869 m / timeout 0.6250`，全部未过正式门禁。R4B/R4C 的 75%/100% 预旋在 default/strict 上明显退化，停止。R4A 随后进入 held-out seeds `32–63` 的 5-seed × 6-tag 队列，结果见下段。

R4A 的 held-out 5-seed 结果已返回：default `0.7188 / 5.663 m / timeout 0.2812`（R1K `0.6937 / 5.994 / 0.3063`），strict125 `0.3937 / 8.945 m / timeout 0.6062`（R1K `0.4000 / 8.831 / 0.6000`），H1 `0.9437 / 3.905 m / sat 0.4859`（R1K `0.9375 / 4.280 / 0.4727`），H3 `0.9750 / 3.343 m / sat 0.5057`（R1K `0.9812 / 3.174 / 0.5047`）。全航线实际地速最低比仍只有 `0.25–0.63`。结论：**R4 fail，不扩 R4B/R4C，不继续调初速/RPM 系数。** 产物：`artifacts/lowlevel/pc_alos_r_r4a_holdout32_5seed_20261104/report/R4A/` 与 `artifacts/lowlevel/pc_alos_r_r4a_holdout32_5seed_20261104/report_vs_r1k/`。

队列入口：`scripts/run_pc_alos_r_20261104.sh`；汇总：`scripts/summarize_pc_alos_r.py`；报告：`artifacts/lowlevel/pc_alos_r_floor75_20261104/report/{R0,R1,R1B}`、`artifacts/lowlevel/pc_alos_r_initfloor_2x2_32seed_20261104/report/{R1G,R1H,R1I,R1J}`、`artifacts/lowlevel/pc_alos_r_r1k_init100_32seed_20261104/report/R1K`、`artifacts/lowlevel/pc_alos_r_r1k_5seed_32eval_20261104/report/R1K`、`artifacts/lowlevel/pc_alos_r_r1k_h4_5seed_32eval_20261104/report/R1K`、`artifacts/lowlevel/pc_alos_r_r2b_r3b_smoke_20261104/report/{R2B,R3B}`。未自动 commit/push。

### S13：DADE 闭环 DAgger（2026-11-04，已完成，整体 fail）

goal13 的 DADE-pose 在严格留出的高速强流格 `grid_hs` 上已有配对正信号：
success `0.9000`（no-DADE `0.8438`）、RMSE `4.90 m`（no-DADE `6.11 m`）。
但 default/strict125 和 H1/H3/H4 的绝对门禁仍未过。S13 不再调 DADE 结构或
超参数，改为用 DADE-pose 学生做一轮闭环 DAgger：在 `lw/ls/hw` 三格执行
student + `0.05` 归一化动作噪声，采集学生访问状态上的 teacher 标签，合并原
demos 后重新监督 DADE 残差；`grid_hs` 继续严格留出，不参与采集或训练。

产物与门禁见 `doc/goals/goal14_dade_closed_loop_dagger.md`。队列入口为
`scripts/run_pc_dade_grid_dagger_20261104.sh`，训练配置为
`configs/lowlevel/odarrl_control_ppo_pc_grid_dade_dagger_frozen_bc_residual_dade_pose_lr3e-4_scale1p0_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json`。
先跑 `seed94030 × 8 eval seed` smoke；通过后跑 `94030–94034 × 32 eval seed`。
这是“利用 DADE + 补闭环状态覆盖”的新实验，不是 DADE-pose 同配方续调。

正式结果已完成：五训练 seed × 32 评估 seed，DADE 基线与 DAgger 各 `110/110`
个评测产物；审计 `failures=[]`、冻结 BC 漂移全为 `0.0`、`grid_hs` 未泄漏。

| Tag | DAgger success / RMSE / timeout / sat | DADE baseline success / RMSE / timeout / sat | 判定 |
|---|---:|---:|---|
| `grid_lw` | `0.9750 / 2.16 m / 0.0250 / 0.1658` | `1.0000 / 1.88 m / 0.0000 / 0.1574` | fail |
| `grid_ls` | `0.6438 / 4.94 m / 0.3563 / 0.1344` | `0.6813 / 4.79 m / 0.3187 / 0.1274` | fail |
| `grid_hw` | `0.9938 / 2.19 m / 0.0063 / 0.4736` | `0.9750 / 3.48 m / 0.0250 / 0.4607` | pass |
| `grid_hs` | `0.8875 / 5.38 m / 0.1125 / 0.4428` | `0.9000 / 4.90 m / 0.1000 / 0.4382` | fail |
| `default` | `0.6625 / 5.10 m / 0.3375 / 0.1244` | `0.6813 / 4.96 m / 0.3187 / 0.1171` | fail |
| `strict125` | `0.4375 / 6.45 m / 0.5625 / 0.1150` | `0.4562 / 6.36 m / 0.5437 / 0.1074` | fail |
| `h1` | `0.9250 / 4.60 m / 0.0750 / 0.4483` | `0.9437 / 3.76 m / 0.0563 / 0.4455` | fail |
| `h3` | `1.0000 / 1.83 m / 0.0000 / 0.5196` | `1.0000 / 1.84 m / 0.0000 / 0.5098` | fail |

`grid_hs` 的 success delta 为 `-0.0125`（配对范围 `-0.0625~+0.03125`），RMSE
ratio `1.1018`，saturation `+0.0046`；没有达到预注册的 `+0.02`/`0.95×`
迁移阈值，绝对 success 和 RMSE 也未过线。DAgger 相对同 run 的零残差 `initial`
确实有学习效果（`grid_hs` success `+0.0438`、RMSE ratio `0.8885`；H3
success `+0.15625`、RMSE ratio `0.1828`），但这些收益已经被 DADE-pose 基线
包含，没有形成增量。H1/H3/H4 的 saturation 仍在 `0.45–0.52`，执行器门禁没有
改善。按预注册规则登记为 **fail**，不追加同配方 DAgger 轮次，不回头调 DADE
dropout/head 或动作噪声来挑 seed。下一步转向实际地速跟踪、执行器权限/饱和奖励
或新的控制层设计。

报告：`artifacts/lowlevel/pc_dade_grid_dagger_report_20261104/report.{json,md}`；
审计：`artifacts/lowlevel/pc_dade_grid_dagger_supervised_residual_20261104/audit.json`。

### S14：实际地速 + 材料饱和奖励修正（2026-11-04 新增）

S10f/S13 说明 DADE-pose 已有 held-out `grid_hs` 的正信号，但继续 DAgger 没有
增量；R1K/R4A 又说明命令速度下限不等于实际地速。S14 因此不再调 DADE 或 ALOS
系数，而是在同一冻结 BC、同一训练 seed、同一 held-out 格上，对已有 S13
no-DADE/DADE-pose 残差做一轮可选奖励修正：

- 新增 `w_route_speed_pc` 和 `route_speed_floor_ratio=0.75`，直接惩罚真实 NED
  地速在路径切向上的不足；该量不使用 `goal_distance`；
- 新增 `w_material_sat_pc`，直接惩罚真实 `material_saturated`，而不是只惩罚
  请求动作越界；
- 两个新增权重默认 `0`，旧配置行为不变；
- S14 arm A 从 S13 no-DADE `final.pt` warm start，arm B 从 S13 DADE-pose
  `final.pt` warm start；不新增残差网络，不改变 actor 输入；
- S14 新环境配置固定 `minimum_speed_ratio=0.75`、20 m 终端航向捕获、
  `terminal_speed_ratio=1.0`、不启用距离制动；起步使用温和非零
  `initial_speed_ratio=1.10`、`initial_propeller_rpm_ratio=0.25`。

执行入口为 `scripts/run_pc_s14_capability_20261104.sh`，完整说明见
`doc/goals/goal15_capability_joint_domain.md`。先跑 `1 seed × 8 eval seed` 的
`grid_hs/default/strict125/h1/h3/h4` smoke；通过后跑 `5 train seed × 32 eval seed`
的 `grid_lw grid_ls grid_hw grid_hs default strict125 h1 h2 h3 h4 c4`。产物根使用
`artifacts/lowlevel/pc_s14_capability_*_20261104/`，不覆盖 S13 结果。

这是一次有界实验：smoke 失败立即停止；smoke 通过只跑一次 5×32；全量 fail 立即
停止；全量 pass 只追加一次 32–63 个 held-out eval seed 复核，复核后无论结果如何
都终止，不再调权重、不补训练 seed、不换 tag。基础设施路径错误不计入实验结论，
但只允许修复后重跑同一个 smoke，不借机扩大实验。

2026-11-04 smoke：首次因独立目录缺少冻结 BC checkpoint 失败；补齐只读软链接后
同一 smoke 完成 1 seed × 8 eval seed × 6 tags。smoke 契约通过，S14-B 的
default/strict125/grid_hs 分别为 `0.8750/3.004 m`、`0.5000/5.396 m`、
`1.0000/1.795 m`；但 H1/H3/H4 的 material saturation 仍为 `0.45–0.52`，高于
门禁。按预注册规则只启动一次 5×32，不用 smoke 指标写正结论。

预注册判定：保留格 success 下降不超过 `0.02`、RMSE 不超过 `1.05×`、实际材料
饱和增量不超过 `0.02`、实际路径切向地速不下降超过 `0.02`；held-out `grid_hs`
success `>=0.85`、RMSE `<=6.0 m`，且相对同 seed 参考至少有 `+0.02` success 或
`0.95×` RMSE。DADE 只有同时满足“相对 no-DADE 有增量”和“绝对门禁”才登记为正结果。

2026-11-04 23:42 全量结果：**S14 有界实验 fail，按协议终止，不启动 32–63
held-out 复核，不补 seed、不调权重、不换 tag。** 10/10 训练和 110/110
`train_seed × arm × tag` 评测全部完成；本地同步报告见
`artifacts/lowlevel/pc_s14_capability_5x32_summary_20261104/summary.md`。

- S14-A no-DADE：default `success=0.6312, RMSE=5.233 m`；strict125
  `0.4125, 6.598 m, timeout=0.5875`；held-out `grid_hs` `0.8313, 6.278 m`；
  H1/H3/H4 分别 `0.9000/0.9625/0.9250` success、`5.013/4.495/6.767 m` RMSE，
  saturation 均高于门禁。
- S14-B DADE-pose：default `0.6250, 5.805 m`；strict125 `0.4062, 7.345 m,
  timeout=0.5938`；held-out `grid_hs` `0.9125, 5.062 m` 通过绝对门禁，且相对
  S14-A 为 `+0.0813` success、`-1.216 m` RMSE；但 default/strict125/H3/H4 退化，
  相对 S13 DADE-pose 的 `grid_hs` 增量只有 `+0.0125` success、`-0.163 m` RMSE，
  未达到预注册的稳定 DADE 正结论标准。
- 两臂 terminal speed ratio 均为 `1.0000`，证明没有终端距离减速回归；strict125
  的主要问题仍是 timeout 和 RMSE，而不是终点前爬行。S14 的 route-speed 奖励没有
  解决 strict125/H1/H3/H4 的执行器饱和与高速跟踪问题。

结论：S14 的 held-out `grid_hs` 正信号不足以抵消其他保留格的失败，不能写成
“能力提升已通过验收”。后续若继续，必须另开有界实验并重新预注册，不能沿用本轮
结果追加调参。

### S15：ABC 前置 P0/P1/P2/P3 与 A/B/C 时间感受野、决策频率、Transformer actor（2026-11-05，进行中：P0–P3、A、B10/B5 已完成；C 待触发判定；S15 未冻结）

对应主文档：`doc/goals/goal9_odarrl_control_reproduction.md` 的 14.4.18。S15 先执行
P0/P1/P2/P3 前置门禁，再执行 A/B/C；每个阶段只改一个变量，禁止把输入、DADE、
奖励、架构、控制频率和历史跨度同时改变后比较：

- **A：时间感受野。** 保持 50 Hz actor 和当前 DADE/MLP 结构，只把历史 stride 从 1 改到
  5/10，使长窗从 1 s 扩到 5/10 s；用于判断当前 1 s 历史是否覆盖不了约 8 s 的海流变化。
- **B：决策频率。** 物理、传感器和安全终止保持 50 Hz，只把 actor 决策降到 10 Hz（先）
  和 5 Hz（后）；用零阶保持和现有速率限制执行动作。B10 必须与 A1 的上下文物理跨度
  相同，B5 必须与 A2 相同。
- **C：actor 信息通路。** 在最佳 A/B 的控制频率和历史采样下，用直接 Transformer actor
  替换 `DADE + MLP`；输入必须包含当前状态、命令和历史 token，先做 BC 蒸馏再有界微调。

#### S15.0 ABC 前置 P0/P1/P2/P3

主文档 14.4.18 的“前置工作”是唯一配方来源；服务器只执行，不另立规则。前置组是
A/B/C 之前的门禁，不是正式性能结论：

当前状态（2026-11-05）：P0 已完成。服务器 smoke 契约通过；screen 三组各完成
`1 × 20` 固定 seed 评估和 `32 × 128 × 256` 训练。A2 在 default 上出现 success `+0.05`
和 3D RMSE 改善 `0.4271 m`，但 `history-zero`/`history-reverse` 均无可解释退化，
按门禁只能算相关性而不能算时间感受野的因果收益，因此 P0 判为 no attributable
signal，P1 基线固定为 `A0_stride1`。产物位于
`artifacts/lowlevel/pc_abc_prework_20261105/{smoke,P0_a_screen}/`。首次 screen 有两个
strict125 历史消融因 CPU 绑定额度超出 384 核失败；修复 worker 槽位后只重跑同命令，
训练预算、评估 seed 和阈值均未改变。

P1 随后完成。smoke 的 `P1_full`/`P1_curriculum` 均完成 32 iterations，training audit
通过，最终 schema 相同且四类路径消融齐全；`P1_curriculum` screen 在 `P1_full` 未通过
归因门禁后由 runner 主动跳过，不是 curriculum 训练失败。正式 screen 使用训练 seed
`91430`、评估 seeds `1000–1019` 和 `32 × 128 × 256` 预算；相对 `A0_stride1`，default
success delta 为 `+0.0500`、3D RMSE 改善 `0.4308 m`，但 `path-zero` 的 success 下降为
`-0.0`、RMSE 仅恶化 `3.13e-05 m`、动作 L2 平均变化约 `4.70e-05`。因此 primary signal
存在但路径归因为 false，`stage_decision=no_path_attribution`、`selected_arm=null`、
`next_action=stop_path_branch_and_write_negative_result`。完整证据见
`doc/goals/goal9_p1_path_observation_report.md`。下一步 P2 回到 P0 `A0_stride1`，不再
继续增加路径变量；P2a 复用 P0 A0，P2b 只比较 base-state raw-history 与 encoded-history
DADE，并必须记录 latency 和 `history-zero`。

- **P0：A 组短筛查。** 只跑 `A0_stride1/A1_stride5/A2_stride10` 的 smoke 和 1 个训练
  seed × 20 个评估 seed；不先跑 A 的 3×32 全量。若无信号，后续以 A0 为基线。
- **P1：路径观测 pilot。** 以 P0 最佳 A 为基线，把 6–8 维路径特征作为 `o_t^a` 的
  一部分送入 actor encoder，得到包含路径信息的 `z_t^a = E_a(o_t^a)`；DADE 和奖励
  不变。路径列复用编码器预留输入槽或作为独立 encoder 输入 block，但不得绕过 encoder
  直接进入 actor 顶层或 DADE。可选 encoder 内部融合
  `z_t^a = f_base(o_base_t) + alpha * f_path(path_t)`，`alpha` 用小非零初值或保持
  零初始化输出层但非零 scale，避免 `alpha=0` 的路径编码器死梯度；不要直接 append 到
  12D state。必须运行 `path-zero` 消融，在送入 encoder 前把 `o_t` 的路径列置零并同时
  作用于当前/历史编码，而不是在最终 `z_t` 上做减法；路径特征只能由
  `sensor_output.p_nav` 投影得到。
- **P1 的模块化 block。** 至少拆分 `state_base`、`path_geom`、`path_context` 和默认
  关闭的 `path_velocity`；每个 block 带 `active_mask`、归一化、schema hash 和固定
  `z_t` 投影槽。P1 允许 `P1-full`（一次性全开）和 `P1-curriculum`（阶段性启用）
  两种模式；课程顺序为 `state_base` 热启动 → 启用 `path_geom` → 启用
  `path_context` → 可选 `path_velocity`。两者必须使用相同总步数、seed 和更新次数，
  最终阶段启用与 `P1-full` 相同的 block 集合；阶段切换点预注册，只改 `active_mask`，
  不能改变 actor/DADE 顶层宽度，也不能看评估结果调整阶段。课程学习不得与 P2 的
  DADE-z 或 P3 的奖励改动同时进行。
- **P2：DADE-on-z_t 消融。** 以 P1 胜出配置为基线，比较 raw-history DADE 与 z-history
  DADE；side channel 保持原值，历史编码在 policy 内完成。当前 DADE 的 current token 被
  `del current`，因此 P2b 是历史 token 编码，不是 current-z 融合；current-z query 只能
  另立 P2d。P1 通过后 `path_t` 已在 `o_t` 中，P2b 的 z-history token 必须使用同一个
  包含路径的 `z_t`；DADE 不得额外接收 raw `path_t`。`dade_state_mode='pose'` 的按列
  语义在 z 编码后失效，必须重新定义或关闭。
- **P3：奖励 pilot。** 只有 P1/P2 显示路径信息被使用但仍有明确误差时启动；只做一个
  固定小权重的 clipped 路径误差奖励，不加入 goal/progress/timeout bonus，不扫描权重。

**P2 服务器结果（2026-11-05，完成）：** `no_signal`。相对 raw-history `A0_stride1`，
default/strict125 success delta 均为 `0.0`，3D RMSE 改善分别为 `0.002811 m` 和
`0.018726 m`；`history-zero` 无 success 下降且 RMSE 没有恶化，不构成历史归因；
p95 latency `1.066 ms` 通过 20 ms 门禁。按协议保留 `A0_stride1`，停止 DADE-z 重构；
P3 记为 `not_applicable`（P1 `no_path_attribution` 且 P2b `path_dim=0`）。证据位于
`artifacts/lowlevel/pc_abc_prework_20261105/P2_dade_z/report/stage_decision.json`。

执行规则：

- P0/P1/P2/P3 都是 1 个训练 seed × 20 个固定评估 seed（`1000–1019`）；只有最终唯一
  候选才允许进入 3×32 和 CPU PVS 64-seed；
- 每个阶段通过主文档 14.4.18 的门禁后才进入下一阶段；fail 时写负结果并停止该分支，
  不得用加步数、换 seed、调阈值或混入第二个变量来救结果；
- P0/P1/P2/P3 未完成或未记录负结果前，不启动 A/B/C 的正式 3×32；A/B/C 不能借用
  “GPU 空闲”跳过依赖门禁；
- P1/P2/P3 的路径特征、schema hash、归一化、checkpoint metadata 和单测必须一起提交，
  不能只改运行时张量；旧 BC 必须能通过编码器输入扩展或明确的 checkpoint 迁移协议复现。
- P1 必须验证 `path_t` 只存在于 `o_t` 和编码后的 `z_t` 中，并检查不存在绕过 encoder
  的 actor/DADE 路径；P2 的 raw/z-history 比较必须使用同一路径输入契约。

前置产物根目录：

```text
artifacts/lowlevel/pc_abc_prework_20261105/
  P0_a_screen/
  P1_path_obs/
  P2_dade_z/
  P3_path_reward/
  smoke/
  report/
```

每个阶段产出 `summary.json`、`summary.md`、manifest、配置/checkpoint hash 和 finite
audit；大型 checkpoint、完整 rollout 和 event 文件只留服务器。前置组通过后才允许在
`artifacts/lowlevel/pc_abc_temporal_20261105/` 启动 A/B/C。

#### 共同契约

- 主环境固定为 `odarrl_control_pc_c3_random_realistic_v1_rewardscale001.json`；C2 mild
  只做回归，不作为主判据。
- episode 物理时长固定为当前 50 s。50 Hz 为 2500 个物理步；10 Hz 为 500 个 actor 决策，
  5 Hz 为 250 个；安全终止仍可在每个物理步触发。
- 动作在决策间隔内零阶保持，并继续经过现有速率限制、幅值和 material saturation 检查；
  reward、return、gamma 和 timeout 必须按物理时间换算，禁止把动作保持期的奖励重复累加。
- 所有 arm 使用同一 BC/教师数据、同一训练物理秒数、同一更新次数和同一评估 seed 集；
  不能给 A/B/C 不同预算后把差异归因于方法。
- actor 不得读取真实海流、故障内部状态或仿真特权量；critic 特权输入不得进入部署路径。
- 任何 Torch 结果只能作为预筛；最终候选必须经过 CPU PVS 64-seed 复核。

#### 磁盘与入口

所有新结果写入独立根目录，禁止覆盖 S0–S14：

```text
artifacts/lowlevel/pc_abc_temporal_20261105/
  A0_stride1/
  A1_stride5/
  A2_stride10/
  B10/
  B5/
  C_transformer/
  smoke/
  screen/
  formal/
  final_pvs64/
  report/
```

预计新增统一入口（实现并通过 smoke 后才允许启动）：

```text
scripts/run_pc_abc_prework_20261105.sh
scripts/summarize_pc_abc_prework.py
scripts/run_pc_abc_temporal_20261105.sh
scripts/summarize_pc_abc_temporal.py
```

如果入口或配置尚不存在，先提交契约、单元测试和 1-iteration smoke；不得用临时 Python
命令直接启动正式训练。

#### 执行阶梯

1. **同步与测试**
   - 同步本地代码到服务器；服务器目录不是 Git 仓库，Git 操作只在本机；
   - 运行 PC 单测、ruff 和 PC schema/import smoke；
   - 确认服务器 `artifacts/lowlevel/pc_abc_temporal_20261105/` 为空或为全新目录。

2. **契约 smoke（A0/A1/A2）**
   - 1 个训练 seed × 8 个评估 seed；
   - 逐项检查物理时长、决策计数、reward 时间尺度、动作保持、历史采样、no-privilege-leak
     和 finite audit；
   - 任一契约失败只修实现并重跑同 smoke，不扩大 seed 或训练预算。

3. **A 组首轮筛查**
   - 1 个训练 seed × 20 个固定评估 seed（`1000–1019`）；
   - `A0_stride1` 为同 seed 基线，比较 `A1_stride5`、`A2_stride10`；
   - 通过阈值：success 或 strict125 至少 `+0.05`，或 3D RMSE 至少改善 `0.10 m`；
     safety 增幅不超过 `0.02`，timeout 增幅不超过 `0.05`；material saturation 仅报告；
   - 同时运行 `history-zero`/`history-reverse`，不能只凭总指标宣称历史有效。

4. **A 组正式比较**
   - 首轮有可解释信号且无灾难性退化时，扩到 3 个训练 seed（`94030–94032`）× 32 个评估
     seed（`0–31`）；
   - 逐 seed 配对；报告 mean/std/worst、失败尾部、timeout、safety、RMSE、saturation 和
     历史消融动作敏感度；
   - 正式门禁与主文档 14.4.18 相同。

5. **B 组决策频率**
   - 先跑 `B10`（10 Hz actor、50 Hz 物理、5 s 上下文），与 `A1_stride5` 配对；
   - B10 的门禁是非劣：success/strict 降幅不超过 `0.02`、RMSE `<=1.05×`、
     timeout 增幅不超过 `0.02`，并记录 p50/p95 时延和决策计算量；material saturation
     只报告；
   - 只有 B10 通过，才允许跑 `B5` 并与 `A2_stride10` 配对；否则停止降频分支。
   - **2026-11-05 结果：** B10 正式 screen `pass_non_inferior`；default success/strict
     降幅 `0.0`、RMSE ratio `1.0014`、latency gate 通过（p95 `0.8907 ms`、compute
     ratio `0.1186`）。B5 正式 screen `fail_non_inferior`；default/strict125 success
     下降 `0.05`、RMSE ratio `1.1153`、timeout 增加 `0.05`。停止 5 Hz 分支，不重跑。

6. **C 组 Transformer actor**
   - 只有 A/B 显示时间尺度有影响且 `DADE + MLP` 仍有明确通路瓶颈时才启动；
   - 当前 A 无历史因果归因、B10 仅效率非劣、B5 fail，默认记录
     `C_not_triggered`；除非已有 artifact 明确满足上述瓶颈条件，否则不训练 C；
   - 先监督蒸馏 BC/最佳基线动作，保证初始闭环；再做一次有界微调；
   - 与相同频率、相同历史跨度、相同训练预算的最佳 A/B 配对；
   - 参数规模和 p95 时延与 DADE+MLP 同量级；若无稳定收益立即停止，不追加层数/head。

7. **最终复核**
   - 只有正式比较通过的单一候选进入 CPU PVS 64-seed；
   - 不根据评估 seed 反向挑 checkpoint；不在最终复核前改写门禁；
   - 通过后写 `doc/goals/goal9_abc_temporal_report.md` 和服务器小型 summary；否则写负结果
     报告并停止该分支。

#### 队列与资源规则

- S15-P0/P1/P2/P3 是当前 PC 最高优先级；新 DADE/DAgger、S14 重复轮次和无关大扩训让路；
- S16 的正式训练不得与 S15 的正式 A/B/C 或最终复核并行；只有 S15 冻结唯一胜出配置后，
  S16 才能启动。S16 的代码、配置和 BC 数据审计可在 schema 冻结后提前准备，但不得提前训练；
- P0/A0/A1/A2 可以在独立 GPU 上并行 smoke，但正式 arm 必须保证同样的 GPU/CPU 软硬件
  条件和独立输出目录；
- P1/P2/P3 只能按依赖顺序启动；前置组报告和 schema 冻结前，A/B/C 正式组不得占用资源；
- B/C 只能在 A 组报告产出后启动，不能用“GPU 空闲”作为跳过依赖门禁的理由；
- 每个 run 记录 GPU 型号、显存、torch/CUDA、配置 hash、checkpoint hash、训练 seed、
  评估 seed 和 wall-clock；
- 大型 checkpoint/rollout 只留服务器；Git 只提交代码、配置、小型 summary 和报告。

#### 预期判定

- P0 完成：得到后续 P1/P2/P3 的统一基线；A1/A2 无信号时以 A0 继续，不跑旧 schema 全量；
- P1 通过：说明 actor 确实缺少路径相对信息，且 `path-zero` 有因果退化；否则停止路径分支；
- P2 通过：说明 z-history DADE 相对 raw-history 有稳定增量；否则保留 raw-history；
- P3 通过：说明小权重路径奖励带来闭环收益；否则保留原 `pc_command_track`；
- A 通过：说明 50 帧 @ 50 Hz 的 1 s 历史确实太短，后续优先保留更长历史跨度，不必先降频；
- B 通过：说明 10 Hz（或 5 Hz）actor 能在保持性能的同时降低决策/推理成本，可以进入
  分层控制实现；
- C 通过：说明真正瓶颈在 DADE+MLP 的信息通路，而不是时间跨度；可把直接 Transformer
  作为下一阶段候选；
- 任一失败：保留负结果，停止该变量，不用追加训练步数或换 seed 掩盖。

### S16：V 系列（V-0D 纯路径主臂；旧 ALOS-free 细节仅作历史）

**2026-11-06 用户最终修正，优先于本节旧定义：** 直接以 `V-0D` 纯路径输入为主臂，
`V-3D` 只作可选诊断控制。不再把 `V-3D → V-2D → V-1D → V-0D` 的命令逐级消融作为
必经主线。

```text
V-0D（主臂）： [] + path observations
V-3D（可选）： [psi_d, theta_d, u_d] + path observations
```

V-0D 的 actor 输入为 `base_state + 8D path`，命令维度为 0；路径只通过 path encoder
融合进 `z_t`，不进入 encoder/DADE/history。当前速度状态可以保留，删除的是目标命令
`u_d`。动作层必须是冻结 BC + 动作残差，不是 BC 热启动的完整策略。

**技术边界：** 不能用现有 3D BC checkpoint 加 `command_mask` 冒充 V-0D。冻结 BC 的
`base_action()` 仍读取 actor command（`src/auv_lowlevel/batch/pc_policy.py:825`），
把命令置零会同时拔掉 BC 的 `psi_d/theta_d/u_d` 输入。必须新增同契约的 0D BC：
`state + 8D path -> action`，训练后冻结，再从零初始化动作残差。当前 BC 侧仍是 3D
命令契约且示范数据没有 path 字段，需要非破坏性扩展。

指标口径（2026-11-06 用户确认）：3D RMSE 是第一主指标；success、strict125、timeout
和 safety 作为护栏，默认允许的绝对恶化不超过 `0.02`。路径归因以 `path-zero` 后 3D
RMSE 至少恶化 5% 为主要证据，不能只看动作 L2 变化。

权威计划：`doc/goals/goal9_v0d_direct_pure_path_plan_20261106.md`。以下 2026-11-05
原文保留为历史，不再作为新实验的输入契约，也不适用于 V-0D 主臂。

对应主文档：`doc/goals/goal9_odarrl_control_reproduction.md` 的 14.4.19。S16 在 S15
冻结唯一基线后，移除 ALOS 的 heading/pitch 制导，让 actor 由包含路径信息的
`z_t = E(o_t)` 输出执行器；V1 再移除显式 `u_d`，由 RPM 和奖励隐式学习速度。S16
不重跑 S15 的 A/B/C，也不把 V0/V1 的失败解释成 ALOS 外置方案失败。

**最终状态（2026-11-05）：** S15 冻结后已实现并执行 V0；修正路径支路后，正式 screen
相对 `A0_stride1` 的 default/strict125/grid_hs success 分别为 `0/20`、`0/20`、
`0/20`，三组共 `60/60` timeout，判定 `fail_v0`。material saturation 仅作为风险记录，
不参与判定；即使忽略 saturation，success/timeout/RMSE 仍远低于门禁。按预注册规则，
V1/V1b/V2/V3 均未启动，S16 候选为空，部署方案保留 ALOS 外置 + S15 `A0_stride1`。
详见 `doc/goals/goal9_v0_report.md` 和 `doc/goals/goal9_final_report.md`。

**启动前置条件**

- S15 的 P0/P1/P2/P3 已完成并留下正结果或负结果；A/B/C 已按门禁结束；
- 已冻结唯一 S16 基线：最佳 A/B/C 组合；若 C 失败，则用最佳 A/B 的 `DADE + MLP`，若
  A 也无信号则用 A0；在该基线上记录配置 hash、checkpoint hash、归一化 schema 和评估 seed；
- S16 不要求 A/B/C 每一臂都有正收益，但禁止在 S16 内重新比较 A/B/C，也禁止根据 V 的
  评估结果反向挑 S15 checkpoint；
- 已确认 actor 只使用可部署观测；ALOS 的 `psi_d/theta_d` 和 V1 的 `u_d` 不得以任何
  side channel 进入学生 actor；真实海流、故障标签和仿真真值不得进入 actor。

**共同契约**

- 主环境固定为 `odarrl_control_pc_c3_random_realistic_v1_rewardscale001.json`；V0/V1
  首轮为 1 个训练 seed × 20 个评估 seed（`1000–1019`），最终唯一候选才进入 3×32 和
  CPU PVS 64-seed；
- 输出根目录固定为 `artifacts/lowlevel/pc_alos_free_v_20261105/`，V0/V1/V1b/V2/V3
  各自使用独立子目录，不覆盖 S15 或历史结果；
- **BC 数据：** 初期允许二选一或混采 `ALOS + PD/PI` teacher 的闭环轨迹与中低速
  teacher/策略轨迹；两类数据都必须记录来源、速度分布、曲率段、恢复状态和 episode
  筛选规则。BC 标签是执行器动作 `[rudder, stern, rpm]`，不是 `psi_d/theta_d/u_d`；
- **初始状态：** 不再假设“初速 = 0、RPM = 0”。训练和评估都必须覆盖非零初始纵荡速度与
  非零起步 RPM；至少包含路径对准起步、横向/垂向偏差起步、航向/俯仰偏差起步，以及
  低速、中速和非零 RPM 起步。初始状态属于环境，不作为 actor 的特权观测；
- **奖励：** 安全/timeout/速度包线为硬约束；路径误差、进度和任务时间为主目标；速度
  平滑高于动作平滑，能耗尽量优化；material saturation 只报告，不作为约束；不加入过冲
  惩罚。过冲只记录为诊断指标，不进入 reward；
- **终端：** 不要求终端减速；success、timeout、捕获半径和安全终止沿用既有口径；
- **路径与 DADE：** V0/V1 的 `o_t` 包含路径特征，当前 actor 使用编码后的 `z_t`；
  若启用 DADE，历史 token 也只能使用同一套包含路径的 `z_t`，不得另开 raw `path_t`
  旁路；路径特征不直接拼接进 actor 顶层；
- **模块化观测复用：** V0/V1 复用 P1 的 block 和 `active_mask`，并把启用状态写入
  checkpoint metadata；V0/V1 的最终观测 schema 必须一致，V1 只删除 `u_d`，不能在
  同一实验里同时增删路径 block；
- **默认不引入 speed head。** V0/V1 的 actor 输出始终是 `[rudder, stern, rpm]`，
  不把速度变成独立动作头，也不硬编码“曲率→目标速度”规则。

#### V0：ALOS-free + 固定速度参考

目的：先只移除 ALOS 的 heading/pitch 制导，保留固定速度参考，隔离“纯路径控制是否
成立”这一件事，不混入速度学习。

- actor 输入：`z_t = E(o_t)`，其中 `o_t` 由机器人状态和路径特征组成；固定 `u_ref`
  作为显式输入；输出：`[rudder, stern, rpm]`；
- 路径投影只能来自 `sensor_output.p_nav`；不使用 ALOS 生成的 `psi_d/theta_d`；
- BC 可以来自 ALOS teacher 或中低速轨迹，但学生训练与评估观测必须一致；
- 奖励以路径误差、进度/时间为主，固定速度跟踪、速度平滑、能耗、动作平滑和安全项为辅；
- 先跑契约 smoke（1 个训练 seed × 8 个评估 seed），检查非零初速/初 RPM、路径特征、
  动作包线、reward 时间尺度和 finite audit，再跑 1×20 筛查；
- 相对最佳 ALOS-external S15 基线，通过条件为：success 或 strict125 至少 `+0.05`，或
  3D RMSE 至少改善 `0.10 m`；time-to-goal 不超过 `1.05×`；safety 增幅不超过 `0.02`，
  timeout 增幅不超过 `0.05`；material saturation 只报告。

**退出规则：** V0 未通过则停止 V 分支，不启动 V1；记录 ALOS 外置仍是更可靠方案的
负结果，不用加步数或换 seed 救回。

#### V1：删除显式速度参考，RPM 隐式学速度

目的：在 V0 证明纯路径控制成立后，让 actor 通过 RPM 和奖励学习速度，而不是依赖显式
`u_d` 或 speed head。

- actor 输入：`z_t = E(o_t)`，其中 `o_t` 由机器人状态和路径特征组成；不含 `u_d`；
  输出：`[rudder, stern, rpm]`；
- 路径切向、曲率、前视和剩余距离进入 `o_t` 并编码进 `z_t`，但不在奖励中硬编码
  曲率-速度映射；由策略自己学习转弯减速；
- BC 可用 ALOS 轨迹或中低速轨迹；没有 `u_d` 时，教师速度行为通过 RPM 标签隐式蒸馏，
  后续再做有界 PPO 微调；
- 速度包线固定 `u_min/u_max`、RPM 上限；material saturation 继续测量并报告；不能为了
  “自己学速度”而放弃最低舵效速度或安全上限；
- 与 V0 同 seed 配对。V1 通过至少要求：success/strict 降幅不超过 `0.02`，3D RMSE
  `<=1.05×`，time-to-goal `<=1.05×`，timeout/safety 增幅不超过 `0.02`；同时记录速度
  标准差、`Δu` p95、RPM²/能耗和转弯段速度-曲率关系，不能以速度平滑改善掩盖 primary
  指标退化；material saturation 只报告。

**分支规则：** 若 V1 的路径与安全指标通过，但转弯段明显不减速，才允许进入 V1b；若
V1 的 primary 指标失败，记录负结果并保留 ALOS 外置方案。

#### V1b：可选曲率软约束（仅在有证据时）

只有 V1 在转弯段明显不减速、且不是路径精度/时间目标的合理高速时才启动。只允许加一个
很小的曲率软约束：

```text
r_turn = -w_turn * relu(u - u_turn_max(kappa))
```

`w_turn` 必须小，不得压过路径精度和时间目标；V1b 与 V1 同 seed 配对，并检查转弯段
速度、success/strict、RMSE、time-to-goal 和 saturation。若只是降低速度但 primary
指标退化，判 fail，停止该分支。

#### V2：可选 speed head（默认不启动）

只有 V1 出现严重速度振荡、速度不可解释、速度包线频繁越界或路径/速度控制互相干扰，
且复核确认不是奖励或观测问题时，才允许测试：

```text
actor -> [rudder, stern, u_cmd] -> speed controller -> rpm
```

启动前必须写清触发证据（例如 `Δu` p95 或速度包线违反显著高于 V0、且 V1b 无法修复）
以及为何不是奖励/观测缺陷。V2 是独立 arm，不能与 V1/V1b 同时改动；默认不启动。
speed head 会改变动作空间、控制器、BC 标签和 checkpoint，属于中等架构改动。

#### V3：正式比较

最终至少比较：`ALOS + PD/PI` teacher、V0、V1；V1b/V2 只有在各自前置证据成立时才加入。
正式门禁沿用 3 个训练 seed × 32 个评估 seed 和 CPU PVS 64-seed。主指标为 success、
strict125、横向/垂向 RMSE、time-to-goal、timeout 和 safety；material saturation 作为
信息性风险指标记录；次级指标记录速度标准差、`Δu` p95、RPM²/能耗、动作变化率、转弯段
速度-曲率关系和过冲诊断。所有结果按训练 seed 配对，报告 mean/std/worst 和失败尾部；
不得根据评估 seed 反向挑 checkpoint。

#### 执行阶梯

1. **实现与审计**：完成 V0/V1 配置、路径特征、初始状态采样、无速度头断言、无过冲奖励
   断言、BC 数据契约和 finite audit；运行 PC 单测、ruff 和 schema/import smoke；
2. **V0 smoke**：1 个训练 seed × 8 个评估 seed，逐项检查非零初速/初 RPM、路径特征、
   动作包线、reward 时间尺度和 no-privilege-leak；失败只修实现并重跑同 smoke；
3. **V0 筛查**：1 个训练 seed × 20 个评估 seed，与 S15 最佳 ALOS-external 基线配对；
   通过后冻结 V0 配置和评估口径；
4. **V1 筛查**：在 V0 通过后启动，1×20 同 seed 配对；未通过则记录负结果并停止；
5. **V1b 可选分支**：仅在 V1 显示转弯不减速且 primary 指标仍可接受时启动；否则跳过；
6. **V2 可选分支**：仅在 V1 速度行为“特别糟糕”且已有书面证据、复核排除奖励/观测
   问题时启动；否则跳过；
7. **V3 正式复核**：只对单一候选跑 3×32 和 CPU PVS 64-seed；通过后写
   `doc/goals/goal9_alos_free_speed_report.md` 与服务器小型 summary，失败则写负结果。

#### 队列与资源规则

- S16 只能在 S15 冻结唯一胜出配置后启动；S16 正式训练不得与 S15 正式 A/B/C 或最终
  CPU PVS 并行；
- V0→V1 严格串行；V1b/V2 是可选分支，前置证据不成立时直接跳过；
- 每个 arm 使用独立 GPU、CPU 核范围和输出目录，记录 GPU 型号、显存、torch/CUDA、
  配置 hash、checkpoint hash、训练 seed、评估 seed 和 wall-clock；
- V3 的 CPU PVS 64-seed 只使用冻结候选，不在评估集上继续调参；大型 checkpoint、
  rollout 和 event 只留服务器，Git 只提交代码、配置、小型 summary 和报告。

#### 预期判定

- V0 通过：纯路径控制成立，可以进入 V1；V0 失败则停止 V 分支；
- V1 通过：RPM 隐式速度可接受，优先保留 V1，不必引入 speed head；
- V1b 通过：曲率软约束只用于修复转弯减速，不推广为默认规则；
- V2 通过：只有在 V1 特别糟糕的情况下，speed head 才成为候选；失败则移除该分支；
- V3 通过：ALOS-free 候选在 3×32 和 CPU PVS 64-seed 上稳定优于或非劣于外部 ALOS
  teacher，并可进入正式报告；否则保留 ALOS 外置，不用评估 seed 反挑 checkpoint。

## 5. Git 与提交规则

- 每次只提交一类产物，禁止把其他已有修改一并提交；
- 当前本地仓库里 `settled_局部规划器_反应式/src/auv_lowlevel/` 整体尚未被 Git 跟踪；首次提交必须把该包作为一个可导入的整体提交，不能只提交 `pc_policy.py` / `pc_bc.py` 而漏掉 `pc_normalization.py`、`pc_env.py`、`checkpoint.py` 等依赖，否则提交无法运行；
- 服务器目录不是 Git 仓库，服务器只做 rsync 同步；Git 提交和 push 只在本机执行；
- 本次改动的文件已写入工作区但本代理没有自动 commit；需要时按 S8 末尾的文件级命令执行；
- 代码提交前运行：

```bash
git diff --check
PYTHONPATH=src:../settled_全局规划器/src .venv/bin/pytest -q tests/test_lowlevel_pc_*.py
.venv/bin/ruff check src/auv_lowlevel tests scripts
```

- 服务器结果先写 `summary.json`、`summary.md`、manifest 和哈希，再决定哪些小型摘要进入 Git；
- 大型 checkpoint、完整 rollout、TensorBoard event 只留服务器；
- 每完成一个阶段更新 goal9 状态和本文勾选项，失败也保留记录。

## 6. 执行顺序与最终状态

> 2026-11-05 起，以下旧队列只作为历史记录；当前以 S15 的
> P0→P1→P2→P3→A→B→C，再到 S16 的 V0→V1→V1b→V2→V3 顺序为准。前置组完成前，
> 暂停新的 DADE/DAgger、S14 重复轮次和无关扩训；P0/P1/P2/P3 必须先于 A/B/C 正式组
> 完成，A/B/C 胜出配置冻结前不得启动 V 的正式训练。

> **2026-11-05 最终状态：** 第 1–10 步已完成；第 11 步 V0 已执行并判为
> `fail_v0`；第 12–14 步因 V0 未通过按协议未启动；第 15 步失败退出规则已执行；
> 第 16 步历史项不阻塞本次结论。

1. 同步并运行服务器 PC 单测、ruff 和 S15 契约 smoke；
2. 启动 S15-P0：A0/A1/A2 的 1 seed × 20 eval seed，逐 seed 配对；不先跑 3×32；
3. **S15-P0/P1 已完成**：P0 为 no attributable signal；P1 的 `P1_full` 有正指标但
   `path-zero` 无可解释退化，stage decision 为 `no_path_attribution`，路径分支停止，
   不重跑 P1，也不继续加路径特征；
4. **已完成** S15-P2：P2b 相对 raw-history 为 `no_signal`，保留 `A0_stride1`；
5. **已完成判定** S15-P3：P1 无路径归因、P2b 无路径输入，正式记录为
   `not_applicable`；
6. **已完成** P0/P1/P2/P3 与 schema 冻结，随后完成 A/B/C 正式组；A 组无因果信号，
   未扩到 3×32；
7. **已完成** B10 非劣筛查；B10 通过后执行 B5，B5=`fail_non_inferior`，停止 5 Hz；
8. **已完成判定** C：A/B 未提供可审计的通路瓶颈，记录为 `C_not_triggered`；
9. **未触发** 正式 CPU PVS：没有 A/B/C 候选通过正式比较，因此没有候选进入该复核；
10. **已冻结** S15 唯一基线为 `A0_stride1`，配置/checkpoint/schema hash 见
    `doc/goals/goal9_s15_freeze.md`；
11. **已完成并失败** S16-V0：ALOS-free + 固定速度参考完成 smoke 与 1×20 screen，
    三组评估 `60/60` timeout，判为 `fail_v0`；
12. **未启动** V1：V0 失败，按协议停止整个 V 分支；
13. **未启动** V1b/V2：前置分支不成立；
14. **未启动** V3：没有通过的单一候选；
15. **已执行** 失败退出规则：保留 ALOS 外置方案，不换 seed、不加步数、不调阈值；
16. 历史 S0–S14 的未完成项不阻塞 S15/S16，也不在 S15/S16 进行期间抢占 GPU/CPU。
