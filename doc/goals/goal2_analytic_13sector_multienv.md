# 任务：P1 解析 13 扇区多环境训练

记录日期：2026-10-28

状态：**ACTIVE**

取代：`doc/goals/goal1.md`。旧文件冻结为单环境历史任务书，旧 C1–C5、SAT、R0–R4 证据继续保留，但不再决定新 campaign 的执行顺序。

工作目录：`/home/bubble/水下机器人/settled_局部规划器_反应式`

## 1. 唯一目标

把 P1 训练从“单环境串行 + OceanSim 每步渲染和点云回读”改为“解析 13 扇区 + PyTorch 张量向量环境 + 单个 DDPG 策略并行采样”，在保持 17 维观测、动作范围、奖励和成功判定不变的前提下，先证明语义等价，再证明端到端吞吐提升，最后用新 campaign 重新筛选 P1 配置。

P1 策略的最终评估仍使用 OceanSim P1-A 单环境链路。本轮不进入 P2，不启动真实 OceanSim 多环境声呐训练，不修改全局规划器。

## 2. 变更原因

- 单环境 P1 campaign 已产生 C1–C5、SAT0–SAT7 和 R0–R4 证据，但没有配置达到可用策略门槛；继续逐个跑 20k 的墙钟成本不适合扩大筛选。
- 解析 13 扇区可以直接在 GPU/CPU 张量上批量计算，避免每环境一次 OceanSim 渲染、`scan()`、点云回读和 NumPy 分箱。
- Cartpole 的 4096 环境只证明物理调度能力，不证明 OceanSim 声呐吞吐；因此不再把提高 Cartpole 环境数作为 P1 的前置工作。
- 这条路线不是“假 OceanSim 结果”：训练和评估使用不同的感知来源，必须通过配对误差、来源标记和最终 OceanSim 评估分开报告。

## 3. 冻结契约

- P1 仍是二维水平运动学，固定深度，无垂向动作。
- 动作仍为 `[u_dot, r]`，范围 `±0.5 m/s²`、`±15°/s`，RL 步长 `0.2 s`，积分步长 `0.02 s`。
- Actor 观测仍为 17 维：13 个前视扇区距离 + 目标距离 + 目标相对艏向 + 纵向速度 `u` + 艏摇角速度 `r`。
- 扇区仍为 130° 前视场、13 个 10° 扇区；局部有效视野 `20 m`，原始物理量程 `120 m`。
- 解析传感器仍使用 1300 条 `0.1°` 射线、艏部原点 `[0.8, 0, 0] m`、垂直切片 `±0.5 m` 和“扇区内取最小距离”规则。
- 归一化、episode 超时、碰撞、越界和目标成功定义保持不变。
- 训练和评估都必须记录 `observation_source`、观测规格、动作规格、配置哈希和代码哈希。
- 不通过缩小动作范围、放宽成功阈值或引入 oracle 到评估路径来提高成功率。

## 4. 训练与评估感知

```text
训练：场景几何 + 状态 → 解析 1300 条射线 → 13 扇区 → 17 维观测
评估：OceanSim 点云 → 1300 条射线 → 13 扇区 → 17 维观测
```

- 训练来源固定标记为 `analytic_13sector`。
- 评估来源固定标记为 `oceansim_simplified_pcl`。
- 两种来源必须共用 13 扇区压缩和 17 维 observation builder，不得各写一套归一化。
- 解析后端首版支持 S1 空场景和 S2 圆柱；更复杂障碍必须显式实现并测试，不能用“最近障碍中心距离”偷换扇区射线距离。

## 5. 等价性验收

在开始长训练前，必须采集同一批状态的配对观测：

- S1 空场景不少于 100 个位姿。
- S2 单圆柱不少于 400 个位姿，覆盖起点、接近、绕行、扇区边界和 2–20 m 距离。
- 每个位姿同时记录 OceanSim P1-A 的 13 个扇区距离和解析 13 个扇区距离。
- 记录机器人状态、场景、初始目标距离、声呐配置哈希和代码哈希。

工程门（初版，只有证据充分时才允许修改）：

- 13 扇区距离平均绝对误差 `MAE <= 0.10 m`。
- 绝对误差 `P95 <= 0.25 m`。
- 最大绝对误差 `<= 0.75 m`，且不得出现“真实有障碍但解析扇区判为空”的危险漏检。
- S1 解析输出与 P1-A 一样全部饱和到局部 `20 m`。
- 17 维观测顺序、归一化和 dtype 完全一致。

未通过时先修几何求交、坐标、射线方向或扇区边界，不得通过调整策略或放宽门来掩盖感知差异。

## 6. 向量化架构

```text
N 个独立 episode 状态（张量）
  → 解析 13 扇区 + 17 维观测
  → 单个 DDPG Actor（每环境独立 history/GRU hidden）
  → N 组 [u_dot, r]
  → 批量二维积分、碰撞、越界、奖励与自动 reset
  → 张量化 replay 与批量 PPER 更新
```

必须满足：

- 每个环境有独立 `env_id`、`episode_id`、`episode_step`、状态和终止原因。
- GRU history 和 hidden state 按环境隔离；任一 episode reset 只清空该环境。
- NoisyNet 噪声按环境独立采样；若改为共享噪声，必须另立消融。
- Replay 序列只能来自同一环境、同一 episode；不能跨环境或跨 reset 拼接。
- 区分 `terminated`、`truncated` 和自动 reset；新 episode 的观测不能写回上一条 transition。
- PPER 支持批量 add 和批量 update priorities；beta 按梯度步更新。
- 默认保持 `update_to_data_ratio=1.0`，不能只因为环境数增加就稀释梯度更新；任何比例变化必须作为预注册消融。

## 7. 阶段与门禁

### G0：文档和协议

- [x] 冻结旧 `goal1`，建立本文件。
- [x] 更新 `doc/01_技术栈.md` 和 `doc/02_执行计划.md`。
- [x] 明确训练、评估和配对标定的来源标记。

通过条件：三份文档和本目标没有相互矛盾的“训练必须使用 OceanSim 多环境”表述。

### G1：解析 13 扇区后端

- [x] 实现解析 1300 射线求交和 13 扇区输出。
- [x] 复用现有 `ForwardSectorObservationBuilder`。
- [x] 单元测试空场景、圆柱、扇区边界、量程裁剪和观测顺序。
- [ ] 完成 OceanSim 配对数据采集和误差报告（2026-10-28 为 PARTIAL：数据已采集，`max <= 0.75 m` 门未通过）。

通过条件：第 5 节等价性门通过。

### G2：张量向量环境

- [x] 实现 `P1TensorVectorEnv`，支持 `N` 个独立 episode。
- [x] 实现批量 reset、50 Hz 子步积分、碰撞、越界、超时、奖励和成功判定。
- [x] 实现自动 reset 及 `terminated/truncated` 分离。
- [x] 用固定动作序列验证 `N=1` 与现有单环境状态、奖励和终止逐项一致。

通过条件：环境语义测试全过，固定动作轨迹无跨环境串扰。

### G3：向量化 DDPG

- [x] 每环境独立 history/GRU hidden。
- [x] 张量化 sequence replay 和 batch PPER。
- [x] 独立 NoisyNet 噪声。
- [x] 记录 `env_steps/s`、`control_steps/s`、`gradient_steps/s`、`update_to_data_ratio`。

通过条件：序列边界、reset bootstrap、checkpoint 重载和确定性评估测试全过。

### G4：吞吐与规模选择

测试 `N = 1, 4, 16, 64, 256, 1024`，每档至少 200 个控制步，重复 3 次。必须包含解析传感器、环境 step 和 DDPG 更新，不能只测空环境。

通过条件：

- `N=64` 端到端吞吐至少为旧单环境基线的 `10x`，并维持每个环境 5 Hz 控制频率。
- 无 OOM、NaN、静默环境降级或 episode 串扰。
- 若 `N=64` 不通过，不进入更大规模；先定位传感器、replay 或更新瓶颈。
- 规模只根据实测选择，不预设 1024 是目标。

### G5：新训练 campaign

- [ ] 使用新 campaign 目录和配置，不覆盖旧单环境结果。
- [ ] 所有配置在新向量环境中重新训练，不直接沿用旧排名作为最终选择。
- [ ] 预算以总环境转移数记录；不能把 `100k` 误解为“每个环境 100k”。
- [ ] 先跑工程 smoke，再跑候选筛选，最后跑单配置长训练。
- [ ] 每次评估加载训练 checkpoint，在 OceanSim P1-A 固定 6 条 episode 上执行。

通过条件：训练来源、评估来源、总转移数、环境数和更新比例全部写入 manifest。

### G6：结论

- [ ] 报告解析与 OceanSim 扇区误差。
- [ ] 报告单环境语义等价和多环境隔离测试。
- [ ] 报告不同环境数的端到端吞吐。
- [ ] 报告新 campaign 的成功率、碰撞/越界/超时、奖励分量和动作饱和。
- [ ] 报告 OceanSim 评估结果，并明确不是 OceanSim 多环境训练。

## 8. 停止条件

出现以下任一情况时停止扩展并报告：

- 配对扇区误差门失败且几何修正后仍无法通过。
- `N=1` 无法与现有单环境保持状态、奖励和终止语义一致。
- 多环境出现 episode/GRU/replay 串扰。
- `N=64` 达不到 10x 端到端加速或 5 Hz 控制频率。
- 训练吞吐上升但更新比例被无意稀释，导致学习信号不可比。
- 需要修改 17 维观测、动作范围、奖励口径或成功定义。
- 需要进入真实 OceanSim 多环境声呐训练或 P2。

## 9. 最终报告格式

1. 状态：COMPLETE / PARTIAL / BLOCKED
2. 修改文件与提交列表
3. 本地测试、ruff、pyright、mypy
4. 解析 13 扇区与 OceanSim 配对误差
5. 单环境语义等价与多环境隔离结果
6. 环境数—吞吐—显存表
7. 新 campaign 配置、总转移数和更新比例
8. 训练结果与 OceanSim 评估结果
9. 未解决风险
10. 下一步建议

报告正文完成后，以下两行必须作为文件最后两行。

## 10. 本轮执行记录（2026-10-28）

### 本地代码与测试

- G1 解析后端和 13 扇区观测已在 `src/auv_rl/perception.py`、`src/auv_rl/observation.py` 实现。
- G2 `P1TensorVectorEnv` 已在 `src/auv_rl/envs/parallel_planar.py` 实现。
- G3 每环境 history/GRU、批量 replay、PPER、独立 NoisyNet 噪声和 `train_parallel_ddpg` 已实现。
- `act_batch` 已改为按每个环境的完整 history 分组构造序列；`N=1` 与旧 `act(evaluate=True)` 连续多步逐项一致。
- replay 序列索引使用 `(env_id, episode_id, episode_step)`，即使 episode_id 冲突也不会跨环境拼接。

本地验证：

```text
../.venv/bin/python -m pytest -q
146 passed

../.venv/bin/python -m ruff check src tests scripts/calibrate_p1_analytic_sensor.py scripts/diagnose_p1_pairing_pose.py
All checks passed!

../.venv/bin/python -m ruff format --check src tests scripts/calibrate_p1_analytic_sensor.py scripts/diagnose_p1_pairing_pose.py
57 files already formatted

../.venv/bin/python -m pyright
0 errors, 0 warnings, 0 informations

../.venv/bin/python -m mypy
Success: no issues found in 55 source files
```

### G1 配对脚本

- 新增 `scripts/calibrate_p1_analytic_sensor.py`。
- 本地 dry-run 生成 100 个 S1 位姿、421 个 S2 位姿，共 6773 个扇区行。
- S2 位姿覆盖起点、东西向接近、南北向绕行、扇区边界和 2–20 m 名义距离。
- dry-run 输出在 `artifacts/m4/p1/parallel/dry_run/`。
- 2026-10-28 已在服务器 OceanSim 环境执行真实配对；最终 artifact 位于服务器
  `artifacts/m4/p1/parallel/analytic_vs_oceansim.csv`、`analytic_equivalence.json` 和
  `pairing_poses.jsonl`，本地对应副本在 `artifacts/m4/p1/parallel/remote_run4_render_sync/`。
- 最终样本为 S1 100 个位姿、S2 421 个位姿，共 6773 个扇区行；每个位姿 13 个扇区，扫描均有效。
- 最终命令记录了 `render_sync=true`、`pre_scan_updates=2`、`pose_updates=1`，代码聚合哈希为
  `e3794f2476dc4403`；`observation_contract` 在 521 个配对样本上验证 shape `(17,)`、dtype
  `float32`、13 扇区顺序、130° 视场和 20 m 归一化一致。

真实配对结果（最终轮）：

| 场景 | 位姿数 | 扇区行 | MAE | P95 | max | 危险漏检 |
|---|---:|---:|---:|---:|---:|---:|
| 总体 | 521 | 6773 | `0.024274 m` | `0.038863 m` | `17.420721 m` | `0` |
| S1 | 100 | 1300 | `0.000000 m` | `0.000000 m` | `0.000000 m` | `0` |
| S2 | 421 | 5473 | `0.030040 m` | `0.046118 m` | `17.420721 m` | `0` |

门禁结论：`MAE` 通过、`P95` 通过、危险漏检通过、S1 全饱和通过、17 维观测契约通过；
`max <= 0.75 m` 失败。G1 仍未完成。

三轮几何/采集修复记录（每轮都保留 artifact）：

| 轮次 | 假设与修改 | MAE | P95 | max | 危险漏检 | artifact |
|---|---|---:|---:|---:|---:|---|
| 0 | 原始结果；`viewTransform` 被错误当作当前位姿 | `1.026835` | `9.807640` | `17.928978` | `335` | `remote_run1/` |
| 1 | 用当前 `State` 重建 world-to-camera 变换 | `0.125526` | `0.045600` | `17.716569` | `0` | `remote_run2/` |
| 2 | 写位姿后增加 `pre_scan_updates=2` | `0.125525` | `0.045600` | `17.716569` | `0` | `remote_run3_pre_scan_updates/` |
| 3 | 用 `rep.orchestrator.step()` 同步 Replicator 注释器 | `0.024274` | `0.038863` | `17.420721` | `0` | `remote_run4_render_sync/` |

剩余 10 个失败行均为“圆柱刚擦过扇区边界”的稀疏点云差异：OceanSim 在该扇区返回
`20 m`，解析后端在边界射线返回实际几何距离；因此危险漏检为 0，但 max 门仍失败。
在冻结的“扇区内取最小距离”语义下，不能通过删除位姿、放宽门禁或改写 OceanSim 字段来通过。

逐行边界分析已写入 `artifacts/m4/p1/parallel/edge_boundary_analysis.json`：

- 10 个失败行来自 9 个位姿，覆盖扇区 `2/3/5/7/9/10`。
- 所有解析命中距离最近扇区边界均不超过 `0.05°`，即 OS 量化为一条 `0.1°` 边界射线。
- 所有失败行的 OceanSim 扇区均为 `20 m` 饱和，且 `dangerous_miss=false`。
- 因此当前阻塞不是坐标、姿态同步或运行版本不一致，而是冻结的 P1-A 点云采样与
  解析连续几何在扇区边界的离散差异；继续改变门禁或点云分辨率会违反冻结契约。

### 服务器状态

2026-10-28 已通过临时凭据连接 `yhx@152.136.183.51:43127`。服务器为 8 张 RTX 4090；
配对采集只使用空闲的 GPU 0，未终止其他用户任务。服务器目录不是 Git 工作树，运行使用：

```bash
cd /data/yhx/underwater-robotics/src/plan_p1/settled_局部规划器_反应式
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=src:/data/yhx/underwater-robotics/src/plan_p1/settled_全局规划器/src \
/data/yhx/underwater-robotics/envs/conda_envs/oceansim/bin/python \
  scripts/calibrate_p1_analytic_sensor.py \
  --render-sync --output artifacts/m4/p1/parallel
```

Isaac Sim 在退出阶段会写出 crash backtrace；配对 CSV/JSON 在退出前已完整写入，结论以 JSON/CSV 为准。

### 当前结论

- G2：本地测试通过。
- G3：本地测试通过；尚未声明端到端吞吐提升，因为 G4 未开始。
- G1：本地代码、单元测试、dry-run 和真实 OceanSim 采集均完成；误差门为 **PARTIAL**，
  `MAE`/`P95`/危险漏检/S1 饱和/17 维契约通过，`max <= 0.75 m` 未通过，不能宣称 G1 完成。

## 报告结尾（固定）

P2 未开始。
多环境训练仅限解析 13 扇区；OceanSim 多环境声呐训练未开始。
