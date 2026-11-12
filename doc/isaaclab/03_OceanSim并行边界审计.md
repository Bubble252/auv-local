# OceanSim 声呐并行边界审计

记录日期：2026-10-28

文档性质：阶段 E 的书面审计，不实现 P1 向量化环境，不修改 P1-A 观测契约。

## 1. 审计范围

本审计只回答：**IsaacLab 的 1024 环境能力，能否以及如何迁移到当前 P1-A OceanSim 感知链。**

证据来自当前代码，而不是 Cartpole 压测结果：

- `src/auv_rl/perception.py`：点云到 `RangeScan` 的适配器和 OceanSim 传感器封装。
- `src/auv_rl/envs/planar_avoidance.py`：单环境 P1 运动学、17 维观测和 reset/step。
- `src/auv_rl/ddpg/agent.py`、`replay.py`、`trainer.py`：单环境 DDPG、序列 replay 和训练循环。
- `configs/rl/p1_planar_oceansim.json`：1300 条射线、1300×200 相机、60 帧预热和场景参数。
- `scripts/train_p1_oceansim.py`：OceanSim 场景、位姿同步、传感器创建和单环境训练入口。

Cartpole 的吞吐只能证明 IsaacLab 的向量化调度和 GPU 仿真能力，不能证明 OceanSim 的渲染、点云回读或 Python 回调可以按同样比例扩展。

## 2. 当前 P1-A 的真实结构

当前 P1-A 是单进程、单场景、单声呐、同步调用：

```text
PlanarAvoidanceEnv.reset/step
    -> OceanSimPointCloudSource.capture(state)
        -> pose_writer(state)
        -> pre_scan_tick() 一次或多次
        -> ImagingSonarSensor.scan()
        -> scan_data["pcl"] + viewTransform
        -> transform_oceansim_points()
        -> PointCloudSonarAdapter.scan_from_points()
        -> 1300 条 0.1° 射线 -> RangeScan
    -> ForwardSectorObservationBuilder
        -> 13 个 10° 扇区 + 目标距离/方位 + u + r
        -> 17 维观测
```

关键事实：

- 声呐几何固定为 `ray_count=1300`、`field_of_view_rad=130°`、`max_range_m=120 m`、`vertical_half_width_m=0.5 m`，见 `src/auv_rl/perception.py:144`。
- 相机分辨率固定为 `1300×200`，见 `configs/rl/p1_planar_oceansim.json:31`。
- 每次 reset 调用 `on_reset` 重建场景，并执行 `warmup_frames=60` 次 tick，见 `scripts/train_p1_oceansim.py:466` 和 `configs/rl/p1_planar_oceansim.json:43`。
- `capture()` 对每个环境单独调用 `sensor.scan()`，默认最多重试 60 次，见 `src/auv_rl/perception.py:577` 和 `configs/rl/p1_planar_oceansim.json:42`。
- 点云从 Warp/OceanSim 转到 NumPy 时使用 `.numpy()` 或 `.to("cpu")`，并转成 `float64`，见 `src/auv_rl/perception.py:53`。
- 训练循环是严格的单环境串行：`env.reset()` 后逐 `env.step()`，没有向量化环境，见 `src/auv_rl/ddpg/trainer.py:373`。

因此，“把 `num_envs` 改成 1024”不会自动把一个 OceanSim 声呐复制成 1024 个独立声呐。

## 3. IsaacLab 能复制什么，不能自动复制什么

IsaacLab 的 `num_envs` 会复制任务中的机器人、刚体和场景实例，并使用批量张量推进物理。对 Cartpole，这足够。

对 P1-A，至少还需要为每个环境复制：

- 独立的声呐相机 prim 或可区分的 render product；
- 独立的相机外参和位姿写入；
- 独立的声呐 annotator / Warp 缓冲；
- 独立的 `scan()` 调用和点云回读；
- 独立的二维运动学状态、碰撞查询和 episode 边界；
- 在 reset 时只重建该环境的障碍物，而不是重建全局 stage。

当前代码的 `OceanSimPointCloudSource` 持有一个 `sensor`、一个 `pose_writer`、一个 `on_reset` 和一个 `_scenario`，见 `src/auv_rl/perception.py:537`。它是单环境对象，不是可按 batch 复用的张量化传感器。

## 4. 瓶颈判断

### 4.1 渲染

单个环境每帧是 `1300×200 = 260,000` 像素。假设每个环境都需要独立渲染：

| 环境数 | 每帧像素 | 5 Hz 时像素/秒 | 20 Hz 时像素/秒 |
| ---: | ---: | ---: | ---: |
| 1 | 0.26 M | 1.3 M | 5.2 M |
| 64 | 16.6 M | 83 M | 332 M |
| 256 | 66.6 M | 333 M | 1.33 G |
| 1024 | 266 M | 1.33 G | 5.33 G |

这些数字只说明量级，不是实测结果。即使 GPU 能渲染，1024 个 render product 的管理、同步和 annotator 开销也不能用单环境数据外推。

### 4.2 射线和扇区聚合

每个环境每帧 1300 条射线，1024 个环境就是 1.33 M 条射线/帧。当前 `PointCloudSonarAdapter.scan_from_points()` 在 NumPy 中为每个环境单独执行：

- 点云坐标变换；
- 方位角计算和 FOV/距离/垂直切片过滤；
- `np.floor` 分箱和 `np.minimum.at` 最近距离聚合；
- 13 扇区聚合。

这部分是 CPU 代码，不是 GPU 批量 kernel。1024 个环境会形成 1024 次 Python 调用和 1024 组 NumPy 临时数组。

### 4.3 GPU 到 CPU 的数据传输

当前适配器把 Warp/OceanSim 点云转成 CPU NumPy，再转 `float64`。这条路径在单环境标定中可接受，但多环境时会成为主要同步点：

- 每个环境每帧都要等待 GPU 标量/点云完成；
- 点云数量和障碍物结构相关，不能用固定小数组估算；
- Python 侧过滤和分箱无法与下一帧渲染流水线重叠；
- 1024 个环境的回读会放大 PCIe、内存分配和 GIL/NumPy 调用开销。

### 4.4 reset 和场景重建

当前每次 reset 都调用 `_build_scene`，并执行 60 次预热 tick。单环境这样做的代价已经体现在 reset 时间中；1024 个环境同时 reset 会产生：

- 全局 stage 变更或大量 prim 变更；
- 统一的 60 帧预热等待；
- 场景构建的串行 Python/USD 调用；
- 首步时间显著高于 steady-state step 时间。

因此，即使 steady-state 的 1024 环境能跑，启动和 reset 也可能成为实验吞吐的瓶颈，必须单独报告。

## 5. 解析 13 扇区训练 + OceanSim 评估的边界

允许的路线是：

```text
训练：解析几何传感器 -> 13 扇区 RangeScan -> 17 维观测 -> 向量化 DDPG
评估：同一策略 -> OceanSim P1-A 点云 -> 13 扇区 RangeScan -> 同一 17 维观测
```

保持观测语义的条件：

- 使用相同的 130° 前视场、120 m 原始量程、20 m 局部有效视野、13 个 10° 扇区和 `[0.8, 0, 0]` 艏部原点；
- 使用相同的垂直切片 `±0.5 m` 和最近回波规则；
- 使用相同的目标距离/方位定义、归一化和 17 维顺序；
- 训练日志必须写 `observation_source="analytic_13sector"`，不能写成 `oceansim_simplified_pcl`；
- 评估日志必须写回 `observation_source="oceansim_simplified_pcl"`；
- 解析传感器的成功率、收敛曲线和 OceanSim 评估结果分表报告，不能混成同一张“OceanSim 训练结果”。

这不是“用解析传感器冒充 OceanSim 实测”，而是把训练吞吐和感知真实性拆成两个可审计的实验轨道。

## 6. DDPG 在 1024 环境下需要的改造

当前 DDPG 是单环境实现，直接扩到 1024 会先遇到 replay 和 episode 语义问题，而不是网络显存问题。

### 6.1 Replay buffer

当前 `PrioritizedSequenceReplayBuffer`：

- 用 `_Transition` Python 对象和 `list` 保存转移，见 `src/auv_rl/ddpg/replay.py:85`；
- 用 `dict[(episode_id, episode_step)]` 查找序列，见 `src/auv_rl/ddpg/replay.py:140`；
- 采样时逐行重建序列，见 `src/auv_rl/ddpg/replay.py:259`；
- 默认 `buffer_size=100000`，见 `configs/rl/p1_planar_oceansim.json:53`。

1024 个环境 × 200 步就是 204,800 个转移，已经超过默认容量；Python 对象开销还会放大内存。需要改成预分配张量 ring buffer，并用 `(env_id, episode_id, episode_step)` 做向量化边界索引。

### 6.2 GRU 和序列

当前 `agent.act()` 维护一个全局 `_history` 列表，见 `src/auv_rl/ddpg/agent.py:271`。多环境必须改成每个环境一份历史，或把 batch 维度显式传入 actor。否则不同环境的序列会串在一起。

训练侧的 GRU 需要：

- 每个采样序列只来自一个环境；
- `history_length=10` 的 padding mask 正确；
- episode reset 时清空该环境的 hidden state；
- 评估时使用确定性 hidden state，不采样噪声。

### 6.3 PPER 和 NoisyNet

- PPER 的 priority tree 需要支持批量 `add` 和批量 `update_priorities`，不能逐条 Python 循环。
- beta 应按梯度步数更新，而不是环境步数；1024 环境的 rollout 会产生不同数量的梯度步。
- NoisyNet 的噪声应按环境独立采样，或明确记录共享噪声的设计；否则 1024 个环境会共享同一探索扰动。

### 6.4 Episode 边界和 bootstrap

向量化环境会在同一个 `step()` 中自动 reset 部分环境。必须区分：

- `terminated`：真实终止，bootstrap 为 0；
- `truncated`：时间上限，按配置决定是否 bootstrap；
- 自动 reset 后的新观测不能写回上一条转移的 `next_observation`；
- replay 序列不能跨环境，也不能跨 episode；
- 每个环境需要独立的 `episode_id`，不能只用一个全局计数器。

## 7. 结论

1. **IsaacLab 的 1024 环境能力可以用于 P1 的向量化训练基础设施，但不能直接等同于 P1-A OceanSim 声呐的 1024 环境能力。**
2. 当前 P1-A 的渲染、点云回读、NumPy 分箱、场景 reset 和单环境 Python 回调都不支持 batch 化；直接开 1024 个 OceanSim 声呐，最可能的失败点是显存、render product 管理、GPU→CPU 回读或 reset，而不是 DDPG 网络。
3. 推荐路线是“解析 13 扇区并行训练 + OceanSim P1-A 评估”，但必须严格保持 17 维观测、13 扇区几何和来源标记。
4. 在投入完整 P1 向量化开发前，先做一个 16/64 环境的 OceanSim 声呐微基准，测每环境每帧的渲染、scan、回读和分箱耗时；如果 64 环境已经不能维持目标决策频率，就不应直接承诺 1024 环境 OceanSim。
5. Cartpole 的吞吐数字只能用于判断 IsaacLab 调度和 GPU 基础能力，不能写成 OceanSim 声呐吞吐。

## 8. 下一步建议

- 先完成阶段 A–D 的单环境、64、256、1024 Cartpole 工程门。
- 另开一个小实验：16/64 环境 OceanSim 声呐复制，测每环境每帧成本和显存；本任务不做完整实现。
- 如果 64 环境 OceanSim 已明显低于 5 Hz，优先实现解析 13 扇区向量化训练，OceanSim 只做评估。
- 在任何向量化 P1 代码合入前，先补 replay buffer、episode 边界和 GRU hidden state 的单元测试。

正式多环境训练未开始。
P1/P2 实验口径未修改。
