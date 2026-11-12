# Reading note：residual policy、AUV path following 与 terminal goal-reaching

**日期**：2026-11-12<br>
**对应 blocker**：`research/blockers/2026-11-12-goal159c-h3-residual-no-closed-loop-gain.md`

## 1. 检索

使用 `paper-search` 查询并核对了以下工作：

- Tom Silver et al., *Residual Policy Learning*, arXiv:1812.06298 (2018)。
- Simen Theie Havenstrøm et al., *Deep Reinforcement Learning Controller for 3D
  Path-following and Collision Avoidance by Autonomous Underwater Vehicles*,
  arXiv:2006.09792 (2020)。
- Gautham Vasan et al., *Revisiting Sparse Rewards for Goal-Reaching Reinforcement
  Learning*, arXiv:2407.00324 (2024)。
- Enrico Anderlini et al., *Docking Control of an Autonomous Underwater Vehicle
  Using Reinforcement Learning*, Applied Sciences 9(17):3456 (2019),
  DOI `10.3390/app9173456`。
- 相关工作还包括 *Residual Policy Learning for Powertrain Control*
  (arXiv:2212.07611) 与 *Three-Dimensional Path-Following Control of an AUV Based
  on Deep Reinforcement Learning* (DOI `10.2478/pomr-2022-0042`)。

## 2. 要点

### Residual Policy Learning

Silver 等人的核心设定是：已有“不错但不完美”的控制器时，学习一个 residual 去改进
它，而不是从零学习。论文强调 residual 适合控制器可微性差、模型不匹配、传感器噪声
和稀疏奖励的长时程任务。对本项目的含义是：H3 已经是强基线，residual 应该只修正
控制器剩余的误差，而不是重新产生一个大幅度的 actuator 指令。

### AUV path following

Havenstrøm 等人的工作说明 DRL 可以在 AUV 3D path following + obstacle avoidance
上工作，但它主要验证的是从零/端到端控制能力，不等于“在强控制器上叠加 residual
一定有增益”。本项目 Goal157/158 已经显示：强 H3 基线下，旧 K2 上的神经增益会消失，
必须做同控制器配对。

### Goal-reaching / terminal capture

Vasan 等人的稀疏奖励研究表明，goal-reaching 任务中 minimum-time/terminal 形式
可以优于 dense reward，且 early goal-hit rate 是学习成功的重要指标。对本项目的含义
是：如果失败集中在 goal plane/terminal capture，应该优先审计 termination、entry
geometry 和 terminal credit，而不是继续增加 dense residual 幅度或放宽 goal radius。

### AUV docking

Anderlini 等人的 docking 工作说明终端对接/捕获需要专门的终端控制与奖励设计；这
与本项目 current10/15 的 `goal_plane_miss`、final distance 集中在 `1.35–1.82 m`
一致。它不是直接支持“换 SAC 算法”，而是支持先做 terminal capture/动作参数化。

## 3. 对本项目的决策含义

1. residual 的合理角色是“在 H3 之后的受约束修正”，不是大幅 actuator 叠加；
2. 当前 action saturation `~0.56`、通道方向跨 seed 变号，说明参数化/信用比算法更
   可疑；
3. 失败是 plane miss，且放宽 success 半径不产生稳定收益，说明 goal radius 不是
   主要修复手段；
4. 在 residual 方向未稳定前，不应启动 REDQ/DroQ 或 5M SAC 对照；先做 command-space
   residual / terminal constrained residual 的预注册 pilot。
