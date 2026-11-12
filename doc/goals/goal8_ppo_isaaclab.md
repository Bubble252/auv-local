# 任务：PPO + IsaacLab 原生多环境实验（P1 PPO 路线）

记录日期：2026-10-29（2026-10-30 更新状态快照）

状态：**G1–G6 PASS；G7 解析最终评估 PASS；G7 OceanSim 50Hz 稳定性 FAIL（仅保留评估用途）；最终总状态 FAIL；G8 阻塞**

最后更新：2026-10-30（lr=1e-4 v3 的 seeds 0/1/2 已完成，G5 以 best checkpoint 通过；
C1/C2/C3 三条课程臂均已完成训练和独立复评，G6 审计全部 PASS；C3 best checkpoint 的
解析最终评估全部 PASS；OceanSim 同配置重复三次延迟门仅 `1/3` 通过；G7.3 又完成关闭
抗锯齿、CPU 绑定、IsaacLab 性能渲染子集和 16 Kit 线程的单变量迭代，CPU 绑定达到 `2/3`
但最差 p95 仍为 `20.75ms`，因此已生成并保留总状态 `FAIL` 的
`final_evaluation.json` 和 `final_report.md`）

编号说明：本文件的 G0–G8 是 PPO 路线自己的阶段门；这里的 G1 指隔离安装 PPO 后端，
不是 `doc/01_技术栈.md` 6.2 节感知配对实验里的 G1 边界门。

本地工作目录：`/home/bubble/水下机器人/settled_局部规划器_反应式`

服务器工作目录：`/data/yhx/underwater-robotics/src/plan_p1/settled_局部规划器_反应式`
（所有 IsaacLab/PPO 训练和评估只在服务器执行，本地不启动 GPU 训练）

关系：本文件是 goal6/goal7 之后的新训练路线。goal6 的 120m/50Hz/BC 口径和 goal7 的
DDPG 稳定性结论继续保留，不覆盖、不删除旧结果。本文件只新增“PPO + IsaacLab 原生多环境”
实验链，不改写已完成实验的结论。

当前结论快照：

| 项目 | 状态 | 说明 |
|---|---|---|
| 解析 PPO 主训练与课程 | **PASS** | G1–G6 完成；最终模型预注册为 C3 综合臂 best |
| 解析最终评估 | **PASS** | C3 best 在 S1/S2、C1/C2/C3 固定评估集全部通过 |
| OceanSim 延迟门 | **FAIL** | `v7/v8/v9` 同配置重复三次仅 `v7` 完整通过；`v8` median/p95 失败，`v9` p95 失败 |
| G7.3 延迟稳定化迭代 | **FAIL** | CPU 绑定后 `pinned_v11/v12/v13` 为 `2/3`；最差 p95 `20.75ms`，仍越冻结门 |
| OceanSim 正式 6 条主评估 | **不作为正式成功结果** | `v7` 的 6 条评估虽成功，但未通过三次稳定性门；保留为诊断证据 |
| G8 声呐并行微基准 | **BLOCKED** | OceanSim 仅能作为评估边界，不扩大渲染并行 |

“单次延迟 PASS”不能写成“50Hz 已稳定”。正式判定要求相同配置重复 3 次均通过；
并且每次都有完整的 `evaluation_summary.json`、逐 episode CSV 和轨迹证据。

## 1. 目标与边界

### 1.1 本轮目标

建立一套独立、可重复、参数可用 JSON 迭代的 PPO 训练链：

```text
IsaacLab DirectRLEnv
  + 4096 级 GPU 批量环境
  + 解析 13 扇区观测
  + 二维水平运动学
  + PPO（rsl_rl）
  → 最终 checkpoint
  → 解析评估 + OceanSim P1-A 单环境评估
```

首先要回答三个问题：

1. PPO 能否在 S1/S2 上稳定收敛，并达到与 E5 可比较的成功率；
2. 扩展到单圆柱随机、左右阻挡、S 形和偏移门后，能否显著改善分布外泛化；
3. 多环境训练吞吐和 episode 边界是否可靠，是否有 NaN、状态串扰或自动 reset 错误。

### 1.2 明确不做

- 不在第一版直接开 1024 个 OceanSim 声呐环境训练。
- 不把 OceanSim 渲染结果当成 PPO 训练吞吐的前提。
- 不修改 `goal6`、`goal7` 的配置、门禁或历史结果。
- 不进入 P2、3D ALOS、PVS 或全局规划器后续阶段。
- 不默认使用 BC 初始化。BC 只作为可选 warm-start 消融，不作为主链路。
- 不为了“跑起来”放宽 17 维观测、动作范围、成功判定、120m 有效范围或 50Hz 决策。

### 1.3 两条训练/评估轨道

| 轨道 | 训练感知 | 动力学 | 用途 |
|---|---|---|---|
| 解析训练 | 批量解析 13 扇区，120m | 二维水平运动学 | PPO 主训练 |
| 真实评估 | OceanSim P1-A 简化点云，120m | 同一 17 维观测适配器 | 最终 checkpoint 评估 |

训练日志必须写 `observation_source="analytic_13sector"`；OceanSim 日志必须写
`observation_source="oceansim_simplified_pcl"`。两类结果分表，不能混写。

现有审计已确认：IsaacLab 的 4096 环境调度可用，Cartpole 压测约
`51.2 万 env steps/s`，见 `artifacts/infra/isaaclab/scaling.csv`；但这不等于 4096 个
OceanSim 声呐可并行。P1 自己的 `DirectRLEnv` 首轮实测在 4096 环境达到约
`136.7 万 env steps/s`、`334 control steps/s`，见
`artifacts/ppo_isaaclab/scaling_probe.json`。该数字仍只覆盖解析几何，不代表 OceanSim
点云、渲染或真实声呐可并行。OceanSim 的并行边界以
`doc/isaaclab/03_OceanSim并行边界审计.md` 为准。

Cartpole 数字只证明 IsaacLab 的调度和 GPU 环境扩展能力，不是 P1 解析传感器吞吐。P1 必须
单独测 `control_steps/s`、`env_steps/s`、sensor 耗时占比和 p50/p95 step 延迟；如果 1300
条射线 × 4096 环境的解析求交成为瓶颈，先优化向量化实现，不能把 Cartpole 数字当作 PPO 吞吐。

“4096 环境”在本任务中首先指**单个训练进程、单张 GPU 上的 `num_envs=4096`**。8 张
RTX 4090 不会因为设置一个 `num_envs` 就自动组成一个 32768 环境的分布式 PPO。第一轮只
在单卡跑通 4096；多卡实验必须启动多个独立进程，每卡一个 seed 或一个配置，分别记录
`gpu_id`、`process_id` 和 `config_sha256`。除非后续明确需要，不引入跨进程 all-reduce
或修改 rsl_rl 的分布式逻辑。

## 2. 硬约束：隔离环境与源码零侵入

### 2.1 允许修改的位置

只允许修改当前项目仓库：

```text
src/auv_isaaclab/                    # 新增的外部 IsaacLab 任务包
configs/ppo/isaaclab/                # 新增 JSON 配置
scripts/                             # 新增 PPO 训练、评估、校验、压测入口
tests/                               # 新增批量环境与配置测试
artifacts/ppo_isaaclab/              # 小体积证据、汇总、CSV、JSON
doc/goals/goal8_ppo_isaaclab.md      # 本文件
```

### 2.2 禁止直接修改的位置

```text
/data/yhx/underwater-robotics/simulator/IsaacLab/source/**
/data/yhx/underwater-robotics/envs/venvs/isaaclab/lib/python3.11/site-packages/**
```

包括 IsaacLab、`isaaclab_tasks`、`isaaclab_rl` 和 `rsl_rl` 的源码。优先顺序必须是：

1. 使用 IsaacLab/rsl_rl 的公开 API；
2. 在当前项目的 `src/auv_isaaclab/` 中写子类、wrapper 或 adapter；
3. 通过环境配置、task registration 或 runner 注入实现；
4. 只有在第 1–3 项都无法完成、且已证明是上游 bug 时，才允许打最小补丁。

任何例外必须同时满足：

- 保存为 `patches/isaaclab/<name>.patch` 或 `patches/rsl_rl/<name>.patch`，不直接手改安装目录；
- 在 `doc/isaaclab/` 记录原因、上游版本、复现步骤和回滚方法；
- 补丁前后各跑一次 `N=1` 对齐测试和 IsaacLab smoke；
- 在提交信息和 manifest 中标注 `upstream_patch=true`。

### 2.3 OceanSim/Isaac Sim 环境隔离

- 基础 conda 环境：`/data/yhx/underwater-robotics/envs/conda_envs/oceansim`
- IsaacLab 独立 venv：`/data/yhx/underwater-robotics/envs/venvs/isaaclab`
- IsaacLab 源码：`/data/yhx/underwater-robotics/simulator/IsaacLab`

所有新包只安装到独立 venv，禁止在 `oceansim` conda 环境执行 `pip install`、`pip uninstall`
或升级 PyTorch/Isaac Sim。安装前后必须记录并比较 `oceansim` 的 `pip freeze` SHA-256；
不一致时停止进入训练阶段。

安装 PPO 后端前记录以下指纹：

```bash
/data/yhx/underwater-robotics/envs/conda_envs/oceansim/bin/python -m pip freeze | sha256sum
/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python -m pip freeze | sha256sum
```

### 2.4 当前版本基线

| 组件 | 当前值 |
|---|---|
| Isaac Sim | 5.0.0.0 |
| IsaacLab 源码 | v2.2.1 |
| IsaacLab venv Python | 3.11.13 |
| PyTorch | 2.7.0+cu128 |
| NumPy | 1.26.0 |
| Gymnasium | 1.2.0 |
| PPO 后端 | `rsl-rl-lib==2.3.3`（已安装到 IsaacLab 独立 venv） |

IsaacLab v2.2.1 的 `isaaclab_rl[rsl-rl]` 固定 `rsl-rl-lib==2.3.3`。如果安装尝试升级
Isaac Sim、PyTorch、NumPy 或 Pillow 并造成冲突，立即停止，不要用 `--force-reinstall`
掩盖问题。

G1 实测的隔离安装结果见 `artifacts/infra/isaaclab/ppo_install_manifest.json`。本次安装只向
IsaacLab venv 新增 `rsl-rl-lib==2.3.3`、`GitPython==3.2.1`、`gitdb==4.0.12`、
`smmap==5.0.3`；Isaac Sim 5.0.0.0、PyTorch 2.7.0+cu128、torchvision 0.22.0+cu128、
NumPy 1.26.0、Pillow 11.2.1 均未变化。`oceansim` conda 环境的 freeze 指纹前后一致。

### 2.5 依赖下载与镜像策略

- 新依赖只允许安装到 `/data/yhx/underwater-robotics/envs/venvs/isaaclab`，禁止在
  `oceansim` conda 环境执行 `pip install`、`pip uninstall` 或 `pip install -U`。
- 优先使用官方源；下载失败时，只对当前命令临时设置代理
  `https_proxy=http://127.0.0.1:7897`、`http_proxy=http://127.0.0.1:7897`，或改用可信的
  国内镜像。不要把代理、镜像或密码写入仓库、全局 `pip.conf`、`.bashrc` 或日志。
- 安装前记录 `pip freeze` 指纹，安装后重新记录并比较；出现任何已有包版本变化就停止。
- 新增依赖必须说明用途，并在 `ppo_install_manifest.json` 记录包名、版本、安装命令和下载源。
- 第一版不安装第二套 PPO 框架（例如 `skrl`），也不安装 `stable-baselines3`；避免同一实验
  出现两套 runner 和两套日志口径。
- 不升级 Isaac Sim、PyTorch、NumPy、Pillow 或 IsaacLab；不使用 `--force-reinstall`、
  `--upgrade`。安装脚本异常时先保留日志，再决定是否回滚，不靠反复重装碰运气。

## 3. 总体架构

```text
configs/ppo/isaaclab/*.json
        │
        ├─ 配置校验 / schema_version / config_hash
        │
        ├─ P1AvoidanceEnvCfg.from_json()
        │       └─ IsaacLab DirectRLEnv（批量张量）
        │
        └─ build_rsl_rl_runner_cfg()
                └─ rsl_rl OnPolicyRunner（不改源码）
                        │
                        ├─ TensorBoard / CSV 日志
                        ├─ checkpoint_latest / checkpoint_best
                        └─ deterministic evaluation

解析训练 ──> 解析评估（固定 S1/S2 + 分布外场景池）
       └──> OceanSim P1-A 单环境评估（最终 checkpoint）
```

### 3.1 建议目录

```text
src/auv_isaaclab/
  __init__.py
  config.py                # 严格 JSON 解析、extends、哈希、派生维度
  tasks/
    __init__.py
    p1_avoidance/
      __init__.py
      env_cfg.py             # JSON → IsaacLab configclass
      batch.py               # 二维状态、碰撞、解析声呐、奖励、history、reset
      env.py                 # DirectRLEnv 封装和动作缩放
      geometry.py            # G6 课程扩展时再拆出
      sensor.py              # G6 课程扩展时再拆出
      scenarios.py           # G6 课程扩展时再拆出
      rewards.py             # G6 奖励消融时再拆出
      metrics.py             # G5 评估和 episode 统计时补齐
      agents/
        rsl_rl_ppo_cfg.py    # JSON → PPO runner config

configs/ppo/isaaclab/
  schema/
    ppo_experiment.schema.json
  p1_ppo_base.json           # 冻结契约和默认环境参数
  p1_ppo_smoke.json          # 小规模连通性实验
  p1_ppo_main.json           # 主训练实验
  p1_ppo_time_bonus.json     # 可选的时间奖励消融
  eval_analytic.json         # 解析评估场景和 seeds
  eval_oceansim.json         # OceanSim P1-A 评估参数

scripts/
  validate_ppo_config.py
  train_p1_isaaclab_ppo.py
  probe_p1_isaaclab_scaling.py
  summarize_p1_isaaclab_scaling.py
  eval_p1_isaaclab_ppo.py
  # eval 入口在 G5 前补齐
```

## 4. JSON 参数体系

### 4.1 原则

- JSON 是实验参数的唯一事实来源；Python 代码只读 JSON，不把可调参数散落在源码里。
- 所有 JSON 必须有 `schema_version`，例如 `auv_ppo_isaaclab.v1`。
- 训练输出目录必须保存一份展开后的完整配置和 `config_sha256`。
- 命令行只允许覆盖运行时字段：`--num-envs`、`--seed`、`--device`、`--output`、
  `--max-iterations`、`--resume`；覆盖值也要写入 manifest。
- 配置解析失败必须直接退出，不能静默使用默认值。
- 契约字段（观测维度、扇区数、动作范围、dt、episode 上限）允许修改 JSON 值，但不允许
  在没有记录的情况下改变语义；校验器发现不兼容时直接报错。
- 每次参数迭代新建带版本号的 JSON（例如 `p1_ppo_main_v2.json`），用 `extends` 继承
  base，不直接覆盖上一次实验的配置；输出目录同时保存展开后的配置和哈希，保证结果可追溯。

### 4.2 建议的 JSON 结构

```json
{
  "schema_version": "auv_ppo_isaaclab.v1",
  "experiment": {
    "name": "p1_ppo_main_v1",
    "seed": 0,
    "output": "outputs/ppo_isaaclab/p1_ppo_main_v1"
  },
  "env": {
    "num_envs": 4096,
    "dt_s": 0.02,
    "integration_step_s": 0.02,
    "max_episode_steps": 3000,
    "sensor_max_range_m": 120.0,
    "local_effective_range_m": 120.0,
    "goal_radius_m": 1.0,
    "action_max_accel_mps2": 0.5,
    "action_max_yaw_rate_rps": 0.2617993877991494
  },
  "observation": {
    "name": "forward13_target_u_r",
    "version": "2.0-120m",
    "history_length": 16,
    "sector_count": 13,
    "field_of_view_rad": 2.2689280275926285,
    "ray_count": 1300,
    "min_range_m": 0.2,
    "origin_body_m": [0.8, 0.0, 0.0],
    "vertical_half_width_m": 0.5,
    "sector_width_deg": 10.0,
    "normalize": true,
    "normalization_max_range_m": 120.0
  },
  "reward": {
    "version": "paper",
    "distance_penalty_factor": -1.0,
    "action_penalty_factor": -0.05,
    "near_obstacle_penalty": -2.0,
    "far_obstacle_penalty": -1.0,
    "collision_penalty": -1000.0,
    "timeout_penalty": -500.0,
    "goal_bonus": 700.0,
    "reward_obstacle_range_m": 20.0,
    "obstacle_influence_fraction": 0.7,
    "reward_step_scale": 0.1,
    "ppo_reward_scale": 0.01
  },
  "curriculum": {
    "stage": "s1_s2",
    "scenario_mix": {
      "s1": 0.25,
      "s2": 0.75
    },
    "start_jitter_xy_m": 0.0,
    "start_jitter_yaw_rad": 0.0
  },
  "ppo": {
    "num_steps_per_env": 32,
    "max_iterations": 5000,
    "save_interval": 100,
    "learning_rate": 0.0003,
    "gamma": 0.99798,
    "lam": 0.95,
    "clip_param": 0.2,
    "entropy_coef": 0.005,
    "value_loss_coef": 1.0,
    "num_learning_epochs": 5,
    "num_mini_batches": 8,
    "max_grad_norm": 1.0,
    "actor_hidden_dims": [256, 256],
    "critic_hidden_dims": [256, 256]
  },
  "evaluation": {
    "eval_every_iterations": 50,
    "seeds": [0, 1, 2],
    "scenario_keys": ["s1", "s2"]
  }
}
```

数值只是首轮候选，不是冻结结论。实际训练前先跑 smoke，并记录每个字段的选择依据。

奖励首轮实测后做了两处修正，均来自可复现的回报对比：

- `collision_penalty` 从 `-80` 改为 `-1000`，并新增 `timeout_penalty=-500`。S2 直行碰撞
  回报约 `-1133`，硬转出界约 `-1057`；若只把碰撞改到 `-700`，策略会改为拖到 3000 步
  超时（约 `-800`）。当前排序为 `goal +650 > timeout ~-800 > OOB ~-1057 > collision -1133`，
  提前终止不再比继续尝试更划算。
- 新增 `action_penalty_factor=-0.05`，对归一化动作做 L2 正则。P2 best checkpoint 曾全程把
  `u_dot` 压在下限 `-1`（速度已到 1.0m/s，动作实际无效），导致全组件饱和率 `0.518`；
  正则后同一 checkpoint 的 yaw 饱和率降到 `0.055`，S1/S2 仍保持 `3/3` 成功。

`distance_penalty_factor` 之前未进入批量核心公式，本轮已修正为实际乘子；默认 `-1.0`
不改变既有数值。

### 4.3 建议拆分的配置文件

| 文件 | 内容 | 是否需要频繁改 |
|---|---|---|
| `p1_ppo_base.json` | 17 维观测、动作范围、120m、50Hz、奖励基础值 | 少 |
| `p1_ppo_smoke.json` | 小环境数、短 rollout、快速连通性 | 中 |
| `p1_ppo_smoke_mixed.json` | 64 环境、S1+S2 混合 smoke | 中 |
| `p1_ppo_main.json` | 主 PPO 参数和训练预算 | 中 |
| `p1_ppo_c1.json` | C1 随机单圆柱/center block | 课程扩展时 |
| `p1_ppo_c2.json` | C2 blocking left/right | 课程扩展时 |
| `p1_ppo_c3.json` | C3 slalom/S3/混合场景 | 课程扩展时 |
| `p1_ppo_time_bonus.json` | 成功时间奖励消融 | 实验时 |
| `p1_ppo_bc_warmstart.json` | BC warm-start 消融，不属于第一版主链 | 实验时 |
| `eval_analytic.json` | 固定评估场景、seeds、成功门 | 少 |
| `eval_oceansim.json` | OceanSim 路径、延迟门、评估 seeds | 少 |

禁止把训练场景列表、奖励系数、网络大小和 seeds 写死在 `train_p1_isaaclab_ppo.py` 中。

OceanSim 的 PPO 评估参数另放在 `configs/rl/`，与 IsaacLab 训练配置分开：

| 文件 | 用途 |
|---|---|
| `p1_ppo_oceansim_120m50hz.json` | 主评估：120m、50Hz、16 帧、CPU readback |
| `p1_ppo_oceansim_120m50hz_preferred_readback_v2.json` | 单变量比较 preferred readback |
| `p1_ppo_oceansim_120m50hz_no_copy_v3.json` | 单变量比较禁用 array copy |
| `p1_ppo_oceansim_120m50hz_aa_off_v4.json` | 单变量比较 `anti_aliasing=0` |
| `p1_ppo_oceansim_120m50hz_aa_off_perf_v5.json` | AA 关闭 + IsaacLab 官方性能渲染子集 |
| `p1_ppo_oceansim_120m50hz_16threads_v6.json` | 单变量比较 `limit_cpu_threads=16` |

这组文件只记录评估运行时参数；修改时必须新建版本号并在新输出目录运行，不覆盖旧的失败证据。
`anti_aliasing` 只允许 `[0, 4]` 的整数；`render_profile` 只接受 `default` 或
`isaaclab_performance_v1`；`limit_cpu_threads` 只接受 `[1, 128]` 的整数，非法值由
`_simulation_app_config()` fail-fast。

## 5. 环境设计

### 5.1 DirectRLEnv

第一版使用 `DirectRLEnv`，不使用 ManagerBasedRLEnv。原因是 P1 的权威状态是自定义二维
运动学，障碍和解析声呐都是几何计算，不需要逐个环境构建复杂 USD 资产管理器。

需要实现并测试：

- `_setup_scene()`：建立批量状态、goal、obstacle 数组和 history buffer；
- `_pre_physics_step()` / `_apply_action()`：把归一化动作映射到物理动作；
- `_get_observations()`：返回 `[N, 17]`，history 单独维护；
- `_get_rewards()`：返回原始奖励和缩放后的 PPO 奖励；
- `_get_dones()`：分别返回 `terminated` 和 `truncated`；
- `_reset_idx()`：只重置指定环境，不重建整个 stage；
- `_episode_metrics()`：记录成功、碰撞、越界、超时、clearance、time-to-goal。

### 5.2 时序与积分步长

三个时间量必须分开记录，不能混用：

| 字段 | 含义 | 本轮值 |
|---|---|---|
| `dt_s` | 策略控制周期；每调用一次 `env.step()` 推进的物理时间 | `0.02 s`（50Hz） |
| `integration_step_s` | 批量运动学内部单次积分的最长时间步 | `0.02 s` |
| `max_episode_steps` | 每个 episode 最多包含多少个策略控制步 | `3000` |

因此本轮 `dt_s / integration_step_s = 1`，每个控制步内部只积分一次；3000 个控制步对应
`60 s` 物理时间。`integration_step_s` 不是 PPO 的 rollout 长度，也不是声呐刷新率。若以后
降低 `dt_s` 或提高控制频率，必须保证 `dt_s` 是 `integration_step_s` 的整数倍，并按
`n_substeps = round(dt_s / integration_step_s)` 执行内部子步。

积分语义必须与当前 `PlanarKinematics.step()` 一致：

```text
u_next = clip(u + u_dot * h, 1.0, 2.5)          # m/s
yaw_next = wrap(yaw + r * h)                     # rad
x_next = x + (u * cos(yaw) + current_x) * h
y_next = y + (u * sin(yaw) + current_y) * h
```

其中 `h = integration_step_s`，`u_dot ∈ [-0.5, 0.5] m/s²`，`r ∈ [-15°, 15°]/s`。如果后续
加入子步，碰撞、越界和最小 clearance 必须在每个子步后检查，不能只在控制步末尾检查。

### 5.3 动作接口

为不改 rsl_rl 源码，IsaacLab 环境的 `action_space` 使用归一化动作：

```text
policy_action ∈ [-1, 1]^2
u_dot = policy_action[0] * 0.5 m/s²
r     = policy_action[1] * 0.2617993877991494 rad/s
```

runner 使用 `clip_actions=true`。物理动作仍保持原范围；
环境内部记录 clip、饱和率和实际物理动作。不能通过修改 PPO 网络源码实现 tanh 限制。

### 5.4 状态、碰撞和 reset

- 批量状态至少包含位置 `[N,2]`、yaw `[N]`、surge `[N]`、previous action `[N,2]`；
- 障碍使用定长 padding 数组，至少支持 4 个 box 或 8 个 cylinder；
- 碰撞和越界必须与当前 `PlanarAvoidanceEnv` 的船体几何和 clearance 定义一致；
- 每个环境有独立 `env_id`、`episode_id`、history 和 episode 统计；
- `terminated` 与 `truncated` 分开；时间上限按 PPO 的 time-limit bootstrap 处理；
- 自动 reset 后的新观测不能写回上一条 transition 的 `next_observation`；
- 任何一个环境 reset 不能清空其他环境的 history。

### 5.5 解析 13 扇区

保持当前几何和归一化：

- 1300 条射线、130° 前视场、120m 最大量程；
- 13 个 10° 扇区；
- 原点 `[0.8, 0, 0]m`；
- 垂直切片 `±0.5m`；
- 扇区内最近回波；
- 归一化到 `[0,1]`。

实现上使用预计算射线方向和批量解析几何求交。必须提供 `N=1` 与现有
`ForwardSectorObservationBuilder` 的逐项对齐测试；不能只比较“成功率”判断等价。

批量实现必须在 torch 上向量化，禁止在 Python 里逐环境或逐射线循环。1300 条射线 × 4096
环境意味着每个控制步约 530 万条射线；50Hz 下约 2.66 亿条射线/秒，必须把传感器耗时单独
计时。先保证 `N=1` 对齐，再测 64/256/1024/4096 的吞吐和显存；如果传感器占比过高，优先
减少无效射线、按场景裁剪或缓存静态几何，不能删减 13 扇区的观测契约。

### 5.6 观测历史

第一版使用 16 帧堆叠：

```text
raw observation: [N, 17]
stacked observation: [N, 16, 17]
policy input: [N, 272]
```

17 维的构成为：13 个扇区最近距离 + 目标距离 + 目标方位角 + 前向速度 + 偏航角速度。第一版
actor 和 critic 都使用同一份 272 维输入，不启用 privileged critic 或非对称观测；如果后续
要加 critic 特权信息，作为单独消融，不改主链路。

MLP PPO 比 recurrent PPO 更容易稳定和排查。后续若要使用 GRU/LSTM，作为单独实验，不在
第一版同时改动。

### 5.7 奖励

第一版保持当前四项奖励和 50Hz 时间归一化，使用原始奖励计算审计指标，同时给 PPO 一个
缩放后的奖励：

```text
distance = -1.0 * tanh(d_t / d_0) * 0.1
obstacle = -0.2 (front obstacle < 14m) / -0.1 (14–20m) / 0
action   = -0.05 * ||policy_action||_2^2
goal     = +700
collision = -1000
timeout = -500
total_reward = distance + obstacle + action + terminal_reward
ppo_reward = 0.01 * total_reward
```

连续项 `distance`、`obstacle` 乘以 `dt_s / 0.2 = 0.1`；动作正则不乘时间尺度，因为它是每个
控制步的即时代价。终端项 `goal`、`collision`、`timeout` 只在终止时给一次，**不乘 0.1**。
越界与碰撞共用碰撞终端惩罚；当前排序约为 `goal +650 > timeout ~-800 > OOB ~-1057 >
collision ~-1133`，因此策略不会为了提前结束而主动出界或碰撞。`reward_obstacle_range_m`
必须固定为 `20.0`，与 120m 观测范围解耦：近障碍阈值是 `0.7 × 20 = 14m`，远障碍阈值是
`20m`。不能因为声呐量程改成 120m 就把奖励影响范围改成 84m/120m，否则同一段轨迹的障碍
惩罚会被放大，且与 goal6 的口径不可比。

`ppo_reward_scale=0.01` 是首轮值，必须同时记录原始 reward 和缩放 reward。奖励审计仍要满足：

- 成功回报为正；
- 超时回报为负；
- 碰撞回报差于超时；
- 成功与超时至少有可测差距。

时间奖励不混入第一版；如果主实验完成后仍然明显“只求成功、不追求快速”，再单独增加
`p1_ppo_time_bonus.json` 做消融，一次只改一个变量。

## 6. PPO 与 rsl_rl 配置

### 6.1 后端选择

第一版使用 `rsl_rl`，通过 `RslRlVecEnvWrapper` 和 `OnPolicyRunner` 接入。理由：

- 与 IsaacLab 的 DirectRLEnv 和 vector env 集成成熟；
- PPO 在 GPU 上按 rollout batch 更新，适合数千环境；
- 不需要引入 DDPG replay buffer、NoisyNet、PPER 或 BC 辅助损失。

`skrl` 只作为备选，不在同一轮同时安装两套训练框架。

### 6.2 首轮 PPO 候选参数

| 参数 | 首轮值 | 说明 |
|---|---:|---|
| `num_envs` | 4096 | smoke 先用 64/1024 |
| `num_steps_per_env` | 32–64 | 每轮约 13–26 万转移 |
| `max_iterations` | 由 JSON 配置 | 先 smoke，再主实验 |
| `num_learning_epochs` | 5 | 不盲目堆 epoch |
| `num_mini_batches` | 4–8 | 视显存和更新吞吐 |
| `learning_rate` | `3e-4` | 固定；`schedule=fixed` |
| `gamma` | `0.99798` | 保持约 10s 物理视界 |
| `lambda` | `0.95` | GAE |
| `clip_param` | `0.2` | PPO 标准值 |
| `entropy_coef` | `0.005` | 先小后调 |
| `value_loss_coef` | `1.0` | 首轮固定 |
| `max_grad_norm` | `1.0` | 防止梯度爆炸 |
| actor/critic hidden | `[256, 256]` | 272 维输入，先中等容量 |

所有参数只通过 JSON 修改。`rsl_rl==2.3.3` 的 adaptive schedule 会在每个 minibatch 内把
学习率在 `1e-5` 与 `1e-2` 之间来回调整；P2 实测熵从 `2.8` 升到 `3.5`，评估在超时和出界
之间震荡。因此主链改为 `schedule=fixed`，只记录实际 PPO 迭代数、rollout 数、value loss、
surrogate loss、entropy、固定学习率、action noise std 和吞吐。KL 与 gradient norm 需要
额外适配器，首轮不以它们作为门槛。

### 6.3 PPO 的 batch 与 DDPG 的 batch_size/ratio 不同

PPO 没有 replay buffer，也没有 DDPG 那种 `batch_size`、`buffer_size`、`learning_starts`
或 behavior ratio。对应的量是：

```text
rollout_transitions = num_envs × num_steps_per_env
minibatch_size      = rollout_transitions / num_mini_batches
updates_per_iter    = num_learning_epochs × num_mini_batches
```

例如 `4096 × 32 = 131072` 条转移，`num_mini_batches=8` 时每个 minibatch 为 16384 条；
每轮做 `5 × 8 = 40` 次参数更新，每条转移被复用 5 次。`num_steps_per_env` 是每个环境的
rollout 长度，不是 episode 长度，episode 可以跨越多个 PPO iteration。

显存不足时，优先降低 `num_envs`、`num_steps_per_env` 或提高 `num_mini_batches`，不要为了
凑一个 batch size 去改 rsl_rl 源码。每次改动都记录 `config_sha256`，不要只保留最后一版配置。

### 6.4 首轮实验矩阵

| ID | 配置 | 环境数 | 场景 | 目的 | 进入条件 |
|---|---|---:|---|---|---|
| P0 | `p1_ppo_smoke.json` | 64 | S1 | 验证 loss、entropy、fixed LR、checkpoint | G2/G3 通过 |
| P1 | `p1_ppo_smoke_mixed.json` | 64 | S1+S2 | 验证不是只会直行 | P0 通过 |
| P2 | `p1_ppo_main.json` | 1024 | S1+S2 | 测吞吐和显存，不追求最终成绩 | P1 通过 |
| P3 | `p1_ppo_main.json` | 4096 | S1+S2 | 主实验，seeds `0,1,2` | P2 无 NaN/OOM/串扰 |
| P4 | `p1_ppo_c1.json` | 4096 | C1 随机单圆柱/center block | 场景扩展 | P3 通过 S1/S2 门 |
| P5 | `p1_ppo_c2.json` | 4096 | C2 blocking left/right | 学习绕行方向 | P4 不退化 |
| P6 | `p1_ppo_c3.json` | 4096 | C3 slalom/S3/混合 | 连续避障和泛化 | P5 不退化 |
| A1 | `p1_ppo_time_bonus.json` | 4096 | S1+S2 | 成功时间奖励消融 | P3 完成后单独跑 |
| A2 | `p1_ppo_bc_warmstart.json` | 4096 | S1+S2 | BC warm-start 消融 | P3 完成后单独跑 |

P0 和 P1 已完成 10-iteration 连通性 smoke，但都只覆盖 320 个控制步，不能据此判断收敛或
避障能力。P2 已完成 300 iteration：best/final 均达到 S1 `3/3`、S2 `3/3`、yaw 饱和
`0.144`。P3 seed 0 已完成 4096×5000 并通过审计；seed 1 正在训练，seed 2 排队等待。
500-iteration 探针已 `PASS` 新评估门，S1/S2 各 `3/3`，平均到达时间约 `10.2s / 10.8s`。

每次只改变一个变量；P4–P6 的课程扩展不能覆盖 P3 的主结果。8 卡并行时，优先把 P3 的
seed 0/1/2 或不同消融拆成独立进程，每个进程固定一张卡，不做跨进程梯度同步。

## 7. 场景课程与评估集

### 7.1 训练课程

课程必须由 JSON 明确描述，不能藏在代码分支里。建议分四个阶段：

| 阶段 | 场景 | 目的 |
|---|---|---|
| C0 | 固定 S1 + S2 | 验证 PPO 和批量环境 |
| C1 | 随机单圆柱 + center block | 学会不同障碍位置 |
| C2 | blocking left/right | 学会选择绕行方向 |
| C3 | slalom + S3 offset gate + 混合场景 | 学会连续和狭窄通道 |

阶段切换必须记录 checkpoint、成功率、碰撞率和时间；如果性能显著下降，回退上一阶段，
不得直接跳到更难场景。第一版不允许加入未在 JSON 中声明的场景。

### 7.2 训练/评估隔离

- 训练场景和评估场景必须分开；
- 固定 S1/S2 评估 seeds 仍为 `0,1,2`，用于与 goal6/goal7 对照；
- 专家场景池使用固定 seed 生成，并额外保留 20%–30% 未参与训练的布局；
- 不允许因为某个 seed 失败就换 seed 或删除失败 episode；
- 报告应包括 success、collision、out_of_bounds、timeout、min clearance、time_to_goal、
  path_length、action saturation 和每个场景族的成功数。

### 7.3 评估门

第一阶段解析评估至少满足：

- S1 `>=2/3`；
- S2 `>=1/3`；
- 无 NaN/Inf、无全零动作；
- yaw 动作饱和率 `<0.5`，两维同时饱和率 `<0.1`；`u_dot` 饱和率单独报告，因为时间最优
  策略会在最大加速度上做 bang-bang 控制，不能把“全程最大推力”误判成失控；
- 全组件饱和率仍写入 `summary.json`，用于横向比较，但不单独作为门；
- 最后 20% PPO 迭代无系统性崩溃；
- 4096 环境 rollout 无状态串扰。

扩展场景只报告，不把其成功率混入 S1/S2 主门。OceanSim P1-A 使用最终选出的 checkpoint，
先通过 50Hz 延迟门，再跑 S1/S2 × seeds `0,1,2`；延迟中位数和 p95 都必须 `<=20ms`。

## 8. 分阶段执行计划

### G0：文档与契约冻结

- [x] 新建本文件。
- [x] 在 `doc/01_技术栈.md` 增加 PPO/IsaacLab 当前路线和两条训练/评估轨道。
- [x] 在 `doc/02_执行计划.md` 增加 G0–G8 阶段表。
- [x] 冻结 17 维观测、动作范围、成功定义、50Hz、120m 和 episode 上限。
- [x] 明确不修改 IsaacLab/rsl_rl 源码；异常走补丁流程。
- [x] 人工审阅本文件、技术栈和执行计划后，才进入 G1（2026-10-29 确认后执行 G1）。

### G1：隔离安装与 PPO 后端

- [x] 记录 `oceansim` conda 环境和 IsaacLab venv 的 pip freeze SHA-256。
- [x] 只在 `/data/yhx/underwater-robotics/envs/venvs/isaaclab` 安装 `rsl_rl`。
- [x] 使用 IsaacLab 官方安装入口，不手写改动 `site-packages`：

```bash
source /data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/activate
cd /data/yhx/underwater-robotics/simulator/IsaacLab
OMNI_KIT_ACCEPT_EULA=YES ./isaaclab.sh --install rsl_rl
```

- [x] 验证 `import rsl_rl`、`isaaclab_rl.rsl_rl`、`RslRlVecEnvWrapper`。
- [x] 重新跑 `scripts/audit_isaaclab_environment.py`。
- [x] 确认 `oceansim` pip freeze 未变化。
- [x] 输出 `artifacts/infra/isaaclab/ppo_install_manifest.json`。
- [x] 执行 `pip check`，确认 rsl_rl 的依赖没有破坏 Isaac Sim 5.0 / PyTorch 2.7。
- [x] 禁止 `--force-reinstall`、`--upgrade` 和 `pip install -U`；安装失败或尝试升级
      Isaac Sim/PyTorch/NumPy/Pillow 时立即停止并报告。

G1 验收结果：

| 检查项 | 结果 |
|---|---|
| 安装包 | `rsl-rl-lib==2.3.3`，`isaaclab_rl==0.2.4`（editable） |
| IsaacLab venv freeze SHA-256 | 安装前 `4e8e7ecc…38883`，安装后 `c6b9ac39…dbf8` |
| `oceansim` freeze SHA-256 | 前后均为 `08f75112…f0548`，未变化 |
| 核心版本 | Isaac Sim 5.0.0.0、PyTorch 2.7.0+cu128、NumPy 1.26.0、Pillow 11.2.1 未变 |
| `pip check` | `No broken requirements found.` |
| 后端导入 | `rsl_rl`、`isaaclab_rl.rsl_rl`、`RslRlVecEnvWrapper` 全部 PASS |
| CUDA | `cuda_available=true`，可见 8 张 GPU |
| IsaacLab 源码 | v2.2.1（`0f00ca2`），`git status` 为空 |

证据：

```text
artifacts/infra/isaaclab/ppo_install_manifest.json
artifacts/infra/isaaclab/ppo_backend_verify.json
artifacts/infra/isaaclab/environment_audit_ppo_g1.json
```

服务器日志保留在 `/data/yhx/underwater-robotics/logs/ppo_isaaclab/`。官方
`--install rsl_rl` 入口在 v2.2.1 会先确保 IsaacLab 各扩展为 editable 安装，因此日志中会
出现 `isaaclab_rl`、`isaaclab`、`isaaclab_assets`、`isaaclab_mimic`、`isaaclab_tasks` 的
安装记录；实际 freeze 差异只有上面四个新增包，已有版本没有变化。脚本末尾因未交互接受
Omniverse EULA 而退出，但 `rsl_rl` 安装和 `isaaclab_rl` 注册在此之前已经完成，后端验证
为 PASS。**后续不要无必要重复执行整个安装脚本。**

### G2：外部任务包与 JSON 配置

- [x] 新建 `src/auv_isaaclab/`，不把任务直接塞进 IsaacLab 源码树。
- [x] 实现 `P1AvoidanceEnvCfg.from_json()` 和严格字段校验。
- [x] 实现 `build_rsl_rl_runner_cfg()`，JSON 字段覆盖 PPO 配置。
- [x] 建立 `configs/ppo/isaaclab/schema/ppo_experiment.schema.json`。
- [x] 至少提供 `p1_ppo_base.json`、`p1_ppo_smoke.json`、`p1_ppo_main.json`。
- [x] 所有输出目录保存 `resolved_config.json` 和 `config_sha256`。
- [x] 配置缺字段、未知字段、维度不匹配时必须 fail-fast。

G2 验收结果：

| 检查项 | 结果 |
|---|---|
| 外部任务包 | `src/auv_isaaclab/`，配置层不依赖 IsaacLab/Omniverse |
| 严格校验 | 未知字段、缺字段、错误类型、错误 dt 比、错误奖励范围均 fail-fast |
| `extends` | 支持递归继承，检测循环，保留 base→child 源文件链 |
| 派生维度 | `17` 维原始观测、`16` 帧历史、`272` 维策略输入 |
| PPO batch | base `4096×32=131072`，8 个 minibatch，每个 `16384` |
| 配置哈希 | 当前 base `2fde0a71…6a78c`、smoke `b30bbc74…49caa`、mixed `6e7ddb41…cfaf6`、main `cc02257b…c44f7` |
| 配置测试 | focused `20 passed`，全量 `223 passed` |
| 静态检查 | `ruff check`、`ruff format --check`、`mypy` 全部 PASS |

证据：`artifacts/ppo_isaaclab/config_validation/summary.json`。`validate_ppo_config.py`
会写 `resolved_config.json` 和 `config_manifest.json`；训练入口复用同一份 `ConfigBundle`，
不需要在训练脚本里重新拼参数。

### G3：批量环境对齐

- [x] 实现批量二维运动学、碰撞、越界、goal 和 reset（`P1BatchCore`，S1/S2）。
- [x] 实现批量解析 13 扇区，保持 1300 射线和 120m 契约。
- [x] 实现 history 环形缓存和每个环境独立 reset。
- [x] `N=1` 与现有 `PlanarAvoidanceEnv` 在同一初始状态和动作序列下逐项对齐。
- [x] 校验观测 `max_abs_diff < 1e-6`、reward `< 1e-6`、termination 一致。
- [x] 增加 64/256/1024/4096 环境无 NaN、无 OOM、无串扰测试。
- [x] 输出吞吐 JSON：control steps/s、env steps/s、GPU 显存、p50/p95 step 延迟。
- [x] 用空 `InteractiveScene` 把批量核心接进 `DirectRLEnv`，不改 IsaacLab 源码。
- [x] 在服务器 GPU 0 上用 `float32` 重跑固定动作序列，S2 最大观测差 `3.13e-7`。
- [x] 64 环境 S1+S2 连续 540 步，验证 S2 碰撞后的自动 reset 和 history 清理。

G3 验收结果：

| 检查项 | 结果 |
|---|---|
| 固定动作序列 | S1/S2 观测、reward、termination/truncation 均对齐 |
| S1 完整 episode | 1100 步到达 goal，`terminated=true`、`truncated=false` |
| S2 完整 episode | 511 步碰撞，`terminated=true`、`truncated=false` |
| 最大观测差 | float64 批量核心：S1 `0.0`、S2 `3.47e-9`；DirectRLEnv float32：S2 约 `3.13e-7`（门限 `<1e-6`） |
| 最大 reward 差 | `5.56e-17`，门限 `<1e-6` |
| DirectRLEnv | 空场景、headless、`N=1` 启动；观测 `[1,272]`，reset 约 `0.16s` |
| 批量状态 | 1/64/256/1024/4096 全部无 NaN/Inf、无状态串扰、无残留 done 环境 |
| 自动 reset | 64 环境混合场景 540 步：32 次碰撞、32 次 reset、history reset 误差 `0.0` |
| 局部测试 | 批量核心与配置 focused `30 passed`，全量 `238 passed` |
| 静态检查 | `ruff`、`ruff format --check`、`mypy` 均 PASS |

G3 吞吐（100 control steps，S1，单进程、单张 RTX 4090、GPU 0）：

| 环境数 | control steps/s | env steps/s | p50 ms | p95 ms | 进程结束后 GPU MiB |
|---:|---:|---:|---:|---:|---:|
| 1 | 327.3 | 327.3 | 3.03 | 3.18 | 2,614 |
| 64 | 313.3 | 20,054 | 3.18 | 3.28 | 2,616 |
| 256 | 389.9 | 99,807 | 2.55 | 2.70 | 2,634 |
| 1024 | 425.2 | 435,442 | 2.35 | 2.43 | 2,670 |
| 4096 | 333.9 | 1,367,459 | 2.96 | 3.34 | 2,788 |

每步固定开销仍然明显：环境数从 64 增到 4096 时，control steps/s 维持在约 313–425，
env steps/s 随环境数近似线性增长；单次运行有正常抖动，因此表中不做三次重复的严格比较。
下一轮若要做大吞吐优化，先定位空 `SimulationContext` 物理步和 Python 调度的固定开销，
不要先改射线求交。

证据：`artifacts/ppo_isaaclab/parity_n1_batch_core.json`、
`artifacts/ppo_isaaclab/scaling_probe.json`、`artifacts/ppo_isaaclab/scaling_probe.csv` 和
`artifacts/ppo_isaaclab/scaling_probe/`。服务器完整日志在
`/data/yhx/underwater-robotics/logs/ppo_isaaclab/`。

### G4：PPO smoke

- [x] 先跑 `num_envs=64`、短 rollout、`max_iterations=10` 的 smoke。
- [x] 只训练 S1，确认 loss、entropy、value、策略参数和 checkpoint 正常。
- [x] 再加入固定 S2，确认混合场景 rollout 正常。
- [x] smoke 失败时只改一个参数；保留失败配置和日志。
- [x] 不把 smoke 结果写成主实验结果。

G4 smoke 结果（只证明训练链连通，不证明收敛）：

| 配置 | 环境数 | 迭代 | transitions | entropy | 最终 checkpoint SHA-256 |
|---|---:|---:|---:|---:|---|
| S1 | 64 | 10 | 20,480 | `2.8264` | `41a403eb…ce3f` |
| S1+S2 | 64 | 10 | 20,480 | `2.8002` | `27dc95ad…b4ec` |

两次训练均满足：loss 全有限、entropy 为正、模型参数有限、确定性动作 shape `[64,2]` 且有限。
S1 训练约 `4.09s`，S1+S2 约 `4.71s`。10 个 iteration 只覆盖 320 个控制步，没有覆盖
S1 的 1100 步 goal 或 S2 的 511 步碰撞，因此不能据此判断成功率，也不能据此说 PPO
“已经学会避障”。

`rsl_rl==2.3.3` 的 `OnPolicyRunner.log()` 不直接暴露每次 PPO update 的 KL；本轮记录了
adaptive learning rate、entropy、value loss、surrogate loss 和 action noise std 作为连通性
检查。G5 如需 KL 门，应在不改上游源码的前提下增加独立日志适配器。

证据：`artifacts/ppo_isaaclab/ppo_smoke/s1/training_summary.json` 和
`artifacts/ppo_isaaclab/ppo_smoke/mixed/training_summary.json`。checkpoint 和 TensorBoard
只留服务器。

### G5：S1/S2 主实验

- [x] 实现 `scripts/eval_p1_isaaclab_ppo.py`，直接复用批量核心，不启动 Omniverse。
- [x] 实现 `scripts/audit_p1_ppo_seed.py`，独立校验 summary、record 数、best/final SHA-256、
      门禁和最后 20% 评估；失败时返回非零退出码并保留证据。
- [x] 每 N 个 iteration 做确定性评估，记录固定 seeds `0,1,2`、S1/S2 共 6 条。
- [x] checkpoint 选择基于 6 条固定解析评估；保存 `checkpoint_best/latest/final.pt` 和 SHA-256。
- [x] P2 1024 环境短主实验：best/final 均 `PASS`，S1 `3/3`、S2 `3/3`。
- [x] P3 4096×500 探针：新门禁 `PASS`，S1 `3/3`、S2 `3/3`，平均到达 `~10.2s / 10.8s`。
- [x] 使用 4096 环境跑满 seeds `0,1,2`；最终采用 `p1_ppo_main_lr1e4_v3.json`，三组
      best checkpoint 均通过固定 S1/S2 评估和独立复评。
- [x] 通过 S1/S2 门后，才进入课程扩展。

G5 评估口径说明：

- 6 条固定评估 = S1/S2 × seeds `0,1,2`；每个 seed 用固定 0.5m/3° 起始扰动，不换 seed、
  不删除失败 episode。
- 评估动作必须先乘回物理范围 `[0.5m/s², 15°/s]` 再送入批量核心。早期版本漏乘，导致偏航
  被放大 4 倍，相关“出界”结论已废弃。
- `action_saturation` 仍按全组件报告；门禁使用 `action_saturation_yaw < 0.5` 且
  `both_action_saturation < 0.1`。时间最优策略持续顶最大加速度是预期行为，不单独判失败。

关键证据：

| 实验 | 环境数 | 迭代 | 结果 | 证据 |
|---|---:|---:|---|---|
| P2 | 1024 | 300 | `PASS`，S1/S2 各 3/3，yaw 饱和 `0.144` | `outputs/ppo_isaaclab/p1_ppo_p2_1024_actionpen_v1/training_summary.json` |
| P3 探针 | 4096 | 500 | `PASS`，S1/S2 各 3/3，到达 `10.18s / 10.79s`，yaw 饱和 `0.292`，both `0.077` | `outputs/ppo_isaaclab/p1_ppo_p3_4096_timeoutpen_v1/training_summary.json` |
| P3 主实验 | 4096 | 5000 | seed 0 `PASS`；best iter 949，S1/S2 各 3/3，到达 `9.88s / 10.17s`，yaw 饱和 `0.211`，both `0.0987`，SHA-256 `406c020f…3bafb` | `outputs/ppo_isaaclab/p1_ppo_p3_main_s0_v1/` |
| P3 主实验 | 4096 | 5000 | seed 1 **最终门禁 FAIL**；best iter 1450 单独复评 PASS（S1/S2 各 3/3，到达 `9.73s / 9.89s`，both `0.0418`）；iter 5000 虽 S1/S2 各 3/3，但 both `0.1717` 超门；最后 20% 中 20/21 次 FAIL | `outputs/ppo_isaaclab/p1_ppo_p3_main_s1_v1/`、`artifacts/ppo_isaaclab/eval_analytic/p3_main_s1_best/summary.json` |
| P3 v3 | 4096 | 5000 | seed 0：best iter 3850，独立 S1/S2 各 3/3，到达 `9.67s / 9.74s` | `artifacts/ppo_isaaclab/p3_lr1e4_s0_audit.json` |
| P3 v3 | 4096 | 5000 | seed 1：best iter 4850，独立 S1/S2 各 3/3，到达 `9.78s / 9.84s` | `artifacts/ppo_isaaclab/p3_lr1e4_s1_audit.json` |
| P3 v3 | 4096 | 5000 | seed 2：best iter 4250，独立 S1/S2 各 3/3，到达 `9.77s / 9.97s` | `artifacts/ppo_isaaclab/p3_lr1e4_s2_audit.json` |

P3 500-iteration 探针的逐点门禁显示：训练中期会阶段性从成功退化为出界，因此不能只看
最后一个 checkpoint。当前正式训练每 50 iteration 做一次固定评估，并保存历史最佳
`checkpoint_best.pt`。`best_score` 按以下顺序比较，前一项相同才看后一项：
`门禁是否通过 → 总成功数 → 平均到达时间（越短越好） → 平均最小 clearance（越大越好）`。

| iter | 门禁 | S1 | S2 | 平均到达 S1/S2 | yaw 饱和 | both 饱和 |
|---:|---|---:|---:|---|---:|---:|
| 50 | FAIL | 0 | 0 | 出界 | 1.000 | 0.000 |
| 100 | FAIL | 0 | 0 | 出界 | 0.000 | 0.000 |
| 150 | PASS | 3 | 3 | 25.42 / 25.77s | 0.080 | 0.000 |
| 200 | PASS | 3 | 3 | 26.22 / 26.37s | 0.100 | 0.000 |
| 250 | PASS | 3 | 3 | 26.71 / 26.57s | 0.111 | 0.000 |
| 300 | PASS | 3 | 3 | 24.12 / 24.21s | 0.098 | 0.000 |
| 350 | FAIL | 0 | 0 | 出界 | 0.235 | 0.172 |
| 400 | FAIL（饱和） | 3 | 3 | 10.21 / 10.47s | 0.156 | 0.135 |
| 450 | FAIL（饱和） | 3 | 3 | 10.05 / 10.25s | 0.218 | 0.210 |
| 500 | PASS | 3 | 3 | 10.18 / 10.79s | 0.292 | 0.077 |

seed 0 的正式训练在 iter 499 得到的 best checkpoint 与 P3 500-iteration 探针的 best 具有相同
SHA-256（`efc327ac7ad832874002a036d5dd6562b83014029480e7cba58f9da5cfbdea7c`），说明在当前
固定 seed、奖励和固定学习率下，前 500 iteration 可重复。iter 449 的 latest checkpoint 虽然
S1/S2 都 `3/3`、到达约 `10s`，但 both 饱和率为 `0.221`，被门禁拒绝；这正是必须保留历史
best、而不是取 latest 的原因。iter 599 又把 best 更新为
SHA-256 `8f878be177ff7f43e409f823f222cb417ca7d8a91ba45918608cc63c14218429`，说明继续训练
仍可能在保持门禁的同时改善到达时间；iter 949 继续更新为
SHA-256 `406c020ff8f8baa286d891b8c519c816ceb75ef60e49029198a4bb3b3383bafb`，但 both
饱和率 `0.0987` 已贴近 `0.1` 门限，说明后续候选可能因动作饱和而回退。

seed 0 的 5000 iteration 已完整结束。最终第 5000 次周期评估为 `PASS`（S1/S2 各 `3/3`，
both 饱和 `0.0656`）。但按最后 20%（iter 4000–5000）的 20 次周期评估统计，有 14 次
`FAIL_GATE`：4000–4550 连续失败，4750 和 4800 再次失败，4600–4700 与 4850–5000 通过。
因此本轮只能声称“最终门通过且保留历史 best”，不能声称训练末段稳定，也不能只看最后一个
iteration。原始 summary 因
`entropy_positive=false` 被判为 `FAIL_ENTROPY`；检查发现 `losses_finite=true`、
`eval_errors=[]`、最终评估 `PASS`、`best_gates.pass=true`，且 Gaussian 策略的微分熵最小值为
`-2.1878`。Gaussian 微分熵可以为负，因此将门禁修正为“熵必须有限”，保留
`entropy_positive` 作为诊断字段。原始 `training_summary.json` 已备份为
`training_summary.pre_reclassification.json`，重分类记录写入 `reclassification.json`。
这次修正只改变审计逻辑，不改变训练、奖励、checkpoint 或评估轨迹。
对 `checkpoint_best.pt` 的独立 CPU 复评同样是 `PASS`（S1/S2 各 `3/3`，both `0.0987`）；
训练内 GPU 评估在 iter 950 记录为 both `0.0885`。两者都过门，微小差异来自 CPU/GPU
float32 运算顺序，不应把 CPU 复评数值当作 GPU 训练内评估的逐位复现。

seed 1 暴露了另一类真实问题：探索噪声没有收敛。`mean_noise_std` 从 iter 0 的 `0.99`
持续升到 iter 4999 的 `6.36`，而 seed 0 同期在约 `0.16–0.75` 间波动。seed 1 的 best
checkpoint（iter 1450）独立 GPU 复评为 `PASS`：S1/S2 各 `3/3`，到达 `9.73s / 9.89s`，
both 饱和 `0.0418`，SHA-256 `1514635…86a`。但最终 iter 5000 的确定性策略 `u_dot` 饱和率为
`1.0`，使 both 饱和率达到 `0.1717`，超过 `0.1` 门限；最后 20% 的 21 次周期评估中 20 次
`FAIL_GATE`。因此不能把 seed 1 记为 PASS，旧 watcher 也按门禁停止，未启动原配置的 seed 2。

单变量迭代配置 `p1_ppo_main_lowent_v2.json` 只把 `entropy_coef` 从 `0.005` 降为 `0.001`，
其余环境、奖励、网络、学习率和评估口径完全不变。seeds `0/1/2` 分别在 GPU `0/4/5` 上以
独立进程并行运行，输出目录为 `outputs/ppo_isaaclab/p1_ppo_p3_lowent_s{0,1,2}_v2/`。
旧 seed 1 输出不覆盖；v2 只有三组都通过同一审计后才可替代 G5 主结果。

v2 在约 iter 1,400–1,850 的 latest checkpoint 独立复评暴露了第二种失稳：s0 与 s1 的
yaw 饱和率均为 `1.0`，S1/S2 六条全部出界；s2 latest 仍为 `PASS`。这不是可接受的稳定
训练，因此停止 v2 的 5000-iteration 矩阵，保留 checkpoint、日志和三次独立复评。证据写入
`artifacts/ppo_isaaclab/p3_lowent_v2_early_stop.json` 及同目录下的三份 summary。

v3 回到 base 的 `entropy_coef=0.005`，只把 `learning_rate` 从 `3e-4` 降到 `1e-4`，用于检验
后期振荡是否主要来自更新步长过大。seeds `0/1/2` 分别在 GPU `0/4/5` 上运行，输出目录为
`outputs/ppo_isaaclab/p1_ppo_p3_lr1e4_s{0,1,2}_v3/`。

v3 三组均完整跑满 5,000 iteration。正式交付按本文件的 checkpoint 选择规则取
`checkpoint_best.pt`；`checkpoint_final.pt` 只用于稳定性诊断，不用于 G6/G7。三组 best
checkpoint 的 SHA-256、训练内门禁和独立复评均由
`scripts/audit_p1_ppo_seed.py` 交叉核对：

| seed | best iter | 独立 S1/S2 | 到达时间 S1/S2 | yaw 饱和 | both 饱和 | 最后 20% PASS/FAIL | final iter 5000 诊断 |
|---:|---:|---:|---|---:|---:|---:|---|
| 0 | 3850 | 3/3, 3/3 | 9.67 / 9.74s | 0.0349 | 0.0349 | 15 / 6 | FAIL_GATE |
| 1 | 4850 | 3/3, 3/3 | 9.78 / 9.84s | 0.0000 | 0.0000 | 18 / 3 | FAIL_GATE |
| 2 | 4250 | 3/3, 3/3 | 9.77 / 9.97s | 0.1266 | 0.0003 | 14 / 7 | FAIL_GATE |

三组最后 20% 均为多数评估通过；末端存在 3–7 次局部坍缩，说明 PPO 在固定学习率下仍会
进入不稳定相位。由于本任务训练的是单向策略并用固定验证集选择 checkpoint，交付和后续
课程评估统一使用 best checkpoint；final checkpoint 的失败不得被隐藏，也不得被描述成
“末段稳定”。汇总证据见 `artifacts/ppo_isaaclab/p3_lr1e4_v3_acceptance.json`。

### G6：场景课程与泛化

G6 的代码准备已完成，并已在 G5 三 seed 通过后启动课程训练。C1/C2/C3 是三条独立的
课程臂，每条都从基础配置开始训练，分别保留自己的 best checkpoint；它们不是共享同一个
checkpoint 的串行课程。

- 新增 `src/auv_isaaclab/tasks/p1_avoidance/scenarios.py`，把 C1/C2/C3 定义为确定性的
  场景组，而不是单个写死布局：
  - `c1`：8 个变体（4 个随机单圆柱 + 4 个 center block）；
  - `c2`：8 个变体（4 个 blocking left + 4 个 blocking right）；
  - `c3`：7 个变体（6 个 slalom + 1 个 offset gate）。
- 训练 reset 可重采样场景组成员；评估可以固定具体 variant，避免“训练随机、评估也随机”的
  不可复现问题。
- 新增 `p1_ppo_c1.json`、`p1_ppo_c2.json`、`p1_ppo_c3.json`，已通过严格 JSON 校验。
- 评估侧支持 `(key, variant)` 和逐实例统计；扩展场景只写入 `scenario_instances`，动作饱和
  与 S1/S2 成功门只按 S1/S2 记录计算。
- `configs/ppo/isaaclab/curriculum_evals/eval_c1.json`、`eval_c2.json`、`eval_c3.json`
  分别覆盖 8/8/7 个固定变体，并保留 S1/S2 × seeds `0,1,2` 回归门。
- C3 的 offset gate 含 4 个 box；64 环境课程 smoke 发现批量射线求交的 `near/far` 初值
  形状是 `[N, rays]` 而轴求交结果是 `[N, rays, boxes]`，已修正为后者，并新增
  `c3_gate_00` 回归测试。C1/C2/C3 的 64 环境零动作 smoke 均通过。
- C1/C2/C3 均以 4096 环境训练到 JSON 指定上限，并通过
  `scripts/audit_p1_curriculum_stage.py` 的单阶段审计；训练内周期评估与独立复评使用同一批
  固定实例和同一 checkpoint SHA。
- 课程训练前的 v3 零样本基线保留为未见布局参照；最终报告中仍需把“课程训练结果”和
  “未参与训练的 holdout 结果”分表，不能合并成同一组数字。
- 最近一次完整本地质量门为全量 `269 passed`，`ruff`、`ruff format --check`、`mypy`
  全部通过；OceanSim 评估脚本后续有改动，正式收尾前必须再跑一次全量质量门。

旧 v1 seed 0 best 的零样本基线为 C1 `23/24`、C2 `22/24`、C3 `18/21`，但 G5 最终选中的
是 lr=`1e-4` 的 v3 seed 0 best，因此 G6 必须使用同一条交付策略重新建立基线。v3 seed 0
（iter 3849）的零样本结果写入
`artifacts/ppo_isaaclab/eval_curriculum_zero_shot_v3_s0/`：

| 评估集 | S1/S2 | 扩展场景成功 | 说明 |
|---|---:|---:|---|
| `eval_c1` | `3/3, 3/3` | `23/24` | 8 个 C1 变体 × 3 seeds；1 次碰撞 |
| `eval_c2` | `3/3, 3/3` | `20/24` | 8 个 C2 变体 × 3 seeds；4 次碰撞 |
| `eval_c3` | `3/3, 3/3` | `11/21` | 6 个 slalom + 1 个 box offset gate；10 次碰撞 |

这只是“已训练 S1/S2 的 checkpoint 在未见布局上的零样本结果”，不是 C1/C2/C3 课程训练结果；
它证明评估链、逐实例统计和 C3 box 求交可以工作，也说明后续课程训练的主要价值应是提高
C3 和更复杂布局的稳健性，而不是重新验证 S1/S2。

G6 预注册验收口径（在课程训练启动前冻结，避免事后挑选标准；v3 基线变化不放宽门限）：

- 任何阶段都必须保留 S1 `>=2/3`、S2 `>=1/3`、yaw 饱和 `<0.5`、both 饱和 `<0.1`；
- C1 最终独立评估至少 `22/24`，C2 至少 `21/24`，C3 至少 `19/21`；
- C1/C2/C3 的绝对门限保持不变；旧 v1 基线和 v3 基线都只作参照，不用于事后降低门限；
- `c3_gate_00` 至少 `2/3`；
- 训练内周期评估与独立复评必须使用同一组固定实例；独立复评失败时，不得改称训练内结果通过；
- 未通过时保留 checkpoint、配置、逐 episode CSV 和失败原因；只允许建立新 JSON/新输出目录做
  单变量迭代，不覆盖失败证据或事后降低门限。

- [x] 将 C1–C3 代码、JSON 和评估配置同步到远端，哈希与本地一致；C1/C2/C3 配置校验 PASS。
- [x] C3 64 环境、2 iteration smoke PASS，批量 box ray-cast 无异常。
- [x] 用 v3 seed 0 best 重建 C1/C2/C3 零样本基线。
- [x] 新增 `scripts/audit_p1_curriculum_stage.py`，把训练完整性、best checkpoint SHA、
      独立复评 SHA、S1/S2 回归门、阶段成功数和 `c3_gate_00` 合并为单阶段审计 JSON。
- [x] 完成 C1 单圆柱/center block 训练（4096 环境）并通过独立复评。
- [x] 完成 C2 blocking left/right 训练（4096 环境）并通过独立复评。
- [x] 完成 C3 slalom + offset gate + 混合训练（4096 环境）并通过独立复评。
- [x] 每个阶段保存独立 checkpoint、配置、指标和失败样本。
- [x] 保留课程训练前的未见布局零样本基线，并在最终报告中与课程结果分表。
- [x] 分布外结果只报告，不修改 S1/S2 主门。

G6 阶段审计汇总（正式交付只看每条课程臂自己的 best checkpoint）：

| 阶段 | best iter | 独立 S1/S2 | 阶段评估 | yaw / both 饱和 | best SHA-256 | 审计 |
|---|---:|---:|---:|---:|---|---|
| C1 | 3600 | 3/3, 3/3 | 24/24 | 0.0075 / 0.0075 | `5205633e…b2a06d` | `PASS` |
| C2 | 2000 | 3/3, 3/3 | 24/24 | 0.0875 / 0.0875 | `ed7d1353…8172872` | `PASS` |
| C3 | 1250 | 3/3, 3/3 | 21/21 | 0.0213 / 0.0213 | `91dea4d4…399c723e` | `PASS` |

三组 final checkpoint 的训练内末段评估均为 `FAIL_GATE`，原因都是 S1/S2 的 both 饱和率超过
`0.1`；该结果保留在 `training_summary.json` 的诊断字段，不能用来替代 best checkpoint，
也不能据此宣称训练末段稳定。

C1 阶段结果（2026-10-30）：

| 项目 | 结果 |
|---|---|
| best iteration | `3600` |
| 独立 S1/S2 | `3/3`、`3/3` |
| 独立 C1 | `24/24`，平均到达 `9.95s`，平均最小 clearance `1.26m` |
| yaw / both 饱和 | `0.0075 / 0.0075` |
| best checkpoint SHA-256 | `5205633eec990c1ab1cead7d1caa416ea773ea9ea1f637ed18d1ef1199b2a06d` |
| 审计 | `artifacts/ppo_isaaclab/g6_c1_audit.json`，`PASS` |

训练内最后一个 checkpoint 的评估仍为 `FAIL_GATE`：C1 成功仍为 `24/24`，但 S1/S2 的
both 饱和率为 `0.1336`，超过冻结门 `<0.1`。该失败保留在训练摘要诊断字段中；正式 C1
结果使用 best checkpoint，且独立复评 SHA 与训练内 best SHA 一致。

C2 阶段结果（2026-10-30）：

| 项目 | 结果 |
|---|---|
| best iteration | `2000` |
| 独立 S1/S2 | `3/3`、`3/3` |
| 独立 C2 | `24/24`，平均到达 `10.01s`，平均最小 clearance `1.26m` |
| yaw / both 饱和 | `0.0875 / 0.0875`（训练内 best 门禁） |
| best checkpoint SHA-256 | `ed7d13539aec1239b62a9464387898678dd4c5c52a14221d57ec197648172872` |
| 审计 | `artifacts/ppo_isaaclab/g6_c2_audit.json`，`PASS` |

C2 末段 20 次训练内评估全部为 `FAIL_GATE`，原因均为 S1/S2 的 both 饱和率超过 `0.1`；
最终 iteration 的成功数仍为 S1/S2 各 `3/3`、C2 `24/24`，但饱和率 `0.320` 不合格。
该失败不会被隐藏，也不据此宣告“末段稳定”；正式 C2 交付仍使用 best checkpoint。

C3 阶段结果（2026-10-30）：

| 项目 | 结果 |
|---|---|
| best iteration | `1250` |
| 独立 S1/S2 | `3/3`、`3/3`，平均到达 `9.68s / 9.89s` |
| 独立 C3 | `21/21`，平均到达 `9.74s`，平均最小 clearance `0.75m` |
| `c3_gate_00` | `3/3`，平均到达 `9.77s`，平均最小 clearance `0.56m` |
| yaw / both 饱和 | `0.0213 / 0.0213` |
| best checkpoint SHA-256 | `91dea4d4612190786c56f22efed02de5313fec9867da0c558bd6a179399c723e` |
| 审计 | `artifacts/ppo_isaaclab/g6_c3_audit.json`，`PASS` |

C3 末段 checkpoint 同样为 `FAIL_GATE`，原因是 S1/S2 的 both 饱和率超过冻结门；成功数未下降，
但动作稳定性不合格。正式交付仍固定使用 C3 best checkpoint，且训练内 best SHA 与独立复评
SHA 一致。至此 G6 三阶段审计均 PASS，可以进入 G7。

最终交付模型预注册（在 G6 结果产生前冻结）：

- G6 阶段门按每条课程臂自己的 best checkpoint 独立判定，不能用另一条臂的结果替代。
- G7 最终交付模型固定选择 **C3 综合臂的 best checkpoint**，因为 C3 同时混合 S1/S2、
  C1、C2 和 C3，是三条臂中覆盖面最广的一条。
- C3 臂的 S1/S2、C1/C2/C3、OceanSim 和可视化必须全部使用同一个 checkpoint SHA-256；
  不允许按测试集结果从三条臂中事后挑“最好看”的模型，也不允许把不同 checkpoint 的结果
  拼成同一份最终证据。
- 若 C3 best 未通过 G6 门，则 G6 整体不通过；应新建版本化配置和输出目录做单变量迭代，
  保留本次失败证据，不降低门限，也不改为挑选 C1/C2 臂规避失败。

### G7：最终评估与可视化

- [x] 对最终 checkpoint 跑解析 S1/S2 主评估和扩展场景评估。
- [x] 在 OceanSim P1-A 上完成三次重复延迟门；结论为 FAIL，未进入正式稳定性通过分支。
- [ ] 输出 `final_evaluation.json`、`final_report.md` 和吞吐摘要。
- [ ] 生成 2D 轨迹/GIF 时，明确标注是解析还是 OceanSim 结果。
- [x] 不改用 5Hz 结果冒充 50Hz。

G7 使用的交付模型已经按预注册规则固定为 **C3 综合臂 best checkpoint**：

```text
checkpoint: outputs/ppo_isaaclab/p1_ppo_c3_lr1e4_v1/checkpoint_best.pt
SHA-256:    91dea4d4612190786c56f22efed02de5313fec9867da0c558bd6a179399c723e
```

以下所有解析最终评估都使用这一 SHA；没有从 C1/C2/C3 三条臂中事后挑选模型。

#### G7.1 解析最终评估（PASS）

固定 seeds `0,1,2`，S1/S2 各 3 条，C1/C2/C3 使用各自的固定实例集。结果如下：

| 评估集 | 成功数 | 平均到达 | 备注 |
|---|---:|---:|---|
| S1 | `3/3` | `9.68s` | 固定 seeds，无碰撞/越界/超时 |
| S2 | `3/3` | `9.89s` | 固定 seeds，无碰撞/越界/超时 |
| C1 | `24/24` | `9.77s` | 8 个变体 × 3 seeds |
| C2 | `24/24` | `9.76s` | 8 个变体 × 3 seeds |
| C3 | `21/21` | `9.74s` | 7 个变体 × 3 seeds；含 `c3_gate_00` 3/3 |

所有最终解析评估的 yaw 饱和率与 both 饱和率均为
`0.02131552729550792`，低于冻结门 `0.5 / 0.1`。这说明 C3 best 在已训练的课程场景上
满足交付门；它不等价于对任意未见场景的泛化保证，也不能替代 OceanSim 结果。

服务器证据目录（需拉回本地后再写入最终报告）：

```text
artifacts/ppo_isaaclab/eval_final/analytic_c3_v1/
artifacts/ppo_isaaclab/eval_final/c1_c3_v1/
artifacts/ppo_isaaclab/eval_final/c2_c3_v1/
artifacts/ppo_isaaclab/eval_curriculum/c3_best_v1/
```

#### G7.2 OceanSim 延迟门与主评估（FAIL）

OceanSim 使用独立的 `oceansim` conda 环境，只做评估，不在其中安装或升级任何包。参数集中在
版本化 JSON 中：

```text
configs/rl/p1_ppo_oceansim_120m50hz.json
configs/rl/p1_ppo_oceansim_120m50hz_preferred_readback_v2.json
configs/rl/p1_ppo_oceansim_120m50hz_no_copy_v3.json
```

延迟门要求 median 和 p95 都 `<=20ms`。目前的有效实验如下：

| 实验 | GPU/回读设置 | median | p95 | 结论 |
|---|---|---:|---:|---|
| `final_c3_v1` | 全部 GPU 可见，CPU readback | `21.24ms` | `24.61ms` | FAIL |
| `final_c3_v1_retry1` | 全部 GPU 可见，CPU readback | `20.83ms` | `31.26ms` | FAIL |
| `final_c3_preferred_v2` | preferred `cuda:0`，全部 GPU 可见 | `27.93ms` | `38.79ms` | FAIL |
| `final_c3_no_copy_v3` | 禁止 array copy，全部 GPU 可见 | `33.53ms` | `52.49ms` | FAIL |
| `final_c3_profile2` | 全部 GPU 可见，CPU readback | `27.62ms` | `41.26ms` | FAIL |
| `final_c3_single_gpu_v4` | `CUDA_VISIBLE_DEVICES=0`，CPU readback | `18.21ms` | `19.46ms` | 单次 PASS，但不能作为稳定结论 |
| `final_c3_single_gpu_v5` | 单 GPU，完整 6 条评估 | `20.76ms` | `30.80ms` | FAIL |
| `final_c3_single_gpu_no_copy_v6` | 单 GPU + no copy | `18.30ms` | `24.56ms` | FAIL（median PASS，p95 FAIL） |
| `single_gpu_retry_v7` | 单 GPU，CPU readback，重复 1 | `18.08ms` | `19.71ms` | PASS；6 条评估也成功 |
| `single_gpu_retry_v8` | 单 GPU，CPU readback，重复 2 | `20.05ms` | `27.04ms` | FAIL，未进入 6 条评估 |
| `single_gpu_retry_v9` | 单 GPU，CPU readback，重复 3 | `19.90ms` | `22.07ms` | FAIL，median PASS / p95 FAIL |

`single_gpu_v4` 的 `trajectories.json` 显示 6 条轨迹都成功，但该次运行没有写出完整的
`evaluation_summary.json`，CSV 也只有表头，因此只能作为“延迟门出现过一次 PASS”的线索，
不能作为正式主评估。CSV 的额外字段序列化 bug 已修复；修复后的完整运行 `v5` 仍因 p95
`30.80ms` 失败，`v6` 仍因 p95 `24.56ms` 失败。

单 GPU、CPU readback 下的耗时拆解约为：

| 子项 | 典型耗时 |
|---|---:|
| policy inference | `0.5–0.7ms` |
| environment step | `17.7ms` |
| render tick（中位数） | `11.3ms` |
| scan | `3.7ms` |
| pose write | `0.8ms` |
| point transform | `0.5ms` |

场景点云约 `11,000` 点。瓶颈不是 PPO 推理，而是 OceanSim 渲染 tick 的偶发停顿，导致
p95 抖动。日志还提示 `1300×200` 低于 DLSS 最小输入尺寸；G7.3 已用官方
`SimulationApp(anti_aliasing=0)` 验证 carb 的 `/rtx/post/aa/op=0` 生效，但 p95 仍未
稳定过门。禁止通过修改 OceanSim 源码、降低分辨率语义、放宽门限或只报告一次最好运行来
“通过”延迟门。

三次重复的正式判定是 `1/3` 通过，因此结论为：**OceanSim 不能稳定维持 50Hz；G7 的
OceanSim 分支 FAIL，只保留“仅评估”用途。** `single_gpu_retry_v7` 的 6 条评估全部成功，
但它是单次运行，不能满足预先冻结的三次稳定门，不能写成正式 OceanSim 成功结论；`v7`
的轨迹和摘要保留为诊断证据。

最终聚合结果已写入：

```text
artifacts/ppo_isaaclab/final_evaluation.json   # status=FAIL
artifacts/ppo_isaaclab/final_report.md
```

最终 JSON 中 OceanSim 的 `latency_stability` 记录三轮 median/p95、每轮是否通过、
要求运行次数和 `all_passed=false`；聚合后的最差 median/p95 分别为 `20.05ms / 27.04ms`。

#### G7.3 延迟稳定化迭代（FAIL）

在 G7.2 的 `1/3` 之后，只对评估运行时做可回滚的单变量迭代，所有运行仍使用同一个 C3
best checkpoint、120m 量程、1300×200 分辨率、50Hz 控制周期、CPU readback 和 `<=20ms`
冻结门。证据集中在：

```text
artifacts/ppo_isaaclab/g7_3_latency_iterations.json
artifacts/ppo_isaaclab/g7_3_latency_iterations.md
```

| 运行 | 单变量变化 | CPU 绑定 | median | p95 | 结果 |
|---|---|---|---:|---:|---|
| `no_aa_v10` | `anti_aliasing=0` | 否 | `18.40ms` | `22.72ms` | FAIL |
| `pinned_v11` | 仅运行时 CPU 绑定 | `0-31` | `15.02ms` | `17.22ms` | PASS；6 条评估成功 |
| `pinned_v12` | 仅运行时 CPU 绑定 | `0-31` | `17.03ms` | `18.90ms` | PASS；6 条评估成功 |
| `pinned_v13` | 仅运行时 CPU 绑定 | `0-31` | `17.33ms` | `20.75ms` | FAIL；未进入 6 条评估 |
| `perf_v14` | AA 关闭 + IsaacLab 性能渲染子集 | `0-31` | `17.41ms` | `21.58ms` | FAIL |
| `threads16_v17` | `limit_cpu_threads=16` | `0-31` | `17.85ms` | `27.04ms` | FAIL |

CPU 绑定把稳态 median 降到 `15–17ms`，是唯一有实质收益的改动；但三次预注册运行仍是
`2/3`，第三轮 p95 越门 `0.75ms`。`pinned_v11` 和 `pinned_v12` 的 `trajectories.json`
与历史 `v7` 的 SHA-256 完全相同，均为
`8aa81281a4481a46c1a6ea9168bc8f4bed95e0482b2bb56a73f524e0a210c384`，说明 CPU 绑定没有
改变观测、动作、碰撞或成功轨迹。关闭抗锯齿、套用官方性能渲染子集、把 Kit 线程从 32
降到 16 都没有提供稳定余量，因此不能把这组结果写成“稳定 50Hz”。

G7 的离线入口和证据生成器已准备：

- `src/auv_rl/ppo_policy.py` 可直接从 rsl_rl checkpoint 重建 actor，维护训练时的 16 帧历史，
  并把归一化动作乘回 `[0.5m/s², 15°/s]`；因此 OceanSim conda 环境不需要安装 rsl_rl。
- `scripts/train_p1_oceansim.py --eval-policy-kind ppo` 复用原有 50Hz 延迟门和 6 条评估，
  输出新增 `policy_kind`、`checkpoint_sha256`，并保留 `observation_source` 分表。
- PPO/OceanSim 的 120m、50Hz、16 帧历史和延迟门参数集中在
  `configs/rl/p1_ppo_oceansim_120m50hz.json`；命令行只覆盖 checkpoint、输出和运行设备。
- `scripts/render_p1_rollout_2d.py --policy-kind ppo` 支持 S1/S2 与 C1-C3 固定变体，标题和
  metadata 明示 `analytic sensor`；真实 OceanSim 轨迹仍需在实际评估后另存并标注。
- `scripts/write_p1_ppo_final_evidence.py` 校验所有 checkpoint SHA、解析/课程门、OceanSim
  延迟门，并生成最终 JSON/Markdown；阈值来自
  `configs/ppo/isaaclab/final_gates.json`，不由报告脚本临时放宽。
- 已用 seed 0 best checkpoint 在本地解析环境复跑 S1/S2，均在约 `9.94s / 10.22s` 到达，
  与训练内评估的约 `9.91s / 10.19s` 一致；这只是入口验证，不替代最终 checkpoint 评估。

### G8：可选 OceanSim 声呐微基准

只有 G7 的 OceanSim 延迟门稳定通过后才允许启动，且只做 16/64 环境、固定时间窗的
microbenchmark。当前 G7 尚未稳定通过，因此 G8 保持 **BLOCKED**，不因“算力充裕”而先扩到
1024 环境。

G7.2 的三次重复是 `1/3`，G7.3 的 CPU 绑定版本也只有 `2/3`，均未达到 `3/3`，因此停止条件
仍然成立，不允许启动 G8。下面保留的是未来若重新开启实验时的检查清单，不是本轮待办；
本轮不运行 16/64 环境微基准。

- [ ] 记录每环境每帧渲染、scan、GPU→CPU 回读和分箱耗时；
- [ ] 记录显存、首步时间、reset 时间；
- [ ] 若 64 环境无法维持目标频率，结论写“OceanSim 只做评估”，不继续扩到 1024；
- [ ] 该实验不替代解析 PPO 主训练，也不改写主结果。

G8 的启动条件：

1. 同一配置（单 GPU + CPU readback）至少重复 3 次延迟门，每次 median/p95 均 `<=20ms`；
2. 每次都有完整摘要、逐 episode CSV 和明确 checkpoint SHA；
3. G7 的 6 条 OceanSim 主评估完成，且 S1 `>=2/3`、S2 `>=1/3`；
4. 若上述任一条件不成立，停止在“OceanSim 仅评估”的结论，不再扩大声呐并行规模。

## 9. 运行命令

以下 IsaacLab 训练/解析评估命令均已在本轮服务器验证。OceanSim 评估使用独立的
`oceansim` conda Python，见本节末尾。

```bash
PY=/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python
REPO=/data/yhx/underwater-robotics/src/plan_p1/settled_局部规划器_反应式
export PYTHONPATH="$REPO/src:/data/yhx/underwater-robotics/src/plan_p1/settled_全局规划器/src"
export OMNI_KIT_ACCEPT_EULA=YES

cd "$REPO"

"$PY" scripts/validate_ppo_config.py \
  --config configs/ppo/isaaclab/p1_ppo_main.json

"$PY" scripts/probe_p1_isaaclab_scaling.py \
  --config configs/ppo/isaaclab/p1_ppo_smoke.json \
  --num-envs 4096 \
  --steps 100 \
  --warmup-steps 10 \
  --seed 0 \
  --device cuda:0 \
  --headless \
  --output artifacts/ppo_isaaclab/scaling_probe/num_envs_4096.json

"$PY" scripts/summarize_p1_isaaclab_scaling.py \
  --input artifacts/ppo_isaaclab/scaling_probe \
  --csv artifacts/ppo_isaaclab/scaling_probe.csv \
  --summary artifacts/ppo_isaaclab/scaling_probe.json

"$PY" scripts/train_p1_isaaclab_ppo.py \
  --config configs/ppo/isaaclab/p1_ppo_smoke.json \
  --num-envs 64 \
  --seed 0 \
  --device cuda:0 \
  --output outputs/ppo_isaaclab/p1_ppo_smoke_s1_v1 \
  --no-eval \
  --headless

"$PY" scripts/train_p1_isaaclab_ppo.py \
  --config configs/ppo/isaaclab/p1_ppo_main.json \
  --num-envs 4096 \
  --seed 0 \
  --device cuda:0 \
  --output outputs/ppo_isaaclab/p1_ppo_p3_main_s0_v1 \
  --eval-config configs/ppo/isaaclab/eval_analytic.json \
  --headless

"$PY" scripts/eval_p1_isaaclab_ppo.py \
  --config configs/ppo/isaaclab/eval_analytic.json \
  --checkpoint outputs/ppo_isaaclab/p1_ppo_p3_main_s0_v1/checkpoint_best.pt \
  --output artifacts/ppo_isaaclab/eval_analytic \
  --device cuda:0 \
  --headless
```

注意：当前 pip 安装的 Isaac Sim 5.0 在 `SimulationApp.close()` 时可能强制以 0 退出，
因此批处理脚本不能只看 shell 退出码。解析/课程评估应读取 `summary.json` 的顶层
`status`；OceanSim 的 `evaluation_summary.json` 没有顶层 `status`，应读取
`main_success_gate` 和 `latency_gate.gate.passed`。

OceanSim 评估仍使用原来的 `oceansim` conda Python，不把 IsaacLab venv 当作 OceanSim 环境。
稳定性复测必须固定 `CUDA_VISIBLE_DEVICES=0`，每次使用新的输出目录，不能覆盖失败证据。
若复测 CPU 绑定方案，使用 `taskset -c 0-31` 包住 Python 进程；这只是运行环境隔离，
不是感知或控制参数。下面的正式基线命令保留原始口径，G7.3 的绑定命令见证据文件：

```bash
OCEANSIM_PY=/data/yhx/underwater-robotics/envs/conda_envs/oceansim/bin/python

CUDA_VISIBLE_DEVICES=0 "$OCEANSIM_PY" scripts/train_p1_oceansim.py \
  --config configs/rl/p1_ppo_oceansim_120m50hz.json \
  --eval-only \
  --eval-checkpoint outputs/ppo_isaaclab/p1_ppo_c3_lr1e4_v1/checkpoint_best.pt \
  --eval-policy-kind ppo \
  --eval-output artifacts/ppo_isaaclab/eval_oceansim/single_gpu_retry_v7 \
  --eval-seeds 0 1 2 \
  --device cuda:0 \
  --latency-threshold-ms 20 \
  --latency-steps 30 \
  --latency-warmup-steps 5 \
  --latency-scenario s2 \
  --calibration outputs/m4/p1/sonar_calibration/calibration.json
```

G7.3 的单变量配置对应命令只需替换 `--config` 和输出目录；例如 CPU 绑定 + 16 线程：

```bash
CUDA_VISIBLE_DEVICES=0 taskset -c 0-31 "$OCEANSIM_PY" \
  scripts/train_p1_oceansim.py \
  --config configs/rl/p1_ppo_oceansim_120m50hz_16threads_v6.json \
  --eval-only \
  --eval-checkpoint outputs/ppo_isaaclab/p1_ppo_c3_lr1e4_v1/checkpoint_best.pt \
  --eval-policy-kind ppo \
  --eval-output artifacts/ppo_isaaclab/eval_oceansim/threads16_v17 \
  --eval-seeds 0 1 2 \
  --device cuda:0 \
  --latency-threshold-ms 20 \
  --latency-steps 30 \
  --latency-warmup-steps 5 \
  --latency-scenario s2 \
  --calibration outputs/m4/p1/sonar_calibration/calibration.json
```

每次运行后必须读取 `evaluation_summary.json`，至少核对：`policy_kind=ppo`、
`observation_source=oceansim_simplified_pcl_120m`、`checkpoint_sha256` 等于 C3 best、
`jitter=false`、`main_success_gate`、`latency_gate.gate.passed`、median/p95 以及
S1/S2 成功数。该文件没有顶层 `status`，不能按 `status` 字段判门。若脚本以 0 退出但
上述门为 `false`，仍按失败处理；若摘要缺失，不能用 `trajectories.json` 代替正式摘要。

稳定性门的正式判定至少重复 3 次相同命令（只改输出目录），3/3 均 median/p95 `<=20ms`
才可进入 6 条主评估。单次 PASS 只能记录为线索。

## 10. 测试与质量门

每个代码阶段至少跑：

```bash
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/mypy
.venv/bin/pyright
.venv/bin/python -m pytest -q
```

IsaacLab 任务另外需要：

- `N=1` 批环境与单环境对齐；
- 批量 sensor 与当前 13 扇区 builder 对齐；
- history reset 独立性；
- terminated/truncated bootstrap；
- 自动 reset 不污染 old `next_observation`；
- 4096 环境初始化和连续 100 步无 NaN；
- JSON 解析器拒绝未知字段和错误维度；
- 4096 环境的 throughput 报告可重跑。

任何一项失败都不能进入下一阶段。

2026-10-30 最终复跑结果：

- `.venv/bin/python -m pytest -q`：`276 passed`；
- `ruff check .` 和 `ruff format --check .`：PASS；
- `mypy`：PASS（93 个 source files）；
- `pyright` 对 `tests/test_m4_p1_latency_gate.py` PASS；`scripts/train_p1_oceansim.py`
  仍有既有的 10 个错误（本地未安装 Isaac Sim/OceanSim 导致的可选导入，以及 NumPy
  `savez_compressed` 类型存根误报），这些不是本轮的配置开关引入的；
- 全仓 `pyright` 仍在既有 `tests/test_ppo_policy.py` 报 6 个 Torch 类型存根问题，该文件
  本轮未修改，不能把它写成本次改动引入的回归；
- 最终证据聚合器新增了多轮延迟门测试，聚焦测试 `11 passed`，GIF 相册脚本通过
  ruff/format/mypy/pyright。

## 11. 证据与目录

小型证据进入 Git：

```text
artifacts/infra/isaaclab/ppo_install_manifest.json
artifacts/infra/isaaclab/ppo_backend_verify.json
artifacts/infra/isaaclab/environment_audit_ppo_g1.json
artifacts/ppo_isaaclab/
  config_validation/summary.json
  p0_audit.json
  parity_n1_batch_core.json
  scaling_probe.json
  scaling_probe.csv
  scaling_probe/num_envs_*.json
  ppo_smoke/s1/training_summary.json
  ppo_smoke/mixed/training_summary.json
  ppo_p2_1024_actionpen_v1/training_summary.json
  ppo_p3_4096_probe_v1/training_summary.json
  p3_main_s0_audit.json
  p3_main_s1_audit.json
  p3_lr1e4_s0_audit.json
  p3_lr1e4_s1_audit.json
  p3_lr1e4_s2_audit.json
  p3_lr1e4_v3_acceptance.json
  p3_lowent_v2_early_stop.json
  p3_lowent_v2_early_stop/
  eval_analytic/p3_4096_it500_v2gate/summary.json
  eval_analytic/p3_main_s1_best/summary.json
  eval_analytic/p3_main_s1_best/episodes.csv
  eval_analytic/p3_lr1e4_s0_best_v3/summary.json
  eval_analytic/p3_lr1e4_s1_best_v3/summary.json
  eval_analytic/p3_lr1e4_s2_best_v3/summary.json
  g6_c1_audit.json
  g6_c2_audit.json
  g6_c3_audit.json
  eval_curriculum/c1_best_v1/summary.json
  eval_curriculum/c1_best_v1/episodes.csv
  eval_curriculum/c2_best_v1/summary.json
  eval_curriculum/c2_best_v1/episodes.csv
  eval_curriculum/c3_best_v1/summary.json
  eval_curriculum/c3_best_v1/episodes.csv
  eval_oceansim/single_gpu_retry_v7/evaluation_summary.json
  eval_oceansim/single_gpu_retry_v7/latency_gate.json
  eval_oceansim/single_gpu_retry_v7/eval_episodes.csv
  eval_oceansim/single_gpu_retry_v8/latency_gate.json
  eval_oceansim/single_gpu_retry_v9/latency_gate.json
  eval_oceansim/no_aa_v10/latency_gate.json
  eval_oceansim/pinned_v11/evaluation_summary.json
  eval_oceansim/pinned_v11/latency_gate.json
  eval_oceansim/pinned_v11/eval_episodes.csv
  eval_oceansim/pinned_v11/trajectories.json
  eval_oceansim/pinned_v12/evaluation_summary.json
  eval_oceansim/pinned_v12/latency_gate.json
  eval_oceansim/pinned_v12/eval_episodes.csv
  eval_oceansim/pinned_v12/trajectories.json
  eval_oceansim/pinned_v13/latency_gate.json
  eval_oceansim/perf_v14/latency_gate.json
  eval_oceansim/threads16_v17/latency_gate.json
  g7_3_latency_iterations.json
  g7_3_latency_iterations.md
  final_evaluation.json
  final_report.md
  throughput_summary.json
  visuals/ppo_c3_analytic_v1/contact_sheet.png
  visuals/ppo_c3_analytic_v1/index.html
  visuals/ppo_c3_analytic_v1/index_standalone.html
  visuals/ppo_c3_analytic_v1/*_seed0.gif
```

已从服务器同步到本地的 G7 解析证据：

```text
artifacts/ppo_isaaclab/eval_final/analytic_c3_v1/
artifacts/ppo_isaaclab/eval_final/c1_c3_v1/
artifacts/ppo_isaaclab/eval_final/c2_c3_v1/
```

最终证据文件已经生成，并如实记录 OceanSim 失败；不能把其中的 `status=FAIL` 改成 `PASS`：

```text
artifacts/ppo_isaaclab/final_evaluation.json
artifacts/ppo_isaaclab/final_report.md
```

以下只留服务器，不进普通 Git：

- `checkpoint_latest.pt`、`checkpoint_best.pt`、optimizer state；
- TensorBoard event、完整 rollout、视频；
- 大型 CSV、OceanSim 点云和 demo 数据。

## 12. Git 检查点与建议提交

每阶段结束先执行：

```bash
cd "/home/bubble/水下机器人"
git status --short
git diff --check
git diff --stat
```

只有用户明确要求时才执行 `git add`、`git commit` 和 `git push`。建议提交信息：

| 阶段 | 建议 commit |
|---|---|
| G0 | `docs: add ppo isaaclab execution plan` |
| G1 | `chore(isaaclab): add isolated ppo backend install and audit` |
| G2 | `feat(isaaclab): add json-configured p1 task skeleton` |
| G3 | `feat(isaaclab): add batched analytic p1 environment` |
| G4 | `test(isaaclab): add n1 parity and vector reset tests` |
| G5 | `feat(ppo): add rsl rl training and evaluation entrypoints` |
| G6 | `feat(ppo): add scenario curriculum configs` |
| G7 | `docs(ppo): record ppo isaaclab results` |
| G8 | `docs(isaaclab): record oceansim sonar microbenchmark boundary` |

## 13. 停止条件

出现以下任一情况，停止当前阶段并保留日志、配置和失败证据：

- `oceansim` conda 环境的 pip freeze 发生变化；
- 安装 PPO 后端需要升级 Isaac Sim、PyTorch、NumPy 或破坏约束文件；
- 需要直接修改 IsaacLab/rsl_rl 源码才能运行，但没有完成补丁审查；
- `N=1` 批量环境与单环境观测、奖励或终止条件不一致；
- 4096 环境出现 NaN、OOM、状态串扰或 history 污染；
- PPO smoke 在 S1 上完全学不动，且不是配置错误；
- 拓展课程导致 S1/S2 主评估持续退化且回退无效；
- OceanSim 延迟中位数或 p95 超过 20ms；
- 只有一次延迟 PASS，或缺少完整 `evaluation_summary.json`，却要据此宣称稳定 50Hz；
- 最终解析、OceanSim 和可视化使用了不同的 checkpoint SHA-256；
- 需要把 5Hz 或非正式 checkpoint 冒充 50Hz/最终结果。

## 14. 当前下一步

G1–G6 已通过，最终交付模型固定为 C3 综合臂 best checkpoint
`91dea4d4612190786c56f22efed02de5313fec9867da0c558bd6a179399c723e`。解析最终评估
（S1/S2、C1/C2/C3）全部通过；G7 OceanSim 分支结论为 FAIL，G8 阻塞。

已完成的收尾项：

1. G7 OceanSim 已按冻结口径完成 `v7/v8/v9` 三次复测：`1/3` 通过，停止继续调 OceanSim
   并行；`v7` 的 6 条成功轨迹仅作诊断，不写成正式 OceanSim 成功结果。
2. 已完成 G7.3 延迟稳定化迭代：CPU 绑定达到 `2/3`，但第三轮 p95 `20.75ms` 仍失败；
   关闭抗锯齿、性能渲染子集和 16 线程均未提供稳定余量。证据见
   `g7_3_latency_iterations.json/.md`，不把 `2/3` 写成通过。
3. 已生成 `final_evaluation.json`（总状态 `FAIL`）、`final_report.md` 和
   `throughput_summary.json`；解析、课程和 OceanSim 的结论分表记录。
4. 已生成 6 个解析 2D GIF、静态轨迹、contact sheet 和 HTML 相册，标题/元数据明确标注
   `analytic sensor`；OceanSim 轨迹没有混入这组图。
5. G8 保持 **BLOCKED**：OceanSim 仅保留评估用途，不扩大到 16/64/1024 个渲染环境。

最终质量门已复跑：`276 passed`、ruff/format PASS、mypy PASS；不再改变门限，也不把
`v7` 单次成功或 CPU 绑定的 `2/3` 升级为最终成功。若无新的硬件隔离或 OceanSim
渲染架构变化，本轮到此停止。

本轮不执行 `git add`、`git commit` 或 `git push`；是否提交由用户明确决定。

