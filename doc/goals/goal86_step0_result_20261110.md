# Goal86 Step 0 结果：BC 先验接口与传统控制器残差后端审计

**日期**：2026-11-10<br>
**状态**：`STEP0_PASS / P-A_READY / P-C_U2_NOT_AVAILABLE / SAC_PC_ADAPTER_MISSING`<br>
**预注册**：`doc/goals/goal86_bc_prior_traditional_residual_20261110.md`<br>
**产物**：`artifacts/goal86/bc_interface_audit.json`、`artifacts/goal86/rl_backend_audit.json`<br>
**远端产物**：`/data/yhx/underwater-robotics/src/plan_p1/settled_局部规划器_反应式/artifacts/goal86/`

## 1. 结论

Step 0 只做只读审计，没有训练、没有写 checkpoint、没有打开 held-out。结论如下：

| 问题 | 结论 |
|---|---|
| Goal85 契约 | 4 个配置全部是 `unbiased_noise_v1` + `position_source=eta`，相关测试通过 |
| D3 BC 动作空间 | `actor_command_dim=0`、`actor_output_dim=3`，输出物理 `[rudder, stern, rpm]` |
| D3 能否初始化 PPO actor | 可以，`load_pc_bc_actor_prior` 探针通过，路径编码器零输出 |
| D3 能否初始化 command residual | 不可以，D3 没有 command 输出列，只能作为 actuator 先验 |
| command teacher | 不存在有效 teacher；现有 teacher 只产出 actuator 标签 |
| PPO 后端 | 入口、actor warmstart、command/actuator residual 配置均可用 |
| SAC 后端 | 上游 SB3 2.9.2 存在且可导入，但没有 PC residual adapter/trainer/eval |
| 算力 | 远端 8×RTX 4090、384 CPU、503 GB RAM；磁盘剩余约 `464 GiB` |

因此 Step 0 判 `PASS`，但只允许启动 `P-A-*` 与 `T1/T2/U1-*` 的 PPO 分支；
`P-C/U2` 标 `NOT_AVAILABLE`，所有 SAC arm 标 `BLOCKED_SAC_PC_ADAPTER_MISSING`。

## 2. Goal85 契约

| 配置 | SHA256 | sensor profile | termination |
|---|---|---|---|
| `pc_goal85_unbiased_default_baseline_20261110.json` | `a8a27775…ae1b` | `unbiased_noise_v1` | `eta` |
| `pc_goal85_unbiased_default_candidate_20261110.json` | `da2c4215…4a82` | `unbiased_noise_v1` | `eta` |
| `pc_goal85_unbiased_strict125_baseline_20261110.json` | `d80ce032…6c79` | `unbiased_noise_v1` | `eta` |
| `pc_goal85_unbiased_strict125_candidate_20261110.json` | `e4b9de87…a962` | `unbiased_noise_v1` | `eta` |

测试：

```text
tests/test_pc_goal85_unbiased_contract.py
tests/test_pc_terminal_proximity.py
tests/test_lowlevel_pc_command_residual.py
tests/test_lowlevel_pc_residual_pretrain.py
tests/test_lowlevel_config.py
72 passed
```

## 3. BC 接口

远端审计了 `12` 个 BC checkpoint。Goal26 D3 的关键字段：

| 字段 | 值 |
|---|---|
| path | `checkpoints/goal26/d3/pc_bc_actor.pt` |
| SHA256 | `7430b26308b701565a52a92ed31c382ccca9cf1721d183ab856e3fafd9f4ab73` |
| `policy_kind` | `pc_bc_actor` |
| `observation_schema` | `pc_v1` |
| `action_envelope` | `pc20` |
| `action_head` | `continuous` |
| `actor_command_dim` | `0` |
| `path_dim` | `8` |
| `actor_output_dim` | `3` |
| 输出语义 | 物理执行器 `[rudder, stern, rpm]` |
| PPO actor warmstart 探针 | `PASS` |
| actor DADE 输出 | 零初始化 |
| path encoder 零输出 | `PASS` |

判定：

- `P-A` 可以用 D3 作为 actuator 残差初始化；actor 主干、state encoder、path encoder
  和 normalization 均可无损加载。
- `P-C` 和 `U2` 不能用 D3 初始化。`actor_command_dim=0` 表示 D3 没有
  `[psi_d, theta_d, u_d]` 输出列；把物理 rudder/stern/rpm 当成 command residual
  标签是语义错误。
- 远端另有 `actor_command_dim=3` 的旧 BC checkpoint，它们只是“输入含 command”，
  输出仍是物理执行器，不是 command teacher。

## 4. Command teacher

仓库中存在的 teacher 全是执行器层：

| 来源 | 输出 | 判定 |
|---|---|---|
| `src/auv_lowlevel/batch/pc_teacher.py::BatchedPCTeacher` | 特权 PD/PI 的 `[rudder, stern, rpm]` | actuator teacher |
| `src/auv_lowlevel/env.py::teacher_command` | `Remus100ActuatorCommand` | actuator teacher |

没有找到满足以下条件的 command teacher：

- 输出 `[psi_d, theta_d, u_d]`；
- 只用可部署观测；
- 闭环表现优于 ALOS；
- 有可审计的标签数据和 checkpoint。

因此 `P-C` 与 `U2` 标 `BC_COMMAND_TEACHER_UNAVAILABLE`，不启动对应训练。
ALOS 自己不能当 teacher，因为标签恒为零，只会制造 `BC_COMMAND_LABEL_COLLAPSED`。

## 5. PPO 与 SAC

### 5.1 PPO

`scripts/train_batch_lowlevel_ppo.py` 满足 Step 0 要求：

- 支持 `--warmstart-actor` 和 `--warmstart-residual`；
- 调用 `load_pc_bc_actor_prior`；
- 存在 command residual 环境与配置；
- 存在 actuator residual 配置；
- `scripts/evaluate_pc_ppo.py` 可做 deterministic 评价。

状态：`READY`。

### 5.2 SAC

远端 `stable-baselines3 2.9.2` 位于
`/data/yhx/underwater-robotics/src/stable-baselines3`，SAC 源码与类均可导入。但项目内：

- 没有 SAC 源文件；
- 没有 SAC PC residual trainer；
- 没有 SAC deterministic eval；
- 没有 SAC checkpoint 契约；
- 没有 SAC 的 BC actor 初始化路径。

最接近的参考是现有 PC command-residual DDPG：

```text
src/auv_rl/ddpg/pc_command_residual_env.py
src/auv_rl/ddpg/pc_vector_trainer.py
scripts/train_pc_command_residual_offpolicy.py
```

它们可以复用 observation/action/replay 结构，但不能把 DDPG 结果写成 SAC 结果。
状态：`SAC_PC_ADAPTER_MISSING`，`ready_for_sac_screen=false`。

## 6. 远端算力与 checkpoint

远端环境：

| 项目 | 值 |
|---|---|
| GPU | 8 × RTX 4090，24564 MiB/卡 |
| torch | `2.7.0+cu128`，`torch.cuda.device_count()=8` |
| 磁盘 | 3.5 TiB 总量，剩余约 `463.59 GiB` |
| 远端 Python | `/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python` |

Goal59 frozen checkpoints 的 SHA256（用于后续 zero-residual 与配对评价，不作为 PPO
warmstart）：

| checkpoint | SHA256 |
|---|---|
| `formal_s240/final.pt` | `f73aeda9b9fb199a38a233e4e72c746e68737589bfb9f9c38b8c791428953ea1` |
| `formal_s752/final.pt` | `edb08156dae5b996ca9b7d70e286ef78fbd4ba08ebd41115308c423d37b34de2` |
| `formal_s880/final.pt` | `a8d5c98b7880d9cb324cba7e704bf2fde4224172aee34e5f62781ab3c560c932` |
| `formal_s1008/final.pt` | `e6ef364119b2b401d06e6788369431d6314aa07000f62af43c9a23e97584b3a1` |

这些文件是 off-policy actor/critic/target_actor bundle，不是 PPO checkpoint；
Goal86 的 PPO arm 仍从零残差或 BC 初始化开始。

## 7. Arm 裁决

| arm | 状态 | 原因 |
|---|---|---|
| `P-A-PPO` | `READY` | D3 actuator 先验可加载 |
| `P-C-PPO` | `NOT_AVAILABLE` | 无有效 command teacher |
| `U2-PPO` | `NOT_AVAILABLE` | 无有效 path command teacher |
| `T1-PPO` | `READY` | command residual 环境与配置存在 |
| `T2-PPO` | `READY` | actuator residual 配置存在 |
| `U1-PPO` | `READY` | path-following + command residual 可配置 |
| `*-SAC` | `BLOCKED` | PC SAC adapter/trainer/eval 缺失 |

## 8. 下一步

Step 1 只做实现，不训练：

1. 实现 PC/SAC adapter、trainer、deterministic eval 和 checkpoint 往返；
2. 为 `P-A` 生成 D3 teacher 到 ALOS base 的 actuator residual 标签；
3. 生成 Goal86 专用 PPO 配置，不覆盖 Goal27/58/59 历史配置；
4. 对 PPO 与 SAC 分别做 zero-residual parity 和短 smoke；
5. 训练前先在远端做 `num_envs × batch` 单卡吞吐基准，再按 seed 分配 8 张 GPU。

## 9. 复现命令

本地：

```bash
cd /home/bubble/水下机器人/settled_局部规划器_反应式
../.venv/bin/python scripts/audit_goal86_bc_interfaces.py \
  --output artifacts/goal86/bc_interface_audit.json
../.venv/bin/python scripts/audit_goal86_rl_backends.py \
  --output artifacts/goal86/rl_backend_audit.json
```

远端：

```bash
sshpass -p 'Bq9068w1' ssh -o StrictHostKeyChecking=no -p 43127 yhx@152.136.183.51
cd /data/yhx/underwater-robotics/src/plan_p1/settled_局部规划器_反应式
/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python \
  scripts/audit_goal86_bc_interfaces.py \
  --output artifacts/goal86/bc_interface_audit.json
```

本步骤没有训练、没有新增 rollout、没有打开 held-out。
