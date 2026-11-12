# Goal86 Step 1：SAC 的 `z_t` 适配与工程 smoke

**日期**：2026-11-10<br>
**状态**：`SAC_ADAPTER_SMOKE_PASS / P-A_BC_BACKBONE_INIT_ONLY / P-A_BC_PRIOR_CONTRACT_MISMATCH / T1_ZERO_INIT_OK / NOT_A_RESULT`<br>
**上游**：`doc/goals/goal86_bc_prior_traditional_residual_20261110.md`<br>
**产物**：`src/auv_rl/sac/`、`scripts/train_pc_command_residual_sac.py`、
`scripts/evaluate_pc_sac.py`、`configs/lowlevel/goal86/`、远端 `artifacts/goal86/`

本步骤只实现并验证 SAC 的项目原生 actor 架构、D3 BC 初始化、checkpoint 往返和确定性
评价入口。远端 100k transitions 是工程 smoke，不是 1M screen，也不能写成 SAC 或 BC
先验的算法效果。未打开 held-out `100–149`。

## 1. 实现结论

| 项目 | 状态 | 说明 |
|---|---|---|
| PC/SAC VecEnv | `PASS` | 固定长度 history stack；done 时 `terminal_observation` 形状正确 |
| `z_t` actor feature extractor | `PASS` | 复用 `PCReservedSlotEncoder`、path/current encoder 和保留槽 |
| D3 BC encoder 初始化 | `PASS` | state/path encoder 严格加载；SHA 校验通过 |
| D3 BC actor 主干初始化 | `PASS` | `[256,128]` 主干和输出层映射通过 |
| zero-residual 初始化 | `PASS` | `mu` 与 command 新列零初始化，确定性动作严格为零 |
| SB3 SAC checkpoint 往返 | `PASS` | `final.zip` 可重新加载，deterministic predict 有限 |
| SAC deterministic eval | `PASS` | 4-seed 闭环、terminal 记录和指标落盘完成 |
| 1M screen | `NOT_STARTED` | 本步骤不训练正式预算 |

## 2. 项目原生 `z_t` 结构

SAC 不再使用普通 `FlattenExtractor + MLP`。actor 和 critic 都使用项目原生提取器：

```text
actor_state(12) -> PCReservedSlotEncoder(12 -> 64, 输入保留 8, 输出保留 4) -> z_t
path prefix    -> path_encoder                                                -> z_t
path tail      -> endpoint_encoder (零初始化)                                  -> z_t
actor_current  -> current_encoder (零初始化)                                   -> z_t

features = [z_t(64), zero_dade(64), command(3)] = 131
latent_pi = 256 -> 128
mu / log_std = 3 维 command residual
```

关键设计：

- 只编码 history stack 的**最后一帧**；10 帧 history 仍保留在 observation-space 中，
  但本 Goal 不用 DADE 做时序融合。这样做是为了保持 actor 可部署，并直接复用 D3 BC。
- `[z_t, zero_dade, command]` 的顺序不是随意的：前 128 列与 D3 BC actor 的
  `[z_t, DADE]` 输入完全对齐，后 3 列才是 SAC 新加的 command。
- DADE 在本 Goal 不进入主模型，因此 `zero_dade` 恒为零；这不是声称“DADE 无效”。
- critic 使用同样的 `z_t` 提取器但独立初始化；actor 与 critic 不共享参数。

## 3. path8 与 path12 的处理

Goal26 D3 BC 的 `path_dim=8`；Goal85 候选配置的 `path_dim=12`。两套列定义不是任意
宽度：

```text
path12 = path8 的 8 个 canonical features + 4 个 endpoint-context features
```

因此本步骤不静默截断，而是显式拆分：

| arm | `path_dim` | `bc_path_dim` | endpoint | 初始化 |
|---|---:|---:|---:|---|
| `T1-SAC` | 12 | 12 | 0 | path encoder 随机；全 `z_t` 从零残差开始 |
| `P-A-SAC` | 12 | 8 | 4 | 前 8 列加载 D3 path encoder；后 4 列零初始化 |

`P-A-SAC` 的 D3 checkpoint：

```text
checkpoints/goal26/d3/pc_bc_actor.pt
SHA256 = 7430b26308b701565a52a92ed31c382ccca9cf1721d183ab856e3fafd9f4ab73
actor_command_dim = 0
actor_output_dim = 3
path_dim = 8
actor_hidden_dims = [256, 128]
```

endpoint encoder 的最后线性层零初始化，因此初始 `z_t` 恰好等于 D3 的 state+path 表征；
endpoint context 的效果由 RL 从零开始学习。配置测试会拒绝把 D3 path8 encoder 直接接到
path12 输入上。

## 4. BC 初始化的准确语义

`initialize_sac_head_from_bc` 做的是**残差 actor 主干初始化**，不是把 D3 当作 command
teacher：

1. `actor_encoder` 和 `path_encoder` 严格加载 D3 参数；
2. D3 actor 的 `actor.0/actor.2` 映射到 SAC 的 `latent_pi.0/latent_pi.2`；
3. SAC 第一层新增的 3 个 command 输入列置零；
4. 最终 `mu` 置零，`log_std` 保持可探索；
5. 因而初始 deterministic residual 严格为 0，actor 从传统控制器附近开始。

D3 BC 的 `actor_command_dim=0`，所以它只能初始化 actuator 残差主干；不能把它解释成
`[psi_d, theta_d, u_d]` command teacher。P-C/U2 仍为 `NOT_AVAILABLE`。

## 5. 代码与配置

新增或修改：

| 文件 | 作用 |
|---|---|
| `src/auv_rl/sac/pc_vec_env.py` | PC batch env 到 SB3 VecEnv 的固定 history 适配 |
| `src/auv_rl/sac/pc_features.py` | `z_t` 提取器、BC encoder/actor 初始化和零残差初始化 |
| `scripts/train_pc_command_residual_sac.py` | SAC 训练入口、UTD、checkpoint 和初始化清单 |
| `scripts/evaluate_pc_sac.py` | SB3 SAC deterministic 闭环评价 |
| `configs/lowlevel/goal86/pc_goal86_t1_sac_default_20261110.json` | 传统控制器 + SAC command residual，全 path12，无 BC |
| `configs/lowlevel/goal86/pc_goal86_pa_sac_default_20261110.json` | D3 先验 + SAC command residual，path8 + endpoint4 |
| `tests/test_pc_sac_features.py` | 特征布局、末帧选择、BC 加载和零残差测试 |
| `tests/test_pc_goal86_sac_config.py` | 配置与 D3 checkpoint 契约测试 |

## 6. 测试

```bash
cd /home/bubble/水下机器人/settled_局部规划器_反应式
PYTHONPATH=src ../.venv/bin/python -m pytest -q \
  tests/test_pc_sac_adapter.py \
  tests/test_pc_sac_features.py \
  tests/test_pc_goal86_sac_config.py
```

结果：

```text
10 passed
```

覆盖项：

- history stack 的 done reset 和 `terminal_observation` 形状；
- 只使用最后一帧，前 9 帧改变不影响 `z_t`；
- `features = [z_t, zero_dade, command]` 的宽度和顺序；
- D3 state/path encoder 严格加载，endpoint/current 保持零初始化；
- path12 直接加载 path8 BC 时明确报错；
- BC 初始化后 `mu=0`、command 新列为 0、deterministic action 为零；
- T1 与 P-A 配置的 `path_dim`、`bc_path_dim`、`net_arch` 和 D3 SHA 契约。

## 7. 本地 smoke

本地 CPU 的 tiny smoke：

| arm | transitions | 结果 |
|---|---:|---|
| `T1-SAC` | 2048 | 完成；checkpoint 可加载；deterministic predict 有限 |
| `P-A-SAC` | 2048 | 完成；D3 SHA 和 path8/endpoint4 初始化正确 |

本地 smoke 只证明适配器工作，不进入效果分析。

## 8. 远端 100k 工程 smoke

服务器：8 × RTX 4090；每张卡运行一个进程；环境 `unbiased_noise_v1`、
`position_source=eta`、零传感器延迟、`path_dim=12`。

| arm | num envs | transitions | wall time | throughput | episodes | success episodes |
|---|---:|---:|---:|---:|---:|---:|
| `T1-SAC` | 64 | 100032 | 365.21 s | 273.90 /s | 64 | 42 |
| `P-A-SAC` | 64 | 100032 | 414.24 s | 241.48 /s | 64 | 41 |

checkpoint SHA256：

```text
T1-SAC: 031c96e507ca011817da91f55ac0f2e0d7431098475ed9076b276ad3a3d00dcb
P-A-SAC: 79b24d001866550233219a4200ddb9690793945c7ecb5fb77dbb7d710f680212
```

说明：这两个 run 没有加 `--smoke`，因此 manifest 中 `smoke=false`；它们是 100k
engineering smoke，不是 1M screen。新增的 `--engineering-smoke` 只写 `run_kind`，不会
改变预算或采样规则。

## 9. 确定性评价 smoke

远端 CPU、float64、dev seed `20–23`：

| 模式 | success | path RMSE | gate-active residual L2 | residual max | policy action max |
|---|---:|---:|---:|---:|---:|
| `T1-SAC` policy | 4/4 | 1.4674 m | 6.06% | 0.0899 | 0.168 |
| `P-A-SAC` policy | 3/4 | 1.4733 m | 4.39% | 0.1394 | 0.821 |
| zero-residual | 4/4 | 1.4680 m | 0% | 0 | 0 |

这些数字只证明评价入口、terminal capture 和 residual 记录可用。4 个 seed 不能用于
判断 BC 先验、SAC 或 path/residual 接口的优劣；P-A 的 `action max=0.821` 也只作为
工程诊断，不能写成饱和结论（阈值是 0.9），更不能据此换算法。

## 10. 当前裁决与下一步

裁决：

```text
SAC_PC_ADAPTER = PASS
D3_Z_T_BACKBONE_INIT = PASS
T1_ZERO_RESIDUAL = PASS
ZERO_RESIDUAL_PARITY = PASS (PPO 112 metrics, SAC 16 metrics, max spread 0.0)
READY_FOR_T1_T2_SCREEN = true
P-A_RESIDUAL_BC = BLOCKED
READY_FOR_P-A_SCREEN = false
```

`P-A` 的阻塞不是算法效果问题，而是实现语义不一致：当前代码只做 D3 表征先验和零残差
初始化，没有拟合 `D3_actuator - ALOS_actuator` 标签。只读标签审计显示，在
`residual_scale=0.25` 下 rudder/stern 的裁剪比例为 `88.45%/73.86%`；只读 scale
网格显示 `0.50` 仍为 `75.66%/52.95%`，`1.00` 才降到 `0.77%/7.47%`。这支持
H86-PA-1，但不能静默把正式协议改成 `1.00`：T2/P-A 必须同 scale，并记录协议修订。
证据与两个假设见 `research/blockers/2026-11-10-goal86-pa-residual-contract.md`。

下一步顺序：

1. `T1/T2 × PPO/SAC` 使用统一 1M environment transitions 先跑 screen；训练入口继续
   使用项目原生 `z_t`，不退回 flatten+MLP；
2. 对 P-A 做 `0.25/0.50/1.00` 的 residual scale 只读标签网格，判断 H86-PA-1；
3. 按真实 `D3-ALOS` 标签实现 P-A 残差 BC，并保持与 T2 相同的 residual scale；
4. BC-only 检查和 zero-residual parity 通过后，才进入 P-A 的 1M screen；
5. held-out `100–149` 保持关闭，不因 P-A 阻塞更换算法或放宽 Gate。

## 11. 复现命令

本地测试：

```bash
cd /home/bubble/水下机器人/settled_局部规划器_反应式
PYTHONPATH=src ../.venv/bin/python -m pytest -q \
  tests/test_pc_sac_adapter.py \
  tests/test_pc_sac_features.py \
  tests/test_pc_goal86_sac_config.py
```

本地 tiny smoke：

```bash
PYTHONPATH=src ../.venv/bin/python scripts/train_pc_command_residual_sac.py \
  --config configs/lowlevel/goal86/pc_goal86_pa_sac_default_20261110.json \
  --output artifacts/goal86/sac_zt_pa_smoke_local_20261110 \
  --smoke --device cpu --overwrite
```

远端 100k engineering smoke：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:$SB3 \
python scripts/train_pc_command_residual_sac.py \
  --config configs/lowlevel/goal86/pc_goal86_t1_sac_default_20261110.json \
  --output artifacts/goal86/sac_zt_t1_smoke_20261110 \
  --num-envs 64 --total-env-transitions 100000 \
  --learning-starts 1000 --batch-size 256 --buffer-size 100000 \
  --device cuda --engineering-smoke --overwrite
```

## 12. Git

本步骤不自动 commit/push。提交时只 add 本文档、Goal86 主文档、SAC adapter/trainer/eval、
Goal86 configs 和对应测试；不 add `artifacts/`、checkpoint、日志或缓存。

```bash
python ../scripts/check_research_ledger.py \
  --goal Goal86 \
  --goal-doc doc/goals/goal86_bc_prior_traditional_residual_20261110.md \
  --strict
git diff --check
git diff --cached --check
```

