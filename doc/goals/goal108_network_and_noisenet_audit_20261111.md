# Goal108 子审计：网络结构、历史谱系与 NoisyNet 排除

**日期**：2026-11-11<br>
**状态**：`READ_ONLY_AUDIT_COMPLETE / PPO_PARITY_OK / SAC_FEATURE_LAYOUT_OK / SAC_PATH_INIT_NONPARITY / NOISENET_ABSENT / CHECKPOINT_PARITY_CONFIRMED`<br>
**性质**：Goal108 的测量前只读审计；不改变训练预算、seed、base checkpoint、reward、terminal、success 或 evaluation 协议。

## 1. 问题

用户提出：当前残差是否可能被 NoisyNet 主导。历史 DDPG 线确实出现过
NoisyNet，因此不能只凭当前代码口头判断，必须和旧 checkpoint、旧代码及
Goal89/95/104/107 的逐张量 parity 审计对照。

## 2. 实际结构对照

### 2.1 历史 DDPG 线（Goal58/59）

旧 DDPG 的 actor/critic 使用 `NoisyLinear`，参数包括
`weight_mu/weight_rho/bias_mu/bias_rho`，actor 还支持 output sigma。评价时使用
`sample_noise=False` 的 mean-weight 网络。

Goal59 已经在 8/8 deterministic mean 上检查过饱和与 sigma，结论是
`NOISENET_DOMINANT=false`。NoisyNet 可以影响探索，但不能解释确定性策略的
动作饱和或闭环失败。

这条 DDPG 线不是当前 BC → PPO/SAC 谱系。

### 2.2 Goal86 frozen BC 与 Goal108 PPO

| 项目 | Goal86 frozen BC | Goal108 PPO residual / frozen BC |
|---|---|---|
| `d_z_total` | 64（60 active  4 reserved） | 64（60 active  4 reserved） |
| state encoder | `20→128→128→60` | `20→128→128→60` |
| path encoder | `12→64→64` | residual `8→64→64`  endpoint `4→32→64` |
| current encoder | `3→64→64` | `3→64→60`  4 reserved |
| actor | `131→256→128→3` | `131→256→128→3` |
| critic | `131→256→128→1` | `131→256→128→1` |
| 融合顺序 | `[z_t64, zero-DADE64, command3]` | 同左 |
| NoisyNet | 无 | 无 |
| 随机参数 | 无 | PPO `std=(3,)`，评价 deterministic |

Goal108 PPO initial checkpoint 的实际扫描结果：

```text
actor_encoder.network.0.weight  (128, 20)
actor_encoder.network.2.weight  (128, 128)
actor_encoder.network.4.weight  (60, 128)
path_encoder.0.weight           (64, 8)
path_endpoint_encoder.0.weight  (64, 4)
actor_current_encoder.0.weight  (64, 3)
actor.0.weight                  (256, 131)
actor.2.weight                  (128, 256)
actor.4.weight                  (3, 128)
critic.0.weight                 (256, 131)
std                             (3,)
noise/sigma parameters          none
```

`model_initial.pt` 的 residual output 为零；manifest 记录：

```text
initial_residual_zero_applied = true
initial_residual_abs_max      = 0.0
```

因此 PPO iteration 0 与 B20 完全一致，dev40 为 default `33/40`、strict125
`20/40`。这不是“先验被网络随机初始化破坏”的问题。

### 2.3 Goal108 SAC

实际加载 `B20_SAC_init/initial.zip` 后扫描得到：

```text
features_dim                        131
actor_encoder.network.0.weight      (128, 20)
actor_encoder.network.2.weight      (128, 128)
actor_encoder.network.4.weight      (60, 128)
path_encoder.0.weight               (64, 12)
path_encoder.2.weight               (64, 64)
current_encoder.0.weight            (64, 3)
latent_pi.0.weight                  (256, 131)
latent_pi.2.weight                  (128, 256)
mu.weight                           (3, 128)
log_std.weight                      (3, 128)
qf0.0.weight                        (256, 134)
qf1.0.weight                        (256, 134)
noise/sigma parameters              none
```

SAC 的 `q` 输入是 `131  action3 = 134`。actor 的 `mu` 被零初始化，初始
deterministic action 为零增量，dev40 同样复现 B20 `33/20`。

需要如实记录的差异是：Goal108 SAC 当前配置的 `bc_path_dim=12`，因此 path
编码器是一个 `12→64→64` MLP；PPO residual 分支使用 `8→64→64` 加
`4→32→64` 的拆分。两者输入信息完全相同，但参数化不同。该差异不是
Goal108 `INVALID` 条件（base parity、summary、seed、契约均有效），但在
解释 SAC 与 PPO 的小幅差异时必须保留这个边界。

### 2.4 与 Goal86 P-A SAC 的代码/初始化对照（结果后补充审计）

用户提示应与历史代码和旧实验对照。重新读取 Goal86 P-A SAC 的配置与
`initialization.json` 后，确认这里不只是“12 与 84 的宽度差异”：

| 项目 | Goal86 P-A SAC | Goal108 B20 SAC | Goal108 B20 PPO residual |
|---|---|---|---|
| `bc_path_dim` | 8 | 12 | 8 |
| path endpoint | 4，独立 `4→32→64`，零末层 | 无独立 endpoint 分支 | 4，独立 `4→32→64`，零末层 |
| `bc_checkpoint` | Goal86 residual BC，非空 | `null` | Goal86 residual BC，非空（frozen branch） |
| BC encoder 加载 | state/path encoder 均加载 | 未加载 | frozen BC 分支加载 |
| trainable path encoder | 8→64→64，来自 BC | 12→64→64，随机初始化 | 8→64→64，末层零初始化 |
| trainable state encoder | BC 初始化 | 随机初始化 | 随机初始化 |
| residual 输出头 | `mu` 零初始化 | `mu` 零初始化 | actor 末层零初始化 |

对 Goal108 两个 initial checkpoint 做逐张量比对：

```text
PPO actor_encoder 与 SAC actor_encoder：3/3 层 torch.equal，max_abs_diff=0.0
PPO actor.4.weight 与 SAC mu.weight：均为全零
PPO actor_current_encoder 与 SAC current_encoder：末层均全零
PPO path_encoder.2.weight：全零（path8 输出被门控）
SAC path_encoder.2.weight：非零（path12 从第一步就进入 latent）
```

因此可以排除两件事：

1. 不是 NoisyNet；Goal108 PPO/SAC 都没有 `rho/sigma/noise` 参数；
2. 不是 state encoder 随机种子错位；PPO 与 SAC 的 trainable state encoder
   在当前两个 initial checkpoint 中逐元素完全相同。

但必须保留一个新的可比性边界：Goal108 SAC 没有沿用 Goal86 P-A SAC 的
`path8  endpoint4  BC encoder` 初始化，而是把 12 维 path 整体送入一个随机
初始化的 path encoder。PPO 的 trainable path 分支则因末层零初始化，在第一步
不会把随机 path 表征注入残差。两者输入信息相同，但**初始优化路径不同**：
SAC 的 `mu` 一旦离开零，path 随机特征会立即参与梯度；PPO 要等 path/actor
末层先获得非零权重，path 信息才会进入闭环。

这不会把 Goal108 的冻结 Gate 改写成 `INVALID`：B20 parity、预算、seed 和评价
协议都有效，SAC 的退化幅度也远超 seed 噪声。但它意味着 **SAC 的 6/3 不能
被解释成“纯 SAC 优化器不如 PPO”**，只能解释为“在这个未对齐的 path 参数化/
初始化下，SAC replay 优化把确定性残差推到了饱和”。若未来还要做 SAC 对照，
应另开预注册实验，固定 `bc_path_dim=8  endpoint4`，并明确是否加载 BC encoder。

### 2.5 实际 checkpoint 级复核（远端只读）

为避免只根据配置或文档下结论，重新从远端读取
`B20_PPO_init/model_initial.pt`、`B20_SAC_init/initial.zip` 和
Goal86 P-A BC checkpoint，并做逐张量比对。结果如下：

| 检查 | 结果 |
|---|---|
| Goal86 frozen BC → Goal108 PPO `frozen_bc_*` | 22/22 条目 `torch.equal`，`max_abs_diff=0.0` |
| PPO 与 SAC 的 state encoder | 8/8 条目 `torch.equal`，`max_abs_diff=0.0` |
| PPO path encoder | `8→64→64`，末层全零 |
| PPO endpoint encoder | `4→32→64`，末层全零 |
| PPO current encoder | `3→64→60`，末层全零 |
| PPO actor 输出层 | 全零，初始确定性增量为 0 |
| SAC path encoder | `12→64→64`，末层非零 |
| SAC endpoint encoder | 不存在独立分支 |
| SAC initial `mu` | 全零；`log_std.bias=-3.0` |
| NoisyNet `rho/sigma/noise` 参数 | 0 个 |

注意：Goal86 P-A SAC 的 `pa_sac_bc_head_scale100_diag/final.zip` 是训练后的
checkpoint，其 encoder 已经偏离初始 BC，不能用 final 权重做初始化 parity。
但它的 `initialization.json` 明确记录 `bc_path_dim=8`、`endpoint_dim=4`、
`encoder_loaded=true`、`path_encoder_loaded=true`，与 Goal108 的
`bc_path_dim=12`、`bc_checkpoint=null` 形成直接对照。

因此这次复核再次排除：

```text
noisenet_dominant              = false
frozen_bc_checkpoint_mismatch  = false
ppo_state_encoder_mismatch     = false
ppo_zero_increment_parity      = true
sac_initial_mu_zero            = true
sac_path_init_parity           = false
```

### 2.6 与历史代码的逐路径对照

用户提示“网络结构问题可以和之前的情况或代码比对”。上面的 checkpoint 级
结论再往下追一层，可以看到当前代码里同时存在四条不同的谱系，不能把它们都
叫作“同一个残差网络”：

| 谱系 | 代码/配置事实 | 对当前判断的含义 |
|---|---|---|
| Goal58/59 DDPG | actor/critic 使用 `NoisyLinear`；评价走 `sample_noise=False` 的 mean-weight 网络 | 旧线的噪声参数确实存在于旧 checkpoint；但 Goal59 的 8/8 deterministic saturation 已是 `1.0`，`noise/det` 中位数仅 `0.019`，所以饱和不是 NoisyNet 造成。该线不是当前 BC → PPO/SAC 谱系。 |
| Goal86 P-A SAC | `bc_path_dim=8`、独立 endpoint `4→32→64`；`load_bc_encoder()` 加载 BC 的 state/path/current encoder；`initialize_sac_head_from_bc()` 复制 actor 主干后把 `mu` 置零 | 这是“BC encoder/主干初始化”，但初始动作不是 BC 动作；没有 action-level BC prior。它的 path 分支是 BC 初始化过的，不能和 Goal108 SAC 的随机 path encoder 直接等同。 |
| Goal108 PPO residual | `base_policy` 加载 Goal86 frozen BC；`pc.bc_path_dim=8`、`path_encoder_zero_init=true`；frozen BC 22/22 `torch.equal`；可训练 path8 与 endpoint4 的末层都为零 | 冻结 BC 提供动作 prior；可训练残差分支在第一步输出严格为 0，随机 path 表征不会立刻进入闭环。 |
| Goal108 B20 SAC | `bc_checkpoint=null`、`bc_path_dim=12`；没有调用 `load_bc_encoder()`；只调用 `zero_initialize_sac_residual()`；path encoder 为 `12→64→64` 随机初始化，且没有独立 endpoint 分支；`base_residual_checkpoint` 在环境侧叠加 BC 残差 | BC 动作 prior 仍在环境动作层存在，但 **SAC 的残差特征网络**没有 BC encoder，path 分支从第一步就可接收梯度；这与 PPO 的 zero-gated path 分支不同。 |

对应的代码路径是：

- PPO 可训练 path/endpoint 的末层零初始化：
  `src/auv_lowlevel/batch/pc_policy.py` 的 `path_encoder`、
  `path_endpoint_encoder` 构造段；
- PPO frozen BC 分支：
  `src/auv_lowlevel/batch/pc_policy.py` 的 `frozen_bc_actor_encoder`、
  `frozen_bc_path_encoder`、`freeze_frozen_bc_actor()`；
- SAC 特征提取器的 path/endpoint 构造与 BC encoder 加载：
  `src/auv_rl/sac/pc_features.py` 的 `PCStatePathFeatureExtractor`、
  `load_bc_encoder()`；
- SAC 训练入口只在 `bc_checkpoint` 非空时调用 `load_bc_encoder()`：
  `scripts/train_pc_command_residual_sac.py`；
- SAC 环境侧的固定 BC 残差叠加：
  `src/auv_rl/ddpg/pc_command_residual_env.py` 的
  `base_residual_policy.act_inference()`。

因此要区分两件事：

```text
动作层 BC prior：
  Goal108 PPO  = frozen BC branch
  Goal108 SAC  = base_residual_checkpoint in env
  Goal86 P-A SAC = 无（mu 零初始化，BC 只初始化 encoder/trunk）

残差特征网络初始化：
  Goal108 PPO  = path8 + endpoint4，末层零初始化
  Goal108 SAC  = path12 单一随机 encoder，无 endpoint split
  Goal86 P-A SAC = path8 BC encoder + endpoint4 零初始化
```

这三层差异共同说明：Goal108 的 PPO/SAC 差异不能写成“纯 PPO 优化器
优于 SAC 优化器”。如果未来还要做 SAC 同协议对照，预注册里必须显式固定
`bc_path_dim=8` + endpoint4，并明确是否加载 BC encoder；否则实验仍然会把
“网络初始化/参数化差异”混进算法对照。

## 3. 结论

```text
network_wiring_bug              = false
noisenet_dominant               = false
frozen_bc_checkpoint_mismatch   = false
ppo_zero_increment_parity       = true
sac_zero_increment_parity       = true
sac_encoder_prior_loaded        = false
sac_path_init_parity            = false
```

当前 PPO 1M screen 的 increment L2 ratio 只有 `0.0060–0.0095`，不是“残差很大”。
如果后续出现大残差，它也不可能是 NoisyNet 注入的，因为当前 PPO/SAC 都没有
NoisyNet，正式评价又使用 deterministic mean。需要查的是动作尺度、reward
credit、terminal capture 和 residual 方向，而不是继续往 NoisyNet 上归因。

## 4. 对 Goal108 判读的影响

- PPO 与 SAC 的 B20 基座和零增量 parity 均成立；
- 网络接线问题可以排除；
- SAC 的 path 参数化与 PPO 不完全同构，且没有沿用 Goal86 P-A SAC 的 BC
  encoder 初始化，因此不能把二者差异解释成纯优化器差异；
- 若 SAC 只出现边缘正向信号，formal 前应补一个 `bc_path_dim=8` 的同构对照组；
- 若 SAC 与 PPO 都不动，则优先进入 terminal capture / reward / residual direction，
  不再扩网络或继续扫 residual scale。

## 5. 证据

- `doc/goals/goal89_network_parity_and_gate_scope_audit_20261111.md`
- `doc/goals/goal95_result_20261111.md`
- `doc/goals/goal104_channel_parameterization_audit_20261111.md`
- `doc/goals/goal107_result_20261111.md`
- `doc/goals/goal86_sac_result_20261110.md`
- `artifacts/lowlevel/goal108_b20_prior_optimizer_20261111/B20_PPO_init/model_initial.pt`
- `artifacts/lowlevel/goal108_b20_prior_optimizer_20261111/B20_SAC_init/initial.zip`
- `artifacts/lowlevel/goal86/pa_residual_bc_path12_cmd3_current_20261110/pc_bc_actor.pt`


## 2.7 PPO 训练动力学与 Goal91 历史对照（结果后补充审计）

网络结构本身已经排除，但“为什么 PPO 增量只有 `0.006–0.0095`”仍需和旧 PPO
run 对照。这里使用的都是远端只读产物，不改变 Goal108 的冻结 Gate。

### 2.7.1 Goal108 PPO 的实际训练协议

从 `B20_PPO_s86201/training_manifest.json` 与
`iteration_records.jsonl` 读取到：

| 项目 | 值 |
|---|---|
| `num_envs` | 1000 |
| `num_steps_per_env` | 25 |
| iterations | 40 |
| 总 transitions | 1,000,000 |
| 每个 env 的决策步数 | `25 × 40 = 1000` |
| `num_learning_epochs` | 1 |
| `num_mini_batches` | 100 |
| `clip_param` | 0.02 |
| `entropy_coef` | 1e-4 |
| actor lr | 3e-4（critic warmup 前 5 个 iteration 为 0） |
| schedule | fixed |

关键事实不是“PPO 跑得不够久”，而是 **1M transitions 在这个任务上约等于
每个环境只跑完一个 episode**：

| seed | mean episode length | 有 episode 日志的 iteration | 最后 iteration resets |
|---:|---:|---:|---:|
| 86201 | 988.79 decisions | 4 / 40 | 24 |
| 86202 | 989.88 decisions | 4 / 40 | 24 |
| 86203 | 988.81 decisions | 4 / 40 | 26 |

也就是前 ~36 个 iteration 只看到 truncated rollout，terminal reward 只在最后
约 3–4 个 iteration 才进入训练。与此同时每个 rollout 只做 1 个 learning epoch
和 100 个 minibatch，整轮训练总共只有约 4000 次 minibatch 更新。

### 2.7.2 参数与优化量的直接证据

`B20_PPO_s86201` 的 checkpoint 逐张量范数：

| 参数 | initial | model_10 | model_20 | model_30 | final |
|---|---:|---:|---:|---:|---:|
| `actor.0.weight` | 9.2304 | 9.0081 | 8.8891 | 8.7872 | 8.7212 |
| `actor.2.weight` | 6.5394 | 6.2752 | 6.0740 | 5.8810 | 5.7494 |
| `actor.4.weight`（残差输出头） | 0.0000 | 0.0093 | 0.0087 | 0.0107 | 0.0177 |
| `path_encoder.2.weight`（末层） | 0.0000 | 0.2568 | 0.3157 | 0.3729 | 0.4176 |
| `path_endpoint_encoder.2.weight` | 0.0000 | 0.0692 | 0.1051 | 0.1184 | 0.1329 |

对应的 iteration 记录：

- `mean_noise_std` 全程约 `0.100`，没有学出更小探索方差；
- `approx_kl` 在 `4.6e-6 – 1.2e-4`，`clip_fraction` 在 `0 – 0.164`；
- `surrogate` 在 `8e-8 – 3.9e-4`，`value_function` 在 `2e-4 – 0.07`；
- actor 主干范数在 40 个 iteration 内持续下降，说明可见的更新主要是在收缩
  初始化主干，而不是形成新的闭环修正方向；
- 第 1 个 iteration 记录的 `actor.4.weight update norm=1.006` 不是优化器步，
  而是 `_zero_actor_output` 把随机初始化清零造成的快照差；`model_0.pt` 的
  `actor.4.weight` 实际为 0，可与该解释逐项对上。

### 2.7.3 与 Goal91 PPO 的对照

Goal91 的 6 个 PPO run 使用相同 `1000 envs × 25 steps × 40`：

| 指标 | Goal91 N03/N05 | Goal108 B20 PPO |
|---|---|---:|
| 总 transitions | 1M | 1M |
| 每个 env 决策步 | 1000 | 1000 |
| 有 episode 日志的 iteration | 6 / 40 | 4 / 40 |
| actor 输出层范数 | 0.58–0.62 → 0.39–0.45（下降） | 0 → 0.0177 |
| entropy | `-6.26 → -6.24`（几乎不动） | `-2.65 → -2.64` |
| mean noise std | ~0.03–0.05 | 0.100 |

Goal91 是从随机 actor 训练，Goal108 是从零初始化残差头训练；两者的动作谱系
不同，但训练动力学一致：

```text
每个 env 只覆盖约一个 episode
+ 每个 rollout 只做 1 个 epoch
+ 极小的 advantage / KL / clip 活动
=> actor 主干与输出头都难以形成有效的 terminal-level 修正
```

因此这不是 NoisyNet、不是 frozen BC checkpoint 错配，也不是 PPO/SAC
state encoder 接错；它是当前 PPO 采样/更新协议与 mission-length episode
不匹配的结果。该判断仍是相关性证据，不能在 Goal117 之前直接改算法或改 reward。

### 2.7.4 对后续实验的约束

- 任何新的 PPO 训练型 Goal 必须预注册：`num_envs`、`num_steps_per_env`、
  `num_learning_epochs`、`num_mini_batches`、总 transitions、每个 env 覆盖的
  episode 数，以及至少 1 个完整 episode 的训练比例；
- 必须同时记录 learned-increment norm、KL、clip fraction、entropy/std、
  advantage 正负分布和 terminal reward credit；
- 如果目标是 terminal closure，不能让 1M transitions 只覆盖一次 989 步
  episode；否则测到的是“单次轨迹上的截断 PPO”，不是可学习的闭环残差；
- 在 Goal117 证明“方向正确但闭环不动且有探索/样本效率证据”之前，不启动
  SAC/REDQ/DroQ 对照。
