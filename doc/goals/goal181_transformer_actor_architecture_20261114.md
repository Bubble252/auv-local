# Goal181：Actor 架构与容量配对实验（MLP / Transformer）

**日期**：2026-11-14<br>
**状态**：`DRAFT_FOR_APPROVAL / BLOCKED_ON_GOAL182_PROFILE / NO_TRAINING_STARTED`<br>
**仓库**：`settled_局部规划器_反应式`<br>
**控制契约组**：`G-PATH-CONDITIONED-POLICY` 与 `G-ALOS-BODY-RL`，两组都做<br>
**架构轴**：`AX-ACTOR-ARCH`（不是第三个控制契约组）

```yaml
goal_id: Goal181
repo: settled_局部规划器_反应式
status: BLOCKED_ON_GOAL182_PROFILE
phase: P7_ACTOR_ARCHITECTURE
created: 2026-11-14
updated: 2026-11-16
parents:
  - Goal175
  - Goal177
  - Goal179
  - Goal180
  - META-G06-PROGRAMS
result_doc: null
matrix: null
legacy: false
```

> **2026-11-16 R8-A 结论**：本 Goal 只保留 actor 架构轴，不再维护独立的 8D path、
> reward 或 16-iteration 预算。formal 必须直接消费 Goal182 的 5D actor、reward、PPO
> profile 和 600–2600 iteration 预算；Goal182 profile 锁定前不启动 formal。旧协议
> 结果只作历史参考，不能与新栈结果直接比较。

## 0. 2026-11-14 架构决策

Transformer 主架构改为 **M 档**，S 档降为可选消融，不再是主结论 arm。

```text
primary actor architecture: A3M_TA_HISTORY_M
primary MLP control:        A1M_MLP_TOKEN（同 token、mean-pool、M 容量）
secondary fused arm:        A2M_TA_FUSED_M
secondary fused control:    A0M_MLP_512x256x128
optional ablation:          A2_TA_FUSED_S / A3_TA_HISTORY_S
```

- `A3M_TA_HISTORY_M` 是主架构，用短窗口 state token、DADE off，检验“M 容量 Transformer
  actor 是否优于同容量 MLP”，同时避免把 DADE 重新变成主线。
- `A1M_MLP_TOKEN` 是 A3M 的主 MLP control：使用与 A3M 完全相同的 token 和 mean-pool，
  hidden dims 由 Step 0 校准到 A3M 参数量的 `±20%`，critic 使用 `[512,256,128]`。
  它排除 tokenization 混因。
- `A2M_TA_FUSED_M` 是 fused-token 对照，只检验“少量 fused token 上的 actor 映射”，
  不作为主结论；它的对照是 `A0M_MLP_512x256x128`。
- `A0M_MLP_512x256x128` 使用 fused `z_t + command + c_hat`：actor 为 `512x256x128`，
  critic 也为 `[512,256,128]`，只作为 A2M 的 fused-token control。
- S 档保留为消融；资源不足时可以不做，不能用 S 的结果单独支撑最终架构结论。

M 档 Transformer 起始配置锁定为：

```text
d_model=112, 2 layers, 4 heads, FFN=224,
dropout=0, pre-LN, GELU, learned positional embedding
```

M 档 actor 与 critic 必须同容量：

```text
A0M / A1M / A2M / A3M / A4 / A5 / A6
  actor:  按各自 token 契约，M 容量
  critic: 512x256x128
```

S 档 actor/critic 才使用 `256x128x64`。不要再把 M arm 的 critic 固定在
`[256,128,64]`。

Step 0 只允许根据实际参数统计在 `d_model=104/112`、`FFN=208/224` 内校准一次；校准后
`A3M` 的 actor 参数量相对 `A1M_MLP_TOKEN` 必须落在 `±20%`，`A2M` 相对
`A0M_MLP_512x256x128` 同规则，否则判 `CAPACITY_MISMATCH`。

## Research Contract

```yaml
autoresearch:
  required: true
  triggers:
    - gate_fail_twice
    - no_metric_movement_after_3_fixes
    - heldout_collapse
    - seed_noise_ambiguous
  ledger: research/LEDGER.jsonl
  blockers_dir: research/blockers/
  reading_dir: research/paper_reading/
  max_outer_loops: 3
  check_command: python ../scripts/check_research_ledger.py --goal Goal181 --goal-doc doc/goals/goal181_transformer_actor_architecture_20261114.md --strict
```

- 本 Goal 是探索型 Goal；Step 0 锁定决策前不训练、不创建 formal matrix、不写收益结论。
- 当前两个控制契约组已经固定为 `G-ALOS-BODY-RL` 和 `G-PATH-CONDITIONED-POLICY`。本 Goal 只增加
  `MLP actor vs Transformer actor` 的正交架构轴，并把 MLP 容量和 Transformer 容量作为
  同一轴下的配对子因子，不把网络结构混成第三个控制契约。
- 本 Goal 覆盖 `G-PATH-CONDITIONED-POLICY` 与 `G-ALOS-BODY-RL` 两个控制组，两组都要完成
  paired comparison。执行顺序只是资源调度，不作为科学变量；两组各自矩阵、各自 seed、
  各自控制契约，结果不得 pooled。
- Goal179 的 head/credit 诊断，以及两个控制组各自的 MLP 基线，是本 Goal 的前置门。
  未解决的动作幅度、BC/PPO 抵消或 reward credit 问题，不能用 Transformer actor 掩盖。
- Gate 记录前不得改 reward、termination、success、goal radius、控制契约、residual authority、
  checkpoint 选择规则、seed 或 held-out。

## 1. 背景与事实

### 1.1 当前代码里“Transformer”到底在哪里

现有 `dual_horizon_transformer` 不是 actor 替换：

- `src/auv_lowlevel/batch/encoded_policy.py:204`：`DualHorizonEncoder` 用
  `nn.TransformerEncoder` 编码短/长历史窗口；
- `src/auv_lowlevel/batch/encoded_policy.py:374`：编码后的 actor 仍是 `_mlp(...)`；
- `src/auv_lowlevel/batch/pc_policy.py:809`：PC actor 的输入由 `z_t + command + DADE feature`
  拼接；
- `src/auv_lowlevel/batch/pc_policy.py:811`：当前 PC actor 仍是 `_mlp(...)`；
- `src/auv_lowlevel/batch/pc_policy.py:1080`：`_actor_residual_mean()` 调用 `self.actor(features)`，
  没有 actor 级别的 attention 或 token 序列。

因此：

```text
encoder.kind = dual_horizon_transformer
    != actor 已替换为 Transformer
```

只改 `encoder.kind`、训练配置名或 metadata 名称不构成本 Goal 的完成条件。必须新增可审计的
actor kind、token 契约、checkpoint 兼容和参数/延迟记录。

### 1.2 历史证据与前置条件

| 证据 | 当前结论 | 对本 Goal 的含义 |
|---|---|---|
| Goal175 | MLP 路线矩阵 `VERTICAL_GUARD_FAIL`，当时停止 Transformer actor 替换 | 基座没有先通过时，架构替换不可解释 |
| Goal177 | head residual 幅度远低于 `0.20`，Stage B 未启动 | 先确认 residual authority 和动作幅度不是零 |
| Goal179 | head-only 剂量、BC/PPO 抵消和 credit 诊断仍待完成 | 本 Goal 不重复其诊断；未完成则 `BLOCKED_ON_HEAD_CREDIT` |
| Goal180 | 路径条件组的 canonical reward/纵向迁移矩阵仍是草案 | 本 Goal 复用其锁定后的 MLP 基线和 reward，不另造基座 |
| Goal9/19/95 | 直接 Transformer actor 曾作为条件实验提出，但没有完成配对证据 | 本 Goal 只把历史提案转成可证伪实验，不宣称已实现 |

### 1.3 为什么现在值得做

若 Goal180 的 latent probe 显示纵向/俯仰信息存在，但 MLP actor 的小扰动 Jacobian 很弱，
则“actor 没有把已有表征转成动作”是一个可检验假设。Transformer actor 只能用来检验这个
假设；如果 MLP 在相同条件下已经能产生有效动作，架构替换不是优先项。

## 2. 假设与证伪

### H181-A：actor 架构是当前动作响应瓶颈

- 机制：MLP 对 `[z_t, command, c_hat]` 的局部映射无法利用路径/历史 token 间的交互；
  Transformer actor 在相同输入、相同 authority 下应产生更大且方向更合理的残差。
- 成立预期：M 主臂 `A3M_TA_HISTORY_M` 相对同 token 的 `A1M_MLP_TOKEN`，applied residual Jacobian p90
  提升 `>=2x`，且 current10 的 cross 或 vertical route p90 改善 `>=5%`、绝对改善 `>=0.02 m`。
- 失败预期：Jacobian 和路线指标都不移动，或只在最后一个 keypoint 改善。
- 最便宜证伪：固定 MLP checkpoint，对 `A1M` 与蒸馏后的 `A3M` 做 eval-only Jacobian；
  不训练 PPO。

### H181-B：收益来自 attention，而不是 tokenization 或初始化

- 机制：如果只是把输入拆成 token，参数匹配的 token-MLP 也应获得相同收益；如果收益只在
  蒸馏初始化出现，则不能归因于架构。
- 成立预期：`A2*` / `A3*` 相对 `A1_MLP_TOKEN` 的 paired 改善仍达到
  H181-A 阈值，并且在 random 与 distill 两个初始化 arm 中至少一个成立、另一个不出现反向塌陷。
- 失败预期：token-MLP 与 Transformer 无差异，或只有蒸馏 arm 有效。
- 最便宜证伪：同一批 dev rollout 上比较 token-MLP、随机初始化 Transformer、蒸馏初始化
  Transformer 的 action MSE、Jacobian 和参数预算。

### H181-C：历史 token 是收益来源

- 机制：当前 token 不足以消除纵向滞后；历史窗口中的 `w/q/vertical_error` 轨迹提供了
  预测所需信息。
- 成立预期：`A3M_TA_HISTORY_M`（actor 自己消费短窗口 state 历史、DADE off）优于
  `A4_TA_CURRENT`，且当前 token 不足以解释收益。
- 失败预期：current-only Transformer 与 history Transformer 无差异；此时不应把收益写成
  “历史建模”。
- 最便宜证伪：固定参数预算，比较 current-only 与 history-token 两种 token 契约。

### H181-D：部署代价可接受

- 机制：Transformer 的序列长度、参数和推理延迟必须满足闭环控制预算，否则离线收益不可用。
- 成立预期：actor 参数不超过 MLP 的 `2x`，actor forward p95 不超过 MLP 的 `2x`，且不出现
  因历史缓存导致的观测/重置语义变化。
- 失败预期：参数或延迟超预算，或需要改变环境 history/reset 契约才能运行。
- 最便宜证伪：Step 0 在目标 CPU/GPU 上做 microbenchmark；不训练。

## 3. 实验组归属与固定契约

### 3.1 归属

```text
控制契约组（不变）:
  G-ALOS-BODY-RL
  G-PATH-CONDITIONED-POLICY

架构轴（本 Goal）:
  AX-ACTOR-ARCH = MLP / token-MLP / Transformer actor
```

两个控制组都在本 Goal 范围内，执行顺序可并行或串行；顺序本身不作为科学结论。两组各自
建立配对矩阵、使用自己的 seed 池和评价契约，不把 cell、seed 或 success 合并。若后续要把
架构轴升级成第三个 program group，必须先修改
`总协调/configs/programs/auv_control_programs.json` 与 schema，不能只改标题。

### 3.2 固定控制契约

| 项目 | `G-PATH-CONDITIONED-POLICY` | `G-ALOS-BODY-RL` 配对实验 |
|---|---|---|
| 制导来源 | 路径条件策略内部隐式学习 | 外部 ALOS |
| path | `path_dim=8`，`actor_command_dim=0` | 按该组已锁定契约 |
| actor 输出 | 3D 归一化执行器残差，叠加冻结 BC | 同左 |
| 外部 ALOS | 不进入 canonical runtime | 是该组 canonical 输入 |
| reward | 使用 Goal180 锁定后的同一 reward | 使用该组已锁定 reward |
| curriculum | `enabled=true`，固定 `stage_mix`，逐 episode provenance | 同左 |
| held-out | `100–149` 关闭 | 同左 |

### 3.3 本 Goal 不改变

- 不改变 success / timeout / safety / termination / goal radius；
- 不改变 BC checkpoint、residual scale、动作通道 mask 或 actuator clipping；
- 不新增外部 ALOS、显式纵向 skip 或第二个残差网络；
- 不把短窗口 state 历史结果与 DADE 后端结果混成一个 arm；
- 不在 screen 前用 formal 预算训练 Transformer，也不先用大预算证明它再补 MLP 对照。

### 3.4 训练 profile

两个控制组从 2026-11-14 起的新训练统一绑定 canonical profile
`PPO-FLAT-4096x48-E5-MB4`：

| 项目 | 值 |
|---|---:|
| `num_envs` | 4096 |
| `num_steps_per_env` | 48 |
| `num_learning_epochs` | 5 |
| `num_mini_batches` | 4 |
| rollout samples / iteration | 196,608 |
| mini-batch size | 49,152 |
| gradient updates / iteration | 20 |

- 容量 screen 和架构 screen 都是 `5 iterations = 983,040 transitions` per arm；
- formal 每个训练 seed `16 iterations = 3,145,728 transitions`，至少 3 seeds；
- 配置顶层必须写 `"training_profile": "PPO-FLAT-4096x48-E5-MB4"`；
- 架构配对必须使用相同 profile，不能给 Transformer arm 更大 iteration 预算；
- 2026-11-14 前的旧配置不追溯修改，缺 profile 声明即 `historical_only`，不得与新协议混算。

本 profile 只固定 rollout/优化形状；学习率、clip、entropy、KL、critic warmup 和 reward
normalization 仍属于 §8.1 待商讨的 credit 配置。

### 3.5 actor/critic 配对规则

这里有两套不同的配对规则，不能混用。

**MLP 容量组**：MLP 容量 screen 不是只放大 actor。`A0/A0D/A0W` 的 actor 和 critic 必须
使用同一容量档：

| 档位 | `policy.actor_hidden_dims` | `policy.critic_hidden_dims` |
|---|---|---|
| `A0_MLP_256x128` | `256x128` | `256x128` |
| `A0D_MLP_256x128x64` | `256x128x64` | `256x128x64` |
| `A0W_MLP_512x256x128` | `512x256x128` | `512x256x128` |

规则：

- 禁止 actor 换成大 MLP、critic 仍保留旧的 `256x128` 或固定宽度；
- `A0M_MLP_512x256x128` 是 Transformer M 架构 control，不属于 MLP 容量 screen；它的
  actor 和 critic 都必须是 `512x256x128`，配置中显式标记 `role=architecture_control`；
- 若确实需要单独研究 critic 容量，必须新开 critic-capacity 因子/Goal，不能混进本轮 MLP
  actor 容量结论；
- 这项约束与 Transformer 无关，适用于任何 MLP actor 容量变化；
- 配置生成后必须直接比较 `actor_hidden_dims` 和 `critic_hidden_dims`，失败标签为
  `MLP_CRITIC_CAPACITY_MISMATCH`。

**Transformer 架构组**：critic 跟随 actor 容量档，不再对所有 arm 固定 `[256,128,64]`：

| arm 组 | `policy.critic_hidden_dims` |
|---|---|
| `A0 / A0D / A0W` | actor 同档（只用于 MLP 容量 screen） |
| `A0M_MLP_512x256x128` | `512x256x128`（A2M fused control） |
| `A1M_MLP_TOKEN` | `512x256x128`（A3M primary control） |
| `A1_MLP_TOKEN` | `256x128x64`（S 档 token-MLP control） |
| `A2M / A3M / A4 / A5 / A6` | `512x256x128`（M 档） |
| `A2_TA_FUSED_S / A3_TA_HISTORY_S` | `256x128x64`（S 档） |
| `Z0_ZERO_RESIDUAL` | 无 critic |

因此：

- Transformer 架构的 primary MLP control 是 `A1M_MLP_TOKEN`；它使用与 `A3M` 完全相同的
  token 和 mean-pool，hidden dims 由 Step 0 参数匹配，critic 使用 `512x256x128`；
- `A0D_MLP_256x128x64` 降为 S 档架构 control；`A0/A0W` 只用于 MLP 容量 screen，
  不能拿来当 M 架构 control；
- `A3M_TA_HISTORY_M` 是主架构，绑定 `A1M_MLP_TOKEN`；`A2M_TA_FUSED_M` 是 fused-token
  对照，绑定 `A0M_MLP_512x256x128`；
- M 档 arm 的 critic 不等于 `512x256x128`，或 S 档 arm 的 critic 不等于 `256x128x64`，
  判 `CRITIC_CAPACITY_MISMATCH`，不进入 formal。

## 4. 方法：独立替换 actor 架构，并让 MLP actor/critic 容量成对

### 4.1 候选 arm

所有 arm 共用同一个 BC checkpoint、reward、curriculum、residual authority 和 seed split。
所有 arm 的 critic 必须与 actor 使用同一容量档：M 档 `512x256x128`，S 档
`256x128x64`。具体映射见 §3.5。

架构实验保留四个正交因子，并增加一项 MLP actor/critic 镜像约束；arm 名不再使用 DADE，
DADE 只作为 `history_backend` 的一个取值：

```text
actor_arch      = mlp | token_mlp | transformer
actor_capacity  = S | M | L
mlp_critic_capacity = mirror_mlp_actor  # A0/A0D/A0W/A0M
critic_capacity     = M=512x256x128 | S=256x128x64
history_backend = dade | short_state_tokens | none
init            = random | distill
```

| arm | actor 结构 | 输入 token | history backend | 用途 |
|---|---|---|---|---|
| `A0_MLP_256x128` | 现有 `_mlp(256,128)`；critic 同档 | `z_t` + optional command token + `c_hat` token | `dade` | 仅 MLP 容量 screen |
| `A0D_MLP_256x128x64` | 更深 MLP；critic 同档 `256x128x64` | 同 `A0` | `dade` | MLP depth/capacity + S 档 control |
| `A0W_MLP_512x256x128` | 更宽 MLP；critic 同档 `512x256x128` | 同 `A0` | `dade` | 仅 MLP width/capacity screen |
| `A0M_MLP_512x256x128` | M 容量 MLP；actor/critic 都是 `512x256x128` | 同 `A0` | `dade` | **A2M fused-token control** |
| `A1M_MLP_TOKEN` | M 容量 token-MLP；hidden dims 由 Step 0 校准到 A3M 的 `±20%`，critic `512x256x128` | 与 `A3M_TA_HISTORY_M` 相同 token 的 mean-pool | `short_state_tokens` | **A3M primary MLP control**；排除 tokenization 混因 |
| `A1_MLP_TOKEN` | S 容量 token-MLP；critic 固定 `256x128x64` | 与 `A3_TA_HISTORY_S` 相同 token 的 mean-pool | `short_state_tokens` | S 档 tokenization 对照 |
| `A2_TA_FUSED_S` | 轻量 Transformer actor，`d_model=64, 2 layers`；critic `256x128x64` | `z_t` + optional command token + `c_hat` token | `dade` | 可选 S 档消融，不进入主结论 |
| `A2M_TA_FUSED_M` | M 容量 Transformer actor，`d_model=112, 2 layers, FFN=224`；critic `512x256x128` | 同 `A2_TA_FUSED_S` | `dade` | fused-token 对照，不作主结论 |
| `A3_TA_HISTORY_S` | 轻量 Transformer actor，`d_model=64, 2 layers`；critic `256x128x64` | current token + `K` 个短窗口 state token | `short_state_tokens` | 可选 S 档 history 消融 |
| `A3M_TA_HISTORY_M` | M 容量 Transformer actor，`d_model=112, 2 layers, FFN=224`；critic `512x256x128` | 同 `A3_TA_HISTORY_S` | `short_state_tokens` | **主架构**：M 容量短历史 token，DADE off |
| `A4_TA_CURRENT` | Transformer actor，M 容量；critic `512x256x128` | 仅 current token | `none` | 无历史对照 |
| `A5_TA_FUSED_DISTILL` | 同 `A2M_TA_FUSED_M` | 同 `A2M_TA_FUSED_M` | `dade` | 初始化对照 |
| `A6_TA_HISTORY_DISTILL` | 同 `A3M_TA_HISTORY_M` | 同 `A3M_TA_HISTORY_M` | `short_state_tokens` | 初始化对照 |
| `Z0_ZERO_RESIDUAL` | 冻结 BC，残差恒 0 | 无 actor | `none` | 绝对 zero baseline |

`A0M` 与 `A0W` 结构相同（actor/critic 都是 `512x256x128`，fused token），区别是角色：
`A0W` 只用于 MLP 容量 screen，`A0M` 是 A2M 的 formal control。实现可以复用同一个模型类，
但配置和结果必须保留两个角色标签，不能把容量 screen 的结果直接当 formal 结论。

`A2M_TA_FUSED_M` 的 token 数是现有 `z_t / command / c_hat` 三个模态自然决定的，不是人为调参：
路径条件组 `command_dim=0` 时是 2 个 token，ALOS-body 组有 command 时是 3 个 token。它只测
“少量 fused token 上的 actor 映射”；如果 2-token 版本没有信号，不能直接推出“attention
对所有序列都无效”，还需要看 `A3M_TA_HISTORY_M`。

screen 分成两小段：

1. **容量 screen**：`A0_MLP_256x128`、`A0D_MLP_256x128x64`、`A0W_MLP_512x256x128`，每个
   arm 同时把 actor 和 critic 换到同一档，先确认 MLP actor-critic 容量本身是否值得升级；
2. **架构 screen**：以 M 档为主，`A3M_TA_HISTORY_M` 对 `A1M_MLP_TOKEN`，
   `A2M_TA_FUSED_M` 对 `A0M_MLP_512x256x128` 作为 fused-token 对照，并用
   `A4_TA_CURRENT` 做无历史对照；
   M 档架构 screen arm 的 critic 固定 `[512,256,128]`；S 档 `A2/A3` 的 critic 固定
   `[256,128,64]`，只作为可选消融，
   资源不足时可跳过；`A5/A6` 至少保留到 formal 前。

formal 只保留 M 主架构、其 `A1M` 主对照、`A0M` 的 fused 对照和必要的 distill 配对；
S 档不进入 formal 主结论。

### 4.2 Token 契约

第一版不改变环境 observation 宽度，直接从现有 `PCPolicyLayout` 取 token。先只加**短窗口
state 历史**，其它历史先不加：

```text
current_token_t = [state_t, path_t, current_t, command_t if command_dim > 0]
history_token_k = state_{t-k}, k = 1..K
K               = 现有 history_short_window（默认 10）
```

- `A2M_TA_FUSED_M` 是 fused-token 对照，只把现有 `z_t`、optional command、`c_hat` 视为
  token，不改变上游 encoder；
- `A3M_TA_HISTORY_M` 是主架构，关闭 DADE，history token 只取 state 的短窗口，不取 path history、
  command history、side-channel history 或 raw action history；
- 第一轮不加 long window；等 `A3M_TA_HISTORY_M` 出现信号后，再单独开
  `A7_TA_LONG_HISTORY`，不能直接把 long window 塞进主 arm；
- 第一轮不加 path history、current/side-channel history 和 action history；这些都需要独立的
  配对 arm，否则无法判断收益来自哪类历史；
- `A4_TA_CURRENT` 只保留 current token，用来区分“历史 token 收益”和“只是拆分当前输入”；
- `A1_MLP_TOKEN` 使用与 `A3M_TA_HISTORY_M` 相同 token 的 mean-pool，避免把“拆 token”误判为
  attention 收益；
- 所有 positional encoding、attention mask、padding 和 reset 语义必须写入 checkpoint metadata。

token 数不是拍脑袋决定的，规则是：

1. 一个语义单元一个 token，不把一个标量拆成一个 token；
2. 第一轮只保留当前契约已经有的模态；
3. 历史 token 只从已经存在、且与纵向/俯仰最相关的 state 短窗口开始；
4. 每增加一类历史，都必须有一个匹配的 no-history/MLP 对照，并且单独成 arm。

### 4.3 初始化

至少保留两类初始化，避免把“架构收益”误判为“初始化收益”：

1. random：`A3M`（主）与 `A2M`（fused 对照）默认初始化；
2. distill：`A5_TA_FUSED_DISTILL` / `A6_TA_HISTORY_DISTILL`，在固定 dev rollout 上用监督
   回归匹配容量匹配的 MLP arm 的 action mean；

蒸馏只用于 warm start，不进入 reward，不使用 held-out，不据验证集结果选择 checkpoint。
蒸馏期间冻结上游 encoder/path encoder/history backend，只训练 actor。若蒸馏 MSE 不能优于
随机初始化，则该 arm 标记 `INVALID_DISTILL_INIT`，不进入 formal。

### 4.4 输出与 authority

- 所有 Transformer arm 继续输出 3D normalized actuator residual；
- 使用与容量匹配的 MLP arm 完全相同的 residual scale、channel mask、rate/magnitude limiter；
- 不把 actor 改成完整动作生成器；完整动作替换属于另一个控制契约 Goal；
- 记录 raw residual、applied residual、composed action，三者不得混报。
- PPO 前测初始 residual RMS；若某个 Transformer arm 超过 MLP 的 `0.5x–2x`，先校准输出层，
  否则标 `INITIAL_RESIDUAL_MISMATCH`，不进入 screen。

### 4.5 容量匹配规则

MLP 深度/宽度和 Transformer 宽度都是 capacity 因子，不是 architecture 因子。所有架构结论
必须来自容量配对的 arm：

- 每个要进入架构结论的 Transformer arm 绑定一个 MLP control，参数量差不超过 `±20%`；
  M 主架构 `A3M` 绑定 `A1M_MLP_TOKEN`，M fused 对照 `A2M` 绑定
  `A0M_MLP_512x256x128`，S 消融绑定 `A0D_MLP_256x128x64`；
- actor forward p95 差不超过 `±20%`，并同时满足绝对控制周期预算；
- 如果 Transformer 参数量比配对 MLP 多 `>20%`，该 arm 只能标成 `capacity arm`，不能用来
  证明“Transformer 架构更好”；
- `A1M_MLP_TOKEN` 与 `A3M_TA_HISTORY_M` 使用完全相同的 token/mean-pool，并做参数和延迟匹配；
- `A1_MLP_TOKEN` 与 `A3_TA_HISTORY_S` 使用完全相同的 token/mean-pool；
- `A0/A0D/A0W` 必须同时报告 actor 和 critic 的参数量、forward p95；只改 actor 的 arm
  直接判 `MLP_CRITIC_CAPACITY_MISMATCH`，不进入 screen 比较；
- M 档 arm `A0M/A1M/A2M/A3M/A4/A5/A6` 的 `policy.critic_hidden_dims` 必须是
  `[512,256,128]`；S 档 arm `A1/A2_TA_FUSED_S/A3_TA_HISTORY_S` 必须是 `[256,128,64]`；
  任一不匹配判 `CRITIC_CAPACITY_MISMATCH`；
- `A0_MLP_256x128` 和 `A0W_MLP_512x256x128` 只做容量 screen，不进入架构 formal；
  `A0D` 只作为 S 消融 control；`A0M` 是 A2M 的 fused control；`A1M` 是 A3M 的
  primary MLP control，`A0M` 与 `A1M` 都允许进入 formal。

推荐起始容量阶梯（最终以 Step 0 实际参数统计为准）：

| 档位 | MLP control | Transformer 起始配置 | 说明 |
|---|---|---|---|
| S（可选） | `256x128x64` | `d_model=64, 2 layers, 4 heads, FFN=128` | 只作消融，不进入 formal 主结论 |
| **M（主）** | `A1M_MLP_TOKEN`（同 token mean-pool）；A2M 用 `A0M` | `d_model=112, 2 layers, 4 heads, FFN=224` | 主架构；A3M 必须与 `A1M` 容量配对 |

Step 0 只允许在 `d_model=104/112`、`FFN=208/224` 内校准一次；校准后参数量仍超出 `±20%`
则判 `CAPACITY_MISMATCH`，不能进入 screen。L 档当前不启用；若以后需要，必须新增独立 arm
和同容量 MLP control，不能直接沿用 M 的结论。

## 5. 执行计划

### Step 0：只读审计与决策锁定

- [ ] 核对 program 仍只有两个控制契约组，`AX-ACTOR-ARCH` 不被误写成第三个组；
- [ ] 核对当前 actor 仍是 `pc_policy.py:811` / `encoded_policy.py:374` 的 MLP；
- [ ] 确认路径组的 Goal180 MLP 基线和 ALOS 组的 Goal175/Goal179 基线是否满足前置门；
- [ ] 锁定 BC checkpoint、reward、residual authority、训练 seed、dev seed、评价 levels/radii；
- [ ] 锁定 M 为主架构，S 为可选消融；`actor_arch × actor_capacity × history_backend × init`
  因子、token 契约和部署预算；
- [ ] 锁定 critic 容量档：M 档 `[512,256,128]`，S 档 `[256,128,64]`；并以
  `A1M_MLP_TOKEN` 作为 A3M primary MLP control，`A0M_MLP_512x256x128` 作为 A2M control；
- [ ] 运行 canonical training-profile 检查，确认所有新训练配置绑定
  `PPO-FLAT-4096x48-E5-MB4`；
- [ ] 统计 `256x128`、`256x128x64`、`512x256x128`、`A0M`、`A1M` 与 M Transformer 的
  实际参数量；
- [ ] 写 `goal181_step0.md`，未锁定前不训练。

### Step 1：实现 actor kind 与兼容层

- [ ] 增加显式 `actor_kind = mlp | token_mlp | transformer`，默认 `mlp`；
- [ ] `A0/A0D/A0W` 容量配置同时写 `policy.actor_hidden_dims` 与同档
  `policy.critic_hidden_dims`，并加测试禁止 actor-only 扩宽；`A0M` 是 M 架构 control，
  必须写 `actor_hidden_dims=[512,256,128]` 和 `critic_hidden_dims=[512,256,128]`；
- [ ] M 档 arm `A0M/A1M/A2M/A3M/A4/A5/A6` 全部写
  `policy.critic_hidden_dims=[512,256,128]`；S 档 `A1/A2_TA_FUSED_S/A3_TA_HISTORY_S`
  写 `[256,128,64]`；加测试禁止 critic 与 actor 容量档不一致；
- [ ] 新增 Transformer actor module，保持输入/输出宽度和 residual 接口不变；
- [ ] checkpoint metadata 记录 actor kind、history backend、token 契约、参数数和序列长度；
- [ ] 旧 checkpoint 必须能无歧义加载为 MLP；缺少 actor kind 时禁止静默推断为 Transformer；
- [ ] 增加 forward、mask、梯度、旧 checkpoint 兼容、参数/延迟测试；
- [ ] 运行 `pytest` 相关测试和 `git diff --check`。

### Step 2：eval-only 归因

- [ ] 用固定 MLP checkpoint 采集 dev rollout；
- [ ] 训练 `A5_TA_FUSED_DISTILL` / `A6_TA_HISTORY_DISTILL`，报告 action MSE、R² 和逐通道误差；
- [ ] 对 `A1M_MLP_TOKEN`、`A0M`、`A3M`（主）、`A2M`（fused 对照）、`A4_TA_CURRENT`
  做同状态 Jacobian；
- [ ] 记录 applied share、sign consistency、参数数、p50/p95 forward latency；
- [ ] 输出 `goal181_step2.json` 和 `goal181_step2.md`。

### Step 3：screen

- [ ] 每个控制组使用自己的已锁定 reward 和 MLP 基座：路径组用 Goal180，ALOS 组用
  Goal175/Goal179；
- [ ] 3A 先跑 MLP 容量 screen：`A0_MLP_256x128`、`A0D_MLP_256x128x64`、
  `A0W_MLP_512x256x128`，每个 arm 的 critic 与 actor 同档；1 个训练 seed、每个 arm
  5 iterations（983,040 transitions）；
- [ ] 3B 以 M 为主：`A3M_TA_HISTORY_M` 对 `A1M_MLP_TOKEN`，
  `A2M_TA_FUSED_M` 对 `A0M_MLP_512x256x128` 作为 fused-token 对照，并用
  `A4_TA_CURRENT` 做无历史对照；
- [ ] 3B 的 primary architecture comparison 固定为 `A3M_TA_HISTORY_M` 对
  `A1M_MLP_TOKEN`；S 档 `A2/A3` 只作可选消融；
- [ ] 只在 `current10` 的 `default` / `strict125` 上做 pilot；
- [ ] 每个 arm 必须有 zero-residual equivalence 检查；
- [ ] 记录每个 arm 的 actor/critic 参数量、forward p95、初始 residual RMS 和容量匹配比；
- [ ] 验证 fused MLP arm `A0/A0D/A0W/A0M` 的 `actor_hidden_dims == critic_hidden_dims`；
  `A0M` 必须是 `512x256x128`；不等则判 `MLP_CRITIC_CAPACITY_MISMATCH`；
- [ ] `A1M_MLP_TOKEN` 的 actor hidden dims 由 Step 0 校准，但 critic 必须是
  `512x256x128`；actor 参数与 A3M 的差不超过 `±20%`；
- [ ] 验证 M 档 arm 的 `critic_hidden_dims == [512,256,128]`，S 档 arm 的
  `critic_hidden_dims == [256,128,64]`；否则判 `CRITIC_CAPACITY_MISMATCH`；
- [ ] 输出 screen 表，不写 formal 结论。

### Step 4：formal matrix

- [ ] 只把有 screen 信号或必要诊断价值的 arm 放入 formal；M 主架构及其 `A1M` 对照优先，
  S 档不进入 formal 主结论；
- [ ] 为两个控制组分别建立 matrix：
  `configs/governance/goal181_matrix_path.json` 与
  `configs/governance/goal181_matrix_alos.json`；
- [ ] 覆盖 `arms × checkpoint_seeds × levels × radii + zero baselines`；
- [ ] 正式训练每个 seed 跑 16 iterations（3,145,728 transitions），至少 3 个训练 checkpoint seeds；
- [ ] 输出分组的 `goal181_result_path_20261114.md` 和
  `goal181_result_alos_20261114.md`，更新 ledger 和 program `matrix_refs`；
- [ ] 两张 matrix 分别挂到对应控制组，各自使用自己的 seed 池，不 pooled。

## 6. Gate

### 6.1 前置门

| check | 阈值 | 失败标签 |
|---|---|---|
| head/credit 前置 | Goal179 或等价诊断已给出 residual authority 可测、无 BC/PPO 完全抵消的结论 | `BLOCKED_ON_HEAD_CREDIT` |
| MLP 基线 | 路径组使用 Goal180 的完整 dev matrix；ALOS 组使用 Goal175/Goal179 已锁定基线；不得用未审计 smoke 代替 | `BLOCKED_ON_MLP_BASELINE` |
| 实现隔离 | 只改变 actor；encoder/path/reward/action pipeline 有配对哈希 | `ACTOR_ISOLATION_BROKEN` |
| 旧 checkpoint | 缺 actor kind 的 checkpoint 只能加载为 MLP | `SILENT_ACTOR_METADATA` |
| 容量匹配 | M 主架构 `A3M` 与 `A1M_MLP_TOKEN` 的参数量差不超过 `±20%`，forward p95 差不超过 `±20%`；A2M 与 `A0M` 同规则 | `CAPACITY_MISMATCH` |
| MLP actor/critic 成对 | `A0/A0D/A0W` 的 `critic_hidden_dims` 必须与 `actor_hidden_dims` 完全相同；`A0M` 必须是 `512x256x128` actor + `512x256x128` critic | `MLP_CRITIC_CAPACITY_MISMATCH` |
| critic 容量档 | M 档 `A0M/A1M/A2M/A3M/A4/A5/A6` 的 `critic_hidden_dims` 必须是 `[512,256,128]`；S 档 `A1/A2_TA_FUSED_S/A3_TA_HISTORY_S` 必须是 `[256,128,64]` | `CRITIC_CAPACITY_MISMATCH` |
| 部署预算 | 参数 `<=2x` MLP；actor forward p95 `<=2x` MLP；无额外 reset 语义变化 | `ACTOR_DEPLOYMENT_BUDGET_FAIL` |

### 6.2 架构信号门

| check | 阈值 | 失败标签 |
|---|---|---|
| Jacobian | M 主臂 `A3M` 的 applied Jacobian p90 相对 `A1M` `>=2x` | `ACTOR_JACOBIAN_NO_SIGNAL` |
| 路线主指标 | current10 的 cross 或 vertical route p90 改善 `>=5%` 且 `>=0.02 m` | `ACTOR_ROUTE_NO_GAIN` |
| tokenization 对照 | 相对 `A1_MLP_TOKEN` 仍保留至少一半改善，否则不能归因于 attention | `TOKENIZATION_CONFOUND` |
| 初始化对照 | random 或 distill 至少一个成立；收益不能只出现在单初始化 | `INITIALIZATION_CONFOUND` |
| 容量归因 | 架构收益在容量匹配的 MLP control 上仍成立；不能用更大的 Transformer 对比小 MLP | `CAPACITY_CONFOUND` |
| 历史归因 | 若声称 history 是来源，`A3M_TA_HISTORY_M` 必须优于 `A4_TA_CURRENT` | `HISTORY_ATTRIBUTION_WEAK` |

### 6.3 formal 门

| check | 阈值 | 失败标签 |
|---|---|---|
| 配对收益 | 至少 2/3 训练 seeds 的 cross 或 vertical p90 改善 `>=4%`，且 3D route RMSE 改善 `>=3%`；对照必须是容量匹配 MLP | `ACTOR_ARCH_UNSTABLE` |
| 负控 | zero-residual 与 MLP baseline 的轨迹契约一致 | `ACTOR_NEGATIVE_CONTROL_FAIL` |
| success/safety | success 下降不超过 2/40；timeout/safety 不增加 | `ACTOR_SUCCESS_GUARD_FAIL` |
| 分段指标 | 至少 3/5 段和 5/8 个 20–90% keypoint 改善 | `ACTOR_SEGMENT_NO_GAIN` |
| 部署 | 参数和延迟仍在 Step 0 预算内 | `ACTOR_DEPLOYMENT_BUDGET_FAIL` |

## 7. 失败裁决

| 现象 | 判读 | 下一步 |
|---|---|---|
| latent probe 好，MLP Jacobian 差，Transformer Jacobian 也差 | 不是 actor 架构单独能解决 | 回到 head/authority/credit 诊断 |
| Transformer 与 token-MLP 无差异 | 收益可能来自 tokenization 或参数容量 | 检查 token 契约和参数匹配，不宣称 attention 收益 |
| 只有蒸馏初始化有效 | 收益可能来自 MLP 教师，不是架构 | 保留 distill 作为 warm start，但 formal 必须与 random 分开 |
| screen 有信号、formal 跨 seed 不稳定 | 可能是 seed 噪声或容量过拟合 | 进入 autoresearch，暂停推广 |
| 路线改善但 success/safety 回退 | 架构改变了风险行为 | Gate 失败，不因单一 RMSE 改善放行 |

## 8. 需要用户拍板的项目与建议默认值

| # | 决策 | 建议默认 | 影响 |
|---|---|---|---|
| 1 | Transformer 是第三个 program group 还是架构轴 | 架构轴 `AX-ACTOR-ARCH`，不新增第三个控制契约组 | 避免控制契约与网络结构混因 |
| 2 | 两个控制组的顺序 | 两组都做；顺序只由资源决定，不作为科学变量 | 避免把调度顺序误写成结论差异 |
| 3 | 主 Transformer arm | `A3M_TA_HISTORY_M` 为主架构（短历史 token，DADE off）；`A2M_TA_FUSED_M` 为 fused-token 对照；S 档只作可选消融 | M 容量主结论；避免回到 DADE 主线 |
| 4 | MLP 容量 | screen 跑 `256x128`、`256x128x64`、`512x256x128`；formal 主对照固定 `A1M_MLP_TOKEN`（同 token、mean-pool、actor/critic 都 M）；A2M 用 `A0M`（actor/critic 都 M） | 先确认容量收益，再做架构配对；防止 tokenization/critic 混因 |
| 5 | history backend | `A3M` 用 `short_state_tokens`；`A2M` 用 `dade` 仅作 fused-token 对照；不混 arm | 避免双重历史编码混因，主架构不依赖 DADE |
| 6 | 输出类型 | 继续输出 3D 执行器残差 | 不改变 BC authority 和动作管线 |
| 7 | 初始化 | 同时保留 random 与 distill | 区分架构收益与教师初始化收益 |
| 8 | Transformer 容量 | 主架构 M：`d_model=112, 2 layers, 4 heads, FFN=224`；Step 0 只允许在 104/112、208/224 内校准一次 | M 与 `A1M` 容量匹配；避免用更大模型冒充架构收益 |
| 9 | screen / formal 预算 | canonical profile `PPO-FLAT-4096x48-E5-MB4`；screen 5 iterations（983,040 transitions）；formal 至少 3 seeds × 16 iterations（3,145,728 transitions/seed） | 先筛后立项，避免大预算偏置 |
| 10 | 部署预算 | 每个配对 `±20%`，总体 `<=2x` 最大 MLP；forward p95 同规则 | 防止只在离线评估有效 |
| 11 | 基线前置 | 路径组等 Goal180；ALOS 组等 Goal175/Goal179 的锁定基线；两组都可失败但必须完整 | 没有可比基座就不做架构结论 |
| 12 | critic 容量 | M 档 actor/critic 都是 `[512,256,128]`；S 档 actor/critic 都是 `[256,128,64]`；primary MLP control 为 `A1M_MLP_TOKEN`，A2M 用 `A0M` | 容量轴同时作用于 actor 和 critic，避免 critic 容量混因 |

### 8.1 仍需和你商讨、当前不锁定

- M 主架构已锁定 `d_model=112, 2 layers, 4 heads, FFN=224, dropout=0, pre-LN, GELU,
  learned positional embedding`；仍需锁定 pooling、token type embedding 和输出层初始化；
- MLP 容量 screen 的 winner 选择规则：只看 route 指标，还是同时按参数/延迟做 Pareto；
- 容量匹配容差是否固定为 `±20%`，还是按实际控制周期和参数统计调整；
- screen 时上游 encoder/path/history backend 是否全部冻结，还是允许只解冻最后一层；
- PPO credit 配置：learning rate、epochs、minibatch、clip、entropy、KL、critic warmup、
  reward normalization；
- 是否等待 Goal182 的 PPO authority / normalization / trust-region profile winner，并在
  本 Goal formal 中统一使用；Goal182 未锁定前不比较架构 formal；
- distill warm start 的步数、损失函数和是否与 random arm 匹配优化预算；
- 每个控制组最终的训练 seed、dev seed、transitions 预算和绝对延迟上限。

除 8.1 的待商讨项外，如果以上默认值都接受，可以回复“其余按推荐”，再把状态推进到
`PROTOCOL_LOCKED` 并创建 matrix；在此之前本 Goal 不训练。

## 9. Git 与仓库边界

- 本文档、actor 实现、matrix 和结果都在 `settled_局部规划器_反应式`；
- 实验组/架构轴的治理说明在 `总协调`，两边分别提交；
- 禁止 `git add -A`；提交前运行 `git diff --cached --check`；
- 不自动 commit / push；
- 不修改或清理 Goal175–180 的未提交 WIP。

## 10. Goal181 执行提示词

```text
你现在执行 Goal181：Actor 架构与容量配对实验（MLP / Transformer）。

先读：
- doc/goals/goal181_transformer_actor_architecture_20261114.md
- doc/goals/goal180_path_conditioned_policy_20261114.md
- doc/goals/goal179_head_only_credit_diagnostics_20261114.md
- doc/goals/goal175_alos_comp_mlp_route_matrix_20261114.md
- doc/goals/goal177_result_20261114.md
- 总协调/configs/programs/auv_control_programs.json

硬约束：
1. 当前两个控制契约组不变；Transformer 只作为 AX-ACTOR-ARCH 正交架构轴；
2. G-PATH-CONDITIONED-POLICY 和 G-ALOS-BODY-RL 都要做；顺序只由资源决定，不作为科学变量；
3. 当前 dual_horizon_transformer 只是历史编码器，不得只改 encoder.kind 冒充 actor 替换；
4. 所有 arm 固定 BC checkpoint、reward、curriculum、residual authority、seed 和评价矩阵；
5. 默认输出 3D 执行器残差，不改变冻结 BC authority；
6. MLP screen 至少包含 256x128、256x128x64、512x256x128 三档；
7. MLP arm `A0/A0D/A0W` 的 actor/critic 必须同档；`A0M` 是 A2M fused control，
   actor/critic 都是 `512x256x128`；`A1M` 是 A3M primary control，actor hidden dims
   校准到 A3M 的 `±20%`，critic 必须是 `512x256x128`；
8. M 档 arm 的 critic 一律 `[512,256,128]`，S 档 arm 的 critic 一律 `[256,128,64]`；
   M primary MLP control 为 `A1M_MLP_TOKEN`，M fused control 为 `A0M_MLP_512x256x128`；
9. 主架构 M 锁定 `d_model=112, 2 layers, 4 heads, FFN=224`；Step 0 只允许在
   `104/112`、`208/224` 内按参数 `±20%` 校准一次；S 档仅作可选消融；
10. 必须有 MLP、token-MLP、Transformer、zero-residual 四类对照；
11. 至少保留 random 和 distill 两种初始化；
12. 新训练配置必须绑定 `PPO-FLAT-4096x48-E5-MB4`；screen 5 iterations、formal 每 seed 16 iterations；
13. 2026-11-14 前配置只作历史，不重写、不当作同一训练协议；
14. 各组前置未满足时标记 BLOCKED，不训练：路径组看 Goal180，ALOS 组看 Goal175/Goal179；
15. held-out 100–149 全程关闭；
16. 不自动 commit/push，不修改 Goal175–180 WIP。

执行顺序：
Step 0 只读审计和决策锁定 -> Step 1 actor kind/兼容实现 -> Step 2 eval-only 归因
-> Step 3A MLP 容量 screen -> Step 3B 容量匹配的架构 screen -> Step 4 formal matrix。
任何 Gate 失败都写 blocker、至少两个可证伪 hypothesis 和 ledger decision，再决定
continue/pivot/stop。
```

## 11. 自主迭代契约

本 Goal 执行 `research/AUTORESEARCH_LOOP.md`：

1. 失败后先归因，不直接换架构或扫 seed；
2. 每个失败假设至少写一个便宜证伪实验；
3. 跨 seed 结果低于噪声、或两个方法不可区分时，按 `seed_noise_ambiguous` 进入 research loop；
4. 任何架构推广都必须同时报告收益、部署代价、success/safety 和负控；
5. 本 Goal 的 `continue` 只表示继续验证同一架构轴，不表示架构收益自动跨控制契约成立。
