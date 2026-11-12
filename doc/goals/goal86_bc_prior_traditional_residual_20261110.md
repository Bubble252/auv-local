# Goal86：BC 先验与传统控制器 RL 残差的分离对照

**日期**：2026-11-10<br>
**状态**：`STEP0_PASS / SAC_Z_T_ADAPTER_SMOKE_PASS / P-A_BC_INIT_ONLY / P-A_BC_PRIOR_CONTRACT_MISMATCH / P-C_U2_NOT_AVAILABLE`<br>
**上游**：Goal85 位置契约；Goal24/25/26 的 BC 底座；Goal27/58/59 的命令残差；Goal85b plant mismatch<br>
**性质**：探索型训练 Goal。先做 BC 先验，再学习残差；历史 BC+残差只读不改，传统控制器+RL 残差是新增独立实验组。新学习 arm 的算法固定为 on-policy PPO 与 off-policy SAC；不打开 held-out。

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
  check_command: python ../scripts/check_research_ledger.py --goal Goal86 --goal-doc doc/goals/goal86_bc_prior_traditional_residual_20261110.md --strict
```

`required: true`：本 Goal 会训练新的 BC 先验和残差策略。Gate 记录前只能做实现、
smoke、screen 和预注册的正式训练；不得根据 dev 结果改 seed 预算、Gate 或动作接口。
触发 blocker 后按根目录 `AGENTS.md` 写 blocker、至少两个可证伪 hypothesis、reading
note 和 `research/LEDGER.jsonl` decision。

## 1. 背景与证据

### 1.1 历史 BC+残差

Goal24/25/26 的路线是：

```text
教师闭环示范
  -> D3 BC（state + path -> [rudder, stern, rpm]）
  -> 冻结 D3 BC
  -> 在 actuator 动作上学习残差
```

这条线有 BC 先验，zero-residual 机制也通过，但 Goal26 的 P1/P2 只让 pre-goal RMSE
改善约 `0.014–0.017 m`，success 没有提升，最终为 `FAIL / STOP`。因此本 Goal 不把
“有 BC”误写成“BC 一定有效”。

### 1.2 最近命令残差没有走 BC

Goal27 C1、Goal58 和 Goal59 有意改成：

```text
ALOS 3D guidance
  -> [psi_d, theta_d, u_d] base command
  -> 3D command residual（从零初始化学习）
  -> PD/PI
  -> [rudder, stern, rpm]
```

Goal58/59 的 command-residual 线没有把 D3 BC 作为 base；Goal58 配置中的
`bc_auxiliary_weight` 为 `0.0`。Goal58 残差饱和且 terminal capture 不改善，Goal59
继续审计终端门控和 12D endpoint 观测。它们回答的是 off-policy/command-residual
问题，不是“BC 先验是否有用”。

### 1.3 本 Goal 的三个独立问题

用户明确要求保留原来的 BC+残差设定，并额外做一个“传统控制器 + RL 残差”实验组。
因此本 Goal 只回答两个问题：

1. **BC 先验**能否让新的残差学习更省样本、更稳，或在同预算下更好；
2. **传统控制器 + RL 残差**能否超过同一个传统控制器的 zero-residual 基线；
3. **PPO 与 SAC** 在同一 base、同一动作接口、同一 BC 初始化下，哪个更稳、更省样本。

这些问题不能在同一个 arm 里混答。历史 BC+残差是只读参考，不修改其算法、网络、
reward、动作接口、训练预算或 checkpoint 选择规则。

### 1.4 依赖与硬约束

- 新训练组使用 Goal85 的 `unbiased_noise_v1`、零传感器延迟和真值 `eta`
  termination/reward/success 契约；Goal85 契约已实现并通过测试，Step 0 审计
  确认 4 个 Goal85 配置均为 `unbiased_noise_v1` + `position_source=eta`。
- `plant_mismatch` 是 Goal85b 的变量，不塞进 Goal86；Goal85b 已完成并判定
  `MISMATCH_TOO_SMALL`，因此 Goal86 的 nominal 结果不能写成 plant-robust 结论。
- Step 0 审计结论见 `doc/goals/goal86_step0_result_20261110.md`：D3 BC 只能作为
  actuator 先验；没有有效 command teacher，`P-C/U2` 标 `NOT_AVAILABLE`；PPO 入口
  可用，SAC 只有上游 SB3 源码，PC residual adapter 尚未实现。
- 项目 `AGENTS.md` 当前规定“不继续 DADE 主线，不新增第二个学习残差网络”。因此
  DADE 不进入 Goal86；若未来要重开，必须另开 Goal 并先修订该硬约束。
- 新学习 arm 只允许 PPO 或 SAC；DDPG/TD3 只保留为 Goal58/59 历史只读结果，不再
  新增训练。
- 一个 arm 只允许一个学习残差网络；command residual 与 actuator residual 不能同时
  叠加。两者只能作为互相独立的 arm 比较。
- `ideal_actor_observations=false`。教师可以使用特权信息生成标签，但 actor 输入必须
  保持可部署观测；不得把真值 `eta`、真实海流或教师特权状态泄漏给 actor。

### 1.5 P-A 残差 BC 契约阻塞（2026-11-10）

`P-A` 的预注册语义是“ALOS+PD/PI base + D3 教师残差 BC”：

```text
label = normalize(D3_actuator - ALOS_actuator)
```

当前 PPO/SAC 的 P-A 初始化只加载了 D3 的 state/path encoder 和 actor 主干，然后把
最终输出层清零；它没有拟合上述残差标签。因此当前 P-A 只能称为“D3 表征先验 +
零残差初始化”，不能称为“BC 残差先验”。

只看现有 25,600 条训练分布标签，在 `residual_scale=0.25` 下，归一化残差的
rudder/stern 超出 `[-1,1]` 的比例分别为 `88.45%` 和 `73.86%`；`scale=0.50`
仍为 `75.66%/52.95%`，`scale=1.00` 才降到 `0.77%/7.47%`。这支持
H86-PA-1（残差包线过小），但把 `1.00` 用于正式协议会改变 T2/P-A 的动作包线，
必须先记录协议修订并让 T2 使用同一 scale。完整证据和两个可证伪假设见
`research/blockers/2026-11-10-goal86-pa-residual-contract.md`。

当前裁决：`T1/T2` 的 PPO/SAC screen 可以继续；`P-A` 在契约修复前不进入 screen。
不换算法、不改 Gate、不打开 held-out。

## 2. 可证伪假设

### H86-A：BC 先验能提高 actuator residual 的样本效率或最终表现

机制：D3 BC 已经学到了可部署的 actuator 动作先验。把
`teacher_action - traditional_controller_action` 作为监督标签预训练残差 actor，
可以让 RL 从有方向的残差附近开始，而不是从全零附近探索。

成立应看到：

- 在同总 environment transitions、同训练 seeds、同评价 seeds 下，`BC-prior + residual`
  在 1M screen 的 true-eta terminal capture 比 `scratch + residual` 高 `>=5 pp`，
  或在达到相同 screen 指标时使用不超过一半的 transitions；
- formal 5M 不出现超过 `2 pp` 的主场景回退；
- residual 不是被 BC 标签压成近零，且 zero-residual parity 通过。

失败应看到：

- BC-prior 与 scratch 的差异小于 seed 噪声；
- BC-prior 只在离线动作误差上更好，闭环 terminal capture 不变；
- BC 标签本身接近零，或残差被 BC 约束得过小。

最便宜证伪：1 个训练 seed、1M transitions 的 screen，在 PPO 内比较 `P-A-PPO` 与
`T2-PPO`，在 SAC 内比较 `P-A-SAC` 与 `T2-SAC`；不把两个算法的结果混算。

### H86-B：传统控制器 + RL 残差能超过同一控制器的 zero-residual

机制：ALOS+PD/PI 或 `path_fixed_speed`+PD/PI 提供强基线，但有界残差可以修正终端
捕获、路径偏差和未建模动力学；前提是残差不能破坏基线的安全与稳定性。

成立应看到：

- 同一 base、同一 dev seeds、同一 termination 契约下，至少一个主场景的
  `eta_tight_at_done` 相对 zero-residual 提升 `>=5 pp`；
- 另一场景回退不超过 `2 pp`；
- safety、timeout、plane-miss 和 material saturation 不出现预注册护栏之外的恶化；
- path RMSE 不恶化超过 `5%`，且 residual 的 gate-active L2 不为零。

失败应看到：residual 很大但 success/terminal capture 不动，或 residual 缩到近零；
此时不能把换 PPO/SAC 或 REDQ/DroQ 当第一反应，先查动作参数化、reward 和
terminal capture。

最便宜证伪：`T0` 分别与 `T1-PPO/T1-SAC` 或 `T2-PPO/T2-SAC` 使用同环境 seeds 配对，
先用 1M screen 看方向和 residual 幅度；screen 不通过不扩 formal。

### H86-C：command residual 比 actuator residual 更适合传统控制器

机制：command residual 在 `[psi_d, theta_d, u_d]` 语义上直接修正制导目标，再由
PD/PI 保持执行器约束；actuator residual 更接近低层动作，但可能被执行器限幅、
速率限制和 PD/PI 反馈吞掉。

成立应看到：在同一个 ALOS+PD/PI base、同动作初始化、同预算、同一个算法内，
command residual 的 true-eta terminal capture 至少一个场景高 `>=5 pp`，另一场景
不低 `>2 pp`，且 safety/timeout 不恶化。PPO 和 SAC 分别判断，不合并。

失败应看到：两者的差异小于 seed 噪声，或 actuator residual 更稳；此时不把
“command space 更高级”写成结论，只按数据选择接口。

最便宜证伪：`T1-PPO` 与 `T2-PPO`、`T1-SAC` 与 `T2-SAC` 分别做 1M screen 配对；
screen 只作机制诊断，不把单 seed 差异写成正式 PASS。

### H86-D：SAC 的 off-policy 样本效率优于 PPO，但 PPO 的最终表现可能追上

机制：SAC 使用 replay、双 Q 和 entropy 正则，可能更早利用已有 transition；PPO 是
on-policy，样本效率通常较低，但在固定总 transitions 足够时可能更稳定。

成立应看到：在同一 base、同一动作接口、同初始化、同一 environment transition 预算
下，SAC 在 1M screen 的 true-eta terminal capture 比 PPO 高 `>=5 pp`，或达到同指标
使用更少 wall-clock。

失败应看到：SAC 与 PPO 的差异小于 seed 噪声，或 SAC 在 formal 5M 出现更大的
safety/timeout 回退。此时不把“off-policy 一定更省样本”写成结论。

最便宜证伪：选一个代表 arm（优先 `T1-PPO` command residual scratch），各跑 1 个 seed、
1M screen；只比较方向和机制，不据此写正式 PASS。

### H86-E：路径跟踪是 guidance 轴，不是 BC 先验轴

机制：`path_fixed_speed` 改变的是 base command 的生成方式，不是残差网络或 BC 数据。
它必须与 `ALOS` 分开比较；zero-residual 路径跟踪也不能冒充“BC-only”。

成立应看到：`U0`（path zero-residual）与 `T0`（ALOS zero-residual）的差异可复现，
且 `U1-PPO/U1-SAC` 相对 `U0` 的 residual 效应独立于 ALOS 的残差效应。

失败应看到：路径跟踪的结果主要由路径控制器或速度策略解释，而不是 residual；
此时不能把路径跟踪收益写成 RL 残差收益。

## 3. 实验组与隔离

### 3.1 G0：历史 BC+残差（只读，不改）

| arm | base | 学习内容 | 算法/预算 | 用途 |
|---|---|---|---|---|
| `B0` | D3 BC | 无 | 原始 Goal26 设置 | BC-only 历史参考 |
| `B1` | D3 frozen BC | actuator residual | 原始 Goal26 P2 设置 | BC+残差历史参考 |

规则：

- `checkpoints/goal26/d3/pc_bc_actor.pt`、Goal26 配置、示范数据、reward、seed 和
  checkpoint 选择规则全部只读；不重训、不覆盖、不改 `w_path_3d` 或 residual scale。
- `B0/B1` 的原始结果保留在旧契约下。若需要在新契约下重评，必须标为
  `B0-contract/B1-contract`，只作跨契约诊断，不能和历史结果混算。
- 不用 `B0/B1` 的好坏反向选择 Goal86 的新 seed 或 Gate。

### 3.2 G1：BC 先验 + 残差（新增）

每个 BC 先验接口都分别跑 PPO 和 SAC；算法不能共用同一 checkpoint。

| arm family | base | 残差空间 | 初始化 | PPO arm | SAC arm | 说明 |
|---|---|---|---|---|---|---|
| `P-A` | ALOS + PD/PI | actuator `[rudder,stern,rpm]` | D3 教师残差 BC | `P-A-PPO` | `P-A-SAC` | actuator BC 先验 |
| `P-C` | ALOS + PD/PI | command `[psi_d,theta_d,u_d]` | command 教师残差 BC | `P-C-PPO` | `P-C-SAC` | **仅当有效 command teacher 存在** |

`P-A` 的教师残差标签：

```text
base_action = ALOS + PD/PI(observation)
teacher_action = D3_BC(observation)
label = normalize(teacher_action - base_action)
```

`P-C` 的教师残差标签必须是：

```text
base_command = ALOS(observation)
teacher_command = 更强的 command teacher(observation)
label = normalize(teacher_command - base_command)
```

如果没有一个闭环表现更好、且只用可部署观测的 command teacher，则 `P-C` 记为
`BC_COMMAND_TEACHER_UNAVAILABLE`，不启动训练。**不得**把 ALOS 自己的命令当教师，
否则标签恒为零，BC 只是把残差初始化为零，不能宣称有 BC 先验。

### 3.3 G2：传统控制器 + RL 残差（新增、独立）

每个学习臂都分别跑 PPO 和 SAC；`T0/U0` 是算法无关的传统控制器基线。

| arm family | guidance/base | 残差空间 | 初始化 | PPO arm | SAC arm | 作用 |
|---|---|---|---|---|---|---|
| `T0` | ALOS + PD/PI | 无 | zero | — | — | ALOS 传统基线 |
| `T1` | ALOS + PD/PI | command | scratch zero | `T1-PPO` | `T1-SAC` | command residual |
| `T2` | ALOS + PD/PI | actuator | scratch zero | `T2-PPO` | `T2-SAC` | actuator residual |
| `U0` | `path_fixed_speed` + PD/PI | 无 | zero | — | — | 传统路径跟踪基线 |
| `U1` | `path_fixed_speed` + PD/PI | command | scratch zero | `U1-PPO` | `U1-SAC` | 路径跟踪 + command residual |
| `U2` | `path_fixed_speed` + PD/PI | command | BC prior | `U2-PPO` | `U2-SAC` | 仅当 path command teacher 有效 |

说明：

- `T1-PPO/T1-SAC`、`T2-PPO/T2-SAC` 是新组的 scratch 主对照；`P-C-PPO/P-C-SAC`、
  `P-A-PPO/P-A-SAC` 是它们各自算法内的 BC 先验版本，其他训练因素必须逐项相同。
- `U0/U1` 是路径跟踪 guidance 轴；`U2` 只在存在与 path base 同契约的有效 command
  teacher 时运行，不能直接把 ALOS 的 teacher 结果当成 path teacher。
- `path_learned_speed` 暂不进入 Goal86。它会额外引入一个学习速度变量，等
  `U0/U1` 出现明确信号后再另立 arm。
- `P-A-*` / `P-C-*` 不能复用 `T1-*` / `T2-*` 的训练 checkpoint；初始化不同就是
  不同训练 run。PPO 与 SAC 之间也不能复用 checkpoint。

### 3.4 不进入 Goal86 的轴

- DADE / no-DADE：项目当前硬约束不允许继续 DADE 主线；另开 Goal 并先修订规则。
- REDQ / DroQ：本 Goal 不进入；只保留为未来样本效率 blocker 的候选，不现在预注册。
- DDPG/TD3：只保留为 Goal58/59 历史只读结果；不新增 DDPG/TD3 arm，也不把
  Goal59 checkpoint 当成 PPO/SAC 的新结果。
- plant mismatch：Goal85b 单独负责；Goal86 只使用 nominal plant。
- command + actuator 双残差：不允许，避免第二个学习残差网络。

## 4. 数据、契约与评价

| 项目 | 固定值 |
|---|---|
| 位置契约 | Goal85 `unbiased_noise_v1`；`p_nav = eta + zero-mean noise` |
| reward / termination / success | 真值 `eta`；`position_source=eta` |
| 传感器延迟 | 全部为 0；执行器一阶动态保留 |
| actor 真值观测 | `ideal_actor_observations=false` |
| 训练分布 | nominal；不包含 plant mismatch |
| dev seeds | `20–59`，40 个 |
| held-out | `100–149` 关闭，不读取 |
| eval 场景 | `default`、`strict125`；strict125 只在 eval 使用 |
| eval 精度 | CPU/float64 |
| termination | `endpoint_stop` |
| checkpoint | 每个训练 seed 的 `final.pt`；不按 dev 选 best |
| 主指标 | `eta_tight_at_done` / true-eta terminal capture |
| 风险指标 | success、timeout、plane-miss、safety、material saturation、path RMSE、residual L2 |
| 近目标次级指标 | `eta_min_distance_m`、near-miss shell、annulus entry、conditional capture、radial closure、radius sensitivity；不参与 PASS/FAIL |

历史 `B0/B1` 保留旧契约；新组 `T*/U*/P*` 使用上表契约。两类结果不得直接混合成
同一个 success 数字；跨契约比较只作诊断并明确标注。

### 4.1 目标附近过渡性评价

所有新 arm 复用 Goal85 §4.2 的定义和固定半径阶梯。每个 formal arm 至少输出：

- `eta_min_distance_m` 的分位数与 CDF；
- `near_miss_shell_{0.25,0.50,1.00}_fraction`；
- 各半径的 `annulus_entry`、`conditional_capture_at_r` 和 `capture_funnel`；
- `radial_closure_speed_at_min_mps`、`post_min_outward_speed_mps` 和近区停留时间；
- plane crossing 的横向/垂向偏移；
- `radius_sensitivity_curve`，并同时给出“含 censored 成功样本”和“仅失败样本”两套结果。

这些指标只解释失败发生在“接近、进入近区、终端对准、擦过外离”的哪一层，不能改变
`goal_radius`、success 或 Gate。`endpoint_stop` 的成功轨迹按 Goal85 规则标记
`censored_at_tight=true`，不能把成功和失败的 `eta_min_distance_m` 直接混算。
只有 trace-only 结果不足以区分近区分布时，才运行 Goal85 §4.2 的 eval-only
`capture_probe`；probe 不训练、不选 checkpoint、不写成正式 success。

## 5. BC 先验协议

### 5.1 数据和标签

- 示范数据必须来自与目标 base 相同的观测/动作包线，不能把旧 15° 或旧 27D 数据
  混入新 20° / 26D 契约。
- 教师动作只作为离线标签；actor 训练输入不能包含教师特权状态、真实 `eta` 或真实
  海流。
- 训练/验证按 episode 切分，不能让同一条轨迹的相邻步跨 split。
- 先归一化再计算 residual label；记录 label mean/std/max 和零标签比例。
- `P-C` 的 label 如果接近零（例如均值绝对值小于 heading scale 的 `1%`），必须判定
  教师没有提供有效命令信息，标记 `BC_COMMAND_LABEL_COLLAPSED`，不得进入 RL。

### 5.2 预训练与进入 RL 的条件

BC 残差 actor 只在训练 split 上拟合，验证集用于 early stopping；不得用 dev
`20–59` 的 success 选 checkpoint。进入 RL 前必须通过：

1. 残差输出维度、归一化、限幅和 rate limit 与目标 arm 一致；
2. 预训练残差不是全零，且 label/pred 统计已保存；
3. 残差置零时与 base 逐元素一致；
4. actor 输入维度与 Goal85 契约一致，`ideal_actor_observations=false`；
5. 无 NaN/Inf、无 teacher/actor 数据泄漏；
6. BC-only 闭环不出现 safety/timeout 的系统性恶化。

不通过则停止该 BC arm；不要用“再训练久一点”救无效标签。

### 5.3 BC+残差与历史组的关系

- `B1` 是历史 PPO + frozen D3 BC + actuator residual，原样保留。
- `P-A` 是新的 ALOS+PD/PI + actuator residual；不能把 `B1` 的 checkpoint
  当作 `P-A` 的训练结果，也不能把 `B1` 的旧 success 当作新契约结果。
- 如果未来要把 `B1` 以完全相同配方在新契约重跑，只能命名 `B1-contract`，并单独
  预注册训练 seed；不在本 Goal 中偷偷重训。

## 6. 残差与训练协议

### 6.1 动作接口

```text
ALOS command residual:
  ALOS -> [psi_d, theta_d, u_d] + actor_delta -> PD/PI -> actuator

ALOS actuator residual:
  ALOS -> PD/PI -> actuator + actor_delta -> actuator

path command residual:
  path_fixed_speed -> [psi_d, theta_d, u_d] + actor_delta -> PD/PI -> actuator
```

- command residual 使用 `command_residual.enabled=true`；actuator residual 使用
  `residual.scale_rudder/stern/rpm`，两者不能同时启用。
- 所有残差必须有绝对限幅、rate limit、零初始化或 BC 初始化校验。
- `T1-*/T2-*/U1-*` 的 actor 从零残差开始；`P-A-*/P-C-*/U2-*` 从 BC 残差开始。
- 训练 reward、terminal gate、success 定义在所有新 arm 之间固定；只允许动作接口
  和初始化不同。

### 6.2 算法

新学习 arm 只允许以下两种算法：

| 算法 | 类型 | 主要配置 | 用途 |
|---|---|---|---|
| `PPO` | on-policy | rollout + clipped policy objective | BC 先验、传统残差、指导线对照 |
| `SAC` | off-policy | replay + twin Q + entropy regularization | 同上，检验样本效率 |

- PPO 使用项目现有 batch PPO 训练入口；SAC 必须先完成 PC residual 环境适配。
  当前仓库只有 `third_party/stable-baselines3/stable_baselines3/sac`，没有已验证的
  PC/SAC trainer；Step 0 必须把这一项标成实现缺口，不能假设可直接训练。
- PPO 和 SAC 必须使用同一动作接口、同一 reward、同一 terminal 契约和同一 BC 先验；
  允许网络架构、rollout/replay、UTD 不同，但必须写清楚。
- 历史 `B0/B1` 保持 Goal26 的 PPO 设置不变；`B1` 不改成 SAC。
- Goal58/59 的 DDPG/TD3 checkpoint 只作历史参考，不参与 PPO/SAC 的 formal 比较。
- NoisyNet 只属于历史 DDPG 线；PPO/SAC 不使用 NoisyNet。若 residual 饱和，先查
  action parameterization 和 reward，不把算法切换当第一反应。

### 6.3 训练规模与算力

| 项目 | PPO | SAC |
|---|---:|---:|
| 训练 seeds | screen 1 / formal 3 | screen 1 / formal 3 |
| environment transitions / seed | screen 1M / formal 5M | screen 1M / formal 5M |
| 并行环境 | 先审计 `512/1024`，正式前冻结 | 先审计 `128/256`，正式前冻结 |
| 每轮 rollout / replay batch | rollout 128 steps；minibatch 按吞吐冻结 | replay batch 256；UTD 0.125 |
| checkpoint | `final` | `final` |
| 评价 seeds | 40 dev | 40 dev |
| 场景 | default/strict125 | default/strict125 |

这里的 `1M/5M` 指**每个训练 run 的 environment transitions 总数**，是所有并行环境
累计的交互步数，不是 episode 数，也不是 gradient update 次数。例如 `128` 个并行
环境各走 `7813` 个控制步，累计约 `1M` transitions。screen 的 `1M` 只用于机制和
工程筛选；formal 的 `5M` 才进入正式 Gate。3 个 formal seed 合计每个 arm 约
`15M` environment transitions。

算力使用规则：

- 先做 PPO `num_envs=512/1024` × rollout 128，SAC `num_envs=128/256` × batch
  `256/512` 的单卡吞吐基准；只允许在正式测量前冻结。
- 训练 seed 分布到 8 张 GPU 并行；BC 数据采集和 BC 预训练可与 screen 并行，但不能
  改变各 arm 的总 environment transitions 或评价 seeds。
- PPO 与 SAC 的 `num_envs`、batch、UTD 可以不同，因为它们服务不同优化器；但两者
  的 environment transition 预算和评价协议必须相同。报告时同时给出 wall-clock、
  gradient updates 和 UTD，不能只报“SAC 更快”。
- 增加环境数或 batch 只用于缩短 wall time，不得让某个 arm 获得更多有效样本。
- screen 失败不扩 formal、不换 seed、不增加预算；按预注册裁决停止或写 blocker。

## 7. Gate

### 7.1 机制 Gate（所有新 arm）

| check | 阈值 |
|---|---:|
| zero-residual parity | 与对应传统 base 逐元素一致，误差 `0`（浮点显示误差除外） |
| NaN / Inf / flew-out | `0` |
| actor 真值观测泄漏 | `0` |
| residual L2（gate-active） | `>=1%`，除非 BC 标签本身被证明无效 |
| residual saturation | deterministic raw saturation `<=0.50` |
| checkpoint 选择 | 只用 `final.pt`，不使用 dev best |
| 近目标次级指标改变 success / Gate | `false` |

### 7.2 BC 先验 Gate

对 `P-A-PPO` vs `T2-PPO`、`P-A-SAC` vs `T2-SAC`（actuator），以及可用的
`P-C-PPO` vs `T1-PPO`、`P-C-SAC` vs `T1-SAC`（command）分别比较：

| check | 通过条件 |
|---|---|
| screen 样本效率 | 1M 时 true-eta capture 提升 `>=5 pp`，或达到同指标使用 `<=50%` transitions |
| formal 不退化 | 5M 时任一主场景回退不超过 `2 pp` |
| formal 增益 | 至少一个场景 true-eta capture 提升 `>=3 pp` |
| 安全护栏 | safety/timeout 不恶化超过 `2 pp`；path RMSE 不恶化超过 `5%` |

若 screen 没有方向，BC 先验判 `NO_GAIN`；不扩 formal，不通过增加 BC epochs 或扫
学习率追结果。

### 7.3 传统控制器 + 残差 Gate

对 `T1-PPO/T1-SAC`、`T2-PPO/T2-SAC` vs `T0`，以及 `U1-PPO/U1-SAC` vs `U0`：

| check | 通过条件 |
|---|---|
| 主效果 | 至少一个场景 `eta_tight_at_done` 提升 `>=5 pp` |
| 另一场景 | 回退不超过 `2 pp` |
| safety / timeout | 恶化不超过 `2 pp` |
| plane-miss | 不出现超过 `5 pp` 的系统性恶化 |
| path RMSE | 不恶化超过 `5%` |
| residual 机制 | gate-active L2 `>=1%`，且不饱和 |

### 7.4 接口比较

`T1-*` vs `T2-*` 只回答 command vs actuator 接口；PPO 与 SAC 内部分别比较。若差异
小于 seed 噪声，判 `NO_INTERFACE_EFFECT`；不能把某一接口的小时好时坏写成通用优势。

### 7.5 算法比较

对同一 arm 的 `PPO` 与 `SAC` 比较：

| check | 通过条件 |
|---|---|
| screen 样本效率 | 1M 时一方 true-eta capture 高 `>=5 pp`，或同指标 wall-clock 低 `>=20%` |
| formal 主效果 | 5M 时一方至少一个场景高 `>=5 pp`，另一场景回退不超过 `2 pp` |
| 安全护栏 | 两个算法的 safety/timeout 都不恶化超过 `2 pp` |
| 资源报告 | 同时报告 env transitions、gradient updates、UTD、wall-clock 和 GPU 数 |

若 PPO/SAC 差异小于 seed 噪声，判 `NO_ALGORITHM_EFFECT`；不能因为 SAC 是 off-policy
就默认它更好，也不能因为 PPO 是 on-policy 就默认它更稳。

### 7.6 失败裁决

| 观察 | 判读 | 下一步 |
|---|---|---|
| BC 标签接近零 | `BC_TEACHER_INVALID` | 停止该 BC arm；不把零标签当先验 |
| BC 先验 screen 无方向 | `BC_PRIOR_NO_GAIN` | 不扩 formal；检查标签/接口后再决定新 Goal |
| residual 饱和且闭环不动 | `SATURATION_OR_TERMINAL_CAPTURE` | 查动作参数化、reward、terminal；不先换算法 |
| residual 很小且闭环不动 | `RESIDUAL_COLLAPSE` | 查 scale/初始化/BC 标签；不扩 seed |
| command 与 actuator 均失败 | `INTERFACE_NOT_CAUSE` | 检查 base、reward、terminal 和 plant mismatch |
| 新契约结果低于旧契约 | `CONTRACT_SHIFT` | 分开报告；不把跨契约下降归因给 BC 或残差 |
| 结果落在 seed 噪声内 | `SEED_NOISE_AMBIGUOUS` | 按 blocker 流程评估，不选择性报 seed |
| SAC 适配器未通过 smoke | `SAC_ADAPTER_INVALID` | 停止 SAC arm；修环境/动作/checkpoint 契约，不分析效果 |
| PPO/SAC 差异为噪声 | `NO_ALGORITHM_EFFECT` | 保留两种算法的描述性结果；不选 seed 或算法救 Gate |

## 8. 执行计划

### Step 0：冻结契约与 BC/动作空间审计

- [x] 确认 Goal85 的 `unbiased_noise_v1` 和 `termination.position_source=eta` 已实现并通过测试；
- [x] 新增 `scripts/audit_goal86_bc_interfaces.py`，只读检查现有 BC checkpoint 的
  `policy_kind`、`actor_output_dim`、`actor_command_dim`、输入 schema、动作包线和 SHA；
- [x] 新增 `scripts/audit_goal86_rl_backends.py`，确认 PPO 训练入口可用，并把 SAC
  PC residual 适配标为实现缺口；SAC smoke 未通过前不启动 SAC screen；
- [x] 输出 `artifacts/goal86/bc_interface_audit.json`，明确 D3 只能作为 actuator BC，
  不能作为 command residual 的初始化；
- [x] 审计 command teacher；没有有效 teacher 时把 `P-C/U2` 标为 `NOT_AVAILABLE`；
- [x] 记录 Goal85 checkpoint/config SHA、GPU 和磁盘状态；
- [x] 本步骤只读，不训练、不写 checkpoint。

运行：

```bash
cd /home/bubble/水下机器人/settled_局部规划器_反应式
python scripts/audit_goal86_bc_interfaces.py --output artifacts/goal86/bc_interface_audit.json
python scripts/audit_goal86_rl_backends.py --output artifacts/goal86/rl_backend_audit.json \
  --checkpoint <goal59-formal-seed>/final.pt
python ../scripts/check_research_ledger.py --goal Goal86 --goal-doc doc/goals/goal86_bc_prior_traditional_residual_20261110.md --strict
git diff --check
```

Git：只 add 审计脚本、测试和本文档；不 add `artifacts/`。用户明确要求后再 commit。

### Step 1：实现新配置和 BC 标签

- Step 1A（SAC `z_t`）已完成，证据见
  `doc/goals/goal86_step1_sac_zt_adapter_20261110.md`。
- [x] 新增 Goal86 专用 `T1-SAC` 与 `P-A-SAC` config，不覆盖历史 config；
- [x] 实现 SAC 的 PC residual VecEnv/history 适配、`z_t` actor、replay、checkpoint 和
  deterministic eval 接口；
- [x] 实现 `P-A-SAC` 的 D3 state/path encoder 与 actor 主干初始化；path8 前缀与
  endpoint4 显式拆分，不静默截断；
- [x] SAC 单测覆盖末帧编码、特征顺序、BC 加载、零残差初始化和 checkpoint 往返；
- [x] 新增 Goal86 专用 PPO config，不覆盖 Goal26/27/58/59 历史 config；
- [x] 为 `T*-PPO/T*-SAC/U*-PPO/U*-SAC/P-*-PPO/P-*-SAC` 生成成对配置；只允许算法和
  优化器参数不同，环境、reward、动作接口和评价协议必须相同；
- [x] 为 `P-A` 生成 D3 teacher 到 ALOS base 的 actuator residual 标签并完成只读审计；
  标签接入 BC 预训练仍未完成，见 §1.5 与 blocker；
- [ ] 修复 `P-A` 的残差 BC 契约：用真实 `D3-ALOS` 标签预训练残差头，并在同
  residual scale 下与 `T2` 配对；修复前不启动 P-A screen；
- [ ] 为 `P-C/U2` 实现 command residual 标签；无 teacher 则停止该分支；
- [x] 保持历史 `B0/B1` 的配置和 checkpoint 只读；
- [x] 补齐 PPO/SAC 的 label shape、动作限幅/rate limit、`z_t` 特征和 zero-residual
  parity 测试。

运行：

```bash
pytest tests/test_lowlevel_pc_command_residual.py tests/test_lowlevel_pc_residual_pretrain.py tests/test_lowlevel_config.py
git diff --check
```

Git：只 add 本次新增的 config、接口、测试；不修改历史 config。不要在未通过测试前
启动训练。

当前执行边界：`T1/T2 × PPO/SAC` 可以按预注册预算先跑 1M screen；`P-A` 在
`P-A_BC_PRIOR_CONTRACT_MISMATCH` 关闭前不进入 screen。P-A 修复时必须保持项目原生
`z_t` 特征布局，不得退回普通 flatten+MLP。

### Step 2：BC 预训练与 BC-only 检查

- [ ] 用固定 episode split 预训练 `P-A` 和可用的 `P-C` 的共享 BC 先验；
- [ ] 记录 label mean/std/max、预测残差均值、验证 loss 和 checkpoint SHA；
- [ ] 检查同一 BC 先验能否被 PPO/SAC 的 actor 主干加载；不能加载时必须显式说明
  哪些层重新初始化，不能悄悄把 BC 效果算给算法；
- [ ] 评估 BC-only 的 finite、safety、timeout、terminal capture，不按 dev success 选 best；
- [ ] 不通过则停止该 BC arm，不进入 RL。

运行示例（参数以最终 config 为准）：

```bash
python scripts/pretrain_pc_residual.py \
  --training-config <goal86_bc_prior_training_config> \
  --demos <goal86_demonstrations> \
  --output artifacts/goal86/bc_prior/<arm> \
  --device cuda
```

Git：只 add 预训练配置/脚本/测试；checkpoint 和 artifacts 不提交。

### Step 3：zero-residual 与 smoke

- [ ] 对 `T0/T1-*/T2-*/U0/U1-*/U2-*/P-A-*/P-C-*` 逐一做 zero-residual parity；
- [ ] 用 1 个 seed、短 rollout 检查 NaN/Inf、显存、吞吐、动作包线和日志字段；
- [ ] 确认 `T1-*/T2-*/U1-*` 从零初始化，`P-A-*/P-C-*/U2-*` 从 BC 初始化，二者不被混用；
- [ ] PPO 与 SAC 各自通过 smoke；SAC 适配 smoke 失败时标记 `SAC_ADAPTER_INVALID`；
- [ ] smoke 失败立即停，不扩 seed、不改 Gate。

运行：

```bash
pytest tests/test_lowlevel_pc_command_residual.py
python scripts/smoke_pc_command_residual_zero.py --config <goal86_config>
```

Git：只 add smoke 相关代码/测试和结果摘要；不 add 日志/checkpoint。

### Step 4：1M screen

- [ ] 每个 `PPO` 与 `SAC` arm 只跑 1 个预注册训练 seed；`P-A` 在残差 BC 契约修复前
  不运行，只先跑 `T1/T2` 四组；
- [ ] 记录 1M 时的 deterministic policy、zero-residual、terminal capture、residual L2；
- [ ] 记录 1M 的 `eta_min_distance_m`、near-miss shell 和 annulus entry；screen 只作
  机制方向判断，不用近目标指标改 Gate；
- [ ] 先判断 BC 先验是否有方向，再判断传统残差是否超过 `T0/U0`，最后比较同一 arm
  的 PPO/SAC；
- [ ] 不因为 screen 好看就跳过 formal 的 3 个 seed。

运行：PPO 使用 batch PPO 入口，SAC 使用 Step 1 验证过的 PC/SAC 入口；两者都固定
`total_env_transitions=1M`，算法专属并行度在 Step 0 冻结，输出到
`artifacts/goal86/screen/<arm>`。

Git：只 add screen 配置、分析脚本和摘要；不 add artifacts。

### Step 5：Formal 5M 与 40-seed 评价

- [ ] PPO 与 SAC 各 3 个独立训练 seed，每 seed 5M environment transitions；两者同预算；
- [ ] 每个 final checkpoint 在 dev `20–59`、default/strict125、CPU/float64 上评价；
- [ ] 每个 checkpoint 都跑同一 seed 的 zero-residual 配对；
- [ ] 结果包含 success、terminal capture、timeout、safety、plane-miss、RMSE、residual L2、saturation；
- [ ] 同时输出 §4.1 的近目标次级指标、`censored_at_tight` 标记和两套半径敏感性曲线；
- [ ] 结果同时记录 env transitions、gradient updates、UTD、wall-clock 和 GPU 数；
- [ ] 不打开 held-out `100–149`。

运行示例：

```bash
setsid nohup bash scripts/run_goal86_formal_20261110.sh \
  > logs/goal86_formal_20261110/nohup.log 2>&1 < /dev/null &
```

Git：只 add formal 配置、运行脚本、结果汇总脚本；不提交 artifacts/checkpoint/日志。

### Step 6：Result、ledger、STATE 与提交

- [ ] 写 `doc/goals/goal86_result_20261110.md`；
- [ ] 写 blocker/reading note（若触发）；
- [ ] 更新 `research/STATE.md`；
- [ ] 运行 ledger checker 和 git 检查；
- [ ] 只有用户明确要求时才 commit/push。

运行：

```bash
python ../scripts/check_research_ledger.py \
  --goal Goal86 \
  --goal-doc doc/goals/goal86_bc_prior_traditional_residual_20261110.md \
  --strict
git diff --check
git diff --cached --check
```

## 9. 报告与判读限制

- 历史 `B0/B1` 与新契约结果分开报告；不得把跨契约 success 直接相减当成 BC 效应。
- BC 先验与 residual 接口分开报告；不能用 `P-A-*`/`P-C-*` 相对各自 `T2-*`/`T1-*`
  的差异同时声称“command residual 更好”。
- 路径跟踪 `U*` 与 ALOS `T*` 分开报告；不能把 guidance 差异归因给 RL。
- 任何 `NOT_AVAILABLE`（例如没有 command teacher）都写清楚，不写成 FAIL。
- 单 seed screen 只作机制诊断，不写成正式结论；formal 必须报告 seed 数和配对方式。
- 近目标次级指标只用于解释接近/捕获漏斗；不得用更大的诊断半径重算 success，也不得
  把 `censored_at_tight` 的成功样本与失败样本的 `eta_min_distance_m` 混算。
- 结果不好时保留失败，不通过换 seed、扩预算、放宽 Gate 或打开 held-out 救结果。

## 10. Git

禁止 `git add -A`。只 add 本次相关的 goal/result 文档、配置、代码、测试和分析脚本；
不提交 `artifacts/`、checkpoint、日志、缓存和大文件。提交前运行：

```bash
python ../scripts/check_research_ledger.py \
  --goal Goal86 \
  --goal-doc doc/goals/goal86_bc_prior_traditional_residual_20261110.md \
  --strict
git diff --cached --check
```

不自动 commit / push；用户明确要求后才执行。

## 11. 自主迭代与停止条件

本 Goal 不是一次性训练任务。正式结果出来后，按 `research/AUTORESEARCH_LOOP.md` 自动进入下一轮：

1. 先按“数据 → 标签 → 评价 → baseline → 动作 → reward/terminal → 算法 → plant/estimator”归因；
2. 每个 blocker 写至少两个可证伪 hypothesis；
3. 用 `paper-search` / `deep-research` 查最接近的论文和可用代码，写 reading note；
4. 先跑最便宜的 falsification，再写 `research/LEDGER.jsonl` 的 `continue` / `pivot` / `stop`；
5. 若 `pivot`，新写 `doc/goals/goalXX_*.md` 和新的 Goal prompt，自动继续执行；
6. 不修改失败 Gate 的阈值、seed、eval split、held-out 或 checkpoint 选择规则。

持续执行的停止条件：

- 达到正式成功标准：一个 arm 在 formal 5M、3 个训练 seed 下满足主指标 `>=5 pp`、
  另一场景回退 `<=2 pp`、safety/timeout 与 plane-miss 护栏通过；最终用 5 个 seed 确认；
- 单 Goal 3 个 outer loop 仍无主指标或近目标机制移动：写 blocker review，不重复同一失败配置；
- 连续 3 个 Goal 没有可复现进展：写 `stop`/`pivot` 报告，等待用户决定新方向或预算；
- safety 回归、NaN/Inf、数据泄漏、预算耗尽：立即停。

以下事情不自动决定，必须向用户报告：改变任务成功定义或 `goal_radius`、打开 held-out、
改变 PPO/SAC 主算法族、重开 DADE、超过预注册预算、commit/push。其他诊断、论文检索、
新 Goal 写作和预注册预算内的实验继续自动执行。

## 12. Goal 提示词

```text
你现在执行 Goal86：BC 先验与传统控制器 RL 残差的分离对照。

先读：
- doc/goals/goal86_bc_prior_traditional_residual_20261110.md
- doc/goals/goal85_unbiased_nav_no_delay_20261110.md
- doc/goals/goal85b_plant_mismatch_mandatory_20261110.md
- doc/goals/goal26_result_20261108.md
- doc/goals/goal58_result_20261109.md
- doc/goals/goal59_result_20261109.md

硬约束：
1. 历史 B0/B1 的 BC+残差配置、checkpoint、reward、seed 和预算只读不改。
2. 新增 G1/G2 是独立实验组；不要用历史 checkpoint 冒充新契约结果。
3. 训练统一使用 Goal85 的 unbiased_noise_v1、零传感器延迟、真值 eta 判分。
4. 新学习 arm 只允许 PPO 与 SAC；DDPG/TD3 只作历史只读参考；REDQ/DroQ 和 DADE
   不进入本 Goal。
5. command residual 与 actuator residual 不叠加；一个 arm 只有一个残差网络。
6. 先做 Step 0 BC/动作空间与 RL backend 审计，再 BC 预训练，再 RL；SAC 适配器
   未通过 smoke 时标 SAC_ADAPTER_INVALID，不分析 SAC 效果。
7. 没有有效 command teacher 就把 P-C/U2 标为 NOT_AVAILABLE，不伪造标签。
8. dev 20-59，held-out 100-149 关闭；checkpoint 只用 final.pt。
9. PPO/SAC 使用相同 environment transition 预算和评价协议，算法专属并行度、
   batch、UTD 必须在正式测量前冻结并报告。
10. 这是一个长跑自主迭代任务：每轮执行 `research/AUTORESEARCH_LOOP.md`，不要等我逐步指令。
11. Gate FAIL 或 hard blocker 后，自动写 blocker、至少两个可证伪 hypothesis、用
    `paper-search` / `deep-research` 查论文、下载 PDF/TXT、找到可用 code、写 reading
    note、跑最便宜的 falsification，并在 `research/LEDGER.jsonl` 写
    `continue` / `pivot` / `stop`。
12. 如果 `pivot`，自动写新的 `doc/goals/goalXX_*.md` 和新的 Goal prompt，并在预注册
    预算内继续执行；不得修改失败 Gate 的阈值、seed、eval split、held-out 或
    checkpoint 规则。
13. 只有改变任务成功定义/`goal_radius`、打开 held-out、改变 PPO/SAC 主算法族、
    重开 DADE、超过预注册预算或 commit/push 时才停下来问我；其他诊断、调研、
    新 Goal 写作和预算内实验自动继续。
14. 不自动 commit / push。

执行顺序：
- 审计现有 BC checkpoint 的动作空间和 SHA；
- 审计 PPO 入口和 SAC PC residual 适配缺口；
- 实现 Goal86 专用 PPO/SAC config 与 BC label 接口；
- BC 预训练、BC-only 检查和 PPO/SAC zero-residual parity；
- 1M screen；
- PPO/SAC 各 3 个 seed × 5M formal；
- 40-seed dev/default+strict125 评价；
- 写 result、ledger、STATE，运行 checker；未获明确要求不 commit/push。
```
